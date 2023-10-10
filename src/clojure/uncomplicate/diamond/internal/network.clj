(ns uncomplicate.diamond.internal.network
  (:require [uncomplicate.commons
             [core :refer [Releaseable release let-release Info info with-release view]]
             [utils :refer [dragan-says-ex]]]
            [uncomplicate.fluokitten.core :refer [fmap join]]
            [uncomplicate.neanderthal
             [core :refer [transfer!]]
             [block :refer [data-accessor]]]
            [uncomplicate.neanderthal.internal.api :refer [destruct]]
            [uncomplicate.diamond.tensor :refer [Transfer input output tensor TensorDescriptor
                                                 shape data-type layout]]
            [uncomplicate.diamond.internal.protocols
             :refer [Backprop forward backward DiamondFactoryProvider diamond-factory
                     native-diamond-factory DiffTransfer diff-input diff-output diff-z Workspace
                     inf-ws-size train-ws-size create-workspace *workspace* DescriptorProvider
                     inf-desc train-desc diff-desc Initializable init neanderthal-factory]])
  (:import [clojure.lang IFn AFn Seqable Indexed ILookup]))

(extend-type java.lang.Object
  Workspace
  (inf-ws-size [this]
    0)
  (train-ws-size [this]
    0))

(defn invoke [f]
  (f))

(defn ^:private layer-info [layer]
  (let [nfo [(info layer :topology) (info layer :shape)]]
    (if-let [act (info layer :activation)]
      (conj nfo act)
      nfo)))

;; ======================== Sequential network ==============================

(deftype SequentialNetworkInference [x-tz forward-layers workspace]
  Releaseable
  (release [this]
    (release x-tz)
    (destruct (data-accessor (neanderthal-factory (diamond-factory this) :byte)) workspace)
    (doseq [l forward-layers] (release l))
    true)
  Object
  (hashCode [_]
    (reduce hash-combine (hash :sequential) forward-layers))
  (equals [this other]
    (if (= SequentialNetworkInference (type other))
      (= forward-layers (seq other))
      (= other this)))
  (toString [_]
    (format "#SequentialNetwork[inference, input:%s, layers:%d]"
            (shape x-tz) (count forward-layers)))
  Info
  (info [x]
    {:topology :sequential
     :batch (first (info (first forward-layers) :shape))
     :layers (fmap layer-info forward-layers)})
  (info [x info-type]
    (case info-type
      :topology :sequential
      :batch (first (info (first forward-layers) :shape))
      :layers (fmap layer-info forward-layers)
      nil))
  DiamondFactoryProvider
  (diamond-factory [_]
    (diamond-factory (peek forward-layers)))
  Seqable
  (seq [x]
    (seq forward-layers))
  Indexed
  (nth [_ i]
    (nth forward-layers i))
  (nth [_ i not-found]
    (nth forward-layers i not-found))
  (count [_]
    (count forward-layers))
  ILookup
  (valAt [_ k]
    (get forward-layers k))
  (valAt [_ k not-found]
    (get forward-layers k not-found))
  Initializable
  (init [this init-fn]
    (doseq [layer forward-layers]
      (init layer init-fn))
    this)
  Transfer
  (input [_] x-tz)
  (output [_] (output (peek forward-layers)))
  DiffTransfer
  (diff-input [this]
    (diff-input (peek forward-layers)))
  (diff-output [_]
    (dragan-says-ex "Inference network does not calculate gradients."))
  IFn
  (invoke [this]
    (peek (mapv invoke forward-layers)))
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(defmethod print-method SequentialNetworkInference
  [nn ^java.io.Writer w]
  (.write w "\n=======================================================================\n")
  (.write w (str nn))
  (doseq [layer nn]
    (.write w "\n-----------------------------------------------------------------------\n")
    (.write w (pr-str layer)))
  (.write w "\n=======================================================================\n"))

(deftype SequentialNetworkTraining [x-mb-tz forward-layers last-layer
                                    rest-backward-layers workspace]
  Releaseable
  (release [_]
    (release x-mb-tz)
    (destruct (data-accessor (neanderthal-factory (diamond-factory last-layer) :byte)) workspace)
    (doseq [l forward-layers] (release l))
    true)
  Object
  (hashCode [_]
    (reduce hash-combine (hash :sequential) forward-layers))
  (equals [_ other]
    (and (or (= SequentialNetworkTraining (type other))
             (= SequentialNetworkInference (type other)))
         (= forward-layers (seq other))))
  (toString [_]
    (format "#SequentialNetwork[train, input:%s, layers:%d]"
            (shape x-mb-tz) (count forward-layers)))
  Info
  (info [x]
    {:topology :sequential
     :batch (get (shape x-mb-tz) 0)
     :layers (fmap layer-info forward-layers)})
  (info [x info-type]
    (info [x info-type]
          (case info-type
            :topology :sequential
            :batch (get (shape x-mb-tz) 0)
            :layers (fmap layer-info forward-layers)
            nil)))
  DiamondFactoryProvider
  (diamond-factory [_]
    (diamond-factory last-layer))
  Seqable
  (seq [x]
    (seq forward-layers))
  Indexed
  (nth [_ i]
    (nth forward-layers i))
  (nth [_ i not-found]
    (nth forward-layers i not-found))
  (count [_]
    (count forward-layers))
  ILookup
  (valAt [_ k]
    (get forward-layers k))
  (valAt [_ k not-found]
    (get forward-layers k not-found))
  Initializable
  (init [this init-fn]
    (doseq [layer forward-layers]
      (init layer init-fn))
    this)
  Transfer
  (input [_] x-mb-tz)
  (output [_] (output last-layer))
  DiffTransfer
  (diff-input [_]
    (diff-input last-layer))
  (diff-z [_]
    (diff-z last-layer))
  (diff-output [_]
    (diff-output (first forward-layers)))
  IFn
  (invoke [this]
    (doseq [layer forward-layers]
      (layer))
    (output last-layer))
  (applyTo [this xs]
    (AFn/applyToHelper this xs))
  Backprop
  (forward [this hyperparam]
    (doseq [layer forward-layers]
      (forward layer hyperparam))
    this)
  (backward [this]
    (backward last-layer)
    this)
  (backward [this hyperparam]
    (backward last-layer hyperparam)
    (doseq [layer rest-backward-layers]
      (backward layer)
      (backward layer hyperparam))
    this))

(defmethod print-method SequentialNetworkTraining
  [nn ^java.io.Writer w]
  (.write w "\n=======================================================================\n")
  (.write w (str nn))
  (doseq [layer nn]
    (.write w "\n-----------------------------------------------------------------------\n")
    (.write w (pr-str layer)))
  (.write w "\n=======================================================================\n"))

(deftype SequentialNetworkBlueprint [fact src-desc layer-blueprints
                                     inf-ws-sz train-ws-sz]
  Releaseable
  (release [_]
    (doseq [l layer-blueprints] (release l))
    true)
  Object
  (hashCode [_]
    (reduce hash-combine (hash :sequential) layer-blueprints))
  (equals [_ n]
    (and (= SequentialNetworkBlueprint (type n))
         (= layer-blueprints (.layer-blueprints ^SequentialNetworkBlueprint n))))
  (toString [_]
    (str layer-blueprints))
  Info
  (info [this]
    {:input (shape src-desc)
     :shape (shape this)
     :topology :sequential})
  (info [this info-type]
    (case info-type
      :input (shape src-desc)
      :shape (shape this)
      :topology :sequential
      nil))
  DiamondFactoryProvider
  (diamond-factory [_]
    fact)
  Seqable
  (seq [x]
    (seq layer-blueprints))
  Indexed
  (nth [_ i]
    (nth layer-blueprints i))
  (nth [_ i not-found]
    (nth layer-blueprints i not-found))
  (count [_]
    (count layer-blueprints))
  ILookup
  (valAt [_ k]
    (get layer-blueprints k))
  (valAt [_ k not-found]
    (get layer-blueprints k not-found))
  DescriptorProvider
  (inf-desc [_]
    (inf-desc (peek layer-blueprints)))
  (train-desc [_]
    (train-desc (peek layer-blueprints)))
  (diff-desc [_]
    (diff-desc (peek layer-blueprints)))
  TensorDescriptor
  (shape [_]
    (shape (peek layer-blueprints)))
  (data-type [_]
    (data-type (peek layer-blueprints)))
  (layout [_]
    (layout (peek layer-blueprints)))
  Workspace
  (inf-ws-size [this]
    inf-ws-sz)
  (train-ws-size [this]
    train-ws-sz)
  IFn
  (invoke [this input-tz optimization]
    (.invoke this input-tz false optimization))
  (invoke [_ input-tz prop-diff? optimization]
    (let-release [input-tz (if input-tz (fmap view input-tz) (fmap (partial tensor fact) src-desc))
                  workspace (create-workspace fact train-ws-sz)]
      (binding [*workspace* workspace]
        (loop [bps (next layer-blueprints)
               backward-layers [((first layer-blueprints) input-tz prop-diff? optimization)]]
          (if bps
            (recur (next bps)
                   (cons ((first bps) (first backward-layers) true optimization)
                         backward-layers))
            (->SequentialNetworkTraining input-tz
                                         (vec (reverse backward-layers))
                                         (first backward-layers)
                                         (rest backward-layers)
                                         workspace))))))
  (invoke [this optimization-or-input-tz]
    (if (keyword? optimization-or-input-tz)
      (this nil optimization-or-input-tz)
      (let-release [input-tz (fmap view optimization-or-input-tz)
                    workspace (create-workspace fact inf-ws-sz)]
        (binding [*workspace* workspace]
          (loop [bps (next layer-blueprints)
                 forward-layers [((first layer-blueprints) input-tz)]]
            (if bps
              (recur (next bps) (conj forward-layers ((first bps) (peek forward-layers))))
              (->SequentialNetworkInference input-tz forward-layers workspace)))))))
  (invoke [this]
    (let-release [input-tz (tensor fact src-desc)]
      (this input-tz)))
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(defmethod print-method SequentialNetworkBlueprint
  [nn ^java.io.Writer w]
  (.write w (str nn)))

(declare eval-layer)

(defn sequential-network [fact src-desc layers]
  (let-release [layers (reduce (fn [lrs layer]
                                 (conj lrs (eval-layer fact (peek lrs) layer)))
                               [(eval-layer fact src-desc (first layers))]
                               (rest layers))]
    (->SequentialNetworkBlueprint fact src-desc layers
                                  (apply max (map inf-ws-size layers))
                                  (apply max (map train-ws-size layers)))))

(defmethod transfer! [SequentialNetworkInference Object]
  [source destination]
  (doall (map transfer! source destination))
  destination)

(defmethod transfer! [SequentialNetworkTraining Object]
  [source destination]
  (doall (map transfer! source destination))
  destination)

;; ============== Parallel network =========================================

(deftype ParallelNetworkInference [x-tzs parallel-layers]
  Releaseable
  (release [_]
    (doseq [x-tz x-tzs] (release x-tz))
    (doseq [l parallel-layers] (release l))
    true)
  Object
  (hashCode [_]
    (reduce hash-combine (hash :parallel) parallel-layers))
  (equals [this other]
    (if (= ParallelNetworkInference (type other))
      (= parallel-layers (seq other))
      (= other this)))
  (toString [_]
    (format "#ParallelNetwork[input:%s, layers:%d]"
            (fmap shape x-tzs) (count parallel-layers)))
  Info
  (info [x]
    {:topology :parallel
     :batch (first (info (first parallel-layers) :shape))
     :layers (fmap layer-info parallel-layers)})
  (info [x info-type]
    (case info-type
      :topology :parallel
      :batch (first (info (first parallel-layers) :shape))
      :layers (fmap layer-info parallel-layers)
      nil))
  DiamondFactoryProvider
  (diamond-factory [_]
    (diamond-factory (peek parallel-layers)))
  Seqable
  (seq [x]
    (seq parallel-layers))
  Indexed
  (nth [_ i]
    (nth parallel-layers i))
  (nth [_ i not-found]
    (nth parallel-layers i not-found))
  (count [_]
    (count parallel-layers))
  ILookup
  (valAt [_ k]
    (get parallel-layers k))
  (valAt [_ k not-found]
    (get parallel-layers k not-found))
  Initializable
  (init [this init-fn]
    (doseq [layer parallel-layers]
      (init layer init-fn))
    this)
  Transfer
  (input [_]
    x-tzs)
  (output [_]
    (fmap output parallel-layers))
  DiffTransfer
  (diff-input [this]
    (fmap diff-input parallel-layers))
  (diff-output [_]
    (dragan-says-ex "Inference network does not calculate gradients."))
  IFn
  (invoke [this]
    (fmap invoke parallel-layers))
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(defmethod print-method ParallelNetworkInference
  [nn ^java.io.Writer w]
  (.write w "\n= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =\n")
  (.write w (str nn))
  (doseq [layer nn]
    (.write w "\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n")
    (.write w (pr-str layer)))
  (.write w "\n= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =\n"))

(deftype ParallelNetworkTraining [x-mb-tzs parallel-layers]
  Releaseable
  (release [_]
    (doseq [x-mb-tz x-mb-tzs] (release x-mb-tz))
    (doseq [l parallel-layers] (release l))
    true)
  Object
  (hashCode [_]
    (reduce hash-combine (hash :parallel) parallel-layers))
  (equals [_ other]
    (and (or (= ParallelNetworkTraining (type other))
             (= ParallelNetworkInference (type other)))
         (= parallel-layers (seq other))))
  (toString [_]
    (format "#ParallelNetwork[input:%s, layers:%d]"
            (fmap shape x-mb-tzs) (count parallel-layers)))
  Info
  (info [x]
    {:topology :parallel
     :batch (fmap (comp first shape) x-mb-tzs)
     :layers (fmap layer-info parallel-layers)})
  (info [x info-type]
    (case info-type
      :topology :parallel
      :batch (fmap (comp first shape) x-mb-tzs)
      :layers (fmap layer-info parallel-layers)
      nil))
  DiamondFactoryProvider
  (diamond-factory [_]
    (diamond-factory (peek parallel-layers)))
  Seqable
  (seq [x]
    (seq parallel-layers))
  Indexed
  (nth [_ i]
    (nth parallel-layers i))
  (nth [_ i not-found]
    (nth parallel-layers i not-found))
  (count [_]
    (count parallel-layers))
  ILookup
  (valAt [_ k]
    (get parallel-layers k))
  (valAt [_ k not-found]
    (get parallel-layers k not-found))
  Initializable
  (init [this init-fn]
    (doseq [layer parallel-layers]
      (init layer init-fn))
    this)
  Transfer
  (input [_]
    x-mb-tzs)
  (output [_]
    (fmap output parallel-layers))
  DiffTransfer
  (diff-input [_]
    (fmap diff-input parallel-layers))
  (diff-z [_]
    (fmap diff-z parallel-layers))
  (diff-output [_]
    (fmap diff-output parallel-layers))
  IFn
  (invoke [this]
    (fmap invoke parallel-layers))
  (applyTo [this xs]
    (AFn/applyToHelper this xs))
  Backprop
  (forward [this hyperparam]
    (doseq [layer parallel-layers]
      (forward layer hyperparam))
    this)
  (backward [this]
    (doseq [layer parallel-layers]
      (backward layer))
    this)
  (backward [this hyperparam]
    (doseq [layer parallel-layers]
      (backward layer hyperparam))
    this))

(defmethod print-method ParallelNetworkTraining
  [nn ^java.io.Writer w]
  (.write w "\n= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =\n")
  (.write w (str nn))
  (doseq [layer nn]
    (.write w "\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n")
    (.write w (pr-str layer)))
  (.write w "\n= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =\n"))

(deftype ParallelNetworkBlueprint [fact src-descs layer-blueprints
                                   inf-ws-sz train-ws-sz]
  Releaseable
  (release [_]
    (doseq [l layer-blueprints] (release l))
    true)
  Object
  (hashCode [_]
    (reduce hash-combine (hash :parallel) layer-blueprints))
  (equals [_ n]
    (and (= ParallelNetworkBlueprint (type n))
         (= layer-blueprints (.layer-blueprints ^ParallelNetworkBlueprint n))))
  (toString [_]
    (str layer-blueprints))
  Info
  (info [this]
    {:input (fmap shape src-descs)
     :shape (shape this)
     :topology :parallel})
  (info [this info-type]
    (case info-type
      :input (fmap shape src-descs)
      :shape (shape this)
      :topology :parallel
      nil))
  DiamondFactoryProvider
  (diamond-factory [_]
    fact)
  DescriptorProvider
  (inf-desc [_]
    (fmap inf-desc layer-blueprints))
  (train-desc [_]
    (fmap train-desc layer-blueprints))
  (diff-desc [_]
    (fmap diff-desc layer-blueprints))
  TensorDescriptor
  (shape [_]
    (fmap shape layer-blueprints))
  (data-type [_]
    (fmap data-type layer-blueprints))
  (layout [_]
    (fmap layout layer-blueprints))
  Workspace
  (inf-ws-size [this]
    (apply max (fmap inf-ws-size layer-blueprints)))
  (train-ws-size [this]
    (apply max (fmap train-ws-size layer-blueprints)))
  IFn
  (invoke [this prev-layer optimization]
    (.invoke this prev-layer false optimization))
  (invoke [_ prev-layer prop-diff? optimization]
    (let [src-tzs (fmap output prev-layer)]
      (->ParallelNetworkTraining src-tzs
                                 (fmap (fn [bp x]
                                         (bp (view x) prop-diff? optimization))
                                       layer-blueprints src-tzs))))
  (invoke [this prev-layer]
    (let [src-tzs (fmap output prev-layer)]
      (->ParallelNetworkInference src-tzs
                                  (fmap (fn [bp x]
                                          (bp (view x)))
                                        layer-blueprints src-tzs))))
  (invoke [this]
    (let-release [input-tzs (fmap (partial tensor fact) src-descs)]
      (this input-tzs)))
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(defmethod print-method ParallelNetworkBlueprint
  [nn ^java.io.Writer w]
  (.write w (str nn)))

(defn parallel-network [fact src-descs parallel-layers]
  (let-release [layers (mapv (partial sequential-network fact) src-descs parallel-layers)]
    (->ParallelNetworkBlueprint fact src-descs layers
                                (apply max (map inf-ws-size layers))
                                (apply max (map train-ws-size layers)))))

(defn eval-layer [fact src-desc layer]
  (if (sequential? layer)
    (parallel-network fact (if (sequential? src-desc) src-desc (train-desc src-desc)) layer)
    (layer fact src-desc)))

(defmethod transfer! [ParallelNetworkInference Object]
  [source destination]
  (doall (map transfer! source destination))
  destination)

(defmethod transfer! [ParallelNetworkTraining Object]
  [source destination]
  (doall (map transfer! source destination))
  destination)
