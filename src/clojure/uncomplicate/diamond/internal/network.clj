(ns uncomplicate.diamond.internal.network
  (:require [uncomplicate.commons
             [core :refer [Releaseable release let-release Info info with-release view]]
             [utils :refer [dragan-says-ex]]]
            [uncomplicate.fluokitten.core :refer [fmap join]]
            [uncomplicate.neanderthal.core :refer [transfer!]]
            [uncomplicate.diamond.tensor :refer [Transfer input output tensor]]
            [uncomplicate.diamond.internal.protocols
             :refer [NeuralNetwork layers Backprop forward backward DiamondFactoryProvider
                     diamond-factory native-diamond-factory DiffTransfer diff-input
                     diff-output diff-z parameters Workspace inf-ws-size train-ws-size
                     create-workspace *workspace*]])
  (:import [clojure.lang IFn AFn]))

(extend-type java.lang.Object
  Workspace
  (inf-ws-size [this]
    0)
  (train-ws-size [this]
    0))

(defn invoke [f]
  (f))

(defn ^:private layer-info [layer]
  [(info layer :topology) (info layer :shape) (info layer :activation)])


;; ======================== Sequential network ==============================

(deftype SequentialNetworkInference [x-tz forward-layers workspace]
  Releaseable
  (release [_]
    (release x-tz)
    (release workspace)
    (doseq [l forward-layers] (release l))
    true)
  Object
  (hashCode [_]
    (reduce hash-combine (hash :sequential) forward-layers))
  (equals [_ n]
    (and (satisfies? NeuralNetwork n) (= forward-layers (layers n))))
  (toString [_]
    (format "[%s]" (apply str forward-layers)))
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
  NeuralNetwork
  (layers [_]
    forward-layers)
  Transfer
  (input [_] x-tz)
  (output [_] (output (peek forward-layers)))
  DiffTransfer
  (diff-input [this]
    (output this))
  (diff-output [_]
    (dragan-says-ex "Inference network does not calculate gradients."))
  IFn
  (invoke [this]
    (peek (mapv invoke forward-layers)))
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(defmethod print-method SequentialNetworkInference
  [nn ^java.io.Writer w]
  (.write w "\n[\n")
  (doseq [layer (layers nn)]
    (.write w (pr-str layer))
    (.write w "\n"))
  (.write w "]"))

(deftype SequentialNetworkTraining [x-mb-tz forward-layers last-layer
                                    rest-backward-layers workspace]
  Releaseable
  (release [_]
    (release x-mb-tz)
    (release workspace)
    (doseq [l forward-layers] (release l))
    true)
  Object
  (hashCode [_]
    (reduce hash-combine (hash :sequential) forward-layers))
  (equals [_ n]
    (and (satisfies? NeuralNetwork n) (= forward-layers (layers n))))
  (toString [_]
    (format "[%s]" (apply str forward-layers)))
  Info
  (info [x]
    {:topology :sequential
     :batch (first (info (first forward-layers) :shape))
     :layers (fmap layer-info forward-layers)})
  (info [x info-type]
    (info [x info-type]
          (case info-type
            :topology :sequential
            :batch (first (info (first forward-layers) :shape))
            :layers (fmap layer-info forward-layers)
            nil)))
  DiamondFactoryProvider
  (diamond-factory [_]
    (diamond-factory last-layer))
  NeuralNetwork
  (layers [_]
    forward-layers)
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
  (.write w "\n[\n")
  (doseq [layer (layers nn)]
    (.write w (pr-str layer))
    (.write w "\n"))
  (.write w "]"))

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
    (and (satisfies? NeuralNetwork n) (= layer-blueprints (layers n))))
  (toString [_]
    (str layer-blueprints))
  DiamondFactoryProvider
  (diamond-factory [_]
    fact)
  NeuralNetwork
  (layers [_]
    (format "[%s]" (apply str layer-blueprints)))
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
                                         (reverse backward-layers)
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
  (doall (map transfer! (layers source) (layers destination)))
  destination)

(defmethod transfer! [SequentialNetworkTraining Object]
  [source destination]
  (doall (map transfer! (layers source) (layers destination)))
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
  (equals [_ n]
    (and (satisfies? NeuralNetwork n) (= parallel-layers (layers n))))
  (toString [_]
    (format "[%s]" (apply str parallel-layers)))
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
  NeuralNetwork
  (layers [_]
    (join (fmap layers parallel-layers)))
  Transfer
  (input [_] x-tzs)
  (output [_]
    (fmap output parallel-layers))
  DiffTransfer
  (diff-input [this]
    (output this))
  (diff-output [_]
    (dragan-says-ex "Inference network does not calculate gradients."))
  IFn
  (invoke [this]
    (fmap invoke parallel-layers))
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(deftype ParallelNetworkTraining [x-mb-tzs parallel-layers]
  Releaseable
  (release [_]
    (doseq [x-mb-tz x-mb-tzs] (release x-mb-tz))
    (doseq [l parallel-layers] (release l))
    true)
  Object
  (hashCode [_]
    (reduce hash-combine (hash :parallel) parallel-layers))
  (equals [_ n]
    (and (satisfies? NeuralNetwork n) (= parallel-layers (layers n))));;TODO sort out layers/parallel-layers
  (toString [_]
    (format "[%s]" (apply str parallel-layers)))
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
  NeuralNetwork
  (layers [_]
    (join (fmap layers parallel-layers)))
  Transfer
  (input [_] x-mb-tzs)
  (output [_] (fmap output parallel-layers))
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

;; TODO print-method

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
    (and (satisfies? NeuralNetwork n) (= layer-blueprints (layers n))))
  (toString [_]
    (str layer-blueprints))
  DiamondFactoryProvider
  (diamond-factory [_]
    fact)
  NeuralNetwork
  (layers [_]
    (format "[%s]" (apply str layer-blueprints)))
  Workspace
  (inf-ws-size [this]
    (apply max (fmap inf-ws-size layer-blueprints)))
  (train-ws-size [this]
    (apply max (fmap train-ws-size layer-blueprints)))
  IFn
  (invoke [this input-tz optimization]
    (.invoke this input-tz false optimization))
  (invoke [_ input-tzs prop-diff? optimization]
    (->ParallelNetworkTraining input-tzs
                               (fmap (fn [bp x]
                                       (bp (view x) prop-diff? optimization))
                                     layer-blueprints input-tzs)))
  (invoke [this input-tzs]
    (->ParallelNetworkInference input-tzs
                                (fmap (fn [bp x]
                                        (bp (view x)))
                                      layer-blueprints input-tzs)))
  (invoke [this]
    (let-release [input-tzs (fmap (partial tensor fact) src-descs)]
      (this input-tzs)))
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(defn parallel-network [fact src-descs parallel-layers]
  (let-release [layers (mapv (partial sequential-network fact) src-descs parallel-layers)]
    (->ParallelNetworkBlueprint fact src-descs layers
                                (apply max (map inf-ws-size layers))
                                (apply max (map train-ws-size layers)))))

(defn eval-layer [fact src-desc layer]
  (if (sequential? src-desc)
    (parallel-network fact src-desc layer)
    (layer fact src-desc)))
