(ns uncomplicate.diamond.internal.network
  (:require [uncomplicate.commons
             [core :refer [Releaseable release let-release Info info with-release view]]
             [utils :refer [dragan-says-ex]]]
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
     :layers (mapv layer-info forward-layers)})
  (info [x info-type]
    (case info-type
      :topology :sequential
      :batch (first (info (first forward-layers) :shape))
      :layers (mapv layer-info forward-layers)
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
     :layers (mapv layer-info forward-layers)})
  (info [x info-type]
    (info [x info-type]
          (case info-type
            :topology :sequential
            :batch (first (info (first forward-layers) :shape))
            :layers (mapv layer-info forward-layers)
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
    (backward last-layer))
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
                                     inf-ws-size train-ws-size]
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
  IFn
  (invoke [_ input-tz optimization]
    (let-release [input-tz (view input-tz)
                  workspace (create-workspace fact train-ws-size)]
      (binding [*workspace* workspace]
        (loop [bps (next layer-blueprints)
               backward-layers [((first layer-blueprints) input-tz false optimization)]]
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
      (let-release [input-tz (tensor fact src-desc)]
        (this input-tz optimization-or-input-tz))
      (let-release [input-tz (view optimization-or-input-tz)
                    workspace (create-workspace fact inf-ws-size)]
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

(defn sequential-network [fact src-desc layers]
  (let-release [layers (reduce (fn [lrs layer-fn]
                                 (conj lrs (layer-fn fact (peek lrs))))
                               [((first layers) fact src-desc)]
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
