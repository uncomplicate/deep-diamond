(ns uncomplicate.diamond.internal.network
  (:require [uncomplicate.commons.core :refer [Releaseable release let-release Info info]]
            [uncomplicate.neanderthal.core :refer [transfer!]]
            [uncomplicate.diamond.tensor :refer [Transfer input output]]
            [uncomplicate.diamond.internal.protocols
             :refer [NeuralNetwork layers Backprop forward backward DiamondFactoryProvider
                     diamond-factory]])
  (:import clojure.lang.IFn))

(defn invoke [f]
  (f))

(defn ^:private layer-info [layer]
  [(info layer :topology) (:shape (info layer :bias)) (info layer :activation)])

(deftype SequentialNetworkInference [forward-layers]
  Releaseable
  (release [_]
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
  (input [_] (input (get forward-layers 0)))
  (output [_] (output (peek forward-layers)))
  IFn
  (invoke [this]
    (peek (mapv invoke forward-layers))))

(defmethod print-method SequentialNetworkInference
  [nn ^java.io.Writer w]
  (.write w "\n[\n")
  (doseq [layer (layers nn)]
    (.write w (pr-str layer))
    (.write w "\n"))
  (.write w "]"))

(deftype SequentialNetworkTraining [forward-layers last-layer rest-backward-layers]
  Releaseable
  (release [_]
    (doseq [l forward-layers] (release l)))
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
  (input [_] (input (first forward-layers)))
  (output [_] (output last-layer))
  IFn
  (invoke [this]
    (doseq [layer forward-layers]
      (layer))
    (output last-layer))
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

(deftype SequentialNetworkBlueprint [layer-blueprints]
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
  NeuralNetwork
  (layers [_]
    (format "[%s]" (apply str layer-blueprints)))
  IFn
  (invoke [_ input-tz optimization]
    (loop [bps (next layer-blueprints)
           backward-layers [((first layer-blueprints) input-tz false optimization)]]
      (if bps
        (recur (next bps)
               (cons ((first bps) (first backward-layers) true optimization)
                     backward-layers))
        (->SequentialNetworkTraining (reverse backward-layers)
                                     (first backward-layers)
                                     (rest backward-layers)))))
  (invoke [this input-tz]
    (loop [bps (next layer-blueprints)
           forward-layers [((first layer-blueprints) input-tz)]]
      (if bps
        (recur (next bps) (conj forward-layers ((first bps) (peek forward-layers))))
        (->SequentialNetworkInference forward-layers)))))

(defn sequential-network [fact src-desc layers]
  (let-release [layers (reduce (fn [lrs layer-fn]
                                 (conj lrs (layer-fn fact (peek lrs))))
                               [((first layers) fact src-desc)]
                               (rest layers))]
    (->SequentialNetworkBlueprint layers)))

(defmethod transfer! [SequentialNetworkInference Object]
  [source destination]
  (doall (map transfer! (layers source) (layers destination)))
  destination)

(defmethod transfer! [SequentialNetworkTraining Object]
  [source destination]
  (doall (map transfer! (layers source) (layers destination)))
  destination)
