(ns uncomplicate.diamond.internal.protocols)

(declare default-diamond-factory)

;; ===================== General ========================================

(defprotocol DiamondFactoryProvider
  (diamond-factory [_])
  (native-diamond-factory [_]))

(defprotocol NeanderthalFactoryProvider
  (neanderthal-factory [this dtype]))

;; ===================== Tensor ========================================

(defprotocol TensorFactory
  (create-tensor-desc [this desc] [this shape type format])
  (create-tensor [this desc init])
  (create-transformer [this in out])
  (create-batcher [this src dst mb-size])
  (create-shuffler [this src dst])
  (create-sum [this scale dst] [this dst scale src scale-srcs])
  (tensor-engine [this tdesc]))

(defprotocol Offset
  (offset [tz n-ofst]))

;; =================== DNN ============================================

(defprotocol DnnFactory
  (activ-blueprint [this src-desc activ alpha beta])
  (inner-product-blueprint [this src-desc dst-desc weights-type])
  (fc-blueprint [this src-desc dst-desc activ alpha beta weights-type]))

(defprotocol CostFactory
  (quadratic-cost [this last-layer train-tz])
  (mean-absolute-cost [this last-layer train-tz])
  (sigmoid-crossentropy-cost [this last-layer train-tz]))

(defprotocol BlueprintProvider
  (blueprint [this]))

(defprotocol DiffParameters
  (diff-bias [this])
  (diff-weights [this]))

(defprotocol Backprop
  (forward [this] [this hyperparam])
  (backward [this] [this hyperparam]))

(defprotocol NeuralNetwork
  (layers [this]))
