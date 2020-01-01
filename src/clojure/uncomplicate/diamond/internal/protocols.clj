(ns uncomplicate.diamond.internal.protocols)

;; ===================== General ========================================

(defprotocol FactoryProvider
  (factory [_]))

(defprotocol ContextProvider
  (context [_]))

(defprotocol DataAccessorProvider
  (data-accessor ^DataAccessor [this dtype]))

;; ===================== Tensor ========================================

(defprotocol TensorFactory
  (create-tensor-desc [this desc] [this shape type format])
  (create-tensor [this desc])
  (create-transformer [this in out])
  (create-batcher [this src dst mb-size])
  (create-shuffler [this src dst])
  (create-sum [this scale dst] [this dst scale src scale-srcs])
  (tensor-engine [this tdesc]))

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
