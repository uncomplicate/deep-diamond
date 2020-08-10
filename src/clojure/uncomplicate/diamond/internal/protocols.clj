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
  (create-sum [this scale dst] [this scale-src src scale-dst dst])
  (tensor-engine [this tdesc]))

(defprotocol Offset
  (offset [tz n-ofst]))

;; =================== DNN ============================================

(defprotocol DnnFactory
  (activ-blueprint [this src-desc activ alpha beta])
  (inner-product-blueprint [this src-desc dst-desc weights-type])
  (fc-blueprint [this src-desc dst-desc activ alpha beta weights-type])
  (convolution-blueprint [this src-desc kernel-desc dst-desc activ
                          strides padding-l padding-r alpha beta])
  (pooling-blueprint [this src-desc dst-desc algo
                      strides kernel padding-l padding-r])
  (gaussian-dropout-blueprint [this src-desc sd]))

(defprotocol CostFactory
  (quadratic-cost [this last-layer train-tz])
  (mean-absolute-cost [this last-layer train-tz])
  (crossentropy-cost [this last-layer train-tz]))

(defprotocol BlueprintProvider
  (blueprint [this]))

(defprotocol Parameters
  (weights [this])
  (bias [this]))

(defprotocol ParametersSeq
  (parameters [this]))

(defprotocol DiffParameters
  (diff-bias [this])
  (diff-weights [this]))

(defprotocol DiffTransfer
  (diff-input [this])
  (diff-z [this])
  (diff-output [this]))

(defprotocol Backprop
  (forward [this] [this hyperparam])
  (backward [this] [this hyperparam]))

(defprotocol NeuralNetwork
  (layers [this]))
