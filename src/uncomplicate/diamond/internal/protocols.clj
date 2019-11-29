(ns uncomplicate.diamond.internal.protocols)

;; ===================== General ========================================

(defprotocol FactoryProvider
  (factory [_]))

(defprotocol ContextProvider
  (context [_]))

;; ===================== Tensor ========================================

(defprotocol TensorFactory
  (create-tensor-desc [this desc] [this shape type format])
  (create-tensor [this desc])
  (create-transformer [this in out])
  (create-shuffler [this src dst]))
