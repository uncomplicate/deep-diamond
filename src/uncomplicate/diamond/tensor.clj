(ns uncomplicate.diamond.tensor
  (:require [uncomplicate.commons.core :refer [release Info]]
            [uncomplicate.diamond.internal.protocols :as api]))

(def ^:dynamic *diamond-factory*)

(defmacro with-diamond [factory-fn params & body]
  `(binding [*diamond-factory* (~factory-fn ~@params)]
     (try ~@body
          (finally (release *diamond-factory*)))))

;; ====================== Public protocols =====================================

(defprotocol TensorDescriptor
  (shape [tz])
  (layout [tz])
  (data-type [tz]))

(extend-type java.util.Collection
  TensorDescriptor
  (shape [this]
    (vec this))
  (data-type [_]
    nil)
  (layout [_]
    nil))

(extend-type clojure.lang.IPersistentVector
  TensorDescriptor
  (shape [this]
    this)
  (data-type [_]
    nil)
  (layout [_]
    nil))

(extend-type java.util.Map
  TensorDescriptor
  (shape [this]
    (get this :shape []))
  (data-type [this]
    (get this :data-type))
  (layout [this]
    (get this :layout)))

(defprotocol TensorContainer
  (view-tz [this] [this sub]))

(defprotocol Revert
  (revert [this]))

(defprotocol Transfer
  (input [this])
  (output [this]))

(defprotocol ConnectorCreator
  (connector [in out]))

;; =============================================================================

(defrecord TensorDescriptorImpl [shape data-type layout]
  Info
  (info [this]
    {:class (class this)
     :device :cpu
     :shape shape
     :data-type data-type
     :layout layout})
  TensorDescriptor
  (shape [_]
    shape)
  (data-type [_]
    data-type)
  (layout [_]
    layout))

(defn desc
  ([shape type layout]
   (->TensorDescriptorImpl shape type layout))
  ([shape layout]
   (desc shape nil layout))
  ([shape]
   (desc (uncomplicate.diamond.tensor/shape shape) nil nil)))

(defn tensor
  ([tz-factory shape type format]
   (let [fact (api/factory tz-factory)]
     (api/create-tensor fact (api/create-tensor-desc fact shape type format))))
  ([shape type format]
   (tensor *diamond-factory* shape type format))
  ([tz-factory desc]
   (let [fact (api/factory tz-factory)]
     (api/create-tensor fact (api/create-tensor-desc fact desc) )))
  ([desc]
   (tensor *diamond-factory* desc)))

(defn transformer [x y]
  (api/create-transformer (api/factory x) x y))

(defn batcher
  ([x y]
   (batcher x y ((shape y) 0)))
  ([x y ^long mb-size]
   (api/create-batcher (api/factory x) x y mb-size)))

(defn shuffler [x y]
  (api/create-shuffler (api/factory x) x y))
