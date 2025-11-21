;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.diamond.internal.neanderthal.factory
  (:require [uncomplicate.commons
             [core :refer [Releaseable release let-release view bytesize]]
             [utils :refer [dragan-says-ex direct-buffer mapped-buffer]]]
            [uncomplicate.clojure-cpp :refer [byte-pointer type-pointer]]
            [uncomplicate.neanderthal.core :refer [entry!]]
            [uncomplicate.neanderthal
             [native :refer [factory-by-type native-float native-int native-byte]]
             [block :refer [create-data-source buffer initialize!]]]
            [uncomplicate.neanderthal.internal.api :refer [FlowProvider vector-engine]]
            [uncomplicate.diamond.tensor :refer [*diamond-factory* desc output shape data-type layout]]
            [uncomplicate.diamond.internal
             [protocols :refer [TensorFactory create-tensor MappedTensorFactory DiamondFactoryProvider
                                CostFactory DnnFactory NeanderthalFactoryProvider
                                inf-desc train-desc diff-desc]
              :as api]
             [utils :refer [default-strides]]
             [cost :refer [quadratic-cost! mean-absolute-cost! crossentropy-cost!]]]
            [uncomplicate.diamond.internal.neanderthal.tensor :refer [neanderthal-tensor]]
            [uncomplicate.diamond.internal.neanderthal.directed
             :refer [neanderthal-fc-blueprint neanderthal-gaussian-dropout-blueprint
                     ->ActivationLayerBlueprint]]))

(def ^{:private true :const true} UNSUPPORTED_DATA_TYPE
  "The requested data type is not supported on the Neanderthal platform.
Please contribute towards making it possible, or use on of the supported types.")

(def ^{:private true :const true} UNSUPPORTED_CONSTRUCT
  "The requested construct is not supported on the Neanderthal tensor factory.")

(defrecord VectorFactory [master native-diamond-fact neand-facts]
  Releaseable
  (release [_]
    (when master
      (release native-diamond-fact)
      (doseq [neand-fact (vals neand-facts)]
        (release neand-fact)))
    true)
  DiamondFactoryProvider
  (diamond-factory [this]
    this)
  (native-diamond-factory [this]
    (or native-diamond-fact this))
  NeanderthalFactoryProvider
  (neanderthal-factory [this]
    this)
  (neanderthal-factory [_ dtype]
    (or (neand-facts dtype)
        (dragan-says-ex UNSUPPORTED_DATA_TYPE {:data-type dtype})))
  TensorFactory
  (create-tensor-desc [this dims dtype format]
    (desc dims (or dtype :float) (or format (default-strides dims))))
  (create-tensor-desc [this tz-desc]
    tz-desc)
  (create-tensor [this tensor-desc init]
    (let [shape (shape tensor-desc)]
      (neanderthal-tensor (api/neanderthal-factory this (or (data-type tensor-desc) :float))
                          shape
                          (or (layout tensor-desc) (default-strides shape))
                          init)))
  (create-tensor [this tensor-desc batch-index init]
    (if (= 0 batch-index)
      (create-tensor this tensor-desc init)
      (dragan-says-ex "Neanderthal supports only 0 as batch index." {:requested batch-index})))
  (create-transformer [_ in-tz out-tz]
    (dragan-says-ex UNSUPPORTED_CONSTRUCT {:construct :transformer}))
  (create-shuffler [_ src-tz dst-tz]
    (dragan-says-ex UNSUPPORTED_CONSTRUCT {:construct :shuffler}))
  (create-batcher [_ src-tz dst-tz mb-size]
    (dragan-says-ex UNSUPPORTED_CONSTRUCT {:construct :batcher}))
  (tensor-engine [this dtype]
    (vector-engine (neand-facts dtype)))
  ;; MappedTensorFactory
  ;; (map-channel [this channel td flag offset-bytes n-index]
  ;;   (let [size (bytesize (desc td))]
  ;;     (let-release [buf ((type-pointer (data-type td)) (mapped-buffer channel offset-bytes size flag))]
  ;;       (dnnl-tensor* this td buf n-index true))))
  ;; DnnFactory
  ;; (activ-op-blueprint [this desc-provider activ alpha beta]
  ;;   (dnnl-activ-blueprint this eng (view (inf-desc desc-provider))
  ;;                         (view (train-desc desc-provider))
  ;;                         (view (diff-desc desc-provider))
  ;;                         activ alpha beta))
  ;; (activ-blueprint [this desc-provider activ alpha beta]
  ;;   (let-release [activ-bluep (dnnl-activ-blueprint this eng (view (inf-desc desc-provider))
  ;;                                                   (view (train-desc desc-provider))
  ;;                                                   (view (diff-desc desc-provider))
  ;;                                                   activ alpha beta)]
  ;;     (->ActivationLayerBlueprint this activ-bluep)))
  ;; (inner-product-blueprint [this src-desc dst-desc weights-type]
  ;;   (dnnl-inner-product-blueprint this eng src-desc dst-desc weights-type))
  ;; (fc-blueprint [this src-desc dst-desc activ alpha beta weights-type]
  ;;   (neanderthal-fc-blueprint this src-desc dst-desc activ alpha beta weights-type))
  )

(defn vector-factory
  ([master native-diamond-fact neand-facts]
   (->VectorFactory master native-diamond-fact neand-facts))
  ([master neand-facts]
   (->VectorFactory master nil neand-facts))
  ([]
   (->VectorFactory false nil factory-by-type)))
