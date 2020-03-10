;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.diamond.internal.neanderthal.factory
  (:require [uncomplicate.commons
             [core :refer [Releaseable release let-release]]
             [utils :refer [dragan-says-ex]]]
            [uncomplicate.neanderthal.native :refer [factory-by-type]]
            [uncomplicate.neanderthal.internal.api :refer [FlowProvider flow]]
            [uncomplicate.diamond.tensor :refer [*diamond-factory* view-tz output]]
            [uncomplicate.diamond.internal
             [protocols
              :refer [TensorFactory DiamondFactoryProvider CostFactory DnnFactory
                      NeanderthalFactoryProvider]]
             [cost :refer [quadratic-cost! mean-absolute-cost! sigmoid-crossentropy-cost!]]]
            [uncomplicate.diamond.internal.dnnl
             [protocols :refer [DescProvider desc DnnlEngineProvider]]
             [core :refer [memory-desc engine stream memory dims]]
             [tensor :refer [dnnl-tensor dnnl-transformer dnnl-batcher dnnl-shuffler]]
             [fully-connected :refer [dnnl-sum-blueprint dnnl-activ-blueprint
                                      dnnl-inner-product-blueprint dnnl-fc-blueprint
                                      dnnl-universal-cost dnnl-custom-cost]]
             [factory :refer [->FloatTensorEngine]]]
            [uncomplicate.diamond.internal.neanderthal.fully-connected
             :refer [neanderthal-fc-blueprint]])
  (:import uncomplicate.diamond.internal.neanderthal.fully_connected.FullyConnectedBlueprint))

(def ^{:private true :const true} UNSUPPORTED_DATA_TYPE
  "The requested data type is not supported on the Neanderthal/DNNL platform.
Please contribute towards making it possible, or use on of the supported types.")

(extend-type FullyConnectedBlueprint
  DescProvider
  (desc [this]
    (desc (.activ-bluep this))))

(defrecord NeanderthalFactory [eng strm master tensor-engines]
  Releaseable
  (release [_]
    (when master
      (release strm)
      (release eng))
    true)
  DiamondFactoryProvider
  (diamond-factory [this]
    this)
  FlowProvider
  (flow [_]
    strm)
  DnnlEngineProvider
  (dnnl-engine [_]
    eng)
  NeanderthalFactoryProvider
  (neanderthal-factory [_ dtype]
    (factory-by-type dtype))
  TensorFactory
  (create-tensor-desc [this dims dtype format]
    (memory-desc dims dtype format))
  (create-tensor-desc [this tz-desc]
    (desc tz-desc))
  (create-tensor [this tensor-desc _]
    (dnnl-tensor this tensor-desc))
  (create-transformer [_ in-tz out-tz]
    (dnnl-transformer eng strm (view-tz in-tz) (view-tz out-tz)))
  (create-shuffler [_ src-tz dst-tz]
    (dnnl-shuffler eng strm (view-tz src-tz) (view-tz dst-tz)))
  (create-batcher [_ src-tz dst-tz mb-size]
    (dnnl-batcher eng strm (view-tz src-tz) (view-tz dst-tz) mb-size))
  (create-sum [_ scale dst]
    (dnnl-sum-blueprint eng strm scale dst))
  (create-sum [_ scale-src src scale-dst dst]
    (dnnl-sum-blueprint eng strm scale-src src scale-dst dst))
  (tensor-engine [this dtype]
    (or (get tensor-engines dtype)
        (dragan-says-ex UNSUPPORTED_DATA_TYPE {:data-type dtype})))
  DnnFactory
  (activ-blueprint [this src-desc activ alpha beta]
    (dnnl-activ-blueprint this eng src-desc src-desc activ alpha beta))
  (inner-product-blueprint [this src-desc dst-desc weights-type]
    (dragan-says-ex "Neanderthal engine does not implement inner product blueprint."))
  (fc-blueprint [this src-desc dst-desc activ alpha beta _]
    (neanderthal-fc-blueprint this src-desc dst-desc activ alpha beta))
  CostFactory
  (quadratic-cost [this prev-layer train-tz]
    (dnnl-universal-cost eng strm prev-layer train-tz quadratic-cost!))
  (mean-absolute-cost [this prev-layer train-tz]
    (dnnl-universal-cost eng strm prev-layer train-tz mean-absolute-cost!))
  (sigmoid-crossentropy-cost [this prev-layer train-tz]
    (dnnl-custom-cost eng strm prev-layer train-tz
                        (partial sigmoid-crossentropy-cost!
                                 ((dims (output prev-layer)) 0)))))

(defn neanderthal-factory
  ([eng strm]
   (->NeanderthalFactory eng strm false {:float (->FloatTensorEngine)}))
  ([]
   (let-release [eng (engine)]
     (->NeanderthalFactory eng (stream eng) true {:float (->FloatTensorEngine)}))))
