(ns uncomplicate.diamond.internal.cudnn.factory
  (:require [uncomplicate.commons.core :refer [Releaseable release let-release]]
            [uncomplicate.clojurecuda.core
             :refer [init device context current-context stream default-stream in-context]]
            [uncomplicate.neanderthal.internal.api :refer [FlowProvider]]
            [uncomplicate.diamond.internal.protocols
             :refer [TensorFactory FactoryProvider ContextProvider CostFactory DnnFactory]]
            [uncomplicate.diamond.internal.cudnn.core :refer [cudnn-handle get-cudnn-stream]])
  (:import jcuda.jcudnn.JCudnn))

(deftype CUDnnFactory [ctx hstream handle master]
  Releaseable
  (release [_]
    (in-context ctx
                (release handle)
                (when master
                  (when-not (= default-stream hstream)
                    (release hstream))
                  (release ctx)))
    true)
  FactoryProvider
  (factory [this]
    this)
  FlowProvider
  (flow [_]
    hstream)
  ContextProvider
  (context [_]
    ctx)
  TensorFactory
  (create-tensor-desc [this dims dtype format]
    )
  (create-tensor-desc [this tz-desc]
    )
  (create-tensor [this tensor-desc]
    )
  (create-transformer [_ in-tz out-tz]
    )
  (create-shuffler [_ src-tz dst-tz]
    )
  (create-batcher [_ src-tz dst-tz mb-size]
    )
  (create-sum [_ scale dst]
    )
  (create-sum [_ dst scale src scale-srcs]
    )
  DnnFactory
  (activ-blueprint [this src-desc activ alpha beta]
    )
  (inner-product-blueprint [this src-desc dst-desc weights-type]
    )
  (fc-blueprint [this src-desc dst-desc activ alpha beta weights-type]
    )
  CostFactory
  (quadratic-cost [this prev-layer train-tz]
    )
  (mean-absolute-cost [this prev-layer train-tz]
    )
  (sigmoid-crossentropy-cost [this prev-layer train-tz]
    ))

(JCudnn/setExceptionsEnabled false)

(defn cudnn-factory
  ([ctx hstream]
   (in-context
    ctx
    (let-release [handle (cudnn-handle hstream)
                  hstream (get-cudnn-stream handle)]
      (->CUDnnFactory (current-context) hstream handle false))))
  ([]
   (init)
   (let-release [ctx (context (device))]
     (in-context
      ctx
      (let-release [hstream (stream)
                    handle (cudnn-handle hstream)]
        (->CUDnnFactory (current-context) hstream handle true))))))
