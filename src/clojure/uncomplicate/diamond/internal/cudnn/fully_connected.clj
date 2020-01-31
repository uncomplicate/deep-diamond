(ns uncomplicate.diamond.internal.cudnn.fully-connected
  (:require [uncomplicate.commons
             [core :refer [Releaseable release]]
             [utils :refer [dragan-says-ex]]]
            [uncomplicate.neanderthal.core :refer [axpby!]]
            [uncomplicate.diamond.internal.cudnn.core :refer [add-tensor transform-tensor]])
  (:import clojure.lang.IFn))

(deftype CUDnnSum [cudnn-hdl scale-src src scale-dst dst]
  IFn
  (invoke [this]
    (axpby! scale-src src scale-dst dst)))

(deftype CUDnnSumBlueprint [cudnn-hdl scale-src scale-dst]
  IFn
  (invoke [this src-and-dst]
    (->CUDnnSum cudnn-hdl scale-src src-and-dst scale-dst src-and-dst))
  (invoke [this src dst]
    (->CUDnnSum cudnn-hdl scale-src src scale-dst dst)))

(defn cudnn-sum-blueprint
  ([cudnn-hdl scale]
   (->CUDnnSumBlueprint cudnn-hdl scale 0.0))
  ([cudnn-hdl scale-src scale-dst]
   (->CUDnnSumBlueprint cudnn-hdl scale-src scale-dst)))
