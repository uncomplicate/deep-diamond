(ns uncomplicate.diamond.internal.cudnn.fully-connected
  (:require [uncomplicate.commons
             [core :refer [Releaseable release let-release with-release Info info]]
             [utils :refer [dragan-says-ex]]]
            [uncomplicate.neanderthal
             [core :refer [axpby! axpy! view dim copy!]]
             [real :refer [nrm2 asum]]
             [math :refer [sqr pow sqrt]]
             [vect-math :refer [linear-frac! linear-frac mul! log! log sqrt! sqr!]]
             [block :refer [cast-prim data-accessor buffer]]]
            [uncomplicate.neanderthal.internal
             [api :refer [flow]]
             [printing :refer [print-vector]]]
            [uncomplicate.diamond
             [tensor :as tz
              :refer [Transfer input output connector view-tz revert shape layout
                      TensorDescriptor shape]]
             [dnn :refer [Parameters bias weights transfer-parameters!]]]
            [uncomplicate.diamond.internal.protocols
             :refer [BlueprintProvider DiamondFactoryProvider DiffParameters
                     diff-bias diff-weights Backprop forward backward blueprint]]
            [uncomplicate.diamond.internal.cudnn
             [core :refer :all]
             [protocols :refer :all]
             [tensor :refer [cudnn-tensor-desc]]]
            [uncomplicate.diamond.internal.neanderthal.fully-connected
             :refer [->FullyConnectedBlueprint]])
  (:import clojure.lang.IFn
           uncomplicate.diamond.internal.neanderthal.fully_connected.FullyConnectedBlueprint))

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

;; ================================ Activation =============================================

(deftype CUDnnActivationInference [cudnn-hdl bluep activation-desc a-tz one zero linear]
  Releaseable
  (release [_]
    true)
  Info
  (info [this]
    {:activation (info bluep :activation)
     :a (info a-tz)})
  (info [this info-type]
    (case info-type
      :a (info a-tz)
      (info bluep info-type)))
  Transfer
  (input [_]
    a-tz)
  (output [_]
    a-tz)
  IFn
  (invoke [_]
    (when-not linear
      (activation-forward cudnn-hdl activation-desc
                          one a-tz (buffer a-tz) zero a-tz (buffer a-tz)))
    a-tz))

(deftype CUDnnActivationTraining [cudnn-hdl bluep activation-desc z-tz a-tz one zero linear]
  Releaseable
  (release [_]
    true)
  Info
  (info [this]
    {:activation (info bluep :activation)
     :z (info z-tz)
     :a (info a-tz)})
  (info [this info-type]
    (case info-type
      :a (info a-tz)
      :z (info z-tz)
      (info bluep info-type)))
  Transfer
  (input [_]
    z-tz)
  (output [_]
    a-tz)
  IFn
  (invoke [_]
    (if-not linear
      (activation-forward cudnn-hdl activation-desc
                          one z-tz (buffer z-tz) zero a-tz (buffer a-tz))
      (copy! z-tz a-tz))
    a-tz)
  Backprop
  (forward [this]
    (if-not linear
      (activation-forward cudnn-hdl activation-desc
                          one z-tz (buffer z-tz) zero a-tz (buffer a-tz))
      (copy! z-tz a-tz))
    this)
  (backward [this]
    (if-not linear
      (activation-backward cudnn-hdl activation-desc
                           one a-tz (buffer a-tz) a-tz (buffer a-tz) z-tz (buffer z-tz)
                           zero z-tz (buffer z-tz))
      (copy! a-tz z-tz))
    this))

(deftype CUDnnActivationBlueprint [fact activ ad linear]
  Releaseable
  (release [_]
    (release ad))
  Info
  (info [this]
    {:activation activ})
  (info [this info-type]
    (case info-type
      :activation activ
      nil))
  IFn
  (invoke [this src-tz]
    (->CUDnnActivationInference (handle fact) this ad src-tz
                                (cast-prim (data-accessor src-tz) 1.0)
                                (cast-prim (data-accessor src-tz) 0.0)
                                linear))
  (invoke [this src-tz dst-tz]
    (->CUDnnActivationTraining (handle fact) this ad src-tz dst-tz
                               (cast-prim (data-accessor src-tz) 1.0)
                               (cast-prim (data-accessor dst-tz) 0.0)
                               linear)))

(defn cudnn-activ-blueprint [fact activ coef]
  (let-release [ad (activation-descriptor activ true coef)]
    (->CUDnnActivationBlueprint fact activ ad (#{:linear :identity} activ))))

;; ============================= Fully Connected Layer ================================

(extend-type FullyConnectedBlueprint
  DescProvider
  (desc [this]
    (desc (.dst-desc this))))

;;TODO unify with neanderthal-fc-blueprint
(defn cudnn-fc-blueprint [fact src-desc dst-desc activ alpha beta]
  (let [dst-shape (shape dst-desc)
        weights-shape [(dst-shape 1) (apply * (rest (shape src-desc)))]]
    (let-release [dst-desc (cudnn-tensor-desc [(dst-shape 0) (apply * (rest dst-shape))]
                                              (or (tz/data-type dst-desc) (data-type src-desc))
                                              :nc)
                  bias-desc (cudnn-tensor-desc [(dst-shape 1)] (data-type dst-desc) :x)
                  weights-desc (cudnn-tensor-desc weights-shape (data-type dst-desc) :oi)
                  activ-bluep (cudnn-activ-blueprint fact activ alpha)]
      (->FullyConnectedBlueprint fact activ-bluep src-desc bias-desc weights-desc dst-desc))))

;; ============================= Cost Function ========================================

(deftype UniversalCost [prev-layer
                        connect-output connect-diff
                        a-y y cost]
  Releaseable
  (release [_]
    (release connect-output)
    (release connect-diff))
  Transfer
  (input [this]
    (input connect-output))
  (output [_]
    (output connect-output))
  Backprop
  (forward [this]
    (connect-output)
    this)
  (backward [this]
    (axpy! -1.0 y a-y)
    (connect-diff)
    (backward prev-layer)
    this)
  IFn
  (invoke [_]
    (connect-output)
    (axpy! -1.0 y a-y)
    (cost a-y)))

(defn cudnn-universal-cost [prev-layer train-tz cost]
  (let [train-desc (desc train-tz)
        output-desc (cudnn-tensor-desc (dims (output prev-layer))
                                       (data-type train-desc) (strides train-desc))]
    (let-release [connect-output (connector (output prev-layer) output-desc)
                  connect-diff (revert connect-output)]
      (->UniversalCost prev-layer
                       connect-output connect-diff
                       (view (output connect-output)) (view train-tz)
                       cost))))

(defn quadratic-cost [a-y]
  (/ (sqr (nrm2 a-y)) (* 2 (dim a-y))))

(defn mean-absolute-cost [a-y]
  (/ (asum a-y) (dim a-y)))

(defn sigmoid-crossentropy-cost [^long n a y]
  (with-release [ylna (mul! (log a) y)
                 y-1 (linear-frac 1.0 y -1.0)]
    (/ (asum (axpy! -1.0 ylna (mul! y-1 (log! (linear-frac! -1.0 a 1.0))))) n)))

(deftype CustomCost [prev-layer
                     connect-output connect-diff
                     a y cost]
  Releaseable
  (release [_]
    (release connect-output)
    (release connect-diff))
  Transfer
  (input [this]
    (input connect-output))
  (output [_]
    (output connect-output))
  Backprop
  (forward [this]
    (connect-output)
    this)
  (backward [this]
    (axpy! -1.0 y a)
    (connect-diff)
    (backward prev-layer)
    this)
  IFn
  (invoke [_]
    (connect-output)
    (cost a y)))

(defn cudnn-custom-cost [prev-layer train-tz cost]
  (let [train-desc (desc train-tz)
        output-desc (cudnn-tensor-desc (dims (output prev-layer))
                                       (data-type train-desc) (strides train-desc))]
    (let-release [connect-output (connector (output prev-layer) output-desc)
                  connect-diff (revert connect-output)]
      (->CustomCost  prev-layer
                    connect-output connect-diff
                    (view (output connect-output)) (view train-tz)
                    cost))))
