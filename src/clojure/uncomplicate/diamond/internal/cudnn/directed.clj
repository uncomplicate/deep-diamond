(ns uncomplicate.diamond.internal.cudnn.directed
  (:require [uncomplicate.commons
             [core :refer [Releaseable release let-release with-release Info info]]
             [utils :refer [dragan-says-ex]]]
            [uncomplicate.clojurecuda.core :refer [mem-alloc]]
            [uncomplicate.neanderthal
             [core :refer [axpby! axpy! view dim copy! transfer!]]
             [block :refer [cast-prim data-accessor buffer]]
             [math :refer [sqrt pow]]
             [vect-math :refer [sqr! linear-frac! sqrt!]]]
            [uncomplicate.diamond
             [tensor :as tz
              :refer [Transfer input output connector view-tz revert shape layout
                      TensorDescriptor shape]]]
            [uncomplicate.diamond.internal.protocols
             :refer [BlueprintProvider DiamondFactoryProvider Backprop forward backward
                     blueprint create-tensor DiffTransfer diff-input diff-output diff-z
                     ParametersSeq]]
            [uncomplicate.diamond.internal.cudnn
             [core :refer :all]
             [protocols :refer :all]
             [tensor :refer [cudnn-tensor-desc cudnn-tensor]]]
            [uncomplicate.diamond.internal.neanderthal.fully-connected
             :refer [->FullyConnectedBlueprint]])
  (:import clojure.lang.IFn
           uncomplicate.diamond.internal.neanderthal.fully_connected.FullyConnectedBlueprint))

(deftype CuDnnSum [cudnn-hdl scale-src src scale-dst dst]
  IFn
  (invoke [this]
    (axpby! scale-src src scale-dst dst)))

(deftype CuDnnSumBlueprint [cudnn-hdl scale-src scale-dst]
  IFn
  (invoke [this src-and-dst]
    (->CuDnnSum cudnn-hdl scale-src src-and-dst scale-dst src-and-dst))
  (invoke [this src dst]
    (->CuDnnSum cudnn-hdl scale-src src scale-dst dst)))

(defn cudnn-sum-blueprint
  ([cudnn-hdl scale]
   (->CuDnnSumBlueprint cudnn-hdl scale 0.0))
  ([cudnn-hdl scale-src scale-dst]
   (->CuDnnSumBlueprint cudnn-hdl scale-src scale-dst)))

;; ================================ Activation =============================================

(deftype CuDnnActivationInference [cudnn-hdl bluep activation-desc
                                   a-tz one zero linear]
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
                          one a-tz (buffer a-tz)
                          zero a-tz (buffer a-tz)))
    a-tz))

(deftype CuDnnLinearActivationTraining [cudnn-hdl bluep activation-desc z-tz a-tz one zero]
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
  DiffTransfer
  (diff-input [_]
    a-tz)
  (diff-output [_]
    z-tz)
  IFn
  (invoke [_]
    (copy! z-tz a-tz)
    a-tz)
  Backprop
  (forward [this]
    (copy! z-tz a-tz)
    this)
  (backward [this]
    (copy! a-tz z-tz)
    this))

(deftype CuDnnActivationTraining [cudnn-hdl bluep activation-desc z-tz a-tz da-tz one zero]
  Releaseable
  (release [_]
    (release da-tz))
  Info
  (info [this]
    {:activation (info bluep :activation)
     :z (info z-tz)
     :a (info a-tz)
     :da (info da-tz)})
  (info [this info-type]
    (case info-type
      :a (info a-tz)
      :z (info z-tz)
      :da (info da-tz)
      (info bluep info-type)))
  Transfer
  (input [_]
    z-tz)
  (output [_]
    a-tz)
  DiffTransfer
  (diff-input [_]
    da-tz)
  (diff-output [_]
    z-tz)
  IFn
  (invoke [_]
    (activation-forward cudnn-hdl activation-desc
                        one z-tz (buffer z-tz) zero a-tz (buffer a-tz))
    a-tz)
  Backprop
  (forward [this]
    (activation-forward cudnn-hdl activation-desc
                        one z-tz (buffer z-tz) zero a-tz (buffer a-tz))
    this)
  (backward [this]
    (activation-backward cudnn-hdl activation-desc
                         one a-tz (buffer a-tz) da-tz (buffer da-tz) z-tz (buffer z-tz)
                         zero z-tz (buffer z-tz))
    this))

(deftype CuDnnActivationBlueprint [fact activ ad]
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
    (->CuDnnActivationInference (handle fact) this ad src-tz
                                (cast-prim (data-accessor src-tz) 1.0)
                                (cast-prim (data-accessor src-tz) 0.0)
                                (or (= :linear activ) (= :identity activ))))
  (invoke [this src-tz dst-tz]
    (cond
      (or (= :linear activ) (= :identity activ))
      (->CuDnnLinearActivationTraining (handle fact) this ad src-tz dst-tz
                                       (cast-prim (data-accessor src-tz) 1.0)
                                       (cast-prim (data-accessor dst-tz) 0.0))
      (or (= :sigmoid activ) (:logistic activ))
      (let-release [diff-tz (create-tensor fact dst-tz false)]
        (->CuDnnActivationTraining (handle fact) this ad src-tz dst-tz diff-tz
                                   (cast-prim (data-accessor src-tz) 1.0)
                                   (cast-prim (data-accessor dst-tz) 0.0)))
      :default
      (->CuDnnActivationTraining (handle fact) this ad src-tz dst-tz (view-tz dst-tz)
                                 (cast-prim (data-accessor src-tz) 1.0)
                                 (cast-prim (data-accessor dst-tz) 0.0)))))

;; ================================ Softmax =============================================

(deftype CuDnnSoftmaxInference [cudnn-hdl bluep z-tz one zero]
  Releaseable
  (release [_]
    true)
  Info
  (info [this]
    {:activation :softmax
     :z (info z-tz)})
  (info [this info-type]
    (case info-type
      :activation :softmax
      :z (info z-tz)
      (info bluep info-type)))
  Transfer
  (input [_]
    z-tz)
  (output [_]
    z-tz)
  IFn
  (invoke [_]
    (softmax-forward cudnn-hdl :accurate :instance
                     one z-tz (buffer z-tz) zero z-tz (buffer z-tz))
    z-tz))

(deftype CuDnnSoftmaxTraining [cudnn-hdl bluep z-tz da-tz one zero]
  Releaseable
  (release [_]
    (release da-tz))
  Info
  (info [this]
    {:activation :softmax
     :z (info z-tz)
     :da (info da-tz)})
  (info [this info-type]
    (case info-type
      :z (info z-tz)
      :da (info da-tz)
      (info bluep info-type)))
  Transfer
  (input [_]
    z-tz)
  (output [_]
    z-tz)
  DiffTransfer
  (diff-input [_]
    da-tz)
  (diff-output [_]
    z-tz)
  IFn
  (invoke [_]
    (softmax-forward cudnn-hdl :accurate :instance
                     one z-tz (buffer z-tz) zero z-tz (buffer z-tz))
    z-tz)
  Backprop
  (forward [this]
    (softmax-forward cudnn-hdl :accurate :instance
                     one z-tz (buffer z-tz) zero z-tz (buffer z-tz))
    this)
  (backward [this]
    (softmax-backward cudnn-hdl :accurate :instance
                      one z-tz (buffer z-tz) da-tz (buffer da-tz)
                      zero z-tz (buffer z-tz))
    this))

(deftype CuDnnSoftmaxBlueprint [fact]
  Releaseable
  (release [_]
    true)
  Info
  (info [this]
    {:activation :softmax})
  (info [this info-type]
    (case info-type
      :activation :softmax
      nil))
  IFn
  (invoke [this src-tz]
    (->CuDnnSoftmaxInference (handle fact) this src-tz
                             (cast-prim (data-accessor src-tz) 1.0)
                             (cast-prim (data-accessor src-tz) 0.0)))
  (invoke [this src-tz dst-tz]
    (->CuDnnSoftmaxTraining (handle fact) this src-tz (view-tz dst-tz)
                            (cast-prim (data-accessor src-tz) 1.0)
                            (cast-prim (data-accessor dst-tz) 0.0))))

(defn cudnn-activ-blueprint [fact activ coef]
  (if (= :softmax activ)
    (->CuDnnSoftmaxBlueprint fact)
    (let-release [ad (activation-descriptor activ true coef)]
      (->CuDnnActivationBlueprint fact activ ad))))

;; ============================= Fully Connected Layer ================================

(extend-type FullyConnectedBlueprint
  DescProvider
  (desc [this]
    (.dst-desc this)))

(defn cudnn-fc-blueprint [fact src-desc dst-desc activ alpha beta]
  (let [dst-shape (shape dst-desc)
        weights-shape [(dst-shape 1) (apply * (rest (shape src-desc)))]]
    (let-release [src-desc (cudnn-tensor-desc (shape src-desc) (data-type src-desc) (layout src-desc))
                  dst-desc (cudnn-tensor-desc [(dst-shape 0) (apply * (rest dst-shape))]
                                              (or (tz/data-type dst-desc) (data-type src-desc))
                                              :nc)
                  bias-desc (cudnn-tensor-desc [(dst-shape 1)] (data-type dst-desc) :x)
                  weights-desc (cudnn-tensor-desc weights-shape (data-type dst-desc) :oi)
                  activ-bluep (cudnn-activ-blueprint fact activ alpha)]
      (->FullyConnectedBlueprint fact activ-bluep src-desc bias-desc weights-desc dst-desc))))

;; ============================= Cost Function ========================================

(deftype CuDnnUniversalCost [prev-layer
                             connect-output connect-diff train-tz
                             a-y y cost]
  Releaseable
  (release [_]
    (release connect-output)
    (release connect-diff)
    (release train-tz))
  Transfer
  (input [this]
    (input connect-output))
  (output [_]
    (output connect-output))
  DiffTransfer
  (diff-input [_]
    train-tz)
  (diff-output [_]
    (output connect-diff))
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
  (let [train-desc (desc train-tz)]
    (let-release [connect-output (connector (output prev-layer) train-desc)
                  connect-diff (connector train-desc (diff-input prev-layer))]
      (->CuDnnUniversalCost prev-layer
                            connect-output connect-diff train-tz
                            (view (input connect-diff)) (view train-tz)
                            cost))))

(deftype CuDnnCustomCost [prev-layer
                          connect-output connect-diff train-tz
                          a y a-y cost]
  Releaseable
  (release [_]
    (release connect-output)
    (release connect-diff)
    (release train-tz))
  Transfer
  (input [this]
    (input connect-output))
  (output [_]
    (output connect-output))
  DiffTransfer
  (diff-input [_]
    (release train-tz))
  (diff-output [_]
    (output connect-diff))
  Backprop
  (forward [this]
    (connect-output)
    this)
  (backward [this]
    (copy! a a-y)
    (axpy! -1.0 y a-y)
    (connect-diff)
    this)
  IFn
  (invoke [_]
    (connect-output)
    (cost y a)))

(defn cudnn-custom-cost [prev-layer train-tz cost]
  (let [train-desc (desc train-tz)]
    (let-release [connect-output (connector (output prev-layer) train-desc)
                  connect-diff (connector train-desc (diff-z prev-layer))]
      (->CuDnnCustomCost prev-layer
                         connect-output connect-diff train-tz
                         (view (output connect-output)) (view train-tz) (view (input connect-diff))
                         cost))))

;; ================================ Pooling =============================================

(deftype CuDnnPoolingInferenceLayer [fact cudnn-hdl bluep pooling-desc
                                     src-tz dst-tz one zero]
  Releaseable
  (release [_]
    (release src-tz)
    (release dst-tz))
  Info
  (info [this]
    {:algo (info bluep :algo)
     :dst (info dst-tz)})
  (info [this info-type]
    (case info-type
      :algo (info bluep :algo)
      :dst (info dst-tz)
      (info bluep info-type)))
  Transfer
  (input [_]
    src-tz)
  (output [_]
    dst-tz)
  ParametersSeq
  (parameters [_]
    [])
  IFn
  (invoke [_]
    (pooling-forward cudnn-hdl pooling-desc
                     one src-tz (buffer src-tz) zero dst-tz (buffer dst-tz))
    dst-tz))

(deftype CuDnnPoolingTrainingLayer [fact cudnn-hdl bluep pooling-desc
                                    src-tz dst-tz diff-dst-tz
                                    one zero prop-diff?]
  Releaseable
  (release [_]
    (release src-tz)
    (release dst-tz)
    (release diff-dst-tz))
  Info
  (info [this]
    {:algo (info bluep :algo)
     :dst (info dst-tz)})
  (info [this info-type]
    (case info-type
      :algo (info bluep :algo)
      :dst (info dst-tz)
      (info bluep info-type)))
  Transfer
  (input [_]
    src-tz)
  (output [_]
    dst-tz)
  DiffTransfer
  (diff-input [_]
    diff-dst-tz)
  (diff-output [_]
    src-tz)
  ParametersSeq
  (parameters [_]
    [])
  IFn
  (invoke [this]
    (forward this nil)
    dst-tz)
  Backprop
  (forward [this]
    this)
  (forward [this _]
    (pooling-forward cudnn-hdl pooling-desc
                     one src-tz (buffer src-tz)
                     zero dst-tz (buffer dst-tz))
    this)
  (backward [this]
    this)
  (backward [this _]
    (when prop-diff?
      (pooling-backward cudnn-hdl pooling-desc
                        one dst-tz (buffer dst-tz) diff-dst-tz (buffer diff-dst-tz)
                        src-tz (buffer src-tz) zero src-tz (buffer src-tz)))
    this))

(deftype CuDnnPoolingBlueprint [fact algo pd dst-desc]
  Releaseable
  (release [_]
    (release pd))
  Info
  (info [this]
    {:algo algo})
  (info [this info-type]
    (case info-type
      :algo algo
      nil))
  DiamondFactoryProvider
  (diamond-factory [_]
    fact)
  BlueprintProvider
  (blueprint [this]
    this)
  DescProvider
  (desc [_]
    dst-desc)
  TensorDescriptor
  (shape [_]
    (shape dst-desc))
  (data-type [_]
    (tz/data-type dst-desc))
  (layout [_]
    (layout dst-desc))
  IFn
  (invoke [this prev-layer]
    (let-release [dst-tz (cudnn-tensor fact dst-desc)]
      (->CuDnnPoolingInferenceLayer fact (handle fact) this pd
                                    (view-tz (output prev-layer)) dst-tz
                                    (cast-prim (data-accessor dst-tz) 1.0)
                                    (cast-prim (data-accessor dst-tz) 0.0))))
  (invoke [this prev-layer prop-diff? _]
    (let-release [dst-tz (cudnn-tensor fact dst-desc)
                  diff-dst-tz (cudnn-tensor fact dst-desc)]
      (->CuDnnPoolingTrainingLayer fact (handle fact) this pd
                                   (view-tz (output prev-layer)) dst-tz diff-dst-tz
                                   (cast-prim (data-accessor dst-tz) 1.0)
                                   (cast-prim (data-accessor dst-tz) 0.0)
                                   prop-diff?))))

(defn cudnn-pooling-blueprint
  [fact dst-desc algo strides kernel padding]
  (let-release [pool-desc (pooling-descriptor algo kernel strides padding)]
    (->CuDnnPoolingBlueprint fact algo pool-desc (desc dst-desc))))

(defmethod transfer! [CuDnnPoolingInferenceLayer Object]
  [source destination]
  destination)

(defmethod transfer! [CuDnnPoolingTrainingLayer Object]
  [source destination]
  destination)
