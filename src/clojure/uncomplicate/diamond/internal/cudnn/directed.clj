(ns uncomplicate.diamond.internal.cudnn.directed
  (:require [uncomplicate.commons
             [core :refer [Releaseable release let-release with-release Info info]]
             [utils :refer [dragan-says-ex]]]
            [uncomplicate.clojurecuda.core :refer [mem-alloc]]
            [uncomplicate.neanderthal
             [core :refer [axpby! axpy! view dim copy! transfer! raw zero]]
             [block :refer [cast-prim data-accessor buffer]]
             [math :refer [sqrt pow]]
             [vect-math :refer [sqr! linear-frac! sqrt!]]]
            [uncomplicate.diamond
             [tensor :as tz
              :refer [Transfer input output connector view-tz revert shape layout
                      TensorDescriptor shape]]]
            [uncomplicate.diamond.internal
             [protocols
              :refer [BlueprintProvider DiamondFactoryProvider Backprop forward backward
                      blueprint create-tensor DiffTransfer diff-input diff-output diff-z
                      ParametersSeq Parameters DiffParameters]]
             [utils :refer [transfer-weights-bias!]]]
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

(defn cudnn-fc-blueprint [fact src-desc dst-desc activ alpha _]
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
                         (view (output connect-output)) (view train-tz)
                         (view (input connect-diff))
                         cost))))

;; ================================ Convolution ===============================================

(deftype CuDnnConvolutionInferenceLayer [fact cudnn-hdl bluep one zero activ
                                         conv-desc filter-desc conv-fwd-algo
                                         src-conn bias-tz weights-tz dst-tz workspace]
  Releaseable
  (release [_]
    (release activ)
    (release conv-desc)
    (release src-conn)
    (release bias-tz)
    (release weights-tz)
    (release dst-tz)
    (release workspace))
  ;; TODO implement equals, hashcode etc.
  Info
  (info [this]
    (assoc (info activ)
           :shape (info bluep :shape)
           :bias (info bias-tz)
           :weights (info weights-tz)
           :dst (info dst-tz)
           :topology :convolution :algorithm :inference))
  (info [this info-type]
    (case info-type
      :shape (info bluep :shape)
      :bias (info bias-tz)
      :weights (info weights-tz)
      :dst (info dst-tz)
      :topology :convolution :algorithm :inference
      (or (info activ info-type) (info bluep info-type))))
  DiamondFactoryProvider
  (diamond-factory [_]
    fact)
  BlueprintProvider
  (blueprint [_]
    bluep)
  Parameters
  (bias [_]
    bias-tz)
  (weights [_]
    weights-tz)
  ParametersSeq
  (parameters [_]
    [weights-tz bias-tz])
  Transfer
  (input [this]
    (input src-conn))
  (output [_]
    (output activ))
  IFn
  (invoke [_]
    (src-conn)
    (convolution-fwd cudnn-hdl conv-desc conv-fwd-algo
                     one (output src-conn) (buffer (output src-conn))
                     filter-desc (buffer weights-tz) zero dst-tz (buffer dst-tz) workspace)
    (add-tensor cudnn-hdl one bias-tz (buffer bias-tz) one dst-tz (buffer dst-tz))
    (activ)))

(deftype CuDnnConvolutionSGDLayer [fact cudnn-hdl bluep one zero scal-n ^long n
                                   activ prop-diff? conv-desc filter-desc
                                   conv-fwd-algo conv-bwd-data-algo conv-bwd-weights-algo
                                   v w diff-weights-vec
                                   src-conn bias-tz weights-tz dst-tz a-tz
                                   diff-src-conn diff-weights-tz workspace]
  Releaseable
  (release [_]
    (release activ)
    (release src-conn)
    (release bias-tz)
    (release weights-tz)
    (release dst-tz)
    (release a-tz);;TODO move the release part to activ since I don't use a-tz here
    (release diff-src-conn)
    (release diff-weights-tz)
    (release workspace))
  ;;TODO Implement equals etc.
  Info
  (info [this]
    (assoc (info activ) :shape (info bluep :shape)
           :shape (info bluep :shape)
           :bias (info bias-tz)
           :weights (info weights-tz)
           :dst (info dst-tz)
           :batch n :algorithm :adam :topology :fc))
  (info [this info-type]
    (case info-type
      :shape (info bluep :shape)
      :bias (info bias-tz)
      :weights (info weights-tz)
      :dst (info dst-tz)
      :batch n :algorithm :adam :topology :fc
      (or (info activ info-type) (info bluep info-type))))
  DiamondFactoryProvider
  (diamond-factory [_]
    fact)
  BlueprintProvider
  (blueprint [_]
    bluep)
  Parameters
  (bias [_]
    bias-tz)
  (weights [_]
    weights-tz)
  ParametersSeq
  (parameters [_]
    [weights-tz bias-tz])
  Transfer
  (input [this]
    (input src-conn))
  (output [_]
    (output activ))
  DiffTransfer
  (diff-input [_]
    (diff-input activ))
  (diff-z [_]
    (diff-output activ))
  (diff-output [_]
    (input src-conn))
  IFn
  (invoke [this]
    (src-conn)
    (convolution-fwd cudnn-hdl conv-desc conv-fwd-algo
                     one (output src-conn) (buffer (output src-conn))
                     filter-desc (buffer weights-tz) zero dst-tz (buffer dst-tz) workspace)
    (add-tensor cudnn-hdl one bias-tz (buffer bias-tz) one dst-tz (buffer dst-tz))
    (activ))
  Backprop
  (forward [this]
    (forward activ)
    this)
  (forward [this [_ _ mu nesterov?]]
    (when nesterov? (axpy! mu v w))
    (src-conn)
    (convolution-fwd cudnn-hdl conv-desc conv-fwd-algo
                     one (output src-conn) (buffer (output src-conn))
                     filter-desc (buffer weights-tz) zero dst-tz (buffer dst-tz) workspace)
    (add-tensor cudnn-hdl one bias-tz (buffer bias-tz) one dst-tz (buffer dst-tz))
    (forward activ)
    this)
  (backward [this]
    (backward activ)
    this)
  (backward [this [_ eta lambda mu nesterov?]]
    (let [eta-avg (- (/ (double eta) n))]
      (when nesterov? (axpy! (- (double mu)) v w))
      (convolution-bwd-filter cudnn-hdl conv-desc conv-bwd-weights-algo
                              one (output src-conn) (buffer (output src-conn))
                              dst-tz (buffer dst-tz)
                              zero filter-desc (buffer diff-weights-tz) workspace)
      (convolution-bwd-bias cudnn-hdl
                            (cast-prim (data-accessor dst-tz) eta) dst-tz (buffer dst-tz)
                            one bias-tz (buffer bias-tz))
      (when prop-diff?
        (convolution-bwd-data cudnn-hdl conv-desc conv-bwd-data-algo
                              one filter-desc (buffer weights-tz)
                              dst-tz (buffer dst-tz)
                              zero (input diff-src-conn) (buffer (input diff-src-conn))
                              workspace)
        (diff-src-conn))
      (axpby! eta-avg diff-weights-vec mu v)
      (axpby! 1.0 v (inc (* eta-avg (double lambda))) w)
      this)))

(defn sgd-layer [fact bluep activ-bluep val-1 val-0 scal-n prop-diff?
                 conv-desc filter-desc conv-fwd-algo conv-bwd-data-algo conv-bwd-weights-algo
                 src-conn bias-tz weights-tz z-tz a-tz
                 diff-weights-tz workspace]
  (let-release [n (get (shape z-tz) 0)
                w (view weights-tz)
                v (zero w)
                a (view a-tz)
                diff-weights-vec (view diff-weights-tz)
                activ (activ-bluep z-tz a-tz)
                diff-src-conn (revert src-conn)]
    (->CuDnnConvolutionSGDLayer fact (handle fact) bluep
                                val-1 val-0 scal-n n
                                activ prop-diff? conv-desc filter-desc
                                conv-fwd-algo conv-bwd-data-algo conv-bwd-weights-algo
                                v w diff-weights-vec
                                src-conn bias-tz weights-tz z-tz a-tz
                                diff-src-conn diff-weights-tz workspace)))

(deftype CuDnnConvolutionAdamLayer [fact cudnn-hdl bluep one zero scal-n ^long n
                                    activ prop-diff? conv-desc filter-desc
                                    conv-fwd-algo conv-bwd-data-algo conv-bwd-weights-algo
                                    g s r w
                                    src-conn bias-tz weights-tz dst-tz a-tz
                                    diff-src-conn diff-weights-tz workspace]
  Releaseable
  (release [_]
    (release activ)
    (release src-conn)
    (release bias-tz)
    (release weights-tz)
    (release dst-tz)
    (release a-tz);;TODO move the release part to activ since I don't use a-tz here
    (release diff-src-conn)
    (release diff-weights-tz)
    (release workspace))
  ;;TODO Implement equals etc.
  Info
  (info [this]
    (assoc (info activ) :shape (info bluep :shape)
           :shape (info bluep :shape)
           :bias (info bias-tz)
           :weights (info weights-tz)
           :dst (info dst-tz)
           :batch n :algorithm :adam :topology :fc))
  (info [this info-type]
    (case info-type
      :shape (info bluep :shape)
      :bias (info bias-tz)
      :weights (info weights-tz)
      :dst (info dst-tz)
      :batch n :algorithm :adam :topology :fc
      (or (info activ info-type) (info bluep info-type))))
  DiamondFactoryProvider
  (diamond-factory [_]
    fact)
  BlueprintProvider
  (blueprint [_]
    bluep)
  Parameters
  (bias [_]
    bias-tz)
  (weights [_]
    weights-tz)
  ParametersSeq
  (parameters [_]
    [weights-tz bias-tz])
  Transfer
  (input [this]
    (input src-conn))
  (output [_]
    (output activ))
  DiffTransfer
  (diff-input [_]
    (diff-input activ))
  (diff-z [_]
    (diff-output activ))
  (diff-output [_]
    (input src-conn))
  IFn
  (invoke [this]
    (src-conn)
    (convolution-fwd cudnn-hdl conv-desc conv-fwd-algo
                     one (output src-conn) (buffer (output src-conn))
                     filter-desc (buffer weights-tz) zero dst-tz (buffer dst-tz) workspace)
    (add-tensor cudnn-hdl one bias-tz (buffer bias-tz) one dst-tz (buffer dst-tz))
    (activ))
  Backprop
  (forward [this]
    (forward activ)
    this)
  (forward [this _]
    (src-conn)
    (convolution-fwd cudnn-hdl conv-desc conv-fwd-algo
                     one (output src-conn) (buffer (output src-conn))
                     filter-desc (buffer weights-tz) zero dst-tz (buffer dst-tz) workspace)
    (add-tensor cudnn-hdl one bias-tz (buffer bias-tz) one dst-tz (buffer dst-tz))
    (forward activ)
    this)
  (backward [this]
    (backward activ)
    this)
  (backward [this [t eta lambda rho1 rho2 epsilon]]
    (let [t (inc (long t))
          eta (double (or eta 0.001))
          lambda (double (or lambda 0.0))
          rho1 (double (or rho1 0.9))
          rho2 (double (or rho2 0.999))
          epsilon (double (or epsilon 1e-6))
          eta-avg (- (/ (double eta) n))]
      (convolution-bwd-filter cudnn-hdl conv-desc conv-bwd-weights-algo
                              scal-n (output src-conn) (buffer (output src-conn))
                              dst-tz (buffer dst-tz)
                              zero filter-desc (buffer diff-weights-tz) workspace)
      (convolution-bwd-bias cudnn-hdl
                            (cast-prim (data-accessor dst-tz) eta) dst-tz (buffer dst-tz)
                            one bias-tz (buffer bias-tz))
      (when prop-diff?
        (convolution-bwd-data cudnn-hdl conv-desc conv-bwd-data-algo
                              one filter-desc (buffer weights-tz)
                              dst-tz (buffer dst-tz)
                              zero (input diff-src-conn) (buffer (input diff-src-conn))
                              workspace)
        (diff-src-conn))
      (axpby! (- 1.0 rho1) g rho1 s);;TODO I might separate this into a optimization-algo type everywhere!
      (axpby! (- 1.0 rho2) (sqr! g) rho2 r)
      (linear-frac! (/ (- eta) (- 1.0 (pow rho1 t))) s 0.0
                    (/ 1.0 (sqrt (- 1.0 (pow rho2 t)))) (sqrt! r g) epsilon g)
      (axpby! 1.0 g (inc (* eta-avg lambda)) w)
      this)))

(defn adam-layer [fact bluep activ-bluep val-1 val-0 scal-n prop-diff?
                  conv-desc filter-desc conv-fwd-algo conv-bwd-data-algo conv-bwd-weights-algo
                  src-conn bias-tz weights-tz z-tz a-tz
                  diff-weights-tz workspace]
  (let-release [n (get (shape z-tz) 0)
                w (view weights-tz)
                a (view a-tz)
                g (raw w)
                s (zero w)
                r (zero w);;TODO rename zero value to val-0
                activ (activ-bluep z-tz a-tz)
                diff-src-conn (revert src-conn)]
    (->CuDnnConvolutionAdamLayer fact (handle fact) bluep
                                 val-1 val-0 scal-n n
                                 activ prop-diff? conv-desc filter-desc
                                 conv-fwd-algo conv-bwd-data-algo conv-bwd-weights-algo
                                 g s r w
                                 src-conn bias-tz weights-tz z-tz a-tz
                                 diff-src-conn diff-weights-tz workspace)))

(deftype CuDnnConvolutionLayerBlueprint [fact activ-bluep conv-desc
                                         conv-fwd-algo conv-bwd-data-algo conv-bwd-weights-algo
                                         src-desc weights-desc filter-desc bias-desc dst-desc]
  ;; TODO implement equals
  Releaseable
  (release [_]
    (release conv-desc)
    (release conv-fwd-algo)
    (release conv-bwd-data-algo)
    (release conv-bwd-weights-algo)
    (release weights-desc)
    (release filter-desc)
    (release bias-desc))
  Info
  (info [this info-type]
    (case info-type
      :bias bias-desc
      :inference {:src src-desc
                  :weights weights-desc
                  :filter filter-desc
                  :dst dst-desc}
      :training {:src src-desc
                 :weights weights-desc
                 :filter filter-desc
                 :dst dst-desc}
      nil))
  (info [this]
    {:bias bias-desc
     :inference {:src src-desc
                 :weights weights-desc
                 :filter filter-desc
                 :dst dst-desc}
     :training {:src src-desc
                :weights weights-desc
                :filter filter-desc
                :dst dst-desc}})
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
    (strides dst-desc))
  IFn
  (invoke [this prev-layer]
    (let-release [src-conn (connector (output prev-layer) src-desc)
                  bias-tz (cudnn-tensor fact bias-desc)
                  weights-tz (cudnn-tensor fact weights-desc)
                  a-tz (cudnn-tensor fact dst-desc)
                  workspace (mem-alloc (convolution-fwd-get-workspace-size
                                        (handle fact) conv-desc conv-fwd-algo
                                        src-desc filter-desc dst-desc))
                  activ (activ-bluep a-tz)]
      (->CuDnnConvolutionInferenceLayer fact (handle fact) this
                                        (cast-prim (data-accessor a-tz) 1.0)
                                        (cast-prim (data-accessor a-tz) 0.0)
                                        activ
                                        conv-desc filter-desc conv-fwd-algo
                                        src-conn bias-tz weights-tz a-tz workspace)))
  (invoke [this prev-layer prop-diff? optimization]
    (let [src-shape (shape src-desc)
          training-layer (case optimization
                           :sgd sgd-layer
                           :adam adam-layer
                           (dragan-says-ex
                            "This optimization algorithm is not available in cuDNN backend."
                            {:optimization optimization}))]
      (let-release [src-conn (connector (output (prev-layer)) src-desc)
                    bias-tz (cudnn-tensor fact bias-desc)
                    weights-tz (cudnn-tensor fact weights-desc)
                    dst-tz (cudnn-tensor fact dst-desc)
                    diff-src-conn (revert src-conn)
                    diff-weights-tz (raw weights-tz)
                    workspace (mem-alloc (max (long (convolution-fwd-get-workspace-size
                                                     (handle fact) conv-desc conv-fwd-algo
                                                     src-desc filter-desc dst-desc))
                                              (long (convolution-bwd-data-get-workspace-size
                                                     (handle fact) conv-desc conv-bwd-data-algo
                                                     filter-desc dst-desc src-desc))
                                              (long (convolution-bwd-filter-get-workspace-size
                                                     (handle fact) conv-desc conv-bwd-weights-algo
                                                     src-desc dst-desc filter-desc))))
                    a-tz (raw dst-tz)]
        (training-layer fact this activ-bluep
                        (cast-prim (data-accessor dst-tz) 1.0)
                        (cast-prim (data-accessor dst-tz) 0.0)
                        (cast-prim (data-accessor dst-tz) (/ 1.0 (long (get src-shape 0))))
                        prop-diff? conv-desc filter-desc
                        conv-fwd-algo conv-bwd-data-algo conv-bwd-weights-algo
                        src-conn bias-tz weights-tz dst-tz a-tz
                        diff-weights-tz workspace))))
  (invoke [this prev-layer prop-diff?]
    (.invoke this prev-layer prop-diff? :sgd)))

(defn cudnn-convolution-layer-blueprint
  [fact src-desc weights-desc dst-desc strides padding dilation
   activ alpha]
  (let-release [src-desc (desc src-desc)
                dst-desc (desc dst-desc)
                dtype (data-type dst-desc)
                weights-desc (cudnn-tensor-desc (shape weights-desc) dtype nil)
                filter-desc (filter-descriptor (shape weights-desc) dtype :nchw);;TODO generalize?
                bias-desc (cudnn-tensor-desc [1 (get (dims dst-desc) 1)] dtype :nc);;TODO maybe I'd need to do into etc.
                conv-desc (convolution-descriptor :cross-correleation dtype padding strides dilation)
                conv-fwd-algo (convolution-fwd-get-algo (handle fact) conv-desc
                                                        src-desc filter-desc dst-desc)
                conv-bwd-data-algo (convolution-bwd-data-get-algo (handle fact) conv-desc
                                                                  filter-desc dst-desc src-desc)
                conv-bwd-weights-algo (convolution-bwd-filter-get-algo (handle fact) conv-desc
                                                                       src-desc dst-desc filter-desc)
                activ-bluep (cudnn-activ-blueprint fact activ alpha)]
    (->CuDnnConvolutionLayerBlueprint fact activ-bluep conv-desc
                                      conv-fwd-algo conv-bwd-data-algo conv-bwd-weights-algo
                                      src-desc weights-desc filter-desc bias-desc dst-desc)))

(defmethod transfer! [CuDnnConvolutionInferenceLayer Object]
  [source destination]
  (transfer-weights-bias! source destination))

(defmethod transfer! [CuDnnConvolutionSGDLayer Object]
  [source destination]
  (transfer-weights-bias! source destination))

(defmethod transfer! [CuDnnConvolutionAdamLayer Object]
  [source destination]
  (transfer-weights-bias! source destination))

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
