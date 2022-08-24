(ns uncomplicate.diamond.internal.cudnn.directed
  (:require [uncomplicate.commons.core
             :refer [Releaseable release let-release with-release Info info view]]
            [uncomplicate.fluokitten.core :refer [fmap]]
            [uncomplicate.clojurecuda.core :refer [mem-alloc]]
            [uncomplicate.neanderthal
             [core :refer [axpby! axpy! copy! transfer! raw view-vctr entry! scal!]]
             [block :refer [cast-prim data-accessor buffer offset]]]
            [uncomplicate.diamond
             [tensor :as tz
              :refer [Transfer input output connector revert shape layout TensorDescriptor ]]]
            [uncomplicate.diamond.internal
             [protocols
              :refer [Parameters bias weights ParametersSeq parameters DescriptorProvider
                      DiamondFactoryProvider DiffParameters Backprop forward backward DiffTransfer
                      diff-input diff-output diff-z LinearBackprop backward-diff inf-desc train-desc
                      Initializable init Workspace inf-ws-size train-ws-size *workspace* create-tensor
                      batch-index]]
             [utils :refer [transfer-weights-bias! concat-strides concat-dst-shape direction-count
                            concat-offsets default-strides]]]
            [uncomplicate.diamond.internal.cudnn
             [core :refer :all]
             [protocols :refer :all]
             [tensor :refer [cudnn-tensor-desc cudnn-tensor cudnn-transformer]]]
            [uncomplicate.diamond.internal.neanderthal.directed
             :refer [->DirectedLayerBlueprint ->GaussianDropoutBlueprint ->NopActivation
                     ->NopActivationBlueprint sgd-layer adam-layer]])
  (:import [clojure.lang IFn AFn]
           [uncomplicate.diamond.internal.neanderthal.directed
            InnerProductBlueprint DirectedLayerBlueprint GaussianDropoutBlueprint]))

(defn cudnn-contiguous-desc [md]
  (let [s (shape md)]
    (if (and (= :float (data-type md))
             (= (size md) (apply * Float/BYTES s)))
      (view md)
      (cudnn-tensor-desc s :float (default-strides s)))))

;; ================================ Activation =============================================

(deftype CUDnnActivationInference [cudnn-hdl bluep activation-desc data-tz one zero linear]
  Releaseable
  (release [_]
    (release data-tz))
  Info
  (info [this]
    {:activation (info bluep :activation)
     :data (info data-tz)})
  (info [this info-type]
    (case info-type
      :data (info data-tz)
      (info bluep info-type)))
  Transfer
  (input [_]
    data-tz)
  (output [_]
    data-tz)
  IFn
  (invoke [_]
    (when-not linear
      (activation-forward cudnn-hdl activation-desc
                          one data-tz (buffer data-tz)
                          zero data-tz (buffer data-tz)))
    data-tz)
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(deftype CUDnnLinearActivationTraining [cudnn-hdl bluep activation-desc
                                        src-tz dst-tz diff-src-tz one zero]
  Releaseable
  (release [_]
    (release src-tz)
    (release dst-tz)
    (release diff-src-tz))
  Info
  (info [this]
    {:activation (info bluep :activation)
     :src (info src-tz)
     :dst (info dst-tz)
     :diff-src (info diff-src-tz)})
  (info [this info-type]
    (case info-type
      :src (info src-tz)
      :dst (info dst-tz)
      :diff-src (info diff-src-tz)
      (info bluep info-type)))
  Transfer
  (input [_]
    src-tz)
  (output [_]
    dst-tz)
  DiffTransfer
  (diff-input [_]
    dst-tz)
  (diff-output [_]
    diff-src-tz)
  IFn
  (invoke [this]
    (forward this)
    dst-tz)
  (applyTo [this xs]
    (AFn/applyToHelper this xs))
  Backprop
  (forward [this]
    (copy! src-tz dst-tz)
    this)
  (backward [this]
    (copy! dst-tz diff-src-tz)
    this))

(deftype CUDnnActivationTraining [cudnn-hdl bluep activation-desc
                                  src-tz dst-tz diff-dst-tz diff-src-tz
                                  one zero]
  Releaseable
  (release [_]
    (release src-tz)
    (release dst-tz)
    (release diff-dst-tz)
    (release diff-src-tz))
  Info
  (info [this]
    {:activation (info bluep :activation)
     :src (info src-tz)
     :dst (info dst-tz)
     :diff-dst (info diff-dst-tz)
     :diff-src (info diff-src-tz)})
  (info [this info-type]
    (case info-type
      :src (info src-tz)
      :dst (info dst-tz)
      :diff-dst (info diff-dst-tz)
      :diff-src (info diff-src-tz)
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
    diff-src-tz)
  IFn
  (invoke [_]
    (activation-forward cudnn-hdl activation-desc
                        one src-tz (buffer src-tz) zero dst-tz (buffer dst-tz))
    dst-tz)
  (applyTo [this xs]
    (AFn/applyToHelper this xs))
  Backprop
  (forward [this]
    (activation-forward cudnn-hdl activation-desc
                        one src-tz (buffer src-tz) zero dst-tz (buffer dst-tz))
    this)
  (backward [this]
    (activation-backward cudnn-hdl activation-desc
                         one dst-tz (buffer dst-tz) diff-dst-tz (buffer diff-dst-tz) src-tz (buffer src-tz)
                         zero diff-src-tz (buffer diff-src-tz))
    this))

(deftype CUDnnActivationBlueprint [fact activ ad inf-desc train-desc diff-desc]
  Releaseable
  (release [_]
    (release inf-desc)
    (release train-desc)
    (release diff-desc)
    (release ad))
  Info
  (info [this]
    {:activation activ})
  (info [this info-type]
    (case info-type
      :activation activ
      nil))
  DiamondFactoryProvider
  (diamond-factory [_]
    fact)
  DescriptorProvider
  (inf-desc [_]
    (view inf-desc))
  (train-desc [_]
    (view train-desc))
  (diff-desc [_]
    (view diff-desc))
  TensorDescriptor
  (shape [_]
    (shape train-desc))
  (data-type [_]
    (data-type train-desc))
  (layout [_]
    (layout train-desc))
  IFn
  (invoke [this src-tz]
    (->CUDnnActivationInference (handle fact) this ad src-tz
                                (cast-prim (data-accessor src-tz) 1.0)
                                (cast-prim (data-accessor src-tz) 0.0)
                                (or (= :linear activ) (= :identity activ))))
  (invoke [this src-tz diff-src-tz]
    (let-release [dst-tz (cudnn-tensor fact (view diff-desc) (batch-index diff-src-tz))]
      (cond
        (or (= :linear activ) (= :identity activ))
        (->CUDnnLinearActivationTraining (handle fact) this ad src-tz dst-tz diff-src-tz
                                         (cast-prim (data-accessor src-tz) 1.0)
                                         (cast-prim (data-accessor src-tz) 0.0))
        (or (= :sigmoid activ) (:logistic activ))
        (let-release [diff-dst-tz (cudnn-tensor fact (view diff-desc) (batch-index dst-tz))]
          (->CUDnnActivationTraining (handle fact) this ad src-tz dst-tz diff-dst-tz diff-src-tz
                                     (cast-prim (data-accessor src-tz) 1.0)
                                     (cast-prim (data-accessor src-tz) 0.0)))
        :default
        (->CUDnnActivationTraining (handle fact) this ad src-tz dst-tz dst-tz diff-src-tz
                                   (cast-prim (data-accessor src-tz) 1.0)
                                   (cast-prim (data-accessor src-tz) 0.0)))))
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

;; ================================ Softmax =============================================

(deftype CUDnnSoftmaxInference [cudnn-hdl bluep data-tz one zero]
  Releaseable
  (release [_]
    (release data-tz))
  Info
  (info [this]
    {:activation :softmax
     :data (info data-tz)})
  (info [this info-type]
    (case info-type
      :activation :softmax
      :data (info data-tz)
      (info bluep info-type)))
  Transfer
  (input [_]
    data-tz)
  (output [_]
    data-tz)
  IFn
  (invoke [_]
    (softmax-forward cudnn-hdl :accurate :instance
                     one data-tz (buffer data-tz) zero data-tz (buffer data-tz))
    data-tz)
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(deftype CUDnnSoftmaxTraining [cudnn-hdl bluep data-tz diff-dst-tz diff-src-tz one zero]
  Releaseable
  (release [_]
    (release data-tz)
    (release diff-dst-tz))
  Info
  (info [this]
    {:activation :softmax
     :data (info data-tz)
     :diff-dst (info diff-dst-tz)})
  (info [this info-type]
    (case info-type
      :data (info data-tz)
      :diff-dst (info diff-dst-tz)
      (info bluep info-type)))
  Transfer
  (input [_]
    data-tz)
  (output [_]
    data-tz)
  DiffTransfer
  (diff-input [_]
    diff-dst-tz)
  (diff-output [_]
    diff-src-tz)
  IFn
  (invoke [_]
    (softmax-forward cudnn-hdl :accurate :instance
                     one data-tz (buffer data-tz) zero data-tz (buffer data-tz))
    data-tz)
  (applyTo [this xs]
    (AFn/applyToHelper this xs))
  Backprop
  (forward [this]
    (softmax-forward cudnn-hdl :accurate :instance
                     one data-tz (buffer data-tz) zero data-tz (buffer data-tz))
    this)
  (backward [this]
    (softmax-backward cudnn-hdl :accurate :instance
                      one data-tz (buffer data-tz) diff-dst-tz (buffer diff-dst-tz)
                      zero diff-src-tz (buffer diff-src-tz))
    this))

(deftype CUDnnSoftmaxBlueprint [fact inf-desc train-desc diff-desc]
  Releaseable
  (release [_]
    (release inf-desc)
    (release train-desc)
    (release diff-desc))
  Info
  (info [this]
    {:activation :softmax})
  (info [this info-type]
    (case info-type
      :activation :softmax
      nil))
  DiamondFactoryProvider
  (diamond-factory [_]
    fact)
  DescriptorProvider
  (inf-desc [_]
    (view inf-desc))
  (train-desc [_]
    (view train-desc))
  (diff-desc [_]
    (view diff-desc))
  TensorDescriptor
  (shape [_]
    (shape train-desc))
  (data-type [_]
    (data-type train-desc))
  (layout [_]
    (layout train-desc))
  IFn
  (invoke [this src-tz]
    (->CUDnnSoftmaxInference (handle fact) this src-tz
                             (cast-prim (data-accessor src-tz) 1.0)
                             (cast-prim (data-accessor src-tz) 0.0)))
  (invoke [this src-tz diff-src-tz]
    (let-release [diff-dst-tz (cudnn-tensor fact (view diff-desc) (batch-index diff-src-tz))]
      (->CUDnnSoftmaxTraining (handle fact) this src-tz diff-dst-tz diff-src-tz
                              (cast-prim (data-accessor src-tz) 1.0)
                              (cast-prim (data-accessor src-tz) 0.0))))
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(defn cudnn-activ-blueprint
  ([fact inf-desc train-desc diff-desc activ coef]
   (let [inf-desc (desc inf-desc)
         train-desc (desc train-desc)
         diff-desc (desc diff-desc)]
     (case activ
       :identity (->NopActivationBlueprint fact inf-desc train-desc diff-desc)
       :softmax (->CUDnnSoftmaxBlueprint fact inf-desc train-desc diff-desc)
       (let-release [ad (activation-descriptor activ true coef)]
         (->CUDnnActivationBlueprint fact activ ad inf-desc train-desc diff-desc)))))
  ([fact data-desc activ coef]
   (cudnn-activ-blueprint fact data-desc data-desc data-desc activ coef)))

;; ============================= Cost Function ========================================

(deftype CUDnnUniversalCost [prev-layer
                             connect-output connect-diff train-tz
                             a-y y cost]
  Releaseable
  (release [_]
    (release connect-output)
    (release connect-diff)
    (release train-tz)
    (release a-y)
    (release y))
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
    (cost a-y))
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(defn cudnn-universal-cost [prev-layer train-tz cost]
  (let [train-desc (desc train-tz)]
    (let-release [connect-output (connector (output prev-layer) train-desc)
                  connect-diff (connector train-desc (diff-input prev-layer))]
      (->CUDnnUniversalCost prev-layer
                            connect-output connect-diff train-tz
                            (view-vctr (input connect-diff)) (view-vctr train-tz)
                            cost))))

(deftype CUDnnCustomCost [prev-layer
                          connect-output connect-diff train-tz
                          a y a-y cost]
  Releaseable
  (release [_]
    (release connect-output)
    (release connect-diff)
    (release train-tz)
    (release a)
    (release y)
    (release a-y))
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
    (copy! a a-y)
    (axpy! -1.0 y a-y)
    (connect-diff)
    this)
  IFn
  (invoke [_]
    (connect-output)
    (cost y a))
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(defn cudnn-custom-cost [prev-layer train-tz cost]
  (let [train-desc (desc train-tz)]
    (let-release [connect-output (connector (output prev-layer) train-desc)
                  connect-diff (connector train-desc (diff-z prev-layer))]
      (->CUDnnCustomCost prev-layer
                         connect-output connect-diff train-tz
                         (view-vctr (output connect-output)) (view-vctr train-tz)
                         (view-vctr (input connect-diff))
                         cost))))

;; ================================ Convolution =====================================

(deftype CUDnnConvolutionInference [fact cudnn-hdl bluep one zero
                                    conv-desc filter-desc conv-fwd-algo
                                    src-conn bias-tz weights-tz dst-tz workspace]
  Releaseable
  (release [_]
    (release src-conn)
    (release bias-tz)
    (release weights-tz)
    (release dst-tz)
    (release workspace)
    (release filter-desc))
  Info
  (info [this]
    {:bias (info bias-tz)
     :weights (info weights-tz)
     :dst (info dst-tz)})
  (info [this info-type]
    (case info-type
      :bias (info bias-tz)
      :weights (info weights-tz)
      :dst (info dst-tz)
      nil))
  Transfer
  (input [_]
    (input src-conn))
  (output [_]
    dst-tz)
  Parameters
  (bias [_]
    bias-tz)
  (weights [_]
    weights-tz)
  ParametersSeq
  (parameters [_]
    [weights-tz bias-tz])
  IFn
  (invoke [_]
    (src-conn)
    (convolution-fwd cudnn-hdl conv-desc conv-fwd-algo
                     one (output src-conn) (buffer (output src-conn))
                     filter-desc (buffer weights-tz) zero dst-tz (buffer dst-tz) workspace)
    (add-tensor cudnn-hdl one bias-tz (buffer bias-tz) one dst-tz (buffer dst-tz))
    dst-tz)
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(deftype CUDnnConvolutionTraining [fact cudnn-hdl bluep da one zero
                                   prop-diff? conv-desc filter-desc
                                   conv-fwd-algo conv-bwd-data-algo conv-bwd-weights-algo
                                   src-conn bias-tz weights-tz dst-tz
                                   diff-weights-tz diff-src-conn workspace]
  Releaseable
  (release [_]
    (release src-conn)
    (release bias-tz)
    (release weights-tz)
    (release dst-tz)
    (release diff-weights-tz)
    (release diff-src-conn)
    (release workspace)
    (release filter-desc))
  Info
  (info [this]
    {:bias (info bias-tz)
     :weights (info weights-tz)
     :dst (info dst-tz)
     :diff-weights (info diff-weights-tz)})
  (info [this info-type]
    (case info-type
      :bias (info bias-tz)
      :weights (info weights-tz)
      :dst (info dst-tz)
      :diff-weights (info diff-weights-tz)
      nil))
  Transfer
  (input [_]
    (input src-conn))
  (output [_]
    dst-tz)
  DiffTransfer
  (diff-input [_]
    dst-tz)
  (diff-output [_]
    (output diff-src-conn))
  Parameters
  (bias [_]
    bias-tz)
  (weights [_]
    weights-tz)
  ParametersSeq
  (parameters [_]
    [weights-tz bias-tz])
  DiffParameters
  (diff-weights [_]
    diff-weights-tz)
  IFn
  (invoke [this]
    (forward this)
    dst-tz)
  (applyTo [this xs]
    (AFn/applyToHelper this xs))
  Backprop
  (forward [this]
    (src-conn)
    (convolution-fwd cudnn-hdl conv-desc conv-fwd-algo
                     one (output src-conn) (buffer (output src-conn))
                     filter-desc (buffer weights-tz) zero dst-tz (buffer dst-tz) workspace)
    (add-tensor cudnn-hdl one bias-tz (buffer bias-tz) one dst-tz (buffer dst-tz))
    this)
  (backward [this]
    (backward-diff this one zero one zero))
  LinearBackprop
  (backward-diff [this scal-diff-w scal-g scal-diff-b scal-b]
    (convolution-bwd-filter cudnn-hdl conv-desc conv-bwd-weights-algo
                            (cast-prim da scal-diff-w)
                            (output src-conn) (buffer (output src-conn))
                            dst-tz (buffer dst-tz)
                            (cast-prim da scal-g) filter-desc (buffer diff-weights-tz)
                            workspace)
    (convolution-bwd-bias cudnn-hdl
                          (cast-prim da scal-diff-b) dst-tz (buffer dst-tz)
                          (cast-prim da scal-b) bias-tz (buffer bias-tz))
    (when prop-diff?
      (convolution-bwd-data cudnn-hdl conv-desc conv-bwd-data-algo
                            one filter-desc (buffer weights-tz)
                            dst-tz (buffer dst-tz)
                            zero (input diff-src-conn) (buffer (input diff-src-conn))
                            workspace)
      (diff-src-conn))
    this))

(deftype CUDnnConvolutionBlueprint [fact conv-desc
                                    conv-fwd-algo conv-bwd-data-algo conv-bwd-weights-algo
                                    src-desc weights-desc filter-desc bias-desc dst-desc]
  Releaseable
  (release [_]
    (release conv-desc)
    (release conv-fwd-algo)
    (release conv-bwd-data-algo)
    (release conv-bwd-weights-algo)
    (release src-desc)
    (release weights-desc)
    (release filter-desc)
    (release bias-desc)
    (release dst-desc))
  Object
  (hashCode [_]
    (-> (hash :convolution)
        (hash-combine src-desc) (hash-combine weights-desc)
        (hash-combine bias-desc) (hash-combine dst-desc)))
  (equals [_ other]
    (and (instance? CUDnnConvolutionBlueprint other)
         (equal-desc? src-desc (.src-desc ^CUDnnConvolutionBlueprint other))
         (equal-desc? weights-desc (.weights-desc ^CUDnnConvolutionBlueprint other))
         (equal-desc? dst-desc (.dst-desc ^CUDnnConvolutionBlueprint other))))
  (toString [this]
    (pr-str {:src src-desc :weights weights-desc :dst dst-desc}))
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
  DescriptorProvider
  (inf-desc [_]
    (view dst-desc))
  (train-desc [_]
    (view dst-desc))
  (diff-desc [_]
    (view dst-desc))
  TensorDescriptor
  (shape [_]
    (shape dst-desc))
  (data-type [_]
    (data-type dst-desc))
  (layout [_]
    (strides dst-desc))
  Workspace
  (inf-ws-size [this]
    (:workspace-size conv-fwd-algo))
  (train-ws-size [this]
    (max (long (:workspace-size conv-fwd-algo))
         (long (:workspace-size conv-bwd-data-algo))
         (long (:workspace-size conv-bwd-weights-algo))))
  IFn
  (invoke [this src-tz]
    (let-release [src-conn (connector src-tz src-desc)
                  bias-tz (cudnn-tensor fact (view bias-desc))
                  weights-tz (cudnn-tensor fact (view weights-desc))
                  a-tz (cudnn-tensor fact (view dst-desc))]
      (->CUDnnConvolutionInference fact (handle fact) this
                                   (cast-prim (data-accessor a-tz) 1.0)
                                   (cast-prim (data-accessor a-tz) 0.0)
                                   conv-desc (view filter-desc) (:algo conv-fwd-algo)
                                   src-conn bias-tz weights-tz a-tz *workspace*)))
  (invoke [this src-tz diff-src-tz prop-diff? _]
    (let [da (data-accessor src-tz)]
      (let-release [src-conn (connector src-tz src-desc)
                    bias-tz (cudnn-tensor fact (view bias-desc))
                    weights-tz (cudnn-tensor fact (view weights-desc))
                    dst-tz (cudnn-tensor fact dst-desc (batch-index src-tz))
                    diff-src-conn (connector src-desc diff-src-tz)
                    diff-weights-tz (create-tensor fact (view weights-desc) true)]
        (->CUDnnConvolutionTraining fact (handle fact) this da
                                    (cast-prim da 1.0) (cast-prim da 0.0)
                                    prop-diff? conv-desc (view filter-desc)
                                    (:algo conv-fwd-algo) (:algo conv-bwd-data-algo)
                                    (:algo conv-bwd-weights-algo)
                                    src-conn bias-tz weights-tz dst-tz
                                    diff-weights-tz diff-src-conn
                                    *workspace*))))
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(defn cudnn-convolution-op-blueprint
  [fact src-desc weights-desc dst-desc strides padding dilation]
  (let-release [src-desc (desc src-desc)
                dst-desc (desc dst-desc)
                dtype (data-type dst-desc)
                weights-desc (cudnn-tensor-desc (shape weights-desc) dtype :nchw)
                filter-desc (filter-descriptor (shape weights-desc) dtype :nchw)
                bias-desc (cudnn-tensor-desc [1 (get (dims dst-desc) 1)] dtype :nc)
                conv-desc (convolution-descriptor :cross-correleation dtype padding strides dilation)
                conv-fwd-algo (convolution-fwd-find-algo (handle fact) conv-desc
                                                         src-desc filter-desc dst-desc)
                conv-bwd-data-algo (convolution-bwd-data-find-algo (handle fact) conv-desc
                                                                   filter-desc dst-desc src-desc)
                conv-bwd-weights-algo (convolution-bwd-filter-find-algo (handle fact) conv-desc
                                                                        src-desc dst-desc filter-desc)]
    (->CUDnnConvolutionBlueprint fact conv-desc
                                 conv-fwd-algo conv-bwd-data-algo conv-bwd-weights-algo
                                 src-desc weights-desc filter-desc bias-desc dst-desc)))

(defn cudnn-convolution-layer-blueprint [fact src-desc weights-desc dst-desc activ
                                         strides padding dilation alpha]
  (let [dtype (or (tz/data-type src-desc) :float)]
    (let-release [src-desc (cudnn-tensor-desc (shape src-desc) dtype (layout src-desc))
                  dst-desc (cudnn-tensor-desc (shape dst-desc)
                                              (or (tz/data-type dst-desc) dtype)
                                              (layout dst-desc))
                  convolution-bluep (cudnn-convolution-op-blueprint
                                     fact src-desc weights-desc dst-desc strides padding dilation)
                  activ-bluep (cudnn-activ-blueprint fact (view dst-desc) activ alpha)]
      (->DirectedLayerBlueprint fact :convolution convolution-bluep activ-bluep))))

(defmethod transfer! [CUDnnConvolutionInference Object]
  [source destination]
  (transfer-weights-bias! source destination))

(defmethod transfer! [CUDnnConvolutionTraining Object]
  [source destination]
  (transfer-weights-bias! source destination))

;; ================================ Pooling =============================================

(deftype CUDnnPoolingInferenceLayer [fact cudnn-hdl bluep pooling-desc
                                     src-tz dst-tz one zero]
  Releaseable
  (release [_]
    (release src-tz)
    (release dst-tz))
  Object
  (hashCode [_]
    (-> (hash :pooling)
        (hash-combine (shape src-tz))
        (hash-combine (shape dst-tz))))
  (equals [_ other]
    (and (instance? CUDnnPoolingInferenceLayer other)
         (= src-tz (.src-tz ^CUDnnPoolingInferenceLayer other))
         (= dst-tz (.dst-tz ^CUDnnPoolingInferenceLayer other))))
  (toString [this]
    (str bluep))
  Info
  (info [this]
    {:algo (info bluep :algo)
     :dst (info dst-tz)
     :shape (shape dst-tz)})
  (info [this info-type]
    (case info-type
      :algo (info bluep :algo)
      :dst (info dst-tz)
      (info bluep info-type)))
  DiamondFactoryProvider
  (diamond-factory [_]
    fact)
  Transfer
  (input [_]
    src-tz)
  (output [_]
    dst-tz)
  ParametersSeq
  (parameters [_]
    [])
  Initializable
  (init [this _]
    this)
  IFn
  (invoke [_]
    (pooling-forward cudnn-hdl pooling-desc
                     one src-tz (buffer src-tz) zero dst-tz (buffer dst-tz))
    dst-tz)
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(defmethod print-method CUDnnPoolingInferenceLayer
  [^CUDnnPoolingInferenceLayer layer ^java.io.Writer w]
  (.write w (format "#Pooling[shape:%s, algo:%s]\n destination: %s\n"
                    (shape (output layer)) (info layer :algo) (pr-str (.dst-tz layer)))))

(deftype CUDnnPoolingTrainingLayer [fact cudnn-hdl bluep pooling-desc
                                    src-tz dst-tz diff-dst-tz diff-src-tz
                                    one zero prop-diff?]
  Releaseable
  (release [_]
    (release src-tz)
    (release dst-tz)
    (release diff-dst-tz)
    (release diff-src-tz))
  Object
  (hashCode [_]
    (-> (hash :pooling)
        (hash-combine (shape src-tz))
        (hash-combine (shape dst-tz))))
  (equals [_ other]
    (and (instance? CUDnnPoolingTrainingLayer other)
         (= src-tz (.src-tz ^CUDnnPoolingTrainingLayer other))
         (= dst-tz (.dst-tz ^CUDnnPoolingTrainingLayer other))))
  (toString [this]
    (str bluep))
  Info
  (info [this]
    {:algo (info bluep :algo)
     :dst (info dst-tz)
     :shape (shape dst-tz)})
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
    diff-src-tz)
  ParametersSeq
  (parameters [_]
    [])
  Initializable
  (init [this _]
    this)
  IFn
  (invoke [this]
    (forward this nil)
    dst-tz)
  (applyTo [this xs]
    (AFn/applyToHelper this xs))
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
                        src-tz (buffer src-tz) zero src-tz (buffer diff-src-tz)))
    this))

(defmethod print-method CUDnnPoolingTrainingLayer
  [^CUDnnPoolingTrainingLayer layer ^java.io.Writer w]
  (.write w (format "#Pooling[shape:%s, algo:%s]\n destination: %s\n"
                    (shape (output layer)) (info layer :algo) (pr-str (.dst-tz layer)))))

(deftype CUDnnPoolingBlueprint [fact algo pd dst-desc]
  Releaseable
  (release [_]
    (release pd)
    (release dst-desc))
  Object
  (hashCode [this]
    (-> (hash :pooling)
        (hash-combine algo)
        (hash-combine (train-desc this))))
  (equals [this other]
    (and (instance? CUDnnPoolingBlueprint other)
         (= algo (.algo ^CUDnnPoolingBlueprint other))
         (= (inf-desc this) (inf-desc other))
         (= (train-desc this) (train-desc other))))
  (toString [this]
    (str {:algo algo
          :shape (shape this)
          :topology :pooling}))
  Info
  (info [this]
    {:algo algo
     :shape (shape dst-desc)
     :topology :pooling})
  (info [this info-type]
    (case info-type
      :algo algo
      :shape (shape dst-desc)
      :topology :pooling
      nil))
  DiamondFactoryProvider
  (diamond-factory [_]
    fact)
  DescriptorProvider
  (inf-desc [_]
    (view dst-desc))
  (train-desc [_]
    (view dst-desc))
  (diff-desc [_]
    "TODO")
  TensorDescriptor
  (shape [_]
    (shape dst-desc))
  (data-type [_]
    (tz/data-type dst-desc))
  (layout [_]
    (layout dst-desc))
  IFn
  (invoke [this prev-layer]
    (let-release [dst-tz (cudnn-tensor fact (view dst-desc))]
      (->CUDnnPoolingInferenceLayer fact (handle fact) this pd
                                    (view (output prev-layer)) dst-tz
                                    (cast-prim (data-accessor dst-tz) 1.0)
                                    (cast-prim (data-accessor dst-tz) 0.0))))
  (invoke [this prev-layer prop-diff? _]
    (let-release [dst-tz (cudnn-tensor fact (view dst-desc))
                  diff-dst-tz (cudnn-tensor fact (view dst-desc))]
      (->CUDnnPoolingTrainingLayer fact (handle fact) this pd
                                   (view (output prev-layer)) dst-tz
                                   diff-dst-tz (view (diff-input prev-layer))
                                   (cast-prim (data-accessor dst-tz) 1.0)
                                   (cast-prim (data-accessor dst-tz) 0.0)
                                   prop-diff?)))
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(defmethod print-method CUDnnPoolingBlueprint
  [bp ^java.io.Writer w]
  (.write w (str bp)))

(defn cudnn-pooling-blueprint
  [fact src-desc dst-desc algo strides kernel padding]
  (let-release [dst-desc (cudnn-tensor-desc (shape dst-desc)
                                            (or (tz/data-type dst-desc) (data-type src-desc))
                                            (or (tz/layout dst-desc) (default-strides (shape dst-desc))))
                pool-desc (pooling-descriptor algo kernel strides padding)]
    (->CUDnnPoolingBlueprint fact algo pool-desc dst-desc)))

(defmethod transfer! [CUDnnPoolingInferenceLayer Object]
  [source destination]
  destination)

(defmethod transfer! [CUDnnPoolingTrainingLayer Object]
  [source destination]
  destination)

;; ====================== Dropout ====================================================

(defn cudnn-gaussian-dropout-blueprint [fact src-desc sd]
  (let-release [mask-desc (cudnn-contiguous-desc (desc src-desc))]
    (->GaussianDropoutBlueprint fact sd mask-desc)))

;; ====================== Batch Normalization =======================================

(deftype CUDnnBatchNormalizationInference [fact cudnn-hdl bluep one zero param-desc
                                           src-conn gamma-tz beta-tz mean-tz var-tz]
  Releaseable
  (release [_]
    (release src-conn)
    (release gamma-tz)
    (release beta-tz)
    (release mean-tz)
    (release var-tz))
  Object
  (hashCode [_]
    (-> (hash :batch-norm)
        (hash-combine (shape gamma-tz))
        (hash-combine (shape beta-tz))
        (hash-combine (shape (input src-conn)))
        (hash-combine (shape (output src-conn)))))
  (equals [_ other]
    (and (instance? CUDnnBatchNormalizationInference other)
         (let [other ^CUDnnBatchNormalizationInference other]
           (and
            (= gamma-tz (.gamma-tz other))
            (= beta-tz (.beta-tz other))
            (= src-conn (.src-conn other))
            (= mean-tz (.mean-tz other))
            (= var-tz (.var-tz other))
            (= (output src-conn) (output (.src-conn other)))))))
  (toString [this]
    (str bluep))
  Info
  (info [this]
    {:gamma (info gamma-tz)
     :beta (info beta-tz)
     :dst (info (output src-conn))})
  (info [this info-type]
    (case info-type
      :gamma (info gamma-tz)
      :beta (info beta-tz)
      :dst (info (output src-conn))
      nil))
  DiamondFactoryProvider
  (diamond-factory [_]
    fact)
  Transfer
  (input [_]
    (input src-conn))
  (output [_]
    (output src-conn))
  Parameters
  (weights [_]
    gamma-tz)
  (bias [_]
    beta-tz)
  ParametersSeq
  (parameters [_]
    [gamma-tz beta-tz mean-tz var-tz])
  Initializable
  (init [this init-fn]
    (init-fn gamma-tz)
    (init-fn beta-tz)
    this)
  IFn
  (invoke [_]
    (src-conn)
    (batch-norm-fwd-inference cudnn-hdl :spatial one zero
                              (output src-conn) (buffer (output src-conn))
                              (output src-conn) (buffer (output src-conn))
                              param-desc (buffer gamma-tz) (buffer beta-tz)
                              (buffer mean-tz) (buffer var-tz))
    (output src-conn))
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(defmethod print-method CUDnnBatchNormalizationInference
  [layer ^java.io.Writer w]
  (.write w (format "#BatchNorm[]\n gamma:\n beta: \n output: %s\n"
                    (weights layer) (bias layer) (pr-str (output layer)))))

(deftype CUDnnBatchNormalizationTraining [fact cudnn-hdl bluep da cnt un-bessel one zero
                                          param-desc src-conn gamma-tz beta-tz dst-tz
                                          mean-tz var-tz saved-mean-tz saved-inv-var-tz
                                          diff-gamma-tz diff-beta-tz diff-src-conn]
  Releaseable
  (release [_]
    (release src-conn)
    (release gamma-tz)
    (release beta-tz)
    (release dst-tz)
    (release mean-tz)
    (release var-tz)
    (release saved-mean-tz)
    (release saved-inv-var-tz)
    (release diff-gamma-tz)
    (release diff-beta-tz)
    (release diff-src-conn))
  Object
  (hashCode [_]
    (-> (hash :batch-norm)
        (hash-combine (shape gamma-tz))
        (hash-combine (shape beta-tz))
        (hash-combine (shape (input src-conn)))
        (hash-combine (shape (output src-conn)))))
  (equals [_ other]
    (and (instance? CUDnnBatchNormalizationInference other)
         (let [other ^CUDnnBatchNormalizationTraining other]
           (and
            (= gamma-tz (.gamma-tz other))
            (= beta-tz (.beta-tz other))
            (= src-conn (.src-conn other))
            (= mean-tz (.mean-tz other))
            (= var-tz (.var-tz other))
            (= dst-tz (output (.dst-tz other)))))))
  (toString [this]
    (str bluep))
  Info
  (info [this]
    {:gamma (info gamma-tz)
     :beta (info beta-tz)
     :dst (info dst-tz)
     :mean (info mean-tz)
     :variance (info var-tz)
     :diff-diff-gamma (info diff-gamma-tz)})
  (info [this info-type]
    (case info-type
      :gamma (info gamma-tz)
      :beta (info beta-tz)
      :dst (info dst-tz)
      :mean (info mean-tz)
      :variance (info var-tz)
      :diff-diff-gamma (info diff-gamma-tz)
      nil))
  DiamondFactoryProvider
  (diamond-factory [_]
    fact)
  Transfer
  (input [_]
    (input src-conn))
  (output [_]
    dst-tz)
  DiffTransfer
  (diff-input [_]
    dst-tz)
  (diff-output [_]
    (output diff-src-conn))
  Parameters
  (weights [_]
    gamma-tz)
  (bias [_]
    beta-tz)
  ParametersSeq
  (parameters [_]
    [gamma-tz beta-tz mean-tz var-tz])
  Initializable
  (init [this init-fn]
    (init-fn gamma-tz)
    (init-fn beta-tz)
    (reset! cnt 0)
    this)
  DiffParameters
  (diff-weights [_]
    diff-gamma-tz)
  IFn
  (invoke [this]
    (src-conn)
    (batch-norm-fwd-training cudnn-hdl :spatial one zero
                             (output src-conn) (buffer (output src-conn))
                             dst-tz (buffer dst-tz)
                             param-desc (buffer gamma-tz) (buffer beta-tz) @cnt
                             (buffer mean-tz) (buffer var-tz)
                             (buffer saved-mean-tz) (buffer saved-inv-var-tz))
    (scal! un-bessel var-tz)
    dst-tz)
  Backprop
  (forward [this]
    (src-conn)
    (batch-norm-fwd-training cudnn-hdl :spatial one zero
                             (output src-conn) (buffer (output src-conn))
                             dst-tz (buffer dst-tz)
                             param-desc (buffer gamma-tz) (buffer beta-tz) (swap! cnt inc)
                             (buffer mean-tz) (buffer var-tz)
                             (buffer saved-mean-tz) (buffer saved-inv-var-tz))
    (scal! un-bessel var-tz)
    this)
  (backward [this]
    (backward-diff this 1.0 0.0 1.0 0.0))
  LinearBackprop
  (backward-diff [this scal-diff-w scal-g _ scal-b]
    (entry! diff-beta-tz 0.0)
    (batch-norm-bwd cudnn-hdl :spatial one zero
                    (cast-prim da scal-diff-w) (cast-prim da scal-g)
                    (output src-conn) (buffer (output src-conn))
                    dst-tz (buffer dst-tz) (input diff-src-conn) (buffer (input diff-src-conn))
                    param-desc (buffer gamma-tz) (buffer diff-gamma-tz) (buffer diff-beta-tz)
                    (buffer saved-mean-tz) (buffer saved-inv-var-tz))
    (axpby! 1.0 diff-beta-tz scal-b beta-tz)
    (diff-src-conn)
    this)
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(defmethod print-method CUDnnBatchNormalizationTraining
  [layer ^java.io.Writer w]
  (.write w (format "#BatchNorm[]\n gamma:\n beta: \n output: %s\n"
                    (weights layer) (bias layer) (pr-str (output layer)))))


(deftype CUDnnBatchNormalizationBlueprint [fact data-desc gamma-desc un-bessel]
  Releaseable
  (release [_]
    (release data-desc)
    (release gamma-desc))
  Object
  (hashCode [_]
    (-> (hash :batch-norm) (hash-combine gamma-desc)))
  (equals [_ other]
    (and (instance? CUDnnBatchNormalizationBlueprint other)
         (equal-desc? data-desc (.data-desc ^CUDnnBatchNormalizationBlueprint other))
         (equal-desc? gamma-desc (.gamma-desc ^CUDnnBatchNormalizationBlueprint other))))
  (toString [this]
    (pr-str {:topology :batch-norm
             :shape (shape this)}))
  Info
  (info [this info-type]
    (case info-type
      :bias gamma-desc
      :inference {:src data-desc
                  :weights gamma-desc
                  :dst data-desc}
      :training {:src data-desc
                 :weights gamma-desc
                 :dst data-desc}
      nil))
  (info [this]
    {:bias gamma-desc
     :inference {:src data-desc
                 :weights gamma-desc
                 :dst data-desc}
     :training {:src data-desc
                :weights gamma-desc
                :dst data-desc}})
  DiamondFactoryProvider
  (diamond-factory [_]
    fact)
  DescriptorProvider
  (inf-desc [_]
    data-desc)
  (train-desc [_]
    data-desc)
  (diff-desc [_]
    data-desc)
  TensorDescriptor
  (shape [this]
    (shape data-desc))
  (data-type [this]
    (data-type data-desc))
  (layout [this]
    (layout data-desc))
  IFn
  (invoke [this src-tz]
    (let-release [src-conn (connector src-tz data-desc)
                  gamma-tz (cudnn-tensor fact (view gamma-desc))
                  beta-tz (cudnn-tensor fact (view gamma-desc))
                  mean-tz (cudnn-tensor fact (view gamma-desc))
                  var-tz (cudnn-tensor fact (view gamma-desc))
                  dst-tz (cudnn-tensor fact (view data-desc))]
      (->CUDnnBatchNormalizationInference fact (handle fact) this
                                          (cast-prim (data-accessor gamma-tz) 1.0)
                                          (cast-prim (data-accessor gamma-tz) 0.0)
                                          gamma-desc src-conn gamma-tz beta-tz mean-tz var-tz)))
  (invoke [this src-tz diff-src-tz prop-diff? _]
    (let [da (data-accessor src-tz)]
      (let-release [src-conn (connector src-tz data-desc)
                    gamma-tz (cudnn-tensor fact (view gamma-desc))
                    beta-tz (cudnn-tensor fact (view gamma-desc))
                    mean-tz (cudnn-tensor fact (view gamma-desc))
                    var-tz (cudnn-tensor fact (view gamma-desc))
                    dst-tz (cudnn-tensor fact (view data-desc) (batch-index src-tz))
                    saved-mean-tz (create-tensor fact (view gamma-desc) true)
                    saved-inv-var-tz (create-tensor fact (view gamma-desc) true)
                    diff-gamma-tz (create-tensor fact (view gamma-desc) true)
                    diff-beta-tz (create-tensor fact (view gamma-desc) true)
                    diff-src-conn (if prop-diff?
                                    (connector data-desc diff-src-tz)
                                    (cudnn-tensor fact data-desc (batch-index diff-src-tz)))]
        (->CUDnnBatchNormalizationTraining fact (handle fact) this da (atom -1) un-bessel
                                           (cast-prim da 1.0) (cast-prim da 0.0)
                                           gamma-desc src-conn gamma-tz beta-tz dst-tz
                                           mean-tz var-tz saved-mean-tz saved-inv-var-tz
                                           diff-gamma-tz diff-beta-tz diff-src-conn))))
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(defmethod print-method CUDnnBatchNormalizationBlueprint
  [bp ^java.io.Writer w]
  (.write w (str bp)))

(defn cudnn-batch-norm-op-blueprint [fact data-desc]
  (let-release [data-desc (desc data-desc)
                gamma-desc (batch-norm-descriptor data-desc :spatial)]
    (let [data-shape (shape data-desc)
          n (long (apply * (get data-shape 0) (drop 2 data-shape)))
          un-bessel (/ (dec n) n)]
      (->CUDnnBatchNormalizationBlueprint fact data-desc gamma-desc un-bessel))))

(defn cudnn-batch-norm-layer-blueprint [fact data-desc activ alpha beta]
  (let-release [batch-norm-bluep (cudnn-batch-norm-op-blueprint fact (view data-desc))
                activ-bluep (cudnn-activ-blueprint fact (view data-desc) activ alpha)]
    (->DirectedLayerBlueprint fact :batch-norm batch-norm-bluep activ-bluep)))

(defmethod transfer! [CUDnnBatchNormalizationInference Object]
  [source destination]
  (doall (map transfer! (parameters source) (parameters destination)))
  destination)

(defmethod transfer! [CUDnnBatchNormalizationTraining Object]
  [source destination]
  (doall (map transfer! (parameters source) (parameters destination)))
  destination)

;; ================================ Branch ======================================

(deftype CUDnnBranchInference [fact bluep branch? data-tz fwd-trans]
  Releaseable
  (release [_]
    (doseq [ft fwd-trans] (release ft))
    true)
  Object
  (hashCode [_]
    (reduce #(hash-combine %1 (shape (output %2)))
            (hash-combine (hash :branch) (shape data-tz))
            fwd-trans))
  (equals [_ other]
    (and (instance? CUDnnBranchInference other)
         (= data-tz (.data-tz ^CUDnnBranchInference other))
         (= (map output fwd-trans) (map output (.fwd-trans ^CUDnnBranchInference other)))))
  (toString [this]
    (str bluep))
  Info
  (info [this]
    {:src (fmap info (input this))
     :dst (fmap info (output this))})
  (info [this info-type]
    (case info-type
      :src (fmap info (input this))
      :dst (fmap info (output this))
      nil))
  DiamondFactoryProvider
  (diamond-factory [_]
    fact)
  Transfer
  (input [_]
    (if branch? data-tz (mapv input fwd-trans)))
  (output [_]
    (if branch? (mapv output fwd-trans) data-tz))
  ParametersSeq
  (parameters [_]
    [])
  Initializable
  (init [this _]
    this)
  IFn
  (invoke [this]
    (doseq [ft fwd-trans] (ft))
    (output this))
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(defmethod print-method CUDnnBranchInference
  [layer ^java.io.Writer w]
  (.write w (format "#Branch[src:%s, dst:%s]" (input layer) (output layer))))

(deftype CUDnnBranchTraining [fact bluep branch? data-tz diff-data-tz
                              fwd-trans bwd-trans prop-diff?]
  Releaseable
  (release [_]
    (release data-tz)
    (release diff-data-tz)
    (doseq [ft fwd-trans] (release ft))
    (doseq [bt bwd-trans] (release bt))
    true)
  Object
  (hashCode [_]
    (reduce #(hash-combine %1 (shape (output %2)))
            (hash-combine (hash :branch) (shape data-tz))
            fwd-trans))
  (equals [_ other]
    (and (instance? CUDnnBranchTraining other)
         (= data-tz (.data-tz ^CUDnnBranchTraining other))
         (= (map output fwd-trans) (map output (.fwd-trans ^CUDnnBranchTraining other)))))
  (toString [this]
    (str bluep))
  Info
  (info [this]
    {:src (fmap info (input this))
     :dst (fmap info (output this))})
  (info [this info-type]
    (case info-type
      :src (fmap info (input this))
      :dst (fmap info (output this))
      nil))
  DiamondFactoryProvider
  (diamond-factory [_]
    fact)
  Transfer
  (input [_]
    (if branch? data-tz (mapv input fwd-trans)))
  (output [_]
    (if branch? (mapv output fwd-trans) data-tz))
  DiffTransfer
  (diff-input [this]
    (if branch? (mapv input bwd-trans) data-tz))
  (diff-output [this]
    (if branch? diff-data-tz (mapv output bwd-trans)))
  ParametersSeq
  (parameters [_]
    [])
  Initializable
  (init [this _]
    this)
  Backprop
  (forward [this]
    this)
  (forward [this _]
    (doseq [ft fwd-trans] (ft))
    this)
  (backward [this]
    this)
  (backward [this _]
    (when prop-diff?
      (doseq [bt bwd-trans] (bt)))
    this)
  IFn
  (invoke [this]
    (doseq [ft fwd-trans] (ft))
    (output this))
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(defmethod print-method CUDnnBranchTraining
  [layer ^java.io.Writer w]
  (.write w (format "#Branch[src:%s, dst:%s]" (input layer) (output layer))))

(deftype CUDnnBranchBlueprint [fact cudnn-hdl src-desc dst-descs sub-descs sub-offsets branch-dim dst-dims]
  Releaseable
  (release [_]
    (doseq [dd dst-descs] (release dd))
    (doseq [sd sub-descs] (release sd))
    (release src-desc))
  Object
  (hashCode [_]
    (hash-combine (reduce hash-combine (hash :branch) dst-descs) src-desc))
  (equals [_ other]
    (and (instance? CUDnnBranchBlueprint other)
         (every? identity (map equal-desc? dst-descs (.dst-descs ^CUDnnBranchBlueprint other)))
         (equal-desc? src-desc (.src-desc ^CUDnnBranchBlueprint other))))
  (toString [this]
    (pr-str {:shape (shape this)
             :topology :branch}))
  DiamondFactoryProvider
  (diamond-factory [_]
    fact)
  DescriptorProvider
  (inf-desc [this]
    dst-descs)
  (train-desc [_]
    dst-descs)
  (diff-desc [_]
    dst-descs)
  TensorDescriptor
  (shape [this]
    (fmap shape dst-descs))
  (data-type [this]
    (fmap data-type dst-descs))
  (layout [this]
    (fmap layout dst-descs))
  IFn
  (invoke [this prev-layer]
    (let [src-tz (output prev-layer)]
      (let-release [dst-tzs (fmap (partial cudnn-tensor fact) dst-descs)
                    sub-tzs (mapv #(cudnn-tensor fact false (buffer src-tz) (+ (offset src-tz) (long %1)) %2)
                                  sub-offsets sub-descs)
                    fwd-trans (mapv (partial cudnn-transformer cudnn-hdl) sub-tzs dst-tzs)]
        (->CUDnnBranchInference fact this true src-tz fwd-trans))))
  (invoke [this prev-layer prop-diff? _]
    (let [src-tz (output prev-layer)
          diff-src-tz (diff-input prev-layer)
          diff-src-sub-offsets (mapv (partial * (get (strides diff-src-tz) branch-dim))
                                     (concat-offsets branch-dim dst-dims))]
      (let-release [dst-tzs (fmap (partial cudnn-tensor fact) dst-descs)
                    src-sub-tzs (mapv #(cudnn-tensor fact false (buffer src-tz) %1 %2)
                                      sub-offsets sub-descs)
                    diff-src-sub-tzs (mapv #(cudnn-tensor fact false (buffer diff-src-tz) %1 %2)
                                           diff-src-sub-offsets sub-descs)
                    fwd-trans (mapv (partial cudnn-transformer cudnn-hdl) src-sub-tzs dst-tzs)
                    bwd-trans (mapv (partial cudnn-transformer cudnn-hdl) dst-tzs diff-src-sub-tzs)]
        (->CUDnnBranchTraining fact this true src-tz diff-src-tz fwd-trans bwd-trans prop-diff?))))
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(defmethod print-method CUDnnBranchBlueprint
  [bp ^java.io.Writer w]
  (.write w (str bp)))

(defn cudnn-branch-blueprint [fact src-desc branch-dim dst-descs]
  (let-release [src-desc (desc src-desc)
                dtype (data-type src-desc)
                strd (strides src-desc)
                dst-dims (map shape dst-descs)
                dst-descs (mapv desc dst-descs)
                sub-descs (mapv #(cudnn-tensor-desc %1 dtype strd) dst-dims)
                sub-offsets (mapv (partial * (get (strides src-desc) branch-dim))
                                  (concat-offsets branch-dim dst-dims))]
    (->CUDnnBranchBlueprint fact (handle fact) src-desc dst-descs sub-descs sub-offsets branch-dim dst-dims)))

(deftype CUDnnConcatBlueprint [fact cudnn-hdl src-descs dst-desc sub-descs sub-offsets]
  Releaseable
  (release [_]
    (doseq [src-desc src-descs] (release src-desc))
    (doseq [sd sub-descs] (release sd))
    (release dst-desc))
  Object
  (hashCode [_]
    (hash-combine (reduce hash-combine (hash :branch) src-descs) dst-desc))
  (equals [_ other]
    (and (instance? CUDnnConcatBlueprint other)
         (every? identity (map equal-desc? src-descs (.src-descs ^CUDnnConcatBlueprint other)))
         (equal-desc? dst-desc (.dst-desc ^CUDnnConcatBlueprint other))))
  (toString [this]
    (pr-str {:shape (shape this)
             :topology :concat}))
  DiamondFactoryProvider
  (diamond-factory [_]
    fact)
  DescriptorProvider
  (inf-desc [this]
    dst-desc)
  (train-desc [_]
    dst-desc)
  (diff-desc [_]
    dst-desc)
  TensorDescriptor
  (shape [this]
    (shape dst-desc))
  (data-type [this]
    (data-type dst-desc))
  (layout [this]
    (layout dst-desc))
  IFn
  (invoke [this prev-layer]
    (let-release [src-tzs (fmap (comp view output) prev-layer)
                  dst-tz (cudnn-tensor fact dst-desc)
                  sub-tzs (mapv #(cudnn-tensor fact false (buffer dst-tz) (+ (offset dst-tz) (long %1)) %2)
                               sub-offsets sub-descs)
                  fwd-trans (mapv (partial cudnn-transformer cudnn-hdl) src-tzs sub-tzs)]
      (->CUDnnBranchInference fact this false dst-tz fwd-trans)))
  (invoke [this prev-layer prop-diff? _]
    (let [src-tzs (fmap (comp view output) prev-layer)
          diff-src-tzs (fmap (comp view diff-input) prev-layer)]
      (let-release [dst-tz (cudnn-tensor fact dst-desc)
                    sub-tzs (mapv #(cudnn-tensor fact false (buffer dst-tz) (+ (offset dst-tz) (long %1)) %2)
                                  sub-offsets sub-descs)
                    fwd-trans (mapv (partial cudnn-transformer cudnn-hdl) src-tzs sub-tzs)
                    bwd-trans (mapv (partial cudnn-transformer cudnn-hdl) sub-tzs diff-src-tzs)]
        (->CUDnnBranchTraining fact this false dst-tz diff-src-tzs fwd-trans bwd-trans prop-diff?))))
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(defmethod print-method CUDnnConcatBlueprint
  [bp ^java.io.Writer w]
  (.write w (str bp)))

(defn cudnn-concat-blueprint [fact src-descs conc-dim dst-type]
  (let-release [src-dims (mapv shape src-descs)
                src-descs (mapv desc src-descs)
                dst-type (or dst-type (tz/data-type (first src-descs)))
                dst-shape (concat-dst-shape conc-dim src-dims)
                dst-strd (default-strides dst-shape)
                dst-desc (cudnn-tensor-desc dst-shape dst-type dst-strd)
                sub-descs (mapv #(cudnn-tensor-desc %1 dst-type dst-strd) src-dims)
                sub-offsets (mapv (partial * (get (strides dst-desc) conc-dim))
                                  (concat-offsets conc-dim src-dims))]
    (->CUDnnConcatBlueprint fact (handle fact) src-descs dst-desc sub-descs sub-offsets)))

;; ============================ Split ====================================================

(deftype CUDnnSplitInference [fact cudnn-hdl bluep n src-tz]
  Releaseable
  (release [_]
    (release src-tz))
  Object
  (hashCode [_]
    (-> (hash :split) (hash-combine n) (hash-combine src-tz)))
  (equals [_ other]
    (and (instance? CUDnnSplitInference other)
         (= n (.n ^CUDnnSplitInference other))))
  (toString [this]
    (str bluep))
  Info
  (info [this]
    {:src (info src-tz)
     :n n})
  (info [this info-type]
    (case info-type
      :src (info src-tz)
      :n n
      nil))
  DiamondFactoryProvider
  (diamond-factory [_]
    fact)
  Transfer
  (input [_]
    src-tz)
  (output [_]
    (vec (repeat n src-tz)))
  ParametersSeq
  (parameters [_]
    [])
  Initializable
  (init [this _]
    this)
  IFn
  (invoke [this]
    (output this))
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(defmethod print-method CUDnnSplitInference
  [^CUDnnSplitInference layer ^java.io.Writer w]
  (.write w (format "#Split[n:%d, src:%s]" (.n layer) (input layer))))

(deftype CUDnnSplitTraining [fact cudnn-hdl bluep ^long n src-tz diff-tzs prop-diff?]
  Releaseable
  (release [_]
    (doseq [dt diff-tzs] (release dt))
    true)
  Object
  (hashCode [_]
    (-> (hash :split) (hash-combine n) (hash-combine src-tz)))
  (equals [_ other]
    (and (instance? CUDnnSplitTraining other)
         (= n (.n ^CUDnnSplitTraining other))))
  (toString [this]
    (str bluep))
  Info
  (info [this]
    {:src (info src-tz)
     :n n})
  (info [this info-type]
    (case info-type
      :src (info src-tz)
      :n n
      nil))
  DiamondFactoryProvider
  (diamond-factory [_]
    fact)
  Transfer
  (input [_]
    src-tz)
  (output [_]
    (vec (repeat n src-tz)))
  DiffTransfer
  (diff-input [_]
    diff-tzs)
  (diff-output [_]
    src-tz)
  ParametersSeq
  (parameters [_]
    [])
  Initializable
  (init [this _]
    this)
  Backprop
  (forward [this]
    this)
  (forward [this _]
    this)
  (backward [this]
    this)
  (backward [this _]
    (when prop-diff?
      (entry! src-tz 0.0)
      (doseq [diff-tz diff-tzs]
        (axpy! (/ 1.0 n) diff-tz src-tz)))
    this)
  IFn
  (invoke [this]
    (output this))
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(defmethod print-method CUDnnSplitTraining
  [^CUDnnSplitTraining layer ^java.io.Writer w]
  (.write w (format "#Split[n:%d, src:%s]" (.n layer) (input layer))))

(deftype CUDnnSplitBlueprint [fact n src-desc]
  Releaseable
  (release [_]
    (release src-desc))
  Object
  (hashCode [_]
    (-> (hash :split)
        (hash-combine n)
        (hash-combine src-desc)))
  (equals [_ other]
    (and (instance? CUDnnSplitBlueprint other)
         (= n (.n ^CUDnnSplitBlueprint other))
         (equal-desc? src-desc (.src-desc ^CUDnnSplitBlueprint other))))
  (toString [this]
    (pr-str {:shape (shape this)
             :topology :split}))
  DiamondFactoryProvider
  (diamond-factory [_]
    fact)
  DescriptorProvider
  (inf-desc [this]
    (train-desc this))
  (train-desc [_]
    (vec (repeat n src-desc)))
  (diff-desc [_]
    "TODO")
  TensorDescriptor
  (shape [_]
    (vec (repeat n (shape src-desc))))
  (data-type [_]
    (vec (repeat n (data-type src-desc))))
  (layout [_]
    (vec (repeat n (layout src-desc))))
  IFn
  (invoke [this prev-layer]
    (->CUDnnSplitInference fact (handle fact) this n (view (output prev-layer))))
  (invoke [this prev-layer prop-diff? _]
    (let-release [src-tz (view (output prev-layer))
                  diff-tzs (mapv (partial cudnn-tensor fact) (repeat n src-desc))]
      (->CUDnnSplitTraining fact (handle fact) this n src-tz diff-tzs prop-diff?)))
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(defmethod print-method CUDnnSplitBlueprint
  [^CUDnnSplitBlueprint layer ^java.io.Writer w]
  (.write w (format "#Split[n:%d, src:%s]" (.n layer) (input layer))))

(defn cudnn-split-blueprint [fact src-desc ^long n]
  (->CUDnnSplitBlueprint fact n (view (desc src-desc))))

(defmethod transfer! [CUDnnSplitInference Object]
  [source destination]
  destination)

(defmethod transfer! [CUDnnSplitTraining Object]
  [source destination]
  destination)

;; ================================ Sum ======================================

(deftype CUDnnSum [fact cudnn-hdl bluep src-tzs prop-diff?]
  Releaseable
  (release [_]
    true)
  Object
  (hashCode [_]
    (reduce #(hash-combine %1 (shape %2)) (hash :sum) src-tzs))
  (equals [_ other]
    (and (instance? CUDnnSum other) (= src-tzs (.src-tzs ^CUDnnSum other))))
  (toString [this]
    (str bluep))
  Info
  (info [this]
    {:src (map info src-tzs)})
  (info [this info-type]
    (case info-type
      :src (map info src-tzs)
      nil))
  DiamondFactoryProvider
  (diamond-factory [_]
    fact)
  Transfer
  (input [_]
    src-tzs)
  (output [_]
    (first src-tzs))
  DiffTransfer
  (diff-input [_]
    (first src-tzs))
  (diff-output [_]
    src-tzs)
  ParametersSeq
  (parameters [_]
    [])
  Initializable
  (init [this _]
    this)
  Backprop
  (forward [this]
    this)
  (forward [this _]
    (let [src0 (get src-tzs 0)]
      (doseq [src-tz (rest src-tzs)]
        (axpy! 1.0 src-tz src0)))
    this)
  (backward [this]
    this)
  (backward [this _]
    (when prop-diff?
      (let [src0 (scal! (/ 1.0 (count src-tzs)) (get src-tzs 0))]
        (doseq [src-tz (rest src-tzs)]
          (copy! src0 src-tz))))
    this)
  IFn
  (invoke [this]
    (let [src0 (get src-tzs 0)]
      (doseq [src-tz (rest src-tzs)]
        (axpy! 1.0 src-tz src0))
      src0))
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(defmethod print-method CUDnnSum
  [layer ^java.io.Writer w]
  (.write w (format "#Sum[srcs:%s]" (input layer))))

(deftype CUDnnSumBlueprint [fact src-descs]
  Releaseable
  (release [_]
    (doseq [sd src-descs]
      (release sd))
    true)
  Object
  (hashCode [_]
    (reduce #(hash-combine %1 (shape %2)) (hash :sum) src-descs))
  (equals [_ other]
    (and (instance? CUDnnSumBlueprint other)
         (every? identity (map equal-desc? src-descs (.src-descs ^CUDnnSumBlueprint other)))))
  (toString [this]
    (pr-str {:shape (shape this)
             :topology :sum}))
  DiamondFactoryProvider
  (diamond-factory [_]
    fact)
  DescriptorProvider
  (inf-desc [this]
    (get src-descs 0))
  (train-desc [_]
    (get src-descs 0))
  (diff-desc [_]
    "TODO")
  TensorDescriptor
  (shape [_]
    (shape (get src-descs 0)))
  (data-type [_]
    (data-type (get src-descs 0)))
  (layout [_]
    (layout (get src-descs 0)))
  IFn
  (invoke [this prev-layer]
    (this prev-layer false nil))
  (invoke [this prev-layer prop-diff? _]
    (->CUDnnSum fact (handle fact) this (fmap output prev-layer) prop-diff?))
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(defmethod print-method CUDnnSum
  [bp ^java.io.Writer w]
  (.write w (str bp)))

(defn cudnn-sum-blueprint [fact src-descs]
  (->CUDnnSumBlueprint fact (mapv (comp view desc) src-descs)))

(defmethod transfer! [CUDnnSum Object]
  [source destination]
  destination)
