;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.diamond.internal.dnnl.directed
  (:require [uncomplicate.commons
             [core :refer [Releaseable release let-release with-release Info info view]]
             [utils :refer [dragan-says-ex]]]
            [uncomplicate.neanderthal
             [core :refer [axpy! axpby! zero dim transfer! scal! copy! view-vctr]]
             [real :refer [nrm2 asum]]
             [block :refer [buffer]]
             [math :refer [sqrt pow]]
             [vect-math :refer [sqr! linear-frac! sqrt! mul!]]]
            [uncomplicate.neanderthal.internal.api :refer [flow]]
            [uncomplicate.diamond.tensor :as tz
             :refer [Transfer input output connector shape layout TensorDescriptor
                     shape view-tz revert]]
            [uncomplicate.diamond.internal
             [protocols
              :refer [Parameters bias weights ParametersSeq parameters
                      DescriptorProvider DiamondFactoryProvider DiffParameters
                      diff-weights Backprop forward backward
                      DiffTransfer diff-output diff-input diff-z
                      LinearBackprop backward-diff inf-desc train-desc]]
             [utils :refer [transfer-weights-bias! default-strides]]]
            [uncomplicate.diamond.internal.dnnl
             [protocols :refer :all]
             [core :refer :all :as dnnl]
             [tensor :refer [dnnl-tensor dnnl-transformer dnnl-args]]]
            [uncomplicate.diamond.internal.neanderthal.directed
             :refer [->DirectedLayerBlueprint ->GaussianDropoutBlueprint]])
  (:import [clojure.lang IFn AFn]
           [uncomplicate.diamond.internal.neanderthal.directed
            DirectedLayerBlueprint GaussianDropoutBlueprint]))

(defn dnnl-contiguous-desc [md]
  (let [shape (dims md)]
    (if (= (size md) (apply * Float/BYTES shape))
      md ;;TODO this should be somehow copied, perhaps...
      (memory-desc (dims md) :float (default-strides shape)))))

;; ================================ Sum ======================================

(deftype DnnlSum [strm dst-tz sum-prim sum-args]
  Releaseable
  (release [_]
    (release sum-prim)
    (release dst-tz))
  IFn
  (invoke [this]
    (execute! strm sum-prim sum-args)
    dst-tz)
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(deftype DnnlSumBlueprint [strm sum-pd]
  Releaseable
  (release [_]
    (release sum-pd))
  IFn
  (invoke [_ src-and-dst]
    (let-release [sum-prim (primitive sum-pd)]
      (->DnnlSum strm src-and-dst sum-prim (args (buffer src-and-dst)))))
  (invoke [_ src-tz dst-tz]
    (let-release [sum-prim (primitive sum-pd)]
      (->DnnlSum strm dst-tz sum-prim (dnnl-args args dst-tz dst-tz src-tz))))
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(defn dnnl-sum-blueprint
  ([eng strm scale dst]
   (->DnnlSumBlueprint strm (sum! eng scale dst)))
  ([eng strm scale-src src scale-dst dst]
   (->DnnlSumBlueprint strm (sum! eng dst scale-dst dst scale-src src))))

;; ================================ Activation =============================================

(deftype DnnlActivationInference [fact strm bluep a-tz
                                  eltwise-fwd-prim eltwise-fwd-args]
  Releaseable
  (release [_]
    (release a-tz)
    (release eltwise-fwd-prim))
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
    (execute! strm eltwise-fwd-prim eltwise-fwd-args)
    a-tz)
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(deftype DnnlActivationTraining [fact strm bluep z-tz a-tz
                                 eltwise-fwd-prim eltwise-fwd-args
                                 eltwise-bwd-prim eltwise-bwd-args]
  Releaseable
  (release [_]
    (release z-tz)
    (release a-tz)
    (release eltwise-fwd-prim)
    (release eltwise-bwd-prim))
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
    (execute! strm eltwise-fwd-prim eltwise-fwd-args)
    a-tz)
  (applyTo [this xs]
    (AFn/applyToHelper this xs))
  Backprop
  (forward [this]
    (execute! strm eltwise-fwd-prim eltwise-fwd-args)
    this)
  (backward [this]
    (execute! strm eltwise-bwd-prim eltwise-bwd-args)
    this))

(deftype DnnlActivationBlueprint [fact activ eltw-infer-pd eltw-train-pd eltw-bwd-pd]
  Releaseable
  (release [_]
    (release eltw-infer-pd)
    (release eltw-train-pd)
    (release eltw-bwd-pd))
  Info
  (info [this]
    {:activation activ})
  (info [this info-type]
    (case info-type
      :activation activ
      nil))
  DescriptorProvider
  (inf-desc [_]
    (dst-md eltw-infer-pd))
  (train-desc [_]
    (dst-md eltw-train-pd))
  TensorDescriptor
  (shape [this]
    (shape (train-desc this)))
  (data-type [this]
    (data-type (train-desc this)))
  (layout [this]
    (layout (train-desc this)))
  IFn
  (invoke [this src-tz]
    (let-release [eltw-fwd-prim (primitive eltw-infer-pd)
                  eltw-fwd-args (fwd-args (buffer src-tz))]
      (->DnnlActivationInference fact (flow fact) this src-tz
                                 eltw-fwd-prim eltw-fwd-args)))
  (invoke [this src-tz dst-tz]
    (let-release [eltw-fwd-prim (primitive eltw-train-pd)
                  eltw-fwd-args (fwd-args (buffer src-tz) (buffer dst-tz))
                  eltw-bwd-prim (primitive eltw-bwd-pd)
                  eltw-bwd-args (dnnl-args eltwise-bwd-args src-tz dst-tz src-tz)]
      (->DnnlActivationTraining fact (flow fact) this src-tz dst-tz
                                eltw-fwd-prim eltw-fwd-args
                                eltw-bwd-prim eltw-bwd-args)))
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(defn dnnl-activation-blueprint
  ([fact eng inf-src-desc train-src-desc diff-desc activ alpha beta]
   (let [inf-src-desc (desc inf-src-desc)
         train-src-desc (desc train-src-desc)
         diff-desc (desc diff-desc)]
     (with-release [eltw-infer-desc (eltwise-fwd-desc :inference activ inf-src-desc alpha beta)
                    eltw-train-desc (eltwise-fwd-desc :training activ train-src-desc alpha beta)
                    eltw-bwd-desc (eltwise-bwd-desc activ diff-desc train-src-desc alpha beta)]
       (let-release [eltw-infer-pd (primitive-desc eng eltw-infer-desc)
                     eltw-train-pd (primitive-desc eng eltw-train-desc)
                     eltw-bwd-pd (primitive-desc eng eltw-bwd-desc eltw-train-pd)]
         (->DnnlActivationBlueprint fact activ eltw-infer-pd eltw-train-pd eltw-bwd-pd)))))
  ([fact eng data-desc activ alpha beta]
   (dnnl-activation-blueprint fact eng data-desc data-desc data-desc activ alpha beta)))

;; ================================ Softmax =============================================

(deftype DnnlSoftmaxTraining [fact strm bluep z-tz da-tz
                              softmax-fwd-prim softmax-fwd-args
                              softmax-bwd-prim softmax-bwd-args]
  Releaseable
  (release [_]
    (release z-tz)
    (release da-tz)
    (release softmax-fwd-prim)
    (release softmax-bwd-prim))
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
    (execute! strm softmax-fwd-prim softmax-fwd-args)
    z-tz)
  (applyTo [this xs]
    (AFn/applyToHelper this xs))
  Backprop
  (forward [this]
    (execute! strm softmax-fwd-prim softmax-fwd-args)
    this)
  (backward [this]
    (execute! strm softmax-bwd-prim softmax-bwd-args)
    this))

(deftype DnnlSoftmaxBlueprint [fact softmax-infer-pd softmax-train-pd softmax-bwd-pd]
  Releaseable
  (release [_]
    (release softmax-infer-pd)
    (release softmax-train-pd)
    (release softmax-bwd-pd))
  Info
  (info [this]
    {:activation :softmax})
  (info [this info-type]
    (case info-type
      :activation :softmax
      nil))
  DescriptorProvider
  (inf-desc [_]
    (dst-md softmax-infer-pd))
  (train-desc [_]
    (dst-md softmax-train-pd))
  TensorDescriptor
  (shape [this]
    (shape (train-desc this)))
  (data-type [this]
    (data-type (train-desc this)))
  (layout [this]
    (layout (train-desc this)))
  IFn
  (invoke [this src-tz]
    (let-release [softmax-fwd-prim (primitive softmax-infer-pd)
                  softmax-fwd-args (fwd-args (buffer src-tz))]
      (->DnnlActivationInference fact (flow fact) this src-tz
                                 softmax-fwd-prim softmax-fwd-args)))
  (invoke [this src-tz diff-tz]
    (let-release [softmax-fwd-prim (primitive softmax-train-pd)
                  softmax-fwd-args (fwd-args (buffer src-tz))
                  softmax-bwd-prim (primitive softmax-bwd-pd)
                  softmax-bwd-args (dnnl-args softmax-bwd-args src-tz diff-tz src-tz)]
      (->DnnlSoftmaxTraining fact (flow fact) this src-tz diff-tz
                             softmax-fwd-prim softmax-fwd-args
                             softmax-bwd-prim softmax-bwd-args)))
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(defn dnnl-softmax-blueprint [fact eng inf-src-desc train-src-desc diff-desc]
  (let [inf-src-desc (desc inf-src-desc)
        train-src-desc (desc train-src-desc)
        diff-desc (desc diff-desc)]
    (with-release [softmax-infer-desc (softmax-fwd-desc :inference inf-src-desc 1);;TODO currently DNNL is optimized for 1
                   softmax-train-desc (softmax-fwd-desc :training train-src-desc 1)
                   softmax-bwd-desc (softmax-bwd-desc diff-desc train-src-desc 1)]
      (let-release [softmax-infer-pd (primitive-desc eng softmax-infer-desc)
                    softmax-train-pd (primitive-desc eng softmax-train-desc)
                    softmax-bwd-pd (primitive-desc eng softmax-bwd-desc softmax-train-pd)]
        (->DnnlSoftmaxBlueprint fact softmax-infer-pd softmax-train-pd softmax-bwd-pd)))))

(defn dnnl-activ-blueprint
  ([fact eng inf-src-desc train-src-desc diff-desc activ alpha beta]
   (if (= :softmax activ)
     (dnnl-softmax-blueprint fact eng inf-src-desc train-src-desc diff-desc)
     (dnnl-activation-blueprint fact eng inf-src-desc train-src-desc diff-desc activ alpha beta)))
  ([fact eng data-desc activ alpha beta]
   (dnnl-activ-blueprint fact eng data-desc data-desc data-desc activ alpha beta)))

;; ================================ Inner Product & Convolution ====================================

(deftype DnnlProductInference [fact strm bluep
                               src-conn bias-tz weights-tz dst-tz
                               fwd-prim fwd-args]
  Releaseable
  (release [_]
    (release src-conn)
    (release bias-tz)
    (release weights-tz)
    (release dst-tz)
    (release fwd-prim))
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
    (execute! strm fwd-prim fwd-args)
    dst-tz)
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(deftype DnnlProductTraining [fact strm bluep
                              bias-tz weights-tz dst-tz
                              diff-weights-tz post-diff-weights-tz diff-bias-tz
                              src-conn weights-conn
                              fwd-prim fwd-args
                              bwd-src-conn diff-dst-conn diff-weights-conn
                              bwd-weights-prim bwd-weights-args
                              diff-dst-data-conn weights-data-conn diff-src-conn
                              bwd-data-prim bwd-data-args]
  Releaseable
  (release [_]
    (release bias-tz)
    (release weights-tz)
    (release dst-tz)
    (release diff-weights-tz)
    (release post-diff-weights-tz)
    (release diff-bias-tz)
    (release src-conn)
    (release weights-conn)
    (release fwd-prim)
    (release bwd-src-conn)
    (release diff-dst-conn)
    (release diff-weights-conn)
    (release bwd-weights-prim)
    (release bwd-weights-args)
    (release diff-dst-data-conn)
    (release weights-data-conn)
    (release diff-src-conn)
    (release bwd-data-prim))
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
    (input src-conn))
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
    post-diff-weights-tz)
  IFn
  (invoke [_]
    (src-conn)
    (weights-conn)
    (execute! strm fwd-prim fwd-args)
    dst-tz)
  (applyTo [this xs]
    (AFn/applyToHelper this xs))
  Backprop
  (forward [this]
    (src-conn)
    (weights-conn)
    (execute! strm fwd-prim fwd-args)
    this)
  (backward [this]
    (backward-diff this 1.0 0.0 1.0 0.0))
  LinearBackprop
  (backward-diff [this scal-diff-w scal-g scal-diff-b scal-b]
    (bwd-src-conn)
    (diff-dst-conn)
    (execute! strm bwd-weights-prim bwd-weights-args)
    (diff-weights-conn)
    (if (= 0.0 scal-g)
      (when-not (= 1.0 scal-diff-w) (scal! scal-diff-w diff-weights-tz))
      (axpby! scal-diff-w diff-weights-tz scal-g post-diff-weights-tz));;TODO check whether post-diff-weights-tz and diff-weights-tz are consistently updated
    (axpby! scal-diff-b diff-bias-tz scal-b bias-tz)
    (when bwd-data-prim
      (diff-dst-data-conn)
      (weights-data-conn)
      (execute! strm bwd-data-prim bwd-data-args)
      (diff-src-conn))
    this))

(deftype DnnlProductBlueprint [fact weights-desc bias-desc infer-pd
                               train-pd bwd-weights-pd bwd-data-pd]
  Object
  (hashCode [_]
    (-> (hash weights-desc) (hash-combine bias-desc)))
  (equals [_ other]
    (and (instance? DnnlProductBlueprint other)
         (= bias-desc (.bias-desc ^DnnlProductBlueprint other))
         (equal-desc? (src-md infer-pd) (src-md (.infer-pd ^DnnlProductBlueprint other)))
         (equal-desc? (weights-md infer-pd) (weights-md (.infer-pd ^DnnlProductBlueprint other)))
         (equal-desc? (dst-md infer-pd) (dst-md (.infer-pd ^DnnlProductBlueprint other)))
         (equal-desc? (src-md train-pd) (src-md (.train-pd ^DnnlProductBlueprint other)))
         (equal-desc? (weights-md train-pd) (weights-md (.train-pd ^DnnlProductBlueprint other)))
         (equal-desc? (dst-md train-pd) (dst-md (.train-pd ^DnnlProductBlueprint other)))))
  (toString [this]
    (pr-str {:src (src-md infer-pd)
             :weights (weights-md infer-pd)
             :dst (dst-md infer-pd)}))
  Releaseable
  (release [_]
    (release weights-desc)
    (release bias-desc)
    (release infer-pd)
    (release train-pd)
    (release bwd-weights-pd)
    (release bwd-data-pd))
  Info
  (info [this info-type]
    (case info-type
      :bias bias-desc
      :inference {:src (src-md infer-pd)
                  :weights (weights-md infer-pd)
                  :dst (dst-md infer-pd)}
      :training {:src (src-md train-pd)
                 :weights (weights-md train-pd)
                 :dst (dst-md train-pd)}
      nil))
  (info [this]
    {:bias bias-desc
     :inference {:src (src-md infer-pd)
                 :weights (weights-md infer-pd)
                 :dst (dst-md infer-pd)}
     :training {:src (src-md train-pd)
                :weights (weights-md train-pd)
                :dst (dst-md train-pd)}})
  DescriptorProvider
  (inf-desc [_]
    (dst-md infer-pd))
  (train-desc [_]
    (dst-md train-pd))
  IFn
  (invoke [this src-tz]
    (let-release [src-conn (connector src-tz (src-md infer-pd))
                  bias-tz (dnnl-tensor fact bias-desc)
                  weights-tz (dnnl-tensor fact (weights-md infer-pd))
                  dst-tz (dnnl-tensor fact (dst-md infer-pd))
                  fwd-prim (primitive infer-pd)
                  fwd-args (dnnl-args fwd-args (output src-conn)
                                      weights-tz bias-tz dst-tz)]
      (->DnnlProductInference fact (flow fact) this
                              src-conn bias-tz weights-tz dst-tz
                              fwd-prim fwd-args)))
  (invoke [this src-tz dst-tz prop-diff? post-process-diff?]
    (let-release [src-conn (connector src-tz (src-md train-pd))
                  bias-tz (dnnl-tensor fact bias-desc)
                  weights-tz (dnnl-tensor fact weights-desc)
                  weights-conn (connector weights-tz (weights-md train-pd))
                  dst-conn (connector (dst-md train-pd) dst-tz)
                  fwd-prim (primitive train-pd)
                  fwd-args (dnnl-args fwd-args (output src-conn)
                                      (output weights-conn) bias-tz (input dst-tz))
                  bwd-src-conn (connector src-conn (src-md bwd-weights-pd))
                  diff-dst-conn (connector dst-tz (diff-dst-md bwd-weights-pd))
                  diff-weights-tz (dnnl-tensor fact weights-desc)
                  post-diff-weights-tz (if post-process-diff? (dnnl-tensor fact weights-desc)
                                           diff-weights-tz)
                  diff-weights-conn (connector (diff-weights-md bwd-weights-pd)
                                               diff-weights-tz)
                  diff-bias-tz (dnnl-tensor fact bias-desc)
                  bwd-weights-prim (primitive bwd-weights-pd)
                  bwd-weights-args (dnnl-args bwd-args (output bwd-src-conn)
                                              (output diff-dst-conn)
                                              (input diff-weights-conn)
                                              diff-bias-tz)]
      (if prop-diff?
        (let-release [diff-dst-data-conn (connector diff-dst-conn (diff-dst-md bwd-data-pd))
                      weights-data-conn (connector weights-tz (weights-md bwd-data-pd))
                      diff-src-conn (connector (diff-src-md bwd-data-pd) src-tz)
                      bwd-data-prim (primitive bwd-data-pd)
                      bwd-data-args (dnnl-args bwd-args
                                               (output diff-dst-data-conn)
                                               (output weights-data-conn)
                                               (input diff-src-conn))]
          (->DnnlProductTraining fact (flow fact) this bias-tz weights-tz dst-tz
                                 diff-weights-tz post-diff-weights-tz diff-bias-tz
                                 src-conn weights-conn
                                 fwd-prim fwd-args
                                 bwd-src-conn diff-dst-conn diff-weights-conn
                                 bwd-weights-prim bwd-weights-args
                                 diff-dst-data-conn weights-data-conn diff-src-conn
                                 bwd-data-prim bwd-data-args))
        (->DnnlProductTraining fact (flow fact) this bias-tz weights-tz dst-tz
                               diff-weights-tz post-diff-weights-tz diff-bias-tz
                               src-conn weights-conn
                               fwd-prim fwd-args
                               bwd-src-conn diff-dst-conn diff-weights-conn
                               bwd-weights-prim bwd-weights-args
                               nil nil nil nil nil))))
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(defn dnnl-inner-product-blueprint
  ([fact eng src-desc dst-desc weights-type]
   (let [src-desc (desc src-desc)
         dst-desc (desc dst-desc)
         dst-shape (dims dst-desc)
         dst-type (data-type dst-desc)
         weights-shape (vec (cons (dst-shape 1) (rest (dims src-desc))))
         weights-type (or weights-type (data-type src-desc) dst-type)
         bias-shape [(dst-shape 1)]]
     (let-release [bias-desc (memory-desc bias-shape dst-type :x)]
       (with-release [weights-desc-any (memory-desc weights-shape weights-type :any)
                      infer-desc (inner-product-fwd-desc :inference src-desc weights-desc-any
                                                         bias-desc dst-desc)
                      train-desc (inner-product-fwd-desc :training src-desc weights-desc-any
                                                         bias-desc dst-desc)
                      bwd-weights-desc (inner-product-bwd-desc src-desc weights-desc-any
                                                               bias-desc dst-desc)
                      bwd-data-desc (inner-product-bwd-desc src-desc weights-desc-any dst-desc)]
         (let-release [infer-pd (primitive-desc eng infer-desc)
                       train-pd (primitive-desc eng train-desc)
                       bwd-weights-pd (primitive-desc eng bwd-weights-desc train-pd)
                       bwd-data-pd (primitive-desc eng bwd-data-desc train-pd)
                       weights-desc-export (dnnl-contiguous-desc (weights-md train-pd))]
           (->DnnlProductBlueprint fact weights-desc-export bias-desc infer-pd
                                   train-pd bwd-weights-pd bwd-data-pd))))))
  ([fact eng src-desc dst-desc]
   (dnnl-inner-product-blueprint fact eng src-desc dst-desc nil)))

;; ================================ Directed Layer ==================================

(defn dnnl-fc-blueprint [fact eng src-desc dst-desc activ alpha beta weights-type]
  (with-release [src-desc (memory-desc (shape src-desc) (or (tz/data-type src-desc) :float) :any)
                 dst-desc (memory-desc [(first (shape dst-desc)) (apply * (rest (shape dst-desc)))]
                                       (or (tz/data-type dst-desc) (tz/data-type src-desc)) :any)]
    (let-release [ip-bluep (dnnl-inner-product-blueprint fact eng src-desc dst-desc weights-type)
                  activ-bluep (dnnl-activ-blueprint fact eng (inf-desc ip-bluep) (train-desc ip-bluep)
                                                    (train-desc ip-bluep) activ alpha beta)]
      (->DirectedLayerBlueprint fact :fc ip-bluep activ-bluep))))

;; ============================= Cost Function ========================================

(deftype DnnlUniversalCost [strm prev-layer
                            sum-prim sum-args
                            connect-output connect-diff train-tz
                            a-y cost]
  Releaseable
  (release [_]
    (release sum-prim)
    (release connect-output)
    (release connect-diff)
    (release train-tz)
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
    (execute! strm sum-prim sum-args)
    (connect-diff)
    (backward prev-layer)
    this)
  IFn
  (invoke [_]
    (connect-output)
    (execute! strm sum-prim sum-args)
    (cost a-y))
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(defn dnnl-universal-cost [eng strm prev-layer train-tz cost]
  (let [train-desc (desc train-tz)]
    (let-release [connect-output (connector (output prev-layer) train-desc)
                  connect-diff (connector train-desc (diff-input prev-layer))]
      (with-release [sum-desc (sum! eng train-desc 1.0 train-desc -1.0 train-tz)]
        (let-release [sum-prim (primitive sum-desc)]
          (->DnnlUniversalCost strm prev-layer sum-prim
                               (dnnl-args args (input connect-diff)
                                          (output connect-output) train-tz)
                               connect-output connect-diff train-tz
                               (view-vctr (input connect-diff))
                               cost))))))

(deftype DnnlCustomCost [strm prev-layer sum-prim sum-args
                         connect-output connect-diff train-tz a y cost]
  Releaseable
  (release [_]
    (release sum-prim)
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
    (execute! strm sum-prim sum-args)
    (connect-diff)
    this)
  IFn
  (invoke [_]
    (connect-output)
    (cost y a))
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(defn dnnl-custom-cost [eng strm prev-layer train-tz cost]
  (let [train-desc (desc train-tz)]
    (let-release [connect-output (connector (output prev-layer) train-desc)
                  connect-diff (connector train-desc (diff-z prev-layer))]
      (with-release [sum-desc (sum! eng train-desc 1.0 train-desc -1.0 train-desc)]
        (let-release [sum-prim (primitive sum-desc)]
          (->DnnlCustomCost strm prev-layer sum-prim
                            (dnnl-args args (input connect-diff)
                                       (output connect-output) train-tz)
                            connect-output connect-diff (view train-tz)
                            (view-vctr (output connect-output)) (view-vctr train-tz)
                            cost))))))

;; =========================== Convolution =============================================

(defn dnnl-convolution-op-blueprint
  [fact eng src-desc weights-desc dst-desc strides dilation padding-l padding-r]
  (let [src-desc (desc src-desc)
        dst-desc (desc dst-desc)]
    (let-release [bias-desc (memory-desc [(get (dims dst-desc) 1)] (data-type dst-desc) :x)]
      (with-release [weights-desc (desc weights-desc)
                     conv-infer-desc (convolution-fwd-desc :inference :auto
                                                           src-desc weights-desc bias-desc dst-desc
                                                           strides dilation padding-l padding-r)
                     conv-train-desc (convolution-fwd-desc :training :auto
                                                           src-desc weights-desc bias-desc dst-desc
                                                           strides dilation padding-l padding-r)
                     conv-bwd-weights-desc (convolution-bwd-desc :auto
                                                                 src-desc weights-desc
                                                                 bias-desc dst-desc
                                                                 strides dilation padding-l padding-r)
                     conv-bwd-data-desc (convolution-bwd-desc :auto
                                                              src-desc weights-desc dst-desc
                                                              strides dilation padding-l padding-r)]
        (let-release [conv-infer-pd (primitive-desc eng conv-infer-desc)
                      conv-train-pd (primitive-desc eng conv-train-desc)
                      conv-bwd-weights-pd (primitive-desc eng conv-bwd-weights-desc conv-train-pd)
                      conv-bwd-data-pd (primitive-desc eng conv-bwd-data-desc conv-train-pd)
                      weights-desc-export (dnnl-contiguous-desc (weights-md conv-train-pd))]
          (->DnnlProductBlueprint fact weights-desc-export bias-desc conv-infer-pd
                                  conv-train-pd conv-bwd-weights-pd conv-bwd-data-pd))))))

(defn dnnl-convolution-layer-blueprint [fact eng src-desc weights-desc dst-desc activ
                                        strides dilation padding-l padding-r alpha beta]
  (let-release [src-desc (memory-desc (shape src-desc) (or (tz/data-type src-desc) :float) :any)
                dst-desc (memory-desc (shape dst-desc)
                                      (or (tz/data-type dst-desc) (data-type src-desc))
                                      :any)
                convolution-bluep (dnnl-convolution-op-blueprint fact eng src-desc weights-desc
                                                                 dst-desc strides dilation
                                                                 padding-l padding-r)
                activ-bluep (dnnl-activ-blueprint fact eng
                                                  (inf-desc convolution-bluep)
                                                  (train-desc convolution-bluep)
                                                  (train-desc convolution-bluep)
                                                  activ alpha beta)]
    (->DirectedLayerBlueprint fact :convolution convolution-bluep activ-bluep)))

;; ================================ Pooling =============================================

(deftype DnnlPoolingInferenceLayer [fact strm bluep src-conn dst-tz
                                    pooling-fwd-prim pooling-fwd-args]
  Releaseable
  (release [_]
    (release src-conn)
    (release dst-tz)
    (release pooling-fwd-prim))
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
    (input src-conn))
  (output [_]
    dst-tz)
  ParametersSeq
  (parameters [_]
    [])
  IFn
  (invoke [_]
    (src-conn)
    (execute! strm pooling-fwd-prim pooling-fwd-args)
    dst-tz)
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(deftype DnnlPoolingTrainingLayer [fact strm bluep src-conn dst-tz workspace-tz
                                   pooling-fwd-prim pooling-fwd-args
                                   diff-dst-conn diff-src-conn
                                   pooling-bwd-prim pooling-bwd-args]
  Releaseable
  (release [_]
    (release src-conn)
    (release dst-tz)
    (release workspace-tz)
    (release pooling-fwd-prim)
    (release diff-dst-conn)
    (release diff-src-conn)
    (release pooling-bwd-prim))
  Info
  (info [this]
    {:algo (info bluep :algo)
     :dst (info dst-tz)
     :workspace (info (desc workspace-tz))
     :shape (shape dst-tz)})
  (info [this info-type]
    (case info-type
      :algo (info bluep :algo)
      :dst (info dst-tz)
      :workspace (info (desc workspace-tz))
      (info bluep info-type)))
  Transfer
  (input [_]
    (input src-conn))
  (output [_]
    dst-tz)
  DiffTransfer
  (diff-input [_]
    dst-tz)
  (diff-output [_]
    (input src-conn))
  ParametersSeq
  (parameters [_]
    [])
  IFn
  (invoke [_]
    (src-conn)
    (execute! strm pooling-fwd-prim pooling-fwd-args)
    dst-tz)
  (applyTo [this xs]
    (AFn/applyToHelper this xs))
  Backprop
  (forward [this]
    this)
  (forward [this _]
    (src-conn)
    (execute! strm pooling-fwd-prim pooling-fwd-args)
    this)
  (backward [this]
    this)
  (backward [this _]
    (when pooling-bwd-prim
      (diff-dst-conn)
      (execute! strm pooling-bwd-prim pooling-bwd-args)
      (diff-src-conn))
    this))

(deftype DnnlPoolingBlueprint [fact algo pool-infer-pd pool-train-pd pool-bwd-pd]
  Releaseable
  (release [_]
    (release pool-infer-pd)
    (release pool-train-pd)
    (release pool-bwd-pd))
  Info
  (info [this]
    {:algo algo
     :shape (shape this)
     :topology :pooling})
  (info [this info-type]
    (case info-type
      :algo algo
      :shape (shape this)
      :topology :pooling
      nil))
  DiamondFactoryProvider
  (diamond-factory [_]
    fact)
  DescriptorProvider
  (inf-desc [_]
    (dst-md pool-infer-pd))
  (train-desc [_]
    (dst-md pool-train-pd))
  TensorDescriptor
  (shape [this]
    (shape (train-desc this)))
  (data-type [this]
    (data-type (train-desc this)))
  (layout [this]
    (layout (train-desc this)))
  IFn
  (invoke [this prev-layer]
    (let-release [src-tz (output prev-layer)
                  src-conn (connector src-tz (src-md pool-infer-pd))
                  dst-tz (dnnl-tensor fact (dst-md pool-infer-pd))
                  pool-infer-prim (primitive pool-infer-pd)
                  pool-infer-args (fwd-args (buffer (output src-conn))
                                            (buffer dst-tz))]
      (->DnnlPoolingInferenceLayer fact (flow fact) this src-conn dst-tz
                                   pool-infer-prim pool-infer-args)))
  (invoke [this prev-layer prop-diff? _]
    (let-release [src-tz (output prev-layer)
                  src-conn (connector src-tz (src-md pool-train-pd))
                  dst-tz (dnnl-tensor fact (dst-md pool-train-pd))
                  workspace-tz (when-let [workspace-desc (workspace-md pool-train-pd)]
                                 (dnnl-tensor fact workspace-desc))
                  pool-train-prim (primitive pool-train-pd)
                  pool-train-args (dnnl-args fwd-args (output src-conn) dst-tz workspace-tz)]
      (if prop-diff?
        (let-release [diff-dst-conn (connector dst-tz (diff-dst-md pool-bwd-pd))
                      diff-src-conn (connector (diff-src-md pool-bwd-pd) src-tz)
                      pool-bwd-prim (primitive pool-bwd-pd)
                      pool-bwd-args (dnnl-args pooling-bwd-args (output diff-dst-conn)
                                               (input diff-src-conn) workspace-tz)]
          (->DnnlPoolingTrainingLayer fact (flow fact) this src-conn dst-tz workspace-tz
                                      pool-train-prim pool-train-args
                                      diff-dst-conn diff-src-conn
                                      pool-bwd-prim pool-bwd-args))
        (->DnnlPoolingTrainingLayer fact (flow fact) this src-conn dst-tz workspace-tz
                                    pool-train-prim pool-train-args
                                    nil nil nil nil))))
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(defn dnnl-pooling-blueprint
  [fact eng src-desc dst-desc algo strides kernel padding-l padding-r]
  (let-release [inf-src-desc (inf-desc src-desc);;TODO maybe this could be with-release...
                train-src-desc (train-desc src-desc)
                inf-dst-desc (memory-desc (shape dst-desc)
                                          (or (tz/data-type dst-desc) (data-type inf-src-desc))
                                          (or (tz/layout dst-desc) :any))
                train-dst-desc (memory-desc (shape dst-desc)
                                            (or (tz/data-type dst-desc) (data-type train-src-desc))
                                            (or (tz/layout dst-desc) :any))]
    (with-release [pool-infer-desc (pooling-fwd-desc :inference algo inf-src-desc inf-dst-desc
                                                     kernel strides padding-l padding-r)
                   pool-train-desc (pooling-fwd-desc :training algo train-src-desc train-dst-desc
                                                     kernel strides padding-l padding-r)
                   pool-bwd-desc (pooling-bwd-desc algo train-src-desc train-dst-desc
                                                   kernel strides padding-l padding-r)]
      (let-release [pool-infer-pd (primitive-desc eng pool-infer-desc)
                    pool-train-pd (primitive-desc eng pool-train-desc)
                    pool-bwd-pd (primitive-desc eng pool-bwd-desc pool-train-pd)]
        (->DnnlPoolingBlueprint fact algo pool-infer-pd pool-train-pd pool-bwd-pd)))))

(defmethod transfer! [DnnlPoolingInferenceLayer Object]
  [source destination]
  destination)

(defmethod transfer! [DnnlPoolingTrainingLayer Object]
  [source destination]
  destination)

;; ====================== Dropout ====================================================

(defn dnnl-gaussian-dropout-blueprint [fact src-desc sd]
  (let-release [mask-desc (dnnl-contiguous-desc (train-desc src-desc))]
    (->GaussianDropoutBlueprint fact sd mask-desc)))

;; ================================ Batch Normalization  ===========================================

(deftype DnnlBatchNormalizationInference [fact strm bluep scaleshift-tz
                                          src-conn gamma-tz beta-tz mean-tz var-tz
                                          fwd-prim fwd-args]
  Releaseable
  (release [_]
    (release src-conn)
    (release scaleshift-tz)
    (release mean-tz)
    (release var-tz)
    (release fwd-prim))
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
  IFn
  (invoke [_]
    (src-conn)
    (execute! strm fwd-prim fwd-args)
    (output src-conn))
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(deftype DnnlBatchNormalizationTraining [fact strm bluep scaleshift-tz diff-scaleshift-tz
                                         src-conn gamma-tz beta-tz dst-tz mean-tz var-tz
                                         diff-gamma-tz diff-beta-tz post-diff-gamma-tz
                                         diff-src-conn
                                         fwd-prim fwd-args bwd-prim bwd-args]
  Releaseable
  (release [_]
    (release scaleshift-tz)
    (release diff-scaleshift-tz)
    (release src-conn)
    (release dst-tz)
    (release mean-tz)
    (release var-tz)
    (release gamma-tz)
    (release beta-tz)
    (release diff-gamma-tz)
    (release post-diff-gamma-tz)
    (release diff-beta-tz)
    (release diff-src-conn)
    (release fwd-prim)
    (release fwd-args)
    (release bwd-prim)
    (release bwd-args))
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
  DiffParameters
  (diff-weights [_]
    post-diff-gamma-tz)
  IFn
  (invoke [_]
    (src-conn)
    (execute! strm fwd-prim fwd-args)
    dst-tz)
  (applyTo [this xs]
    (AFn/applyToHelper this xs))
  Backprop
  (forward [this]
    (src-conn)
    (execute! strm fwd-prim fwd-args)
    this)
  (backward [this]
    (backward-diff this 1.0 0.0 1.0 0.0))
  LinearBackprop
  (backward-diff [this scal-diff-w scal-g scal-diff-b scal-b]
    (execute! strm bwd-prim bwd-args)
    (if (= 0.0 scal-g)
      (when-not (= 1.0 scal-diff-w) (scal! scal-diff-w diff-gamma-tz))
      (axpby! scal-diff-w diff-gamma-tz scal-g post-diff-gamma-tz))
    (axpby! scal-diff-b diff-beta-tz scal-b beta-tz)
    (diff-src-conn)
    this))

(deftype DnnlBatchNormalizationBlueprint [fact data-desc scaleshift-desc gamma-desc
                                          infer-pd train-pd bwd-pd]
  Object
  (hashCode [_]
    (-> (hash :batch-normalization) (hash-combine scaleshift-desc)))
  (equals [_ other]
    (and (instance? DnnlBatchNormalizationBlueprint other)
         (= scaleshift-desc (.scaleshift-desc ^DnnlBatchNormalizationBlueprint other))
         (equal-desc? (src-md infer-pd) (src-md (.infer-pd ^DnnlBatchNormalizationBlueprint other)))
         (equal-desc? (src-md train-pd) (src-md (.train-pd ^DnnlBatchNormalizationBlueprint other)))
         (equal-desc? (dst-md train-pd) (dst-md (.train-pd ^DnnlBatchNormalizationBlueprint other)))))
  (toString [this]
    (pr-str {:src (src-md infer-pd)
             :dst (dst-md infer-pd)}))
  Releaseable
  (release [_]
    (release infer-pd)
    (release train-pd)
    (release bwd-pd))
  Info
  (info [this info-type]
    (case info-type
      :bias gamma-desc
      :inference {:src (src-md infer-pd)
                  :weights gamma-desc
                  :dst (dst-md infer-pd)}
      :training {:src (src-md train-pd)
                 :weights gamma-desc
                 :dst (dst-md train-pd)}
      nil))
  (info [this]
    {:bias gamma-desc
     :inference {:src (src-md infer-pd)
                 :weights gamma-desc
                 :dst (dst-md infer-pd)}
     :training {:src (src-md train-pd)
                :weights gamma-desc
                :dst (dst-md train-pd)}})
  DescriptorProvider
  (inf-desc [_]
    (dst-md infer-pd))
  (train-desc [_]
    (dst-md train-pd))
  TensorDescriptor
  (shape [this]
    (shape (train-desc train-pd)))
  (data-type [this]
    (data-type (train-desc train-pd)))
  (layout [this]
    (layout (train-desc train-pd)))
  IFn
  (invoke [this src-tz]
    (let-release [src-conn (connector src-tz data-desc)
                  scaleshift-tz (dnnl-tensor fact scaleshift-desc)
                  gamma-tz (view-tz scaleshift-tz gamma-desc)
                  beta-tz (tz/offset! (view-tz scaleshift-tz gamma-desc) 1)
                  mean-tz (dnnl-tensor fact gamma-desc)
                  var-tz (dnnl-tensor fact gamma-desc)
                  fwd-prim (primitive infer-pd)
                  fwd-args (dnnl-args batch-norm-fwd-args (output src-conn) (output src-conn)
                                      scaleshift-tz mean-tz var-tz)]
      (->DnnlBatchNormalizationInference fact (flow fact) this scaleshift-tz
                                         src-conn gamma-tz beta-tz mean-tz var-tz
                                         fwd-prim fwd-args)))
  (invoke [this src-tz dst-tz _ post-process-diff?]
    (let-release [src-conn (connector src-tz data-desc)
                  scaleshift-tz (dnnl-tensor fact scaleshift-desc)
                  diff-scaleshift-tz (dnnl-tensor fact scaleshift-desc)
                  gamma-tz (view-tz scaleshift-tz gamma-desc)
                  beta-tz (tz/offset! (view-tz scaleshift-tz gamma-desc) 1)
                  mean-tz (dnnl-tensor fact gamma-desc)
                  var-tz (dnnl-tensor fact gamma-desc)
                  diff-gamma-tz (view-tz diff-scaleshift-tz gamma-desc)
                  diff-beta-tz (tz/offset! (view-tz diff-scaleshift-tz gamma-desc) 1)
                  post-diff-gamma-tz (if post-process-diff? (dnnl-tensor fact gamma-desc)
                                         diff-gamma-tz)
                  diff-src-conn (revert src-conn)
                  fwd-prim (primitive train-pd)
                  fwd-args (dnnl-args batch-norm-fwd-args (output src-conn) dst-tz scaleshift-tz
                                            mean-tz var-tz)
                  bwd-prim (primitive bwd-pd)
                  bwd-args (dnnl-args batch-norm-bwd-args dst-tz (output src-conn) scaleshift-tz
                                      mean-tz var-tz (output src-conn) diff-scaleshift-tz)]
      (->DnnlBatchNormalizationTraining fact (flow fact) this scaleshift-tz diff-scaleshift-tz
                                        src-conn gamma-tz beta-tz dst-tz mean-tz var-tz
                                        diff-gamma-tz diff-beta-tz post-diff-gamma-tz diff-src-conn
                                        fwd-prim fwd-args
                                        bwd-prim bwd-args)))
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(defn dnnl-batch-norm-op-blueprint
  [fact eng data-desc]
  (let [data-desc (desc data-desc)
        c (get (dims data-desc) 1)
        scaleshift-desc (memory-desc [2 c] :float :ab)
        gamma-desc (memory-desc [1 c] :float :ab)]
    (with-release [bnorm-infer-desc (batch-norm-fwd-desc :inference data-desc :scaleshift :global-stats)
                   bnorm-train-desc (batch-norm-fwd-desc :training data-desc :scaleshift)
                   bnorm-bwd-desc (batch-norm-bwd-desc :backward data-desc data-desc :scaleshift)]
      (let-release [bnorm-infer-pd (primitive-desc eng bnorm-infer-desc)
                    bnorm-train-pd (primitive-desc eng bnorm-train-desc)
                    bnorm-bwd-pd (primitive-desc eng bnorm-bwd-desc bnorm-train-pd)]
        (->DnnlBatchNormalizationBlueprint fact data-desc scaleshift-desc gamma-desc
                                           bnorm-infer-pd bnorm-train-pd bnorm-bwd-pd)))))

(defn dnnl-batch-norm-layer-blueprint [fact eng src-desc activ alpha beta]
  (let-release [data-desc (dnnl-contiguous-desc (desc src-desc))
                batch-norm-op-bluep (dnnl-batch-norm-op-blueprint fact eng data-desc)
                activ-bluep (dnnl-activ-blueprint fact eng (inf-desc batch-norm-op-bluep)
                                                  (train-desc batch-norm-op-bluep)
                                                  (train-desc batch-norm-op-bluep)
                                                  activ alpha beta)]
    (->DirectedLayerBlueprint fact :batch-normalization batch-norm-op-bluep activ-bluep)))
