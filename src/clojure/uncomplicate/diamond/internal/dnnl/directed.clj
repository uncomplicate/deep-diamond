;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.diamond.internal.dnnl.directed
  (:require [uncomplicate.commons.core
             :refer [Releaseable release let-release with-release Info info view]]
            [uncomplicate.fluokitten.core :refer [fmap]]
            [uncomplicate.neanderthal.core :refer [axpby! dim transfer! scal! view-vctr]]
            [uncomplicate.neanderthal.internal.api :refer [flow]]
            [uncomplicate.diamond.tensor :as tz
             :refer [Transfer input output connector revert shape layout TensorDescriptor view-tz]]
            [uncomplicate.diamond.internal
             [protocols
              :refer [Parameters bias weights ParametersSeq parameters DescriptorProvider
                      DiamondFactoryProvider DiffParameters diff-weights Backprop forward backward
                      DiffTransfer diff-input diff-output diff-z LinearBackprop backward-diff
                      inf-desc train-desc diff-desc Initializable init batch-index create-tensor]]
             [utils :refer [transfer-weights-bias! concat-strides concat-dst-shape direction-count]]]
            [uncomplicate.diamond.internal.dnnl
             [protocols :refer :all]
             [core :refer :all :as dnnl]
             [tensor :refer [dnnl-tensor dnnl-transformer]]]
            [uncomplicate.diamond.internal.neanderthal.directed
             :refer [->DirectedLayerBlueprint ->GaussianDropoutBlueprint ->NopActivation
                     ->NopActivationBlueprint]])
  (:import [clojure.lang IFn AFn]
           [uncomplicate.diamond.internal.neanderthal.directed DirectedLayerBlueprint GaussianDropoutBlueprint]))

;; ================================ Activation =============================================

(deftype DnnlActivationInference [strm bluep data-conn eltwise-fwd-prim eltwise-fwd-args]
  Releaseable
  (release [_]
    (release data-conn)
    (release eltwise-fwd-prim))
  Info
  (info [this]
    {:activation (info bluep :activation)
     :data (info data-conn)})
  (info [this info-type]
    (case info-type
      :data (info data-conn)
      (info bluep info-type)))
  Transfer
  (input [_]
    (input data-conn))
  (output [_]
    (output data-conn))
  Initializable
  (init [this _]
    this)
  IFn
  (invoke [_]
    (data-conn)
    (execute! strm eltwise-fwd-prim eltwise-fwd-args)
    (output data-conn))
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(deftype DnnlActivationTraining [strm bluep src-conn dst-tz diff-dst-tz diff-src-conn
                                 eltwise-fwd-prim eltwise-fwd-args
                                 eltwise-bwd-prim eltwise-bwd-args]
  Releaseable
  (release [_]
    (release src-conn)
    (release dst-tz)
    (release diff-dst-tz)
    (release diff-src-conn)
    (release eltwise-fwd-prim)
    (release eltwise-bwd-prim))
  Info
  (info [this]
    {:activation (info bluep :activation)
     :src (info (input src-conn))
     :dst (info dst-tz)
     :diff-dst (info diff-dst-tz)
     :diff-src (info diff-src-conn)})
  (info [this info-type]
    (case info-type
      :src (info (input src-conn))
      :dst (info dst-tz)
      :diff-dst (info diff-dst-tz)
      :diff-src (info diff-src-conn)
      (info bluep info-type)))
  Transfer
  (input [_]
    (input src-conn))
  (output [_]
    dst-tz)
  DiffTransfer
  (diff-input [_]
    diff-dst-tz)
  (diff-output [_]
    (output diff-src-conn))
  Initializable
  (init [this _]
    this)
  IFn
  (invoke [_]
    (src-conn)
    (execute! strm eltwise-fwd-prim eltwise-fwd-args)
    dst-tz)
  (applyTo [this xs]
    (AFn/applyToHelper this xs))
  Backprop
  (forward [this]
    (src-conn)
    (execute! strm eltwise-fwd-prim eltwise-fwd-args)
    this)
  (backward [this]
    (execute! strm eltwise-bwd-prim eltwise-bwd-args)
    (diff-src-conn)
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
  DiamondFactoryProvider
  (diamond-factory [_]
    fact)
  DescriptorProvider
  (inf-desc [_]
    (dst-md eltw-infer-pd))
  (train-desc [_]
    (dst-md eltw-train-pd))
  (diff-desc [_]
    (diff-dst-md eltw-bwd-pd))
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
                  eltw-fwd-args (fwd-args src-tz)]
      (->DnnlActivationInference (flow fact) this src-tz
                                 eltw-fwd-prim eltw-fwd-args)))
  (invoke [this src-tz diff-tz]
    (let-release [dst-tz (dnnl-tensor fact (dst-md eltw-train-pd) (batch-index src-tz))
                  eltw-fwd-prim (primitive eltw-train-pd)
                  eltw-fwd-args (fwd-args src-tz dst-tz)
                  eltw-bwd-prim (primitive eltw-bwd-pd)
                  diff-src-conn (connector (diff-src-md eltw-bwd-pd) diff-tz)
                  eltw-bwd-args (eltwise-bwd-args src-tz dst-tz (input diff-src-conn))]
      (->DnnlActivationTraining (flow fact) this src-tz dst-tz dst-tz diff-src-conn
                                eltw-fwd-prim eltw-fwd-args
                                eltw-bwd-prim eltw-bwd-args)))
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

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
  DiamondFactoryProvider
  (diamond-factory [_]
    fact)
  DescriptorProvider
  (inf-desc [_]
    (dst-md softmax-infer-pd))
  (train-desc [_]
    (dst-md softmax-train-pd))
  (diff-desc [_]
    (diff-dst-md softmax-bwd-pd))
  TensorDescriptor
  (shape [this]
    (shape (train-desc this)))
  (data-type [this]
    (data-type (train-desc this)))
  (layout [this]
    (layout (train-desc this)))
  Initializable
  (init [this _]
    this)
  IFn
  (invoke [this src-tz]
    (let-release [src-conn (connector src-tz (src-md softmax-infer-pd))
                  softmax-fwd-prim (primitive softmax-infer-pd)
                  softmax-fwd-args (fwd-args (output src-conn))]
      (->DnnlActivationInference (flow fact) this src-conn
                                 softmax-fwd-prim softmax-fwd-args)))
  (invoke [this src-tz diff-tz]
    (let-release [src-conn (connector src-tz (src-md softmax-infer-pd))
                  diff-dst-tz (dnnl-tensor fact (diff-dst-md softmax-bwd-pd) (batch-index src-tz))
                  softmax-fwd-prim (primitive softmax-train-pd)
                  softmax-fwd-args (fwd-args (output src-conn))
                  softmax-bwd-prim (primitive softmax-bwd-pd)
                  softmax-bwd-args (softmax-bwd-args (output src-conn) diff-dst-tz diff-tz)]
      (->DnnlActivationTraining (flow fact) this src-conn (output src-conn) diff-dst-tz diff-tz
                                softmax-fwd-prim softmax-fwd-args
                                softmax-bwd-prim softmax-bwd-args)))
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(defn dnnl-nop-activation-blueprint
  [fact inf-src-desc train-src-desc diff-desc]
  (let-release [inf-src-desc (view (desc inf-src-desc))
                train-src-desc (view (desc train-src-desc))
                diff-desc (view (desc diff-desc))]
    (->NopActivationBlueprint fact inf-src-desc train-src-desc diff-desc)))

(defn dnnl-activation-blueprint
  ([fact eng inf-src-desc train-src-desc activ alpha beta]
   (let-release [inf-src-desc (desc inf-src-desc)
                 train-src-desc (desc train-src-desc)
                 eltw-infer-pd (eltwise-fwd eng :inference activ inf-src-desc alpha beta)
                 eltw-train-pd (eltwise-fwd eng :training activ train-src-desc alpha beta)
                 eltw-bwd-pd (eltwise-bwd eng eltw-train-pd activ train-src-desc train-src-desc alpha beta)]
     (->DnnlActivationBlueprint fact activ eltw-infer-pd eltw-train-pd eltw-bwd-pd)))
  ([fact eng data-desc activ alpha beta]
   (dnnl-activation-blueprint fact eng data-desc data-desc activ alpha beta)))

(defn dnnl-softmax-blueprint [fact eng inf-src-desc train-src-desc]
  (with-release [inf-desc (case (count (shape inf-src-desc))
                            2 (memory-desc (shape inf-src-desc) (data-type inf-src-desc) :ab)
                            4 (memory-desc (shape inf-src-desc) (data-type inf-src-desc) :acdb)
                            (view (desc inf-src-desc)))
                 train-desc (case (count (shape train-src-desc))
                              2 (memory-desc (shape train-src-desc) (data-type train-src-desc) :ab)
                              4 (memory-desc (shape train-src-desc) (data-type train-src-desc) :acdb)
                              (view (desc train-src-desc)))]
    (let-release [softmax-infer-pd (softmax-fwd eng :inference :accurate inf-desc 1) ;;TODO currently DNNL is optimized for 1
                  softmax-train-pd (softmax-fwd eng :training :accurate train-desc 1)
                  softmax-bwd-pd (softmax-bwd eng softmax-train-pd :accurate train-desc train-desc 1)]
      (->DnnlSoftmaxBlueprint fact softmax-infer-pd softmax-train-pd softmax-bwd-pd))))

(defn dnnl-activ-blueprint
  ([fact eng inf-src-desc train-src-desc diff-desc activ alpha beta]
   (case activ
     :identity (dnnl-nop-activation-blueprint fact inf-src-desc train-src-desc diff-desc)
     :softmax (dnnl-softmax-blueprint fact eng inf-src-desc train-src-desc)
     (dnnl-activation-blueprint fact eng inf-src-desc train-src-desc activ alpha beta)))
  ([fact eng data-desc activ alpha beta]
   (dnnl-activ-blueprint fact eng data-desc data-desc data-desc activ alpha beta)))

;; ================================ Inner Product & Convolution ====================================

(deftype DnnlProductInference [strm bluep
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
  Initializable
  (init [this init-fn]
    (init-fn weights-tz)
    (init-fn bias-tz)
    this)
  IFn
  (invoke [_]
    (src-conn)
    (execute! strm fwd-prim fwd-args)
    dst-tz)
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(deftype DnnlProductTraining [strm bluep
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
    post-diff-weights-tz)
  Initializable
  (init [this init-fn]
    (init-fn weights-tz)
    (init-fn bias-tz)
    this)
  IFn
  (invoke [this]
    (forward this)
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
         (let [other ^DnnlProductBlueprint other]
           (and
            (= bias-desc (.bias-desc other))
            (equal-desc? (src-md infer-pd) (src-md (.infer-pd ^DnnlProductBlueprint other)))
            (equal-desc? (weights-md infer-pd) (weights-md (.infer-pd ^DnnlProductBlueprint other)))
            (equal-desc? (dst-md infer-pd) (dst-md (.infer-pd ^DnnlProductBlueprint other)))
            (equal-desc? (src-md train-pd) (src-md (.train-pd ^DnnlProductBlueprint other)))
            (equal-desc? (weights-md train-pd) (weights-md (.train-pd ^DnnlProductBlueprint other)))
            (equal-desc? (dst-md train-pd) (dst-md (.train-pd ^DnnlProductBlueprint other)))))))
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
  (diff-desc [_]
    (dst-md train-pd))
  IFn
  (invoke [this src-tz]
    (let-release [src-conn (connector src-tz (src-md infer-pd))
                  bias-tz (dnnl-tensor fact bias-desc)
                  weights-tz (dnnl-tensor fact (weights-md infer-pd))
                  dst-tz (dnnl-tensor fact (dst-md infer-pd) (batch-index src-tz))
                  fwd-prim (primitive infer-pd)
                  fwd-args (fwd-args (output src-conn) weights-tz bias-tz dst-tz)]
      (->DnnlProductInference (flow fact) this
                              src-conn bias-tz weights-tz dst-tz
                              fwd-prim fwd-args)))
  (invoke [this src-tz diff-src-tz prop-diff? post-process-diff?]
    (let-release [src-conn (connector src-tz (src-md train-pd))
                  bias-tz (dnnl-tensor fact bias-desc)
                  weights-tz (dnnl-tensor fact weights-desc)
                  weights-conn (connector weights-tz (weights-md train-pd))
                  dst-tz (dnnl-tensor fact (dst-md train-pd) (batch-index src-tz))
                  fwd-prim (primitive train-pd)
                  fwd-args (fwd-args (output src-conn) (output weights-conn) bias-tz dst-tz)
                  bwd-src-conn (connector src-conn (src-md bwd-weights-pd))
                  diff-dst-conn (connector dst-tz (diff-dst-md bwd-weights-pd))
                  diff-weights-tz (dnnl-tensor fact weights-desc)
                  post-diff-weights-tz (if post-process-diff? (dnnl-tensor fact weights-desc)
                                           diff-weights-tz)
                  diff-weights-conn (connector (diff-weights-md bwd-weights-pd) diff-weights-tz)
                  diff-bias-tz (dnnl-tensor fact bias-desc)
                  bwd-weights-prim (primitive bwd-weights-pd)
                  bwd-weights-args (bwd-args (output bwd-src-conn)
                                             (output diff-dst-conn)
                                             (input diff-weights-conn)
                                             diff-bias-tz)]
      (if prop-diff?
        (let-release [diff-dst-data-conn (connector diff-dst-conn (diff-dst-md bwd-data-pd))
                      weights-data-conn (connector weights-tz (weights-md bwd-data-pd)) ;; TODO perhaps I have to use weights-weights-conn too? between forward and backward? Yes, according to the documentation!
                      diff-src-conn (connector (diff-src-md bwd-data-pd) diff-src-tz)
                      bwd-data-prim (primitive bwd-data-pd)
                      bwd-data-args (bwd-args (output diff-dst-data-conn)
                                              (output weights-data-conn)
                                              (input diff-src-conn))]
          (->DnnlProductTraining (flow fact) this bias-tz weights-tz dst-tz
                                 diff-weights-tz post-diff-weights-tz diff-bias-tz
                                 src-conn weights-conn
                                 fwd-prim fwd-args
                                 bwd-src-conn diff-dst-conn diff-weights-conn
                                 bwd-weights-prim bwd-weights-args
                                 diff-dst-data-conn weights-data-conn diff-src-conn
                                 bwd-data-prim bwd-data-args))
        (->DnnlProductTraining (flow fact) this bias-tz weights-tz dst-tz
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
     (with-release [weights-desc-any (memory-desc weights-shape weights-type :any)]
       (let-release [bias-desc (memory-desc bias-shape dst-type :x)
                     infer-pd (inner-product-fwd eng :inference src-desc weights-desc-any
                                                 bias-desc dst-desc)
                     train-pd (inner-product-fwd eng :training src-desc weights-desc-any
                                                 bias-desc dst-desc)
                     bwd-weights-pd (inner-product-bwd eng train-pd src-desc weights-desc-any
                                                       bias-desc dst-desc)
                     bwd-data-pd (inner-product-bwd eng train-pd src-desc weights-desc-any dst-desc)
                     weights-desc-export (dnnl-contiguous-desc (weights-md train-pd))]
         (->DnnlProductBlueprint fact weights-desc-export bias-desc infer-pd
                                 train-pd bwd-weights-pd bwd-data-pd)))))
  ([fact eng src-desc dst-desc]
   (dnnl-inner-product-blueprint fact eng src-desc dst-desc nil)))

(defmethod transfer! [DnnlProductInference Object]
  [source destination]
  (transfer-weights-bias! source destination))

(defmethod transfer! [DnnlProductTraining Object]
  [source destination]
  (transfer-weights-bias! source destination))

;; ================================ Directed Layer ==================================

(defn dnnl-fc-blueprint [fact eng src-desc dst-desc activ alpha beta weights-type]
  (with-release [src-desc (memory-desc (shape src-desc) (or (tz/data-type src-desc) :float) :any)
                 dst-desc (memory-desc [(first (shape dst-desc)) (apply * (rest (shape dst-desc)))]
                                       (or (tz/data-type dst-desc) (tz/data-type src-desc)) :any)]
    (let-release [ip-bluep (dnnl-inner-product-blueprint fact eng src-desc dst-desc weights-type)
                  activ-bluep (dnnl-activ-blueprint fact eng
                                                    (inf-desc ip-bluep)
                                                    (train-desc ip-bluep)
                                                    (diff-desc ip-bluep)
                                                    activ alpha beta)]
      (->DirectedLayerBlueprint fact :dense ip-bluep activ-bluep))))

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
    (with-release [sum-pd (sum! eng train-desc 1.0 train-desc -1.0 train-desc)]
      (let-release [connect-output (connector (output prev-layer) train-desc)
                    connect-diff (connector train-desc (diff-input prev-layer))
                    sum-prim (primitive sum-pd)]
        (->DnnlUniversalCost strm prev-layer sum-prim
                             (multi-args (input connect-diff) (output connect-output) train-tz)
                             connect-output connect-diff train-tz
                             (view-vctr (input connect-diff))
                             cost)))))

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
    (with-release [sum-pd (sum! eng train-desc 1.0 train-desc -1.0 train-desc)]
      (let-release [connect-output (connector (output prev-layer) train-desc)
                    connect-diff (connector train-desc (diff-z prev-layer))
                    sum-prim (primitive sum-pd)]
        (->DnnlCustomCost strm prev-layer sum-prim
                          (multi-args (input connect-diff) (output connect-output) train-tz)
                          connect-output connect-diff (view train-tz)
                          (view-vctr (output connect-output)) (view-vctr train-tz)
                          cost)))))

;; =========================== Convolution =============================================

(defn dnnl-convolution-op-blueprint
  [fact eng src-desc weights-desc dst-desc strides dilation padding-l padding-r]
  (let-release [src-desc (desc src-desc)
                dst-desc (desc dst-desc)
                weights-desc (desc weights-desc)
                bias-desc (memory-desc [(get (dims dst-desc) 1)] (data-type dst-desc) :x)
                conv-infer-pd (convolution-fwd eng :inference :auto
                                               src-desc weights-desc bias-desc dst-desc
                                               strides dilation padding-l padding-r)
                conv-train-pd (convolution-fwd eng :training :auto
                                               src-desc weights-desc bias-desc dst-desc
                                               strides dilation padding-l padding-r)
                conv-bwd-weights-pd (convolution-bwd-weights eng conv-train-pd :auto
                                                             src-desc weights-desc
                                                             bias-desc dst-desc
                                                             strides dilation padding-l padding-r)
                conv-bwd-data-pd (convolution-bwd-data eng conv-train-pd :auto
                                                       src-desc weights-desc dst-desc
                                                       strides dilation padding-l padding-r)
                weights-desc-export (dnnl-contiguous-desc (weights-md conv-train-pd))]
    (->DnnlProductBlueprint fact weights-desc-export bias-desc conv-infer-pd
                            conv-train-pd conv-bwd-weights-pd conv-bwd-data-pd)))

(defn dnnl-convolution-layer-blueprint [fact eng src-desc weights-desc dst-desc activ
                                        strides dilation padding-l padding-r alpha beta]
  (with-release [src-desc (memory-desc (shape src-desc) (or (tz/data-type src-desc) :float) :any)
                 dst-desc (memory-desc (shape dst-desc)
                                       (or (tz/data-type dst-desc) (data-type src-desc))
                                       :any)]
    (let-release [convolution-bluep (dnnl-convolution-op-blueprint fact eng src-desc weights-desc
                                                                   dst-desc strides dilation
                                                                   padding-l padding-r)
                  activ-bluep (dnnl-activ-blueprint fact eng
                                                    (inf-desc convolution-bluep)
                                                    (train-desc convolution-bluep)
                                                    (train-desc convolution-bluep)
                                                    activ alpha beta)]
      (->DirectedLayerBlueprint fact :convolution convolution-bluep activ-bluep))))

;; ================================ Pooling =============================================

(deftype DnnlPoolingInferenceLayer [fact strm bluep src-conn dst-tz
                                    pooling-fwd-prim pooling-fwd-args]
  Releaseable
  (release [_]
    (release src-conn)
    (release dst-tz)
    (release pooling-fwd-prim))
  Object
  (hashCode [_]
    (-> (hash :pooling)
        (hash-combine (shape (input src-conn)))
        (hash-combine (shape dst-tz))))
  (equals [_ other]
    (and (instance? DnnlPoolingInferenceLayer other)
         (= src-conn (.src-conn ^DnnlPoolingInferenceLayer other))
         (= dst-tz (.dst-tz ^DnnlPoolingInferenceLayer other))))
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
    (input src-conn))
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
    (src-conn)
    (execute! strm pooling-fwd-prim pooling-fwd-args)
    dst-tz)
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(defmethod print-method DnnlPoolingInferenceLayer
  [^DnnlPoolingInferenceLayer layer ^java.io.Writer w]
  (.write w (format "#Pooling[shape:%s, algo:%s]\n destination: %s\n"
                    (shape (output layer)) (info layer :algo) (pr-str (.dst-tz layer)))))

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
  Object
  (hashCode [_]
    (-> (hash :pooling)
        (hash-combine (shape (input src-conn)))
        (hash-combine (shape dst-tz))))
  (equals [_ other]
    (and (instance? DnnlPoolingTrainingLayer other)
         (= src-conn (.src-conn ^DnnlPoolingTrainingLayer other))
         (= dst-tz (.dst-tz ^DnnlPoolingTrainingLayer other))))
  (toString [this]
    (str bluep))
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
  ParametersSeq
  (parameters [_]
    [])
  Initializable
  (init [this _]
    this)
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

(defmethod print-method DnnlPoolingTrainingLayer
  [^DnnlPoolingTrainingLayer layer ^java.io.Writer w]
  (.write w (format "#Pooling[shape:%s, algo:%s]\n destination: %s\n"
                    (shape (output layer)) (info layer :algo) (pr-str (.dst-tz layer)))))

(deftype DnnlPoolingBlueprint [fact algo pool-infer-pd pool-train-pd pool-bwd-pd]
  Releaseable
  (release [_]
    (release pool-infer-pd)
    (release pool-train-pd)
    (release pool-bwd-pd))
  Object
  (hashCode [this]
    (-> (hash :pooling)
        (hash-combine algo)
        (hash-combine (train-desc this))))
  (equals [this other]
    (and (instance? DnnlPoolingBlueprint other)
         (= algo (.algo ^DnnlPoolingBlueprint other))
         (= (inf-desc this) (inf-desc other))
         (= (train-desc this) (train-desc other))))
  (toString [this]
    (str {:algo algo
          :shape (shape this)
          :topology :pooling}))
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
  (diff-desc [_]
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
                  dst-tz (dnnl-tensor fact (dst-md pool-infer-pd) (batch-index src-tz))
                  pool-infer-prim (primitive pool-infer-pd)
                  pool-infer-args (fwd-args (output src-conn) dst-tz)]
      (->DnnlPoolingInferenceLayer fact (flow fact) this src-conn dst-tz
                                   pool-infer-prim pool-infer-args)))
  (invoke [this prev-layer prop-diff? _]
    (let-release [src-tz (output prev-layer)
                  src-conn (connector src-tz (src-md pool-train-pd))
                  dst-tz (dnnl-tensor fact (dst-md pool-train-pd) (batch-index src-tz))
                  workspace-tz (when-let [workspace-desc (workspace-md pool-train-pd)]
                                 (dnnl-tensor fact workspace-desc))
                  pool-train-prim (primitive pool-train-pd)
                  pool-train-args (fwd-args (output src-conn) dst-tz workspace-tz)]
      (if prop-diff?
        (let-release [diff-dst-conn (connector dst-tz (diff-dst-md pool-bwd-pd))
                      diff-src-conn (connector (diff-src-md pool-bwd-pd) (diff-input prev-layer))
                      pool-bwd-prim (primitive pool-bwd-pd)
                      pool-bwd-args (pooling-bwd-args (output diff-dst-conn) (input diff-src-conn) workspace-tz)]
          (->DnnlPoolingTrainingLayer fact (flow fact) this src-conn dst-tz workspace-tz
                                      pool-train-prim pool-train-args
                                      diff-dst-conn diff-src-conn
                                      pool-bwd-prim pool-bwd-args))
        (->DnnlPoolingTrainingLayer fact (flow fact) this src-conn dst-tz workspace-tz
                                    pool-train-prim pool-train-args
                                    nil nil nil nil))))
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(defmethod print-method DnnlPoolingBlueprint
  [bp ^java.io.Writer w]
  (.write w (str bp)))

(defn dnnl-pooling-blueprint
  [fact eng src-desc dst-desc algo strides kernel padding-l padding-r]
  (let [inf-src-desc (inf-desc src-desc)
        train-src-desc (train-desc src-desc)]
    (with-release [inf-dst-desc (memory-desc (shape dst-desc)
                                             (or (tz/data-type dst-desc) (data-type inf-src-desc))
                                             (or (tz/layout dst-desc) :any))
                   train-dst-desc (memory-desc (shape dst-desc)
                                               (or (tz/data-type dst-desc) (data-type train-src-desc))
                                               (or (tz/layout dst-desc) :any))]
      (let-release [pool-infer-pd (pooling-fwd eng :inference algo inf-src-desc inf-dst-desc
                                               kernel strides padding-l padding-r)
                    pool-train-pd (pooling-fwd eng :training algo train-src-desc train-dst-desc
                                               kernel strides padding-l padding-r)
                    pool-bwd-pd (pooling-bwd eng pool-train-pd algo train-src-desc train-dst-desc
                                             kernel strides padding-l padding-r)]
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

(deftype DnnlBatchNormalizationInference [fact strm bluep
                                          src-conn gamma-tz beta-tz mean-tz var-tz
                                          fwd-prim fwd-args]
  Releaseable
  (release [_]
    (release src-conn)
    (release gamma-tz)
    (release beta-tz)
    (release mean-tz)
    (release var-tz)
    (release fwd-prim))
  Object
  (hashCode [_]
    (-> (hash :batch-norm)
        (hash-combine (shape gamma-tz))
        (hash-combine (shape beta-tz))
        (hash-combine (shape (input src-conn)))
        (hash-combine (shape (output src-conn)))))
  (equals [_ other]
    (and (instance? DnnlBatchNormalizationInference other)
         (let [other ^DnnlBatchNormalizationInference other]
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
    (execute! strm fwd-prim fwd-args)
    (output src-conn))
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(defmethod print-method DnnlBatchNormalizationInference
  [layer ^java.io.Writer w]
  (.write w (format "#BatchNorm[]\n gamma:\n beta: \n output: %s\n"
                    (weights layer) (bias layer) (pr-str (output layer)))))

(deftype DnnlBatchNormalizationTraining [fact strm bluep
                                         src-conn gamma-tz beta-tz dst-tz mean-tz var-tz
                                         diff-gamma-tz post-diff-gamma-tz
                                         diff-src-conn fwd-prim fwd-args bwd-prim bwd-args]
  Releaseable
  (release [_]
    (release src-conn)
    (release dst-tz)
    (release mean-tz)
    (release var-tz)
    (release gamma-tz)
    (release beta-tz)
    (release diff-gamma-tz)
    (release post-diff-gamma-tz)
    (release diff-src-conn)
    (release fwd-prim)
    (release fwd-args)
    (release bwd-prim)
    (release bwd-args))
  Object
  (hashCode [_]
    (-> (hash :batch-norm)
        (hash-combine (shape gamma-tz))
        (hash-combine (shape beta-tz))
        (hash-combine (shape (input src-conn)))
        (hash-combine (shape dst-tz))))
  (equals [_ other]
    (and (instance? DnnlBatchNormalizationTraining other)
         (= gamma-tz (.gamma-tz ^DnnlBatchNormalizationTraining other))
         (= beta-tz (.beta-tz ^DnnlBatchNormalizationTraining other))
         (= src-conn (.src-conn ^DnnlBatchNormalizationTraining other))
         (= mean-tz (.mean-tz ^DnnlBatchNormalizationTraining other))
         (= var-tz (.var-tz ^DnnlBatchNormalizationTraining other))
         (= dst-tz (.dst-tz ^DnnlBatchNormalizationTraining other))))
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
    this)
  DiffParameters
  (diff-weights [_]
    post-diff-gamma-tz)
  IFn
  (invoke [this]
    (forward this)
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
    (when (not= 1.0 scal-b) (scal! scal-b beta-tz))
    (diff-src-conn)
    this))

(defmethod print-method DnnlBatchNormalizationTraining
  [layer ^java.io.Writer w]
  (.write w (format "#BatchNorm[]\n gamma:\n beta: \n output: %s\n"
                    (weights layer) (bias layer) (pr-str (output layer)))))

(deftype DnnlBatchNormalizationBlueprint [fact data-desc scaleshift-desc
                                          infer-pd train-pd bwd-pd ^long c]
  Releaseable
  (release [_]
    (release data-desc)
    (release scaleshift-desc)
    (release infer-pd)
    (release train-pd)
    (release bwd-pd))
  Object
  (hashCode [_]
    (-> (hash :batch-norm) (hash-combine scaleshift-desc)))
  (equals [_ other]
    (and (instance? DnnlBatchNormalizationBlueprint other)
         (let [other ^DnnlBatchNormalizationBlueprint other]
           (and
            (equal-desc? scaleshift-desc (.scaleshift-desc other))
            (equal-desc? (src-md infer-pd) (src-md (.infer-pd other)))
            (equal-desc? (src-md train-pd) (src-md (.train-pd other)))
            (equal-desc? (dst-md train-pd) (dst-md (.train-pd other)))))))
  (toString [this]
    (pr-str {:topology :batch-norm
             :shape (shape this)}))
  Info
  (info [this info-type]
    (case info-type
      :bias scaleshift-desc
      :inference {:src (src-md infer-pd)
                  :weights scaleshift-desc
                  :dst (dst-md infer-pd)}
      :training {:src (src-md train-pd)
                 :weights scaleshift-desc
                 :dst (dst-md train-pd)}
      nil))
  (info [this]
    {:bias scaleshift-desc
     :inference {:src (src-md infer-pd)
                 :weights scaleshift-desc
                 :dst (dst-md infer-pd)}
     :training {:src (src-md train-pd)
                :weights scaleshift-desc
                :dst (dst-md train-pd)}})
  DiamondFactoryProvider
  (diamond-factory [_]
    fact)
  DescriptorProvider
  (inf-desc [_]
    (dst-md infer-pd))
  (train-desc [_]
    (dst-md train-pd))
  (diff-desc [_]
    (dst-md train-pd))
  TensorDescriptor
  (shape [this]
    (dims (train-desc train-pd)))
  (data-type [this]
    (data-type (train-desc train-pd)))
  (layout [this]
    (strides (train-desc train-pd)))
  IFn
  (invoke [this src-tz]
    (let-release [src-conn (connector src-tz data-desc)
                  gamma-tz (dnnl-tensor fact scaleshift-desc)
                  beta-tz (dnnl-tensor fact scaleshift-desc)
                  mean-tz (dnnl-tensor fact scaleshift-desc)
                  var-tz (dnnl-tensor fact scaleshift-desc)
                  fwd-prim (primitive infer-pd)
                  fwd-args (batch-norm-fwd-args (output src-conn) (output src-conn)
                                                gamma-tz beta-tz mean-tz var-tz)]
      (->DnnlBatchNormalizationInference fact (flow fact) this
                                         src-conn gamma-tz beta-tz mean-tz var-tz
                                         fwd-prim fwd-args)))
  (invoke [this src-tz diff-src-tz prop-diff? post-process-diff?]
    (let-release [src-conn (connector src-tz data-desc)
                  dst-tz (dnnl-tensor fact (dst-md train-pd) (batch-index src-tz))
                  gamma-tz (dnnl-tensor fact scaleshift-desc)
                  beta-tz (dnnl-tensor fact scaleshift-desc)
                  mean-tz (dnnl-tensor fact scaleshift-desc)
                  var-tz (dnnl-tensor fact scaleshift-desc)
                  diff-gamma-tz (create-tensor fact scaleshift-desc true)
                  post-diff-gamma-tz (if post-process-diff? (create-tensor fact scaleshift-desc true)
                                         diff-gamma-tz)
                  diff-src-conn (if prop-diff?
                                  (connector data-desc diff-src-tz)
                                  (dnnl-tensor fact (src-md bwd-pd) (batch-index diff-src-tz)))
                  fwd-prim (primitive train-pd)
                  fwd-args (batch-norm-fwd-args (output src-conn) dst-tz gamma-tz beta-tz mean-tz var-tz)
                  bwd-prim (primitive bwd-pd)
                  bwd-args (batch-norm-bwd-args dst-tz (output src-conn) gamma-tz beta-tz
                                                mean-tz var-tz (input diff-src-conn) diff-gamma-tz)]
      (->DnnlBatchNormalizationTraining fact (flow fact) this
                                        src-conn gamma-tz beta-tz dst-tz mean-tz var-tz
                                        diff-gamma-tz  post-diff-gamma-tz diff-src-conn
                                        fwd-prim fwd-args
                                        bwd-prim bwd-args)))
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(defmethod print-method DnnlBatchNormalizationBlueprint
  [bp ^java.io.Writer w]
  (.write w (str bp)))

(defn dnnl-batch-norm-op-blueprint [fact eng data-desc]
  (let-release [data-desc (desc data-desc)
                c (get (dims data-desc) 1)
                scaleshift-desc (memory-desc [1 c] :float :ab)
                bnorm-infer-pd (batch-norm-fwd eng :inference data-desc :scale :shift :global-stats)
                bnorm-train-pd (batch-norm-fwd eng :training data-desc :scale :shift)
                bnorm-bwd-pd (batch-norm-bwd eng bnorm-train-pd :backward data-desc data-desc :scale)]
    (->DnnlBatchNormalizationBlueprint fact data-desc scaleshift-desc
                                       bnorm-infer-pd bnorm-train-pd bnorm-bwd-pd c)))

(defn dnnl-batch-norm-layer-blueprint [fact eng src-desc activ alpha beta]
  (let-release [data-desc (dnnl-contiguous-desc (desc src-desc))
                batch-norm-op-bluep (dnnl-batch-norm-op-blueprint fact eng data-desc)
                activ-bluep (dnnl-activ-blueprint fact eng (inf-desc batch-norm-op-bluep)
                                                  (train-desc batch-norm-op-bluep)
                                                  (train-desc batch-norm-op-bluep)
                                                  activ alpha beta)]
    (->DirectedLayerBlueprint fact :batch-normalization batch-norm-op-bluep activ-bluep)))

(defmethod transfer! [DnnlBatchNormalizationInference Object]
  [source destination]
  (doall (map transfer! (parameters source) (parameters destination)))
  destination)

(defmethod transfer! [DnnlBatchNormalizationTraining Object]
  [source destination]
  (doall (map transfer! (parameters source) (parameters destination)))
  destination)

;; ================================ Branch ======================================

(deftype DnnlBranchInference [fact strm bluep src-tz dst-tzs branch-prims branch-args]
  Releaseable
  (release [_]
    (doseq [sp branch-prims] (release sp))
    (doseq [dt dst-tzs] (release dt))
    true)
  Object
  (hashCode [_]
    (reduce #(hash-combine %1 (shape %2))
            (hash-combine (hash :branch) (shape src-tz))
            dst-tzs))
  (equals [_ other]
    (and (instance? DnnlBranchInference other)
         (= src-tz (.src-tz ^DnnlBranchInference other))
         (= dst-tzs (.dst-tzs ^DnnlBranchInference other))))
  (toString [this]
    (str bluep))
  Info
  (info [this]
    {:dst (map info dst-tzs)
     :src (info src-tz)})
  (info [this info-type]
    (case info-type
      :dst (map info dst-tzs)
      :src (info src-tz)
      nil))
  DiamondFactoryProvider
  (diamond-factory [_]
    fact)
  Transfer
  (input [_]
    src-tz)
  (output [_]
    dst-tzs)
  ParametersSeq
  (parameters [_]
    [])
  Initializable
  (init [this _]
    this)
  IFn
  (invoke [_]
    (doall (map (partial execute! strm) branch-prims branch-args))
    dst-tzs)
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(defmethod print-method DnnlBranchInference
  [layer ^java.io.Writer w]
  (.write w (format "#Branch[src:%s, dst:%s]" (input layer) (output layer))))

(deftype DnnlBranchTraining [fact strm bluep src-tz dst-tzs diff-src-conn
                             concat-prim concat-args
                             branch-prims branch-args
                             prop-diff?]
  Releaseable
  (release [_]
    (release src-tz)
    (doseq [dt dst-tzs] (release dt))
    (doseq [sp branch-prims] (release sp))
    (release concat-prim))
  Object
  (hashCode [_]
    (reduce #(hash-combine %1 (shape %2))
            (hash-combine (hash :branch) (shape src-tz))
            dst-tzs))
  (equals [_ other]
    (and (instance? DnnlBranchTraining other)
         (= src-tz (.src-tz ^DnnlBranchTraining other))
         (= dst-tzs (.dst-tzs ^DnnlBranchTraining other))))
  (toString [this]
    (str bluep))
  Info
  (info [this]
    {:dst (map info dst-tzs)
     :src (info src-tz)})
  (info [this info-type]
    (case info-type
      :dst (map info dst-tzs)
      :src (info src-tz)
      nil))
  DiamondFactoryProvider
  (diamond-factory [_]
    fact)
  Transfer
  (input [_]
    src-tz)
  (output [_]
    dst-tzs)
  DiffTransfer
  (diff-input [_]
    dst-tzs)
  (diff-output [_]
    (output diff-src-conn))
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
    (doall (map (partial execute! strm) branch-prims branch-args))
    this)
  (backward [this]
    this)
  (backward [this _]
    (when prop-diff?
      (execute! strm concat-prim concat-args)
      (diff-src-conn))
    this)
  IFn
  (invoke [_]
    (doall (map (partial execute! strm) branch-prims branch-args))
    dst-tzs)
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(defmethod print-method DnnlBranchTraining
  [layer ^java.io.Writer w]
  (.write w (format "#Branch[src:%s, dst:%s]" (input layer) (output layer))))

(deftype DnnlBranchBlueprint [fact src-desc dst-descs concat-pd branch-pds]
  Releaseable
  (release [_]
    (release concat-pd)
    (doseq [sp branch-pds] (release sp))
    (doseq [dd dst-descs] (release dd))
    (release src-desc))
  Object
  (hashCode [_]
    (hash-combine (reduce hash-combine (hash :branch) dst-descs) src-desc))
  (equals [_ other]
    (and (instance? DnnlBranchBlueprint other)
         (every? identity (map equal-desc? dst-descs (.dst-descs ^DnnlBranchBlueprint other)))
         (equal-desc? src-desc (.src-desc ^DnnlBranchBlueprint other))))
  (toString [this]
    (pr-str {:shape (shape this)
             :topology :branch}))
  DiamondFactoryProvider
  (diamond-factory [_]
    fact)
  DescriptorProvider
  (inf-desc [_]
    dst-descs)
  (train-desc [_]
    dst-descs)
  (diff-desc [_]
    dst-descs)
  TensorDescriptor
  (shape [_]
    (fmap shape dst-descs))
  (data-type [_]
    (fmap data-type dst-descs))
  (layout [_]
    (fmap layout dst-descs))
  IFn
  (invoke [this prev-layer]
    (let [src-tz (output prev-layer)]
      (let-release [dst-tzs (fmap (partial dnnl-tensor fact) dst-descs)
                    branch-prims (mapv primitive branch-pds)
                    branch-args (mapv (partial fwd-args src-tz) dst-tzs)]
        (->DnnlBranchInference fact (flow fact) this src-tz dst-tzs branch-prims branch-args))))
  (invoke [this prev-layer prop-diff? _]
    (let [src-tz (output prev-layer)
          diff-src-conn (connector src-desc (diff-input prev-layer))]
      (let-release [dst-tzs (fmap (partial dnnl-tensor fact) dst-descs)
                    branch-prims (mapv primitive branch-pds)
                    branch-args (mapv (partial fwd-args src-tz) dst-tzs)
                    concat-prim (primitive concat-pd)
                    concat-args (apply multi-args (input diff-src-conn) dst-tzs)]
        (->DnnlBranchTraining fact (flow fact) this src-tz dst-tzs diff-src-conn
                              concat-prim concat-args branch-prims branch-args
                              prop-diff?))))
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(defmethod print-method DnnlBranchBlueprint
  [bp ^java.io.Writer w]
  (.write w (str bp)))

(defn dnnl-branch-blueprint [fact eng src-desc branch-dim dst-descs]
  (let [dst-dims (map shape dst-descs)
        dst-descs (mapv (comp view desc) dst-descs)
        src-desc (desc src-desc)]
    (let-release [concat-pd (apply concatenate eng src-desc branch-dim dst-descs)
                  src-desc (dst-md concat-pd)]
      (with-release [src-subs (mapv (partial submemory-desc src-desc) ;;TODO I can either forward diff-desc, or, better just move diff related creatin stuff to (.invoke)
                                    dst-dims
                                    (concat-strides branch-dim dst-dims))]
        (let-release [branch-pds (mapv (partial reorder eng) src-subs dst-descs)]
          (->DnnlBranchBlueprint fact src-desc dst-descs concat-pd branch-pds))))))

(defmethod transfer! [DnnlBranchInference Object]
  [source destination]
  destination)

(defmethod transfer! [DnnlBranchTraining Object]
  [source destination]
  destination)

;; ================================ Concat ======================================

(deftype DnnlConcatInference [fact strm bluep src-tzs dst-tz concat-prim concat-args]
  Releaseable
  (release [_]
    (release concat-prim)
    (release dst-tz))
  Object
  (hashCode [_]
    (reduce #(hash-combine %1 (shape %2))
            (hash-combine (hash :concat) (shape dst-tz))
            src-tzs))
  (equals [_ other]
    (and (instance? DnnlConcatInference other)
         (= dst-tz (.dst-tz ^DnnlConcatInference other))
         (= src-tzs (.src-tzs ^DnnlConcatInference other))))
  (toString [this]
    (str bluep))
  Info
  (info [this]
    {:src (map info src-tzs)
     :dst (info dst-tz)})
  (info [this info-type]
    (case info-type
      :src (map info src-tzs)
      :dst (info dst-tz)
      nil))
  DiamondFactoryProvider
  (diamond-factory [_]
    fact)
  Transfer
  (input [_]
    src-tzs)
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
    (execute! strm concat-prim concat-args)
    dst-tz)
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(defmethod print-method DnnlConcatInference
  [layer ^java.io.Writer w]
  (.write w (format "#Concat[srcs:%s, dst:%s]" (input layer) (output layer))))

(deftype DnnlConcatTraining [fact strm bluep src-tzs dst-tz diff-src-cons
                             concat-prim concat-args
                             branch-prims branch-args
                             prop-diff?]
  Releaseable
  (release [_]
    (release concat-prim)
    (release dst-tz)
    (doseq [sp branch-prims] (release sp))
    true)
  Object
  (hashCode [_]
    (reduce #(hash-combine %1 (shape %2))
            (hash-combine (hash :concat) (shape dst-tz))
            src-tzs))
  (equals [_ other]
    (and (instance? DnnlConcatTraining other)
         (= src-tzs (.src-tzs ^DnnlConcatTraining other))
         (= dst-tz (.dst-tz ^DnnlConcatTraining other))))
  (toString [this]
    (str bluep))
  Info
  (info [this]
    {:src (map info src-tzs)
     :dst (info dst-tz)})
  (info [this info-type]
    (case info-type
      :src (map info src-tzs)
      :dst (info dst-tz)
      nil))
  DiamondFactoryProvider
  (diamond-factory [_]
    fact)
  Transfer
  (input [_]
    src-tzs)
  (output [_]
    dst-tz)
  DiffTransfer
  (diff-input [_]
    dst-tz)
  (diff-output [_]
    (map output diff-src-cons))
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
    (execute! strm concat-prim concat-args)
    this)
  (backward [this]
    this)
  (backward [this _]
     (when prop-diff?
       (doall (map (partial execute! strm) branch-prims branch-args))
       (doseq [diff-src-conn diff-src-cons]
        (diff-src-conn)))
    this)
  IFn
  (invoke [_]
    (execute! strm concat-prim concat-args)
    dst-tz)
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(defmethod print-method DnnlConcatTraining
  [layer ^java.io.Writer w]
  (.write w (format "#Concat[srcs:%s, dst:%s]" (input layer) (output layer))))

(deftype DnnlConcatBlueprint [fact src-descs dst-desc concat-pd branch-pds]
  Releaseable
  (release [_]
    (doseq [sp branch-pds] (release sp))
    (doseq [sd src-descs] (release sd))
    (release concat-pd)
    (release dst-desc))
  Object
  (hashCode [_]
    (hash-combine (reduce hash-combine (hash :concat) src-descs) dst-desc))
  (equals [_ other]
    (and (instance? DnnlConcatBlueprint other)
         (every? identity (map equal-desc? src-descs (.src-descs ^DnnlConcatBlueprint other)))
         (equal-desc? dst-desc (.dst-desc ^DnnlConcatBlueprint other))))
  (toString [this]
    (pr-str {:srcs (mapv shape src-descs)
             :shape (shape this)
             :topology :concat}))
  Info
  (info [this]
    {:src (map info src-descs)
     :dst (info dst-desc)})
  (info [this info-type]
    (case info-type
      :src (map info src-descs)
      :dst (info dst-desc)
      nil))
  DiamondFactoryProvider
  (diamond-factory [_]
    fact)
  DescriptorProvider
  (inf-desc [_]
    dst-desc)
  (train-desc [_]
    dst-desc)
  (diff-desc [_]
    dst-desc)
  TensorDescriptor
  (shape [_]
    (shape dst-desc))
  (data-type [_]
    (data-type dst-desc))
  (layout [_]
    (layout dst-desc))
  IFn
  (invoke [this prev-layer]
    (let [src-tzs (fmap (comp view output) prev-layer)]
      (let-release [dst-tz (dnnl-tensor fact dst-desc)
                    concat-prim (primitive concat-pd)
                    concat-args (apply multi-args dst-tz src-tzs)]
        (->DnnlConcatInference fact (flow fact) this src-tzs dst-tz concat-prim concat-args))))
  (invoke [this prev-layer prop-diff? _]
    (let [src-tzs (fmap (comp view output) prev-layer)
          diff-src-cons (fmap connector (fmap desc src-tzs) (fmap diff-input prev-layer))]
      (let-release [dst-tz (dnnl-tensor fact dst-desc)
                    concat-prim (primitive concat-pd)
                    concat-args (apply multi-args dst-tz src-tzs)
                    branch-prims (mapv primitive branch-pds)
                    branch-args (mapv (partial fwd-args dst-tz) (map input diff-src-cons))]
        (->DnnlConcatTraining fact (flow fact) this src-tzs dst-tz diff-src-cons
                              concat-prim concat-args branch-prims branch-args
                              prop-diff?))))
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(defmethod print-method DnnlConcatBlueprint
  [bp ^java.io.Writer w]
  (.write w (str bp)))

(defn dnnl-concat-blueprint
  [fact eng src-descs conc-dim dst-type]
  (let [src-dims (mapv shape src-descs)
        src-descs (mapv (comp view desc) src-descs)
        dst-type (or dst-type (tz/data-type (first src-descs)))
        dst-shape (concat-dst-shape conc-dim src-dims)]
    (let-release [dst-desc (memory-desc dst-shape dst-type :any)
                  concat-pd (apply concatenate eng dst-desc conc-dim src-descs)
                  dst-desc (dst-md concat-pd)]
      (with-release [dst-subs (mapv (partial submemory-desc dst-desc)
                                    src-dims
                                    (concat-strides conc-dim src-dims))]
        (let-release [branch-pds (mapv (partial reorder eng) dst-subs src-descs)]
          (->DnnlConcatBlueprint fact src-descs dst-desc concat-pd branch-pds))))))

(defmethod transfer! [DnnlConcatInference Object]
  [source destination]
  destination)

(defmethod transfer! [DnnlConcatTraining Object]
  [source destination]
  destination)

;; ============================ Split ====================================================

(deftype DnnlSplitInference [fact strm bluep n src-tz]
  Releaseable
  (release [_]
    (release src-tz))
  Object
  (hashCode [_]
    (-> (hash :split) (hash-combine n) (hash-combine src-tz)))
  (equals [_ other]
    (and (instance? DnnlSplitInference other)
         (= n (.n ^DnnlSplitInference other))))
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

(defmethod print-method DnnlSplitInference
  [^DnnlSplitInference layer ^java.io.Writer w]
  (.write w (format "#Split[n:%d, src:%s]" (.n layer) (input layer))))

(deftype DnnlSplitTraining [fact strm bluep n src-tz diff-tzs diff-src-tz sum-prim sum-args prop-diff?]
  Releaseable
  (release [_]
    (release src-tz)
    (release diff-src-tz)
    (doseq [dt diff-tzs] (release dt))
    (release sum-prim))
  Object
  (hashCode [_]
    (-> (hash :split) (hash-combine n) (hash-combine src-tz)))
  (equals [_ other]
    (and (instance? DnnlSplitTraining other)
         (= n (.n ^DnnlSplitTraining other))))
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
    diff-src-tz)
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
      (execute! strm sum-prim sum-args))
    this)
  IFn
  (invoke [this]
    (output this))
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(defmethod print-method DnnlSplitTraining
  [^DnnlSplitTraining layer ^java.io.Writer w]
  (.write w (format "#Split[n:%d, src:%s]" (.n layer) (input layer))))

(deftype DnnlSplitBlueprint [fact n src-desc sum-pd]
  Releaseable
  (release [_]
    (release sum-pd)
    (release src-desc))
  Object
  (hashCode [_]
    (-> (hash :split)
        (hash-combine n)
        (hash-combine src-desc)))
  (equals [_ other]
    (and (instance? DnnlSplitBlueprint other)
         (= n (.n ^DnnlSplitBlueprint other))
         (equal-desc? src-desc (.src-desc ^DnnlSplitBlueprint other))))
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
  (diff-desc [this]
    (train-desc this))
  TensorDescriptor
  (shape [_]
    (vec (repeat n (shape src-desc))))
  (data-type [_]
    (vec (repeat n (data-type src-desc))))
  (layout [_]
    (vec (repeat n (layout src-desc))))
  IFn
  (invoke [this prev-layer]
    (->DnnlSplitInference fact (flow fact) this n (view (output prev-layer))))
  (invoke [this prev-layer prop-diff? _]
    (let-release [src-tz (view (output prev-layer))
                  diff-src-tz (view (diff-output prev-layer))
                  diff-tzs (mapv (partial dnnl-tensor fact) (repeat n src-desc))
                  sum-prim (primitive sum-pd)
                  sum-args (apply multi-args diff-src-tz diff-tzs)]
      (->DnnlSplitTraining fact (flow fact) this n src-tz diff-tzs diff-src-tz
                           sum-prim sum-args prop-diff?)))
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(defmethod print-method DnnlSplitBlueprint
  [^DnnlSplitBlueprint layer ^java.io.Writer w]
  (.write w (format "#Split[n:%d, src:%s]" (.n layer) (input layer))))

(defn dnnl-split-blueprint [fact eng src-desc ^long n]
  (let-release [src-desc (view (desc src-desc))
                sum-pd (apply sum! eng src-desc (interleave (repeat (/ 1.0 n)) (repeat n src-desc)))]
    (->DnnlSplitBlueprint fact n src-desc sum-pd)))

(defmethod transfer! [DnnlSplitInference Object]
  [source destination]
  destination)

(defmethod transfer! [DnnlSplitTraining Object]
  [source destination]
  destination)

;; ================================ Sum ======================================

(deftype DnnlSum [fact strm bluep src-tzs diff-src-tzs
                  sum-prim sum-args diff-transformers prop-diff?]
  Releaseable
  (release [_]
    (doseq [src-tz src-tzs] (release src-tz))
    (doseq [diff-src-tz diff-src-tzs] (release diff-src-tz))
    (doseq [dt diff-transformers] (release dt))
    (release sum-prim))
  Object
  (hashCode [_]
    (reduce #(hash-combine %1 (shape %2)) (hash :sum) src-tzs))
  (equals [_ other]
    (and (instance? DnnlSum other) (= src-tzs (.src-tzs ^DnnlSum other))))
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
    (get src-tzs 0))
  DiffTransfer
  (diff-input [_]
    (get diff-src-tzs 0))
  (diff-output [_]
    diff-src-tzs)
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
    (execute! strm sum-prim sum-args)
    this)
  (backward [this]
    this)
  (backward [this _]
    (when prop-diff?
      (scal! (/ 1.0 (count diff-src-tzs)) (get diff-src-tzs 0))
      (doseq [diff-trans diff-transformers]
        (diff-trans)))
    this)
  IFn
  (invoke [this]
    (execute! strm sum-prim sum-args)
    (get src-tzs 0))
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(defmethod print-method DnnlSum
  [layer ^java.io.Writer w]
  (.write w (format "#Sum[srcs:%s]" (input layer))))

(deftype DnnlSumBlueprint [fact eng src-descs sum-pd]
  Releaseable
  (release [_]
    (doseq [sd src-descs]
      (release sd))
    (release sum-pd))
  Object
  (hashCode [_]
    (reduce #(hash-combine %1 (shape %2)) (hash :sum) src-descs))
  (equals [_ other]
    (and (instance? DnnlSumBlueprint other)
         (every? identity (map equal-desc? src-descs (.src-descs ^DnnlSumBlueprint other)))))
  (toString [this]
    (pr-str {:shape (shape this)
             :topology :sum}))
  DiamondFactoryProvider
  (diamond-factory [_]
    fact)
  DescriptorProvider
  (inf-desc [this]
    (train-desc this))
  (train-desc [_]
    (dst-md sum-pd))
  (diff-desc [_]
    (dst-md sum-pd))
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
    (let [src-tzs (fmap (comp view output) prev-layer)
          diff-src-tzs (fmap (comp view diff-input) prev-layer)
          strm (flow fact)]
      (let-release [sum-prim (primitive sum-pd)
                    sum-args (apply multi-args (get src-tzs 0) src-tzs)
                    diff-transformers (mapv (partial dnnl-transformer eng strm (get diff-src-tzs 0))
                                            (rest diff-src-tzs))]
        (->DnnlSum fact (flow fact) this src-tzs diff-src-tzs
                   sum-prim sum-args diff-transformers prop-diff?))))
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(defmethod print-method DnnlSum
  [bp ^java.io.Writer w]
  (.write w (str bp)))

(defn dnnl-sum-blueprint [fact eng src-descs]
  (let-release [src-descs (mapv (comp view desc) src-descs)
                n (count src-descs)
                sum-pd (apply sum! eng (first src-descs) (interleave (repeat 1.0) src-descs))]
    (->DnnlSumBlueprint fact eng src-descs sum-pd)))

(defmethod transfer! [DnnlSum Object]
  [source destination]
  destination)
