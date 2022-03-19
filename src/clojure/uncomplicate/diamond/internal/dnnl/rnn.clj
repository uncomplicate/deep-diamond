;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.diamond.internal.dnnl.rnn ;; TODO remove unused
  (:require [uncomplicate.commons.core :refer [Releaseable release let-release with-release Info info view]]
            [uncomplicate.neanderthal
             [core :refer [axpby! dim transfer! scal! view-vctr]]
             [block :refer [buffer]]]
            [uncomplicate.neanderthal.internal.api :refer [flow]]
            [uncomplicate.diamond.tensor :as tz
             :refer [Transfer input output connector revert shape layout TensorDescriptor view-tz]]
            [uncomplicate.diamond.internal
             [protocols
              :refer [Parameters bias weights ParametersSeq parameters DescriptorProvider
                      DiamondFactoryProvider DiffParameters diff-weights Backprop forward backward
                      DiffTransfer diff-input diff-output diff-z LinearBackprop backward-diff
                      inf-desc train-desc Initializable init]]
             [utils :refer [transfer-weights-bias! concat-strides concat-dst-shape direction-count]]]
            [uncomplicate.diamond.internal.dnnl
             [protocols :refer :all]
             [core :refer :all :as dnnl]
             [tensor :refer [dnnl-tensor dnnl-transformer dnnl-args]]])
  (:import [clojure.lang IFn AFn]))

;; ================================ RNN ====================================================

(deftype DnnlRnnInference [strm bluep
                           src-conn bias-tz weights-tz dst-tz
                           src-iter-tz weights-iter-tz dst-iter-tz
                           fwd-prim fwd-args]
  Releaseable
  (release [_]
    (release src-conn)
    (release bias-tz)
    (release weights-tz)
    (release dst-tz)
    (release src-iter-tz)
    (release weights-iter-tz)
    (release dst-iter-tz)
    (release fwd-prim))
  Info
  (info [this]
    {:bias (info bias-tz)
     :weights (info weights-tz)
     :weights-iter (info weights-iter-tz)
     :dst (info dst-tz)})
  (info [this info-type]
    (case info-type
      :bias (info bias-tz)
      :weights (info weights-tz)
      :weights-iter (info weights-iter-tz)
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
    [weights-tz bias-tz weights-iter-tz])
  IFn
  (invoke [_]
    (src-conn)
    (execute! strm fwd-prim fwd-args)
    dst-tz)
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(deftype DnnlRnnBlueprint [fact weights-desc bias-desc
                           infer-pd train-pd bwd-pd]
  Object
  (hashCode [_]
    (-> (hash weights-desc) (hash-combine bias-desc)))
  (equals [_ other]
    (and (instance? DnnlRnnBlueprint other)
         (let [other-infer-pd (.infer-pd ^DnnlRnnBlueprint other);;TODO use similar let elsewhere.
               other-train-pd (.train-pd ^DnnlRnnBlueprint other)]
           (= bias-desc (.bias-desc ^DnnlRnnBlueprint other))
           (equal-desc? (src-md infer-pd) (src-md other-infer-pd))
           (equal-desc? (weights-md infer-pd) (weights-md other-infer-pd))
           (equal-desc? (dst-md infer-pd) (dst-md other-infer-pd))
           (equal-desc? (arg-md infer-pd :src-iter) (arg-md other-infer-pd :src-iter))
           (equal-desc? (arg-md infer-pd :weights-iter) (arg-md other-infer-pd :weights-iter))
           (equal-desc? (arg-md infer-pd :dst-iter) (arg-md other-infer-pd :dst-iter))
           (equal-desc? (src-md train-pd) (src-md other-train-pd))
           (equal-desc? (weights-md train-pd) (weights-md other-train-pd))
           (equal-desc? (dst-md train-pd) (dst-md other-train-pd))
           (equal-desc? (arg-md train-pd :src-iter) (arg-md other-train-pd :src-iter))
           (equal-desc? (arg-md train-pd :weights-iter) (arg-md other-train-pd :weights-iter))
           (equal-desc? (arg-md train-pd :dst-iter) (arg-md other-train-pd :dst-iter)))))
  (toString [this]
    (pr-str {:src (src-md infer-pd)
             :weights (weights-md infer-pd)
             :weights-iter (arg-md infer-pd :weights-iter)
             :dst (dst-md infer-pd)}))
  Releaseable
  (release [_]
    (release weights-desc)
    (release bias-desc)
    (release infer-pd)
    (release train-pd)
    (release bwd-pd))
  Info
  (info [this info-type]
    (case info-type
      :bias bias-desc
      :inference {:src (src-md infer-pd)
                  :weights (weights-md infer-pd)
                  :weights-iter (arg-md infer-pd :weights-iter)
                  :dst (dst-md infer-pd)}
      :training {:src (src-md train-pd)
                 :weights (weights-md train-pd)
                 :weights-iter (arg-md train-pd :weights-iter)
                 :dst (dst-md train-pd)}
      nil))
  (info [this]
    {:bias bias-desc
     :inference {:src (src-md infer-pd)
                 :weights (weights-md infer-pd)
                 :weights-iter (arg-md infer-pd :weights-iter)
                 :dst (dst-md infer-pd)}
     :training {:src (src-md train-pd)
                :weights (weights-md train-pd)
                :weights-iter (arg-md train-pd :weights-iter)
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
                  src-iter-tz (dnnl-tensor fact (arg-md infer-pd :src-iter))
                  weights-iter-tz (dnnl-tensor fact (arg-md infer-pd :weights-iter))
                  dst-iter-tz (dnnl-tensor fact (arg-md infer-pd :dst-iter))
                  workspace-tz (when-let [workspace-desc (arg-md infer-pd :workspace)]
                                 (dnnl-tensor fact workspace-desc))
                  fwd-prim (primitive infer-pd)
                  fwd-args (dnnl-args args {:src-layer (output src-conn)
                                            :weights-layer weights-tz
                                            :bias bias-tz
                                            :dst-layer dst-tz
                                            :src-iter src-iter-tz
                                            :weights-iter weights-iter-tz
                                            :dst-iter dst-iter-tz
                                            :workspace workspace-tz})]
      (->DnnlRnnInference (flow fact) this
                          src-conn bias-tz weights-tz dst-tz
                          src-iter-tz weights-iter-tz dst-iter-tz
                          fwd-prim fwd-args)))
  (invoke [this src-tz dst-tz prop-diff? post-process-diff?]
    #_(let-release [src-conn (connector src-tz (src-md train-pd))
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
                    diff-weights-conn (connector (diff-weights-md bwd-weights-pd) diff-weights-tz)
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

(defn dnnl-rnn-blueprint [fact eng src-desc dst-desc weights-type activ dir lrs]
  (let [src-desc (desc src-desc)
        src-shape (shape src-desc)
        dst-desc (desc dst-desc)
        ch (last src-shape)
        dirs (direction-count dir)
        gts 1
        src-iter-shape (into [lrs dirs] (rest src-shape))
        dst-iter-shape (into [lrs dirs] (rest (shape dst-desc)))
        dst-type (data-type dst-desc)
        weights-shape [lrs dirs ch gts ch]
        weights-type (or weights-type (data-type src-desc) dst-type)
        bias-shape [lrs dirs gts ch]]
    (let-release [src-iter-desc (memory-desc src-iter-shape dst-type :any)
                  dst-iter-desc (memory-desc dst-iter-shape dst-type :any)
                  bias-desc (memory-desc bias-shape dst-type :ldgo)]
      (with-release [weights-desc-any (memory-desc weights-shape weights-type :any)
                     infer-desc (vanilla-rnn-fwd-desc :inference activ dir src-desc src-iter-desc
                                                      weights-desc-any weights-desc-any bias-desc
                                                      dst-desc dst-iter-desc)
                     train-desc (vanilla-rnn-fwd-desc :training activ dir src-desc src-iter-desc
                                                      weights-desc-any weights-desc-any bias-desc
                                                      dst-desc dst-iter-desc)
                     bwd-desc (vanilla-rnn-bwd-desc activ dir src-desc src-iter-desc
                                                    weights-desc-any weights-desc-any bias-desc
                                                    dst-desc dst-iter-desc
                                                    src-desc src-iter-desc
                                                    weights-desc-any weights-desc-any bias-desc
                                                    dst-desc dst-iter-desc)]
        (let-release [infer-pd (primitive-desc eng infer-desc)
                      train-pd (primitive-desc eng train-desc)
                      bwd-pd (primitive-desc eng bwd-desc train-pd)
                      weights-desc-export (dnnl-contiguous-desc (weights-md train-pd))]
          (->DnnlRnnBlueprint fact weights-desc-export bias-desc infer-pd train-pd bwd-pd))))))
