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
             [core :refer [axpby! dim transfer! scal! view-vctr entry!]]
             [block :refer [buffer]]]
            [uncomplicate.neanderthal.internal.api :refer [flow]]
            [uncomplicate.diamond.tensor :as tz
             :refer [Transfer input output connector revert shape layout TensorDescriptor view-tz
                     transformer]]
            [uncomplicate.diamond.internal
             [protocols
              :refer [Parameters bias weights ParametersSeq parameters DescriptorProvider
                      DiamondFactoryProvider DiffParameters diff-weights Backprop forward backward
                      DiffTransfer diff-input diff-output diff-z LinearBackprop backward-diff
                      inf-desc train-desc Initializable init RnnParameters DiffRnnParameters]]
             [utils :refer [transfer-weights-bias! concat-strides concat-dst-shape direction-count]]]
            [uncomplicate.diamond.internal.dnnl
             [protocols :refer :all]
             [core :refer :all :as dnnl]
             [tensor :refer [dnnl-tensor dnnl-transformer dnnl-args]]])
  (:import [clojure.lang IFn AFn]))

;; ================================ RNN ====================================================

(deftype DnnlRnnInference [strm bluep srcs dsts
                           src-conn src-iter-conn
                           bias-tz weights-tz weights-iter-tz dst-tz dst-iter-tz
                           fwd-prim fwd-args]
  Releaseable
  (release [_]
    (release src-conn)
    (release bias-tz)
    (release weights-tz)
    (release dst-tz)
    (release src-iter-conn)
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
    srcs)
  (output [_]
    dsts)
  Parameters
  (bias [_]
    bias-tz)
  (weights [_]
    weights-tz)
  RnnParameters
  (weights-iter [_]
    weights-iter-tz)
  ParametersSeq
  (parameters [_]
    [weights-tz bias-tz weights-iter-tz])
  IFn
  (invoke [_]
    (src-conn)
    (when src-iter-conn (src-iter-conn))
    (execute! strm fwd-prim fwd-args)
    dsts)
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(deftype DnnlRnnTraining [strm bluep srcs dsts
                          bias-tz weights-tz weights-iter-tz dst-tz dst-iter-tz
                          diff-weights-tz post-diff-weights-tz
                          diff-weights-iter-tz post-diff-weights-iter-tz diff-bias-tz
                          src-conn src-iter-conn weights-conn weights-iter-conn
                          fwd-dst-tz dst-transformer fwd-dst-iter-tz dst-iter-transformer
                          fwd-prim fwd-args
                          bwd-src-conn bwd-src-iter-conn bwd-weights-conn bwd-weights-iter-conn
                          bwd-dst-conn bwd-dst-iter-conn
                          diff-dst-conn diff-dst-iter-conn
                          bwd-prim bwd-args
                          diff-src-conn diff-src-iter-conn diff-weights-conn diff-weights-iter-conn]
  Releaseable
  (release [_]
    (release bias-tz)
    (release weights-tz)
    (release dst-tz)
    (release weights-iter-tz)
    (release dst-iter-tz)
    (release diff-weights-tz)
    (release post-diff-weights-tz)
    (release diff-bias-tz)
    (release diff-weights-iter-tz)
    (release post-diff-weights-iter-tz)
    (release src-conn)
    (release src-iter-conn)
    (release fwd-dst-iter-tz)
    (release dst-transformer)
    (release fwd-dst-iter-tz)
    (release dst-iter-transformer)
    (release weights-conn)
    (release weights-iter-conn)
    (release fwd-prim)
    (release bwd-src-conn)
    (release bwd-src-iter-conn)
    (release bwd-weights-conn)
    (release bwd-weights-iter-conn)
    (release bwd-dst-conn)
    (release bwd-dst-iter-conn)
    (release diff-dst-conn)
    (release diff-dst-iter-conn)
    (release bwd-prim)
    (release bwd-args)
    (release diff-src-conn)
    (release diff-src-iter-conn)
    (release diff-weights-conn)
    (release diff-weights-iter-conn))
  Info
  (info [this]
    {:bias (info bias-tz)
     :weights (info weights-tz)
     :dst (info dst-tz)
     :diff-weights (info diff-weights-tz)
     :weights-iter (info weights-iter-tz)
     :diff-weights-iter (info diff-weights-iter-tz)})
  (info [this info-type]
    (case info-type
      :bias (info bias-tz)
      :weights (info weights-tz)
      :dst (info dst-tz)
      :diff-weights (info diff-weights-tz)
      :weights-iter (info weights-iter-tz)
      :diff-weights-iter (info diff-weights-iter-tz)
      nil))
  Transfer
  (input [_]
    srcs)
  (output [_]
    dsts)
  DiffTransfer
  (diff-input [_]
    dsts)
  (diff-output [_]
    srcs)
  Parameters
  (bias [_]
    bias-tz)
  (weights [_]
    weights-tz)
  RnnParameters
  (weights-iter [_]
    weights-iter-tz)
  ParametersSeq
  (parameters [_]
    [weights-tz bias-tz weights-iter-tz])
  DiffParameters
  (diff-weights [_]
    post-diff-weights-tz)
  DiffRnnParameters
  (diff-weights-iter [_]
    post-diff-weights-iter-tz)
  IFn
  (invoke [this]
    (forward this)
    dsts)
  (applyTo [this xs]
    (AFn/applyToHelper this xs))
  Backprop
  (forward [this]
    (src-conn)
    (when src-iter-conn (src-iter-conn))
    (weights-conn)
    (weights-iter-conn)
    (execute! strm fwd-prim fwd-args)
    (dst-transformer)
    (when dst-iter-transformer (dst-iter-transformer))
    this)
  (backward [this]
    (backward-diff this 1.0 0.0 1.0 0.0))
  LinearBackprop
  (backward-diff [this scal-diff-w scal-g scal-diff-b scal-b]
    (bwd-src-conn)
    (when bwd-src-iter-conn (bwd-src-iter-conn))
    (bwd-weights-conn)
    (bwd-weights-iter-conn)
    (bwd-dst-conn)
    (when bwd-dst-iter-conn (bwd-dst-iter-conn))
    (diff-dst-conn)
    (when diff-dst-iter-conn (diff-dst-iter-conn))
    (entry! (input diff-weights-conn) 0.0)
    (entry! (input diff-weights-iter-conn) 0.0)
    (entry! diff-bias-tz 0.0)
    (execute! strm bwd-prim bwd-args)
    (diff-weights-conn)
    (when diff-weights-iter-conn (diff-weights-iter-conn))
    (if (= 0.0 scal-g)
      (when-not (= 1.0 scal-diff-w)
        (scal! scal-diff-w diff-weights-tz)
        (scal! scal-diff-w diff-weights-iter-tz))
      (do (axpby! scal-diff-w diff-weights-tz scal-g post-diff-weights-tz)
          (axpby! scal-diff-w diff-weights-iter-tz scal-g post-diff-weights-iter-tz)))
    (axpby! scal-diff-b diff-bias-tz scal-b bias-tz)
    (when diff-src-conn (diff-src-conn))
    (when diff-src-iter-conn (diff-src-iter-conn))
    this))

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
  (invoke [this srcs]
    (let [[src-tz src-iter-tz] (if (sequential? srcs) srcs [srcs nil])]
      (let-release [src-conn (connector src-tz (src-md infer-pd))
                    bias-tz (dnnl-tensor fact bias-desc)
                    weights-tz (dnnl-tensor fact (weights-md infer-pd))
                    dst-tz (dnnl-tensor fact (dst-md infer-pd))
                    src-iter-conn (when-let [src-iter-desc (arg-md infer-pd :src-iter)]
                                    (connector src-iter-tz src-iter-desc))
                    weights-iter-tz (dnnl-tensor fact (arg-md infer-pd :weights-iter))
                    dst-iter-tz (when-let [dst-iter-desc (arg-md infer-pd :dst-iter)]
                                  (dnnl-tensor fact dst-iter-desc))
                    workspace-tz (when-let [workspace-desc (arg-md infer-pd :workspace)]
                                   (dnnl-tensor fact workspace-desc))
                    fwd-prim (primitive infer-pd)
                    fwd-args (dnnl-args args {:src-layer (output src-conn)
                                              :src-iter (when src-iter-conn (output src-iter-conn))
                                              :weights-layer weights-tz
                                              :weights-iter weights-iter-tz
                                              :bias bias-tz
                                              :dst-layer dst-tz
                                              :dst-iter dst-iter-tz
                                              :workspace workspace-tz})
                    srcs (if src-iter-conn [(input src-conn) (input src-iter-conn)] (input src-conn))
                    dsts (if dst-iter-tz [dst-tz dst-iter-tz] dst-tz)]
        (->DnnlRnnInference (flow fact) this srcs dsts
                            src-conn src-iter-conn bias-tz weights-tz weights-iter-tz
                            dst-tz dst-iter-tz
                            fwd-prim fwd-args))))
  (invoke [this srcs dsts _ post-process-diff?]
    (let [[src-tz src-iter-tz] (if (sequential? srcs) srcs [srcs nil])
          [dst-tz dst-iter-tz] (if (sequential? dsts) dsts [dsts nil])]
      (let-release [src-conn (connector src-tz (src-md train-pd))
                    src-iter-conn (when-let [src-iter-desc (arg-md train-pd :src-iter)]
                                    (connector src-iter-tz src-iter-desc))
                    bias-tz (dnnl-tensor fact bias-desc)
                    weights-tz (dnnl-tensor fact weights-desc)
                    weights-conn (connector weights-tz (weights-md train-pd))
                    weights-iter-tz (dnnl-tensor fact weights-desc)
                    weights-iter-conn (connector weights-iter-tz (arg-md train-pd :weights-iter))
                    fwd-dst-tz (dnnl-tensor fact (dst-md train-pd))
                    dst-transformer (transformer fwd-dst-tz dst-tz)
                    fwd-dst-iter-tz (when-let [dst-iter-desc (arg-md train-pd :dst-iter)]
                                      (dnnl-tensor fact dst-iter-desc))
                    dst-iter-transformer (when (and fwd-dst-iter-tz dst-iter-tz)
                                           (transformer fwd-dst-iter-tz dst-iter-tz))
                    workspace-tz (when-let [workspace-desc (arg-md train-pd :workspace)]
                                   (dnnl-tensor fact workspace-desc))
                    fwd-prim (primitive train-pd)
                    fwd-args (dnnl-args args {:src-layer (output src-conn)
                                              :src-iter (when src-iter-conn (output src-iter-conn))
                                              :weights-layer (output weights-conn)
                                              :weights-iter (output weights-iter-conn)
                                              :bias bias-tz
                                              :dst-layer fwd-dst-tz
                                              :dst-iter fwd-dst-iter-tz
                                              :workspace workspace-tz})
                    bwd-src-conn (connector src-conn (src-md bwd-pd))
                    bwd-src-iter-conn (when-let [src-iter-desc (arg-md bwd-pd :src-iter)]
                                        (connector src-iter-conn src-iter-desc))
                    bwd-weights-conn (connector weights-conn (arg-md bwd-pd :weights))
                    bwd-weights-iter-conn (connector weights-iter-conn (arg-md bwd-pd :weights-iter))
                    bwd-dst-conn (connector fwd-dst-tz (dst-md bwd-pd))
                    bwd-dst-iter-conn (when-let [dst-iter-desc (arg-md bwd-pd :dst-iter)]
                                        (connector fwd-dst-iter-tz dst-iter-desc))
                    diff-dst-conn (connector dst-tz (diff-dst-md bwd-pd))
                    diff-dst-iter-conn (when-let [diff-dst-iter-desc (arg-md bwd-pd :diff-dst-iter)]
                                         (connector dst-iter-tz diff-dst-iter-desc))
                    diff-src-conn (connector (diff-src-md bwd-pd) src-tz)
                    diff-src-iter-conn (when-let [diff-src-iter-desc (arg-md bwd-pd :diff-src-iter)]
                                         (connector diff-src-iter-desc src-iter-tz))
                    diff-weights-tz (dnnl-tensor fact weights-desc)
                    post-diff-weights-tz (if post-process-diff? (dnnl-tensor fact weights-desc)
                                             diff-weights-tz)
                    diff-weights-conn (connector (diff-weights-md bwd-pd) diff-weights-tz)
                    diff-weights-iter-tz (dnnl-tensor fact weights-desc)
                    post-diff-weights-iter-tz (if post-process-diff? (dnnl-tensor fact weights-desc)
                                                  diff-weights-iter-tz)
                    diff-weights-iter-conn (connector (arg-md bwd-pd :diff-weights-iter) diff-weights-iter-tz)
                    diff-bias-tz (dnnl-tensor fact bias-desc)
                    bwd-prim (primitive bwd-pd)
                    bwd-args (dnnl-args args {:src-layer (output bwd-src-conn)
                                              :src-iter (when bwd-src-iter-conn (output bwd-src-iter-conn))
                                              :weights-layer (output bwd-weights-conn)
                                              :weights-iter (output bwd-weights-iter-conn)
                                              :bias bias-tz
                                              :dst-layer (output bwd-dst-conn)
                                              :dst-iter (when bwd-dst-iter-conn (output bwd-dst-iter-conn))
                                              :diff-dst-layer (output diff-dst-conn)
                                              :diff-dst-iter (when diff-dst-iter-conn (output diff-dst-iter-conn))
                                              :workspace workspace-tz
                                              :diff-src-layer (input diff-src-conn)
                                              :diff-src-iter (when diff-src-iter-conn (input diff-src-iter-conn))
                                              :diff-weights-layer (input diff-weights-conn)
                                              :diff-weights-iter (input diff-weights-iter-conn)
                                              :diff-bias bias-tz})
                    srcs (if src-iter-conn [(input src-conn) (input src-iter-conn)] (input src-conn))
                    dsts (if dst-iter-tz [dst-tz dst-iter-tz] dst-tz)]
        (->DnnlRnnTraining (flow fact) this srcs dsts
                           bias-tz weights-tz weights-iter-tz dst-tz dst-iter-tz
                           diff-weights-tz post-diff-weights-tz
                           diff-weights-iter-tz post-diff-weights-iter-tz diff-bias-tz
                           src-conn src-iter-conn weights-conn weights-iter-conn
                           fwd-dst-tz dst-transformer fwd-dst-iter-tz dst-iter-transformer
                           fwd-prim fwd-args
                           bwd-src-conn bwd-src-iter-conn bwd-weights-conn bwd-weights-iter-conn
                           bwd-dst-conn bwd-dst-iter-conn
                           diff-dst-conn diff-dst-iter-conn
                           bwd-prim bwd-args
                           diff-src-conn diff-src-iter-conn diff-weights-conn diff-weights-iter-conn))))
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(defn dnnl-rnn-blueprint [fact eng src-desc dst-desc weights-type activ dir lrs src-iter? dst-iter?]
  (let [src-desc (desc src-desc)
        src-shape (shape src-desc)
        dst-desc (desc dst-desc)
        ch (long (last src-shape))
        dirs (direction-count dir)
        gts 1
        src-iter-shape (into [lrs dirs] (rest src-shape))
        dst-iter-shape (conj (into [lrs dirs] (butlast (rest (shape dst-desc))))
                             (if (= :bidirectional-concat activ) (* 2 ch) ch))
        dst-type (data-type dst-desc)
        weights-shape [lrs dirs ch gts ch]
        weights-type (or weights-type (data-type src-desc) dst-type)
        bias-shape [lrs dirs gts ch]]
    (let-release [src-iter-desc (when src-iter? (memory-desc src-iter-shape dst-type :any))
                  dst-iter-desc (when dst-iter? (memory-desc dst-iter-shape dst-type :any))
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
