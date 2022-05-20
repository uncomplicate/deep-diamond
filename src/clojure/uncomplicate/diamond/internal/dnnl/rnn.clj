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
             [block :refer [buffer initialize]]]
            [uncomplicate.neanderthal.internal.api :refer [flow]]
            [uncomplicate.diamond.tensor :as tz
             :refer [Transfer input output connector revert shape layout TensorDescriptor view-tz]]
            [uncomplicate.diamond.internal
             [protocols
              :refer [Parameters bias weights ParametersSeq parameters DescriptorProvider
                      DiamondFactoryProvider DiffParameters diff-weights Backprop forward backward
                      DiffTransfer diff-input diff-output diff-z LinearBackprop backward-diff
                      inf-desc train-desc Initializable init RnnParameters DiffRnnParameters
                      batch-index]]
             [utils :refer [default-strides transfer-weights-bias! concat-strides concat-dst-shape direction-count]]]
            [uncomplicate.diamond.internal.dnnl
             [protocols :refer :all]
             [core :refer :all :as dnnl]
             [tensor :refer [dnnl-tensor dnnl-transformer dnnl-args]]]
            [uncomplicate.diamond.internal.neanderthal.rnn :refer [->RnnLayerBlueprint]])
  (:import [clojure.lang IFn AFn]))

;; ================================ RNN ====================================================

(deftype DnnlRnnInference [strm bluep srcs dsts
                           src-conn src-iter-conn src-iter-c-conn
                           bias-tz weights-tz weights-iter-tz
                           dst-tz dst-iter-tz dst-iter-c-tz
                           fwd-prim fwd-args]
  Releaseable
  (release [_]
    (release src-conn)
    (release src-iter-conn)
    (release src-iter-c-conn)
    (release bias-tz)
    (release weights-tz)
    (release weights-iter-tz)
    (release dst-tz)
    (release dst-iter-tz)
    (release dst-iter-c-tz)
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
  Initializable
  (init [this init-fn]
    (init-fn weights-tz)
    (init-fn weights-iter-tz)
    (when src-iter-conn
      (let [src-iter-tz (input src-iter-conn)]
        (initialize src-iter-tz (data (buffer src-iter-tz)) 0.0)))
    (when src-iter-c-conn
      (let [src-iter-c-tz (input src-iter-c-conn)]
        (initialize src-iter-c-tz (data (buffer src-iter-c-tz)) 0.0)))
    this)
  IFn
  (invoke [_]
    (src-conn)
    (when src-iter-conn (src-iter-conn))
    (when src-iter-c-conn (src-iter-c-conn))
    (execute! strm fwd-prim fwd-args)
    dsts)
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(deftype DnnlRnnTraining [strm bluep srcs dsts diff-dsts
                          src-conn src-iter-conn src-iter-c-conn bias-tz
                          weights-tz weights-conn weights-iter-tz weights-iter-conn
                          dst-tz dst-iter-tz dst-iter-c-tz
                          workspace-tz fwd-prim fwd-args
                          bwd-src-conn bwd-src-iter-conn bwd-src-iter-c-conn
                          bwd-weights-conn bwd-weights-iter-conn
                          bwd-dst-conn bwd-dst-iter-conn bwd-dst-iter-c-conn
                          diff-dst-tz diff-dst-iter-tz diff-dst-iter-c-tz
                          diff-src-conn diff-src-iter-conn diff-src-iter-c-conn
                          diff-weights-tz post-diff-weights-tz diff-weights-conn
                          diff-weights-iter-tz post-diff-weights-iter-tz diff-weights-iter-conn
                          diff-bias-tz
                          bwd-prim bwd-args]
  Releaseable
  (release [_]
    (release src-conn)
    (release src-iter-conn)
    (release src-iter-c-conn)
    (release bias-tz)
    (release weights-tz)
    (release weights-conn)
    (release weights-iter-tz)
    (release weights-iter-conn)
    (release dst-tz)
    (release dst-iter-tz)
    (release dst-iter-c-tz)
    (release workspace-tz)
    (release fwd-prim)
    (release bwd-src-conn)
    (release bwd-src-iter-conn)
    (release bwd-src-iter-c-conn)
    (release bwd-weights-conn)
    (release bwd-weights-iter-conn)
    (release bwd-dst-conn)
    (release bwd-dst-iter-conn)
    (release bwd-dst-iter-c-conn)
    (release diff-dst-tz)
    (release diff-dst-iter-tz)
    (release diff-dst-iter-c-tz)
    (release diff-src-conn)
    (release diff-src-iter-conn)
    (release diff-src-iter-c-conn)
    (release diff-weights-tz)
    (release post-diff-weights-tz)
    (release diff-weights-conn)
    (release diff-weights-iter-tz)
    (release post-diff-weights-iter-tz)
    (release diff-weights-iter-conn)
    (release diff-bias-tz)
    (release bwd-prim)
    (release bwd-args))
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
    diff-dsts)
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
  Initializable
  (init [this init-fn]
    (init-fn weights-tz)
    (init-fn weights-iter-tz)
    (when src-iter-conn
      (let [src-iter-tz (input src-iter-conn)]
        (initialize src-iter-tz (data (buffer src-iter-tz)) 0.0)))
    (when src-iter-c-conn
      (let [src-iter-c-tz (input src-iter-c-conn)]
        (initialize src-iter-c-tz (data (buffer src-iter-c-tz)) 0.0)))
    this)
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
    (when src-iter-c-conn (src-iter-c-conn))
    (weights-conn)
    (weights-iter-conn)
    (execute! strm fwd-prim fwd-args)
    this)
  (backward [this]
    (backward-diff this 1.0 0.0 1.0 0.0))
  LinearBackprop
  (backward-diff [this scal-diff-w scal-g scal-diff-b scal-b]
    (bwd-src-conn)
    (when bwd-src-iter-conn (bwd-src-iter-conn))
    (when bwd-src-iter-c-conn (bwd-src-iter-c-conn))
    (bwd-weights-conn)
    (bwd-weights-iter-conn)
    (bwd-dst-conn)
    (when bwd-dst-iter-conn (bwd-dst-iter-conn))
    (when bwd-dst-iter-c-conn (bwd-dst-iter-c-conn))
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
    (when diff-src-iter-c-conn (diff-src-iter-c-conn))
    this))

(deftype DnnlRnnBlueprint [fact weights-desc weights-iter-desc bias-desc
                           infer-pd train-pd bwd-pd]
  Object
  (hashCode [_]
    (-> (hash weights-desc) (hash-combine weights-iter-desc) (hash-combine bias-desc)))
  (equals [_ other]
    (and (instance? DnnlRnnBlueprint other)
         (let [other-infer-pd (.infer-pd ^DnnlRnnBlueprint other)
               other-train-pd (.train-pd ^DnnlRnnBlueprint other)]
           (and
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
            (equal-desc? (arg-md train-pd :dst-iter) (arg-md other-train-pd :dst-iter))))))
  (toString [this]
    (pr-str {:src (src-md infer-pd)
             :weights (weights-md infer-pd)
             :weights-iter (arg-md infer-pd :weights-iter)
             :dst (dst-md infer-pd)}))
  Releaseable
  (release [_]
    (release weights-desc)
    (release weights-iter-desc)
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
  DiamondFactoryProvider
  (diamond-factory [_]
    fact)
  DescriptorProvider
  (inf-desc [_]
    (dst-md infer-pd))
  (train-desc [_]
    (dst-md train-pd))
  TensorDescriptor
  (shape [this]
    (dims (train-desc this)))
  (data-type [this]
    (data-type (train-desc this)))
  (layout [this]
    (strides (train-desc this)))
  IFn
  (invoke [this srcs]
    (let [[src-tz src-iter-tz src-iter-c-tz] (if (sequential? srcs) srcs [srcs nil nil])]
      (let-release [src-conn (connector src-tz (src-md infer-pd))
                    src-iter-conn (when-let [src-iter-desc (arg-md infer-pd :src-iter)]
                                    (connector src-iter-tz src-iter-desc))
                    src-iter-c-conn (when-let [src-iter-c-desc (arg-md infer-pd :src-iter-c)]
                                      (connector src-iter-c-tz src-iter-c-desc))
                    bias-tz (dnnl-tensor fact bias-desc)
                    weights-tz (dnnl-tensor fact (weights-md infer-pd))
                    weights-iter-tz (dnnl-tensor fact (arg-md infer-pd :weights-iter))
                    dst-tz (dnnl-tensor fact (dst-md infer-pd) 1)
                    dst-iter-tz (when-let [dst-iter-desc (arg-md infer-pd :dst-iter)]
                                  (dnnl-tensor fact dst-iter-desc))
                    dst-iter-c-tz (when-let [dst-iter-c-desc (arg-md infer-pd :dst-iter-c)]
                                    (dnnl-tensor fact dst-iter-c-desc))
                    workspace-tz (when-let [workspace-desc (arg-md infer-pd :workspace)]
                                   (dnnl-tensor fact workspace-desc))
                    fwd-prim (primitive infer-pd)
                    fwd-args (dnnl-args args {:src-layer (output src-conn)
                                              :src-iter (when src-iter-conn (output src-iter-conn))
                                              :src-iter-c (when src-iter-c-conn (output src-iter-c-conn))
                                              :weights-layer weights-tz
                                              :weights-iter weights-iter-tz
                                              :bias bias-tz
                                              :dst-layer dst-tz
                                              :dst-iter dst-iter-tz
                                              :dst-iter-c dst-iter-c-tz
                                              :workspace workspace-tz})
                    srcs (if src-iter-conn
                           (if src-iter-c-conn
                             [(input src-conn) (input src-iter-conn) (input src-iter-c-conn)]
                             [(input src-conn) (input src-iter-conn)])
                           (input src-conn))
                    dsts (if dst-iter-tz
                           (if dst-iter-c-tz
                             [dst-tz dst-iter-tz dst-iter-c-tz]
                             [dst-tz dst-iter-tz])
                           dst-tz)]
        (->DnnlRnnInference (flow fact) this srcs dsts
                            src-conn src-iter-conn src-iter-c-conn
                            bias-tz weights-tz weights-iter-tz
                            dst-tz dst-iter-tz dst-iter-c-tz
                            fwd-prim fwd-args))))
  (invoke [this srcs _ post-process-diff?]
    (let [[src-tz src-iter-tz src-iter-c-tz] (if (sequential? srcs) srcs [srcs nil nil])]
      (let-release [src-conn (connector src-tz (src-md train-pd))
                    src-iter-conn (when-let [src-iter-desc (arg-md train-pd :src-iter)]
                                    (connector src-iter-tz src-iter-desc))
                    src-iter-c-conn (when-let [src-iter-c-desc (arg-md infer-pd :src-iter-c)]
                                      (connector src-iter-c-tz src-iter-c-desc))
                    bias-tz (dnnl-tensor fact bias-desc)
                    weights-tz (dnnl-tensor fact weights-desc)
                    weights-conn (connector weights-tz (weights-md train-pd))
                    weights-iter-tz (dnnl-tensor fact weights-iter-desc)
                    weights-iter-conn (connector weights-iter-tz (arg-md train-pd :weights-iter))
                    dst-tz (dnnl-tensor fact (dst-md train-pd) 1)
                    dst-iter-tz (when-let [dst-iter-desc (arg-md train-pd :dst-iter)]
                                  (dnnl-tensor fact dst-iter-desc))
                    dst-iter-c-tz (when-let [dst-iter-c-desc (arg-md infer-pd :dst-iter-c)]
                                    (dnnl-tensor fact dst-iter-c-desc))
                    workspace-tz (when-let [workspace-desc (arg-md train-pd :workspace)]
                                   (dnnl-tensor fact workspace-desc))
                    fwd-prim (primitive train-pd)
                    fwd-args (dnnl-args args {:src-layer (output src-conn)
                                              :src-iter (when src-iter-conn (output src-iter-conn))
                                              :src-iter-c (when src-iter-c-conn (output src-iter-c-conn))
                                              :weights-layer (output weights-conn)
                                              :weights-iter (output weights-iter-conn)
                                              :bias bias-tz
                                              :dst-layer dst-tz
                                              :dst-iter dst-iter-tz
                                              :dst-iter-c dst-iter-c-tz
                                              :workspace workspace-tz})
                    bwd-src-conn (connector src-conn (src-md bwd-pd))
                    bwd-src-iter-conn (when-let [src-iter-desc (arg-md bwd-pd :src-iter)]
                                        (connector src-iter-conn src-iter-desc))
                    bwd-src-iter-c-conn (when-let [src-iter-c-desc (arg-md bwd-pd :src-iter-c)]
                                          (connector src-iter-c-conn src-iter-c-desc))
                    bwd-weights-conn (connector weights-conn (arg-md bwd-pd :weights))
                    bwd-weights-iter-conn (connector weights-iter-conn (arg-md bwd-pd :weights-iter))
                    bwd-dst-conn (connector dst-tz (dst-md bwd-pd))
                    bwd-dst-iter-conn (when-let [dst-iter-desc (arg-md bwd-pd :dst-iter)]
                                        (connector dst-iter-tz dst-iter-desc))
                    bwd-dst-iter-c-conn (when-let [dst-iter-c-desc (arg-md bwd-pd :dst-iter-c)]
                                          (connector dst-iter-c-tz dst-iter-c-desc))
                    diff-dst-tz (dnnl-tensor fact (diff-dst-md bwd-pd) 1)
                    diff-dst-iter-tz (when-let [diff-dst-iter-desc (arg-md bwd-pd :diff-dst-iter)]
                                       (dnnl-tensor fact diff-dst-iter-desc))
                    diff-dst-iter-c-tz (when-let [diff-dst-iter-c-desc (arg-md bwd-pd :diff-dst-iter-c)]
                                         (dnnl-tensor fact diff-dst-iter-c-desc))
                    diff-weights-tz (dnnl-tensor fact weights-desc)
                    post-diff-weights-tz (if post-process-diff? (dnnl-tensor fact weights-desc)
                                             diff-weights-tz)
                    diff-weights-conn (connector (diff-weights-md bwd-pd) diff-weights-tz)
                    diff-weights-iter-tz (dnnl-tensor fact weights-iter-desc)
                    post-diff-weights-iter-tz (if post-process-diff? (dnnl-tensor fact weights-iter-desc)
                                                  diff-weights-iter-tz)
                    diff-weights-iter-conn (connector (arg-md bwd-pd :diff-weights-iter) diff-weights-iter-tz)
                    diff-bias-tz (dnnl-tensor fact bias-desc)
                    diff-src-conn (connector (diff-src-md bwd-pd) src-tz)
                    diff-src-iter-conn (when-let [diff-src-iter-desc (arg-md bwd-pd :diff-src-iter)]
                                         (connector diff-src-iter-desc src-iter-tz))
                    diff-src-iter-c-conn (when-let [diff-src-iter-c-desc (arg-md bwd-pd :diff-src-iter-c)]
                                           (connector diff-src-iter-c-desc src-iter-c-tz))
                    bwd-prim (primitive bwd-pd)
                    bwd-args (dnnl-args args {:src-layer (output bwd-src-conn)
                                              :src-iter (when bwd-src-iter-conn (output bwd-src-iter-conn))
                                              :src-iter-c (when bwd-src-iter-c-conn (output bwd-src-iter-c-conn))
                                              :weights-layer (output bwd-weights-conn)
                                              :weights-iter (output bwd-weights-iter-conn)
                                              :bias bias-tz
                                              :dst-layer (output bwd-dst-conn)
                                              :dst-iter (when bwd-dst-iter-conn (output bwd-dst-iter-conn))
                                              :dst-iter-c (when bwd-dst-iter-c-conn (output bwd-dst-iter-c-conn))
                                              :diff-dst-layer diff-dst-tz
                                              :diff-dst-iter diff-dst-iter-tz
                                              :diff-dst-iter-c diff-dst-iter-c-tz
                                              :workspace workspace-tz
                                              :diff-src-layer (input diff-src-conn)
                                              :diff-src-iter (when diff-src-iter-conn (input diff-src-iter-conn))
                                              :diff-src-iter-c (when diff-src-iter-c-conn (input diff-src-iter-c-conn))
                                              :diff-weights-layer (input diff-weights-conn)
                                              :diff-weights-iter (input diff-weights-iter-conn)
                                              :diff-bias diff-bias-tz})
                    srcs (if src-iter-conn
                           (if src-iter-c-conn
                             [(input src-conn) (input src-iter-conn) (input src-iter-c-conn)]
                             [(input src-conn) (input src-iter-conn)])
                           (input src-conn))
                    dsts (if dst-iter-tz
                           (if dst-iter-c-tz
                             [dst-tz dst-iter-tz dst-iter-c-tz]
                             [dst-tz dst-iter-tz])
                           dst-tz)
                    diff-dsts (if diff-dst-iter-tz
                                (if diff-dst-iter-c-tz
                                  [diff-dst-tz diff-dst-iter-tz diff-dst-iter-c-tz]
                                  [diff-dst-tz diff-dst-iter-tz])
                                diff-dst-tz)]
        (->DnnlRnnTraining (flow fact) this srcs dsts diff-dsts
                           src-conn src-iter-conn src-iter-c-conn bias-tz
                           weights-tz weights-conn weights-iter-tz weights-iter-conn
                           dst-tz dst-iter-tz dst-iter-c-tz
                           workspace-tz
                           fwd-prim fwd-args
                           bwd-src-conn bwd-src-iter-conn bwd-src-iter-c-conn
                           bwd-weights-conn bwd-weights-iter-conn
                           bwd-dst-conn bwd-dst-iter-conn bwd-dst-iter-c-conn
                           diff-dst-tz diff-dst-iter-tz diff-dst-iter-c-tz
                           diff-src-conn diff-src-iter-conn diff-src-iter-c-conn
                           diff-weights-tz post-diff-weights-tz diff-weights-conn
                           diff-weights-iter-tz post-diff-weights-iter-tz diff-weights-iter-conn
                           diff-bias-tz
                           bwd-prim bwd-args))))
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

;; ================================ Vanilla RNN =================================================

(defn dnnl-rnn-op-blueprint [fact eng src-desc dst-desc weights-type activ alpha beta
                             dir lrs src-iter? dst-iter?]
  (let [src-desc (desc src-desc)
        src-shape (shape src-desc)
        src-ch (long (last src-shape))
        dst-desc (desc dst-desc)
        dst-ch (long (last (shape dst-desc)))
        dirs (direction-count dir)
        gts 1
        dst-iter-shape (conj (into [lrs dirs] (butlast (rest (shape dst-desc))))
                             (if (= :bidirectional-concat dir) (quot dst-ch 2) dst-ch))
        src-iter-shape dst-iter-shape
        dst-type (data-type dst-desc)
        weights-shape [lrs dirs src-ch gts dst-ch]
        weights-type (or weights-type (data-type src-desc) dst-type)
        weights-iter-shape [lrs dirs dst-ch gts dst-ch]
        bias-shape [lrs dirs gts dst-ch]]
    (let-release [src-iter-desc (when src-iter? (memory-desc src-iter-shape dst-type :any))
                  dst-iter-desc (when dst-iter? (memory-desc dst-iter-shape dst-type :any))
                  bias-desc (memory-desc bias-shape dst-type :ldgo)]
      (with-release [weights-desc-any (memory-desc weights-shape weights-type :any)
                     weights-iter-desc-any (memory-desc weights-iter-shape weights-type :any)
                     infer-desc (vanilla-rnn-fwd-desc :inference activ dir src-desc src-iter-desc
                                                      weights-desc-any weights-iter-desc-any bias-desc
                                                      dst-desc dst-iter-desc alpha beta)
                     train-desc (vanilla-rnn-fwd-desc :training activ dir src-desc src-iter-desc
                                                      weights-desc-any weights-iter-desc-any bias-desc
                                                      dst-desc dst-iter-desc alpha beta)
                     bwd-desc (vanilla-rnn-bwd-desc activ dir src-desc src-iter-desc
                                                    weights-desc-any weights-iter-desc-any bias-desc
                                                    dst-desc dst-iter-desc
                                                    src-desc src-iter-desc
                                                    weights-desc-any weights-iter-desc-any bias-desc
                                                    dst-desc dst-iter-desc alpha beta)]
        (let-release [infer-pd (primitive-desc eng infer-desc)
                      train-pd (primitive-desc eng train-desc)
                      bwd-pd (primitive-desc eng bwd-desc train-pd)
                      weights-desc-export (dnnl-contiguous-desc (weights-md train-pd))
                      weights-iter-desc-export (dnnl-contiguous-desc (arg-md train-pd :weights-iter))]
          (->DnnlRnnBlueprint fact weights-desc-export weights-iter-desc-export bias-desc
                              infer-pd train-pd bwd-pd))))))

(defn dnnl-rnn-blueprint [fact eng src-desc dst-desc lrs activ alpha beta weights-type src-iter? dst-iter?]
  (with-release [src-desc (memory-desc (shape src-desc) (or (tz/data-type src-desc) :float) :any)
                 dst-desc (memory-desc (shape dst-desc)
                                       (or (tz/data-type dst-desc) (tz/data-type src-desc))
                                       :any)]
    (let-release [rnn-op-bluep (dnnl-rnn-op-blueprint fact eng src-desc dst-desc weights-type
                                                      activ alpha beta :unidirectional lrs
                                                      src-iter? dst-iter?)]
      (->RnnLayerBlueprint fact :rnn rnn-op-bluep))))

;; ================================ LSTM =======================================================

(defn dnnl-gated-op-blueprint [fwd-desc-fn bwd-desc-fn gts
                               fact eng src-desc dst-desc weights-type
                               dir lrs src-iter? dst-iter?]
  (let [src-desc (desc src-desc)
        src-shape (shape src-desc)
        src-ch (long (last src-shape))
        dst-desc (desc dst-desc)
        dst-ch (long (last (shape dst-desc)))
        dirs (direction-count dir)
        dst-iter-shape (conj (into [lrs dirs] (butlast (rest (shape dst-desc))))
                             (if (= :bidirectional-concat dir) (quot dst-ch 2) dst-ch))
        src-iter-shape dst-iter-shape
        dst-type (data-type dst-desc)
        weights-shape [lrs dirs src-ch gts dst-ch]
        weights-type (or weights-type (data-type src-desc) dst-type)
        weights-iter-shape [lrs dirs dst-ch gts dst-ch]
        bias-shape [lrs dirs gts dst-ch]]
    (let-release [src-iter-desc (when src-iter? (memory-desc src-iter-shape dst-type :any))
                  dst-iter-desc (when dst-iter? (memory-desc dst-iter-shape dst-type :any))
                  bias-desc (memory-desc bias-shape dst-type :ldgo)]
      (with-release [weights-desc-any (memory-desc weights-shape weights-type :any)
                     weights-iter-desc-any (memory-desc weights-iter-shape weights-type :any)
                     infer-desc (fwd-desc-fn :inference dir src-desc src-iter-desc
                                             weights-desc-any weights-iter-desc-any bias-desc
                                             dst-desc dst-iter-desc)
                     train-desc (fwd-desc-fn :training dir src-desc src-iter-desc
                                             weights-desc-any weights-iter-desc-any bias-desc
                                             dst-desc dst-iter-desc)
                     bwd-desc (bwd-desc-fn dir src-desc src-iter-desc
                                           weights-desc-any weights-iter-desc-any bias-desc
                                           dst-desc dst-iter-desc
                                           src-desc src-iter-desc
                                           weights-desc-any weights-iter-desc-any bias-desc
                                           dst-desc dst-iter-desc)]
        (let-release [infer-pd (primitive-desc eng infer-desc)
                      train-pd (primitive-desc eng train-desc)
                      bwd-pd (primitive-desc eng bwd-desc train-pd)
                      weights-desc-export (dnnl-contiguous-desc (weights-md train-pd))
                      weights-iter-desc-export (dnnl-contiguous-desc (arg-md train-pd :weights-iter))]
          (->DnnlRnnBlueprint fact weights-desc-export weights-iter-desc-export bias-desc
                              infer-pd train-pd bwd-pd))))))

(def dnnl-lstm-op-blueprint (partial dnnl-gated-op-blueprint lstm-fwd-desc lstm-bwd-desc 4))

(defn dnnl-lstm-blueprint [fact eng src-desc dst-desc lrs weights-type src-iter? dst-iter?]
  (with-release [src-desc (memory-desc (shape src-desc) (or (tz/data-type src-desc) :float) :any)
                 dst-desc (memory-desc (shape dst-desc)
                                       (or (tz/data-type dst-desc) (tz/data-type src-desc))
                                       :any)]
    (let-release [lstm-op-bluep (dnnl-lstm-op-blueprint fact eng src-desc dst-desc weights-type
                                                        :unidirectional lrs src-iter? dst-iter?)]
      (->RnnLayerBlueprint fact :lstm lstm-op-bluep))))

;; ================================ GRU =======================================================

(def dnnl-gru-op-blueprint (partial dnnl-gated-op-blueprint gru-fwd-desc gru-bwd-desc 3))

(defn dnnl-gru-blueprint [fact eng src-desc dst-desc lrs weights-type src-iter? dst-iter?]
  (with-release [src-desc (memory-desc (shape src-desc) (or (tz/data-type src-desc) :float) :any)
                 dst-desc (memory-desc (shape dst-desc)
                                       (or (tz/data-type dst-desc) (tz/data-type src-desc))
                                       :any)]
    (let-release [gru-op-bluep (dnnl-gru-op-blueprint fact eng src-desc dst-desc weights-type
                                                      :unidirectional lrs src-iter? dst-iter?)]
      (->RnnLayerBlueprint fact :gru gru-op-bluep))))

;; ================================= Ending Layer ==============================

(deftype DnnlEnding [fact bluep transform-src dst-tz transform-diff]
  Releaseable
  (release [_]
    (release transform-src)
    (release dst-tz)
    (release transform-diff))
  Object
  (hashCode [_]
    (hash-combine (hash :ending) (shape dst-tz)))
  (equals [_ other]
    (and (instance? DnnlEnding other)
         (= transform-src (.transform-src ^DnnlEnding other))
         (= transform-diff (.transform-diff ^DnnlEnding other))
         (= dst-tz (.dst-tz ^DnnlEnding other))))
  (toString [this]
    (str bluep))
  Info
  (info [this]
    {:src transform-src
     :dst dst-tz
     :topology :ending})
  (info [this info-type]
    (case info-type
      :src transform-src
      :dst dst-tz
      :topology :ending
      (info bluep info-type)))
  DiamondFactoryProvider
  (diamond-factory [_]
    fact)
  Transfer
  (input [_]
    (input transform-src))
  (output [_]
    dst-tz)
  DiffTransfer
  (diff-input [_]
    dst-tz)
  (diff-output [_]
    (output transform-diff))
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
    (transform-src)
    this)
  (backward [this]
    this)
  (backward [this _]
    (transform-diff)
    this)
  IFn
  (invoke [this]
    (transform-src)
    dst-tz)
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(defmethod print-method DnnlEnding
  [layer ^java.io.Writer w]
  (.write w (format "#Ending[dst:%s]" (output layer))))

(deftype DnnlEndingBlueprint [fact eng src-desc dst-desc]
  Releaseable
  (release [_]
    (release src-desc)
    (release dst-desc))
  Object
  (hashCode [_]
    (-> (hash :ending)
        (hash-combine src-desc)
        (hash-combine dst-desc)))
  (equals [_ other]
    (and (instance? DnnlEndingBlueprint other)
         (equal-desc? src-desc (.src-desc ^DnnlEndingBlueprint other))
         (equal-desc? dst-desc (.dst-desc ^DnnlEndingBlueprint other))))
  (toString [this]
    (str {:shape (shape this)
          :topology :ending}))
  Info
  (info [this]
    {:shape (shape this)
     :topology :ending})
  (info [this info-type]
    (case info-type
      :shape (shape this)
      :topology :ending
      nil))
  DiamondFactoryProvider
  (diamond-factory [_]
    fact)
  DescriptorProvider
  (inf-desc [this]
    dst-desc)
  (train-desc [_]
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
    (let [src-tz (output prev-layer)]
      (let-release [src-sub (tz/offset! (view-tz src-tz (shape dst-desc))
                                        (* (dec (long (get (shape src-tz) 0)))
                                           (long (get (strides src-tz) 0))))
                    dst-tz (dnnl-tensor fact dst-desc)
                    transform-src (dnnl-transformer eng (flow fact) src-sub dst-tz)
                    transform-diff (dnnl-transformer eng (flow fact) dst-tz src-sub)]
        (->DnnlEnding fact this transform-src dst-tz transform-diff))))
  (invoke [this prev-layer _ _]
    (let [src-tz (output prev-layer)
          diff-tz (diff-input prev-layer)]
      (let-release [src-sub (tz/offset! (view-tz src-tz (shape dst-desc))
                                        (* (dec (long (get (shape src-tz) 0)))
                                           (long (get (strides src-tz) 0))))
                    diff-sub (tz/offset! (view-tz (diff-input prev-layer) (shape dst-desc))
                                         (* (dec (long (get (shape diff-tz) 0)))
                                            (long (get (strides diff-tz) 0))))
                    dst-tz (dnnl-tensor fact dst-desc)
                    transform-src (dnnl-transformer eng (flow fact) src-sub dst-tz)
                    transform-diff (dnnl-transformer eng (flow fact) dst-tz diff-sub)]
        (->DnnlEnding fact this transform-src dst-tz transform-diff))))
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(defmethod print-method DnnlEnding
  [bp ^java.io.Writer w]
  (.write w (str bp)))

(defn dnnl-ending-blueprint [fact eng src-desc dst-type]
  (let-release [src-desc (desc src-desc)
                dst-shape (vec (rest (shape src-desc)))
                dst-desc (memory-desc dst-shape (or dst-type (data-type src-desc))
                                      (default-strides dst-shape))]
    (->DnnlEndingBlueprint fact eng src-desc dst-desc)))

(defmethod transfer! [DnnlEnding Object]
  [source destination]
  destination)
