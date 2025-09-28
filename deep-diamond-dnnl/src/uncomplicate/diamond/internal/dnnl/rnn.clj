;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.diamond.internal.dnnl.rnn
  (:require [uncomplicate.commons
             [core :refer [Releaseable release let-release with-release Info info view]]
             [utils :refer [dragan-says-ex]]]
            [uncomplicate.neanderthal
             [core :refer [axpby! transfer! scal!]]
             [random :refer [rand-normal!]]
             [block :refer [buffer initialize!]]]
            [uncomplicate.neanderthal.internal.api :refer [flow]]
            [uncomplicate.diamond.tensor :as tz
             :refer [Transfer input output connector TensorDescriptor shape layout view-tz]]
            [uncomplicate.diamond.internal
             [protocols
              :refer [Parameters ParametersSeq DescriptorProvider DiamondFactoryProvider
                      DiffParameters Backprop forward DiffTransfer diff-input diff-output LinearBackprop
                      backward-diff train-desc Initializable init RnnParameters batch-index inf-desc
                      train-desc diff-desc]]
             [utils :refer [default-strides direction-count transfer-rnn-weights-bias!]]]
            [uncomplicate.diamond.internal.dnnl
             [protocols :refer :all]
             [core :refer :all :as dnnl]
             [tensor :refer [dnnl-tensor dnnl-transformer]]
             [directed :refer [dnnl-nop-activation-blueprint]]]
            [uncomplicate.diamond.internal.neanderthal.directed :refer [->DirectedLayerBlueprint]])
  (:import [clojure.lang IFn AFn]))

;; ================================ RNN ====================================================

(deftype DnnlRnnInference [strm bluep
                           src-conn src-iter-conn src-iter-c-conn
                           bias-tz weights-tz weights-iter-tz
                           dst-tz dst-iter-tz dst-iter-c-tz
                           workspace-tz fwd-prim fwd-args]
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
    (release workspace-tz)
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
    (dragan-says-ex "Fused bias not available in RNNInference. Please use bias-layer and bias-iter."))
  (weights [_]
    (dragan-says-ex "Fused weights not available in RNNInference. Please use weights-layer and weights-iter."))
  RnnParameters
  (weights-layer [this]
    weights-tz)
  (weights-iter [this]
    weights-iter-tz)
  (bias-layer [this]
    bias-tz)
  (bias-iter [this]
    nil)
  ParametersSeq
  (parameters [_]
    [weights-tz weights-iter-tz bias-tz])
  Initializable
  (init [this init-fn]
    (init-fn weights-tz)
    (init-fn weights-iter-tz)
    (initialize! bias-tz (buffer bias-tz))
    (when src-iter-conn
      (let [src-iter-tz (input src-iter-conn)]
        (initialize! src-iter-tz (buffer src-iter-tz))))
    (when src-iter-c-conn
      (let [src-iter-c-tz (input src-iter-c-conn)]
        (initialize! src-iter-c-tz (buffer src-iter-c-tz))))
    this)
  IFn
  (invoke [_]
    (src-conn)
    (when src-iter-conn (src-iter-conn))
    (when src-iter-c-conn (src-iter-c-conn))
    (execute! strm fwd-prim fwd-args)
    dst-tz)
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(deftype DnnlRnnTraining [strm bluep
                          src-conn src-iter-conn src-iter-c-conn bias-tz
                          fused-weights-tz
                          weights-tz weights-conn weights-iter-tz weights-iter-conn
                          dst-tz dst-iter-tz dst-iter-c-tz
                          workspace-tz fwd-prim fwd-args
                          bwd-src-conn bwd-src-iter-conn bwd-src-iter-c-conn
                          bwd-weights-conn bwd-weights-iter-conn
                          bwd-dst-conn bwd-dst-iter-conn bwd-dst-iter-c-conn
                          diff-dst-tz diff-dst-iter-tz diff-dst-iter-c-tz
                          diff-src-conn diff-src-iter-conn diff-src-iter-c-conn
                          fused-diff-weights-tz post-diff-weights-tz
                          diff-weights-tz diff-weights-conn
                          diff-weights-iter-tz diff-weights-iter-conn
                          diff-bias-tz
                          bwd-prim bwd-args]
  Releaseable
  (release [_]
    (release src-conn)
    (release src-iter-conn)
    (release src-iter-c-conn)
    (release bias-tz)
    (release fused-weights-tz)
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
    (release fused-diff-weights-tz)
    (release post-diff-weights-tz)
    (release diff-weights-tz)
    (release diff-weights-conn)
    (release diff-weights-iter-tz)
    (release diff-weights-iter-conn)
    (release diff-bias-tz)
    (release bwd-prim)
    (release bwd-args))
  Info
  (info [this]
    {:bias (info bias-tz)
     :weights (info fused-weights-tz)
     :dst (info dst-tz)
     :diff-weights (info post-diff-weights-tz)})
  (info [this info-type]
    (case info-type
      :bias (info bias-tz)
      :weights (info fused-weights-tz)
      :dst (info dst-tz)
      :diff-weights (info post-diff-weights-tz)
      nil))
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
  Parameters
  (bias [_]
    bias-tz)
  (weights [_]
    fused-weights-tz)
  RnnParameters
  (weights-layer [this]
    (weights-tz))
  (weights-iter [this]
    (weights-iter-tz))
  (bias-layer [this]
    bias-tz)
  (bias-iter [this]
    nil)
  ParametersSeq
  (parameters [_]
    [fused-weights-tz bias-tz])
  DiffParameters
  (diff-weights [_]
    post-diff-weights-tz)
  Initializable
  (init [this init-fn]
    (init-fn fused-weights-tz)
    (initialize! bias-tz (buffer bias-tz))
    (when src-iter-conn
      (let [src-iter-tz (input src-iter-conn)]
        (initialize! src-iter-tz (buffer src-iter-tz))))
    (when src-iter-c-conn
      (let [src-iter-c-tz (input src-iter-c-conn)]
        (initialize! src-iter-c-tz (buffer src-iter-c-tz))))
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
    (execute! strm bwd-prim bwd-args)
    (diff-weights-conn)
    (diff-weights-iter-conn)
    (if (= 0.0 scal-g)
      (when-not (= 1.0 scal-diff-w)
        (scal! scal-diff-w fused-diff-weights-tz))
      (axpby! scal-diff-w fused-diff-weights-tz scal-g post-diff-weights-tz))
    (axpby! scal-diff-b diff-bias-tz scal-b bias-tz)
    (diff-src-conn)
    (when diff-src-iter-conn (diff-src-iter-conn))
    (when diff-src-iter-c-conn (diff-src-iter-c-conn))
    this))

(deftype DnnlRnnBlueprint [fact fused-weights-desc weights-desc weights-iter-desc bias-desc
                           infer-pd train-pd bwd-pd]
  Releaseable
  (release [_]
    (release fused-weights-desc)
    (release weights-desc)
    (release weights-iter-desc)
    (release bias-desc)
    (release infer-pd)
    (release train-pd)
    (release bwd-pd))
  Object
  (hashCode [_]
    (-> (hash :rnn) (hash-combine weights-desc) (hash-combine bias-desc)))
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
  (diff-desc [_]
    (diff-dst-md bwd-pd))
  TensorDescriptor
  (shape [this]
    (dims (train-desc this)))
  (data-type [this]
    (data-type (train-desc this)))
  (layout [this]
    (strides (train-desc this)))
  IFn
  (invoke [this src-tz]
    (let [[src-iter-tz src-iter-c-tz] [nil nil]]
      (let-release [src-conn (connector src-tz (src-md infer-pd))
                    src-iter-conn (when-let [src-iter-desc (arg-md infer-pd :src-iter)]
                                    (if src-iter-tz
                                      (connector src-iter-tz src-iter-desc)
                                      (dnnl-tensor fact src-iter-desc)))
                    src-iter-c-conn (when-let [src-iter-c-desc (arg-md infer-pd :src-iter-c)]
                                      (if src-iter-c-tz
                                        (connector src-iter-c-tz src-iter-c-desc)
                                        (dnnl-tensor fact src-iter-c-desc)))
                    bias-tz (dnnl-tensor fact bias-desc)
                    weights-tz (dnnl-tensor fact (weights-md infer-pd))
                    weights-iter-tz (dnnl-tensor fact (arg-md infer-pd :weights-iter))
                    dst-tz (dnnl-tensor fact (dst-md infer-pd) (batch-index src-tz))
                    dst-iter-tz (when-let [dst-iter-desc (arg-md infer-pd :dst-iter)]
                                  (dnnl-tensor fact dst-iter-desc))
                    dst-iter-c-tz (when-let [dst-iter-c-desc (arg-md infer-pd :dst-iter-c)]
                                    (dnnl-tensor fact dst-iter-c-desc))
                    workspace-tz (when-let [workspace-desc (arg-md infer-pd :workspace)]
                                   (dnnl-tensor fact workspace-desc))
                    fwd-prim (primitive infer-pd)
                    fwd-args (args {:src-layer (output src-conn)
                                    :src-iter (when src-iter-conn (output src-iter-conn))
                                    :src-iter-c (when src-iter-c-conn (output src-iter-c-conn))
                                    :weights-layer weights-tz
                                    :weights-iter weights-iter-tz
                                    :bias bias-tz
                                    :dst-layer dst-tz
                                    :dst-iter dst-iter-tz
                                    :dst-iter-c dst-iter-c-tz
                                    :workspace workspace-tz})]
        (->DnnlRnnInference (flow fact) this
                            src-conn src-iter-conn src-iter-c-conn
                            bias-tz weights-tz weights-iter-tz
                            dst-tz dst-iter-tz dst-iter-c-tz
                            workspace-tz
                            fwd-prim fwd-args))))
  (invoke [this src-tz diff-src-tz prop-diff? post-process-diff?]
    (let [[src-iter-tz src-iter-c-tz] [nil nil]]
      (let-release [src-conn (connector src-tz (src-md train-pd))
                    src-iter-conn (when-let [src-iter-desc (arg-md train-pd :src-iter)]
                                    (if src-iter-tz
                                      (connector src-iter-tz src-iter-desc)
                                      (dnnl-tensor fact src-iter-desc)))
                    src-iter-c-conn (when-let [src-iter-c-desc (arg-md train-pd :src-iter-c)]
                                      (if src-iter-c-tz
                                        (connector src-iter-c-tz src-iter-c-desc)
                                        (dnnl-tensor fact src-iter-c-desc)))
                    bias-tz (dnnl-tensor fact bias-desc)
                    fused-weights-tz (dnnl-tensor fact fused-weights-desc)
                    weights-tz (view-tz fused-weights-tz weights-desc)
                    weights-conn (connector weights-tz (weights-md train-pd))
                    weights-iter-tz (tz/offset! (view-tz fused-weights-tz weights-iter-desc)
                                                (apply * (shape weights-desc)))
                    weights-iter-conn (connector weights-iter-tz (arg-md train-pd :weights-iter))
                    dst-tz (dnnl-tensor fact (dst-md train-pd) (batch-index src-tz))
                    dst-iter-tz (when-let [dst-iter-desc (arg-md train-pd :dst-iter)]
                                  (dnnl-tensor fact dst-iter-desc))
                    dst-iter-c-tz (when-let [dst-iter-c-desc (arg-md train-pd :dst-iter-c)]
                                    (dnnl-tensor fact dst-iter-c-desc))
                    workspace-tz (when-let [workspace-desc (arg-md train-pd :workspace)]
                                   (dnnl-tensor fact workspace-desc))
                    fwd-prim (primitive train-pd)
                    fwd-args (args {:src-layer (output src-conn)
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
                    diff-dst-tz (dnnl-tensor fact (diff-dst-md bwd-pd) (batch-index dst-tz))
                    diff-dst-iter-tz (when-let [diff-dst-iter-desc (arg-md bwd-pd :diff-dst-iter)]
                                       (dnnl-tensor fact diff-dst-iter-desc))
                    diff-dst-iter-c-tz (when-let [diff-dst-iter-c-desc (arg-md bwd-pd :diff-dst-iter-c)]
                                         (dnnl-tensor fact diff-dst-iter-c-desc))
                    fused-diff-weights-tz (dnnl-tensor fact fused-weights-desc)
                    post-diff-weights-tz (if post-process-diff? (dnnl-tensor fact fused-weights-desc)
                                             fused-diff-weights-tz)
                    diff-weights-tz (view-tz fused-diff-weights-tz weights-desc)
                    diff-weights-conn (connector (diff-weights-md bwd-pd) diff-weights-tz)
                    diff-weights-iter-tz (tz/offset! (view-tz fused-diff-weights-tz weights-iter-desc)
                                                     (apply * (shape weights-desc)))
                    diff-weights-iter-conn (connector (arg-md bwd-pd :diff-weights-iter) diff-weights-iter-tz)
                    diff-bias-tz (dnnl-tensor fact bias-desc)
                    diff-src-conn (if prop-diff?
                                    (connector (diff-src-md bwd-pd) diff-src-tz)
                                    (dnnl-tensor fact (diff-src-md bwd-pd) (batch-index diff-src-tz)))
                    diff-src-iter-conn (when-let [diff-src-iter-desc (arg-md bwd-pd :diff-src-iter)]
                                         (if src-iter-tz
                                           (connector diff-src-iter-desc src-iter-tz)
                                           (dnnl-tensor fact diff-src-iter-desc)))
                    diff-src-iter-c-conn (when-let [diff-src-iter-c-desc (arg-md bwd-pd :diff-src-iter-c)]
                                           (if src-iter-c-tz
                                             (connector diff-src-iter-c-desc src-iter-c-tz)
                                             (dnnl-tensor fact diff-src-iter-c-desc)))
                    bwd-prim (primitive bwd-pd)
                    bwd-args (args {:src-layer (output bwd-src-conn)
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
                                    :diff-bias diff-bias-tz})]
        (->DnnlRnnTraining (flow fact) this
                           src-conn src-iter-conn src-iter-c-conn bias-tz
                           fused-weights-tz
                           weights-tz weights-conn weights-iter-tz weights-iter-conn
                           dst-tz dst-iter-tz dst-iter-c-tz
                           workspace-tz
                           fwd-prim fwd-args
                           bwd-src-conn bwd-src-iter-conn bwd-src-iter-c-conn
                           bwd-weights-conn bwd-weights-iter-conn
                           bwd-dst-conn bwd-dst-iter-conn bwd-dst-iter-c-conn
                           diff-dst-tz diff-dst-iter-tz diff-dst-iter-c-tz
                           diff-src-conn diff-src-iter-conn diff-src-iter-c-conn
                           fused-diff-weights-tz post-diff-weights-tz
                           diff-weights-tz diff-weights-conn
                           diff-weights-iter-tz diff-weights-iter-conn
                           diff-bias-tz
                           bwd-prim bwd-args))))
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(defmethod transfer! [DnnlRnnInference Object]
  [source destination]
  (transfer-rnn-weights-bias! source destination))

(defmethod transfer! [DnnlRnnTraining Object]
  [source destination]
  (transfer-rnn-weights-bias! source destination))

;; ================================ Vanilla RNN =================================================

(defn dnnl-rnn-op-blueprint [fact eng src-desc dst-desc weights-type activ alpha
                             dir lrs src-iter? dst-iter?]
  (let [src-desc (desc src-desc)
        src-shape (shape src-desc)
        [T N src-ch] src-shape
        dst-desc (desc dst-desc)
        [_ _ dst-ch] (shape dst-desc)
        dirs (direction-count dir)
        gts 1
        dst-iter-shape [lrs dirs N (if (= :bidirectional-concat dir) (* (long dst-ch) 2) dst-ch)]
        src-iter-shape dst-iter-shape
        dst-type (data-type dst-desc)
        weights-shape [lrs dirs src-ch gts dst-ch]
        weights-type (or weights-type (data-type src-desc) dst-type)
        weights-iter-shape [lrs dirs dst-ch gts dst-ch]
        bias-shape [lrs dirs gts dst-ch]
        fused-weights-shape [lrs dirs (+ (long src-ch) (long dst-ch)) gts dst-ch]]
    (with-release [weights-desc-any (memory-desc weights-shape weights-type :any)
                   weights-iter-desc-any (memory-desc weights-iter-shape weights-type :any)]
      (let-release [src-iter-desc (memory-desc src-iter-shape (data-type src-desc) :any)
                    dst-iter-desc (when dst-iter? (memory-desc dst-iter-shape dst-type :any))
                    bias-desc (memory-desc bias-shape dst-type :ldgo)
                    infer-pd (vanilla-rnn-fwd eng :inference activ dir src-desc src-iter-desc
                                              weights-desc-any weights-iter-desc-any bias-desc
                                              dst-desc dst-iter-desc alpha)
                    train-pd (vanilla-rnn-fwd eng :training activ dir src-desc src-iter-desc
                                              weights-desc-any weights-iter-desc-any bias-desc
                                              dst-desc dst-iter-desc alpha)
                    bwd-pd (vanilla-rnn-bwd eng train-pd activ dir src-desc src-iter-desc
                                            weights-desc-any weights-iter-desc-any bias-desc
                                            dst-desc dst-iter-desc
                                            src-desc src-iter-desc
                                            weights-desc-any weights-iter-desc-any bias-desc
                                            dst-desc dst-iter-desc alpha)
                    weights-desc-export (dnnl-contiguous-desc (weights-md train-pd))
                    weights-iter-desc-export (dnnl-contiguous-desc (arg-md train-pd :weights-iter))
                    fused-weights-desc (memory-desc fused-weights-shape weights-type :ldigo)] ;;TODO ldgoi is not exactly true unless I do weights/weihgts-iter striding! Improve this as part of cudnn/dnnl unification. Did I fix this then?
        (->DnnlRnnBlueprint fact fused-weights-desc weights-desc-export weights-iter-desc-export bias-desc
                            infer-pd train-pd bwd-pd)))))

(defn dnnl-rnn-blueprint [fact eng src-desc dst-desc lrs activ alpha weights-type src-iter? dst-iter?]
  (with-release [src-desc (memory-desc (shape src-desc) (or (tz/data-type src-desc) :float) :any)
                 dst-desc (memory-desc (shape dst-desc)
                                       (or (tz/data-type dst-desc) (tz/data-type src-desc))
                                       :any)]
    (let-release [rnn-op-bluep (dnnl-rnn-op-blueprint fact eng src-desc dst-desc weights-type
                                                      activ alpha :unidirectional lrs
                                                      src-iter? dst-iter?)
                  nop-activ-bluep (dnnl-nop-activation-blueprint fact
                                                                 (inf-desc rnn-op-bluep)
                                                                 (train-desc rnn-op-bluep)
                                                                 (diff-desc rnn-op-bluep))]
      (->DirectedLayerBlueprint fact :rnn rnn-op-bluep nop-activ-bluep))))

;; ================================ LSTM =======================================================

;; TODO maybe unify all these variants of rnn. Lots of code is the same.

(defn dnnl-gated-op-blueprint [fwd-desc-fn bwd-desc-fn gts
                               fact eng src-desc dst-desc weights-type
                               dir lrs src-iter? dst-iter?]
  (let [src-desc (desc src-desc)
        src-shape (shape src-desc)
        [T N src-ch] src-shape
        dst-desc (desc dst-desc)
        [_ _ dst-ch] (shape dst-desc)
        dirs (direction-count dir)
        dst-iter-shape [lrs dirs N (if (= :bidirectional-concat dir) (* (long dst-ch) 2) dst-ch)]
        src-iter-shape dst-iter-shape
        dst-type (data-type dst-desc)
        weights-shape [lrs dirs src-ch gts dst-ch]
        weights-type (or weights-type (data-type src-desc) dst-type)
        weights-iter-shape [lrs dirs dst-ch gts dst-ch]
        bias-shape [lrs dirs gts dst-ch]
        fused-weights-shape [lrs dirs (+ (long src-ch) (long dst-ch)) gts dst-ch]]
    (with-release [weights-desc-any (memory-desc weights-shape weights-type :any)
                   weights-iter-desc-any (memory-desc weights-iter-shape weights-type :any)]
      (let-release [src-iter-desc (memory-desc src-iter-shape (data-type src-desc) :any)
                    dst-iter-desc (when dst-iter? (memory-desc dst-iter-shape dst-type :any))
                    bias-desc (memory-desc bias-shape dst-type :ldgo)
                    infer-pd (fwd-desc-fn eng :inference dir src-desc src-iter-desc
                                          weights-desc-any weights-iter-desc-any bias-desc
                                          dst-desc dst-iter-desc)
                    train-pd (fwd-desc-fn eng :training dir src-desc src-iter-desc
                                          weights-desc-any weights-iter-desc-any bias-desc
                                          dst-desc dst-iter-desc)
                    bwd-pd (bwd-desc-fn eng train-pd dir src-desc src-iter-desc
                                        weights-desc-any weights-iter-desc-any bias-desc
                                        dst-desc dst-iter-desc
                                        src-desc src-iter-desc
                                        weights-desc-any weights-iter-desc-any bias-desc
                                        dst-desc dst-iter-desc)
                    weights-desc-export (dnnl-contiguous-desc (weights-md train-pd))
                    weights-iter-desc-export (dnnl-contiguous-desc (arg-md train-pd :weights-iter))
                    fused-weights-desc (memory-desc fused-weights-shape weights-type :ldigo)]
        (->DnnlRnnBlueprint fact fused-weights-desc weights-desc-export weights-iter-desc-export bias-desc
                            infer-pd train-pd bwd-pd)))))

(def dnnl-lstm-op-blueprint (partial dnnl-gated-op-blueprint lstm-fwd lstm-bwd 4))

(defn dnnl-lstm-blueprint [fact eng src-desc dst-desc lrs weights-type src-iter? dst-iter?]
  (with-release [src-desc (memory-desc (shape src-desc) (or (tz/data-type src-desc) :float) :any)
                 dst-desc (memory-desc (shape dst-desc)
                                       (or (tz/data-type dst-desc) (tz/data-type src-desc))
                                       :any)]
    (let-release [lstm-op-bluep (dnnl-lstm-op-blueprint fact eng src-desc dst-desc weights-type
                                                        :unidirectional lrs src-iter? dst-iter?)
                  nop-activ-bluep (dnnl-nop-activation-blueprint fact
                                                                 (inf-desc lstm-op-bluep)
                                                                 (train-desc lstm-op-bluep)
                                                                 (diff-desc lstm-op-bluep))]
      (->DirectedLayerBlueprint fact :lstm lstm-op-bluep nop-activ-bluep))))

;; ================================ GRU =======================================================

(def dnnl-gru-op-blueprint (partial dnnl-gated-op-blueprint gru-fwd gru-bwd 3))

(defn dnnl-gru-blueprint [fact eng src-desc dst-desc lrs weights-type src-iter? dst-iter?]
  (with-release [src-desc (memory-desc (shape src-desc) (or (tz/data-type src-desc) :float) :any)
                 dst-desc (memory-desc (shape dst-desc)
                                       (or (tz/data-type dst-desc) (tz/data-type src-desc))
                                       :any)]
    (let-release [gru-op-bluep (dnnl-gru-op-blueprint fact eng src-desc dst-desc weights-type
                                                      :unidirectional lrs src-iter? dst-iter?)
                  nop-activ-bluep (dnnl-nop-activation-blueprint fact
                                                                 (inf-desc gru-op-bluep)
                                                                 (train-desc gru-op-bluep)
                                                                 (diff-desc gru-op-bluep))]
      (->DirectedLayerBlueprint fact :gru gru-op-bluep nop-activ-bluep))))

;; ================================= Abbreviate Layer ==============================

(deftype DnnlAbbreviate [fact strm bluep transform-forward dst-tz transform-diff diff-sub]
  Releaseable
  (release [_]
    (release transform-forward)
    (release dst-tz)
    (release transform-diff)
    (release diff-sub))
  Object
  (hashCode [_]
    (hash-combine (hash :abbreviate) (shape dst-tz)))
  (equals [_ other]
    (and (instance? DnnlAbbreviate other)
         (let [other ^DnnlAbbreviate other]
           (equal-desc? dst-tz (.dst-tz other))
           (equal-desc? diff-sub (.diff-sub other)))))
  (toString [this]
    (str bluep))
  Info
  (info [this]
    {:src (input transform-forward)
     :dst dst-tz
     :topology :abbreviate})
  (info [this info-type]
    (case info-type
      :src (input transform-forward)
      :dst dst-tz
      :topology :abbreviate
      (info bluep info-type)))
  DiamondFactoryProvider
  (diamond-factory [_]
    fact)
  Transfer
  (input [_]
    (input transform-forward))
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
    (transform-forward)
    this)
  (backward [this]
    this)
  (backward [this _]
    (when diff-sub (initialize! diff-sub (buffer diff-sub)))
    (transform-diff)
    this)
  IFn
  (invoke [this]
    (transform-forward)
    dst-tz)
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(defmethod print-method DnnlAbbreviate
  [layer ^java.io.Writer w]
  (.write w (format "#Abbreviate[dst:%s]" (output layer))))

(deftype DnnlAbbreviateBlueprint [fact eng src-desc dst-desc]
  Releaseable
  (release [_]
    (release src-desc)
    (release dst-desc))
  Object
  (hashCode [_]
    (-> (hash :abbreviate)
        (hash-combine src-desc)
        (hash-combine dst-desc)))
  (equals [_ other]
    (and (instance? DnnlAbbreviateBlueprint other)
         (equal-desc? src-desc (.src-desc ^DnnlAbbreviateBlueprint other))
         (equal-desc? dst-desc (.dst-desc ^DnnlAbbreviateBlueprint other))))
  (toString [this]
    (str {:shape (shape this)
          :topology :abbreviate}))
  Info
  (info [this]
    {:shape (shape this)
     :topology :abbreviate})
  (info [this info-type]
    (case info-type
      :shape (shape this)
      :topology :abbreviate
      nil))
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
                    transform-forward (dnnl-transformer eng (flow fact) src-sub dst-tz)
                    transform-diff (dnnl-transformer eng (flow fact) dst-tz src-sub)]
        (->DnnlAbbreviate fact (flow fact) this transform-forward dst-tz transform-diff nil))))
  (invoke [this prev-layer _ _]
    (let [src-tz (output prev-layer)
          diff-tz (diff-input prev-layer)
          strm (flow fact)
          target-shape (update (shape src-tz) 0 dec)]
      (let-release [src-sub1 (tz/offset! (view-tz src-tz (shape dst-desc))
                                         (* (long (get (shape target-shape) 0))
                                            (long (get (strides src-tz) 0))))
                    diff-sub1 (tz/offset! (view-tz diff-tz (shape dst-desc))
                                          (* (long (get (shape target-shape) 0))
                                             (long (get (strides diff-tz) 0))))
                    diff-sub (when (pos? (long (get target-shape 0)))
                               (view-tz diff-tz target-shape))
                    dst-tz (dnnl-tensor fact dst-desc)
                    transform-forward (dnnl-transformer eng strm src-sub1 dst-tz)
                    transform-diff (dnnl-transformer eng strm dst-tz diff-sub1)]
        (->DnnlAbbreviate fact strm this transform-forward dst-tz transform-diff diff-sub))))
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(defmethod print-method DnnlAbbreviate
  [bp ^java.io.Writer w]
  (.write w (str bp)))

(defn dnnl-abbreviate-blueprint [fact eng src-desc dst-type]
  (let-release [src-desc (view (desc src-desc))
                dst-shape (vec (rest (shape src-desc)))
                dst-desc (memory-desc dst-shape (or dst-type (data-type src-desc))
                                      (default-strides dst-shape))]
    (->DnnlAbbreviateBlueprint fact eng src-desc dst-desc)))

(defmethod transfer! [DnnlAbbreviate Object]
  [source destination]
  destination)
