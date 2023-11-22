;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.diamond.internal.cudnn.rnn
  (:require [uncomplicate.commons
             [core :refer [Releaseable release let-release with-release Info info view bytesize]]
             [utils :refer [dragan-says-ex]]]
            [uncomplicate.clojure-cpp :refer [int-pointer pointer]]
            [uncomplicate.clojurecuda.core :refer [cuda-malloc cuda-free! memset! memcpy-to-device!]]
            [uncomplicate.neanderthal
             [core :refer [axpby! transfer! scal! entry!]]
             [block :refer [buffer initialize]]
             [random :refer [rand-normal!]]]
            [uncomplicate.neanderthal.internal.api :refer [flow]]
            [uncomplicate.diamond.tensor :as tz
             :refer [Transfer input output connector TensorDescriptor shape layout view-tz]]
            [uncomplicate.diamond.internal
             [protocols
              :refer [Parameters ParametersSeq DescriptorProvider DiamondFactoryProvider
                      DiffParameters Backprop forward DiffTransfer diff-input diff-output LinearBackprop
                      backward-diff train-desc Initializable init Workspace inf-ws-size train-ws-size
                      RnnParameters batch-index]]
             [utils :refer [default-strides direction-count transfer-rnn-weights-bias!]]]
            [uncomplicate.diamond.internal.dnnl.core :refer [memory-desc]]
            [uncomplicate.diamond.internal.cudnn
             [protocols :refer :all]
             [core :refer :all]
             [tensor :refer [cudnn-tensor-desc cudnn-tensor cudnn-transformer]]
             [directed :refer [cudnn-activ-blueprint]]]
            [uncomplicate.diamond.internal.neanderthal.directed :refer [->DirectedLayerBlueprint]])
  (:import [clojure.lang IFn AFn]))

;; ================================ RNN ====================================================

(deftype CUDnnRnnInference [fact cudnn-hdl bluep dev-seq-lengths
                            src-conn src-iter-tz src-iter-c-tz
                            bias-tz bias-iter-tz weights-tz weights-iter-tz
                            dst-tz dst-iter-tz dst-iter-c-tz
                            rnn-desc rnn-src-desc rnn-dst-desc iter-desc
                            weights-mem work reserve]
  Releaseable
  (release [_]
    (cuda-free! dev-seq-lengths)
    (release src-conn)
    (release src-iter-tz)
    (release src-iter-c-tz)
    (release bias-tz)
    (release weights-tz)
    (release weights-iter-tz)
    (release dst-tz)
    (release dst-iter-tz)
    (release dst-iter-c-tz)
    (release iter-desc)
    (cuda-free! weights-mem)
    (cuda-free! work)
    (cuda-free! reserve))
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
    (dragan-says-ex "Fused bias not available in RNNInference. Please use weights-layer and weights-iter."))
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
    bias-iter-tz)
  ParametersSeq
  (parameters [_]
    (dragan-says-ex "Fused weights not available in RNNInference. Please use weights-layer and weights-iter."))
  Initializable
  (init [this init-fn]
    (init-fn weights-tz)
    (init-fn weights-iter-tz)
    (entry! bias-tz 0.0)
    (entry! bias-iter-tz 0.0)
    (when src-iter-tz
      (initialize src-iter-tz (buffer src-iter-tz) 0.0))
    (when src-iter-c-tz
      (initialize src-iter-c-tz (buffer src-iter-c-tz) 0.0))
    this)
  IFn
  (invoke [_]
    (src-conn)
    (rnn-fwd cudnn-hdl rnn-desc :inference dev-seq-lengths
             rnn-src-desc (buffer (output src-conn)) rnn-dst-desc (buffer dst-tz)
             iter-desc (when src-iter-tz (buffer src-iter-tz))
             (when dst-iter-tz (buffer dst-iter-tz))
             iter-desc (when src-iter-c-tz (buffer src-iter-c-tz))
             (when dst-iter-c-tz (buffer dst-iter-c-tz))
             weights-mem work reserve)
    dst-tz)
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(deftype CUDnnRnnTraining [fact cudnn-hdl bluep dev-seq-lengths
                           src-conn src-iter-tz src-iter-c-tz
                           fused-bias-tz bias-tz bias-iter-tz
                           fused-weights-tz weights-tz weights-iter-tz
                           dst-tz dst-iter-tz dst-iter-c-tz
                           rnn-desc rnn-src-desc rnn-dst-desc iter-desc
                           diff-dst-tz diff-dst-iter-tz diff-dst-iter-c-tz
                           diff-src-conn diff-src-iter-tz diff-src-iter-c-tz
                           fused-diff-weights-tz post-diff-weights-tz
                           diff-weights-tz diff-weights-iter-tz
                           fused-diff-bias-tz diff-bias-tz diff-bias-iter-tz
                           weights-mem diff-weights-mem work reserve]
  Releaseable
  (release [_]
    (cuda-free! dev-seq-lengths)
    (release src-conn)
    (release src-iter-tz)
    (release src-iter-c-tz)
    (release fused-bias-tz)
    (release bias-tz)
    (release bias-iter-tz)
    (release fused-weights-tz)
    (release weights-tz)
    (release weights-iter-tz)
    (release dst-tz)
    (release dst-iter-tz)
    (release dst-iter-c-tz)
    (release rnn-desc)
    (release rnn-src-desc)
    (release rnn-dst-desc)
    (release iter-desc)
    (release diff-dst-tz)
    (release diff-dst-iter-tz)
    (release diff-dst-iter-c-tz)
    (release diff-src-conn)
    (release diff-src-iter-tz)
    (release diff-src-iter-c-tz)
    (release fused-diff-weights-tz)
    (release post-diff-weights-tz)
    (release diff-weights-tz)
    (release diff-weights-iter-tz)
    (release fused-diff-bias-tz)
    (release diff-bias-tz)
    (release diff-bias-iter-tz)
    (cuda-free! weights-mem)
    (cuda-free! diff-weights-mem)
    (cuda-free! work)
    (cuda-free! reserve))
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
    fused-bias-tz)
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
    bias-iter-tz)
  ParametersSeq
  (parameters [_]
    [fused-weights-tz fused-bias-tz])
  DiffParameters
  (diff-weights [_]
    post-diff-weights-tz)
  Initializable
  (init [this init-fn]
    (init-fn fused-weights-tz)
    (entry! fused-bias-tz 0.0)
    (memset! diff-weights-mem 0)
    (when src-iter-tz
      (entry! src-iter-tz 0.0))
    (when src-iter-c-tz
      (entry! src-iter-c-tz 0.0))
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
    (rnn-fwd cudnn-hdl rnn-desc :training dev-seq-lengths
             rnn-src-desc (buffer (output src-conn)) rnn-dst-desc (buffer dst-tz)
             iter-desc (when src-iter-tz (buffer src-iter-tz))
             (when dst-iter-tz (buffer dst-iter-tz))
             iter-desc (when src-iter-c-tz (buffer src-iter-c-tz))
             (when dst-iter-c-tz (buffer dst-iter-c-tz))
             weights-mem work reserve)
    this)
  (backward [this]
    (backward-diff this 1.0 0.0 1.0 0.0))
  LinearBackprop
  (backward-diff [this scal-diff-w scal-g scal-diff-b scal-b]
    (rnn-bwd-data cudnn-hdl rnn-desc dev-seq-lengths
                  rnn-dst-desc (buffer dst-tz) (buffer diff-dst-tz)
                  rnn-src-desc (buffer (input diff-src-conn))
                  iter-desc (when src-iter-tz (buffer src-iter-tz))
                  (when diff-dst-iter-tz (buffer diff-dst-iter-tz))
                  (when diff-src-iter-tz (buffer diff-src-iter-tz))
                  iter-desc (when src-iter-c-tz (buffer src-iter-c-tz))
                  (when diff-dst-iter-c-tz (buffer diff-dst-iter-c-tz))
                  (when diff-src-iter-c-tz (buffer diff-src-iter-c-tz))
                  weights-mem work reserve)
    (memset! diff-weights-mem 0)
    (rnn-bwd-weights cudnn-hdl rnn-desc :add dev-seq-lengths
                     rnn-src-desc (buffer (input src-conn))
                     iter-desc (when src-iter-tz (buffer src-iter-tz))
                     rnn-dst-desc (buffer dst-tz)
                     diff-weights-mem work reserve)
    (if (= 0.0 scal-g)
      (when-not (= 1.0 scal-diff-w)
        (scal! scal-diff-w fused-diff-weights-tz))
      (axpby! scal-diff-w fused-diff-weights-tz scal-g post-diff-weights-tz))
    (axpby! scal-diff-b fused-diff-bias-tz scal-b fused-bias-tz)
    (diff-src-conn)
    this))

(deftype CUDnnRnnBlueprint [fact cudnn-hdl rnn-desc weights-type ^long weights-size
                            ^long inf-work-size ^long inf-reserve-size
                            ^long train-work-size ^long train-reserve-size
                            seq-lengths ^long weights-offset
                            ^long bias-offset ^long bias-iter-offset
                            rnn-src-desc rnn-dst-desc
                            src-desc fused-weights-desc weights-desc
                            fused-bias-desc bias-desc dst-desc
                            ldnc-iter-desc iter-desc src-iter? dst-iter? iter-c?]
  Releaseable
  (release [_]
    (release rnn-desc)
    (release rnn-src-desc)
    (release rnn-dst-desc)
    (release src-desc)
    (release fused-weights-desc)
    (release weights-desc)
    (release bias-desc)
    (release dst-desc)
    (release ldnc-iter-desc)
    (release iter-desc)
    (release seq-lengths))
  Object
  (hashCode [_]
    (-> (hash :rnn) (hash-combine weights-desc) (hash-combine bias-desc)))
  (equals [_ other]
    (and (instance? CUDnnRnnBlueprint other)
         (let [other ^CUDnnRnnBlueprint other]
           (and (equal-desc? src-desc (.src-desc ^CUDnnRnnBlueprint other))
                (equal-desc? fused-weights-desc (.fused-weights-desc ^CUDnnRnnBlueprint other))
                (equal-desc? dst-desc (.dst-desc ^CUDnnRnnBlueprint other))
                (equal-desc? ldnc-iter-desc (.ldnc-iter-desc ^CUDnnRnnBlueprint other))
                (equal-desc? iter-desc (.iter-desc ^CUDnnRnnBlueprint other))
                (= src-iter? (.src-iter? other))
                (= dst-iter? (.dst-iter? other))
                (= iter-c? (.iter-c? other))))))
  (toString [this]
    (pr-str {:src src-desc :weights weights-desc :dst dst-desc}))
  Info
  (info [this info-type]
    (case info-type
      :bias bias-desc
      :inference {:src src-desc
                  :weights weights-desc
                  :dst dst-desc}
      :training {:src src-desc
                 :weights weights-desc
                 :dst dst-desc}
      nil))
  (info [this]
    {:bias bias-desc
     :inference {:src src-desc
                 :weights weights-desc
                 :dst dst-desc}
     :training {:src src-desc
                :weights weights-desc
                :dst dst-desc}})
  DiamondFactoryProvider
  (diamond-factory [_]
    fact)
  DescriptorProvider
  (inf-desc [_]
    (view dst-desc)) ;;TODO check whether I still need all these views, since DNNL does not.
  (train-desc [_]
    (view dst-desc))
  (diff-desc [_]
    (view dst-desc))
  TensorDescriptor
  (shape [this]
    (shape (train-desc this)))
  (data-type [this]
    (data-type (train-desc this)))
  (layout [this]
    (strides (train-desc this)))
  Workspace
  (inf-ws-size [this]
    inf-work-size)
  (train-ws-size [this]
    train-work-size)
  IFn
  (invoke [this src-tz]
    (let [[src-iter-tz src-iter-c-tz] [nil nil]]
      (let-release [src-conn (connector src-tz src-desc)
                    dev-seq-lengths (cuda-malloc (bytesize seq-lengths) :int)
                    weights (cuda-malloc weights-size weights-type)
                    weights-tz (cudnn-tensor fact false weights (view weights-desc))
                    weights-iter-tz (cudnn-tensor fact false (pointer weights weights-offset)
                                                  (view weights-desc))
                    bias-tz (cudnn-tensor fact false (pointer weights bias-offset) (view bias-desc))
                    bias-iter-tz (cudnn-tensor fact false (pointer weights bias-iter-offset)
                                               (view bias-desc))
                    dst-tz (cudnn-tensor fact (view dst-desc) 1) ;;TODO check whether cuda uses :tnc or :ntc => determined by rnn-src-desc!
                    dst-iter-tz (when dst-iter? (cudnn-tensor fact (view ldnc-iter-desc)))
                    dst-iter-c-tz (when (and dst-iter? iter-c?) (cudnn-tensor fact (view ldnc-iter-desc)))
                    work (cuda-malloc inf-work-size);;TODO here we can use global workspace
                    reserve (cuda-malloc inf-reserve-size)]
        (memcpy-to-device! seq-lengths dev-seq-lengths)
        (memset! weights 0)
        (->CUDnnRnnInference fact cudnn-hdl this dev-seq-lengths
                             src-conn src-iter-tz src-iter-c-tz
                             bias-tz bias-iter-tz weights-tz weights-iter-tz
                             dst-tz dst-iter-tz dst-iter-c-tz
                             (view rnn-desc) (view rnn-src-desc) (view rnn-dst-desc) (view iter-desc)
                             weights work reserve))))
  (invoke [this src-tz diff-src-tz prop-diff? post-process-diff?];;TODO keep in mind that some of source tensors might have to be views!
    (let [[src-iter-tz src-iter-c-tz] [nil nil]]
      (let-release [src-conn (connector src-tz src-desc)
                    dev-seq-lengths (cuda-malloc (bytesize seq-lengths) :int)
                    weights (cuda-malloc weights-size weights-type)
                    fused-weights-tz (cudnn-tensor fact false weights (view fused-weights-desc))
                    weights-tz (cudnn-tensor fact false weights (view weights-desc))
                    weights-iter-tz (cudnn-tensor fact false (pointer weights weights-offset)
                                                  (view weights-desc))
                    fused-bias-tz (cudnn-tensor fact false (pointer weights bias-offset)
                                                (view fused-bias-desc))
                    bias-tz (cudnn-tensor fact false (pointer weights bias-offset) (view bias-desc))
                    bias-iter-tz (cudnn-tensor fact false (pointer weights bias-iter-offset)
                                               (view bias-desc))
                    dst-tz (cudnn-tensor fact (view dst-desc) 1)
                    dst-iter-tz (when dst-iter? (cudnn-tensor fact (view ldnc-iter-desc)))
                    dst-iter-c-tz (when (and dst-iter? iter-c?)
                                    (cudnn-tensor fact (view ldnc-iter-desc)))
                    work (cuda-malloc train-work-size);;TODO here we can use global workspace
                    reserve (cuda-malloc train-reserve-size)
                    diff-dst-tz (cudnn-tensor fact (view dst-desc) 1)
                    diff-dst-iter-tz (when dst-iter? (cudnn-tensor fact (view ldnc-iter-desc)))
                    diff-dst-iter-c-tz (when (and dst-iter? iter-c?)
                                         (cudnn-tensor fact (view ldnc-iter-desc)))
                    diff-src-conn (if prop-diff?
                                    (connector src-desc diff-src-tz)
                                    (cudnn-tensor fact src-desc (batch-index diff-src-tz)))
                    diff-src-iter-tz (when src-iter-tz (view-tz src-iter-tz (view ldnc-iter-desc)))
                    diff-src-iter-c-tz (when src-iter-c-tz (view-tz src-iter-tz (view ldnc-iter-desc)))
                    diff-weights (cuda-malloc weights-size weights-type)
                    fused-diff-weights-tz (cudnn-tensor fact false diff-weights (view fused-weights-desc))
                    post-diff-weights-tz (if post-process-diff?
                                           (cudnn-tensor fact (view fused-weights-desc))
                                           fused-diff-weights-tz)
                    diff-weights-tz (cudnn-tensor fact false diff-weights (view weights-desc))
                    diff-weights-iter-tz (cudnn-tensor fact false (pointer diff-weights weights-offset)
                                                       (view weights-desc))
                    fused-diff-bias-tz (cudnn-tensor fact false (pointer diff-weights bias-offset)
                                                     (view fused-bias-desc))
                    diff-bias-tz (cudnn-tensor fact false (pointer diff-weights bias-offset)
                                               (view bias-desc))
                    diff-bias-iter-tz (cudnn-tensor fact false (pointer diff-weights bias-iter-offset)
                                                    (view bias-desc))]
        (memcpy-to-device! seq-lengths dev-seq-lengths)
        (memset! weights 0)
        (memset! diff-weights 0)
        (->CUDnnRnnTraining fact cudnn-hdl this dev-seq-lengths
                            src-conn src-iter-tz src-iter-c-tz
                            fused-bias-tz bias-tz bias-iter-tz
                            fused-weights-tz weights-tz weights-iter-tz
                            dst-tz dst-iter-tz dst-iter-c-tz
                            (view rnn-desc) (view rnn-src-desc) (view rnn-dst-desc) (view iter-desc)
                            diff-dst-tz diff-dst-iter-tz diff-dst-iter-c-tz
                            diff-src-conn diff-src-iter-tz diff-src-iter-c-tz
                            fused-diff-weights-tz post-diff-weights-tz
                            diff-weights-tz diff-weights-iter-tz
                            fused-diff-bias-tz diff-bias-tz diff-bias-iter-tz
                            weights diff-weights work reserve))))
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(defmethod transfer! [CUDnnRnnInference Object]
  [source destination]
  (transfer-rnn-weights-bias! source destination))

(defmethod transfer! [CUDnnRnnTraining Object]
  [source destination]
  (transfer-rnn-weights-bias! source destination))

;; ================================ Vanilla RNN =================================================

(defn cudnn-rnn-op-blueprint [fact cudnn-hdl src-desc dst-desc weights-type
                              activ dir lrs src-iter? dst-iter?]
  (let [gts (case activ
              :gru 3
              :lstm 4
              1)
        mode (case (long gts) 3 :gru 4 :lstm activ)
        algo :standard ;; TODO :standard is the fastest one for examples that I've tried. Needs more exploration.
        src-shape (shape src-desc)
        [T N src-ch] src-shape
        dst-shape (shape dst-desc)
        [_ _ dst-ch] dst-shape
        dirs (direction-count dir)
        seq-lengths (repeat N T)
        ldnc-iter-shape [lrs dirs N dst-ch]
        iter-shape [(* (long lrs) dirs) N dst-ch]
        weights-shape [lrs dirs src-ch gts dst-ch]
        weights-strd (with-release [md (memory-desc weights-shape :float :ldgoi)] (layout md))
        weights-offset (get weights-strd 0)
        weights-stride (update weights-strd 0 (partial * 2))
        bias-shape [lrs dirs gts dst-ch]
        bias-strd (default-strides bias-shape)
        bias-offset (long (get bias-strd 0))
        bias-stride (update bias-strd 0 (partial * 2))
        fused-bias-shape [lrs dirs gts (* 2 (long dst-ch))]
        fused-bias-stride (with-release [md (memory-desc fused-bias-shape :float :ldgo)] (layout md))
        fused-weights-shape [lrs dirs (+ (long src-ch) (long dst-ch)) gts dst-ch]
        fused-weights-stride (with-release [md (memory-desc fused-weights-shape :float :ldgoi)] (layout md))]
    (let-release [src-desc (desc src-desc)
                  dst-desc (desc dst-desc)
                  dtype (data-type dst-desc)
                  weights-type (or weights-type dtype)
                  ldnc-iter-desc (cudnn-tensor-desc ldnc-iter-shape dtype :nchw)
                  iter-desc (cudnn-tensor-desc iter-shape dtype :nchw)
                  fused-weights-desc (cudnn-tensor-desc fused-weights-shape weights-type fused-weights-stride)
                  weights-desc (cudnn-tensor-desc weights-shape weights-type weights-stride)
                  fused-bias-desc (cudnn-tensor-desc fused-bias-shape weights-type fused-bias-stride)
                  bias-desc (cudnn-tensor-desc bias-shape weights-type bias-stride)
                  rnn-desc (rnn-descriptor algo mode :double :unidirectional :linear
                                           :float :float :default src-ch dst-ch dst-ch lrs
                                           nil :padded-io-enabled)
                  rnn-src-desc (rnn-data-descriptor :float :seq-mayor-packed src-ch seq-lengths 0.0)
                  rnn-dst-desc (rnn-data-descriptor :float :seq-mayor-packed dst-ch seq-lengths 0.0)]
      (when (= :dynamic algo) (build-rnn-dynamic! cudnn-hdl rnn-desc N))
      (let [weights-size (rnn-weights-space-size cudnn-hdl rnn-desc)
            [inf-work-size inf-reserve-size] (rnn-temp-space-size cudnn-hdl rnn-desc rnn-src-desc :inference)
            [train-work-size train-reserve-size] (rnn-temp-space-size cudnn-hdl rnn-desc rnn-src-desc :training)]
        (->CUDnnRnnBlueprint fact cudnn-hdl rnn-desc weights-type weights-size
                             (max 1 (long inf-work-size)) (max 1 (long inf-reserve-size))
                             (max 1 (long train-work-size)) (max 1 (long train-reserve-size))
                             (int-pointer seq-lengths) weights-offset (apply * fused-weights-shape)
                             (+ (long (apply * fused-weights-shape)) bias-offset)
                             rnn-src-desc rnn-dst-desc
                             src-desc fused-weights-desc weights-desc
                             fused-bias-desc bias-desc dst-desc
                             ldnc-iter-desc iter-desc src-iter? dst-iter? (= :lstm activ))))))

(defn cudnn-rnn-blueprint [fact cudnn-hdl src-desc dst-desc lrs activ weights-type src-iter? dst-iter?]
  (let-release [src-desc (cudnn-tensor-desc (shape src-desc) (or (tz/data-type src-desc) :float)
                                            (or (layout src-desc) :tnc))
                dst-desc (cudnn-tensor-desc (shape dst-desc)
                                            (or (tz/data-type dst-desc) (tz/data-type src-desc) :float)
                                            (or (layout dst-desc) :tnc))
                rnn-op-bluep (cudnn-rnn-op-blueprint fact cudnn-hdl src-desc dst-desc weights-type
                                                     activ :unidirectional lrs src-iter? dst-iter?)
                nop-activ-bluep (cudnn-activ-blueprint fact (train-desc rnn-op-bluep) :identity nil)]
    (->DirectedLayerBlueprint fact :rnn rnn-op-bluep nop-activ-bluep)))

;; ================================= Abbreviate Layer ==============================

(deftype CUDnnAbbreviate [fact cudnn-hdl bluep transform-forward dst-tz transform-diff diff-sub]
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
    (and (instance? CUDnnAbbreviate other)
         (let [other ^CUDnnAbbreviate other]
           (and (= transform-forward (.transform-forward other))
                (= transform-diff (.transform-diff other))
                (= dst-tz (.dst-tz other))
                (= diff-sub (.diff-sub other))))))
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
    (when diff-sub (entry! diff-sub 0.0))
    (transform-diff)
    this)
  IFn
  (invoke [this]
    (transform-forward)
    dst-tz)
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(defmethod print-method CUDnnAbbreviate
  [layer ^java.io.Writer w]
  (.write w (format "#Abbreviate[dst:%s]" (output layer))))

(deftype CUDnnAbbreviateBlueprint [fact src-desc dst-desc sub-shape sub1-shape]
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
    (and (instance? CUDnnAbbreviateBlueprint other)
         (equal-desc? src-desc (.src-desc ^CUDnnAbbreviateBlueprint other))
         (equal-desc? dst-desc (.dst-desc ^CUDnnAbbreviateBlueprint other))))
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
      (let-release [src-sub1 (tz/offset! (view-tz src-tz sub1-shape)
                                         (* (long (get (shape sub-shape) 0))
                                            (long (get (strides src-tz) 0))))
                    dst-tz (cudnn-tensor fact (view dst-desc))
                    dst-sub1 (view-tz dst-tz (tz/desc sub1-shape (data-type dst-desc)
                                                      (default-strides sub1-shape)))
                    transform-forward (cudnn-transformer (handle fact) src-sub1 dst-sub1)]
        (->CUDnnAbbreviate fact (handle fact) this transform-forward dst-tz nil nil))))
  (invoke [this prev-layer _ _]
    (let [src-tz (output prev-layer)
          diff-tz (diff-input prev-layer)
          cudnn-hdl (handle fact)]
      (let-release [src-sub1 (tz/offset! (view-tz src-tz sub1-shape)
                                         (* (long (get (shape sub-shape) 0))
                                            (long (get (strides src-tz) 0))))
                    diff-sub1 (tz/offset! (view-tz diff-tz sub1-shape)
                                          (* (long (get (shape sub-shape) 0))
                                             (long (get (strides diff-tz) 0))))
                    diff-sub (when (pos? (long (get sub-shape 0)))
                               (view-tz diff-tz sub-shape))
                    dst-tz (cudnn-tensor fact (view dst-desc))
                    dst-sub1 (view-tz dst-tz (tz/desc sub1-shape (data-type dst-desc)
                                                      (default-strides sub1-shape)))
                    transform-forward (cudnn-transformer cudnn-hdl src-sub1 dst-sub1)
                    transform-diff (cudnn-transformer cudnn-hdl dst-sub1 diff-sub1)]
        (->CUDnnAbbreviate fact (handle fact) this transform-forward dst-tz transform-diff diff-sub))))
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(defmethod print-method CUDnnAbbreviate
  [bp ^java.io.Writer w]
  (.write w (str bp)))

(defn cudnn-abbreviate-blueprint [fact src-desc dst-type]
  (let-release [src-desc (desc src-desc)
                src-shape (shape src-desc)
                dst-shape (vec (rest src-shape))
                sub-shape (update src-shape 0 dec)
                sub1-shape (assoc src-shape 0 1)
                dst-desc (cudnn-tensor-desc dst-shape (or dst-type (data-type src-desc))
                                            (default-strides dst-shape))]
    (->CUDnnAbbreviateBlueprint fact src-desc dst-desc sub-shape sub1-shape)))

(defmethod transfer! [CUDnnAbbreviate Object]
  [source destination]
  destination)
