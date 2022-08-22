;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.diamond.internal.cudnn.rnn
  (:require [uncomplicate.commons
             [core :refer [Releaseable release let-release with-release Info info view]]
             [utils :refer [dragan-says-ex]]]
            [uncomplicate.clojurecuda.core :refer [mem-alloc memset! memcpy-host!]]
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
             [tensor :refer :all]
             [directed :refer [cudnn-activ-blueprint]]]
            [uncomplicate.diamond.internal.neanderthal.directed :refer [->DirectedLayerBlueprint]])
  (:import [clojure.lang IFn AFn]))

;; ================================ RNN ====================================================

(deftype CUDnnRnnInference [fact cudnn-hdl bluep dev-seq-lengths
                            src-conn src-iter-tz src-iter-c-tz
                            bias-tz weights-tz weights-iter-tz
                            dst-tz dst-iter-tz dst-iter-c-tz
                            rnn-desc rnn-src-desc rnn-dst-desc iter-desc
                            weights-mem work reserve]
  Releaseable
  (release [_]
    (release dev-seq-lengths)
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
    (release weights-mem)
    (release work)
    (release reserve))
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
    (dragan-says-ex "Fused weights not available in RNNInference. Please use weights-layer and weights-iter."))
  RnnParameters
  (weights-layer [this]
    (weights-tz))
  (weights-iter [this]
    (weights-iter-tz))
  ParametersSeq
  (parameters [_]
    (dragan-says-ex "Fused weights not available in RNNInference. Please use weights-layer and weights-iter."))
  Initializable
  (init [this init-fn]
    (rand-normal! 0.0 (/ 1.0 (long (get (shape weights-tz) 2))) weights-tz)
    (rand-normal! 0.0 (/ 1.0 (long (get (shape weights-iter-tz) 2))) weights-iter-tz)
    (entry! bias-tz 0.0)
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
                           bias-tz fused-weights-tz weights-tz weights-iter-tz
                           dst-tz dst-iter-tz dst-iter-c-tz
                           rnn-desc rnn-src-desc rnn-dst-desc iter-desc
                           diff-dst-tz diff-dst-iter-tz diff-dst-iter-c-tz
                           diff-src-conn diff-src-iter-tz diff-src-iter-c-tz
                           fused-diff-weights-tz post-diff-weights-tz
                           diff-weights-tz diff-weights-iter-tz diff-bias-tz
                           weights-mem diff-weights-mem work reserve]
  Releaseable
  (release [_]
    (release dev-seq-lengths)
    (release src-conn)
    (release src-iter-tz)
    (release src-iter-c-tz)
    (release bias-tz)
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
    (release diff-bias-tz)
    (release weights-mem)
    (release work)
    (release reserve))
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
  ParametersSeq
  (parameters [_]
    [fused-weights-tz bias-tz])
  DiffParameters
  (diff-weights [_]
    post-diff-weights-tz)
  Initializable
  (init [this init-fn]
    (rand-normal! 0.0 (/ 2.0 (long (get (shape fused-weights-tz) 2))) fused-weights-tz)
    (entry! bias-tz 0.0)
    (memset! diff-weights-mem 0.0)
    (when src-iter-tz
      (initialize src-iter-tz (buffer src-iter-tz) 0.0))
    (when src-iter-c-tz
      (initialize src-iter-c-tz (buffer src-iter-c-tz) 0.0))
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
    (memset! diff-weights-mem 0.0)
    (rnn-bwd-weights cudnn-hdl rnn-desc :add dev-seq-lengths
                     rnn-src-desc (buffer (input src-conn))
                     iter-desc (when src-iter-tz (buffer src-iter-tz))
                     rnn-dst-desc (buffer dst-tz)
                     diff-weights-mem work reserve)
    (if (= 0.0 scal-g)
      (when-not (= 1.0 scal-diff-w)
        (scal! scal-diff-w fused-diff-weights-tz))
      (axpby! scal-diff-w fused-diff-weights-tz scal-g post-diff-weights-tz))
    (axpby! scal-diff-b diff-bias-tz scal-b bias-tz)
    (diff-src-conn)
    this))

(deftype CUDnnRnnBlueprint [fact cudnn-hdl rnn-desc ^long weights-size
                            ^long inf-work-size ^long inf-reserve-size
                            ^long train-work-size ^long train-reserve-size
                            ^ints seq-lengths ^long weights-offset ^long bias-offset
                            rnn-src-desc rnn-dst-desc
                            src-desc fused-weights-desc weights-desc bias-desc dst-desc
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
    (release iter-desc))
  Object
  (hashCode [_]
    (-> (hash :inner-product)
        (hash-combine src-desc) (hash-combine weights-desc)
        (hash-combine bias-desc) (hash-combine dst-desc)
        ;;TODO
        ))
  (equals [_ other]
    (and (instance? CUDnnRnnBlueprint other)
         (equal-desc? src-desc (.src-desc ^CUDnnRnnBlueprint other))
         (equal-desc? weights-desc (.weights-desc ^CUDnnRnnBlueprint other))
         (equal-desc? dst-desc (.dst-desc ^CUDnnRnnBlueprint other))
         ;;TODO
         ))
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
    (view dst-desc))
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
                    dev-seq-lengths (mem-alloc (* Float/BYTES (alength seq-lengths)))
                    weights (mem-alloc weights-size)
                    weights-tz (cudnn-tensor fact false weights 0 (view weights-desc))
                    weights-iter-tz (cudnn-tensor fact false weights weights-offset (view weights-desc))
                    bias-tz (cudnn-tensor fact false weights bias-offset (view bias-desc))
                    dst-tz (cudnn-tensor fact (view dst-desc) 1);;TODO check whether cuda uses :tnc or :ntc => determined by rnn-src-desc!
                    dst-iter-tz (when dst-iter? (cudnn-tensor fact (view ldnc-iter-desc)))
                    dst-iter-c-tz (when (and dst-iter? iter-c?) (cudnn-tensor fact (view ldnc-iter-desc)))
                    work (mem-alloc inf-work-size);;TODO here we can use global workspace
                    reserve (mem-alloc inf-reserve-size)]
        (memcpy-host! seq-lengths dev-seq-lengths)
        (->CUDnnRnnInference fact cudnn-hdl this dev-seq-lengths
                             src-conn src-iter-tz src-iter-c-tz
                             bias-tz weights-tz weights-iter-tz
                             dst-tz dst-iter-tz dst-iter-c-tz
                             (view rnn-desc) (view rnn-src-desc) (view rnn-dst-desc) (view iter-desc)
                             weights work reserve))))
  (invoke [this src-tz diff-src-tz prop-diff? post-process-diff?];;TODO keep in mind that some of source tensors might have to be views!
    (let [[src-iter-tz src-iter-c-tz] [nil nil]]
      (let-release [src-conn (connector src-tz src-desc)
                    dev-seq-lengths (mem-alloc (* Float/BYTES (alength seq-lengths)))
                    weights (mem-alloc weights-size)
                    fused-weights-tz (cudnn-tensor fact false weights 0 (view fused-weights-desc))
                    weights-tz (cudnn-tensor fact false weights 0 (view weights-desc))
                    weights-iter-tz (cudnn-tensor fact false weights weights-offset (view weights-desc))
                    bias-tz (cudnn-tensor fact false weights bias-offset (view bias-desc))
                    dst-tz (cudnn-tensor fact (view dst-desc) 1)
                    dst-iter-tz (when dst-iter? (cudnn-tensor fact (view ldnc-iter-desc)))
                    dst-iter-c-tz (when (and dst-iter? iter-c?) (cudnn-tensor fact (view ldnc-iter-desc)))
                    work (mem-alloc train-work-size);;TODO here we can use global workspace
                    reserve (mem-alloc train-reserve-size)
                    diff-dst-tz (cudnn-tensor fact (view dst-desc) 1)
                    diff-dst-iter-tz (when dst-iter? (cudnn-tensor fact (view ldnc-iter-desc)))
                    diff-dst-iter-c-tz (when (and dst-iter? iter-c?) (cudnn-tensor fact (view ldnc-iter-desc)))
                    diff-src-conn (if prop-diff?
                                    (connector src-desc diff-src-tz)
                                    (cudnn-tensor fact src-desc (batch-index diff-src-tz)))
                    diff-src-iter-tz (when src-iter-tz (view-tz src-iter-tz (view ldnc-iter-desc)))
                    diff-src-iter-c-tz (when src-iter-c-tz (view-tz src-iter-tz (view ldnc-iter-desc)))
                    diff-weights (mem-alloc weights-size)
                    fused-diff-weights-tz (cudnn-tensor fact false diff-weights 0 (view fused-weights-desc))
                    post-diff-weights-tz (if post-process-diff?
                                           (cudnn-tensor fact (view fused-weights-desc))
                                           fused-diff-weights-tz)
                    diff-weights-tz (cudnn-tensor fact false diff-weights 0 (view weights-desc))
                    diff-weights-iter-tz (cudnn-tensor fact false diff-weights weights-offset (view weights-desc))
                    diff-bias-tz (cudnn-tensor fact false diff-weights bias-offset (view bias-desc))]
        (memcpy-host! seq-lengths dev-seq-lengths)
        (->CUDnnRnnTraining fact cudnn-hdl this dev-seq-lengths
                            src-conn src-iter-tz src-iter-c-tz
                            bias-tz fused-weights-tz weights-tz weights-iter-tz
                            dst-tz dst-iter-tz dst-iter-c-tz
                            (view rnn-desc) (view rnn-src-desc) (view rnn-dst-desc) (view iter-desc)
                            diff-dst-tz diff-dst-iter-tz diff-dst-iter-c-tz
                            diff-src-conn diff-src-iter-tz diff-src-iter-c-tz
                            fused-diff-weights-tz post-diff-weights-tz
                            diff-weights-tz diff-weights-iter-tz diff-bias-tz
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
        src-shape (shape src-desc)
        [T N src-ch] src-shape
        dst-shape (shape dst-desc)
        [_ _ dst-ch] dst-shape
        dirs (direction-count dir)
        seq-lengths (int-array (repeat N T))
        ldnc-iter-shape [lrs dirs N dst-ch]
        iter-shape [(* (long lrs) dirs) N dst-ch]
        weights-shape [lrs dirs src-ch gts dst-ch]
        default-strd (with-release [md (memory-desc weights-shape :float :ldgoi)] (layout md))
        weights-offset (get default-strd 0)
        weights-stride (update default-strd 0 (partial * 2))
        bias-shape [lrs dirs gts dst-ch]
        fused-weights-shape [lrs dirs (* 2 (long dst-ch)) gts dst-ch]
        fused-stride (with-release [md (memory-desc fused-weights-shape :float :ldgoi)] (layout md))]
    (let-release [src-desc (desc src-desc)
                  dst-desc (desc dst-desc)
                  dtype (data-type dst-desc)
                  weights-type (or weights-type dtype)
                  ldnc-iter-desc (cudnn-tensor-desc ldnc-iter-shape dtype :nchw)
                  iter-desc (cudnn-tensor-desc iter-shape dtype :nchw)
                  fused-weights-desc (cudnn-tensor-desc fused-weights-shape dtype fused-stride)
                  weights-desc (cudnn-tensor-desc weights-shape dtype weights-stride)
                  bias-desc (cudnn-tensor-desc bias-shape dtype :nchw)
                  rnn-desc (rnn-descriptor :standard mode :single :unidirectional :linear
                                           :float :float :default src-ch dst-ch dst-ch lrs
                                           nil :padded-io-enabled)
                  rnn-src-desc (rnn-data-descriptor :float :seq-mayor-packed src-ch seq-lengths 0.0)
                  rnn-dst-desc (rnn-data-descriptor :float :seq-mayor-packed dst-ch seq-lengths 0.0)]
      (let [weights-size (rnn-weights-space-size cudnn-hdl rnn-desc)
            [inf-work-size inf-reserve-size] (rnn-temp-space-size cudnn-hdl rnn-desc rnn-src-desc :inference)
            [train-work-size train-reserve-size] (rnn-temp-space-size cudnn-hdl rnn-desc rnn-src-desc :training)]
        (->CUDnnRnnBlueprint fact cudnn-hdl rnn-desc weights-size
                             (max 1 (long inf-work-size)) (max 1 (long inf-reserve-size))
                             (max 1 (long train-work-size)) (max 1 (long train-reserve-size))
                             seq-lengths weights-offset (apply * fused-weights-shape)
                             rnn-src-desc rnn-dst-desc
                             src-desc fused-weights-desc weights-desc bias-desc dst-desc
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
