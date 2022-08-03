;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.diamond.internal.cudnn.rnn
  (:require [uncomplicate.commons.core
             :refer [Releaseable release let-release with-release Info info view]]
            [uncomplicate.clojurecuda.core :refer [mem-alloc memset! memcpy-host!]]
            [uncomplicate.neanderthal
             [core :refer [axpby! transfer! scal! entry!]]
             [block :refer [buffer initialize]]]
            [uncomplicate.neanderthal.internal.api :refer [flow]]
            [uncomplicate.diamond.tensor :as tz
             :refer [Transfer input output connector TensorDescriptor shape layout view-tz]]
            [uncomplicate.diamond.internal
             [protocols
              :refer [Parameters ParametersSeq DescriptorProvider DiamondFactoryProvider
                      DiffParameters Backprop forward DiffTransfer diff-input diff-output LinearBackprop
                      backward-diff train-desc Initializable init Workspace inf-ws-size train-ws-size
                      RnnParameters DiffRnnParameters batch-index]]
             [utils :refer [default-strides direction-count]]]
            [uncomplicate.diamond.internal.dnnl.core :refer [memory-desc]]
            [uncomplicate.diamond.internal.cudnn
             [protocols :refer :all]
             [core :refer :all]
             [tensor :refer [cudnn-tensor cudnn-tensor-desc cudnn-transformer]]]
            [uncomplicate.diamond.internal.neanderthal.rnn :refer [->RnnLayerBlueprint]])
  (:import [clojure.lang IFn AFn]))

;; ================================ RNN ====================================================

(deftype CUDnnRnnInference [fact cudnn-hdl bluep srcs dsts dev-seq-lengths
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
    (initialize bias-tz (buffer bias-tz) 0.0)
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
    dsts)
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(deftype CUDnnRnnTraining [fact cudnn-hdl bluep srcs dsts diff-dsts dev-seq-lengths
                           src-conn src-iter-tz src-iter-c-tz
                           bias-tz weights-tz weights-iter-tz
                           dst-tz dst-iter-tz dst-iter-c-tz
                           rnn-desc rnn-src-desc rnn-dst-desc iter-desc
                           diff-dst-tz diff-dst-iter-tz diff-dst-iter-c-tz
                           diff-src-tz diff-src-iter-tz diff-src-iter-c-tz
                           diff-src-transformer diff-src-iter-transformer diff-src-iter-c-transformer
                           diff-weights-tz diff-weights-iter-tz diff-bias-tz
                           post-diff-weights-tz post-diff-weights-iter-tz
                           weights-mem diff-weights-mem work reserve]
  Releaseable
  (release [_];;TODO
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
    (release diff-weights-tz)
    (release post-diff-weights-tz)
    (release diff-bias-tz)
    (release weights-mem)
    (release work)
    (release reserve))
  Info
  (info [this]
    {:bias (info bias-tz)
     :weights (info weights-tz)
     :dst (info dst-tz)
     :diff-weights (info diff-weights-tz)
     :weights-iter (info weights-iter-tz)})
  (info [this info-type]
    (case info-type
      :bias (info bias-tz)
      :weights (info weights-tz)
      :dst (info dst-tz)
      :diff-weights (info diff-weights-tz)
      :weights-iter (info weights-iter-tz)
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
    post-diff-weights-tz);;TODO think about fusing weights-tz and weights-iter-tz
  DiffRnnParameters
  (diff-weights-iter [_]
    post-diff-weights-iter-tz)
  Initializable
  (init [this init-fn]
    (init-fn weights-tz)
    (init-fn weights-iter-tz)
    (initialize bias-tz (buffer bias-tz) 0.0)
    (memset! diff-weights-mem 0.0)
    (when src-iter-tz
      (initialize src-iter-tz (buffer src-iter-tz) 0.0))
    (when src-iter-c-tz
      (initialize src-iter-c-tz (buffer src-iter-c-tz) 0.0))
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
                  rnn-src-desc (buffer diff-src-tz)
                  iter-desc (when src-iter-tz (buffer src-iter-tz))
                  (when diff-dst-iter-tz (buffer diff-dst-iter-tz))
                  (when diff-src-iter-tz (buffer diff-src-iter-tz))
                  iter-desc (when src-iter-c-tz (buffer src-iter-c-tz))
                  (when diff-dst-iter-c-tz (buffer diff-dst-iter-c-tz))
                  (when diff-src-iter-c-tz (buffer diff-src-iter-c-tz))
                  weights-mem work reserve)
    (memset! diff-weights-mem 0.0)
    (rnn-bwd-weights cudnn-hdl rnn-desc :add dev-seq-lengths
                     rnn-src-desc (buffer (output src-conn))
                     iter-desc (when src-iter-tz (buffer src-iter-tz))
                     rnn-dst-desc (buffer dst-tz)
                     diff-weights-mem work reserve)
    (if (= 0.0 scal-g)
      (when-not (= 1.0 scal-diff-w)
        (scal! scal-diff-w diff-weights-tz)
        (scal! scal-diff-w diff-weights-iter-tz))
      (do (axpby! scal-diff-w diff-weights-tz scal-g post-diff-weights-tz)
          (axpby! scal-diff-w diff-weights-iter-tz scal-g post-diff-weights-iter-tz)))
    (axpby! scal-diff-b diff-bias-tz scal-b bias-tz)
    (diff-src-transformer)
    (when diff-src-iter-transformer (diff-src-iter-transformer))
    (when diff-src-iter-c-transformer (diff-src-iter-c-transformer))
    this))

(deftype CUDnnRnnBlueprint [fact cudnn-hdl rnn-desc ^long weights-size
                            ^long inf-work-size ^long inf-reserve-size
                            ^long train-work-size ^long train-reserve-size
                            ^ints seq-lengths ^long weights-offset ^long bias-offset
                            rnn-src-desc rnn-dst-desc
                            src-desc weights-desc bias-desc dst-desc
                            ldnc-iter-desc iter-desc src-iter? dst-iter? iter-c?]
  Releaseable
  (release [_];;TODO
    (release rnn-desc)
    (release src-desc)
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
  TensorDescriptor
  (shape [_]
    (shape dst-desc))
  (data-type [_]
    (data-type dst-desc))
  (layout [_]
    (strides dst-desc))
  Workspace
  (inf-ws-size [this]
    inf-work-size)
  (train-ws-size [this]
    train-work-size)
  IFn
  (invoke [this srcs]
    (let [[src-tz src-iter-tz src-iter-c-tz] (if (sequential? srcs) srcs [srcs nil nil])]
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
                    reserve (mem-alloc inf-reserve-size)
                    srcs (if src-iter-tz
                           (if src-iter-c-tz
                             [(input src-conn) src-iter-tz src-iter-c-tz]
                             [(input src-conn) src-iter-tz])
                           (input src-conn))
                    dsts (if dst-iter-tz
                           (if dst-iter-c-tz
                             [dst-tz dst-iter-tz dst-iter-c-tz]
                             [dst-tz dst-iter-tz])
                           dst-tz)]
        (memcpy-host! seq-lengths dev-seq-lengths)
        (->CUDnnRnnInference fact cudnn-hdl this srcs dsts dev-seq-lengths
                             src-conn src-iter-tz src-iter-c-tz
                             bias-tz weights-tz weights-iter-tz
                             dst-tz dst-iter-tz dst-iter-c-tz
                             rnn-desc rnn-src-desc rnn-dst-desc (view iter-desc)
                             weights work reserve))))
  (invoke [this srcs _ post-process-diff?];;TODO keep in mind that some of source tensors might have to be views!
    (let [[src-tz src-iter-tz src-iter-c-tz] (if (sequential? srcs) srcs [srcs nil nil])]
      (let-release [src-conn (connector src-tz src-desc)
                    dev-seq-lengths (mem-alloc (* Float/BYTES (alength seq-lengths)))
                    weights (mem-alloc weights-size)
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
                    diff-src-tz (cudnn-tensor fact (view src-desc) 1)
                    diff-src-iter-tz (when src-iter-tz (cudnn-tensor fact (view ldnc-iter-desc)))
                    diff-src-iter-c-tz (when src-iter-c-tz (cudnn-tensor fact (view ldnc-iter-desc)))
                    diff-src-transformer  (cudnn-transformer cudnn-hdl diff-src-tz src-tz)
                    diff-src-iter-transformer (when diff-src-iter-tz
                                                (cudnn-transformer cudnn-hdl diff-src-iter-tz src-iter-tz))
                    diff-src-iter-c-transformer (when diff-src-iter-c-tz
                                                  (cudnn-transformer cudnn-hdl diff-src-iter-c-tz src-iter-c-tz))
                    diff-weights (mem-alloc weights-size)
                    diff-weights-tz (cudnn-tensor fact false diff-weights 0 (view weights-desc))
                    diff-weights-iter-tz (cudnn-tensor fact false diff-weights weights-offset (view weights-desc))
                    post-diff-weights-tz (if post-process-diff?
                                           (cudnn-tensor fact (view weights-desc))
                                           diff-weights-tz)
                    post-diff-weights-iter-tz (if post-process-diff?
                                                (cudnn-tensor fact (view weights-desc))
                                                diff-weights-iter-tz)
                    diff-bias-tz (cudnn-tensor fact false diff-weights bias-offset (view bias-desc))
                    srcs (if src-iter-tz
                           (if src-iter-c-tz
                             [(input src-conn) src-iter-tz src-iter-c-tz]
                             [(input src-conn) src-iter-tz])
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
        (memcpy-host! seq-lengths dev-seq-lengths)
        (->CUDnnRnnTraining fact cudnn-hdl this srcs dsts diff-dsts dev-seq-lengths
                            src-conn src-iter-tz src-iter-c-tz
                            bias-tz weights-tz weights-iter-tz
                            dst-tz dst-iter-tz dst-iter-c-tz
                            rnn-desc rnn-src-desc rnn-dst-desc (view iter-desc)
                            diff-dst-tz diff-dst-iter-tz diff-dst-iter-c-tz
                            diff-src-tz diff-src-iter-tz diff-src-iter-c-tz
                            diff-src-transformer diff-src-iter-transformer diff-src-iter-c-transformer
                            diff-weights-tz diff-weights-iter-tz diff-bias-tz
                            post-diff-weights-tz post-diff-weights-iter-tz
                            weights diff-weights work reserve))))
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

;; ================================ Vanilla RNN =================================================

(defn cudnn-rnn-op-blueprint [gts mode fact cudnn-hdl src-desc dst-desc weights-type
                              dir lrs src-iter? dst-iter? iter-c?]
  (let [src-shape (shape src-desc)
        [T N src-ch] src-shape
        dst-shape (shape dst-desc)
        [_ _ dst-ch] dst-shape
        dirs (direction-count dir)
        ldnc-iter-shape [lrs dirs N dst-ch]
        iter-shape [(* (long lrs) dirs) N dst-ch]
        weights-shape [lrs dirs src-ch gts dst-ch]
        bias-shape [lrs dirs gts dst-ch]
        seq-lengths (int-array (repeat N T))
        default-strd (with-release [md (memory-desc weights-shape :float :ldgoi)] (layout md))
        weights-offset (get default-strd 0)
        weights-stride (update default-strd 0 (partial * 2))]
    (let-release [src-descnn (desc src-desc)
                  dst-desc (desc dst-desc)
                  dtype (data-type dst-desc)
                  ldnc-iter-desc (cudnn-tensor-desc ldnc-iter-shape dtype :nchw)
                  iter-desc (cudnn-tensor-desc iter-shape dtype :nchw)
                  weights-type (or weights-type dtype)
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
                             seq-lengths weights-offset (* 2 (long (apply * weights-shape)) )
                             rnn-src-desc rnn-dst-desc
                             src-desc weights-desc bias-desc dst-desc
                             ldnc-iter-desc iter-desc src-iter? dst-iter? iter-c?)))))

(defn cudnn-rnn-blueprint [fact cudnn-hdl src-desc dst-desc lrs activ weights-type src-iter? dst-iter?]
  (let-release [src-desc (cudnn-tensor-desc (shape src-desc) (or (tz/data-type src-desc) :float)
                                            (or (layout src-desc) :tnc))
                dst-desc (cudnn-tensor-desc (shape dst-desc)
                                            (or (tz/data-type dst-desc) (tz/data-type src-desc) :float)
                                            (or (layout dst-desc) :tnc))
                rnn-op-bluep (cudnn-rnn-op-blueprint 1 activ fact cudnn-hdl src-desc dst-desc weights-type
                                                     :unidirectional lrs
                                                     src-iter? dst-iter? false)]
    (->RnnLayerBlueprint fact :rnn rnn-op-bluep)))

;; TODO I'll have to see how to support weights-iter as part of global-weights to be able to
;; make DNNL and CUDA RNN uniform. Continue with that when you come around to cuda RNN after a pause.
;; It looks to me that fused weights and weights-iter would straighen out errors in cudnn...
