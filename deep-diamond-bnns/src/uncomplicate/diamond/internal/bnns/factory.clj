;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.diamond.internal.bnns.factory
  (:require [uncomplicate.commons
             [core :refer [Releaseable release let-release view bytesize]]
             [utils :refer [dragan-says-ex mapped-buffer]]]
            [uncomplicate.clojure-cpp :refer [byte-pointer type-pointer]]
            [uncomplicate.neanderthal
             [native :refer [factory-by-type]]
             [block :refer [create-data-source buffer initialize!]]]
            [uncomplicate.neanderthal.internal.api :refer [FlowProvider]]
            [uncomplicate.neanderthal.internal.cpp.lapack :refer [with-lapack-check]]
            [uncomplicate.neanderthal.internal.cpp.accelerate.factory
             :refer [->FloatVectorEngine ->IntVectorEngine ->ByteVectorEngine]]
            [uncomplicate.diamond.tensor
             :refer [*diamond-factory* output shape data-type layout]]
            [uncomplicate.diamond.internal
             [protocols
              :refer [TensorFactory MappedTensorFactory DiamondFactoryProvider CostFactory
                      DnnFactory RnnFactory NeanderthalFactoryProvider diamond-factory]]
             [utils :refer [check-contiguous]]
             [cost :refer [quadratic-cost! mean-absolute-cost! crossentropy-cost!]]]
            [uncomplicate.diamond.internal.neanderthal.directed :refer [neanderthal-fc-blueprint]]
            [uncomplicate.diamond.internal.bnns
             [protocols :refer [desc]]
             [core :refer [nda-desc dims]]
             [tensor :refer [bnns-tensor bnns-transformer bnns-batcher bnns-shuffler]]
             [constants :refer [bnns-data-type-pointer]]
             [directed :refer [bnns-activ-blueprint bnns-universal-cost bnns-custom-cost]]]))

(def ^{:private true :const true} INEFFICIENT_OPERATION_MSG
  "This operation would be inefficient because it does not use BNNS capabilities.
  Please use dedicated tensor operations.")

(def ^{:private true :const true} UNSUPPORTED_DATA_TYPE
  "The requested data type is not supported on the BNNS platform.
Please contribute towards making it possible, or use on of the supported types.")

(deftype BnnsFactory [tensor-engines]
  DiamondFactoryProvider
  (diamond-factory [this]
    this)
  (native-diamond-factory [this]
    this)
  NeanderthalFactoryProvider
  (neanderthal-factory [_ dtype]
    (factory-by-type dtype))
  TensorFactory
  (create-tensor-desc [this shape dtype format]
    (nda-desc shape dtype format))
  (create-tensor-desc [this tz-desc]
    (desc tz-desc))
  (create-tensor [this tensor-desc init]
    (let-release [res (bnns-tensor this tensor-desc)]
      (when init
        (initialize! res (buffer res)))
      res))
  (create-tensor [this tensor-desc batch-index init]
    (let-release [res (bnns-tensor this tensor-desc batch-index)]
      (when init
        (initialize! res (buffer res)))
      res))
  (create-transformer [_ in-tz out-tz]
    (bnns-transformer in-tz out-tz)) ;;TODO I had to use direct tensors instead of view because otherwise bnns data wouldn't track offsets of the original tensors. See whether I can make this consistent in DNNL adn cuDNN...
  (create-shuffler [_ src-tz dst-tz]
    (bnns-shuffler (view src-tz) (view dst-tz)))
  (create-batcher [_ src-tz dst-tz mb-size]
    (bnns-batcher (view src-tz) (view dst-tz) mb-size))
  (tensor-engine [this dtype]
    (or (get tensor-engines dtype)
        (dragan-says-ex UNSUPPORTED_DATA_TYPE {:data-type dtype})))
  MappedTensorFactory
  (map-channel [this channel td flag offset-bytes n-index]
    (let [size (bytesize (desc td))]
      (let-release [buf ((bnns-data-type-pointer (data-type td))
                         (mapped-buffer channel offset-bytes size flag))]
        (bnns-tensor this td buf n-index true))))
  DnnFactory
  (activ-blueprint [this src-desc activ alpha beta]
    (bnns-activ-blueprint this src-desc activ alpha beta))
  (inner-product-blueprint [this src-desc dst-desc weights-type]
    (dragan-says-ex "BNNS engine does not implement the inner product blueprint."))
  (fc-blueprint [this src-desc dst-desc activ alpha beta weights-type]
    (neanderthal-fc-blueprint this src-desc dst-desc activ alpha beta weights-type))
  ;; (convolution-blueprint [this src-desc weights-desc dst-desc activ
  ;;                         strides padding dilation alpha beta]
  ;;   (dnnl-convolution-layer-blueprint this eng src-desc weights-desc dst-desc activ
  ;;                                     strides (mapv dec dilation) padding padding alpha beta))
  ;; (pooling-blueprint [this src-desc dst-desc algo strides kernel padding]
  ;;   (dnnl-pooling-blueprint this eng src-desc dst-desc algo strides kernel padding padding))
  ;; (gaussian-dropout-blueprint [this src-desc sd]
  ;;   (dnnl-gaussian-dropout-blueprint this src-desc sd))
  ;; (batch-norm-blueprint [this src-desc activ alpha beta]
  ;;   (dnnl-batch-norm-layer-blueprint this eng src-desc activ alpha beta))
  ;; (concat-blueprint [this src-descs conc-dim dst-type]
  ;;   (dnnl-concat-blueprint this eng src-descs conc-dim dst-type))
  ;; (branch-blueprint [this src-desc branch-dim dst-descs]
  ;;   (dnnl-branch-blueprint this eng src-desc branch-dim dst-descs))
  ;; (split-blueprint [this src-desc n]
  ;;   (dnnl-split-blueprint this eng src-desc n))
  ;; (sum-blueprint [this src-descs]
  ;;   (dnnl-sum-blueprint this eng src-descs))
  (create-workspace [_ byte-size]
    (create-data-source (factory-by-type :byte) (max 1 (long byte-size))))
  ;; RnnFactory
  ;; (rnn-op-blueprint [this src-desc dst-desc weights-type activ dir lrs src-iter? dst-iter?]
  ;;   (case activ
  ;;     :gru (dnnl-gru-op-blueprint this eng src-desc dst-desc weights-type dir lrs src-iter? dst-iter?)
  ;;     :lstm (dnnl-lstm-op-blueprint this eng src-desc dst-desc weights-type dir lrs src-iter? dst-iter?)
  ;;     (dnnl-rnn-op-blueprint this eng src-desc dst-desc weights-type activ 0.0 dir lrs src-iter? dst-iter?)))
  ;; (rnn-blueprint [fact src-desc dst-desc lrs activ alpha weights-type src-iter? dst-iter?]
  ;;   (case activ
  ;;     :gru (dnnl-gru-blueprint fact eng src-desc dst-desc lrs weights-type src-iter? dst-iter?)
  ;;     :lstm (dnnl-lstm-blueprint fact eng src-desc dst-desc lrs weights-type src-iter? dst-iter?)
  ;;     (dnnl-rnn-blueprint fact eng src-desc dst-desc lrs activ alpha weights-type src-iter? dst-iter?)))
  ;; (abbreviate-blueprint [fact src-desc dst-type]
  ;;   (dnnl-abbreviate-blueprint fact eng src-desc dst-type))
  CostFactory
  (quadratic-cost [this prev-layer train-tz]
    (bnns-universal-cost prev-layer train-tz quadratic-cost!))
  (mean-absolute-cost [this prev-layer train-tz]
    (bnns-universal-cost prev-layer train-tz mean-absolute-cost!))
  (crossentropy-cost [this prev-layer train-tz]
    (bnns-custom-cost prev-layer train-tz
                      (partial crossentropy-cost!
                               ((dims (output prev-layer)) 0)))))

(defn bnns-factory []
  (->BnnsFactory {:float (->FloatVectorEngine)
                  :int (->IntVectorEngine)
                  :byte (->ByteVectorEngine)
                  :uint8 (->ByteVectorEngine)}))
