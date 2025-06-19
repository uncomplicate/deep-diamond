;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.diamond.internal.dnnl.factory
  (:require [uncomplicate.commons
             [core :refer [Releaseable release let-release view bytesize]]
             [utils :refer [dragan-says-ex mapped-buffer]]]
            [uncomplicate.clojure-cpp :refer [byte-pointer type-pointer]]
            [uncomplicate.neanderthal.block :refer [create-data-source buffer initialize!]]
            [uncomplicate.neanderthal.internal.api :refer [FlowProvider]]
            [uncomplicate.neanderthal.internal.cpp.lapack :refer [with-lapack-check]]
            [uncomplicate.neanderthal.internal.cpp.mkl.factory
             :refer [->FloatVectorEngine ->IntVectorEngine ->ByteVectorEngine
                     mkl-double mkl-float mkl-long mkl-int mkl-short mkl-byte]]
            [uncomplicate.diamond.tensor
             :refer [*diamond-factory* output shape data-type layout]]
            [uncomplicate.diamond.internal
             [protocols
              :refer [TensorFactory MappedTensorFactory DiamondFactoryProvider CostFactory
                      DnnFactory RnnFactory NeanderthalFactoryProvider diamond-factory]]
             [utils :refer [check-contiguous]]
             [cost :refer [quadratic-cost! mean-absolute-cost! crossentropy-cost!]]]
            [uncomplicate.diamond.internal.dnnl
             [protocols :refer [desc DnnlEngineProvider]]
             [core :refer [memory-desc engine stream memory dims primitive-cache-capacity!]]
             [tensor :refer [dnnl-tensor dnnl-tensor* dnnl-transformer dnnl-batcher dnnl-shuffler]]
             [directed :refer [dnnl-sum-blueprint dnnl-activ-blueprint dnnl-inner-product-blueprint
                               dnnl-universal-cost dnnl-custom-cost dnnl-convolution-layer-blueprint
                               dnnl-split-blueprint dnnl-concat-blueprint dnnl-fc-blueprint
                               dnnl-gaussian-dropout-blueprint dnnl-batch-norm-layer-blueprint
                               dnnl-pooling-blueprint dnnl-branch-blueprint dnnl-sum-blueprint]]
             [rnn :refer [dnnl-rnn-op-blueprint dnnl-rnn-blueprint dnnl-abbreviate-blueprint
                          dnnl-lstm-op-blueprint dnnl-lstm-blueprint
                          dnnl-gru-op-blueprint dnnl-gru-blueprint]]]))

(def ^{:private true :const true} INEFFICIENT_OPERATION_MSG
  "This operation would be inefficient because it does not use DNNL capabilities.
  Please use dedicated tensor operations.")

(def ^{:private true :const true} UNSUPPORTED_DATA_TYPE
  "The requested data type is not supported on the DNNL platform.
Please contribute towards making it possible, or use on of the supported types.")

(defn factory-by-type [data-type]
  (case data-type
    :float mkl-float
    :double mkl-double
    :int mkl-int
    :long mkl-long
    :short mkl-short
    :byte mkl-byte
    :uint8 mkl-byte
    (cond
      (= Float/TYPE data-type) mkl-float
      (= Double/TYPE data-type) mkl-double
      (= Integer/TYPE data-type) mkl-int
      (= Long/TYPE data-type) mkl-long
      (= Short/TYPE data-type) mkl-short
      (= Byte/TYPE data-type) mkl-byte
      (= float data-type) mkl-float
      (= double data-type) mkl-double
      (= int data-type) mkl-int
      (= long data-type) mkl-long
      (= short data-type) mkl-short
      (= byte data-type) mkl-byte
      :default (dragan-says-ex "You requested a factory for an unsupported data type."
                               {:requested data-type
                                :available [:float :int :double :long :short :byte
                                            Float/TYPE Double/TYPE
                                            Integer/TYPE Long/TYPE Short/TYPE Byte/TYPE]}))))

(deftype DnnlFactory [eng strm master tensor-engines]
  Releaseable
  (release [_]
    (when master
      (release strm)
      (release eng))
    true)
  DiamondFactoryProvider
  (diamond-factory [this]
    this)
  (native-diamond-factory [this]
    this)
  FlowProvider
  (flow [_]
    strm)
  DnnlEngineProvider
  (dnnl-engine [_]
    eng)
  NeanderthalFactoryProvider
  (neanderthal-factory [_ dtype]
    (factory-by-type dtype))
  TensorFactory
  (create-tensor-desc [this shape dtype format]
    (memory-desc shape dtype format))
  (create-tensor-desc [this tz-desc]
    (desc tz-desc))
  (create-tensor [this tensor-desc init]
    (let-release [res (dnnl-tensor this tensor-desc)]
      (when init
        (initialize! res (buffer res)))
      res))
  (create-tensor [this tensor-desc batch-index init]
    (let-release [res (dnnl-tensor this tensor-desc batch-index)]
      (when init
        (initialize! res (buffer res)))
      res))
  (create-transformer [_ in-tz out-tz]
    (dnnl-transformer eng strm (view in-tz) (view out-tz)))
  (create-shuffler [_ src-tz dst-tz]
    (dnnl-shuffler eng strm (view src-tz) (view dst-tz)))
  (create-batcher [_ src-tz dst-tz mb-size]
    (dnnl-batcher eng strm (view src-tz) (view dst-tz) mb-size))
  (tensor-engine [this dtype]
    (or (get tensor-engines dtype)
        (dragan-says-ex UNSUPPORTED_DATA_TYPE {:data-type dtype})))
  MappedTensorFactory
  (map-channel [this channel td flag offset-bytes n-index]
    (let [size (bytesize (desc td))]
      (let-release [buf ((type-pointer (data-type td)) (mapped-buffer channel offset-bytes size flag))]
        (dnnl-tensor* this td buf n-index true))))
  DnnFactory
  (activ-blueprint [this src-desc activ alpha beta]
    (dnnl-activ-blueprint this eng src-desc activ alpha beta))
  (inner-product-blueprint [this src-desc dst-desc weights-type]
    (dnnl-inner-product-blueprint this eng src-desc dst-desc weights-type))
  (fc-blueprint [this src-desc dst-desc activ alpha beta weights-type]
    (dnnl-fc-blueprint this eng src-desc dst-desc activ alpha beta weights-type))
  (convolution-blueprint [this src-desc weights-desc dst-desc activ
                          strides padding dilation alpha beta]
    (dnnl-convolution-layer-blueprint this eng src-desc weights-desc dst-desc activ
                                      strides (mapv dec dilation) padding padding alpha beta))
  (pooling-blueprint [this src-desc dst-desc algo strides kernel padding]
    (dnnl-pooling-blueprint this eng src-desc dst-desc algo strides kernel padding padding))
  (gaussian-dropout-blueprint [this src-desc sd]
    (dnnl-gaussian-dropout-blueprint this src-desc sd))
  (batch-norm-blueprint [this src-desc activ alpha beta]
    (dnnl-batch-norm-layer-blueprint this eng src-desc activ alpha beta))
  (concat-blueprint [this src-descs conc-dim dst-type]
    (dnnl-concat-blueprint this eng src-descs conc-dim dst-type))
  (branch-blueprint [this src-desc branch-dim dst-descs]
    (dnnl-branch-blueprint this eng src-desc branch-dim dst-descs))
  (split-blueprint [this src-desc n]
    (dnnl-split-blueprint this eng src-desc n))
  (sum-blueprint [this src-descs]
    (dnnl-sum-blueprint this eng src-descs))
  (create-workspace [_ byte-size]
    (create-data-source mkl-byte (max 1 (long byte-size))))
  RnnFactory
  (rnn-op-blueprint [this src-desc dst-desc weights-type activ dir lrs src-iter? dst-iter?]
    (case activ
      :gru (dnnl-gru-op-blueprint this eng src-desc dst-desc weights-type dir lrs src-iter? dst-iter?)
      :lstm (dnnl-lstm-op-blueprint this eng src-desc dst-desc weights-type dir lrs src-iter? dst-iter?)
      (dnnl-rnn-op-blueprint this eng src-desc dst-desc weights-type activ 0.0 dir lrs src-iter? dst-iter?)))
  (rnn-blueprint [fact src-desc dst-desc lrs activ alpha weights-type src-iter? dst-iter?]
    (case activ
      :gru (dnnl-gru-blueprint fact eng src-desc dst-desc lrs weights-type src-iter? dst-iter?)
      :lstm (dnnl-lstm-blueprint fact eng src-desc dst-desc lrs weights-type src-iter? dst-iter?)
      (dnnl-rnn-blueprint fact eng src-desc dst-desc lrs activ alpha weights-type src-iter? dst-iter?)))
  (abbreviate-blueprint [fact src-desc dst-type]
    (dnnl-abbreviate-blueprint fact eng src-desc dst-type))
  CostFactory
  (quadratic-cost [this prev-layer train-tz]
    (dnnl-universal-cost eng strm prev-layer train-tz quadratic-cost!))
  (mean-absolute-cost [this prev-layer train-tz]
    (dnnl-universal-cost eng strm prev-layer train-tz mean-absolute-cost!))
  (crossentropy-cost [this prev-layer train-tz]
    (dnnl-custom-cost eng strm prev-layer train-tz
                      (partial crossentropy-cost!
                               ((dims (output prev-layer)) 0)))))

(primitive-cache-capacity! 0)

(defn dnnl-factory
  ([eng strm]
   (->DnnlFactory eng strm false {:float (->FloatVectorEngine)
                                  :int (->IntVectorEngine)
                                  :byte (->ByteVectorEngine)
                                  :uint8 (->ByteVectorEngine)}))
  ([]
   (let-release [eng (engine)
                 strm (stream eng)]
     (->DnnlFactory eng strm true {:float (->FloatVectorEngine)
                                   :int (->IntVectorEngine)
                                   :byte (->ByteVectorEngine)
                                   :uint8 (->ByteVectorEngine)}))))
