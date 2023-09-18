;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.diamond.internal.dnnl.impl
  (:require [uncomplicate.commons
             [core :refer [Releaseable release let-release with-release Info Wrapper Wrappable wrap
                           extract info Viewable view Bytes bytesize Entries sizeof* sizeof size]]
             [utils :refer [dragan-says-ex]]]
            [uncomplicate.clojure-cpp
             :refer [null? pointer int-pointer long-pointer long-ptr pointer-vec
                     get-entry put-entry! fill! PointerCreator pointer-pointer get-pointer
                     capacity! memcpy! byte-pointer]]
            [uncomplicate.diamond.internal.utils :refer [extend-dnnl-pointer]]
            [uncomplicate.diamond.internal.dnnl
             [protocols :refer :all]
             [constants :refer :all]])
  (:import [org.bytedeco.javacpp PointerPointer LongPointer FloatPointer]
           org.bytedeco.dnnl.global.dnnl
           [org.bytedeco.dnnl dnnl_engine dnnl_stream dnnl_primitive_desc
            dnnl_primitive dnnl_exec_arg_t dnnl_memory_desc dnnl_memory
            dnnl_primitive_attr]))

(defn dnnl-error
  ([^long err-code details]
   (let [err (dec-status err-code)]
     (ex-info (format "DNNL error %d %s." err-code err)
              {:code err-code :error err :type :dnnl-error :details details})))
  ([err-code]
   (dnnl-error err-code nil)))

(defmacro with-check
  ([status form]
   `(let [status# ~status
          form# ~form]
      (if (= 0 status#)
        form#
        (throw (dnnl-error status# (if (satisfies? Info form#) (info form#) (str form#)))))))
  ([status form details]
   `(let [status# ~status
          form# ~form]
      (if (= 0 status#)
        form#
        (throw (dnnl-error status# ~details))))))

(extend-type nil
  DescProvider
  (desc [_]
    nil))

;; ===================== Engine ========================================================

(extend-dnnl-pointer dnnl_engine dnnl/dnnl_engine_destroy dnnl-error)

(defn engine*
  ([^long id ^long runtime]
   (let-release [res (dnnl_engine.)]
     (with-check
       (dnnl/dnnl_engine_create res runtime id)
       res)))
  ([^long id]
   (let-release [res (dnnl_engine.)]
     (with-check (dnnl/dnnl_engine_create res dnnl/dnnl_cpu id) res))))

(defn engine-count*
  (^long []
   (dnnl/dnnl_engine_get_count dnnl/dnnl_cpu))
  (^long [^long runtime]
   (dnnl/dnnl_engine_get_count runtime)))

(defn engine-kind*
  (^long [^dnnl_engine eng]
   (let [kind (int-array 1)]
     (with-check
       (dnnl/dnnl_engine_get_kind eng kind)
       (aget kind 0)))))

;; ===================== Stream ========================================================

(extend-dnnl-pointer dnnl_stream dnnl/dnnl_stream_destroy dnnl-error)

(defn stream*
  ([^dnnl_engine eng]
   (stream* eng dnnl/dnnl_stream_default_flags))
  ([^dnnl_engine eng ^long flags]
   (let-release [strm (dnnl_stream.)]
     (with-check
       (dnnl/dnnl_stream_create strm eng flags)
       strm))))

(defn wait* [^dnnl_stream strm]
  (with-check (dnnl/dnnl_stream_wait strm) strm))

;; ===================== Primitive descriptor ===========================================

(extend-dnnl-pointer dnnl_primitive_desc dnnl/dnnl_primitive_desc_destroy dnnl-error)

(extend-type dnnl_primitive_desc
  DnnlCloneable
  (clone [this]
    (let-release [pd (dnnl_primitive_desc.)]
      (dnnl/dnnl_primitive_desc_clone pd this))))

(defn query-md*
  ([^dnnl_primitive_desc pd ^long what ^long index]
   (dnnl/dnnl_primitive_desc_query_md pd what index))
  ([^dnnl_primitive_desc pd ^long what]
   (query-md* pd what 0)))

;; ===================== Primitive ======================================================

(extend-dnnl-pointer dnnl_primitive dnnl/dnnl_primitive_destroy dnnl-error)

(defn primitive* [^dnnl_primitive_desc pd]
  (let-release [p (dnnl_primitive.)]
    (with-check (dnnl/dnnl_primitive_create p pd) p)))

(defn execute* [strm p ^dnnl_exec_arg_t args]
  (with-check
    (dnnl/dnnl_primitive_execute p strm (.capacity args) (.position args 0))
    strm))

(defn args* [^dnnl_exec_arg_t args ^long i ^long arg-key arg]
  (doto (.position args i)
    (.arg arg-key)
    (.memory arg))
  args)

;; ===================== Memory =========================================================

(extend-type java.lang.Long
  BlockedDesc
  (memory-desc* [tag dims data-type]
    (let-release [res (dnnl_memory_desc.)]
      (with-check
        (dnnl/dnnl_memory_desc_create_with_tag res (size dims) (long-ptr dims)
                                               (int data-type) tag)
        res
        {:tag (dec-format tag)
         :dims (pointer-vec dims)
         :data-type (dec-data-type data-type)}))))

(extend-type java.lang.Integer
  BlockedDesc
  (memory-desc* [tag dims data-type]
    (let-release [res (dnnl_memory_desc.)]
      (with-check
        (dnnl/dnnl_memory_desc_create_with_tag res (size dims) (long-ptr dims)
                                               (int data-type) tag)
        res
        {:tag (dec-format tag)
         :dims (pointer-vec dims)
         :data-type (dec-data-type data-type)}))))

(extend-type LongPointer
  BlockedDesc
  (memory-desc* [strides dims data-type]
    (let-release [res (dnnl_memory_desc.)]
      (with-check
        (dnnl/dnnl_memory_desc_create_with_strides res (size dims) (long-ptr dims)
                                                   (int data-type) strides)
        res
        {:strides (pointer-vec strides)
         :dims (pointer-vec dims)
         :data-type (dec-data-type data-type)}))))

(defn data-type* ^long [^dnnl_memory_desc mem-desc]
  (with-release [res (int-pointer 1)]
    (with-check (dnnl/dnnl_memory_desc_query mem-desc dnnl/dnnl_query_data_type res)
      (get-entry res 0))))

(defn ndims* ^long [^dnnl_memory_desc mem-desc]
  (let-release [ndims (int-pointer 1)]
    (with-check (dnnl/dnnl_memory_desc_query mem-desc dnnl/dnnl_query_ndims_s32 ndims)
      (get-entry ndims 0))))

(defn dims* [^dnnl_memory_desc mem-desc]
  (with-release [res (pointer-pointer 1)]
    (with-check (dnnl/dnnl_memory_desc_query mem-desc dnnl/dnnl_query_dims res)
      (capacity! (get-pointer (get-entry res 0) :long 0) (ndims* mem-desc)))))

(defn strides* [^dnnl_memory_desc mem-desc]
  (with-release [res (pointer-pointer 1)]
    (with-check (dnnl/dnnl_memory_desc_query mem-desc dnnl/dnnl_query_strides res)
      (capacity! (get-pointer (get-entry res 0) :long 0) (ndims* mem-desc)))))

(defn submemory-desc*
  ([^dnnl_memory_desc parent-desc ^LongPointer dims ^LongPointer offsets]
   (let-release [res (dnnl_memory_desc.)]
     (with-check
       (dnnl/dnnl_memory_desc_create_submemory res parent-desc dims offsets)
       res)))
  ([^dnnl_memory_desc parent-desc ^long n]
   (let [ndims (ndims* parent-desc)]
     (with-release [dims (long-pointer ndims)
                    strides (fill! (long-pointer ndims) 0)]
       (put-entry! (memcpy! (dims* parent-desc) dims (* Long/BYTES ndims)) 0 n)
       (submemory-desc* parent-desc dims strides)))))

(extend-dnnl-pointer dnnl_memory_desc dnnl/dnnl_memory_desc_destroy dnnl-error)

(extend-type dnnl_memory_desc
  DescProvider
  (desc [this]
    this)
  Viewable
  (view [this]
    (let-release [res (dnnl_memory_desc.)]
      (with-check
        (dnnl/dnnl_memory_desc_clone res this)
        res)))
  Bytes
  (bytesize* [this]
    (dnnl/dnnl_memory_desc_get_size this)))

(extend-type dnnl_memory
  Releaseable
  (release [_]
    (dragan-says-ex "You should never directly release dnn_memory. Please use MemoryImpl!")))

(deftype MemoryImpl [^dnnl_memory mem mem-desc data master]
  Releaseable
  (release [this]
    (locking mem
      (when-not (null? mem)
        (with-check (dnnl/dnnl_memory_destroy mem)
          (do (.deallocate mem)
              (.setNull mem)
              (when master
                (release data))))))
    true)
  Wrapper
  (extract [this]
    (if-not (null? mem) mem nil))
  DescProvider
  (desc [this]
    mem-desc)
  PointerCreator
  (pointer* [this]
    (if-not (or (null? mem) (null? data)) data nil))
  (pointer* [this i]
    (if-not (or (null? mem) (null? data)) (pointer data i) nil))
  Bytes
  (bytesize* [_]
    (dnnl/dnnl_memory_desc_get_size mem-desc))
  Entries
  (sizeof* [_]
    (sizeof* data))
  (size* [this]
    (quot (bytesize this) (sizeof data) )))

(defn memory* [^dnnl_memory_desc desc ^dnnl_engine eng data master]
  (let-release [mem (dnnl_memory.)
                data-pointer (pointer data 0)]
    (with-check (dnnl/dnnl_memory_create mem desc eng (byte-pointer data-pointer))
      (->MemoryImpl mem desc data-pointer master))))

(defn get-engine* [^dnnl_memory mem]
  (let-release [res (dnnl_engine.)]
    (with-check (dnnl/dnnl_memory_get_engine mem res) res)))

;; ===================== Eltwise  =========================================================

(defn eltwise-forward* [^dnnl_engine eng prop-kind alg-kind ^dnnl_memory_desc mem-desc
                        alpha beta ^dnnl_primitive_attr attr]
  (let-release [eltw-desc (dnnl_primitive_desc.)]
    (with-check
      (dnnl/dnnl_eltwise_forward_primitive_desc_create
       eltw-desc eng (int prop-kind) (int alg-kind) mem-desc mem-desc (float alpha) (float beta) attr)
      eltw-desc)))

(defn eltwise-backward* [^dnnl_engine eng alg-kind
                         ^dnnl_memory_desc diff-data-desc ^dnnl_memory_desc data-desc
                         alpha beta
                         ^dnnl_primitive_desc hint-fwd-pd ^dnnl_primitive_attr attr]
  (let-release [eltw-desc (dnnl_primitive_desc.)]
    (with-check
      (dnnl/dnnl_eltwise_backward_primitive_desc_create
       eltw-desc eng (int alg-kind) diff-data-desc diff-data-desc data-desc
       (float alpha) (float beta) hint-fwd-pd attr)
      eltw-desc)))

;; ======================= Sum ============================================================

(defn sum* [^dnnl_engine eng ^dnnl_memory_desc dst ^FloatPointer scales ^dnnl_memory_desc src
            ^dnnl_primitive_attr attr]
  (let-release [pd (dnnl_primitive_desc.)]
    (with-check
      (dnnl/dnnl_sum_primitive_desc_create pd eng dst (size scales) scales src attr)
      pd)))

(defn sum-pp* [^dnnl_engine eng ^dnnl_memory_desc dst ^FloatPointer scales ^PointerPointer srcs
               ^dnnl_primitive_attr attr]
  (with-release [pds (pointer-pointer 1)]
    (put-entry! pds 0 (dnnl_primitive_desc.))
    (with-check
      (dnnl/dnnl_sum_primitive_desc_create (.position pds 0) eng dst (size scales)
                                           scales (.position srcs 0) attr)
      (dnnl_primitive_desc. (.get pds 0)))))

;; ======================= Binary ============================================================

(defn binary* [^dnnl_engine eng alg-kind ^dnnl_memory_desc src0 ^dnnl_memory_desc src1
               ^dnnl_memory_desc dst ^dnnl_primitive_attr attr]
  (let-release [binary-desc (dnnl_primitive_desc.)]
    (with-check
      (dnnl/dnnl_binary_primitive_desc_create binary-desc eng (int alg-kind) src0 src1 dst attr)
      binary-desc)))

;; ======================= Reorder ========================================================

(defn reorder* [^dnnl_memory_desc input ^dnnl_engine input-eng
                ^dnnl_memory_desc output ^dnnl_engine output-eng]
  (let-release [pd (dnnl_primitive_desc.)]
    (with-check (dnnl/dnnl_reorder_primitive_desc_create pd input input-eng output output-eng nil)
      pd)))

;; ======================== Inner Product =======================================================

(defn inner-product-forward*
  [^dnnl_engine eng prop-kind ^dnnl_memory_desc src-desc ^dnnl_memory_desc weights-desc
   ^dnnl_memory_desc bias-desc ^dnnl_memory_desc dst-desc  ^dnnl_primitive_attr attr]
  (let-release [ip-desc (dnnl_primitive_desc.)]
    (with-check
      (dnnl/dnnl_inner_product_forward_primitive_desc_create
       ip-desc eng (int prop-kind) src-desc weights-desc bias-desc dst-desc attr)
      ip-desc)))

(defn inner-product-backward-data*
  [^dnnl_engine eng ^dnnl_memory_desc diff-src-desc ^dnnl_memory_desc weights-desc
   ^dnnl_memory_desc diff-dst-desc ^dnnl_primitive_desc hint-fwd-pd ^dnnl_primitive_attr attr]
  (let-release [ip-desc (dnnl_primitive_desc.)]
    (with-check
      (dnnl/dnnl_inner_product_backward_data_primitive_desc_create
       ip-desc eng diff-src-desc weights-desc diff-dst-desc hint-fwd-pd attr)
      ip-desc)))

(defn inner-product-backward-weights*
  [^dnnl_engine eng ^dnnl_memory_desc src-desc ^dnnl_memory_desc diff-weights-desc
   ^dnnl_memory_desc diff-bias-desc ^dnnl_memory_desc diff-dst-desc
   ^dnnl_primitive_desc hint-fwd-pd ^dnnl_primitive_attr attr]
  (let-release [ip-desc (dnnl_primitive_desc.)]
    (with-check
      (dnnl/dnnl_inner_product_backward_weights_primitive_desc_create
       ip-desc eng src-desc diff-weights-desc diff-bias-desc diff-dst-desc hint-fwd-pd attr)
      ip-desc)))

;; =========================== Softmax ==========================================

(defn softmax-forward* [^dnnl_engine eng prop-kind alg-kind ^dnnl_memory_desc mem-desc
                        axis ^dnnl_primitive_attr attr]
  (let-release [softmax-desc (dnnl_primitive_desc.)]
    (with-check
      (dnnl/dnnl_softmax_forward_primitive_desc_create
       softmax-desc eng (int prop-kind) (int alg-kind) mem-desc mem-desc (int axis) attr)
      softmax-desc)))

(defn softmax-backward* [^dnnl_engine eng alg-kind ^dnnl_memory_desc diff-data-desc
                         ^dnnl_memory_desc data-desc axis
                         ^dnnl_primitive_desc hint-fwd-pd ^dnnl_primitive_attr attr]
  (let-release [softmax-desc (dnnl_primitive_desc.)]
    (with-check
      (dnnl/dnnl_softmax_backward_primitive_desc_create
       softmax-desc eng (int alg-kind) diff-data-desc diff-data-desc data-desc (int axis) hint-fwd-pd attr)
      softmax-desc)))

;; ======================= Convolution ====================================================

(defn convolution-forward*
  [^dnnl_engine eng prop-kind alg-kind
   ^dnnl_memory_desc src-desc
   ^dnnl_memory_desc weights-desc ^dnnl_memory_desc bias-desc
   ^dnnl_memory_desc dst-desc
   ^LongPointer strides ^LongPointer dilates ^LongPointer padding-l ^LongPointer padding-r
   ^dnnl_primitive_attr attr]
  (let-release [conv-desc (dnnl_primitive_desc.)]
    (with-check
      (dnnl/dnnl_convolution_forward_primitive_desc_create
       conv-desc eng (int prop-kind) (int alg-kind) src-desc weights-desc bias-desc dst-desc
       strides dilates padding-l padding-r attr)
      conv-desc)))

(defn convolution-backward-data*
  [^dnnl_engine eng alg-kind
   ^dnnl_memory_desc diff-src-desc ^dnnl_memory_desc weights-desc ^dnnl_memory_desc diff-dst-desc
   ^LongPointer strides ^LongPointer dilates ^LongPointer padding-l ^LongPointer padding-r
   ^dnnl_primitive_desc hint-fwd-pd ^dnnl_primitive_attr attr]
  (let-release [conv-desc (dnnl_primitive_desc.)]
    (with-check
      (dnnl/dnnl_convolution_backward_data_primitive_desc_create
       conv-desc eng (int alg-kind) diff-src-desc weights-desc diff-dst-desc
       strides dilates padding-l padding-r hint-fwd-pd attr)
      conv-desc)))

(defn convolution-backward-weights*
  [^dnnl_engine eng alg-kind
   ^dnnl_memory_desc src-desc ^dnnl_memory_desc diff-weights-desc
   ^dnnl_memory_desc diff-bias-desc ^dnnl_memory_desc diff-dst-desc
   ^LongPointer strides ^LongPointer dilates ^LongPointer padding-l ^LongPointer padding-r
   ^dnnl_primitive_desc hint-fwd-pd ^dnnl_primitive_attr attr]
  (let-release [conv-desc (dnnl_primitive_desc.)]
    (with-check
      (dnnl/dnnl_convolution_backward_weights_primitive_desc_create
       conv-desc eng (int alg-kind) src-desc diff-weights-desc diff-bias-desc diff-dst-desc
       strides dilates padding-l padding-r hint-fwd-pd attr)
      conv-desc)))

(defn deconvolution-forward*
  [^dnnl_engine eng prop-kind alg-kind
   ^dnnl_memory_desc src-desc
   ^dnnl_memory_desc weights-desc ^dnnl_memory_desc bias-desc
   ^dnnl_memory_desc dst-desc
   ^LongPointer strides ^LongPointer dilates ^LongPointer padding-l ^LongPointer padding-r
   ^dnnl_primitive_attr attr]
  (let-release [conv-desc (dnnl_primitive_desc.)]
    (with-check
      (dnnl/dnnl_deconvolution_forward_primitive_desc_create
       conv-desc eng (int prop-kind) (int alg-kind) src-desc weights-desc bias-desc dst-desc
       strides dilates padding-l padding-r attr)
      conv-desc)))

(defn deconvolution-backward-data*
  [^dnnl_engine eng alg-kind
   ^dnnl_memory_desc diff-src-desc ^dnnl_memory_desc weights-desc ^dnnl_memory_desc diff-dst-desc
   ^LongPointer strides ^LongPointer dilates ^LongPointer padding-l ^LongPointer padding-r
   ^dnnl_primitive_desc hint-fwd-pd ^dnnl_primitive_attr attr]
  (let-release [conv-desc (dnnl_primitive_desc.)]
    (with-check
      (dnnl/dnnl_deconvolution_backward_data_primitive_desc_create
       conv-desc eng (int alg-kind) diff-src-desc weights-desc diff-dst-desc
       strides dilates padding-l padding-r hint-fwd-pd attr)
      conv-desc)))

(defn deconvolution-backward-weights*
  [^dnnl_engine eng alg-kind
   ^dnnl_memory_desc src-desc ^dnnl_memory_desc diff-weights-desc
   ^dnnl_memory_desc diff-bias-desc ^dnnl_memory_desc diff-dst-desc
   ^LongPointer strides ^LongPointer dilates ^LongPointer padding-l ^LongPointer padding-r
   ^dnnl_primitive_desc hint-fwd-pd ^dnnl_primitive_attr attr]
  (let-release [conv-desc (dnnl_primitive_desc.)]
    (with-check
      (dnnl/dnnl_deconvolution_backward_weights_primitive_desc_create
       conv-desc eng (int alg-kind) src-desc diff-weights-desc diff-bias-desc diff-dst-desc
       strides dilates padding-l padding-r hint-fwd-pd attr)
      conv-desc)))

;; ======================== Pooling ================================================================

(defn pooling-forward*
  [^dnnl_engine eng prop-kind alg-kind
   ^dnnl_memory_desc src-desc ^dnnl_memory_desc dst-desc
   ^LongPointer strides ^LongPointer kernel ^LongPointer
   dilates ^LongPointer padding-l ^LongPointer padding-r
   ^dnnl_primitive_attr attr]
  (let-release [pool-desc (dnnl_primitive_desc.)]
    (with-check
      (dnnl/dnnl_pooling_forward_primitive_desc_create
       pool-desc eng (int prop-kind) (int alg-kind) src-desc dst-desc
       strides kernel dilates padding-l padding-r attr)
      pool-desc)))

(defn pooling-backward*
  [^dnnl_engine eng alg-kind
   ^dnnl_memory_desc diff-src-desc ^dnnl_memory_desc diff-dst-desc
   ^LongPointer strides ^LongPointer kernel
   ^LongPointer dilates ^LongPointer padding-l ^LongPointer padding-r
   ^dnnl_primitive_desc hint-fwd-pd ^dnnl_primitive_attr attr]
  (let-release [pool-desc (dnnl_primitive_desc.)]
    (with-check
      (dnnl/dnnl_pooling_backward_primitive_desc_create pool-desc eng (int alg-kind)
                                            diff-src-desc diff-dst-desc
                                            strides kernel dilates padding-l padding-r hint-fwd-pd attr)
      pool-desc)))

;; ======================== Batch Normalization ===================================================

(defn batch-normalization-forward*
  [^dnnl_engine eng prop-kind ^dnnl_memory_desc data-desc epsilon flags ^dnnl_primitive_attr attr]
  (let-release [bnrm-desc (dnnl_primitive_desc.)]
    (with-check
      (dnnl/dnnl_batch_normalization_forward_primitive_desc_create
       bnrm-desc eng (int prop-kind) data-desc data-desc (float epsilon) (int flags) attr)
      bnrm-desc)))

(defn batch-normalization-backward*
  [^dnnl_engine eng prop-kind ^dnnl_memory_desc diff-data-desc ^dnnl_memory_desc data-desc epsilon flags
   ^dnnl_primitive_desc hint-fwd-pd ^dnnl_primitive_attr attr]
  (let-release [bnrm-desc (dnnl_primitive_desc.)]
    (with-check
      (dnnl/dnnl_batch_normalization_backward_primitive_desc_create
       bnrm-desc eng (int prop-kind) diff-data-desc diff-data-desc data-desc
       (float epsilon) (int flags) hint-fwd-pd attr)
      bnrm-desc)))

;; ======================= Reduction ========================================================

(defn reduction* [^dnnl_engine eng alg-kind ^dnnl_memory_desc src-desc ^dnnl_memory_desc dst-desc
                  p epsilon ^dnnl_primitive_attr attr]
  (let-release [rd (dnnl_primitive_desc.)]
    (with-check (dnnl/dnnl_reduction_primitive_desc_create
                 rd eng (int alg-kind) src-desc dst-desc (float p) (float epsilon) attr)
      rd)))

;; ======================= Concat  ========================================================

(defn concat*
  ([^dnnl_engine eng ^dnnl_memory_desc dst concat-dimension ^dnnl_memory_desc src
    ^dnnl_primitive_attr attr]
   (let-release [pd (dnnl_primitive_desc.)]
     (with-check
       (dnnl/dnnl_concat_primitive_desc_create pd eng dst 1 (int concat-dimension) src attr)
       pd)))
  ([^dnnl_engine eng ^dnnl_memory_desc dst n concat-dimension ^PointerPointer srcs
    ^dnnl_primitive_attr attr]
   (with-release [pds (pointer-pointer 1)]
     (put-entry! pds 0 (dnnl_primitive_desc.))
     (with-check
       (dnnl/dnnl_concat_primitive_desc_create (.position pds 0) eng dst (int n) (int concat-dimension)
                                               (.position srcs 0) attr)
       (dnnl_primitive_desc. (.get pds 0))))))

;; ======================= RNN ============================================================

(defn vanilla-rnn-forward* [^dnnl_engine eng prop-kind activation direction
                            ^dnnl_memory_desc src-desc ^dnnl_memory_desc src-iter-desc
                            ^dnnl_memory_desc weights-desc ^dnnl_memory_desc weights-iter-desc
                            ^dnnl_memory_desc bias-desc ^dnnl_memory_desc dst-desc
                            ^dnnl_memory_desc dst-iter-desc alpha beta
                            ^dnnl_primitive_attr attr]
  (let-release [rnn-desc (dnnl_primitive_desc.)]
    (with-check
      (dnnl/dnnl_vanilla_rnn_forward_primitive_desc_create
       rnn-desc eng (int prop-kind) (int activation) (int direction)
       src-desc src-iter-desc weights-desc weights-iter-desc bias-desc
       dst-desc dst-iter-desc dnnl/dnnl_rnn_flags_undef (float alpha) (float beta) attr)
      rnn-desc)))

(defn vanilla-rnn-backward* [^dnnl_engine eng activation direction
                             ^dnnl_memory_desc src-desc ^dnnl_memory_desc src-iter-desc
                             ^dnnl_memory_desc weights-desc ^dnnl_memory_desc weights-iter-desc
                             ^dnnl_memory_desc bias-desc
                             ^dnnl_memory_desc dst-desc ^dnnl_memory_desc dst-iter-desc
                             ^dnnl_memory_desc diff-src-desc ^dnnl_memory_desc diff-src-iter-desc
                             ^dnnl_memory_desc diff-weights-desc ^dnnl_memory_desc diff-weights-iter-desc
                             ^dnnl_memory_desc diff-bias-desc
                             ^dnnl_memory_desc diff-dst-desc ^dnnl_memory_desc diff-dst-iter-desc
                             [alpha beta]
                             ^dnnl_primitive_desc hint-fwd-pd ^dnnl_primitive_attr attr]
  (let-release [rnn-desc (dnnl_primitive_desc.)]
    (with-check
      (dnnl/dnnl_vanilla_rnn_backward_primitive_desc_create
       rnn-desc eng dnnl/dnnl_backward (int activation) (int direction)
       src-desc src-iter-desc weights-desc weights-iter-desc bias-desc
       dst-desc dst-iter-desc
       diff-src-desc diff-src-iter-desc
       diff-weights-desc diff-weights-iter-desc diff-bias-desc
       diff-dst-desc diff-dst-iter-desc
       dnnl/dnnl_rnn_flags_undef (float alpha) (float beta) hint-fwd-pd nil)
      rnn-desc)))

;; ======================= LSTM ============================================================

(defn lstm-forward* [^dnnl_engine eng prop-kind direction ^dnnl_memory_desc src-desc
                     ^dnnl_memory_desc src-iter-desc ^dnnl_memory_desc src-iter-c-desc
                     ^dnnl_memory_desc weights-desc ^dnnl_memory_desc weights-iter-desc
                     ^dnnl_memory_desc weights-peephole-desc ^dnnl_memory_desc weights-projection-desc
                     ^dnnl_memory_desc bias-desc ^dnnl_memory_desc dst-desc
                     ^dnnl_memory_desc dst-iter-desc ^dnnl_memory_desc dst-iter-c-desc
                     ^dnnl_primitive_attr attr]
  (let-release [lstm-desc (dnnl_primitive_desc.)]
    (with-check
      (dnnl/dnnl_lstm_forward_primitive_desc_create
       lstm-desc eng (int prop-kind) (int direction) src-desc src-iter-desc src-iter-c-desc
       weights-desc weights-iter-desc weights-peephole-desc weights-projection-desc
       bias-desc dst-desc dst-iter-desc dst-iter-c-desc dnnl/dnnl_rnn_flags_undef attr)
      lstm-desc)))

(defn lstm-backward* [^dnnl_engine eng direction
                      ^dnnl_memory_desc src-desc ^dnnl_memory_desc src-iter-desc
                      ^dnnl_memory_desc src-iter-c-desc
                      weights-iter-peephole-projection ^dnnl_memory_desc bias-desc
                      ^dnnl_memory_desc dst-desc ^dnnl_memory_desc dst-iter-desc
                      ^dnnl_memory_desc dst-iter-c-desc
                      ^dnnl_memory_desc diff-src-desc ^dnnl_memory_desc diff-src-iter-desc
                      ^dnnl_memory_desc diff-src-iter-c-desc
                      diff-weights-iter-peephole-projection ^dnnl_memory_desc diff-bias-desc
                      ^dnnl_memory_desc diff-dst-desc ^dnnl_memory_desc diff-dst-iter-desc
                      ^dnnl_memory_desc diff-dst-iter-c-desc
                      ^dnnl_primitive_desc hint-fwd-pd ^dnnl_primitive_attr attr]
  (let-release [lstm-desc (dnnl_primitive_desc.)]
    (let [[^dnnl_memory_desc weights-desc ^dnnl_memory_desc weights-iter-desc
           ^dnnl_memory_desc weights-peephole-desc ^dnnl_memory_desc weights-projection-desc]
          weights-iter-peephole-projection
          [^dnnl_memory_desc diff-weights-desc ^dnnl_memory_desc diff-weights-iter-desc
           ^dnnl_memory_desc diff-weights-peephole-desc ^dnnl_memory_desc diff-weights-projection-desc]
          diff-weights-iter-peephole-projection]
      (with-check
        (dnnl/dnnl_lstm_backward_primitive_desc_create
         lstm-desc eng dnnl/dnnl_backward (int direction)
         src-desc src-iter-desc src-iter-c-desc
         weights-desc weights-iter-desc weights-peephole-desc weights-projection-desc bias-desc
         dst-desc dst-iter-desc dst-iter-c-desc
         diff-src-desc diff-src-iter-desc diff-src-iter-c-desc
         diff-weights-desc diff-weights-iter-desc
         diff-weights-peephole-desc diff-weights-projection-desc diff-bias-desc
         diff-dst-desc diff-dst-iter-desc diff-dst-iter-c-desc
         dnnl/dnnl_rnn_flags_undef hint-fwd-pd attr)
        lstm-desc))))

;; ======================= GRU ============================================================

(defn gru-forward* [^dnnl_engine eng prop-kind direction
                    ^dnnl_memory_desc src-desc ^dnnl_memory_desc src-iter-desc
                    ^dnnl_memory_desc weights-desc ^dnnl_memory_desc weights-iter-desc
                    ^dnnl_memory_desc bias-desc
                    ^dnnl_memory_desc dst-desc ^dnnl_memory_desc dst-iter-desc
                    ^dnnl_primitive_attr attr]
  (let-release [gru-desc (dnnl_primitive_desc.)]
    (with-check
      (dnnl/dnnl_gru_forward_primitive_desc_create gru-desc eng (int prop-kind) (int direction)
                                       src-desc src-iter-desc
                                       weights-desc weights-iter-desc bias-desc
                                       dst-desc dst-iter-desc
                                       dnnl/dnnl_rnn_flags_undef attr)
      gru-desc)))

(defn gru-backward* [^dnnl_engine eng direction
                     ^dnnl_memory_desc src-desc ^dnnl_memory_desc src-iter-desc
                     ^dnnl_memory_desc weights-desc ^dnnl_memory_desc weights-iter-desc
                     ^dnnl_memory_desc bias-desc
                     ^dnnl_memory_desc dst-desc ^dnnl_memory_desc dst-iter-desc
                     ^dnnl_memory_desc diff-src-desc ^dnnl_memory_desc diff-src-iter-desc
                     ^dnnl_memory_desc diff-weights-desc ^dnnl_memory_desc diff-weights-iter-desc
                     ^dnnl_memory_desc diff-bias-desc
                     ^dnnl_memory_desc diff-dst-desc ^dnnl_memory_desc diff-dst-iter-desc
                     ^dnnl_primitive_desc hint-fwd-pd ^dnnl_primitive_attr attr]
  (let-release [rnn-desc (dnnl_primitive_desc.)]
    (with-check
      (dnnl/dnnl_gru_backward_primitive_desc_create rnn-desc eng dnnl/dnnl_backward (int direction)
                                                    src-desc src-iter-desc
                                                    weights-desc weights-iter-desc bias-desc
                                                    dst-desc dst-iter-desc
                                                    diff-src-desc diff-src-iter-desc
                                                    diff-weights-desc diff-weights-iter-desc diff-bias-desc
                                                    diff-dst-desc diff-dst-iter-desc
                                                    dnnl/dnnl_rnn_flags_undef hint-fwd-pd attr)
      rnn-desc)))
