(ns uncomplicate.diamond.internal.dnnl.impl
  (:require [uncomplicate.commons
             [core :refer [Releaseable release let-release with-release]]
             [utils :as cu :refer [dragan-says-ex]]]
            [uncomplicate.diamond.internal.dnnl
             [protocols :refer :all]
             [constants :refer :all]])
  (:import java.nio.ByteBuffer
           [org.bytedeco.javacpp Pointer PointerPointer]
           org.bytedeco.dnnl.global.dnnl
           [org.bytedeco.dnnl dnnl_engine dnnl_stream dnnl_primitive_desc
            dnnl_primitive dnnl_exec_arg_t dnnl_memory_desc_t dnnl_memory
            dnnl_primitive_desc const_dnnl_op_desc_t dnnl_primitive_attr
            dnnl_eltwise_desc_t]))

(defn error
  ([^long err-code details]
   (let [err (dec-status err-code)]
     (ex-info (format "DNNL error %d %s." err-code err)
              {:code err-code :error err :type :dnnl-error :details details})))
  ([err-code]
   (error err-code nil)))

(defmacro with-check
  ([status form]
   `(cu/with-check error ~status ~form)))

(extend-type nil
  Wrapper
  (extract [_]
    nil)
  Wrappable
  (wrap [this]
    nil))

(extend-type Pointer
  Releaseable
  (release [this]
    (.deallocate this)
    true))

(defmacro ^:private deftype-wrapper [name release-method]
  (let [name-str (str name)]
    `(deftype ~name [ref#]
       Object
       (hashCode [this#]
         (hash (deref ref#)))
       (equals [this# other#]
         (= (deref ref#) (extract other#)))
       (toString [this#]
         (format "#%s[%s]" ~name-str (deref ref#)))
       Wrapper
       (extract [this#]
         (deref ref#))
       Releaseable
       (release [this#]
         (locking ref#
           (when-let [d# (deref ref#)]
             (locking d#
               (with-check (~release-method d#) (vreset! ref# nil)))))
         true))))


;; ===================== Engine ========================================================

(deftype-wrapper Engine dnnl/dnnl_engine_destroy)

(extend-type dnnl_engine
  Wrappable
  (wrap [this]
    (->Engine (volatile! this))))

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

(deftype-wrapper Stream dnnl/dnnl_stream_destroy)

(extend-type dnnl_stream
  Wrappable
  (wrap [this]
    (->Stream (volatile! this))))

(defn stream*
  ([^dnnl_engine eng]
   (stream* eng dnnl/dnnl_stream_default_flags))
  ([^dnnl_engine eng ^long flags]
   (let-release [s (dnnl_stream.)]
     (with-check
       (dnnl/dnnl_stream_create s eng flags)
       s))))

(defn wait* [^dnnl_stream s]
  (with-check (dnnl/dnnl_stream_wait s) s))

;; ===================== Primitive descriptor ===========================================

(deftype-wrapper PrimitiveDesc dnnl/dnnl_primitive_desc_destroy)

(extend-type dnnl_primitive_desc
  Wrappable
  (wrap [this]
    (->PrimitiveDesc (volatile! this)))
  DnnlCloneable
  (clone [this]
    (let-release [pd (dnnl_primitive_desc.)]
      (dnnl/dnnl_primitive_desc_clone pd this))))

(defn query-md*
  ([pd ^long what ^long index]
   (dnnl/dnnl_primitive_desc_query_md pd what index))
  ([pd ^long what]
   (query-md* pd what 0)))

;; ===================== Primitive ======================================================

(deftype-wrapper Primitive dnnl/dnnl_primitive_destroy)

(extend-type dnnl_primitive
  Wrappable
  (wrap [this]
    (->Primitive (volatile! this))))

(defn primitive* [^dnnl_primitive_desc pd]
  (let-release [p (dnnl_primitive.)]
    (with-check (dnnl/dnnl_primitive_create p pd) p)))

(defn execute* [s p ^dnnl_exec_arg_t args]
  (with-check
    (dnnl/dnnl_primitive_execute p s (.capacity args) (.position args 0))
    s))

(defn args* [^dnnl_exec_arg_t args ^long i ^long arg-key arg]
  (doto (.position args i)
    (.arg arg-key)
    (.memory arg)))

;; ===================== Memory =========================================================

(extend-type dnnl_memory_desc_t
  DescProvider
  (desc [this]
    this))

(extend-type java.lang.Long
  BlockedDesc
  (memory-desc* [tag dims data-type]
    (let-release [res (dnnl_memory_desc_t.)]
      (with-check
        (dnnl/dnnl_memory_desc_init_by_tag res (alength ^longs dims) ^longs dims
                                           ^long data-type tag)
        res))))

(extend-type java.lang.Integer
  BlockedDesc
  (memory-desc* [tag dims data-type]
    (let-release [res (dnnl_memory_desc_t.)]
      (with-check
        (dnnl/dnnl_memory_desc_init_by_tag res (alength ^longs dims) ^longs dims
                                           ^long data-type tag)
        res))))

(extend-type (class (long-array 0))
  BlockedDesc
  (memory-desc* [strides dims data-type]
    (let-release [res (dnnl_memory_desc_t.)]
      (with-check
        (dnnl/dnnl_memory_desc_init_by_strides res (alength ^longs dims) ^longs dims
                                               ^long data-type ^longs strides)
        res))))

(defn data-type* ^long [^dnnl_memory_desc_t mem-desc]
  (.data_type mem-desc))

(defn dims* ^longs [^dnnl_memory_desc_t mem-desc]
  (let [dims (long-array (.ndims mem-desc))]
    (.get (.dims mem-desc) dims)
    dims))

(defn strides* [^dnnl_memory_desc_t mem-desc]
  (let [strides (.strides (.format_desc_blocking mem-desc))
        res (long-array (.ndims mem-desc))]
    (.get (.position strides 0) res)
    res))

(defn submemory-desc*
  ([^dnnl_memory_desc_t parent-desc ^longs dims ^longs offsets]
   (let-release [res (dnnl_memory_desc_t.)]
     (with-check
       (dnnl/dnnl_memory_desc_init_submemory res parent-desc dims offsets)
       res)))
  ([^dnnl_memory_desc_t parent-desc ^long n]
   (let [dims (dims* parent-desc)]
     (aset dims 0 n)
     (submemory-desc* parent-desc dims (long-array (alength dims))))))

(deftype MemoryImpl [vmem mem-desc d ^Pointer d-ptr master]
  Releaseable
  (release [this]
    (locking vmem
      (when-let [mem @vmem]
        (locking mem
          (with-check (dnnl/dnnl_memory_destroy mem)
            (do (vreset! vmem nil)
                (when master
                  (release d)
                  (.deallocate d-ptr))))))))
  Wrapper
  (extract [this]
    @vmem)
  DescProvider
  (desc [this]
    mem-desc)
  Memory
  (data [this]
    (if @vmem d nil))
  (ptr [this]
    (if @vmem d-ptr nil)))

(extend-type ByteBuffer
  PointerCreator
  (pointer [buf]
    (Pointer. ^ByteBuffer buf)))

(defn memory* [^dnnl_memory_desc_t desc ^dnnl_engine eng data master]
  (let-release [mem (dnnl_memory.)
                data-pointer (pointer data)]
    (with-check (dnnl/dnnl_memory_create mem desc eng ^Pointer data-pointer)
      (->MemoryImpl (volatile! mem) desc data data-pointer master))))

(defn get-engine* [^dnnl_memory mem]
  (let-release [res (dnnl_engine.)]
    (with-check (dnnl/dnnl_memory_get_engine mem res) res)))

;; ===================== Eltwise  =========================================================

(extend-type const_dnnl_op_desc_t
  PrimitiveDescCreator
  (primitive-desc*
    ([desc eng hint-pd]
     (let-release [pd (dnnl_primitive_desc.)]
       (with-check (dnnl/dnnl_primitive_desc_create pd desc nil
                                                    ^dnnl_engine eng
                                                    ^dnnl_primitive_desc hint-pd)
         pd)))
    ([desc eng hint-pd attr]
     (let-release [pd (dnnl_primitive_desc.)]
       (with-check (dnnl/dnnl_primitive_desc_create pd desc
                                                    ^dnnl_primitive_attr attr
                                                    ^dnnl_engine eng
                                                    ^dnnl_primitive_desc hint-pd)
         pd)))))

(extend-type dnnl_eltwise_desc_t
  PrimitiveDescCreator
  (primitive-desc*
    ([desc eng]
     (primitive-desc* (const_dnnl_op_desc_t. desc) eng nil))
    ([desc eng hint-pd]
     (primitive-desc* (const_dnnl_op_desc_t. desc) eng hint-pd)))
  PrimitiveKind
  (primitive-kind* [desc]
    (.primitive_kind desc)))

(defn eltwise-forward-desc* [prop-kind alg-kind mem-desc alpha beta]
  (let-release [eltw-desc (dnnl_eltwise_desc_t.)]
    (with-check
      (dnnl/dnnl_eltwise_forward_desc_init eltw-desc (int prop-kind) (int alg-kind)
                                           mem-desc (float alpha) (float beta))
      eltw-desc)))

(defn eltwise-backward-desc* [alg-kind diff-data-desc data-desc alpha beta]
  (let-release [eltw-desc (dnnl_eltwise_desc_t.)]
    (with-check
      (dnnl/dnnl_eltwise_backward_desc_init eltw-desc (int alg-kind) diff-data-desc
                                            data-desc (float alpha) (float beta))
      eltw-desc)))

;; ======================= Sum ============================================================

(defn sum* [^dnnl_memory_desc_t dst ^floats scales ^dnnl_memory_desc_t src ^dnnl_engine eng]
  (let-release [pd (dnnl_primitive_desc.)]
    (with-check
      (dnnl/dnnl_sum_primitive_desc_create pd dst (alength scales) scales
                                           (.position src 0) nil eng)
      pd)))

;; ======================= Reorder ========================================================

(defn reorder* [^dnnl_memory_desc_t input ^dnnl_engine input-eng
                ^dnnl_memory_desc_t output ^dnnl_engine output-eng]
  (let-release [pd (dnnl_primitive_desc.)]
    (with-check (dnnl/dnnl_reorder_primitive_desc_create
                 pd input input-eng output output-eng nil)
      pd)))
