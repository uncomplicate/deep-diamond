(ns uncomplicate.diamond.internal.dnnl
  (:require [uncomplicate.commons
             [core :refer [let-release with-release]]
             [utils :refer [enc-keyword direct-buffer capacity dragan-says-ex mask]]]
            [uncomplicate.diamond.internal.dnnl
             [impl :refer :all]
             [constants :refer :all]
             [protocols :refer :all]])
  (:import org.bytedeco.javacpp.Pointer
           org.bytedeco.dnnl.global.dnnl
           [org.bytedeco.dnnl dnnl_engine dnnl_memory_desc_t]))

;; ===================== Engine ===============================================

(defn engine
  "Creates an engine for the device `id` of the specified keyword `kind`.

   Supported engine kinds are `:cpu`, `:gpu`, and `:any`. The default kind is `:cpu`.
   Engine has to be `release`d.

  Throws an ExceptionInfo if the `id` does not correspond to a physical device
  or if `kind` is not supported."
  ([^long id kind]
   (wrap (engine* id (enc-keyword dnnl-engine-kind kind))))
  ([^long id]
   (wrap (engine* id)))
  ([]
   (engine 0)))

(defn engine-count
  "Returns the number of physical engines of the specified `kind` (`:cpu`, `:gpu`, `:any`).

  Throws an ExceptionInfo if `kind` is not supported."
  (^long []
   (engine-count*))
  (^long [kind]
   (engine-count* (enc-keyword dnnl-engine-kind kind))))

(defn engine-kind
  "Returns the kind of an engine as a keyword. Typical values are `:gpu` and `:cpu`.

  Throws an ExceptionInfo if `kind` is not supported."
  ([eng]
   (dec-engine-kind (engine-kind* (extract eng)))))

;; ===================== Stream ===============================================

(defn stream
  "Creates a stream for executing primitive operations for engine `eng`.

  Stream execution can be further specified by `flags`, defined in the
  [[constants/dnnl-stream-flags]].
  Stream has to be `release`d."
  [eng & flags]
  (wrap (if flags
          (stream* (extract eng) (mask dnnl-stream-flags flags))
          (stream* (extract eng)))))

(defn wait!
  "Waits until stream `s` completes execution of all queued operations."
  [strm]
  (wait* (extract strm))
  strm)

(defn execute!
  "Queues the operation primitive `p` for execution in stream `strm`.

  Returns `strm`. Throws an ExceptionInfo if the DNNL stream is not valid,
  or the primitive cannot be executed."
  [strm p args]
  (execute* (extract strm) (extract p) args)
  strm)

;; ===================== Desc =================================================

(defn primitive-kind [desc]
  (dec-primitive-kind (primitive-kind* desc)))

(defn primitive-desc
  ([eng desc]
   (wrap (primitive-desc* desc (extract eng))))
  ([eng desc hint-pd]
   (wrap (primitive-desc* desc (extract eng) (extract hint-pd))))
  ([eng desc hint-pd attr]
   (wrap (primitive-desc* desc (extract eng) (extract hint-pd) (extract attr)))))

;; ===================== Primitive ============================================

(defn primitive [pd]
  (wrap (primitive* (extract pd))))

;; ===================== Memory ===============================================

(defn memory-desc
  ([dims data-type format]
   (memory-desc* (if (keyword? format)
                   (enc-keyword dnnl-format format)
                   (long-array format))
                 (long-array dims) (enc-keyword dnnl-data-type data-type)))
  ([dims format]
   (memory-desc dims :float format))
  ([dims]
   (memory-desc dims :float :any)))

(defn submemory-desc
  ([parent-desc dims offsets]
   (submemory-desc* (desc parent-desc) (long-array dims) (long-array offsets)))
  ([parent-desc dim-a]
   (submemory-desc* (desc parent-desc) dim-a)))

(defn equal-desc? [x y]
  (= 1 (dnnl/dnnl_memory_desc_equal (desc x) (desc y))))

(defn data-type [mem-desc]
  (dec-data-type (data-type* (desc mem-desc))))

(defn ndims ^long [mem-desc]
  (.ndims ^dnnl_memory_desc_t (desc mem-desc)))

(defn dims [mem-desc]
  (vec (dims* (desc mem-desc))))

(defn size ^long [mem-desc]
  (dnnl/dnnl_memory_desc_get_size (desc mem-desc)))

(defn strides [mem-desc]
  (vec (strides* (desc mem-desc))))

(defn memory
  ([eng mem-desc buf master]
   (if (<= (size (desc mem-desc)) (capacity buf))
     (memory* (desc mem-desc) (extract eng) buf master)
     (dragan-says-ex "The buffer has to be large enough for mem-desc"
                     {:size (size (desc mem-desc)) :capacity (capacity buf)})))
  ([eng mem-desc buf]
   (memory eng mem-desc buf false))
  ([eng mem-desc]
   (let-release [buf (direct-buffer (size (desc mem-desc)))]
     (memory* (desc mem-desc) (extract eng) buf true))))

(defn offset! [mem ^long n]
  (let [p (ptr mem)]
    (if (and (<= 0 n) (<= n (.capacity ^Pointer p)))
      (with-check (dnnl/dnnl_memory_set_data_handle
                   (extract mem) (.position ^Pointer p n))
        mem)
      (dragan-says-ex "There is not enough capacity in the underlying buffer for this offset."
                      {:n n :requested n :available (.capacity ^Pointer p)}))))
