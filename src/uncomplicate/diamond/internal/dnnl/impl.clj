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
           [org.bytedeco.dnnl dnnl_engine dnnl_stream]))

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
