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
           [org.bytedeco.dnnl dnnl_engine]))

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
