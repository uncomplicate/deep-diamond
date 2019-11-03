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

   Supported engine kinds are `:cpu`, `:gpu`, and `:any`. The default kind is `:cpu`."
  ([^long id kind]
   (wrap (engine* id (enc-keyword dnnl-engine-kind kind))))
  ([^long id]
   (wrap (engine* id)))
  ([]
   (engine 0)))

(defn engine-count
  "Returns the number of physical engines of the specified `kind` (`:cpu`, `:gpu`, `:any`)."
  (^long []
   (engine-count*))
  (^long [kind]
   (engine-count* (enc-keyword dnnl-engine-kind kind))))

(defn engine-kind
  "Returns the kind of an engine as a keyword. Typical values are `:gpu` and `:cpu`."
  ([eng]
   (dec-engine-kind (engine-kind* (extract eng)))))
