;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.diamond.internal.cudnn.impl
  (:require [uncomplicate.commons
             [core :refer [Releaseable release with-release let-release Info
                           Wrapper Wrappable wrap extract]]
             [utils :refer [with-check dragan-says-ex]]]
            [uncomplicate.clojurecuda.internal.impl :refer [native-pointer]]
            [uncomplicate.diamond.internal.cudnn.protocols :refer :all])
  (:import java.nio.ByteBuffer
           jcuda.runtime.cudaStream_t
           jcuda.driver.CUstream
           [jcuda.jcudnn JCudnn cudnnHandle cudnnStatus cudnnTensorDescriptor]))

(defn cudnn-error [^long err-code details]
  (let [err (cudnnStatus/stringFor err-code)]
    (ex-info (format "cuDNN error: %s." err)
             {:name err :code err-code :type :cudnn-error :details details})))

(defmacro ^:private deftype-wrapper [name release-method]
  (let [name-str (str name)]
    `(deftype ~name [ref#]
       Object
       (hashCode [this#]
         (hash (deref ref#)))
       (equals [this# other#]
         (= (deref ref#) (extract other#)))
       (toString [this#]
         (format "#%s[0x%s]" ~name-str
                 (Long/toHexString (native-pointer (deref ref#)))))
       Wrapper
       (extract [this#]
         (deref ref#))
       Releaseable
       (release [this#]
         (locking ref#
           (when-let [d# (deref ref#)]
             (locking d#
               (with-check cudnn-error (~release-method d#) (vreset! ref# nil)))))
         true))))

;; =========================== cuDNN Handle =================================

(deftype-wrapper CUDnnHandle JCudnn/cudnnDestroy)

(extend-type cudnnHandle
  Wrappable
  (wrap [handle]
    (->CUDnnHandle (volatile! handle))))

(defn cudnn-handle*
  "Creates a cuDNN context handler on the specific `stream`."
  [^CUstream hstream]
  (let [handle (cudnnHandle.)
        cuda-stream (cudaStream_t. hstream)]
    (with-check cudnn-error (JCudnn/cudnnCreate handle)
      (with-check cudnn-error (JCudnn/cudnnSetStream handle cuda-stream) handle))))

(defn get-cudnn-stream* [handle]
  (let [res (cudaStream_t.)]
    (with-check cudnn-error (JCudnn/cudnnGetStream handle res) (CUstream. res))))

;; =========================== Tensor Descriptor ============================

(deftype-wrapper CUTensorDescriptor JCudnn/cudnnDestroyTensorDescriptor)

(extend-type cudnnTensorDescriptor
  Wrappable
  (wrap [td]
    (->CUTensorDescriptor (volatile! td))))

(defn tensor-descriptor* []
  (let [res (cudnnTensorDescriptor.)]
    (with-check cudnn-error
      (JCudnn/cudnnCreateTensorDescriptor res)
      res)))

(defn tensor-4d-descriptor* [^cudnnTensorDescriptor td
                             ^ints shape ^long data-type ^long format]
  (with-check cudnn-error
    (JCudnn/cudnnSetTensor4dDescriptor
     td format data-type (aget shape 0) (aget shape 1) (aget shape 2) (aget shape 3))
    td))
