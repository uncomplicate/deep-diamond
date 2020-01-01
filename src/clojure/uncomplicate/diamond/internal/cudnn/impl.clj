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
            [uncomplicate.diamond.tensor :refer [TensorDescriptor]]
            [uncomplicate.diamond.internal.cudnn
             [constants :refer :all]
             [protocols :refer :all]])
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

(deftype CUTensorDescriptor [^cudnnTensorDescriptor td dims data-type strides]
  Object
  (hashCode [this]
    (hash (deref td)))
  (equals [this other]
    (= @td (extract other)))
  (toString [this]
    (format "#CUTensorDescriptor[0x%s]" (Long/toHexString (native-pointer @td))))
  Wrapper
  (extract [this]
    @td)
  DescProvider
  (desc [this]
    this)
  Releaseable
  (release [this]
    (locking td
      (when-let [d @td]
        (locking d
          (with-check cudnn-error
            (JCudnn/cudnnDestroyTensorDescriptor d)
            (vreset! td nil)))))
    true))

(defn get-tensor-nd-descriptor* ^long [^cudnnTensorDescriptor td
                                       ^ints data-type ^ints dims ^ints strides]
  (let [nbdims (int-array 1)]
    (with-check cudnn-error
      (JCudnn/cudnnGetTensorNdDescriptor td (alength dims) data-type nbdims dims strides)
      (aget nbdims 0))))

(extend-type cudnnTensorDescriptor
  Wrappable
  (wrap [td]
    (let [data-type (int-array 1)
          dims (int-array JCudnn/CUDNN_DIM_MAX)
          strides (int-array JCudnn/CUDNN_DIM_MAX)]
      (let [nbdims (get-tensor-nd-descriptor* td data-type dims strides)]
        (->CUTensorDescriptor (volatile! td) (vec (take nbdims dims))
                              (dec-data-type (aget data-type 0))
                              (vec (take nbdims strides)))))))

(defn tensor-descriptor* []
  (let [res (cudnnTensorDescriptor.)]
    (with-check cudnn-error
      (JCudnn/cudnnCreateTensorDescriptor res)
      res)))

(defn tensor-4d-descriptor*
  ([^cudnnTensorDescriptor td ^long format ^long data-type shape]
   (with-check cudnn-error
     (JCudnn/cudnnSetTensor4dDescriptor
      td data-type format (get shape 0 0) (get shape 1 1) (get shape 2 1) (get shape 3 1))
     td)))

(defn tensor-4d-descriptor-ex* [^cudnnTensorDescriptor td ^long data-type shape stride]
  (with-check cudnn-error
    (JCudnn/cudnnSetTensor4dDescriptorEx
     td data-type (get shape 0 0) (get shape 1 1) (get shape 2 1) (get shape 3 1)
     (get stride 0 0) (get stride 1 1) (get stride 2 1) (get stride 3 1))
    td))

(defn tensor-nd-descriptor*
  ([^cudnnTensorDescriptor td ^long data-type ^ints dims ^ints strides]
   (with-check cudnn-error
     (JCudnn/cudnnSetTensorNdDescriptor td data-type (alength dims) dims strides)
     td)))

(defn tensor-nd-descriptor-ex*
  ([^cudnnTensorDescriptor td ^long format ^long data-type ^ints dims]
   (with-check cudnn-error
     (JCudnn/cudnnSetTensorNdDescriptorEx td format data-type (alength dims) dims)
     td)))

(defn size*
  "Queries the tensor descriptor for its dimensions."
  ^long [td]
  (let [res (long-array 1)]
    (with-check cudnn-error
      (JCudnn/cudnnGetTensorSizeInBytes td res) (aget res 0))))

;;TODO wait for bugfix in JCuda 10.2
#_(defn set-tensor* [cudnn-handle td buf-ptr value-buf]
  (with-check cudnn-error
    (JCudnn/cudnnSetTensor cudnn-handle td buf-ptr value-buf)
    cudnn-handle))

(defn add-tensor* [cudnn-handle alpha desc-x buf-x beta desc-y buf-y]
  (with-check cudnn-error
    (JCudnn/cudnnAddTensor cudnn-handle alpha desc-x buf-x beta desc-y buf-y)
    cudnn-handle))

(defn transform-tensor* [cudnn-handle alpha desc-x buf-x beta desc-y buf-y]
  (with-check cudnn-error
    (JCudnn/cudnnTransformTensor cudnn-handle alpha desc-x buf-x beta desc-y buf-y)
    cudnn-handle))

(defn scale-tensor* [cudnn-handle td buf alpha]
  (with-check cudnn-error
    (JCudnn/cudnnScaleTensor cudnn-handle td buf alpha)
    cudnn-handle))
