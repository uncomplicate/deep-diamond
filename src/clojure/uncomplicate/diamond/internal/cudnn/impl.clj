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
                           Wrapper Wrappable wrap extract info]]
             [utils :refer [dragan-says-ex]]]
            [uncomplicate.clojurecuda.internal
             [protocols :refer [size]]
             [impl :refer [native-pointer]]]
            [uncomplicate.diamond.internal.utils :refer [deftype-wrapper]]
            [uncomplicate.diamond.internal.cudnn
             [constants :refer :all]
             [protocols :refer :all]])
  (:import java.nio.ByteBuffer
           jcuda.runtime.cudaStream_t
           jcuda.driver.CUstream
           [jcuda.jcudnn JCudnn cudnnHandle cudnnStatus cudnnTensorDescriptor
            cudnnActivationDescriptor cudnnReduceTensorDescriptor cudnnIndicesType
            cudnnReduceTensorIndices cudnnReduceTensorOp]))

(defn cudnn-error [^long err-code details]
  (let [err (cudnnStatus/stringFor err-code)]
    (ex-info (format "cuDNN error: %s." err)
             {:name err :code err-code :type :cudnn-error :details details})))

(defmacro with-check
  ([status form]
   `(let [status# ~status
          form# ~form]
      (if (= 0 status#)
        form#
        (throw (cudnn-error status# (if (satisfies? Info form#) (info form#) form#)))))))

;; =========================== cuDNN Handle =================================

(deftype-wrapper CUDnnHandle JCudnn/cudnnDestroy cudnn-error)

(extend-type cudnnHandle
  Wrappable
  (wrap [hdl]
    (->CUDnnHandle (volatile! hdl))))

(defn cudnn-handle*
  "Creates a cuDNN context handle on the specific `hstream`."
  [^CUstream hstream]
  (let [hdl (cudnnHandle.)
        cuda-stream (cudaStream_t. hstream)]
    (with-check (JCudnn/cudnnCreate hdl)
      (with-check  (JCudnn/cudnnSetStream hdl cuda-stream) hdl))))

(defn get-cudnn-stream* [hdl]
  (let [res (cudaStream_t.)]
    (with-check (JCudnn/cudnnGetStream hdl res) (CUstream. res))))

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
  Info
  (info [td]
    {:class (class td)
     :device :cuda
     :shape (.dims td)
     :data-type (.data-type td)
     :layout (.strides td)})
  (info [td info-type]
    (case info-type
      :class (class td)
      :device :cuda
      :shape (.dims td)
      :data-type (.data-type td)
      :layout (.strides td)
      nil))
  Releaseable
  (release [this]
    (locking td
      (when-let [d @td]
        (locking d
          (with-check
            (JCudnn/cudnnDestroyTensorDescriptor d)
            (vreset! td nil)))))
    true))

(defn get-tensor-nd-descriptor* ^long [^cudnnTensorDescriptor td
                                       ^ints data-type ^ints dims ^ints strides]
  (let [nbdims (int-array 1)]
    (with-check
      (JCudnn/cudnnGetTensorNdDescriptor td (alength dims) data-type nbdims dims strides)
      (aget nbdims 0))))

(extend-type cudnnTensorDescriptor
  Info
  (info [td]
    (let [data-type (int-array 1)
          dims (int-array JCudnn/CUDNN_DIM_MAX)
          strides (int-array JCudnn/CUDNN_DIM_MAX)
          nbdims (get-tensor-nd-descriptor* td data-type dims strides)]
      {:class (class td)
       :device :cuda
       :shape (vec (take nbdims dims))
       :data-type (dec-data-type (aget data-type 0))
       :layout (vec (take nbdims strides))}))
  Wrappable
  (wrap [td]
    (let [data-type (int-array 1)
          dims (int-array JCudnn/CUDNN_DIM_MAX)
          strides (int-array JCudnn/CUDNN_DIM_MAX)
          nbdims (get-tensor-nd-descriptor* td data-type dims strides)]
      (->CUTensorDescriptor (volatile! td) (vec (take nbdims dims))
                            (dec-data-type (aget data-type 0))
                            (vec (take nbdims strides))))))

(defn tensor-descriptor* []
  (let [res (cudnnTensorDescriptor.)]
    (with-check
      (JCudnn/cudnnCreateTensorDescriptor res)
      res)))

(defn tensor-4d-descriptor*
  ([^cudnnTensorDescriptor td ^long format ^long data-type shape]
   (with-check
     (JCudnn/cudnnSetTensor4dDescriptor
      td data-type format (get shape 0 0) (get shape 1 1) (get shape 2 1) (get shape 3 1))
     td)))

(defn tensor-4d-descriptor-ex* [^cudnnTensorDescriptor td ^long data-type shape stride]
  (with-check
    (JCudnn/cudnnSetTensor4dDescriptorEx
     td data-type (get shape 0 0) (get shape 1 1) (get shape 2 1) (get shape 3 1)
     (get stride 0 0) (get stride 1 1) (get stride 2 1) (get stride 3 1))
    td))

(defn tensor-nd-descriptor*
  ([^cudnnTensorDescriptor td ^long data-type ^ints dims ^ints strides]
   (with-check
     (JCudnn/cudnnSetTensorNdDescriptor td data-type (alength dims) dims strides)
     td)))

(defn tensor-nd-descriptor-ex*
  ([^cudnnTensorDescriptor td ^long format ^long data-type ^ints dims]
   (with-check
     (JCudnn/cudnnSetTensorNdDescriptorEx td format data-type (alength dims) dims)
     td)))

(defn size*
  "Queries the tensor descriptor for its dimensions."
  ^long [td]
  (let [res (long-array 1)]
    (with-check
      (JCudnn/cudnnGetTensorSizeInBytes td res) (aget res 0))))

(defn set-tensor* [cudnn-handle td buf value]
  (with-check
    (JCudnn/cudnnSetTensor cudnn-handle td buf value)
    cudnn-handle))

(defn add-tensor* [cudnn-handle alpha desc-x buf-x beta desc-y buf-y]
  (with-check
    (JCudnn/cudnnAddTensor cudnn-handle alpha desc-x buf-x beta desc-y buf-y)
    cudnn-handle))

(defn transform-tensor* [cudnn-handle alpha desc-x buf-x beta desc-y buf-y]
  (with-check
    (JCudnn/cudnnTransformTensor cudnn-handle alpha desc-x buf-x beta desc-y buf-y)
    cudnn-handle))

(defn scale-tensor* [cudnn-handle td buf alpha]
  (with-check
    (JCudnn/cudnnScaleTensor cudnn-handle td buf alpha)
    cudnn-handle))

;; ======================= Activation ===================================

(deftype-wrapper CUDnnActivationDescriptor
  JCudnn/cudnnDestroyActivationDescriptor cudnn-error)

(extend-type cudnnActivationDescriptor
  Wrappable
  (wrap [ad]
    (->CUDnnActivationDescriptor (volatile! ad))))

(defn activation-descriptor*
  ([]
   (let [res (cudnnActivationDescriptor.)]
     (with-check
       (JCudnn/cudnnCreateActivationDescriptor res)
       res)))
  ([^cudnnActivationDescriptor ad ^long mode ^long relu-nan-opt ^double coef]
   (with-check
     (JCudnn/cudnnSetActivationDescriptor ad mode relu-nan-opt coef)
     ad)))

(defn get-activation-descriptor* [^cudnnActivationDescriptor ad
                                  ^ints mode ^ints relu-nan-opt ^doubles coef]
  (with-check
    (JCudnn/cudnnGetActivationDescriptor ad mode relu-nan-opt coef)
    ad))

(defn activation-forward* [cudnn-handle ad alpha desc-x buf-x beta desc-y buf-y]
  (with-check
    (JCudnn/cudnnActivationForward cudnn-handle ad alpha desc-x buf-x beta desc-y buf-y)
    cudnn-handle))

(defn activation-backward* [cudnn-handle ad
                            alpha desc-y buf-y desc-dy buf-dy
                            desc-x buf-x beta desc-dx buf-dx]
  (with-check
    (JCudnn/cudnnActivationBackward cudnn-handle ad
                                    alpha desc-y buf-y desc-dy buf-dy
                                    desc-x buf-x beta desc-dx buf-dx)
    cudnn-handle))

;; ========================== Reduce ===================================

(deftype-wrapper CUDnnReduceTensorDescriptor
  JCudnn/cudnnDestroyReduceTensorDescriptor cudnn-error)

(extend-type cudnnReduceTensorDescriptor
  Wrappable
  (wrap [rtd]
    (->CUDnnReduceTensorDescriptor (volatile! rtd))))

(defn reduce-tensor-descriptor*
  ([]
   (let [res (cudnnReduceTensorDescriptor.)]
     (with-check
       (JCudnn/cudnnCreateReduceTensorDescriptor res)
       res)))
  ([^cudnnReduceTensorDescriptor rtd ^long op ^long comp-type ^long nan-opt]
   (with-check
     (JCudnn/cudnnSetReduceTensorDescriptor rtd op comp-type nan-opt
                                            cudnnReduceTensorIndices/CUDNN_REDUCE_TENSOR_NO_INDICES
                                            cudnnIndicesType/CUDNN_32BIT_INDICES)
     rtd))
  ([^cudnnReduceTensorDescriptor rtd op comp-type nan-opt indices]
   (let [comp-type (int comp-type)]
     (with-check
       (JCudnn/cudnnSetReduceTensorDescriptor
        rtd (int op) (int comp-type) (int nan-opt)
        (if (or (= cudnnReduceTensorOp/CUDNN_REDUCE_TENSOR_AMAX comp-type)
                (= cudnnReduceTensorOp/CUDNN_REDUCE_TENSOR_MAX comp-type)
                (= cudnnReduceTensorOp/CUDNN_REDUCE_TENSOR_MIN comp-type))
          (int indices)
          cudnnReduceTensorIndices/CUDNN_REDUCE_TENSOR_NO_INDICES)
        cudnnIndicesType/CUDNN_32BIT_INDICES)
       rtd))))

(defn reduce-tensor* [cudnn-handle rtd
                      indices indices-size workspace workspace-size
                      alpha desc-x buf-x beta desc-y buf-y]
  (with-check
    (JCudnn/cudnnReduceTensor cudnn-handle rtd
                              indices indices-size workspace workspace-size
                              alpha desc-x buf-x beta desc-y buf-y)
    cudnn-handle))

(defn reduction-indices-size* ^long [cudnn-handle rtd desc-x desc-y]
  (let [size-arr (long-array 1)]
    (with-check
      (JCudnn/cudnnGetReductionIndicesSize cudnn-handle rtd desc-x desc-y size-arr)
      (aget size-arr 0))))

(defn reduction-workspace-size* ^long [cudnn-handle rtd desc-x desc-y]
  (let [size-arr (long-array 1)]
    (with-check
      (JCudnn/cudnnGetReductionWorkspaceSize cudnn-handle rtd desc-x desc-y size-arr)
      (aget size-arr 0))))
