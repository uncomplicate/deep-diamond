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
            cudnnReduceTensorIndices cudnnReduceTensorOp cudnnConvolutionDescriptor
            cudnnFilterDescriptor cudnnConvolutionFwdPreference]))

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
  (info [td info-type]
    (case info-type
      :class (class td)
      :device :cuda
      :shape (.dims td)
      :data-type (.data-type td)
      :layout (.strides td)
      nil))
  (info [td]
    {:class (class td)
     :device :cuda
     :shape (.dims td)
     :data-type (.data-type td)
     :layout (.strides td)})
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
      td format data-type (get shape 0 0) (get shape 1 1) (get shape 2 1) (get shape 3 1))
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

;; ======================= Softmax ===================================

(defn softmax-forward* [cudnn-handle algo mode alpha desc-x buf-x beta desc-y buf-y]
  (with-check
    (JCudnn/cudnnSoftmaxForward cudnn-handle algo mode alpha desc-x buf-x beta desc-y buf-y)
    cudnn-handle))

(defn softmax-backward* [cudnn-handle algo mode
                         alpha desc-y buf-y desc-dy buf-dy
                         beta desc-dx buf-dx]
  (with-check
    (JCudnn/cudnnSoftmaxBackward cudnn-handle algo mode
                                 alpha desc-y buf-y desc-dy buf-dy
                                 beta desc-dx buf-dx)
    cudnn-handle))

;; ======================== Filter ==============================

(deftype CUFilterDescriptor [^cudnnFilterDescriptor fd
                             ^cudnnTensorDescriptor td
                             dims data-type format strides]
  Object
  (hashCode [this]
    (hash (deref td)))
  (equals [this other]
    (= @td (extract other)))
  (toString [this]
    (format "#CUFilterDescriptor[0x%s]" (Long/toHexString (native-pointer @td))))
  Wrapper
  (extract [this]
    @td)
  DescProvider
  (desc [this]
    this)
  Info
  (info [td info-type]
    (case info-type
      :class (class td)
      :device :cuda
      :shape (.dims td)
      :data-type (.data-type td)
      :format (.format td)
      nil))
  (info [td]
    {:class (class td)
     :device :cuda
     :shape (.dims td)
     :data-type (.data-type td)
     :format (.format td)})
  Releaseable
  (release [this]
    (locking td
      (when-let [d @td]
        (locking d
          (with-check
            (JCudnn/cudnnDestroyTensorDescriptor d)
            (vreset! td nil)))))
    (locking fd
      (when-let [d @fd]
        (locking d
          (with-check
            (JCudnn/cudnnDestroyFilterDescriptor d)
            (vreset! fd nil)))))
    true))

(defn get-filter-nd-descriptor* ^long [^cudnnFilterDescriptor fd
                                       ^ints data-type ^ints format ^ints dims]
  (let [nbdims (int-array 1)]
    (with-check
      (JCudnn/cudnnGetFilterNdDescriptor fd (alength dims) data-type format nbdims dims)
      (aget nbdims 0))))

(extend-type cudnnFilterDescriptor
  Info
  (info [fd]
    (let [data-type (int-array 1)
          format (int-array 1)
          dims (int-array JCudnn/CUDNN_DIM_MAX)
          nbdims (get-filter-nd-descriptor* fd data-type format dims)]
      {:class (class fd)
       :device :cuda
       :shape (vec (take nbdims dims))
       :data-type (dec-data-type (aget data-type 0))
       :format (dec-format (aget format 0))}))
  Wrappable
  (wrap [fd]
    (let [data-type-arr (int-array 1)
          format-arr (int-array 1)
          dims-arr (int-array JCudnn/CUDNN_DIM_MAX)
          strides-arr (int-array JCudnn/CUDNN_DIM_MAX)
          nbdims (get-filter-nd-descriptor* fd data-type-arr format-arr dims-arr)
          format (aget format-arr 0)
          data-type (aget data-type-arr 0)
          dims (vec (take nbdims dims-arr))]
      (let-release [td (if (< 4 nbdims)
                         (tensor-nd-descriptor-ex* (tensor-descriptor*) format data-type dims-arr)
                         (tensor-4d-descriptor* (tensor-descriptor*) format data-type dims))]
        (get-tensor-nd-descriptor* td data-type-arr dims-arr strides-arr)
        (->CUFilterDescriptor (volatile! fd) (volatile! td) dims
                              (dec-data-type data-type)
                              (dec-format format)
                              (vec (take nbdims strides-arr)))))))

(defn filter-descriptor* []
  (let [res (cudnnFilterDescriptor.)]
    (with-check
      (JCudnn/cudnnCreateFilterDescriptor res)
      res)))

(defn filter-4d-descriptor*
  ([^cudnnFilterDescriptor fd ^long data-type ^long format shape]
   (with-check
     (JCudnn/cudnnSetFilter4dDescriptor
      fd data-type format (get shape 0 0) (get shape 1 1) (get shape 2 1) (get shape 3 1))
     fd)))

(defn filter-nd-descriptor*
  ([^cudnnFilterDescriptor fd ^long data-type ^long format ^ints dims]
   (with-check
     (JCudnn/cudnnSetFilterNdDescriptor fd data-type format (alength dims) dims)
     fd)))

;; ======================== Convolution ==============================

(deftype-wrapper CUDnnConvolutionDescriptor
  JCudnn/cudnnDestroyConvolutionDescriptor cudnn-error)

(extend-type cudnnConvolutionDescriptor
  Wrappable
  (wrap [cd]
    (->CUDnnConvolutionDescriptor (volatile! cd))))

(defn convolution-descriptor* []
  (let [res (cudnnConvolutionDescriptor.)]
    (with-check
      (JCudnn/cudnnCreateConvolutionDescriptor res)
      res)))

(defn convolution-2d-descriptor* [^cudnnConvolutionDescriptor cd
                                  pad stride dilation mode data-type]
  (with-check
    (JCudnn/cudnnSetConvolution2dDescriptor
     cd (get pad 0 0) (get pad 1 0) (get stride 0 1) (get stride 1 1)
     (get dilation 0 1) (get dilation 0 1) mode data-type)
    cd))

(defn convolution-nd-descriptor* [^cudnnConvolutionDescriptor cd
                                  ^ints pad ^ints stride ^ints dilation mode data-type]
  (with-check
    (JCudnn/cudnnSetConvolutionNdDescriptor cd (alength pad) pad stride dilation (int mode) (int data-type))
    cd))

(defn convolution-forward* [cudnn-handle cd algo alpha desc-x buf-x
                            ^cudnnFilterDescriptor desc-w buf-w
                            beta desc-y buf-y workspace ws-size]
  (with-check
    (JCudnn/cudnnConvolutionForward cudnn-handle alpha desc-x buf-x desc-w buf-w
                                    cd algo workspace ws-size beta desc-y buf-y)
    cudnn-handle))

(defn convolution-get-fwd-algo*
  ([cudnn-handle cd desc-x ^cudnnFilterDescriptor desc-w desc-y preference limit-bytes]
   (let-release [algo (int-array 1)]
     (with-check
       (JCudnn/cudnnGetConvolutionForwardAlgorithm cudnn-handle desc-x desc-w cd desc-y
                                                   (int preference) (long limit-bytes) algo)
       (aget algo 0))))
  ([cudnn-handle cd desc-x ^cudnnFilterDescriptor desc-w desc-y limit-bytes]
   (convolution-get-fwd-algo* cudnn-handle cd desc-x desc-w desc-y
                                       cudnnConvolutionFwdPreference/CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT
                                       limit-bytes))
  ([cudnn-handle cd desc-x ^cudnnFilterDescriptor desc-w desc-y]
   (convolution-get-fwd-algo* cudnn-handle cd desc-x desc-w desc-y
                                       cudnnConvolutionFwdPreference/CUDNN_CONVOLUTION_FWD_PREFER_FASTEST
                                       0)))

(defn convolution-get-fwd-workspace-size*
  [cudnn-handle cd algo desc-x ^cudnnFilterDescriptor desc-w desc-y]
  (let-release [wsize (long-array 1)]
    (with-check
      (JCudnn/cudnnGetConvolutionForwardWorkspaceSize cudnn-handle desc-x desc-w
                                                      cd desc-y (int algo) wsize)
      (aget wsize 0))))
