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
                           Wrapper Wrappable wrap extract info Viewable view]]
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
            cudnnFilterDescriptor cudnnConvolutionFwdPreference
            cudnnConvolutionBwdDataPreference cudnnConvolutionBwdFilterPreference
            cudnnPoolingDescriptor]))

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

(deftype CUTensorDescriptor [^cudnnTensorDescriptor td dims data-type strides master]
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
  Viewable
  (view [_]
    (CUTensorDescriptor. td dims data-type strides false))
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
    (when master
      (locking td
        (when-let [d @td]
          (locking d
            (with-check
              (JCudnn/cudnnDestroyTensorDescriptor d)
              (vreset! td nil))))))
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
                            (vec (take nbdims strides)) true))))

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
  ([ad ^long mode ^long relu-nan-opt ^double coef]
   (with-check
     (JCudnn/cudnnSetActivationDescriptor ad mode relu-nan-opt coef)
     ad)))

(defn get-activation-descriptor* [ad ^ints mode ^ints relu-nan-opt ^doubles coef]
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
  ([rtd ^long op ^long comp-type ^long nan-opt]
   (with-check
     (JCudnn/cudnnSetReduceTensorDescriptor rtd op comp-type nan-opt
                                            cudnnReduceTensorIndices/CUDNN_REDUCE_TENSOR_NO_INDICES
                                            cudnnIndicesType/CUDNN_32BIT_INDICES)
     rtd))
  ([rtd op comp-type nan-opt indices]
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
                             dims data-type format master]
  Object
  (hashCode [this]
    (hash (deref fd)))
  (equals [this other]
    (= @fd (extract other)))
  (toString [this]
    (format "#CUFilterDescriptor[0x%s]" (Long/toHexString (native-pointer @fd))))
  Wrapper
  (extract [this]
    @fd)
  DescProvider
  (desc [this]
    this)
  Viewable
  (view [_]
    (CUFilterDescriptor. fd dims data-type format false))
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
    (when master
      (locking fd
        (when-let [d @fd]
          (locking d
            (with-check
              (JCudnn/cudnnDestroyFilterDescriptor d)
              (vreset! fd nil))))))
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
    (let [data-type (int-array 1)
          format (int-array 1)
          dims (int-array JCudnn/CUDNN_DIM_MAX)
          nbdims (get-filter-nd-descriptor* fd data-type format dims)]
      (->CUFilterDescriptor (volatile! fd) (vec (take nbdims dims))
                            (dec-data-type (aget data-type 0))
                            (dec-format (aget format 0))
                            true))))

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

(defn convolution-2d-descriptor* [cd pad stride dilation mode data-type]
  (with-check
    (JCudnn/cudnnSetConvolution2dDescriptor
     cd (get pad 0 0) (get pad 1 0) (get stride 0 1) (get stride 1 1)
     (get dilation 0 1) (get dilation 0 1) mode data-type)
    cd))

(defn convolution-nd-descriptor* [cd ^ints pad ^ints stride ^ints dilation mode data-type]
  (with-check
    (JCudnn/cudnnSetConvolutionNdDescriptor cd (alength pad) pad stride dilation mode data-type)
    cd))

(defn convolution-fwd-get-algo*
  ([cudnn-handle cd desc-x ^cudnnFilterDescriptor desc-w desc-y preference limit-bytes]
   (let-release [algo (int-array 1)]
     (with-check
       (JCudnn/cudnnGetConvolutionForwardAlgorithm cudnn-handle desc-x desc-w cd desc-y
                                                   preference limit-bytes algo)
       (aget algo 0))))
  ([cudnn-handle cd desc-x ^cudnnFilterDescriptor desc-w desc-y limit-bytes]
   (convolution-fwd-get-algo* cudnn-handle cd desc-x desc-w desc-y
                              cudnnConvolutionFwdPreference/CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT
                              limit-bytes))
  ([cudnn-handle cd desc-x ^cudnnFilterDescriptor desc-w desc-y]
   (convolution-fwd-get-algo* cudnn-handle cd desc-x desc-w desc-y
                              cudnnConvolutionFwdPreference/CUDNN_CONVOLUTION_FWD_PREFER_FASTEST
                              0)))

(defn convolution-fwd-get-workspace-size*
  [cudnn-handle cd algo desc-x ^cudnnFilterDescriptor desc-w desc-y]
  (let-release [wsize (long-array 1)]
    (with-check
      (JCudnn/cudnnGetConvolutionForwardWorkspaceSize cudnn-handle desc-x desc-w
                                                      cd desc-y algo wsize)
      (aget wsize 0))))

(defn convolution-fwd*
  ([cudnn-handle cd algo alpha desc-x buf-x ^cudnnFilterDescriptor desc-w buf-w
    beta desc-y buf-y workspace ws-size]
   (with-check
     (JCudnn/cudnnConvolutionForward cudnn-handle alpha desc-x buf-x desc-w buf-w
                                     cd algo workspace ws-size beta desc-y buf-y)
     cudnn-handle))
  ([cudnn-handle cd algo ad alpha1 desc-x buf-x ^cudnnFilterDescriptor desc-w buf-w
    alpha2 buf-z desc-bias buf-bias desc-y buf-y workspace ws-size]
   (with-check
     (JCudnn/cudnnConvolutionBiasActivationForward
      cudnn-handle alpha1 desc-x buf-x desc-w buf-w cd algo workspace ws-size
      alpha2 desc-y buf-z desc-bias buf-bias ad desc-y buf-y)
     cudnn-handle)))

(defn convolution-bwd-bias*
  [cudnn-handle alpha desc-dy buf-dy beta desc-db buf-db]
  (with-check
    (JCudnn/cudnnConvolutionBackwardBias cudnn-handle alpha desc-dy buf-dy
                                         beta desc-db buf-db)
    cudnn-handle))

(defn convolution-bwd-data-get-algo*
  ([cudnn-handle cd ^cudnnFilterDescriptor desc-w desc-dy desc-dx preference limit-bytes]
   (let-release [algo (int-array 1)]
     (with-check
       (JCudnn/cudnnGetConvolutionBackwardDataAlgorithm
        cudnn-handle desc-w desc-dy cd desc-dx preference limit-bytes algo)
       (aget algo 0))))
  ([cudnn-handle cd ^cudnnFilterDescriptor desc-w desc-dy desc-dx limit-bytes]
   (convolution-bwd-data-get-algo* cudnn-handle cd desc-w desc-dy desc-dx
                                   cudnnConvolutionBwdDataPreference/CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT
                                   limit-bytes))
  ([cudnn-handle cd ^cudnnFilterDescriptor desc-w desc-dy desc-dx]
   (convolution-bwd-data-get-algo* cudnn-handle cd desc-w desc-dy desc-dx
                                   cudnnConvolutionBwdDataPreference/CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST
                                   0)))

(defn convolution-bwd-data-get-workspace-size*
  [cudnn-handle cd algo ^cudnnFilterDescriptor desc-w desc-dy desc-dx]
  (let-release [wsize (long-array 1)]
    (with-check
      (JCudnn/cudnnGetConvolutionBackwardDataWorkspaceSize
       cudnn-handle desc-w desc-dy cd desc-dx algo wsize)
      (aget wsize 0))))

(defn convolution-bwd-data*
  [cudnn-handle cd algo alpha desc-w buf-w desc-dy buf-dy
   beta desc-dx buf-dx workspace ws-size]
  (with-check
    (JCudnn/cudnnConvolutionBackwardData cudnn-handle
                                         alpha desc-w buf-w desc-dy buf-dy
                                         cd algo workspace ws-size
                                         beta desc-dx buf-dx)
    cudnn-handle))

(defn convolution-bwd-filter-get-algo*
  ([cudnn-handle cd desc-x desc-dy ^cudnnFilterDescriptor desc-dw preference limit-bytes]
   (let-release [algo (int-array 1)]
     (with-check
       (JCudnn/cudnnGetConvolutionBackwardFilterAlgorithm
        cudnn-handle desc-x desc-dy cd desc-dw preference limit-bytes algo)
       (aget algo 0))))
  ([cudnn-handle cd desc-x desc-dy ^cudnnFilterDescriptor desc-dw limit-bytes]
   (convolution-bwd-filter-get-algo*
    cudnn-handle cd desc-x desc-dy desc-dw
    cudnnConvolutionBwdFilterPreference/CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT
    limit-bytes))
  ([cudnn-handle cd desc-x desc-dy ^cudnnFilterDescriptor desc-dw]
   (convolution-bwd-filter-get-algo*
    cudnn-handle cd desc-x desc-dy desc-dw
    cudnnConvolutionBwdFilterPreference/CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST
    0)))

(defn convolution-bwd-filter-get-workspace-size*
  [cudnn-handle cd algo desc-x desc-dy ^cudnnFilterDescriptor desc-dw]
  (let-release [wsize (long-array 1)]
    (with-check
      (JCudnn/cudnnGetConvolutionBackwardFilterWorkspaceSize
       cudnn-handle desc-x desc-dy cd desc-dw algo wsize)
      (aget wsize 0))))

(defn convolution-bwd-filter*
  [cudnn-handle cd algo alpha desc-x buf-x desc-dy buf-dy
   beta desc-dw buf-dw workspace ws-size]
  (with-check
    (JCudnn/cudnnConvolutionBackwardFilter cudnn-handle
                                           alpha desc-x buf-x desc-dy buf-dy
                                           cd algo workspace ws-size
                                           beta desc-dw buf-dw)
    cudnn-handle))

(defn convolution-bwd-bias*
  [cudnn-handle alpha desc-dy buf-dy beta desc-db buf-db]
  (with-check
    (JCudnn/cudnnConvolutionBackwardBias cudnn-handle alpha desc-dy buf-dy beta desc-db buf-db)
    cudnn-handle))

;; ======================== Pooling ================================================================

(deftype-wrapper CUDnnPoolingDescriptor
  JCudnn/cudnnDestroyPoolingDescriptor cudnn-error)

(extend-type cudnnPoolingDescriptor
  Wrappable
  (wrap [pd]
    (->CUDnnPoolingDescriptor (volatile! pd))))

(defn pooling-descriptor* []
  (let [res (cudnnPoolingDescriptor.)]
    (with-check
      (JCudnn/cudnnCreatePoolingDescriptor res)
      res)))

(defn pooling-2d-descriptor* [pd mode nan-opt kernel strides padding]
  (with-check
    (JCudnn/cudnnSetPooling2dDescriptor
     pd mode nan-opt (get kernel 0 0) (get kernel 1 0) (get padding 0 1) (get padding 1 1)
     (get strides 0 1) (get strides 0 1))
    pd))

(defn pooling-nd-descriptor* [pd mode nan-opt ^ints kernel ^ints stride ^ints padding]
  (with-check
    (JCudnn/cudnnSetPoolingNdDescriptor pd mode nan-opt (alength kernel) kernel padding stride)
    pd))

(defn pooling-forward* [cudnn-handle pd alpha desc-x buf-x beta desc-y buf-y]
  (with-check
    (JCudnn/cudnnPoolingForward cudnn-handle pd alpha desc-x buf-x beta desc-y buf-y)
    cudnn-handle))

(defn pooling-backward* [cudnn-handle pd
                         alpha desc-y buf-y desc-dy buf-dy desc-x buf-x
                         beta desc-dx buf-dx]
  (with-check
    (JCudnn/cudnnPoolingBackward cudnn-handle pd
                                 alpha desc-y buf-y desc-dy buf-dy desc-x buf-x
                                 beta desc-dx buf-dx)
    cudnn-handle))
