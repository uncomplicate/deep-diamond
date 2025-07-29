;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.diamond.internal.cudnn.impl
  (:require [uncomplicate.commons
             [core :refer [Releaseable release with-release let-release Info info Viewable view size
                           Bytes]]
             [utils :refer [dragan-says-ex]]]
            [uncomplicate.fluokitten.protocols :refer [Comonad extract]]
            [uncomplicate.clojure-cpp
             :refer [null? int-pointer size-t-pointer get-entry address pointer-seq ptr* Accessor]]
            [uncomplicate.diamond.internal.utils :refer [extend-pointer]]
            [uncomplicate.diamond.internal.cudnn
             [constants :refer :all]
             [protocols :refer :all]])
  (:import [org.bytedeco.javacpp Pointer IntPointer FloatPointer DoublePointer PointerPointer]
           org.bytedeco.cuda.global.cudnn
           org.bytedeco.cuda.cudart.CUstream_st
           [org.bytedeco.cuda.cudnn cudnnContext cudnnTensorStruct cudnnActivationStruct
            cudnnReduceTensorStruct cudnnFilterStruct cudnnConvolutionStruct cudnnConvolutionFwdAlgoPerf_t
            cudnnConvolutionBwdDataAlgoPerf_t cudnnConvolutionBwdFilterAlgoPerf_t cudnnPoolingStruct
            cudnnDropoutStruct cudnnRNNStruct cudnnRNNDataStruct]))

(defn cudnn-error [^long err-code details]
  (let [err (get cudnn-status-codes err-code err-code)]
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

(extend-pointer cudnnContext cudnn/cudnnDestroy cudnn-error)

(defn cudnn-context*
  "Creates a cuDNN context handle on the specific `hstream`."
  [^CUstream_st hstream]
  (let-release [handle (cudnnContext.)]
    (with-check (cudnn/cudnnCreate handle)
      (with-check (cudnn/cudnnSetStream handle hstream) handle))))

(defn get-cudnn-stream* [handle]
  (let-release [res (CUstream_st.)]
    (with-check (cudnn/cudnnGetStream handle res) res)))

;; =========================== Tensor Descriptor ============================

(defmacro extend-tensor-descriptor [t destructor]
  `(extend-type ~t
     Releaseable
     (release [this#]
       (let [td# (.-td this#)]
         (locking td#
           (when (and (.-master this#) (not (null? td#)))
             (with-check
               (. cudnn ~destructor td#)
               (do (.deallocate (ptr* td#))
                   (.setNull (ptr* td#)))))))
       true)
     Comonad
     (extract [this#]
       (if-not (null? (.-td this#)) (.-td this#) nil))
     DescProvider
     (desc [this#]
       this#)
     Viewable
     (view [this#]
       (new ~t (.-td this#) (.-dims this#) (.-data-type this#) (.-layout this#) false))
     Info
     (info
       ([this# info-type#]
        (case info-type#
          :class (class this#)
          :device :cuda
          :shape (.-dims this#)
          :data-type (.-data-type this#)
          :layout (.-layout this#)
          nil))
       ([this#]
        {:class (class this#)
         :device :cuda
         :shape (.-dims this#)
         :data-type (.-data-type this#)
         :layout (.-layout this#)}))))

(deftype CUTensorDescriptor [^cudnnTensorStruct td dims data-type layout master]
  Object
  (hashCode [this]
    (hash td))
  (equals [this other]
    (or (identical? this other)
        (and (instance? CUTensorDescriptor other)
             (let [td2 ^CUTensorDescriptor other]
               (or (= td (extract other))
                   (and (= (.dims this) (.dims td2)) (= (.data-type this) (.data-type td2))
                        (= (.layout this) (.layout td2))))))))
  (toString [this]
    (format "#CUTensorDescriptor[0x%s, master: %s]" (address td) master))
  Bytes
  (bytesize* [_]
    (if (null? td)
      nil
      (with-release [res (size-t-pointer 1)]
        (with-check
          (cudnn/cudnnGetTensorSizeInBytes td res) (get-entry res 0))))))

(extend-tensor-descriptor CUTensorDescriptor cudnnDestroyTensorDescriptor)

(defn get-tensor-nd-descriptor* ^long [^cudnnTensorStruct td
                                       ^IntPointer data-type ^IntPointer dims ^IntPointer strides]
  (with-release [nbdims (int-pointer 1)]
    (with-check
      (cudnn/cudnnGetTensorNdDescriptor td (size dims) data-type nbdims dims strides)
      (get-entry nbdims 0))))

(extend-type cudnnTensorStruct
  Info
  (info [td]
    (with-release [data-type (int-pointer 1)
                   dims (int-pointer cudnn/CUDNN_DIM_MAX)
                   strides (int-pointer cudnn/CUDNN_DIM_MAX)
                   nbdims (get-tensor-nd-descriptor* td data-type dims strides)]
      {:class (class td)
       :device :cuda
       :shape (vec (take nbdims (pointer-seq dims)))
       :data-type (dec-data-type (get-entry data-type 0))
       :layout (vec (take nbdims (pointer-seq strides)))})))

(defn wrap-tensor-struct [^cudnnTensorStruct td]
  (with-release [data-type (int-pointer 1)
                 dims (int-pointer cudnn/CUDNN_DIM_MAX)
                 strides (int-pointer cudnn/CUDNN_DIM_MAX)
                 nbdims (get-tensor-nd-descriptor* td data-type dims strides)]
    (->CUTensorDescriptor td (vec (take nbdims (pointer-seq dims)))
                          (dec-data-type (get-entry data-type 0))
                          (vec (take nbdims (pointer-seq strides))) true)))

(defn tensor-descriptor* []
  (let [res (cudnnTensorStruct.)]
    (with-check
      (cudnn/cudnnCreateTensorDescriptor res)
      res)))

(defn tensor-4d-descriptor*
  ([^cudnnTensorStruct td ^long format ^long data-type shape]
   (with-check
     (cudnn/cudnnSetTensor4dDescriptor
      td format data-type (get shape 0 0) (get shape 1 1) (get shape 2 1) (get shape 3 1))
     td)))

(defn tensor-4d-descriptor-ex* [^cudnnTensorStruct td ^long data-type shape stride]
  (with-check
    (cudnn/cudnnSetTensor4dDescriptorEx
     td data-type (get shape 0 0) (get shape 1 1) (get shape 2 1) (get shape 3 1)
     (get stride 0 0) (get stride 1 1) (get stride 2 1) (get stride 3 1))
    td))

(defn tensor-nd-descriptor*
  ([^cudnnTensorStruct td ^long data-type ^IntPointer dims ^IntPointer strides]
   (with-check
     (cudnn/cudnnSetTensorNdDescriptor td data-type (size dims) dims strides)
     td)))

(defn tensor-nd-descriptor-ex*
  ([^cudnnTensorStruct td ^long format ^long data-type ^IntPointer dims]
   (with-check
     (cudnn/cudnnSetTensorNdDescriptorEx td format data-type (size dims) dims)
     td)))

(defn set-tensor* [cudnn-context td buf value]
  (with-check
    (cudnn/cudnnSetTensor cudnn-context td buf value)
    cudnn-context))

(defn add-tensor* [cudnn-context alpha desc-x buf-x beta desc-y buf-y]
  (with-check
    (cudnn/cudnnAddTensor cudnn-context alpha desc-x buf-x beta desc-y buf-y)
    cudnn-context))

(defn transform-tensor* [cudnn-context alpha desc-x buf-x beta desc-y buf-y]
  (with-check
    (cudnn/cudnnTransformTensor cudnn-context alpha desc-x buf-x beta desc-y buf-y)
    cudnn-context))

(defn scale-tensor* [cudnn-context td buf alpha]
  (with-check
    (cudnn/cudnnScaleTensor cudnn-context td buf alpha)
    cudnn-context))

;; ======================= Activation ===================================

(extend-pointer cudnnActivationStruct cudnn/cudnnDestroyActivationDescriptor cudnn-error)

(defn activation-descriptor*
  ([]
   (let [res (cudnnActivationStruct.)]
     (with-check
       (cudnn/cudnnCreateActivationDescriptor res)
       res)))
  ([ad ^long mode ^long relu-nan-opt ^double coef]
   (with-check
     (cudnn/cudnnSetActivationDescriptor ad mode relu-nan-opt coef)
     ad)))

(defn get-activation-descriptor* [^cudnnActivationStruct ad
                                  ^IntPointer mode ^IntPointer relu-nan-opt ^DoublePointer coef]
  (with-check
    (cudnn/cudnnGetActivationDescriptor ad mode relu-nan-opt coef)
    ad))

(defn activation-forward* [cudnn-context ad alpha desc-x buf-x beta desc-y buf-y]
  (with-check
    (cudnn/cudnnActivationForward cudnn-context ad alpha desc-x buf-x beta desc-y buf-y)
    cudnn-context))

(defn activation-backward* [cudnn-context ad
                            alpha desc-y buf-y desc-dy buf-dy
                            desc-x buf-x beta desc-dx buf-dx]
  (with-check
    (cudnn/cudnnActivationBackward cudnn-context ad
                                   alpha desc-y buf-y desc-dy buf-dy
                                   desc-x buf-x beta desc-dx buf-dx)
    cudnn-context))

;; ========================== Reduce ===================================

(extend-pointer cudnnReduceTensorStruct cudnn/cudnnDestroyReduceTensorDescriptor cudnn-error)

(defn reduce-tensor-descriptor*
  ([]
   (let-release [res (cudnnReduceTensorStruct.)]
     (with-check
       (cudnn/cudnnCreateReduceTensorDescriptor res)
       res)))
  ([rtd ^long op ^long comp-type ^long nan-opt]
   (with-check
     (cudnn/cudnnSetReduceTensorDescriptor rtd op comp-type nan-opt
                                           cudnn/CUDNN_REDUCE_TENSOR_NO_INDICES
                                           cudnn/CUDNN_32BIT_INDICES)
     rtd))
  ([rtd op comp-type nan-opt indices]
   (let [comp-type (int comp-type)]
     (with-check
       (cudnn/cudnnSetReduceTensorDescriptor
        rtd (int op) (int comp-type) (int nan-opt)
        (if (or (= cudnn/CUDNN_REDUCE_TENSOR_AMAX comp-type)
                (= cudnn/CUDNN_REDUCE_TENSOR_MAX comp-type)
                (= cudnn/CUDNN_REDUCE_TENSOR_MIN comp-type))
          (int indices)
          cudnn/CUDNN_REDUCE_TENSOR_NO_INDICES)
        cudnn/CUDNN_32BIT_INDICES)
       rtd))))

(defn reduce-tensor* [cudnn-context rtd
                      indices indices-size workspace workspace-size
                      alpha desc-x buf-x beta desc-y buf-y]
  (with-check
    (cudnn/cudnnReduceTensor cudnn-context rtd
                             indices indices-size workspace workspace-size
                             alpha desc-x buf-x beta desc-y buf-y)
    cudnn-context))

(defn reduction-indices-size* ^long [cudnn-context rtd desc-x desc-y]
  (with-release [size (size-t-pointer 1)]
    (with-check
      (cudnn/cudnnGetReductionIndicesSize cudnn-context rtd desc-x desc-y size)
      (get-entry size 0))))

(defn reduction-workspace-size* ^long [cudnn-context rtd desc-x desc-y]
  (with-release [size (size-t-pointer 1)]
    (with-check
      (cudnn/cudnnGetReductionWorkspaceSize cudnn-context rtd desc-x desc-y size)
      (get-entry size 0))))

;; ======================= Softmax ===================================

(defn softmax-forward* [cudnn-context algo mode alpha desc-x buf-x beta desc-y buf-y]
  (with-check
    (cudnn/cudnnSoftmaxForward cudnn-context algo mode alpha desc-x buf-x beta desc-y buf-y)
    cudnn-context))

(defn softmax-backward* [cudnn-context algo mode
                         alpha desc-y buf-y desc-dy buf-dy
                         beta desc-dx buf-dx]
  (with-check
    (cudnn/cudnnSoftmaxBackward cudnn-context algo mode
                                alpha desc-y buf-y desc-dy buf-dy
                                beta desc-dx buf-dx)
    cudnn-context))

;; ======================== Filter ==============================

(deftype CUFilterDescriptor [^cudnnFilterStruct td
                             dims data-type layout master]
  Object
  (hashCode [this]
    (hash td))
  (equals [this other]
    (or (identical? this other)
        (and (instance? CUFilterDescriptor other)
             (let [td2 ^CUFilterDescriptor other]
               (or (= td (extract other))
                   (and (= (.dims this) (.dims td2)) (= (.data-type this) (.data-type td2))
                        (= (.layout this) (.layout td2))))))))
  (toString [this]
    (format "#CUFilterDescriptor[0x%s, master: %s]" (address td) master))
  Bytes
  (bytesize* [_]
    (if (null? td)
      nil
      (with-release [res (size-t-pointer 1)]
        (with-check
          (cudnn/cudnnGetFilterSizeInBytes td res) (get-entry res 0))))))

(extend-tensor-descriptor CUFilterDescriptor cudnnDestroyFilterDescriptor)

(defn get-filter-nd-descriptor* ^long [^cudnnFilterStruct fd
                                       ^IntPointer data-type ^IntPointer format ^IntPointer dims]
  (with-release [nbdims (int-pointer 1)]
    (with-check
      (cudnn/cudnnGetFilterNdDescriptor fd (size dims) data-type format nbdims dims)
      (get-entry nbdims 0))))

(extend-type cudnnFilterStruct
  Info
  (info [fd]
    (with-release [data-type (int-pointer 1)
                   format (int-pointer 1)
                   dims (int-pointer cudnn/CUDNN_DIM_MAX)
                   nbdims (get-filter-nd-descriptor* fd data-type format dims)]
      {:class (class fd)
       :device :cuda
       :shape (vec (take nbdims (pointer-seq dims)))
       :data-type (dec-data-type (get-entry data-type 0))
       :format (dec-format (get-entry format 0))})))

(defn wrap-filter-struct [^cudnnFilterStruct fd]
      (with-release [data-type (int-pointer 1)
                     format (int-pointer 1)
                     dims (int-pointer cudnn/CUDNN_DIM_MAX)
                     nbdims (get-filter-nd-descriptor* fd data-type format dims)]
        (->CUFilterDescriptor fd (vec (take nbdims (pointer-seq dims)))
                              (dec-data-type (get-entry data-type 0))
                              (dec-format (get-entry format 0))
                              true)))

(defn filter-descriptor* []
  (let-release [res (cudnnFilterStruct.)]
    (with-check
      (cudnn/cudnnCreateFilterDescriptor res)
      res)))

(defn filter-4d-descriptor*
  ([^cudnnFilterStruct fd ^long data-type ^long format shape]
   (with-check
     (cudnn/cudnnSetFilter4dDescriptor
      fd data-type format (get shape 0 0) (get shape 1 1) (get shape 2 1) (get shape 3 1))
     fd)))

(defn filter-nd-descriptor*
  ([^cudnnFilterStruct fd ^long data-type ^long format ^IntPointer dims]
   (with-check
     (cudnn/cudnnSetFilterNdDescriptor fd data-type format (size dims) dims)
     fd)))

;; ======================== Convolution ==============================

(extend-type cudnnConvolutionFwdAlgoPerf_t
  Accessor
  (get-entry
    ([this#]
     (.getPointer this# (.position this#)))
    ([this# i#]
     (.getPointer this# (long i#)))))

(extend-type cudnnConvolutionBwdDataAlgoPerf_t
  Accessor
  (get-entry
    ([this#]
     (.getPointer this# (.position this#)))
    ([this# i#]
     (.getPointer this# (long i#)))))

(extend-type cudnnConvolutionBwdFilterAlgoPerf_t
  Accessor
  (get-entry
    ([this#]
     (.getPointer this# (.position this#)))
    ([this# i#]
     (.getPointer this# (long i#)))))

(extend-pointer cudnnConvolutionStruct cudnn/cudnnDestroyConvolutionDescriptor cudnn-error)

(defn convolution-descriptor* []
  (let-release [res (cudnnConvolutionStruct.)]
    (with-check
      (cudnn/cudnnCreateConvolutionDescriptor res)
      res)))

(defn convolution-2d-descriptor* [cd pad stride dilation mode data-type]
  (with-check
    (cudnn/cudnnSetConvolution2dDescriptor
     cd (get pad 0 0) (get pad 1 0) (get stride 0 1) (get stride 1 1)
     (get dilation 0 1) (get dilation 1 1) mode data-type)
    cd))

(defn convolution-nd-descriptor* [^cudnnConvolutionStruct cd
                                  ^IntPointer pad ^IntPointer stride ^IntPointer dilation mode data-type]
  (with-check
    (cudnn/cudnnSetConvolutionNdDescriptor cd (size pad) pad stride dilation (int mode) (int data-type))
    cd))

(defn convolution-fwd-find-algo*
  ([^cudnnContext cudnn-context ^cudnnConvolutionStruct cd
    ^cudnnTensorStruct desc-x ^cudnnFilterStruct desc-w ^cudnnTensorStruct desc-y algo-count]
   (with-release [algo-perf-count (int-pointer 1)]
     (let-release [algo-perf (cudnnConvolutionFwdAlgoPerf_t. (long algo-count))]
       (with-check
         (cudnn/cudnnFindConvolutionForwardAlgorithm cudnn-context desc-x desc-w cd desc-y
                                                     (int algo-count) algo-perf-count algo-perf)
         (take (get-entry algo-perf-count 0) (pointer-seq algo-perf))))))
  ([cudnn-context cd desc-x ^cudnnFilterStruct desc-w desc-y]
   (convolution-fwd-find-algo* cudnn-context cd desc-x desc-w desc-y 1)))

(defn convolution-fwd-algo* ^long [^cudnnConvolutionFwdAlgoPerf_t algo-perf]
  (.algo algo-perf))

(defn convolution-fwd-status* ^long [^cudnnConvolutionFwdAlgoPerf_t algo-perf]
  (.status algo-perf))

(defn convolution-fwd-memory* ^long [^cudnnConvolutionFwdAlgoPerf_t algo-perf]
  (.memory algo-perf))

(defn convolution-fwd-time* ^double [^cudnnConvolutionFwdAlgoPerf_t algo-perf]
  (.time algo-perf))

(defn convolution-fwd-determinism* ^long [^cudnnConvolutionFwdAlgoPerf_t algo-perf]
  (.determinism algo-perf))

(defn convolution-fwd-math-type* ^long [^cudnnConvolutionFwdAlgoPerf_t algo-perf]
  (.mathType algo-perf))

(defn convolution-fwd*
  ([cudnn-context cd algo alpha desc-x buf-x ^cudnnFilterStruct desc-w buf-w
    beta desc-y buf-y workspace ws-size]
   (with-check
     (cudnn/cudnnConvolutionForward cudnn-context alpha desc-x buf-x desc-w buf-w
                                    cd algo workspace ws-size beta desc-y buf-y)
     cudnn-context))
  ([cudnn-context cd algo ad alpha1 desc-x buf-x ^cudnnFilterStruct desc-w buf-w
    alpha2 buf-z desc-bias buf-bias desc-y buf-y workspace ws-size]
   (with-check
     (cudnn/cudnnConvolutionBiasActivationForward
      cudnn-context alpha1 desc-x buf-x desc-w buf-w cd algo workspace ws-size
      alpha2 desc-y buf-z desc-bias buf-bias ad desc-y buf-y)
     cudnn-context)))

(defn convolution-bwd-bias*
  [cudnn-context alpha desc-dy buf-dy beta desc-db buf-db]
  (with-check
    (cudnn/cudnnConvolutionBackwardBias cudnn-context alpha desc-dy buf-dy
                                        beta desc-db buf-db)
    cudnn-context))

(defn convolution-bwd-data-find-algo*
  ([^cudnnContext cudnn-context ^cudnnConvolutionStruct cd
    ^cudnnFilterStruct desc-w ^cudnnTensorStruct desc-dy ^cudnnTensorStruct desc-dx algo-count]
   (with-release [algo-perf-count (int-pointer 1)]
     (let-release [algo-perf (cudnnConvolutionBwdDataAlgoPerf_t. (long algo-count))]
       (with-check
         (cudnn/cudnnFindConvolutionBackwardDataAlgorithm cudnn-context desc-w desc-dy cd desc-dx
                                                          (int algo-count) algo-perf-count algo-perf)
         (take (get-entry algo-perf-count 0) (pointer-seq algo-perf))))))
  ([cudnn-context cd ^cudnnFilterStruct desc-w desc-dy desc-dx]
   (convolution-bwd-data-find-algo* cudnn-context cd desc-w desc-dy desc-dx 1)))

(defn convolution-bwd-data*
  [cudnn-context cd algo alpha desc-w buf-w desc-dy buf-dy
   beta desc-dx buf-dx workspace ws-size]
  (with-check
    (cudnn/cudnnConvolutionBackwardData cudnn-context
                                        alpha desc-w buf-w desc-dy buf-dy
                                        cd algo workspace ws-size
                                        beta desc-dx buf-dx)
    cudnn-context))

(defn convolution-bwd-filter-find-algo*
  ([^cudnnContext cudnn-context ^cudnnConvolutionStruct cd
    ^cudnnTensorStruct desc-x ^cudnnTensorStruct desc-dy ^cudnnFilterStruct desc-dw preference algo-count]
   (with-release [algo-perf-count (int-pointer 1)]
     (let-release [algo-perf (cudnnConvolutionBwdFilterAlgoPerf_t. (long algo-count))]
       (with-check
         (cudnn/cudnnFindConvolutionBackwardFilterAlgorithm cudnn-context desc-x desc-dy cd desc-dw
                                                            (int algo-count) algo-perf-count algo-perf)
         (take (get-entry algo-perf-count 0) (pointer-seq algo-perf))))))
  ([cudnn-context cd desc-x desc-dy ^cudnnFilterStruct desc-dw preference]
   (convolution-bwd-filter-find-algo* cudnn-context cd desc-x desc-dy desc-dw preference 1)))

(defn convolution-bwd-filter*
  [cudnn-context cd algo alpha desc-x buf-x desc-dy buf-dy
   beta desc-dw buf-dw workspace ws-size]
  (with-check
    (cudnn/cudnnConvolutionBackwardFilter cudnn-context
                                          alpha desc-x buf-x desc-dy buf-dy
                                          cd algo workspace ws-size
                                          beta desc-dw buf-dw)
    cudnn-context))

(defn convolution-bwd-bias*
  [cudnn-context alpha desc-dy buf-dy beta desc-db buf-db]
  (with-check
    (cudnn/cudnnConvolutionBackwardBias cudnn-context alpha desc-dy buf-dy beta desc-db buf-db)
    cudnn-context*))

;; ======================== Pooling ================================================================

(extend-pointer cudnnPoolingStruct cudnn/cudnnDestroyPoolingDescriptor cudnn-error)

(defn pooling-descriptor* []
  (let-release [res (cudnnPoolingStruct.)]
    (with-check
      (cudnn/cudnnCreatePoolingDescriptor res)
      res)))

(defn pooling-2d-descriptor* [pd mode nan-opt kernel strides padding]
  (with-check
    (cudnn/cudnnSetPooling2dDescriptor
     pd mode nan-opt (get kernel 0 0) (get kernel 1 0) (get padding 0 1) (get padding 1 1)
     (get strides 0 1) (get strides 1 1))
    pd))

(defn pooling-nd-descriptor* [^cudnnPoolingStruct pd mode nan-opt
                              ^IntPointer kernel ^IntPointer stride ^IntPointer padding]
  (with-check
    (cudnn/cudnnSetPoolingNdDescriptor pd (int mode) (int nan-opt) (size kernel) kernel padding stride)
    pd))

(defn pooling-forward* [cudnn-context pd alpha desc-x buf-x beta desc-y buf-y]
  (with-check
    (cudnn/cudnnPoolingForward cudnn-context pd alpha desc-x buf-x beta desc-y buf-y)
    cudnn-context))

(defn pooling-backward* [cudnn-context pd
                         alpha desc-y buf-y desc-dy buf-dy desc-x buf-x
                         beta desc-dx buf-dx]
  (with-check
    (cudnn/cudnnPoolingBackward cudnn-context pd
                                alpha desc-y buf-y desc-dy buf-dy desc-x buf-x
                                beta desc-dx buf-dx)
    cudnn-context))

;; ====================== Batch Normalization ===========================================

(defn batch-norm-param-descriptor* [desc-x mode]
  (let-release [res (tensor-descriptor*)]
    (with-check
      (cudnn/cudnnDeriveBNTensorDescriptor res desc-x mode)
      res)))

(defn batch-norm-fwd-inference* [cudnn-context mode alpha beta desc-x buf-x desc-y buf-y
                                 desc-param buf-scale buf-shift buf-mean buf-var epsilon]
  (with-check
    (cudnn/cudnnBatchNormalizationForwardInference
     cudnn-context mode alpha beta desc-x buf-x desc-y buf-y
     desc-param buf-scale buf-shift buf-mean buf-var epsilon)
    cudnn-context))

(defn batch-norm-fwd-training* [cudnn-context mode alpha beta desc-x buf-x desc-y buf-y
                                desc-param buf-scale buf-shift exp-avg
                                buf-running-mean buf-running-var epsilon
                                buf-save-mean buf-save-inv-var]
  (with-check
    (cudnn/cudnnBatchNormalizationForwardTraining
     cudnn-context mode alpha beta desc-x buf-x desc-y buf-y
     desc-param buf-scale buf-shift exp-avg
     buf-running-mean buf-running-var epsilon buf-save-mean buf-save-inv-var)
    cudnn-context))

(defn batch-norm-backward* [cudnn-context mode alpha-data beta-data alpha-param beta-param
                            desc-x buf-x desc-dy buf-dy desc-dx buf-dx desc-param
                            buf-scale buf-scale-diff buf-shift-diff epsilon buf-mean buf-inv-var]
  (with-check
    (cudnn/cudnnBatchNormalizationBackward
     cudnn-context mode alpha-data beta-data alpha-param beta-param
     desc-x buf-x desc-dy buf-dy desc-dx buf-dx
     desc-param buf-scale buf-scale-diff buf-shift-diff epsilon buf-mean buf-inv-var)
    cudnn-context))

;; ====================== Dropout ======================================================

(extend-pointer cudnnDropoutStruct cudnn/cudnnDestroyDropoutDescriptor cudnn-error)

(defn dropout-descriptor*
  ([]
   (let-release [res (cudnnDropoutStruct.)]
     (with-check
       (cudnn/cudnnCreateDropoutDescriptor res)
       res)))
  ([cudnn-context dd dropout states state-size seed]
   (with-check
     (cudnn/cudnnSetDropoutDescriptor dd cudnn-context dropout states state-size seed)
     dd)))

(defn dropout-states-size* [cudnn-context]
  (with-release [size (size-t-pointer 1)]
    (with-check
      (cudnn/cudnnDropoutGetStatesSize cudnn-context size)
      (get-entry size 0))))

(defn dropout-reserve-space-size* [cudnn-context]
  (with-release [size (size-t-pointer 1)]
    (with-check
      (cudnn/cudnnDropoutGetReserveSpaceSize cudnn-context size)
      (get-entry size 0))))

;; ====================== RNN ===========================================================

(extend-pointer cudnnRNNStruct cudnn/cudnnDestroyRNNDescriptor cudnn-error)

(defn rnn-descriptor*
  ([]
   (let-release [res (cudnnRNNStruct.)]
     (with-check
       (cudnn/cudnnCreateRNNDescriptor res)
       res)))
  ([rd algo cell-mode bias-mode dir-mode input-mode data-type math-prec
    math-type input-size hidden-size proj-size num-nayers dropout-desc aux-flags]
   (with-check
     (cudnn/cudnnSetRNNDescriptor_v8
      rd
      algo cell-mode bias-mode dir-mode input-mode data-type math-prec math-type
      input-size hidden-size proj-size num-nayers dropout-desc aux-flags)
     rd)))

(defn get-rnn-descriptor* [^cudnnRNNStruct rd
                           ^IntPointer algo ^IntPointer cell-mode ^IntPointer bias-mode
                           ^IntPointer dir-mode ^IntPointer  input-mode ^IntPointer  data-type
                           ^IntPointer math-prec ^IntPointer math-type ^IntPointer input-size
                           ^IntPointer hidden-size ^IntPointer proj-size ^IntPointer num-nayers
                           ^cudnnDropoutStruct dropout-desc ^IntPointer aux-flags]
  (with-check
    (cudnn/cudnnGetRNNDescriptor_v8 rd algo cell-mode bias-mode dir-mode input-mode
                                    data-type math-prec math-type input-size hidden-size proj-size
                                    num-nayers dropout-desc aux-flags)
    rd))

(defn rnn-weight-params* [^cudnnContext cudnn-context ^cudnnRNNStruct rd pseudo-layer
                          weight-space-size ^Pointer weight-space lin-layer-id
                          ^cudnnTensorStruct w-desc ^PointerPointer w-addr
                          ^cudnnTensorStruct b-desc ^PointerPointer b-addr]
  (with-check
    (cudnn/cudnnGetRNNWeightParams cudnn-context rd (int pseudo-layer) (long weight-space-size)
                                   weight-space (int lin-layer-id) w-desc w-addr b-desc b-addr)
    [w-desc w-addr b-desc b-addr]))

(defn rnn-weight-space-size* [cudnn-context rd]
  (with-release [size (size-t-pointer 1)]
    (with-check
      (cudnn/cudnnGetRNNWeightSpaceSize cudnn-context rd size)
      (get-entry size 0))))

(extend-pointer cudnnRNNDataStruct cudnn/cudnnDestroyRNNDataDescriptor cudnn-error)

(defn rnn-data-descriptor*
  ([]
   (let-release [res (cudnnRNNDataStruct.)]
     (with-check
       (cudnn/cudnnCreateRNNDataDescriptor res)
       res)))
  ([^cudnnRNNDataStruct rd data-type layout vector-size ^IntPointer seq-lengths ^Pointer padding-fill]
   (with-check
     (cudnn/cudnnSetRNNDataDescriptor rd (int data-type) (int layout)
                                      (int (apply max 0 (pointer-seq seq-lengths)))
                                      (size seq-lengths) (int vector-size) seq-lengths padding-fill)
     rd)))

(defn rnn-temp-space-size* [cudnn-context rd ^long forward-mode x-desc]
  (with-release [workspace-size (size-t-pointer 1)
                 reserve-size (size-t-pointer 1)]
    (with-check
      (cudnn/cudnnGetRNNTempSpaceSizes cudnn-context rd forward-mode x-desc
                                       workspace-size reserve-size)
      [(get-entry workspace-size 0) (get-entry reserve-size 0)])))

(defn rnn-fwd* [^cudnnContext cudnn-context ^cudnnRNNStruct rd forward-mode ^IntPointer dev-seq-lengths
                ^cudnnRNNDataStruct desc-x ^Pointer buf-x ^cudnnRNNDataStruct desc-y ^Pointer buf-y
                ^cudnnTensorStruct desc-h ^Pointer buf-hx ^Pointer buf-hy
                ^cudnnTensorStruct desc-c ^Pointer buf-cx ^Pointer buf-cy
                weight-space-size ^Pointer weight-space work-space-size ^Pointer work-space
                reserve-size ^Pointer reserve-space]
  (with-check
    (cudnn/cudnnRNNForward
     cudnn-context rd (int forward-mode) dev-seq-lengths
     desc-x buf-x desc-y buf-y desc-h buf-hx buf-hy desc-c buf-cx buf-cy
     (long weight-space-size) weight-space (long work-space-size) work-space
     (long reserve-size) reserve-space)
    cudnn-context))

(defn rnn-bwd-data* [^cudnnContext cudnn-context ^cudnnRNNStruct rd ^IntPointer dev-seq-lengths
                     ^cudnnRNNDataStruct desc-y ^Pointer buf-y ^Pointer buf-dy
                     ^cudnnRNNDataStruct desc-x ^Pointer buf-dx
                     ^cudnnTensorStruct desc-h ^Pointer buf-hx ^Pointer buf-dhy ^Pointer buf-dhx
                     ^cudnnTensorStruct desc-c ^Pointer buf-cx ^Pointer buf-dcy ^Pointer buf-dcx
                     [weight-space-size ^Pointer weight-space work-space-size ^Pointer work-space
                      reserve-size ^Pointer reserve-space]]
  (with-check
    (cudnn/cudnnRNNBackwardData_v8 cudnn-context rd ^IntPointer dev-seq-lengths
                                   desc-y buf-y buf-dy desc-x buf-dx desc-h buf-hx buf-dhy buf-dhx
                                   desc-c buf-cx buf-dcy buf-dcx
                                   (long weight-space-size) weight-space (long work-space-size) work-space
                                   (long reserve-size) reserve-space)
    cudnn-context))

(defn rnn-bwd-weights* [^cudnnContext cudnn-context ^cudnnRNNStruct rd add-grad
                        ^IntPointer dev-seq-lengths
                        ^cudnnRNNDataStruct desc-x ^Pointer buf-dx
                        ^cudnnTensorStruct desc-h ^Pointer buf-hx
                        ^cudnnRNNDataStruct desc-y ^Pointer buf-y
                        weight-space-size ^Pointer weight-space work-space-size ^Pointer work-space
                        reserve-size ^Pointer reserve-space]
  (with-check
    (cudnn/cudnnRNNBackwardWeights_v8 cudnn-context rd (int add-grad) dev-seq-lengths
                                      desc-x buf-dx desc-h buf-hx desc-y buf-y
                                      (long weight-space-size) weight-space
                                      (long work-space-size) work-space
                                      (long reserve-size) reserve-space)
    cudnn-context))
