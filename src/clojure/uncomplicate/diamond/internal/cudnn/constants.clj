;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.diamond.internal.cudnn.constants
  (:require [uncomplicate.commons.utils :refer [dragan-says-ex]])
  (:import [jcuda.jcudnn cudnnTensorFormat cudnnDataType cudnnActivationMode
            cudnnReduceTensorOp cudnnReduceTensorIndices cudnnNanPropagation
            cudnnIndicesType cudnnSoftmaxAlgorithm cudnnSoftmaxMode
            cudnnConvolutionMode cudnnConvolutionFwdAlgo cudnnConvolutionFwdPreference
            cudnnConvolutionFwdAlgoPerf]))

(defn enc-nan-propagation ^long [nan]
  (if nan
    cudnnNanPropagation/CUDNN_PROPAGATE_NAN
    cudnnNanPropagation/CUDNN_NOT_PROPAGATE_NAN))

(defn dec-nan-propagation [^long nan]
  (if (= cudnnNanPropagation/CUDNN_PROPAGATE_NAN nan) true false))

(def ^:const cudnn-format
  {:nchw cudnnTensorFormat/CUDNN_TENSOR_NCHW
   :nhwc cudnnTensorFormat/CUDNN_TENSOR_NHWC
   :nchw-vect-c cudnnTensorFormat/CUDNN_TENSOR_NCHW_VECT_C})

(defn dec-format [^long format]
  (case format
    0 :nchw
    1 :nhwc
    2 :nchw-vect-c
    (dragan-says-ex "This format is not supported by cuDNN."
                    {:format format :available (keys cudnn-format)})))

(def ^:const cudnn-data-type
  {:float cudnnDataType/CUDNN_DATA_FLOAT
   Float/TYPE cudnnDataType/CUDNN_DATA_FLOAT
   Float cudnnDataType/CUDNN_DATA_FLOAT
   :double cudnnDataType/CUDNN_DATA_DOUBLE
   Double/TYPE cudnnDataType/CUDNN_DATA_DOUBLE
   Double cudnnDataType/CUDNN_DATA_DOUBLE
   :half cudnnDataType/CUDNN_DATA_HALF
   :int8 cudnnDataType/CUDNN_DATA_INT8
   :byte cudnnDataType/CUDNN_DATA_INT8
   Byte/TYPE cudnnDataType/CUDNN_DATA_INT8
   Byte cudnnDataType/CUDNN_DATA_INT8
   :int cudnnDataType/CUDNN_DATA_INT32
   Integer/TYPE cudnnDataType/CUDNN_DATA_INT32
   Integer cudnnDataType/CUDNN_DATA_INT32
   :int8x4 cudnnDataType/CUDNN_DATA_INT8x4
   :uint8 cudnnDataType/CUDNN_DATA_UINT8
   :u8 cudnnDataType/CUDNN_DATA_UINT8
   :uit8x4 cudnnDataType/CUDNN_DATA_UINT8x4
   :int8x32 cudnnDataType/CUDNN_DATA_INT8x32})

(defn dec-data-type [^long data-type]
  (case data-type
    0 :float
    1 :double
    2 :half
    3 :byte
    4 :int
    5 :int8x4
    6 :uint8
    7 :uint8x4
    8 :int8x32
    (dragan-says-ex "This data type is not supported by cuDNN."
                    {:data-type data-type :available (keys cudnn-data-type)})))

(defn data-type-width ^long [data-type]
  (case data-type
    :float 4
    Float/TYPE 4
    Float 4
    :double 8
    Double/TYPE 8
    Double 8
    :half 2
    :byte 1
    Byte/TYPE 1
    Byte 1
    :int8 1
    :int 4
    Integer/TYPE 4
    Integer 4
    :int8x4 4
    :uint8 1
    :u8 1
    :uit8x4 4
    :int8x32 32
    (dragan-says-ex "This data type is not supported by cuDNN."
                    {:data-type data-type :available (keys cudnn-data-type)})))

(def ^:const cudnn-activation-mode
  {:logistic cudnnActivationMode/CUDNN_ACTIVATION_SIGMOID
   :sigmoid cudnnActivationMode/CUDNN_ACTIVATION_SIGMOID
   :relu cudnnActivationMode/CUDNN_ACTIVATION_RELU
   :tanh cudnnActivationMode/CUDNN_ACTIVATION_TANH
   :clipped-relu cudnnActivationMode/CUDNN_ACTIVATION_CLIPPED_RELU
   :elu cudnnActivationMode/CUDNN_ACTIVATION_ELU
   :identity cudnnActivationMode/CUDNN_ACTIVATION_IDENTITY
   :linear cudnnActivationMode/CUDNN_ACTIVATION_IDENTITY})

(defn dec-activation-mode [^long mode]
  (case mode
    0 :logistic
    1 :relu
    2 :tanh
    3 :clipped-relu
    4 :elu
    5 :identity
    (dragan-says-ex "This mode is not supported by cuDNN."
                    {:mode mode :available (keys cudnn-activation-mode)})))

(def ^:const cudnn-reduce-tensor-op
  {:add cudnnReduceTensorOp/CUDNN_REDUCE_TENSOR_ADD
   :amax cudnnReduceTensorOp/CUDNN_REDUCE_TENSOR_AMAX
   :avg cudnnReduceTensorOp/CUDNN_REDUCE_TENSOR_AVG
   :max cudnnReduceTensorOp/CUDNN_REDUCE_TENSOR_MAX
   :min cudnnReduceTensorOp/CUDNN_REDUCE_TENSOR_MIN
   :mul cudnnReduceTensorOp/CUDNN_REDUCE_TENSOR_MUL
   :mul-no-zeros cudnnReduceTensorOp/CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS
   :norm1 cudnnReduceTensorOp/CUDNN_REDUCE_TENSOR_NORM1
   :norm2 cudnnReduceTensorOp/CUDNN_REDUCE_TENSOR_NORM2})

(def ^:const cudnn-reduce-tensor-indices
  {:flattened cudnnReduceTensorIndices/CUDNN_REDUCE_TENSOR_FLATTENED_INDICES
   :no-indices cudnnReduceTensorIndices/CUDNN_REDUCE_TENSOR_NO_INDICES})

(def ^:const cudnn-softmax-algorithm
  {:fast cudnnSoftmaxAlgorithm/CUDNN_SOFTMAX_FAST
   :accurate cudnnSoftmaxAlgorithm/CUDNN_SOFTMAX_ACCURATE
   :log cudnnSoftmaxAlgorithm/CUDNN_SOFTMAX_LOG})

(def ^:const cudnn-softmax-mode
  {:instance cudnnSoftmaxMode/CUDNN_SOFTMAX_MODE_INSTANCE
   :channel cudnnSoftmaxMode/CUDNN_SOFTMAX_MODE_CHANNEL})

(def ^:const cudnn-convolution-mode
  {:convolution cudnnConvolutionMode/CUDNN_CONVOLUTION
   :cross-correleation cudnnConvolutionMode/CUDNN_CROSS_CORRELATION})

(def ^:const cudnn-convolution-fwd-algo
  {:count cudnnConvolutionFwdAlgo/CUDNN_CONVOLUTION_FWD_ALGO_COUNT
   :direct cudnnConvolutionFwdAlgo/CUDNN_CONVOLUTION_FWD_ALGO_DIRECT
   :fft cudnnConvolutionFwdAlgo/CUDNN_CONVOLUTION_FWD_ALGO_FFT
   :fft-tiling cudnnConvolutionFwdAlgo/CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING
   :gemm cudnnConvolutionFwdAlgo/CUDNN_CONVOLUTION_FWD_ALGO_GEMM
   :implicit-gemm cudnnConvolutionFwdAlgo/CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM
   :implicit-precomp-gemm cudnnConvolutionFwdAlgo/CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
   :winograd cudnnConvolutionFwdAlgo/CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD
   :winograd-nonfused cudnnConvolutionFwdAlgo/CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED})

(defn dec-convolution-fwd-algo [^long algo]
  (case algo
    0 :implicit-gemm
    1 :implicit-precomp-gemm
    2 :gemm
    3 :direct
    4 :fft
    5 :fft-tiling
    6 :winograd
    7 :winograd-nonfused
    8 :count
    (dragan-says-ex "This algorithm is not supported by cuDNN."
                    {:algo algo :available (keys cudnn-convolution-fwd-algo)})))

(def ^:const cudnn-convolution-fwd-preference
  {:no-workspace cudnnConvolutionFwdPreference/CUDNN_CONVOLUTION_FWD_NO_WORKSPACE
   :fastest cudnnConvolutionFwdPreference/CUDNN_CONVOLUTION_FWD_PREFER_FASTEST
   :workspace-limit cudnnConvolutionFwdPreference/CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT})
