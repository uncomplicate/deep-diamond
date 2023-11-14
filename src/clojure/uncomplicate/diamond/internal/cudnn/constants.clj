;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.diamond.internal.cudnn.constants
  (:require [uncomplicate.commons.utils :refer [dragan-says-ex]])
  (:import org.bytedeco.cuda.global.cudnn))

(def ^{:const true
       :doc "CUDA Error messages as defined in cuDNN."}
  cudnn-status-codes
  {cudnn/CUDNN_STATUS_SUCCESS :success
   cudnn/CUDNN_STATUS_NOT_INITIALIZED :not_initialized
   cudnn/CUDNN_STATUS_ALLOC_FAILED :alloc-failed
   cudnn/CUDNN_STATUS_BAD_PARAM :bad-param
   cudnn/CUDNN_STATUS_INTERNAL_ERROR :internal-error
   cudnn/CUDNN_STATUS_INVALID_VALUE :invalid-value
   cudnn/CUDNN_STATUS_ARCH_MISMATCH :arch-mismatch
   cudnn/CUDNN_STATUS_MAPPING_ERROR :mapping-error
   cudnn/CUDNN_STATUS_EXECUTION_FAILED :execution-failed
   cudnn/CUDNN_STATUS_NOT_SUPPORTED :not-supported
   cudnn/CUDNN_STATUS_LICENSE_ERROR :license-error
   cudnn/CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING :runtime-prerequisites-missing
   cudnn/CUDNN_STATUS_RUNTIME_IN_PROGRESS :runtime-in-progress
   cudnn/CUDNN_STATUS_RUNTIME_FP_OVERFLOW :fp-overflow
   cudnn/CUDNN_STATUS_VERSION_MISMATCH :version-mismatch})

(defn enc-nan-propagation ^long [nan]
  (if nan
    cudnn/CUDNN_PROPAGATE_NAN
    cudnn/CUDNN_NOT_PROPAGATE_NAN))

(defn dec-nan-propagation [^long nan]
  (if (= cudnn/CUDNN_PROPAGATE_NAN nan) true false))

(def ^:const cudnn-format
  {:nchw cudnn/CUDNN_TENSOR_NCHW
   :nhwc cudnn/CUDNN_TENSOR_NHWC
   :nchw-vect-c cudnn/CUDNN_TENSOR_NCHW_VECT_C})

(defn dec-format [^long format]
  (case format
    0 :nchw
    1 :nhwc
    2 :nchw-vect-c
    (dragan-says-ex "This format is not supported by cuDNN."
                    {:format format :available (keys cudnn-format)})))

(def ^:const cudnn-data-type
  {:float cudnn/CUDNN_DATA_FLOAT
   Float/TYPE cudnn/CUDNN_DATA_FLOAT
   Float cudnn/CUDNN_DATA_FLOAT
   :double cudnn/CUDNN_DATA_DOUBLE
   Double/TYPE cudnn/CUDNN_DATA_DOUBLE
   Double cudnn/CUDNN_DATA_DOUBLE
   :half cudnn/CUDNN_DATA_HALF
   :int8 cudnn/CUDNN_DATA_INT8
   :byte cudnn/CUDNN_DATA_INT8
   Byte/TYPE cudnn/CUDNN_DATA_INT8
   Byte cudnn/CUDNN_DATA_INT8
   :int cudnn/CUDNN_DATA_INT32
   Integer/TYPE cudnn/CUDNN_DATA_INT32
   Integer cudnn/CUDNN_DATA_INT32
   :int8x4 cudnn/CUDNN_DATA_INT8x4
   :uint8 cudnn/CUDNN_DATA_UINT8
   :u8 cudnn/CUDNN_DATA_UINT8
   :uint8x4 cudnn/CUDNN_DATA_UINT8x4
   :int8x32 cudnn/CUDNN_DATA_INT8x32})

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
  {:logistic cudnn/CUDNN_ACTIVATION_SIGMOID
   :sigmoid cudnn/CUDNN_ACTIVATION_SIGMOID
   :relu cudnn/CUDNN_ACTIVATION_RELU
   :tanh cudnn/CUDNN_ACTIVATION_TANH
   :clipped-relu cudnn/CUDNN_ACTIVATION_CLIPPED_RELU
   :elu cudnn/CUDNN_ACTIVATION_ELU
   :identity cudnn/CUDNN_ACTIVATION_IDENTITY
   :linear cudnn/CUDNN_ACTIVATION_IDENTITY})

(defn dec-activation-mode [^long mode]
  (case mode
    0 :logistic
    1 :relu
    2 :tanh
    3 :clipped-relu
    4 :elu
    5 :linear
    (dragan-says-ex "This mode is not supported by cuDNN."
                    {:mode mode :available (keys cudnn-activation-mode)})))

(def ^:const cudnn-reduce-tensor-op
  {:add cudnn/CUDNN_REDUCE_TENSOR_ADD
   :amax cudnn/CUDNN_REDUCE_TENSOR_AMAX
   :avg cudnn/CUDNN_REDUCE_TENSOR_AVG
   :max cudnn/CUDNN_REDUCE_TENSOR_MAX
   :min cudnn/CUDNN_REDUCE_TENSOR_MIN
   :mul cudnn/CUDNN_REDUCE_TENSOR_MUL
   :mul-no-zeros cudnn/CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS
   :norm1 cudnn/CUDNN_REDUCE_TENSOR_NORM1
   :norm2 cudnn/CUDNN_REDUCE_TENSOR_NORM2})

(def ^:const cudnn-reduce-tensor-indices
  {:flattened cudnn/CUDNN_REDUCE_TENSOR_FLATTENED_INDICES
   :no-indices cudnn/CUDNN_REDUCE_TENSOR_NO_INDICES})

(def ^:const cudnn-softmax-algorithm
  {:fast cudnn/CUDNN_SOFTMAX_FAST
   :accurate cudnn/CUDNN_SOFTMAX_ACCURATE
   :log cudnn/CUDNN_SOFTMAX_LOG})

(def ^:const cudnn-softmax-mode
  {:instance cudnn/CUDNN_SOFTMAX_MODE_INSTANCE
   :channel cudnn/CUDNN_SOFTMAX_MODE_CHANNEL})

(def ^:const cudnn-convolution-mode
  {:convolution cudnn/CUDNN_CONVOLUTION
   :cross-correleation cudnn/CUDNN_CROSS_CORRELATION})

(def ^:const cudnn-convolution-fwd-algo
  {:count cudnn/CUDNN_CONVOLUTION_FWD_ALGO_COUNT
   :direct cudnn/CUDNN_CONVOLUTION_FWD_ALGO_DIRECT
   :fft cudnn/CUDNN_CONVOLUTION_FWD_ALGO_FFT
   :fft-tiling cudnn/CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING
   :gemm cudnn/CUDNN_CONVOLUTION_FWD_ALGO_GEMM
   :implicit-gemm cudnn/CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM
   :implicit-precomp-gemm cudnn/CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
   :winograd cudnn/CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD
   :winograd-nonfused cudnn/CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED})

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


(def ^:const cudnn-convolution-bwd-data-algo
  {:algo0 cudnn/CUDNN_CONVOLUTION_BWD_DATA_ALGO_0
   :algo1 cudnn/CUDNN_CONVOLUTION_BWD_DATA_ALGO_1
   :count cudnn/CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT
   :fft cudnn/CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT
   :fft-tiling cudnn/CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING
   :winograd cudnn/CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD
   :winograd-nonfused cudnn/CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED})

(defn dec-convolution-bwd-data-algo [^long algo]
  (case algo
    0 :algo0
    1 :algo1
    2 :fft
    3 :fft-tiling
    4 :winograd
    5 :winograd-nonfused
    6 :count))

(def ^:const cudnn-convolution-bwd-filter-algo
  {:algo0 cudnn/CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0
   :algo1 cudnn/CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1
   :algo3 cudnn/CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3
   :count cudnn/CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT
   :fft cudnn/CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT
   :fft-tiling cudnn/CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING
   :winograd cudnn/CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD
   :winograd-nonfused cudnn/CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED})

(defn dec-convolution-bwd-filter-algo [^long algo]
  (case algo
    0 :algo0
    1 :algo1
    2 :fft
    3 :algo3
    4 :winograd
    5 :winograd-nonfused
    6 :fft-tiling
    7 :count))

(def ^:const cudnn-pooling-mode
  {:max cudnn/CUDNN_POOLING_MAX
   :max-deterministic cudnn/CUDNN_POOLING_MAX_DETERMINISTIC
   :avg cudnn/CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING
   :avg-padding cudnn/CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING
   :avg-include-padding cudnn/CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING
   :avg-exclude-padding cudnn/CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING})

(defn dec-math-type [^long math-type]
  (case math-type
    0 :default
    1 :tensor-op
    2 :tensor-op-allow-conversion
    3 :fma))

(def ^:const cudnn-math-type
  {:default cudnn/CUDNN_DEFAULT_MATH
   :tensor-op cudnn/CUDNN_TENSOR_OP_MATH
   :tensor-op-allow-conversion cudnn/CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION
   :fma cudnn/CUDNN_FMA_MATH})

(defn dec-determinism [^long determinism]
  (case determinism
    0 cudnn/CUDNN_NON_DETERMINISTIC
    1 cudnn/CUDNN_DETERMINISTIC))

(def ^:const cudnn-determinism
  {:non-deterministic cudnn/CUDNN_NON_DETERMINISTIC
   :deterministic cudnn/CUDNN_DETERMINISTIC})

(def ^:const cudnn-batch-norm-mode
  {:per-activation cudnn/CUDNN_BATCHNORM_PER_ACTIVATION
   :spatial cudnn/CUDNN_BATCHNORM_SPATIAL
   :spatial-persistent cudnn/CUDNN_BATCHNORM_SPATIAL_PERSISTENT})

(def ^:const cudnn-err-query-mode
  {:blocking cudnn/CUDNN_ERRQUERY_BLOCKING
   :non-blocking cudnn/CUDNN_ERRQUERY_NONBLOCKING
   :rawcode cudnn/CUDNN_ERRQUERY_RAWCODE})

(defn ^:const dec-err-query-mode [^long mode]
  (case mode
    0 :rawcode
    1 :non-blocking
    2 :blocking
    :unknown))

(def ^:const cudnn-rnn-algo-mode
  {:standard cudnn/CUDNN_RNN_ALGO_STANDARD
   :static cudnn/CUDNN_RNN_ALGO_PERSIST_STATIC
   :dynamic cudnn/CUDNN_RNN_ALGO_PERSIST_DYNAMIC
   :small cudnn/CUDNN_RNN_ALGO_PERSIST_STATIC_SMALL_H})

(defn ^:const dec-rnn-algo-mode [^long mode]
  (case mode
    0 :standard
    1 :static
    2 :dynamic
    3 :small
    :unknown))

(def ^:const cudnn-rnn-cell-mode
  {:relu cudnn/CUDNN_RNN_RELU
   :tanh cudnn/CUDNN_RNN_TANH
   :lstm cudnn/CUDNN_LSTM
   :gru  cudnn/CUDNN_GRU})

(defn ^:const dec-rnn-cell-mode [^long mode]
  (case mode
    0 :relu
    1 :tanh
    2 :lstm
    3 :gru
    :unknown))

(def ^:const cudnn-rnn-bias-mode
  {:no-bias cudnn/CUDNN_RNN_NO_BIAS
   :single cudnn/CUDNN_RNN_SINGLE_INP_BIAS
   :single-inp cudnn/CUDNN_RNN_SINGLE_INP_BIAS
   :double cudnn/CUDNN_RNN_DOUBLE_BIAS
   :single-rec cudnn/CUDNN_RNN_SINGLE_REC_BIAS})

(defn ^:const dec-rnn-bias-mode [^long mode]
  (case mode
    0 :no-bias
    1 :single
    2 :double
    3 :single-rec
    :unknown))

(def ^:const cudnn-direction-mode
  {:unidirectional cudnn/CUDNN_UNIDIRECTIONAL
   :bidirectional cudnn/CUDNN_BIDIRECTIONAL})

(defn ^:const dec-direction-mode [^long mode]
  (case mode
    0 :unidirectional
    1 :bidirectional
    :unknown))

(def ^:const cudnn-rnn-input-mode
  {:linear cudnn/CUDNN_LINEAR_INPUT
   :skip cudnn/CUDNN_SKIP_INPUT})

(defn ^:const dec-rnn-input-mode [^long mode]
  (case mode
    0 :linear
    1 :skip
    :unknown))

(def ^:const cudnn-rnn-aux-mode
  {:padded-io-disabled cudnn/CUDNN_RNN_PADDED_IO_DISABLED
   :padded-io-enabled cudnn/CUDNN_RNN_PADDED_IO_ENABLED})

(defn ^:const dec-rnn-aux-mode [^long mode]
  (case mode
    0 :padded-io-disabled
    1 :padded-io-enabled
    :unknown))

(def ^:const cudnn-forward-mode
  {:inference cudnn/CUDNN_FWD_MODE_INFERENCE
   :training cudnn/CUDNN_FWD_MODE_TRAINING})

(def ^:const cudnn-rnn-data-layout
  {:seq-mayor-unpacked 0
   :seq-mayor-packed 1
   :batch-mayor 2})

(def ^:const cudnn-grad-mode
  {:add cudnn/CUDNN_WGRAD_MODE_ADD
   :set cudnn/CUDNN_WGRAD_MODE_SET})
