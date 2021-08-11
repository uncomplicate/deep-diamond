;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.diamond.internal.dnnl.constants
  (:require [uncomplicate.commons.utils :refer [dragan-says-ex]])
  (:import org.bytedeco.dnnl.global.dnnl))

(defn dec-status [^long status]
  (case status
    0 :success
    1 :out-of-memory
    2 :invalid-arguments
    3 :unimplemented
    4 :iterator-ends
    5 :runtime-error
    6 :not-required
    :unknown))

(def ^:const dnnl-engine-kind
  {:cpu dnnl/dnnl_cpu
   :gpu dnnl/dnnl_gpu
   :any dnnl/dnnl_any_engine})

(defn dec-engine-kind [^long kind]
  (case kind
    1 :cpu
    2 :gpu
    0 :any
    :unknown))

(def ^:const dnnl-stream-flags
  {:default-order dnnl/dnnl_stream_default_flags
   :in-order dnnl/dnnl_stream_in_order
   :out-of-order dnnl/dnnl_stream_out_of_order
   :default-flags dnnl/dnnl_stream_default_flags})

(defn dec-primitive-kind [^long primitive-kind]
  (case primitive-kind
    0 :undef
    1 :reorder
    2 :shuffle
    3 :concat
    4 :sum
    5 :convolution
    6 :deconvolution
    7 :eltwise
    8 :softmax
    9 :pooling
    10 :lrn
    11 :batch-normalization
    12 :layer-normalization
    13 :inner-product
    14 :rnn
    15 :gemm
    16 :binary
    17 :logsoftmax
    18 :matmul
    19 :resampling
    20 :pooling-v2
    21 :reduction
    22 :prelu
    (dragan-says-ex "Unknown primitive kind." {:primitive-kind primitive-kind})))

(defn dec-format [^long format]
  (case format
    0 :undef
    1 :any
    2 :a
    3 :ab
    4 :abc
    5 :abcd
    6 :acbd
    7 :abcde
    8 :abcdef
    9 :abcdefg
    10 :abcdefgh
    11 :abcdefghi
    12 :abcdefghij
    13 :abcdefghijk
    14 :abcdefghijkl
    15 :abdc
    16 :abdec
    17 :acb
    18 :acbde
    19 :acbdef
    20 :acdb
    21 :acdeb
    22 :ba
    23 :bac
    24 :bacd
    25 :bacde
    26 :bca
    27 :bcda
    28 :bcdea
    29 :cba
    30 :cdba
    31 :dcab
    32 :cdeba
    33 :decab
    34 :defcab
    35 :abced
    36 :abcdfe
    37 :abcdegf
    38 :abcdefhg
    39 :abcdefgih
    40 :abcdefghji
    41 :abcdefghikj
    42 :abcdefghijlk
    (if (< 28 format dnnl/dnnl_format_tag_last) :opaque
        (dragan-says-ex "Unknown format." {:format format}))))

(def ^:const dnnl-format
  {:undef dnnl/dnnl_format_tag_undef
   :any dnnl/dnnl_format_tag_any
   :blocked dnnl/dnnl_blocked
   :x dnnl/dnnl_x
   :nc dnnl/dnnl_nc
   :cn dnnl/dnnl_cn
   :tn dnnl/dnnl_tn
   :nt dnnl/dnnl_nt
   :ncw dnnl/dnnl_ncw
   :nwc dnnl/dnnl_nwc
   :nchw dnnl/dnnl_nchw
   :nhwc dnnl/dnnl_nhwc
   :chwn dnnl/dnnl_chwn
   :ncdhw dnnl/dnnl_ncdhw
   :ndhwc dnnl/dnnl_ndhwc
   :oi dnnl/dnnl_oi
   :io dnnl/dnnl_io
   :oiw dnnl/dnnl_oiw
   :owi dnnl/dnnl_owi
   :wio dnnl/dnnl_wio
   :iwo dnnl/dnnl_iwo
   :oihw dnnl/dnnl_oihw
   :hwio dnnl/dnnl_hwio
   :ohwi dnnl/dnnl_ohwi
   :ihwo dnnl/dnnl_ihwo
   :iohw dnnl/dnnl_iohw
   :oidhw dnnl/dnnl_oidhw
   :iodhw dnnl/dnnl_iodhw
   :dhwio dnnl/dnnl_dhwio
   :odhwi dnnl/dnnl_odhwi
   :idhwo dnnl/dnnl_idhwo
   :goiw dnnl/dnnl_goiw
   :gowi dnnl/dnnl_gowi
   :wigo dnnl/dnnl_wigo
   :goihw dnnl/dnnl_goihw
   :hwigo dnnl/dnnl_hwigo
   :giohw dnnl/dnnl_giohw
   :goidhw dnnl/dnnl_goidhw
   :giodhw dnnl/dnnl_giodhw
   :dhwigo dnnl/dnnl_dhwigo
   :tnc dnnl/dnnl_tnc
   :ntc dnnl/dnnl_ntc
   :ldnc dnnl/dnnl_ldnc
   :ldigo dnnl/dnnl_ldigo
   :ldgoi dnnl/dnnl_ldgoi
   :ldio dnnl/dnnl_ldio
   :ldoi dnnl/dnnl_ldoi
   :ldgo dnnl/dnnl_ldgo
   :a dnnl/dnnl_a
   :ab dnnl/dnnl_ab
   :abc dnnl/dnnl_abc
   :abcd dnnl/dnnl_abcd
   :acbd dnnl/dnnl_acbd
   :abcde dnnl/dnnl_abcde
   :abcdef dnnl/dnnl_abcdef
   :abcdefg dnnl/dnnl_abcdefg
   :abcdefgh dnnl/dnnl_abcdefgh
   :abcdefghi dnnl/dnnl_abcdefghi
   :abcdefghij dnnl/dnnl_abcdefghij
   :abcdefghijk dnnl/dnnl_abcdefghijk
   :abcdefghijkl dnnl/dnnl_abcdefghijkl
   :abdc dnnl/dnnl_abdc
   :abdec dnnl/dnnl_abdec
   :acb dnnl/dnnl_acb
   :acbde dnnl/dnnl_acbde
   :acbdef dnnl/dnnl_acbdef
   :acdb dnnl/dnnl_acdb
   :acdeb dnnl/dnnl_acdeb
   :ba dnnl/dnnl_ba
   :bac dnnl/dnnl_bac
   :bacd dnnl/dnnl_bacd
   :bacde dnnl/dnnl_bacde
   :bca dnnl/dnnl_bca
   :bcda dnnl/dnnl_bcda
   :bcdea dnnl/dnnl_bcdea
   :cba dnnl/dnnl_cba
   :cdba dnnl/dnnl_cdba
   :dcab dnnl/dnnl_dcab
   :cdeba dnnl/dnnl_cdeba
   :decab dnnl/dnnl_decab
   :defcab dnnl/dnnl_defcab
   :abced dnnl/dnnl_abced
   :abcdfe dnnl/dnnl_abcdfe
   :abcdegf dnnl/dnnl_abcdegf
   :abcdefhg dnnl/dnnl_abcdefhg
   :abcdefgih dnnl/dnnl_abcdefgih
   :abcdefghji dnnl/dnnl_abcdefghji
   :abcdefghikj dnnl/dnnl_abcdefghikj
   :abcdefghijlk dnnl/dnnl_abcdefghijlk})

(defn dec-data-type [^long data-type]
  (case data-type
    3 :float
    1 :half
    2 :bf16
    4 :int
    5 :byte
    6 :uint8
    0 :undef
    (dragan-says-ex "Unknown data type." {:data-type data-type})))

(def ^:const dnnl-data-type
  {:float dnnl/dnnl_f32
   Float/TYPE dnnl/dnnl_f32
   Float dnnl/dnnl_f32
   :half dnnl/dnnl_f16
   :f16 dnnl/dnnl_f16
   :bf16 dnnl/dnnl_bf16
   :int dnnl/dnnl_s32
   Integer/TYPE dnnl/dnnl_s32
   Integer dnnl/dnnl_s32
   :byte dnnl/dnnl_s8
   Byte/TYPE dnnl/dnnl_s8
   Byte dnnl/dnnl_s8
   :u8 dnnl/dnnl_u8
   :uint8 dnnl/dnnl_u8
   :undef dnnl/dnnl_data_type_undef})

(def ^:const dnnl-forward-prop-kind
  {:training dnnl/dnnl_forward_training
   :inference dnnl/dnnl_forward_inference
   :scoring dnnl/dnnl_forward_scoring})

(def ^:const dnnl-backward-prop-kind
  {:backward dnnl/dnnl_backward
   :data dnnl/dnnl_backward_data
   :weights dnnl/dnnl_backward_weights
   :bias dnnl/dnnl_backward_bias})

(def ^:const dnnl-eltwise-alg-kind
  {:relu dnnl/dnnl_eltwise_relu
   :tanh dnnl/dnnl_eltwise_tanh
   :elu dnnl/dnnl_eltwise_elu
   :square dnnl/dnnl_eltwise_square
   :abs dnnl/dnnl_eltwise_abs
   :sqrt dnnl/dnnl_eltwise_sqrt
   :linear dnnl/dnnl_eltwise_linear
   :bounded-relu dnnl/dnnl_eltwise_bounded_relu
   :soft-relu dnnl/dnnl_eltwise_soft_relu
   :logistic dnnl/dnnl_eltwise_logistic
   :sigmoid dnnl/dnnl_eltwise_logistic
   :exp dnnl/dnnl_eltwise_exp
   :gelu dnnl/dnnl_eltwise_gelu
   :gelu-tanh dnnl/dnnl_eltwise_gelu_tanh
   :gelu-erf dnnl/dnnl_eltwise_gelu_erf
   :swish dnnl/dnnl_eltwise_swish
   :hardswish dnnl/dnnl_eltwise_hardswish
   :log dnnl/dnnl_eltwise_log
   :clip dnnl/dnnl_eltwise_clip
   :pow dnnl/dnnl_eltwise_pow
   :logsigmoid dnnl/dnnl_eltwise_logsigmoid
   :mish dnnl/dnnl_eltwise_mish
   :clip-v2 dnnl/dnnl_eltwise_clip_v2
   :relu-dst-bwd dnnl/dnnl_eltwise_relu_use_dst_for_bwd
   :exp-dst-bwd dnnl/dnnl_eltwise_exp_use_dst_for_bwd
   :tanh-dst-bwd dnnl/dnnl_eltwise_tanh_use_dst_for_bwd
   :elu-dst-bwd dnnl/dnnl_eltwise_elu_use_dst_for_bwd
   :square-dst-bwd dnnl/dnnl_eltwise_sqrt_use_dst_for_bwd
   :logistic-dst-bwd dnnl/dnnl_eltwise_logistic_use_dst_for_bwd
   :sigmoid-dst-bwd dnnl/dnnl_eltwise_logistic_use_dst_for_bwd
   :clip-v2-dst-bwd dnnl/dnnl_eltwise_clip_v2_use_dst_for_bwd
   :round dnnl/dnnl_eltwise_round})

(defn entry-bytes ^long [data-type]
  (case data-type
    :float Float/BYTES
    :half 2
    :f16 2
    :bf16 2
    :int Integer/BYTES
    :byte 1
    :u8 1
    :uint8 1
    (dragan-says-ex "unknown data type" {:data-type data-type})))

(def ^:const dnnl-convolution-alg-kind
  {:auto dnnl/dnnl_convolution_auto
   :direct dnnl/dnnl_convolution_direct
   :winograd dnnl/dnnl_convolution_winograd})

(def ^:const dnnl-deconvolution-alg-kind
  {:direct dnnl/dnnl_deconvolution_direct
   :winograd dnnl/dnnl_deconvolution_winograd})

(def ^:const dnnl-pooling-alg-kind
  {:max dnnl/dnnl_pooling_max
   :avg dnnl/dnnl_pooling_avg
   :avg-padding dnnl/dnnl_pooling_avg_include_padding
   :avg-include-padding dnnl/dnnl_pooling_avg_include_padding
   :avg-exclude-padding dnnl/dnnl_pooling_avg_exclude_padding})

(def ^:const dnnl-normalization-flags
  {:none dnnl/dnnl_normalization_flags_none
   :global-stats dnnl/dnnl_use_global_stats
   :scaleshift dnnl/dnnl_use_scaleshift
   :fuse-relu dnnl/dnnl_fuse_norm_relu})

(def ^:const dnnl-binary-alg-kind
  {:add dnnl/dnnl_binary_add
   :mul dnnl/dnnl_binary_mul
   :max dnnl/dnnl_binary_max
   :min dnnl/dnnl_binary_min
   :div dnnl/dnnl_binary_div
   :sub dnnl/dnnl_binary_sub
   :ge dnnl/dnnl_binary_ge
   :gt dnnl/dnnl_binary_gt
   :le dnnl/dnnl_binary_le
   :lt dnnl/dnnl_binary_lt
   :eq dnnl/dnnl_binary_eq
   :ne dnnl/dnnl_binary_ne})
