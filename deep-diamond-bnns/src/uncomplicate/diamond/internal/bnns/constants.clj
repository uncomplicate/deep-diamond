;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.diamond.internal.bnns.constants
  (:require [uncomplicate.commons.utils :refer [dragan-says-ex]]
            [uncomplicate.clojure-cpp
             :refer [ptr* byte-pointer short-pointer int-pointer float-pointer
                     type-pointer]])
  (:import [uncomplicate.javacpp.accelerate.global bnns bnns$BNNSArithmeticUnary
            bnns$BNNSArithmeticBinary bnns$BNNSArithmeticTernary]))

(defn dec-data-type [^long data-type]
  (case data-type
    65568 :float
    65552 :half
    98320 :bf16
    131104 :int
    131080 :byte
    262152 :uint8
    262160 :uint16
    262176 :uint32
    0 :undef
    (dragan-says-ex "Unknown data type." {:data-type data-type})))

(def ^:const bnns-data-type
  {:float bnns/BNNSDataTypeFloat32
   Float/TYPE bnns/BNNSDataTypeFloat32
   Float bnns/BNNSDataTypeFloat32
   :half bnns/BNNSDataTypeFloat16
   :f16 bnns/BNNSDataTypeFloat16
   :bf16 bnns/BNNSDataTypeBFloat16
   :int bnns/BNNSDataTypeInt32
   Integer/TYPE bnns/BNNSDataTypeInt32
   Integer bnns/BNNSDataTypeInt32
   :byte bnns/BNNSDataTypeInt8
   Byte/TYPE bnns/BNNSDataTypeInt8
   Byte bnns/BNNSDataTypeInt8
   :u8 bnns/BNNSDataTypeUInt8
   :uint8 bnns/BNNSDataTypeUInt8
   :u16 bnns/BNNSDataTypeUInt16
   :uint16 bnns/BNNSDataTypeUInt16
   :u32 bnns/BNNSDataTypeUInt32
   :uint32 bnns/BNNSDataTypeUInt32})

(def ^:const bnns-data-type-size
  {:float 4
   :half 2
   :f16 2
   :bf16 2
   :int 4
   :byte 1
   :u8 1
   :uint8 1
   :u16 2
   :uint16 2
   :u32 4
   :uint32 4
   bnns/BNNSDataTypeFloat32 4
   bnns/BNNSDataTypeFloat16 2
   bnns/BNNSDataTypeBFloat16 2
   bnns/BNNSDataTypeInt32 4
   bnns/BNNSDataTypeInt8 1
   bnns/BNNSDataTypeUInt16 2
   bnns/BNNSDataTypeUInt32 4})

(defn bnns-data-type-pointer [t]
  (or (type-pointer t)
      (case t
        65568 float-pointer
        :half short-pointer
        65552 short-pointer
        :f16 short-pointer
        :bf16 short-pointer
        131104 int-pointer
        Integer int-pointer
  o131080 byte-pointer
        Byte/TYPE byte-pointer
        Byte byte-pointer
        :u8 byte-pointer
        262152 byte-pointer
        :uint8 byte-pointer
        :u16 short-pointer
        262160 short-pointer
        :uint16 short-pointer
        :u32 int-pointer
        262176 int-pointer
        :uint32 int-pointer
        ptr*)))

(def ^:const bnns-data-layout
  {:a bnns/BNNSDataLayoutVector
   :x bnns/BNNSDataLayoutVector
   :row bnns/BNNSDataLayoutRowMajorMatrix
   :column bnns/BNNSDataLayoutColumnMajorMatrix
   :nc bnns/BNNSDataLayout2DLastMajor
   :cn bnns/BNNSDataLayoutRowMajorMatrix
   :oi bnns/BNNSDataLayout2DLastMajor
   :io bnns/BNNSDataLayoutRowMajorMatrix
   :ab bnns/BNNSDataLayout2DLastMajor
   :ba bnns/BNNSDataLayout2DFirstMajor
   :sparse bnns/BNNSDataLayoutFullyConnectedSparse
   :chw bnns/BNNSDataLayoutImageCHW
   :sne bnns/BNNSDataLayoutSNE
   :tnc bnns/BNNSDataLayoutSNE
   :nse bnns/BNNSDataLayoutNSE
   :ntc bnns/BNNSDataLayoutNSE
   :mha-dhk bnns/BNNSDataLayoutMHA_DHK
   :abc bnns/BNNSDataLayout3DLastMajor
   :cba bnns/BNNSDataLayout3DFirstMajor
   :oihw bnns/BNNSDataLayoutConvolutionWeightsOIHW
   :oihrwr bnns/BNNSDataLayoutConvolutionWeightsOIHrWr
   :iohrwr bnns/BNNSDataLayoutConvolutionWeightsIOHrWr
   :oihw-pack32 bnns/BNNSDataLayoutConvolutionWeightsOIHW_Pack32
   :abcd bnns/BNNSDataLayout4DLastMajor
   :dcba bnns/BNNSDataLayout4DFirstMajor
   :nchw bnns/BNNSDataLayout4DLastMajor
   :abcde bnns/BNNSDataLayout5DLastMajor
   :edcba bnns/BNNSDataLayout5DFirstMajor
   :abcdef bnns/BNNSDataLayout6DLastMajor
   :fedcba bnns/BNNSDataLayout6DFirstMajor
   :abcdefg bnns/BNNSDataLayout7DLastMajor
   :gfedcba bnns/BNNSDataLayout7DFirstMajor
   :abcdefgh bnns/BNNSDataLayout8DLastMajor
   :hgfedcba bnns/BNNSDataLayout8DFirstMajor})

(defn bnns-default-layout
  ([major? rank]
   (if major?
     (bnns-default-layout rank)
     (case rank
       0 :a
       1 :a
       2 :ba
       3 :cba
       4 :dcba
       5 :edcba
       6 :fedcba
       7 :gfedcba
       8 :hgfedcba)))
  ([rank]
   (case rank
     0 :a
     1 :a
     2 :ab
     3 :abc
     4 :abcd
     5 :abcde
     6 :abcdef
     7 :abcdefg
     8 :abcdefgh)))

(defn dec-data-layout [^long data-layout]
  (case data-layout
    0x10000 :x
    0x20000 :row
    0x20001 :column
    0x28000 :ab
    0x28001 :ba
    0x21001 :sparse
    0x30000 :chw
    0x30001 :sne
    0x30002 :nse
    0x30003 :mha-dhk
    0x38000 :abc
    0x38001 :cba
    0x40000 :oihw
    0x40001 :oihrwr
    0x40002 :iohrwr
    0x40010 :oihw-pack32
    0x48000 :abcd
    0x48001 :dcba
    0x58000 :abcde
    0x58001 :edcba
    0x68000 :abcdef
    0x68001 :fedcba
    0x78000 :abcdefg
    0x78001 :gfedcba
    0x88000 :abcdefgh
    0x88001 :hgfedcba
    0 :undef
    (dragan-says-ex "Unknown data layout." {:data-layout data-layout})))

(def ^:const bnns-activation-function-enc
  {:identity bnns/BNNSActivationFunctionIdentity
   :relu bnns/BNNSActivationFunctionRectifiedLinear
   :lrelu bnns/BNNSActivationFunctionLeakyRectifiedLinear
   :sigmoid bnns/BNNSActivationFunctionSigmoid
   :logistic bnns/BNNSActivationFunctionSigmoid
   :tanh bnns/BNNSActivationFunctionTanh
   :scaled-tanh bnns/BNNSActivationFunctionScaledTanh
   :abs bnns/BNNSActivationFunctionAbs
   :linear bnns/BNNSActivationFunctionLinear
   :clamp bnns/BNNSActivationFunctionClamp
   :linear-saturate bnns/BNNSActivationFunctionIntegerLinearSaturate
   :linear-saturate-channel bnns/BNNSActivationFunctionIntegerLinearSaturatePerChannel
   :softmax bnns/BNNSActivationFunctionSoftmax
   :gelu-approx bnns/BNNSActivationFunctionGELUApproximation
   :gumbel bnns/BNNSActivationFunctionGumbel
   :gumbel-max bnns/BNNSActivationFunctionGumbelMax
   :softplus bnns/BNNSActivationFunctionSoftplus
   :hard-sigmoid bnns/BNNSActivationFunctionHardSigmoid
   :softsign bnns/BNNSActivationFunctionSoftsign
   :elu bnns/BNNSActivationFunctionELU
   :clamp-lrelu bnns/BNNSActivationFunctionClampedLeakyRectifiedLinear
   :linear-bias bnns/BNNSActivationFunctionLinearWithBias
   :log-softmax bnns/BNNSActivationFunctionLogSoftmax
   :log-sigmoid bnns/BNNSActivationFunctionLogSigmoid
   :selu bnns/BNNSActivationFunctionSELU
   :celu bnns/BNNSActivationFunctionCELU
   :hard-shrink bnns/BNNSActivationFunctionHardShrink
   :soft-shrink bnns/BNNSActivationFunctionSoftShrink
   :tanh-shrink bnns/BNNSActivationFunctionTanhShrink
   :threshold bnns/BNNSActivationFunctionThreshold
   :prelu-channel bnns/BNNSActivationFunctionPReLUPerChannel
   :hard-swish bnns/BNNSActivationFunctionHardSwish
   :silu bnns/BNNSActivationFunctionSiLU
   :relu6 bnns/BNNSActivationFunctionReLU6
   :erf bnns/BNNSActivationFunctionErf
   :gelu bnns/BNNSActivationFunctionGELU
   :gelu-approx-sigmoid bnns/BNNSActivationFunctionGELUApproximationSigmoid})

(def ^:const bnns-activation-function-dec
  (clojure.set/map-invert bnns-activation-function-enc))

(def ^:const bnns-descriptor-type
  {:constant bnns/BNNSConstant
   :sample bnns/BNNSSample
   :parameter bnns/BNNSParameter})

(def ^:const bnns-arithmetic-function-enc
  {:add bnns/BNNSArithmeticAdd
   :sub bnns/BNNSArithmeticSubtract
   :mult bnns/BNNSArithmeticMultiply
   :div bnns/BNNSArithmeticDivide
   :sqrt bnns/BNNSArithmeticSquareRoot
   :inv-sqrt bnns/BNNSArithmeticReciprocalSquareRoot
   :ceil bnns/BNNSArithmeticCeil
   :floor bnns/BNNSArithmeticFloor
   :round bnns/BNNSArithmeticRound
   :sin bnns/BNNSArithmeticSin
   :cos bnns/BNNSArithmeticCos
   :tan bnns/BNNSArithmeticTan
   :asin bnns/BNNSArithmeticAsin
   :acos bnns/BNNSArithmeticAcos
   :atan bnns/BNNSArithmeticAtan
   :sinh bnns/BNNSArithmeticSinh
   :cosh bnns/BNNSArithmeticCosh
   :tanh bnns/BNNSArithmeticTanh
   :asinh bnns/BNNSArithmeticAsinh
   :acosh bnns/BNNSArithmeticAcosh
   :atanh bnns/BNNSArithmeticAtanh
   :pow bnns/BNNSArithmeticPow
   :exp bnns/BNNSArithmeticExp
   :exp2 bnns/BNNSArithmeticExp2
   :log bnns/BNNSArithmeticLog
   :log2 bnns/BNNSArithmeticLog2
   :mult-no-nan bnns/BNNSArithmeticMultiplyNoNaN
   :div-no-nan bnns/BNNSArithmeticDivideNoNaN
   :mult-add bnns/BNNSArithmeticMultiplyAdd
   :min bnns/BNNSArithmeticMinimum
   :max bnns/BNNSArithmeticMaximum
   :select bnns/BNNSArithmeticSelect
   :abs bnns/BNNSArithmeticAbs
   :sign bnns/BNNSArithmeticSign
   :negate bnns/BNNSArithmeticNegate
   :inv bnns/BNNSArithmeticReciprocal
   :sqr bnns/BNNSArithmeticSquare
   :floor-div bnns/BNNSArithmeticFloorDivide
   :trunc-div bnns/BNNSArithmeticTruncDivide
   :trunc-rem bnns/BNNSArithmeticTruncRemainder
   :erf bnns/BNNSArithmeticErf})

(def ^:const bnns-arithmetic-functon-dec
  (clojure.set/map-invert bnns-arithmetic-function-enc))

(def ^:const bnns-arithmetic-function-fields
  {bnns/BNNSArithmeticAdd bnns$BNNSArithmeticBinary
   bnns/BNNSArithmeticSubtract bnns$BNNSArithmeticBinary
   bnns/BNNSArithmeticMultiply bnns$BNNSArithmeticBinary
   bnns/BNNSArithmeticDivide bnns$BNNSArithmeticBinary
   bnns/BNNSArithmeticSquareRoot bnns$BNNSArithmeticUnary
   bnns/BNNSArithmeticReciprocalSquareRoot bnns$BNNSArithmeticUnary
   bnns/BNNSArithmeticCeil bnns$BNNSArithmeticUnary
   bnns/BNNSArithmeticFloor bnns$BNNSArithmeticUnary
   bnns/BNNSArithmeticRound bnns$BNNSArithmeticUnary
   bnns/BNNSArithmeticSin bnns$BNNSArithmeticUnary
   bnns/BNNSArithmeticCos bnns$BNNSArithmeticUnary
   bnns/BNNSArithmeticTan bnns$BNNSArithmeticUnary
   bnns/BNNSArithmeticAsin bnns$BNNSArithmeticUnary
   bnns/BNNSArithmeticAcos bnns$BNNSArithmeticUnary
   bnns/BNNSArithmeticAtan bnns$BNNSArithmeticUnary
   bnns/BNNSArithmeticSinh bnns$BNNSArithmeticUnary
   bnns/BNNSArithmeticCosh bnns$BNNSArithmeticUnary
   bnns/BNNSArithmeticTanh bnns$BNNSArithmeticUnary
   bnns/BNNSArithmeticAsinh bnns$BNNSArithmeticUnary
   bnns/BNNSArithmeticAcosh bnns$BNNSArithmeticUnary
   bnns/BNNSArithmeticAtanh bnns$BNNSArithmeticUnary
   bnns/BNNSArithmeticPow bnns$BNNSArithmeticUnary
   bnns/BNNSArithmeticExp bnns$BNNSArithmeticUnary
   bnns/BNNSArithmeticExp2 bnns$BNNSArithmeticUnary
   bnns/BNNSArithmeticLog bnns$BNNSArithmeticUnary
   bnns/BNNSArithmeticLog2 bnns$BNNSArithmeticUnary
   bnns/BNNSArithmeticMultiplyNoNaN bnns$BNNSArithmeticBinary
   bnns/BNNSArithmeticDivideNoNaN bnns$BNNSArithmeticBinary
   bnns/BNNSArithmeticMultiplyAdd bnns$BNNSArithmeticTernary
   bnns/BNNSArithmeticMinimum bnns$BNNSArithmeticBinary
   bnns/BNNSArithmeticMaximum bnns$BNNSArithmeticBinary
   bnns/BNNSArithmeticSelect bnns$BNNSArithmeticTernary
   bnns/BNNSArithmeticAbs bnns$BNNSArithmeticUnary
   bnns/BNNSArithmeticSign bnns$BNNSArithmeticUnary
   bnns/BNNSArithmeticNegate bnns$BNNSArithmeticUnary
   bnns/BNNSArithmeticReciprocal bnns$BNNSArithmeticUnary
   bnns/BNNSArithmeticSquare bnns$BNNSArithmeticUnary
   bnns/BNNSArithmeticFloorDivide bnns$BNNSArithmeticBinary
   bnns/BNNSArithmeticTruncDivide bnns$BNNSArithmeticBinary
   bnns/BNNSArithmeticTruncRemainder bnns$BNNSArithmeticBinary
   bnns/BNNSArithmeticErf bnns$BNNSArithmeticUnary})

(def ^:const bnns-filter-type
  {:convo bnns/BNNSConvolution
   :convolution bnns/BNNSConvolution
   :dense bnns/BNNSFullyConnected
   :fc bnns/BNNSFullyConnected
   :fully-connected bnns/BNNSFullyConnected
   :batch-norm bnns/BNNSBatchNorm
   :instance-norm bnns/BNNSInstanceNorm
   :layer-norm bnns/BNNSLayerNorm
   :group-norm bnns/BNNSGroupNorm
   :trans-convo bnns/BNNSTransposedConvolution
   :transposed-convolution bnns/BNNSTransposedConvolution
   :quant bnns/BNNSQuantization
   :quantization bnns/BNNSQuantization
   :arithmetic bnns/BNNSArithmetic})
