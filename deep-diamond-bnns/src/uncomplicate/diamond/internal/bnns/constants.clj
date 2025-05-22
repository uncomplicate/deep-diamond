;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.diamond.internal.bnns.constants
  (:require [uncomplicate.commons.utils :refer [dragan-says-ex]])
  (:import uncomplicate.javacpp.accelerate.global.bnns))

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

(def ^:const bnns-data-layout
  {:x bnns/BNNSDataLayoutVector
   :row bnns/BNNSDataLayoutRowMajorMatrix
   :column bnns/BNNSDataLayoutColumnMajorMatrix
   :2d-last bnns/BNNSDataLayout2DLastMajor
   :2d-first bnns/BNNSDataLayout2DFirstMajor
   :sparse bnns/BNNSDataLayoutFullyConnectedSparse
   :chw bnns/BNNSDataLayoutImageCHW
   :sne bnns/BNNSDataLayoutSNE
   :nse bnns/BNNSDataLayoutNSE
   :mha-dhk bnns/BNNSDataLayoutMHA_DHK
   :3d-last bnns/BNNSDataLayout3DLastMajor
   :3d-first bnns/BNNSDataLayout3DFirstMajor
   :oihw bnns/BNNSDataLayoutConvolutionWeightsOIHW
   :oihrwr bnns/BNNSDataLayoutConvolutionWeightsOIHrWr
   :iohrr bnns/BNNSDataLayoutConvolutionWeightsIOHrWr
   :oihw-pack32 bnns/BNNSDataLayoutConvolutionWeightsOIHW_Pack32
   :4d-last bnns/BNNSDataLayout4DLastMajor
   :4d-first bnns/BNNSDataLayout4DFirstMajor
   :5d-last bnns/BNNSDataLayout5DLastMajor
   :5d-first bnns/BNNSDataLayout5DFirstMajor
   :6d-last bnns/BNNSDataLayout6DLastMajor
   :6d-first bnns/BNNSDataLayout6DFirstMajor
   :7d-last bnns/BNNSDataLayout7DLastMajor
   :7d-first bnns/BNNSDataLayout7DFirstMajor
   :8d-last bnns/BNNSDataLayout8DLastMajor
   :8d-first bnns/BNNSDataLayout8DFirstMajor})

(defn dec-data-layout [^long data-layout]
  (case data-layout
    0x10000 :x
    0x20000 :row
    0x20001 :column
    0x28000 :2d-last
    0x28001 :2d-first
    0x21001 :sparse
    0x30000 :chw
    0x30001 :sne
    0x30002 :nse
    0x30003 :mha-dhk
    0x38000 :3d-last
    0x38001 :3d-first
    0x40000 :oihw
    0x40001 :oihrwr
    0x40002 :iohrr
    0x40010 :oihw-pack32
    0x48000 :4d-last
    0x48001 :4d-first
    0x58000 :5d-last
    0x58001 :5d-first
    0x68000 :6d-last
    0x68001 :6d-first
    0x78000 :7d-last
    0x78001 :7d-first
    0x88000 :8d-last
    0x88001 :8d-first
    0 :undef
    (dragan-says-ex "Unknown data layout." {:data-layout data-layout})))
