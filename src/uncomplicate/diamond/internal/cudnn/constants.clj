;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.diamond.internal.cudnn.constants
  (:require [uncomplicate.commons.utils :refer [dragan-says-ex]])
  (:import [jcuda.jcudnn cudnnTensorFormat cudnnDataType]))

(defn dec-format [^long format]
  (case format
    0 :nchw
    1 :nhwc
    2 :nchw-vect-c
    (dragan-says-ex "This format is not supported by cuDNN. Please use another engine."
                    {:format format})))

(def ^:const cudnn-format
  {:nchw cudnnTensorFormat/CUDNN_TENSOR_NCHW
   :nhwc cudnnTensorFormat/CUDNN_TENSOR_NHWC
   :nchw-vect-c cudnnTensorFormat/CUDNN_TENSOR_NCHW_VECT_C})

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
    (dragan-says-ex "This data type is not supported by cuDNN. Please use another "
                    {:data-type data-type})))

(def ^:const cudnn-data-type
  {:float cudnnDataType/CUDNN_DATA_FLOAT
   :double cudnnDataType/CUDNN_DATA_DOUBLE
   :half cudnnDataType/CUDNN_DATA_HALF
   :byte cudnnDataType/CUDNN_DATA_INT8
   :int8 cudnnDataType/CUDNN_DATA_INT8
   :int cudnnDataType/CUDNN_DATA_INT32
   :int8x4 cudnnDataType/CUDNN_DATA_INT8x4
   :uint8 cudnnDataType/CUDNN_DATA_UINT8
   :u8 cudnnDataType/CUDNN_DATA_UINT8
   :uit8x4 cudnnDataType/CUDNN_DATA_UINT8x4
   :int8x32 cudnnDataType/CUDNN_DATA_INT8x32})
