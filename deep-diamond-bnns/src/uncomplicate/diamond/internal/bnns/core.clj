;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.diamond.internal.bnns.core
  (:require [uncomplicate.commons
             [core :refer [let-release with-release view Info bytesize size]]
             [utils :refer [enc-keyword dragan-says-ex mask]]]
            [uncomplicate.fluokitten.protocols :refer [extract]]
            [uncomplicate.clojure-cpp
             :refer [int-pointer long-pointer float-pointer pointer-pointer get-entry
                     fill! pointer-vec safe byte-pointer position! get-pointer put-entry!
                     type-pointer pointer position capacity null?]]
            [uncomplicate.diamond.internal.utils :refer [default-strides]]
            [uncomplicate.diamond.internal.bnns
             [impl :refer :all]
             [constants :refer :all]
             [protocols :refer :all]])
  (:import clojure.lang.ExceptionInfo
           uncomplicate.javacpp.accelerate.global.bnns
           [uncomplicate.javacpp.accelerate.global bnns$BNNSTensor bnns$BNNSNDArrayDescriptor]))

(defn tensor-descriptor [shape data-type layout]
  (let [rank (count shape)
        dtype (enc-keyword bnns-data-type data-type)]
    (if (keyword? layout)
      (let [nda (bnns$BNNSNDArrayDescriptor.)]
        (try
          (let [dlayout (enc-keyword bnns-data-layout layout)]
            (ndarray-descriptor* nda shape dtype dlayout)
            (->BNNSNDArrayDescriptor nda shape data-type layout true))
          (catch Exception e
            (.deallocate nda)
            (throw e))))
      (if (and (<= 0 rank bnns/BNNS_MAX_TENSOR_DIMENSION) (= rank (count layout)))
        (let [td (bnns$BNNSTensor.)]
          (try
            (tensor-descriptor* td shape dtype layout)
            (->BNNSTensorDescriptor td shape data-type layout true)
            (catch Exception e
              (.deallocate td)
              (throw e))))
        (dragan-says-ex (format "Shapes and strides must have equal rank between 0 and %s."
                                bnns/BNNS_MAX_TENSOR_DIMENSION)
                        {:shape shape :strides layout :max-rank bnns/BNNS_MAX_TENSOR_DIMENSION})))))
