;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.diamond.internal.bnns.core
  (:require [uncomplicate.commons
             [core :refer [let-release with-release Viewable view Info bytesize size]]
             [utils :refer [enc-keyword dragan-says-ex mask]]]
            [uncomplicate.fluokitten.protocols :refer [extract]]
            [uncomplicate.clojure-cpp
             :refer [ptr* size-t-pointer int-pointer long-pointer float-pointer
                     pointer-pointer get-entry zero! pointer-vec safe byte-pointer
                     position! get-pointer put-entry! pointer pointer-seq
                     position capacity capacity! null?]]
            [uncomplicate.diamond.internal.utils :refer [default-strides]]
            [uncomplicate.diamond.internal.bnns
             [impl :refer :all]
             [constants :refer :all]
             [protocols :refer :all]])
  (:import clojure.lang.ExceptionInfo
           [uncomplicate.javacpp.accelerate.global bnns bnns$BNNSFilterParameters
            bnns$BNNSLayerParametersActivation bnns$BNNSLayerParametersArithmetic
            bnns$BNNSFilter bnns$BNNSActivation
            bnns$BNNSLayerParametersFullyConnected bnns$BNNSLayerParametersConvolution]
           [uncomplicate.diamond.internal.bnns.impl BnnsTensorImpl
            BnnsTensorDescriptorImpl BnnsNdArrayDescriptorImpl]))

(defn equal-desc?
  "Compares two BNNS descriptor providers `td1` and `td2` for descriptro equality.
  `td1` and `td2` can be any objects that can provide BNNS descriptors (tensors,
  descriptors themselves,  etc.)"
  [td1 td2]
  (= (desc td1) (desc td2)))

(defn compatible-desc?
  "Compares two BNNS descriptor providers `td1` and `td2` for descriptro compatibility.
  Descriptors are compatible if they have equal shape, data type, and layout.
  `td1` and `td2` can be any objects that can provide BNNS descriptors (tensors,
  descriptors themselves,  etc.)"
  [td1 td2]
  (let [d1 (desc td1)
        d2 (desc td2)]
    (and (= (pointer-seq (dims* d1)) (pointer-seq (dims* d2)))
         (= (data-type* d1) (data-type* d2))
         (= (layout* d1) (layout* d2)))))

(defn data-type
  "Queries the data type of a Descriptor."
  [nd]
  (dec-data-type (data-type* (desc nd))))

(defn layout
  "Queries the layout of a Descriptor."
  [nd]
  (dec-data-layout (layout* (desc nd))))

(defn layout!
  [nd layout]
  (layout* (desc nd) (enc-keyword bnns-data-layout layout)))

(defn dims
  "Queries the dimensions of a Descriptor."
  [nd]
  (let [dsc (desc nd)]
    (vec ((if (major* dsc) reverse identity)
          (pointer-vec (dims* dsc))))))

(defn rank
  "Queries the rank of a Descriptor."
  [nd]
  (rank* (desc nd)))

(defn major-strides? [strd]
  (if (seq strd)
    (let [strd (remove zero? strd)]
      (if (seq strd)
        (or (apply = strd) (not (apply <= strd)))
        true))
    false))

(defn strides
  "Queries the strides of a Descriptor."
  [nd]
  (let [dsc (desc nd)]
    (vec ((if (major* dsc) reverse identity)
          (pointer-vec (strides* dsc))))))

(defn data [d]
  (let [dsc (desc d)]
    (when-let [res (data* dsc)]
      (capacity! ((bnns-data-type-pointer (data-type* dsc)) res)
                 (size dsc)))))

(defprotocol Parameters
  (w-desc [this])
  (i-desc [this])
  (o-desc [this])
  (b-desc [this]))

(defmacro extend-bnns-parameters [t]
  `(extend-type ~t
     Parameters
     (i-desc [this#]
       (.i_desc (extract this#)))
     (w-desc [this#]
       (.w_desc (extract this#)))
     (b-desc [this#]
       (.bias (extract this#)))
     (o-desc [this#]
       (.o_desc (extract this#)))))

(extend-type BnnsNdArrayDescriptorImpl
  Info
  (info
    ([this info-type]
     (case info-type
       :class (class this)
       :device :cpu
       :shape (dims this)
       :data-type (data-type this)
       :layout (layout this)
       :strides (strides this)
       nil))
    ([this]
     {:class (class this)
      :device :cpu
      :shape (dims this)
      :data-type (data-type this)
      :layout (layout this)
      :strides (strides this)})))

(defmacro extend-tensor-info [t]
  `(extend-type ~t
     Info
     (info
       ([this# info-type#]
        (case info-type#
          :class (class this#)
          :device :cpu
          :shape (dims this#)
          :data-type (data-type this#)
          :strides (strides this#)
          nil))
       ([this#]
        {:class (class this#)
         :device :cpu
         :shape (dims this#)
         :data-type (data-type this#)
         :strides (strides this#)}))))

(extend-tensor-info BnnsTensorDescriptorImpl)

(defn bnns-default-strides [major? shape]
  (default-strides ((if major? identity reverse) shape)))

(defn nda-desc
  ([shape data-type layout strides]
   (let [rank (count shape)
         dtype (enc-keyword bnns-data-type data-type)
         dlayout (enc-keyword bnns-data-layout
                              (or layout
                                  (bnns-default-layout (major-strides? strides) rank)))
         shape ((if (major? dlayout) reverse identity) shape)
         strides (or (seq (reverse strides)) (repeat rank 0))]
     (if (<= 0 (count strides) rank bnns/BNNS_MAX_TENSOR_DIMENSION)
       (let-release [shape (size-t-pointer shape)
                     strides (size-t-pointer strides)]
         (->BnnsNdArrayDescriptorImpl (ndarray-descriptor* shape dtype dlayout strides)
                                      rank true))
       (dragan-says-ex (format "Shapes must have rank between 0 and %s, while strides can have less than or equeal count as shapes."
                               bnns/BNNS_MAX_TENSOR_DIMENSION)
                       {:shape shape :layout layout :strides strides :max-rank bnns/BNNS_MAX_TENSOR_DIMENSION}))))
  ([shape data-type layout]
   (if (keyword? layout)
     (nda-desc shape data-type layout (default-strides shape))
     (nda-desc shape data-type nil layout)))
  ([shape data-type]
   (nda-desc shape data-type nil (default-strides shape)))
  ([shape]
   (nda-desc shape :float)))

(defn bnns-default-desc [dsc]
  (let [dsc (desc dsc)
        shp (dims dsc)]
    (nda-desc shp (data-type dsc) (layout dsc)
              (default-strides shp))))

(defn tensor-desc
  ([shape data-type strides]
   (let[rank (count shape)
        shape (reverse shape)
        dtype (enc-keyword bnns-data-type data-type)
        strides (or (seq (reverse strides)) (repeat rank 0))]
     (if (<= 0 (count strides) rank bnns/BNNS_MAX_TENSOR_DIMENSION)
       (let-release [shape (size-t-pointer shape)
                     strides (size-t-pointer strides)]
         (->BnnsTensorDescriptorImpl (tensor-descriptor* shape dtype strides)
                                     rank true))
       (dragan-says-ex (format "Shapes and strides must have equal rank between 0 and %s."
                               bnns/BNNS_MAX_TENSOR_DIMENSION)
                       {:shape shape :strides strides :max-rank bnns/BNNS_MAX_TENSOR_DIMENSION}))))
  ([shape data-type]
   (tensor-desc shape data-type (default-strides shape)))
  ([shape]
   (tensor-desc shape :float)))

(defn bnns-contiguous-nda-desc [dsc]
  (let [shape (dims dsc)]
    (if (= (size dsc) (long (apply * shape)))
      (view (desc dsc))
      (nda-desc shape :float (default-strides shape)))))

(defn bnns-contiguous-tensor-desc [dsc]
  (let [shape (dims dsc)]
    (if (= (size dsc) (long (apply * shape)))
      (view (desc dsc))
      (tensor-desc shape :float (default-strides shape)))))

(defn tensor
  "Used just in testing/api dscoveiry, not in DD implementations."
  ([shape data-type layout strides data master]
   (let [dsc (if layout
               (nda-desc shape data-type layout strides)
               (tensor-desc shape data-type strides))]
     (if (<= 0 (bytesize dsc) (bytesize data))
       (let [data-pointer (pointer data 0)]
         (data* dsc data-pointer)
         (->BnnsTensorImpl dsc data-pointer master))
       (dragan-says-ex "The buffer has to be large enough for the descriptor."
                       {:desc-bytes (bytesize dsc) :buffer-bytes (bytesize data)}))))
  ([shape data-type layout strides data]
   (tensor shape data-type layout strides data false))
  ([shape data-type layout strides]
   (let-release [buf ((bnns-data-type-pointer data-type)
                      ((if layout
                         (nda-shape-size (major? layout) shape size)
                         (tensor-shape-size shape strides))))]
     (tensor shape data-type layout strides (safe buf) true)))
  ([dsc data master]
   (let [dsc (desc dsc)]
     (if (and (<= 0 (bytesize dsc) (bytesize data)))
       (let [data-pointer (pointer data 0)
             dsc (clone* dsc)]
         (data* dsc data-pointer)
         (->BnnsTensorImpl dsc data-pointer master))
       (dragan-says-ex "The buffer has to be large enough for the descriptor."
                       {:desc-bytes (bytesize dsc) :buffer-bytes (bytesize data)}))))
  ([dsc buf]
   (tensor dsc buf false))
  ([dsc]
   (if dsc
     (let-release [buf (byte-pointer (bytesize dsc))]
       (tensor dsc
               (safe ((bnns-data-type-pointer (data-type* dsc)) buf))
               true))
     nil)))

(extend-tensor-info BnnsTensorImpl)
(extend-type BnnsTensorImpl
  Viewable
  (view [this]
    (tensor (.-dsc this) (.-data this) false)))

;; ===================== Filter ================================================

(def default-filter-params (filter-params*))

(defn safe-data [dsc]
  (if-not (null? (data* (desc dsc)))
    dsc
    (dragan-says-ex "Null is not allowed in this descriptor's data.")))

(defn apply-filter [^bnns$BNNSFilter filter in out]
  (filter-apply* (safe filter) (safe (data* (desc in))) (safe (data* (desc out)))))

(defn layer
  ([layer-params filter-params]
   (layer* (safe layer-params) (safe filter-params)))
  ([layer-params]
   (layer* (safe layer-params) (safe default-filter-params))))

;; ===================== Tensor Copy ===========================================

(defn copy
  ([src dst params]
   (if (compatible-desc? src dst)
     (copy* (safe (extract (desc src))) (safe (extract (desc dst))) (safe params))
     (dragan-says-ex "Copy can work only on compatible descriptors. Otherwise it crashes the JVM! Use activation-layer instead."))
   dst)
  ([src dst]
   (copy src dst default-filter-params)
   dst))

;; ===================== Activation ============================================

(defn activation
  ([function]
   (activation* (enc-keyword bnns-activation-function-enc function)))
  ([function ^double alpha]
   (activation* (enc-keyword bnns-activation-function-enc function) alpha))
  ([function ^double alpha ^double beta]
   (activation* (enc-keyword bnns-activation-function-enc function) alpha beta))
  ([function alpha beta scale offset shift]
   (activation* (enc-keyword bnns-activation-function-enc function) alpha beta
                scale offset shift)))

(defn activation-params
  ([^bnns$BNNSActivation activ in out]
   (let [in-desc (desc in)
         out-desc (desc out)]
     (if-let [res (activation-params* (safe activ)
                                      (safe (extract in-desc))
                                      (safe (extract out-desc)))]
       res
       (let [in-zero-stride (some zero? (pointer-seq (strides* in-desc)))
             out-zero-stride (some zero? (pointer-seq (strides* out-desc)))]
         (if-let [res (cond (and in-zero-stride out-zero-stride)
                            nil
                            in-zero-stride
                            (with-release [out-clone (clone* out-desc)]
                              (layout* out-clone (layout* in-desc))
                              (activation-params* (safe activ)
                                                  (safe (extract in-desc))
                                                  out-clone))
                            out-zero-stride
                            (with-release [in-clone (clone* in-desc)]
                              (layout* in-clone (layout* out-desc))
                              (activation-params* (safe activ)
                                                  in-clone
                                                  (safe (extract out-desc))))
                            :default nil)]
           res
           (dragan-says-ex "Activation parameters descriptors are not compatible!"))))))
  ([^bnns$BNNSActivation activ desc]
   (activation-params activ desc desc)))

;; ===================== Arithmetic ============================================

(defn arithmetic
  ([in-desc in-type out-desc out-type]
   (arithmetic* (safe (extract (desc in-desc)))
                (enc-keyword bnns-descriptor-type in-type)
                (safe (extract (desc out-desc)))
                (enc-keyword bnns-descriptor-type out-type)))
  ([in1-desc in1-type in2-desc in2-type out-desc out-type]
   (arithmetic* (safe (extract (desc in1-desc)))
                (enc-keyword bnns-descriptor-type in1-type)
                (safe (extract (desc in2-desc)))
                (enc-keyword bnns-descriptor-type in2-type)
                (safe (extract (desc out-desc)))
                (enc-keyword bnns-descriptor-type out-type)))
  ([in1-desc in1-type in2-desc in2-type in3-desc in3-type out-desc out-type]
   (arithmetic* (safe (extract (desc in1-desc)))
                (enc-keyword bnns-descriptor-type in1-type)
                (safe (extract (desc in2-desc)))
                (enc-keyword bnns-descriptor-type in2-type)
                (safe (extract (desc in3-desc)))
                (enc-keyword bnns-descriptor-type in3-type)
                (safe (extract (desc out-desc)))
                (enc-keyword bnns-descriptor-type out-type))))

(defn arithmetic-params
  ([function arithm ^bnns$BNNSActivation activation]
   (let [fun (enc-keyword bnns-arithmetic-function-enc function)
         fun-type (bnns-arithmetic-function-fields fun)]
     (arithmetic-params* fun
                         (if (instance? fun-type arithm)
                           (safe arithm)
                           (dragan-says-ex "fields has to be an instance of appropriate arithmetic kind."
                                           {:type (type arithm)
                                            :required fun-type}))
                         activation)))
  ([function arithm]
   (arithmetic-params function arithm nil)))

(defn apply-arithmetic  ;;TODO Support batches
  ([^bnns$BNNSFilter filter in out]
   (with-release [pp (pointer-pointer 1)]
     (put-entry! pp 0 (safe (data* (desc in))))
     (arithmetic-apply* (safe filter) pp (safe (data* (desc out))))))
  ([^bnns$BNNSFilter filter in1 in2 out]
   (with-release [pp (pointer-pointer 2)]
     (put-entry! pp 0 (safe (data* (desc in1))))
     (put-entry! pp 1 (safe (data* (desc in2))))
     (arithmetic-apply* (safe filter) pp (safe (data* (desc out))))))
  ([^bnns$BNNSFilter filter in1 in2 in3 out]
   (with-release [pp (pointer-pointer 2)]
     (put-entry! pp 0 (safe (data* (desc in1))))
     (put-entry! pp 1 (safe (data* (desc in2))))
     (put-entry! pp 2 (safe (data* (desc in3))))
     (arithmetic-apply* (safe filter) pp (safe (data* (desc out)))))))

;; ================= Fully Connected ===========================

(defn fully-connected-params [^bnns$BNNSActivation activ in-desc w-desc b-desc out-desc]
  (fully-connected-params* (safe activ)
                           (safe (extract (desc in-desc)))
                           (safe (extract (desc (safe-data w-desc))))
                           (safe (extract (desc (safe-data b-desc))))
                           (safe (extract (desc out-desc)))))

(extend-bnns-parameters bnns$BNNSLayerParametersFullyConnected)

;; ================= Convolution  ===========================

(defn convolution-params
  ([^bnns$BNNSActivation activ in-desc w-desc b-desc out-desc]
   (convolution-params* (safe activ)
                        (safe (extract (desc in-desc)))
                        (safe (extract (desc (safe-data w-desc))))
                        (safe (extract (desc (safe-data b-desc))))
                        (safe (extract (desc out-desc)))))
  ([^bnns$BNNSActivation activ in-desc w-desc b-desc out-desc strides dilates]
   (convolution-params activ in-desc w-desc b-desc out-desc strides dilates nil nil nil))
  ([^bnns$BNNSActivation activ in-desc w-desc b-desc out-desc
    strides dilates padding groups pad]
   (convolution-params* (safe activ)
                        (safe (extract (desc in-desc)))
                        (safe (extract (desc (safe-data w-desc))))
                        (safe (extract (desc (safe-data b-desc))))
                        (safe (extract (desc out-desc)))
                        (get strides 0 1) (get strides 1 1)
                        (get dilates 0 1) (get dilates 1 1)
                        (get padding 0 0) (get padding 1 0)
                        (or groups 1) (vec pad))))

(extend-bnns-parameters bnns$BNNSLayerParametersConvolution)
