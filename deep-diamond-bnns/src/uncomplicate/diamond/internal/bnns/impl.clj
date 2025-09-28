;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.diamond.internal.bnns.impl
  (:require [uncomplicate.commons
             [core :refer [Releaseable release let-release with-release Info
                           info Viewable view Bytes bytesize Entries sizeof* bytesize*
                           sizeof size]]
             [utils :refer [dragan-says-ex with-check]]]
            [uncomplicate.fluokitten
             [protocols :refer [Comonad extract]]
             [core :refer [foldmap fold]]]
            [uncomplicate.clojure-cpp
             :refer [null? get-entry safe2 ptr* pointer byte-pointer size-t-pointer
                     address PointerCreator pointer-seq]]
            [uncomplicate.diamond.internal.utils
             :refer [extend-pointer default-strides]]
            [uncomplicate.diamond.internal.bnns
             [protocols :refer :all]
             [constants :refer :all]])
  (:import [org.bytedeco.javacpp PointerPointer LongPointer FloatPointer Pointer]
           [uncomplicate.javacpp.accelerate.global bnns bnns$BNNSTensor bnns$BNNSNDArrayDescriptor
            bnns$BNNSActivation bnns$BNNSLayerParametersActivation bnns$BNNSFilterParameters
            bnns$BNNSFilter bnns$BNNSArithmeticUnary bnns$BNNSArithmeticBinary
            bnns$BNNSArithmeticTernary bnns$BNNSLayerParametersArithmetic
            bnns$BNNSLayerParametersFullyConnected bnns$BNNSLayerParametersConvolution]))

;; ===================== Tensor and NDArray descriptors ===========================================

(declare ->BnnsTensorDescriptorImpl ->BnnsNdArrayDescriptorImpl ->BnnsTensorImpl
         tensor-descriptor* ndarray-descriptor*)

(defn bnns-error
  ([^long err-code details]
   (ex-info (format "BNNS error: %d." err-code)
            {:error err-code :type :bnns-error :details details}))
  ([^long err-code]
   (bnns-error err-code nil)))

(defn equal-desc-properties? [d1 d2]
  (and (= (pointer-seq (dims* d1)) (pointer-seq (dims* d2)))
       (= (data-type* d1) (data-type* d2))
       (= (pointer-seq (strides* d1)) (pointer-seq (strides* d2)))))

(defmacro extend-bnns-pointer [t]
  `(extend-type ~t
     Releaseable
     (release [this#]
       (locking this#
         (when-not (null? this#)
           (do (.deallocate this#)
               (.setNull this#))))
       true)))

(defn major? [data-layout]
  (if (#{0x28001 :ba 0x38001 :cba 0x48001 :dcba 0x58001 :edcba 0x68001 :fedcba
         0x78001 :gfedcba 0x88001 :hgfedcba}
       data-layout)
    false
    true))

(defn tensor-shape-size-major ^long [shape strides]
  (inc (long (foldmap + 0
                      (fn [^long x ^long y]
                        (* (dec x) y))
                      shape strides))))

(defn tensor-shape-size-minor ^long [shape strides]
  (let [n (size shape)]
    (loop [i 0 j (dec n) acc 0]
      (if (< i n)
        (recur (inc i) (dec j) (+ acc
                                  (* (dec (long (get-entry shape i)))
                                     (long (get-entry strides j)))))
        acc))))

(defn tensor-shape-size ^long [major shape strides]
  (if major
    (tensor-shape-size-major shape strides)
    (tensor-shape-size-minor shape strides)))

(defn nda-shape-size ^long [major shape strides]
  (if (and strides (< 0 (long (fold * strides))))
    (tensor-shape-size major shape strides)
    (long (fold * shape))))

(defmacro extend-tensor-descriptor [t]
  `(extend-type ~t
     Releaseable
     (release [this#]
       (let [td# (.-td this#)]
         (locking td#
           (when (and (.-master this#) (not (null? td#)))
             (do (.deallocate (ptr* td#))
                 (.setNull (ptr* td#))))))
       true)
     PointerCreator
     (pointer* [this#]
       (if-not (null? (.-td this#)) (.-td this#) nil))
     Comonad
     (extract [this#]
       (if-not (null? (.-td this#)) (.-td this#) nil))
     DescProvider
     (desc [this#]
       this#)
     Entries
     (sizeof* [this#]
       (bnns-data-type-size (data-type* this#)))
     (size* [this#]
       (quot (long (bytesize* this#)) (long (sizeof* this#))))))

(deftype BnnsTensorDescriptorImpl [^bnns$BNNSTensor td ^long rank master]
  Object
  (hashCode [this#]
    (hash td))
  (equals [this other]
    (or (identical? this other)
        (and (instance? BnnsTensorDescriptorImpl other)
             (or (= td (extract other))
                 (equal-desc-properties? this other)))))
  (toString [this]
    (format "#BnnsTensorDescriptorImpl[0x%s, type: %s, master: %s]"
            (address td) (dec-data-type (data-type* this)) master))
  Descriptor
  (strides* [this]
    (when-not (null? td)
      (.capacity (.stride ^bnns$BNNSTensor td) rank)))
  (major* [this]
    true)
  (data-type* [this]
    (when-not (null? td)
      (.data_type ^bnns$BNNSTensor td)))
  (dims* [this]
    (when-not (null? td)
      (.capacity (.shape ^bnns$BNNSTensor td) rank)))
  (rank* [_]
    rank)
  (data* [this]
    (when-not (null? td)
      (.data ^bnns$BNNSTensor td)))
  (data* [this p]
    (when-not (null? td)
      (.data ^bnns$BNNSTensor td (byte-pointer (extract p)))))
  (clone* [this]
    (->BnnsTensorDescriptorImpl
     (tensor-descriptor* (dims* this) (data-type* this) (strides* this))
     rank true))
  Bytes
  (bytesize* [_]
    (if (null? td)
      nil
      (bnns/BNNSTensorGetAllocationSize td)))
  Viewable
  (view [this]
    (->BnnsTensorDescriptorImpl td rank false)))

(extend-tensor-descriptor BnnsTensorDescriptorImpl)

(extend-type nil
  Descriptor
  (data* [_] nil)
  (data* [_ _] nil)
  (strides* [_] nil)
  (layout* [_] nil)
  (major* [_] true)
  (data-type* [_] nil)
  (dims* [_] nil)
  (rank* [_] 0))

(deftype BnnsNdArrayDescriptorImpl [^bnns$BNNSNDArrayDescriptor td rank master]
  Object
  (hashCode [this]
    (hash td))
  (equals [this other]
    (or  (identical? this other)
         (and (instance? BnnsNdArrayDescriptorImpl other)
              (or (= td (extract other))
                  (and (= (.layout td) (.layout ^bnns$BNNSNDArrayDescriptor (extract other)))
                       (equal-desc-properties? this other))))))
  (toString [this]
    (format "#BnnsNdArrayDescriptorImpl[0x%s,type: %s, layout: %s, master: %s]"
            (address td) (dec-data-type (data-type* this))
            (dec-data-layout (layout* this)) master))
  Descriptor
  (strides* [_]
    (when-not (null? td)
      (.capacity (.stride ^bnns$BNNSNDArrayDescriptor td) rank)))
  (layout* [_]
    (when-not (null? td)
      (.layout ^bnns$BNNSNDArrayDescriptor td)))
  (layout* [this layout]
    (when-not (null? td)
      (.layout ^bnns$BNNSNDArrayDescriptor td (long layout*))))
  (major* [this]
    (or (null? td)
        (major? (.layout ^bnns$BNNSNDArrayDescriptor td))))
  (data-type* [this]
    (when-not (null? td)
      (.data_type ^bnns$BNNSNDArrayDescriptor td)))
  (dims* [this]
    (when-not (null? td)
      (.capacity (.size ^bnns$BNNSNDArrayDescriptor td) rank)))
  (rank* [_]
    rank)
  (data* [this]
    (when-not (null? td)
      (.data ^bnns$BNNSNDArrayDescriptor (extract this))))
  (data* [this p]
    (when-not (null? td)
      (.data ^bnns$BNNSNDArrayDescriptor td (byte-pointer (extract p)))))
  (clone* [this];;TODO remove in favor of view
    (->BnnsNdArrayDescriptorImpl
     (ndarray-descriptor* (dims* this) (data-type* this) (layout* this) (strides* this))
     rank true))
  Bytes
  (bytesize* [this]
    (if (null? td)
      0
      (* (long (bnns-data-type-size (data-type* this)))
         (nda-shape-size (major* this) (dims* this) (strides* this)))))
  Viewable
  (view [this]
    (clone* this)))

(extend-tensor-descriptor BnnsNdArrayDescriptorImpl)

(extend-type bnns$BNNSNDArrayDescriptor
  Releaseable
  (release [_]
    (dragan-says-ex "You should never directly release bnns$BNNSNDArrayDescriptor. Please use NdarrayDescImpl!")))

(extend-type bnns$BNNSTensor
  Releaseable
  (release [_]
    (dragan-says-ex "You should never directly release bnns$BNNSTensor. Please use BnnsTensorImpl!")))

(defn tensor-descriptor*
  ([shape ^long data-type strides]
   (let [rank (size shape)
         td (bnns$BNNSTensor.)]
     (if (= rank (size strides))
       (try
         (.data-type td data-type)
         (.rank td rank)
         (dotimes [i bnns/BNNS_MAX_TENSOR_DIMENSION]
           (if (< i rank)
             (.shape td i (get-entry shape i))
             (.shape td i 0)))
         (if (null? strides)
           (dotimes [i bnns/BNNS_MAX_TENSOR_DIMENSION]
             (.stride td i 0))
           (dotimes [i bnns/BNNS_MAX_TENSOR_DIMENSION]
             (if (< i rank)
               (.stride td i (get-entry strides i))
               (.stride td i 0))))
         td
         (catch Exception e
           (.deallocate td)
           (throw e)))
       (dragan-says-ex "Shape and strides must have the same rank!"
                       {:shape (info shape) :strides (info strides)}))))
  ([dsc]
   (tensor-descriptor* (dims* dsc) (data-type* dsc) (strides* dsc))))

(defn ndarray-descriptor*
  ([shape ^long data-type ^long layout strides]
   (let [rank (size shape)
         nda (bnns$BNNSNDArrayDescriptor.)]
     (if (= rank (size strides))
       (try
         (.data-type nda data-type)
         (.layout nda layout)
         (dotimes [i bnns/BNNS_MAX_TENSOR_DIMENSION]
           (if (< i rank)
             (.size nda i (get-entry shape i))
             (.size nda i 0)))
         (if (null? strides)
           (dotimes [i bnns/BNNS_MAX_TENSOR_DIMENSION]
             (.stride nda i 0))
           (dotimes [i bnns/BNNS_MAX_TENSOR_DIMENSION]
             (if (< i rank)
               (.stride nda i (get-entry strides i))
               (.stride nda i 0))))
         nda
         (catch Exception e
           (.deallocate nda)
           (throw e)))
       (dragan-says-ex "Shape and strides must have the same rank!"
                       {:shape (info shape) :strides (info strides)}))))
  ([dsc]
   (ndarray-descriptor* (dims* dsc) (data-type* dsc) (layout* dsc) (strides* dsc))))

(deftype BnnsTensorImpl [dsc data master]
  Object
  (hashCode [_]
    (hash dsc))
  (equals [this other]
    (or (identical? this other)
        (and (instance? BnnsTensorImpl other)
             (= dsc (.-dsc ^BnnsTensorImpl other))
             (= data (.-data ^BnnsTensorImpl other)))))
  (toString [this]
    (format "#BnnsTensorImpl[0x%s, master: %s]" (address (extract dsc)) master))
  Releaseable
  (release [this]
    (when master
      (release data))
    (release dsc)
    true)
  DescProvider
  (desc [_]
    dsc)
  Comonad
  (extract [_]
    (if-not (null? data) (extract data) nil))
  PointerCreator
  (pointer* [_]
    (if (and (extract dsc) (extract data)) data nil))
  (pointer* [_ i]
    (if (and (extract dsc) (extract data)) (pointer data i) nil))
  Descriptor
  (strides* [_]
    (strides* dsc))
  (data-type* [_]
    (data-type* dsc))
  (dims* [_]
    (dims* dsc))
  (data* [_]
    (data* dsc))
  (data* [this p]
    (data* dsc p)
    this)
  (rank* [_]
    (rank* dsc))
  Bytes
  (bytesize* [_]
    (bytesize* dsc))
  Entries
  (sizeof* [_]
    (sizeof* data))
  (size* [this]
    (quot (bytesize this) (sizeof data))))

;; ===================== Filter ================================================

(extend-bnns-pointer bnns$BNNSFilterParameters)

(defn filter-params*
  ([^long flags ^long nthreads]
   (let-release [res (bnns$BNNSFilterParameters.)]
     (.n_threads res nthreads)
     (.flags res flags)
     res))
  ([^long nthreads]
   (filter-params* bnns/BNNSFlagsUseClientPtr))
  ([]
   (filter-params* 0 bnns/BNNSFlagsUseClientPtr)))

(defn filter-apply* [^bnns$BNNSFilter filter ^Pointer in ^Pointer out]
  (with-check bnns-error
    (bnns/BNNSFilterApply filter in out)
    filter))

(defn filter-apply-forward* [^bnns$BNNSFilter filter ^long n ^Pointer in ^Pointer out]
  (with-check bnns-error
    (bnns/BNNSFilterApplyBatch filter n in (quot (size in) n) out (quot (size out) n))
    filter))

(defn filter-apply-backward*
  ([^bnns$BNNSFilter filter ^bnns$BNNSNDArrayDescriptor in-delta
    ^Pointer out ^bnns$BNNSNDArrayDescriptor out-delta]
   (with-check bnns-error
     (bnns/BNNSFilterApplyBackwardBatch
      filter 1 nil 0 in-delta (size in-delta) out (size out)
      out-delta (size out-delta) nil nil)
     filter))
  ([^bnns$BNNSFilter filter
    ^Pointer in ^bnns$BNNSNDArrayDescriptor in-delta
    ^Pointer out ^bnns$BNNSNDArrayDescriptor out-delta]
   (with-check bnns-error
     (bnns/BNNSFilterApplyBackwardBatch
      filter 1 in (size in) in-delta (size in-delta)
      out (size out) out-delta (size out-delta) nil nil)
     filter)))

;;TODO delete. Crashes when n>1 and not used anyway since BNNS fully-connected is a mess
(defn filter-apply-backward-n*
  ([^bnns$BNNSFilter filter n ^bnns$BNNSNDArrayDescriptor in-delta
    ^Pointer out ^bnns$BNNSNDArrayDescriptor out-delta]
   (with-check bnns-error
     (bnns/BNNSFilterApplyBackwardBatch
      filter (long n) nil 0 in-delta (quot (size in-delta) n) out (quot (size out) n)
      out-delta (quot (size out-delta) n) nil nil)
     filter))
  ([^bnns$BNNSFilter filter n
    ^Pointer in ^bnns$BNNSNDArrayDescriptor in-delta
    ^Pointer out ^bnns$BNNSNDArrayDescriptor out-delta]
   (with-check bnns-error
     (bnns/BNNSFilterApplyBackwardBatch
      filter (long n) in (quot (size in) n) in-delta (quot (size in-delta) n)
      out (quot (size out) n) out-delta (quot (size out-delta) n) nil nil)
     filter))
  ([^bnns$BNNSFilter filter n
    ^Pointer in ^bnns$BNNSNDArrayDescriptor in-delta
    ^Pointer out ^bnns$BNNSNDArrayDescriptor out-delta
    ^bnns$BNNSNDArrayDescriptor weights-delta
    ^bnns$BNNSNDArrayDescriptor bias-delta]
   (with-check bnns-error
     (bnns/BNNSFilterApplyBackwardBatch
      filter (long n) in (quot (size in) n) in-delta (quot (size in-delta) n)
      out (quot (size out) n) out-delta (quot (size out-delta) n)
      weights-delta bias-delta)
     filter)))

(extend-type bnns$BNNSFilter
  Comonad
  (extract [this]
    (if (null? this) nil this))
  Releaseable
  (release [this#]
    (locking this#
      (when-not (null? this#)
        (bnns/BNNSFilterDestroy this#)
        (.deallocate this#)
        (.setNull this#)))
    true))

;; ===================== Tensor Copy ===========================================

(defn copy* [^bnns$BNNSNDArrayDescriptor src
             ^bnns$BNNSNDArrayDescriptor dst
             ^bnns$BNNSFilterParameters params]
  (with-check bnns-error
    (bnns/BNNSCopy dst src params)
    dst))

;; ===================== Activation ============================================

(extend-bnns-pointer bnns$BNNSActivation)

(defn activation*
  ([^long function]
   (activation* function 0.0 0.0))
  ([^long function ^double alpha]
   (activation* function alpha 0.0))
  ([^long function ^double alpha ^double beta]
   (let-release [res (bnns$BNNSActivation.)]
     (.function res function)
     (.alpha res alpha)
     (.beta res beta)
     res))
  ([function alpha beta iscale ioffset ishift]
   (let-release [res (activation* function alpha beta)]
     (.iscale res iscale)
     (.ioffset res ioffset)
     (.ishift res ishift)
     res)))

(extend-bnns-pointer bnns$BNNSLayerParametersActivation)

(extend-type bnns$BNNSLayerParametersActivation
  LayerCreator
  (layer* [layer-params filter-params]
    (bnns/BNNSFilterCreateLayerActivation layer-params filter-params)))

(defn activation-params* [^bnns$BNNSActivation activ
                          ^bnns$BNNSNDArrayDescriptor in-desc
                          ^bnns$BNNSNDArrayDescriptor out-desc]
  (let-release [res (bnns$BNNSLayerParametersActivation.)]
    (.activation res activ)
    (.i_desc res in-desc)
    (.o_desc res out-desc)
    res))

;; ===================== Arithmetic ============================================

(extend-bnns-pointer bnns$BNNSArithmeticUnary)
(extend-bnns-pointer bnns$BNNSArithmeticBinary)
(extend-bnns-pointer bnns$BNNSArithmeticTernary)

(defn arithmetic*
  ([^bnns$BNNSNDArrayDescriptor in-desc ^long in-type
    ^bnns$BNNSNDArrayDescriptor out-desc ^long out-type]
   (let-release [res (bnns$BNNSArithmeticUnary.)]
     (.in res in-desc)
     (.in_type res in-type)
     (.out res out-desc)
     (.out_type res out-type)
     res))
  ([^bnns$BNNSNDArrayDescriptor in1-desc in1-type
    ^bnns$BNNSNDArrayDescriptor in2-desc in2-type
    ^bnns$BNNSNDArrayDescriptor out-desc out-type]
   (let-release [res (bnns$BNNSArithmeticBinary.)]
     (.in1 res in1-desc)
     (.in1_type res (int in1-type))
     (.in2 res in2-desc)
     (.in2_type res (int in2-type))
     (.out res out-desc)
     (.out_type res (int out-type))
     res))
  ([^bnns$BNNSNDArrayDescriptor in1-desc in1-type
    ^bnns$BNNSNDArrayDescriptor in2-desc in2-type
    ^bnns$BNNSNDArrayDescriptor in3-desc in3-type
    ^bnns$BNNSNDArrayDescriptor out-desc out-type]
   (let-release [res (bnns$BNNSArithmeticTernary.)]
     (.in1 res in1-desc)
     (.in1_type res (int in1-type))
     (.in2 res in2-desc)
     (.in2_type res (int in2-type))
     (.in3 res in3-desc)
     (.in3_type res (int in3-type))
     (.out res out-desc)
     (.out_type res (int out-type))
     res)))

(extend-bnns-pointer bnns$BNNSLayerParametersArithmetic)

(extend-type bnns$BNNSLayerParametersArithmetic
  LayerCreator
  (layer* [layer-params filter-params]
    (bnns/BNNSFilterCreateLayerArithmetic layer-params filter-params)))

(defn arithmetic-params*
  ([^long function
    ^Pointer fields
    ^bnns$BNNSActivation activation]
   (let-release [res (arithmetic-params* function fields)]
     (when activation (.activation res activation))
     res))
  ([^long function
    ^Pointer fields]
   (let-release [res (bnns$BNNSLayerParametersArithmetic.)]
     (.arithmetic_function res function)
     (.arithmetic_function_fields res fields)
     res)))

(defn arithmetic-apply*
  ([^bnns$BNNSFilter filter ^PointerPointer in ^Pointer out]
   (with-check bnns-error
     (bnns/BNNSArithmeticFilterApplyBatch filter 1 (size in) in (size-t-pointer [(size in)]) out (size out))
     filter)))

;; ================= Fully Connected ===========================

(extend-bnns-pointer bnns$BNNSLayerParametersFullyConnected)

(extend-type bnns$BNNSLayerParametersFullyConnected
  LayerCreator
  (layer* [layer-params filter-params]
    (bnns/BNNSFilterCreateLayerFullyConnected layer-params filter-params)))

(defn fully-connected-params* [^bnns$BNNSActivation activ
                               ^bnns$BNNSNDArrayDescriptor in-desc
                               ^bnns$BNNSNDArrayDescriptor w-desc
                               ^bnns$BNNSNDArrayDescriptor b-desc
                               ^bnns$BNNSNDArrayDescriptor out-desc]
  (let-release [res (bnns$BNNSLayerParametersFullyConnected.)]
    (.activation res activ)
    (.i_desc res in-desc)
    (.w_desc res w-desc)
    (.bias res b-desc)
    (.o_desc res out-desc)
    res))

;; ================= Convolution ===========================

(extend-bnns-pointer bnns$BNNSLayerParametersConvolution)

(extend-type bnns$BNNSLayerParametersConvolution
  LayerCreator
  (layer* [layer-params filter-params]
    (bnns/BNNSFilterCreateLayerConvolution layer-params filter-params)))

(defn convolution-params*
  ([^bnns$BNNSActivation activ
    ^bnns$BNNSNDArrayDescriptor in-desc
    ^bnns$BNNSNDArrayDescriptor w-desc
    ^bnns$BNNSNDArrayDescriptor b-desc
    ^bnns$BNNSNDArrayDescriptor out-desc]
   (let-release [res (bnns$BNNSLayerParametersConvolution.)]
     (.activation res activ)
     (.i_desc res in-desc)
     (.w_desc res w-desc)
     (.bias res b-desc)
     (.o_desc res out-desc)
     res))
  ([activ in-desc w-desc b-desc out-desc
    x-stride y-stride x-dilation y-dilation
    x-padding y-padding]
   (let-release [res (convolution-params* activ in-desc w-desc b-desc out-desc)]
     (.x_stride res x-stride)
     (.y_stride res y-stride)
     (.x_dilation_stride res x-dilation)
     (.y_dilation_stride res y-dilation)
     (.x_padding res x-padding)
     (.y_padding res y-padding)
     res))
  ([activ in-desc w-desc b-desc out-desc
    x-stride y-stride x-dilation y-dilation
    x-padding y-padding groups pad]
   (let-release [res (convolution-params* activ in-desc w-desc b-desc out-desc
                                          x-stride y-stride x-dilation y-dilation
                                          x-padding y-padding)]
     (.groups res groups)
     (dotimes [i 4]
       (.pad res i (get pad i 0)))
     res)))
