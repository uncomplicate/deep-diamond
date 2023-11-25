;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.diamond.internal.cudnn.tensor
  (:require [uncomplicate.commons
             [core :refer [Releaseable release let-release with-release Info info Viewable view
                           bytesize]]
             [utils :refer [dragan-says-ex]]]
            [uncomplicate.fluokitten.protocols :refer [Comonad extract]]
            [uncomplicate.clojure-cpp :refer [pointer byte-pointer capacity position!]]
            [uncomplicate.clojurecuda.core :refer [memcpy-to-host! cuda-malloc]]
            [uncomplicate.neanderthal
             [core :refer [transfer! dim vctr copy! native]]
             [block :refer [entry-width buffer data-accessor create-data-source cast-prim]]
             [cuda :refer [factory-by-type]]]
            [uncomplicate.neanderthal.internal.api
             :refer [flow equals-block compatible? set-all MemoryContext
                     EngineProvider Container DataAccessorProvider FactoryProvider
                     native-factory zero raw host factory fits? DenseContainer view-vctr]]
            [uncomplicate.neanderthal.internal.cpp.cuda.structures
             :refer [cu-block-vector set-vector! get-vector!]]
            [uncomplicate.diamond.tensor
             :as diamond
             :refer [TensorDescriptor shape layout data-type TensorContainer Transfer
                     input output Revert ConnectorCreator connector view-tz batch-size]]
            [uncomplicate.diamond.internal
             [protocols
              :refer [TensorFactory DiamondFactoryProvider create-tensor create-tensor-desc
                      diamond-factory neanderthal-factory tensor-engine native-diamond-factory
                      Offset offset DiffTransfer diff-input diff-output BatchDescriptor
                      batch-index]]
             [utils :refer [check-contiguous default-strides]]]
            [uncomplicate.diamond.internal.dnnl
             [protocols :as dnnl]
             [core :as dnnl-core :refer [memory-desc]]
             [tensor :refer []]]
            [uncomplicate.diamond.internal.cudnn
             [core :refer [tensor-descriptor equal-desc? dims strides transform-tensor]]
             [protocols :refer [DescProvider desc handle]]
             [constants :refer [cudnn-format]]])
  (:import [clojure.lang IFn ExceptionInfo AFn]
           [uncomplicate.neanderthal.internal.api Block Changeable DataAccessor VectorSpace]
           uncomplicate.diamond.tensor.TensorDescriptorImpl
           uncomplicate.diamond.internal.dnnl.tensor.DnnlTensor
           [uncomplicate.diamond.internal.cudnn.impl CUTensorDescriptor CUFilterDescriptor]))

(def ^{:private true :const true} INEFFICIENT_OPERATION_MSG
  "This operation would be inefficient because it uses memory transfer.
  Please use transfer! to be reminded of that.")

(def ^{:private true :const true} DOES_NOT_FIT_MSG
  "Source and destination shapes have to fit.")

(defn ^:private not-available []
  (throw (UnsupportedOperationException. "Not available in CUDA. Please use a host instance.")))

(declare ->CUDnnTensor cudnn-transformer cudnn-tensor ->CUDnnShuffler cudnn-batcher cudnn-tensor-desc)

(defn cudnn-shape-padding [shape]
  (into shape (repeat (- 4 (count shape)) 1)))

(extend-type java.util.Collection
  DescProvider
  (desc [this]
    (cudnn-tensor-desc (shape this) :float nil)))

(extend-type java.lang.Number
  DescProvider
  (desc [this]
    (cudnn-tensor-desc [this] :float nil)))

(extend-type java.util.Map
  DescProvider
  (desc [this]
    (cudnn-tensor-desc (shape this) (or (:data-type this) :float) (layout this))))

(extend-type TensorDescriptorImpl
  DescProvider
  (desc [this]
    (cudnn-tensor-desc (.shape this) (or (.data-type this) :float) (layout this))))

(extend-type Object
  DescProvider
  (desc [this]
    (cudnn-tensor-desc (shape this) (or (data-type this) :float) (layout this))))

(extend-type CUTensorDescriptor
  TensorDescriptor
  (shape [this]
    (.dims this))
  (data-type [this]
    (.data-type this))
  (layout [this]
    (.layout this))
  ConnectorCreator
  (connector [in-desc out]
    (if (equal-desc? in-desc (input out))
      (view out)
      (let [out-tz (output out)]
        (if (equal-desc? in-desc out-tz)
          (view out-tz)
          (let [fact (diamond-factory out-tz)]
            (let-release [in-tz (cudnn-tensor fact (view in-desc) (batch-index out-tz))]
              (cudnn-transformer (handle fact) in-tz (view out-tz)))))))))

(defmethod print-method CUTensorDescriptor
  [^CUTensorDescriptor d ^java.io.Writer w]
  (.write w (pr-str {:shape (.dims d) :data-type (.data-type d) :layout (.layout d)})))

(defn cudnn-tensor-desc [shape dtype format]
  (let [format (or format (default-strides shape))]
    (if (or (cudnn-format format)
            (and (sequential? format) (<= 4 (count format)) (= (count format) (count shape))))
      (tensor-descriptor shape dtype format)
      (with-release [md (memory-desc shape dtype format)]
        (let [padding-4 (repeat (- 4 (count shape)) 1)]
          (tensor-descriptor (into shape padding-4) dtype (into (layout md) padding-4)))))))

(extend-type CUFilterDescriptor
  TensorDescriptor
  (shape [this]
    (.dims this))
  (data-type [this]
    (.data-type this))
  (layout [this]
    (.layout this))
  ConnectorCreator
  (connector [in-desc out]
    (if (equal-desc? in-desc (input out))
      (view out)
      (let [out-tz (output out)]
        (if (equal-desc? in-desc out-tz)
          (view out-tz)
          (let [fact (diamond-factory out-tz)]
            (let-release [in-tz (cudnn-tensor fact (view in-desc) (batch-index out-tz))]
              (cudnn-transformer (handle fact) in-tz (view out-tz)))))))))

(defmethod print-method CUFilterDescriptor
  [^CUFilterDescriptor d ^java.io.Writer w]
  (.write w (pr-str {:shape (.dims d) :data-type (.data-type d) :format (.layout d)})))

;; =================== Transformer ==============================================

(deftype CUDnnTransformer [cudnn-hdl in-tz out-tz in-da out-da]
  Releaseable
  (release [_]
    (release in-tz)
    (release out-tz))
  Object
  (hashCode [_]
    (-> (hash :transformer)
        (hash-combine (shape in-tz))
        (hash-combine (shape out-tz))))
  (equals [_ other]
    (and (instance? CUDnnTransformer other)
         (= (shape in-tz) (shape (.in-tz ^CUDnnTransformer other)))
         (= out-tz (.out-tz ^CUDnnTransformer other))))
  (toString [this]
    (str {:input in-tz
          :output out-tz}))
  Revert
  (revert [_]
    (cudnn-transformer cudnn-hdl (view out-tz) (view in-tz)))
  Viewable
  (view [_]
    (cudnn-transformer cudnn-hdl (view in-tz) (view out-tz)))
  Transfer
  (input [_]
    in-tz)
  (output [_]
    out-tz)
  IFn
  (invoke [this]
    (.invoke this cudnn-hdl)
    out-tz)
  (invoke [_ cudnn-hdl2]
    (transform-tensor cudnn-hdl2
                      (cast-prim in-da 1.0) in-tz (byte-pointer (buffer in-tz))
                      (cast-prim out-da 0.0) out-tz (byte-pointer (buffer out-tz)) )
    out-tz)
  (applyTo [this xs]
    (AFn/applyToHelper this xs))
  ConnectorCreator
  (connector [this out-desc]
    (if (equal-desc? out-tz out-desc)
      this
      (connector in-tz out-desc))))

(defn cudnn-transformer [cudnn-hdl in-tz out-tz]
  (->CUDnnTransformer cudnn-hdl in-tz out-tz (data-accessor in-tz) (data-accessor out-tz)))

;; =================== Batcher ==================================================

(deftype CUDnnBatcher [cudnn-hdl src-sub dst-sub src-tz dst-tz ^long mb-size
                       ^long src-cnt ^long src-stride-n ^long dst-cnt ^long dst-stride-n]
  Releaseable
  (release [_]
    (release src-tz)
    (release dst-tz)
    (release src-sub)
    (release dst-sub))
  Object
  (hashCode [_]
    (-> (hash :batcher)
        (hash-combine (shape src-sub))
        (hash-combine (shape dst-sub))))
  (equals [_ other]
    (and (instance? CUDnnBatcher other)
         (= (shape dst-tz) (shape (.dst-tz ^CUDnnBatcher other)))
         (= src-tz (.src-tz ^CUDnnBatcher other))))
  (toString [_]
    (str {:input src-tz
          :output dst-tz
          :mb-size mb-size}))
  Viewable
  (view [_]
    (cudnn-batcher cudnn-hdl (view src-tz) (view dst-tz) mb-size))
  Transfer
  (input [_]
    src-tz)
  (output [_]
    dst-tz)
  IFn
  (invoke [this]
    (.invoke this cudnn-hdl 0 0))
  (invoke [this src-n]
    (.invoke this cudnn-hdl src-n 0))
  (invoke [this src-n dst-n]
    (.invoke this cudnn-hdl src-n dst-n))
  (invoke [_ cudnn-hdl2 src-n dst-n]
    (let [src-n (long src-n)
          dst-n (long dst-n)]
      (if (and (<= 0 src-n (- src-cnt mb-size)) (<= 0 dst-n (- dst-cnt mb-size)))
        (do (offset src-sub (* src-stride-n src-n))
            (offset dst-sub (* dst-stride-n dst-n))
            (transform-tensor cudnn-hdl2
                             (cast-prim (data-accessor src-sub) 1.0) src-sub
                             (byte-pointer (buffer src-sub))
                             (cast-prim (data-accessor dst-sub) 0.0) dst-sub
                             (byte-pointer (buffer dst-sub))))
        (dragan-says-ex "Requested subtensor is outside of bounds."
                        {:src-index src-n :src-cnt src-cnt :dst-index dst-n :dst-cnt dst-cnt
                         :mb-size mb-size})))
    dst-tz)
  (applyTo [this xs]
    (AFn/applyToHelper this xs))
  ConnectorCreator
  (connector [this dst-desc]
    (if (equal-desc? dst-tz dst-desc)
      this
      (connector src-tz dst-desc))))

(defn cudnn-batcher [cudnn-hdl src-tz dst-tz mb-size]
  (let [mb-size (max 1 (long mb-size))]
    (let-release [src-sub (view (view-tz src-tz mb-size))
                  dst-sub (view (view-tz dst-tz mb-size))]
      (->CUDnnBatcher cudnn-hdl src-sub dst-sub
                      src-tz dst-tz mb-size
                      ((dims src-tz) (batch-index src-tz)) ((strides src-sub) (batch-index src-tz))
                      ((dims dst-tz) (batch-index dst-tz)) ((strides dst-sub) (batch-index dst-tz))))))

(deftype CUDnnShuffler [cudnn-hdl batcher batch-size mb-size]
  Releaseable
  (release [_]
    (release batcher))
  (hashCode [_]
    (hash-combine (hash :shuffler) (hash batcher)))
  (equals [_ other]
    (and (instance? CUDnnShuffler other)
         (= batch-size (.batch-size ^CUDnnShuffler other))
         (= mb-size (.mb-size ^CUDnnShuffler other))
         (= batcher (.batcher ^CUDnnShuffler other))))
  (toString [this]
    (str {:input (input this)
          :output (output this)
          :mb-size mb-size}))
  Viewable
  (view [_]
    (->CUDnnShuffler cudnn-hdl (view batcher) batch-size mb-size))
  Transfer
  (input [_]
    (input batcher))
  (output [_]
    (output batcher))
  IFn
  (invoke [_]
    (dotimes [i mb-size]
      (batcher cudnn-hdl (rand-int batch-size) i))
    (output batcher))
  (invoke [this cols]
    (.invoke this cudnn-hdl cols))
  (invoke [_ cudnn-hdl2 cols]
    (loop [src-n (first cols) cols (rest cols) dst-n 0]
      (when src-n
        (batcher cudnn-hdl src-n dst-n)
        (recur (first cols) (rest cols) (inc dst-n))))
    (output batcher))
  (applyTo [this xs]
    (AFn/applyToHelper this xs))
  ConnectorCreator
  (connector [this dst-desc]
    (if (equal-desc? (output batcher) dst-desc)
      this
      (connector batcher dst-desc))))

(defn cudnn-shuffler [cudnn-hdl src-tz dst-tz]
  (->CUDnnShuffler cudnn-hdl (cudnn-batcher cudnn-hdl src-tz dst-tz 1)
                   (batch-size src-tz) (batch-size dst-tz)))

;; ================================ Tensor ======================================

(deftype CUDnnTensor [diamond-fact eng vect-buf ^CUTensorDescriptor cu-desc ^long n-index]
  Object
  (hashCode [x]
    (-> (hash :CUDnnTensor) (hash-combine (hash cu-desc))))
  (equals [x y]
    (or (identical? x y)
        (and (instance? CUDnnTensor y) (equal-desc? cu-desc (desc y))
             (.isContiguous x) (.isContiguous ^CUDnnTensor y)
             (= vect-buf (.-vect-buf ^CUDnnTensor y)))))
  (toString [this]
    (pr-str {:shape (.dims cu-desc) :data-type (.data-type cu-desc)
             :layout (.layout cu-desc)}))
  Info
  (info [x]
    {:data-type (.data-type cu-desc)
     :class (class x)
     :device :cuda
     :shape (.dims cu-desc)
     :strides (.layout cu-desc)
     :master (info vect-buf :master)
     :engine eng})
  (info [x info-type]
    (case info-type
      :data-type (.data-type cu-desc)
      :class (class x)
      :device :cuda
      :shape (.dims cu-desc)
      :strides (.layout cu-desc)
      :master (info vect-buf :master)
      :engine eng
      nil))
  Releaseable
  (release [_]
    (release vect-buf)
    (release cu-desc)
    true)
  Comonad
  (extract [_]
    (extract vect-buf))
  EngineProvider
  (engine [_]
    eng)
  DiamondFactoryProvider
  (diamond-factory [_]
    diamond-fact)
  (native-diamond-factory [this]
    (native-diamond-factory diamond-fact))
  FactoryProvider
  (factory [_]
    (factory vect-buf))
  (native-factory [_]
    (native-factory vect-buf))
  DataAccessorProvider
  (data-accessor [_]
    (data-accessor vect-buf))
  Container
  (raw [x]
    (raw x diamond-fact))
  (raw [_ fact]
    (let [df (diamond-factory fact)]
      (create-tensor df (create-tensor-desc df cu-desc) n-index false)))
  (zero [x]
    (zero x diamond-fact))
  (zero [_ fact]
    (let [df (diamond-factory fact)]
      (create-tensor df (create-tensor-desc df cu-desc) n-index true)))
  (host [x]
    (let-release [res (raw x (native-diamond-factory diamond-fact))]
      (get-vector! vect-buf (view-vctr res))
      res))
  (native [x]
    (host x))
  MemoryContext
  (compatible? [_ y]
    (compatible? (factory vect-buf) (factory y)))
  (fits? [_ y]
    (= (.dims cu-desc) (cudnn-shape-padding (shape y))))
  (device [_]
    :cuda)
  VectorSpace
  (dim [_]
    (apply * (.dims cu-desc)))
  Block
  (buffer [_]
    (buffer vect-buf))
  (offset [_]
    0)
  (stride [_]
    (dragan-says-ex "Tensors do not have a single stride. You're doing something wrong."))
  (isContiguous [_]
    (= (dim vect-buf) (apply * (.dims cu-desc))))
  Changeable
  (setBoxed [x v]
    (set-all eng v x)
    x)
  (setBoxed [_ _ _]
    (dragan-says-ex INEFFICIENT_OPERATION_MSG))
  (alter [_ _]
    (dragan-says-ex INEFFICIENT_OPERATION_MSG))
  (alter [_ _ _]
    (dragan-says-ex INEFFICIENT_OPERATION_MSG))
  Revert
  (revert [this]
    this)
  Transfer
  (input [this]
    this)
  (output [this]
    this)
  DiffTransfer
  (diff-input [this]
    this)
  (diff-output [this]
    this)
  IFn
  (invoke [this]
    this)
  (applyTo [this xs]
    (AFn/applyToHelper this xs))
  DescProvider
  (desc [_]
    cu-desc)
  TensorDescriptor
  (shape [_]
    (.dims cu-desc))
  (data-type [_]
    (.data-type cu-desc))
  (layout [_]
    (.layout cu-desc))
  BatchDescriptor
  (batch-index [_]
    n-index)
  Viewable
  (view [_]
    (->CUDnnTensor diamond-fact eng (view vect-buf) (view cu-desc) n-index))
  DenseContainer
  (view-vctr [_]
    vect-buf)
  TensorContainer
  (view-tz [this]
    this)
  (view-tz [_ sub]
    (let-release [sub-desc (if (number? sub)
                             (cudnn-tensor-desc (assoc (dims cu-desc) n-index sub)
                                                (.data-type cu-desc)
                                                (.layout cu-desc))
                             (cudnn-tensor-desc (shape sub)
                                                (or (data-type sub) (.data-type cu-desc))
                                                (or (layout sub) (.layout cu-desc))))]
      (cudnn-tensor diamond-fact false (pointer (buffer vect-buf) 0) sub-desc n-index)))
  Offset
  (offset [this ofst]
    (if (<= 0 (long ofst) (capacity (buffer vect-buf)))
      (position! (buffer vect-buf) ofst)
      (dragan-says-ex "There isn't enough capacity in the underlying buffer for this offset."
                      {:requested ofst :available (capacity (buffer vect-buf))}))
    this)
  ConnectorCreator
  (connector [in-tz out-desc]
    (if (equal-desc? cu-desc out-desc)
      (view in-tz)
      (let-release [out-tz (cudnn-tensor diamond-fact out-desc (batch-index in-tz))]
        (cudnn-transformer (handle diamond-fact) (view in-tz) out-tz)))))

(defn cudnn-tensor
  ([diamond-fact master buf tdesc n-index]
   (let [tdesc (desc tdesc)
         neand-fact (neanderthal-factory diamond-fact (data-type tdesc))
         tz-cnt (apply * (dims tdesc))]
     (if (<= 0 (bytesize tdesc) (bytesize buf))
       (let-release [vect-buf (cu-block-vector neand-fact master buf tz-cnt 1)]
         (->CUDnnTensor diamond-fact
                        (tensor-engine diamond-fact (data-type tdesc))
                        vect-buf tdesc n-index))
       (throw (ex-info "Insufficient buffer size."
                       {:desc-size (bytesize tdesc) :buffer-size (bytesize buf)})))))
  ([diamond-fact master buf tdesc]
   (cudnn-tensor diamond-fact master buf tdesc 0))
  ([diamond-fact tdesc n-index]
   (let [tdesc (desc tdesc)]
     (let-release [buf (cuda-malloc (max 1 (bytesize tdesc)) (data-type tdesc))]
       (cudnn-tensor diamond-fact true buf tdesc n-index))))
  ([diamond-fact tdesc]
   (cudnn-tensor diamond-fact tdesc 0)))

(defmethod print-method CUDnnTensor
  [^CUDnnTensor x ^java.io.Writer w]
  (.write w (str x))
  (.write w "\n")
  (with-release [native-x (native (view-vctr x))]
    (print-method (doall (take *print-length* (seq native-x))) w)))

(defmethod transfer! [CUDnnTensor CUDnnTensor]
  [source destination]
  (copy! source destination))

(defmethod transfer! [DnnlTensor CUDnnTensor]
  [src dest]
  (check-contiguous src dest)
  (if (fits? dest src)
    (if (and (= (data-type src) (data-type dest))
             (= (cudnn-shape-padding (layout src)) (strides dest)))
      (set-vector! (view-vctr src) (view-vctr dest))
      (with-release [dnnl-mid (raw dest src)
                     dnnl-view (view-tz src (diamond/desc (cudnn-shape-padding (shape src))
                                                          (data-type src)
                                                          (cudnn-shape-padding (layout src))))]
        (transfer! dnnl-view dnnl-mid)
        (set-vector! (view-vctr dnnl-mid) (view-vctr dest))))
    (dragan-says-ex DOES_NOT_FIT_MSG
                    {:source (dnnl/desc src) :destination (desc dest)
                     :compatible? (compatible? src dest)}))
  dest)

(defmethod transfer! [CUDnnTensor DnnlTensor]
  [src dest]
  (check-contiguous src dest)
  (if (fits? src dest)
    (if (and (= (data-type src) (data-type dest))
             (= (strides src) (cudnn-shape-padding (layout dest))))
      (get-vector! (view-vctr src) (view-vctr dest))
      (with-release [dnnl-mid (raw src dest)
                     dnnl-view (view-tz dest (diamond/desc (cudnn-shape-padding (shape dest))
                                                           (data-type dest)
                                                           (cudnn-shape-padding (layout dest))))]
        (get-vector! (view-vctr src) (view-vctr dnnl-mid))
        (transfer! dnnl-mid dnnl-view)))
    (dragan-says-ex DOES_NOT_FIT_MSG
                    {:source (desc src) :destination (dnnl/desc dest)
                     :compatible? (compatible? src dest)}))
  dest)

(defmethod transfer! [CUDnnTensor Object]
  [source destination]
  (with-release [src (host source)]
    (transfer! src destination)))

(defmethod transfer! [Object CUDnnTensor]
  [source cuda]
  (check-contiguous cuda)
  (with-release [dest (raw cuda (native-diamond-factory cuda))]
    (transfer! source dest)
    (set-vector! (view-vctr dest) (view-vctr cuda))
    cuda))

(defmethod transfer! [Object CUDnnTransformer]
  [source destination]
  (transfer! source (input destination))
  destination)

(defmethod transfer! [CUDnnTransformer Object]
  [source destination]
  (transfer! (output source) destination))
