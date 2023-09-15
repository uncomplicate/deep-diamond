;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.diamond.internal.dnnl.tensor
  (:require [uncomplicate.commons
             [core :refer [Releaseable release let-release with-release Info info Viewable view
                           bytesize Wrapper extract size]]
             [utils :refer [dragan-says-ex]]]
            [uncomplicate.fluokitten
             [protocols :refer [Magma Monoid Applicative Functor]]
             [core :refer [fmap foldmap]]]
            [uncomplicate.clojure-cpp :refer [pointer]]
            [uncomplicate.neanderthal
             [core :refer [transfer! dim copy!]]
             [block :refer [entry-width data-accessor buffer count-entries contiguous?]]]
            [uncomplicate.neanderthal.internal.api
             :refer [flow FactoryProvider EngineProvider DataAccessorProvider
                     Container raw copy MemoryContext set-all compatible? factory native-factory
                     create-vector DenseContainer view-vctr]]
            [uncomplicate.neanderthal.internal.cpp.structures :refer [real-block-vector]]
            [uncomplicate.diamond.tensor
             :refer [TensorDescriptor shape layout TensorContainer Transfer input output
                     Revert ConnectorCreator connector view-tz transformer batch-size]
             :as tz]
            [uncomplicate.diamond.internal
             [protocols
              :refer [TensorFactory DiamondFactoryProvider diamond-factory create-tensor
                      neanderthal-factory tensor-engine native-diamond-factory Offset offset
                      DiffTransfer diff-input diff-output create-tensor-desc parameters
                      DescriptorProvider BatchDescriptor batch-index]]
             [utils :refer [check-contiguous]]]
            [uncomplicate.diamond.internal.dnnl
             [core :refer [memory-desc dims data-type memory strides submemory-desc
                           equal-desc? execute! reorder primitive fwd-args offset! ndims]
              :as dnnl-core]
             [constants :refer [entry-bytes]]
             [protocols :refer [DescProvider desc dnnl-engine]]])
  (:import org.bytedeco.javacpp.Pointer
           [clojure.lang Seqable IFn AFn]
           [uncomplicate.neanderthal.internal.api Block VectorSpace Changeable]
           org.bytedeco.dnnl.dnnl_memory_desc
           uncomplicate.diamond.tensor.TensorDescriptorImpl))

(declare ->DnnlTensor dnnl-transformer dnnl-tensor dnnl-batcher ->DnnlShuffler)

(extend-type java.util.Collection
  DescProvider
  (desc [this]
    (memory-desc this :float :any)))

(extend-type java.lang.Number
  DescProvider
  (desc [this]
    (memory-desc [this] :float :x)))

(extend-type java.util.Map
  DescProvider
  (desc [this]
    (memory-desc (:shape this) (or (:data-type this) :float) (or (layout this) :any))))

(extend-type TensorDescriptorImpl
  DescProvider
  (desc [this]
    (memory-desc (.shape this) (or (.data-type this) :float) (or (layout this) :any))))

(extend-type Object
  DescProvider
  (desc [this]
    (memory-desc (shape this) (or (tz/data-type this) :float) (or (layout this) :any))))

(extend-type dnnl_memory_desc
  TensorDescriptor
  (shape [this]
    (dims this))
  (tz/data-type [this]
    (data-type this))
  (layout [this]
    (strides this))
  ConnectorCreator
  (connector [in-desc out]
    (if (equal-desc? in-desc (input out))
      (view out)
      (let [out-tz (output out)]
        (if (equal-desc? in-desc out-tz)
          (view out-tz)
          (let [fact (diamond-factory out-tz)]
            (let-release [in-tz (dnnl-tensor fact in-desc (batch-index out-tz))]
              (dnnl-transformer (dnnl-engine fact) (flow fact) in-tz (view out-tz)))))))))

(defmethod print-method dnnl_memory_desc
  [^dnnl_memory_desc d ^java.io.Writer w]
  (.write w (pr-str {:shape (dims d) :data-type (data-type d) :layout (strides d)})))

;; =================== Transformer ==============================================

(deftype DnnlTransformer [eng strm reorder reorder-args in-tz out-tz]
  Releaseable
  (release [_]
    (release in-tz)
    (release out-tz)
    (release reorder))
  Object
  (hashCode [_]
    (-> (hash :transformer)
        (hash-combine (shape in-tz))
        (hash-combine (shape out-tz))))
  (equals [_ other]
    (and (instance? DnnlTransformer other)
         (= (shape in-tz) (shape (.in-tz ^DnnlTransformer other)))
         (= out-tz (.out-tz ^DnnlTransformer other))))
  (toString [this]
    (str {:input in-tz
          :output out-tz}))
  Revert
  (revert [_]
    (dnnl-transformer eng strm (view out-tz) (view in-tz)))
  Viewable
  (view [_]
    (dnnl-transformer eng strm (view in-tz) (view out-tz)))
  Transfer
  (input [_]
    in-tz)
  (output [_]
    out-tz)
  IFn
  (invoke [_]
    (execute! strm reorder reorder-args)
    out-tz)
  (invoke [_ strm2]
    (execute! strm2 reorder reorder-args)
    out-tz)
  (applyTo [this xs]
    (AFn/applyToHelper this xs))
  ConnectorCreator
  (connector [this out-desc]
    (if (equal-desc? out-tz out-desc)
      this
      (connector in-tz out-desc))))

(defn dnnl-transformer [eng strm in-tz out-tz]
  (with-release [reorder-pd (reorder eng in-tz out-tz)]
    (let-release [reorder-prim (primitive reorder-pd)]
      (->DnnlTransformer eng strm reorder-prim (fwd-args in-tz out-tz)
                         in-tz out-tz))))

;; =================== Batcher ==================================================

(deftype DnnlBatcher [eng strm reorder reorder-args
                      src-sub dst-sub src-tz dst-tz ^long mb-size
                      ^long src-cnt ^long src-stride-n
                      ^long dst-cnt ^long dst-stride-n]
  Releaseable
  (release [_]
    (release src-tz)
    (release dst-tz)
    (release src-sub)
    (release dst-sub)
    (release reorder))
  Object
  (hashCode [_]
    (-> (hash :batcher)
        (hash-combine (shape src-sub))
        (hash-combine (shape dst-sub))))
  (equals [_ other]
    (and (instance? DnnlBatcher other)
         (= (shape src-tz) (shape (.src-tz ^DnnlBatcher other)))
         (= (shape dst-tz) (shape (.dst-tz ^DnnlBatcher other)))
         (= src-tz (.src-tz ^DnnlBatcher other))))
  (toString [_]
    (str {:input src-tz
          :output dst-tz
          :mb-size mb-size}))
  Viewable
  (view [_]
    (dnnl-batcher eng strm (view src-tz) (view dst-tz) mb-size))
  Transfer
  (input [_]
    src-tz)
  (output [_]
    dst-tz)
  IFn
  (invoke [this]
    (.invoke this strm 0 0))
  (invoke [this src-n]
    (.invoke this strm src-n 0))
  (invoke [this src-n dst-n]
    (.invoke this strm src-n dst-n))
  (invoke [_ strm2 src-n dst-n]
    (let [src-n (long src-n)
          dst-n (long dst-n)]
      #dbg (if (and (<= 0 src-n (- src-cnt mb-size)) (<= 0 dst-n (- dst-cnt mb-size)))
        (do
          (offset src-sub (* src-stride-n src-n))
          (offset dst-sub (* dst-stride-n dst-n))
          (execute! strm2 reorder reorder-args))
        (dragan-says-ex "Requested subtensor is outside of bounds."
                        {:src-index src-n :src-cnt src-cnt
                         :dst-index dst-n :dst-cnt dst-cnt
                         :mb-size mb-size})))
    dst-tz)
  (applyTo [this xs]
    (AFn/applyToHelper this xs))
  ConnectorCreator
  (connector [this dst-desc]
    (if (equal-desc? dst-tz dst-desc)
      this
      (connector src-tz dst-desc))))

(defn dnnl-batcher [eng strm src-tz dst-tz mb-size]
  (let [mb-size (max 1 (long mb-size))]
    (let-release [src-sub (view-tz src-tz mb-size)
                  dst-sub (view-tz dst-tz mb-size)]
      (with-release [reorder-pd (reorder eng src-sub dst-sub)]
        (let-release [reorder-prim (primitive reorder-pd)]
          (->DnnlBatcher eng strm reorder-prim
                         (fwd-args src-sub dst-sub)
                         src-sub dst-sub
                         src-tz dst-tz
                         mb-size
                         ((dims src-tz) (batch-index src-tz)) ((strides src-sub) (batch-index src-tz))
                         ((dims dst-tz) (batch-index dst-tz)) ((strides dst-sub) (batch-index dst-tz))))))))

(deftype DnnlShuffler [strm batcher batch-size mb-size]
  Releaseable
  (release [_]
    (release batcher))
  Object
  (hashCode [_]
    (hash-combine (hash :shuffler) (hash batcher)))
  (equals [_ other]
    (and (instance? DnnlShuffler other)
         (= batch-size (.batch-size ^DnnlShuffler other))
         (= mb-size (.mb-size ^DnnlShuffler other))
         (= batcher (.batcher ^DnnlShuffler other))))
  (toString [this]
    (str {:input (input this)
          :output (output this)
          :mb-size mb-size}))
  Viewable
  (view [_]
    (->DnnlShuffler strm (view batcher) batch-size mb-size))
  Transfer
  (input [_]
    (input batcher))
  (output [_]
    (output batcher))
  IFn
  (invoke [_]
    (dotimes [i mb-size]
      (batcher strm (rand-int batch-size) i))
    (output batcher))
  IFn
  (invoke [this cols]
    (.invoke this strm cols))
  (invoke [_ strm2 cols]
    (loop [src-n (first cols) cols (rest cols) dst-n 0]
      (when src-n
        (batcher strm2 src-n dst-n)
        (recur (first cols) (rest cols) (inc dst-n))))
    (output batcher))
  (applyTo [this xs]
    (AFn/applyToHelper this xs))
  ConnectorCreator
  (connector [this dst-desc]
    (if (equal-desc? (output batcher) dst-desc)
      this
      (connector batcher dst-desc))))

(defn dnnl-shuffler [eng strm src-tz dst-tz]
  (->DnnlShuffler strm (dnnl-batcher eng strm src-tz dst-tz 1)
                  (batch-size src-tz) (batch-size dst-tz)))

;; ================================ Tensor ======================================

(deftype DnnlTensor [diamond-fact neand-fact eng master tz-mem
                     ^long n ^long c ^long n-index]
  Object
  (hashCode [x]
    (-> (hash :DnnlTensor) (hash-combine (hash tz-mem))))
  (equals [x y]
    (or (identical? x y)
        (and (instance? DnnlTensor y) (equal-desc? tz-mem (desc y))
             (.isContiguous x) (.isContiguous ^DnnlTensor y)
             (= (view-vctr x) (view-vctr y)))))
  (toString [this]
    (pr-str {:shape (dims tz-mem) :data-type (data-type tz-mem) :layout (strides tz-mem)}))
  Info
  (info [x]
    {:data-type (data-type tz-mem)
     :class (class x)
     :device :cpu
     :shape (shape x)
     :strides (strides tz-mem)
     :master master
     :engine eng})
  (info [x info-type]
    (case info-type
      :data-type (data-type tz-mem)
      :class (class x)
      :device :cpu
      :shape (shape x)
      :strides (strides tz-mem)
      :master master
      :engine eng
      nil))
  Releaseable
  (release [_]
    (if master
      (release tz-mem)
      true))
  Wrapper
  (extract [_]
    (extract tz-mem))
  EngineProvider
  (engine [_]
    eng)
  DiamondFactoryProvider
  (diamond-factory [_]
    diamond-fact)
  (native-diamond-factory [_]
    (native-diamond-factory diamond-fact))
  FactoryProvider
  (factory [_]
    neand-fact)
  (native-factory [_]
    (native-factory neand-fact))
  DataAccessorProvider
  (data-accessor [_]
    (data-accessor neand-fact))
  Container
  (raw [_]
    (dnnl-tensor diamond-fact tz-mem n-index))
  (raw [_ fact]
    (let [df (diamond-factory fact)]
      (create-tensor df (create-tensor-desc df (desc tz-mem)) n-index false)))
  (zero [x]
    (dnnl-tensor diamond-fact tz-mem n-index))
  (zero [_ fact]
    (let [df (diamond-factory fact)]
      (create-tensor df (create-tensor-desc df (desc tz-mem)) n-index true)))
  (host [x]
    (let-release [res (raw x)]
      (copy eng x res)))
  (native [x]
    x)
  Seqable
  (seq [this]
    (seq (view-vctr this)))
  MemoryContext
  (compatible? [_ y]
    (compatible? neand-fact (factory y)))
  (fits? [_ y]
    (= (dims tz-mem) (shape y)))
  (device [_]
    :cpu)
  Monoid
  (id [_]
    (dnnl-tensor diamond-fact
                 (memory-desc (repeat (ndims tz-mem) 0)
                              (data-type tz-mem)
                              (repeat (ndims tz-mem) 0))
                 n-index))
  Functor
  (fmap [x f]
    (f x))
  (fmap [x f xs]
    (apply f x xs))
  Applicative
  (pure [x v]
    (let-release [res (dnnl-tensor diamond-fact
                                   (memory-desc (repeat (ndims tz-mem) 1)
                                                (data-type tz-mem)
                                                (repeat (ndims tz-mem) 1))
                                   n-index)]
      (set-all eng v x)))
  (pure [x v vs]
    (let [vs (cons v vs)]
      (let-release [res (dnnl-tensor diamond-fact
                                     (memory-desc (cons (count vs) (repeat (dec (ndims tz-mem)) 1))
                                                  (data-type tz-mem) (repeat (ndims tz-mem) 1))
                                     n-index)]
        (transfer! vs res))))
  Changeable
  (setBoxed [x v]
    (set-all eng v x)
    x)
  (setBoxed [x i val]
    (dragan-says-ex "Tensors do not support editing of specific entries. Please use tensor's vector view."))
  (alter [x f]
    (check-contiguous x)
    (alter (view-vctr x) f)
    x)
  (alter [x i f]
    (check-contiguous x)
    (alter (view-vctr x) i f)
    x)
  VectorSpace
  (dim [_]
    (* n c))
  Block
  (buffer [_]
    (pointer tz-mem))
  (offset [_]
    0)
  (stride [_]
    1)
  (isContiguous [_]
    (= (size tz-mem) (apply * (dims tz-mem)) ))
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
    (desc tz-mem))
  DescriptorProvider
  (inf-desc [_]
    (desc tz-mem))
  (train-desc [_]
    (desc tz-mem))
  (diff-desc [_]
    (desc tz-mem))
  TensorDescriptor
  (shape [_]
    (dims tz-mem))
  (data-type [_]
    (data-type tz-mem))
  (layout [_]
    (strides tz-mem))
  BatchDescriptor
  (batch-index [_]
    n-index)
  Viewable
  (view [this]
    (->DnnlTensor diamond-fact neand-fact eng false tz-mem n c n-index))
  DenseContainer
  (view-vctr [this]
    (if (<= (dim this) (size tz-mem))
      (create-vector neand-fact false (pointer tz-mem 0) (dim this) 0 1)
      (dragan-says-ex "Strided tensors cannot be viewed as vectors."
                      {:tensor (info this) :dim (dim this) :size (size tz-mem)})))
  TensorContainer
  (view-tz [this]
    this)
  (view-tz [_ sub]
    (let-release [sub-desc (if (number? sub)
                             (if (= 0 n-index)
                               (submemory-desc tz-mem sub)
                               (submemory-desc tz-mem (assoc (dims tz-mem) n-index sub)))
                             (memory-desc (shape sub) (or (tz/data-type sub) (data-type tz-mem))
                                          (or (layout sub) (strides tz-mem))))
                  sub-mem (memory (dnnl-engine diamond-fact) sub-desc (pointer tz-mem 0) false)
                  shp (dims sub-mem)]
      (->DnnlTensor diamond-fact neand-fact eng false sub-mem (first shp) (apply * (rest shp))
                    (if (= (count shp) (count (dims tz-mem))) n-index 0))))
  Offset
  (offset [this ofst]
    (offset! tz-mem ofst)
    this)
  ConnectorCreator
  (connector [in-tz out-desc]
    (if (equal-desc? tz-mem out-desc)
      (view in-tz)
      (let-release [out-tz (dnnl-tensor diamond-fact neand-fact eng out-desc (batch-index in-tz))]
        (dnnl-transformer (dnnl-engine diamond-fact) (flow diamond-fact) (view in-tz) out-tz)))))

(defn dnnl-tensor
  ([diamond-fact neand-fact eng mem-desc n-index]
   (let [mem-desc (desc mem-desc)
         tz-mem (memory (dnnl-engine diamond-fact) mem-desc)
         shp (dims mem-desc)]
     (->DnnlTensor diamond-fact neand-fact eng true tz-mem (first shp) (apply * (rest shp)) n-index)))
  ([diamond-fact neand-fact eng mem-desc]
   (dnnl-tensor diamond-fact neand-fact eng mem-desc 0))
  ([diamond-fact mem-desc n-index]
   (let [dtype (tz/data-type (desc mem-desc))]
     (dnnl-tensor diamond-fact (neanderthal-factory diamond-fact dtype)
                  (tensor-engine diamond-fact dtype) mem-desc n-index)))
  ([diamond-fact mem-desc]
   (dnnl-tensor diamond-fact mem-desc 0)))

(defn dnnl-tensor* [diamond-fact mem-desc buf n-index master]
 (let [mem-desc (desc mem-desc)
       tz-mem (memory (dnnl-engine diamond-fact) mem-desc buf master)
       shp (dims mem-desc)
       dtype (tz/data-type mem-desc)]
   (->DnnlTensor diamond-fact
                 (neanderthal-factory diamond-fact dtype)
                 (tensor-engine diamond-fact dtype)
                 true tz-mem (first shp) (apply * (rest shp))
                 n-index)))

(defmethod print-method DnnlTensor
  [^DnnlTensor x ^java.io.Writer w]
  (.write w (str x))
  (.write w " ")
  (if (contiguous? x)
    (print-method (doall (take *print-length* (seq x))) w)
    (.write w "(... non-printable ...)")))

(defmethod transfer! [DnnlTensor DnnlTensor]
  [source destination]
  (if (equal-desc? source destination)
    (copy! source destination)
    (with-release [transform! (transformer source destination)]
      (transform!)))
  destination)

(defmethod transfer! [Object DnnlTensor]
  [source destination]
  (transfer! source (view-vctr destination))
  destination)

(defmethod transfer! [DnnlTensor Object]
  [source destination]
  (transfer! (view-vctr source) destination))

(defmethod transfer! [Object DnnlTransformer]
  [source destination]
  (transfer! source (view-vctr (input destination)))
  destination)

(defmethod transfer! [DnnlTransformer Object]
  [source destination]
  (transfer! (view-vctr (output source)) destination))

(defmethod transfer! [DnnlTensor DnnlTransformer]
  [source destination]
  (transfer! source (input destination))
  destination)

(defmethod transfer! [DnnlTransformer DnnlTensor]
  [source destination]
  (transfer! (output source) destination))

(prefer-method transfer! [DnnlTensor Object] [Object DnnlTensor])
(prefer-method transfer! [DnnlTensor DnnlTensor] [Object DnnlTensor])
(prefer-method transfer! [DnnlTensor DnnlTensor] [DnnlTensor Object])
