;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.diamond.internal.dnnl.tensor
  (:require [uncomplicate.commons
             [core :refer [Releaseable release let-release with-release Info info]]
             [utils :refer [dragan-says-ex]]]
            [uncomplicate.fluokitten.protocols :refer [Magma Monoid Foldable Applicative pure]]
            [uncomplicate.neanderthal
             [core :refer [transfer! dim copy!]]
             [block :refer [entry-width data-accessor buffer count-entries]]]
            [uncomplicate.neanderthal.internal.api
             :refer [Viewable view flow FactoryProvider EngineProvider DataAccessorProvider
                     Container raw copy MemoryContext set-all compatible? factory native-factory
                     create-vector]]
            [uncomplicate.neanderthal.internal.host.buffer-block :refer [real-block-vector]]
            [uncomplicate.diamond.tensor
             :refer [TensorDescriptor shape layout TensorContainer Transfer input output
                     Revert ConnectorCreator connector view-tz transformer]
             :as tz]
            [uncomplicate.diamond.internal
             [protocols
              :refer [TensorFactory DiamondFactoryProvider diamond-factory create-tensor
                      neanderthal-factory tensor-engine native-diamond-factory Offset
                      DiffTransfer diff-input diff-output create-tensor-desc layers parameters]]
             [utils :refer [check-contiguous]]]
            [uncomplicate.diamond.internal.dnnl
             [core :refer [memory-desc dims data-type memory size strides submemory-desc
                           equal-desc? execute! reorder primitive fwd-args offset! ndims]
              :as dnnl-core]
             [constants :refer [entry-bytes]]
             [protocols :refer [DescProvider desc data ptr dnnl-engine]]])
  (:import org.bytedeco.javacpp.Pointer
           [clojure.lang Seqable IFn]
           [uncomplicate.neanderthal.internal.api Block VectorSpace Changeable]
           org.bytedeco.dnnl.dnnl_memory_desc_t
           uncomplicate.diamond.tensor.TensorDescriptorImpl))

(declare ->DnnlTensor dnnl-transformer dnnl-tensor dnnl-shuffler)

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

(extend-type dnnl_memory_desc_t
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
      out
      (let [out-tz (output out)]
        (if (equal-desc? in-desc out-tz)
          (view-tz out-tz)
          (let [fact (diamond-factory out-tz)]
            (let-release [in-tz (dnnl-tensor fact in-desc)]
              (dnnl-transformer (dnnl-engine fact) (flow fact) in-tz (view-tz out-tz)))))))))

(defmethod print-method dnnl_memory_desc_t
  [^dnnl_memory_desc_t d ^java.io.Writer w]
  (.write w (pr-str {:shape (dims d) :data-type (data-type d) :layout (strides d)})))

;; =================== Transformer ==============================================

(deftype DnnlTransformer [eng strm reorder reorder-args in-tz out-tz]
  Releaseable
  (release [_]
    (release in-tz)
    (release out-tz)
    (release reorder))
  Revert
  (revert [_]
    (dnnl-transformer eng strm (view-tz out-tz) (view-tz in-tz)))
  Transfer
  (input [_]
    in-tz)
  (output [_]
    out-tz)
  DiffTransfer
  (diff-input [_]
    out-tz)
  (diff-output [_]
    in-tz)
  IFn
  (invoke [_]
    (execute! strm reorder reorder-args)
    out-tz)
  (invoke [_ strm2]
    (execute! strm2 reorder reorder-args)
    out-tz)
  ConnectorCreator
  (connector [this out-desc]
    (if (equal-desc? out-tz out-desc)
      this
      (connector in-tz out-desc))))

(defn dnnl-transformer [eng strm in-tz out-tz]
  (with-release [reorder-pd (reorder eng (buffer in-tz) (buffer out-tz))]
    (let-release [reorder-prim (primitive reorder-pd)]
      (->DnnlTransformer eng strm reorder-prim
                         (fwd-args (buffer in-tz) (buffer out-tz))
                         in-tz out-tz))))

;; =================== Batcher ==================================================

(deftype DnnlBatcher [eng strm reorder reorder-args
                      src-submem dst-submem src-tz dst-tz ^long mb-size
                      ^long src-cnt ^long src-stride-n ^long src-entry-width
                      ^long dst-cnt ^long dst-stride-n ^long dst-entry-width]
  Releaseable
  (release [_]
    (release src-tz)
    (release dst-tz)
    (release src-submem)
    (release dst-submem)
    (release reorder))
  Transfer
  (input [_]
    src-tz)
  (output [_]
    dst-tz)
  DiffTransfer
  (diff-input [_]
    dst-tz)
  (diff-output [_]
    src-tz)
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
      (if (and (<= 0 src-n (- src-cnt mb-size)) (<= 0 dst-n (- dst-cnt mb-size)))
        (do
          (offset! src-submem (* src-entry-width src-stride-n src-n))
          (offset! dst-submem (* dst-entry-width dst-stride-n dst-n))
          (execute! strm2 reorder reorder-args))
        (dragan-says-ex "Requested subtensor is outside of bounds."
                        {:src-index src-n :src-cnt src-cnt
                         :dst-index dst-n :dst-cnt dst-cnt
                         :mb-size mb-size})))
    dst-tz)
  ConnectorCreator
  (connector [this dst-desc]
    (if (equal-desc? dst-tz dst-desc)
      this
      (connector src-tz dst-desc))))

(defn dnnl-batcher [eng strm src-tz dst-tz mb-size]
  (let [mb-size (max 1 (long mb-size))]
    (let-release [src-sub (view-tz src-tz mb-size)
                  dst-sub (view-tz dst-tz mb-size)]
      (with-release [reorder-pd (reorder eng (buffer src-sub) (buffer dst-sub))]
        (let-release [reorder-prim (primitive reorder-pd)]
          (->DnnlBatcher eng strm reorder-prim
                         (fwd-args (buffer src-sub) (buffer dst-sub))
                         (buffer src-sub) (buffer dst-sub)
                         (view-tz src-tz) (view-tz dst-tz)
                         mb-size
                         ((dims src-tz) 0) ((strides src-sub) 0)
                         (entry-bytes (data-type src-tz))
                         ((dims dst-tz) 0) ((strides dst-sub) 0)
                         (entry-bytes (data-type dst-tz))))))))


(deftype DnnlShuffler [strm batcher]
  Releaseable
  (release [_]
    (release batcher))
  Transfer
  (input [_]
    (input batcher))
  (output [_]
    (output batcher))
  DiffTransfer
  (diff-input [_]
    (diff-input batcher))
  (diff-output [_]
    (diff-output batcher))
  IFn
  (invoke [this cols]
    (.invoke this strm cols))
  (invoke [_ strm2 cols]
    (loop [src-n (first cols) cols (rest cols) dst-n 0]
      (when src-n
        (batcher strm2 src-n dst-n)
        (recur (first cols) (rest cols) (inc dst-n))))
    (output batcher))
  ConnectorCreator
  (connector [this dst-desc]
    (if (equal-desc? (output batcher) dst-desc)
      this
      (connector batcher dst-desc))))

(defn dnnl-shuffler [eng strm src-tz dst-tz]
  (->DnnlShuffler strm (dnnl-batcher eng strm src-tz dst-tz 1)))

;; ================================ Tensor ======================================

#_(let [array? (partial instance? (type (long-array 0)))]
    (defn offset
      ([sa]
       (cond
         (integer? sa)
         (let [sa (long sa)
               strides (long-array [sa])]
           (fn
             (^longs []
              strides)
             (^long [^long a]
              (* a sa))))
         (array? sa)
         (let [strides ^longs sa
               n (alength strides)]
           (fn
             (^longs []
              strides)
             (^long [^longs indices]
              (loop [i 0 res 0]
                (if (< i n)
                  (recur (inc i) (+ res (* (aget indices i) (aget strides i))))
                  res)))))
         (sequential? sa)
         (offset (long-array sa))
         :default (ex-info "Offset function cannot accept this type of stride collection."
                           {:type (type sa)})))
      ([^long sa ^long sb]
       (let [strides (long-array [sa sb])]
         (fn
           (^longs []
            strides)
           (^long [^long a ^long b]
            (+ (* a sa) (* b sb))))))
      ([^long sa ^long sb ^long sc]
       (let [strides (long-array [sa sb sc])]
         (fn
           (^longs []
            strides)
           (^long [^long a ^long b ^long c]
            (+ (* a sa) (* b sb) (* c sc))))))
      ([^long sa ^long sb ^long sc ^long sd]
       (let [strides (long-array [sa sb sc sd])]
         (fn
           (^longs []
            strides)
           (^long [^long a ^long b ^long c ^long d]
            (+ (* a sa) (* b sb) (* c sc) (* d sd))))))))

(deftype DnnlTensor [diamond-fact neand-fact eng master tz-mem
                     ^long n ^long c]
  Object
  (hashCode [x]
    (-> (hash :DnnlTensor) (hash-combine (hash tz-mem))))
  (equals [x y]
    (cond
      (nil? y) false
      (identical? x y) true
      (and (instance? DnnlTensor y) (equal-desc? tz-mem (desc y)))
      (= (view x) (view y))
      :default false))
  (toString [this]
    (pr-str {:shape (dims tz-mem) :data-type (data-type tz-mem) :layout (strides tz-mem)}))
  Info
  (info [x]
    {:data-type (data-type tz-mem)
     :class (class x)
     :device :cpu
     :shape (shape x)
     :strides (strides tz-mem)
     :offset (dnnl-core/offset tz-mem)
     :master master
     :engine eng})
  (info [x info-type]
    (case info-type
      :data-type (data-type tz-mem)
      :class (class x)
      :device :cpu
      :shape (shape x)
      :strides (strides tz-mem)
      :offset (dnnl-core/offset tz-mem)
      :master master
      :engine eng
      nil))
  Releaseable
  (release [_]
    (if master
      (release tz-mem)
      true))
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
    (dnnl-tensor diamond-fact tz-mem))
  (raw [_ fact]
    (let [df (diamond-factory fact)]
      (create-tensor df (create-tensor-desc df (desc tz-mem)) false)))
  (zero [x]
    (dnnl-tensor diamond-fact tz-mem))
  (zero [_ fact]
    (let [df (diamond-factory fact)]
      (create-tensor df (create-tensor-desc df (desc tz-mem)) true)))
  (host [x]
    (let-release [res (raw x)]
      (copy eng x res)))
  (native [x]
    x)
  Viewable
  (view [_]
    (let [ewidth (entry-width (data-accessor neand-fact))
          n (apply * (dims tz-mem))]
      (if (= (* (long n) ewidth) (size tz-mem))
        (create-vector neand-fact false (data tz-mem) n
                       (/ (.position ^Pointer (ptr tz-mem)) ewidth) 1)
        (dragan-says-ex "Strided tensors cannot be viewed as vectors."))))
  Seqable
  (seq [this]
    (seq (view this)))
  MemoryContext
  (compatible? [_ y]
    (compatible? neand-fact (factory y)))
  (fits? [_ y]
    (= (dims tz-mem) (shape y)))
  (device [_]
    :cpu)
  Monoid
  (id [_]
    (dnnl-tensor diamond-fact (memory-desc (repeat (ndims tz-mem) 0)
                                           (data-type tz-mem)
                                           (repeat (ndims tz-mem) 0))))
  Applicative
  (pure [x v]
    (let-release [res (dnnl-tensor diamond-fact (memory-desc (repeat (ndims tz-mem) 1)
                                                             (data-type tz-mem)
                                                             (repeat (ndims tz-mem) 1)))]
      (set-all eng v x)))
  (pure [x v vs]
    (let [vs (cons v vs)]
      (let-release [res (dnnl-tensor diamond-fact
                                     (memory-desc (cons (count vs) (repeat (dec (ndims tz-mem)) 1))
                                                  (data-type tz-mem) (repeat (ndims tz-mem) 1)))]
        (transfer! vs res))))
  Changeable
  (setBoxed [x v]
    (set-all eng v x)
    x)
  (setBoxed [x i val]
    (dragan-says-ex "Tensors do not support editing of specific entries. Please use tensor's vector view."))
  (alter [x f]
    (check-contiguous x)
    (alter (view x) f)
    x)
  (alter [x i f]
    (check-contiguous x)
    (alter (view x) i f)
    x)
  VectorSpace
  (dim [_]
    (* n c))
  Block
  (buffer [_]
    tz-mem)
  (offset [_]
    (quot (dnnl-core/offset tz-mem) (entry-width (data-accessor neand-fact))))
  (stride [_]
    (dragan-says-ex "Tensors do not have a single stride. You're doing something wrong."))
  (isContiguous [_]
    (= (size tz-mem)
       (apply * (entry-width (data-accessor neand-fact)) (dims tz-mem)) ))
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
  DescProvider
  (desc [_]
    (desc tz-mem))
  TensorDescriptor
  (shape [_]
    (dims tz-mem))
  (data-type [_]
    (data-type tz-mem))
  (layout [_]
    (strides tz-mem))
  TensorContainer
  (view-tz [_]
    (->DnnlTensor diamond-fact neand-fact eng false tz-mem n c))
  (view-tz [_ sub]
    (let-release [sub-desc (if (number? sub)
                             (submemory-desc tz-mem sub)
                             (memory-desc (shape sub) (or (tz/data-type sub) (data-type tz-mem))
                                          (or (layout sub) (strides tz-mem))))
                  sub-mem (memory (dnnl-engine diamond-fact) sub-desc (data tz-mem) false)
                  shp (dims sub-mem)]
      (->DnnlTensor diamond-fact neand-fact eng false sub-mem
                    (first shp) (apply * (rest shp)))))
  Offset
  (offset [this n-ofst]
    (offset! tz-mem (* (long n-ofst) (long (get (strides tz-mem) 0))
                       (entry-width (data-accessor neand-fact))))
    this)
  ConnectorCreator
  (connector [in-tz out-desc]
    (if (equal-desc? tz-mem out-desc)
      (view-tz in-tz)
      (let-release [out-tz (dnnl-tensor diamond-fact neand-fact eng out-desc)]
        (dnnl-transformer (dnnl-engine diamond-fact) (flow diamond-fact) (view-tz in-tz) out-tz)))))

(defn dnnl-tensor
  ([diamond-fact neand-fact eng mem-desc]
   (let [mem-desc (desc mem-desc)
         tz-mem (memory (dnnl-engine diamond-fact) mem-desc)
         shp (dims mem-desc)]
     (->DnnlTensor diamond-fact neand-fact eng true tz-mem (first shp) (apply * (rest shp)))))
  ([diamond-fact mem-desc]
   (let [dtype (tz/data-type (desc mem-desc))]
     (dnnl-tensor diamond-fact (neanderthal-factory diamond-fact dtype)
                  (tensor-engine diamond-fact dtype) mem-desc))))

(defn dnnl-tensor* [diamond-fact mem-desc buf master]
 (let [mem-desc (desc mem-desc)
       tz-mem (memory (dnnl-engine diamond-fact) mem-desc buf master)
       shp (dims mem-desc)
       dtype (tz/data-type mem-desc)]
   (->DnnlTensor diamond-fact
                 (neanderthal-factory diamond-fact dtype)
                 (tensor-engine diamond-fact dtype)
                 true tz-mem (first shp) (apply * (rest shp)))))

(defmethod print-method DnnlTensor
  [^DnnlTensor x ^java.io.Writer w]
  (.write w (str x))
  (.write w "\n")
  (print-method (doall (take *print-length* (seq (view x)))) w))

(defmethod transfer! [DnnlTensor DnnlTensor]
  [source destination]
  (if (equal-desc? source destination)
    (copy! source destination)
    (with-release [transform! (transformer source destination)]
      (transform!)))
  destination)

(defmethod transfer! [Object DnnlTensor]
  [source destination]
  (transfer! source (view destination))
  destination)

(defmethod transfer! [DnnlTensor Object]
  [source destination]
  (transfer! (view source) destination))

(defmethod transfer! [Object DnnlTransformer]
  [source destination]
  (transfer! source (view (input destination)))
  destination)

(defmethod transfer! [DnnlTransformer Object]
  [source destination]
  (transfer! (view (output source)) destination))

(defmethod transfer! [DnnlTensor DnnlTransformer]
  [source destination]
  (transfer! source (input destination))
  destination)

(defmethod transfer! [DnnlTransformer DnnlTensor]
  [source destination]
  (transfer! (output source) destination))
