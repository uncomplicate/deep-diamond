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
            [uncomplicate.neanderthal
             [core :refer [transfer! dim]]
             [block :refer [entry-width data-accessor buffer count-entries]]
             [native :refer [factory-by-type]]]
            [uncomplicate.neanderthal.internal
             [api :as neand :refer [Viewable view flow]]
             [printing :refer [print-vector]]]
            [uncomplicate.neanderthal.internal.host.buffer-block :refer [real-block-vector]]
            [uncomplicate.diamond.tensor
             :refer [TensorDescriptor shape layout TensorContainer Transfer input output
                     Revert ConnectorCreator connector view-tz]
             :as tz]
            [uncomplicate.diamond.internal.protocols
             :refer [TensorFactory FactoryProvider ContextProvider factory context]]
            [uncomplicate.diamond.internal.dnnl
             [core :refer [memory-desc dims data-type memory size strides submemory-desc
                           equal-desc? execute! reorder primitive fwd-args offset!]
              :as dnnl-core]
             [constants :refer [entry-bytes]]
             [protocols :refer [DescProvider desc data ptr]]])
  (:import org.bytedeco.javacpp.Pointer
           [clojure.lang Seqable IFn]
           uncomplicate.neanderthal.internal.api.Block
           org.bytedeco.dnnl.dnnl_memory_desc_t
           uncomplicate.diamond.tensor.TensorDescriptorImpl))

(declare ->DnnlTensor dnnl-transformer dnnl-tensor dnnl-shuffler)

(extend-type java.util.Collection
  DescProvider
  (desc [this]
    (memory-desc this :undef :undef)))

(extend-type java.util.Map
  DescProvider
  (desc [this]
    (memory-desc (:shape this) (or (:data-type this) :undef) (or (:layout this) :undef))))

(extend-type TensorDescriptorImpl
  DescProvider
  (desc [this]
    (memory-desc (.shape this) (or (.data-type this) :undef) (or (.layout this) :undef)))
  ConnectorCreator
  (connector [in-desc out]
    (connector (desc in-desc) out)))

(extend-type dnnl_memory_desc_t
  TensorDescriptor
  (shape [this]
    (dims this))
  (tz/data-type [this]
    (data-type this))
  ConnectorCreator
  (connector [in-desc out]
    (if (equal-desc? in-desc (input out))
      out
      (let [out-tz (output out)]
        (if (equal-desc? in-desc out-tz)
          (view-tz out-tz)
          (let [fact (factory out-tz)]
            (let-release [in-tz (dnnl-tensor fact in-desc)]
              (dnnl-transformer (context fact) (flow fact) in-tz (view-tz out-tz)))))))))

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

;; =================== Shuffler ==================================================

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
      (if (and (< -1 src-n) (< (+ mb-size src-n) (inc src-cnt))
               (< -1 dst-n) (< (+ mb-size dst-n) (inc dst-cnt)))
        (do
          (offset! src-submem (* src-entry-width src-stride-n src-n))
          (offset! dst-submem (* dst-entry-width dst-stride-n dst-n))
          (execute! strm2 reorder reorder-args))
        (dragan-says-ex "Requested subtensor is outside of bounds."
                        {:src-index src-n :src-cnt src-cnt :dst-index dst-n :dst-cnt dst-cnt
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

(let [array? (partial instance? (type (long-array 0)))]
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

(deftype DnnlTensor [fact neand-fact eng offset-fn master tz-mem]
  Object
  (hashCode [x]
    (-> (hash :DnnlTensor) (hash-combine (hash tz-mem))))
  (equals [x y]
    (cond
      (nil? y) false
      (identical? x y) true
      (and (instance? DnnlTensor y) (= (dims tz-mem) (shape y)))
      (= (view x) (view y))
      :default false))
  (toString [this]
    (format "#DnnlTensor[%s, shape:%s, strides:%s]"
            (name (data-type tz-mem)) (dims tz-mem) (strides tz-mem)))
  Info
  (info [x]
    {:entry-type (data-type tz-mem)
     :class (class x)
     :device :cpu
     :shape (shape x)
     :strides (strides tz-mem)
     :offset (dnnl-core/offset tz-mem)})
  (info [x info-type]
    (case info-type
      :entry-type (data-type tz-mem)
      :class (class x)
      :device :cpu
      :shape (shape x)
      :strides (strides tz-mem)
      :offset (dnnl-core/offset tz-mem)
      nil))
  Releaseable
  (release [_]
    (if master
      (release tz-mem)
      true))
  FactoryProvider
  (factory [_]
    fact)
  Block
  (buffer [_]
    tz-mem)
  (isContiguous [_]
    (= (size tz-mem)
       (apply * (entry-width (data-accessor neand-fact)) (dims tz-mem)) ))
  Viewable
  (view [_]
    (let [ewidth (entry-width (data-accessor neand-fact))
          n (apply * (dims tz-mem))]
      (if (= (* (long n) ewidth) (size tz-mem))
        (real-block-vector neand-fact false (data tz-mem) n
                           (/ (.position ^Pointer (ptr tz-mem)) ewidth) 1)
        (dragan-says-ex "Strided tensors cannot be viewed as vectors."))))
  Revert
  (revert [this]
    this)
  Transfer
  (input [this]
    this)
  (output [this]
    this)
  Seqable
  (seq [this]
    (seq (view this)))
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
    (->DnnlTensor fact neand-fact eng offset-fn false tz-mem))
  (view-tz [_ sub]
    (let-release [sub-desc (if (number? sub)
                             (submemory-desc tz-mem sub)
                             (memory-desc (shape sub) (or (tz/data-type sub) (data-type tz-mem))
                                          (or (layout sub) (strides tz-mem))))
                  sub-mem (memory eng sub-desc (data tz-mem) false)]
      (->DnnlTensor fact neand-fact eng offset-fn (number? sub) sub-mem)))
  ConnectorCreator
  (connector [in-tz out-desc]
    (if (equal-desc? tz-mem out-desc)
      (view-tz in-tz)
      (let-release [out-tz (dnnl-tensor fact neand-fact eng out-desc)]
        (dnnl-transformer eng (flow fact) (view-tz in-tz) out-tz)))))

(defn dnnl-tensor
  ([fact neand-fact eng mem-desc]
   (let [mem-desc (desc mem-desc)
         tz-mem (memory eng mem-desc)]
     (->DnnlTensor fact neand-fact eng (offset (strides mem-desc)) true tz-mem)))
  ([fact mem-desc]
   (dnnl-tensor fact (factory-by-type (tz/data-type mem-desc))
                (context fact) mem-desc)))

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

(defmethod transfer! [DnnlTensor DnnlTensor]
  [source destination]
  (if (equal-desc? (buffer source) (buffer destination))
    (do
      (transfer! (view source) (view destination))
      destination)
    (dragan-says-ex "You need a specialized transformer to transfer these two MKL-DNN tensors.")))

(defmethod print-method DnnlTensor
  [^DnnlTensor x ^java.io.Writer w]
  (.write w (str x))
  (with-release [view-x (view x)]
    (when (< 0 (dim view-x))
      (print-vector w view-x))))
