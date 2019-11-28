(ns uncomplicate.diamond.internal.dnnl.tensor
  (:require [uncomplicate.commons
             [core :refer [Releaseable release let-release with-release]]
             [utils :refer [dragan-says-ex]]]
            [uncomplicate.neanderthal
             [core :refer [transfer!]]
             [block :refer [entry-width data-accessor buffer count-entries]]
             [native :refer [factory-by-type]]]
            [uncomplicate.neanderthal.internal.api :as neand
             :refer [Viewable view flow]]
            [uncomplicate.neanderthal.internal.host.buffer-block :refer [real-block-vector]]
            [uncomplicate.diamond.tensor
             :refer [TensorDescriptor shape layout TensorContainer Transfer input output
                     Revert ConnectorCreator connector view-tz]
             :as tz]
            [uncomplicate.diamond.internal.protocols
             :refer [TensorFactory FactoryProvider ContextProvider factory context]]
            [uncomplicate.diamond.internal.dnnl
             [core :refer [memory-desc dims data-type memory size strides submemory-desc
                           equal-desc? execute! reorder primitive fwd-args]]
             [protocols :refer [DescProvider desc data ptr]]])
  (:import org.bytedeco.javacpp.Pointer
           [clojure.lang Seqable IFn]
           uncomplicate.neanderthal.internal.api.Block
           org.bytedeco.dnnl.dnnl_memory_desc_t
           uncomplicate.diamond.tensor.TensorDescriptorImpl))

(declare ->DnnlTensor dnnl-transformer dnnl-tensor)

(extend-type java.util.Collection
  DescProvider
  (desc [this]
    (memory-desc this :undef :undef)))

(extend-type java.util.Map
  DescProvider
  (desc [this]
    (memory-desc (:shape this) (or (:data-type this) :undef) (or (:data-type this) :undef))))

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
       :default (ex-info "Offset function can not accept this type of stride collection."
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

(defn dnnl-tensor
  ([fact neand-fact eng mem-desc]
   (let [mem-desc (desc mem-desc)
         tz-mem (memory eng mem-desc)]
     (->DnnlTensor fact neand-fact eng (offset (strides mem-desc)) true tz-mem)))
  ([fact mem-desc]
   (dnnl-tensor fact (factory-by-type (tz/data-type mem-desc))
                  (context fact) mem-desc)))

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
