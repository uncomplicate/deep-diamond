;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.diamond.internal.cudnn.tensor
  (:require [uncomplicate.commons
             [core :refer [Releaseable release let-release with-release Info info]]
             [utils :refer [dragan-says-ex]]]
            [uncomplicate.neanderthal
             [core :refer [transfer! dim vctr]]
             [block :refer [entry-width buffer count-entries]]
             [cuda :refer [factory-by-type]]]
            [uncomplicate.neanderthal.internal
             [api :refer [Viewable view flow equals-block compatible? set-all MemoryContext
                          EngineProvider]]
             [printing :refer [print-vector]]]
            [uncomplicate.neanderthal.internal.device.cublock :refer [cu-block-vector]]
            [uncomplicate.diamond.tensor
             :refer [TensorDescriptor shape layout data-type TensorContainer Transfer
                     input output Revert ConnectorCreator connector view-tz]]
            [uncomplicate.diamond.internal.protocols
             :refer [TensorFactory FactoryProvider ContextProvider
                     factory context data-accessor tensor-engine]]
            [uncomplicate.diamond.internal.cudnn
             [core :refer [tensor-descriptor equal-desc? size]]
             [protocols :refer [DescProvider desc]]])
  (:import clojure.lang.IFn
           [uncomplicate.neanderthal.internal.api Block RealChangeable DataAccessor VectorSpace]
           uncomplicate.diamond.tensor.TensorDescriptorImpl
           uncomplicate.diamond.internal.cudnn.impl.CUTensorDescriptor))

(def ^{:private true :const true} INEFFICIENT_OPERATION_MSG
  "This operation would be inefficient because it uses memory transfer.
  Please use transfer! to be reminded of that.")

(declare ->CUDnnTensor cudnn-transformer cudnn-tensor cudnn-shuffler)

(extend-type java.util.Collection
  DescProvider
  (desc [this]
    (tensor-descriptor this :float :nchw)))

(extend-type java.util.Map
  DescProvider
  (desc [this]
    (tensor-descriptor (:shape this) (or (:data-type this) :float) (or (layout this) :nchw))))

(extend-type TensorDescriptorImpl
  DescProvider
  (desc [this]
    (tensor-descriptor (.shape this) (or (.data-type this) :float) (or (layout this) :float)))
  ConnectorCreator
  (connector [in-desc out]
    (connector (desc in-desc) out)))

(extend-type CUTensorDescriptor
  TensorDescriptor
  (shape [this]
    (.dims this))
  (data-type [this]
    (.data-type this))
  (layout [this]
    (.strides this))
  ConnectorCreator
  (connector [in-desc out]
    (if (equal-desc? in-desc (input out))
      out
      (let [out-tz (output out)]
        (if (equal-desc? in-desc out-tz)
          (view-tz out-tz)
          (let [fact (factory out-tz)]
            (let-release [in-tz (cudnn-tensor fact in-desc)]
              (cudnn-transformer (context fact) (flow fact) in-tz (view-tz out-tz)))))))))

(defmethod print-method CUTensorDescriptor
  [^CUTensorDescriptor d ^java.io.Writer w]
  (.write w (pr-str {:shape (.dims d) :data-type (.data-type d) :layout (.strides d)})))

(deftype CUDnnTensor [fact ^DataAccessor da eng master buf ofst ^CUTensorDescriptor cu-desc]
  Object
  (hashCode [x]
    (-> (hash :CUDnnTensor) (hash-combine (hash cu-desc))))
  (equals [x y]
    (cond
      (nil? y) false
      (identical? x y) true
      (and (instance? CUDnnTensor y)
           (compatible? da (.da ^CUDnnTensor y)) (equal-desc? cu-desc (desc y)))
      (equals-block eng x y)
      :default false))
  (toString [this];;TODO
    (pr-str {:shape (.dims cu-desc) :data-type (.data-type cu-desc) :layout (.strides cu-desc)}))
  Info
  (info [x]
    {:data-type (.data-type cu-desc)
     :class (class x)
     :device :cpu
     :shape (.dims cu-desc)
     :strides (.strides cu-desc)})
  (info [x info-type]
    (case info-type
      :data-type (.data-type cu-desc)
      :class (class x)
      :device :cpu
      :shape (.dims cu-desc)
      :strides (.strides cu-desc)
      nil))
  Releaseable
  (release [_]
    (when master
      (release buf)
      (release cu-desc))
    true)
  EngineProvider
  (engine [_]
    eng)
  FactoryProvider
  (factory [_]
    fact)
  DescProvider
  (desc [_]
    cu-desc)
  Block
  (buffer [_]
    buf)
  (offset [_]
    ofst)
  (stride [_]
    (dragan-says-ex "Tensors do not have a single stride. You're doing something wrong."))
  (isContiguous [_]
    (= (size cu-desc)
       (apply * (entry-width da) (.dims cu-desc))))
  Viewable
  (view [_]
    "TODO")
  MemoryContext
  (compatible? [_ y]
    (instance? CUDnnTensor y))
  (fits? [_ y]
    (= (.dims cu-desc) (shape y)))
  (device [_]
    :cuda)
  VectorSpace
  (dim [_]
    (apply * (.dims cu-desc)))
  Revert
  (revert [this]
    this)
  Transfer
  (input [this]
    this)
  (output [this]
    this)
  IFn
  (invoke [this]
    this)
  DescProvider
  (desc [_]
    (desc cu-desc))
  RealChangeable
  (set [x val]
    (set-all eng val x)
    x)
  (set [_ _ _]
    (dragan-says-ex INEFFICIENT_OPERATION_MSG))
  (setBoxed [x val]
    (.set x val))
  (setBoxed [x i val]
    (.set x i val))
  (alter [_ _]
    (dragan-says-ex INEFFICIENT_OPERATION_MSG))
  (alter [_ _ _]
    (dragan-says-ex INEFFICIENT_OPERATION_MSG))
  TensorDescriptor
  (shape [_]
    (.dims cu-desc))
  (data-type [_]
    (.data-type cu-desc))
  (layout [_]
    (.strides cu-desc))
  TensorContainer
  (view-tz [_]
    (->CUDnnTensor fact da eng false buf ofst cu-desc))
  #_(view-tz [_ sub];;TODO
    (let-release [sub-desc (if (number? sub)
                             (submemory-desc tz-mem sub)
                             (memory-desc (shape sub) (or (tz/data-type sub) (data-type tz-mem))
                                          (or (layout sub) (strides tz-mem))))
                  sub-mem (memory sub-desc (data tz-mem) false)]
      (->DnnlTensor fact neand-fact eng offset-fn (number? sub) sub-mem)))
  ;;ConnectorCreator
  #_(connector [in-tz out-desc];;TODO
    (if (equal-desc? tz-mem out-desc)
      (view-tz in-tz)
      (let-release [out-tz (dnnl-tensor fact neand-fact eng out-desc)]
        (dnnl-transformer eng (flow fact) (view-tz in-tz) out-tz)))))

(defn cudnn-tensor
  ([fact ^DataAccessor da master buf tdesc]
   (let [n (apply * (shape tdesc))]
     (if (and (<= 0 n (.count da buf)))
       (->CUDnnTensor fact da (tensor-engine fact (data-type tdesc)) master buf 0 tdesc)
       (throw (ex-info "Insufficient buffer size." {:size n :buffer-size (.count da buf)})))))
  ([fact tdesc]
   (let-release [da (data-accessor fact (data-type tdesc))
                 buf (.createDataSource da (apply * (shape tdesc)))]
     (cudnn-tensor fact da true buf tdesc))))

(defmethod print-method CUDnnTensor;;TODO see about printing entries...
  [^CUDnnTensor x ^java.io.Writer w]
  (.write w (str x)))
