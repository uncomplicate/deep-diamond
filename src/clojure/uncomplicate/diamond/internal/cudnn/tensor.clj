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
            [uncomplicate.clojurecuda.core :refer [memcpy-host! mem-alloc]]
            [uncomplicate.clojurecuda.internal.protocols :as cuda]
            [uncomplicate.neanderthal
             [core :refer [transfer! dim vctr]]
             [block :refer [entry-width buffer data-accessor count-entries create-data-source]]
             [cuda :refer [factory-by-type]]]
            [uncomplicate.neanderthal.internal
             [api :refer [Viewable view flow equals-block compatible? set-all MemoryContext
                          EngineProvider Container DataAccessorProvider FactoryProvider
                          native-factory zero raw host factory]]
             [printing :refer [print-vector]]]
            [uncomplicate.neanderthal.internal.device.cublock :refer [cu-block-vector]]
            [uncomplicate.diamond.tensor
             :refer [TensorDescriptor shape layout data-type TensorContainer Transfer
                     input output Revert ConnectorCreator connector view-tz]]
            [uncomplicate.diamond.internal.protocols
             :refer [TensorFactory DiamondFactoryProvider ContextProvider create-tensor
                     diamond-factory context neanderthal-factory tensor-engine]]
            [uncomplicate.diamond.internal.dnnl.protocols :refer [data]]
            [uncomplicate.diamond.internal.cudnn
             [core :refer [tensor-descriptor equal-desc? size]]
             [protocols :refer [DescProvider desc]]])
  (:import clojure.lang.IFn
           [uncomplicate.neanderthal.internal.api Block RealChangeable DataAccessor VectorSpace]
           uncomplicate.diamond.tensor.TensorDescriptorImpl
           uncomplicate.diamond.internal.dnnl.tensor.DnnlTensor
           uncomplicate.diamond.internal.cudnn.impl.CUTensorDescriptor))

(def ^{:private true :const true} INEFFICIENT_OPERATION_MSG
  "This operation would be inefficient because it uses memory transfer.
  Please use transfer! to be reminded of that.")


(defn ^:private not-available []
  (throw (UnsupportedOperationException. "Not available in CUDA. Please use a host instance.")))

(defn set-tensor! [host cuda]
  (memcpy-host! (data (buffer host)) (buffer cuda)
                (flow (diamond-factory cuda)))
  cuda)

(defn get-tensor! [cuda host]
  (memcpy-host! (buffer cuda) (data (buffer host))
                (flow (diamond-factory cuda)))
  host)

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
    (tensor-descriptor (.shape this) (or (.data-type this) :float) (or (layout this) :nchw)))
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
          (let [fact (diamond-factory out-tz)]
            (let-release [in-tz (cudnn-tensor fact in-desc)]
              (cudnn-transformer (context fact) (flow fact) in-tz (view-tz out-tz)))))))))

(defmethod print-method CUTensorDescriptor
  [^CUTensorDescriptor d ^java.io.Writer w]
  (.write w (pr-str {:shape (.dims d) :data-type (.data-type d) :layout (.strides d)})))

(deftype CUDnnTensor [diamond-fact eng vect-view master buf ofst ^CUTensorDescriptor cu-desc]
  Object
  (hashCode [x]
    (-> (hash :CUDnnTensor) (hash-combine (hash cu-desc))))
  (equals [x y]
    (cond
      (nil? y) false
      (identical? x y) true
      (and (instance? CUDnnTensor y) (equal-desc? cu-desc (desc y)))
      (equals-block eng x y)
      :default false))
  (toString [this]
    (pr-str {:shape (.dims cu-desc) :data-type (.data-type cu-desc) :layout (.strides cu-desc)}))
  Info
  (info [x]
    {:data-type (.data-type cu-desc)
     :class (class x)
     :device :cuda
     :shape (.dims cu-desc)
     :offset ofst
     :strides (.strides cu-desc)
     :master master
     :engine eng})
  (info [x info-type]
    (case info-type
      :data-type (.data-type cu-desc)
      :class (class x)
      :device :cuda
      :shape (.dims cu-desc)
      :offset ofst
      :strides (.strides cu-desc)
      :master master
      :engine eng
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
  DiamondFactoryProvider
  (diamond-factory [_]
    diamond-fact)
  FactoryProvider
  (factory [_]
    (factory vect-view))
  DataAccessorProvider
  (data-accessor [_]
    (data-accessor vect-view))
  Container
  (raw [_]
    (cudnn-tensor diamond-fact cu-desc false))
  (raw [_ fact]
    (create-tensor fact cu-desc false))
  (zero [x]
    (zero x diamond-fact))
  (zero [_ fact]
    (create-tensor diamond-fact cu-desc true))
  (host [x]
    (let-release [res (raw x (native-factory diamond-fact))]
      (get-tensor! x res)))
  (native [x]
    (host x))
  Block
  (buffer [_]
    buf)
  (offset [_]
    ofst)
  (stride [_]
    (dragan-says-ex "Tensors do not have a single stride. You're doing something wrong."))
  (isContiguous [_]
    (= (size cu-desc)
       (apply * (entry-width (data-accessor vect-view)) (.dims cu-desc))))
  Viewable
  (view [_]
    vect-view)
  MemoryContext
  (compatible? [_ y]
    (and (instance? CUDnnTensor y) (compatible? vect-view (view y))))
  (fits? [_ y]
    (= (.dims cu-desc) (shape y)))
  (device [_]
    :cuda)
  VectorSpace
  (dim [_]
    (apply * (.dims cu-desc)))
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
    cu-desc)
  TensorDescriptor
  (shape [_]
    (.dims cu-desc))
  (data-type [_]
    (.data-type cu-desc))
  (layout [_]
    (.strides cu-desc))
  TensorContainer
  (view-tz [_]
    (->CUDnnTensor diamond-fact vect-view eng false buf ofst cu-desc))
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
  ([diamond-fact master buf tdesc]
   (let [neand-fact (neanderthal-factory diamond-fact (data-type tdesc))
         tz-cnt (apply * (shape tdesc))]
     (if (<= 0 (size tdesc) (cuda/size buf))
       (let-release [vect-view (cu-block-vector neand-fact false buf tz-cnt 0 1)]
         (->CUDnnTensor diamond-fact vect-view
                        (tensor-engine diamond-fact (data-type tdesc))
                        master buf 0 tdesc))
       (throw (ex-info "Insufficient buffer size." {:size (size tdesc) :buffer-size (cuda/size buf)})))))
  ([diamond-fact tdesc]
   (let-release [buf (mem-alloc (max 1 (size tdesc)))]
     (cudnn-tensor diamond-fact true buf tdesc))))

(defmethod print-method CUDnnTensor;;TODO see about printing entries...
  [^CUDnnTensor x ^java.io.Writer w]
  (.write w (str x)))

(defmethod transfer! [DnnlTensor CUDnnTensor]
  [source destination]
  (set-tensor! source destination))

(defmethod transfer! [CUDnnTensor DnnlTensor]
  [source destination]
  (get-tensor! source destination))
