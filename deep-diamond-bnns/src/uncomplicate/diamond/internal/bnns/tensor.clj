;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.diamond.internal.bnns.tensor
  (:require [uncomplicate.commons
             [core :refer [Releaseable release let-release with-release Info info
                           Viewable view size Bytes bytesize* bytesize]]
             [utils :refer [dragan-says-ex]]]
            [uncomplicate.fluokitten.protocols
             :refer [Magma Monoid Applicative Functor Foldable Comonad extract fold foldmap]]
            [uncomplicate.clojure-cpp :refer [pointer safe null? get-entry zero! memcpy!]]
            [uncomplicate.neanderthal
             [core :refer [transfer! copy! dim]]
             [block :refer [entry-width data-accessor buffer contiguous?]]]
            [uncomplicate.neanderthal.internal.api
             :refer [create-vector flow FactoryProvider EngineProvider DataAccessorProvider
                     Container raw zero copy MemoryContext compatible? factory native-factory
                     create-vector* DenseContainer view-vctr]]
            [uncomplicate.neanderthal.internal.cpp.structures :refer [real-block-vector]]
            [uncomplicate.diamond.tensor
             :refer [TensorDescriptor shape TensorContainer Transfer input output
                     Revert ConnectorCreator connector view-tz transformer batch-size
                     default-desc]
             :as tz]
            [uncomplicate.diamond.internal
             [protocols
              :refer [TensorFactory DiamondFactoryProvider diamond-factory create-tensor
                      neanderthal-factory tensor-engine native-diamond-factory Offset offset
                      DiffTransfer diff-input diff-output create-tensor-desc parameters
                      DescriptorProvider BatchDescriptor batch-index]]
             [utils :refer [check-contiguous default-strides]]]
            [uncomplicate.diamond.internal.bnns
             [impl :refer [nda-shape-size]]
             [core :as bnns
              :refer [equal-desc? nda-desc dims data-type layout strides]]
             [constants :refer [bnns-data-type-pointer]]
             [protocols :refer [DescProvider desc data*]]])
  (:import org.bytedeco.javacpp.Pointer
           [clojure.lang Seqable IFn AFn]
           [uncomplicate.neanderthal.internal.api Block VectorSpace Changeable Vector]
           uncomplicate.diamond.tensor.TensorDescriptorImpl
           [uncomplicate.diamond.internal.bnns.impl BnnsTensorDescriptorImpl BnnsNdArrayDescriptorImpl BnnsTensorImpl]))

(declare ->BnnsTensor bnns-tensor)

(extend-type TensorDescriptorImpl
  DescProvider
  (desc [this]
    (nda-desc (.shape this) (or (.data-type this) :float)
              (or (tz/layout this) (default-strides this)))))

(extend-type BnnsTensorDescriptorImpl
  TensorDescriptor
  (shape [this]
    (dims this))
  (data-type [this]
    (data-type this))
  (layout [this]
    (strides this))
  ;;TODO
  ;; ConnectorCreator
  ;; (connector [in-desc out]
  ;;   (if (equal-desc? in-desc (input out))
  ;;     (view out)
  ;;     (let [out-tz (output out)]
  ;;       (if (equal-desc? in-desc out-tz)
  ;;         (view out-tz)
  ;;         (let [fact (diamond-factory out-tz)]
  ;;           (let-release [in-tz (bnns-tensor fact (view in-desc) (batch-index out-tz))]
  ;;             (cudnn-transformer (handle fact) in-tz (view out-tz))))))))
  )

(extend-type BnnsTensorImpl
  TensorDescriptor
  (shape [this]
    (dims this))
  (data-type [this]
    (data-type this))
  (layout [this]
    (strides this))
  ;;TODO
  ;; ConnectorCreator
  ;; (connector [in-desc out]
  ;;   (if (equal-desc? in-desc (input out))
  ;;     (view out)
  ;;     (let [out-tz (output out)]
  ;;       (if (equal-desc? in-desc out-tz)
  ;;         (view out-tz)
  ;;         (let [fact (diamond-factory out-tz)]
  ;;           (let-release [in-tz (bnns-tensor fact (view in-desc) (batch-index out-tz))]
  ;;             (cudnn-transformer (handle fact) in-tz (view out-tz))))))))
  )

(defmethod print-method BnnsTensorDescriptorImpl
  [^BnnsTensorDescriptorImpl d ^java.io.Writer w]
  (.write w (pr-str {:shape (tz/shape d) :data-type (tz/data-type d) :layout (tz/layout d)})))

(extend-type BnnsNdArrayDescriptorImpl
  TensorDescriptor
  (shape [this]
    (dims this))
  (data-type [this]
    (data-type this))
  (layout [this]
    (bnns/layout this))
  ;;TODO
  ;; ConnectorCreator
  ;; (connector [in-desc out]
  ;;   (if (equal-desc? in-desc (input out))
  ;;     (view out)
  ;;     (let [out-tz (output out)]
  ;;       (if (equal-desc? in-desc out-tz)
  ;;         (view out-tz)
  ;;         (let [fact (diamond-factory out-tz)]
  ;;           (let-release [in-tz (bnns-tensor fact (view in-desc) (batch-index out-tz))]
  ;;             (cudnn-transformer (handle fact) in-tz (view out-tz))))))))
  )

(defmethod print-method BnnsNdArrayDescriptorImpl
  [^BnnsNdArrayDescriptorImpl d ^java.io.Writer w]
  (.write w (pr-str {:shape (tz/shape d) :data-type (tz/data-type d) :layout (tz/layout d)})))

(deftype BnnsTensor [diamond-fact neand-fact eng master tz-desc vector-view ^long nc ^long n-index]
  Object
  (hashCode [x]
    (-> (hash :BnnsTensor) (hash-combine (hash tz-desc))))
  (equals [x y]
    (or (identical? x y)
        (and (instance? BnnsTensor y) (equal-desc? tz-desc y)
             (= (strides tz-desc) (tz/layout y))
             vector-view (.isContiguous ^BnnsTensor y)
             (= vector-view (view-vctr y)))))
  (toString [this]
    (pr-str {:shape (dims tz-desc) :data-type (data-type tz-desc) :layout (strides tz-desc)}))
  Info
  (info [x]
    {:data-type (data-type tz-desc)
     :class (class x)
     :device :cpu
     :shape (dims tz-desc)
     :layout (strides tz-desc)
     :master master
     :engine eng})
  (info [x info-type]
    (case info-type
      :data-type (data-type tz-desc)
      :class (class x)
      :device :cpu
      :shape (dims tz-desc)
      :layout (strides tz-desc)
      :master master
      :engine eng
      nil))
  Releaseable
  (release [_]
    (if master
      (release tz-desc)
      true))
  Bytes
  (bytesize* [_]
    (bytesize* tz-desc))
  Comonad
  (extract [_]
    (extract tz-desc))
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
  (raw [x]
    (raw x diamond-fact))
  (raw [_ fact]
    (create-tensor (diamond-factory fact) (default-desc tz-desc) n-index false))
  (zero [x]
    (zero x diamond-fact))
  (zero [_ fact]
    (create-tensor (diamond-factory fact) (default-desc tz-desc) n-index true))
  (host [x]
    (let-release [res (raw x)]
      (memcpy! (buffer x) (buffer res))))
  (native [x]
    x)
  Seqable
  (seq [this]
    (seq vector-view))
  MemoryContext
  (compatible? [_ y]
    (compatible? neand-fact (factory y)))
  (fits? [_ y]
    (= (dims tz-desc) (dims y)))
  (device [_]
    :cpu)
  Monoid
  (id [_]
    (with-release [ndims (count (dims tz-desc))
                   md (nda-desc (repeat ndims 0)
                                (data-type tz-desc)
                                (repeat ndims 0))]
      (bnns-tensor diamond-fact md n-index)))
  Functor
  (fmap [x f]
    (f x))
  (fmap [x f xs]
    (apply f x xs))
  Foldable
  (fold [this]
    (check-contiguous this)
    (fold vector-view))
  (fold [this f init]
    (check-contiguous this)
    (fold vector-view f init))
  (fold [this f init y]
    (check-contiguous this y)
    (fold vector-view f init (view-vctr y)))
  (fold [this f init y z]
    (check-contiguous this y z)
    (fold vector-view f init (view-vctr y) (view-vctr z)))
  (fold [this f init y z v]
    (check-contiguous this y z v)
    (fold vector-view f init (view-vctr y) (view-vctr z) (view-vctr v)))
  (fold [this f init y z v ws]
    (check-contiguous this y z v)
    (doseq [w ws] (check-contiguous w))
    (fold vector-view f (view-vctr y) (view-vctr z) (view-vctr v) (map view-vctr ws)))
  (foldmap [this g]
    (check-contiguous this)
    (foldmap vector-view g))
  (foldmap [this g f init]
    (check-contiguous this)
    (foldmap vector-view g f init))
  (foldmap [this g f init y]
    (check-contiguous this y)
    (foldmap vector-view f init (view-vctr y)))
  (foldmap [this g f init y z]
    (check-contiguous this y z)
    (foldmap vector-view g f init (view-vctr y) (view-vctr z)))
  (foldmap [this g f init y z v]
    (check-contiguous this y z v)
    (foldmap vector-view g f init (view-vctr y) (view-vctr z) (view-vctr v)))
  (foldmap [this g f init y z v ws]
    (check-contiguous this y z v)
    (doseq [w ws] (check-contiguous w))
    (foldmap vector-view g f (view-vctr y) (view-vctr z) (view-vctr v) (map view-vctr ws)))
  Applicative
  (pure [_ v]
    (with-release [ndims (count (shape tz-desc))
                   md (nda-desc (repeat ndims 1)
                                (tz/data-type tz-desc)
                                (repeat ndims 1))]
      (let-release [res (bnns-tensor diamond-fact md n-index)]
        (zero! (buffer res))
        res)))
  (pure [_ v vs]
    (let [vs (cons v vs)]
      (with-release [ndims (count (shape tz-desc))
                     md (nda-desc (cons (count vs) (repeat (dec ndims) 1))
                                  (tz/data-type tz-desc) (repeat ndims 1))]
        (let-release [res (bnns-tensor diamond-fact md n-index)]
          (transfer! vs res)))))
  VectorSpace
  (dim [_]
    nc)
  Vector
  (boxedEntry [_ i]
    (.boxedEntry ^Vector vector-view i))
  Block
  (buffer [_]
    (extract tz-desc))
  (offset [_]
    0)
  (stride [_]
    (.stride ^Block vector-view))
  (isContiguous [_]
    (boolean vector-view))
  Changeable
  (setBoxed [x v]
    (.setBoxed ^Changeable vector-view v)
    x)
  (setBoxed [_ _ _]
    (dragan-says-ex "Tensors do not support editing of specific entries. Please use tensor's vector view."))
  (alter [x f]
    (.alter ^Changeable vector-view f)
    x)
  (alter [x i f]
    (.alter ^Changeable vector-view i f)
    x)
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
    (desc tz-desc))
  DescriptorProvider
  (inf-desc [_]
    (desc tz-desc))
  (train-desc [_]
    (desc tz-desc))
  (diff-desc [_]
    (desc tz-desc))
  TensorDescriptor
  (shape [_]
    (dims tz-desc))
  (data-type [_]
    (data-type tz-desc))
  (layout [_]
    (strides tz-desc))
  BatchDescriptor
  (batch-index [_]
    n-index)
  Viewable
  (view [this]
    (->BnnsTensor diamond-fact neand-fact eng false tz-desc vector-view nc n-index))
  DenseContainer
  (view-vctr [this]
    (or vector-view (dragan-says-ex "Strided tensors cannot be viewed nor used as vectors."
                                    {:tensor (info this) :dim nc :size (size tz-desc)})))
  TensorContainer
  (view-tz [this]
    this)
  (view-tz [_ sub]
    (let-release [sub-desc (if (number? sub)
                             (nda-desc (assoc (dims tz-desc) n-index sub)
                                       (data-type tz-desc)
                                       (bnns/layout (desc tz-desc))
                                       (strides tz-desc))
                             (let [sub-layout (tz/layout sub)
                                   dtype (or (tz/data-type sub) (data-type tz-desc))]
                               (if (keyword? sub-layout)
                                 (nda-desc (shape sub) dtype sub-layout
                                           (strides tz-desc))
                                 (nda-desc (shape sub) dtype
                                           (bnns/layout (desc tz-desc))
                                           sub-layout))))]
      (let [dtype (data-type sub-desc)
            neand-fact (neanderthal-factory diamond-fact dtype)]
        (let-release [sub-buf (create-vector neand-fact
                                             false (pointer (buffer vector-view) 0)
                                             (size sub-desc) 0 1)
                      sub-tz-desc (bnns/tensor sub-desc (buffer sub-buf))]
          (->BnnsTensor diamond-fact neand-fact (tensor-engine diamond-fact dtype)
                        true sub-tz-desc sub-buf (dim sub-buf) n-index)))))
  ;; Offset
  ;; (offset [this ofst]
  ;;   (offset! tz-desc ofst)
  ;;   this)
  ;; ConnectorCreator
  ;; (connector [in-tz out-desc]
  ;;   (if (equal-desc? tz-desc out-desc)
  ;;     (view in-tz)
  ;;     (let-release [out-tz (bnns-tensor diamond-fact neand-fact eng out-desc (batch-index in-tz))]
  ;;       (bnns-transformer (bnns-engine diamond-fact) (flow diamond-fact) (view in-tz) out-tz))))
  )

(defn bnns-tensor
  ([diamond-fact tdesc buf n-index master]
   (let [tdesc (desc tdesc)]
     (if (null? (data* tdesc))
       (let [dtype (data-type tdesc)
             neand-fact (neanderthal-factory diamond-fact dtype)
             nc (apply * (dims tdesc))
             tz-bytesize (bytesize tdesc)]
         (if (<= 0 tz-bytesize (bytesize buf))
           (let-release [vect-view (if (= nc (size tdesc))
                                     (create-vector neand-fact master buf nc 0 1)
                                     nil)
                         tz-desc (bnns/tensor tdesc buf)]
             (->BnnsTensor diamond-fact neand-fact
                           (tensor-engine diamond-fact dtype)
                           master tz-desc vect-view nc n-index))
           (throw (dragan-says-ex "Insufficient buffer size."
                                  {:desc-size (bytesize tdesc) :buffer-size (bytesize buf)}))))
       (throw (dragan-says-ex "We cannot overwrite NDA descriptor's existing data pointer! Please provide a fresh NDA descriptor."
                              {:nda-desc tdesc})))))
  ([diamond-fact tdesc buf master]
   (bnns-tensor diamond-fact tdesc buf 0 master))
  ([diamond-fact tdesc n-index]
   (let [tdesc (desc tdesc)]
     (let-release [buf ((bnns-data-type-pointer (data-type tdesc))
                        (max 1 (size tdesc)))]
       (bnns-tensor diamond-fact tdesc buf n-index true))))
  ([diamond-fact tdesc]
   (bnns-tensor diamond-fact tdesc 0)))

(defmethod print-method BnnsTensor
  [^BnnsTensor x ^java.io.Writer w]
  (.write w (str x))
  (.write w " ")
  (if (contiguous? x)
    (print-method (doall (take (or *print-length* 16) (seq x))) w)
    (.write w "(... non-printable ...)")))

(defmethod transfer! [BnnsTensor BnnsTensor]
  [source destination]
  (if (and (equal-desc? source destination) (contiguous? source) (contiguous? destination))
    (copy! source destination)
    (with-release [transform! (transformer source destination)]
      (transform!)))
  destination)

(defmethod transfer! [Object BnnsTensor]
  [source destination]
  (if (contiguous? destination)
    (transfer! source (view-vctr destination))
    (with-release [connect (connector (default-desc destination) destination)]
      (transfer! source (view-vctr (input connect)))
      (connect)))
  destination)

(defmethod transfer! [BnnsTensor Object]
  [source destination]
  (if (contiguous? source)
    (transfer! (view-vctr source) destination)
    (with-release [connect (connector source (default-desc source))]
      (connect)
      (transfer! (view-vctr (output source)) destination)))
  (transfer! (view-vctr source) destination))

#_(defmethod transfer! [Object BnnsTransformer]
  [source destination]
  (transfer! source (input destination))
  destination)

#_(defmethod transfer! [BnnsTransformer Object]
  [source destination]
  (transfer! (output source) destination))

(prefer-method transfer! [BnnsTensor Object] [Object BnnsTensor])
(prefer-method transfer! [BnnsTensor BnnsTensor] [Object BnnsTensor])
(prefer-method transfer! [BnnsTensor BnnsTensor] [BnnsTensor Object])
