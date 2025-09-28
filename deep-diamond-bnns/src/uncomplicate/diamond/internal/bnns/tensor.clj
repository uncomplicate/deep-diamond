;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.diamond.internal.bnns.tensor
  (:require [uncomplicate.commons
             [core :refer [Releaseable release let-release with-release Info
                           info Viewable view size Bytes bytesize* bytesize sizeof]]
             [utils :refer [dragan-says-ex]]]
            [uncomplicate.fluokitten.protocols
             :refer [Magma Monoid Applicative Functor Foldable Comonad
                     extract fold foldmap]]
            [uncomplicate.clojure-cpp :refer
             [pointer safe null? get-entry zero! memcpy! capacity capacity! position!]]
            [uncomplicate.neanderthal
             [core :refer [transfer! copy! dim]]
             [block :refer [entry-width data-accessor buffer contiguous?]]]
            [uncomplicate.neanderthal.internal.api
             :refer [create-vector flow FactoryProvider EngineProvider
                     DataAccessorProvider Container raw zero MemoryContext
                     compatible? factory native-factory create-vector*
                     DenseContainer view-vctr]]
            [uncomplicate.neanderthal.internal.cpp.structures
             :refer [real-block-vector]]
            [uncomplicate.diamond.tensor
             :refer [TensorDescriptor shape TensorContainer Transfer input
                     output Revert ConnectorCreator connector view-tz
                     transformer batch-size default-desc]
             :as tz]
            [uncomplicate.diamond.internal
             [protocols
              :refer [TensorFactory DiamondFactoryProvider diamond-factory
                      create-tensor neanderthal-factory tensor-engine
                      native-diamond-factory Offset offset DiffTransfer
                      diff-input diff-output create-tensor-desc parameters
                      DescriptorProvider BatchDescriptor batch-index]]
             [utils :refer [check-contiguous default-strides]]]
            [uncomplicate.diamond.internal.bnns
             [core :as bnns
              :refer [equal-desc? compatible-desc? nda-desc dims data-type
                      layout strides rank data bnns-default-desc
                      apply-filter copy activation activation-params layer]]
             [constants :refer [bnns-data-type-pointer]]
             [protocols :refer [DescProvider desc data* clone*]]])
  (:import org.bytedeco.javacpp.Pointer
           [clojure.lang Seqable IFn AFn]
           [uncomplicate.neanderthal.internal.api Block VectorSpace Changeable
            Vector]
           uncomplicate.diamond.tensor.TensorDescriptorImpl
           [uncomplicate.diamond.internal.bnns.impl BnnsTensorDescriptorImpl
            BnnsNdArrayDescriptorImpl BnnsTensorImpl]))

(declare ->BnnsTensor ->BnnsShuffler
         bnns-tensor bnns-transformer bnns-batcher bnns-shuffler)

(extend-type java.util.Collection
  DescProvider
  (desc [this]
    (nda-desc (shape this))))

(extend-type java.lang.Number
  DescProvider
  (desc [this]
    (nda-desc [this] :float :x [1])))

(extend-type java.util.Map
  DescProvider
  (desc [this]
    (nda-desc (shape this) (or (:data-type this) :float) (tz/layout this))))

(extend-type Object
  DescProvider
  (desc [this]
    (nda-desc (shape this) (or (tz/data-type this) :float) (tz/layout this))))

(extend-type TensorDescriptorImpl
  DescProvider
  (desc [this]
    (nda-desc (.shape this) (or (.data-type this) :float) (tz/layout this))))

(extend-type BnnsTensorDescriptorImpl
  TensorDescriptor
  (shape [this]
    (dims this))
  (data-type [this]
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
            (let-release [in-tz (bnns-tensor fact (view in-desc) (batch-index out-tz))]
              (bnns-transformer in-tz (view out-tz)))))))))

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
  ConnectorCreator
  (connector [in-desc out]
    (if (equal-desc? in-desc (input out))
      (view out)
      (let [out-tz (output out)]
        (if (equal-desc? in-desc out-tz)
          (view out-tz)
          (let [fact (diamond-factory out-tz)]
            (let-release [in-tz (bnns-tensor fact (view in-desc) (batch-index out-tz))]
              (bnns-transformer in-tz (view out-tz)))))))))

(defmethod print-method BnnsNdArrayDescriptorImpl
  [^BnnsNdArrayDescriptorImpl d ^java.io.Writer w]
  (.write w (pr-str {:shape (tz/shape d) :data-type (tz/data-type d) :layout (tz/layout d)})))

;; =================== Transformer ==============================================

(deftype BnnsTransformer [reorder in-tz out-tz]
  Releaseable
  (release [_]
    (release reorder))
  Object
  (hashCode [_]
    (-> (hash :transformer)
        (hash-combine (shape in-tz))
        (hash-combine (shape out-tz))))
  (equals [this other]
    (or (identical? this other)
        (and (instance? BnnsTransformer other)
             (= (shape in-tz) (shape (.in-tz ^BnnsTransformer other)))
             (= out-tz (.out-tz ^BnnsTransformer other)))))
  (toString [this]
    (str {:input in-tz
          :output out-tz}))
  Revert
  (revert [_]
    (bnns-transformer (view out-tz) (view in-tz)))
  Viewable
  (view [_]
    (bnns-transformer (view in-tz) (view out-tz)))
  Transfer
  (input [_]
    in-tz)
  (output [_]
    out-tz)
  IFn
  (invoke [_]
    (if reorder
      (apply-filter reorder in-tz out-tz)
      (copy in-tz out-tz))
    out-tz)
  (invoke [this _]
    (.invoke this))
  (applyTo [this xs]
    (AFn/applyToHelper this xs))
  ConnectorCreator
  (connector [this out-desc]
    (if (equal-desc? out-tz out-desc)
      this
      (connector in-tz out-desc))))

;; =================== Batcher ==================================================

(deftype BnnsBatcher [reorder src-sub dst-sub src-tz dst-tz ^long mb-size
                      ^long src-cnt ^long src-stride-n
                      ^long dst-cnt ^long dst-stride-n]
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
  (equals [this other]
    (or (identical? this other)
        (and (instance? BnnsBatcher other)
             (= (shape dst-tz) (shape (.dst-tz ^BnnsBatcher other)))
             (= src-tz (.src-tz ^BnnsBatcher other)))))
  (toString [_]
    (str {:input src-tz
          :output dst-tz
          :mb-size mb-size}))
  Viewable
  (view [_]
    (bnns-batcher (view src-tz) (view dst-tz) mb-size))
  Transfer
  (input [_]
    src-tz)
  (output [_]
    dst-tz)
  IFn
  (invoke [this]
    (.invoke this 0 0))
  (invoke [this src-n]
    (.invoke this src-n 0))
  (invoke [_ src-n dst-n]
    (let [src-n (long src-n)
          dst-n (long dst-n)]
      (if (and (<= 0 src-n (- src-cnt mb-size)) (<= 0 dst-n (- dst-cnt mb-size)))
        (do (offset src-sub (* src-stride-n src-n))
            (offset dst-sub (* dst-stride-n dst-n))
            (if reorder
              (apply-filter reorder src-sub dst-sub)
              (copy src-sub dst-sub)))
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
      (connector dst-tz dst-desc))))

(let [activ (activation :identity)]

  (defn bnns-transformer [in-tz out-tz]
    (if (and (not (identical? in-tz out-tz))
             (equal-desc? in-tz out-tz))
      (->BnnsTransformer nil in-tz out-tz)
      (with-release [activ-params (activation-params activ in-tz out-tz)]
        (let-release [reorder (layer activ-params)]
          (if (null? reorder)
            (dragan-says-ex "BNNS cannot create a transformer for this combination of input and output."
                            {:in (info in-tz) :out (info out-tz)})
            (->BnnsTransformer reorder in-tz out-tz))))))

  (defn bnns-batcher [src-tz dst-tz mb-size]
    (let [mb-size (max 1 (long mb-size))]
      (let-release [src-sub (view-tz src-tz mb-size)
                    dst-sub (view-tz dst-tz mb-size)]
        (->BnnsBatcher (if (and (not (identical? src-tz dst-tz))
                                (equal-desc? src-sub dst-sub))
                         nil
                         (with-release [activ-params (activation-params activ src-sub dst-sub)]
                           (let-release [reorder (layer activ-params)]
                             (if (null? reorder)
                               (dragan-says-ex "BNNS cannot create a batcher for this combination of input and output."
                                               {:src (info src-tz) :dst (info dst-tz)})
                               reorder))))
                       src-sub dst-sub src-tz dst-tz mb-size
                       ((dims src-tz) (batch-index src-tz))
                       ((strides src-sub) (batch-index src-tz))
                       ((dims dst-tz) (batch-index dst-tz))
                       ((strides dst-sub) (batch-index dst-tz)))))))

(deftype BnnsShuffler [batcher batch-sizeoo mb-size]
  Releaseable
  (release [_]
    (release batcher))
  (hashCode [_]
    (hash-combine (hash :shuffler) (hash batcher)))
  (equals [this other]
    (or (identical? this other)
        (and (instance? BnnsShuffler other)
             (= batch-size (.batch-size ^BnnsShuffler other))
             (= mb-size (.mb-size ^BnnsShuffler other))
             (= batcher (.batcher ^BnnsShuffler other)))))
  (toString [this]
    (str {:input (input this)
          :output (output this)
          :mb-size mb-size}))
  Viewable
  (view [_]
    (->BnnsShuffler (view batcher) batch-size mb-size))
  Transfer
  (input [_]
    (input batcher))
  (output [_]
    (output batcher))
  IFn
  (invoke [_]
    (dotimes [i mb-size]
      (batcher (rand-int batch-size) i))
    (output batcher))
  (invoke [_ cols]
    (loop [src-n (first cols) cols (rest cols) dst-n 0]
      (when src-n
        (batcher src-n dst-n)
        (recur (first cols) (rest cols) (inc dst-n))))
    (output batcher))
  (applyTo [this xs]
    (AFn/applyToHelper this xs))
  ConnectorCreator
  (connector [this dst-desc]
    (if (equal-desc? (output batcher) dst-desc)
      this
      (connector batcher dst-desc))))

(defn bnns-shuffler [src-tz dst-tz]
  (->BnnsShuffler (bnns-batcher src-tz dst-tz 1)
                  (batch-size src-tz) (batch-size dst-tz)))

;; =================== Tensor =================================================

(deftype BnnsTensor [diamond-fact neand-fact eng master tz-desc vector-view
                     buf ^long nc ^long n-index]
  Object
  (hashCode [x]
    (-> (hash :BnnsTensor) (hash-combine (hash tz-desc))))
  (equals [x y]
    (or (identical? x y)
        (and (instance? BnnsTensor y)
             (equal-desc? tz-desc y)
             (= (layout tz-desc) (layout y))
             (if (and vector-view (.isContiguous ^BnnsTensor y))
               (= vector-view (view-vctr y))
               (= (data tz-desc) (data (desc y)))))))
  (toString [this]
    (pr-str {:shape (dims tz-desc) :data-type (data-type tz-desc) :layout (strides tz-desc)}))
  Info
  (info [x]
    {:data-type (data-type tz-desc)
     :class (class x)
     :device :cpu
     :rank (rank tz-desc)
     :shape (dims tz-desc)
     :layout (strides tz-desc)
     :master master
     :engine eng})
  (info [x info-type]
    (case info-type
      :data-type (data-type tz-desc)
      :class (class x)
      :device :cpu
      :rank (rank tz-desc)
      :shape (dims tz-desc)
      :layout (strides tz-desc)
      :master master
      :engine eng
      nil))
  Releaseable
  (release [_]
    (locking tz-desc
      (if master
        (release buf)
        (data* tz-desc nil))
      (release tz-desc))
    true)
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
    (create-tensor (diamond-factory fact)
                   (default-desc (dims tz-desc) (data-type tz-desc))
                   n-index false))
  (zero [x]
    (zero x diamond-fact))
  (zero [_ fact]
    (create-tensor (diamond-factory fact)
                   (default-desc (dims tz-desc) (data-type tz-desc))
                   n-index true))
  (host [x]
    (let-release [res (raw x)]
      (memcpy! (buffer x) (buffer res))))
  (native [x]
    x)
  Seqable
  (seq [this]
    (seq (view-vctr this)))
  MemoryContext
  (compatible? [_ y]
    (compatible? neand-fact (factory y)))
  (fits? [_ y]
    (= (dims tz-desc) (dims y)))
  (device [_]
    :cpu)
  Monoid
  (id [_]
    (with-release [ndims (rank tz-desc)
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
    (fold (view-vctr this)))
  (fold [this f init]
    (check-contiguous this)
    (fold (view-vctr this) f init))
  (fold [this f init y]
    (check-contiguous this y)
    (fold (view-vctr this) f init (view-vctr y)))
  (fold [this f init y z]
    (check-contiguous this y z)
    (fold (view-vctr this) f init (view-vctr y) (view-vctr z)))
  (fold [this f init y z v]
    (check-contiguous this y z v)
    (fold (view-vctr this) f init (view-vctr y) (view-vctr z) (view-vctr v)))
  (fold [this f init y z v ws]
    (check-contiguous this y z v)
    (doseq [w ws] (check-contiguous w))
    (fold (view-vctr this) f (view-vctr y) (view-vctr z) (view-vctr v) (map view-vctr ws)))
  (foldmap [this g]
    (check-contiguous this)
    (foldmap (view-vctr this) g))
  (foldmap [this g f init]
    (check-contiguous this)
    (foldmap (view-vctr this) g f init))
  (foldmap [this g f init y]
    (check-contiguous this y)
    (foldmap (view-vctr this) f init (view-vctr y)))
  (foldmap [this g f init y z]
    (check-contiguous this y z)
    (foldmap (view-vctr this) g f init (view-vctr y) (view-vctr z)))
  (foldmap [this g f init y z v]
    (check-contiguous this y z v)
    (foldmap (view-vctr this) g f init (view-vctr y) (view-vctr z) (view-vctr v)))
  (foldmap [this g f init y z v ws]
    (check-contiguous this y z v)
    (doseq [w ws] (check-contiguous w))
    (foldmap (view-vctr this) g f (view-vctr y) (view-vctr z)
             (view-vctr v) (map view-vctr ws)))
  Applicative
  (pure [_ v]
    (with-release [ndims (rank tz-desc)
                   md (nda-desc (repeat ndims 1)
                                (data-type tz-desc)
                                (repeat ndims 1))]
      (let-release [res (bnns-tensor diamond-fact md n-index)]
        (zero! (buffer res))
        res)))
  (pure [_ v vs]
    (let [vs (cons v vs)]
      (with-release [ndims (rank tz-desc)
                     md (nda-desc (cons (count vs) (repeat (dec ndims) 1))
                                  (data-type tz-desc) (repeat ndims 1))]
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
    (capacity! (data tz-desc) (capacity buf)))
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
    (let [strd (strides tz-desc)]
      (if (or (some zero? strd) (some zero? (dims tz-desc)))
        (bnns/layout tz-desc)
        strd)))
  BatchDescriptor
  (batch-index [_]
    n-index)
  Viewable
  (view [this]
    (let-release [tz-desc-clone (clone* tz-desc)]
      (data* tz-desc-clone (data* tz-desc))
      (->BnnsTensor diamond-fact neand-fact eng false tz-desc-clone vector-view
                    buf nc n-index)))
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
                                 (nda-desc (shape sub) dtype sub-layout (strides tz-desc))
                                 (if (seq sub-layout)
                                   (nda-desc (shape sub) dtype (layout tz-desc) sub-layout)
                                   (nda-desc (shape sub) dtype (layout tz-desc))))))]
      (bnns-tensor diamond-fact sub-desc buf n-index false)))
  Offset
  (offset [this ofst]
    (if (<= 0 (long ofst) (capacity buf))
      (do (position! buf ofst)
          (data* tz-desc buf)
          (when vector-view (position! (buffer vector-view) ofst)))
      (dragan-says-ex "There isn't enough capacity in the underlying buffer for this offset."
                      {:requested ofst :available (size tz-desc)}))
    this)
  ConnectorCreator
  (connector [in-tz out-desc]
    (if (equal-desc? tz-desc out-desc)
      (view in-tz)
      (let-release [out-tz (bnns-tensor diamond-fact (view (desc out-desc))
                                        (batch-index in-tz))]
        (bnns-transformer (view in-tz) out-tz)))))

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
                                     (create-vector neand-fact false buf nc 0 1)
                                     nil)]
             (data* tdesc buf)
             (->BnnsTensor diamond-fact neand-fact
                           (tensor-engine diamond-fact dtype)
                           master tdesc vect-view
                           buf nc n-index))
           (throw (dragan-says-ex "Insufficient buffer size."
                                  {:desc-size (bytesize tdesc) :buffer-size (bytesize buf)}))))
       (throw (dragan-says-ex "We cannot overwrite NDA descriptor's existing data pointer! Please provide a fresh NDA descriptor."
                              {:nda-desc tdesc})))))
  ([diamond-fact tdesc buf master]
   (bnns-tensor diamond-fact tdesc buf 0 master))
  ([diamond-fact tdesc n-index]
   (let [tdesc (desc tdesc)]
     (layout tdesc)
     (strides tdesc)
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
  [src dst]
  (if (and (equal-desc? src dst) (contiguous? src) (contiguous? dst))
    (copy! src dst)
    (with-release [transform! (transformer src dst)]
      (transform!)))
  dst)

(defmethod transfer! [Object BnnsTensor]
  [src dst]
  (if (contiguous? dst)
    (transfer! src (view-vctr dst))
    (with-release [connect (connector (bnns-default-desc dst) dst)]
      (transfer! src (view-vctr (input connect)))
      (connect)))
  dst)

(defmethod transfer! [BnnsTensor Object]
  [src dst]
  (if (contiguous? src)
    (transfer! (view-vctr src) dst)
    (with-release [connect (connector src (bnns-default-desc src))]
      (connect)
      (transfer! (view-vctr (output connect)) dst))))

(defmethod transfer! [Object BnnsTransformer]
  [src dst]
  (transfer! src (input dst))
  dst)

(defmethod transfer! [BnnsTransformer Object]
  [source destination]
  (transfer! (output source) destination))

(prefer-method transfer! [BnnsTensor Object] [Object BnnsTensor])
(prefer-method transfer! [BnnsTensor BnnsTensor] [Object BnnsTensor])
(prefer-method transfer! [BnnsTensor BnnsTensor] [BnnsTensor Object])
