;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.diamond.internal.neanderthal.tensor
  (:require [uncomplicate.commons ;;TODO clean up requires
             [core :refer [let-release with-release]]
             [utils :refer [dragan-says-ex]]]
            [uncomplicate.neanderthal
             [block :refer [column? entry-type stride data-accessor buffer]]]
            [uncomplicate.neanderthal.internal.api
             :refer [factory native-factory create-vector create-ge]]
            [uncomplicate.diamond.tensor
             :refer [TensorDescriptor shape data-type layout TensorContainer Transfer
                     Revert ConnectorCreator]]
            [uncomplicate.diamond.internal.protocols :refer [DiffTransfer DescriptorProvider BatchDescriptor]]
            [uncomplicate.diamond.internal.neanderthal.constants :refer [data-type-dec]])
  (:import [uncomplicate.neanderthal.internal.api VectorSpace Vector GEMatrix]))

;; ================================ Tensor ======================================

(defn equal-desc? [td1 td2]
  (and (= (shape td1) (shape td2))
       (= (data-type td1) (data-type td2))
       (= (layout td1) (layout td2))))

(extend-type VectorSpace
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
  DescriptorProvider
  (inf-desc [this]
    this)
  (train-desc [this]
    this)
  (diff-desc [this]
    this)
  BatchDescriptor
  (batch-index [_]
    0)
  TensorContainer
  (view-tz [this]
    this))

(extend-type Vector
  TensorDescriptor
  (shape [this]
    [(.dim this)])
  (data-type [this]
    (data-type-dec (entry-type (data-accessor this))))
  (layout [this]
    [(stride this)]))

(extend-type GEMatrix
  TensorDescriptor
  (shape [this]
    [(.mrows this) (.ncols this)])
  (data-type [this]
    (data-type-dec (entry-type (data-accessor this))))
  (layout [this]
    (if (column? this)
      [1 (stride this)]
      [(stride this) 1])))

(defn neanderthal-tensor
  ([neand-fact shape layout init]
   (let [fact (factory neand-fact)]
     (let [[m n k] shape]
       (cond k (dragan-says-ex "Matrices can't support more than 2 dimensions." {:requested shape})
             n (cond
                 (or (not layout) (#{:nc :ab :oi} layout)) (create-ge neand-fact m n false init)
                 (#{:cn :ba :io} layout) (create-ge neand-fact m n true init)
                 :default
                 (let [[sm sn] layout]
                   (cond
                     (= 1 sn) (create-ge neand-fact m n false init)
                     (= m sn) (create-ge neand-fact m n true init)
                     :default (dragan-says-ex "Matrices do not support non-dense tensor strides." {:requested sn :required m}))))
             m (if (or (not layout) (#{:x :a [1]} layout))
                 (create-vector neand-fact (get shape 0) init)
                 (dragan-says-ex "Vector as a tensor should not be strided." {:requested shape}))
             :default (dragan-says-ex "Vectors can't support 0 dimensions." {:requested shape})))))
  ([neand-fact shape init]
   (case (count shape)
     1 (create-vector neand-fact (get shape 0) init)
     2 (create-ge neand-fact (get shape 0) (get shape 1) false init)
     (dragan-says-ex "Vectors and matrices only support 1 or 2 dimensions." {:requested shape})))
  ([neand-fact shape]
   (neanderthal-tensor neand-fact shape true)))
