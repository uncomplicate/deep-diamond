;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.diamond.internal.utils
  (:require [uncomplicate.commons
             [core :refer [Wrapper Releaseable with-release extract]]
             [utils :refer [dragan-says-ex with-check]]]
            [uncomplicate.fluokitten.core :refer [foldmap]]
            [uncomplicate.clojure-cpp :refer [null?]]
            [uncomplicate.neanderthal.core :refer [transfer! axpy entry!]]
            [uncomplicate.diamond.internal.protocols
             :refer [weights bias weights-layer weights-iter bias-layer bias-iter]])
  (:import uncomplicate.neanderthal.internal.api.Block))

(defmacro extend-pointer [name release-method error]
  (let [name-str (str name)]
    `(extend-type ~name
       Releaseable
       (release [this#]
         (locking this#
           (when-not (null? this#)
             (with-check ~error (~release-method this#)
               (do (.deallocate this#)
                   (.setNull this#)))))
         true))))

(defn check-contiguous
  ([^Block x]
   (when-not (.isContiguous x)
     (dragan-says-ex "This operation is supported only on contiguous tensors.
Please use a copy or create a transformer."
                     {:x (str x)})))
  ([^Block x ^Block y]
   (check-contiguous x)
   (check-contiguous y))
  ([^Block x ^Block y ^Block z]
   (check-contiguous x)
   (check-contiguous y)
   (check-contiguous z)))

(defn default-strides [shape]
  (let [cnt (count shape)]
    (case cnt
      0 []
      1 [1]
      (let [res (long-array cnt)]
        (aset res (dec cnt) 1)
        (loop [res res i (dec cnt)]
          (if (< 0 i)
            (do (aset res (dec i) (* (aget res i) (long (get shape i))))
                (recur res  (dec i)))
            (vec res)))))))

(defn transfer-weights-bias! [source destination]
  (transfer! (bias source) (bias destination))
  (transfer! (weights source) (weights destination))
  destination)

(defn transfer-rnn-weights-bias! [source destination]
  (if (bias-iter source)
    (if (bias-iter destination)
      (do (transfer! (bias-layer source) (bias-layer destination))
          (transfer! (bias-iter source) (bias-iter destination)))
      (with-release [bias-sum (axpy 1.0 (bias-layer source) (bias-iter source))]
        (transfer! bias-sum (bias-layer destination))))
    (do (transfer! (bias-layer source) (bias-layer destination))
        (when (bias-iter destination)
          (entry! (bias-iter destination) 0.0))))
  (transfer! (weights-layer source) (weights-layer destination))
  (transfer! (weights-iter source) (weights-iter destination))
  destination)

(defn concat-offsets [split-dim sub-shapes]
  (pop (reduce (fn [acc sub-shape]
                 (conj acc (+ (long (peek acc)) (long (get sub-shape split-dim)))))
               [0]
               sub-shapes)))

(defn concat-strides [split-dim sub-shapes]
  (let [stride-vec (vec (repeat (count (first sub-shapes)) 0))]
    (map (partial assoc stride-vec split-dim) (concat-offsets split-dim sub-shapes))))

(defn concat-dst-shape [conc-dim src-shapes]
  (assoc (get src-shapes 0) conc-dim
         (foldmap + #(get % conc-dim) src-shapes)))

(defn direction-count ^long [direction]
  (case direction
    :bidirectional-concat 2
    :bidirectional-sum 2
    1))
