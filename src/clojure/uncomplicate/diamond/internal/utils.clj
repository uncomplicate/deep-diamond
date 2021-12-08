;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.diamond.internal.utils
  (:require [uncomplicate.commons
             [core :refer [Wrapper Releaseable extract]]
             [utils :refer [dragan-says-ex with-check]]]
            [uncomplicate.neanderthal.core :refer [transfer!]]
            [uncomplicate.diamond.internal.protocols :refer [weights bias]])
  (:import uncomplicate.neanderthal.internal.api.Block))

(defmacro deftype-wrapper [name release-method error]
  (let [name-str (str name)]
    `(deftype ~name [ref#]
       Object
       (hashCode [this#]
         (hash (deref ref#)))
       (equals [this# other#]
         (= (deref ref#) (extract other#)))
       (toString [this#]
         (format "#%s[%s]" ~name-str (deref ref#)))
       Wrapper
       (extract [this#]
         (deref ref#))
       Releaseable
       (release [this#]
         (locking ref#
           (when-let [d# (deref ref#)]
             (locking d#
               (with-check ~error (~release-method d#) (vreset! ref# nil)))))
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

(defn concat-strides [split-dim src-shape sub-shapes]
  (let [stride-vec (vec (repeat (count src-shape) 0))]
    (loop [strd 0 strds [] sub-shapes sub-shapes]
      (if sub-shapes
        (recur (+ strd (long (get (first sub-shapes) split-dim)))
               (conj strds (assoc stride-vec split-dim strd))
               (next sub-shapes))
        strds))))
