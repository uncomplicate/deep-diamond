;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.diamond.internal.utils
  (:require [uncomplicate.commons.utils :refer [dragan-says-ex]]
            [uncomplicate.diamond.tensor :refer [layout]])
  (:import uncomplicate.neanderthal.internal.api.Block))

(defn check-contiguous
  ([^Block x]
   (when-not (.isContiguous x)
     (dragan-says-ex "This operation is supported only on contiguous tensors.
Please use a copy or create a transformer."
                     {:strides (layout x)})))
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
