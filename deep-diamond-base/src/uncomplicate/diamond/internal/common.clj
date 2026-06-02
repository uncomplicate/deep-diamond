;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.diamond.internal.common
  (:require [uncomplicate.commons.core :refer [with-release]]
            [uncomplicate.neanderthal
             [core :refer [transfer! view-vctr]]
             [block :refer [contiguous?]]]
            [uncomplicate.diamond.tensor :refer [input output shape data-type default-desc connector]]))

(defn transfer-object-tensor [src dst]
  (if (= :half (data-type dst))
    (with-release [float-connect (connector (default-desc (shape dst) :float) dst)]
      (transfer! src (view-vctr (input float-connect)))
      (float-connect))
    (if (contiguous? dst)
      (transfer! src (view-vctr dst))
      (with-release [connect (connector (default-desc dst) dst)]
        (transfer! src (view-vctr (input connect)))
        (connect))))
  dst)

(defn transfer-tensor-object [src dst]
  [src dst]
  (if (= :half (data-type src))
    (with-release [float-connect (connector src (default-desc (shape src) :float))]
      (float-connect)
      (transfer! (view-vctr (output float-connect)) dst))
    (if (contiguous? src)
      (transfer! (view-vctr src) dst)
      (with-release [connect (connector src (default-desc dst))]
        (connect)
        (transfer! (view-vctr (output src)) dst))))
  dst)
