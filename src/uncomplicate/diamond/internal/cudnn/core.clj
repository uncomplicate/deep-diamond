;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.diamond.internal.cudnn.core
  (:require [uncomplicate.commons
             [core :refer [let-release wrap extract]]
             [utils :refer [dragan-says-ex with-check enc-keyword]]]
            [uncomplicate.clojurecuda.internal.protocols :refer [ptr]]
            [uncomplicate.neanderthal.block :refer [buffer]]
            [uncomplicate.diamond.internal.cudnn
             [protocols :refer :all]
             [constants :refer :all]
             [impl :refer :all]])
  (:import [jcuda.jcudnn JCudnn]))

(defn cudnn-handle [stream]
  (wrap (cudnn-handle* (extract stream))))

(defn get-cudnn-stream [handle]
  (wrap (get-cudnn-stream* (extract handle))))

(defn tensor-descriptor []
  (wrap (tensor-descriptor*)))

(defn tensor-4d-descriptor [shape data-type format]
  (if (= 4 (count shape))
    (let-release [td (wrap (tensor-descriptor*))]
      (tensor-4d-descriptor* (extract td) (int-array shape)
                             (enc-keyword cudnn-data-type data-type)
                             (enc-keyword cudnn-format format))
      td)
    (dragan-says-ex "The shape of a 4d tensor descriptor must have 4 entries"
                    {:shape shape})))

(defn add-tensor [cudnn-handle alpha a beta b]
  (with-check cudnn-error
    (JCudnn/cudnnAddTensor cudnn-handle (ptr alpha) (desc a) (extract (buffer a))
                           (ptr beta) (desc b) (extract (buffer b)))
    b))
