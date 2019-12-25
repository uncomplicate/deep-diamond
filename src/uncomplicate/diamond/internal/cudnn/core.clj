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
  (:import java.lang.Exception
           jcuda.jcudnn.JCudnn
           uncomplicate.diamond.internal.cudnn.impl.CUTensorDescriptor))

(defn cudnn-handle [stream]
  (wrap (cudnn-handle* (extract stream))))

(defn get-cudnn-stream [handle]
  (wrap (get-cudnn-stream* (extract handle))))

(defn tensor-descriptor [shape data-type layout]
  (let [d (count shape)
        dtype (enc-keyword cudnn-data-type data-type)]
    (let [td (tensor-descriptor*)]
      (try
        (wrap (if (keyword? layout)
                (let [format (enc-keyword cudnn-format layout)]
                  (if (< 4 d)
                    (tensor-4d-descriptor* td format dtype shape)
                    (tensor-nd-descriptor-ex* td format dtype (int-array shape))))
                (if (= d (count layout))
                  (if (< 4 d)
                    (tensor-4d-descriptor-ex* td dtype shape layout)
                    (tensor-nd-descriptor* td dtype (int-array shape) (int-array layout)))
                  (dragan-says-ex "Shape and strides must have the same length."
                                  {:shape shape :strides layout}))))
        (catch Exception e
          (with-check cudnn-error
            (JCudnn/cudnnDestroyTensorDescriptor td)
            (throw e)))))))

(defn equal-desc? [td1 td2]
  (and (instance? CUTensorDescriptor td1) (instance? CUTensorDescriptor td2)
       (let [td1 ^CUTensorDescriptor td1
             td2 ^CUTensorDescriptor td2]
         (and (= (.dims td1) (.dims td2)) (= (.data-type td1) (.data-type td2))
              (= (.strides td1) (.strides td2))))))

(defn size
  "Queries the tensor descriptor for its dimensions."
  ^long [td]
  (size* (extract td)))

(defn add-tensor [cudnn-handle alpha a beta b]
  (with-check cudnn-error
    (JCudnn/cudnnAddTensor cudnn-handle (ptr alpha) (desc a) (extract (buffer a))
                           (ptr beta) (desc b) (extract (buffer b)))
    b))
