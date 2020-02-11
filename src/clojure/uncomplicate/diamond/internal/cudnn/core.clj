;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.diamond.internal.cudnn.core
  (:require [uncomplicate.commons
             [core :refer [let-release with-release wrap extract]]
             [utils :refer [dragan-says-ex enc-keyword]]]
            [uncomplicate.clojurecuda.internal.protocols :refer [ptr with-offset]]
            [uncomplicate.neanderthal.block :refer [buffer]]
            [uncomplicate.diamond.internal.cudnn
             [protocols :refer :all]
             [constants :refer :all]
             [impl :refer :all]])
  (:import java.lang.Exception
           [jcuda.jcudnn JCudnn cudnnNanPropagation]
           uncomplicate.diamond.internal.cudnn.impl.CUTensorDescriptor))

(defn cudnn-handle [stream]
  (wrap (cudnn-handle* (extract stream))))

(defn get-cudnn-stream [handle]
  (wrap (get-cudnn-stream* (extract handle))))

(defn tensor-descriptor [shape data-type layout]
  (let [d (count shape)
        dtype (enc-keyword cudnn-data-type data-type)]
    (let-release [td (wrap (tensor-descriptor*))]
      (if (keyword? layout)
        (let [format (enc-keyword cudnn-format layout)]
          (if (< 4 d)
            (tensor-4d-descriptor* (extract td) format dtype shape)
            (tensor-nd-descriptor-ex* (extract td) format dtype (int-array shape))))
        (if (= d (count layout))
          (if (< 4 d)
            (tensor-4d-descriptor-ex* (extract td) dtype shape layout)
            (tensor-nd-descriptor* (extract td) dtype (int-array shape) (int-array layout)))
          (dragan-says-ex "Shape and strides must have the same length."
                          {:shape shape :strides layout})))
      td)))

(defn equal-desc? [td1 td2]
  (let [td1 (desc td1)
        td2 (desc td2)]
    (and (instance? CUTensorDescriptor td1) (instance? CUTensorDescriptor td2)
         (let [td1 ^CUTensorDescriptor td1
               td2 ^CUTensorDescriptor td2]
           (and (= (.dims td1) (.dims td2)) (= (.data-type td1) (.data-type td2))
                (= (.strides td1) (.strides td2)))))))

(defn data-type
  "Returns the data type of a tensor descriptor."
  [td]
  (.data-type ^CUTensorDescriptor (desc td)))

(defn dims
  "Returns the dimensions of a tensor descriptor."
  [td]
  (.dims ^CUTensorDescriptor (desc td)))

(defn ndims
  "Returns the number of dimensions of a tensor descriptor."
  ^long [td]
  (count (dims td)))

(defn size
  "Returns the size of a tensor descriptor."
  ^long [td]
  (size* (extract td)))

(defn strides
  "Queries the strides of a tensor descriptor."
  [td]
  (.strides ^CUTensorDescriptor (desc td)))

(defn set-tensor
  "TODO: attention: primitive numbers need explicit casting to double, float etc."
  ([cudnn-handle desc-x buf-x ofst-x value]
   (set-tensor* (extract cudnn-handle)
                (extract (desc desc-x)) (with-offset buf-x ofst-x) (ptr value))
   cudnn-handle)
  ([cudnn-handle desc-x buf-x value]
   (set-tensor* (extract cudnn-handle) (extract (desc desc-x)) (extract buf-x) (ptr value))
   cudnn-handle))

(defn scale-tensor
  "TODO: attention: primitive numbers need explicit casting to double, float etc."
  ([cudnn-handle alpha desc-x buf-x ofst-x]
   (scale-tensor* (extract cudnn-handle)
                  (extract (desc desc-x)) (with-offset buf-x ofst-x) (ptr alpha))
   cudnn-handle)
  ([cudnn-handle alpha desc-x buf-x]
   (scale-tensor* (extract cudnn-handle) (extract (desc desc-x)) (extract buf-x) (ptr alpha))
   cudnn-handle))

(defn add-tensor
  "TODO: attention: primitive numbers need explicit casting to double, float etc."
  ([cudnn-handle alpha desc-x buf-x ofst-x beta desc-y buf-y ofst-y]
   (add-tensor* (extract cudnn-handle)
                (ptr alpha) (extract (desc desc-x)) (with-offset buf-x ofst-x)
                (ptr beta) (extract (desc desc-y)) (with-offset buf-y ofst-y))
   cudnn-handle)
  ([cudnn-handle alpha desc-x buf-x beta desc-y buf-y]
   (add-tensor* (extract cudnn-handle)
                (ptr alpha) (extract (desc desc-x)) (extract buf-x)
                (ptr beta) (extract (desc desc-y)) (extract buf-y))
   cudnn-handle))

(defn transform-tensor
  ([cudnn-handle alpha desc-x buf-x ofst-x beta desc-y buf-y ofst-y]
   (transform-tensor* (extract cudnn-handle)
                      (ptr alpha) (extract (desc desc-x)) (with-offset buf-x ofst-x)
                      (ptr beta) (extract (desc desc-y)) (with-offset buf-y ofst-y))
   cudnn-handle)
  ([cudnn-handle alpha desc-x buf-x beta desc-y buf-y]
   (transform-tensor* (extract cudnn-handle)
                      (ptr alpha) (extract (desc desc-x)) (extract buf-x)
                      (ptr beta) (extract (desc desc-y)) (extract buf-y))
   cudnn-handle))

;; =========================== Activation ============================================

(defn activation-descriptor [mode relu-nan-opt ^double coef]
  (let-release [ad (wrap (activation-descriptor*))]
    (activation-descriptor* (extract ad) (enc-keyword cudnn-activation-mode mode)
                            (if relu-nan-opt
                              cudnnNanPropagation/CUDNN_PROPAGATE_NAN
                              cudnnNanPropagation/CUDNN_NOT_PROPAGATE_NAN)
                            coef)
    ad))

(defn get-activation-descriptor [ad]
  (let [mode (int-array 1)
        relu-nan-opt (int-array 1)
        coef (double-array 1)]
    (get-activation-descriptor* (extract ad) mode relu-nan-opt coef)
    {:mode (dec-activation-mode (aget mode 0))
     :relu-nan-opt (if (= (aget relu-nan-opt 0) cudnnNanPropagation/CUDNN_PROPAGATE_NAN) true false)
     :coef (aget coef 0)}))

(defn activation-forward [cudnn-handle ad alpha desc-x buf-x beta desc-y buf-y]
  (activation-forward* (extract cudnn-handle) (extract ad)
                       (ptr alpha) (extract (desc desc-x)) (extract buf-x)
                       (ptr beta) (extract (desc desc-y)) (extract buf-y))
  cudnn-handle)

(defn activation-backward
  "TODO: why is y even needed?"
  [cudnn-handle ad
                           alpha desc-y buf-y desc-dy buf-dy
                           desc-x buf-x beta desc-dx buf-dx]
  (activation-backward* (extract cudnn-handle) (extract ad)
                        (ptr alpha) (extract (desc desc-y)) (extract buf-y)
                        (extract (desc desc-dy)) (extract buf-dy)
                        (extract (desc desc-x)) (extract buf-x)
                        (ptr beta) (extract (desc desc-dx)) (extract buf-dx))
  cudnn-handle)
