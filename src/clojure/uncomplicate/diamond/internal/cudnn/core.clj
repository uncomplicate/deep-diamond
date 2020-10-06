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
            [uncomplicate.clojurecuda.core :refer [mem-alloc]]
            [uncomplicate.clojurecuda.internal.protocols
             :as cuda
             :refer [ptr with-offset]]
            [uncomplicate.neanderthal.block :refer [buffer]]
            [uncomplicate.diamond.internal.cudnn
             [protocols :refer :all]
             [constants :refer :all]
             [impl :refer :all]])
  (:import java.lang.Exception
           [jcuda.jcudnn JCudnn cudnnConvolutionFwdAlgoPerf cudnnConvolutionBwdDataAlgoPerf
            cudnnConvolutionBwdFilterAlgoPerf]
           [uncomplicate.diamond.internal.cudnn.impl CUTensorDescriptor CUFilterDescriptor]))

(defprotocol AlgoPerf
  (algo [this])
  (workspace-size [this])
  (status [this])
  (algo-time [this])
  (determinism [this])
  (math-type [this]))

(defn cudnn-handle [stream]
  (wrap (cudnn-handle* (extract stream))))

(defn get-cudnn-stream [handle]
  (wrap (get-cudnn-stream* (extract handle))))

(defn tensor-descriptor [shape data-type layout]
  (let [d (count shape)
        dtype (enc-keyword cudnn-data-type data-type)]
    (let [td (tensor-descriptor*)]
      (try
        (wrap
         (if (keyword? layout)
           (let [format (enc-keyword cudnn-format layout)]
             (if (< 4 d)
               (tensor-nd-descriptor-ex* td format dtype (int-array shape))
               (tensor-4d-descriptor* td format dtype shape)))
           (if (= d (count layout))
             (if (< 4 d)
               (tensor-nd-descriptor* td dtype (int-array shape) (int-array layout))
               (tensor-4d-descriptor-ex* td dtype shape layout))
             (dragan-says-ex "Shape and strides must have the same length."
                             {:shape shape :strides layout}))))
        (catch Exception e
          (with-check (JCudnn/cudnnDestroyTensorDescriptor td)
            (throw e)))))))

(defn equal-desc? [td1 td2]
  (let [td1 (desc td1)
        td2 (desc td2)]
    (or (= td1 td2)
        (and (instance? CUTensorDescriptor td1) (instance? CUTensorDescriptor td2)
             (let [td1 ^CUTensorDescriptor td1
                   td2 ^CUTensorDescriptor td2]
               (and (= (.dims td1) (.dims td2)) (= (.data-type td1) (.data-type td2))
                    (= (.strides td1) (.strides td2)))))
        (and (instance? CUFilterDescriptor td1) (instance? CUFilterDescriptor td2)
             (let [td1 ^CUFilterDescriptor td1
                   td2 ^CUFilterDescriptor td2]
               (and (= (.dims td1) (.dims td2)) (= (.data-type td1) (.data-type td2))
                    (= (.format td1) (.format td2))))))))

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

(defn activation-descriptor
  ([mode relu-nan-opt ^double coef]
   (let-release [ad (wrap (activation-descriptor*))]
     (activation-descriptor* (extract ad) (enc-keyword cudnn-activation-mode mode)
                             (enc-nan-propagation relu-nan-opt)
                             coef)
     ad))
  ([mode ^double coef]
   (activation-descriptor mode true coef)))

(defn get-activation-descriptor [ad]
  (let [mode (int-array 1)
        relu-nan-opt (int-array 1)
        coef (double-array 1)]
    (get-activation-descriptor* (extract ad) mode relu-nan-opt coef)
    {:mode (dec-activation-mode (aget mode 0))
     :relu-nan-opt (dec-nan-propagation (aget relu-nan-opt 0))
     :coef (aget coef 0)}))

(defn activation-forward [cudnn-handle ad alpha desc-x buf-x beta desc-y buf-y]
  (activation-forward* (extract cudnn-handle) (extract ad)
                       (ptr alpha) (extract (desc desc-x)) (extract buf-x)
                       (ptr beta) (extract (desc desc-y)) (extract buf-y))
  cudnn-handle)

(defn activation-backward [cudnn-handle ad
                           alpha desc-y buf-y desc-dy buf-dy
                           desc-x buf-x beta desc-dx buf-dx]
  (activation-backward* (extract cudnn-handle) (extract ad)
                        (ptr alpha) (extract (desc desc-y)) (extract buf-y)
                        (extract (desc desc-dy)) (extract buf-dy)
                        (extract (desc desc-x)) (extract buf-x)
                        (ptr beta) (extract (desc desc-dx)) (extract buf-dx))
  cudnn-handle)

;; ============================ Reduce ==============================================

(defn reduce-tensor-descriptor
  ([op comp-type nan-opt indices]
   (let-release [rtd (wrap (reduce-tensor-descriptor*))]
     (reduce-tensor-descriptor* (extract rtd) (enc-keyword cudnn-reduce-tensor-op op)
                                (enc-keyword cudnn-data-type comp-type)
                                (enc-nan-propagation nan-opt)
                                (enc-keyword cudnn-reduce-tensor-indices indices))
     rtd))
  ([op comp-type nan-opt]
   (let-release [rtd (wrap (reduce-tensor-descriptor*))]
     (reduce-tensor-descriptor* (extract rtd) (enc-keyword cudnn-reduce-tensor-op op)
                                (enc-keyword cudnn-data-type comp-type)
                                (enc-nan-propagation nan-opt))
     rtd))
  ([op comp-type]
   (reduce-tensor-descriptor op comp-type true)))

(defn reduction-indices-size ^long [cudnn-handle rtd desc-x desc-y]
  (reduction-indices-size* (extract cudnn-handle) (extract rtd)
                           (extract (desc desc-x)) (extract (desc desc-y))))

(defn reduction-workspace-size ^long [cudnn-handle rtd desc-x desc-y]
  (reduction-workspace-size* (extract cudnn-handle) (extract rtd)
                             (extract (desc desc-x)) (extract (desc desc-y))))

(defn reduce-tensor
  ([cudnn-handle rtd indices workspace alpha desc-x buf-x beta desc-y buf-y]
   (reduce-tensor* (extract cudnn-handle) (extract rtd)
                   (extract indices) (cuda/size indices)
                   (extract workspace) (cuda/size workspace)
                   (ptr alpha) (extract (desc desc-x)) (extract buf-x)
                   (ptr beta) (extract (desc desc-y)) (extract buf-y))
   cudnn-handle)
  ([cudnn-handle rtd alpha desc-x buf-x beta desc-y buf-y]
   (let [indices-size (reduction-indices-size cudnn-handle rtd desc-x desc-y)
         workspace-size (reduction-workspace-size cudnn-handle rtd desc-x desc-y)]
     (let-release [indices (mem-alloc (max 1 indices-size))
                   workspace (mem-alloc (max 1 workspace-size))]
       (reduce-tensor cudnn-handle rtd indices workspace
                      alpha desc-x buf-x beta desc-y buf-y)))))

;; =========================== Softmax ============================================

(defn softmax-forward [cudnn-handle algo mode alpha desc-x buf-x beta desc-y buf-y]
  (softmax-forward* (extract cudnn-handle) (enc-keyword cudnn-softmax-algorithm algo)
                    (enc-keyword cudnn-softmax-mode mode)
                    (ptr alpha) (extract (desc desc-x)) (extract buf-x)
                    (ptr beta) (extract (desc desc-y)) (extract buf-y))
  cudnn-handle)

(defn softmax-backward [cudnn-handle algo mode
                        alpha desc-y buf-y desc-dy buf-dy
                        beta desc-dx buf-dx]
  (softmax-backward* (extract cudnn-handle)  (enc-keyword cudnn-softmax-algorithm algo)
                     (enc-keyword cudnn-softmax-mode mode)
                     (ptr alpha) (extract (desc desc-y)) (extract buf-y)
                     (extract (desc desc-dy)) (extract buf-dy)
                     (ptr beta) (extract (desc desc-dx)) (extract buf-dx))
  cudnn-handle)

;; ============================ Filter =============================================

(defn filter-descriptor [shape data-type format]
  (let [d (count shape)
        dtype (enc-keyword cudnn-data-type data-type)
        format (enc-keyword cudnn-format format)]
    (let [fd (filter-descriptor*)]
      (try
        (wrap
         (if (< 4 d)
           (filter-nd-descriptor* fd dtype format (int-array shape))
           (filter-4d-descriptor* fd dtype format shape)))
        (catch Exception e
          (with-check (JCudnn/cudnnDestroyFilterDescriptor fd)
            (throw e)))))))

(defn extract-filter [ ^CUFilterDescriptor filter-desc]
  (deref (.fd filter-desc)))

;; ============================ Convolution ========================================

(defn convolution-descriptor [mode data-type pad stride dilation]
  (let-release [cd (wrap (convolution-descriptor*))]
    (let [mode (enc-keyword cudnn-convolution-mode mode)
          dtype (enc-keyword cudnn-data-type data-type)]
      (try
        (wrap
         (if (< 2 (count pad))
           (convolution-nd-descriptor* (extract cd) (int-array pad) (int-array stride)
                                       (int-array dilation) mode dtype)
           (convolution-2d-descriptor* (extract cd) pad stride dilation mode dtype)))))
    cd))

(extend-type cudnnConvolutionFwdAlgoPerf
  AlgoPerf
  (algo [this]
    (dec-convolution-fwd-algo (.algo this)))
  (workspace-size [this]
    (.memory this))
  (status [this]
    (.status this))
  (algo-time [this]
    (.time this))
  (determinism [this]
    (dec-determinism (.determinism this)))
  (math-type [this]
    (dec-math-type (.mathType this))))

(defn algo-perf [cudnn-algo-perf]
  {:algo (algo cudnn-algo-perf)
   :workspace-size (workspace-size cudnn-algo-perf)
   :status (status cudnn-algo-perf)
   :algo-time (algo-time cudnn-algo-perf)
   :determinism (determinism cudnn-algo-perf)
   :math-type (math-type cudnn-algo-perf)})

(defn convolution-fwd-find-algo
  ([cudnn-handle cd desc-x desc-w desc-y algo-count]
   (map algo-perf
        (convolution-fwd-find-algo* (extract cudnn-handle) (extract cd) (extract (desc desc-x))
                                    (extract-filter desc-w) (extract (desc desc-y))
                                    algo-count)))
  ([cudnn-handle cd desc-x desc-w desc-y]
   (first (convolution-fwd-find-algo cudnn-handle cd desc-x desc-w desc-y 1))))

(defn convolution-fwd
  "TODO
  The :convolution algorithm uses flipped kernels as real convolution from the books.
  To match DNNL (more practical), use :cross-correlation."
  ([cudnn-handle cd algo alpha desc-x buf-x desc-w buf-w beta desc-y buf-y workspace]
   (convolution-fwd* (extract cudnn-handle) (extract cd)
                     (enc-keyword cudnn-convolution-fwd-algo algo)
                     (ptr alpha) (extract (desc desc-x)) (extract buf-x)
                     (extract-filter desc-w) (extract buf-w)
                     (ptr beta) (extract (desc desc-y)) (extract buf-y)
                     (extract workspace) (if workspace (cuda/size workspace) 0))

   cudnn-handle)
  ([cudnn-handle cd algo ad alpha1 desc-x buf-x desc-w buf-w alpha2 buf-z
    desc-bias buf-bias desc-y buf-y workspace]
   (convolution-fwd* (extract cudnn-handle) (extract cd)
                     (enc-keyword cudnn-convolution-fwd-algo algo) (extract ad)
                     (ptr alpha1) (extract (desc desc-x)) (extract buf-x)
                     (extract-filter desc-w) (extract buf-w)
                     (ptr alpha2) (extract buf-z)
                     (extract (desc desc-bias)) (extract buf-bias)
                     (extract (desc desc-y)) (extract buf-y)
                     (extract workspace) (if workspace (cuda/size workspace) 0))
   cudnn-handle))

(defn convolution-bwd-bias
  ([cudnn-handle alpha desc-dy buf-dy beta desc-db buf-db]
   (convolution-bwd-bias* (extract cudnn-handle)
                          (ptr alpha) (extract (desc desc-dy)) (extract buf-dy)
                          (ptr beta) (extract (desc desc-db)) (extract buf-db))))

(extend-type cudnnConvolutionBwdDataAlgoPerf
  AlgoPerf
  (algo [this]
    (dec-convolution-bwd-data-algo (.algo this)))
  (workspace-size [this]
    (.memory this))
  (status [this]
    (.status this))
  (algo-time [this]
    (.time this))
  (determinism [this]
    (dec-determinism (.determinism this)))
  (math-type [this]
    (dec-math-type (.mathType this))))

(defn convolution-bwd-data-find-algo
  ([cudnn-handle cd desc-w desc-dy desc-dx algo-count]
   (map algo-perf
        (convolution-bwd-data-find-algo* (extract cudnn-handle) (extract cd) (extract-filter desc-w)
                                         (extract (desc desc-dy)) (extract (desc desc-dx))
                                         algo-count)))
  ([cudnn-handle cd desc-w desc-dy desc-dx]
   (first (convolution-bwd-data-find-algo cudnn-handle cd desc-w desc-dy desc-dx 1))))

(defn convolution-bwd-data
  [cudnn-handle cd algo alpha desc-w buf-w desc-dy buf-dy
   beta desc-dx buf-dx workspace]
  (convolution-bwd-data* (extract cudnn-handle) (extract cd)
                         (enc-keyword cudnn-convolution-bwd-data-algo algo)
                         (ptr alpha) (extract-filter desc-w) (extract buf-w)
                         (extract (desc desc-dy)) (extract buf-dy)
                         (ptr beta) (extract (desc desc-dx)) (extract buf-dx)
                         (extract workspace) (if workspace (cuda/size workspace) 0))
  cudnn-handle)

(extend-type cudnnConvolutionBwdFilterAlgoPerf
  AlgoPerf
  (algo [this]
    (dec-convolution-bwd-filter-algo (.algo this)))
  (workspace-size [this]
    (.memory this))
  (status [this]
    (.status this))
  (algo-time [this]
    (.time this))
  (determinism [this]
    (dec-determinism (.determinism this)))
  (math-type [this]
    (dec-math-type (.mathType this))))

(defn convolution-bwd-filter-find-algo
  ([cudnn-handle cd desc-x desc-dy desc-dw algo-count]
   (map algo-perf
    (convolution-bwd-filter-find-algo* (extract cudnn-handle) (extract cd) (extract (desc desc-x))
                                       (extract (desc desc-dy)) (extract-filter desc-dw)
                                       algo-count)))
  ([cudnn-handle cd desc-x desc-dy desc-dw]
   (first (convolution-bwd-filter-find-algo cudnn-handle cd desc-x desc-dy desc-dw 1))))

(defn convolution-bwd-filter
  [cudnn-handle cd algo alpha desc-x buf-x desc-dy buf-dy
   beta desc-dw buf-dw workspace]
  (convolution-bwd-filter* (extract cudnn-handle) (extract cd)
                           (enc-keyword cudnn-convolution-bwd-filter-algo algo)
                           (ptr alpha) (extract (desc desc-x)) (extract buf-x)
                           (extract (desc desc-dy)) (extract buf-dy)
                           (ptr beta) (extract-filter desc-dw) (extract buf-dw)
                           (extract workspace) (if workspace (cuda/size workspace) 0))
  cudnn-handle)

(defn convolution-bwd-bias
  [cudnn-handle alpha desc-dy buf-dy beta desc-db buf-db]
  (convolution-bwd-bias* (extract cudnn-handle)
                         (ptr alpha) (extract (desc desc-dy)) (extract buf-dy)
                         (ptr beta) (extract (desc desc-db)) (extract buf-db))
  cudnn-handle)

;; ======================== Pooling ================================================================

(defn pooling-descriptor
  ([mode nan-opt kernel stride padding]
   (let-release [pd (wrap (pooling-descriptor*))]
     (let [mode (enc-keyword cudnn-pooling-mode mode)
           nan-opt (enc-nan-propagation nan-opt)]
       (try
         (wrap
          (if (< 2 (count kernel))
            (pooling-nd-descriptor* (extract pd) mode nan-opt
                                    (int-array kernel) (int-array stride) (int-array padding))
            (pooling-2d-descriptor* (extract pd) mode nan-opt kernel stride padding)))))
     pd))
  ([mode kernel stride padding]
   (pooling-descriptor mode true kernel stride padding)))

(defn pooling-forward [cudnn-handle pd alpha desc-x buf-x beta desc-y buf-y]
  (pooling-forward* (extract cudnn-handle) (extract pd)
                    (ptr alpha) (extract (desc desc-x)) (extract buf-x)
                    (ptr beta) (extract (desc desc-y)) (extract buf-y)))

(defn pooling-backward [cudnn-handle pd alpha
                        desc-y buf-y desc-dy buf-dy desc-x buf-x
                        beta desc-dx buf-dx]
  (pooling-backward* (extract cudnn-handle) (extract pd)
                     (ptr alpha) (extract (desc desc-y)) (extract buf-y)
                     (extract (desc desc-dy)) (extract buf-dy)
                     (extract (desc desc-x)) (extract buf-x)
                     (ptr beta) (extract (desc desc-dx)) (extract buf-dx)))
