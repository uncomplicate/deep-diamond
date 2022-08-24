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
             [utils :refer [dragan-says-ex enc-keyword mask]]]
            [uncomplicate.clojurecuda.core :refer [mem-alloc]]
            [uncomplicate.clojurecuda.internal
             [protocols :as cuda :refer [ptr with-offset]]
             [impl :refer [cu-linear-memory]]]
            [uncomplicate.diamond.internal.cudnn
             [protocols :refer :all]
             [constants :refer :all]
             [impl :refer :all]])
  (:import java.lang.Exception
           jcuda.driver.CUdeviceptr
           [jcuda.jcudnn JCudnn cudnnConvolutionFwdAlgoPerf cudnnConvolutionBwdDataAlgoPerf
            cudnnConvolutionBwdFilterAlgoPerf cudnnTensorDescriptor cudnnRNNDataDescriptor]
           [uncomplicate.diamond.internal.cudnn.impl CUTensorDescriptor CUFilterDescriptor]))

(defprotocol AlgoPerf
  (algo [this])
  (workspace-size [this])
  (status [this])
  (algo-time [this])
  (determinism [this])
  (math-type [this]))

(defn cudnn-handle [stream]
  (let [status (remove zero? [(JCudnn/cudnnOpsInferVersionCheck)
                              (JCudnn/cudnnOpsTrainVersionCheck)
                              (JCudnn/cudnnCnnInferVersionCheck)
                              (JCudnn/cudnnCnnTrainVersionCheck)])]
    (if (empty? status)
      (wrap (cudnn-handle* (extract stream)))
      (throw (cudnn-error (first status) "cuDNN version mismatch.")))))

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
   (activation-descriptor mode true coef))
  ([ad]
   (let [mode (int-array 1)
         relu-nan-opt (int-array 1)
         coef (double-array 1)]
     (get-activation-descriptor* (extract ad) mode relu-nan-opt coef)
     {:mode (dec-activation-mode (aget mode 0))
      :relu-nan-opt (dec-nan-propagation (aget relu-nan-opt 0))
      :coef (aget coef 0)})))

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
                    (ptr beta) (extract (desc desc-y)) (extract buf-y))
  cudnn-handle)

(defn pooling-backward [cudnn-handle pd alpha
                        desc-y buf-y desc-dy buf-dy desc-x buf-x
                        beta desc-dx buf-dx]
  (pooling-backward* (extract cudnn-handle) (extract pd)
                     (ptr alpha) (extract (desc desc-y)) (extract buf-y)
                     (extract (desc desc-dy)) (extract buf-dy)
                     (extract (desc desc-x)) (extract buf-x)
                     (ptr beta) (extract (desc desc-dx)) (extract buf-dx))
  cudnn-handle)

;; ====================== Batch Normalization ===========================================

(defn batch-norm-descriptor [desc-x mode]
  (wrap (batch-norm-param-descriptor* (extract desc-x) (enc-keyword cudnn-batch-norm-mode mode))))

(defn batch-norm-runtime-err? [cudnn-handle err-mode]
  (let [status (int-array 1)]
    (JCudnn/cudnnQueryRuntimeError (extract cudnn-handle) status
                                   (enc-keyword cudnn-err-query-mode err-mode) nil)
    (= 0 (aget status 0))))

(defn batch-norm-fwd-inference [cudnn-handle mode alpha beta desc-x buf-x desc-y buf-y
                                desc-param buf-scale buf-shift buf-mean buf-var]
  (batch-norm-fwd-inference* (extract cudnn-handle) (enc-keyword cudnn-batch-norm-mode mode)
                             (ptr alpha) (ptr beta) (extract (desc desc-x)) (extract buf-x)
                             (extract (desc desc-y)) (extract buf-y) (extract desc-param)
                             (extract buf-scale) (extract buf-shift) (extract buf-mean) (extract buf-var)
                             (max JCudnn/CUDNN_BN_MIN_EPSILON 1e-8))
  cudnn-handle)

(defn batch-norm-fwd-training [cudnn-handle mode alpha beta desc-x buf-x desc-y buf-y
                               desc-param buf-scale buf-shift n
                               buf-running-mean buf-running-var buf-save-mean buf-save-inv-var]
  (let [exp-avg (double (/ 1 (inc (long n))))]
    (batch-norm-fwd-training* (extract cudnn-handle) (enc-keyword cudnn-batch-norm-mode mode)
                              (ptr alpha) (ptr beta) (extract (desc desc-x)) (extract buf-x)
                              (extract (desc desc-y)) (extract buf-y) (extract desc-param)
                              (extract buf-scale) (extract buf-shift) exp-avg
                              (extract buf-running-mean) (extract buf-running-var)
                              (max JCudnn/CUDNN_BN_MIN_EPSILON 1e-8)
                              (extract buf-save-mean) (extract buf-save-inv-var))
    cudnn-handle))

(defn batch-norm-bwd [cudnn-handle mode alpha-data beta-data alpha-param beta-param
                      desc-x buf-x desc-dy buf-dy desc-dx buf-dx desc-param
                      buf-scale buf-scale-diff buf-shift-diff buf-saved-mean buf-saved-inv-var]
  (batch-norm-backward* (extract cudnn-handle) (enc-keyword cudnn-batch-norm-mode mode)
                        (ptr alpha-data) (ptr beta-data) (ptr alpha-param) (ptr beta-param)
                        (extract (desc desc-x)) (extract buf-x)
                        (extract (desc desc-dy)) (extract buf-dy)
                        (extract (desc desc-dx)) (extract buf-dx) (extract desc-param)
                        (extract buf-scale) (extract buf-scale-diff) (extract buf-shift-diff)
                        (max JCudnn/CUDNN_BN_MIN_EPSILON 1e-8)
                        (extract buf-saved-mean) (extract buf-saved-inv-var))
  cudnn-handle)

;; ====================== Dropout ======================================================

(defn dropout-descriptor
  ([cudnn-handle dropout states state-size]
   (dropout-descriptor cudnn-handle dropout states state-size (rand-int Integer/MAX_VALUE)))
  ([cudnn-handle dropout states state-size seed]
   (let-release [dd (wrap (rnn-descriptor*))]
     (dropout-descriptor* (extract cudnn-handle) (extract dd) (float dropout)
                          (ptr states) (long state-size) (long seed))
     dd)))

(defn dropout-states-size ^long [cudnn-handle]
  (dropout-states-size* (extract cudnn-handle)))

(defn dropout-reserve-space-size ^long [cudnn-handle]
  (dropout-reserve-space-size* (extract cudnn-handle)))

;; ======================== RNN ==============================================================

(defn build-rnn-dynamic! [cudnn-handle rd ^long mini-batch]
  (with-check (JCudnn/cudnnBuildRNNDynamic (extract cudnn-handle) (extract rd) mini-batch)
    rd))

(defn rnn-descriptor
  ([algo mode bias-mode direction-mode input-mode data-type math-prec math-type
    input-size hidden-size proj-size num-layers dropout-desc & aux-flags]
   (let-release [rd (wrap (rnn-descriptor*))]
     (rnn-descriptor* (extract rd) (enc-keyword cudnn-rnn-algo-mode algo)
                      (enc-keyword cudnn-rnn-cell-mode mode)
                      (enc-keyword cudnn-rnn-bias-mode bias-mode)
                      (enc-keyword cudnn-direction-mode direction-mode)
                      (enc-keyword cudnn-rnn-input-mode input-mode)
                      (enc-keyword cudnn-data-type data-type)
                      (enc-keyword cudnn-data-type math-prec)
                      (enc-keyword cudnn-math-type math-type)
                      input-size hidden-size proj-size num-layers
                      (extract dropout-desc) (mask cudnn-rnn-aux-mode aux-flags))
     rd))
  ([rd]
   (let [algo (int-array 1)
         mode (int-array 1)
         bias-mode (int-array 1)
         direction-mode (int-array 1)
         input-mode (int-array 1)
         data-type (int-array 1)
         math-prec (int-array 1)
         math-type (int-array 1)
         input-size (int-array 1)
         hidden-size (int-array 1)
         proj-size (int-array 1)
         num-layers (int-array 1)
         dropout-desc (dropout-descriptor*)
         aux-flags (int-array 1)]
     (get-rnn-descriptor* (extract rd)
                          algo mode bias-mode direction-mode input-mode data-type math-prec math-type
                          input-size hidden-size proj-size num-layers dropout-desc aux-flags)
     {:algo (dec-rnn-algo-mode (aget algo 0))
      :mode (dec-rnn-cell-mode (aget mode 0))
      :bias (dec-rnn-bias-mode (aget bias-mode 0))
      :direction (dec-direction-mode (aget direction-mode 0))
      :input (dec-rnn-input-mode (aget input-mode 0))
      :data-type (dec-data-type (aget data-type 0))
      :math-prec (dec-data-type (aget math-prec 0))
      :math-type (dec-math-type (aget math-type 0))
      :input-size (aget input-size 0)
      :hidden-size (aget hidden-size 0)
      :proj-size (aget proj-size 0)
      :layers (aget num-layers 0)
      :dropout (wrap dropout-desc)
      :aux-flags (aget aux-flags 0)})))

(defn rnn-weight-params [cudnn-handle rd pseudo-layer weight-space lin-layer-id]
  (let-release [w-desc (wrap (tensor-descriptor*))
                w-addr (CUdeviceptr.)
                b-desc (wrap (tensor-descriptor*))
                b-addr (CUdeviceptr.)]
    (rnn-weight-params* (extract cudnn-handle) (extract rd) pseudo-layer
                        (cuda/size weight-space) (extract weight-space) lin-layer-id
                        (extract (desc w-desc)) w-addr (extract (desc b-desc)) b-addr)
    (let-release [w-buf (cu-linear-memory w-addr (size w-desc) false)
                  b-buf (cu-linear-memory b-addr (size b-desc) false)]
      [w-desc w-buf b-desc b-buf])))

(defn rnn-weights-space-size ^long [cudnn-handle rd]
  (rnn-weight-space-size* (extract cudnn-handle) (extract rd)))

(defn rnn-temp-space-size [cudnn-handle rd x-desc forward-mode]
  (rnn-temp-space-size* (extract cudnn-handle) (extract rd)
                        (enc-keyword cudnn-forward-mode forward-mode)
                        (extract x-desc)))

(defn rnn-data-descriptor
  ([data-type layout vector-size seq-lengths padding-fill]
   (let-release [rd (wrap (rnn-data-descriptor*))]
     (rnn-data-descriptor* (extract rd) (enc-keyword cudnn-data-type data-type)
                           (enc-keyword cudnn-rnn-data-layout layout) vector-size
                           (if (sequential? seq-lengths) (int-array seq-lengths) seq-lengths)
                           (ptr padding-fill))
     rd))
  ([vector-size seq-lengths]
   (rnn-data-descriptor :float :seq-mayor-unpacked vector-size seq-lengths 0)))

(defn rnn-fwd [cudnn-handle rd forward-mode dev-seq-lengths
               desc-x buf-x desc-y buf-y desc-h buf-hx buf-hy desc-c buf-cx buf-cy
               weight-space work-space reserve-space]
  (rnn-fwd* (extract cudnn-handle) (extract rd) (enc-keyword cudnn-forward-mode forward-mode)
            (extract dev-seq-lengths)
            (extract desc-x) (extract buf-x) (extract desc-y) (extract buf-y)
            (extract desc-h) (extract buf-hx) (extract buf-hy)
            (extract desc-c) (extract buf-cx) (extract buf-cy)
            (cuda/size weight-space) (extract weight-space)
            (cuda/size work-space) (extract work-space)
            (cuda/size reserve-space) (extract reserve-space))
  cudnn-handle)

(defn rnn-bwd-data [cudnn-handle rd dev-seq-lengths
                    desc-y buf-y buf-dy desc-x buf-dx desc-h buf-hx buf-dhy buf-dhx
                    desc-c buf-cx buf-dcy buf-dcx weight-space work-space reserve-space]
  (rnn-bwd-data* (extract cudnn-handle) (extract rd)
                 (extract dev-seq-lengths)
                 (extract desc-y) (extract buf-y) (extract buf-dy)
                 (extract desc-x) (extract buf-dx) (extract desc-h)
                 (extract buf-hx) (extract buf-dhy) (extract buf-dhx)
                 (extract desc-c) (extract buf-cx) (extract buf-dcy) (extract buf-dcx)
                 [(cuda/size weight-space) (extract weight-space)
                  (cuda/size work-space) (extract work-space)
                  (cuda/size reserve-space) (extract reserve-space)])
  cudnn-handle)

(defn rnn-bwd-weights [cudnn-handle rd add-grad dev-seq-lengths
                       desc-x buf-dx desc-h buf-hx desc-y buf-y
                       weight-space work-space reserve-space]
  (rnn-bwd-weights* (extract cudnn-handle) (extract rd) (enc-keyword cudnn-grad-mode add-grad)
                    (extract dev-seq-lengths)
                    (extract desc-x) (extract buf-dx)
                    (extract desc-h) (extract buf-hx) (extract desc-y) (extract buf-y)
                    (cuda/size weight-space) (extract weight-space)
                    (cuda/size work-space) (extract work-space)
                    (cuda/size reserve-space) (extract reserve-space))
  cudnn-handle)
