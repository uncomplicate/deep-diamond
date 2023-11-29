;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.diamond.internal.cudnn.core
  (:require [uncomplicate.commons
             [core :refer [let-release with-release size bytesize info]]
             [utils :refer [dragan-says-ex enc-keyword mask]]]
            [uncomplicate.fluokitten.protocols :refer [extract]]
            [uncomplicate.clojure-cpp
             :refer [pointer safe int-pointer long-pointer double-pointer get-entry ptr ptr2
                     type-pointer pointer-pointer]]
            [uncomplicate.clojurecuda.core :refer [cuda-malloc cuda-free!]]
            [uncomplicate.clojurecuda.internal.impl :refer [->CUDevicePtr]]
            [uncomplicate.diamond.internal.cudnn
             [protocols :refer :all]
             [constants :refer :all]
             [impl :refer :all]])
  (:import java.lang.Exception
           org.bytedeco.cuda.global.cudnn
           [org.bytedeco.cuda.cudnn cudnnContext cudnnRuntimeTag_t cudnnConvolutionFwdAlgoPerf_t
            cudnnConvolutionBwdDataAlgoPerf_t cudnnConvolutionBwdFilterAlgoPerf_t]
           [uncomplicate.diamond.internal.cudnn.impl CUTensorDescriptor CUFilterDescriptor]))

(defn safe-ptr [x]
  (safe (pointer x)))

(defprotocol AlgoPerf
  (algo [this])
  (workspace-size [this])
  (status [this])
  (algo-time [this])
  (determinism [this])
  (math-type [this]))

(defn cudnn-context [stream]
  (let [status (remove zero? [(cudnn/cudnnOpsInferVersionCheck)
                              (cudnn/cudnnOpsTrainVersionCheck)
                              (cudnn/cudnnCnnInferVersionCheck)
                              (cudnn/cudnnCnnTrainVersionCheck)])]
    (if (empty? status)
      (cudnn-context* (extract stream))
      (throw (cudnn-error (first status) "cuDNN version mismatch.")))))

(defn get-cudnn-stream [handle]
  (get-cudnn-stream* (extract handle)))

(defn tensor-descriptor [shape data-type layout]
  (let [d (count shape)
        dtype (enc-keyword cudnn-data-type data-type)]
    (let [td (tensor-descriptor*)]
      (try
        (wrap-tensor-struct
         (if (keyword? layout)
           (let [format (enc-keyword cudnn-format layout)]
             (if (< 4 d)
               (with-release [shape (int-pointer shape)]
                 (tensor-nd-descriptor-ex* td format dtype shape))
               (tensor-4d-descriptor* td format dtype shape)))
           (if (= d (count layout))
             (if (< 4 d)
               (with-release [shape (int-pointer shape)
                              layout (int-pointer layout)]
                 (tensor-nd-descriptor* td dtype shape layout))
               (tensor-4d-descriptor-ex* td dtype shape layout))
             (dragan-says-ex "Shape and strides must have the same length."
                             {:shape shape :strides layout}))))
        (catch Exception e
          (with-check (cudnn/cudnnDestroyTensorDescriptor td)
            (throw e)))))))

(defn equal-desc? [td1 td2]
  (= (desc td1) (desc td2)))

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

(defn strides
  "Queeries the strides of a tensor descriptor."
  [td]
  (.layout ^CUTensorDescriptor (desc td)))

(defn set-tensor
  "Sets all elements of tensor `x` to scalar `value`. Attention: `value` needs
  explicit casting to the proper primitive type (`double`, `float`, etc.) "
  [cudnn-handle desc-x x value]
  (with-release [value (pointer value)]
    (set-tensor* (extract cudnn-handle) (extract (desc desc-x)) (ptr x) value))
  cudnn-handle)

(defn scale-tensor
  "Scales all elements of tensor `x` by scalar `alpha`. Attention: `alpha` needs
  explicit casting to the proper primitive type (`double`, `float`, etc.)
  "
  [cudnn-handle alpha desc-x x]
  (with-release [alpha (pointer alpha)]
    (scale-tensor* (extract cudnn-handle) (extract (desc desc-x)) (ptr x) alpha))
  cudnn-handle)

(defn add-tensor
  "Adds elements of tensor `x`, scaled by scalar `alpha`, to tensor `y`.
  Attention: `alpha` needs explicit casting to the proper primitive type (`double`,
  `float`, etc.). Shapes in `desc-x` and `desc-y` must match, or at least be
  `1` in `desc-y`. This is intended for adding bias. Supports broadcasting when a dimension is `1`.
  "
  [cudnn-handle alpha desc-x x beta desc-y y]
  (with-release [alpha (pointer alpha)
                 beta (pointer beta)]
    (add-tensor* (extract cudnn-handle)
                 alpha (extract (desc desc-x)) (ptr x)
                 beta (extract (desc desc-y)) (ptr y)))
  cudnn-handle)

(defn transform-tensor
  "Adds elements of tensor `x`, scaled by scalar `alpha`, to tensor `y` with a different
  layout. Attention: `alpha` needs explicit casting to the proper primitive type (`double`,
  `float`, etc.). Only supports 4D and 5D tensors. Does not support broadcasting.
  Tensor memory must not overlap. Can be used to convert a tensor with an unsupported format
  to a supported one.
  "
  [cudnn-handle alpha desc-x buf-x beta desc-y buf-y]
  (with-release [alpha (pointer alpha)
                 beta (pointer beta)]
    (transform-tensor* (extract cudnn-handle)
                       alpha (extract (desc desc-x)) (ptr buf-x)
                       beta (extract (desc desc-y)) (ptr buf-y)))
  cudnn-handle)

;; =========================== Activation ============================================

(defn activation-descriptor
  ([mode relu-nan-opt ^double coef]
   (let-release [ad (activation-descriptor*)]
     (activation-descriptor* ad (enc-keyword cudnn-activation-mode mode)
                             (enc-nan-propagation relu-nan-opt)
                             coef)
     ad))
  ([mode ^double coef]
   (activation-descriptor mode true coef))
  ([ad]
   (with-release [mode (int-pointer 1)
                  relu-nan-opt (int-pointer 1)
                  coef (double-pointer 1)]
     (get-activation-descriptor* (extract ad) mode relu-nan-opt coef)
     {:mode (dec-activation-mode (get-entry mode 0))
      :relu-nan-opt (dec-nan-propagation (get-entry relu-nan-opt 0))
      :coef (get-entry coef 0)})))

(defn activation-forward [cudnn-handle ad alpha desc-x buf-x beta desc-y buf-y]
  (with-release [alpha (pointer alpha)
                 beta (pointer beta)]
    (activation-forward* (extract cudnn-handle) (extract ad)
                         alpha (extract (desc desc-x)) (ptr buf-x)
                         beta (extract (desc desc-y)) (ptr buf-y)))
  cudnn-handle)

(defn activation-backward [cudnn-handle ad
                           alpha desc-y buf-y desc-dy buf-dy
                           desc-x buf-x beta desc-dx buf-dx]
  (with-release [alpha (pointer alpha)
                 beta (pointer beta)]
    (activation-backward* (extract cudnn-handle) (extract ad)
                          alpha (extract (desc desc-y)) (ptr buf-y)
                          (extract (desc desc-dy)) (ptr buf-dy)
                          (extract (desc desc-x)) (ptr buf-x)
                          beta (extract (desc desc-dx)) (ptr buf-dx)))
  cudnn-handle)

;; ============================ Reduce ==============================================

(defn reduce-tensor-descriptor
  ([op comp-type nan-opt indices]
   (let-release [rtd (reduce-tensor-descriptor*)]
     (reduce-tensor-descriptor* (extract rtd) (enc-keyword cudnn-reduce-tensor-op op)
                                (enc-keyword cudnn-data-type comp-type)
                                (enc-nan-propagation nan-opt)
                                (enc-keyword cudnn-reduce-tensor-indices indices))
     rtd))
  ([op comp-type nan-opt]
   (let-release [rtd (reduce-tensor-descriptor*)]
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
   (with-release [alpha (pointer alpha)
                  beta (pointer beta)]
     (reduce-tensor* (extract cudnn-handle) (extract rtd)
                     (ptr indices) (bytesize indices)
                     (ptr workspace) (bytesize workspace)
                     alpha (extract (desc desc-x)) (ptr buf-x)
                     beta (extract (desc desc-y)) (ptr buf-y)))
   cudnn-handle)
  ([cudnn-handle rtd alpha desc-x buf-x beta desc-y buf-y]
   (let [indices-size (reduction-indices-size cudnn-handle rtd desc-x desc-y)
         workspace-size (reduction-workspace-size cudnn-handle rtd desc-x desc-y)]
     (let [indices (cuda-malloc (max 1 indices-size))
           workspace (cuda-malloc (max 1 workspace-size))]
       (try
         (reduce-tensor cudnn-handle rtd indices workspace alpha desc-x buf-x beta desc-y buf-y)
         (finally (cuda-free! indices)
                  (cuda-free! workspace)))))))

;; =========================== Softmax ============================================

(defn softmax-forward [cudnn-handle algo mode alpha desc-x buf-x beta desc-y buf-y]
  (with-release [alpha (pointer alpha)
                 beta (pointer beta)]
    (softmax-forward* (extract cudnn-handle) (enc-keyword cudnn-softmax-algorithm algo)
                      (enc-keyword cudnn-softmax-mode mode)
                      alpha (extract (desc desc-x)) (ptr buf-x)
                      beta (extract (desc desc-y)) (ptr buf-y)))
  cudnn-handle)

(defn softmax-backward [cudnn-handle algo mode
                        alpha desc-y buf-y desc-dy buf-dy
                        beta desc-dx buf-dx]
  (with-release [alpha (pointer alpha)
                 beta (pointer beta)]
    (softmax-backward* (extract cudnn-handle)  (enc-keyword cudnn-softmax-algorithm algo)
                       (enc-keyword cudnn-softmax-mode mode)
                       alpha (extract (desc desc-y)) (ptr buf-y)
                       (extract (desc desc-dy)) (ptr buf-dy)
                       beta (extract (desc desc-dx)) (ptr buf-dx)))
  cudnn-handle)

;; ============================ Filter =============================================

(defn filter-descriptor [shape data-type format]
  (let [d (count shape)
        dtype (enc-keyword cudnn-data-type data-type)
        format (enc-keyword cudnn-format format)]
    (wrap-filter-struct
     (let-release [fd (filter-descriptor*)]
       (if (< 4 d)
         (with-release [shape (int-pointer shape)]
           (filter-nd-descriptor* fd dtype format shape))
         (filter-4d-descriptor* fd dtype format shape))))))

;; ============================ Convolution ========================================

(defn convolution-descriptor [mode data-type pad stride dilation]
  (let-release [cd (convolution-descriptor*)]
    (let [mode (enc-keyword cudnn-convolution-mode mode)
          dtype (enc-keyword cudnn-data-type data-type)]
      (if (< 2 (count pad))
        (with-release [pad (int-pointer pad)
                       stride (int-pointer stride)
                       dilation (int-pointer dilation)]
          (convolution-nd-descriptor* (extract cd) pad stride dilation mode dtype))
        (convolution-2d-descriptor* (extract cd) pad stride dilation mode dtype)))
    cd))

(extend-type cudnnConvolutionFwdAlgoPerf_t
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
                                    (extract (desc desc-w)) (extract (desc desc-y))
                                    algo-count)))
  ([cudnn-handle cd desc-x desc-w desc-y]
   (first (convolution-fwd-find-algo cudnn-handle cd desc-x desc-w desc-y 1))))

(defn convolution-fwd
  "The forward convolution operation.
  cuDNN `:convolution` algorithm uses flipped kernels as real convolution from the books.
  To match DNNL (more practical), use `:cross-correlation.`"
  ([cudnn-handle cd algo alpha desc-x buf-x desc-w buf-w beta desc-y buf-y workspace]
   (with-release [alpha (pointer alpha)
                  beta (pointer beta)]
     (convolution-fwd* (extract cudnn-handle) (extract cd)
                       (enc-keyword cudnn-convolution-fwd-algo algo)
                       alpha (extract (desc desc-x)) (ptr buf-x)
                       (extract (desc desc-w)) (ptr buf-w)
                       beta (extract (desc desc-y)) (ptr buf-y)
                       (ptr workspace) (bytesize workspace)))

   cudnn-handle)
  ([cudnn-handle cd algo ad alpha1 desc-x buf-x desc-w buf-w alpha2 buf-z
    desc-bias buf-bias desc-y buf-y workspace]
   (with-release [alpha1 (pointer alpha1)
                  alpha2 (pointer alpha2)]
     (convolution-fwd* (extract cudnn-handle) (extract cd)
                       (enc-keyword cudnn-convolution-fwd-algo algo) (extract ad)
                       alpha1 (extract (desc desc-x)) (ptr buf-x)
                       (extract (desc desc-w)) (ptr buf-w)
                       alpha2 (ptr buf-z)
                       (extract (desc desc-bias)) (ptr buf-bias)
                       (extract (desc desc-y)) (ptr buf-y)
                       (extract workspace) (bytesize workspace)))
   cudnn-handle))

(defn convolution-bwd-bias
  ([cudnn-handle alpha desc-dy buf-dy beta desc-db buf-db]
   (with-release [alpha (pointer alpha)
                  beta (pointer beta)]
     (convolution-bwd-bias* (extract cudnn-handle)
                            alpha (extract (desc desc-dy)) (ptr buf-dy)
                            beta (extract (desc desc-db)) (ptr buf-db)))))

(extend-type cudnnConvolutionBwdDataAlgoPerf_t
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
        (convolution-bwd-data-find-algo* (extract cudnn-handle) (extract cd) (extract (desc desc-w))
                                         (extract (desc desc-dy)) (extract (desc desc-dx))
                                         algo-count)))
  ([cudnn-handle cd desc-w desc-dy desc-dx]
   (first (convolution-bwd-data-find-algo cudnn-handle cd desc-w desc-dy desc-dx 1))))

(defn convolution-bwd-data
  [cudnn-handle cd algo alpha desc-w buf-w desc-dy buf-dy
   beta desc-dx buf-dx workspace]
  (with-release [alpha (pointer alpha)
                 beta (pointer beta)]
    (convolution-bwd-data* (extract cudnn-handle) (extract cd)
                           (enc-keyword cudnn-convolution-bwd-data-algo algo)
                           alpha (extract (desc desc-w)) (ptr buf-w)
                           (extract (desc desc-dy)) (ptr buf-dy)
                           beta (extract (desc desc-dx)) (ptr buf-dx)
                           (extract workspace) (bytesize workspace)))
  cudnn-handle)

(extend-type cudnnConvolutionBwdFilterAlgoPerf_t
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
                                           (extract (desc desc-dy)) (extract (desc desc-dw))
                                           algo-count)))
  ([cudnn-handle cd desc-x desc-dy desc-dw]
   (first (convolution-bwd-filter-find-algo cudnn-handle cd desc-x desc-dy desc-dw 1))))

(defn convolution-bwd-filter
  [cudnn-handle cd algo alpha desc-x buf-x desc-dy buf-dy
   beta desc-dw buf-dw workspace]
  (with-release [alpha (pointer alpha)
                 beta (pointer beta)]
    (convolution-bwd-filter* (extract cudnn-handle) (extract cd)
                             (enc-keyword cudnn-convolution-bwd-filter-algo algo)
                             alpha (extract (desc desc-x)) (ptr buf-x)
                             (extract (desc desc-dy)) (ptr buf-dy)
                             beta (extract (desc desc-dw)) (ptr buf-dw)
                             (extract workspace) (bytesize workspace)))
  cudnn-handle)

(defn convolution-bwd-bias
  [cudnn-handle alpha desc-dy buf-dy beta desc-db buf-db]
  (with-release [alpha (pointer alpha)
                 beta (pointer beta)]
    (convolution-bwd-bias* (extract cudnn-handle)
                           alpha (extract (desc desc-dy)) (ptr buf-dy)
                           beta (extract (desc desc-db)) (ptr buf-db)))
  cudnn-handle)

;; ======================== Pooling ================================================================

(defn pooling-descriptor
  ([mode nan-opt kernel stride padding]
   (let-release [pd (pooling-descriptor*)]
     (let [mode (enc-keyword cudnn-pooling-mode mode)
           nan-opt (enc-nan-propagation nan-opt)]
       (if (< 2 (count kernel))
         (with-release [kernel (int-pointer kernel)
                        stride (int-pointer stride)
                        padding(int-pointer padding)]
           (pooling-nd-descriptor* (extract pd) mode nan-opt kernel stride padding))
         (pooling-2d-descriptor* (extract pd) mode nan-opt kernel stride padding)))
     pd))
  ([mode kernel stride padding]
   (pooling-descriptor mode true kernel stride padding)))

(defn pooling-forward [cudnn-handle pd alpha desc-x buf-x beta desc-y buf-y]
  (with-release [alpha (pointer alpha)
                 beta (pointer beta)]
    (pooling-forward* (extract cudnn-handle) (extract pd)
                      alpha (extract (desc desc-x)) (ptr buf-x)
                      beta (extract (desc desc-y)) (ptr buf-y)))
  cudnn-handle)

(defn pooling-backward [cudnn-handle pd alpha
                        desc-y buf-y desc-dy buf-dy desc-x buf-x
                        beta desc-dx buf-dx]
  (with-release [alpha (pointer alpha)
                 beta (pointer beta)]
    (pooling-backward* (extract cudnn-handle) (extract pd)
                       alpha (extract (desc desc-y)) (ptr buf-y)
                       (extract (desc desc-dy)) (ptr buf-dy)
                       (extract (desc desc-x)) (ptr buf-x)
                       beta (extract (desc desc-dx)) (ptr buf-dx)))
  cudnn-handle)

;; ====================== Batch Normalization ===========================================

(defn batch-norm-descriptor [desc-x mode]
  (wrap-tensor-struct (batch-norm-param-descriptor* (extract desc-x)
                                                    (enc-keyword cudnn-batch-norm-mode mode))))

(defn batch-norm-runtime-err? [cudnn-handle err-mode]
  (with-release [status (int-pointer 1)]
    (cudnn/cudnnQueryRuntimeError ^cudnnContext (extract cudnn-handle) status
                                  (int (enc-keyword cudnn-err-query-mode err-mode)) nil)
    (= 0 (get-entry status 0))))

(defn batch-norm-fwd-inference [cudnn-handle mode alpha beta desc-x buf-x desc-y buf-y
                                desc-param buf-scale buf-shift buf-mean buf-var]
  (with-release [alpha (pointer alpha)
                 beta (pointer beta)]
    (batch-norm-fwd-inference* (extract cudnn-handle) (enc-keyword cudnn-batch-norm-mode mode)
                               alpha beta (extract (desc desc-x)) (ptr buf-x)
                               (extract (desc desc-y)) (ptr buf-y) (extract desc-param)
                               (ptr buf-scale) (ptr buf-shift) (ptr buf-mean) (ptr buf-var)
                               (max cudnn/CUDNN_BN_MIN_EPSILON 1e-8)))
  cudnn-handle)

(defn batch-norm-fwd-training [cudnn-handle mode alpha beta desc-x buf-x desc-y buf-y
                               desc-param buf-scale buf-shift n
                               buf-running-mean buf-running-var buf-save-mean buf-save-inv-var]
  (let [exp-avg (double (/ 1 (inc (long n))))]
    (with-release [alpha (pointer alpha)
                   beta (pointer beta)]
      (batch-norm-fwd-training* (extract cudnn-handle) (enc-keyword cudnn-batch-norm-mode mode)
                                alpha beta (extract (desc desc-x)) (ptr buf-x)
                                (extract (desc desc-y)) (ptr buf-y) (extract desc-param)
                                (ptr buf-scale) (ptr buf-shift) exp-avg
                                (ptr buf-running-mean) (ptr buf-running-var)
                                (max cudnn/CUDNN_BN_MIN_EPSILON 1e-8)
                                (ptr buf-save-mean) (ptr buf-save-inv-var)))
    cudnn-handle))

(defn batch-norm-bwd [cudnn-handle mode alpha-data beta-data alpha-param beta-param
                      desc-x buf-x desc-dy buf-dy desc-dx buf-dx desc-param
                      buf-scale buf-scale-diff buf-shift-diff buf-saved-mean buf-saved-inv-var]
  (with-release [alpha-data (pointer alpha-data)
                 beta-data (pointer beta-data)
                 alpha-param (pointer alpha-param)
                 beta-param (pointer beta-param)]
    (batch-norm-backward* (extract cudnn-handle) (enc-keyword cudnn-batch-norm-mode mode)
                          alpha-data beta-data alpha-param beta-param
                          (extract (desc desc-x)) (ptr buf-x)
                          (extract (desc desc-dy)) (ptr buf-dy)
                          (extract (desc desc-dx)) (ptr buf-dx) (extract desc-param)
                          (ptr buf-scale) (ptr buf-scale-diff) (ptr buf-shift-diff)
                          (max cudnn/CUDNN_BN_MIN_EPSILON 1e-8)
                          (ptr buf-saved-mean) (ptr buf-saved-inv-var)))
  cudnn-handle)

;; ====================== Dropout ======================================================

(defn dropout-descriptor
  ([cudnn-handle dropout states state-size]
   (dropout-descriptor cudnn-handle dropout states state-size (rand-int Integer/MAX_VALUE)))
  ([cudnn-handle dropout states state-size seed]
   (let-release [dd (rnn-descriptor*)]
     (dropout-descriptor* (extract cudnn-handle) (extract dd) (float dropout)
                          (ptr states) (long state-size) (long seed))
     dd)))

(defn dropout-states-size ^long [cudnn-handle]
  (dropout-states-size* (extract cudnn-handle)))

(defn dropout-reserve-space-size ^long [cudnn-handle]
  (dropout-reserve-space-size* (extract cudnn-handle)))

;; ======================== RNN ==============================================================

(defn build-rnn-dynamic! [cudnn-handle rd ^long mini-batch]
  (with-check (cudnn/cudnnBuildRNNDynamic (extract cudnn-handle) (extract rd) mini-batch)
    rd))

(defn rnn-descriptor
  ([algo mode bias-mode direction-mode input-mode data-type math-prec math-type
    input-size hidden-size proj-size num-layers dropout-desc & aux-flags]
   (let-release [rd (rnn-descriptor*)]
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
   (with-release [algo (int-pointer 1)
                  mode (int-pointer 1)
                  bias-mode (int-pointer 1)
                  direction-mode (int-pointer 1)
                  input-mode (int-pointer 1)
                  data-type (int-pointer 1)
                  math-prec (int-pointer 1)
                  math-type (int-pointer 1)
                  input-size (int-pointer 1)
                  hidden-size (int-pointer 1)
                  proj-size (int-pointer 1)
                  num-layers (int-pointer 1)
                  dropout-desc (dropout-descriptor*)
                  aux-flags (int-pointer 1)]
     (get-rnn-descriptor* (extract rd)
                          algo mode bias-mode direction-mode input-mode data-type math-prec math-type
                          input-size hidden-size proj-size num-layers dropout-desc aux-flags)
     {:algo (dec-rnn-algo-mode (get-entry algo 0))
      :mode (dec-rnn-cell-mode (get-entry mode 0))
      :bias (dec-rnn-bias-mode (get-entry bias-mode 0))
      :direction (dec-direction-mode (get-entry direction-mode 0))
      :input (dec-rnn-input-mode (get-entry input-mode 0))
      :data-type (dec-data-type (get-entry data-type 0))
      :math-prec (dec-data-type (get-entry math-prec 0))
      :math-type (dec-math-type (get-entry math-type 0))
      :input-size (get-entry input-size 0)
      :hidden-size (get-entry hidden-size 0)
      :proj-size (get-entry proj-size 0)
      :layers (get-entry num-layers 0)
      :dropout (info dropout-desc)
      :aux-flags (get-entry aux-flags 0)})))

(defn rnn-weight-params [cudnn-handle rd pseudo-layer weight-space lin-layer-id]
  (let-release [w-desc (wrap-tensor-struct (tensor-descriptor*))
                b-desc (wrap-tensor-struct (tensor-descriptor*))]
    (with-release [w-addr (pointer-pointer 1)
                   b-addr (pointer-pointer 1)]
      (rnn-weight-params* (extract cudnn-handle) (extract rd) pseudo-layer
                          (bytesize weight-space) (ptr weight-space) lin-layer-id
                          (extract w-desc) w-addr (extract b-desc) b-addr)
      (let-release [w-addr (long-pointer (get-entry w-addr 0))
                    b-addr (long-pointer (get-entry b-addr 0))
                    w-buf (->CUDevicePtr w-addr (bytesize w-desc) false)
                    b-buf (->CUDevicePtr b-addr (bytesize b-desc) false)]
        [w-desc w-buf b-desc b-buf]))))

(defn rnn-weights-space-size ^long [cudnn-handle rd]
  (rnn-weight-space-size* (extract cudnn-handle) (extract rd)))

(defn rnn-temp-space-size [cudnn-handle rd x-desc forward-mode]
  (rnn-temp-space-size* (extract cudnn-handle) (extract rd)
                        (enc-keyword cudnn-forward-mode forward-mode)
                        (extract x-desc)))

(defn rnn-data-descriptor
  ([data-type layout vector-size seq-lengths padding-fill]
   (with-release [seq-lengths (int-pointer seq-lengths)
                  padding-fill ()]
     (let-release [rd (rnn-data-descriptor*)]
       (rnn-data-descriptor* (extract rd) (enc-keyword cudnn-data-type data-type)
                             (enc-keyword cudnn-rnn-data-layout layout) vector-size
                             seq-lengths ((type-pointer data-type) padding-fill))
       rd)))
  ([vector-size seq-lengths]
   (rnn-data-descriptor :float :seq-mayor-unpacked vector-size seq-lengths 0)))

(defn rnn-fwd [cudnn-handle rd forward-mode dev-seq-lengths
               desc-x buf-x desc-y buf-y desc-h buf-hx buf-hy desc-c buf-cx buf-cy
               weight-space work-space reserve-space]
  (rnn-fwd* (extract cudnn-handle) (extract rd) (enc-keyword cudnn-forward-mode forward-mode)
            (ptr dev-seq-lengths)
            (extract desc-x) (ptr buf-x) (extract desc-y) (ptr buf-y)
            (extract desc-h) (ptr2 buf-hx) (ptr2 buf-hy)
            (extract desc-c) (ptr2 buf-cx) (ptr2 buf-cy)
            (bytesize weight-space) (ptr weight-space)
            (bytesize work-space) (ptr work-space)
            (bytesize reserve-space) (ptr reserve-space))
  cudnn-handle)

(defn rnn-bwd-data [cudnn-handle rd dev-seq-lengths
                    desc-y buf-y buf-dy desc-x buf-dx desc-h buf-hx buf-dhy buf-dhx
                    desc-c buf-cx buf-dcy buf-dcx weight-space work-space reserve-space]
  (rnn-bwd-data* (extract cudnn-handle) (extract rd)
                 (extract dev-seq-lengths)
                 (extract desc-y) (ptr buf-y) (ptr buf-dy)
                 (extract desc-x) (ptr2 buf-dx)
                 (extract desc-h) (ptr2 buf-hx) (ptr2 buf-dhy) (ptr2 buf-dhx)
                 (extract desc-c) (ptr2 buf-cx) (ptr2 buf-dcy) (ptr2 buf-dcx)
                 [(bytesize weight-space) (ptr weight-space)
                  (bytesize work-space) (ptr work-space)
                  (bytesize reserve-space) (ptr reserve-space)])
  cudnn-handle)

(defn rnn-bwd-weights [cudnn-handle rd add-grad dev-seq-lengths
                       desc-x buf-dx desc-h buf-hx desc-y buf-y
                       weight-space work-space reserve-space]
  (rnn-bwd-weights* (extract cudnn-handle) (extract rd) (enc-keyword cudnn-grad-mode add-grad)
                    (extract dev-seq-lengths)
                    (extract desc-x) (ptr buf-dx)
                    (extract desc-h) (ptr2 buf-hx) (extract desc-y) (ptr buf-y)
                    (bytesize weight-space) (ptr weight-space)
                    (bytesize work-space) (ptr work-space)
                    (bytesize reserve-space) (ptr reserve-space))
  cudnn-handle)
