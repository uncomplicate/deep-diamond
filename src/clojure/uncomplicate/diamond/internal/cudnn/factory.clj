;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.diamond.internal.cudnn.factory
  (:require [clojure.java.io :as io]
            [uncomplicate.commons
             [core :refer [Releaseable release let-release with-release view extract]]
             [utils :refer [dragan-says-ex]]]
            [uncomplicate.fluokitten.core :refer [op]]
            [uncomplicate.clojure-cpp :refer [byte-pointer]]
            [uncomplicate.clojurecuda
             [core :refer :all]
             [toolbox :refer [read-int]]]
            [uncomplicate.neanderthal
             [cuda :refer [cuda-float cuda-double cuda-long cuda-int cuda-byte]]
             [block :refer [buffer offset]]]
            [uncomplicate.neanderthal.internal.api :refer :all :exclude [device]]
            [uncomplicate.diamond.tensor :refer [shape data-type layout output]]
            [uncomplicate.diamond.internal
             [protocols :refer [TensorFactory DiamondFactoryProvider NeanderthalFactoryProvider
                                CostFactory DnnFactory RnnFactory]]
             [utils :refer [check-contiguous]]
             [cost :refer [quadratic-cost! mean-absolute-cost! crossentropy-cost!]]]
            [uncomplicate.diamond.internal.dnnl.factory :refer [dnnl-factory]]
            [uncomplicate.diamond.internal.neanderthal.directed :refer [neanderthal-fc-blueprint]]
            [uncomplicate.diamond.internal.cudnn
             [protocols :refer [HandleProvider desc]]
             [core :refer [cudnn-context get-cudnn-stream ndims dims
                           strides transform-tensor set-tensor scale-tensor add-tensor]]
             [tensor :refer [cudnn-tensor cudnn-transformer cudnn-batcher cudnn-shuffler
                             cudnn-tensor-desc]]
             [directed :refer [cudnn-activ-blueprint
                               cudnn-universal-cost cudnn-custom-cost cudnn-pooling-blueprint
                               cudnn-convolution-layer-blueprint cudnn-gaussian-dropout-blueprint
                               cudnn-batch-norm-layer-blueprint cudnn-branch-blueprint
                               cudnn-concat-blueprint cudnn-sum-blueprint cudnn-split-blueprint]]
             [rnn :refer [cudnn-rnn-op-blueprint cudnn-rnn-blueprint cudnn-abbreviate-blueprint]]]))

(def ^{:private true :const true} INEFFICIENT_OPERATION_MSG
  "This operation would be inefficient because it does not use cuDNN capabilities.
  Please use dedicated tensor operations.")

(def ^{:private true :const true} UNSUPPORTED_DATA_TYPE
  "The requested data type is not supported on the CUDA platform.
Please contribute towards making it possible, or use on of the supported types.")

(defn ^:private tensor-1d-equals [modl hstream x y]
  (with-release [equals-kernel (function modl "tensor_1d_equals")
                 eq-flag-buf (mem-alloc-driver Integer/BYTES)]
    (let [n (first (dims x))]
      (memset! eq-flag-buf 0)
      (launch! equals-kernel (grid-1d n) hstream
               (parameters n
                           (extract x) (offset x) (first (strides x))
                           (extract y) (offset y) (first (strides y))
                           eq-flag-buf))
      (= 0 (read-int hstream eq-flag-buf)))))

(defn ^:private tensor-2d-equals [modl hstream x y]
  (with-release [equals-kernel (function modl "tensor_2d_equals")
                 eq-flag-buf (mem-alloc-driver Integer/BYTES)]
    (let [[n c] (dims x)
          [nx cx] (strides x)
          [ny cy] (strides y)]
      (memset! eq-flag-buf 0)
      (launch! equals-kernel (grid-2d n c) hstream
               (parameters n c
                           (extract x) (int (offset x)) (int nx) (int cx) ;;TODO perhaps remove int casts. Neanderthal does not need them any more.
                           (extract y) (int (offset y)) (int ny) (int cy)
                           eq-flag-buf))
      (= 0 (read-int hstream eq-flag-buf)))))

(defn ^:private tensor-3d-equals [modl hstream x y]
  (with-release [equals-kernel (function modl "tensor_3d_equals")
                 eq-flag-buf (mem-alloc-driver Integer/BYTES)]
    (let [[n c h] (dims x)
          [nx cx hx] (strides x)
          [ny cy hy] (strides y)]
      (memset! eq-flag-buf 0)
      (launch! equals-kernel (grid-2d n c h) hstream
               (parameters (int n) (int c) (int h)
                           (extract x) (int (offset x)) (int nx) (int cx) (int hx)
                           (extract y) (int (offset y)) (int ny) (int cy) (int hy)
                           eq-flag-buf))
      (= 0 (read-int hstream eq-flag-buf)))))

(defn ^:private tensor-4d-equals [modl hstream x y]
  (with-release [equals-kernel (function modl "tensor_4d_equals")
                 eq-flag-buf (mem-alloc-driver Integer/BYTES)]
    (let [[n c h w] (dims x)
          [nx cx hx wx] (strides x)
          [ny cy hy wy] (strides y)]
      (memset! eq-flag-buf 0)
      (launch! equals-kernel (grid-3d n c h) hstream
               (parameters (int n) (int c) (int h) (int w)
                           (extract x) (int (offset x)) (int nx) (int cx) (int hx) (int wx)
                           (extract y) (int (offset y)) (int ny) (int cy) (int hy) (int wy)
                           eq-flag-buf))
      (= 0 (read-int hstream eq-flag-buf)))))

(defn ^:private tensor-5d-equals [modl hstream x y]
  (with-release [equals-kernel (function modl "tensor_5d_equals")
                 eq-flag-buf (mem-alloc-driver Integer/BYTES)]
    (let [[n c d h w] (dims x)
          [nx cx dx hx wx] (strides x)
          [ny cy dy hy wy] (strides y)]
      (memset! eq-flag-buf 0)
      (launch! equals-kernel (grid-3d n c h) hstream
               (parameters (int n) (int c) (int d) (int h) (int w)
                           (extract x) (int (offset x)) (int nx) (int cx) (int dx) (int hx) (int wx)
                           (extract y) (int (offset y)) (int ny) (int cy) (int dy) (int hy) (int wy)
                           eq-flag-buf))
      (= 0 (read-int hstream eq-flag-buf)))))

(defn ^:private tensor-equals [modl hstream x y]
  (let [cnt (int (apply * (dims x)))]
    (if (< 0 cnt)
      (case (ndims x)
        1 (tensor-1d-equals modl hstream x y)
        2 (tensor-2d-equals modl hstream x y)
        3 (tensor-3d-equals modl hstream x y)
        4 (tensor-4d-equals modl hstream x y)
        5 (tensor-5d-equals modl hstream x y)
        (dragan-says-ex "Equals is supported only up to 5 dimensions." {:shape (dims x)}))
      (= 0 (int (apply * (dims y)))))))

(defn tensor-method
  ([method x]
   (let [vx (view-vctr x)]
     (check-contiguous x)
     (method (engine vx) vx)))
  ([method x y]
   (let [vx (view-vctr x)]
     (check-contiguous x y)
     (method (engine vx) vx (view-vctr y))))
  ([method x y z]
   (let [vx (view-vctr x)]
     (check-contiguous x y z)
     (method (engine vx) vx (view-vctr y) (view-vctr z)))))

(defn tensor-math
  ([method a y]
   (let [va (view-vctr a)]
     (check-contiguous a y)
     (method (engine va) va (view-vctr y))
     y))
  ([method a b y]
   (let [va (view-vctr a)]
     (check-contiguous a b y)
     (method (engine va) va (view-vctr b) (view-vctr y))
     y)))

(deftype TensorEngine [cudnn-hdl modl hstream cast]
  BlockEngine
  (equals-block [_ x y]
    (tensor-equals modl hstream x y))
  Blas
  (copy [_ x y]
    (transform-tensor cudnn-hdl (cast 1.0) x (byte-pointer (buffer x)) (cast 0.0) y (byte-pointer (buffer y)))
    y)
  (axpy [_ alpha x y]
    (add-tensor cudnn-hdl (cast alpha) x (byte-pointer (buffer x)) (cast 1.0) y (byte-pointer (buffer y)))
    y)
  (swap [_ x y]
    (tensor-method swap x y)
    x)
  (asum [_ x]
    (tensor-method asum x))
  (nrm1 [_ x]
    (tensor-method nrm1 x))
  (nrm2 [_ x]
    (tensor-method nrm2 x))
  (nrmi [this x]
    (tensor-method nrmi x))
  (scal [_ alpha x]
    (scale-tensor cudnn-hdl (cast alpha) x (buffer x))
    x)
  BlasPlus
  (amax [_ x]
    (tensor-method amax x))
  (sum [_ x]
    (tensor-method sum x))
  (set-all [_ value x]
    (set-tensor cudnn-hdl x (byte-pointer (buffer x)) (cast value))
    x)
  (axpby [_ alpha x beta y]
    (add-tensor cudnn-hdl (cast alpha) x (buffer x) (cast beta) y (buffer y))
    y)
  VectorMath
  (sqr [_ a y]
    (tensor-math sqr a y))
  (mul [_ a b y]
    (tensor-math mul a b y))
  (div [_ a b y]
    (tensor-math div a b y))
  (inv [_ a y]
    (tensor-math inv a y))
  (abs [_ a y]
    (tensor-math abs a y))
  (linear-frac [_ a b scalea shifta scaleb shiftb y]
    (let [va (view-vctr a)]
      (linear-frac (engine va) va (view-vctr b) scalea shifta scaleb shiftb (view-vctr y))
      y))
  (fmod [_ a b y]
    (tensor-math fmod a b y)
    a)
  (frem [_ a b y]
    (tensor-math frem a b y))
  (sqrt [_ a y]
    (tensor-math sqrt a y))
  (inv-sqrt [_ a y]
    (tensor-math inv-sqrt a y))
  (cbrt [_ a y]
    (tensor-math cbrt a y))
  (inv-cbrt [_ a y]
    (tensor-math inv-cbrt a y))
  (pow2o3 [_ a y]
    (tensor-math pow2o3 a y))
  (pow3o2 [_ a y]
    (tensor-math pow3o2 a y))
  (pow [_ a b y]
    (tensor-math pow a b y))
  (powx [_ a b y]
    (powx (engine (view-vctr a)) (view-vctr a) b (view-vctr y))
    y)
  (hypot [_ a b y]
    (tensor-math hypot a b y))
  (exp [_ a y]
    (tensor-math exp a y))
  (expm1 [_ a y]
    (tensor-math expm1 a y))
  (log [_ a y]
    (tensor-math log a y))
  (log10 [_ a y]
    (tensor-math log10 a y))
  (sin [_ a y]
    (tensor-math sin a y))
  (cos [_ a y]
    (tensor-math cos a y))
  (tan [_ a y]
    (tensor-math tan a y))
  (sincos [_ a y z]
    (tensor-math sincos a y z))
  (asin [_ a y]
    (tensor-math asin a y))
  (acos [_ a y]
    (tensor-math acos a y))
  (atan [_ a y]
    (tensor-math atan a y))
  (atan2 [_ a b y]
    (tensor-math atan a b y))
  (sinh [_ a y]
    (tensor-math sinh a y))
  (cosh [_ a y]
    (tensor-math cosh a y))
  (tanh [_ a y]
    (tensor-math tanh a y))
  (asinh [_ a y]
    (tensor-math asinh a y))
  (acosh [_ a y]
    (tensor-math acosh a y))
  (atanh [_ a y]
    (tensor-math atanh a y))
  (erf [_ a y]
    (tensor-math erf a y))
  (erfc [_ a y]
    (tensor-math erfc a y))
  (erf-inv [_ a y]
    (tensor-math erf-inv a y))
  (erfc-inv [_ a y]
    (tensor-math erfc-inv a y))
  (cdf-norm [_ a y]
    (tensor-math cdf-norm a y))
  (cdf-norm-inv [_ a y]
    (tensor-math cdf-norm-inv a y))
  (gamma [_ a y]
    (tensor-math gamma a y))
  (lgamma [_ a y]
    (tensor-math lgamma a y))
  (expint1 [_ a y]
    (tensor-math expint1 a y))
  (floor [_ a y]
    (tensor-math floor a y))
  (fceil [_ a y]
    (tensor-math fceil a y))
  (trunc [_ a y]
    (tensor-math trunc a y))
  (round [_ a y]
    (tensor-math round a y))
  (modf [_ a y z]
    (tensor-math modf a y z))
  (frac [_ a y]
    (tensor-math frac a y))
  (fmin [_ a b y]
    (tensor-math fmin a y))
  (fmax [_ a b y]
    (tensor-math fmax a y))
  (copy-sign [_ a b y]
    (tensor-math copy-sign a b y))
  (sigmoid [this a y]
    (dragan-says-ex INEFFICIENT_OPERATION_MSG))
  (ramp [this a y]
    (dragan-says-ex INEFFICIENT_OPERATION_MSG))
  (relu [this alpha a y]
    (dragan-says-ex INEFFICIENT_OPERATION_MSG))
  (elu [this alpha a y]
    (dragan-says-ex INEFFICIENT_OPERATION_MSG))
  RandomNumberGenerator
  (rand-uniform [_ rng-stream lower upper x]
    (rand-uniform (engine (view-vctr x)) rng-stream lower upper (view-vctr x))
    x)
  (rand-normal [_ rng-stream mu sigma x]
    (rand-normal (engine (view-vctr x)) rng-stream mu sigma (view-vctr x))
    x))

(deftype CUDnnFactory [ctx hstream cudnn-hdl master
                       native-diamond-fact
                       neand-facts tensor-engines]
  Releaseable
  (release [_]
    (in-context
     ctx
     (doseq [neand-fact (vals neand-facts)]
       (release neand-fact))
     (doseq [eng (vals tensor-engines)]
       (release eng))
     (release cudnn-hdl))
    (when master
      (when-not (= default-stream hstream)
        (release hstream))
      (release ctx))
    true)
  DiamondFactoryProvider
  (diamond-factory [this]
    this)
  (native-diamond-factory [_]
    native-diamond-fact)
  FlowProvider
  (flow [_]
    hstream)
  HandleProvider
  (handle [_]
    cudnn-hdl)
  NeanderthalFactoryProvider
  (neanderthal-factory [_ dtype]
    (or (get neand-facts dtype)
        (dragan-says-ex UNSUPPORTED_DATA_TYPE {:data-type dtype})))
  TensorFactory
  (create-tensor-desc [this shape dtype format]
    (cudnn-tensor-desc shape dtype format))
  (create-tensor-desc [this tz-desc]
    (cudnn-tensor-desc (shape tz-desc) (data-type tz-desc) (layout tz-desc)))
  (create-tensor [this tensor-desc init]
    (let-release [res (cudnn-tensor this tensor-desc)]
      (when init
        (set-all (engine res) 0 res))
      res))
  (create-tensor [this tensor-desc batch-index init]
    (let-release [res (cudnn-tensor this tensor-desc batch-index)]
      (when init
        (set-all (engine res) 0 res))
      res))
  (create-transformer [_ in-tz out-tz]
    (cudnn-transformer cudnn-hdl (view in-tz) (view out-tz)))
  (create-shuffler [_ src-tz dst-tz]
    (cudnn-shuffler cudnn-hdl (view src-tz) (view dst-tz)))
  (create-batcher [_ src-tz dst-tz mb-size]
    (cudnn-batcher cudnn-hdl (view src-tz) (view dst-tz) mb-size))
  (tensor-engine [this dtype]
    (or (get tensor-engines dtype)
        (dragan-says-ex UNSUPPORTED_DATA_TYPE {:data-type dtype})))
  DnnFactory
  (activ-blueprint [this src-desc activ coef _]
    (cudnn-activ-blueprint this src-desc activ coef))
  (inner-product-blueprint [this src-desc dst-desc weights-type]
    (dragan-says-ex "cuDNN engine does not implement the inner product blueprint."))
  (fc-blueprint [this src-desc dst-desc activ alpha beta weights-type]
    (neanderthal-fc-blueprint this src-desc dst-desc activ alpha beta weights-type))
  (convolution-blueprint [this src-desc weights-desc dst-desc activ
                          strides padding dilation alpha _]
    (cudnn-convolution-layer-blueprint this src-desc weights-desc dst-desc activ
                                       strides padding dilation alpha))
  (pooling-blueprint [this src-desc dst-desc algo strides kernel padding]
    (cudnn-pooling-blueprint this src-desc dst-desc algo strides kernel padding))
  (gaussian-dropout-blueprint [this src-desc sd]
    (cudnn-gaussian-dropout-blueprint this src-desc sd))
  (batch-norm-blueprint [this src-desc activ alpha beta]
    (cudnn-batch-norm-layer-blueprint this src-desc activ alpha beta))
  (concat-blueprint [this src-descs conc-dim dst-type]
    (cudnn-concat-blueprint this src-descs conc-dim dst-type))
  (branch-blueprint [this src-desc branch-dim dst-descs]
    (cudnn-branch-blueprint this src-desc branch-dim dst-descs))
  (split-blueprint [this src-desc n]
    (cudnn-split-blueprint this src-desc n))
  (sum-blueprint [this src-descs]
    (cudnn-sum-blueprint this src-descs))
  (create-workspace [_ byte-size]
    (in-context
     ctx
     (mem-alloc-driver (max 1 (long byte-size)))))
  RnnFactory
  (rnn-op-blueprint [this src-desc dst-desc weights-type activ dir lrs src-iter? dst-iter?]
    (cudnn-rnn-op-blueprint this cudnn-hdl src-desc dst-desc weights-type
                            activ dir lrs src-iter? dst-iter?))
  (rnn-blueprint [fact src-desc dst-desc lrs activ _ weights-type src-iter? dst-iter?]
    (cudnn-rnn-blueprint fact cudnn-hdl src-desc dst-desc lrs activ weights-type src-iter? dst-iter?))
  (abbreviate-blueprint [fact src-desc dst-type]
    (cudnn-abbreviate-blueprint fact src-desc dst-type))
  CostFactory
  (quadratic-cost [_ prev-layer train-tz]
    (cudnn-universal-cost prev-layer train-tz quadratic-cost!))
  (mean-absolute-cost [_ prev-layer train-tz]
    (cudnn-universal-cost prev-layer train-tz mean-absolute-cost!))
  (crossentropy-cost [_ prev-layer train-tz]
    (cudnn-custom-cost prev-layer train-tz
                       (partial crossentropy-cost!
                                ((dims (output prev-layer)) 0)))))

(defn ^:private create-module [src dtype]
  (with-release [prog (compile! (program src) [(str "-DTYPE=" dtype) "-default-device"])]
    (module prog)))

(let [src (slurp (io/resource "uncomplicate/diamond/internal/cuda/blas-plus.cu"))]
  (defn ^:private create-cudnn-factory [ctx hstream cudnn-hdl master]
    (let-release [float-modl (create-module src "float")
                  double-modl (create-module src "double")
                  long-modl (create-module src "long")
                  int-modl (create-module src "int")
                  byte-modl (create-module src "char")
                  native-diamond-fact (dnnl-factory)
                  float-fact (cuda-float ctx hstream)
                  double-fact (cuda-double ctx hstream)
                  long-fact (cuda-long ctx hstream)
                  int-fact (cuda-int ctx hstream)
                  byte-fact (cuda-byte ctx hstream)
                  float-engine (->TensorEngine cudnn-hdl float-modl hstream float)
                  double-engine (->TensorEngine cudnn-hdl double-modl hstream double)
                  long-engine (->TensorEngine cudnn-hdl long-modl hstream long)
                  int-engine (->TensorEngine cudnn-hdl int-modl hstream int)
                  byte-engine (->TensorEngine cudnn-hdl byte-modl hstream byte)]
      (->CUDnnFactory ctx hstream cudnn-hdl master
                      native-diamond-fact
                      {:float float-fact
                       :double double-fact
                       :long long-fact
                       :int int-fact
                       :byte byte-fact
                       :uint8 byte-fact}
                      {:float float-engine
                       :double double-engine
                       :int int-engine
                       :long long-engine
                       :byte byte-engine
                       :uint8 byte-engine}))))

(defn cudnn-factory
  ([ctx hstream]
   (in-context
    ctx
    (let-release [cudnn-hdl (cudnn-context hstream)
                  hstream (get-cudnn-stream cudnn-hdl)]
      (create-cudnn-factory ctx hstream cudnn-hdl false))))
  ([]
   (init)
   (let-release [ctx (context (device))]
     (in-context
      ctx
      (let-release [hstream (stream)
                    cudnn-hdl (cudnn-context hstream)]
        (create-cudnn-factory ctx hstream cudnn-hdl true))))))
