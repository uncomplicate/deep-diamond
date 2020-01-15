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
             [core :refer [Releaseable release let-release with-release]]
             [utils :refer [dragan-says-ex]]]
            [uncomplicate.fluokitten.core :refer [op]]
            [uncomplicate.clojurecuda
             [core :refer :all]
             [toolbox :refer [read-int]]]
            [uncomplicate.neanderthal
             [cuda :refer [cuda-float cuda-double]]
             [block :refer [buffer offset]]]
            [uncomplicate.neanderthal.internal.api :refer :all :exclude [device]]
            [uncomplicate.diamond.internal.protocols
             :refer [TensorFactory DiamondFactoryProvider ContextProvider
                     NeanderthalFactoryProvider CostFactory DnnFactory]]
            [uncomplicate.diamond.internal.cudnn
             [protocols :refer [desc]]
             [core :refer [cudnn-handle get-cudnn-stream tensor-descriptor
                           ndims dims strides transform-tensor add-tensor]]
             [tensor :refer [cudnn-tensor]]])
  (:import jcuda.jcudnn.JCudnn
           uncomplicate.neanderthal.internal.api.Block))

(def ^{:private true :const true} INEFFICIENT_OPERATION_MSG
  "This operation would be inefficient because it does not use cuDNN capabilities.
  Please use dedicated tensor operations.")

(def ^{:private true :const true} UNSUPPORTED_DATA_TYPE
  "The requested data type is not supported on the CUDA platform.
Please contribute towards making it possible, or use on of the supported types.")

(defn check-contiguous
  ([^Block x]
   (when-not (.isContiguous x)
     (dragan-says-ex "Neanderthal API is supported only on contiguous tensors. Please use a copy."
                     {:strides (strides ~x)})))
  ([^Block x ^Block y]
   (check-contiguous x)
   (check-contiguous y))
  ([^Block x ^Block y ^Block z]
   (check-contiguous x)
   (check-contiguous y)
   (check-contiguous z)))

(defn ^:private tensor-1d-equals [modl hstream x y]
  (with-release [equals-kernel (function modl "tensor_1d_equals")
                 eq-flag-buf (mem-alloc Integer/BYTES)]
    (let [n (first (dims x))]
      (memset! eq-flag-buf 0)
      (launch! equals-kernel (grid-1d n) hstream
               (parameters (int n)
                           (buffer x) (int (offset x)) (int (first (strides x)))
                           (buffer y) (int (offset y)) (int (first (strides y)))
                           eq-flag-buf))
      (= 0 (read-int hstream eq-flag-buf)))))

(defn ^:private tensor-2d-equals [modl hstream x y]
  (with-release [equals-kernel (function modl "tensor_2d_equals")
                 eq-flag-buf (mem-alloc Integer/BYTES)]
    (let [[n c] (dims x)
          [nx cx] (strides x)
          [ny cy] (strides y)]
      (memset! eq-flag-buf 0)
      (launch! equals-kernel (grid-2d n c) hstream
               (parameters (int n) (int c)
                           (buffer x) (int (offset x)) (int nx) (int cx)
                           (buffer y) (int (offset y)) (int ny) (int cy)
                           eq-flag-buf))
      (= 0 (read-int hstream eq-flag-buf)))))

(defn ^:private tensor-3d-equals [modl hstream x y]
  (with-release [equals-kernel (function modl "tensor_3d_equals")
                 eq-flag-buf (mem-alloc Integer/BYTES)]
    (let [[n c h] (dims x)
          [nx cx hx] (strides x)
          [ny cy hy] (strides y)]
      (memset! eq-flag-buf 0)
      (launch! equals-kernel (grid-2d n c h) hstream
               (parameters (int n) (int c) (int h)
                           (buffer x) (int (offset x)) (int nx) (int cx) (int hx)
                           (buffer y) (int (offset y)) (int ny) (int cy) (int hy)
                           eq-flag-buf))
      (= 0 (read-int hstream eq-flag-buf)))))

(defn ^:private tensor-4d-equals [modl hstream x y]
  (with-release [equals-kernel (function modl "tensor_4d_equals")
                 eq-flag-buf (mem-alloc Integer/BYTES)]
    (let [[n c h w] (dims x)
          [nx cx hx wx] (strides x)
          [ny cy hy wy] (strides y)]
      (memset! eq-flag-buf 0)
      (launch! equals-kernel (grid-3d n c h) hstream
               (parameters (int n) (int c) (int h) (int w)
                           (buffer x) (int (offset x)) (int nx) (int cx) (int hx) (int wx)
                           (buffer y) (int (offset y)) (int ny) (int cy) (int hy) (int wy)
                           eq-flag-buf))
      (= 0 (read-int hstream eq-flag-buf)))))

(defn ^:private tensor-5d-equals [modl hstream x y]
  (with-release [equals-kernel (function modl "tensor_5d_equals")
                 eq-flag-buf (mem-alloc Integer/BYTES)]
    (let [[n c d h w] (dims x)
          [nx cx dx hx wx] (strides x)
          [ny cy dy hy wy] (strides y)]
      (memset! eq-flag-buf 0)
      (launch! equals-kernel (grid-3d n c h) hstream
               (parameters (int n) (int c) (int d) (int h) (int w)
                           (buffer x) (int (offset x)) (int nx) (int cx) (int dx) (int hx) (int wx)
                           (buffer y) (int (offset y)) (int ny) (int cy) (int dy) (int hy) (int wy)
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

(defn ^:private tensor-set [modl hstream x value]
  (with-release [set-kernel (function modl "tensor_5d_set")]
    (let [[n c d h w] (dims x)
          [nx cx dx hx wx] (strides x)]
      (launch! set-kernel (grid-3d (or n 1) (or c 1) (or h 1)) hstream
               (parameters (int (or n 0)) (int (or c 1)) (int (or d 1)) (int (or h 1)) (int (or w 1))
                           value (buffer x) (int (offset x))
                           (int (or nx 0)) (int (or cx 0)) (int (or dx 0)) (int (or hx 0)) (int (or wx 0))))
      x)))

(defn tensor-method
  ([method x]
   (let [vx (view x)]
     (check-contiguous x)
     (method (engine vx) vx)))
  ([method x y]
   (let [vx (view x)]
     (check-contiguous x)
     (check-contiguous y)
     (method (engine vx) vx (view y))))
  ([method x y z]
   (let [vx (view x)]
     (check-contiguous x)
     (check-contiguous y)
     (check-contiguous z)
     (method (engine vx) vx (view y) (view z)))))

(defn tensor-math
  ([method a y]
   (let [va (view a)]
     (check-contiguous a)
     (check-contiguous y)
     (method (engine va) va (view y))
     y))
  ([method a b y]
   (let [va (view a)]
     (check-contiguous a)
     (check-contiguous b)
     (check-contiguous y)
     (method (engine va) va (view b) (view y))
     y)))

(deftype TensorEngine [cudnn-hdl modl hstream cast ^long byte-cnt]
  BlockEngine
  (equals-block [_ x y]
    (tensor-equals modl hstream x y))
  Blas
  (copy [_ x y]
    (transform-tensor cudnn-hdl (cast 1.0) x (buffer x) (* (offset y) byte-cnt)
                      (cast 0.0) y (buffer y) (* (offset y) byte-cnt))
    y)
  (axpy [_ alpha x y]
    (add-tensor cudnn-hdl (cast alpha) x (buffer x) (* (offset y) byte-cnt)
                (cast 1.0) y (buffer y) (* (offset y) byte-cnt))
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
    (let [vx (view x)]
      (check-contiguous x)
      (scal (engine vx) alpha vx))
    x)
  BlasPlus
  (amax [_ x]
    (tensor-method amax x))
  (sum [_ x]
    (tensor-method sum x))
  (set-all [_ value x]
    (tensor-set modl hstream x (cast value)))
  (axpby [_ alpha x beta y]
    (add-tensor cudnn-hdl (cast alpha) x (buffer x) (* (offset y) byte-cnt)
                (cast beta) y (buffer y) (* (offset y) byte-cnt))
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
    (let [va (view a)]
      (linear-frac (engine va) va (view b) scalea shifta scaleb shiftb (view y))
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
    (powx (engine (view a)) (view a) b (view y))
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
    (rand-uniform (engine (view x)) rng-stream lower upper (view x)))
  (rand-normal [_ rng-stream mu sigma x]
    (rand-normal (engine (view x)) rng-stream mu sigma (view x))))

(deftype CUDnnFactory [ctx hstream cudnn-hdl master
                       neand-facts tensor-engines]
  Releaseable
  (release [_]
    (in-context ctx
                (release cudnn-hdl)
                (doseq [eng (vals tensor-engines)]
                  (release eng))
                (doseq [neand-fact (vals neand-facts)]
                  (release neand-fact))
                (when master
                  (when-not (= default-stream hstream)
                    (release hstream))
                  (release ctx)))
    true)
  DiamondFactoryProvider
  (diamond-factory [this]
    this)
  FlowProvider
  (flow [_]
    hstream)
  ContextProvider
  (context [_]
    ctx)
  NeanderthalFactoryProvider
  (neanderthal-factory [_ dtype]
    (or (get neand-facts dtype)
        (dragan-says-ex UNSUPPORTED_DATA_TYPE {:data-type dtype})))
  TensorFactory
  (create-tensor-desc [this shape dtype format]
    (tensor-descriptor shape dtype format))
  (create-tensor-desc [this tz-desc]
    (desc tz-desc))
  (create-tensor [this tensor-desc init]
    (let-release [res (cudnn-tensor this tensor-desc)]
      (when init
        (set-all (engine res) 0 res))
      res))
  (create-transformer [_ in-tz out-tz]
    )
  (create-shuffler [_ src-tz dst-tz]
    )
  (create-batcher [_ src-tz dst-tz mb-size]
    )
  (create-sum [_ scale dst]
    )
  (create-sum [_ dst scale src scale-srcs]
    )
  (tensor-engine [this dtype]
    (or (get tensor-engines dtype)
        (dragan-says-ex UNSUPPORTED_DATA_TYPE {:data-type dtype})))
  DnnFactory
  (activ-blueprint [this src-desc activ alpha beta]
    )
  (inner-product-blueprint [this src-desc dst-desc weights-type]
    )
  (fc-blueprint [this src-desc dst-desc activ alpha beta weights-type]
    )
  CostFactory
  (quadratic-cost [this prev-layer train-tz]
    )
  (mean-absolute-cost [this prev-layer train-tz]
    )
  (sigmoid-crossentropy-cost [this prev-layer train-tz]
    ))

(JCudnn/setExceptionsEnabled false)

(defn ^:private create-module [src dtype]
  (with-release [prog (compile! (program src)
                                [(str "-DTYPE=" dtype) "-arch=compute_30" "-default-device"])]
    (module prog)))

(let [src (slurp (io/resource "uncomplicate/diamond/internal/cuda/blas-plus.cu"))]

  (defn ^:private create-cudnn-factory [ctx hstream cudnn-hdl master]
    (in-context
     ctx
     (let-release [float-modl (create-module src "float")
                   double-modl (create-module src "double")
                   int-modl (create-module src "int")
                   long-modl (create-module src "long")
                   float-fact (cuda-float ctx hstream)
                   double-fact (cuda-double ctx hstream)
                   int-fact nil  ;;TODO
                   long-fact nil ;;TODO
                   float-engine (->TensorEngine cudnn-hdl float-modl hstream float Float/BYTES)
                   double-engine (->TensorEngine cudnn-hdl double-modl hstream double Double/BYTES)
                   int-engine (->TensorEngine cudnn-hdl int-modl hstream int Integer/BYTES)
                   long-engine (->TensorEngine cudnn-hdl long-modl hstream long Long/BYTES)]
       (->CUDnnFactory ctx hstream cudnn-hdl master
                       {:float float-fact
                        :double double-fact
                        :int int-fact
                        :long long-fact}
                       {:float float-engine
                        :double double-engine
                        :int int-engine
                        :long long-engine})))))

(defn cudnn-factory
  ([ctx hstream]
   (in-context
    ctx
    (let-release [cudnn-hdl (cudnn-handle hstream)
                  hstream (get-cudnn-stream cudnn-hdl)]
      (create-cudnn-factory ctx hstream cudnn-hdl false))))
  ([]
   (init)
   (let-release [ctx (context (device))]
     (in-context
      ctx
      (let-release [hstream (stream)
                    cudnn-hdl (cudnn-handle hstream)]
        (create-cudnn-factory ctx hstream cudnn-hdl true))))))
