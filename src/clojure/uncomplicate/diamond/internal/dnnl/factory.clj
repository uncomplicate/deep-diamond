;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.diamond.internal.dnnl.factory
  (:require [uncomplicate.commons
             [core :refer [Releaseable release let-release]]
             [utils :refer [dragan-says-ex]]]
            [uncomplicate.neanderthal
             [native :refer [factory-by-type]]
             [block :refer [data-accessor]]]
            [uncomplicate.neanderthal.internal.api :as neand
             :refer [FlowProvider Blas BlasPlus sum view factory amax RandomNumberGenerator
                     VectorMath rand-uniform rand-normal]]
            [uncomplicate.neanderthal.internal.host.lapack :refer [with-lapack-check]]
            [uncomplicate.diamond.tensor :refer [*diamond-factory* view-tz output layout]]
            [uncomplicate.diamond.internal.protocols
             :refer [TensorFactory DiamondFactoryProvider ContextProvider CostFactory
                     DnnFactory NeanderthalFactoryProvider]]
            [uncomplicate.diamond.internal.dnnl
             [protocols :refer [desc data]]
             [core :refer [memory-desc engine stream memory dims]]
             [tensor :refer [dnnl-tensor dnnl-transformer dnnl-batcher dnnl-shuffler
                             check-contiguous]]
             [fully-connected :refer [dnnl-sum-blueprint dnnl-activ-blueprint
                                      dnnl-inner-product-blueprint dnnl-fc-blueprint
                                      dnnl-universal-cost quadratic-cost mean-absolute-cost
                                      dnnl-custom-cost sigmoid-crossentropy-cost]]])
  (:import [uncomplicate.neanderthal.internal.host CBLAS LAPACK MKL]
           uncomplicate.neanderthal.internal.api.RealBufferAccessor
           uncomplicate.diamond.internal.dnnl.tensor.DnnlTensor))

(def ^{:private true :const true} INEFFICIENT_OPERATION_MSG
  "This operation would be inefficient because it does not use DNNL capabilities.
  Please use dedicated tensor operations.")

(defmacro tensor-method
  ([method x]
   `(do
      (check-contiguous ~x)
      (~method (.dim ~x) (data (.buffer ~x)) (.offset ~x) 1)))
  ([method x y]
   `(do
      (check-contiguous ~x ~y)
      (~method (.dim ~x) (data (.buffer ~x)) (.offset ~x) 1 (data (.buffer ~y)) (.offset ~y) 1))))

(defmacro tensor-amax [iamax da x]
  `(if (< 0 (.dim ~x))
     (do
       (check-contiguous ~x)
       (Math/abs (.get ~da (data (.buffer ~x)) (+ (.offset ~x) (tensor-method ~iamax ~x)))))
     0.0))

(defmacro tensor-laset [method alpha x]
  `(do
     (with-lapack-check
       (~method CBLAS/ORDER_ROW_MAJOR (int \g) (.dim ~x) 1 ~alpha ~alpha (data (.buffer ~x)) (.offset ~x) 1))
     ~x))

(defmacro tensor-axpy
  ([method alpha x y]
   `(do
      (check-contiguous ~x ~y)
      (~method (.dim ~x) ~alpha (data (.buffer ~x)) (.offset ~x) 1 (data (.buffer ~y)) (.offset ~y) 1)
      ~y))
  ([method alpha x beta y]
   `(do
      (check-contiguous ~x ~y)
      (~method (.dim ~x) ~alpha (data (.buffer ~x)) (.offset ~x) 1 ~beta (data (.buffer ~y)) (.offset ~y) 1)
      ~y)))

(defmacro tensor-scal [method alpha x]
 `(do
    (check-contiguous ~x)
    (~method (.dim ~x) ~alpha (data (.buffer ~x)) (.offset ~x) 1)
    ~x))

(defmacro tensor-math
  ([method a y]
   ` (do
       (check-contiguous ~a ~y)
       (~method (.dim ~a) (data (.buffer ~a)) (.offset ~a) (data (.buffer ~y)) (.offset ~y))
       ~y))
  ([method a b y]
   `(do
      (check-contiguous ~a ~b ~y)
      (~method (.dim ~a) (data (.buffer ~a)) (.offset ~a)
       (data (.buffer ~b)) (.offset ~b) (data (.buffer ~y)) (.offset ~y))
      ~y)))

(defmacro tensor-powx [method a b y]
  `(do
     (check-contiguous ~a ~y)
     (~method (.dim ~a) (data (.buffer ~a)) (.offset ~a) ~b (data (.buffer ~y)) (.offset ~y))
     ~y))

(defmacro tensor-linear-frac [method a b scalea shifta scaleb shiftb y]
  `(do
     (check-contiguous ~a ~b ~y)
     (~method (.dim ~a) (data (.buffer ~a)) (.offset ~a) (data (.buffer ~b)) (.offset ~b)
      ~scalea ~shifta ~scaleb ~shiftb (data (.buffer ~y)) (.offset ~y))
     ~y))

(deftype FloatTensorEngine []
  Blas
  (swap [_ x y]
    (tensor-method CBLAS/sswap ^DnnlTensor x ^DnnlTensor y)
    x)
  (copy [_ x y]
    (tensor-method CBLAS/scopy ^DnnlTensor x ^DnnlTensor y)
    y)
  (asum [_ x]
    (tensor-method CBLAS/sasum ^DnnlTensor x))
  (nrm1 [this x]
    (tensor-method CBLAS/sasum ^DnnlTensor x))
  (nrm2 [_ x]
    (tensor-method CBLAS/snrm2 ^DnnlTensor x))
  (nrmi [this x]
    (amax this x))
  (scal [_ alpha x]
    (tensor-scal CBLAS/sscal alpha ^DnnlTensor x)
    x)
  (axpy [_ alpha x y]
    (tensor-axpy CBLAS/saxpy alpha ^DnnlTensor x ^DnnlTensor y)
    y)
  BlasPlus
  (amax [_ x]
    (tensor-amax CBLAS/isamax ^RealBufferAccessor (data-accessor (factory ~x)) ^DnnlTensor x))
  (sum [_ x]
    (let [view-x (view x)]
      (sum (neand/engine x) (view x))))
  (set-all [_ alpha x]
    (tensor-laset LAPACK/slaset alpha ^DnnlTensor x))
  (axpby [_ alpha x beta y]
    (tensor-axpy MKL/saxpby alpha ^DnnlTensor x beta ^DnnlTensor y)
    y)
  VectorMath
  (sqr [_ a y]
    (tensor-math MKL/vdSqr ^DnnlTensor a ^DnnlTensor y))
  (mul [_ a b y]
    (tensor-math MKL/vdMul ^DnnlTensor a ^DnnlTensor b ^DnnlTensor y))
  (div [_ a b y]
    (tensor-math MKL/vdDiv ^DnnlTensor a ^DnnlTensor b ^DnnlTensor y))
  (inv [_ a y]
    (tensor-math MKL/vdInv ^DnnlTensor a ^DnnlTensor y))
  (abs [_ a y]
    (tensor-math MKL/vdAbs ^DnnlTensor a ^DnnlTensor y))
  (linear-frac [_ a b scalea shifta scaleb shiftb y]
    (tensor-linear-frac MKL/vdLinearFrac ^DnnlTensor a ^DnnlTensor b
                        scalea shifta scaleb shiftb ^DnnlTensor y))
  (fmod [_ a b y]
    (tensor-math MKL/vdFmod ^DnnlTensor a ^DnnlTensor b ^DnnlTensor y))
  (frem [_ a b y]
    (tensor-math MKL/vdRemainder ^DnnlTensor a ^DnnlTensor b ^DnnlTensor y))
  (sqrt [_ a y]
    (tensor-math MKL/vdSqrt ^DnnlTensor a ^DnnlTensor y))
  (inv-sqrt [_ a y]
    (tensor-math MKL/vdInvSqrt ^DnnlTensor a ^DnnlTensor y))
  (cbrt [_ a y]
    (tensor-math MKL/vdCbrt ^DnnlTensor a ^DnnlTensor y))
  (inv-cbrt [_ a y]
    (tensor-math MKL/vdInvCbrt ^DnnlTensor a ^DnnlTensor y))
  (pow2o3 [_ a y]
    (tensor-math MKL/vdPow2o3 ^DnnlTensor a ^DnnlTensor y))
  (pow3o2 [_ a y]
    (tensor-math MKL/vdPow3o2 ^DnnlTensor a ^DnnlTensor y))
  (pow [_ a b y]
    (tensor-math MKL/vdPow ^DnnlTensor a ^DnnlTensor b ^DnnlTensor y))
  (powx [_ a b y]
    (tensor-powx MKL/vdPowx ^DnnlTensor a b ^DnnlTensor y))
  (hypot [_ a b y]
    (tensor-math MKL/vdHypot ^DnnlTensor a ^DnnlTensor b ^DnnlTensor y))
  (exp [_ a y]
    (tensor-math MKL/vdExp ^DnnlTensor a ^DnnlTensor y))
  (expm1 [_ a y]
    (tensor-math MKL/vdExpm1 ^DnnlTensor a ^DnnlTensor y))
  (log [_ a y]
    (tensor-math MKL/vdLn ^DnnlTensor a ^DnnlTensor y))
  (log10 [_ a y]
    (tensor-math MKL/vdLog10 ^DnnlTensor a ^DnnlTensor y))
  (sin [_ a y]
    (tensor-math MKL/vdSin ^DnnlTensor a ^DnnlTensor y))
  (cos [_ a y]
    (tensor-math MKL/vdCos ^DnnlTensor a ^DnnlTensor y))
  (tan [_ a y]
    (tensor-math MKL/vdTan ^DnnlTensor a ^DnnlTensor y))
  (sincos [_ a y z]
    (tensor-math MKL/vdSinCos ^DnnlTensor a ^DnnlTensor y ^DnnlTensor z))
  (asin [_ a y]
    (tensor-math MKL/vdAsin ^DnnlTensor a ^DnnlTensor y))
  (acos [_ a y]
    (tensor-math MKL/vdAcos ^DnnlTensor a ^DnnlTensor y))
  (atan [_ a y]
    (tensor-math MKL/vdAtan ^DnnlTensor a ^DnnlTensor y))
  (atan2 [_ a b y]
    (tensor-math MKL/vdAtan2 ^DnnlTensor a ^DnnlTensor b ^DnnlTensor y))
  (sinh [_ a y]
    (tensor-math MKL/vdSinh ^DnnlTensor a ^DnnlTensor y))
  (cosh [_ a y]
    (tensor-math MKL/vdCosh ^DnnlTensor a ^DnnlTensor y))
  (tanh [_ a y]
    (tensor-math MKL/vdTanh ^DnnlTensor a ^DnnlTensor y))
  (asinh [_ a y]
    (tensor-math MKL/vdAsinh ^DnnlTensor a ^DnnlTensor y))
  (acosh [_ a y]
    (tensor-math MKL/vdAcosh ^DnnlTensor a ^DnnlTensor y))
  (atanh [_ a y]
    (tensor-math MKL/vdAtanh ^DnnlTensor a ^DnnlTensor y))
  (erf [_ a y]
    (tensor-math MKL/vdErf ^DnnlTensor a ^DnnlTensor y))
  (erfc [_ a y]
    (tensor-math MKL/vdErfc ^DnnlTensor a ^DnnlTensor y))
  (erf-inv [_ a y]
    (tensor-math MKL/vdErfInv ^DnnlTensor a ^DnnlTensor y))
  (erfc-inv [_ a y]
    (tensor-math MKL/vdErfcInv ^DnnlTensor a ^DnnlTensor y))
  (cdf-norm [_ a y]
    (tensor-math MKL/vdCdfNorm ^DnnlTensor a ^DnnlTensor y))
  (cdf-norm-inv [_ a y]
    (tensor-math MKL/vdCdfNormInv ^DnnlTensor a ^DnnlTensor y))
  (gamma [_ a y]
    (tensor-math MKL/vdGamma ^DnnlTensor a ^DnnlTensor y))
  (lgamma [_ a y]
    (tensor-math MKL/vdLGamma ^DnnlTensor a ^DnnlTensor y))
  (expint1 [_ a y]
    (tensor-math MKL/vdExpInt1 ^DnnlTensor a ^DnnlTensor y))
  (floor [_ a y]
    (tensor-math MKL/vdFloor ^DnnlTensor a ^DnnlTensor y))
  (fceil [_ a y]
    (tensor-math MKL/vdCeil ^DnnlTensor a ^DnnlTensor y))
  (trunc [_ a y]
    (tensor-math MKL/vdTrunc ^DnnlTensor a ^DnnlTensor y))
  (round [_ a y]
    (tensor-math MKL/vdRound ^DnnlTensor a ^DnnlTensor y))
  (modf [_ a y z]
    (tensor-math MKL/vdModf ^DnnlTensor a ^DnnlTensor y ^DnnlTensor z))
  (frac [_ a y]
    (tensor-math MKL/vdFrac ^DnnlTensor a ^DnnlTensor y))
  (fmin [_ a b y]
    (tensor-math MKL/vdFmin ^DnnlTensor a ^DnnlTensor b ^DnnlTensor y))
  (fmax [_ a b y]
    (tensor-math MKL/vdFmax ^DnnlTensor a ^DnnlTensor b ^DnnlTensor y))
  (copy-sign [_ a b y]
    (tensor-math MKL/vdCopySign ^DnnlTensor a ^DnnlTensor b ^DnnlTensor y))
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
    (check-contiguous)
    (let [view-x (view x)]
      (rand-uniform (neand/engine x) rng-stream lower upper x)))
  (rand-normal [_ rng-stream mu sigma x]
    (check-contiguous)
    (let [view-x (view x)]
      (rand-normal (neand/engine x) rng-stream mu sigma x))))

(deftype DnnlFactory [eng strm master tensor-engines]
  Releaseable
  (release [_]
    (when master
      (release strm)
      (release eng))
    true)
  DiamondFactoryProvider
  (diamond-factory [this]
    this)
  FlowProvider
  (flow [_]
    strm)
  ContextProvider
  (context [_]
    eng)
  NeanderthalFactoryProvider
  (neanderthal-factory [_ dtype]
    (factory-by-type dtype))
  TensorFactory
  (create-tensor-desc [this shape dtype format]
    (memory-desc shape dtype format))
  (create-tensor-desc [this tz-desc]
    (desc tz-desc))
  (create-tensor [this tensor-desc _]
    (dnnl-tensor this tensor-desc))
  (create-transformer [_ in-tz out-tz]
    (dnnl-transformer eng strm (view-tz in-tz) (view-tz out-tz)))
  (create-shuffler [_ src-tz dst-tz]
    (dnnl-shuffler eng strm (view-tz src-tz) (view-tz dst-tz)))
  (create-batcher [_ src-tz dst-tz mb-size]
    (dnnl-batcher eng strm (view-tz src-tz) (view-tz dst-tz) mb-size))
  (create-sum [_ scale dst]
    (dnnl-sum-blueprint eng strm scale dst))
  (create-sum [_ dst scale src scale-srcs]
    (dnnl-sum-blueprint eng strm dst scale src scale-srcs))
  (tensor-engine [this dtype]
    (get tensor-engines dtype nil))
  DnnFactory
  (activ-blueprint [this src-desc activ alpha beta]
    (dnnl-activ-blueprint this eng src-desc src-desc activ alpha beta))
  (inner-product-blueprint [this src-desc dst-desc weights-type]
    (dnnl-inner-product-blueprint this eng src-desc dst-desc weights-type))
  (fc-blueprint [this src-desc dst-desc activ alpha beta weights-type]
    (dnnl-fc-blueprint this eng src-desc dst-desc activ alpha beta weights-type))
  CostFactory
  (quadratic-cost [this prev-layer train-tz]
    (dnnl-universal-cost eng strm prev-layer train-tz quadratic-cost))
  (mean-absolute-cost [this prev-layer train-tz]
    (dnnl-universal-cost eng strm prev-layer train-tz mean-absolute-cost))
  (sigmoid-crossentropy-cost [this prev-layer train-tz]
    (dnnl-custom-cost eng strm prev-layer train-tz
                        (partial sigmoid-crossentropy-cost
                                 ((dims (output prev-layer)) 0)))))

(defn dnnl-factory
  ([eng strm]
   (->DnnlFactory eng strm false {:float (->FloatTensorEngine)}))
  ([]
   (let-release [eng (engine)
                 strm (stream eng)]
     (->DnnlFactory eng strm true {:float (->FloatTensorEngine)}))))
