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
                     VectorMath rand-uniform rand-normal swap copy set-all]]
            [uncomplicate.neanderthal.internal.host.lapack :refer [with-lapack-check]]
            [uncomplicate.diamond.tensor
             :refer [*diamond-factory* view-tz output shape data-type layout]]
            [uncomplicate.diamond.internal
             [protocols
              :refer [TensorFactory DiamondFactoryProvider CostFactory DnnFactory
                      NeanderthalFactoryProvider diamond-factory]]
             [utils :refer [check-contiguous]]
             [cost :refer [quadratic-cost! mean-absolute-cost! crossentropy-cost!]]]
            [uncomplicate.diamond.internal.dnnl
             [protocols :refer [desc data DnnlEngineProvider]]
             [core :refer [memory-desc engine stream memory dims size]]
             [tensor :refer [dnnl-tensor dnnl-transformer dnnl-batcher dnnl-shuffler]]
             [directed :refer [dnnl-sum-blueprint dnnl-activ-blueprint
                               dnnl-inner-product-blueprint dnnl-fc-blueprint
                               dnnl-universal-cost dnnl-custom-cost
                               dnnl-convolution-layer-blueprint dnnl-pooling-blueprint
                               dnnl-gaussian-dropout-blueprint]]])
  (:import [uncomplicate.neanderthal.internal.host CBLAS LAPACK MKL]
           uncomplicate.neanderthal.internal.api.RealBufferAccessor
           uncomplicate.diamond.internal.dnnl.tensor.DnnlTensor))

(def ^{:private true :const true} INEFFICIENT_OPERATION_MSG
  "This operation would be inefficient because it does not use DNNL capabilities.
  Please use dedicated tensor operations.")

(def ^{:private true :const true} UNSUPPORTED_DATA_TYPE
  "The requested data type is not supported on the DNNL platform.
Please contribute towards making it possible, or use on of the supported types.")

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
    (tensor-amax CBLAS/isamax ^RealBufferAccessor (data-accessor (factory x)) ^DnnlTensor x))
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
    (tensor-math MKL/vsSqr ^DnnlTensor a ^DnnlTensor y))
  (mul [_ a b y]
    (tensor-math MKL/vsMul ^DnnlTensor a ^DnnlTensor b ^DnnlTensor y))
  (div [_ a b y]
    (tensor-math MKL/vsDiv ^DnnlTensor a ^DnnlTensor b ^DnnlTensor y))
  (inv [_ a y]
    (tensor-math MKL/vsInv ^DnnlTensor a ^DnnlTensor y))
  (abs [_ a y]
    (tensor-math MKL/vsAbs ^DnnlTensor a ^DnnlTensor y))
  (linear-frac [_ a b scalea shifta scaleb shiftb y]
    (tensor-linear-frac MKL/vsLinearFrac ^DnnlTensor a ^DnnlTensor b
                        scalea shifta scaleb shiftb ^DnnlTensor y))
  (fmod [_ a b y]
    (tensor-math MKL/vsFmod ^DnnlTensor a ^DnnlTensor b ^DnnlTensor y))
  (frem [_ a b y]
    (tensor-math MKL/vsRemainder ^DnnlTensor a ^DnnlTensor b ^DnnlTensor y))
  (sqrt [_ a y]
    (tensor-math MKL/vsSqrt ^DnnlTensor a ^DnnlTensor y))
  (inv-sqrt [_ a y]
    (tensor-math MKL/vsInvSqrt ^DnnlTensor a ^DnnlTensor y))
  (cbrt [_ a y]
    (tensor-math MKL/vsCbrt ^DnnlTensor a ^DnnlTensor y))
  (inv-cbrt [_ a y]
    (tensor-math MKL/vsInvCbrt ^DnnlTensor a ^DnnlTensor y))
  (pow2o3 [_ a y]
    (tensor-math MKL/vsPow2o3 ^DnnlTensor a ^DnnlTensor y))
  (pow3o2 [_ a y]
    (tensor-math MKL/vsPow3o2 ^DnnlTensor a ^DnnlTensor y))
  (pow [_ a b y]
    (tensor-math MKL/vsPow ^DnnlTensor a ^DnnlTensor b ^DnnlTensor y))
  (powx [_ a b y]
    (tensor-powx MKL/vsPowx ^DnnlTensor a b ^DnnlTensor y))
  (hypot [_ a b y]
    (tensor-math MKL/vsHypot ^DnnlTensor a ^DnnlTensor b ^DnnlTensor y))
  (exp [_ a y]
    (tensor-math MKL/vsExp ^DnnlTensor a ^DnnlTensor y))
  (expm1 [_ a y]
    (tensor-math MKL/vsExpm1 ^DnnlTensor a ^DnnlTensor y))
  (log [_ a y]
    (tensor-math MKL/vsLn ^DnnlTensor a ^DnnlTensor y))
  (log10 [_ a y]
    (tensor-math MKL/vsLog10 ^DnnlTensor a ^DnnlTensor y))
  (sin [_ a y]
    (tensor-math MKL/vsSin ^DnnlTensor a ^DnnlTensor y))
  (cos [_ a y]
    (tensor-math MKL/vsCos ^DnnlTensor a ^DnnlTensor y))
  (tan [_ a y]
    (tensor-math MKL/vsTan ^DnnlTensor a ^DnnlTensor y))
  (sincos [_ a y z]
    (tensor-math MKL/vsSinCos ^DnnlTensor a ^DnnlTensor y ^DnnlTensor z))
  (asin [_ a y]
    (tensor-math MKL/vsAsin ^DnnlTensor a ^DnnlTensor y))
  (acos [_ a y]
    (tensor-math MKL/vsAcos ^DnnlTensor a ^DnnlTensor y))
  (atan [_ a y]
    (tensor-math MKL/vsAtan ^DnnlTensor a ^DnnlTensor y))
  (atan2 [_ a b y]
    (tensor-math MKL/vsAtan2 ^DnnlTensor a ^DnnlTensor b ^DnnlTensor y))
  (sinh [_ a y]
    (tensor-math MKL/vsSinh ^DnnlTensor a ^DnnlTensor y))
  (cosh [_ a y]
    (tensor-math MKL/vsCosh ^DnnlTensor a ^DnnlTensor y))
  (tanh [_ a y]
    (tensor-math MKL/vsTanh ^DnnlTensor a ^DnnlTensor y))
  (asinh [_ a y]
    (tensor-math MKL/vsAsinh ^DnnlTensor a ^DnnlTensor y))
  (acosh [_ a y]
    (tensor-math MKL/vsAcosh ^DnnlTensor a ^DnnlTensor y))
  (atanh [_ a y]
    (tensor-math MKL/vsAtanh ^DnnlTensor a ^DnnlTensor y))
  (erf [_ a y]
    (tensor-math MKL/vsErf ^DnnlTensor a ^DnnlTensor y))
  (erfc [_ a y]
    (tensor-math MKL/vsErfc ^DnnlTensor a ^DnnlTensor y))
  (erf-inv [_ a y]
    (tensor-math MKL/vsErfInv ^DnnlTensor a ^DnnlTensor y))
  (erfc-inv [_ a y]
    (tensor-math MKL/vsErfcInv ^DnnlTensor a ^DnnlTensor y))
  (cdf-norm [_ a y]
    (tensor-math MKL/vsCdfNorm ^DnnlTensor a ^DnnlTensor y))
  (cdf-norm-inv [_ a y]
    (tensor-math MKL/vsCdfNormInv ^DnnlTensor a ^DnnlTensor y))
  (gamma [_ a y]
    (tensor-math MKL/vsGamma ^DnnlTensor a ^DnnlTensor y))
  (lgamma [_ a y]
    (tensor-math MKL/vsLGamma ^DnnlTensor a ^DnnlTensor y))
  (expint1 [_ a y]
    (tensor-math MKL/vsExpInt1 ^DnnlTensor a ^DnnlTensor y))
  (floor [_ a y]
    (tensor-math MKL/vsFloor ^DnnlTensor a ^DnnlTensor y))
  (fceil [_ a y]
    (tensor-math MKL/vsCeil ^DnnlTensor a ^DnnlTensor y))
  (trunc [_ a y]
    (tensor-math MKL/vsTrunc ^DnnlTensor a ^DnnlTensor y))
  (round [_ a y]
    (tensor-math MKL/vsRound ^DnnlTensor a ^DnnlTensor y))
  (modf [_ a y z]
    (tensor-math MKL/vsModf ^DnnlTensor a ^DnnlTensor y ^DnnlTensor z))
  (frac [_ a y]
    (tensor-math MKL/vsFrac ^DnnlTensor a ^DnnlTensor y))
  (fmin [_ a b y]
    (tensor-math MKL/vsFmin ^DnnlTensor a ^DnnlTensor b ^DnnlTensor y))
  (fmax [_ a b y]
    (tensor-math MKL/vsFmax ^DnnlTensor a ^DnnlTensor b ^DnnlTensor y))
  (copy-sign [_ a b y]
    (tensor-math MKL/vsCopySign ^DnnlTensor a ^DnnlTensor b ^DnnlTensor y))
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
    (check-contiguous x)
    (let [vx (view x)]
      (rand-uniform (neand/engine x) rng-stream lower upper vx))
    x)
  (rand-normal [_ rng-stream mu sigma x]
    (check-contiguous x)
    (let [vx (view x)]
      (rand-normal (neand/engine x) rng-stream mu sigma vx))
    x))

(deftype ViewTensorEngine []
  Blas
  (swap [_ x y]
    (let [vx (view x)]
      (swap (neand/engine vx) vx (view y)))
    x)
  (copy [_ x y]
    (let [vx (view x)]
      (copy (neand/engine vx) vx (view y)))
    y)
  BlasPlus
  (set-all [_ alpha x]
    (let [vx (view x)]
      (set-all (neand/engine vx) alpha (view x)))
    x))

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
  (native-diamond-factory [this]
    this)
  FlowProvider
  (flow [_]
    strm)
  DnnlEngineProvider
  (dnnl-engine [_]
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
  (create-sum [_ scale-src src scale-dst dst]
    (dnnl-sum-blueprint eng strm scale-src src scale-dst dst))
  (tensor-engine [this dtype]
    (or (get tensor-engines dtype)
        (dragan-says-ex UNSUPPORTED_DATA_TYPE {:data-type dtype})))
  DnnFactory
  (activ-blueprint [this src-desc activ alpha beta]
    (dnnl-activ-blueprint this eng src-desc activ alpha beta))
  (inner-product-blueprint [this src-desc dst-desc weights-type]
    (dnnl-inner-product-blueprint this eng src-desc dst-desc weights-type))
  (fc-blueprint [this src-desc dst-desc activ alpha beta weights-type]
    (dnnl-fc-blueprint this eng src-desc dst-desc activ alpha beta weights-type))
  (convolution-blueprint [this src-desc weights-desc dst-desc activ
                          strides padding-l padding-r alpha beta]
    (dnnl-convolution-layer-blueprint this eng src-desc weights-desc dst-desc activ
                                      strides padding-l padding-r alpha beta))
  (pooling-blueprint [this src-desc dst-desc algo strides kernel padding-l padding-r]
    (dnnl-pooling-blueprint this eng src-desc dst-desc algo
                            strides kernel padding-l padding-r))
  (gaussian-dropout-blueprint [this src-desc sd]
    (dnnl-gaussian-dropout-blueprint this src-desc sd))
  CostFactory
  (quadratic-cost [this prev-layer train-tz]
    (dnnl-universal-cost eng strm prev-layer train-tz quadratic-cost!))
  (mean-absolute-cost [this prev-layer train-tz]
    (dnnl-universal-cost eng strm prev-layer train-tz mean-absolute-cost!))
  (crossentropy-cost [this prev-layer train-tz]
    (dnnl-custom-cost eng strm prev-layer train-tz
                      (partial crossentropy-cost!
                               ((dims (output prev-layer)) 0)))))

(defn dnnl-factory
  ([eng strm]
   (let [view-engine (->ViewTensorEngine)]
     (->DnnlFactory eng strm false {:float (->FloatTensorEngine)
                                    :int view-engine
                                    :byte view-engine
                                    :uint8 view-engine})))
  ([]
   (let-release [eng (engine)
                 strm (stream eng)]
     (let [view-engine (->ViewTensorEngine)]
       (->DnnlFactory eng strm true {:float (->FloatTensorEngine)
                                     :int view-engine
                                     :byte view-engine
                                     :uint8 view-engine})))))
