;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.diamond.internal.cudnn.core-test
  (:require [midje.sweet :refer [facts throws => roughly just]]
            [uncomplicate.commons
             [core :refer [with-release release info]]
             [utils :refer [capacity direct-buffer put-float get-float]]]
            [uncomplicate.fluokitten.core :refer [fmap!]]
            [uncomplicate.clojurecuda.core
             :refer [with-default default-stream mem-alloc memcpy-host! synchronize!]]
            [uncomplicate.diamond.internal.cudnn
             [core :refer :all]
             [protocols :as api]]))

(with-default
  (with-release [cudnn-hdl (cudnn-handle default-stream)
                 relu-desc (activation-descriptor :relu true 42.0)
                 desc-x (tensor-descriptor [2 3 4 5] :float :nchw)
                 host-x (float-array (range -2 70))
                 gpu-x (mem-alloc (size desc-x))
                 gpu-dx (mem-alloc (size desc-x))
                 host-y (float-array (range 120))
                 gpu-y (mem-alloc (size desc-x))
                 host-dx (float-array (range -6 6 0.1))
                 gpu-dx (mem-alloc (size desc-x))
                 host-dy (float-array (range -0.6 6 0.01))
                 gpu-dy (mem-alloc (size desc-x))]

    (facts "ReLU Activation descriptor."
           (activation-descriptor relu-desc) => {:mode :relu :relu-nan-opt true :coef 42.0})

    (memcpy-host! host-x gpu-x)
    (memcpy-host! host-y gpu-y)

    (facts "Activation forward ReLU operation."
           (activation-forward cudnn-hdl relu-desc (float 3.0) desc-x gpu-x (float 2.0) desc-x gpu-y)
           => cudnn-hdl
           (memcpy-host! gpu-x host-x)
           (memcpy-host! gpu-y host-y)
           (take 5 host-x) => [-2.0 -1.0 0.0 1.0 2.0]
           (take 5 host-y) => [0.0 2.0 4.0 9.0 14.0])

    (facts "Activation backward ReLU operation."
           (memcpy-host! host-dx gpu-dx)
           (memcpy-host! host-dy gpu-dy)
           (activation-backward cudnn-hdl relu-desc (float 300.0) desc-x gpu-y desc-x gpu-dy
                                desc-x gpu-x (float 200.0) desc-x gpu-dx)
           => cudnn-hdl
           (memcpy-host! gpu-x host-x)
           (memcpy-host! gpu-y host-y)
           (memcpy-host! gpu-dx host-dx)
           (memcpy-host! gpu-dy host-dy)
           (take 5 host-x) => [-2.0 -1.0 0.0 1.0 2.0]
           (take 5 host-y) => [0.0 2.0 4.0 9.0 14.0]
           (take 5 host-dx) => [-1200.0 -1180.0 -1160.0 -1311.0 -1288.0]
           (take 5 host-dy) => (just [(roughly -0.6) (roughly -0.59) (roughly -0.58)
                                      (roughly -0.57) (roughly -0.56)]))))

(with-default
  (with-release [cudnn-hdl (cudnn-handle default-stream)
                 relu-desc (activation-descriptor :sigmoid true 42.0)
                 desc-x (tensor-descriptor [1 1 1 1] :float :nchw)
                 host-x (float-array [-0.5])
                 gpu-x (mem-alloc (size desc-x))
                 gpu-dx (mem-alloc (size desc-x))
                 gpu-y (mem-alloc (size desc-x))
                 gpu-dx (mem-alloc (size desc-x))
                 host-dy (float-array [-0.1])
                 gpu-dy (mem-alloc (size desc-x))]

    (facts "Sigmoid Activation descriptor."
           (activation-descriptor relu-desc) => {:mode :logistic :relu-nan-opt true :coef 42.0})

    (memcpy-host! host-x gpu-x)

    (facts "Activation forward sigmoid operation."
           (activation-forward cudnn-hdl relu-desc (float 1.0) desc-x gpu-x (float 0.0) desc-x gpu-y)
           => cudnn-hdl
           (first (memcpy-host! gpu-x (float-array 1))) => -0.5
           (first (memcpy-host! gpu-y (float-array 1))) => (roughly 0.3775407))

    (facts "Activation backward sigmoid operation."
           (memcpy-host! host-dy gpu-dy)
           (first (memcpy-host! gpu-dy (float-array 1))) => (float -0.1)
           (activation-backward cudnn-hdl relu-desc (float 1.0) desc-x gpu-y desc-x gpu-dy
                                desc-x gpu-x (float 0.0) desc-x gpu-dx)
           => cudnn-hdl
           (first (memcpy-host! gpu-x (float-array 1))) => -0.5
           (first (memcpy-host! gpu-y (float-array 1))) => (roughly 0.3775407)
           (first (memcpy-host! gpu-dx (float-array 1))) => (roughly -0.02350037172436714)
           (first (memcpy-host! gpu-dy (float-array 1))) => (float -0.1))))

(with-default
  (with-release [cudnn-hdl (cudnn-handle default-stream)
                 linear-desc (activation-descriptor :linear true 2.0)
                 desc-x (tensor-descriptor [1 1 1 1] :float [1 1 1 1])
                 host-x (float-array [3.0])
                 gpu-x (mem-alloc (size desc-x))
                 host-y (float-array [50.0])
                 gpu-y (mem-alloc (size desc-x))]

    (facts "Activation forward linear operation does not support forward and backward operations."
           (memcpy-host! host-x gpu-x)
           (memcpy-host! host-y gpu-y)
           (activation-forward cudnn-hdl linear-desc (float 3.0) desc-x gpu-x (float 2.0) desc-x gpu-y)
           => (throws clojure.lang.ExceptionInfo))))

(with-default
  (with-release [cudnn-hdl (cudnn-handle default-stream)
                 add-desc (reduce-tensor-descriptor :add :float)
                 max-desc (reduce-tensor-descriptor :max :float)
                 mul-desc (reduce-tensor-descriptor :mul :float)
                 desc-x (tensor-descriptor [2 3 1 1] :float :nchw)
                 host-x (float-array [1 2 3 4 5 6])
                 gpu-x (mem-alloc (size desc-x))
                 desc-y (tensor-descriptor [1 1 1 1] :float :nchw)
                 host-y (float-array 1)
                 gpu-y (mem-alloc (size desc-x))]

    (memcpy-host! host-x gpu-x)
    (memcpy-host! host-y gpu-y)

    (facts "Reduce tensor."
           (reduce-tensor cudnn-hdl add-desc (float 3.0) desc-x gpu-x (float 2.0) desc-y gpu-y)
           => cudnn-hdl
           (memcpy-host! gpu-x host-x)
           (memcpy-host! gpu-y host-y)
           (seq host-x) => [1.0 2.0 3.0 4.0 5.0 6.0]
           (first host-y) => (* 3.0 (double (apply + host-x)))
           (reduce-tensor cudnn-hdl max-desc (float 2.5) desc-x gpu-x (float 0.0) desc-y gpu-y)
           => cudnn-hdl
           (first (memcpy-host! gpu-y host-y)) => (* 2.5 (double (apply max host-x)))
           (reduce-tensor cudnn-hdl mul-desc (float 1.5) desc-x gpu-x (float 0.0) desc-y gpu-y)
           => cudnn-hdl
           (first (memcpy-host! gpu-y host-y)) => (* 1.5 (double (apply * host-x))))))

(with-default
  (with-release [cudnn-hdl (cudnn-handle default-stream)
                 desc-x (tensor-descriptor [2 3] :float :nchw)
                 host-x (float-array [1 3 3 2 4 8])
                 gpu-x (mem-alloc (size desc-x))
                 gpu-dx (mem-alloc (size desc-x))
                 gpu-y (mem-alloc (size desc-x))
                 host-dy (float-array [0 -2.135335400336505 0 0 0 -1.0207943791746268])
                 gpu-dy (mem-alloc (size desc-x))]

    (memcpy-host! host-x gpu-x)

    (facts "Softmax forward operation."
           (softmax-forward cudnn-hdl :accurate :instance
                            (float 1.0) desc-x gpu-x (float 0.0) desc-x gpu-x)
           => cudnn-hdl
           (seq (memcpy-host! gpu-x (float-array 6)))
           => (map float [0.06337894 0.4683105 0.4683105 0.002428258 0.017942535 0.9796292]))

    (facts "Softmax backward operation."
           (memcpy-host! host-dy gpu-dy)
           (softmax-backward cudnn-hdl :accurate :instance
                             (float 1.0) desc-x gpu-x desc-x gpu-dy (float 0.0) desc-x gpu-x)
           => cudnn-hdl
           (seq (memcpy-host! gpu-x (float-array 6)))
           => (map float [0.06337894 -0.5316895 0.4683105 0.002428258 0.017942535 -0.020370794]))))

(with-default
  (with-release [cudnn-hdl (cudnn-handle default-stream)
                 desc-x (tensor-descriptor [2 1 4 4] :float :nchw)
                 host-x (float-array [0 43 3 30 0 98 0 0 7 38 0 0 19 20 175 50
                                      0 0 7 19 43 98 38 20 3 0 0 175 30 0 0 50])
                 gpu-x (mem-alloc (size desc-x))

                 desc-w (tensor-descriptor [1 1 3 3] :float :nchw)
                 filter-w (filter-descriptor [1 1 3 3] :float :nchw)
                 host-w (float-array [-2 0 1 0 1 0 -1 -2 0])
                 gpu-w (mem-alloc (size desc-w))

                 desc-bias (tensor-descriptor [1 1 1 1] :float :nchw)
                 gpu-bias (mem-alloc (size desc-bias))

                 desc-y (tensor-descriptor [2 1 2 2] :float :nchw)
                 gpu-y (mem-alloc (size desc-y))
                 gpu-z (mem-alloc (size desc-y))

                 convo-desc (convolution-descriptor :cross-correleation :float [0 0] [1 1] [1 1])
                 convo-fwd-algo-perf (convolution-fwd-find-algo cudnn-hdl convo-desc desc-x filter-w desc-y)
                 convo-fwd-algo (:algo convo-fwd-algo-perf)
                 activ-desc (activation-descriptor :relu false 1.0)
                 convo-fwd-ws (when (< 0 (long (:workspace-size convo-fwd-algo-perf)))
                                (mem-alloc (:workspace-size convo-fwd-algo-perf)))]

    (memcpy-host! host-x gpu-x)
    (memcpy-host! host-w gpu-w)
    (memcpy-host! (float-array [0.5]) gpu-bias)
    (memcpy-host! (float-array (repeat 8 1.0)) gpu-z)

    (facts "Fused Convoluton ReLU forward operation."
           (convolution-fwd cudnn-hdl convo-desc convo-fwd-algo activ-desc (float 1.0) desc-x gpu-x
                            filter-w gpu-w (float 2.0) gpu-z desc-bias gpu-bias desc-y gpu-y convo-fwd-ws)
           => cudnn-hdl
           (seq (memcpy-host! gpu-y (float-array 8))) => [20.5 0.0 0.0 0.0 104.5 59.5 0.0 0.0]
           (seq (memcpy-host! gpu-z (float-array 8))) => (repeat 8 1.0))))

(with-default
  (with-release [cudnn-hdl (cudnn-handle default-stream)

                 desc-x (tensor-descriptor [2 1 4 4] :float :nchw)
                 gpu-x (mem-alloc (size desc-x))
                 host-x (float-array [0 43 3 30 0 98 0 0 7 38 0 0 19 20 175 50
                                      0 0 7 19 43 98 38 20 3 0 0 175 30 0 0 50])

                 desc-w (tensor-descriptor [1 1 3 3] :float :nchw)
                 filter-w (filter-descriptor [1 1 3 3] :float :nchw)
                 gpu-w (mem-alloc (size desc-w))
                 gpu-dw (mem-alloc (size desc-w))
                 host-w (float-array [-2 0 1 0 1 0 -1 -2 0])

                 desc-y (tensor-descriptor [2 1 2 2] :float :nchw)
                 gpu-y (mem-alloc (size desc-y))
                 host-dy (float-array [0.2 0.3 0.8 1 1 1 1 1])

                 convo-desc (convolution-descriptor :cross-correleation :float [0 0] [1 1] [1 1])
                 convo-fwd-algo-perf (convolution-fwd-find-algo cudnn-hdl convo-desc desc-x filter-w desc-y)
                 convo-fwd-algo (:algo convo-fwd-algo-perf)
                 convo-bwd-data-algo-perf (convolution-bwd-data-find-algo cudnn-hdl convo-desc
                                                                          filter-w desc-y desc-x)
                 convo-bwd-filter-algo-perf (convolution-bwd-filter-find-algo cudnn-hdl convo-desc
                                                                              desc-x desc-y filter-w)
                 convo-ws (mem-alloc (max 1
                                          (long (:workspace-size convo-fwd-algo-perf))
                                          (long (:workspace-size convo-bwd-data-algo-perf))
                                          (long (:workspace-size convo-bwd-filter-algo-perf))))]

    (memcpy-host! host-x gpu-x)
    (memcpy-host! host-w gpu-w)
    (facts "Convoluton forward operation."
           (convolution-fwd cudnn-hdl convo-desc convo-fwd-algo (float 1.0) desc-x gpu-x
                            filter-w gpu-w (float 0.0) desc-y gpu-y convo-ws) => cudnn-hdl
           (seq (memcpy-host! gpu-y (float-array 8)))
           => [18.0 -94.0 -21.0 -566.0 102.0 57.0 -78.0 -176.0])

    (memcpy-host! host-dy gpu-y)

    (facts "Convolution backward filter operation."
           (convolution-bwd-filter cudnn-hdl convo-desc (:algo convo-bwd-filter-algo-perf)
                                   (float 1.0) desc-x gpu-x desc-y gpu-y
                                   (float 0.0) filter-w gpu-dw convo-ws) => cudnn-hdl
           (map float (seq (memcpy-host! gpu-dw (float-array 9))))
           => (map float [251.9 230.9 93.6 217.0 186.0 233.0 81.0 198.6 415.0]))


    (facts "Convolution backward data operation."
           (convolution-bwd-data cudnn-hdl convo-desc (:algo convo-bwd-data-algo-perf)
                                 (float 1.0) filter-w gpu-w desc-y gpu-y
                                 (float 0.0) desc-x gpu-x convo-ws) => cudnn-hdl
           (map float (seq (memcpy-host! gpu-x (float-array 32))))
           => (map float [-0.4 -0.6 0.2 0.3 -1.6 -1.8 1.1 1.0 -0.2
                          0.099999994 0.39999998 0.0 -0.8 -2.6 -2.0 0.0 -2.0 -2.0
                          1.0 1.0 -2.0 -1.0 2.0 1.0 -1.0 -2.0 -1.0 0.0 -1.0 -3.0 -2.0 0.0]))))

(with-release [cudnn-hdl (cudnn-handle default-stream)
               desc-x (tensor-descriptor [2 1 4 4] :float :nchw)
               gpu-x (mem-alloc (size desc-x))
               host-x (float-array [0 43 3 30 0 98 0 0 7 38 0 0 19 20 175 50
                                    0 0 7 19 43 98 38 20 3 0 0 175 30 0 0 50])
               gpu-dx (mem-alloc (size desc-x))
               desc-y (tensor-descriptor [2 1 2 2] :float :nchw)
               gpu-y (mem-alloc (size desc-y))
               host-dy (float-array (repeat 8 2.0))
               gpu-dy (mem-alloc (size desc-y))
               pool-desc (pooling-descriptor :max-deterministic [2 2] [2 2] [0 0])]

  (memcpy-host! host-x gpu-x)

  (facts "Max pooling forward."
         (pooling-forward cudnn-hdl pool-desc
                          (float 1.0) desc-x gpu-x (float 0.0) desc-y gpu-y)
         (seq (memcpy-host! gpu-x (float-array 32))) => (seq host-x)
         (seq (memcpy-host! gpu-y (float-array 8))) => [98.0 30.0 38.0 175.0 98.0 38.0 30.0 175.0])

  (memcpy-host! host-dy gpu-dy)

  (facts "Max pooling backward."
         (pooling-backward cudnn-hdl pool-desc
                           (float 1.0) desc-y gpu-y desc-y gpu-dy desc-x gpu-x
                           (float 0.0) desc-x gpu-x)
         (seq (memcpy-host! gpu-y (float-array 8))) => [98.0 30.0 38.0 175.0 98.0 38.0 30.0 175.0]
         (seq (memcpy-host! gpu-dy (float-array 8))) => (seq host-dy)
         (seq (memcpy-host! gpu-x (float-array 32)))
         => [0.0 0.0 0.0 2.0 0.0 2.0 0.0 0.0 0.0 2.0 0.0 0.0 0.0 0.0 2.0 0.0
             0.0 0.0 0.0 0.0 0.0 2.0 2.0 0.0 0.0 0.0 0.0 2.0 2.0 0.0 0.0 0.0]))

(defn population-variance [^long n ^double sample-variance]
  (/ (* sample-variance (dec n)) n))

(with-release [cudnn-hdl (cudnn-handle default-stream)
               desc-x (tensor-descriptor [2 1 2 2] :float :nchw)
               gpu-x (mem-alloc (size desc-x))
               gpu-y (mem-alloc (size desc-x))
               host-x (float-array (range -1 7))
               desc-param (batch-norm-descriptor desc-x :spatial)
               gpu-gamma (mem-alloc (size desc-param))
               gpu-beta (mem-alloc (size desc-param))
               gpu-mean (mem-alloc (size desc-param))
               gpu-var (mem-alloc (size desc-param))
               gpu-saved-mean (mem-alloc (size desc-param))
               gpu-saved-inv-var (mem-alloc (size desc-param))
               host-gamma (float-array [0.5])
               host-beta (float-array [1.5])
               host-mean (float-array [2.5])
               host-var (float-array [6.0000005])]

  (memcpy-host! host-x gpu-x)
  (memcpy-host! host-gamma gpu-gamma)
  (memcpy-host! host-beta gpu-beta)

  (facts "Batch normalization forward training."
         (batch-norm-fwd-training cudnn-hdl :spatial (float 1.0) (float 0.0)
                                  desc-x gpu-x desc-x gpu-y desc-param
                                  gpu-gamma gpu-beta 0 gpu-mean gpu-var
                                  gpu-saved-mean gpu-saved-inv-var)
         (seq (memcpy-host! gpu-mean (float-array 1))) => (seq host-mean)
         (seq (memcpy-host! gpu-var (float-array 1))) => (seq host-var)
         (seq (memcpy-host! gpu-saved-mean (float-array 1))) => [2.5]
         (seq (memcpy-host! gpu-saved-inv-var (float-array 1))) => [(float 0.43643576)]
         (seq (memcpy-host! gpu-x (float-array 8))) => (seq host-x)
         (seq (memcpy-host! gpu-y (float-array 8))) => [0.7362374067306519 0.9544553160667419
                                                        1.172673225402832 1.3908910751342773
                                                        1.6091089248657227 1.827326774597168
                                                        2.0455446243286133 2.2637624740600586])
  (fmap! (partial population-variance 8) host-var)
  (memcpy-host! host-var gpu-var)
  (facts "Batch normalization forward inference."
         (batch-norm-fwd-inference cudnn-hdl :spatial (float 1.0) (float 0.0)
                                   desc-x gpu-x desc-x gpu-y desc-param
                                   gpu-gamma gpu-beta gpu-mean gpu-var)
         (seq (memcpy-host! gpu-x (float-array 8))) => (seq host-x)
         (seq (memcpy-host! gpu-y (float-array 8))) => [0.7362374067306519 0.9544553160667419
                                                        1.172673225402832 1.3908910751342773
                                                        1.6091089248657227 1.827326774597168
                                                        2.0455446243286133 2.2637624740600586]))

(with-release [cudnn-hdl (cudnn-handle default-stream)
               desc-x (tensor-descriptor [1 2 2 2] :float :nchw)
               gpu-x (mem-alloc (size desc-x))
               gpu-y (mem-alloc (size desc-x))
               host-x (float-array (range -1 7))
               desc-param (batch-norm-descriptor desc-x :spatial)
               gpu-gamma (mem-alloc (size desc-param))
               gpu-gamma-diff (mem-alloc (size desc-param))
               gpu-beta (mem-alloc (size desc-param))
               gpu-beta-diff (mem-alloc (size desc-param))
               gpu-mean (mem-alloc (size desc-param))
               gpu-var (mem-alloc (size desc-param))
               gpu-saved-mean (mem-alloc (size desc-param))
               gpu-saved-inv-var (mem-alloc (size desc-param))
               host-gamma (float-array [0.5 1.5])
               host-beta (float-array [1 1])
               host-mean (float-array [0.5 4.5])
               host-var (float-array [1.6666667 1.6666667])
               host-diff (float-array [-5 10 0.3 0.2 -0.5 0.6 0.9 -3])]

  (memcpy-host! host-x gpu-x)
  (memcpy-host! host-gamma gpu-gamma)
  (memcpy-host! host-beta gpu-beta)

  (facts "Batch normalization forward."
         (batch-norm-fwd-training cudnn-hdl :spatial (float 1.0) (float 0.0)
                                  desc-x gpu-x desc-x gpu-y desc-param
                                  gpu-gamma gpu-beta 0 gpu-mean gpu-var
                                  gpu-saved-mean gpu-saved-inv-var)
         (seq (memcpy-host! gpu-mean (float-array 2))) => (seq host-mean)
         (seq (memcpy-host! gpu-var (float-array 2))) => (seq host-var)
         (seq (memcpy-host! gpu-saved-mean (float-array 2))) => [0.5 4.5]
         (seq (memcpy-host! gpu-saved-inv-var (float-array 2))) => [(float 0.8944271) (float 0.8944271)]
         (seq (memcpy-host! gpu-x (float-array 8))) => (seq host-x)
         (seq (memcpy-host! gpu-y (float-array 8)))
         => (mapv float [0.32917967 0.77639323 1.2236068 1.6708204 -1.012461 0.32917976 1.6708205 3.0124612]))

  (facts "Batch normalization backward."
         (memcpy-host! host-diff gpu-y)

         (batch-norm-bwd cudnn-hdl :spatial (float 1.0) (float 0.0) (float -1.0) (float 1.0)
                         desc-x gpu-x desc-x gpu-y desc-x gpu-x desc-param
                         gpu-gamma gpu-gamma-diff gpu-beta gpu-saved-mean gpu-saved-inv-var)
         (seq (memcpy-host! gpu-saved-mean (float-array 2))) => [0.5 4.5]
         (seq (memcpy-host! gpu-saved-inv-var (float-array 2))) => [(float 0.8944271) (float 0.8944271)]
         (seq (memcpy-host! gpu-gamma-diff (float-array 2))) => [(float -2.63856) (float 3.2199378)]
         (seq (memcpy-host! gpu-beta (float-array 2))) => [-4.5 3.0]
         (seq (memcpy-host! gpu-x (float-array 8))) => (mapv float [-2.4552026 3.9891448 -0.6126826
                                                                    -0.9212599 -1.4489719 0.9928142
                                                                    2.3612874 -1.9051301])))

(with-default
  (let [T 2
        N 1
        C 2
        G 1
        L 2
        D 1
        src-dim [T N C]
        src-iter-dim [L D N C]
        weights-dim [L D C G C]
        weights-strides [(* 2 D C G C) (* C G C) (* C G) C 1]
        bias-dim [L D G C]]
    (with-release [cudnn-hdl (cudnn-handle default-stream)

                   desc-x (tensor-descriptor src-dim :float :nchw)
                   gpu-x (mem-alloc (size desc-x))
                   host-x (float-array [2 3 0.2 0.3])


                   ;;desc-h (tensor-descriptor src-iter-dim :float :nchw)
                   desc-h1 (tensor-descriptor [L N C] :float [(* N C) C 1])
                   gpu-hx (mem-alloc (size desc-h1))
                   host-hx (float-array (apply * src-iter-dim))
                   gpu-hy (mem-alloc (size desc-h1))

                   gpu-cx (mem-alloc (size desc-h1))
                   gpu-cy (mem-alloc (size desc-h1))


                   rnn-desc (rnn-descriptor :standard :relu :single :unidirectional :linear
                                            :float :float :default C C C L nil :padded-io-disabled)
                   weights-size (rnn-weights-space-size cudnn-hdl rnn-desc)
                   gpu-w (mem-alloc weights-size)
                   desc-w (tensor-descriptor weights-dim :float weights-strides)
                   host-w (float-array [0.1 0.3 0.2 0.4 100 300 200 400 0.3 0.5 0.4 0.6 0.01 0.03 0.02 0.04 0.3 0.7 1 2])
                   rnn-tn-desc (rnn-data-descriptor :float :seq-mayor-packed C (repeat N T) 0.0)
                   temp (rnn-temp-space-size cudnn-hdl rnn-desc rnn-tn-desc :inference)
                   work (mem-alloc (first temp))
                   reserve (mem-alloc (+ 2048 (long (second temp))))
                   weight-params-0 (rnn-weight-params cudnn-hdl rnn-desc 0 gpu-w 0)
                   weight-iter-params-0 (rnn-weight-params cudnn-hdl rnn-desc 0 gpu-w 1)
                   weight-params-1 (rnn-weight-params cudnn-hdl rnn-desc 1 gpu-w 0)
                   weight-iter-params-1 (rnn-weight-params cudnn-hdl rnn-desc 1 gpu-w 1)
                   gpu-y (mem-alloc (size desc-x))
                   host-y (float-array 4)

                   ]
      (memcpy-host! host-x gpu-x)
      (memcpy-host! host-w gpu-w)
      (memcpy-host! host-y gpu-y)

      (facts "CUDA RNN basic functionality."
             (let [rd (rnn-descriptor rnn-desc)]
               (dissoc rd :dropout) => {:algo :standard :aux-flags 0 :bias :single :data-type :float
                                        :direction :unidirectional :hidden-size C :input :linear
                                        :input-size C :layers L :math-prec :float :math-type :default
                                        :mode :relu :proj-size C}
               (release (:dropout rd)))
             (rnn-weights-space-size cudnn-hdl rnn-desc) => 80
             (rnn-temp-space-size cudnn-hdl rnn-desc rnn-tn-desc :inference) => [16777312 0]
             (map size (take-nth 2 weight-params-0)) => [16 8]
             (map size (take-nth 2 weight-iter-params-0)) => [16 0]
             (map size (take-nth 2 weight-params-1)) => [16 8]
             (map size (take-nth 2 weight-iter-params-1)) => [16 0]
             (seq (memcpy-host! gpu-x (float-array 5))) => (map float [2.0 3.0 0.2 0.3 0.0])
             (seq (memcpy-host! gpu-w (float-array 20)))
             => (map float [0.1 0.3 0.2 0.4 100.0 300.0 200.0 400.0 ;; weights and weights-iter layer 0
                            0.3 0.5 0.4 0.6 0.01 0.03 0.02 0.04 ;; weights and weights-iter layer 1
                            0.3 0.7 1 2]) ;; bias layer 0 and 1
             (seq (memcpy-host! (weight-params-0 1) (float-array 5))) => (map float [0.1 0.3 0.2 0.4 0.0])
             (seq (memcpy-host! (weight-params-0 3) (float-array 3))) => (map float [0.3 0.7 0.0])
             (seq (memcpy-host! (weight-iter-params-0 1) (float-array 5))) => (map float [100.0 300.0 200.0 400.0 0.0])
             (seq (memcpy-host! (weight-iter-params-0 3) (float-array 5))) => (map float [0.0 0.0 0.0 0.0 0.0])
             (seq (memcpy-host! (weight-params-1 1) (float-array 5))) => (map float [0.3 0.5 0.4 0.6 0.0])
             (seq (memcpy-host! (weight-params-1 3) (float-array 3))) => (map float [1.0 2.0 0.0])
             (seq (memcpy-host! (weight-iter-params-1 1) (float-array 5))) => (map float [0.01 0.03 0.02 0.04 0.0])
             (seq (memcpy-host! (weight-iter-params-1 3) (float-array 5))) => (map float [0.0 0.0 0.0 0.0 0.0])

             (rnn-fwd cudnn-hdl rnn-desc :inference (repeat N T) rnn-tn-desc gpu-x
                      rnn-tn-desc gpu-y desc-h1 gpu-hx gpu-hy desc-h1 nil nil
                      gpu-w work reserve) => cudnn-hdl

             (synchronize! (get-cudnn-stream cudnn-hdl))

             (seq (memcpy-host! gpu-y (float-array 5)))
             => (map float [2.5700002 3.9400000 850.6969 1054.8889 0.0])

             )


      )))
