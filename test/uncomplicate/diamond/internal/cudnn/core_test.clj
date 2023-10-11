;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.diamond.internal.cudnn.core-test
  (:require [midje.sweet :refer [facts throws => roughly just truthy]]
            [uncomplicate.commons.core :refer [with-release release info bytesize size extract]]
            [uncomplicate.fluokitten.core :refer [fmap!]]
            [uncomplicate.clojure-cpp
             :refer [pointer int-pointer float-pointer pointer-seq get-entry null? fill!]]
            [uncomplicate.clojurecuda.core
             :refer [with-default default-stream mem-alloc-runtime memcpy-host! synchronize! memset!
                     cuda-malloc cuda-free! memcpy-to-host! memcpy-to-device!]]
            [uncomplicate.diamond.internal.cudnn
             [core :refer :all]
             [protocols :as api]]))

(with-default
  (with-release [cudnn-hdl (cudnn-context default-stream)
                 desc-x (tensor-descriptor [2 3 4 5] :float :nchw)
                 desc-y (tensor-descriptor [2 3 4 5] :float :nchw)
                 desc-z (tensor-descriptor [2 3 4 5] :int :nchw)]

    (facts "Tensor descriptor test."
           (= desc-x desc-x) => true
           (= desc-x desc-y) => true
           (= desc-x desc-z) => false
           (data-type desc-x) => :float)))

(with-default
  (with-release [cudnn-hdl (cudnn-context default-stream)
                 relu-desc (activation-descriptor :relu true 42.0)
                 desc-x (tensor-descriptor [2 3 4 5] :float :nchw)
                 host-x (float-pointer (range -2 70))
                 gpu-x (mem-alloc-runtime (bytesize desc-x))
                 gpu-dx (mem-alloc-runtime (bytesize desc-x))
                 host-y (float-pointer (range 120))
                 gpu-y (mem-alloc-runtime (bytesize desc-x))
                 host-dx (float-pointer (range -6 6 0.1))
                 gpu-dx (mem-alloc-runtime (bytesize desc-x))
                 host-dy (float-pointer (range -0.6 6 0.01))
                 gpu-dy (mem-alloc-runtime (bytesize desc-x))]

    (facts "ReLU Activation descriptor."
           (activation-descriptor relu-desc) => {:mode :relu :relu-nan-opt true :coef 42.0})

    (memcpy-host! host-x gpu-x)
    (memcpy-host! host-y gpu-y)

    (facts "Activation forward ReLU operation."
           (activation-forward cudnn-hdl relu-desc (float 3.0) desc-x (pointer gpu-x)
                               (float 2.0) desc-x (pointer gpu-y))
           => cudnn-hdl
           (memcpy-host! gpu-x host-x)
           (memcpy-host! gpu-y host-y)
           (take 5 (pointer-seq host-x)) => [-2.0 -1.0 0.0 1.0 2.0]
           (take 5 (pointer-seq host-y)) => [0.0 2.0 4.0 9.0 14.0])

    (facts "Activation backward ReLU operation."
           (memcpy-host! host-dx gpu-dx)
           (memcpy-host! host-dy gpu-dy)
           (activation-backward cudnn-hdl relu-desc (float 300.0) desc-x (pointer gpu-y)
                                desc-x (pointer gpu-dy)
                                desc-x (pointer gpu-x) (float 200.0) desc-x (pointer gpu-dx))
           => cudnn-hdl
           (memcpy-host! gpu-x host-x)
           (memcpy-host! gpu-y host-y)
           (memcpy-host! gpu-dx host-dx)
           (memcpy-host! gpu-dy host-dy)
           (take 5 (pointer-seq host-x)) => [-2.0 -1.0 0.0 1.0 2.0]
           (take 5 (pointer-seq host-y)) => [0.0 2.0 4.0 9.0 14.0]
           (take 5 (pointer-seq host-dx)) => [-1200.0 -1180.0 -1160.0 -1311.0 -1288.0]
           (take 5 (pointer-seq host-dy)) => (just [(roughly -0.6) (roughly -0.59) (roughly -0.58)
                                                    (roughly -0.57) (roughly -0.56)]))))

(with-default
  (with-release [cudnn-hdl (cudnn-context default-stream)
                 relu-desc (activation-descriptor :sigmoid true 42.0)
                 desc-x (tensor-descriptor [1 1 1 1] :float :nchw)
                 host-x (float-pointer [-0.5])
                 gpu-x (mem-alloc-runtime (bytesize desc-x))
                 gpu-dx (mem-alloc-runtime (bytesize desc-x))
                 gpu-y (mem-alloc-runtime (bytesize desc-x))
                 gpu-dx (mem-alloc-runtime (bytesize desc-x))
                 host-dy (float-pointer [-0.1])
                 gpu-dy (mem-alloc-runtime (bytesize desc-x))]

    (facts "Sigmoid Activation descriptor."
           (activation-descriptor relu-desc) => {:mode :logistic :relu-nan-opt true :coef 42.0})

    (memcpy-host! host-x gpu-x)

    (facts "Activation forward sigmoid operation."
           (activation-forward cudnn-hdl relu-desc (float 1.0) desc-x (pointer gpu-x)
                               (float 0.0) desc-x (pointer gpu-y))
           => cudnn-hdl
           (get-entry (memcpy-host! gpu-x (float-pointer 1)) 0) => -0.5
           (get-entry (memcpy-host! gpu-y (float-pointer 1)) 0) => (roughly 0.3775407))

    (facts "Activation backward sigmoid operation."
           (memcpy-host! host-dy gpu-dy)
           (get-entry (memcpy-host! gpu-dy (float-pointer 1)) 0) => (float -0.1)
           (activation-backward cudnn-hdl relu-desc (float 1.0) desc-x (pointer gpu-y)
                                desc-x (pointer gpu-dy)
                                desc-x (pointer gpu-x) (float 0.0) desc-x (pointer gpu-dx))
           => cudnn-hdl
           (get-entry (memcpy-host! gpu-x (float-pointer 1)) 0) => -0.5
           (get-entry (memcpy-host! gpu-y (float-pointer 1)) 0) => (roughly 0.3775407)
           (get-entry (memcpy-host! gpu-dx (float-pointer 1)) 0) => (roughly -0.02350037172436714)
           (get-entry (memcpy-host! gpu-dy (float-pointer 1)) 0) => (float -0.1))))

(with-default
  (with-release [cudnn-hdl (cudnn-context default-stream)
                 linear-desc (activation-descriptor :linear true 2.0)
                 desc-x (tensor-descriptor [1 1 1 1] :float [1 1 1 1])
                 host-x (float-pointer [3.0])
                 gpu-x (mem-alloc-runtime (bytesize desc-x))
                 host-y (float-pointer [50.0])
                 gpu-y (mem-alloc-runtime (bytesize desc-x))]

    (facts "Activation forward linear operation does not support forward and backward operations."
           (memcpy-host! host-x gpu-x)
           (memcpy-host! host-y gpu-y)
           (activation-forward cudnn-hdl linear-desc (float 3.0) desc-x (pointer gpu-x)
                               (float 2.0) desc-x (pointer gpu-y))
           => (throws clojure.lang.ExceptionInfo))))

(with-default
  (with-release [cudnn-hdl (cudnn-context default-stream)
                 add-desc (reduce-tensor-descriptor :add :float)
                 max-desc (reduce-tensor-descriptor :max :float)
                 mul-desc (reduce-tensor-descriptor :mul :float)
                 desc-x (tensor-descriptor [2 3 1 1] :float :nchw)
                 host-x (float-pointer [1 2 3 4 5 6])
                 gpu-x (cuda-malloc (bytesize desc-x))
                 desc-y (tensor-descriptor [1 1 1 1] :float :nchw)
                 host-y (float-pointer 1)
                 gpu-y (cuda-malloc (bytesize desc-x))]

    (memcpy-to-device! host-x gpu-x)
    (memset! gpu-y 0)

    (facts "Reduce tensor."
           (reduce-tensor cudnn-hdl add-desc (float 3.0) desc-x gpu-x (float 2.0) desc-y gpu-y)
           => cudnn-hdl
           (memcpy-to-host! gpu-x host-x)
           (memcpy-to-host! gpu-y host-y)
           (pointer-seq host-x) => [1.0 2.0 3.0 4.0 5.0 6.0]
           (get-entry host-y 0) => (* 3.0 (double (apply + (pointer-seq host-x))))
           (reduce-tensor cudnn-hdl max-desc (float 2.5) desc-x gpu-x (float 0.0) desc-y gpu-y)
           => cudnn-hdl
           (get-entry (memcpy-to-host! gpu-y host-y) 0) => (* 2.5 (double (apply max (pointer-seq host-x))))
           (reduce-tensor cudnn-hdl mul-desc (float 1.5) desc-x gpu-x (float 0.0) desc-y gpu-y)
           => cudnn-hdl
           (get-entry (memcpy-to-host! gpu-y host-y) 0) => (* 1.5 (double (apply * (pointer-seq host-x))))
           (null? (cuda-free! gpu-x)) => true
           (null? (cuda-free! gpu-y)) => true)))

(with-default
  (with-release [cudnn-hdl (cudnn-context default-stream)
                 desc-x (tensor-descriptor [2 3] :float :nchw)
                 host-x (float-pointer [1 3 3 2 4 8])
                 gpu-x (cuda-malloc (bytesize desc-x))
                 gpu-dx (cuda-malloc (bytesize desc-x))
                 gpu-y (cuda-malloc (bytesize desc-x))
                 host-dy (float-pointer [0 -2.135335400336505 0 0 0 -1.0207943791746268])
                 gpu-dy (cuda-malloc (bytesize desc-x))]

    (memcpy-to-device! host-x gpu-x)

    (facts "Softmax forward operation."
           (softmax-forward cudnn-hdl :accurate :instance
                            (float 1.0) desc-x gpu-x (float 0.0) desc-x gpu-x)
           => cudnn-hdl
           (pointer-seq (memcpy-to-host! gpu-x (float-pointer 6)))
           => (map float [0.06337894 0.4683105 0.4683105 0.002428258 0.017942535 0.9796292]))

    (facts "Softmax backward operation."
           (memcpy-to-device! host-dy gpu-dy)
           (softmax-backward cudnn-hdl :accurate :instance
                             (float 1.0) desc-x gpu-x desc-x gpu-dy (float 0.0) desc-x gpu-x)
           => cudnn-hdl
           (pointer-seq (memcpy-to-host! gpu-x (float-pointer 6)))
           => (map float [0.06337894 -0.5316895 0.4683105 0.002428258 0.017942535 -0.020370794])
           (null? (cuda-free! gpu-x)) => true
           (null? (cuda-free! gpu-y)) => true
           (null? (cuda-free! gpu-dx)) => true
           (null? (cuda-free! gpu-dy)) => true)))

(with-default
  (with-release [cudnn-hdl (cudnn-context default-stream)
                 desc-x (tensor-descriptor [2 1 4 4] :float :nchw)
                 host-x (float-pointer [0 43 3 30 0 98 0 0 7 38 0 0 19 20 175 50
                                      0 0 7 19 43 98 38 20 3 0 0 175 30 0 0 50])
                 gpu-x (cuda-malloc (bytesize desc-x))

                 desc-w (tensor-descriptor [1 1 3 3] :float :nchw)
                 filter-w (filter-descriptor [1 1 3 3] :float :nchw)
                 host-w (float-pointer [-2 0 1 0 1 0 -1 -2 0])
                 gpu-w (cuda-malloc (bytesize desc-w))

                 desc-bias (tensor-descriptor [1 1 1 1] :float :nchw)
                 gpu-bias (cuda-malloc (bytesize desc-bias))

                 desc-y (tensor-descriptor [2 1 2 2] :float :nchw)
                 gpu-y (cuda-malloc (bytesize desc-y))
                 gpu-z (cuda-malloc (bytesize desc-y))

                 convo-desc (convolution-descriptor :cross-correleation :float [0 0] [1 1] [1 1])
                 convo-fwd-algo-perf (convolution-fwd-find-algo cudnn-hdl convo-desc desc-x filter-w desc-y)
                 convo-fwd-algo (:algo convo-fwd-algo-perf)
                 activ-desc (activation-descriptor :relu false 1.0)
                 convo-fwd-ws (when (< 0 (long (:workspace-size convo-fwd-algo-perf)))
                                (cuda-malloc (:workspace-size convo-fwd-algo-perf)))]

    (memcpy-to-device! host-x gpu-x)
    (memcpy-to-device! host-w gpu-w)
    (memcpy-to-device! (float-pointer [0.5]) gpu-bias)
    (memcpy-to-device! (float-pointer (repeat 8 1.0)) gpu-z)

    (facts "Fused Convoluton ReLU forward operation."
           (convolution-fwd cudnn-hdl convo-desc convo-fwd-algo activ-desc (float 1.0) desc-x gpu-x
                            filter-w gpu-w (float 2.0) gpu-z desc-bias gpu-bias desc-y gpu-y convo-fwd-ws)
           => cudnn-hdl
           (pointer-seq (memcpy-to-host! gpu-y (float-pointer 8))) => [20.5 0.0 0.0 0.0 104.5 59.5 0.0 0.0]
           (pointer-seq (memcpy-to-host! gpu-z (float-pointer 8))) => (repeat 8 1.0)
           (mapv cuda-free! [gpu-x gpu-y gpu-z gpu-w convo-fwd-ws]) => truthy)))

(with-default
  (with-release [cudnn-hdl (cudnn-context default-stream)

                 desc-x (tensor-descriptor [2 1 4 4] :float :nchw)
                 gpu-x (mem-alloc-runtime (bytesize desc-x))
                 host-x (float-pointer [0 43 3 30 0 98 0 0 7 38 0 0 19 20 175 50
                                      0 0 7 19 43 98 38 20 3 0 0 175 30 0 0 50])

                 desc-w (tensor-descriptor [1 1 3 3] :float :nchw)
                 filter-w (filter-descriptor [1 1 3 3] :float :nchw)
                 gpu-w (mem-alloc-runtime (bytesize desc-w))
                 gpu-dw (mem-alloc-runtime (bytesize desc-w))
                 host-w (float-pointer [-2 0 1 0 1 0 -1 -2 0])

                 desc-y (tensor-descriptor [2 1 2 2] :float :nchw)
                 gpu-y (mem-alloc-runtime (bytesize desc-y))
                 host-dy (float-pointer [0.2 0.3 0.8 1 1 1 1 1])

                 convo-desc (convolution-descriptor :cross-correleation :float [0 0] [1 1] [1 1])
                 convo-fwd-algo-perf (convolution-fwd-find-algo cudnn-hdl convo-desc desc-x filter-w desc-y)
                 convo-fwd-algo (:algo convo-fwd-algo-perf)
                 convo-bwd-data-algo-perf (convolution-bwd-data-find-algo cudnn-hdl convo-desc
                                                                          filter-w desc-y desc-x)
                 convo-bwd-filter-algo-perf (convolution-bwd-filter-find-algo cudnn-hdl convo-desc
                                                                              desc-x desc-y filter-w)
                 convo-ws (mem-alloc-runtime (max 1
                                          (long (:workspace-size convo-fwd-algo-perf))
                                          (long (:workspace-size convo-bwd-data-algo-perf))
                                          (long (:workspace-size convo-bwd-filter-algo-perf))))]

    (memcpy-host! host-x gpu-x)
    (memcpy-host! host-w gpu-w)
    (facts "Convoluton forward operation."
           (convolution-fwd cudnn-hdl convo-desc convo-fwd-algo (float 1.0) desc-x (pointer gpu-x)
                            filter-w (pointer gpu-w) (float 0.0) desc-y (pointer gpu-y) (pointer convo-ws)) => cudnn-hdl
           (pointer-seq (memcpy-host! gpu-y (float-pointer 8)))
           => [18.0 -94.0 -21.0 -566.0 102.0 57.0 -78.0 -176.0])

    (memcpy-host! host-dy gpu-y)

    (facts "Convolution backward filter operation."
           (convolution-bwd-filter cudnn-hdl convo-desc (:algo convo-bwd-filter-algo-perf)
                                   (float 1.0) desc-x (pointer gpu-x) desc-y (pointer gpu-y)
                                   (float 0.0) filter-w (pointer gpu-dw) (pointer convo-ws)) => cudnn-hdl
           (map float (pointer-seq (memcpy-host! gpu-dw (float-pointer 9))))
           => (map float [251.9 230.9 93.6 217.0 186.0 233.0 81.0 198.6 415.0]))


    (facts "Convolution backward data operation."
           (convolution-bwd-data cudnn-hdl convo-desc (:algo convo-bwd-data-algo-perf)
                                 (float 1.0) filter-w (pointer gpu-w) desc-y (pointer gpu-y)
                                 (float 0.0) desc-x (pointer gpu-x) (pointer convo-ws)) => cudnn-hdl
           (map float (pointer-seq (memcpy-host! gpu-x (float-pointer 32))))
           => (map float [-0.4 -0.6 0.2 0.3 -1.6 -1.8 1.1 1.0 -0.2
                          0.099999994 0.39999998 0.0 -0.8 -2.6 -2.0 0.0 -2.0 -2.0
                          1.0 1.0 -2.0 -1.0 2.0 1.0 -1.0 -2.0 -1.0 0.0 -1.0 -3.0 -2.0 0.0]))))

(with-release [cudnn-hdl (cudnn-context default-stream)
               desc-x (tensor-descriptor [2 1 4 4] :float :nchw)
               gpu-x (mem-alloc-runtime (bytesize desc-x))
               host-x (float-pointer [0 43 3 30 0 98 0 0 7 38 0 0 19 20 175 50
                                    0 0 7 19 43 98 38 20 3 0 0 175 30 0 0 50])
               gpu-dx (mem-alloc-runtime (bytesize desc-x))
               desc-y (tensor-descriptor [2 1 2 2] :float :nchw)
               gpu-y (mem-alloc-runtime (bytesize desc-y))
               host-dy (float-pointer (repeat 8 2.0))
               gpu-dy (mem-alloc-runtime (bytesize desc-y))
               pool-desc (pooling-descriptor :max-deterministic [2 2] [2 2] [0 0])]

  (memcpy-host! host-x gpu-x)

  (facts "Max pooling forward."
         (pooling-forward cudnn-hdl pool-desc
                          (float 1.0) desc-x (pointer gpu-x) (float 0.0) desc-y (pointer gpu-y))
         (pointer-seq (memcpy-host! gpu-x (float-pointer 32))) => (pointer-seq host-x)
         (pointer-seq (memcpy-host! gpu-y (float-pointer 8))) => [98.0 30.0 38.0 175.0 98.0 38.0 30.0 175.0])

  (memcpy-host! host-dy gpu-dy)

  (facts "Max pooling backward."
         (pooling-backward cudnn-hdl pool-desc
                           (float 1.0) desc-y (pointer gpu-y) desc-y (pointer gpu-dy) desc-x (pointer gpu-x)
                           (float 0.0) desc-x (pointer gpu-x))
         (pointer-seq (memcpy-host! gpu-y (float-pointer 8))) => [98.0 30.0 38.0 175.0 98.0 38.0 30.0 175.0]
         (pointer-seq (memcpy-host! gpu-dy (float-pointer 8))) => (pointer-seq host-dy)
         (pointer-seq (memcpy-host! gpu-x (float-pointer 32)))
         => [0.0 0.0 0.0 2.0 0.0 2.0 0.0 0.0 0.0 2.0 0.0 0.0 0.0 0.0 2.0 0.0
             0.0 0.0 0.0 0.0 0.0 2.0 2.0 0.0 0.0 0.0 0.0 2.0 2.0 0.0 0.0 0.0]))

(defn population-variance [^long n ^double sample-variance]
  (/ (* sample-variance (dec n)) n))

(with-release [cudnn-hdl (cudnn-context default-stream)
               desc-x (tensor-descriptor [2 1 2 2] :float :nchw)
               gpu-x (mem-alloc-runtime (bytesize desc-x))
               gpu-y (mem-alloc-runtime (bytesize desc-x))
               host-x (float-pointer (range -1 7))
               desc-param (batch-norm-descriptor desc-x :spatial)
               gpu-gamma (mem-alloc-runtime (bytesize desc-param))
               gpu-beta (mem-alloc-runtime (bytesize desc-param))
               gpu-mean (mem-alloc-runtime (bytesize desc-param))
               gpu-var (mem-alloc-runtime (bytesize desc-param))
               gpu-saved-mean (mem-alloc-runtime (bytesize desc-param))
               gpu-saved-inv-var (mem-alloc-runtime (bytesize desc-param))
               host-gamma (float-pointer [0.5])
               host-beta (float-pointer [1.5])
               host-mean (float-pointer [2.5])
               host-var (float-pointer [(population-variance 8 6.0000005)])]

  (memcpy-host! host-x gpu-x)
  (memcpy-host! host-gamma gpu-gamma)
  (memcpy-host! host-beta gpu-beta)

  (facts "Batch normalization forward training."

         (batch-norm-fwd-training cudnn-hdl :spatial (float 1.0) (float 0.0)
                                  desc-x (pointer gpu-x) desc-x (pointer gpu-y) desc-param
                                  (pointer gpu-gamma) (pointer gpu-beta) 0 (pointer gpu-mean) (pointer gpu-var)
                                  (pointer gpu-saved-mean) (pointer gpu-saved-inv-var))
         (pointer-seq (memcpy-host! gpu-mean (float-pointer 1))) => (pointer-seq host-mean)
         (pointer-seq (memcpy-host! gpu-var (float-pointer 1))) => [(float 6.0000005)]
         (pointer-seq (memcpy-host! gpu-saved-mean (float-pointer 1))) => [2.5]
         (pointer-seq (memcpy-host! gpu-saved-inv-var (float-pointer 1))) => [(float 0.43643576)]
         (pointer-seq (memcpy-host! gpu-x (float-pointer 8))) => (pointer-seq host-x)
         (pointer-seq (memcpy-host! gpu-y (float-pointer 8))) => [0.7362374067306519 0.9544553160667419
                                                        1.172673225402832 1.3908910751342773
                                                        1.6091089248657227 1.827326774597168
                                                        2.0455446243286133 2.2637624740600586])
  (memcpy-host! host-var gpu-var);; Check slight difference between sample variance and population variance... 5.25 vs 6.0000005
  (facts "Batch normalization forward inference."
         (batch-norm-fwd-inference cudnn-hdl :spatial (float 1.0) (float 0.0)
                                   desc-x (pointer gpu-x) desc-x (pointer gpu-y) desc-param
                                   (pointer gpu-gamma) (pointer gpu-beta) (pointer gpu-mean) (pointer gpu-var))
         (pointer-seq (memcpy-host! gpu-x (float-pointer 8))) => (pointer-seq host-x)
         (pointer-seq (memcpy-host! gpu-y (float-pointer 8))) => [0.7362374067306519 0.9544553160667419
                                                        1.172673225402832 1.3908910751342773
                                                        1.6091089248657227 1.827326774597168
                                                        2.0455446243286133 2.2637624740600586]))

(with-release [cudnn-hdl (cudnn-context default-stream)
               desc-x (tensor-descriptor [1 2 2 2] :float :nchw)
               gpu-x (mem-alloc-runtime (bytesize desc-x))
               gpu-y (mem-alloc-runtime (bytesize desc-x))
               host-x (float-pointer (range -1 7))
               desc-param (batch-norm-descriptor desc-x :spatial)
               gpu-gamma (mem-alloc-runtime (bytesize desc-param))
               gpu-gamma-diff (mem-alloc-runtime (bytesize desc-param))
               gpu-beta (mem-alloc-runtime (bytesize desc-param))
               gpu-beta-diff (mem-alloc-runtime (bytesize desc-param))
               gpu-mean (mem-alloc-runtime (bytesize desc-param))
               gpu-var (mem-alloc-runtime (bytesize desc-param))
               gpu-saved-mean (mem-alloc-runtime (bytesize desc-param))
               gpu-saved-inv-var (mem-alloc-runtime (bytesize desc-param))
               host-gamma (float-pointer [0.5 1.5])
               host-beta (float-pointer [1 1])
               host-mean (float-pointer [0.5 4.5])
               host-var (float-pointer [1.6666667 1.6666667])
               host-diff (float-pointer [-5 10 0.3 0.2 -0.5 0.6 0.9 -3])]

  (memcpy-host! host-x gpu-x)
  (memcpy-host! host-gamma gpu-gamma)
  (memcpy-host! host-beta gpu-beta)

  (facts "Batch normalization forward."
         (batch-norm-fwd-training cudnn-hdl :spatial (float 1.0) (float 0.0)
                                  desc-x (pointer gpu-x) desc-x (pointer gpu-y) desc-param
                                  (pointer gpu-gamma) (pointer gpu-beta) 0 (pointer gpu-mean) (pointer gpu-var)
                                  (pointer gpu-saved-mean) (pointer gpu-saved-inv-var))
         (pointer-seq (memcpy-host! gpu-mean (float-pointer 2))) => (pointer-seq host-mean)
         (pointer-seq (memcpy-host! gpu-var (float-pointer 2))) => (pointer-seq host-var)
         (pointer-seq (memcpy-host! gpu-saved-mean (float-pointer 2))) => [0.5 4.5]
         (pointer-seq (memcpy-host! gpu-saved-inv-var (float-pointer 2))) => [(float 0.8944271) (float 0.8944271)]
         (pointer-seq (memcpy-host! gpu-x (float-pointer 8))) => (pointer-seq host-x)
         (pointer-seq (memcpy-host! gpu-y (float-pointer 8)))
         => (mapv float [0.32917967 0.77639323 1.2236068 1.6708204 -1.012461 0.32917976 1.6708205 3.0124612]))

  (facts "Batch normalization backward."
         (memcpy-host! host-diff gpu-y)

         (batch-norm-bwd cudnn-hdl :spatial (float 1.0) (float 0.0) (float -1.0) (float 1.0)
                         desc-x (pointer gpu-x) desc-x (pointer gpu-y) desc-x (pointer gpu-x) desc-param
                         (pointer gpu-gamma) (pointer gpu-gamma-diff) (pointer gpu-beta)
                         (pointer gpu-saved-mean) (pointer gpu-saved-inv-var))
         (pointer-seq (memcpy-host! gpu-saved-mean (float-pointer 2))) => [0.5 4.5]
         (pointer-seq (memcpy-host! gpu-saved-inv-var (float-pointer 2))) => [(float 0.8944271) (float 0.8944271)]
         (pointer-seq (memcpy-host! gpu-gamma-diff (float-pointer 2))) => [(float -2.63856) (float 3.2199378)]
         (pointer-seq (memcpy-host! gpu-beta (float-pointer 2))) => [-4.5 3.0]
         (pointer-seq (memcpy-host! gpu-x (float-pointer 8))) => (mapv float [-2.4552026 3.9891448 -0.6126826
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
    (with-release [cudnn-hdl (cudnn-context default-stream)

                   desc-x (tensor-descriptor src-dim :float :nchw)
                   gpu-x (mem-alloc-runtime (bytesize desc-x))
                   host-x (float-pointer [2 3 0.2 0.3])

                   ;;desc-h (tensor-descriptor src-iter-dim :float :nchw)
                   desc-h1 (tensor-descriptor [L N C] :float [(* N C) C 1])
                   gpu-hx (mem-alloc-runtime (bytesize desc-h1))
                   host-hx (float-pointer (apply * src-iter-dim))
                   gpu-hy (mem-alloc-runtime (bytesize desc-h1))

                   gpu-cx (mem-alloc-runtime (bytesize desc-h1))
                   gpu-cy (mem-alloc-runtime (bytesize desc-h1))

                   rnn-desc (rnn-descriptor :standard :relu :single :unidirectional :linear
                                            :float :float :default C C C L nil :padded-io-disabled)
                   weights-size (rnn-weights-space-size cudnn-hdl rnn-desc)
                   gpu-w (mem-alloc-runtime weights-size)
                   ;; desc-w (tensor-descriptor weights-dim :float weights-strides)
                   host-w (float-pointer [0.1 0.3 0.2 0.4 100 300 200 400
                                          0.3 0.5 0.4 0.6 0.01 0.03 0.02 0.04 0.3 0.7 1 2])
                   rnn-tn-desc (rnn-data-descriptor :float :seq-mayor-unpacked C (repeat N T) 0.0)
                   dev-seq-lengths (mem-alloc-runtime (* Integer/BYTES N) :int)
                   temp (rnn-temp-space-size cudnn-hdl rnn-desc rnn-tn-desc :inference)
                   work (mem-alloc-runtime (first temp))
                   reserve (mem-alloc-runtime (+ 2048 (long (second temp))))
                   weight-params-0 (rnn-weight-params cudnn-hdl rnn-desc 0 (pointer gpu-w) 0)
                   weight-iter-params-0 (rnn-weight-params cudnn-hdl rnn-desc 0 (pointer gpu-w) 1)
                   weight-params-1 (rnn-weight-params cudnn-hdl rnn-desc 1 (pointer gpu-w) 0)
                   weight-iter-params-1 (rnn-weight-params cudnn-hdl rnn-desc 1 (pointer gpu-w) 1)
                   gpu-y (mem-alloc-runtime (bytesize desc-x))]
      ;;(build-rnn-dynamic! cudnn-hdl rnn-desc N) ;;TODO it seems it's no longer supported, or JCuda didn't complain before
      (memcpy-host! host-x gpu-x)
      (memcpy-host! host-w gpu-w)
      (memcpy-host! (int-pointer (repeat N T)) dev-seq-lengths)

      (facts "CUDA RNN basic functionality."
             (let [rd (rnn-descriptor rnn-desc)]
               (dissoc rd :dropout) => {:algo :standard :aux-flags 0 :bias :single :data-type :float
                                        :direction :unidirectional :hidden-size C :input :linear
                                        :input-size C :layers L :math-prec :float :math-type :default
                                        :mode :relu :proj-size C}
               (release (:dropout rd)))
             (rnn-weights-space-size cudnn-hdl rnn-desc) => 80
             (rnn-temp-space-size cudnn-hdl rnn-desc rnn-tn-desc :inference) => [16777312 0]
             (map bytesize (take-nth 2 weight-params-0)) => [16 8]
             (map bytesize (take-nth 2 weight-iter-params-0)) => [16 0]
             (map bytesize (take-nth 2 weight-params-1)) => [16 8]
             (map bytesize (take-nth 2 weight-iter-params-1)) => [16 0]
             (pointer-seq (memcpy-host! gpu-x (fill! (float-pointer 5) 0))) => (map float [2.0 3.0 0.2 0.3 0.0])
             (pointer-seq (memcpy-host! gpu-w (float-pointer 20)))
             => (map float [0.1 0.3 0.2 0.4 100.0 300.0 200.0 400.0 ;; weights and weights-iter layer 0
                            0.3 0.5 0.4 0.6 0.01 0.03 0.02 0.04 ;; weights and weights-iter layer 1
                            0.3 0.7 1 2]) ;; bias layer 0 and 1
             ;;TODO
             ;;(pointer-seq (memcpy-host! (weight-params-0 1) (float-pointer 5))) => (map float [0.1 0.3 0.2 0.4 0.0])
             ;; (pointer-seq (memcpy-host! (weight-params-0 3) (float-pointer 3))) => (map float [0.3 0.7 0.0])
             ;; (pointer-seq (memcpy-host! (weight-iter-params-0 1) (float-pointer 5))) => (map float [100.0 300.0 200.0 400.0 0.0])
             ;; (pointer-seq (memcpy-host! (weight-iter-params-0 3) (float-pointer 5))) => (map float [0.0 0.0 0.0 0.0 0.0])
             ;; (pointer-seq (memcpy-host! (weight-params-1 1) (float-pointer 5))) => (map float [0.3 0.5 0.4 0.6 0.0])
             ;; (pointer-seq (memcpy-host! (weight-params-1 3) (float-pointer 3))) => (map float [1.0 2.0 0.0])
             ;; (pointer-seq (memcpy-host! (weight-iter-params-1 1) (float-pointer 5))) => (map float [0.01 0.03 0.02 0.04 0.0])
             ;; (pointer-seq (memcpy-host! (weight-iter-params-1 3) (float-pointer 5))) => (map float [0.0 0.0 0.0 0.0 0.0])

             (rnn-fwd cudnn-hdl rnn-desc :inference (pointer dev-seq-lengths) rnn-tn-desc (pointer gpu-x)
                      rnn-tn-desc (pointer gpu-y) desc-h1 (pointer gpu-hx) (pointer gpu-hy) desc-h1 nil nil
                      (pointer gpu-w) (pointer work) (pointer reserve)) => cudnn-hdl

             (pointer-seq (memcpy-host! gpu-y (float-pointer 5)))
             => (map float [2.5700002 3.9400000 850.6969 1054.8889 0.0])))))

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
    (with-release [cudnn-hdl (cudnn-context default-stream)

                   desc-x (tensor-descriptor src-dim :float :nchw)
                   gpu-x (mem-alloc-runtime (bytesize desc-x))
                   host-x (float-pointer [2 3 0.2 0.3])

                   desc-h1 (tensor-descriptor [L N C] :float [(* N C) C 1])
                   gpu-hx (mem-alloc-runtime (bytesize desc-h1))
                   host-hx (float-pointer (apply * src-iter-dim))
                   gpu-hy (mem-alloc-runtime (bytesize desc-h1))

                   rnn-desc (rnn-descriptor :standard :relu :single :unidirectional :linear
                                            :float :float :default C C C L nil :padded-io-enabled)
                   weights-size (rnn-weights-space-size cudnn-hdl rnn-desc)
                   gpu-w (mem-alloc-runtime weights-size)
                   ;; desc-w (tensor-descriptor weights-dim :float weights-strides)
                   host-w (float-pointer [0.1 0.3 0.2 0.4 100 300 200 400
                                          0.3 0.5 0.4 0.6 0.01 0.03 0.02 0.04 0.3 0.7 1 2])
                   rnn-tn-desc (rnn-data-descriptor :float :seq-mayor-unpacked C (repeat N T) 0.0)
                   dev-seq-lengths (mem-alloc-runtime (* Integer/BYTES N) :int)
                   temp (rnn-temp-space-size cudnn-hdl rnn-desc rnn-tn-desc :training)
                   work (mem-alloc-runtime (first temp))
                   reserve (mem-alloc-runtime (long (second temp)))
                   weight-params-0 (rnn-weight-params cudnn-hdl rnn-desc 0 (pointer gpu-w) 0)
                   weight-iter-params-0 (rnn-weight-params cudnn-hdl rnn-desc 0 (pointer gpu-w) 1)
                   weight-params-1 (rnn-weight-params cudnn-hdl rnn-desc 1 (pointer gpu-w) 0)
                   weight-iter-params-1 (rnn-weight-params cudnn-hdl rnn-desc 1 (pointer gpu-w) 1)
                   gpu-y (mem-alloc-runtime (bytesize desc-x))
                   gpu-dy (mem-alloc-runtime (bytesize desc-x))
                   host-dy (float-pointer [1.1 -2.2 3.3 -4.4])
                   gpu-dx (mem-alloc-runtime (bytesize desc-x))
                   gpu-dhy (mem-alloc-runtime (bytesize desc-h1))
                   host-dhy (float-pointer [-1 2 0.1 -0.2])
                   gpu-dhx (mem-alloc-runtime (bytesize desc-h1))]

      (memcpy-host! host-x gpu-x)
      (memcpy-host! host-w gpu-w)
      ;;(memcpy-host! host-hx gpu-hx)
      (memcpy-host! (int-pointer (repeat N T)) dev-seq-lengths)

      (facts "CUDA Vanilla RNN training."
             (let [rd (rnn-descriptor rnn-desc)]
               (dissoc rd :dropout) => {:algo :standard :aux-flags 1 :bias :single :data-type :float
                                        :direction :unidirectional :hidden-size C :input :linear
                                        :input-size C :layers L :math-prec :float :math-type :default
                                        :mode :relu :proj-size C}
               (release (:dropout rd)));;TODO perhaps this will no longer be needed since I switched it to info...
             (rnn-weights-space-size cudnn-hdl rnn-desc) => 80
             (rnn-temp-space-size cudnn-hdl rnn-desc rnn-tn-desc :training) => [16777472 64]
             (map bytesize (take-nth 2 weight-params-0)) => [16 8]
             (map bytesize (take-nth 2 weight-iter-params-0)) => [16 0]
             (map bytesize (take-nth 2 weight-params-1)) => [16 8]
             (map bytesize (take-nth 2 weight-iter-params-1)) => [16 0]
             (pointer-seq (memcpy-host! gpu-x (fill! (float-pointer 5) 0))) => (map float [2.0 3.0 0.2 0.3 0.0])
             (pointer-seq (memcpy-host! gpu-w (float-pointer 20)))
             => (map float [0.1 0.3 0.2 0.4 100.0 300.0 200.0 400.0 ;; weights and weights-iter layer 0
                            0.3 0.5 0.4 0.6 0.01 0.03 0.02 0.04 ;; weights and weights-iter layer 1
                            0.3 0.7 1 2]) ;; bias layer 0 and 1
             ;; TODO now I have to figure out why these adresses cause segfault (some JavaCPP/CUDA api shenanigans)
             ;; (pointer-seq (memcpy-host! (weight-params-0 1) (float-pointer 5))) => (map float [0.1 0.3 0.2 0.4 0.0])
             ;; (pointer-seq (memcpy-host! (weight-params-0 3) (float-pointer 3))) => (map float [0.3 0.7 0.0])
             ;; (pointer-seq (memcpy-host! (weight-iter-params-0 1) (float-pointer 5))) => (map float [100.0 300.0 200.0 400.0 0.0])
             ;; (pointer-seq (memcpy-host! (weight-iter-params-0 3) (float-pointer 5))) => (map float [0.0 0.0 0.0 0.0 0.0])
             ;; (pointer-seq (memcpy-host! (weight-params-1 1) (float-pointer 5))) => (map float [0.3 0.5 0.4 0.6 0.0])
             ;; (pointer-seq (memcpy-host! (weight-params-1 3) (float-pointer 3))) => (map float [1.0 2.0 0.0])
             ;; (pointer-seq (memcpy-host! (weight-iter-params-1 1) (float-pointer 5))) => (map float [0.01 0.03 0.02 0.04 0.0])
             ;; (pointer-seq (memcpy-host! (weight-iter-params-1 3) (float-pointer 5))) => (map float [0.0 0.0 0.0 0.0 0.0])

             (rnn-fwd cudnn-hdl rnn-desc :training (pointer dev-seq-lengths) rnn-tn-desc (pointer gpu-x)
                      rnn-tn-desc (pointer gpu-y) desc-h1 (pointer gpu-hx) (pointer gpu-hy) desc-h1 nil nil
                      (pointer gpu-w) (pointer work) (pointer reserve)) => cudnn-hdl
             (pointer-seq (memcpy-host! gpu-y (float-pointer 5)))
             => (map float [2.5700002 3.9400000 850.6969 1054.8889 0.0])
             (pointer-seq (memcpy-host! gpu-hy (float-pointer 5)))
             => (map float [830.41 1200.86 850.6969 1054.8889 0.0])

             (memcpy-host! host-dy gpu-dy)
             (memcpy-host! host-dhy gpu-dhy)

             (rnn-bwd-data cudnn-hdl rnn-desc (pointer dev-seq-lengths)
                           rnn-tn-desc (pointer gpu-y) (pointer gpu-dy)
                           rnn-tn-desc (pointer gpu-dx)
                           desc-h1 (pointer gpu-hx) (pointer gpu-dhy) (pointer gpu-dhx)
                           desc-h1 nil nil nil
                           (pointer gpu-w) (pointer work) (pointer reserve)) => cudnn-hdl

             (pointer-seq (memcpy-host! gpu-dx (float-pointer 5)))
             => (map float [-33.62967 -66.71936 0.0059999824 -0.17000008 0.0])
             (pointer-seq (memcpy-host! gpu-dhx (float-pointer 5)))
             => (map float [-33629.67 -66719.36 -0.035219997 -0.060019996 0.0])

             (pointer-seq (memcpy-host! gpu-y (float-pointer 5)))
             => (map float [2.5700002 3.9400000 850.6969 1054.8889 0.0])

             (rnn-bwd-weights cudnn-hdl rnn-desc :add (pointer dev-seq-lengths)
                              rnn-tn-desc (pointer gpu-x) desc-h1 (pointer gpu-hx) rnn-tn-desc (pointer gpu-y)
                              (pointer gpu-w) (pointer work) (pointer reserve)) => cudnn-hdl

             (pointer-seq (memcpy-host! gpu-w (float-pointer 21)))
             => (map float [10.535569 15.953354 -341.30847 -511.8627
                            97.4520034790039 295.8139953613281 201.3159942626953 402.1619873046875
                            2825.152587890625 4085.8203125 -3822.6806640625 -5528.6044921875
                            8.748000144958496 13.425999641418457 -11.802001 -18.083999633789062
                            3.8797843 -169.20824 5.442 -4.882 0.0])

             (pointer-seq (memcpy-host! gpu-y (float-pointer 5)))
             => (map float [2.5700002 3.9400000 850.6969 1054.8889 0.0])
             ))))

(with-default
  (let [T 2
        N 1
        C 2
        G 3
        L 2
        D 1
        src-dim [T N C]
        src-iter-dim [L D N C]
        weights-dim [L D C G C]
        weights-strides [(* 2 D C G C) (* C G C) (* C G) C 1]
        bias-dim [L D G C]]
    (with-release [cudnn-hdl (cudnn-context default-stream)

                   desc-x (tensor-descriptor src-dim :float :nchw)
                   gpu-x (mem-alloc-runtime (bytesize desc-x))
                   host-x (float-pointer [2 3 0.2 0.3])

                   desc-h1 (tensor-descriptor [L N C] :float [(* N C) C 1])
                   gpu-hx (mem-alloc-runtime (bytesize desc-h1))
                   host-hx (fill! (float-pointer (apply * src-iter-dim)) 0)
                   gpu-hy (mem-alloc-runtime (bytesize desc-h1))

                   rnn-desc (rnn-descriptor :standard :gru :single :unidirectional :linear
                                            :float :float :default C C C L nil :padded-io-enabled)
                   weights-size (rnn-weights-space-size cudnn-hdl rnn-desc)
                   gpu-w (mem-alloc-runtime weights-size)
                   ;; desc-w (tensor-descriptor weights-dim :float weights-strides)
                   host-w (float-pointer [0.111 0.211 0.112 0.212 0.121 0.221 0.122 0.222 0.131 0.231 0.132 0.232
                                        100 300 200 400 100 300 200 400 100 300 200 400
                                        0.311 0.411 0.312 0.412 0.321 0.421 0.322 0.422 0.331 0.431 0.332 0.432
                                        0.01 0.03 0.02 0.04 0.01 0.03 0.02 0.04 0.01 0.03 0.02 0.04
                                        0.3 0.7 0.3 0.7 0.3 0.7
                                        1 2 1 2 1 2])
                   rnn-tn-desc (rnn-data-descriptor :float :seq-mayor-unpacked C (repeat N T) 0.0)
                   dev-seq-lengths (mem-alloc-runtime (* Integer/BYTES N) :int)
                   temp (rnn-temp-space-size cudnn-hdl rnn-desc rnn-tn-desc :training)
                   work (mem-alloc-runtime (first temp))
                   reserve (mem-alloc-runtime (long (second temp)))
                   weight-params-00 (rnn-weight-params cudnn-hdl rnn-desc 0 (pointer gpu-w) 0)
                   weight-params-01 (rnn-weight-params cudnn-hdl rnn-desc 0 (pointer gpu-w) 1)
                   weight-params-02 (rnn-weight-params cudnn-hdl rnn-desc 0 (pointer gpu-w) 2)
                   weight-params-10 (rnn-weight-params cudnn-hdl rnn-desc 1 (pointer gpu-w) 0)
                   weight-params-11 (rnn-weight-params cudnn-hdl rnn-desc 1 (pointer gpu-w) 1)
                   weight-params-12 (rnn-weight-params cudnn-hdl rnn-desc 1 (pointer gpu-w) 2)
                   weight-params-1 (rnn-weight-params cudnn-hdl rnn-desc 1 (pointer gpu-w) 0)
                   weight-iter-params-00 (rnn-weight-params cudnn-hdl rnn-desc 0 (pointer gpu-w) 3)
                   weight-iter-params-10 (rnn-weight-params cudnn-hdl rnn-desc 1 (pointer gpu-w) 3)
                   gpu-y (mem-alloc-runtime (bytesize desc-x))
                   gpu-dy (mem-alloc-runtime (bytesize desc-x))
                   host-dy (float-pointer [1.1 -2.2 3.3 -4.4])
                   gpu-dx (mem-alloc-runtime (bytesize desc-x))
                   gpu-dhy (mem-alloc-runtime (bytesize desc-h1))
                   host-dhy (float-pointer [-1 2 0.1 -0.2])
                   gpu-dhx (mem-alloc-runtime (bytesize desc-h1))]
      (memcpy-host! host-x gpu-x)
      (memcpy-host! host-w gpu-w)
      (memcpy-host! host-hx gpu-hx)
      (memcpy-host! (int-pointer (repeat N T)) dev-seq-lengths)

      (facts "CUDA GRU training."
             (let [rd (rnn-descriptor rnn-desc)]
               (dissoc rd :dropout) => {:algo :standard :aux-flags 1 :bias :single :data-type :float
                                        :direction :unidirectional :hidden-size C :input :linear
                                        :input-size C :layers L :math-prec :float :math-type :default
                                        :mode :gru :proj-size C}
               (release (:dropout rd)))
             (rnn-weights-space-size cudnn-hdl rnn-desc) => 240
             (rnn-temp-space-size cudnn-hdl rnn-desc rnn-tn-desc :training) => [16777600 224]

             (pointer-seq (memcpy-host! gpu-x (fill! (float-pointer 5) 0))) => (map float [2.0 3.0 0.2 0.3 0.0])
             (pointer-seq (memcpy-host! gpu-w (fill! (float-pointer 61) 0)))
             => (map float [0.111 0.211 0.112 0.212 0.121 0.221 0.122 0.222 0.131 0.231 0.132 0.232
                            100 300 200 400 100 300 200 400 100 300 200 400
                            0.311 0.411 0.312 0.412 0.321 0.421 0.322 0.422 0.331 0.431 0.332 0.432
                            0.01 0.03 0.02 0.04 0.01 0.03 0.02 0.04 0.01 0.03 0.02 0.04
                            0.3 0.7 0.3 0.7 0.3 0.7
                            1 2 1 2 1 2 0])
             ;; (pointer-seq (memcpy-host! (weight-params-00 1) (float-pointer 5))) => (map float [0.111 0.211 0.112 0.212 0.0])
             ;; (pointer-seq (memcpy-host! (weight-params-01 1) (float-pointer 5))) => (map float [0.121 0.221 0.122 0.222 0.0])
             ;; (pointer-seq (memcpy-host! (weight-params-02 1) (float-pointer 5))) => (map float [0.131 0.231 0.132 0.232 0.0])
             ;; (pointer-seq (memcpy-host! (weight-params-00 3) (float-pointer 3))) => (map float [0.3 0.7 0.0])
             ;; (pointer-seq (memcpy-host! (weight-params-01 3) (float-pointer 3))) => (map float [0.3 0.7 0.0])
             ;; (pointer-seq (memcpy-host! (weight-params-02 3) (float-pointer 3))) => (map float [0.3 0.7 0.0])
             ;; (pointer-seq (memcpy-host! (weight-params-10 1) (float-pointer 5))) => (map float [0.311 0.411 0.312 0.412 0.0])
             ;; (pointer-seq (memcpy-host! (weight-params-11 1) (float-pointer 5))) => (map float [0.321 0.421 0.322 0.422 0.0])
             ;; (pointer-seq (memcpy-host! (weight-params-12 1) (float-pointer 5))) => (map float [0.331 0.431 0.332 0.432 0.0])
             ;; (pointer-seq (memcpy-host! (weight-params-10 3) (float-pointer 3))) => (map float [1 2 0.0])
             ;; (pointer-seq (memcpy-host! (weight-params-11 3) (float-pointer 3))) => (map float [1 2 0.0])
             ;; (pointer-seq (memcpy-host! (weight-params-12 3) (float-pointer 3))) => (map float [1 2 0.0])
             ;; (pointer-seq (memcpy-host! (weight-iter-params-00 1) (float-pointer 5))) => (map float [100.0 300.0 200.0 400.0 0.0])
             ;; (pointer-seq (memcpy-host! (weight-iter-params-00 3) (float-pointer 5))) => (map float [0.0 0.0 0.0 0.0 0.0])
             ;; (pointer-seq (memcpy-host! (weight-iter-params-10 1) (float-pointer 5))) => (map float [0.01 0.03 0.02 0.04 0.0])
             ;; (pointer-seq (memcpy-host! (weight-iter-params-10 3) (float-pointer 5))) => (map float [0.0 0.0 0.0 0.0 0.0])

             (rnn-fwd cudnn-hdl rnn-desc :training (pointer dev-seq-lengths) rnn-tn-desc (pointer gpu-x)
                      rnn-tn-desc (pointer gpu-y) desc-h1 (pointer gpu-hx) (pointer gpu-hy) desc-h1 nil nil
                      (pointer gpu-w) (pointer work) (pointer reserve)) => cudnn-hdl
             (pointer-seq (memcpy-host! gpu-y (float-pointer 5)))
             => (map float [0.1986464262008667 0.10329369455575943 0.3485546410083771 0.19498808681964874 0.0])
             (pointer-seq (memcpy-host! gpu-hy (float-pointer 5)))
             => (map float [0.20356373488903046 0.161529079079628 0.3485546410083771 0.19498808681964874 0.0])

             (memcpy-host! host-dy gpu-dy)
             (memcpy-host! host-dhy gpu-dhy)

             (rnn-bwd-data cudnn-hdl rnn-desc (pointer dev-seq-lengths)
                           rnn-tn-desc (pointer gpu-y) (pointer gpu-dy)
                           rnn-tn-desc (pointer gpu-dx)
                           desc-h1 (pointer gpu-hx) (pointer gpu-dhy) (pointer gpu-dhx)
                           desc-h1 nil nil nil
                           (pointer gpu-w) (pointer work) (pointer reserve)) => cudnn-hdl

             (pointer-seq (memcpy-host! gpu-dx (float-pointer 5)))
             => (map float [-0.019561385735869408 -0.036919500678777695 0.0 0.0 0.0])
             (pointer-seq (memcpy-host! gpu-dhx (float-pointer 5)))
             => (map float [-43.723384857177734 -75.57097625732422 2.7878310680389404 -5.621692180633545 0.0])

             (pointer-seq (memcpy-host! gpu-y (float-pointer 5)))
             => (map float [0.20008478 0.10380766 0.349865 0.19589604 0.0])

             (rnn-bwd-weights cudnn-hdl rnn-desc :add (pointer dev-seq-lengths)
                              rnn-tn-desc (pointer gpu-x) desc-h1 (pointer gpu-hx)
                              rnn-tn-desc (pointer gpu-y)
                              (pointer gpu-w) (pointer work) (pointer reserve)) => cudnn-hdl

             (pointer-seq (memcpy-host! gpu-w (float-pointer 61))) ;;TODO doesn't match DNNL, probably due to different layouts
             => (map float [0.111 0.211 0.112 0.212 0.36748418 0.59072626 -0.45590958 -0.6448644 0.026169404 0.07375412 0.23240864 0.38261294 100.0 300.0 200.0 400.0 100.0 300.0 200.0 400.0 100.0 300.0 200.0 400.0 0.31105167 0.41104087 0.31199607 0.41199687 0.13757727 0.27592492 0.51014656 0.57081133 0.4461669 0.5220893 0.3196769 0.42225328 0.0100523345 0.030027272 0.019996008 0.03999792 -0.06637962 -0.009802721 0.094934866 0.07904983 0.05212988 0.051954597 0.015350954 0.0375773 0.3 0.7 0.4232421 0.4110452 0.24758472 0.7502043 1.0002637 1.9999799 0.06381178 2.9602985 1.587811 1.937103 0.0])

             (pointer-seq (memcpy-host! gpu-y (float-pointer 5)))
             => (map float [0.20008478 0.10380766 0.349865 0.19589604 0.0])))))
