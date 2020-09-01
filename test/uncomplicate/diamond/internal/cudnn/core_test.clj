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
             [core :refer [with-release]]
             [utils :refer [capacity direct-buffer put-float get-float]]]
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
           (get-activation-descriptor relu-desc) => {:mode :relu :relu-nan-opt true :coef 42.0})

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
           (get-activation-descriptor relu-desc) => {:mode :logistic :relu-nan-opt true :coef 42.0})

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
           => (map float [0.06337894 0.4683105 0.4683105 0.0024282578 0.017942535 0.9796292]))

    (facts "Softmax backward operation."
           (memcpy-host! host-dy gpu-dy)
           (softmax-backward cudnn-hdl :accurate :instance
                             (float 1.0) desc-x gpu-x desc-x gpu-dy (float 0.0) desc-x gpu-x)
           => cudnn-hdl
           (seq (memcpy-host! gpu-x (float-array 6)))
           => (map float [0.06337894 -0.5316895 0.4683105 0.0024282578 0.017942535 -0.020370794]))))

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
                 convo-fwd-algo (convolution-fwd-get-algo cudnn-hdl convo-desc desc-x filter-w desc-y)
                 activ-desc (activation-descriptor :relu false 1.0)
                 convo-fwd-ws (mem-alloc (convolution-fwd-get-workspace-size
                                          cudnn-hdl convo-desc convo-fwd-algo desc-x filter-w desc-y))]

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
                 convo-fwd-algo (convolution-fwd-get-algo cudnn-hdl convo-desc desc-x filter-w desc-y)

                 convo-bwd-data-algo (convolution-bwd-data-get-algo cudnn-hdl convo-desc
                                                                    filter-w desc-y desc-x)
                 convo-bwd-filter-algo (convolution-bwd-filter-get-algo cudnn-hdl convo-desc
                                                                        desc-x desc-y filter-w)
                 convo-ws (mem-alloc (max (long (convolution-fwd-get-workspace-size
                                                 cudnn-hdl convo-desc convo-fwd-algo desc-x filter-w desc-y))
                                          (long (convolution-bwd-data-get-workspace-size
                                                 cudnn-hdl convo-desc convo-bwd-data-algo
                                                 filter-w desc-y desc-x))
                                          (long (convolution-bwd-filter-get-workspace-size
                                                 cudnn-hdl convo-desc convo-bwd-filter-algo
                                                 desc-x desc-y filter-w))))]

    (memcpy-host! host-x gpu-x)
    (memcpy-host! host-w gpu-w)
    (facts "Convoluton forward operation."
           (convolution-fwd cudnn-hdl convo-desc convo-fwd-algo (float 1.0) desc-x gpu-x
                            filter-w gpu-w (float 0.0) desc-y gpu-y convo-ws) => cudnn-hdl
           (seq (memcpy-host! gpu-y (float-array 8)))
           => [18.0 -94.0 -21.0 -566.0 102.0 57.0 -78.0 -176.0])

    (memcpy-host! host-dy gpu-y)

    (facts "Convolution backward filter operation."
           (convolution-bwd-filter cudnn-hdl convo-desc convo-bwd-filter-algo
                                   (float 1.0) desc-x gpu-x desc-y gpu-y
                                   (float 0.0) filter-w gpu-dw convo-ws) => cudnn-hdl
           (map float (seq (memcpy-host! gpu-dw (float-array 9))))
           => (map float [251.9 230.9 93.6 217.0 186.0 233.0 81.0 198.6 415.0]))


    (facts "Convolution backward data operation."
           (convolution-bwd-data cudnn-hdl convo-desc convo-bwd-data-algo
                                 (float 1.0) filter-w gpu-w desc-y gpu-y
                                 (float 0.0) desc-x gpu-x convo-ws) => cudnn-hdl
           (map float (seq (memcpy-host! gpu-x (float-array 32))))
           => (map float [-0.40000004 -0.6 0.20000002 0.3 -1.6 -1.8 1.0999999 1.0 -0.20000005
                          0.100000024 0.39999995 -8.940697E-8 -0.8000001 -2.6 -2.0 0.0 -2.0 -2.0
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
