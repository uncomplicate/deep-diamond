(ns uncomplicate.diamond.internal.dnnl.fully-connected-test
  (:require [midje.sweet :refer [facts throws => roughly]]
            [uncomplicate.commons [core :refer [with-release]]]
            [uncomplicate.neanderthal
             [core :refer [transfer! view native view-ge cols]]
             [real :refer [entry! entry]]
             [native :refer [fv]]
             [random :refer [rand-uniform!]]
             [math :as math]]
            [uncomplicate.diamond
             [tensor :refer [*diamond-factory* tensor connector transformer
                             desc revert shape input output view-tz shuffler]]
             [dnn :refer [weights bias sum activation inner-product fully-connected
                          network init! train cost sgd-train]]]
            [uncomplicate.diamond.internal.protocols
             :refer [diff-bias diff-weights forward backward layers]]
            [uncomplicate.diamond.internal.dnnl.factory :refer [dnnl-factory]])
  (:import clojure.lang.ExceptionInfo))

(defn test-sum [factory]
  (facts
   "Tensor sum test"
   (with-release [tz-x (tensor factory [2 3 4 5] :float :nchw)
                  tz-y (tensor factory [2 3 4 5] :float :nhwc)
                  sum-bp (sum factory tz-y 2.0 tz-x)
                  sum-xy (sum-bp tz-y tz-x)]
     (entry (native (transfer! (range) (view tz-x))) 119) => 119.0
     (entry (native (view tz-y)) 119) => 0.0
     (sum-xy) => tz-y
     (entry (native (view tz-y)) 119) => 238.0)))

(test-sum *diamond-factory*)

(facts "Activation tests"
       (with-release [fact (dnnl-factory)
                      src-tz (tensor fact [1 3 2 1] :float :nchw)
                      dst-tz (tensor fact [1 3 2 1] :float :nchw)
                      activ-bluep (activation fact src-tz :relu)
                      activ-infer (activ-bluep src-tz)
                      activ-train (activ-bluep src-tz dst-tz)]
         (transfer! [-0.5 0 0.2 1 0.3 -0.7] src-tz)
         (view (activ-infer)) => (fv 0 0 0.2 1.0 0.3 0)
         (view (input activ-infer)) => (fv 0 0 0.2 1.0 0.3 0)
         (view (output activ-infer)) => (fv 0 0 0.2 1.0 0.3 0)
         (transfer! [-0.5 0 0.2 1 0.3 -0.7] src-tz)
         (forward activ-train)
         (view (input activ-train)) => (fv -0.5 0 0.2 1 0.3 -0.7)
         (view (output activ-train)) => (fv 0 0 0.2 1.0 0.3 0)
         (transfer! [-0.1 0.1 1 2 7 -0.6] dst-tz)
         (backward activ-train)
         (view (output activ-train)) => (fv -0.1 0.1 1 2 7 -0.6)
         (view (input activ-train)) => (fv 0 0 1 2 7.0 0)))

(facts "Inner product tests."
       (with-release [fact (dnnl-factory)
                      src-tz (tensor fact [1 3 2 1] :float :nchw)
                      dst-tz (tensor fact [1 2] :float :nc)
                      ip (inner-product fact src-tz dst-tz)
                      ip-infer (ip src-tz)
                      ip-train (ip src-tz dst-tz true)]
         (transfer! [-0.5 0 0.2 1 0.3 -0.7] src-tz)
         (transfer! [-0.1 0.1 0.2 -0.7 -0.1 0.1 0.2 -0.7 -0.1 0.1 0.2 -0.7] (weights ip-infer))
         (transfer! [-0.1 0.2] (bias ip-infer))
         (view (ip-infer)) => (fv -0.81 0.72999996)
         (view (input ip-infer)) => (fv -0.5 0 0.2 1 0.3 -0.7)
         (view (output ip-infer)) => (fv -0.81 0.72999996)
         (transfer! [-0.1 0.1 0.2 -0.7 -0.1 0.1 0.2 -0.7 -0.1 0.1 0.2 -0.7] (weights ip-train))
         (transfer! [-0.1 0.2] (bias ip-train))
         (view (ip-train)) => (fv -0.81 0.72999996)
         (transfer! [-0.1 0.8299999594688416] (view (output ip-train)))
         (backward ip-train) => ip-train
         (view (diff-bias ip-train)) => (view (output ip-train))
         (view (diff-weights ip-train)) => (fv 0.05 0 -0.020000001 -0.1 -0.030000001 0.07
                                               -0.415 0.0 0.166 0.83 0.249 -0.581)))

(facts "Inner product backprop step by step."
       (with-release [fact (dnnl-factory)
                      src-tz (tensor fact [1 1] :float :nc)
                      dst-tz (tensor fact [1 1] :float :nc)
                      ip (inner-product fact src-tz dst-tz)
                      ip-infer (ip src-tz)
                      ip-train (ip src-tz dst-tz true)]
         (transfer! [-0.5] src-tz)
         (transfer! [-0.1] (weights ip-infer))
         (transfer! [ 0.2] (bias ip-infer))
         (view (ip-infer)) => (fv 0.25)
         (view (input ip-infer)) => (fv -0.5)
         (view (output ip-infer)) => (fv 0.25)
         (transfer! [-0.1] (weights ip-train))
         (transfer! [0.2] (bias ip-train))
         (view (ip-train)) => (fv 0.25)
         (transfer! [0.4] (view (output ip-train)))
         (backward ip-train) => ip-train
         (view (diff-bias ip-train)) => (view (output ip-train))
         (view (diff-weights ip-train)) => (fv -0.2)))

(facts "Fully connected inference layer"
       (with-release [fact (dnnl-factory)
                      input-tz (tensor fact [1 3 2 1] :float :nchw)
                      fc-bluep (fully-connected fact input-tz [1 2] :relu)
                      fc (fc-bluep input-tz)
                      connect-output (connector (output fc) (desc [1 2] :float :nc))]
         (transfer! [-0.5 0 0.2 1 0.3 -0.7] input-tz)
         (transfer! [-0.1 0.1 0.2 -0.7 -0.1 0.1 0.2 -0.7 -0.1 0.1 0.2 -0.7] (weights fc))
         (transfer! [-0.1 0.2] (bias fc))
         (view (output connect-output)) => (fv 0.0 0.0)
         (fc) => (output fc)
         (view (connect-output)) => (fv 0.0 0.72999996)))

(facts "Fully connected training layer"
       (with-release [fact (dnnl-factory)
                      input-tz (tensor fact [1 3 2 1] :float :nchw)
                      fc-bluep (fully-connected fact input-tz [1 2] :relu)
                      fc (fc-bluep input-tz false)
                      train-tz (tensor fact [1 2] :float :nc)
                      fc-output (cost fc train-tz)]
         (transfer! [-0.5 0 0.2 1 0.3 -0.7] input-tz)
         (transfer! [-0.1 0.1 0.2 -0.7 -0.1 0.1 0.2 -0.7 -0.1 0.1 0.2 -0.7] (weights fc))
         (transfer! [-0.1 0.2] (bias fc))
         (forward fc [nil 1 0 0 false]) => fc
         (view (output fc)) => (fv 0.0 0.7299999594688416)
         (forward fc-output) => fc-output
         (transfer! [-0.1 0.8299999594688416] (view train-tz))
         (backward fc-output)
         (backward fc) => fc
         (backward fc [nil 1 0 0 false]) => fc
         (view input-tz) => (fv -0.5 0 0.2 1.0 0.3 -0.69999999)
         (view (weights fc)) => (fv -0.1 0.1 0.2 -0.7 -0.1 0.1 0.2 -0.7 -0.1 0.1 0.2 -0.7)
         (view (bias fc)) => (fv -0.1 0.2)))

(facts "Fully connected layer step by step"
       (with-release [fact (dnnl-factory)
                      input-tz (tensor fact [1 1] :float :nc)
                      fc-bluep (fully-connected fact input-tz [1 1] :linear)
                      fc (fc-bluep input-tz false)
                      train-tz (tensor fact [1 1] :float :nc)
                      fc-output (cost fc train-tz)]
         (transfer! [-0.5] input-tz)
         (transfer! [-0.1] (weights fc))
         (transfer! [0.2] (bias fc))
         (forward fc [nil 1 0 0 false]) => fc
         (view (output fc)) => (fv 0.25)
         (forward fc-output) => fc-output
         (transfer! [-0.15] (view train-tz))
         (backward fc-output)
         (backward fc) => fc
         (view (output fc)) => (fv 0.4)
         (view input-tz) => (fv -0.5)
         (backward fc [nil 1 0 0 false]) => fc
         (view input-tz) => (fv -0.5)
         (view (weights fc)) => (fv 0.1)
         (view (bias fc)) => (fv -0.2)))

(facts "Fully connected, 2 layers step by step"
       (with-release [fact (dnnl-factory)
                      input-tz (tensor fact [2 1] :float :nc)
                      fc1-bluep (fully-connected fact input-tz [2 1] :linear)
                      fc2-bluep (fully-connected fact fc1-bluep [2 1] :linear)
                      fc1 (fc1-bluep input-tz true)
                      fc2 (fc2-bluep fc1 true)
                      train-tz (tensor fact [2 1] :float :nc)
                      fc-output (cost fc2 train-tz)]
         (transfer! [-0.5 -0.5] input-tz)
         (transfer! [-0.1 -0.1] (weights fc1))
         (transfer! [0.2 0.2] (bias fc1))
         (transfer! [0.8 0.8] (weights fc2))
         (transfer! [0.5 0.5] (bias fc2))
         (forward fc1 [nil 1 0 0 false]) => fc1
         (view (output fc1)) => (fv 0.25 0.25)
         (output fc1) => (input fc2)
         (forward fc2 [nil 1 0 0 false]) => fc2
         (view (output fc2)) => (fv 0.7 0.7)
         (forward fc-output) => fc-output
         (transfer! [0.25 0.25] (view train-tz))
         (backward fc-output)
         (backward fc2) => fc2
         (view (output fc2)) => (fv 0.45 0.45)
         (view (input fc2)) => (fv 0.25 0.25)
         (backward fc2 [nil 1 0 0 false]) => fc2
         (view (input fc2)) => (fv 0.35999998 0.35999998)
         (view (weights fc2)) => (fv 0.6875)
         (view (bias fc2)) => (fv 0.050000012)
         (backward fc1) = fc1
         (view (output fc1)) => (fv 0.35999998 0.35999998)
         (backward fc1 [nil 1 0 0 false]) => fc1
         (view (input fc1)) => (fv -0.036 -0.036)
         (view (weights fc1)) => (fv 0.07999999)
         (view (bias fc1)) => (fv -0.15999998)))

(facts "Sequential network"
       (with-release [fact (dnnl-factory)
                      input-tz (tensor fact [1 16] :float :nc)
                      train-tz (tensor fact [1 2] :float :nc)
                      net-bp (network fact input-tz
                                      [(fully-connected [1 64] :relu)
                                       (fully-connected [1 64] :relu)
                                       (fully-connected [1 2] :linear)])
                      net (init! (net-bp input-tz :sgd))
                      quad-cost (cost net train-tz :quadratic)]
         (transfer! (range 16) input-tz)
         (train net quad-cost 10 [0.01 0 0 false]) => (roughly 0.0 0.0001)))

(facts "Sequential network step by step"
       (with-release [fact (dnnl-factory)
                      input-tz (tensor fact [2 1] :float :nc)
                      train-tz (tensor fact [2 1] :float :nc)
                      net-bp (network fact input-tz
                                      [(fully-connected [2 1] :linear)
                                       (fully-connected [2 1] :linear)])
                      net (net-bp input-tz :sgd)
                      quad-cost (cost net train-tz :quadratic)]
         (transfer! [-0.5 -0.5] input-tz)
         (transfer! [-0.1 -0.1] (weights (first (layers net))))
         (transfer! [0.2 0.2] (bias (first (layers net))))
         (transfer! [0.8 0.8] (weights (second (layers net))))
         (transfer! [0.5 0.5] (bias (second (layers net))))
         (transfer! [0.25 0.25] train-tz)
         (train net quad-cost 1 [1 0 0 false]) => 0.056953115582683234
         (view (weights (first (layers net)))) => (fv 0.07999999)
         (view (bias (first (layers net)))) => (fv -0.15999998)
         (view (weights (second (layers net)))) => (fv 0.6875)
         (view (bias (second (layers net)))) => (fv 0.050000012)))

(facts "Quadratic cost"
       (with-release [fact (dnnl-factory)
                      input-tz (tensor fact [2 1] :float :nc)
                      train-tz (tensor fact [2 1] :float :nc)
                      net-bp (network fact input-tz
                                      [(fully-connected [2 1] :relu)
                                       (fully-connected [2 1] :linear)])
                      net (net-bp input-tz :sgd)
                      quad-cost (cost net train-tz :quadratic)]
         (transfer! [0.25 0.35] train-tz)
         (transfer! [0.4 -1.3] (output net))
         (quad-cost) => 0.6862499438341274
         (view (output net)) => (fv 0.15 -1.64999998)))

(facts "Sequential network"
       (with-release [fact (dnnl-factory)
                      input-tz (tensor fact [1 16] :float :nc)
                      train-tz (tensor fact [1 2] :float :nc)
                      net-bp (network fact input-tz
                                      [(fully-connected [1 64] :relu)
                                       (fully-connected [1 64] :relu)
                                       (fully-connected [1 2] :sigmoid)])
                      net (init! (net-bp input-tz :sgd))
                      quad-cost (cost net train-tz :sigmoid-crossentropy)]
         (transfer! (range 16) input-tz)
         (train net quad-cost 1000 [0.01 0 0 false]) => (roughly 0.0 0.1)))

(defn my-fn ^double [xs]
  (+ (math/sin (entry xs 0))
     (math/cos (entry xs 1))
     (math/tanh (entry xs 2))
     (math/sqr (entry xs 3))))

(facts "Gradient descent"
       (with-release [fact (dnnl-factory)
                      x-tz (tensor fact [10000 4] :float :nc)
                      y-tz (tensor fact [10000 1] :float :nc)
                      net-bp (network fact x-tz
                                      [(fully-connected [10000 64] :relu) ;;TODO implement (fc [64] :relu)
                                       (fully-connected [10000 64] :relu)
                                       (fully-connected [10000 1] :linear)])
                      net (init! (net-bp x-tz :sgd))
                      quad-cost (cost net y-tz :quadratic)]
         (rand-uniform! (view x-tz))
         (transfer! (map my-fn (cols (view-ge (view x-tz) 4 10000))) (view y-tz))
         (time (train net quad-cost 20 [0.003 0 0 false])) => (roughly 0.0 0.2)))

(facts "Stochastic gradient descent"
       (with-release [fact (dnnl-factory)
                      x-tz (tensor fact [10000 4] :float :nc)
                      x-mb-tz (tensor fact [100 4] :float :nc)
                      x-shuff (shuffler x-tz x-mb-tz)
                      y-tz (tensor fact [10000 1] :float :nc)
                      y-mb-tz (tensor fact [100 1] :float :nc)
                      y-shuff (shuffler y-tz y-mb-tz)
                      net-bp (network fact x-mb-tz
                                      [(fully-connected [100 64] :relu)
                                       (fully-connected [100 64] :relu)
                                       (fully-connected [100 1] :linear)])
                      net (init! (net-bp x-mb-tz :sgd))
                      quad-cost (cost net y-mb-tz :quadratic)]
         (rand-uniform! (view x-tz))
         (transfer! (map my-fn (cols (view-ge (view x-tz) 4 10000))) (view y-tz))
         (time (sgd-train net x-shuff y-shuff quad-cost 1 [0.01 0 0 false])) => (roughly 0.0 0.2)))

(facts "Adam gradient descent"
       (with-release [fact (dnnl-factory)
                      x-tz (tensor fact [10000 4] :float :nc)
                      x-mb-tz (tensor fact [100 4] :float :nc)
                      x-shuff (shuffler x-tz x-mb-tz)
                      y-tz (tensor fact [10000 1] :float :nc)
                      y-mb-tz (tensor fact [100 1] :float :nc)
                      y-shuff (shuffler y-tz y-mb-tz)
                      net-bp (network fact x-mb-tz
                                      [(fully-connected [100 64] :relu)
                                       (fully-connected [100 64] :relu)
                                       (fully-connected [100 1] :linear)])
                      net (init! (net-bp x-mb-tz :adam))
                      quad-cost (cost net y-mb-tz :quadratic)]
         (rand-uniform! (view x-tz))
         (transfer! (map my-fn (cols (view-ge (view x-tz) 4 10000))) (view y-tz))
         (time (sgd-train net x-shuff y-shuff quad-cost 1 [0.01])) => (roughly 0.0 0.2)))

#_(facts "Fully connected deep network"
       (with-release [input-tz (tensor [1024 1] :float :nc)
                      net-bp (network input-tz input-tz
                                      [(fully-connected [1024 1024] :relu)
                                       (fully-connected [1024 349] :logistic)
                                       (fully-connected [1024 4024] :tanh)
                                       (fully-connected [1024 1] :elu)])
                      net (init! (net-bp input-tz))]
         (time (dotimes [i 100]
                 (forward net [1 0 0 false])
                 (backward net [1 0 0 false])))))
