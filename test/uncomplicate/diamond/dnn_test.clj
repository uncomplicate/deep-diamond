;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.diamond.dnn-test
  (:require [midje.sweet :refer [facts throws => roughly]]
            [uncomplicate.commons [core :refer [with-release]]]
            [uncomplicate.neanderthal
             [core :refer [entry! entry native transfer! view vctr cols view-ge nrm2]]
             [random :refer [rand-uniform!]]
             [math :as math]]
            [uncomplicate.diamond
             [dnn :refer :all]
             [tensor :refer :all]
             [dnn-test :refer :all]]
            [uncomplicate.diamond.internal.protocols
             :refer [diff-bias diff-weights forward backward layers]]))

(defn test-sum [factory]
  (with-release [tz-x (tensor factory [2 3 4 5] :float :nchw)
                 tz-y (tensor factory [2 3 4 5] :float :nchw)
                 sum-bp (sum factory 2.0 tz-x 3.0 tz-y)
                 sum-xy (sum-bp tz-x tz-y)]
    (facts
     "Tensor sum test."
     (entry (native (transfer! (range) (view tz-x))) 119) => 119.0
     (entry (native (transfer! (range 0 10000 10) (view tz-y))) 119) => 1190.0
     (entry (native (view tz-x)) 1) => 1.0
     (entry (native (view tz-y)) 1) => 10.0
     (sum-xy) => tz-y
     (entry (native (view tz-x)) 1) => 1.0
     (entry (native (view tz-y)) 1) => 32.0
     (entry (native (view tz-x)) 119) => 119.0
     (entry (native (view tz-y)) 119) => 3808.0)))

(defn test-activation [fact]
  (with-release [src-tz (tensor fact [1 3 2 1] :float :nchw)
                 dst-tz (tensor fact [1 3 2 1] :float :nchw)
                 activ-bluep (activation fact src-tz :relu)
                 activ-infer (activ-bluep src-tz)
                 activ-train (activ-bluep src-tz dst-tz)]

    (transfer! [-0.5 0 0.2 1 0.3 -0.7] src-tz)

    (facts
     "Activation inference test."
     (view (activ-infer)) => (vctr src-tz [0 0 0.2 1.0 0.3 0])
     (view (input activ-infer)) => (vctr src-tz [0 0 0.2 1.0 0.3 0])
     (view (output activ-infer)) => (vctr src-tz [0 0 0.2 1.0 0.3 0]))

    (transfer! [-0.5 0 0.2 1 0.3 -0.7] src-tz)

    (facts
     "Activation forward test."
     (forward activ-train)
     (view (input activ-train)) => (vctr src-tz [-0.5 0 0.2 1 0.3 -0.7])
     (view (output activ-train)) => (vctr src-tz [0 0 0.2 1.0 0.3 0]))

    (facts
     "Activation backward test."
     (transfer! [-0.1 0.1 1 2 7 -0.6] dst-tz)
     (backward activ-train)
     (view (output activ-train)) => (vctr src-tz [-0.1 0.1 1 2 7 -0.6])
     (view (input activ-train)) => (vctr src-tz [0 0 1 2 7.0 0]))))

(defn test-fully-connected-inference [fact]
  (with-release [input-tz (tensor fact [1 3 2 1] :float :nchw)
                 fc-bluep (fully-connected fact input-tz [1 2] :relu)
                 fc (fc-bluep input-tz)
                 connect-output (connector (output fc) (desc [1 2] :float :nc))]
    (facts "Fully connected inference layer"
           (transfer! [-0.5 0 0.2 1 0.3 -0.7] input-tz)
           (transfer! [-0.1 0.1 0.2 -0.7 -0.1 0.1 0.2 -0.7 -0.1 0.1 0.2 -0.7] (weights fc))
           (transfer! [-0.1 0.2] (bias fc))
           (view (output connect-output)) => (vctr (output fc) [0.0 0.0])
           (fc) => (output fc)
           (view (output connect-output)) => (vctr (output fc) [0.0 0.72999996]))))

(defn test-fully-connected-transfer [fact]
  (with-release [input-tz (tensor fact [1 3 2 1] :float :nchw)
                 input-tz (tensor fact [1 3 2 1] :float :nchw)
                 fc-bluep (fully-connected fact input-tz [1 2] :relu)
                 fc (fc-bluep input-tz)
                 fc-1 (fc-bluep input-tz)]
    (facts "Inference layer transfer test."
           (transfer! [-0.1 0.1 0.2 -0.7 -0.1 0.1 0.2 -0.7 -0.1 0.1 0.2 -0.7] (weights fc))
           (transfer! [-0.1 0.2] (bias fc))
           (transfer! fc fc-1) => fc-1
           (bias fc-1) => (bias fc)
           (weights fc-1) => (weights fc))))

(defn test-fully-connected-training [fact]
  (with-release [input-tz (tensor fact [1 3 2 1] :float :nchw)
                 fc-bluep (fully-connected fact input-tz [1 2] :linear)
                 fc (fc-bluep input-tz false)
                 train-tz (tensor fact [1 2] :float :nc)
                 fc-output (cost fc train-tz)]
    (facts "Fully connected training layer"
           (transfer! [-0.5 0 0.2 1 0.3 -0.7] input-tz)
           (transfer! [-0.1 0.1 0.2 -0.7 -0.1 0.1 0.2 -0.7 -0.1 0.1 0.2 -0.7] (weights fc))
           (transfer! [-0.1 0.2] (bias fc))
           (forward fc [nil 1 0 0 false]) => fc
           (view (output fc)) => (vctr train-tz -0.8100000023841858 0.7299999594688416)
           (forward fc-output) => fc-output
           (transfer! [-0.71 -0.1] (view train-tz))
           (backward fc-output)
           (backward fc) => fc
           (backward fc [nil 1 0 0 false]) => fc
           (view input-tz) => (vctr train-tz -0.5 0 0.2 1.0 0.3 -0.69999999)
           (view (weights fc))
           => (vctr train-tz -0.15000000596046448 0.10000000149011612 0.2200000137090683
                    -0.5999999642372131 -0.06999999284744263 0.0299999862909317
                    0.6150000095367432 -0.699999988079071 -0.26600000262260437
                    -0.7299999594688416 -0.04899999499320984 -0.11900001764297485)
           (view (bias fc)) => (vctr train-tz 2.2351741790771484E-8 -0.6299999952316284))))

(defn test-fully-connected-training-adam [fact]
  (with-release [input-tz (tensor fact [1 3 2 1] :float :nchw)
                 fc-bluep (fully-connected fact input-tz [1 2] :linear)
                 fc (fc-bluep input-tz false :adam)
                 train-tz (tensor fact [1 2] :float :nc)
                 fc-output (cost fc train-tz)]
    (facts "Fully connected training layer"
           (transfer! [-0.5 0 0.2 1 0.3 -0.7] input-tz)
           (transfer! [-0.1 0.1 0.2 -0.7 -0.1 0.1 0.2 -0.7 -0.1 0.1 0.2 -0.7] (weights fc))
           (transfer! [-0.1 0.2] (bias fc))
           (forward fc []) => fc
           (view (output fc)) => (vctr train-tz -0.8100000023841858 0.7299999594688416)
           (forward fc-output) => fc-output
           (transfer! [-0.71 -0.1] (view train-tz))
           (let [reflection-warn *warn-on-reflection*]
             (set! *warn-on-reflection* true)
             (transfer! [0.1 0.3 -0.4 -0.2 0.2 0.3 -0.3 -0.1 -0.15 0.12 0.25 -0.25] (.s fc))
             (transfer! [0.01 0.03 -0.04 -0.02 0.02 0.03 -0.03 -0.01 -0.015 0.012 0.025 -0.025] (.s fc))
             (set! *warn-on-reflection* reflection-warn))
           (backward fc-output)
           (backward fc) => fc
           (backward fc [1 1]) => fc
           (view input-tz) => (vctr train-tz -0.5 0 0.2 1.0 0.3 -0.69999999)
           (nrm2 (view (weights fc))) => (float 149791.78)
           (view (bias fc)) => (vctr train-tz 2.2351741790771484E-8 -0.6299999952316284))))

(defn test-fully-connected-layer-1 [fact]
  (with-release [input-tz (tensor fact [1 1] :float :nc)
                 fc-bluep (fully-connected fact input-tz [1 1] :linear)
                 fc (fc-bluep input-tz false)
                 train-tz (tensor fact [1 1] :float :nc)
                 fc-output (cost fc train-tz)]
    (facts "Fully connected layer step by step"
           (transfer! [-0.5] input-tz)
           (transfer! [-0.1] (weights fc))
           (transfer! [0.2] (bias fc))
           (forward fc [nil 1 0 0 false]) => fc
           (entry (native (view (output fc))) 0) => 0.25
           (forward fc-output) => fc-output
           (transfer! [-0.15] (view train-tz))
           (backward fc-output)
           (backward fc) => fc
           (entry (native (view (output fc))) 0) => (roughly 0.4)
           (entry (native (view input-tz)) 0) => -0.5
           (backward fc [nil 1 0 0 false]) => fc
           (entry (native (view input-tz)) 0) => -0.5
           (entry (native (view (weights fc))) 0) => (roughly 0.1)
           (entry (native (view (bias fc))) 0) => (roughly -0.2))))

(defn test-fully-connected-layer-2 [fact]
  (with-release [input-tz (tensor fact [2 1] :float :nc)
                 fc1-bluep (fully-connected fact input-tz [2 1] :linear)
                 fc2-bluep (fully-connected fact fc1-bluep [2 1] :linear)
                 fc1 (fc1-bluep input-tz true)
                 fc2 (fc2-bluep fc1 true)
                 train-tz (tensor fact [2 1] :float :nc)
                 fc-output (cost fc2 train-tz)]
    (facts "Fully connected, 2 layers step by step."
           (transfer! [-0.5 -0.5] input-tz)
           (transfer! [-0.1 -0.1] (weights fc1))
           (transfer! [0.2 0.2] (bias fc1))
           (transfer! [0.8 0.8] (weights fc2))
           (transfer! [0.5 0.5] (bias fc2))
           (forward fc1 [nil 1 0 0 false]) => fc1
           (view (output fc1)) => (vctr train-tz 0.25 0.25)
           (output fc1) => (input fc2)
           (forward fc2 [nil 1 0 0 false]) => fc2
           (view (output fc2)) => (vctr train-tz 0.7 0.7)
           (forward fc-output) => fc-output
           (transfer! [0.25 0.25] (view train-tz))
           (backward fc-output)
           (backward fc2) => fc2
           (view (output fc2)) => (vctr train-tz 0.45 0.45)
           (view (input fc2)) => (vctr train-tz 0.25 0.25)
           (backward fc2 [nil 1 0 0 false]) => fc2
           (view (input fc2)) => (vctr train-tz 0.35999998 0.35999998)
           (view (weights fc2)) => (vctr train-tz [0.6875])
           (view (bias fc2)) => (vctr train-tz [0.050000012])
           (backward fc1) => fc1
           (view (output fc1)) => (vctr train-tz 0.35999998 0.35999998)
           (backward fc1 [nil 1 0 0 false]) => fc1
           (view (input fc1)) => (vctr train-tz -0.036 -0.036)
           (view (weights fc1)) => (vctr train-tz [0.07999999])
           (view (bias fc1)) => (vctr train-tz [-0.15999998]))))

(defn test-sequential-network-linear [fact]
  (with-release [input-tz (tensor fact [1 16] :float :nc)
                 train-tz (tensor fact [1 2] :float :nc)
                 net-bp (network fact input-tz
                                 [(fully-connected [64] :relu)
                                  (fully-connected [64] :relu)
                                  (fully-connected [2] :linear)])
                 net (init! (net-bp input-tz :sgd))
                 quad-cost (cost net train-tz :quadratic)]
    (facts "Sequential network with linear/quadratic cost."
           (transfer! (range 16) input-tz)
           (train net quad-cost 10 [0.01 0 0 false]) => (roughly 0.0 0.001))))

(defn test-sequential-network-detailed [fact]
  (with-release [input-tz (tensor fact [2 1] :float :nc)
                 train-tz (tensor fact [2 1] :float :nc)
                 net-bp (network fact input-tz
                                 [(fully-connected [1] :linear)
                                  (fully-connected [1] :linear)])
                 net (net-bp input-tz :sgd)
                 quad-cost (cost net train-tz :quadratic)]
    (facts "Sequential network step by step."
           (transfer! [-0.5 -0.5] input-tz)
           (transfer! [-0.1 -0.1] (weights (first (layers net))))
           (transfer! [0.2 0.2] (bias (first (layers net))))
           (transfer! [0.8 0.8] (weights (second (layers net))))
           (transfer! [0.5 0.5] (bias (second (layers net))))
           (transfer! [0.25 0.25] train-tz)
           (train net quad-cost 1 [1 0 0 false]) => 0.056953115582683234
           (entry (native (view (weights (first (layers net))))) 0) => (roughly 0.08)
           (entry (native (view (bias (first (layers net))))) 0) => (roughly -0.16)
           (entry (native (view (weights (second (layers net))))) 0) => 0.6875
           (entry (native (view (bias (second (layers net))))) 0) => (roughly 0.05))))

(defn test-sequential-network-batched [fact]
  (with-release [input-tz (tensor fact [4 2] :float :nc)
                 x-mb-tz (tensor fact [2 2] :float :nc)
                 x-batcher (batcher input-tz x-mb-tz)
                 train-tz (tensor fact [4 1] :float :nc)
                 y-mb-tz (tensor fact [2 1] :float :nc)
                 y-batcher (batcher train-tz y-mb-tz)
                 net-bp (network fact x-mb-tz
                                 [(fully-connected [1] :linear)
                                  (fully-connected [1] :linear)])
                 net (net-bp x-mb-tz :adam)
                 quad-cost (cost net y-mb-tz :quadratic)]
    (facts "Sequential network step by step."
           (transfer! [-0.5 -0.5 5 5 0.5 0.5 -5 -5] input-tz)
           (transfer! [-0.1 -0.1] (weights (first (layers net))))
           (transfer! [0.2 0.2] (bias (first (layers net))))
           (transfer! [0.8 0.8] (weights (second (layers net))))
           (transfer! [0.5 0.5] (bias (second (layers net))))
           (transfer! [0.25 0.25 2.5 2.5] train-tz)
           (train net x-batcher y-batcher quad-cost 2 [1]) => (roughly 5.5345)
           (entry (native (view (weights (first (layers net))))) 0) => (roughly 0.97108)
           (entry (native (view (bias (first (layers net))))) 0) => (roughly 1.0245)
           (entry (native (view (weights (second (layers net))))) 0) = (roughly 0.79295)
           (entry (native (view (bias (second (layers net))))) 0) => (roughly -1.38267))))

(defn test-quadratic-cost [fact]
  (with-release [input-tz (tensor fact [2 1] :float :nc)
                 train-tz (tensor fact [2 1] :float :nc)
                 net-bp (network fact input-tz
                                 [(fully-connected [1] :relu)
                                  (fully-connected [1] :linear)])
                 net (net-bp input-tz :sgd)
                 quad-cost (cost net train-tz :quadratic)]
    (facts "Quadratic cost."
           (transfer! [0.25 0.35] train-tz)
           (transfer! [0.4 -1.3] (output net))
           (quad-cost) => 0.6862499438341274
           (view (output net)) => (vctr train-tz 0.15 -1.64999998))))

(defn test-sequential-network-sigmoid-sgd [fact]
  (facts "Sequential network with sigmoid cross-entropy."
         (with-release [input-tz (tensor fact [1 16] :float :nc)
                        train-tz (tensor fact [1 2] :float :nc)
                        net-bp (network fact input-tz
                                        [(fully-connected [64] :relu)
                                         (fully-connected [64] :relu)
                                         (fully-connected [2] :sigmoid)])
                        net (init! (net-bp input-tz :sgd))
                        quad-cost (cost net train-tz :sigmoid-crossentropy)]
           (transfer! (range 16) input-tz)
           (transfer! [1 2] train-tz)
           (train net quad-cost 3 [0.01 0 0 false]) => (roughly 6 5))))

(defn test-sequential-network-sigmoid-adam [fact]
  (facts "Sequential network with sigmoid cross-entropy."
         (with-release [input-tz (tensor fact [1 16] :float :nc)
                        train-tz (tensor fact [1 2] :float :nc)
                        net-bp (network fact input-tz
                                        [(fully-connected [64] :relu)
                                         (fully-connected [64] :relu)
                                         (fully-connected [2] :sigmoid)])
                        net (init! (net-bp input-tz :adam))
                        quad-cost (cost net train-tz :sigmoid-crossentropy)]
           (transfer! (range 16) input-tz)
           (transfer! [1 2] train-tz)
           (train net quad-cost 3 []) => (roughly 6 5))))

(defn my-fn ^double [xs]
  (+ (math/sin (entry xs 0))
     (math/cos (entry xs 1))
     (math/tanh (entry xs 2))
     (math/sqr (entry xs 3))))

(defn test-gradient-descent [fact]
  (with-release [x-tz (tensor fact [10000 4] :float :nc)
                 y-tz (tensor fact [10000 1] :float :nc)
                 net-bp (network fact x-tz
                                 [(fully-connected [64] :relu)
                                  (fully-connected [64] :relu)
                                  (fully-connected [1] :linear)])
                 net (init! (net-bp x-tz :sgd))
                 quad-cost (cost net y-tz :quadratic)]
    (facts "Gradient descent."
           (rand-uniform! (view x-tz))
           (transfer! (map my-fn (cols (native (view-ge (view x-tz) 4 10000)))) (view y-tz))
           (time (train net quad-cost 30 [0.003 0 0 false])) => (roughly 0.0 0.2))))

(defn test-stochastic-gradient-descent-sgd [fact]
  (with-release [x-tz (tensor fact [10000 4] :float :nc)
                 x-mb-tz (tensor fact [100 4] :float :nc)
                 x-shuff (batcher x-tz x-mb-tz)
                 y-tz (tensor fact [10000 1] :float :nc)
                 y-mb-tz (tensor fact [100 1] :float :nc)
                 y-shuff (batcher y-tz y-mb-tz)
                 net-bp (network fact x-mb-tz
                                 [(fully-connected [64] :relu)
                                  (fully-connected [64] :relu)
                                  (fully-connected [1] :linear)])
                 net (init! (net-bp x-mb-tz :sgd))
                 quad-cost (cost net y-mb-tz :quadratic)]
    (facts "Vanilla stochastic gradient descent."
           (rand-uniform! (view x-tz))
           (transfer! (map my-fn (cols (native (view-ge (view x-tz) 4 10000)))) (view y-tz))
           (time (train net x-shuff y-shuff quad-cost 1 [0.01 0 0 false])) => (roughly 0.0 0.2))))

(defn test-stochastic-gradient-descent-adam [fact]
  (with-release [x-tz (tensor fact [10000 4] :float :nc)
                 x-mb-tz (tensor fact [100 4] :float :nc)
                 x-shuff (batcher x-tz x-mb-tz)
                 y-tz (tensor fact [10000 1] :float :nc)
                 y-mb-tz (tensor fact [100 1] :float :nc)
                 y-shuff (batcher y-tz y-mb-tz)
                 net-bp (network fact x-mb-tz
                                 [(fully-connected [64] :relu)
                                  (fully-connected [64] :relu)
                                  (fully-connected [1] :linear)])
                 net (init! (net-bp x-mb-tz :adam))
                 quad-cost (cost net y-mb-tz :quadratic)]
    (facts "Stochastic gradient descent with Adam."
           (rand-uniform! (view x-tz))
           (transfer! (map my-fn (cols (native (view-ge (view x-tz) 4 10000)))) (view y-tz))
           (time (train net x-shuff y-shuff quad-cost 1 [0.01])) => (roughly 0.0 0.01))))

(defn bench-wide-layers [fact]
  (with-release [input-tz (tensor fact [1024 1] :float :nc)
                 net-bp (network input-tz input-tz
                                 [(fully-connected [1024] :relu)
                                  (fully-connected [349] :logistic)
                                  (fully-connected [4024] :tanh)
                                  (fully-connected [1] :elu)])
                 net (init! (net-bp input-tz :sgd))]
    (time (do (dotimes [i 100]
                (forward net [0 1 0 0 false])
                (backward net [0 1 0 0 false]))
              (net)))))
