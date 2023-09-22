;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.diamond.dnn-test
  (:require [midje.sweet :refer [facts throws => roughly just]]
            [uncomplicate.commons [core :refer [with-release release]]] ;;TODO remove release
            [uncomplicate.neanderthal
             [core :refer [entry! entry native transfer! view-vctr vctr
                           cols view-ge nrm2 axpy!]]
             [random :refer [rand-uniform!]]
             [math :as math]
             [vect-math :refer [div!]]]
            [uncomplicate.diamond
             [dnn :refer :all]
             [tensor :refer :all]]
            [uncomplicate.diamond.internal.protocols
             :refer [diff-weights forward backward diff-input diff-output
                     weights bias *workspace* inf-ws-size train-ws-size create-workspace
                     parameters weights-layer weights-iter bias-layer]]))

(defn test-activation-relu [fact]
  (with-release [src-tz (tensor fact [1 3 2 1] :float :nchw)
                 activ-bluep (activation fact src-tz :relu)
                 activ-infer (activ-bluep src-tz)
                 activ-train (activ-bluep src-tz src-tz)]

    (transfer! [-0.5 0 0.2 1 0.3 -0.7] src-tz)

    (facts
     "Activation inference test."
     (view-vctr (activ-infer)) => (vctr src-tz [0 0 0.2 1.0 0.3 0])
     (view-vctr (input activ-infer)) => (vctr src-tz [0 0 0.2 1.0 0.3 0])
     (view-vctr (output activ-infer)) => (vctr src-tz [0 0 0.2 1.0 0.3 0]))

    (transfer! [-0.5 0 0.2 1 0.3 -0.7] src-tz)

    (facts
     "Activation forward test."
     (forward activ-train)
     (view-vctr (input activ-train)) => (vctr src-tz [-0.5 0 0.2 1 0.3 -0.7])
     (view-vctr (output activ-train)) => (vctr src-tz [0 0 0.2 1.0 0.3 0]))

    (facts
     "Activation backward test."
     (transfer! [-0.1 0.1 1 2 7 -0.6] (output activ-train))
     (backward activ-train)
     (view-vctr (output activ-train)) => (vctr src-tz [-0.1 0.1 1 2 7 -0.6])
     (view-vctr (input activ-train)) => (vctr src-tz [0 0 1 2 7.0 0]))))

(defn test-activation-sigmoid [fact]
  (with-release [src-tz (tensor fact [1 1 1 1] :float :nchw)
                 activ-bluep (activation fact src-tz :sigmoid)
                 activ-infer (activ-bluep src-tz)
                 activ-train (activ-bluep src-tz src-tz)]

    (transfer! [0.7] src-tz)

    (facts
     "Activation inference test."
     (first (native (view-vctr (activ-infer)))) => (roughly 0.6681877374649048)
     (first (native (view-vctr (input activ-infer)))) => (roughly 0.6681877374649048)
     (first (native (view-vctr (output activ-infer)))) => (roughly 0.6681877374649048))

    (transfer! [-0.5] src-tz)

    (facts
     "Activation forward test."
     (forward activ-train)
     (view-vctr (input activ-train)) => (vctr src-tz [-0.5])
     (entry (native (view-vctr (output activ-train))) 0) => (roughly 0.3775407))

    (facts
     "Activation backward test."
     (transfer! [-0.1] (diff-input activ-train))
     (backward activ-train)
     (view-vctr (diff-input activ-train)) => (vctr src-tz [-0.10000000149011612])
     (view-vctr (input activ-train)) => (vctr src-tz [-0.02350037172436714]))))

(defn test-fully-connected-inference [fact]
  (with-release [input-tz (tensor fact [1 3 2 1] :float :nchw)
                 fc-bluep (fully-connected fact input-tz [1 2] :relu)
                 fc (fc-bluep input-tz)
                 connect-output (connector (output fc) (desc [1 2] :float :nc))]
    (facts "Fully connected inference layer"
           (transfer! [-0.5 0 0.2 1 0.3 -0.7] input-tz)
           (transfer! [-0.1 0.1 0.2 -0.7 -0.1 0.1 0.2 -0.7 -0.1 0.1 0.2 -0.7] (weights fc))
           (transfer! [-0.1 0.2] (bias fc))
           (fc) => (output fc)
           (view-vctr (output connect-output)) => (vctr (output fc) [0.0 0.72999996]))))

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

(defn test-inner-product-training [fact]
  (with-release [input-tz (tensor fact [1 3 2 1] :float :nchw)
                 ip-bluep (inner-product fact input-tz [1 2])
                 dst-tz (tensor fact [1 2] :float :nc)
                 ip (ip-bluep input-tz dst-tz false false)]
    (facts "Inner product training operation."
           (transfer! [-0.5 0 0.2 1 0.3 -0.7] input-tz)
           (transfer! [-0.1 0.1 0.2 -0.7 -0.1 0.1 0.2 -0.7 -0.1 0.1 0.2 -0.7] (weights ip))
           (transfer! [-0.1 0.2] (bias ip))
           (forward ip) => ip
           (view-vctr (output ip)) => (vctr input-tz -0.8100000023841858 0.7299999594688416)
           (transfer! [-0.71 -0.1] (diff-input ip))
           (backward ip) => ip
           (seq (native input-tz)) => (map float [-0.5 0 0.2 1.0 0.3 -0.69999999])
           (seq (native (weights ip)))
           => (map float [-0.1 0.1 0.2 -0.7 -0.1 0.1 0.2 -0.7 -0.1 0.1 0.2 -0.7])
           (seq (native (diff-weights ip)))
           => (map float [0.355 0.0 -0.142 -0.71
                          -0.213 0.49699998 0.05 0.0
                          -0.0200000014 -0.1 -0.030000001 0.07])
           (seq (native (bias ip))) => (map float [-0.71 -0.1]))))

(defn test-fully-connected-training [fact]
  (with-release [input-tz (tensor fact [1 3 2 1] :float :nchw)
                 fc-bluep (fully-connected fact input-tz [1 2] :linear)
                 fc (fc-bluep input-tz false)
                 train-tz (tensor fact [1 2] :float :nc)
                 fc-output (cost fc train-tz :quadratic)]
    (facts "Fully connected training layer"
           (transfer! [-0.5 0 0.2 1 0.3 -0.7] input-tz)
           (transfer! [-0.1 0.1 0.2 -0.7 -0.1 0.1 0.2 -0.7 -0.1 0.1 0.2 -0.7] (weights fc))
           (transfer! [-0.1 0.2] (bias fc))
           (forward fc [nil 1 0 0 false]) => fc
           (nrm2 (axpy! -1 (view-vctr (output fc))
                        (vctr train-tz -0.8100000023841858 0.7299999594688416))) => (roughly 0 0.0000001)
           (forward fc-output) => fc-output
           (transfer! [-0.71 -0.1] (view-vctr train-tz))
           (backward fc-output)
           (backward fc) => fc
           (backward fc [nil 1 0 0 false]) => fc
           (view-vctr input-tz) => (vctr train-tz -0.5 0 0.2 1.0 0.3 -0.69999999)
           (nrm2 (axpy! -1 (view-vctr (weights fc))
                        (vctr train-tz -0.15000000596046448 0.10000000149011612 0.2200000137090683
                              -0.5999999642372131 -0.06999999284744263 0.0299999862909317
                              0.6150000095367432 -0.699999988079071 -0.26600000262260437
                              -0.7299999594688416 -0.04899999499320984 -0.11900001764297485)))
           => (roughly 0 0.0000001)
           (nrm2 (axpy! -1 (view-vctr (bias fc))
                        (vctr train-tz 2.2351741790771484E-8 -0.6299999952316284))) => (roughly 0 0.0000001))))

(defn test-fully-connected-training-adam [fact]
  (with-release [input-tz (tensor fact [1 3 2 1] :float :nchw)
                 fc-bluep (fully-connected fact input-tz [1 2] :linear)
                 fc (fc-bluep input-tz false :adam)
                 train-tz (tensor fact [1 2] :float :nc)
                 fc-output (cost fc train-tz :quadratic)]
    (facts "Fully connected training layer - adam"
           (transfer! [-0.5 0 0.2 1 0.3 -0.7] input-tz)
           (transfer! [-0.1 0.1 0.2 -0.7 -0.1 0.1 0.2 -0.7 -0.1 0.1 0.2 -0.7] (weights fc))
           (transfer! [-0.1 0.2] (bias fc))
           (forward fc []) => fc
           (nrm2 (axpy! -1 (view-vctr (output fc))
                        (vctr train-tz -0.8100000023841858 0.7299999594688416))) => (roughly 0 0.0000001)
           (forward fc-output) => fc-output
           (transfer! [-0.71 -0.1] (view-vctr train-tz))
           (let [reflection-warn *warn-on-reflection*]
             (set! *warn-on-reflection* false)
             (transfer! [0.1 0.3 -0.4 -0.2 0.2 0.3 -0.3 -0.1 -0.15 0.12 0.25 -0.25] (.s fc))
             (transfer! [0.01 0.03 -0.04 -0.02 0.02 0.03 -0.03 -0.01 -0.015 0.012 0.025 -0.025] (.s fc))
             (set! *warn-on-reflection* reflection-warn))
           (backward fc-output)
           (backward fc) => fc
           (backward fc [ 1 1]) => fc
           (view-vctr input-tz) => (vctr train-tz -0.5 0 0.2 1.0 0.3 -0.69999999)
           (nrm2 (view-vctr (weights fc))) => (float 149791.78)
           (nrm2 (axpy! -1 (view-vctr (bias fc))
                        (vctr train-tz 2.2351741790771484E-8 -0.6299999952316284)))))) => (roughly 0 0.0000001)

(defn test-fully-connected-layer-1 [fact]
  (with-release [input-tz (tensor fact [1 1] :float :nc)
                 fc-bluep (fully-connected fact input-tz [1 1] :linear)
                 fc (fc-bluep input-tz false)
                 train-tz (tensor fact [1 1] :float :nc)
                 fc-output (cost fc train-tz :quadratic)]
    (facts "Fully connected layer step by step"
           (transfer! [-0.5] input-tz)
           (transfer! [-0.1] (weights fc))
           (transfer! [0.2] (bias fc))
           (forward fc [nil 1 0 0 false]) => fc
           (entry (native (view-vctr (output fc))) 0) => 0.25
           (forward fc-output) => fc-output
           (transfer! [-0.15] (view-vctr train-tz))
           (backward fc-output)
           (backward fc) => fc
           (entry (native (view-vctr (output fc))) 0) => (roughly 0.4)
           (entry (native (view-vctr input-tz)) 0) => -0.5
           (backward fc [nil 1 0 0 false]) => fc
           (entry (native (view-vctr input-tz)) 0) => -0.5
           (entry (native (view-vctr (weights fc))) 0) => (roughly 0.1)
           (entry (native (view-vctr (bias fc))) 0) => (roughly -0.2))))

(defn test-fully-connected-layer-2 [fact]
  (with-release [input-tz (tensor fact [2 1] :float :nc)
                 fc1-bluep (fully-connected fact input-tz [2 1] :linear)
                 fc2-bluep (fully-connected fact fc1-bluep [2 1] :linear)
                 fc1 (fc1-bluep input-tz true)
                 fc2 (fc2-bluep fc1 true)
                 train-tz (tensor fact [2 1] :float :nc)
                 fc-output (cost fc2 train-tz :quadratic)]
    (facts "Fully connected, 2 layers step by step."
           (transfer! [-0.5 -0.5] input-tz)
           (transfer! [-0.1 -0.1] (weights fc1))
           (transfer! [0.2 0.2] (bias fc1))
           (transfer! [0.8 0.8] (weights fc2))
           (transfer! [0.5 0.5] (bias fc2))
           (forward fc1 [nil 1 0 0 false]) => fc1
           (view-vctr (output fc1)) => (vctr train-tz 0.25 0.25)
           (output fc1) => (input fc2)
           (forward fc2 [nil 1 0 0 false]) => fc2
           (view-vctr (output fc2)) => (vctr train-tz 0.7 0.7)
           (forward fc-output) => fc-output
           (transfer! [0.25 0.25] (view-vctr train-tz))
           (backward fc-output)
           (backward fc2) => fc2
           (view-vctr (output fc2)) => (vctr train-tz 0.45 0.45)
           (view-vctr (input fc2)) => (vctr train-tz 0.25 0.25)
           (backward fc2 [nil 1 0 0 false]) => fc2
           (view-vctr (input fc2)) => (vctr train-tz 0.35999998 0.35999998)
           (view-vctr (weights fc2)) => (vctr train-tz [0.6875])
           (view-vctr (bias fc2)) => (vctr train-tz [0.050000012])
           (backward fc1) => fc1
           (view-vctr (diff-input fc1)) => (vctr train-tz 0.35999998 0.35999998)
           (backward fc1 [nil 1 0 0 false]) => fc1
           (view-vctr (diff-output fc1)) => (vctr train-tz -0.036 -0.036)
           (view-vctr (weights fc1)) => (vctr train-tz [0.07999999])
           (view-vctr (bias fc1)) => (vctr train-tz [-0.15999998]))))

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
           (train net quad-cost 10 [0.01 0 0 false]) => (roughly 0.0 0.002))))

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
           (transfer! [-0.1 -0.1] (weights (first net)))
           (transfer! [0.2 0.2] (bias (first net)))
           (transfer! [0.8 0.8] (weights (second net)))
           (transfer! [0.5 0.5] (bias (second net)))
           (transfer! [0.25 0.25] train-tz)
           (train net quad-cost 1 [1 0 0 false]) => 0.056953115582683234
           (entry (native (view-vctr (weights (first net)))) 0) => (roughly 0.08)
           (entry (native (view-vctr (bias (first net)))) 0) => (roughly -0.16)
           (entry (native (view-vctr (weights (second net)))) 0) => 0.6875
           (entry (native (view-vctr (bias (second net)))) 0) => (roughly 0.05))))

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
           (transfer! [-0.1 -0.1] (weights (first net)))
           (transfer! [0.2 0.2] (bias (first net)))
           (transfer! [0.8 0.8] (weights (second net)))
           (transfer! [0.5 0.5] (bias (second net)))
           (transfer! [0.25 0.25 2.5 2.5] train-tz)
           (train net x-batcher y-batcher quad-cost 2 [1]) => (roughly 5.5345)
           (entry (native (view-vctr (weights (first net)))) 0) => (roughly 0.97108)
           (entry (native (view-vctr (bias (first net)))) 0) => (roughly 1.0245)
           (entry (native (view-vctr (weights (second net)))) 0) = (roughly 0.79295)
           (entry (native (view-vctr (bias (second net)))) 0) => (roughly -1.38267))))

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
           (view-vctr (output net)) => (vctr train-tz 0.15 -1.64999998))))

(defn test-crossentropy-cost [fact]
  (with-release [input-tz (tensor fact [2 1] :float :nc)
                 train-tz (tensor fact [2 1] :float :nc)
                 net-bp (network fact input-tz
                                 [(fully-connected [1] :relu)
                                  (fully-connected [1] :sigmoid)])
                 net (net-bp input-tz :sgd)
                 crossentropy-cost (cost net train-tz :crossentropy)]
    (facts "Sigmoid crossentropy cost."
           (transfer! [0.25 0.65] train-tz)
           (transfer! [0.4 0.1] (output net))
           (crossentropy-cost) => 1.0728740692138672)))

(defn test-sequential-network-sigmoid-sgd [fact]
  (facts "Sequential SGD network with sigmoid cross-entropy."
         (with-release [input-tz (tensor fact [1 16] :float :nc)
                        train-tz (tensor fact [1 2] :float :nc)
                        net-bp (network fact input-tz
                                        [(fully-connected [64] :relu)
                                         (fully-connected [64] :relu)
                                         (fully-connected [2] :sigmoid)])
                        net (init! (net-bp input-tz :sgd))
                        crossentropy-cost (cost net train-tz :crossentropy)]
           (transfer! (range 16) input-tz)
           (transfer! [0.9 0.1] train-tz)
           (train net crossentropy-cost 3 [0.01 0 0 false]) => (roughly 2 1.3))))

(defn test-sequential-network-sigmoid-adam [fact]
  (facts "Sequential Adam network with sigmoid cross-entropy."
         (with-release [input-tz (tensor fact [1 16] :float :nc)
                        train-tz (tensor fact [1 2] :float :nc)
                        net-bp (network fact input-tz
                                        [(fully-connected [64] :relu)
                                         (fully-connected [64] :relu)
                                         (fully-connected [2] :sigmoid)])
                        net (init! (net-bp input-tz :adam))
                        crossentropy-cost (cost net train-tz :crossentropy)]
           (transfer! (range 16) input-tz)
           (transfer! [0.9 0.1] train-tz)
           (train net crossentropy-cost 3 []) => (roughly 1.7 1.1))))

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
           (rand-uniform! (view-vctr x-tz))
           (transfer! (map my-fn (cols (native (view-ge (view-vctr x-tz) 4 10000)))) (view-vctr y-tz))
           (train net quad-cost 30 [0.003 0 0 false]) => (roughly 0.0 0.2))))

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
           (rand-uniform! (view-vctr x-tz))
           (transfer! (map my-fn (cols (native (view-ge (view-vctr x-tz) 4 10000)))) (view-vctr y-tz))
           (train net x-shuff y-shuff quad-cost 1 [0.01 0 0 false]) => (roughly 0.0 0.2))))

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
           (rand-uniform! (view-vctr x-tz))
           (transfer! (map my-fn (cols (native (view-ge (view-vctr x-tz) 4 10000)))) (view-vctr y-tz))
           (train net x-shuff y-shuff quad-cost 1 [0.01]) => (roughly 0.0 0.01))))

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
              (net)
              nil))))

(defn test-convolution-inference [fact]
  (with-release [input-tz (tensor fact [2 1 4 4] :float :nchw)
                 conv-bluep (convolution fact [2 1 4 4] [1 1 3 3] [1] :linear)
                 ws (create-workspace fact (inf-ws-size conv-bluep))]
    (binding [*workspace* ws]
      (with-release [conv (conv-bluep input-tz)
                     connect-output (connector (output conv) (desc [2 1 2 2] :float :nchw))
                     input-weights (connector (desc [1 1 3 3] :float :nchw) (weights conv))
                     output-weights (revert input-weights)]
        (facts "Convolution inference layer."
               (transfer! [0 43 3 30 0 98 0 0 7 38 0 0 19 20 175 50
                           0 0 7 19 43 98 38 20 3 0 0 175 30 0 0 50]
                          input-tz)
               (transfer! [-2 0 1 0 1 0 -1 -2 0] (input input-weights))
               (input-weights)
               (transfer! [0.5] (bias conv))
               (conv) => (output conv)
               (seq (native (connect-output))) => [18.5 -93.5 -20.5 -565.5 102.5 57.5 -77.5 -175.5])))))

(defn test-convolution-inference-relu [fact]
  (with-release [input-tz (tensor fact [2 1 4 4] :float :nchw)
                 conv-bluep (convolution fact [2 1 4 4] [1 1 3 3] [1] :relu)
                 ws (create-workspace fact (inf-ws-size conv-bluep))]
    (binding [*workspace* ws]
      (with-release [conv (conv-bluep input-tz)
                     connect-output (connector (output conv) (desc [2 1 2 2] :float :nchw))
                     input-weights (connector (desc [1 1 3 3] :float :nchw) (weights conv))
                     output-weights (revert input-weights)]
        (facts "Convolution inference layer."
               (transfer! [0 43 3 30 0 98 0 0 7 38 0 0 19 20 175 50
                           0 0 7 19 43 98 38 20 3 0 0 175 30 0 0 50]
                          input-tz)
               (transfer! [-2 0 1 0 1 0 -1 -2 0] (input input-weights))
               (input-weights)
               (transfer! [0.5] (bias conv))
               (conv) => (output conv)
               (seq (native (connect-output))) => [18.5 0.0 0.0 0.0 102.5 57.5 0.0 0.0])))))

(defn test-convolution-training [fact]
  (with-release [input-tz (tensor fact [2 1 4 4] :float :nchw)
                 conv-bluep (convolution fact [2 1 4 4] [1 1 3 3] [1] :linear)
                 ws (create-workspace fact (train-ws-size conv-bluep))]
    (binding [*workspace* ws]
      (with-release [conv (conv-bluep input-tz true)
                     input-weights (connector (desc [1 1 3 3] :float :nchw) (weights conv))
                     output-weights (revert input-weights)
                     train-tz (tensor fact [2 1 2 2] :float :nchw)
                     conv-output (cost conv train-tz :quadratic)]
        (facts "Convolution training layer."
               (transfer! [0 43 3 30 0 98 0 0 7 38 0 0 19 20 175 50
                           0 0 7 19 43 98 38 20 3 0 0 175 30 0 0 50]
                          input-tz)
               (transfer! [-2 0 1 0 1 0 -1 -2 0] (input input-weights))
               (input-weights)
               (transfer! [0.5] (bias conv))
               (forward conv [nil 1 0 0 false]) => conv
               (forward conv-output)
               (seq (native (output conv-output))) => [18.5 -93.5 -20.5 -565.5 102.5 57.5 -77.5 -175.5]
               (transfer! [18.3 -93.8 -21.3 -566.5 101.5 56.5 -78.5 -176.5] (view-vctr train-tz))
               (backward conv-output) => conv-output
               (backward conv) => conv
               (backward conv [nil 1 0 0 false]) => conv
               (nrm2 (axpy! -1 (view-vctr (weights conv))
                            (vctr train-tz -127.950065 -115.449982 -45.800049 -108.500145
                                  -92.000023 -116.5 -41.500053 -101.300003 -207.499939))) => (roughly 0 0.0001)
               (view-vctr (input conv)) => (vctr train-tz [-0.40000152587890625 -0.600006103515625 0.20000076293945312
                                                           0.3000030517578125 -1.5999984741210938 -1.7999992370605469
                                                           1.1000022888183594 1.0 -0.20000076293945312 0.09999465942382812
                                                           0.399993896484375 0.0 -0.7999992370605469 -2.5999984741210938
                                                           -2.0 0.0 -2.0 -2.0 1.0 1.0 -2.0 -1.0 2.0 1.0 -1.0 -2.0 -1.0
                                                           0.0 -1.0 -3.0 -2.0 0.0]))))))

(defn test-pooling-max [fact]
  (with-release [src-tz (tensor fact [2 1 4 4] :float :nchw)
                 pool-bluep (pooling fact src-tz [2 2] :max)
                 pool-infer (pool-bluep src-tz)
                 pool-train (pool-bluep src-tz true nil)]

    (transfer! [0 43 3 30 0 98 0 0 7 38 0 0 19 20 175 50
                0 0 7 19 43 98 38 20 3 0 0 175 30 0 0 50] src-tz)

    (facts
     "Pooling inference test."
     (view-vctr (pool-infer)) => (vctr src-tz [98.0 30.0 38.0 175.0 98.0 38.0 30.0 175.0])
     (view-vctr (input pool-infer)) => (vctr src-tz [0 43 3 30 0 98 0 0 7 38 0 0 19 20 175 50
                                                     0 0 7 19 43 98 38 20 3 0 0 175 30 0 0 50])
     (view-vctr (output pool-infer)) => (vctr src-tz [98.0 30.0 38.0 175.0 98.0 38.0 30.0 175.0]))

    (facts
     "Pooling forward test."
     (entry! (output pool-train) 0.0)
     (forward pool-train nil) => pool-train
     (view-vctr (input pool-train)) => (vctr src-tz [0 43 3 30 0 98 0 0 7 38 0 0 19 20 175 50
                                                     0 0 7 19 43 98 38 20 3 0 0 175 30 0 0 50])
     (view-vctr (output pool-train)) => (vctr src-tz [98.0 30.0 38.0 175.0 98.0 38.0 30.0 175.0]))

    (facts
     "Pooling backward test."
     (entry! (diff-input pool-train) 2.0)
     (backward pool-train nil)
     (view-vctr (diff-output pool-train))
     => (vctr src-tz [0.0 0.0 0.0 2.0 0.0 2.0 0.0 0.0 0.0 2.0 0.0 0.0 0.0 0.0 2.0 0.0
                      0.0 0.0 0.0 0.0 0.0 2.0 2.0 0.0 0.0 0.0 0.0 2.0 2.0 0.0 0.0 0.0]))))

(defn test-pooling-avg [fact]
  (with-release [src-tz (tensor fact [2 1 4 4] :float :nchw)
                 pool-bluep (pooling fact src-tz [2 2] :avg)
                 pool-infer (pool-bluep src-tz)
                 pool-train (pool-bluep src-tz true nil)]

    (transfer! [0 43 3 30 0 98 0 0 7 38 0 0 19 20 175 50
                0 0 7 19 43 98 38 20 3 0 0 175 30 0 0 50] src-tz)

    (facts
     "Pooling inference test."
     (view-vctr (pool-infer)) => (vctr src-tz [35.25 8.25 21.0 56.25 35.25 21.0 8.25 56.25])
     (view-vctr (input pool-infer)) => (vctr src-tz [0 43 3 30 0 98 0 0 7 38 0 0 19 20 175 50
                                                     0 0 7 19 43 98 38 20 3 0 0 175 30 0 0 50])
     (view-vctr (output pool-infer)) => (vctr src-tz [35.25 8.25 21.0 56.25 35.25 21.0 8.25 56.25]))

    (facts
     "Pooling forward test."
     (entry! (output pool-train) 0.0)
     (forward pool-train nil) => pool-train
     (view-vctr (input pool-train)) => (vctr src-tz [0 43 3 30 0 98 0 0 7 38 0 0 19 20 175 50
                                                     0 0 7 19 43 98 38 20 3 0 0 175 30 0 0 50])
     (view-vctr (output pool-train)) => (vctr src-tz [35.25 8.25 21.0 56.25 35.25 21.0 8.25 56.25]))

    (facts
     "Pooling backward test."
     (entry! (diff-input pool-train) 2.0)
     (backward pool-train nil)
     (view-vctr (diff-output pool-train)) => (vctr src-tz (repeat 32 0.5)))))

(defn test-sequential-network-convolution-adam [fact]
  (facts "Convolutional network with softmax cross-entropy."
         (with-release [input-tz (tensor fact [2 1 8 8] :float :nhwc)
                        train-tz (tensor fact [2 2] :float :nc)
                        net-bp (network fact input-tz
                                        [(convolution [1] [3 3] :relu)
                                         (pooling [2 2] :max)
                                         (convolution [1] [2 2] :relu {:padding [1 1]})
                                         (fully-connected [4] :relu)
                                         (fully-connected [2] :softmax)])
                        net (init! (net-bp input-tz :adam))
                        crossentropy-cost (cost net train-tz :crossentropy)]
           (transfer! (range 1 10 0.2) input-tz)
           (transfer! [0.9 0.1 0 0] train-tz)
           (train net crossentropy-cost 3 []) => (roughly 1.7 1))))

(defn test-gaussian-dropout [fact]
  (with-release [src-tz (tensor fact [2 1 4 4] :float :nchw)
                 drop-bluep (dropout fact src-tz 1.0)
                 drop-train (drop-bluep src-tz nil true)]

    (transfer! (repeat 1) src-tz)

    (facts
     "Dropout forward test."
     (forward drop-train nil) => drop-train
     (view-vctr (div! (output drop-train) (.mask-tz drop-train))) => (vctr src-tz (repeat 32 1.0)))

    (facts
     "Dropout backward test."
     (backward drop-train nil)
     (view-vctr (div! (output drop-train) (.mask-tz drop-train))) => (vctr src-tz (repeat 32 1.0)))))

(defn test-batch-normalization-inference [fact]
  (with-release [input-tz (tensor fact [2 1 2 2] :float :nchw)
                 bnorm-bluep (batch-norm fact input-tz :linear nil)
                 bnorm-infer (bnorm-bluep input-tz)]

    (transfer! [-1 0 1 2 3 4 5 6] input-tz)
    (doall (map transfer! [[0.5] [1.5] [2.5] [5.25]] (parameters bnorm-infer)))

    (facts
     "Batch normalization inference test."
     (seq (native (input bnorm-infer))) => [-1.0 0.0 1.0 2.0 3.0 4.0 5.0 6.0]
     (seq (native (bnorm-infer)))
     => [0.7362374067306519 0.9544553160667419 1.172673225402832 1.3908910751342773
         1.6091089248657227 1.827326774597168 2.0455446243286133 2.2637624740600586]
     (seq (native (output bnorm-infer)))
     => [0.7362374067306519 0.9544553160667419 1.172673225402832 1.3908910751342773
         1.6091089248657227 1.827326774597168 2.0455446243286133 2.2637624740600586]
     (input bnorm-infer) => (output bnorm-infer))))

(defn test-batch-normalization-training [fact]
  (with-release [input-tz (tensor fact [1 2 2 2] :float :nchw)
                 bnorm-bluep (batch-norm fact input-tz :linear nil)
                 bnorm-train (bnorm-bluep input-tz true)]

    (transfer! [-1 0 1 2 3 4 5 6] input-tz)
    (facts
     "Batch normalization forward test."
     (doall (map transfer! [[0.5 1.5] [1 1] [0] [0]] (parameters bnorm-train)))
     (seq (native (input bnorm-train))) => [-1.0 0.0 1.0 2.0 3.0 4.0 5.0 6.0]
     (forward bnorm-train [nil 1 0 1 false]) => bnorm-train
     (forward bnorm-train)
     (nrm2 (view-vctr (output bnorm-train))) => (roughly 4.2426405)
     (map (comp seq native view-vctr) (parameters bnorm-train))
     => [[0.5 1.5] [1.0 1.0] [0.5 4.5] [1.25 1.25]])

    (facts
     "Batch normalization backward test."
     (transfer! [-5 10 0.3 0.2 -0.5 0.6 0.9 -3] (diff-input bnorm-train))
     (backward bnorm-train) => bnorm-train
     (backward bnorm-train [nil 1 0 1 false]) => bnorm-train
     (seq (native (weights bnorm-train))) => (just [(roughly -2.1385) (roughly 4.7199)])
     (seq (native (bias bnorm-train))) => [-4.5 3.0]
     (nrm2 (native (view-vctr (diff-output bnorm-train)))) => (roughly 5.954477))))

(defn test-concatenate [fact]
  (with-release [input0-tz (tensor fact [1 1 2 2] :float :nchw)
                 input1-tz (tensor fact [1 2 2 2] :float :nchw)
                 concat-bluep (concatenate fact 1 [input0-tz input1-tz input0-tz] nil)
                 concat-inf (concat-bluep [input0-tz input1-tz input0-tz])
                 concat-train (concat-bluep [input0-tz input1-tz input0-tz] true nil)]

    (transfer! (range 4) input0-tz)
    (transfer! (range 10 90 10) input1-tz)

    (facts
     "Concatenate inference test."
     (seq (native (concat-inf)))
     => [0.0 1.0 2.0 3.0 10.0 20.0 30.0 40.0 50.0 60.0 70.0 80.0 0.0 1.0 2.0 3.0])

    (facts
     "Concatenate training test."
     (forward concat-train nil) => concat-train
     (seq (native (output concat-train)))
     => [0.0 1.0 2.0 3.0 10.0 20.0 30.0 40.0 50.0 60.0 70.0 80.0 0.0 1.0 2.0 3.0]

     (transfer! (repeat 0.0) input0-tz)
     (transfer! (repeat 0.0) input1-tz)

     (seq (native input0-tz)) => [0.0 0.0 0.0 0.0]
     (backward concat-train nil) => concat-train
     (seq (native input0-tz)) => [0.0 1.0 2.0 3.0]
     (seq (native input1-tz)) => [10.0 20.0 30.0 40.0 50.0 60.0 70.0 80.0])))

(defn test-branch [fact]
  (with-release [dst0-desc (desc [1 1 2 2] :float :nchw)
                 dst1-desc (desc [1 2 2 2] :float :nchw)
                 input-tz (tensor fact [1 3 2 2] :float :nchw)
                 branch-bluep (branch fact input-tz 1 [dst0-desc dst1-desc])
                 branch-inf (branch-bluep input-tz)
                 branch-train (branch-bluep input-tz true nil)]

    (transfer! [0.0 1.0 2.0 3.0 10.0 20.0 30.0 40.0 50.0 60.0 70.0 80.0] input-tz)

    (facts
     "Branch inference test."

     (map (comp seq native) (branch-inf)) => [[0.0 1.0 2.0 3.0]
                                              [10.0 20.0 30.0 40.0 50.0 60.0 70.0 80.0]])

    (facts
     "Branch training test."
     (forward branch-train nil) => branch-train
     (map (comp seq native) (output branch-train)) => [[0.0 1.0 2.0 3.0]
                                                       [10.0 20.0 30.0 40.0 50.0 60.0 70.0 80.0]]
     (transfer! (repeat 0.0) input-tz)
     (seq (native input-tz)) => (repeat 12 0.0)
     (backward branch-train nil) => branch-train
     (seq (native input-tz)) => [0.0 1.0 2.0 3.0 10.0 20.0 30.0 40.0 50.0 60.0 70.0 80.0])))

(defn test-parallel-network-solo [fact]
  (with-release [input0-tz (tensor fact [1 1 1 1] :float :nchw)
                 input1-tz (tensor fact [1 2 1 1] :float :nchw)
                 net-bp (network fact [input0-tz input1-tz]
                                 [[[(dense [1] :linear) (dense [2] :linear)]
                                   [(dense [1] :linear)]]])
                 net-train (net-bp [input0-tz input1-tz] true :adam)]
    (transfer! [1] input0-tz)
    (transfer! [2 3] input1-tz)

    (facts
     "Parallel training test, solo."
     (doall (map transfer! [[0.1] [0.0]] (parameters (get-in net-train [0 0 0]))))
     (doall (map transfer! [[10 20] [0 0]] (parameters (get-in net-train [0 0 1]))))
     (doall (map transfer! [[0.1 0.2] [0 0]] (parameters (get-in net-train [0 1 0]))))
     (forward net-train [0 1 0 0 false]) => net-train
     (map (comp seq native) (output net-train)) => [[1.0 2.0] [0.800000011920929]]

     (transfer! (repeat 0.0) input0-tz)
     (transfer! (repeat 0.0) input1-tz)

     (transfer! [0.5 1] (first (diff-input net-train)))
     (transfer! [0.1] (second (diff-input net-train)))
     (backward net-train)
     (backward net-train [0 1 0 0 false]) => net-train
     (map (comp seq native output) (first net-train)) => [[0.5 1.0] [0.10000000149011612]]
     (seq (native input0-tz)) => [2.5]
     (seq (native input1-tz)) => [0.010000000707805157 0.020000001415610313])))

(defn test-network-concat [fact]
  (with-release [input0-tz (tensor fact [1 1 1 1] :float :nchw)
                 input1-tz (tensor fact [1 2 1 1] :float :nchw)
                 net-bp (network fact [input0-tz input1-tz]
                                 [(conc 1)])
                 net-train (net-bp [input0-tz input1-tz] true :adam)]
    (let [parallel-layers 1]
      (transfer! [1] input0-tz)
      (transfer! [2 3] input1-tz)

      (facts
       "Network training test, concat."
       (forward net-train [0 1 0 0 false]) => net-train
       (seq (native (output net-train))) => [1.0 2.0 3.0]

       (transfer! (repeat 0.0) input0-tz)
       (transfer! (repeat 0.0) input1-tz)

       (transfer! [0.5 1 0.1] (diff-input net-train))
       (backward net-train [0 1 0 0 false]) => net-train
       (seq (native input0-tz)) => [0.5]
       (seq (native input1-tz)) => [1.0 0.10000000149011612]))))

(defn test-network-branch-concat [fact]
  (with-release [input-tz (tensor fact [1 4 1 1] :float :nchw)
                 net-bp (network fact input-tz
                                 [(branch 1 [(desc [1 1 1 1] :float :nchw)
                                             (desc [1 2 1 1] :float :nchw)
                                             (desc [1 1 1 1] :float :nchw)])
                                  (conc 1)])
                 net-train (net-bp input-tz true :adam)]
    (facts
     "Branch-concat test."
     (transfer! [1 2 3 1] input-tz)
     (forward net-train [0 1 0 0 false]) => net-train
     (seq (native (output net-train))) => [1.0 2.0 3.0 1.0]
     (transfer! (repeat 0.0) input-tz)
     (transfer! [0.5 1 0.1 0.5] (diff-input net-train))
     (backward net-train [0 1 0 0 false]) => net-train
     (seq (native input-tz)) => [0.5 1.0 0.10000000149011612 0.5])))

(defn test-network-branch-concat-simplified [fact]
  (with-release [input-tz (tensor fact [1 4 1 1] :float :nchw)
                 net-bp (network fact input-tz
                                 [(branch 1 [1 2 1])
                                  (conc 1)])
                 net-train (net-bp input-tz true :adam)]
    (facts
     "Branch-concat test, with simplified branch destination specification."
     (transfer! [1 2 3 1] input-tz)
     (forward net-train [0 1 0 0 false]) => net-train
     (seq (native (output net-train))) => [1.0 2.0 3.0 1.0]
     (transfer! (repeat 0.0) input-tz)
     (transfer! [0.5 1 0.1 0.5] (diff-input net-train))
     (backward net-train [0 1 0 0 false]) => net-train
     (seq (native input-tz)) => [0.5 1.0 0.10000000149011612 0.5])))

(defn test-parallel-network-concat [fact]
  (with-release [input0-tz (tensor fact [1 1 1 1] :float :nchw)
                 input1-tz (tensor fact [1 2 1 1] :float :nchw)
                 net-bp (network fact [input0-tz input1-tz]
                                 [[[(dense [1] :linear) (dense [2] :linear)]
                                   [(dense [1] :linear)]]
                                  (conc 1)])
                 net-train (net-bp [input0-tz input1-tz] true :adam)]
    (transfer! [1] input0-tz)
    (transfer! [2 3] input1-tz)

    (facts
     "Parallel training test, concat."
     (doall (map transfer! [[0.1] [0]] (parameters (get-in net-train [0 0 0]))))
     (doall (map transfer! [[10 20] [0 0]] (parameters (get-in net-train [0 0 1]))))
     (doall (map transfer! [[0.1 0.2] [0 0]] (parameters (get-in net-train [0 1 0]))))
     (forward net-train [0 1 0 0 false]) => net-train
     (seq (native (output net-train))) => [1.0 2.0 0.800000011920929]

     (transfer! (repeat 0.0) input0-tz)
     (transfer! (repeat 0.0) input1-tz)

     (transfer! [0.5 1 0.1] (diff-input net-train))
     (backward net-train [0 1 0 0 false]) => net-train
     (map (comp seq native output) (first net-train)) => [[0.5 1.0] [0.10000000149011612]]
     (seq (native input0-tz)) => [2.5]
     (seq (native input1-tz)) => [0.010000000707805157 0.020000001415610313])))

(defn test-parallel-network-nested [fact]
  (with-release [input-tz (tensor fact [1 4 1 1] :float :nchw)
                 net-bp (network fact input-tz
                                 [(branch 1 [1 2 1])
                                  [[(dense [1] :linear) (dense [2] :linear)]
                                   [(dense [1] :linear)]
                                   [(dense [1] :linear) (dense [2] :linear)]]
                                  (conc 1)]);;
                 net-train (net-bp input-tz true :adam)]
    (transfer! [1 2 3 1] input-tz)

    (facts
     "Parallel training test, nested."
     (doall (map transfer! [[0.1] [0]] (parameters (get-in net-train [1 0 0]))))
     (doall (map transfer! [[10 20] [0 0]] (parameters (get-in net-train [1 0 1]))))
     (doall (map transfer! [[0.1 0.2] [0 0]] (parameters (get-in net-train [1 1 0]))))
     (doall (map transfer! [[0.1] [0]] (parameters (get-in net-train [1 2 0]))))
     (doall (map transfer! [[10 20] [0 0]] (parameters (get-in net-train [1 2 1]))))
     (forward net-train [0 1 0 0 false]) => net-train
     (seq (native (output net-train))) => [1.0 2.0 0.800000011920929 1.0 2.0]

     (transfer! (repeat 0.0) input-tz)
     (transfer! [0.5 1 0.1 0.5 1] (diff-input net-train))
     (backward net-train [0 1 0 0 false]) => net-train
     (map (comp seq native output) (second net-train))
     => [[0.5 1.0] [0.10000000149011612] [0.5 1.0]]
     (seq (native input-tz)) => [2.5 0.010000000707805157 0.020000001415610313 2.5]
     (map (comp seq native) (output (get net-train 0)))
     => [[2.5] [0.010000000707805157 0.020000001415610313] [2.5]])))

(defn test-sum [fact]
  (with-release [input0-tz (tensor fact [1 2 1 1] :float :nchw)
                 input1-tz (tensor fact [1 2 1 1] :float :nchw)
                 sum-bluep (sum fact [input0-tz input1-tz])
                 sum-train (sum-bluep [input0-tz input1-tz] true nil)]

    (transfer! [1 2] input0-tz)
    (transfer! [3 4] input1-tz)

    (facts
     "Sum training test."
     (forward sum-train nil) => sum-train
     (seq (native (output sum-train))) => [4.0 6.0]

     (transfer! [0 0] input1-tz)

     (seq (native input1-tz)) => [0.0 0.0]
     (backward sum-train nil) => sum-train
     (seq (native input0-tz)) => [2.0 3.0]
     (seq (native input1-tz)) => [2.0 3.0])))

(defn test-split [fact]
  (with-release [input-tz (tensor fact [1 2 1 1] :float :nchw)
                 split-bluep (split fact input-tz 2)
                 split-inf (split-bluep input-tz)
                 split-train (split-bluep input-tz true nil)]

    (transfer! [1.0 2.0] input-tz)

    (facts
     "Split inference test."
     (map (comp seq native) (split-inf)) => [[1.0 2.0] [1.0 2.0]])

    (facts
     "Split training test."
     (forward split-train nil) => split-train
     (map (comp seq native) (output split-train)) => [[1.0 2.0] [1.0 2.0]]

     (transfer! (repeat 0.0) input-tz)
     (seq (native input-tz)) => [0.0 0.0]
     (transfer! [1.0 2.0] (first (diff-input split-train)))
     (transfer! [3.0 4.0] (second (diff-input split-train)))
     (backward split-train nil) => split-train
     (seq (native input-tz)) => [2.0 3.0])))

;; ============================= RNN tests ==================================================

(defn test-vanilla-rnn-inference [fact]
  (with-release [input-tz (tensor fact [2 1 2] :float :tnc)
                 rnn-bluep-no-iter (rnn-op fact input-tz [2 1 2] 2)
                 rnn-no-iter (rnn-bluep-no-iter input-tz)
                 input-weights (connector (desc [2 1 2 1 2] :float :ldigo) (weights-layer rnn-no-iter))
                 input-weights-iter (connector (desc [2 1 2 1 2] :float :ldigo) (weights-iter rnn-no-iter))
                 input-bias (connector (desc [2 1 1 2] :float :ldgo) (bias-layer rnn-no-iter))
                 output-bias (revert input-bias)]
    (facts "Vanilla RNN inference operation without iter."
           (init! rnn-no-iter)
           (transfer! [2 3 0.2 0.3] input-tz)
           (transfer! [0.1 0.2 0.3 0.4 0.3 0.4 0.5 0.6] (input input-weights))
           (input-weights)
           (transfer! [100 200 300 400 0.01 0.02 0.03 0.04] (input input-weights-iter))
           (input-weights-iter)
           (transfer! [0.3 0.7 1 2] (input input-bias))
           (input-bias)
           (seq (native (rnn-no-iter)))
           => (just [2.570000171661377 3.940000057220459 850.6968994140625 (roughly 1054.889)])
           (seq (native (output rnn-no-iter)))
           => (just [2.570000171661377 3.940000057220459 850.6968994140625 (roughly 1054.889)])
           (seq (native (output-bias)))
           => (just [(roughly 0.3) (roughly 0.7) (roughly 1.0) (roughly 2.0)]))))

(defn test-vanilla-rnn-training [fact]
  (with-release [input-tz (tensor fact [2 1 2] :float :tnc)
                 dst-desc (desc [2 1 2] :float :tnc)
                 rnn-bluep-no-iter (rnn-op fact input-tz [2 1 2] 2 false false)
                 rnn-no-iter (rnn-bluep-no-iter input-tz input-tz false false)
                 input-weights (connector (desc [2 1 2 1 2] :float :ldigo) (weights-layer rnn-no-iter))
                 input-weights-iter (connector (desc [2 1 2 1 2] :float :ldigo) (weights-iter rnn-no-iter))
                 output-weights (revert input-weights)
                 output-weights-iter (connector (weights-iter rnn-no-iter)
                                                (desc [2 1 2 1 2] :float :ldigo))
                 input-bias (connector (desc [2 1 1 2] :float :ldgo) (bias-layer rnn-no-iter))
                 output-bias (revert input-bias)]
    (facts "Vanilla RNN training operation without iter."
           (init! rnn-no-iter)
           (transfer! [2 3 0.2 0.3] input-tz)
           (transfer! [0.1 0.2 0.3 0.4 0.3 0.4 0.5 0.6] (input input-weights))
           (input-weights)
           (transfer! [100 200 300 400 0.01 0.02 0.03 0.04] (input input-weights-iter))
           (input-weights-iter)
           (transfer! [0.3 0.7 1 2] (input input-bias))
           (input-bias)
           (forward rnn-no-iter)
           (seq (native (output rnn-no-iter)))
           => (just [(roughly 2.57) (roughly 3.94) (roughly 850.697) (roughly 1054.889)])
           (transfer! [1.1 -2.2 3.3 -4.4] (diff-input rnn-no-iter))
           (backward rnn-no-iter)
           (seq (native (diff-output rnn-no-iter)))
           => (just [(roughly -153.1285) (roughly -333.8167) (roughly -0.275) (roughly -0.627)])
           (seq (native (output-weights))) => (map float [0.1 0.2 0.3 0.4 0.3 0.4 0.5 0.6])
           (map float (seq (native (output-weights-iter)))) => (map float [100 200 300 400 0.01 0.02 0.03 0.04])
           (nrm2 (diff-weights rnn-no-iter)) => (roughly 8408.49)
           (seq (native (output-bias)))
           => (just [(roughly -276.36731) (roughly -628.83374) (roughly 4.345) (roughly -6.677)]))))

(defn test-rnn-inference [fact]
  (with-release [input-tz (tensor fact [2 1 2] :float :tnc)
                 rnn-bluep-iter (rnn fact input-tz [2 1 2] 2 :relu {:src-iter true :dst-iter true})
                 rnn-iter (rnn-bluep-iter input-tz)
                 input-weights (connector (desc [2 1 2 1 2] :float :ldigo) (weights-layer (.op rnn-iter)))
                 input-weights-iter (connector (desc [2 1 2 1 2] :float :ldigo) (weights-iter (.op rnn-iter)))
                 input-bias (connector (desc [2 1 1 2] :float :ldgo) (bias-layer rnn-iter))]
    (facts "Vanilla RNN layer inference."
           (init! rnn-iter)
           (transfer! [2 3 0.2 0.3] input-tz)
           (transfer! [0.1 0.2 0.3 0.4 0.3 0.4 0.5 0.6] (input input-weights))
           (input-weights)
           (transfer! [100 200 300 400 0.01 0.02 0.03 0.04] (input input-weights-iter))
           (input-weights-iter)
           (transfer! [0.3 0.7 1 2] (input input-bias))
           (input-bias)
           (seq (native (rnn-iter)))
           => (just [2.570000171661377 3.940000057220459 850.6968994140625 (roughly 1054.889)])
           (seq (native (output rnn-iter)))
           => (just [2.570000171661377 3.940000057220459 850.6968994140625 (roughly 1054.889)]))
    (release input-tz)
    (release input-weights)
    (release input-weights-iter)
    (release input-bias)
    (release rnn-iter)
    (release rnn-bluep-iter)))

(defn test-rnn-training-no-iter [fact]
  (with-release [input-tz (tensor fact [2 1 2] :float :tnc)
                 rnn-bluep-no-iter (rnn fact input-tz [2 1 2] 2 :relu nil)
                 rnn-no-iter (rnn-bluep-no-iter input-tz nil :sgd)
                 input-weights (connector (desc [2 1 2 1 2] :float :ldigo) (weights-layer (.op rnn-no-iter)))
                 input-weights-iter (connector (desc [2 1 2 1 2] :float :ldigo) (weights-iter (.op rnn-no-iter)))
                 input-bias (connector (desc [2 1 1 2] :float :ldgo) (bias-layer rnn-no-iter))]
    (facts "Vanilla RNN layer training without iter."
           (init! rnn-no-iter)
           (transfer! [2 3 0.2 0.3] input-tz)
           (transfer! [0.1 0.2 0.3 0.4 0.3 0.4 0.5 0.6] (input input-weights))
           (input-weights)
           (transfer! (repeat 8 0) (input input-weights-iter))
           (input-weights-iter)
           (transfer! [0.3 0.7 1 2] (input input-bias))
           (input-bias)
           (forward rnn-no-iter [1 0 0 true])
           (seq (native (output rnn-no-iter)))
           => [2.570000171661377 3.940000057220459 1.5529999732971191 2.680000066757202]
           (transfer! [1.1 -2.2 3.3 -4.4] (diff-input rnn-no-iter))
           (backward rnn-no-iter [nil 1 0 0 true])
           (seq (native (diff-output rnn-no-iter)))
           => (just [(roughly -0.209) (roughly -0.473) (roughly -0.275) (roughly -0.627)]))))

(defn test-lstm-training-no-iter [fact]
  (with-release [input-tz (tensor fact [2 1 2] :float :tnc)
                 lstm-bluep-no-iter (rnn fact input-tz [2 1 2] 2 :lstm nil)
                 lstm-no-iter (lstm-bluep-no-iter input-tz nil :sgd)
                 input-weights (connector (desc [2 1 2 4 2] :float :ldigo) ;;TODO support just :ldigo as desc.
                                          (weights-layer (.op lstm-no-iter)))
                 input-weights-iter (connector (desc [2 1 2 4 2] :float :ldigo) (weights-iter (.op lstm-no-iter)))
                 input-bias (connector (desc [2 1 4 2] :float :ldgo) (bias-layer lstm-no-iter))]
    (facts "LSTM layer training SGD."
           (init! (.op lstm-no-iter))
           (transfer! [2 3 0.2 0.3] input-tz)
           (transfer! [0.1 0.2 0.3 0.4 0.3 0.4 0.5 0.6
                       0.1 0.2 0.3 0.4 0.3 0.4 0.5 0.6
                       0.1 0.2 0.3 0.4 0.3 0.4 0.5 0.6
                       0.1 0.2 0.3 0.4 0.3 0.4 0.5 0.6]
                      (input input-weights))
           (input-weights)
           (transfer! (repeat 32 0) (input input-weights-iter))
           (input-weights-iter)
           (transfer! [0.3 0.7 1 2 0.3 0.7 1 2 0.3 0.7 1 2 0.3 0.7 1 2] (input input-bias))
           (input-bias)
           (forward lstm-no-iter [1 0 0 true])
           (seq (native (output lstm-no-iter)))
           => (just [(roughly 0.2837 0.001) (roughly 0.5043 0.001) (roughly 0.4724 0.001) (roughly 0.771 0.001)])
           (seq (native (output lstm-no-iter)))
           => (just [(roughly 0.2837 0.001) (roughly 0.5043 0.001) (roughly 0.4724 0.001) (roughly 0.771 0.001)])
           (transfer! [1.1 -2.2 3.3 -4.4] (diff-input lstm-no-iter))
           (backward lstm-no-iter [nil 1 0 0 true])
           (seq (native (diff-output lstm-no-iter)))
           => (just [(roughly 5.369E-5 0.001) (roughly 5.369E-5 0.001) (roughly 0.0236 0.001) (roughly 0.0236 0.001)]))))

(defn test-lstm-training-no-iter-adam [fact]
  (with-release [input-tz (tensor fact [2 1 2] :float :tnc)
                 lstm-bluep-no-iter (rnn fact input-tz [2 1 2] 2 :lstm nil)
                 lstm-no-iter (lstm-bluep-no-iter input-tz nil :adam)
                 input-weights (connector (desc [2 1 2 4 2] :float :ldigo) ;;TODO support just :ldigo as desc.
                                          (weights-layer (.op lstm-no-iter)))
                 input-weights-iter (connector (desc [2 1 2 4 2] :float :ldigo) (weights-iter (.op lstm-no-iter)))
                 input-bias (connector (desc [2 1 4 2] :float :ldgo) (bias-layer lstm-no-iter))]
    (facts "LSTM layer training Adam."
           (init! lstm-no-iter)
           (transfer! [2 3 0.2 0.3] input-tz)
           (transfer! [0.1 0.2 0.3 0.4 0.3 0.4 0.5 0.6
                       0.1 0.2 0.3 0.4 0.3 0.4 0.5 0.6
                       0.1 0.2 0.3 0.4 0.3 0.4 0.5 0.6
                       0.1 0.2 0.3 0.4 0.3 0.4 0.5 0.6]
                      (input input-weights))
           (input-weights)
           (transfer! (repeat 32 0) (input input-weights-iter))
           (input-weights-iter)
           (transfer! [0.3 0.7 1 2 0.3 0.7 1 2 0.3 0.7 1 2 0.3 0.7 1 2] (input input-bias))
           (input-bias)
           (forward lstm-no-iter [])
           (seq (native (output lstm-no-iter)))
           => (just [(roughly 0.2837 0.001) (roughly 0.5043 0.001) (roughly 0.4724 0.001) (roughly 0.771 0.001)])
           (seq (native (output lstm-no-iter)))
           => (just [(roughly 0.2837 0.001) (roughly 0.5043 0.001) (roughly 0.4724 0.001) (roughly 0.771 0.001)])
           (transfer! [1.1 -2.2 3.3 -4.4] (diff-input lstm-no-iter))
           (backward lstm-no-iter [1 1])
           (seq (native (diff-output lstm-no-iter)))
           => (just [(roughly 5.369E-5 0.001) (roughly 5.369E-5 0.001) (roughly 0.0236 0.001) (roughly 0.0236 0.001)]))))

(defn test-gru-training-no-iter-adam [fact]
  (with-release [input-tz (tensor fact [2 1 2] :float :tnc)
                 gru-bluep-no-iter (rnn fact input-tz [2 1 2] 2 :gru nil)
                 gru-no-iter (gru-bluep-no-iter input-tz nil :adam)
                 input-weights (connector (desc [2 1 2 3 2] :float :ldigo) ;;TODO support just :ldigo as desc.
                                          (weights-layer (.op gru-no-iter)))
                 input-weights-iter (connector (desc [2 1 2 3 2] :float :ldigo) (weights-iter (.op gru-no-iter)))
                 input-bias (connector (desc [2 1 3 2] :float :ldgo) (bias-layer gru-no-iter))]
    (facts "GRU layer training Adam."
           (init! gru-no-iter)
           (transfer! [2 3 0.2 0.3] input-tz)
           (transfer! [0.111 0.112 0.121 0.122 0.131 0.132
                       0.211 0.212 0.221 0.222 0.231 0.232
                       0.311 0.312 0.321 0.322 0.331 0.332
                       0.411 0.412 0.421 0.422 0.431 0.432]
                      (input input-weights))
           (input-weights)
           (transfer! (repeat 24 0) (input input-weights-iter))
           (input-weights-iter)
           (transfer! [0.3 0.7 0.3 0.7 0.3 0.7 1 2 1 2 1 2] (input input-bias))
           (input-bias)
           (forward gru-no-iter [])
           (seq (native (output gru-no-iter)))
           => (just [(roughly 0.198 0.01) (roughly 0.103 0.01) (roughly 0.3457 0.01) (roughly 0.189 0.01)])
           (seq (native (output gru-no-iter)))
           => (just [(roughly 0.198 0.01) (roughly 0.103 0.01) (roughly 0.3457 0.01) (roughly 0.189 0.01)])
           (transfer! [1.1 -2.2 3.3 -4.4] (diff-input gru-no-iter))
           (backward gru-no-iter [1 1])
           (seq (native (diff-output gru-no-iter)))
           => (just [(roughly -0.003 0.01) (roughly -0.006 0.01) (roughly 0.003 0.01) (roughly 0.005 0.01)]))))

(defn test-abbreviate [fact]
  (with-release [src-tz (tensor fact [3 4 2] :float :tnc)
                 edg-blueprint (abbreviate fact src-tz nil)
                 edg (edg-blueprint src-tz nil nil)]
    (facts "Test transfer of an abbreviation of a time series."
           (transfer! (range) src-tz)
           (forward edg nil)
           (take 2 (shape (output edg))) => [4 2]
           (take 2 (layout (output edg))) => [2 1]
           (seq (native (output edg))) => (range 16.0 24.0)
           (transfer! (range 100 200) (diff-input edg))
           (backward edg nil)
           (seq (native (output edg))) => (range 100.0 108.0)
           (seq (native src-tz)) => (into (vec (repeat 16 0.0)) (range 100.0 108.0)))))
