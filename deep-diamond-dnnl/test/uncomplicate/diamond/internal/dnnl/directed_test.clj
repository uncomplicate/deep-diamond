;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.diamond.internal.dnnl.directed-test
  (:require [midje.sweet :refer [facts throws => roughly]]
            [uncomplicate.commons [core :refer [with-release release]]]
            [uncomplicate.neanderthal
             [core :refer [transfer! native view-vctr view-ge cols]]
             [real :refer [entry! entry]]
             [native :refer [fv]]
             [random :refer [rand-uniform!]]
             [math :as math]]
            [uncomplicate.diamond
             [tensor :refer [*diamond-factory* tensor connector transformer desc shape input output
                             view-tz batcher]]
             [dnn :refer [sum activation inner-product fully-connected network init! train! cost]]
             [dnn-test :refer :all]]
            [uncomplicate.diamond.internal.protocols
             :refer [diff-weights forward backward weights bias]]
            [uncomplicate.diamond.internal.dnnl.factory :refer [dnnl-factory]])
  (:import clojure.lang.ExceptionInfo))

(facts "Inner product tests."
       (let [fact (dnnl-factory)
                      src-tz (tensor fact [1 3 2 1] :float :nchw)
                      dst-desc (desc [1 2] :float :nc)
                      ip (inner-product fact src-tz dst-desc)
                      ip-infer (ip src-tz)
                      ip-train (ip src-tz src-tz true false)]
         (transfer! [-0.5 0 0.2 1 0.3 -0.7] src-tz)
         (transfer! [-0.1 0.1 0.2 -0.7 -0.1 0.1 0.2 -0.7 -0.1 0.1 0.2 -0.7] (weights ip-infer))
         (transfer! [-0.1 0.2] (bias ip-infer))
         (view-vctr (ip-infer)) => (fv -0.81 0.72999996)
         (view-vctr (input ip-infer)) => (fv -0.5 0 0.2 1 0.3 -0.7)
         (view-vctr (output ip-infer)) => (fv -0.81 0.72999996)
         (transfer! [-0.1 0.1 0.2 -0.7 -0.1 0.1 0.2 -0.7 -0.1 0.1 0.2 -0.7] (weights ip-train))
         (transfer! [-0.1 0.2] (bias ip-train))
         (view-vctr (ip-train)) => (fv -0.81 0.72999996)
         (transfer! [-0.1 0.8299999594688416] (view-vctr (output ip-train)))
         (backward ip-train) => ip-train
         (view-vctr (diff-weights ip-train)) => (fv 0.05 0 -0.020000001 -0.1 -0.030000001 0.07
                                                    -0.415 0.0 0.166 0.83 0.249 -0.581)
         (release ip-train)
         (release ip-infer)
         (release ip)))

(facts "Inner product backprop step by step."
       (with-release [fact (dnnl-factory)
                      src-tz (tensor fact [1 1] :float :nc)
                      dst-desc (desc [1 1] :float :nc)
                      ip (inner-product fact src-tz dst-desc)
                      ip-infer (ip src-tz)
                      ip-train (ip src-tz src-tz true false)]
         (transfer! [-0.5] src-tz)
         (transfer! [-0.1] (weights ip-infer))
         (transfer! [ 0.2] (bias ip-infer))
         (view-vctr (ip-infer)) => (fv 0.25)
         (view-vctr (input ip-infer)) => (fv -0.5)
         (view-vctr (output ip-infer)) => (fv 0.25)
         (transfer! [-0.1] (weights ip-train))
         (transfer! [0.2] (bias ip-train))
         (view-vctr (ip-train)) => (fv 0.25)
         (transfer! [0.4] (view-vctr (output ip-train)))
         (backward ip-train) => ip-train
         (view-vctr (diff-weights ip-train)) => (fv -0.2)))

(with-release [fact (dnnl-factory)]
  (test-activation-relu fact)
  (test-activation-sigmoid fact)
  (test-fully-connected-inference fact)
  (test-fully-connected-transfer fact)
  (test-fully-connected-training fact)
  (test-fully-connected-training-adam fact)
  (test-fully-connected-layer-1 fact)
  (test-fully-connected-layer-2 fact)
  (test-sequential-network-linear fact)
  (test-sequential-network-detailed fact)
  (test-sequential-network-batched fact)
  (test-quadratic-cost fact)
  (test-sequential-network-sigmoid-sgd fact)
  (test-sequential-network-sigmoid-adam fact)
  (test-gradient-descent fact)
  (test-stochastic-gradient-descent-sgd fact)
  (test-stochastic-gradient-descent-adam fact)
  (test-crossentropy-cost fact)
  (test-inner-product-training fact)
  (test-convolution-inference fact)
  (test-convolution-inference-relu fact)
  (test-convolution-training fact)
  (test-pooling-max fact)
  (test-pooling-avg fact)
  (test-sequential-network-convolution-adam fact)
  (test-gaussian-dropout fact)
  (test-batch-normalization-inference fact)
  (test-batch-normalization-training fact)
  (test-concatenate fact)
  (test-branch fact)
  (test-network-concat fact)
  (test-network-branch-concat fact)
  (test-network-branch-concat-simplified fact)
  (test-parallel-network-solo fact)
  (test-parallel-network-concat fact)
  (test-parallel-network-nested fact)
  (test-sum fact)
  (test-split fact))

#_(with-release [fact (dnnl-factory)]
  (bench-wide-layers fact))
;; Intel i9 9900X (2019)
;; "Elapsed time: 1430.487041 msecs"
