;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.diamond.internal.bnns.directed-test
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
             [dnn-test :refer :all]]
            [uncomplicate.diamond.internal.protocols
             :refer [diff-weights forward backward weights bias]]
            [uncomplicate.diamond.internal.bnns.factory :refer [bnns-factory]])
  (:import clojure.lang.ExceptionInfo))

(with-release [fact (bnns-factory)]
  (test-activation-relu fact)
  ;; (test-activation-sigmoid fact)
  ;; (test-fully-connected-inference fact)
  ;; (test-fully-connected-transfer fact)
  ;; (test-fully-connected-training fact)
  ;; (test-fully-connected-training-adam fact)
  ;; (test-fully-connected-layer-1 fact)
  ;; (test-fully-connected-layer-2 fact)
  ;; (test-sequential-network-linear fact)
  ;; (test-sequential-network-detailed fact)
  ;; (test-sequential-network-batched fact)
  ;; (test-quadratic-cost fact)
  ;; (test-sequential-network-sigmoid-sgd fact)
  ;; (test-sequential-network-sigmoid-adam fact)
  ;; (test-gradient-descent fact)
  ;; (test-stochastic-gradient-descent-sgd fact)
  ;; (test-stochastic-gradient-descent-adam fact)
  ;; (test-crossentropy-cost fact)
  ;; (test-inner-product-training fact)
  ;; (test-convolution-inference fact)
  ;; (test-convolution-inference-relu fact)
  ;; (test-convolution-training fact)
  ;; (test-pooling-max fact)
  ;; (test-pooling-avg fact)
  ;; (test-sequential-network-convolution-adam fact)
  ;; (test-gaussian-dropout fact)
  ;; (test-batch-normalization-inference fact)
  ;; (test-batch-normalization-training fact)
  ;; (test-concatenate fact)
  ;; (test-branch fact)
  ;; (test-network-concat fact)
  ;; (test-network-branch-concat fact)
  ;; (test-network-branch-concat-simplified fact)
  ;; (test-parallel-network-solo fact)
  ;; (test-parallel-network-concat fact)
  ;; (test-parallel-network-nested fact)
  ;; (test-sum fact)
  ;; (test-split fact)
  )

#_(with-release [fact (bnns-factory)]
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
;; "Elapsed time: 4990.836368 msecs"
