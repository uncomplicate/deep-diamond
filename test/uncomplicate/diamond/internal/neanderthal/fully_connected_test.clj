(ns uncomplicate.diamond.internal.neanderthal.fully-connected-test
  (:require [midje.sweet :refer [facts throws => roughly]]
            [uncomplicate.commons [core :refer [with-release]]]
            [uncomplicate.diamond.dnn-test :refer :all]
            [uncomplicate.diamond.internal.neanderthal.factory :refer [neanderthal-factory]])
  (:import clojure.lang.ExceptionInfo))

(with-release [fact (neanderthal-factory)]
  (test-sum fact)
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
  (test-stochastic-gradient-descent-adam fact))

;; (with-release [fact (neanderthal-factory)]
;;   (bench-wide-layers fact))
;; "Elapsed time: 6235.015802 msecs"
