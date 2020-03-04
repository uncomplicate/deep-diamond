(ns uncomplicate.diamond.internal.neanderthal.fully-connected-test
  (:require [midje.sweet :refer [facts throws => roughly]]
            [uncomplicate.commons [core :refer [with-release]]]
            [uncomplicate.diamond.dnn-test :refer :all]
            [uncomplicate.diamond.internal.neanderthal.factory :refer [neanderthal-factory]])
  (:import clojure.lang.ExceptionInfo))

(with-release [fact (neanderthal-factory)]
  (test-sum fact)
  (test-activation fact)
  (test-fully-connected-inference fact)
  (test-fully-connected-transfer fact)
  (test-fully-connected-training fact)
  (test-fully-connected-layer-1 fact)
  (test-fully-connected-layer-2 fact)
  (test-sequential-network-linear fact)
  (test-sequential-network-detailed fact)
  (test-quadratic-cost fact)
  (test-sequential-network-sigmoid fact)
  (test-gradient-descent fact)
  (test-stochastic-gradient-descent fact)
  (test-adam-gradient-descent fact))

;; (with-release [fact (neanderthal-factory)]
;;   (bench-wide-layers fact))
;; "Elapsed time: 6235.015802 msecs"
