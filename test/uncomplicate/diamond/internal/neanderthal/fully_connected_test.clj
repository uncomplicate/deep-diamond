(ns uncomplicate.diamond.internal.neanderthal.fully-connected-test
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
                             desc revert shape input output view-tz batcher]]
             [dnn :refer [weights bias sum activation inner-product fully-connected
                          network init! train cost train]]]
            [uncomplicate.diamond.internal.protocols
             :refer [diff-bias diff-weights forward backward layers]]
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

#_(facts "Fully connected deep network"
       (with-release [fact (neanderthal-factory)
                      input-tz (tensor fact [1024 1] :float :nc)
                      net-bp (network input-tz input-tz
                                      [(fully-connected [1024] :relu)
                                       (fully-connected [349] :logistic)
                                       (fully-connected [4024] :tanh)
                                       (fully-connected [1] :elu)])
                      net (init! (net-bp input-tz :sgd))]
         (time (dotimes [i 100]
                 (forward net [0 1 0 0 false])
                 (backward net [0 1 0 0 false])))))
