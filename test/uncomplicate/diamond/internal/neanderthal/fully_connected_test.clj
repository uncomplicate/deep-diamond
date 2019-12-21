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

(facts "Fully connected inference layer"
       (with-release [fact (neanderthal-factory)
                      input-tz (tensor fact [1 6] :float :nc)
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
       (with-release [fact (neanderthal-factory)
                      input-tz (tensor fact [1 6] :float :nc)
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

(facts "Fully connected, 2 layers step by step"
       (with-release [fact (neanderthal-factory)
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
       (with-release [fact (neanderthal-factory)
                      input-tz (tensor fact [1 16] :float :nc)
                      train-tz (tensor fact [1 2] :float :nc)
                      net-bp (network fact input-tz
                                      [(fully-connected [64] :relu)
                                       (fully-connected [64] :relu)
                                       (fully-connected [2] :linear)])
                      net (init! (net-bp input-tz :sgd))
                      quad-cost (cost net train-tz :quadratic)]
         (transfer! (range 16) input-tz)
         (train net quad-cost 10 [0.01 0 0 false]) => (roughly 0.0 0.0001)))

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
