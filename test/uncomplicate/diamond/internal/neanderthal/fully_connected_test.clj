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
         (transfer! [-0.1 0.2 0.1 -0.7 0.2 -0.1 -0.7 0.1 -0.1 0.2 0.1 -0.7] (weights fc))
         (transfer! [-0.1 0.2] (bias fc))
         (view (output connect-output)) => (fv 0.0 0.0)
         (fc) => (output fc)
         (view (connect-output)) => (fv 0.0 0.72999996)))
