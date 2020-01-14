(ns uncomplicate.diamond.internal.dnnl.dnnl-tensor-test
  (:require [midje.sweet :refer [facts throws =>]]
            [uncomplicate.commons.core :refer [with-release]]
            [uncomplicate.diamond.tensor :refer [with-diamond *diamond-factory*]]
            [uncomplicate.diamond.internal.dnnl
             [core :refer [engine stream]]
             [factory :refer [dnnl-factory]]]
            [uncomplicate.diamond.tensor-test :refer :all]))

(with-release [eng (engine)
               strm (stream eng)
               diamond-factory (dnnl-factory eng strm)]

  (test-tensor diamond-factory)
  (test-create-tensor diamond-factory)
  (test-transformer diamond-factory)
  (test-pull-different diamond-factory)
  (test-pull-same diamond-factory)
  (test-push-different diamond-factory)
  (test-push-same diamond-factory)
  (test-subtensor diamond-factory)
  (test-shuffler diamond-factory))

(with-release [eng (engine)
               strm (stream eng)]
  (with-diamond dnnl-factory [eng strm]

    (test-tensor *diamond-factory*)
    (test-create-tensor *diamond-factory*)
    (test-transformer *diamond-factory*)
    (test-pull-different *diamond-factory*)
    (test-pull-same *diamond-factory*)
    (test-push-different *diamond-factory*)
    (test-push-same *diamond-factory*)
    (test-subtensor *diamond-factory*)
    (test-shuffler *diamond-factory*)))
