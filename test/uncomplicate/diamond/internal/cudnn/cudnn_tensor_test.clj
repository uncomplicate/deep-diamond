(ns uncomplicate.diamond.internal.cudnn.cudnn-tensor-test
  (:require [midje.sweet :refer [facts throws =>]]
            [uncomplicate.commons.core :refer [with-release]]
            [uncomplicate.diamond.tensor :refer [with-diamond *diamond-factory*]]
            [uncomplicate.diamond.internal.cudnn.factory :refer [cudnn-factory]]
            [uncomplicate.diamond.tensor-test :refer :all]))

(with-diamond cudnn-factory []

  (test-tensor *diamond-factory*)
  (test-create-tensor *diamond-factory*)
  #_(test-transformer *diamond-factory*)
  #_(test-pull-different *diamond-factory*)
  #_(test-pull-same *diamond-factory*)
  #_(test-push-different *diamond-factory*)
  #_(test-push-same *diamond-factory*)
  #_(test-subtensor *diamond-factory*)
  #_(test-shuffler *diamond-factory*))
