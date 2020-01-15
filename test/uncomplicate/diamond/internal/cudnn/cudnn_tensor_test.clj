(ns uncomplicate.diamond.internal.cudnn.cudnn-tensor-test
  (:require [midje.sweet :refer [facts throws =>]]
            [uncomplicate.commons.core :refer [with-release]]
            [uncomplicate.neanderthal.core :refer [dim]]
            [uncomplicate.diamond.tensor :refer [with-diamond *diamond-factory* tensor]]
            [uncomplicate.diamond.internal.cudnn.factory :refer [cudnn-factory]]
            [uncomplicate.diamond.tensor-test :refer :all])
  (:import clojure.lang.ExceptionInfo))

(defn test-cudnn-create [fact]
  (facts
   "Test cuDNN specific constraints."
   (tensor fact [0 1 1 1] :float :nchw) => (throws ExceptionInfo)
   (tensor fact [2 3] :int :nc) => (throws ExceptionInfo)
   (tensor fact [2 3] :long :nc) => (throws ExceptionInfo)
   (with-release [t1 (tensor fact [2 3 2 2] :double :nchw)]
     (dim t1) => 24)))

(with-diamond cudnn-factory []

  (test-tensor *diamond-factory*)
  (test-create *diamond-factory*)
  (test-cudnn-create *diamond-factory*)
  (test-equality *diamond-factory*)
  #_(test-transformer *diamond-factory*)
  #_(test-pull-different *diamond-factory*)
  #_(test-pull-same *diamond-factory*)
  #_(test-push-different *diamond-factory*)
  #_(test-push-same *diamond-factory*)
  #_(test-subtensor *diamond-factory*)
  #_(test-shuffler *diamond-factory*))
