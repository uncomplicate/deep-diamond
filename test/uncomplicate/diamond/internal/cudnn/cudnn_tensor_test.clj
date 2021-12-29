;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.diamond.internal.cudnn.cudnn-tensor-test
  (:require [midje.sweet :refer [facts throws =>]]
            [uncomplicate.commons.core :refer [with-release]]
            [uncomplicate.neanderthal.core :refer [dim asum native transfer!]]
            [uncomplicate.diamond.tensor :refer [with-diamond *diamond-factory* tensor offset! view-tz]]
            [uncomplicate.diamond.internal.cudnn.factory :refer [cudnn-factory]]
            [uncomplicate.diamond.internal.dnnl.factory :refer [dnnl-factory]]
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

(defn test-cudnn-transfer [fact0 fact1]
  (with-release [tz-x (tensor fact0 [6 2] :byte :nc)
                 sub-x (offset! (view-tz tz-x 2) 4)
                 tz-y (tensor fact1 [6 2] :float :nc)
                 sub-y (offset! (view-tz tz-y 2) 2)
                 tz-z (tensor fact0 [6 2] :uint8 :nc)
                 sub-z (offset! (view-tz tz-z 2) 1)]
    (facts "Test heterogenous transfer."
           (transfer! (range -6 6) tz-x)
           (seq (native tz-x)) => (range -6 6)
           (asum (native (transfer! tz-x tz-y))) => 36.0
           (seq (native (transfer! tz-y tz-z))) => [0 0 0 0 0 0 0 1 2 3 4 5]
           (asum (native (transfer! sub-x sub-y))) => 14.0
           (seq (native (transfer! sub-y sub-z))) => [2 3 4 5]
           (seq (native tz-z)) => [0 0 2 3 4 5 0  1 2 3 4 5])))

(with-release [dnnl-fact (dnnl-factory)]
  (with-diamond cudnn-factory []
    (test-tensor *diamond-factory*)
    (test-create *diamond-factory*)
    (test-cudnn-create *diamond-factory*)
    (test-equality *diamond-factory*)
    (test-release *diamond-factory*)
    (test-transfer *diamond-factory* dnnl-fact)
    (test-contiguous *diamond-factory*)
    (test-subtensor *diamond-factory*)
    (test-transformer *diamond-factory*)
    (test-pull-different *diamond-factory*)
    (test-pull-same *diamond-factory*)
    (test-push-different *diamond-factory*)
    (test-push-same *diamond-factory*)
    (test-batcher *diamond-factory*)
    (test-shuffler *diamond-factory*)
    (test-cudnn-transfer dnnl-fact *diamond-factory*)))
