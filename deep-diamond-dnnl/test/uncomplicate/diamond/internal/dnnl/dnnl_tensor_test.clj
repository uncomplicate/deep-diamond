;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.diamond.internal.dnnl.dnnl-tensor-test
  (:require [midje.sweet :refer [facts throws => truthy]]
            [uncomplicate.commons.core :refer [with-release]]
            [uncomplicate.neanderthal.core :refer [dim view-vctr transfer!]]
            [uncomplicate.diamond.tensor :refer [with-diamond *diamond-factory* tensor]]
            [uncomplicate.diamond.internal.dnnl
             [core :refer [engine stream]]
             [factory :refer [dnnl-factory]]]
            [uncomplicate.diamond.tensor-test :refer :all])
  (:import clojure.lang.ExceptionInfo))

(defn test-dnnl-create [fact]
  (facts
   "Test DNNL specific constraints."
   (with-release [t0 (tensor fact [0 1 1 1] :float :nchw)
                  tnc (tensor fact [2 3] :float :nc)
                  non-contiguous-tensor (tensor fact [2 3 2 1] :float [48 8 2 2])]
     (dim t0) => 0
     (dim tnc) => 6
     (view-vctr non-contiguous-tensor) => (throws ExceptionInfo)
     (tensor fact [2 3] :double :nc) => (throws ExceptionInfo)
     (tensor fact [2 3] :int :nc) => truthy
     (tensor fact [2 3] :long :nc) => (throws ExceptionInfo))))

(with-release [eng (engine)
               strm (stream eng)
               diamond-factory (dnnl-factory eng strm)
               eng1 (engine)
               strm1 (stream eng1)
               diamond-factory1 (dnnl-factory eng1 strm1)]

  (test-tensor diamond-factory)
  (test-zero diamond-factory)
  (test-create diamond-factory)
  (test-dnnl-create diamond-factory)
  (test-equality diamond-factory)
  (test-release diamond-factory)
  (test-transfer diamond-factory diamond-factory)
  (test-contiguous diamond-factory)
  (test-subtensor diamond-factory)
  (test-transformer diamond-factory)
  (test-transformer-any diamond-factory)
  (test-transfer-any diamond-factory)
  (test-transfer-view-tz diamond-factory)
  (test-pull-different diamond-factory)
  (test-pull-same diamond-factory)
  (test-push-different diamond-factory)
  (test-push-same diamond-factory)
  (test-batcher diamond-factory)
  (test-shuffler diamond-factory)
  (test-batcher-tnc diamond-factory)
  (test-tensor-fold diamond-factory)
  (test-tensor-reducible diamond-factory)
  (test-heterogenous-transfer diamond-factory diamond-factory1))

(with-release [eng (engine)
               strm (stream eng)]
  (with-diamond dnnl-factory [eng strm]

    (test-tensor *diamond-factory*)
    (test-zero *diamond-factory*)
    (test-create *diamond-factory*)
    (test-dnnl-create *diamond-factory*)
    (test-equality *diamond-factory*)
    (test-release *diamond-factory*)
    (test-transfer *diamond-factory* *diamond-factory*)
    (test-contiguous *diamond-factory*)
    (test-subtensor *diamond-factory*)
    (test-transformer *diamond-factory*)
    (test-transformer-any *diamond-factory*)
    (test-transfer-any *diamond-factory*)
    (test-transfer-view-tz *diamond-factory*)
    (test-pull-different *diamond-factory*)
    (test-pull-same *diamond-factory*)
    (test-push-different *diamond-factory*)
    (test-push-same *diamond-factory*)
    (test-batcher *diamond-factory*)
    (test-shuffler *diamond-factory*)
    (test-batcher-tnc *diamond-factory*)
    (test-tensor-fold *diamond-factory*)
    (test-tensor-reducible *diamond-factory*)
    (test-heterogenous-transfer *diamond-factory* *diamond-factory*)))
