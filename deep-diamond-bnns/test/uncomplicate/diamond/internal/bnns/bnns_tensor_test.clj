;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.diamond.internal.bnns.bnns-tensor-test
  (:require [midje.sweet :refer [facts throws => truthy]]
            [uncomplicate.clojure-cpp :refer [position!]]
            [uncomplicate.commons.core :refer [with-release bytesize]]
            [uncomplicate.neanderthal.core :refer [dim view-vctr transfer! native]]
            [uncomplicate.diamond.tensor
             :refer [with-diamond *diamond-factory* tensor layout shape data-type desc]]
            [uncomplicate.diamond.internal.bnns
             [factory :refer [bnns-factory]]]
            [uncomplicate.diamond.tensor-test :refer :all])
  (:import clojure.lang.ExceptionInfo))

(defn test-bnns-create [fact]
  (facts
    "Test BNNS specific constraints."
    (with-release [t0 (tensor fact [0 1 1 1] :float :nchw)
                   tnc (tensor fact [2 3] :float :nc)
                   non-contiguous-tensor (tensor fact [2 3 2 1] :float [48 8 2 2])]

      (dim t0) => 0
      (layout t0) => [0 0 0 0]
      (dim tnc) => 6
      (shape tnc) => [2 3]
      (layout tnc) => [0 0]
      (data-type tnc) => :float
      (view-vctr non-contiguous-tensor) => (throws ExceptionInfo)
      (tensor fact [2 3] :double :nc) => (throws ExceptionInfo)
      (tensor fact [2 3] :int :nc) => truthy
      (tensor fact [2 3] :long :nc) => (throws ExceptionInfo)
      (tensor fact [1 -1 1 1] :float :nchw) => (throws ExceptionInfo))))

(defn test-bnns-equality [fact]
  (with-release [x1 (tensor fact [2 1 2 3] :float :4d-last)
                 y1 (tensor fact [2 1 2 3] :float :4d-last)
                 y3 (tensor fact [2 1 2 3] :float :4d-first)
                 y4 (tensor fact [2 1 2 2] :float :4d-first)
                 x5 (tensor fact [2 2 2 2] :float :4d-first)
                 y5 (tensor fact [2 2 2 2] :float :4d-last)]
    (facts "Equality and hash code tests."
           (.equals x1 nil) => false
           (= x1 y1) => true
           (= x1 y3) => false
           (= x1 y4) => false
           (= x5 y5) => false
           (transfer! (range) x1) => (transfer! (range) y1))))


(defn test-bnns-subtensor [fact]
  (with-release [tz-x (tensor fact [6 1 1 1] :float [1 1 1 1])
                 sub-x (view-tz tz-x [2 1 1 1])
                 sub-y (view-tz tz-x (desc [1 3 1 1] [3 1 1 1]))
                 sub-z (view-tz tz-x 4)]
    (facts "Test subtensors and offsets."
           (transfer! (range) tz-x)
           (seq (native tz-x)) => [0.0 1.0 2.0 3.0 4.0 5.0]
           (seq (native sub-x)) => [0.0 1.0]
           (seq (native sub-y)) => [0.0 1.0 2.0]
           (seq (native sub-z)) => [0.0 1.0 2.0 3.0]
           (position! (.buffer sub-y) 3)
           (seq (native sub-y)) => [3.0 4.0 5.0]
           (seq (native sub-x)) => [0.0 1.0]
           (position! (.buffer sub-z) 1)
           (seq (native sub-z)) => [1.0 2.0 3.0 4.0]
           (seq (native sub-x)) => [0.0 1.0]
           (seq (native tz-x)) => [0.0 1.0 2.0 3.0 4.0 5.0]
    ;;       (initialize! sub-y (buffer sub-y))
           (seq (native tz-x)) => [0.0 1.0 2.0 0.0 0.0 0.0])))


(with-release [diamond-factory (bnns-factory)]
  (test-bnns-create diamond-factory)
  (test-tensor diamond-factory)
  (test-zero diamond-factory)
  (test-bnns-equality diamond-factory)
  (test-release diamond-factory)
  ;; (test-transfer diamond-factory diamond-factory)
  (test-contiguous diamond-factory)
  (test-bnns-subtensor diamond-factory)
  ;; (test-transformer diamond-factory)
  ;; (test-transformer-any diamond-factory)
  ;; (test-transfer-any diamond-factory)
  ;; (test-pull-different diamond-factory)
  ;; (test-pull-same diamond-factory)
  ;; (test-push-different diamond-factory)
  ;; (test-push-same diamond-factory)
  ;; (test-batcher diamond-factory)
  ;; (test-shuffler diamond-factory)
  ;; (test-batcher-tnc diamond-factory)
  ;; (test-tensor-fold diamond-factory)
  ;; (test-tensor-reducible diamond-factory)
  )

#_(with-release [eng (engine)
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
      (test-pull-different *diamond-factory*)
      (test-pull-same *diamond-factory*)
      (test-push-different *diamond-factory*)
      (test-push-same *diamond-factory*)
      (test-batcher *diamond-factory*)
      (test-shuffler *diamond-factory*)
      (test-batcher-tnc *diamond-factory*)
      (test-tensor-fold *diamond-factory*)
      (test-tensor-reducible *diamond-factory*)))
