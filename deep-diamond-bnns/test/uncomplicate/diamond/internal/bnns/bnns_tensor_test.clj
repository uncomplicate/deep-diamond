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
            [uncomplicate.clojure-cpp :refer [position! pointer-seq]]
            [uncomplicate.commons.core :refer [with-release bytesize]]
            [uncomplicate.neanderthal
             [core :refer [dim view-vctr transfer! native entry! entry asum]]
             [block :refer [buffer contiguous? initialize!]]]
            [uncomplicate.diamond.tensor
             :refer [with-diamond *diamond-factory* tensor transformer layout
                     shape data-type desc view-tz input output connector revert
                     batcher shuffler offset!]]
            [uncomplicate.diamond.internal.bnns [factory :refer [bnns-factory]]]
            [uncomplicate.diamond.internal [protocols :refer [offset]]]
            [uncomplicate.diamond.tensor-test :refer :all])
  (:import clojure.lang.ExceptionInfo))

(defn test-bnns-create [fact]
  (facts
    "Test BNNS specific constraints."
    (with-release [t0 (tensor fact [0 1 1 1] :float :nchw)
                   tnc (tensor fact [2 3] :float :nc)
                   non-contiguous-tensor (tensor fact [2 3 2 1] :float [48 8 2 2])]

      (dim t0) => 0
      (layout t0) => :abcd
      (dim tnc) => 6
      (shape tnc) => [2 3]
      (layout tnc) => [3 1]
      (data-type tnc) => :float
      (view-vctr non-contiguous-tensor) => (throws ExceptionInfo)
      (tensor fact [2 3] :double :nc) => (throws ExceptionInfo)
      (tensor fact [2 3] :int :nc) => truthy
      (tensor fact [2 3] :long :nc) => (throws ExceptionInfo)
      (tensor fact [1 -1 1 1] :float :nchw) => (throws ExceptionInfo))))

(defn test-bnns-equality [fact]
  (with-release [x1 (tensor fact [2 1 2 3] :float :abcd)
                 y1 (tensor fact [2 1 2 3] :float :abcd)
                 y3 (tensor fact [2 1 2 3] :float :dcba)
                 y4 (tensor fact [2 1 2 2] :float :dcba)
                 x5 (tensor fact [2 2 2 2] :float :dcba)
                 y5 (tensor fact [2 2 2 2] :float :abcd)]
    (facts "BNNS Equality and hash code tests."
           (.equals x1 nil) => false
           (= x1 y1) => true
           (= x1 y3) => false
           (= x1 y4) => false
           (= x5 y5) => false
           (transfer! (range) x1) => (transfer! (range) y1))))

(defn test-bnns-transformer [fact]
  (with-release [tz-x (tensor fact [2 3 4 5] :float :nchw)
                 tz-y (tensor fact [2 3 4 5] :float :nchw)
                 tz-sub-x (view-tz tz-x [1 3 4 5])
                 tz-sub-y (view-tz tz-y [1 3 4 5])
                 transform (transformer tz-x tz-y)
                 sub-transform (transformer tz-sub-x tz-sub-y)]
    (facts "Tensor BNNS transformer."
           (entry (view-vctr (native (transfer! (range) tz-x))) 119) => 119.0
           (entry (view-vctr (native tz-y)) 119) => 0.0
           (buffer (input transform)) => (buffer tz-x)
           (buffer (output transform)) => (buffer tz-y)
           (transform) => tz-y
           (asum tz-y) => (asum tz-x)
           (entry (view-vctr (native tz-y)) 119) => 119.0
           (transfer! (range 0 1000 10) tz-x)
           (sub-transform) => tz-sub-y
           (seq (view-vctr (native tz-y))) => (into (vec (range 0.0 600.0 10.0)) (range 60.0 120 1.0))
           (offset tz-sub-y 60)
           (sub-transform)
           (seq (view-vctr (native tz-y))) => (into (vec (range 0.0 600.0 10.0)) (range 0.0 600.0 10.0)))))

(defn test-bnns-transformer-any [fact]
  (with-release [tz-x (tensor fact [2 3 2 1] :float [48 14 4 2]) ;; TODO strides must have sense! In the original dnnl/cuda test, the memory is overlapping!
                 in-x (connector (desc [2 3 2 1] :float :nchw) tz-x)
                 out-x (revert in-x)
                 tz-y (tensor fact [2 3 2 1] :float [48 12 4 1])
                 out-y (connector tz-y (desc [2 3 2 1] :float :nchw))
                 transform (transformer tz-x tz-y)]
    (facts "Tensor BNNS transformer for arbitrary strides."
           (transfer! (range) (input in-x))
           (in-x)
           (transfer! (repeat 10 0.0) (input in-x))
           (seq (out-x)) => (map float (range 0 12))
           (transform)
           (out-y)
           (seq (output out-y)) => (seq (input in-x)))))

(defn test-bnns-batcher [fact]
  (with-release [tz-x (tensor fact [7 2 1 1] :float [2 1 1 1])
                 tz-y (tensor fact [3 2 1 1] :float [4 2 1 1])
                 batch (batcher tz-x tz-y 3)
                 batch-1 (batcher tz-x tz-y 1)
                 batch-2 (batcher tz-x tz-y 2)]
    (facts "BNNS batcher test."
           (transfer! (range 1 15) tz-x)
           (seq (transfer! tz-x (float-array 14))) => (range 1.0 15.0)
           (batch 0 0) => tz-y
           (seq (transfer! tz-y (float-array 6))) => (range 1.0 7.0)
           (transfer! (repeat 0) tz-y)
           (batch 1 0) => tz-y
           (seq (transfer! tz-y (float-array 6))) => (range 3.0 9.0)
           (batch-2 0 0) => tz-y
           (seq (transfer! tz-y (float-array 6))) => [1.0 2.0 3.0 4.0 7.0 8.0]
           (transfer! (repeat 0) tz-y)
           (batch-1 1 0) => tz-y
           (seq (transfer! tz-y (float-array 6))) => [3.0 4.0 0.0 0.0 0.0 0.0]
           (transfer! (repeat 0) tz-y)
           (batch-1 1 1) => tz-y
           (seq (transfer! tz-y (float-array 6))) => [0.0 0.0 3.0 4.0 0.0 0.0]
           (batch-1 2 0) => tz-y
           (seq (transfer! tz-y (float-array 6))) => [5.0 6.0 3.0 4.0 0.0 0.0]
           (batch-1 3 2) => tz-y
           (seq (transfer! tz-y (float-array 6))) => [5.0 6.0 3.0 4.0 7.0 8.0]
           (batch 8) => (throws ExceptionInfo)
           (batch 0 -1) => (throws ExceptionInfo)
           (batch 7 -1) => (throws ExceptionInfo)
           (batch -1) => (throws ExceptionInfo))))

(defn test-bnns-pull-different [fact]
  (with-release [tz-x (tensor fact [2 3 4 5] :float :abcd)
                 tz-y-desc (desc [2 3 4 5] :int :abcd)
                 connection (connector tz-x tz-y-desc)]
    (facts "BNNS Tensor pull connector with different destination"

           (entry (view-vctr (native (transfer! (range) tz-x))) 119) => 119.0
           (= (buffer (input connection)) (buffer tz-x)) => true
           (= (buffer (input connection)) (buffer (output connection))) => false
           (entry (native (view-vctr (connection))) 119) => 119)))

(defn test-bnns-push-different [fact]
  (with-release [tz-y (tensor fact [2 3 4 5] :float :abcd)
                 tz-x-desc (desc [2 3 4 5] :int :abcd)
                 connection (connector tz-x-desc tz-y)]
    (facts "BNNS Tensor push connector with different destination"
           (entry (native (transfer! (range) (view-vctr (input connection)))) 119) => 119
           (= (buffer (output connection)) (buffer tz-y)) => true
           (= (buffer (input connection)) (buffer (output connection))) => false
           (entry (native (view-vctr (connection))) 119) => 119.0)))

(defn test-bnns-shuffler [fact]
  (with-release [tz-x (tensor fact [6 2 1 1] :float [2 1 1 1])
        tz-y (tensor fact [3 2 1 1] :int [4 2 1 1])
        shuff (shuffler tz-x tz-y)]
    (facts "BNNS shuffler test."
           (transfer! (range 1 13) tz-x)
           (seq (transfer! tz-x (float-array 12))) => (range 1.0 13.0)
           (shuff [2 0 1])
           (seq (transfer! tz-y (int-array 6))) => [5 6 1 2 3 4]
           (shuff [0 2 1 1]) => (throws ExceptionInfo)
           (shuff [0 2 8]) => (throws ExceptionInfo)
           (shuff [0 1]) => tz-y)))

(defn test-bnns-batcher-tnc [fact]
  (with-release [tz-x (tensor fact [2 7 1] :float :tnc)
                 tz-y (tensor fact [2 3 1] :float :tnc)
                 sub-x (view-tz tz-x 3)
                 batch (batcher tz-x tz-y 3)
                 batch-2 (batcher tz-x tz-y 2)]
    (facts "BNNS batcher test."
           (transfer! [1.0 3.0 5.0 7.0 9.0 11.0 13.0 2.0 4.0 6.0 8.0 10.0 12.0 14.0] tz-x)

           (seq (native tz-x)) => [1.0 3.0 5.0 7.0 9.0 11.0 13.0 2.0 4.0 6.0 8.0 10.0 12.0 14.0]
           (seq (native tz-y)) => (repeat 6 0.0)
           (batch 0 0) => tz-y
           (seq (native tz-y)) => [1.0 3.0 5.0 2.0 4.0 6.0]
           (transfer! (repeat 0) tz-y)
           (batch 1 0) => tz-y
           (seq (native tz-y)) => [3.0 5.0 7.0 4.0 6.0 8.0]
           (transfer! (repeat 0) tz-y)
           (batch-2 1 1) => tz-y
           (seq (native tz-y)) => [0.0 3.0 5.0 0.0 4.0 6.0]
           (batch 8) => (throws ExceptionInfo)
           (batch 0 -1) => (throws ExceptionInfo)
           (batch 7 -1) => (throws ExceptionInfo)
           (batch -1) => (throws ExceptionInfo))))

(with-release [diamond-factory (bnns-factory)]
  (test-bnns-create diamond-factory)
  (test-bnns-equality diamond-factory)
  (test-tensor diamond-factory)
  (test-zero diamond-factory)
  (test-release diamond-factory)
  (test-transfer diamond-factory diamond-factory)
  (test-contiguous diamond-factory)
  (test-subtensor diamond-factory)
  (test-bnns-transformer diamond-factory)
  (test-transformer-any diamond-factory)
  (test-bnns-transformer-any diamond-factory)
  (test-transfer-any diamond-factory)
  (test-bnns-pull-different diamond-factory)
  (test-pull-same diamond-factory)
  (test-bnns-push-different diamond-factory)
  (test-push-same diamond-factory)
  (test-bnns-batcher diamond-factory)
  (test-bnns-shuffler diamond-factory)
  (test-bnns-batcher-tnc diamond-factory)
  (test-tensor-fold diamond-factory)
  (test-tensor-reducible diamond-factory))

(with-diamond bnns-factory []
  (test-bnns-create *diamond-factory*)
  (test-bnns-equality *diamond-factory*)
  (test-tensor *diamond-factory*)
  (test-zero *diamond-factory*)
  (test-release *diamond-factory*)
  (test-transfer *diamond-factory* *diamond-factory*)
  (test-contiguous *diamond-factory*)
  (test-subtensor *diamond-factory*)
  (test-bnns-transformer *diamond-factory*)
  (test-transformer-any *diamond-factory*)
  (test-bnns-transformer-any *diamond-factory*)
  (test-transfer-any *diamond-factory*)
  (test-bnns-pull-different *diamond-factory*)
  (test-pull-same *diamond-factory*)
  (test-bnns-push-different *diamond-factory*)
  (test-push-same *diamond-factory*)
  (test-bnns-batcher *diamond-factory*)
  (test-bnns-shuffler *diamond-factory*)
  (test-bnns-batcher-tnc *diamond-factory*)
  (test-tensor-fold *diamond-factory*)
  (test-tensor-reducible *diamond-factory*))
