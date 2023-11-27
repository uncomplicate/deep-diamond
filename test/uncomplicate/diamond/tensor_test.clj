;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.diamond.tensor-test
  (:require [midje.sweet :refer [facts throws =>]]
            [uncomplicate.commons.core :refer [with-release release]]
            [uncomplicate.fluokitten.core :refer [fmap! fmap fold]]
            [uncomplicate.clojure-cpp :refer [position! pointer get-entry]]
            [uncomplicate.neanderthal
             [core :refer [asum view-vctr transfer! native entry entry! dim]]
             [block :refer [buffer contiguous?]]]
            [uncomplicate.diamond.tensor :refer :all]
            [uncomplicate.diamond.internal.protocols :refer [offset]])
  (:import clojure.lang.ExceptionInfo))

(defn test-tensor [fact]
  (with-release [tz (tensor fact [2 3 4 5] :float :nchw)]
    (facts "Basic tensor tests"
           (asum tz) => 0.0
           (asum (entry! tz 1)) => 120.0
           (shape tz) => [2 3 4 5])))

(defn test-tensor1 [fact]
  (with-release [tz (tensor fact [2 3 4 5] :float :nchw)]
    (
     (asum tz)
     (asum (entry! tz 1))
     (shape tz))))

(defn test-create [fact]
  (with-release [t1 (tensor fact [1 1 1 1] :float :nchw)
                 td3221 (desc [3 2 2 1] :float :nhwc)
                 t3221 (tensor fact td3221)]
    (facts "Basic tensor creation tests."
           (tensor fact [1 -1 1 1] :float :nchw) => (throws ExceptionInfo)
           (dim t1) => 1
           (shape t1) => [1 1 1 1]
           (layout t1) => [1 1 1 1]
           (data-type t1) => :float
           (dim t3221) => 12
           (layout t3221) => [4 1 2 2])))

(defn test-equality [fact]
  (with-release [x1 (tensor fact [2 1 2 3] :float :nchw)
                 y1 (tensor fact [2 1 2 3] :float :nchw)
                 y3 (tensor fact [2 1 2 3] :float :nhwc)
                 y4 (tensor fact [2 1 2 2] :float :nchw)
                 x5 (tensor fact [2 2 2 2] :float :nchw)
                 y5 (tensor fact [2 2 2 2] :float :nhwc)]
    (facts "Equality and hash code tests."
           (.equals x1 nil) => false
           (= x1 y1) => true
           (= x1 y3) => false
           (= x1 y4) => false
           (= x5 y5) => false
           (transfer! (range) x1) => (transfer! (range) y1))))

(defn test-release [fact]
  (let [t1 (tensor fact [2 3 1 1] :float :nchw)]
    (facts "Release tensor."
           (release (view-tz t1)) => true
           (release t1) => true)))

(defn test-transfer [fact0 fact1]
  (with-release [x1 (tensor fact0 [2 1 2 1] :float :nchw)
                 x2 (tensor fact0 [2 1 2 1] :float :nchw)
                 y1 (tensor fact1 [2 1 2 1] :float :nchw)
                 y2 (tensor fact1 [2 1 1 1] :float [1 1 1 1])]
    (facts "Tensor transfer."
           (transfer! (float-array [1 2 3 4]) x1) => x1
           (seq (transfer! x1 (float-array 4))) => [1.0 2.0 3.0 4.0]
           (seq (native (transfer! (float-array [4 3 2 1]) x1))) => [4.0 3.0 2.0 1.0]
           (seq (native (transfer! [10 20 30 40] x1))) => [10.0 20.0 30.0 40.0]
           (transfer! x1 x2) => x2
           (seq (native x2)) => [10.0 20.0 30.0 40.0]
           (transfer! x1 y1) => y1
           (seq (native y1)) => [10.0 20.0 30.0 40.0]
           (entry! y1 100) => y1
           (seq (native (transfer! y1 x1))) => [100.0 100.0 100.0 100.0]
           (transfer! x2 y2) => (throws ExceptionInfo))))

(defn test-contiguous [fact]
  (with-release [x1 (tensor fact [2 3 2 2] :float :nchw)]
    (facts "Test whether a tensor is contiguous."
           (contiguous? x1) => true
           (transfer! (range) x1)
           (view-tz x1 [2 3 2 2]) => x1
           (seq (native (view-tz x1 [1 3 2 2]))) => (range 0.0 12.0)
           (contiguous? x1) => true)))

(defn test-subtensor [fact]
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
           (position! (buffer sub-y) 3)
           (seq (native sub-y)) => [3.0 4.0 5.0]
           (seq (native sub-x)) => [0.0 1.0]
           (position! (buffer sub-z) 1)
           (seq (native sub-z)) => [1.0 2.0 3.0 4.0]
           (seq (native sub-x)) => [0.0 1.0])))

(defn test-tensor-fold [fact]
  (with-release [x (transfer! [1 2 3 4 5 6] (tensor fact [2 3 1 1] :float :nchw))
                 *' (fn ^double [^double x ^double y]
                      (* x y))
                 +' (fn ^double [^double x ^double y]
                      (+ x y))]
    (facts "Fold implementation for tensors."

           (fold x) => 21.0
           (fold *' 1.0 x) => 720.0
           (fold +' 0.0 x) => (fold x))))

(defn test-tensor-reducible [fact]
  (with-release [y (transfer! [1 2 3 4 5 6] (tensor fact [2 3 1 1] :float :nchw))
                 x (transfer! [10 20 30 40 50 60] (tensor fact [2 3 1 1] :float :nchw))
                 pf1 (fn ^double [^double res ^double x] (+ x res))
                 pf1o (fn [res ^double x] (conj res x))]
    (facts "Reducible implementation for tensors."

           (fold pf1 1.0 x) => 211.0
           (fold pf1o [] x) => [10.0 20.0 30.0 40.0 50.0 60.0]

           (fold pf1 1.0 x y) => 232.0
           (fold pf1o [] x y) => [11.0 22.0 33.0 44.0 55.0 66.0]

           (fold pf1 1.0 x y y) => 253.0
           (fold pf1o [] x y y) => [12.0 24.0 36.0 48.0 60.0 72.0]

           (fold pf1 1.0 x y y y) => 274.0
           (fold pf1o [] x y y y) => [13.0 26.0 39.0 52.0 65.0 78.0]

           (fold + 1.0 x y y y) => 274.0)))

(defn test-transformer [fact]
  (with-release [tz-x (tensor fact [2 3 4 5] :float :nchw)
                 tz-y (tensor fact [2 3 4 5] :float :nhwc)
                 tz-sub-x (view-tz tz-x [1 3 4 5])
                 tz-sub-y (view-tz tz-y [1 3 4 5])
                 transform (transformer tz-x tz-y)
                 sub-transform (transformer tz-sub-x tz-sub-y)]
    (facts "Tensor transformer"
           (entry (view-vctr (native (transfer! (range) tz-x))) 119) => 119.0
           (entry (view-vctr (native tz-y)) 119) => 0.0
           (buffer (input transform)) => (buffer tz-x)
           (buffer (output transform)) => (buffer tz-y)
           (transform) => tz-y
           (asum tz-y) => (asum tz-x)
           (entry (view-vctr (native tz-y)) 119) => 119.0
           (transfer! (range 0 1000 10) tz-x)
           (sub-transform) => tz-sub-y
           (entry (view-vctr (native tz-y)) 34) => 310.0)))

(defn test-pull-different [fact]
  (with-release [tz-x (tensor fact [2 3 4 5] :float :nchw)
                 tz-y-desc (desc [2 3 4 5] :float :nhwc)
                 connection (connector tz-x tz-y-desc)]
    (facts "Tensor pull connector with different destination"
           (entry (view-vctr (native (transfer! (range) tz-x))) 119) => 119.0
           (= (buffer (input connection)) (buffer tz-x)) => true
           (= (buffer (input connection)) (buffer (output connection))) => false
           (entry (native (view-vctr (connection))) 119) => 119.0)))

(defn test-pull-same [fact]
  (with-release [tz-x (tensor fact [2 3 4 5] :float :nchw)
                 tz-y-desc (desc [2 3 4 5] :float :nchw)
                 connection (connector tz-x tz-y-desc)]
    (facts "Tensor pull connector with the same destination"
           (entry (view-vctr (native (transfer! (range) tz-x))) 119) => 119.0
           (= (buffer (input connection)) (buffer tz-x)) => true
           (= (buffer (input connection)) (buffer (output connection))) => true
           (entry (native (view-vctr (connection))) 119) => 119.0)))

(defn test-push-different [fact]
  (with-release [tz-y (tensor fact [2 3 4 5] :float :nchw)
                 tz-x-desc (desc [2 3 4 5] :float :nhwc)
                 connection (connector tz-x-desc tz-y)]
    (facts "Tensor push connector with different destination"
           (entry (native (transfer! (range) (view-vctr (input connection)))) 119) => 119.0
           (= (buffer (output connection)) (buffer tz-y)) => true
           (= (buffer (input connection)) (buffer (output connection))) => false
           (entry (native (view-vctr (connection))) 119) => 119.0)))

(defn test-push-same [fact]
  (with-release [tz-y (tensor fact [2 3 4 5] :float :nchw)
                 tz-x-desc (desc [2 3 4 5] :float :nchw)
                 connection (connector tz-x-desc tz-y)]
    (facts "Tensor push connector with the same destination"
           (entry (native (transfer! (range) (view-vctr (input connection)))) 119) => 119.0
           (= (buffer (output connection)) (buffer tz-y)) => true
           (= (buffer (input connection)) (buffer (output connection))) => true
           (entry (native (view-vctr (connection))) 119) => 119.0)))

(defn test-shuffler [fact]
  (with-release [tz-x (tensor fact [6 2 1 1] :float [2 1 1 1])
                 tz-y (tensor fact [3 2 1 1] :float [1 3 1 1])
                 shuff (shuffler tz-x tz-y)]
    (facts "shuffler test."
           (transfer! (range 1 13) tz-x)
           (seq (native tz-x)) => (range 1.0 13.0)
           (seq (native tz-y)) => [0.0 0.0 0.0 0.0 0.0 0.0]
           (shuff [0 2 1])
           (seq (native tz-y)) => [1.0 5.0 3.0 2.0 6.0 4.0]
           (shuff [0 2 1 1]) => (throws ExceptionInfo)
           (shuff [0 2 8]) => (throws ExceptionInfo)
           (shuff [0 1]) => tz-y)))

(defn test-batcher [fact]
  (with-release [tz-x (tensor fact [7 2 1 1] :float [2 1 1 1])
                 tz-y (tensor fact [3 2 1 1] :float [1 3 1 1])
                 batch (batcher tz-x tz-y 3)
                 batch-2 (batcher tz-x tz-y 2)]
    (facts "batcher test."
           (transfer! (range 1 15) tz-x)
           (seq (native tz-x)) => (range 1.0 15.0)
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

(defn test-batcher-tnc [fact]
  (with-release [tz-x-ntc (tensor fact [2 7 1] :float :ntc)
                 tz-x (tensor fact [2 7 1] :float :tnc)
                 tz-y (tensor fact [2 3 1] :float :tnc)
                 sub-x (view-tz tz-x 3)
                 batch (batcher tz-x tz-y 3)
                 batch-2 (batcher tz-x tz-y 2)]
    (facts "batcher test."
           (transfer! (range 1 15) tz-x)
           (seq (native (transfer! sub-x (tensor fact [2 3 1] :float :tnc)))) => [1.0 2.0 3.0 8.0 9.0 10.0]
           (seq (native (transfer! (doto sub-x (offset! 3))
                                   (tensor fact [2 3 1] :float :tnc)))) => [4.0 5.0 6.0 11.0 12.0 13.0]
           (transfer! (range 1 15) tz-x-ntc)
           (transfer! tz-x-ntc tz-x)
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
