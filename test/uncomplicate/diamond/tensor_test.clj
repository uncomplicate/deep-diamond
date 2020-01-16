(ns uncomplicate.diamond.tensor-test
  (:require [midje.sweet :refer [facts throws =>]]
            [uncomplicate.commons
             [core :refer [with-release release]]]
            [uncomplicate.neanderthal
             [core :refer [asum view transfer! native entry entry! dim]]
             [block :refer [buffer contiguous?]]]
            [uncomplicate.diamond.tensor :refer :all])
  (:import clojure.lang.ExceptionInfo))

(defn test-tensor [factory]
  (with-release [tz (tensor factory [2 3 4 5] :float :nchw)]
    (facts "Basic tensor tests"
           (asum tz) => 0.0
           (asum (entry! tz 1)) => 120.0
           (shape tz) => [2 3 4 5])))

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
                 y4 (tensor fact [2 1 2 2] :float :nchw)]
    (facts "Equality and hash code tests."
           (.equals x1 nil) => false
           (= x1 y1) => true
           (= x1 y3) => false
           (= x1 y4) => false
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
                 y2 (tensor fact1 [2 1] :float :nc)]
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

(defn test-transformer [factory]
  (with-release [tz-x (tensor factory [2 3 4 5] :float :nchw)
                 tz-y (tensor factory [2 3 4 5] :float :nhwc)
                 transform (transformer tz-x tz-y)]
    (facts "Tensor transformator"
           (entry (view (native (transfer! (range) tz-x))) 119) => 119.0
           (entry (view (native tz-y)) 119) => 0.0
           (buffer (input transform)) => (buffer tz-x)
           (buffer (output transform)) => (buffer tz-y)
           (transform) => tz-y
           (entry (view (native tz-y)) 119) => 119.0)))

(defn test-pull-different [factory]
  (with-release [tz-x (tensor factory [2 3 4 5] :float :nchw)
                 tz-y-desc (desc [2 3 4 5] :float :nhwc)
                 connection (connector tz-x tz-y-desc)]
    (facts "Tensor pull connector with different destination"
           (entry (view (transfer! (range) tz-x)) 119) => 119.0
           (identical? (buffer (input connection)) (buffer tz-x)) => true
           (identical? (buffer (input connection)) (buffer (output connection))) => false
           (entry (view (connection)) 119) => 119.0)))

(defn test-pull-same [factory]
  (with-release [tz-x (tensor factory [2 3 4 5] :float :nchw)
                 tz-y-desc (desc [2 3 4 5] :float :nchw)
                 connection (connector tz-x tz-y-desc)]
    (facts "Tensor pull connector with the same destination"
           (entry (view (transfer! (range) tz-x)) 119) => 119.0
           (identical? (buffer (input connection)) (buffer tz-x)) => true
           (identical? (buffer (input connection)) (buffer (output connection))) => true
           (entry (view (connection)) 119) => 119.0)))

(defn test-push-different [factory]
  (with-release [tz-y (tensor factory [2 3 4 5] :float :nchw)
                 tz-x-desc (desc [2 3 4 5] :float :nhwc)
                 connection (connector tz-x-desc tz-y)]
    (facts "Tensor push connector with different destination"
           (entry (transfer! (range) (view (input connection))) 119) => 119.0
           (identical? (buffer (output connection)) (buffer tz-y)) => true
           (identical? (buffer (input connection)) (buffer (output connection))) => false
           (entry (view (connection)) 119) => 119.0)))

(defn test-push-same [factory]
  (with-release [tz-y (tensor factory [2 3 4 5] :float :nchw)
                 tz-x-desc (desc [2 3 4 5] :float :nchw)
                 connection (connector tz-x-desc tz-y)]
    (facts "Tensor push connector with the same destination"
           (entry (transfer! (range) (view connection)) 119) => 119.0
           (identical? (buffer (output connection)) (buffer tz-y)) => true
           (identical? (buffer (input connection)) (buffer (output connection))) => true
           (entry (view (connection)) 119) => 119.0)))

(defn test-subtensor [factory]
  (with-release [tz-x (tensor factory [6] :float :x)
                 sub-x (view-tz tz-x [2])
                 sub-y (view-tz tz-x (desc [1 3] :nc))
                 sub-z (view-tz tz-x 4)]
    (facts "Test subtensors and offsets."
           (transfer! (range) tz-x)
           (seq tz-x) => [0.0 1.0 2.0 3.0 4.0 5.0]
           (seq sub-x) => [0.0 1.0]
           (seq sub-y) => [0.0 1.0 2.0]
           (seq sub-z) => [0.0 1.0 2.0 3.0]
           (uncomplicate.diamond.internal.dnnl.core/offset! (buffer sub-y) Float/BYTES);;TODO generalize
           (seq sub-y) => [1.0 2.0 3.0]
           (seq sub-x) => [0.0 1.0]
           (uncomplicate.diamond.internal.dnnl.core/offset! (buffer sub-z) Float/BYTES)
           (seq sub-z) => [1.0 2.0 3.0 4.0]
           (seq sub-x) => [0.0 1.0])))

(defn test-shuffler [factory]
  (with-release [tz-x (tensor factory [6 2] :float :nc)
                 tz-y (tensor factory [3 2] :float :cn)
                 shuff (shuffler tz-x tz-y)]
    (facts "shuffler test."
           (transfer! (range 1 13) tz-x)
           (seq tz-x) => (range 1.0 13.0)
           (seq tz-y) => [0.0 0.0 0.0 0.0 0.0 0.0]
           (shuff [0 2 1])
           (seq tz-y) => [1.0 5.0 3.0 2.0 6.0 4.0]
           (shuff [0 2 1 1]) => (throws ExceptionInfo)
           (shuff [0 2 8]) => (throws ExceptionInfo)
           (shuff [0 1]) => tz-y)))

(defn test-batcher [factory]
  (with-release [tz-x (tensor factory [7 2] :float :nc)
                 tz-y (tensor factory [3 2] :float :cn)
                 batch (batcher tz-x tz-y 3)
                 batch-2 (batcher tz-x tz-y 2)]
    (facts "batcher test."
           (transfer! (range 1 15) tz-x)
           (seq tz-x) => (range 1.0 15.0)
           (seq tz-y) => (repeat 6 0.0)
           (batch 0 0) => tz-y
           (seq tz-y) => [1.0 3.0 5.0 2.0 4.0 6.0]
           (transfer! (repeat 0) tz-y)
           (batch 1 0) => tz-y
           (seq tz-y) => [3.0 5.0 7.0 4.0 6.0 8.0]
           (transfer! (repeat 0) tz-y)
           (batch-2 1 1) => tz-y
           (seq tz-y) => [0.0 3.0 5.0 0.0 4.0 6.0]
           (batch 8) => (throws ExceptionInfo)
           (batch 0 -1) => (throws ExceptionInfo)
           (batch 7 -1) => (throws ExceptionInfo)
           (batch -1) => (throws ExceptionInfo))))
