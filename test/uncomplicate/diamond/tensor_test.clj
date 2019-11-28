(ns uncomplicate.diamond.tensor-test
  (:require [midje.sweet :refer [facts throws =>]]
            [uncomplicate.commons
             [core :refer [with-release]]]
            [uncomplicate.neanderthal
             [core :refer [asum view transfer! native]]
             [real :refer [entry! entry]]
             [block :refer [buffer]]]
            [uncomplicate.diamond.internal.protocols :as api]
            [uncomplicate.diamond.tensor :refer :all])
  (:import clojure.lang.ExceptionInfo))

(defn test-tensor [factory]
  (facts
   "Basic tensor tests"
   (with-release [tz (tensor factory [2 3 4 5] :float :nchw)]
     (asum (view tz)) => 0.0
     (asum (entry! (view tz) 1)) => 120.0
     (shape tz) => [2 3 4 5])))

(defn test-transformer [factory]
  (facts
   "Tensor transformator"
   (with-release [tz-x (tensor factory [2 3 4 5] :float :nchw)
                  tz-y (tensor factory [2 3 4 5] :float :nhwc)
                  transform (transformer tz-x tz-y)]
     (entry (native (transfer! (range) (view tz-x))) 119) => 119.0
     (entry (native (view tz-y)) 119) => 0.0
     (buffer (input transform)) => (buffer tz-x)
     (buffer (output transform)) => (buffer tz-y)
     (transform) => tz-y
     (entry (native (view tz-y)) 119) => 119.0)))

(defn test-pull-different [factory]
  (facts
   "Tensor pull connector with different destination"
   (with-release [tz-x (tensor factory [2 3 4 5] :float :nchw)
                  tz-y-desc (desc [2 3 4 5] :float :nhwc)
                  connection (connector tz-x tz-y-desc)]
     (entry (transfer! (range) (view tz-x)) 119) => 119.0
     (identical? (buffer (input connection)) (buffer tz-x)) => true
     (identical? (buffer (input connection)) (buffer (output connection))) => false
     (entry (view (connection)) 119) => 119.0)))

(defn test-pull-same [factory]
  (facts
   "Tensor pull connector with the same destination"
   (with-release [tz-x (tensor factory [2 3 4 5] :float :nchw)
                  tz-y-desc (desc [2 3 4 5] :float :nchw)
                  connection (connector tz-x tz-y-desc)]
     (entry (transfer! (range) (view tz-x)) 119) => 119.0
     (identical? (buffer (input connection)) (buffer tz-x)) => true
     (identical? (buffer (input connection)) (buffer (output connection))) => true
     (entry (view (connection)) 119) => 119.0)))

(defn test-push-different [factory]
  (facts
   "Tensor push connector with different destination"
   (with-release [tz-y (tensor factory [2 3 4 5] :float :nchw)
                  tz-x-desc (desc [2 3 4 5] :float :nhwc)
                  connection (connector tz-x-desc tz-y)]
     (entry (transfer! (range) (view (input connection))) 119) => 119.0
     (identical? (buffer (output connection)) (buffer tz-y)) => true
     (identical? (buffer (input connection)) (buffer (output connection))) => false
     (entry (view (connection)) 119) => 119.0)))

(defn test-push-same [factory]
  (facts
   "Tensor push connector with the same destination"
   (with-release [tz-y (tensor factory [2 3 4 5] :float :nchw)
                  tz-x-desc (desc [2 3 4 5] :float :nchw)
                  connection (connector tz-x-desc tz-y)]
     (entry (transfer! (range) (view connection)) 119) => 119.0
     (identical? (buffer (output connection)) (buffer tz-y)) => true
     (identical? (buffer (input connection)) (buffer (output connection))) => true
     (entry (view (connection)) 119) => 119.0)))
