;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.diamond.neanderthal-test
  (:require [midje.sweet :refer [facts throws =>]]
            [uncomplicate.commons.core :refer [with-release]]
            [uncomplicate.neanderthal
             [core :refer [asum trans]]
             [native :refer [factory-by-type]]
             [block :refer [column?]]]
            [uncomplicate.diamond.tensor :refer :all]
            [uncomplicate.diamond.internal.neanderthal.factory :refer [neanderthal-factory]])
  (:import clojure.lang.ExceptionInfo))

(defn test-vector [fact dtype]
  (with-release [tz-vect (tensor fact [2] dtype :a)]
    (facts "Basic Tensor Descriptor tests for vectors."
           (shape tz-vect) => [2]
           (layout tz-vect) => [1])))

(defn test-matrix [fact dtype]
  (with-release [tz-ge (tensor fact [20 3])
                 tz-trans (trans tz-ge)
                 tz-ge-stride (tensor fact [20 3] dtype [1 20])]
    (facts "Basic Tensor Descriptor tests for GE matrices."
           (column? tz-ge) => false
           (shape tz-ge) => [20 3]
           (layout tz-ge) => [3 1]
           (shape tz-trans) => [3 20]
           (layout tz-trans) => [1 3]
           (shape tz-ge-stride) => [20 3]
           (layout tz-ge-stride) => [1 20])))

(defn test-all [fact]
  (doseq [dtype [:float :double :long :int :short :byte]]
    (test-vector fact dtype)
    (test-matrix fact dtype)))

(test-all (neanderthal-factory))
