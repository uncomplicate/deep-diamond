(ns native-test
  (:require [clojure.test :refer [deftest is]]
            [hello-world.native :as native]))

(deftest yeah-test
  (is (= 15.0 (native/yeah))))
