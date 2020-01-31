;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.diamond.dnn-test
  (:require [midje.sweet :refer [facts throws => roughly]]
            [uncomplicate.commons [core :refer [with-release]]]
            [uncomplicate.neanderthal
             [core :refer [entry! entry native transfer! view]]]
            [uncomplicate.diamond
             [dnn :refer :all]
             [tensor :refer :all]
             [dnn-test :refer :all]]))

(defn test-sum [factory]
  (facts
   "Tensor sum test"
   (with-release [tz-x (tensor factory [2 3 4 5] :float :nchw)
                  tz-y (tensor factory [2 3 4 5] :float :nchw)
                  sum-bp (sum factory 2.0 tz-x 3.0 tz-y)
                  sum-xy (sum-bp tz-x tz-y)]
     (entry (native (transfer! (range) (view tz-x))) 119) => 119.0
     (entry (native (transfer! (range 0 10000 10) (view tz-y))) 119) => 1190.0
     (entry (native (view tz-x)) 1) => 1.0
     (entry (native (view tz-y)) 1) => 10.0
     (sum-xy) => tz-y
     (entry (native (view tz-x)) 1) => 1.0
     (entry (native (view tz-y)) 1) => 32.0
     (entry (native (view tz-x)) 119) => 119.0
     (entry (native (view tz-y)) 119) => 3808.0)))
