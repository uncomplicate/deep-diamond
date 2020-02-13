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
             [core :refer [entry! entry native transfer! view vctr]]]
            [uncomplicate.diamond
             [dnn :refer :all]
             [tensor :refer :all]
             [dnn-test :refer :all]]
            [uncomplicate.diamond.internal.protocols
             :refer [diff-bias diff-weights forward backward layers]]))

(defn test-sum [factory]
  (with-release [tz-x (tensor factory [2 3 4 5] :float :nchw)
                 tz-y (tensor factory [2 3 4 5] :float :nchw)
                 sum-bp (sum factory 2.0 tz-x 3.0 tz-y)
                 sum-xy (sum-bp tz-x tz-y)]
    (facts
     "Tensor sum test."
     (entry (native (transfer! (range) (view tz-x))) 119) => 119.0
     (entry (native (transfer! (range 0 10000 10) (view tz-y))) 119) => 1190.0
     (entry (native (view tz-x)) 1) => 1.0
     (entry (native (view tz-y)) 1) => 10.0
     (sum-xy) => tz-y
     (entry (native (view tz-x)) 1) => 1.0
     (entry (native (view tz-y)) 1) => 32.0
     (entry (native (view tz-x)) 119) => 119.0
     (entry (native (view tz-y)) 119) => 3808.0)))

(defn test-activation [fact]
  (with-release [src-tz (tensor fact [1 3 2 1] :float :nchw)
                 dst-tz (tensor fact [1 3 2 1] :float :nchw)
                 activ-bluep (activation fact src-tz :relu)
                 activ-infer (activ-bluep src-tz)
                 activ-train (activ-bluep src-tz dst-tz)]

    (transfer! [-0.5 0 0.2 1 0.3 -0.7] src-tz)

    (facts
     "Activation inference test."
     (view (activ-infer)) => (vctr src-tz [0 0 0.2 1.0 0.3 0])
     (view (input activ-infer)) => (vctr src-tz [0 0 0.2 1.0 0.3 0])
     (view (output activ-infer)) => (vctr src-tz [0 0 0.2 1.0 0.3 0]))

    (transfer! [-0.5 0 0.2 1 0.3 -0.7] src-tz)

    (facts
     "Activation forward test."
     (forward activ-train)
     (view (input activ-train)) => (vctr src-tz [-0.5 0 0.2 1 0.3 -0.7])
     (view (output activ-train)) => (vctr src-tz [0 0 0.2 1.0 0.3 0]))

    (facts
     "Activation backward test."
     (transfer! [-0.1 0.1 1 2 7 -0.6] dst-tz)
     (backward activ-train)
     (view (output activ-train)) => (vctr src-tz [-0.1 0.1 1 2 7 -0.6])
     (view (input activ-train)) => (vctr src-tz [0 0 1 2 7.0 0]))))
