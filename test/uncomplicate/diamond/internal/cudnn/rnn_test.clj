;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.diamond.internal.cudnn.rnn-test
  (:require [midje.sweet :refer [facts throws => roughly]]
            [uncomplicate.commons [core :refer [with-release]]]
            [uncomplicate.diamond.dnn-test :refer :all]
            [uncomplicate.diamond.internal.protocols :refer [create-workspace *workspace*]]
            [uncomplicate.diamond.internal.cudnn.factory :refer [cudnn-factory]])
  (:import clojure.lang.ExceptionInfo))

(with-release [fact (cudnn-factory)]
  (test-vanilla-rnn-inference fact)
  (test-vanilla-rnn-training fact)
  (test-rnn-inference fact)
  (test-rnn-training-no-iter fact)
  (test-lstm-training-no-iter fact)
  (test-lstm-training-no-iter-adam fact)
  (test-gru-training-no-iter-adam fact)
  (test-abbreviate fact))
