;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.diamond.internal.dnnl.rnn-test
  (:require [midje.sweet :refer [facts throws => roughly]]
            [uncomplicate.commons [core :refer [with-release]]]
            [uncomplicate.neanderthal
             [core :refer [transfer! native view-vctr view-ge cols]]
             [real :refer [entry! entry]]
             [native :refer [fv]]
             [random :refer [rand-uniform!]]
             [math :as math]]
            [uncomplicate.diamond
             [tensor :refer [*diamond-factory* tensor connector transformer
                             desc shape input output view-tz batcher]]
             [dnn :refer [network init! train cost train rnn]]
             [dnn-test :refer :all]]
            [uncomplicate.diamond.internal.protocols
             :refer [diff-weights forward backward weights bias]]
            [uncomplicate.diamond.internal.dnnl.factory :refer [dnnl-factory]])
  (:import clojure.lang.ExceptionInfo))

(with-release [fact (dnnl-factory)]
  (test-vanilla-rnn-inference fact)
  (test-vanilla-rnn-inference-no-iter fact)
  (test-vanilla-rnn-training fact))
