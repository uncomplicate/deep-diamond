;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.diamond.internal.cudnn.constants-test
  (:require [midje.sweet :refer [facts =>]]
            [uncomplicate.diamond.internal.cudnn.constants :refer :all]))

(facts "cuDNN data-type tests."
       (remove identity (map #(= % (cudnn-data-type (dec-data-type %))) (range 8))) => [])

(facts "cuDNN activation mode tests."
       (remove identity (map #(= % (cudnn-activation-mode (dec-activation-mode %))) (range 6)))
       => [])

(facts "cuDNN convolution forward algorithm tests."
       (remove identity
               (map #(= % (cudnn-convolution-fwd-algo (dec-convolution-fwd-algo %))) (range 9)))
       => [])

(facts "cuDNN convolution backward data algorithm tests."
       (remove identity
               (map #(= % (cudnn-convolution-bwd-data-algo (dec-convolution-bwd-data-algo %)))
                    (range 7)))
       => [])

(facts "cuDNN convolution backward filter algorithm tests."
       (remove identity
               (map #(= % (cudnn-convolution-bwd-filter-algo (dec-convolution-bwd-filter-algo %)))
                    (range 8)))
       => [])
