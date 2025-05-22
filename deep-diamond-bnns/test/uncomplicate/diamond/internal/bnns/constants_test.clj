;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.diamond.internal.bnns.constants-test
  (:require [midje.sweet :refer [facts =>]]
            [uncomplicate.diamond.internal.bnns.constants :refer :all]))

(facts "BNNS data-type tests."
       (remove identity (map #(= % (bnns-data-type (dec-data-type %))) (vals bnns-data-type)))
       => [])

(facts "BNNS data-layout tests."
       (remove identity (map #(= % (bnns-data-layouqt (dec-data-layout %))) (vals bnns-data-layout)))
       => [])
