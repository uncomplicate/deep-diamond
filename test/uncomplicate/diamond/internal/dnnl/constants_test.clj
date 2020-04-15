;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.diamond.internal.dnnl.constants-test
  (:require [midje.sweet :refer [facts =>]]
            [uncomplicate.diamond.internal.dnnl.constants :refer [dnnl-format dec-format]])
  (:import clojure.lang.ExceptionInfo java.nio.ByteBuffer))

(facts "DNNL format tests."
       (count (remove identity (map #(= % (dnnl-format (dec-format %))) (range 23)))) => 0)
