;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.diamond.internal.cost
  (:require [uncomplicate.commons.core :refer [with-release]]
            [uncomplicate.neanderthal
             [core :refer [dim axpy!]]
             [real :refer [nrm2 asum]]
             [math :refer [sqr pow sqrt]]
             [vect-math :refer [linear-frac! linear-frac mul! log! log sqrt! sqr! round!]]]
            [uncomplicate.diamond.tensor :refer [shape]]))

(defn quadratic-cost!
  ([a-y]
   (/ (sqr (nrm2 a-y)) (* 2 (dim a-y))))
  ([y a!]
   (let [a-y (axpy! -1.0 y a!)]
     (/ (sqr (nrm2 a-y)) (* 2 (dim a-y))))))

(defn mean-absolute-cost!
  ([a-y]
   (/ (asum a-y) (dim a-y)))
  ([y a!]
   (let [a-y (axpy! -1.0 y a!)]
     (/ (asum a-y) (dim a-y)))))

(defn sigmoid-crossentropy-cost!
  ([^long n y a]
   (with-release [ylna (mul! (log a) y)
                  y-1 (linear-frac 1.0 y -1.0)]
     (/ (asum (axpy! -1.0 ylna (mul! y-1 (log! (linear-frac! -1.0 a 1.0))))) n)))
  ([y a]
   (sigmoid-crossentropy-cost! ((shape a) 0) a y)))

(defn binary-accuracy!
  ([y a!]
   (- 1.0 (/ (asum (axpy! -1.0 y (round! a!))) (dim y)))))
