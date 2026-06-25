;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns hello-world.native
  (:require [uncomplicate.neanderthal.core :refer [transfer! asum]]
            [uncomplicate.diamond
             [tensor :refer [tensor]]
             [native :refer []]]))

(defn yeah []
  (let [t (tensor [2 3] :float :nc)]
    (transfer! (range) t)
  ;; ... and compute an absolute sum of its entries
    (asum t)))
