;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns hello-world.cuda
  (:require [uncomplicate.commons.core :refer [with-release]]
            [uncomplicate.neanderthal.core :refer [transfer! asum]]
            [uncomplicate.diamond.tensor :refer [tensor with-diamond]]
            [uncomplicate.diamond.internal.cudnn.factory :refer [cudnn-factory]]))

(with-release [cudnn (cudnn-factory)
               t (tensor cudnn [2 3] :float :nc)]
  (transfer! (range) t)
  (asum t))

;; If you see something like this:
;; 15.0
;; It means that everything is set up and you can enjoy programming with Deep Diamond :)

;; Alternatively, you can also use:

(with-diamond cudnn-factory []
  (with-release [t (tensor [2 3] :float :nc)]
    (transfer! (range) t)
    (asum t)))
