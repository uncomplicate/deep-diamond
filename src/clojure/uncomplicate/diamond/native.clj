;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.diamond.native
  "Entry point to Deep Diamond's CPU engine (currently based on Intel's OneDNNL).
  By evaluating this namespace, you're altering [[uncomplicate.diamond.tensor/*diamond-factory*]]
  var root to the engine produced by [[uncomplicate.diamond.internal.dnnl.factory/dnnl-factory]].
  "
  (:require [uncomplicate.commons.utils :refer [channel]]
            [uncomplicate.diamond.internal.protocols :refer [create-tensor-desc]]
            [uncomplicate.diamond.internal.dnnl
             [factory :refer [dnnl-factory]]
             [file-channel :refer [map-channel]]])
  (:import java.nio.channels.FileChannel))

(alter-var-root #'uncomplicate.diamond.tensor/*diamond-factory*
                (constantly (dnnl-factory)))

(defn map-tensor
  "Creates a new tensor that controls memory block mapped to `file`.

  Arguments:
  - `shape`, `type`, and `format` are standard argument used by Deep Diamonds constructors from the `tensor` namespace.
  - `flag`s - one of `:read-write`, `:read` (or `:read-only`), `:private` (or `:copy-on-write`).
  - `offset-bytes` - offset in the file where tensor's memory block begins.

  If you want to see an example of `map-tensor` in action, please see how this function is used in
  [MNIST functional test](https://github.com/uncomplicate/deep-diamond/blob/master/test/uncomplicate/diamond/functional/mnist/mnist_classification_test.clj)
  and if you need to explore the details, please see the implementation of `map channel` in
  [Neanderthal's internals](https://github.com/uncomplicate/neanderthal/blob/master/src/clojure/uncomplicate/neanderthal/internal/cpp/structures.clj).
  "
  ([file shape type format flag offset-bytes]
   (map-tensor file
               (create-tensor-desc uncomplicate.diamond.tensor/*diamond-factory* shape type format)
               flag offset-bytes))
  ([file desc flag offset-bytes]
   (map-channel uncomplicate.diamond.tensor/*diamond-factory*
                (if (instance? FileChannel file) file (channel file))
                desc flag offset-bytes))
  ([file desc flag]
   (map-tensor file desc flag 0))
  ([file shape type format flag]
   (map-tensor file shape type format flag 0))
  ([file desc]
   (map-tensor file desc :read-write)))
