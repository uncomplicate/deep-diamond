;;   Copyright (c) Dragan Djuric. All rights reserved. use and distribution terms for this software are covered by the
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
  (:require [clojure.string :refer [includes? lower-case]]
            [clojure.tools.logging :refer [warn error info]]
            [uncomplicate.commons.utils :refer [dragan-says-ex channel]]
            [uncomplicate.diamond.internal.protocols :refer [map-channel create-tensor-desc]])
  (:import java.nio.channels.FileChannel))

(defn load-class [^String classname]
  (try (.loadClass (clojure.lang.DynamicClassLoader.) classname)
       (catch Exception e
         (info (format "Class %s is not available." classname))
         nil)))

(defmacro load-dnnl []
  `(do (require 'uncomplicate.diamond.internal.dnnl.factory)
       (alter-var-root #'uncomplicate.diamond.tensor/*diamond-factory*
                       (constantly (uncomplicate.diamond.internal.dnnl.factory/dnnl-factory)))
       (info "DNNL (Intel oneDNN) native backend loaded.")))

(defmacro load-bnns []
  `(do (require 'uncomplicate.diamond.internal.bnns.factory)
       (alter-var-root #'uncomplicate.diamond.tensor/*diamond-factory*
                       (constantly (uncomplicate.diamond.internal.bnns.factory/bnns-factory)))
       (info "BNNS (Apple Accelerate BNNS) native backend loaded.")))

(defn find-default-backend []
  (info "Searching for a suitable backend.")
  (cond (load-class "org.bytedeco.dnnl.global.dnnl")
        :dnnl
        (and (includes? (lower-case (System/getProperty "os.name")) "mac")
             (load-class "uncomplicate.javacpp.accelerate.global.bnns"))
        :bnns
        :default nil))

(defmacro load-backend
  ([]
   `(load-backend ~(find-default-backend)))
  ([backend]
   (let [backend# backend]
     (info (format "Loading %s backend. It may take a few seconds. Please stand by." backend#))
     (case backend#
       :bnns (if (load-class "uncomplicate.javacpp.accelerate.global.bnns")
                     `(load-bnns)
                     (do (error "Apple BNNS is not available in your classpath!")
                         (info "If you want to use Accelerate, please ensure deep-diamond-bnns is in your project dependencies.")
                         (dragan-says-ex "Accelerate cannot be loaded!  Please check yor project's dependencies.")))
       :dnnl (if (load-class "org.bytedeco.dnnl.global.dnnl")
               `(load-dnnl)
              (do (error "DNNL is not available in your classpath!")
                  (info "If you want to use DNNL, please ensure deep-diamond-dnnl and org.bytedeco/dnnl are in your project dependencies.")
                  (dragan-says-ex "DNNL be loaded! Please check yor project's dependencies.")))
       nil (error "This project has no native engine available, so nothing was loaded!")
       (dragan-says-ex (format "Unknown native backend \"%s\". Please use one of: :dnnl :bnns." backend#)
                       {:requested backend# :expected [:dnnl :bnns nil]})))))

(load-backend)

(defn map-tensor
  "Creates a new tensor that controls memory block mapped to `file`.

  Arguments:

  - `shape`, `type`, and `format` are standard parameters used by Deep Diamonds constructors from the `tensor` namespace.
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
                desc flag offset-bytes 0))
  ([file desc flag]
   (map-tensor file desc flag 0))
  ([file shape type format flag]
   (map-tensor file shape type format flag 0))
  ([file desc]
   (map-tensor file desc :read-write)))
