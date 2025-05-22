;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.diamond.internal.bnns.impl
  (:require [uncomplicate.commons
             [core :refer [Releaseable release let-release with-release Info
                           info Viewable view Bytes bytesize Entries sizeof* bytesize*
                           sizeof size]]
             [utils :refer [dragan-says-ex]]]
            [uncomplicate.fluokitten.protocols :refer [Comonad extract]]
            [uncomplicate.clojure-cpp
             :refer [null? ptr*]]
            [uncomplicate.diamond.internal.utils :refer [extend-pointer]]
            [uncomplicate.diamond.internal.bnns
             [protocols :refer :all]
             [constants :refer :all]])
  (:import [org.bytedeco.javacpp PointerPointer LongPointer FloatPointer]
           uncomplicate.javacpp.accelerate.global.bnns
           [uncomplicate.javacpp.accelerate.global bnns$BNNSTensor bnns$BNNSNDArrayDescriptor]))

;; ===================== Tensor and NDArray descriptors ===========================================

(defmacro extend-tensor-descriptor [t]
  `(extend-type ~t
     Releaseable
     (release [this#]
       (let [td# (.-td this#)]
         (locking td#
           (when (and (.-master this#) (not (null? td#)))
             (do (.deallocate (ptr* td#))
                 (.setNull (ptr* td#))))))
       true)
     Comonad
     (extract [this#]
       (if-not (null? (.-td this#)) (.-td this#) nil))
     DescProvider
     (desc [this#]
       this#)
     Viewable
     (view [this#]
       (new ~t (.-td this#) (.-dims this#) (.-data-type this#) (.-layout this#) false))
     Info
     (info
       ([this# info-type#]
        (case info-type#
          :class (class this#)
          :device :cpu
          :shape (.-dims this#)
          :data-type (.-data-type this#)
          :layout (.-layout this#)
          nil))
       ([this#]
        {:class (class this#)
         :device :cpu
         :shape (.-dims this#)
         :data-type (.-data-type this#)
         :layout (.-layout this#)}))))

(deftype BNNSTensorDescriptor [^bnns$BNNSTensor td dims data-type layout master]
  Object
  (hashCode [this]
    (hash td))
  (equals [this other]
    (and (instance? BNNSTensorDescriptor other)
         (let [td2 ^BNNSTensorDescriptor other]
           (or (= td (extract other))
               (and (= (.dims this) (.dims td2)) (= (.data-type this) (.data-type td2))
                    (= (.layout this) (.layout td2)))))))
  (toString [this]
    (format "#BNNSTensorDescriptor[0x%s, master: %s]" (address td) master))
  Bytes
  (bytesize* [_]
    (if (null? td)
      nil
      (.data_size_in_bytes td))))

(extend-tensor-descriptor BNNSTensorDescriptor)

(deftype BNNSNDArrayDescriptor [^bnns$BNNSNDArrayDescriptor nda dims data-type layout master]
  Object
  (hashCode [this]
    (hash nda))
  (equals [this other]
    (and (instance? BNNSNdarrayDescriptor other)
         (let [nda2 ^BNNSNdarrayDescriptor other]
           (or (= nda (extract other))
               (and (= (.dims this) (.dims nda2)) (= (.data-type this) (.data-type nda2))
                    (= (.layout this) (.layout nda2)))))))
  (toString [this]
    (format "#BNNSDescriptor[0x%s, master: %s]" (address nda) master))
  Bytes
  (bytesize* [_]
    (if (null? nda)
      nil
      (int "TODO"))))

(extend-tensor-descriptor BNNSNDArrayDescriptor)

(extend-type bnns$BNNSNDArrayDescriptor
  Releaseable
  (release [_]
    (dragan-says-ex "You should never directly release bnns$BNNSNDArrayDescriptor. Please use NdarrayDescImpl!")))

(extend-type bnns$BNNSTensor
  Releaseable
  (release [_]
    (dragan-says-ex "You should never directly release dnnl_memorybnns$BNNSTensor. Please use BnnsTensorImpl!")))

(defn tensor-descriptor*
  ([^bnns$BNNSTensor td shape ^long data-type strides]
   (let [rank (count shape)]
     (.data-type td data-type)
     (.rank td rank)
     (dotimes [i rank]
       (.shape td i (get shape i))
       (.stride td i (get strides i)))
     td)))

(defn ndarray-descriptor*
  ([^bnns$BNNSNDArrayDescriptor nda shape ^long data-type ^long layout ]
   (let [rank (count shape)]
     (.data-type nda data-type)
     (.layout nda layout)
     (dotimes [i rank]
       (.size nda i (get shape i)))
     nda)))
