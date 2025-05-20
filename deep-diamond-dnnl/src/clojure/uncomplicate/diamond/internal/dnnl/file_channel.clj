;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.diamond.internal.dnnl.file-channel
  (:require [uncomplicate.commons
             [core :refer [let-release with-release bytesize]]
             [utils :refer [dragan-says-ex mapped-buffer]]]
            [uncomplicate.clojure-cpp :refer [type-pointer]]
            [uncomplicate.neanderthal.core :refer [transfer!]]
            [uncomplicate.diamond.tensor :refer [data-type]]
            [uncomplicate.diamond.internal
             [protocols :refer [diamond-factory native-diamond-factory parameters create-mapped-tensor]]
             [network :refer []]])
  (:import java.nio.channels.FileChannel
           [uncomplicate.diamond.internal.network SequentialNetworkInference
            SequentialNetworkTraining]))

(defn map-channel
  "Maps a new tensor to a channel via `commons/utils/mapped-buffer`. Please note that the mapping
  remains active and uses resources until the tensor is released *and all references to it removed*,
  because the mapping is only managed by Java's GC, and there is no way of directly affecting that mapping."
  ([fact channel td flag offset-bytes n-index]
   (create-mapped-tensor (diamond-factory fact) channel td flag offset-bytes n-index))
  ([fact channel td flag offset-bytes]
   (map-channel fact channel td flag offset-bytes 0))
  ([fact channel td n-index]
   (map-channel fact channel td :read-write n-index))
  ([fact channel td]
   (map-channel fact channel td :read-write 0)))

(defn ^:private transfer-network! [net channel option]
  (reduce (fn [pos layer]
            (reduce (fn [^long pos param]
                      (with-release [mapped-param (map-channel (native-diamond-factory param)
                                                               channel param option pos)]
                        (case option
                          :read-write (transfer! param mapped-param)
                          :read (transfer! mapped-param param)
                          (dragan-says-ex "You can only :read or :read-write a channel!"))
                        (+ pos (bytesize mapped-param)))) ;;TODO since i removed desc from requires, Tensors will now have to provide the bytesize!
                    pos (parameters layer)))
          0 net)
  channel)

(defmethod transfer! [SequentialNetworkInference FileChannel]
  [net channel]
  (transfer-network! net channel :read-write)
  channel)

(defmethod transfer! [SequentialNetworkTraining FileChannel]
  [net channel]
  (transfer-network! net channel :read-write)
  channel)

(defmethod transfer! [FileChannel SequentialNetworkInference]
  [channel net]
  (transfer-network! net channel :read)
  net)

(defmethod transfer! [FileChannel SequentialNetworkTraining]
  [channel net]
  (transfer-network! net channel :read)
  net)
