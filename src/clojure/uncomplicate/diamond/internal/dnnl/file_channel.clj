(ns uncomplicate.diamond.internal.dnnl.file-channel
  (:require [uncomplicate.commons
             [core :refer [let-release with-release]]
             [utils :refer [dragan-says-ex mapped-buffer]]]
            [uncomplicate.neanderthal.core :refer [transfer!]]
            [uncomplicate.diamond.internal
             [protocols :refer [diamond-factory native-diamond-factory parameters layers]]
             [network :refer []]]
            [uncomplicate.diamond.internal.dnnl
             [core :refer [size]]
             [protocols :refer [desc]]
             [tensor :refer [dnnl-tensor*]]])
  (:import java.nio.channels.FileChannel
           [uncomplicate.diamond.internal.network SequentialNetworkInference
            SequentialNetworkTraining]))

(defn map-channel
  ([fact channel td flag offset-bytes]
   (let [fact (diamond-factory fact)
         size (size (desc td))]
     (let-release [buf (mapped-buffer channel offset-bytes size flag)]
       (dnnl-tensor* fact td buf true))))
  ([fact channel td flag]
   (map-channel fact channel td flag 0))
  ([fact channel td]
   (map-channel fact channel td :read-write)))

(defn ^:private transfer-network! [net channel option]
  (reduce (fn [pos layer]
            (reduce (fn [^long pos param]
                      (with-release [mapped-param (map-channel (native-diamond-factory param)
                                                               channel param option pos)]
                        (case option
                          :read-write (transfer! param mapped-param)
                          :read (transfer! mapped-param param)
                          (dragan-says-ex "You can only :read or :read-write a channel!"))
                        (+ pos (size mapped-param))))
                    pos (parameters layer)))
          0 (layers net))
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
