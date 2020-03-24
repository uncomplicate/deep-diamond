;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.diamond.native
  (:require [uncomplicate.commons.utils :refer [channel]]
            [uncomplicate.diamond.internal.protocols :refer [create-tensor-desc]]
            [uncomplicate.diamond.internal.dnnl.factory :refer [dnnl-factory map-channel]])
  (:import java.nio.channels.FileChannel))

(alter-var-root #'uncomplicate.diamond.tensor/*diamond-factory*
                (constantly (dnnl-factory)))

(defn map-file
  ([file shape type format flag offset-bytes]
   (map-file file
             (create-tensor-desc uncomplicate.diamond.tensor/*diamond-factory* shape type format)
             flag offset-bytes))
  ([file desc flag offset-bytes]
   (map-channel uncomplicate.diamond.tensor/*diamond-factory*
                (if (instance? FileChannel file) file (channel file))
                desc flag offset-bytes))
  ([file desc flag]
   (map-file file desc flag 0))
  ([file shape type format flag]
   (map-file file shape type format flag 0))
  ([file desc]
   (map-file file desc :read-write)))
