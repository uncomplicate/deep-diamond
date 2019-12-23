;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.diamond.internal.cudnn.tensor
  (:require [uncomplicate.commons
             [core :refer [Releaseable release let-release with-release Info info]]
             [utils :refer [dragan-says-ex]]]
            [uncomplicate.neanderthal
             [core :refer [transfer! dim]]
             [block :refer [entry-width data-accessor buffer count-entries]]
             [native :refer [factory-by-type]]]
            [uncomplicate.neanderthal.internal
             [api :as neand :refer [Viewable view flow]]
             [printing :refer [print-vector]]]
            [uncomplicate.diamond.tensor
             :refer [TensorDescriptor shape layout TensorContainer Transfer input output
                     Revert ConnectorCreator connector view-tz]]
            [uncomplicate.diamond.internal.protocols
             :refer [TensorFactory FactoryProvider ContextProvider factory context]]
            [uncomplicate.diamond.internal.cudnn
             [core :refer [tensor-descriptor equal-desc?]]
             [protocols :refer [DescProvider desc]]])
  (:import [clojure.lang Seqable IFn]
           uncomplicate.neanderthal.internal.api.Block
           uncomplicate.diamond.tensor.TensorDescriptorImpl
           uncomplicate.diamond.internal.cudnn.impl.CUTensorDescriptor))

(declare ->CUDnnTensor cudnn-transformer cudnn-tensor cudnn-shuffler)

(extend-type java.util.Collection
  DescProvider
  (desc [this]
    (tensor-descriptor this :float :nchw)))

(extend-type java.util.Map
  DescProvider
  (desc [this]
    (tensor-descriptor (:shape this) (or (:data-type this) :float) (or (layout this) :nchw))))

(extend-type TensorDescriptorImpl
  DescProvider
  (desc [this]
    (tensor-descriptor (.shape this) (or (.data-type this) :float) (or (layout this) :float)))
  ConnectorCreator
  (connector [in-desc out]
    (connector (desc in-desc) out)))

(extend-type CUTensorDescriptor
  TensorDescriptor
  (shape [this]
    (.dims this))
  (data-type [this]
    (.data-type this))
  (layout [this]
    (.strides this))
  ConnectorCreator
  (connector [in-desc out]
    (if (equal-desc? in-desc (input out))
      out
      (let [out-tz (output out)]
        (if (equal-desc? in-desc out-tz)
          (view-tz out-tz)
          (let [fact (factory out-tz)]
            (let-release [in-tz (cudnn-tensor fact in-desc)]
              (cudnn-transformer (context fact) (flow fact) in-tz (view-tz out-tz)))))))))

(defmethod print-method CUTensorDescriptor
  [^CUTensorDescriptor d ^java.io.Writer w]
  (.write w (pr-str {:shape (.dims d) :data-type (.data-type d) :layout (.strides d)})))
