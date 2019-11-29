;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.diamond.internal.dnnl.factory
  (:require [uncomplicate.commons.core :refer [Releaseable release let-release]]
            [uncomplicate.neanderthal.internal.api :refer [FlowProvider flow]]
            [uncomplicate.diamond.tensor :refer [*diamond-factory* view-tz]]
            [uncomplicate.diamond.internal.protocols
             :refer [TensorFactory FactoryProvider ContextProvider]]
            [uncomplicate.diamond.internal.dnnl
             [protocols :refer [desc]]
             [core :refer [memory-desc engine stream memory dims]]
             [tensor :refer [dnnl-tensor dnnl-transformer dnnl-shuffler]]]))

(defrecord DnnlFactory [eng strm master]
  Releaseable
  (release [_]
    (when master
      (release strm)
      (release eng))
    true)
  FactoryProvider
  (factory [this]
    this)
  FlowProvider
  (flow [_]
    strm)
  ContextProvider
  (context [_]
    eng)
  TensorFactory
  (create-tensor-desc [this dims dtype format]
    (memory-desc dims dtype format))
  (create-tensor-desc [this tz-desc]
    (desc tz-desc))
  (create-tensor [this tensor-desc]
    (dnnl-tensor this tensor-desc))
  (create-transformer [_ in-tz out-tz]
    (dnnl-transformer eng strm (view-tz in-tz) (view-tz out-tz)))
  (create-shuffler [_ src-tz dst-tz]
    (dnnl-shuffler eng strm (view-tz src-tz) (view-tz dst-tz))))

(defn dnnl-factory
  ([eng strm]
   (->DnnlFactory eng strm false))
  ([]
   (let-release [eng (engine)]
     (->DnnlFactory eng (stream eng) true))))

(alter-var-root #'uncomplicate.diamond.tensor/*diamond-factory*
                (constantly (dnnl-factory)))
