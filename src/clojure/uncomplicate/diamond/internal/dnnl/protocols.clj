;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.diamond.internal.dnnl.protocols)

(defprotocol PointerCreator
  (pointer [this]))

(defprotocol DnnlCloneable
  (clone [this]))

(defprotocol BlockedDesc
  (memory-desc* [this dims data-type]))

(defprotocol Memory
  (data [this])
  (ptr [this]))

(defprotocol DescProvider
  (desc [this]))

(defprotocol PrimitiveKind
  (primitive-kind* [this]))

(defprotocol DnnlEngineProvider
  (dnnl-engine [_]))
