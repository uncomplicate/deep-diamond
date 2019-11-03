(ns uncomplicate.diamond.internal.dnnl.protocols)

(defprotocol PointerCreator
  (pointer [this]))

(defprotocol Wrapper
  (extract [this]))

(defprotocol Wrappable
  (wrap [this]))
