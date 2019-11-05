(ns uncomplicate.diamond.internal.dnnl.protocols)

(defprotocol PointerCreator
  (pointer [this]))

(defprotocol Wrapper
  (extract [this]))

(defprotocol Wrappable
  (wrap [this]))

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

(defprotocol PrimitiveDescCreator
  (primitive-desc* [this eng] [this eng hint-pd] [this attr eng hint-pd]))
