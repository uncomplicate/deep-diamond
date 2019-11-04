(ns uncomplicate.diamond.internal.dnnl.constants
  (:import org.bytedeco.dnnl.global.dnnl))

(defn dec-status [^long status]
  (case status
    0 :success
    1 :out-of-memory
    2 :invalid-arguments
    3 :unimplemented
    4 :iterator-ends
    5 :runtime-error
    6 :not-required
    :unknown))

(def ^:const dnnl-engine-kind
  {:cpu dnnl/dnnl_cpu
   :gpu dnnl/dnnl_gpu
   :any dnnl/dnnl_any_engine})

(defn dec-engine-kind [^long kind]
  (case kind
    1 :cpu
    2 :gpu
    0 :any
    :unknown))

(def ^:const dnnl-stream-flags
  {:default-order dnnl/dnnl_stream_default_flags
   :in-order dnnl/dnnl_stream_in_order
   :out-of-order dnnl/dnnl_stream_out_of_order
   :default-flags dnnl/dnnl_stream_default_flags})
