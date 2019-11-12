(ns uncomplicate.diamond.internal.dnnl.dnnl-test
  (:require [midje.sweet :refer [facts throws => roughly]]
            [uncomplicate.commons
             [core :refer [with-release]]
             [utils :refer [capacity direct-buffer put-float get-float]]]
            [uncomplicate.diamond.internal.dnnl :refer :all]
            [uncomplicate.diamond.internal.dnnl.protocols :as api])
  (:import clojure.lang.ExceptionInfo java.nio.ByteBuffer))

(facts "Engine count tests."
       (pos? (engine-count)) => true
       (pos? (engine-count :cpu)) => true
       (engine-count :gpu) => 0
       (engine-count :any) => 0
       (engine-count :something) => (throws ExceptionInfo))

(facts "Engine kind tests."
       (with-release [eng (engine)]
         (engine-kind eng) => :cpu
         (engine 10 :gpu) => (throws ExceptionInfo)
         (engine 0 :gpu) => (throws ExceptionInfo)
         (engine 0 :any) => (throws ExceptionInfo)))

(facts "Memory descriptor by strides."
       (with-release [strds [120 3 4 5]
                      dimensions [2 3 4 5]
                      md (memory-desc dimensions :float strds)]
         (data-type md) => :float
         (ndims md) => (count dimensions)
         (dims md) => dimensions
         (strides md) => strds
         (size md) => (* (long (first strds)) (long (first dimensions)) Float/BYTES)
         (memory-desc [1 1] :f64 [1 1]) => (throws ExceptionInfo)
         (data-type (memory-desc [1 1])) => :float
         (strides (memory-desc [2 3])) => [0 0]
         (strides (memory-desc [2 3] :float :nc)) => [3 1]))

(facts "Memory descriptor by tag."
       (with-release [md (memory-desc [2 3 4 5] :float :nchw)]
         (data-type md) => :float
         (ndims md) => 4
         (dims md) => [2 3 4 5]
         (strides md) => [60 20 5 1]
         (memory-desc [1 1] :f64 :nchw) => (throws ExceptionInfo)
         (memory-desc [1 1] :s8 :xxx) => (throws ExceptionInfo)))

(facts "Basic memory integration."
       (with-release [eng (engine)
                      md (memory-desc [2 3 4 5] :float :nchw)
                      mem (memory eng md)]
         (instance? ByteBuffer (api/data mem)) => true
         (capacity (api/data mem)) => 480))

(facts "Memory offset operation."
       (with-release [eng (engine)
                      s (stream eng)
                      md (memory-desc [2 3 4 5] :float :nchw)
                      buf (direct-buffer (+ 8 (size md)))
                      mem (memory eng md buf)
                      relu-desc (eltwise-fwd-desc :inference :relu md)
                      relu-pd (primitive-desc eng relu-desc)
                      relu (primitive relu-pd)
                      relu-args (eltwise-args mem)]
         (primitive-kind relu-desc) => :eltwise
         (put-float buf 0 -100)
         (put-float buf 1 -20)
         (put-float buf 2 -200)
         (put-float buf 120 -400)
         (put-float buf 121 -500)
         (offset! mem 4)
         (execute! s relu relu-args) => s
         (get-float buf 0) => -100.0
         (get-float buf 1) => 0.0
         (get-float buf 2) => 0.0
         (get-float buf 120) => 0.0
         (get-float buf 121) => -500.0
         (offset! mem 489) => (throws ExceptionInfo)))

(facts "Submemory descriptor."
       (with-release [eng (engine)
                      s (stream eng)
                      md (memory-desc [2 3 4 5] :float :nchw)
                      buf (direct-buffer (size md))
                      sub-md (submemory-desc md 1)
                      sub-mem (memory eng sub-md buf)
                      relu-desc (eltwise-fwd-desc :inference :relu sub-md)
                      relu-pd (primitive-desc eng relu-desc)
                      relu (primitive relu-pd)
                      relu-args (eltwise-args sub-mem)]
         (primitive-kind relu-desc) => :eltwise
         (put-float buf 0 -100)
         (put-float buf 1 -20)
         (put-float buf 2 -200)
         (put-float buf 59 -1)
         (put-float buf 60 -2)
         (put-float buf 119 -400)
         (offset! sub-mem (* Float/BYTES 60))
         (execute! s relu relu-args) => s
         (get-float buf 0) => -100.0
         (get-float buf 1) => -20.0
         (get-float buf 2) => -200.0
         (get-float buf 59) => -1.0
         (get-float buf 60) => 0.0
         (get-float buf 119) => 0.0
         (offset! sub-mem (* Float/BYTES 1))
         (execute! s relu relu-args) => s
         (get-float buf 0) => -100.0
         (get-float buf 1) => 0.0
         (get-float buf 2) => 0.0
         (get-float buf 59) => 0.0
         (get-float buf 60) => 0.0
         (get-float buf 119) => 0.0))
