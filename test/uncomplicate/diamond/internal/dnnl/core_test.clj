;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.diamond.internal.dnnl.core-test
  (:require [midje.sweet :refer [facts throws => roughly truthy just]]
            [uncomplicate.commons
             [core :refer [with-release]]
             [utils :refer [capacity direct-buffer put-float get-float]]]
            [uncomplicate.neanderthal
             [core :refer [zero nrm2 entry! entry transfer!]]
             [native :refer [fv]]
             [block :refer [buffer]]
             [math :refer [sqr sqrt]]]
            [uncomplicate.diamond.internal.dnnl
             [core :refer :all]
             [protocols :as api]])
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
       (with-release [strds [120 1 20 4]
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
                      relu-args (fwd-args mem)]
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
                      relu-args (fwd-args sub-mem)]
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

(facts "Submemory descriptor with offsets."
       (with-release [eng (engine)
                      s (stream eng)
                      md (memory-desc [2 2 3 2] :float :nchw)
                      buf (direct-buffer (size md))
                      sub-md (submemory-desc md [2 1 3 2] [0 1 0 0])
                      sub-mem (memory eng sub-md buf)
                      relu-desc (eltwise-fwd-desc :inference :abs sub-md)
                      relu-pd (primitive-desc eng relu-desc)
                      relu (primitive relu-pd)
                      relu-args (fwd-args sub-mem)]
         (dotimes [i 24]
           (put-float buf i (- i)))
         (execute! s relu relu-args) => s
         (get-float buf 5) => -5.0
         (get-float buf 6) => 6.0
         (get-float buf 11) => 11.0
         (get-float buf 12) => -12.0
         (get-float buf 23) => 23.0))

(facts "In-place Sum operation"
       (with-release [eng (engine)
                      s (stream eng)
                      md (memory-desc [2 3 4 5] :float :nchw)
                      buf (direct-buffer (size md))
                      src (memory eng md buf)
                      sum-pd (sum! eng 2.0 md)
                      sum-prim (primitive sum-pd)
                      sum-args (multi-args src)]
         (put-float buf 0 -100)
         (put-float buf 1 20)
         (execute! s sum-prim sum-args) => s
         (get-float buf 0) => -200.0
         (get-float buf 1) => 40.0))

(facts "In-place Sum operation with two sources"
       (with-release [eng (engine)
                      s (stream eng)
                      md (memory-desc [2 3 4 5] :float :nchw)
                      buf-src (direct-buffer (size md))
                      src (memory eng md buf-src)
                      buf-dst (direct-buffer (size md))
                      dst (memory eng md buf-dst)
                      sum-pd (sum! eng md 2.0 md 3.0 md)
                      sum-prim (primitive sum-pd)
                      sum-args (multi-args dst src dst)]
         (put-float buf-src 0 -100)
         (put-float buf-src 1 10)
         (put-float buf-dst 0 -200)
         (put-float buf-dst 1 20)
         (execute! s sum-prim sum-args) => s
         (get-float buf-src 0) => -100.0
         (get-float buf-src 1) => 10.0
         (get-float buf-dst 0) => -800.0
         (get-float buf-dst 1) => 80.0))

(facts "Out of place Sum operation"
       (with-release [eng (engine)
                      s (stream eng)
                      md (memory-desc [2 3 4 5] :float :nchw)
                      src0-buf (direct-buffer (size md))
                      src1-buf (direct-buffer (size md))
                      dst-buf (direct-buffer (size md))
                      src0 (memory eng md src0-buf)
                      src1 (memory eng md src1-buf)
                      dst (memory eng md dst-buf)
                      sum-pd (sum! eng md 2.0 md 3.0 md)
                      sum-prim (primitive sum-pd)
                      sum-args (multi-args dst src0 src1)]
         (put-float src0-buf 0 -100)
         (put-float src0-buf 1 20)
         (put-float src1-buf 0 -1000)
         (put-float src1-buf 1 200)
         (execute! s sum-prim sum-args) => s
         (get-float dst-buf 0) => -3200.0
         (get-float dst-buf 1) => 640.0))

(facts "Reordering memory."
       (let [dims [2 2 3 2]]
         (with-release [eng (engine)
                        s (stream eng)
                        a-desc (memory-desc dims :float :nchw)
                        b-desc (memory-desc dims :float :nchw)
                        c-desc (memory-desc dims :float :nhwc)
                        reorder-a-c-pd (reorder eng a-desc c-desc)
                        reorder-a-b-pd (reorder eng a-desc b-desc)
                        a-vec (fv (range (apply * dims)))
                        b-vec (zero a-vec)
                        c-vec (zero a-vec)
                        a-mem (memory eng a-desc (buffer a-vec))
                        b-mem (memory eng a-desc (buffer b-vec))
                        c-mem (memory eng a-desc (buffer c-vec))
                        reorder-a-c (primitive reorder-a-c-pd)
                        reorder-a-b (primitive reorder-a-b-pd)
                        reorder-a-c-args (fwd-args a-mem c-mem)
                        reorder-a-b-args (fwd-args a-mem b-mem)]
           (equal-desc? a-desc a-desc) => true
           (equal-desc? a-desc b-desc) => true
           (equal-desc? a-desc c-desc) => false
           (= a-vec c-vec) => false
           (= (nrm2 a-vec) (nrm2 c-vec)) => false
           (execute! s reorder-a-c reorder-a-c-args) => s
           (= a-vec c-vec) => false
           (= (nrm2 a-vec) (nrm2 c-vec)) => true
           (= a-vec b-vec) => false
           (= (nrm2 a-vec) (nrm2 b-vec)) => false
           (execute! s reorder-a-b reorder-a-b-args) => s
           (= a-vec b-vec) => true)))

(facts "Elementwise forward ReLU operation."
       (with-release [eng (engine)
                      s (stream eng)
                      md (memory-desc [2 3 4 5] :float :nchw)
                      buf (direct-buffer (size md))
                      mem (memory eng md buf)
                      relu-desc (eltwise-fwd-desc :inference :relu md)
                      relu-pd (primitive-desc eng relu-desc)
                      relu (primitive relu-pd)
                      relu-args (fwd-args mem)]
         (primitive-kind relu-desc) => :eltwise
         (put-float buf 0 -100)
         (put-float buf 1 20)
         (execute! s relu relu-args) => s
         (get-float buf 0) => 0.0
         (get-float buf 1) => 20.0))

(facts "Elementwise backward ReLU operation."
       (with-release [eng (engine)
                      s (stream eng)
                      md (memory-desc [2 3] :float :nc)
                      buf (direct-buffer (size md))
                      mem (memory eng md buf)
                      relu-desc (eltwise-fwd-desc :training :relu md)
                      relu-pd (primitive-desc eng relu-desc)
                      relu (primitive relu-pd)
                      relu-args (fwd-args mem)
                      diff-dst-vec (fv (range 2 8))
                      diff-dst-desc (memory-desc [2 3] :float :nc)
                      relu-bwd-desc (eltwise-bwd-desc :relu diff-dst-desc md)
                      relu-bwd-pd (primitive-desc eng relu-bwd-desc relu-pd)
                      diff-dst-mem (memory eng (diff-dst-md relu-bwd-pd) (buffer diff-dst-vec))
                      relu-bwd (primitive relu-bwd-pd)
                      relu-bwd-args (eltwise-bwd-args mem diff-dst-mem diff-dst-mem)]
         (primitive-kind relu-desc) => :eltwise
         (put-float buf 0 -100)
         (put-float buf 1 20)
         (execute! s relu relu-args) => s
         (get-float buf 0) => 0.0
         (get-float buf 1) => 20.0
         diff-dst-vec => (fv 2 3 4 5 6 7)
         (execute! s relu-bwd relu-bwd-args) => s
         diff-dst-vec => (fv 0 3 0 0 0 0)))

(facts "Elementwise backward Logistic operation."
       (with-release [eng (engine)
                      s (stream eng)
                      src-vec (fv [-1 -0.5 0 0.1 0.5 0.7])
                      src-md (memory-desc [2 3] :float :nc)
                      src-mem (memory eng src-md (buffer src-vec))
                      dst-vec (fv 6)
                      dst-mem (memory eng src-md (buffer dst-vec))
                      logistic-desc (eltwise-fwd-desc :training :logistic src-md)
                      logistic-pd (primitive-desc eng logistic-desc)
                      logistic (primitive logistic-pd)
                      logistic-args (fwd-args src-mem dst-mem)
                      diff-dst-vec (fv [-0.5 -0.2 -0.4 0 0.2 0.3])
                      diff-dst-desc (memory-desc [2 3] :float :nc)
                      diff-src-vec (fv 6)
                      logistic-bwd-desc (eltwise-bwd-desc :logistic diff-dst-desc src-md)
                      logistic-bwd-pd (primitive-desc eng logistic-bwd-desc logistic-pd)
                      diff-dst-mem (memory eng (diff-dst-md logistic-bwd-pd) (buffer diff-dst-vec))
                      diff-src-mem (memory eng (diff-src-md logistic-bwd-pd) (buffer diff-src-vec))
                      logistic-bwd (primitive logistic-bwd-pd)
                      logistic-bwd-args (eltwise-bwd-args src-mem diff-dst-mem diff-src-mem)]
         (primitive-kind logistic-desc) => :eltwise
         (execute! s logistic logistic-args)
         (execute! s logistic-bwd logistic-bwd-args)
         (seq dst-vec) => [0.2689414322376251 0.377540647983551 0.5 0.5249792337417603
                           0.622459352016449 0.6681877970695496];; this is aL - y
         diff-src-vec => (fv [-0.09830597043037415 -0.04700074344873428 -0.10000000149011612
                              0.0 0.04700074344873428 0.06651385873556137])));; this is deltaL

(facts "Inner product forward."
       (with-release [eng (engine)
                      s (stream eng)
                      src-desc (memory-desc [1 1 3 3] :float :nchw)
                      weights-desc (memory-desc [2 1 3 3] :float :any)
                      bias-desc (memory-desc [2] :float :x)
                      dst-desc (memory-desc [1 2] :float :nc)
                      ip-desc (inner-product-fwd-desc :inference src-desc weights-desc bias-desc dst-desc)
                      ip-pd (primitive-desc eng ip-desc)
                      src-vec (fv (take 9 (range 1 2 0.1)))
                      src-mem (memory eng (src-md ip-pd) (buffer src-vec))
                      weights-vec (fv (take 18 (range 0 1 0.02)))
                      weights-mem (memory eng (weights-md ip-pd) (buffer weights-vec))
                      bias-vec (fv [0.3 0.7])
                      bias-mem (memory eng bias-desc (buffer bias-vec))
                      dst-vec (fv 2)
                      dst-mem (memory eng (dst-md ip-pd) (buffer dst-vec))
                      ip (primitive ip-pd)
                      ip-args (fwd-args src-mem weights-mem bias-mem dst-mem)]
         (primitive-kind ip-desc) => :inner-product
         (execute! s ip ip-args) => s
         dst-vec => (fv [1.428 4.095999717712402])))

(facts "Inner product backward."
       (with-release [eng (engine)
                      s (stream eng)
                      src-desc (memory-desc [2 2] :float :nc)
                      weights-desc (memory-desc [3 2] :float :io)
                      bias-desc (memory-desc [3] :float :x)
                      dst-desc (memory-desc [2 3] :float :nc)
                      ip-desc (inner-product-fwd-desc :training src-desc weights-desc bias-desc dst-desc)
                      ip-pd (primitive-desc eng ip-desc)
                      src-vec (fv [2 3 1 1])
                      src-mem (memory eng (src-md ip-pd) (buffer src-vec))
                      weights-vec (fv [0.1 0.2 0.3 0.4 0.5 0.6])
                      weights-mem (memory eng (weights-md ip-pd) (buffer weights-vec))
                      bias-vec (fv [1 2 3])
                      bias-mem (memory eng bias-desc (buffer bias-vec))
                      dst-vec (fv 6)
                      dst-mem (memory eng (dst-md ip-pd) (buffer dst-vec))
                      ip (primitive ip-pd)
                      ip-args (fwd-args src-mem weights-mem bias-mem dst-mem)
                      diff-src-desc (memory-desc [2 2] :float :nc)
                      diff-dst-desc (memory-desc [2 3] :float :nc)
                      ip-bwd-data-desc (inner-product-bwd-desc diff-src-desc weights-desc diff-dst-desc)
                      ip-bwd-data-pd (primitive-desc eng ip-bwd-data-desc ip-pd)
                      diff-src-vec (fv 4)
                      diff-dst-vec (fv [0.2 0.3 0.8 1 1 1])
                      diff-src-mem (memory eng (diff-src-md ip-bwd-data-pd) (buffer diff-src-vec))
                      diff-dst-mem (memory eng (diff-dst-md ip-bwd-data-pd) (buffer diff-dst-vec))
                      ip-bwd-data (primitive ip-bwd-data-pd)
                      ip-bwd-data-args (bwd-args diff-dst-mem weights-mem diff-src-mem)
                      diff-weights-desc (memory-desc [3 2] :float :nc)
                      diff-bias-desc (memory-desc [3] :float :x)
                      ip-bwd-weights-desc (inner-product-bwd-desc src-desc diff-weights-desc
                                                                  diff-bias-desc diff-dst-desc)
                      ip-bwd-weights-pd (primitive-desc eng ip-bwd-weights-desc ip-pd)
                      diff-weights-vec (fv [1 0 0 0 0 0])
                      diff-bias-vec (fv [5 5 5])
                      diff-weights-mem (memory eng (diff-weights-md ip-bwd-weights-pd)
                                               (buffer diff-weights-vec))
                      diff-bias-mem (memory eng diff-bias-desc (buffer diff-bias-vec))
                      ip-bwd-weights (primitive ip-bwd-weights-pd)
                      ip-bwd-weights-args (bwd-args src-mem diff-dst-mem
                                                    diff-weights-mem diff-bias-mem)]
         (primitive-kind ip-desc) => :inner-product
         (execute! s ip ip-args) => s
         dst-vec => (fv 2.4 3.9 5.4 1.5 2.7 3.9)
         diff-src-vec => (fv 4)
         (execute! s ip-bwd-data ip-bwd-data-args) => s
         diff-src-vec => (fv 0.320000022649765 0.7100000381469727 0.6000000238418579 1.5)
         (execute! s ip-bwd-weights ip-bwd-weights-args) => s
         diff-weights-vec => (fv 1.4 1.6 1.6 1.9000000953674316 2.6 3.4)
         diff-bias-vec => (fv 1.2 1.3 1.8))

       (with-release [eng (engine)
                      s (stream eng)
                      src-desc (memory-desc [1 1] :float :nc)
                      weights-desc (memory-desc [1 1] :float :any)
                      bias-desc (memory-desc [1] :float :x)
                      dst-desc (memory-desc [1 1] :float :nc)
                      ip-desc (inner-product-fwd-desc :training src-desc weights-desc bias-desc dst-desc)
                      ip-pd (primitive-desc eng ip-desc)
                      src-vec (fv [-0.5])
                      src-mem (memory eng (src-md ip-pd) (buffer src-vec))
                      weights-vec (fv [-0.1])
                      weights-mem (memory eng (weights-md ip-pd) (buffer weights-vec))
                      bias-vec (fv [0.2])
                      bias-mem (memory eng bias-desc (buffer bias-vec))
                      dst-vec (fv 1)
                      dst-mem (memory eng (dst-md ip-pd) (buffer dst-vec))
                      ip (primitive ip-pd)
                      ip-args (fwd-args src-mem weights-mem bias-mem dst-mem)
                      diff-src-desc (memory-desc [1 1] :float :nc)
                      diff-dst-desc (memory-desc [1 1] :float :nc)
                      ip-bwd-data-desc (inner-product-bwd-desc diff-src-desc weights-desc diff-dst-desc)
                      ip-bwd-data-pd (primitive-desc eng ip-bwd-data-desc ip-pd)
                      diff-src-vec (fv 1)
                      diff-dst-vec (fv [0.4])
                      diff-src-mem (memory eng (diff-src-md ip-bwd-data-pd) (buffer diff-src-vec))
                      diff-dst-mem (memory eng (diff-dst-md ip-bwd-data-pd) (buffer diff-dst-vec))
                      ip-bwd-data (primitive ip-bwd-data-pd)
                      ip-bwd-data-args (bwd-args diff-dst-mem weights-mem diff-src-mem)
                      diff-weights-desc (memory-desc [1 1] :float :nc)
                      diff-bias-desc (memory-desc [1] :float :x)
                      ip-bwd-weights-desc (inner-product-bwd-desc src-desc diff-weights-desc
                                                                  diff-bias-desc diff-dst-desc)
                      ip-bwd-weights-pd (primitive-desc eng ip-bwd-weights-desc ip-pd)
                      diff-weights-vec (fv [1000])
                      diff-bias-vec (fv [5000])
                      diff-weights-mem (memory eng (diff-weights-md ip-bwd-weights-pd)
                                               (buffer diff-weights-vec))
                      diff-bias-mem (memory eng diff-bias-desc (buffer diff-bias-vec))
                      ip-bwd-weights (primitive ip-bwd-weights-pd)
                      ip-bwd-weights-args (bwd-args src-mem diff-dst-mem
                                                    diff-weights-mem diff-bias-mem)]
         (primitive-kind ip-desc) => :inner-product
         (execute! s ip ip-args) => s
         dst-vec => (fv 0.25)
         (execute! s ip-bwd-data ip-bwd-data-args) => s
         diff-src-vec => (fv -0.04000000283122063)
         (execute! s ip-bwd-weights ip-bwd-weights-args) => s
         diff-weights-vec => (fv -0.2)
         ;;Note that diff-bias is equal to diff-dst
         diff-bias-vec => (fv 0.4)))

(facts "Softmax forward operation"
       (with-release [eng (engine)
                      s (stream eng)
                      md (memory-desc [2 3] :float :nc)
                      buf (direct-buffer (size md))
                      mem (memory eng md buf)
                      axis 1
                      softmax-desc (softmax-fwd-desc :inference md axis)
                      softmax-pd (primitive-desc eng softmax-desc)
                      softmax (primitive softmax-pd)
                      softmax-args (fwd-args mem)]
         (primitive-kind softmax-desc) => :softmax
         (put-float buf 0 1)
         (put-float buf 1 3)
         (put-float buf 2 3)
         (put-float buf 3 2)
         (put-float buf 4 4)
         (put-float buf 5 8)
         (execute! s softmax softmax-args) => s
         (get-float buf 0) => 0.06337893754243851
         (get-float buf 1) => 0.46831050515174866
         (get-float buf 2) => 0.46831050515174866
         (get-float buf 3) => 0.0024282580707222223
         (get-float buf 4) => 0.017942532896995544
         (get-float buf 5) => 0.9796292185783386))

(facts "Softmax backward operation."
       (with-release [eng (engine)
                      s (stream eng)
                      md (memory-desc [2 3] :float :nc)
                      buf (direct-buffer (size md))
                      mem-vec (fv 1 3 3 2 4 8)
                      mem (memory eng md (buffer mem-vec))
                      axis 1
                      softmax-desc (softmax-fwd-desc :training md axis)
                      softmax-pd (primitive-desc eng softmax-desc)
                      softmax (primitive softmax-pd)
                      softmax-args (fwd-args mem)
                      diff-dst-vec (fv 0 -2.135335400336505 0
                                       0 0 -1.0207943791746268) ;; -ti/aLi
                      diff-dst-desc (memory-desc [2 3] :float :nc)
                      softmax-bwd-desc (softmax-bwd-desc diff-dst-desc md axis)
                      softmax-bwd-pd (primitive-desc eng softmax-bwd-desc softmax-pd)
                      diff-dst-mem (memory eng (diff-dst-md softmax-bwd-pd) (buffer diff-dst-vec))
                      softmax-bwd (primitive softmax-bwd-pd)
                      softmax-bwd-args (softmax-bwd-args mem diff-dst-mem mem)]
         (primitive-kind softmax-desc) => :softmax
         (execute! s softmax softmax-args) => s
         mem-vec => (fv 0.06337893754243851 0.46831050515174866 0.46831050515174866
                        0.0024282580707222223 0.017942532896995544 0.9796292185783386)
         (execute! s softmax-bwd softmax-bwd-args) => s
         mem-vec => (fv 0.06337893754243851 -0.5316895246505737 0.46831050515174866
                        0.0024282580707222223 0.017942532896995544 -0.02037079446017742))) ; aLi - ti

(facts "Convolution forward."
       (with-release [eng (engine)
                      s (stream eng)
                      src-desc (memory-desc [2 1 4 4] :float :nchw)
                      weights-desc (memory-desc [1 1 3 3] :float :nchw)
                      bias-desc (memory-desc [1] :float :x)
                      dst-desc (memory-desc [2 1 2 2] :float :nchw)
                      conv-desc (convolution-forward-desc
                                 :inference :auto src-desc weights-desc bias-desc dst-desc
                                 [1 1] [0 0])
                      conv-pd (primitive-desc eng conv-desc)
                      src-vec (fv 0 43 3 30 0 98 0 0 7 38 0 0 19 20 175 50
                                  0 0 7 19 43 98 38 20 3 0 0 175 30 0 0 50)
                      src-mem (memory eng (src-md conv-pd) (buffer src-vec))
                      weights-vec (fv -2 0 1 0 1 0 -1 -2 0)
                      weights-mem (memory eng (weights-md conv-pd) (buffer weights-vec))
                      bias-vec (fv 0.5)
                      bias-mem (memory eng bias-desc (buffer bias-vec))
                      dst-vec (fv (* 2 1 2 2))
                      dst-mem (memory eng (dst-md conv-pd) (buffer dst-vec))
                      conv (primitive conv-pd)
                      conv-args (fwd-args src-mem weights-mem bias-mem dst-mem)]
         (primitive-kind conv-desc) => :convolution
         (execute! s conv conv-args) => s
         (seq dst-vec) => [18.5 -93.5 -20.5 -565.5 102.5 57.5 -77.5 -175.5]))

(facts "Convolution backward."
       (with-release [eng (engine)
                      s (stream eng)
                      src-desc (memory-desc [2 1 4 4] :float :nchw)
                      weights-desc (memory-desc [1 1 3 3] :float :nchw)
                      bias-desc (memory-desc [1] :float :x)
                      dst-desc (memory-desc [2 1 2 2] :float :nchw)
                      conv-desc (convolution-forward-desc
                                 :training :auto src-desc weights-desc bias-desc dst-desc
                                 [1 1] [0 0])
                      conv-pd (primitive-desc eng conv-desc)
                      src-vec (fv 0 43 3 30 0 98 0 0 7 38 0 0 19 20 175 50
                                  0 0 7 19 43 98 38 20 3 0 0 175 30 0 0 50)
                      src-mem (memory eng (src-md conv-pd) (buffer src-vec))
                      weights-vec (fv -2 0 1 0 1 0 -1 -2 0)
                      weights-mem (memory eng (weights-md conv-pd) (buffer weights-vec))
                      bias-vec (fv 0.5)
                      bias-mem (memory eng bias-desc (buffer bias-vec))
                      dst-vec (fv 8)
                      dst-mem (memory eng (dst-md conv-pd) (buffer dst-vec))
                      conv (primitive conv-pd)
                      conv-args (fwd-args src-mem weights-mem bias-mem dst-mem)

                      diff-src-desc (memory-desc [2 1 4 4] :float :nchw)
                      diff-dst-desc (memory-desc [2 1 2 2] :float :nchw)
                      conv-bwd-data-desc (convolution-backward-desc
                                          :auto diff-src-desc weights-desc diff-dst-desc
                                          [1 1] [0 0] [0 0])
                      conv-bwd-data-pd (primitive-desc eng conv-bwd-data-desc conv-pd)
                      diff-src-vec (fv 32)
                      diff-dst-vec (fv [0.2 0.3 0.8 1 1 1 1 1])
                      diff-src-mem (memory eng (diff-src-md conv-bwd-data-pd) (buffer diff-src-vec))
                      diff-dst-mem (memory eng (diff-dst-md conv-bwd-data-pd) (buffer diff-dst-vec))
                      conv-bwd-data (primitive conv-bwd-data-pd)
                      conv-bwd-data-args (bwd-args diff-dst-mem weights-mem diff-src-mem)
                      conv-bwd-weights-desc (convolution-backward-desc
                                             :auto src-desc weights-desc bias-desc dst-desc
                                             [1 1] [0 0] [0 0])
                      conv-bwd-weights-pd (primitive-desc eng conv-bwd-weights-desc conv-pd)
                      diff-weights-vec (fv 9)
                      diff-bias-vec (fv [1.5])
                      diff-weights-mem (memory eng (diff-weights-md conv-bwd-weights-pd)
                                               (buffer diff-weights-vec))
                      diff-bias-mem (memory eng bias-desc (buffer diff-bias-vec))
                      conv-bwd-weights (primitive conv-bwd-weights-pd)
                      conv-bwd-weights-args (bwd-args src-mem diff-dst-mem
                                                      diff-weights-mem diff-bias-mem)]
         (primitive-kind conv-desc) => :convolution
         (execute! s conv conv-args) => s
         (seq dst-vec) => [18.5 -93.5 -20.5 -565.5 102.5 57.5 -77.5 -175.5]
         diff-src-vec => (fv 32)
         (execute! s conv-bwd-data conv-bwd-data-args) => s
         (seq diff-src-vec)
         => (map float [-0.4 -0.6 0.2 0.3 -1.6 -1.8 1.1 1.0 -0.2 0.09999999403953552
                        0.3999999761581421 0.0 -0.8 -2.6 -2.0 0.0 -2.0 -2.0 1.0 1.0
                        -2.0 -1.0 2.0 1.0 -1.0 -2.0 -1.0 0.0 -1.0 -3.0 -2.0 0.0])
         (execute! s conv-bwd-weights conv-bwd-weights-args) => s
         (seq diff-weights-vec) => (map float [251.9 230.9 93.6 217.0 186.0 233.0 81.0 198.6 415.0])
         (entry diff-bias-vec 0) => (float 6.3)))

(facts "Dilated convolution forward."
       (with-release [eng (engine)
                      s (stream eng)
                      src-desc (memory-desc [2 1 4 4] :float :nchw)
                      weights-desc (memory-desc [1 1 3 3] :float :nchw)
                      bias-desc (memory-desc [1] :float :x)
                      dst-desc (memory-desc [2 1 2 2] :float :nchw)
                      conv-desc (dilated-convolution-forward-desc
                                 :inference :auto src-desc weights-desc bias-desc dst-desc
                                 [1 1] [0 0] [0 0])
                      conv-pd (primitive-desc eng conv-desc)
                      src-vec (fv 0 43 3 30 0 98 0 0 7 38 0 0 19 20 175 50
                                  0 0 7 19 43 98 38 20 3 0 0 175 30 0 0 50)
                      src-mem (memory eng (src-md conv-pd) (buffer src-vec))
                      weights-vec (fv -2 0 1 0 1 0 -1 -2 0)
                      weights-mem (memory eng (weights-md conv-pd) (buffer weights-vec))
                      bias-vec (fv 0.5)
                      bias-mem (memory eng bias-desc (buffer bias-vec))
                      dst-vec (fv (* 2 1 2 2))
                      dst-mem (memory eng (dst-md conv-pd) (buffer dst-vec))
                      conv (primitive conv-pd)
                      conv-args (fwd-args src-mem weights-mem bias-mem dst-mem)]
         (primitive-kind conv-desc) => :convolution
         (execute! s conv conv-args) => s
         (seq dst-vec) => [18.5 -93.5 -20.5 -565.5 102.5 57.5 -77.5 -175.5]))

(facts "Convolution backward."
       (with-release [eng (engine)
                      s (stream eng)
                      src-desc (memory-desc [2 1 4 4] :float :nchw)
                      weights-desc (memory-desc [1 1 3 3] :float :nchw)
                      bias-desc (memory-desc [1] :float :x)
                      dst-desc (memory-desc [2 1 2 2] :float :nchw)
                      conv-desc (dilated-convolution-forward-desc
                                 :training :auto src-desc weights-desc bias-desc dst-desc
                                 [1 1] [0 0] [0 0])
                      conv-pd (primitive-desc eng conv-desc)
                      src-vec (fv 0 43 3 30 0 98 0 0 7 38 0 0 19 20 175 50
                                  0 0 7 19 43 98 38 20 3 0 0 175 30 0 0 50)
                      src-mem (memory eng (src-md conv-pd) (buffer src-vec))
                      weights-vec (fv -2 0 1 0 1 0 -1 -2 0)
                      weights-mem (memory eng (weights-md conv-pd) (buffer weights-vec))
                      bias-vec (fv 0.5)
                      bias-mem (memory eng bias-desc (buffer bias-vec))
                      dst-vec (fv 8)
                      dst-mem (memory eng (dst-md conv-pd) (buffer dst-vec))
                      conv (primitive conv-pd)
                      conv-args (fwd-args src-mem weights-mem bias-mem dst-mem)

                      diff-src-desc (memory-desc [2 1 4 4] :float :nchw)
                      diff-dst-desc (memory-desc [2 1 2 2] :float :nchw)
                      conv-bwd-data-desc (dilated-convolution-backward-desc
                                          :auto diff-src-desc weights-desc diff-dst-desc
                                          [1 1] [0 0] [0 0] [0 0])
                      conv-bwd-data-pd (primitive-desc eng conv-bwd-data-desc conv-pd)
                      diff-src-vec (fv 32)
                      diff-dst-vec (fv [0.2 0.3 0.8 1 1 1 1 1])
                      diff-src-mem (memory eng (diff-src-md conv-bwd-data-pd) (buffer diff-src-vec))
                      diff-dst-mem (memory eng (diff-dst-md conv-bwd-data-pd) (buffer diff-dst-vec))
                      conv-bwd-data (primitive conv-bwd-data-pd)
                      conv-bwd-data-args (bwd-args diff-dst-mem weights-mem diff-src-mem)
                      conv-bwd-weights-desc (dilated-convolution-backward-desc
                                             :auto src-desc weights-desc bias-desc dst-desc
                                             [1 1] [0 0] [0 0] [0 0])
                      conv-bwd-weights-pd (primitive-desc eng conv-bwd-weights-desc conv-pd)
                      diff-weights-vec (fv 9)
                      diff-bias-vec (fv [1.5])
                      diff-weights-mem (memory eng (diff-weights-md conv-bwd-weights-pd)
                                               (buffer diff-weights-vec))
                      diff-bias-mem (memory eng bias-desc (buffer diff-bias-vec))
                      conv-bwd-weights (primitive conv-bwd-weights-pd)
                      conv-bwd-weights-args (bwd-args src-mem diff-dst-mem
                                                      diff-weights-mem diff-bias-mem)]
         (primitive-kind conv-desc) => :convolution
         (execute! s conv conv-args) => s
         (seq dst-vec) => [18.5 -93.5 -20.5 -565.5 102.5 57.5 -77.5 -175.5]
         diff-src-vec => (fv 32)
         (execute! s conv-bwd-data conv-bwd-data-args) => s
         (seq diff-src-vec)
         => (map float [-0.4 -0.6 0.2 0.3 -1.6 -1.8 1.1 1.0 -0.2 0.09999999403953552
                        0.3999999761581421 0.0 -0.8 -2.6 -2.0 0.0 -2.0 -2.0 1.0 1.0
                        -2.0 -1.0 2.0 1.0 -1.0 -2.0 -1.0 0.0 -1.0 -3.0 -2.0 0.0])
         (execute! s conv-bwd-weights conv-bwd-weights-args) => s
         (seq diff-weights-vec) => (map float [251.9 230.9 93.6 217.0 186.0 233.0 81.0 198.6 415.0])
         (entry diff-bias-vec 0) => (float 6.3)))

(facts "Max pooling forward."
       (with-release [eng (engine)
                      s (stream eng)
                      src-desc (memory-desc [2 1 4 4] :float :nchw)
                      dst-desc (memory-desc [2 1 2 2] :float :nchw)
                      pool-desc (pooling-fwd-desc :inference :max src-desc dst-desc [2 2] [2 2] [0 0])
                      pool-pd (primitive-desc eng pool-desc)
                      src-vec (fv 0 43 3 30 0 98 0 0 7 38 0 0 19 20 175 50
                                  0 0 7 19 43 98 38 20 3 0 0 175 30 0 0 50)
                      src-mem (memory eng src-desc (buffer src-vec))
                      dst-vec (fv (* 2 1 2 2))
                      dst-mem (memory eng dst-desc (buffer dst-vec))
                      pool (primitive pool-pd)
                      pool-args (fwd-args src-mem dst-mem)]
         (primitive-kind pool-desc) => :pooling
         (execute! s pool pool-args) => s
         src-vec => (fv 0 43 3 30 0 98 0 0 7 38 0 0 19 20 175 50
                        0 0 7 19 43 98 38 20 3 0 0 175 30 0 0 50)
         (seq dst-vec) => [98.0 30.0 38.0 175.0 98.0 38.0 30.0 175.0]))

(facts "Max pooling backward."
       (with-release [eng (engine)
                      s (stream eng)
                      src-desc (memory-desc [2 1 4 4] :float :nchw)
                      dst-desc (memory-desc [2 1 2 2] :float :nchw)
                      pool-desc (pooling-fwd-desc :training :max src-desc dst-desc [2 2] [2 2] [0 0])
                      pool-pd (primitive-desc eng pool-desc)
                      src-vec (fv 0 43 3 30 0 98 0 0 7 38 0 0 19 20 175 50
                                  0 0 7 19 43 98 38 20 3 0 0 175 30 0 0 50)
                      src-mem (memory eng src-desc (buffer src-vec))
                      dst-vec (fv (* 2 1 2 2))
                      dst-mem (memory eng dst-desc (buffer dst-vec))
                      workspace-mem (memory eng (workspace-md pool-pd))
                      pool (primitive pool-pd)
                      pool-args (fwd-args src-mem dst-mem workspace-mem)

                      pool-bwd-desc (pooling-bwd-desc :max src-desc dst-desc [2 2] [2 2] [0 0])
                      diff-dst-vec (entry! (zero src-vec) 2.0)
                      diff-src-vec (entry! (zero src-vec) 0.0)
                      pool-bwd-pd (primitive-desc eng pool-bwd-desc pool-pd)
                      diff-dst-mem (memory eng (diff-dst-md pool-bwd-pd) (buffer diff-dst-vec))
                      diff-src-mem (memory eng (diff-src-md pool-bwd-pd) (buffer diff-src-vec))
                      pool-bwd (primitive pool-bwd-pd)
                      pool-bwd-args (pooling-bwd-args diff-dst-mem diff-src-mem workspace-mem)]
         (primitive-kind pool-desc) => :pooling
         (execute! s pool pool-args) => s
         src-vec => (fv 0 43 3 30 0 98 0 0 7 38 0 0 19 20 175 50
                        0 0 7 19 43 98 38 20 3 0 0 175 30 0 0 50)
         (seq dst-vec) => [98.0 30.0 38.0 175.0 98.0 38.0 30.0 175.0]
         (execute! s pool-bwd pool-bwd-args)
         (seq diff-src-vec)
         => [0.0 0.0 0.0 2.0 0.0 2.0 0.0 0.0 0.0 2.0 0.0 0.0 0.0 0.0 2.0 0.0
             0.0 0.0 0.0 0.0 0.0 2.0 2.0 0.0 0.0 0.0 0.0 2.0 2.0 0.0 0.0 0.0]))

(facts "Batch normalization forward."
       (with-release [eng (engine)
                      s (stream eng)
                      data-desc (memory-desc [2 1 2 2] :float :nchw)
                      scaleshift-desc (memory-desc [2 1] :float :nc)
                      bnrm-desc (batch-norm-fwd-desc :inference data-desc :scaleshift)
                      bnrm-pd (primitive-desc eng bnrm-desc)
                      src-vec (fv (range -1 7))
                      src-mem (memory eng data-desc (buffer src-vec))
                      dst-vec (fv 8)
                      dst-mem (memory eng data-desc (buffer dst-vec))
                      scaleshift-vec (fv [0.5 1.5])
                      scaleshift-mem (memory eng scaleshift-desc (buffer scaleshift-vec))
                      bnrm (primitive bnrm-pd)
                      bnrm-args (batch-norm-fwd-args src-mem dst-mem scaleshift-mem)]
         (primitive-kind bnrm-desc) => :batch-normalization
         (execute! s bnrm bnrm-args) => s
         (seq src-vec) => (range -1.0 7.0)
         (seq dst-vec) => [0.7362374067306519 0.9544553160667419 1.172673225402832 1.3908910751342773
                           1.6091089248657227 1.827326774597168 2.0455446243286133 2.2637624740600586]))

(facts "Batch normalization backward."
       (with-release [eng (engine)
                      s (stream eng)
                      data-desc (memory-desc [1 2 2 2] :float :nchw)
                      stats-desc (memory-desc [1 2] :float :nc)
                      bnrm-desc (batch-norm-fwd-desc :training data-desc :scaleshift)
                      bnrm-pd (primitive-desc eng bnrm-desc)
                      src-vec (fv (range -1 7))
                      src-mem (memory eng data-desc (buffer src-vec))
                      dst-vec (fv 8)
                      dst-mem (memory eng data-desc (buffer dst-vec))
                      scaleshift-desc (memory-desc [2 2] :float :nc)
                      scaleshift-vec (fv [0.5 1.5 1 1])
                      scaleshift-mem (memory eng stats-desc (buffer scaleshift-vec))
                      mean-vec (fv 2)
                      mean-mem (memory eng stats-desc (buffer mean-vec))
                      variance-vec (fv 2)
                      variance-mem (memory eng stats-desc (buffer variance-vec))
                      bnrm (primitive bnrm-pd)
                      bnrm-args (batch-norm-fwd-args src-mem dst-mem scaleshift-mem
                                                     mean-mem variance-mem)
                      bnrm-bwd-desc (batch-norm-bwd-desc :backward data-desc data-desc :scaleshift)
                      bnrm-bwd-pd (primitive-desc eng bnrm-bwd-desc bnrm-pd)
                      diff-dst-vec (fv [-5 10 0.3 0.2 -0.5 0.6 0.9 -3])
                      diff-dst-mem (memory eng (diff-dst-md bnrm-bwd-pd) (buffer diff-dst-vec))
                      diff-src-vec (fv 8)
                      diff-src-mem (memory eng (diff-src-md bnrm-bwd-pd) (buffer diff-src-vec))
                      diff-scaleshift-vec (fv 4)
                      diff-scaleshift-mem (memory eng scaleshift-desc (buffer diff-scaleshift-vec))
                      bnrm-bwd (primitive bnrm-bwd-pd)
                      bnrm-bwd-args (batch-norm-bwd-args diff-dst-mem src-mem scaleshift-mem
                                                         mean-mem variance-mem
                                                         diff-src-mem diff-scaleshift-mem)]
         (primitive-kind bnrm-desc) => :batch-normalization
         (execute! s bnrm bnrm-args) => s
         (seq src-vec) => (range -1.0 7.0)
         (seq dst-vec)
         => [0.32917964458465576 0.7763931751251221 1.223606824874878 1.6708203554153442
             -1.0124611854553223 0.32917964458465576 1.6708203554153442 3.0124611854553223]
         (seq mean-vec) => [0.5 4.5]
         (seq variance-vec) => [1.25 1.25]
         (execute! s bnrm-bwd bnrm-bwd-args)
         (seq diff-src-vec)
         => [-2.455202579498291 3.989145278930664 -0.6126827001571655 -0.9212599992752075
             -1.4489718675613403 0.9928141236305237 2.3612875938415527 -1.9051299095153809]
         (seq diff-scaleshift-vec) => [2.6385602951049805 -3.219937801361084 5.5 -2.0]))

(facts "In-place Binary operation"
       (with-release [eng (engine)
                      s (stream eng)
                      md (memory-desc [2 3 4 5] :float :nchw)
                      buf0 (direct-buffer (size md))
                      src0 (memory eng md buf0)
                      buf1 (direct-buffer (size md))
                      src1 (memory eng md buf1)
                      add-desc (binary-desc :add md)
                      add-pd (primitive-desc eng add-desc)
                      add-prim (primitive add-pd)
                      add-args (binary-args src0 src1)]
         (put-float buf0 0 -100)
         (put-float buf0 1 20)
         (put-float buf1 0 -200)
         (put-float buf1 1 30)
         (execute! s add-prim add-args) => s
         (get-float buf0 0) => -300.0
         (get-float buf0 1) => 50.0))

(facts "Reduction sum operation"
       (with-release [eng (engine)
                      s (stream eng)
                      src-md (memory-desc [2 3 4 5] :float :nchw)
                      src-buf (direct-buffer (size src-md))
                      src (memory eng src-md src-buf)
                      dst-md (memory-desc [2 3 4 1] :float :nchw)
                      dst-buf (direct-buffer (size dst-md))
                      dst (memory eng dst-md dst-buf)
                      sum-desc (reduction-desc :sum src-md dst-md)
                      sum-pd (primitive-desc eng sum-desc)
                      sum-prim (primitive sum-pd)
                      sum-args (fwd-args src dst)]
         (dotimes [i 120]
           (put-float src-buf i i))
         (execute! s sum-prim sum-args) => s
         (get-float dst-buf 0) => (float (apply + (range 5)))
         (get-float dst-buf 1) => (float (apply + (range 5 10)))
         (get-float dst-buf 2) => (float (apply + (range 10 15)))))

(facts "Reduction max operation"
       (with-release [eng (engine)
                      s (stream eng)
                      src-md (memory-desc [2 3 4 5] :float :nchw)
                      src-buf (direct-buffer (size src-md))
                      src (memory eng src-md src-buf)
                      dst-md (memory-desc [1 3 1 1] :float :nchw)
                      dst-buf (direct-buffer (size dst-md))
                      dst (memory eng dst-md dst-buf)
                      max-desc (reduction-desc :max src-md dst-md)
                      max-pd (primitive-desc eng max-desc)
                      max-prim (primitive max-pd)
                      max-args (fwd-args src dst)]
         (dotimes [i 120]
           (put-float src-buf i i))
         (execute! s max-prim max-args) => s
         (get-float dst-buf 0) => 79.0
         (get-float dst-buf 1) => 99.0
         (get-float dst-buf 2) => 119.0))

(facts "Reduction L2 operation"
       (with-release [eng (engine)
                      s (stream eng)
                      src-md (memory-desc [2 3 4 5] :float :nchw)
                      src-buf (direct-buffer (size src-md))
                      src (memory eng src-md src-buf)
                      dst-md (memory-desc [2 3 4 1] :float :nchw)
                      dst-buf (direct-buffer (size dst-md))
                      dst (memory eng dst-md dst-buf)
                      norm-desc (reduction-desc :norm-lp-sum src-md dst-md 2.0 0.0)
                      norm-pd (primitive-desc eng norm-desc)
                      norm-prim (primitive norm-pd)
                      norm-args (fwd-args src dst)]
         (dotimes [i 120]
           (put-float src-buf i i))
         (execute! s norm-prim norm-args) => s
         (get-float dst-buf 0) => (float (sqrt (apply + (map sqr (range 5)))))
         (get-float dst-buf 1) => (float (sqrt (apply + (map sqr (range 5 10)))))
         (get-float dst-buf 2) => (float (sqrt (apply + (map sqr (range 10 15)))))))

(facts "Concatenate operation with one source"
       (with-release [eng (engine)
                      s (stream eng)
                      md (memory-desc [2 3 4 5] :float :nchw)
                      src-buf (direct-buffer (size md))
                      src (memory eng md src-buf)
                      dst-buf (direct-buffer (size md))
                      dst (memory eng md dst-buf)
                      concat-pd (concatenate eng md 0 md)
                      concat-prim (primitive concat-pd)
                      concat-args (multi-args dst src)]
         (dotimes [i 120]
           (put-float src-buf i i))
         (execute! s concat-prim concat-args) => s
         (get-float dst-buf 0) => 0.0
         (get-float dst-buf 100) => 100.0))

(facts "Concatenate operation with two homogeneous sources"
       (with-release [eng (engine)
                      s (stream eng)
                      src-md (memory-desc [2 3 4 5] :float :nchw)
                      dst-md (memory-desc [4 3 4 5] :float :nchw)
                      src0-buf (direct-buffer (size src-md))
                      src0 (memory eng src-md src0-buf)
                      src1-buf (direct-buffer (size src-md))
                      src1 (memory eng src-md src1-buf)
                      dst-buf (direct-buffer (size dst-md))
                      dst (memory eng dst-md dst-buf)
                      concat-pd (concatenate eng dst-md 0 src-md src-md)
                      concat-prim (primitive concat-pd)
                      concat-args (multi-args dst src0 src1)]
         (dotimes [i 120]
           (put-float src0-buf i i)
           (put-float src1-buf i (* 1000.0 i)))
         (execute! s concat-prim concat-args) => s
         (get-float dst-buf 0) => 0.0
         (get-float dst-buf 100) => 100.0
         (get-float dst-buf 121) => 1000.0
         (get-float dst-buf 220) => 100000.0))

(facts "Concatenate operation with three heterogeneous sources"
       (with-release [eng (engine)
                      s (stream eng)
                      src0-md (memory-desc [1 1 2 1] :float :nchw)
                      src1-md (memory-desc [1 1 1 1] :float :nchw)
                      src2-md (memory-desc [1 1 1 1] :float :nchw)
                      dst-md (memory-desc [1 1 4 1] :float :nchw)
                      src0-buf (direct-buffer (size src0-md))
                      src0 (memory eng src1-md src0-buf)
                      src1-buf (direct-buffer (size src1-md))
                      src1 (memory eng src1-md src1-buf)
                      src2-buf (direct-buffer (size src2-md))
                      src2 (memory eng src2-md src2-buf)
                      dst-buf (direct-buffer (size dst-md))
                      dst (memory eng dst-md dst-buf)
                      concat-pd (concatenate eng dst-md 2 src0-md src1-md src2-md)
                      concat-prim (primitive concat-pd)
                      concat-args (multi-args dst src0 src1 src2)]
         (put-float src0-buf 0 1.0)
         (put-float src0-buf 1 2.0)
         (put-float src1-buf 0 10.0)
         (put-float src2-buf 0 5.0)
         (execute! s concat-prim concat-args) => s
         (get-float dst-buf 0) => 1.0
         (get-float dst-buf 1) => 2.0
         (get-float dst-buf 2) => 10.0
         (get-float dst-buf 3) => 5.0))

(facts
 "Vanilla RNN dimensions."
 (let [T 2
       N 1
       SC 4
       DC 2
       G 1
       L 1
       D 1
       src-dim [T N SC]
       src-iter-dim [L D N DC]
       weights-dim [L D SC G DC]
       weights-iter-dim [L D DC G DC]
       bias-dim [L D G DC]
       dst-dim [T N DC]
       dst-iter-dim [L D N DC]]
   (with-release
     [eng (engine)
      s (stream eng)
      src-desc (memory-desc src-dim :float :tnc)
      src-iter-desc (memory-desc src-iter-dim :float :ldnc)
      weights-desc (memory-desc weights-dim :float :ldigo)
      weights-iter-desc (memory-desc weights-iter-dim :float :ldigo)
      bias-desc (memory-desc bias-dim :float :ldgo)
      dst-desc (memory-desc dst-dim :float :tnc)
      dst-iter-desc (memory-desc dst-iter-dim :float :ldnc)]
     (vanilla-rnn-fwd-desc :inference :relu :unidirectional
                           src-desc nil weights-desc weights-iter-desc bias-desc
                           dst-desc nil) => truthy
     (vanilla-rnn-fwd-desc :inference :relu :unidirectional
                           src-desc src-iter-desc weights-desc weights-iter-desc bias-desc
                           dst-desc dst-iter-desc) => truthy)))

(facts
 "LSTM dimensions."
 (let [T 2
       N 1
       SC 4
       DC 2
       G 4
       L 2
       D 1
       src-dim [T N SC]
       src-iter-dim [L D N DC]
       weights-dim [L D SC G DC]
       weights-iter-dim [L D DC G DC]
       bias-dim [L D G DC]
       dst-dim [T N DC]
       dst-iter-dim [L D N DC]]
   (with-release
     [eng (engine)
      s (stream eng)
      src-desc (memory-desc src-dim :float :tnc)
      src-iter-desc (memory-desc src-iter-dim :float :ldnc)
      src-iter-c-desc (memory-desc src-iter-dim :float :ldnc)
      weights-desc (memory-desc weights-dim :float :ldigo)
      weights-iter-desc (memory-desc weights-iter-dim :float :ldigo)
      bias-desc (memory-desc bias-dim :float :ldgo)
      dst-desc (memory-desc dst-dim :float :tnc)
      dst-iter-desc (memory-desc dst-iter-dim :float :ldnc)
      dst-iter-c-desc (memory-desc dst-iter-dim :float :ldnc)]
     (lstm-fwd-desc :inference :unidirectional
                                      src-desc nil nil weights-desc weights-desc bias-desc
                                      dst-desc nil nil) => (throws Exception)
     (lstm-fwd-desc :inference :unidirectional
                    src-desc nil nil weights-desc weights-iter-desc bias-desc
                    dst-desc nil nil) => truthy
     (lstm-fwd-desc :inference :unidirectional
                    src-desc src-iter-desc src-iter-c-desc
                    weights-desc weights-iter-desc bias-desc
                    dst-desc dst-iter-desc dst-iter-c-desc) => truthy)))

(facts
 "GRU dimensions."
 (let [T 2
       N 1
       SC 4
       DC 2
       G 3
       L 1
       D 1
       src-dim [T N SC]
       src-iter-dim [L D N DC]
       weights-dim [L D SC G DC]
       weights-iter-dim [L D DC G DC]
       bias-dim [L D G DC]
       dst-dim [T N DC]
       dst-iter-dim [L D N DC]]
   (with-release
     [eng (engine)
      s (stream eng)
      src-desc (memory-desc src-dim :float :tnc)
      src-iter-desc (memory-desc src-iter-dim :float :ldnc)
      weights-desc (memory-desc weights-dim :float :ldigo)
      weights-iter-desc (memory-desc weights-iter-dim :float :ldigo)
      bias-desc (memory-desc bias-dim :float :ldgo)
      dst-desc (memory-desc dst-dim :float :tnc)
      dst-iter-desc (memory-desc dst-iter-dim :float :ldnc)]
     (gru-fwd-desc :inference :unidirectional
                   src-desc nil weights-desc weights-desc bias-desc
                   dst-desc nil) => (throws Exception)
     (gru-fwd-desc :inference :unidirectional
                   src-desc nil weights-desc weights-iter-desc bias-desc
                   dst-desc nil) => truthy
     (gru-fwd-desc :inference :unidirectional
                   src-desc src-iter-desc
                   weights-desc weights-iter-desc bias-desc
                   dst-desc dst-iter-desc) => truthy)))

(facts
 "Vanilla RNN forward."
 (let [T 2
       N 1
       C 2
       G 1
       L 2
       D 1
       src-dim [T N C]
       src-iter-dim [L D N C]
       weights-dim [L D C G C]
       bias-dim [L D G C]]
   (with-release
     [eng (engine)
      s (stream eng)
      src-desc (memory-desc src-dim :float :tnc)
      src-iter-desc (memory-desc src-iter-dim :float :ldnc)
      weights-desc (memory-desc weights-dim :float :ldigo)
      bias-desc (memory-desc bias-dim :float :ldgo)
      dst-desc (memory-desc src-dim :float :tnc)
      dst-iter-desc (memory-desc src-iter-dim :float :ldnc)
      rnn-desc (vanilla-rnn-fwd-desc :inference :relu :unidirectional
                                     src-desc src-iter-desc weights-desc weights-desc bias-desc
                                     dst-desc dst-iter-desc)
      rnn-no-iter-desc (vanilla-rnn-fwd-desc :inference :relu :unidirectional
                                             src-desc nil weights-desc weights-desc bias-desc
                                             dst-desc nil)
      rnn-no-iter-pd (primitive-desc eng rnn-no-iter-desc)
      rnn-pd (primitive-desc eng rnn-desc)
      src-vec (fv [2 3 0.2 0.3])
      src-mem (memory eng (arg-md rnn-pd :src) (buffer src-vec))
      src-iter-vec (fv (apply * src-iter-dim))
      src-iter-mem (memory eng (arg-md rnn-pd :src-iter) (buffer src-iter-vec))
      weights-vec (fv [0.1 0.2 0.3 0.4 0.3 0.4 0.5 0.6])
      weights-mem (memory eng (arg-md rnn-pd :weights) (buffer weights-vec))
      weights-iter-vec (fv [100 200 300 400 0.01 0.02 0.03 0.04])
      weights-iter-mem (memory eng (arg-md rnn-pd :weights-iter) (buffer weights-iter-vec))
      bias-vec (fv [0.3 0.7 1 2])
      bias-mem (memory eng bias-desc (buffer bias-vec))
      dst-vec (fv (apply * src-dim))
      dst-mem (memory eng (arg-md rnn-pd :dst) (buffer dst-vec))
      dst-iter-vec (fv (apply * src-dim))
      dst-iter-mem (memory eng (arg-md rnn-pd :dst-iter) (buffer dst-iter-vec))
      workspace-mem (memory eng (arg-md rnn-pd :workspace))
      rnn (primitive rnn-pd)
      rnn-args (args {:src-layer src-mem
                      :src-iter src-iter-mem
                      :weights-layer weights-mem
                      :weights-iter weights-iter-mem
                      :bias bias-mem
                      :dst-layer dst-mem
                      :dst-iter dst-iter-mem
                      :workspace workspace-mem})
      rnn-no-iter (primitive rnn-no-iter-pd)
      rnn-no-iter-args (args {:src-layer src-mem
                              :src-iter nil
                              :weights-layer weights-mem
                              :weights-iter weights-iter-mem
                              :bias bias-mem
                              :dst-layer dst-mem
                              :dst-iter nil
                              :workspace workspace-mem})]
     (primitive-kind rnn-desc) => :rnn
     (execute! s rnn rnn-args) => s
     (seq src-iter-vec) => [0.0 0.0 0.0 0.0]
     (seq dst-vec) => [2.570000171661377 3.940000057220459 850.6968994140625 1054.8890380859375]
     (seq dst-iter-vec) => [830.4099731445312 1200.8599853515625 850.6968994140625 1054.8890380859375]
     (entry! dst-vec 0)
     (primitive-kind rnn-no-iter-desc) => :rnn
     (arg-md rnn-no-iter-pd :src-iter) => nil
     (zero-desc? (arg-md rnn-no-iter-pd :src-iter)) => true
     (execute! s rnn-no-iter rnn-no-iter-args) => s
     (seq dst-vec) => [2.570000171661377 3.940000057220459 850.6968994140625 1054.8890380859375])))

(facts
 "Vanilla RNN training."
 (let [T 2
       N 1
       C 2
       G 1
       L 2
       D 1
       src-dim [T N C]
       src-iter-dim [L D N C]
       weights-dim [L D C G C]
       bias-dim [L D G C]]
   (with-release
     [eng (engine)
      s (stream eng)
      src-desc (memory-desc src-dim :float :tnc)
      src-iter-desc (memory-desc src-iter-dim :float :ldnc)
      weights-desc (memory-desc weights-dim :float :ldigo)
      bias-desc (memory-desc bias-dim :float :ldgo)
      dst-desc (memory-desc src-dim :float :tnc)
      dst-iter-desc (memory-desc src-iter-dim :float :ldnc)
      rnn-fwd-desc (vanilla-rnn-fwd-desc :training :relu :unidirectional src-desc src-iter-desc
                                         weights-desc weights-desc bias-desc dst-desc dst-iter-desc)
      rnn-fwd-pd (primitive-desc eng rnn-fwd-desc)
      src-vec (fv [2 3 0.2 0.3])
      src-mem (memory eng (arg-md rnn-fwd-pd :src) (buffer src-vec))
      src-iter-vec (fv (apply * src-iter-dim))
      src-iter-mem (memory eng (arg-md rnn-fwd-pd :src-iter) (buffer src-iter-vec))
      weights-vec (fv [0.1 0.2 0.3 0.4 0.3 0.4 0.5 0.6])
      weights-mem (memory eng (arg-md rnn-fwd-pd :weights) (buffer weights-vec))
      weights-iter-vec (fv [100 200 300 400 0.01 0.02 0.03 0.04])
      weights-iter-mem (memory eng (arg-md rnn-fwd-pd :weights-iter) (buffer weights-iter-vec))
      bias-vec (fv [0.3 0.7 1 2])
      bias-mem (memory eng bias-desc (buffer bias-vec))
      dst-vec (fv (apply * src-dim))
      dst-mem (memory eng (arg-md rnn-fwd-pd :dst-iter) (buffer dst-vec))
      dst-iter-vec (fv (apply * src-dim))
      dst-iter-mem (memory eng (arg-md rnn-fwd-pd :dst-iter) (buffer dst-iter-vec))
      workspace-mem (memory eng (arg-md rnn-fwd-pd :workspace))
      rnn-fwd (primitive rnn-fwd-pd)
      rnn-fwd-args (args {:src-layer src-mem
                          :src-iter src-iter-mem
                          :weights-layer weights-mem
                          :weights-iter weights-iter-mem
                          :bias bias-mem
                          :dst-layer dst-mem
                          :dst-iter dst-iter-mem
                          :workspace workspace-mem})
      bwd-weights-desc (memory-desc weights-dim :float :any)
      rnn-bwd-desc (vanilla-rnn-bwd-desc :relu :unidirectional src-desc src-iter-desc
                                         bwd-weights-desc bwd-weights-desc bias-desc
                                         dst-desc dst-iter-desc
                                         src-desc src-iter-desc
                                         bwd-weights-desc bwd-weights-desc bias-desc
                                         dst-desc dst-iter-desc)
      rnn-bwd-pd (primitive-desc eng rnn-bwd-desc rnn-fwd-pd)
      bwd-weights-mem (memory eng (arg-md rnn-bwd-pd :weights))
      reorder-weights-fb-pd (reorder eng weights-mem bwd-weights-mem)
      reorder-weights-bf-pd (reorder eng bwd-weights-mem weights-mem)
      reorder-weights-fb (primitive reorder-weights-fb-pd)
      reorder-weights-bf (primitive reorder-weights-bf-pd)
      bwd-weights-iter-mem (memory eng (arg-md rnn-bwd-pd :weights-iter))
      reorder-weights-iter-fb-pd (reorder eng weights-iter-mem bwd-weights-iter-mem)
      reorder-weights-iter-bf-pd (reorder eng bwd-weights-iter-mem weights-iter-mem)
      reorder-weights-iter-fb (primitive reorder-weights-iter-fb-pd)
      reorder-weights-iter-bf (primitive reorder-weights-iter-bf-pd)
      diff-weights-vec (fv [0.1 0.2 0.3 0.4 0.3 0.4 0.5 0.6])
      diff-weights-packed-mem (memory eng weights-desc (buffer diff-weights-vec))
      diff-weights-mem (memory eng (arg-md rnn-bwd-pd :diff-weights))
      reorder-diff-weights-pack-pd (reorder eng diff-weights-mem diff-weights-packed-mem)
      reorder-diff-weights-unpack-pd (reorder eng diff-weights-packed-mem diff-weights-mem)
      reorder-diff-weights-pack (primitive reorder-diff-weights-pack-pd)
      reorder-diff-weights-unpack (primitive reorder-diff-weights-unpack-pd)
      diff-weights-iter-vec (fv [100 200 300 400 0.01 0.02 0.03 0.04])
      diff-weights-iter-packed-mem (memory eng weights-desc (buffer diff-weights-iter-vec))
      diff-weights-iter-mem (memory eng (arg-md rnn-bwd-pd :diff-weights-iter))
      reorder-diff-weights-iter-pack-pd (reorder eng diff-weights-iter-mem diff-weights-iter-packed-mem)
      reorder-diff-weights-iter-unpack-pd (reorder eng diff-weights-iter-packed-mem diff-weights-iter-mem)
      reorder-diff-weights-iter-pack (primitive reorder-diff-weights-iter-pack-pd)
      reorder-diff-weights-iter-unpack (primitive reorder-diff-weights-iter-unpack-pd)
      diff-dst-vec (fv [1.1 -2.2 3.3 -4.4])
      diff-dst-mem (memory eng (arg-md rnn-bwd-pd :diff-dst) (buffer diff-dst-vec))
      diff-dst-iter-vec (fv [-1 2 0.1 -0.2])
      diff-dst-iter-mem (memory eng (arg-md rnn-fwd-pd :diff-dst-iter) (buffer diff-dst-iter-vec))
      rnn-bwd-args (args {:src-layer src-mem
                          :src-iter src-iter-mem
                          :weights-layer bwd-weights-mem
                          :weights-iter bwd-weights-iter-mem
                          :bias bias-mem
                          :dst-layer dst-mem
                          :dst-iter dst-iter-mem
                          :workspace workspace-mem
                          :diff-src-layer src-mem
                          :diff-src-iter src-iter-mem
                          :diff-weights-layer diff-weights-mem
                          :diff-weights-iter diff-weights-iter-mem
                          :diff-bias bias-mem
                          :diff-dst-layer diff-dst-mem
                          :diff-dst-iter diff-dst-iter-mem})
      rnn-bwd (primitive rnn-bwd-pd)]

     (primitive-kind rnn-fwd-desc) => :rnn
     (execute! s rnn-fwd rnn-fwd-args) => s
     (seq dst-vec) => [2.570000171661377 3.940000057220459 850.6968994140625 1054.8890380859375]
     (seq src-vec) => (map float [2.0 3.0 0.2 0.3])
     (primitive-kind rnn-bwd-desc) => :rnn
     (execute! s reorder-weights-fb (fwd-args weights-mem bwd-weights-mem)) => s
     (execute! s reorder-weights-iter-fb (fwd-args weights-iter-mem bwd-weights-iter-mem)) => s
     (execute! s reorder-diff-weights-unpack (fwd-args diff-weights-packed-mem diff-weights-mem))
     (execute! s reorder-diff-weights-iter-unpack (fwd-args diff-weights-iter-packed-mem diff-weights-iter-mem))
     (execute! s rnn-bwd rnn-bwd-args) => s
     (execute! s reorder-weights-bf (fwd-args bwd-weights-mem weights-mem)) => s
     (execute! s reorder-weights-iter-bf (fwd-args bwd-weights-iter-mem weights-iter-mem)) => s
     (execute! s reorder-diff-weights-pack (fwd-args diff-weights-mem diff-weights-packed-mem))
     (execute! s reorder-diff-weights-iter-pack (fwd-args diff-weights-iter-mem diff-weights-iter-packed-mem))
     (seq weights-vec) => (map float [0.1 0.2 0.3 0.4 0.3 0.4 0.5 0.6])
     (seq dst-vec) => [2.570000171661377 3.940000057220459 850.6968994140625 1054.8890380859375]
     (seq src-vec) => [-33.62968063354492 -66.7193832397461 0.0059999702498316765 -0.1700000762939453]
     (seq src-iter-vec) => [-33629.6796875 -66719.3828125 -0.03522000089287758 -0.060019999742507935]
     (seq diff-weights-vec)
     => (map float [10.535527229309082 -341.3085632324219 15.953291893005371 -511.8628845214844
                    2825.152587890625 -3822.6806640625 4085.8203125 -5528.6044921875])
     (seq diff-weights-iter-vec)
     => (map float [97.4520034790039 201.3159942626953 295.8139953613281 402.1619873046875
                    8.748000144958496 -11.802000045776367 13.425999641418457 -18.083999633789062])
     (seq bias-vec) => [3.879763603210449 -169.20828247070312 5.441999435424805 -4.881999969482422])))

(facts
 "Vanilla RNN training no-iter. Demonstrates possible bug in DNNL RNN when src-iter is nil, which work the same as if it was 0, but doesn't."
 (let [T 2
       N 1
       C 2
       G 1
       L 2
       D 1
       src-dim [T N C]
       src-iter-dim [L D N C]
       weights-dim [L D C G C]
       bias-dim [L D G C]]
   (with-release
     [eng (engine)
      s (stream eng)
      src-desc (memory-desc src-dim :float :tnc)
      src-iter-desc (memory-desc src-iter-dim :float :ldnc)
      weights-desc (memory-desc weights-dim :float :ldigo)
      bias-desc (memory-desc bias-dim :float :ldgo)
      dst-desc (memory-desc src-dim :float :tnc)
      dst-iter-desc (memory-desc src-iter-dim :float :ldnc)
      rnn-fwd-desc (vanilla-rnn-fwd-desc :training :relu :unidirectional src-desc src-iter-desc
                                         weights-desc weights-desc bias-desc dst-desc dst-iter-desc)
      rnn-fwd-pd (primitive-desc eng rnn-fwd-desc)
      rnn-no-iter-fwd-desc (vanilla-rnn-fwd-desc :training :relu :unidirectional src-desc src-iter-desc
                                                 weights-desc weights-desc bias-desc dst-desc nil)
      rnn-no-iter-fwd-pd (primitive-desc eng rnn-no-iter-fwd-desc)

      src-vec (fv [2 3 0.2 0.3])
      src-mem (memory eng (arg-md rnn-fwd-pd :src) (buffer src-vec))
      diff-src-vec (fv (apply * src-dim))
      diff-src-mem (memory eng (arg-md rnn-fwd-pd :diff-src) (buffer diff-src-vec))
      src-iter-vec (fv [0 0 0 0])
      src-iter-mem (memory eng (arg-md rnn-fwd-pd :src-iter) (buffer src-iter-vec))
      diff-src-iter-vec (fv [0 0 0 0])
      diff-src-iter-mem (memory eng (arg-md rnn-fwd-pd :diff-src-iter) (buffer diff-src-iter-vec))
      weights-vec (fv [0.1 0.2 0.3 0.4 0.3 0.4 0.5 0.6])
      weights-mem (memory eng (arg-md rnn-fwd-pd :weights) (buffer weights-vec))
      weights-iter-vec (fv [100 200 300 400 0.01 0.02 0.03 0.04])
      weights-iter-mem (memory eng (arg-md rnn-fwd-pd :weights-iter) (buffer weights-iter-vec))
      bias-vec (fv [0.3 0.7 1 2])
      bias-mem (memory eng bias-desc (buffer bias-vec))
      diff-bias-vec (fv [0.3 0.7 1 2])
      diff-bias-mem (memory eng bias-desc (buffer diff-bias-vec))
      dst-vec (fv (apply * src-dim))
      dst-mem (memory eng (arg-md rnn-fwd-pd :dst-iter) (buffer dst-vec))
      dst-iter-vec (fv (apply * src-dim))
      dst-iter-mem (memory eng (arg-md rnn-fwd-pd :dst-iter) (buffer dst-iter-vec))
      workspace-mem (memory eng (arg-md rnn-fwd-pd :workspace))
      rnn-fwd (primitive rnn-fwd-pd)
      rnn-fwd-args (args {:src-layer src-mem
                          :src-iter src-iter-mem
                          :weights-layer weights-mem
                          :weights-iter weights-iter-mem
                          :bias bias-mem
                          :dst-layer dst-mem
                          :dst-iter dst-iter-mem
                          :workspace workspace-mem})
      rnn-no-iter-fwd (primitive rnn-no-iter-fwd-pd)
      rnn-no-iter-fwd-args (args {:src-layer src-mem
                                  :src-iter src-iter-mem
                                  :weights-layer weights-mem
                                  :weights-iter weights-iter-mem
                                  :bias bias-mem
                                  :dst-layer dst-mem
                                  :dst-iter dst-iter-mem
                                  :workspace workspace-mem})
      bwd-weights-desc (memory-desc weights-dim :float :any)
      rnn-bwd-desc (vanilla-rnn-bwd-desc :relu :unidirectional src-desc src-iter-desc
                                         bwd-weights-desc bwd-weights-desc bias-desc
                                         dst-desc dst-iter-desc
                                         src-desc src-iter-desc
                                         bwd-weights-desc bwd-weights-desc bias-desc
                                         dst-desc dst-iter-desc)
      rnn-bwd-pd (primitive-desc eng rnn-bwd-desc rnn-fwd-pd)
      rnn-no-iter-bwd-desc (vanilla-rnn-bwd-desc :relu :unidirectional src-desc src-iter-desc
                                                 bwd-weights-desc bwd-weights-desc bias-desc
                                                 dst-desc nil
                                                 src-desc src-iter-desc
                                                 bwd-weights-desc bwd-weights-desc bias-desc
                                                 dst-desc nil)
      rnn-no-iter-bwd-pd (primitive-desc eng rnn-no-iter-bwd-desc rnn-no-iter-fwd-pd)
      bwd-weights-mem (memory eng (arg-md rnn-bwd-pd :weights))
      reorder-weights-fb-pd (reorder eng weights-mem bwd-weights-mem)
      reorder-weights-bf-pd (reorder eng bwd-weights-mem weights-mem)
      reorder-weights-fb (primitive reorder-weights-fb-pd)
      reorder-weights-bf (primitive reorder-weights-bf-pd)
      bwd-weights-iter-mem (memory eng (arg-md rnn-bwd-pd :weights-iter))
      reorder-weights-iter-fb-pd (reorder eng weights-iter-mem bwd-weights-iter-mem)
      reorder-weights-iter-bf-pd (reorder eng bwd-weights-iter-mem weights-iter-mem)
      reorder-weights-iter-fb (primitive reorder-weights-iter-fb-pd)
      reorder-weights-iter-bf (primitive reorder-weights-iter-bf-pd)
      diff-weights-vec (fv [0.1 0.2 0.3 0.4 0.3 0.4 0.5 0.6])
      diff-weights-packed-mem (memory eng weights-desc (buffer diff-weights-vec))
      diff-weights-mem (memory eng (arg-md rnn-bwd-pd :diff-weights))
      reorder-diff-weights-pack-pd (reorder eng diff-weights-mem diff-weights-packed-mem)
      reorder-diff-weights-unpack-pd (reorder eng diff-weights-packed-mem diff-weights-mem)
      reorder-diff-weights-pack (primitive reorder-diff-weights-pack-pd)
      reorder-diff-weights-unpack (primitive reorder-diff-weights-unpack-pd)
      diff-weights-iter-vec (fv [100 200 300 400 0.01 0.02 0.03 0.04])
      diff-weights-iter-packed-mem (memory eng weights-desc (buffer diff-weights-iter-vec))
      diff-weights-iter-mem (memory eng (arg-md rnn-bwd-pd :diff-weights-iter))
      reorder-diff-weights-iter-pack-pd (reorder eng diff-weights-iter-mem diff-weights-iter-packed-mem)
      reorder-diff-weights-iter-unpack-pd (reorder eng diff-weights-iter-packed-mem diff-weights-iter-mem)
      reorder-diff-weights-iter-pack (primitive reorder-diff-weights-iter-pack-pd)
      reorder-diff-weights-iter-unpack (primitive reorder-diff-weights-iter-unpack-pd)
      diff-dst-vec (fv [1.1 -2.2 3.3 -4.4])
      diff-dst-mem (memory eng (arg-md rnn-bwd-pd :diff-dst) (buffer diff-dst-vec))
      diff-dst-iter-vec (fv [0 0 0 0])
      diff-dst-iter-mem (memory eng (arg-md rnn-fwd-pd :diff-dst-iter) (buffer diff-dst-iter-vec))
      rnn-bwd-args (args {:src-layer src-mem
                          :src-iter src-iter-mem
                          :weights-layer bwd-weights-mem
                          :weights-iter bwd-weights-iter-mem
                          :bias bias-mem
                          :dst-layer dst-mem
                          :dst-iter dst-iter-mem
                          :workspace workspace-mem
                          :diff-src-layer diff-src-mem
                          :diff-src-iter diff-src-iter-mem
                          :diff-weights-layer diff-weights-mem
                          :diff-weights-iter diff-weights-iter-mem
                          :diff-bias diff-bias-mem
                          :diff-dst-layer diff-dst-mem
                          :diff-dst-iter diff-dst-iter-mem})
      rnn-bwd (primitive rnn-bwd-pd)
      rnn-no-iter-bwd-args (args {:src-layer src-mem
                                  :src-iter src-iter-mem
                                  :weights-layer bwd-weights-mem
                                  :weights-iter bwd-weights-iter-mem
                                  :bias bias-mem
                                  :dst-layer dst-mem
                                  :dst-iter dst-iter-mem
                                  :workspace workspace-mem
                                  :diff-src-layer diff-src-mem
                                  :diff-src-iter diff-src-iter-mem
                                  :diff-weights-layer diff-weights-mem
                                  :diff-weights-iter diff-weights-iter-mem
                                  :diff-bias diff-bias-mem
                                  :diff-dst-layer diff-dst-mem
                                  :diff-dst-iter diff-dst-iter-mem})
      rnn-no-iter-bwd (primitive rnn-no-iter-bwd-pd)]

     (primitive-kind rnn-fwd-desc) => :rnn
     (execute! s rnn-fwd rnn-fwd-args) => s
     (seq dst-vec) => [2.570000171661377 3.940000057220459 850.6968994140625 1054.8890380859375]
     (seq dst-iter-vec) => [830.4099731445312 1200.8599853515625 850.6968994140625 1054.8890380859375]
     (seq src-vec) => (map float [2.0 3.0 0.2 0.3])
     (primitive-kind rnn-bwd-desc) => :rnn
     (execute! s reorder-weights-fb (fwd-args weights-mem bwd-weights-mem)) => s
     (execute! s reorder-weights-iter-fb (fwd-args weights-iter-mem bwd-weights-iter-mem)) => s
     (execute! s reorder-diff-weights-unpack (fwd-args diff-weights-packed-mem diff-weights-mem))
     (execute! s reorder-diff-weights-iter-unpack (fwd-args diff-weights-iter-packed-mem diff-weights-iter-mem))
     (execute! s rnn-bwd rnn-bwd-args) => s
     (execute! s reorder-weights-bf (fwd-args bwd-weights-mem weights-mem)) => s
     (execute! s reorder-weights-iter-bf (fwd-args bwd-weights-iter-mem weights-iter-mem)) => s
     (execute! s reorder-diff-weights-pack (fwd-args diff-weights-mem diff-weights-packed-mem))
     (execute! s reorder-diff-weights-iter-pack (fwd-args diff-weights-iter-mem diff-weights-iter-packed-mem))
     (seq weights-vec) => (map float [0.1 0.2 0.3 0.4 0.3 0.4 0.5 0.6])
     (seq dst-vec) => [2.570000171661377 3.940000057220459 850.6968994140625 1054.8890380859375]
     (seq dst-iter-vec) => [830.4099731445312 1200.8599853515625 850.6968994140625 1054.8890380859375]
     (seq diff-dst-vec) => [1.100000023841858 -2.200000047683716 3.299999952316284 -4.400000095367432]
     (seq diff-dst-iter-vec) => [0.0 0.0 0.0 0.0]
     (seq src-vec) => [2.0 3.0 0.20000000298023224 0.30000001192092896]
     (seq diff-src-vec) => [-153.12847900390625 -333.81671142578125 -0.27500003576278687 -0.627000093460083]
     (seq src-iter-vec) => [0.0 0.0 0.0 0.0]
     (seq diff-src-iter-vec) => [-153128.484375 -333816.6875 -0.035089995712041855 -0.05972999334335327]
     (seq diff-weights-vec)
     => (map float [-551.2486572265625 -1255.685546875 -826.7230224609375 -1883.42822265625
                    2742.11572265625 -3656.591796875 3965.741455078125 -5288.42138671875])
     (seq diff-weights-iter-vec)
     => (map float [98.9219970703125 198.61399841308594 298.22900390625 397.7229919433594
                    8.49100112915039 -11.288000106811523 13.031999588012695 -17.29599952697754])
     (seq bias-vec) => [0.30000001192092896 0.699999988079071 1.0 2.0]
     (seq diff-bias-vec) => [-276.06732177734375 -628.1337280273438 5.345000267028809 -4.677000045776367]

     (transfer! [2 3 0.2 0.3] src-vec)
     (transfer! [0 0 0 0] src-iter-vec)
     (transfer! [0 0 0 0] dst-iter-vec)
     (transfer! [0.1 0.2 0.3 0.4 0.3 0.4 0.5 0.6] weights-vec)
     (transfer! [100 200 300 400 0.01 0.02 0.03 0.04] weights-iter-vec)
     (transfer! [0.3 0.7 1 2] bias-vec)
     (transfer! [0 0 0 0] diff-bias-vec)
     (transfer! [0.1 0.2 0.3 0.4 0.3 0.4 0.5 0.6] diff-weights-vec)
     (transfer! [100 200 300 400 0.01 0.02 0.03 0.04] diff-weights-iter-vec)
     (execute! s rnn-no-iter-fwd rnn-no-iter-fwd-args) => s
     (seq dst-vec) => [2.570000171661377 3.940000057220459 850.6968994140625 1054.8890380859375]
     (execute! s reorder-weights-fb (fwd-args weights-mem bwd-weights-mem)) => s
     (execute! s reorder-weights-iter-fb (fwd-args weights-iter-mem bwd-weights-iter-mem)) => s
     (execute! s reorder-diff-weights-unpack (fwd-args diff-weights-packed-mem diff-weights-mem))
     (execute! s reorder-diff-weights-iter-unpack (fwd-args diff-weights-iter-packed-mem diff-weights-iter-mem))
     (execute! s rnn-no-iter-bwd rnn-no-iter-bwd-args)
     (execute! s reorder-weights-bf (fwd-args bwd-weights-mem weights-mem)) => s
     (execute! s reorder-weights-iter-bf (fwd-args bwd-weights-iter-mem weights-iter-mem)) => s
     (execute! s reorder-diff-weights-pack (fwd-args diff-weights-mem diff-weights-packed-mem))
     (execute! s reorder-diff-weights-iter-pack (fwd-args diff-weights-iter-mem diff-weights-iter-packed-mem))
     (seq weights-vec) => (map float [0.1 0.2 0.3 0.4 0.3 0.4 0.5 0.6])
     (seq dst-vec) => [2.570000171661377 3.940000057220459 850.6968994140625 1054.8890380859375]
     (seq src-vec) => [2.0 3.0 0.20000000298023224 0.30000001192092896]
     (seq diff-src-vec) => [-153.12847900390625 -333.81671142578125 -0.27500003576278687 -0.627000093460083]
     (seq src-iter-vec) => [0.0 0.0 0.0 0.0]
     (seq diff-src-iter-vec) => [-153128.484375 -333816.6875 -0.035089995712041855 -0.05972999334335327]
     (seq diff-weights-vec)
     => (just [(roughly -551) (roughly -1255) (roughly -826) (roughly -1883)
               (roughly 2742) (roughly -3656) (roughly 3965) (roughly -5288)])
     (seq diff-weights-iter-vec)
     => (map float [98.9219970703125 198.61399841308594 298.22900390625 397.7229919433594
                    8.49100112915039 -11.288000106811523 13.031999588012695 -17.29599952697754])
     (seq bias-vec) => [0.30000001192092896 0.699999988079071 1.0 2.0]
     (seq diff-bias-vec) => [-276.3673095703125 -628.833740234375 4.345000267028809 -6.677000045776367])))

(facts
 "LSTM forward."
 (let [T 2
       N 1
       C 2
       G 4
       L 2
       D 1
       src-dim [T N C]
       src-iter-dim [L D N C]
       weights-dim [L D C G C]
       bias-dim [L D G C]]
   (with-release
     [eng (engine)
      s (stream eng)
      src-desc (memory-desc src-dim :float :tnc)
      src-iter-desc (memory-desc src-iter-dim :float :ldnc)
      src-iter-c-desc (memory-desc src-iter-dim :float :ldnc)
      weights-desc (memory-desc weights-dim :float :ldigo)
      bias-desc (memory-desc bias-dim :float :ldgo)
      dst-desc (memory-desc src-dim :float :tnc)
      dst-iter-desc (memory-desc src-iter-dim :float :ldnc)
      dst-iter-c-desc (memory-desc src-iter-dim :float :ldnc)
      lstm-desc (lstm-fwd-desc :inference :unidirectional
                               src-desc src-iter-desc src-iter-c-desc
                               weights-desc weights-desc bias-desc
                               dst-desc dst-iter-desc dst-iter-c-desc)
      lstm-no-iter-desc (lstm-fwd-desc :inference :unidirectional
                                       src-desc nil nil weights-desc weights-desc bias-desc
                                       dst-desc dst-iter-desc dst-iter-c-desc)
      lstm-no-iter-pd (primitive-desc eng lstm-no-iter-desc)
      lstm-pd (primitive-desc eng lstm-desc)
      src-vec (fv [2 3 0.2 0.3])
      src-mem (memory eng (arg-md lstm-pd :src) (buffer src-vec))
      src-iter-vec (fv (apply * src-iter-dim))
      src-iter-mem (memory eng (arg-md lstm-pd :src-iter) (buffer src-iter-vec))
      src-iter-c-vec (fv (apply * src-iter-dim))
      src-iter-c-mem (memory eng (arg-md lstm-pd :src-iter-c) (buffer src-iter-c-vec))
      weights-vec (fv [0.1 0.2 0.3 0.4 0.3 0.4 0.5 0.6
                       0.1 0.2 0.3 0.4 0.3 0.4 0.5 0.6
                       0.1 0.2 0.3 0.4 0.3 0.4 0.5 0.6
                       0.1 0.2 0.3 0.4 0.3 0.4 0.5 0.6])
      weights-mem (memory eng (arg-md lstm-pd :weights) (buffer weights-vec))
      weights-iter-vec (fv [100 200 300 400 0.01 0.02 0.03 0.04
                            100 200 300 400 0.01 0.02 0.03 0.04
                            100 200 300 400 0.01 0.02 0.03 0.04
                            100 200 300 400 0.01 0.02 0.03 0.04])
      weights-iter-mem (memory eng (arg-md lstm-pd :weights-iter) (buffer weights-iter-vec))
      bias-vec (fv [0.3 0.7 1 2
                    0.3 0.7 1 2
                    0.3 0.7 1 2
                    0.3 0.7 1 2])
      bias-mem (memory eng bias-desc (buffer bias-vec))
      dst-vec (fv (apply * src-dim))
      dst-mem (memory eng (arg-md lstm-pd :dst) (buffer dst-vec))
      dst-iter-vec (fv (apply * src-dim))
      dst-iter-mem (memory eng (arg-md lstm-pd :dst-iter) (buffer dst-iter-vec))
      dst-iter-c-vec (fv (apply * src-iter-dim))
      dst-iter-c-mem (memory eng (arg-md lstm-pd :dst-iter-c) (buffer dst-iter-vec))
      workspace-mem (memory eng (arg-md lstm-pd :workspace))
      lstm (primitive lstm-pd)
      lstm-args (args {:src-layer src-mem
                       :src-iter src-iter-mem
                       :src-iter-c src-iter-c-mem
                       :weights-layer weights-mem
                       :weights-iter weights-iter-mem
                       :bias bias-mem
                       :dst-layer dst-mem
                       :dst-iter dst-iter-mem
                       :dst-iter-c dst-iter-c-mem
                       :workspace workspace-mem})
      lstm-no-iter (primitive lstm-no-iter-pd)
      lstm-no-iter-args (args {:src-layer src-mem
                               :weights-layer weights-mem
                               :weights-iter weights-iter-mem
                               :bias bias-mem
                               :dst-layer dst-mem
                               :dst-iter dst-iter-mem
                               :dst-iter-c dst-iter-c-mem
                               :workspace workspace-mem})]
     (primitive-kind lstm-desc) => :rnn
     (execute! s lstm lstm-args) => s
     (seq src-iter-vec) => [0.0 0.0 0.0 0.0]
     (seq dst-vec) => [0.28369858860969543 0.5042773485183716 0.6443434953689575 0.8513875007629395]
     (seq dst-iter-vec) => [0.623127281665802 0.8365733027458191 0.6443434953689575 0.8513875007629395]
     (entry! dst-vec 0)
     (primitive-kind lstm-no-iter-desc) => :rnn
     (arg-md lstm-no-iter-pd :src-iter) => nil
     (zero-desc? (arg-md lstm-no-iter-pd :src-iter)) => true
     (execute! s lstm-no-iter lstm-no-iter-args) => s
     (seq dst-vec) => [0.28369858860969543 0.5042773485183716 0.6443434953689575 0.8513875007629395])))

(facts
 "LSTM training."
 (let [T 2
       N 1
       C 2
       G 4
       L 2
       D 1
       src-dim [T N C]
       src-iter-dim [L D N C]
       weights-dim [L D C G C]
       bias-dim [L D G C]]
   (with-release
     [eng (engine)
      s (stream eng)
      src-desc (memory-desc src-dim :float :tnc)
      src-iter-desc (memory-desc src-iter-dim :float :ldnc)
      weights-desc (memory-desc weights-dim :float :ldigo)
      bias-desc (memory-desc bias-dim :float :ldgo)
      dst-desc (memory-desc src-dim :float :tnc)
      dst-iter-desc (memory-desc src-iter-dim :float :ldnc)
      dst-iter-c-desc (memory-desc src-iter-dim :float :ldnc)
      lstm-fwd-desc (lstm-fwd-desc :training  :unidirectional
                                   src-desc src-iter-desc src-iter-desc
                                   weights-desc weights-desc bias-desc
                                   dst-desc dst-iter-desc dst-iter-desc)
      lstm-fwd-pd (primitive-desc eng lstm-fwd-desc)
      src-vec (fv [2 3 0.2 0.3])
      src-mem (memory eng (arg-md lstm-fwd-pd :src) (buffer src-vec))
      src-iter-vec (fv (apply * src-iter-dim))
      src-iter-mem (memory eng (arg-md lstm-fwd-pd :src-iter) (buffer src-iter-vec))
      src-iter-c-vec (fv (apply * src-iter-dim))
      src-iter-c-mem (memory eng (arg-md lstm-fwd-pd :src-iter-c) (buffer src-iter-c-vec))
      weights-vec (fv [0.1 0.2 0.3 0.4 0.3 0.4 0.5 0.6
                       0.1 0.2 0.3 0.4 0.3 0.4 0.5 0.6
                       0.1 0.2 0.3 0.4 0.3 0.4 0.5 0.6
                       0.1 0.2 0.3 0.4 0.3 0.4 0.5 0.6])
      weights-mem (memory eng (arg-md lstm-fwd-pd :weights) (buffer weights-vec))
      weights-iter-vec (fv [100 200 300 400 0.01 0.02 0.03 0.04
                            100 200 300 400 0.01 0.02 0.03 0.04
                            100 200 300 400 0.01 0.02 0.03 0.04
                            100 200 300 400 0.01 0.02 0.03 0.04])
      weights-iter-mem (memory eng (arg-md lstm-fwd-pd :weights-iter) (buffer weights-iter-vec))
      bias-vec (fv [0.3 0.7 1 2
                    0.3 0.7 1 2
                    0.3 0.7 1 2
                    0.3 0.7 1 2])
      bias-mem (memory eng bias-desc (buffer bias-vec))
      dst-vec (fv (apply * src-dim))
      dst-mem (memory eng (arg-md lstm-fwd-pd :dst) (buffer dst-vec))
      dst-iter-vec (fv (apply * src-dim))
      dst-iter-mem (memory eng (arg-md lstm-fwd-pd :dst-iter) (buffer dst-iter-vec))
      dst-iter-c-vec (fv (apply * src-iter-dim))
      dst-iter-c-mem (memory eng (arg-md lstm-fwd-pd :dst-iter-c) (buffer dst-iter-vec))
      workspace-mem (memory eng (arg-md lstm-fwd-pd :workspace))
      lstm-fwd (primitive lstm-fwd-pd)
      lstm-fwd-args (args {:src-layer src-mem
                           :src-iter src-iter-mem
                           :src-iter-c src-iter-c-mem
                           :weights-layer weights-mem
                           :weights-iter weights-iter-mem
                           :bias bias-mem
                           :dst-layer dst-mem
                           :dst-iter dst-iter-mem
                           :dst-iter-c dst-iter-c-mem
                           :workspace workspace-mem})
      bwd-weights-desc (memory-desc weights-dim :float :any)
      lstm-bwd-desc (lstm-bwd-desc :unidirectional src-desc src-iter-desc src-iter-desc
                                   bwd-weights-desc bwd-weights-desc bias-desc
                                   dst-desc dst-iter-desc dst-iter-desc
                                   src-desc src-iter-desc src-iter-desc
                                   bwd-weights-desc bwd-weights-desc bias-desc
                                   dst-desc dst-iter-desc dst-iter-desc)
      lstm-bwd-pd (primitive-desc eng lstm-bwd-desc lstm-fwd-pd)
      bwd-weights-mem (memory eng (arg-md lstm-bwd-pd :weights))
      reorder-weights-fb-pd (reorder eng weights-mem bwd-weights-mem)
      reorder-weights-bf-pd (reorder eng bwd-weights-mem weights-mem)
      reorder-weights-fb (primitive reorder-weights-fb-pd)
      reorder-weights-bf (primitive reorder-weights-bf-pd)
      bwd-weights-iter-mem (memory eng (arg-md lstm-bwd-pd :weights-iter))
      reorder-weights-iter-fb-pd (reorder eng weights-iter-mem bwd-weights-iter-mem)
      reorder-weights-iter-bf-pd (reorder eng bwd-weights-iter-mem weights-iter-mem)
      reorder-weights-iter-fb (primitive reorder-weights-iter-fb-pd)
      reorder-weights-iter-bf (primitive reorder-weights-iter-bf-pd)
      diff-weights-vec (fv [0.1 0.2 0.3 0.4 0.3 0.4 0.5 0.6
                            0.1 0.2 0.3 0.4 0.3 0.4 0.5 0.6
                            0.1 0.2 0.3 0.4 0.3 0.4 0.5 0.6
                            0.1 0.2 0.3 0.4 0.3 0.4 0.5 0.6])
      diff-weights-packed-mem (memory eng weights-desc (buffer diff-weights-vec))
      diff-weights-mem (memory eng (arg-md lstm-bwd-pd :diff-weights))
      reorder-diff-weights-pack-pd (reorder eng diff-weights-mem diff-weights-packed-mem)
      reorder-diff-weights-unpack-pd (reorder eng diff-weights-packed-mem diff-weights-mem)
      reorder-diff-weights-pack (primitive reorder-diff-weights-pack-pd)
      reorder-diff-weights-unpack (primitive reorder-diff-weights-unpack-pd)
      diff-weights-iter-vec (fv [100 200 300 400 0.01 0.02 0.03 0.04
                                 100 200 300 400 0.01 0.02 0.03 0.04
                                 100 200 300 400 0.01 0.02 0.03 0.04
                                 100 200 300 400 0.01 0.02 0.03 0.04])
      diff-weights-iter-packed-mem (memory eng weights-desc (buffer diff-weights-iter-vec))
      diff-weights-iter-mem (memory eng (arg-md lstm-bwd-pd :diff-weights-iter))
      reorder-diff-weights-iter-pack-pd (reorder eng diff-weights-iter-mem diff-weights-iter-packed-mem)
      reorder-diff-weights-iter-unpack-pd (reorder eng diff-weights-iter-packed-mem diff-weights-iter-mem)
      reorder-diff-weights-iter-pack (primitive reorder-diff-weights-iter-pack-pd)
      reorder-diff-weights-iter-unpack (primitive reorder-diff-weights-iter-unpack-pd)
      diff-dst-vec (fv [1.1 -2.2 3.3 -4.4])
      diff-dst-mem (memory eng (arg-md lstm-bwd-pd :diff-dst) (buffer diff-dst-vec))
      diff-dst-iter-vec (fv [-1 2 0.1 -0.2])
      diff-dst-iter-mem (memory eng (arg-md lstm-fwd-pd :diff-dst-iter) (buffer diff-dst-iter-vec))
      diff-dst-iter-c-vec (fv [-0.1 0.2 0.01 -0.02])
      diff-dst-iter-c-mem (memory eng (arg-md lstm-fwd-pd :diff-dst-iter-c) (buffer diff-dst-iter-c-vec))
      lstm-bwd-args (args {:src-layer src-mem
                           :src-iter src-iter-mem
                           :src-iter-c src-iter-c-mem
                           :weights-layer bwd-weights-mem
                           :weights-iter bwd-weights-iter-mem
                           :bias bias-mem
                           :dst-layer dst-mem
                           :dst-iter dst-iter-mem
                           :dst-iter-c dst-iter-c-mem
                           :workspace workspace-mem
                           :diff-src-layer src-mem
                           :diff-src-iter src-iter-mem
                           :diff-src-iter-c src-iter-c-mem
                           :diff-weights-layer diff-weights-mem
                           :diff-weights-iter diff-weights-iter-mem
                           :diff-bias bias-mem
                           :diff-dst-layer diff-dst-mem
                           :diff-dst-iter diff-dst-iter-mem
                           :diff-dst-iter-c diff-dst-iter-c-mem})
      lstm-bwd (primitive lstm-bwd-pd)]

     (primitive-kind lstm-fwd-desc) => :rnn
     (execute! s lstm-fwd lstm-fwd-args) => s
     (seq dst-vec) => [0.28369858860969543 0.5042773485183716 0.6443434953689575 0.8513875007629395]
     (seq src-vec) => (map float [2.0 3.0 0.2 0.3])
     (primitive-kind lstm-bwd-desc) => :rnn
     (execute! s reorder-weights-fb (fwd-args weights-mem bwd-weights-mem)) => s
     (execute! s reorder-weights-iter-fb (fwd-args weights-iter-mem bwd-weights-iter-mem)) => s
     (execute! s reorder-diff-weights-unpack (fwd-args diff-weights-packed-mem diff-weights-mem))
     (execute! s reorder-diff-weights-iter-unpack (fwd-args diff-weights-iter-packed-mem diff-weights-iter-mem))
     (execute! s lstm-bwd lstm-bwd-args) => s
     (execute! s reorder-weights-bf (fwd-args bwd-weights-mem weights-mem)) => s
     (execute! s reorder-weights-iter-bf (fwd-args bwd-weights-iter-mem weights-iter-mem)) => s
     (execute! s reorder-diff-weights-pack (fwd-args diff-weights-mem diff-weights-packed-mem))
     (execute! s reorder-diff-weights-iter-pack (fwd-args diff-weights-iter-mem diff-weights-iter-packed-mem))
     (map float (seq weights-vec)) => (map float [0.1 0.2 0.3 0.4 0.3 0.4 0.5 0.6
                                                  0.1 0.2 0.3 0.4 0.3 0.4 0.5 0.6
                                                  0.1 0.2 0.3 0.4 0.3 0.4 0.5 0.6
                                                  0.1 0.2 0.3 0.4 0.3 0.4 0.5 0.6])
     (seq dst-vec) => [0.28369858860969543 0.5042773485183716 0.6443434953689575 0.8513875007629395]
     (seq src-vec) => [0.011007515713572502 0.011007515713572502 0.1259019672870636 0.1259019672870636]
     (seq src-iter-vec) => [18.216032028198242 18.216032028198242 -87.13029479980469 -87.13029479980469]
     (seq diff-weights-vec)
     => (map float [-0.15334531664848328 0.5088344812393188 0.30000001192092896 0.4000000059604645
                    0.11885079741477966 0.5528948307037354 0.48091015219688416 0.6224913597106934
                    -0.2800179719924927 0.6632516980171204 0.30000001192092896 0.4000000059604645
                    0.028276175260543823 0.6293423175811768 0.47136521339416504 0.6337370276451111
                    0.31738361716270447 -0.15135137736797333 0.30000001192092896 0.4000000059604645
                    1.6538493633270264 -0.43319636583328247 0.680149257183075 0.46523377299308777
                    0.36541545391082764 -0.22898398339748383 0.30000001192092896 0.4000000059604645
                    2.0426719188690186 -0.6620708703994751 0.7383583784103394 0.4236163794994354])
     (seq diff-weights-iter-vec)
     => (map float [100.0 200.0 300.0 400.0 -0.24559931457042694 0.34523943066596985
                    -0.014603456482291222 0.1050318107008934 100.0 200.0 300.0 400.0
                    -0.3020751178264618 0.41710248589515686 -0.024458782747387886
                    0.11940087378025055 100.0 200.0 300.0 400.0 0.34582412242889404
                    -0.14766573905944824 0.09891565144062042 -0.004336059093475342 100.0 200.0
                    300.0 400.0 0.6069310307502747 -0.2780276834964752 0.15249831974506378
                    -0.03880783170461655]))))


(facts
 "GRU forward."
 (let [T 2
       N 1
       C 2
       G 3
       L 2
       D 1
       src-dim [T N C]
       src-iter-dim [L D N C]
       weights-dim [L D C G C]
       bias-dim [L D G C]]
   (with-release
     [eng (engine)
      s (stream eng)
      src-desc (memory-desc src-dim :float :tnc)
      src-iter-desc (memory-desc src-iter-dim :float :ldnc)
      weights-desc (memory-desc weights-dim :float :ldigo)
      bias-desc (memory-desc bias-dim :float :ldgo)
      dst-desc (memory-desc src-dim :float :tnc)
      dst-iter-desc (memory-desc src-iter-dim :float :ldnc)
      gru-desc (gru-fwd-desc :inference :unidirectional
                             src-desc src-iter-desc weights-desc weights-desc bias-desc
                             dst-desc dst-iter-desc)
      gru-no-iter-desc (gru-fwd-desc :inference :unidirectional
                                     src-desc nil weights-desc weights-desc bias-desc
                                     dst-desc dst-iter-desc)
      gru-no-iter-pd (primitive-desc eng gru-no-iter-desc)
      gru-pd (primitive-desc eng gru-desc)
      src-vec (fv [2 3 0.2 0.3])
      src-mem (memory eng (arg-md gru-pd :src) (buffer src-vec))
      src-iter-vec (fv (apply * src-iter-dim))
      src-iter-mem (memory eng (arg-md gru-pd :src-iter) (buffer src-iter-vec))
      weights-vec (fv [0.1 0.2 0.1 0.2 0.1 0.2
                       0.3 0.4 0.3 0.4 0.3 0.4
                       0.3 0.4 0.3 0.4 0.3 0.4
                       0.5 0.6 0.5 0.6 0.5 0.6])
      weights-mem (memory eng (arg-md gru-pd :weights) (buffer weights-vec))
      weights-iter-vec (fv [100 200 100 200 100 200
                            300 400 300 400 300 400
                            0.01 0.02 0.01 0.02 0.01 0.02
                            0.03 0.04 0.03 0.04 0.03 0.04])
      weights-iter-mem (memory eng (arg-md gru-pd :weights-iter) (buffer weights-iter-vec))
      bias-vec (fv [0.3 0.7 0.3 0.7 0.3 0.7
                    1 2 1 2 1 2])
      bias-mem (memory eng bias-desc (buffer bias-vec))
      dst-vec (fv (apply * src-dim))
      dst-mem (memory eng (arg-md gru-pd :dst) (buffer dst-vec))
      dst-iter-vec (fv (apply * src-dim))
      dst-iter-mem (memory eng (arg-md gru-pd :dst-iter) (buffer dst-iter-vec))
      workspace-mem (memory eng (arg-md gru-pd :workspace))
      gru (primitive gru-pd)
      gru-args (args {:src-layer src-mem
                      :src-iter src-iter-mem
                      :weights-layer weights-mem
                      :weights-iter weights-iter-mem
                      :bias bias-mem
                      :dst-layer dst-mem
                      :dst-iter dst-iter-mem
                      :workspace workspace-mem})
      gru-no-iter (primitive gru-no-iter-pd)
      gru-no-iter-args (args {:src-layer src-mem
                              :weights-layer weights-mem
                              :weights-iter weights-iter-mem
                              :bias bias-mem
                              :dst-layer dst-mem
                              :dst-iter dst-iter-mem
                              :workspace workspace-mem})]
     (primitive-kind gru-desc) => :rnn
     (execute! s gru gru-args) => s
     (seq src-iter-vec) => [0.0 0.0 0.0 0.0]
     (seq dst-vec) => [0.20008479058742523 0.10380759835243225 0.3499049246311188 0.19589270651340485]
     (seq dst-iter-vec) => [0.17513683438301086 0.08930931240320206 0.3499049246311188 0.19589270651340485]
     (entry! dst-vec 0)
     (primitive-kind gru-no-iter-desc) => :rnn
     (arg-md gru-no-iter-pd :src-iter) => nil
     (zero-desc? (arg-md gru-no-iter-pd :src-iter)) => true
     (execute! s gru-no-iter gru-no-iter-args) => s
     (seq dst-vec) => [0.20008479058742523 0.10380759835243225 0.3499049246311188 0.19589270651340485])))

(facts
 "GRU training."
 (let [T 2
       N 1
       C 2
       G 3
       L 2
       D 1
       src-dim [T N C]
       src-iter-dim [L D N C]
       weights-dim [L D C G C]
       bias-dim [L D G C]]
   (with-release
     [eng (engine)
      s (stream eng)
      src-desc (memory-desc src-dim :float :tnc)
      src-iter-desc (memory-desc src-iter-dim :float :ldnc)
      weights-desc (memory-desc weights-dim :float :ldigo)
      bias-desc (memory-desc bias-dim :float :ldgo)
      dst-desc (memory-desc src-dim :float :tnc)
      dst-iter-desc (memory-desc src-iter-dim :float :ldnc)
      gru-fwd-desc (gru-fwd-desc :training :unidirectional src-desc src-iter-desc
                                 weights-desc weights-desc bias-desc dst-desc dst-iter-desc)
      gru-fwd-pd (primitive-desc eng gru-fwd-desc)
      src-vec (fv [2 3 0.2 0.3])
      src-mem (memory eng (arg-md gru-fwd-pd :src) (buffer src-vec))
      src-iter-vec (fv (apply * src-iter-dim))
      src-iter-mem (memory eng (arg-md gru-fwd-pd :src-iter) (buffer src-iter-vec))
      weights-vec (fv [0.111 0.112 0.121 0.122 0.131 0.132
                       0.211 0.212 0.221 0.222 0.231 0.232
                       0.311 0.312 0.321 0.322 0.331 0.332
                       0.411 0.412 0.421 0.422 0.431 0.432])
      weights-mem (memory eng (arg-md gru-fwd-pd :weights) (buffer weights-vec))
      weights-iter-vec (fv [100 200 100 200 100 200
                            300 400 300 400 300 400
                            0.01 0.02 0.01 0.02 0.01 0.02
                            0.03 0.04 0.03 0.04 0.03 0.04])
      weights-iter-mem (memory eng (arg-md gru-fwd-pd :weights-iter) (buffer weights-iter-vec))
      bias-vec (fv [0.3 0.7 0.3 0.7 0.3 0.7
                    1 2 1 2 1 2])
      bias-mem (memory eng bias-desc (buffer bias-vec))
      dst-vec (fv (apply * src-dim))
      dst-mem (memory eng (arg-md gru-fwd-pd :dst-iter) (buffer dst-vec))
      dst-iter-vec (fv (apply * src-dim))
      dst-iter-mem (memory eng (arg-md gru-fwd-pd :dst-iter) (buffer dst-iter-vec))
      workspace-mem (memory eng (arg-md gru-fwd-pd :workspace))
      gru-fwd (primitive gru-fwd-pd)
      gru-fwd-args (args {:src-layer src-mem
                          :src-iter src-iter-mem
                          :weights-layer weights-mem
                          :weights-iter weights-iter-mem
                          :bias bias-mem
                          :dst-layer dst-mem
                          :dst-iter dst-iter-mem
                          :workspace workspace-mem})
      bwd-weights-desc (memory-desc weights-dim :float :any)
      gru-bwd-desc (gru-bwd-desc :unidirectional src-desc src-iter-desc
                                 bwd-weights-desc bwd-weights-desc bias-desc
                                 dst-desc dst-iter-desc
                                 src-desc src-iter-desc
                                 bwd-weights-desc bwd-weights-desc bias-desc
                                 dst-desc dst-iter-desc)
      gru-bwd-pd (primitive-desc eng gru-bwd-desc gru-fwd-pd)
      bwd-weights-mem (memory eng (arg-md gru-bwd-pd :weights))
      reorder-weights-fb-pd (reorder eng weights-mem bwd-weights-mem)
      reorder-weights-bf-pd (reorder eng bwd-weights-mem weights-mem)
      reorder-weights-fb (primitive reorder-weights-fb-pd)
      reorder-weights-bf (primitive reorder-weights-bf-pd)
      bwd-weights-iter-mem (memory eng (arg-md gru-bwd-pd :weights-iter))
      reorder-weights-iter-fb-pd (reorder eng weights-iter-mem bwd-weights-iter-mem)
      reorder-weights-iter-bf-pd (reorder eng bwd-weights-iter-mem weights-iter-mem)
      reorder-weights-iter-fb (primitive reorder-weights-iter-fb-pd)
      reorder-weights-iter-bf (primitive reorder-weights-iter-bf-pd)
      diff-weights-vec (fv [0.111 0.112 0.121 0.122 0.131 0.132
                            0.211 0.212 0.221 0.222 0.231 0.232
                            0.311 0.312 0.321 0.322 0.331 0.332
                            0.411 0.412 0.421 0.422 0.431 0.432])
      diff-weights-packed-mem (memory eng weights-desc (buffer diff-weights-vec))
      diff-weights-mem (memory eng (arg-md gru-bwd-pd :diff-weights))
      reorder-diff-weights-pack-pd (reorder eng diff-weights-mem diff-weights-packed-mem)
      reorder-diff-weights-unpack-pd (reorder eng diff-weights-packed-mem diff-weights-mem)
      reorder-diff-weights-pack (primitive reorder-diff-weights-pack-pd)
      reorder-diff-weights-unpack (primitive reorder-diff-weights-unpack-pd)
      diff-weights-iter-vec (fv [100 200 100 200 100 200
                                 300 400 300 400 300 400
                                 0.01 0.02 0.01 0.02 0.01 0.02
                                 0.03 0.04 0.03 0.04 0.03 0.04])
      diff-weights-iter-packed-mem (memory eng weights-desc (buffer diff-weights-iter-vec))
      diff-weights-iter-mem (memory eng (arg-md gru-bwd-pd :diff-weights-iter))
      reorder-diff-weights-iter-pack-pd (reorder eng diff-weights-iter-mem diff-weights-iter-packed-mem)
      reorder-diff-weights-iter-unpack-pd (reorder eng diff-weights-iter-packed-mem diff-weights-iter-mem)
      reorder-diff-weights-iter-pack (primitive reorder-diff-weights-iter-pack-pd)
      reorder-diff-weights-iter-unpack (primitive reorder-diff-weights-iter-unpack-pd)
      diff-dst-vec (fv [1.1 -2.2 3.3 -4.4])
      diff-dst-mem (memory eng (arg-md gru-bwd-pd :diff-dst) (buffer diff-dst-vec))
      diff-dst-iter-vec (fv [-1 2 0.1 -0.2])
      diff-dst-iter-mem (memory eng (arg-md gru-fwd-pd :diff-dst-iter) (buffer diff-dst-iter-vec))
      gru-bwd-args (args {:src-layer src-mem
                          :src-iter src-iter-mem
                          :weights-layer bwd-weights-mem
                          :weights-iter bwd-weights-iter-mem
                          :bias bias-mem
                          :dst-layer dst-mem
                          :dst-iter dst-iter-mem
                          :workspace workspace-mem
                          :diff-src-layer src-mem
                          :diff-src-iter src-iter-mem
                          :diff-weights-layer diff-weights-mem
                          :diff-weights-iter diff-weights-iter-mem
                          :diff-bias bias-mem
                          :diff-dst-layer diff-dst-mem
                          :diff-dst-iter diff-dst-iter-mem})
      gru-bwd (primitive gru-bwd-pd)]

     (primitive-kind gru-fwd-desc) => :rnn
     (execute! s gru-fwd gru-fwd-args) => s
     (seq dst-vec) => [0.1986464262008667 0.10329369455575943 0.3485546410083771 0.19498808681964874]
     (seq dst-iter-vec) => [0.20356373488903046 0.161529079079628 0.3485546410083771 0.19498808681964874]
     (seq src-vec) => (map float [2.0 3.0 0.2 0.3])
     (primitive-kind gru-bwd-desc) => :rnn
     (execute! s reorder-weights-fb (fwd-args weights-mem bwd-weights-mem)) => s
     (execute! s reorder-weights-iter-fb (fwd-args weights-iter-mem bwd-weights-iter-mem)) => s
     (execute! s reorder-diff-weights-unpack (fwd-args diff-weights-packed-mem diff-weights-mem))
     (execute! s reorder-diff-weights-iter-unpack (fwd-args diff-weights-iter-packed-mem diff-weights-iter-mem))
     (execute! s gru-bwd gru-bwd-args) => s
     (execute! s reorder-weights-bf (fwd-args bwd-weights-mem weights-mem)) => s
     (execute! s reorder-weights-iter-bf (fwd-args bwd-weights-iter-mem weights-iter-mem)) => s
     (execute! s reorder-diff-weights-pack (fwd-args diff-weights-mem diff-weights-packed-mem))
     (execute! s reorder-diff-weights-iter-pack (fwd-args diff-weights-iter-mem diff-weights-iter-packed-mem))
     (map float (seq weights-vec)) => (map float [0.111 0.112 0.121 0.122 0.131 0.132
                                                  0.211 0.212 0.221 0.222 0.231 0.232
                                                  0.311 0.312 0.321 0.322 0.331 0.332
                                                  0.411 0.412 0.421 0.422 0.431 0.432])
     (seq dst-vec) => [0.1986464262008667 0.10329369455575943 0.3485546410083771 0.19498808681964874]
     (seq src-vec) => [-0.019561385735869408 -0.036919500678777695 0.0 0.0]
     (seq src-iter-vec) => [-43.723384857177734 -75.57097625732422 2.7878310680389404 -5.621692180633545]
     (map float (seq diff-weights-vec))
     => (map float [0.36496845 -0.48433483 0.121 0.122 0.02170933 0.23649475
                    0.5919527 -0.6825022 0.221 0.222 0.067064 0.38874215
                    0.12007338 0.50730044 0.32101682 0.32201445 0.44945595 0.31934232
                    0.25949857 0.5669721 0.42101333 0.42201144 0.5249955 0.42195606])
     (seq diff-weights-iter-vec)
     => (map float [100.0 200.0 100.0 200.0 100.0 200.0
                    300.0 400.0 300.0 400.0 300.0 400.0
                    -0.06661617755889893 0.09495311975479126 0.010016393847763538
                    0.020014090463519096 0.0518181174993515 0.016099922358989716
                    -0.009839469566941261 0.07897470146417618 0.030008524656295776
                    0.0400073267519474 0.05569284409284592 0.03760381042957306]))))
