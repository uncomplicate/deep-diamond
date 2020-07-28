;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.diamond.internal.dnnl.core-test
  (:require [midje.sweet :refer [facts throws => roughly]]
            [uncomplicate.commons
             [core :refer [with-release]]
             [utils :refer [capacity direct-buffer put-float get-float]]]
            [uncomplicate.neanderthal
             [core :refer [zero nrm2 entry! entry]]
             [native :refer [fv]]
             [block :refer [buffer]]]
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
                      md (memory-desc [2 2 4 6] :float :nchw)
                      buf (direct-buffer (size md))
                      sub-md (submemory-desc md [1 2 4 6] [0 0 0 0])
                      sub-mem (memory eng sub-md buf)
                      relu-desc (eltwise-fwd-desc :inference :relu sub-md)
                      relu-pd (primitive-desc eng relu-desc)
                      relu (primitive relu-pd)
                      relu-args (fwd-args sub-mem)]
         (put-float buf 24 -1000)
         (execute! s relu relu-args) => s
         (get-float buf 24) => 0.0))

(facts "In-place Sum operation"
       (with-release [eng (engine)
                      s (stream eng)
                      md (memory-desc [2 3 4 5] :float :nchw)
                      buf (direct-buffer (size md))
                      src (memory eng md buf)
                      sum-pd (sum! eng 2.0 md)
                      sum-prim (primitive sum-pd)
                      sum-args (args src)]
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
                      sum-args (args dst src dst)]
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
                      sum-args (args dst src0 src1)]
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
                      weights-desc (memory-desc [3 2] :float :any)
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
         diff-src-vec => (fv 0.51 0.64000004529953 0.9 1.2)
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
                      conv-desc (convolution-fwd-desc :inference :auto
                                                      src-desc weights-desc bias-desc dst-desc
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
                      conv-desc (convolution-fwd-desc :training :auto
                                                      src-desc weights-desc bias-desc dst-desc
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
                      conv-bwd-data-desc (convolution-bwd-desc :auto diff-src-desc
                                                               weights-desc diff-dst-desc
                                                               [1 1] [0 0] [0 0])
                      conv-bwd-data-pd (primitive-desc eng conv-bwd-data-desc conv-pd)
                      diff-src-vec (fv 32)
                      diff-dst-vec (fv [0.2 0.3 0.8 1 1 1 1 1])
                      diff-src-mem (memory eng (diff-src-md conv-bwd-data-pd) (buffer diff-src-vec))
                      diff-dst-mem (memory eng (diff-dst-md conv-bwd-data-pd) (buffer diff-dst-vec))
                      conv-bwd-data (primitive conv-bwd-data-pd)
                      conv-bwd-data-args (bwd-args diff-dst-mem weights-mem diff-src-mem)
                      conv-bwd-weights-desc (convolution-bwd-desc :auto
                                                                  src-desc weights-desc
                                                                  bias-desc dst-desc
                                                                  [1 1] [0 0] [0 0])
                      conv-bwd-weights-pd (primitive-desc eng conv-bwd-weights-desc conv-pd)
                      diff-weights-vec (fv 32)
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
         (seq diff-src-vec) => (map float [-0.4 -0.6 0.2 0.3 -1.6 -1.8 1.1 1.0 -0.2
                                           0.09999999403953552 0.3999999761581421
                                           0.0 -0.8 -2.6 -2.0 0.0
                                           -2.0 -2.0 1.0 1.0 -2.0 -1.0 2.0 1.0
                                           -1.0 -2.0 -1.0 0.0 -1.0 -3.0 -2.0 0.0])
         (execute! s conv-bwd-weights conv-bwd-weights-args) => s
         (seq diff-weights-vec) => (map float [251.9 230.9 93.6 217.0 186.0 233.0 81.0 198.6
                                               415.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
                                               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
                                               0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0])
         (entry diff-bias-vec 0) => (float 6.3)))
