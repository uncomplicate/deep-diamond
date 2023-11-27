;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
  uncomplicate.diamond.internal.dnnl.core-test
  (:require [midje.sweet :refer [facts throws => roughly truthy just]]
            [uncomplicate.commons.core :refer [with-release bytesize size release]]
            [uncomplicate.fluokitten.core :refer [extract]]
            [uncomplicate.clojure-cpp
             :refer [pointer put-float! get-float byte-pointer float-pointer put-entry! get-entry
                     position pointer-seq position! null?]]
            [uncomplicate.neanderthal
             [core :refer [zero nrm2 entry! entry transfer!]]
             [native :refer [fv]]
             [block :refer [buffer]]
             [math :refer [sqr sqrt]]]
            [uncomplicate.diamond.internal.dnnl
             [core :refer :all]
             [protocols :as api]])
  (:import clojure.lang.ExceptionInfo))

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
         (bytesize md) => (* (long (first strds)) (long (first dimensions)) Float/BYTES)
         (memory-desc [1 1] :f64 [1 1]) => (throws ExceptionInfo)
         (data-type (memory-desc [1 1])) => :float
         (dims (memory-desc [2 3])) => [2 3]
         (strides (memory-desc [2 3])) => [0 0]
         (strides (memory-desc [2 3] :float :nc)) => [3 1]))

(facts "Memory descriptor by tag."
       (with-release [md (memory-desc [2 3 4 5] :float :nchw)
                      md1 (memory-desc [2 1 2 3] :float :nchw)
                      md2 (memory-desc [2 1 2 3] :float :nhwc)]
         (= md md1) => false
         (equal-desc? md1 md2) => true ;; surprisingly! but DNNL's dnnl_memory_desc_equal say s so by returning 1...
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
         (pointer mem) => truthy
         (bytesize (pointer mem)) => 480))

(facts "Memory offset operation."
       (with-release [eng (engine)
                      s (stream eng)
                      md (memory-desc [2 3 4 5] :float :nchw)
                      buf (byte-pointer (+ 8 (bytesize md)))
                      mem (memory eng md buf)
                      relu-pd (eltwise-fwd eng :inference :relu md)
                      relu (primitive relu-pd)
                      relu-args (fwd-args mem)]
         (put-float! buf 0 -100)
         (put-float! buf 1 -20)
         (put-float! buf 2 -200)
         (put-float! buf 120 -400)
         (put-float! buf 121 -500)
         (offset! mem 4)
         (position (pointer mem)) => 4
         (position buf) => 0
         (execute! s relu relu-args) => s
         (get-float buf 0) => -100.0
         (get-float buf 1) => 0.0
         (get-float buf 2) => 0.0
         (get-float buf 120) => 0.0
         (get-float buf 121) => -500.0
         (offset! mem 489) => (throws ExceptionInfo)))

(facts "Memory with typed pointers (float)."
       (with-release [eng (engine)
                      s (stream eng)
                      md (memory-desc [2] :float :x)
                      buf (byte-pointer (+ (bytesize md) Float/BYTES))
                      mem (memory eng md (position! buf 4))
                      relu-pd (eltwise-fwd eng :inference :relu md)
                      relu (primitive relu-pd)
                      relu-args (fwd-args mem)]
         (position! buf 0)
         (put-float! buf 0 -1)
         (put-float! buf 1 -2)
         (put-float! buf 2 -3)
         (position (pointer mem)) => 4
         (position buf) => 0
         (execute! s relu relu-args) => s
         (get-float buf 0) => -1.0
         (get-float buf 1) => 0.0
         (get-float buf 2) => 0.0
         (size buf) => 12
         (position buf) => 0
         (pointer-seq (float-pointer buf)) => [-1.0 0.0 0.0]))

(facts "Memory descriptor lifecycle."
       (with-release [eng (engine)
                      s (stream eng)
                      md (memory-desc [2] :float :x)
                      buf (byte-pointer (+ (bytesize md) Float/BYTES))
                      mem (memory eng md (position! buf 4))
                      mem-master (memory eng md)
                      relu-pd (eltwise-fwd eng :inference :relu md)
                      relu (primitive relu-pd)
                      relu-args (fwd-args mem)]
         (null? (extract mem-master)) => false
         (release mem-master) => true
         (null? (extract mem-master)) => true
         (null? (extract md)) => false
         (release md) => true
         (release md) => true
         (null? (extract md)) => true
         (extract md) => nil
         (null? (extract mem)) => false
         (null? (pointer mem)) => false
         (execute! s relu relu-args) => s))

(facts "Submemory descriptor."
       (with-release [eng (engine)
                      s (stream eng)
                      md (memory-desc [2 3 4 5] :float :nchw)
                      buf (byte-pointer (bytesize md))
                      sub-md (submemory-desc md 1)
                      sub-mem (memory eng sub-md buf)
                      relu-pd (eltwise-fwd eng :inference :relu sub-md)
                      relu (primitive relu-pd)
                      relu-args (fwd-args sub-mem)]
         (dims md) => [2 3 4 5]
         (strides md) => [60 20 5 1]
         (dims sub-md) => [1 3 4 5]
         (strides sub-md) => [60 20 5 1]
         (put-float! buf 0 -100)
         (put-float! buf 1 -20)
         (put-float! buf 2 -200)
         (put-float! buf 59 -1)
         (put-float! buf 60 -2)
         (put-float! buf 119 -400)
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
                      buf (float-pointer (quot (bytesize md) Float/BYTES))
                      sub-md (submemory-desc md [2 1 3 2] [0 1 0 0])
                      sub-mem (memory eng sub-md buf)
                      relu-pd (eltwise-fwd eng :inference :abs sub-md)
                      relu (primitive relu-pd)
                      relu-args (fwd-args sub-mem)]
         (dotimes [i 24]
           (put-entry! buf i (- i)))
         (execute! s relu relu-args) => s
         (get-entry buf 5) => -5.0
         (get-entry buf 6) => 6.0
         (get-entry buf 11) => 11.0
         (get-entry buf 12) => -12.0
         (get-entry buf 23) => 23.0))

(facts "In-place Sum operation"
       (with-release [eng (engine)
                      s (stream eng)
                      md (memory-desc [2 3 4 5] :float :nchw)
                      buf (byte-pointer (bytesize md))
                      src (memory eng md buf)
                      sum-pd (sum! eng 2.0 md)
                      sum-prim (primitive sum-pd)
                      sum-args (multi-args src)]
         (put-float! buf 0 -100)
         (put-float! buf 1 20)
         (execute! s sum-prim sum-args) => s
         (get-float buf 0) => -200.0
         (get-float buf 1) => 40.0))

(facts "In-place Sum operation with two sources"
       (with-release [eng (engine)
                      s (stream eng)
                      md (memory-desc [2 3 4 5] :float :nchw)
                      buf-src (byte-pointer (bytesize md))
                      src (memory eng md buf-src)
                      buf-dst (byte-pointer (bytesize md))
                      dst (memory eng md buf-dst)
                      sum-pd (sum! eng md 2.0 md 3.0 md)
                      sum-prim (primitive sum-pd)
                      sum-args (multi-args dst src dst)]
         (put-float! buf-src 0 -100)
         (put-float! buf-src 1 10)
         (put-float! buf-dst 0 -200)
         (put-float! buf-dst 1 20)
         (execute! s sum-prim sum-args) => s
         (get-float buf-src 0) => -100.0
         (get-float buf-src 1) => 10.0
         (get-float buf-dst 0) => -800.0
         (get-float buf-dst 1) => 80.0))

(facts "Out of place Sum operation"
       (with-release [eng (engine)
                      s (stream eng)
                      md (memory-desc [2 3 4 5] :float :nchw)
                      src0-buf (byte-pointer (bytesize md))
                      src1-buf (byte-pointer (bytesize md))
                      dst-buf (byte-pointer (bytesize md))
                      src0 (memory eng md src0-buf)
                      src1 (memory eng md src1-buf)
                      dst (memory eng md dst-buf)
                      sum-pd (sum! eng md 2.0 md 3.0 md)
                      sum-prim (primitive sum-pd)
                      sum-args (multi-args dst src0 src1)]
         (put-float! dst-buf 0 100000.0)
         (put-float! src0-buf 0 -100)
         (put-float! src0-buf 1 20)
         (put-float! src1-buf 0 -1000)
         (put-float! src1-buf 1 200)
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

(facts "Reordering memory with offsets."
       (let [dims [2 2 3 2]]
         (with-release [eng (engine)
                        s (stream eng)
                        a-desc (memory-desc dims :float :nchw)
                        b-desc (memory-desc dims :float :nchw)
                        reorder-a-b-pd (reorder eng a-desc b-desc)
                        a-vec (fv (range (apply * 3 dims)))
                        b-vec (zero a-vec)
                        a-mem (memory eng a-desc (buffer a-vec))
                        b-mem (memory eng a-desc (position! (buffer b-vec) 5))
                        reorder-a-b (primitive reorder-a-b-pd)
                        reorder-a-b-args (fwd-args a-mem b-mem)]
           (= a-vec b-vec) => false
           (= (nrm2 a-vec) (nrm2 b-vec)) => false
           (take 3 (pointer-seq (pointer (offset! a-mem 2)))) => [2.0 3.0 4.0]
           (execute! s reorder-a-b reorder-a-b-args) => s
           (take 6 (pointer-seq (pointer (offset! b-mem 2)))) => [0.0 0.0 0.0 2.0 3.0 4.0]
           (pointer-seq (pointer a-mem)) => (range 2.0 (apply * 3 dims))
           (take 25 (pointer-seq (pointer b-mem))) => (into [0.0 0.0 0.0] (range 2.0 (apply * dims))))))

(facts "Elementwise forward ReLU operation."
       (with-release [eng (engine)
                      s (stream eng)
                      md (memory-desc [2 3 4 5] :float :nchw)
                      buf (byte-pointer (bytesize md))
                      mem (memory eng md buf)
                      relu-pd (eltwise-fwd eng :inference :relu md)
                      relu (primitive relu-pd)
                      relu-args (fwd-args mem)]
         (put-float! buf 0 -100)
         (put-float! buf 1 20)
         (execute! s relu relu-args) => s
         (get-float buf 0) => 0.0
         (get-float buf 1) => 20.0))

(facts "Elementwise backward ReLU operation."
       (with-release [eng (engine)
                      s (stream eng)
                      md (memory-desc [2 3] :float :nc)
                      buf (byte-pointer (bytesize md))
                      mem (memory eng md buf)
                      relu-pd (eltwise-fwd eng :training :relu md)
                      relu (primitive relu-pd)
                      relu-args (fwd-args mem)
                      diff-dst-vec (fv (range 2 8))
                      diff-dst-desc (memory-desc [2 3] :float :nc)
                      relu-bwd-pd (eltwise-bwd eng relu-pd :relu diff-dst-desc md)
                      diff-dst-mem (memory eng (diff-dst-md relu-bwd-pd) (buffer diff-dst-vec))
                      relu-bwd (primitive relu-bwd-pd)
                      relu-bwd-args (eltwise-bwd-args mem diff-dst-mem diff-dst-mem)]
         (put-float! buf 0 -100)
         (put-float! buf 1 20)
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
                      logistic-pd (eltwise-fwd eng :training :logistic src-md)
                      logistic (primitive logistic-pd)
                      logistic-args (fwd-args src-mem dst-mem)
                      diff-dst-vec (fv [-0.5 -0.2 -0.4 0 0.2 0.3])
                      diff-dst-desc (memory-desc [2 3] :float :nc)
                      diff-src-vec (fv 6)
                      logistic-bwd-pd (eltwise-bwd eng logistic-pd :logistic diff-dst-desc src-md)
                      diff-dst-mem (memory eng (diff-dst-md logistic-bwd-pd) (buffer diff-dst-vec))
                      diff-src-mem (memory eng (diff-src-md logistic-bwd-pd) (buffer diff-src-vec))
                      logistic-bwd (primitive logistic-bwd-pd)
                      logistic-bwd-args (eltwise-bwd-args src-mem diff-dst-mem diff-src-mem)]
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
                      ip-pd (inner-product-fwd eng :inference src-desc weights-desc bias-desc dst-desc)
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
         (execute! s ip ip-args) => s
         dst-vec => (fv [1.428 4.095999717712402])))

(facts "Inner product backward."
       (with-release [eng (engine)
                      s (stream eng)
                      src-desc (memory-desc [2 2] :float :nc)
                      weights-desc (memory-desc [3 2] :float :io)
                      bias-desc (memory-desc [3] :float :x)
                      dst-desc (memory-desc [2 3] :float :nc)
                      ip-pd (inner-product-fwd eng :training src-desc weights-desc bias-desc dst-desc)
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
                      ip-bwd-data-pd (inner-product-bwd eng ip-pd diff-src-desc weights-desc diff-dst-desc)
                      diff-src-vec (fv 4)
                      diff-dst-vec (fv [0.2 0.3 0.8 1 1 1])
                      diff-src-mem (memory eng (diff-src-md ip-bwd-data-pd) (buffer diff-src-vec))
                      diff-dst-mem (memory eng (diff-dst-md ip-bwd-data-pd) (buffer diff-dst-vec))
                      ip-bwd-data (primitive ip-bwd-data-pd)
                      ip-bwd-data-args (bwd-args diff-dst-mem weights-mem diff-src-mem)
                      diff-weights-desc (memory-desc [3 2] :float :nc)
                      diff-bias-desc (memory-desc [3] :float :x)
                      ip-bwd-weights-pd (inner-product-bwd eng ip-pd src-desc diff-weights-desc
                                                           diff-bias-desc diff-dst-desc)
                      diff-weights-vec (fv [1 0 0 0 0 0])
                      diff-bias-vec (fv [5 5 5])
                      diff-weights-mem (memory eng (diff-weights-md ip-bwd-weights-pd)
                                               (buffer diff-weights-vec))
                      diff-bias-mem (memory eng diff-bias-desc (buffer diff-bias-vec))
                      ip-bwd-weights (primitive ip-bwd-weights-pd)
                      ip-bwd-weights-args (bwd-args src-mem diff-dst-mem
                                                    diff-weights-mem diff-bias-mem)]
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
                      ip-pd (inner-product-fwd eng :training src-desc weights-desc bias-desc dst-desc)
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
                      ip-bwd-data-pd (inner-product-bwd eng ip-pd diff-src-desc weights-desc diff-dst-desc)
                      diff-src-vec (fv 1)
                      diff-dst-vec (fv [0.4])
                      diff-src-mem (memory eng (diff-src-md ip-bwd-data-pd) (buffer diff-src-vec))
                      diff-dst-mem (memory eng (diff-dst-md ip-bwd-data-pd) (buffer diff-dst-vec))
                      ip-bwd-data (primitive ip-bwd-data-pd)
                      ip-bwd-data-args (bwd-args diff-dst-mem weights-mem diff-src-mem)
                      diff-weights-desc (memory-desc [1 1] :float :nc)
                      diff-bias-desc (memory-desc [1] :float :x)
                      ip-bwd-weights-pd (inner-product-bwd eng ip-pd src-desc diff-weights-desc
                                                           diff-bias-desc diff-dst-desc)
                      diff-weights-vec (fv [1000])
                      diff-bias-vec (fv [5000])
                      diff-weights-mem (memory eng (diff-weights-md ip-bwd-weights-pd)
                                               (buffer diff-weights-vec))
                      diff-bias-mem (memory eng diff-bias-desc (buffer diff-bias-vec))
                      ip-bwd-weights (primitive ip-bwd-weights-pd)
                      ip-bwd-weights-args (bwd-args src-mem diff-dst-mem diff-weights-mem diff-bias-mem)]
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
                      buf (byte-pointer (bytesize md))
                      mem (memory eng md buf)
                      axis 1
                      softmax-pd (softmax-fwd eng :inference :accurate md axis)
                      softmax (primitive softmax-pd)
                      softmax-args (fwd-args mem)]
         (put-float! buf 0 1)
         (put-float! buf 1 3)
         (put-float! buf 2 3)
         (put-float! buf 3 2)
         (put-float! buf 4 4)
         (put-float! buf 5 8)
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
                      buf (byte-pointer (bytesize md))
                      mem-vec (fv 1 3 3 2 4 8)
                      mem (memory eng md (buffer mem-vec))
                      axis 1
                      softmax-pd (softmax-fwd eng :training :accurate md axis)
                      softmax (primitive softmax-pd)
                      softmax-args (fwd-args mem)
                      diff-dst-vec (fv 0 -2.135335400336505 0 0 0 -1.0207943791746268) ;; -ti/aLi
                      diff-dst-desc (memory-desc [2 3] :float :nc)
                      softmax-bwd-pd (softmax-bwd eng softmax-pd :accurate diff-dst-desc md axis)
                      diff-dst-mem (memory eng (diff-dst-md softmax-bwd-pd) (buffer diff-dst-vec))
                      softmax-bwd (primitive softmax-bwd-pd)
                      softmax-bwd-args (softmax-bwd-args mem diff-dst-mem mem)]
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
                      conv-pd (convolution-fwd eng :inference :auto
                                               src-desc weights-desc bias-desc dst-desc
                                               [1 1] [0 0])
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
         (execute! s conv conv-args) => s
         (seq dst-vec) => [18.5 -93.5 -20.5 -565.5 102.5 57.5 -77.5 -175.5]))

(facts "Convolution backward."
       (with-release [eng (engine)
                      s (stream eng)
                      src-desc (memory-desc [2 1 4 4] :float :nchw)
                      weights-desc (memory-desc [1 1 3 3] :float :nchw)
                      bias-desc (memory-desc [1] :float :x)
                      dst-desc (memory-desc [2 1 2 2] :float :nchw)
                      conv-pd (convolution-fwd eng :training :auto
                                               src-desc weights-desc bias-desc dst-desc
                                               [1 1] [0 0])
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
                      conv-bwd-data-pd (convolution-bwd-data eng conv-pd :auto
                                                             diff-src-desc weights-desc diff-dst-desc
                                                             [1 1] [0 0] [0 0])
                      diff-src-vec (fv 32)
                      diff-dst-vec (fv [0.2 0.3 0.8 1 1 1 1 1])
                      diff-src-mem (memory eng (diff-src-md conv-bwd-data-pd) (buffer diff-src-vec))
                      diff-dst-mem (memory eng (diff-dst-md conv-bwd-data-pd) (buffer diff-dst-vec))
                      conv-bwd-data (primitive conv-bwd-data-pd)
                      conv-bwd-data-args (bwd-args diff-dst-mem weights-mem diff-src-mem)
                      conv-bwd-weights-pd (convolution-bwd-weights eng conv-pd :auto
                                                                   src-desc weights-desc bias-desc dst-desc
                                                                   [1 1] [0 0] [0 0])
                      diff-weights-vec (fv 9)
                      diff-bias-vec (fv [1.5])
                      diff-weights-mem (memory eng (diff-weights-md conv-bwd-weights-pd)
                                               (buffer diff-weights-vec))
                      diff-bias-mem (memory eng bias-desc (buffer diff-bias-vec))
                      conv-bwd-weights (primitive conv-bwd-weights-pd)
                      conv-bwd-weights-args (bwd-args src-mem diff-dst-mem
                                                      diff-weights-mem diff-bias-mem)]
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
                      conv-pd (convolution-fwd eng :inference :auto
                                               src-desc weights-desc bias-desc dst-desc
                                               [1 1] [0 0] [0 0])
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
         (execute! s conv conv-args) => s
         (seq dst-vec) => [18.5 -93.5 -20.5 -565.5 102.5 57.5 -77.5 -175.5]))

(facts "Dilated convolution backward."
       (with-release [eng (engine)
                      s (stream eng)
                      src-desc (memory-desc [2 1 4 4] :float :nchw)
                      weights-desc (memory-desc [1 1 3 3] :float :nchw)
                      bias-desc (memory-desc [1] :float :x)
                      dst-desc (memory-desc [2 1 2 2] :float :nchw)
                      conv-pd (convolution-fwd eng :training :auto
                                               src-desc weights-desc bias-desc dst-desc
                                               [1 1] [0 0] [0 0])
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
                      conv-bwd-data-pd (convolution-bwd-data eng conv-pd :auto diff-src-desc
                                                             weights-desc diff-dst-desc
                                                             [1 1] [0 0] [0 0] [0 0])
                      diff-src-vec (fv 32)
                      diff-dst-vec (fv [0.2 0.3 0.8 1 1 1 1 1])
                      diff-src-mem (memory eng (diff-src-md conv-bwd-data-pd) (buffer diff-src-vec))
                      diff-dst-mem (memory eng (diff-dst-md conv-bwd-data-pd) (buffer diff-dst-vec))
                      conv-bwd-data (primitive conv-bwd-data-pd)
                      conv-bwd-data-args (bwd-args diff-dst-mem weights-mem diff-src-mem)
                      conv-bwd-weights-pd (convolution-bwd-weights
                                           eng conv-pd :auto src-desc weights-desc bias-desc dst-desc
                                           [1 1] [0 0] [0 0] [0 0])
                      diff-weights-vec (fv 9)
                      diff-bias-vec (fv [1.5])
                      diff-weights-mem (memory eng (diff-weights-md conv-bwd-weights-pd)
                                               (buffer diff-weights-vec))
                      diff-bias-mem (memory eng bias-desc (buffer diff-bias-vec))
                      conv-bwd-weights (primitive conv-bwd-weights-pd)
                      conv-bwd-weights-args (bwd-args src-mem diff-dst-mem
                                                      diff-weights-mem diff-bias-mem)]
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
                      pool-pd (pooling-fwd eng :inference :max src-desc dst-desc [2 2] [2 2] [0 0])
                      src-vec (fv 0 43 3 30 0 98 0 0 7 38 0 0 19 20 175 50
                                  0 0 7 19 43 98 38 20 3 0 0 175 30 0 0 50)
                      src-mem (memory eng src-desc (buffer src-vec))
                      dst-vec (fv (* 2 1 2 2))
                      dst-mem (memory eng dst-desc (buffer dst-vec))
                      pool (primitive pool-pd)
                      pool-args (fwd-args src-mem dst-mem)]
         (execute! s pool pool-args) => s
         src-vec => (fv 0 43 3 30 0 98 0 0 7 38 0 0 19 20 175 50
                        0 0 7 19 43 98 38 20 3 0 0 175 30 0 0 50)
         (seq dst-vec) => [98.0 30.0 38.0 175.0 98.0 38.0 30.0 175.0]))

(facts "Max pooling backward."
       (with-release [eng (engine)
                      s (stream eng)
                      src-desc (memory-desc [2 1 4 4] :float :nchw)
                      dst-desc (memory-desc [2 1 2 2] :float :nchw)
                      pool-pd (pooling-fwd eng :training :max src-desc dst-desc [2 2] [2 2] [0 0])
                      src-vec (fv 0 43 3 30 0 98 0 0 7 38 0 0 19 20 175 50
                                  0 0 7 19 43 98 38 20 3 0 0 175 30 0 0 50)
                      src-mem (memory eng src-desc (buffer src-vec))
                      dst-vec (fv (* 2 1 2 2))
                      dst-mem (memory eng dst-desc (buffer dst-vec))
                      workspace-mem (memory eng (workspace-md pool-pd))
                      pool (primitive pool-pd)
                      pool-args (fwd-args src-mem dst-mem workspace-mem)

                      pool-bwd-pd (pooling-bwd eng pool-pd :max src-desc dst-desc [2 2] [2 2] [0 0])
                      diff-dst-vec (entry! (zero src-vec) 2.0)
                      diff-src-vec (entry! (zero src-vec) 0.0)
                      diff-dst-mem (memory eng (diff-dst-md pool-bwd-pd) (buffer diff-dst-vec))
                      diff-src-mem (memory eng (diff-src-md pool-bwd-pd) (buffer diff-src-vec))
                      pool-bwd (primitive pool-bwd-pd)
                      pool-bwd-args (pooling-bwd-args diff-dst-mem diff-src-mem workspace-mem)]
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
                      scaleshift-desc (memory-desc [1] :float :x)
                      bnrm-pd (batch-norm-fwd eng :inference data-desc :scale :shift)
                      src-vec (fv (range -1 7))
                      src-mem (memory eng data-desc (buffer src-vec))
                      dst-vec (fv 8)
                      dst-mem (memory eng data-desc (buffer dst-vec))
                      scale-vec (fv [0.5])
                      scale-mem (memory eng scaleshift-desc (buffer scale-vec))
                      shift-vec (fv [1.5])
                      shift-mem (memory eng scaleshift-desc (buffer shift-vec))
                      bnrm (primitive bnrm-pd)
                      bnrm-args (batch-norm-fwd-args src-mem dst-mem scale-mem shift-mem true)]
         (execute! s bnrm bnrm-args) => s
         (seq src-vec) => (range -1.0 7.0)
         (seq dst-vec) => [0.7362374067306519 0.9544553160667419 1.172673225402832 1.3908910751342773
                           1.6091089248657227 1.827326774597168 2.0455446243286133 2.2637624740600586]))

(facts "Batch normalization backward."
       (with-release [eng (engine)
                      s (stream eng)
                      data-desc (memory-desc [1 2 2 2] :float :nchw)
                      stats-desc (memory-desc [2] :float :x)
                      bnrm-pd (batch-norm-fwd eng :training data-desc :scale :shift)
                      src-vec (fv (range -1 7))
                      src-mem (memory eng data-desc (buffer src-vec))
                      dst-vec (fv 8)
                      dst-mem (memory eng data-desc (buffer dst-vec))
                      scaleshift-desc (memory-desc [2] :float :x)
                      scale-vec (fv [0.5 1.5])
                      scale-mem (memory eng scaleshift-desc (buffer scale-vec))
                      shift-vec (fv [1 1])
                      shift-mem (memory eng scaleshift-desc (buffer shift-vec))
                      mean-vec (fv 2)
                      mean-mem (memory eng stats-desc (buffer mean-vec))
                      variance-vec (fv 2)
                      variance-mem (memory eng stats-desc (buffer variance-vec))
                      bnrm (primitive bnrm-pd)
                      bnrm-args (batch-norm-fwd-args src-mem dst-mem scale-mem shift-mem
                                                     mean-mem variance-mem)
                      ;; :shift is no longer supported by DNNL, which is a pity. (September 2023)
                      bnrm-bwd-pd (batch-norm-bwd eng bnrm-pd :backward data-desc data-desc :scale)
                      diff-dst-vec (fv [-5 10 0.3 0.2 -0.5 0.6 0.9 -3])
                      diff-dst-mem (memory eng (diff-dst-md bnrm-bwd-pd) (buffer diff-dst-vec))
                      diff-src-vec (fv 8)
                      diff-src-mem (memory eng (diff-src-md bnrm-bwd-pd) (buffer diff-src-vec))
                      diff-scale-vec (fv 2)
                      diff-scale-mem (memory eng scaleshift-desc (buffer diff-scale-vec))
                      bnrm-bwd (primitive bnrm-bwd-pd)
                      bnrm-bwd-args (batch-norm-bwd-args diff-dst-mem src-mem scale-mem shift-mem
                                                         mean-mem variance-mem
                                                         diff-src-mem diff-scale-mem)]
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
         (seq diff-scale-vec) => [2.6385602951049805 -3.219937801361084]))

(facts "In-place Binary operation"
       (with-release [eng (engine)
                      s (stream eng)
                      md (memory-desc [2 3 4 5] :float :nchw)
                      buf0 (byte-pointer (bytesize md))
                      src0 (memory eng md buf0)
                      buf1 (byte-pointer (bytesize md))
                      src1 (memory eng md buf1)
                      add-pd (binary eng :add md)
                      add-prim (primitive add-pd)
                      add-args (binary-args src0 src1)]
         (put-float! buf0 0 -100)
         (put-float! buf0 1 20)
         (put-float! buf1 0 -200)
         (put-float! buf1 1 30)
         (execute! s add-prim add-args) => s
         (get-float buf0 0) => -300.0
         (get-float buf0 1) => 50.0))

(facts "Reduction sum operation"
       (with-release [eng (engine)
                      s (stream eng)
                      src-md (memory-desc [2 3 4 5] :float :nchw)
                      src-buf (byte-pointer (bytesize src-md))
                      src (memory eng src-md src-buf)
                      dst-md (memory-desc [2 3 4 1] :float :nchw)
                      dst-buf (byte-pointer (bytesize dst-md))
                      dst (memory eng dst-md dst-buf)
                      sum-pd (reduction eng :sum src-md dst-md)
                      sum-prim (primitive sum-pd)
                      sum-args (fwd-args src dst)]
         (dotimes [i 120]
           (put-float! src-buf i i))
         (execute! s sum-prim sum-args) => s
         (get-float dst-buf 0) => (float (apply + (range 5)))
         (get-float dst-buf 1) => (float (apply + (range 5 10)))
         (get-float dst-buf 2) => (float (apply + (range 10 15)))))

(facts "Reduction max operation"
       (with-release [eng (engine)
                      s (stream eng)
                      src-md (memory-desc [2 3 4 5] :float :nchw)
                      src-buf (byte-pointer (bytesize src-md))
                      src (memory eng src-md src-buf)
                      dst-md (memory-desc [1 3 1 1] :float :nchw)
                      dst-buf (byte-pointer (bytesize dst-md))
                      dst (memory eng dst-md dst-buf)
                      max-pd (reduction eng :max src-md dst-md)
                      max-prim (primitive max-pd)
                      max-args (fwd-args src dst)]
         (dotimes [i 120]
           (put-float! src-buf i i))
         (execute! s max-prim max-args) => s
         (get-float dst-buf 0) => 79.0
         (get-float dst-buf 1) => 99.0
         (get-float dst-buf 2) => 119.0))

(facts "Reduction L2 operation"
       (with-release [eng (engine)
                      s (stream eng)
                      src-md (memory-desc [2 3 4 5] :float :nchw)
                      src-buf (byte-pointer (bytesize src-md))
                      src (memory eng src-md src-buf)
                      dst-md (memory-desc [2 3 4 1] :float :nchw)
                      dst-buf (byte-pointer (bytesize dst-md))
                      dst (memory eng dst-md dst-buf)
                      norm-pd (reduction eng :norm-lp-sum src-md dst-md 2.0 0.0)
                      norm-prim (primitive norm-pd)
                      norm-args (fwd-args src dst)]
         (dotimes [i 120]
           (put-float! src-buf i i))
         (execute! s norm-prim norm-args) => s
         (get-float dst-buf 0) => (float (sqrt (apply + (map sqr (range 5)))))
         (get-float dst-buf 1) => (float (sqrt (apply + (map sqr (range 5 10)))))
         (get-float dst-buf 2) => (float (sqrt (apply + (map sqr (range 10 15)))))))

(facts "Concatenate operation with one source"
       (with-release [eng (engine)
                      s (stream eng)
                      md (memory-desc [2 3 4 5] :float :nchw)
                      src-buf (byte-pointer (bytesize md))
                      src (memory eng md src-buf)
                      dst-buf (byte-pointer (bytesize md))
                      dst (memory eng md dst-buf)
                      concat-pd (concatenate eng md 0 md)
                      concat-prim (primitive concat-pd)
                      concat-args (multi-args dst src)]
         (dotimes [i 120]
           (put-float! src-buf i i))
         (execute! s concat-prim concat-args) => s
         (get-float dst-buf 0) => 0.0
         (get-float dst-buf 100) => 100.0))

(facts "Concatenate operation with two homogeneous sources"
       (with-release [eng (engine)
                      s (stream eng)
                      src-md (memory-desc [2 3 4 5] :float :nchw)
                      dst-md (memory-desc [4 3 4 5] :float :nchw)
                      src0-buf (byte-pointer (bytesize src-md))
                      src0 (memory eng src-md src0-buf)
                      src1-buf (byte-pointer (bytesize src-md))
                      src1 (memory eng src-md src1-buf)
                      dst-buf (byte-pointer (bytesize dst-md))
                      dst (memory eng dst-md dst-buf)
                      concat-pd (concatenate eng dst-md 0 src-md src-md)
                      concat-prim (primitive concat-pd)
                      concat-args (multi-args dst src0 src1)]
         (dotimes [i 120]
           (put-float! src0-buf i i)
           (put-float! src1-buf i (* 1000.0 i)))
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
                      src0-buf (byte-pointer (bytesize src0-md))
                      src0 (memory eng src1-md src0-buf)
                      src1-buf (byte-pointer (bytesize src1-md))
                      src1 (memory eng src1-md src1-buf)
                      src2-buf (byte-pointer (bytesize src2-md))
                      src2 (memory eng src2-md src2-buf)
                      dst-buf (byte-pointer (bytesize dst-md))
                      dst (memory eng dst-md dst-buf)
                      concat-pd (concatenate eng dst-md 2 src0-md src1-md src2-md)
                      concat-prim (primitive concat-pd)
                      concat-args (multi-args dst src0 src1 src2)]
         (put-float! src0-buf 0 1.0)
         (put-float! src0-buf 1 2.0)
         (put-float! src1-buf 0 10.0)
         (put-float! src2-buf 0 5.0)
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
     (vanilla-rnn-fwd eng :inference :relu :unidirectional
                      src-desc nil weights-desc weights-iter-desc bias-desc
                      dst-desc nil) => truthy
     (vanilla-rnn-fwd eng :inference :relu :unidirectional
                      src-desc src-iter-desc weights-desc weights-iter-desc bias-desc
                      dst-desc dst-iter-desc) => truthy)))

(facts
 "LSTM dimensions."
 (let [T 2
       N 1
       SC 4
       DC 2
       G 4
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
      src-iter-c-desc (memory-desc src-iter-dim :float :ldnc)
      weights-desc (memory-desc weights-dim :float :ldigo)
      weights-iter-desc (memory-desc weights-iter-dim :float :ldigo)
      bias-desc (memory-desc bias-dim :float :ldgo)
      dst-desc (memory-desc dst-dim :float :tnc)
      dst-iter-desc (memory-desc dst-iter-dim :float :ldnc)
      dst-iter-c-desc (memory-desc dst-iter-dim :float :ldnc)]
     (lstm-fwd eng :inference :unidirectional
               src-desc nil nil weights-desc weights-desc nil nil bias-desc
               dst-desc nil nil) => (throws Exception)
     (lstm-fwd eng :inference :unidirectional
               src-desc nil nil weights-desc weights-iter-desc nil nil bias-desc
               dst-desc nil nil) => truthy
     (lstm-fwd eng :inference :unidirectional
               src-desc src-iter-desc src-iter-c-desc
               weights-desc weights-iter-desc nil nil bias-desc
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
     (gru-fwd eng :inference :unidirectional
              src-desc nil weights-desc weights-desc bias-desc
              dst-desc nil) => (throws Exception)
     (gru-fwd eng :inference :unidirectional
              src-desc nil weights-desc weights-iter-desc bias-desc
              dst-desc nil) => truthy
     (gru-fwd eng :inference :unidirectional
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
      rnn-pd (vanilla-rnn-fwd eng :inference :relu :unidirectional
                              src-desc src-iter-desc weights-desc weights-desc bias-desc
                              dst-desc dst-iter-desc)
      rnn-no-iter-pd (vanilla-rnn-fwd eng :inference :relu :unidirectional
                                      src-desc nil weights-desc weights-desc bias-desc
                                      dst-desc nil)
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
     (execute! s rnn rnn-args) => s
     (seq src-iter-vec) => [0.0 0.0 0.0 0.0]
     (seq dst-vec) => [2.570000171661377 3.940000057220459 850.6968994140625 1054.8890380859375]
     (seq dst-iter-vec) => [830.4099731445312 1200.8599853515625 850.6968994140625 1054.8890380859375]
     (entry! dst-vec 0)
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
      rnn-fwd-pd (vanilla-rnn-fwd eng :training :relu :unidirectional src-desc src-iter-desc
                                  weights-desc weights-desc bias-desc dst-desc dst-iter-desc)
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
      rnn-bwd-pd (vanilla-rnn-bwd eng rnn-fwd-pd :relu :unidirectional src-desc src-iter-desc
                                  bwd-weights-desc bwd-weights-desc bias-desc
                                  dst-desc dst-iter-desc src-desc src-iter-desc
                                  bwd-weights-desc bwd-weights-desc bias-desc
                                  dst-desc dst-iter-desc)
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

     (execute! s rnn-fwd rnn-fwd-args) => s
     (seq dst-vec) => [2.570000171661377 3.940000057220459 850.6968994140625 1054.8890380859375]
     (seq src-vec) => (map float [2.0 3.0 0.2 0.3])
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
     => (map float [10.435526847839355 -341.5085754394531 15.653291702270508 -512.2628784179688
                    2824.8525390625 -3823.08056640625 4085.3203125 -5529.20458984375])
     (seq diff-weights-iter-vec)
     => (map float [-2.5480000972747803 1.3159998655319214 -4.185999870300293 2.161999464035034
                    8.73799991607666 -11.822000503540039 13.395999908447266 -18.124000549316406])
     (seq bias-vec) => [3.579763650894165 -169.9082794189453 4.441999912261963 -6.881999969482422])))

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
      rnn-fwd-pd (vanilla-rnn-fwd eng :training :relu :unidirectional src-desc src-iter-desc
                                  weights-desc weights-desc bias-desc dst-desc dst-iter-desc)
      rnn-no-iter-fwd-pd (vanilla-rnn-fwd eng :training :relu :unidirectional src-desc src-iter-desc
                                          weights-desc weights-desc bias-desc dst-desc nil)
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
      rnn-bwd-pd (vanilla-rnn-bwd eng rnn-fwd-pd :relu :unidirectional src-desc src-iter-desc
                                  bwd-weights-desc bwd-weights-desc bias-desc
                                  dst-desc dst-iter-desc
                                  src-desc src-iter-desc
                                  bwd-weights-desc bwd-weights-desc bias-desc
                                  dst-desc dst-iter-desc)
      rnn-no-iter-bwd-pd (vanilla-rnn-bwd eng rnn-fwd-pd :relu :unidirectional src-desc src-iter-desc
                                          bwd-weights-desc bwd-weights-desc bias-desc
                                          dst-desc nil
                                          src-desc src-iter-desc
                                          bwd-weights-desc bwd-weights-desc bias-desc
                                          dst-desc nil)
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

     (execute! s rnn-fwd rnn-fwd-args) => s
     (seq dst-vec) => [2.570000171661377 3.940000057220459 850.6968994140625 1054.8890380859375]
     (seq dst-iter-vec) => [830.4099731445312 1200.8599853515625 850.6968994140625 1054.8890380859375]
     (seq src-vec) => (map float [2.0 3.0 0.2 0.3])
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
     => (map float [-551.3486328125 -1255.885498046875 -827.0230102539062 -1883.8282470703125
                    2741.815673828125 -3656.99169921875 3965.241455078125 -5289.021484375])
     (seq diff-weights-iter-vec)
     => (map float [-1.0780001878738403 -1.386000394821167 -1.7710000276565552 -2.2770004272460938
                    8.481000900268555 -11.308000564575195 13.001999855041504 -17.336000442504883])
     (seq bias-vec) => [0.30000001192092896 0.699999988079071 1.0 2.0]
     (seq diff-bias-vec) => [-276.3673095703125 -628.833740234375 4.345000267028809 -6.677000045776367]
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
     (seq diff-weights-vec) => [-551.3486328125 -1255.885498046875 -827.0230102539062 -1883.8282470703125
                                2741.81591796875 -3656.99169921875 3965.241455078125 -5289.02099609375]
     (seq diff-weights-iter-vec)
     => (map float [-1.0780001878738403 -1.386000394821167 -1.7710000276565552 -2.2770004272460938
                    8.481000900268555 -11.308000564575195 13.001999855041504 -17.336000442504883])
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
      lstm-pd (lstm-fwd eng :inference :unidirectional
                        src-desc src-iter-desc src-iter-c-desc
                        weights-desc weights-desc nil nil bias-desc
                        dst-desc dst-iter-desc dst-iter-c-desc)
      lstm-no-iter-pd (lstm-fwd eng :inference :unidirectional
                                src-desc nil nil weights-desc weights-desc nil nil bias-desc
                                dst-desc dst-iter-desc dst-iter-c-desc)
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
     (execute! s lstm lstm-args) => s
     (seq src-iter-vec) => [0.0 0.0 0.0 0.0]
     (seq dst-vec) => [0.28369858860969543 0.5042773485183716 0.6443434953689575 0.8513875007629395]
     (seq dst-iter-vec) => [0.623127281665802 0.8365733027458191 0.6443434953689575 0.8513875007629395]
     (entry! dst-vec 0)
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
      lstm-fwd-pd (lstm-fwd eng :training :unidirectional
                            src-desc src-iter-desc src-iter-desc
                            weights-desc weights-desc nil nil bias-desc
                            dst-desc dst-iter-desc dst-iter-desc)
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
      lstm-bwd-pd (lstm-bwd eng lstm-fwd-pd :unidirectional src-desc src-iter-desc src-iter-desc
                            [bwd-weights-desc bwd-weights-desc nil nil] bias-desc
                            dst-desc dst-iter-desc dst-iter-desc
                            src-desc src-iter-desc src-iter-desc
                            [bwd-weights-desc bwd-weights-desc nil nil] bias-desc
                            dst-desc dst-iter-desc dst-iter-desc)
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

     (execute! s lstm-fwd lstm-fwd-args) => s
     (seq dst-vec) => [0.28369858860969543 0.5042773485183716 0.6443434953689575 0.8513875007629395]
     (seq src-vec) => (map float [2.0 3.0 0.2 0.3])
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
     => (map float [-0.2533453106880188 0.3088344633579254 0.0 0.0 -0.1811492145061493 0.15289485454559326
                    -0.019089847803115845 0.022491352632641792 -0.3800179660320282 0.4632517099380493
                    0.0 0.0 -0.27172383666038513 0.2293422818183899 -0.028634771704673767
                    0.03373702988028526 0.21738360822200775 -0.35135138034820557 0.0 0.0
                    1.3538494110107422 -0.8331964015960693 0.18014927208423615 -0.13476626574993134
                    0.2654154598712921 -0.42898398637771606 0.0 0.0 1.7426719665527344
                    -1.0620708465576172 0.23835836350917816 -0.17638367414474487])
     (seq diff-weights-iter-vec)
     => (map float [0.0 0.0 0.0 0.0 -0.25559931993484497 0.3252394199371338 -0.04460345581173897
                    0.06503181159496307 0.0 0.0 0.0 0.0 -0.31207510828971863 0.3971024751663208
                    -0.05445878207683563 0.07940087467432022 0.0 0.0 0.0 0.0 0.3358241319656372
                    -0.1676657348871231 0.06891565024852753 -0.04433605819940567 0.0 0.0 0.0 0.0
                    0.5969310402870178 -0.2980276942253113 0.12249832600355148 -0.07880783081054688]))))

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
      gru-pd (gru-fwd eng :inference :unidirectional
                      src-desc src-iter-desc weights-desc weights-desc bias-desc
                      dst-desc dst-iter-desc)
      gru-no-iter-pd (gru-fwd eng :inference :unidirectional
                              src-desc nil weights-desc weights-desc bias-desc
                              dst-desc dst-iter-desc)
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
     (execute! s gru gru-args) => s
     (seq src-iter-vec) => [0.0 0.0 0.0 0.0]
     (seq dst-vec) => [0.20008479058742523 0.10380759835243225 0.3499049246311188 0.19589270651340485]
     (seq dst-iter-vec) => [0.17513683438301086 0.08930931240320206 0.3499049246311188 0.19589270651340485]
     (entry! dst-vec 0)
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
      gru-fwd-pd (gru-fwd eng :training :unidirectional src-desc src-iter-desc
                          weights-desc weights-desc bias-desc dst-desc dst-iter-desc)
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
      gru-bwd-pd (gru-bwd eng gru-fwd-pd :unidirectional src-desc src-iter-desc
                          bwd-weights-desc bwd-weights-desc bias-desc
                          dst-desc dst-iter-desc
                          src-desc src-iter-desc
                          bwd-weights-desc bwd-weights-desc bias-desc
                          dst-desc dst-iter-desc)
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

     (execute! s gru-fwd gru-fwd-args) => s
     (seq dst-vec) => [0.1986464262008667 0.10329369455575943 0.3485546410083771 0.19498808681964874]
     (seq dst-iter-vec) => [0.20356373488903046 0.161529079079628 0.3485546410083771 0.19498808681964874]
     (seq src-vec) => (map float [2.0 3.0 0.2 0.3])
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
     => (map float [0.25396845 -0.5963348 0.0 0.0 -0.10929067 0.10449476 0.38095266 -0.8945022 0.0 0.0
                    -0.163936 0.15674214 -0.19092661 0.19530042 1.6800186E-5 1.4439758E-5 0.11845594
                    -0.012657661 -0.15150145 0.15497208 1.3331051E-5 1.1458037E-5 0.09399552 -0.010043932])
     (seq diff-weights-iter-vec)
     => (map float [0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 -0.07661617547273636
                    0.0749531239271164 1.639435868128203E-5 1.4090949662204366E-5 0.04181811586022377
                    -0.0039000778924673796 -0.03983946889638901 0.03897470235824585 8.52486482472159E-6
                    7.327119874389609E-6 0.02569284476339817 -0.002396188909187913] ))))
