;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.diamond.internal.bnns.core-test
  (:require [midje.sweet :refer [facts throws => roughly truthy just]]
            [uncomplicate.commons.core :refer [with-release bytesize size release view]]
            [uncomplicate.fluokitten.core :refer [extract]]
            [uncomplicate.clojure-cpp
             :refer [pointer put! put-float! get-float byte-pointer float-pointer
                     put-entry! get-entry position pointer-seq position! null?
                     fill! capacity!]]
            [uncomplicate.neanderthal
             [core :refer [zero raw nrm2 entry! entry transfer!]]
             [native :refer [fv]]
             [block :refer [buffer]]
             [math :refer [sqr sqrt]]]
            [uncomplicate.diamond.internal.bnns
             [core :refer :all]
             [protocols :as api]])
  (:import clojure.lang.ExceptionInfo))

(facts "NDA descriptor memory safety strides."
       (with-release [strds [120 1 20 4]
                      dimensions [2 3 4 5]
                      nd (nda-desc dimensions :float :abcd strds)
                      tz (tensor nd)]
         (data-type nd) => :float
         (release nd) => true
         (release nd) => true
         (count (pointer-seq (data tz))) => 199
         (release tz) => true
         (release tz) => true
         (data tz) => nil))

(facts "TD descriptor by strides."
       (with-release [strds [120 1 20 4]
                      dimensions [2 3 4 5]
                      td (tensor-desc dimensions :float strds)]
         (data-type td) => :float
         (rank td) => (count dimensions)
         (dims td) => dimensions
         (strides td) => strds
         (bytesize td) => 796
         (bytesize (tensor-desc [2 3] :float [0 0])) => 4
         (tensor-desc [1 1] :f64 [1 1]) => (throws ExceptionInfo)
         (data-type (tensor-desc [1 1])) => :float
         (dims (tensor-desc [2 3])) => [2 3]
         (strides (tensor-desc [2 3])) => [3 1]
         (strides (tensor-desc [2 3] :float [3 1])) => [3 1]))

(facts "NDA descriptor by strides."
       (with-release [strds [120 1 20 4]
                      dimensions [2 3 4 5]
                      nda (nda-desc dimensions :float strds)
                      nda23 (nda-desc [2 3] :float :ab [3 1])]
         (data-type nda) => :float
         (layout nda) => :abcd
         (rank nda) => (count dimensions)
         (dims nda) => dimensions
         (strides nda) => strds
         (bytesize nda) => 796
         (bytesize (nda-desc [2 3] :uint16 [0 0])) => 12
         (bytesize (nda-desc [2] :uint16 [4])) => 10
         (bytesize (nda-desc [2] :uint16 [0])) => 4
         (nda-desc [1 1] :f64 [1 1]) => (throws ExceptionInfo)
         (data-type (nda-desc [1 1])) => :float
         (bytesize (nda-desc [2 3 4 5])) => 480
         (strides (nda-desc [2 3 4 5])) => [60 20 5 1]
         (dims (nda-desc [2 3])) => [2 3]
         (strides nda23) => [3 1]
         (data-type nda23) => :float
         (layout nda23) => :ab
         (pointer-seq (api/strides* nda23)) => [1 3]
         (bytesize nda23) => 24
         (strides (nda-desc [2 3] :float [3 1])) => [3 1]))

(facts "Basic tensor memory integration."
       (with-release [n-dsc (nda-desc [2 3 4 5] :float :abcd)
                      t-dsc (tensor-desc [2 3 4 5] :float)
                      n-tz (tensor n-dsc)
                      t-tz (tensor t-dsc)
                      t-view (view t-tz)
                      large-dsc (nda-desc [50000000] :float :x)]
         (pointer n-tz) => truthy
         (pointer t-tz) => truthy
         (bytesize n-dsc) => (bytesize t-dsc)
         (bytesize n-tz) => (bytesize t-tz)
         (size large-dsc) => 50000000
         (null? (data large-dsc)) => true
         (null? (data n-dsc)) => true
         (null? (data t-dsc)) => true
         (count (pointer-seq (data t-tz))) => 120
         (api/data* t-view nil) => t-view
         (count (pointer-seq (data t-view))) => 0))

(facts "Test activation forward."
       (with-release [activ (activation :linear 2.0)
                      nda (nda-desc [3] :float :x [1])
                      in-tz (tensor nda)
                      out-tz (tensor nda)
                      activ-params (activation-params activ in-tz out-tz)
                      activ-layer (layer activ-params)]
         (pointer-seq (pointer in-tz)) => [0.0 0.0 0.0]
         (pointer-seq (pointer out-tz)) => [0.0 0.0 0.0]
         (put! (pointer in-tz) [-1 0 1])
         (pointer-seq (pointer in-tz)) => [-1.0 0.0 1.0]
         activ-params => truthy
         activ-layer => truthy
         (apply-filter activ-layer in-tz out-tz)
         (pointer-seq (pointer in-tz)) => [-1.0 0.0 1.0]
         (pointer-seq (pointer out-tz)) => [-2.0 0.0 2.0]))

(facts "Test activation ReLU forward operation."
       (with-release [nda (nda-desc [2 3 4 5] :float :nchw)
                      in-tz (tensor nda)
                      activ (activation :relu)
                      relu-params (activation-params activ in-tz)
                      relu-layer (layer relu-params)]
         (put! (pointer in-tz) [-1 0 1])
         (apply-filter relu-layer in-tz) => relu-layer
         (take 4 (pointer-seq (pointer in-tz))) => [0.0 0.0 1.0 0.0]))

(facts "Test activation ReLU bakward operation."
       (with-release [nda (nda-desc [20 3] :float :nc)
                      in-tz (tensor nda)
                      out-tz (tensor nda)
                      activ (activation :relu)
                      relu-params (activation-params activ nda)
                      relu-layer (layer relu-params)]
         (put! (pointer in-tz) [-100 20])
         (apply-filter relu-layer in-tz out-tz) => relu-layer
         (take 2 (pointer-seq (pointer out-tz))) => [0.0 20.0]
         (put! (pointer in-tz) (range 2 8))
         (apply-filter-backward relu-layer in-tz out-tz in-tz) => relu-layer
         (take 6 (pointer-seq (pointer in-tz))) => [0.0 3.0 0.0 0.0 0.0 0.0]))

(facts "NDArray offset operation."
       (with-release [nda (nda-desc [2 3 4 5] :float :abcd)
                      buf (byte-pointer (+ 8 (bytesize nda)))
                      in-tz (tensor nda buf)
                      out-tz (tensor nda)
                      activ (activation :relu)
                      activ-params (activation-params activ nda nda) ;;It appears to me that BNNS NDA controls the data pointer and you can't do much with offsetting it through the buff pointer itself.
                      activ-layer (layer activ-params)]
         (put-float! buf 0 100)
         (put-float! buf 1 20)
         (put-float! buf 2 -200)
         (put-float! buf 3 300)
         (put-float! buf 120 -400)
         (put-float! buf 121 -500)
         (position! buf 4)
         (apply-filter activ-layer in-tz out-tz)
         (get-entry (pointer out-tz) 0) => 100.0
         (api/data* in-tz buf);; We have to explicitly set the new offset data.
         (apply-filter activ-layer in-tz out-tz)
         (get-entry (pointer out-tz) 0) => 20.0
         (get-entry (pointer out-tz) 1) => 0.0
         (position! buf 489) => (throws IndexOutOfBoundsException)))

(facts "Test copy and transform."
       (with-release [nda (nda-desc [3 2] :float :column)
                      in-tz (tensor nda)
                      out-tz (tensor nda)
                      nda-row (nda-desc [3 2] :byte :row)
                      nda-row1 (nda-desc [3 2] :byte :column [4 2])
                      out-tz-row (tensor nda-row)
                      out-tz-row1 (tensor nda-row1)
                      activ (activation :identity)
                      activ-params (activation-params activ nda nda-row)
                      activ-layer (layer activ-params)
                      activ-params1 (activation-params activ nda nda-row1)
                      activ-layer1 (layer activ-params1)]
         (pointer-seq (pointer in-tz)) => [0.0 0.0 0.0 0.0 0.0 0.0]
         (pointer-seq (pointer out-tz)) => [0.0 0.0 0.0 0.0 0.0 0.0]
         (pointer-seq (pointer out-tz-row)) => [0 0 0 0 0 0]
         (put! (pointer in-tz) [-1 0 1 -2 10 2])
         (pointer-seq (pointer in-tz)) => [-1.0 0.0 1.0 -2.0 10.0 2.0]
         (copy in-tz out-tz) => out-tz
         (pointer-seq (pointer in-tz)) => [-1.0 0.0 1.0 -2.0 10.0 2.0]
         (pointer-seq (pointer out-tz)) => [-1.0 0.0 1.0 -2.0 10.0 2.0]
         (copy in-tz out-tz-row) => (throws ExceptionInfo)
         (copy in-tz out-tz-row1) => (throws ExceptionInfo)
         ;; Default strides don't work
         (apply-filter activ-layer in-tz out-tz-row) => (throws IllegalArgumentException)
         ;; It happily accepts whatever explicit strides and works with them
         (apply-filter activ-layer1 in-tz out-tz-row1)
         (pointer-seq (pointer out-tz-row1)) => [-1 0 0 0 1 0 -2 0 10 0 2]
         (put! (pointer in-tz) [-10 0 10 -2 10 2])
         (with-release [activ-layer2 (with-release [activ-params2 (activation-params activ nda nda-row1)]
                                       (layer activ-params2))]
           (apply-filter activ-layer2 in-tz out-tz-row1))
         (pointer-seq (pointer out-tz-row1)) => [-10 0 0 0 10 0 -2 0 10 0 2]))

(facts "Test 3d transform."
       (with-release [nda1 (nda-desc [3 2 1] :float :abc [3 1 1])
                      nda2 (nda-desc [3 2 1] :byte :abc [7 2 1])
                      in-tz (tensor nda1)
                      out-tz (tensor nda2)
                      activ (activation :identity)
                      activ-params (activation-params activ nda1 nda2)
                      activ-layer (layer activ-params)]
         (apply-filter activ-layer in-tz out-tz)))

(facts "Test arithmetic."
       (with-release [activ (activation :linear 2.0)
                      nda (nda-desc [3] :float :x)
                      in-tz (tensor nda)
                      out-tz (tensor nda)
                      activ-params (activation-params activ nda nda)
                      activ-layer (layer activ-params)
                      arith (arithmetic in-tz :constant out-tz :constant)
                      arith-params (arithmetic-params :sqr arith activ)
                      arith-layer (layer arith-params)]
         (pointer-seq (pointer in-tz)) => [0.0 0.0 0.0]
         (pointer-seq (pointer out-tz)) => [0.0 0.0 0.0]
         (put! (pointer in-tz) [-1 0 1])
         (pointer-seq (pointer in-tz)) => [-1.0 0.0 1.0]
         activ-params => truthy
         activ-layer => truthy
         (apply-filter activ-layer in-tz out-tz)
         (pointer-seq (pointer in-tz)) => [-1.0 0.0 1.0]
         (pointer-seq (pointer out-tz)) => [-2.0 0.0 2.0]
         (apply-arithmetic arith-layer in-tz out-tz)
         (pointer-seq (pointer in-tz)) => [-1.0 0.0 1.0]
         (pointer-seq (pointer out-tz)) => [2.0 0.0 2.0]))

(facts "Inner product forward 1D."
       (with-release [src-desc (nda-desc [1] :float :x)
                      weights-desc (nda-desc [1 1] :float :row)
                      bias-desc (nda-desc [1] :float :x)
                      dst-desc (nda-desc [1] :float :x)
                      activ (activation :linear 2.0)
                      src-tz (tensor src-desc)
                      weights-tz (tensor weights-desc)
                      bias-tz (tensor bias-desc)
                      dst-tz (tensor dst-desc)
                      _ (put! (pointer weights-tz) [2.0]) ;;TODO Has to be set before fc-params!
                      _ (put! (pointer bias-tz) [1])
                      fc-params (fully-connected-params activ src-desc weights-tz bias-tz dst-desc)
                      fc-layer (layer fc-params)]
         (put! (pointer src-tz) [1.0])
         (pointer-seq (capacity! (float-pointer (.data (api/b-desc fc-params))) 1)) => [1.0]
         (null? (.data (api/b-desc fc-params))) => false
         (put! (capacity! (float-pointer (.data (api/b-desc fc-params))) 1) [2000]) ;; doesn't do anything for fc-layer!
         (pointer-seq (capacity! (float-pointer (.data (api/b-desc fc-params))) 1)) => [2000.0]
         (apply-filter fc-layer src-tz dst-tz)
         (pointer-seq (pointer src-tz)) => [1.0]
         (pointer-seq (pointer dst-tz)) => [6.0]
         (apply-filter (layer fc-params) src-tz dst-tz) ;; Layer gets its own copy of data, ovbiously!
         (pointer-seq (pointer src-tz)) => [1.0]
         (pointer-seq (pointer dst-tz)) => [4004.0]))

#_(facts "Inner product backward 2D." TODO this deprecated functionality is rather wierd, and deprecated in BNNS for a reason...
       (with-release [src-desc (nda-desc [2 2] :float :x)
                      weights-desc (nda-desc [3 2] :float :io)
                      bias-desc (nda-desc [3] :float :x)
                      dst-desc (nda-desc [2 3] :float :x)
                      activ (activation :linear 1.0)
                      src-tz (tensor src-desc)
                      weights-tz (tensor weights-desc)
                      bias-tz (tensor bias-desc)
                      dst-tz (tensor dst-desc)
                      src-diff-tz (tensor src-desc)
                      weights-diff-tz (tensor weights-desc)
                      bias-diff-tz (tensor bias-desc)
                      dst-diff-tz (tensor dst-desc)
                      _ (put! (pointer weights-tz) [0.1 0.4 0.2 0.5 0.3 0.6]) ;; row-oriented
                      _ (put! (pointer bias-tz) [1 2 3])
                      fc-params (fully-connected-params activ src-desc weights-tz bias-tz dst-desc)
                      fc-layer (layer fc-params)]
         fc-layer =not=> nil
         (put! (pointer src-tz) [2 3 1 1])
         (apply-filter-forward fc-layer src-tz dst-tz)
         (pointer-seq (pointer src-tz)) => [2.0 3.0 1.0 1.0]
         (pointer-seq (pointer dst-tz)) => (map float [2.4 3.9 5.4 1.5 2.7 3.9])
         (put! (pointer dst-diff-tz) [0.2 0.3 0.8 1 1 1])
         (put! (pointer weights-diff-tz) [1 0 0 0 0 0])
         (put! (pointer bias-diff-tz) [5 5 5])
         (apply-filter-backward fc-layer src-tz src-diff-tz dst-tz dst-diff-tz
                                weights-diff-tz bias-tz)
         (apply-filter-backward fc-layer src-tz src-diff-tz dst-tz dst-diff-tz
                                weights-diff-tz bias-tz)
         (pointer-seq (pointer src-diff-tz)) => (map float [0.320000022649765 0.7100000381469727 0.6000000238418579 1.5])
         (pointer-seq (pointer dst-diff-tz)) => (map float [0.2 0.3 0.8 1.0 1.0 1.0])
         (pointer-seq (pointer weights-diff-tz)) => [1.4 1.6 1.6 1.9000000953674316 2.6 3.4]
         (pointer-seq (pointer bias-tz)) => (map float [1.2 1.3 1.8])))

(facts "Inner product forward 4D squashed to 1D."
       (with-release [src-desc (nda-desc [9] :float :x)
                      weights-desc (nda-desc [2 9] :float :row)
                      bias-desc (nda-desc [2] :float :x)
                      dst-desc (nda-desc [2] :float :x)
                      activ (activation :linear 1.0)
                      src-tz (tensor src-desc)
                      weights-tz (tensor weights-desc)
                      bias-tz (tensor bias-desc)
                      dst-tz (tensor dst-desc)
                      fc-params (fully-connected-params activ src-desc weights-tz bias-tz dst-desc)]
         (put! (pointer weights-tz) (take (size weights-desc) (range 0 1 0.02)))
         (put! (pointer bias-tz) [0.3 0.7])
         (put! (pointer src-tz) (take 9 (range 1 2 0.1)))
         (apply-filter (layer fc-params) src-tz dst-tz)
         (pointer-seq (pointer dst-tz)) => (map float [1.428 4.0959997])))

(facts "Convolution forward 1D."
       (with-release [src-desc (nda-desc [1 1 1] :float :chw)
                      weights-desc (nda-desc [1 1 1 1] :float :oihw)
                      bias-desc (nda-desc [1] :float :x)
                      dst-desc (nda-desc [1 1 1] :float :chw)
                      activ (activation :linear 1.0)
                      src-tz (tensor src-desc)
                      weights-tz (tensor weights-desc)
                      bias-tz (tensor bias-desc)
                      dst-tz (tensor dst-desc)
                      conv-params (convolution-params activ
                                                      src-desc weights-tz bias-tz dst-desc
                                                      [1 1] [0 0])]
         (put! (pointer src-tz) [2.0])
         (put! (pointer weights-tz) [-2])
         (put! (pointer bias-tz) [1])
         (apply-filter (layer conv-params) src-tz dst-tz)
         (pointer-seq (pointer dst-tz)) => [-3.0]))

;;TODO as we can see, shape is backwards in BNNS...
(facts "Convolution forward."
       (with-release [src-desc (nda-desc [1 4 4] :float :chw)
                      weights-desc (nda-desc [2 1 3 3] :float :oihw)
                      bias-desc (nda-desc [2] :float :x)
                      dst-desc (nda-desc [2 2 2] :float :chw)
                      activ (activation :linear 1.0)
                      src-tz (tensor src-desc)
                      weights-tz (tensor weights-desc)
                      bias-tz (tensor bias-desc)
                      dst-tz (tensor dst-desc)
                      conv-params (convolution-params activ
                                                      src-desc weights-tz bias-tz dst-desc)]
         (put! (pointer src-tz) [0 43 3 30 0 98 0 0 7 38 0 0 19 20 175 50])
         (put! (pointer weights-tz) [-2 0 1 0 1 0 -1 -2 0
                                     -2 0 1 0 1 0 -1 -2 0])
         (put! (pointer bias-tz) [0.5 0.5 ])
         (apply-filter (layer conv-params) src-tz dst-tz)
         (pointer-seq (pointer dst-tz)) => [18.5 -93.5 -20.5 -565.5
                                            18.5 -93.5 -20.5 -565.5]))
