;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.diamond.internal.dnnl.neanderthal-test
  (:require [midje.sweet :refer [facts throws => roughly]]
            [uncomplicate.commons [core :refer [with-release]]]
            [uncomplicate.diamond.dnn-test :refer :all]
            [uncomplicate.diamond.internal.dnnl.neanderthal :refer [neanderthal-tz-factory]])
  (:import clojure.lang.ExceptionInfo))

(with-release [fact (neanderthal-tz-factory)]
  (test-sum fact)
  (test-activation-relu fact)
  (test-activation-sigmoid fact)
  (test-inner-product-training fact)
  (test-fully-connected-inference fact)
  (test-fully-connected-transfer fact)
  (test-fully-connected-training fact)
  (test-fully-connected-training-adam fact)
  (test-fully-connected-layer-1 fact)
  (test-fully-connected-layer-2 fact)
  (test-sequential-network-linear fact)
  (test-sequential-network-detailed fact)
  (test-sequential-network-batched fact)
  (test-quadratic-cost fact)
  (test-sequential-network-sigmoid-sgd fact)
  (test-sequential-network-sigmoid-adam fact)
  ;; TODO works fine in REPL, MKL crashes in terminal (but only after the move to the new namespace for neand-tz-fact).
  ;; Not severe, since this engine is only used for testing purposes, but still, needs investigation.
  ;; It may be just some slight javacpp version incompatibility.
  ;; java: symbol lookup error: /home/dragan/.javacpp/cache/mkl-2025.2-1.5.12-linux-x86_64-redist.jar/org/bytedeco/mkl/linux-x86_64/libmkl_intel_thread.so.2: undefined symbol: __kmpc_global_thread_num
  ;;(test-gradient-descent fact)
  ;;(test-stochastic-gradient-descent-sgd fact)
  ;;(test-stochastic-gradient-descent-adam fact)
  (test-crossentropy-cost fact)
  (test-inner-product-training fact)
  (test-convolution-inference fact)
  (test-convolution-inference-relu fact)
  (test-convolution-training fact)
  (test-pooling-avg fact)
  (test-gaussian-dropout fact)
  (test-batch-normalization-inference fact)
  (test-batch-normalization-training fact)
  (test-concatenate fact)
  (test-branch fact)
  (test-network-concat fact)
  (test-network-branch-concat fact)
  (test-network-branch-concat-simplified fact)
  (test-parallel-network-solo fact)
  (test-parallel-network-concat fact)
  (test-parallel-network-nested fact)
  (test-sum fact)
  (test-split fact)
  )

;; (with-release [fact (neanderthal-tz-factory)]
;;   (bench-wide-layers fact))
;; "Elapsed time: 6235.015802 msecs"
