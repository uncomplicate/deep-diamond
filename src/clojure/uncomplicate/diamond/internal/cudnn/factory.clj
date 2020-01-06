;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.diamond.internal.cudnn.factory
  (:require [clojure.java.io :as io]
            [uncomplicate.commons
             [core :refer [Releaseable release let-release with-release]]
             [utils :refer [dragan-says-ex]]]
            [uncomplicate.fluokitten.core :refer [op]]
            [uncomplicate.clojurecuda
             [core :refer :all]
             [toolbox :refer [read-int]]]
            [uncomplicate.neanderthal
             [cuda :refer [cuda-float cuda-double]]
             [block :refer [buffer offset]]]
            [uncomplicate.neanderthal.internal.api
             :refer [FlowProvider BlockEngine Blas BlasPlus engine set-all]]
            [uncomplicate.diamond.internal.protocols
             :refer [TensorFactory DiamondFactoryProvider ContextProvider
                     NeanderthalFactoryProvider CostFactory DnnFactory]]
            [uncomplicate.diamond.internal.cudnn
             [protocols :refer [desc]]
             [core :refer [cudnn-handle get-cudnn-stream tensor-descriptor
                           ndims dims strides transform-tensor add-tensor]]
             [tensor :refer [cudnn-tensor]]])
  (:import jcuda.jcudnn.JCudnn))

(defn ^:private tensor-1d-equals [modl hstream x y]
  (with-release [equals-kernel (function modl "tensor_1d_equals")
                 eq-flag-buf (mem-alloc Integer/BYTES)]
    (let [n (first (dims x))]
      (memset! eq-flag-buf 0)
      (launch! equals-kernel (grid-1d n) hstream
               (parameters (int n)
                           (buffer x) (int (offset x)) (int (first (strides x)))
                           (buffer y) (int (offset y)) (int (first (strides y)))
                           eq-flag-buf))
      (= 0 (read-int hstream eq-flag-buf)))))

(defn ^:private tensor-2d-equals [modl hstream x y]
  (with-release [equals-kernel (function modl "tensor_2d_equals")
                 eq-flag-buf (mem-alloc Integer/BYTES)]
    (let [[n c] (dims x)
          [nx cx] (strides x)
          [ny cy] (strides y)]
      (memset! eq-flag-buf 0)
      (launch! equals-kernel (grid-2d n c) hstream
               (parameters (int n) (int c)
                           (buffer x) (int (offset x)) (int nx) (int cx)
                           (buffer y) (int (offset y)) (int ny) (int cy)
                           eq-flag-buf))
      (= 0 (read-int hstream eq-flag-buf)))))

(defn ^:private tensor-3d-equals [modl hstream x y]
  (with-release [equals-kernel (function modl "tensor_3d_equals")
                 eq-flag-buf (mem-alloc Integer/BYTES)]
    (let [[n c h] (dims x)
          [nx cx hx] (strides x)
          [ny cy hy] (strides y)]
      (memset! eq-flag-buf 0)
      (launch! equals-kernel (grid-2d n c h) hstream
               (parameters (int n) (int c) (int h)
                           (buffer x) (int (offset x)) (int nx) (int cx) (int hx)
                           (buffer y) (int (offset y)) (int ny) (int cy) (int hy)
                           eq-flag-buf))
      (= 0 (read-int hstream eq-flag-buf)))))

(defn ^:private tensor-4d-equals [modl hstream x y]
  (with-release [equals-kernel (function modl "tensor_4d_equals")
                 eq-flag-buf (mem-alloc Integer/BYTES)]
    (let [[n c h w] (dims x)
          [nx cx hx wx] (strides x)
          [ny cy hy wy] (strides y)]
      (memset! eq-flag-buf 0)
      (launch! equals-kernel (grid-3d n c h) hstream
               (parameters (int n) (int c) (int h) (int w)
                           (buffer x) (int (offset x)) (int nx) (int cx) (int hx) (int wx)
                           (buffer y) (int (offset y)) (int ny) (int cy) (int hy) (int wy)
                           eq-flag-buf))
      (= 0 (read-int hstream eq-flag-buf)))))

(defn ^:private tensor-5d-equals [modl hstream x y]
  (with-release [equals-kernel (function modl "tensor_5d_equals")
                 eq-flag-buf (mem-alloc Integer/BYTES)]
    (let [[n c d h w] (dims x)
          [nx cx dx hx wx] (strides x)
          [ny cy dy hy wy] (strides y)]
      (memset! eq-flag-buf 0)
      (launch! equals-kernel (grid-3d n c h) hstream
               (parameters (int n) (int c) (int d) (int h) (int w)
                           (buffer x) (int (offset x)) (int nx) (int cx) (int dx) (int hx) (int wx)
                           (buffer y) (int (offset y)) (int ny) (int cy) (int dy) (int hy) (int wy)
                           eq-flag-buf))
      (= 0 (read-int hstream eq-flag-buf)))))

(defn ^:private tensor-equals [modl hstream x y]
  (let [cnt (int (apply * (dims x)))]
    (if (< 0 cnt)
      (case (ndims x)
        1 (tensor-1d-equals modl hstream x y)
        2 (tensor-2d-equals modl hstream x y)
        3 (tensor-3d-equals modl hstream x y)
        4 (tensor-4d-equals modl hstream x y)
        5 (tensor-5d-equals modl hstream x y)
        (dragan-says-ex "Equals is supported only up to 5 dimensions." {:shape (dims x)}))
      (= 0 (int (apply * (dims y)))))))

(defn ^:private tensor-set [modl hstream x value]
  (with-release [set-kernel (function modl "tensor_5d_set")]
    (let [[n c d h w] (dims x)
          [nx cx dx hx wx] (strides x)]
      (launch! set-kernel (grid-3d (or n 1) (or c 1) (or h 1)) hstream
               (parameters (int (or n 0)) (int (or c 1)) (int (or d 1)) (int (or h 1)) (int (or w 1))
                           value (buffer x) (int (offset x))
                           (int (or nx 0)) (int (or cx 0)) (int (or dx 0)) (int (or hx 0)) (int (or wx 0))))
      x)))

(deftype FloatTensorEngine [handle modl hstream]
  BlockEngine
  (equals-block [_ x y]
    (tensor-equals modl hstream x y))
  Blas
  (copy [_ x y]
    (transform-tensor handle (float 1.0) x (buffer x) (* (offset y) Float/BYTES)
                      (float 0.0) y (buffer y) (* (offset y) Float/BYTES))
    y)
  (axpy [_ alpha x y]
    (add-tensor handle (float alpha) x (buffer x) (* (offset y) Float/BYTES)
                (float 1.0) y (buffer y) (* (offset y) Float/BYTES))
    y)
  BlasPlus
  (set-all [_ value x]
    (tensor-set modl hstream x (float value)))
  (axpby [_ alpha x beta y]
    (add-tensor handle (float alpha) x (buffer x) (* (offset y) Float/BYTES)
                (float beta) y (buffer y) (* (offset y) Float/BYTES))
    y))

(deftype DoubleTensorEngine [handle modl hstream]
  BlockEngine
  (equals-block [_ x y]
    (tensor-equals modl hstream x y))
  Blas
  (copy [_ x y]
    (transform-tensor handle (double 1.0) x (buffer x) (* (offset y) Double/BYTES)
                      (double 0.0) y (buffer y) (* (offset y) Double/BYTES))
    y)
  (axpy [_ alpha x y]
    (add-tensor handle (double alpha) x (buffer x) (* (offset y) Double/BYTES)
                (double 1.0) y (buffer y) (* (offset y) Double/BYTES))
    y)
  BlasPlus
  (set-all [_ value x]
    (tensor-set modl hstream x (double value)))
  (axpby [_ alpha x beta y]
    (add-tensor handle (double alpha) x (buffer x) (* (offset y) Double/BYTES)
                (double beta) y (buffer y) (* (offset y) Double/BYTES))
    y))

(deftype IntTensorEngine [handle modl hstream]
  BlockEngine
  (equals-block [_ x y]
    (tensor-equals modl hstream x y))
  Blas
  (copy [_ x y]
    (transform-tensor handle (int 1.0) x (buffer x) (* (offset y) Integer/BYTES)
                      (int 0.0) y (buffer y) (* (offset y) Integer/BYTES))
    y)
  (axpy [_ alpha x y]
    (add-tensor handle (int alpha) x (buffer x) (* (offset y) Integer/BYTES)
                (int 1.0) y (buffer y) (* (offset y) Integer/BYTES))
    y)
  BlasPlus
  (set-all [_ value x]
    (tensor-set modl hstream x (int value)))
  (axpby [_ alpha x beta y]
    (add-tensor handle (int alpha) x (buffer x) (* (offset y) Integer/BYTES)
                (int beta) y (buffer y) (* (offset y) Integer/BYTES))
    y))

(deftype LongTensorEngine [handle modl hstream]
  BlockEngine
  (equals-block [_ x y]
    (tensor-equals modl hstream x y))
  Blas
  (copy [_ x y]
    (transform-tensor handle (long 1.0) x (buffer x) (* (offset y) Long/BYTES)
                      (long 0.0) y (buffer y) (* (offset y) Long/BYTES))
    y)
  (axpy [_ alpha x y]
    (add-tensor handle (long alpha) x (buffer x) (* (offset y) Long/BYTES)
                (long 1.0) y (buffer y) (* (offset y) Long/BYTES))
    y)
  BlasPlus
  (set-all [_ value x]
    (tensor-set modl hstream x (long value)))
  (axpby [_ alpha x beta y]
    (add-tensor handle (long alpha) x (buffer x) (* (offset y) Long/BYTES)
                (long beta) y (buffer y) (* (offset y) Long/BYTES))
    y))

(deftype CUDnnFactory [ctx hstream handle master
                       neand-facts tensor-engines]
  Releaseable
  (release [_]
    (in-context ctx
                (release handle)
                (doseq [eng (vals tensor-engines)]
                  (release eng))
                (doseq [neand-fact (vals neand-facts)]
                  (release neand-fact))
                (when master
                  (when-not (= default-stream hstream)
                    (release hstream))
                  (release ctx)))
    true)
  DiamondFactoryProvider
  (diamond-factory [this]
    this)
  FlowProvider
  (flow [_]
    hstream)
  ContextProvider
  (context [_]
    ctx)
  NeanderthalFactoryProvider
  (neanderthal-factory [_ dtype]
    (get neand-facts dtype nil))
  TensorFactory
  (create-tensor-desc [this shape dtype format]
    (tensor-descriptor shape dtype format))
  (create-tensor-desc [this tz-desc]
    (desc tz-desc))
  (create-tensor [this tensor-desc init]
    (let-release [res (cudnn-tensor this tensor-desc)]
      (when init
        (set-all (engine res) 0 res))
      res))
  (create-transformer [_ in-tz out-tz]
    )
  (create-shuffler [_ src-tz dst-tz]
    )
  (create-batcher [_ src-tz dst-tz mb-size]
    )
  (create-sum [_ scale dst]
    )
  (create-sum [_ dst scale src scale-srcs]
    )
  (tensor-engine [this dtype]
    (get tensor-engines dtype nil))
  DnnFactory
  (activ-blueprint [this src-desc activ alpha beta]
    )
  (inner-product-blueprint [this src-desc dst-desc weights-type]
    )
  (fc-blueprint [this src-desc dst-desc activ alpha beta weights-type]
    )
  CostFactory
  (quadratic-cost [this prev-layer train-tz]
    )
  (mean-absolute-cost [this prev-layer train-tz]
    )
  (sigmoid-crossentropy-cost [this prev-layer train-tz]
    ))

(JCudnn/setExceptionsEnabled false)

(defn ^:private create-module [src dtype]
  (with-release [prog (compile! (program src)
                                [(str "-DTYPE=" dtype) "-arch=compute_30" "-default-device"])]
    (module prog)))

(let [src (slurp (io/resource "uncomplicate/diamond/internal/cuda/blas-plus.cu"))]

  (defn ^:private create-cudnn-factory [ctx hstream handle master]
    (in-context
     ctx
     (let-release [float-modl (create-module src "float")
                   double-modl (create-module src "double")
                   int-modl (create-module src "int")
                   long-modl (create-module src "long")
                   float-fact (cuda-float ctx hstream)
                   double-fact (cuda-double ctx hstream)
                   int-fact nil  ;;TODO
                   long-fact nil ;;TODO
                   float-engine (->FloatTensorEngine handle float-modl hstream)
                   double-engine (->DoubleTensorEngine handle double-modl hstream)
                   int-engine (->IntTensorEngine handle int-modl hstream)
                   long-engine (->LongTensorEngine handle long-modl hstream)]
       (->CUDnnFactory ctx hstream handle master
                       {:float float-fact
                        :double double-fact
                        :int int-fact
                        :long long-fact}
                       {:float float-engine
                        :double double-engine
                        :int int-engine
                        :long long-engine})))))

(defn cudnn-factory
  ([ctx hstream]
   (in-context
    ctx
    (let-release [handle (cudnn-handle hstream)
                  hstream (get-cudnn-stream handle)]
      (create-cudnn-factory ctx hstream handle false))))
  ([]
   (init)
   (let-release [ctx (context (device))]
     (in-context
      ctx
      (let-release [hstream (stream)
                    handle (cudnn-handle hstream)]
        (create-cudnn-factory ctx hstream handle true))))))
