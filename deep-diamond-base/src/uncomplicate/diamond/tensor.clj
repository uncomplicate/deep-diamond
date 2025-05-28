;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.diamond.tensor
  "Contains type-agnostic general tensor functions. Does not contain functions related
  to deep neural networks (DNN); see the [[dnn]] namespace for these.

  ### How to use

      (ns test
        (:require [uncomplicate.diamond
                   [tensor :refer :all]
                   [native :refer :all]]))

  ### Examples

  The best and most accurate examples can be found in the
  [comprehensive test suite](https://github.com/uncomplicate/deep-diamond/tree/master/test/uncomplicate/diamond),
  [full examples](https://github.com/uncomplicate/deep-diamond/tree/master/test/uncomplicate/diamond/functional),
  [core tensor examples](https://github.com/uncomplicate/deep-diamond/blob/master/test/uncomplicate/diamond/tensor_test.clj)
  [core DNN examples](https://github.com/uncomplicate/deep-diamond/blob/master/test/uncomplicate/diamond/dnn_test.clj)
  [internal CPU engine tests](https://github.com/uncomplicate/deep-diamond/tree/master/test/uncomplicate/diamond/internal/dnnl),
  [internal GPU engine tests](https://github.com/uncomplicate/deep-diamond/tree/master/test/uncomplicate/diamond/internal/cudnn),


  There are quite a few tutorials [on my blog dragan.rocks](http://dragan.rocks).

  For the comprehensive real-world examples, with detailed tutorials and guides, see the
  [Interactive Programming for Artificial intelligence book series](aiprobook.com), and specifically
  the [Deep Learning for Programmers](https://aiprobook.com/deep-learning-for-programmers) book.

    ### Cheat Sheet

  * Default engine (factory) binding: [[*diamond-factory*]], [[with-diamond]]. All functions can also receive
    engine through parameters.

  * Tensor descriptor: [[desc]], [[shape]], [[layout]], [[data-type]],

  * Create : [[tensor]], [[transformer]], [[batcher]], [[shuffler]]

  * Inspect, transform, and manage tensors: [[view-tz]], [[revert]], [[input]], [[output]],
  [[connector]], [[batch-size]], [[offset!]].


  Tensors support typical core Fluokitten functions.
  "
  (:require [uncomplicate.commons
             [core :refer [release let-release Info view]]
             [utils :refer [dragan-says-ex cond-into]]]
            [uncomplicate.fluokitten.protocols :refer [Magma Monoid Applicative Functor pure]]
            [uncomplicate.diamond.internal
             [protocols :as api]
             [utils :refer [default-strides]]]))

(def ^:dynamic *diamond-factory*
  "The default factory binding. Most polymorphic functions that have factory
  as a parameter, have a version that uses this binding. The default is `nil`, though, and the
  user has to require a namespace that initializes and binds the default engine. The most common
  way is to require [[uncomplicate.diamond.native]], which binds the DNNL based CPU engine as the default,
  but you are free to provide another implementation of your own, either by this root binding,
  or through Clojure's dynamic binding through [[with-diamond]].

  Please see [[dnnl-tensor-test]](https://github.com/uncomplicate/deep-diamond/blob/master/test/uncomplicate/diamond/internal/dnnl/dnnl_tensor_test.clj)
  for examples.
  "
  nil)

(defmacro with-diamond
  "Dynamically binds a factory created by `factory-fn` with parameters
  `params`, and evaluates `body` in the context of that factory.

  See [[*diamond-factory]].
  "
  [factory-fn params & body]
  `(binding [*diamond-factory* (~factory-fn ~@params)]
     (try ~@body
          (finally (release *diamond-factory*)))))

;; ====================== Public protocols =====================================

(defprotocol TensorDescriptor
  "Generic tensor descriptor. Most Clojure and Java data types and structures (such as numbers,
  collections, persistent structures, etc. can be used as tensor descriptors). Technology-specific
  engines often provide their own implementation internally. Please see Deep Diamond tests, examples,
  and the [Deep Learning for Programmers](https://aiprobook.com/deep-learning-for-programmers) book."
  (shape [tz] "Tensor shape, as a vector ([2 3 4]).")
  (layout [tz] "Tensor layout, as a keyword (:nchw).")
  (data-type [tz] "Data type of tensor entries (:float)."))

(extend-type java.util.Collection
  TensorDescriptor
  (shape [this]
    (vec this))
  (data-type [_]
    nil)
  (layout [_]
    nil))

(extend-type java.lang.Number
  TensorDescriptor
  (shape [this]
    [this])
  (data-type [_]
    nil)
  (layout [_]
    [1]))

(extend-type clojure.lang.IPersistentVector
  TensorDescriptor
  (shape [this]
    this)
  (data-type [_]
    nil)
  (layout [_]
    nil))

(extend-type java.util.Map
  TensorDescriptor
  (shape [this]
    (get this :shape []))
  (data-type [this]
    (get this :data-type))
  (layout [this]
    (or (get this :layout) (get this :format) (get this :strides))))

(defprotocol TensorContainer
  "An object that can be viewed as a tensor."
  (view-tz [this] [this sub] "View the object's underlying data through a (sub)tensor"))

(defprotocol Revert
  "A directed transformation whose direction can be reverted."
  (revert [this] "Revert the direction of the transformation"))

(defprotocol Transfer
  "An operation that transfers input to output."
  (input [this] "The input tensor of the Transfer.")
  (output [this] "The output tensor of the Transfer."))

(defprotocol ConnectorCreator
  "An object that can create a connection and provide its state in another shape,
  or gets its state from an object with another shape."
  (connector [in out] "Creates a connector between `in` and `out` descriptor or tensor."))

(extend-type clojure.lang.Keyword
  ConnectorCreator
  (connector [in-format out]
    (let [in-out (input out)
          fact (api/diamond-factory in-out)
          in-desc (api/create-tensor-desc fact (shape (input in-out)) (data-type (input in-out)) in-format)]
      (if (= (layout in-desc) (layout in-out))
        (do (release in-desc)
            (view (output out)))
        (let-release [in-tz (api/create-tensor fact in-desc (api/batch-index (output out)) false)]
          (api/create-transformer fact in-tz (view (output out))))))))

;; =============================================================================

(defrecord TensorDescriptorImpl [shape data-type layout]
  Info
  (info [this]
    {:class (class this)
     :device :cpu
     :shape shape
     :data-type data-type
     :layout layout})
  Monoid
  (id [_]
    (let [zero-vec (vec (repeat (count shape) 0))]
      (TensorDescriptorImpl. zero-vec data-type zero-vec)))
  Applicative
  (pure [x v]
    (let [v-vec (vec (repeat (count shape) v))]
      (TensorDescriptorImpl. v-vec data-type v-vec)))
  (pure [x v vs]
    (let [vs (into (into [v] vs) (repeat (- (count shape) (count vs) 1) 0))]
      (TensorDescriptorImpl. vs data-type (default-strides vs))))
  Functor
  (fmap [x f]
    (f x))
  (fmap [x f xs]
    (apply f x xs))
  TensorDescriptor
  (shape [_]
    shape)
  (data-type [_]
    data-type)
  (layout [_]
    layout)
  ConnectorCreator
  (connector [in-desc out]
    (connector (api/create-tensor-desc (api/diamond-factory out) in-desc) out)))

(defn desc
  "Creates a general, technology-agnostic, tensor descriptor.

  The required parameter `shape` is a vector of tensor dimensions.
  Optionally, `type` and `layout` can be provided as keywords, that
  are later transformed to appropriate internal implementations
  supported by specific backends. Alternatively, `layout` might be
  provided as a vector of offsets.

  Examples:

      (desc [2 3 2]) => {:shape [2 3 2], :data-type nil, :layout nil}
      (desc [2 3] :nc) => {:shape [2 3], :data-type nil, :layout :nc}
      (desc [2 3 2 4] :nhwc) => {:shape [2 3 2 4], :data-type nil, :layout :nhwc}
  "
  ([shape type layout]
   (if (and (vector? shape)
            (or (keyword? type) (class? type) (nil? type))
            (or (and (vector? layout) (= (count shape) (count layout)))
                (and (keyword? layout) (= (count shape) (count (name layout))))
                (nil? layout)))
     (->TensorDescriptorImpl shape type layout)
     (dragan-says-ex "The requested descriptor would most likely be inconsistent."
                     {:shape shape :type type :layout layout :errors
                      (cond-into []
                                 (not (vector? shape)) "shape is not a vector"
                                 (not (or (keyword? type) (symbol? type) (nil? type)))
                                 "type is not a keyword, class, or nil"
                                 (not (or (vector? layout) (keyword? layout) (nil? layout)))
                                 "layout is not a keyword, vector, or nil"
                                 (and (vector? layout) (not= (count shape) (count layout)))
                                 "shape and layout must have the same length"
                                 (and (keyword? layout) (not= (count shape) (count (name layout))))
                                 "shape and layout must have the same length")})))
  ([shape layout]
   (desc shape nil layout))
  ([tdesc]
   (desc (shape tdesc) (data-type tdesc) (layout tdesc))))

(defn default-desc
  "Creates a general, technology-agnostic, tensor descriptor
  with default contiguous stride appropritate for input descriptors' shape.
  "
  ([shape type]
   (desc shape type (default-strides shape)))
  ([tdesc]
   (default-desc (shape tdesc) (data-type tdesc))))

(defn tensor
  "Creates a technology-specific tensor.

  The backend is determined by `tz-factory`, while the structure
  is provided either from a general or technology specific descriptor
  `desc`, or from directly provided descriptor parameters `shape`,
  `type`, and `layout`. If `tz-factory` is not provided, it is
  taken from the `*diamond-factory*` binding, which is by default
  [[internal/dnnl]].

  Examples:

      (tensor {:shape [2 3] :data-type :float :layout :nc})
      =>
      {:shape [2 3], :data-type :float, :layout [3 1]}
      [   0.00    0.00    0.00    ⋯      0.00    0.00 ]

      (tensor (desc [2 3] :float :nc))
      =>
      {:shape [2 3], :data-type :float, :layout [3 1]}
      [   0.00    0.00    0.00    ⋯      0.00    0.00 ]

      (tensor [2 3] :float :nc)
      =>
      {:shape [2 3], :data-type :float, :layout [3 1]}
      [   0.00    0.00    0.00    ⋯      0.00    0.00 ]
  "
  ([tz-factory shape type layout batch-index]
   (if (and (sequential? shape) (<= 0 (long batch-index) (dec (count shape))))
     (let [fact (api/diamond-factory tz-factory)]
       (api/create-tensor fact (api/create-tensor-desc fact shape type layout) batch-index true))
     (dragan-says-ex "Incompatible tensor constructor arguments."
                     {:shape (class shape) :batch-index batch-index :errors
                      (cond-into []
                                 (not (sequential? shape)) "shape is not sequential"
                                 (<= 0 (long batch-index) (dec (count shape))) "batch-index is not within bounds of shape dimension")})))
  ([tz-factory shape type layout]
   (tensor tz-factory shape type layout
           (if (and (keyword? layout) (clojure.string/includes? (name layout) "t")) 1 0)))
  ([shape type layout]
   (tensor *diamond-factory* shape type layout))
  ([tz-factory desc]
   (let [fact (api/diamond-factory tz-factory)
         layout (layout desc)]
     (api/create-tensor fact (api/create-tensor-desc fact desc)
                        (if (and (keyword? layout) (clojure.string/includes? (name layout) "t")) 1 0)
                        true)))
  ([desc]
   (tensor *diamond-factory* desc)))

(defn offset!
  "Destructively moves tensor's beginning to a different place in the underlying memory."
  [tz ^long n]
  (api/offset tz n))

(defn transformer
  "Creates a function optimized for transferring data from tensor
  `x` to tensor `y`.

  Example:

      (def t1 (tensor [2 3] :float :nc))
      (transfer! (range) t1)
      (def t2 (tensor [3 2] :float :cn))
      (def tr-1-2 (transformer t1 t2))

      (tr-1-2)
      =>
      {:shape [2 3], :data-type :float, :layout [1 2]}
      [   0.00    3.00    1.00    ⋯      2.00    5.00 ]

      t2
      =>
      {:shape [2 3], :data-type :float, :layout [1 2]}
      [   0.00    3.00    1.00    ⋯      2.00    5.00 ]
  "
  [x y]
  (api/create-transformer (api/diamond-factory x) x y))

(defn batch-size
  "Determines batch size from tensor or descriptor's shape, depending on the context.
  Usually batch size is the first shape entry, but in some cases (RNN for example) it
  may be the second entry, or even something else.
  "
  ^long [x]
  ((shape x) (api/batch-index x)))

(defn batcher
  "Creates a function that can transfer mini-batches of tensor `x` to tensor `y`.

  Useful for, but not limited to, cycling through mini-batches
  during stochastic gradient descent. Assumes that the input data has
  already been shuffled (but does not depend on it technically).

  Example:

      (def t1 (tensor [2 3] :float :nc))
      (def t2 (tensor [1 3] :float :cn))
      (transfer! (range) t1)
      (def bat (batcher t1 t2))

      (bat)
      =>
      {:shape [1 3], :data-type :float, :layout [1 1]}
      [   0.00    1.00    2.00 ]

      (bat 1)
      =>
      {:shape [1 3], :data-type :float, :layout [1 1]}
      [   3.00    4.00    5.00 ]
  "
  ([x y]
   (batcher x y (min (long ((shape x) (api/batch-index x))) (long ((shape y) (api/batch-index y))))))
  ([x y ^long mb-size]
   (api/create-batcher (api/diamond-factory x) x y mb-size)))

(defn shuffler
  "Creates a function that can transfer randomly selected columns
  of tensor `x` to tensor `y` in mini-batches.

  Useful for, but not limited to, cycling through mini-batches during
  stochastic gradient descent. The difference from batcher is that it
  randomly selects the grouping of columns. Shuffler is generally
  slower than plain batcher; it may or may not make a difference in speed.

  Example:

      (def t1 (tensor [2 3] :float :nc))
      (def t2 (tensor [1 3] :float :cn))
      (transfer! (range) t1)
      (def bat (batcher t1 t2))

      (bat)
      =>
      {:shape [1 3], :data-type :float, :layout [1 1]}
      [   0.00    1.00    2.00 ]

      (bat 1)
      =>
      {:shape [1 3], :data-type :float, :layout [1 1]}
      [   3.00    4.00    5.00 ]
  "
  [x y]
  (api/create-shuffler (api/diamond-factory x) x y))
