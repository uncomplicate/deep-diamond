;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.diamond.dnn
  "Contains type-agnostic deep neural networks (DNN) functions.

  ### Examples

  The [Deep Learning for Programmers](https://aiprobook.com/deep-learning-for-programmers) book
  contains very detailed examples and explanations. Please check it out.

  The most up-to-date examples can be found in the
  [comprehensive test suite](https://github.com/uncomplicate/deep-diamond/tree/master/test/uncomplicate/diamond),
  [full examples](https://github.com/uncomplicate/deep-diamond/tree/master/test/uncomplicate/diamond/functional),
  [core tensor examples](https://github.com/uncomplicate/deep-diamond/blob/master/test/uncomplicate/diamond/tensor_test.clj)
  [core DNN examples](https://github.com/uncomplicate/deep-diamond/blob/master/test/uncomplicate/diamond/dnn_test.clj)
  [internal CPU engine tests](https://github.com/uncomplicate/deep-diamond/tree/master/test/uncomplicate/diamond/internal/dnnl),
  [internal GPU engine tests](https://github.com/uncomplicate/deep-diamond/tree/master/test/uncomplicate/diamond/internal/cudnn),

  ### Cheat Sheet

  * Basic dense layers: [[activation]], [[inner-product]], [[fully-connected]], [[dense]].

  * Convolutional layers: [[convolution]], [[convo]].

  * Recurrent layers [[rnn-op]], [[rnn]], [[abbreviate]].

  * Training optimizations: [[pooling]], [[dropout-mask]], [[dropout]], [[batch-norm]].

  * [[concatenate]], [[conc]], [[branch]], [[split]], [[sum]].

  * Training and using the network: [[cost]], [[network]], [[init!]], [[train]], [[train-shuffle]], [[infer]].
  "
  (:require [uncomplicate.commons
             [core :refer [with-release let-release release view]]
             [utils :refer [dragan-says-ex]]]
            [uncomplicate.fluokitten.core :refer [fmap foldmap]]
            [uncomplicate.neanderthal
             [core :refer [ncols transfer! view-vctr]]
             [random :refer [rand-normal! rng-state]]]
            [uncomplicate.diamond.tensor
             :refer [*diamond-factory* shape input output batcher shuffler TensorContainer
                     tensor data-type layout desc batch-size]]
            [uncomplicate.diamond.internal
             [protocols :as api]
             [network :refer [sequential-network parallel-network]]
             [utils :refer [default-strides]]]))

(defn activation
  "Creates an activation blueprint, which is also a function that can create activation
  (usually non-linear) that can then be attached to the end of a network layer. It can be
  used in many ways, from a relatively low-level structure, to the fully automatic piece
  in a description of neural network.

  Arguments:

  - `fact`: technology-specific engine factory.
  - `src-desc`: tensor descriptor (or even just a relevant part of its shape) of the activation
  input and output.
  - `activ`: keyword that determines the activation function (:relu, :elu, etc.).
  See activation functions supported by DNNL ([[uncomplicate.diamond.internal.dnnl.constants/dnnl-eltwise-alg-kind]]),
  and cuDNN ([[uncomplicate.diamond.internal.cudnn.constants/cudnn-activation-mode]]).
  Keywords are the same, but not all keywords are supported by all backends, in general.
  - `alpha`: the first scalar constant (if supported by the chosen `activ`).
  - `beta`: the second scalar constant (if supported by the chosen `activ`).

  Most of these arguments can be automatically inferred when this blueprint is used
  in a DNN DSL in the context of a network.

  See examples in [dnn-test](https://github.com/uncomplicate/deep-diamond/blob/master/test/uncomplicate/diamond/dnn_test.clj),
  and [functional-tests](https://github.com/uncomplicate/deep-diamond/tree/master/test/uncomplicate/diamond/functional).
  "
  ([fact src-desc activ alpha beta]
   (api/activ-blueprint (api/diamond-factory fact) src-desc activ alpha beta))
  ([fact src-desc activ alpha]
   (api/activ-blueprint (api/diamond-factory fact) src-desc activ alpha 0.0))
  ([fact src-desc activ]
   (api/activ-blueprint (api/diamond-factory fact) src-desc activ 0.0 0.0))
  ([src-desc activ]
   (api/activ-blueprint *diamond-factory* src-desc activ 0.0 0.0)))

(defn inner-product
  "Creates an inner-product blueprint, which is also a function that can create an inner product
  operation structure that can be the main building block of the linear part of a network layer.
  If you find this description a bit cryptic, just think about matrix multiplication operation,
  generalized to more than two dimensions. Even the ND inner product can be efficiently implemented
  with 2D matrix multiplication, but even more optimization can be provided by DNNL, cuDNN, etc.

  Arguments:

  - `fact`: technology-specific engine factory.
  - `src-desc`: tensor descriptor (or even just a relevant part of its shape) of the product input.
  - `dst-desc`: tensor descriptor (or even just a relevant part of its shape) of the product output.
  - `weights-type`: type of weights and biases.

  Most of these arguments can be automatically inferred when this blueprint is used
  in a DNN DSL in the context of a network.

  See examples in [dnn-test](https://github.com/uncomplicate/deep-diamond/blob/master/test/uncomplicate/diamond/dnn_test.clj),
  and [functional-tests](https://github.com/uncomplicate/deep-diamond/tree/master/test/uncomplicate/diamond/functional).
  "
  ([fact src-desc dst-desc weights-type]
   (api/inner-product-blueprint (api/diamond-factory fact) src-desc dst-desc weights-type))
  ([fact src-desc dst-desc]
   (api/inner-product-blueprint (api/diamond-factory fact) src-desc dst-desc nil))
  ([src-desc dst-desc]
   (api/inner-product-blueprint *diamond-factory* src-desc dst-desc nil)))

(defn ^:private coerce-fc-dst
  [src-desc dst-desc]
  (let [src-shape (shape src-desc)
        dst-shape (shape dst-desc)
        n (get src-shape 0)]
    (if (< 1 (count dst-shape))
      dst-desc
      {:shape [n (get dst-shape 0)]
       :data-type (data-type dst-desc)
       :layout (layout dst-desc)})))

(defn fully-connected
  "Creates a dense aka fully connected neural network layer blueprint, which is also a function that
  can create the actual layer either when directly called or when used in the neural network description.

  Arguments:

  - `fact`: technology-specific engine factory.
  - `src-desc`: tensor descriptor (or even just a relevant part of its shape) of the layer input.
  - `dst-desc`: tensor descriptor (or even just a relevant part of its shape) of the layer output.
  - `activ`: keyword that determines the activation function (:relu, :elu, etc.). See [[activation]].
  - `args`: a map of additional arguments such as `:alpha`, `:beta`, `:weights-type`, or some of
  the technology specific options supported by the underlying engine.

  Most of these arguments can be automatically inferred when this blueprint is used
  in a DNN DSL in the context of a network.

  See examples in [dnn-test](https://github.com/uncomplicate/deep-diamond/blob/master/test/uncomplicate/diamond/dnn_test.clj),
  and [functional tests](https://github.com/uncomplicate/deep-diamond/tree/master/test/uncomplicate/diamond/functional).
  "
  ([fact src-desc dst-desc activ args]
   (let [alpha (or (:alpha args) (if (= activ :linear) 1.0 0.0))
         beta (or (:beta args) 0.0)
         dst-desc (coerce-fc-dst src-desc dst-desc)]
     (api/fc-blueprint (api/diamond-factory fact) src-desc dst-desc
                       activ alpha beta (:weights-type args))))
  ([fact src-desc dst-desc activ]
   (fully-connected fact src-desc dst-desc activ nil))
  ([dst-desc activ args]
   (fn
     ([fact src-desc]
      (fully-connected fact src-desc dst-desc activ args))
     ([src-desc]
      (fully-connected *diamond-factory* src-desc dst-desc activ args))))
  ([dst-desc activ]
   (fully-connected dst-desc activ nil)))

(defn dense
  "A simpler version of [[fully-connected]] layer. You'll usually use this function in the network
  description.

  Arguments:

  - `dst-desc`: tensor descriptor (or even just a relevant part of its shape) of the layer output.
  - `activ`: keyword that determines the activation function (:relu, :elu, etc.). See [[activation]].
  - `args`: a map of additional arguments such as `:alpha`, `:beta`, `:weights-desc`, or some of
  the technology specific options supported by the underlying engine.

  See [[fully-connected]].
  "
  ([dst-desc activ args]
   (fully-connected dst-desc activ args))
  ([dst-desc activ]
   (fully-connected dst-desc activ nil)))

(defn  ^:private coerce-conv-shapes [src-shape kernel-shape dst-shape
                                     strides padding dilation]
  (let [cnt (count src-shape)
        [n ic & idhw] src-shape
        [n oc :as dst-shape] (if (< (count dst-shape) cnt)
                               (into [n] dst-shape)
                               dst-shape)
        [_ _ & kdhw :as kernel-shape] (if (< (count kernel-shape) cnt)
                                        (into [oc ic] kernel-shape)
                                        kernel-shape)
        kdhw-dilated (map (fn [^long k ^long d]
                            (inc (* (dec k) d)))
                          kdhw dilation)]
    [kernel-shape (if (< (count dst-shape) cnt)
                    (into dst-shape
                          (map (fn [^long i ^long k ^long s ^long p]
                                 (inc (quot (+ (- i k) (* 2 p)) s)))
                               idhw kdhw-dilated strides padding))
                    dst-shape)]))

(defn convolution
  "Creates a convolution neural network layer blueprint, which is also a function that
  can create the actual layer either when directly called or when used in the neural network
  description.

  Arguments:

  - `fact`: technology-specific engine factory.
  - `src-desc`: tensor descriptor (or even just a relevant part of its shape) of the layer input.
  - `weights-desc` tensor descriptor (or even just a relevant part of its shape) of the weights and biases.
  - `dst-desc`: tensor descriptor (or even just a relevant part of its shape) of the layer output.
  - `activ`: keyword that determines the activation function (:relu, :elu, etc.). See [[activation]].
  - `args`: a map of additional arguments such as `:alpha`, `:beta`, `:strides`, `:padding`, `dilation`,
  or some of the technology specific options supported by the underlying engine.

  Most of these arguments can be automatically inferred when this blueprint is used
  in a DNN DSL in the context of a network.

  See examples in [dnn-test](https://github.com/uncomplicate/deep-diamond/blob/master/test/uncomplicate/diamond/dnn_test.clj),
  and [functional tests](https://github.com/uncomplicate/deep-diamond/tree/master/test/uncomplicate/diamond/functional).
  "
  ([fact src-desc weights-desc dst-desc activ args]
   (let [conv-dim (- (count (shape src-desc)) 2)
         strides (or (:strides args) (vec (repeat conv-dim 1)))
         padding (or (:padding args) (vec (repeat conv-dim 0)))
         dilation (or (:dilation args) (vec (repeat conv-dim 1)))
         alpha (or (:alpha args) (if (= activ :linear) 1.0 0.0))
         beta (or (:beta args) 0.0)
         [weights-shape dst-shape] (coerce-conv-shapes (shape src-desc) (shape weights-desc)
                                                       (shape dst-desc)
                                                       strides padding dilation)
         weights-desc (if (= weights-shape (shape weights-desc))
                        weights-desc
                        (desc weights-shape (data-type weights-desc) (layout weights-desc)))
         dst-desc (if (= dst-shape (shape dst-desc))
                    dst-desc
                    (desc dst-shape (data-type dst-desc) (layout dst-desc)))]
     (if (= (count (shape src-desc)) (count weights-shape) (count dst-shape))
       (api/convolution-blueprint (api/diamond-factory fact)
                                  src-desc weights-desc dst-desc activ
                                  strides padding dilation alpha beta)
       (dragan-says-ex "TODO message."))))
  ([fact src-desc weights-desc dst-desc activ]
   (convolution fact src-desc weights-desc dst-desc activ nil))
  ([dst-desc kernel-desc activ args]
   (fn
     ([fact src-desc]
      (convolution fact src-desc kernel-desc dst-desc activ args))
     ([src-desc]
      (convolution *diamond-factory* src-desc kernel-desc dst-desc activ args))))
  ([dst-desc kernel-desc activ]
   (convolution dst-desc kernel-desc activ nil)))

(defn convo
  "A simpler version of [[convolution]]. You'll usually use this function in the network description.

  Arguments:

  - `dst-desc`: tensor descriptor (or even just a relevant part of its shape) of the layer output.
  - `activ`: keyword that determines the activation function (:relu, :elu, etc.). See [[activation]].
  - `args`: a map of additional arguments such as `:alpha`, `:beta`, `:weights-desc`, or some of
  the technology specific options supported by the underlying engine.

  See [[fully-connected]].
  "
  ([dst-desc kernel-desc activ args]
   (convolution dst-desc kernel-desc activ args))
  ([dst-desc kernel-desc activ]
   (convolution dst-desc kernel-desc activ nil)))

(defn ^:private coerce-pooling-dst [src-shape dst-shape]
  (let [[n c] src-shape
        missing-cnt (- (count src-shape) (count dst-shape))]
    (if (= 0 missing-cnt)
      dst-shape
      (into (if (= 1 missing-cnt) [n] [n c]) dst-shape))))

(defn pooling
  "Creates a pooling neural network layer blueprint, which is also a function that can create
  the actual pooling layer either when directly called or when used in the neural network description.

  Arguments:

  - `fact`: technology-specific engine factory.
  - `src-desc`: tensor descriptor (or even just a relevant part of its shape) of the layer input.
  - `kernel`: kernel shape.
  - `algo`: keyword that determines pooling algorithm (`:avg`, `max`, etc.).
  See pooling algorithms supported by DNNL ([[uncomplicate.diamond.internal.dnnl.constants/dnnl-pooling-alg-kind]]), and cuDNN ([[uncomplicate.diamond.internal.cudnn.constants/cudnn-pooling-mode]]). Keywords are the same when possible, but not all keywords are supported by all backends, in general.
  - `args`: a map of additional arguments such as `:strides` or `:padding`.

  Most of these arguments can be automatically inferred when this blueprint is used
  in a DNN DSL in the context of a network.

  See examples in [dnn-test](https://github.com/uncomplicate/deep-diamond/blob/master/test/uncomplicate/diamond/dnn_test.clj),
  and [functional tests](https://github.com/uncomplicate/deep-diamond/tree/master/test/uncomplicate/diamond/functional).
  "
  ([fact src-desc kernel algo args]
   (let [conv-dim (count kernel)
         padding (or (:padding args) (vec (repeat conv-dim 0)))
         strides (or (:strides args) kernel)
         dst-shape (coerce-pooling-dst
                    (shape src-desc)
                    (map (fn [^long src ^long stride ^long p]
                           (quot (+ src p p) stride))
                         (take-last conv-dim (shape src-desc))
                         strides
                         padding))]
     (api/pooling-blueprint (api/diamond-factory fact)
                            src-desc dst-shape algo strides kernel padding)))
  ([fact src-desc kernel algo]
   (pooling fact src-desc kernel algo nil))
  ([kernel algo args]
   (fn
     ([fact src-desc]
      (pooling fact src-desc kernel algo args))
     ([src-desc]
      (pooling *diamond-factory* src-desc kernel algo args))))
  ([kernel algo]
   (pooling kernel algo nil)))

(defn dropout-mask
  "Keeps last `mask-dim` elements from `src-desc` and pads the elements before with `1`."
  [src-desc ^long mask-dim]
  (let [src-shape (shape src-desc)
        ones-dim (- (count src-shape) mask-dim)]
    (into (vec (repeat ones-dim 1)) (drop ones-dim src-shape))))

(defn dropout
  "Creates a dropout neural network layer blueprint, which is also a function that can create
  the actual dropout layer either when directly called or when used in the neural network description.

  Arguments:

  - `fact`: technology-specific engine factory.
  - `src-desc`: tensor descriptor (or even just a relevant part of its shape) of the layer input.
  - `sd`: standard deviation of swing around the layer weight.

  Most of these arguments can be automatically inferred when this blueprint is used
  in a DNN DSL in the context of a network.

  See examples in [dnn-test](https://github.com/uncomplicate/deep-diamond/blob/master/test/uncomplicate/diamond/dnn_test.clj),
  and [functional tests](https://github.com/uncomplicate/deep-diamond/tree/master/test/uncomplicate/diamond/functional).
  "
  ([fact src-desc sd]
   (api/gaussian-dropout-blueprint fact src-desc sd))
  ([sd]
   (fn
     ([fact src-desc]
      (dropout fact src-desc sd))
     ([src-desc]
      (dropout *diamond-factory* src-desc sd))))
  ([]
   (dropout 1.0)))

(defn batch-norm
  "Creates a batch normalization neural network layer blueprint, which is also a function that can
  create the actual batch normalization layer either when directly called or when used in the neural
  network description.

  Arguments:

  - `fact`: technology-specific engine factory.
  - `src-desc`: tensor descriptor (or even just a relevant part of its shape) of the layer input.
  - `activ`: keyword that determines the activation function (:relu, :elu, etc.). See [[activation]].
  - `args`: a map of additional arguments such as `:alpha`, `:beta`, or some of the technology
  specific options supported by the underlying engine.

  Most of these arguments can be automatically inferred when this blueprint is used
  in a DNN DSL in the context of a network.

  See examples in [dnn-test](https://github.com/uncomplicate/deep-diamond/blob/master/test/uncomplicate/diamond/dnn_test.clj),
  and [functional tests](https://github.com/uncomplicate/deep-diamond/tree/master/test/uncomplicate/diamond/functional).
  "
  ([fact src-desc activ args]
   (let [alpha (or (:alpha args) (if (= activ :linear) 1.0 0.0))
         beta (or (:beta args) 0.0)]
     (api/batch-norm-blueprint fact src-desc activ alpha beta)))
  ([activ args]
   (fn
     ([fact src-desc]
      (batch-norm fact src-desc activ args))
     ([src-desc]
      (batch-norm *diamond-factory* src-desc activ args))))
  ([activ]
   (batch-norm activ nil))
  ([]
   (batch-norm :linear nil)))

(defn concatenate
  "Creates a concatenation blueprint, which is also a function that can create the actual
  concatenation layer either when directly called or when used in the neural network description.
  Concatenation stitches multiple input tensors into one output tensor. Also see [[branch]] and [[split]].

  Arguments:

  - `fact`: technology-specific engine factory.
  - `conc-dim`: the dimension where concatenation is going to expand the output.
  - `src-descs`: tensor descriptors (or even just a relevant parts of their shape) of layer inputs.
  - `dst-type`: output type.

  Most of these arguments can be automatically inferred when this blueprint is used
  in a DNN DSL in the context of a network.

  See examples in [dnn-test](https://github.com/uncomplicate/deep-diamond/blob/master/test/uncomplicate/diamond/dnn_test.clj),
  and [functional tests](https://github.com/uncomplicate/deep-diamond/tree/master/test/uncomplicate/diamond/functional).
  "
  ([fact ^long conc-dim src-descs dst-type]
   (let [src-descs (if (sequential? src-descs) src-descs (api/train-desc src-descs))]
     (api/concat-blueprint fact src-descs conc-dim dst-type)))
  ([fact conc-dim src-descs]
   (concatenate fact conc-dim src-descs nil))
  ([conc-dim dst-type]
   (fn
     ([fact src-descs]
      (concatenate fact conc-dim src-descs dst-type))
     ([src-descs]
      (concatenate *diamond-factory* conc-dim src-descs dst-type))))
  ([conc-dim]
   (concatenate conc-dim nil))
  ([]
   (concatenate 0 nil)))

(defn conc
  "A simpler version of [[concatenate]]. You'll usually use this function in the network description.
  Concatenation stitches multiple input tensors into one output tensor. Also see [[branch]] and [[split]].

  Arguments:

  - `conc-dim`: the dimension where concatenation is going to expand the output.

  See examples in [dnn-test](https://github.com/uncomplicate/deep-diamond/blob/master/test/uncomplicate/diamond/dnn_test.clj),
  and [functional tests](https://github.com/uncomplicate/deep-diamond/tree/master/test/uncomplicate/diamond/functional).
  "
  ([^long conc-dim]
   (concatenate conc-dim))
  ([]
   (concatenate 0)))

(defn  ^:private coerce-branch-dst [src-desc branch-dim dst-descs]
  (let [src-shape (shape src-desc)]
    (fmap (fn [dst-desc]
            (let [dst-shape (shape dst-desc)
                  dst-layout (layout dst-desc)
                  dst-cnt (count dst-shape)]
              (if (< 1 dst-cnt)
                dst-desc
                (let [dst-shape (assoc src-shape branch-dim (get dst-shape 0))]
                  {:shape dst-shape
                   :data-type (data-type dst-desc)
                   :layout (if (or (keyword? dst-layout) (= dst-cnt (count src-shape)))
                             dst-layout
                             (default-strides dst-shape))}))))
          dst-descs)))

(defn branch
  "Creates a branch blueprint, which is also a function that can create the actual
  branching layer either when directly called or when used in the neural network description.
  Branching divides the input tensor into multiple output tensors. Also see [[concatenate]] and [[split]].

  Arguments:

  - `fact`: technology-specific engine factory.
  - `src-desc`: tensor descriptor (or even just a relevant part of its shape) of the layer input.
  - `branch-dim`: the dimension where branching is going to divide the input.
  - `dst-descs`: tensor descriptors (or even just a relevant parts of their shape) of layer outputs.

  Most of these arguments can be automatically inferred when this blueprint is used
  in a DNN DSL in the context of a network.

  As an example, branch divides tensor shape  `[1 2 1 1]` by `branch-dim = 1`
  into `[1 1 1 1]` and `[1 1 1 1]`.

  See examples in [dnn-test](https://github.com/uncomplicate/deep-diamond/blob/master/test/uncomplicate/diamond/dnn_test.clj),
  and [functional tests](https://github.com/uncomplicate/deep-diamond/tree/master/test/uncomplicate/diamond/functional).
  "
  ([fact src-desc ^long branch-dim dst-descs]
   (api/branch-blueprint fact src-desc branch-dim
                         (coerce-branch-dst src-desc branch-dim dst-descs)))
  ([^long branch-dim dst-descs]
   (fn
     ([fact src-desc]
      (branch fact src-desc branch-dim dst-descs))
     ([src-desc]
      (branch *diamond-factory* src-desc branch-dim dst-descs))))
  ([dst-descs]
   (branch 0 dst-descs)))

(defn split
  "Creates a split blueprint, which is also a function that can create the actual
  split layer either when directly called or when used in the neural network description.
  Splitting clones the input tensor into multiple output tensors. Also see [[concatenate]] and [[branch]].

  Arguments:

  - `fact`: technology-specific engine factory.
  - `src-desc`: tensor descriptor (or even just a relevant part of its shape) of the layer input.
  - `n`: number of output clones.

  As an example, split clones tensor shape `[1 2 1 1]` `n = 3` times into three tensors
  shaped `[1 2 1 1]`, `[1 2 1 1]`, and `[1 2 1 1]`.


  See examples in [dnn-test](https://github.com/uncomplicate/deep-diamond/blob/master/test/uncomplicate/diamond/dnn_test.clj),
  and [functional tests](https://github.com/uncomplicate/deep-diamond/tree/master/test/uncomplicate/diamond/functional).
  "
  ([fact src-desc ^long n]
   (api/split-blueprint fact src-desc n))
  ([^long n]
   (fn
     ([fact src-desc]
      (split fact src-desc n))
     ([src-desc]
      (split *diamond-factory* src-desc n)))))

(defn sum
  "Creates a sum blueprint, which is also a function that can create the actual summing layer either
  when directly called or when used in the neural network description. The summing layer will
  sum all respective entries from input tensors into one output tensor of the same shape.

  Arguments:

  - `fact`: technology-specific engine factory.
  - `src-descs`: tensor descriptors (or even just a relevant parts of their shape) of layer inputs.

  See examples in [dnn-test](https://github.com/uncomplicate/deep-diamond/blob/master/test/uncomplicate/diamond/dnn_test.clj),
  and [functional tests](https://github.com/uncomplicate/deep-diamond/tree/master/test/uncomplicate/diamond/functional).
  "
  ([fact src-descs]
   (api/sum-blueprint fact src-descs))
  ([]
   (fn
     ([fact src-desc]
      (sum fact src-desc))
     ([src-desc]
      (sum *diamond-factory* src-desc)))))

(defn cost
  "Creates cost that goes at the output of the network and drives the minimization
  of the cost function in respect to the `train-tz` tensor. Currently supported `cost-kw` 's are:
  `:quadratic`, `:mean-absolute`, and `:crossentropy`.

  Arguments:

  - `layer`: the last layer in the network, which provides the estimated ('predicted') output.
  - `train-tz`: the target training tensor.

  See examples in [dnn-test](https://github.com/uncomplicate/deep-diamond/blob/master/test/uncomplicate/diamond/dnn_test.clj),
  and [functional tests](https://github.com/uncomplicate/deep-diamond/tree/master/test/uncomplicate/diamond/functional).
  "
  ([layer train-tz cost-kw]
   ((case cost-kw
      :quadratic api/quadratic-cost
      :mean-absolute api/mean-absolute-cost
      :crossentropy api/crossentropy-cost
      (dragan-says-ex "This cost function is not supported." {:cost cost-kw}))
    (api/diamond-factory layer) layer (view train-tz)))
  ([layer cost-kw]
   (let-release [train-tz (tensor (output layer) (output layer))]
     (cost layer train-tz cost-kw)))
  ([layer]
   (cost layer :quadratic)))

(defn network
  "Creates a neural network blueprint from the specific input (`src-desc`), and blueprints provided by `layers`.
  This function is very flexible and tries to accommodate to diverse `layers` data. Please see the test
  folder for detailed examples.

  The [Deep Learning for Programmers](https://aiprobook.com/deep-learning-for-programmers) book
  contains very detailed examples and explanations. Please check it out.

  See examples in [dnn-test](https://github.com/uncomplicate/deep-diamond/blob/master/test/uncomplicate/diamond/dnn_test.clj),
  and [functional tests](https://github.com/uncomplicate/deep-diamond/tree/master/test/uncomplicate/diamond/functional).
  "
  ([fact src-desc layers]
   (sequential-network (api/diamond-factory fact) src-desc layers))
  ([src-desc layers]
   (network *diamond-factory* src-desc layers))
  ([layers]
   (fn
     ([fact src-descs]
      (network fact src-descs layers))
     ([src-descs]
      (network *diamond-factory* src-descs layers)))))

(defn init!
  "Destructively initializes the parameters (weights, biases, etc.) of the network using [Xavier initialization](),
  which is a good default. You are, of course, free to provide different `init-fn`, which is an
  one-argument function that receives every tensor that needs initialization in each layer.
  This is an automatic default for the 99% of cases that need standard stuff. If you need even more
  liberal initialization, you are free to implement a function that access the parameters using
  the internal API, and do whatever you want.

  The [Deep Learning for Programmers](https://aiprobook.com/deep-learning-for-programmers) book
  contains a detailed discussion about different trade-offs that should be considered when initializing
  the network.
  "
  ([net!]
   (with-release [rng (rng-state (view-vctr (input net!)))]
     (api/init net! (fn [x] (rand-normal! rng 0.0 (/ 1.0 (double (apply * (rest (shape x))))) x))))
   net!)
  ([net! init-fn]
   (api/init net! init-fn)
   net!))

(defn ^:private linear-decay
  [^long t ^long tau ^double eta-0 ^double eta-tau]
  (let [alpha (min (double (/ t tau)) 1.0)]
    (+  (* (- 1.0 alpha) eta-0) (* alpha eta-tau))))

(defn ^:private train*
  ([net cost! epochs hyperparam]
   (let [hyperparam (transient (into [0] hyperparam))]
     (dotimes [t epochs]
       (assoc! hyperparam 0 t)
       (api/forward net hyperparam)
       (api/forward cost!)
       (api/backward cost!)
       (api/backward net hyperparam)))
   (net)
   (cost!))
  ([net cost! options]
   (map (fn [[epochs hyperparam]]
          (train* net cost! epochs hyperparam))
        options))
  ([selector net in-batcher out-batcher cost! epochs hyperparam]
   (let [b-size (batch-size (input in-batcher))
         mb-size (batch-size (output in-batcher))
         mb-count (quot b-size mb-size)
         [eta-decay eta-0 eta-tau]
         (let [eta (first hyperparam)]
           (cond
             (number? eta) [linear-decay eta (* 0.01 (double eta))]
             (sequential? eta) (cons linear-decay eta)
             :default (cons (constantly nil) eta)))
         hyperparam (transient (into [0 0] (rest hyperparam)))]
     (dotimes [t (long epochs)]
       (assoc! hyperparam 0 t)
       (assoc! hyperparam 1 (eta-decay t epochs eta-0 eta-tau))
       (dotimes [n mb-count]
         (let [selection (selector n mb-size)]
           (in-batcher selection)
           (out-batcher selection)
           (api/forward net hyperparam)
           (api/forward cost!)
           (api/backward cost!)
           (api/backward net hyperparam))))
     (net)
     (cost!)))
  ([selector net in-batcher out-batcher cost! options]
   (map (fn [[epochs hyperparam]]
          (train* selector net in-batcher out-batcher cost! epochs hyperparam))
        options)))

(defn train
  "This is the magic function that trains your network `net` using `cost!`, trough
  a number of `epochs`, using hyperparameters `hyperparam`. It is rather flexible
  and will automatically figure out how to do mini-batches needed.

  Arguments:

  - `net`: the network that needs to be trained.
  - `cost!`: the cost function. See [[cost]].
  - `epochs`: the number training cycles that process all training data points.
  - `hyperparam`: a vector of hyperparameters relevant in the context of the chosen training algorithm
  that was provided at the moment of creation of the network from its blueprint (`:sgd`, `adam`, etc.).
  Typically contains learning rate (`eta`), decay, etc. Please see the [DLFP](https://aiprobook.com/deep-learning-for-programmers)
  book or some other resource for the explanation of many possible parameters connected with various learning algorithms.
  - `options`: multiple `epochs` and `hyperparam` pairs can be provided in `options` sequence.
  - `in`: the input of the network. Typically a tensor or a connector, but really anything that can
  accept [[uncomplicate.diamond.tensor/output]]. If it's bigger than the network's input shape,
  the training will be done in mini-batches.
  - `out`: the output of the network. Typically a tensor or a connector, but really anything that can
  accept [[uncomplicate.diamond.tensor/input]]. If it's bigger than the network's output shape,
  the training will be done in mini-batches.

  If you need stochastic re-shuffling of the mini-batches, please consider [[train-shullfle]].

  Explaining all that this function can do would require a whole essay, so it's best
  to study the examples from many examples provided by Deep Diamond's [functional tests](https://github.com/uncomplicate/deep-diamond/tree/master/test/uncomplicate/diamond/functional). Even better resource is the [Deep Learning for Programmers](https://aiprobook.com/deep-learning-for-programmers) book
  book. Please check it out.
  "
  ([net cost! epochs hyperparam]
   (if (keyword? cost!)
     (with-release [cost! (cost net cost!)]
       (train* net cost! epochs hyperparam))
     (train* net cost! epochs hyperparam)))
  ([net cost! options]
   (if (keyword? cost!)
     (with-release [cost! (cost net cost!)]
       (doall (train* net cost! options)))
     (train* net cost! options)))
  ([net in out cost! epochs hyperparam]
   (cond (keyword? cost!)
         (with-release [cost! (cost net cost!)]
           (train net in out cost! epochs hyperparam))
         (satisfies? TensorContainer in)
         (with-release [in-batcher (batcher in (input net))]
           (train net in-batcher out cost! epochs hyperparam))
         (satisfies? TensorContainer out)
         (with-release [out-batcher (batcher out (api/diff-input cost!))]
           (train* * net in out-batcher cost! epochs hyperparam))
         :default (train* * net in out cost! epochs hyperparam)))
  ([net in out cost! options]
   (cond (keyword? cost!)
         (with-release [cost! (cost net cost!)]
           (doall (train net in out cost! options)))
         (satisfies? TensorContainer in)
         (with-release [in-batcher (batcher in (input net))]
           (doall (train net in-batcher out cost! options)))
         (satisfies? TensorContainer out)
         (with-release [out-batcher (batcher out (api/diff-input cost!))]
           (doall (train* * net in out-batcher cost! options)))
         :default (train* * net in out cost! options))))

(defn ^:private random-cols [_ mb-size]
  (shuffle (range mb-size)))

(defn train-shuffle
  "Similar to [[train]], but does stochastic reshuffling of the mini-batches.

  Arguments:

  - `net`: the network that needs to be trained.
  - `cost!`: the cost function. See [[cost]].
  - `epochs`: the number training cycles that process all training data points.
  - `hyperparam`: a vector of hyperparameters relevant in the context of the chosen training algorithm
  that was provided at the moment of creation of the network from its blueprint (`:sgd`, `adam`, etc.).
  Typically contains learning rate (`eta`), decay, etc. Please see the [DLFP](https://aiprobook.com/deep-learning-for-programmers)
  book or some other resource for the explanation of many possible parameters connected with various learning algorithms.
  - `options`: multiple `epochs` and `hyperparam` pairs can be provided in `options` sequence.
  - `in`: the input of the network. Typically a tensor or a connector, but really anything that can
  accept [[uncomplicate.diamond.tensor/output]]. If it's bigger than the network's input shape,
  the training will be done in mini-batches.
  - `out`: the output of the network. Typically a tensor or a connector, but really anything that can
  accept [[uncomplicate.diamond.tensor/input]]. If it's bigger than the network's output shape,
  the training will be done in mini-batches.

  See examples in [dnn-test](https://github.com/uncomplicate/deep-diamond/blob/master/test/uncomplicate/diamond/dnn_test.clj),
  and [functional tests](https://github.com/uncomplicate/deep-diamond/tree/master/test/uncomplicate/diamond/functional).
  "
  ([net in out cost! epochs hyperparam]
   (cond (keyword? cost!)
         (with-release [cost! (cost net cost!)]
           (train net in out cost! epochs hyperparam))
         (satisfies? TensorContainer in)
         (with-release [in-shuffler (shuffler in (input net))]
           (train net in-shuffler out cost! epochs hyperparam))
         (satisfies? TensorContainer out)
         (with-release [out-shuffler (shuffler out (api/diff-input cost!))]
           (train* random-cols net in out-shuffler cost! epochs hyperparam))
         :default (train* random-cols net in out cost! epochs hyperparam)))
  ([net in out cost! options]
   (cond (keyword? cost!)
         (with-release [cost! (cost net cost!)]
           (doall (train net in out cost! options)))
         (satisfies? TensorContainer in)
         (with-release [in-shuffler (shuffler in (input net))]
           (doall (train net in-shuffler out cost! options)))
         (satisfies? TensorContainer out)
         (with-release [out-shuffler (shuffler out (api/diff-input cost!))]
           (doall (train* random-cols net in out-shuffler cost! options)))
         :default (train* random-cols net in out cost! options))))

(defn ^:private infer* [net in-batcher out-batcher]
  (let [b-size (batch-size (input in-batcher))
        mb-size (batch-size (input net))
        mb-count (long (quot b-size mb-size))
        mb-rem (long (rem b-size mb-size))]
    (dotimes [n mb-count]
      (in-batcher (* n mb-size))
      (net)
      (out-batcher 0 (* n mb-size)))
    (when (< 0 mb-rem)
      (in-batcher (- b-size mb-size))
      (net)
      (out-batcher 0 (- b-size mb-size)))
    (output out-batcher)))

(defn infer
  "Estimates output `out` for the provided input `in`. Works with tensors, connectors, and anything
  that can provide Does [[uncomplicate.diamond.tensor/input]] and Does [[uncomplicate.diamond.tensor/output]].
  If `in` and `out` are bigger than the network shapes, automatically does the inference in mini-batches.

  Please also see [[train]].

  See examples in [dnn-test](https://github.com/uncomplicate/deep-diamond/blob/master/test/uncomplicate/diamond/dnn_test.clj),
  and [functional tests](https://github.com/uncomplicate/deep-diamond/tree/master/test/uncomplicate/diamond/functional).
  "
  ([net in out]
   (cond (satisfies? TensorContainer in)
         (with-release [in-batcher (batcher in (input net))]
           (infer net in-batcher out))
         (satisfies? TensorContainer out)
         (with-release [out-batcher (batcher (output net) out)]
           (infer* net in out-batcher))
         :default (infer* net in out)))
  ([net in]
   (let [net-out (output net)
         shape-out (shape net-out)]
     (let-release [out (tensor net (assoc shape-out (api/batch-index net-out)
                                          (get (shape (input in)) (api/batch-index (input in))))
                               (data-type net-out) (layout net-out))]
       (infer net in out)))))

;;TODO train should be renamed to train! and infer! (perhaps)

;; ========================== Recurrent networks =========================================

(defn  ^:private coerce-rnn-dst [src-desc dst-desc]
  (let [dst-shape (shape dst-desc)
        [t n c] (shape src-desc)]
    (case (count dst-shape)
      0 src-desc
      1 {:shape [t n (get dst-shape 0)]
         :data-type (data-type dst-desc)
         :layout :tnc}
      2 {:shape [(get dst-shape 0) n (get (dst-shape 1))]
         :data-type (data-type dst-desc)
         :layout :tnc}
      dst-desc)))

(defn rnn-op
  "The RNN operation blueprint. You are probably looking for the [[rnn]] function instead."
  ([fact src-desc dst-desc weights-type activ dir lrs src-iter? dst-iter?]
   (api/rnn-op-blueprint (api/diamond-factory fact) src-desc (coerce-rnn-dst src-desc dst-desc)
                         weights-type activ dir lrs src-iter? dst-iter?))
  ([fact src-desc dst-desc activ dir lrs src-iter? dst-iter?]
   (api/rnn-op-blueprint (api/diamond-factory fact) src-desc (coerce-rnn-dst src-desc dst-desc)
                         nil activ dir lrs src-iter? dst-iter?))
  ([fact src-desc dst-desc lrs src-iter? dst-iter?]
   (api/rnn-op-blueprint (api/diamond-factory fact) src-desc (coerce-rnn-dst src-desc dst-desc)
                         nil :relu :unidirectional lrs src-iter? dst-iter?))
  ([fact src-desc dst-desc lrs]
   (rnn-op fact src-desc dst-desc lrs false false))
  ([src-desc dst-desc lrs]
   (rnn-op *diamond-factory* src-desc dst-desc lrs)))

(defn rnn
  "Creates a recurrent neural network (RNN) layer blueprint, which is also a function that
  can create the actual RNN layer either when directly called or when used in the neural network
  description.

  Arguments:

  - `fact`: technology-specific engine factory.
  - `src-desc`: tensor descriptor (or even just a relevant part of its shape) of the layer input.
  - `dst-desc`: tensor descriptor (or even just a relevant part of its shape) of the layer output.
  - `lrs`: the number of recurrent layers.
  - `activ`: keyword that determines the activation function (:relu, :elu, etc.) for vanilla RNN,
  or the specialized RNN algorithm (`:lstm`, `:gru` etc.) supported by DNNL ([[uncomplicate.diamond.internal.dnnl.constants/dnnl-rnn-alg-kind]]), and cuDNN ([[uncomplicate.diamond.internal.cudnn.constants/cudnn-cell-mode]]). Also see [[activation]].
  - `args`: a map of additional arguments such as `:weights-type`, `:src-iter`, `:dst-iter`,
  or some of the technology specific options supported by the underlying engine.

  Most of these arguments can be automatically inferred when this blueprint is used
  in a DNN DSL in the context of a network.

  See examples in [dnn-test](https://github.com/uncomplicate/deep-diamond/blob/master/test/uncomplicate/diamond/rnn_test.clj),
  and [MasterCard functional test](https://github.com/uncomplicate/deep-diamond/tree/master/test/uncomplicate/diamond/functional/mastercard).
  "
  ([fact src-desc dst-desc lrs activ args]
   (let [alpha (or (:alpha args) (if (= activ :linear) 1.0 0.0))]
     (api/rnn-blueprint (api/diamond-factory fact) src-desc (coerce-rnn-dst src-desc dst-desc)
                        lrs activ alpha
                        (:weights-type args) (:src-iter args) (:dst-iter args))))
  ([fact src-desc dst-desc activ args]
   (rnn fact src-desc dst-desc 1 activ args))
  ([dst-desc lrs activ args]
   (fn
     ([fact src-desc]
      (rnn fact src-desc dst-desc lrs activ args))
     ([src-desc]
      (rnn *diamond-factory* src-desc dst-desc lrs activ args))))
  ([lrs activ args]
   (rnn [] lrs activ args))
  ([lrs-or-dst-desc activ]
   (if (number? lrs-or-dst-desc)
     (rnn [] lrs-or-dst-desc activ nil)
     (rnn lrs-or-dst-desc 1 activ nil)))
  ([param]
   (if (keyword param)
     (rnn 1 param)
     (rnn param :gru)))
  ([]
   (rnn [] 1 :gru nil)))

(defn abbreviate
  "Extract the relevant part of RNN output sequence."
  ([fact src-desc dst-type]
   (api/abbreviate-blueprint fact src-desc (or dst-type (data-type src-desc))))
  ([dst-type]
   (fn
     ([fact src-desc]
      (abbreviate fact src-desc dst-type))
     ([src-desc]
      (abbreviate *diamond-factory* src-desc dst-type))))
  ([]
   (abbreviate nil)))
