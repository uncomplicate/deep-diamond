(ns ^{:author "Dragan Djuric"}
    uncomplicate.diamond.internal.protocols
  "Internal Deep Diamond API mostly relevant to programmers who write tools, extensions
  and additional backend engines.

  When implementing your own backend, in addition to the light documentation provided in this
  namespace, please refer to the DNNL implementation and its tests. It is currently the most
  complete one. Next, see the cuDNN GPU backend to see how it solves differences in general
  approach to computation. There are countless small and not so small gotchas with these
  complex technologies, so expect lots of debugging.
  ")

;; ===================== General ========================================

(defprotocol DiamondFactoryProvider
  "An object that can provide a DD backend factory, and native backend factory."
  (diamond-factory [this] "Returens the backend factory related to this object.")
  (native-diamond-factory [this] "Returns native backend factory related to this object."))

(defprotocol NeanderthalFactoryProvider
  "An object that can provide relevant Neanderthal backend factory for the provided
  `dtype` (typically a keyword such as `:float`, `:int`, etc.). Most backends are
  supposed to provide these factories for Java primitive types supported by Neanderthal.

  Anyway, don't forget to check out both DNNL and cuDNN implementations, AND the relevant **tests**.
  They are the final sources of truth, not these docstrings; if the docs and tests don't match,
  the tests are right, and the docstrings are wrong.
  "
  (neanderthal-factory [this dtype] "Returns Neanderthal backend factory related to this DD object."))

;; ===================== Tensor ========================================

(defprotocol TensorFactory
  "A backend factory for tensor-related structures, such as the tensors themselves,
  tensor decsciptors, tensor transformers, batchers, shufflers, and engines.
  "
  (create-tensor-desc [this desc] [this shape type format]
    "Create technology-specific tensor descriptor from another descriptor `desc`,
     or from `shape`, `type`, and `format`.")
  (create-tensor [this desc batch-index init] [this desc init]
    "Create tensor from the provided descriptor `desc`, and initializes it to zero if `init` is true.")
  (create-transformer [this in out]
    "Create transformer from `in` to `out`")
  (create-batcher [this src dst mb-size]
    "Create batcher from `src` to `dst` that has mini-batch capacity `mb-size`.")
  (create-shuffler [this src dst]
    "Create shuffluer from `src` to `dst`")
  (tensor-engine [this dtype]
    "Provides DD tensor engine for data type `dtype` (`:float`, etc.)"))

(defprotocol MappedTensorFactory
  "A backend factory for memory-mapped tensor-related structures, such as the mapped tensors themselves.
  "
  (map-channel [this channel desc flag offset-bytes n-index]
    "Create a mapped tensor from the provided `FileChannel`, descriptor `desc`, mapping `flags`."))

(defprotocol Offset
  "An object for which the memory region that it controls can be (destructively) changed."
  (offset [tz n-ofst]
    "Offsets the underlying memory region for this `tz` by `n-ofst` batches
     (not in bytes or scalar entries)."))

;; =================== DNN ============================================

(defprotocol DnnFactory
  "A backend factory for neural networks related structures, that typically
  creates blueprints for common layers such as fully connected layers,
  convolutions, batch normalization, etc.
  "
  (activ-op-blueprint [this src-desc activ alpha beta]
    "Create technology specific activation blueprint from `src-desc`,
     `activ` keyword, `alpha`, and `beta` scalar parameters.")
  (activ-blueprint [this src-provider activ alpha beta]
    "Create technology specific activation layer blueprint from `src-provider`,
     `activ` keyword, `alpha`, and `beta` scalar parameters.")
  (inner-product-blueprint [this src-desc dst-desc weights-type]
    "Create technology specific inner product blueprint with input `src-desc`,
    output `dst-desc`, and `weights-type` keyword.")
  (fc-blueprint [this src-desc dst-desc activ alpha beta weights-type]
    "Create fully connected (dense) layer blueprint with input `src-desc`,
    output `dst-desc`, `activ` keyword, `alpha`, and `beta` scalar parameters,
    and `weights-type` keyword.")
  (convolution-blueprint [this src-desc kernel-desc dst-desc activ
                          strides padding dilation alpha beta]
    "Create convolution layer blueprint with input `src-desc`, kernel `kernel-desc`,
    output `dst-desc`, algorithm keyword `algo`, `activ` keyword, `strides`, `padding` and
    `dilation` Clojure vectors, `alpha`, and `beta` scalar parameters.")
  (pooling-blueprint [this src-desc dst-desc algo strides kernel padding]
    "Create pooling layer blueprint with input `src-desc`, output `dst-desc`,
    algorithm keyword `algo`, `strides`, `kernel`, and `padding` Clojure vectors.")
  (gaussian-dropout-blueprint [this src-desc sd]
    "Create dropout layer blueprint with input `src-desc`, and standard deviation scalar `sd`.")
  (batch-norm-blueprint [this src-desc activ alpha beta]
    "Create batch normalization layer blueprint with input `src-desc`, `activ` keyword,
    `alpha`, and `beta` scalar parameters.")
  (concat-blueprint [this src-descs conc-dim dst-type]
    "Create concatenation layer blueprint with inputs `src-descs`, scalar `conc-dim`,
    and `dst-type` keyword.")
  (branch-blueprint [this src-desc split-dim dst-descs]
    "Create branch layer blueprint with input `src-desc`, scalar `split-dim`,
    and `dst-descs` outputs.")
  (split-blueprint [this src-desc n]
    "Create split layer blueprint with input `src-desc`, and scalar `n`, the number of splits at the output.")
  (sum-blueprint [this src-descs]
    "Create summing layer blueprint with inputs `src-descs`.")
  (create-workspace [this byte-size]
    "Creates technology specific workspace of `byte-size` size."))

(defprotocol RnnFactory
  "A backend factory for recurrent neural networks (RNN) related structures, that typically
  creates blueprints for RNN operations and layers.
  "
  (rnn-op-blueprint [this src-desc dst-desc weights-type activ dir lrs src-iter? dst-iter?]
    "Create RNN operation with `src-desc`, `dst-desc`, `weights-type` keyword, `activ` keyword,
     direction `dir`, number of layers `lrs`, and booleans `src-iter?`, and `dst-iter?`.")
  (rnn-blueprint [fact src-desc dst-desc lrs activ alpha weights-type src-iter? dst-iter?]
    "Create RNN layer with `src-desc`, `dst-desc`, `activ` keyword, `weights-type` keyword,
     and booleans `src-iter?`, and `dst-iter?`.")
  (abbreviate-blueprint [fact src-desc dst-type]
    "Creates a RNN output sequence abbreviation blueprint."))

(defprotocol CostFactory
  "A backend factory for supported cost implementations."
  (quadratic-cost [this last-layer train-tz] "Create quadratic cost blueprint.")
  (mean-absolute-cost [this last-layer train-tz] "Create absolute cost blueprint.")
  (crossentropy-cost [this last-layer train-tz] "Create cross-entropy cost blueprint."))

(defprotocol DescriptorProvider
  "An object that can provide several descriptors for connecting with other layers
  in different contexts (inference, training, and gradients.)"
  (inf-desc [this] "Descriptor used in inference layer.")
  (train-desc [this] "Descriptor used in training layer.")
  (diff-desc [this] "Descriptor used for propagating gradients."))

(defprotocol Parameters
  "An object that has trainable parameters."
  (weights [this] "Provides tensor that contains weights, or anything that can be considered weights in the context of learning algorithms.")
  (bias [this] "Provides tensor that contains bias, or anything that can be considered bias in the context of learning algorithms."))

(defprotocol RnnParameters
  "An object that has trainable parameters in the context of recurrent networks."
  (weights-layer [this] "Weights tensor.")
  (weights-iter [this] "Weights iteration tensor.")
  (bias-layer [this] "Bias tensor.")
  (bias-iter [this] "Bias iteration tensor."))

(defprotocol ParametersSeq  ;;TODO consider making parameters a map instead of a vector.
  "An object that has a sequence of parameters."
  (parameters [this]))

(defprotocol DiffParameters
  "An object that has weights gradients (typically objects that can be used in training)."
  (diff-weights [this] "Gradient weights tensor."))

(defprotocol Initializable
  "An object that can be initialized with a provided function `init-fn`."
  (init [this init-fn] "Initialize `this` with `init-fn`."))

(defprotocol DiffTransfer
  "An object that can transfer gradient data."
  (diff-input [this] "Gradients at the input of `this`.")
  (diff-z [this] "Gradients after the linear part of `this` transformation.")
  (diff-output [this] "Gradients at the output of `this`."))

(defprotocol Backprop
  "An object that can be trained with the backpropagation algorithm."
  (forward [this] [this hyperparam]
    "Do the forward pass with the provided `hyperparam` s relevant to the encompassing backprop
     variants such as stochastic gradient descent or Adam.")
  (backward [this] [this hyperparam]
    "Do the backward pass with the provided `hyperparam` s relevant to the encompassing backprop
     variants such as stochastic gradient descent or Adam."))

(defprotocol LinearBackprop
  "An object that provides hooks of the linear part of the backward pass."
  (backward-diff [this scal-diff-w scal-g scal-diff-b scal-b]
    "Compute the linear part of the backward gradient computation of the backprop algorithm."))

(defprotocol Workspace
  "An object that can inform interested parties of the capacity of workspace that it needs."
  (inf-ws-size [this] "Necessary workspace size during inference.")
  (train-ws-size [this] "Necessary workspace size during training."))

(def ^:dynamic *workspace* nil)

(defprotocol BatchDescriptor
  "An object that can provide information relevant for batch based computation."
  (batch-index [this] "The index of batch in tensor shape (typically 0, but can be 1 for RNN)."))
