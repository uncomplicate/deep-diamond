(ns uncomplicate.diamond.internal.neanderthal.rnn ;;TODO clean up unnecessary imports
  (:require [uncomplicate.commons
             [core :refer [Releaseable release let-release with-release Info info view]]
             [utils :refer [dragan-says-ex]]]
            [uncomplicate.fluokitten.core :refer [fmap]]
            [uncomplicate.neanderthal
             [core :refer [rk! mm! mv! trans axpy! axpby! view-vctr view-ge mrows
                           ncols vctr zero dim transfer! raw]]
             [real :refer [entry! nrm2 asum]]
             [math :refer [sqr pow sqrt]]
             [vect-math :refer [linear-frac! linear-frac mul! log! log sqrt! sqr!]]
             [random :refer [rand-normal! rng-state]]
             [block :refer [buffer]]]
            [uncomplicate.neanderthal.internal.api :refer [flow]]
            [uncomplicate.diamond
             [tensor :refer [Transfer input output connector view-tz revert shape
                             layout data-type TensorDescriptor]]]
            [uncomplicate.diamond.internal
             [protocols :refer [Parameters RnnParameters bias weights weights-iter ParametersSeq parameters
                                DescriptorProvider DiamondFactoryProvider DiffParameters
                                diff-weights diff-weights-iter Backprop forward backward
                                activ-blueprint DiffTransfer diff-input
                                diff-output diff-z LinearBackprop
                                backward-diff Workspace inf-ws-size train-ws-size
                                neanderthal-factory inf-desc train-desc Initializable init]]
             [utils :refer [transfer-weights-bias!]]])
  (:import [clojure.lang IFn AFn]))

(deftype InferenceRnnLayer [fact bluep op]
  Releaseable
  (release [_]
    (release op))
  Object
  (hashCode [_]
    (-> (hash (info bluep :topology))
        (hash-combine (weights op)) (hash-combine (bias op))
        (hash-combine (weights-iter op))))
  (equals [_ layer]
    (and (satisfies? Parameters layer) (= (info bluep :topology) (info layer :topology))
         (= (bias op) (bias layer)) (= (weights op) (weights layer))
         (= (weights-iter op) (weights-iter layer))))
  (toString [_]
    (str bluep))
  Info
  (info [x]
    (assoc (info op) :shape (info bluep :shape)
           :topology (info bluep :topology) :algorithm :inference))
  (info [x info-type]
    (case info-type
      :algorithm :inference
      (or (info op info-type) (info bluep info-type))))
  DiamondFactoryProvider
  (diamond-factory [_]
    fact)
  Parameters
  (bias [_]
    (bias op))
  (weights [_]
    (weights op))
  RnnParameters
  (weights-iter [_]
    (weights-iter op))
  ParametersSeq
  (parameters [_]
    (parameters op))
  Initializable
  (init [this init-fn]
    (init op init-fn)
    this)
  Transfer
  (input [this]
    (input op))
  (output [_]
    (output op))
  IFn
  (invoke [_]
    (op))
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(defmethod print-method InferenceRnnLayer
  [layer ^java.io.Writer w]
  (.write w (format "#Inference[topology%s, shape:%s]\n..........\n weights: %s\n..........\n weights-iter: %s\n..........\n bias: %s"
                    (info layer :topology) (info layer :shape)
                    (pr-str (weights layer)) (pr-str (weights-iter layer)) (pr-str (bias layer)))))

(deftype SGDRnnLayer [fact bluep op ^long n v w b v-iter w-iter]
  Releaseable
  (release [_]
    (release op)
    (release v)
    (release w)
    (release b))
  Object
  (hashCode [_]
    (-> (hash (info bluep :topology))
        (hash-combine (weights op)) (hash-combine (weights-iter op)) (hash-combine (bias op))))
  (equals [_ layer]
    (and (satisfies? Parameters layer)
         (= (info bluep :topology) (info layer :topology))
         (= (bias op) (bias layer)) (= (weights op) (weights layer))
         (= (weights-iter op) (weights-iter layer))))
  (toString [_]
    (str bluep))
  Info
  (info [x]
    (assoc (info op) :shape (info bluep :shape)
           :batch n :algorithm :sgd :topology (info bluep :topology) ))
  (info [x info-type]
    (case info-type
      :batch n
      :algorithm :sgd
      (or (info op info-type) (info bluep info-type))))
  DiamondFactoryProvider
  (diamond-factory [_]
    fact)
  Transfer
  (input [this]
    (input op))
  (output [_]
    (output op))
  DiffTransfer
  (diff-input [_]
    (diff-input op))
  (diff-z [_]
    (diff-output op))
  (diff-output [_]
    (diff-output op))
  Parameters
  (weights [_]
    (weights op))
  (bias [_]
    (bias op))
  RnnParameters
  (weights-iter [_]
    (weights-iter op))
  ParametersSeq
  (parameters [_]
    (parameters op))
  Initializable
  (init [this init-fn]
    (init op init-fn)
    this)
  IFn
  (invoke [_]
    (op))
  (applyTo [this xs]
    (AFn/applyToHelper this xs))
  Backprop
  (forward [this]
    this)
  (forward [this [_ _ mu nesterov?]]
    (when nesterov?
      (axpy! mu v w)
      (axpy! mu v-iter w-iter))
    (forward op)
    this)
  (backward [this]
    this)
  (backward [this [_ eta lambda mu nesterov?]]
    (let [eta-avg (- (/ (double eta) n))]
      (when nesterov?
        (axpy! (- (double mu)) v w)
        (axpy! (- (double mu)) v-iter w-iter))
      (backward-diff op eta-avg mu eta-avg 1.0)
      (axpby! 1.0 v (inc (* eta-avg (double lambda))) w)
      (axpby! 1.0 v-iter (inc (* eta-avg (double lambda))) w-iter)
      this)))

(defn sgd-rnn-layer [fact bluep op-bluep srcs prop-diff?]
  (let-release [op (op-bluep srcs prop-diff? true)]
    (->SGDRnnLayer fact bluep op (second (shape bluep))
                   (view-vctr (diff-weights op)) (view-vctr (weights op))
                   (view-vctr (bias op))
                   (view-vctr (diff-weights-iter op)) (view-vctr (weights-iter op)))))

(defmethod print-method SGDRnnLayer
  [layer ^java.io.Writer w]
  (.write w (format "#SGD[topology:%s, shape:%s]\n..........\n weights: %s\n..........\n weights-iter: %s\n..........\n bias: %s"
                    (info layer :topology) (info layer :shape)
                    (pr-str (weights layer)) (pr-str (weights-iter layer)) (pr-str (bias layer)))))

(deftype RnnLayerBlueprint [fact topology op-bluep]
  Releaseable
  (release [_]
    (release op-bluep))
  Object
  (hashCode [_]
    (-> (hash topology) (hash-combine op-bluep)))
  (equals [_ other]
    (and (instance? RnnLayerBlueprint other)
         (= op-bluep (.op-bluep ^RnnLayerBlueprint other))))
  (toString [this]
    (str {:shape (shape this)
          :topology topology}))
  Info
  (info [x]
    (assoc (info op-bluep)
           :shape (shape op-bluep) :topology topology))
  (info [x info-type]
    (case info-type
      :shape (shape op-bluep)
      :topology topology
      (info op-bluep info-type)))
  DiamondFactoryProvider
  (diamond-factory [_]
    fact)
  DescriptorProvider
  (inf-desc [_]
    (inf-desc op-bluep))
  (train-desc [_]
    (train-desc op-bluep))
  TensorDescriptor
  (shape [_]
    (shape op-bluep))
  (data-type [_]
    (data-type op-bluep))
  (layout [_]
    (layout op-bluep))
  Workspace
  (inf-ws-size [this]
    (long (inf-ws-size op-bluep)))
  (train-ws-size [this]
    (long (train-ws-size op-bluep)))
  IFn
  (invoke [this prev-layer]
    (->InferenceRnnLayer fact this (op-bluep (fmap output prev-layer))))
  (invoke [this prev-layer prop-diff? optimization]
    (let [training-layer (case optimization
                           :sgd sgd-rnn-layer
                           ;;TODO implement                 :adam adam-rnn-layer
                           (dragan-says-ex
                            (format "Optimization algorithm %s is not available." optimization)
                            {:optimization optimization}))]
      (training-layer fact this op-bluep (fmap output prev-layer) prop-diff?)))
  (invoke [this prev-layer prop-diff?]
    (.invoke this prev-layer prop-diff? :sgd))
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(defmethod print-method RnnLayerBlueprint
  [bp ^java.io.Writer w]
  (.write w (str bp)))

(defn transfer-rnn! [source destination]
  (transfer! (bias source) (bias destination))
  (transfer! (weights source) (weights destination))
  (transfer! (weights-iter source) (weights-iter destination))
  destination)

(defmethod transfer! [InferenceRnnLayer Object]
  [source destination]
  (transfer-rnn! source destination))

#_(defmethod transfer! [AdamRnnLayer Object]
    [source destination]
    (transfer-rnn! source destination))

(defmethod transfer! [SGDRnnLayer Object]
  [source destination]
  (transfer-rnn! source destination))
