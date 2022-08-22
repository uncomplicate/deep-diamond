;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.diamond.internal.neanderthal.directed
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
             [protocols :refer [Parameters bias weights ParametersSeq parameters DescriptorProvider
                                DiamondFactoryProvider DiffParameters diff-weights Backprop forward
                                backward create-tensor activ-blueprint DiffTransfer diff-input
                                diff-output diff-z create-tensor-desc LinearBackprop backward-diff
                                Workspace inf-ws-size train-ws-size neanderthal-factory inf-desc
                                train-desc diff-desc Initializable init batch-index
                                RnnParameters weights-layer weights-iter]]
             [utils :refer [transfer-weights-bias!]]])
  (:import [clojure.lang IFn AFn]))

(deftype NopActivation [bluep data diff]
  Releaseable
  (release [_]
    (release data)
    (release diff))
  Info
  (info [this]
    {:activation (info bluep :activation)
     :data (fmap info data)
     :diff (fmap info diff)})
  (info [this info-type]
    (case info-type
      :data (fmap info data)
      :diff (fmap info diff)
      (info bluep info-type)))
  Transfer
  (input [_]
    data)
  (output [_]
    data)
  DiffTransfer
  (diff-input [_]
    diff)
  (diff-output [_]
    diff)
  IFn
  (invoke [_]
    data)
  (applyTo [this xs]
    (AFn/applyToHelper this xs))
  Backprop
  (forward [this]
    this)
  (backward [this]
    this))

(deftype NopActivationBlueprint [fact inf-desc train-desc diff-desc]
  Releaseable
  (release [_]
    (release inf-desc)
    (release train-desc))
  Info
  (info [this]
    {:activation :identity})
  (info [this info-type]
    (case info-type
      :activation :identity
      nil))
  DiamondFactoryProvider
  (diamond-factory [_]
    fact)
  DescriptorProvider
  (inf-desc [_]
    inf-desc)
  (train-desc [_]
    train-desc)
  (diff-desc [_]
    diff-desc)
  TensorDescriptor
  (shape [this]
    (shape train-desc))
  (data-type [this]
    (data-type train-desc))
  (layout [this]
    (layout train-desc))
  IFn
  (invoke [this data-tz]
    (->NopActivation this data-tz data-tz))
  (invoke [this data-tz diff-data-tz]
    (->NopActivation this data-tz diff-data-tz))
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(deftype InnerProductInference [ones b w a-1 z
                                src-conn bias-tz weights-tz dst-tz]
  Releaseable
  (release [_]
    (release ones)
    (release b)
    (release w)
    (release a-1)
    (release z)
    (release src-conn)
    (release bias-tz)
    (release weights-tz)
    (release dst-tz))
  Info
  (info [this]
    {:bias (info bias-tz)
     :weights (info weights-tz)
     :dst (info dst-tz)})
  (info [this info-type]
    (case info-type
      :bias (info bias-tz)
      :weights (info weights-tz)
      :dst (info dst-tz)
      nil))
  Transfer
  (input [_]
    (input src-conn))
  (output [_]
    dst-tz)
  Parameters
  (bias [_]
    bias-tz)
  (weights [_]
    weights-tz)
  ParametersSeq
  (parameters [_]
    [weights-tz bias-tz])
  IFn
  (invoke [_]
    (src-conn)
    (rk! 1.0 b ones (mm! 1.0 w a-1 0.0 z))
    dst-tz)
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(deftype InnerProductTraining [ones prop-diff? b w a-1 z diff-w diff-b
                               src-conn bias-tz weights-tz dst-tz
                               diff-weights-tz diff-src-conn]
  Releaseable
  (release [_]
    (release ones)
    (release b)
    (release w)
    (release a-1)
    (release z)
    (release diff-w)
    (release diff-b)
    (release src-conn)
    (release bias-tz)
    (release weights-tz)
    (release dst-tz)
    (release diff-weights-tz)
    (release diff-src-conn))
  Info
  (info [this]
    {:bias (info bias-tz)
     :weights (info weights-tz)
     :dst (info dst-tz)
     :diff-weights (info diff-weights-tz)})
  (info [this info-type]
    (case info-type
      :bias (info bias-tz)
      :weights (info weights-tz)
      :dst (info dst-tz)
      :diff-weights (info diff-weights-tz)
      nil))
  Transfer
  (input [_]
    (input src-conn))
  (output [_]
    dst-tz)
  DiffTransfer
  (diff-input [_]
    dst-tz)
  (diff-output [_]
    (output diff-src-conn))
  Parameters
  (bias [_]
    bias-tz)
  (weights [_]
    weights-tz)
  ParametersSeq
  (parameters [_]
    [weights-tz bias-tz])
  DiffParameters
  (diff-weights [_]
    diff-weights-tz)
  IFn
  (invoke [this]
    (forward this)
    dst-tz)
  (applyTo [this xs]
    (AFn/applyToHelper this xs))
  Backprop
  (forward [this]
    (src-conn)
    (rk! 1.0 b ones (mm! 1.0 w a-1 0.0 z))
    this)
  (backward [this]
    (backward-diff this 1.0 0.0 1.0 0.0))
  LinearBackprop
  (backward-diff [this scal-diff-w scal-g scal-diff-b scal-b]
    (mm! scal-diff-w z (trans a-1) scal-g diff-w)
    (mv! scal-diff-b z ones scal-b diff-b)
    (when prop-diff?
      (mm! 1.0 (trans w) z 0.0 a-1)
      (diff-src-conn))
    this))

(deftype InnerProductBlueprint [fact ones src-desc weights-desc bias-desc dst-desc]
  Object
  (hashCode [_]
    (-> (hash :inner-product)
        (hash-combine src-desc) (hash-combine weights-desc)
        (hash-combine bias-desc) (hash-combine dst-desc)))
  (equals [_ other]
    (and (instance? InnerProductBlueprint other)
         (= src-desc (.src-desc ^InnerProductBlueprint other))
         (= dst-desc (.dst-desc ^InnerProductBlueprint other))))
  (toString [this]
    (str {:src src-desc :weights weights-desc :dst dst-desc}))
  Releaseable
  (release [_]
    (release ones)
    (release src-desc)
    (release weights-desc)
    (release bias-desc)
    (release dst-desc))
  Info
  (info [this info-type]
    (case info-type
      :bias bias-desc
      :inference {:src src-desc
                  :weights weights-desc
                  :dst dst-desc}
      :training {:src src-desc
                 :weights weights-desc
                 :dst dst-desc}
      nil))
  (info [this]
    {:bias bias-desc
     :inference {:src src-desc
                 :weights weights-desc
                 :dst dst-desc}
     :training {:src src-desc
                :weights weights-desc
                :dst dst-desc}})
  DiamondFactoryProvider
  (diamond-factory [_]
    fact)
  DescriptorProvider
  (inf-desc [_]
    (view dst-desc))
  (train-desc [_]
    (view dst-desc))
  (diff-desc [_]
    (view dst-desc))
  TensorDescriptor
  (shape [_]
    (shape dst-desc))
  (data-type [_]
    (data-type dst-desc))
  (layout [_]
    (layout dst-desc))
  IFn
  (invoke [_ src-tz]
    (let [src-shape (shape src-tz)]
      (let-release [src-conn (connector src-tz src-desc)
                    bias-tz (create-tensor fact (view bias-desc) false)
                    weights-tz (create-tensor fact (view weights-desc) false)
                    dst-tz (create-tensor fact (view dst-desc) false)
                    x (view-ge (view-vctr (output src-conn))
                               (apply * (rest src-shape)) (long (get src-shape 0)))
                    b (view-vctr bias-tz)]
        (->InnerProductInference (view ones)
                                 b (trans (view-ge (view-vctr weights-tz) (mrows x) (dim b)))
                                 x (view-ge (view-vctr dst-tz) (dim b) (ncols x))
                                 src-conn bias-tz weights-tz dst-tz))))
  (invoke [_ src-tz diff-src-tz prop-diff? _]
    (let [src-shape (shape src-tz)]
      (let-release [src-conn (connector src-tz src-desc)
                    bias-tz (create-tensor fact (view bias-desc) false)
                    weights-tz (create-tensor fact (view weights-desc) false)
                    dst-tz (create-tensor fact (view dst-desc) false)
                    diff-src-conn (connector src-desc diff-src-tz)
                    diff-weights-tz (raw weights-tz)
                    x (view-ge (view-vctr (output src-conn))
                               (apply * (rest src-shape)) (long (get src-shape 0)))
                    b (view-vctr bias-tz)
                    w (trans (view-ge (view-vctr weights-tz) (mrows x) (dim b)))
                    diff-w (trans (view-ge (view-vctr diff-weights-tz) (mrows x) (dim b)))
                    a (view-ge (view-vctr dst-tz) (dim b) (ncols x))]
        (->InnerProductTraining (view ones) prop-diff?
                                b w x a diff-w b
                                src-conn bias-tz weights-tz dst-tz
                                diff-weights-tz diff-src-conn))))
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(defn inner-product-blueprint
  ([fact src-desc dst-desc weights-type]
   (let [src-shape (shape src-desc)
         dst-shape (shape dst-desc)
         src-type (data-type src-desc)
         dst-type (data-type dst-desc)
         weights-shape [(get dst-shape 1) (apply * (rest src-shape))]
         weights-type (or weights-type (data-type src-desc) dst-type)]
     (let-release [dst-desc (create-tensor-desc fact
                                                [(dst-shape 0) (apply * (rest dst-shape))]
                                                (or dst-type src-type)
                                                :nc)
                   bias-desc (create-tensor-desc fact [(dst-shape 1)] dst-type :x)
                   weights-desc (create-tensor-desc fact weights-shape weights-type :oi)
                   ones (entry! (vctr (neanderthal-factory fact src-type) (long (get src-shape 0))) 1.0)]
       (->InnerProductBlueprint fact ones src-desc weights-desc bias-desc dst-desc)))))

(defmethod transfer! [InnerProductInference Object]
  [source destination]
  (transfer-weights-bias! source destination))

(defmethod transfer! [InnerProductTraining Object]
  [source destination]
  (transfer-weights-bias! source destination))

(deftype InferenceLayer [fact bluep op activ]
  Releaseable
  (release [_]
    (release op)
    (release activ))
  Object
  (hashCode [_]
    (-> (hash (info bluep :topology)) (hash-combine (info activ :activation))
        (hash-combine (weights op)) (hash-combine (bias op))))
  (equals [_ layer]
    (and (satisfies? Parameters layer) (= (info bluep :topology) (info layer :topology))
         (= (info activ :activation) (info layer :activation))
         (= (bias op) (bias layer)) (= (weights op) (weights layer))))
  (toString [_]
    (str bluep))
  Info
  (info [x]
    (assoc (into (info op) (info activ)) :shape (info bluep :shape)
           :topology (info bluep :topology) :algorithm :inference))
  (info [x info-type]
    (case info-type
      :algorithm :inference
      (or (info activ info-type) (info op info-type) (info bluep info-type))))
  DiamondFactoryProvider
  (diamond-factory [_]
    fact)
  Parameters
  (bias [_]
    (bias op))
  (weights [_]
    (weights op))
  RnnParameters
  (weights-layer [this]
    (weights-layer op))
  (weights-iter [this]
    (weights-iter op))
  ParametersSeq
  (parameters [_]
    (parameters op))
  Initializable
  (init [this init-fn]
    (init-fn (bias op))
    (init-fn (weights op))
    this)
  Transfer
  (input [this]
    (input op))
  (output [_]
    (output activ))
  DiffTransfer
  (diff-input [this]
    (output activ))
  (diff-output [_]
    (dragan-says-ex "Inference layer does not calculate gradients."))
  IFn
  (invoke [_]
    (op)
    (activ))
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(defmethod print-method InferenceLayer
  [layer ^java.io.Writer w]
  (.write w (format "#Inference[topology%s, shape:%s, activation: %s]\n..........\n weights: %s\n..........\n bias: %s"
                    (info layer :topology) (info layer :shape) (info layer :activation)
                    (pr-str (weights layer)) (pr-str (bias layer)))))

(deftype SGDLayer [fact bluep op activ ^long n v w b]
  Releaseable
  (release [_]
    (release op)
    (release activ)
    (release v)
    (release w)
    (release b))
  Object
  (hashCode [_]
    (-> (hash (info bluep :topology)) (hash-combine (info activ :activation))
        (hash-combine (weights op)) (hash-combine (bias op))))
  (equals [_ layer]
    (and (satisfies? Parameters layer)
         (= (info bluep :topology) (info layer :topology))
         (= (info activ :activation) (info layer :activation))
         (= (bias op) (bias layer)) (= (weights op) (weights layer))))
  (toString [_]
    (str bluep))
  Info
  (info [x]
    (assoc (into (info op) (info activ)) :shape (info bluep :shape)
           :batch n :algorithm :sgd :topology (info bluep :topology) ))
  (info [x info-type]
    (case info-type
      :batch n
      :algorithm :sgd
      (or (info activ info-type) (info op info-type) (info bluep info-type))))
  DiamondFactoryProvider
  (diamond-factory [_]
    fact)
  Transfer
  (input [this]
    (input op))
  (output [_]
    (output activ))
  DiffTransfer
  (diff-input [_]
    (diff-input activ))
  (diff-z [_]
    (diff-output activ))
  (diff-output [_]
    (diff-output op))
  Parameters
  (weights [_]
    (weights op))
  (bias [_]
    (bias op))
  RnnParameters
  (weights-layer [this]
    (weights-layer op))
  (weights-iter [this]
    (weights-iter op))
  ParametersSeq
  (parameters [_]
    (parameters op))
  Initializable
  (init [this init-fn]
    (init-fn (bias op))
    (init-fn (weights op))
    this)
  IFn
  (invoke [_]
    (op)
    (activ))
  (applyTo [this xs]
    (AFn/applyToHelper this xs))
  Backprop
  (forward [this]
    (forward activ)
    this)
  (forward [this [_ _ _ mu nesterov?]]
    (when nesterov? (axpy! mu v w))
    (forward op)
    (forward activ)
    this)
  (backward [this]
    (backward activ)
    this)
  (backward [this [_ eta lambda mu nesterov?]]
    (let [eta-avg (- (/ (double eta) n))]
      (when nesterov? (axpy! (- (double mu)) v w))
      (backward-diff op eta-avg mu eta-avg 1.0)
      (axpby! 1.0 v (inc (* eta-avg (double lambda))) w)
      this)))

(defn sgd-layer [fact bluep op-bluep activ-bluep prev-layer prop-diff?]
  (let-release [src-tz (output prev-layer)
                op (op-bluep (output prev-layer) (diff-input prev-layer) prop-diff? true)
                activ (activ-bluep (output op) (diff-input op))]
    (->SGDLayer fact bluep op activ (get (shape src-tz) (batch-index src-tz))
                (view-vctr (diff-weights op)) (view-vctr (weights op))
                (view-vctr (bias op)))))

(defmethod print-method SGDLayer
  [layer ^java.io.Writer w]
  (.write w (format "#SGD[topology:%s, shape:%s, activation: %s]\n..........\n weights: %s\n..........\n bias: %s"
                    (info layer :topology) (info layer :shape) (info layer :activation)
                    (pr-str (weights layer)) (pr-str (bias layer)))))

(deftype AdamLayer [fact bluep op activ ^long n
                    s r w g b]
  Releaseable
  (release [_]
    (release op)
    (release activ)
    (release s)
    (release r)
    (release w)
    (release g)
    (release b))
  Object
  (hashCode [_]
    (-> (hash (info bluep :topology)) (hash-combine (info activ :activation))
        (hash-combine (weights op)) (hash-combine (bias op))))
  (equals [_ layer]
    (and (satisfies? Parameters layer)
         (= (info bluep :topology) (info layer :topology))
         (= (info activ :activation) (info layer :activation))
         (= (bias op) (bias layer)) (= (weights op) (weights layer))))
  (toString [_]
    (str bluep))
  Info
  (info [x]
    (assoc (into (info op) (info activ)) :shape (info bluep :shape)
           :batch n :algorithm :adam :topology (info bluep :topology) ))
  (info [x info-type]
    (case info-type
      :batch n
      :algorithm :adam
      (or (info activ info-type) (info op info-type) (info bluep info-type))))
  DiamondFactoryProvider
  (diamond-factory [_]
    fact)
  Transfer
  (input [this]
    (input op))
  (output [_]
    (output activ))
  DiffTransfer
  (diff-input [_]
    (diff-input activ))
  (diff-z [_]
    (diff-output activ))
  (diff-output [_]
    (diff-output op))
  Parameters
  (weights [_]
    (weights op))
  (bias [_]
    (bias op))
  RnnParameters
  (weights-layer [this]
    (weights-layer op))
  (weights-iter [this]
    (weights-iter op))
  ParametersSeq
  (parameters [_]
    (parameters op))
  Initializable
  (init [this init-fn]
    (init-fn (bias op))
    (init-fn (weights op))
    this)
  IFn
  (invoke [_]
    (op)
    (activ))
  (applyTo [this xs]
    (AFn/applyToHelper this xs))
  Backprop
  (forward [this]
    (forward activ)
    this)
  (forward [this _]
    (forward op)
    (forward activ)
    this)
  (backward [this]
    (backward activ)
    this)
  (backward [this [t eta lambda rho1 rho2 epsilon]]
    (let [t (inc (long t))
          eta (double (or eta 0.001))
          lambda (double (or lambda 0.0))
          rho1 (double (or rho1 0.9))
          rho2 (double (or rho2 0.999))
          epsilon (double (or epsilon 1e-6))
          eta-avg (- (/ (double eta) n))
          scal-n (/ 1.0 n)]
      (backward-diff op scal-n 0.0 eta-avg 1.0)
      (axpby! (- 1.0 rho1) g rho1 s)
      (axpby! (- 1.0 rho2) (sqr! g) rho2 r)
      (linear-frac! (/ (- eta) (- 1.0 (pow rho1 t))) s 0.0
                    (/ 1.0 (sqrt (- 1.0 (pow rho2 t)))) (sqrt! r g) epsilon g)
      (axpby! 1.0 g (inc (* eta-avg lambda)) w)
      this)))

(defn adam-layer [fact bluep op-bluep activ-bluep prev-layer prop-diff?]
  (let-release [src-tz (output prev-layer)
                op (op-bluep src-tz (diff-input prev-layer) prop-diff? false)
                activ (activ-bluep (output op) (diff-input op))
                w (view-vctr (weights op))
                s (zero w)
                r (zero w)]
    (->AdamLayer fact bluep op activ (get (shape src-tz) (batch-index src-tz))
                 s r w (view-vctr (diff-weights op)) (view-vctr (bias op)))))

(defmethod print-method AdamLayer
  [layer ^java.io.Writer w]
  (.write w (format "#Adam[topology:%s, shape:%s, activation: %s]\n..........\n weights: %s\n..........\n bias: %s"
                    (info layer :topology) (info layer :shape) (info layer :activation)
                    (pr-str (weights layer)) (pr-str (bias layer)))))

(deftype DirectedLayerBlueprint [fact topology op-bluep activ-bluep]
  Releaseable
  (release [_]
    (release op-bluep)
    (release activ-bluep))
  Object
  (hashCode [_]
    (-> (hash topology) (hash-combine activ-bluep) (hash-combine op-bluep)))
  (equals [_ other]
    (and (instance? DirectedLayerBlueprint other)
         (= activ-bluep (.activ-bluep ^DirectedLayerBlueprint other))
         (= op-bluep (.op-bluep ^DirectedLayerBlueprint other))))
  (toString [this]
    (str {:shape (shape this)
          :topology topology
          :activation (info activ-bluep :activation)}))
  Info
  (info [x]
    (assoc (into (info op-bluep) (info activ-bluep))
           :shape (shape activ-bluep) :topology topology))
  (info [x info-type]
    (case info-type
      :shape (shape activ-bluep)
      :topology topology
      (or (info activ-bluep info-type) (info op-bluep info-type))))
  DiamondFactoryProvider
  (diamond-factory [_]
    fact)
  DescriptorProvider
  (inf-desc [_]
    (inf-desc activ-bluep))
  (train-desc [_]
    (train-desc activ-bluep))
  (diff-desc [_]
    (diff-desc activ-bluep))
  TensorDescriptor
  (shape [_]
    (shape activ-bluep))
  (data-type [_]
    (data-type activ-bluep))
  (layout [_]
    (layout activ-bluep))
  Workspace
  (inf-ws-size [this]
    (max (long (inf-ws-size op-bluep)) (long (inf-ws-size activ-bluep))))
  (train-ws-size [this]
    (max (long (train-ws-size op-bluep)) (long (train-ws-size activ-bluep))))
  IFn
  (invoke [this prev-layer]
    (let-release [op (op-bluep (fmap output prev-layer))
                  activ (activ-bluep (output op))]
      (->InferenceLayer fact this op activ)))
  (invoke [this prev-layer prop-diff? optimization]
    (if-let [training-layer (case optimization
                              :sgd sgd-layer
                              :adam adam-layer
                              (dragan-says-ex
                               (format "Optimization algorithm %s is not available." optimization)
                               {:requested optimization :available [:sgd :adam]}))]
      (training-layer fact this op-bluep activ-bluep prev-layer prop-diff?)))
  (invoke [this prev-layer prop-diff?]
    (.invoke this prev-layer prop-diff? :sgd))
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(defmethod print-method DirectedLayerBlueprint
  [bp ^java.io.Writer w]
  (.write w (str bp)))

(defn neanderthal-fc-blueprint [fact src-desc dst-desc activ alpha beta weights-type]
  (let [dst-shape (shape dst-desc)
        weights-shape [(dst-shape 1) (apply * (rest (shape src-desc)))]]
    (let-release [src-desc (create-tensor-desc fact src-desc)
                  dst-desc (create-tensor-desc fact
                                               [(dst-shape 0) (apply * (rest dst-shape))]
                                               (or (data-type dst-desc) (data-type src-desc))
                                               :nc)
                  ip-bluep (inner-product-blueprint fact src-desc dst-desc weights-type)
                  activ-bluep (activ-blueprint fact (view dst-desc) activ alpha beta)]
      (->DirectedLayerBlueprint fact :dense ip-bluep activ-bluep))))

(defmethod transfer! [InferenceLayer Object]
  [^InferenceLayer source destination]
  (transfer! (.op source) destination))

(defmethod transfer! [AdamLayer Object]
  [^AdamLayer source destination]
  (transfer! (.op source) destination))

(defmethod transfer! [SGDLayer Object]
  [^SGDLayer source destination]
  (transfer! (.op source) destination))

;; ================================ Gaussian Dropout ======================================

(deftype IdentityLayer [fact data-conn]
  Releaseable
  (release [_]
    (release data-conn))
  Object
  (hashCode [_]
    (hash-combine (hash :identity) (output data-conn)))
  (equals [_ other]
    (and (instance? IdentityLayer other)
         (let [conn (.data-conn ^IdentityLayer other)]
           (or (= (input data-conn) (input conn))
               (= (output data-conn) (output conn))))))
  (toString [this]
    (str {:shape (shape (output data-conn))
          :topology :identity}))
  DiamondFactoryProvider
  (diamond-factory [_]
    fact)
  Transfer
  (input [_]
    (input data-conn))
  (output [_]
    (output data-conn))
  ParametersSeq
  (parameters [_]
    [])
  Initializable
  (init [this _]
    this)
  IFn
  (invoke [_]
    (data-conn))
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(defmethod print-method IdentityLayer
  [layer ^java.io.Writer w]
  (.write w (format "#Identity[shape:%s]\n" (shape (output layer)))))

(deftype GaussianDropoutLayer [fact bluep prev-layer ^double sd rand-state mask-tz data-conn]
  Releaseable
  (release [_]
    (release mask-tz)
    (release data-conn))
  Object
  (hashCode [_]
    (hash-combine (hash :gaussian-dropout) mask-tz))
  (equals [_ other]
    (and (instance? GaussianDropoutLayer other)
         (= mask-tz (.mask-tz ^GaussianDropoutLayer other))
         (= (output data-conn) (output (.data-conn  ^GaussianDropoutLayer other)))))
  (toString [this]
    (str bluep))
  Info
  (info [this]
    {:rand-state rand-state
     :mask mask-tz
     :data (info (output data-conn))
     :topology :gaussian-dropout
     :shape (info bluep :shape)})
  (info [this info-type]
    (case info-type
      :rand-state rand-state
      :mask mask-tz
      :data (info (output data-conn))
      :topology :gaussian-dropout
      (info bluep info-type)))
  DiamondFactoryProvider
  (diamond-factory [_]
    fact)
  Transfer
  (input [_]
    (input data-conn))
  (output [_]
    (output data-conn))
  DiffTransfer
  (diff-input [_]
    (output data-conn))
  (diff-z [_]
    (diff-z prev-layer))
  (diff-output [_]
    (input data-conn))
  ParametersSeq
  (parameters [_]
    [])
  Initializable
  (init [this _]
    this)
  IFn
  (invoke [this]
    (data-conn))
  (applyTo [this xs]
    (AFn/applyToHelper this xs))
  Backprop
  (forward [this]
    this)
  (forward [this _]
    (data-conn)
    (mul! (output data-conn) (rand-normal! rand-state 1.0 sd mask-tz))
    this)
  (backward [this]
    this)
  (backward [this _]
    (data-conn)
    (mul! (output data-conn) mask-tz)
    this))

(defmethod print-method GaussianDropoutLayer
  [^GaussianDropoutLayer layer ^java.io.Writer w]
  (.write w (format "#GaussianDropout[shape:%s]\n output: %s\n"
                    (shape (output layer)) (pr-str (.mask-tz layer)))))

(deftype GaussianDropoutBlueprint [fact ^double sd mask-desc]
  Releaseable
  (release [_]
    (release mask-desc))
  Object
  (hashCode [_]
    (-> (hash :gaussian-dropout) (hash-combine mask-desc)))
  (equals [_ other]
    (and (instance? GaussianDropoutBlueprint other)
         (= mask-desc (.mask-desc ^GaussianDropoutBlueprint other))))
  (toString [this]
    (str {:shape (shape mask-desc)
          :topology :gaussian-dropout}))
  Info
  (info [x]
    {:shape (shape mask-desc)
     :topology :gaussian-dropout})
  (info [x info-type]
    (case info-type
      :shape (shape mask-desc)
      :topology :gaussian-dropout
      nil))
  DiamondFactoryProvider
  (diamond-factory [_]
    fact)
  DescriptorProvider
  (inf-desc [_]
    (view mask-desc))
  (train-desc [_]
    (view mask-desc))
  (diff-desc [_]
    (view mask-desc))
  TensorDescriptor
  (shape [this]
    (shape mask-desc))
  (data-type [this]
    (data-type mask-desc))
  (layout [this]
    (layout mask-desc))
  IFn
  (invoke [this prev-layer]
    (let-release [data-conn (connector (output prev-layer) (view mask-desc))]
      (->IdentityLayer fact data-conn)))
  (invoke [this prev-layer _ _]
    (let-release [src-tz (output prev-layer)
                  data-conn (connector src-tz (view mask-desc))
                  mask-tz (create-tensor fact (view mask-desc) false)]
      (->GaussianDropoutLayer fact this prev-layer sd (rng-state mask-tz) mask-tz data-conn)))
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(defn neanderthal-gaussian-dropout-blueprint [fact src-desc sd]
  (let-release [mask-desc (create-tensor-desc fact src-desc)]
    (->GaussianDropoutBlueprint fact sd mask-desc)))

(defmethod print-method GaussianDropoutBlueprint
  [bp ^java.io.Writer w]
  (.write w (str bp)))

(defmethod transfer! [IdentityLayer Object]
  [source destination]
  destination)

(defmethod transfer! [GaussianDropoutLayer Object]
  [source destination]
  destination)
