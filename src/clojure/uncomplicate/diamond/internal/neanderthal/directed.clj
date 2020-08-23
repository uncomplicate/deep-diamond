;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.diamond.internal.neanderthal.directed
  (:require [uncomplicate.commons
             [core :refer [Releaseable release let-release with-release Info info]]
             [utils :refer [dragan-says-ex]]]
            [uncomplicate.neanderthal
             [core :refer [rk! mm! mv! trans axpy! axpby! view view-ge mrows
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
             [protocols :refer [Parameters bias weights ParametersSeq parameters
                                BlueprintProvider DiamondFactoryProvider DiffParameters
                                diff-weights Backprop forward backward
                                blueprint create-tensor activ-blueprint DiffTransfer
                                diff-input diff-output diff-z create-tensor-desc]]
             [utils :refer [transfer-weights-bias!]]])
  (:import clojure.lang.IFn))

(deftype InnerProductInference [fact bluep ones b w a-1 z
                                src-conn bias-tz weights-tz dst-tz]
  Releaseable
  (release [_]
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
    dst-tz))

(deftype InnerProductTraining [fact bluep ones prop-diff? b w a-1 z diff-w diff-b
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
    (input src-conn))
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
  (invoke [_]
    (src-conn)
    (rk! 1.0 b ones (mm! 1.0 w a-1 0.0 z))
    dst-tz)
  Backprop
  (forward [this]
    (src-conn)
    (rk! 1.0 b ones (mm! 1.0 w a-1 0.0 z))
    this)
  (backward [this]
    (backward this 1.0 0.0 1.0 0.0))
  (backward [this scal-diff-w scal-g scal-diff-b scal-b]
    (mm! scal-diff-w z (trans a-1) scal-g diff-w)
    (mv! scal-diff-b z ones scal-b diff-b)
    (when prop-diff?
      (mm! 1.0 (trans w) z 0.0 a-1)
      (diff-src-conn))
    this))

(deftype InnerProductBlueprint [fact src-desc weights-desc bias-desc dst-desc]
  ;; TODO implement equals
  Releaseable
  (release [_]
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
      :training {:bias bias-desc
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
  TensorDescriptor
  (shape [_]
    (shape dst-desc))
  (data-type [_]
    (data-type dst-desc))
  (layout [_]
    (layout dst-desc))
  IFn
  (invoke [this src-tz]
    (let [src-shape (shape src-tz)]
      (let-release [src-conn (connector src-tz src-desc)
                    bias-tz (create-tensor fact bias-desc false)
                    weights-tz (create-tensor fact weights-desc false)
                    dst-tz (create-tensor fact dst-desc false)
                    x (view-ge (view (output src-conn))
                               (apply * (rest src-shape)) (long (get src-shape 0)))
                    b (view bias-tz)
                    ones (entry! (vctr x (ncols x)) 1.0)]
        (->InnerProductInference fact this ones
                                 b (trans (view-ge (view weights-tz) (mrows x) (dim b)))
                                 x (view-ge (view dst-tz) (dim b) (ncols x))
                                 src-conn bias-tz weights-tz dst-tz))))
  (invoke [this src-tz dst-tz prop-diff? _]
    (let [src-shape (shape src-tz)]
      (let-release [src-conn (connector src-tz src-desc)
                    bias-tz (create-tensor fact bias-desc false)
                    weights-tz (create-tensor fact weights-desc false)
                    diff-src-conn (revert src-conn)
                    diff-weights-tz (zero weights-tz);;TODO raw should be enough.
                    x (view-ge (view (output src-conn))
                               (apply * (rest src-shape)) (long (get src-shape 0)))
                    b (view bias-tz)
                    w (trans (view-ge (view weights-tz) (mrows x) (dim b)))
                    diff-w (trans (view-ge (view diff-weights-tz) (mrows x) (dim b)))
                    a (view-ge (view dst-tz) (dim b) (ncols x))
                    ones (entry! (vctr x (ncols x)) 1.0)]
        (->InnerProductTraining fact this ones prop-diff?
                                b w x a diff-w b
                                src-conn bias-tz weights-tz dst-tz
                                diff-weights-tz diff-src-conn)))))

(defn inner-product-blueprint
  ([fact src-desc dst-desc weights-type]
   (let [dst-shape (shape dst-desc)
         dst-type (data-type dst-desc)
         weights-shape [(dst-shape 1) (apply * (rest (shape src-desc)))]
         weights-type (or weights-type (data-type src-desc) dst-type)]
    (let-release [dst-desc (create-tensor-desc fact
                                               [(dst-shape 0) (apply * (rest dst-shape))]
                                               (or (data-type dst-desc) (data-type src-desc))
                                               :nc)
                  bias-desc (create-tensor-desc fact [(dst-shape 1)] (data-type dst-desc) :x)
                  weights-desc (create-tensor-desc fact weights-shape weights-type :oi)]
      (->InnerProductBlueprint fact src-desc weights-desc bias-desc dst-desc)))))

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
  BlueprintProvider
  (blueprint [_]
    bluep)
  Parameters
  (bias [_]
    (bias op))
  (weights [_]
    (weights op))
  ParametersSeq
  (parameters [_]
    (parameters op))
  Transfer
  (input [this]
    (input op))
  (output [_]
    (output activ))
  IFn
  (invoke [_]
    (op)
    (activ)))

(defmethod print-method InferenceLayer
  [layer ^java.io.Writer w]
  (let [bluep (blueprint layer)]
    (.write w (pr-str {:weights (weights layer) :bias (bias layer)
                       :shape (info bluep :shape)
                       :topology (info bluep :topology)
                       :activation (info bluep :activation)}))))

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
  BlueprintProvider
  (blueprint [_]
    bluep)
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
  ParametersSeq
  (parameters [_]
    (parameters op))
  IFn
  (invoke [_]
    (op)
    (activ))
  Backprop
  (forward [this]
    (forward activ)
    this)
  (forward [this [_ _ mu nesterov?]]
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
      (backward op eta-avg mu eta-avg 1.0);;TODO rename to backward-diff or something
      (axpby! 1.0 v (inc (* eta-avg (double lambda))) w)
      this)))

(defn sgd-layer [fact bluep op-bluep activ-bluep src-tz prop-diff?]
  (let-release [z-tz (create-tensor fact op-bluep false)
                a-tz (create-tensor fact activ-bluep false)
                op (op-bluep src-tz z-tz prop-diff? true)
                activ (activ-bluep z-tz a-tz)]
    (->SGDLayer fact bluep op activ (first (shape bluep))
                (view (diff-weights op)) (view (weights op)) (view (bias op)))))

(defmethod print-method SGDLayer
  [layer ^java.io.Writer w]
  (let [bluep (blueprint layer)]
    (.write w (pr-str {:weights (weights layer) :bias (bias layer)
                       :shape (info bluep :shape)
                       :topology (info bluep :topology)
                       :activation (info bluep :activation)}))))

(deftype AdamLayer [fact bluep op activ ^long n
                    s r w g b]
  Releaseable
  (release [_]
    (release op)
    (release activ)
    (release s)
    (release r)
    (release w))
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
  BlueprintProvider
  (blueprint [_]
    bluep)
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
  ParametersSeq
  (parameters [_]
    (parameters op))
  IFn
  (invoke [_]
    (op)
    (activ))
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
      (backward op scal-n 0.0 eta-avg 1.0)
      (axpby! (- 1.0 rho1) g rho1 s)
      (axpby! (- 1.0 rho2) (sqr! g) rho2 r)
      (linear-frac! (/ (- eta) (- 1.0 (pow rho1 t))) s 0.0
                    (/ 1.0 (sqrt (- 1.0 (pow rho2 t)))) (sqrt! r g) epsilon g)
      (axpby! 1.0 g (inc (* eta-avg lambda)) w)
      this)))

(defn adam-layer [fact bluep op-bluep activ-bluep src-tz prop-diff?]
  (let-release [z-tz (create-tensor fact op-bluep false)
                a-tz (create-tensor fact activ-bluep false)
                op (op-bluep src-tz z-tz prop-diff? false)
                activ (activ-bluep z-tz a-tz)
                w (view (weights op))
                s (zero w)
                r (zero w)]
    (->AdamLayer fact bluep op activ (first (shape bluep))
                 s r w (view (diff-weights op)) (view (bias op)))))

(defmethod print-method AdamLayer
  [layer ^java.io.Writer w]
  (let [bluep (blueprint layer)]
    (.write w (pr-str {:weights (weights layer) :bias (bias layer)
                       :shape (info bluep :shape)
                       :topology (info bluep :topology)
                       :activation (info bluep :activation)}))))

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
    (pr-str {:shape (shape this)
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
  BlueprintProvider
  (blueprint [this]
    this)
  TensorDescriptor
  (shape [_]
    (shape activ-bluep))
  (data-type [_]
    (data-type activ-bluep))
  (layout [_]
    (layout activ-bluep))
  IFn
  (invoke [this prev-layer]
    (let-release [src-tz (output prev-layer)
                  op (op-bluep src-tz)
                  activ (activ-bluep (output op))]
      (->InferenceLayer fact this op activ)))
  (invoke [this prev-layer prop-diff? optimization]
    (let [src-tz (output prev-layer)
          training-layer (case optimization
                           :sgd sgd-layer
                           :adam adam-layer
                           (dragan-says-ex
                            (format "Optimization algorithm %s is not available." optimization)
                            {:optimization optimization}))]
      (training-layer fact this op-bluep activ-bluep src-tz prop-diff?)))
  (invoke [this prev-layer prop-diff?]
    (.invoke this prev-layer prop-diff? :sgd)))

(defn neanderthal-fc-blueprint [fact src-desc dst-desc activ alpha beta weights-type]
  (let [dst-shape (shape dst-desc)
        weights-shape [(dst-shape 1) (apply * (rest (shape src-desc)))]]
    (let-release [src-desc (create-tensor-desc fact src-desc)
                  dst-desc (create-tensor-desc fact
                                               [(dst-shape 0) (apply * (rest dst-shape))]
                                               (or (data-type dst-desc) (data-type src-desc))
                                               :nc)
                  ip-bluep (inner-product-blueprint fact src-desc dst-desc weights-type)
                  activ-bluep (activ-blueprint fact dst-desc activ alpha beta)]
      (->DirectedLayerBlueprint fact :fc ip-bluep activ-bluep))))

(defmethod transfer! [InferenceLayer Object]
  [source destination]
  (transfer-weights-bias! source destination))

(defmethod transfer! [AdamLayer Object]
  [source destination]
  (transfer-weights-bias! source destination))

(defmethod transfer! [SGDLayer Object]
  [source destination]
  (transfer-weights-bias! source destination))

;; ================================ Gaussian Dropout ======================================

(deftype IdentityLayer [prev-layer]
  Transfer
  (input [_]
    (input prev-layer))
  (output [_]
    (output prev-layer))
  ParametersSeq
  (parameters [_]
    [])
  IFn
  (invoke [_]
    (output prev-layer)))

(deftype GaussianDropoutLayer [fact bluep prev-layer ^double sd rand-state
                               mask-tz data-conn revert-data-conn]
  Releaseable
  (release [_]
    (release mask-tz))
  Info
  (info [this]
    {:rand-state rand-state
     :mask mask-tz
     :data (info (output data-conn))})
  (info [this info-type]
    (case info-type
      :rand-state rand-state
      :mask mask-tz
      :data (info (output data-conn))
      (info bluep info-type)))
  DiamondFactoryProvider
  (diamond-factory [_]
    fact)
  Transfer
  (input [_]
    (input data-conn))
  (output [_]
    (input data-conn))
  DiffTransfer
  (diff-input [_]
    (input data-conn))
  (diff-z [_]
    (diff-z prev-layer))
  (diff-output [_]
    (input data-conn))
  ParametersSeq
  (parameters [_]
    [])
  IFn
  (invoke [this]
    (input data-conn))
  Backprop
  (forward [this _]
    (data-conn)
    (mul! (output data-conn) (rand-normal! rand-state 1.0 sd mask-tz))
    (revert-data-conn)
    this)
  (forward [this]
    this)
  (backward [this _]
    (data-conn)
    (mul! (output data-conn) mask-tz)
    (revert-data-conn)
    this)
  (backward [this]
    this))

(deftype GaussianDropoutBlueprint [fact ^double sd data-desc mask-desc]
  Releaseable
  (release [_]
    (release mask-desc))
  DiamondFactoryProvider
  (diamond-factory [_]
    fact)
  BlueprintProvider
  (blueprint [this]
    this)
  TensorDescriptor
  (shape [this]
    (shape data-desc))
  (data-type [this]
    (data-type data-desc))
  (layout [this]
    (layout data-desc))
  IFn
  (invoke [this prev-layer]
    (->IdentityLayer prev-layer))
  (invoke [this prev-layer _ _]
    (let-release [src-tz (output prev-layer)
                  data-conn (connector src-tz data-desc)
                  revert-data-conn (revert data-conn)
                  mask-tz (create-tensor fact data-desc false)]
      (->GaussianDropoutLayer fact this prev-layer
                              sd (rng-state mask-tz)
                              mask-tz data-conn revert-data-conn))))

(defmethod transfer! [IdentityLayer Object]
  [source destination]
  destination)

(defmethod transfer! [GaussianDropoutLayer Object]
  [source destination]
  destination)
