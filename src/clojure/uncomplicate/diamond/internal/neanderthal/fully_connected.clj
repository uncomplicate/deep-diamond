;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.diamond.internal.neanderthal.fully-connected
  (:require [uncomplicate.commons
             [core :refer [Releaseable release let-release with-release Info info]]
             [utils :refer [dragan-says-ex]]]
            [uncomplicate.neanderthal
             [core :refer [rk! mm! mv! trans axpy! axpby! view view-ge mrows
                           ncols vctr zero dim transfer!]]
             [real :refer [entry! nrm2 asum]]
             [math :refer [sqr pow sqrt]]
             [vect-math :refer [linear-frac! linear-frac mul! log! log sqrt! sqr!]]
             [block :refer [buffer]]]
            [uncomplicate.neanderthal.internal
             [api :refer [flow]]
             [printing :refer [print-vector]]]
            [uncomplicate.diamond
             [tensor :as tz
              :refer [Transfer input output connector view-tz revert shape layout
                      TensorDescriptor shape]]
             [dnn :refer [Parameters bias weights transfer-parameters!]]]
            [uncomplicate.diamond.internal.protocols
             :refer [BlueprintProvider DiamondFactoryProvider DiffParameters
                     diff-bias diff-weights Backprop forward backward blueprint
                     create-tensor activ-blueprint]]
            [uncomplicate.diamond.internal.dnnl.core :refer [memory-desc data-type]])
  (:import clojure.lang.IFn))

(deftype FullyConnectedInference [fact bluep ones activ
                                  x b w a
                                  src-conn bias-tz weights-tz dst-tz]
  Releaseable
  (release [_]
    (release src-conn)
    (release bias-tz)
    (release weights-tz)
    (release dst-tz)
    (release activ))
  Object
  (hashCode [_]
    (-> (hash :fc) (hash-combine (info activ :activation))
        (hash-combine weights-tz) (hash-combine weights-tz)))
  (equals [_ layer]
    (and (satisfies? Parameters layer) (= :fc (info layer :topology))
         (= (info activ :activation) (info layer :activation))
         (= bias-tz (bias layer)) (= weights-tz (weights layer))))
  (toString [_]
    (str bluep))
  Info
  (info [x]
    (assoc (info activ)
           :shape (info bluep :shape)
           :bias (info bias-tz)
           :weights (info weights-tz)
           :dst (info dst-tz)
           :topology :fc :algorithm :inference))
  (info [x info-type]
    (case info-type
      :topology :fc :algorithm :inference
      :bias (info bias-tz)
      :weights (info weights-tz)
      :dst (info dst-tz)
      (or (info activ info-type) (info bluep info-type))))
  DiamondFactoryProvider
  (diamond-factory [_]
    fact)
  BlueprintProvider
  (blueprint [_]
    bluep)
  Parameters
  (bias [_]
    bias-tz)
  (weights [_]
    weights-tz)
  Transfer
  (input [this]
    (input src-conn))
  (output [_]
    (output activ))
  IFn
  (invoke [_]
    (src-conn)
    (rk! 1.0 b ones (mm! 1.0 w x 0.0 a))
    (activ)))

(deftype FullyConnectedSGD [fact bluep ones activ prop-diff?
                            v a-1 b w z
                            src-conn bias-tz weights-tz dst-tz]
  Releaseable
  (release [_]
    "TODO"
    (release activ)
    (release v))
  Object
  (hashCode [_]
    (-> (hash :fc) (hash-combine (info activ :activation))
        (hash-combine weights-tz) (hash-combine bias-tz)))
  (equals [_ layer]
    (and (satisfies? Parameters layer) (= :fc (info layer :topology))
         (= (info activ :activation) (info layer :activation))
         (= bias-tz (bias layer)) (= weights-tz (weights layer))))
  (toString [_]
    (str bluep))
  Info
  (info [x]
    (assoc (info activ) :shape (info bluep :shape)
           :shape (info bluep :shape)
           :bias (info bias-tz)
           :weights (info weights-tz)
           :dst (info dst-tz)
           :batch (dim ones) :algorithm :sgd :topology :fc))
  (info [x info-type]
    (case info-type
      :batch (dim ones) :algorithm :sgd :topology :fc
      :bias (info bias-tz)
      :weights (info weights-tz)
      :dst (info dst-tz)
      (or (info activ info-type) (info bluep info-type))))
  DiamondFactoryProvider
  (diamond-factory [_]
    fact)
  BlueprintProvider
  (blueprint [_]
    bluep)
  Parameters
  (bias [_]
    bias-tz)
  (weights [_]
    weights-tz)
  Transfer
  (input [this]
    (input src-conn))
  (output [_]
    (output activ))
  IFn
  (invoke [_]
    (src-conn)
    (rk! 1.0 b ones (mm! 1.0 w a-1 0.0 z))
    (activ))
  Backprop
  (forward [this]
    (forward activ)
    this)
  (forward [this [_ _ mu nesterov?]]
    (when nesterov? (axpy! mu v w))
    (src-conn)
    (rk! 1.0 b ones (mm! 1.0 w a-1 0.0 z))
    (forward activ)
    this)
  (backward [this]
    (backward activ)
    this)
  (backward [this [_ eta lambda mu nesterov?]]
    (let [eta-avg (- (/ (double eta) (dim ones)))]
      (when nesterov? (axpy! (- (double mu)) v w))
      (mm! eta-avg z (trans a-1) mu v)
      (when prop-diff?
        (mm! 1.0 (trans w) z 0.0 a-1))
      (mv! eta-avg z ones 1.0 b)
      (axpby! 1.0 v (inc (* eta-avg (double lambda))) w)
      this)))

(defn sgd-layer [fact bluep activ-bluep ones prop-diff?
                 a-1 b w a src-conn bias-tz weights-tz a-tz]
  (let-release [z-tz (create-tensor fact a-tz false)
                z (trans (view-ge (view z-tz) (ncols a-1) (dim b)))
                v (zero w)
                activ (activ-bluep z-tz a-tz)]
    (->FullyConnectedSGD fact bluep ones activ prop-diff?
                         v a-1 b w z
                         src-conn bias-tz weights-tz a-tz)))

(deftype FullyConnectedBlueprint [fact activ-bluep src-desc bias-desc weights-desc dst-desc]
  Releaseable
  (release [_]
    (release activ-bluep))
  Object
  (hashCode [_]
    (-> (hash :fc) (hash-combine activ-bluep)))
  (equals [_ other]
    (and (instance? FullyConnectedBlueprint other)
         (= activ-bluep (.activ-bluep ^FullyConnectedBlueprint other))
         ));;TODO implement equals
  (toString [_]
    (pr-str {:shape (shape dst-desc)
             :topology :fc
             :activation (info activ-bluep :activation)}))
  Info
  (info [x]
    "TODO")
  (info [x info-type]
    "TODO")
  DiamondFactoryProvider
  (diamond-factory [_]
    fact)
  BlueprintProvider
  (blueprint [this]
    this)
  ;; DescProvider
  ;; (desc [_]
  ;;   (desc activ-bluep))
  TensorDescriptor
  (shape [_]
    (shape dst-desc))
  (data-type [_]
    (tz/data-type dst-desc))
  (layout [_]
    (layout dst-desc))
  IFn
  (invoke [this prev-layer]
    (let [src-shape (shape src-desc)
          n (long (get src-shape 0))]
      (let-release [src-conn (connector (output prev-layer) src-desc)
                    bias-tz (create-tensor fact bias-desc false)
                    weights-tz (create-tensor fact weights-desc false)
                    a-tz (create-tensor fact dst-desc false)
                    x (trans (view-ge (view (output src-conn))
                                      (long (get src-shape 0)) (apply * (rest src-shape))))
                    b (view bias-tz)
                    ones (entry! (vctr x (ncols x)) 1.0)
                    activ (activ-bluep a-tz)]
        (->FullyConnectedInference fact this ones activ x b
                                   (trans (view-ge (view weights-tz) (mrows x) (dim b)))
                                   (trans (view-ge (view a-tz) n (dim b)))
                                   src-conn bias-tz weights-tz a-tz))))
  (invoke [this prev-layer prop-diff? optimization]
    (let [src-shape (shape src-desc)
          training-layer (case optimization
                           :sgd sgd-layer
                           :adam nil ;;TODO adam-layer
                           (dragan-says-ex
                            "This optimization algorithm is not available in Neanderthal backend."
                            {:optimization optimization}))]
      (let-release [src-conn (connector (output prev-layer) src-desc)
                    bias-tz (create-tensor fact bias-desc false)
                    weights-tz (create-tensor fact weights-desc false)
                    a-tz (create-tensor fact dst-desc false)
                    x (trans (view-ge (view (output src-conn))
                                      (long (get src-shape 0)) (apply * (rest src-shape))))
                    b (view bias-tz)
                    w (trans (view-ge (view weights-tz) (mrows x) (dim b)))
                    a (trans (view-ge (view a-tz) (ncols x) (dim b)))
                    ones (entry! (vctr x (ncols x)) 1.0)]
        (training-layer fact this activ-bluep ones prop-diff? x b w a
                        src-conn bias-tz weights-tz a-tz))))
  (invoke [this prev-layer prop-diff?]
    (.invoke this prev-layer prop-diff? :sgd)))

(defn neanderthal-fc-blueprint [fact src-desc dst-desc activ alpha beta]
  (let-release [dst-desc (memory-desc (shape dst-desc)
                                      (or (tz/data-type dst-desc) (tz/data-type src-desc))
                                      :nc)
                bias-desc (memory-desc (rest (shape dst-desc)) (tz/data-type dst-desc) :x)
                weights-desc (memory-desc (into [(first (shape bias-desc))] (rest (shape src-desc)))
                                          (tz/data-type dst-desc) :oi)
                activ-bluep (activ-blueprint fact dst-desc activ alpha beta)]
    (->FullyConnectedBlueprint fact activ-bluep src-desc bias-desc weights-desc dst-desc)))


(defmethod transfer! [FullyConnectedInference Object]
  [source destination]
  (transfer-parameters! source destination))

#_(defmethod transfer! [FullyConnectedAdam Object]
  [source destination]
  (transfer-parameters! source destination))

(defmethod transfer! [FullyConnectedSGD Object]
  [source destination]
  (transfer-parameters! source destination))
