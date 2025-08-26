;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.diamond.internal.bnns.directed
  (:require [uncomplicate.commons.core
             :refer [Releaseable release let-release with-release Info info view]]
            [uncomplicate.fluokitten.core :refer [fmap]]
            [uncomplicate.neanderthal.core :refer [axpby! dim transfer! scal! view-vctr]]
            [uncomplicate.neanderthal.internal.api :refer [flow]]
            [uncomplicate.diamond.tensor :as tz
             :refer [Transfer input output connector revert shape TensorDescriptor view-tz]]
            [uncomplicate.diamond.internal
             [protocols
              :refer [bias weights Parameters ParametersSeq parameters DescriptorProvider
                      DiamondFactoryProvider DiffParameters diff-weights Backprop forward backward
                      DiffTransfer diff-input diff-output diff-z LinearBackprop backward-diff
                      inf-desc train-desc diff-desc Initializable init batch-index create-tensor]]
             [utils :refer [transfer-weights-bias! concat-strides concat-dst-shape direction-count]]]
            [uncomplicate.diamond.internal.bnns
             [protocols :refer [desc]]
             [core :refer :all]
             [tensor :refer [bnns-tensor bnns-transformer]]]
            [uncomplicate.diamond.internal.neanderthal.directed
             :refer [->DirectedLayerBlueprint ->GaussianDropoutBlueprint ->NopActivation
                     ->NopActivationBlueprint]])
  (:import [clojure.lang IFn AFn]
           [uncomplicate.diamond.internal.neanderthal.directed DirectedLayerBlueprint GaussianDropoutBlueprint]))

;; ================================ Activation =============================================

(deftype BnnsActivationInference [bluep activ data-tz]
  Releaseable
  (release [_]
    (release data-tz))
  Info
  (info [this]
    {:activation (info bluep :activation)
     :data (info data-tz)})
  (info [this info-type]
    (case info-type
      :data (info data-tz)
      (info bluep info-type)))
  Transfer
  (input [_]
    data-tz)
  (output [_]
    data-tz)
  Initializable
  (init [this _]
    this)
  IFn
  (invoke [_]
    (apply-filter activ data-tz data-tz)
    data-tz)
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(deftype BnnsActivationTraining [bluep activ
                                 src-tz dst-tz diff-dst-tz diff-src-tz]
  Releaseable
  (release [_]
    (release src-tz)
    (release dst-tz)
    (release diff-dst-tz)
    (release diff-src-tz))
  Info
  (info [this]
    {:activation (info bluep :activation)
     :src (info src-tz)
     :dst (info dst-tz)
     :diff-dst (info diff-dst-tz)
     :diff-src (info diff-src-tz)})
  (info [this info-type]
    (case info-type
      :src (info src-tz)
      :dst (info dst-tz)
      :diff-dst (info diff-dst-tz)
      :diff-src (info diff-src-tz)
      (info bluep info-type)))
  Transfer
  (input [_]
    src-tz)
  (output [_]
    dst-tz)
  DiffTransfer
  (diff-input [_]
    diff-dst-tz)
  (diff-output [_]
    diff-src-tz)
  Initializable
  (init [this _]
    this)
  IFn
  (invoke [_]
    (apply-filter activ src-tz dst-tz)
    dst-tz)
  (applyTo [this xs]
    (AFn/applyToHelper this xs))
  Backprop
  (forward [this]
    (apply-filter activ src-tz dst-tz)
    this)
  (backward [this]
    (apply-filter-backward activ diff-src-tz dst-tz diff-dst-tz)
    this))

(deftype BnnsActivationBlueprint [fact activ activ-inf activ-train
                                  inf-desc train-desc diff-desc]
  Releaseable
  (release [_]
    (release activ-inf)
    (release activ-train)
    (release inf-desc)
    (release train-desc)
    (release diff-desc))
  Info
  (info [this]
    {:activation activ})
  (info [this info-type]
    (case info-type
      :activation activ
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
  (shape [_]
    (shape train-desc))
  (data-type [_]
    (data-type train-desc))
  (layout [_]
    (layout train-desc))
  IFn
  (invoke [this src-tz]
    (->BnnsActivationInference this activ-inf src-tz))
  (invoke [this src-tz diff-src-tz]
    (let-release [dst-tz (bnns-tensor fact (view train-desc) (batch-index src-tz))]
      (->BnnsActivationTraining this activ-train src-tz dst-tz diff-src-tz diff-src-tz)))
  (applyTo [this xs]
    (AFn/applyToHelper this xs)))

(defn bnns-nop-activation-blueprint
  [fact inf-src-desc train-src-desc diff-desc]
  (let-release [inf-src-desc (view (desc inf-src-desc))
                train-src-desc (view (desc train-src-desc))
                diff-desc (view (desc diff-desc))]
    (->NopActivationBlueprint fact inf-src-desc train-src-desc diff-desc)))

(defn bnns-activation-blueprint
  ([fact inf-desc train-desc activ alpha beta]
   (let-release [inf-desc (view (desc inf-desc))
                 train-desc (view (desc train-desc))]
     (with-release [activ-fn (activation activ alpha beta)
                    inf-params (activation-params activ-fn inf-desc)
                    train-params (activation-params activ-fn train-desc)]
       (let-release [activ-inf (layer inf-params)
                     activ-train (layer train-params)]
         (->BnnsActivationBlueprint fact activ activ-inf activ-train inf-desc train-desc train-desc)))))
  ([fact data-desc activ alpha beta]
   (bnns-activation-blueprint fact data-desc data-desc activ alpha beta)))

(defn bnns-activ-blueprint
  ([fact inf-src-desc train-src-desc diff-desc activ alpha beta]
   (case activ
     :identity (bnns-nop-activation-blueprint fact inf-src-desc train-src-desc diff-desc)
     ;;:softmax (bnns-softmax-blueprint fact eng inf-src-desc train-src-desc)
     (bnns-activation-blueprint fact inf-src-desc train-src-desc  activ alpha beta)))
  ([fact data-desc activ alpha beta]
   (bnns-activ-blueprint fact data-desc data-desc data-desc activ alpha beta)))

;; ================================ Inner Product & Convolution ====================================
