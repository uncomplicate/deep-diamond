;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.diamond.dnn
  (:require [uncomplicate.commons
             [core :refer [with-release let-release release view]]
             [utils :refer [dragan-says-ex]]]
            [uncomplicate.fluokitten.core :refer [fmap foldmap]]
            [uncomplicate.neanderthal
             [core :refer [ncols transfer! view-vctr]]
             [random :refer [rand-normal! rng-state]]]
            [uncomplicate.diamond.tensor
             :refer [*diamond-factory* shape input output batcher TensorContainer
                     tensor data-type layout desc]]
            [uncomplicate.diamond.internal
             [protocols :as api]
             [network :refer [sequential-network parallel-network]]
             [utils :refer [default-strides]]]))

(defn activation
  ([fact src-desc activ alpha beta]
   (api/activ-blueprint (api/diamond-factory fact) src-desc activ alpha beta))
  ([fact src-desc activ alpha]
   (api/activ-blueprint (api/diamond-factory fact) src-desc activ alpha 0.0))
  ([fact src-desc activ]
   (api/activ-blueprint (api/diamond-factory fact) src-desc activ 0.0 0.0))
  ([src-desc activ]
   (api/activ-blueprint *diamond-factory* src-desc activ 0.0 0.0)))

(defn inner-product
  ([fact src-desc dst-desc weights-type]
   (api/inner-product-blueprint (api/diamond-factory fact) src-desc dst-desc weights-type))
  ([fact src-desc dst-desc]
   (api/inner-product-blueprint (api/diamond-factory fact) src-desc dst-desc nil))
  ([src-desc dst-desc]
   (api/inner-product-blueprint *diamond-factory* src-desc dst-desc nil)))

(defn coerce-fc-dst [src-desc dst-desc]
  (let [src-shape (shape src-desc)
        dst-shape (shape dst-desc)
        n (get src-shape 0)]
    (if (< 1 (count dst-shape))
      dst-desc
      {:shape [n (get dst-shape 0)]
       :data-type (data-type dst-desc)
       :layout (layout dst-desc)})))

(defn fully-connected
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
  "Another name for fully-connected."
  ([dst-desc activ args]
   (fully-connected dst-desc activ args))
  ([dst-desc activ]
   (fully-connected dst-desc activ nil)))

(defn coerce-conv-shapes [src-shape kernel-shape dst-shape
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
  "TODO"
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
  ([dst-desc kernel-desc activ args]
   (convolution dst-desc kernel-desc activ args))
  ([dst-desc kernel-desc activ]
   (convolution dst-desc kernel-desc activ nil)))

(defn coerce-pooling-dst [src-shape dst-shape]
  (let [[n c] src-shape
        missing-cnt (- (count src-shape) (count dst-shape))]
    (if (= 0 missing-cnt)
      dst-shape
      (into (if (= 1 missing-cnt) [n] [n c]) dst-shape))))

(defn pooling
  "TODO"
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

(defn dropout-mask [src-desc ^long mask-dim]
  (let [src-shape (shape src-desc)]
    (into (vec (repeat (- (count src-shape) mask-dim) 1)) (take-last mask-dim src-shape))))

(defn dropout
  "TODO"
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
  "TODO"
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
  "TODO"
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
  "TODO"
  ([^long concat-dimension]
   (concatenate concat-dimension))
  ([]
   (concatenate 0)))

(defn coerce-branch-dst [src-desc branch-dim dst-descs]
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
  "TODO"
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
  "TODO"
  ([fact src-desc ^long n]
   (api/split-blueprint fact src-desc n))
  ([^long n]
   (fn
     ([fact src-desc]
      (split fact src-desc n))
     ([src-desc]
      (split *diamond-factory* src-desc n)))))

(defn sum
  "TODO"
  ([fact src-descs]
   (api/sum-blueprint fact src-descs))
  ([]
   (fn
     ([fact src-desc]
      (sum fact src-desc))
     ([src-desc]
      (sum *diamond-factory* src-desc)))))

(defn cost
  "TODO"
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
  "TODO"
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
  "TODO"
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
  ([net in-batcher out-batcher cost! epochs hyperparam]
   (let [b-size (api/source-size in-batcher)
         mb-size (api/minibatch-size in-batcher)
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
         (in-batcher (* n mb-size))
         (out-batcher (* n mb-size))
         (api/forward net hyperparam)
         (api/forward cost!)
         (api/backward cost!)
         (api/backward net hyperparam)))
     (net)
     (cost!)))
  ([net in-batcher out-batcher cost! options]
   (map (fn [[epochs hyperparam]]
          (train* net in-batcher out-batcher cost! epochs hyperparam))
        options)))

(defn train
  "TODO"
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
           (train* net in out-batcher cost! epochs hyperparam))
         :default (train* net in out cost! epochs hyperparam)))
  ([net in out cost! options]
   (cond (keyword? cost!)
         (with-release [cost! (cost net cost!)]
           (doall (train net in out cost! options)))
         (satisfies? TensorContainer in)
         (with-release [in-batcher (batcher in (input net))]
           (doall (train net in-batcher out cost! options)))
         (satisfies? TensorContainer out)
         (with-release [out-batcher (batcher out (api/diff-input cost!))]
           (doall (train* net in out-batcher cost! options)))
         :default (train* net in out cost! options))))

(defn ^:private infer* [net in-batcher out-batcher]
  (let [b-size (api/source-size in-batcher)
        mb-size (api/minibatch-size in-batcher)
        mb-count (quot b-size mb-size)
        mb-rem (rem b-size mb-size)]
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
  "TODO"
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

(defn coerce-rnn-dst [src-desc dst-desc]
  (let [dst-shape (shape dst-desc)
        [t n c] (shape src-desc)]
    (case (count dst-shape)
      0 src-desc
      1 {:shape [t n (get dst-shape 0)]
         :data-type (data-type dst-desc)
         :layout (layout dst-desc)}
      2 {:shape [(get dst-shape 0) n (get (dst-shape 1))]
         :data-type (data-type dst-desc)
         :layout (layout dst-desc)}
      dst-desc)))

(defn rnn-op
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
  ([fact src-desc dst-desc lrs activ args]
   (let [alpha (or (:alpha args) (if (= activ :linear) 1.0 0.0))
         beta (or (:beta args) 0.0)]
     (api/rnn-blueprint (api/diamond-factory fact) src-desc (coerce-rnn-dst src-desc dst-desc)
                        lrs activ alpha beta
                        (:weights-type args) (:src-iter args) (:dst-iter args))))
  ([fact src-desc dst-desc activ args]
   (rnn fact src-desc dst-desc 1 activ args))
  ([dst-desc lrs activ args]
   (fn
     ([fact src-desc]
      (rnn fact src-desc dst-desc lrs activ args))
     ([src-desc]
      (rnn *diamond-factory* src-desc dst-desc lrs activ args))))
  ([dst-desc activ args]
   (rnn dst-desc 1 activ args))
  ([dst-desc activ]
   (rnn dst-desc 1 activ nil))
  ([dst-desc]
   (rnn dst-desc 1 :relu nil))
  ([]
   (rnn [] 1 :relu nil)))

(defn lstm-op
  ([fact src-desc dst-desc weights-type dir lrs src-iter? dst-iter?]
   (api/lstm-op-blueprint (api/diamond-factory fact) src-desc (coerce-rnn-dst src-desc dst-desc)
                          weights-type dir lrs src-iter? dst-iter?))
  ([fact src-desc dst-desc dir lrs src-iter? dst-iter?]
   (api/lstm-op-blueprint (api/diamond-factory fact) src-desc (coerce-rnn-dst src-desc dst-desc)
                          nil dir lrs src-iter? dst-iter?))
  ([fact src-desc dst-desc lrs src-iter? dst-iter?]
   (api/lstm-op-blueprint (api/diamond-factory fact) src-desc (coerce-rnn-dst src-desc dst-desc)
                          nil :unidirectional lrs src-iter? dst-iter?))
  ([fact src-desc dst-desc lrs]
   (lstm-op fact src-desc dst-desc lrs false false))
  ([src-desc dst-desc lrs]
   (lstm-op *diamond-factory* src-desc dst-desc lrs)))

(defn lstm
  ([fact src-desc dst-desc lrs args]
   (api/lstm-blueprint (api/diamond-factory fact) src-desc (coerce-rnn-dst src-desc dst-desc)
                       lrs (:weights-type args) (:src-iter args) (:dst-iter args)))
  ([fact src-desc dst-desc args]
   (lstm fact src-desc dst-desc 1 args))
  ([dst-desc lrs args]
   (fn
     ([fact src-desc]
      (lstm fact src-desc dst-desc lrs args))
     ([src-desc]
      (lstm *diamond-factory* src-desc dst-desc lrs args))))
  ([dst-desc args]
   (lstm dst-desc 1 args))
  ([dst-desc]
   (lstm dst-desc 1 nil))
  ([]
   (lstm [] 1 nil)))

(defn gru-op
  ([fact src-desc dst-desc weights-type dir lrs src-iter? dst-iter?]
   (api/gru-op-blueprint (api/diamond-factory fact) src-desc (coerce-rnn-dst src-desc dst-desc)
                          weights-type dir lrs src-iter? dst-iter?))
  ([fact src-desc dst-desc dir lrs src-iter? dst-iter?]
   (api/gru-op-blueprint (api/diamond-factory fact) src-desc (coerce-rnn-dst src-desc dst-desc)
                          nil dir lrs src-iter? dst-iter?))
  ([fact src-desc dst-desc lrs src-iter? dst-iter?]
   (api/gru-op-blueprint (api/diamond-factory fact) src-desc (coerce-rnn-dst src-desc dst-desc)
                          nil :unidirectional lrs src-iter? dst-iter?))
  ([fact src-desc dst-desc lrs]
   (gru-op fact src-desc dst-desc lrs false false))
  ([src-desc dst-desc lrs]
   (gru-op *diamond-factory* src-desc dst-desc lrs)))

(defn gru
  ([fact src-desc dst-desc lrs args]
   (api/gru-blueprint (api/diamond-factory fact) src-desc (coerce-rnn-dst src-desc dst-desc)
                       lrs (:weights-type args) (:src-iter args) (:dst-iter args)))
  ([fact src-desc dst-desc args]
   (gru fact src-desc dst-desc 1 args))
  ([dst-desc lrs args]
   (fn
     ([fact src-desc]
      (gru fact src-desc dst-desc lrs args))
     ([src-desc]
      (gru *diamond-factory* src-desc dst-desc lrs args))))
  ([dst-desc args]
   (gru dst-desc 1 args))
  ([dst-desc]
   (gru dst-desc 1 nil))
  ([]
   (gru [] 1 nil)))

(defn ending
  ([fact src-desc dst-type]
   (api/ending-blueprint fact src-desc (or dst-type (data-type src-desc))))
  ([dst-type]
   (fn
     ([fact src-desc]
      (ending fact src-desc dst-type))
     ([src-desc]
      (ending *diamond-factory* src-desc dst-type))))
  ([]
   (ending nil)))
