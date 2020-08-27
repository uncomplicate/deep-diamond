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
            [uncomplicate.neanderthal
             [core :refer [ncols transfer! view-vctr]]
             [random :refer [rand-normal! rand-uniform! rng-state]]]
            [uncomplicate.diamond.tensor
             :refer [*diamond-factory* shape input output batcher TensorContainer
                     tensor data-type layout desc]]
            [uncomplicate.diamond.internal
             [protocols :as api]
             [network :refer [sequential-network]]]))

(defn sum
  ([^double scale dst]
   (let [dst (api/create-tensor-desc *diamond-factory* dst)]
     (api/create-sum *diamond-factory* scale dst)))
  ([^double scale-src src ^double scale-dst dst]
   (sum *diamond-factory* scale-src src scale-dst dst))
  ([fact scale-src src scale-dst dst]
   (api/create-sum (api/diamond-factory fact)
                   scale-src src
                   scale-dst dst)))

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
      (fully-connected fact src-desc dst-desc activ args))))
  ([dst-desc activ]
   (fully-connected dst-desc activ nil)))

(defn dense
  "TODO Same as fully-connected."
  ([dst-desc activ args]
   (fully-connected dst-desc activ args))
  ([dst-desc activ]
   (fully-connected dst-desc activ nil)))

(defn coerce-conv-shapes [src-shape kernel-shape dst-shape
                          strides padding dilation];;TODO dilation
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
      (convolution fact src-desc kernel-desc dst-desc activ args))))
  ([dst-desc kernel-desc activ]
   (convolution dst-desc kernel-desc activ nil)))

(defn convo
  ([dst-desc kernel-desc activ args]
   (fn
     ([fact src-desc]
      (convolution fact src-desc kernel-desc dst-desc activ args))))
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
      (pooling fact src-desc kernel algo args))))
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
      (dropout fact src-desc sd))))
  ([]
   (dropout 1.0)))

(defn cost
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
  ([fact src-desc layers]
   (sequential-network (api/diamond-factory fact) src-desc layers))
  ([src-desc layers]
   (network *diamond-factory* src-desc layers)))

(defn init! [net!]
  (with-release [rng (rng-state (view-vctr (api/bias (first (api/layers net!)))))]
    (doseq [layer (api/layers net!)]
      (doseq [params (api/parameters layer)]
        (rand-normal! rng 0.0 (/ 1.0 (double (apply * (rest (shape params))))) params))))
  net!)

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
   (let [b-size (long (first (shape (input in-batcher))))
         mb-size (long (first (shape (output in-batcher))))
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
     (cost!))))

(defn train
  ([net cost! epochs hyperparam]
   (if (keyword? cost!)
     (with-release [cost! (cost net cost!)]
       (train* net cost! epochs hyperparam))
     (train* net cost! epochs hyperparam)))
  ([net cost! options]
   (if (keyword? cost!)
     (with-release [cost! (cost net cost!)]
       (train* net cost! options))
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
           (train net in out cost! options))
         (satisfies? TensorContainer in)
         (with-release [in-batcher (batcher in (input net))]
           (train net in-batcher out cost! options))
         (satisfies? TensorContainer out)
         (with-release [out-batcher (batcher out (api/diff-input cost!))]
           (train* net in out-batcher cost! options))
         :default (train* net in out cost! options))))

(defn ^:private infer* [net in-batcher out-batcher]
  (let [b-size (long (first (shape (input in-batcher))))
        mb-size (long (first (shape (output in-batcher))))
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
  ([net in out]
   (cond (satisfies? TensorContainer in)
         (with-release [in-batcher (batcher in (input net))]
           (infer net in-batcher out))
         (satisfies? TensorContainer out)
         (with-release [out-batcher (batcher (output net) out)]
           (infer* net in out-batcher))
         :default (infer* net in out)))
  ([net in]
   (let [net-out (output net)]
     (let-release [out (tensor net (into [(get (shape (input in)) 0)] (rest (shape net-out)))
                               (data-type net-out) (layout net-out))]
       (infer net in out)))))
