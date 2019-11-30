(ns uncomplicate.diamond.dnn
  (:require [uncomplicate.commons.utils :refer [dragan-says-ex]]
            [uncomplicate.neanderthal
             [core :refer [ncols view]]
             [random :refer [rand-normal! rand-uniform! rng-state]]]
            [uncomplicate.diamond.tensor :refer [*diamond-factory* shape input]]
            [uncomplicate.diamond.internal
             [protocols :as api]
             [network :refer [sequential-network]]]))

(defprotocol Parameters
  (weights [this])
  (bias [this]))

(defn sum
  ([^double scale dst]
   (let [dst (api/create-tensor-desc *diamond-factory* dst)]
     (api/create-sum *diamond-factory* scale dst)))
  ([dst ^double scale src]
   (let [fact *diamond-factory*]
     (api/create-sum fact (api/create-tensor-desc fact dst)
                     scale (api/create-tensor-desc fact src) nil)))
  ([fact dst scale src & scales-srcs]
   (api/create-sum (api/factory fact) (api/create-tensor-desc fact dst)
                   scale (api/create-tensor-desc fact src)
                   (map #(if (number? %) % (api/create-tensor-desc fact %))
                        scales-srcs))))

(defn activation
  ([fact src-desc activ alpha beta]
   (api/activ-blueprint (api/factory fact) src-desc activ alpha beta))
  ([fact src-desc activ alpha]
   (api/activ-blueprint (api/factory fact) src-desc activ alpha 0.0))
  ([fact src-desc activ]
   (api/activ-blueprint (api/factory fact) src-desc activ 0.0 0.0))
  ([src-desc activ]
   (api/activ-blueprint *diamond-factory* src-desc activ 0.0 0.0)))

(defn inner-product
  ([fact src-desc dst-desc weights-type]
   (api/inner-product-blueprint (api/factory fact) src-desc dst-desc weights-type))
  ([fact src-desc dst-desc]
   (api/inner-product-blueprint (api/factory fact) src-desc dst-desc nil))
  ([src-desc dst-desc]
   (api/inner-product-blueprint *diamond-factory* src-desc dst-desc nil)))

(defn fully-connected
  ([fact src-desc dst-desc activ args]
   (let [alpha (or (:alpha args) (if (= activ :linear) 1.0 0.0))
         beta (or (:beta args) 0.0)]
     (api/fc-blueprint (api/factory fact) src-desc dst-desc
                       activ alpha beta (:weights-type args))))
  ([fact src-desc dst-desc activ]
   (api/fc-blueprint (api/factory fact) src-desc dst-desc activ
                     (if (= activ :linear) 1.0 0.0) 0.0 nil))
  ([dst-desc activ args]
   (fn
     ([fact src-desc]
      (fully-connected fact src-desc dst-desc activ args))
     ([]
      dst-desc)))
  ([dst-desc activ]
   (fn
     ([fact src-desc]
      (fully-connected fact src-desc dst-desc activ))
     ([]
      dst-desc))))

(defn cost
  ([layer train-tz cost]
   ((case cost
      :quadratic api/quadratic-cost
      :mean-absolute api/mean-absolute-cost
      :sigmoid-crossentropy api/sigmoid-crossentropy-cost
      (dragan-says-ex "This cost function is not supported." {:cost cost}))
    (api/factory layer) layer train-tz))
  ([layer train-tz]
   (api/quadratic-cost (api/factory layer) layer train-tz)))

(defn network
  ([fact src-desc layers]
   (sequential-network (api/factory fact) src-desc layers))
  ([src-desc layers]
   (network *diamond-factory* src-desc layers)))

(defn train
  ([network cost epochs hyperparam]
   (dotimes [n epochs]
     (api/forward network hyperparam)
     (api/forward cost)
     (api/backward cost)
     (api/backward network hyperparam))
   (network)
   (cost))
  ([network cost options]
   (map (fn [[epochs hyperparam]] (train network cost epochs hyperparam)) options)))

(defn init! [network!]
  (doseq [layer (api/layers network!)]
    (rand-normal! 0.0 (/ 1.0 (double (apply * (rest (shape (input layer)))))) (view (weights layer)))
    (rand-normal! (view (bias layer))))
  network!)

(defn sgd-train
  ([network in-shuff out-shuff cost epochs hyperparam]
   (let [indices (range (first (shape (input in-shuff))))
         batch-size (first (shape (input network)))]
     (dotimes [n (long epochs)]
       (let [batches (partition batch-size (shuffle indices))]
         (doseq [batch batches]
           (in-shuff batch)
           (out-shuff batch)
           (api/forward network hyperparam)
           (api/forward cost)
           (api/backward cost)
           (api/backward network hyperparam))))
     (network)
     (cost)))
  ([network in-shuff out-shuff cost options]
   (map (fn [[epochs hyperparam]] (sgd-train network in-shuff out-shuff cost epochs hyperparam)) options)))
