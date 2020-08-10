(ns uncomplicate.diamond.functional.boston-housing-prices.regression-test
  (:require [midje.sweet :refer [facts throws => roughly]]
            [clojure.java.io :as io]
            [clojure.data.csv :as csv]
            [clojure.string :as string]
            [uncomplicate.commons [core :refer [with-release let-release]]]
            [uncomplicate.neanderthal
             [core :refer [transfer transfer! view native view-ge
                           cols mv! rk! raw col row nrm2 scal! ncols dim rows]]
             [real :refer [entry! entry]]
             [native :refer [native-float fv]]
             [random :refer [rand-uniform!]]
             [math :as math :refer [sqrt]]]
            [uncomplicate.diamond
             [tensor :refer [*diamond-factory* tensor connector transformer
                             desc revert shape input output view-tz batcher]]
             [dnn :refer [sum activation inner-product fully-connected
                          network init! train cost train]]]
            [uncomplicate.diamond.internal.dnnl.factory :refer [dnnl-factory]]
            [uncomplicate.diamond.internal.neanderthal.factory :refer [neanderthal-factory]]
#_            [uncomplicate.diamond.internal.cudnn.factory :refer [cudnn-factory]]))

(defonce boston-housing-raw
  (csv/read-csv (slurp (io/resource "uncomplicate/diamond/functional/boston_housing_prices/boston-housing.csv"))))

(defonce boston-housing
  (doall (shuffle (map #(mapv (fn [^String x] (Double/valueOf x)) %) (drop 1 boston-housing-raw)))))

(defonce x-train (transfer native-float (map (partial take 13) (take 404 boston-housing))))

(defonce y-train (transfer native-float (map (partial drop 13) (take 404 boston-housing))))

(defonce x-test (transfer native-float (map (partial take 13) (drop 404 boston-housing))))

(defonce y-test (transfer native-float (map (partial drop 13) (drop 404 boston-housing))))

(defn standardize!
  ([a!]
   (let-release [row-means (raw (col a! 0))]
     (when (< 0 (dim a!))
       (with-release [ones (entry! (raw (row a! 0)) 1)]
         (mv! (/ -1.0 (ncols a!)) a! ones row-means)
         (standardize! row-means a!)))
     row-means))
  ([row-means a!]
   (when (< 0 (dim a!))
     (with-release [ones (entry! (raw (row a! 0)) 1)]
       (rk! row-means ones a!)
       (doseq [x-mean (rows a!)]
         (let [s (double (nrm2 x-mean))]
           (if (= 0.0 s)
             x-mean
             (scal! (/ (sqrt (ncols a!)) s) x-mean))))))
   a!))

(standardize! x-train)
(standardize! x-test)

(defn test-boston-regression [fact]
  (with-release [x-tz (tensor fact [404 13] :float :nc)
                 x-mb-tz (tensor fact [16 13] :float :nc)
                 y-tz (tensor fact [404 1] :float :nc)
                 y-mb-tz (tensor fact [16 1] :float :nc)
                 net-bp (network fact x-mb-tz
                                 [(fully-connected [64] :relu)
                                  (fully-connected [64] :relu)
                                  (fully-connected [1] :linear)])
                 net (init! (net-bp x-mb-tz :adam))
                 net-infer (net-bp x-mb-tz)
                 quad-cost (cost net y-mb-tz :quadratic)
                 mean-abs-cost (cost net-infer y-mb-tz :mean-absolute)
                 x-batcher (batcher x-tz (input net))
                 y-batcher (batcher y-tz y-mb-tz)]
    (facts "Adam gradient descent - Boston Housing Prices."
           (transfer! x-train (view x-tz))
           (transfer! y-train (view y-tz))
           (time (train net x-batcher y-batcher quad-cost 80 [])) => (roughly 6.0 5)true

           (transfer! net net-infer)
           (net-infer)
           (mean-abs-cost) => (roughly 3 2))))

(with-release [fact (dnnl-factory)]
  (test-boston-regression fact))

(with-release [fact (neanderthal-factory)]
  (test-boston-regression fact))

#_(with-release [fact (cudnn-factory)];;TODO
  (test-boston-regression fact))
