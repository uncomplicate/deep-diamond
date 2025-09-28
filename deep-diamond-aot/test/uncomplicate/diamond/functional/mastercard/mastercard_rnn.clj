(ns ^{:author "Dragan Djuric"}
    uncomplicate.diamond.functional.mastercard.mastercard-rnn
  (:require [midje.sweet :refer [facts throws => roughly]]
            [clojure.java.io :as io]
            [clojure.data.csv :as csv]
            [uncomplicate.commons.core :refer [with-release let-release release]]
            [uncomplicate.neanderthal
             [core :refer [ge dim amax submatrix subvector mrows trans transfer transfer! view-vctr
                           native view-ge cols mv! rk! raw col row nrm2 scal! ncols dim rows axpby!]]
             [native :refer [fge native-float fv]]]
            [uncomplicate.diamond
             [tensor :refer [*diamond-factory* tensor offset! connector transformer
                             desc revert shape input output view-tz batcher]]
             [dnn :refer [rnn infer! sum activation inner-product dense
                          network init! train! cost train-shuffle! abbreviate]]]
            [uncomplicate.diamond.internal.dnnl.factory :refer [dnnl-factory]]
            [uncomplicate.diamond.internal.neanderthal.factory :refer [neanderthal-factory]]
            [uncomplicate.diamond.internal.cudnn.factory :refer [cudnn-factory]]))

(defonce mastercard-raw
  (csv/read-csv (slurp (io/resource "mastercard/Mastercard_stock_history.csv"))))

(def mastercard
  (doall (map #(Double/valueOf ^String (get % 2)) (drop 1 mastercard-raw))))

(def mastercard-train (transfer! (take 2904 mastercard) (fge 1 2904)))

(def mastercard-test (transfer! (drop 2904 mastercard) (fge 1 968)))

(defn squash! [a!]
  (scal! (/ (amax a!)) a!))

(squash! mastercard-train)
(squash! mastercard-test)

(defn split-series [fact s ^long t]
  (let [n (- (ncols s) t)
        c (mrows s)]
    (let-release [x-tz (tensor fact [t n c] :float :tnc)
                  y-tz (tensor fact [n c] :float :nc)
                  x-ge (trans (view-ge (view-vctr x-tz) (* n c) t))
                  s-vctr (view-vctr s)]
      (transfer! (submatrix s 0 t c n) (view-ge (view-vctr y-tz) c n))
      (dotimes [j t]
        (transfer! (subvector s-vctr (* j c) (* c n)) (row x-ge j)))
      [x-tz y-tz])))

(defn test-timeseries [fact activ]
  (let [[x-train y-train] (split-series fact mastercard-train 64)
        [x-test y-test] (split-series fact mastercard-test 64)]
    (with-release [x-train x-train
                   y-train y-train
                   x-test x-test
                   y-test y-test
                   net-bp (network fact (desc [64 32 1] :float :tnc)
                                   [(rnn [128] activ)
                                    (rnn 2)
                                    (abbreviate)
                                    (dense [128] :relu)
                                    (dense [1] :linear)])
                   net (init! (net-bp :adam))
                   net-infer (init! (net-bp) :zero)]

      (facts (format "Adam gradient descent - learning MasterCard stock prices with RNN and %s." (str activ))

             (time (train-shuffle! net x-train y-train :quadratic 50 [0.001])) => (roughly 0.0 0.2)
             (transfer! net net-infer)
             (nrm2 (axpby! 1 y-test -1.0 (infer! net-infer x-test))) => (roughly 0.0 3.0)))))

(with-release [fact (dnnl-factory)]
  (test-timeseries fact :gru))
;; "Elapsed time: 56580.76743 msecs" (50 epochs)

#_(with-release [fact (dnnl-factory)]
  (test-timeseries fact :lstm))

#_(with-release [fact (dnnl-factory)]
  (test-timeseries fact :relu))

(with-release [fact (cudnn-factory)]
  (test-timeseries fact :gru)) ;; (50 epochs)
;; "Elapsed time: 21057.697005 msecs" (:standard)
;; "Elapsed time: 57792.445161 msecs" (:static)
;; "Elapsed time: 40416.253165 msecs" (:dynamic)
