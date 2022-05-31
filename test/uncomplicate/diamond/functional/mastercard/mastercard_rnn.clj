(ns uncomplicate.diamond.functional.mastercard.mastercard-rnn
  (:require [midje.sweet :refer [facts throws => roughly]]
            [clojure.java.io :as io]
            [clojure.data.csv :as csv]
            [clojure.string :as string]
            [uncomplicate.commons [core :refer [with-release let-release release]]]
            [uncomplicate.neanderthal
             [core :refer [ge dim amax submatrix subvector mrows trans transfer transfer! view-vctr native view-ge
                           cols mv! rk! raw col row nrm2 scal! ncols dim rows]]
             [real :refer [entry! entry]]
             [native :refer [fge native-float fv]]
             [random :refer [rand-uniform!]]
             [math :refer [pi sqrt]]
             [vect-math :refer [linear-frac!]]]
            [uncomplicate.diamond
             [tensor :refer [*diamond-factory* tensor offset! connector transformer
                             desc revert shape input output view-tz batcher]]
             [dnn :refer [lstm gru infer sum activation inner-product fully-connected
                          network init! train cost train ending]]]
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

(defn test-timeseries [fact rnn]
  (let [[x-train y-train] (split-series fact mastercard-train 64)]
    (try (with-release [net-bp (network fact (desc [64 32 1] :float :tnc)
                                        [(rnn 128)
                                         (ending)
                                         (fully-connected [1] :linear)])
                        net (init! (net-bp :adam))
                        net-infer (net-bp)]

           (facts "Adam gradient descent - learning increment with RNN."

                  (time (train net x-train y-train :quadratic 1 [])) => (roughly 0.0 0.5)
                  (transfer! net net-infer)
                  ))
         (finally
           (release x-train)
           (release y-train)
           ))))

(with-release [fact (dnnl-factory)]
  (test-timeseries fact lstm))
;; "Elapsed time: 348.86187 msecs"

(with-release [fact (dnnl-factory)]
  (test-timeseries fact gru))
;;"Elapsed time: 445.922226 msecs"



;; (with-release [fact (neanderthal-factory)]
;;   (test-boston-regression fact))

;; (with-release [fact (cudnn-factory)]
;;   (test-boston-regression fact))
