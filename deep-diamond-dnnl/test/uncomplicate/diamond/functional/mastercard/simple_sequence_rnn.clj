(ns uncomplicate.diamond.functional.mastercard.simple-sequence-rnn
  #_(:require [midje.sweet :refer [facts throws => roughly]]
            [uncomplicate.commons.core :refer [with-release let-release release]]
            [uncomplicate.neanderthal
             [core :refer [ge dim amax submatrix subvector mrows trans transfer transfer! view-vctr
                           native view-ge cols mv! rk! raw col row nrm2 scal! ncols dim rows axpby!]]
             [native :refer [fge native-float fv iv]]]
            [uncomplicate.diamond
             [tensor :refer [*diamond-factory* tensor offset! connector transformer
                             desc revert shape input output view-tz batcher]]
             [dnn :refer [rnn infer sum activation inner-product dense
                          network init! train! cost train-shuffle! abbreviate]]
             [native :refer []]]
            [uncomplicate.diamond.internal.dnnl.factory :refer [dnnl-factory]]
            [uncomplicate.diamond.internal.neanderthal.factory :refer [neanderthal-factory]]
            [uncomplicate.diamond.internal.cudnn.factory :refer [cudnn-factory]]))


;; (def simple-sequence (range -100 100))

;; (def full-series (fge 1 200 simple-sequence))

;; (defn split-series [fact s ^long t]
;;   (let [n (- (ncols s) t)
;;         c (mrows s)]
;;     (let-release [x-tz (tensor fact [t n c] :float :tnc)
;;                   y-tz (tensor fact [n c] :float :nc)
;;                   x-ge (trans (view-ge (view-vctr x-tz) (* n c) t))
;;                   s-vctr (view-vctr s)]
;;       (transfer! (submatrix s 0 t c n) (view-ge (view-vctr y-tz) c n))
;;       (dotimes [j t]
;;         (transfer! (subvector s-vctr (* j c) (* c n)) (row x-ge j)))
;;       [x-tz y-tz])))


;; (def train-data (split-series *diamond-factory* full-series 5))

;; (def net-bp (network (desc [5 32 1] :float :tnc)
;;                      [(rnn [128] :gru)
;;                       (rnn 2)
;;                       (abbreviate)
;;                       (dense [128] :relu)
;;                       (dense [1] :linear)]))

;; (def net (init! (net-bp :adam)))

;; (time (train-shuffle! net (train-data 0) (train-data 1) :quadratic 50 [0.005]))

;; (def net-infer (net-bp))

;; (transfer! net net-infer)

;; (def question (tensor [5 1 1] :float :tnc))
;; (transfer! [1 2 3 4 5] question)

;; (infer! net question)

;; (transfer! [1 2 3 4 5] (view-ge (view-vctr (input net)) 5 32))

;; (def net1-bp (network (desc [5 1 1] :float :tnc)
;;                       [(rnn [128] :gru)
;;                        (rnn 2 :gru)
;;                        (abbreviate)
;;                        (dense [128] :relu)
;;                        (dense [1] :linear)]))

;; (def net1-infer (net1-bp))

;; (transfer! net net1-infer)

;; (net1-infer question)

;; (transfer! [10 12 14 16 18] question)

;; (infer! net1-infer question)

;; (transfer! [10 1 100 16 34] question)

;; (infer! net1-infer question)

;; (transfer! [1000 1001 1002 1003 1004] question)

;; (infer! net1-infer question)

;; (transfer! [37.4 38.4 39.4 40.4 41.4] question)

;; (infer! net1-infer question)

;; (def architecture [(rnn [128] :gru)
;;                    (rnn 2)
;;                    (abbreviate)
;;                    (dense [128] :relu)
;;                    (dense [1] :linear)])

;; (def nvidia (cudnn-factory))
;; (def gpu-net-bp (network nvidia (desc [5 32 1] :float :tnc)
;;                          architecture))
;; (def gpu-net (init! (gpu-net-bp :adam)))

;; (def gpu-train-data (split-series nvidia full-series 5))

;; (time (train-shuffle! gpu-net (gpu-train-data 0) (gpu-train-data 1) :quadratic 1000 [0.005]))

;; (transfer! gpu-net net1-infer)
