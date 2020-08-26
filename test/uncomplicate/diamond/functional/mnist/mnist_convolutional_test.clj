(ns uncomplicate.diamond.functional.mnist.mnist-convolutional-test
  (:require [midje.sweet :refer [facts throws => roughly]]
            [uncomplicate.commons
             [core :refer [let-release Releaseable release with-release view]]
             [utils :refer [random-access channel]]]
            [uncomplicate.neanderthal
             [core :refer [subvector view-ge transfer transfer! dim entry ge mrows
                           transfer! amax imax col native]]
             [real :refer [entry!]]
             [native :as neand :refer [native-byte native-float fge]]]
            [uncomplicate.diamond
             [tensor :refer [tensor transformer connector transformer
                             desc revert shape input output view-tz batcher with-diamond]]
             [native :refer [map-tensor]]
             [dnn :refer [dense convo fully-connected convolution pooling dropout
                          network init! train cost infer]]
             [metrics :refer [confusion-matrix contingency-totals classification-metrics]]]
            [uncomplicate.diamond.internal.dnnl.factory :refer [dnnl-factory]]
            [uncomplicate.diamond.internal.cudnn.factory :refer [cudnn-factory]]
            [uncomplicate.diamond.functional.mnist.mnist-classification-test :as mnist]))

(defonce train-images-file (random-access "data/mnist/train-images.idx3-ubyte"))
(defonce train-labels-file (random-access "data/mnist/train-labels.idx1-ubyte"))
(defonce test-images-file (random-access "data/mnist/t10k-images.idx3-ubyte"))
(defonce test-labels-file (random-access "data/mnist/t10k-labels.idx1-ubyte"))

(defonce train-images (map-tensor train-images-file [60000 1 28 28] :uint8 :nchw :read 16))
(defonce train-labels (map-tensor train-labels-file [60000] :uint8 :x :read 8))
(defonce test-images (map-tensor test-images-file [10000 1 28 28] :uint8 :nchw :read 16))
(defonce test-labels (map-tensor test-labels-file [10000] :uint8 :x :read 8))

(defonce train-labels-float (transfer! train-labels (tensor [60000] :float :x)))
(defonce y-train (mnist/enc-categories train-labels-float))
(defonce test-labels-float (transfer! test-labels (tensor [10000] :float :x)))
(defonce y-test (mnist/enc-categories test-labels-float))

(defn test-mnist-convolutional [fact]
  (with-release [net-bp (network fact (desc [128 1 28 28] :float :nchw)
                                 [(convo [32] [3 3] :relu)
                                  (convo [64] [3 3] :relu)
                                  (pooling [12 12] [2 2] :max);; TODO see whether I need dst-desc here... use default kernel equal to stride?
                                  (dropout)
                                  (dense [128] :relu)
                                  (dropout)
                                  (dense [10] :softmax)])
                 net (init! (net-bp :adam))
                 net-infer (net-bp)
                 train-images (transfer! train-images (tensor fact [60000 1 28 28] :uint8 :nchw))
                 train-labels-float (transfer! train-labels (tensor fact [60000] :float :x))
                 y-train (mnist/enc-categories train-labels-float)]
    (facts "MNIST classification tests."
           (time (train net train-images y-train :crossentropy 2 [])) => (roughly 0.02 0.1)
           (transfer! net net-infer)
           (take 8 (mnist/dec-categories (infer net-infer test-images)))
           => (list 7.0 2.0 1.0 0.0 4.0 1.0 4.0 9.0))))

(with-release [fact (dnnl-factory)]
  (test-mnist-convolutional fact))

;; "Elapsed time: 52966.299469 msecs"

(with-diamond cudnn-factory []
  (with-release [x-mb-tz (tensor [128 1 28 28] :float :nchw)
                 y-mb-tz (tensor [128 10] :float :nc)
                 net-bp (network x-mb-tz
                                 [(convo [32] [3 3] :relu)
                                  (convo [64] [3 3] :relu)
                                  (pooling [12 12] [2 2] :max)
                                  (dropout)
                                  (dense [128] :relu)
                                  (dropout)
                                  (dense [10] :softmax)])
                 net (init! (net-bp x-mb-tz :adam))
                 net-infer (net-bp x-mb-tz)
                 crossentropy-cost (cost net y-mb-tz :crossentropy)
                 train-images (transfer! train-images (tensor [60000 1 28 28] :float :nchw))
                 y-train (transfer! y-train (tensor y-train))
                 test-images (transfer! test-images (tensor [10000 1 28 28] :float :nchw))
                 y-test (transfer! y-test (tensor y-test))]
    (facts "cuDNN MNIST classification tests."
           (time (train net train-images y-train crossentropy-cost 2 [])) => (roughly 0.02 0.1)
           (transfer! net net-infer)
           (take 8 (mnist/dec-categories (native (infer net-infer test-images))))
           => (list 7.0 2.0 1.0 0.0 4.0 1.0 4.0 9.0))))

;; "Elapsed time: 3487.728516 msecs"
