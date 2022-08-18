(ns uncomplicate.diamond.functional.mnist.mnist-convolutional-test
  (:require [midje.sweet :refer [facts throws => roughly]]
            [uncomplicate.commons
             [core :refer [with-release]]
             [utils :refer [random-access channel]]]
            [uncomplicate.neanderthal
             [core :refer [transfer! native]]]
            [uncomplicate.diamond
             [tensor :refer [tensor desc with-diamond]]
             [native :refer [map-tensor]]
             [dnn :refer [dense convo pooling dropout network init! train infer batch-norm]]
             [metrics :refer [classification-metrics]]]
            [uncomplicate.diamond.internal.dnnl.factory :refer [dnnl-factory]]
            [uncomplicate.diamond.internal.cudnn.factory :refer [cudnn-factory]]
            [uncomplicate.diamond.functional.mnist.mnist-classification-test :as mnist]))

(def train-images-file (random-access "data/mnist/train-images.idx3-ubyte"))
(def train-labels-file (random-access "data/mnist/train-labels.idx1-ubyte"))
(def test-images-file (random-access "data/mnist/t10k-images.idx3-ubyte"))
(def test-labels-file (random-access "data/mnist/t10k-labels.idx1-ubyte"))

(def train-images (map-tensor train-images-file [60000 1 28 28] :uint8 :nchw :read 16))
(def train-labels (map-tensor train-labels-file [60000] :uint8 :x :read 8))
(def test-images (map-tensor test-images-file [10000 1 28 28] :uint8 :nchw :read 16))
(def test-labels (map-tensor test-labels-file [10000] :uint8 :x :read 8))

(def train-labels-float (transfer! train-labels (tensor [60000] :float :x)))
(def y-train (mnist/enc-categories train-labels-float))
(def test-labels-float (transfer! test-labels (tensor [10000] :float :x)))
(def y-test (mnist/enc-categories test-labels-float))

(with-diamond dnnl-factory []
  (with-release [net-bp (network (desc [128 1 28 28] :float :nchw)
                                 [(convo [32] [3 3] :relu)
                                  (convo [64] [3 3] :relu)
                                  (pooling [2 2] :max)
                                  (dropout)
                                  (dense [128] :relu)
                                  ;;(batch-norm)
                                  (dropout)
                                  (dense [10] :softmax)])
                 net (init! (net-bp :adam))
                 net-infer (net-bp)]
    (facts "MNIST classification tests."
           (time (train net train-images y-train :crossentropy 2 [])) => (roughly 0.0 0.03)
           (transfer! net net-infer)
           (with-release [inf (infer net-infer test-images)
                          pred (mnist/dec-categories inf)
                          metrics (:metrics (classification-metrics test-labels-float pred))]
             (:accuracy metrics) => (roughly 0.985 0.005)
             (:f1 metrics) => (roughly 0.985 0.005)
             (take 8 pred) => (list 7.0 2.0 1.0 0.0 4.0 1.0 4.0 9.0)))))

;; "Elapsed time: 52966.299469 msecs"

(with-diamond cudnn-factory []
  (with-release [x-mb-tz (tensor [128 1 28 28] :float :nchw)
                 y-mb-tz (tensor [128 10] :float :nc)
                 net-bp (network x-mb-tz
                                 [(convo [32] [3 3] :relu)
                                  (convo [64] [3 3] :relu)
                                  (pooling [2 2] :max)
                                  (dropout)
                                  (dense [128] :relu)
                                  (dropout)
                                  (dense [10] :softmax)])
                 net (init! (net-bp x-mb-tz :adam))
                 net-infer (net-bp x-mb-tz)
                 train-images (transfer! train-images (tensor [60000 1 28 28] :float :nchw))
                 y-train (transfer! y-train (tensor [60000 10] :float :nchw))
                 test-images (transfer! test-images (tensor [10000 1 28 28] :float :nchw))
                 y-test (transfer! y-test (tensor [10000 10] :float :nchw))]
    (facts "cuDNN MNIST classification tests."
           (time (train net train-images y-train :crossentropy 2 [])) => (roughly 0.0 0.03)
           (transfer! net net-infer)
           (with-release [inf (infer net-infer test-images)
                          native-inf (native inf)
                          pred (mnist/dec-categories native-inf)
                          metrics (:metrics (classification-metrics test-labels-float pred))]
             (:accuracy metrics) => (roughly 0.965 0.02)
             (:f1 metrics) => (roughly 0.965 0.02)
             (take 8 pred) => (list 7.0 2.0 1.0 0.0 4.0 1.0 4.0 9.0)))))

;; "Elapsed time: 3487.728516 msecs"
