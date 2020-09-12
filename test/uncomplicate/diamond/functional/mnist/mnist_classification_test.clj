(ns uncomplicate.diamond.functional.mnist.mnist-classification-test
  (:require [midje.sweet :refer [facts throws => roughly]]
            [uncomplicate.commons
             [core :refer [let-release Releaseable release with-release]]
             [utils :refer [random-access channel]]]
            [uncomplicate.neanderthal
             [core :refer [subvector view-ge transfer transfer! dim entry ge mrows
                           transfer! view-vctr amax imax col native]]
             [real :refer [entry!]]
             [native :as neand :refer [native-byte native-float fge]]]
            [uncomplicate.diamond
             [tensor :refer [tensor transformer connector transformer
                             desc revert shape input output view-tz batcher with-diamond]]
             [native :refer [map-tensor]]
             [dnn :refer [fully-connected network init! train cost infer]]
             [metrics :refer [confusion-matrix contingency-totals classification-metrics]]]
            [uncomplicate.diamond.internal.dnnl.factory :refer [dnnl-factory]]
            [uncomplicate.diamond.internal.cudnn.factory :refer [cudnn-factory]]))

(defonce train-images-file (random-access "data/mnist/train-images.idx3-ubyte"))
(defonce train-labels-file (random-access "data/mnist/train-labels.idx1-ubyte"))
(defonce test-images-file (random-access "data/mnist/t10k-images.idx3-ubyte"))
(defonce test-labels-file (random-access "data/mnist/t10k-labels.idx1-ubyte"))

(defonce train-images (map-tensor train-images-file [60000 1 28 28] :uint8 :nchw :read 16))
(defonce train-labels (map-tensor train-labels-file [60000] :uint8 :x :read 8))
(defonce test-images (map-tensor test-images-file [10000 1 28 28] :uint8 :nchw :read 16))
(defonce test-labels (map-tensor test-labels-file [10000] :uint8 :x :read 8))

(defn enc-categories [val-tz]
  (let [val-vector (view-vctr val-tz)]
    (let-release [cat-tz (tensor val-tz [(first (shape val-tz)) (inc (long (amax val-vector)))] :float :nc )
                  cat-matrix (view-ge (view-vctr cat-tz) (second (shape cat-tz)) (first (shape cat-tz)))]
      (dotimes [j (dim val-vector)]
        (entry! cat-matrix (entry val-vector j) j 1.0))
      cat-tz)))

(defn dec-categories [cat-tz]
  (let [cat-matrix (view-ge (view-vctr cat-tz) (second (shape cat-tz)) (first (shape cat-tz)))]
    (let-release [val-tz (tensor cat-tz [(first (shape cat-tz))] :float :x)
                  val-vector (view-vctr val-tz)]
      (dotimes [j (dim val-vector)]
        (entry! val-vector j (imax (col cat-matrix j))))
      val-tz)))

(defonce train-labels-float (transfer! train-labels (tensor [60000] :float :x)))
(defonce y-train (enc-categories train-labels-float))
(defonce test-labels-float (transfer! test-labels (tensor [10000] :float :x)))
(defonce y-test (enc-categories test-labels-float))

(with-release [x-mb-tz (tensor [128 1 28 28] :float :nchw)
               y-mb-tz (tensor [128 10] :float :nc)
               net-bp (network x-mb-tz
                               [(fully-connected [512] :relu)
                                (fully-connected [10] :softmax)])
               net (init! (net-bp x-mb-tz :adam))
               net-infer (net-bp x-mb-tz)
               crossentropy-cost (cost net y-mb-tz :crossentropy)
               x-train-bat (batcher train-images (input net))
               y-train (transfer! y-train (tensor y-train))
               y-train-bat (batcher y-train y-mb-tz)
               x-test-bat (batcher test-images (input net-infer))
               y-test (transfer! y-test (tensor y-test))
               y-test-bat (batcher (output net-infer) y-test)]
  (facts "DNNL MNIST classification tests."
         (time (train net x-train-bat y-train-bat crossentropy-cost 2 [])) => (roughly 0.02 0.1)
         (transfer! net net-infer)
         (with-release [inf (infer net-infer test-images)
                        pred (dec-categories inf)
                        metrics (:metrics (classification-metrics test-labels-float pred))]
           (:accuracy metrics) => (roughly 0.965 0.005)
           (:f1 metrics) => (roughly 0.965 0.006)
           (take 8 pred) => (list 7.0 2.0 1.0 0.0 4.0 1.0 4.0 9.0))))

;; "Elapsed time: 2074.615346 msecs"

(with-diamond cudnn-factory []
  (with-release [x-mb-tz (tensor [128 1 28 28] :float :nchw)
                 y-mb-tz (tensor [128 10] :float :nc)
                 net-bp (network x-mb-tz
                                 [(fully-connected [512] :relu)
                                  (fully-connected [10] :softmax)])
                 net (init! (net-bp x-mb-tz :adam))
                 net-infer (net-bp x-mb-tz)
                 crossentropy-cost (cost net y-mb-tz :crossentropy)
                 train-images (transfer! train-images (tensor [60000 1 28 28] :float :nchw))
                 x-train-bat (batcher train-images (input net))
                 y-train (transfer! y-train (tensor y-train))
                 y-train-bat (batcher y-train y-mb-tz)
                 test-images (transfer! test-images (tensor [10000 1 28 28] :float :nchw))
                 x-test-bat (batcher test-images (input net-infer))
                 y-test (transfer! y-test (tensor y-test))
                 y-test-bat (batcher (output net-infer) y-test)]
    (facts "cuDNN MNIST classification tests."
           (time (train net x-train-bat y-train-bat crossentropy-cost 2 [])) => (roughly 0.02 0.1)
           (transfer! net net-infer)
           (with-release [inf (infer net-infer test-images)
                          native-inf (native inf)
                          pred (dec-categories native-inf)
                          metrics (:metrics (classification-metrics test-labels-float pred))]
             (:accuracy metrics) => (roughly 0.965 0.005)
             (:f1 metrics) => (roughly 0.965 0.005)
             (take 8 pred) => (list 7.0 2.0 1.0 0.0 4.0 1.0 4.0 9.0)))))

;; "Elapsed time: 213.328266 msecs"

(defn test-mnist-classification-internal-input [fact]
  (with-release [net-bp (network fact (desc [512 1 28 28] :float :nchw)
                                 [(fully-connected [256] :relu)
                                  (fully-connected [256] :relu)
                                  (fully-connected [10] :sigmoid)])
                 net (init! (net-bp :adam))
                 net-infer (net-bp)
                 crossentropy-cost (cost net :crossentropy)
                 train-images (transfer! train-images (tensor fact [60000 1 28 28] :uint8 :nchw))
                 y-train (enc-categories train-labels-float)]
    (facts "MNIST classification tests."
           (time (train net train-images y-train crossentropy-cost 2 [])) => (roughly 0.25 0.1)
           (transfer! net net-infer)
           (transfer! (view-tz test-images 512) (input net-infer))
           (take 8 (dec-categories (net-infer))) => (list 7.0 2.0 1.0 0.0 4.0 1.0 4.0 9.0))))

(defn test-mnist-classification-internal-cost [fact]
  (with-release [net-bp (network fact (desc [512 1 28 28] :float :nchw)
                                 [(fully-connected [256] :relu)
                                  (fully-connected [256] :relu)
                                  (fully-connected [10] :sigmoid)])
                 net (init! (net-bp :adam))
                 net-infer (net-bp)
                 train-images (transfer! train-images (tensor fact [60000 1 28 28] :uint8 :nchw))
                 y-train (enc-categories train-labels-float)]
    (facts "MNIST classification tests."
           (time (train net train-images y-train :crossentropy 2 [])) => (roughly 0.25 0.1)
           (transfer! net net-infer)
           (transfer! (view-tz test-images 512) (input net-infer))
           (take 8 (dec-categories (net-infer))) => (list 7.0 2.0 1.0 0.0 4.0 1.0 4.0 9.0))))

(defn test-mnist-classification-internal-infer [fact]
  (with-release [net-bp (network fact (desc [512 1 28 28] :float :nchw)
                                 [(fully-connected [256] :relu)
                                  (fully-connected [256] :relu)
                                  (fully-connected [10] :sigmoid)])
                 net (init! (net-bp :adam))
                 net-infer (net-bp)
                 train-images (transfer! train-images (tensor fact [60000 1 28 28] :uint8 :nchw))
                 y-train (enc-categories train-labels-float)]
    (facts "MNIST classification tests."
           (time (train net train-images y-train :crossentropy 2 [])) => (roughly 0.25 0.1)
           (transfer! net net-infer)
           (take 8 (dec-categories (infer net-infer test-images)))
           => (list 7.0 2.0 1.0 0.0 4.0 1.0 4.0 9.0))))

(defn test-mnist-classification-softmax [fact]
  (with-release [net-bp (network fact (desc [512 1 28 28] :float :nchw)
                                 [(fully-connected [256] :relu)
                                  (fully-connected [256] :relu)
                                  (fully-connected [10] :softmax)])
                 net (init! (net-bp :adam))
                 net-infer (net-bp)
                 train-images (transfer! train-images (tensor fact [60000 1 28 28] :uint8 :nchw))
                 y-train (enc-categories train-labels-float)]
    (facts "MNIST classification tests."
           (time (train net train-images y-train :crossentropy 2 [])) => (roughly 0.25 0.15)
           (transfer! net net-infer)
           (take 8 (dec-categories (infer net-infer test-images)))
           => (list 7.0 2.0 1.0 0.0 4.0 1.0 4.0 9.0))))

(with-release [fact (dnnl-factory)]
  (test-mnist-classification-internal-input fact)
  (test-mnist-classification-internal-cost fact)
  (test-mnist-classification-internal-infer fact)
  (test-mnist-classification-softmax fact))

;; "Elapsed time: 762.952065 msecs"
;; "Elapsed time: 810.657392 msecs"
;; "Elapsed time: 821.330076 msecs"
;; "Elapsed time: 798.699342 msecs"
