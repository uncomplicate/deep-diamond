(ns uncomplicate.diamond.functional.imdb-sentiment.classification-test
  (:require [midje.sweet :refer [facts throws => roughly]]
            [clojure.java.io :as io]
            [clojure.data.csv :as csv]
            [clojure.string :as string]
            [uncomplicate.commons [core :refer [with-release let-release]]]
            [uncomplicate.neanderthal
             [core :refer [transfer transfer! view-vctr native view-ge mrows ge
                           cols mv! rk! raw col row nrm2 scal! ncols dim rows]]
             [real :refer [entry! entry]]
             [native :refer [native-float fv fge]]
             [random :refer [rand-uniform!]]
             [math :as math :refer [sqrt]]]
            [uncomplicate.diamond
             [tensor :refer [*diamond-factory* tensor connector transformer
                             desc revert shape input output view-tz batcher]]
             [dnn :refer [fully-connected dropout network init! train! cost]]]
            [uncomplicate.diamond.internal.cost :refer [binary-accuracy!]]
            [uncomplicate.diamond.internal.dnnl.factory :refer [dnnl-factory]]
            [uncomplicate.diamond.internal.neanderthal.factory :refer [neanderthal-factory]]
            [uncomplicate.diamond.internal.cudnn.factory :refer [cudnn-factory]]))

(defn read-imdb-master
  ([]
   (->> (io/resource "imdb-sentiment/imdb_master.csv")
        (slurp)
        (csv/read-csv)
        (drop 1)))
  ([cnt]
   (take cnt (read-imdb-master))))

(defn word-frequencies [reviews]
  (apply merge-with + (pmap #(frequencies (% 1)) reviews)))

(defn word-vec [reviews cnt]
  (->> (word-frequencies reviews)
       (sort-by val >)
       (map #(% 0))
       (take cnt)
       (into [])))

(defn word-map [word-vector]
  (into {} (map #(vector (word-vector %) %) (range (count word-vector)))))

(defn split-review [review]
  (vector (review 1) (string/split (review 2) #" ") (review 3)))

(defonce wvec (word-vec (pmap split-review (read-imdb-master)) 10000))
(defonce wmap (word-map wvec))

(defn encode-review [word-map review x y]
  (let [[_ words sentiment] (split-review review)]
    (doseq [idx (map word-map words)]
      (when idx (entry! x idx 1.0)))
    (entry! y 0 (case sentiment "neg" 0 "pos" 1)))
  x)

(defn decode-review [word-vec code-vec]
  (filter identity
          (map #(if (< 0.5 (entry code-vec %))
                  (word-vec %)
                  nil)
               (range (dim code-vec)))))

(defn encode-reviews [wmap reviews]
  (let-release [in (fge 10000 25000)
                out (fge 1 25000)]
    (doall (map #(encode-review wmap %1 %2 %3) reviews (cols in) (cols out)))
    [in out]))

(defonce data (doall (map #(encode-reviews wmap (shuffle %))
                      (split-at 25000 (read-imdb-master 50000)))))

(def x-test ((first data) 0))
(def y-test ((first data) 1))

(def x-train ((second data) 0))
(def y-train ((second data) 1))

(defonce x-minibatch (ge x-train (mrows x-train) 512))

(defn test-imdb-classification [fact]
  (with-release [x-tz (tensor fact [25000 10000] :float :nc)
                 x-mb-tz (tensor fact [512 10000] :float :nc)
                 y-tz (tensor fact [25000 1] :float :nc)
                 y-mb-tz (tensor fact [512 1] :float :nc)
                 net-bp (network fact x-mb-tz
                                 [(fully-connected [16] :relu)
                                  (fully-connected [16] :relu)
                                  (fully-connected [1] :sigmoid)])
                 net (init! (net-bp x-mb-tz :adam))
                 net-infer (net-bp x-mb-tz)
                 crossentropy-cost (cost net y-mb-tz :crossentropy)
                 x-batcher (batcher x-tz (input net))
                 y-batcher (batcher y-tz y-mb-tz)]
    (transfer! x-train (view-vctr x-tz))
    (transfer! y-train (view-vctr y-tz))
    (facts "Adam gradient descent - IMDB sentiment classification."
           (time (train! net x-batcher y-batcher crossentropy-cost 5 [])) => (roughly 0.3 0.2)
           (transfer! net net-infer)
           (binary-accuracy! y-mb-tz (net-infer)) => (roughly 1 0.2))))

(with-release [fact (dnnl-factory)]
  (test-imdb-classification fact))

(with-release [fact (neanderthal-factory)]
  (test-imdb-classification fact))

(with-release [fact (cudnn-factory)]
  (test-imdb-classification fact))
