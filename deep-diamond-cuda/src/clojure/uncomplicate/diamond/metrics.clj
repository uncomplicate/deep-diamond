;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns  ^{:author "Dragan Djuric"}
    uncomplicate.diamond.metrics
  "Classification evaluation functions."
  (:require [uncomplicate.commons
             [core :refer [with-release let-release]]
             [utils :refer [dragan-says-ex cond-into]]]
            [uncomplicate.neanderthal
             [core :refer [dim ge mrows amax raw col axpy axpy! copy
                           vctr mv mv! copy! dia trans view-vctr entry entry!]]
             [real :as r :refer [asum]]
             [vect-math :refer [linear-frac]]]))

(defn confusion-matrix
  "Returns the confusion matrix between vectors, matrices, or tensors with
  `real` and `pred` (predicted) categories. For example, a typical confusion matrix
  with two categories, shows the number of true positives (TP), true negatives (TN),
  false positives (FP), and false negatives (FN). This function generalizes this
  to `n` categories, where there are `n-1` different ways to be wrong, that is,
  when the predicted category is not the same as the real category, we can treat
  these mistakes per each category, instead in bulk.
  The categories are natural numbers in theory, and this function can process
  both positive integers and floats. If the data in `real` or `prod` floats, their
  whole value part will be consider to be the category index (for example, both
  `4.0` and `4.889` fall into category `4`).

  Arguments:

  - `real`: values as they are supposed to be.
  - `pred`: values predicted by an algorithm.
  - `n`: number of categories in `real` and `pred`.
  "
  ([real pred ^long n]
   (let [real (view-vctr real)
         pred (view-vctr pred)]
     (let-release [cm (ge real n n)]
       (dotimes [k (dim real)]
         (let [i (entry pred k)
               j (entry real k)]
           (entry! cm i j (inc (double (entry cm i j))))))
       cm)))
  ([real pred]
   (confusion-matrix real pred (inc (double (amax real))))))

(defn contingency-totals
  "Calculates true positives, false positives, and false negatives from the confusion matrix `cm`.
  Returns a `3 x n` matrix, where columns are categories, and rows are TP, FP, and FN.
  Since multi-category confusion matrix is per-category, there is no point in making distinction
  between true positive and true negative; both are 'true this', whatever 'this' may be (this category
  is 'positive', and all other categories are 'negative').
  "
  [cm]
  (let-release [totals (ge cm (mrows cm) 3)]
    (with-release [ones (entry! (vctr cm (mrows cm)) 1.0)]
      (let [tp (col totals 0)
            fp (col totals 1)
            fn (col totals 2)]
        (copy! (dia cm) tp)
        (axpy! -1.0 tp (mv! cm ones fp))
        (axpy! -1.0 tp (mv! (trans cm) ones fn))))
    totals))

(defn classification-metrics
  "Computes most common classification metrics from confusion matrix `cm`
  (which can be provided, or computed internally from `real` and `prod`).
  The result is a Clojure map, with keys `:metrics` and `:macro`.
  `:metrics` are returned as a map of neanderthal vectors: `:accuracy`,
  `:f1`, `sensitivity`, `:specificity`, etc. The `:macro` key contains
  rates from `cm` (true positive rate, true negative rate, etc.).

  Arguments:

  - `cm`: pre-computed confusion matrix.
  - `real`: values as they are supposed to be.
  - `pred`: values predicted by an algorithm.
  - `n`: number of categories in `real` and `pred`.

  Please see relevant [Wikipedia article](https://en.wikipedia.org/wiki/Evaluation_of_binary_classifiers)
  for the quick reference.
  "
  ([cm]
   (let [eps 1e-9]
     (with-release [tp (copy (dia cm))
                    ones (entry! (raw tp) 1.0)
                    tp+fn (mv (trans cm) ones)
                    fn (axpy -1.0 tp tp+fn)
                    tp+fp (mv cm ones)
                    fp (axpy -1.0 tp tp+fp)
                    tn (linear-frac -1.0 tp (asum tp))
                    tn+fp (axpy tn fp)
                    tn+fn (axpy tn fn)
                    tpr (linear-frac 1.0 tp eps 1.0 tp+fn eps)
                    fpr (linear-frac 1.0 fp eps 1.0 tn+fp eps)
                    ppv (linear-frac 1.0 tp eps 1.0 tp+fp eps)
                    for (linear-frac 1.0 fn eps 1.0 tn+fn eps)]
       (let [cnt (dim ones)
             tpr-mac (/ (asum tpr) cnt)
             fpr-mac (/ (asum fpr) cnt)
             ppv-mac (/ (asum ppv) cnt)
             for-mac (/ (asum for) cnt)
             tnr-mac (- 1.0 fpr-mac)
             npv-mac (- 1.0 for-mac)
             fnr-mac (- 1.0 tpr-mac)
             fdr-mac (- 1.0 ppv-mac)]
         {:metrics {:accuracy (/ 1.0 (+ 1.0 (/ (asum fp) (asum tp))))
                    :f1 (/ (* 2 ppv-mac tpr-mac) (+ ppv-mac tpr-mac))
                    :ba (* 0.5 (+ tpr-mac tnr-mac))
                    :sensitivity tpr-mac
                    :specificity tnr-mac
                    :precision ppv-mac
                    :fall-out fpr-mac}
          :macro {:tpr tpr-mac
                  :tnr tnr-mac
                  :ppv ppv-mac
                  :npv npv-mac
                  :fnr fnr-mac
                  :fpr fpr-mac
                  :fdr fdr-mac
                  :for for-mac}}))))
  ([real pred n]
   (with-release [cm (confusion-matrix real pred n)]
     (classification-metrics cm)))
  ([real pred]
   (with-release [cm (confusion-matrix real pred)]
     (classification-metrics cm))))
