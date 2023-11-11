;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.diamond.metrics
  (:require [uncomplicate.commons
             [core :refer [with-release let-release]]
             [utils :refer [dragan-says-ex cond-into]]]
            [uncomplicate.neanderthal
             [core :refer [dim ge mrows amax raw col axpy axpy! copy
                           vctr mv mv! copy! dia trans view-vctr entry entry!]]
             [real :as r :refer [asum]]
             [vect-math :refer [linear-frac]]]))

(defn confusion-matrix
  ([real pred n]
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

(defn contingency-totals [cm]
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
