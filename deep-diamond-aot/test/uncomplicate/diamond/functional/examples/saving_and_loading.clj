(ns uncomplicate.diamond.functional.examples.saving-and-loading
  (:require [midje.sweet :refer [facts throws => =not=>]]
            [clojure.java.io :as io]
            [clojure.data.csv :as csv]
            [clojure.string :as string]
            [uncomplicate.commons
             [core :refer [with-release let-release]]
             [utils :refer [random-access channel]]]
            [uncomplicate.neanderthal
             [core :refer [entry! entry zero asum transfer transfer! view-vctr native view-ge
                           cols mv! rk! raw col row nrm2 scal! ncols dim rows]]
             [native :refer [native-float fv]]
             [random :refer [rand-uniform!]]
             [math :as math :refer [sqrt]]]
            [uncomplicate.diamond
             [tensor :refer [*diamond-factory* tensor connector transformer
                             desc revert shape input output view-tz batcher]]
             [dnn :refer [sum activation inner-product fully-connected network init! train! cost rnn abbreviate]]
             [native :refer [map-tensor]]]))

(with-release [td1 (desc [2 3] :float :nc)
               tf1 (random-access "td1.tz")
               tz0 (tensor *diamond-factory* [2 3] :float :nc)
               tz1 (map-tensor tf1 td1)
               tz2 (map-tensor tf1 td1)
               tz3 (entry! (zero tz0) 3)]

  (facts "Saving and loading a tensor to/from a file."
         (entry! tz1 0) => tz0
         (entry! tz1 1) =not=> tz0
         tz2 => tz1
         (entry! tz2 2) => tz1
         tz1 =not=> tz0
         (transfer! tz3 tz1) => tz1
         (asum tz2) => 18.0))

(with-release [network-blueprint (network (desc [512 1 28 28] :float :nchw)
                                          [(fully-connected [512] :relu)
                                           (fully-connected [10] :sigmoid)])
               net1 (init! (network-blueprint :adam))
               net2 (network-blueprint :adam)
               nfc1 (channel (random-access "network.trained"))]
  (facts "Saving and loading and network by transferring it to/from a Java NIO file channel."
         net1 =not=> net2
         (transfer! net1 nfc1) => nfc1
         (transfer! nfc1 net2)
         net1 => net2))

(with-release [network-blueprint (network (desc [4 2 1] :float :tnc)
                                          [(rnn [4] :gru)
                                           (rnn 2)
                                           (abbreviate)
                                           (fully-connected [2] :relu)
                                           (fully-connected [1] :linear)])
               net1 (init! (network-blueprint :adam))
               net2 (init! (network-blueprint :adam) :zero)
               nfc1 (channel (random-access "network2.trained"))]
  (facts "Saving and loading an RNN network by transferring it to/from a Java NIO file channel."
         net1 =not=> net2
         (transfer! (range 2 20) (input net1))
         (net1)
         (transfer! net1 nfc1) => nfc1
         (transfer! nfc1 net2) => net1
         (transfer! (input net1) (input net2))
         (net2) => (output net1)
         net2 => net1))
