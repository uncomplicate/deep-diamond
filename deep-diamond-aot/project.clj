;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(defproject uncomplicate/deep-diamond "0.39.0-SNAPSHOT"
  :description "Fast Clojure Deep Learning Library"
  :author "Dragan Djuric"
  :url "http://github.com/uncomplicate/deep-diamond"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.12.2"]
                 [org.uncomplicate/neanderthal-base "0.57.0"]
                 [org.uncomplicate/neanderthal-mkl "0.57.1"]
                 [org.uncomplicate/neanderthal-cuda "0.58.0"]
                 [org.uncomplicate/neanderthal-accelerate "0.57.0"]
                 [org.uncomplicate/deep-diamond-base "0.39.0-SNAPSHOT"]
                 [org.uncomplicate/deep-diamond-dnnl "0.39.0-SNAPSHOT"]
                 [org.uncomplicate/deep-diamond-cuda "0.37.0"]
                 [org.uncomplicate/deep-diamond-bnns "0.36.1"]]

  :aot [uncomplicate.neanderthal.internal.cpp.structures
        uncomplicate.neanderthal.internal.cpp.factory
        uncomplicate.neanderthal.internal.cpp.mkl.factory
        uncomplicate.neanderthal.internal.cpp.openblas.factory
        uncomplicate.neanderthal.internal.cpp.cuda.structures
        uncomplicate.neanderthal.internal.cpp.cuda.factory
        uncomplicate.neanderthal.internal.cpp.accelerate.factory]

  :profiles {:dev [:dev/all ~(leiningen.core.utils/get-os)]
             :dev/all {:plugins [[lein-midje "3.2.1"]
                                 [lein-codox "0.10.8"]]
                       :resource-paths ["data"]
                       :global-vars {*warn-on-reflection* true
                                     *assert* false
                                     *unchecked-math* :warn-on-boxed
                                     *print-length* 128}
                       :dependencies [[midje "1.10.10"]
                                      [codox-theme-rdash "0.1.2"]
                                      [org.clojure/data.csv "1.1.0"]]
                       :codox {:metadata {:doc/format :markdown}
                               :source-uri "http://github.com/uncomplicate/deep-diamond/blob/master/{filepath}#L{line}"
                               :themes [:rdash]
                               :source-paths ["../deep-diamond-base/src/"
                                              "../deep-diamond-dnnl/src"]
                               :namespaces [uncomplicate.diamond.tensor
                                            uncomplicate.diamond.dnn
                                            uncomplicate.diamond.metrics
                                            uncomplicate.diamond.native
                                            uncomplicate.diamond.internal.protocols
                                            uncomplicate.diamond.internal.dnnl.core
                                            uncomplicate.diamond.internal.dnnl.constants]
                               :output-path "../docs/codox"}}
             :linux {:dependencies [[org.uncomplicate/neanderthal-mkl "0.57.1"]
                                    [org.bytedeco/mkl "2025.2-1.5.12" :classifier "linux-x86_64-redist"]
                                    [org.bytedeco/cuda "12.9-9.10-1.5.13-20250913.041224-9" :classifier "linux-x86_64-redist"]
                                    [org.bytedeco/cuda "12.9-9.10-1.5.13-20250913.041224-9" :classifier "linux-x86_64-redist-cublas"]
                                    [org.bytedeco/cuda "12.9-9.10-1.5.13-20250913.041224-9" :classifier "linux-x86_64-redist-cudnn"]]
                     :codox {:namespaces [uncomplicate.diamond.internal.cudnn.core
                                          uncomplicate.diamond.internal.cudnn.constants]
                             :source-paths ["../deep-diamond-cuda/src/clojure/"]}}
             :windows {:dependencies [[org.uncomplicate/neanderthal-mkl "0.57.1"]
                                      [org.bytedeco/mkl "2025.2-1.5.12" :classifier "windows-x86_64-redist"]
                                      [org.bytedeco/cuda "12.9-9.10-1.5.13-20250913.041224-9" :classifier "windows-x86_64-redist"]
                                      [org.bytedeco/cuda "12.9-9.10-1.5.13-20250913.041224-9" :classifier "windows-x86_64-redist-cublas"]
                                      [org.bytedeco/cuda "12.9-9.10-1.5.13-20250913.041224-9" :classifier "windows-x86_64-redist-cudnn"]]
                       :codox {:namespaces [uncomplicate.diamond.internal.cudnn.core
                                            uncomplicate.diamond.internal.cudnn.constants]
                               :source-paths ["../deep-diamond-cuda/src/clojure/"]}}
             :macosx {:dependencies [[org.uncomplicate/neanderthal-accelerate "0.57.0"]
                                     [org.bytedeco/openblas "0.3.30-1.5.12" :classifier "macosx-arm64"]]
                      :codox {:namespaces [uncomplicate.diamond.internal.bnns.core
                                           uncomplicate.diamond.internal.constants.constants]
                              :source-paths ["../deep-diamond-bnns/src"]}}}

  :repositories [["maven-central-snapshots" "https://central.sonatype.com/repository/maven-snapshots"]]

  :jvm-opts ^:replace ["-Dclojure.compiler.direct-linking=true"
                       "--enable-native-access=ALL-UNNAMED"]

  :javac-options ["-target" "1.8" "-source" "1.8" "-Xlint:-options"])
