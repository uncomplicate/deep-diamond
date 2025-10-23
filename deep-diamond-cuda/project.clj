;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(defproject org.uncomplicate/deep-diamond-cuda "0.40.0"
  :description "Fast Clojure Deep Learning Library"
  :author "Dragan Djuric"
  :url "http://github.com/uncomplicate/deep-diamond"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.12.3"]
                 [org.uncomplicate/neanderthal-cuda  "0.59.0"]
                 [org.uncomplicate/deep-diamond-base "0.39.0"]
                 [org.uncomplicate/deep-diamond-dnnl "0.39.2"]]

  :profiles {:dev [:dev/all ~(leiningen.core.utils/get-os)]
             :dev/all {:plugins [[lein-midje "3.2.1"]]
                       :resource-paths ["data"]
                       :global-vars {*warn-on-reflection* true
                                     *assert* false
                                     *unchecked-math* :warn-on-boxed
                                     *print-length* 128}
                       :dependencies [[midje "1.10.10"]
                                      [org.clojure/data.csv "1.1.0"]
                                      [org.uncomplicate/deep-diamond-test "0.39.0"]]
                       :jvm-opts ^:replace ["-Dclojure.compiler.direct-linking=true"
                                            "--enable-native-access=ALL-UNNAMED"]}
             :linux {:dependencies [[org.uncomplicate/neanderthal-mkl "0.57.1"]
                                    [org.bytedeco/mkl "2025.2-1.5.12" :classifier "linux-x86_64-redist"]
                                    [org.bytedeco/cuda-redist "13.0-9.14-1.5.13-20251022.164318-20" :classifier "linux-x86_64"]
                                    [org.bytedeco/cuda-redist-cublas "13.0-9.14-1.5.13-20251022.164348-20" :classifier "linux-x86_64"]
                                    [org.bytedeco/cuda-redist-cudnn "13.0-9.14-1.5.13-20251022.164345-20" :classifier "linux-x86_64"]]}
             :windows {:dependencies [[org.uncomplicate/neanderthal-mkl "0.57.1"]
                                      [org.bytedeco/mkl "2025.2-1.5.12" :classifier "windows-x86_64-redist"]
                                      [org.bytedeco/cuda-redist "13.0-9.14-1.5.13-20251022.164318-20" :classifier "windows-x86_64"]
                                      [org.bytedeco/cuda-redist-cublas "13.0-9.14-1.5.13-20251022.164318-20" :classifier "windows-x86_64"]
                                      [org.bytedeco/cuda-redist-cudnn "13.0-9.14-1.5.13-20251022.164345-20" :classifier "windows-x86_64"]]}
             :macosx {:dependencies [[org.uncomplicate/neanderthal-accelerate "0.57.0"]
                                     [org.bytedeco/openblas "0.3.30-1.5.12" :classifier "macosx-arm64"]]}}

  :repositories [["maven-central-snapshots" "https://central.sonatype.com/repository/maven-snapshots"]]

  :javac-options ["-target" "1.8" "-source" "1.8" "-Xlint:-options"]
  :source-paths ["src/clojure" "src/device"])
