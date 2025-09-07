;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(defproject org.uncomplicate/deep-diamond-bnns "0.36.0-SNAPSHOT"
  :description "Fast Clojure Deep Learning Library"
  :author "Dragan Djuric"
  :url "http://github.com/uncomplicate/deep-diamond"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.12.2"]
                 [org.uncomplicate/neanderthal-accelerate "0.57.0-SNAPSHOT"]
                 [org.uncomplicate/deep-diamond-base "0.36.0-SNAPSHOT"]
                 [org.uncomplicate/accelerate-platform "0.1.0-1.5.12"]]

  :profiles {:dev {:plugins [[lein-midje "3.2.1"]]
                   :global-vars {*warn-on-reflection* true
                                 *assert* false
                                 *unchecked-math* :warn-on-boxed
                                 *print-length* 128}
                   :dependencies [[midje "1.10.10"]
                                  [org.uncomplicate/deep-diamond-test "0.36.0-SNAPSHOT"]
                                  [org.bytedeco/openblas "0.3.30-1.5.12" :classifier "macosx-arm64"]]
                   :jvm-opts ^:replace ["-Dclojure.compiler.direct-linking=true"
                                        "--enable-native-access=ALL-UNNAMED"]}}

  :javac-options ["-target" "1.8" "-source" "1.8" "-Xlint:-options"])
