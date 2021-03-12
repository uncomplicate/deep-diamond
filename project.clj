;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(defproject uncomplicate/deep-diamond "0.20.0"
  :description "Fast Clojure Deep Learning Library"
  :author "Dragan Djuric"
  :url "http://github.com/uncomplicate/deep-diamond"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.10.3"]
                 [uncomplicate/neanderthal "0.39.0"]
                 [org.bytedeco/dnnl-platform "2.1.1-1.5.5"]
                 [org.jcuda/jcudnn "11.1.1"]]

  :profiles {:dev {:plugins [[lein-midje "3.2.1"]
                             [lein-codox "0.10.6"]]
                   :resource-paths ["data"]
                   :global-vars {*warn-on-reflection* true
                                 *assert* false
                                 *unchecked-math* :warn-on-boxed
                                 *print-length* 128}
                   :dependencies [[midje "1.9.10"]
                                  [org.clojure/data.csv "1.0.0"]]}}

  :repositories [["snapshots" {:url "https://oss.sonatype.org/content/repositories/snapshots/"
                               :snapshots true :sign-releases false :checksum :warn :update :daily}]]

  :codox {:metadata {:doc/format :markdown}
          :src-dir-uri "http://github.com/uncomplicate/deep-diamond/blob/master/"
          :src-linenum-anchor-prefix "L"
          :output-path "docs/codox"}

  :jvm-opts ^:replace ["-Dclojure.compiler.direct-linking=true" "-XX:+UseLargePages"
                       #_"--add-opens=java.base/jdk.internal.ref=ALL-UNNAMED"]

  :javac-options ["-target" "1.8" "-source" "1.8" "-Xlint:-options"]
  :source-paths ["src/clojure" "src/device"])
