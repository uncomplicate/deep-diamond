;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(defproject uncomplicate/deep-diamond "0.27.0-SNAPSHOT"
  :description "Fast Clojure Deep Learning Library"
  :author "Dragan Djuric"
  :url "http://github.com/uncomplicate/deep-diamond"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.11.1"]
                 [uncomplicate/neanderthal "0.48.0-SNAPSHOT"]
                 [org.bytedeco/dnnl-platform "3.2.1-1.5.10-SNAPSHOT"]]

  :profiles {:dev {:plugins [[lein-midje "3.2.1"]
                             [lein-codox "0.10.7"]]
                   :resource-paths ["data"]
                   :global-vars {*warn-on-reflection* true
                                 *assert* false
                                 *unchecked-math* :warn-on-boxed
                                 *print-length* 128}
                   :dependencies [[midje "1.10.9"]
                                  [org.clojure/data.csv "1.0.1"]
                                  [org.bytedeco/cuda-platform-redist "12.1-8.9-1.5.10-SNAPSHOT"]]

                   :jvm-opts ^:replace ["-Dclojure.compiler.direct-linking=true"]}}

  :repositories [["snapshots" "https://oss.sonatype.org/content/repositories/snapshots"]]

  :codox {:metadata {:doc/format :markdown}
          :src-dir-uri "http://github.com/uncomplicate/deep-diamond/blob/master/"
          :src-linenum-anchor-prefix "L"
          :output-path "docs/codox"}

  :source-paths ["src/clojure" "src/device"])
