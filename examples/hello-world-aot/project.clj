(defproject hello-world-aot "0.36.1"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.12.2"]
                 [uncomplicate/deep-diamond "0.36.1"]]

  :profiles {:default [:default/all ~(leiningen.core.utils/get-os)]
             :default/all {:dependencies []}
             :linux {:dependencies [[org.uncomplicate/neanderthal-mkl "0.57.1"]
                                    [org.bytedeco/mkl "2025.2-1.5.12" :classifier "linux-x86_64-redist"]
                                    ;; optional, if you want GPU computing with CUDA. Beware: the jar size is 3GB!
                                    [org.uncomplicate/deep-diamond-cuda "0.36.1"]
                                    [org.bytedeco/cuda "12.9-9.10-1.5.12-20250612.143830-1" :classifier "linux-x86_64-redist"]]}
             :windows {:dependencies [[org.uncomplicate/neanderthal-mkl "0.57.1"]
                                      [org.bytedeco/mkl "2025.2-1.5.12" :classifier "windows-x86_64-redist"]
                                      ;; optional, if you want GPU computing with CUDA. Beware: the jar size is 3GB!
                                      [org.uncomplicate/deep-diamond-cuda "0.36.1"]
                                      [org.bytedeco/cuda "12.9-9.10-1.5.12-20250612.145546-3" :classifier "windows-x86_64-redist"]]}
             :macosx {:dependencies [[org.uncomplicate/neanderthal-accelerate "0.57.0"]
                                     [org.bytedeco/openblas "0.3.30-1.5.12" :classifier "macosx-arm64"]]}}

  ;; Wee need this for the CUDA binaries, which are not available in the Maven Central due to its huge size (3GB, vs 1GB limit)!
  :repositories [["snapshots" "https://oss.sonatype.org/content/repositories/snapshots"]
                 ["maven-central-snapshots" "https://central.sonatype.com/repository/maven-snapshots"]]

  ;; We need direct linking for properly resolving types in heavy macros and avoiding reflection warnings!
  :jvm-opts ^:replace ["-Dclojure.compiler.direct-linking=true"
                       "--enable-native-access=ALL-UNNAMED"]

  ;; :global-vars {*warn-on-reflection* true
  ;;               *assert* false
  ;;               *unchecked-math* :warn-on-boxed
  ;;               *print-length* 16}
  )
