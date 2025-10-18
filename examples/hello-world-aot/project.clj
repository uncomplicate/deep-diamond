(defproject hello-world-aot "0.39.1"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.12.3"]
                 [uncomplicate/deep-diamond "0.39.1"]]

  ;; uncomplicate/deep-diamond is AOT compiled for fast loading and developer convenience, which
  ;; might cause issues since it freezes org.clojure/core.async to the specific version (see ClojureCUDA).

  ;; FOR PRODUCTION USE, PLEASE USE org.uncomplicate/deep-diamond-base AND OTHER PARTICULAR DEPENDENCIES

  :profiles {:linux {:dependencies [[org.uncomplicate/neanderthal-mkl "0.57.1"]
                                    [org.bytedeco/mkl "2025.2-1.5.12" :classifier "linux-x86_64-redist"]
                                    ;; optional, if you want GPU computing with CUDA. Beware: the cuda redist jars are very large!
                                    [org.bytedeco/cuda-redist "12.9-9.10-1.5.12" :classifier "linux-x86_64"]
                                    [org.bytedeco/cuda-redist-cublas "12.9-9.10-1.5.12" :classifier "linux-x86_64"]
                                    [org.bytedeco/cuda-redist-cudnn "12.9-9.10-1.5.12" :classifier "linux-x86_64"]]}
             :windows {:dependencies [[org.uncomplicate/neanderthal-mkl "0.57.1"]
                                      [org.bytedeco/mkl "2025.2-1.5.12" :classifier "windows-x86_64-redist"]
                                      ;; optional, if you want GPU computing with CUDA. Beware: the cuda redist jars are very large!
                                      [org.bytedeco/cuda-redist "12.9-9.10-1.5.12" :classifier "windows-x86_64"]
                                      [org.bytedeco/cuda-redist-cublas "12.9-9.10-1.5.12" :classifier "windows-x86_64"]
                                      [org.bytedeco/cuda-redist-cudnn "12.9-9.10-1.5.12" :classifier "windows-x86_64"]]}
             :macosx {:dependencies [[org.uncomplicate/neanderthal-accelerate "0.57.0"]
                                     [org.bytedeco/openblas "0.3.30-1.5.12" :classifier "macosx-arm64"]]}}

  ;; Wee need this for the DNNL binaries, for the latest version is not available in the Maven Central yet
  :repositories [["maven-central-snapshots" "https://central.sonatype.com/repository/maven-snapshots"]]

  ;; We need direct linking for properly resolving types in heavy macros and avoiding reflection warnings!
  :jvm-opts ^:replace ["-Dclojure.compiler.direct-linking=true"
                       "--enable-native-access=ALL-UNNAMED"]

  ;; :global-vars {*warn-on-reflection* true
  ;;               *assert* false
  ;;               *unchecked-math* :warn-on-boxed
  ;;               *print-length* 16}
  )
