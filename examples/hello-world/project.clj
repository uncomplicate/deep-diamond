(defproject hello-world-on-the-fly "0.39.1"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.12.3"]
                 [org.uncomplicate/neanderthal-base "0.57.0"]
                 [org.uncomplicate/deep-diamond-base "0.39.0"]
                 [org.uncomplicate/deep-diamond-dnnl "0.39.0"]
                 [org.uncomplicate/deep-diamond-bnns "0.39.0"]]

  ;; If you'd like AOT compiled Deep Diamond for fast namespace loading (1-2 seconds instead of 20),
  ;; see hello-world-aot example for reference.

  ;; Practically, all platform specific dependencies are optional.
  ;; You can use either MKL, OpenBLAS, or Accelerate for CPU computing as you wish (when your hardware supports them)
  ;; For the GPU, choose between CUDA (PC) or OpenCL (PC or MacOS x86_64)
  :profiles {:linux {:dependencies [[org.uncomplicate/neanderthal-mkl "0.57.1"]
                                    [org.bytedeco/mkl "2025.2-1.5.12" :classifier "linux-x86_64-redist"]
                                    [org.uncomplicate/deep-diamond-cuda "0.39.1"]
                                    [org.bytedeco/cuda-redist "12.9-9.10-1.5.12" :classifier "linux-x86_64"]
                                    [org.bytedeco/cuda-redist-cublas "12.9-9.10-1.5.12" :classifier "linux-x86_64"]
                                    [org.bytedeco/cuda-redist-cudnn "12.9-9.10-1.5.12" :classifier "linux-x86_64"]]}
             :windows {:dependencies [[org.uncomplicate/neanderthal-mkl "0.57.1"]
                                      [org.bytedeco/mkl "2025.2-1.5.12" :classifier "windows-x86_64-redist"]
                                      [org.uncomplicate/deep-diamond-cuda "0.39.1"]
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
