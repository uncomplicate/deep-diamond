# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Deep Diamond is a Clojure library for fast tensors and neural network computations. It provides a unified API across multiple backend implementations (CPU and GPU) backed by highly optimized native libraries.

## Build System & Commands

Deep Diamond uses **Leiningen** as its build tool. Each subproject has its own `project.clj` with OS-specific profiles.

### Running Tests

```bash
# Run all tests in a subproject
cd deep-diamond-base && lein test

# Run specific test namespace
cd deep-diamond-dnnl && lein test uncomplicate.diamond.internal.dnnl.directed-test

# Run tests with continuous refresh (requires lein-midje plugin)
cd examples/hello-world-aot && make tr
# or
lein do clean, test-refresh :changes-only

# Run single test (using Midje's :only selector)
lein test :only namespace/test-name
```

### Building & Installation

```bash
# Install a subproject to local Maven repo
cd deep-diamond-base && lein install

# Generate API documentation (from deep-diamond-aot only)
cd deep-diamond-aot && lein codox
# Output: docs/codox/
```

### REPL Development

```bash
# Start REPL in any subproject
cd deep-diamond-dnnl && lein repl

# The REPL will automatically select the appropriate backend based on platform
# JVM options include --enable-native-access=ALL-UNNAMED for native library access
```

## Project Structure

Deep Diamond is organized into **7 subprojects**:

### Core Modules
- **deep-diamond-base**: Platform-agnostic abstractions, protocols, and public APIs
- **deep-diamond-test**: Shared test specifications that run against all backends

### Backend Implementations
- **deep-diamond-dnnl**: CPU backend using Intel oneDNN (formerly DNNL)
- **deep-diamond-cuda**: NVIDIA GPU backend using cuDNN
- **deep-diamond-bnns**: Apple Silicon backend using Accelerate/BNNS framework

### Distribution
- **deep-diamond-aot**: Convenience package bundling all backends (ahead-of-time compiled)
- **examples/**: Hello-world examples demonstrating CPU and GPU usage

### Dependency Hierarchy

```
deep-diamond-base (foundation)
    ├── deep-diamond-dnnl (depends on: base + org.bytedeco/dnnl)
    ├── deep-diamond-cuda (depends on: base + dnnl + CUDA/cuDNN)
    └── deep-diamond-bnns (depends on: base + Accelerate)

deep-diamond-aot (depends on: base + all backends)
deep-diamond-test (depends on: base)
```

**Important**: The CUDA backend depends on DNNL for certain operations (like fully-connected layers on CPU), demonstrating a hybrid CPU/GPU approach.

## Architecture

### Protocol-Oriented Design

The entire system is built around **18 core protocols** defined in `deep-diamond-base/src/uncomplicate/diamond/internal/protocols.clj`. Each backend provides factory implementations of these protocols.

**Key Protocol Categories:**

1. **Factory Protocols** (backend selection):
   - `DiamondFactoryProvider` - Provides backend factory instances
   - `TensorFactory` - Creates tensors, transformers, batchers, shufflers
   - `DnnFactory` - Creates neural network layer blueprints
   - `RnnFactory` - Creates recurrent network blueprints
   - `CostFactory` - Creates cost/loss functions

2. **Layer & Network Protocols**:
   - `Backprop` - Forward/backward pass implementation
   - `Parameters` / `DiffParameters` - Trainable weights and gradients
   - `Initializable` - Weight initialization
   - `DescriptorProvider` - Tensor descriptors for different contexts

### Backend Selection Mechanism

Backend loading is **automatic** and happens in `uncomplicate.diamond.native`:

1. On namespace load, the system auto-detects available backends:
   - Checks for DNNL classes → loads DNNL backend
   - On macOS, checks for BNNS classes → loads BNNS backend
   - Falls back to nil if nothing available

2. The selected factory is bound to `*diamond-factory*` dynamic var

3. Users can explicitly choose backends:
   ```clojure
   (require '[uncomplicate.diamond.native])
   ;; Auto-loads default backend, or manually:
   (load-backend :dnnl)
   (load-backend :bnns)
   ```

### Main Public APIs

User-facing namespaces in `deep-diamond-base/src/uncomplicate/diamond/`:

- **tensor.clj** - Core tensor operations: `tensor`, `desc`, `shape`, `transformer`, `batcher`
- **dnn.clj** - Neural network DSL: layer constructors (`activation`, `fully-connected`, `convolution`, `pooling`, `batch-norm`, `dropout`), training (`cost`, `network`, `init!`, `train`, `infer`)
- **metrics.clj** - Evaluation: `confusion-matrix`, `classification-metrics`
- **native.clj** - Backend loading: `load-backend`, `map-tensor`
- **neanderthal.clj** - Integration bridge with Neanderthal linear algebra library

### Internal Organization

- `internal/protocols.clj` - Protocol definitions
- `internal/network.clj` - Sequential/parallel network implementations
- `internal/cost.clj` - Cost function implementations
- `internal/neanderthal/` - Neanderthal integration layer
- Backend-specific: `internal/dnnl/`, `internal/cudnn/`, `internal/bnns/` - Native library bindings

## Testing Strategy

**Framework**: Midje (behavior-driven testing)

**Test Organization**:
- `deep-diamond-test/` contains **generic test specifications** that are reusable across all backends
- Each backend runs the same test suite to ensure consistent behavior
- Backend-specific tests validate native library integration details

**Test Structure**:
```
deep-diamond-test/src/
  └── tensor_test.clj, dnn_test.clj  # Generic tests

deep-diamond-{dnnl,cuda,bnns}/test/uncomplicate/diamond/internal/{backend}/
  ├── {backend}_tensor_test.clj
  ├── directed_test.clj      # Layer tests
  ├── rnn_test.clj           # Recurrent network tests
  └── core_test.clj          # Core functionality
```

## Development Configuration

### JVM Options (in all project.clj files)

```clojure
:jvm-opts ["-Dclojure.compiler.direct-linking=true"
           "--enable-native-access=ALL-UNNAMED"]

:global-vars {*warn-on-reflection* true
              *assert* false
              *unchecked-math* :warn-on-boxed
              *print-length* 128}
```

**Why these settings matter**:
- **Direct linking** - Required for performance optimization
- **Native access** - Required for native library JNI calls
- **Reflection warnings** - Catch performance issues at compile time
- **Unchecked math** - Warn about boxed math operations

### Platform-Specific Dependencies

Each subproject uses OS-specific Leiningen profiles (`:linux`, `:windows`, `:macosx`) to pull in the correct native library classifiers:

- Linux: MKL, CUDA (if using CUDA backend)
- Windows: MKL, CUDA (if using CUDA backend)
- macOS: Accelerate framework, OpenBLAS (ARM64)

### Repository Configuration

All projects require the Maven Central Snapshots repository for bleeding-edge bytedeco dependencies:

```clojure
:repositories [["maven-central-snapshots"
                "https://central.sonatype.com/repository/maven-snapshots"]]
```

## Key Architectural Patterns

1. **Protocol-Oriented Design**: Polymorphism across backends via protocols, not inheritance
2. **Factory Pattern**: Each backend provides a factory implementing the complete protocol suite
3. **Blueprint Pattern**: Neural network layers are "blueprints" that can be instantiated with specific factories
4. **Hybrid Execution**: Backends can delegate operations to other backends (e.g., CUDA → DNNL for CPU fallback)
5. **Memory Management**: Explicit resource management via Neanderthal's `Releaseable` protocol
6. **Neanderthal Integration**: Deep integration with Neanderthal library for linear algebra operations

## Important Notes for Development

### When Working with Backends

- **DNNL**: Full implementation (all layer types, RNN support)
- **cuDNN**: Full GPU implementation + CPU fallback via DNNL for fully-connected layers
- **BNNS**: Partial implementation (activation layers, basic operations; convolutions/RNN support incomplete)

### When Adding New Layers

1. Add blueprint constructor in `deep-diamond-base/src/uncomplicate/diamond/dnn.clj`
2. Implement layer protocols in each backend's `directed.clj`
3. Add factory method in each backend's `factory.clj`
4. Add tests to `deep-diamond-test` for cross-backend validation
5. Add backend-specific tests if needed

### When Modifying Protocols

Protocol changes in `deep-diamond-base/src/uncomplicate/diamond/internal/protocols.clj` require updates to **all three backends** (dnnl, cuda, bnns) to maintain consistency.

### Memory and Resource Management

Deep Diamond relies on Neanderthal's resource management. Always use `with-release` for tensors, networks, and other resources that hold native memory. Failing to release resources will cause memory leaks.

### Integration with Neanderthal

Deep Diamond is tightly integrated with [Neanderthal](https://github.com/uncomplicate/neanderthal) for linear algebra. Tensors can be converted to/from Neanderthal matrices, and many operations delegate to Neanderthal for CPU computation.

## Common Development Workflows

### Workflow 1: Adding a New Activation Function

Activation functions are simpler than full layers and demonstrate the basic cross-backend pattern.

**Steps:**

1. **Add the blueprint constructor** in `deep-diamond-base/src/uncomplicate/diamond/dnn.clj`:
   ```clojure
   (defn swish
     "Swish activation: x * sigmoid(x)"
     [fact src-desc]
     (->ActivationBlueprint fact src-desc :swish))
   ```

2. **Update each backend's activation implementation** in `directed.clj`:
   - `deep-diamond-dnnl/src/uncomplicate/diamond/internal/dnnl/directed.clj`
   - `deep-diamond-cuda/src/clojure/uncomplicate/diamond/internal/cudnn/directed.clj`
   - `deep-diamond-bnns/src/uncomplicate/diamond/internal/bnns/directed.clj`

   Add the `:swish` case to the activation multimethod/conditional.

3. **Add constant mappings** in each backend's `constants.clj`:
   - Map `:swish` to the native library's constant (e.g., `CUDNN_ACTIVATION_SWISH`)

4. **Install base and test**:
   ```bash
   cd deep-diamond-base && lein install
   cd ../deep-diamond-dnnl && lein test uncomplicate.diamond.internal.dnnl.directed-test
   ```

5. **Test across all available backends**:
   ```bash
   cd ../deep-diamond-cuda && lein test  # If CUDA available
   cd ../deep-diamond-bnns && lein test  # On macOS
   ```

6. **Add to generic test suite** in `deep-diamond-test/src/uncomplicate/diamond/dnn_test.clj` if needed.

### Workflow 2: Fixing a Backend-Specific Bug

Example: Fix incorrect gradient computation in DNNL convolution layer.

**Steps:**

1. **Reproduce the bug** with a minimal test:
   ```bash
   cd deep-diamond-dnnl
   lein repl
   ```
   ```clojure
   (require '[uncomplicate.diamond.dnn :refer :all])
   (require '[uncomplicate.diamond.tensor :refer :all])
   ;; Create minimal reproduction case
   ```

2. **Locate the implementation**:
   - Convolution layers: `deep-diamond-dnnl/src/uncomplicate/diamond/internal/dnnl/directed.clj`
   - Look for `ConvolutionInference` or `ConvolutionTraining` record

3. **Check the native bindings** in `deep-diamond-dnnl/src/uncomplicate/diamond/internal/dnnl/core.clj`:
   - Verify parameters passed to native `convolution-backward-*` functions

4. **Make the fix** in `directed.clj`

5. **Test the specific layer**:
   ```bash
   lein test uncomplicate.diamond.internal.dnnl.directed-test
   ```

6. **Run full backend test suite**:
   ```bash
   lein test
   ```

7. **Verify against generic tests**:
   ```bash
   # Generic tests import the backend, so they'll pick up your changes
   lein test uncomplicate.diamond.dnn-test
   ```

8. **Install and test downstream** (if needed):
   ```bash
   lein install
   cd ../deep-diamond-cuda  # CUDA uses DNNL for some operations
   lein test
   ```

### Workflow 3: Adding a New Tensor Operation

Example: Add a `clip` operation to limit tensor values to a range.

**Steps:**

1. **Add protocol method** to `deep-diamond-base/src/uncomplicate/diamond/internal/protocols.clj`:
   ```clojure
   (defprotocol TensorMath
     (clip! [this min-val max-val] "Clip tensor values to [min-val, max-val]"))
   ```

2. **Add public API** to `deep-diamond-base/src/uncomplicate/diamond/tensor.clj`:
   ```clojure
   (defn clip!
     "Clips tensor values to the specified range."
     [x min-val max-val]
     (tensor-math/clip! x min-val max-val))
   ```

3. **Implement in each backend's tensor module**:
   - `deep-diamond-dnnl/src/uncomplicate/diamond/internal/dnnl/tensor.clj`
   - `deep-diamond-cuda/src/clojure/uncomplicate/diamond/internal/cudnn/tensor.clj`
   - `deep-diamond-bnns/src/uncomplicate/diamond/internal/bnns/tensor.clj`

4. **Check if native library supports the operation**:
   - If yes: use native implementation via bindings in `core.clj`
   - If no: implement using existing operations or fall back to Neanderthal

5. **Install base and rebuild backends**:
   ```bash
   cd deep-diamond-base && lein install
   cd ../deep-diamond-dnnl && lein install
   cd ../deep-diamond-cuda && lein install
   cd ../deep-diamond-bnns && lein install
   ```

6. **Add tests** to `deep-diamond-test/src/uncomplicate/diamond/tensor_test.clj`

7. **Run tests across all backends**:
   ```bash
   cd deep-diamond-dnnl && lein test
   cd ../deep-diamond-cuda && lein test
   cd ../deep-diamond-bnns && lein test
   ```

### Workflow 4: Testing Changes Across Multiple Subprojects

When you modify `deep-diamond-base`, you need to test all downstream projects.

**Steps:**

1. **Make changes** in `deep-diamond-base/src/`

2. **Install base** to local Maven repo:
   ```bash
   cd deep-diamond-base
   lein install
   ```

3. **Test each backend** in sequence:
   ```bash
   cd ../deep-diamond-dnnl
   lein test

   cd ../deep-diamond-cuda
   lein test  # Only if CUDA available

   cd ../deep-diamond-bnns
   lein test  # Only on macOS
   ```

4. **Test the AOT distribution**:
   ```bash
   cd ../deep-diamond-aot
   lein test
   ```

5. **Test examples** to ensure end-user experience works:
   ```bash
   cd ../examples/hello-world-aot
   lein test
   ```

6. **Run with continuous feedback** during development:
   ```bash
   cd examples/hello-world-aot
   make tr  # Watches for changes and reruns tests
   ```

### Workflow 5: Debugging Native Library Issues

Example: Segfault or incorrect results from native library calls.

**Steps:**

1. **Enable verbose native library logging** (if available):
   - DNNL: Set environment variable `ONEDNN_VERBOSE=1`
   - CUDA: Set `CUDNN_LOGINFO_DBG=1` or `CUDNN_LOGDEST_DBG=stdout`

2. **Run with JVM assertions enabled**:
   ```bash
   cd deep-diamond-dnnl
   lein repl
   ```
   Edit `project.clj` temporarily to set `:global-vars {*assert* true}`

3. **Check parameter marshalling** in `core.clj`:
   - Verify pointer types match native function signatures
   - Check for off-by-one errors in array indexing
   - Validate memory layout expectations (row-major vs column-major)

4. **Inspect tensor descriptors**:
   ```clojure
   (require '[uncomplicate.diamond.tensor :refer :all])
   (def t (tensor [2 3 4]))
   (desc t)  ; Check dimensions, strides, data type
   ```

5. **Test with minimal native library example**:
   - Create a minimal test using only the `core.clj` native bindings
   - Compare with equivalent C/C++ code using the same library
   - Check the native library's examples directory

6. **Common issues**:
   - **Memory not released**: Check for missing `with-release` calls
   - **Descriptor mismatch**: Input/output tensor descriptors incompatible
   - **Uninitialized tensors**: Forgot to call `init!` before training
   - **Wrong data layout**: NCHW vs NHWC format mismatch

### Workflow 6: Adding Tests to the Shared Test Suite

When adding features that should work across all backends, add tests to `deep-diamond-test`.

**Steps:**

1. **Add test to** `deep-diamond-test/src/uncomplicate/diamond/dnn_test.clj` or `tensor_test.clj`:
   ```clojure
   (facts "New feature test"
     (with-release [factory (dnnl-factory)]  ; Backend-agnostic
       (with-release [x (tensor factory [2 3])]
         ;; Test code here
         )))
   ```

2. **The test should be backend-agnostic**:
   - Use factory parameter passed from backend tests
   - Don't assume specific backend behavior unless necessary
   - Test the public API, not internal implementation

3. **Each backend test imports and runs these tests**:
   - `deep-diamond-dnnl/test/.../directed_test.clj` imports from `deep-diamond-test`
   - CUDA and BNNS backends do the same

4. **Install test suite**:
   ```bash
   cd deep-diamond-test
   lein install
   ```

5. **Run in each backend**:
   ```bash
   cd ../deep-diamond-dnnl && lein test
   cd ../deep-diamond-cuda && lein test
   cd ../deep-diamond-bnns && lein test
   ```

### Workflow 7: Working with the REPL for Exploratory Development

**Steps:**

1. **Start REPL** in the backend you're working with:
   ```bash
   cd deep-diamond-dnnl
   lein repl
   ```

2. **Load necessary namespaces**:
   ```clojure
   (require '[uncomplicate.diamond.dnn :refer :all])
   (require '[uncomplicate.diamond.tensor :refer :all])
   (require '[uncomplicate.commons.core :refer [with-release]])
   (require '[uncomplicate.neanderthal.core :as n])
   ```

3. **Create test tensors and networks**:
   ```clojure
   (with-release [x (tensor [2 3 4] :float :nchw)]
     (n/entry! (n/view-ge x) 0.5)  ; Fill with 0.5
     (shape x))
   ```

4. **Experiment with layers**:
   ```clojure
   (with-release [factory (native-factory)
                  input-desc (desc [128 3 224 224] :float :nchw)
                  net (network factory input-desc
                        [(convolutional [64] [3 3])
                         (activation :relu)
                         (pooling [2 2] :max)])]
     (init! net)
     ;; Test forward pass, inspect shapes, etc.
     )
   ```

5. **Reload after making changes**:
   ```clojure
   (require '[uncomplicate.diamond.dnn :refer :all] :reload-all)
   ```

6. **Check for reflection warnings** in REPL output:
   - Shows where type hints are needed for performance

### Workflow 8: Updating Dependencies

When updating native library versions (e.g., newer oneDNN, cuDNN, or bytedeco libraries):

**Steps:**

1. **Update version** in `project.clj`:
   ```clojure
   :dependencies [[org.bytedeco/dnnl-platform "3.10.0-1.5.14"]]
   ```

2. **Check for API changes**:
   - Review the native library's changelog
   - Check if constants changed in `constants.clj`
   - Verify function signatures in `core.clj`

3. **Update constants** if needed in `{backend}/constants.clj`:
   - Map any new enum values
   - Remove deprecated constants

4. **Test incrementally**:
   ```bash
   lein clean  # Important! Clear old compiled classes
   lein test uncomplicate.diamond.internal.dnnl.core-test  # Basic native bindings
   lein test uncomplicate.diamond.internal.dnnl.directed-test  # Layer tests
   lein test  # Full suite
   ```

5. **Update all OS-specific profiles**:
   - `:linux`, `:windows`, `:macosx` may need different versions
   - Test on each platform if possible

6. **Check examples**:
   ```bash
   cd examples/hello-world-aot
   lein clean
   lein test
   ```
