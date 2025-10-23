# Llaminar V2 Test Infrastructure

This directory contains the test suite for Llaminar V2 architecture.

## Test File Naming Conventions

**IMPORTANT**: V2 tests follow a specific naming convention:

- **Format**: `Test__ClassName.cpp` where `ClassName` is the **class under test**
- **Test Suite**: Must match the filename: `TEST(Test__ClassName, TestName)`
- **Examples**:
  - Testing `ModelLoader` class:
    - File: `Test__ModelLoader.cpp`
    - Suite: `TEST(Test__ModelLoader, LoadsGGUFHeader) { ... }`
  - Testing `Qwen2Pipeline` class:
    - File: `Test__Qwen2Pipeline.cpp`
    - Suite: `TEST(Test__Qwen2Pipeline, InitializesLayers) { ... }`
  - Testing `FP32Tensor` class:
    - File: `Test__FP32Tensor.cpp`
    - Suite: `TEST(Test__FP32Tensor, FP32Creation) { ... }`

**Why This Convention?**
- **Clear association**: Test file name directly indicates which class is being tested
- **Easy navigation**: Search for `Test__ClassName` to find tests for any class
- **Consistent naming**: Test suite matches file name (both use `Test__ClassName`)
- **Matches V2 style**: CamelCase aligns with V2's naming conventions
- **Distinguishes from V1**: V1 tests use snake_case (e.g., `test_batch_correctness.cpp`)

## Test Organization

Tests are organized into four categories:

### 1. **Unit Tests** (`unit/`)
- **Purpose**: Fast, isolated tests for individual components
- **Scope**: Tensors, kernels, utilities
- **Requirements**: No model loading, no MPI (or single-rank MPI only)
- **Runtime**: < 1 second per test
- **Example**: `Test__FP32Tensor.cpp`

### 2. **Integration Tests** (`integration/`)
- **Purpose**: Medium-complexity tests involving multiple components
- **Scope**: Model loading, pipeline initialization, data flow validation
- **Requirements**: May load models, single or multi-rank MPI
- **Runtime**: < 10 seconds per test
- **Example**: TBD

### 3. **End-to-End Tests** (`e2e/`)
- **Purpose**: Full pipeline tests with real models and inference
- **Scope**: Complete workflows (load → prefill → decode → validate)
- **Requirements**: Requires GGUF models, multi-rank MPI
- **Runtime**: 10-60 seconds per test
- **Example**: TBD

### 4. **PyTorch Parity Tests** (`pytorch_parity/`)
- **Purpose**: Ground truth validation against PyTorch reference implementation
- **Scope**: Layer-by-layer outputs, logits, attention scores
- **Requirements**: PyTorch, NumPy, GGUF models
- **Runtime**: 30-120 seconds per test
- **Example**: TBD

### 5. **Performance Tests** (`performance/`)
- **Purpose**: Benchmarks for throughput, latency, memory usage
- **Scope**: GEMM kernels, attention, full pipeline
- **Requirements**: Release build, isolated environment
- **Runtime**: Variable (1-300 seconds)
- **Note**: Not part of standard CTest suites (run manually or in CI performance stage)

---

## Running Tests

### Prerequisites

All V2 tests require GGUF models to be present. The test infrastructure automatically handles this via a **fixture test** that runs before any test suite:

```bash
# The fixture test runs automatically, but you can trigger it manually:
cd build_v2
ctest -R V2_FetchModelsFixture
```

Models are fetched from HuggingFace and cached in `models/` directory. The fetch is skipped if models are already present.

### Run All V2 Tests

```bash
cd build_v2
ctest -L V2                     # All V2 tests
ctest -L V2 --output-on-failure # Show test output on failure
ctest -L V2 -V                  # Verbose output
```

### Run Specific Test Categories

```bash
# Unit tests only
ctest -L "V2;Unit"

# Integration tests only
ctest -L "V2;Integration"

# End-to-end tests only
ctest -L "V2;E2E"

# PyTorch parity tests only
ctest -L "V2;Parity"
```

### Run Specific Tests by Name

```bash
# Run all tensor-related tests
ctest -R "V2_Unit_Tensor.*"

# Run a specific test
ctest -R "V2_Unit_TensorBasics"

# Run with verbose output
ctest -R "V2_Unit_TensorBasics" -V
```

---

## Test Fixtures and Dependencies

The V2 test infrastructure uses CMake's **FIXTURES** system to manage dependencies:

### Model Fetching Fixture

- **Fixture Name**: `V2_Models`
- **Setup Test**: `V2_FetchModelsFixture`
- **Purpose**: Ensures GGUF models are downloaded before tests run
- **Timeout**: 600 seconds (10 minutes for large downloads)
- **Behavior**: Always passes (script returns 0 even on failure)
- **MPI**: Does NOT use mpirun (runs directly with bash)
- **Environment Variables**:
  - `LLAMINAR_SKIP_MODEL_DOWNLOAD=1` - Skip all downloads (use existing)
  - `LLAMINAR_FETCH_EXPERIMENTAL=1` - Include experimental/404-prone models
  - `LLAMINAR_FETCH_FP32=1` - Include large FP32 models (~2GB)
  - `LLAMINAR_FETCH_7B_IQ_MODELS=1` - Include large 7B IQ models (~1-3GB each)
  - `LLAMINAR_ENFORCE_MODELS=1` - Fail if no models present after fetch

### MPI and OpenMP Configuration

**All V2 tests (except the fixture) run with optimal MPI/OpenMP settings:**

- **MPI Configuration**:
  - Processes: **Minimum 2 ranks** (ensures multi-rank testing even on single-socket systems)
  - Default: 1 per socket on multi-socket systems (e.g., 2 on dual-socket)
  - Single-socket systems: Uses `--oversubscribe` flag to run 2 ranks on 1 socket
  - Binding: `--bind-to socket --map-by socket`
  - Optimizations: Memory pinning, NUMA-aware communication
  
- **OpenMP Configuration** (auto-detected from CPU topology):
  - Threads: Physical cores per socket (e.g., 28 on dual-socket with 28 cores/socket)
  - Placement: `OMP_PLACES=sockets` (threads placed on sockets)
  - Binding: `OMP_PROC_BIND=close` (threads bound close together)
  - Nested: Disabled (`OMP_NESTED=false`)
  - Dynamic: Disabled (`OMP_DYNAMIC=false`)
  
- **BLAS Threading**:
  - `OPENBLAS_NUM_THREADS` = threads per socket
  - `MKL_NUM_THREADS` = threads per socket
  - `GOTO_NUM_THREADS` = threads per socket

- **Intel OpenMP Tuning**:
  - `KMP_AFFINITY=granularity=fine,compact,1,0`
  - `KMP_BLOCKTIME=0` (reduce thread blocking time)

**Example for 2-socket system with 28 cores/socket:**
```
mpirun -np 2 --bind-to socket --map-by socket \
  --mca mpi_leave_pinned 1 \
  --mca btl_vader_single_copy_mechanism none \
  v2_test_tensor_basics

Environment:
  OMP_NUM_THREADS=28
  OPENBLAS_NUM_THREADS=28
  MKL_NUM_THREADS=28
```

**Example for 1-socket system with 8 cores:**
```
mpirun -np 2 --bind-to socket --map-by socket \
  --oversubscribe \
  --mca mpi_leave_pinned 1 \
  --mca btl_vader_single_copy_mechanism none \
  v2_test_tensor_basics

Environment:
  OMP_NUM_THREADS=8
  OPENBLAS_NUM_THREADS=8
  MKL_NUM_THREADS=8
```

This configuration ensures:
- ✅ Optimal CPU utilization (all physical cores)
- ✅ NUMA-aware memory allocation
- ✅ Minimal inter-socket communication overhead
- ✅ Consistent performance across test runs
- ✅ **Multi-rank testing on all systems** (minimum 2 ranks, even single-socket)

### How Fixtures Work

1. **Fixture Setup**: `V2_FetchModelsFixture` provides the `V2_Models` fixture
2. **Fixture Requirement**: All V2 tests require the `V2_Models` fixture
3. **Execution Order**: CMake ensures `V2_FetchModelsFixture` runs before any test needing models
4. **One-Time Execution**: Fixture setup runs only once per CTest invocation
5. **Graceful Degradation**: If model fetch fails, tests still run (may skip if model unavailable)

---

## Adding New Tests

### 1. Create Test Source File

Place your test in the appropriate subdirectory:

```bash
# Unit test (use CamelCase: Test__ClassName.cpp)
tests/v2/unit/Test__MyFeature.cpp

# Integration test
tests/v2/integration/Test__MyIntegration.cpp

# E2E test
tests/v2/e2e/Test__MyWorkflow.cpp

# PyTorch parity test
tests/v2/pytorch_parity/Test__MyParity.cpp
```

### 2. Write Test Using GTest

```cpp
#include <gtest/gtest.h>
#include "tensors/Tensors.h"
// ... other includes

using namespace llaminar2;

TEST(MyTestSuite, MyTest) {
    // Your test code
    EXPECT_EQ(actual, expected);
}
```

### 3. Add Test to CMakeLists.txt

Edit `tests/v2/CMakeLists.txt`:

```cmake
# Add executable
add_executable(v2_test_my_feature unit/test_my_feature.cpp)
target_link_libraries(v2_test_my_feature 
    llaminar2_core 
    GTest::gtest
    GTest::gtest_main
)

# Register test with fixture dependency and MPI wrapper
add_v2_test(V2_Unit_MyFeature 
    COMMAND v2_test_my_feature
    LABELS "V2;Unit;MyFeature"
)

# Optional: Override MPI process count (default = number of sockets)
add_v2_test(V2_Unit_MyFeatureSingleRank
    COMMAND v2_test_my_feature
    LABELS "V2;Unit;MyFeature"
    MPI_PROCS 1  # Force single-rank execution
)
```

**Note**: The `add_v2_test()` function automatically:
- Wraps the test command with `mpirun` and optimal settings
- Detects CPU topology (sockets, cores per socket)
- **Ensures at least 2 MPI ranks** (adds `--oversubscribe` on single-socket systems)
- Sets all OpenMP/MPI environment variables
- Makes the test depend on the `V2_Models` fixture

### 4. Build and Run

```bash
cd build_v2
cmake --build . --target v2_test_my_feature --parallel
ctest -R V2_Unit_MyFeature -V
```

---

## Test Labels

Tests are labeled for easy filtering:

| Label | Meaning |
|-------|---------|
| `V2` | All V2 tests |
| `Unit` | Unit tests |
| `Integration` | Integration tests |
| `E2E` | End-to-end tests |
| `Parity` | PyTorch parity tests |
| `Tensor` | Tensor-related tests |
| `Kernel` | Kernel-related tests |
| `Pipeline` | Pipeline-related tests |
| `Models` | Model loading tests |
| `Fixture` | Fixture setup tests |

**Combine labels** to run specific subsets:

```bash
ctest -L "V2;Unit;Tensor"    # V2 unit tests for tensors
ctest -L "V2;Integration"    # All V2 integration tests
```

---

## Current Test Status

### ✅ Implemented

- **Fixture System**: Model fetching before all tests
- **Unit Tests**:
  - `V2_Unit_TensorBasics` - Tensor creation and properties (5 tests)

### ⏳ TODO

- **Unit Tests**:
  - TensorFactory with NUMA binding
  - Quantized tensor types (IQ4_NL, Q8_0, Q4_0, etc.)
  - Kernel interfaces (GEMM, RoPE, SwiGLU, etc.)
  - ModelLoader basic functionality
- **Integration Tests**:
  - Model loading from GGUF
  - Pipeline initialization
  - Weight verification
- **E2E Tests**:
  - Qwen2Pipeline full inference
  - Multi-rank MPI execution
- **PyTorch Parity Tests**:
  - Prefill phase parity
  - Decode phase parity
  - Layer-by-layer comparison

---

## Debugging Tests

### Run Single Test with Verbose Output

```bash
cd build_v2
ctest -R V2_Unit_TensorBasics -V
```

### Run Test Executable Directly

```bash
cd build_v2
./tests/v2/v2_test_tensor_basics         # All tests
./tests/v2/v2_test_tensor_basics --help  # GTest options
./tests/v2/v2_test_tensor_basics --gtest_filter="TensorBasics.FP32*"  # Specific test
```

### Enable Debug Logging

```bash
# Set environment variables before running
export LLAMINAR_LOG_LEVEL=DEBUG
ctest -R V2_Unit_TensorBasics -V
```

### Debugging with GDB

```bash
cd build_v2
gdb ./tests/v2/v2_test_tensor_basics
(gdb) run
(gdb) bt  # Backtrace on crash
```

---

## Best Practices

### Unit Tests

- ✅ **DO**: Test single components in isolation
- ✅ **DO**: Use small, synthetic data
- ✅ **DO**: Keep tests < 1 second runtime
- ❌ **DON'T**: Load models or use MPI (unless single-rank)
- ❌ **DON'T**: Test multiple components together (use integration tests)

### Integration Tests

- ✅ **DO**: Test component interactions
- ✅ **DO**: Use small GGUF models (0.5B)
- ✅ **DO**: Verify data flow between components
- ❌ **DON'T**: Test full inference (use E2E tests)

### E2E Tests

- ✅ **DO**: Test complete workflows
- ✅ **DO**: Use realistic prompts and sequences
- ✅ **DO**: Verify output quality
- ❌ **DON'T**: Test individual components (use unit tests)

### PyTorch Parity Tests

- ✅ **DO**: Use NumPy snapshots for ground truth
- ✅ **DO**: Compare layer-by-layer outputs
- ✅ **DO**: Use appropriate tolerances (FP16: 1e-3, FP32: 1e-5)
- ❌ **DON'T**: Compare only final outputs (not enough granularity)

---

## File Structure

```
tests/v2/
├── CMakeLists.txt                  # Test configuration
├── README.md                       # This file
├── unit/                           # Unit tests
│   ├── test_tensor_basics.cpp      ✅ Implemented
│   ├── test_tensor_factory.cpp     ⏳ TODO
│   └── test_quantized_tensors.cpp  ⏳ TODO
├── integration/                    # Integration tests
│   ├── test_model_loader.cpp       ⏳ TODO
│   └── test_pipeline_init.cpp      ⏳ TODO
├── e2e/                            # End-to-end tests
│   └── test_qwen_inference.cpp     ⏳ TODO
├── pytorch_parity/                 # PyTorch parity tests
│   ├── test_pytorch_parity.cpp     ⏳ TODO
│   └── npz_to_npy.py              # Utility script
└── performance/                    # Performance tests
    └── benchmark_gemm.cpp          ⏳ TODO
```

---

## Related Documentation

- **V2 Architecture**: `.github/instructions/llaminar-v2-architecture.instructions.md`
- **Development Guide**: `.github/copilot-instructions.md`
- **Model Fetching Script**: `scripts/fetch_test_models.sh`
- **V1 Parity Framework**: `.github/instructions/parity-test-framework.instructions.md` (reference)

---

## Contact

For questions or issues with the V2 test infrastructure, refer to the main project documentation or open an issue.
