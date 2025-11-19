# Llaminar Project Development Guidelines

This document provides comprehensive guidelines for working with the Llaminar LLM inference engine, including build processes, debugging techniques, and kernel development best practices learned from our MPI and performance optimization work.

**⚠️ Architecture Note**: Llaminar has two parallel architectures:
- **V1 (Production)**: `src/` - Operator-based MPI distributed inference (fully functional)
- **V2 (Development)**: `src/v2/` - Operator-free kernel-centric design (in development)

Most sections in this document apply to **V1**. For V2-specific guidance, see `.github/instructions/llaminar-v2-architecture.instructions.md`.

## Table of Contents
- [Architecture Overview (V1 vs V2)](#architecture-overview-v1-vs-v2)
- [Project Overview (V1)](#project-overview-v1)
- [Build System](#build-system)
- [Testing Guidelines (V1)](#testing-guidelines-v1)
- [CTest Label Best Practices](#ctest-label-best-practices)
- [Debugging with GDB](#debugging-with-gdb)
- [Kernel Development (V1)](#kernel-development-v1)
- [Kernel Development (V2)](#kernel-development-v2)
- [MPI Development Best Practices](#mpi-development-best-practices)
- [Performance Optimization](#performance-optimization)
- [COSMA vs OpenBLAS Integration (V1)](#cosma-vs-openblas-integration-v1)
- [Code Quality Guidelines](#code-quality-guidelines)
- [Documentation Standards](#documentation-standards)

## Architecture Overview 

### V2 Architecture (Development - `src/v2/`)

**Design Philosophy**: Operator-free Tensor-centric design

**Key Characteristics**:
- 🔄 **In Development**: Basic pipeline structure, not feature-complete
- 🎯 **No Operator Layer**: Pipelines call kernels directly
- 🎯 **Per-Tensor Device Affinity**: Each tensor knows its device placement
- 🎯 **Heterogeneous Execution**: Mix CPU, CUDA, ROCm in single run
- 🎯 **IActivationTensor / ITensorKernel Pattern**: Kernel ops are focused around the IActivationTensors which will accept computation results (GEMM, SwiGLU, RoPE, etc). Activation tensors instantiate kernels via ITensorKernel and take the weights or other activation tensors as arguments, and the kernels mutate the IActivationTensor buffer in-place.

**See Also**: `.github/instructions/llaminar-v2-architecture.instructions.md` for comprehensive V2 documentation.

---

## Build System

### V2 Build Process

```bash
# V2 Debug build (experimental)
cmake -B build_v2 -S src/v2 -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build_v2 --parallel

# V2 Release build
cmake -B build_v2_release -S src/v2 -DCMAKE_BUILD_TYPE=Release
cmake --build build_v2_release --parallel

# Run V2 (device enumeration)
./build_v2/src/v2/llaminar2 --list-devices
```

**V2 Build Targets**:
- `llaminar2_core`: Core V2 library
- `llaminar2`: V2 executable (minimal functionality)

## Canonical Runtime Configuration

### Running Llaminar with Optimal Settings

Always use the canonical launch script for optimal performance:

```bash
# Canonical way to run Llaminar (RELEASE mode)
./run_llaminar.sh [arguments]

# DEBUG mode version (for smoke testing new features with debug logging etc):
./run_llaminar_debug.sh [arguments]

# Examples (RELEASE mode builds)
./run_llaminar.sh -v --print-topology
./run_llaminar.sh -m models/qwen2.5-0.5b-instruct-q4_0.gguf -v # [any other arguments...]
```

## Benchmark Mode

### Running Performance Benchmarks

Llaminar provides a dedicated `--benchmark` mode for clean performance measurement:

```bash
# Recommended: Use canonical launcher (RELEASE mode)
./run_llaminar.sh --benchmark \
  -m models/qwen2.5-0.5b-instruct-q8_0.gguf \
  -p "Your prompt here" \
  -n 50
```

### Benchmark Features

- **Separate prefill/decode timing** - Independent measurement of both phases
- **Clean output** - Minimal logging (ERROR level only) with formatted metrics
- **Greedy sampling** - Deterministic token selection for reproducible results
- **MPI-aware** - Handles tokenization on rank 0 with proper broadcast to all ranks
- **Professional formatting** - Box-drawing characters for clear metric display

### Benchmark Output Metrics

```
╔══════════════════════════════════════════════════════════════╗
║ PREFILL PHASE                                                ║
║   Tokens:              8 tokens                              ║
║   Time:          1216.49 ms                                  ║
║   Throughput:       6.58 tok/s                               ║
╠══════════════════════════════════════════════════════════════╣
║ DECODE PHASE                                                 ║
║   Tokens:             50 tokens                              ║
║   Time:         48095.52 ms                                  ║
║   Throughput:       1.04 tok/s                               ║
╚══════════════════════════════════════════════════════════════╝
```

### Implementation Details

- **Source**: `src/BenchmarkRunner.{h,cpp}` (~300 lines)
- **Integration**: `src/Main.cpp`, `src/ArgumentParser.{h,cpp}`
- **Key patterns**:
  - Logits fetched via `pipeline.logits(latest_logits)` after prefill/decode
  - Token broadcast from rank 0 to all ranks via MPI_Bcast
  - DummyTokenizer on non-rank-0 processes for API compatibility
  - Greedy sampling: argmax over last logits row

### Benchmark Defaults and Configuration

**Standard Benchmark:**
```bash
# Uses intelligent defaults if -p not provided:
#   - Auto-generated ~512 token prompt (mixed technical/narrative)
#   - 128 decode tokens
./run_llaminar.sh --benchmark -m model.gguf
```

**Phase-Specific Benchmarks:**
```bash
# Prefill-only (decode skipped when -n 0)
./run_llaminar.sh --benchmark -m model.gguf -p "Long context..." -n 0

# Decode-only (prefill skipped when -p "")
./run_llaminar.sh --benchmark -m model.gguf -p "" -n 128
```

**Implementation Notes:**
- Auto-prompt generation in `src/ArgumentParser.cpp` (lines ~189-198)
- Conditional execution in `src/BenchmarkRunner.cpp`:
  - Prefill: Skipped if `token_count == 0`
  - Decode: Skipped if `n_predict == 0`
- Output shows "(SKIPPED)" for omitted phases
- TOTAL section omitted when only one phase runs

### Benchmarking Best Practices

```bash
# 1. Use Release builds for accurate timing
cmake -B build_v2_release -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build_v2_release --parallel

# 2. Disable snapshotting
unset LLAMINAR_SNAPSHOT_STAGES

# 3. Test various prompt lengths (prefill scales with tokens)
./run_llaminar.sh --benchmark -m model.gguf -p "Short" -n 50
./run_llaminar.sh --benchmark -m model.gguf -p "Much longer prompt..." -n 50

# 4. Vary decode length to measure sustained performance
./run_llaminar.sh --benchmark -m model.gguf -p "Test" -n 20   # Quick
./run_llaminar.sh --benchmark -m model.gguf -p "Test" -n 200  # Sustained

# 5. Use phase-specific modes to isolate performance
./run_llaminar.sh --benchmark -m model.gguf -p "" -n 128  # Decode-only
./run_llaminar.sh --benchmark -m model.gguf -n 0          # Prefill-only
```

### Performance Notes

- **Prefill scales with token count**: 2 tokens → 1.68 tok/s, 8 tokens → 6.58 tok/s, 512 tokens → 33.60 tok/s
- **Decode is consistent**: ~1.04 tok/s regardless of decode length (in Debug)
- **Debug vs Release**: Release builds expected to be 5-10x faster
- **Backend routing**: Small ops use OpenBLAS, large prefills may use COSMA (if threshold met)
- **NUMA optimization**: K/V cache and activations benefit from first-touch allocation (+10-40% on large models)

## Development Profiling (Advanced)

### Canonical Environment Variables

```bash
# OpenMP Settings (automatically set by run_llaminar.sh)
export OMP_NUM_THREADS=<cores_per_socket>  # Auto-detected physical cores per socket
export OMP_PLACES=sockets          # Place threads on sockets
export OMP_PROC_BIND=close         # Bind threads close together
export OMP_NESTED=false            # Disable nested parallelism
export OMP_DYNAMIC=false           # Disable dynamic adjustment
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=0             # Reduce blocking time

# MPI Settings (automatically set by run_llaminar.sh)
export OMPI_MCA_mpi_leave_pinned=1
export OMPI_MCA_btl_vader_single_copy_mechanism=none
export OMPI_MCA_btl_openib_allow_ib=1
```

### Manual MPI Execution (if needed)

```bash
# 2-socket system (most common)
export OMP_NUM_THREADS=28 OMP_PLACES=sockets OMP_PROC_BIND=close OMP_NESTED=false OMP_DYNAMIC=false 
export KMP_AFFINITY=granularity=fine,compact,1,0 KMP_BLOCKTIME=0 OPENBLAS_NUM_THREADS=28 
export GOTO_NUM_THREADS=28 MKL_NUM_THREADS=28 MKL_DYNAMIC=false OMPI_MCA_mpi_leave_pinned=1
export OMPI_MCA_btl_vader_single_copy_mechanism=none OMPI_MCA_btl_openib_allow_ib=1
export LLAMINAR_LOG_LEVEL=DEBUG
mpirun -np 2 --bind-to socket --map-by socket \
  --mca mpi_leave_pinned 1 \
  --mca btl_vader_single_copy_mechanism none \
  --report-bindings ./build/llaminar
```

## Testing Guidelines (V1)

**Note**: This section describes V1 testing infrastructure. V2 testing infrastructure is documented separately below.

### V2 Testing Conventions

**Test File Naming** (V2 only):
- Test files MUST use CamelCase format: `Test__ClassName.cpp`
- `ClassName` is the **class under test** (the class being tested)
- The test suite name in the file MUST match: `TEST(Test__ClassName, ...)`
- Examples:
  - Testing `ModelLoader` class → File: `Test__ModelLoader.cpp`, Suite: `Test__ModelLoader`
  - Testing `Qwen2Pipeline` class → File: `Test__Qwen2Pipeline.cpp`, Suite: `Test__Qwen2Pipeline`
  - Testing `FP32Tensor` class → File: `Test__FP32Tensor.cpp`, Suite: `Test__FP32Tensor`
- The double underscore `__` separates "Test" prefix from the class name

**Rationale**: Clear association between test and the specific class being tested, matches V2's CamelCase conventions.

### V2 Test Organization

Llaminar V2 has a comprehensive test suite organized into several categories:

1. **Unit Tests** in `tests/v2/unit/`: Individual component validation (no model loading, fast)
2. **Integration Tests** in `tests/v2/integration/`: Full pipeline tests with model loading. These are longer running than unit tests. **NOTE**: Integration tests always run against the `build_v2_release` build when run via ctest!
3. **E2E Tests** in `tests/v2/e2e/`: Ground truth parity comparison against PyTorch reference. Leverages helpers in `tests/v2/pytorch_parity/` to compare per-layer snapshots between llaminar and Pytorch reference at different stages of inferencing
5. **Performance Tests** in `tests/v2/performance/`: Performance tests of various components of the llaminar stack, especially kernels

**PYTORCH SNAPSHOT FRAMEWORK:**
- **Pytorch Parity - Python project sources**: See `python/reference/` in the workspace.
- **Comprehensive documentation**: See `docs/v2/SNAPSHOT_FRAMEWORK_DESIGN.md`

### Running Tests

```bash
# All tests
cd /workspaces/llaminar && ctest --test-dir build_v2 -R --output-on-failure

# Just unit tests
cd /workspaces/llaminar && ctest --test-dir build_v2 -R "^V2_Unit_" --output-on-failure

# Run a specific test in a single test file
cd /workspaces/llaminar && GTEST_FILTER=The.Specific.GTest.Name ctest --test-dir build_v2 -R V2_My_Test_File -V
```

### CTest Label Best Practices

**Philosophy**: Labels should describe **what** is tested, not **when** it was developed.

V2 tests use a **4-tier hierarchical labeling system** for flexible filtering and self-documentation:

#### Tier 1 - Test Type (Required)
- `Unit` - Isolated component tests (no model loading, fast)
- `Integration` - Multi-component tests (may load models)
- `E2E` - End-to-end pipeline tests (full inference workflow)
- `Parity` - Ground truth validation (PyTorch/llama.cpp comparison)
- `Performance` - Benchmarks (run manually, not in standard CTest suites)

**Folder-to-Label Mapping** (`tests/v2/`):
- `tests/v2/unit/` → Must include `Unit` label
- `tests/v2/integration/` → Must include `Integration` label
- `tests/v2/e2e/` → Must include `E2E` label
- `tests/v2/performance/` → Must include `Performance` label
- Future `tests/v2/parity/` → Must include `Parity` label

#### Tier 2 - Architecture (Optional)
- `V2` - V2 architecture tests
- `V1` - V1 architecture tests (when needed for disambiguation)

#### Tier 3 - Component (Specific)
Use labels that describe the **system component** being tested:
- `DeviceManagement` - Device orchestration, discovery, selection
- `TensorOperations` - Tensor creation, manipulation, conversion
- `Kernels` - Computational kernels (GEMM, RoPE, Attention)
- `ModelLoading` - GGUF parsing, weight loading, verification
- `PipelineExecution` - Pipeline lifecycle, forward pass, factory
- `WeightPlacement` - Weight distribution strategies
- `DataTransfer` - Cross-device buffer management
- `ArgumentParsing` - CLI argument handling

#### Tier 4 - Feature (Granular)
Use labels that describe **specific capabilities** being tested:
- `Orchestration` - Orchestrator-specific features
- `Quantization` - Quantized tensor formats
- `MultiDevice` - Heterogeneous execution
- `CrossDevice` - Cross-device operations
- `HeterogeneousExecution` - Mixed device types
- `BasicFeatures` / `AdvancedFeatures` - Capability level
- Format-specific: `IQ4_NL`, `FP32`, `BF16`
- Operation-specific: `GEMM`, `RoPE`, `Attention`

#### Naming Conventions

**DO**:
- ✅ Use **CamelCase** for all labels (matches V2 file naming: `Test__ClassName.cpp`)
- ✅ Be **specific**: `PipelineFactory` not `Factory`, `FP32Tensor` not `Tensor`
- ✅ Use **feature names**: `WeightPlacement`, `DeviceOrchestration`
- ✅ Apply **multiple labels** per test for flexible filtering
- ✅ Use standard abbreviations: `CPU`, `GPU`, `MPI`, `GEMM`, `GGUF`

**DON'T**:
- ❌ Use **timeline-based labels**: `Phase1`, `Phase2`, `Milestone3`
- ❌ Use **generic labels** without context: `Factory`, `Pipeline`, `Tensor`
- ❌ Use **ambiguous abbreviations**: `P1`, `Mgmt`, `Exec`
- ❌ Mix **naming styles**: Stick to CamelCase throughout

#### Example: Good vs Bad Labels

```cmake
# ❌ BAD: Timeline-based, generic, unhelpful
set_tests_properties(v2_test_orchestrator 
    PROPERTIES LABELS "Phase1;Unit;V2")

# ❌ BAD: Too generic, unclear what's tested
set_tests_properties(v2_test_factory 
    PROPERTIES LABELS "Factory;Unit;V2")

# ✅ GOOD: Feature-descriptive, hierarchical, filterable
set_tests_properties(v2_test_device_orchestrator 
    PROPERTIES LABELS "V2;Unit;DeviceManagement;Orchestration;BasicFeatures")

# ✅ GOOD: Specific component, multiple filter dimensions
set_tests_properties(v2_test_iq4nl_gemm 
    PROPERTIES LABELS "V2;Unit;TensorOperations;Quantization;IQ4_NL;GEMM")
```

#### Filtering Examples

Multiple labels enable flexible test filtering:

```bash
# By test type (all unit tests) - Used by precommit hook
ctest -R "^V2_Unit_"  # Regex filter by test name
ctest -L Unit         # Label filter (also works)

# By component (all device management tests)
ctest -L DeviceManagement

# By feature (all quantization tests)
ctest -L Quantization

# Combined filters (device management orchestration tests)
ctest -L "DeviceManagement" -L "Orchestration"

# Exclude specific labels (all tests except performance)
ctest -LE Performance

# Precommit hook command (unit tests only)
ctest --test-dir build_v2 -R "^V2_Unit_" --output-on-failure
```

#### Adding New Tests

When adding tests to `tests/v2/CMakeLists.txt`, follow this pattern:

```cmake
# 1. Create test executable
add_executable(v2_test_my_feature Test__MyFeature.cpp)
target_link_libraries(v2_test_my_feature llaminar2_core GTest::gtest_main)

# 2. Add test with hierarchical labels
add_v2_test(V2_Unit_MyFeature
    COMMAND v2_test_my_feature
    LABELS "V2;Unit;ComponentName;FeatureName;SpecificDetail"
)
```

**Label Selection Checklist**:
1. ✅ Does the label describe **what** is tested, not **when**?
2. ✅ Would a new developer understand what the label means?
3. ✅ Can tests be logically filtered using this label?
4. ✅ Is the label **specific** enough to be useful?
5. ✅ Does the label follow **CamelCase** convention?

#### Documentation

Test label conventions are documented in:
- **`tests/v2/CMakeLists.txt`**: 50-line comment block with tier definitions and examples
- **`changelog/2025-10-24-ctest-label-standardization.md`**: Complete migration guide

**See Also**:
- CTest documentation: https://cmake.org/cmake/help/latest/manual/ctest.1.html#ctest-labels
- V2 test organization: `tests/v2/CMakeLists.txt` (lines 1-100)

### V2 Performance Testing

**V2 Performance Test Suite** (`tests/v2/performance/`)

V2 includes a dedicated performance test framework for benchmarking component-level throughput, latency, and memory usage. Performance tests are integrated into CTest with optimal MPI/OpenMP settings.

**⚠️ CRITICAL**: V2 **requires `-march=native`** for SIMD optimizations (AVX512/AVX2/FMA). Without it, performance is 4-12× slower due to scalar fallback. This is now enabled by default in `src/v2/CMakeLists.txt`.

#### Running V2 Performance Tests

```bash
# From workspace root - first ensure Release build exists
cmake -B build_v2_release -S src/v2 -DCMAKE_BUILD_TYPE=Release
cmake --build build_v2_release --parallel

# Run all V2 performance tests
cd build_v2
ctest -L Performance --verbose

# Run specific benchmarks
ctest -L "Performance;GEMM" --verbose          # All GEMM benchmarks
ctest -L "Performance;IQ4_NL" --verbose        # IQ4_NL quantized tests
ctest -R "V2_Perf_IQ4NL_GEMM" --verbose        # Specific test by name
```

#### Available V2 Benchmarks

**IQ4_NL GEMM Performance** (`Perf__IQ4_NL_GEMM`)
- Benchmarks IQ4_NL quantized matrix multiplication with AVX512/AVX2 SIMD
- Test cases: Single token, small batch (32), medium batch (128), large batch (512)
- Metrics: Time/iteration (ms), throughput (GFLOPS), memory bandwidth (GB/s)
- Uses real Qwen 2.5 0.5B IQ4_NL weights
- **Expected Performance** (Release with AVX512):
  - Small batch (32): ~8 GFLOPS
  - Medium batch (128): ~20 GFLOPS
  - Large batch (512): ~25 GFLOPS

**Optimal Settings Automatically Applied:**
- CPU topology detection (sockets, cores per socket)
- OpenMP: `OMP_NUM_THREADS=$CORES_PER_SOCKET`, `OMP_PLACES=sockets`, `OMP_PROC_BIND=close`
- MPI core pinning: `--bind-to socket`, `--map-by socket`, `--report-bindings`
- BLAS threading: `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` auto-configured
- **SIMD Optimization**: `-march=native -mtune=native` (enables AVX512/AVX2/FMA)

#### Adding New V2 Performance Tests

```bash
# 1. Create performance test (e.g., Perf__MyFeature.cpp)
# File: tests/v2/performance/Perf__MyFeature.cpp

# 2. Add to tests/v2/CMakeLists.txt
add_executable(v2_perf_my_feature performance/Perf__MyFeature.cpp)
target_link_libraries(v2_perf_my_feature llaminar2_core GTest::gtest GTest::gtest_main)

add_v2_perf_test(V2_Perf_MyFeature
    COMMAND v2_perf_my_feature
    LABELS "V2;Performance;MyComponent;MyFeature"
    MPI_PROCS 1  # Single rank for pure performance
)

# 3. Build and run
cmake --build build_v2_release --target v2_perf_my_feature --parallel
cd build_v2 && ctest -L Performance -R "MyFeature" --verbose
```

**Best Practices:**
- Use Release builds (`-O3 -DNDEBUG -march=native`)
- Include warmup iterations (3-10) before timed section
- Use MPI barriers around timed code
- Test multiple workload sizes (small/medium/large)
- Report multiple metrics (time, GFLOPS, bandwidth)
- Use realistic input data (not all zeros)

**Documentation:**
- Comprehensive guide: `tests/v2/performance/README.md` (500+ lines)
- Framework details: `changelog/2025-10-24-v2-performance-test-framework.md`
- V2 architecture: `.github/instructions/llaminar-v2-architecture.instructions.md`

## Checking Files for Compile Errors / Problems

It is possible to use the `problems` tool against a particular filename in order to get a list of problems (compile errors) in the file. Do this after editing a file but before building, as the feedback loop is faster.

## Debugging with GDB

### Basic GDB Setup

Ensure you're working on a DEBUG build or there won't be any symbols.

### Debugging Segfaults: MPI-Specific Debugging: Getting a Backtrace for a Segfault with GDB

**Recommended: GDB Command File Approach** (works in containers, no GUI needed)

```bash
# 1. Create GDB command file to avoid interactive prompts
cat > /tmp/gdbcommands.txt << 'EOF'
set debuginfod enabled off
set pagination off
handle SIGSEGV stop print
run
thread apply all bt full
quit
EOF

# 2. Set environment variables necessary for OMP / MPI and backend blas to work properly
export OMP_NUM_THREADS=28 OMP_PLACES=sockets OMP_PROC_BIND=close OMP_NESTED=false OMP_DYNAMIC=false 
export KMP_AFFINITY=granularity=fine,compact,1,0 KMP_BLOCKTIME=0 OPENBLAS_NUM_THREADS=28 
export GOTO_NUM_THREADS=28 MKL_NUM_THREADS=28 MKL_DYNAMIC=false OMPI_MCA_mpi_leave_pinned=1
export OMPI_MCA_btl_vader_single_copy_mechanism=none OMPI_MCA_btl_openib_allow_ib=1
export LLAMINAR_LOG_LEVEL=DEBUG

# 3. Run both MPI ranks under GDB with command file and proper OMP/MPI settings
cd /workspaces/llaminar
timeout 120 mpirun -np 2 \
  gdb -x /tmp/gdbcommands.txt --args \
  ./build_v2/tests/v2/v2_test_qwen2_e2e_correctness \
  --gtest_filter=Qwen2E2ECorrectness.SingleTokenInference 2>&1 | tee gdb_output.log

# 4. Analyze backtrace
grep -A 50 "Program received signal" gdb_output.log
```

**Key Points**:
- `--args` separates GDB options from executable arguments (CRITICAL!)
- Command file prevents blocking on debuginfod/pagination prompts
- `thread apply all bt full` shows all threads with variables
- Works in all container environments
- Timeout prevents infinite hangs

**Alternative: Per-Rank Logging** (when backtrace differs by rank)

```bash
export OMP_NUM_THREADS=28 OMP_PLACES=sockets OMP_PROC_BIND=close OMP_NESTED=false OMP_DYNAMIC=false 
export KMP_AFFINITY=granularity=fine,compact,1,0 KMP_BLOCKTIME=0 OPENBLAS_NUM_THREADS=28 
export GOTO_NUM_THREADS=28 MKL_NUM_THREADS=28 MKL_DYNAMIC=false OMPI_MCA_mpi_leave_pinned=1
export OMPI_MCA_btl_vader_single_copy_mechanism=none OMPI_MCA_btl_openib_allow_ib=1
export LLAMINAR_LOG_LEVEL=DEBUG

# Capture separate log per MPI rank
timeout 120 bash -c 'mpirun -np 2 bash -c "gdb -x /tmp/gdbcommands.txt --args ./build/my_test 2>&1 | tee /tmp/gdb_rank_\$OMPI_COMM_WORLD_RANK.log"'

# Then examine rank-specific backtraces
grep -A 50 "Program received signal" /tmp/gdb_rank_0.log
grep -A 50 "Program received signal" /tmp/gdb_rank_1.log
```

#### ASAN for more complex, double-free style segfaults

Use ASAN for localizing more complex double-free style issues:

```bash
cmake -B build_v2 -S src/v2 -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS="-g3 -O0 -fno-omit-frame-pointer -fsanitize=address" -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON 

ASAN_OPTIONS=halt_on_error=0:detect_leaks=0 timeout 240 mpirun -np 2 ./build/<your_test> --gtest_filter=<your_test_filter>
```

Just don't forget to reconfigure cmake to disable ASAN when you're done debugging!

### Common Debugging Scenarios

```bash
# Memory issues (use with caution - slows execution significantly)
export OMP_NUM_THREADS=28 OMP_PLACES=sockets OMP_PROC_BIND=close OMP_NESTED=false OMP_DYNAMIC=false 
export KMP_AFFINITY=granularity=fine,compact,1,0 KMP_BLOCKTIME=0 OPENBLAS_NUM_THREADS=28 
export GOTO_NUM_THREADS=28 MKL_NUM_THREADS=28 MKL_DYNAMIC=false OMPI_MCA_mpi_leave_pinned=1
export OMPI_MCA_btl_vader_single_copy_mechanism=none OMPI_MCA_btl_openib_allow_ib=1
export LLAMINAR_LOG_LEVEL=DEBUG
mpirun -np 2 valgrind --tool=memcheck --leak-check=full ./build/llaminar

# NUMA binding issues
numactl --hardware
numactl --show
export OMP_NUM_THREADS=28 OMP_PLACES=sockets OMP_PROC_BIND=close OMP_NESTED=false OMP_DYNAMIC=false 
export KMP_AFFINITY=granularity=fine,compact,1,0 KMP_BLOCKTIME=0 OPENBLAS_NUM_THREADS=28 
export GOTO_NUM_THREADS=28 MKL_NUM_THREADS=28 MKL_DYNAMIC=false OMPI_MCA_mpi_leave_pinned=1
export OMPI_MCA_btl_vader_single_copy_mechanism=none OMPI_MCA_btl_openib_allow_ib=1
export LLAMINAR_LOG_LEVEL=DEBUG
mpirun -np 2 numactl --cpubind=0 ./build/llaminar : numactl --cpubind=1 ./build/llaminar

# COSMA hanging issues (use timeouts)
export OMP_NUM_THREADS=28 OMP_PLACES=sockets OMP_PROC_BIND=close OMP_NESTED=false OMP_DYNAMIC=false 
export KMP_AFFINITY=granularity=fine,compact,1,0 KMP_BLOCKTIME=0 OPENBLAS_NUM_THREADS=28 
export GOTO_NUM_THREADS=28 MKL_NUM_THREADS=28 MKL_DYNAMIC=false OMPI_MCA_mpi_leave_pinned=1
export OMPI_MCA_btl_vader_single_copy_mechanism=none OMPI_MCA_btl_openib_allow_ib=1
export LLAMINAR_LOG_LEVEL=DEBUG
timeout 60 mpirun -np 2 ./build/llaminar
```

## Kernel Development (V1)

**Note**: This section describes V1 operator-based kernel development. For V2's operator-free approach, see [Kernel Development (V2)](#kernel-development-v2).

### Kernel Base Classes

All V1 kernels should inherit from appropriate base classes:
```cpp
// For MPI-aware kernels
class MyKernel : public MPIKernelBase {
public:
    bool execute(const std::vector<std::shared_ptr<TensorBase>>& inputs,
                std::vector<std::shared_ptr<TensorBase>>& outputs) override;
};

// For simple kernels
// Historical example (pre-MPI refactor). All production kernels now derive from MPIKernelBase.
class SimpleKernel : public KernelBase {
    // Similar interface without MPI context
};
```

### Tensor Integration

Use the hybrid tensor system appropriately:
```cpp
// Automatic tensor selection based on size and MPI context
auto tensor = TensorFactory::create_auto({1024, 1024});

// Explicit tensor types
auto simple = TensorFactory::create_simple({512, 512});
auto cosma = TensorFactory::create_cosma({2048, 2048}, "operation_name", mpi_rank);

// Legacy compatibility
std::shared_ptr<Tensor> legacy = std::make_shared<Tensor>({256, 256});
auto upgraded = TensorFactory::from_tensor(legacy);
```

**NUMA Considerations:**
- SimpleTensor automatically applies NUMA first-touch for allocations ≥128KB
- No code changes needed - optimization is transparent
- Controlled via `LLAMINAR_NUMA_FIRST_TOUCH` environment variable
- Most beneficial for:
  - K/V cache tensors (96MB-4GB)
  - Large activation buffers
  - Intermediate computation results

**When Adding New Tensor Allocations:**
1. Use SimpleTensor for standard row-major tensors
2. First-touch happens automatically if size ≥128KB
3. Verify with `LLAMINAR_NUMA_VERIFY_LOCALITY=1` during testing
4. For custom allocations, follow SimpleTensor::numaFirstTouch pattern

### Error Handling Patterns

```cpp
bool MyKernel::execute(const std::vector<std::shared_ptr<TensorBase>>& inputs,
                      std::vector<std::shared_ptr<TensorBase>>& outputs) {
    try {
        // Validate inputs
        if (inputs.size() != expected_input_count) {
            LOG_ERROR("Invalid input count: " << inputs.size());
            return false;
        }
        
        // Check tensor dimensions
        const auto& input_shape = inputs[0]->shape();
        if (input_shape.size() != 2) {
            LOG_ERROR("Expected 2D tensor, got " << input_shape.size() << "D");
            return false;
        }
        
        // Perform computation
        // ... implementation ...
        
        LOG_DEBUG("Kernel execution successful");
        return true;
        
    } catch (const std::exception& e) {
        LOG_ERROR("Kernel execution failed: " << e.what());
        return false;
    }
}
```

## Kernel Development (V2)

**Purpose**: V2 kernel development follows the **operator-free design** with direct pipeline orchestration and the **IBlockDecoder strategy pattern** for quantized tensors.

### Core V2 Principles

1. **No Operator Layer**: Pipelines call kernels directly (no `MPILinearOperator`, etc.)
2. **Per-Tensor Device Affinity**: Tensors know their device placement
3. **Strategy Pattern**: Generic kernels + format-specific decode strategies
4. **ITensor Interfaces**: `ITensorGemm`, `ITensorAttention`, `ITensorRoPE`, etc.

### IBlockDecoder Strategy Pattern

**Problem**: Quantized tensors (IQ4_NL, Q6_K, Q8_0) need format-specific decode logic, but we want a **single generic GEMM kernel**.

**Solution**: IBlockDecoder interface + QuantizedGemmKernel

```cpp
// src/v2/tensors/TensorKernels.h
class IBlockDecoder {
public:
    __attribute__((always_inline))
    virtual void decode_block_at(size_t row_idx, size_t k_block_offset, float* output) const = 0;
    
    __attribute__((always_inline))
    virtual const void* get_raw_block_at(size_t row_idx, size_t k_block_offset) const = 0;
    
    __attribute__((always_inline))
    virtual size_t block_size() const = 0;
};

// src/v2/tensors/Tensors.h
class IQ4_NLTensor : public TensorBase, public IBlockDecoder {
    // Inline implementation for zero overhead
    void decode_block_at(size_t row_idx, size_t k_block_offset, float* output) const override {
        const IQ4_NLBlock& block = blocks_[row_idx * blocks_per_row_ + k_block_offset];
        decodeBlock(block, output);  // Format-specific decode
    }
    
    size_t block_size() const override { return 32; }  // IQ4_NL: 32 elements/block
    
    std::unique_ptr<ITensorGemm> createGemm() const override {
        return std::make_unique<QuantizedGemmKernel>(this);  // Generic kernel
    }
};

// src/v2/kernels/cpu/QuantizedGemm.h
class QuantizedGemmKernel : public ITensorGemm {
    const IBlockDecoder* decoder_;  // Strategy interface
    
    bool multiply(...) override {
        // Generic implementation works for all quantized formats
        decoder_->decode_block_at(j, kb, B_block);  // Inlined (zero overhead)
        // ... accumulate ...
    }
};
```

**Benefits**:
- ✅ **Code Reuse**: ~350 lines generic kernel vs ~1000 lines per format
- ✅ **Zero Overhead**: `always_inline` eliminates virtual dispatch
- ✅ **Extensibility**: New formats just implement IBlockDecoder

### Adding New Quantized Formats (V2)

```cpp
// tensors/Q6_KTensor.h
class Q6_KTensor : public TensorBase, public IBlockDecoder {
public:
    void decode_block_at(size_t row_idx, size_t k_block_offset, float* output) const override {
        const Q6_KBlock& block = blocks_[row_idx * blocks_per_row_ + k_block_offset];
        decodeQ6KBlock(block, output);  // Q6_K-specific decode
    }
    
    size_t block_size() const override { return 256; }  // Q6_K: 256 elements/block
    
    std::unique_ptr<ITensorGemm> createGemm() const override {
        return std::make_unique<QuantizedGemmKernel>(this);  // Reuse generic kernel!
    }
    
private:
    static void decodeQ6KBlock(const Q6_KBlock& block, float* output);
};
```

**Result**: Single QuantizedGemmKernel works for IQ4_NL, Q6_K, Q8_0, etc.

### V2 Kernel Interface Design

All V2 kernels implement ITensor* interfaces:

```cpp
// src/v2/tensors/TensorKernels.h
class ITensorGemm {
public:
    virtual bool multiply(const float* A, float* C, int m, int n, int k, ...) = 0;
};

class ITensorAttention {
public:
    virtual bool execute(const TensorBase* Q, const TensorBase* K, 
                        const TensorBase* V, TensorBase* output, ...) = 0;
};

class ITensorRoPE {
public:
    virtual bool execute(TensorBase* tensor, const RoPEParams& params) = 0;
};
```

**Key Difference from V1**: No MPI context in kernel interfaces (handled by pipeline)

### V2 Pipeline Orchestration

```cpp
// src/v2/pipelines/QwenPipeline.cpp
bool QwenPipeline::attention_block(int layer, TensorBase* in, TensorBase* out) {
    // Direct kernel calls (no operators!)
    auto device = device_mgr_.selectDevice(in->size_bytes(), OperationType::ATTENTION);
    auto gemm = device->getGemmKernel();
    auto rope = device->getRoPEKernel();
    
    auto Q = allocateTensor({seq_len, d_model});
    gemm->execute(in, weights_.wq[layer], Q.get(), {});  // Q projection
    
    rope->execute(Q.get(), {.seq_offset = 0, .theta_base = 10000.0f});
    // ... etc
}
```

**See Also**: `.github/instructions/llaminar-v2-architecture.instructions.md` for complete V2 documentation.

## MPI Development Best Practices

### Critical MPI Patterns

Based on our experience with hangs and synchronization issues:

```cpp
// ALWAYS use barriers around collective operations
MPI_Barrier(MPI_COMM_WORLD);
auto t0 = std::chrono::high_resolution_clock::now();
cosma::multiply(A, B, C, strategy, MPI_COMM_WORLD, 1.0f, 0.0f);
MPI_Barrier(MPI_COMM_WORLD);
auto t1 = std::chrono::high_resolution_clock::now();

// Safe MPI reduction pattern
float local_sum = compute_local_sum();
float global_sum;
MPI_Barrier(MPI_COMM_WORLD);  // Ensure all ranks ready
MPI_Allreduce(&local_sum, &global_sum, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
MPI_Barrier(MPI_COMM_WORLD);  // Ensure completion before proceeding
```

### Thread Safety Considerations

```cpp
// MPI thread support initialization
int provided;
MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
if (provided < MPI_THREAD_MULTIPLE) {
    LOG_WARN("MPI thread support insufficient");
}

// OpenMP + MPI hybrid parallelization
#pragma omp parallel for
for (size_t i = 0; i < local_work_size; ++i) {
    // Thread-safe local computation
    local_result[i] = compute(local_data[i]);
}
// Implicit barrier at end of parallel region
MPI_Allreduce(local_result, global_result, work_size, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
```

## Performance Optimization

### NUMA-Aware Allocation Patterns

**Critical Rule**: All allocations ≥128KB on hot paths (K/V cache, activations, weights) MUST use NUMA first-touch initialization.

**SimpleTensor Pattern** (src/tensors/SimpleTensor.h):
```cpp
void resize(const std::vector<size_t>& new_shape) {
    size_t new_size = 1;
    for (auto dim : new_shape) new_size *= dim;
    
    data_.resize(new_size);
    shape_ = new_shape;
    
    // NUMA first-touch for large tensors
    if (debugEnv().loader.numa_first_touch) {
        numaFirstTouch(data_.data(), new_size);
    }
}

static void numaFirstTouch(float* data, size_t count) {
    constexpr size_t THRESHOLD = 128 * 1024 / sizeof(float);  // 128KB
    if (count < THRESHOLD) return;
    
    #pragma omp parallel for
    for (size_t i = 0; i < count; ++i) {
        data[i] = 0.0f;
    }
}
```

**Why This Matters:**
- K/V cache: 96MB-4GB per model → wrong NUMA node = 2-3x slower access
- Activations: Frequent allocation/deallocation during decode
- First-touch policy: OS places pages where they're first written
- Parallel init: Each thread writes its portion → local NUMA placement

**Configuration:**
- Control: `debugEnv().loader.numa_first_touch` (default: true)
- Verification: `debugEnv().loader.numa_verify_locality` (diagnostic logging)
- Threshold: 128KB (consistent across ModelLoader and SimpleTensor)

**Performance Impact:**
- Small models (≤1B): +1-3%
- Large models (7B-13B): +10-40% on multi-socket systems
- Primary benefit: K/V cache access during autoregressive decode

### Threading Strategy

Our empirical findings for optimal performance:

```cpp
// Small operations: Single-threaded to avoid overhead
if (total_elements < 8192) {
    openblas_set_num_threads(1);
}

// Medium operations: Multi-threaded within socket
else if (total_elements < 1048576) {
    openblas_set_num_threads(omp_get_max_threads());
}

// Large operations: Distributed across MPI processes
else {
    // Use MPI distribution with local multi-threading
    openblas_set_num_threads(cores_per_numa_node);
}
```

### Memory Considerations

```cpp
// Memory-aware operation sizing
size_t available_memory = get_numa_memory_size() / 4;  // Conservative estimate
size_t max_matrix_elements = available_memory / sizeof(float) / 3;  // A, B, C matrices

if (m * n * k > max_matrix_elements) {
    LOG_WARN("Matrix too large for available memory, consider tiling");
    return false;
}
```

### Performance Measurement

```cpp
// Standard timing pattern
auto start = std::chrono::high_resolution_clock::now();
bool success = perform_operation();
auto end = std::chrono::high_resolution_clock::now();

if (success) {
    double ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
    double flops = 2.0 * m * n * k;  // multiply + add
    double gflops = flops / (ms * 1e6);
    LOG_INFO("Operation: " << ms << "ms, " << gflops << " GFLOPS");
}
```

## Code Quality Guidelines

### Logging Standards

```cpp
// Use appropriate log levels
LOG_ERROR("Critical failures that prevent operation");
LOG_WARN("Concerning conditions that don't prevent operation");
LOG_INFO("Important runtime information");
LOG_DEBUG("Detailed debugging information");
LOG_TRACE("Very verbose execution tracing");

// Include context in logs
LOG_DEBUG("Processing batch " << batch_id << "/" << total_batches 
          << ", sequence length: " << seq_len);
```

### Exception Handling

```cpp
// Graceful degradation pattern
try {
    return perform_optimal_operation();
} catch (const MPIException& e) {
    LOG_WARN("MPI operation failed: " << e.what() << ", using local fallback");
    return perform_local_operation();
} catch (const std::bad_alloc& e) {
    LOG_ERROR("Memory allocation failed: " << e.what());
    return false;
} catch (const std::exception& e) {
    LOG_ERROR("Unexpected error: " << e.what());
    return false;
}
```

### Testing New Features

```cpp
// Always add corresponding tests
TEST_CASE("MyNewKernel basic functionality") {
    auto kernel = std::make_unique<MyNewKernel>();
    auto input = TensorFactory::create_simple({32, 32});
    auto output = TensorFactory::create_simple({32, 32});
    
    std::vector<std::shared_ptr<TensorBase>> inputs = {input};
    std::vector<std::shared_ptr<TensorBase>> outputs = {output};
    
    REQUIRE(kernel->execute(inputs, outputs));
    // Validate outputs...
}
```

### Centralized Environment Access (debugEnv)

All new or refactored code on hot paths (kernels, matmul selection, attention assembly, tensor partition loops) MUST avoid direct `std::getenv` calls. Instead:

1. Add any new environment flag/knob to the structured snapshot in `src/utils/DebugEnv.h` inside the appropriate group (or create one).
2. Parse it once in `src/utils/DebugEnv.cpp` (lazy static initialization already provided) and expose typed fields.
3. Consume via `const auto &snap = debugEnv();` then reference `snap.<group>.<field>`.

Rationale:
- Eliminates repeated libc getenv lookups in tight loops (measurable on small decode shapes).
- Provides a single discoverable registry of supported flags with defaults and validation.
- Enables future dynamic reconfiguration (e.g., reload snapshot) without editing all call sites.

Migration Guidance:
- When touching a file that still has raw `std::getenv` for a flag already present in the snapshot, replace it immediately.
- If a flag is only used in test or one-off tooling code (not perf‑sensitive), direct calls are acceptable but still discouraged.
- For transient experimental flags, stage them in `debug_env` early; this prevents entropy and drift.

Example (Before → After):
```cpp
// BEFORE (hot path)
if (std::getenv("LLAMINAR_ATTN_MICRO_TRACE") && rank == 0) { ... }

// AFTER
const auto &env = debugEnv();
if (env.attention.micro_trace && rank == 0) { ... }
```

Do NOT add another ad-hoc snapshot facility; extend the existing one. If grouping is unclear, prefer adding a new subgroup struct within `src/utils/DebugEnv.h` rather than mixing unrelated flags.


## Documentation Standards
### Doxygen

All files and functions should be documented with the Doxygen format for readability and easy understanding of their purpose. For @author, use David Sanftenberg.


## Debug / Instrumentation Environment Variables

Use these to control verbosity, backend selection, and validation when working on kernels and distributed execution:

| Variable | Description | Default / Activation | Primary Scope |
|----------|-------------|----------------------|---------------|
| `LLAMINAR_DEQUANT_STATS` | Logs per-tensor dequant stats (min/max/mean/sample). | Disabled unless set to non-zero. | Quantized tensor loading |
| `LLAMINAR_DEQUANT_ANOMALIES` | Emits warnings for anomalous Q6_K (and future) values (NaN/Inf/huge). | Disabled unless set. | Dequant safety diagnostics |
| `LLAMINAR_QUANT_OUTPUT_BF16` | Enable BF16 activation storage (Phase 5). | 0 (off) | Activation memory optimization |
| `LLAMINAR_FORCE_FP32_RMSNORM` | Force FP32 for RMSNorm (safety-first). | 1 (on, force FP32) | Numerical stability |
| `LLAMINAR_ALLOW_BF16_RMSNORM` | Allow BF16 RMSNorm (experimental). | 0 (off) | Numerical stability |
| `LLAMINAR_FORCE_FP32_SOFTMAX` | Force FP32 for Softmax. | 1 (on, force FP32) | Numerical stability |
| `LLAMINAR_ALLOW_BF16_SOFTMAX` | Allow BF16 Softmax (experimental). | 0 (off) | Numerical stability |
| `LLAMINAR_FORCE_FP32_LOGITS` | Force FP32 for logits output. | 1 (on, force FP32) | Output precision |
| `LLAMINAR_COSMA_PREFILL_THRESHOLD` | Sequence length threshold to switch to COSMA prefill path. | 4096 | Prefill path gating |
| `ADAPTIVE_DISABLE_COSMA` | Force all operations down OpenBLAS/adaptive local path. | Unset (COSMA enabled when threshold met) | Backend selection override |
| `LLAMINAR_COSMA_MAX_RESIDENT_MB` | Soft memory budget for COSMA working set; fallback if exceeded. | 2048 | Memory safety / preflight |
| `LLAMINAR_COSMA_VALIDATE_TILE` | If >0, performs small tile OpenBLAS vs COSMA correctness spot-check (relative L2). | 0 (off) | Debug validation |
| `LLAMINAR_COSMA_LOG_LEVEL` | Verbosity for COSMA prefill instrumentation (`trace`..`error`). | info | Prefill diagnostics |
| `OMP_NUM_THREADS` / `OMP_PLACES` / `OMP_PROC_BIND` | Governs OpenMP thread placement & counts (run script sets). | Auto-set by `run_llaminar.sh` | Threading performance |
| `KMP_AFFINITY` / `KMP_BLOCKTIME` | Fine tuning for Intel OpenMP runtime responsiveness. | Script default | Latency tuning |

### Usage Patterns

```bash
# Rich dequant + anomaly diagnostics
export LLAMINAR_DEQUANT_STATS=1
export LLAMINAR_DEQUANT_ANOMALIES=1

# Enable BF16 activation storage (Phase 5)
export LLAMINAR_QUANT_OUTPUT_BF16=1
export LLAMINAR_FORCE_FP32_RMSNORM=0     # Must explicitly disable force flag
export LLAMINAR_ALLOW_BF16_RMSNORM=1     # Then enable BF16 support

# NOTE: FORCE_FP32 and ALLOW_BF16 flags are INDEPENDENT (not mutually exclusive)
# To enable BF16 for an operation, you must:
#   1. Set ALLOW_BF16_<operation>=1
#   2. Set FORCE_FP32_<operation>=0
# Both conditions must be true for BF16 to be used.

# Force baseline path (skip COSMA) for A/B perf comparison
export ADAPTIVE_DISABLE_COSMA=1

# Lower threshold to test COSMA with shorter prompts
export LLAMINAR_COSMA_PREFILL_THRESHOLD=1024
```

Keep heavy validation (stats, validate tiles) off for performance benchmarks.


```cpp
/**
 * @brief Performs distributed matrix multiplication with adaptive backend selection
 * 
 * @param A Input matrix A (m×k)
 * @param B Input matrix B (k×n)  
 * @param C Output matrix C (m×n)
 * @param m Number of rows in A and C
 * @param n Number of columns in B and C
 * @param k Number of columns in A and rows in B
 * @param is_prefill Whether this is a prefill operation (affects backend selection)
 * 
 * @return true if operation succeeded, false otherwise
 * 
 * @note This function automatically selects between OpenBLAS and COSMA based on
 *       operation size and MPI context. Small operations use single-threaded OpenBLAS,
 *       medium operations use multi-threaded OpenBLAS, and large prefill operations
 *       may use distributed computation.
 */
bool adaptiveMatMul(const float* A, const float* B, float* C,
                   int m, int n, int k, bool is_prefill = false);
```

## Common Pitfalls and Solutions

### MPI Hanging Issues

**Problem**: Infinite hangs in MPI collective operations
**Solution**: Always use MPI_Barrier before and after collective operations

### Memory Fragmentation

**Problem**: Large tensor allocations fail despite available memory
**Solution**: Use NUMA-aware allocation and conservative memory limits

### Performance Regression

**Problem**: New optimizations actually slow down common cases
**Solution**: Always benchmark against realistic workloads, not just synthetic tests

### Race Conditions

**Problem**: Inconsistent results in multi-threaded code
**Solution**: Use proper OpenMP pragmas and avoid shared mutable state

## Conclusion

This project emphasizes production reliability over peak theoretical performance. The lessons learned from our MPI and performance work show that:

1. **Empirical testing beats theoretical assumptions** - Always measure real performance
2. **MPI barriers are critical** - Prevent hangs with proper synchronization
3. **Adaptive approaches work** - Different backends for different problem sizes
4. **Graceful degradation** - Always have reliable fallback paths
5. **Threading is complex** - Start simple, optimize incrementally

When in doubt, prioritize correctness and reliability over raw performance.

## Extended COSMA Execution & Test Debug Controls (Addendum)

This addendum documents additional environment toggles and watchdog policies added after the primary guidelines were authored. They refine control over distributed GEMM path selection, correctness validation, and hang mitigation.

### Path Selection & Forcing
| Variable | Purpose | Default / Activation | Notes |
|----------|---------|----------------------|-------|
| `LLAMINAR_COSMA_FORCE_DIRECT` | Force direct COSMA distributed multiply path, bypassing fast/replicated heuristic. | Unset (0) | Use to stress-test COSMA on marginal sizes. |
| `LLAMINAR_COSMA_DIRECT_THRESHOLD_OPS` | Volume (m*n*k) threshold for auto-direct path when not forced. | Internal tuned default | Adjust only during tuning experiments. |
| `LLAMINAR_COSMA_FAST_PATH_THRESHOLD` | Volume below which replicated OpenBLAS + broadcast fast path is used. | Internal constant | Keeps tiny ops off COSMA. |
| `LLAMINAR_COSMA_FORCE_REPLICATED` | Force always replicated execution even when large. | Unset | Isolate COSMA issues by comparison. |
| `LLAMINAR_COSMA_FORCE_REPLICATED_DIAG` | Replicated path + extra diagnostics. | Unset | Superset of FORCE_REPLICATED. |
| `ADAPTIVE_DISABLE_COSMA` | Disable COSMA entirely (already documented). | Unset | Highest precedence bypass. |
| `LLAMINAR_COSMA_FORCE_DISTRIBUTED_ACT` | Force COSMA activation allocation even when fast path would skip. | Unset | Useful for tests needing distributed elementwise views. |

### Correctness / Validation
| Variable | Purpose | Notes |
|----------|---------|-------|
| `LLAMINAR_COSMA_COMPARE_REPLICATED` | Execute a full replicated OpenBLAS reference after direct COSMA and compare rel L2 + max abs. | High cost; not for perf runs. |
| `LLAMINAR_COSMA_VALIDATE_TILE` | Small tile OpenBLAS check (existing). | Prefer before full replicate compare. |
| `LLAMINAR_COSMA_FAST_UNVERIFIED` | Skip validations for fast path ops. | Use to reduce overhead with confidence. |
| `LLAMINAR_COSMA_AUTO_FIX_TRANSPOSE` | Attempt automatic orientation correction if mismatch detected. | Logs a warning when used. |
| `LLAMINAR_COSMA_DEBUG_RECON` | Verbose gather / ownership normalization logs. | Aids debugging reconstruction anomalies. |

### Allocation / Memory Safety
| Variable | Purpose | Notes |
|----------|---------|-------|
| `LLAMINAR_COSMA_MAX_RESIDENT_MB` | Single-allocation soft guard (existing). | Now also interacts with small-op allocation skip. |

### Structural / Small Matrix Diagnostics
| Variable | Purpose | Notes |
|----------|---------|-------|
| `LLAMINAR_COSMA_DUMP_SMALL` | Dump structural info for small fast-path matrices. | Per-rank concise summary. |

### Dequant & Fused Path Controls (Recap)
| Variable | Purpose | Notes |
|----------|---------|-------|
| `LLAMINAR_COSMA_DISABLE_FUSED_DEQUANT` | Disable fused quantized weight dequant+population. | Debug fallback. |

### Test Watchdog & Timeout Policy
All COSMA-associated tests have an external 60s CTest timeout. Internally, watchdog threads enforce earlier detection:

| Variable | Purpose | Behavior |
|----------|---------|----------|
| `LLAMINAR_COSMA_TEST_PHASE_TIMEOUT_MS` | Per-phase soft limit (stream, matmul, recon). | Warns / stack capture if exceeded (hard cap). |
| `LLAMINAR_COSMA_TEST_INTERNAL_TIMEOUT_MS` | Global internal watchdog for the whole test. | Aborts with per-rank stack dump. |
| `LLAMINAR_SKIP_MPI_IN_SINGLE_TEST` | Avoid multi-rank collectives in single-rank mode. | Bypasses MPI collectives safely. |
| `LLAMINAR_COSMA_TEST_TRACE` | Escalate verbosity for COSMA tests to trace. | Any non-empty value. |

### Recommended Debug Flows
1. Fast path validation: `LLAMINAR_COSMA_VALIDATE_TILE=64` + multi-rank small test.
2. Direct path correctness: `LLAMINAR_COSMA_FORCE_DIRECT=1 LLAMINAR_COSMA_COMPARE_REPLICATED=1`.
3. Reconstruction tracing: `LLAMINAR_COSMA_DEBUG_RECON=1 LLAMINAR_COSMA_TEST_TRACE=1`.
4. Orientation suspicion: enable `LLAMINAR_COSMA_AUTO_FIX_TRANSPOSE` plus replicated compare for before/after diff.

Disable heavy validation & trace logging prior to performance measurement to ensure representative timing.

## Documentation and Project Resources

### Key Documentation Files

**Developer Guidelines:**
- **`.github/copilot-instructions.md`** (this file): Comprehensive development guidelines for V1 and V2
- **`.github/instructions/llaminar-v2-architecture.instructions.md`**: Complete V2 architecture documentation
- **`.github/instructions/parity-test-framework.instructions.md`**: V1 parity testing guide

**Performance and Benchmarking:**
- `./run_llaminar.sh`: Canonical V1 launcher with optimal MPI/OpenMP settings (RELEASE mode: `build_release/`)
- `./run_llaminar_debug.sh`: V1 debug version of the above launcher (DEBUG mode: `build/`)
- `./run_batch_performance.sh`: V1 batch vs sequential performance comparison
- `./run_pytorch_parity_test.sh`: V1 PyTorch parity testing with metrics
- `./run_performance_demo.sh`: V1 production-style adaptive matmul demo

**Model Support:**
- **V1**: Qwen 2.5 family (0.5B-72B), LLaMA 3.x (prototype)
- **V2**: Basic infrastructure only (not production-ready)
- **Quantization**: Q4_0, Q6_K, Q8_0, IQ4_NL, F16, F32

**Key Source Directories (V1):**
- `src/`: Core V1 inference engine
- `src/operators/`: MPI-aware transformer operators
- `src/tensors/`: Hybrid tensor system (SimpleTensor, COSMATensor)
- `src/backends/`: Backend implementations (OpenBLAS, COSMA, CUDA/ROCm stubs)
- `src/weights/`: Weight loading and verification
- `src/utils/`: Utilities (logging, performance tracing, environment config)
- `tests/`: Comprehensive test suite (unit, integration, parity)

**Key Source Directories (V2):**
- `src/v2/`: V2 operator-free architecture
- `src/v2/tensors/`: Tensor base classes, quantized tensors (IQ4_NL)
- `src/v2/kernels/`: Kernel implementations (CPU, CUDA, ROCm, Vulkan)
- `src/v2/pipelines/`: Pipeline orchestration (QwenPipeline)
- `src/v2/backends/`: Backend device management
- `src/v2/utils/`: V2-specific utilities (MPIContext, CPUFeatures)

### Writing Documentation and Changelogs

When we write documentation and changelogs at the end of our work runs:

1. **Location**: Place in `changelog/` folder to avoid cluttering the repo root
2. **Naming**: Use ISO date prefix: `YYYY-MM-DD-description.md`
   ```bash
   # Get current date
   date +%Y-%m-%d
   # Example: 2025-10-16-batch-parity-testing-implementation.md
   ```
3. **Content**: Include:
   - Summary of changes
   - Key findings or results
   - Code changes with file paths
   - Test results (pass/fail counts)
   - Performance metrics (before/after)
   - Next steps or follow-up tasks

4. **Session Summaries**: For major work sessions, create a summary document:
   - `changelog/YYYY-MM-DD-session-summary.md`
   - Include: objectives, discoveries, outcomes, remaining work

### Project Status (October 2025)

**V1 Production Ready:**
- ✅ Sequential Qwen inference (0.5B-72B)
- ✅ Multi-rank MPI distribution
- ✅ OpenBLAS backend (all sizes)
- ✅ Intel MKL backend (BF16 quantized inference)
- ✅ COSMA backend (large prefill ≥8K tokens)
- ✅ NUMA-aware memory allocation
- ✅ PyTorch parity tests (prefill + decode)
- ✅ Batch attention parity (17/17 stages)

**V1 In Progress:**
- 🔄 Batch pipeline full validation (attention ✅, FFN/LM head ✅)
- 🔄 LLaMA adapter completion
- 🔄 CUDA/ROCm backend integration

**V2 Development Status:**
- ✅ Basic infrastructure (tensors, kernels, backends)
- ✅ IQ4_NL quantized tensor with IBlockDecoder pattern
- ✅ Generic QuantizedGemmKernel (335-451 GFLOPS)
- ✅ FP32/BF16 tensor types
- ✅ Device enumeration (CPU, CUDA, ROCm, Vulkan stubs)
- ❌ Full pipeline implementation (incomplete)
- ❌ MPI distribution (not yet ported)
- ❌ Production testing (not validated)

**Deprecated/Experimental:**
- 🧪 Graph-based execution (V1, deprecated, may be removed)
- 🧪 Adaptive transformer pipeline (V1, experimental)

### Getting Help

**Common issues:**
1. **MPI hangs**: Check barriers around collective operations, verify all ranks participate
2. **Numerical divergence**: Run parity tests to identify diverging stage
3. **Performance issues**: Profile with `run_batch_performance.sh`, check OpenMP thread settings
4. **Memory errors**: Enable NUMA verification, check first-touch allocation
5. **Build errors**: Clean build directory, verify dependencies (OpenBLAS, MPI, ScaLAPACK)

**Debug environment variables**: See "Debug / Instrumentation Environment Variables" section for full list of `LLAMINAR_*` flags.