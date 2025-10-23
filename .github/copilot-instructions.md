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
- [Debugging with GDB](#debugging-with-gdb)
- [Kernel Development (V1)](#kernel-development-v1)
- [Kernel Development (V2)](#kernel-development-v2)
- [MPI Development Best Practices](#mpi-development-best-practices)
- [Performance Optimization](#performance-optimization)
- [COSMA vs OpenBLAS Integration (V1)](#cosma-vs-openblas-integration-v1)
- [Code Quality Guidelines](#code-quality-guidelines)
- [Documentation Standards](#documentation-standards)

## Architecture Overview (V1 vs V2)

### V1 Architecture (Production - `src/`)

**Design Philosophy**: Operator-based MPI distributed inference

**Key Characteristics**:
- ✅ **Production Ready**: Fully functional Qwen inference with MPI distribution
- ✅ **Operator Abstraction**: `MPILinearOperator`, `MPIAttentionOperator`, etc.
- ✅ **Centralized Backend Selection**: MatMulBackendSelector for OpenBLAS/COSMA/MKL
- ✅ **Parity Testing**: PyTorch ground truth validation
- ✅ **Batch Processing**: BatchQwenPipeline for multi-sequence throughput

**When to use V1**:
- Production inference workloads
- MPI multi-node distributed execution
- Batch processing (multi-user serving)
- When you need proven, tested code

### V2 Architecture (Development - `src/v2/`)

**Design Philosophy**: Operator-free kernel-centric design

**Key Characteristics**:
- 🔄 **In Development**: Basic pipeline structure, not feature-complete
- 🎯 **No Operator Layer**: Pipelines call kernels directly
- 🎯 **Per-Tensor Device Affinity**: Each tensor knows its device placement
- 🎯 **Heterogeneous Execution**: Mix CUDA, ROCm, Vulkan in single run
- 🎯 **IBlockDecoder Pattern**: Generic quantized GEMM kernel (see V2 kernel development)

**When to use V2**:
- Experimenting with new architectures
- Multi-GPU heterogeneous execution (future)
- Learning the next-generation design
- Contributing to V2 development

**Migration Status**:
- ✅ Basic infrastructure (tensors, kernels, backends)
- ✅ IQ4_NL quantized tensor with fused GEMM
- ✅ Generic QuantizedGemmKernel (IBlockDecoder pattern)
- ❌ Full pipeline implementation (incomplete)
- ❌ MPI distribution (not yet ported)
- ❌ Production testing (not validated)

**See Also**: `.github/instructions/llaminar-v2-architecture.instructions.md` for comprehensive V2 documentation.

---

## Project Overview (V1)

Llaminar is a high-performance, distributed LLM inference engine built on:
- **Multi-Architecture Pipeline System**: Factory-based extensible architecture for Qwen, LLaMA, and future models
- **Centralized Backend Selection**: Single decision point via MatMulBackendSelector for OpenBLAS, COSMA, and Intel MKL
- **Hybrid Tensor System**: Zero-copy COSMA optimization with backward compatibility
- **MPI Distribution**: Multi-node inference with NUMA-aware deployment
- **Adaptive Backends**: OpenBLAS (baseline), COSMA (distributed), Intel MKL (BF16 acceleration)
- **Comprehensive Observability**: Structured environment snapshot, performance counters, validation framework

### Pipeline Architecture

**Sequential vs Batch Execution Modes:**
- **Sequential Pipelines**: Process one sequence at a time (QwenPipeline, LlamaPipelineAdapter)
  - Lower memory footprint
  - Simpler implementation
  - Ideal for single-user inference
  
- **Batch Pipelines**: Process multiple sequences simultaneously (BatchQwenPipeline)
  - Higher throughput for multi-user scenarios
  - More complex tensor management (sequence padding, batch dimension handling)
  - Requires careful attention to MPI aggregation and dimension ordering
  - **Parity Testing**: Automated validation that batch and sequential paths produce identical results (see `.github/instructions/parity-test-framework.instructions.md`)

**Operator-Based Architecture:**
All transformer operations are implemented as MPI-aware operators in `src/operators/`:
- `MPIEmbeddingOperator`: Token embedding lookup with vocabulary partitioning
- `MPILinearOperator` / `MPILinearBatchOperator`: Matrix multiplication with weight sharding
- `MPIAttentionOperator` / `MPIAttentionBatchOperator`: Multi-head attention with KV cache
- `MPIRMSNormOperator`: RMS normalization
- `MPIRoPEOperator`: Rotary position embeddings
- `MPISwiGLUOperator` / `MPISwiGLUBatchOperator`: SwiGLU activation with gating
- `MPIResidualOperator`: Residual connection

Each operator handles:
- Tensor partition specifications (which dimensions are sharded vs replicated)
- MPI collective operations (Allreduce, broadcast, gather)
- Backend selection (OpenBLAS, COSMA, or Intel MKL based on operation characteristics)
- NUMA-aware memory allocation

### Batch Processing Architecture

**Batch processing** enables efficient multi-sequence inference by processing multiple prompts simultaneously:

#### Key Differences: Sequential vs Batch

| Aspect | Sequential Pipeline | Batch Pipeline |
|--------|--------------------|-----------------|
| **Throughput** | 1 sequence at a time | N sequences simultaneously |
| **Memory** | Lower footprint | Higher (padded sequences) |
| **Complexity** | Simpler tensor shapes | Complex dimension handling |
| **MPI Aggregation** | Standard reduction | Requires careful batch-aware aggregation |
| **Use Case** | Single-user inference | Multi-user serving, batch jobs |

#### Batch Pipeline Implementation (`BatchQwenPipeline`)

**Core challenges solved:**

1. **Sequence Padding**: Variable-length sequences must be padded to uniform length
   - Handled by `BatchPaddingUtils`: Pads to longest sequence in batch
   - Tracks original lengths to avoid processing padding tokens

2. **Dimension Ordering**: Batch dimension handling throughout pipeline
   - Input: `[batch_size, max_seq_len]`
   - Activations: `[batch_size, seq_len, d_model]`
   - Attention: Special handling for batch dimension in score computation

3. **MPI Aggregation**: Batch-aware collective operations
   - Weight partitioning: Same as sequential (sharded across ranks)
   - Activation gathering: Must preserve batch dimension
   - Attention reduction: Per-sequence reduction within batch

4. **Memory Management**: Efficient allocation for batched tensors
   - Pre-allocated buffers sized for `batch_size × max_seq_len`
   - NUMA-aware first-touch for large activations
   - KV cache: Separate cache entries per sequence

#### Batch Operators

All operators have batch-aware variants:

- `MPILinearBatchOperator`: Batched linear projections (Q/K/V, FFN)
- `MPIAttentionBatchOperator`: Batched multi-head attention with per-sequence masking
- `MPISwiGLUBatchOperator`: Batched SwiGLU activation
- Shared operators: `MPIRMSNormOperator`, `MPIRoPEOperator` (dimension-agnostic)

**Key implementation pattern:**
```cpp
// Sequential: [seq_len, d_model]
auto output_seq = linear_op->forward(input, weight);

// Batch: [batch_size, seq_len, d_model]
auto output_batch = linear_batch_op->forward(input, weight, batch_size);
```

#### Parity Testing

**Critical validation**: Batch and sequential paths must produce identical results for the same input.

- **Test**: `tests/test_batch_correctness.cpp` (`BatchCorrectnessTest.BatchedAttentionStagesParity`)
- **Status**: ✅ **17/17 stages passing** with excellent numerical agreement (completed Oct 17, 2025)
- **Coverage**: Full pipeline validation from embedding through LM head
  - Embedding layer
  - 9 attention stages (norm, Q/K/V projections, RoPE, scores, weights, context, output, residual)
  - 6 FFN stages (norm, gate, up, SwiGLU, down, residual)
  - Final norm + LM head
- **Methodology**: Snapshot-based comparison using `SnapshotRegistry` and `SnapshotComparator`
- **Critical Bug Fixed**: Missing final RMSNorm in `BatchQwenPipeline::projectOutput()` (divergence reduced 470,000×)

For details, see:
- `.github/instructions/parity-test-framework.instructions.md` - Comprehensive parity testing guide
- `changelog/2025-10-17-batch-parity-extended-to-ffn-lm-head.md` - Detailed test results and bug fix
- `changelog/2025-10-17-batch-parity-and-performance-complete.md` - Complete session summary

#### Performance Characteristics

**When to use batch processing:**
- ✅ Multi-user serving (requests can be batched)
- ✅ Batch inference jobs (many prompts to process)
- ✅ High-throughput scenarios (latency less critical)

**When to use sequential:**
- ✅ Single-user interactive inference (low latency)
- ✅ Memory-constrained environments
- ✅ Variable-length sequences with large variance

**Benchmark scripts:**
```bash
# Compare batch vs sequential performance
./run_batch_performance.sh

# Shows:
# - Throughput improvement (tokens/sec)
# - Memory overhead
# - Scaling with batch size

# Run any benchmark with optimal MPI/OpenMP settings
./run_benchmark.sh <benchmark_executable> [args...]

# Examples:
./run_benchmark.sh benchmark_iq4nl_gemm
./run_benchmark.sh test_batch_performance --gtest_filter='*.Scaling'
```

### Key Architecture Components

**Core Pipeline Infrastructure:**
- `src/Main.cpp`: Application entry point with pipeline factory-based execution
- `src/AbstractPipeline.{h,cpp}`: Pipeline interface defining prefill/decode lifecycle
- `src/PipelineBase.{h,cpp}`: Base implementation for pipelines

**Model Implementations:**
- `src/QwenPipeline.{h,cpp}`: Sequential Qwen model implementation
- `src/BatchQwenPipeline.{h,cpp}`: Batched Qwen model implementation (multi-sequence processing)
- `src/QwenPipelineAdapter.{h,cpp}`: Qwen adapter implementing AbstractPipeline
- `src/BatchQwenPipelineAdapter.{h,cpp}`: Batch Qwen adapter implementing AbstractPipeline
- `src/LlamaPipelineAdapter.{h,cpp}`: LLaMA adapter (prototype)

**Backend and Execution:**
- `src/MatmulBackendSelection.{h,cpp}`: Centralized backend decision logic (OpenBLAS, COSMA, Intel MKL)
- `src/BackendSelector.{h,cpp}`: Backend selection infrastructure
- `src/AdaptiveMatmul.{h,cpp}`: Adaptive matrix multiplication with BF16 quantization support
- `src/backends/MKLBackend.{h,cpp}`: Intel MKL BF16 GEMM backend (optional, enabled with -DUSE_MKL=ON)
- `src/CosmaPrefillManager.{h,cpp}`: COSMA distributed prefill coordination
- `src/CosmaPrefillProvider.{h,cpp}`: COSMA-specific prefill implementation
- `src/OpenblasPrefillProvider.{h,cpp}`: OpenBLAS prefill implementation
- `src/PrefillProvider.{h,cpp}`: Base prefill provider interface
- `src/PrefillProviderBaseImpl.{h,cpp}`: Shared prefill provider implementation

**Validation and Testing:**
- `src/PrefillDiagnostics.{h,cpp}`: Modular prefill validation and comparison
- `src/ParityHooks.{h,cpp}`: Snapshot capture hooks for parity testing
- `src/PipelineSnapshotManager.{h,cpp}`: Pipeline-level snapshot management
- `src/PipelineStages.h`: Stage enumeration for snapshot capture

**Component Systems:**
- `src/operators/`: MPI-aware operators (attention, linear, embedding, RMSNorm, RoPE, SwiGLU)
- `src/tensors/`: Hybrid tensor system (SimpleTensor + COSMATensor)
- `src/utils/`: Utilities (DebugEnv, logging, performance tracing)
- `src/weights/`: Weight loading and management (ModelLoader, ModelWeightsProvider)
- `src/backends/`: Backend-specific implementations

**Batch Processing Utilities:**
- `src/BatchPaddingUtils.{h,cpp}`: Sequence padding and alignment for batch processing
- `src/SequentialBatchBenchmark.{h,cpp}`: Performance comparison between batch and sequential modes

## Build System

### V1 Build Process

```bash
# Debug build (development)
cmake -B build -S . -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build --parallel

# Release build (production)
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release \
  -DCOSMA_WITH_PROFILING=OFF \
  -DBUILD_SHARED_LIBS=OFF \
  -DCOSMA_SCALAPACK=CUSTOM \
  -DCOSMA_SCALAPACK_LINK_LIBRARIES=/usr/lib/x86_64-linux-gnu/libscalapack-openmpi.so
cmake --build build --parallel

# Intel MKL-enabled build (BF16 support)
cmake -B build_mkl -S . -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="/opt/intel/oneapi/mkl/latest" \
  -DUSE_MKL=ON \
  -DCOSMA_WITH_PROFILING=OFF \
  -DBUILD_SHARED_LIBS=OFF
cmake --build build_mkl --parallel
```

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

### Available Build Targets (V1)

```bash
# Core library only
cmake --build build --target llaminar_core --parallel

# Main executable
cmake --build build --target llaminar --parallel

# COSMA library
cmake --build build --target cosma --parallel

# Clean and rebuild
cmake --build build --target clean
cmake --build build --parallel
```

### Using VS Code Tasks

The project includes predefined VS Code tasks:
- `cmake: configure` - Configure with debug settings
- `cmake: build` - Build entire project
- `cmake: build core` - Build core library only
- `cmake: configure Release` - Configure for optimized production build
- `cmake: build Release` - Build in release mode

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
║   Time:          1216.49 ms                                 ║
║   Throughput:       6.58 tok/s                             ║
╠══════════════════════════════════════════════════════════════╣
║ DECODE PHASE                                                 ║
║   Tokens:             50 tokens                              ║
║   Time:         48095.52 ms                                 ║
║   Throughput:       1.04 tok/s                             ║
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
cmake -B build_release -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build_release --parallel

# 2. Disable heavy instrumentation
unset LLAMINAR_COSMA_VALIDATE_TILE
unset LLAMINAR_DEQUANT_STATS
unset LLAMINAR_EMBED_TRACE

# 3. Test various prompt lengths (prefill scales with tokens)
./run_llaminar.sh --benchmark -m model.gguf -p "Short" -n 50
./run_llaminar.sh --benchmark -m model.gguf -p "Much longer prompt..." -n 50

# 4. Vary decode length to measure sustained performance
./run_llaminar.sh --benchmark -m model.gguf -p "Test" -n 20   # Quick
./run_llaminar.sh --benchmark -m model.gguf -p "Test" -n 200  # Sustained

# 5. Use phase-specific modes to isolate performance
./run_llaminar.sh --benchmark -m model.gguf -p "" -n 128  # Decode-only
./run_llaminar.sh --benchmark -m model.gguf -n 0          # Prefill-only

# 6. For component benchmarks, use the generic benchmark runner
./run_benchmark.sh benchmark_iq4nl_gemm         # IQ4_NL GEMM performance
./run_benchmark.sh test_batch_performance       # Batch scaling tests
```

### Performance Notes

- **Prefill scales with token count**: 2 tokens → 1.68 tok/s, 8 tokens → 6.58 tok/s, 512 tokens → 33.60 tok/s
- **Decode is consistent**: ~1.04 tok/s regardless of decode length (in Debug)
- **Debug vs Release**: Release builds expected to be 5-10x faster
- **Backend routing**: Small ops use OpenBLAS, large prefills may use COSMA (if threshold met)
- **NUMA optimization**: K/V cache and activations benefit from first-touch allocation (+10-40% on large models)

## Development Profiling (Advanced)

### Performance Testing Scripts

Llaminar includes several specialized performance testing scripts:

```bash
# Batch vs Sequential Performance Comparison
./run_batch_performance.sh

# Production-style adaptive matmul demonstration
./run_performance_demo.sh

# PyTorch parity testing with performance metrics
./run_pytorch_parity_test.sh
```

**Important Distinction:**
- **`run_llaminar.sh --benchmark`**: Production inference benchmarking with real models
  - Use this for: End-to-end performance measurement, model comparisons, user-facing benchmarks
  - Outputs: Clean tok/s metrics for prefill/decode phases
  
- **Performance test executables**: Development profiling with GTest suite
  - Use this for: Analyzing specific components, batch vs sequential comparison, parity testing
  - Outputs: Detailed GTest metrics with component-level timing
  
- **`run_batch_performance.sh`**: Compares batch and sequential execution performance
  - Validates throughput improvements from batching
  - Measures memory overhead and scaling characteristics

### Generic Benchmark Runner

For component-level performance benchmarking, use the **generic benchmark runner** which provides canonical MPI/OpenMP configuration for any benchmark executable:

```bash
# Run any benchmark with optimal settings (auto-detects in build_release/)
./run_benchmark.sh benchmark_iq4nl_gemm

# With full path
./run_benchmark.sh ./build_release/test_performance

# Pass arguments (GTest filters, etc)
./run_benchmark.sh test_batch_performance --gtest_filter='*.ThroughputScaling'

# Override MPI/OpenMP settings if needed
LLAMINAR_MPI_PROCS=4 ./run_benchmark.sh my_benchmark
OMP_NUM_THREADS=16 ./run_benchmark.sh my_benchmark
```

**Features:**
- **Automatic path detection**: Searches `build_release/` and `build/` directories
- **Topology auto-detection**: Detects sockets, cores, hyperthreading
- **Canonical settings**: Same MPI/OpenMP configuration as `run_llaminar.sh`
- **Argument forwarding**: Passes all extra arguments to the benchmark
- **Environment overrides**: Customize via `LLAMINAR_MPI_PROCS`, `OMP_NUM_THREADS`

**Convenience Wrappers:**
```bash
# Specialized wrapper for IQ4_NL GEMM benchmark
./run_iq4nl_benchmark.sh  # Delegates to run_benchmark.sh

# These are equivalent:
./run_iq4nl_benchmark.sh
./run_benchmark.sh benchmark_iq4nl_gemm
```

**Creating New Benchmarks:**
```bash
# 1. Create benchmark source (e.g., tests/benchmark_my_feature.cpp)
# 2. Add to CMakeLists.txt
add_executable(benchmark_my_feature tests/benchmark_my_feature.cpp)
target_link_libraries(benchmark_my_feature llaminar_core)

# 3. Build it
cmake --build build_release --target benchmark_my_feature --parallel

# 4. Run with optimal settings
./run_benchmark.sh benchmark_my_feature
```

**See also:** `BENCHMARK_RUNNER_GUIDE.md` for comprehensive documentation including usage examples, configuration details, and troubleshooting.

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
- This convention applies ONLY to `tests/v2/` directory
- V1 tests in `tests/` continue using snake_case (e.g., `test_batch_correctness.cpp`)

**Rationale**: Clear association between test and the specific class being tested, matches V2's CamelCase conventions.

### Test Organization

Llaminar V1 has a comprehensive test suite organized into several categories:

1. **Smoke Tests** (~5s): Fast sanity checks for core functionality
2. **Unit Tests** (~2m30s): Individual component validation (no model loading)
3. **Parity Tests** (~4m): Ground truth comparison against PyTorch and llama.cpp
4. **Integration Tests** (~3m): Full pipeline tests with model loading
5. **Batch Correctness Tests**: Validation that batch and sequential execution produce identical results

**Key Testing Infrastructure:**
- `src/ParityHooks.{h,cpp}`: Snapshot capture hooks for parity testing
- `src/PipelineSnapshotManager.{h,cpp}`: Pipeline-level snapshot management
- `tests/TestParityFramework.cpp`: PyTorch ground truth comparison tests
- `tests/test_batch_correctness.cpp`: Batch vs sequential parity validation
- **Comprehensive documentation**: See `.github/instructions/parity-test-framework.instructions.md`

### Running Tests

```bash
# All tests
ctest --test-dir build --output-on-failure --parallel

# Verbose output
ctest --test-dir build --output-on-failure --verbose --parallel

# Test status summary
ctest --test-dir build --output-on-failure --parallel | tail -20

# Smoke Tests (1.16s)
ctest --test-dir build --output-on-failure --parallel \
  -R "^(BasicTest|NumaTest|ModelLoaderGoldenTest|PipelineFactoryTest|DequantTest|TPPartitionSpecTest|LargeMatmulPlanTest|WeightRoleClassification|MPILinearKernelTest|MPIRMSNormKernelTest|MPIAttentionKernelTest|MPISoftmaxCorrectnessTest|RMSNormCoreCorrectness|SoftmaxCoreCorrectness|LinearOrientationCorrectnessTest)$"

# Unit Tests (2m30s)
ctest --test-dir build --output-on-failure --parallel \
  -E "(Integration|ParityFrameworkTest|Incremental|Qwen|Prefill|.*Stress.*)"

# Parity Integration (220s) (Long running, verbose test suite - use `tee` and grep details from the logfile)
ctest --test-dir build --output-on-failure --verbose \
  -R "(ParityFrameworkTest|AbstractPipelineParity)" 2>&1 | tee test_output.log | tail -150

# Parity Integration (220s) with a GTEST filter to target only PyTorch parity tests in the suite:
GTEST_FILTER="ParityFramework.COSMAPrefillVsPyTorch:ParityFramework.OpenBLASPrefillVsPyTorch:ParityFramework.MKLPrefillVsPyTorch:ParityFramework.TrueIncrementalDecodeVsPyTorch" ctest --test-dir build --output-on-failure --verbose -R "ParityFrameworkTest" 2>&1 | tee test_output.log | tail -150

# Integration Tests (3m0s)
ctest --test-dir build --output-on-failure --verbose \
  -R "(Integration|Incremental|Qwen|Prefill|End2End|KVCache)"
```

### MPI Testing

```bash
# Run with 2 MPI processes (optimal for 2-socket systems)
mpirun -np 2 ./build/llaminar

# Run specific tests with MPI
mpirun -np 2 ./build/test_cosma

# DEBUGGING: Capture the backtrace of a segfault in MPI with gdb
bash -lc 'set -m; mpirun -np 2 gdb -q --batch -ex "handle SIGUSR1 pass nostop noprint" -ex run -ex bt -ex bt full --args ./build/test_incremental_decode_parity --gtest_filter=IncrementalDecodeParity.ReplayVsIncrementalMultiRank 2>&1 | tee gdb_mpi_bt.log'
```

### Test Categories

- **BasicTest**: MPI initialization and basic functionality
- **NumaTest**: NUMA topology detection and affinity
- **PipelineFactoryTest**: Pipeline factory registration and creation
- **QwenPipelineTest**: Qwen pipeline functionality
- **BatchCorrectnessTest**: Batch vs sequential execution parity (8/8 attention stages passing)
- **AbstractPipelineParity**: Prefill vs incremental decode equivalence
- **ParityFrameworkTest**: PyTorch ground truth comparison (OpenBLAS, COSMA, incremental decode)
- **CosmaPrefillTests**: COSMA distributed prefill correctness
- **AdaptiveMatmulTests**: Backend decision logic validation
- **MPIOperatorTests**: MPI operator correctness (Linear, Attention, RMSNorm, Embedding, RoPE, SwiGLU)
- **AttentionTests**: Attention mechanism validation (bias, sharding, output modes)
- **RMSNormTests**: RMSNorm parity and edge cases
- **KVCacheTests**: KV cache capacity and state management
- **TPTests**: Tensor partition correctness
- **WeightTests**: Weight loading, role classification, and verification

**Removed Historical Tests:**
- ❌ `test_graph.cpp`: Generic compute graph (architecture removed)
- ❌ `LinearKernelTest`: Legacy non-MPI kernel (retired after MPI migration)


## Checking Files for Compile Errors / Problems

It is possible to use the `problems` tool against a particular filename in order to get a list of problems (compile errors) in the file. Do this after editing a file but before building, as the feedback loop is faster.

## Debugging with GDB

### Basic GDB Setup

```bash
# Compile with debug symbols
cmake -B build -S . -DCMAKE_BUILD_TYPE=Debug

# Single process debugging
gdb ./build/llaminar
(gdb) run --verbose --print-topology

# MPI debugging (attach to specific rank)
mpirun -np 2 xterm -e gdb ./build/llaminar
```

### Debugging Segfaults

```bash
# Run with core dumps enabled
ulimit -c unlimited
mpirun -np 2 ./build/llaminar

# Analyze core dump
gdb ./build/llaminar core
(gdb) bt         # Get backtrace
(gdb) bt full    # Get full backtrace with variable values
(gdb) info registers
(gdb) x/10i $pc  # Examine instructions around crash
```

Use ASAN for localizing more complex double-free style issues:

```bash
cmake -B build -S . -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS="-fsanitize=address -fno-omit-frame-pointer -g" -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

ASAN_OPTIONS=halt_on_error=0:detect_leaks=0 timeout 240 mpirun -np 2 ./build/<your_test> --gtest_filter=<your_test_filter>
```

Just don't forget to reconfigure cmake to disable ASAN when you're done debugging!

### MPI-Specific Debugging

```bash
# Debug specific MPI rank
mpirun -np 2 -host localhost:1 gdb ./build/llaminar : -host localhost:1 ./build/llaminar

# Use GDB with MPI (tmux/screen recommended)
mpirun -np 2 xterm -hold -e gdb -ex run --args ./build/llaminar --verbose

# Debugging hanging MPI programs
# In separate terminal:
ps aux | grep llaminar
gdb -p <PID>
(gdb) bt    # Get backtrace of hanging process
```

### Common Debugging Scenarios

```bash
# Memory issues (use with caution - slows execution significantly)
mpirun -np 2 valgrind --tool=memcheck --leak-check=full ./build/llaminar

# NUMA binding issues
numactl --hardware
numactl --show
mpirun -np 2 numactl --cpubind=0 ./build/llaminar : numactl --cpubind=1 ./build/llaminar

# COSMA hanging issues (use timeouts)
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

### Performance Baselines vs llama.cpp

**Benchmark Configuration** (October 2025):
- Model: Qwen 2.5 0.5B Instruct Q8_0 (638 MB)
- Hardware: 2-socket system (56 physical cores, 112 with HT)
- llama.cpp: Release build with `-march=native`, 28 threads (physical cores only)
- Llaminar: Release build with MPI (2 ranks), OpenBLAS, 28 threads per rank

**llama.cpp Performance Baseline (pp512 - 512 token prompts):**

| Batch Size | Throughput | Speedup vs Batch=1 | Notes |
|------------|------------|---------------------|-------|
| 1 | 25.2 tok/s | 1.0× | Single sequence baseline |
| 8 | 120.5 tok/s | 4.8× | |
| 16 | 249.7 tok/s | 9.9× | ~10× scaling milestone |
| 32 | 643.4 tok/s | 25.5× | **Exceeds 22× target** |
| 64 | 854.3 tok/s | 33.9× | |
| 128 | 1065.3 tok/s | 42.3× | Strong scaling continues |
| 256 | 1148.2 tok/s | 45.5× | Approaching plateau |
| 512 | 1210.1 tok/s | 48.0× | **Peak performance** |

**Key Findings:**
- **Short sequences (8 tokens) severely limit batch scaling**: Only ~6× speedup at batch=32
- **Long sequences (512 tokens) unlock true batch potential**: 48× speedup at batch=512
- **Performance plateaus around batch=256-512**: ~1150-1210 tok/s
- **The 22× target is achievable**: Requires longer sequences (256+ tokens) and batch sizes ≥32

**Reproducing llama.cpp Baseline:**

```bash
# Build llama.cpp with optimizations
cd llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS="-march=native" \
  -DCMAKE_C_FLAGS="-march=native" \
  -DGGML_NATIVE=ON -DGGML_OPENMP=ON
cmake --build build --target llama-bench --parallel

# Run benchmark (pp512 = 512 token prompt, n=0 means no generation)
OMP_NUM_THREADS=28 OMP_PLACES=cores OMP_PROC_BIND=close \
  ./build/bin/llama-bench \
  -m ../models/qwen2.5-0.5b-instruct-q8_0.gguf \
  -p 512 -n 0 -b 1,2,4,8,16,32,64,128,256,512 -t 28
```

**Llaminar Performance Goals:**
- **Target**: Match or exceed llama.cpp's 1210 tok/s at batch=512, pp512
- **Current Status** (as of Oct 2025):
  - Short sequences (8 tokens): 415 tok/s @ batch=32 (3× speedup) - limited by sequence length
  - Long sequences (512 tokens): **To be benchmarked**
- **Advantages**: Llaminar shows 3.5-4× better single-sequence performance (136 vs 39 tok/s with 8 tokens)
- **Next Steps**: Extend batch performance tests to 128-512 token sequences to measure true scaling potential

**Running Llaminar Batch Performance Tests:**

```bash
# Build Release version
cmake -B build_release -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build_release --target test_batch_performance --parallel

# Run with proper environment configuration
./run_batch_performance.sh

# Or run specific test filters
./run_batch_performance.sh --filter '*PrefillThroughputScaling'
```

## COSMA vs OpenBLAS Integration (V1)

**Note**: This section applies to V1 architecture. V2 backend selection works differently (per-tensor device affinity).

### When to Use Each Backend

Based on our empirical performance testing:

```cpp
// OpenBLAS dominates for small operations
// - Single token generation: 1x896x896 - OpenBLAS 134x faster
// - Small batches: <64 tokens - OpenBLAS wins by large margins
// - Communication overhead dominates COSMA performance

// COSMA becomes competitive at scale
// - Prefill operations: ≥8K tokens - COSMA starts showing advantages
// - Very large operations: ≥64K tokens - COSMA can be 3.6x faster
// - Memory-bound operations benefit from COSMA's optimized layout
```

### Backend Selection Logic

```cpp
MatMulBackend selectOptimalBackend(int m, int n, int k, bool is_prefill) {
    size_t total_elements = static_cast<size_t>(m) * n * k;
    
    // Small operations: single-threaded for minimal overhead
    if (total_elements < 8192) {
        return MatMulBackend::SINGLE_THREADED_OPENBLAS;
    }
    
    // Large prefill: consider distributed if beneficial
    if (is_prefill && m >= 8192 && total_elements >= 8388608) {
        return MatMulBackend::DISTRIBUTED_OPENBLAS;  // or COSMA in future
    }
    
    // Medium operations: multi-threaded local
    return MatMulBackend::MULTI_THREADED_OPENBLAS;
}
```

### Integration Patterns

```cpp
// Hybrid execution with fallback
bool execute_with_fallback(const TensorInputs& inputs) {
    try {
        // Try optimal backend first
        if (should_use_cosma(inputs)) {
            return execute_cosma_path(inputs);
        }
    } catch (const std::exception& e) {
        LOG_WARN("COSMA execution failed, falling back to OpenBLAS: " << e.what());
    }
    
    // Fallback to reliable OpenBLAS
    return execute_openblas_path(inputs);
}

// BF16 quantized path with MKL default (OpenBLAS verified working)
bool execute_bf16_matmul(const float* A, const bfloat16_t* B, float* C, int m, int n, int k) {
    #ifdef HAVE_MKL
        // Try Intel MKL cblas_sbgemm (default when MKL available - hardware accelerated on Ice Lake+)
        try {
            return MKLBackend::multiply_bf16(A, B, C, m, n, k);
        } catch (const std::exception& e) {
            LOG_WARN("MKL BF16 failed: " << e.what() << ", trying OpenBLAS");
        }
    #endif
    
    // Fallback to OpenBLAS BF16 (verified working in v0.3.26 - software emulation reliable)
    if (debugEnv().quant.bf16_prefer_mkl == 0) {
        try {
            return OpenBLASBackend::multiply_bf16(A, B, C, m, n, k);
        } catch (const std::exception& e) {
            LOG_WARN("OpenBLAS BF16 failed: " << e.what() << ", expanding to FP32");
        }
    }
    
    // Final fallback: FP32 expansion (rarely needed - both backends work)
    return multiply_fp32_fallback(A, B, C, m, n, k);
}
```

## Parity Testing Framework

Llaminar includes a comprehensive **parity testing framework** that validates correctness by comparing against ground truth implementations (PyTorch, llama.cpp) and between execution modes (batch vs sequential, prefill vs decode).

### Framework Components

**Infrastructure:**
- `src/ParityHooks.{h,cpp}`: Hooks for capturing intermediate activations (snapshots)
- `src/PipelineSnapshotManager.{h,cpp}`: Pipeline-level snapshot orchestration
- `src/PipelineStages.h`: Enumeration of all capturable stages (18 stages per layer)
- `tests/ParityTestFramework.{h,cpp}`: Test utilities and comparison framework
- `tests/WeightVerifier.h`: Weight loading verification against ground truth

**Test Suites:**
- `tests/TestParityFramework.cpp`: PyTorch ground truth comparison (prefill, decode)
- `tests/test_batch_correctness.cpp`: Batch vs sequential validation
- Various operator-level parity tests in `tests/`

### Snapshot Capture System

**18 Pipeline Stages per Layer:**
1. `EMBEDDING`: Token embedding lookup
2. `ATTENTION_NORM`: Pre-attention RMSNorm
3. `Q_PROJECTION`: Query projection
4. `K_PROJECTION`: Key projection
5. `V_PROJECTION`: Value projection
6. `ROPE_APPLICATION`: Rotary position embeddings applied
7. `ATTENTION_SCORES`: Q·K^T scores
8. `ATTENTION_WEIGHTS`: Softmax attention weights
9. `ATTENTION_CONTEXT`: Weighted sum of values
10. `ATTENTION_OUTPUT`: Output projection
11. `ATTENTION_RESIDUAL`: Post-attention residual connection
12. `FFN_NORM`: Pre-FFN RMSNorm
13. `FFN_GATE`: FFN gate projection
14. `FFN_UP`: FFN up projection
15. `FFN_SWIGLU`: SwiGLU activation output
16. `FFN_DOWN`: FFN down projection
17. `FFN_RESIDUAL`: Post-FFN residual connection
18. `FINAL_NORM`: Final RMSNorm before LM head
19. `LM_HEAD`: Logits over vocabulary

**Usage Example:**
```cpp
// In pipeline code:
captureSnapshot(PipelineStage::Q_PROJECTION, layer_idx, data_ptr, seq_len, d_model);

// In test code:
SnapshotRegistry& registry = SnapshotRegistry::instance();
TensorSnapshot snapshot;
std::string key = registry.make_key("OpenBLAS", "Q_PROJECTION", 0);
if (registry.get_snapshot(key, snapshot)) {
    // Compare against ground truth
    auto result = SnapshotComparator::compare(snapshot, ground_truth, tolerance);
    EXPECT_TRUE(result.passed());
}
```

### Parity Test Types

#### 1. PyTorch Parity (`ParityFrameworkTest`)

**Purpose**: Validate Llaminar produces identical outputs to PyTorch reference implementation

**Tests:**
- `COSMAPrefillVsPyTorch`: COSMA backend vs PyTorch (prefill phase)
- `OpenBLASPrefillVsPyTorch`: OpenBLAS backend vs PyTorch (prefill phase)
- `MKLPrefillVsPyTorch`: Intel MKL BF16 backend vs PyTorch (prefill phase)
- `TrueIncrementalDecodeVsPyTorch`: Incremental decode vs PyTorch (autoregressive)

**Status**: ✅ All passing with <0.1% relative error

**Run:**
```bash
# All PyTorch parity tests
ctest --test-dir build -R "ParityFrameworkTest" --output-on-failure --verbose

# Specific backend
GTEST_FILTER="ParityFramework.OpenBLASPrefillVsPyTorch" \
  ctest --test-dir build -R "ParityFrameworkTest" --verbose
```

#### 2. Batch vs Sequential Parity (`BatchCorrectnessTest`)

**Purpose**: Validate batch and sequential pipelines produce identical results

**Tests:**
- `BatchedAttentionStagesParity`: ✅ **8/8 attention stages passing** (exact matches)
- `PrefillBatchVsSequential`: 🔄 In progress (extending to FFN/LM head)

**Status**: Attention mechanism fully validated, FFN stages under investigation

**Run:**
```bash
mpirun -np 2 ./build/test_batch_correctness \
  --gtest_filter="BatchCorrectnessTest.BatchedAttentionStagesParity"
```

#### 3. Prefill vs Decode Parity (`AbstractPipelineParity`)

**Purpose**: Validate incremental decode produces same results as prefill

**Status**: ✅ Passing - proves KV cache and incremental attention correctness

#### 4. Weight Verification

**Purpose**: Validate weights loaded from GGUF match PyTorch checkpoint

**Coverage:**
- Embedding table: Token embeddings
- Layer weights: Q/K/V/O projections, FFN gate/up/down, RMSNorm gamma
- LM head: Output projection weights

**Verification:**
```cpp
WeightVerifier verifier("path/to/pytorch_weights.npz");
auto result = verifier.verifyEmbedding(model_loader.getEmbeddingTable());
EXPECT_TRUE(result.passed()) << result.summary();
```

### Debugging with Snapshots

**When a parity test fails:**

1. **Identify divergence point**: Framework reports first mismatching stage
   ```
   ✓ Q_PROJECTION layer 0 (max_diff=0)
   ✗ ATTENTION_SCORES layer 0 FAILED (max_diff=0.523)
   ```

2. **Inspect tensors**: Snapshots contain shape, data, metadata
   ```cpp
   std::cout << "Shape: " << snapshot.seq_len << " × " << snapshot.feature_dim << std::endl;
   std::cout << "First values: " << snapshot.data[0] << ", " << snapshot.data[1] << std::endl;
   ```

3. **Compare metrics**: Relative L2, max absolute difference, mismatch count
   ```cpp
   std::cout << "Relative L2: " << result.metrics.rel_l2 << std::endl;
   std::cout << "Max abs diff: " << result.metrics.max_abs_diff << std::endl;
   std::cout << "Mismatches: " << result.metrics.num_mismatches << "/" << total << std::endl;
   ```

4. **Isolate bug**: Stage-by-stage validation narrows down to specific operation

### Best Practices

1. **Always add snapshots** when implementing new pipeline stages
2. **Run parity tests** before merging pipeline changes
3. **Use strict tolerances** for early stages (embedding, projections): `1e-4`
4. **Use relaxed tolerances** for late stages (accumulated error): `1e-3`
5. **Capture on rank 0 only** to avoid MPI synchronization overhead
6. **Clear registry** between test cases to avoid stale snapshots

### Documentation

For comprehensive guide including API reference, debugging workflows, and advanced usage:
- **`.github/instructions/parity-test-framework.instructions.md`** (2600+ lines)

For architectural overview and batch parity testing:
- **`.github/instructions/llaminar-architecture.instructions.md`**

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