# Llaminar Project Development Guidelines

This document provides comprehensive guidelines for working with the Llaminar LLM inference engine, including build processes, debugging techniques, and kernel development best practices learned from our MPI and performance optimization work.

## Table of Contents
- [Project Overview](#project-overview)
- [Build System](#build-system)
- [Testing Guidelines](#testing-guidelines)
- [Debugging with GDB](#debugging-with-gdb)
- [Kernel Development](#kernel-development)
- [MPI Development Best Practices](#mpi-development-best-practices)
- [Performance Optimization](#performance-optimization)
- [COSMA vs OpenBLAS Integration](#cosma-vs-openblas-integration)
- [Code Quality Guidelines](#code-quality-guidelines)

## Project Overview

Llaminar is a high-performance, distributed LLM inference engine built on:
- **Multi-Architecture Pipeline System**: Factory-based extensible architecture for Qwen, LLaMA, and future models
- **Centralized Backend Selection**: Single decision point via MatMulBackendSelector for OpenBLAS vs COSMA
- **Hybrid Tensor System**: Zero-copy COSMA optimization with backward compatibility
- **MPI Distribution**: Multi-node inference with NUMA-aware deployment
- **Adaptive Backends**: OpenBLAS for small operations, COSMA for large-scale distributed computing
- **Comprehensive Observability**: Structured environment snapshot, performance counters, validation framework

### Key Architecture Components
- `src/main.cpp`: Application entry point with pipeline factory-based execution
- `src/abstract_pipeline.h`: Pipeline interface defining prefill/decode lifecycle
- `src/pipeline_factory.h`: Factory registration system for model architectures
- `src/qwen_pipeline.{h,cpp}`: Production Qwen model implementation
- `src/qwen_pipeline_adapter.{h,cpp}`: Qwen adapter implementing AbstractPipeline
- `src/llama_pipeline_adapter.{h,cpp}`: LLaMA adapter (prototype)
- `src/matmul_backend_selection.{h,cpp}`: Centralized backend decision logic
- `src/cosma_prefill_manager.{h,cpp}`: COSMA distributed prefill coordination
- `src/prefill_diagnostics.{h,cpp}`: Modular prefill validation and comparison
- `src/kernels/`: MPI-aware matrix operations, attention, and transformer components
- `src/tensors/`: Hybrid tensor system (SimpleTensor + COSMATensor)
- `src/debug_env.{h,cpp}`: Centralized environment snapshot (replaces scattered getenv)

## Build System

### Standard Build Process

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
```

### Available Build Targets

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
# Canonical way to run Llaminar
./run_llaminar.sh [arguments]

# Examples
./run_llaminar.sh -v --print-topology
./run_llaminar.sh -m models/qwen2.5-0.5b-instruct-q4_0.gguf -v
```

## Benchmark Mode

### Running Performance Benchmarks

Llaminar provides a dedicated `--benchmark` mode for clean performance measurement:

```bash
# Recommended: Use canonical launcher
./run_llaminar.sh --benchmark \
  -m models/qwen2.5-0.5b-instruct-q8_0.gguf \
  -p "Your prompt here" \
  -n 50

# Direct MPI execution (2 processes)
mpirun -np 2 --bind-to socket --map-by socket \
  ./build/llaminar --benchmark \
  -m models/qwen2.5-0.5b-instruct-q8_0.gguf \
  -p "Explain machine learning." \
  -n 100
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
- **Integration**: `src/Main.cpp` section 6, `src/ArgumentParser.{h,cpp}`
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
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel

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
```

### Performance Notes

- **Prefill scales with token count**: 2 tokens → 1.68 tok/s, 8 tokens → 6.58 tok/s, 512 tokens → 33.60 tok/s
- **Decode is consistent**: ~1.04 tok/s regardless of decode length (in Debug)
- **Debug vs Release**: Release builds expected to be 5-10x faster
- **Backend routing**: Small ops use OpenBLAS, large prefills may use COSMA (if threshold met)
- **NUMA optimization**: K/V cache and activations benefit from first-touch allocation (+10-40% on large models)

## Development Profiling (Advanced)

### Performance Bench Script for Developers

For detailed parallelization efficiency analysis, use the GTest-based profiling suite:

```bash
# Run all prefill performance benchmarks
./run_performance_bench.sh

# Run specific test suites
./run_performance_bench.sh --filter "OpenBLAS_StrongScaling*"
./run_performance_bench.sh --filter "COSMA_ModelShapes*"
```

**Important Distinction:**
- **`run_llaminar.sh --benchmark`**: Production inference benchmarking with real models
  - Use this for: End-to-end performance measurement, model comparisons, user-facing benchmarks
  - Outputs: Clean tok/s metrics for prefill/decode phases
  
- **`run_performance_bench.sh`**: Development profiling with GTest suite
  - Use this for: Analyzing parallelization efficiency, tuning threading strategies, backend selection optimization
  - Outputs: Detailed GTest metrics with efficiency percentages (>90% excellent, <50% poor)

The performance bench script runs the `test_prefill_performance_bench` executable with identical OpenMP/MPI configuration to the canonical launcher, but provides developer-focused metrics for optimization work.

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

## Testing Guidelines

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
GTEST_FILTER="ParityFramework.COSMAPrefillVsPyTorch:ParityFramework.OpenBLASPrefillVsPyTorch:ParityFramework.TrueIncrementalDecodeVsPyTorch" ctest --test-dir build --output-on-failure --verbose -R "ParityFrameworkTest" 2>&1 | tee test_output.log | tail -150

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
- **QwenPipelineTest**: Qwen pipeline functionality (4 test cases)
- **AbstractPipelineParity**: Prefill vs incremental decode equivalence
- **CosmaTest**: Matrix multiplication and COSMA integration (some precision edge cases)
- **CosmaPrefillTests**: Fused COSMA correctness and statistics
- **AdaptiveMatmulTests**: Backend decision logic validation
- **MPILinearKernelTest**: Distributed linear projection
- **AttentionTests**: Attention mechanism validation  
- **RMSNormTests**: RMSNorm parity and edge cases
- **KVCacheTests**: KV cache capacity management
- **TPTests**: Tensor partition correctness

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

## Kernel Development

### Kernel Base Classes

All kernels should inherit from appropriate base classes:
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

## COSMA vs OpenBLAS Integration

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

1. Add any new environment flag/knob to the structured snapshot in `DebugEnv.h` inside the appropriate group (or create one).
2. Parse it once in `debug_env.cpp` (lazy static initialization already provided) and expose typed fields.
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

Do NOT add another ad-hoc snapshot facility; extend the existing one. If grouping is unclear, prefer adding a new subgroup struct within `DebugEnv.h` rather than mixing unrelated flags.


## Documentation Standards
### Doxygen

All files and functions should be documented with the Doxygen format for readability and easy understanding of their purpose. For @author, use David Sanftenberg.


## Debug / Instrumentation Environment Variables

Use these to control verbosity, backend selection, and validation when working on kernels and distributed execution:

| Variable | Description | Default / Activation | Primary Scope |
|----------|-------------|----------------------|---------------|
| `LLAMINAR_DEQUANT_STATS` | Logs per-tensor dequant stats (min/max/mean/sample). | Disabled unless set to non-zero. | Quantized tensor loading |
| `LLAMINAR_DEQUANT_ANOMALIES` | Emits warnings for anomalous Q6_K (and future) values (NaN/Inf/huge). | Disabled unless set. | Dequant safety diagnostics |
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

# Writing Documentation and Changelogs

When we write documentation and changelogs at the end of our work runs, we place them in the `changelog/` folder in the workspace to avoid cluttering up the rest of the repo. We also use the `date` command to get the current date before naming changelog files.