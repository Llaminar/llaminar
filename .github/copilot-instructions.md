# Llaminar V2 Project Development Guidelines

This document provides practical guidelines for working with the **Llaminar V2** LLM inference engine, including build processes, testing, debugging, and kernel / MPI / attention development best practices.

**Architecture Note (V2)**: The active architecture is **Llaminar V2** in `src/v2/`, an operator-free, kernel-centric design with **GraphOrchestrator** as the sole execution path.

- For a **high-level architecture map** of tensors, kernels, attention, MPI orchestration, and graph execution, see:
    - `.github/instructions/llaminar-architecture-v2.instructions.md`
- For additional V2-specific implementation details, see:
    - `.github/instructions/llaminar-v2-architecture.instructions.md`

## Table of Contents
- [Architecture Overview](#architecture-overview)
- [Build System](#build-system)
- [Canonical Runtime Configuration](#canonical-runtime-configuration)
- [Benchmark Mode](#benchmark-mode)
- [Testing Guidelines](#testing-guidelines)
- [Debugging](#debugging)
- [Snapshot Framework and E2E Testing](#snapshot-framework-and-e2e-testing)
- [Kernel Development](#kernel-development)
- [Weight Sharding and Tensor Parallelism](#weight-sharding-and-tensor-parallelism)
- [MPI Development Best Practices](#mpi-development-best-practices)
- [Performance Optimization](#performance-optimization)
- [Code Quality Guidelines](#code-quality-guidelines)
- [Environment Variables Reference](#environment-variables-reference)
- [Documentation and Project Resources](#documentation-and-project-resources)

---

## Architecture Overview

**Design Philosophy**: Operator-free, tensor-centric, kernel-oriented design with declarative graph execution.

**Key Characteristics**:
- **GraphOrchestrator**: Single execution path via declarative compute DAGs
- **No Operator Layer**: Graph stages orchestrate kernels directly
- **Per-Tensor Device Affinity**: Each tensor knows its device placement
- **Heterogeneous Execution**: Designed to mix CPU, CUDA, ROCm via device backends
- **Centralized Kernel Dispatch**: `KernelFactory` provides unified kernel creation with caching
- **Automatic Weight Sharding**: Megatron-style tensor parallelism across MPI ranks

**Execution Flow**:
> **GraphOrchestrator** → builds **ComputeGraph** → **GraphExecutor** runs **ComputeStages** → stages use **KernelFactory** to get kernels → kernels operate on local buffers. MPI synchronization via **AllreduceStage** / **AllGatherStage**.

See `.github/instructions/llaminar-architecture-v2.instructions.md` for a full-stack walkthrough.

---

## Build System

### Build Commands

```bash
# Debug build (for development)
cmake -B build_v2 -S src/v2 -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build_v2 --parallel

# Release build (for performance)
cmake -B build_v2_release -S src/v2 -DCMAKE_BUILD_TYPE=Release
cmake --build build_v2_release --parallel

# E2E Release build (for parity testing with snapshots)
cmake -B build_v2_e2e_release -S src/v2 -DCMAKE_BUILD_TYPE=Release -DENABLE_PIPELINE_SNAPSHOTS=ON
cmake --build build_v2_e2e_release --parallel
```

**Build Targets**:
- `llaminar2_core`: Core library (linked by tests and tools)
- `llaminar2`: Main executable

**CMake Options**:
| Option | Default | Description |
|--------|---------|-------------|
| `HAVE_CUDA` | OFF | Enable CUDA backend |
| `HAVE_ROCM` | OFF | Enable ROCm backend |
| `ENABLE_PIPELINE_SNAPSHOTS` | OFF | Enable tensor snapshot capture for E2E tests |

---

## Canonical Runtime Configuration

### Running Llaminar

The `llaminar2` executable automatically bootstraps MPI and configures the runtime environment:

```bash
# Standard inference (auto-detects topology, self-launches MPI)
./build_v2_release/llaminar2 -m models/qwen2.5-0.5b-instruct-q4_0.gguf -p "Hello, world!" -n 50

# Debug logging
LLAMINAR_LOG_LEVEL=DEBUG ./build_v2/llaminar2 -m models/qwen2.5-0.5b-instruct-q4_0.gguf -p "Hello, world!" -n 50

# Specify MPI process count (e.g., 2 ranks for tensor parallelism)
./build_v2_release/llaminar2 --mpi-procs 2 -m models/qwen2.5-7b-instruct-q4_0.gguf -p "Hello!" -n 50

# Dry-run to preview configuration without execution
./build_v2_release/llaminar2 --dry-run -m models/qwen2.5-0.5b-instruct-q4_0.gguf

# Disable MPI bootstrap (single-rank only)
./build_v2_release/llaminar2 --no-mpi-bootstrap -m model.gguf -p "Hello"
```

The executable automatically configures:
- **CPU Topology Detection**: Parses `/proc/cpuinfo` to detect sockets, cores, NUMA nodes
- **OpenMP**: `OMP_NUM_THREADS` (cores/socket), `OMP_PLACES=sockets`, `OMP_PROC_BIND=close`
- **MPI**: `OMPI_MCA_mpi_leave_pinned=1`, socket binding, process mapping
- **BLAS**: `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`

### MPI Bootstrap Options

| Option | Description |
|--------|-------------|
| `--mpi-procs N` | Number of MPI processes (default: auto-detect from topology) |
| `--hostfile PATH` | MPI hostfile for multi-machine setup (OpenMPI format) |
| `--oversubscribe` | Allow more MPI ranks than available CPU slots |
| `--dry-run` | Show configuration without executing |
| `--mpi-verbose` | Verbose MPI launch output |
| `--no-mpi-bootstrap` | Disable auto-bootstrap, run single-rank |

---

## Benchmark Mode

### Running Benchmarks

```bash
# Standard benchmark with auto-generated prompt
./build_v2_release/llaminar2 --benchmark -m model.gguf

# Custom prompt and decode length
./build_v2_release/llaminar2 --benchmark -m model.gguf -p "Your prompt here" -n 100

# With kernel profiling
LLAMINAR_PROFILE_KERNELS=1 ./build_v2_release/llaminar2 --benchmark -m model.gguf -n 50
```

**Features**:
- 1 warmup run + 3 benchmark runs averaged
- Separate prefill/decode timing
- Greedy sampling for reproducibility
- KV cache cleared between runs

**Example Output**:
```
+--------------------------------------------------------------+
|                    BENCHMARK RESULTS                         |
|              (average of 3 runs after warmup)                |
+--------------------------------------------------------------+
| PREFILL PHASE                                                |
|   Tokens:           596 tokens                               |
|   Throughput:    304.19 tok/s                                |
+--------------------------------------------------------------+
| DECODE PHASE                                                 |
|   Tokens:           128 tokens                               |
|   Throughput:     54.00 tok/s                                |
+--------------------------------------------------------------+
```

### Kernel Profiling

Enable per-kernel timing breakdown:

```bash
LLAMINAR_PROFILE_KERNELS=1 ./build_v2_release/llaminar2 --benchmark -m model.gguf -n 50
```

**Profiled Operations**: `GEMM_Q8`, `ATTENTION`, `FFN_DOWN`, `FFN_GATE`, `FFN_UP`, `LM_HEAD`, `QUANTIZE_Q8`, `RMS_NORM`, `SWIGLU`, `ROPE`, `RESIDUAL_ADD`, `EMBEDDING`

---

## Testing Guidelines

### Test Organization

| Category | Location | Purpose |
|----------|----------|---------|
| Unit | `tests/v2/unit/` | Fast, isolated component tests (no model loading) |
| Integration | `tests/v2/integration/` | Full pipeline tests with models |
| E2E | `tests/v2/e2e/` | Ground truth parity vs PyTorch reference |
| Performance | `tests/v2/performance/` | Benchmark tests |

### Test File Naming Convention

- Files: `Test__ClassName.cpp` (CamelCase with double underscore)
- Suite: `TEST(Test__ClassName, TestName)`
- Example: `Test__ModelLoader.cpp` → `TEST(Test__ModelLoader, LoadsQwen2)`

### Running Tests

```bash
# Unit tests (fast)
ctest --test-dir build_v2 -R "^V2_Unit_" --output-on-failure --parallel

# Integration tests (require models)
ctest --test-dir build_v2_release -R "^V2_Integration_" --output-on-failure

# E2E parity tests (require snapshot build)
ctest --test-dir build_v2_e2e_release -R "^V2_E2E_" --output-on-failure

# Performance benchmarks
ctest --test-dir build_v2_release -R "^V2_Perf_" --verbose
```

### CTest Labels

Tests use hierarchical labels for flexible filtering:
- **Tier 1 (Type)**: `Unit`, `Integration`, `E2E`, `Performance`
- **Tier 2 (Architecture)**: `V2`
- **Tier 3 (Component)**: `DeviceManagement`, `TensorOperations`, `Kernels`, `ModelLoading`
- **Tier 4 (Feature)**: `Quantization`, `GEMM`, `Attention`, `IQ4_NL`

```bash
ctest -L DeviceManagement          # All device tests
ctest -L "Quantization"            # All quantization tests
ctest -LE Performance              # Exclude performance tests
```

### TestTensorFactory Utility

Header-only utility for unit tests (`tests/v2/utils/TestTensorFactory.h`):

```cpp
#include "utils/TestTensorFactory.h"
using namespace llaminar2::test;

auto input = TestTensorFactory::createFP32Random({32, 896});
auto weights = TestTensorFactory::createQ8_0Random({1024, 896});
float mse = TestTensorFactory::computeMSE(a->data(), b->data(), count);
```

---

## Debugging

### GDB for Segfaults (MPI-aware)

```bash
# Create GDB command file
cat > /tmp/gdbcommands.txt << 'EOF'
set debuginfod enabled off
set pagination off
handle SIGSEGV stop print
run
thread apply all bt full
quit
EOF

# Set environment
export OMP_NUM_THREADS=28 OMP_PLACES=sockets OMP_PROC_BIND=close
export LLAMINAR_LOG_LEVEL=DEBUG

# Run under GDB
timeout 120 mpirun -np 2 \
  gdb -x /tmp/gdbcommands.txt --args \
  ./build_v2/tests/v2/v2_test_my_feature 2>&1 | tee gdb_output.log

# Analyze
grep -A 50 "Program received signal" gdb_output.log
```

### ASAN for Memory Issues

```bash
cmake -B build_v2_asan -S src/v2 -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS="-g3 -O0 -fno-omit-frame-pointer -fsanitize=address" \
  -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address"
cmake --build build_v2_asan --parallel

ASAN_OPTIONS=halt_on_error=0:detect_leaks=0 \
  timeout 240 mpirun -np 2 ./build_v2_asan/tests/v2/<your_test>
```

### Valgrind for Memory Leaks

```bash
mpirun -np 2 valgrind --tool=memcheck --leak-check=full ./build_v2/tests/v2/v2_test_my_feature
```

### Common Issues and Solutions

| Problem | Solution |
|---------|----------|
| MPI hangs | Use `MPI_Barrier` before/after collective operations |
| Numerical divergence | Run E2E parity tests, check layer-by-layer snapshots |
| Performance regression | Use `--benchmark` mode, verify Release build with `-march=native` |
| Memory allocation failures | Enable NUMA verification, check first-touch allocation |
| Race conditions | Use `OMP_WORKSHARE_REGION` macro, avoid shared mutable state |
| Inference produces garbage | Use `-t 0` (greedy sampling) to eliminate randomness |

---

## Snapshot Framework and E2E Testing

### Overview

The snapshot framework captures intermediate tensors during inference for comparison against PyTorch ground truth. This is essential for debugging numerical divergence.

**Key Concept**: Both Llaminar and PyTorch load the **same GGUF weights** (e.g., Q4_0). PyTorch dequantizes to FP32; Llaminar uses quantized GEMM. Minor divergence is expected; token predictions should match.

### Build Configuration

```bash
# E2E builds require ENABLE_PIPELINE_SNAPSHOTS
cmake -B build_v2_e2e_release -S src/v2 \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_PIPELINE_SNAPSHOTS=ON
cmake --build build_v2_e2e_release --parallel
```

### Snapshot Capture API

```cpp
// Enable snapshot capture before forward pass
IInferenceRunner* runner = createInferenceRunner(model_ctx, mpi_ctx);
runner->enableSnapshotCapture("/tmp/llaminar_snapshots");

// Run inference
runner->forward(token_ids.data(), seq_len);

// Retrieve snapshots
auto keys = runner->getSnapshotKeys();
for (const auto& key : keys) {
    size_t size;
    const float* data = runner->getSnapshot(key, size);
    // Compare with PyTorch reference...
}
```

### Snapshot Keys (Qwen2)

| Stage | Key Format |
|-------|------------|
| Embedding | `EMBEDDING` |
| Attention norm | `layer{N}_ATTENTION_NORM` |
| QKV projections | `layer{N}_Q_PROJECTION`, `layer{N}_K_PROJECTION`, `layer{N}_V_PROJECTION` |
| RoPE | `layer{N}_Q_ROPE`, `layer{N}_K_ROPE` |
| Attention output | `layer{N}_ATTENTION_CONTEXT`, `layer{N}_ATTENTION_OUTPUT` |
| FFN stages | `layer{N}_FFN_NORM`, `layer{N}_FFN_GATE`, `layer{N}_FFN_UP`, `layer{N}_FFN_SWIGLU`, `layer{N}_FFN_DOWN` |
| Residuals | `layer{N}_ATTENTION_RESIDUAL`, `layer{N}_FFN_RESIDUAL` |
| Final | `FINAL_NORM`, `LM_HEAD` |

### Tensor Dump for Debugging

Dump raw tensor data to disk for analysis:

```bash
# Dump specific layers and stages
LLAMINAR_SNAPSHOT_TENSOR_DUMP=1 \
LLAMINAR_SNAPSHOT_DUMP_LAYERS=3,4,5 \
LLAMINAR_SNAPSHOT_DUMP_STAGES=FFN_RESIDUAL,ATTENTION_RESIDUAL \
LLAMINAR_SNAPSHOT_DUMP_DIR=/tmp/layer_debug \
./build_v2_release/llaminar2 -m model.gguf -p "test prompt"
```

**Output Files**:
- `<stage>_rank<N>_fp32.bin` - FP32 dequantized data
- `<stage>_rank<N>_metadata.txt` - Shape and type info

### Analyzing Dumps

```python
import numpy as np

# Compare FP32 vs Q8_1 at specific element
dump_dir = "/tmp/layer_debug"
for layer in range(24):
    fp32 = np.fromfile(f"{dump_dir}/layer{layer}_FFN_RESIDUAL_rank0_fp32.bin", dtype=np.float32)
    q8_1 = np.fromfile(f"{dump_dir}/layer{layer}_FFN_RESIDUAL_rank0_q8_1.bin", dtype=np.float32)
    diff = np.max(np.abs(fp32 - q8_1))
    print(f"Layer {layer}: max diff = {diff:.6f}")
```

### PyTorch Reference Framework

Generate ground truth snapshots from GGUF:

```bash
# Generate snapshots
python python/reference/generate_qwen2_pipeline_snapshots.py \
    --model models/qwen2.5-0.5b-instruct-q4_0.gguf \
    --output pytorch_qwen2_snapshots \
    --decode-steps 5

# Run inference
python python/reference/run_reference.py \
    --model qwen \
    --checkpoint models/qwen2.5-0.5b-instruct-q4_0.gguf \
    --prompt "The quick brown fox" \
    --max-tokens 20
```

### Debugging Inference Issues

**Step 1**: Use greedy sampling to eliminate randomness:
```bash
./build_v2_release/llaminar2 -m model.gguf -p "prompt" -n 10 -t 0
```

**Step 2**: Compare top-5 predictions:
```bash
LLAMINAR_LOG_LEVEL=TRACE ./build_v2_release/llaminar2 -m model.gguf -p "prompt" -n 1 -t 0 2>&1 | grep "Top-5"
```

**Step 3**: Run E2E parity tests:
```bash
ctest --test-dir build_v2_e2e_release -R Qwen2FP32Parity -V
```

**Step 4**: Enable stage tracing for full visibility:
```bash
LLAMINAR_STAGE_DUMP=1 LLAMINAR_MPI_LOG_COLLECTIVES=1 \
./build_v2_release/llaminar2 --mpi-procs 2 -m model.gguf -p "test"
```

---

## Kernel Development

### Core Principles

1. **Tensor-centric Design**: Operations focused on Tensors
2. **Per-Tensor Device Affinity**: Tensors know their device placement
3. **Strategy Pattern**: Generic kernels + format-specific decode via `ITensorGemmTileDataProvider`
4. **ITensor Interfaces**: `ITensorGemm`, `ITensorAttention`, `ITensorRoPE`, etc.

### ITensorGemmTileDataProvider Strategy Pattern

Quantized tensors implement decode strategies; a single generic GEMM kernel works for all formats:

```cpp
class IQ4_NLTensor : public TensorBase, public ITensorGemmTileDataProvider {
    void decode_block_at(size_t row_idx, size_t k_block_offset, float* output) const override {
        const IQ4_NLBlock& block = blocks_[row_idx * blocks_per_row_ + k_block_offset];
        decodeBlock(block, output);  // Format-specific decode
    }
    
    size_t block_size() const override { return 32; }
    
    std::unique_ptr<ITensorGemm> createGemm(const MPIContext& mpi_ctx, int device_idx) const override {
        return std::make_unique<QuantizedGemmKernel>(this, mpi_ctx, device_idx);
    }
};
```

### KernelFactory

Centralized kernel creation with caching:

```cpp
// Preferred: cached kernel (pack once, use many)
ITensorGemm* gemm = KernelFactory::getOrCreateGemm(weight_tensor.get());
gemm->multiply(activations, output, m, n, k);

// Alternative: uncached (for one-off use)
auto gemm = KernelFactory::createGemm(dynamic_cast<IQ4_NLTensor*>(tensor), DeviceType::CPU);
```

### SIMD Guidelines

1. **Exploit ILP**: Unroll loops, interleaved loads/stores
2. **Vectorized Tail Handling**: AVX512 16-way → AVX2 8-way → AVX 4-way → SSE 2-way → scalar
3. **Prefetch**: Prefetch upcoming sequential reads

### OpenMP Nested-Safe Pattern

Use `OMP_WORKSHARE_REGION` macro to avoid fork/join overhead:

```cpp
#include "utils/OpenMPUtils.h"

void my_kernel(float* data, int n) {
    auto do_work = [&]() {
        #pragma omp for schedule(static)
        for (int i = 0; i < n; ++i) {
            data[i] = process(data[i]);
        }
    };
    OMP_WORKSHARE_REGION(do_work);
}
```

---

## Weight Sharding and Tensor Parallelism

### Overview

Weight sharding enables Megatron-style tensor parallelism by distributing weights across MPI ranks.

**Default Behavior** (December 2025):
- Single rank: No sharding
- Multiple ranks: Automatic sharding (equivalent to `--shard-weights`)

### Sharding Modes

| Weight | Mode | Description |
|--------|------|-------------|
| `attn_q`, `attn_k`, `attn_v` | COLUMN_PARALLEL | Split output dim (heads) |
| `attn_output` (Wo) | ROW_PARALLEL | Split input dim, allreduce after |
| `ffn_gate`, `ffn_up` | COLUMN_PARALLEL | Split output dim |
| `ffn_down` | INPUT_PARALLEL | Split input dim, allreduce after |
| `output` (LM head) | COLUMN_PARALLEL | Split vocab, allgather logits |
| Norms, embeddings | REPLICATE | Full copy on each rank |

### Memory Savings

With 2 MPI ranks: ~50% reduction for sharded weights, ~25-30% overall.

```bash
# Automatic sharding when world_size > 1
./build_v2_release/llaminar2 --mpi-procs 2 -m model.gguf -p "Hello"
```

---

## MPI Development Best Practices

### Critical Patterns

```cpp
// ALWAYS use barriers around collective operations
MPI_Barrier(MPI_COMM_WORLD);
cosma::multiply(A, B, C, strategy, MPI_COMM_WORLD, 1.0f, 0.0f);
MPI_Barrier(MPI_COMM_WORLD);

// Safe reduction pattern
float local_sum = compute_local_sum();
float global_sum;
MPI_Barrier(MPI_COMM_WORLD);
MPI_Allreduce(&local_sum, &global_sum, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
MPI_Barrier(MPI_COMM_WORLD);
```

### MPI Debugging

```bash
# Log all MPI collectives with timing
LLAMINAR_MPI_LOG_COLLECTIVES=1 LLAMINAR_MPI_LOG_TIMING=1 \
./build_v2_release/llaminar2 --mpi-procs 2 -m model.gguf -p "test"

# Enable checksum verification (slow)
LLAMINAR_MPI_VERIFY_CHECKSUMS=1 ./build_v2_release/llaminar2 --mpi-procs 2 -m model.gguf -p "test"
```

### Rank Comparison Testing

```bash
./scripts/compare_ranks.sh -m models/qwen2.5-0.5b-instruct-q4_0.gguf -p "Hello" -n 20
```

---

## Performance Optimization

### NUMA-Aware Allocation

All allocations >=128KB on hot paths MUST use NUMA first-touch initialization:

```cpp
// Parallel init for local NUMA placement
#pragma omp parallel for
for (size_t i = 0; i < buffer->numel(); ++i) {
    buffer->mutable_data()[i] = 0.0f;
}
```

**Impact**: +10-40% on multi-socket systems for large models (K/V cache access).

### Threading Strategy

```cpp
if (total_elements < 8192) {
    openblas_set_num_threads(1);  // Single-threaded
} else if (total_elements < 1048576) {
    openblas_set_num_threads(omp_get_max_threads());  // Multi-threaded
} else {
    // Use MPI distribution + local threading
}
```

---

## Code Quality Guidelines

### Logging Standards

```cpp
LOG_ERROR("Critical failures that prevent operation");
LOG_WARN("Concerning conditions");
LOG_INFO("Important runtime information");
LOG_DEBUG("Detailed debugging");
LOG_TRACE("Verbose execution tracing");
```

### Centralized Environment Access (debugEnv)

All hot-path code MUST use `debugEnv()` instead of `std::getenv`:

```cpp
// BEFORE (bad)
if (std::getenv("LLAMINAR_ATTN_MICRO_TRACE")) { ... }

// AFTER (good)
const auto& env = debugEnv();
if (env.attention.micro_trace) { ... }
```

### Testing New Features

```cpp
TEST(Test__MyNewKernel, BasicFunctionality) {
    auto input = TestTensorFactory::createFP32Random({32, 32});
    auto output = TestTensorFactory::createFP32({32, 32});
    
    auto kernel = KernelFactory::createMyKernel(input.get());
    ASSERT_TRUE(kernel->execute(output.get()));
    
    EXPECT_FALSE(TestTensorFactory::hasNaNOrInf(output.get()));
}
```

---

## Environment Variables Reference

| Variable | Description | Default |
|----------|-------------|---------|
| `LLAMINAR_LOG_LEVEL` | Logging verbosity (ERROR/WARN/INFO/DEBUG/TRACE) | INFO |
| `LLAMINAR_PROFILE_KERNELS` | Enable per-kernel timing in benchmark mode | Disabled |
| `LLAMINAR_EXECUTOR_PROFILING` | Enable per-stage profiling in GraphExecutor | Disabled |
| `LLAMINAR_EXECUTOR_VALIDATION` | Enable output validation after each stage | Disabled |
| `LLAMINAR_STAGE_DUMP` | Dump per-stage tensor outputs | Disabled |
| `LLAMINAR_DETERMINISTIC` | Force deterministic execution | Disabled |
| `LLAMINAR_SNAPSHOT_TENSOR_DUMP` | Enable raw tensor dump to disk | Disabled |
| `LLAMINAR_SNAPSHOT_DUMP_DIR` | Output directory for tensor dumps | `/tmp/llaminar_tensor_dumps` |
| `LLAMINAR_SNAPSHOT_DUMP_LAYERS` | Comma-separated layer indices to dump | `all` |
| `LLAMINAR_SNAPSHOT_DUMP_STAGES` | Comma-separated stage names to dump | `all` |
| `LLAMINAR_MPI_LOG_COLLECTIVES` | Log MPI collective operations | Disabled |
| `LLAMINAR_MPI_LOG_TIMING` | Log timing of MPI operations | Disabled |
| `LLAMINAR_MPI_VERIFY_CHECKSUMS` | Verify checksums before/after MPI ops | Disabled |
| `LLAMINAR_DEQUANT_STATS` | Log per-tensor dequant stats | Disabled |

For the full list, see `src/v2/utils/DebugEnv.h`.

---

## Documentation and Project Resources

### Key Documentation

| File | Purpose |
|------|---------|
| `.github/copilot-instructions.md` | This file - development guidelines |
| `.github/instructions/llaminar-architecture-v2.instructions.md` | High-level architecture map |
| `.github/instructions/llaminar-v2-architecture.instructions.md` | Detailed implementation notes |
| `docs/v2/DISTRIBUTED_ARCHITECTURE_IMPLEMENTATION.md` | MPI tensor parallelism design |
| `docs/v2/SNAPSHOT_FRAMEWORK_DESIGN.md` | Snapshot system documentation |

### Key Source Directories

| Directory | Purpose |
|-----------|---------|
| `src/v2/inference/` | IInferenceRunner interface and factory |
| `src/v2/execution/` | ComputeGraph, GraphExecutor, ComputeStages |
| `src/v2/pipelines/qwen/` | GraphOrchestrator, Qwen2Graph, buffer specs |
| `src/v2/kernels/cpu/` | CPU kernels (GEMM, attention, primitives) |
| `src/v2/tensors/` | Tensor types (FP32, BF16, quantized) |
| `src/v2/loaders/` | GGUF loading, WeightManager |
| `src/v2/utils/` | MPIContext, MPITopology, logging |

### Writing Changelogs

Place in `changelog/` with ISO date prefix:
```bash
# Example: changelog/2025-12-22-feature-description.md
```

Include: summary, code changes with paths, test results, performance metrics, next steps.

---

## Conclusion

This project emphasizes production reliability over peak theoretical performance:

1. **Empirical testing beats theoretical assumptions** - Always measure real performance
2. **MPI barriers are critical** - Prevent hangs with proper synchronization
3. **Adaptive approaches work** - Different backends for different problem sizes
4. **Graceful degradation** - Always have reliable fallback paths
5. **Threading is complex** - Start simple, optimize incrementally

When in doubt, prioritize correctness and reliability over raw performance.
