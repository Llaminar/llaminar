# Llaminar V2 Project Development Guidelines

This document provides practical guidelines for working with the **Llaminar V2** LLM inference engine, including build processes, testing, debugging, and kernel / MPI / attention development best practices.

**Architecture Note (V2)**: The active architecture for all new development is **Llaminar V2** in `src/v2/`, an operator-free, kernel-centric design.

- For a **high-level architecture map** of tensors, kernels, attention, MPI orchestration, and pipelines, see:
    - `.github/instructions/llaminar-architecture-v2.instructions.md`
- For additional V2-specific implementation details and historical notes, see:
    - `.github/instructions/llaminar-v2-architecture.instructions.md`

## Table of Contents
- [Architecture Overview](#architecture-overview)
- [Build System](#build-system)
- [Testing Guidelines (V2)](#testing-guidelines-v2)
- [CTest Label Best Practices](#ctest-label-best-practices)
- [Debugging with GDB](#debugging-with-gdb)
- [Kernel Development (V2)](#kernel-development-v2)
- [KernelFactory: Centralized Kernel Dispatch and Caching](#kernelfactory-centralized-kernel-dispatch-and-caching)
- [Weight Sharding and Tensor Parallelism](#weight-sharding-and-tensor-parallelism)
- [MPI Development Best Practices](#mpi-development-best-practices)
- [Performance Optimization](#performance-optimization)
- [Code Quality Guidelines](#code-quality-guidelines)
- [Documentation Standards](#documentation-standards)
- [Pytorch Parity Testing](#pytorch-parity-testing)

## Architecture Overview

### V2 Architecture (`src/v2/`)

**Design Philosophy**: Operator-free, tensor-centric, kernel-oriented design.

**Key Characteristics**:
- 🎯 **No Operator Layer**: Pipelines orchestrate kernels directly (no `MPILinearOperator`, etc.).
- 🎯 **Per-Tensor Device Affinity**: Each tensor knows its device placement and how to create appropriate kernels.
- 🎯 **Heterogeneous Execution**: Designed to mix CPU, CUDA, ROCm, etc. within a single run via device backends.
- 🎯 **IActivationTensor / ITensor* Pattern**: Activation tensors expose narrow interfaces (`ITensorGemm`, `ITensorAttention`, `ITensorRoPE`, etc.) and kernels mutate activation buffers in-place.
- 🎯 **Centralized Kernel Dispatch**: `KernelFactory` provides unified kernel creation with caching and automatic lifecycle management.
- 🎯 **Weight Sharding**: Automatic tensor parallelism via `--shard-weights` flag distributes weights across MPI ranks.

The high-level flow is:

> **Pipeline** → chooses **devices** → allocates **tensors** → `KernelFactory` creates/caches **kernels** → kernels operate on local buffers. Any MPI/multi-rank work is coordinated by small **orchestrators**, not kernels.

See `.github/instructions/llaminar-architecture-v2.instructions.md` for a full-stack walkthrough.

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
- `llaminar2_core`: Core V2 library (linked by tests and tools)
- `llaminar2`: V2 executable (minimal functionality / device listing)

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
./run_llaminar.sh -- --benchmark \
  -m models/qwen2.5-0.5b-instruct-q8_0.gguf \
  -p "Your prompt here" \
  -n 50
```

### Benchmark Features

- **Warmup + averaged runs** - 1 warmup run (discarded), then 3 benchmark runs averaged
- **Separate prefill/decode timing** - Independent measurement of both phases
- **Clean output** - Minimal logging (ERROR level only) with formatted metrics
- **Greedy sampling** - Deterministic token selection for reproducible results
- **MPI-aware** - Handles tokenization on rank 0 with proper broadcast to all ranks
- **KV cache reset** - Cache cleared between runs for consistent measurements
- **Professional formatting** - Box-drawing characters for clear metric display

### Benchmark Output Metrics

```
╔══════════════════════════════════════════════════════════════╗
║                    BENCHMARK RESULTS                         ║
║              (average of 3 runs after warmup)                 ║
╠══════════════════════════════════════════════════════════════╣
║ PREFILL PHASE                                                ║
║   Tokens:           596 tokens                              ║
║   Time:         1959.28 ms                                  ║
║   Throughput:    304.19 tok/s                               ║
╠══════════════════════════════════════════════════════════════╣
║ DECODE PHASE                                                 ║
║   Tokens:           128 tokens                              ║
║   Time:         2370.41 ms                                  ║
║   Throughput:     54.00 tok/s                               ║
╠══════════════════════════════════════════════════════════════╣
║ TOTAL                                                        ║
║   Time:         4329.69 ms                                  ║
║   Overall:       167.22 tok/s (avg)                         ║
╚══════════════════════════════════════════════════════════════╝

✓ Benchmark completed successfully.
```

### Implementation Details

- **Source**: `src/v2/utils/BenchmarkRunner.{h,cpp}` (~460 lines)
- **Integration**: `src/v2/Main.cpp`, `src/v2/utils/ArgParser.{h,cpp}`
- **Key patterns**:
  - 1 warmup run + 3 benchmark runs averaged
  - `pipeline->clear_cache()` called before each run
  - Logits fetched via `pipeline->logits()` after prefill/decode
  - Token broadcast from rank 0 to all ranks via MPI_Bcast
  - Greedy sampling: argmax over last logits row (deterministic)
### Benchmark Defaults and Configuration

**Standard Benchmark:**
```bash
# Uses intelligent defaults if -p not provided:
#   - Auto-generated ~512 token prompt (mixed technical/narrative), tokenizes to ~596 tokens
#   - 128 decode tokens
#   - 1 warmup run + 3 benchmark runs averaged
./run_llaminar.sh -- --benchmark -m model.gguf
``` - 128 decode tokens
./run_llaminar.sh -- --benchmark -m model.gguf
```

**Phase-Specific Benchmarks:**
```bash
# Prefill-only (decode skipped when -n 0)
./run_llaminar.sh -- --benchmark -m model.gguf -p "Long context..." -n 0

# Custom prompt with specific decode count
./run_llaminar.sh -- --benchmark -m model.gguf -p "Hello world" -n 50
```

**Implementation Notes:**
- Auto-prompt generation in `src/v2/utils/BenchmarkRunner.cpp`
- Conditional execution:
  - Decode: Skipped if `n_predict == 0`
- Output shows "(SKIPPED)" for omitted phases

### Benchmarking Best Practices

```bash
# 1. Use Release builds for accurate timing
cmake -B build_v2_release -S src/v2 -DCMAKE_BUILD_TYPE=Release
cmake --build build_v2_release --parallel

# 2. Disable snapshotting
unset LLAMINAR_SNAPSHOT_STAGES

# 3. Test various prompt lengths (prefill scales with tokens)
./run_llaminar.sh -- --benchmark -m model.gguf -p "Short" -n 50
./run_llaminar.sh -- --benchmark -m model.gguf -p "Much longer prompt..." -n 50

# 4. Vary decode length to measure sustained performance
./run_llaminar.sh -- --benchmark -m model.gguf -p "Test" -n 20   # Quick
./run_llaminar.sh -- --benchmark -m model.gguf -p "Test" -n 200  # Sustained

# 5. Use phase-specific modes to isolate performance
./run_llaminar.sh -- --benchmark -m model.gguf -n 0          # Prefill-only (with default ~512 token prompt)
```

### Performance Notes

- **Prefill scales with token count**: 2 tokens → 1.68 tok/s, 8 tokens → 6.58 tok/s, 512 tokens → 33.60 tok/s
- **Decode is consistent**: ~1.04 tok/s regardless of decode length (in Debug)
- **Debug vs Release**: Release builds expected to be 5-10x faster
- **Backend routing**: Small ops use OpenBLAS, large prefills may use COSMA (if threshold met)
- **NUMA optimization**: K/V cache and activations benefit from first-touch allocation (+10-40% on large models)

### Kernel Profiling

Enable per-operation timing breakdown to identify bottlenecks:

```bash
# Run benchmark with kernel profiling
LLAMINAR_PROFILE_KERNELS=1 ./run_llaminar.sh -- --benchmark -m model.gguf -n 50
```

**Profiled Operations:**
- `GEMM_Q8`: Quantized GEMM (QKV projections, output projections)
- `ATTENTION`: Full attention computation (Q*K^T, softmax, *V)
- `FFN_DOWN`: FFN down projection (tracked separately)
- `FFN_GATE`, `FFN_UP`: FFN gate/up projections
- `LM_HEAD`: Final vocabulary projection
- `QUANTIZE_Q8`: FP32 → Q8_1 activation quantization
- `RMS_NORM`: RMS normalization
- `SWIGLU`: SwiGLU activation
- `ROPE`: Rotary position encoding
- `RESIDUAL_ADD`: Residual connections
- `EMBEDDING`: Token embedding lookup

**Example Output:**
```
╔══════════════════════════════════════════════════════════════════════════════╗
║                         KERNEL PROFILING SUMMARY                             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Kernel Type      │  Calls   │  Total (ms)  │  Avg (µs)  │  Min/Max (µs)     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  GEMM_Q8         │    7128  │     8947.85  │   1255.31  │   33.47/32651.11 ║
║  ATTENTION       │    1188  │     5915.90  │   4979.71  │  181.00/86178.09 ║
║  FFN_DOWN        │    1188  │     3891.00  │   3275.25  │  429.99/34938.14 ║
║  ...             │          │              │            │                   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  TOTAL KERNEL TIME:   19993.79 ms   ( 30.31 kernel tok/s)                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

**Typical Profile by Model Size:**

| Model Size | GEMM % | Attention % | FFN % | Notes |
|------------|--------|-------------|-------|-------|
| 0.5B | ~20% | ~57% | ~11% | Attention-bound |
| 3B | ~45% | ~30% | ~20% | Compute-bound |
| 7B+ | ~50%+ | ~25% | ~20% | Compute-bound |

**Key Insight**: Smaller models are attention-bound (OpenMP overhead in Q*K^T); larger models are compute-bound (JIT GEMM dominates).

**Implementation**: See `src/v2/utils/KernelProfiler.h` and `src/v2/utils/DebugEnv.h` (`ProfileConfig`).

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
  --report-bindings ./build_v2/src/v2/llaminar2
```

## Testing Guidelines (V2)

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

### Running Tests (V2)

```bash
# All tests
cd /workspaces/llaminar && ctest --test-dir build_v2 -R --output-on-failure

# Just unit tests
cd /workspaces/llaminar && ctest --test-dir build_v2 -R "^V2_Unit_" --output-on-failure --parallel

# Run a specific test in a single test file
cd /workspaces/llaminar && GTEST_FILTER=The.Specific.GTest.Name ctest --test-dir build_v2 -R V2_My_Test_File -V
```

### Snapshot Framework and E2E Testing

**Overview**: Llaminar V2 includes a comprehensive snapshot framework for capturing and comparing intermediate pipeline activations during inference. This is essential for E2E (end-to-end) correctness validation and debugging divergences.

**Key Concepts**:
- **Snapshots**: Captured intermediate tensors at specific pipeline stages (embeddings, Q/K/V projections, attention outputs, FFN outputs, layer norms, etc.)
- **Snapshot Keys**: Human-readable identifiers for each captured tensor (e.g., `"layer0_ATTENTION_CONTEXT"`, `"layer5_FFN_DOWN"`)
- **Ground Truth Comparison**: Compare Llaminar outputs against PyTorch reference implementation to validate correctness

#### Snapshot-Enabled Builds

**CRITICAL**: E2E tests require `ENABLE_PIPELINE_SNAPSHOTS` compile flag. This is **only enabled in specific build configurations**:

1. **E2ERelease Build** (recommended for E2E tests):
   ```bash
   # Create E2E-specific build with snapshots enabled
   cmake -B build_v2_e2e -S src/v2 \
     -DCMAKE_BUILD_TYPE=Release \
     -DENABLE_PIPELINE_SNAPSHOTS=ON \
     -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
   
   cmake --build build_v2_e2e --parallel
   ```

2. **Debug Build** (snapshots enabled by default):
   ```bash
   # Debug builds automatically enable snapshots
   cmake -B build_v2 -S src/v2 -DCMAKE_BUILD_TYPE=Debug
   cmake --build build_v2 --parallel
   ```

3. **Release Build** (snapshots **DISABLED**):
   ```bash
   # Standard release builds have snapshots disabled for performance
   cmake -B build_v2_release -S src/v2 -DCMAKE_BUILD_TYPE=Release
   cmake --build build_v2_release --parallel
   # ⚠️ E2E tests will FAIL to compile against this build!
   ```

#### Running Qwen2 E2E Tests

**Basic E2E Test Execution**:
```bash
# Set up optimal MPI/OpenMP environment
export OMP_NUM_THREADS=28 OMP_PLACES=sockets OMP_PROC_BIND=close \
       OMP_NESTED=false OMP_DYNAMIC=false \
       KMP_AFFINITY=granularity=fine,compact,1,0 KMP_BLOCKTIME=0 \
       OPENBLAS_NUM_THREADS=28 GOTO_NUM_THREADS=28 MKL_NUM_THREADS=28 MKL_DYNAMIC=false \
       OMPI_MCA_mpi_leave_pinned=1 OMPI_MCA_btl_vader_single_copy_mechanism=none \
       OMPI_MCA_btl_openib_allow_ib=1 LLAMINAR_LOG_LEVEL=ERROR

# Run all E2E tests (requires 2 MPI ranks)
timeout 180 mpirun -np 2 --bind-to socket --map-by socket \
  ./build_v2_e2e/tests/v2/v2_test_qwen2_e2e_correctness

# Run specific E2E test
timeout 180 mpirun -np 2 --bind-to socket --map-by socket \
  ./build_v2_e2e/tests/v2/v2_test_qwen2_e2e_correctness \
  --gtest_filter="Qwen2E2ECorrectness.SingleTokenInference"
```

**Available E2E Tests** (`tests/v2/e2e/Test__Qwen2E2ECorrectness.cpp`):
- `SingleTokenInference`: Single token forward pass validation
- `MultiSequenceBatch`: Batched inference with multiple sequences
- `BatchScaling`: Batch size scaling validation
- `ComprehensiveBatchParity`: Sequential vs batched execution comparison

#### Snapshot Capture API

**Pipeline-Level Snapshot Control**:
```cpp
// Enable snapshot capture (must be done before forward pass)
pipeline->enableSnapshotCapture("/tmp/llaminar_snapshots");

// Run inference (snapshots automatically captured at instrumented points)
pipeline->forward(input_ids, seq_len);

// Retrieve captured snapshots
auto snapshot_keys = pipeline->getSnapshotKeys();  // List of all captured stages
auto tensor_data = pipeline->getSnapshot("layer0_ATTENTION_CONTEXT");  // Get specific snapshot
```

**Common Snapshot Keys** (Qwen2 Pipeline):
- `EMBEDDING`: Token embeddings after embedding layer
- `layer{N}_ATTENTION_NORM`: Pre-attention RMS normalization
- `layer{N}_Q_PROJECTION`, `layer{N}_K_PROJECTION`, `layer{N}_V_PROJECTION`: QKV projections
- `layer{N}_Q_ROPE`, `layer{N}_K_ROPE`: After RoPE application
- `layer{N}_ATTENTION_CONTEXT`: Attention output before output projection
- `layer{N}_ATTENTION_OUTPUT`: After attention output projection
- `layer{N}_ATTENTION_RESIDUAL`: After attention residual connection
- `layer{N}_FFN_NORM`: Pre-FFN RMS normalization
- `layer{N}_FFN_GATE`, `layer{N}_FFN_UP`: FFN gate and up projections
- `layer{N}_FFN_SWIGLU`: After SwiGLU activation
- `layer{N}_FFN_DOWN`: FFN down projection
- `layer{N}_FFN_RESIDUAL`: After FFN residual connection
- `FINAL_NORM`: Final RMS normalization before LM head
- `LM_HEAD`: Logits output

#### Debugging with Snapshots

**Layer-by-Layer Divergence Detection**:
```cpp
// Example from ComprehensiveBatchParity test
auto seq_keys = pipeline_sequential->getSnapshotKeys();
for (const auto& key : seq_keys) {
    auto seq_snapshot = pipeline_sequential->getSnapshot(key);
    auto batch_snapshot = pipeline_batched->getSnapshot(key);
    
    // Extract corresponding sequence from batch
    // Compare tensors with tolerance
    if (max_diff > tolerance) {
        LOG_ERROR("First divergence at stage: " << key);
        LOG_ERROR("Max diff: " << max_diff << ", Mean diff: " << mean_diff);
        break;
    }
}
```

**Snapshot Output Directories**:
- `/tmp/llaminar_snapshots/`: Default snapshot directory
- Snapshots are **NOT automatically cleaned up** - useful for post-test analysis
- Each snapshot is a binary file containing FP32 tensor data

#### Debugging Layer Divergence with Tensor Dumps

When Q8_1 (or other quantized) inference produces different results than FP32, use the tensor dump feature to trace divergence layer-by-layer and element-by-element. This is the systematic approach used to find and fix issues like gamma weight clipping bugs.

**Step 1: Run the comparison test with tensor dumps enabled**

First, identify which layer shows divergence by running the parity test with tensor dumps:

```bash
# Clean previous dumps
rm -rf /tmp/layer_debug

# Run with tensor dump enabled for all layers
export LLAMINAR_SNAPSHOT_TENSOR_DUMP=1
export LLAMINAR_SNAPSHOT_DUMP_DIR=/tmp/layer_debug
export LLAMINAR_SNAPSHOT_DUMP_LAYERS=all
export LLAMINAR_SNAPSHOT_DUMP_STAGES=FFN_RESIDUAL,ATTENTION_RESIDUAL
export LLAMINAR_LOG_LEVEL=ERROR

# Run the Q8_1 vs FP32 parity test
timeout 180 mpirun -np 2 --bind-to socket --map-by socket \
  ./build_v2_e2e/tests/v2/v2_test_q8_1_layer_divergence \
  --gtest_filter="Test__Q8_1_LayerByLayer.SnapshotComparison"
```

**Step 2: Analyze dumps to find first divergence point**

Use Python to trace a specific element through layers:

```python
import numpy as np
import os

dump_dir = "/tmp/layer_debug"
element_idx = 62  # Index to trace (find by looking at max diff elements)

for layer in range(24):
    for stage in ["ATTENTION_RESIDUAL", "FFN_RESIDUAL"]:
        fp32_file = f"{dump_dir}/fp32_pipeline_layer{layer}_{stage}_rank0_fp32.bin"
        q8_1_file = f"{dump_dir}/q8_1_pipeline_layer{layer}_{stage}_rank0_fp32.bin"
        
        if not os.path.exists(fp32_file):
            continue
            
        fp32_data = np.fromfile(fp32_file, dtype=np.float32)
        q8_1_data = np.fromfile(q8_1_file, dtype=np.float32)
        
        diff = abs(fp32_data[element_idx] - q8_1_data[element_idx])
        print(f"L{layer} {stage}: FP32={fp32_data[element_idx]:.4f}, "
              f"Q8_1={q8_1_data[element_idx]:.4f}, diff={diff:.4f}")
```

**Step 3: Drill down into the divergent layer**

Once you identify which layer diverges (e.g., Layer 3 FFN), capture all stages for that layer:

```bash
rm -rf /tmp/layer3_detail
export LLAMINAR_SNAPSHOT_DUMP_DIR=/tmp/layer3_detail
export LLAMINAR_SNAPSHOT_DUMP_LAYERS=3
export LLAMINAR_SNAPSHOT_DUMP_STAGES=all  # All stages within layer 3

# Re-run the test
timeout 180 mpirun -np 2 ... ./build_v2_e2e/tests/v2/v2_test_q8_1_layer_divergence
```

Then analyze each stage to pinpoint where divergence begins:

```python
stage_order = [
    "ATTENTION_NORM", "Q_PROJECTION", "K_PROJECTION", "V_PROJECTION",
    "ATTENTION_CONTEXT", "ATTENTION_OUTPUT", "ATTENTION_RESIDUAL",
    "FFN_NORM", "FFN_GATE", "FFN_UP", "FFN_SWIGLU", "FFN_DOWN", "FFN_RESIDUAL"
]

for stage in stage_order:
    # Load and compare FP32 vs Q8_1 for each stage
    # Find where the big jump happens (e.g., FFN_NORM if gamma clipping issue)
```

**Step 4: Check model weights for problematic values**

If the divergence originates in a norm layer, check the gamma weights:

```python
from gguf import GGUFReader

reader = GGUFReader("models/qwen2.5-0.5b-instruct-q8_0.gguf")
for tensor in reader.tensors:
    if "blk.3.ffn_norm" in tensor.name:
        weights = np.frombuffer(tensor.data, dtype=np.float32)
        print(f"Min: {weights.min()}, Max: {weights.max()}")
        # Check for values exceeding kernel limits (e.g., |gamma| > 8 for Q12)
        large = np.where(np.abs(weights) > 8)[0]
        print(f"Elements > 8: {large}")
```

**Real-World Example: Q8_1 RMSNorm Gamma Clipping Bug**

Using this approach, we discovered that the Q8_1 RMSNorm kernel used Q12 fixed-point format for gamma weights with a [-8, 8] clamp range. In Qwen2.5-0.5B, gamma[62] = 13.015625, which got clipped to 8.0, causing 38% error that cascaded through layers and eventually caused sign flips in layer 21.

The fix was to expand the gamma range from Q12 [-8, 8] to Q11 [-16, 16] in `src/v2/kernels/cpu/primitives/RMSNormPrimitives.cpp`.

**Key Indicators of Common Issues**:
- **Sudden large diff at specific element**: Check if model weights exceed kernel limits
- **Gradual accumulation across layers**: Normal quantization noise, may need tighter tolerances
- **Sign flips in residual connections**: Usually caused by large errors in inputs that cancel incorrectly
- **Specific layer consistently bad**: Check that layer's norm weights or attention parameters

#### Compile-Time Safety

E2E tests include compile-time checks to prevent building against non-snapshot builds:

```cpp
// In Test__Qwen2E2ECorrectness.cpp
#ifndef ENABLE_PIPELINE_SNAPSHOTS
#error "Test requires ENABLE_PIPELINE_SNAPSHOTS. Build with: \
cmake -B build_v2_e2e -S src/v2 -DCMAKE_BUILD_TYPE=Release -DENABLE_PIPELINE_SNAPSHOTS=ON"
#endif
```

This ensures clear error messages if you accidentally try to build E2E tests against `build_v2_release`.

#### Performance Impact

**Snapshot Overhead**:
- **Disabled (Release builds)**: Zero overhead - snapshot capture code compiled out
- **Enabled (E2E/Debug builds)**: ~10-20% slowdown due to tensor copying and disk I/O
- Snapshots are **only captured when explicitly enabled** via `enableSnapshotCapture()`

**Best Practices**:
- Use E2ERelease build for correctness testing (snapshots enabled, optimizations on)
- Use standard Release build for performance benchmarking (snapshots disabled)
- Snapshots are not needed for unit or integration tests (use Debug build for those)

#### Further Reading

For in-depth documentation on the snapshot framework architecture, implementation details, and PyTorch reference comparison:
- **`.github/instructions/llaminar-v2-architecture.instructions.md`** - Complete snapshot system design
- **`docs/v2/SNAPSHOT_FRAMEWORK_DESIGN.md`** - Detailed technical documentation
- **`python/reference/`** - PyTorch reference implementation for ground truth generation

### Model Loading in Tests

**Best Practice**: Use `ModelContext` to manage model loading and tensor creation in tests. This ensures consistent initialization of `ModelLoader`, `WeightManager`, and `TensorFactory`.

#### 1. E2E / Integration Tests (Real Model)

For tests that require a real GGUF model (e.g., E2E correctness tests), use `ModelContext::create()`:

```cpp
// In SetUp()
const std::string model_path = "models/qwen2.5-0.5b-instruct-q4_0.gguf";

// Create MPI context first
auto mpi_ctx = std::make_shared<MPIContext>(rank, world_size, MPI_COMM_WORLD);

// Load model with MPI context
auto model_ctx = ModelContext::create(model_path, mpi_ctx);
if (!model_ctx) {
    FAIL() << "Failed to load model: " << model_path;
}

// Create pipeline using the context
auto pipeline = PipelineFactory::create(
    model_ctx->architecture(), 
    model_ctx, 
    mpi_ctx
);
```

#### 2. Unit Tests (Mock/Test Model)

For unit tests that need a pipeline structure but don't require real weights (or want to avoid loading a large file), use `ModelContext::createForTesting()`:

```cpp
// Create a test-only context (initializes minimal valid metadata)
auto model_ctx = ModelContext::createForTesting(
    "dummy_path.gguf", 
    nullptr, // No MPI context for simple unit tests
    24       // block_count (optional)
);

// You can now create a pipeline or access the loader
// The loader will have initialized metadata but no actual weights
```

#### 3. Tensor Creation

Avoid manual `new FP32Tensor(...)`. Use `TensorFactory` to ensure correct NUMA placement and memory alignment:

```cpp
// Get factory from context or create one
TensorFactory factory(mpi_ctx);

// Create tensor with specific placement
auto tensor = factory.create({32, 4096}, TensorType::FP32, device_idx);
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
ctest --test-dir build_v2 -R "^V2_Unit_" --output-on-failure --parallel
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
cmake --build build_v2_release --parallel # IMPORTANT: always run with `--parallel` for faster builds!

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
cd build_v2 && ctest -L Performance -R "MyFeature" --verbose --parallel
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
timeout 120 bash -c 'mpirun -np 2 bash -c "gdb -x /tmp/gdbcommands.txt --args ./build_v2/tests/v2/v2_test_my_feature 2>&1 | tee /tmp/gdb_rank_\$OMPI_COMM_WORLD_RANK.log"'

# Then examine rank-specific backtraces
grep -A 50 "Program received signal" /tmp/gdb_rank_0.log
grep -A 50 "Program received signal" /tmp/gdb_rank_1.log
```

#### ASAN for more complex, double-free style segfaults

Use ASAN for localizing more complex double-free style issues:

```bash
cmake -B build_v2 -S src/v2 -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS="-g3 -O0 -fno-omit-frame-pointer -fsanitize=address" -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON 

ASAN_OPTIONS=halt_on_error=0:detect_leaks=0 timeout 240 mpirun -np 2 ./build_v2/tests/v2/<your_test> --gtest_filter=<your_test_filter>
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
mpirun -np 2 valgrind --tool=memcheck --leak-check=full ./build_v2/tests/v2/v2_test_my_feature

# NUMA binding issues
numactl --hardware
numactl --show
export OMP_NUM_THREADS=28 OMP_PLACES=sockets OMP_PROC_BIND=close OMP_NESTED=false OMP_DYNAMIC=false 
export KMP_AFFINITY=granularity=fine,compact,1,0 KMP_BLOCKTIME=0 OPENBLAS_NUM_THREADS=28 
export GOTO_NUM_THREADS=28 MKL_NUM_THREADS=28 MKL_DYNAMIC=false OMPI_MCA_mpi_leave_pinned=1
export OMPI_MCA_btl_vader_single_copy_mechanism=none OMPI_MCA_btl_openib_allow_ib=1
export LLAMINAR_LOG_LEVEL=DEBUG
mpirun -np 2 numactl --cpubind=0 ./build_v2_release/src/v2/llaminar2 : numactl --cpubind=1 ./build_v2_release/src/v2/llaminar2

# COSMA hanging issues (use timeouts)
export OMP_NUM_THREADS=28 OMP_PLACES=sockets OMP_PROC_BIND=close OMP_NESTED=false OMP_DYNAMIC=false 
export KMP_AFFINITY=granularity=fine,compact,1,0 KMP_BLOCKTIME=0 OPENBLAS_NUM_THREADS=28 
export GOTO_NUM_THREADS=28 MKL_NUM_THREADS=28 MKL_DYNAMIC=false OMPI_MCA_mpi_leave_pinned=1
export OMPI_MCA_btl_vader_single_copy_mechanism=none OMPI_MCA_btl_openib_allow_ib=1
export LLAMINAR_LOG_LEVEL=DEBUG
timeout 60 mpirun -np 2 ./build_v2_release/src/v2/llaminar2
```

## Kernel Development (V2)

**Purpose**: V2 kernel development follows the **operator-free design** with direct pipeline orchestration and the **ITensorGemmTileDataProvider strategy pattern** for quantized tensors.

### Core V2 Principles

1. **Tensor-centric Design**: All operations are focused on Tensors
2. **Per-Tensor Device Affinity**: Tensors know their device placement
3. **Strategy Pattern**: Generic kernels + format-specific decode strategies
4. **ITensor Interfaces Expose Kernels**: `ITensorGemm`, `ITensorAttention`, `ITensorRoPE`, etc.

### ITensorGemmTileDataProvider Strategy Pattern

**Problem**: Quantized tensors (IQ4_NL, Q6_K, Q8_0) need format-specific decode logic, but we want a **single generic GEMM kernel**.

**Solution**: ITensorGemmTileDataProvider interface + QuantizedGemmKernel

```cpp
// src/v2/tensors/TensorKernels.h
class ITensorGemmTileDataProvider {
public:
    __attribute__((always_inline))
    virtual void decode_block_at(size_t row_idx, size_t k_block_offset, float* output) const = 0;
    
    virtual const void* get_raw_block_at(size_t row_idx, size_t k_block_offset) const = 0;
    
    virtual size_t decoder_rows() const = 0;
    virtual size_t decoder_cols() const = 0;
    virtual size_t block_size() const = 0;
};

// src/v2/tensors/Tensors.h
class IQ4_NLTensor : public TensorBase, public ITensorGemmTileDataProvider {
    // Inline implementation for zero overhead
    void decode_block_at(size_t row_idx, size_t k_block_offset, float* output) const override {
        const IQ4_NLBlock& block = blocks_[row_idx * blocks_per_row_ + k_block_offset];
        decodeBlock(block, output);  // Format-specific decode
    }
    
    size_t block_size() const override { return 32; }  // IQ4_NL: 32 elements/block
    size_t decoder_rows() const override { return rows_; }
    size_t decoder_cols() const override { return cols_; }
    
    std::unique_ptr<ITensorGemm> createGemm(const MPIContext& mpi_ctx, int device_idx) const override {
        return std::make_unique<QuantizedGemmKernel>(this, mpi_ctx, device_idx);
    }
};

// src/v2/kernels/cpu/QuantizedGemm.h
class QuantizedGemmKernel : public ITensorGemm {
    const ITensorGemmTileDataProvider* decoder_;  // Strategy interface
    
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
- ✅ **Extensibility**: New formats just implement ITensorGemmTileDataProvider

### Adding New Quantized Formats (V2)

```cpp
// src/v2/tensors/Q6_KTensor.h
class Q6_KTensor : public TensorBase, public ITensorGemmTileDataProvider {
public:
    void decode_block_at(size_t row_idx, size_t k_block_offset, float* output) const override {
        const Q6_KBlock& block = blocks_[row_idx * blocks_per_row_ + k_block_offset];
        decodeQ6KBlock(block, output);  // Q6_K-specific decode
    }
    
    size_t block_size() const override { return 256; }  // Q6_K: 256 elements/block
    size_t decoder_rows() const override { return rows_; }
    size_t decoder_cols() const override { return cols_; }
    
    std::unique_ptr<ITensorGemm> createGemm(const MPIContext& mpi_ctx, int device_idx) const override {
        return std::make_unique<QuantizedGemmKernel>(this, mpi_ctx, device_idx);  // Reuse generic kernel!
    }
    
private:
    static void decodeQ6KBlock(const Q6_KBlock& block, float* output);
};
```

**Result**: Single QuantizedGemmKernel works for IQ4_NL, Q6_K, Q8_0, etc.

### V2 Kernel Interface Design

All V2 kernels implement ITensor* interfaces and receive MPIContext + device_idx:

```cpp
// src/v2/tensors/TensorKernels.h
class ITensorGemm {
public:
    virtual ~ITensorGemm() = default;
    virtual bool multiply(
        const float* A,        // Activation matrix (m × k)
        float* C,              // Output matrix (m × n)
        int m, int n, int k,
        float alpha = 1.0f, float beta = 0.0f
    ) = 0;
};

class ITensorAttention {
public:
    virtual ~ITensorAttention() = default;
    virtual bool compute(
        TensorBase* Q, TensorBase* K, TensorBase* V,
        TensorBase* output,
        const AttentionWorkspace& workspace,
        const AttentionConfig& config
    ) = 0;
};

class ITensorRoPE {
public:
    virtual ~ITensorRoPE() = default;
    virtual bool apply(
        TensorBase* tensor,
        int seq_len, int head_dim,
        int pos_offset, float theta_base
    ) = 0;
};
```

**Design Principles**:
- Kernels receive `MPIContext` and `device_idx` at construction (from tensor factory methods)
- No global state - all context passed explicitly
- Activation tensors implement `IActivationTensor` with `createGemm()`, `createAttention()`, `createRoPE()` factory methods

### V2 Pipeline Orchestration

```cpp
// src/v2/pipelines/qwen/Qwen2Pipeline.cpp
bool Qwen2Pipeline::attention_block(const LayerWeights& layer, int layer_idx, int effective_seq_len) {
    // Determine execution device based on weight placement
    int attn_device = placement_map_ ? getWeightDevice("attn_q", -1) : device_idx_;
    
    // Get device-appropriate buffers (pre-allocated, no hot-path allocation!)
    auto& buffers = placement_map_ ? getBuffersForDevice(attn_device) : activation_buffers_;
    
    // Pre-attention RMSNorm via IActivationTensor interface
    auto* activation_tensor = dynamic_cast<IActivationTensor*>(buffers.normalized.get());
    VALIDATE_OP(activation_tensor->applyRMSNorm(
        layer.attn_norm->data(), effective_seq_len, d_model_, rms_norm_eps_
    ));
    
    // Attention computation via MpiAttentionOrchestrator → GQAAttention → ITensorAttention
    VALIDATE_OP(mpi_attention_orchestrator_->compute(
        buffers.normalized.get(),  // Input
        layer.wq.get(), layer.wk.get(), layer.wv.get(), layer.wo.get(),  // Weights
        kv_cache_.get(), layer_idx, effective_seq_len,
        buffers.attn_output.get()  // Output
    ));
    
    // Residual connection (using pre-allocated buffers)
    // ... etc
}
```

**See Also**: `.github/instructions/llaminar-v2-architecture.instructions.md` for complete V2 documentation.

### SIMD Writing Guidelines for v2 Kernels

When writing SIMD in v2, we follow these principles:

1. **Exploit ILP**: Unroll loops, do interleaved loads and stores to exploit dual load ports
2. **Vectorized Tail Handling in All Loops**: AVX512 16-way, AVX2 8-way, AVX 4-way, SSE 2-way, scalar tail
3. **Prefetch**: Prefetch upcoming sequential reads

### OpenMP Nested-Safe Parallel Regions (OMP_WORKSHARE_REGION)

Location: `src/v2/utils/OpenMPUtils.h`

**Purpose**: Provides macros for creating OpenMP parallel regions that are safe to call from within existing parallel regions. This enables "layer-level fusion" where an outer parallel region can encompass multiple kernel calls, eliminating thread fork/join overhead between operations.

**Problem**: Creating a new `#pragma omp parallel` region has ~10-50µs overhead per invocation. With 482 parallel region entries per token (24 layers × ~20 operations), this adds up to significant overhead (measured at 29% of decode time in profiling).

**Solution**: Use `OMP_WORKSHARE_REGION` macro which checks `omp_in_parallel()` and only creates a new parallel region if not already inside one.

#### API Reference

```cpp
#include "utils/OpenMPUtils.h"

// Main macro - creates parallel region only if not already in one
#define OMP_WORKSHARE_REGION(work_fn)

// With barrier after work (when subsequent code depends on completion)
#define OMP_WORKSHARE_REGION_SYNC(work_fn)

// Helper functions
inline bool isInParallelRegion();  // Wrapper for omp_in_parallel()
inline int getThreadId();          // Wrapper for omp_get_thread_num()
inline int getNumThreads();        // Wrapper for omp_get_num_threads()

// RAII guard for temporary thread count changes
class ThreadCountGuard;
```

#### Usage Pattern

**ALWAYS use this pattern** when writing kernels with OpenMP parallelization:

```cpp
#include "utils/OpenMPUtils.h"

void my_kernel(float* data, int n) {
    // Define work as a lambda with worksharing constructs
    auto do_work = [&]() {
        #pragma omp for schedule(static)
        for (int i = 0; i < n; ++i) {
            data[i] = process(data[i]);
        }
    };
    
    // Execute: creates parallel region only if not already in one
    OMP_WORKSHARE_REGION(do_work);
}
```

**DO NOT** write verbose if/else patterns manually:

```cpp
// ❌ BAD - verbose, error-prone, hard to maintain
if (omp_in_parallel()) {
    #pragma omp for schedule(static)
    for (int i = 0; i < n; ++i) { /* ... */ }
} else {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) { /* ... */ }
}

// ✅ GOOD - use the macro
auto do_work = [&]() {
    #pragma omp for schedule(static)
    for (int i = 0; i < n; ++i) { /* ... */ }
};
OMP_WORKSHARE_REGION(do_work);
```

#### When to Use

| Scenario | Use |
|----------|-----|
| Kernel with `#pragma omp parallel for` | `OMP_WORKSHARE_REGION` |
| Multiple independent loops in one function | Wrap each in separate `OMP_WORKSHARE_REGION` with `nowait` |
| Work that must complete before continuing | `OMP_WORKSHARE_REGION_SYNC` |
| Single-threaded operation inside parallel region | Just call it (no macro needed) |

#### Implementation Files Using This Pattern

- `src/v2/kernels/cpu/gemm_v4/QuantisedGemmKernel.h` - All GEMM functions
- `src/v2/kernels/cpu/primitives/SwiGLUPrimitives.cpp` - All SwiGLU variants
- Future: All new kernels should use this pattern

### KernelFactory: Centralized Kernel Dispatch and Caching

Location: `src/v2/kernels/KernelFactory.h`, `src/v2/kernels/KernelFactory.cpp`

**Purpose**: KernelFactory provides centralized kernel creation and caching, eliminating duplicate device routing code across tensor types and ensuring weight packing happens only once.

#### Key Features

1. **Device Type Abstraction**: Simplified `DeviceType` enum (CPU, CUDA, ROCm, Vulkan, Metal) that groups backend variants
2. **Type-Safe Dispatch**: Template overloads for each quantized tensor type
3. **Kernel Caching**: Pack once, use many pattern with automatic lifecycle management
4. **Automatic Cache Invalidation**: TensorBase destructor calls `clearCacheFor(this)` to prevent stale entries

#### API Reference

```cpp
namespace llaminar::v2::kernels {

class KernelFactory {
public:
    // Device resolution
    static DeviceType getDeviceType(int device_idx);
    
    // Create kernel (uncached - prefer getOrCreateGemm)
    static std::unique_ptr<ITensorGemm> createGemm(const IQ4_NLTensor* tensor, DeviceType dev);
    static std::unique_ptr<ITensorGemm> createGemm(const Q4_0Tensor* tensor, DeviceType dev);
    // ... overloads for all tensor types
    
    // Cached GEMM API (preferred - pack once, use many)
    static ITensorGemm* getOrCreateGemm(const TensorBase* tensor);
    
    // Cache management
    static void clearCacheFor(const TensorBase* tensor);  // Called by TensorBase destructor
    static void clearCache();                              // Clear all entries
    static std::pair<size_t, size_t> cacheStats();         // (cache_size, packed_bytes)
};

}
```

#### Usage in Pipelines

```cpp
// Preferred pattern - cached kernel (pack once)
ITensorGemm* gemm = KernelFactory::getOrCreateGemm(weight_tensor.get());
gemm->multiply(activations, output, m, n, k);

// Alternative - uncached (for one-off use)
auto gemm = KernelFactory::createGemm(dynamic_cast<IQ4_NLTensor*>(tensor), DeviceType::CPU);
```

#### Cache Lifecycle

The kernel cache uses `TensorBase*` pointers as keys. To prevent use-after-free bugs when tensors are destroyed and memory addresses are reused:

1. **Automatic Invalidation**: `TensorBase::~TensorBase()` calls `KernelFactory::clearCacheFor(this)`
2. **Manual Cleanup**: Call `clearCache()` when tensor lifetimes are uncertain
3. **Thread Safety**: Cache operations protected by mutex

**Unit Tests**: `tests/v2/unit/Test__KernelFactoryCacheInvalidation.cpp` (10 tests)

---

## Weight Sharding and Tensor Parallelism

### Overview

Weight sharding enables tensor parallelism by distributing weight matrices across MPI ranks, reducing per-rank memory usage for large models.

**Default Behavior** (since December 2025):
- Single rank: No sharding
- Multiple MPI ranks: Automatic weight sharding (equivalent to `--shard-weights`)

### Sharding Modes

Location: `src/v2/loaders/WeightManager.h`

```cpp
enum class ShardingMode {
    REPLICATE,        // Full copy on each rank (default for small weights)
    COLUMN_PARALLEL,  // Split output dimension (Phase 2)
    ROW_PARALLEL      // Split input dimension (Phase 1)
};
```

**Phase 1 Strategy (Row-Parallel Only)**:
| Weight Pattern | Mode | Rationale |
|----------------|------|-----------|
| `attn_output.weight` | ROW_PARALLEL | Split input dim, allreduce output |
| `ffn_down.weight` | ROW_PARALLEL | Split input dim, allreduce output |
| QKV, Gate/Up, norms, embeddings | REPLICATE | Attention expects full tensors |

### CLI Usage

```bash
# Auto-enabled when world_size > 1 (default)
mpirun -np 2 ./run_llaminar.sh -m model.gguf -p "Hello"

# Explicit (redundant when np > 1)
./run_llaminar.sh -m model.gguf --shard-weights -p "Hello"

# Single-rank without sharding (no MPI)
./run_llaminar.sh -m model.gguf -p "Hello"
```

### Memory Savings

With 2 MPI ranks and row-parallel sharding:
- **Wo (attention output)**: 50% reduction (896×896 → 896×448 per rank)
- **Down (FFN)**: 50% reduction (4864×896 → 4864×448 per rank)
- **Overall**: ~25-30% memory reduction for row-parallel weights

### Implementation Details

**Weight Loading** (`WeightManager::getShardedWeight()`):
1. Load full tensor from GGUF
2. Determine sharding mode from weight name pattern
3. Slice tensor based on mode and rank position
4. Free original tensor, return sliced

**Pipeline Integration** (`PipelineBase::project_row_parallel()`):
```cpp
// Row-parallel projection with automatic allreduce
bool project_row_parallel(TensorBase* input, TensorBase* weight, TensorBase* output,
                          int m, int n, int k, const std::string& snapshot_key);
// Handles:
// - Sharded weights: Local slice GEMM + MPI_Allreduce
// - Replicated weights: Full GEMM + scale factor
```

### Future Work (Phase 2)

Column-parallel sharding (QKV, Gate/Up) requires:
1. Modify `MpiAttentionOrchestrator` to accept pre-sharded Q/K/V
2. Update buffer allocation for local dimensions
3. Update FusedGEMM to pass local N dimensions

**Documentation**: See `changelog/2025-06-30-weight-sharding-phase1.md` for detailed implementation notes.


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

**V2 Activation Buffer Pattern** (src/v2/pipelines/qwen/Qwen2Pipeline.cpp):
```cpp
// Pre-allocate buffers once during pipeline initialization
void Qwen2Pipeline::allocate_activation_buffers(int max_seq_len, int device_idx) {
    auto& buffers = activation_buffers_;
    buffers.max_seq_len = max_seq_len;
    
    // Allocate with device affinity
    buffers.residual = TensorFactory::create_fp32(
        {static_cast<size_t>(max_seq_len), d_model_}, device_idx
    );
    buffers.normalized = TensorFactory::create_fp32(
        {static_cast<size_t>(max_seq_len), d_model_}, device_idx
    );
    // ... etc
    
    // NUMA first-touch initialization (OpenMP parallel init)
    // Each thread touches its portion → local NUMA placement
    #pragma omp parallel for
    for (size_t i = 0; i < buffers.residual->numel(); ++i) {
        buffers.residual->mutable_data()[i] = 0.0f;
    }
    // ... repeat for all buffers
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
- Threshold: 128KB (V2 tensor allocations)

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
// Always add corresponding tests (V2 uses GTest)
// File: tests/v2/unit/Test__MyNewKernel.cpp
TEST(Test__MyNewKernel, BasicFunctionality) {
    MPIContext mpi_ctx(0);  // Rank 0
    int device_idx = 0;     // CPU device
    
    // Create tensors via factory
    auto input = TensorFactory::create_fp32({32, 32}, device_idx);
    auto output = TensorFactory::create_fp32({32, 32}, device_idx);
    
    // Create kernel via tensor interface
    auto* activation = dynamic_cast<IActivationTensor*>(input.get());
    ASSERT_NE(activation, nullptr);
    auto kernel = activation->createMyKernel(mpi_ctx, device_idx);
    
    // Execute and validate
    ASSERT_TRUE(kernel->execute(output.get(), /* ... */));
    
    // Validate outputs
    const float* out_data = output->data();
    EXPECT_NEAR(out_data[0], expected_value, 1e-5f);
}
```

### Centralized Environment Access (debugEnv)

All new or refactored code on hot paths (kernels, matmul selection, attention assembly, tensor partition loops) MUST avoid direct `std::getenv` calls. Instead:

1. Add any new environment flag/knob to the structured snapshot in `src/v2/utils/DebugEnv.h` inside the appropriate group (or create one).
2. Parse it once in `src/v2/utils/DebugEnv.cpp` (lazy static initialization already provided) and expose typed fields.
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

Do NOT add another ad-hoc snapshot facility; extend the existing one. If grouping is unclear, prefer adding a new subgroup struct within `src/v2/utils/DebugEnv.h` rather than mixing unrelated flags.


## Documentation Standards
### Doxygen

All files and functions should be documented with the Doxygen format for readability and easy understanding of their purpose. For @author, use David Sanftenberg.


## Debug / Instrumentation Environment Variables

Llaminar uses a "DebugEnv" singleton that loads all environment variables at startup and exposes them to classes at runtime. This avoids heavy getEnv() calls.

For a full list of environment variables available, check `src/v2/utils/DebugEnv.h`.

| Variable | Description | Default / Activation | Primary Scope |
|----------|-------------|----------------------|---------------|
| `LLAMINAR_PROFILE_KERNELS` | Enable per-kernel timing breakdown in benchmark mode. | Disabled unless set to non-zero. | Performance profiling |
| `LLAMINAR_DEQUANT_STATS` | Logs per-tensor dequant stats (min/max/mean/sample). | Disabled unless set to non-zero. | Quantized tensor loading |
| `OMP_NUM_THREADS` / `OMP_PLACES` / `OMP_PROC_BIND` | Governs OpenMP thread placement & counts (run script sets). | Auto-set by `run_llaminar.sh` | Threading performance |
| `LLAMINAR_USE_LAYER_EXECUTOR` | Enable LayerExecutor (Qwen2Graph) for declarative compute graphs. | Disabled (0) | Execution framework |
| `LLAMINAR_USE_GRAPH_BUFFER_MANAGEMENT` | When LayerExecutor is enabled, use GraphBufferManager for automatic buffer allocation with aliasing optimization. | Disabled (0) | Buffer management |
| `LLAMINAR_SNAPSHOT_TENSOR_DUMP` | Enable raw tensor dump to disk for debugging. | Disabled (0) | Snapshot framework |
| `LLAMINAR_SNAPSHOT_DUMP_DIR` | Output directory for tensor dumps. | `/tmp/llaminar_tensor_dumps` | Snapshot framework |
| `LLAMINAR_SNAPSHOT_DUMP_LAYERS` | Comma-separated layer indices to dump. | `all` | Snapshot framework |
| `LLAMINAR_SNAPSHOT_DUMP_STAGES` | Comma-separated stage names to dump. | `all` | Snapshot framework |
| `LLAMINAR_SNAPSHOT_DUMP_RANK` | Only dump from this MPI rank (-1=all). | `0` | Snapshot framework |

### Graph Buffer Management

When both `LLAMINAR_USE_LAYER_EXECUTOR=1` and `LLAMINAR_USE_GRAPH_BUFFER_MANAGEMENT=1` are set, the GraphExecutor takes over buffer responsibilities from the pipeline. This enables automatic aliasing optimization for SCRATCH buffers that don't have overlapping lifetimes.

```bash
# Enable graph-managed buffers with LayerExecutor
LLAMINAR_USE_LAYER_EXECUTOR=1 \
LLAMINAR_USE_GRAPH_BUFFER_MANAGEMENT=1 \
./run_llaminar.sh -m model.gguf -p "Hello world"
```

Benefits of graph-managed buffers:
- **Memory savings**: Non-overlapping SCRATCH buffers can share physical memory
- **Automatic aliasing**: Q↔gate, K↔up, V↔ffn_output aliasing
- **Centralized allocation**: All buffers managed by GraphBufferManager with NUMA awareness

### Tensor Dump Feature

The tensor dump feature allows you to save raw tensor data (both FP32 dequantized and native Q8_1 blocks) to disk for detailed debugging and analysis. This is integrated with the snapshot framework but works in any build (including Release).

**Example: Debug layer 21 FFN residual add divergence**
```bash
# Dump only layer 21 FFN stages
LLAMINAR_SNAPSHOT_TENSOR_DUMP=1 \
LLAMINAR_SNAPSHOT_DUMP_LAYERS=21 \
LLAMINAR_SNAPSHOT_DUMP_STAGES=FFN_INPUT_RESIDUAL,FFN_DOWN,FFN_RESIDUAL \
LLAMINAR_SNAPSHOT_DUMP_DIR=/tmp/layer21_debug \
./run_llaminar.sh -m model.gguf -p "test prompt"
```

**Output Files:**
- `<stage>_rank<N>_fp32.bin` - FP32 dequantized data (raw float32 binary)
- `<stage>_rank<N>_q8_1_blocks.bin` - Raw Q8_1 blocks (for Q8_1 tensors)
- `<stage>_rank<N>_metadata.txt` - Shape, type, and sample scale information

**Stage Names:**
- Global: `EMBEDDING`, `FINAL_NORM`, `LM_HEAD`
- Per-layer: `ATTENTION_NORM`, `Q_PROJECTION`, `K_PROJECTION`, `V_PROJECTION`, `Q_ROPE`, `K_ROPE`, `ATTENTION_CONTEXT`, `ATTENTION_OUTPUT`, `ATTENTION_RESIDUAL`, `FFN_NORM`, `FFN_GATE`, `FFN_UP`, `FFN_SWIGLU`, `FFN_DOWN`, `FFN_INPUT_RESIDUAL`, `FFN_RESIDUAL`


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

## Documentation and Project Resources

### Key Documentation Files

**Developer Guidelines:**
- **`.github/copilot-instructions.md`** (this file): Comprehensive V2 development guidelines
- **`.github/instructions/llaminar-architecture-v2.instructions.md`**: High-level V2 architecture map
- **`.github/instructions/llaminar-v2-architecture.instructions.md`**: Detailed V2 implementation notes

**V2 Build Targets:**
- `build_v2/`: Debug build (for development, `CMAKE_BUILD_TYPE=Debug`)
- `build_v2_release/`: Release build (for benchmarking, `CMAKE_BUILD_TYPE=Release`)
- `build_v2/src/v2/llaminar2`: V2 executable (device enumeration)
- `build_v2/tests/v2/`: V2 test executables

**Model Support:**
- **V2**: Qwen 2.5 family (0.5B-72B)
- **Quantization**: Q4_0, Q6_K, Q8_0, IQ4_NL, F16, F32

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

### Getting Help

**Common issues:**
1. **MPI hangs**: Check barriers around collective operations, verify all ranks participate
2. **Numerical divergence**: Run parity tests to identify diverging stage
3. **Performance issues**: Use `--benchmark` mode with `./run_llaminar.sh`, check OpenMP thread settings
4. **Memory errors**: Enable NUMA verification, check first-touch allocation
5. **Build errors**: Clean build directory, verify dependencies (OpenBLAS, MPI, ScaLAPACK)

**Debug environment variables**: See "Debug / Instrumentation Environment Variables" section for full list of `LLAMINAR_*` flags.

## PyTorch Comparison Testing and Snapshotting Framework

We have a PyTorch comparison/snapshotting framework in the `python/reference/` folder. This framework allows us to inference the same GGUFs that Llaminar uses, capturing stage-by-stage, layer-by-layer snapshots in PyTorch for comparison against Llaminar's outputs.

### Key Concepts

**CRITICAL**: Agents often get confused and think the PyTorch framework uses FP32 weights from HuggingFace. **This is NOT true!** PyTorch and Llaminar **use the exact same GGUF weights** (e.g., Q4_0) loaded from the `models/` folder. The key differences are:

1. **PyTorch**: Dequantizes Q4_0 → FP32 upfront, then runs FP32 matmuls
2. **Llaminar**: Does GEMM in the integer domain with AVX512-VNNI, converting activations to Q8_1 block format

This means Llaminar's matmuls are "noisy" compared to PyTorch's FP32 ground truth, but this is acceptable for inference. We only worry if cosine similarity plummets in the first few layers.

### E2E Test Configuration

The E2E parity tests use these standardized constants (defined in `tests/v2/e2e/qwen2/Test__Qwen2FP32Parity.cpp`):

```cpp
// Model path - MUST match what Python uses
static constexpr const char* TEST_MODEL_PATH = "models/qwen2.5-0.5b-instruct-q4_0.gguf";

// Test prompt - MUST match Python snapshot generation
static constexpr const char* TEST_PROMPT = "The quick brown fox jumps over the lazy dog";

// Token IDs (from Qwen2.5 tokenizer) - MUST match between C++ and Python
static const std::vector<int> TEST_TOKEN_IDS = {785, 3974, 13876, 38835, 34208, 916, 279, 15678, 5562};

// Snapshot directory
static constexpr const char* TEST_SNAPSHOT_DIR = "pytorch_qwen2_snapshots";
```

**CRITICAL**: E2E tests use **greedy sampling (temperature = 0.0)** for deterministic, reproducible token prediction. When debugging inference issues, always test with `-t 0` to match E2E behavior:

```bash
# Greedy sampling (matches E2E tests)
./run_llaminar.sh -m models/qwen2.5-0.5b-instruct-q4_0.gguf -p "The quick brown fox jumps over the lazy dog" -n 10 -t 0

# Default sampling (temperature=0.8) produces probabilistic/non-deterministic output
./run_llaminar.sh -m models/qwen2.5-0.5b-instruct-q4_0.gguf -p "The quick brown fox jumps over the lazy dog" -n 10
```

### Parity Criteria

Our comparison criteria between Llaminar and PyTorch:

| Metric | Threshold | Notes |
|--------|-----------|-------|
| Cosine similarity (layers 0-5) | ≥ 0.999 | Expected to diverge in deeper layers |
| Top-1 token match | 100% | Final token prediction must match |
| Top-5 overlap | ≥ 60% | Order may differ slightly |
| KL divergence | Low | Logits distribution alignment |

Cosine similarity is expected to diverge in layers ≥ 6 due to accumulated quantization noise. What matters is **final token choice alignment**.

### Running PyTorch Reference Inference

#### Generate Snapshots from GGUF

```bash
cd /workspaces/llaminar

# Generate full pipeline snapshots (prefill + decode)
python python/reference/generate_qwen2_pipeline_snapshots.py \
    --model models/qwen2.5-0.5b-instruct-q4_0.gguf \
    --output pytorch_qwen2_snapshots \
    --decode-steps 5 \
    --verbose

# Generate snapshots for specific layers only
python python/reference/generate_qwen2_pipeline_snapshots.py \
    --model models/qwen2.5-0.5b-instruct-q4_0.gguf \
    --layers 0,1,2 \
    --output pytorch_layer012_snapshots

# Custom prompt (must match C++ test if comparing)
python python/reference/generate_qwen2_pipeline_snapshots.py \
    --model models/qwen2.5-0.5b-instruct-q4_0.gguf \
    --prompt "Hello, world!" \
    --output custom_snapshots
```

#### Run Simple Inference (Text Generation)

```bash
# Generate text with PyTorch reference
python python/reference/run_reference.py \
    --model qwen \
    --checkpoint models/qwen2.5-0.5b-instruct-q4_0.gguf \
    --prompt "The quick brown fox" \
    --max-tokens 20 \
    --verbose
```

### Snapshot System Architecture

#### Directory Structure

Snapshots are saved as NumPy `.npy` files:

```
pytorch_qwen2_snapshots/
├── metadata.txt              # Model config, token IDs, decode tokens
├── EMBEDDING.npy             # Prefill: [1, seq_len, d_model]
├── FINAL_NORM.npy            # Prefill: [1, seq_len, d_model]
├── LM_HEAD.npy               # Prefill: [1, seq_len, vocab_size]
├── layer0_ATTENTION_NORM.npy
├── layer0_Q_PROJECTION.npy
├── layer0_K_PROJECTION.npy
├── layer0_V_PROJECTION.npy
├── layer0_ATTENTION_CONTEXT.npy
├── layer0_ATTENTION_OUTPUT.npy
├── layer0_ATTENTION_RESIDUAL.npy
├── layer0_FFN_NORM.npy
├── layer0_FFN_GATE.npy
├── layer0_FFN_UP.npy
├── layer0_FFN_SWIGLU.npy
├── layer0_FFN_DOWN.npy
├── layer0_FFN_RESIDUAL.npy
├── ... (layers 1-23)
├── decode_step0_EMBEDDING.npy    # Decode step 0
├── decode_step0_layer0_*.npy
├── decode_step0_LM_HEAD.npy
├── decode_step1_*.npy            # Decode step 1
└── ...
```

#### Metadata File Format

```
Model: models/qwen2.5-0.5b-instruct-q4_0.gguf
Architecture: Qwen2Config
n_layers: 24
n_heads: 14
n_kv_heads: 2
d_model: 896
d_head: 64
d_ff: 4864
vocab_size: 151936
num_snapshots: 2466
decode_tokens: 256,40400,50735,29285,29285,60119
```

The `decode_tokens` field shows the tokens PyTorch generated during decode (with greedy sampling). These should match Llaminar's output with `-t 0`.

### Investigating Logits and Token Predictions

#### Load and Analyze PyTorch Logits

```python
import numpy as np

# Load prefill logits (after processing full prompt)
logits = np.load('pytorch_qwen2_snapshots/LM_HEAD.npy')
print(f'Shape: {logits.shape}')  # (1, seq_len, vocab_size)

# Get last position logits (prediction for next token)
last_logits = logits[0, -1, :]

# Find top-5 predictions
top_indices = np.argsort(last_logits)[::-1][:5]
print('PyTorch Top-5 predictions:')
for i, idx in enumerate(top_indices):
    print(f'  {i+1}. Token {idx} score={last_logits[idx]:.4f}')
```

#### Compare Llaminar vs PyTorch Logits

```bash
# Run Llaminar with TRACE logging to see top-5 predictions
LLAMINAR_LOG_LEVEL=TRACE ./run_llaminar.sh \
    -m models/qwen2.5-0.5b-instruct-q4_0.gguf \
    -p "The quick brown fox jumps over the lazy dog" \
    -n 5 -t 0 2>&1 | grep -E "Top-5|Token [0-9]|Sampled"
```

Expected output when working correctly:
```
[TRACE] [Rank 0] Top-5 logits at decode iteration 0:
[TRACE]   1. Token 256 score=14.5406
[TRACE]   2. Token 8159 score=14.0087
[TRACE]   3. Token 100160 score=13.1267
[DEBUG] [Rank 0] Sampled token: 256
```

The top-1 token should match PyTorch's prediction (`decode_tokens` in metadata).

### Llaminar Snapshot Capture

Llaminar can also capture snapshots during inference for comparison:

#### Enable Snapshot Capture (E2E Build Required)

```bash
# Build with snapshot support
cmake -B build_v2_e2e -S src/v2 \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_PIPELINE_SNAPSHOTS=ON

cmake --build build_v2_e2e --parallel
```

#### Capture Snapshots in Code

```cpp
// Enable snapshot capture before forward pass
pipeline->enableSnapshotCapture("/tmp/llaminar_snapshots");

// Run inference
pipeline->forward(token_ids.data(), seq_len);

// Retrieve snapshots
auto keys = pipeline->getSnapshotKeys();
for (const auto& key : keys) {
    auto data = pipeline->getSnapshot(key);
    // Compare with PyTorch reference...
}
```

### Debugging Workflow for Inference Issues

When Llaminar produces unexpected output, follow this systematic approach:

#### Step 1: Verify with Greedy Sampling

```bash
# First, eliminate sampling randomness
./run_llaminar.sh -m model.gguf -p "Your prompt" -n 10 -t 0 2>&1 | tail -20
```

If output is garbage with `-t 0`, there's a real inference bug. If output is reasonable with `-t 0` but random without, that's expected probabilistic sampling behavior.

#### Step 2: Compare Top-5 Predictions

```bash
# Get Llaminar's top-5
LLAMINAR_LOG_LEVEL=TRACE ./run_llaminar.sh -m model.gguf -p "prompt" -n 1 -t 0 2>&1 | grep "Top-5"

# Compare to PyTorch's top-5 (from snapshots)
python -c "
import numpy as np
logits = np.load('pytorch_qwen2_snapshots/LM_HEAD.npy')[0, -1, :]
top5 = np.argsort(logits)[::-1][:5]
print('PyTorch Top-5:', list(top5))
print('Scores:', [logits[i] for i in top5])
"
```

#### Step 3: Identify Divergence Point

If top predictions don't match, run layer-by-layer comparison:

```bash
# Run E2E parity tests with verbose output
cd build_v2_e2e && ctest -R Qwen2FP32 -V
```

The test output shows cosine similarity per layer, helping identify where divergence begins.

#### Step 4: Check Token IDs Match

Ensure the tokenizer produces the same token IDs:

```bash
# Python tokenization
python -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')
print(tok.encode('The quick brown fox jumps over the lazy dog', add_special_tokens=False))
"
# Expected: [785, 3974, 13876, 38835, 34208, 916, 279, 15678, 5562]
```

### Common Investigation Pitfalls

1. **Forgetting `-t 0`**: Default temperature (0.8) produces random output. Always use greedy sampling when comparing to PyTorch ground truth.

2. **Prompt mismatch**: PyTorch snapshots are generated with a specific prompt. Using a different prompt in Llaminar will produce different (but possibly correct) output.

3. **Token ID mismatch**: If C++ uses different token IDs than Python (e.g., including BOS token), results won't match even if the pipeline is correct.

4. **MPI rank issues**: When running with MPI, only rank 0 does sampling. Other ranks must still participate in forward passes. Check for MPI deadlocks if inference hangs.

5. **Chat template confusion**: Instruct models expect chat-formatted input. Raw prompts may produce nonsensical output even when the pipeline is correct. The E2E tests use raw prompts (no chat template) for simplicity.

6. **Looking at wrong position**: For multi-token prompts, the "next token" prediction comes from the **last position's logits**, not the first.

### Running E2E Parity Tests

```bash
# Build with snapshot support
cmake -B build_v2_e2e_release -S src/v2 \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_PIPELINE_SNAPSHOTS=ON
cmake --build build_v2_e2e_release --parallel

# Run all E2E tests
ctest --test-dir build_v2_e2e_release -R "^V2_E2E_" --output-on-failure

# Run specific parity test
ctest --test-dir build_v2_e2e_release -R Qwen2FP32Parity -V
```

### Key Files Reference

| File | Purpose |
|------|---------|
| `python/reference/generate_qwen2_pipeline_snapshots.py` | Generate PyTorch snapshots |
| `python/reference/loaders/gguf_loader.py` | Load GGUF files into PyTorch |
| `python/reference/run_reference.py` | CLI for PyTorch inference |
| `tests/v2/e2e/qwen2/Test__Qwen2FP32Parity.cpp` | C++ E2E parity tests |
| `pytorch_qwen2_snapshots/` | Default snapshot output directory |
| `src/v2/pipelines/qwen/Qwen2Pipeline.cpp` | Llaminar pipeline implementation |