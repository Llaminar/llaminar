# Llaminar V2 - High-Performance LLM Inference Engine

**Status**: Production-ready CPU inference with full tensor parallelism support

Llaminar is a high-performance LLM inference engine designed for CPU clusters with MPI-based tensor parallelism. It features a declarative graph execution system, automatic weight sharding, and support for a wide range of quantization formats.

## Current Features (December 2025)

- ✅ **Graph-based Execution**: `GraphOrchestrator` executes declarative compute DAGs
- ✅ **Megatron-style Tensor Parallelism**: Automatic weight sharding across MPI ranks
- ✅ **22+ Quantization Formats**: Q4_0, Q8_0, IQ4_NL, Q6_K, and many more
- ✅ **Qwen2 Model Family**: Full support for Qwen2.5 models (0.5B-72B)
- ✅ **AVX512-VNNI Optimized GEMM**: JIT-compiled quantized matrix multiplication
- ✅ **KV Cache Sharding**: Per-rank KV cache for memory efficiency
- ✅ **Benchmark Mode**: Built-in performance measurement with kernel profiling
- 🔄 **GPU Backends**: CUDA infrastructure in place, kernels in development

## Quick Start

```bash
# Build (Release mode for performance)
cmake -B build_v2_release -S src/v2 -DCMAKE_BUILD_TYPE=Release
cmake --build build_v2_release --parallel

# Run inference (auto-detects topology, self-launches MPI if needed)
./build_v2_release/llaminar2 -m models/qwen2.5-0.5b-instruct-q4_0.gguf -p "Hello, world!" -n 50

# Specify MPI process count manually
./build_v2_release/llaminar2 --mpi-procs 2 -m models/qwen2.5-7b-instruct-q4_0.gguf -p "Hello!" -n 50

# Benchmark mode
./build_v2_release/llaminar2 --benchmark -m models/qwen2.5-0.5b-instruct-q8_0.gguf -n 50

# Dry-run to see detected configuration
./build_v2_release/llaminar2 --dry-run -m models/qwen2.5-0.5b-instruct-q4_0.gguf
```

## Architecture Overview

Llaminar V2 uses a **kernel-centric, operator-free** architecture:

```
+-----------------------------------------------------------------------------+
|                          INFERENCE LAYER                                     |
|                                                                              |
|   createInferenceRunner()  -->  IInferenceRunner (GraphOrchestrator)        |
|                                      |                                       |
|                                      v                                       |
|                        +-------------------------+                          |
|                        |   GraphOrchestrator     |                          |
|                        |  (Declarative graphs)   |                          |
|                        +-------------------------+                          |
|                                      |                                       |
|                                      v                                       |
|                        +-------------------------+                          |
|                        |   ComputeGraph DAG      |                          |
|                        |  + GraphExecutor        |                          |
|                        +-------------------------+                          |
|                                                                              |
+-----------------------------------------------------------------------------+
                                      |
                                      v
+-----------------------------------------------------------------------------+
|                           KERNEL LAYER                                       |
|                                                                              |
|   KernelFactory  -->  ITensorGemm, ITensorAttention, IRMSNorm, etc.         |
|                                                                              |
|   +-----------------+  +-----------------+  +-----------------------------+  |
|   |  CPU Kernels    |  |  CUDA Kernels   |  |  Quantized Kernels         |  |
|   |  (oneDNN/BLAS)  |  |  (in progress)  |  |  (IQ4_NL, Q8_0, Q6_K...)   |  |
|   +-----------------+  +-----------------+  +-----------------------------+  |
|                                                                              |
+-----------------------------------------------------------------------------+
                                      |
                                      v
+-----------------------------------------------------------------------------+
|                           TENSOR LAYER                                       |
|                                                                              |
|   TensorBase  -->  FP32Tensor, BF16Tensor, IQ4_NLTensor, Q8_0Tensor, etc.   |
|                                                                              |
+-----------------------------------------------------------------------------+
```

### Key Components

| Component | Description |
|-----------|-------------|
| **GraphOrchestrator** | Executes declarative compute DAGs for transformer inference |
| **Qwen2Graph** | Builds attention and FFN graphs for Qwen2 models |
| **KernelFactory** | Centralized kernel dispatch with caching |
| **WeightManager** | Automatic tensor parallelism via weight sharding |
| **MPITopology** | Work distribution across MPI ranks |
| **UnifiedKVCache** | Sharded KV cache for memory efficiency |

## Project Structure

```
src/v2/
 inference/          # IInferenceRunner interface and factory
 execution/          # ComputeGraph, GraphExecutor, ComputeStages
 pipelines/          # Model-specific orchestrators (Qwen2)
   └── qwen/           # GraphOrchestrator, Qwen2Graph, Qwen2BufferSpec
 kernels/            # Compute kernels
   ├── cpu/            # CPU kernels (GEMM, attention, primitives)
   └── cuda/           # CUDA kernels (in development)
 tensors/            # Tensor types (FP32, BF16, quantized)
 loaders/            # GGUF model loading, WeightManager
 utils/              # MPIContext, MPITopology, logging, etc.
 backends/           # Device abstraction (CPU, CUDA, ROCm)

tests/v2/
 unit/               # Fast unit tests (no model loading)
 integration/        # Full pipeline tests with models
 e2e/                # End-to-end parity tests vs PyTorch
 performance/        # Benchmark tests
```

## Tensor Parallelism

Llaminar implements **Megatron-style tensor parallelism** with automatic weight sharding:

| Weight | Sharding Mode | Description |
|--------|---------------|-------------|
| `attn_q.weight`, `attn_k.weight`, `attn_v.weight` | COLUMN_PARALLEL | Split output dim (heads) |
| `attn_output.weight` (Wo) | ROW_PARALLEL | Split input dim, allreduce after |
| `ffn_gate.weight`, `ffn_up.weight` | COLUMN_PARALLEL | Split output dim (d_ff) |
| `ffn_down.weight` | INPUT_PARALLEL | Split input dim, allreduce after |
| `output.weight` (LM head) | COLUMN_PARALLEL | Split vocab, allgather logits |
| Norms, embeddings | REPLICATE | Full copy on each rank |

**Memory Savings**: With 2 MPI ranks, sharded weights use ~50% less memory per rank.

```bash
# Automatic sharding when world_size > 1
./build_v2_release/llaminar2 --mpi-procs 2 -m model.gguf -p "Hello"
```

## Supported Quantization Formats

Llaminar supports all major GGUF quantization formats:

| Format | Bits/Weight | Description |
|--------|-------------|-------------|
| Q4_0, Q4_1 | 4.5 | 4-bit quantization |
| Q5_0, Q5_1 | 5.5 | 5-bit quantization |
| Q8_0, Q8_1 | 8.5 | 8-bit quantization |
| Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K | Variable | K-quant formats |
| IQ1_M, IQ1_S, IQ2_S, IQ2_XS, IQ2_XXS | 1-2 | Importance-based low-bit |
| IQ3_S, IQ3_XXS, IQ4_NL, IQ4_XS | 3-4 | Importance-based mid-bit |
| FP16, BF16, FP32 | 16-32 | Full precision |

## Building

### Prerequisites

- CMake 3.16+
- C++17 compiler (GCC 9+, Clang 10+)
- OpenMPI or MPICH
- OpenBLAS or oneDNN

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

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `HAVE_CUDA` | OFF | Enable CUDA backend |
| `HAVE_ROCM` | OFF | Enable ROCm backend |
| `ENABLE_PIPELINE_SNAPSHOTS` | OFF | Enable tensor snapshot capture for E2E tests |

## Testing

```bash
# Unit tests (fast, no model loading)
ctest --test-dir build_v2 -R "^V2_Unit_" --output-on-failure --parallel

# Integration tests (require models)
ctest --test-dir build_v2_release -R "^V2_Integration_" --output-on-failure

# E2E parity tests (compare against PyTorch reference)
ctest --test-dir build_v2_e2e_release -R "^V2_E2E_" --output-on-failure

# Performance benchmarks
ctest --test-dir build_v2_release -R "^V2_Perf_" --verbose
```

## Benchmarking

```bash
# Run benchmark with default settings
./build_v2_release/llaminar2 --benchmark -m model.gguf

# With kernel profiling
LLAMINAR_PROFILE_KERNELS=1 ./build_v2_release/llaminar2 --benchmark -m model.gguf -n 50

# Custom prompt and decode length
./build_v2_release/llaminar2 --benchmark -m model.gguf -p "Your prompt here" -n 100
```

**Benchmark Output:**
```
+--------------------------------------------------------------+
|                    BENCHMARK RESULTS                         |
|              (average of 3 runs after warmup)                |
+--------------------------------------------------------------+
| PREFILL PHASE                                                |
|   Tokens:           596 tokens                               |
|   Time:         1959.28 ms                                   |
|   Throughput:    304.19 tok/s                                |
+--------------------------------------------------------------+
| DECODE PHASE                                                 |
|   Tokens:           128 tokens                               |
|   Time:         2370.41 ms                                   |
|   Throughput:     54.00 tok/s                                |
+--------------------------------------------------------------+
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `LLAMINAR_LOG_LEVEL` | Logging verbosity (ERROR, WARN, INFO, DEBUG, TRACE) |
| `LLAMINAR_PROFILE_KERNELS` | Enable per-kernel timing in benchmark mode |
| `LLAMINAR_SNAPSHOT_TENSOR_DUMP` | Dump tensors to disk for debugging |
| `OMP_NUM_THREADS` | OpenMP thread count (auto-configured by llaminar2) |

## Documentation

- **Architecture Guide**: [.github/instructions/llaminar-architecture-v2.instructions.md](.github/instructions/llaminar-architecture-v2.instructions.md)
- **Development Guidelines**: [.github/copilot-instructions.md](.github/copilot-instructions.md)
- **Distributed Architecture**: [docs/v2/DISTRIBUTED_ARCHITECTURE_IMPLEMENTATION.md](docs/v2/DISTRIBUTED_ARCHITECTURE_IMPLEMENTATION.md)

## License

MIT License - see LICENSE file for details.

## Author

David Sanftenberg

---

**Note**: The legacy V1 codebase in `src/` has been removed. All development now uses the V2 architecture in `src/v2/`.
