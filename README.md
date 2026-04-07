# Llaminar V2 — High-Performance LLM Inference Engine

Llaminar is a from-scratch LLM inference engine with an **operator-free, kernel-centric** architecture. It executes transformer models as declarative compute DAGs across CPU, CUDA, and ROCm devices with automatic Megatron-style tensor parallelism and pipeline parallelism.

## Features

- **Declarative Graph Execution** — `GraphOrchestrator` builds and executes `ComputeGraph` DAGs via `DeviceGraphExecutor`, with graph caching for decode mode
- **Multi-Backend GPU Inference** — CUDA (Ampere/Turing TensorCore GEMM, FlashAttention-2) and ROCm (MatrixCore GEMM, FlashAttention) with format-aware tile heuristics and stream-K persistent kernels
- **CPU Inference** — AVX-512 VNNI JIT-compiled GEMM, JIT attention microkernels (Xbyak), OpenBLAS/MKL fallback
- **Tensor Parallelism** — Automatic Megatron-style weight sharding across devices with NCCL, RCCL, PCIe BAR, UPI, MPI, and host-staging collective backends
- **Pipeline Parallelism** — Layer distribution across devices/ranks with named TP domains for heterogeneous PP+TP setups (e.g., CUDA + ROCm in one pipeline)
- **22+ Quantization Formats** — All GGUF formats: Q4_0/1, Q5_0/1, Q8_0/1, K-quants (Q2_K–Q8_K), importance quants (IQ1–IQ4), FP16, BF16, FP32
- **Qwen Model Family** — Qwen2.5 (0.5B–72B), Qwen3, and Qwen3.5 with hybrid GDN (Gated Delta Network) + full attention layers
- **Execution Modes** — Completion, interactive chat, single-shot chat, OpenAI-compatible HTTP server, and benchmark modes
- **KV Cache Quantization** — Configurable precision: FP32, FP16, Q8_1, Q16_1, TurboQuant TQ4 (rotation-based 4-bit), and heterogeneous TQ (TQ8 K + TQ4 V)
- **Weight Streaming** — VRAM-constrained inference with configurable memory budget, layer prefetching, and LRU/FIFO eviction
- **GPU-Side Sampling** — Greedy and top-k/top-p sampling on device, avoiding full logit D2H transfers
- **Built-in Profiling** — Benchmark mode (warmup + averaged runs), per-kernel timing, GPU event-based stage timing, Nsight ncu/nsys compatible
- **Parity Testing** — Layer-by-layer PyTorch ground truth comparison with cosine similarity, KL divergence, and top-K overlap metrics
- **Debug Infrastructure** — Stage dump framework (async binary tensor dumps), stage output print facility, tensor verification with automatic NaN/zero detection, transfer tracing

## Quick Start

```bash
# Build (Release mode, GPU optional)
cmake -B build_v2_release -S src/v2 -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build_v2_release --parallel

# Single-device inference
./build_v2_release/llaminar2 -m models/qwen2.5-0.5b-instruct-q4_0.gguf -p "Hello, world!" -n 50

# Explicit GPU device
./build_v2_release/llaminar2 -d cuda:0 -m model.gguf -p "Hello" -n 50

# Chat mode (applies chat template automatically)
./build_v2_release/llaminar2 --chat-single -m model.gguf -p "What is the capital of France?" -n 100

# Interactive chat (multi-turn)
./build_v2_release/llaminar2 --chat -m model.gguf

# OpenAI-compatible HTTP server
./build_v2_release/llaminar2 --serve -m model.gguf

# 2-way tensor parallelism
./build_v2_release/llaminar2 --tp-devices "cuda:0,cuda:1" -m model.gguf -p "Hello" -n 50

# Heterogeneous PP+TP (CUDA + ROCm)
./build_v2_release/llaminar2 \
  --define-domain "gpu_fast=cuda:0,cuda:1;backend=nccl" \
  --define-domain "gpu_slow=rocm:0,rocm:1;backend=rccl" \
  --pp-stage "0=gpu_fast:0-13" \
  --pp-stage "1=gpu_slow:14-27" \
  -m model.gguf -p "Hello" -n 50

# KV cache quantization (TurboQuant 4-bit)
./build_v2_release/llaminar2 --kv-cache-precision tq4 -m model.gguf -p "Hello" -n 50

# Benchmark mode (1 warmup + 3 averaged runs)
./build_v2_release/llaminar2 --benchmark -m model.gguf -n 128

# Dry-run to preview configuration
./build_v2_release/llaminar2 --dry-run -m model.gguf
```

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  CLI / YAML                                                                 │
│  OrchestrationConfigParser → OrchestrationConfig → ConfigValidator          │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Orchestration Runner                                                       │
│  IOrchestrationRunner → OrchestrationRunner                                 │
│    → ExecutionPlanBuilder → RankExecutionPlan (pre-parsed RuntimeConfig)     │
│    → Weight loading + sharding                                              │
│    → Build path: Single │ TP (MultiDeviceOrchestrator) │ PP (compiled)      │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Local Execution                                                            │
│  DeviceGraphOrchestrator (per-device inference, graph caching)              │
│    → Qwen2Graph (stateless graph builder)                                   │
│      → GraphSchema → GraphResolver → ComputeGraph (DAG of ComputeNodes)    │
│    → DeviceGraphExecutor (cohere → verify → execute → mark dirty → snap)   │
│    → BufferArena (centralized buffer management, coherence tracking)        │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Compute Stages                                                             │
│  RMSNorm │ FusedResidualNorm │ GatedRMSNorm │ QKNorm │ GEMM │ FusedQKV    │
│  FusedGateUp │ SwiGLU │ AttentionOutputGate │ RoPE │ Attention              │
│  GDNProjection │ GDNRecurrence │ ShortConv1d │ QGateSplit                   │
│  ResidualAdd │ Embedding │ LMHead │ QuantizeQ16_1 │ KVCacheAppend          │
│  Allreduce │ AllGather │ AllGatherV │ LocalPP/GlobalPP │ Send/Receive (PP)  │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Kernel Layer                                                               │
│  KernelFactory → ITensorGemm, ITensorAttention, ITensorRoPE, ...           │
│  ┌──────────────┐ ┌──────────────────┐ ┌──────────────────┐                │
│  │ CPU Kernels   │ │ CUDA Kernels     │ │ ROCm Kernels     │                │
│  │ AVX-512 VNNI  │ │ TensorCore WMMA  │ │ MatrixCore       │                │
│  │ JIT Attention  │ │ FlashAttention-2 │ │ FlashAttention   │                │
│  │ OpenBLAS/MKL  │ │ Stream-K GEMM    │ │ INT8 VNNI        │                │
│  └──────────────┘ └──────────────────┘ └──────────────────┘                │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Tensor Layer                                                               │
│  TensorBase → TypedTensorBase<Derived, DataType> (CRTP zero-overhead)      │
│  27 tensor types: FP32, BF16, FP16, Q8_0/1, Q4_0/1, IQ4_NL, Q6_K, ...    │
│  Per-tensor device coherence (host ↔ GPU automatic sync)                   │
│  UnifiedKVCache (POSITION_MAJOR / HEAD_MAJOR layouts)                      │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Backend + Collective Layer                                                 │
│  DeviceRegistry │ BackendManager │ IBackend (CPU, CUDA, ROCm)              │
│  BackendRouter → NCCL │ RCCL │ PCIeBAR │ UPI │ MPI │ Host                 │
│  ILocalTPContext │ ILocalPPContext │ GlobalTPContext                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Design Principles

- **No operator layer** — Graph stages orchestrate kernels directly via `KernelFactory`
- **Per-tensor device affinity** — Each tensor tracks its own host/device coherence state
- **Parse once, copy always** — `RuntimeConfig` is pre-parsed once, carried through the entire config chain
- **Declarative graphs** — `GraphSchema` + `GraphResolver` produce concrete `ComputeGraph` DAGs
- **Graph caching** — Decode-mode graphs (seq_len=1) are cached and reused, eliminating rebuild overhead

## Parallelism

### Tensor Parallelism (TP)

Megatron-style automatic weight sharding with proportional split support for heterogeneous devices:

| Weight | Sharding Mode | Description |
|--------|---------------|-------------|
| `attn_q`, `attn_k`, `attn_v` | COLUMN_PARALLEL | Split output dim (heads) |
| `attn_output` (Wo) | ROW_PARALLEL | Split input dim, allreduce after |
| `ffn_gate`, `ffn_up` | COLUMN_PARALLEL | Split output dim (d_ff) |
| `ffn_down` | INPUT_PARALLEL | Split input dim, allreduce after |
| `output` (LM head) | COLUMN_PARALLEL | Split vocab, allgather logits |
| GDN `attn_q`, `attn_k` | REPLICATE | Full copy (not head-divisible for GDN) |
| GDN `attn_v` | COLUMN_PARALLEL | Split V heads across devices |
| Norms, embeddings | REPLICATE | Full copy on each device |

```bash
# Simple TP (auto-detects devices)
./build_v2_release/llaminar2 -tp 2 -m model.gguf -p "Hello"

# Explicit devices with proportional weights (73%/27% for heterogeneous GPUs)
./build_v2_release/llaminar2 --tp-devices "cuda:0,rocm:0" --tp-weights "0.73,0.27" -m model.gguf -p "Hello"

# Preview TP placement
./build_v2_release/llaminar2 -tp 4 --explain-placement --dry-run -m model.gguf
```

### Pipeline Parallelism (PP)

Layer distribution across devices with named TP domains for heterogeneous setups:

```bash
# 2-stage PP with independent TP domains
./build_v2_release/llaminar2 \
  --define-domain "gpu_fast=cuda:0,cuda:1;backend=nccl" \
  --define-domain "gpu_slow=rocm:0,rocm:1;backend=rccl" \
  --pp-stage "0=gpu_fast:0-13" \
  --pp-stage "1=gpu_slow:14-27" \
  -m model.gguf -p "Hello" -n 50
```

### Collective Backends

| Backend | Use Case | Latency |
|---------|----------|---------|
| NCCL | All CUDA devices | ~5μs |
| RCCL | All ROCm devices | ~5μs |
| PCIe BAR | Mixed CUDA+ROCm (same node) | ~25μs |
| UPI | Cross-socket CPU | ~10μs |
| MPI | Cross-node distributed | ~10–50μs |
| Host | Fallback (any mix) | ~200μs |

## Supported Quantization Formats

| Format | Bits/Weight | Description |
|--------|-------------|-------------|
| Q4_0, Q4_1 | ~4.5 | 4-bit block quantization |
| Q5_0, Q5_1 | ~5.5 | 5-bit block quantization |
| Q8_0, Q8_1 | ~8.5 | 8-bit block quantization |
| Q2_K–Q8_K | Variable | K-quant formats |
| IQ1_M, IQ1_S | ~1.5 | Ultra-low-bit importance quants |
| IQ2_S, IQ2_XS, IQ2_XXS | ~2 | Low-bit importance quants |
| IQ3_S, IQ3_XXS | ~3 | Mid-bit importance quants |
| IQ4_NL, IQ4_XS | ~4 | Non-linear importance quants |
| FP16, BF16, FP32 | 16–32 | Full precision |

## Building

### Prerequisites

- CMake 3.20+, Ninja build system
- C++20 compiler (GCC 11+, Clang 14+)
- OpenMPI
- **Optional**: CUDA toolkit (for CUDA backend), ROCm (for ROCm backend)

### Build Commands

```bash
# Debug build (development, assertions enabled)
cmake -B build_v2 -S src/v2 -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build_v2 --parallel

# Release build (performance, assertions compiled out)
cmake -B build_v2_release -S src/v2 -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build_v2_release --parallel

# Integration build (Release optimizations + debug symbols + snapshots + assertions)
cmake -B build_v2_integration -S src/v2 -G Ninja -DCMAKE_BUILD_TYPE=Integration
cmake --build build_v2_integration --parallel

# With GPU backends
cmake -B build_v2_release -S src/v2 -G Ninja -DCMAKE_BUILD_TYPE=Release -DHAVE_CUDA=ON -DHAVE_ROCM=ON
cmake --build build_v2_release --parallel
```

### CMake Build Types

| Build Type | Optimization | Debug Symbols | Snapshots | Assertions | Use Case |
|------------|-------------|---------------|-----------|------------|----------|
| Debug | Off | Yes | Yes | Yes | Development, debugging |
| Release | Full (-O3) | No | No | No | Production, benchmarks |
| Integration | Full (-O3) | Yes | Yes | Yes | Unit/integration/parity tests |
| E2ERelease | Full (-O3) | No | Yes | No | End-to-end tests with snapshots |

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `HAVE_CUDA` | OFF | Enable CUDA backend |
| `HAVE_ROCM` | OFF | Enable ROCm backend |
| `ENABLE_SNAPSHOTS` | OFF | Enable tensor snapshot capture (auto-enabled for Debug/Integration) |

## Testing

```bash
# Unit tests (fast, no model loading — Integration build)
ctest --test-dir build_v2_integration -R "^V2_Unit_" --output-on-failure --parallel

# Integration tests (full pipeline with models)
ctest --test-dir build_v2_integration -R "^V2_Integration_" --output-on-failure --parallel

# Parity tests (Llaminar vs PyTorch reference)
ctest --test-dir build_v2_integration -R "^V2_Integration_Parity_" --output-on-failure

# End-to-end tests (server, full pipeline parity)
ctest --test-dir build_v2_e2e_release -R "^V2_E2E_" --output-on-failure

# Performance benchmarks (Release build)
ctest --test-dir build_v2_release -R "^V2_Perf_" --verbose
```

## Profiling and Benchmarking

```bash
# Standard benchmark (1 warmup + 3 averaged runs, greedy sampling)
./build_v2_release/llaminar2 --benchmark -m model.gguf -n 128

# Full profiling (kernel timing + executor overhead + GPU stage timing)
LLAMINAR_PROFILING=1 ./build_v2_release/llaminar2 --benchmark -m model.gguf -n 128

# GPU event-based per-stage timing
LLAMINAR_GPU_STAGE_TIMING=1 ./build_v2_release/llaminar2 --benchmark -m model.gguf -n 128

# NVIDIA Nsight profiling (use --no-mpi-bootstrap to avoid attaching to mpirun wrapper)
sudo /usr/local/cuda/bin/ncu --kernel-name "myKernel" --launch-skip 1 --launch-count 1 \
  ./build_v2_release/llaminar2 --no-mpi-bootstrap -d cuda:0 -m model.gguf -p "test" -n 5
```

## Project Structure

```
src/v2/
├── config/                    # OrchestrationConfig, TP/PP/Placement configs, ConfigValidator
├── inference/                 # IInferenceRunner interface and factory
├── interfaces/                # ICollectiveContext, IMultiDeviceOrchestrator, IMPIContext
├── execution/
│   ├── runner/                # IOrchestrationRunner, OrchestrationRunner
│   ├── factory/               # InferenceRunnerFactory
│   ├── mpi_orchestration/     # RankExecutionPlan, ExecutionPlanBuilder
│   ├── config/                # RuntimeConfig, ExecutionPolicy
│   ├── local_execution/
│   │   ├── orchestrators/     # DeviceGraphOrchestrator, MultiDeviceOrchestrator
│   │   ├── model/             # IModelExecutor, ModelExecutor
│   │   ├── graph/             # GraphSchema, GraphResolver, DeviceGraphExecutor, ComputeGraph
│   │   ├── coherence/         # StageCoherence, GpuCoherence (RAII wrappers)
│   │   └── collective/        # CollectiveContext
│   └── compute_stages/        # IComputeStage, all stage implementations
├── memory/                    # BufferArena, BufferId, CoherenceTracker, StageBoundBuffers
├── models/
│   ├── qwen/                  # Qwen2Graph, Qwen2GraphConfigBuilder, Qwen2BufferSpec
│   ├── qwen3/                 # Qwen3Schema (extends Qwen2 with per-head norm)
│   └── qwen35/                # Qwen35Graph, Qwen35Schema (hybrid GDN + full attention)
├── app/
│   └── modes/                 # ServerMode, InteractiveChatMode, SingleShotChatMode, BenchmarkMode
├── kernels/
│   ├── cpu/                   # AVX-512 VNNI GEMM, JIT attention (Xbyak), OpenBLAS, primitives
│   │   ├── jit/               # JIT infrastructure (RegisterGuard, RegisterAllocation, Xbyak)
│   │   ├── turboquant/        # TurboQuant rotation-based quantization (TQ3/TQ4/TQ8)
│   │   └── gdn/               # Gated Delta Network CPU kernels
│   ├── cuda/                  # TensorCore GEMM, FlashAttention-2, stream-K, ops
│   ├── rocm/                  # MatrixCore GEMM, FlashAttention, INT8 VNNI, ops
│   └── KernelFactory.h        # Centralized dispatch with caching
├── tensors/                   # ITensor, TensorBase, 27 typed tensor classes, UnifiedKVCache
├── transfer/                  # TransferEngine, TransferMethod (H2D/D2H/P2P/BAR dispatch)
├── collective/                # ILocalTPContext, BackendRouter, NCCL/RCCL/PCIeBAR/UPI/MPI/Host
├── backends/                  # IBackend, BackendManager, DeviceRegistry, DeviceId
├── loaders/                   # GGUF loading, WeightManager, WeightStreamer
└── utils/                     # MPIContext, MPITopology, Tokenizer, Sampler, logging

tests/v2/
├── unit/                      # Fast isolated tests (no model loading)
├── integration/               # Full pipeline tests with models
│   └── parity/                # PyTorch ground truth parity tests
├── e2e/
│   ├── parity/                # End-to-end parity tests
│   └── server/                # HTTP server end-to-end tests
├── performance/               # Benchmark and kernel perf tests
└── utils/                     # TestTensorFactory, test utilities
```

## Key Environment Variables

| Variable | Description |
|----------|-------------|
| `LLAMINAR_LOG_LEVEL` | Logging verbosity: ERROR, WARN, INFO, DEBUG, TRACE |
| `LLAMINAR_PROFILING` | Enable all profiling (kernel + executor + GPU stage timing) |
| `LLAMINAR_GPU_STAGE_TIMING` | GPU event-based per-stage timing |
| `LLAMINAR_VALIDATE_BUFFERS` | Buffer validation after stage execution (auto in Debug/Integration) |
| `LLAMINAR_STAGE_DUMP_ENABLED` | Dump stage input/output tensors to disk for debugging |
| `LLAMINAR_STAGE_OUTPUT_PRINT` | Print stage tensor samples to log output |
| `LLAMINAR_TRACE_TRANSFERS` | Trace host↔device memory transfers |
| `LLAMINAR_WEIGHT_STREAMING` | Enable weight streaming for VRAM-constrained inference |
| `LLAMINAR_STREAM_MEMORY_MB` | GPU memory budget for weight streaming cache (0 = auto) |

See `src/v2/utils/DebugEnv.h` for the full list.

## Documentation

| Document | Description |
|----------|-------------|
| [.github/copilot-instructions.md](.github/copilot-instructions.md) | Development guidelines, build system, testing, debugging |
| [.github/instructions/llaminar-v2-architecture.instructions.md](.github/instructions/llaminar-v2-architecture.instructions.md) | Detailed implementation architecture |
| [tests/v2/integration/parity/README.md](tests/v2/integration/parity/README.md) | Parity test framework documentation |

## License

MIT License — see LICENSE file for details.

## Author

David Sanftenberg
