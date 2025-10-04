# Llaminar LLM Inference Engine - Architecture Documentation

*Last Updated: October 3, 2025*

## Overview

Llaminar is a high-performance, MPI-first LLM inference engine focused on low‑latency decode and scalable prefill. The original generic compute graph and non‑MPI kernels have been fully retired; all production execution flows through **MPI-aware kernels**, an **Abstract Pipeline** adapter layer, and a centralized **adaptive backend selector** (OpenBLAS vs COSMA) tuned for sequence length and operation size.

Key pillars:
- **Abstract Pipeline Adapters**: Model‑family specific orchestration (e.g. Qwen adapter) implementing prefill + decode lifecycle over a fixed semantic stage graph (RMSNorm → Attention primitives → MLP Gate/Up/SwiGLU/Down → Residuals → Output Projection).
- **Adaptive MatMul**: Single decision point (`adaptive_matmul.h`) governing local vs COSMA distributed matmuls; avoids policy drift.
- **COSMA Prefill Manager**: High‑throughput large prompt (prefill) GEMMs with fused RMSNorm+QKV path and validation / orientation diagnostics.
- **Tensor Sharding**: Current column (feature) partition for linear projections & planned evolution to hybrid 1D→2D sharding for future multi‑node scaling (see Roadmap).
- **Centralized Environment Snapshot**: `debugEnv()` captures all tuning / debug flags once (no repeated `getenv` in hot loops).
- **Observability**: Structured perf counters, stage timers (norm, attention, linear, activation), optional per-layer token diffs, dequant stats, COSMA tile validation & distributed GEMM forensic logs.

## Core Design Principles (Current State)
1. **MPI Everywhere**: All compute kernels derive from `MPIKernelBase`; no legacy single‑process fallbacks remain.
2. **Semantic Stages Over Generic Graphs**: Explicit transformer stages replace run‑time node scheduling overhead.
3. **Single Policy Surface**: Backend selection & tuning centralized (predictable & testable).
4. **Data Locality First**: Column partition weight shards + fused prefill pathways minimize movement.
5. **Graceful Degradation**: COSMA failures auto‑fallback to OpenBLAS with logged reason.
6. **Deterministic Debugging**: Environment snapshot + opt‑in reference comparisons (RMSNorm, attention output, FFN) for surgical diagnostics.
7. **Incremental Evolution**: Design allows swapping in 2D sharding & mixed precision without re‑plumbing pipeline logic.

## Architecture Components

### 1. Entry Point & Orchestration

**File**: `src/main.cpp` (219 lines)
- **Purpose**: Application entry point and execution orchestration
- **Key Features**:
  - MPI initialization with thread support
  - 7-stage execution pipeline
  - Exception handling and graceful shutdown
  - Performance measurement and reporting

**Execution Flow (Current)**:
1. Parse CLI → build `LlaminarParams` (includes `--kv-stats`, verbosity, model path).
2. Initialize logging + environment snapshot (`debugEnv()` lazy init).
3. Detect topology (NUMA, cores) & optionally print.
4. Register MPI kernels (attention, linear, rmsnorm, residual, swiglu) & adaptive backend.
5. Load GGUF model → weight tensors (auto layout decisions) & create model adapter.
6. Instantiate distributed Qwen pipeline (formerly MPITransformerPipeline + adapter) – adapter layer removed; factory now returns DistributedTransformerPipeline directly.
7. Execute prefill (prompt) then iterative decode until token budget / stop condition.
8. Emit KV cache summary if requested; output perf counters & adaptive backend stats.

### 2. Command Line Interface

**Files**: `src/argument_parser.h/cpp`
- **Class**: `ArgumentParser`
- **Structure**: `LlaminarParams`
- **Features**:
  - POSIX-style argument parsing (`-v`, `--verbose`, etc.)
  - Multi-level verbosity (`-v`, `-vv`, `-vvv`)
  - Model file specification (`-m`, `--model`)
  - System configuration flags
  - Legacy COSMA parameters
  - Comprehensive help and version information

**Supported Arguments**:
```bash
# Model and Logging
-m, --model <file>     # GGUF model file path
-v/-vv/-vvv           # Verbosity levels (INFO/DEBUG/TRACE)

# System Configuration  
--print-topology      # Display system topology
--enable-hyperthreading # Use HT cores
--detect-gpus         # Enable GPU detection

# Performance
--profile             # Enable kernel profiling
--validate            # Enable result validation
--matrix-size <size>  # Matrix dimensions for benchmarks
--repeat <num>        # Benchmark iterations
```

### 3. Logging System

**Files**: `src/logger.h`, `src/log_level.h`
- **Pattern**: Header-only singleton with macros
- **Levels**: ERROR(0), WARN(1), INFO(2), DEBUG(3), TRACE(4)
- **Features**:
  - Timestamped output with millisecond precision
  - File and line number tracking
  - Environment variable configuration
  - Thread-safe singleton pattern
  - Convenient macros (`LOG_INFO`, `LOG_DEBUG`, etc.)

**Usage Example**:
```cpp
LOG_INFO("Model loaded successfully");
LOG_DEBUG("Processing iteration " << i << "/" << total);
LOG_ERROR("Failed to load model: " << filename);
```

### 4. System Topology Detection

**Files**: `src/topology_manager.h/cpp`, `src/common.h/cpp`
- **Classes**: `TopologyManager`, `CPUTopology`, `SystemTopology`
- **Features**:
  - CPU architecture detection (sockets, cores, hyperthreading)
  - NUMA topology mapping
  - GPU device enumeration (CUDA/ROCm support planned)
  - Memory capacity reporting
  - MPI environment information

**Detection Capabilities**:
- **CPU**: Socket count, cores per socket, hyperthreading status
- **NUMA**: Node count, CPU affinity, memory distribution  
- **Memory**: Total and per-NUMA-node capacity
- **GPU**: Device enumeration (framework-agnostic detection)
- **MPI**: Rank, size, process distribution

**Example Output**:
```
=== CPU Topology ===
Total CPUs: 112, Physical cores: 56, Sockets: 2
Hyperthreading: Yes, Using: No

=== NUMA Topology ===  
Node 0: 56 CPUs, 376 GB memory
Node 1: 56 CPUs, 377 GB memory
```

### 2. Hybrid / Sharded Tensor Architecture

**Files**: `src/tensors/tensor_base.h`, `src/tensors/tensor_factory.cpp`, `src/tensor.h`
- **Classes**: `TensorBase`, `SimpleTensor`, `COSMATensor`, `TensorFactory`
- **Pattern**: Abstract interface with concrete implementations
- **Purpose**: Zero-copy COSMA optimization with legacy compatibility

**Key Components**:

#### TensorBase Abstract Interface
```cpp
class TensorBase {
public:
    virtual const std::vector<int>& shape() const = 0;
    virtual float* data() = 0;
    virtual const float* data() const = 0;
    virtual std::string type_name() const = 0;
    virtual bool is_distributed() const = 0;
};
```

#### SimpleTensor (Minimal Host Row-Major)
- Keeps row‑major buffers for small ops & latency‑critical decode.
- Still used for many activation intermediates (seq_len × hidden_size) to minimize transformation overhead.

#### COSMATensor (COSMA Optimization)
- **Purpose**: Direct integration with `cosma::CosmaMatrix<float>`
- **Data Storage**: COSMA-optimized distributed layout
- **Use Cases**: Large matrices (≥256×256), multi-process operations
- **Performance**: Zero-copy access to COSMA operations

#### TensorFactory (Smart Selection)
- **Auto-Selection Logic**:
  - **Matrix Size**: COSMATensor for matrices ≥256×256 elements
  - **MPI Context**: COSMATensor when multiple processes available
  - **Operation Type**: COSMATensor preferred for matmul/attention
  - **Fallback**: SimpleTensor for compatibility

**Usage Examples**:
```cpp
// Automatic tensor selection
auto tensor = TensorFactory::create_auto({1024, 1024});
// → COSMATensor in MPI environment

// Explicit type creation
auto simple = TensorFactory::create_simple({512, 512});
auto cosma = TensorFactory::create_cosma({2048, 2048}, "matmul_A", mpi_rank);

// Legacy compatibility
std::shared_ptr<Tensor> legacy = std::make_shared<Tensor>({256, 256});
auto upgraded = TensorFactory::from_tensor(legacy);
```

**Performance Benefits**:
- **Zero-Copy COSMA**: Direct `cosma::multiply()` calls without data copying
- **Automatic Optimization**: Large matrices automatically use COSMA layout
- **Legacy Compatibility**: No performance penalty for existing code
- **Smart Conversion**: Only copies data when necessary

### 3. Abstract Pipeline Architecture

The legacy generic compute graph (ComputeGraph / ComputeNode / MatMulNode) has been removed. It has been replaced by a higher‑level, purpose‑built inference pipeline abstraction that maps directly onto transformer model structure and execution phases.

**Key Components**:
- `AbstractPipeline` (interface): Defines lifecycle (prefill / decode), tensor staging, and logits access.
- Concrete adapter (e.g. `QwenPipelineAdapter`): Specializes AbstractPipeline for specific model families (Qwen2.5 today) translating model config + weights into orchestrated kernel invocations.
- `DistributedTransformerPipeline` (formerly `MPITransformerPipeline`): Unified distributed transformer implementation (adapters removed).
- `cosma_prefill_manager`: Orchestrates large prefill GEMMs (possibly distributed) and feeds activations into attention / MLP kernels.

**Execution Phases**:
1. **Prefill**: Fused RMSNorm+QKV (COSMA path if threshold & env allow) → attention primitives → MLP (gate/up/down) with adaptive backend per matmul.
2. **Decode**: Single token; all GEMMs forced OpenBLAS (small shapes) & minimal allocations; KV cache append + attention over growing context.
3. **Output Projection**: Adaptive matmul (prefill may use COSMA, decode stays OpenBLAS) → logits provided to sampler layer (external / future integration).

**Design Principles vs Removed Graph**:
- Eliminates generic node scheduling overhead in favor of fixed semantic stages (clear perf boundaries: attention phases, MLP phases).
- Centralized environment snapshot (`debug_env`) governs all hot path toggles (no per-node ad‑hoc env checks).
- Explicit performance counters instrumentation at semantic phases (RMSNorm, Attention QK, Attention Apply, MLP Gate/Up/Down) replaces per-node timing.
- Simplifies memory lifetime: tensors owned / recycled within pipeline stages instead of heap of generic node objects.

**Benefits**:
- Eliminated graph scheduler overhead & dynamic allocations churn.
- Uniform perf instrumentation surface (matmul counters, stage timers).
- Consistent environment gating (no latent `getenv` in loops).
- Simplified feature rollout (add adapter, not node graph rewrites).

**Migration Summary**:
- Removed files: `src/graph_compute.h`, `src/graph_compute.cpp`, `tests/test_graph.cpp`.
- Deprecated concepts: generic topology sort, per-node cycle detection, MatMulNode micro-benchmark path.
- Replacement: direct pipeline invocation inside `main.cpp` (abstract or legacy transformer), with fallback benchmark path removed.

**Future Extensions**:
- Plugin registry for model-family adapters.
- Mixed-precision policy injection (prefill vs decode) at pipeline boundary.
- Streaming KV cache compaction & eviction policies integrated into pipeline state.


### 4. Adaptive MatMul & Prefill Backend

**File**: `src/adaptive_matmul.h`
**Concepts**:
- `AdaptiveMatMulManager` chooses backend per (m,n,k,is_prefill, MPI size, env overrides).
- Prefill: Potential COSMA offload if seq_len ≥ threshold & world_size>1.
- Decode: Always OpenBLAS (saves collective overhead > latency).
- Supports batched Q/K/V path (currently OpenBLAS-loop; future fused COSMA batch candidate).

**File**: `src/cosma_prefill_manager*`
- Fused RMSNorm + QKV extraction with orientation / validation hooks.
- Unified COSMA strategy selection to prevent inconsistent tilings across operands.
- Optional tile verification, reconstruction diagnostics & orientation auto-fix env flags.

**Recent Extensions**:
- Output projection & all FFN (gate/up/down) matmuls now routed through adaptive backend (COSMA allowed on large prefill only).
- Performance counters record every GEMM (size, backend, time) for post-run summary.

**Fallback Semantics**:
1. Attempt COSMA via prefill manager.
2. On exception or validation failure → log + single-rank OpenBLAS fallback transparently.

### 5. Distributed Linear Projection Kernel

**File**: `src/kernels/MPILinearKernel.cpp`
**Role**: Column-partitioned dense layer implementation used when adaptive path selects OpenBLAS and we want distributed weight shards.
**Flow**: Distribute weight slice → local GEMM (OpenBLAS) → Allgatherv assemble full output rows → optional bias.
**Adaptive Interaction**: Invokes `adaptiveMatMul(... distributed_partition=true)` to ensure COSMA is not (incorrectly) applied to partial matrices.

### 6. Attention Primitives
- Fused QKV path (prefill COSMA) or kernel-based multi-head attention (decode & small prefill).
- RoPE applied per head with direct loop (possible future vectorization / LUT sincos reuse).
- All reductions / softmax executed locally (causal masking enforced row-wise); future: distributed softmax for large multi-rank KV spans.

### 7. KV Cache Management
**State**: `KVCacheState` tracked in `AbstractPipeline` & CLI flag `--kv-stats` prints usage & capacity.
**Growth**: Capacity ensured before decode; parity tests validate state after prefill vs incremental decode path.
**Future**: Segment compaction & eviction policies (LRU / sliding window) pluggable at pipeline layer.

### 8. Environment & Observability
**Central Snapshot**: `debugEnv()` groups: attention, baseline, cosma, adaptive, pipeline, linear, embedding, layer_capture, prefill_debug, etc.
**Rules**: No hot path `std::getenv`; all flags registered & typed (bool/int/strings) once.
**Diagnostics**:
- Per-stage diff (optional reference recomputation) with rel_l2 & top sample logging.
- Row capture for RMSNorm & FFN forensic modes.
- COSMA orientation / reconstruction debug channels.
- Perf counters + adaptive backend aggregated summary.

### 9. Model Loading & Quantization
**File**: `src/model_loader.*`
- Parses GGUF → instantiates weight tensors (Simple/COSMA) based on size & env.
- Supports quant formats (Q4K / Q6K etc.) with dequant stats & anomaly detection flags.
**Repacker**: Performs any layout adaptation required by fused prefill manager (e.g., contiguous block ordering for QKV concatenated strategies).

### 10. Testing Infrastructure (Current Representative Set)
Key test families (non-exhaustive):
- `test_prefill_attention_golden`, `test_cosma_prefill_*`: Fused COSMA correctness & stats.
- `test_adaptive_matmul*`: Backend decision correctness & performance constraints.
- `test_mpi_transformer_pipeline`, `test_abstract_pipeline_parity`: Adapter parity & end-to-end invariants.
- `test_kv_cache_growth*`: KV capacity correctness across modes.
- `test_rmsnorm_*`, `test_attention_*`: Primitive parity & edge cases.
- `test_tp_*`, `test_mlp_tp_parity`: Tensor partition correctness.

Removed tests: `test_graph.cpp`, `LinearKernelTest` (documented as historical in guidelines).

### 11. Performance Strategy Summary
| Phase | Typical Matmuls | Backend Policy | Rationale |
|-------|-----------------|----------------|-----------|
| Prefill (short) | many small/medium | OpenBLAS (single/multi-thread) | COSMA overhead not amortized |
| Prefill (large ≥4K tokens) | tall skinny (seq_len × hidden) * (hidden × proj) | COSMA | Collective reuse + fused QKV normalization |
| Decode | 1 × hidden * hidden × proj | OpenBLAS single-thread (often) | Min latency, avoids collectives |
| FFN large prefill | seq_len × hidden * hidden × d_ff | COSMA candidate | Throughput scaling |
| LM Head (future) | seq_len × hidden * hidden × vocab | Policy TBD (likely still local) | Avoid huge all-gather unless 2D shard implemented |

### 12. Tensor Sharding Roadmap (Forward Looking)
Current: 1D column partition for linear weights; activations replicated.
Planned Phases:
1. Activation micro-sharding for very large batch prefill (reduce per-rank memory).
2. 2D block cyclic strategy for extreme vocab or d_ff expansions (align w/ COSMA tiling to eliminate post-gather).
3. Owner-aware KV cache distribution (shard past sequence to reduce memory duplication) + distributed attention softmax.
4. Mixed precision (FP16 / BF16) activation buffers with on-the-fly dequant for matmuls.

### 13. Future Enhancements
- 2D sharded matmul path unifying MPILinearKernel + COSMA strategy (remove gather step where possible).
- Adaptive decode micro-batching (multiple next-token candidates) with latency guardrails.
- Runtime reconfiguration: dynamic environment snapshot refresh & remote control channel.
- Streaming KV compaction & eviction strategies.
- Quantized fused prefill (direct dequant into COSMA layout without intermediate float buffer).

### 14. Current vs Target End-State Snapshot
| Aspect | Current | Target |
|--------|---------|--------|
| Pipeline | Abstract adapters + legacy MPI pipeline (deprecated) | Pure adapters (legacy removed) |
| MatMul Policy | Central adaptive manager | Same + 2D shard extension |
| Attention Prefill | Fused RMSNorm+QKV (COSMA) + local softmax | Fully distributed softmax + partial KV ownership |
| Linear Sharding | 1D column gather | 1D/2D hybrid (gather-less) |
| KV Cache | Replicated per rank | Sharded + eviction/compaction |
| Mixed Precision | Primarily FP32 (with weight quant) | Activation FP16/BF16 + selective FP32 accum |
| Env Flags | Static snapshot at init | Hot-reloadable snapshot |

---
**Deprecations**:
- Compute Graph (removed)
- Non-MPI kernels (removed)
- `DistributedTransformerPipeline` (successor to deprecated `MPITransformerPipeline`; extend this for distributed transformer behavior)
- Legacy MatMulKernel (replaced by adaptive path + MPILinearKernel + COSMA prefill manager)

**Authoritative Hot Paths**:
1. `executePrefillAttentionCosma` (fused + distributed GEMMs)
2. FFN matmuls via adaptive backend (gate/up/down)
3. Attention decode kernel (MPI primitives + RoPE + softmax)
4. Output projection adaptiveMatMul

Maintain invariants:
- No direct `getenv` in hot loops.
- All distributed collectives bracketed by safe ordering (barriers where needed) to avoid hidden hangs.
- Fallback to OpenBLAS must never silently change numerical rank agreement; diffs logged if validation toggles enabled.

This document supersedes any earlier graph-centric or non‑MPI kernel guidance.

### 8. Model Loading System

**Files**: `src/model_loader.h/cpp`
- **Class**: `ModelLoader`
- **Format**: GGUF (GPT-Generated Unified Format)
- **Features**:
  - GGUF file parsing and validation
  - Metadata extraction (architecture, parameters)
  - **Hybrid Tensor Creation**: Automatic selection of optimal tensor types
  - Tensor loading with format conversion
  - Quantization support (Q8_0 implemented)

**GGUF Support**:
- **Architectures**: Qwen2.5, LLaMA family support
- **Quantization**: Q8_0, F16, F32 (extensible)
- **Metadata**: Model parameters, tokenizer info, training details
- **Validation**: Magic number verification, version compatibility
- **Tensor Optimization**: Large weight matrices automatically use COSMATensor

### 9. Data Format Conversion

**Files**: `src/repacker.h/cpp`
- **Class**: `Repacker`
- **Purpose**: Convert between GGUF and hybrid tensor formats
- **Features**:
  - Memory layout transformation
  - Type conversion (F16 ↔ F32, quantized formats)
  - **Tensor Type Selection**: Automatic SimpleTensor vs COSMATensor choice
  - Efficient memory management
  - Distributed data placement

### 10. Build System & Dependencies

**File**: `CMakeLists.txt`
- **Pattern**: Modern CMake with submodules
- **Library Structure**: Core library + executables
- **Dependencies**:
  - **COSMA**: High-performance matrix operations
  - **GGML/LLaMA.cpp**: Model format support and inference kernels
  - **MPI**: Distributed computing (OpenMPI)
  - **OpenMP**: Shared-memory parallelism
  - **NUMA**: Memory affinity management
  - **CUDA/ROCm**: GPU acceleration (optional)

**Build Targets**:
```bash
llaminar_core    # Core library with all components
llaminar         # Main executable
test_*          # Unit test executables
```

## Data Flow Architecture

### Inference Pipeline (Planned)

1. **Model Loading**: GGUF → Parsed tensors → Distributed placement
2. **Input Processing**: Tokenization → Embeddings → Attention preparation  
3. **Forward Pass**: Transformer blocks → Matrix operations → Activations
4. **Output Generation**: Logits → Sampling → Token generation
5. **Result Collection**: Distributed gather → Response formatting

### Current State (Abstract Pipeline)

1. **Initialization**: MPI setup → Topology detection → Environment snapshot
2. **Model Load**: GGUF parse → Weight tensors (auto Simple/COSMA selection) → Optional adapter wrapping
3. **Prefill**: Abstract pipeline issues RMSNorm, QKV projection, attention primitives, MLP phases (distributed prefill manager engaged above thresholds)
4. **Decode Loop** (interactive / generation): Repeated attention + MLP using latency‑optimized local backends
5. **Output**: Logits retrieval → Sampling / chat interface → Optional validation & perf summary

## Performance Characteristics

### Scalability
- **MPI Distributed**: Multi-node prefill + optional future distributed decode
- **NUMA Aware**: Memory locality & process pinning still handled before pipeline creation
- **Thread Parallel**: OpenMP within each rank; adaptive single vs multi-thread based on op size

### Memory Management
- **Hybrid Tensor System**: Automatic Simple vs COSMA tensor selection remains intact
- **Prefill Working Set Control**: Environment-governed memory caps (e.g., `LLAMINAR_COSMA_MAX_RESIDENT_MB`) consulted by prefill manager
- **KV Cache Growth**: Managed outside generic graph; pipeline directly appends to caches with validation tests ensuring shape parity
- **Format Optimization**: COSMA layout for optimal cache utilization
- **Legacy Compatibility**: SimpleTensor maintains existing memory patterns
- **Smart Allocation**: TensorFactory selects optimal memory layout
- **Quantization**: Reduced precision for memory efficiency

### Compute Optimization
- **COSMA Integration**: State-of-the-art matrix multiplication algorithms
- **Kernel Registration**: Pluggable optimization for different operations
- **Topology Awareness**: Hardware-specific optimizations

## Testing Infrastructure

**Directory**: `tests/`
- **test_basic.cpp**: MPI initialization and basic functionality
- **test_numa.cpp**: NUMA topology detection and affinity
- **test_cosma.cpp**: Matrix multiplication and COSMA integration
- **test_graph.cpp**: Compute graph construction and execution

**Test Coverage**:
- Component initialization and configuration
- MPI communication and coordination
- System topology detection accuracy
- Matrix operation correctness and performance

## Development Patterns

### Error Handling
- Exception-based error propagation
- MPI-aware error coordination
- Graceful degradation for optional features
- Comprehensive logging for debugging

### Memory Management
- RAII patterns for resource management
- Smart pointers for automatic cleanup
- MPI memory coordination
- NUMA-aware allocation strategies

### Extensibility Points
- **Kernel Registration**: Add new operations via inheritance
- **Model Formats**: Extend ModelLoader for new formats
- **Topology Detection**: Platform-specific detection modules
- **Communication**: Custom MPI communication patterns

## Configuration & Environment

### Environment Variables
```bash
LLAMINAR_LOG_LEVEL    # Override default log level
MPI_THREAD_MULTIPLE   # Enable MPI threading support
OMP_NUM_THREADS       # Control OpenMP parallelism
CUDA_VISIBLE_DEVICES  # GPU device selection
```

### CMake Configuration
```bash
-DCMAKE_BUILD_TYPE=Debug|Release
-DENABLE_CUDA=ON|OFF
-DENABLE_ROCM=ON|OFF
-DCOSMA_SCALAPACK_LINK_LIBRARIES=<path>
```

## Future Architecture Enhancements

### Planned Components
1. **Attention Kernels**: Multi-head attention with hybrid tensor optimization
2. **Transformer Blocks**: Complete layer implementations using TensorBase interface
3. **Advanced Tensor Operations**: Extend hybrid system to all kernel types
4. **Communication Layer**: Optimized MPI communication patterns
5. **Inference Engine**: Complete LLM inference pipeline with zero-copy optimization
6. **Model Zoo**: Pre-trained model repository integration

### Performance Optimizations
1. **GPU Acceleration**: CUDA/ROCm kernel implementations
2. **Mixed Precision**: FP16/BF16 computation paths
3. **Pipeline Parallelism**: Layer-wise distribution
4. **Tensor Parallelism**: Within-layer distribution
5. **Memory Optimization**: KV-cache management

### Scalability Improvements
1. **Dynamic Load Balancing**: Adaptive work distribution
2. **Hierarchical Communication**: Optimized MPI topologies
3. **Asynchronous Execution**: Overlapped computation/communication
4. **Elastic Scaling**: Runtime process addition/removal

## Canonical Runtime Configuration

### Optimal Launch Settings

Llaminar includes a canonical launch script with empirically-optimized settings:

```bash
# Always use the canonical launcher
./run-llaminar.sh [arguments]

# The script automatically configures:
# - OpenMP: Auto-detected cores per socket, socket placement, close binding
# - MPI: 1 process per socket, memory pinning, NUMA-aware binding  
# - Threading: Single-threaded for small ops, multi-threaded for medium, distributed for large
# - Topology Detection: Mirrors C++ logic from src/common.cpp for consistent results
```

### Environment Configuration

**OpenMP Settings** (automatically configured):
```bash
OMP_NUM_THREADS=<detected>       # Auto-detected physical cores per socket
OMP_PLACES=sockets               # Thread placement strategy
OMP_PROC_BIND=close              # Bind threads close together
KMP_AFFINITY=granularity=fine,compact,1,0  # Intel threading
KMP_BLOCKTIME=0                  # Minimize thread blocking
```

**MPI Settings** (automatically configured):
```bash
OMPI_MCA_mpi_leave_pinned=1                     # Memory pinning
OMPI_MCA_btl_vader_single_copy_mechanism=none   # NUMA optimization
OMPI_MCA_btl_openib_allow_ib=1                  # InfiniBand support
```

### System Requirements

- **CPU**: Multi-socket x86_64 with NUMA support
- **Memory**: 16GB+ RAM, preferably balanced across NUMA nodes
- **MPI**: OpenMPI 4.0+ with thread support
- **OpenMP**: libgomp or Intel OpenMP runtime
- **Optimal**: 2-socket system, 1 MPI process per socket

## Usage Examples

### Basic Execution
```bash
# Topology detection and system info
./run-llaminar.sh -v --print-topology

# Performance benchmarking
./run-llaminar.sh -vv --matrix-size 2048

# GPU detection and profiling
./run-llaminar.sh --detect-gpus --profile --trace
```

### Model Inference
```bash
# Load and run inference with Qwen 2.5 model
./run-llaminar.sh -m models/qwen2.5-0.5b-instruct-q4_0.gguf -v

# Verbose inference with profiling
./run-llaminar.sh -m models/qwen2.5-0.5b-instruct-q4_0.gguf -vv --profile
```

### Advanced Usage (Manual MPI)
```bash
# Manual MPI execution (if canonical script unavailable)
mpirun -np 2 --bind-to socket --map-by socket \
  --mca mpi_leave_pinned 1 --report-bindings \
  ./build/llaminar -m models/qwen2.5-0.5b-instruct-q4_0.gguf
```

This architecture provides a solid foundation for high-performance, distributed LLM inference while maintaining modularity, extensibility, and observability throughout the system.

---

## Backend Abstraction Layer (Prefill & Inference)

To prepare for future GPU offload while retaining the current CPU-only implementation, Llaminar now includes lightweight backend abstraction interfaces that wrap all latency / throughput sensitive GEMMs inside attention and MLP kernels.

### Goals
1. Decouple kernel call sites from concrete matmul implementation details (OpenBLAS vs COSMA vs future GPU libraries)
2. Preserve existing adaptive CPU heuristics (single-thread / multi-thread / distributed) without duplication
3. Provide zero-cost indirection today (simple inline / single virtual hop) with fast fallback to existing `adaptive_matmul`
4. Centralize prefill-vs-inference path decisions for consistent logging and experimentation

### Components
**Files**: `src/backends/prefill_backend.h/cpp`, `src/backends/inference_backend.h/cpp`, `src/backends/device_kind.h`

| Element | Description |
|---------|-------------|
| `DeviceKind` | Enum declaring `CPU`, `GPU` (future) |
| `PrefillBackendInterface` | Interface for large, throughput-oriented (prompt / prefill) GEMMs |
| `InferenceBackendInterface` | Interface for small, latency-critical (decode) GEMMs |
| `CpuPrefillBackend` | Delegates to `adaptive_matmul` honoring COSMA thresholds |
| `CpuInferenceBackend` | Delegates to `adaptive_matmul` optimized for small/medium ops |
| `Gpu*BackendStub` | Placeholder implementations returning `Unsupported` status |
| `PrefillBackendFactory` / `InferenceBackendFactory` | Simple factories (later: pluggable registry) |

### Invocation Pattern (Attention / MLP)
At each projection (Q, K, V, Out, Gate, Up, Down) the kernel:
1. Computes `is_prefill_like = seq_len >= debugEnv().cosma.prefill_threshold`
2. Chooses Prefill vs Inference backend interface
3. Attempts backend `launch()` with a small descriptor struct (M,N,K + flags)
4. On non-success (`Unsupported`, `Error`) logs a WARN and falls back to `adaptive_matmul`

All fallbacks preserve the exact previous semantics; there is no behavioral drift.

### Descriptor Structures
| Prefill | Inference |
|---------|-----------|
| `PrefillOpDesc { kind, M,N,K, is_prefill }` | `InferenceOpDesc { kind, M,N,K, latency_critical }` |
| `PrefillLaunchContext { A,B,C }` | `InferenceLaunchContext { A,B,C }` |

Currently only `MatMul` kind is implemented; future kinds (fused epilogs, attention assembly, quant-dequant fusion) can extend these enums without touching call sites.

### Logging & Observability
First use per kernel (rank 0 only) emits a single structured line:
```
BACKEND_DECISION_SUMMARY component=Attention seq_len=... threshold=... path=prefill prefill_backend=cpu_stub inference_backend=cpu_stub phases=QKV+Out fallback=adaptive_matmul
```
This enables quick grep-based validation of which path dominated a run without log spam.

### Relationship to Adaptive MatMul
`adaptive_matmul` remains the authoritative CPU arbitration layer (single-thread vs multi-thread vs distributed COSMA). Backends **delegate** rather than re-implement heuristics, ensuring a single tuning locus. When GPU support lands, only backend implementations change; kernel code and adaptive CPU logic remain intact.

### Future Extensions (Non-Breaking)
| Extension | Impact |
|-----------|--------|
| GPU BLAS (cuBLAS / rocBLAS) | Implement `GpuPrefillBackend` & `GpuInferenceBackend` with stream / handle caching |
| Triton / CK kernels | Register specialized fused attention or MLP epilogs via additional `OpKind` values |
| Auto-tuning | Backends collect lightweight perf stats, feed heuristic refinement engine |
| JSON decision trace | Optional per-op structured emission for offline analysis |

---

## Distribution Modes & Tensor Parallel Simplification

The legacy intra-rank tensor parallel splitter logic has been removed. Llaminar now treats **Tensor Parallel (TP)** strictly as *inter-socket / inter-process* sharding. Two high-level deployment modes are defined:

| Mode | Default Threshold | Characteristics |
|------|-------------------|-----------------|
| `ReplicatedDataParallel` | < ~32B params (configurable) | All weights replicated per rank; simpler, lower latency for small/medium models |
| `ShardedTensorParallel` | ≥ threshold | Parameter matrices partitioned across ranks; reduces memory footprint |

Runtime selection is driven by environment snapshot fields parsed once in `debug_env` (`DistributionEnvConfig`). This centralization:
1. Eliminates repeated `getenv()` overhead in hot loops
2. Provides a discoverable registry of supported knobs
3. Enables consistent experiment repro (single summary line on startup)

### Rationale for Simplification
Previous intra-rank column/row splitter heuristics added complexity and branch misprediction overhead for marginal gains on small decode shapes. Inter-rank sharding (true TP) captures the meaningful memory scaling while leaving per-rank math contiguous and cache-friendly.

### Interaction with Backends
- Prefill path: large prompt → may trigger distributed COSMA via `adaptive_matmul` once thresholds hit
- Inference path: small decode batches remain local; sharding only affects weight residency & gather/scatter steps external to backend invocation

---

## Environment Snapshot (`debugEnv`) Consolidation

Hot-path code (kernels, backend selection) consumes a pre-parsed immutable snapshot from `debug_env.{h,cpp}` instead of raw `std::getenv` calls.

Benefits:
1. Reduced libc call overhead on small matmuls
2. Single source of truth for defaults & validation
3. Easier documentation & test harness override

Key groups include:
| Group | Example Fields |
|-------|----------------|
| `cosma` | `prefill_threshold`, `max_resident_mb`, validation toggles |
| `distribution` | Mode overrides, sharding thresholds |
| `attention` | Trace / micro diagnostics flags |

---