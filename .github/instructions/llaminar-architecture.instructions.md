# Llaminar LLM Inference Engine - Architecture Documentation

*Last Updated: October 6, 2025*

## Overview

Llaminar is a high-performance, MPI-first LLM inference engine focused on low‑latency decode and scalable prefill. The architecture is built on a **multi-architecture pipeline abstraction** with pluggable model-family adapters, **strategy-pattern prefill providers**, and comprehensive observability.

### Core Architecture Pillars

1. **Abstract Pipeline System** ✨
   - **Multi-Architecture Support**: Factory pattern for Qwen, LLaMA, and future model families
   - **Clean Abstraction**: `AbstractPipeline` interface with `prefill()` / `decode()` lifecycle
   - **Concrete Implementations**: `QwenPipeline` (production), `LlamaPipeline` (prototype)
   - **Adapters**: `QwenPipelineAdapter`, `LlamaPipelineAdapter` implement standard interface
   - **Factory Registration**: Automatic model-family selection via `PipelineFactory`

2. **Prefill Provider Abstraction** ✨ *NEW*
   - **Strategy Pattern**: Swappable prefill backends (OpenBLAS, COSMA, future GPU)
   - **Built-in Snapshot Capture**: Base class provides parity testing utilities for all providers
   - **Runtime Selection**: `PrefillProviderFactory` chooses optimal provider based on sequence length and MPI context
   - **Isolated Testing**: Each provider testable in isolation with unified metrics
   - **Clean Separation**: Pipeline orchestrates, providers execute with stage-by-stage instrumentation

3. **COSMA Prefill Manager**
   - High‑throughput large prompt (prefill) GEMMs with fused RMSNorm+QKV path
   - Validation / orientation diagnostics and automatic fallback
   - Integrated with `COSMAPrefillProvider` for distributed execution

4. **Tensor Sharding**
   - Current: 1D column partition for linear projections
   - Planned: Hybrid 1D→2D sharding for multi‑node scaling

5. **Centralized Environment Snapshot** ✨
   - `debugEnv()`: Structured, typed access to all configuration flags
   - Eliminates repeated `getenv()` calls in hot loops
   - Single source of truth for tuning parameters

6. **Comprehensive Observability**
   - Structured perf counters and stage timers
   - Provider-integrated snapshot capture for parity testing
   - COSMA tile validation and distributed GEMM diagnostics
   - Prefill diagnostics module for baseline comparison

## Core Design Principles

1. **MPI-First Architecture**: All compute kernels derive from `MPIKernelBase`
2. **Multi-Architecture Pipeline**: Explicit model-family adapters (Qwen, LLaMA) behind unified interface
3. **Strategy Pattern for Backends**: Prefill providers encapsulate execution logic with runtime selection
4. **Stage-Driven Execution**: Semantic transformer stages (prefill/decode) replace generic graph scheduling
5. **Data Locality First**: Column partition weight shards + fused prefill minimize communication
6. **Graceful Degradation**: Provider factory falls back to OpenBLAS with diagnostics
7. **Deterministic Debugging**: Environment snapshot + opt‑in validation for surgical diagnostics
8. **Clean Abstraction Boundaries**: Pipeline orchestrates, kernels compute, providers execute

## Recent Architectural Improvements (October 2025)

### Prefill Provider Refactoring ✨

**Motivation**: The original `QwenPipeline` contained monolithic prefill logic with scattered backend selection, making it difficult to:
- Test COSMA vs OpenBLAS execution paths in isolation
- Capture consistent snapshots across backends for parity testing
- Switch backends without modifying pipeline code
- Add new execution backends (GPU) without tangled dependencies

**Solution**: Extracted prefill execution into a **strategy pattern** with pluggable providers:

#### Before (Monolithic Pipeline)
```cpp
class QwenPipeline {
    bool prefill(...) {
        // 500+ lines of prefill logic
        if (use_cosma) {
            // COSMA-specific execution
            executePrefillAttentionCosma(...);
        } else {
            // OpenBLAS-specific execution  
            executeAttentionLocal(...);
        }
        // Ad-hoc snapshot capture scattered throughout
    }
};
```

**Problems**:
- ❌ Backend logic tangled with pipeline orchestration
- ❌ Hard to test backends in isolation (need full pipeline)
- ❌ Snapshot capture inconsistent between backends
- ❌ Adding GPU backend requires pipeline modifications
- ❌ No unified metrics across backends

#### After (Provider Abstraction)
```cpp
// Base abstraction
class PrefillProvider {
    virtual bool execute(tokens, weights, output, ctx, metrics) = 0;
protected:
    void captureSnapshot(...);  // Built-in for all providers
};

// Concrete implementations
class OpenBLASPrefillProvider : public PrefillProvider { ... };
class COSMAPrefillProvider : public PrefillProvider { ... };

// Factory selection
auto provider = PrefillProviderFactory::create(config, mpi_ctx, seq_len);
bool success = provider->execute(tokens, weights, output, ctx, metrics);
```

**Benefits**:
- ✅ **Separation of Concerns**: Pipeline orchestrates, providers execute
- ✅ **Isolated Testing**: Each provider testable independently with mocked weights
- ✅ **Consistent Snapshots**: Base class provides capture utilities for all providers
- ✅ **Runtime Selection**: Factory chooses optimal provider based on workload
- ✅ **Extensible**: GPU provider can be added without touching pipeline/kernel code
- ✅ **Unified Metrics**: `PrefillMetrics` struct tracks timing/FLOPS/snapshots consistently
- ✅ **Parity Testing**: Both providers capture at identical stages for A/B comparison

#### Migration Impact

**Files Refactored**:
- **New**: `src/prefill_provider.{h,cpp}` - Base abstraction and factory
- **New**: `src/openblas_prefill_provider.{h,cpp}` - Baseline CPU provider
- **New**: `src/cosma_prefill_provider.{h,cpp}` - Distributed COSMA provider
- **Modified**: `src/qwen_pipeline.cpp` - Now delegates to provider factory
- **Tests**: `tests/test_prefill_providers.cpp` - Isolated provider tests
- **Tests**: `tests/test_parity_framework.cpp` - Provider-aware parity tests

**No Breaking Changes**:
- External API unchanged (`AbstractPipeline::prefill()` signature preserved)
- Environment variables honored (`ADAPTIVE_DISABLE_COSMA`, etc.)
- Backend selection logic preserved (sequence length thresholds)
- Performance characteristics identical

**Testing Coverage**:
- ✅ `test_prefill_providers`: Isolated provider unit tests
- ✅ `test_embedding_parity`: Embedding layer validation
- ✅ `test_parity_framework`: End-to-end multi-provider parity testing
- ✅ `test_embedding_standalone`: Standalone embedding correctness

#### Future Extensions Enabled

This refactoring makes the following additions straightforward:

1. **GPU Provider** (future):
   ```cpp
   class GpuPrefillProvider : public PrefillProvider {
       // cuBLAS/rocBLAS matmuls, device memory management
   };
   ```
   - No pipeline changes needed
   - Factory adds GPU detection logic
   - Inherits snapshot capture automatically

2. **Decode Provider** (future):
   ```cpp
   class DecodeProvider {
       virtual bool execute(token, weights, output, ctx, metrics) = 0;
   };
   ```
   - Mirror prefill pattern for decode phase
   - Enables latency-optimized GPU decode

3. **Fused Kernels**:
   - Providers can use specialized fused ops (FlashAttention, etc.)
   - Interface unchanged, implementation swapped

4. **Multi-Model Support**:
   - Factory can select provider based on model architecture
   - Different strategies for different model families

## Architecture Components

### 1. Entry Point & Orchestration

**File**: `src/main.cpp` (219 lines)
- **Purpose**: Application entry point and execution orchestration
- **Key Features**:
  - MPI initialization with thread support
  - 7-stage execution pipeline
  - Exception handling and graceful shutdown
  - Performance measurement and reporting

**Execution Flow**:
1. **Initialization**: Parse CLI → build `LlaminarParams` (includes `--kv-stats`, verbosity, model path)
2. **Environment Setup**: Initialize logging + `debugEnv()` snapshot
3. **Topology Detection**: Detect NUMA, cores; optionally print system info
4. **Pipeline Registration**: Register Qwen and LLaMA pipeline creators with `PipelineFactory`
5. **Model Loading**: Load GGUF model → weight tensors → wrap in architecture-specific `IModelWeights`
6. **Pipeline Creation**: Use `PipelineFactory::create(model_config)` to instantiate appropriate pipeline
7. **Inference**: Execute prefill (prompt) then iterative decode until completion
8. **Summary**: Emit KV cache stats, performance counters, backend statistics

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

### 3. Multi-Architecture Pipeline System ✨

The abstract pipeline architecture provides clean separation between model-family-specific logic and the core inference engine.

#### Architecture Components

**Files**: `src/abstract_pipeline.h`, `src/pipeline_base.h`, `src/pipeline_factory.h`

**Core Classes**:
- **`AbstractPipeline`**: Pure virtual interface defining pipeline lifecycle
- **`PipelineBase`**: MPI-aware base with kernel composition and tensor utilities  
- **`PipelineFactory`**: Singleton registry for architecture-specific creators
- **`IModelWeights`**: Polymorphic weight access interface
- **`StageContext`**: Metadata for prefill/decode stage tracking
- **`KVCacheState`**: KV cache capacity and usage tracking

#### Pipeline Interface

```cpp
class AbstractPipeline {
public:
    virtual const ModelConfig& config() const = 0;
    virtual bool prefill(const std::vector<int>& tokens,
                        const IModelWeights& weights,
                        StageContext& ctx) = 0;
    virtual bool decode(int next_token,
                       const IModelWeights& weights,
                       StageContext& ctx) = 0;
    virtual bool logits(std::shared_ptr<TensorBase>& out_logits) = 0;
    virtual std::unique_ptr<IModelWeights> loadWeights(const std::string& path) = 0;
    virtual std::string name() const = 0;
};
```

#### Concrete Implementations

**QwenPipeline** (`src/qwen_pipeline.{h,cpp}`):
- Production implementation for Qwen 2.5 model family
- Formerly `DistributedTransformerPipeline` (renamed for clarity)
- Implements complete transformer execution with MPI distribution
- Supports both prefill and decode with adaptive backend selection

**QwenPipelineAdapter** (`src/qwen_pipeline_adapter.{h,cpp}`):
- Wraps `QwenPipeline` behind `AbstractPipeline` interface
- Provides `QwenModelWeights` implementing `IModelWeights`
- Handles stage context management and logits caching
- Implements `loadWeights()` for Qwen-specific GGUF parsing

**LlamaPipelineAdapter** (`src/llama_pipeline_adapter.{h,cpp}`):
- Prototype implementation for LLaMA model family
- Currently delegates to `QwenPipeline` (similar architecture)
- Demonstrates multi-architecture extensibility
- Future: Specialized for LLaMA-specific features (GQA variations, etc.)

#### Pipeline Factory Pattern

```cpp
// Registration (done at startup)
PipelineFactory::instance().registerCreator("qwen", 
    [](const ModelConfig& cfg) {
        return std::make_unique<QwenPipelineAdapter>(cfg);
    });

// Usage in main.cpp
auto pipeline = PipelineFactory::instance().create(model_config);
```

#### Stage-Driven Execution

**Prefill Phase**:
1. Embedding lookup
2. Layer loop:
   - RMSNorm (attention)
   - QKV projection (may use fused COSMA path)
   - Attention primitives (RoPE, scaled dot-product, output projection)
   - Residual connection
   - RMSNorm (FFN)
   - MLP (Gate/Up/SwiGLU/Down with adaptive backend)
   - Residual connection
3. Output RMSNorm + LM head projection

**Decode Phase**:
1. Embedding lookup (single token)
2. Layer loop (same structure, optimized for single token):
   - Local OpenBLAS for all matmuls (latency-optimized)
   - KV cache append
   - Causal attention over growing context
3. Output projection → logits

#### Benefits Over Legacy Compute Graph

- ✅ **Eliminated**: Generic node scheduling overhead
- ✅ **Simplified**: Fixed semantic stages vs dynamic DAG
- ✅ **Centralized**: Environment controls via `debugEnv()` snapshot
- ✅ **Modular**: Clean adapter boundaries for new architectures
- ✅ **Testable**: Explicit stage transitions and parity validation


### 4. Prefill Provider Abstraction ✨

**Files**: `src/prefill_provider.{h,cpp}`, `src/openblas_prefill_provider.{h,cpp}`, `src/cosma_prefill_provider.{h,cpp}`

The prefill provider architecture implements a **strategy pattern** for swappable prefill execution backends, enabling:
- Multiple prefill implementations (OpenBLAS, COSMA, future GPU)
- Built-in snapshot capture for parity testing
- Runtime backend selection based on workload characteristics
- Isolated testing of individual providers
- Stage-by-stage instrumentation and validation

#### Architecture Hierarchy

```
AbstractPipeline::prefill()
  └─> QwenPipeline::prefill()
       └─> PrefillProvider::execute()  [with snapshot hooks]
            ├─> OpenBLASPrefillProvider  (baseline, CPU matmuls)
            ├─> COSMAPrefillProvider     (distributed matmuls)
            └─> (future) GPUPrefillProvider
```

#### PrefillProvider Base Class

**Purpose**: Abstract interface for prefill execution with built-in observability

**Key Features**:
- **Strategy Pattern**: Swap providers at runtime based on config/workload
- **Snapshot Utilities**: Base class provides capture methods inherited by all providers
- **Zero Overhead**: Snapshots compiled out in release builds
- **MPI-Aware**: Providers handle distributed execution and rank coordination
- **Metrics Tracking**: Timing, FLOP counting, and stage-level instrumentation

**Core Interface**:
```cpp
class PrefillProvider {
public:
    virtual bool execute(
        const std::vector<int>& tokens,
        const IModelWeights& weights,
        std::shared_ptr<TensorBase>& output,
        StageContext& ctx,
        PrefillMetrics& metrics) = 0;
    
    virtual std::string name() const = 0;
    
protected:
    void captureSnapshot(PipelineStage stage, int layer_index,
                        const float* data, int seq_len, int feature_dim);
    bool isSnapshotEnabled() const;
};
```

#### OpenBLASPrefillProvider

**File**: `src/openblas_prefill_provider.{h,cpp}`

**Purpose**: CPU-based prefill using OpenBLAS for matrix multiplications

**Characteristics**:
- **Baseline Implementation**: Well-tested, predictable behavior
- **Optimal For**: Small to medium sequences (< 4K tokens), single-node setups
- **MatMul Backend**: OpenBLAS (single/multi-threaded based on operation size)
- **Kernel Architecture**: Uses existing MPI kernel infrastructure
  - `MPIEmbeddingKernel`: Token embedding lookup
  - `MPIRMSNormKernel`: Layer normalization (sequence-wise distribution)
  - `MPIAttentionKernel`: Complete attention (Q/K/V/O projections + primitives)
  - `MPILinearKernel`: FFN linear projections
  - `MPISwiGLUKernel`: SwiGLU activation

**Stage Flow**:
1. **EMBEDDING**: Token embedding lookup
2. **Per Layer**:
   - ATTENTION_NORM → Attention (Q/K/V/RoPE/scores/softmax/context/output) → ATTENTION_RESIDUAL
   - FFN_NORM → FFN_GATE/UP → FFN_SWIGLU → FFN_DOWN → FFN_RESIDUAL
3. **FINAL_NORM** → **LM_HEAD**

**Snapshot Capture**: All standardized stages for PyTorch comparison

#### COSMAPrefillProvider

**File**: `src/cosma_prefill_provider.{h,cpp}`

**Purpose**: Distributed prefill using COSMA for large-scale matrix multiplications

**Characteristics**:
- **Optimal For**: Large sequences (≥ 4K tokens), multi-node setups (2+ MPI ranks)
- **Performance**: Up to 3.6x faster than OpenBLAS for large operations (≥64K tokens)
- **MatMul Backend**: COSMA for distributed compute, with adaptive fallback
- **Fused Operations**: Combines RMSNorm + QKV projection for efficiency

**Key Differences from OpenBLAS**:
- **Attention**: Fused norm+QKV via COSMA → CPU attention primitives → adaptive output projection
- **FFN**: Uses `adaptiveMatMul` (may use COSMA for gate/up/down based on size)
- **Memory**: Distributed weight layout, higher communication overhead
- **Tradeoff**: Better throughput for large ops, worse for small ops (<4K tokens)

**Snapshot Alignment**:
- Captures at **same stages** as OpenBLASPrefillProvider
- Enables A/B testing: run both providers, compare snapshots stage-by-stage
- Identifies divergence source (COSMA matmul vs attention primitives vs etc.)

#### PrefillProviderFactory

**File**: `src/prefill_provider.cpp`

**Purpose**: Automatic provider selection based on configuration and workload

**Selection Logic**:
```cpp
std::unique_ptr<PrefillProvider> PrefillProviderFactory::create(
    const ModelConfig& config,
    const MPIContext& mpi_ctx,
    int seq_len) {
    
    const auto& env = debugEnv();
    
    // Check global COSMA disable
    if (env.adaptive.disable_cosma) {
        return std::make_unique<OpenBLASPrefillProvider>(config, mpi_ctx);
    }
    
    // Check for forced COSMA execution (debug/validation)
    if (env.cosma.force_direct || env.cosma.force_replicated) {
        return std::make_unique<COSMAPrefillProvider>(config, mpi_ctx);
    }
    
    // Sequence length-based decision
    int threshold = env.cosma.prefill_threshold;  // default: 4096
    if (seq_len >= threshold && mpi_ctx.size > 1) {
        return std::make_unique<COSMAPrefillProvider>(config, mpi_ctx);
    }
    
    // Default: OpenBLAS for small/medium sequences
    return std::make_unique<OpenBLASPrefillProvider>(config, mpi_ctx);
}
```

**Decision Criteria**:
1. **Environment Override**: `ADAPTIVE_DISABLE_COSMA=1` → always OpenBLAS
2. **Debug Forcing**: `LLAMINAR_COSMA_FORCE_DIRECT=1` → always COSMA
3. **Sequence Length**: `seq_len >= threshold` (default 4096) → COSMA (if multi-rank)
4. **Default Fallback**: OpenBLAS for small sequences or single-rank

#### PrefillMetrics

**Purpose**: Comprehensive instrumentation for prefill execution

**Tracked Metrics**:
- **Timing**: Per-stage breakdown (embedding, attention, FFN, norm, LM head)
- **Compute**: Total FLOP count and GFLOPS throughput
- **Execution**: Layers executed, snapshots captured
- **Backend**: Provider name for performance attribution

**Usage Example**:
```cpp
PrefillMetrics metrics;
auto provider = PrefillProviderFactory::create(config, mpi_ctx, seq_len);
bool success = provider->execute(tokens, weights, output, ctx, metrics);

if (success) {
    LOG_INFO("Prefill completed: " << metrics.total_ms() << "ms, "
             << metrics.gflops() << " GFLOPS, "
             << "backend=" << metrics.backend_name);
}
```

#### Environment Controls

**Provider Selection**:
- `ADAPTIVE_DISABLE_COSMA=1`: Force OpenBLAS provider globally
- `LLAMINAR_COSMA_PREFILL_THRESHOLD=<tokens>`: Sequence length threshold (default: 4096)
- `LLAMINAR_COSMA_FORCE_DIRECT=1`: Force COSMA provider (debug)
- `LLAMINAR_COSMA_FORCE_REPLICATED=1`: Force COSMA replicated path (validation)

**COSMA-Specific Controls**:
- `LLAMINAR_COSMA_MAX_RESIDENT_MB=<mb>`: Memory budget (default: 2048)
- `LLAMINAR_COSMA_VALIDATE_TILE=<size>`: Enable correctness validation (debug)
- `LLAMINAR_COSMA_COMPARE_REPLICATED=1`: Full replicated comparison (expensive)
- `LLAMINAR_COSMA_AUTO_FIX_TRANSPOSE=1`: Automatic orientation correction

**Snapshot Controls**:
- `LLAMINAR_PARITY_CAPTURE=1`: Enable snapshot capture in debug builds
- `LLAMINAR_COSMA_TEST_TRACE=1`: Escalate COSMA test verbosity

#### Integration with Pipeline

The pipeline uses the factory for automatic provider selection:

```cpp
// In QwenPipeline::prefill()
PrefillMetrics metrics;
auto provider = PrefillProviderFactory::create(config_, mpi_ctx_, tokens.size());

bool success = provider->execute(tokens, weights, hidden_states, ctx, metrics);

if (success) {
    // Cache logits, update metrics, etc.
    ctx.backend_stats.prefill_backend = metrics.backend_name;
    ctx.backend_stats.prefill_gflops = metrics.gflops();
}
```

#### Benefits Over Previous Architecture

**Before (Monolithic QwenPipeline)**:
- ✗ Prefill logic scattered across pipeline class
- ✗ Hard to test backends in isolation
- ✗ Snapshot capture ad-hoc and inconsistent
- ✗ Backend switching required code changes

**After (Provider Abstraction)**:
- ✓ Clean separation: pipeline orchestrates, providers execute
- ✓ Testable in isolation: unit tests for each provider
- ✓ Consistent snapshots: base class provides utilities
- ✓ Runtime selection: factory chooses optimal provider
- ✓ Extensible: GPU provider can be added without pipeline changes

### 5. COSMA Prefill Manager

**Files**: `src/cosma_prefill_manager.{h,cpp}`, `src/prefill_diagnostics.{h,cpp}`

The COSMA prefill manager provides fused, distributed execution of large prefill operations with integrated diagnostics and validation.

#### Core Functionality

**Fused RMSNorm + QKV**:
- Combines layer normalization and QKV projection into single distributed operation
- Avoids intermediate activation materialization
- Unified COSMA strategy selection prevents inconsistent tiling

**Orientation & Layout Management**:
- Automatic orientation detection and correction
- Validation hooks for debugging mismatches
- Auto-fix capability via `LLAMINAR_COSMA_AUTO_FIX_TRANSPOSE`

**Distributed Execution**:
- Coordinates across MPI ranks with barriers
- Handles partial matrix reconstruction and gathering
- Manages COSMA buffer allocation lifecycle

#### Diagnostics Integration

**Prefill Diagnostics Module** (`src/prefill_diagnostics.{h,cpp}`):
- **BufferStats**: Min/max/mean/L2 norm computation
- **DiffSummary**: Relative L2 error and maximum absolute difference
- **PrefillBaselineRegistry**: Reference execution for validation

**Usage Pattern**:
```cpp
// Optional baseline comparison
if (env.prefill_debug.compare_baseline) {
    auto baseline_result = PrefillBaselineRegistry::computeBaseline(...);
    auto diff = PrefillBaselineRegistry::compareResults(
        cosma_result, baseline_result);
    LOG_INFO("Prefill diff: rel_l2=" << diff.rel_l2 
             << ", max_abs=" << diff.max_abs);
}
```

#### Performance Counters

Tracks all GEMM operations with:
- Operation dimensions (m × n × k)
- Backend used (COSMA vs OpenBLAS)
- Execution time
- GFLOPS achieved

Post-run summary shows aggregate statistics for optimization.

#### Fallback Behavior

1. **Attempt COSMA Path**: Try distributed execution via COSMA
2. **Validation Check**: Optionally verify results against OpenBLAS tile
3. **On Failure**: Log warning and fall back to single-rank OpenBLAS
4. **Transparent Recovery**: Pipeline continues without user intervention

#### Environment Controls

| Variable | Purpose | Default |
|----------|---------|---------|
| `LLAMINAR_COSMA_FORCE_DIRECT` | Force COSMA direct path | Unset |
| `LLAMINAR_COSMA_COMPARE_REPLICATED` | Full OpenBLAS validation | Unset (expensive) |
| `LLAMINAR_COSMA_VALIDATE_TILE` | Small tile validation | 0 (off) |
| `LLAMINAR_COSMA_DEBUG_RECON` | Verbose reconstruction logs | Unset |
| `LLAMINAR_COSMA_AUTO_FIX_TRANSPOSE` | Auto-correct orientation | Unset |
| `LLAMINAR_COSMA_LOG_LEVEL` | Prefill logging verbosity | info |

#### Integration Points

- Called from `COSMAPrefillProvider` for attention QKV projection
- Used for MLP gate/up/down projections in large prefill
- Coordinates with `PrefillProviderFactory` for automatic backend selection
- Reports statistics via `debugEnv().prefill_debug` snapshot

### 6. Distributed Linear Projection

**File**: `src/kernels/MPILinearKernel.cpp`

Provides column-partitioned dense layer implementation for distributed weight storage with local OpenBLAS computation.

#### Architecture

**Weight Distribution**:
- Weights partitioned by columns across MPI ranks
- Each rank holds `[hidden_size, proj_size / num_ranks]` slice
- Input activations replicated across all ranks

**Execution Flow**:
1. **Local GEMM**: Each rank computes `input @ local_weight_slice`
2. **Gather Results**: `MPI_Allgatherv` assembles full output across ranks  
3. **Optional Bias**: Add bias vector if provided

**Backend Integration**:
- Explicitly uses OpenBLAS for local matmul
- Used by `OpenBLASPrefillProvider` for all linear projections
- Ensures consistent execution with provider-based architecture

#### Use Cases

- Distributed weight storage when model exceeds single-node memory
- Balanced computation across MPI ranks
- Compatible with both prefill and decode phases

#### Performance Characteristics

**Advantages**:
- Distributes memory footprint across ranks
- Balances computation load
- Leverages fast local OpenBLAS for small/medium operations

**Overhead**:
- `MPI_Allgatherv` communication cost
- Beneficial when communication < local GEMM time savings

#### Interaction with Adaptive Backend

The kernel ensures COSMA is not applied to partial weight slices:

```cpp
// In MPILinearKernel::execute()
adaptiveMatMul(input, local_weight, local_output,
               /* distributed_partition= */ true);
// Flag prevents COSMA selection for partial matrix
```

This maintains correctness while allowing adaptive selection at higher pipeline levels.

### 7. Attention Implementation

**Files**: `src/kernels/MPIAttentionKernel.{h,cpp}`, `src/kernels/attention_primitives.{h,cpp}`

The attention system provides multi-head attention with RoPE positional encoding, supporting both fused COSMA prefill and local kernel-based execution.

#### Execution Paths

**Large Prefill (COSMA Path)**:
1. Fused RMSNorm + QKV projection via `cosma_prefill_manager`
2. Reshape to multi-head format: `[batch, seq_len, num_heads, head_dim]`
3. Apply RoPE positional encoding per head
4. Scaled dot-product attention: `softmax(QK^T / sqrt(d_k)) V`
5. Output projection (adaptive backend)

**Decode / Small Prefill (Local Path)**:
1. Separate RMSNorm, Q/K/V projection kernels
2. RoPE encoding
3. KV cache append
4. Causal attention over growing context
5. Local OpenBLAS output projection

#### RoPE (Rotary Position Embedding)

**Implementation**: Direct per-head loop with sin/cos computation
**Format**: Applied to Q and K tensors before attention
**Future Optimization**: Vectorization, LUT-based sin/cos reuse

```cpp
// Simplified RoPE application
for (int head = 0; head < num_heads; ++head) {
    for (int pos = 0; pos < seq_len; ++pos) {
        float theta = pos / pow(10000.0, 2.0 * dim / head_dim);
        float cos_theta = cos(theta);
        float sin_theta = sin(theta);
        // Apply rotation to Q[head][pos] and K[head][pos]
    }
}
```

#### Attention Mechanism

**Scaled Dot-Product**:
- Query-Key product: `scores = Q @ K^T`
- Scaling: `scores /= sqrt(head_dim)`
- Causal masking: Set future positions to -inf
- Softmax: `probs = softmax(scores)` (row-wise normalization)
- Apply to values: `output = probs @ V`

**Execution**:
- All reductions and softmax executed locally (single rank)
- Causal masking enforced row-wise
- Future enhancement: Distributed softmax for large multi-rank KV spans

#### Multi-Head Attention

**Head Configuration**:
- Number of heads from `ModelConfig::n_heads`
- Head dimension: `hidden_size / n_heads`
- Supports Grouped Query Attention (GQA) variations

**Reshaping**:
- Input: `[batch, seq_len, hidden_size]`
- Multi-head: `[batch, seq_len, num_heads, head_dim]`
- Transposed for attention: `[batch, num_heads, seq_len, head_dim]`

#### Performance Optimizations

- **RoPE Caching**: Future LUT-based sin/cos precomputation
- **Fused Kernels**: Combined operations reduce memory traffic
- **Adaptive Backend**: Large operations use COSMA, small use OpenBLAS
- **KV Cache**: Avoid recomputing past positions in decode

### 8. KV Cache Management

**Files**: `src/abstract_pipeline.h`, `src/qwen_pipeline.cpp`

The KV cache system stores past key and value tensors to avoid recomputation during autoregressive decode.

#### KVCacheState Structure

```cpp
struct KVCacheState {
    int max_seq_len;      // Maximum capacity
    int current_pos;      // Current fill position
    bool initialized;     // Cache allocation status
    
    // Per-layer storage
    std::vector<std::shared_ptr<TensorBase>> key_cache;   
    std::vector<std::shared_ptr<TensorBase>> value_cache;
};
```

#### Lifecycle

**Initialization**: During first prefill
- Allocate `[num_layers, max_seq_len, num_heads, head_dim]` tensors
- Set `max_seq_len` from config or CLI `--max-seq-len`
- Mark `initialized = true`

**Prefill**: Write initial sequence
- Store computed K and V for all positions
- Set `current_pos = prefill_length`

**Decode**: Append new tokens
1. Check capacity: `current_pos + 1 <= max_seq_len`
2. Compute K, V for new token
3. Append to cache at `current_pos`
4. Increment `current_pos`

#### Observability

**CLI Flag**: `--kv-stats`
- Prints cache usage and capacity after each operation
- Shows per-layer memory footprint
- Reports fill percentage

**Example Output**:
```
KV Cache Stats:
  Position: 127/512 (24.8%)
  Layers: 24
  Memory per layer: 2.1 MB
  Total KV cache: 50.4 MB
```

#### Testing

**Parity Tests**: `test_abstract_pipeline_parity.cpp`
- Validates KV cache state after prefill
- Compares incremental decode vs full replay
- Ensures `current_pos` consistency

#### Future Enhancements

**Planned Features**:
- **Eviction Policies**: LRU, sliding window, selective retention
- **Compaction**: Remove unused sequence segments
- **Distribution**: Shard KV cache across MPI ranks (owner-aware)
- **Quantization**: Store K/V in reduced precision (INT8, FP16)

**Pluggable Strategy**:
```cpp
class KVCacheEvictionPolicy {
public:
    virtual void compact(KVCacheState& cache, int min_retained) = 0;
};
```

Integration point: `AbstractPipeline::decode()` invokes policy before cache append.

### 9. Environment & Observability

**Files**: `src/debug_env.{h,cpp}`

Centralized environment configuration system eliminating hot-path `std::getenv` calls with structured, typed snapshots.

#### Debug Environment Snapshot

**Access Pattern**:
```cpp
const auto& env = debugEnv();
if (env.attention.micro_trace && rank == 0) {
    LOG_TRACE("Attention Q shape: " << q_shape);
}
```

**Configuration Groups**:
- `attention`: Attention kernel diagnostics and tracing
- `baseline`: Reference execution and comparison
- `cosma`: COSMA-specific controls and validation
- `adaptive`: Backend selection overrides
- `pipeline`: Pipeline-level instrumentation
- `linear`: Linear kernel diagnostics
- `embedding`: Embedding layer controls
- `layer_capture`: Per-layer forensic capture
- `prefill_debug`: Prefill diagnostics and validation

#### Design Principles

**Mandatory Rules**:
1. **No Hot-Path getenv**: All environment variables parsed once at initialization
2. **Typed Fields**: Boolean, integer, and string fields with proper defaults
3. **Single Registry**: All flags documented in `debug_env.h`
4. **Lazy Initialization**: Snapshot created on first `debugEnv()` call

**Migration Guidance**:
- New flags MUST be added to appropriate group in `debug_env.h`
- Existing raw `std::getenv` calls should be migrated on file touch
- Experimental flags staged in snapshot early to prevent drift

#### Diagnostic Capabilities

**Per-Stage Validation**:
- Optional reference recomputation for correctness checking
- Relative L2 error computation
- Top sample value logging
- Per-layer diff summaries

**Forensic Modes**:
- Row capture for RMSNorm intermediate values
- FFN gate/up/down activation dumps
- COSMA orientation and reconstruction debugging
- Attention QKV tensor inspection

**Performance Instrumentation**:
- Per-operation timing (via performance counters)
- Adaptive backend decision logging
- GEMM statistics aggregation
- Post-run summary reports

#### Common Environment Variables

| Variable | Purpose | Group | Type |
|----------|---------|-------|------|
| `LLAMINAR_ATTN_MICRO_TRACE` | Detailed attention tracing | attention | bool |
| `ADAPTIVE_DISABLE_COSMA` | Force OpenBLAS path | adaptive | bool |
| `LLAMINAR_COSMA_PREFILL_THRESHOLD` | Sequence length threshold | cosma | int |
| `LLAMINAR_COSMA_VALIDATE_TILE` | Tile validation size | cosma | int |
| `LLAMINAR_DEQUANT_STATS` | Dequantization statistics | pipeline | bool |
| `LLAMINAR_COSMA_LOG_LEVEL` | COSMA log verbosity | cosma | string |

#### Usage Examples

```cpp
// Attention tracing
const auto& env = debugEnv();
if (env.attention.micro_trace) {
    LOG_TRACE("QK product shape: " << qk_shape);
}

// Backend override
if (env.adaptive.disable_cosma) {
    return MatMulBackend::MULTI_THREADED_OPENBLAS;
}

// Validation control
if (env.cosma.validate_tile > 0) {
    performTileValidation(env.cosma.validate_tile);
}
```

#### Future Extensions

- **Hot Reload**: Dynamic snapshot refresh without restart
- **Remote Control**: External configuration channel
- **Profiling Modes**: Predefined diagnostic profiles
- **Conditional Compilation**: Debug-only snapshot fields

### 10. Model Loading & Weight Management

**Files**: `src/model_loader.{h,cpp}`, `src/qwen_pipeline_adapter.cpp`, `src/llama_pipeline_adapter.cpp`

Model loading system parses GGUF format files and instantiates architecture-specific weight implementations.

#### Loading Pipeline

**1. Format Detection**:
- Read GGUF header and magic bytes
- Validate file version compatibility
- Parse metadata section

**2. Configuration Extraction**:
- Model dimensions (vocab size, hidden size, layer count)
- Architecture parameters (attention heads, MLP ratio)
- Tokenizer vocabulary
- Create `ModelConfig` structure

**3. Tensor Enumeration**:
- Scan tensor directory
- Map tensor names to pipeline weight keys
- Record shapes and data types

**4. Quantization Handling**:
- Detect quantization format (Q4_K, Q6_K, etc.)
- Dequantize to FP32 on load (per-tensor)
- Optional statistics logging via `LLAMINAR_DEQUANT_STATS`
- Anomaly detection via `LLAMINAR_DEQUANT_ANOMALIES`

**5. MPI Distribution**:
- Rank 0 reads GGUF file
- Broadcast metadata to all ranks
- For distributed tensors: partition and distribute slices
- For replicated tensors: broadcast full tensor

**6. Tensor Instantiation**:
- Select tensor type (SimpleTensor vs COSMATensor) based on size
- Apply environment-driven selection policy
- Populate weight containers

#### IModelWeights Implementations

**QwenModelWeights**:
```cpp
class QwenModelWeights : public IModelWeights {
    std::shared_ptr<TensorBase> getEmbedding(int token_id) const override;
    std::shared_ptr<TensorBase> getWeight(const std::string& key) const override;
    bool hasWeight(const std::string& key) const override;
    ModelConfig config() const override;
private:
    std::unordered_map<std::string, std::shared_ptr<TensorBase>> weights_;
    std::shared_ptr<TensorBase> embedding_table_;
};
```

**LlamaModelWeights**:
- Currently delegates to QwenModelWeights (similar architecture)
- Future: Handle LLaMA-specific weight layouts

#### Weight Naming Conventions

**Standard Keys**:
- Embedding: `token_embd.weight`
- QKV projections: `layers.{L}.attention.{q,k,v}_weight`
- Attention output: `layers.{L}.attention.wo_weight`
- MLP gates: `layers.{L}.mlp.{gate,up,down}_proj`
- Layer norms: `layers.{L}.{attn,mlp}_norm.weight`
- Output projection: `output.weight`

#### Quantization Support

**Supported Formats**:
- **Q4_K**: 4-bit with block-wise quantization
- **Q6_K**: 6-bit with block-wise quantization  
- **FP32**: Full precision (no quantization)

**Dequantization**:
- Performed on load (eager dequantization)
- Per-tensor statistics logged if enabled
- Anomaly detection: NaN, Inf, extreme values

**Statistics Example**:
```
Dequant [layers.0.attention.q_weight]:
  Min: -0.234, Max: 0.198
  Mean: 0.003, Std: 0.067
  Samples: [-0.012, 0.045, -0.023, ...]
```

#### Layout Adaptation

**Repacker**: Performs layout transformations for fused kernels
- Contiguous block ordering for QKV concatenation
- COSMA-compatible memory layouts
- Alignment requirements for vectorization

#### Future Enhancements

- **Lazy Dequantization**: Keep quantized format, dequant on-the-fly
- **Mixed Precision**: FP16/BF16 activation buffers
- **Streaming Load**: Memory-mapped incremental loading
- **Weight Sharding**: Automatic partitioning for large models

### 11. Testing Infrastructure

**Files**: `tests/test_*.cpp`, `tests/parity_*.cpp`

Comprehensive test suite validating pipeline correctness, backend selection, and distributed execution.

#### Test Categories

**Pipeline Tests**:
- **`test_pipeline_factory.cpp`**: Factory registration and creation
- **`test_abstract_pipeline_parity.cpp`**: Prefill vs incremental decode equivalence
- **`test_qwen_pipeline.cpp`**: Qwen-specific pipeline functionality (4 test cases)

**Backend & COSMA Tests**:
- **`test_cosma_prefill_*.cpp`**: Fused COSMA correctness and statistics
- **`test_adaptive_matmul*.cpp`**: Backend decision logic validation
- **`test_cosma.cpp`**: Core COSMA integration

**Primitive Kernel Tests**:
- **`test_rmsnorm_*.cpp`**: RMSNorm parity and edge cases
- **`test_attention_*.cpp`**: Attention mechanism validation
- **`test_rope_*.cpp`**: RoPE positional encoding
- **`test_softmax_*.cpp`**: Softmax numerical stability

**Distributed Execution Tests**:
- **`test_tp_*.cpp`**: Tensor partition correctness
- **`test_mlp_tp_parity.cpp`**: MLP distributed parity
- **`test_mpi_linear_kernel.cpp`**: MPILinearKernel validation

**Infrastructure Tests**:
- **`test_basic.cpp`**: MPI initialization and basic functionality
- **`test_numa.cpp`**: NUMA topology detection and affinity
- **`test_kv_cache_growth*.cpp`**: KV cache capacity management

#### Removed Tests

Historical tests no longer needed after architecture refactor:
- ❌ `test_graph.cpp`: Generic compute graph (removed architecture)
- ❌ `LinearKernelTest`: Legacy non-MPI linear kernel (retired)

#### Parity Testing Framework

**Purpose**: Ensure mathematical equivalence across execution paths

**Key Parity Tests**:
1. **Prefill vs Incremental Decode**: 
   - Full sequence prefill should match token-by-token decode
   - KV cache state must be identical
   - Logits must match within numerical tolerance

2. **COSMA vs OpenBLAS**:
   - Large operations should produce equivalent results
   - Relative L2 error < 1e-4 threshold
   - Maximum absolute difference tracking

3. **Distributed vs Single-Rank**:
   - Multi-rank execution matches single-rank reference
   - Gather/reduction correctness
   - No data loss or corruption

**Example Parity Check**:
```cpp
TEST_CASE("Prefill vs Incremental Parity") {
    // Full sequence prefill
    pipeline->prefill(all_tokens, weights, prefill_ctx);
    auto prefill_logits = pipeline->getLogits();
    
    // Incremental decode
    pipeline->prefill({all_tokens[0]}, weights, decode_ctx);
    for (int i = 1; i < all_tokens.size(); ++i) {
        pipeline->decode(all_tokens[i], weights, decode_ctx);
    }
    auto decode_logits = pipeline->getLogits();
    
    // Validate equivalence
    float rel_error = computeRelativeL2(prefill_logits, decode_logits);
    REQUIRE(rel_error < 1e-4);
}
```

#### Test Execution

**Run All Tests**:
```bash
ctest --test-dir build --output-on-failure --parallel
```

**Core Tests Only** (recommended during development):
```bash
ctest --test-dir build -R "^(BasicTest|PipelineFactoryTest|QwenPipelineTest)$"
```

**With Verbose Output**:
```bash
ctest --test-dir build --output-on-failure --verbose
```

**MPI Tests**:
```bash
mpirun -np 2 ./build/test_abstract_pipeline_parity
```

#### Current Test Status

**Passing Tests**:
- ✅ BasicTest (MPI initialization)
- ✅ PipelineFactoryTest (factory mechanics)  
- ✅ QwenPipelineTest (3/4 subtests)

**Known Issues**:
- ⚠️ QwenPipelineTest.ValidationTests (pre-existing precision issue, unrelated to refactor)
- ⚠️ Some COSMA tests have numerical precision edge cases

#### Testing Best Practices

1. **Always run tests after kernel changes**
2. **Use parity tests to validate optimizations**
3. **Check both single-rank and multi-rank execution**
4. **Verify KV cache state consistency**
5. **Enable validation flags during development** (`LLAMINAR_COSMA_VALIDATE_TILE`, etc.)
6. **Disable heavy validation for benchmarking**

### 12. Performance Strategy Summary

Empirically-tuned backend selection based on operation characteristics and execution phase.

#### Backend Selection by Phase

| Phase | Sequence Length | Matrix Dims | Backend Policy | Rationale |
|-------|----------------|-------------|----------------|-----------|
| **Prefill (short)** | < 4K tokens | Many small/medium | OpenBLAS (single/multi-thread) | COSMA overhead not amortized |
| **Prefill (large)** | ≥ 4K tokens | seq_len × hidden → projections | COSMA | Collective reuse + fused QKV |
| **Decode** | 1 token | 1 × hidden → projections | OpenBLAS single-thread | Min latency, avoid collectives |
| **FFN (large prefill)** | ≥ 4K tokens | seq_len × hidden → d_ff | COSMA candidate | Throughput scaling |
| **LM Head** | Any | seq_len × hidden → vocab | Local (policy TBD) | Avoid huge all-gather unless 2D shard |

#### Empirical Performance Data

**Small Operations (< 8K elements)**:
- OpenBLAS single-threaded: **134x faster** than COSMA for 1×896×896
- Communication overhead dominates COSMA performance
- Recommendation: Always use local OpenBLAS

**Medium Operations (8K - 8M elements)**:
- OpenBLAS multi-threaded competitive for < 64 tokens
- COSMA overhead still significant
- Recommendation: Multi-threaded OpenBLAS

**Large Operations (≥ 8M elements)**:
- COSMA becomes competitive at ≥ 8K tokens
- **COSMA 3.6x faster** at 64K tokens vs single-rank OpenBLAS
- Distributed memory bandwidth advantage
- Recommendation: COSMA for large prefill

#### Threading Strategy

**Small Operations (< 8K elements)**:
```cpp
openblas_set_num_threads(1);  // Minimize overhead
```

**Medium Operations (8K - 1M elements)**:
```cpp
openblas_set_num_threads(omp_get_max_threads());  // Full socket
```

**Large Distributed Operations**:
```cpp
openblas_set_num_threads(cores_per_numa_node);  // Per-rank threading
// + COSMA distributed execution
```

#### Memory Considerations

**COSMA Buffer Allocation**:
- Soft limit: 2048 MB per rank (configurable via `LLAMINAR_COSMA_MAX_RESIDENT_MB`)
- Fallback to OpenBLAS if allocation exceeds budget
- Prevents memory exhaustion on large models

**Tensor Type Selection**:
- **SimpleTensor**: < 256×256 elements or latency-critical
- **COSMATensor**: ≥ 256×256 and distributed execution

#### Optimization Priorities

1. **Latency** (Decode): Single-threaded OpenBLAS, minimal overhead
2. **Throughput** (Large Prefill): COSMA distributed, fused kernels
3. **Memory Efficiency**: Adaptive tensor types, buffer reuse
4. **Correctness**: Parity validation, fallback paths

### 13. Tensor Sharding & Future Roadmap

Current distributed execution capabilities and planned enhancements for scaling to larger models and longer contexts.

#### Current State (1D Column Sharding)

**Weight Distribution**:
- Linear weights partitioned by columns across MPI ranks
- Each rank holds `[hidden_size, proj_size / num_ranks]` slice
- Activations replicated across all ranks

**Communication Pattern**:
- Local GEMM on weight slice
- `MPI_Allgatherv` to assemble full output
- Works well for moderate model sizes

**Limitations**:
- Activations fully replicated (memory scaling bottleneck)
- All-gather communication cost grows with output size
- KV cache duplicated across ranks

#### Planned Enhancements

**Phase 1: Activation Micro-Sharding**
- **Goal**: Reduce per-rank memory for very large batch prefill
- **Approach**: Partition activation tensors across sequence dimension
- **Benefit**: Linear memory scaling with rank count
- **Challenges**: Attention computation requires gather or ring-reduce

**Phase 2: 2D Block-Cyclic Distribution**
- **Goal**: Eliminate all-gather for extreme vocab or d_ff expansions
- **Approach**: Align 2D tensor distribution with COSMA tiling strategy
- **Benefit**: Zero-copy distributed matmul without post-gather
- **Integration**: Seamless with COSMA's `BlockCyclicMatrix` layout

**Phase 3: Distributed KV Cache**
- **Goal**: Shard past sequence to reduce memory duplication
- **Components**:
  - Owner-aware KV cache distribution
  - Distributed attention softmax (ring-reduce pattern)
  - Partial sequence ownership per rank
- **Benefit**: Support longer contexts with same memory budget

**Phase 4: Mixed Precision Execution**
- **Goal**: Reduce activation memory footprint by 50%
- **Approach**: 
  - FP16/BF16 activation buffers
  - On-the-fly dequantization for matmuls
  - Selective FP32 accumulation for numerical stability
- **Benefit**: 2x memory reduction, potential speedup on modern hardware

#### Implementation Strategy

**Incremental Rollout**:
1. Implement and validate each phase independently
2. Maintain backward compatibility with existing 1D path
3. Add environment flags to enable/disable features
4. Comprehensive parity testing at each phase

**Environment Controls** (Future):
- `LLAMINAR_ACTIVATION_SHARDING={none,1d,2d}`: Activation distribution mode
- `LLAMINAR_KV_CACHE_DISTRIBUTED=1`: Enable distributed KV cache
- `LLAMINAR_MIXED_PRECISION={fp32,fp16,bf16}`: Activation precision
- `LLAMINAR_2D_SHARD_THRESHOLD=<size>`: Minimum size for 2D sharding

#### Research Directions

**Adaptive Sharding**:
- Automatically select 1D vs 2D based on model dimensions
- Hybrid strategies for different layers (e.g., 2D for MLP, 1D for attention)

**Memory-Optimal Schedules**:
- Recompute vs cache trade-offs for activations
- Gradient checkpointing patterns (if extending to fine-tuning)

**Communication Optimization**:
- Overlapped computation and communication
- Hierarchical reduction trees for large rank counts
- NCCL/RCCL integration for GPU backends

### 14. Additional Future Enhancements

Beyond tensor sharding, several high-impact features are planned for production deployment and advanced use cases.

#### Runtime Reconfiguration

**Dynamic Environment Snapshot**:
- Reload `debugEnv()` snapshot without process restart
- Hot-swap backend policies during long-running inference
- Remote control channel for runtime configuration

**Use Cases**:
- A/B testing different backend strategies
- Adaptive optimization based on workload
- Debug flag toggle without restart

**Implementation**:
```cpp
// Future API
debugEnv().reload();
debugEnv().setRemoteChannel("tcp://controller:5555");
```

#### Adaptive Decode Micro-Batching

**Goal**: Process multiple next-token candidates in parallel

**Approach**:
- Beam search or speculative decoding
- Batch multiple decode operations
- Latency guardrails to prevent slowdown

**Challenges**:
- KV cache management for multiple hypotheses
- Memory overhead for parallel paths
- Efficient pruning and selection

#### Streaming KV Cache Management

**Compaction Strategies**:
- **LRU**: Evict least-recently-used sequence segments
- **Sliding Window**: Fixed-size context window
- **Selective Retention**: Keep important tokens (e.g., system prompt)

**Eviction Policies**:
```cpp
class StreamingKVCache {
    void compact(int target_size);
    void evictLRU(int num_tokens);
    void slidingWindow(int window_size);
    void selectiveRetain(const std::vector<int>& important_positions);
};
```

**Integration**: Pluggable policy at `AbstractPipeline::decode()` level

#### Quantized Fused Prefill

**Goal**: Direct dequantization into COSMA layout

**Current**: 
1. Load quantized weights → dequantize to FP32 → SimpleTensor
2. Convert to COSMATensor for COSMA operations

**Optimized**:
1. Load quantized weights → direct dequant into COSMA layout
2. Skip intermediate FP32 buffer allocation
3. Fuse dequant kernel with COSMA distribution

**Benefits**:
- 50%+ memory reduction during load
- Faster initialization
- Reduced allocation pressure

#### GPU Backend Support

**Multi-Backend Architecture**:
- Unified `TensorBase` interface supports CPU and GPU
- CUDA/HIP kernels for attention, matmul, softmax
- NCCL for GPU-to-GPU communication
- Hybrid CPU/GPU execution

**Challenges**:
- Unified memory management
- CPU ↔ GPU data transfer overhead
- NUMA awareness for PCIe topology

#### Production Deployment Features

**Model Serving**:
- gRPC/REST API for inference requests
- Request batching and queue management
- Dynamic model loading/unloading
- Multi-tenancy support

**Observability**:
- Prometheus metrics export
- OpenTelemetry tracing integration
- Performance profiling hooks
- Health check endpoints

**Fault Tolerance**:
- Checkpoint/resume for long sequences
- Rank failure recovery (MPI fault tolerance)
- Graceful degradation on resource exhaustion

#### Research Integration

**Experimental Features**:
- Flash Attention integration
- PagedAttention for KV cache
- Mixture-of-Experts (MoE) support
- Continuous batching (Orca-style)

**Plugin System**:
```cpp
class InferencePlugin {
    virtual bool shouldIntercept(const Operation& op) = 0;
    virtual bool execute(const Operation& op) = 0;
};

// Register custom optimizations
PipelineFactory::registerPlugin(std::make_unique<FlashAttentionPlugin>());
```

---

### 15. Parity Test Framework Integration ✨

**Overview**: The parity test framework provides comprehensive snapshot capture and comparison capabilities for validating pipeline execution correctness. It's now deeply integrated into the core pipeline architecture for automatic, zero-overhead parity testing.

#### Architecture

**Core Components**:

1. **PipelineStage Enum** (`src/pipeline_stages.h`)
   - Standardized 22-stage enumeration covering all transformer operations
   - Shared between production code and tests
   - Stages: EMBEDDING, ATTENTION_NORM, QKV_PROJECTION, ROPE_APPLICATION, ATTENTION_SCORES, ATTENTION_SOFTMAX, ATTENTION_CONTEXT, ATTENTION_OUTPUT, ATTENTION_RESIDUAL, FFN_NORM, FFN_GATE, FFN_UP, FFN_SWIGLU, FFN_DOWN, FFN_RESIDUAL, FINAL_NORM, LM_HEAD, CUSTOM
   - Utility functions: `stage_to_string()`, `string_to_stage()` (inline, zero overhead)

2. **Parity Hooks** (`src/parity_hooks.h/cpp`)
   - Production-safe interface with default no-op implementation
   - Environment-driven activation via `LLAMINAR_PARITY_CAPTURE`
   - Tests provide real implementation via `parity_test_framework.cpp`
   - Zero overhead when disabled (functions inline to empty)

3. **Pipeline Integration**
   - **AbstractPipeline**: Virtual `captureStageSnapshot()` and `isParityEnabled()` methods
   - **PipelineBase**: Convenience `captureIfEnabled()` helpers with rank filtering
   - **QwenPipeline**: 8 strategic capture points at key computation stages

#### Capture Points in QwenPipeline

**Prefill/Decode Path** (automatic capture when `LLAMINAR_PARITY_CAPTURE=1`):

| Stage | Location | Description |
|-------|----------|-------------|
| `EMBEDDING` | After token embedding | Input to first transformer layer |
| `ATTENTION_NORM` | Pre-attention RMSNorm | Input to QKV projection |
| `ATTENTION_OUTPUT` | After W_o projection | Attention block output |
| `ATTENTION_RESIDUAL` | After attention residual add | Input to FFN block |
| `FFN_NORM` | Pre-FFN RMSNorm | Input to gate/up projections |
| `FFN_DOWN` | After down projection | FFN block output |
| `FFN_RESIDUAL` | After FFN residual add | Output of transformer layer |
| `FINAL_NORM` | After final RMSNorm | Input to LM head |
| `LM_HEAD` | Language model head output | Final logits |

**Key Design Features**:
- **Rank 0 Only**: Captures happen on rank 0 to avoid MPI duplication
- **Automatic Shape Extraction**: Captures sequence length and feature dimension from tensors
- **Layer-Aware**: Each capture includes layer index (or -1 for non-layer stages)
- **Zero Overhead When Disabled**: `isParityEnabled()` check inlines to false

#### Usage Patterns

**Basic Parity Testing**:
```bash
# Enable automatic snapshot capture
export LLAMINAR_PARITY_CAPTURE=1

# Run Llaminar inference (captures will be stored in SnapshotRegistry)
mpirun -np 2 ./build/llaminar -m model.gguf -v

# Compare with reference implementation in test
./build/test_parity_framework
```

**Test Integration**:
```cpp
#include "parity_test_framework.h"

TEST(ParityTest, QwenPrefillVsReference) {
    // Clear previous captures
    parity::SnapshotRegistry::instance().clear();
    
    // Enable parity capture
    parity::LlaminarSnapshotHook::set_enabled(true);
    
    // Run Llaminar pipeline (automatic capture via captureIfEnabled calls)
    auto pipeline = PipelineFactory::create(config);
    pipeline->prefill(tokens, weights, ctx);
    
    // Run reference implementation and capture
    // ... reference execution ...
    
    // Compare snapshots
    auto tolerance = parity::ComparisonTolerance(1e-3f, 1e-4);
    for (int layer = 0; layer < num_layers; ++layer) {
        auto key_llama = registry.make_key("llaminar", PipelineStage::ATTENTION_OUTPUT, layer);
        auto key_ref = registry.make_key("reference", PipelineStage::ATTENTION_OUTPUT, layer);
        
        TensorSnapshot snap_llama, snap_ref;
        ASSERT_TRUE(registry.get_snapshot(key_llama, snap_llama));
        ASSERT_TRUE(registry.get_snapshot(key_ref, snap_ref));
        
        auto result = SnapshotComparator::compare(snap_ref, snap_llama, tolerance);
        EXPECT_TRUE(result.passed()) << "Layer " << layer << " failed parity";
    }
}
```

**Custom Capture Points** (extending to new architectures):
```cpp
class MyCustomPipeline : public PipelineBase, public AbstractPipeline {
    bool execute(...) override {
        // ... computation ...
        
        // Capture at custom stage
        captureIfEnabled(PipelineStage::CUSTOM, layer_idx, my_tensor);
        
        // ... more computation ...
        return true;
    }
};
```

#### Comparison Metrics

**Supported Metrics**:
- **Max Absolute Difference**: `max(|expected - actual|)`
- **Mean Absolute Difference**: `mean(|expected - actual|)`
- **Relative L2 Norm**: `||expected - actual||₂ / ||expected||₂`
- **Worst Element Tracking**: Index and values of maximum difference

**Configurable Tolerances**:
```cpp
// Strict tolerance for early layers
auto strict = ComparisonTolerance(1e-4f, 1e-5);

// Relaxed tolerance for final logits
auto relaxed = ComparisonTolerance(1e-2f, 1e-3);
```

#### Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `LLAMINAR_PARITY_CAPTURE` | Enable automatic snapshot capture | Disabled (0) |
| `LLAMINAR_LAYER_TOKEN_DIFF` | Legacy layer diff diagnostics | Disabled (0) |

**Note**: The new parity framework (`LLAMINAR_PARITY_CAPTURE`) is complementary to the legacy layer token diff system. They can coexist but serve different purposes:
- **Parity Framework**: Cross-implementation validation (Llaminar vs llama.cpp)
- **Layer Token Diff**: Incremental decode vs replay validation within Llaminar

#### Integration with Existing Diagnostics

**Relationship to Baseline Comparison**:
- **Baseline**: Per-stage FP32 snapshots for prefill GEMM validation
- **Parity**: Cross-pipeline snapshot comparison for correctness validation
- Both use `handle_prefill_stage_snapshot()` infrastructure
- Can be used together for comprehensive validation

**Relationship to Layer Token Diff**:
- **Layer Token Diff**: Legacy diagnostic for incremental decode parity
- **Parity Framework**: Modern unified approach for all validation
- Migration path: Gradually replace layer token diff with parity framework

#### Performance Considerations

**Overhead When Disabled**:
- **Zero Overhead**: `isParityEnabled()` compiles to constant `false`
- Capture calls eliminated by compiler dead code elimination
- No runtime checks in hot paths

**Overhead When Enabled**:
- **Memory**: Stores full tensor snapshots (can be large for long sequences)
- **Compute**: Tensor copy per capture point (~microseconds for typical sizes)
- **Recommendation**: Use only for validation, not performance benchmarks

#### Future Enhancements

**Planned Features**:
1. **Selective Stage Capture**: Environment variable to capture only specific stages
2. **Streaming Snapshots**: Write captures to disk instead of memory for long sequences
3. **Differential Snapshots**: Only capture changed regions for incremental decode
4. **Automatic Tolerance Tuning**: Learn tolerances from successful runs
5. **Cross-Rank Comparison**: Validate distributed state consistency across MPI ranks
6. **Integration with CI/CD**: Automated parity regression tests on every commit

**Plugin Architecture** (future):
```cpp
class ParityPlugin {
    virtual bool shouldCapture(PipelineStage stage, int layer) = 0;
    virtual void onCapture(const TensorSnapshot& snapshot) = 0;
};

// Register custom parity validation logic
parity::registerPlugin(std::make_unique<MyCustomValidator>());
```

#### Files

**Core**:
- `src/pipeline_stages.h`: PipelineStage enum and conversion utilities
- `src/parity_hooks.h/cpp`: Production-safe hook interface
- `src/abstract_pipeline.h`: Virtual parity methods (base interface)
- `src/pipeline_base.h/cpp`: Convenience helpers (`captureIfEnabled`)
- `src/qwen_pipeline.cpp`: 8 capture points in production pipeline

**Tests**:
- `tests/parity_test_framework.h/cpp`: Full snapshot capture and comparison implementation
- `tests/test_parity_framework.cpp`: Parity framework unit tests

**Total LOC**: ~500 lines (framework), ~10 lines added to QwenPipeline

---

## Additional Technical Details

### Build System & Dependencies

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

## Execution Flow

### Runtime Lifecycle

1. **Initialization**: 
   - MPI setup (`MPI_Init_thread` with `MPI_THREAD_MULTIPLE`)
   - NUMA topology detection
   - Environment snapshot creation (`debugEnv()`)
   - OpenMP thread configuration

2. **Model Loading**:
   - GGUF file parsing (metadata + tensor directory)
   - Weight tensor instantiation (auto SimpleTensor/COSMATensor selection)
   - MPI broadcast/distribution of weights
   - Pipeline creation via `PipelineFactory::create(config)`

3. **Prefill Phase**:
   - Token sequence → embedding lookup
   - Per-layer execution:
     - RMSNorm (attention)
     - QKV projection (COSMA path if threshold met via `cosma_prefill_manager`)
     - Multi-head attention (RoPE + scaled dot-product)
     - Residual connection
     - RMSNorm (MLP)
     - MLP (Gate/Up/SwiGLU/Down with adaptive backend)
     - Residual connection
   - Output RMSNorm + LM head projection
   - KV cache initialization

4. **Decode Loop** (Autoregressive Generation):
   - Single token → embedding lookup
   - Per-layer execution (same structure, latency-optimized):
     - Local OpenBLAS for all matmuls
     - KV cache append
     - Causal attention over growing context
   - Output projection → logits
   - External sampling/chat interface
   - Repeat until stopping condition

5. **Completion**:
   - Optional performance summary (if perf counters enabled)
   - Optional KV cache statistics (if `--kv-stats` flag set)
   - Validation logging (if diagnostics enabled)
   - MPI cleanup (`MPI_Finalize`)

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

## Testing Strategy

**Directory**: `tests/`

**Core Infrastructure Tests**:
- `test_basic.cpp`: MPI initialization and basic functionality
- `test_numa.cpp`: NUMA topology detection and affinity  
- `test_pipeline_factory.cpp`: Pipeline factory registration and creation

**Pipeline & Parity Tests**:
- `test_qwen_pipeline.cpp`: Qwen pipeline functionality (4 test cases)
- `test_abstract_pipeline_parity.cpp`: Prefill vs incremental decode equivalence
- `test_kv_cache_growth*.cpp`: KV cache capacity management

**Backend & Kernel Tests**:
- `test_cosma_prefill_*.cpp`: Fused COSMA correctness and statistics
- `test_adaptive_matmul*.cpp`: Backend selection validation
- `test_mpi_linear_kernel.cpp`: Distributed linear projection
- `test_cosma.cpp`: Core COSMA integration

**Primitive Tests**:
- `test_rmsnorm_*.cpp`: RMSNorm correctness and parity
- `test_attention_*.cpp`: Attention mechanism validation
- `test_rope_*.cpp`: RoPE positional encoding
- `test_softmax_*.cpp`: Softmax numerical stability
- `test_mlp_tp_parity.cpp`: MLP distributed parity
- `test_tp_*.cpp`: Tensor partition correctness

**Test Coverage Focus**:
- ✅ Mathematical equivalence across execution paths (parity tests)
- ✅ MPI communication correctness and coordination
- ✅ Backend selection policy validation
- ✅ KV cache state management
- ✅ NUMA topology detection accuracy
- ✅ Distributed execution correctness

**Removed Historical Tests**:
- ❌ `test_graph.cpp`: Generic compute graph (architecture removed)
- ❌ `LinearKernelTest`: Legacy non-MPI kernel (retired after MPI migration)

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
**Current Architecture**: `src/prefill_provider.{h,cpp}`, `src/openblas_prefill_provider.{h,cpp}`, `src/cosma_prefill_provider.{h,cpp}`

| Element | Description |
|---------|-------------|
| `PrefillProvider` | Abstract base class for prefill execution strategies |
| `OpenBLASPrefillProvider` | CPU-based prefill using OpenBLAS (baseline, optimal for <4K tokens) |
| `COSMAPrefillProvider` | Distributed prefill using COSMA (optimal for ≥4K tokens, multi-rank) |
| `PrefillProviderFactory` | Automatic provider selection based on sequence length and MPI context |
| `PrefillMetrics` | Comprehensive instrumentation (timing, FLOPS, snapshots) |
| `GpuPrefillProvider` (future) | GPU-accelerated prefill with device memory management |

**Decode Path**: Currently uses local OpenBLAS in `QwenPipeline::decode()` for latency optimization. Future work may extract decode provider abstraction.

### Invocation Pattern (from QwenPipeline)
```cpp
// In QwenPipeline::prefill()
PrefillMetrics metrics;
auto provider = PrefillProviderFactory::create(config_, mpi_ctx_, tokens.size());
bool success = provider->execute(tokens, weights, output, ctx, metrics);

if (success) {
    // Cache results, update context
    ctx.backend_stats.prefill_backend = metrics.backend_name;
    ctx.backend_stats.prefill_gflops = metrics.gflops();
}
```

Provider selection logic:
1. **Environment Override**: `ADAPTIVE_DISABLE_COSMA=1` → OpenBLAS
2. **Debug Forcing**: `LLAMINAR_COSMA_FORCE_DIRECT=1` → COSMA
3. **Sequence Length**: `seq_len >= threshold` (default 4096) + multi-rank → COSMA
4. **Default**: OpenBLAS for small sequences or single-rank

### Descriptor Structures
| Provider Metrics | Captured Data |
|------------------|---------------|
| `PrefillMetrics::embedding_ms` | Embedding lookup time |
| `PrefillMetrics::attention_ms` | Total attention time across layers |
| `PrefillMetrics::ffn_ms` | Total FFN time across layers |
| `PrefillMetrics::norm_ms` | Normalization time |
| `PrefillMetrics::total_flops` | Floating point operations count |
| `PrefillMetrics::backend_name` | Provider identifier ("OpenBLAS", "COSMA") |

All providers capture snapshots at standardized stages for parity testing when `LLAMINAR_PARITY_CAPTURE=1`.

### Logging & Observability
Providers emit structured metrics on completion:
```
[INFO] OpenBLASPrefillProvider completed: 347.2ms, 12.3 GFLOPS, 24 layers, 147 snapshots
[INFO] COSMAPrefillProvider completed: 96.5ms, 44.1 GFLOPS, 24 layers, 147 snapshots
```

Each provider captures at identical stages for A/B comparison:
- EMBEDDING, ATTENTION_NORM, Q/K/V_PROJECTION, ROPE_APPLICATION
- ATTENTION_SCORES, ATTENTION_SOFTMAX, ATTENTION_CONTEXT, ATTENTION_OUTPUT
- FFN_NORM, FFN_GATE, FFN_UP, FFN_SWIGLU, FFN_DOWN
- FINAL_NORM, LM_HEAD

### Relationship to COSMA Prefill Manager
`COSMAPrefillProvider` delegates to `CosmaPrefillManager` for distributed matmul coordination, maintaining separation of concerns:
- **Provider**: Orchestrates transformer layer execution, snapshot capture, metrics
- **Manager**: Handles COSMA-specific matmul, orientation validation, memory management

### Future Extensions (Non-Breaking)
| Extension | Impact |
|-----------|--------|
| GPU Provider | Implement `GpuPrefillProvider` with cuBLAS/rocBLAS, no pipeline changes |
| Decode Provider | Extract `DecodeProvider` abstraction mirroring prefill pattern |
| Fused Kernels | Providers can use specialized fused ops without interface changes |
| Auto-tuning | Providers collect perf stats for heuristic refinement |
| Multi-Model | Factory can select provider based on model architecture

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

## PyTorch Reference Implementation for Ground-Truth Validation 🐍

**Location**: `python/reference/`  
**Status**: Production-ready infrastructure (test integration pending API alignment)  
**Purpose**: Stage-by-stage numerical validation against HuggingFace transformer models

### Overview

The PyTorch reference implementation provides **21 granular capture stages** (vs llama.cpp's 2-stage capture) for comprehensive parity testing. It enables debugging of quantization errors, operator implementation issues, and numerical drift by comparing Llaminar's execution against ground-truth PyTorch/HuggingFace models.

### Architecture

**Three-Component Design**:

1. **Python Reference Implementation** (`python/reference/`)
   - Abstract base class: `AbstractReferenceModel`
   - Concrete implementations: `QwenReferenceModel` (production), `LlamaReferenceModel` (prototype)
   - Stage-by-stage hook system for capturing intermediate activations
   - Quantization simulation (Q4_0, Q6_K) matching GGUF formats
   - CLI tool: `run_reference.py` for snapshot generation

2. **Snapshot Bridge** (`tests/npz_loader.h`, `tests/npz_to_npy.py`)
   - Header-only C++ .npy parser (zero external dependencies)
   - Python extraction helper (.npz → individual .npy files)
   - Cross-language compatibility layer

3. **C++ Parity Integration** (`tests/test_parity_framework.cpp`)
   - Test template for PyTorch snapshot comparison
   - Configurable tolerances for FP32/FP16/Q6_K/Q4_0
   - Stage-by-stage comparison with detailed diagnostics
   - Status: Infrastructure complete, awaiting API refactor (see §15.7)

### 21 Capture Stages

Granular snapshots covering the full transformer pipeline:

| Category | Stages | Purpose |
|----------|--------|---------|
| **Input** | `EMBEDDING`, `POSITIONAL_ENCODING` | Input validation, vocabulary alignment |
| **Attention (per-layer)** | `ATTENTION_NORM`, `QKV_PROJECTION`, `ROPE_APPLICATION`, `ATTENTION_SCORES`, `ATTENTION_PROBS`, `ATTENTION_CONTEXT`, `ATTENTION_OUTPUT`, `ATTENTION_RESIDUAL` | Attention mechanism correctness |
| **FFN (per-layer)** | `FFN_NORM`, `FFN_GATE`, `FFN_UP`, `FFN_ACTIVATION`, `FFN_DOWN`, `FFN_RESIDUAL` | Feed-forward network validation |
| **Per-Layer** | `LAYER_OUTPUT` | Layer-wise correctness |
| **Output** | `FINAL_NORM`, `LM_HEAD`, `FINAL_LOGITS`, `PROBABILITIES` | Final output validation |

**Note**: Llaminar's `QwenPipeline` currently captures 8 strategic stages (see Parity Test Framework §13). The PyTorch reference provides all 21 for comprehensive debugging.

### Quick Start Workflow

**1. Generate PyTorch Reference Snapshots**:
```bash
# Install dependencies (pre-installed in devcontainer)
pip install -r python/reference/requirements.txt

# Generate FP32 snapshots
python python/reference/run_reference.py \
    --model qwen \
    --checkpoint Qwen/Qwen2-0.5B-Instruct \
    --tokens 1,2,3,4,5 \
    --output pytorch_snapshots.npz \
    --verbose

# Generate quantized snapshots for Q4_0 validation
python python/reference/run_reference.py \
    --model qwen \
    --checkpoint Qwen/Qwen2-0.5B-Instruct \
    --tokens 1,2,3,4,5 \
    --quantization q4_0 \
    --output pytorch_q4_snapshots.npz
```

**2. Extract Snapshots to C++-Compatible Format**:
```bash
# Extract .npz archive to individual .npy files
python tests/npz_to_npy.py pytorch_snapshots.npz pytorch_snapshots/

# Directory structure after extraction:
# pytorch_snapshots/
#   EMBEDDING_-1.npy              # Non-layer stage (layer_idx = -1)
#   ATTENTION_OUTPUT_0.npy        # Layer 0 attention output
#   FFN_DOWN_0.npy                # Layer 0 FFN down projection
#   FINAL_NORM_-1.npy             # Final normalization
#   LM_HEAD_-1.npy                # Language model head
```

**3. Load and Compare in C++**:
```cpp
#include "npz_loader.h"
using namespace llaminar::parity;

// Load PyTorch reference snapshot
PyTorchSnapshotLoader pytorch_loader("pytorch_snapshots/");
NpyArray pytorch_embedding;
if (pytorch_loader.load_snapshot("EMBEDDING", -1, pytorch_embedding)) {
    // pytorch_embedding.shape = {1, seq_len, hidden_dim}
    // pytorch_embedding.data = std::vector<float>
    
    // Compare with Llaminar snapshot from SnapshotRegistry
    auto llaminar_snapshot = SnapshotRegistry::get("EMBEDDING", -1);
    auto metrics = SnapshotComparator::compare(pytorch_embedding, llaminar_snapshot);
    
    LOG_INFO("EMBEDDING max_abs_diff: " << metrics.max_abs_diff 
             << ", rel_l2: " << metrics.rel_l2);
}
```

### File Format & Naming Convention

**NumPy .npy Format** (header-only parser in `npz_loader.h`):
- Magic bytes: `\x93NUMPY`
- Version: 1-3 (2 bytes)
- Header: Python dict with `descr` (dtype), `fortran_order`, `shape`
- Data: Raw float32 blob (little-endian)

**Naming Convention**:
```
{STAGE_NAME}_{LAYER_INDEX}.npy

Examples:
  EMBEDDING_-1.npy           # Global stage (no layer)
  ATTENTION_OUTPUT_0.npy     # Layer 0 attention output
  FFN_DOWN_5.npy             # Layer 5 FFN down projection
  FINAL_NORM_-1.npy          # Final norm (no layer)
```

### Comparison Metrics & Tolerances

**Standard Metrics**:
- `max_abs_diff`: max|A - B| across all elements
- `mean_abs_diff`: mean|A - B|
- `rel_l2`: ||A - B||₂ / ||A||₂ (relative L2 norm)

**Recommended Tolerances**:

| Precision | max_abs_diff | rel_l2 | Use Case |
|-----------|--------------|--------|----------|
| FP32 | 1e-4 | 1e-5 | Numerical precision only |
| Q6_K | 5e-3 | 1e-2 | High-quality quantization |
| Q4_0 | 1e-2 | 5e-2 | Aggressive quantization |

**Adaptive Tolerance Strategy** (recommended):
```cpp
// Stricter for early stages (error doesn't accumulate)
if (stage == "EMBEDDING" || stage.find("NORM") != npos) {
    return is_quantized ? 5e-3f : 1e-4f;
}

// Relaxed for late stages (accumulated error)
if (stage == "FINAL_LOGITS") {
    return is_quantized ? 5e-2f : 1e-3f;
}
```

### Environment Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `PYTORCH_SNAPSHOT_DIR` | Directory with extracted .npy files | `pytorch_snapshots/` |
| `PYTORCH_SNAPSHOT_TOKENS` | Token IDs used for generation | `1,2,3,4,5` |
| `PYTORCH_MODEL_PATH` | HuggingFace checkpoint path | `Qwen/Qwen2-0.5B-Instruct` |
| `PYTORCH_QUANTIZATION` | Quantization format | `q4_0` or `q6_k` |

### Model Family Support

**Production**:
- **Qwen**: `QwenReferenceModel` with full 21-stage capture
  - Supports Qwen2-0.5B, Qwen2-1.5B, Qwen2-7B
  - Quantization: Q4_0, Q6_K matching GGUF
  - RoPE, SwiGLU FFN, RMSNorm validation

**Prototype**:
- **LLaMA**: `LlamaReferenceModel` with basic structure
  - Needs attention implementation completion
  - Quantization hooks in place

**Future** (extensible via `AbstractReferenceModel`):
- DeepSeek (similar to LLaMA)
- Mistral (sliding window attention)
- GPT-2 (learned positional embeddings)

### Integration Status

**✅ Complete**:
- Python reference implementation (800+ lines)
- CLI tool for snapshot generation
- Header-only C++ .npy parser (303 lines, zero dependencies)
- Python extraction helper (100 lines)
- Comprehensive documentation (850+ lines total)

**⚠️ Pending** (see §15.7):
- Test case needs API refactoring (~1 hour)
  - Current: Uses old `GGUFContext` API
  - Target: New `ModelConfig`-based `AbstractPipeline` API
- CI/CD integration (GitHub Actions workflow)

### Advanced Usage Patterns

**Quantization Error Analysis**:
```bash
# Generate FP32 baseline
python python/reference/run_reference.py --model qwen \
    --checkpoint Qwen/Qwen2-0.5B-Instruct --tokens 1,2,3 \
    --output fp32.npz

# Generate Q6_K quantized
python python/reference/run_reference.py --model qwen \
    --checkpoint Qwen/Qwen2-0.5B-Instruct --tokens 1,2,3 \
    --quantization q6_k --output q6k.npz

# Generate Q4_0 quantized
python python/reference/run_reference.py --model qwen \
    --checkpoint Qwen/Qwen2-0.5B-Instruct --tokens 1,2,3 \
    --quantization q4_0 --output q4.npz

# Compare all three in C++ to isolate quantization vs implementation errors
```

**Selective Stage Testing**:
```python
# Only capture specific stages for faster iteration
python python/reference/run_reference.py \
    --model qwen --checkpoint Qwen/Qwen2-0.5B-Instruct \
    --tokens 1,2,3,4,5 \
    --stages EMBEDDING ATTENTION_OUTPUT FINAL_NORM \
    --output selective.npz
```

**Long Sequence Validation**:
```bash
# Test with sequences up to context length
python python/reference/run_reference.py \
    --model qwen --checkpoint Qwen/Qwen2-0.5B-Instruct \
    --tokens $(seq 1 100 | tr '\n' ',') \
    --output long_seq.npz
```

### Troubleshooting Guide

**Shape Mismatch**:
- **Symptom**: `Shape mismatch: PyTorch [1,5,512] vs Llaminar [5,512]`
- **Cause**: Different batch dimension handling
- **Fix**: Ensure consistent token count (`PYTORCH_SNAPSHOT_TOKENS` matches input)

**High Numeric Drift**:
- **Symptom**: `rel_l2 = 0.5` (expected < 0.01)
- **Causes**: Quantization errors, operator bugs, different RNG seeds
- **Debug**: Compare stage-by-stage to isolate error source

**File Not Found**:
- **Symptom**: `FileNotFoundError: EMBEDDING_-1.npy`
- **Fix**: Run extraction step: `python tests/npz_to_npy.py snapshots.npz output/`
- **Verify**: `ls -la $PYTORCH_SNAPSHOT_DIR`

**Parser Error**:
- **Symptom**: `Invalid .npy header`
- **Cause**: Corrupted file or unsupported NumPy version
- **Fix**: Regenerate with latest NumPy: `pip install -U numpy`

### API Migration Path (§15.7)

**Current State**: Test infrastructure complete, awaiting `ModelConfig` API alignment

**Required Changes** (~1 hour):

```cpp
// 1. Model Loading (OLD → NEW)
// auto gguf_ctx = loader.load(model_path);
ModelConfig config = createConfigFromGGUF(model_path);

// 2. Pipeline Creation (OLD → NEW)
// auto pipeline = std::make_unique<QwenPipelineAdapter>(gguf_ctx);
auto pipeline = PipelineFactory::create(config);

// 3. Prefill Execution (OLD → NEW)
// auto result = pipeline->prefill(token_ids);
IModelWeights* weights = /* load from model */;
StageContext ctx = /* create context */;
bool success = pipeline->prefill(token_ids, weights, ctx);

// 4. Enable Test
// Remove DISABLED_ prefix from test name
TEST(ParityFramework, DistributedPipelineVsPyTorchReference) { ... }
```

### Documentation References

- **Python Reference README**: `python/reference/README.md` (300+ lines comprehensive guide)
- **C++ Integration Guide**: `tests/PYTORCH_INTEGRATION.md` (450+ lines with troubleshooting)
- **Test Framework Guide**: `tests/AGENTS.md` §14 (PyTorch reference usage)
- **Status Report**: `PYTORCH_INTEGRATION_STATUS.md` (implementation summary)
- **Parity Framework**: `tests/AGENTS.md` §13 (Llaminar snapshot capture system)

### Best Practices

1. **Version Control Snapshots**: Commit .npz files or document HuggingFace checkpoint versions
2. **Test Incrementally**: Validate embedding → layer 0 → layer N → final output
3. **Document Tolerances**: Add comments justifying relaxed tolerances for quantized models
4. **Isolate Errors**: Compare FP32 first, then add quantization progressively
5. **Cache Models**: Use `~/.cache/huggingface` to avoid re-downloading multi-GB checkpoints
6. **Validate Shapes First**: Shape mismatch indicates fundamental API issues
7. **Use Verbose Logging**: Enable `-vvv` and `--verbose` for debugging

### Future Enhancements

1. **Full .npz Support**: Eliminate extraction step by adding ZIP parsing to `npz_loader.h`
2. **Model Coverage**: Add DeepSeek, Mistral, GPT-2 reference implementations
3. **Automatic Tolerance Learning**: Adaptive tolerances based on successful runs
4. **GPU Validation**: Compare CUDA/ROCm kernels against PyTorch GPU execution
5. **Visual Diff Reports**: HTML output with per-stage heatmaps and statistics
6. **Streaming Snapshots**: Disk-backed storage for very long sequences
7. **CI/CD Integration**: Automated snapshot generation and comparison on every PR

---