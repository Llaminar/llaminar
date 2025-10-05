# Llaminar LLM Inference Engine - Architecture Documentation

*Last Updated: October 5, 2025*

## Overview

Llaminar is a high-performance, MPI-first LLM inference engine focused on low‑latency decode and scalable prefill. The architecture is built on a **multi-architecture pipeline abstraction** with pluggable model-family adapters, centralized backend selection, and comprehensive observability.

### Core Architecture Pillars

1. **Abstract Pipeline System** ✨
   - **Multi-Architecture Support**: Factory pattern for Qwen, LLaMA, and future model families
   - **Clean Abstraction**: `AbstractPipeline` interface with `prefill()` / `decode()` lifecycle
   - **Concrete Implementations**: `QwenPipeline` (production), `LlamaPipeline` (prototype)
   - **Adapters**: `QwenPipelineAdapter`, `LlamaPipelineAdapter` implement standard interface
   - **Factory Registration**: Automatic model-family selection via `PipelineFactory`

2. **Centralized Backend Selection** ✨
   - **MatMulBackendSelector**: Single decision point for all matrix multiplication paths
   - **Stage-Aware**: Considers operation size, stage context (prefill vs decode), and MPI topology
   - **Intelligent Routing**: Small ops → single-threaded OpenBLAS, medium → multi-threaded, large prefill → COSMA
   - **Zero Duplication**: Kernels delegate to selector, no scattered threshold logic

3. **COSMA Prefill Manager**
   - High‑throughput large prompt (prefill) GEMMs with fused RMSNorm+QKV path
   - Validation / orientation diagnostics and automatic fallback

4. **Tensor Sharding**
   - Current: 1D column partition for linear projections
   - Planned: Hybrid 1D→2D sharding for multi‑node scaling

5. **Centralized Environment Snapshot** ✨
   - `debugEnv()`: Structured, typed access to all configuration flags
   - Eliminates repeated `getenv()` calls in hot loops
   - Single source of truth for tuning parameters

6. **Comprehensive Observability**
   - Structured perf counters and stage timers
   - Optional per-layer token diffs and validation
   - COSMA tile validation and distributed GEMM diagnostics
   - Prefill diagnostics module for baseline comparison

## Core Design Principles

1. **MPI-First Architecture**: All compute kernels derive from `MPIKernelBase`
2. **Multi-Architecture Pipeline**: Explicit model-family adapters (Qwen, LLaMA) behind unified interface
3. **Centralized Policy**: Backend selection and tuning consolidated in `MatMulBackendSelector`
4. **Stage-Driven Execution**: Semantic transformer stages (prefill/decode) replace generic graph scheduling
5. **Data Locality First**: Column partition weight shards + fused prefill minimize communication
6. **Graceful Degradation**: COSMA failures auto‑fallback to OpenBLAS with diagnostics
7. **Deterministic Debugging**: Environment snapshot + opt‑in validation for surgical diagnostics
8. **Clean Abstraction Boundaries**: Pipeline orchestrates, kernels compute, selector chooses backend

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


### 4. Adaptive MatMul & Prefill Backend

**File**: `src/adaptive_matmul.h`
**Concepts**:
- `AdaptiveMatMulManager` chooses backend per (m,n,k,is_prefill, MPI size, env overrides).
- Prefill: Potential COSMA offload if seq_len ≥ threshold & world_size>1.
- Decode: Always OpenBLAS (saves collective overhead > latency).
- Supports batched Q/K/V path (currently OpenBLAS-loop; future fused COSMA batch candidate).

### 4. Backend Selection & Adaptive MatMul

**Files**: `src/matmul_backend_selection.{h,cpp}`, `src/prefill_backend.{h,cpp}`, `src/inference_backend.{h,cpp}`

The backend selection system provides centralized policy for choosing between OpenBLAS and COSMA based on operation characteristics, system configuration, and execution phase.

#### MatMulBackendSelector

**Purpose**: Centralized decision logic for all matrix multiplication operations

**Selection Criteria**:
- **Operation Size**: Total elements (m × n × k)
- **Execution Phase**: Prefill vs decode
- **Sequence Length**: Token count in current batch
- **MPI Context**: Number of ranks available
- **Environment Overrides**: `ADAPTIVE_DISABLE_COSMA`, `LLAMINAR_COSMA_PREFILL_THRESHOLD`

**Decision Logic**:
```cpp
MatMulBackend selectBackend(int m, int n, int k, 
                           int seq_len, 
                           bool is_prefill) {
    // Very small: single-threaded OpenBLAS (minimize overhead)
    if (total_elements < 8192) {
        return MatMulBackend::SINGLE_THREADED_OPENBLAS;
    }
    
    // Large prefill: COSMA for distributed throughput
    if (is_prefill && seq_len >= cosma_threshold && 
        total_elements >= 8388608) {
        return MatMulBackend::DISTRIBUTED_COSMA;
    }
    
    // Medium: multi-threaded OpenBLAS
    return MatMulBackend::MULTI_THREADED_OPENBLAS;
}
```

#### Backend Factories

**PrefillBackend** (`src/prefill_backend.{h,cpp}`):
- Handles large batch token processing
- Delegates to `MatMulBackendSelector` for policy
- Routes to COSMA path via `cosma_prefill_manager` when selected
- Falls back to OpenBLAS on COSMA failure/validation errors

**InferenceBackend** (`src/inference_backend.{h,cpp}`):
- Handles single-token decode operations
- Always uses local OpenBLAS (latency-optimized)
- Avoids MPI collective overhead
- Single-threaded for minimal context switch overhead

#### Performance Characteristics

Based on empirical benchmarking:

| Operation Size | Tokens | Backend | Reason |
|----------------|--------|---------|--------|
| < 8K elements | Any | Single-threaded OpenBLAS | Overhead dominates |
| 8K - 8M elements | < 4K | Multi-threaded OpenBLAS | Communication cost too high |
| ≥ 8M elements | ≥ 4K | COSMA (if enabled) | Distributed throughput wins |
| Decode (any) | 1 | Single-threaded OpenBLAS | Latency critical |

#### Environment Controls

- `ADAPTIVE_DISABLE_COSMA=1`: Force all operations to OpenBLAS
- `LLAMINAR_COSMA_PREFILL_THRESHOLD=<tokens>`: Override sequence length threshold (default: 4096)
- `LLAMINAR_COSMA_MAX_RESIDENT_MB=<mb>`: Memory budget for COSMA operations (default: 2048)
- `LLAMINAR_COSMA_VALIDATE_TILE=<size>`: Enable correctness validation (debug only)

#### Integration with Pipeline

The pipeline uses backend selection through kernel composition:

```cpp
// In QwenPipeline::prefill()
auto backend_choice = MatMulBackendSelector::selectBackend(
    seq_len, hidden_size, proj_size, seq_len, true);

if (backend_choice == MatMulBackend::DISTRIBUTED_COSMA) {
    // Route through COSMA prefill manager
    executePrefillAttentionCosma(...);
} else {
    // Use local OpenBLAS kernel
    executeAttentionLocal(...);
}
```

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

- Called from `QwenPipeline::prefill()` for attention QKV projection
- Used for MLP gate/up/down projections in large prefill
- Coordinates with `MatMulBackendSelector` for policy enforcement
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
- Coordinates with `MatMulBackendSelector` to avoid COSMA on partial matrices
- Ensures consistent execution with adaptive backend policy

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