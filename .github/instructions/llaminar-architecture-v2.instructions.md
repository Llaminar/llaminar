# Llaminar V2 Architecture

This document is a high-level but concrete map of the Llaminar V2 stack: tensors, kernels, devices, inference execution, MPI orchestration, and attention. It is intended as a **quick-start reference for future agents** so they can safely modify V2 without re-deriving the architecture from scratch.

---

## 1. Design Goals

V2 is a **kernel-centric, operator-free** architecture:

- **Per-tensor device affinity** – each tensor knows which device it lives on.
- **Declarative graph execution** – `GraphOrchestrator` executes DAGs of compute stages.
- **Heterogeneous execution** – CPU / CUDA / ROCm / (future) backends can be mixed in one run.
- **Quantization-aware kernels** – unified GEMM/attention interfaces that work with FP32/BF16 and quantized formats.
- **MPI-aware orchestration** – multi-rank inference (tensor parallelism) lives in orchestrators, not kernels.
- **Centralized kernel dispatch** – `KernelFactory` provides unified kernel creation with caching.
- **Weight sharding** – automatic tensor parallelism distributes weight matrices across MPI ranks.
- **Graph caching** – `Qwen2Graph` builds DAGs that are cached and reused during decode for performance.

The mental model:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          INFERENCE LAYER                                     │
│                                                                              │
│   createInferenceRunner()  ──→  IInferenceRunner (GraphOrchestrator)        │
│                                      │                                       │
│                                      ▼                                       │
│                        ┌─────────────────────────┐                          │
│                        │   GraphOrchestrator     │                          │
│                        │  (Declarative graphs)   │                          │
│                        └─────────────────────────┘                          │
│                                      │                                       │
│                                      │ DAG stages                            │
│                                      ▼                                       │
│                        ┌─────────────────────────┐                          │
│                        │   ComputeGraph DAG      │                          │
│                        │  + GraphExecutor        │                          │
│                        └─────────────────────────┘                          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           KERNEL LAYER                                       │
│                                                                              │
│   KernelFactory  ──→  ITensorGemm, ITensorAttention, IRMSNorm, etc.         │
│                                                                              │
│   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐ │
│   │  CPU Kernels    │  │  CUDA Kernels   │  │  Quantized Kernels         │ │
│   │  (OpenBLAS/MKL) │  │  (future)       │  │  (IQ4_NL, Q8_0, Q6_K...)   │ │
│   └─────────────────┘  └─────────────────┘  └─────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TENSOR LAYER                                       │
│                                                                              │
│   TensorBase  ──→  FP32Tensor, BF16Tensor, IQ4_NLTensor, Q8_0Tensor, etc.   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Inference Interface

### 2.1 IInferenceRunner

Location: `src/v2/inference/IInferenceRunner.h`

The `IInferenceRunner` interface provides the public API for inference execution:

```cpp
class IInferenceRunner {
public:
    virtual ~IInferenceRunner() = default;
    
    // Core inference
    virtual bool forward(const int* tokens, int seq_len) = 0;
    virtual const float* logits() const = 0;
    
    // State management
    virtual void clear_cache() = 0;
    virtual int get_position() const = 0;
    
    // Metadata
    virtual int vocab_size() const = 0;
    virtual const char* architecture() const = 0;
    
    // Snapshot capture (for E2E testing)
    virtual void enableSnapshotCapture(const std::string& dir) = 0;
    virtual std::vector<std::string> getSnapshotKeys() const = 0;
    virtual const float* getSnapshot(const std::string& key, size_t& size) const = 0;
};
```

### 2.2 Factory Function

Location: `src/v2/inference/InferenceRunner.h`

```cpp
// Factory creates a GraphOrchestrator-based runner
std::unique_ptr<IInferenceRunner> createInferenceRunner(
    std::shared_ptr<ModelContext> model_ctx,
    std::shared_ptr<MPIContext> mpi_ctx,
    int device_idx,
    const InferenceRunnerConfig& config = {});
```

The factory creates a `GraphOrchestrator` for the model architecture (e.g., Qwen2).

### 2.3 Usage Pattern

```cpp
// Create runner
auto runner = createInferenceRunner(model_ctx, mpi_ctx, device_idx);

// Inference
runner->forward(tokens.data(), seq_len);
const float* logits = runner->logits();

// Sampling
int next_token = argmax(logits, runner->vocab_size());

// Decode loop
while (next_token != eos_token) {
    runner->forward(&next_token, 1);
    next_token = argmax(runner->logits(), runner->vocab_size());
}

// Reset for new sequence
runner->clear_cache();
```

---

## 3. GraphOrchestrator System

### 3.1 Architecture Overview

The **GraphOrchestrator** is the graph-based execution system for transformer inference. It orchestrates declarative compute graphs for high-performance execution with:

- **Declarative graph construction** – Build computation as a DAG of stages
- **Automatic dependency tracking** – Stages execute in correct order
- **Graph caching** – Pre-built graphs reused across decode steps
- **MPI-aware execution** – `AllreduceStage` handles tensor-parallel synchronization
- **State management** – Owns KV cache, position tracking, and activation buffers

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        GraphOrchestrator                                     │
│                                                                              │
│   ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────────────┐  │
│   │  InferenceState │   │  GraphExecutor  │   │   LayerGraphCache       │  │
│   │  - hidden       │   │  - execute()    │   │   - attention_decode    │  │
│   │  - logits       │   │  - topo sort    │   │   - ffn_decode          │  │
│   │  - kv_cache     │   │                 │   │   - per-layer caching   │  │
│   │  - positions    │   │                 │   │                         │  │
│   └─────────────────┘   └─────────────────┘   └─────────────────────────┘  │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  Qwen2Graph (IGraphBuilder)                                         │   │
│   │  - buildAttentionGraph() → ComputeGraph                             │   │
│   │  - buildFFNGraph() → ComputeGraph                                   │   │
│   │  - Declarative stage definition                                      │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 ComputeGraph and Stages

Location: `src/v2/execution/ComputeGraph.h`, `src/v2/execution/ComputeStage.h`

A **ComputeGraph** is a DAG of `IComputeStage` nodes:

```cpp
class ComputeGraph {
public:
    void addNode(const std::string& name, std::unique_ptr<IComputeStage> stage, int device_idx);
    void addDependency(const std::string& dependent, const std::string& dependency);
    std::vector<std::string> getExecutionOrder() const;  // Topological sort
    void reset();  // Reset all stages for reuse
};
```

**Available Stage Types:**

| Stage Class | Purpose | MPI Sync? |
|-------------|---------|-----------|
| `RMSNormStage` | RMS normalization | No |
| `GEMMStage` | Matrix multiplication | No |
| `SwiGLUStage` | SwiGLU activation | No |
| `RoPEStage` | Rotary position embeddings | No |
| `ResidualAddStage` | Residual connections | No |
| `AttentionComputeStage` | Full attention computation | No |
| `KVCacheAppendStage` | KV cache management | No |
| `AllreduceStage` | **MPI synchronization** | **Yes** |

### 3.3 Execution Flow

```
forward(tokens, seq_len)
        │
        ▼
┌───────────────────────────────────────┐
│  1. Embedding Lookup                  │
│     token_ids → hidden [seq, d_model] │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│  2. For each layer (0..N-1):          │
│                                       │
│     ┌─────────────────────────────┐   │
│     │  executeAttention()         │   │
│     │  - Build/cache graph        │   │
│     │  - RMSNorm → Q/K/V → RoPE   │   │
│     │  - KV Cache → Attention     │   │
│     │  - Wo projection            │   │
│     │  - AllreduceStage (if MPI)  │   │
│     │  - Residual add             │   │
│     └─────────────────────────────┘   │
│              │                        │
│              ▼                        │
│     ┌─────────────────────────────┐   │
│     │  executeFFN()               │   │
│     │  - Build/cache graph        │   │
│     │  - RMSNorm → Gate/Up        │   │
│     │  - SwiGLU → Down            │   │
│     │  - AllreduceStage (if MPI)  │   │
│     │  - Residual add             │   │
│     └─────────────────────────────┘   │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│  3. Final Norm + LM Head              │
│     hidden → logits [seq, vocab]      │
└───────────────────────────────────────┘
        │
        ▼
    return logits
```

### 3.4 MPI Tensor Parallelism

The GraphOrchestrator implements **Megatron-style tensor parallelism** where weight matrices are split across MPI ranks:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RANK 0                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  GraphOrchestrator (owns InferenceState, GraphExecutor)              │    │
│  │  ┌──────────────────────────────────────────────────────────────┐   │    │
│  │  │  ComputeGraph (FFN layer)                                     │   │    │
│  │  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌──────────────┐ ┌───────┐ │   │    │
│  │  │  │ RMSNorm│→│Gate/Up │→│ SwiGLU │→│ Down GEMM    │→│Allreduce│   │    │
│  │  │  └────────┘ └────────┘ └────────┘ └──────────────┘ └───────┘ │   │    │
│  │  └──────────────────────────────────────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  Weights (SHARD 0):                                                          │
│  • ffn_gate: [d_ff/2, d_model]  ← first half of rows                        │
│  • ffn_up:   [d_ff/2, d_model]  ← first half of rows                        │
│  • ffn_down: [d_model, d_ff/2]  ← first half of columns                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                   │
                                   │ MPI_Allreduce (SUM)
                                   ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RANK 1                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  GraphOrchestrator ← SEPARATE INSTANCE, IDENTICAL GRAPH STRUCTURE    │    │
│  │  ┌──────────────────────────────────────────────────────────────┐   │    │
│  │  │  ComputeGraph (structurally identical)                        │   │    │
│  │  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌──────────────┐ ┌───────┐ │   │    │
│  │  │  │ RMSNorm│→│Gate/Up │→│ SwiGLU │→│ Down GEMM    │→│Allreduce│   │    │
│  │  │  └────────┘ └────────┘ └────────┘ └──────────────┘ └───────┘ │   │    │
│  │  └──────────────────────────────────────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  Weights (SHARD 1):                                                          │
│  • ffn_gate: [d_ff/2, d_model]  ← second half of rows                       │
│  • ffn_up:   [d_ff/2, d_model]  ← second half of rows                       │
│  • ffn_down: [d_model, d_ff/2]  ← second half of columns                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key Points:**
- Each rank has its own `GraphOrchestrator` instance
- Graphs are structurally identical across ranks
- Weight tensors point to different shards
- `AllreduceStage` synchronizes partial results after Wo and Down projections

### 3.5 Graph Caching (Decode Optimization)

For decode mode (seq_len=1), graphs are cached and reused:

```cpp
bool GraphOrchestrator::executeAttention(...) {
    if (graph_caching_enabled_ && seq_len == 1) {
        auto& cache = layer_graph_cache_[layer_idx];
        
        if (cache.attention_decode && cache.valid) {
            // Update only dynamic parameters (position offset)
            updateCachedGraphParams(*cache.attention_decode, pos_offset, seq_len);
            
            // Execute cached graph
            bool success = executor_.execute(*cache.attention_decode, ctx);
            cache.attention_decode->reset();
            return success;
        }
        
        // Build graph and cache for future reuse
        cache.attention_decode = std::make_unique<ComputeGraph>(
            graph_builder_->buildAttentionGraph(...));
        cache.valid = true;
    }
    // Execute (cached or freshly built)
}
```

**Benefits:**
- Eliminates graph construction overhead during decode
- Position offset updated via `updateDynamicParams()` without rebuilding
- ~10-20% decode speedup for long sequences

### 3.6 InferenceState

The `GraphOrchestrator` owns all mutable inference state:

```cpp
struct InferenceState {
    // Core buffers
    std::shared_ptr<TensorBase> hidden;    // [batch * seq, d_model]
    std::shared_ptr<TensorBase> logits;    // [batch * seq, vocab_size]
    
    // KV Cache
    std::unique_ptr<IUnifiedKVCache> kv_cache;
    
    // Position tracking
    std::vector<int> positions;           // Per-sequence position offset
    std::vector<int> sequence_lengths;    // For variable-length batches
    
    // Activation buffers
    std::shared_ptr<TensorBase> normalized, residual;
    std::shared_ptr<TensorBase> Q, K, V;
    std::shared_ptr<TensorBase> attn_output, attn_proj;
    std::shared_ptr<TensorBase> gate, up, ffn_output;
    
    // Attention workspace
    std::shared_ptr<TensorBase> workspace_scores, workspace_context, workspace_mask;
};
```

---

## 4. Tensors and Tensor Interfaces

### 4.1 Core Tensor Types

Location: `src/v2/tensors/`

- `TensorBase` – abstract base: shape, dtype, device, virtual hooks to create kernels
- Concrete tensors:
  - `FP32Tensor`, `BF16Tensor`, `FP16Tensor` – dense float tensors
  - Quantized: `IQ4_NLTensor`, `Q4_0Tensor`, `Q6_KTensor`, `Q8_0Tensor`, etc.
  - View/alias tensors for cheap slicing

Each tensor exposes factory methods for kernels:
```cpp
std::unique_ptr<ITensorGemm> createGemm() const;
std::unique_ptr<ITensorAttention> createAttention() const;
```

### 4.1.1 TypedTensorBase and `typed_data()` Pattern

All 27 tensor classes inherit from `TypedTensorBase<Derived, DataType>`, a CRTP base that provides
**zero-overhead typed access** to native storage:

```cpp
template<typename Derived, typename DataType>
class TypedTensorBase {
public:
    const DataType* typed_data() const;        // Native type access
    DataType* mutable_typed_data();            // Mutable native type access
};
```

**Usage Pattern** – After `dynamic_cast<>`, use `typed_data()` for consistent access:

```cpp
// KV cache copy (UnifiedKVCache.cpp)
template <>
void UnifiedKVCache<ActivationPrecision::Q8_1>::copy_append_data(
    Q8_1Tensor *dst, const Q8_1Tensor *src, int offset, int tokens)
{
    Q8_1Block *dst_blocks = dst->mutable_typed_data();  // ✅ Unified pattern
    const Q8_1Block *src_blocks = src->typed_data();
    std::memcpy(dst_blocks + offset, src_blocks, tokens * sizeof(Q8_1Block));
}

// GEMM kernel (QuantisedGemmKernel.h)
if (c_type == TensorType::Q8_1) {
    auto *A_q8 = static_cast<const Q8_1Tensor *>(A);
    auto *C_q8 = static_cast<Q8_1Tensor *>(C);
    return multiply_q8_1_to_q8_1(
        A_q8->typed_data(),           // ✅ const Q8_1Block*
        C_q8->mutable_typed_data(),   // ✅ Q8_1Block*
        m, n, k);
}
```

**Supported Types**:
| Tensor Class | `typed_data()` Returns |
|--------------|------------------------|
| `FP32Tensor` | `float*` |
| `BF16Tensor`, `FP16Tensor` | `uint16_t*` |
| `Q8_1Tensor` | `Q8_1Block*` |
| `Q16_1Tensor` | `Q16_1Block*` |
| `Q8_0Tensor`, `Q4_0Tensor`, etc. | Respective block types |

**Legacy Accessors** (deprecated but still available):
- `q8_1_blocks()` / `mutable_q8_1_blocks()` → Use `typed_data()` instead
- `bf16_data()` / `mutable_bf16_data()` → Use `typed_data()` instead
- `fp16_data()` / `mutable_fp16_data()` → Use `typed_data()` instead

### 4.2 Tensor Kernel Interfaces

Location: `src/v2/tensors/TensorKernels.h`

- `ITensorGemm` – GEMM between activations and/or weights
- `ITensorAttention` – Attention over Q/K/V
- Other per-op interfaces (RMSNorm, SwiGLU, RoPE)

---

## 5. Kernels and KernelFactory

### 5.1 CPU Kernels

Location: `src/v2/kernels/cpu/`

- GEMM: oneDNN / OpenBLAS / AVX-512 VNNI JIT
- Attention: `CpuAttentionKernelT<T>` for FP32/BF16
- Vectorized primitives in `primitives/`

### 5.2 Quantized GEMM and Block Decoders

Quantized tensors implement `IBlockDecoder`:
```cpp
class IBlockDecoder {
    virtual void decode_block_at(size_t row, size_t k_block, float* out) const = 0;
    virtual size_t block_size() const = 0;
};
```

A generic `QuantizedGemmKernel` uses `IBlockDecoder` strategy pattern:
- One implementation works for all quantized formats
- `decode_block_at` is `always_inline` for zero virtual call overhead

### 5.3 KernelFactory

Location: `src/v2/kernels/KernelFactory.h`

Centralized kernel dispatch with caching:

```cpp
class KernelFactory {
    static DeviceType getDeviceType(int device_idx);
    static ITensorGemm* getOrCreateGemm(const TensorBase* tensor);  // Cached
    static void clearCacheFor(const TensorBase* tensor);            // Auto cleanup
    static void clearCache();                                        // Manual cleanup
};
```

**Usage:**
```cpp
// Preferred: cached kernel (pack once, use many)
ITensorGemm* gemm = KernelFactory::getOrCreateGemm(weight_tensor.get());
gemm->multiply(activations, output, m, n, k);
```

---

## 6. MPI Layer

### 6.1 MPIContext

Location: `src/v2/utils/MPIContext.h`

```cpp
class MPIContext {
    int rank() const;
    int world_size() const;
    void allreduce_sum(float* buffer, size_t count);
    void broadcast(void* data, size_t count, int root);
    void barrier();
};
```

### 6.2 Weight Sharding

Location: `src/v2/loaders/WeightManager.h`

Automatic tensor parallelism:

| Weight Pattern | Sharding Mode | Rationale |
|----------------|---------------|-----------|
| `attn_output.weight` (Wo) | ROW_PARALLEL | Split input dim, allreduce output |
| `ffn_down.weight` | ROW_PARALLEL | Split input dim, allreduce output |
| QKV, Gate/Up, norms | REPLICATE | Full tensors needed for attention |

**Memory Savings (2 ranks):** ~50% reduction for sharded weights

---

## 7. Attention

### 7.1 ITensorAttention

Location: `src/v2/tensors/TensorKernels.h`

```cpp
class ITensorAttention {
    virtual bool compute(
        const float* Q, const float* K, const float* V, float* output,
        int seq_len, int n_heads, int n_kv_heads, int head_dim,
        bool causal, int window_size, /* workspaces */, MPIContext* mpi_ctx) = 0;
};
```

### 7.2 AttentionComputeStage

Location: `src/v2/execution/stages/AttentionComputeStage.h`

The attention compute stage encapsulates full attention computation:
- RoPE application to Q/K
- KV cache append/gather
- Attention scoring with causal masking
- Context aggregation

Created via `KernelFactory::createAttention()` which dispatches to the appropriate kernel based on tensor type and device.

---

## 8. Snapshot System

### 8.1 Overview

The snapshot system captures intermediate tensor states for debugging and parity testing:

- **Compile-time conditional** – Only in Debug/E2ERelease builds
- **In-memory storage** – `std::map<std::string, std::vector<float>>`
- **Zero overhead in Release** – Compiles away to NOOPs

### 8.2 Usage

```cpp
// Enable capture
runner->enableSnapshotCapture("/tmp/snapshots");

// Run inference
runner->forward(tokens, seq_len);

// Retrieve snapshots
auto keys = runner->getSnapshotKeys();
for (const auto& key : keys) {
    size_t size;
    const float* data = runner->getSnapshot(key, size);
    // Compare with reference...
}
```

### 8.3 Snapshot Keys

- **Global:** `EMBEDDING`, `FINAL_NORM`, `LM_HEAD`
- **Per-layer:** `layer{i}_{STAGE}` where stage is:
  - Attention: `ATTENTION_NORM`, `Q_PROJECTION`, `K_PROJECTION`, `V_PROJECTION`, `Q_ROPE`, `K_ROPE`, `ATTENTION_CONTEXT`, `ATTENTION_OUTPUT`, `ATTENTION_RESIDUAL`
  - FFN: `FFN_NORM`, `FFN_GATE`, `FFN_UP`, `FFN_SWIGLU`, `FFN_DOWN`, `FFN_RESIDUAL`

---

## 9. Quick Reference

### 9.1 File Locations

| Component | Location |
|-----------|----------|
| Inference interface | `src/v2/inference/IInferenceRunner.h` |
| Inference factory | `src/v2/inference/InferenceRunner.{h,cpp}` |
| GraphOrchestrator | `src/v2/pipelines/qwen/GraphOrchestrator.{h,cpp}` |
| Graph builder | `src/v2/pipelines/qwen/Qwen2Graph.{h,cpp}` |
| Buffer allocation | `src/v2/pipelines/qwen/Qwen2BufferSpec.{h,cpp}` |
| ComputeGraph/Stages | `src/v2/execution/` |
| KernelFactory | `src/v2/kernels/KernelFactory.{h,cpp}` |
| Tensors | `src/v2/tensors/` |
| CPU kernels | `src/v2/kernels/cpu/` |
| MPI utilities | `src/v2/utils/MPIContext.h`, `src/v2/utils/MPITopology.h` |
| Weight sharding | `src/v2/loaders/WeightManager.{h,cpp}` |
| Placement strategy | `src/v2/execution/PlacementStrategy.h` |

### 9.2 Key Design Rules

1. **Use `IInferenceRunner`** for inference – Don't call `GraphOrchestrator` directly from Main/ChatUI
2. **Keep MPI out of kernels** – MPI sync lives in `AllreduceStage` and `AllGatherStage`
3. **Use KernelFactory** – Prefer `getOrCreateGemm()` for cached kernel access
4. **Declarative graphs** – Build computation as DAGs of `ComputeStage` nodes
5. **Graph caching** – Reuse cached graphs in decode mode for performance

### 9.3 Environment Variables

| Variable | Effect |
|----------|--------|
| `LLAMINAR_PROFILE_KERNELS=1` | Enable per-kernel timing in benchmark mode |
| `LLAMINAR_LOG_LEVEL=DEBUG` | Set logging verbosity |
| `LLAMINAR_SNAPSHOT_TENSOR_DUMP=1` | Enable raw tensor dumps for debugging |

### 9.4 Running Tests

```bash
# Unit tests
ctest --test-dir build_v2 -R "^V2_Unit_" --output-on-failure --parallel

# Integration tests
ctest --test-dir build_v2_release -R "^V2_Integration_" --output-on-failure

# E2E parity tests (requires E2ERelease build)
ctest --test-dir build_v2_e2e_release -R "^V2_E2E_" --output-on-failure
```

---

This architecture is intentionally modular: small, focused abstractions connected by narrow interfaces. The `IInferenceRunner` interface ensures that callers (Main.cpp, ChatUI, tests) remain decoupled from implementation details of the `GraphOrchestrator`.

---

## 10. MPI Tensor Parallelism Details

### 10.1 Sharding Patterns

Llaminar implements **Megatron-style tensor parallelism** with automatic weight sharding:

| Weight | Sharding Mode | Description |
|--------|---------------|-------------|
| `attn_q.weight`, `attn_k.weight`, `attn_v.weight` | COLUMN_PARALLEL | Split output dim (heads) |
| `attn_output.weight` (Wo) | ROW_PARALLEL | Split input dim, allreduce after |
| `ffn_gate.weight`, `ffn_up.weight` | COLUMN_PARALLEL | Split output dim (d_ff) |
| `ffn_down.weight` | INPUT_PARALLEL | Split input dim, allreduce after |
| `output.weight` (LM head) | COLUMN_PARALLEL | Split vocab, allgather logits |
| Norms, embeddings | REPLICATE | Full copy on each rank |

### 10.2 MPI Synchronization Stages

| Stage | When Used | Operation |
|-------|-----------|----------|
| `AllreduceStage` | After Wo, after Down | MPI_Allreduce(SUM) |
| `AllGatherStage` | After LM head | MPI_Allgather for full logits |

### 10.3 MPITopology

Location: `src/v2/utils/MPITopology.h`

Manages work distribution across ranks:

```cpp
class MPITopology {
    // Work range calculations
    WorkRange get_head_range(int total_heads) const;     // For attention
    WorkRange get_column_range(int total_cols) const;    // For column-parallel
    WorkRange get_vocab_range(int vocab_size) const;     // For LM head
    
    // Device capabilities
    const std::vector<RankPlacement>& all_placements() const;
};
```
