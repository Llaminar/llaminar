# Llaminar V2 Architecture Reference

This document provides an architecture-level reference for agents working with **Llaminar V2**, the operator-free, kernel-centric LLM inference engine. It's designed to give context when navigating the codebase and understanding how components connect.

> **For development guidelines**, build commands, testing procedures, and debugging instructions, see `.github/copilot-instructions.md`.
> 
> **For execution scenario diagrams** including Memory Management Flow, Workspace Lifecycle, and multi-rank scenarios, see `docs/v2/ARCHITECTURE_EXECUTION_SCENARIOS.md`.

---

## 1. Design Goals and Mental Model

### 1.1 Design Goals

V2 is a **kernel-centric, operator-free** architecture:

1. **Per-tensor device affinity** – each tensor knows which device it lives on (CPU, CUDA:0, ROCm:1)
2. **Declarative graph execution** – `GraphOrchestrator` executes DAGs of compute stages
3. **Heterogeneous execution** – CPU / CUDA / ROCm backends can be mixed in one inference run
4. **Quantization-aware kernels** – unified GEMM/attention interfaces for FP32/BF16 and quantized formats
5. **MPI-aware orchestration** – multi-rank tensor parallelism lives in orchestrators, not kernels
6. **Centralized kernel dispatch** – `KernelFactory` provides unified kernel creation with caching
7. **Automatic weight sharding** – Megatron-style tensor parallelism distributes weights across MPI ranks
8. **Graph caching** – `LayerGraphCache` builds DAGs that are cached and reused during decode
9. **Zero-allocation hot path** – all buffers pre-allocated via `GraphBufferManager` + `DeviceWorkspaceManager`
10. **Automatic coherence** – GPU↔CPU memory sync handled by `StageCoherence` at stage boundaries

### 1.2 Mental Model

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          APPLICATION LAYER                                   │
│                                                                              │
│   llaminar2 CLI  ──→  InferenceRunnerFactory::createInferenceRunner()       │
│                              │                                               │
│                              ▼                                               │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  IInferenceRunner (GraphOrchestrator implements)                    │   │
│   │    • forward(tokens, seq_len) → logits                             │   │
│   │    • enableSnapshotCapture() → parity testing                      │   │
│   │    • getSnapshot(key) → per-stage tensor data                      │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ORCHESTRATION LAYER                                   │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  GraphOrchestrator                                                  │   │
│   │    • InferenceState (hidden, logits, kv_cache, activations)        │   │
│   │    • LayerGraphCache (decode-mode graph reuse)                     │   │
│   │    • Qwen2Graph (IGraphBuilder - declarative graph construction)   │   │
│   │    • CollectiveContext (ICollectiveContext - MPI backend routing)  │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                      │                                       │
│         ┌────────────────────────────┼────────────────────────────┐          │
│         ▼                            ▼                            ▼          │
│   ┌──────────────────┐   ┌─────────────────────────┐   ┌──────────────────┐ │
│   │   MPITopology    │   │   PlacementStrategy     │   │   BackendRouter  │ │
│   │ (IMPITopology)   │   │ (device→stage mapping)  │   │ (NCCL/RCCL/MPI)  │ │
│   └──────────────────┘   └─────────────────────────┘   └──────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                             GRAPH LAYER                                      │
│                                                                              │
│   ┌───────────────────────┐        ┌────────────────────────────────────┐   │
│   │  ComputeGraph (DAG)   │        │  GraphBufferManager                │   │
│   │    • ComputeNode      │        │    • LivenessAnalyzer (aliasing)   │   │
│   │    • dependencies     │        │    • DeviceWorkspaceManager        │   │
│   │    • getExecutionOrder│        │    • WorkspaceBudgetConfig         │   │
│   └───────────────────────┘        └────────────────────────────────────┘   │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  GraphExecutor (IGraphExecutor)                                     │   │
│   │    • execute(graph, ctx) → topological order                        │   │
│   │    • StageCoherence (GPU↔CPU sync at boundaries)                   │   │
│   │    • TensorVerification (debug builds: NaN/zero detection)         │   │
│   │    • setSnapshotCallback() → parity test hooks                     │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         COMPUTE STAGE LAYER                                  │
│                                                                              │
│   IComputeStage Interface:                                                   │
│     • execute(ctx) → run computation                                         │
│     • type() → ComputeStageType enum                                        │
│     • getDumpInfo() → StageDumpInfo (inputs/outputs/weights for debugging)  │
│     • coherencePolicy() → FULL/INPUT/OUTPUT/NONE                            │
│                                                                              │
│   ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌───────────┐│
│   │GEMM Stages │ │Attn Stages │ │Norm Stages │ │Activ Stages│ │MPI Stages ││
│   ├────────────┤ ├────────────┤ ├────────────┤ ├────────────┤ ├───────────┤│
│   │GEMMStage   │ │FusedAttnWo │ │RMSNormStage│ │SwiGLUStage │ │Allreduce  ││
│   │FusedQKV    │ │AttnCompute │ │EmbedStage  │ │QuantQ16_1  │ │AllGather  ││
│   │FusedGateUp │ │KVCacheOps  │ │LMHeadStage │ │RoPEStage   │ │           ││
│   └────────────┘ └────────────┘ └────────────┘ └────────────┘ └───────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           KERNEL LAYER                                       │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  KernelFactory (centralized dispatch with caching)                  │   │
│   │    • getDeviceType(DeviceId) → DeviceType (CPU/CUDA/ROCm)          │   │
│   │    • getOrCreateGemm(tensor) → ITensorGemm* (cached)               │   │
│   │    • createKVCache(config) → IKVCache                              │   │
│   │    • createAttention(config) → ITensorAttention                    │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│   Kernel Interfaces:                                                         │
│   ┌──────────────┐ ┌──────────────────┐ ┌──────────────┐ ┌───────────────┐  │
│   │ ITensorGemm  │ │ ITensorAttention │ │ ITensorRoPE  │ │IWorkspaceCons │  │
│   │ multiply()   │ │ compute()        │ │ apply()      │ │BindWorkspace()│  │
│   └──────────────┘ └──────────────────┘ └──────────────┘ └───────────────┘  │
│                                                                              │
│   Backend Implementations:                                                   │
│   ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐            │
│   │   CPU Kernels    │ │   CUDA Kernels   │ │   ROCm Kernels   │            │
│   │ OpenBLAS/MKL GEMM│ │ cuBLAS GEMM      │ │ rocBLAS GEMM     │            │
│   │ JIT Attention    │ │ FlashAttention   │ │ comp_kernel      │            │
│   │ AVX-512 VNNI     │ │ TensorCore WMMA  │ │ MatrixCore       │            │
│   │ QuantizedGEMM    │ │ CUDAQuantGEMM    │ │ ROCmQuantGEMM    │            │
│   └──────────────────┘ └──────────────────┘ └──────────────────┘            │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           BACKEND LAYER                                      │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  BackendManager (unified device memory interface)                   │   │
│   │    • getBackendFor(DeviceId) → IBackend*                           │   │
│   │    • getCPUBackend(), getCUDABackend(), getROCmBackend()           │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│   IBackend Interface:                                                        │
│     • allocate(size) → void*, free(ptr)                                     │
│     • deviceToHost(dst, src, size), hostToDevice(dst, src, size)            │
│     • deviceMemoryTotal(), deviceMemoryFree()                               │
│     • synchronize()                                                          │
│                                                                              │
│   ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐               │
│   │   CPUBackend    │ │  CUDABackend    │ │  ROCmBackend    │               │
│   │ Rank-local NUMA │ │ cudaMalloc      │ │ hipMalloc       │               │
│   │ 64B alignment   │ │ cudaMemcpy      │ │ hipMemcpy       │               │
│   └─────────────────┘ └─────────────────┘ └─────────────────┘               │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TENSOR LAYER                                       │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  ITensor (runtime polymorphism interface)                           │   │
│   │    • native_type(), dtype_name()                                   │   │
│   │    • shape(), rows(), cols(), numel(), size_bytes()               │   │
│   │    • home_device() → DeviceId                                      │   │
│   │    • typed_as<T>(), try_as<T>()                                   │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  CPUTensorBase (device coherence + host storage)                    │   │
│   │    • ensureOnDevice(device) → upload to GPU                         │   │
│   │    • mark_device_dirty() → mark GPU as authoritative                │   │
│   │    • data() → host data (syncs from GPU if device-dirty)           │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  TypedTensorBase<Derived, DataType> (CRTP zero-overhead access)     │   │
│   │    • typed_data() → const DataType*                                │   │
│   │    • mutable_typed_data() → DataType*                              │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│   Concrete Types:                                                            │
│   ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌───────────────────┐  │
│   │ Activations  │ │Quant Weights │ │ KV Cache     │ │ Special Purpose   │  │
│   ├──────────────┤ ├──────────────┤ ├──────────────┤ ├───────────────────┤  │
│   │ FP32Tensor   │ │ IQ4_NLTensor │ │ Q16_1Tensor  │ │ INT32Tensor (acc) │  │
│   │ FP16Tensor   │ │ Q4_0Tensor   │ │ Q8_1Tensor   │ │ INT8Tensor        │  │
│   │ BF16Tensor   │ │ Q6_KTensor   │ │              │ │                   │  │
│   └──────────────┘ └──────────────┘ └──────────────┘ └───────────────────┘  │
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
    
    // Snapshot capture (for parity testing)
    virtual void enableSnapshotCapture(const std::string& dir) = 0;
    virtual std::vector<std::string> getSnapshotKeys() const = 0;
    virtual const float* getSnapshot(const std::string& key, size_t& size) const = 0;
};
```

### 2.2 Factory Functions

Location: `src/v2/execution/InferenceRunnerFactory.h`

```cpp
// Standard factory - creates GraphOrchestrator with full model context
std::unique_ptr<IInferenceRunner> createInferenceRunner(
    std::shared_ptr<IModelContext> model_ctx,
    std::shared_ptr<MPIContext> mpi_ctx,
    DeviceId device,                               // DeviceId (not int)
    const InferenceRunnerConfig& config = {});

// Testable factory - allows injecting mock IModelContext for unit tests
std::unique_ptr<IInferenceRunner> createTestableInferenceRunner(
    IModelContext* model_ctx,                      // Non-owning pointer for DI
    DeviceId device,
    const InferenceRunnerConfig& config = {});
```

**InferenceRunnerConfig:**
```cpp
struct InferenceRunnerConfig {
    bool enable_benchmarking = false;              // Enable timing instrumentation
    bool enable_tensor_parallelism = true;         // MPI sharding (auto if world_size > 1)
    size_t max_context_length = 4096;              // Maximum sequence length
    std::string cache_dir = "";                    // Optional: snapshot/cache directory
};
```

### 2.3 IModelContext Interface

Location: `src/v2/interfaces/IModelContext.h`

The `IModelContext` abstracts model metadata for testing:

```cpp
class IModelContext {
public:
    virtual int n_vocab() const = 0;
    virtual int n_embd() const = 0;
    virtual int n_head() const = 0;
    virtual int n_kv_head() const = 0;
    virtual int n_layer() const = 0;
    virtual int n_ff() const = 0;
    virtual int n_ctx() const = 0;
    virtual float rope_freq_base() const = 0;
    virtual const char* arch() const = 0;
    
    // Weight access (returns nullptr for mock contexts)
    virtual ITensor* get_weight(const std::string& name) const = 0;
};
```

### 2.4 Usage Pattern

```cpp
// Create runner
auto runner = createInferenceRunner(model_ctx, mpi_ctx, DeviceId::cpu());

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
- **Automatic dependency tracking** – Stages execute in topological order
- **Graph caching** – Pre-built graphs reused across decode steps (`LayerGraphCache`)
- **MPI-aware execution** – `AllreduceStage` / `AllGatherStage` handle tensor-parallel sync
- **State management** – Owns KV cache, position tracking, and activation buffers
- **Collective context** – Routes MPI operations to appropriate backends (NCCL/RCCL/MPI/Host)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        GraphOrchestrator                                     │
│                                                                              │
│   ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────────────┐   │
│   │  InferenceState │   │  GraphExecutor  │   │   LayerGraphCache       │   │
│   │  - hidden       │   │  - execute()    │   │   - attention_decode    │   │
│   │  - logits       │   │  - topo sort    │   │   - ffn_decode          │   │
│   │  - kv_cache     │   │  - coherence    │   │   - layer_prefetch_fn   │   │
│   │  - positions[]  │   │  - verification │   │   - per-layer graphs    │   │
│   │  - activations  │   │                 │   │                         │   │
│   └─────────────────┘   └─────────────────┘   └─────────────────────────┘   │
│                                                                              │
│   ┌───────────────────────────────────────┐   ┌─────────────────────────┐   │
│   │  CollectiveContext                    │   │  PlacementPlan          │   │
│   │  - executeAllreduce()                 │   │  - layers[]             │   │
│   │  - executeAllgather()                 │   │  - decode_devices[]     │   │
│   │  - BackendRouter (NCCL/RCCL/MPI/Host) │   │  - weight_fractions[]   │   │
│   └───────────────────────────────────────┘   └─────────────────────────┘   │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  Qwen2Graph (IGraphBuilder)                                          │   │
│   │    • buildAttentionGraph(params) → ComputeGraph                      │   │
│   │    • buildFFNGraph(params) → ComputeGraph                            │   │
│   │    • buildLayerGraph(layer_idx) → ComputeGraph                       │   │
│   │    • buildForwardGraph() → complete embedding→layers→LM head         │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 InferenceState

Location: `src/v2/execution/GraphOrchestrator.h`

The **InferenceState** struct holds all buffers for inference:

```cpp
struct InferenceState {
    // Core buffers
    std::unique_ptr<FP32Tensor> hidden;           // Current hidden state
    std::unique_ptr<FP32Tensor> logits;           // Full logits (allgathered)
    std::unique_ptr<FP32Tensor> logits_local;     // Local shard (column-parallel LM head)
    
    // KV cache
    std::unique_ptr<IKVCache> kv_cache;
    
    // Position tracking
    std::vector<int> positions;                   // Per-sequence positions
    std::vector<int> sequence_lengths;            // For batched inference
    
    // Activation buffers (reused across layers)
    std::unique_ptr<FP32Tensor> normalized;       // After RMS norm
    std::unique_ptr<FP32Tensor> residual;         // For residual connections
    std::unique_ptr<FP32Tensor> Q, K, V;          // Attention projections
    std::unique_ptr<FP32Tensor> attn_output;      // Attention context
    std::unique_ptr<FP32Tensor> attn_proj;        // After Wo projection
    std::unique_ptr<FP32Tensor> gate, up;         // FFN activations
    std::unique_ptr<FP32Tensor> ffn_output;       // After down projection
    
    // HybridQ16 mode buffers (optional)
    std::unique_ptr<Q16_1Tensor> Q_rope, K_rope;  // Quantized Q/K after RoPE
    std::unique_ptr<FP32Tensor> V_dequant;        // Dequantized V for attention
    
    // Workspace buffers
    std::unique_ptr<FP32Tensor> workspace_scores; // Attention scores
    std::unique_ptr<FP32Tensor> workspace_context;// Attention context scratch
    std::unique_ptr<INT32Tensor> workspace_mask;  // Causal mask
};
```

### 3.3 ComputeGraph and ComputeNode

Location: `src/v2/execution/GraphExecutor.h`

A **ComputeGraph** is a DAG of `ComputeNode` structures:

```cpp
struct ComputeNode {
    std::string name;                              // Unique node identifier
    std::shared_ptr<IComputeStage> stage;          // The compute stage
    std::vector<std::string> dependencies;         // Input node names
    DeviceId device;                               // Target device
    bool completed = false;                        // Execution status
};

class ComputeGraph {
public:
    void addNode(const std::string& name, std::shared_ptr<IComputeStage> stage, 
                 DeviceId device = DeviceId::cpu());
    void addDependency(const std::string& dependent, const std::string& dependency);
    std::vector<const ComputeNode*> getExecutionOrder() const;  // Topological sort
    void merge(const ComputeGraph& other);                       // Combine subgraphs
    std::vector<const ComputeNode*> getRootNodes() const;        // Entry points
    std::vector<const ComputeNode*> getLeafNodes() const;        // Output nodes
    void reset();                                                 // Reset for reuse
};
```

### 3.4 IComputeStage Interface

Location: `src/v2/execution/compute_stages/IComputeStage.h`

```cpp
class IComputeStage {
public:
    virtual ~IComputeStage() = default;
    
    // Core execution
    virtual bool execute(void* ctx = nullptr) = 0;
    
    // Metadata
    virtual ComputeStageType type() const = 0;
    virtual const char* name() const = 0;
    
    // Debugging/parity (REQUIRED for all stages)
    virtual StageDumpInfo getDumpInfo() const = 0;
    
    // Buffer requirements for GraphBufferManager
    virtual StageBufferRequirements getBufferRequirements() const { return {}; }
    
    // Coherence policy for automatic GPU↔CPU sync
    virtual CoherencePolicy coherencePolicy() const { return CoherencePolicy::FULL; }
};
```

### 3.5 ComputeStageType Enum

```cpp
enum class ComputeStageType {
    // Matrix operations
    GEMM,                  // General GEMM
    GEMM_BIAS,             // GEMM with bias add
    GEMM_FUSED_QKV,        // Fused Q/K/V projection
    GEMM_FUSED_GATE_UP,    // Fused gate/up projection
    
    // Normalization
    RMS_NORM,              // RMS normalization
    LAYER_NORM,            // Layer normalization
    
    // Activations
    SWIGLU,                // SwiGLU activation
    GELU,                  // GELU activation
    SILU,                  // SiLU activation
    
    // Attention
    ROPE,                  // Rotary position embeddings
    ATTENTION,             // Full attention (deprecated)
    FUSED_ATTENTION_WO,    // Attention + Wo projection
    ATTENTION_COMPUTE,     // Pure attention computation
    
    // Element-wise
    ADD_RESIDUAL,          // Residual connections
    SCALE,                 // Scalar multiplication
    
    // MoE (future)
    MOE_ROUTER,            // MoE routing
    MOE_EXPERT_FFN,        // MoE expert computation
    MOE_COMBINE,           // MoE output combination
    
    // MPI collective
    ALLREDUCE,             // MPI_Allreduce(SUM)
    ALLGATHER,             // MPI_Allgather
    
    // Model-level
    EMBEDDING,             // Token embedding lookup
    LM_HEAD,               // Language model head
    FINAL_NORM,            // Final layer norm
    
    // KV cache
    KV_CACHE_APPEND,       // Append to KV cache
    KV_CACHE_GATHER,       // Gather from KV cache
    
    // Quantization
    QUANTIZE_Q8_1,         // Quantize to Q8_1
    QUANTIZE_Q16_1,        // Quantize to Q16_1
    DEQUANTIZE,            // Dequantize to FP32
};
```

### 3.6 Available Compute Stages

| Stage Class | Type | Purpose | MPI Sync? |
|-------------|------|---------|-----------|
| `RMSNormStage` | RMS_NORM | RMS normalization | No |
| `GEMMStage` | GEMM | Matrix multiplication | No |
| `FusedQKVGEMMStage` | GEMM_FUSED_QKV | Q/K/V projection | No |
| `FusedGateUpGEMMStage` | GEMM_FUSED_GATE_UP | Gate/Up projection | No |
| `SwiGLUStage` | SWIGLU | SwiGLU activation | No |
| `RoPEStage` | ROPE | Rotary position embeddings | No |
| `ResidualAddStage` | ADD_RESIDUAL | Residual connections | No |
| `FusedAttentionWoStage` | FUSED_ATTENTION_WO | Attention + Wo | No |
| `AttentionComputeStage` | ATTENTION_COMPUTE | Pure attention | No |
| `KVCacheAppendStage` | KV_CACHE_APPEND | KV cache management | No |
| `EmbeddingStage` | EMBEDDING | Token embedding | No |
| `LMHeadStage` | LM_HEAD | Final projection | No |
| `QuantizeQ16_1Stage` | QUANTIZE_Q16_1 | Activation quantization | No |
| `AllreduceStage` | ALLREDUCE | **MPI sync** | **Yes** |
| `AllGatherStage` | ALLGATHER | **MPI sync** | **Yes** |

### 3.7 Execution Flow

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
- `AllreduceStage` / `AllGatherStage` synchronize partial results

### 3.9 Graph Caching (Decode Optimization)

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

---

## 4. Collective Layer and MPI

### 4.1 CollectiveContext

Location: `src/v2/execution/CollectiveContext.h`

The **CollectiveContext** bridges compute stages to MPI backends:

```cpp
class CollectiveContext : public ICollectiveContext {
public:
    struct Config {
        std::shared_ptr<MPIContext> mpi_ctx;
        std::shared_ptr<IMPITopology> topology;
        CollectiveBackendType preferred_backend = CollectiveBackendType::AUTO;
    };
    
    // Execute collective operations
    bool executeAllreduce(TensorBase* buffer, CollectiveOp op, DeviceId device);
    bool executeAllgather(TensorBase* local, TensorBase* full, int seq_len, DeviceId device);
    bool executeBroadcast(TensorBase* buffer, int root, DeviceId device);
    
    // Backend selection
    ICollectiveBackend* getBackend(const DeviceGroup& group);
};
```

### 4.2 BackendRouter

Location: `src/v2/collective/BackendRouter.h`

Routes collective operations to appropriate backends:

```cpp
class BackendRouter : public IBackendRouter {
public:
    BackendSelection selectBackend(const DeviceGroup& group);
    ICollectiveBackend* getBackend(const DeviceGroup& group);
    
    // Special case: heterogeneous device groups (CPU+GPU)
    bool executeHeterogeneousAllReduce(
        const std::vector<DeviceId>& devices,
        TensorBase* buffer,
        CollectiveOp op);
};

struct BackendSelection {
    CollectiveBackendType type;
    ICollectiveBackend* backend;
    std::string reason;  // For debugging
};
```

### 4.3 CollectiveBackendType

```cpp
enum class CollectiveBackendType {
    AUTO,       // Auto-select based on DeviceGroup topology
    MPI,        // MPI (fallback, inter-node)
    NCCL,       // NVIDIA Collective Communication Library
    RCCL,       // ROCm Collective Communication Library
    PCIE_BAR,   // Direct PCIe BAR mapping (CUDA↔ROCm)
    HOST,       // Host-staged fallback (heterogeneous)
};
```

### 4.4 Backend Selection Logic

| Device Group | Selected Backend | Reason |
|--------------|------------------|--------|
| All CUDA | NCCL | NVLink/PCIe fast path |
| All ROCm | RCCL | xGMI/PCIe fast path |
| Mixed CUDA+ROCm | PCIE_BAR or HOST | Direct BAR or staged |
| Cross-node | MPI | Infiniband/Ethernet |
| CPU only | MPI | Standard MPI |

---

## 5. Buffer Management

### 5.1 GraphBufferManager

Location: `src/v2/execution/GraphBufferManager.h`

Manages memory allocation with buffer aliasing for reduced footprint:

```cpp
class GraphBufferManager : public IGraphBufferManager {
public:
    struct MemoryInfo {
        size_t total_bytes;
        size_t free_bytes;
        size_t usable_bytes;  // Accounting for fragmentation
    };
    
    MemoryInfo queryAvailableMemory(DeviceId device);
    size_t computeWorkspaceBudget(DeviceId device, const WorkspaceBudgetConfig& config);
    void allocateWithAliasing(const ComputeGraph& graph);
    TensorBase* getBuffer(const std::string& node, const std::string& buffer);
};
```

### 5.2 LivenessAnalyzer

Location: `src/v2/execution/LivenessAnalyzer.h`

Analyzes buffer lifetimes to enable SCRATCH buffer aliasing:

```cpp
class LivenessAnalyzer {
public:
    struct BufferLiveness {
        std::string buffer_name;
        int first_use_stage;   // Stage index where buffer is first written
        int last_use_stage;    // Stage index where buffer is last read
        size_t size_bytes;
    };
    
    std::vector<BufferLiveness> analyze(const ComputeGraph& graph);
    std::vector<AliasingGroup> computeAliasingGroups();
    std::pair<size_t, size_t> computeMemoryUsage();  // (original, optimized)
};
```

**Example Aliasing:**
```
Stage 0: Q = GEMM(hidden)           Q live [0, 2]
Stage 1: K = GEMM(hidden)           K live [1, 2]
Stage 2: attn = Attention(Q, K, V)  Q, K dead after stage 2
Stage 3: gate = GEMM(hidden)        gate live [3, 4] ← can reuse Q's memory!
```

### 5.3 DeviceWorkspaceManager

Location: `src/v2/execution/DeviceWorkspaceManager.h`

Per-device workspace allocation with zero-allocation hot path:

```cpp
class DeviceWorkspaceManager {
public:
    DeviceWorkspaceManager(DeviceId device, size_t total_bytes);
    
    // Single contiguous allocation at startup
    void* allocateContiguous(size_t bytes, size_t alignment = 64);
    
    // Named sub-buffer allocation
    void* getBuffer(const std::string& name);
    size_t getBufferSize(const std::string& name);
    
    // Zero-allocation during inference - all buffers pre-allocated
    bool isFullyAllocated() const;
};
```

### 5.4 WorkspaceBudgetConfig

```cpp
struct WorkspaceBudgetConfig {
    float gpu_workspace_fraction = 0.1f;  // 10% of free GPU memory
    float cpu_workspace_fraction = 0.2f;  // 20% of available RAM
    size_t min_workspace_bytes = 64 * 1024 * 1024;  // 64 MB minimum
    size_t max_workspace_bytes = 2ULL * 1024 * 1024 * 1024;  // 2 GB max
};
```

---

## 6. Tensors and Tensor Interfaces

### 6.1 Core Tensor Types

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

### 6.1.1 TypedTensorBase and `typed_data()` Pattern

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

### 6.2 Tensor Kernel Interfaces

Location: `src/v2/tensors/TensorKernels.h`

- `ITensorGemm` – GEMM between activations and/or weights
- `ITensorAttention` – Attention over Q/K/V
- Other per-op interfaces (RMSNorm, SwiGLU, RoPE)

### 4.3 TensorLayout Contracts

Location: `src/v2/tensors/TensorLayout.h`

Tensor memory layout contracts define canonical shapes for attention tensors. Layouts are orthogonal to quantization format—a Q16_1Tensor can have any layout depending on pipeline position.

**Available Layouts:**

| Layout | Shape | Use Case |
|--------|-------|----------|
| `Q_SEQ_HEAD_DIM` | `[seq_len][n_heads][head_dim]` | Query after GEMM |
| `Q_HEAD_SEQ_DIM` | `[n_heads][seq_len][head_dim]` | Per-head parallel attention |
| `KV_POS_HEAD_DIM` | `[position][n_kv_heads][head_dim]` | KV cache POSITION_MAJOR |
| `KV_HEAD_POS_DIM` | `[n_kv_heads][position][head_dim]` | KV cache HEAD_MAJOR |
| `ROW_MAJOR_2D` | `[rows][cols]` | Embeddings, hidden states |
| `ROW_MAJOR_1D` | `[elements]` | Flat vectors |

**Layout Validation:**

```cpp
// LayoutExpectation contains model dimensions for validation
LayoutExpectation expect{
    .head_dim = 128,
    .n_heads = 32,
    .n_kv_heads = 8,
    .d_model = 4096
};

// Validate tensor shape matches expected layout
auto result = validateTensorLayout(tensor, TensorLayout::Q_SEQ_HEAD_DIM, expect);
if (!result.passed) {
    throw LayoutValidationError(result.error_reason);
}
```

### 4.4 TensorVerification System

Location: `src/v2/tensors/TensorVerification.h`

The verification system provides fail-fast validation at stage boundaries in Debug/Integration builds:

**Build Type Behavior:**

| Build Type | `LLAMINAR_ASSERTIONS_ACTIVE` | Verification Active | Behavior |
|------------|------------------------------|---------------------|----------|
| Debug | 1 | Yes | Throws `VerificationFailure` |
| Integration | 1 | Yes | Throws `VerificationFailure` |
| Release | 0 | No | Compiled out (zero overhead) |

**Verification Checks:**

- **NaN/Inf detection**: Sampled check for floating-point anomalies
- **Null pointer detection**: Catches uninitialized buffers
- **All-zero detection**: Warns on suspiciously empty tensors
- **Layout validation**: Verifies shape matches expected layout

**GraphExecutor Integration:**

```cpp
// Automatically called at stage boundaries
void GraphExecutor::execute(ComputeGraph& graph, ExecutionContext& ctx) {
    for (auto& stage : topological_order) {
        // BEFORE stage execution
        verifyStageEntry(stage, layer_idx);
        
        // Execute stage
        stage->execute();
        
        // AFTER stage execution (before snapshot callback)
        verifyStageExit(stage, layer_idx);
        
        // Snapshot callback (if enabled)
        if (snapshot_callback_) {
            invokeSnapshotCallback(stage);
        }
    }
}
```

**VerificationFailure Exception:**

When verification fails, a `VerificationFailure` exception is thrown with full context:

```
╔══════════════════════════════════════════════════════════════════╗
║               TENSOR VERIFICATION FAILED                          ║
╠══════════════════════════════════════════════════════════════════╣
║ Layer:  3
║ Stage:  FusedAttentionWoStage
║ Phase:  EXIT
║ Tensor: attention_output
║ Reason: Contains 5 NaN values in first 8 rows
║
║ Dump:   /tmp/llaminar_verification_dump/...
╚══════════════════════════════════════════════════════════════════╝
```

**Environment Variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `LLAMINAR_VALIDATE_BUFFERS` | Auto (Debug/Integration) | Enable/disable buffer validation |
| `LLAMINAR_FAIL_ON_NAN` | 1 (Debug/Integration) | Throw on NaN/Inf detection |
| `LLAMINAR_FAIL_ON_ZERO` | 0 | Throw on all-zero tensors |
| `LLAMINAR_DUMP_ON_FAILURE` | 1 | Dump buffers to disk on failure |

### 4.5 UnifiedKVCache and Layout Modes

Location: `src/v2/tensors/UnifiedKVCache.h`

The `UnifiedKVCache` manages KV cache storage with configurable memory layouts:

**KVCacheLayoutMode:**

```cpp
enum class KVCacheLayoutMode : uint8_t {
    POSITION_MAJOR,  // [position][n_kv_heads][head_dim] - cache-append friendly
    HEAD_MAJOR       // [n_kv_heads][position][head_dim] - attention-compute friendly
};
```

**Layout Selection:**

| Layout | Memory Pattern | Best For |
|--------|----------------|----------|
| `POSITION_MAJOR` | Append is contiguous | FP32/BF16 attention (standard) |
| `HEAD_MAJOR` | Per-head iteration is contiguous | Q16_INTEGER attention |

**GraphOrchestrator Auto-Selection:**

The `GraphOrchestrator` automatically selects the optimal layout based on KV cache precision:

```cpp
// In GraphOrchestrator::createKVCache()
KVCacheLayoutMode layout_mode = (kv_precision == ActivationPrecision::Q16_1)
    ? KVCacheLayoutMode::HEAD_MAJOR   // Optimal for Q16IntegerAttention
    : KVCacheLayoutMode::POSITION_MAJOR;  // Default for FP32/BF16
```

**HEAD_MAJOR Benefits for Q16_INTEGER:**

- Per-head K/V access via `get_k_for_kv_head(kv_head_idx)` returns contiguous buffer
- Eliminates transpose for Q16 integer dot product kernel
- Natural fit for `[n_kv_heads][position][head_dim]` attention iteration

**Multi-Block Q16_1 Support:**

For Q16_1 precision, the cache stores multi-block data per head when `head_dim > block_size`:

```
head_dim=256, BLOCK_128 → 2 blocks per head per position
head_dim=384, BLOCK_128 → 3 blocks per head per position
```

**IUnifiedKVCache Interface:**

```cpp
class IUnifiedKVCache {
    virtual void append(TensorBase* K, TensorBase* V, int seq_len) = 0;
    virtual void shift(int positions_to_evict) = 0;
    virtual int cached_tokens() const = 0;
    virtual KVCacheLayoutMode layout_mode() const = 0;
    
    // HEAD_MAJOR accessors (returns nullptr for POSITION_MAJOR)
    virtual TensorBase* get_k_for_kv_head(int kv_head_idx) = 0;
    virtual TensorBase* get_v_for_kv_head(int kv_head_idx) = 0;
    
    // POSITION_MAJOR accessors
    virtual TensorBase* get_k_cache() = 0;
    virtual TensorBase* get_v_cache() = 0;
};
```

### 4.6 Device Coherence Protocol

Location: `src/v2/tensors/cpu/CPUTensors.h`, `src/v2/execution/StageCoherence.h`, `src/v2/execution/GpuCoherence.h`

Llaminar uses a **dual-validity coherence protocol** to manage tensor data movement between host (CPU) and device (GPU) memory. Each `CPUTensorBase` tracks two independent validity flags: `host_valid_` and `device_valid_`.

#### Coherence State Machine

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      TENSOR COHERENCE STATE MACHINE                              │
│                                                                                  │
│  Two independent validity flags: host_valid_ and device_valid_                   │
│  A tensor can be valid on both, either, or neither location.                    │
└─────────────────────────────────────────────────────────────────────────────────┘

    ┌────────────────────┐                    ┌────────────────────┐
    │   INITIAL STATE    │                    │   MAPPED MEMORY    │
    │  host_valid = true │                    │  (Zero-Copy Mode)  │
    │  device_valid = false                   │                    │
    │  gpu_data_ptr = nullptr                 │  host_valid = true │
    └─────────┬──────────┘                    │  device_valid = true
              │                               │  (always both)     │
              │ ensureOnDevice()              └────────────────────┘
              │ [pins host memory]
              │ [allocates GPU buffer]
              │ [uploads data]
              ▼
    ┌────────────────────┐
    │   BOTH VALID       │
    │  host_valid = true │  ←─────────────────────────────────────┐
    │  device_valid = true                                        │
    │  gpu_data_ptr = 0x...                                       │
    └─────────┬──────────┘                                        │
              │                                                   │
              │ mark_device_dirty()                               │
              │ [after GPU kernel writes]                         │
              ▼                                                   │
    ┌────────────────────┐                                        │
    │   GPU AUTHORITATIVE│                                        │
    │  host_valid = false│                                        │
    │  device_valid = true                                        │
    └─────────┬──────────┘                                        │
              │                                                   │
              │ data() or ensureOnHost()                          │
              │ [downloads from GPU]                              │
              │ [waits on completion event if set]                │
              └───────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────────────────┐
    │  DESTRUCTOR (~CPUTensorBase):                                                │
    │    1. freeMappedMemory() - if using zero-copy                               │
    │    2. Free GPU buffer via IBackend::free() - if gpu_data_ptr_ != nullptr    │
    │    3. Destroy completion event - if device_completion_event_ exists          │
    │    4. unpinHostMemory() - if host_pinned_ == true                           │
    │    5. Clear kernel cache entry                                               │
    └─────────────────────────────────────────────────────────────────────────────┘
```

#### Internal State Variables

```cpp
class CPUTensorBase {
protected:
    // ===== Coherence State =====
    bool host_valid_ = true;          // Host buffer contains valid data
    bool device_valid_ = false;       // GPU buffer contains valid data
    void* gpu_data_ptr_ = nullptr;    // GPU buffer pointer (nullptr if not allocated)
    std::optional<DeviceId> gpu_device_;  // Which GPU owns the buffer
    
    // ===== Pinned Memory (for fast GPU transfers) =====
    bool host_pinned_ = false;        // True if host buffer is page-locked
    size_t pinned_bytes_ = 0;         // Size of pinned region
    void* pinned_host_ptr_ = nullptr; // Pointer used when pinning (for destructor)
    
    // ===== Mapped Memory (zero-copy) =====
    bool is_mapped_ = false;          // True if using shared host/device memory
    void* mapped_ptr_ = nullptr;      // Mapped memory pointer
    
    // ===== Fine-Grained Sync =====
    void* device_completion_event_ = nullptr;  // CUDA/HIP event for async sync
};
```

#### Core CPUTensorBase Methods

```cpp
class CPUTensorBase {
public:
    // ===== Device Coherence =====
    
    // Upload tensor to GPU (lazy-allocates GPU buffer, pins host memory)
    // First call: pins host → allocates GPU buffer → uploads
    // Subsequent calls: uploads if host_valid_ && !device_valid_
    bool ensureOnDevice(DeviceId device);
    
    // Download from GPU to host (if device_valid_ && !host_valid_)
    // Waits on completion event if set (fine-grained sync)
    bool ensureOnHost();
    
    // Mark GPU as having authoritative data
    // Sets device_valid_ = true, host_valid_ = false (unless mapped)
    void mark_device_dirty();
    
    // Mark GPU dirty AND record a completion event for fine-grained sync
    void mark_device_dirty_with_event();
    
    // ===== Data Access (Coherence-Aware) =====
    
    // Returns host pointer - calls ensureOnHost() if device_valid_ && !host_valid_
    const float* data();
    
    // Returns mutable host pointer - also sets host_valid_ = true
    float* mutable_data();
    
    // ===== Query Methods =====
    
    void* gpu_data_ptr();              // Raw GPU buffer pointer (nullptr if not on GPU)
    bool isOnGPU() const;              // True if gpu_data_ptr_ != nullptr
    bool isOnCPU() const;              // True if host_valid_
    bool isDeviceValid() const;        // True if device_valid_ && gpu_data_ptr_
    bool isMapped() const;             // True if using zero-copy memory
    
    // ===== Lifecycle =====
    
    bool releaseDeviceMemory();        // Download if needed, then free GPU buffer
    void invalidateGpuData();          // Free GPU buffer without downloading
};
```

#### Automatic Coherence via GraphExecutor

Location: `src/v2/execution/GraphExecutor.cpp`, `src/v2/execution/StageCoherence.h`

The `GraphExecutor` handles coherence **automatically** at stage boundaries using the stage's `coherencePolicy()`:

```cpp
// GraphExecutor stage execution flow (simplified)
bool GraphExecutor::executeStage(const ComputeNode& node) {
    CoherencePolicy policy = node.stage->coherencePolicy();
    auto dump_info = node.stage->getDumpInfo();
    
    // 1. ENTRY: Cohere inputs to GPU
    if (policy == CoherencePolicy::INPUT || policy == CoherencePolicy::FULL) {
        auto inputs = extractInputBuffers(dump_info);
        cohereInputs(inputs, target_device);      // ensureOnDevice() on each
        
        auto weights = extractWeightBuffers(dump_info);
        cohereInputs(weights, target_device);     // Weights are read-only inputs
    }
    
    // 2. ENTRY: Allocate GPU buffers for outputs (CRITICAL for GPU kernels)
    if (policy == CoherencePolicy::OUTPUT || policy == CoherencePolicy::FULL) {
        auto outputs = extractOutputBuffers(dump_info);
        cohereOutputs(outputs, target_device);    // ensureOnDevice() to allocate
    }
    
    // 3. EXECUTE: Run the kernel
    bool success = node.stage->execute();
    
    // 4. EXIT: Mark outputs as GPU-authoritative
    if (policy == CoherencePolicy::OUTPUT || policy == CoherencePolicy::FULL) {
        auto outputs = extractOutputBuffers(dump_info);
        markOutputsDirty(outputs);                // mark_device_dirty() on each
    }
    
    return success;
}
```

**Key Insight: `cohereOutputs()` vs `cohereInputs()`**

Both call `ensureOnDevice()`, but serve different purposes:
- **`cohereInputs()`**: Upload data that the kernel will READ
- **`cohereOutputs()`**: Allocate GPU buffer that the kernel will WRITE TO

For GPU execution, outputs MUST have their GPU buffers allocated BEFORE the kernel runs, otherwise the kernel has nowhere to write results. This is why `CoherencePolicy::FULL` is the default for most stages - it ensures both inputs are uploaded AND outputs are allocated.

**CoherencePolicy Enum:**

| Policy | Entry: Inputs | Entry: Outputs | Exit |
|--------|---------------|----------------|------|
| `FULL` (default) | Upload inputs + weights | Allocate output buffers | Mark outputs dirty |
| `INPUT` | Upload inputs + weights | No allocation | No-op |
| `OUTPUT` | No upload | Allocate output buffers | Mark outputs dirty |
| `NONE` | No-op | No-op | No-op |

**When to Use Each Policy:**

```cpp
class GEMMStage : public IComputeStage {
    // FULL (default): Most compute stages
    CoherencePolicy coherencePolicy() const override { return CoherencePolicy::FULL; }
};

class LMHeadStage : public IComputeStage {
    // FULL: Needs GPU buffer for logits output, reads hidden state input
    CoherencePolicy coherencePolicy() const override { return CoherencePolicy::FULL; }
};

class AllreduceStage : public IComputeStage {
    // NONE: MPI collective manages its own synchronization
    CoherencePolicy coherencePolicy() const override { return CoherencePolicy::NONE; }
};

class ResidualAddStage : public IComputeStage {
    // FULL: In-place operation still needs input on GPU, output marked dirty
    CoherencePolicy coherencePolicy() const override { return CoherencePolicy::FULL; }
};
```

#### Pinned Host Memory

For optimal GPU transfer performance, `CPUTensorBase` lazily pins host memory on first GPU upload:

```cpp
bool CPUTensorBase::ensureOnDevice(DeviceId device) {
    // 1. Pin host memory (first time only) - enables DMA transfers
    if (!host_pinned_) {
        ensureHostPinned();  // cudaHostRegister / hipHostRegister
    }
    
    // 2. Allocate GPU buffer (first time only)
    if (!gpu_data_ptr_) {
        gpu_data_ptr_ = backend->allocate(byte_size(), device.gpu_ordinal());
    }
    
    // 3. Upload if host is newer than device
    if (host_valid_ && !device_valid_) {
        backend->hostToDevice(gpu_data_ptr_, raw_host_data_ptr(), byte_size(), ...);
        device_valid_ = true;
    }
    
    return true;
}
```

**Pinned Memory Lifecycle:**
- Pinned lazily on first `ensureOnDevice()` call
- Unpinned in destructor (`unpinHostMemory()`)
- Stored in `pinned_host_ptr_` for proper cleanup even if tensor is reallocated

#### Mapped Memory (Zero-Copy)

For tensors where GPU reads/writes directly to host memory via PCIe:

```cpp
// Allocate mapped memory (shared host/device)
bool initMappedMemory(size_t bytes, DeviceId target_device);

// When mapped:
// - ensureOnDevice() and ensureOnHost() are no-ops
// - Both host_valid_ and device_valid_ are always true
// - mark_device_dirty() keeps both valid (shared memory)
// - GPU reads/writes directly via PCIe BAR
```

#### Manual Coherence Utilities

Location: `src/v2/execution/GpuCoherence.h`

For tests and custom pipelines that bypass GraphExecutor, use RAII utilities:

```cpp
#include "execution/GpuCoherence.h"

// Pattern 1: Lambda wrapper (preferred - handles both inputs and outputs)
bool ok = with_gpu_coherence(
    gpu_device,
    {input.get()},                           // inputs to upload
    {output_q.get(), output_k.get()},        // outputs to allocate + mark dirty
    [&] {
        return kernel->multiply_fused(input.get(), projections, M, K);
    });

// Pattern 2: RAII wrapper for single output
{
    auto output = GpuOutput<FP32Tensor>(output_tensor.get(), gpu_device);
    kernel->compute(input.get(), output.get(), ...);
}  // ← output marked dirty on scope exit

// Pattern 3: RAII for read-only inputs
{
    auto weights = GpuInput<Q4_0Tensor>(weight_tensor.get(), gpu_device);
    kernel->compute(weights.get(), ...);
}  // ← weights NOT marked dirty (read-only)
```

**Anti-Pattern (WRONG):**

```cpp
// ❌ BAD: Forgetting to allocate output GPU buffer
input->ensureOnDevice(gpu);
// output->ensureOnDevice(gpu);  // MISSING! Kernel has nowhere to write!
kernel->compute(input->gpu_data_ptr(), output->gpu_data_ptr(), ...);  // CRASH: output->gpu_data_ptr() == nullptr

// ❌ BAD: Forgetting to mark output dirty
input->ensureOnDevice(gpu);
output->ensureOnDevice(gpu);
kernel->compute(...);
// output->mark_device_dirty();  // MISSING! Host thinks its data is still valid
const float* result = output->data();  // Returns STALE host data!

// ✅ CORRECT: Use with_gpu_coherence() or GpuOutput RAII
```

**C++20 Concept:**

```cpp
template <typename T>
concept CoherableTensor = requires(T *t, DeviceId d) {
    { t->ensureOnDevice(d) } -> std::same_as<bool>;
    { t->mark_device_dirty() } -> std::same_as<void>;
};
```

#### Fine-Grained Synchronization

For optimal performance, use event-based synchronization instead of full device sync:

```cpp
// After GPU kernel writes to tensor:
tensor->mark_device_dirty_with_event();  // Records completion event

// Later, when reading on host:
tensor->data();  // Waits on just this tensor's event, not full device sync
```

This is especially useful when multiple kernels run concurrently - each tensor tracks its own completion status.

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

### 7.3 HybridQ16 Mode and Q16IntegerAttention

Location: `src/v2/kernels/cpu/attention/q16/ref/Q16IntegerAttentionRef.h`

**HybridQ16 Mode** is an experimental attention path that uses Q16_1 quantized activations with integer-only attention computation. This mode is designed for memory-efficient inference where:

- Activations (Q, K, V) are stored as Q16_1 (16-bit quantized blocks)
- KV cache uses Q16_1 precision with HEAD_MAJOR layout
- Attention computation uses integer dot products avoiding FP32 dequantization

**Activation Modes:**

| Mode | Q/K/V Precision | KV Cache Layout | Attention Kernel |
|------|-----------------|-----------------|------------------|
| Standard (FP32) | FP32 | POSITION_MAJOR | `CpuAttentionKernelT<float>` |
| BF16 | BF16 | POSITION_MAJOR | `CpuAttentionKernelT<bfloat16>` |
| HybridQ16 | Q16_1 | HEAD_MAJOR | `Q16IntegerAttentionRef` |

**Q16IntegerAttention Algorithm:**

```cpp
// Integer-only attention (no FP32 intermediate)
// Q: [seq_q][n_heads][head_dim] as Q16_1
// K: [n_kv_heads][kv_len][head_dim] as Q16_1 (HEAD_MAJOR)
// V: [n_kv_heads][kv_len][head_dim] as Q16_1 (HEAD_MAJOR)

for each query head h:
    kv_head = h / (n_heads / n_kv_heads)  // GQA mapping
    
    // Integer dot product: Q[h] · K[kv_head]^T
    scores = q16_dot_product(Q[h], K[kv_head])  // INT32 accumulator
    
    // Fixed-point softmax (avoids FP32)
    weights = q16_softmax(scores)  // INT16 weights summing to 32767
    
    // Weighted sum: weights · V[kv_head]
    context[h] = q16_weighted_sum(weights, V[kv_head])
```

**Status:** HybridQ16 is under active development. The pipeline is wired but numerical parity tests are pending (see `PROJECT_Q16_INTEGER_ATTENTION_V2.md`).

---

## 8. Stage Debugging and Snapshot System

### 8.1 StageDumpInfo Infrastructure

Location: `src/v2/execution/compute_stages/IComputeStage.h`

The **StageDumpInfo** system is the foundation for all stage debugging, introspection, and snapshot capture. Every `IComputeStage` must implement `getDumpInfo()` to expose its inputs, outputs, weights, and scalar parameters:

```cpp
struct StageDumpInfo {
    struct InputBuffer {
        const char* name;
        const void* data;
        size_t rows, cols;
        const char* dtype;  // "FP32", "Q8_1", "Q16_1", etc.
    };
    
    struct OutputBuffer { /* same fields */ };
    struct WeightBuffer { /* includes TensorBase* for full metadata */ };
    struct ScalarParam { const char* name; double value; const char* dtype; };
    
    std::vector<InputBuffer> inputs;
    std::vector<OutputBuffer> outputs;
    std::vector<WeightBuffer> weights;
    std::vector<ScalarParam> scalars;
    
    // Fluent builder API
    StageDumpInfo& addInput(const char* name, const float* data, size_t rows, size_t cols);
    StageDumpInfo& addOutput(const char* name, const TensorBase* tensor, size_t rows, size_t cols);
    StageDumpInfo& addWeight(const char* name, const TensorBase* tensor);
    StageDumpInfo& addScalarInt(const char* name, int value);
};
```

**Every stage must implement getDumpInfo()** - This is required for:
1. **TensorVerification** - Entry/exit validation of inputs and outputs
2. **Snapshot callbacks** - parity testing against PyTorch
3. **StageDumper** - Full binary dumps for debugging

### 8.2 StageDumper Utility

Location: `src/v2/execution/StageDumper.h`

The `StageDumper` provides comprehensive disk-based debugging of stage execution:

**Output Structure:**
```
<dump_dir>/stage_<counter>_<type>_layer<N>_iter<I>/
  metadata.txt        - Human-readable description
  params.bin          - Binary parameters for replay
  inputs/             - Input tensors directory
    A.bin             - FP32 activation matrix
    A_q8_1.bin        - Q8_1 block data
  weights/            - Weight tensors directory
    B_dequant.bin     - Dequantized weights (FP32)
    B_raw.bin         - Raw quantized blocks
    B_metadata.txt    - Quant type, shape, scale info
  outputs/            - Output tensors directory
    C.bin             - Output matrix
```

**Environment Variables:**

| Variable | Effect |
|----------|--------|
| `LLAMINAR_STAGE_DUMP=1` | Enable stage dumping |
| `LLAMINAR_STAGE_DUMP_DIR` | Output directory (default: `/tmp/llaminar_stage_dumps`) |
| `LLAMINAR_STAGE_DUMP_TYPES` | Comma-separated stage types to dump (e.g., `GEMM,ATTENTION`) |
| `LLAMINAR_STAGE_DUMP_LAYERS` | Comma-separated layer indices to dump |

**Usage (automatic via GraphExecutor):**
```bash
LLAMINAR_STAGE_DUMP=1 LLAMINAR_STAGE_DUMP_LAYERS=0,1 ./build_v2/llaminar2 -m model.gguf -p "test"
```

### 8.3 Snapshot System (Subset of StageDumpInfo)

> **📖 For parity test implementation details, see `tests/v2/integration/parity/README.md`**

The **snapshot system** uses `StageDumpInfo` outputs for parity testing. It captures output tensors in-memory for comparison against PyTorch reference:

**Architecture:**
```
StageDumpInfo.outputs  ->  GraphExecutor.snapshot_callback_  ->  In-memory storage
                                     |
                          IInferenceRunner.getSnapshot()  ->  parity test comparison
```

- **Compile-time conditional** - Only in Debug/Integration builds (`ENABLE_PIPELINE_SNAPSHOTS`)
- **In-memory storage** - `std::map<std::string, std::vector<float>>`
- **Zero overhead in Release** - Compiles away to NOOPs

**GraphExecutor Integration:**
```cpp
// After stage execution, if snapshot callback is configured
if (success && config_.snapshot_callback) {
    auto dump_info = node.stage->getDumpInfo();  // Reuse StageDumpInfo
    config_.snapshot_callback(node.name, dump_info);
}
```

**API Usage:**
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
    // Compare with PyTorch reference...
}
```

### 8.4 Snapshot Keys

- **Global:** `EMBEDDING`, `FINAL_NORM`, `LM_HEAD`
- **Per-layer:** `layer{i}_{STAGE}` where stage is:
  - Attention: `ATTENTION_NORM`, `Q_PROJECTION`, `K_PROJECTION`, `V_PROJECTION`, `Q_ROPE`, `K_ROPE`, `ATTENTION_CONTEXT`, `ATTENTION_OUTPUT`, `ATTENTION_RESIDUAL`
  - FFN: `FFN_NORM`, `FFN_GATE`, `FFN_UP`, `FFN_SWIGLU`, `FFN_DOWN`, `FFN_RESIDUAL`

### 8.5 Integration with TensorVerification

The TensorVerification system (Section 4.4) also uses `StageDumpInfo`:

```cpp
void GraphExecutor::verifyStageExit(const ComputeNode& node, int layer_idx) {
    auto dump_info = node.stage->getDumpInfo();  // Get all outputs
    
    for (const auto& output : dump_info.outputs) {
        auto result = verifyRawBuffer(output.data, output.rows, output.cols, ...);
        if (!result.passed) {
            // Dump ALL buffers for debugging (inputs + outputs + weights)
            dumpStageBuffers(node.name, layer_idx, "EXIT", dump_info, ...);
            throw VerificationFailure(...);
        }
    }
}
```

**Key Insight:** `StageDumpInfo` is the single source of truth for stage introspection. Snapshots, verification, and debugging all consume the same data structure, ensuring consistency.

---

## 9. Quick Reference

### 9.1 File Locations

| Component | Location |
|-----------|----------|
| **Application Layer** | |
| Inference interface | `src/v2/execution/IInferenceRunner.h` |
| Inference factory | `src/v2/execution/InferenceRunnerFactory.h` |
| Model context interface | `src/v2/interfaces/IModelContext.h` |
| **Orchestration Layer** | |
| GraphOrchestrator | `src/v2/execution/GraphOrchestrator.h` |
| Graph builder interface | `src/v2/execution/IGraphBuilder.h` |
| Qwen2 graph builder | `src/v2/models/qwen/Qwen2Graph.h` |
| **Graph Layer** | |
| Graph executor | `src/v2/execution/GraphExecutor.h` |
| Compute stages | `src/v2/execution/compute_stages/` |
| Stage base interface | `src/v2/execution/compute_stages/IComputeStage.h` |
| **Buffer Management** | |
| Graph buffer manager | `src/v2/execution/GraphBufferManager.h` |
| Liveness analyzer | `src/v2/execution/LivenessAnalyzer.h` |
| Workspace manager | `src/v2/execution/DeviceWorkspaceManager.h` |
| **Collective Layer** | |
| Collective context | `src/v2/execution/CollectiveContext.h` |
| Backend router | `src/v2/collective/BackendRouter.h` |
| Backend interface | `src/v2/collective/ICollectiveBackend.h` |
| NCCL/RCCL/MPI backends | `src/v2/collective/backends/` |
| **Coherence** | |
| Stage coherence | `src/v2/execution/StageCoherence.h` |
| GPU coherence utilities | `src/v2/execution/GpuCoherence.h` |
| **Kernel Layer** | |
| KernelFactory | `src/v2/kernels/KernelFactory.h` |
| CPU kernels | `src/v2/kernels/cpu/` |
| CUDA kernels | `src/v2/kernels/cuda/` |
| ROCm kernels | `src/v2/kernels/rocm/` |
| **Backend Layer** | |
| Backend manager | `src/v2/backends/BackendManager.h` |
| Backend interface | `src/v2/backends/IBackend.h` |
| DeviceId | `src/v2/backends/DeviceId.h` |
| **Tensor Layer** | |
| ITensor interface | `src/v2/tensors/ITensor.h` |
| CPU tensors | `src/v2/tensors/cpu/CPUTensors.h` |
| Tensor factory | `src/v2/tensors/TensorFactory.h` |
| **MPI Layer** | |
| MPI context | `src/v2/utils/MPIContext.h` |
| MPI topology | `src/v2/utils/MPITopology.h` |
| NUMA topology | `src/v2/utils/NUMATopology.h` |
| Placement strategy | `src/v2/execution/PlacementStrategy.h` |
| Weight sharding | `src/v2/loaders/WeightManager.h` |

### 9.2 Key Design Rules

1. **Use `IInferenceRunner`** for inference – Don't call `GraphOrchestrator` directly from Main/ChatUI
2. **Keep MPI out of kernels** – MPI sync lives in `AllreduceStage` and `AllGatherStage`
3. **Use KernelFactory** – Prefer `getOrCreateGemm()` for cached kernel access
4. **Declarative graphs** – Build computation as DAGs of `ComputeStage` nodes
5. **Graph caching** – Reuse cached graphs in decode mode for performance
6. **Use StageCoherence** – Let GraphExecutor handle GPU↔CPU sync automatically
7. **Implement getDumpInfo()** – All stages must support introspection
8. **DeviceId not int** – Use typed `DeviceId` for device identification
9. **Collective via BackendRouter** – Let the router select optimal MPI backend

### 9.3 Environment Variables

| Variable | Effect |
|----------|--------|
| `LLAMINAR_LOG_LEVEL=DEBUG` | Set logging verbosity |
| `LLAMINAR_PROFILE_KERNELS=1` | Enable per-kernel timing in benchmark mode |
| `LLAMINAR_EXECUTOR_PROFILING=1` | Enable per-stage profiling in GraphExecutor |
| `LLAMINAR_VALIDATE_BUFFERS=1` | Enable buffer validation at stage boundaries |
| `LLAMINAR_VALIDATE_INPUTS=1` | Enable input validation before stage execution |
| `LLAMINAR_FAIL_ON_NAN=1` | Throw exception on NaN/Inf detection (default in Debug) |
| `LLAMINAR_FAIL_ON_ZERO=1` | Throw exception on all-zero output tensors |
| `LLAMINAR_DUMP_ON_FAILURE=1` | Dump buffers to disk when verification fails |
| `LLAMINAR_STAGE_DUMP_ENABLED=1` | Enable full stage dumping |
| `LLAMINAR_STAGE_DUMP_DIR` | Output directory for stage dumps |
| `LLAMINAR_STAGE_DUMP_TYPES` | Comma-separated stage types to dump |
| `LLAMINAR_STAGE_DUMP_LAYERS` | Comma-separated layer indices to dump |
| `LLAMINAR_STAGE_OUTPUT_PRINT=1` | Print stage outputs to log |
| `LLAMINAR_MPI_LOG_COLLECTIVES=1` | Log MPI collective operations |

### 9.4 Running Tests

```bash
# Unit tests (Debug build)
ctest --test-dir build_v2 -R "^V2_Unit_" --output-on-failure --parallel

# Integration tests (Integration build - has snapshots + debug symbols)
ctest --test-dir build_v2_integration -R "^V2_Integration_" --output-on-failure

# Parity tests (Llaminar vs PyTorch reference)
ctest --test-dir build_v2_integration -R "^V2_Integration_Parity_" --output-on-failure

# Performance benchmarks (Release build)
ctest --test-dir build_v2_release -R "^V2_Perf_" --verbose
```

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
class MPITopology : public IMPITopology {
    // Identification
    int rank() const;
    int world_size() const;
    int node_rank() const;
    int local_rank() const;
    int ranks_per_node() const;
    const std::vector<std::string>& all_hostnames() const;
    
    // Work range calculations
    WorkRange get_head_range(int total_heads) const;     // For attention
    WorkRange get_column_range(int total_cols) const;    // For column-parallel
    WorkRange get_vocab_range(int vocab_size) const;     // For LM head
    
    // Device capabilities
    const std::vector<RankPlacement>& all_placements() const;
    const ClusterInventory& cluster_inventory() const;
};

struct WorkRange {
    int start;
    int end;        // Exclusive
    int count() const { return end - start; }
};

struct RankPlacement {
    int rank;
    std::string hostname;
    int numa_node;
    DeviceId primary_device;
    DeviceCapability capability;
};
```

### 10.4 PlacementStrategy

Location: `src/v2/execution/PlacementStrategy.h`

Determines device-to-layer mapping for heterogeneous systems:

```cpp
class PlacementStrategy {
public:
    virtual PlacementPlan createPlan(const PlacementInput& input) = 0;
};

// Available strategies
class CPUOnlyStrategy : public PlacementStrategy;
class GPUFirstStrategy : public PlacementStrategy;
class HybridOptimalStrategy : public PlacementStrategy;  // Bandwidth-proportional

struct PlacementInput {
    int n_layers, n_heads, hidden_dim, vocab_size;
    size_t bytes_per_layer, total_model_bytes;
    int world_size;
    const ClusterInventory* cluster_inventory;
    float gpu_memory_bandwidth, cpu_memory_bandwidth;
    
    // Phase-aware methods
    PhaseDeviceWeights getPhaseDeviceWeights(InferencePhase phase) const;
    bool cpuShouldParticipate(InferencePhase phase) const;  // True if CPU has ≥5% bandwidth
    float getTotalDecodeBandwidth() const;
};

enum class InferencePhase { PREFILL, DECODE };
```

---

This architecture is intentionally modular: small, focused abstractions connected by narrow interfaces. The `IInferenceRunner` interface ensures that callers (Main.cpp, ChatUI, tests) remain decoupled from implementation details of the `GraphOrchestrator`.
