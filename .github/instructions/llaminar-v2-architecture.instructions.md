# Llaminar V2 Detailed Implementation Notes

This document provides detailed implementation-level architecture notes for agents working with **Llaminar V2**. It covers the full stack from CLI config parsing through to kernel execution.

> **For development guidelines**, build commands, testing, and debugging, see `.github/copilot-instructions.md`.
>
> **For the high-level architecture map** (design goals, mental model, layer diagrams), see `.github/instructions/llaminar-architecture-v2.instructions.md`.

---

## 1. Configuration Pipeline

Llaminar V2 uses a **5-phase configuration pipeline** that parses user input once and carries pre-parsed values through the entire system. This avoids redundant string parsing and ensures consistency.

```
Phase 0: CLI/YAML → OrchestrationConfig (raw user-facing config, strings)
Phase 1: ExecutionPlanBuilder → RankExecutionPlan (per-rank contract, parsed RuntimeConfig)
Phase 2: IGraphConfigBuilder → GraphConfig (model-specific execution config)
Phase 3: InferenceRunnerFactory → DeviceGraphOrchestrator / MultiDeviceOrchestrator
Phase 4: IGraphBuilder (via GraphBuilderRegistry) + GraphResolver → ComputeGraph (declarative DAG of stages)
```

### 1.1 Phase 0: OrchestrationConfig

Location: `src/v2/config/OrchestrationConfig.h`

User-facing configuration parsed from CLI flags or YAML. Contains raw strings and enums.

**Key Enums:**
- `TPScope`: AUTO, LOCAL, GLOBAL, HYBRID
- `DeviceAssignmentMode`: AUTO, LOCAL_GPU, ROUND_ROBIN, EXPLICIT
- `PPSplitMode`: EQUAL, WEIGHTED, MANUAL

**Key Structures:**
- `DomainDefinition` — Named TP domain (devices, weights, backend). Format: `"name=device1,device2[;weights=w1,w2][;backend=type]"`
- `PPStageDefinition` — Pipeline stage mapping. Format: `"stage_id=domain_name:first_layer-last_layer"`

Five mutually exclusive device selection modes (enforced by `ConfigValidator`):
1. **Single Device**: `-d cuda:0`
2. **Simple TP**: `-tp 2`
3. **Explicit TP**: `--tp-devices "cuda:0,cuda:1"`
4. **Named Domains**: `--define-domain` + `--pp-stage`
5. **Device Map**: `--device-map "0=cuda:0,1=cuda:1"`

### 1.2 Phase 1: RuntimeConfig (Parse Once, Copy Always)

Location: `src/v2/execution/config/RuntimeConfig.h`

Lightweight struct with **pre-parsed** runtime parameters. Created once via `RuntimeConfig::fromOrchestrationConfig()` during plan building, then carried through the entire config chain.

```cpp
struct RuntimeConfig {
    int max_seq_len = 2048;
    int batch_size = 1;
    ActivationPrecision activation_precision = ActivationPrecision::FP32;
    FusedAttentionBackend fused_attention_backend = FusedAttentionBackend::JIT;
    float kv_cache_scale = 256.0f;
    KVCachePrecision kv_cache_precision = KVCachePrecision::AUTO;
    
    static RuntimeConfig fromOrchestrationConfig(int max_seq_len,
        const std::string& precision_str, const std::string& kv_precision_str, ...);
};
```

**Key Enums:**
- `ActivationPrecision`: FP32, BF16, FP16, Q8_1, Q16_1, Hybrid, HybridQ16
- `FusedAttentionBackend`: JIT, REFERENCE, TILED, Q16_INTEGER
- `KVCachePrecision`: AUTO, FP32, FP16, Q8_1

### 1.3 Phase 1: RankExecutionPlan

Location: `src/v2/execution/mpi_orchestration/RankExecutionPlan.h`

The **contract** between cluster orchestration and rank-local execution. Each MPI rank receives exactly one plan.

**Key Fields:**
- **Identity**: rank, hostname, numa_node
- **PP**: pp_stage_id, first_layer, last_layer, has_embedding, has_lm_head, prev_rank, next_rank
- **TP**: tp_scope, local_tp_devices, local_tp_weights, local_tp_backend
- **Weight Sharding**: shard_index, total_shards, work_fraction
- **Runtime**: `RuntimeConfig runtime` — the pre-parsed config values

### 1.4 Phase 2: GraphConfig

Location: `src/v2/models/GraphTypes.h`

Model-specific execution configuration. Built by `IGraphConfigBuilder` (implemented by `Qwen2GraphConfigBuilder`) from a `RankExecutionPlan`.

**Key Fields:**
- Architecture: n_layers, d_model, n_heads, n_kv_heads, head_dim, d_ff, vocab_size
- TP assignments: head_start, local_n_heads, local_n_kv_heads, qkv_column_parallel
- FFN sharding: d_ff_local, ffn_column_parallel
- LM head sharding: vocab_local, lm_head_column_parallel
- Precision: activation_precision, fused_attention_backend, kv_cache_precision
- Contexts: local_tp_ctx (ILocalTPContext*), pp_contexts, domain_tp_contexts
- Optional: TensorParallelConfig*, PipelineConfig*

### 1.5 Phase 3-4: Runner Creation

Three build paths from `OrchestrationRunner::buildComputeGraph()`:

| Scenario | Method | Creates |
|----------|--------|---------|
| Single device | `buildSingleDeviceComputeGraph()` | `DeviceGraphOrchestrator` |
| LOCAL TP | `buildMultiDeviceComputeGraph()` | `MultiDeviceOrchestrator` (owns N `DeviceGraphOrchestrator`s) |
| LOCAL PP | `buildLocalPPComputeGraph()` | `TreeToRunnerCompiler` → N `DeviceGraphOrchestrator`s |

### 1.6 File Locations

| Component | Location |
|-----------|----------|
| OrchestrationConfig | `src/v2/config/OrchestrationConfig.h` |
| OrchestrationConfigParser | `src/v2/config/OrchestrationConfigParser.h` |
| ConfigValidator | `src/v2/config/ConfigValidator.h` |
| RuntimeConfig | `src/v2/execution/config/RuntimeConfig.h` |
| RankExecutionPlan | `src/v2/execution/mpi_orchestration/RankExecutionPlan.h` |
| IExecutionPlanBuilder | `src/v2/execution/mpi_orchestration/IExecutionPlanBuilder.h` |
| GraphConfig (GraphTypes) | `src/v2/models/GraphTypes.h` |
| IGraphConfigBuilder | `src/v2/models/IGraphConfigBuilder.h` |
| Qwen2GraphConfigBuilder | `src/v2/models/qwen/Qwen2GraphConfigBuilder.h` |
| PipelineConfig | `src/v2/config/PipelineConfig.h` |
| TensorParallelConfig | `src/v2/config/TensorParallelConfig.h` |

---

## 2. Orchestration Runner Layer

### 2.1 IOrchestrationRunner

Location: `src/v2/execution/runner/IOrchestrationRunner.h`

Top-level inference interface for end-to-end generation:

```cpp
class IOrchestrationRunner {
    virtual bool initialize() = 0;
    virtual void shutdown() = 0;
    virtual GenerationResult generate(const std::vector<int32_t>& prompt, int max_new, const SamplingParams&) = 0;
    virtual bool prefill(const std::vector<int32_t>& tokens) = 0;
    virtual std::pair<int32_t, float> decodeStep(const SamplingParams&) = 0;
    virtual void reset() = 0;
    virtual const OrchestrationConfig& config() const = 0;
    virtual const RankExecutionPlan& executionPlan() const = 0;
    virtual ILocalTPContext* localTPContext() = 0;
};
```

### 2.2 OrchestrationRunner

Location: `src/v2/execution/runner/OrchestrationRunner.h`

Implements `IOrchestrationRunner`. Lifecycle:
1. **initialize()**: Build execution plan, create TP/PP contexts, load weights, build compute graph
2. **prefill()** / **decodeStep()**: Delegate to internal `IInferenceRunner`
3. **shutdown()**: Clean teardown

Ownership chain:
- `OrchestrationConfig config_` ← user configuration
- `RankExecutionPlan plan_` ← per-rank contract (with pre-parsed `RuntimeConfig`)
- `ILocalTPContext`, `ILocalPPContext` ← multi-device contexts
- `ModelContext` ← model weights + metadata
- `IInferenceRunner local_runner_` ← one of: DeviceGraphOrchestrator, MultiDeviceOrchestrator, or compiled PP graph

### 2.3 IOrchestrationRunnerFactory

Location: `src/v2/execution/runner/IOrchestrationRunnerFactory.h`

Factory with multiple entry points:
- `createFromArgs(argc, argv)` — CLI parsing
- `createFromConfig(path)` — YAML parsing
- `createFromOrchestrationConfig(config)` — programmatic
- `createSimple(model_path, device)` — quick tests

---

## 3. Local Execution Layer

### 3.1 IInferenceRunner

Location: `src/v2/execution/local_execution/orchestrators/IInferenceRunner.h`

Single-device/rank inference interface:

```cpp
class IInferenceRunner {
    // Core inference
    virtual bool forward(const int* tokens, int seq_len) = 0;
    virtual const float* logits() const = 0;
    virtual int vocab_size() const = 0;
    virtual void clear_cache() = 0;
    virtual int get_position() const = 0;
    
    // GPU-side sampling (avoids D2H transfer of full logits)
    virtual int sampleGreedyOnDevice();
    virtual int sampleOnDevice(const SamplingParams& params);
    
    // TP optimization: skip AllGather when logits aren't consumed
    virtual void setSkipLogitsGatherDecode(bool);
    virtual void setSkipLogitsGatherPrefill(bool);
    
    // Batch inference
    virtual bool forward_batch(const std::vector<std::vector<int>>& token_batches);
    virtual int batch_size() const;
    
    // Hidden state API (for PP nesting)
    virtual TensorBase* getHiddenState();
    virtual void setHiddenState(TensorBase*);
    
    // Snapshot capture (parity testing)
    virtual void enableSnapshotCapture(const std::string& output_dir);
    virtual const float* getSnapshot(const std::string& key, size_t& out_size) const;
    virtual SnapshotInfo getSnapshotWithShape(const std::string& key) const;
    
    // Profiling
    virtual const GraphExecutorStats* executorStats() const;
};
```

### 3.2 DeviceGraphOrchestrator

Location: `src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h`

**Imperative execution layer** for single-device inference. Manages graph execution, state, and caching.

**Key Responsibilities:**
- Graph execution via `DeviceGraphExecutor`
- Lazy device context initialization per device
- Graph caching for decode mode (seq_len=1)
- Inference state management (buffers, KV cache, positions)
- BufferArena integration (Phase 2)
- PP stage nesting support

**Three Constructors:**
1. `DeviceGraphOrchestrator(Dependencies deps, const GraphConfig&, const GraphCacheConfig&)` — DI (preferred for testing)
2. `DeviceGraphOrchestrator(shared_ptr<IGraphBuilder>, shared_ptr<MPIContext>, ...)` — pre-built graph builder
3. `DeviceGraphOrchestrator(const GraphConfig&, shared_ptr<MPIContext>, ...)` — creates internal builder via `GraphBuilderRegistry`

**InferenceState** (all mutable per-inference state):
```cpp
struct InferenceState {
    // Core buffers
    shared_ptr<TensorBase> hidden, logits, logits_local;
    
    // KV cache (single-device or per-PP-stage)
    unique_ptr<IKVCache> kv_cache;
    unordered_map<DeviceId, unique_ptr<IKVCache>> pp_kv_caches;
    
    // Position tracking
    vector<int> positions, sequence_lengths;
    
    // Activation buffers (reused across layers)
    shared_ptr<TensorBase> normalized, residual;
    shared_ptr<TensorBase> Q, K, V, Q_rope, K_rope, V_dequant;
    shared_ptr<TensorBase> attn_output, attn_proj;
    shared_ptr<TensorBase> gate, up, ffn_output;
    
    // Workspace buffers
    shared_ptr<TensorBase> workspace_scores, workspace_context, workspace_mask;
};
```

**Key Execution Methods:**
- `executeForward(ForwardInput, ForwardOutput)` — full forward (embedding → layers → LM head)
- `executeAttention(...)` — single attention block
- `executeFFN(...)` — single FFN block
- `executeLayer(...)` — attention + FFN

**Phase-Aware Inference:** Supports `setPhase(InferencePhase)` to switch between PREFILL and DECODE behavior. Phase-aware weight access returns different weights per phase (e.g., for weight streaming).

**Graph Caching (Decode Optimization):**
```cpp
// For decode mode (seq_len=1), graphs are cached and reused
if (seq_len == 1 && layer_graph_cache_[layer_idx].valid) {
    updateCachedGraphParams(*cache.attention_decode, pos_offset, seq_len);
    executor_.execute(*cache.attention_decode, ctx);
}
```

### 3.3 MultiDeviceOrchestrator

Location: `src/v2/execution/local_execution/orchestrators/MultiDeviceOrchestrator.h`

Coordinates multiple `DeviceGraphOrchestrator` instances for **LOCAL tensor parallelism (TP)** across multiple devices within a single MPI rank. Also supports **Pipeline Parallelism (PP)** via nested orchestrators.

**Config Structure:**
```cpp
struct Config {
    ParallelismMode mode;                        // AUTO, TP, PP, TP_PP
    vector<GlobalDeviceAddress> devices;         // TP devices
    vector<float> weights;                       // Proportional weights (e.g., {0.73, 0.27})
    CollectiveBackendType backend;               // NCCL, RCCL, HOST, etc.
    vector<PPStageConfig> pp_stages;             // For PP mode
    size_t max_seq_len;
    int batch_size;
    ActivationPrecision activation_precision;
    KVCachePrecision kv_cache_precision;
    bool use_mapped_memory;                      // GPU zero-copy access
    
    ParallelismMode detectMode() const;
    static Config fromPlan(const RankExecutionPlan& plan);
};
```

**Key Members:**
- `vector<unique_ptr<DeviceGraphOrchestrator>> device_runners_` — per-device runners
- `unique_ptr<ILocalTPContext> tp_ctx_` — LOCAL TP context (NCCL/RCCL/HOST)
- `unique_ptr<ILocalPPContext> pp_ctx_` — inter-stage transfers
- `unique_ptr<TensorBase> combined_logits_` — AllGather output
- `unique_ptr<TPWorkerPool> tp_worker_pool_` — persistent thread pool for TP
- `bool skip_logits_gather_decode_` / `skip_logits_gather_prefill_` — skip D2H optimization

**Execution Flow (TP Mode):**
```
forward(tokens, seq_len)
  → forwardTP()
    → Each device runner executes in parallel (via TPWorkerPool)
      → runner->forward(tokens, seq_len)   [all devices process same tokens]
      → AllReduce after row-parallel ops (Wo, FFN down) via tp_ctx_
    → Optional: gatherLogits() — AllGather partial logits
      → Skip if skip_logits_gather_{decode,prefill}_ set
  → Return combined_logits_
```

**Execution Flow (PP Mode):**
```
forward(tokens, seq_len)
  → forwardPP()
    → Stage 0: process tokens → hidden_state
    → pp_ctx_->transferHidden(0, 1)
    → Stage 1: process hidden → hidden_state
    → ... continue through stages ...
    → Final stage produces logits
    → copyLogitsFromStage(final_stage_idx)
```

**GPU-Side Sampling:**
- `sampleGreedyOnDevice()` — GPU argmax, D2H only the (token, logprob) pair
- `sampleOnDevice(SamplingParams)` — GPU top-k/top-p sampling

---

## 4. Memory Management: BufferArena

Location: `src/v2/memory/`

The **BufferArena** system provides centralized buffer management with typed registration, runtime coherence tracking, borrow validation, and aliasing support.

### 4.1 BufferId

Location: `src/v2/memory/BufferId.h`

Enum identifying all activation, workspace, and staging buffers:

```cpp
enum class BufferId : uint32_t {
    // Cross-layer persistent
    HIDDEN_STATE, LOGITS, LOGITS_LOCAL,
    
    // Per-layer activations (recycled)
    NORMALIZED, RESIDUAL,
    Q_PROJ, K_PROJ, V_PROJ, Q_ROPE, K_ROPE, V_DEQUANT,
    ATTN_OUTPUT, ATTN_PROJ,
    GATE_PROJ, UP_PROJ, FFN_OUTPUT, FFN_NORMALIZED,
    
    // Workspace / scratch
    ATTN_SCORES_WORKSPACE, ATTN_CONTEXT_WORKSPACE, GEMM_WORKSPACE,
    
    // Collective staging
    ALLREDUCE_STAGING, ALLGATHER_STAGING,
    
    // Quantized buffers
    Q_QUANTIZED, K_QUANTIZED,
    
    _COUNT  // 32 entries total
};
```

### 4.2 BufferArena

Location: `src/v2/memory/BufferArena.h`

**Single source of truth** for all activation buffers. Fixed-size array of `ManagedBuffer` entries.

**Lifecycle:**
1. **Registration**: `registerBuffer()` / `registerExternalBuffer()` — declare buffers with shape, dtype, device
2. **Aliasing**: `registerAlias(a, b)` — mark scratch buffers as sharing storage
3. **Allocation**: `allocate()` — create all TensorBase instances
4. **Runtime**: `prepareForRead/Write()`, `markWritten()` — coherence management per stage
5. **Destruction**: `~BufferArena()` — frees all owned tensors

**ManagedBuffer (internal):**
```cpp
struct ManagedBuffer {
    bool registered;
    shared_ptr<TensorBase> owned_tensor;    // Arena-owned
    ITensor* external_tensor;               // Externally owned (weights)
    size_t rows, cols;
    const char* dtype;
    DeviceId home_device;
    CoherenceState coherence;               // HOST, DEVICE, or UNINITIALIZED
    int active_read_borrows;                // Debug: borrow tracking
    bool active_write_borrow;
    int alias_group;                        // Aliasing group ID
};
```

**API Categories:**

| Category | Methods |
|----------|---------|
| Registration | `registerBuffer()`, `registerExternalBuffer()`, `registerAlias()` |
| Allocation | `allocate()` |
| Coherence | `prepareForRead()`, `prepareForWrite()`, `markWritten()`, `markWrittenFlagsOnly()` |
| Borrow Tracking | `acquireReadBorrow()`, `acquireWriteBorrow()`, `releaseReadBorrow()`, `releaseWriteBorrow()`, `validateNoBorrowsActive()` |
| Access | `getTensor()`, `getDevicePtr()`, `getRows()`, `getCols()` |
| Introspection | `getCoherenceState()`, `isRegistered()`, `registeredCount()`, `isAllocated()` |

### 4.3 CoherenceTracker

Location: `src/v2/memory/CoherenceTracker.h`

**Internal to BufferArena** — manages per-buffer host↔device transfers.

```cpp
struct CoherenceState {
    enum Authority { UNINITIALIZED, HOST, DEVICE };
    Authority authority;
    DeviceId authoritative_device;
    bool needsTransferTo(DeviceId target) const;
};

class CoherenceTracker {  // All static methods
    static bool prepareForRead(TensorBase*, CoherenceState&, DeviceId target);
    static bool prepareForWrite(TensorBase*, CoherenceState&, DeviceId target);
    static void markWritten(CoherenceState&, DeviceId);
    static void markWrittenWithEvent(TensorBase*, CoherenceState&, DeviceId, void* stream);
    static void markWrittenFlagsOnly(TensorBase*, CoherenceState&, DeviceId);
};
```

### 4.4 StageBoundBuffers

Location: `src/v2/memory/StageBoundBuffers.h`

**Immutable buffer collection** built from BufferArena borrows, passed to `stage->execute()`:

```cpp
class StageBoundBuffers {
    // Typed read/write access with compile-time access control
    template<typename T> BufferView<T, READ> input(BufferId id) const;
    template<typename T> BufferView<T, WRITE> output(BufferId id) const;
    template<typename T> BufferView<T, READ> weight(BufferId id) const;
    template<typename T> BufferView<T, READWRITE> inout(BufferId id) const;
    
    ITensor* weightTensor(BufferId id) const;
    void* workspace(const char* name) const;
    bool has(BufferId id) const;
};
```

### 4.5 StageBufferContract

Location: `src/v2/memory/StageBufferContract.h`

**Declarative I/O specification** for compute stages — describes what buffers a stage reads, writes, and needs as workspace. Returned by `IComputeStage::bufferContract()`.

```cpp
struct BufferBinding {
    BufferId id;
    BufferAccess access;       // READ, WRITE, READWRITE
    const char* dtype;         // "FP32", "Q8_1", etc.
};

struct WorkspaceDesc {
    const char* name;
    size_t size_bytes;
    size_t alignment;
};

struct StageBufferContract {
    vector<BufferBinding> inputs;
    vector<BufferBinding> outputs;
    vector<BufferBinding> weight_tensors;
    vector<BufferBinding> inouts;
    vector<WorkspaceDesc> workspaces;
};
```

Fluent builder pattern: `StageBufferContract::build().addInput(...).addOutput(...).addWeight(...)`

The executor uses the contract to drive BufferArena borrows and coherence transitions automatically.

### 4.6 BufferArena Integration with DeviceGraphExecutor

```
DeviceGraphExecutor::executeStage(node):
  1. BufferArena::acquireReadBorrow(inputs)
  2. BufferArena::prepareForRead(inputs) → CoherenceTracker (H2D if needed)
  3. BufferArena::acquireWriteBorrow(outputs)
  4. BufferArena::prepareForWrite(outputs) → CoherenceTracker (allocate GPU buffer)
  5. Build StageBoundBuffers from arena
  6. stage->execute(StageBoundBuffers)
  7. BufferArena::markWritten(outputs)
  8. BufferArena::releaseReadBorrow(inputs)
  9. BufferArena::releaseWriteBorrow(outputs)
```

---

## 5. Graph System

### 5.1 GraphSchema (Declarative Graph Definition)

Location: `src/v2/execution/local_execution/graph/GraphSchema.h`

The schema system provides a **declarative** specification of the compute graph, separate from execution.

**Key Structures:**

| Structure | Purpose |
|-----------|---------|
| `GraphSchema` | Top-level schema (name, embedding, layer template, LM head stages, buffer specs, alias groups) |
| `LayerTemplate` | Per-layer stage specs (attention_stages, ffn_stages) |
| `StageSpec` | Single stage definition (name, type, inputs, outputs, dependencies, TP mode, parameters) |
| `BufferSpec` | Buffer definition (name, shape formulas, dtype, alias group) |
| `AliasGroupSpec` | Alias group (name, buffer names, estimated savings) |
| `WeightShardingConfig` | Per-weight sharding rules (patterns, exact matches, dimension types) |

**StageType Enum:** RMSNorm, LayerNorm, GEMM, FusedQKVGEMM, FusedGateUpGEMM, RoPE, KVCacheAppend, KVCacheGather, AttentionCompute, SwiGLU, GELU, ResidualAdd, Embedding, LMHead, Allreduce, Allgather, MoERouter, MoEFFN, Quantize, Dequantize, QKNorm

**TPMode Enum:** None, ColumnParallel, RowParallel, ExpertParallel

**ISchemaFactory Interface:**
```cpp
class ISchemaFactory {
    virtual GraphSchema createSchema() const = 0;
    virtual WeightShardingConfig getWeightShardingConfig() const = 0;
    virtual StageShardingConfig getStageShardingConfig() const = 0;
};
```

### 5.2 GraphResolver

Location: `src/v2/execution/local_execution/graph/GraphResolver.h`

Resolves a declarative `GraphSchema` into concrete `ResolvedGraphSpec` by binding tensors, sequence lengths, and TP sharding info.

**GraphResolverConfig:**
```cpp
struct GraphResolverConfig {
    // MPI/TP
    int world_size, rank;
    const MPIContext* mpi_ctx;
    unordered_map<string, ShardingInfo> weight_sharding;
    
    // Execution policy (stage-level enable/disable)
    ExecutionPolicyFlags exec_policy;
    
    // Sequence/batch
    int batch_size, seq_len, position_offset;
    bool has_kv_cache;
    int cached_tokens;
    
    // Model config
    int n_layers, d_model, n_heads, n_kv_heads, head_dim, d_ff, vocab_size;
    int local_n_heads, local_n_kv_heads, local_d_ff, local_vocab;
    float rms_norm_eps, rope_theta, kv_cache_scale;
    DeviceId default_device;
};
```

**TensorContext** — binds weight names and buffers to actual tensors:
```cpp
struct TensorContext {
    unordered_map<string, TensorBase*> model_weights;
    function<TensorBase*(int, const string&)> get_layer_weight;
    unordered_map<string, TensorBase*> buffers;
    IKVCache* kv_cache;
    const int* position_ids, *token_ids;
    
    TensorBase* resolve(const TensorRef& ref, int layer_idx = -1) const;
};
```

**Resolution Flow:**
```
GraphSchema  +  GraphResolverConfig  +  TensorContext
       ↓              ↓                      ↓
  GraphResolver::resolve() → ResolvedGraphSpec
       ↓
  ResolvedStage[] (each with bound tensors, computed parameters, dependencies)
```

### 5.3 ComputeGraph and DeviceGraphExecutor

Location: `src/v2/execution/local_execution/graph/DeviceGraphExecutor.h`

**ComputeGraph** — DAG of `ComputeNode` structures:
```cpp
struct ComputeNode {
    string name;
    unique_ptr<IComputeStage> stage;
    vector<string> dependencies;
    DeviceId device;
    bool completed;
    bool weights_cohered;      // Fast-path: skip re-upload for cached graphs
    bool is_final_output;      // Mark for D2H transfer avoidance
};

class ComputeGraph {
    ComputeGraph& addNode(const string& name, unique_ptr<IComputeStage> stage, DeviceId device);
    ComputeGraph& addDependency(const string& node, const string& depends_on);
    ComputeGraph& merge(ComputeGraph&& other, const string& connect_from = "");
    const vector<string>& getExecutionOrder() const;  // Topological sort
    void reset();  // For graph reuse in decode mode
};
```

**DeviceGraphExecutor** — executes ComputeGraph nodes via a **single `runStages()` loop** parameterized by `StageRunPolicy`, eliminating divergent code path bugs:

```cpp
struct StageRunPolicy {
    bool coherence = true;            // Arena contract-based input/output coherence
    bool weight_coherence = true;     // Upload weights to device
    bool mark_dirty = true;           // Mark outputs device-authoritative (ALWAYS ON)
    bool validation = true;           // NaN/Inf output validation
    bool profiling = true;            // Per-stage timing breakdown
    bool collective_intercept = true; // Use CollectiveContext for allreduce/allgather
    bool timeline = false;            // GPU event-based per-stage profiling
    bool stage_dump = true;           // Stage dump framework
    bool snapshot_callback = true;    // Invoke snapshot callback
    bool pointer_validation = false;  // GPU pointer device validation
    
    static StageRunPolicy full();       // Prefill, cache miss: everything on
    static StageRunPolicy fastDecode(); // Cached decode (M=1): minimal overhead
    static StageRunPolicy debug();      // Full + timeline + pointer validation
};

class DeviceGraphExecutor : public IGraphExecutor {
    bool execute(ComputeGraph& graph, IDeviceContext* ctx);
    bool executeMultiDevice(ComputeGraph& graph, const unordered_map<DeviceId, IDeviceContext*>&);
    
    void setCollectiveContext(ICollectiveContext* ctx);
    void setSnapshotCallback(StageSnapshotCallback callback);
    void setProfilingEnabled(bool enabled);
    void setValidationEnabled(bool enabled);
    void setCurrentLayerIdx(int layer_idx);
};
```

**Execution Loop** (driven by StageRunPolicy):
1. If `policy.coherence`: Compute `StageBufferContract` from stage, drive BufferArena borrows and H2D transfers
2. If `policy.weight_coherence`: Upload weight tensors to device
3. If `policy.validation`: `verifyStageEntry()` — NaN/zero check on inputs
4. `node.stage->execute()` — run compute stage
5. If `policy.mark_dirty`: Mark outputs as GPU-authoritative via BufferArena
6. If `policy.validation`: `verifyStageExit()` — NaN/zero check on outputs
7. If `policy.snapshot_callback`: Invoke snapshot callback (parity testing)

### 5.4 Model-Agnostic Graph Building

The graph system is fully decoupled from any specific model architecture via registries and a single registration file.

#### GraphBuilderRegistry

Location: `src/v2/execution/local_execution/graph/GraphBuilderRegistry.h`

Factory registry mapping architecture strings (e.g., `"qwen2"`, `"qwen3"`) to `IGraphBuilder` implementations:

```cpp
class GraphBuilderRegistry {
    static void registerFactory(const string& arch, FactoryFn fn);
    static unique_ptr<IGraphBuilder> create(const string& arch, const GraphConfig& config,
                                            shared_ptr<MPIContext> mpi_ctx);
    static bool isSupported(const string& arch);
    static vector<string> supportedArchitectures();
};
```

Uses `std::call_once` via `ensureBuiltins()` for lazy initialization.

#### SchemaFactoryRegistry

Location: `src/v2/execution/local_execution/graph/SchemaFactoryRegistry.h`

Parallel registry for `ISchemaFactory` implementations providing weight sharding and stage sharding configs per architecture:

```cpp
class SchemaFactoryRegistry {
    static ISchemaFactory* getFactory(const string& arch);
    static WeightShardingConfig getWeightShardingConfig(const string& arch);
    static StageShardingConfig getStageShardingConfig(const string& arch);
    static bool isSupported(const string& arch);
    static void registerFactory(const string& arch, FactoryFn fn);
};
```

#### ModelRegistrations

Location: `src/v2/models/ModelRegistrations.cpp`

**Single ~56-line file** that registers all built-in model architectures:

```cpp
void registerBuiltinModels() {
    // Graph builders
    GraphBuilderRegistry::registerFactory("qwen2", ...Qwen2Graph...);
    GraphBuilderRegistry::registerFactory("qwen3", ...Qwen2Graph...);  // Reuses Qwen2Graph
    
    // Schema factories
    SchemaFactoryRegistry::registerFactory("qwen2", ...Qwen2SchemaFactory...);
    SchemaFactoryRegistry::registerFactory("qwen3", ...Qwen3SchemaFactory...);
}
```

**Key insight**: Qwen2 and Qwen3 share the same `Qwen2Graph` builder but differ in `ISchemaFactory` (Qwen3 adds QK-norm stages). To add a new model architecture, add entries to `ModelRegistrations.cpp` and implement an `ISchemaFactory`.

#### IGraphBuilder

Location: `src/v2/execution/local_execution/graph/IGraphBuilder.h`

Interface for **stateless, declarative** compute graph builders:

```cpp
class IGraphBuilder {
    virtual ComputeGraph buildFullForwardGraph(const ForwardInput&, ForwardOutput&) = 0;
    virtual ComputeGraph buildEmbeddingGraph(...) = 0;
    virtual ComputeGraph buildTransformerLayersGraph(...) = 0;
    virtual ComputeGraph buildLayerGraph(int layer_idx, ...) = 0;
    virtual ComputeGraph buildLMHeadGraph(...) = 0;
    virtual GraphSchema getSchema() const = 0;
    virtual GraphResolverConfig getResolverConfig(int seq_len) const = 0;
};
```

#### Qwen2Graph (Concrete Builder)

Location: `src/v2/models/qwen/Qwen2Graph.h`

Implements `IGraphBuilder` for Qwen2/Qwen3 architectures. Stateless — creates `ComputeGraph` DAGs from model configuration. Configuration methods register parallelism contexts (`setPipelineConfig()`, `setTPContext()`), model state (`setWeights()`, `setBuffers()`), and allocators (`setTensorFactory()`).

#### Model Directory Layout

```
src/v2/models/
├── ModelRegistrations.cpp/.h       # Central model registration (add new models here)
├── GraphTypes.h                    # GraphConfig struct
├── IGraphConfigBuilder.h           # Interface for graph config building
├── qwen/
│   ├── Qwen2Graph.h/.cpp          # IGraphBuilder for Qwen2/Qwen3
│   ├── Qwen2Schema.h              # ISchemaFactory for Qwen2
│   ├── Qwen2GraphConfigBuilder.h  # IGraphConfigBuilder
│   └── Qwen2BufferSpec.h/.cpp     # Buffer specifications
└── qwen3/
    └── Qwen3Schema.h              # ISchemaFactory for Qwen3 (QK-norm, no QKV bias)
```

---

## 6. Compute Stages

### 6.1 IComputeStage Interface

Location: `src/v2/execution/compute_stages/IComputeStage.h`

```cpp
class IComputeStage {
    virtual bool execute(void* ctx = nullptr) = 0;
    virtual ComputeStageType type() const = 0;
    virtual const char* name() const = 0;
    virtual StageDumpInfo getDumpInfo() const = 0;       // REQUIRED: introspection
    virtual StageBufferRequirements getBufferRequirements() const;
    virtual StageBufferContract bufferContract() const;  // Declarative I/O specification
    virtual CoherencePolicy coherencePolicy() const;     // Default: FULL
};
```

**CoherencePolicy:**

| Policy | Entry: Inputs | Entry: Outputs | Exit |
|--------|---------------|----------------|------|
| `FULL` (default) | Upload inputs + weights | Allocate output buffers | Mark outputs dirty |
| `INPUT` | Upload inputs + weights | No allocation | No-op |
| `OUTPUT` | No upload | Allocate output buffers | Mark outputs dirty |
| `NONE` | No-op | No-op | No-op (MPI stages manage own sync) |

### 6.2 StageDumpInfo

Every stage must implement `getDumpInfo()` — the single source of truth for:
1. **TensorVerification** — entry/exit validation
2. **Snapshot callbacks** — parity testing against PyTorch
3. **StageDumper** — binary dumps for debugging

```cpp
struct StageDumpInfo {
    struct InputBuffer { const char* name; const void* data; size_t rows, cols; const char* dtype; };
    struct OutputBuffer { /* same fields */ };
    struct WeightBuffer { /* includes TensorBase* for full metadata */ };
    struct ScalarParam { const char* name; double value; const char* dtype; };
    
    vector<InputBuffer> inputs;
    vector<OutputBuffer> outputs;
    vector<WeightBuffer> weights;
    vector<ScalarParam> scalars;
    
    // Fluent builder
    StageDumpInfo& addInput(const char* name, const float* data, size_t rows, size_t cols);
    StageDumpInfo& addOutput(const char* name, const TensorBase* tensor, size_t rows, size_t cols);
    StageDumpInfo& addWeight(const char* name, const TensorBase* tensor);
};
```

### 6.3 Available Compute Stages

| Stage Class | Type | Purpose | MPI? |
|-------------|------|---------|------|
| `RMSNormStage` | RMS_NORM | RMS normalization | No |
| `GEMMStage` | GEMM | Matrix multiplication | No |
| `FusedQKVGEMMStage` | GEMM_FUSED_QKV | Q/K/V projection | No |
| `FusedGateUpGEMMStage` | GEMM_FUSED_GATE_UP | Gate/Up projection | No |
| `SwiGLUStage` | SWIGLU | SwiGLU activation | No |
| `RoPEStage` | ROPE | Rotary position embeddings | No |
| `ResidualAddStage` | ADD_RESIDUAL | Residual connections | No |
| `AttentionComputeStage` | ATTENTION_COMPUTE | Pure attention | No |
| `KVCacheAppendStage` | KV_CACHE_APPEND | KV cache management | No |
| `EmbeddingStage` | EMBEDDING | Token embedding | No |
| `LMHeadStage` | LM_HEAD | Final projection | No |
| `QuantizeQ16_1Stage` | QUANTIZE_Q16_1 | Activation quantization | No |
| `AllreduceStage` | ALLREDUCE | MPI_Allreduce(SUM) | **Yes** |
| `AllGatherStage` | ALLGATHER | MPI_Allgather | **Yes** |
| `AllGatherVStage` | ALLGATHER_V | Variable-count AllGather | **Yes** |
| `SendActivationsStage` | SEND_ACTIVATIONS | PP send | **Yes** |
| `ReceiveActivationsStage` | RECEIVE_ACTIVATIONS | PP receive | **Yes** |

---

## 7. Model Executor Layer

Location: `src/v2/execution/local_execution/model/`

### 7.1 IModelExecutor

Location: `src/v2/execution/local_execution/model/IModelExecutor.h`

Abstract interface for model-level execution with phase timing and graph building:

```cpp
struct ForwardInput {
    const int* token_ids;
    const vector<vector<int>>* batches;
    int batch_size, seq_len;
    int* position_ids, position_offset;
    IKVCache* kv_cache;
    DeviceId device;
};

struct ForwardOutput {
    TensorBase* logits, *hidden_states;
    bool return_hidden_states;
};

class IModelExecutor {
    virtual ComputeGraph buildFullForwardGraph(const ForwardInput&, ForwardOutput&) = 0;
    virtual ComputeGraph buildEmbeddingGraph(const ForwardInput&, TensorBase* output) = 0;
    virtual ComputeGraph buildTransformerLayersGraph(TensorBase*, IKVCache*, const int*, DeviceId) = 0;
    virtual ComputeGraph buildLayerGraph(int layer_idx, TensorBase*, IKVCache*, const int*, DeviceId) = 0;
    virtual ComputeGraph buildLMHeadGraph(TensorBase* hidden, TensorBase* logits, int tokens, DeviceId, TensorBase* logits_local = nullptr) = 0;
    virtual bool executeForward(const ForwardInput&, ForwardOutput&) = 0;
    virtual bool execute(ComputeGraph&, IDeviceContext*) = 0;
};
```

### 7.2 ModelExecutorConfig

```cpp
struct ModelExecutorConfig {
    LayerExecutorConfig layer_config;
    int n_layers, max_batch_size, max_seq_len;
    bool enable_kv_cache = true;
    bool fuse_embedding = true, fuse_lm_head = true;
    bool per_layer_graphs = false;
    bool profile_per_phase = false;
};
```

**Note:** `ILayerExecutor` is an alias for `DeviceGraphExecutor`. The ModelExecutor sits above DeviceGraphExecutor and provides model-level graph building and phase-aware execution.

---

## 8. Tensor System

### 8.1 Type Hierarchy

```
ITensor (runtime polymorphism interface)
  └── TensorBase (device coherence + host storage)
        └── TypedTensorBase<Derived, DataType> (CRTP zero-overhead access)
              ├── FP32Tensor (float*)
              ├── BF16Tensor (uint16_t*)
              ├── FP16Tensor (uint16_t*)
              ├── Q8_1Tensor (Q8_1Block*)
              ├── Q16_1Tensor (Q16_1Block*)
              ├── Q8_0Tensor (Q8_0Block*)
              ├── Q4_0Tensor (Q4_0Block*)
              ├── IQ4_NLTensor (IQ4_NLBlock*)
              ├── Q6_KTensor (Q6_KBlock*)
              ├── INT32Tensor (int32_t*)
              └── ... (27 total)
```

**typed_data() Pattern:**
```cpp
if (auto* q8 = dynamic_cast<Q8_1Tensor*>(tensor)) {
    Q8_1Block* blocks = q8->mutable_typed_data();  // Zero-overhead access
}
```

### 8.2 TensorLayout Contracts

Location: `src/v2/tensors/TensorLayout.h`

| Layout | Shape | Use Case |
|--------|-------|----------|
| `Q_SEQ_HEAD_DIM` | `[seq_len][n_heads][head_dim]` | Query after GEMM |
| `Q_HEAD_SEQ_DIM` | `[n_heads][seq_len][head_dim]` | Per-head parallel attention |
| `KV_POS_HEAD_DIM` | `[position][n_kv_heads][head_dim]` | KV cache POSITION_MAJOR |
| `KV_HEAD_POS_DIM` | `[n_kv_heads][position][head_dim]` | KV cache HEAD_MAJOR |
| `ROW_MAJOR_2D` | `[rows][cols]` | Embeddings, hidden states |

### 8.3 Device Coherence Protocol

Llaminar has two coherence layers: a **new explicit state machine** (`TensorCoherenceState`) and the **legacy per-tensor methods** still on `TensorBase`.

#### TensorCoherenceState (New)

Location: `src/v2/tensors/CoherenceState.h`

Explicit compile-time-verifiable state machine replacing the implicit boolean combinations:

```cpp
enum class TensorCoherenceState : uint8_t {
    HOST_ONLY,              // Data only on host. No GPU buffer allocated.
    HOST_AUTHORITATIVE,     // Both exist, host modified more recently.
    DEVICE_AUTHORITATIVE,   // Both exist, device modified more recently.
    SYNCED,                 // Both exist and are identical.
    MAPPED,                 // Zero-copy mapped memory. Both always valid.
    INVALID,                // Error state — should never be reached.
};

enum class CoherenceOp : uint8_t {
    UPLOAD, DOWNLOAD, MARK_DEVICE_DIRTY, MUTABLE_HOST_ACCESS, RELEASE_DEVICE
};

enum class MemoryResidency : uint8_t {
    STANDARD, MAPPED
};
```

**Transition table**: `COHERENCE_TRANSITIONS[state][op] → {new_state, valid}` — constexpr, compile-time verified. Invalid transitions (e.g., `MARK_DEVICE_DIRTY` on `HOST_ONLY`) produce `INVALID` with `valid = false`.

**CoherenceAuditLog** (`src/v2/tensors/CoherenceAuditLog.h`): Per-tensor ring buffer (32 entries) recording state transitions for debugging. Dumped automatically on verification failure.

#### TransferEngine

Location: `src/v2/transfer/TransferEngine.h`

Unified, stateless data movement with **plan/execute separation**:

```cpp
class TransferEngine {
    static TransferMethod planTransfer(DeviceId src, DeviceId dst, MemoryResidency residency);
    static bool execute(const TransferRequest& request);
    static bool upload(TensorBase* tensor, DeviceId device);
    static bool download(TensorBase* tensor);
    static bool transferActivation(TensorBase* tensor, DeviceId device);
};
```

**TransferMethod enum** (`src/v2/transfer/TransferMethod.h`):

| Method | When Used |
|--------|-----------|
| `NOOP` | Same device — no transfer needed |
| `HOST_TO_DEVICE` | Standard H2D via `IBackend::hostToDevice()` |
| `DEVICE_TO_HOST` | Standard D2H via `IBackend::deviceToHost()` |
| `DEVICE_TO_DEVICE_SAME_BACKEND` | P2P within same vendor (CUDA↔CUDA or ROCm↔ROCm) |
| `HOST_STAGED` | Cross-vendor: D2H → memcpy → H2D via host staging |
| `MAPPED_NOOP` | Zero-copy mapped memory — no transfer needed |

**MemoryDescriptor** — snapshot of where a tensor's data physically lives (host_ptr, device_ptr, mapped pointers, residency). Created via `MemoryDescriptor::fromTensor()` to decouple transfer logic from `TensorBase` internals.

#### Legacy Per-Tensor Methods (TensorBase)

The original per-tensor methods still exist on `TensorBase` and are used by the `BufferArena::CoherenceTracker` internally:

| State | Meaning | `data()` Returns |
|-------|---------|------------------|
| Host valid, device invalid | Initial or host-written | Host data directly |
| Host valid, device valid | Both up-to-date | Host data directly |
| Host invalid, device valid | GPU kernel wrote | Syncs GPU→host first |
| Mapped memory | Zero-copy shared | Either directly |

**Core Methods:**
- `ensureOnDevice(DeviceId)` — upload to GPU (lazy-allocate, pin host memory)
- `mark_device_dirty()` — mark GPU as authoritative
- `mark_device_dirty_with_event()` — record completion event for fine-grained sync
- `data()` — host pointer (syncs from GPU if device-dirty)
- `mutable_data()` — host pointer + marks host as authoritative
- `gpu_data_ptr()` — raw GPU buffer pointer

**Automatic Coherence** is handled by `BufferArena` + `CoherenceTracker` in the `DeviceGraphExecutor` via `StageBufferContract`. Legacy `StageCoherence` (`src/v2/execution/local_execution/coherence/StageCoherence.h`) still exists for stages not yet migrated to the contract system. Manual coherence for tests uses `with_gpu_coherence()` or `GpuOutput<T>` RAII wrappers from `src/v2/execution/local_execution/coherence/GpuCoherence.h`.

### 8.4 UnifiedKVCache

Location: `src/v2/tensors/UnifiedKVCache.h`

**Layout Modes:**
- `POSITION_MAJOR`: `[position][n_kv_heads][head_dim]` — cache-append friendly (default for FP32/BF16)
- `HEAD_MAJOR`: `[n_kv_heads][position][head_dim]` — attention-compute friendly (optimal for Q16_1)

**Interface:**
```cpp
class IUnifiedKVCache {
    virtual void append(TensorBase* K, TensorBase* V, int seq_len) = 0;
    virtual void shift(int positions_to_evict) = 0;
    virtual int cached_tokens() const = 0;
    virtual KVCacheLayoutMode layout_mode() const = 0;
    virtual TensorBase* get_k_for_kv_head(int kv_head_idx) = 0;  // HEAD_MAJOR only
    virtual TensorBase* get_v_for_kv_head(int kv_head_idx) = 0;
    virtual TensorBase* get_k_cache() = 0;                        // POSITION_MAJOR
    virtual TensorBase* get_v_cache() = 0;
};
```

### 8.5 TensorVerification System

Location: `src/v2/tensors/TensorVerification.h`

Automatic stage boundary validation in Debug/Integration builds:
- NaN/Inf detection (fails by default)
- Null pointer detection
- All-zero detection (warns)
- Automatic buffer dump on failure to `/tmp/llaminar_verification_dump/`

**Environment Variables:**
| Variable | Default | Description |
|----------|---------|-------------|
| `LLAMINAR_VALIDATE_BUFFERS` | Auto (Debug/Integration) | Enable buffer validation |
| `LLAMINAR_FAIL_ON_NAN` | 1 | Throw on NaN/Inf |
| `LLAMINAR_FAIL_ON_ZERO` | 0 | Throw on all-zero |
| `LLAMINAR_DUMP_ON_FAILURE` | 1 | Dump buffers on failure |

---

## 9. Kernel Layer

### 9.1 KernelFactory

Location: `src/v2/kernels/KernelFactory.h`

Centralized kernel dispatch with caching. Used primarily during weight loading and preparation; graph builders configure stages with their kernels during graph construction.

```cpp
class KernelFactory {
    static DeviceType getDeviceType(int device_idx);
    static ITensorGemm* getOrCreateGemm(const TensorBase* tensor);  // Cached
    static void clearCacheFor(const TensorBase* tensor);
    static void clearCache();
};
```

### 9.2 Kernel Interfaces

| Interface | Purpose |
|-----------|---------|
| `ITensorGemm` | GEMM between activations and/or weights |
| `ITensorAttention` | Attention over Q/K/V |
| `ITensorRoPE` | Rotary position embeddings |
| `IWorkspaceConsumer` | Workspace binding for scratch memory |
| `IBlockDecoder` / `ITensorGemmTileDataProvider` | Format-specific decode strategy for quantized GEMM |

### 9.3 Backend Implementations

| Backend | GEMM | Attention | Specialties |
|---------|------|-----------|-------------|
| CPU | OpenBLAS/MKL, AVX-512 VNNI JIT, QuantizedGEMM | CpuAttentionKernelT, JIT Q16IntegerAttention | SIMD vectorized primitives |
| CUDA | cuBLAS, TensorCore WMMA, CUDAQuantGEMM | FlashAttention-style | Compute capability dispatch |
| ROCm | rocBLAS, MatrixCore, ROCmQuantGEMM | comp_kernel attention | CDNA/RDNA dispatch |

---

## 10. Collective Layer

### 10.1 CollectiveContext

Location: `src/v2/execution/local_execution/collective/CollectiveContext.h`

Bridges compute stages to collective backends:
```cpp
class CollectiveContext : public ICollectiveContext {
    bool executeAllreduce(TensorBase* buffer, CollectiveOp op, DeviceId device);
    bool executeAllgather(TensorBase* local, TensorBase* full, int seq_len, DeviceId device);
    bool executeAllgatherV(TensorBase* local, TensorBase* full, const vector<int>& counts, DeviceId device);
    bool executeBroadcast(TensorBase* buffer, int root, DeviceId device);
};
```

**Single-device optimization:** Returns success immediately when only one device is active (no collective needed).

### 10.2 BackendRouter

Location: `src/v2/collective/BackendRouter.h`

| Device Group | Backend | Latency |
|--------------|---------|---------|
| All CUDA | NCCL | ~5μs |
| All ROCm | RCCL | ~5μs |
| Mixed CUDA+ROCm (same node) | HOST | host-staged |
| Cross-node | MPI | ~10-50μs |
| CPU only | MPI | ~1-5μs |

### 10.3 ILocalTPContext

Location: `src/v2/collective/ILocalTPContext.h`

Interface for LOCAL tensor parallelism (multiple devices within a single MPI rank):
```cpp
class ILocalTPContext {
    virtual const vector<GlobalDeviceAddress>& devices() const = 0;
    virtual const vector<float>& weights() const = 0;
    virtual int degree() const = 0;
    virtual bool allreduce(TensorBase* tensor) = 0;
    virtual bool allgather(TensorBase* local, TensorBase* gathered) = 0;
    virtual bool allgatherV(TensorBase* local, TensorBase* gathered, const vector<int>& counts) = 0;
    virtual WorkRange headRange(int device_idx, int total_heads) const = 0;
    virtual WorkRange ffnRange(int device_idx, int total_d_ff) const = 0;
    virtual WorkRange vocabRange(int device_idx, int vocab_size) const = 0;
    virtual void barrier() = 0;
};
```

---

## 11. Weight Sharding and Tensor Parallelism

### 11.1 TensorParallelConfig

Location: `src/v2/config/TensorParallelConfig.h`

Per-device TP assignments with proportional splits:

```cpp
struct DeviceShardingAssignment {
    DeviceId device;
    int head_start, head_count;        // Q heads
    int kv_head_start, kv_head_count;  // KV heads (GQA)
    int d_ff_start, d_ff_count;        // FFN slice
    int vocab_start, vocab_count;      // LM head slice
    float work_fraction;               // e.g., 0.73
    int local_rank;
};
```

**Factory Methods:**
- `TensorParallelConfig::equalSplit(world_size, n_heads, n_kv_heads, d_ff, vocab)`
- `TensorParallelConfig::proportionalSplit(devices, fractions, n_heads, n_kv_heads, d_ff, vocab)`
- `TensorParallelConfig::singleDevice(device, n_heads, n_kv_heads, d_ff, vocab)`

### 11.2 Sharding Patterns

| Weight | Mode | Description |
|--------|------|-------------|
| `attn_q`, `attn_k`, `attn_v` | COLUMN_PARALLEL | Split output dim (heads) |
| `attn_output` (Wo) | ROW_PARALLEL | Split input dim, allreduce after |
| `ffn_gate`, `ffn_up` | COLUMN_PARALLEL | Split output dim (d_ff) |
| `ffn_down` | INPUT_PARALLEL | Split input dim, allreduce after |
| `output` (LM head) | COLUMN_PARALLEL | Split vocab, allgather logits |
| Norms, embeddings | REPLICATE | Full copy on each device |

### 11.3 WeightShardingConfig (Schema-Level)

Location: `src/v2/execution/local_execution/graph/GraphSchema.h`

Declarative weight sharding rules used by GraphResolver:
```cpp
struct WeightShardingConfig {
    vector<WeightShardingPattern> patterns;           // Regex-like patterns
    unordered_map<string, WeightShardingMode> exact_matches;
    WeightShardingMode default_mode = Replicate;
    
    WeightShardingMode getMode(const string& name) const;
    WeightDimensionType getDimensionType(const string& name) const;
};
```

---

## 12. Hybrid Parallelism

### 12.1 Three-Level Architecture

```
LEVEL 1: Cross-Rank Pipeline Parallelism (PP)
  • Layers distributed across MPI ranks
  • MPI P2P for activation forwarding
  • PipelineParallelConfig manages layer ranges

LEVEL 2: Intra-Rank Layer Placement
  • Layers assigned to CPU or GPU within a rank
  • LayerPlacementConfig tracks device per layer
  • TransitionPoints identify CPU↔GPU transfers

LEVEL 3: Heterogeneous Tensor Parallelism (TP)
  • Heads/FFN split across multiple GPUs (same rank)
  • Proportional splits for mixed vendors
  • AllGatherV for variable-sized output collection
```

### 12.2 PipelineConfig

Location: `src/v2/config/PipelineConfig.h`

PP+TP composition:
```cpp
struct PipelineConfig {
    vector<TPDomainConfig> tp_domains;      // {name, devices, backend}
    vector<PPStageConfig> pp_stages;        // {stage_id, domain, layers}
    map<pair<int,int>, CollectiveBackendType> pp_transfer_backends;
    int total_layers;
    
    const TPDomainConfig* getDomainForStage(int stage_id) const;
    const PPStageConfig* getStageForLayer(int layer_idx) const;
    DeviceId getDeviceForLayer(int layer_idx) const;
    
    // Factory methods
    static PipelineConfig singleDevice(int num_layers, DeviceId device);
    static PipelineConfig tensorParallel(int num_layers, const vector<DeviceId>&, CollectiveBackendType);
    static PipelineConfig pipelineParallel2Stage(int num_layers, DeviceId, int split, DeviceId, CollectiveBackendType);
};
```

### 12.3 PipelineParallelConfig

Location: `src/v2/config/PipelineParallelConfig.h`

Layer distribution across MPI ranks:
```cpp
class PipelineParallelConfig {
    static PipelineParallelConfig equalSplit(int num_ranks, int total_layers);
    static PipelineParallelConfig customSplit(const vector<pair<int,int>>& ranges);
    
    const LayerRange& forRank(int rank) const;
    int rankForLayer(int layer) const;
    bool isFirstStage(int rank) const;
    bool isLastStage(int rank) const;
    int prevRank(int rank) const;
    int nextRank(int rank) const;
};
```

### 12.4 LayerPlacementConfig

Location: `src/v2/config/LayerPlacementConfig.h`

Per-layer device assignment within a rank:
```cpp
class LayerPlacementConfig {
    static LayerPlacementConfig allOnDevice(DeviceId, int n_layers);
    static LayerPlacementConfig cpuFirstLayers(int cpu_layers, int total, DeviceId gpu);
    static LayerPlacementConfig cpuLastLayers(int cpu_layers, int total, DeviceId gpu);
    
    DeviceId deviceForLayer(int layer) const;
    vector<TransitionPoint> transitionPoints() const;
};
```

---

## 13. Device Management

### 13.1 GlobalDeviceAddress

Location: `src/v2/backends/GlobalDeviceAddress.h`

Fully-qualified device address: `hostname:numa:type:ordinal`
```cpp
struct GlobalDeviceAddress {
    string hostname;
    int numa_node;
    DeviceType device_type;
    int device_ordinal;
    
    static GlobalDeviceAddress parse(const string& spec);  // "cuda:0", "node1:0:cuda:0"
    string toString() const;
};
```

### 13.2 DeviceRegistry

Location: `src/v2/backends/DeviceRegistry.h`

Singleton device discovery and management:
```cpp
class DeviceRegistry : public IDeviceRegistry {
    static DeviceRegistry& instance();
    void discover();
    vector<GlobalDeviceAddress> allDevices() const;
    vector<GlobalDeviceAddress> devicesByType(DeviceType) const;
    size_t memoryCapacity(const GlobalDeviceAddress&) const;
    bool canP2P(const GlobalDeviceAddress&, const GlobalDeviceAddress&) const;
    IBackend* backendFor(const GlobalDeviceAddress&);
};
```

### 13.3 BackendManager

Location: `src/v2/backends/BackendManager.h`

Unified device memory interface:
```cpp
class BackendManager {
    IBackend* getBackendFor(DeviceId);
    IBackend* getCPUBackend(), getCUDABackend(), getROCmBackend();
};

// IBackend: allocate(), free(), deviceToHost(), hostToDevice(), synchronize()
```

---

## 14. Quick Reference

### 14.1 File Locations

| Component | Location |
|-----------|----------|
| **Config Pipeline** | |
| OrchestrationConfig | `src/v2/config/OrchestrationConfig.h` |
| OrchestrationConfigParser | `src/v2/config/OrchestrationConfigParser.h` |
| ConfigValidator | `src/v2/config/ConfigValidator.h` |
| RuntimeConfig | `src/v2/execution/config/RuntimeConfig.h` |
| RankExecutionPlan | `src/v2/execution/mpi_orchestration/RankExecutionPlan.h` |
| IExecutionPlanBuilder | `src/v2/execution/mpi_orchestration/IExecutionPlanBuilder.h` |
| GraphConfig (GraphTypes) | `src/v2/models/GraphTypes.h` |
| IGraphConfigBuilder | `src/v2/models/IGraphConfigBuilder.h` |
| TensorParallelConfig | `src/v2/config/TensorParallelConfig.h` |
| PipelineConfig | `src/v2/config/PipelineConfig.h` |
| PipelineParallelConfig | `src/v2/config/PipelineParallelConfig.h` |
| LayerPlacementConfig | `src/v2/config/LayerPlacementConfig.h` |
| **Runner Layer** | |
| IOrchestrationRunner | `src/v2/execution/runner/IOrchestrationRunner.h` |
| OrchestrationRunner | `src/v2/execution/runner/OrchestrationRunner.h` |
| IOrchestrationRunnerFactory | `src/v2/execution/runner/IOrchestrationRunnerFactory.h` |
| **Local Execution** | |
| IInferenceRunner | `src/v2/execution/local_execution/orchestrators/IInferenceRunner.h` |
| DeviceGraphOrchestrator | `src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h` |
| MultiDeviceOrchestrator | `src/v2/execution/local_execution/orchestrators/MultiDeviceOrchestrator.h` |
| MultiDomainOrchestrator | `src/v2/execution/local_execution/orchestrators/MultiDomainOrchestrator.h` |
| **Memory Management** | |
| BufferArena | `src/v2/memory/BufferArena.h` |
| BufferId | `src/v2/memory/BufferId.h` |
| CoherenceTracker | `src/v2/memory/CoherenceTracker.h` |
| StageBoundBuffers | `src/v2/memory/StageBoundBuffers.h` |
| StageBufferContract | `src/v2/memory/StageBufferContract.h` |
| BufferAccess | `src/v2/memory/BufferAccess.h` |
| NUMAAllocator | `src/v2/memory/NUMAAllocator.h` |
| **Transfer** | |
| TransferEngine | `src/v2/transfer/TransferEngine.h` |
| TransferMethod | `src/v2/transfer/TransferMethod.h` |
| **Coherence (Tensor)** | |
| CoherenceState | `src/v2/tensors/CoherenceState.h` |
| CoherenceAuditLog | `src/v2/tensors/CoherenceAuditLog.h` |
| **Graph System** | |
| GraphSchema | `src/v2/execution/local_execution/graph/GraphSchema.h` |
| GraphResolver | `src/v2/execution/local_execution/graph/GraphResolver.h` |
| DeviceGraphExecutor | `src/v2/execution/local_execution/graph/DeviceGraphExecutor.h` |
| IGraphBuilder | `src/v2/execution/local_execution/graph/IGraphBuilder.h` |
| DeviceGraphBufferManager | `src/v2/execution/local_execution/graph/DeviceGraphBufferManager.h` |
| LivenessAnalyzer | `src/v2/execution/local_execution/graph/LivenessAnalyzer.h` |
| **Model Execution** | |
| IModelExecutor | `src/v2/execution/local_execution/model/IModelExecutor.h` |
| ModelExecutor | `src/v2/execution/local_execution/model/ModelExecutor.h` |
| ILayerExecutor | `src/v2/execution/local_execution/model/ILayerExecutor.h` |
| **Graph Building** | |
| GraphBuilderRegistry | `src/v2/execution/local_execution/graph/GraphBuilderRegistry.h` |
| SchemaFactoryRegistry | `src/v2/execution/local_execution/graph/SchemaFactoryRegistry.h` |
| IGraphBuilder | `src/v2/execution/local_execution/graph/IGraphBuilder.h` |
| ModelRegistrations | `src/v2/models/ModelRegistrations.cpp` |
| Qwen2Graph | `src/v2/models/qwen/Qwen2Graph.h` |
| Qwen2Schema | `src/v2/models/qwen/Qwen2Schema.h` |
| Qwen3Schema | `src/v2/models/qwen3/Qwen3Schema.h` |
| Qwen2GraphConfigBuilder | `src/v2/models/qwen/Qwen2GraphConfigBuilder.h` |
| Qwen2BufferSpec | `src/v2/models/qwen/Qwen2BufferSpec.h` |
| ISchemaFactory | `src/v2/execution/local_execution/graph/GraphSchema.h` |
| **Compute Stages** | |
| IComputeStage | `src/v2/execution/compute_stages/IComputeStage.h` |
| Stage implementations | `src/v2/execution/compute_stages/stages/` |
| **Collective Layer** | |
| CollectiveContext | `src/v2/execution/local_execution/collective/CollectiveContext.h` |
| BackendRouter | `src/v2/collective/BackendRouter.h` |
| ILocalTPContext | `src/v2/collective/ILocalTPContext.h` |
| LocalTPContext | `src/v2/collective/LocalTPContext.h` |
| ICollectiveBackend | `src/v2/collective/ICollectiveBackend.h` |
| **Coherence (Execution)** | |
| StageCoherence | `src/v2/execution/local_execution/coherence/StageCoherence.h` |
| GpuCoherence | `src/v2/execution/local_execution/coherence/GpuCoherence.h` |
| **Kernel Layer** | |
| KernelFactory | `src/v2/kernels/KernelFactory.h` |
| CPU kernels | `src/v2/kernels/cpu/` |
| CUDA kernels | `src/v2/kernels/cuda/` |
| ROCm kernels | `src/v2/kernels/rocm/` |
| **Backend Layer** | |
| BackendManager | `src/v2/backends/BackendManager.h` |
| IBackend | `src/v2/backends/IBackend.h` |
| DeviceId | `src/v2/backends/DeviceId.h` |
| GlobalDeviceAddress | `src/v2/backends/GlobalDeviceAddress.h` |
| DeviceRegistry | `src/v2/backends/DeviceRegistry.h` |
| DeviceAddressAdapter | `src/v2/backends/DeviceAddressAdapter.h` |
| **Tensor Layer** | |
| ITensor | `src/v2/tensors/ITensor.h` |
| TensorClasses | `src/v2/tensors/TensorClasses.h` |
| TensorFactory | `src/v2/tensors/TensorFactory.h` |
| TensorLayout | `src/v2/tensors/TensorLayout.h` |
| TensorVerification | `src/v2/tensors/TensorVerification.h` |
| UnifiedKVCache | `src/v2/tensors/UnifiedKVCache.h` |
| **MPI Layer** | |
| MPIContext | `src/v2/utils/MPIContext.h` |
| MPITopology | `src/v2/utils/MPITopology.h` |
| MPITags | `src/v2/utils/MPITags.h` |
| **Loaders** | |
| WeightManager | `src/v2/loaders/WeightManager.h` |

### 14.2 Key Design Rules

1. **Use `IOrchestrationRunner`** for complex multi-device scenarios
2. **Use `IInferenceRunner`** for simple single-device inference
3. **Parse once, copy always** — RuntimeConfig is pre-parsed in ExecutionPlanBuilder, carried through the chain
4. **Keep MPI out of kernels** — MPI sync lives in `AllreduceStage` and `AllGatherStage`
5. **Model-agnostic graph building** — Register new models in `ModelRegistrations.cpp`; use `GraphBuilderRegistry` and `SchemaFactoryRegistry`
6. **Declarative graphs** — Build computation as DAGs via GraphSchema + GraphResolver
7. **Graph caching** — Reuse cached graphs in decode mode for performance
8. **Use BufferArena** — Centralized buffer management with coherence tracking
9. **Use StageBufferContract** — Stages declare I/O via `bufferContract()` for automatic coherence
10. **DeviceId not int** — Use typed `DeviceId` for device identification
11. **Collective via BackendRouter** — Let the router select optimal backend
12. **Implement getDumpInfo()** — All stages must support introspection
13. **Use OMP_WORKSHARE_REGION** — Nested-safe OpenMP parallelism in all kernels

### 14.3 Execution Path Summary

```
main()
  → IOrchestrationRunnerFactory::createFromArgs(argc, argv)
    → OrchestrationConfigParser::parseArgs() → OrchestrationConfig
    → OrchestrationRunner::initialize()
      → ExecutionPlanBuilder → RankExecutionPlan (RuntimeConfig parsed once)
      → loadWeights()
      → buildComputeGraph()
        ├── Single: DeviceGraphOrchestrator (IGraphBuilder via registry → ComputeGraph → DeviceGraphExecutor)
        ├── TP: MultiDeviceOrchestrator (N DeviceGraphOrchestrators, TPWorkerPool)
        └── PP: TreeToRunnerCompiler (N DeviceGraphOrchestrators, PPContext)
  → runner.prefill(tokens)
    → forward(tokens, seq_len)
      → GraphSchema → GraphResolver → ResolvedGraphSpec → ComputeGraph
      → DeviceGraphExecutor::execute(graph)  [StageRunPolicy::full()]
        → For each node: contract → cohere → verify → execute → mark dirty → verify → snapshot
  → runner.decodeStep()
    → forward(tokens, 1)  [cached graphs reused, StageRunPolicy::fastDecode()]
```

### 14.4 Directory Structure

```
src/v2/
├── config/                          # Configuration (OrchestrationConfig, TP/PP/Placement configs)
├── execution/
│   ├── runner/                      # Tier 1: IOrchestrationRunner, OrchestrationRunner
│   ├── mpi_orchestration/           # Tier 2: RankExecutionPlan, ExecutionPlanBuilder
│   ├── config/                      # RuntimeConfig, ExecutionPolicy
│   ├── local_execution/
│   │   ├── orchestrators/           # Tier 3: DeviceGraphOrchestrator, MultiDeviceOrchestrator
│   │   ├── model/                   # IModelExecutor, ModelExecutor, ILayerExecutor
│   │   ├── graph/                   # GraphSchema, GraphResolver, DeviceGraphExecutor, ComputeGraph,
│   │   │                            #   GraphBuilderRegistry, SchemaFactoryRegistry, IGraphBuilder
│   │   ├── coherence/               # StageCoherence, GpuCoherence (legacy + test helpers)
│   │   ├── collective/              # CollectiveContext
│   │   └── device/                  # DeviceContext, DeviceWorkspaceManager
│   ├── factory/                     # InferenceRunnerFactory (internal)
│   └── compute_stages/              # IComputeStage, all stage implementations
├── memory/                          # BufferArena, BufferId, CoherenceTracker, StageBoundBuffers,
│                                    #   StageBufferContract, NUMAAllocator
├── transfer/                        # TransferEngine, TransferMethod
├── models/
│   ├── ModelRegistrations.cpp/.h    # Central model registration (add new models here)
│   ├── qwen/                        # Qwen2Graph, Qwen2Schema, Qwen2GraphConfigBuilder, Qwen2BufferSpec
│   ├── qwen3/                       # Qwen3Schema (QK-norm variant of Qwen2)
│   ├── GraphTypes.h                 # GraphConfig
│   └── IGraphConfigBuilder.h        # Interface
├── kernels/
│   ├── cpu/                         # CPU kernels (GEMM, attention, JIT, primitives)
│   ├── cuda/                        # CUDA kernels
│   ├── rocm/                        # ROCm kernels
│   └── KernelFactory.h              # Centralized dispatch
├── tensors/                         # ITensor, TensorBase, all typed tensors, KV cache,
│                                    #   CoherenceState, CoherenceAuditLog
├── backends/                        # IBackend, BackendManager, DeviceRegistry, DeviceId
├── collective/                      # ILocalTPContext, BackendRouter, HOST, NCCL, RCCL
├── loaders/                         # GGUF loading, WeightManager
├── app/                             # Application modes (ChatCompletionHandler, etc.)
└── utils/                           # MPIContext, MPITopology, Tokenizer, Sampler, logging
```
