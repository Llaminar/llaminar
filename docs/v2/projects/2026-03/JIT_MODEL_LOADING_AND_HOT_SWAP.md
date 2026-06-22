# JIT Model Loading and Hot-Swap Design

**Date**: 2025-03-05  
**Status**: Research Complete, Implementation Planned  
**Depends On**: [Main.cpp Refactor Plan](MAIN_CPP_REFACTOR_PLAN.md), [OpenAI HTTP Server Design](OPENAI_HTTP_SERVER_DESIGN.md)

---

## Overview

This document analyzes what is required to support **JIT (just-in-time) model loading** and **runtime model hot-swap** in Llaminar's HTTP server mode. The target use case: a REST request arrives for a model that isn't presently loaded but could be. The server loads it on demand, serves the request, and manages multiple models with GPU memory eviction.

---

## Current Architecture: One Model, One Process

Today, the lifecycle is strictly **linear**:

```
parse config → build plan → load weights → build graph → serve → shutdown → exit
```

Everything is per-process. `OrchestrationRunner` owns the full stack (`model_ctx_` → `weight_manager_` → tensors → GPU buffers → KV caches → compute graphs) and `shutdown()` tears it all down via RAII cascades.

---

## Resource Ownership Analysis

### OrchestrationRunner::initialize() — Allocation Steps

| Step | Method | What It Allocates |
|------|--------|-------------------|
| 1 | `initializeMPI()` | `mpi_ctx_` — shared_ptr to `MPIContext` (or gets global singleton) |
| 2 | `buildExecutionPlan()` | `plan_` — `RankExecutionPlan` (value type, no heap) |
| 3 | `setupLocalTPContext()` | `local_tp_ctx_` — unique_ptr to `ILocalTPContext` (NCCL/RCCL comms, BAR tensors, device group) |
| 3.5 | `setupLocalPPContext()` | `local_pp_ctx_` — unique_ptr to `ILocalPPContext` |
| 4 | `loadWeights()` | `model_ctx_` — shared_ptr to `ModelContext` (mmap + weight cache); also creates `tokenizer_` |
| 5 | `validateTPPPConfiguration()` | None (validation only) |
| 6 | `buildComputeGraph()` | `runner_` — unique_ptr to `IInferenceRunner` (KV cache, activation buffers, compute graphs, device contexts) |

### OrchestrationRunner::shutdown() — Deallocation

```cpp
runner_.reset();         // Destroys IInferenceRunner (KV cache, graph caches, activation buffers, GPU workspace)
local_pp_ctx_.reset();   // Destroys PP context
local_tp_ctx_.reset();   // Destroys TP context (NCCL comms, BAR registrations)
model_ctx_.reset();      // Destroys ModelContext → WeightManager cache → weight tensors → GPU memory + mmap
```

### ModelContext & ModelLoader — Weight Memory

**ModelContext** owns:
- `ModelLoader loader_` (by value) — holds the GGUF file handle and mmap mapping
- `shared_ptr<WeightManager> weight_manager_` — caches loaded weight tensors

**ModelLoader** owns:
- `ifstream file_stream_` + `vector<ifstream> split_streams_` — GGUF file handles
- `shared_ptr<MmapRegion> mmap_region_` — the mmap'd model file (entire GGUF, MAP_PRIVATE|MAP_POPULATE on Linux)
- `vector<shared_ptr<MmapRegion>> split_mmap_regions_` — for multi-part GGUF files
- `GGUFModel model_` — parsed metadata (in-memory, relatively small)

**WeightManager** owns:
- `unordered_map` cache of `shared_ptr<TensorBase>` keyed by `(name, DeviceId)`
- For TP: per-device clones of weight tensors (each with independent coherence state)
- Supports `clearCache()` to free all cached weights

### DeviceGraphOrchestrator — Per-Model State

| Category | Members | Memory Type |
|----------|---------|-------------|
| **KV Cache** | `state_.kv_cache` + `state_.pp_kv_caches` | CPU RAM or GPU VRAM (large — `n_layers × max_seq_len × n_kv_heads × head_dim × dtype`) |
| **Activation Buffers** | `state_.hidden`, `state_.logits`, `state_.Q/K/V`, `state_.attn_output`, `state_.gate/up/ffn_output` | CPU or GPU |
| **Graph Caches** | `layer_graph_cache_` (per layer) + `forward_graph_cache_` (keyed by signature) | CPU heap |
| **Device Context Cache** | `device_contexts_` (map of `IDeviceContext` per DeviceId) | GPU handles (cuBLAS, hipBLAS, streams) |
| **Buffer Manager** | `buffer_manager_` + `owned_buffers_` | CPU/GPU |
| **Weight References** | `weight_manager_`, `weight_placement_map_` (shared_ptrs) | Shared with ModelContext |
| **PP/TP Contexts** | `pp_contexts_`, `domain_tp_contexts_` | Communication state |
| **Weight Streamer** | `weight_streamer_` (shared_ptr) | GPU cache for streaming weights |

### TensorBase — Memory Ownership

Each `TensorBase` can own up to **six kinds of memory**:

| Memory | Lifecycle |
|--------|-----------|
| **Host data** | Owned via concrete class (`std::vector<float>`, `AlignedVector<Q8_1Block>`, etc.), freed on destruction |
| **GPU data** | `gpu_data_ptr_` allocated via `IBackend::allocate()`, freed in `~TensorBase()` via `backend->free()` |
| **Mmap zero-copy** | `shared_ptr<MmapRegion>` — stays alive until all tensor refs + ModelLoader are destroyed |
| **BAR-backed** | PCIe BAR memory for cross-vendor P2P, freed via `freeBARBackedMemory()` |
| **Secondary device buffers** | For multi-device transfers, freed in `~TensorBase()` |
| **Pinned host memory** | Unpinned in `~TensorBase()` |

### Global State / Singletons

| Global | Type | Hot-Swap Impact |
|--------|------|-----------------|
| **`DeviceManager::instance()`** | Meyers singleton | **Safe** — hardware inventory, model-independent, no reset needed |
| **`GPUDeviceContextPool::instance()`** | Meyers singleton | **Safe** — cuBLAS/hipBLAS handles per device, reusable across models |
| **`KernelFactory` static caches** | Static maps | **BLOCKER** — caches keyed by raw `TensorBase*` pointers. After unloading model A, pointer values may be recycled by model B. Must call `KernelFactory::clearCache()` between model swaps. Also holds `prepared_gemm_registry_` and `device_gemm_engine_registry_` with weight-specific GPU handles. |
| **Backend singletons** (`getCUDABackend()`, `getROCmBackend()`, `getCPUBackend()`) | `call_once` globals | **Safe** — stateless allocator interfaces |
| **`MPIContextFactory::global()`** | Shared global | **Safe** — process-level, not model-specific |

---

## Per-Resource Hot-Swap Safety Summary

| Resource | Per-Model? | Released by `shutdown()`? | Hot-Swap Safe? |
|----------|-----------|--------------------------|---------------|
| Model mmap (GGUF file) | Yes | Yes — `shared_ptr<MmapRegion>` | Clean — RAII |
| Weight tensors (CPU) | Yes | Yes — `WeightManager` cache | Clean |
| Weight tensors (GPU) | Yes | Yes — `TensorBase::~TensorBase()` frees via backend | Clean |
| KV cache | Yes | Yes — `IKVCache` unique_ptr | Clean |
| Activation buffers | Yes | Yes — `InferenceState` shared_ptrs | Clean |
| Compute graph caches | Yes | Yes — within runner | Clean |
| TP/PP collective contexts | Yes | Yes — unique_ptr in runner | Clean |
| KernelFactory static caches | **Global** | **No** | **Must clear between swaps** |
| DeviceManager | Global singleton | No (reused) | Safe — hardware inventory |
| GPU device context pool | Global singleton | No (reused) | Safe — model-independent |
| MPI context | Per-process | No (shared) | Safe — all models share |

**Key finding**: The RAII ownership model is already sound — `shutdown()` cleanly releases everything per-model. The **only global-state fix** needed is `KernelFactory` cache scoping.

---

## Required Changes

### 1. Model Registry (New Component)

A `ModelRegistry` that maps model names to GGUF paths, manages the lifecycle of multiple `OrchestrationRunner` instances, and handles load/unload/swap coordination:

```cpp
class ModelRegistry {
public:
    explicit ModelRegistry(const ServerConfig& server_config);

    /// Register a model path (available but not loaded)
    void registerModel(const std::string& model_id, const std::string& gguf_path);

    /// Scan a directory for .gguf files and register all
    void scanModelDirectory(const std::string& dir_path);

    /// Get a loaded runner (JIT-loads if not loaded, evicts LRU if needed)
    /// Thread-safe. Blocks during load.
    IOrchestrationRunner* getOrLoad(const std::string& model_id);

    /// Explicitly preload a model
    bool preload(const std::string& model_id);

    /// Explicitly unload a model (frees all resources)
    bool unload(const std::string& model_id);

    /// List all known models (loaded + available)
    std::vector<ModelInfo> listModels() const;

    /// Total GPU memory used by all loaded models
    size_t totalGpuMemoryUsed() const;

private:
    struct LoadedModel {
        std::unique_ptr<IOrchestrationRunner> runner;
        std::string gguf_path;
        std::chrono::steady_clock::time_point last_used;
        size_t estimated_gpu_bytes;
    };

    bool evictLRU(size_t bytes_needed);

    std::unordered_map<std::string, LoadedModel> loaded_models_;
    std::unordered_map<std::string, std::string> available_models_;  // id → path
    mutable std::shared_mutex mutex_;
    ServerConfig server_config_;
};
```

The server would scan a model directory (e.g., `--model-dir ./models/`) at startup and register all `.gguf` files as available-but-not-loaded.

### 2. KernelFactory Cache Scoping — The One Real Blocker

`KernelFactory` uses static caches keyed by raw `TensorBase*` pointers. After unloading model A, those pointer values may be recycled by model B's tensors, returning stale kernels with wrong weight data.

**Minimum fix**: Call `KernelFactory::clearCache()` between model swaps.

**Proper fix**: Make the cache **model-aware** — either:
- **Scoped per-runner** (move from static to instance member on `DeviceGraphOrchestrator`), or
- **Keyed by stable identity** (model_id + tensor_name + device_id) rather than raw pointer

The per-runner approach is cleaner and allows concurrent models without cache thrashing.

### 3. OrchestrationConfig Split

Today, `OrchestrationConfig` mixes process-level settings (MPI, device topology, listen port) with model-level settings (model path, quant type, max_seq_len). These need to be separated:

```cpp
// Process-level (singleton, set once at startup)
struct ServerConfig {
    std::string host = "0.0.0.0";
    int port = 8080;
    std::string api_key;
    int queue_size = 64;
    std::string model_dir;          // Directory to scan for models
    size_t max_gpu_memory_mb = 0;   // 0 = use all available

    // Device topology, MPI settings (fixed for process lifetime)
    DeviceAssignmentMode device_mode;
    int tp_degree;
    TPScope tp_scope;
    CollectiveBackendType default_backend;
    // ... (all topology/MPI fields from OrchestrationConfig)
};

// Model-level (one per loaded model)
struct ModelConfig {
    std::string model_path;
    int max_seq_len = 2048;
    bool use_mmap = true;
    std::string activation_precision = "fp32";
    std::string kv_cache_precision = "auto";
    bool use_fused_attention = false;
    FusedAttentionBackend fused_attention_backend = FusedAttentionBackend::JIT;
    // Per-model TP/PP overrides if different from server defaults
};
```

`OrchestrationConfig` can be reconstructed from `ServerConfig + ModelConfig` when creating a new runner.

### 4. GPU Memory Budget Management

Today, a model greedily allocates GPU memory for weights + KV cache + activations. With multiple models:

- **Memory accounting**: Track per-model GPU memory footprint before committing to a load
- **Pre-flight estimation**: Estimate GPU memory from GGUF metadata (layer count × parameter shapes × quant type) before loading
- **Eviction policy**: When GPU memory is insufficient, unload the LRU model first
- **Fail-fast**: If the new model can't fit even after evicting everything, return `HTTP 507 Insufficient Storage`

**Estimation formula** (approximate):
```
weight_bytes = sum(tensor_shape × quant_bytes_per_element) for all layers
kv_cache_bytes = 2 × n_layers × max_seq_len × n_kv_heads × head_dim × kv_dtype_size
activation_bytes ≈ 2 × max_seq_len × d_model × sizeof(float)  // rough upper bound
total ≈ weight_bytes + kv_cache_bytes + activation_bytes + workspace_overhead
```

This can be computed from GGUF metadata without loading weights.

### 5. MPI Rank Coordination Protocol Extension

The [HTTP server design](OPENAI_HTTP_SERVER_DESIGN.md) defines a command broadcast protocol (PREFILL, DECODE, CLEAR_CACHE, SHUTDOWN). Hot-swap adds two new commands:

```cpp
enum class CommandType : int32_t {
    PREFILL,
    DECODE_STEP,
    CLEAR_CACHE,
    SHUTDOWN,
    LOAD_MODEL,      // NEW
    UNLOAD_MODEL,     // NEW
    SWITCH_MODEL,     // NEW: set active model for subsequent PREFILL/DECODE
};
```

All ranks must load/unload in lockstep because weights are sharded across TP ranks and PP stages. The sequence:

1. Rank 0 broadcasts `LOAD_MODEL { model_path, config_overrides }`
2. Every rank creates its own `OrchestrationRunner`, builds plan, loads its weight shard
3. Every rank broadcasts ready/error status
4. On success, model becomes available for inference
5. Before inference, Rank 0 broadcasts `SWITCH_MODEL { model_id }` so all ranks know which runner to use

### 6. Request Routing Changes

The `InferenceWorker` in the HTTP server design currently assumes a single runner. With hot-swap:

```cpp
void InferenceWorker::processRequest(InferenceRequest& req) {
    // Resolve model → runner (JIT load if needed)
    auto* runner = model_registry_.getOrLoad(req.model);
    if (!runner) {
        req.response->error("Model '" + req.model + "' not found or failed to load");
        return;
    }

    // If active model changed, broadcast SWITCH_MODEL to all ranks
    if (runner != active_runner_) {
        broadcastSwitchModel(req.model);
        active_runner_ = runner;
    }

    // Proceed with inference
    runner->clearCache();
    runner->prefill(tokens);
    // ...decode loop...
}
```

---

## Loading Latency Analysis

Model loading is **slow** (seconds to tens of seconds for large models: mmap + weight sharding + graph build + GPU upload). For a REST API, this translates to unacceptable first-request latency:

| Model Size | Estimated Load Time | Bottleneck |
|-----------|-------------------|------------|
| 0.5B (Q4_0) | 1-3s | Graph build + GPU upload |
| 7B (Q4_0) | 5-10s | Mmap + GPU upload |
| 70B (Q4_0) | 30-60s | Mmap + multi-GPU sharding |

### Mitigation Strategies

| Strategy | First-Request Latency | Memory Cost |
|----------|----------------------|-------------|
| **Preload at startup** (current) | Zero | Full memory committed |
| **Eager background load** | Near-zero (if load wins the race) | Predictive — may waste memory |
| **JIT load on first request** | 5-30s first hit, then cached | On-demand — minimal waste |
| **Hybrid**: preload primary + JIT for secondary | Primary = instant, secondary = slow first hit | Balanced |

**Recommended approach**: **Hybrid**. Preload the model specified on the command line (or the first in the model directory). JIT-load others on demand. While loading, the HTTP handler can either:

- **Block** with a long timeout and return the result once loaded
- **Return `HTTP 202 Accepted`** with a `Retry-After: 10` header, letting the client poll
- **Return `HTTP 503`** with `"message": "Model loading, try again in ~10s"` for simpler clients

---

## Concurrent Model Serving (Future)

Full concurrent multi-model serving (different requests hitting different models without unloading) would require:

1. **Per-model KernelFactory caches** (not static — move to runner instance)
2. **GPU memory partitioning**: Pre-allocate budgets per model or use a shared pool with reservations
3. **MPI-level model multiplexing**: Non-rank-0 processes must know which model's collectives to participate in for each request. The `SWITCH_MODEL` command handles this.
4. **Independent KV caches**: Already per-runner, so this is free.

This is a natural extension of the single-model swap design — the same components, just without evicting the previous model.

---

## Implementation Phases

### Phase A: KernelFactory Fix (Prerequisite)

1. Add `KernelFactory::clearCache()` call in `OrchestrationRunner::shutdown()`
2. Long-term: Refactor `KernelFactory` caches to be keyed by `(model_id, tensor_name, device)` instead of raw `TensorBase*`

### Phase B: Config Split

1. Extract `ServerConfig` from `OrchestrationConfig`
2. Extract `ModelConfig` from `OrchestrationConfig`
3. Add `OrchestrationConfig::fromServerAndModel(ServerConfig, ModelConfig)` factory
4. Migrate `ServerMode` and `AppLifecycle` to use split configs

### Phase C: ModelRegistry

1. Implement `ModelRegistry` with `registerModel()`, `getOrLoad()`, `unload()`, `listModels()`
2. Add `--model-dir` CLI flag and model directory scanning
3. GPU memory estimation from GGUF metadata (pre-flight check)
4. LRU eviction policy

### Phase D: MPI Protocol Extension

1. Add `LOAD_MODEL`, `UNLOAD_MODEL`, `SWITCH_MODEL` commands
2. Implement distributed load coordination (all ranks load/report in lockstep)
3. Implement distributed unload coordination
4. Follower loop extension on non-rank-0

### Phase E: HTTP Integration

1. Update `InferenceWorker` to use `ModelRegistry` instead of single runner
2. Update model endpoint (`GET /v1/models`) to return all registered models with load status
3. Add `POST /v1/models/{model_id}/load` admin endpoint (optional preload trigger)
4. Add `POST /v1/models/{model_id}/unload` admin endpoint
5. Implement 202/503 responses during JIT loading

### Phase F: Concurrent Multi-Model (Future)

1. Scope `KernelFactory` caches per-runner
2. GPU memory budgeting with reservations
3. Remove LRU eviction (keep all loaded models resident)
4. Performance testing with model switching overhead

---

## Effort Estimate

| Work Item | Effort | Risk |
|-----------|--------|------|
| `KernelFactory` cache clear on shutdown | Small | Low |
| `KernelFactory` cache scoping (per-runner) | Medium | Medium — must not break performance |
| Config split (`ServerConfig` + `ModelConfig`) | Medium | Low — refactor only |
| `ModelRegistry` (load/unload/eviction) | Large | Medium — GPU memory accounting |
| MPI command protocol extension (LOAD/UNLOAD/SWITCH) | Medium | High — distributed coordination |
| `InferenceWorker` multi-model routing | Small | Low — lookup only |
| GPU memory pre-flight estimation | Medium | Medium — quant-dependent sizing |
| Background loading with progress tracking | Medium | Low |
| Model directory scanning + auto-registration | Small | Low |
| HTTP admin endpoints (load/unload) | Small | Low |

**Total estimated effort**: ~3-4 weeks of focused work, with the MPI coordination being the highest-risk item.

---

## Key Findings

1. **The RAII ownership model is already sound.** `OrchestrationRunner::shutdown()` cleanly releases everything per-model via cascading `unique_ptr::reset()` and `shared_ptr` reference counting. No manual cleanup needed.

2. **The only global-state blocker is `KernelFactory`.** Its static caches are keyed by raw `TensorBase*` pointers, which become dangling after model unload. Fix: call `clearCache()` on swap (short-term) or scope caches per-runner (long-term).

3. **Mmap lifecycle is correct.** `MmapRegion` is held as `shared_ptr` in both `ModelLoader` and individual tensors. The mmap stays alive as long as any tensor references it.

4. **GPU memory is per-tensor.** Each `TensorBase` frees its own `gpu_data_ptr_` in its destructor via the backend allocator. No global GPU memory pool to drain.

5. **The hard parts are distributed coordination** (all MPI ranks must load/unload in lockstep) **and GPU memory budgeting** (knowing whether a model will fit before attempting to load it).
