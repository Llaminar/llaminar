# MoE Expert Weights in PreparedWeightStore — Design

**Date**: 2026-05-03  
**Status**: RFC  
**Parent**: `MODEL_WEIGHT_LIFETIME_REDESIGN_PLAN.md` (Phase 5/7)  
**Scope**: Moving MoE expert GEMM weight ownership from KernelFactory static registries into PreparedWeightStore, and enabling RebalancedExpertReplica bindings.

---

## 1. Problem Statement

MoE expert GEMM weights currently live in two overlapping ownership systems:

| System | What it owns | Lifetime |
|--------|-------------|----------|
| `KernelFactory::prepared_gemm_registry_` | Per-expert `PreparedGemmHandle` (VNNI-repacked buffers) | Static / process-wide |
| `MoEExpertComputeStage::Params::moe_owned_kernels` | `shared_ptr<ITensorGemm>` for GPU engines | Stage-local, rebuilt per graph |
| `MoEExpertComputeStage::Params::moe_packed_*_lifetime` | `shared_ptr<void>` GPU VRAM pool lifetimes | Stage-local |
| `MoEExpertWeightService` | Manages create/destroy/serialize but doesn't own | Stateless service |

This causes problems:

1. **`~TensorBase()` linear-scans the global registry**: Qwen3.5-35B has 768 expert weight entries (256 experts × 3 weight groups). Each expert view destruction scans `sliced_cache_` + `fused_gate_up_cache_` under global `cache_mutex_` — 57ms/token overhead (fixed by `cache_.has_value()` guard, but the structural problem remains).

2. **Dual-key hack**: After host data release, `raw_data()` returns nullptr so registry lookups fail. We added a tensor-pointer alias key as a workaround. MoE amplifies this 768× per model.

3. **No model-scoped cleanup**: If you load model A (256 experts), then model B, old expert entries persist in the global registry until the tensor views are destroyed. There's no "release all experts for model X" operation.

4. **Dynamic rebalance operates outside the weight binding model**: `registerAndPrepareNewExperts()` creates new `PreparedGemmHandle` entries in KernelFactory and hands back raw `ITensorGemm*` pointers. These dynamically-created entries have no `WeightIdentity`, no `WeightBinding`, and no traceability.

5. **ExpertReplicaSet has no weight-level representation**: When a socket replicates an expert, both sockets get GEMM engines but the weight lifecycle plan doesn't know about it. A replicated expert is invisible to the audit and cleanup systems.

---

## 2. Current Data Flow

```
GGUF 3D tensor [cols, rows_per_expert, num_experts]
     │
     ▼ WeightManager (Replicate mode — full 3D on each device)
     │
     ▼ GraphConfigBuilder → params.gate_exps (TensorBase* to 3D)
     │
     ▼ MoEExpertWeightService::extractExpertViews()
     │   → params.expert_gate_views[e] = shared_ptr<TensorBase> (2D view)
     │
    ▼ MoEExpertWeightService::prepareGemmEngines()
    │   ├── CPU: PreparedWeightStore / local expert GEMM preparation
    │   │        → store-owned prepared handle or expert slab entry
    │   │        → ITensorGemm* stored in params.prepared_gate_gemm[e]
     │   │
     │   └── GPU: LoadOrchestrator pipeline → moe_owned_kernels + packed_lifetime
     │
     ▼ MoEExpertComputeStage::execute()
         └── ensureGemmEnginesCached() copies to cached_*_gemm_
```

**Dynamic rebalance flow (decode)**:
```
DecodeExpertHistogram → MoERebalanceController::proposeReplicas()
     │
     ▼ OrchestrationRunner::setExpertReplicaSet()
     │   → DeviceGraphOrchestrator → MoEExpertComputeStage params
     │
     ▼ MoEExpertWeightService::registerAndPrepareNewExperts(new_expert_ids)
     │   ├── Check KernelFactory cache hit for each expert
     │   ├── Miss? → LoadOrchestrator pipeline (serialize→transfer→repack)
     │   └── Store ITensorGemm* in params.prepared_*_gemm[e]
     │
     ▼ ExpertReplicaSet::assignForToken() — deterministic dispatch
```

---

## 3. Target Architecture

### 3.1 PreparedWeightStore Gains Expert-Aware API

```cpp
class PreparedWeightStore {
public:
    // === Existing API ===
    PreparedWeightRef prepareGemm(const WeightBinding& binding);
    ITensorGemm* gemmKernel(const PreparedWeightRef& ref) const;
    std::optional<PreparedWeightRef> preparedRefForBinding(uint64_t binding_id, DeviceId device) const;

    // === New: MoE Expert Slab API ===

    /// Register a full set of expert engines for one weight group at one layer.
    /// Typically called during initial graph preparation (256 experts × 3 groups × N layers).
    ExpertSlabRef registerExpertSlab(const ExpertSlabDescriptor& desc);

    /// Get the GEMM engine for a specific expert within a slab.
    ITensorGemm* expertGemmKernel(const ExpertSlabRef& slab, int expert_id) const;

    /// Fused gate+up kernel for an expert (keys on the two per-expert handles).
    ITensorFusedGateUpGemm* expertFusedGateUpKernel(
        const ExpertSlabRef& gate_slab,
        const ExpertSlabRef& up_slab,
        int expert_id) const;

    /// Dynamic rebalance: register newly-arrived expert engines (from transfer or repack).
    /// Returns the expert IDs that were actually new (not cache hits).
    std::vector<int> registerArrivedExperts(
        const ExpertSlabRef& slab,
        const std::vector<ExpertArrival>& arrivals);

    /// Dynamic rebalance: release departed expert engines (free VRAM/memory).
    void releaseDepartedExperts(
        const ExpertSlabRef& slab,
        const std::vector<int>& expert_ids);

    /// Query: which experts in this slab have prepared engines?
    std::vector<bool> expertAvailabilityMask(const ExpertSlabRef& slab) const;

    /// Cleanup: release entire slab (model unload).
    void releaseExpertSlab(const ExpertSlabRef& slab);
};
```

### 3.2 New Types

```cpp
/// Identifies a "slab" of expert GEMM weights for one weight group × one layer.
struct ExpertSlabRef {
    ModelContextId model_id;
    uint64_t slab_id;           // Unique within model
    int layer_idx;
    WeightRole role;            // MoEExpertGate, MoEExpertUp, MoEExpertDown
    DeviceId device;
};

/// Descriptor used to register a slab.
struct ExpertSlabDescriptor {
    int layer_idx;
    WeightRole role;
    DeviceId device;
    int num_experts;            // Total expert slots (e.g., 256)
    int local_expert_start;    // First expert this slab covers
    int local_expert_count;    // How many experts are populated initially
    size_t rows_per_expert;
    size_t cols_per_expert;

    // Source identity for traceability
    WeightIdentity source_identity;  // Identity of the 3D parent tensor
};

/// Describes one expert arriving (from rebalance transfer or initial load).
struct ExpertArrival {
    int expert_id;
    ITensorGemm* engine;                           // Prepared GEMM engine
    std::shared_ptr<ITensorGemm> engine_lifetime;  // Ownership
    std::shared_ptr<TensorBase> view_lifetime;     // 2D view ownership (may be null for GPU)
    WeightDerivationKind derivation;               // ExpertSlice or RebalancedExpertReplica
    std::optional<DeviceId> source_device;         // Where it came from (for replicas)
};
```

### 3.3 Internal Storage

```cpp
// Inside PreparedWeightStore:
struct ExpertSlabEntry {
    ExpertSlabDescriptor descriptor;
    ExpertSlabRef ref;

    // Per-expert state [indexed by expert_id - local_expert_start]
    struct ExpertEntry {
        ITensorGemm* engine = nullptr;
        std::shared_ptr<ITensorGemm> engine_lifetime;
        std::shared_ptr<TensorBase> view_lifetime;
        WeightDerivationKind derivation = WeightDerivationKind::ExpertSlice;
        std::optional<DeviceId> source_device;  // Non-null for replicas
        bool available = false;
    };
    std::vector<ExpertEntry> experts;  // [num_experts]

    // Fused gate+up cache (keyed by expert_id)
    std::unordered_map<int, std::unique_ptr<ITensorFusedGateUpGemm>> fused_cache;

    // GPU lifetime pools (shared across all experts in slab)
    std::shared_ptr<void> packed_lifetime;
};

std::unordered_map<uint64_t, ExpertSlabEntry> expert_slabs_;  // keyed by slab_id
```

---

## 4. RebalancedExpertReplica Design

### 4.1 Concept

A `RebalancedExpertReplica` is a prepared GEMM engine for an expert that was **replicated** to a second device (typically another CPU socket or GPU) based on runtime usage histograms. It differs from the original `ExpertSlice` binding in:

| Property | ExpertSlice (original) | RebalancedExpertReplica |
|----------|----------------------|--------------------------|
| Source | 3D GGUF tensor | Another device's prepared state |
| Lifetime | Graph lifetime (frozen) | Dynamic — can be evicted |
| Identity | `{model, layer, expert, device}` | Same + `source_device` |
| When created | Graph build / initial load | Runtime (during decode) |
| Who creates | `MoEExpertWeightService::prepareGemmEngines()` | `MoEExpertWeightService::registerAndPrepareNewExperts()` |
| Evictable? | No | Yes (via `releaseDepartedExperts()`) |

### 4.2 Lifecycle

```
Step 1: Histogram Accumulation
    MoEExpertComputeStage::execute() → DecodeExpertHistogram::record()
    (Zero-allocation sliding window counter)

Step 2: Rebalance Proposal
    MoERebalanceController::proposeReplicas(max_per_socket)
    → ExpertReplicaSet { is_replicated[e], owner_socket[e] }
    (Runs every N tokens or on explicit trigger)

Step 3: Weight Preparation
    MoEExpertWeightService::registerAndPrepareNewExperts(new_expert_ids)
    ├── For each new expert:
    │   ├── Check PreparedWeightStore cache (expertGemmKernel returns non-null?)
    │   ├── Cache hit → skip
    │   └── Cache miss:
    │       ├── CPU: PreparedWeightStore expert slab lookup or local expert GEMM preparation
    │       │   (view/payload exists according to the binding host-retention policy)
    │       └── GPU: LoadOrchestrator pipeline from blob/payload/raw
    │
    └── PreparedWeightStore::registerArrivedExperts(slab, arrivals)
        where each arrival has derivation = RebalancedExpertReplica
        and source_device = the original owner socket/device

Step 4: Dispatch Registration
    OrchestrationRunner::setExpertReplicaSet(replicas, socket_id)
    → MoEExpertComputeStage → params.replica_set = replicas
    (Both sockets now have engines; per-token dispatch is deterministic)

Step 5: Execution
    MoEExpertComputeStage::execute()
    ├── Prefill: Uses replica_set.prefill_mask (prebuilt at step 4)
    └── Decode: replica_set.assignForToken() → compute_here[k]

Step 6: Eviction (optional — when histograms shift)
    MoERebalanceController notices expert usage dropped below threshold
    → releaseDepartedExperts(slab, [expert_ids])
    → PreparedWeightStore clears engines, frees memory
    → ExpertReplicaSet is updated (is_replicated[e] = false)
```

### 4.3 Source Data for Replicas

Three source tiers (same priority as existing `registerAndPrepareNewExpertsGPU()`):

| Priority | Source | When available |
|----------|--------|----------------|
| 1 | **MPI blob** (cross-rank transfer) | NodeLocalTP/GlobalTP with expert migration |
| 2 | **PayloadProvider** (host-side packed) | After initial CPU prep, before raw release |
| 3 | **Raw GGUF view** (`view->raw_data()`) | Before `MADV_DONTNEED` / host release |

**Key insight**: For **CPU multi-socket** rebalancing (the most common case), priority 3 is always available because `LLAMINAR_MOE_RELEASE_RAW_WEIGHTS=false` by default, and CPU expert weights are VNNI-repacked from the raw view. The raw 3D parent tensor is kept alive by `params.gate_exps` (the 3D `TensorBase*`).

For GPU expert rebalancing, priority 2 (`ExpertWeightPayloadProvider`) is the fast path — it provides pre-staged host data without re-reading from mmap.

### 4.4 Identity and Traceability

A replica binding has:
```cpp
WeightIdentity {
    .model_id = model_42,
    .logical_id = 7,                    // Same as source expert
    .instance_id = unique_new_id,       // Distinct instance
    .canonical_name = "blk.5.ffn_gate_exps.weight[expert=42]",
    .role = WeightRole::MoEExpertGate,
    .derivation = WeightDerivationKind::RebalancedExpertReplica,
    .source_instance_id = original_binding.instance_id,
    .layer = 5,
    .expert = 42,
    .tp_rank_or_device_index = 1,       // Socket/device it was replicated TO
};
```

This means the weight audit can answer: "Expert 42 at layer 5 has a replica on socket 1, derived from the original on socket 0, prepared as VNNI-packed CPU GEMM."

---

## 5. Integration Points

### 5.1 Graph Build (Initial Load)

**Before** (current):
```cpp
// In Qwen35MoEGraph::buildFFNGraph():
MoEExpertComputeStage::extractExpertViews(params);  // 2D views from 3D
MoEExpertComputeStage::prepareExpertGemmEngines(params);  // KernelFactory registration
```

**After** (new):
```cpp
// In Qwen35MoEGraph::buildFFNGraph():
MoEExpertComputeStage::extractExpertViews(params);  // 2D views from 3D (unchanged)

// Register slabs in PreparedWeightStore
params.gate_slab = store->registerExpertSlab({
    .layer_idx = layer,
    .role = WeightRole::MoEExpertGate,
    .device = device,
    .num_experts = 256,
    .local_expert_start = params.local_expert_start,
    .local_expert_count = params.local_expert_count,
    .rows_per_expert = intermediate,
    .cols_per_expert = d_model,
    .source_identity = gate_exps_identity,
});

// Prepare initial engines (same underlying VNNI/GPU repack)
MoEExpertWeightService::prepareAndRegisterExperts(ctx, store);
```

**Key difference**: The prepared engines go into `PreparedWeightStore::expert_slabs_` owned by ModelContext, not into `KernelFactory::prepared_gemm_registry_`.

### 5.2 Stage Execution (Decode — No Rebalance)

**Before**:
```cpp
void MoEExpertComputeStage::ensureGemmEnginesCached() {
    // Copy from params.prepared_*_gemm → cached_*_gemm_
    // No fallback to global KernelFactory preparation; refs/slabs must be store-backed.
}
```

**After**:
```cpp
void MoEExpertComputeStage::ensureGemmEnginesCached() {
    for (int e = local_start; e < local_end; ++e) {
        if (!cached_gate_gemm_[e]) {
            cached_gate_gemm_[e] = params_.prepared_store->expertGemmKernel(
                params_.gate_slab, e);
            // Hard-fail if null — initialization bug
            LLAMINAR_ASSERT(cached_gate_gemm_[e], "Expert engine missing");
        }
    }
}
```

### 5.3 Dynamic Rebalance

**Before**:
```cpp
MoEExpertWeightService::registerAndPrepareNewExperts(ctx, new_expert_ids);
// → KernelFactory registration + ITensorGemm* into params
```

**After**:
```cpp
// Phase 1: Prepare engines (same underlying work)
auto arrivals = MoEExpertWeightService::prepareExpertArrivals(ctx, new_expert_ids);

// Phase 2: Register in PreparedWeightStore with RebalancedExpertReplica identity
auto actually_new = store->registerArrivedExperts(params_.gate_slab, arrivals);

// Phase 3: Invalidate cached engines for newly-arrived experts
for (int e : actually_new) {
    cached_gate_gemm_[e] = nullptr;  // Will re-resolve from store on next execute
}
```

### 5.4 Model Unload / Context Teardown

**Before**: `~TensorBase()` for each expert view → `KernelFactory::clearCacheFor()` × 768 entries

**After**:
```cpp
// ModelContext destructor:
prepared_store_->releaseExpertSlab(gate_slab);  // Batch release, one lock acquisition
prepared_store_->releaseExpertSlab(up_slab);
prepared_store_->releaseExpertSlab(down_slab);
// Expert views destructed without touching KernelFactory
```

---

## 6. Migration Strategy

### Phase A: Add Expert Slab API to PreparedWeightStore (additive)

- Add `ExpertSlabRef`, `ExpertSlabDescriptor`, `ExpertArrival` types
- Add `registerExpertSlab()`, `expertGemmKernel()`, `registerArrivedExperts()`, `releaseDepartedExperts()`
- Internal storage uses `expert_slabs_` map
- **No behavioral changes** — existing MoE path continues to work via KernelFactory

### Phase B: Dual-Path Registration

- `MoEExpertWeightService::prepareGemmEngines()` registers in BOTH KernelFactory AND PreparedWeightStore
- Stage execution still reads from `params.prepared_*_gemm[]` (unchanged)
- Add validation that store and KernelFactory agree on engine pointers

### Phase C: Stage Reads from Store

- `ensureGemmEnginesCached()` resolves from `PreparedWeightStore::expertGemmKernel()`
- `params.prepared_*_gemm[]` becomes the initial-population cache (for latency)
- KernelFactory registration becomes the compatibility shim
- Dynamic rebalance uses `registerArrivedExperts()` + `releaseDepartedExperts()`

### Phase D: Remove KernelFactory Expert State

- Expert views no longer register in `prepared_gemm_registry_`
- `fused_gate_up_cache_` for experts moves into `ExpertSlabEntry::fused_cache`
- `~TensorBase()` no longer needs to scan expert entries
- KernelFactory `sliced_cache_` no longer contains expert-related entries

---

## 7. Testing Strategy

### 7.1 Unit Tests

| Test | What it validates |
|------|-------------------|
| `Test__PreparedWeightStore_ExpertSlab_RegisterAndLookup` | Register slab → populate experts → lookup engines |
| `Test__PreparedWeightStore_ExpertSlab_PartialPopulation` | Only some experts populated → others return nullptr |
| `Test__PreparedWeightStore_ExpertSlab_ArrivedExperts` | Register arrivals → verify availability mask updates |
| `Test__PreparedWeightStore_ExpertSlab_DepartedExperts` | Release experts → verify engines freed, mask updated |
| `Test__PreparedWeightStore_ExpertSlab_ReplicaIdentity` | Arrivals with `RebalancedExpertReplica` derivation retain source_device |
| `Test__PreparedWeightStore_ExpertSlab_FusedGateUp` | Fused kernel cache is per-slab, not global |
| `Test__PreparedWeightStore_ExpertSlab_Release` | `releaseExpertSlab()` frees all engines and removes fused cache |
| `Test__PreparedWeightStore_ExpertSlab_ModelUnload` | Two slabs → unload one → other still works |

### 7.2 Integration Tests

| Test | What it validates |
|------|-------------------|
| `V2_Integration_MoE_ExpertSlab_SingleDevice` | Qwen3.5-0.8B runs through store-backed expert engines |
| `V2_Integration_MoE_ExpertSlab_Rebalance` | 2-socket rebalance creates replicas via store |
| `V2_Integration_MoE_ExpertSlab_Eviction` | Replica eviction frees memory, execution continues on owner |
| `V2_Integration_MoE_ExpertSlab_NodeLocalTP` | 2xMPI expert parallelism through store |

### 7.3 Rebalance-Specific Tests

| Test | Scenario |
|------|----------|
| `Test__RebalancedExpertReplica_Identity` | Verify replica WeightIdentity has source_instance_id and RebalancedExpertReplica derivation |
| `Test__RebalancedExpertReplica_DeterministicDispatch` | Two sockets with replicated expert → same token routing → same dispatch decisions |
| `Test__RebalancedExpertReplica_EvictionPreservesOriginal` | Evict replica on socket 1 → socket 0 original still functional |
| `Test__RebalancedExpertReplica_HistogramDriven` | Feed skewed histogram → verify promotion/demotion proposals |
| `Test__RebalancedExpertReplica_ConcurrentPrepAndExecute` | Prepare new replicas while decode is running (thread safety) |

### 7.4 Performance Regression Gate

```bash
# Before/after comparison on Qwen3.5-35B decode:
LLAMINAR_PROFILING=1 ./build_v2_release/llaminar2 benchmark -m qwen35-35b.gguf -d cpu

# Key metric: decode tok/s should NOT regress from store indirection.
# Expert lookup is O(1) (vector index), not O(n) (linear scan).
```

---

## 8. Open Questions

1. **Fused gate+up cache scope**: Should fused kernels live in the slab (per-layer) or in a global model-context cache (shared across layers if experts have same shape)? Slab-local is simpler but wastes memory if all layers share the same intermediate/d_model dimensions. → **Decision**: Slab-local for now. Expert shapes are uniform within a model, but slab-local avoids cross-layer lifetime confusion.

2. **GPU VRAM pool sharing across slabs**: Currently `moe_packed_*_lifetime` is one pool per weight group per layer. Should we share across layers? → **Decision**: Keep per-slab pools. GPU memory accounting is already per-layer. Sharing would complicate cleanup.

3. **Thread safety model**: Rebalance can happen concurrently with decode (on different layers). The store needs per-slab locks, not one global lock. → **Decision**: `std::shared_mutex` per slab. Read (execute) takes shared lock; write (arrive/depart) takes exclusive lock. The slab is layer-scoped so different layers don't contend.

4. **MPI expert transfer integration**: `ExpertWeightTransfer::executeMigration()` currently serializes to `CPUPackedWeightsWithNativeBlocks` and sends via MPI. After transfer, it calls `registerAndPrepareNewExperts()`. The new path should call `registerArrivedExperts()` instead. → **Decision**: Phase C migration — adapter in `ExpertWeightTransfer` calls store API.

5. **ExpertParallel sharding mode**: Currently unused by any graph builder. Should we activate it in Qwen35MoESchema? → **Decision**: Not in this work package. Expert parallelism via `ExpertParallel` sharding is orthogonal — it pre-splits the 3D tensor at load time (static EP). Rebalancing requires all experts available. Keep `Replicate` + dynamic dispatch.

---

## 9. Relationship to MODEL_WEIGHT_LIFETIME_REDESIGN_PLAN.md

This design implements **Phase 5** (GPU Pipeline Registration) and parts of **Phase 7** (Stage Execution Uses PreparedWeightRef) specifically for MoE expert weights.

| Plan Phase | What this design covers |
|------------|------------------------|
| Phase 4 (PreparedWeightStore Compat) | Expert slab API is an extension of PreparedWeightStore |
| Phase 5 (GPU Pipeline → Store) | `LoadOrchestrator` products go into store slabs |
| Phase 7 (Stage Uses PreparedWeightRef) | `ensureGemmEnginesCached()` reads from store |
| Phase 8 (Slim KernelFactory) | Expert entries removed from `prepared_gemm_registry_` |
| Phase 9 (Host Release) | Expert slab descriptor carries host policy; `releaseRawWeights()` respects it |

The `RebalancedExpertReplica` derivation kind maps directly to the plan's `WeightDerivationKind::RebalancedExpertReplica` enum value and satisfies the requirement:

> "Dynamic expert replicas are future `RebalancedExpertReplica` bindings with source identity preserved."

---

## 10. Summary

The core insight is that expert weights are a **slab** — a fixed-size array of identically-shaped prepared GEMM engines, where individual slots can be populated, evicted, and replaced at runtime. This is fundamentally different from dense layer weights (frozen after preparation) and requires a different storage abstraction.

The `ExpertSlabRef` + `ExpertSlabEntry` pattern gives us:
- **O(1) lookup** by expert_id (vector index, not hash/linear scan)
- **Per-slab locking** (decode doesn't contend across layers)
- **Batch cleanup** on model unload (one operation per slab, not N×768 tensor destructions)
- **Traceability** (each slot knows its derivation, source, and device)
- **Runtime mutability** for rebalance (arrive/depart without rebuilding the graph)
