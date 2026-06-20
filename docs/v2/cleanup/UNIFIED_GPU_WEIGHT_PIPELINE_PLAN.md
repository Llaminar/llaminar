# Unified GPU Weight Pipeline

## Problem Statement

Llaminar has two structurally identical but independently invoked GPU weight loading pipelines:

| Pipeline | Location | What It Loads | When It Runs |
|----------|----------|---------------|--------------|
| **Dense GEMM** | `WeightManager::packGemmWeightsViaPipeline()` | 2D GEMM weights (attn_q, ffn_gate, etc.) | `finalizeForDevice()` before graph build |
| **MoE Expert** | `MoEExpertWeightService::prepareGemmEnginesGPU()` | Per-expert 2D views from 3D `_exps.weight` tensors | Graph builder (or was lazy at execution time) |

Both pipelines use the **identical** underlying mechanism:
1. `LoadOrchestrator` → `WeightVRAMPool` (plan → allocate → H2D + GPU repack → create GEMM kernels)
2. Pipelined streaming: host → pinned ring → staging slot → repack kernel → final VRAM slot
3. Shared-pointer lifetime: GEMM kernels hold `shared_ptr<LoadOrchestrator>` keeping VRAM alive

The MoE pipeline was historically invoked **lazily at execution time** because 256 experts × 40 layers × 3 weights = 30,720 GEMM engines seemed too large for VRAM. However:
- The 35B-Q4_K model's expert weights total ~13 GB repacked, fitting in 31 GB MI60 alongside ~7 GB for dense weights + KV cache
- The "lazy" path required retaining 3D tensor host data in RAM indefinitely — a RAM bloat we've now eliminated (Phase 9)
- Without the lazy fallback, GPU MoE inference is broken

**Solution**: Merge both into a **single unified pipeline** invocation in `WeightManager::packGemmWeightsViaPipeline()` that handles dense GEMM weights AND MoE expert weights in one shot.

---

## Architecture After Unification

```
WeightManager::packGemmWeightsViaPipeline(device)
    │
    ├── Step 1: Collect dense GEMM weights (existing)
    │   └── isGemmWeight(name) → attn_q, attn_k, ffn_gate, ffn_up, ffn_down, output.weight, ...
    │
    ├── Step 2: Collect MoE expert weights (NEW)
    │   ├── Find 3D _exps.weight tensors in cache
    │   ├── Extract 2D expert views (create_view per expert)
    │   └── Create per-expert WeightJobs (gate_L0_e0, up_L0_e0, down_L0_e0, ...)
    │
    ├── Step 3: Plan ALL jobs into single LoadOrchestrator
    │   └── One WeightVRAMPool allocation for everything
    │
    ├── Step 4: Execute pipeline (pipelined H2D + GPU repack)
    │   └── DeviceLoadPipeline::processJobs() — streaming, overlapped
    │
    ├── Step 5: Create GEMM kernels from pool slots
    │   ├── Dense weights → KernelFactory::registerPreparedGemmFromTransfer()
    │   └── Expert weights → ExpertGemmRegistry (new, per-layer per-expert lookup)
    │
    └── Step 6: Release host data
        ├── Dense weights → tensor->release_host_weight_data() (existing)
        └── MoE 3D tensors → tensor->release_host_weight_data() (NEW — safe now)
```

**Graph builders** (e.g., `Qwen35MoEGraph`) no longer call `prepareExpertGemmEngines()` at build time. Instead they **query** the `ExpertGemmRegistry` for pre-prepared engines.

**Dynamic rebalancing** is unaffected — `MoEExpertWeightService::registerAndPrepareNewExpertsGPU()` still creates small one-off LoadOrchestrators for newly-arrived experts at runtime, using `ExpertWeightBlobs` (serialized packed data) as the host source.

---

## Key Design Decisions

### 1. Single VRAM Allocation vs. Separate Pools

**Decision**: Single allocation for all weights (dense + expert) via one `LoadOrchestrator`.

**Rationale**:
- One large allocation is faster and less fragmented than many small ones
- The LoadOrchestrator pattern already handles this well
- The staging ring buffer is sized to `max(all weight raw_bytes)` — expert slices are typically smaller than the largest dense weight (LM head), so no staging waste increase

### 2. Expert GEMM Registry

**Decision**: New `ExpertGemmRegistry` class owned by `WeightManager`, keyed by `(device, layer_idx, expert_id, role)`.

**Why not KernelFactory?** KernelFactory is keyed by `TensorBase*` pointer. Expert views are temporary objects created during pipeline execution — their pointers become invalid after the pipeline. We need a stable lookup mechanism for graph builders.

**Interface**:
```cpp
class ExpertGemmRegistry {
public:

    struct Key { DeviceId device; int layer; int expert; WeightRole role; };

    void registerEngine(Key key, ITensorGemm* engine);
    ITensorGemm* getEngine(Key key) const;

    // Bulk query for graph builder
    struct LayerEngines {
        std::vector<ITensorGemm*> gate;  // [num_experts]
        std::vector<ITensorGemm*> up;    // [num_experts]
        std::vector<ITensorGemm*> down;  // [num_experts]
    };
    LayerEngines getLayerEngines(DeviceId device, int layer, int num_experts) const;
};
```

### 3. Dynamic Rebalancing Compatibility

Expert engines prepared by the unified pipeline are **identical** to those created by the old per-layer `prepareGemmEnginesGPU()` — same `ROCmQuantisedGemmKernel` / `CUDAQuantisedGemmKernel` objects with the same VRAM pool lifetime pattern.

Dynamic rebalancing (`registerAndPrepareNewExpertsGPU`) creates **new** small `LoadOrchestrator` instances for just the arriving experts. These produce GEMM kernels that are stored in the same `moe_owned_kernels` vector and `prepared_*_gemm` arrays. The `ExpertGemmRegistry` is updated to reflect the new engine, evicting the old one.

### 4. Host Data Release Timing

After the unified pipeline completes:
- Dense tensors: `release_host_weight_data()` inline (existing behavior)
- MoE 3D tensors: `release_host_weight_data()` inline (NEW — safe because all experts are now on GPU)
- Expert 2D views: become dangling (but they're temporary locals, not stored)

The 3D tensors can now properly report `isInPreparedGemmRegistry() == true` (or equivalent) so `releaseAllHostWeightData()` also handles them correctly.

### 5. VRAM Budget Validation

Before allocation, verify the planned VRAM fits within available memory:
```
Available VRAM = total_vram - reserved_kv_cache - reserved_activations - safety_margin
Required = dense_weights_repacked + expert_weights_repacked + staging_slots
```

If it doesn't fit → fail with clear error message suggesting `LLAMINAR_WEIGHT_STREAMING=1` or a smaller model.

---

## Phases

### Phase 1: ExpertGemmRegistry

**Goal**: Create the registry that stores pre-prepared expert GEMM engines, queryable by graph builders.

**Files**:
- NEW: `src/v2/loaders/ExpertGemmRegistry.h`
- NEW: `src/v2/loaders/ExpertGemmRegistry.cpp`
- MODIFY: `src/v2/loaders/WeightManager.h` — add `ExpertGemmRegistry` member + accessor

**Interface**:
```cpp
// src/v2/loaders/ExpertGemmRegistry.h
class ExpertGemmRegistry {
public:
    enum class WeightRole : uint8_t { GATE, UP, DOWN };

    void registerEngine(DeviceId device, int layer, int expert, WeightRole role,
                        ITensorGemm* engine, std::shared_ptr<ITensorGemm> ownership);

    ITensorGemm* getEngine(DeviceId device, int layer, int expert, WeightRole role) const;

    // Bulk populate for a stage's Params struct
    bool populateExpertEngines(DeviceId device, int layer, int num_experts,
                               std::vector<ITensorGemm*>& gate_out,
                               std::vector<ITensorGemm*>& up_out,
                               std::vector<ITensorGemm*>& down_out) const;

    // Replace engine (for dynamic rebalancing)
    void replaceEngine(DeviceId device, int layer, int expert, WeightRole role,
                       ITensorGemm* engine, std::shared_ptr<ITensorGemm> ownership);

    size_t size() const;
    void clear();
};
```

**Tests** (unit):
- `tests/v2/unit/loaders/Test__ExpertGemmRegistry.cpp`
- Register/query/replace operations
- `populateExpertEngines` fills vectors correctly
- Thread safety (concurrent reads)

**Acceptance criteria**:
- Unit tests pass
- No functional changes to inference (registry not yet wired in)

---

### Phase 2: Unified Pipeline — Include MoE Experts

**Goal**: Extend `packGemmWeightsViaPipeline()` to include MoE expert weights alongside dense GEMM weights in a single LoadOrchestrator invocation.

**Files**:
- MODIFY: `src/v2/loaders/WeightManager.cpp` — `packGemmWeightsViaPipeline()`:
  - After collecting dense GEMM weights, also collect 3D `_exps.weight` tensors
  - Extract 2D expert views from each 3D tensor
  - Create per-expert WeightJobs with names like `moe_L{layer}_gate_e{expert}`
  - After pipeline execution, create expert GEMM kernels and register in `ExpertGemmRegistry`
  - Release 3D tensor host data after all experts are prepared
- MODIFY: `src/v2/loaders/WeightManager.h` — expose `ExpertGemmRegistry` accessor
- NOTE: `GraphSchema::isMoEExpertWeight(name)` was considered but left out; `WeightManager`
    handles MoE expert parent parsing locally so the graph schema remains declarative.

**Key implementation details**:

```cpp
// Inside packGemmWeightsViaPipeline(), after collecting dense weights:

// Step 2: Collect MoE expert weights
struct MoELayer {
    int layer_idx;
    TensorBase* gate_exps;  // 3D: [cols, rows_per_expert, num_experts]
    TensorBase* up_exps;
    TensorBase* down_exps;
    int num_experts;
    int expert_intermediate;
};
std::vector<MoELayer> moe_layers;

for (auto& [name, tensor] : cache_) {
    if (name.find("ffn_gate_exps.weight") != std::string::npos) {
        // Parse layer index from "blk.N.ffn_gate_exps.weight"
        int layer = parseLayerIndex(name);
        // Find matching up_exps and down_exps
        // Extract expert metadata (num_experts from shape[2], intermediate from shape[1])
        // Add to moe_layers
    }
}

// For each MoE layer, create per-expert view jobs:
for (auto& moe : moe_layers) {
    for (int e = 0; e < moe.num_experts; ++e) {
        // gate view
        auto gate_view = moe.gate_exps->create_view({rows, cols}, e * elements_per_expert);
        // Plan + create job for "moe_L{layer}_gate_e{e}"
        // ... same for up and down
    }
}
```

**Tests** (integration):
- `tests/v2/integration/loaders/Test__UnifiedGPUPipeline.cpp`
- Load a MoE model (Qwen35 35B Q4_K)
- Verify dense weights AND expert weights are all in the same VRAM pool
- Verify `ExpertGemmRegistry` is populated for all layers/experts
- Verify host data for both dense and 3D MoE tensors is released
- Verify GEMM engines produce correct matmul results (spot-check one expert)

**Acceptance criteria**:
- All dense weights + all MoE experts loaded in single pipeline invocation
- Single `LoadOrchestrator` / `WeightVRAMPool` allocation
- Host data fully released after pipeline
- `ExpertGemmRegistry` queryable

---

### Phase 3: Graph Builder Consumes Registry

**Goal**: Update `Qwen35MoEGraph` to query pre-prepared expert GEMM engines from the `ExpertGemmRegistry` instead of calling `prepareExpertGemmEngines()` at graph-build time.

**Files**:
- MODIFY: `src/v2/models/qwen35moe/Qwen35MoEGraph.cpp`:
  - Remove the `if (device.is_gpu() && expert_params.expert_mask.empty())` skip block
  - Remove the `else { MoEExpertComputeStage::prepareExpertGemmEngines(expert_params); }` call
  - Instead: query `ExpertGemmRegistry` from WeightManager to populate `expert_params.prepared_*_gemm`
- MODIFY: `src/v2/models/qwen35moe/Qwen35MoEGraph.h` — add WeightManager/Registry accessor if needed
- MODIFY: `src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h/cpp` — expose registry access to graph builders

**Key change in Qwen35MoEGraph::build()**:
```cpp
// OLD:
if (device.is_gpu() && expert_params.expert_mask.empty()) {
    expert_params.prepared_gate_gemm.assign(num_experts, nullptr);
    expert_params.prepared_up_gemm.assign(num_experts, nullptr);
    expert_params.prepared_down_gemm.assign(num_experts, nullptr);
} else {
    MoEExpertComputeStage::prepareExpertGemmEngines(expert_params);
}

// NEW:
if (device.is_gpu()) {
    // Query pre-prepared engines from unified pipeline
    auto* registry = weight_manager_->expertGemmRegistry();
    if (!registry->populateExpertEngines(device, layer_idx, num_experts,
            expert_params.prepared_gate_gemm,
            expert_params.prepared_up_gemm,
            expert_params.prepared_down_gemm)) {
        LOG_ERROR("[Qwen35MoEGraph] Expert engines not found in registry for layer " << layer_idx);
        return false;
    }
} else {
    // CPU path: prepare GEMM engines inline (VNNI pack, no GPU pipeline)
    MoEExpertComputeStage::prepareExpertGemmEngines(expert_params);
}
```

**Tests**:
- Run existing MoE parity test on ROCm: `v2_integration_parity_qwen35moe_single_device --gtest_filter="*ROCm*"`
- Run existing MoE parity test on CUDA (if available)
- Verify token predictions match PyTorch reference

**Acceptance criteria**:
- Graph builder no longer calls `prepareExpertGemmEngines()` for GPU
- Parity tests pass (ROCm + CUDA)
- No lazy fallback path exists

---

### Phase 4: Remove Dead Code + Host Data Release

**Goal**: Clean up the old lazy path remnants and ensure 3D tensor host data is properly released.

**Files**:
- MODIFY: `src/v2/execution/compute_stages/stages/MoEExpertComputeStage.cpp`:
  - `ensureGemmEnginesForExperts()` — remove the hard-failure path for "no payload provider" (now all engines are pre-resolved OR come from rebalancing)
  - Simplify: if engine is NULL and no payload provider → error (this should never happen after Phase 3)
- MODIFY: `src/v2/loaders/WeightManager.cpp`:
  - `releaseAllHostWeightData()` — 3D MoE tensors now have their engines in `ExpertGemmRegistry`, so they can be safely released
  - Add registry check: `if (expert_registry_.hasEnginesFor(tensor))` → safe to release
- REMOVE: The `isNonGemmWeight` exclusion for `_exps.weight` is kept (these tensors are NOT in the dense GEMM filter; they're handled by the new MoE section of the unified pipeline)

**Tests**:
- Verify RAM usage after pipeline: 3D MoE tensors should be released
- Run `releaseAllHostWeightData()` and confirm expert 3D tensors are freed
- Integration test: full Qwen35 inference with LLAMINAR_LOG_LEVEL=DEBUG, grep for "RETAINED" — should show zero MoE tensor retentions

**Acceptance criteria**:
- Zero host RAM retained for MoE expert weights after pipeline
- Dynamic rebalancing still works (separate test with ExpertWeightBlobs)
- No "RETAINED host data for blk.*.ffn_*_exps.weight" in logs

---

### Phase 5: Dynamic Rebalancing Integration

**Goal**: Ensure dynamic expert rebalancing works with the unified pipeline's registry.

**Files**:
- MODIFY: `src/v2/execution/moe/MoEExpertWeightService.cpp`:
  - `registerAndPrepareNewExpertsGPU()` — after creating new GEMM engine, also update `ExpertGemmRegistry`
  - `releaseDepartedExperts()` — remove engine from registry when departing
- MODIFY: `src/v2/execution/compute_stages/stages/MoEExpertComputeStage.cpp`:
  - `updateExpertMask()` — verify it correctly handles registry-populated engines

**Tests**:
- Unit test: simulate expert arrival (ExpertWeightBlobs → new engine in registry)
- Unit test: simulate expert departure (engine removed from registry)
- Integration test: run with `LLAMINAR_MOE_REBALANCE_MODE=dynamic` and verify no crashes

**Acceptance criteria**:
- Expert arrival creates new GEMM engine via one-off LoadOrchestrator
- Engine registered in ExpertGemmRegistry
- Departed experts properly cleaned up
- No memory leaks

---

### Phase 6: CPU Path Alignment (Optional)

**Goal**: Ensure CPU-only models (no GPU device) still work correctly with the unified pipeline. This phase is low-priority since CPU path was never broken.

**Files**:
- Verify `packGemmWeightsViaPipeline()` early-returns for CPU devices (existing behavior)
- Verify `Qwen35MoEGraph` falls back to `prepareExpertGemmEngines()` for CPU device (Phase 3 already handles this)

**Tests**:
- Run MoE parity test with `-d cpu` device
- Verify CPU GEMM engines prepared via old VNNI interleave path

**Acceptance criteria**:
- CPU inference unaffected by changes

---

## VRAM Budget Analysis

### Qwen3.5-35B-A3B (Q4_K quantization)

| Component | Count | Per-item | Total |
|-----------|-------|----------|-------|
| Dense GEMM (attn_q/k/v/o × 40L) | 160 | ~2.5 MB | ~400 MB |
| Dense GEMM (shared_expert gate/up/down × 40L) | 120 | ~2.5 MB | ~300 MB |
| Dense GEMM (LM head) | 1 | ~50 MB | ~50 MB |
| MoE gate experts (64 × 40L) | 2,560 | ~110 KB | ~275 MB |
| MoE up experts (64 × 40L) | 2,560 | ~110 KB | ~275 MB |
| MoE down experts (64 × 40L) | 2,560 | ~110 KB | ~275 MB |
| Norms, embeddings (non-GEMM) | ~85 | small | ~20 MB |
| Staging ring (3 slots × max_weight) | 3 | ~50 MB | ~150 MB |
| **Total VRAM** | | | **~1.75 GB** |

Wait — 64 experts × 768 intermediate × 2560 d_model at Q4_K (144 bytes per 256 elements):
- Per expert gate weight: 768 × 2560 / 256 × 144 = ~1.1 MB raw, repacked payload ≈ similar
- Per expert × 3 roles = ~3.3 MB
- All experts all layers: 64 × 40 × 3.3 = ~8.4 GB

**Revised total**: ~750 MB dense + ~8.4 GB MoE + ~150 MB staging ≈ **9.3 GB**

On 31 GB MI60 with ~4 GB KV cache + ~2 GB activations = **~25 GB available**. Fits comfortably.

---

## Risk Analysis

| Risk | Mitigation |
|------|-----------|
| Single VRAM alloc too large for small GPUs | Fail with clear error, suggest weight streaming |
| Expert view extraction in WeightManager creates coupling | Keep view extraction as a utility function, minimal coupling |
| Staging slot waste (max weight determines slot size) | LM head is typically largest; expert slices are smaller. No additional waste. |
| Dynamic rebalancing creates orphaned VRAM | Existing shared_ptr pattern handles this (departed engine released → old orchestrator may be freed) |
| Pipeline ordering (must complete before graph build) | Already the case: `finalizeForDevice()` runs before `orchestrator->initializeBuffers()` |

---

## File Inventory

### New Files
| File | Purpose |
|------|---------|
| `src/v2/loaders/ExpertGemmRegistry.h` | Expert GEMM engine registry (header) |
| `src/v2/loaders/ExpertGemmRegistry.cpp` | Expert GEMM engine registry (impl) |
| `tests/v2/unit/loaders/Test__ExpertGemmRegistry.cpp` | Unit tests for registry |
| `tests/v2/integration/loaders/Test__UnifiedGPUPipeline.cpp` | Integration test (if needed) |

### Modified Files
| File | Change |
|------|--------|
| `src/v2/loaders/WeightManager.h` | Add ExpertGemmRegistry member + accessor |
| `src/v2/loaders/WeightManager.cpp` | Extend `packGemmWeightsViaPipeline()` for MoE |
| `src/v2/models/qwen35moe/Qwen35MoEGraph.cpp` | Query registry instead of inline prepare |
| `src/v2/execution/moe/MoEExpertWeightService.cpp` | Update registry on rebalance |
| `src/v2/execution/compute_stages/stages/MoEExpertComputeStage.cpp` | Simplify GPU engine lookup |
| `src/v2/execution/local_execution/graph/GraphSchema.h` | No change — MoE expert parsing remains local to `WeightManager` |
| `tests/v2/CMakeLists.txt` | Add new test targets |

### Removed Code (Phase 4)
- `Qwen35MoEGraph.cpp`: The `if (device.is_gpu() && expert_mask.empty()) { assign nullptrs }` block
- `MoEExpertComputeStage.cpp`: Lazy fallback error path (replaced with clean error)
- `WeightManager.cpp`: `hostDataRequired()` no longer needs special MoE handling (already removed in Phase 9)

---

## Implementation Order

```
Phase 1 (ExpertGemmRegistry)     ← Pure additive, no behavioral change
    │
    ▼
Phase 2 (Unified Pipeline)       ← Behavioral change: MoE experts loaded in GPU pipeline
    │
    ▼
Phase 3 (Graph Builder Query)    ← Behavioral change: graph builder uses registry
    │
    ▼
Phase 4 (Dead Code + Release)    ← Cleanup: remove lazy path, release host data
    │
    ▼
Phase 5 (Rebalancing)            ← Integration: rebalance updates registry
    │
    ▼
Phase 6 (CPU Path)               ← Verification only
```

Each phase has clear unit/integration test gates. Phases 1-3 are the critical path. Phases 4-6 are cleanup and verification.
