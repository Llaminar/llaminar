# MemoryPlanner Implementation Plan

## Problem Statement

Llaminar V2 has no unified memory budget. The five major VRAM consumers — weights, KV cache, activations, kernel workspace, and ad-hoc kernel buffers — allocate independently with no pre-flight "will this fit?" check. OOM surfaces as a raw CUDA/HIP runtime error mid-initialization.

The `llaminar plan` CLI subcommand exists as a stub. It should produce a complete execution plan with memory breakdown per device, answering "will this model fit on these devices with this parallelism strategy?" before any allocation occurs.

## Goals

1. Pre-flight memory validation in `OrchestrationRunner::initialize()` — fail fast with a diagnostic before any device allocation.
2. Full `llaminar plan` implementation via thin MPI — leverages existing `gatherClusterInventory()` + GGUF header parsing for cross-node inventory.
3. Migrate highest-impact ad-hoc kernel allocations to `IWorkspaceConsumer` so workspace budgets are accurate.
4. Add `estimateBytes()` APIs to KV cache and weight subsystems.

## Architecture

### MemoryPlanner (New)

Location: `src/v2/planning/MemoryPlanner.h`

Pure estimation class. No allocations. Takes model metadata + cluster inventory + config, returns per-device memory plans.

```
MemoryPlanner::plan(ModelMemoryProfile, ClusterInventory, OrchestrationConfig)
  → MemoryPlan {
      per_device: vector<DeviceMemoryPlan>
      total_bytes, total_available
      fits: bool
      diagnostics: vector<string>
    }
```

### Data Flow

```
GGUF Header (metadata only, no mmap)
  ↓
ModelMemoryProfile              ← NEW: extracted from GGUFModel.tensors[]
  weight_bytes_native: size_t   ← sum(GGUFTensorInfo.size_bytes)
  weight_bytes_packed: size_t   ← per-tensor: getCUDAPackedBytesPerWeight(K) × elements
  tensor_infos: vector<TensorSizeInfo>  ← name, bytes, quant_type per tensor
  architecture, n_layers, d_model, d_ff, n_heads, n_kv_heads, head_dim, vocab_size
  ↓
MemoryPlanner::plan()
  ├── WeightMemoryEstimator::estimate(profile, shard_config)
  │     → per-device weight bytes (accounts for TP column/row sharding + replication)
  ├── KVCacheMemoryEstimator::estimate(kv_config)
  │     → per-device KV cache bytes (uses existing getKVCacheBytesPerElement())
  ├── ActivationMemoryEstimator::estimate(schema_params)
  │     → per-device activation bytes (evaluates BufferSpec formulas)
  ├── WorkspaceMemoryEstimator::estimate(model_dims)
  │     → per-device workspace bytes (uses existing computeModelAwareBudgetFloor() formula)
  └── Compare totals vs DeviceInfo.free_memory_bytes per device
        → MemoryPlan with fits/diagnostics
```

### Call Sites

**Site 1: `OrchestrationRunner::initialize()` — Step 5.5**

Between `validateTPPPConfiguration()` (Step 5) and `buildComputeGraph()` (Step 6). At this point:
- `plan_` is built (has device assignments, TP degree, layer ranges, shard info)
- `model_ctx_` exists (GGUF fully loaded, `GGUFModel.tensors` available)
- `ClusterInventory` was gathered in `buildExecutionPlan()` — store on runner

```cpp
// In OrchestrationRunner::initialize(), after Step 5b:

// Step 5c: Validate memory plan
if (!validateMemoryPlan())
{
    syncInitStep(false, "validateMemoryPlan");
    return false;
}
if (!syncInitStep(true, "validateMemoryPlan"))
{
    return false;
}
```

Each rank validates its own device assignment against its local `DeviceInfo.free_memory_bytes`. If any rank fails, `syncInitStep()` propagates the failure cluster-wide.

**Site 2: `PlanCommand::execute()` — `llaminar plan` CLI**

Thin MPI mode: `mpirun -np N llaminar2 plan -m model.gguf`

```
1. Parse flags (-m, -s, -o, --format)
2. Initialize MPI (reuse MPIContextFactory::global())
3. gatherClusterInventory()          ← existing MPI_Allgatherv exchange
4. Read GGUF header (metadata only)  ← existing ModelLoader with setUseMmap(false)
5. Build ModelMemoryProfile from GGUFModel.tensors
6. For each candidate strategy:
     MemoryPlanner::plan(profile, inventory, candidate_config)
7. Select best feasible strategy
8. Rank 0: emit YAML plan + memory breakdown table
9. MPI_Finalize()
```

Also supports single-node (no MPI): if `world_size == 1`, `gatherClusterInventory()` already handles the local-only path (existing code, lines 748-800 of OrchestrationRunner.cpp).

## Implementation Phases

### Phase 1: Core Estimators + MemoryPlan Struct

**New files:**
- `src/v2/planning/MemoryPlan.h` — Output structs
- `src/v2/planning/ModelMemoryProfile.h` — GGUF-derived model sizing data
- `src/v2/planning/MemoryPlanner.h` / `.cpp` — Orchestrates the four estimators
- `src/v2/planning/WeightMemoryEstimator.h` / `.cpp`
- `src/v2/planning/KVCacheMemoryEstimator.h` / `.cpp`
- `src/v2/planning/ActivationMemoryEstimator.h` / `.cpp`
- `src/v2/planning/WorkspaceMemoryEstimator.h` / `.cpp`

#### MemoryPlan.h

```cpp
struct DeviceMemoryPlan {
    DeviceId device;
    size_t weight_bytes = 0;        // Model weights (packed for device)
    size_t kv_cache_bytes = 0;      // KV cache for assigned layers
    size_t activation_bytes = 0;    // BufferArena activation buffers
    size_t workspace_bytes = 0;     // DeviceWorkspaceManager (kernel scratch)

    size_t total_bytes() const;
    size_t device_total_bytes = 0;  // From DeviceInfo.memory_bytes
    size_t device_free_bytes = 0;   // From DeviceInfo.free_memory_bytes
    size_t headroom_bytes = 128 * 1024 * 1024;  // 128 MB default

    bool fits() const { return total_bytes() + headroom_bytes <= device_free_bytes; }
    size_t deficit() const;         // 0 if fits, else how much over
    std::string summary() const;    // Human-readable breakdown
};

struct MemoryPlan {
    std::vector<DeviceMemoryPlan> devices;
    std::vector<std::string> diagnostics;  // Warnings/errors
    bool fits() const;                     // All devices fit
    std::string renderTable() const;       // libfort-formatted breakdown
};
```

#### ModelMemoryProfile.h

```cpp
struct TensorSizeInfo {
    std::string name;
    size_t native_bytes;     // GGUF on-disk size
    std::string quant_type;  // "Q8_0", "Q4_K_M", etc.
    size_t elements;         // Total element count
    size_t K;                // Inner dimension (for packed size estimation)
};

struct ModelMemoryProfile {
    // Architecture
    std::string architecture;
    int n_layers, d_model, d_ff, n_heads, n_kv_heads, head_dim, vocab_size;

    // Weight sizing
    size_t total_native_bytes;    // Sum of all GGUFTensorInfo.size_bytes
    std::vector<TensorSizeInfo> tensors;

    // Factory
    static ModelMemoryProfile fromGGUF(const GGUFModel& model);
    static ModelMemoryProfile fromModelLoader(const ModelLoader& loader);

    // Query helpers
    size_t weightBytesForLayers(int first, int last) const;  // PP slice
    size_t embeddingBytes() const;
    size_t lmHeadBytes() const;
};
```

#### Estimator APIs

```cpp
// WeightMemoryEstimator
struct WeightEstimate {
    size_t native_bytes;   // As-is from GGUF
    size_t packed_bytes;   // After device-specific packing
    size_t upload_bytes;   // What actually lands on device
};

static WeightEstimate estimate(
    const ModelMemoryProfile& profile,
    DeviceId device,
    int shard_index = 0,      // TP shard
    int total_shards = 1,     // TP degree
    int first_layer = 0,      // PP range
    int last_layer = -1       // -1 = all
);

// KVCacheMemoryEstimator
static size_t estimate(
    int n_layers, int batch_size, int max_seq_len,
    int n_kv_heads, int head_dim,
    ActivationPrecision kv_precision,
    DeviceId device
);

// ActivationMemoryEstimator
static size_t estimate(
    int batch_size, int max_seq_len,
    int d_model, int d_ff,
    int n_heads, int n_kv_heads, int head_dim,
    int vocab_size,
    ActivationPrecision activation_precision,
    bool is_gpu  // GPU skips CPU-only attention workspace
);

// WorkspaceMemoryEstimator
static size_t estimate(
    int batch_size, int max_seq_len,
    int d_model, int d_ff,
    int vocab_size,
    DeviceId device
);
```

#### Reuse of Existing Code

| Estimator | Reuses |
|-----------|--------|
| Weight | `PlacementStrategy::getNativeBytesPerWeight()`, `getCUDAPackedBytesPerWeight()`, `getCPUPackedBytesPerWeight()` |
| KV Cache | `PlacementStrategy::getKVCacheBytesPerElement()` |
| Activation | `PlacementStrategy::estimateActivationBuffers()` formula |
| Workspace | `WorkspaceAllocator::computeModelAwareBudgetFloor()` formula |

These are currently private/static in their respective .cpp files. Phase 1 extracts the formulas into the estimator classes and calls them from both the planner and the existing code paths (no duplication).

### Phase 2: Integration with OrchestrationRunner

**Modified files:**
- `src/v2/execution/runner/OrchestrationRunner.h` — Add `ClusterInventory cluster_inventory_` member, `validateMemoryPlan()` method
- `src/v2/execution/runner/OrchestrationRunner.cpp` — Store inventory from `buildExecutionPlan()`, add Step 5c

Changes to `buildExecutionPlan()`:
```cpp
// Store inventory for later use by memory planner
cluster_inventory_ = gatherClusterInventory();
// ... existing plan building code uses cluster_inventory_ ...
```

New method `validateMemoryPlan()`:
```cpp
bool OrchestrationRunner::validateMemoryPlan()
{
    auto profile = ModelMemoryProfile::fromModelLoader(*model_loader_);
    auto plan = MemoryPlanner::plan(profile, cluster_inventory_, config_, plan_);

    if (!plan.fits())
    {
        std::string msg = "Memory plan validation failed:\n" + plan.renderTable();
        for (const auto& d : plan.diagnostics)
            msg += "\n  " + d;
        return setError(msg);
    }

    LOG_INFO("[MemoryPlanner] " << plan.renderTable());
    return true;
}
```

### Phase 3: `llaminar plan` via Thin MPI

**Modified files:**
- `src/v2/app/commands/PlanCommand.h` — Add MPI support flags
- `src/v2/app/commands/PlanCommand.cpp` — Full implementation

New CLI flags for plan:
```
-m, --model <path>             GGUF model file (required)
-s, --strategy <type>          auto|single-gpu|tp|pp|hybrid|cpu-only
-o, --output <file>            Write plan to file (default: stdout)
    --format <fmt>             yaml|json|table
    --max-seq-len <n>          Context length for KV cache sizing (default: model max)
    --batch-size <n>           Batch size (default: 1)
    --kv-precision <type>      fp16|q8_1|fp32|tq|auto
    --activation-precision     fp32|bf16|hybrid
    --headroom <mb>            Reserved headroom per device (default: 128)
```

Implementation flow:

```
execute():
  1. initializeLogging()
  2. Parse PlanConfig via CliSpec
  3. Initialize MPI (MPIContextFactory::global() or local-only)
  4. ClusterInventory = gatherInventory()
       - Reuses OrchestrationRunner::gatherClusterInventory() logic
       - Extract into standalone function: planning::gatherClusterInventory(mpi_ctx)
  5. Read GGUF header only (ModelLoader with setUseMmap(false))
  6. Build ModelMemoryProfile from loader
  7. For strategy == "auto":
       - Enumerate candidates: single-gpu, tp:2, tp:4, pp:2, hybrid
       - For each: MemoryPlanner::plan(profile, inventory, candidate_config)
       - Select first that fits (preference order: single > tp > pp > hybrid > cpu)
     For explicit strategy:
       - Build config for that strategy
       - MemoryPlanner::plan() — report fits/deficit
  8. Rank 0: renderTable() to stdout or file
  9. Rank 0: emit YAML config (consumable by `llaminar serve --config`)
  10. MPI_Finalize()
```

#### Extracting gatherClusterInventory

Currently lives on `OrchestrationRunner` and depends on `config_`, `mpi_ctx_`, `DeviceManager`. Extract a free function:

```cpp
// src/v2/planning/ClusterInventoryGatherer.h
namespace llaminar2::planning {

    /// Gather cluster inventory from all MPI ranks.
    /// Works for both multi-rank (MPI_Allgatherv) and single-rank (local only).
    ClusterInventory gatherClusterInventory(
        const std::shared_ptr<IMPIContext>& mpi_ctx,
        const std::vector<DeviceAddress>& explicit_tp_devices = {},
        const std::string& hostfile = ""
    );

} // namespace llaminar2::planning
```

This extracts the body of `OrchestrationRunner::gatherClusterInventory()` into a standalone function. `OrchestrationRunner` delegates to it. `PlanCommand` calls it directly.

### Phase 4: KV Cache `estimateBytes()` on IKVCache / KVCacheConfig

**Modified files:**
- `src/v2/kernels/KernelFactory.h` — Add static `estimateBytes()` to `KVCacheConfig`

```cpp
struct KVCacheConfig {
    // ... existing fields ...

    /// Estimate total device memory for this KV cache configuration.
    /// Does not allocate. Uses the same formulas as actual constructors.
    size_t estimateBytes() const;
};
```

Implementation dispatches on `precision` and `device.is_gpu()`:

| Precision | Formula (per layer per batch) |
|-----------|-------------------------------|
| FP32 | `4 × max_seq × kv_heads × head_dim × 4` (K+V+scratch×2) |
| FP16 | `4 × max_seq × kv_heads × head_dim × 2` |
| Q8_1 | `2 × ceil(max_seq × kv_heads × head_dim / 32) × 36` (K+V, block_q8_1 = 36 bytes/32 elements) |
| TQ | Ring buffers + FP16 scratch + quant temps + dequant params |
| Hybrid (GDN) | GDN state (`d_model × d_model × 4` per GDN layer) + attention KV for FA layers |

`local_n_kv_heads` is used when sharded (TP), else `n_kv_heads`.

### Phase 5: Migrate Top Ad-Hoc Allocators to IWorkspaceConsumer

Priority order by impact:

#### 5a. CUDAGatedDeltaNet / ROCmGatedDeltaNet

Current: `gpu_state_` and `deinterleave_scratch_` via raw `cudaGDN_gpu_malloc`. Not an `IWorkspaceConsumer`.

Change: Add `IWorkspaceConsumer` implementation to `CUDAGatedDeltaNet`:
```cpp
WorkspaceRequirements getWorkspaceRequirements(int m, int n, int k) const override {
    WorkspaceRequirements reqs;
    reqs.buffers.push_back({"gdn_state", state_size_bytes_, 256, true});
    reqs.buffers.push_back({"gdn_deinterleave_scratch", scratch_size_bytes_, 256, false});
    return reqs;
}
```

Fallback: Keep `cudaGDN_gpu_malloc` behind `if (!hasWorkspace())` for standalone tests.

#### 5b. CUDARingKVCacheTQ / ROCmRingKVCacheTQ

Current: 12 separate `cudaMalloc` calls for entries, scratch, params, temps.

Change: Report all buffers through `getWorkspaceRequirements()`. The KV cache is created before workspace allocation, so this requires reordering: estimate KV workspace needs during planning, allocate workspace first, then create KV cache with bound workspace. This is the most invasive change and may be deferred to Phase 6.

#### 5c. CUDANativeVNNIGemvTuned `kpar_partials`

Current: Grow-only `cudaMalloc` in C-level struct, hot-path decode.

Change: Add workspace buffer `"gemv_kpar_partials"` to `CUDAQuantisedGemmKernel::getWorkspaceRequirements()` for the decode (M=1) case. Size: `kpar × N × sizeof(float)` where kpar is the parallelism factor.

#### 5d. TurboQuant per-call alloc/free

Current: `d_k_temp`/`d_v_temp` allocated and freed every TQ quantize call.

Change: Pre-allocate as workspace buffers. Size: `max_seq × kv_dim × sizeof(__half)` each.

## Unit Tests

### Phase 1 Tests

Location: `tests/v2/unit/planning/`

```
Test__ModelMemoryProfile.cpp
  - FromGGUF_ExtractsArchitecture
  - FromGGUF_SumsTensorBytes
  - WeightBytesForLayers_FullRange
  - WeightBytesForLayers_PPSlice
  - EmbeddingBytes_MatchesVocabTimesD

Test__WeightMemoryEstimator.cpp
  - SingleDevice_NativeBytes
  - SingleDevice_CUDAPackedBytes
  - SingleDevice_CPUPackedBytes
  - TPSharded_HalvesColumnParallelWeights
  - TPSharded_ReplicatesNormWeights
  - PPSlice_OnlyCountsAssignedLayers

Test__KVCacheMemoryEstimator.cpp
  - FP16_BasicFormula
  - FP32_BasicFormula
  - Q8_1_BlockSizeRounding
  - TQ_IncludesScratchAndParams
  - Sharded_UsesLocalKVHeads
  - ZeroLayers_ReturnsZero

Test__ActivationMemoryEstimator.cpp
  - FP32_MatchesPlacementStrategyFormula
  - GPU_SkipsAttentionWorkspace
  - CPU_IncludesAttentionWorkspace
  - TP_UsesLocalDimensions

Test__WorkspaceMemoryEstimator.cpp
  - GPU_MatchesBudgetFloorFormula
  - CPU_ReturnsZero
  - LargeVocab_DominatedByLMHead

Test__MemoryPlanner.cpp
  - SingleGPU_Fits
  - SingleGPU_DoesNotFit_ReportsDeficit
  - TP2_SplitsWeightsAndKVCache
  - PP2_SplitsLayersAcrossDevices
  - MixedDevices_CUDAAndROCm
  - CPUOnly_NoWorkspace
  - RenderTable_ProducesValidOutput
  - Diagnostics_WarnsOnTightHeadroom
```

### Phase 2 Tests

Location: `tests/v2/unit/execution/runner/`

```
Test__OrchestrationRunner_MemoryValidation.cpp
  - ValidateMemoryPlan_Passes_WhenFits
  - ValidateMemoryPlan_Fails_WhenExceedsVRAM
  - ValidateMemoryPlan_DiagnosticMessage_IncludesBreakdown
```

### Phase 3 Tests

Location: `tests/v2/unit/planning/`

```
Test__ClusterInventoryGatherer.cpp
  - SingleRank_PopulatesLocalDevices
  - ExplicitTPDevices_OverridesDetected

Test__PlanCommand.cpp (integration)
  - SingleNode_ProducesYAML
  - ModelNotFound_ReturnsError
  - AutoStrategy_SelectsFeasible
```

### Phase 4 Tests

Location: `tests/v2/unit/kernels/`

```
Test__KVCacheConfig_EstimateBytes.cpp
  - FP16_MatchesActualAllocation
  - FP32_MatchesActualAllocation
  - Q8_1_MatchesActualAllocation
  - Sharded_ReducesByTPDegree
  - ZeroBatchSize_ReturnsZero
```

### Phase 5 Tests

Existing workspace consumer tests + new entries:

```
Test__CUDAGatedDeltaNet_Workspace.cpp
  - ReportsNonEmptyRequirements
  - FallbackWhenWorkspaceNotBound
  - WorkspaceBound_UsesExternalBuffer
```

## File Layout Summary

```
src/v2/planning/
├── MemoryPlan.h                    # Output structs (DeviceMemoryPlan, MemoryPlan)
├── ModelMemoryProfile.h            # GGUF-derived model sizing
├── ModelMemoryProfile.cpp
├── MemoryPlanner.h                 # Top-level planner
├── MemoryPlanner.cpp
├── WeightMemoryEstimator.h         # Per-component estimators
├── WeightMemoryEstimator.cpp
├── KVCacheMemoryEstimator.h
├── KVCacheMemoryEstimator.cpp
├── ActivationMemoryEstimator.h
├── ActivationMemoryEstimator.cpp
├── WorkspaceMemoryEstimator.h
├── WorkspaceMemoryEstimator.cpp
├── ClusterInventoryGatherer.h      # Extracted from OrchestrationRunner
└── ClusterInventoryGatherer.cpp

tests/v2/unit/planning/
├── Test__ModelMemoryProfile.cpp
├── Test__WeightMemoryEstimator.cpp
├── Test__KVCacheMemoryEstimator.cpp
├── Test__ActivationMemoryEstimator.cpp
├── Test__WorkspaceMemoryEstimator.cpp
├── Test__MemoryPlanner.cpp
└── Test__ClusterInventoryGatherer.cpp
```

## Dependency Graph

```
MemoryPlanner
  ├── ModelMemoryProfile        (depends on: GGUFModel, GGUFTensorInfo)
  ├── WeightMemoryEstimator     (depends on: PlacementStrategy sizing functions)
  ├── KVCacheMemoryEstimator    (depends on: PlacementStrategy::getKVCacheBytesPerElement)
  ├── ActivationMemoryEstimator (depends on: PlacementStrategy::estimateActivationBuffers)
  ├── WorkspaceMemoryEstimator  (depends on: WorkspaceAllocator budget formula)
  └── ClusterInventoryGatherer  (depends on: DeviceManager, MPIContext, NodeDetection)

PlanCommand
  ├── MemoryPlanner
  ├── ClusterInventoryGatherer
  ├── ModelLoader (header-only mode)
  └── OrchestrationConfigParser (for YAML emission)

OrchestrationRunner
  ├── MemoryPlanner
  └── ModelMemoryProfile (from already-loaded model_ctx_)
```

## Implementation Order

| Step | What | Depends On | Effort |
|------|------|-----------|--------|
| 1 | `MemoryPlan.h` structs | Nothing | S |
| 2 | `ModelMemoryProfile` + fromGGUF() | GGUFModel | S |
| 3 | `WeightMemoryEstimator` | ModelMemoryProfile, PlacementStrategy | M |
| 4 | `KVCacheMemoryEstimator` | PlacementStrategy | S |
| 5 | `ActivationMemoryEstimator` | PlacementStrategy | S |
| 6 | `WorkspaceMemoryEstimator` | WorkspaceAllocator | S |
| 7 | `MemoryPlanner` + renderTable() | Steps 3-6 | M |
| 8 | Unit tests for Steps 1-7 | Steps 1-7 | M |
| 9 | `ClusterInventoryGatherer` extraction | OrchestrationRunner | M |
| 10 | `OrchestrationRunner::validateMemoryPlan()` | Steps 7, 9 | S |
| 11 | `PlanCommand` full implementation | Steps 7, 9 | M |
| 12 | `KVCacheConfig::estimateBytes()` | KVCacheConfig | S |
| 13 | GDN workspace migration | CUDAGatedDeltaNet | M |
| 14 | Gemv kpar_partials workspace | CUDAQuantisedGemmKernel | M |
| 15 | TurboQuant per-call fix | CUDATurboQuantKernels | S |

Steps 1-8 are the core deliverable. Steps 9-11 are the `llaminar plan` integration. Steps 12-15 improve workspace budget accuracy.

## Output Format

`llaminar plan -m model.gguf` produces:

```
=== Memory Plan ===
╔═══════════╦══════════╦══════════╦══════════╦══════════╦══════════╦══════════╦══════╗
║  Device   ║ Weights  ║ KV Cache ║  Activ.  ║ Wkspace  ║  Total   ║ Avail.   ║  OK  ║
╠═══════════╬══════════╬══════════╬══════════╬══════════╬══════════╬══════════╬══════╣
║  CUDA:0   ║ 3620 MB  ║  256 MB  ║   42 MB  ║ 1200 MB  ║ 5118 MB  ║ 23535 MB ║  ✓   ║
╚═══════════╩══════════╩══════════╩══════════╩══════════╩══════════╩══════════╩══════╝

--- plan.yaml ---
model_path: "models/Qwen3.5-4B-Q8_0.gguf"
device: cuda:0
max_seq_len: 4096
kv_precision: fp16
memory:
  weights_mb: 3620
  kv_cache_mb: 256
  activations_mb: 42
  workspace_mb: 1200
  total_mb: 5118
  device_free_mb: 23535
  headroom_mb: 18417
```

For multi-device TP:

```
╔═══════════╦══════════╦══════════╦══════════╦══════════╦══════════╦══════════╦══════╗
║  Device   ║ Weights  ║ KV Cache ║  Activ.  ║ Wkspace  ║  Total   ║ Avail.   ║  OK  ║
╠═══════════╬══════════╬══════════╬══════════╬══════════╬══════════╬══════════╬══════╣
║  CUDA:0   ║ 1850 MB  ║  128 MB  ║   42 MB  ║  800 MB  ║ 2820 MB  ║ 23535 MB ║  ✓   ║
║  ROCm:0   ║ 1850 MB  ║  128 MB  ║   42 MB  ║  800 MB  ║ 2820 MB  ║ 31744 MB ║  ✓   ║
╚═══════════╩══════════╩══════════╩══════════╩══════════╩══════════╩══════════╩══════╝
  Strategy: TP-2 (local)
  Replicated: embedding (20 MB), norms (0.4 MB)
  Sharded: QKV, Wo, FFN gate/up/down, LM head
```
