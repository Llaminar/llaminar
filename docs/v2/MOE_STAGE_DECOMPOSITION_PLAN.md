# MoE Stage Decomposition: From Monolith to Device-Agnostic Stages

**Status:** Active project plan — phased implementation  
**Branch:** `feat/qwen35-moe`  
**Prerequisite:** Dynamic rebalancing path is stable (Milestone 5 of `QWEN35_MOE_EXPERT_REBALANCING_PLAN.md`)  
**Related:** `docs/v2/QWEN35_MOE_EXPERT_REBALANCING_PLAN.md` line 258

---

## Problem

`MoEFFNStage.cpp` is 1803 lines doing 14 distinct jobs in a single monolithic stage:

| Concern | Lines (approx) | Device-specific? |
|---------|----------------|------------------|
| Routing (softmax top-k) | 30 | No (delegates to IMoEKernel) |
| Token grouping / expert lists | 40 | No |
| Gather tokens per expert | 20 | No (delegates to IMoEKernel) |
| Expert FFN prefill (gate+up+SwiGLU+down) | 150 | Via ITensorGemm engines |
| Expert FFN decode (batched GEMV + fused down) | 280 | Via ITensorGemm engines + CPU SwiGLU primitives |
| Scatter-add weighted output | 20 | No (delegates to IMoEKernel) |
| Histogram recording | 10 | No |
| EP filtering (expert_mask + local_expert_start) | 30 | No |
| GEMM engine lazy init / caching | 60 | **Yes** — device-specific dispatch |
| Weight view extraction (3D→2D) | 110 | No |
| VNNI/CPU weight packing | 80 | **Yes** — `CPUNativeVNNIGemmKernel.h`, `CPUPackedWeights.h` |
| CUDA batch packing | 90 | **Yes** — `#ifdef HAVE_CUDA` |
| ROCm batch packing | 90 | **Yes** — `#ifdef HAVE_ROCM` |
| Dynamic rebalance lifecycle | 200 | **Yes** — VNNI repack, madvise, NUMA audit |
| Weight serialization / transfer | 100 | **Yes** — `CPUPackedWeights` serialization |
| NUMA auditing | 50 | **Yes** — `numaif.h`, `sched.h` |
| Scratch buffer management | 60 | No |
| Dump info / diagnostics | 80 | No |

The stage directly includes CPU-specific headers (`CPUNativeVNNIGemmKernel.h`, `CPUPackedWeights.h`, `numaif.h`, `sched.h`) and has `#ifdef HAVE_CUDA` / `#ifdef HAVE_ROCM` blocks constructing device-specific kernel objects. This makes it impossible to unit-test routing or expert computation without pulling in an entire backend toolchain.

`TPMode::ExpertParallel` exists in the enum ([GraphSchema.h](../../src/v2/execution/local_execution/graph/GraphSchema.h)) but `GraphResolver::resolveTPCollective()` has a TODO stub. EP is instead handled imperatively in `Qwen35MoEGraph::buildFFNGraph()` with `local_expert_start`/`local_expert_count` and a `FusedAddAllreduceStage`.

## Goals

1. **Device-agnostic stages** — stages are glue/orchestration (CPU-side), dispatching to device-specific kernels (CPU, CUDA, ROCm) via existing kernel interfaces (`IMoEKernel`, `ITensorGemm`).
2. **Unit-testable / mockable** — each stage can be tested with mock kernels, no GPU required.
3. **Graph-level EP** — `TPMode::ExpertParallel` in `GraphResolver` auto-inserts AllReduce collectives, replacing imperative wiring in the graph builder.
4. **Preserve dynamic rebalancing** — the phased rebalance API (`releaseDepartedExperts`, `registerAndPrepareNewExperts`, `applyExpertMask`) continues to work through the weight service.
5. **Zero behavioral change** — each phase is individually testable; parity tests pass after every phase.

## Non-Goals

- Changing MoE numerics or routing algorithm.
- Introducing new GPU kernels — we're restructuring, not rewriting.
- Removing the decode fast-path (`executeSingleToken`) — it stays as a specialized path inside `MoEExpertComputeStage`.
- Splitting `SharedExpertFFNStage` or `SharedExpertGateStage` — they're already well-factored.

---

## Current Architecture

```
Qwen35MoEGraph::buildFFNGraph()
    ├── FusedResidualNormStage          ← residual + pre-FFN norm
    ├── MoEFFNStage (1803 lines)        ← route + gather + expert FFN + scatter + histogram + EP filter + rebalance + ...
    ├── SharedExpertFFNStage            ← always-active dense SwiGLU
    ├── SharedExpertGateStage           ← sigmoid gating
    └── FusedAddAllreduceStage          ← moe_output + shared_output + allreduce (EP path)
```

## Target Architecture

```
Qwen35MoEGraph::buildFFNGraph()
    ├── FusedResidualNormStage                        ← residual + pre-FFN norm
    ├── MoERoutingStage [TPMode::None]                ← route + histogram + EP mask
    ├── MoEExpertComputeStage [TPMode::ExpertParallel]← gather + expert FFN + scatter
    │       └── [AllReduceStage auto-inserted by GraphResolver]
    ├── SharedExpertFFNStage [TPMode::RowParallel]    ← always-active dense SwiGLU
    │       └── [AllReduceStage auto-inserted by GraphResolver]
    ├── SharedExpertGateStage                         ← sigmoid gating
    └── ResidualAddStage                              ← moe_output + shared_output
```

Plus a non-stage service:
```
MoEExpertWeightService ← weight lifecycle: view extraction, GEMM prep, serialization, rebalance, NUMA audit
```

---

## Decomposition Design

### Stage 1: `MoERoutingStage` (~200 lines)

**Type:** `ComputeStageType::MOE_ROUTER`  
**TP Mode:** `TPMode::None` (routing is replicated — all ranks see the same result)

**Params:**
```
input              TensorBase* [seq_len, d_model]     READ
gate_weights       TensorBase* [num_experts, d_model]  READ (weight)
num_experts        int
top_k              int
norm_topk_prob     bool
expert_mask        vector<bool>                        EP filtering
local_expert_start int                                 EP range
local_expert_count int                                 EP range
decode_histogram   DecodeExpertHistogram*               optional, not owned
layer_idx          int                                 for histogram
replica_set        ExpertReplicaSet                    for per-token dispatch
my_socket_id       int                                 for replica dispatch
```

**Outputs (arena buffers):**
```
routing_indices    int32 [seq_len, top_k]              expert IDs
routing_weights    float [seq_len, top_k]              normalized weights
router_logits      float [seq_len, num_experts]        raw logits (for dump/parity)
```

**What it does:**
1. Delegates to `IMoEKernel::route()` → `MoERoutingResult`
2. If EP active: zeros weights for non-local experts (same logic as current `execute()`)
3. Records decode histogram if histogram pointer non-null
4. Stashes routing results for dump info
5. If replicas active: annotates which socket computes each expert per token

**New BufferIds needed:** `MOE_ROUTING_INDICES`, `MOE_ROUTING_WEIGHTS`

### Stage 2: `MoEExpertComputeStage` (~500 lines, renamed from MoEFFNStage)

**Type:** `ComputeStageType::MOE_EXPERT_FFN`  
**TP Mode:** `TPMode::ExpertParallel` (partial results; allreduce after)

**Params:**
```
input              TensorBase* [seq_len, d_model]     READ
routing_indices    TensorBase* [seq_len, top_k]       READ (from routing stage)
routing_weights    TensorBase* [seq_len, top_k]       READ (from routing stage)
expert_intermediate int
num_experts        int
top_k              int
local_expert_start int
local_expert_count int
expert_mask        vector<bool>

# Per-expert GEMM engines (set by MoEExpertWeightService at build time)
prepared_gate_gemm  vector<ITensorGemm*>
prepared_up_gemm    vector<ITensorGemm*>
prepared_down_gemm  vector<ITensorGemm*>

# GPU lifetime management (owned by service, referenced here)
moe_owned_kernels         vector<shared_ptr<ITensorGemm>>
moe_packed_*_lifetime     shared_ptr<void>

# Scratch buffers
gate_scratch       TensorBase*
up_scratch         TensorBase*

# Output
output             TensorBase* [seq_len, d_model]     WRITE
```

**What it does:**
- **Prefill path** (`seq_len > 1`): Group tokens by expert → `gatherTokenBatch()` → gate+up GEMM → SwiGLU → down GEMM → `scatterAddWeighted()`
- **Decode path** (`seq_len == 1`): Batched gate+up GEMV → fused SwiGLU+down → weighted accumulate
- Dispatches all compute via `ITensorGemm` (device-agnostic) and `IMoEKernel` (gather/scatter/SwiGLU)
- **No device-specific `#include`s** — no `CPUNativeVNNIGemmKernel.h`, no `numaif.h`, no `#ifdef HAVE_CUDA`

### Service: `MoEExpertWeightService` (~500 lines)

**Not a compute stage** — called at graph-build time and during dynamic rebalancing.

**Responsibilities:**
1. **`extractExpertViews()`** — Create per-expert 2D tensor views from 3D packed tensors, handling EP slicing
2. **`prepareGemmEngines()`** — Dispatch to CPU VNNI repacking, CUDA batch packing, or ROCm batch packing based on device type
3. **`prepareGemmEnginesCPU()`** — VNNI weight repacking via `KernelFactory`
4. **`prepareGemmEnginesCUDA()`** — `packMoEExpertsCUDA()` → per-expert `CUDAQuantisedGemmKernel`
5. **`prepareGemmEnginesROCm()`** — `packMoEExpertsROCm()` → per-expert `ROCmQuantisedGemmKernel`
6. **`releaseRawWeights()`** — `MmapRegion::adviseDontneedRange()` after GPU upload
7. **`serializeExpert()` / `registerTransferredExpert()`** — Weight serialization for inter-socket transfer
8. **`releaseDepartedExperts()` / `registerAndPrepareNewExperts()` / `applyExpertMask()`** — Phased rebalance lifecycle
9. **NUMA auditing** — `queryNUMANode()`, `auditExpertNUMA()` (move_pages, sched_getcpu)

**Device-specific code lives here:** This is the only place with `#ifdef HAVE_CUDA`/`HAVE_ROCM` and CPU-specific includes. The service produces device-agnostic `ITensorGemm*` engines that stages consume.

### TPMode::ExpertParallel in GraphResolver

Currently a TODO stub at [GraphResolver.cpp line 470](../../src/v2/execution/local_execution/graph/GraphResolver.cpp):

```cpp
case TPMode::ExpertParallel:
    LOG_WARN("[GraphResolver] ExpertParallel not yet implemented");
    return std::nullopt;
```

**New behavior:** When a stage declares `TPMode::ExpertParallel`, the resolver inserts an `AllReduceStage` after it (same as `RowParallel`). The only difference from `RowParallel` is semantic — EP reduces partial expert outputs rather than partial matmul results, but the collective operation is identical (`MPI_SUM` or `NCCL allreduce`).

This replaces the imperative `FusedAddAllreduceStage` wiring in `Qwen35MoEGraph::buildFFNGraph()`. The graph builder no longer needs to know about TP collective insertion for MoE.

---

## Buffer Flow Diagram

```
                   BufferId::NORMALIZED
                          │
            ┌─────────────┼──────────────┐
            │             │              │
            ▼             ▼              ▼
    MoERoutingStage  SharedExpertFFN  SharedExpertGate
            │             │              │
            ▼             │              │
  MOE_ROUTING_INDICES     │              │
  MOE_ROUTING_WEIGHTS     │              │
            │             │              │
            ▼             │              │
  MoEExpertComputeStage   │              │
            │             │              │
            ▼             ▼              ▼
  MOE_COMBINED_OUTPUT  MOE_SHARED_EXPERT_OUTPUT
            │             │
            ▼             ▼
    [AllReduceStage]  [AllReduceStage]    ← auto-inserted by GraphResolver
            │             │
            └──────┬──────┘
                   ▼
           ResidualAddStage
                   │
                   ▼
           BufferId::ATTN_PROJ
```

---

## Phased Implementation Plan

### Phase 1: Extract `MoEExpertWeightService`

**Goal:** Move all weight lifecycle code out of `MoEFFNStage` into a standalone service. No behavioral change — the stage calls the service instead of doing weight work inline.

**Files to create:**
- `src/v2/execution/moe/MoEExpertWeightService.h`
- `src/v2/execution/moe/MoEExpertWeightService.cpp`

**Files to modify:**
- `src/v2/execution/compute_stages/stages/MoEFFNStage.h` — Remove static methods `extractExpertViews`, `prepareExpertGemmEngines`; remove phased rebalance API; remove `releaseRawExpertWeights`; remove `registerTransferredExpert`; remove `serializeExpert`; remove `detachAndSerializeExpert`
- `src/v2/execution/compute_stages/stages/MoEFFNStage.cpp` — Remove ~600 lines of weight lifecycle code; call `MoEExpertWeightService` where graph builders previously called static methods
- `src/v2/models/qwen35moe/Qwen35MoEGraph.cpp` — Call `MoEExpertWeightService::extractExpertViews()` and `prepareGemmEngines()` instead of `MoEFFNStage::extractExpertViews()` and `MoEFFNStage::prepareExpertGemmEngines()`
- `src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp` — Update any references to MoEFFNStage rebalance API to use MoEExpertWeightService
- CMakeLists.txt — Add new source files

**Methods to extract:**

| From MoEFFNStage | To MoEExpertWeightService |
|-------------------|---------------------------|
| `extractExpertViews(Params&)` (static) | `extractExpertViews(WeightContext&)` |
| `prepareExpertGemmEngines(Params&)` (static) | `prepareGemmEngines(WeightContext&)` |
| `prepareExpertGemmEnginesCUDA(Params&)` (static) | `prepareGemmEnginesCUDA(WeightContext&)` |
| `prepareExpertGemmEnginesROCm(Params&)` (static) | `prepareGemmEnginesROCm(WeightContext&)` |
| `releaseRawExpertWeights()` | `releaseRawWeights(WeightContext&)` |
| `detachAndSerializeExpert(int)` | `detachAndSerializeExpert(WeightContext&, int)` |
| `serializeExpert(int)` | `serializeExpert(const WeightContext&, int)` |
| `registerTransferredExpert(int, blobs)` | `registerTransferredExpert(WeightContext&, int, blobs)` |
| `updateExpertMaskAndPrepareEngines(mask, weights)` | `updateMaskAndPrepare(WeightContext&, mask, weights)` |
| `releaseDepartedExperts(mask)` | `releaseDepartedExperts(WeightContext&, mask)` |
| `registerAndPrepareNewExperts(mask, weights)` | `registerAndPrepareNewExperts(WeightContext&, mask, weights)` |
| `applyExpertMask(mask)` | via `MoEFFNStage::updateExpertMask(mask)` (kept in stage) |
| NUMA audit helpers | `queryNUMANode()`, `auditExpertNUMA()` |

**WeightContext struct** — lightweight reference struct pointing to the Params fields that the service operates on:
```cpp
struct MoEWeightContext {
    DeviceId device_id;
    int num_experts, expert_intermediate, d_model;
    int local_expert_start, local_expert_count;
    std::vector<bool>& expert_mask;
    TensorBase* gate_exps;   // 3D packed
    TensorBase* up_exps;     // 3D packed  
    TensorBase* down_exps;   // 3D packed
    std::vector<std::shared_ptr<TensorBase>>& expert_gate_views;
    std::vector<std::shared_ptr<TensorBase>>& expert_up_views;
    std::vector<std::shared_ptr<TensorBase>>& expert_down_views;
    std::vector<ITensorGemm*>& prepared_gate_gemm;
    std::vector<ITensorGemm*>& prepared_up_gemm;
    std::vector<ITensorGemm*>& prepared_down_gemm;
    std::vector<std::shared_ptr<ITensorGemm>>& moe_owned_kernels;
    std::shared_ptr<void>& moe_packed_gate_lifetime;
    std::shared_ptr<void>& moe_packed_up_lifetime;
    std::shared_ptr<void>& moe_packed_down_lifetime;
};
```

**Tests to write:**
- `tests/v2/unit/execution/moe/Test__MoEExpertWeightService.cpp`
  - Test `extractExpertViews` with mock 3D tensors (verify correct 2D slicing)
  - Test `prepareGemmEngines` calls KernelFactory for each view
  - Test `releaseDepartedExperts` nulls correct engine pointers
  - Test `registerAndPrepareNewExperts` only preps newly-acquired experts
  - Test `serializeExpert` / `registerTransferredExpert` round-trip

**Exit criteria:**
- All existing MoE parity tests pass unchanged
- All existing unit tests pass
- New unit tests cover weight service lifecycle
- `MoEFFNStage.cpp` loses ~600 lines

---

### Phase 2: Extract `MoERoutingStage`

**Goal:** Split routing logic (top-k selection, EP masking, histogram recording) into a standalone stage. Wire as two stages in the graph builder.

**Files to create:**
- `src/v2/execution/compute_stages/stages/MoERoutingStage.h`
- `src/v2/execution/compute_stages/stages/MoERoutingStage.cpp`

**Files to modify:**
- `src/v2/execution/compute_stages/stages/MoEFFNStage.h` — Remove routing-related Params fields (`gate_weights`, `norm_topk_prob`, `decode_histogram`, `layer_idx`, `replica_set`, `my_socket_id`); add `routing_indices`/`routing_weights` input fields
- `src/v2/execution/compute_stages/stages/MoEFFNStage.cpp` — Remove routing code from `execute()` and `executeSingleToken()`; read routing results from input buffers
- `src/v2/execution/compute_stages/ComputeStageFactory.h/.cpp` — Wire `createMoERouter()` to `MoERoutingStage`
- `src/v2/models/qwen35moe/Qwen35MoEGraph.cpp` — Add `MoERoutingStage` node before `MoEFFNStage` node; add dependency edge
- `src/v2/memory/BufferId.h` — Add `MOE_ROUTING_INDICES`, `MOE_ROUTING_WEIGHTS`
- `src/v2/models/qwen35moe/Qwen35MoESchema.h` — Declare routing buffer specs

**Routing output format:**
- `routing_indices`: `int32_t [seq_len × top_k]` — flat array, row-major
- `routing_weights`: `float [seq_len × top_k]` — normalized weights (0 for non-local experts under EP)

**Tests to write:**
- `tests/v2/unit/execution/stages/Test__MoERoutingStage.cpp`
  - Test basic routing with mock IMoEKernel
  - Test EP mask zeroing (non-local experts get weight 0)
  - Test histogram recording
  - Test replica dispatch annotation
  - Test `allowsZeroOutput()` returns false (routing always produces output)

**Exit criteria:**
- All MoE parity tests pass
- Routing stage is independently testable with mock kernel
- Graph builder wires two-stage pipeline

---

### Phase 3: Rename to `MoEExpertComputeStage` + cleanup

**Goal:** Rename the remaining `MoEFFNStage` to `MoEExpertComputeStage`. Remove all device-specific `#include`s. The stage becomes pure orchestration over `ITensorGemm` and `IMoEKernel`.

**Files to modify:**
- `src/v2/execution/compute_stages/stages/MoEFFNStage.h` → rename to `MoEExpertComputeStage.h`
- `src/v2/execution/compute_stages/stages/MoEFFNStage.cpp` → rename to `MoEExpertComputeStage.cpp`
- Remove `#include` of `CPUNativeVNNIGemmKernel.h`, `CPUPackedWeights.h`, `numaif.h`, `sched.h`
- Remove `#ifdef HAVE_CUDA` / `#ifdef HAVE_ROCM` blocks (already extracted to service in Phase 1)
- Update all references in graph builders, factory, tests
- Wire `createMoEExpert()` factory method to `MoEExpertComputeStage`

**Tests to write:**
- `tests/v2/unit/execution/stages/Test__MoEExpertComputeStage.cpp`
  - Test prefill path with mock ITensorGemm engines
  - Test decode path (single-token fast path)
  - Test EP partial output (some experts skipped → zero regions)
  - Test `allowsZeroOutput()` returns true under EP

**Exit criteria:**
- `MoEExpertComputeStage` has zero device-specific includes
- All parity tests pass
- File renamed, all references updated

---

### Phase 4: Implement `TPMode::ExpertParallel` in GraphResolver

**Goal:** The `GraphResolver::resolveTPCollective()` stub becomes a real implementation. When a stage declares `TPMode::ExpertParallel`, the resolver auto-inserts an `AllReduceStage` after it.

**Files to modify:**
- `src/v2/execution/local_execution/graph/GraphResolver.cpp` — Implement `ExpertParallel` case in `resolveTPCollective()` (insert AllReduce, same as RowParallel)
- `src/v2/models/qwen35moe/Qwen35MoEGraph.cpp` — Simplify `buildFFNGraph()`: remove imperative `FusedAddAllreduceStage` wiring; let resolver handle collective insertion
- `src/v2/execution/compute_stages/stages/MoEExpertComputeStage.h` — Add `requiresAllreduce()` override returning true when EP is active; or declare `TPMode::ExpertParallel` via schema

**Design decision — two options:**

*Option A: Stage declares TPMode*
```cpp
// In schema
stage("moe_expert_compute", TPMode::ExpertParallel);
// Resolver sees ExpertParallel → inserts AllReduce after
```

*Option B: Stage overrides `requiresAllreduce()`*
```cpp
bool MoEExpertComputeStage::requiresAllreduce() const override {
    return params_.local_expert_count >= 0;  // EP active
}
```

Option A is preferred because it keeps the TP mode visible in the declarative schema, making it inspectable by the graph compiler and profiler.

**Tests to write:**
- `tests/v2/unit/execution/graph/Test__GraphResolver_ExpertParallel.cpp`
  - Test resolver inserts AllReduce after ExpertParallel stage
  - Test resolver does NOT insert AllReduce when TPMode::None
  - Test resolver handles mixed ExpertParallel + ColumnParallel in same graph

**Exit criteria:**
- `Qwen35MoEGraph::buildFFNGraph()` no longer manually creates `FusedAddAllreduceStage`
- GraphResolver handles EP collective insertion
- All parity tests pass

---

### Phase 5: Comprehensive unit tests

**Goal:** Full test coverage for all new stages and the weight service, with mocked kernels.

**Test files:**
- `tests/v2/unit/execution/moe/Test__MoEExpertWeightService.cpp` (from Phase 1)
- `tests/v2/unit/execution/stages/Test__MoERoutingStage.cpp` (from Phase 2)
- `tests/v2/unit/execution/stages/Test__MoEExpertComputeStage.cpp` (from Phase 3)
- `tests/v2/unit/execution/graph/Test__GraphResolver_ExpertParallel.cpp` (from Phase 4)
- `tests/v2/unit/execution/moe/Test__MoERoutingToCompute_Integration.cpp` — End-to-end two-stage pipeline test with mock kernels

**Mock infrastructure needed:**
- `MockIMoEKernel` — mock `route()`, `gatherTokenBatch()`, `scatterAddWeighted()`, `swiGLU()`
- `MockITensorGemm` — mock `multiply_tensor()`, `multiply_fused_tensor()`
- Use existing `TestTensorFactory` for tensor creation

**Exit criteria:**
- All new code has unit tests
- Tests run in `build_v2_integration` under `V2_Unit_*` prefix
- No GPU required for unit tests

---

## File Inventory

### New files (created)

| File | Phase | Purpose |
|------|-------|---------|
| `src/v2/execution/moe/MoEExpertWeightService.h` | 1 | Weight lifecycle service header |
| `src/v2/execution/moe/MoEExpertWeightService.cpp` | 1 | Weight lifecycle service impl |
| `src/v2/execution/compute_stages/stages/MoERoutingStage.h` | 2 | Routing stage header |
| `src/v2/execution/compute_stages/stages/MoERoutingStage.cpp` | 2 | Routing stage impl |
| `tests/v2/unit/execution/moe/Test__MoEExpertWeightService.cpp` | 1 | Weight service tests |
| `tests/v2/unit/execution/stages/Test__MoERoutingStage.cpp` | 2 | Routing stage tests |
| `tests/v2/unit/execution/stages/Test__MoEExpertComputeStage.cpp` | 3 | Expert compute tests |
| `tests/v2/unit/execution/graph/Test__GraphResolver_ExpertParallel.cpp` | 4 | Resolver EP tests |
| `tests/v2/unit/execution/moe/Test__MoERoutingToCompute_Integration.cpp` | 5 | Pipeline integration test |

### Modified files

| File | Phases | Change |
|------|--------|--------|
| `src/v2/execution/compute_stages/stages/MoEFFNStage.h` | 1,2,3 | Progressively slimmed, then renamed |
| `src/v2/execution/compute_stages/stages/MoEFFNStage.cpp` | 1,2,3 | Progressively slimmed, then renamed |
| `src/v2/models/qwen35moe/Qwen35MoEGraph.cpp` | 1,2,4 | Updated stage creation calls |
| `src/v2/execution/compute_stages/ComputeStageFactory.h` | 2 | Wire factory methods |
| `src/v2/execution/compute_stages/ComputeStageFactory.cpp` | 2 | Wire factory methods |
| `src/v2/memory/BufferId.h` | 2 | Add routing buffer IDs |
| `src/v2/models/qwen35moe/Qwen35MoESchema.h` | 2 | Add routing buffer specs |
| `src/v2/execution/local_execution/graph/GraphResolver.cpp` | 4 | Implement EP case |
| `src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp` | 1 | Update rebalance calls |
| CMakeLists.txt (src + tests) | 1,2 | Add new sources |

### Renamed files

| Old | New | Phase |
|-----|-----|-------|
| `MoEFFNStage.h` | `MoEExpertComputeStage.h` | 3 |
| `MoEFFNStage.cpp` | `MoEExpertComputeStage.cpp` | 3 |

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Routing output format mismatch between stages | Use typed BufferIds with explicit shape contracts; unit test round-trip |
| Dynamic rebalancing breaks after weight service extraction | Phase 1 preserves exact call sites; integration tests verify rebalance cycle |
| GPU batch packing regression | Existing CUDA/ROCm parity tests catch regressions |
| Graph resolver EP collective wrong type | EP uses same AllReduce(SUM) as RowParallel; unit test resolver output |
| Performance regression from stage boundary overhead | Routing stage is ~1μs (just top-k + mask); negligible vs expert GEMM |

## Validation Strategy

After each phase:
1. `ctest --test-dir build_v2_integration -R "^V2_Unit_" --output-on-failure --parallel`
2. `ctest --test-dir build_v2_integration -R "^V2_Parity_" --output-on-failure --verbose` (includes MoE parity)
3. `ctest --test-dir build_v2_integration -R "^V2_Integration_" --output-on-failure`

After Phase 4:
4. Full MoE E2E test: `ctest --test-dir build_v2_e2e_release -R "Qwen35MoE" --verbose`
