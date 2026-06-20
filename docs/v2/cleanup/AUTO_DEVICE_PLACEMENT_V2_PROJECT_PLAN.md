# AUTO Device Placement V2 Project Plan

**Date**: 2026-05-28
**Status**: Proposed
**Scope**: Consolidate Llaminar's older MPI placement richness into the V2 `OrchestrationRunner` execution reality, starting with memory-aware AUTO device selection and ending with one shared planner for CLI analysis and runtime execution.

---

## Summary

Llaminar currently has two parallel planning systems:

1. The V2 runtime path: `OrchestrationRunner` builds a `RankExecutionPlan`, then dispatches to single-device, local TP, or local PP graph construction.
2. The older MPI placement path: `PlacementStrategy`, `PlacementPlan`, `WeightPlacementMap`, and `MPITopology::computePlacement()` contain richer placement ideas such as GPU-first memory fitting, hybrid CPU/GPU decode participation, heterogeneous domains, and deterministic cluster-wide planning.

The immediate bug is narrow: when a model is selected on the command line and no device flags are provided, simple V2 AUTO placement chooses the first enumerated GPU and waits until memory validation to fail. The larger cleanup is architectural: preserve the useful placement intelligence from the older path, but make V2's `RankExecutionPlan` the runtime contract.

This plan moves in phases. Each phase must leave the tree in a coherent state, add tests using full MPI-gathered inventory mocks, and make runtime AUTO more capable without changing explicit user configurations.

---

## Current Findings

### V2 Runtime Contract

`OrchestrationRunner` is the canonical runtime entry point. During initialization it:

1. gathers `ClusterInventory`,
2. reads model metadata,
3. asks `ExecutionPlanBuilder` for a `RankExecutionPlan`,
4. creates TP/PP contexts,
5. loads weights,
6. validates memory,
7. builds the graph from the plan.

The graph dispatch depends on `RankExecutionPlan` fields:

- `primary_device` for single-device execution,
- `local_tp_devices` and `local_tp_weights` for local tensor parallelism,
- `local_pp_devices`, `local_pp_layer_boundaries`, and `local_pp_stage_tp_info` for local pipeline parallelism and PP+TP composition,
- global TP participation fields for cross-rank TP.

This means V2 AUTO should produce or enrich `RankExecutionPlan`. `WeightPlacementMap` can remain a loader-side helper, but it should not become the V2 runtime source of truth.

### Existing Richer Placement Machinery

The older placement machinery is real and useful:

- `PlacementInput` can describe model shape, quantization, cluster inventory, aggregate memory, and bandwidth.
- `LayerPlacementStrategyFactory::autoSelect()` chooses among CPU-only, GPU-first, hybrid optimal, and heterogeneous multi-domain strategies.
- `GPUFirstLayerPlacementStrategy` already models memory fitting and GPU spillover.
- `HybridOptimalLayerPlacementStrategy` models CPU participation in decode.
- `HeterogeneousMultiDomainStrategy` models cross-vendor and multi-domain assignments.
- `WeightPlacementMap::applyPlan()` can consume `PlacementPlan` for loader decisions.
- `MPITopology::computePlacement()` still invokes this path deterministically.

The issue is that normal V2 inference does not call this machinery when no device flags are specified.

### Newer Memory Planning Machinery

The `llaminar2 plan` command already has another piece of the answer:

- it reads GGUF metadata into `ModelMemoryProfile`,
- evaluates candidate memory plans with `MemoryPlanner`,
- enumerates single GPU, TP, PP, and CPU-only candidates,
- reports whether candidates fit using `DeviceInfo.free_memory_bytes`.

That logic is closer to what runtime AUTO needs than the older `PlacementPlan` contract, but it is currently isolated in the CLI command.

---

## Design Principles

1. **`RankExecutionPlan` is the V2 runtime contract.** AUTO must populate the fields V2 already executes.
2. **Explicit user configuration wins.** `-d`, `--tp-devices`, `--define-domain`, `--topology`, `--device-map`, and explicit TP/PP flags must keep their existing behavior.
3. **AUTO is deterministic.** All ranks must compute identical cluster-level decisions from identical inventory and model inputs.
4. **Memory feasibility comes before performance scoring.** A fast device that cannot fit the model is not a candidate.
5. **The planner must explain itself.** Candidate rejection and selection reasons should feed `--explain-placement`, `--dry-run`, and debug logs.
6. **CLI planning and runtime AUTO share the same pipeline.** `llaminar plan` and inline `llaminar serve` planning must call the same MPI gather/probe, planner, YAML materializer, and plan applier code paths.
7. **YAML is a serialized plan artifact, not a separate planning path.** A supplied orchestration plan YAML should deserialize into the same in-memory `OrchestrationPlan` object that inline planning would have produced.
8. **Old richness migrates by translation, not takeover.** Reuse `PlacementStrategy` ideas, but translate expressible outcomes into V2 plan fields.
9. **Tests use full gathered inventories.** Unit tests must build `ClusterInventory` through a gather-equivalent mock path, not by stuffing a minimal `gpus` vector into one rank.

---

## Target End State: One Planning Pipeline

The desired end state is not merely shared helper functions. The desired end state is one planning pipeline with two front doors.

```text
        ┌────────────────────────────────────────────────────┐
        │                 Planning Inputs                    │
        │  CLI/YAML config + GGUF metadata + MPI inventory   │
        └───────────────────────┬────────────────────────────┘
                    │
                    ▼
        ┌────────────────────────────────────────────────────┐
        │ PlanningProbe                                      │
        │ - MPI gather of all rank inventories               │
        │ - model metadata/profile extraction                │
        │ - optional bandwidth/topology probes               │
        └───────────────────────┬────────────────────────────┘
                    │
                    ▼
        ┌────────────────────────────────────────────────────┐
        │ SharedOrchestrationPlanner                         │
        │ - candidate enumeration                            │
        │ - MemoryPlanner validation                         │
        │ - scoring and diagnostics                          │
        └───────────────────────┬────────────────────────────┘
                    │
                    ▼
        ┌────────────────────────────────────────────────────┐
        │ OrchestrationPlan                                  │
        │ - schema/version/planner metadata                  │
        │ - model/profile fingerprint                        │
        │ - inventory/probe fingerprint                      │
        │ - selected candidate and rejected candidates        │
        │ - per-rank RankExecutionPlan payloads              │
        │ - memory plan and explanation reports              │
        └───────────────┬────────────────────┬───────────────┘
                │                    │
                ▼                    ▼
          YAML serialize/parse      OrchestrationPlanApplier
                           │
                           ▼
                       OrchestrationRunner
```

The two user-facing modes differ only in how the `OrchestrationPlan` object is obtained:

```text
llaminar plan -m model.gguf ...
  -> bootstrap MPI
  -> PlanningProbe::gatherAndProbe(...)
  -> SharedOrchestrationPlanner::plan(...)
  -> OrchestrationPlanYaml::write(...)

llaminar serve --config orchestration-plan.yaml ...
  -> OrchestrationPlanYaml::read(...)
  -> optional PlanningProbe::gatherAndProbe(...) for validation/freshness
  -> OrchestrationPlanApplier::planForRank(...)
  -> OrchestrationRunner::initializeFromPlan(...)

llaminar serve -m model.gguf ...        # no supplied plan
  -> bootstrap MPI
  -> PlanningProbe::gatherAndProbe(...)
  -> SharedOrchestrationPlanner::plan(...)
  -> OrchestrationPlanApplier::planForRank(...)
  -> OrchestrationRunner::initializeFromPlan(...)
```

This is the critical invariant: once an `OrchestrationPlan` exists, supplied-plan serve and inline-plan serve must exercise the same applier, validation, weight loading, context creation, and graph-building paths. The only difference is whether the plan came from YAML deserialization or from a just-run MPI gather/probe/planner round.

### Proposed Core Types

```cpp
struct PlanningSnapshot
{
  OrchestrationConfig config;
  ModelMemoryProfile model_profile;
  ClusterInventory cluster_inventory;
  std::vector<DeviceProbeResult> device_probes;
  std::string model_fingerprint;
  std::string inventory_fingerprint;
};

struct OrchestrationPlan
{
  int schema_version = 1;
  std::string planner_version;
  std::string model_fingerprint;
  std::string inventory_fingerprint;
  AutoPlacementResult placement;
  std::vector<RankExecutionPlan> rank_plans;
};
```

The YAML file should serialize `OrchestrationPlan`, not a loosely related config. If a user edits YAML manually, the parser should still produce the same in-memory `OrchestrationPlan` object before execution begins.

### Required Same-Path Tests

Every major planning feature should include three equivalence tests:

1. `plan` mode builds an `OrchestrationPlan` from gathered inventory and writes YAML.
2. `serve --config generated.yaml` reads the YAML and produces the same per-rank `RankExecutionPlan` fingerprints.
3. `serve` without YAML gathers/probes/plans inline and produces the same `OrchestrationPlan` fingerprint as `plan` mode for the same inputs.

Optional validation for supplied YAML may gather the current inventory and compare fingerprints, but it must not use a separate planner path.

When a later phase adds a new candidate kind, that candidate kind is not complete until these same-path tests exist for it.

---

## Test Harness Standard

Before feature phases begin, establish a reusable inventory mock harness. Every phase below depends on it.

### Full MPI-Gathered Inventory Mock

Create a test utility, for example:

```text
tests/v2/utils/MockClusterInventoryBuilder.h
tests/v2/utils/MockClusterInventoryBuilder.cpp
```

The helper should model the same data shape produced by `gatherClusterInventory()`:

1. Build one complete `RankInventory` per rank.
2. Populate CPU socket information, CPU memory, GPU total/free memory, NUMA node, device type, ordinal, name, optional bandwidth, and optional P2P matrices.
3. Pass each rank inventory through the same serialization/deserialization helpers used by MPI inventory exchange, or through a test-only gather shim that exercises equivalent serialization boundaries.
4. Assign deterministic node IDs from hostnames.
5. Call `ClusterInventory::buildNodeAggregations()`.
6. Return the final `ClusterInventory` that all ranks would see after `MPI_Allgatherv`.

The point is to catch bugs that only appear when inventory is rank-indexed, host-indexed, NUMA-scoped, or cross-rank. Tests should not rely on a local-only shortcut unless the scenario itself is local-only.

### Standard Inventory Scenarios

Create named helpers for common fixtures:

- `cpuOnlyTwoSocketNode()`
- `singleCuda24GB()`
- `singleRocm32GB()`
- `mixed3090AndFourMI60()`
- `asymmetricCudaPair24GBAnd48GB()`
- `sameVendorFourGpuNode()`
- `crossVendorCudaRocmNode()`
- `twoNodeOneGpuPerRank()`
- `twoNodeCpuAndGpuHybrid()`
- `freeMemoryConstrainedNode()`
- `bandwidthRankedMixedNode()`

Each helper should permit overriding:

- total memory,
- free memory,
- memory bandwidth,
- device names,
- NUMA node,
- rank-local visibility,
- P2P capabilities.

### Required Test Pattern

Every phase should include tests with this shape:

```cpp
auto cluster = MockClusterInventoryBuilder()
    .addNode(...)
    .addRank(...)
    .addGPU(...)
    .gatherLikeMPI();

for (int rank = 0; rank < cluster.world_size; ++rank)
{
    auto plan = planner.buildPlanForRank(config, model_profile, cluster, rank);
    ASSERT_TRUE(plan.validate().empty());
    // phase-specific assertions
}
```

For multi-rank scenarios, tests must build all rank plans and assert cross-rank consistency, not only inspect rank 0.

---

## Phase 0: Baseline Characterization and Test Harness

**Goal**: Lock in the current behavior and create the gathered-inventory test foundation.

### Implementation

- Add `MockClusterInventoryBuilder` and named scenario helpers.
- Add helpers for synthetic model profiles:
  - 0.5B class,
  - 7B class,
  - 27B class,
  - 70B class,
  - MoE profile with expert-heavy tensors.
- Add utilities to create both `ModelConfig` and `ModelMemoryProfile` from the same synthetic profile so tests can cover both current and target planner APIs.
- Add a baseline test that proves current simple AUTO chooses `gpus[0]` when no device flags are set.
- Add baseline tests documenting that explicit modes are already handled outside AUTO.

### Tests

New tests:

```text
tests/v2/unit/execution/mpi_orchestration/Test__AutoPlacementBaseline.cpp
tests/v2/unit/utils/Test__MockClusterInventoryBuilder.cpp
```

Required cases:

- CPU-only gathered inventory yields CPU fallback in current simple planning.
- Mixed RTX 3090 + MI60 inventory currently chooses the first enumerated GPU.
- Explicit `-d` bypasses AUTO.
- Explicit `--tp-devices` bypasses AUTO.
- Named domains bypass AUTO.
- Multi-rank gathered inventory has identical node totals and rank counts on every test path.

### Exit Criteria

- Current behavior is documented by failing-or-passing baseline assertions.
- All later tests can use the same full gathered inventory mock helper.

---

## Phase 1: Shared Planning Pipeline and OrchestrationPlan Artifact

**Goal**: Extract a pure planning pipeline shared by `llaminar2 plan`, supplied-plan `llaminar serve`, and inline-plan `llaminar serve`.

### Implementation

Create planning modules under `src/v2/planning/` or `src/v2/execution/mpi_orchestration/`:

```text
AutoPlacementCandidate.h
AutoPlacementCandidate.cpp
AutoPlacementPlanner.h
AutoPlacementPlanner.cpp
PlanningSnapshot.h
PlanningProbe.h
PlanningProbe.cpp
OrchestrationPlan.h
OrchestrationPlanYaml.h
OrchestrationPlanYaml.cpp
OrchestrationPlanApplier.h
OrchestrationPlanApplier.cpp
```

The split of responsibilities should be explicit:

- `PlanningProbe` gathers MPI inventory, reads model metadata, and records optional probe measurements.
- `AutoPlacementPlanner` turns a `PlanningSnapshot` into an `AutoPlacementResult`.
- `OrchestrationPlan` owns the selected candidate, diagnostics, memory reports, and per-rank `RankExecutionPlan` payloads.
- `OrchestrationPlanYaml` serializes/deserializes that exact object.
- `OrchestrationPlanApplier` extracts and validates the current rank's `RankExecutionPlan` for `OrchestrationRunner`.

Suggested structures:

```cpp
enum class AutoPlacementKind
{
    SingleDevice,
    LocalTP,
    LocalPP,
    LocalPPTP,
    GlobalTP,
    GlobalPP,
    HeterogeneousMultiDomain,
    CPUOnly,
};

struct AutoPlacementCandidate
{
    AutoPlacementKind kind;
    std::string name;
    std::vector<GlobalDeviceAddress> devices;
    std::vector<DevicePlanConfig> memory_configs;
    std::vector<int> layer_boundaries;
    std::vector<float> weights;
    CollectiveBackendType backend;
    float estimated_score;
    std::vector<std::string> reasons;
};

struct AutoPlacementResult
{
    bool found_feasible;
    AutoPlacementCandidate selected;
    MemoryPlan selected_memory_plan;
    std::vector<AutoPlacementCandidateReport> reports;
};
```

Move candidate construction from `PlanCommand` into reusable functions:

- single-device candidates for every visible GPU,
- CPU-only fallback,
- local TP group candidates,
- local PP candidates,
- later phases can extend this without changing command code.

Add `OrchestrationPlan` materialization even before all candidate kinds are supported. At this phase it may carry only single-device and CPU-only candidates, but it must already be the object passed between planning, YAML, and execution.

At this phase, keep scoring simple and deterministic:

1. Reject candidates that fail `MemoryPlanner::plan()`.
2. Rank feasible GPU candidates ahead of CPU-only.
3. Within the same kind, rank by `memory_bandwidth_gbps` if available, then free memory, then total memory, then stable device order.

### Tests

New tests:

```text
tests/v2/unit/planning/Test__AutoPlacementCandidates.cpp
tests/v2/unit/planning/Test__OrchestrationPlanYaml.cpp
tests/v2/unit/execution/mpi_orchestration/Test__OrchestrationPlanApplier.cpp
```

Required cases using gathered inventory mocks:

- Every GPU in `mixed3090AndFourMI60()` becomes a single-device candidate.
- A GPU with insufficient `free_memory_bytes` is rejected even if total VRAM is large.
- CPU-only candidate is always generated and evaluated.
- Bandwidth-ranked candidates choose the faster fitting GPU.
- Stable tie-breaking is deterministic across repeated runs.
- Multi-rank gathered inventories produce the same candidate list on every rank.
- `OrchestrationPlanYaml` round-trips all per-rank `RankExecutionPlan` payloads without changing plan fingerprints.
- `OrchestrationPlanApplier` returns the same rank plan whether the plan object came from inline planning or YAML deserialization.
- Plan fingerprints include model and inventory fingerprints so stale supplied YAML can be detected.

### Exit Criteria

- `PlanCommand` can call the shared planning pipeline without changing output semantics.
- `OrchestrationRunner` can accept a prebuilt `OrchestrationPlan` through the applier in tests.
- Candidate reports contain enough information to explain why a device was rejected.
- YAML is proven to be a serialization format for the in-memory plan, not a parallel planning path.

---

## Phase 2: Conservative Runtime AUTO for Single Device and CPU Fallback

**Goal**: Fix the immediate AUTO bug through the shared `OrchestrationPlan` pipeline without enabling new parallel execution shapes yet.

### Implementation

- Add a narrow integration point where unspecified AUTO mode asks the shared planner for an `OrchestrationPlan`, then uses `OrchestrationPlanApplier` to obtain the current rank's `RankExecutionPlan`.
- Keep `ExecutionPlanBuilder::buildSimplePlan()` as the final explicit-mode builder, but do not add a separate runtime-only AUTO solver there.
- Detect unspecified AUTO with the same semantics as `ConfigValidator::detectDeviceSelectionMode()`:
  - no `device_for_this_rank`,
  - no `device_map`,
  - no `tp_devices`,
  - no simple `-tp N`,
  - no named domains,
  - no topology tree,
  - `device_mode == AUTO`.
- For this phase, ask the shared planner only for `SingleDevice` and `CPUOnly` candidates.
- Convert the selected candidate to `RankExecutionPlan.primary_device`.
- Store selection diagnostics in a lightweight plan explanation object, or log them until an explanation API lands.
- Do not auto-enable TP or PP in this phase.
- Add a `serve --config orchestration-plan.yaml` path, or a test-visible equivalent, that reads the YAML, applies the plan, and reaches the same `RankExecutionPlan` as inline AUTO.

### Tests

New tests:

```text
tests/v2/unit/execution/mpi_orchestration/Test__AutoDevicePlannerSingleDevice.cpp
tests/v2/unit/execution/runner/Test__ServePlanEquivalenceSingleDevice.cpp
```

Required cases using gathered inventory mocks:

- 7B class model on one 24 GB CUDA GPU selects CUDA.
- 27B class model on mixed 24 GB CUDA + 32 GB ROCm selects a fitting ROCm device when CUDA is too tight.
- Same model with CUDA free memory artificially reduced rejects CUDA and selects ROCm.
- If multiple GPUs fit, planner selects higher memory bandwidth.
- If no GPU fits, planner selects CPU.
- CPU-only machine selects CPU.
- Explicit `-d cuda:0` still chooses CUDA even if AUTO would choose ROCm; memory validation remains responsible for failure.
- Multi-rank gathered inventory remains deterministic and only assigns devices visible to the rank being planned.
- `llaminar plan` style planning and inline `serve` planning produce the same `OrchestrationPlan` fingerprint for single-device cases.
- `serve --config generated.yaml` and inline `serve` produce identical `RankExecutionPlan` fingerprints for every rank.

### Exit Criteria

- The original issue is fixed for no-device single-rank invocation.
- Explicit modes are untouched.
- Runtime memory validation should normally pass for the selected AUTO single-device candidate.
- Supplied-plan serve and inline-plan serve differ only in whether `OrchestrationPlan` was deserialized or freshly computed.

---

## Phase 3: Runtime AUTO for Local Tensor Parallelism

**Goal**: When no single GPU fits, allow AUTO to select local TP if V2 can execute it.

### Implementation

- Extend candidate enumeration to local TP groups:
  - same-backend groups first: CUDA-only, ROCm-only,
  - then cross-vendor groups only if the current local collective backend can support them,
  - prefer smaller TP degree that fits before larger TP degree unless score clearly favors larger.
- Use `MemoryPlanner` with TP shard configs for all devices in a group.
- Translate selected local TP candidate into:
  - `plan.local_tp_devices`,
  - `plan.local_tp_weights`, if heterogeneous proportional splitting is supported,
  - `plan.local_tp_backend`,
  - `plan.primary_device = local_tp_devices[0]`,
  - `plan.tp_scope = TPScope::LOCAL`,
  - `plan.weight_shard.total_shards = local_tp_devices.size()`.
- Reuse the existing `RankOrchestrator` path. Do not create a new executor.
- Keep global TP and local PP out of AUTO for this phase.

### Tests

New tests:

```text
tests/v2/unit/execution/mpi_orchestration/Test__AutoDevicePlannerLocalTP.cpp
```

Required cases using gathered inventory mocks:

- 70B class model on four 32 GB ROCm GPUs selects local TP when no single GPU fits.
- 70B class model on one 24 GB CUDA + four 32 GB ROCm GPUs selects ROCm-only TP before cross-vendor TP.
- Two 24 GB CUDA GPUs select TP-2 when the model does not fit on either alone but fits sharded.
- TP candidate rejection includes per-device memory deficits.
- `local_tp_backend` is NCCL for CUDA-only, RCCL for ROCm-only, and HOST or AUTO for cross-vendor according to existing backend selection.
- `RankExecutionPlan::validate()` passes for all generated plans.
- Existing explicit `-tp N` behavior is not changed by AUTO candidate scoring.
- Plan YAML round-trip and inline serve produce identical local TP rank-plan fingerprints.

### Exit Criteria

- AUTO can select local TP for memory feasibility.
- `OrchestrationRunner::buildMultiDeviceComputeGraph()` is reached only through existing plan fields.

---

## Phase 4: Runtime AUTO for Local Pipeline Parallelism

**Goal**: Preserve the older GPU-first layer distribution idea by translating simple layer-split plans into V2 local PP.

### Implementation

- Add local PP candidates for devices on the same rank.
- Generate equal layer splits first, then memory-balanced splits.
- Use `MemoryPlanner` per stage with stage-specific `first_layer` and `last_layer`.
- Translate selected local PP candidate into:
  - `plan.local_pp_devices`,
  - `plan.local_pp_layer_boundaries`,
  - `plan.local_pp_backend`,
  - `plan.primary_device = local_pp_devices[0]`,
  - stage embedding and LM-head ownership through boundaries.
- Keep local PP lower priority than local TP unless local TP cannot fit or backend compatibility makes TP undesirable.
- Do not yet support PP+TP composition in AUTO; that lands in Phase 5.

### Tests

New tests:

```text
tests/v2/unit/execution/mpi_orchestration/Test__AutoDevicePlannerLocalPP.cpp
```

Required cases using gathered inventory mocks:

- Model too large for one GPU but fitting as local PP across two GPUs selects local PP when TP is disabled or unsupported.
- Asymmetric 24 GB + 48 GB GPUs receive memory-balanced layer boundaries.
- Equal split is used when devices are symmetric.
- Boundaries cover all layers exactly once.
- Stage 0 owns embedding and last stage owns LM head through the generated plan.
- `OrchestrationRunner` would route to `buildLocalPPComputeGraph()` because `plan.usesLocalPP()` is true.
- Explicit named-domain PP remains unaffected.
- Plan YAML round-trip and inline serve produce identical local PP rank-plan fingerprints.

### Exit Criteria

- AUTO can express simple GPU-first spillover as V2 local PP.
- No older `PlacementPlan` is required at runtime for this case.

---

## Phase 5: AUTO PP+TP Composition and Heterogeneous Local Domains

**Goal**: Reach the first meaningful subset of V1 richness: local pipeline stages that can themselves be local TP domains.

### Implementation

- Add `LocalPPTP` candidates:
  - each PP stage can be a single device or a local TP group,
  - stage-level TP details populate `RankExecutionPlan::local_pp_stage_tp_info`,
  - global `local_tp_devices` must stay empty when local PP owns per-stage TP, preserving existing V2 dispatch semantics.
- Add domain grouping heuristics:
  - group same-backend devices together,
  - avoid cross-vendor TP inside a stage unless no same-backend solution fits,
  - use PP between vendor groups when possible.
- Prefer PP between CUDA and ROCm domains over cross-vendor TP for large models.
- Extend explanation output to show stage domains, layer ranges, and per-stage memory.

### Tests

New tests:

```text
tests/v2/unit/execution/mpi_orchestration/Test__AutoDevicePlannerLocalPPTP.cpp
```

Required cases using gathered inventory mocks:

- Mixed one CUDA + four ROCm node chooses ROCm local TP stage for most layers and CUDA single-device or CPU fallback only if useful.
- Two CUDA + two ROCm GPUs prefer two PP stages with same-backend TP inside each stage over four-way cross-vendor TP.
- Per-stage TP info is parallel to `local_pp_devices` and validates for each stage.
- `plan.local_tp_devices` remains empty when `plan.local_pp_stage_tp_info` contains stage TP groups.
- Memory validation creates per-stage per-shard `DevicePlanConfig` entries and passes.
- Candidate reports include rejected PP+TP shapes with reason strings.
- Plan YAML round-trip and inline serve produce identical local PP+TP rank-plan fingerprints.

### Exit Criteria

- V2 AUTO can express a practical heterogeneous local multi-domain plan without invoking the old executor path.
- The plan shape matches what `RankOrchestrator::Config::fromPlan()` already expects for local PP with nested TP.

---

## Phase 6: PlacementStrategy Adapter Layer

**Goal**: Reuse the old strategy intelligence directly, but only through an adapter that emits V2-executable candidates.

### Implementation

- Introduce an adapter that converts selected `PlacementPlan` outcomes into `AutoPlacementCandidate` or `RankExecutionPlan` where possible.
- Supported translations in this phase:
  - all layers on one GPU -> `SingleDevice`,
  - all layers on CPU -> `CPUOnly`,
  - layers distributed across local GPUs -> `LocalPP`,
  - same-layer same-rank multi-GPU sharding -> `LocalTP`, if represented,
  - local PP stages with same-backend TP -> `LocalPPTP`.
- Unsupported translations must fail closed with clear diagnostics:
  - phase-aware CPU+GPU decode participation,
  - cross-rank domains not represented in V2 plan fields yet,
  - per-layer attention/FFN split if the graph cannot execute it,
  - dynamic routing or residency concepts that require graph changes.
- Use the adapter as a candidate source, not as the only candidate source. The shared planner should still generate direct V2 candidates.
- Add a comparison mode in tests: direct V2 candidates vs adapted old strategies.

### Tests

New tests:

```text
tests/v2/unit/execution/mpi_orchestration/Test__PlacementPlanToRankExecutionPlan.cpp
```

Required cases using gathered inventory mocks:

- `CPUOnlyLayerPlacementStrategy` adapts to CPU `primary_device`.
- `GPUFirstLayerPlacementStrategy` all-on-one-GPU adapts to single-device.
- `GPUFirstLayerPlacementStrategy` multi-GPU layer distribution adapts to local PP when layers are contiguous per device.
- Non-contiguous layer placement is rejected with a diagnostic instead of silently producing invalid PP boundaries.
- `HybridOptimalLayerPlacementStrategy` CPU decode participation is recognized but marked unsupported for V2 runtime until Phase 8.
- `HeterogeneousMultiDomainStrategy` local same-rank domains adapt where expressible and reject cross-rank domains until Phase 7.
- All adapter outputs validate as `RankExecutionPlan` or fail closed.

### Exit Criteria

- Old placement strategies can contribute to V2 AUTO when their output matches V2 execution capabilities.
- Unsupported old richness is visible in diagnostics rather than lost.

---

## Phase 7: Cross-Rank AUTO Domains

**Goal**: Bring global TP and global PP AUTO decisions into the V2 planning model.

### Implementation

- Extend candidates to include cross-rank shapes:
  - one device per rank global TP,
  - multi-rank PP with one rank per stage,
  - local TP inside rank plus global PP between ranks,
  - global TP stages inside global PP where supported by `GlobalOrchestrator` or equivalent runtime.
- Translate candidates into existing V2 global fields when available:
  - `global_tp_domain_id`,
  - `global_tp_rank_in_domain`,
  - `global_tp_domain_size`,
  - `prev_rank`,
  - `next_rank`,
  - rank-specific layer ranges,
  - cross-rank backend.
- If the current runtime cannot execute a richer domain shape, keep it candidate-only and mark it blocked by executor support.
- Ensure rank 0 and all worker ranks derive compatible plans from the same gathered inventory.

### Tests

New tests:

```text
tests/v2/unit/execution/mpi_orchestration/Test__AutoDevicePlannerGlobalDomains.cpp
tests/v2/integration/utils/mpi/Test__AutoPlacement_MPI_Integration.cpp
```

Required cases using gathered inventory mocks:

- Two ranks with one GPU each select global TP when no local candidate fits and global TP does.
- Four ranks across two nodes select PP stages with correct `prev_rank` and `next_rank`.
- Rank-specific `first_layer` and `last_layer` ranges cover all layers once for PP.
- Global TP rank indices are stable and within domain size.
- Cross-rank backend selection is MPI unless a more specific backend is explicitly available.
- All ranks compute the same selected global candidate identity.
- MPI integration test verifies deterministic plan fingerprints across real `mpirun -np 2`.
- `plan` mode YAML, supplied-plan serve, and inline-plan serve produce identical per-rank plans under real `mpirun -np 2`.

### Exit Criteria

- AUTO can reason over the full gathered cluster, not only rank-local devices.
- Cross-rank plans are either executable or explicitly marked as blocked with diagnostics.

---

## Phase 8: Phase-Aware Decode and CPU Participation

**Goal**: Recover the old hybrid decode insight in V2: CPU memory bandwidth can participate in decode while GPU dominates prefill.

### Implementation

- Define an explicit V2 runtime representation for phase-aware placement. Do not hide it inside `WeightPlacementMap` only.
- Options:
  - extend `RankExecutionPlan` with per-layer decode participants,
  - add a model graph planning layer that consumes an immutable placement table,
  - or introduce a `RuntimePlacementPlan` owned by `OrchestrationRunner` and passed to graph builders.
- Map old `LayerPlacement::decode_devices`, `decode_weight_fractions`, and `cpu_participates_in_decode` into the new V2 representation.
- Teach graph builders or executors to honor phase-aware decode only where kernels and collectives support it.
- Keep phase-aware plans disabled by default until correctness and performance are proven.

### Tests

New tests:

```text
tests/v2/unit/execution/mpi_orchestration/Test__AutoDevicePlannerPhaseAwareDecode.cpp
tests/v2/integration/orchestration/Test__PhaseAwareDecodePlacement.cpp
```

Required cases using gathered inventory mocks:

- Hybrid strategy output with CPU decode participation maps to the new V2 representation.
- CPU decode participants are not accidentally used during prefill.
- Device fractions sum to 1.0 per layer.
- CPU participation is omitted when CPU bandwidth is below threshold.
- Generated graph or config rejects phase-aware placement when a required kernel path is unavailable.
- A small integration model confirms prefill and decode select the expected execution paths.
- Plan YAML round-trip and inline serve preserve phase-aware decode placement exactly.

### Exit Criteria

- The main missing piece of old `HybridOptimalLayerPlacementStrategy` has a real V2 execution surface.
- Hybrid decode remains opt-in or guarded until parity and performance are acceptable.

---

## Phase 9: Planner Explanation, Dry Run, and CLI Hardening

**Goal**: Make runtime AUTO inspectable and harden the already-shared `llaminar2 plan` front end.

### Implementation

- Add structured planner reports:
  - candidate name,
  - devices,
  - memory plan table,
  - score,
  - rejection reasons,
  - selected reason.
- Wire reports into:
  - `--explain-placement`,
  - `--dry-run`,
  - `llaminar2 plan --format table|yaml|json`,
  - startup banner summary.
- Ensure `PlanCommand` has no fallback candidate enumeration outside the shared planner.
- Add YAML generation for selected V2 plan shapes through `OrchestrationPlanYaml` only.
- Where runtime AUTO selects a shape that YAML cannot currently represent, treat that as a schema gap and fail the phase rather than adding an alternate output path.

### Tests

New tests:

```text
tests/v2/unit/app/commands/Test__PlanCommandAutoPlanner.cpp
tests/v2/unit/execution/mpi_orchestration/Test__AutoPlacementExplanation.cpp
```

Required cases using gathered inventory mocks:

- `PlanCommand` and `ExecutionPlanBuilder` select the same candidate for the same model and inventory.
- `--explain-placement` includes rejected first GPU reason for the mixed 3090 + MI60 case.
- JSON output contains all candidate reports and selected candidate fields.
- YAML output is emitted for single-device, local TP, and local PP plans.
- Dry run does not load full weights or build a graph.
- Candidate report order is stable for snapshot-style tests.
- `PlanCommand` tests assert that command output came from `OrchestrationPlanYaml`, not command-local YAML construction.

### Exit Criteria

- Users can see why AUTO picked a device or parallelism shape.
- CLI plan analysis and runtime planning cannot drift silently.

---

## Phase 10: Consolidation and Legacy Path Cleanup

**Goal**: End with one planner brain and clear ownership boundaries.

### Implementation

- Decide final status of the older APIs:
  - keep as strategy plugins behind the shared planner,
  - or mark direct `MPITopology::computePlacement()` usage as legacy.
- Remove duplicate candidate enumeration from command code and tests.
- Document the planning pipeline in the V2 architecture instructions:

```text
CLI/YAML -> OrchestrationConfig
Model metadata -> ModelMemoryProfile
Cluster gather -> ClusterInventory
Shared AUTO planner -> AutoPlacementResult
AutoPlacementResult -> RankExecutionPlan
RankExecutionPlan -> contexts, weight loading, graph build
```

- Add a migration note explaining which old `PlacementPlan` concepts are supported in V2 and which require executor work.
- Update developer docs so future model additions know where placement policy belongs.

### Tests

Required cases using gathered inventory mocks:

- No test depends on `rank_inv.gpus[0]` fallback for AUTO.
- `llaminar2 plan` and runtime AUTO share golden candidate fingerprints.
- Old strategy adapter tests still pass or are explicitly marked legacy.
- Multi-rank deterministic tests cover both mock-gathered and real MPI paths.
- Config validation tests prove explicit device selection modes still bypass AUTO.

### Exit Criteria

- There is one shared AUTO planner used by both runtime and CLI analysis.
- V2 executes through `RankExecutionPlan` only.
- Older richness is either supported through adapters or documented as awaiting runtime executor support.

---

## Suggested Implementation Order

1. Phase 0: inventory mock harness and baseline tests.
2. Phase 1: shared planning pipeline, candidate model, `OrchestrationPlan`, YAML round-trip, and applier.
3. Phase 2: single-device/CPU runtime AUTO.
4. Phase 3: local TP runtime AUTO.
5. Phase 9 partial: explanation output for the shapes already supported.
6. Phase 4: local PP runtime AUTO.
7. Phase 5: local PP+TP and heterogeneous local domains.
8. Phase 6: old strategy adapter.
9. Phase 7: cross-rank domains.
10. Phase 8: phase-aware decode.
11. Phase 10: consolidation.

This order fixes the user-visible bug early, reduces code duplication early, and postpones executor-sensitive richness until the planner has a stable representation and test bed.

---

## Acceptance Matrix

| Capability | Phase | Runtime Path | Test Requirement |
| --- | ---: | --- | --- |
| Full MPI-gathered inventory mocks | 0 | Test utility | Mock gather/serialize/deserialization equivalent |
| Shared `OrchestrationPlan` object | 1 | Planner output | Inline object and YAML round-trip fingerprints match |
| Supplied-plan serve path | 2 | `OrchestrationPlanApplier` | `serve --config` and inline serve rank plans match |
| Memory-aware single GPU AUTO | 2 | `primary_device` | Mixed GPU and free-memory constrained cases |
| CPU fallback | 2 | `primary_device = cpu` | CPU-only and no-fitting-GPU cases |
| Local TP AUTO | 3 | `local_tp_devices` | Same-vendor and cross-vendor grouped candidates |
| Local PP AUTO | 4 | `local_pp_devices` | Symmetric and asymmetric layer boundaries |
| Local PP+TP AUTO | 5 | `local_pp_stage_tp_info` | Vendor-domain stages and nested TP validation |
| Old strategy reuse | 6 | Adapter to V2 candidates | Supported translations and fail-closed diagnostics |
| Global TP/PP AUTO | 7 | Global plan fields | Per-rank deterministic plan fingerprints |
| Phase-aware decode | 8 | New V2 placement surface | Prefill/decode participant separation |
| CLI/runtime planner unification | 1, 9 | Shared planner and YAML artifact | Same candidate selected by both paths, command-local enumeration removed |
| Legacy consolidation | 10 | One planner brain | No duplicate AUTO fallback behavior |

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| AUTO changes explicit behavior | High | Gate AUTO on unspecified device mode only; add bypass tests for every explicit mode. |
| Planner selects a shape V2 cannot execute | High | Candidate kinds must declare executor support; unsupported kinds are diagnostic-only. |
| Memory estimates are too optimistic | High | Keep headroom, use free memory, and keep runtime `validateMemoryPlan()` as a hard gate. |
| Cross-vendor TP is technically possible but slow | Medium | Prefer same-backend TP and PP across vendor domains before cross-vendor TP. |
| Old `PlacementPlan` semantics do not map cleanly | Medium | Adapter fails closed and reports unsupported features. |
| Plan command and runtime drift again | Medium | Make both front ends produce/consume `OrchestrationPlan`; test YAML and inline fingerprints against each other. |
| Mock inventories are too synthetic | Medium | Build mocks through gather-equivalent serialization and include real MPI integration tests. |

---

## Open Decisions

1. Should local PP ever outrank local TP when both fit, or should PP only be a fallback for unsupported TP backends and asymmetric memory?
2. What minimum headroom should AUTO reserve by default for runtime selection: the current 128 MB planner default, a percentage of device memory, or both?
3. Should cross-vendor TP be enabled automatically, or require an opt-in until HOST backend performance is characterized?
4. Should phase-aware CPU decode be represented in `RankExecutionPlan`, a new `RuntimePlacementPlan`, or graph-builder-specific config?
5. Should `MPITopology::computePlacement()` become a wrapper over the shared planner, or remain as a legacy compatibility API with adapter tests?

---

## Definition of Done

The cleanup is complete when:

1. Running with a model path and no device flags selects a memory-feasible plan when one exists.
2. `llaminar plan` performs MPI gather/probe, produces an `OrchestrationPlan`, and writes that plan as YAML.
3. `llaminar serve --config plan.yaml` deserializes the same `OrchestrationPlan` schema and runs through the same `OrchestrationPlanApplier` and `OrchestrationRunner` path as inline planning.
4. `llaminar serve` without supplied YAML performs MPI gather/probe/planning inline, creates the same in-memory `OrchestrationPlan`, and then runs through the same applier path.
5. Runtime AUTO and `llaminar plan` use the same candidate enumeration and scoring code.
6. V2 execution is driven by `RankExecutionPlan`, not by a parallel placement contract.
7. The old placement strategies either contribute through a tested adapter or are documented as legacy.
8. Every supported AUTO shape has gathered-inventory mock tests and at least one real MPI determinism test where applicable.
9. The startup banner, dry run, and explain placement output tell the user why the plan was selected.