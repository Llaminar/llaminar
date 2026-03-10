# Config-to-Execution Flow Cleanup Proposal

**Date**: 2026-03-10  
**Status**: Proposed  
**Scope**: `OrchestrationRunner`, `ExecutionPlanBuilder`, `MultiDeviceOrchestrator`, `InferenceRunnerFactory`

---

## Problem Statement

The setup process that translates CLI/YAML configuration into an inferencing plan is ad-hoc and haphazardly wired. Single Device, Tensor Parallel, and Pipeline Parallel modes each have their own one-off config-parsing paths that duplicate field copies, re-parse the same strings, and reach into different intermediate structs inconsistently. This makes the code fragile, hard to extend, and unnecessarily verbose.

---

## Current Architecture

### The Config Type Chain (5 Layers Deep)

The journey from CLI flags to inference traverses a 5-struct degradation chain, where each layer extracts a subset from the previous:

| # | Type | Location | Role |
|---|------|----------|------|
| 1 | `OrchestrationConfig` | `src/v2/config/OrchestrationConfig.h` | User-facing CLI/YAML config. ~150 fields (raw strings, enums, everything). |
| 2 | `RankExecutionPlan` | `src/v2/execution/mpi_orchestration/RankExecutionPlan.h` | Per-rank contract: "what devices + layers this rank owns." Topology only. |
| 3 | `MultiDeviceOrchestrator::Config` | `src/v2/execution/local_execution/orchestrators/MultiDeviceOrchestrator.h` | Multi-device orchestration: devices, weights, PP stages, mode. |
| 4 | `InferenceRunnerConfig` | `src/v2/execution/factory/InferenceRunnerFactory.h` | Per-device runner: seq_len, precision, TP context pointer. |
| 5 | `GraphConfig` | `src/v2/models/GraphTypes.h` | Final per-orchestrator config: architecture dims, TP slices, device. ~50 fields. |

### The Three Build Paths

The critical branching point is `OrchestrationRunner::buildComputeGraph()` (line ~1012):

```
buildComputeGraph()
    ├─ hasLocalTP()   → buildMultiDeviceComputeGraph()   [TP path]
    ├─ usesLocalPP()  → buildLocalPPComputeGraph()       [PP path]
    └─ else           → buildSingleDeviceComputeGraph()  [Single-device path]
```

Each path manually constructs its own intermediate config struct and calls a different factory method.

#### Single Device Flow

```
OrchestrationConfig
  → ExecutionPlanBuilder::buildSimplePlan()     → RankExecutionPlan
  → buildSingleDeviceComputeGraph()             → InferenceRunnerConfig (manual field copy)
  → createInferenceRunner()                     → InferenceRunnerFactory
  → createDeviceGraphOrchestratorImpl()         → GraphConfig (via IGraphConfigBuilder + manual copy)
  → setFullDimensions()                         [no TP sharding]
  → DeviceGraphOrchestrator
```

#### Local TP Flow (Multi-Device, Single Rank)

```
OrchestrationConfig
  → ExecutionPlanBuilder::buildSimplePlan()     → RankExecutionPlan (local_tp_devices populated)
  → buildMultiDeviceConfig()                    → MDO::Config (manual field copy)
  → buildMultiDeviceComputeGraph()
  → MultiDeviceOrchestrator(model_ctx, tp_ctx, config)  [TP constructor]
  → initializeDeviceRunners()                   → FOR EACH device:
      → InferenceRunnerConfig (manual field copy from MDO::Config)
      → createTestableInferenceRunner()
      → createDeviceGraphOrchestratorImpl()     → GraphConfig
      → applyLocalTPAssignment()                [TP sharding]
      → DeviceGraphOrchestrator (one per device)
```

#### Local PP Flow (Pipeline Parallel)

```
OrchestrationConfig
  → ExecutionPlanBuilder::buildSimplePlan()     → RankExecutionPlan (local_pp_devices populated)
  → buildLocalPPComputeGraph()                  → MDO::Config (manual field copy, mode=PP hardcoded)
  → MultiDeviceOrchestrator(model_ctx, config)  [Config-only constructor]
  → initializePPDeviceRunners()                 → FOR EACH stage:
      → ModelContext::createForPPStage()        [partitioned model context]
      → InferenceRunnerConfig (manual field copy from MDO::Config)
      → FactoryPPStageConfig (layer ranges, has_embedding, has_lm_head)
      → createPPStageRunner() or nested MDO (for TP domains within PP)
      → DeviceGraphOrchestrator (one per PP stage)
```

---

## Identified Code Smells

### Smell 1: Triplicated Precision Parsing

`config_.activation_precision` is a raw `std::string` that gets parsed via `parseActivationPrecisionString()` at three independent call sites:

| Site | File:Line |
|------|-----------|
| `buildMultiDeviceConfig()` | `OrchestrationRunner.cpp:1056` |
| `buildLocalPPComputeGraph()` | `OrchestrationRunner.cpp:1162` |
| `buildSingleDeviceComputeGraph()` | `OrchestrationRunner.cpp:1269` |

Same issue with `parseKVCachePrecision()`. The string→enum conversion should happen **once** during config initialization.

### Smell 2: Triplicated Field-Copy Boilerplate

The pattern `max_seq_len = X; activation_precision = parse(...); kv_cache_precision = parse(...)` is manually repeated at **five** sites:

1. `buildSingleDeviceComputeGraph()` → populates `InferenceRunnerConfig`
2. `buildMultiDeviceConfig()` → populates `MDO::Config`
3. `buildLocalPPComputeGraph()` → populates `MDO::Config`
4. `MDO::initializeDeviceRunners()` → copies `MDO::Config` → `InferenceRunnerConfig`
5. `MDO::initializePPDeviceRunners()` → copies `MDO::Config` → `InferenceRunnerConfig`

No single "translate config once" function exists. Each path manually copies the same ~6 fields.

### Smell 3: Three Entirely Separate Build Methods

`buildSingleDeviceComputeGraph()`, `buildMultiDeviceComputeGraph()`, and `buildLocalPPComputeGraph()` are three separate ~80-line methods with significant structural overlap:

- All validate device availability via `DeviceManager::instance()`
- All log execution strategy
- All construct some form of runner config
- All call some factory method

But each does it its own way with no shared abstraction.

### Smell 4: MDO Has Three Constructors for Different Modes

| Constructor Signature | Line | Purpose |
|----------------------|------|---------|
| `MDO(model_ctx, config)` | `MultiDeviceOrchestrator.cpp:245` | Config-only: detects TP vs PP via `effectiveMode()` |
| `MDO(model_ctx, tp_ctx, config)` | `MultiDeviceOrchestrator.cpp:301` | Pre-made TP context: forces TP mode |
| `MDO(model_ctx, runners, tp_ctx, config)` | `MultiDeviceOrchestrator.cpp:333` | Test injection: pre-made runners |

The first constructor dispatches to `initializeDeviceRunners()` (TP) or `initializePPDeviceRunners()` (PP) — two completely separate init paths inside the same class.

### Smell 5: InferenceRunnerFactory Has a 4-Way TP Cascade

Inside `createDeviceGraphOrchestratorImpl()` (~line 498), there's a cascading `if/else if/else if/else` for TP assignment:

```cpp
if (local_tp_ctx && degree > 1 && sharded)        → applyLocalTPAssignment()
else if (tp_config && sharded)                     → proportional global TP
else if (mpi world > 1 && sharded)                 → legacy equal-split global TP
else                                               → setFullDimensions() (no TP)
```

Each TP variant was bolted on as a new `else if` branch rather than being unified under a single strategy.

### Smell 6: PP and TP Read From Different Intermediate Structs

- **TP path**: `buildMultiDeviceConfig()` reads from `plan_` (RankExecutionPlan), which was already translated from OrchestrationConfig.
- **PP path**: `buildLocalPPComputeGraph()` reads partly from `plan_` (devices, boundaries) and partly from `config_` (OrchestrationConfig) directly for runtime settings.

The two paths don't even read from the same intermediate struct consistently.

### Smell 7: `RankExecutionPlan` Doesn't Carry Runtime Config

`RankExecutionPlan` carries topology info (devices, layers, weights, backends) but **not** runtime config (max_seq_len, activation_precision, kv_cache_precision). This is why all three build methods must reach back to `config_` (OrchestrationConfig) to get runtime values. If `RankExecutionPlan` carried the pre-parsed runtime config, the triplicated parsing would vanish.

---

## Root Cause

`RankExecutionPlan` was designed as a **topology document** ("what devices and layers does this rank own") rather than a **complete execution contract** ("everything this rank needs to run, fully parsed"). This forces `OrchestrationRunner` to serve as a manual bridge, re-reading `OrchestrationConfig` raw fields for every build path.

Current flow:

```
OrchestrationConfig (raw CLI)
    → partially translated into RankExecutionPlan (topology only)
    → OrchestrationRunner selects build path
        → each path re-reads OrchestrationConfig for runtime fields
        → each path manually constructs MDO::Config or InferenceRunnerConfig
            → MDO::Config re-copies fields into InferenceRunnerConfig
                → InferenceRunnerFactory re-copies into GraphConfig
```

That's **4–5 manual field-copy hops** with 3 parallel paths doing the same copies differently.

---

## Proposed Changes

### Phase 1: Parse Once, Copy Never

**Goal**: Eliminate triplicated string→enum parsing and field-copy boilerplate.

1. **Add a `RuntimeConfig` sub-struct** to `RankExecutionPlan`:

```cpp
struct RuntimeConfig {
    int max_seq_len = 2048;
    int batch_size = 1;
    ActivationPrecision activation_precision = ActivationPrecision::FP32;
    KVCachePrecision kv_cache_precision = KVCachePrecision::FP32;
    float kv_cache_scale = 3.0f;
    FusedAttentionBackend fused_attention_backend = FusedAttentionBackend::AUTO;
    bool use_mapped_memory = false;
};
```

2. **Parse raw strings once** in `ExecutionPlanBuilder::buildSimplePlan()` (and `buildPlanWithDomains()`), storing pre-parsed enums into `plan.runtime`.

3. **MDO::Config and InferenceRunnerConfig embed `RuntimeConfig`** (or a const reference) instead of having their own copies of the same fields.

4. **Remove** `parseActivationPrecisionString()` and `parseKVCachePrecision()` calls from all three build methods in OrchestrationRunner.

**Files changed**: `RankExecutionPlan.h`, `ExecutionPlanBuilder.cpp`, `OrchestrationRunner.cpp`, `MultiDeviceOrchestrator.h`, `InferenceRunnerFactory.h`.

**Risk**: Low. Purely structural — no behavioral change.

### Phase 2: Unify Build Paths

**Goal**: Replace the three `build*ComputeGraph()` methods with a single method that dispatches based on `RankExecutionPlan` attributes.

1. **Extract shared logic** (device validation, strategy logging) into helper methods.

2. **Single `buildComputeGraph()` implementation** that constructs the appropriate runner based on plan attributes, eliminating the three-way `if/else if/else` entirely. Pseudocode:

```cpp
bool OrchestrationRunner::buildComputeGraph() {
    validateDevices(plan_);   // shared
    logStrategy(plan_);       // shared

    if (plan_.usesLocalTP() || plan_.usesLocalPP()) {
        auto mdo_config = MDO::Config::fromPlan(plan_);  // single translation point
        runner_ = std::make_unique<MultiDeviceOrchestrator>(model_ctx_, mdo_config);
    } else {
        auto runner_config = InferenceRunnerConfig::fromPlan(plan_);
        runner_ = createInferenceRunner(model_ctx_, mpi_ctx_, plan_.primaryDevice(), runner_config);
    }
    return runner_ != nullptr;
}
```

3. **`MDO::Config::fromPlan()`** and **`InferenceRunnerConfig::fromPlan()`** become the single, canonical translation points from plan to sub-config.

**Files changed**: `OrchestrationRunner.cpp`, `MultiDeviceOrchestrator.h/.cpp`, `InferenceRunnerFactory.h`.

**Risk**: Medium. Changes the construction flow but not the runtime behavior. Existing tests provide safety net.

### Phase 3: Consolidate MDO Constructors

**Goal**: Reduce MDO's three constructors to two (production + test injection).

1. **Merge the Config-only and TP-context constructors**. The Config-only constructor already creates a TP context for TP mode — the pre-made-TP-context constructor is redundant. Have the caller set the TP context on the config or pass it alongside.

2. **Keep the test-injection constructor** (it serves a legitimate purpose for unit testing with mocked runners).

**Files changed**: `MultiDeviceOrchestrator.h/.cpp`, `OrchestrationRunner.cpp`.

**Risk**: Medium. Must verify no external callers depend on the pre-made-TP-context constructor signature.

### Phase 4: Unify TP Assignment in InferenceRunnerFactory

**Goal**: Replace the 4-way `if/else if/else if/else` TP cascade with a strategy pattern.

1. **Define `ITPAssignment` interface** with a single method: `apply(GraphConfig&)`.

2. **Four implementations**: `LocalTPAssignment`, `ProportionalGlobalTPAssignment`, `EqualSplitGlobalTPAssignment`, `NoTPAssignment`.

3. **Factory selects the appropriate strategy** once, then calls `strategy->apply(graph_config)`.

**Files changed**: `InferenceRunnerFactory.cpp`, new `TPAssignment.h`.

**Risk**: Low. Localized refactor within the factory.

---

## Verification Plan

Each phase must pass:

1. **All 374+ unit tests**: `ctest --test-dir build_v2_integration -R "^V2_Unit_" --output-on-failure --parallel`
2. **All 7+ parity tests**: `ctest --test-dir build_v2_integration -R "^V2_Integration_Parity_" --output-on-failure`
3. **Integration tests**: `ctest --test-dir build_v2_integration -R "^V2_Integration_" --output-on-failure --parallel`
4. **Manual smoke test**: Single device, TP (2-way), PP (2-way) inference with `--dry-run` and live execution.

---

## Estimated Impact

| Metric | Before | After |
|--------|--------|-------|
| `parseActivationPrecisionString()` call sites | 3 | 1 |
| Manual field-copy sites | 5 | 1–2 |
| `build*ComputeGraph()` methods | 3 | 1 |
| MDO constructors (non-test) | 2 | 1 |
| TP assignment branches in factory | 4-way cascade | Strategy dispatch |
| Net lines removed (estimated) | — | ~200–300 |

---

## Non-Goals

- **No runtime behavioral changes.** This is purely structural cleanup.
- **No new parallelism modes.** The cleanup makes adding them easier, but that's future work.
- **No changes to `GraphConfig` population.** The `IGraphConfigBuilder` → `GraphConfig` path inside the factory is already clean.
- **No changes to PP/TP forward paths.** `forwardPP()`, `forwardTP()`, collective operations, coherence — all untouched.
