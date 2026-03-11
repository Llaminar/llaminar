# Config-to-Execution Flow Cleanup Proposal

**Date**: 2026-03-10  
**Status**: Complete (all 4 phases done)  
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

### Smell 1: Triplicated Precision Parsing ✅ FIXED (Phase 1)

~~`config_.activation_precision` is a raw `std::string` that gets parsed via `parseActivationPrecisionString()` at three independent call sites.~~ **Fixed**: Parsing now happens once in `ExecutionPlanBuilder` via `RuntimeConfig::fromOrchestrationConfig()`. All three build methods read pre-parsed enums from `plan_.runtime`.

### Smell 2: Triplicated Field-Copy Boilerplate ✅ FIXED (Phases 1+2)

The pattern `max_seq_len = X; activation_precision = ...; kv_cache_precision = ...` was manually repeated at **five** sites. Phase 1 eliminated the string re-parsing. Phase 2 eliminated the remaining field copies via `MDO::Config::fromPlan()` and `InferenceRunnerConfig::fromPlan()`.

Sites remaining after Phase 1:
1. ~~`buildMultiDeviceConfig()` → copies `plan_.runtime` fields to `MDO::Config`~~ ✅ Replaced by `MDO::Config::fromPlan()`
2. ~~`buildLocalPPComputeGraph()` → copies `plan_.runtime` fields to `MDO::Config`~~ ✅ Replaced by `MDO::Config::fromPlan()`
3. ~~`buildSingleDeviceComputeGraph()` → copies `plan_.runtime` fields to `InferenceRunnerConfig`~~ ✅ Replaced by `InferenceRunnerConfig::fromPlan()`
4. `MDO::initializeDeviceRunners()` → copies `MDO::Config` → `InferenceRunnerConfig`
5. `MDO::initializePPDeviceRunners()` → copies `MDO::Config` → `InferenceRunnerConfig`

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

### Smell 7: `RankExecutionPlan` Doesn't Carry Runtime Config ✅ FIXED (Phase 1)

~~`RankExecutionPlan` carries topology info but **not** runtime config.~~ **Fixed**: `RankExecutionPlan` now has a `RuntimeConfig runtime` member populated by `ExecutionPlanBuilder`. All three build methods read from `plan_.runtime` instead of reaching back to `config_`.

---

## Root Cause ⬜ PARTIALLY ADDRESSED

`RankExecutionPlan` was designed as a **topology document** rather than a **complete execution contract**. Phase 1 fixed this by adding `RuntimeConfig runtime` — the plan now carries pre-parsed runtime config. Phase 2 centralized the downstream translation into `fromPlan()` factories on `MDO::Config` and `InferenceRunnerConfig`, eliminating all manual field copies from OrchestrationRunner.

---

## Proposed Changes

### Phase 1: Parse Once, Copy Never ✅ COMPLETED

**Goal**: Eliminate triplicated string→enum parsing and field-copy boilerplate.

**Decision**: Instead of creating a new `PlanRuntimeConfig` sub-struct as originally proposed, we **slimmed the existing `RuntimeConfig` struct** (which was dead code — only one test instantiated it, but its header was `#include`d by ~15 files for the enum types) and promoted it to be the canonical first-class config carrier. This avoids creating a new type and cleans up the dead code.

**What was done**:

1. **Slimmed `RuntimeConfig`** from ~20 fields to 6: removed `n_threads`, `use_mmap`, `seed`, `use_fused_attention`, 6 `executor_*` flags. Kept: `max_seq_len`, `batch_size`, `activation_precision`, `fused_attention_backend`, `kv_cache_scale`, `kv_cache_precision`.

2. **Added `parseActivationPrecision()`** to `RuntimeConfig.h` (promoted from anonymous-namespace function in OrchestrationRunner.cpp).

3. **Added `RuntimeConfig::fromOrchestrationConfig()`** static factory that parses raw strings once.

4. **Added `RuntimeConfig runtime` member** to `RankExecutionPlan`.

5. **Populated `plan.runtime`** in `ExecutionPlanBuilder` — both `buildSimplePlan()` and `buildPlanWithDomains()` call `RuntimeConfig::fromOrchestrationConfig()`.

6. **Updated all 3 `OrchestrationRunner` build paths** to read from `plan_.runtime` instead of re-parsing `config_` strings.

7. **Removed** the anonymous `parseActivationPrecisionString()` from OrchestrationRunner.cpp.

**Files changed**: `RuntimeConfig.h`, `RankExecutionPlan.h`, `ExecutionPlanBuilder.cpp`, `OrchestrationRunner.cpp`.

**Verification**: 1609 build targets, 0 errors. 374/374 unit tests pass.

### Phase 2: Centralize Config Translation ✅ COMPLETED

**Goal**: Replace manual field-copy boilerplate in OrchestrationRunner with canonical `fromPlan()` factory methods on the config structs.

**Refined scope** (updated after code analysis): The original proposal suggested replacing all three `build*ComputeGraph()` methods with a single method. After examining the code, each path has genuinely distinct logic:
- **TP path**: Passes pre-created `local_tp_ctx_`, uses `createMultiDeviceOrchestrator()` factory
- **PP path**: Builds `PPStageConfig` entries with cross-vendor detection, creates MDO directly, initializes `GlobalBackendRouter`
- **Single-device path**: Resolves device from multiple sources with NUMA support, passes `mpi_ctx_`, uses `createInferenceRunner()` factory

Forcing these into a single method would create an unreadable if/else. Instead, Phase 2 focuses on the higher-value win: **centralizing config translation** via `fromPlan()` factories.

**Changes**:

1. **Add `MDO::Config::fromPlan(const RankExecutionPlan&)`** static factory that handles both TP and PP config construction. For TP: copies devices, weights, backend. For PP: sets `mode=PP`, builds `PPStageConfig` entries from plan boundaries with cross-vendor BAR detection.

2. **Add `InferenceRunnerConfig::fromPlan(const RankExecutionPlan&)`** static factory — eliminates the field-copy block in `buildSingleDeviceComputeGraph()`.

3. **Remove `buildMultiDeviceConfig()`** helper from OrchestrationRunner (its logic moves into `MDO::Config::fromPlan()`).

4. **Simplify the three build methods** to use the new factories. They keep their path-specific logic (device validation, strategy logging, runner creation) but lose the config-construction boilerplate.

**Files changed**: `MultiDeviceOrchestrator.h`, `InferenceRunnerFactory.h`, `OrchestrationRunner.h`, `OrchestrationRunner.cpp`.

**Risk**: Low. Config translation moves but its logic is unchanged.

### Phase 3: Consolidate MDO Constructors ✅ COMPLETED

**Goal**: Reduce MDO's three constructors to two (production + test injection).

1. **Merged the Config-only and TP-context constructors** into a single constructor with an optional `tp_ctx` parameter: `MultiDeviceOrchestrator(model_ctx, config, tp_ctx = nullptr)`. If `tp_ctx` is provided, uses it directly (TP mode). Otherwise, auto-detects mode from config and creates TP context if needed.

2. **Kept the test-injection constructor** (private, via `createForTest()`) — unchanged.

**Files changed**: `MultiDeviceOrchestrator.h/.cpp`, `InferenceRunnerFactory.cpp`, `Qwen2ParityTestBase.h`.

**Result**: 3 constructors → 2 (unified production + private test injection). All 374 unit tests pass.

### Phase 4: Unify TP Assignment in InferenceRunnerFactory ✅ COMPLETED

**Goal**: Replace the 4-way `if/else if/else if/else` TP cascade with named functions and a clean selector.

**Refined scope** (updated during implementation): The original proposal called for a virtual `ITPAssignment` interface with 4 class implementations and a new header file. Since all strategies are internal to one file and unlikely to grow, a lighter approach was used: named static functions with a selector, avoiding unnecessary abstraction.

1. **Extracted `applyProportionalGlobalTPAssignment()`** — proportional GLOBAL TP via `TensorParallelConfig`.

2. **Extracted `applyEqualSplitGlobalTPAssignment()`** — equal 1/world_size GLOBAL TP via MPI context.

3. **Created `applyTPAssignment()` selector** — picks the right strategy based on precedence (LOCAL > Proportional > Equal-Split > No TP) and delegates. Replaces the inline 4-way cascade.

4. **Existing functions preserved**: `applyLocalTPAssignment()` (already extracted) and `setFullDimensions()` (no-TP path).

**Files changed**: `InferenceRunnerFactory.cpp` only (no new header needed).

**Result**: ~100-line inline cascade → 5 focused static functions + 1 selector. Each TP mode is independently readable and testable. All 374 unit tests pass.

---

## Verification Plan

Each phase must pass:

1. **All 374+ unit tests**: `ctest --test-dir build_v2_integration -R "^V2_Unit_" --output-on-failure --parallel`
2. **All 7+ parity tests**: `ctest --test-dir build_v2_integration -R "^V2_Integration_Parity_" --output-on-failure`
3. **Integration tests**: `ctest --test-dir build_v2_integration -R "^V2_Integration_" --output-on-failure --parallel`
4. **Manual smoke test**: Single device, TP (2-way), PP (2-way) inference with `--dry-run` and live execution.

---

## Estimated Impact

| Metric | Before | Phase 1 | All Phases |
|--------|--------|---------|------------|
| `parseActivationPrecisionString()` call sites | 3 | **1** ✅ | 1 ✅ |
| Manual field-copy sites (OrchestrationRunner) | 5 | **3** ✅ | 0 ✅ |
| `build*ComputeGraph()` methods | 3 | 3 | 3 (simplified) ✅ |
| MDO constructors (non-test) | 2 | 2 | 1 ✅ |
| TP assignment branches in factory | 4-way cascade | 4-way cascade | Selector + named functions ✅ |
| Net lines removed (cumulative) | — | ~40 | ~200–300 |

---

## Non-Goals

- **No runtime behavioral changes.** This is purely structural cleanup.
- **No new parallelism modes.** The cleanup makes adding them easier, but that's future work.
- **No changes to `GraphConfig` population.** The `IGraphConfigBuilder` → `GraphConfig` path inside the factory is already clean.
- **No changes to PP/TP forward paths.** `forwardPP()`, `forwardTP()`, collective operations, coherence — all untouched.
