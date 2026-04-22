# Audit: BROKEN Pipeline Parallelism Code Path (Unified Graph Approach)

**Date**: February 3, 2026  
**Status**: To Be Removed  
**Replacement**: RankOrchestrator-based PP (per-stage orchestrators)

---

## Executive Summary

The current PP implementation uses a "single DeviceGraphOrchestrator with unified graph" approach that is **architecturally broken**. The fundamental problem: **a single graph cannot span multiple GPU execution contexts**. Each GPU has its own execution queue, command buffer, and synchronization primitives - you cannot interleave stages across devices in a single DAG.

This document audits exactly what code needs to be removed so the new RankOrchestrator-based PP can replace it cleanly.

---

## 1. Factory Function: `createUnifiedPipelineRunner()`

### Location
- **Header**: [src/v2/execution/factory/InferenceRunnerFactory.h](src/v2/execution/factory/InferenceRunnerFactory.h#L288-L292)
- **Implementation**: [src/v2/execution/factory/InferenceRunnerFactory.cpp](src/v2/execution/factory/InferenceRunnerFactory.cpp#L1141-L1260)

### What It Does
Creates a single `DeviceGraphOrchestrator` with:
1. A `PipelineConfig` injected via `setPipelineConfig()`
2. Layer→device weight mapping via a lambda accessor
3. Per-device KV caches created in `initializeInferenceState()`
4. Expects `buildUnifiedPipelineGraph()` to create a single monolithic graph

### Signature
```cpp
std::unique_ptr<IInferenceRunner> createUnifiedPipelineRunner(
    std::shared_ptr<ModelContext> model_ctx,
    std::shared_ptr<PipelineConfig> pipeline_config,
    const InferenceRunnerConfig &config = {});
```

### Helper Function
- `configureUnifiedPipelineWeightsImpl()` - Lines 1095-1140
- Sets up layer weight accessor that uses `pipeline_config_ptr->getDeviceForLayer(layer_idx)`

### **ACTION**: Remove entire function (~150 lines)
Also remove `configureUnifiedPipelineWeightsImpl()` (~45 lines).

---

## 2. Graph Building: `buildUnifiedPipelineGraph()`

### Location
- **Declaration**: [src/v2/models/qwen/Qwen2Graph.h](src/v2/models/qwen/Qwen2Graph.h#L696-L703)
- **Implementation**: [src/v2/models/qwen/Qwen2Graph.cpp](src/v2/models/qwen/Qwen2Graph.cpp#L750-L1100) (approximately)

### What It Does
Creates a single `ComputeGraph` spanning ALL PP stages:
```cpp
for (const auto &pp_stage : config_.pipeline_config->pp_stages)
{
    // 1. Set device for this stage
    // 2. Build embedding if has_embedding
    // 3. Build transformer layers for this stage
    // 4. Insert LocalPPTransferStage between stages
    // 5. Build LM head if has_lm_head
}
```

### Why It's Wrong
- A single `ComputeGraph` cannot span multiple device execution contexts
- Assumes `DeviceGraphExecutor::executeMultiDevice()` can magically switch contexts mid-execution
- The "multi-device" execution is a lie - it just picks different contexts per stage node, but real GPU kernels don't work that way

### Contrast with `buildPartialForwardGraph()`
The `buildPartialForwardGraph()` method is **CORRECT** - it builds a graph for a single PP stage:
```cpp
ComputeGraph buildPartialForwardGraph(
    const Qwen2ForwardInput &input,
    Qwen2ForwardOutput &output,
    int first_layer,      // Start of this stage's layers
    int last_layer,       // End of this stage's layers
    bool has_embedding,   // Include embedding lookup?
    bool has_lm_head);    // Include LM head?
```

This is what the new RankOrchestrator uses - one partial graph per stage.

### **ACTION**: Remove `buildUnifiedPipelineGraph()` (~350 lines in Qwen2Graph.cpp)
Keep `buildPartialForwardGraph()` - it's the correct approach.

---

## 3. PP-Specific Code in DeviceGraphOrchestrator

### 3.1 `InferenceState::pp_kv_caches` Map

**Location**: [src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h](src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h#L166)

```cpp
struct InferenceState
{
    // ...
    /// Per-device KV caches for Pipeline Parallelism
    std::unordered_map<DeviceId, std::unique_ptr<IKVCache>> pp_kv_caches;
    // ...
};
```

Also in `InferenceState::clear()` at line 238:
```cpp
for (auto &[device, cache] : pp_kv_caches)
{
    if (cache)
        cache->clear();
}
```

**ACTION**: Remove `pp_kv_caches` member and its iteration in `clear()`

### 3.2 Multi-Device KV Cache Creation in `initializeInferenceState()`

**Location**: [src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp](src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp#L1726-L1810)

```cpp
if (pipeline_config_ && pipeline_config_->hasPP())
{
    // Create a KV cache for each PP stage's device
    for (const auto &pp_stage : pipeline_config_->pp_stages)
    {
        // ... ~80 lines of per-device cache creation
        state_.pp_kv_caches[stage_device] = KernelFactory::createKVCache(kv_config);
    }
}
```

**ACTION**: Remove entire PP KV cache creation block (~85 lines)

### 3.3 `executeMultiDevice()` Calls in `executeForward()`

**Location**: [src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp](src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp#L650-L665)

```cpp
if (pipeline_config_ && pipeline_config_->hasPP())
{
    // Build device contexts map for all devices in the pipeline
    std::unordered_map<DeviceId, IDeviceContext*> contexts;
    for (const auto& device : pipeline_config_->getAllDevices())
    {
        contexts[device] = getDeviceContext(device);
    }
    success = executor_.executeMultiDevice(graph, contexts);
}
```

**ACTION**: Remove the multi-device execution branch (keep single-device path)

### 3.4 `hasUnifiedPP()` Method

**Location**: [src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h](src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h#L729)

```cpp
bool hasUnifiedPP() const { return pipeline_config_ && pipeline_config_->hasPP(); }
```

**ACTION**: Remove this method

### 3.5 PP Branch in `executeForward()` Graph Building

**Location**: [src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp](src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp#L574-L610)

```cpp
if (pipeline_config_ && pipeline_config_->hasPP())
{
    // Unified PP+TP path: single graph spanning all PP stages
    LOG_DEBUG("[DeviceGraphOrchestrator] Building UNIFIED PIPELINE graph...");

    // Initialize PP and TP contexts
    if (!pp_contexts_initialized_ && !initializePPContexts()) { ... }
    if (!tp_contexts_initialized_ && !initializeTPContexts()) { ... }

    // Wire contexts and build unified graph
    return session.withPipelineConfig(pipeline_config_).buildUnified();
}
```

**ACTION**: Remove entire unified PP branch

### 3.6 `initializePPContexts()` Method

**Location**: [src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp](src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp#L2561-L2700)

Creates `ILocalPPContext` instances for inter-stage transfers within the unified graph approach.

**ACTION**: Remove `initializePPContexts()` (~140 lines)
Also remove:
- `pp_contexts_` map (header line 2062)
- `pp_contexts_initialized_` flag (header line 2070)

### 3.7 `GraphBuildSession::buildUnified()` Method

**Location**: [src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp](src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp#L2969-L3015)

```cpp
GraphBuildResult GraphBuildSession::buildUnified()
{
    // ... validation
    graph_builder->setPipelineConfig(pipeline_config_);
    // Wire PP contexts
    // Wire TP contexts
    ComputeGraph graph = graph_builder->buildUnifiedPipelineGraph(prepared_input, output);
    // ...
}
```

**ACTION**: Remove `buildUnified()` method (~50 lines)

### 3.8 `pipeline_config_` Member Variable

**Location**: [src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h](src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h#L1113)

```cpp
std::shared_ptr<PipelineConfig> pipeline_config_;
```

**NOTE**: Keep this! The `PipelineConfig` data structure is still useful:
- It describes the PP topology
- Used by `createPPStageRunner()` for the correct approach
- Could be renamed to clarify it's configuration, not execution state

### 3.9 `pipelineConfig()` Accessor

**Location**: [src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h](src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h#L723)

```cpp
std::shared_ptr<PipelineConfig> pipelineConfig() const { return pipeline_config_; }
```

**ACTION**: Keep - useful for introspection

---

## 4. Test Setup: `setupLocalPPPipeline()`

### Location
- **Base Implementation**: [tests/v2/integration/parity/ParityTestBase.h](tests/v2/integration/parity/ParityTestBase.h#L1741-L1855)
- **Qwen2 Override**: [tests/v2/integration/parity/qwen2/Qwen2ParityTestBase.h](tests/v2/integration/parity/qwen2/Qwen2ParityTestBase.h#L426-L429) (delegates to base)

### What It Does
```cpp
bool setupLocalPPPipeline()
{
    // 1. Load model with REPLICATED strategy
    // 2. Calculate layer boundaries for N stages
    // 3. Create PipelineConfig with PP stages
    // 4. Call createUnifiedPipelineRunner(model_ctx_, pipeline_config, runner_config)
    // 5. Enable snapshot capture
}
```

### Contrast with `setupLocalTPPipeline()`
The TP pipeline uses `RankOrchestrator` - the CORRECT approach:
```cpp
bool setupLocalTPPipeline()
{
    // ... create multi_orch_ with RankOrchestrator
    runner_.reset(multi_orch_.release());
}
```

### **ACTION**: Rewrite `setupLocalPPPipeline()` to use RankOrchestrator
- Remove call to `createUnifiedPipelineRunner()`
- Create per-stage orchestrators via `createPPStageRunner()`
- Coordinate execution via `RankOrchestrator` pattern

---

## 5. Test Files to Update/Remove

### Tests Using `buildUnifiedPipelineGraph()` Directly
| File | Status |
|------|--------|
| `tests/v2/unit/models/qwen/Test__Qwen2Graph_PP.cpp` | **REMOVE** (1257 lines) |

### Tests Using `executeMultiDevice()` with Unified Graph
| File | Status |
|------|--------|
| `tests/v2/integration/pp/Test__UnifiedPP_Execution.cpp` | **REMOVE** (966 lines) |

### Tests Using `setupLocalPPPipeline()` (via `createUnifiedPipelineRunner()`)
| File | Status |
|------|--------|
| `tests/v2/integration/parity/qwen2/Test__Qwen2_ParityMatrix.cpp` | **MIGRATE** - calls `setupPipeline()` which dispatches to `setupLocalPPPipeline()` |
| `tests/v2/integration/parity/qwen2/Qwen2ParityTestBase.h` | **MIGRATE** - `setupPipeline()` dispatch and `setupLocalPPPipeline()` |
| `tests/v2/integration/parity/ParityTestBase.h` | **MIGRATE** - base `setupLocalPPPipeline()` implementation |

### Tests that SHOULD NOT be affected (correct approach)
| File | Status |
|------|--------|
| `tests/v2/integration/execution/runner/Test__PPStageRunner.cpp` | **KEEP** - tests `createPPStageRunner()` (correct approach) |
| `tests/v2/unit/execution/compute_stages/stages/Test__LocalPPTransferStage.cpp` | **KEEP** - tests transfer stage (reusable) |

---

## 6. Summary: What to Remove vs Keep

### REMOVE (Broken Unified PP Path)

| Item | Location | Lines |
|------|----------|-------|
| `createUnifiedPipelineRunner()` | InferenceRunnerFactory.cpp | ~150 |
| `configureUnifiedPipelineWeightsImpl()` | InferenceRunnerFactory.cpp | ~45 |
| `buildUnifiedPipelineGraph()` | Qwen2Graph.cpp | ~350 |
| `pp_kv_caches` map | DeviceGraphOrchestrator.h | ~15 |
| PP KV cache creation in `initializeInferenceState()` | DeviceGraphOrchestrator.cpp | ~85 |
| Multi-device execution branch in `executeForward()` | DeviceGraphOrchestrator.cpp | ~15 |
| `hasUnifiedPP()` | DeviceGraphOrchestrator.h | ~1 |
| PP branch in `executeForward()` graph building | DeviceGraphOrchestrator.cpp | ~40 |
| `initializePPContexts()` | DeviceGraphOrchestrator.cpp | ~140 |
| `pp_contexts_` map | DeviceGraphOrchestrator.h | ~3 |
| `pp_contexts_initialized_` | DeviceGraphOrchestrator.h | ~1 |
| `GraphBuildSession::buildUnified()` | DeviceGraphOrchestrator.cpp | ~50 |
| Test__Qwen2Graph_PP.cpp | tests/v2/unit/models/qwen/ | ~1257 |
| Test__UnifiedPP_Execution.cpp | tests/v2/integration/pp/ | ~966 |
| **TOTAL** | | **~3100 lines** |

### KEEP (Reusable Infrastructure)

| Item | Reason |
|------|--------|
| `PipelineConfig` | Describes PP topology - used by correct approach |
| `TPDomainConfig` | Describes TP domains - reusable |
| `PPStageConfig` | Describes individual PP stages - reusable |
| `pipeline_config_` member | Configuration storage - useful |
| `pipelineConfig()` accessor | Introspection - useful |
| `createPPStageRunner()` | Correct factory - creates single-stage orchestrators |
| `buildPartialForwardGraph()` | Correct graph building - per-stage graphs |
| `setPipelineConfig()` | Sets config - may need refactoring but useful |
| `LocalPPTransferStage` | PP activation transfer - reusable with modifications |
| `ILocalPPContext` | PP context interface - reusable |

### MIGRATE (Need Updates)

| Item | Change Needed |
|------|---------------|
| `setupLocalPPPipeline()` | Use RankOrchestrator instead of unified runner |
| Tests calling `setupLocalPPPipeline()` | Update to use new API |

---

## 7. Dependency Graph for Removal

```
createUnifiedPipelineRunner()
    └── configureUnifiedPipelineWeightsImpl()
    └── DeviceGraphOrchestrator.setPipelineConfig()
        └── executeForward() PP branch
            └── initializePPContexts()
            └── GraphBuildSession::buildUnified()
                └── Qwen2Graph::buildUnifiedPipelineGraph()
                    └── pp_kv_caches usage
            └── executor_.executeMultiDevice()
```

**Removal Order**:
1. Tests first (Test__Qwen2Graph_PP.cpp, Test__UnifiedPP_Execution.cpp)
2. Factory function (createUnifiedPipelineRunner)
3. Graph building (buildUnifiedPipelineGraph, buildUnified)
4. Orchestrator internals (initializePPContexts, pp_kv_caches, etc.)
5. Cleanup dead code paths

---

## 8. The Correct Approach (for reference)

The new RankOrchestrator-based PP:

```
RankOrchestrator
├── PPStageRunner[0] (DeviceGraphOrchestrator with partial graph)
│   └── buildPartialForwardGraph(layers 0..N/2, has_embedding=true)
│   └── KV cache for layers 0..N/2 only
│
├── PPStageRunner[1] (DeviceGraphOrchestrator with partial graph)
│   └── buildPartialForwardGraph(layers N/2..N, has_lm_head=true)
│   └── KV cache for layers N/2..N only
│
└── Inter-stage activation transfer via GlobalBackendRouter
```

Each orchestrator owns its own:
- Device context(s)
- KV cache (for its layers only)
- Partial compute graph
- Execution loop

The RankOrchestrator coordinates:
- Prefill/decode phase transitions
- Activation transfers between stages
- KV cache state synchronization (positions, etc.)

This matches how real multi-GPU inference works - each GPU executes independently, with explicit synchronization points.
