# Declarative Graph Builder Refactoring Plan

**Date**: December 22, 2025  
**Status**: In Progress  
**Goal**: Make model graph builders (Qwen2Graph, future Qwen3Graph, DeepseekGraph) purely declarative

---

## Executive Summary

The current `Qwen2Graph` class mixes declarative graph building with imperative execution logic, caching, buffer management, and device context handling. This violates the architectural principle that graph builders should **only build graphs**.

This refactoring separates concerns so that:
- **Graph builders** (Qwen2Graph, etc.) are purely declarative - they construct `ComputeGraph` objects
- **GraphOrchestrator** owns execution infrastructure (executor, device contexts, caching)
- **Shared utilities** are extracted to reusable headers

---

## Current Problems

### 1. Execution Methods in Graph Builder (~300 lines)

| Method | Lines | Problem |
|--------|-------|---------|
| `executeForward()` | ~50 | Builds AND executes graph |
| `executeAttention()` | ~100 | Builds AND executes with caching |
| `executeFFN()` | ~65 | Builds AND executes with caching |
| `executeLayer()` | ~20 | Orchestrates attention + FFN |
| `execute()` | ~3 | Direct executor delegation |

### 2. Runtime State Owned by Graph Builder

| State | Purpose | Problem |
|-------|---------|---------|
| `executor_` | DeviceGraphExecutor instance | Execution, not building |
| `device_contexts_` | IDeviceContext map | Execution infrastructure |
| `layer_graph_cache_` | Cached ComputeGraphs | Runtime optimization |
| `graph_caching_enabled_` | Cache flag | Runtime state |
| `last_pos_offset_` | Position tracking | Runtime state |
| `position_ids_buffer_` | Runtime buffer | Should be external |

### 3. Buffer Management (~200 lines)

| Method | Purpose | Problem |
|--------|---------|---------|
| `initializeBuffers()` | Allocate activation buffers | Memory lifecycle |
| `releaseBuffers()` | Free buffers | Memory lifecycle |
| `bindGraphManagedBuffers()` | Wire buffers to struct | Memory management |
| `buffer_manager_` | DeviceGraphBufferManager instance | Memory infrastructure |
| `buffer_spec_builder_` | Qwen2BufferSpecBuilder | Could stay or move |

### 4. Helper Functions That Should Be Shared

```cpp
static bool isRowParallelSharded(const TensorBase* weight)  // MPI utility
static void* getMPIComm(const MPIContext* mpi_ctx)          // MPI utility  
void addFinalNormToGraph(...)                               // Common pattern
std::vector<int> buildPositionIds(...)                      // Common utility
```

---

## Refactoring Phases

### Phase 1: Create GraphBuildUtils.h ✅

Extract shared helper functions to a new utility header.

**New File**: `src/v2/execution/GraphBuildUtils.h`

**Functions to Move**:
- `isRowParallelSharded()` - Check if tensor is row-parallel sharded
- `getMPICommPtr()` - Get MPI communicator as void*
- `buildPositionIds()` - Generate position ID sequence (kept in Qwen2Graph as static)

**Estimated Impact**: ~50 lines moved, 0 behavioral change

---

### Phase 2: Move Execution to GraphOrchestrator ✅

**Completed**: Execution is now exclusively in `GraphOrchestrator`.

**What was done**:
1. Updated `Test__LayerExecutor_Q8_1_vs_FP32_Parity.cpp` to use `GraphOrchestrator`
2. Removed from `Qwen2Graph.h`:
   - `executeForward()`, `execute()`, `executeAttention()`, `executeFFN()`, `executeLayer()`
   - `stats()`, `resetStats()`, `clearCache()`
   - `isGraphCachingEnabled()`, `getCacheSize()`, `hasValidCachedGraph()`
   - `canUseCachedGraph()`, `getOrBuildAttentionGraph()`, `getOrBuildFFNGraph()`
   - `updateCachedGraphParams()`, `getDeviceContext()`
   - `executor_` member
   - `device_contexts_` map
   - `position_ids_buffer_`
   - Graph caching infrastructure (`graph_caching_enabled_`, `CachedLayerGraphs`, `layer_graph_cache_`, `last_pos_offset_`)
3. Simplified constructors (no executor setup)
4. Updated `setSnapshotCallback()` to only store callback
5. Removed `Test__Qwen2GraphCaching.cpp` (caching now tested via `GraphOrchestrator`)

**Impact**: ~600 lines removed from Qwen2Graph, 163 unit tests pass

---

### Phase 3: Move Buffer Management to GraphOrchestrator ✅

**Completed**: Buffer management is now exclusively in `GraphOrchestrator`.

Buffer lifecycle is an execution concern, not a graph building concern.

**What was done**:
1. Added to `GraphOrchestrator`:
   - `setTensorFactory()` / `tensorFactory()` 
   - `initializeBuffers(int seq_len)` 
   - `releaseBuffers()`
   - `hasGraphManagedBuffers()`
   - `getInternalBuffers()` / `getModelBuffers()`
   - `bufferStats()`
   - `bindGraphManagedBuffers(int seq_len)` (private)
   - Member variables: `tensor_factory_`, `buffer_manager_`, `owned_buffers_`, `buffer_spec_builder_`, `managed_buffers_`
2. Added `createBufferSpecBuilder()` factory method to `Qwen2Graph` for architecture-specific buffer specification
3. Removed from `Qwen2Graph`:
   - `initializeBuffers()`, `releaseBuffers()`, `hasGraphManagedBuffers()`, `getInternalBuffers()`, `getModelBuffers()`, `bufferStats()`
   - `buffer_manager_`, `owned_buffers_`, `bindGraphManagedBuffers()`
4. Renamed test file `Test__Qwen2GraphBufferManagement.cpp` → `Test__GraphOrchestratorBufferManagement.cpp`

**Kept in Qwen2Graph**:
- `createBufferSpecBuilder()` - factory method for architecture-specific buffer specs
- `setTensorFactory()` - needed for buffer spec builder

**Impact**: ~200 lines moved, 163 unit tests pass (test renamed)

---

### Phase 4: Schema-Based Graph Building (Major Refactor)

**Status**: In Progress

#### The Problem

After Phases 1-3, Qwen2Graph still contains ~1158 lines of **mixed declarative + imperative** code:

| Element | Declarative? | Problem |
|---------|--------------|---------|
| Stage parameters (`RMSNormStage::Params`) | ✅ Yes | Could be YAML |
| Graph structure (`addNode`, `addDependency`) | ✅ Yes | Could be YAML |
| `if (env.execution.exec_gemm)` conditionals | ❌ No | Runtime toggle |
| `if (has_multi_rank && wo_is_sharded)` | ❌ No | Runtime MPI logic |
| `detect_attention_mode()` | ❌ No | Runtime mode selection |
| `if (cached_tokens > 0)` | ❌ No | KV cache state logic |

**Scaling Problem**: When adding DeepSeek R1 MoE, Qwen3 MoE, Mistral, etc., we'd have to copy-paste all this imperative logic into each new graph builder.

#### The Solution: Three-Layer Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Layer 1: GraphSchema (YAML / C++ struct)                       │
│  - Purely declarative structure                                 │
│  - No conditionals, just placeholders and annotations           │
│  - One schema per architecture: qwen2, deepseek_r1, llama3      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Layer 2: GraphResolver (C++ - SHARED across all models)        │
│  - Evaluates ALL conditionals ONCE before graph building        │
│  - Input: Schema + RuntimeConfig → ResolvedGraphSpec            │
│  - MPI/TP logic lives HERE (shared across ALL models)           │
│  - debugEnv logic lives HERE (shared across ALL models)         │
│  - Attention mode detection lives HERE                          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Layer 3: GraphBuilder (C++ - trivial, no conditionals)         │
│  - Pure function: ResolvedGraphSpec → ComputeGraph              │
│  - Just loops over resolved stages and calls addNode()          │
└─────────────────────────────────────────────────────────────────┘
```

#### Phase 4a: Define GraphSchema and ResolvedGraphSpec ✅

**Completed**: December 22, 2025

**New Files Created**:
- `src/v2/execution/GraphSchema.h` (~330 lines) - Declarative schema structs
- `src/v2/execution/GraphResolver.h` (~340 lines) - Resolver interface and configs
- `src/v2/pipelines/qwen/Qwen2Schema.h` (~280 lines) - Qwen2 architecture schema factory

**Key Structures**:

```cpp
// GraphSchema.h - Declarative stage specification
struct StageSpec {
    std::string name;           // "attn_norm", "qkv_proj", etc.
    StageType type;             // RMSNorm, GEMM, Attention, etc.
    std::vector<TensorRef> inputs;     // Buffer/weight references
    std::vector<TensorRef> outputs;    // Buffer references
    std::vector<std::string> dependencies;  // Stage dependencies
    
    // Annotations for resolver (not runtime conditionals!)
    TPMode tp_mode = TPMode::None;      // column_parallel, row_parallel
    bool requires_kv_cache = false;     // Stage only emitted if KV cache present
    bool is_optional = false;           // Can be disabled by exec policy
    std::string exec_policy_key;        // debugEnv key for optional stages
};

struct LayerTemplate {
    std::vector<StageSpec> attention_stages;
    std::vector<StageSpec> ffn_stages;
};

struct GraphSchema {
    std::string name;           // "qwen2", "deepseek_r1"
    std::string version;
    StageSpec embedding;
    LayerTemplate layer_template;
    std::vector<StageSpec> lm_head_stages;
};

// ResolvedGraphSpec - After resolver evaluates all conditionals
struct ResolvedStage {
    std::string name;
    StageType type;
    std::vector<TensorBase*> inputs;   // Actual tensor pointers
    std::vector<TensorBase*> outputs;
    std::vector<std::string> dependencies;
    int device_idx;
    std::unordered_map<std::string, std::any> params;  // Stage-specific params
};

struct ResolvedGraphSpec {
    std::string name;
    std::vector<ResolvedStage> stages;  // Flat list, ready to build
    ResolutionStats stats;              // For diagnostics
};
```

**Estimated Impact**: ~950 lines new code, 0 behavioral change (schema not used yet)

---

#### Phase 4b: Implement GraphResolver ✅

**Completed**: December 22, 2025

**New File**: `src/v2/execution/GraphResolver.cpp` (~640 lines)
**Test File**: `tests/v2/unit/Test__GraphResolver.cpp` (~420 lines, 18 tests)

**Resolver Implementation** (all imperative code moves here):

```cpp
class GraphResolver {
public:
    ResolvedGraphSpec resolve(
        const GraphSchema& schema,
        const GraphResolverConfig& runtime,
        const TensorContext& tensors);

private:
    // All imperative logic consolidated here:
    bool shouldEmitStage(const StageSpec& spec, const GraphResolverConfig& cfg);
    std::optional<ResolvedStage> resolveTPCollective(...);  // Insert allreduce/allgather
    AttentionMode detectAttentionMode(const GraphResolverConfig& cfg);
    std::vector<ResolvedStage> resolveLayer(const LayerTemplate&, int layer_idx, ...);
    std::optional<ResolvedStage> resolveStage(const StageSpec&, int layer_idx, ...);
    void populateStageParams(ResolvedStage&, const StageSpec&, ...);
};

// GraphBuilder - Trivial conversion to ComputeGraph
class GraphBuilder {
public:
    static ComputeGraph build(const ResolvedGraphSpec& spec);
};
```

**Key Features**:
- `ExecutionPolicyFlags` - Maps debugEnv toggles to bool flags
- `TensorContext` - Resolves string tensor references to TensorBase pointers
- Automatic TP collective insertion (allreduce for RowParallel, allgather for ColumnParallel)
- Attention mode detection (prefill vs decode)
- Layer template expansion with proper naming (layer0_*, layer1_*, etc.)

**Test Coverage**:
- ExecutionPolicyFlags (defaults, shouldExecute, fromDebugEnv)
- Schema resolution (empty, single layer, stage naming)
- Stage filtering (disabled stages skipped)
- TP collectives (single rank no collectives, multi-rank allreduce/allgather)
- Attention mode detection
- TensorContext (buffer, model weight, layer weight resolution)
- Qwen2Schema factory validation
- GraphBuilder integration

**Impact**: All 164 unit tests pass

---

#### Phase 4c: YAML Parser (Optional, Future) ⏳

Once we have 2-3 models working with C++ schemas, add YAML parsing:

```yaml
# schemas/qwen2.yaml
name: qwen2
layer_template:
  attention:
    - name: attn_norm
      type: RMSNorm
      inputs: [hidden, weights.attn_norm]
      outputs: [normalized]
    - name: qkv_proj
      type: FusedQKVGEMM
      tp_mode: column_parallel
      # ...
```

---

#### Phase 4d: Migrate Qwen2Graph to Schema

Replace imperative graph building with schema + resolver:

```cpp
// Before (imperative)
ComputeGraph Qwen2Graph::buildAttentionGraph(...) {
    if (env.execution.exec_rmsnorm) {
        graph.addNode("attn_norm", ...);
    }
    if (env.execution.exec_gemm && layer.wq) {
        // 50 lines of QKV setup
    }
    // ... 300 more lines
}

// After (declarative)
ComputeGraph Qwen2Graph::buildAttentionGraph(...) {
    auto resolved = resolver_.resolve(schema_.layer_template.attention, runtime_cfg_, ...);
    return builder_.build(resolved);
}
```

**Estimated Impact**: Qwen2Graph shrinks from ~1158 to ~200 lines

---

#### Phase 4e: Add Second Model (Validation)

Add Llama3 or Mistral using the same resolver to prove the design:

```cpp
// Same resolver, different schema
GraphSchema llama3_schema = Llama3Schema::create();
auto resolved = resolver_.resolve(llama3_schema, runtime_cfg, ...);
```

---

#### Benefits of Schema-Based Approach

| Benefit | Description |
|---------|-------------|
| **New model = new schema only** | ~100 lines YAML vs ~1000 lines C++ |
| **Shared TP/MPI logic** | Written once in GraphResolver |
| **Testable in isolation** | Unit test resolver with mock schemas |
| **Self-documenting** | Schema IS the architecture spec |
| **Debuggable** | Dump resolved graph before execution |
| **Future: hot-reload schemas** | Change YAML without recompile |

---

#### Implementation Order

| Sub-phase | Description | Effort | Risk |
|-----------|-------------|--------|------|
| **4a** | Define `GraphSchema`, `ResolvedGraphSpec` structs | 1-2 days | Low |
| **4b** | Implement `GraphResolver` | 2-3 days | Medium |
| **4c** | YAML parser (optional) | 1 day | Low |
| **4d** | Migrate Qwen2Graph to schema | 1-2 days | Medium |
| **4e** | Add Llama3/Mistral schema | 1 day | Low |

---

## File Changes Summary

### New Files
| File | Purpose |
|------|---------|
| `src/v2/execution/GraphBuildUtils.h` | Shared graph building utilities |

### Modified Files
| File | Changes |
|------|---------|
| `src/v2/execution/GraphOrchestrator.h` | Add execution methods, caching, buffer mgmt |
| `src/v2/execution/GraphOrchestrator.cpp` | Implement moved methods |
| `src/v2/pipelines/qwen/Qwen2Graph.h` | Remove execution members/methods |
| `src/v2/pipelines/qwen/Qwen2Graph.cpp` | Remove ~900 lines of execution code |

### Test Updates
- Tests calling `Qwen2Graph::execute*()` → call `GraphOrchestrator` instead
- May need to update test fixtures

---

## Implementation Order

1. **Phase 1**: Create `GraphBuildUtils.h`, move helpers (low risk)
2. **Phase 2**: Move execution methods to GraphOrchestrator (medium risk)
3. **Phase 3**: Move buffer management (medium risk)
4. **Phase 4**: Clean up Qwen2Graph header/impl (low risk)
5. **Verification**: Run full test suite after each phase

---

## Benefits

1. **Future Model Support**: Adding Qwen3Graph, DeepseekGraph becomes trivial
2. **Single Responsibility**: Graph builders build, orchestrators orchestrate
3. **Testability**: Graph building can be tested without execution infrastructure
4. **Code Reuse**: Shared utilities available to all graph builders
5. **Maintainability**: Smaller, focused classes

---

## Risks

1. **Test Breakage**: Tests may directly call `Qwen2Graph::execute*()` methods
2. **GraphOrchestrator Complexity**: May become too large (mitigate by extracting sub-components)
3. **Performance**: Additional indirection (likely negligible)

---

## Success Criteria

- [ ] All 164 unit tests pass
- [ ] `Qwen2Graph` contains no `execute*()` methods
- [ ] `Qwen2Graph` owns no executor or device contexts
- [ ] Shared utilities in `GraphBuildUtils.h`
- [ ] Integration tests pass
- [ ] Benchmark performance unchanged
