# Project Plan: Making Qwen2Graph 100% Declarative

**Date**: December 19, 2025  
**Author**: David Sanftenberg  
**Status**: Phase 5 Complete ✅ (Phases 1-5 Done)  
**Branch**: `feature/typed-residuals`

## Executive Summary

This project refactors `Qwen2Graph` from a mixed imperative/declarative design to a **purely declarative graph builder**. The goal is to establish a clean architecture pattern that can be reused for future model architectures (Qwen3Graph, LlamaGraph, etc.).

### Current State

`Qwen2Graph` currently has **two responsibilities**:
1. ✅ **Graph Building** (declarative) - `build*Graph()` methods construct ComputeGraph DAGs
2. ❌ **Graph Execution** (imperative) - `execute*()` methods run graphs and manage caching

This violates Single Responsibility Principle and makes the class harder to test, extend, and reason about.

### Target State

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DECLARATIVE LAYER                                   │
│  ┌─────────────────────┐   ┌─────────────────────┐   ┌──────────────────┐  │
│  │   Qwen2GraphBuilder │   │  Qwen3GraphBuilder  │   │  LlamaGraphBuilder│  │
│  │   (stateless)       │   │  (stateless)        │   │  (stateless)     │  │
│  └──────────┬──────────┘   └──────────┬──────────┘   └────────┬─────────┘  │
│             │                         │                       │             │
│             └─────────────────────────┼───────────────────────┘             │
│                                       │                                     │
│                                       ▼                                     │
│                          ┌────────────────────────┐                         │
│                          │    IGraphBuilder       │                         │
│                          │    (interface)         │                         │
│                          └────────────────────────┘                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          EXECUTION LAYER                                     │
│  ┌─────────────────────┐   ┌─────────────────────┐   ┌──────────────────┐  │
│  │   DeviceGraphExecutor     │   │  GraphCacheManager  │   │ DeviceContextMgr │  │
│  │   (executes DAGs)   │   │  (Phase 10 caching) │   │ (device contexts)│  │
│  └─────────────────────┘   └─────────────────────┘   └──────────────────┘  │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Qwen2PipelineOrchestrator                         │   │
│  │   - Owns GraphBuilder, Executor, CacheManager                        │   │
│  │   - Coordinates execution flow                                       │   │
│  │   - Manages state (position offset, seq_len changes)                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Issues Identified

### Issue 1: Execute Methods in Graph Builder
**Location**: `Qwen2Graph.cpp` lines 912-1175  
**Methods**: `executeForward()`, `executeAttention()`, `executeFFN()`, `executeLayer()`

```cpp
// CURRENT (mixed concerns)
bool Qwen2Graph::executeAttention(...) {
    ComputeGraph graph = buildAttentionGraph(...);  // Builds
    return executor_.execute(graph, ctx);            // Executes
}
```

**Problem**: Graph builder shouldn't execute graphs.

### Issue 2: Mutable State in Graph Builder
**Location**: `Qwen2Graph.cpp` lines 211-213, `Qwen2Graph.h` lines 520-525

```cpp
// State that gets mutated during graph building
int current_batch_size_ = 0;
int current_seq_len_ = 0;
std::vector<int> position_ids_buffer_;
int last_pos_offset_ = -1;
```

**Problem**: Declarative builders should be stateless.

### Issue 3: Graph Caching Logic in Graph Builder
**Location**: `Qwen2Graph.cpp` lines 1000-1130

```cpp
// Caching logic embedded in execute methods
if (graph_caching_enabled_ && seq_len == 1 && ...) {
    auto& cache = layer_graph_cache_[layer_idx];
    if (cache.attention_decode && cache.valid) {
        updateCachedGraphParams(...);
        return executor_.execute(*cache.attention_decode, ctx);
    }
}
```

**Problem**: Caching strategy shouldn't be coupled to graph definition.

### Issue 4: Placeholder Nodes with nullptr Stages
**Location**: `Qwen2Graph.cpp` lines 248-254

```cpp
for (int layer = 0; layer < config_.n_layers; ++layer) {
    graph.addNode(layer_name, nullptr, device_idx);  // Hole in graph!
}
```

**Problem**: Graphs should be complete and self-describing.

### Issue 5: Device Context Lazy Initialization
**Location**: `Qwen2Graph.cpp` lines 162-180

```cpp
IDeviceContext* Qwen2Graph::getDeviceContext(int device_idx) {
    // Side effect: creates and caches context
    device_contexts_[device_idx] = std::move(ctx);
}
```

**Problem**: Graph builder shouldn't manage device lifecycle.

### Issue 6: Position IDs Built During Graph Construction
**Location**: `Qwen2Graph.cpp` line 243

```cpp
position_ids_buffer_ = buildPositionIds(input.seq_len, input.batch_size, input.position_offset);
```

**Problem**: Runtime data computation in graph builder.

---

## Implementation Phases

### Phase 1: Extract Execution to Orchestrator ✅ COMPLETE

**Goal**: Move all `execute*()` methods out of Qwen2Graph into a new `GraphOrchestrator` class.

**Status**: ✅ Complete (December 19, 2025)

**Files Created**:
- `src/v2/pipelines/qwen/GraphOrchestrator.h` (~297 lines)
- `src/v2/pipelines/qwen/GraphOrchestrator.cpp` (~500 lines)
- `tests/v2/unit/Test__GraphOrchestrator.cpp` (~340 lines)

**Changes Made**:
1. ✅ Created `GraphOrchestrator` that wraps:
   - `Qwen2Graph` (as `graph_builder_` via non-owning shared_ptr)
   - `DeviceGraphExecutor` (references `graph_builder_->executor()`)
   - `IDeviceContext` cache (`device_contexts_`)
   - Graph cache (`layer_graph_cache_`)

2. ✅ Implemented these methods in `GraphOrchestrator`:
   - `executeLayer()` - Orchestrates attention + FFN for a layer
   - `executeAttention()` - Builds and executes attention graph with caching
   - `executeFFN()` - Builds and executes FFN graph with caching
   - `getDeviceContext()` - Lazy device context creation
   - Cache management: `initializeGraphCache()`, `clearCache()`, `invalidateGraphCache()`
   - Statistics: `getCacheStats()`

3. ✅ Updated `Qwen2Pipeline` to use `GraphOrchestrator`:
   - Added `orchestrator_` member
   - Create orchestrator after `layer_executor_` initialization
   - `run_layer()` prefers orchestrator when available

4. ✅ All tests pass:
   - 21/21 GraphOrchestrator unit tests (18 passed, 3 skipped for device context)
   - 15/15 Graph caching tests pass
   - All V2 unit tests pass

**Actual LOC**: ~1100 new lines (including tests)

**Note**: The original `execute*()` methods remain in Qwen2Graph for backward compatibility during migration. They will be deprecated in Phase 2.

---

### Phase 2: Remove Mutable State from Graph Builder

**Goal**: Make `Qwen2Graph` stateless by passing all context explicitly.

**Status**: Not Started

**Changes**:
1. Remove member variables:
   - `current_batch_size_`
   - `current_seq_len_`
   - `position_ids_buffer_`
   - `last_pos_offset_`
   - `device_contexts_`
   - `layer_graph_cache_`
   - `graph_caching_enabled_`

2. Update method signatures to pass context explicitly:
   ```cpp
   // BEFORE
   ComputeGraph buildLMHeadGraph(TensorBase* hidden, TensorBase* logits, int device_idx);
   
   // AFTER
   ComputeGraph buildLMHeadGraph(
       TensorBase* hidden, TensorBase* logits, 
       int seq_len, int batch_size,  // Explicit context
       int device_idx);
   ```

3. Move state to `GraphOrchestrator`

**Estimated LOC**: ~100 changes

---

### Phase 3: Eliminate Placeholder Nodes

**Goal**: Build complete graphs without `nullptr` stages.

**Changes**:
1. Remove placeholder node pattern from `buildFullForwardGraph()`

2. Option A: **Inlined Layer Graphs**
   ```cpp
   ComputeGraph buildFullForwardGraph(...) {
       // ... embedding ...
       for (int layer = 0; layer < n_layers; ++layer) {
           auto attn_graph = buildAttentionGraph(layer, ...);
           auto ffn_graph = buildFFNGraph(layer, ...);
           graph.merge(attn_graph, prev_node);
           graph.merge(ffn_graph, attn_last_node);
       }
       // ... lm_head ...
   }
   ```

3. Option B: **Composite Stage** (cleaner)
   ```cpp
   // New stage type that wraps a sub-graph
   class CompositeStage : public IComputeStage {
       ComputeGraph subgraph_;
       bool execute(IDeviceContext*) override {
           return DeviceGraphExecutor::execute(subgraph_, ctx);
       }
   };
   ```

**Estimated LOC**: ~150 changes

---

### Phase 4: Position IDs as External Input ✅ COMPLETE

**Goal**: Remove position ID computation from graph builder.

**Status**: ✅ Complete (December 19, 2025)

**Implementation**: Chose Option A (External Input) for simplicity.

**Changes Made**:
1. ✅ Added `const int* position_ids` to `Qwen2ForwardInput` struct
2. ✅ Made `buildPositionIds()` a public static utility method
3. ✅ Marked `position_ids_buffer_` member as deprecated
4. ✅ Updated `buildFullForwardGraph()` to use `input.position_ids` with fallback
5. ✅ Updated `GraphOrchestrator::executeForward()` to build and pass position IDs

**API Change**:
```cpp
struct Qwen2ForwardInput {
    const int* token_ids = nullptr;
    const int* position_ids = nullptr;  // NEW: Pre-computed externally (required)
    int batch_size = 1;
    int seq_len = 0;
    int position_offset = 0;  // Legacy: used if position_ids == nullptr
    // ...
};

// Static utility for callers to build position IDs
static std::vector<int> Qwen2Graph::buildPositionIds(int seq_len, int batch_size, int offset);
```

**Backward Compatibility**:
- If `position_ids == nullptr`, the deprecated internal path is used (with LOG_DEBUG warning)
- Existing code continues to work without changes

**Tests**: All 172 V2 unit tests pass

---

### Phase 5: Extract IGraphBuilder Interface ✅ COMPLETE

**Goal**: Define common interface for all model graph builders, with MockGraphBuilder for unit testing.

**Status**: ✅ Complete (December 19, 2025)

**Files Created**:
- `src/v2/execution/IGraphBuilder.h` (~350 lines) - Interface + MockGraphBuilder + input/output structs
- `tests/v2/unit/Test__IGraphBuilder.cpp` (~490 lines) - Comprehensive unit tests

**Files Modified**:
- `src/v2/pipelines/qwen/Qwen2Graph.h` - Added IGraphBuilder inheritance
- `src/v2/pipelines/qwen/Qwen2Graph.cpp` - Added interface implementations
- `tests/v2/CMakeLists.txt` - Added v2_test_igraph_builder target

**Interface Design**:
```cpp
// Generic input/output structs
struct ForwardInput {
    const int* token_ids = nullptr;
    const int* position_ids = nullptr;
    int batch_size = 1;
    int seq_len = 0;
    int position_offset = 0;
    int device_idx = 0;
    void* kv_cache = nullptr;  // Type-erased for interface
};

struct ForwardOutput {
    TensorBase* logits = nullptr;
    TensorBase* hidden = nullptr;
};

struct LayerContext {
    int layer_idx = 0;
    int seq_len = 0;
    int device_idx = 0;
    const int* position_ids = nullptr;
    void* kv_cache = nullptr;
};

// Abstract interface
class IGraphBuilder {
public:
    virtual ~IGraphBuilder() = default;
    
    // Core graph building methods (generic signatures)
    virtual ComputeGraph buildForwardGraph(const ForwardInput& input, ForwardOutput& output) = 0;
    virtual ComputeGraph buildLayerGraph(const LayerContext& ctx) = 0;
    
    // Configuration queries
    virtual int numLayers() const = 0;
    virtual int hiddenDim() const = 0;
    virtual bool isInitialized() const = 0;
    
    // Static utility
    static std::vector<int> buildPositionIds(int seq_len, int batch_size, int offset);
};
```

**MockGraphBuilder Design**:
Uses factory pattern to avoid ComputeGraph copy issues (deleted copy ctor):
```cpp
class MockGraphBuilder : public IGraphBuilder {
public:
    // Factory setters - create graphs on demand
    void setForwardGraphFactory(std::function<ComputeGraph(const ForwardInput&, ForwardOutput&)> factory);
    void setLayerGraphFactory(std::function<ComputeGraph(const LayerContext&)> factory);
    void setLayerGraphFactory(int layer_idx, std::function<ComputeGraph(const LayerContext&)> factory);
    
    // Call tracking
    int buildForwardGraphCallCount() const;
    int buildLayerGraphCallCount() const;
    const ForwardInput* lastForwardInput() const;
    const LayerContext& lastLayerContext() const;
    
    // Configuration
    void setNumLayers(int n);
    void setHiddenDim(int d);
    void setInitialized(bool init);
    void resetCallCounts();
};
```

**Qwen2Graph Integration**:
```cpp
class Qwen2Graph : public IGraphBuilder {
public:
    // IGraphBuilder interface implementations
    ComputeGraph buildForwardGraph(const ForwardInput& input, ForwardOutput& output) override;
    ComputeGraph buildLayerGraph(const LayerContext& ctx) override;
    int numLayers() const override { return config_.n_layers; }
    int hiddenDim() const override { return config_.d_model; }
    bool isInitialized() const override { return weights_ != nullptr; }
};
```

**Test Coverage** (27 tests):
- MockGraphBuilder basics: 3 tests
- buildForwardGraph: 6 tests
- buildLayerGraph: 6 tests
- Reset/state management: 1 test
- buildPositionIds utility: 4 tests
- Polymorphism: 2 tests
- Structure defaults: 3 tests
- Edge cases: 2 tests

**Results**: All 173 V2 unit tests pass (was 172, +1 for IGraphBuilder)

---

### Phase 6: Device Context Factory (Optional)

**Goal**: Move device context creation out of graph builder into a dedicated factory.

**Status**: Not Started

---

## Migration Strategy

### Step 1: Create Orchestrator (Non-Breaking) ✅
- Add `GraphOrchestrator` alongside existing code
- `Qwen2Pipeline` can use either path via feature flag

### Step 2: Deprecate Execute Methods
- Mark `Qwen2Graph::execute*()` as `[[deprecated]]`
- Log warnings when called
- Update all callers to use orchestrator

### Step 3: Remove Deprecated Code
- Delete `execute*()` methods from `Qwen2Graph`
- Delete mutable state members
- `Qwen2Graph` becomes stateless

### Step 4: Rename for Clarity
- Rename `Qwen2Graph` → `Qwen2GraphBuilder` (optional, for clarity)

---

## Test Plan

### Unit Tests to Add
1. `Test__GraphOrchestrator.cpp` - Orchestrator execution tests
2. Update `Test__Qwen2GraphCaching.cpp` - Test caching via orchestrator
3. `Test__Qwen2GraphBuilderStateless.cpp` - Verify no state leakage

### Integration Tests
1. Verify `Qwen2Pipeline` works with orchestrator
2. Verify graph caching still works (Phase 10)
3. Verify MPI tensor parallelism unchanged

### E2E Tests
1. Run existing E2E parity tests
2. Benchmark to ensure no performance regression

---

## Success Criteria

| Criterion | Metric |
|-----------|--------|
| No execute methods in Qwen2Graph | 0 `execute*()` methods |
| No mutable state in Qwen2Graph | 0 mutable member variables (except config) |
| No placeholder nodes | 0 `nullptr` stages in built graphs |
| Tests pass | 100% existing tests + new tests |
| No performance regression | < 1% throughput change |

---

## Timeline

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Phase 1: Extract Orchestrator | 2-3 hours | None |
| Phase 2: Remove State | 1-2 hours | Phase 1 |
| Phase 3: Eliminate Placeholders | 1-2 hours | Phase 2 |
| Phase 4: Position IDs | 30 min | Phase 1 |
| Phase 5: IGraphBuilder | 1 hour | Phase 2 |
| Testing & Validation | 1-2 hours | All |

**Total Estimated Time**: 6-10 hours

---

## Files Affected

### New Files
- `src/v2/pipelines/qwen/GraphOrchestrator.h`
- `src/v2/pipelines/qwen/GraphOrchestrator.cpp`
- `tests/v2/unit/Test__GraphOrchestrator.cpp`

### Modified Files
- `src/v2/pipelines/qwen/Qwen2Graph.h` - Remove execute methods, state
- `src/v2/pipelines/qwen/Qwen2Graph.cpp` - Remove execute methods, state
- `src/v2/pipelines/qwen/Qwen2Pipeline.cpp` - Use orchestrator
- `src/v2/pipelines/qwen/CMakeLists.txt` - Add new files
- `tests/v2/CMakeLists.txt` - Add orchestrator tests

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Performance regression | Low | High | Benchmark before/after each phase |
| Breaking MPI tensor parallelism | Low | High | Run MPI tests after each phase |
| Graph caching broken | Medium | Medium | Dedicated caching tests |
| Large refactor scope | Medium | Medium | Incremental phases with feature flags |

---

## Appendix: Current vs Target Method Locations

| Method | Current Location | Target Location |
|--------|-----------------|-----------------|
| `buildAttentionGraph()` | Qwen2Graph | Qwen2Graph (unchanged) |
| `buildFFNGraph()` | Qwen2Graph | Qwen2Graph (unchanged) |
| `buildFullForwardGraph()` | Qwen2Graph | Qwen2Graph (modified) |
| `executeForward()` | Qwen2Graph | **GraphOrchestrator** |
| `executeAttention()` | Qwen2Graph | **GraphOrchestrator** |
| `executeFFN()` | Qwen2Graph | **GraphOrchestrator** |
| `executeLayer()` | Qwen2Graph | **GraphOrchestrator** |
| `getDeviceContext()` | Qwen2Graph | **GraphOrchestrator** |
| Graph caching | Qwen2Graph | **GraphOrchestrator** |
| `buildPositionIds()` | Qwen2Graph | External / Stage |
