# DGO-Qwen2 Decoupling Plan

**Date**: March 31, 2026
**Status**: ✅ COMPLETE (all phases A-E done, all gates passed)
**Goal**: Make DeviceGraphOrchestrator and ForwardExecutionEngine fully model-type-agnostic by removing all direct dependencies on `Qwen2Graph`, `Qwen2ForwardInput`, and `Qwen2ForwardOutput`.

---

## Problem Statement

DeviceGraphOrchestrator (DGO) directly references Qwen2-specific types in 44+ locations across 4 files. This prevents supporting new model architectures (Qwen3, Llama, etc.) without modifying the orchestrator layer.

### Current Coupling Points

| File | `Qwen2Graph` | `Qwen2ForwardInput` | `Qwen2ForwardOutput` | Total |
|------|:---:|:---:|:---:|:---:|
| DeviceGraphOrchestrator.h | 6 | 6 | 0 | 12 |
| DeviceGraphOrchestrator.cpp | 6 | 9 | 7 | 22 |
| ForwardExecutionEngine.h | 0 | 4 | 3 | 7 |
| ForwardGraphTypes.h | 0 | 0 | 3 | 3 |

**Key insight**: `IGraphBuilder` already exists and `Qwen2Graph` already implements it, but IGraphBuilder only exposes 2 of the ~15 methods DGO calls.

---

## Acceptance Gates

Every phase must pass **all** of the following before proceeding:

1. **Build**: `cmake --build build_v2_integration --parallel` — all targets compile
2. **Unit tests**: `ctest --test-dir build_v2_integration -R "^V2_Unit_" --output-on-failure --parallel` — all pass
3. **Parity tests** (run serially):
   - `V2_Integration_Parity_Qwen2_SingleDevice`
   - `V2_Integration_Parity_Qwen3_SingleDevice`
   - `V2_Integration_Parity_Qwen2_LocalPP`
   - `V2_Integration_Parity_Qwen2_LocalTP`

---

## Phase A: Consolidate ForwardInput / ForwardOutput

**Rationale**: `Qwen2ForwardInput` extends `ForwardInput` with 5 fields that are
architecture-generic pipeline features (PP KV caches, external hidden state,
variable-length batching). `Qwen2ForwardOutput` is identical to `ForwardOutput`.
Rather than maintaining two parallel hierarchies, move the fields into the generic
types and eliminate the Qwen2-specific types.

### A.1 — Extend `ForwardInput` in `IGraphBuilder.h`

Move these fields from `Qwen2ForwardInput` into `ForwardInput`:

```cpp
struct ForwardInput {
    // ... existing fields ...

    // Pipeline Parallelism
    const std::unordered_map<DeviceId, IKVCache*>* pp_kv_caches = nullptr;
    TensorBase* external_hidden_state = nullptr;

    // Variable-length batching
    const std::vector<int>* sequence_lengths = nullptr;

    // Batched input (alternative to token_ids)
    struct Batch { const int* tokens; int len; int offset; };
    const Batch* batches = nullptr;
    int num_batches = 0;

    // Helper
    IKVCache* getKVCacheForDevice(const DeviceId& device) const;
};
```

### A.2 — Create type aliases in `Qwen2Graph.h`

Replace `struct Qwen2ForwardInput` and `struct Qwen2ForwardOutput` with:

```cpp
using Qwen2ForwardInput = ForwardInput;
using Qwen2ForwardOutput = ForwardOutput;
```

This makes the transition invisible to all existing callers — zero breakage.

### A.3 — Replace `Qwen2ForwardOutput` in ForwardGraphTypes.h

Change `GraphBuildResult` and `ForwardGraphCache` to use `ForwardOutput` directly.
Remove the `#include "Qwen2Graph.h"` from `ForwardGraphTypes.h`.

**Gate**: Build + all tests pass. Existing code still compiles because the aliases
resolve to the same types.

---

## Phase B: Extend `IGraphBuilder` Interface

**Rationale**: DGO calls ~15 methods on `graph_builder_`, but `IGraphBuilder` only
defines 4. We need to widen the interface so DGO can work through `IGraphBuilder*`
instead of `Qwen2Graph*`.

### Methods to add to `IGraphBuilder`

Grouped by concern. All have default no-op implementations so existing mocks and
tests compile without change.

#### Configuration access

```cpp
virtual const GraphConfig& config() const = 0;  // pure virtual — every builder has config
virtual std::string architectureName() const { return "unknown"; }
```

#### Weight / buffer management

```cpp
virtual void setWeights(const ModelWeights& weights) {}
virtual void setBuffers(const ModelBuffers& buffers) {}
virtual void setArena(BufferArena* arena) {}
virtual const ModelBuffers& buffers() const;  // default returns empty static
```

#### Schema / resolver

```cpp
virtual GraphSchema getSchema() const { return {}; }
virtual GraphResolverConfig getResolverConfig(int seq_len) const { return {}; }
```

#### Pipeline configuration

```cpp
virtual void setPipelineConfig(std::shared_ptr<PipelineConfig> config) {}
virtual void setPPContext(int from, int to, ILocalPPContext* ctx) {}
virtual void setTPContext(const std::string& domain, ILocalTPContext* ctx) {}
```

#### Graph building (PP variants)

```cpp
virtual ComputeGraph buildFullForwardGraph(const ForwardInput& input, ForwardOutput& output);
virtual ComputeGraph buildPartialForwardGraph(
    const ForwardInput& input, ForwardOutput& output,
    int first_layer, int last_layer, bool has_embedding, bool has_lm_head);
virtual ComputeGraph buildUnifiedPipelineGraph(const ForwardInput& input, ForwardOutput& output);
```

Default implementations throw `std::logic_error("not implemented")`.

#### Snapshot callback

```cpp
virtual void setSnapshotCallback(StageSnapshotCallback callback) {}
```

### Update `Qwen2Graph` overrides

Add `override` to all newly-virtual methods in `Qwen2Graph`. Nothing else changes
in `Qwen2Graph` since it already implements all of these as concrete methods.

### Update `MockGraphBuilder`

Add mock overrides for configuration access (`config()`) to the existing
`MockGraphBuilder` in `IGraphBuilder.h` so unit tests work.

**Gate**: Build + all tests pass.

---

## Phase C: Decouple DGO from Qwen2Graph

**Rationale**: With the interface now complete, DGO can operate through `IGraphBuilder*`.

### C.1 — Change member type

In `DeviceGraphOrchestrator.h`:

```cpp
// Before:
std::shared_ptr<Qwen2Graph> graph_builder_;
Qwen2Graph* graphBuilder();

// After:
std::shared_ptr<IGraphBuilder> graph_builder_;
IGraphBuilder* graphBuilder();
```

### C.2 — Remove Qwen2 includes from DGO.h

Replace:
```cpp
#include "../../../models/qwen/Qwen2Graph.h"
```
with forward declarations and `IGraphBuilder.h` include (already included transitively).

### C.3 — Use generic types in DGO APIs

Replace all `Qwen2ForwardInput` → `ForwardInput`, `Qwen2ForwardOutput` → `ForwardOutput`
in DGO.h and DGO.cpp. Since Phase A made these aliases, this is a search-and-replace.

### C.4 — Replace `Qwen2Graph::buildPositionIds()` static call

Change to `IGraphBuilder::buildPositionIds()` (already defined on the interface
as a static method).

### C.5 — Delegate `architecture()`

```cpp
// Before:
std::string architecture() const override { return "qwen2"; }

// After:
std::string architecture() const override {
    return graph_builder_ ? graph_builder_->architectureName() : "unknown";
}
```

### C.6 — Move Qwen2Graph construction out of DGO

The two constructors that call `std::make_shared<Qwen2Graph>(...)` must instead
accept `shared_ptr<IGraphBuilder>`. The factory (`InferenceRunnerFactory`) creates
the concrete `Qwen2Graph` and passes it in. 

DGO constructor signatures become:

```cpp
// Dependencies constructor (already generic)
DeviceGraphOrchestrator(Dependencies deps, const GraphConfig& config,
                        const GraphCacheConfig& cache_config = {});

// Graph builder injection constructor (already exists, change type)
DeviceGraphOrchestrator(std::shared_ptr<IGraphBuilder> graph_builder,
                        std::shared_ptr<MPIContext> mpi_ctx,
                        const GraphCacheConfig& cache_config = {});

// REMOVE: the (GraphConfig, MPIContext, GraphCacheConfig) constructor
// that internally creates Qwen2Graph. Move Qwen2Graph creation to callers.
```

### C.7 — Update DGO.cpp

Replace all `Qwen2ForwardInput` / `Qwen2ForwardOutput` references.
Replace the hardcoded `"Qwen2Schema"` string with the generic reference.

### C.8 — Update `GraphBuildSession`

Change `std::optional<Qwen2ForwardInput> input_` → `std::optional<ForwardInput> input_`.
Change `prepareInput()` return type → `ForwardInput`.
Change `forInput()` parameter → `const ForwardInput&`.

**Gate**: Build + all tests pass.

---

## Phase D: Propagate to Engine and Types

### D.1 — Decouple ForwardExecutionEngine.h

Replace all `Qwen2ForwardInput` → `ForwardInput`, `Qwen2ForwardOutput` → `ForwardOutput`
in `ForwardExecutionEngine.h/.cpp`.

### D.2 — Decouple ForwardGraphTypes.h

Replace `Qwen2ForwardOutput` → `ForwardOutput` in `GraphBuildResult` and
`ForwardGraphCache`. Remove `#include "Qwen2Graph.h"`.

### D.3 — Decouple IForwardExecutionHost

`buildForwardGraph(const ForwardInput&)`, `resolvePPCopyInfo(const ForwardInput&)`.

### D.4 — Update test files

Replace `Qwen2ForwardInput` / `Qwen2ForwardOutput` in:
- `Test__DeviceGraphOrchestrator.cpp`
- `Test__ForwardExecutionEngine.cpp`
- `Test__ForwardGraphTypes.cpp`
- `Test__MultiDomainOrchestrator.cpp`
- `Test__RankOrchestrator.cpp`

### D.5 — Update InferenceRunnerFactory.cpp

Factory constructs `std::make_shared<Qwen2Graph>(...)` and passes it to DGO via the
injection constructor. This is the **only file** that should know about `Qwen2Graph`
in the orchestration layer.

**Gate**: Build + all tests pass.

---

## Phase E: Cleanup

### E.1 — Remove Qwen2ForwardInput/Output aliases from Qwen2Graph.h

Once all callers use `ForwardInput` / `ForwardOutput` directly, delete the
`using Qwen2ForwardInput = ForwardInput;` aliases.

### E.2 — Verify no Qwen2 references remain in orchestration layer

```bash
grep -rn 'Qwen2' src/v2/execution/local_execution/orchestrators/
grep -rn 'Qwen2' src/v2/execution/local_execution/engine/
```

Should return zero matches.

**Gate**: Build + all tests pass. Final verification.

---

## File Change Summary

| File | Change |
|------|--------|
| `src/v2/execution/local_execution/graph/IGraphBuilder.h` | Extend ForwardInput, add ~12 virtual methods |
| `src/v2/models/qwen/Qwen2Graph.h` | Replace structs with aliases, add `override` keywords, add `architectureName()` |
| `src/v2/models/qwen/Qwen2Graph.cpp` | No changes (methods already exist) |
| `src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h` | Change member type, replace all Qwen2 types |
| `src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp` | Replace all Qwen2 types, remove direct construction |
| `src/v2/execution/local_execution/engine/ForwardExecutionEngine.h` | Replace all Qwen2 types |
| `src/v2/execution/local_execution/engine/ForwardExecutionEngine.cpp` | Replace all Qwen2 types |
| `src/v2/execution/local_execution/engine/ForwardGraphTypes.h` | Replace Qwen2ForwardOutput, remove include |
| `src/v2/execution/factory/InferenceRunnerFactory.cpp` | Construct Qwen2Graph explicitly, pass as IGraphBuilder |
| `tests/v2/unit/.../Test__DeviceGraphOrchestrator.cpp` | Replace types, use IGraphBuilder |
| `tests/v2/unit/.../Test__ForwardExecutionEngine.cpp` | Replace types |
| `tests/v2/unit/.../Test__ForwardGraphTypes.cpp` | Replace types |
| `tests/v2/unit/.../Test__MultiDomainOrchestrator.cpp` | Replace types |
| `tests/v2/unit/.../Test__RankOrchestrator.cpp` | Replace types |

---

## Non-Goals

- Renaming `DeviceGraphOrchestrator` class itself
- Refactoring `Qwen2Graph` internals
- Adding a second model architecture (Qwen3Graph already exists separately)
- Re-designing the `GraphBuildSession` builder pattern (it stays, just uses generic types)
