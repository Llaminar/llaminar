# Graph Infrastructure Decoupling Plan

**Status**: IN PROGRESS  
**Created**: April 2026  
**Goal**: Remove all Qwen-specific coupling from generic graph infrastructure so that adding a new model architecture (Llama, DeepSeek, Mistral) requires only creating files under `models/<arch>/` — no edits to `execution/`, `loaders/`, `config/`, or `kernels/`.

---

## Current State

The audit found **28 coupling points** (5 HIGH, 7 MEDIUM, ~16 LOW). The infrastructure is ~90% decoupled — coupling is concentrated in two factory files and two weight-naming files.

**Existing abstractions already in place**:
- `IGraphBuilder` interface (Qwen2Graph implements it)
- `ISchemaFactory` / `SchemaFactoryRegistry` (maps arch → schema)
- `IGraphConfigBuilder` / `createGraphConfigBuilder()` (maps arch → config builder)
- `GraphConfig` struct is fully model-agnostic

---

## Phase 1: GraphBuilderRegistry — Decouple factory from Qwen2Graph ✅ DONE

**Severity**: HIGH — blocks new model support  
**Files touched**: 3 new, 2 modified

### Problem
`InferenceRunnerFactory.cpp` directly `#include`s `Qwen2Graph.h` and has 4× `std::make_shared<Qwen2Graph>(graph_config, mpi_ctx)`. Adding a new model requires editing this generic factory.

### Solution
Create `GraphBuilderRegistry` (parallel to `SchemaFactoryRegistry`):
- Maps `architecture string → IGraphBuilder` factory function
- Lives in `src/v2/execution/local_execution/graph/GraphBuilderRegistry.h/cpp`
- Model-specific registration happens in model TUs (self-registration via static init)
- `InferenceRunnerFactory` calls `GraphBuilderRegistry::create("qwen2", config, mpi)` instead of `std::make_shared<Qwen2Graph>(config, mpi)`

### New files
- `src/v2/execution/local_execution/graph/GraphBuilderRegistry.h` — Registry class
- `src/v2/execution/local_execution/graph/GraphBuilderRegistry.cpp` — Implementation
- `tests/v2/unit/execution/local_execution/graph/Test__GraphBuilderRegistry.cpp` — Unit tests

### Changes
- `src/v2/execution/factory/InferenceRunnerFactory.cpp` — Replace `#include Qwen2Graph.h` + 4 instantiations with registry calls
- `src/v2/models/qwen/Qwen2Graph.cpp` — Add static self-registration
- `src/v2/models/qwen3/Qwen3Graph.cpp` — Add static self-registration (if exists, or Qwen3 reuses Qwen2Graph)

### Tests
- `Create_KnownArchitecture` — "qwen2" returns valid IGraphBuilder
- `Create_UnknownArchitecture_Throws` — "unknown_arch" throws
- `IsSupported_ReturnsCorrectly` — true for qwen2/qwen3, false for unknown
- `SupportedArchitectures_ListsAll` — returns all registered architectures
- `Create_MatchesArchitectureName` — returned builder's `architectureName()` matches

---

## Phase 2: Unify registries + eliminate SchemaFactoryRegistry model includes ✅ DONE

**Severity**: HIGH  
**Files touched**: 3 modified, 1 new

### Problem
`SchemaFactoryRegistry.cpp` directly `#include`s `Qwen2Schema.h` and `Qwen3Schema.h` with an if/else chain. Same pattern as the graph builder coupling.

### Solution
Apply the same self-registration pattern to `SchemaFactoryRegistry`:
- Model TUs register their schema factory during static init
- Remove `#include` of model headers from the registry `.cpp`
- Keep the existing `SchemaFactoryRegistry` API unchanged

### Changes
- `src/v2/execution/local_execution/graph/SchemaFactoryRegistry.h` — Added `registerFactory()`, `SchemaFactoryRegistrar`, `REGISTER_SCHEMA_FACTORY` macro
- `src/v2/execution/local_execution/graph/SchemaFactoryRegistry.cpp` — Replaced includes + if/else with dynamic map lookup (no model includes)
- `src/v2/models/qwen/Qwen2Graph.cpp` — Added schema self-registration for both "qwen2" and "qwen3"
- `src/v2/execution/local_execution/orchestrators/RankOrchestrator.cpp` — Made createForTest tolerant of unregistered architectures

### Tests
- `tests/v2/unit/execution/local_execution/graph/Test__SchemaFactoryRegistry.cpp` — 8 tests
- Existing `SchemaFactoryRegistry` tests continue to pass (API unchanged)

---

## Phase 3: Eliminate hardcoded Qwen constants from generic code

**Severity**: MEDIUM  
**Files touched**: 5 modified

### Problem
4 sites use `151936` (Qwen2.5 vocab) and `896` (Qwen2.5 hidden dim) as fallback defaults in generic infrastructure. These produce wrong workspace estimates for non-Qwen models.

### Sites
1. `DeviceGraphOrchestrator.cpp:884` — `hints.vocab_size = ... : 151936`
2. `OrchestrationRunnerFactory.cpp:157` — `compile_ctx.vocab_size = 151936`
3. `TreeToRunnerCompiler.h:125` — `int vocab_size = 151936`
4. `CUDAOpsKernels.cpp:1444` — `DEFAULT_VOCAB_SIZE = 151936`

### Solution
Replace hardcoded Qwen constants with generic sentinel values:
- Use `0` as sentinel → caller must provide actual value
- Where fallbacks are needed for backward compat, use conservative generic defaults (e.g., 65536 vocab, 4096 hidden) with a `LOG_WARN`
- Add `static_assert` or runtime checks where 0 would cause division-by-zero

### Changes
- `DeviceGraphOrchestrator.cpp` — Replace `151936` → `65536`, `896` → `4096` with warning
- `OrchestrationRunnerFactory.cpp` — Get values from model metadata, no hardcoded fallback
- `TreeToRunnerCompiler.h` — Use `0` sentinel, require caller to set
- `CUDAOpsKernels.cpp` — Require actual vocab_size parameter

### Tests
- Existing tests pass (values now come from model metadata)
- Add: `WorkspaceSizing_UsesModelMetadata_NotHardcodedDefaults`

---

## Phase 4: Move ModelConfig presets to model-specific code

**Severity**: MEDIUM  
**Files touched**: 2 modified, 1 new

### Problem
`IExecutionPlanBuilder.h` (generic interface) defines `ModelConfig::qwen2_0_5b()`, `qwen2_7b()`, `qwen2_72b()` convenience constructors.

### Solution
Move preset constructors to `src/v2/models/qwen/Qwen2ModelPresets.h`. Keep `ModelConfig` struct in the generic interface (it's model-agnostic).

### Changes
- `src/v2/execution/mpi_orchestration/IExecutionPlanBuilder.h` — Remove static preset methods
- `src/v2/models/qwen/Qwen2ModelPresets.h` — New file with preset functions
- Update any callers of `ModelConfig::qwen2_*()` to use new location

---

## Phase 5: Weight naming abstraction for LayerWeightStreamer

**Severity**: HIGH  
**Files touched**: 3 modified

### Problem
`LayerWeightStreamer::getLayerWeightNames()` hardcodes the 9 GGUF component names (attn_q, attn_k, etc.) that are specific to the Qwen2 transformer block structure. MoE models or models with different components won't work with weight streaming.

Note: The `blk.N.` prefix is GGUF-standard (used by all models in GGUF format), but the component names after the prefix ARE architecture-specific.

### Solution
Get weight component names from the schema (which already knows the architecture):
- Add `getLayerWeightComponents()` to `ISchemaFactory` (returns `{"attn_q.weight", "attn_k.weight", ...}`)
- `LayerWeightStreamer` queries the schema instead of hardcoding names
- `WeightPlacementMap` similar refactor

### Changes
- `src/v2/execution/local_execution/graph/GraphSchema.h` — Add `layerWeightComponents` to schema
- `src/v2/loaders/LayerWeightStreamer.cpp` — Use schema for weight names
- `src/v2/loaders/WeightPlacementMap.cpp` — Use schema for weight names

### Tests
- `LayerWeightStreamer_UsesSchemaComponentNames`
- `WeightPlacementMap_UsesSchemaComponentNames`

---

## Phase 6: Remove deprecated WeightManager methods

**Severity**: LOW  
**Files touched**: 2 modified

### Problem
8 `[[deprecated]]` methods (`isQKVWeight()`, `isFFNGateUpWeight()`, etc.) have zero live callers but remain in the codebase, cluttering the API.

### Solution
Delete the declarations from `WeightManager.h` and implementations from `WeightManager.cpp`.

---

## Phase 7: Fix Qwen-specific comments in generic code

**Severity**: LOW  
**Files touched**: ~15 modified (comment-only)

### Problem
~22 comment sites reference Qwen2 as the only example, creating a Qwen-centric impression.

### Solution
Update comments to use generic phrasing: "Model graphs (e.g., Qwen2Graph, LlamaGraph)" or remove model-specific references where not needed.

### Key files
- `DeviceGraphOrchestrator.h/cpp` — File-level doc comments, error messages
- `CollectiveContext.h`, `ICollectiveBackend.h` — Doc references
- `DeviceGraphExecutor.h` — Performance notes
- `InferenceRunnerFactory.h` — Scale comment

---

## Phase Order

| Phase | Severity | Estimated Scope | Dependency |
|-------|----------|-----------------|------------|
| 1     | HIGH     | 3 new + 3 modified files | None |
| 2     | HIGH     | 3 modified files | Phase 1 (same pattern) |
| 3     | MEDIUM   | 5 modified files | None |
| 4     | MEDIUM   | 2 modified + 1 new | None |
| 5     | HIGH     | 3 modified files | Phase 2 (uses schema) |
| 6     | LOW      | 2 modified files | None |
| 7     | LOW      | ~15 comment-only | None |

Phases 1-2 are the critical blockers for new model support.
Phases 3-5 eliminate friction and wrong defaults.
Phases 6-7 are cleanup.
