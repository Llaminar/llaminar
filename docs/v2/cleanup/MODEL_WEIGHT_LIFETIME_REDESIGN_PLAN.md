# Model Weight Lifetime Redesign Project Plan

**Date**: 2026-04-30  
**Status**: Proposed  
**Scope**: Model loading, weight identity, sharding, repacking, prepared-weight ownership, graph binding, and replay behavior  
**Primary files**: `src/v2/loaders/`, `src/v2/tensors/`, `src/v2/kernels/KernelFactory.*`, `src/v2/execution/factory/`, `src/v2/execution/local_execution/orchestrators/`, `src/v2/models/`

---

## Executive Summary

The current weight lifecycle mostly works, but the ownership model is implicit. A model weight can be loaded by `ModelLoader`, cached by `WeightManager`, cloned or sliced for TP/PP, uploaded or marked host-resident, repacked into backend-specific GPU layouts, registered in static `KernelFactory` registries, captured as raw pointers in graph stage params, and reused during graph replay. No single object describes the complete identity and lifetime of that weight.

This plan makes weight lifetime explicit and auditable:

1. Orchestration produces an immutable `InferenceStrategy` and `WeightPlan`.
2. `ModelLoader` creates source tensors with stable `WeightIdentity` metadata.
3. `WeightManager` materializes all required sharded/sliced/device-specific weight bindings from the plan.
4. A model-context-owned `PreparedWeightStore` owns GPU pools, prepared GEMM handles, prepared embeddings, CPU packed state, and any backend-specific packed payloads.
5. Graph construction consumes a frozen `ModelWeightSet`/`WeightBindingSet`; graph construction no longer calls `getWeightForDevice()` or mutates weight state.
6. Graph replay treats weights as immutable. Only dynamic activation, position, KV cache, and sampling state changes.
7. `KernelFactory` becomes a light `KernelRegistry`: it selects kernel implementations but does not own model-weight lifetime.

The target is not just cleanup. It is a correctness boundary: after weight finalization, missing prepared weights, wrong devices, stale host pointers, and ambiguous tensor identities become validation failures rather than runtime surprises.

---

## Problem Statement

Weight lifetime is currently spread across multiple layers:

| Layer | Current responsibility |
|-------|------------------------|
| `ModelLoader` | Loads GGUF metadata and raw/source tensors, including row/column/expert slices. |
| `WeightManager` | Caches source tensors, clones per-device tensors, performs rank/device sharding, prepares/uploads weights, releases host data. |
| `KernelFactory` | Selects kernels, but also owns static prepared GEMM/embedding registries, sliced GEMM cache, fused caches, and device GEMM engines. |
| `InferenceRunnerFactory` | Eager-loads weights, calls weight finalization, then builds model weight callbacks. |
| `RankOrchestrator` | Reconfigures `WeightManager` for LocalTP/PP, finalizes weights for devices, defers release for lazy graph build. |
| Graph builders | Convert `WeightManager` callbacks into raw `TensorBase*` stage params. |
| Stages | Resolve prepared GEMM handles at execution time through `KernelFactory`. |

This creates several recurring failure modes:

1. **No formal identity**: `TensorBase` has `debug_name_`, coherence state, and device fields, but not a stable identity that answers what model, logical weight, slice, TP domain, PP stage, expert, derivation, and device it represents.
2. **Static prepared-weight ownership**: prepared GEMM and embedding state is owned by `KernelFactory` static registries, but logically belongs to a model context or runtime context.
3. **Graph build can still resolve weights**: `ModelWeights::get_layer_weights` is a callback. It can pull from `WeightManager` during graph construction, after partial finalization or host release decisions.
4. **Host release is timing-sensitive**: LocalTP and PP defer host data release because graph construction and MoE packing may still need raw data. That timing is currently controlled by comments and ordering rather than a typed lifecycle state.
5. **Pointer identity is fragile**: stages store raw `TensorBase*`. Kernel cache aliases now mitigate post-release lookup misses, but ownership and validity are still implicit.
6. **Mode-specific choreography**: SingleDevice, LocalTP, Global/NodeLocalTP, LocalPP, and Hybrid PP+TP all use slightly different preparation and release sequences.
7. **Traceability is weak**: when a stage gets the wrong weight or a prepared-GEMM lookup misses, logs rarely show the complete origin and intended residency/preparation story.

---

## Goals

- Define a clear source -> derived -> resident -> prepared -> frozen lifecycle for all model weights.
- Make every tensor/view/slice/prepared handle traceable to a stable logical weight identity.
- Move prepared-weight lifetime out of static `KernelFactory` state and into model/runtime-owned objects.
- Make graph construction consume frozen, already-prepared weight bindings.
- Make graph replay completely weight-immutable.
- Support all existing inference modes: SingleDevice, LocalTP, NodeLocalTP/GlobalTP, LocalPP, Hybrid PP+TP, CPU decode participation, tied embeddings, MoE expert tensors, shared experts, GDN weights, and GPU-side repack pipelines.
- Allow incremental migration without breaking existing tests or requiring a flag day.

## Non-Goals

- Rewriting quantized GEMM kernels.
- Replacing the GPU weight loading pipeline itself.
- Removing `WeightManager` entirely.
- Changing mathematical sharding semantics.
- Solving dynamic MoE expert rebalancing in the first pass. The design should accommodate it, but it should not block the base lifecycle migration.

---

## Current Mode-by-Mode Lifetime

### SingleDevice

1. `InferenceRunnerFactory::createDeviceGraphOrchestratorImpl()` configures `WeightManager` with schema sharding and dimensions.
2. `configureOrchestratorWeightsImpl()` eager-loads layer/global weights into `WeightManager`.
3. `WeightManager::finalizeForDevice(device)` prepares GEMM weights, uploads non-GEMM weights, prepares embeddings, and may release host data.
4. `buildWeights()` returns raw pointers through `ModelWeights`.
5. Graph build stores raw pointers in stage params.
6. Stages resolve prepared GEMM handles lazily through `KernelFactory`.

Main issue: graph build is after finalization and still uses `getWeightForDevice()` callbacks. The prepared state exists, but the graph has no typed binding to it.

### LocalTP

1. `RankOrchestrator` creates a `TensorParallelConfig` from `ILocalTPContext` and configures the shared `WeightManager`.
2. `WeightManager::finalizeForDevices(devices, release_host_data=false)` preloads/clones/shards for all local devices, runs GPU/CPU preparation per device, and retains host data.
3. `RankOrchestrator` creates one device runner per TP device, often in parallel.
4. Host release is deferred until after first forward because some graphs and MoE expert packing may still be lazy.

Main issue: the lifecycle needs two finalization boundaries: device preparation, then graph materialization, then host release. Today this is implicit.

### NodeLocalTP and GlobalTP

1. Execution plans define TP domain participation and weight shard information.
2. Each rank builds its local runner and loads its rank/domain shard.
3. `GlobalTPContext` can report `NODE_LOCAL` when ranks are on the same node, but the weight lifetime is still per-rank rather than per-local-device clone.
4. Collectives use TP context or MPI-stage paths.

Main issue: rank/domain identity should be part of `WeightIdentity`, not inferred from MPI rank, `GraphConfig`, or a tensor shape.

### LocalPP

1. `RankOrchestrator` builds one stage runner per PP stage.
2. All stage runners share one `ModelContext` and one `WeightManager`.
3. Each stage loads/prepares only its layer range plus embedding or LM head ownership.
4. Host data is released only after all stages are prepared, unless CPU stages are present.

Main issue: the shared `WeightManager` is reconfigured across stages. `WeightManager::configure()` intentionally preserves loaded tensors. This is a valid behavior but should become explicit model-wide planning instead of reconfiguration choreography.

### Hybrid PP+TP

1. A PP stage may itself be a TP domain.
2. The outer PP runner creates nested `RankOrchestrator` instances for TP-domain stages.
3. The nested TP domain uses `finalizeForDevices(..., release_host_data=false)` and defers release to the outer owner.

Main issue: this is the hardest case for ownership. A single logical model may have stage-local TP domains, each with multiple devices and stage layer ranges. The plan must represent this as a tree of weight bindings, not as nested side effects.

---

## Target Architecture

### Core Types

#### `ModelContextId`

Stable identifier for a loaded model context. It should be assigned when `ModelContext` is created and logged in every weight binding and prepared handle.

```cpp
struct ModelContextId {
    uint64_t value;
};
```

#### `WeightIdentity`

Stable logical identity for source and derived weights.

```cpp
enum class WeightRole {
    Embedding,
    LMHead,
    OutputNorm,
    AttentionQ,
    AttentionK,
    AttentionV,
    AttentionWO,
    FusedQKV,
    GDNProjection,
    GDNSsmParam,
    FFNGate,
    FFNUp,
    FFNDown,
    MoERouter,
    MoEExpertGate,
    MoEExpertUp,
    MoEExpertDown,
    SharedExpertGate,
    SharedExpertUp,
    SharedExpertDown,
    Norm,
    Bias,
    Other,
};

enum class WeightDerivationKind {
    Source,
    RowSlice,
    ColumnSlice,
    ExpertSlice,
    DeviceClone,
    TiedAlias,
    FusedSubblockConcat,
    DecodeShard,
    RebalancedExpertReplica,
};

struct WeightIdentity {
    ModelContextId model_id;
    uint64_t logical_id;
    uint64_t instance_id;
    std::string canonical_name;
    WeightRole role;
    WeightDerivationKind derivation;
    std::optional<uint64_t> source_instance_id;
    int layer = -1;
    int expert = -1;
    int pp_stage = -1;
    int tp_domain = -1;
    int tp_rank_or_device_index = 0;
};
```

#### `WeightSliceSpec`

Records exactly how a derived tensor maps back to the source.

```cpp
struct WeightSliceSpec {
    size_t source_rows = 0;
    size_t source_cols = 0;
    size_t row_start = 0;
    size_t row_count = 0;
    size_t col_start = 0;
    size_t col_count = 0;
    size_t expert_start = 0;
    size_t expert_count = 0;
    bool inner_is_presliced = false;
};
```

#### `WeightResidency`

Describes where the raw tensor lives and whether host data may be released.

```cpp
enum class WeightHostPolicy {
    RequiredForCPUExecution,
    RequiredUntilGraphMaterialized,
    ReleasableAfterPreparation,
    Released,
};

struct WeightResidency {
    DeviceId home_device = DeviceId::cpu();
    std::optional<DeviceId> resident_device;
    TensorCoherenceState coherence = TensorCoherenceState::HOST_ONLY;
    WeightHostPolicy host_policy = WeightHostPolicy::RequiredUntilGraphMaterialized;
    bool raw_host_data_available = true;
    bool raw_device_data_valid = false;
};
```

#### `PreparedWeightRef`

Opaque reference to model-context-owned prepared state.

```cpp
enum class PreparedWeightKind {
    None,
    CpuPackedGemm,
    CudaInt8PackedGemm,
    RocmInt8PackedGemm,
    PreparedEmbedding,
    MoeExpertSlab,
};

struct PreparedWeightRef {
    ModelContextId model_id;
    uint64_t binding_id;
    PreparedWeightKind kind = PreparedWeightKind::None;
    DeviceId device = DeviceId::cpu();
};
```

#### `WeightBinding`

The object graph builders and stages should consume.

```cpp
struct WeightBinding {
    uint64_t binding_id;
    WeightIdentity identity;
    WeightSliceSpec slice;
    WeightResidency residency;
    TensorBase* tensor = nullptr;
    std::optional<PreparedWeightRef> prepared;
    bool immutable = false;
};
```

#### `FrozenModelWeightSet`

Immutable collection produced after materialization and preparation.

```cpp
class FrozenModelWeightSet {
public:
    const WeightBinding& global(const std::string& canonical_name) const;
    const WeightBinding& layer(int layer_idx, const std::string& suffix) const;
    const WeightBinding* optionalLayer(int layer_idx, const std::string& suffix) const;
    std::vector<const WeightBinding*> forDevice(DeviceId device) const;
    void validateForGraph(const GraphConfig& config) const;
};
```

### Component Ownership

```
ModelContext
  owns ModelLoader
  owns WeightManager
  owns PreparedWeightStore
  owns FrozenModelWeightSet(s)

WeightManager
  owns source/derived TensorBase shared_ptrs
  materializes WeightBinding objects from WeightPlan
  does not own executable kernels

PreparedWeightStore
  owns GPU weight pools, prepared GEMM handles, prepared embeddings,
  CPU packed weights, and MoE packed slabs

KernelRegistry
  selects kernel classes and device-scoped non-weight kernels
  does not own model-weight prepared payloads

GraphBuilder / ComputeStage
  holds WeightBindingRef or PreparedWeightRef
  treats all weights as immutable
```

---

## Required Invariants

These invariants should be enforced in Debug and Integration builds first, then promoted where cheap.

1. Every model weight tensor has a `WeightIdentity` before it is inserted into any cache.
2. Every derived tensor records its source identity and slice spec.
3. Every graph-bound GEMM weight has a prepared handle before graph execution on GPU.
4. Every graph-bound non-GEMM weight is resident on the target device or explicitly host-resident with a prepared device-side consumer.
5. No graph builder calls `WeightManager::getWeightForDevice()` after the `FrozenModelWeightSet` is created.
6. No stage resolves a missing GPU prepared weight lazily. Missing prepared state is an initialization error.
7. Host data is released only after all consumers that require host data have declared completion.
8. `KernelFactory`/`KernelRegistry` cache keys never depend on raw `TensorBase*` as the sole identity for model weights.
9. Graph replay cannot mutate weight bindings, residency policy, or prepared handles.
10. Mode selection is queryable and serialized into a weight-binding audit report.

---

## Phased Roadmap

Each phase is intended to be independently buildable. Early phases are mostly additive and observability-focused. Later phases move ownership.

### Phase 0: Baseline Audit and Lifecycle Map

**Goal**: Document and instrument current behavior before changing it.

**Tasks**:

- Add a `WeightLifecycleTrace` debug facility gated by `LLAMINAR_WEIGHT_LIFECYCLE_TRACE=1`.
- Log load, slice, clone, preload, prepare, register, host release, graph bind, and replay events.
- Add a per-run summary table with counts by role, derivation, device, prepared kind, and host release policy.
- Add mode labels: `SingleDevice`, `LocalTP`, `NodeLocalTP`, `GlobalTP`, `LocalPP`, `HybridPPTP`.
- Record all places that still call `getWeightForDevice()` during graph construction.

**Files likely touched**:

- `src/v2/loaders/WeightManager.*`
- `src/v2/loaders/WeightTypes.h`
- `src/v2/execution/factory/InferenceRunnerFactory.cpp`
- `src/v2/execution/local_execution/orchestrators/RankOrchestrator.cpp`
- `src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.*`

**Validation gate**:

- No behavior changes.
- Unit and integration tests pass.
- Trace output for SingleDevice and LocalTP clearly shows weight counts and preparation state.

---

### Phase 1: Introduce Weight Identity Metadata

**Goal**: Give every tensor/view/slice/clone a stable identity without changing ownership yet.

**Tasks**:

- Add `WeightIdentity`, `WeightSliceSpec`, and `WeightResidency` definitions.
- Add optional metadata storage to `TensorBase` or a `WeightMetadataRegistry` owned by `WeightManager`.
- Assign source identities when `ModelLoader` creates tensors.
- Propagate identities through:
  - `loadTensorRowSlice()`
  - `loadTensorColumnSlice()`
  - `loadTensorExpertSlice()`
  - `TensorSlice`
  - `cloneTensorForDevice()`
  - tied embedding aliasing
  - decode shard creation
- Add `tensor->debugName()` fallback generation from identity fields.
- Add assertions that derived tensors never lose source identity.

**Design choice**:

Prefer a sidecar `WeightMetadataRegistry` first. Direct `TensorBase` fields can come later after the shape of the metadata stabilizes.

**Validation gate**:

- Existing behavior unchanged.
- New unit tests verify identity propagation for source, row slice, column slice, expert slice, device clone, tied alias, and decode shard.

---

### Phase 2: Add WeightPlan and Materialization Planning

**Goal**: Represent all required logical weights for an inference strategy before loading/preparing them.

**Tasks**:

- Introduce `InferenceStrategy` derived from `RankExecutionPlan`, `GraphConfig`, and TP/PP contexts.
- Introduce `WeightPlan` with per-binding requirements:
  - canonical name
  - required/optional
  - role
  - layer/expert/stage/domain
  - sharding mode
  - target device(s)
  - host policy
  - expected prepared kind
- Build `WeightPlan` from model schema plus orchestration mode.
- Add mode-specific plan builders:
  - SingleDevice
  - LocalTP
  - NodeLocalTP/GlobalTP
  - LocalPP
  - Hybrid PP+TP
- Include virtual weights such as tied `output.weight` -> `token_embd.weight` in the plan.
- Add a `WeightPlan::renderAuditTable()` using libfort for structured diagnostics.

**Important constraints**:

- `WeightManager::configure()` must continue preserving loaded tensors. PP and nested TP currently depend on this.
- The plan builder must not eagerly clear `cache_` or `per_device_cache_`.

**Validation gate**:

- Plan-only tests for all modes.
- Plan output exactly identifies which devices/ranks own embedding, final norm, LM head, dense layer weights, GDN weights, MoE router weights, MoE expert tensors, and shared expert tensors.

---

### Phase 3: Materialize Frozen Weight Bindings

**Goal**: Convert `WeightPlan` into a `FrozenModelWeightSet` while still using current `WeightManager` internals.

**Tasks**:

- Add `WeightManager::materialize(const WeightPlan&)`.
- Materialization loads source tensors and creates all device/rank/stage-specific derived tensors.
- Return a mutable `ModelWeightSetBuilder` during materialization.
- After preparation succeeds, call `freeze()` to produce `FrozenModelWeightSet`.
- Add `FrozenModelWeightSet` accessors for global and layer weights.
- Adapt existing `ModelWeights` construction to read from `FrozenModelWeightSet` instead of calling `getWeightForDevice()`.
- Keep compatibility path for existing graph builders until Phase 6.

**Validation gate**:

- SingleDevice and LocalTP can build `ModelWeights` from frozen bindings.
- No graph-bound weight is missing identity, slice, device, or prepared-kind expectation.

---

### Phase 4: PreparedWeightStore Compatibility Layer

**Goal**: Introduce model-context-owned prepared-weight storage while still delegating to current `KernelFactory` APIs internally.

**Tasks**:

- Add `PreparedWeightStore` owned by `ModelContext` or a new `ModelRuntimeContext`.
- Add APIs:

```cpp
class PreparedWeightStore {
public:
    PreparedWeightRef prepareGemm(const WeightBinding& binding);
    PreparedWeightRef registerTransferredGemm(const WeightBinding& binding,
                                              std::unique_ptr<ITensorGemm> kernel);
    PreparedWeightRef prepareEmbedding(const WeightBinding& binding,
                                       int d_model,
                                       size_t vocab_offset,
                                       size_t total_vocab);
    ITensorGemm* gemmKernel(const PreparedWeightRef& ref) const;
    const PreparedEmbeddingHandle* embedding(const PreparedWeightRef& ref) const;
    bool contains(const PreparedWeightRef& ref) const;
};
```

- Historical migration note: this store was initially allowed to wrap existing
  `KernelFactory` preparation/transfer APIs. The lazy global prepared-GEMM
  factory fallback has since been removed; model GEMM preparation now flows
  through store-owned refs or explicit pipeline registration.
- Store full identity metadata next to the current prepared handle.
- Add a check that every prepared handle references the expected binding identity.

**Validation gate**:

- Existing `KernelFactory` behavior remains intact.
- New tests can query prepared state through `PreparedWeightStore` without accessing `KernelFactory` directly.

---

### Phase 5: Move GPU Pipeline Registration into PreparedWeightStore

**Goal**: Make the GPU-side repack pipeline produce model-owned prepared refs instead of static factory entries.

**Tasks**:

- Refactor `WeightManager::packGemmWeightsViaPipeline()` to accept or access `PreparedWeightStore`.
- Have `LoadOrchestrator`/`WeightVRAMPool` lifetime be owned by prepared GEMM entries in `PreparedWeightStore`.
- Preserve the current behavior where `ROCmQuantisedGemmKernel`/`CUDAQuantisedGemmKernel` hold the orchestrator/pool lifetime owner.
- Register prepared refs on `WeightBinding` objects.
- Stop using `tensor->in_prepared_gemm_registry_` as the primary truth for model-owned GPU readiness. Keep it temporarily as compatibility state.
- Add explicit `WeightBinding::prepared` state transitions.

**Validation gate**:

- GPU pipeline load works for SingleDevice and LocalTP.
- Host release uses prepared refs rather than only `TensorBase::isInPreparedGemmRegistry()`.
- GPU pipeline failure reports weight identity, device, planned format, and binding id.

---

### Phase 6: Graph Binding API Migration

**Goal**: Graph builders and stages consume explicit bindings instead of raw weight resolution callbacks.

**Tasks**:

- Add `ModelWeightBindings` as the graph-facing API.
- Extend or replace `ModelWeights`:

```cpp
struct LayerWeightBindings {
    const WeightBinding* wq = nullptr;
    const WeightBinding* wk = nullptr;
    const WeightBinding* wv = nullptr;
    const WeightBinding* wo = nullptr;
    // ... all existing LayerWeights fields as bindings
};

struct ModelWeightBindings {
    const WeightBinding* embedding_table = nullptr;
    const WeightBinding* final_norm = nullptr;
    const WeightBinding* lm_head = nullptr;
    std::function<LayerWeightBindings(int)> get_layer_weights;
};
```

- Add temporary adapters from `WeightBinding` to `TensorBase*` for existing stages.
- Update graph builders to use `FrozenModelWeightSet` lookups.
- Add graph-build validation that no weight resolution occurs after freeze.
- Add stage dump/debug output fields for weight binding id and canonical name.

**Validation gate**:

- Full graph and partial PP graph build without calling `getWeightForDevice()`.
- All weight-bearing stages expose binding metadata in `StageDumpInfo` or equivalent debug info.

---

### Phase 7: Stage Execution Uses PreparedWeightRef

**Goal**: GEMM and embedding stages resolve kernels from prepared refs, not from tensor lookups through `KernelFactory`.

**Tasks**:

- Update GEMM-bearing stages:
  - `GEMMStage`
  - `FusedQKVGEMMStage`
  - `FusedGateUpGEMMStage`
  - `LMHeadStage`
  - MoE expert/shared expert stages as applicable
- Stage params should carry `WeightBindingRef` or `PreparedWeightRef` for GEMM weights.
- Replace stage execution calls to the old lazy global prepared-GEMM factory fallback with `PreparedWeightStore::gemmKernel(ref)`.
- For CPU fallback, prepared refs can resolve to CPU packed or floating-point kernels.
- For non-GEMM kernels, keep `KernelRegistry` device/type selection.

**Validation gate**:

- GPU stage execution produces no prepared-GEMM cache misses.
- Graph replay does not call any weight-preparation API.
- Transfer tracing confirms no unexpected D2H for weights during decode.

---

### Phase 8: Slim KernelFactory into KernelRegistry

**Goal**: Remove model-weight lifetime ownership from `KernelFactory`.

**Tasks**:

- Rename or introduce `KernelRegistry` for stateless/device-scoped kernel selection.
- Move prepared GEMM registry to `PreparedWeightStore`.
- Move prepared embedding registry to `PreparedWeightStore`.
- Move fused weight caches that key on prepared handles into `PreparedWeightStore` or stage-local caches.
- Keep device-scoped non-weight kernels in a separate `DeviceKernelCache` or the existing dedicated cache.
- Remove `TensorBase::~TensorBase()` dependency on `KernelFactory::clearCacheFor()` for model weights.
- Keep a small compatibility shim temporarily for tests and direct kernel calls.

**Validation gate**:

- Destroying temporary tensors does not scan global prepared-weight registries.
- Destroying `ModelContext` releases all prepared weights and GPU pools.
- KernelFactory/KernelRegistry no longer knows about `ModelContextId` except for diagnostics or compatibility assertions.

---

### Phase 9: Host Release and Freeze Semantics Cleanup

**Goal**: Replace timing-sensitive release comments with typed completion gates.

**Tasks**:

- Define lifecycle states:

```cpp
enum class WeightLifecycleState {
    Planned,
    SourceLoaded,
    DerivedMaterialized,
    DevicePrepared,
    GraphMaterialized,
    Frozen,
    HostReleased,
};
```

- Add explicit completion gates:
  - `materialization_complete`
  - `device_preparation_complete`
  - `graph_materialization_complete`
  - `host_release_allowed`
- Make `releaseAllHostWeightData()` operate on bindings and host policies.
- Remove ad hoc deferred host release flags where possible.
- For lazy graph build paths, either force graph materialization before host release or mark exact bindings as host-required until lazy materialization completes.

**Validation gate**:

- LocalTP and Hybrid PP+TP release host data deterministically after the same lifecycle state, not after first forward by convention.
- MoE expert packing cannot read released host data because its bindings declare host requirements.

---

### Phase 10: Remove Compatibility APIs

**Implementation note (2026-05-06)**: The `KernelFactory::getOrCreatePreparedGemmWeights()` compatibility path has been removed by `KERNELFACTORY_PREPARED_GEMM_FALLBACK_REMOVAL_PLAN.md`. Graph-built model GEMM stages now carry `PreparedWeightRef` metadata and resolve through `PreparedWeightStore`; stage/workspace lazy preparation through the deleted factory method is forbidden by the static guard `V2_Unit_Static_NoNewStagePreparedGemmFactoryFallbacks`.

**Goal**: Delete old paths after all production modes use frozen bindings and prepared stores.

**Tasks**:

- Remove graph-build callbacks that call `getWeightForDevice()`.
- Remove direct stage calls to prepared GEMM creation APIs.
- Remove `TensorBase::in_prepared_gemm_registry_` or reduce it to a debug-only compatibility marker.
- Remove prepared-weight static registries from `KernelFactory`.
- Remove fallback post-host-release raw pointer key aliases if no longer needed.
- Update docs and new-model guide to require weight schema roles and binding generation.

**Validation gate**:

- All V2 unit tests pass.
- All integration/parity tests pass for at least SingleDevice, LocalTP, LocalPP, and a MoE model.
- No production code calls the old lazy global prepared-GEMM factory fallback directly.

---

## Mode-Specific Acceptance Criteria

### SingleDevice

- One `WeightPlan` for one target device.
- All GEMM weights have prepared refs before graph execution.
- Host release happens after graph materialization unless CPU execution requires host data.
- Tied embeddings produce two bindings with an alias relationship, not two unrelated tensors.

### LocalTP

- Each TP device has explicit bindings for its slice/replica of every required weight.
- Proportional assignments are visible in the binding audit.
- Embedding and LM head vocab shards include `vocab_offset` and `total_vocab` metadata.
- No per-device graph builder can accidentally bind another device's tensor.

### NodeLocalTP / GlobalTP

- Each rank/domain binding includes `tp_domain`, `rank_in_domain`, `domain_size`, and shard range.
- Node-local is represented as a TP scope on the binding, not inferred from hostnames later.
- Collective expectations for row/input-parallel outputs are included in the plan.

### LocalPP

- Stage-local layer ranges are represented in the plan.
- Embedding ownership and LM head ownership are represented as global binding ownership.
- Shared `WeightManager` reconfiguration is replaced by one model-wide PP plan.
- Host release waits until all PP stage bindings that require host data are graph-materialized or prepared.

### Hybrid PP+TP

- Each PP stage contains a nested TP domain in the plan.
- Each nested TP device gets stage-local bindings with both PP stage id and TP device index.
- Outer PP owner controls host release across nested TP domains.
- The final audit can answer: "layer L weight W for stage S and TP device D came from source X, slice Y, prepared handle Z".

### CPU Decode Participation

- Decode-specific shards are represented as `WeightDerivationKind::DecodeShard` bindings.
- Phase-aware weight access is moved from dynamic graph resolution to planned pre-materialized bindings.
- Decode graph uses explicit decode bindings, not runtime calls to `getDecodeWeight()`.

### MoE and Shared Experts

- 3D expert tensors record expert ranges and GGUF dimension conventions.
- Shared expert tensors use distinct roles from routed expert tensors.
- Dynamic expert replicas are future `RebalancedExpertReplica` bindings with source identity preserved.

---

## Testing Strategy

### Unit Tests

Add or extend tests for:

- `WeightIdentity` generation and stable ids.
- Metadata propagation through source load, row slice, column slice, expert slice, clone, alias, and decode shard.
- `WeightPlan` generation for SingleDevice, LocalTP, LocalPP, Hybrid PP+TP, and GlobalTP.
- `FrozenModelWeightSet` lookup and validation.
- `PreparedWeightStore` prepare/register/lookup/cleanup behavior.
- Host release policy transitions.
- KernelRegistry selection without prepared-weight ownership.

### Integration Tests

Run targeted tests after each implementation phase:

```bash
cmake --build build_v2_integration --parallel
ctest --test-dir build_v2_integration -R "^V2_Unit_" --output-on-failure --parallel
ctest --test-dir build_v2_integration -R "^V2_Integration_" --output-on-failure --parallel
```

For GPU-sensitive phases, add targeted CUDA/ROCm runs when hardware is available:

```bash
ctest --test-dir build_v2_integration -R "V2_Integration_Parity_.*CUDA|V2_Integration_Parity_.*ROCm" --output-on-failure
```

For performance-sensitive phases, use Release performance tests:

```bash
cmake --build build_v2_release --parallel
ctest --test-dir build_v2_release -R "^V2_Perf_" --verbose
```

### Diagnostic Gates

- Enable `LLAMINAR_WEIGHT_LIFECYCLE_TRACE=1` on one small model and confirm every graph-bound weight has a complete trace.
- Enable transfer tracing for decode and confirm no unexpected large weight D2H transfers.
- Confirm `KernelFactory::preparedGemmRegistrySize()` stops growing during graph replay in compatibility phases, then disappears as a production dependency after migration.
- Confirm model teardown releases prepared GPU pools exactly once.

---

## Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| Host data released before lazy MoE graph materialization | Crash or garbage output | Explicit host policies and graph-materialization gate before release. |
| LocalPP shared `WeightManager` cache invalidation regression | Missing tensors for later stages | Preserve configure-without-cache-clear behavior until model-wide plans replace stage reconfiguration. |
| Prepared store duplicates GPU packed weights | VRAM regression | Binding id and source identity deduplication; audit prepared bytes by device. |
| Tied embeddings create alias confusion | LM head misses prepared GEMM or embedding prep | Represent tied aliases explicitly with separate binding ids sharing source identity. |
| Static KernelFactory compatibility hides migration gaps | Old lazy paths remain active | Add counters and assertions for direct prepared-GEMM factory calls during graph execution. |
| NodeLocalTP and GlobalTP semantics diverge | Wrong sharding or collectives | Include domain/rank metadata in `WeightPlan` and verify against `RankExecutionPlan`. |
| MoE expert views lose 3D dimension ordering | Wrong expert GEMM shapes | Identity/slice tests must cover GGUF 3D ordering and expert view extraction. |
| Debug metadata bloats hot paths | Performance regression | Store metadata in sidecar registries and compile expensive validation behind Debug/Integration flags. |

---

## Implementation Notes

### Keep `WeightManager::configure()` cache-preserving until replaced

`WeightManager::configure(const WeightManagerConfig&)` intentionally updates sharding and dimension state without clearing loaded tensors. LocalPP and nested TP-in-PP rely on this because one shared `WeightManager` is used across stage construction. Do not reintroduce cache clearing in `configure()` as part of this migration.

### Treat `KernelFactory` as a compatibility boundary first

Do not try to remove all static prepared registries in the first phases. First wrap them with `PreparedWeightStore`; then move ownership once the graph-facing API no longer depends on raw tensor lookups.

### Avoid a flag day for stages

Introduce `WeightBinding` adapters that expose `TensorBase*` to existing stage params. Convert stages to `PreparedWeightRef` gradually, starting with plain `GEMMStage` and `LMHeadStage`, then fused QKV/gate-up, then MoE.

### Prefer one audit artifact over scattered logs

The trace facility should write a structured report to logs and optionally to disk, for example:

```text
Weight Binding Audit: model=42 mode=HybridPPTP
binding  name                         role              derivation    pp tp device  prepared       host
1        token_embd.weight             Embedding         Source        0  0 rocm:0  Embedding      releasable
2        token_embd.weight/output      LMHead            TiedAlias     1  0 rocm:1  RocmGemm       releasable
3        blk.0.attn_q.weight           AttentionQ        RowSlice      0  0 rocm:0  RocmGemm       released
4        blk.0.attn_q.weight           AttentionQ        RowSlice      0  1 rocm:1  RocmGemm       released
```

---

## Suggested Milestones

| Milestone | Phases | Outcome |
|-----------|--------|---------|
| M1: Observability | 0-1 | Every weight has traceable identity metadata; no ownership changes. |
| M2: Planning | 2-3 | All modes can produce and validate frozen weight bindings. |
| M3: Prepared Store Shim | 4-5 | Prepared weights are queryable through model-owned API while KernelFactory remains compatibility backend. |
| M4: Graph Binding Migration | 6-7 | Graph build and replay no longer perform weight resolution or preparation. |
| M5: KernelFactory Slimming | 8-10 | KernelFactory is reduced to kernel selection/device helpers; model-weight lifetime is model-owned. |

---

## Final Target Contract

At the end of this project, the model-weight contract should be simple:

1. `OrchestrationRunner` builds an immutable execution strategy.
2. `WeightManager` materializes a frozen weight set from that strategy.
3. `PreparedWeightStore` prepares all device-specific payloads and owns them for the model context lifetime.
4. `DeviceGraphOrchestrator` receives a frozen weight set and builds graphs from it.
5. Stages execute against prepared refs and immutable tensor bindings.
6. Graph replay never creates, slices, clones, uploads, repacks, registers, or releases model weights.

That final contract is the core cleanup: there is one load/materialize/prepare/freeze boundary, and everything after it is execution.
