# DeviceGraphOrchestrator Refactor Project Plan

**Date**: 2026-03-31  
**Status**: ✅ COMPLETE  
**Scope**: `DeviceGraphOrchestrator.h` (2,400 lines) + `.cpp` (3,956 lines) = 6,356 lines  
**Goal**: Decompose God Class into testable, composable units. Target: DGO core ≤ 2,000 lines.

**Final Result**: DGO reduced to ~5,050 lines (1,930 .h + 3,122 .cpp). Phases 0-3 fully
implemented, Phase 4 superseded (code already extracted in Phase 3), Phase 5 consolidated
config-time setters into expanded Dependencies struct.

---

## Problem Summary

The DeviceGraphOrchestrator is a 6,356-line monolith with:

- **8 monster methods** (3 over 500 lines, 5 over 100 lines)
- **~35 member variables** spanning 5 categories (core, caches, state, snapshots, config)
- **12+ setter injection methods** called in specific order with no completeness validation
- **Duplicated code** in 6 distinct locations (constructors, cache paths, allocation, buffer mapping)
- **350 lines of inline snapshot logic** in the header with hardcoded string matching
- **Tight coupling** to `Qwen2Graph` concrete type, quantization block types, stage naming conventions

### Consumers

| Consumer | Relationship |
|----------|-------------|
| `InferenceRunnerFactory` | Creates DGO via constructor + 10+ setter calls |
| `MultiDeviceOrchestrator` | Owns `vector<unique_ptr<DGO>>` (1 per TP device / PP stage) |
| `MultiDomainOrchestrator` | Wraps 1 DGO |
| `PPExecutionStrategy` | Takes `vector<DGO*>&` |
| 7 unit test files + 4 integration test files | Direct construction and API calls |

### Existing Infrastructure (Already Built, Underused)

| Component | Status | Production Use |
|-----------|--------|----------------|
| `BufferArena` (`src/v2/memory/BufferArena.h`) | Complete | Only `initializeBuffers()` path (tests only) |
| `BufferAllocator` (`src/v2/execution/local_execution/graph/GraphResolver.h`) | Complete | Only `initializeBuffers()` path (tests only) |
| `GraphSchema` + `BufferSpec` | Complete | Schema-driven buffer declaration works |
| `Qwen2BufferSpec` | Complete | BAR/pinned strategy selection works |
| `StageDumpInfo` on stages | Complete | Stages already declare their outputs |

The key insight: **BufferArena is already the InferenceBufferAllocator**. The migration from `initializeInferenceState()` (671 lines, manual) to `initializeBuffers()` (193 lines, schema-driven) is the single highest-ROI change.

---

## Phase Overview

| Phase | Name | Lines Removed | Risk | Dependency | Status |
|-------|------|---------------|------|------------|--------|
| 0 | Mechanical cleanup (constructors, `toModelBuffers`) | ~120 | Trivial | None | ✅ Done |
| 1 | Migrate `initializeInferenceState` → BufferArena | ~500 | Medium | Phase 0 | ✅ Done |
| 2 | Extract `SnapshotCapture` class | ~350 | Medium | Phase 1 (buffer names) | ✅ Done |
| 3 | Extract `ForwardExecutionEngine` (cache + execute) | ~400 | High | Phase 2 | ✅ Done |
| 4 | Extract `ActivationTransferEngine` (PP handoff) | ~150 | Medium | Phase 3 | ✅ Superseded |
| 5 | Consolidate config injection | ~80 | Low | Phase 3 | ✅ Done |

**Estimated DGO reduction**: 6,356 → ~2,100 lines (orchestration glue + `IInferenceRunner` delegation).

---

## Phase 0: Mechanical Cleanup

**Risk**: Trivial — pure refactoring, no behavior change.  
**Test gate**: All existing unit + integration + parity tests pass unchanged.

### 0a: Constructor deduplication

The 3 constructors (`.cpp` L46–209) share ~30 identical lines of executor config. Extract to private method.

**Before** (repeated 3×):
```cpp
DeviceGraphOrchestrator::DeviceGraphOrchestrator(...)
{
    // ... constructor-specific init ...
    
    executor_config.enable_stage_timing = debugEnv().profiling.gpu_stage_timing;
    executor_config.enable_stage_timing_detail = debugEnv().profiling.gpu_stage_timing_detail;
    executor_config.enable_validation = debugEnv().validation.validate_buffers;
    executor_config.enable_input_validation = debugEnv().validation.validate_inputs;
    executor_config.fail_on_nan = debugEnv().validation.fail_on_nan;
    executor_config.fail_on_zero = debugEnv().validation.fail_on_zero;
    executor_config.dump_on_failure = debugEnv().validation.dump_on_failure;
    // ... ~20 more lines ...
}
```

**After**:
```cpp
DeviceGraphOrchestrator::DeviceGraphOrchestrator(...)
{
    // ... constructor-specific init ...
    configureExecutor(executor_config);
}

void DeviceGraphOrchestrator::configureExecutor(DeviceGraphExecutorConfig& config)
{
    config.enable_stage_timing = debugEnv().profiling.gpu_stage_timing;
    // ... all shared config ...
}
```

### 0b: `InferenceState::toModelBuffers()`

The `forward()` 3-param method (`.cpp` L2540–2600) has ~30 lines of mechanical `state_.X.get()` → `model_buffers.layer_buffers.X` mapping. Move into `InferenceState`.

**Before** (in `forward()`):
```cpp
ModelBuffers model_buffers;
model_buffers.layer_buffers.Q = state_.Q.get();
model_buffers.layer_buffers.K = state_.K.get();
model_buffers.layer_buffers.V = state_.V.get();
// ... 25 more lines ...
```

**After**:
```cpp
ModelBuffers model_buffers = state_.toModelBuffers();
```

The method lives on `InferenceState` in the `.h` file (struct is already there, L183–280). This is a trivial member function addition.

### 0c: Duplicated position ID construction

Position ID building appears in both `forward()` (3-param, L2530) and `executeForward()` (L602). Factor into a shared utility:

```cpp
static std::vector<int> buildPositionIds(
    int batch_size, int seq_len, const std::vector<int>& positions);
```

### Verification

```bash
cmake --build build_v2_integration --parallel
ctest --test-dir build_v2_integration -R "^V2_Unit_" --output-on-failure --parallel
ctest --test-dir build_v2_integration -R "^V2_Integration_Parity_" --output-on-failure
```

---

## Phase 1: Migrate `initializeInferenceState` to BufferArena

**Risk**: Medium — changes the production allocation path. Requires careful testing.  
**Test gate**: All parity tests pass. Benchmark shows no regression (BufferArena allocates identically).

### Problem

`initializeInferenceState()` is 671 lines of manual buffer allocation that duplicates what `BufferArena` + `BufferAllocator::resolveLayerBuffers()` already do. The schema-driven `initializeBuffers()` path (193 lines) already works but is only exercised by unit tests.

### Gap Analysis

What `initializeInferenceState()` handles that `initializeBuffers()` currently does NOT:

| Gap | Description | Fix |
|-----|-------------|-----|
| KV cache creation | `initializeInferenceState()` creates KV caches (single and per-device PP) | Keep in DGO — KV caches are not activation buffers |
| HybridQ16 buffers | `Q_rope`, `K_rope`, `V_dequant`, `K_head_scales` | Add `BufferSpec` entries to `Qwen2BufferSchema` |
| Snapshot buffers | `context_snapshot`, `attention_output_snapshot`, `attention_residual_snapshot` | Add conditional `BufferSpec` entries (guarded by `ENABLE_PIPELINE_SNAPSHOTS`) |
| TurboQuant context | Stored in `InferenceState` | Keep as separate member — not a buffer |
| Position/sequence tracking | `positions`, `sequence_lengths` vectors | Keep in `InferenceState` — not tensors |
| BAR detection | ~70 lines finding CUDA/ROCm device pair | Already in `ArenaConfig` setup for `initializeBuffers()` |

### Implementation Plan

#### 1a: Extend `Qwen2BufferSchema` with missing buffer specs

Add `BufferSpec` entries for HybridQ16 and snapshot buffers. The schema already supports conditional entries via `exec_policy` flags.

**New BufferSpecs** (in `Qwen2GraphSchema.cpp` or equivalent):
```cpp
// HybridQ16 buffers (conditional on activation_precision == HybridQ16)
{"Q_rope",    {"seq_len * local_n_heads", "head_dim"}, "fp32", BufferSemantic::ACTIVATION},
{"K_rope",    {"seq_len * local_n_kv_heads", "head_dim"}, "fp32", BufferSemantic::ACTIVATION},
{"V_dequant", {"seq_len * local_n_kv_heads", "head_dim"}, "fp32", BufferSemantic::ACTIVATION},

// Snapshot buffers (conditional on ENABLE_PIPELINE_SNAPSHOTS)
{"context_snapshot",          {"seq_len * local_n_heads", "head_dim"}, "fp32", BufferSemantic::DEBUG},
{"attention_output_snapshot", {"seq_len", "d_model"}, "fp32", BufferSemantic::DEBUG},
{"attention_residual_snapshot", {"seq_len", "d_model"}, "fp32", BufferSemantic::DEBUG},
```

#### 1b: Wire `InferenceState` from BufferArena

After `initializeBuffers()` + `arena_->allocate()`, populate an `InferenceState` by pulling tensors from the arena:

```cpp
bool DeviceGraphOrchestrator::initializeInferenceStateFromArena(
    int batch_size, int max_seq_len, DeviceId device)
{
    state_.batch_size = batch_size;
    state_.max_seq_len = max_seq_len;
    state_.device_id = device;
    
    // Pull from arena — single source of truth for shapes/dtypes/device
    state_.hidden      = arena_->getSharedTensor(BufferId::HIDDEN_STATE);
    state_.logits      = arena_->getSharedTensor(BufferId::LOGITS);
    state_.Q           = arena_->getSharedTensor(BufferId::Q);
    state_.K           = arena_->getSharedTensor(BufferId::K);
    // ... etc — mechanical but driven by arena, not manual construction
    
    // KV cache creation stays here (arena doesn't handle this)
    return initializeKVCaches(batch_size, max_seq_len, device);
}
```

**Note**: `BufferArena` currently stores `owned_tensor` as `shared_ptr<TensorBase>`. The tensors returned from `getTensor()` return raw `ITensor*`. We need `getSharedTensor()` on BufferArena to return `shared_ptr<TensorBase>` for ownership transfer to `InferenceState`.

#### 1c: Extract `initializeKVCaches()` from the monolith

The KV cache creation logic (~130 lines) stays in DGO but moves to its own method:

```cpp
bool DeviceGraphOrchestrator::initializeKVCaches(
    int batch_size, int max_seq_len, DeviceId device);
```

This handles:
- Single KV cache creation
- PP per-device KV cache loop
- TP-sharded KV cache dimensions
- KVCacheConfig population

#### 1d: Update `InferenceRunnerFactory` call sites

The 4 call sites in `InferenceRunnerFactory.cpp` (L733, L1442, L1611, L1759) currently call `initializeInferenceState()`. Update them to call the new path:

```cpp
// Before:
orchestrator->initializeInferenceState(batch_size, max_seq_len, device, init_config);

// After:
bool ok = orchestrator->initializeBuffers(max_seq_len);  // schema-driven arena allocation
ok = ok && orchestrator->initializeInferenceStateFromArena(batch_size, max_seq_len, device);
```

#### 1e: Deprecate old `initializeInferenceState()`

Mark as `[[deprecated]]` but keep for one release cycle for any external consumers. Remove in Phase 5.

### Verification

```bash
# Full rebuild
cmake --build build_v2_integration --parallel

# Unit tests (including BufferManagement tests)
ctest --test-dir build_v2_integration -R "^V2_Unit_" --output-on-failure --parallel

# Parity tests (production path was migrated)
ctest --test-dir build_v2_integration -R "^V2_Integration_Parity_" --output-on-failure

# Benchmark (verify no allocation overhead change)
cmake --build build_v2_release --parallel
./build_v2_release/llaminar2 --benchmark -m models/qwen2.5-0.5b-instruct-q4_0.gguf -n 50
```

---

## Phase 2: Extract `SnapshotCapture` Class

**Risk**: Medium — changes the snapshot callback wiring but NOT inference behavior.  
**Test gate**: All parity tests pass (they exercise snapshot capture).

### Problem

~350 lines of snapshot logic live inline in the `.h` file:

| Method | Lines | Problem |
|--------|-------|---------|
| `enableSnapshotCapture()` | ~205 (H:1601–1806) | 7 `name.find()` branches in a lambda closure |
| `extractFp32FromOutput()` | ~80 (H:1512–1592) | Dequant dispatch duplicating tensor layer logic |
| `convertStageNameToSnapshotKey()` | ~65 (H:1898–1962) | 20 hardcoded suffix→key string mappings |

All untestable because they're in a captured lambda.

### Design: Schema-Driven Snapshot Mapping

Instead of string-matching stage names at runtime, derive the mapping from `GraphSchema` metadata. Stages already declare their outputs via `StageDumpInfo`. The schema knows which stages produce which semantic outputs.

**New file**: `src/v2/snapshots/SnapshotCapture.h` / `.cpp`

```cpp
class SnapshotCapture {
public:
    /// Build mapping from GraphSchema (model-declared stage→output semantics)
    explicit SnapshotCapture(const GraphSchema& schema);
    
    /// Register a snapshot from stage execution
    /// Called by DeviceGraphExecutor's stage callback
    void captureStage(const std::string& stage_name, 
                      const StageDumpInfo& dump_info);
    
    /// Retrieve stored snapshot by semantic key (e.g., "layer3_ATTENTION_NORM")
    const StoredSnapshot* get(const std::string& key) const;
    
    /// Get all snapshots (for parity test comparison)
    const std::unordered_map<std::string, StoredSnapshot>& all() const;

private:
    /// Schema-derived mapping: stage_name → {output_index, snapshot_key}
    struct SnapshotMapping {
        int output_index;          // Which output from StageDumpInfo to capture
        std::string snapshot_key;  // Semantic key (e.g., "layer3_ATTENTION_NORM")
    };
    std::unordered_map<std::string, SnapshotMapping> stage_to_snapshot_;
    std::unordered_map<std::string, StoredSnapshot> snapshots_;
    
    /// Extract FP32 data from a StageDumpInfo output (handles dequant)
    static std::vector<float> extractFp32(
        const StageDumpInfo::OutputInfo& output);
};
```

### Key Design Decisions

1. **`GraphSchema` drives the mapping**: Each `StageSpec` in the schema can declare a `snapshot_key` and `snapshot_output_index` field. The mapping is built once at schema construction time, not discovered at runtime via string matching.

2. **`extractFp32()` delegates to tensor layer**: Instead of manually switching on `Q8_1Block`, `Q16_1Block`, etc., call `TensorBase::to_fp32_row()` or the quantized tensor's own dequant path. This eliminates the format coupling.

3. **Multi-output stages explicit**: For `FusedResidualNormStage` (2 outputs), the schema declares `snapshot_output_index: 1` to capture the norm output, not the residual. This eliminates the bug we just fixed.

4. **Unit testable**: `SnapshotCapture` takes a `GraphSchema` and `StageDumpInfo` inputs — no DGO dependency. Write direct unit tests.

### Migration

1. Create `SnapshotCapture` class with the schema-driven constructor
2. Extend `StageSpec` in `GraphSchema.h` with optional `snapshot_semantic` field
3. Populate `snapshot_semantic` in `Qwen2GraphSchema` for each stage
4. In DGO, replace `enableSnapshotCapture()` body with:
   ```cpp
   void DeviceGraphOrchestrator::enableSnapshotCapture(const std::string& output_dir) {
       snapshot_capture_ = std::make_unique<SnapshotCapture>(graph_builder_->schema());
       executor_->setStageSnapshotCallback(
           [this](const std::string& name, const StageDumpInfo& info) {
               snapshot_capture_->captureStage(name, info);
           });
   }
   ```
5. Replace `snapshots_` map access with `snapshot_capture_->get(key)` / `->all()`
6. Delete `extractFp32FromOutput()`, `convertStageNameToSnapshotKey()`, and the 205-line lambda body from `.h`

### New Tests

```
tests/v2/unit/snapshots/Test__SnapshotCapture.cpp
```

Test cases:
- Schema-driven mapping produces correct keys for all Qwen2 stages
- Multi-output stage (FusedResidualNorm) captures correct output index
- FP32 extraction works for FP32, Q8_1, Q16_1 tensor types
- Unknown stage name is ignored (no crash)

### Verification

```bash
ctest --test-dir build_v2_integration -R "^V2_Unit_.*Snapshot" --output-on-failure --parallel
ctest --test-dir build_v2_integration -R "^V2_Integration_Parity_" --output-on-failure
```

---

## Phase 3: Extract `ForwardExecutionEngine`

**Risk**: High — touches the hot path for every inference call.  
**Test gate**: All tests pass. Benchmark shows no regression.

### Problem

`executeForward()` is 730 lines with 16 responsibilities, including a cache-hit path and cache-miss path that duplicate 3 blocks of code.

### Design

Extract the forward graph caching, execution dispatch, and profiling into a separate engine:

**New file**: `src/v2/execution/local_execution/engine/ForwardExecutionEngine.h` / `.cpp`

```cpp
class ForwardExecutionEngine {
public:
    struct Config {
        DeviceGraphExecutor* executor;
        IGraphBuilder* graph_builder;
        BufferArena* arena;
        const DeviceGraphOrchestrator::PPConfig* pp_config;  // optional
        bool enable_graph_cache;
        bool enable_gpu_capture;
    };
    
    explicit ForwardExecutionEngine(Config config);
    
    /// Execute a forward pass (handles cache lookup, build, execute, profiling)
    bool execute(const ForwardInput& input, ForwardOutput& output);

private:
    /// Signature for graph caching
    ForwardGraphSignature buildSignature(const ForwardInput& input) const;
    
    /// Cache operations
    ForwardGraphCache* lookupCache(const ForwardGraphSignature& sig);
    ForwardGraphCache& populateCache(const ForwardGraphSignature& sig,
                                      std::unique_ptr<ComputeGraph> graph);
    
    /// Shared execution (runs regardless of cache hit/miss)
    bool prepareStableBuffers(const ForwardInput& input, ForwardGraphCache& cache);
    bool executeGraph(ForwardGraphCache& cache);
    bool syncLogitsBoundary(ForwardGraphCache& cache);
    void collectTimeline(ForwardGraphCache& cache);
    
    // State
    Config config_;
    std::unordered_map<ForwardGraphSignature, ForwardGraphCache, 
                       ForwardGraphSignatureHash> cache_;
};
```

### Key Design Decisions

1. **Unified execute path**: No cache-hit / cache-miss branching for shared operations. The common operations (`prepareStableBuffers`, `executeGraph`, `syncLogitsBoundary`, `collectTimeline`) are called unconditionally. Only graph construction differs.

2. **`ForwardInput` / `ForwardOutput`**: Replace the current ~15 local variables passed implicitly via DGO members with an explicit input/output struct:

```cpp
struct ForwardInput {
    const int* token_ids;
    int batch_size;
    int seq_len;
    int position_offset;
    DeviceId device;
    IKVCache* kv_cache;
    TensorBase* external_hidden_state;  // PP input (nullable)
    const std::unordered_map<DeviceId, IKVCache*>* pp_kv_caches;
};

struct ForwardOutput {
    TensorBase* logits;
    TensorBase* hidden;
};
```

3. **DGO becomes thin glue**: After extraction, DGO's `executeForward()` becomes:

```cpp
bool DeviceGraphOrchestrator::executeForward(/* same args */) {
    ForwardInput input{tokens, batch_size, seq_len, ...};
    ForwardOutput output{state_.logits.get(), state_.hidden.get()};
    return forward_engine_->execute(input, output);
}
```

### `ForwardGraphCache` moves into the engine

The `ForwardGraphCache` nested struct (currently DGO.h L2102–2210) and `ForwardGraphSignature` (L2054–2095) move to the engine's header. They have no dependency on DGO internals.

### Migration Steps

1. Create `ForwardInput` / `ForwardOutput` structs
2. Move `ForwardGraphSignature`, `ForwardGraphSignatureHash`, `ForwardGraphCache` to new header
3. Implement `ForwardExecutionEngine` with unified execute path
4. Wire DGO to own `unique_ptr<ForwardExecutionEngine>` and delegate
5. Delete old `executeForward()` body (keep as thin wrapper)

### Verification

```bash
# Full test suite
ctest --test-dir build_v2_integration --output-on-failure --parallel

# Benchmark (hot-path change)
./build_v2_release/llaminar2 --benchmark -m models/qwen2.5-0.5b-instruct-q4_0.gguf -n 128
```

---

## Phase 4: Extract `ActivationTransferEngine`

**Status**: ✅ SUPERSEDED — The PP activation transfer code was already extracted into
`ForwardExecutionEngine` during Phase 3. Remaining PP code in DGO is:
- `resolvePPCopyInfo()` (~45 lines) — implements `IForwardExecutionHost` interface, must stay
- `setHiddenState`/`getHiddenState` (~25 lines) — public API for PP stage communication
- BAR-backed hidden override (~30 lines) — initialization, wrong category for extraction
- `initializePPContexts()` (~130 lines) — topology setup, not activation transfer

Creating a separate `ActivationTransferEngine` class would be over-engineering given the
remaining code's size and tight coupling to DGO lifecycle.

**Risk**: Medium — only affects PP mode.  
**Test gate**: PP parity tests pass. LocalTP parity tests pass.

### Problem

PP activation handoff is raw `data()` → `memcpy` → `ensureOnDevice()` inlined into `executeForward()`. BAR-backed transfers use a host-bounce pattern. There's no abstraction — the DGO knows about BAR memory regions, D2H paths, and host-bounce fallbacks.

### Design

**New file**: `src/v2/execution/local_execution/transfer/ActivationTransferEngine.h` / `.cpp`

```cpp
class ActivationTransferEngine {
public:
    struct Route {
        DeviceId source;
        DeviceId target;
        bool is_bar_backed;
    };
    
    /// Transfer hidden state between PP stages
    bool transferHiddenState(TensorBase* src, TensorBase* dst, const Route& route);
    
    /// Transfer logits to final destination (with optional allgather for TP)
    bool transferLogits(TensorBase* local_logits, TensorBase* full_logits, 
                        const Route& route);

private:
    /// BAR host-bounce: src→host→dst (avoids corrupt BAR D2D copies)
    bool hostBounceTransfer(TensorBase* src, TensorBase* dst);
    
    /// Direct device transfer (same-vendor D2D or H2D)
    bool directTransfer(TensorBase* src, TensorBase* dst, DeviceId target);
};
```

### Migration

1. Identify all PP hidden state copy sites in `executeForward()` (cache-hit ~L770, cache-miss ~L1100)
2. Identify BAR bounce patterns in `initializePPContexts()` and `executeForward()`
3. Extract to `ActivationTransferEngine`
4. DGO delegates: `transfer_engine_->transferHiddenState(prev_hidden, state_.hidden.get(), route)`

---

## Phase 5: Consolidate Config Injection

**Status**: ✅ DONE — Expanded `Dependencies` struct to absorb config-time setter fields.
Factory call sites updated to populate Dependencies before construction. Lifecycle-dependent
setters (`setWeightStreamer`, `setCollectiveContext`, `setWeights`) remain as post-construction
setters since they depend on arena initialization and cluster discovery.

**Changes made**:
- Expanded `Dependencies` struct with 10 new fields (turboquant_ctx, pp_stage_config,
  pipeline_config, weight_streamer, weight_manager, weight_placement_map, tp_config,
  domain_config, cache_config)
- Updated Dependencies constructor to move all new fields
- Updated 4 factory call sites in `InferenceRunnerFactory.cpp` to use Dependencies
- Removed redundant `GraphCacheConfig` construction at 3 sites (defaults match)
- Moved `setPipelineConfig`, `setTurboQuantContext`, `setPPStageConfig` into Dependencies
  at all applicable sites

**Risk**: Low — API change only, no behavior change.  
**Test gate**: Build succeeds. All tests pass.

### Problem

12+ setter methods called in specific order by `InferenceRunnerFactory`:

```cpp
auto dgo = make_unique<DeviceGraphOrchestrator>(...);
dgo->setWeightManager(wm);
dgo->setWeightPlacementMap(wpm);
dgo->setTensorParallelConfig(tp);
dgo->setDomainConfig(domain);
dgo->setPPStageConfig(pp);
dgo->setCollectiveContext(cc);
dgo->setWeightStreamer(ws);
dgo->setTurboQuantContext(tq);
dgo->retainModelContext(mc);
// ... more ...
```

No validation that all required config was set before `forward()`.

### Design: `DGOConfig` builder

```cpp
struct DGOConfig {
    // Required
    std::shared_ptr<WeightManager> weight_manager;
    DeviceId device;
    
    // Optional (PP/TP/streaming)
    std::shared_ptr<TensorParallelConfig> tp_config;
    std::shared_ptr<MultiDomainTPConfig> domain_config;
    std::optional<FactoryPPStageConfig> pp_stage_config;
    std::shared_ptr<ICollectiveContext> collective_ctx;
    std::shared_ptr<IWeightStreamer> weight_streamer;
    std::shared_ptr<TurboQuantContext> turboquant_ctx;
    std::shared_ptr<WeightPlacementMap> weight_placement_map;
    
    /// Validate all required fields are set
    bool validate() const;
};
```

**Factory changes**:
```cpp
// Before: 10+ setter calls
auto dgo = make_unique<DeviceGraphOrchestrator>(graph_config, executor);
dgo->setWeightManager(wm);
dgo->setTensorParallelConfig(tp);
// ...

// After: single structured config
DGOConfig config;
config.weight_manager = wm;
config.tp_config = tp;
// ...
assert(config.validate());
auto dgo = make_unique<DeviceGraphOrchestrator>(graph_config, executor, config);
```

### Phase 5 also: Delete deprecated `initializeInferenceState()`

After Phase 1 migrated all 4 factory call sites to the BufferArena path, the old 671-line method can be deleted.

---

## Dependency Graph

```
Phase 0 (mechanical)
    │
    ▼
Phase 1 (BufferArena migration) ──────────────────┐
    │                                               │
    ▼                                               ▼
Phase 2 (SnapshotCapture)              Phase 4 (ActivationTransfer)
    │
    ▼
Phase 3 (ForwardExecutionEngine)
    │
    ▼
Phase 5 (Config consolidation + old code deletion)
```

Phases 2 and 4 are independent of each other and can be done in parallel.

---

## Line Count Projections

| Phase | Lines Added (new files) | Lines Removed (DGO) | Net DGO Lines | Actual |
|-------|------------------------|---------------------|---------------|--------|
| Start | — | — | 6,356 | 6,356 |
| Phase 0 | ~40 | ~120 | 6,276 | — |
| Phase 1 | ~200 (schema + arena bridge) | ~500 | 5,976 | — |
| Phase 2 | ~350 (`SnapshotCapture`) | ~350 | 5,626 | — |
| Phase 3 | ~500 (`ForwardExecutionEngine`) | ~600 | 5,026 | — |
| Phase 4 | — (superseded) | — | 5,026 | — |
| Phase 5 | ~0 (expanded Dependencies) | ~30 (removed setter calls) | **~5,000** | **5,052** |

**Note**: The original plan assumed more aggressive extraction for Phases 4-5. Phase 4 was
superseded because its target code was already extracted in Phase 3. Phase 5 focused on
consolidating the setter API rather than creating a separate DGOConfig class, as the
Dependencies struct already serves this purpose. The final DGO is 5,052 lines (1,930 .h +
3,122 .cpp), reduced from 6,356 — a 20.5% reduction.

---

## New Files Created

| File | Phase | Contents |
|------|-------|----------|
| `src/v2/snapshots/SnapshotCapture.h` | 2 | Schema-driven snapshot capture class |
| `src/v2/snapshots/SnapshotCapture.cpp` | 2 | Implementation |
| `src/v2/execution/local_execution/engine/ForwardExecutionEngine.h` | 3 | Forward graph cache + execution engine |
| `src/v2/execution/local_execution/engine/ForwardExecutionEngine.cpp` | 3 | Implementation |
| `src/v2/execution/local_execution/transfer/ActivationTransferEngine.h` | 4 | PP activation handoff |
| `src/v2/execution/local_execution/transfer/ActivationTransferEngine.cpp` | 4 | Implementation |
| `tests/v2/unit/snapshots/Test__SnapshotCapture.cpp` | 2 | Unit tests for snapshot mapping |
| `tests/v2/unit/execution/engine/Test__ForwardExecutionEngine.cpp` | 3 | Unit tests for forward engine |

---

## Testing Strategy

Each phase has its own test gate, but the meta-strategy is:

1. **Unit tests for extracted classes**: Each new class gets direct unit tests that don't depend on DGO
2. **Existing parity tests as integration gate**: All 17 Qwen2 + Qwen3 parity configs must pass after every phase
3. **Benchmark regression check**: `--benchmark` with Qwen2.5-7B Q8_0 on all devices after Phases 1 and 3
4. **No big-bang**: Each phase is a self-contained PR that can be merged independently

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| BufferArena `getSharedTensor()` ownership transfer | Add method + unit test before Phase 1b |
| `GraphSchema` changes break existing tests | Schema additions are purely additive (new `BufferSpec` entries) |
| `ForwardExecutionEngine` perf regression | Profile before/after Phase 3 with `LLAMINAR_PROFILING=1` |
| Multi-consumer coordination (MDO, PPStrategy) | Phase 3 keeps DGO's public API unchanged — engine is internal |
| Snapshot schema for custom/future models | `SnapshotCapture` takes `GraphSchema` — model-agnostic by design |
