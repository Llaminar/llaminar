# InferenceRunner & InferenceRunnerFactory Refactor Plan

## Current State

### File sizes
| File | Lines | Role |
|------|-------|------|
| `InferenceRunnerFactory.cpp` | 2008 | God-factory: GraphConfig building, TP assignment, weight loading, orchestrator wiring, collective setup, PP stage config |
| `OrchestrationRunner.cpp` | 1670 | God-class: lifecycle, MPI coordination, sampling, PP communication, TP setup, weight loading, plan building |
| `IOrchestrationRunner.h` | ~380 | Interface (bloated: 30+ virtual methods across 8 concerns) |
| `IInferenceRunner.h` | ~450 | Low-level device interface (also bloated) |

### Key Problems

1. **`InferenceRunnerFactory.cpp` is a 2000-line monolith** with 6+ static helper functions that each handle unrelated concerns (TP assignment, weight upload, collective setup, graph config building). It's untestable in isolation.

2. **`OrchestrationRunner` mixes too many concerns**: lifecycle management, MPI broadcast coordination, CPU sampling with penalty tracking, PP communication, cluster inventory gathering, model metadata loading, TP/PP context creation, and compute graph building. The `initialize()` method alone has 7 sequential steps.

3. **`IOrchestrationRunner` has ~30 virtual methods** spanning snapshots, profiling, GPU-side sampling, MPI worker loops, timeline suppression, and a growing list of setters. Most callers use only 3-5 methods.  Most of the advanced methods have default no-op implementations in the interface, suggesting they're optional capabilities, not core API.

4. **`InferenceRunnerConfig` is a grab-bag struct** (15+ fields) that conflates runtime config (max_seq_len, batch_size), precision config (activation, KV cache scales), parallelism config (tp_ctx, pp_stage_config), and memory config (use_mapped_memory, use_bar_backed_hidden).

5. **Duplicated weight-loading logic**: `configureOrchestratorWeightsImpl`, `configurePPStageWeightsImpl`, and `configureUnifiedPipelineWeightsImpl` share 60%+ structure (eager-load, validate schema, pack, upload, set orchestrator weights) but are separate 100+ line functions.

6. **`createDeviceGraphOrchestratorImpl()` is ~300 lines** that sequentially: build GraphConfig, apply TP assignment, create TurboQuant/rotation contexts, create GlobalTPContext, create orchestrator, init graph cache, init inference state, configure weights, create weight streamer, build collective context.

---

## Proposed Refactoring

### Phase 1: Extract Weight Preparation Service

**Goal**: Eliminate the 3 duplicated `configure*WeightsImpl` functions.

**New interface**: `IWeightPreparationService`
```
src/v2/execution/factory/IWeightPreparationService.h
src/v2/execution/factory/WeightPreparationService.h
src/v2/execution/factory/WeightPreparationService.cpp
```

```cpp
class IWeightPreparationService {
public:
    virtual ~IWeightPreparationService() = default;

    struct WeightLoadSpec {
        std::string architecture;
        DeviceId target_device;
        int first_layer = 0;        // For PP: subset of layers
        int last_layer = -1;        // -1 = all layers
        bool has_embedding = true;
        bool has_lm_head = true;
    };

    /// Validate, load, pack, upload weights. Returns configured ModelWeights.
    virtual std::optional<ModelWeights> prepareWeights(
        const WeightLoadSpec& spec,
        std::shared_ptr<IModelContext> model_ctx) = 0;
};
```

**Impact**: Replaces `configureOrchestratorWeightsImpl`, `configurePPStageWeightsImpl`, and `configureUnifiedPipelineWeightsImpl` with a single polymorphic call. Unit tests can inject a mock that returns pre-built `ModelWeights`.

**Tests**:
```
tests/v2/unit/execution/factory/Test__WeightPreparationService.cpp
```
- Test schema validation with missing required weights
- Test optional weight skipping
- Test PP partial weight loading (layer range filtering)
- Test parallel eager loading (mock weight manager)

---

### Phase 2: Extract TP Assignment Strategy

**Goal**: Replace 4 static `apply*TPAssignment` functions with a strategy interface.

**New interface**: `ITPAssignmentStrategy`
```
src/v2/execution/factory/ITPAssignmentStrategy.h
src/v2/execution/factory/TPAssignmentStrategy.cpp
```

```cpp
class ITPAssignmentStrategy {
public:
    virtual ~ITPAssignmentStrategy() = default;

    /// Apply TP dimensions (head_start, local_n_heads, d_ff_local, vocab_local)
    /// to the given GraphConfig.
    virtual bool apply(GraphConfig& graph_config) const = 0;
};

/// Factory: selects the right strategy based on context
std::unique_ptr<ITPAssignmentStrategy> createTPAssignment(
    ITPContext* tp_ctx,
    int tp_device_idx,
    const TensorParallelConfig* tp_config,
    const std::shared_ptr<IMPIContext>& mpi_ctx,
    bool weights_sharded);
```

Concrete implementations:
- `NoTPAssignment` — `setFullDimensions()`, single device
- `LocalTPAssignment` — proportional split via `ILocalTPContext`
- `ProportionalGlobalTPAssignment` — heterogeneous multi-rank via `TensorParallelConfig`
- `EqualSplitGlobalTPAssignment` — homogeneous 1/world_size via MPI

**Impact**: Each strategy is independently testable with a mock `GraphConfig`. The factory function replaces the current `applyTPAssignment()` static dispatcher.

**Tests**:
```
tests/v2/unit/execution/factory/Test__TPAssignmentStrategy.cpp
```
- Test each strategy with known inputs → expected GraphConfig fields
- Test factory selection logic

---

### Phase 3: Extract Graph Config Builder Pipeline

**Goal**: Replace the inline GraphConfig population in `createDeviceGraphOrchestratorImpl()` with a composable pipeline.

**New class**: `GraphConfigPipeline`
```
src/v2/execution/factory/GraphConfigPipeline.h
src/v2/execution/factory/GraphConfigPipeline.cpp
```

```cpp
class GraphConfigPipeline {
public:
    struct Context {
        std::shared_ptr<IModelContext> model_ctx;
        DeviceId device;
        InferenceRunnerConfig runner_config;
        ITPContext* tp_ctx = nullptr;
        int tp_device_idx = 0;
        std::shared_ptr<IMPIContext> mpi_ctx;
    };

    struct Result {
        GraphConfig graph_config;
        std::shared_ptr<TurboQuantContext> turboquant_ctx;   // ownership
        std::shared_ptr<ActivationRotation> kv_rotation;     // ownership
        std::shared_ptr<IGlobalTPContext> global_tp_ctx;      // ownership
    };

    /// Build a fully-configured GraphConfig + owned auxiliary contexts
    static std::optional<Result> build(const Context& ctx);
};
```

This replaces ~150 lines of inline code in `createDeviceGraphOrchestratorImpl` with a single call. The auxiliary context objects (TurboQuant, rotation, GlobalTP) are co-created and returned for ownership transfer to the orchestrator.

**Impact**: `createDeviceGraphOrchestratorImpl` shrinks from ~300 lines to ~50.

**Tests**:
```
tests/v2/unit/execution/factory/Test__GraphConfigPipeline.cpp
```
- Test TurboQuant context creation for TQ4/TQ KV cache
- Test KV rotation creation for Q16_1
- Test GlobalTPContext creation for multi-rank
- Test full pipeline with mock IModelContext

---

### Phase 4: Split `IOrchestrationRunner` Into Focused Interfaces

**Goal**: Apply Interface Segregation Principle to the 30-method interface.

Split into:

```cpp
// Core inference — what 90% of callers need
class IInferenceEngine {
public:
    virtual ~IInferenceEngine() = default;
    virtual bool initialize() = 0;
    virtual void shutdown() = 0;
    virtual bool prefill(const std::vector<int32_t>& tokens) = 0;
    virtual GenerationResult decodeStep() = 0;
    virtual GenerationResult generate(const std::vector<int32_t>& prompt_tokens,
                                       int max_new_tokens, const SamplingParams& sampling) = 0;
    virtual void clearCache() = 0;
    virtual int vocabSize() const = 0;
    virtual int currentPosition() const = 0;
    virtual bool isInitialized() const = 0;
    virtual const std::string& lastError() const = 0;
    virtual std::shared_ptr<ITokenizer> tokenizer() const = 0;
    virtual const float* lastLogits() const = 0;
    virtual void setStopTokens(const std::vector<int32_t>& stop_tokens) = 0;
    virtual void setSamplingParams(const SamplingParams& params) = 0;
    virtual SamplingParams getRecommendedSamplingParams() const = 0;
};

// Snapshot capability (parity tests)
class ISnapshotCapable {
public:
    virtual ~ISnapshotCapable() = default;
    virtual void enableSnapshotCapture(const std::string& output_dir = "") = 0;
    virtual void disableSnapshotCapture() = 0;
    virtual void clearSnapshots() = 0;
    virtual const float* getSnapshot(const std::string& key, size_t& out_size) const = 0;
    virtual std::vector<std::string> getSnapshotKeys() const = 0;
};

// GPU-side sampling optimization
class IGPUSamplingCapable {
public:
    virtual ~IGPUSamplingCapable() = default;
    virtual int sampleGreedyOnDevice() = 0;
    virtual int sampleOnDevice(const SamplingParams& params) = 0;
    virtual void setSkipLogitsGatherDecode(bool skip) = 0;
    virtual void setSkipLogitsGatherPrefill(bool skip) = 0;
};

// Profiling capability
class IProfilable {
public:
    virtual ~IProfilable() = default;
    virtual const GraphExecutorStats* executorStats() const = 0;
    virtual void resetExecutorStats() = 0;
    virtual void setSuppressTimeline(bool suppress) = 0;
    virtual void setAccumulatePrefill(bool accumulate) = 0;
    virtual void flushStageTimeline() = 0;
};

// MPI coordination (server mode only)
class IMPICoordinatable {
public:
    virtual ~IMPICoordinatable() = default;
    virtual void runMPIWorkerLoop() = 0;
    virtual void shutdownMPIWorkers() = 0;
    virtual void setMPICoordinatedMode(bool enabled) = 0;
};

// Orchestration metadata (plan inspection)
class IOrchestrationInspectable {
public:
    virtual ~IOrchestrationInspectable() = default;
    virtual const RankExecutionPlan& executionPlan() const = 0;
    virtual const OrchestrationConfig& config() const = 0;
};
```

**`IOrchestrationRunner` becomes a composite**:
```cpp
class IOrchestrationRunner
    : public IInferenceEngine
    , public ISnapshotCapable
    , public IGPUSamplingCapable
    , public IProfilable
    , public IMPICoordinatable
    , public IOrchestrationInspectable
{
    // Inherits all ~30 methods via composition of focused interfaces
};
```

**Impact**: 
- `ChatCompletionHandler` depends only on `IInferenceEngine` (not the full 30-method interface)
- `BenchmarkRunner` depends on `IInferenceEngine` + `IProfilable` + `IGPUSamplingCapable`
- Parity tests depend on `IInferenceEngine` + `ISnapshotCapable`
- Server mode depends on `IInferenceEngine` + `IMPICoordinatable`
- Mocking becomes granular: `MockInferenceEngine` has 12 methods, not 30+

**Migration**: `IOrchestrationRunner` still exists as a composite, so no existing code breaks. New code can depend on the narrower interfaces.

---

### Phase 5: Extract MPI Coordinator

**Goal**: Extract the MPI broadcast coordination logic from `OrchestrationRunner` into a standalone class.

```
src/v2/execution/runner/IMPICoordinator.h
src/v2/execution/runner/MPICoordinator.h
src/v2/execution/runner/MPICoordinator.cpp
```

```cpp
class IMPICoordinator {
public:
    virtual ~IMPICoordinator() = default;

    /// Broadcast a prefill command + tokens to all worker ranks
    virtual void coordinatePrefill(const std::vector<int32_t>& tokens) = 0;

    /// Broadcast a decode step command
    virtual void coordinateDecodeStep() = 0;

    /// Broadcast clear cache command
    virtual void coordinateClearCache() = 0;

    /// Broadcast sampling params
    virtual void coordinateSetSampling(const SamplingParams& params) = 0;

    /// Broadcast skip-logits-gather for decode
    virtual void coordinateSkipLogitsDecode(bool skip) = 0;

    /// Signal workers to shut down
    virtual void shutdownWorkers() = 0;

    /// Run worker loop (non-root ranks)
    virtual void runWorkerLoop(IInferenceEngine& engine) = 0;

    /// Check if coordination is active
    virtual bool isActive() const = 0;
};
```

**Impact**: `OrchestrationRunner::prefill()`, `decodeStep()`, `clearCache()`, `setSamplingParams()`, `setSkipLogitsGatherDecode()` each lose their 3-line MPI broadcast boilerplate. The worker loop (`runMPIWorkerLoop`) moves entirely into `MPICoordinator`.

**Tests**:
```
tests/v2/unit/execution/runner/Test__MPICoordinator.cpp
```
- Test command serialization/deserialization with mock MPI
- Test worker loop dispatch
- Test no-op when inactive (single rank)

---

### Phase 6: Simplify `OrchestrationRunner::initialize()`

After Phases 1-3, `initialize()` becomes:

```cpp
bool OrchestrationRunner::initialize() {
    if (initialized_) return true;

    if (!initializeMPI()) return false;
    if (!buildExecutionPlan()) return false;
    if (!setupLocalTPContext()) return false;
    if (!setupLocalPPContext()) return false;
    if (!loadWeights()) return false;
    if (!validateTPPPConfiguration()) return false;
    if (!validateContextLength()) return false;
    if (!buildComputeGraph()) return false;

    initialized_ = true;
    cacheRecommendedSampling();
    return true;
}
```

The MPI sync-init-step pattern can be extracted into a helper:
```cpp
bool OrchestrationRunner::syncStep(bool ok, const char* name) {
    if (!mpi_ctx_ || mpi_ctx_->world_size() <= 1) return ok;
    int val = ok ? 1 : 0, global = 0;
    MPI_Allreduce(&val, &global, 1, MPI_INT, MPI_MIN, mpi_ctx_->communicator());
    if (!global && ok) setError(std::string("Failed on another rank: ") + name);
    return global != 0;
}
```

---

### Phase 7: Collapse Factory Free Functions

**Goal**: Replace the 7 free functions in `InferenceRunnerFactory.h` with a single factory class.

```cpp
class DeviceRunnerFactory {
public:
    explicit DeviceRunnerFactory(
        std::shared_ptr<IWeightPreparationService> weight_service = nullptr);

    /// Create single-device runner (standard path)
    std::unique_ptr<IInferenceRunner> createSingleDevice(
        std::shared_ptr<IModelContext> model_ctx,
        DeviceId device,
        const InferenceRunnerConfig& config);

    /// Create PP stage runner
    std::unique_ptr<IInferenceRunner> createPPStage(
        std::shared_ptr<IModelContext> model_ctx,
        DeviceId device,
        const FactoryPPStageConfig& pp_config,
        const InferenceRunnerConfig& config);

    /// Create multi-device LOCAL TP runner
    std::unique_ptr<IRankOrchestrator> createMultiDevice(
        std::shared_ptr<IModelContext> model_ctx,
        std::unique_ptr<ILocalTPContext> tp_ctx,
        const RankOrchestrator::Config& config);

    /// Create unified PP pipeline runner
    std::unique_ptr<IInferenceRunner> createPipeline(
        std::shared_ptr<ModelContext> model_ctx,
        std::shared_ptr<PipelineConfig> pipeline_config,
        const InferenceRunnerConfig& config);
};
```

**Impact**: 
- `IWeightPreparationService` is injected → testable without real weights
- Single class with clear ownership (vs scattered free functions)
- Deprecated free functions can forward to `DeviceRunnerFactory` during migration

---

## Summary: Before vs After

### Before
```
InferenceRunnerFactory.cpp   2008 lines  (monolithic factory with 10+ static helpers)
OrchestrationRunner.cpp      1670 lines  (god-class: lifecycle + MPI + sampling + PP + TP)
IOrchestrationRunner.h        380 lines  (30+ virtual methods, 8 concerns)
```

### After
```
# New focused components
IWeightPreparationService.h    ~30 lines  (weight load/pack/upload interface)
WeightPreparationService.cpp  ~200 lines  (unified weight pipeline, replaces 3 functions)
ITPAssignmentStrategy.h        ~25 lines  (TP dimension assignment interface)
TPAssignmentStrategy.cpp      ~150 lines  (4 concrete strategies)
GraphConfigPipeline.cpp       ~120 lines  (config + auxiliary context builder)
IMPICoordinator.h              ~30 lines  (MPI broadcast coordination interface)
MPICoordinator.cpp            ~120 lines  (extracted from OrchestrationRunner)
DeviceRunnerFactory.cpp       ~200 lines  (replaces free functions)

# Slimmed originals
InferenceRunnerFactory.cpp   ~600 lines  (delegates to extracted services)
OrchestrationRunner.cpp     ~1000 lines  (lifecycle + inference; MPI/sampling delegated)
IOrchestrationRunner.h       ~380 lines  (composite of 6 focused interfaces, backward compatible)

# New interfaces enable granular mocking
IInferenceEngine.h             ~20 lines  (core 12-method interface for 90% of callers)
ISnapshotCapable.h             ~10 lines
IGPUSamplingCapable.h          ~10 lines
IProfilable.h                  ~10 lines
IMPICoordinatable.h            ~10 lines
IOrchestrationInspectable.h    ~10 lines
```

### Execution Order

| Phase | Effort | Risk | Reward |
|-------|--------|------|--------|
| 1. Weight Preparation Service | Medium | Low | Eliminates 3-way duplication, enables mock weight loading |
| 2. TP Assignment Strategy | Low | Low | Clean testable strategies, eliminates 4 static functions |
| 3. Graph Config Pipeline | Medium | Low | Shrinks factory by ~150 lines |
| 4. Interface Segregation | Low | **None** | Backward-compatible narrower interfaces for mocking |
| 5. MPI Coordinator | Medium | Low | Extracts MPI from OrchestrationRunner |
| 6. Simplify initialize() | Low | Low | Cleanup, depends on 1-3 |
| 7. Collapse Factory Functions | Medium | Low | Single injectable factory class |

**Phase 4 is the quick win** — it's purely additive (new interfaces, existing composite inherits them), zero risk, and immediately improves testability for new code.

**Phases 1-2 provide the most test coverage improvement** — weight loading and TP assignment are currently only testable via full integration tests.

---

## Migration Strategy

All phases are backward-compatible:
- `IOrchestrationRunner` continues to exist as a composite interface
- Free functions in `InferenceRunnerFactory.h` can forward to `DeviceRunnerFactory`
- Existing tests continue to work via `MockOrchestrationRunner`
- New tests can mock narrower interfaces (`MockInferenceEngine`, `MockWeightPreparationService`)

No existing caller needs to change. New callers should prefer the narrow interfaces.
