# Orchestration Consolidation Plan

> **Goal**: Unify the diverged code paths by completing `OrchestrationRunner` and deprecating `InferenceRunnerFactory` as a public API.

## Background

The codebase has **two parallel paths** for creating inference runners:

| Path | Entry Point | Used By |
|------|-------------|---------|
| **Legacy** | `InferenceRunnerFactory::createInferenceRunner()` | Tests, parity framework |
| **Modern** | `OrchestrationRunnerFactory::createFromArgs()` | CLI, production |

This divergence causes maintenance burden and blocks Pipeline Parallelism support because the paths have different assumptions about weight loading.

## Architecture Decision

### ✅ Keep: OrchestrationRunner as the Canonical Path

`OrchestrationRunner` is designed for PP from the ground up:
- Uses `RankExecutionPlan` with `first_layer`, `last_layer`, `has_embedding`, `has_lm_head`
- Supports LOCAL TP, GLOBAL TP, and LOCAL PP
- Clean separation between configuration and execution

### ⚠️ Demote: InferenceRunnerFactory to Internal Implementation

`InferenceRunnerFactory` should become an **internal implementation detail** of `OrchestrationRunner`:
- `OrchestrationRunner` already uses it internally (line 827, 886)
- Remove from public headers
- Keep functionality for backward compatibility during transition

### ✅ Unify: ModelContext Layer Range API

One `ModelContext::create()` that handles all scenarios via optional parameters:

```cpp
// Unified factory method signature
static std::shared_ptr<ModelContext> create(
    const std::string& model_path,
    const ModelContextConfig& config = ModelContextConfig::defaults());
```

---

## Phase 1: Unified ModelContext API (Week 1)

### 1.1 Create `ModelContextConfig` struct

**File**: `src/v2/loaders/ModelContextConfig.h`

```cpp
/**
 * @brief Configuration for ModelContext creation
 * 
 * Supports all scenarios: single device, TP, PP
 */
struct ModelContextConfig {
    // MPI context (optional - nullptr for single-rank)
    std::shared_ptr<MPIContext> mpi_ctx;
    
    // Weight distribution
    WeightDistributionStrategy strategy = WeightDistributionStrategy::REPLICATED;
    WeightPrecision weight_precision = WeightPrecision::NATIVE;
    
    // Layer range (PP support)
    int first_layer = 0;       // First layer to load (inclusive)
    int last_layer = -1;       // Last layer to load (-1 = all layers)
    bool has_embedding = true; // Load embedding weights?
    bool has_lm_head = true;   // Load output norm + LM head?
    
    // Weight sharding (TP support)
    int shard_index = 0;       // Which shard to load
    int total_shards = 1;      // Total shards (1 = no sharding)
    float work_fraction = 1.0f; // For proportional TP
    
    // Advanced
    std::shared_ptr<WeightPlacementMap> placement_map;
    TensorFactory* factory = nullptr;
    
    // Factory helpers
    static ModelContextConfig defaults();
    static ModelContextConfig forPPStage(int stage_idx, int total_stages, int n_layers);
    static ModelContextConfig forTPShard(int shard_idx, int total_shards);
    static ModelContextConfig fromExecutionPlan(const RankExecutionPlan& plan);
};
```

### 1.2 Modify `ModelContext::create()` to use config

**File**: `src/v2/loaders/ModelContext.cpp`

```cpp
std::shared_ptr<ModelContext> ModelContext::create(
    const std::string& model_path,
    const ModelContextConfig& config)
{
    // Create with strategy based on config
    auto strategy = config.strategy;
    
    // Auto-select strategy if layer range specified
    if (config.first_layer != 0 || config.last_layer != -1 || 
        !config.has_embedding || !config.has_lm_head) {
        strategy = WeightDistributionStrategy::LAYER_PARTITIONED;
    }
    
    auto ctx = std::make_shared<ModelContext>(
        model_path,
        config.mpi_ctx,
        config.placement_map,
        config.factory,
        strategy);
    
    // Configure layer range if partitioned
    if (strategy == WeightDistributionStrategy::LAYER_PARTITIONED) {
        auto& wm = ctx->concreteWeightManager();
        wm->setLayerRange(config.first_layer, config.last_layer);
        wm->setHasEmbedding(config.has_embedding);
        wm->setHasLmHead(config.has_lm_head);
    }
    
    // Configure sharding if needed
    if (config.total_shards > 1) {
        auto& wm = ctx->concreteWeightManager();
        wm->setShardInfo(config.shard_index, config.total_shards, config.work_fraction);
    }
    
    return ctx;
}
```

### 1.3 Keep legacy signatures as deprecated wrappers

```cpp
// Deprecated: Use create(path, ModelContextConfig) instead
[[deprecated("Use create(path, config) with ModelContextConfig")]]
static std::shared_ptr<ModelContext> create(
    const std::string& model_path,
    std::shared_ptr<MPIContext> mpi_ctx,
    std::shared_ptr<WeightPlacementMap> placement_map,
    TensorFactory* factory,
    WeightDistributionStrategy strategy,
    WeightPrecision weight_precision);

// Deprecated: Use create(path, ModelContextConfig::forPPStage()) instead
[[deprecated("Use create(path, ModelContextConfig::forPPStage())")]]
static std::shared_ptr<ModelContext> createForPPStage(...);
```

---

## Phase 2: Complete OrchestrationRunner PP Support (Week 2)

### 2.1 Update `OrchestrationRunner::loadWeights()`

**File**: `src/v2/execution/runner/OrchestrationRunner.cpp` (line 680)

```cpp
bool OrchestrationRunner::loadWeights()
{
    std::string model_path = config_.model_path;
    if (model_path.empty()) {
        LOG_DEBUG("No model path specified, skipping weight loading");
        return true;
    }

    // Build config from execution plan
    auto ctx_config = ModelContextConfig::fromExecutionPlan(plan_);
    ctx_config.mpi_ctx = mpi_ctx_;
    
    model_ctx_ = ModelContext::create(model_path, ctx_config);
    if (!model_ctx_) {
        return setError("Failed to create ModelContext for: " + model_path);
    }

    tokenizer_ = createTokenizer(model_ctx_);
    if (!tokenizer_) {
        LOG_WARN("Failed to create tokenizer from model context");
    }

    LOG_INFO("Model context created from: " << model_path
             << " [layers " << plan_.first_layer << "-" << plan_.last_layer
             << ", embedding=" << plan_.has_embedding
             << ", lm_head=" << plan_.has_lm_head << "]");
    return true;
}
```

### 2.2 Implement `ModelContextConfig::fromExecutionPlan()`

```cpp
ModelContextConfig ModelContextConfig::fromExecutionPlan(const RankExecutionPlan& plan)
{
    ModelContextConfig config;
    
    // Layer range from PP
    config.first_layer = plan.first_layer;
    config.last_layer = plan.last_layer;
    config.has_embedding = plan.has_embedding;
    config.has_lm_head = plan.has_lm_head;
    
    // Sharding from TP
    config.shard_index = plan.weight_shard.shard_index;
    config.total_shards = plan.weight_shard.total_shards;
    config.work_fraction = plan.weight_shard.work_fraction;
    
    // Auto-select strategy
    bool has_pp = (plan.first_layer != 0 || plan.last_layer != -1 ||
                   !plan.has_embedding || !plan.has_lm_head);
    bool has_tp = (plan.weight_shard.total_shards > 1);
    
    if (has_pp && has_tp) {
        config.strategy = WeightDistributionStrategy::LAYER_PARTITIONED; // PP + TP
    } else if (has_pp) {
        config.strategy = WeightDistributionStrategy::LAYER_PARTITIONED;
    } else if (has_tp) {
        config.strategy = WeightDistributionStrategy::SHARDED;
    } else {
        config.strategy = WeightDistributionStrategy::REPLICATED;
    }
    
    return config;
}
```

---

## Phase 3: Migrate Tests to OrchestrationRunner (Week 3-4)

### 3.1 Create Test Helper: `TestOrchestrationHelper`

**File**: `tests/v2/utils/TestOrchestrationHelper.h`

```cpp
/**
 * @brief Helper for creating OrchestrationRunner in tests
 * 
 * Provides the same simplicity as createInferenceRunner() but uses
 * the modern OrchestrationRunner path.
 */
class TestOrchestrationHelper {
public:
    /**
     * @brief Create a simple single-device runner for testing
     */
    static std::unique_ptr<IOrchestrationRunner> createSimple(
        const std::string& model_path,
        DeviceId device = DeviceId::cpu());
    
    /**
     * @brief Create a runner with custom config
     */
    static std::unique_ptr<IOrchestrationRunner> create(
        const std::string& model_path,
        const OrchestrationConfig& config);
    
    /**
     * @brief Create OrchestrationConfig from legacy InferenceRunnerConfig
     * 
     * Bridge for migrating tests incrementally.
     */
    static OrchestrationConfig fromLegacyConfig(
        const std::string& model_path,
        DeviceId device,
        const InferenceRunnerConfig& legacy_config);
};
```

### 3.2 Migration Order (by impact)

| Priority | File | Usages | Migration Notes |
|----------|------|--------|-----------------|
| 1 | `ParityTestBase.h` | 1 | **Highest impact** - migrates ALL parity tests |
| 2 | `Qwen2ParityTestBase.h` | 1 | Qwen2-specific override |
| 3 | Pipeline tests (5 files) | ~25 | Medium effort |
| 4 | Other integration tests | ~10 | Straightforward |
| 5 | Unit tests | ~5 | May require mock updates |

### 3.3 ParityTestBase Migration

**File**: `tests/v2/integration/parity/ParityTestBase.h` (line 1392)

```cpp
// BEFORE:
runner_ = createInferenceRunner(model_ctx_, mpi_ctx_, device, inf_config);

// AFTER:
auto orch_config = TestOrchestrationHelper::fromLegacyConfig(
    model_ctx_->path(), device, inf_config);
auto runner = TestOrchestrationHelper::create(model_ctx_->path(), orch_config);
runner->initialize();
runner_ = std::move(runner);
```

---

## Phase 4: Deprecate InferenceRunnerFactory (Week 5)

### 4.1 Mark public functions as deprecated

**File**: `src/v2/execution/factory/InferenceRunnerFactory.h`

```cpp
// Add deprecation warnings
[[deprecated("Use OrchestrationRunnerFactory::createFromOrchestrationConfig() instead")]]
std::unique_ptr<IInferenceRunner> createInferenceRunner(...);

[[deprecated("Use OrchestrationRunner with TestOrchestrationHelper instead")]]
std::unique_ptr<IInferenceRunner> createTestableInferenceRunner(...);

// Keep multi-device - still needed internally
std::unique_ptr<IMultiDeviceOrchestrator> createMultiDeviceOrchestrator(...);
```

### 4.2 Move to internal header

After tests are migrated:

```
src/v2/execution/factory/InferenceRunnerFactory.h        → DELETE (public)
src/v2/execution/factory/internal/InferenceRunnerImpl.h  → NEW (internal)
```

---

## Phase 5: Cleanup (Week 6)

### 5.1 Remove deprecated code paths

- Delete `createInferenceRunner()` and `createTestableInferenceRunner()`
- Keep `createMultiDeviceOrchestrator()` (used by OrchestrationRunner)
- Simplify `MultiDeviceOrchestrator` to use OrchestrationConfig directly

### 5.2 Update documentation

- Update `.github/copilot-instructions.md`
- Update `.github/instructions/llaminar-architecture-v2.instructions.md`
- Update README diagrams

---

## Testing Strategy

### Unit Tests for New Code
- `Test__ModelContextConfig.cpp` - Config validation, factory helpers
- `Test__OrchestrationRunner_LayerPartitioned.cpp` - PP weight loading

### Integration Tests
- Run existing parity tests after migration (should be green)
- Add PP-specific parity tests using new API

### Regression Tests
- Compare old vs new path outputs before removing old path
- Benchmark to ensure no performance regression

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Breaking existing tests | Deprecation warnings first, fix one file at a time |
| Performance regression | Benchmark before/after each phase |
| Missing functionality | Map all InferenceRunnerConfig fields to OrchestrationConfig |
| PP bugs | Validate with existing PP tests before removing old path |

---

## Success Criteria

1. ✅ All tests pass using OrchestrationRunner
2. ✅ InferenceRunnerFactory removed from public API
3. ✅ PP parity tests pass with LAYER_PARTITIONED (not REPLICATED)
4. ✅ Single code path for ModelContext weight loading
5. ✅ No performance regression in benchmarks

---

## Timeline

| Week | Phase | Deliverables |
|------|-------|--------------|
| 1 | ModelContext unification | `ModelContextConfig`, unified `create()` |
| 2 | OrchestrationRunner PP | `loadWeights()` uses plan, PP tests pass |
| 3-4 | Test migration | All tests use OrchestrationRunner |
| 5 | Deprecation | Warnings added, internal header created |
| 6 | Cleanup | Old code removed, docs updated |

---

## Appendix: Dependency Graph

```
                    ┌─────────────────────────────────────────┐
                    │      OrchestrationRunnerFactory        │
                    │  (CLI entry point - modern path)       │
                    └────────────────┬────────────────────────┘
                                     │
                                     ▼
                    ┌─────────────────────────────────────────┐
                    │         OrchestrationRunner            │
                    │  (owns ModelContext, execution plan)   │
                    └────────────────┬────────────────────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              │                      │                      │
              ▼                      ▼                      ▼
    ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
    │  ModelContext   │   │ IInferenceRunner│   │ ILocalTPContext │
    │ (unified API)   │   │ (internal impl) │   │ (collectives)   │
    └────────┬────────┘   └────────┬────────┘   └─────────────────┘
             │                     │
             ▼                     ▼
    ┌─────────────────┐   ┌─────────────────────────────────────┐
    │  WeightManager  │   │    InferenceRunnerFactory          │
    │ (layer ranges)  │   │  (INTERNAL - not public anymore)   │
    └─────────────────┘   └─────────────────────────────────────┘
```
