# Project: Orchestration Consolidation

> **Status**: Active  
> **Started**: 2026-02-01  
> **Owner**: David Sanftenberg  
> **Branch**: `tensor-parallel`

## Executive Summary

Consolidate the diverged inference runner code paths by completing `OrchestrationRunner` as the canonical entry point and deprecating `InferenceRunnerFactory` as a public API. Create a unified `ModelContextConfig` API that handles all weight loading scenarios (single device, TP, PP) through a single code path.

---

## Problem Statement

The codebase has two parallel paths for creating inference runners:

| Path | Entry Point | Weight Loading | PP Support |
|------|-------------|----------------|------------|
| **Legacy** | `InferenceRunnerFactory::createInferenceRunner()` | Hardcodes global weight requirement | ❌ Blocked |
| **Modern** | `OrchestrationRunnerFactory::createFromArgs()` | Uses `RankExecutionPlan` | ✅ Designed for PP |

This divergence:
1. Creates maintenance burden (two code paths to update)
2. Blocks PP support in tests (parity tests use legacy path)
3. Causes confusion about which API to use

---

## Goals

| # | Goal | Success Metric |
|---|------|----------------|
| G1 | Single code path for weight loading | `ModelContextConfig` used everywhere |
| G2 | PP parity tests work with `LAYER_PARTITIONED` | 18 PP parity tests pass without `REPLICATED` workaround |
| G3 | `InferenceRunnerFactory` removed from public API | No external usages remain |
| G4 | No performance regression | Benchmark ≥ baseline throughput |

---

## Non-Goals

- Pipeline parallelism micro-batching (future work)
- New collective backends (separate project)
- Model architecture changes

---

## Architecture

### Before (Diverged)

```
CLI ─────────────────────────────────────────────────────────────────┐
                                                                      │
                                        ┌──────────────────────────┐  │
Tests ──► InferenceRunnerFactory ──────►│ DeviceGraphOrchestrator  │  │
                                        └──────────────────────────┘  │
                                                                      │
                                        ┌──────────────────────────┐  │
CLI ────► OrchestrationRunnerFactory ──►│   OrchestrationRunner    │──┘
                                        │  (uses InferenceRunner   │
                                        │   internally)            │
                                        └──────────────────────────┘
```

### After (Unified)

```
                                        ┌──────────────────────────┐
CLI ────► OrchestrationRunnerFactory ──►│   OrchestrationRunner    │
Tests ──► TestOrchestrationHelper ──────►│  (canonical entry point) │
                                        └────────────┬─────────────┘
                                                     │
                                                     ▼
                                        ┌──────────────────────────┐
                                        │     ModelContext         │
                                        │  (unified config API)    │
                                        └────────────┬─────────────┘
                                                     │
                               ┌─────────────────────┼─────────────────────┐
                               │                     │                     │
                               ▼                     ▼                     ▼
                        Single Device              TP Shard           PP Stage
                        (full model)           (sharded weights)  (layer partition)
```

---

## Phases

### Phase 1: Unified ModelContext API
**Duration**: 2-3 days  
**Risk**: Low  
**Dependencies**: None

Create `ModelContextConfig` struct and update `ModelContext::create()` to use it.

### Phase 2: OrchestrationRunner PP Integration  
**Duration**: 1-2 days  
**Risk**: Medium  
**Dependencies**: Phase 1

Wire `OrchestrationRunner::loadWeights()` to use `ModelContextConfig::fromExecutionPlan()`.

### Phase 3: Test Migration
**Duration**: 3-5 days  
**Risk**: Medium  
**Dependencies**: Phase 2

Migrate all tests from `InferenceRunnerFactory` to `OrchestrationRunner`.

### Phase 4: Deprecation and Cleanup
**Duration**: 1-2 days  
**Risk**: Low  
**Dependencies**: Phase 3

Remove `InferenceRunnerFactory` from public API.

---

## Phase 1: Unified ModelContext API

### Tasks

| ID | Task | File(s) | Est. |
|----|------|---------|------|
| 1.1 | Create `ModelContextConfig` struct | `src/v2/loaders/ModelContextConfig.h` | 1h |
| 1.2 | Add factory helper methods | `src/v2/loaders/ModelContextConfig.cpp` | 1h |
| 1.3 | Add `ModelContext::create(path, config)` overload | `src/v2/loaders/ModelContext.h/cpp` | 2h |
| 1.4 | Add `setHasEmbedding()`/`setHasLmHead()` to WeightManager | `src/v2/loaders/WeightManager.h/cpp` | 1h |
| 1.5 | Unit tests for `ModelContextConfig` | `tests/v2/unit/loaders/Test__ModelContextConfig.cpp` | 2h |
| 1.6 | Integration test: PP stage loading | `tests/v2/integration/loaders/Test__ModelContext_PPStage.cpp` | 2h |

### Task 1.1: Create ModelContextConfig Struct

**File**: `src/v2/loaders/ModelContextConfig.h`

```cpp
#pragma once

#include "WeightManager.h"
#include "../utils/MPIContext.h"
#include "../execution/mpi_orchestration/RankExecutionPlan.h"
#include <memory>
#include <string>

namespace llaminar2 {

/**
 * @brief Configuration for ModelContext creation
 * 
 * Unified configuration that supports all scenarios:
 * - Single device (full model)
 * - Tensor Parallelism (sharded weights)
 * - Pipeline Parallelism (layer partitions)
 * - Combined TP + PP
 */
struct ModelContextConfig {
    // =========================================================================
    // Core Settings
    // =========================================================================
    
    /// MPI context for distributed weight management (nullptr = single rank)
    std::shared_ptr<MPIContext> mpi_ctx;
    
    /// Weight distribution strategy (auto-selected if not specified)
    WeightDistributionStrategy strategy = WeightDistributionStrategy::REPLICATED;
    
    /// Weight loading precision
    WeightPrecision weight_precision = WeightPrecision::NATIVE;
    
    // =========================================================================
    // Pipeline Parallelism (Layer Range)
    // =========================================================================
    
    /// First layer to load (inclusive, 0-indexed)
    int first_layer = 0;
    
    /// Last layer to load (inclusive, -1 = all remaining layers)
    int last_layer = -1;
    
    /// Whether to load embedding weights (typically true for first PP stage)
    bool has_embedding = true;
    
    /// Whether to load output_norm and lm_head (typically true for last PP stage)
    bool has_lm_head = true;
    
    // =========================================================================
    // Tensor Parallelism (Weight Sharding)
    // =========================================================================
    
    /// Which shard this rank loads (0-indexed)
    int shard_index = 0;
    
    /// Total number of shards (1 = no sharding)
    int total_shards = 1;
    
    /// Work fraction for proportional TP (default: equal split)
    float work_fraction = 1.0f;
    
    // =========================================================================
    // Advanced Settings
    // =========================================================================
    
    /// Fine-grained weight placement decisions (nullptr = default)
    std::shared_ptr<WeightPlacementMap> placement_map;
    
    /// Custom tensor factory for NUMA-aware allocation (nullptr = default)
    TensorFactory* factory = nullptr;
    
    // =========================================================================
    // Factory Helpers
    // =========================================================================
    
    /// Default configuration (single device, full model)
    static ModelContextConfig defaults();
    
    /// Configuration for a PP stage
    /// @param stage_idx Which stage (0-indexed)
    /// @param total_stages Total PP stages
    /// @param n_layers Total layers in model
    static ModelContextConfig forPPStage(int stage_idx, int total_stages, int n_layers);
    
    /// Configuration for a TP shard
    /// @param shard_idx Which shard (0-indexed)
    /// @param total_shards Total shards
    static ModelContextConfig forTPShard(int shard_idx, int total_shards);
    
    /// Configuration from RankExecutionPlan
    /// @param plan Execution plan containing PP/TP settings
    static ModelContextConfig fromExecutionPlan(const RankExecutionPlan& plan);
    
    // =========================================================================
    // Validation
    // =========================================================================
    
    /// Validate configuration
    /// @return Empty vector if valid, otherwise list of error messages
    std::vector<std::string> validate() const;
    
    /// Check if this config specifies layer partitioning
    bool isLayerPartitioned() const;
    
    /// Check if this config specifies weight sharding
    bool isSharded() const;
    
    /// String representation for debugging
    std::string toString() const;
};

} // namespace llaminar2
```

### Task 1.2: Factory Helper Implementations

**File**: `src/v2/loaders/ModelContextConfig.cpp`

```cpp
#include "ModelContextConfig.h"
#include <sstream>
#include <cmath>

namespace llaminar2 {

ModelContextConfig ModelContextConfig::defaults()
{
    return ModelContextConfig{};
}

ModelContextConfig ModelContextConfig::forPPStage(int stage_idx, int total_stages, int n_layers)
{
    if (total_stages <= 0 || stage_idx < 0 || stage_idx >= total_stages) {
        return defaults(); // Invalid params, return default
    }
    
    ModelContextConfig config;
    config.strategy = WeightDistributionStrategy::LAYER_PARTITIONED;
    
    // Divide layers evenly across stages
    int layers_per_stage = n_layers / total_stages;
    int remainder = n_layers % total_stages;
    
    // Calculate this stage's layer range
    config.first_layer = stage_idx * layers_per_stage + std::min(stage_idx, remainder);
    int next_first = (stage_idx + 1) * layers_per_stage + std::min(stage_idx + 1, remainder);
    config.last_layer = next_first - 1;
    
    // First stage gets embedding, last stage gets lm_head
    config.has_embedding = (stage_idx == 0);
    config.has_lm_head = (stage_idx == total_stages - 1);
    
    return config;
}

ModelContextConfig ModelContextConfig::forTPShard(int shard_idx, int total_shards)
{
    ModelContextConfig config;
    config.strategy = WeightDistributionStrategy::SHARDED;
    config.shard_index = shard_idx;
    config.total_shards = total_shards;
    config.work_fraction = 1.0f / total_shards;
    return config;
}

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
    if (config.isLayerPartitioned()) {
        config.strategy = WeightDistributionStrategy::LAYER_PARTITIONED;
    } else if (config.isSharded()) {
        config.strategy = WeightDistributionStrategy::SHARDED;
    } else {
        config.strategy = WeightDistributionStrategy::REPLICATED;
    }
    
    return config;
}

std::vector<std::string> ModelContextConfig::validate() const
{
    std::vector<std::string> errors;
    
    if (first_layer < 0) {
        errors.push_back("first_layer must be >= 0");
    }
    if (last_layer >= 0 && last_layer < first_layer) {
        errors.push_back("last_layer must be >= first_layer (or -1 for all)");
    }
    if (shard_index < 0) {
        errors.push_back("shard_index must be >= 0");
    }
    if (total_shards < 1) {
        errors.push_back("total_shards must be >= 1");
    }
    if (shard_index >= total_shards) {
        errors.push_back("shard_index must be < total_shards");
    }
    if (work_fraction <= 0.0f || work_fraction > 1.0f) {
        errors.push_back("work_fraction must be in (0.0, 1.0]");
    }
    
    return errors;
}

bool ModelContextConfig::isLayerPartitioned() const
{
    return first_layer != 0 || last_layer != -1 || !has_embedding || !has_lm_head;
}

bool ModelContextConfig::isSharded() const
{
    return total_shards > 1;
}

std::string ModelContextConfig::toString() const
{
    std::ostringstream ss;
    ss << "ModelContextConfig{";
    ss << "layers=[" << first_layer << "," << last_layer << "]";
    ss << ", emb=" << has_embedding;
    ss << ", lm=" << has_lm_head;
    if (isSharded()) {
        ss << ", shard=" << shard_index << "/" << total_shards;
    }
    ss << ", strategy=" << static_cast<int>(strategy);
    ss << "}";
    return ss.str();
}

} // namespace llaminar2
```

### Task 1.3: ModelContext::create() with Config

Add new overload to `ModelContext.h`:

```cpp
/**
 * @brief Create model context with unified configuration
 *
 * This is the preferred factory method. Supports all scenarios:
 * - Single device (default config)
 * - TP (config.total_shards > 1)
 * - PP (config.first_layer != 0 or config.last_layer != -1)
 * - Combined TP + PP
 *
 * @param model_path Path to GGUF model file
 * @param config Configuration for weight loading
 * @return Shared pointer to context, or nullptr on error
 */
static std::shared_ptr<ModelContext> create(
    const std::string& model_path,
    const ModelContextConfig& config);
```

Implementation in `ModelContext.cpp`:

```cpp
std::shared_ptr<ModelContext> ModelContext::create(
    const std::string& model_path,
    const ModelContextConfig& config)
{
    // Validate config
    auto errors = config.validate();
    if (!errors.empty()) {
        for (const auto& e : errors) {
            LOG_ERROR("[ModelContext] Config error: " << e);
        }
        return nullptr;
    }
    
    // Determine strategy
    WeightDistributionStrategy strategy = config.strategy;
    if (config.isLayerPartitioned() && strategy == WeightDistributionStrategy::REPLICATED) {
        strategy = WeightDistributionStrategy::LAYER_PARTITIONED;
    }
    
    // Create context using existing private constructor
    auto ctx = std::shared_ptr<ModelContext>(new ModelContext(
        model_path,
        config.mpi_ctx,
        config.placement_map,
        config.factory,
        strategy));
    
    if (!ctx || !ctx->weight_manager_) {
        LOG_ERROR("[ModelContext] Failed to create context for: " << model_path);
        return nullptr;
    }
    
    // Configure layer range if partitioned
    if (strategy == WeightDistributionStrategy::LAYER_PARTITIONED) {
        ctx->weight_manager_->setLayerRange(config.first_layer, config.last_layer);
        ctx->weight_manager_->setHasEmbedding(config.has_embedding);
        ctx->weight_manager_->setHasLmHead(config.has_lm_head);
    }
    
    // Configure sharding if needed
    if (config.isSharded()) {
        ctx->weight_manager_->setShardInfo(
            config.shard_index,
            config.total_shards,
            config.work_fraction);
    }
    
    LOG_DEBUG("[ModelContext] Created with " << config.toString());
    return ctx;
}
```

### Task 1.4: WeightManager Global Weight Flags

Add to `WeightManager.h`:

```cpp
/**
 * @brief Set whether this weight manager should provide embedding weights
 * @param has_embedding True if embedding should be loaded
 */
void setHasEmbedding(bool has_embedding);

/**
 * @brief Set whether this weight manager should provide lm_head weights
 * @param has_lm_head True if output_norm and lm_head should be loaded
 */
void setHasLmHead(bool has_lm_head);
```

Add to `WeightManager.cpp`:

```cpp
void WeightManager::setHasEmbedding(bool has_embedding)
{
    has_embedding_ = has_embedding;
    LOG_DEBUG("[WeightManager] has_embedding = " << has_embedding);
}

void WeightManager::setHasLmHead(bool has_lm_head)
{
    has_lm_head_ = has_lm_head;
    LOG_DEBUG("[WeightManager] has_lm_head = " << has_lm_head);
}
```

Update `isWeightInLayerRange()` to check global flags:

```cpp
bool WeightManager::isWeightInLayerRange(const std::string& name) const
{
    // Check global weights
    if (name == "token_embd.weight") {
        return has_embedding_;
    }
    if (name == "output_norm.weight" || name == "output.weight") {
        return has_lm_head_;
    }
    
    // Check layer weights
    // ... existing layer extraction logic ...
}
```

### Task 1.5: Unit Tests

**File**: `tests/v2/unit/loaders/Test__ModelContextConfig.cpp`

```cpp
#include <gtest/gtest.h>
#include "loaders/ModelContextConfig.h"

using namespace llaminar2;

class Test__ModelContextConfig : public ::testing::Test {};

TEST(Test__ModelContextConfig, Defaults_FullModel)
{
    auto config = ModelContextConfig::defaults();
    
    EXPECT_EQ(config.first_layer, 0);
    EXPECT_EQ(config.last_layer, -1);
    EXPECT_TRUE(config.has_embedding);
    EXPECT_TRUE(config.has_lm_head);
    EXPECT_EQ(config.total_shards, 1);
    EXPECT_FALSE(config.isLayerPartitioned());
    EXPECT_FALSE(config.isSharded());
}

TEST(Test__ModelContextConfig, ForPPStage_FirstStage)
{
    auto config = ModelContextConfig::forPPStage(0, 2, 24);
    
    EXPECT_EQ(config.first_layer, 0);
    EXPECT_EQ(config.last_layer, 11);  // 0-11 = 12 layers
    EXPECT_TRUE(config.has_embedding);
    EXPECT_FALSE(config.has_lm_head);
    EXPECT_TRUE(config.isLayerPartitioned());
}

TEST(Test__ModelContextConfig, ForPPStage_LastStage)
{
    auto config = ModelContextConfig::forPPStage(1, 2, 24);
    
    EXPECT_EQ(config.first_layer, 12);
    EXPECT_EQ(config.last_layer, 23);  // 12-23 = 12 layers
    EXPECT_FALSE(config.has_embedding);
    EXPECT_TRUE(config.has_lm_head);
}

TEST(Test__ModelContextConfig, ForPPStage_ThreeStages)
{
    // 24 layers / 3 stages = 8 layers each
    auto stage0 = ModelContextConfig::forPPStage(0, 3, 24);
    auto stage1 = ModelContextConfig::forPPStage(1, 3, 24);
    auto stage2 = ModelContextConfig::forPPStage(2, 3, 24);
    
    EXPECT_EQ(stage0.first_layer, 0);
    EXPECT_EQ(stage0.last_layer, 7);
    EXPECT_TRUE(stage0.has_embedding);
    
    EXPECT_EQ(stage1.first_layer, 8);
    EXPECT_EQ(stage1.last_layer, 15);
    EXPECT_FALSE(stage1.has_embedding);
    EXPECT_FALSE(stage1.has_lm_head);
    
    EXPECT_EQ(stage2.first_layer, 16);
    EXPECT_EQ(stage2.last_layer, 23);
    EXPECT_TRUE(stage2.has_lm_head);
}

TEST(Test__ModelContextConfig, ForTPShard)
{
    auto config = ModelContextConfig::forTPShard(1, 4);
    
    EXPECT_EQ(config.shard_index, 1);
    EXPECT_EQ(config.total_shards, 4);
    EXPECT_FLOAT_EQ(config.work_fraction, 0.25f);
    EXPECT_TRUE(config.isSharded());
    EXPECT_FALSE(config.isLayerPartitioned());
}

TEST(Test__ModelContextConfig, Validation_Valid)
{
    auto config = ModelContextConfig::defaults();
    auto errors = config.validate();
    EXPECT_TRUE(errors.empty());
}

TEST(Test__ModelContextConfig, Validation_InvalidShardIndex)
{
    auto config = ModelContextConfig::defaults();
    config.shard_index = 5;
    config.total_shards = 4;
    
    auto errors = config.validate();
    EXPECT_FALSE(errors.empty());
    EXPECT_TRUE(errors[0].find("shard_index") != std::string::npos);
}

TEST(Test__ModelContextConfig, Validation_InvalidLayerRange)
{
    auto config = ModelContextConfig::defaults();
    config.first_layer = 10;
    config.last_layer = 5;
    
    auto errors = config.validate();
    EXPECT_FALSE(errors.empty());
}

TEST(Test__ModelContextConfig, FromExecutionPlan)
{
    RankExecutionPlan plan;
    plan.first_layer = 12;
    plan.last_layer = 23;
    plan.has_embedding = false;
    plan.has_lm_head = true;
    plan.weight_shard.shard_index = 0;
    plan.weight_shard.total_shards = 2;
    plan.weight_shard.work_fraction = 0.6f;
    
    auto config = ModelContextConfig::fromExecutionPlan(plan);
    
    EXPECT_EQ(config.first_layer, 12);
    EXPECT_EQ(config.last_layer, 23);
    EXPECT_FALSE(config.has_embedding);
    EXPECT_TRUE(config.has_lm_head);
    EXPECT_EQ(config.shard_index, 0);
    EXPECT_EQ(config.total_shards, 2);
    EXPECT_FLOAT_EQ(config.work_fraction, 0.6f);
}
```

### Acceptance Criteria (Phase 1)

- [ ] `ModelContextConfig` struct compiles and validates
- [ ] Factory helpers produce correct configurations
- [ ] `ModelContext::create(path, config)` works
- [ ] WeightManager respects `has_embedding`/`has_lm_head` flags
- [ ] All unit tests pass
- [ ] Integration test loads partial weights correctly

---

## Phase 2: OrchestrationRunner PP Integration

### Tasks

| ID | Task | File(s) | Est. |
|----|------|---------|------|
| 2.1 | Update `loadWeights()` to use `ModelContextConfig` | `OrchestrationRunner.cpp` | 1h |
| 2.2 | Integration test: OrchestrationRunner with PP | `tests/v2/integration/` | 2h |
| 2.3 | Verify PP parity tests work | existing tests | 1h |

### Task 2.1: Update loadWeights()

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
    ModelContextConfig ctx_config = ModelContextConfig::fromExecutionPlan(plan_);
    ctx_config.mpi_ctx = mpi_ctx_;
    ctx_config.weight_precision = WeightPrecision::NATIVE;
    
    // Create ModelContext with unified API
    model_ctx_ = ModelContext::create(model_path, ctx_config);
    
    if (!model_ctx_) {
        return setError("Failed to create ModelContext for: " + model_path);
    }

    // Create tokenizer
    tokenizer_ = createTokenizer(model_ctx_);
    if (!tokenizer_) {
        LOG_WARN("Failed to create tokenizer from model context");
    }

    LOG_INFO("Model loaded: " << model_path << " " << ctx_config.toString());
    return true;
}
```

### Acceptance Criteria (Phase 2)

- [ ] `OrchestrationRunner::loadWeights()` uses `ModelContextConfig::fromExecutionPlan()`
- [ ] PP stages load only their assigned weights
- [ ] Memory usage reduced ~50% for 2-stage PP
- [ ] PP parity tests pass with `LAYER_PARTITIONED` (not `REPLICATED` workaround)

---

## Phase 3: Test Migration

### Tasks

| ID | Task | File(s) | Est. |
|----|------|---------|------|
| 3.1 | Create `TestOrchestrationHelper` | `tests/v2/utils/TestOrchestrationHelper.h/cpp` | 2h |
| 3.2 | Migrate `ParityTestBase.h` | `tests/v2/integration/parity/ParityTestBase.h` | 2h |
| 3.3 | Migrate `Qwen2ParityTestBase.h` | `tests/v2/integration/parity/qwen2/Qwen2ParityTestBase.h` | 1h |
| 3.4 | Migrate pipeline tests | 5 files | 4h |
| 3.5 | Migrate other integration tests | ~10 files | 4h |
| 3.6 | Remove/update factory unit tests | `Test__InferenceRunnerFactory*.cpp` | 2h |

### Task 3.1: TestOrchestrationHelper

**File**: `tests/v2/utils/TestOrchestrationHelper.h`

```cpp
#pragma once

#include "execution/runner/IOrchestrationRunner.h"
#include "execution/runner/IOrchestrationRunnerFactory.h"
#include "execution/factory/InferenceRunnerFactory.h"
#include "config/OrchestrationConfig.h"
#include "backends/DeviceId.h"
#include <memory>
#include <string>

namespace llaminar2::test {

/**
 * @brief Helper for creating OrchestrationRunner in tests
 * 
 * Provides simple factory methods that match the convenience of the
 * old InferenceRunnerFactory API while using the modern OrchestrationRunner.
 */
class TestOrchestrationHelper {
public:
    /**
     * @brief Create a simple single-device runner
     * 
     * Equivalent to old: createInferenceRunner(model_ctx, nullptr, device, {})
     */
    static std::unique_ptr<IOrchestrationRunner> createSimple(
        const std::string& model_path,
        DeviceId device = DeviceId::cpu());
    
    /**
     * @brief Create runner with full configuration
     */
    static std::unique_ptr<IOrchestrationRunner> create(
        const OrchestrationConfig& config);
    
    /**
     * @brief Convert legacy InferenceRunnerConfig to OrchestrationConfig
     * 
     * Bridge for incremental migration of tests.
     */
    static OrchestrationConfig fromLegacyConfig(
        const std::string& model_path,
        DeviceId device,
        const InferenceRunnerConfig& legacy);
    
    /**
     * @brief Create config for a PP stage
     */
    static OrchestrationConfig forPPStage(
        const std::string& model_path,
        int stage_idx,
        int total_stages,
        DeviceId device);
};

} // namespace llaminar2::test
```

### Migration Pattern for Tests

```cpp
// BEFORE (legacy):
auto model_ctx = ModelContext::create(model_path, mpi_ctx);
InferenceRunnerConfig config;
config.max_seq_len = 2048;
config.activation_precision = ActivationPrecision::FP32;
auto runner = createInferenceRunner(model_ctx, mpi_ctx, device, config);

// AFTER (modern):
OrchestrationConfig config = OrchestrationConfig::defaults();
config.model_path = model_path;
config.max_seq_len = 2048;
config.device_for_this_rank = GlobalDeviceAddress::fromDeviceId(device);
auto runner = TestOrchestrationHelper::create(config);
runner->initialize();
```

### Acceptance Criteria (Phase 3)

- [ ] `TestOrchestrationHelper` provides equivalent convenience to old API
- [ ] All parity tests migrated and passing
- [ ] All pipeline tests migrated and passing
- [ ] No remaining usages of `createInferenceRunner` in tests
- [ ] No remaining usages of `createTestableInferenceRunner` in tests

---

## Phase 4: Deprecation and Cleanup

### Tasks

| ID | Task | File(s) | Est. |
|----|------|---------|------|
| 4.1 | Mark public functions deprecated | `InferenceRunnerFactory.h` | 30m |
| 4.2 | Move to internal header | `src/v2/execution/factory/internal/` | 1h |
| 4.3 | Update `RankOrchestrator` | `RankOrchestrator.cpp` | 1h |
| 4.4 | Remove deprecated legacy overloads from `ModelContext` | `ModelContext.h/cpp` | 30m |
| 4.5 | Update documentation | `.github/instructions/*.md`, `README.md` | 2h |

### Acceptance Criteria (Phase 4)

- [ ] `InferenceRunnerFactory.h` not in public include path
- [ ] No deprecation warnings in clean build
- [ ] `createRankOrchestrator` still works (used internally)
- [ ] Documentation updated
- [ ] No test failures

---

## Test Plan

### Unit Tests (New)

| Test | File | Purpose |
|------|------|---------|
| `Test__ModelContextConfig` | `tests/v2/unit/loaders/` | Config validation, factory helpers |
| `Test__TestOrchestrationHelper` | `tests/v2/unit/utils/` | Test helper functionality |

### Integration Tests (New)

| Test | File | Purpose |
|------|------|---------|
| `Test__ModelContext_PPStage` | `tests/v2/integration/loaders/` | PP stage weight loading |
| `Test__OrchestrationRunner_PP` | `tests/v2/integration/runner/` | Full PP inference flow |

### Regression Tests (Existing)

Run all existing tests after each phase:
```bash
# Unit tests
ctest --test-dir build_v2 -R "^V2_Unit_" --output-on-failure --parallel

# Integration tests  
ctest --test-dir build_v2_integration -R "^V2_Integration_" --output-on-failure

# PP parity tests
ctest --test-dir build_v2_integration -R "V2_Integration_Parity_.*PP" --output-on-failure
```

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Breaking existing tests | Medium | High | Deprecation warnings first; migrate one file at a time |
| Performance regression | Low | Medium | Benchmark after each phase |
| Missing edge cases | Medium | Medium | Keep legacy path working until all tests pass |
| Complex merge conflicts | Low | Medium | Complete in focused branch, merge frequently |

---

## Subagent Delegation Plan

### Phase 1 Tasks (Can Parallelize)

| Task | Subagent Prompt Summary |
|------|------------------------|
| 1.1-1.2 | "Create ModelContextConfig.h and .cpp with struct definition and factory helpers" |
| 1.4 | "Add setHasEmbedding/setHasLmHead methods to WeightManager" |
| 1.5 | "Create unit tests for ModelContextConfig" |

### Phase 1 Tasks (Sequential)

| Task | Dependency |
|------|------------|
| 1.3 | After 1.1, 1.4 |
| 1.6 | After 1.3 |

### Phase 2-4

Execute sequentially, validating tests pass after each phase.

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Test pass rate | 100% | `ctest` results |
| PP memory reduction | ≥40% | `/usr/bin/time -v` peak RSS |
| API surface reduction | ≥2 public functions removed | Count exported symbols |
| Code duplication | 0 parallel paths | Grep for `createInferenceRunner` |

---

## Timeline

| Day | Phase | Deliverables |
|-----|-------|--------------|
| 1 | 1.1-1.4 | `ModelContextConfig`, WeightManager updates |
| 2 | 1.5-1.6, 2.1 | Unit tests, integration test, OrchestrationRunner update |
| 3 | 2.2-2.3 | PP integration tests, verify parity |
| 4 | 3.1-3.2 | TestOrchestrationHelper, ParityTestBase migration |
| 5 | 3.3-3.4 | Qwen2 parity, pipeline tests migration |
| 6 | 3.5-3.6 | Remaining tests, factory test updates |
| 7 | 4.1-4.5 | Deprecation, cleanup, documentation |

---

## Appendix: Files to Modify

### New Files
- `src/v2/loaders/ModelContextConfig.h`
- `src/v2/loaders/ModelContextConfig.cpp`
- `tests/v2/utils/TestOrchestrationHelper.h`
- `tests/v2/utils/TestOrchestrationHelper.cpp`
- `tests/v2/unit/loaders/Test__ModelContextConfig.cpp`
- `tests/v2/integration/loaders/Test__ModelContext_PPStage.cpp`

### Modified Files
- `src/v2/loaders/ModelContext.h` - Add `create(path, config)` overload
- `src/v2/loaders/ModelContext.cpp` - Implement new overload
- `src/v2/loaders/WeightManager.h` - Add `setHasEmbedding`, `setHasLmHead`
- `src/v2/loaders/WeightManager.cpp` - Implement new methods
- `src/v2/execution/runner/OrchestrationRunner.cpp` - Update `loadWeights()`
- `tests/v2/integration/parity/ParityTestBase.h` - Migrate to OrchestrationRunner
- `tests/v2/integration/parity/qwen2/Qwen2ParityTestBase.h` - Migrate to OrchestrationRunner
- (15+ additional test files)

### Deprecated/Removed Files
- `src/v2/execution/factory/InferenceRunnerFactory.h` → Move to internal/
- `src/v2/execution/factory/InferenceRunnerFactory.cpp` → Keep but demote

### Documentation Updates
- `.github/copilot-instructions.md`
- `.github/instructions/llaminar-architecture-v2.instructions.md`
- `README.md`
- `docs/v2/projects/2026-02/ORCHESTRATION_RUNNER_MIGRATION_PLAN.md`
