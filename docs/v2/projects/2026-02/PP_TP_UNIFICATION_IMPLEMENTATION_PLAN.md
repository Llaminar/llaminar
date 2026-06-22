# PP-TP Unification Implementation Plan

## Phase 4: Unified Pipeline Parallelism with Tensor Parallelism

**Document Version**: 1.0  
**Date**: February 2026  
**Author**: AI Architecture Assistant  
**Status**: Implementation Plan

---

## Executive Summary

This document provides a detailed implementation plan for unifying Pipeline Parallelism (PP) with the existing Tensor Parallelism (TP) pattern in Llaminar V2. The key insight is:

- **TP = "parallel multi-orchestrator"** - Multiple devices run the same layers simultaneously with sharded weights
- **PP = "sequential multi-orchestrator"** - Multiple devices run different layers sequentially with activation transfers
- **TP+PP = "sequential (parallel multi-orchestrator)"** - PP stages where each stage is a TP domain

The current single-orchestrator PP approach (`buildUnifiedPipelineGraph()`) is architecturally broken. This plan migrates to a multi-orchestrator approach matching the successful TP pattern.

---

## Table of Contents

1. [Current State Analysis](#1-current-state-analysis)
2. [Target Architecture](#2-target-architecture)
3. [Implementation Phases](#3-implementation-phases)
4. [Phase 4.1: Core Infrastructure](#4-phase-41-core-infrastructure)
5. [Phase 4.2: DeviceGraphOrchestrator Enhancements](#5-phase-42-devicegraphorchestrator-enhancements)
6. [Phase 4.3: RankOrchestrator PP Support](#6-phase-43-multideviceorchestrator-pp-support)
7. [Phase 4.4: Factory Function Migration](#7-phase-44-factory-function-migration)
8. [Phase 4.5: Deprecation and Cleanup](#8-phase-45-deprecation-and-cleanup)
9. [Test Plan](#9-test-plan)
10. [Risk Assessment](#10-risk-assessment)
11. [Migration Checklist](#11-migration-checklist)

---

## 1. Current State Analysis

### 1.1 TP Implementation (Working ✓)

**Pattern**: RankOrchestrator creates one DeviceGraphOrchestrator per device.

```
RankOrchestrator
├── DeviceGraphOrchestrator[cuda:0] ─┬─ forward() ─┐
├── DeviceGraphOrchestrator[cuda:1] ─┴─ forward() ─┤ std::async parallel
└── ILocalTPContext (allreduce, allgather)         │
                                                   ▼
                                           gatherLogits() → combined output
```

**Key Files**:
- `src/v2/execution/local_execution/orchestrators/RankOrchestrator.h` (526 lines)
- `src/v2/execution/local_execution/orchestrators/RankOrchestrator.cpp` (1371 lines)
- `src/v2/collective/ILocalTPContext.h` (373 lines)

### 1.2 PP Implementation (Broken ✗)

**Current Pattern**: Single DeviceGraphOrchestrator with `buildUnifiedPipelineGraph()`.

```
DeviceGraphOrchestrator (single instance)
├── setPipelineConfig()
├── initializeInferenceState() → creates pp_kv_caches map
├── buildUnifiedPipelineGraph() → single monolithic graph
└── Problems:
    - All weights must be accessible from single orchestrator
    - KV cache management is complex (pp_kv_caches map)
    - Device transitions within graph are error-prone
    - No activation buffer management between stages
```

**Key Files**:
- `src/v2/execution/factory/InferenceRunnerFactory.cpp` - `createUnifiedPipelineRunner()` (lines 1141-1280)
- `src/v2/models/qwen/Qwen2Graph.h` - `buildUnifiedPipelineGraph()` declaration
- `tests/v2/unit/models/qwen/Test__Qwen2Graph_PP.cpp` - 1200+ lines of tests

### 1.3 What Needs to Change

| Aspect | Current PP | Target PP |
|--------|------------|-----------|
| Orchestrator count | 1 | N (one per PP stage) |
| KV cache | `pp_kv_caches` map | Each orchestrator owns its KV cache |
| Graph building | `buildUnifiedPipelineGraph()` | `buildPartialForwardGraph()` per stage |
| Activation transfer | None (in-graph) | `ILocalPPContext.sendActivations()` |
| Weight loading | Complex lambda | Normal per-device loading |
| Factory | `createUnifiedPipelineRunner()` | `RankOrchestrator(pp_config)` |

---

## 2. Target Architecture

### 2.1 PP-Only (2-stage example)

```
RankOrchestrator (parallelism_mode = PP)
├── PPStageOrchestrator[stage=0, cuda:0]
│   ├── DeviceGraphOrchestrator
│   │   ├── layers 0-11
│   │   ├── embedding (has_embedding=true)
│   │   └── KV cache for layers 0-11
│   └── forward_hidden() → activations
├── PPStageOrchestrator[stage=1, cpu]
│   ├── DeviceGraphOrchestrator
│   │   ├── layers 12-23
│   │   ├── lm_head (has_lm_head=true)
│   │   └── KV cache for layers 12-23
│   └── forward_hidden() → logits
└── ILocalPPContext
    └── sendActivations(stage0→stage1)
```

### 2.2 TP-Only (existing, unchanged)

```
RankOrchestrator (parallelism_mode = TP)
├── DeviceGraphOrchestrator[cuda:0] ─┐
├── DeviceGraphOrchestrator[cuda:1] ─┤ parallel forward
└── ILocalTPContext                  │
    └── allreduce, allgather         ▼
```

### 2.3 TP+PP (composite)

```
RankOrchestrator (parallelism_mode = TP_PP)
├── PPStageOrchestrator[stage=0]
│   └── RankOrchestrator (TP)
│       ├── DeviceGraphOrchestrator[rocm:0] ─┐
│       ├── DeviceGraphOrchestrator[rocm:1] ─┤ parallel
│       └── ILocalTPContext (RCCL)           │
├── PPStageOrchestrator[stage=1]             ▼
│   └── DeviceGraphOrchestrator[cuda:0]
│       └── ILocalTPContext (none, single device)
└── ILocalPPContext
    └── sendActivations(stage0→stage1)
```

---

## 3. Implementation Phases

### 3.1 Phase Overview

```
Phase 4.1: Core Infrastructure (Week 1)
    ├── ParallelismMode enum
    ├── IStageOrchestrator interface
    ├── PPStageConfig enhancements
    └── Unit tests for new types

Phase 4.2: DeviceGraphOrchestrator Enhancements (Week 1-2)
    ├── Partial graph building (first_layer, last_layer)
    ├── forward_hidden() method
    ├── has_embedding / has_lm_head flags
    └── Unit tests for partial graphs

Phase 4.3: RankOrchestrator PP Support (Week 2-3)
    ├── Accept PipelineConfig
    ├── Strategy pattern (TP vs PP vs TP+PP)
    ├── Sequential forward logic for PP
    ├── Activation transfer via ILocalPPContext
    └── Integration tests for PP

Phase 4.4: Factory Function Migration (Week 3)
    ├── Deprecate createUnifiedPipelineRunner()
    ├── Create new factory path
    ├── Update test helpers
    └── Migration guide for tests

Phase 4.5: Deprecation and Cleanup (Week 4)
    ├── Remove buildUnifiedPipelineGraph()
    ├── Remove pp_kv_caches from InferenceState
    ├── Remove old PP-specific code
    └── Documentation updates
```

### 3.2 Dependency Graph

```
Phase 4.1 ──────────────────────────────────────┐
    │                                           │
    ▼                                           │
Phase 4.2 ──────────────────────────────────────┤
    │                                           │
    ├───────────────────┐                       │
    ▼                   ▼                       │
Phase 4.3           Phase 4.4                   │
    │                   │                       │
    └───────┬───────────┘                       │
            ▼                                   │
        Phase 4.5 ◄─────────────────────────────┘
```

---

## 4. Phase 4.1: Core Infrastructure

### 4.1.1 New File: ParallelismMode.h

**Path**: `src/v2/config/ParallelismMode.h`  
**LOC**: ~80 lines

```cpp
/**
 * @file ParallelismMode.h
 * @brief Parallelism mode enum for RankOrchestrator
 */

#pragma once

#include <string>

namespace llaminar2 {

/**
 * @brief Parallelism mode for multi-device execution
 */
enum class ParallelismMode {
    /// Single device, no parallelism
    SINGLE_DEVICE,
    
    /// Tensor Parallelism: parallel execution across devices
    /// Each device runs ALL layers with SHARDED weights
    TENSOR_PARALLEL,
    
    /// Pipeline Parallelism: sequential execution across devices
    /// Each device runs DIFFERENT layers with FULL weights (for that layer)
    PIPELINE_PARALLEL,
    
    /// Combined TP+PP: PP stages where each stage uses TP
    /// Sequential across stages, parallel within each stage
    TENSOR_PIPELINE_PARALLEL
};

/**
 * @brief Convert parallelism mode to string
 */
constexpr const char* parallelismModeToString(ParallelismMode mode) {
    switch (mode) {
        case ParallelismMode::SINGLE_DEVICE: return "SINGLE_DEVICE";
        case ParallelismMode::TENSOR_PARALLEL: return "TENSOR_PARALLEL";
        case ParallelismMode::PIPELINE_PARALLEL: return "PIPELINE_PARALLEL";
        case ParallelismMode::TENSOR_PIPELINE_PARALLEL: return "TENSOR_PIPELINE_PARALLEL";
        default: return "UNKNOWN";
    }
}

/**
 * @brief Determine parallelism mode from PipelineConfig
 * 
 * @param config Pipeline configuration
 * @return Detected parallelism mode
 */
ParallelismMode detectParallelismMode(const PipelineConfig& config);

} // namespace llaminar2
```

**Implementation Notes**:
- `detectParallelismMode()` checks:
  - `config.numStages() == 1 && config.maxTPDegree() == 1` → `SINGLE_DEVICE`
  - `config.numStages() == 1 && config.maxTPDegree() > 1` → `TENSOR_PARALLEL`
  - `config.numStages() > 1 && config.maxTPDegree() == 1` → `PIPELINE_PARALLEL`
  - `config.numStages() > 1 && config.maxTPDegree() > 1` → `TENSOR_PIPELINE_PARALLEL`

---

### 4.1.2 New File: IStageOrchestrator.h

**Path**: `src/v2/execution/local_execution/orchestrators/IStageOrchestrator.h`  
**LOC**: ~120 lines

```cpp
/**
 * @file IStageOrchestrator.h
 * @brief Interface for PP stage execution
 * 
 * Abstracts over:
 * - Single DeviceGraphOrchestrator (PP-only stage)
 * - RankOrchestrator (TP domain as PP stage)
 */

#pragma once

#include "../../../tensors/ITensor.h"
#include "../../../backends/DeviceId.h"
#include <memory>

namespace llaminar2 {

class TensorBase;
class IKVCache;

/**
 * @brief Configuration for a PP stage orchestrator
 */
struct PPStageOrchestratorConfig {
    int stage_id = 0;           ///< Stage index in pipeline
    int first_layer = 0;        ///< First layer (inclusive)
    int last_layer = 0;         ///< Last layer (exclusive)
    bool has_embedding = false; ///< Stage includes embedding lookup
    bool has_lm_head = false;   ///< Stage includes LM head projection
    DeviceId primary_device;    ///< Primary device for this stage
};

/**
 * @brief Interface for PP stage execution
 * 
 * A PP stage can be:
 * - A single DeviceGraphOrchestrator (simple PP)
 * - A RankOrchestrator (TP domain)
 */
class IStageOrchestrator {
public:
    virtual ~IStageOrchestrator() = default;

    // =========================================================================
    // Stage Information
    // =========================================================================
    
    /// Get stage configuration
    virtual const PPStageOrchestratorConfig& stageConfig() const = 0;
    
    /// Get stage ID
    virtual int stageId() const = 0;
    
    /// Check if this is the first stage (has embedding)
    virtual bool isFirstStage() const = 0;
    
    /// Check if this is the last stage (has lm_head)
    virtual bool isLastStage() const = 0;

    // =========================================================================
    // Hidden State Forward Pass
    // =========================================================================
    
    /**
     * @brief Forward pass with hidden state I/O
     * 
     * Used by non-first stages that receive activations from previous stage.
     * 
     * @param hidden_input Input hidden states [batch, seq_len, d_model]
     * @param seq_len Sequence length for this forward pass
     * @return Output hidden states (or logits if last stage)
     */
    virtual TensorBase* forward_hidden(TensorBase* hidden_input, int seq_len) = 0;

    /**
     * @brief Forward pass from tokens (first stage only)
     * 
     * @param tokens Input token IDs
     * @param seq_len Sequence length
     * @return Output hidden states (or logits if single stage)
     */
    virtual TensorBase* forward_tokens(const int* tokens, int seq_len) = 0;

    // =========================================================================
    // State Access
    // =========================================================================
    
    /// Get output hidden states buffer
    virtual TensorBase* outputHidden() = 0;
    
    /// Get KV cache for this stage's layers
    virtual IKVCache* kvCache() = 0;
    
    /// Clear KV cache
    virtual void clearCache() = 0;
    
    /// Get current position in sequence
    virtual int getPosition() const = 0;
};

} // namespace llaminar2
```

---

### 4.1.3 Modify: PPStageConfig.h

**Path**: `src/v2/config/PPStageConfig.h`  
**Changes**: Add orchestrator-friendly accessors

```cpp
// Add to existing PPStageConfig struct:

/**
 * @brief Convert to PPStageOrchestratorConfig
 */
PPStageOrchestratorConfig toOrchestratorConfig(DeviceId primary_device) const {
    PPStageOrchestratorConfig config;
    config.stage_id = stage_id;
    config.first_layer = first_layer;
    config.last_layer = last_layer;
    config.has_embedding = has_embedding;
    config.has_lm_head = has_lm_head;
    config.primary_device = primary_device;
    return config;
}
```

**Estimated Changes**: +15 lines

---

### 4.1.4 Test File: Test__ParallelismMode.cpp

**Path**: `tests/v2/unit/config/Test__ParallelismMode.cpp`  
**LOC**: ~150 lines

```cpp
TEST(Test__ParallelismMode, DetectsSingleDevice) {
    auto config = PipelineConfig::singleDevice(24, DeviceId::cuda(0));
    EXPECT_EQ(detectParallelismMode(config), ParallelismMode::SINGLE_DEVICE);
}

TEST(Test__ParallelismMode, DetectsTensorParallel) {
    auto config = PipelineConfig::tensorParallel(24, 
        {DeviceId::cuda(0), DeviceId::cuda(1)}, 
        CollectiveBackendType::NCCL);
    EXPECT_EQ(detectParallelismMode(config), ParallelismMode::TENSOR_PARALLEL);
}

TEST(Test__ParallelismMode, DetectsPipelineParallel) {
    auto config = PipelineConfig::pipelineParallel2Stage(24,
        DeviceId::cuda(0), 12,
        DeviceId::cpu(),
        CollectiveBackendType::HOST);
    EXPECT_EQ(detectParallelismMode(config), ParallelismMode::PIPELINE_PARALLEL);
}

TEST(Test__ParallelismMode, DetectsTPPP) {
    PipelineConfig config;
    config.total_layers = 24;
    config.tp_domains = {
        {"stage0_tp", {DeviceId::cuda(0), DeviceId::cuda(1)}, CollectiveBackendType::NCCL},
        {"stage1_tp", {DeviceId::rocm(0), DeviceId::rocm(1)}, CollectiveBackendType::RCCL}
    };
    config.pp_stages = {
        PPStageConfig::firstStage(0, "stage0_tp", 0, 12),
        PPStageConfig::lastStage(1, "stage1_tp", 12, 24)
    };
    EXPECT_EQ(detectParallelismMode(config), ParallelismMode::TENSOR_PIPELINE_PARALLEL);
}
```

---

### 4.1.5 Phase 4.1 Summary

| File | Action | LOC | Dependencies |
|------|--------|-----|--------------|
| `src/v2/config/ParallelismMode.h` | Create | 80 | PipelineConfig.h |
| `src/v2/config/ParallelismMode.cpp` | Create | 40 | ParallelismMode.h |
| `src/v2/execution/local_execution/orchestrators/IStageOrchestrator.h` | Create | 120 | ITensor.h |
| `src/v2/config/PPStageConfig.h` | Modify | +15 | - |
| `tests/v2/unit/config/Test__ParallelismMode.cpp` | Create | 150 | ParallelismMode.h |
| `tests/v2/CMakeLists.txt` | Modify | +10 | - |

**Total**: ~415 lines new code

---

## 5. Phase 4.2: DeviceGraphOrchestrator Enhancements

### 5.2.1 New Method: buildPartialForwardGraph()

**File**: `src/v2/models/qwen/Qwen2Graph.h`  
**Add**: New graph building method for PP stages

```cpp
/**
 * @brief Build a partial forward graph for PP stages
 * 
 * Unlike buildFullForwardGraph(), this method builds a graph for only
 * a subset of layers, optionally including embedding and/or LM head.
 * 
 * @param input Forward input (hidden states if not first stage, tokens if first)
 * @param output Forward output
 * @param stage_config PP stage configuration
 * @return ComputeGraph for this stage
 */
ComputeGraph buildPartialForwardGraph(
    const Qwen2ForwardInput& input,
    Qwen2ForwardOutput& output,
    const PPStageOrchestratorConfig& stage_config);
```

**Implementation** (in Qwen2Graph.cpp):

```cpp
ComputeGraph Qwen2Graph::buildPartialForwardGraph(
    const Qwen2ForwardInput& input,
    Qwen2ForwardOutput& output,
    const PPStageOrchestratorConfig& stage_config)
{
    ComputeGraph graph;
    
    // Embedding (first stage only)
    if (stage_config.has_embedding) {
        buildEmbeddingGraph(graph, input, output);
    }
    
    // Transformer layers [first_layer, last_layer)
    for (int layer_idx = stage_config.first_layer; 
         layer_idx < stage_config.last_layer; 
         ++layer_idx) {
        buildLayerGraph(graph, input, output, layer_idx);
    }
    
    // Final norm + LM head (last stage only)
    if (stage_config.has_lm_head) {
        buildFinalNormGraph(graph, input, output);
        buildLMHeadGraph(graph, input, output);
    }
    
    return graph;
}
```

**Estimated Changes**: +100 lines in Qwen2Graph.cpp

---

### 5.2.2 New Method: forward_hidden()

**File**: `src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h`  
**Add**: Method for PP stage forward with hidden state I/O

```cpp
// In DeviceGraphOrchestrator class:

/**
 * @brief Forward pass with hidden state input (for PP stages)
 * 
 * For non-first PP stages, accepts hidden states from previous stage
 * instead of tokens. The hidden states are written to the internal
 * state.hidden buffer before graph execution.
 * 
 * @param hidden_input Hidden states from previous stage [batch, seq_len, d_model]
 * @param seq_len Sequence length for this forward pass
 * @return Pointer to output (hidden states or logits depending on stage)
 */
TensorBase* forward_hidden(TensorBase* hidden_input, int seq_len);

/**
 * @brief Get output hidden states (for PP non-last stages)
 * 
 * Returns the hidden states buffer after the last layer processed
 * by this orchestrator. Used by PP to transfer activations.
 * 
 * @return Pointer to hidden states tensor, or nullptr if last stage
 */
TensorBase* outputHidden();
```

**Implementation**:

```cpp
TensorBase* DeviceGraphOrchestrator::forward_hidden(TensorBase* hidden_input, int seq_len)
{
    if (!state_.isInitialized()) {
        LOG_ERROR("DeviceGraphOrchestrator::forward_hidden: State not initialized");
        return nullptr;
    }

    // Copy hidden input to internal state
    // (Alternatively, use directly if coherence allows)
    if (hidden_input != state_.hidden.get()) {
        // Copy hidden_input data to state_.hidden
        std::memcpy(state_.hidden->mutable_data(),
                    hidden_input->data(),
                    seq_len * model_config_.d_model * sizeof(float));
    }

    // Build and execute partial graph
    // ... (similar to existing forward() but uses partial graph)
    
    // Return output (hidden or logits depending on stage)
    if (pp_stage_config_ && pp_stage_config_->has_lm_head) {
        return state_.logits.get();
    } else {
        return state_.hidden.get();
    }
}
```

**Estimated Changes**: +150 lines in DeviceGraphOrchestrator.cpp

---

### 5.2.3 New Field: pp_stage_config_

**File**: `src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h`  
**Add**: PP stage configuration storage

```cpp
private:
    // Existing members...
    
    /// PP stage configuration (set when this orchestrator is a PP stage)
    /// nullptr means full-model orchestration (no PP)
    std::optional<PPStageOrchestratorConfig> pp_stage_config_;
    
public:
    /**
     * @brief Set PP stage configuration
     * 
     * Configures this orchestrator as a PP stage with partial layers.
     * Must be called before initializeInferenceState().
     * 
     * @param config Stage configuration
     */
    void setPPStageConfig(const PPStageOrchestratorConfig& config);
    
    /**
     * @brief Get PP stage configuration
     * @return Optional containing stage config, or empty if not a PP stage
     */
    const std::optional<PPStageOrchestratorConfig>& ppStageConfig() const {
        return pp_stage_config_;
    }
    
    /**
     * @brief Check if this orchestrator is a PP stage
     */
    bool isPPStage() const { return pp_stage_config_.has_value(); }
```

---

### 5.2.4 Modify: initializeInferenceState()

**File**: `src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp`  
**Changes**: Handle partial layer KV cache allocation

```cpp
bool DeviceGraphOrchestrator::initializeInferenceState(
    int batch_size, int max_seq_len, DeviceId device_id,
    const InferenceStateInitConfig& init_config)
{
    // ... existing code ...
    
    // KV Cache allocation - respect PP stage boundaries
    int kv_n_layers = model_config_.n_layers;
    int kv_first_layer = 0;
    
    if (pp_stage_config_.has_value()) {
        // PP stage: only allocate KV for this stage's layers
        kv_n_layers = pp_stage_config_->last_layer - pp_stage_config_->first_layer;
        kv_first_layer = pp_stage_config_->first_layer;
        
        LOG_INFO("DeviceGraphOrchestrator: PP stage " << pp_stage_config_->stage_id
                 << " allocating KV cache for layers " << kv_first_layer
                 << " to " << pp_stage_config_->last_layer);
    }
    
    // Create KV cache for this stage's layers
    state_.kv_cache = KernelFactory::createKVCache(/* ... */);
    // Note: The KV cache internally maps layer_idx - kv_first_layer
    
    // ... rest of initialization ...
}
```

**Estimated Changes**: +80 lines modification

---

### 5.2.5 Test File: Test__DeviceGraphOrchestrator_PP.cpp

**Path**: `tests/v2/unit/orchestrators/Test__DeviceGraphOrchestrator_PP.cpp`  
**LOC**: ~400 lines

```cpp
/**
 * Unit tests for DeviceGraphOrchestrator PP stage functionality
 */

TEST(Test__DeviceGraphOrchestrator_PP, SetPPStageConfig) {
    // Setup orchestrator
    auto model_ctx = MockModelContextBuilder().usePreset(ModelPreset::QWEN2_05B).build();
    DeviceGraphOrchestrator::Dependencies deps;
    deps.model_ctx = model_ctx;
    
    Qwen2GraphConfig config = buildConfigFromContext(model_ctx);
    DeviceGraphOrchestrator orchestrator(std::move(deps), config);
    
    // Configure as PP stage
    PPStageOrchestratorConfig stage_config;
    stage_config.stage_id = 0;
    stage_config.first_layer = 0;
    stage_config.last_layer = 12;
    stage_config.has_embedding = true;
    stage_config.has_lm_head = false;
    stage_config.primary_device = DeviceId::cpu();
    
    orchestrator.setPPStageConfig(stage_config);
    
    EXPECT_TRUE(orchestrator.isPPStage());
    EXPECT_EQ(orchestrator.ppStageConfig()->first_layer, 0);
    EXPECT_EQ(orchestrator.ppStageConfig()->last_layer, 12);
}

TEST(Test__DeviceGraphOrchestrator_PP, InitializeStateAllocatesPartialKVCache) {
    // ... test that KV cache only covers stage's layers ...
}

TEST(Test__DeviceGraphOrchestrator_PP, ForwardHiddenExecutesPartialGraph) {
    // ... test forward_hidden() with hidden state input ...
}

TEST(Test__DeviceGraphOrchestrator_PP, OutputHiddenReturnsCorrectBuffer) {
    // ... test outputHidden() returns state_.hidden ...
}
```

---

### 5.2.6 Phase 4.2 Summary

| File | Action | LOC | Dependencies |
|------|--------|-----|--------------|
| `src/v2/models/qwen/Qwen2Graph.h` | Modify | +30 | PPStageOrchestratorConfig |
| `src/v2/models/qwen/Qwen2Graph.cpp` | Modify | +100 | - |
| `src/v2/execution/.../DeviceGraphOrchestrator.h` | Modify | +60 | PPStageOrchestratorConfig |
| `src/v2/execution/.../DeviceGraphOrchestrator.cpp` | Modify | +230 | - |
| `tests/v2/unit/orchestrators/Test__DeviceGraphOrchestrator_PP.cpp` | Create | 400 | - |

**Total**: ~820 lines

---

## 6. Phase 4.3: RankOrchestrator PP Support

### 6.3.1 Add: ParallelismMode field and strategy selection

**File**: `src/v2/execution/local_execution/orchestrators/RankOrchestrator.h`  
**Changes**: Accept PipelineConfig, add strategy pattern

```cpp
// Add to Config struct:
struct Config {
    // ... existing fields ...
    
    /// Full pipeline configuration (for PP and TP+PP modes)
    /// If set, parallelism_mode is auto-detected from this config
    std::shared_ptr<PipelineConfig> pipeline_config;
    
    /// Explicit parallelism mode override (AUTO = detect from config)
    ParallelismMode parallelism_mode = ParallelismMode::TENSOR_PARALLEL;
};

// Add to class:
private:
    /// Parallelism mode (TP, PP, or TP+PP)
    ParallelismMode parallelism_mode_ = ParallelismMode::TENSOR_PARALLEL;
    
    /// PP stage orchestrators (only for PP and TP+PP modes)
    std::vector<std::unique_ptr<IStageOrchestrator>> pp_stage_orchestrators_;
    
    /// PP context for activation transfers
    std::unique_ptr<ILocalPPContext> pp_ctx_;

public:
    /**
     * @brief Get parallelism mode
     */
    ParallelismMode parallelismMode() const { return parallelism_mode_; }
    
    /**
     * @brief Check if running in PP mode
     */
    bool hasPipelineParallelism() const {
        return parallelism_mode_ == ParallelismMode::PIPELINE_PARALLEL ||
               parallelism_mode_ == ParallelismMode::TENSOR_PIPELINE_PARALLEL;
    }
```

---

### 6.3.2 Modify: forward() for PP

**File**: `src/v2/execution/local_execution/orchestrators/RankOrchestrator.cpp`  
**Changes**: Add sequential execution path for PP

```cpp
bool RankOrchestrator::forward(const int* tokens, int seq_len)
{
    if (parallelism_mode_ == ParallelismMode::TENSOR_PARALLEL) {
        // Existing TP path: parallel forward across devices
        return forwardTP(tokens, seq_len);
    }
    else if (parallelism_mode_ == ParallelismMode::PIPELINE_PARALLEL ||
             parallelism_mode_ == ParallelismMode::TENSOR_PIPELINE_PARALLEL) {
        // New PP path: sequential forward across stages
        return forwardPP(tokens, seq_len);
    }
    else {
        // Single device: delegate to first runner
        return device_runners_[0]->forward(tokens, seq_len);
    }
}

bool RankOrchestrator::forwardTP(const int* tokens, int seq_len)
{
    // Existing parallel implementation...
    // (Move current forward() logic here)
}

bool RankOrchestrator::forwardPP(const int* tokens, int seq_len)
{
    if (pp_stage_orchestrators_.empty()) {
        LOG_ERROR("RankOrchestrator::forwardPP: No PP stages configured");
        return false;
    }
    
    TensorBase* current_hidden = nullptr;
    
    for (size_t stage_idx = 0; stage_idx < pp_stage_orchestrators_.size(); ++stage_idx) {
        auto& stage = pp_stage_orchestrators_[stage_idx];
        
        if (stage_idx == 0) {
            // First stage: forward from tokens
            current_hidden = stage->forward_tokens(tokens, seq_len);
        } else {
            // Subsequent stages: forward from hidden
            // Transfer activations via PP context
            if (pp_ctx_) {
                pp_ctx_->sendActivations(
                    pp_stage_orchestrators_[stage_idx - 1]->outputHidden(),
                    stage_idx - 1,
                    stage_idx);
            }
            current_hidden = stage->forward_hidden(current_hidden, seq_len);
        }
        
        if (!current_hidden) {
            LOG_ERROR("RankOrchestrator::forwardPP: Stage " << stage_idx << " failed");
            return false;
        }
    }
    
    // Store final output
    if (pp_stage_orchestrators_.back()->isLastStage()) {
        // Last stage output is logits
        combined_logits_ = dynamic_cast<FP32Tensor*>(current_hidden);
    }
    
    return true;
}
```

**Estimated Changes**: +200 lines

---

### 6.3.3 New File: PPStageOrchestrator.h

**Path**: `src/v2/execution/local_execution/orchestrators/PPStageOrchestrator.h`  
**LOC**: ~200 lines

```cpp
/**
 * @file PPStageOrchestrator.h
 * @brief Wrapper that adapts DeviceGraphOrchestrator to IStageOrchestrator
 * 
 * For simple PP (no TP within stages), this wraps a single DeviceGraphOrchestrator.
 * For TP+PP, use RankOrchestratorStage instead.
 */

#pragma once

#include "IStageOrchestrator.h"
#include "DeviceGraphOrchestrator.h"
#include <memory>

namespace llaminar2 {

/**
 * @brief Adapts a single DeviceGraphOrchestrator to IStageOrchestrator
 */
class PPStageOrchestrator : public IStageOrchestrator {
public:
    /**
     * @brief Construct with pre-configured DeviceGraphOrchestrator
     * 
     * The orchestrator must have setPPStageConfig() already called.
     */
    explicit PPStageOrchestrator(
        std::unique_ptr<DeviceGraphOrchestrator> orchestrator,
        const PPStageOrchestratorConfig& config);
    
    // IStageOrchestrator implementation
    const PPStageOrchestratorConfig& stageConfig() const override { return config_; }
    int stageId() const override { return config_.stage_id; }
    bool isFirstStage() const override { return config_.has_embedding; }
    bool isLastStage() const override { return config_.has_lm_head; }
    
    TensorBase* forward_hidden(TensorBase* hidden_input, int seq_len) override;
    TensorBase* forward_tokens(const int* tokens, int seq_len) override;
    
    TensorBase* outputHidden() override;
    IKVCache* kvCache() override;
    void clearCache() override;
    int getPosition() const override;

private:
    std::unique_ptr<DeviceGraphOrchestrator> orchestrator_;
    PPStageOrchestratorConfig config_;
};

/**
 * @brief Adapts RankOrchestrator (TP domain) to IStageOrchestrator
 * 
 * For TP+PP mode where each PP stage is a TP domain.
 */
class RankOrchestratorStage : public IStageOrchestrator {
public:
    explicit RankOrchestratorStage(
        std::unique_ptr<RankOrchestrator> orchestrator,
        const PPStageOrchestratorConfig& config);
    
    // IStageOrchestrator implementation
    // ... similar to PPStageOrchestrator ...

private:
    std::unique_ptr<RankOrchestrator> orchestrator_;
    PPStageOrchestratorConfig config_;
};

} // namespace llaminar2
```

---

### 6.3.4 Modify: Constructor for PP initialization

**File**: `src/v2/execution/local_execution/orchestrators/RankOrchestrator.cpp`  
**Changes**: Initialize PP stages when pipeline_config is provided

```cpp
void RankOrchestrator::initializePPStages()
{
    if (!config_.pipeline_config) {
        return;
    }
    
    const auto& pp_config = *config_.pipeline_config;
    parallelism_mode_ = detectParallelismMode(pp_config);
    
    if (parallelism_mode_ == ParallelismMode::TENSOR_PARALLEL) {
        // TP-only: existing initialization path
        return;
    }
    
    LOG_INFO("RankOrchestrator: Initializing " << pp_config.numStages() 
             << " PP stages, mode=" << parallelismModeToString(parallelism_mode_));
    
    pp_stage_orchestrators_.reserve(pp_config.numStages());
    
    for (const auto& stage : pp_config.pp_stages) {
        const auto* domain = pp_config.getDomain(stage.domain_name);
        if (!domain) {
            throw std::runtime_error("PP stage " + std::to_string(stage.stage_id) +
                                   " references unknown domain: " + stage.domain_name);
        }
        
        auto stage_config = stage.toOrchestratorConfig(domain->primary_device());
        
        if (domain->devices.size() == 1) {
            // Single device: create PPStageOrchestrator
            auto device_orch = createDeviceOrchestratorForStage(stage_config);
            pp_stage_orchestrators_.push_back(
                std::make_unique<PPStageOrchestrator>(std::move(device_orch), stage_config));
        } else {
            // TP domain: create RankOrchestratorStage
            auto tp_orch = createTPOrchestratorForStage(*domain, stage_config);
            pp_stage_orchestrators_.push_back(
                std::make_unique<RankOrchestratorStage>(std::move(tp_orch), stage_config));
        }
    }
    
    // Create PP context for activation transfers
    pp_ctx_ = createLocalPPContext(pp_config);
}
```

**Estimated Changes**: +250 lines

---

### 6.3.5 Test File: Test__RankOrchestrator_PP.cpp

**Path**: `tests/v2/integration/execution/Test__RankOrchestrator_PP.cpp`  
**LOC**: ~600 lines

```cpp
/**
 * Integration tests for RankOrchestrator with Pipeline Parallelism
 */

class RankOrchestratorPPTest : public ::testing::Test {
protected:
    void SetUp() override {
        model_ctx_ = MockModelContextBuilder()
            .usePreset(ModelPreset::QWEN2_05B)
            .build();
    }
    
    std::shared_ptr<IModelContext> model_ctx_;
};

TEST_F(RankOrchestratorPPTest, DetectsTPMode) {
    auto config = PipelineConfig::tensorParallel(24,
        {DeviceId::cpu(), DeviceId::cpu()},  // Two CPU "devices" for testing
        CollectiveBackendType::HOST);
    
    RankOrchestrator::Config orch_config;
    orch_config.devices = {GlobalDeviceAddress::cpu(), GlobalDeviceAddress::cpu()};
    orch_config.pipeline_config = std::make_shared<PipelineConfig>(config);
    
    RankOrchestrator orch(model_ctx_, orch_config);
    
    EXPECT_EQ(orch.parallelismMode(), ParallelismMode::TENSOR_PARALLEL);
    EXPECT_FALSE(orch.hasPipelineParallelism());
}

TEST_F(RankOrchestratorPPTest, DetectsPPMode) {
    auto config = PipelineConfig::pipelineParallel2Stage(24,
        DeviceId::cpu(), 12,
        DeviceId::cpu(),
        CollectiveBackendType::HOST);
    
    RankOrchestrator::Config orch_config;
    orch_config.devices = {GlobalDeviceAddress::cpu(), GlobalDeviceAddress::cpu()};
    orch_config.pipeline_config = std::make_shared<PipelineConfig>(config);
    
    RankOrchestrator orch(model_ctx_, orch_config);
    
    EXPECT_EQ(orch.parallelismMode(), ParallelismMode::PIPELINE_PARALLEL);
    EXPECT_TRUE(orch.hasPipelineParallelism());
}

TEST_F(RankOrchestratorPPTest, ForwardPPExecutesStagesSequentially) {
    // Create 2-stage PP config
    auto config = PipelineConfig::pipelineParallel2Stage(24,
        DeviceId::cpu(), 12,
        DeviceId::cpu(),
        CollectiveBackendType::HOST);
    
    // ... setup orchestrator ...
    
    // Forward pass
    std::vector<int> tokens = {1, 2, 3, 4, 5};
    EXPECT_TRUE(orch.forward(tokens.data(), tokens.size()));
    
    // Verify logits are valid
    const float* logits = orch.logits();
    EXPECT_NE(logits, nullptr);
    
    // Verify not all zeros or NaN
    bool has_non_zero = false;
    for (int i = 0; i < orch.vocab_size(); ++i) {
        EXPECT_FALSE(std::isnan(logits[i]));
        if (logits[i] != 0.0f) has_non_zero = true;
    }
    EXPECT_TRUE(has_non_zero);
}

TEST_F(RankOrchestratorPPTest, PPClearsCacheOnAllStages) {
    // ... test that clear_cache() propagates to all PP stages ...
}

TEST_F(RankOrchestratorPPTest, PPPositionTrackingIsCorrect) {
    // ... test that get_position() returns correct cumulative position ...
}
```

---

### 6.3.6 Phase 4.3 Summary

| File | Action | LOC | Dependencies |
|------|--------|-----|--------------|
| `src/v2/execution/.../RankOrchestrator.h` | Modify | +100 | ParallelismMode.h, IStageOrchestrator.h |
| `src/v2/execution/.../RankOrchestrator.cpp` | Modify | +450 | PPStageOrchestrator.h |
| `src/v2/execution/.../PPStageOrchestrator.h` | Create | 200 | IStageOrchestrator.h |
| `src/v2/execution/.../PPStageOrchestrator.cpp` | Create | 300 | DeviceGraphOrchestrator.h |
| `tests/v2/integration/execution/Test__RankOrchestrator_PP.cpp` | Create | 600 | - |

**Total**: ~1650 lines

---

## 7. Phase 4.4: Factory Function Migration

### 7.4.1 Deprecate: createUnifiedPipelineRunner()

**File**: `src/v2/execution/factory/InferenceRunnerFactory.h`  
**Changes**: Add deprecation notice

```cpp
/**
 * @brief [DEPRECATED] Factory function to create a unified LOCAL PP runner
 * 
 * @deprecated Use RankOrchestrator with PipelineConfig instead:
 * @code
 * RankOrchestrator::Config config;
 * config.pipeline_config = your_pipeline_config;
 * auto runner = std::make_unique<RankOrchestrator>(model_ctx, config);
 * @endcode
 * 
 * This function will be removed in a future release.
 */
[[deprecated("Use RankOrchestrator with pipeline_config instead")]]
std::unique_ptr<IInferenceRunner> createUnifiedPipelineRunner(
    std::shared_ptr<ModelContext> model_ctx,
    std::shared_ptr<PipelineConfig> pipeline_config,
    const InferenceRunnerConfig& config = {});
```

---

### 7.4.2 New Function: createMultiDevicePipelineRunner()

**File**: `src/v2/execution/factory/InferenceRunnerFactory.h`  
**Add**: New factory function using multi-orchestrator pattern

```cpp
/**
 * @brief Create a multi-device runner with full parallelism support
 * 
 * Creates a RankOrchestrator configured for the specified pipeline:
 * - Single device → Direct DeviceGraphOrchestrator
 * - TP only → RankOrchestrator with parallel forward
 * - PP only → RankOrchestrator with sequential stages
 * - TP+PP → RankOrchestrator with nested TP domains per PP stage
 * 
 * @param model_ctx Model context with weights
 * @param pipeline_config Complete pipeline configuration
 * @param config Runner configuration
 * @return IInferenceRunner (concrete type depends on parallelism mode)
 */
std::unique_ptr<IInferenceRunner> createMultiDevicePipelineRunner(
    std::shared_ptr<IModelContext> model_ctx,
    std::shared_ptr<PipelineConfig> pipeline_config,
    const InferenceRunnerConfig& config = {});
```

**Implementation**:

```cpp
std::unique_ptr<IInferenceRunner> createMultiDevicePipelineRunner(
    std::shared_ptr<IModelContext> model_ctx,
    std::shared_ptr<PipelineConfig> pipeline_config,
    const InferenceRunnerConfig& config)
{
    if (!pipeline_config) {
        LOG_ERROR("createMultiDevicePipelineRunner: pipeline_config is null");
        return nullptr;
    }
    
    std::string error;
    if (!pipeline_config->validate(&error)) {
        LOG_ERROR("createMultiDevicePipelineRunner: Invalid config - " << error);
        return nullptr;
    }
    
    ParallelismMode mode = detectParallelismMode(*pipeline_config);
    
    if (mode == ParallelismMode::SINGLE_DEVICE) {
        // Single device: use DeviceGraphOrchestrator directly
        return createInferenceRunner(model_ctx, nullptr,
            pipeline_config->getAllDevices()[0], config);
    }
    
    // Multi-device: create RankOrchestrator
    RankOrchestrator::Config orch_config;
    orch_config.pipeline_config = pipeline_config;
    orch_config.max_seq_len = config.max_seq_len;
    orch_config.batch_size = config.batch_size;
    orch_config.activation_precision = config.activation_precision;
    
    // Extract devices from config
    orch_config.devices.clear();
    for (const auto& device : pipeline_config->getAllDevices()) {
        orch_config.devices.push_back(GlobalDeviceAddress::fromDeviceId(device));
    }
    
    return std::make_unique<RankOrchestrator>(model_ctx, orch_config);
}
```

---

### 7.4.3 Update: ParityTestBase.h

**File**: `tests/v2/integration/parity/ParityTestBase.h`  
**Changes**: Migrate from createUnifiedPipelineRunner

```cpp
// Before (line ~1855):
runner_ = createUnifiedPipelineRunner(model_ctx_, pipeline_config, runner_config);

// After:
runner_ = createMultiDevicePipelineRunner(model_ctx_, pipeline_config, runner_config);
```

**Also update Qwen2ParityTestBase.h** (line ~420):
```cpp
// Before:
// Uses createUnifiedPipelineRunner() for complete production code coverage

// After:
// Uses createMultiDevicePipelineRunner() for unified TP/PP support
```

---

### 7.4.4 Migration Script

Create a sed script for batch migration:

**File**: `scripts/migrate_pp_factories.sh`

```bash
#!/bin/bash
# Migrate createUnifiedPipelineRunner -> createMultiDevicePipelineRunner

find tests/ src/ -name "*.cpp" -o -name "*.h" | while read file; do
    if grep -q "createUnifiedPipelineRunner" "$file"; then
        echo "Migrating: $file"
        sed -i 's/createUnifiedPipelineRunner/createMultiDevicePipelineRunner/g' "$file"
    fi
done
```

---

### 7.4.5 Phase 4.4 Summary

| File | Action | LOC | Dependencies |
|------|--------|-----|--------------|
| `src/v2/execution/factory/InferenceRunnerFactory.h` | Modify | +50 | ParallelismMode.h |
| `src/v2/execution/factory/InferenceRunnerFactory.cpp` | Modify | +100 | RankOrchestrator.h |
| `tests/v2/integration/parity/ParityTestBase.h` | Modify | +10 | - |
| `tests/v2/integration/parity/qwen2/Qwen2ParityTestBase.h` | Modify | +10 | - |
| `scripts/migrate_pp_factories.sh` | Create | 20 | - |

**Total**: ~190 lines

---

## 8. Phase 4.5: Deprecation and Cleanup

### 8.5.1 Remove: buildUnifiedPipelineGraph()

**Files to modify**:

| File | Lines to Remove | Notes |
|------|-----------------|-------|
| `src/v2/models/qwen/Qwen2Graph.h` | ~30 | Remove method declaration |
| `src/v2/models/qwen/Qwen2Graph.cpp` | ~200 | Remove method implementation |
| `tests/v2/unit/models/qwen/Test__Qwen2Graph_PP.cpp` | ~1200 | Remove entire file or migrate tests |

**Migration strategy for tests**:
1. Tests that verify graph structure → Migrate to use `buildPartialForwardGraph()`
2. Tests that verify end-to-end PP → Migrate to `Test__RankOrchestrator_PP.cpp`

---

### 8.5.2 Remove: pp_kv_caches from InferenceState

**File**: `src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h`

```cpp
// Remove from InferenceState struct (lines ~200-207):
// DELETE:
/// Per-device KV caches for Pipeline Parallelism
/// When PP is enabled, each PP stage device has its own KV cache containing
/// only the layers processed by that stage. Key is DeviceId, value is the cache.
/// Only populated when pipeline_config->hasPP() is true.
std::unordered_map<DeviceId, std::unique_ptr<IKVCache>> pp_kv_caches;
```

**Also update**:
- `InferenceState::clear()` - Remove `pp_kv_caches` iteration
- Any code that accesses `state_.pp_kv_caches`

---

### 8.5.3 Remove: Old PP-specific code in DeviceGraphOrchestrator

**File**: `src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.cpp`

Areas to clean up:
1. Remove PP KV cache initialization in `initializeInferenceState()`
2. Remove graph building code that handles multi-device in single graph
3. Remove any `pipeline_config_->hasPP()` branching that's replaced by stage-based approach

**Estimated removal**: ~300 lines

---

### 8.5.4 Remove: createUnifiedPipelineRunner() implementation

**File**: `src/v2/execution/factory/InferenceRunnerFactory.cpp`

```cpp
// Remove lines 1141-1280 (approximately)
// The entire createUnifiedPipelineRunner() function and configureUnifiedPipelineWeightsImpl()
```

---

### 8.5.5 Update: CMakeLists.txt

**File**: `tests/v2/CMakeLists.txt`

```cmake
# Remove test target (around line 5726):
# DELETE:
# Test: Qwen2Graph buildUnifiedPipelineGraph (Pipeline Parallelism graph building)
# add_executable(v2_unit_qwen2graph_pp Test__Qwen2Graph_PP.cpp)
# ...
```

---

### 8.5.6 Documentation Updates

**Files to update**:

| File | Changes |
|------|---------|
| `docs/v2/projects/2026-02/UNIFIED_PP_GRAPH_ARCHITECTURE_PLAN.md` | Mark as superseded |
| `.github/copilot-instructions.md` | Update PP section |
| `README.md` | Update parallelism examples |

---

### 8.5.7 Phase 4.5 Summary

| File | Action | LOC Removed |
|------|--------|-------------|
| `src/v2/models/qwen/Qwen2Graph.h` | Remove buildUnifiedPipelineGraph | -30 |
| `src/v2/models/qwen/Qwen2Graph.cpp` | Remove implementation | -200 |
| `src/v2/execution/.../DeviceGraphOrchestrator.h` | Remove pp_kv_caches | -10 |
| `src/v2/execution/.../DeviceGraphOrchestrator.cpp` | Remove PP graph code | -300 |
| `src/v2/execution/factory/InferenceRunnerFactory.cpp` | Remove old factory | -150 |
| `tests/v2/unit/models/qwen/Test__Qwen2Graph_PP.cpp` | Remove or migrate | -1200 |
| `tests/v2/CMakeLists.txt` | Remove test target | -20 |

**Total removed**: ~1910 lines

---

## 9. Test Plan

### 9.1 TP Regression Tests (Must Not Break)

| Test Suite | File | Purpose |
|------------|------|---------|
| `V2_Unit_RankOrchestrator` | Existing | TP forward, allgather |
| `V2_Integration_LocalTP_*` | Existing | End-to-end TP inference |
| `V2_Integration_Parity_TP_*` | Existing | TP parity vs PyTorch |

**Regression CI check**: All TP tests must pass before and after each PR.

### 9.2 New PP Unit Tests

| Test | File | Coverage |
|------|------|----------|
| ParallelismMode detection | `Test__ParallelismMode.cpp` | Mode detection from config |
| PP stage config | `Test__PPStageConfig.cpp` | Config validation |
| DeviceGraphOrchestrator PP | `Test__DeviceGraphOrchestrator_PP.cpp` | Partial graphs, forward_hidden |
| PPStageOrchestrator | `Test__PPStageOrchestrator.cpp` | IStageOrchestrator interface |

### 9.3 New PP Integration Tests

| Test | File | Coverage |
|------|------|----------|
| MultiDevice PP 2-stage | `Test__RankOrchestrator_PP.cpp` | Sequential stage execution |
| PP activation transfer | `Test__PPActivationTransfer.cpp` | Hidden state passing |
| PP+TP composite | `Test__TPPP_Composite.cpp` | TP domains as PP stages |

### 9.4 New PP Parity Tests

| Test | Model | Configuration |
|------|-------|---------------|
| PP 2-stage CPU | Qwen2-0.5B | Layers 0-12 CPU, 12-24 CPU |
| PP CUDA+CPU | Qwen2-0.5B | Layers 0-12 CUDA, 12-24 CPU |
| TP+PP 2x2 | Qwen2-1.5B | 2 TP devices × 2 PP stages |

### 9.5 Test Execution Order

```
Phase 4.1 tests:
  ctest -R "V2_Unit_ParallelismMode"
  ctest -R "V2_Unit_PPStageConfig"

Phase 4.2 tests:
  ctest -R "V2_Unit_DeviceGraphOrchestrator_PP"
  ctest -R "V2_Unit_.*" (regression)

Phase 4.3 tests:
  ctest -R "V2_Integration_RankOrchestrator_PP"
  ctest -R "V2_Integration_LocalTP" (regression)

Phase 4.4 tests:
  ctest -R "V2_Integration_Parity" (all parity tests)

Phase 4.5 tests:
  ctest --parallel  # Full test suite
```

---

## 10. Risk Assessment

### 10.1 Risk Matrix

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| TP regression | Medium | High | Run TP tests after each change |
| Memory leaks in PP | Medium | Medium | ASAN testing during Phase 4.3 |
| Performance regression | Low | Medium | Benchmark before/after Phase 4.5 |
| Incomplete test migration | Medium | Low | Track test count before/after |
| Build breaks | Low | Low | CI gates on each PR |

### 10.2 Rollback Strategy

Each phase can be rolled back independently:

**Phase 4.1**: Delete new files, revert CMakeLists.txt  
**Phase 4.2**: Revert DeviceGraphOrchestrator changes  
**Phase 4.3**: Revert RankOrchestrator changes  
**Phase 4.4**: Re-enable deprecated function  
**Phase 4.5**: Restore removed code from git history

### 10.3 Performance Considerations

| Aspect | TP | PP | Concern |
|--------|----|----|---------|
| Forward latency | Parallel | Sequential | PP adds inter-stage transfer overhead |
| Memory per device | Sharded | Full (per stage layers) | PP may use more total memory |
| Startup time | One graph build | Multiple graph builds | PP builds N graphs |

**Mitigations**:
- Use PCIeBAR for activation transfer to minimize latency
- Pre-allocate activation transfer buffers
- Cache partial graphs like we cache decode graphs

---

## 11. Migration Checklist

### 11.1 Pre-Implementation

- [ ] Read this plan completely
- [ ] Review current TP implementation (RankOrchestrator.cpp)
- [ ] Review ILocalPPContext interface
- [ ] Identify all tests using createUnifiedPipelineRunner

### 11.2 Phase 4.1 Checklist

- [ ] Create ParallelismMode.h/cpp
- [ ] Create IStageOrchestrator.h
- [ ] Modify PPStageConfig.h
- [ ] Create Test__ParallelismMode.cpp
- [ ] Update CMakeLists.txt
- [ ] All unit tests pass

### 11.3 Phase 4.2 Checklist

- [ ] Add buildPartialForwardGraph() to Qwen2Graph
- [ ] Add forward_hidden() to DeviceGraphOrchestrator
- [ ] Add pp_stage_config_ field
- [ ] Modify initializeInferenceState() for partial KV
- [ ] Create Test__DeviceGraphOrchestrator_PP.cpp
- [ ] TP regression tests pass

### 11.4 Phase 4.3 Checklist

- [ ] Add ParallelismMode to RankOrchestrator
- [ ] Implement forwardPP()
- [ ] Create PPStageOrchestrator.h/cpp
- [ ] Create RankOrchestratorStage
- [ ] Create Test__RankOrchestrator_PP.cpp
- [ ] TP regression tests pass
- [ ] New PP tests pass

### 11.5 Phase 4.4 Checklist

- [ ] Deprecate createUnifiedPipelineRunner()
- [ ] Create createMultiDevicePipelineRunner()
- [ ] Update ParityTestBase.h
- [ ] Update Qwen2ParityTestBase.h
- [ ] Run migration script on test files
- [ ] All parity tests pass

### 11.6 Phase 4.5 Checklist

- [ ] Remove buildUnifiedPipelineGraph()
- [ ] Remove pp_kv_caches
- [ ] Remove createUnifiedPipelineRunner() (after deprecation period)
- [ ] Remove/migrate Test__Qwen2Graph_PP.cpp
- [ ] Update documentation
- [ ] Full test suite passes
- [ ] No memory leaks (ASAN clean)

---

## Appendix A: File Change Summary

### New Files (9 files, ~1500 LOC)

| File | LOC |
|------|-----|
| `src/v2/config/ParallelismMode.h` | 80 |
| `src/v2/config/ParallelismMode.cpp` | 40 |
| `src/v2/execution/.../IStageOrchestrator.h` | 120 |
| `src/v2/execution/.../PPStageOrchestrator.h` | 200 |
| `src/v2/execution/.../PPStageOrchestrator.cpp` | 300 |
| `tests/v2/unit/config/Test__ParallelismMode.cpp` | 150 |
| `tests/v2/unit/orchestrators/Test__DeviceGraphOrchestrator_PP.cpp` | 400 |
| `tests/v2/integration/execution/Test__RankOrchestrator_PP.cpp` | 600 |
| `scripts/migrate_pp_factories.sh` | 20 |

### Modified Files (10 files, ~1200 LOC added)

| File | Lines Added |
|------|-------------|
| `src/v2/config/PPStageConfig.h` | +15 |
| `src/v2/models/qwen/Qwen2Graph.h` | +30 |
| `src/v2/models/qwen/Qwen2Graph.cpp` | +100 |
| `src/v2/execution/.../DeviceGraphOrchestrator.h` | +60 |
| `src/v2/execution/.../DeviceGraphOrchestrator.cpp` | +230 |
| `src/v2/execution/.../RankOrchestrator.h` | +100 |
| `src/v2/execution/.../RankOrchestrator.cpp` | +450 |
| `src/v2/execution/factory/InferenceRunnerFactory.h` | +50 |
| `src/v2/execution/factory/InferenceRunnerFactory.cpp` | +100 |
| `tests/v2/integration/parity/*.h` | +20 |
| `tests/v2/CMakeLists.txt` | +30 |

### Removed Files (Phase 4.5) (~1900 LOC removed)

| File | LOC Removed |
|------|-------------|
| `tests/v2/unit/models/qwen/Test__Qwen2Graph_PP.cpp` | 1200 |
| Various inline removals | 700 |

### Net Change

**Total New Code**: ~2700 LOC  
**Total Removed Code**: ~1900 LOC  
**Net Addition**: ~800 LOC

---

## Appendix B: Interface Contracts

### B.1 IStageOrchestrator Contract

```cpp
// Invariants:
// 1. stageConfig() never returns reference to moved-from object
// 2. forward_hidden() only valid for non-first stages
// 3. forward_tokens() only valid for first stage
// 4. outputHidden() returns nullptr for last stage
// 5. kvCache() returns cache covering [first_layer, last_layer)
// 6. getPosition() returns cumulative position across all tokens processed

// Preconditions:
// - forward_hidden(): hidden_input->numel() >= seq_len * d_model
// - forward_tokens(): tokens points to valid array of seq_len ints

// Postconditions:
// - forward_hidden() returns: non-null tensor on success
// - forward_tokens() returns: non-null tensor on success
// - After clear_cache(): getPosition() == 0, kvCache()->position() == 0
```

### B.2 RankOrchestrator PP Contract

```cpp
// Invariants:
// 1. parallelismMode() reflects actual execution strategy
// 2. hasPipelineParallelism() ⟺ pp_stage_orchestrators_.size() > 0
// 3. logits() returns combined/final logits regardless of mode

// Preconditions for PP mode:
// - pipeline_config must be valid (validate() == true)
// - All stage domains must exist in tp_domains

// Postconditions:
// - After construction with PP config: pp_stage_orchestrators_.size() == numStages()
// - After forward(): logits() returns valid pointer
// - After clear_cache(): all stage KV caches cleared
```

---

## Appendix C: Estimated Timeline

| Phase | Duration | Parallel Work |
|-------|----------|---------------|
| 4.1 | 2-3 days | Can start immediately |
| 4.2 | 3-4 days | After 4.1 complete |
| 4.3 | 4-5 days | After 4.2 complete |
| 4.4 | 2-3 days | Can overlap with 4.3 testing |
| 4.5 | 2-3 days | After 4.3 and 4.4 complete |

**Total estimated time**: 2-3 weeks with buffer for issues

---

*End of Implementation Plan*
