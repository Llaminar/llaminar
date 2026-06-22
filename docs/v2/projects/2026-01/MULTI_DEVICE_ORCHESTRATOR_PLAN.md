# Multi-Device Orchestrator Implementation Plan

**Author**: Copilot  
**Date**: January 25, 2026  
**Status**: ⚠️ Phase 6 In Progress (Weight Sharding Integration)
**Target**: LOCAL TP (intra-rank multi-device tensor parallelism)

## Implementation Status

| Phase | Status | Tests |
|-------|--------|-------|
| Phase 1: GraphOrchestrator Rename | ✅ Complete | All existing tests pass |
| Phase 2: RankOrchestrator Implementation | ✅ Complete | 37/37 unit tests pass |
| Phase 3: ILocalTPContext Wiring | ✅ Complete | 48/48 unit tests pass |
| Phase 4: Test Infrastructure | ✅ Complete | 8/8 integration tests pass |
| Phase 5: Integration and Validation | ✅ Complete | 21/21 factory tests + 11/11 orchestration tests pass |
| Phase 6: Weight Sharding Integration | 🔄 In Progress | See gaps below |

---

## ⚠️ Phase 6: Weight Sharding Integration Gap Analysis

**Research Date**: January 25, 2026

### Current State

The `RankOrchestrator::initializeDeviceRunners()` method correctly sets LOCAL TP parameters:
```cpp
runner_config.local_tp_ctx = tp_ctx_.get();
runner_config.local_tp_device_index = device_idx;
```

### Critical Gap: `createTestableInferenceRunner()` Ignores LOCAL TP Config

**Location**: `src/v2/execution/InferenceRunnerFactory.cpp` lines 738-881

The `createTestableInferenceRunner()` function (used by `RankOrchestrator`) does **NOT** process `config.local_tp_ctx` and `config.local_tp_device_index`. Instead, it hardcodes single-rank configuration:

```cpp
// Lines 789-796 - HARDCODED single-rank, ignores LOCAL TP!
graph_config.head_start = 0;
graph_config.local_n_heads = graph_config.n_heads;
graph_config.local_n_kv_heads = graph_config.n_kv_heads;
graph_config.qkv_column_parallel = false;
graph_config.d_ff_local = graph_config.d_ff;
graph_config.ffn_column_parallel = false;
graph_config.vocab_local = graph_config.vocab_size;
graph_config.lm_head_column_parallel = false;
```

**Compare with**: `createInferenceRunner()` (lines 294-369) which properly handles LOCAL TP when `local_tp_ctx && local_tp_ctx->degree() > 1 && weights_sharded`.

### Required Changes for Phase 6

1. **`createTestableInferenceRunner()` in InferenceRunnerFactory.cpp**:
   - Check `config.local_tp_ctx` and `config.local_tp_device_index`
   - Compute head sharding, FFN sharding, vocab sharding based on device index and weights
   - Set `graph_config.local_tp_ctx` and `graph_config.local_tp_device_idx`
   - Port the LOCAL TP configuration logic from `createInferenceRunner()` (lines 294-369)

2. **Weight Loading via `ILocalTPWeightSharder`**:
   - Currently: All device runners load **full weights** via `model_ctx->getWeight()`
   - Required: Each device loads only its **sharded portion** of column/row-parallel weights
   - Use `ILocalTPWeightSharder::getColumnShard()` for Q, K, V, gate, up projections
   - Use `ILocalTPWeightSharder::getRowShard()` for Wo, FFN down projections

3. **Interface Consideration**:
   - `IModelContext::getWeight()` returns full weights
   - Need either:
     a) `IModelContext::getShardedWeight(name, tp_ctx, device_idx)` method, OR
     b) Shard weights in factory after loading full weights (less memory efficient)

### Why This Matters

Without weight sharding:
- All devices load **identical full weights** → wastes 50%+ memory
- Buffer dimensions don't match weight dimensions → incorrect GEMM output
- No actual parallelism benefit (each device duplicates all work)

### Test Coverage Needed

- [ ] Unit test: `createTestableInferenceRunner` with `local_tp_ctx` sets correct graph_config
- [ ] Unit test: Sharded head counts match device weights
- [ ] Integration test: 2-GPU LOCAL TP loads sharded weights correctly
- [ ] Parity test: 2-GPU LOCAL TP output matches single-GPU output

---

## Executive Summary

This document outlines the implementation plan for **Option A**: creating a new `RankOrchestrator` class that manages multiple `DeviceGraphOrchestrator` instances (renamed from `GraphOrchestrator`) for LOCAL tensor parallelism.

### Key Changes

1. **Rename**: `GraphOrchestrator` → `DeviceGraphOrchestrator` (clarifies single-device scope)
2. **New Class**: `RankOrchestrator` coordinates N `DeviceGraphOrchestrator` instances
3. **Parallel Execution**: Thread pool executes device graphs concurrently
4. **Collective Integration**: `ILocalTPContext` handles allreduce/allgather between devices

### Why Option A?

| Consideration | Option A (New Class) | Option B (Extend GraphOrch) | Option C (Extend MultiDomain) |
|--------------|---------------------|----------------------------|------------------------------|
| Single-device code changes | **None** | Extensive | Moderate |
| Risk to existing functionality | **Low** | High | Medium |
| Code clarity | **Clean separation** | Mixed concerns | Overloaded wrapper |
| Testing complexity | **Isolated tests** | Coupled tests | Wrapper tests |

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Phase 1: GraphOrchestrator Rename](#2-phase-1-graphorchestrator-rename)
3. [Phase 2: RankOrchestrator Implementation](#3-phase-2-multideviceorchestrator-implementation)
4. [Phase 3: ILocalTPContext Wiring](#4-phase-3-ilocalptcontext-wiring)
5. [Phase 4: Test Infrastructure](#5-phase-4-test-infrastructure)
6. [Phase 5: Integration and Validation](#6-phase-5-integration-and-validation)
7. [File Change Summary](#7-file-change-summary)
8. [Risk Assessment](#8-risk-assessment)
9. [Timeline Estimate](#9-timeline-estimate)

---

## 1. Architecture Overview

### 1.1 Current Architecture (Single-Device)

```
┌─────────────────────────────────────────────────────────────┐
│                     IInferenceRunner                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    GraphOrchestrator                         │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ InferenceState (single device_id)                     │   │
│  │   - hidden, logits, Q, K, V (full-size or sharded)   │   │
│  │   - kv_cache (single instance)                       │   │
│  │   - device_id = cuda:0                               │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ DeviceGraphExecutor → Qwen2Graph → ComputeStages           │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

**Limitation**: Single `device_id`, single KV cache, single buffer set. Cannot natively support LOCAL TP where multiple devices must execute in parallel.

### 1.2 Target Architecture (Multi-Device LOCAL TP)

```
┌─────────────────────────────────────────────────────────────┐
│                     IInferenceRunner                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  RankOrchestrator                     │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ ILocalTPContext* tp_ctx                              │    │
│  │   - devices: [cuda:0, cuda:1]                       │    │
│  │   - weights: [0.5, 0.5] or [0.73, 0.27]            │    │
│  │   - backend: NCCL / RCCL / PCIeBAR                  │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌──────────────────────┐  ┌──────────────────────┐         │
│  │DeviceGraphOrchestrator│  │DeviceGraphOrchestrator│        │
│  │     (device 0)        │  │     (device 1)        │        │
│  │  ┌────────────────┐   │  │  ┌────────────────┐   │        │
│  │  │InferenceState  │   │  │  │InferenceState  │   │        │
│  │  │ device=cuda:0  │   │  │  │ device=cuda:1  │   │        │
│  │  │ local_n_heads=7│   │  │  │ local_n_heads=7│   │        │
│  │  │ kv_cache_shard │   │  │  │ kv_cache_shard │   │        │
│  │  └────────────────┘   │  │  └────────────────┘   │        │
│  │  ┌────────────────┐   │  │  ┌────────────────┐   │        │
│  │  │ Qwen2Graph     │   │  │  │ Qwen2Graph     │   │        │
│  │  │ (sharded stages│   │  │  │ (sharded stages│   │        │
│  │  │  + local TP    │   │  │  │  + local TP    │   │        │
│  │  │  allreduce)    │   │  │  │  allreduce)    │   │        │
│  │  └────────────────┘   │  │  └────────────────┘   │        │
│  └──────────────────────┘  └──────────────────────┘         │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ ThreadPool: parallel forward() across devices        │    │
│  │                                                      │    │
│  │  Thread 0 ──→ device_runners_[0]->forward(tokens)   │    │
│  │  Thread 1 ──→ device_runners_[1]->forward(tokens)   │    │
│  │       └───────── barrier (implicit via allreduce) ──┘    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### 1.3 Data Flow for LOCAL TP

```
Input Tokens
     │
     ▼ (replicated to all devices)
┌────────────────────────────────────────────────────────────┐
│  Embedding Layer (each device computes full embedding)     │
└────────────────────────────────────────────────────────────┘
     │
     ▼
┌────────────────────────────────────────────────────────────┐
│  For each transformer layer:                                │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ RMSNorm (replicated on each device)                  │   │
│  └─────────────────────────────────────────────────────┘   │
│                      │                                      │
│                      ▼                                      │
│  ┌────────────────────────┬────────────────────────┐       │
│  │ Device 0: QKV Proj     │ Device 1: QKV Proj     │       │
│  │ (heads 0-6)            │ (heads 7-13)           │       │
│  │ COLUMN_PARALLEL        │ COLUMN_PARALLEL        │       │
│  └────────────────────────┴────────────────────────┘       │
│                      │                                      │
│                      ▼                                      │
│  ┌────────────────────────┬────────────────────────┐       │
│  │ Device 0: Attention    │ Device 1: Attention    │       │
│  │ (local heads + KV)     │ (local heads + KV)     │       │
│  └────────────────────────┴────────────────────────┘       │
│                      │                                      │
│                      ▼                                      │
│  ┌────────────────────────┬────────────────────────┐       │
│  │ Device 0: Wo Proj      │ Device 1: Wo Proj      │       │
│  │ ROW_PARALLEL           │ ROW_PARALLEL           │       │
│  └────────────────────────┴────────────────────────┘       │
│                      │                                      │
│                      ▼                                      │
│  ┌─────────────────────────────────────────────────────┐   │
│  │        LocalTPAllreduceStage (via ILocalTPContext)   │   │
│  │        NCCL/RCCL/PCIeBAR allreduce across devices    │   │
│  └─────────────────────────────────────────────────────┘   │
│                      │                                      │
│                      ▼                                      │
│  ┌─────────────────────────────────────────────────────┐   │
│  │        Residual Add (reconstructed hidden state)     │   │
│  └─────────────────────────────────────────────────────┘   │
│                      │                                      │
│  (... similar for FFN: gate_up parallel → down → allreduce)│
│                                                             │
└────────────────────────────────────────────────────────────┘
     │
     ▼
┌────────────────────────────────────────────────────────────┐
│  LM Head: COLUMN_PARALLEL → AllGather logits               │
└────────────────────────────────────────────────────────────┘
     │
     ▼
Full Logits (vocab_size) on primary device
```

---

## 2. Phase 1: GraphOrchestrator Rename

**Goal**: Rename `GraphOrchestrator` to `DeviceGraphOrchestrator` to clarify it manages a single device.

### 2.1 Files to Rename

| Current Path | New Path |
|--------------|----------|
| `src/v2/execution/GraphOrchestrator.h` | `src/v2/execution/DeviceGraphOrchestrator.h` |
| `src/v2/execution/GraphOrchestrator.cpp` | `src/v2/execution/DeviceGraphOrchestrator.cpp` |
| `tests/v2/unit/Test__GraphOrchestrator.cpp` | `tests/v2/unit/Test__DeviceGraphOrchestrator.cpp` |
| `tests/v2/unit/Test__GraphOrchestratorSnapshots.cpp` | `tests/v2/unit/Test__DeviceGraphOrchestratorSnapshots.cpp` |
| `tests/v2/unit/Test__GraphOrchestratorBufferManagement.cpp` | `tests/v2/unit/Test__DeviceGraphOrchestratorBufferManagement.cpp` |
| `tests/v2/unit/Test__GraphOrchestratorDomainWiring.cpp` | `tests/v2/unit/Test__DeviceGraphOrchestratorDomainWiring.cpp` |
| `tests/v2/unit/Test__GraphOrchestratorPhaseAwareWeights.cpp` | `tests/v2/unit/Test__DeviceGraphOrchestratorPhaseAwareWeights.cpp` |
| `tests/v2/unit/Test__GraphOrchestratorWeightStreaming.cpp` | `tests/v2/unit/Test__DeviceGraphOrchestratorWeightStreaming.cpp` |

### 2.2 Source Code Changes

#### 2.2.1 Header File Changes (DeviceGraphOrchestrator.h)

```cpp
// OLD
class GraphOrchestrator : public IInferenceRunner {
    ...
};

// NEW
class DeviceGraphOrchestrator : public IInferenceRunner {
    ...
};
```

**Changes Required**:
- Class name: `GraphOrchestrator` → `DeviceGraphOrchestrator`
- Header guard: `GRAPH_ORCHESTRATOR_H_` → `DEVICE_GRAPH_ORCHESTRATOR_H_`
- All 5 constructors renamed
- Copy/move operators renamed
- All comments updated

#### 2.2.2 Implementation File Changes (DeviceGraphOrchestrator.cpp)

- Include directive: `#include "GraphOrchestrator.h"` → `#include "DeviceGraphOrchestrator.h"`
- All ~36 member function scope operators: `GraphOrchestrator::` → `DeviceGraphOrchestrator::`

#### 2.2.3 Factory Changes (InferenceRunnerFactory.cpp)

| Line | Current | New |
|------|---------|-----|
| 15 | `#include "GraphOrchestrator.h"` | `#include "DeviceGraphOrchestrator.h"` |
| 497 | `std::make_unique<GraphOrchestrator>(...)` | `std::make_unique<DeviceGraphOrchestrator>(...)` |
| 813 | `std::make_unique<GraphOrchestrator>(...)` | `std::make_unique<DeviceGraphOrchestrator>(...)` |
| Comments | All references to `GraphOrchestrator` | Update to `DeviceGraphOrchestrator` |

#### 2.2.4 Wrapper Changes (MultiDomainOrchestrator.h/.cpp)

| Line | Current | New |
|------|---------|-----|
| 32 | `class GraphOrchestrator;` | `class DeviceGraphOrchestrator;` |
| 216 | `GraphOrchestrator *getInnerOrchestrator() const;` | `DeviceGraphOrchestrator *getInnerOrchestrator() const;` |
| 224 | `dynamic_cast<GraphOrchestrator *>(inner_runner_.get())` | `dynamic_cast<DeviceGraphOrchestrator *>(inner_runner_.get())` |

#### 2.2.5 Interface Documentation (IInferenceRunner.h)

Update comments mentioning `GraphOrchestrator` to `DeviceGraphOrchestrator`.

#### 2.2.6 CMakeLists.txt Changes

```cmake
# OLD
execution/GraphOrchestrator.cpp

# NEW
execution/DeviceGraphOrchestrator.cpp
```

Test target renames:
- `v2_test_graph_orchestrator_snapshots` → `v2_test_device_graph_orchestrator_snapshots`
- `V2_Unit_GraphOrchestratorSnapshots` → `V2_Unit_DeviceGraphOrchestratorSnapshots`
- Similar for all other GraphOrchestrator test targets

### 2.3 Files Requiring Updates (Complete List)

| File | Change Type |
|------|-------------|
| `src/v2/execution/GraphOrchestrator.h` | Rename + class rename |
| `src/v2/execution/GraphOrchestrator.cpp` | Rename + scope operator updates |
| `src/v2/execution/InferenceRunnerFactory.cpp` | Include + instantiations |
| `src/v2/execution/InferenceRunnerFactory.h` | Forward declaration |
| `src/v2/execution/MultiDomainOrchestrator.h` | Forward decl + method return type |
| `src/v2/execution/MultiDomainOrchestrator.cpp` | Include + dynamic_cast |
| `src/v2/interfaces/IInferenceRunner.h` | Comments only |
| `src/v2/execution/Qwen2Graph.h` | Comments only |
| `src/v2/CMakeLists.txt` | Source file path |
| `tests/v2/CMakeLists.txt` | Test targets + labels |
| `tests/v2/unit/Test__GraphOrchestrator.cpp` | Rename + all references |
| `tests/v2/unit/Test__GraphOrchestratorSnapshots.cpp` | Rename + all references |
| `tests/v2/unit/Test__GraphOrchestratorBufferManagement.cpp` | Rename + all references |
| `tests/v2/unit/Test__GraphOrchestratorDomainWiring.cpp` | Rename + all references |
| `tests/v2/unit/Test__GraphOrchestratorPhaseAwareWeights.cpp` | Rename + all references |
| `tests/v2/unit/Test__GraphOrchestratorWeightStreaming.cpp` | Rename + all references |
| `tests/v2/utils/MockGraphOrchestrator.h` | Class rename → `MockDeviceGraphOrchestrator` |
| `tests/v2/integration/parity/*.cpp` | Comments/includes if present |

### 2.4 Verification

After rename:
```bash
# Build
cmake -B build_v2 -S src/v2 -DCMAKE_BUILD_TYPE=Debug
cmake --build build_v2 --parallel

# Verify no compilation errors
# Run all unit tests
ctest --test-dir build_v2 -R "^V2_Unit_" --output-on-failure --parallel
```

---

## 3. Phase 2: RankOrchestrator Implementation

**Goal**: Create the new multi-device orchestration class.

### 3.1 Class Definition

**Location**: `src/v2/execution/RankOrchestrator.h`

```cpp
#pragma once

#include "interfaces/IInferenceRunner.h"
#include "collectives/ILocalTPContext.h"
#include <memory>
#include <vector>
#include <thread>
#include <future>

namespace llaminar2 {

class DeviceGraphOrchestrator;
class IModelContext;

/**
 * @brief Coordinates multiple DeviceGraphOrchestrator instances for LOCAL TP.
 *
 * RankOrchestrator manages tensor-parallel inference across multiple
 * devices within a single MPI rank. It:
 *   1. Creates N DeviceGraphOrchestrator instances (one per LOCAL TP device)
 *   2. Executes forward() in parallel across devices via thread pool
 *   3. Coordinates collective operations via ILocalTPContext
 *   4. Returns combined logits from the primary device
 *
 * Thread model:
 *   - forward() spawns parallel tasks, one per device
 *   - Each task calls its device runner's forward()
 *   - LocalTPAllreduceStage stages use ILocalTPContext for synchronization
 *   - AllGather combines logits at the end
 */
class RankOrchestrator : public IInferenceRunner {
public:
    /**
     * @brief Configuration for RankOrchestrator.
     */
    struct Config {
        std::vector<GlobalDeviceAddress> devices;    ///< Participating devices
        std::vector<float> weights;                  ///< Proportional weights (sum=1.0)
        CollectiveBackendType backend = CollectiveBackendType::AUTO;
        size_t max_seq_len = 4096;
        int batch_size = 1;
        ActivationPrecision activation_precision = ActivationPrecision::FP32;
        FusedAttentionBackend attention_backend = FusedAttentionBackend::JIT;
        float kv_cache_scale = 1.0f;
        bool use_mapped_memory = false;
    };

    /**
     * @brief Construct RankOrchestrator with ILocalTPContext.
     * @param model_ctx Shared model context (weights, metadata)
     * @param tp_ctx LOCAL TP context for collective operations
     * @param config Configuration for all device runners
     */
    RankOrchestrator(
        std::shared_ptr<IModelContext> model_ctx,
        std::unique_ptr<ILocalTPContext> tp_ctx,
        const Config& config);

    ~RankOrchestrator() override;

    // Disable copy
    RankOrchestrator(const RankOrchestrator&) = delete;
    RankOrchestrator& operator=(const RankOrchestrator&) = delete;

    // Allow move
    RankOrchestrator(RankOrchestrator&&) = default;
    RankOrchestrator& operator=(RankOrchestrator&&) = default;

    // ==================== IInferenceRunner Interface ====================

    /// @brief Execute forward pass across all devices in parallel.
    bool forward(const int* tokens, int seq_len) override;

    /// @brief Get logits from primary device (after AllGather if vocab-sharded).
    const float* logits() const override;

    /// @brief Clear KV cache on ALL devices.
    void clear_cache() override;

    /// @brief Get position (synced across devices).
    int get_position() const override;

    /// @brief Get vocabulary size.
    int vocab_size() const override;

    /// @brief Get model architecture name.
    const char* architecture() const override;

    /// @brief Returns ExecutionPath::GRAPH (multi-device graph execution).
    ExecutionPath executionPath() const override;

    // ==================== Snapshot Support ====================

    void enableSnapshotCapture(const std::string& dir = "") override;
    void disableSnapshotCapture() override;
    void clearSnapshots() override;
    const float* getSnapshot(const std::string& key, size_t& size) const override;
    std::vector<std::string> getSnapshotKeys() const override;

    // ==================== Batch Interface ====================

    bool forward_batch(const std::vector<std::vector<int>>& token_batches) override;
    const float* getLogits(int seq_idx = 0) const override;
    int batch_size() const override;
    int padded_seq_len() const override;
    const std::vector<int>& sequence_lengths() const override;

    // ==================== Profiling ====================

    GraphExecutorStats* executorStats() override;
    void resetExecutorStats() override;

    // ==================== Multi-Device Specific ====================

    /// @brief Get number of devices.
    int device_count() const { return static_cast<int>(device_runners_.size()); }

    /// @brief Get ILocalTPContext.
    ILocalTPContext* localTPContext() const { return tp_ctx_.get(); }

    /// @brief Get device runner at index.
    DeviceGraphOrchestrator* deviceRunner(int idx);
    const DeviceGraphOrchestrator* deviceRunner(int idx) const;

private:
    void initializeDeviceRunners(const Config& config);
    bool executeParallelForward(const int* tokens, int seq_len);

    std::shared_ptr<IModelContext> model_ctx_;
    std::unique_ptr<ILocalTPContext> tp_ctx_;
    std::vector<std::unique_ptr<DeviceGraphOrchestrator>> device_runners_;
    
    // Thread pool for parallel execution
    std::vector<std::thread> thread_pool_;
    
    // Aggregated state
    int vocab_size_ = 0;
    int position_ = 0;
    int batch_size_ = 1;
    std::vector<int> sequence_lengths_;
    
    // Combined logits (after AllGather)
    std::unique_ptr<FP32Tensor> combined_logits_;
    
    // Aggregated profiling stats
    mutable GraphExecutorStats aggregated_stats_;
};

} // namespace llaminar2
```

### 3.2 Implementation Outline

**Location**: `src/v2/execution/RankOrchestrator.cpp`

```cpp
#include "RankOrchestrator.h"
#include "DeviceGraphOrchestrator.h"
#include "InferenceRunnerFactory.h"
#include <execution>

namespace llaminar2 {

RankOrchestrator::RankOrchestrator(
    std::shared_ptr<IModelContext> model_ctx,
    std::unique_ptr<ILocalTPContext> tp_ctx,
    const Config& config)
    : model_ctx_(std::move(model_ctx))
    , tp_ctx_(std::move(tp_ctx))
{
    LLAMINAR_ASSERT_NOT_NULL(model_ctx_.get(), "model_ctx");
    LLAMINAR_ASSERT_NOT_NULL(tp_ctx_.get(), "tp_ctx");
    LLAMINAR_ASSERT(tp_ctx_->degree() >= 2, 
        "RankOrchestrator requires at least 2 devices");
    
    initializeDeviceRunners(config);
    
    LOG_INFO("[RankOrchestrator] Initialized with " 
             << device_runners_.size() << " devices");
}

void RankOrchestrator::initializeDeviceRunners(const Config& config)
{
    const int tp_degree = tp_ctx_->degree();
    device_runners_.reserve(tp_degree);
    
    for (int i = 0; i < tp_degree; ++i) {
        const auto& device = tp_ctx_->deviceAt(i);
        
        // Configure per-device InferenceRunnerConfig
        InferenceRunnerConfig runner_config;
        runner_config.max_seq_len = config.max_seq_len;
        runner_config.batch_size = config.batch_size;
        runner_config.activation_precision = config.activation_precision;
        runner_config.fused_attention_backend = config.attention_backend;
        runner_config.kv_cache_scale = config.kv_cache_scale;
        runner_config.use_mapped_memory = config.use_mapped_memory;
        
        // LOCAL TP configuration
        runner_config.local_tp_ctx = tp_ctx_.get();
        runner_config.local_tp_device_index = i;
        
        // Create DeviceGraphOrchestrator for this device
        auto runner = createInferenceRunner(
            model_ctx_,
            nullptr,  // No MPI context for LOCAL TP
            DeviceId::fromGlobalAddress(device),
            runner_config
        );
        
        // Dynamic cast to DeviceGraphOrchestrator
        auto* device_orch = dynamic_cast<DeviceGraphOrchestrator*>(runner.release());
        LLAMINAR_ASSERT_NOT_NULL(device_orch, 
            "Failed to create DeviceGraphOrchestrator for device " + std::to_string(i));
        
        device_runners_.emplace_back(device_orch);
    }
    
    vocab_size_ = device_runners_[0]->vocab_size();
    
    // Allocate combined logits buffer (full vocab)
    combined_logits_ = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(config.batch_size), 
                           static_cast<size_t>(vocab_size_)},
        DeviceId::cpu());
}

bool RankOrchestrator::forward(const int* tokens, int seq_len)
{
    return executeParallelForward(tokens, seq_len);
}

bool RankOrchestrator::executeParallelForward(const int* tokens, int seq_len)
{
    const int num_devices = device_runners_.size();
    
    // Launch parallel forward on each device
    std::vector<std::future<bool>> futures;
    futures.reserve(num_devices);
    
    for (int i = 0; i < num_devices; ++i) {
        futures.push_back(std::async(std::launch::async, [&, i]() {
            return device_runners_[i]->forward(tokens, seq_len);
        }));
    }
    
    // Wait for all devices to complete
    bool all_success = true;
    for (auto& future : futures) {
        all_success = all_success && future.get();
    }
    
    if (!all_success) {
        LOG_ERROR("[RankOrchestrator] One or more devices failed forward()");
        return false;
    }
    
    // Update position (should be synced across devices)
    position_ = device_runners_[0]->get_position();
    
    return true;
}

const float* RankOrchestrator::logits() const
{
    // If vocab is column-parallel, AllGather was already done in the graph
    // Return logits from primary device
    return device_runners_[0]->logits();
}

void RankOrchestrator::clear_cache()
{
    for (auto& runner : device_runners_) {
        runner->clear_cache();
    }
    position_ = 0;
}

int RankOrchestrator::get_position() const
{
    return position_;
}

int RankOrchestrator::vocab_size() const
{
    return vocab_size_;
}

const char* RankOrchestrator::architecture() const
{
    return device_runners_[0]->architecture();
}

ExecutionPath RankOrchestrator::executionPath() const
{
    return ExecutionPath::GRAPH;
}

// ==================== Snapshot Support ====================

void RankOrchestrator::enableSnapshotCapture(const std::string& dir)
{
    for (auto& runner : device_runners_) {
        runner->enableSnapshotCapture(dir);
    }
}

void RankOrchestrator::disableSnapshotCapture()
{
    for (auto& runner : device_runners_) {
        runner->disableSnapshotCapture();
    }
}

void RankOrchestrator::clearSnapshots()
{
    for (auto& runner : device_runners_) {
        runner->clearSnapshots();
    }
}

const float* RankOrchestrator::getSnapshot(const std::string& key, size_t& size) const
{
    // Route to appropriate device based on key
    // For now, return from primary device
    return device_runners_[0]->getSnapshot(key, size);
}

std::vector<std::string> RankOrchestrator::getSnapshotKeys() const
{
    // Merge keys from all devices (deduplicate)
    std::set<std::string> all_keys;
    for (const auto& runner : device_runners_) {
        auto keys = runner->getSnapshotKeys();
        all_keys.insert(keys.begin(), keys.end());
    }
    return std::vector<std::string>(all_keys.begin(), all_keys.end());
}

// ==================== Batch Interface ====================

bool RankOrchestrator::forward_batch(const std::vector<std::vector<int>>& token_batches)
{
    // Parallel batched forward across devices
    const int num_devices = device_runners_.size();
    std::vector<std::future<bool>> futures;
    
    for (int i = 0; i < num_devices; ++i) {
        futures.push_back(std::async(std::launch::async, [&, i]() {
            return device_runners_[i]->forward_batch(token_batches);
        }));
    }
    
    bool all_success = true;
    for (auto& future : futures) {
        all_success = all_success && future.get();
    }
    
    return all_success;
}

const float* RankOrchestrator::getLogits(int seq_idx) const
{
    return device_runners_[0]->getLogits(seq_idx);
}

int RankOrchestrator::batch_size() const
{
    return batch_size_;
}

int RankOrchestrator::padded_seq_len() const
{
    return device_runners_[0]->padded_seq_len();
}

const std::vector<int>& RankOrchestrator::sequence_lengths() const
{
    return sequence_lengths_;
}

// ==================== Profiling ====================

GraphExecutorStats* RankOrchestrator::executorStats()
{
    // Aggregate stats from all devices
    aggregated_stats_ = GraphExecutorStats{};
    for (auto& runner : device_runners_) {
        auto* stats = runner->executorStats();
        if (stats) {
            // Sum up timing stats
            aggregated_stats_.total_execution_time_ns += stats->total_execution_time_ns;
            // ... aggregate other fields ...
        }
    }
    return &aggregated_stats_;
}

void RankOrchestrator::resetExecutorStats()
{
    for (auto& runner : device_runners_) {
        runner->resetExecutorStats();
    }
    aggregated_stats_ = GraphExecutorStats{};
}

// ==================== Multi-Device Specific ====================

DeviceGraphOrchestrator* RankOrchestrator::deviceRunner(int idx)
{
    LLAMINAR_ASSERT(idx >= 0 && idx < device_runners_.size(), 
        "Device index out of range");
    return device_runners_[idx].get();
}

const DeviceGraphOrchestrator* RankOrchestrator::deviceRunner(int idx) const
{
    LLAMINAR_ASSERT(idx >= 0 && idx < device_runners_.size(), 
        "Device index out of range");
    return device_runners_[idx].get();
}

} // namespace llaminar2
```

### 3.3 Factory Integration

Add factory function in `InferenceRunnerFactory.h`:

```cpp
/**
 * @brief Create RankOrchestrator for LOCAL TP.
 * @param model_ctx Shared model context
 * @param tp_ctx LOCAL TP context (takes ownership)
 * @param config Multi-device configuration
 * @return Unique pointer to RankOrchestrator as IInferenceRunner
 */
std::unique_ptr<IInferenceRunner> createRankOrchestrator(
    std::shared_ptr<IModelContext> model_ctx,
    std::unique_ptr<ILocalTPContext> tp_ctx,
    const RankOrchestrator::Config& config);
```

### 3.4 CMakeLists.txt Updates

```cmake
# Add to llaminar2_core sources
execution/RankOrchestrator.cpp
```

---

## 4. Phase 3: ILocalTPContext Wiring

**Goal**: Wire `ILocalTPContext` collective operations to real backends.

### 4.1 Current State

The `LocalTPContext` implementation has **placeholder collectives**:
- `allreduce()` logs warning and returns true
- `allgather()` logs warning and returns true
- `reduceScatter()` logs warning and returns true

### 4.2 Required Changes

#### 4.2.1 Wire to NCCLBackend Multi-GPU Operations

**Location**: `src/v2/collectives/LocalTPContext.cpp`

```cpp
bool LocalTPContext::allreduce(TensorBase* tensor)
{
    if (degree() == 1) return true;  // Single device no-op
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Get backend based on device types
    ICollectiveBackend* backend = getBackendForDevices();
    LLAMINAR_ASSERT_NOT_NULL(backend, "collective backend");
    
    // Build per-device buffer list
    std::vector<void*> buffers(degree());
    for (int i = 0; i < degree(); ++i) {
        // Ensure tensor is on device i
        tensor->ensureOnDevice(DeviceId::fromGlobalAddress(devices_[i]));
        buffers[i] = tensor->mutable_device_data(devices_[i]);
    }
    
    // Execute multi-GPU allreduce
    return backend->allreduceMulti(
        buffers, 
        tensor->numel(),
        toCollectiveDataType(tensor->dtype()),
        CollectiveOp::SUM);
}
```

#### 4.2.2 Backend Selection Logic

```cpp
ICollectiveBackend* LocalTPContext::getBackendForDevices()
{
    // Check cached backend
    if (backend_) return backend_.get();
    
    // Determine backend based on device types
    bool all_cuda = std::all_of(devices_.begin(), devices_.end(),
        [](const GlobalDeviceAddress& d) { return d.type == DeviceType::CUDA; });
    bool all_rocm = std::all_of(devices_.begin(), devices_.end(),
        [](const GlobalDeviceAddress& d) { return d.type == DeviceType::ROCm; });
    
    if (all_cuda && backend_type_ != CollectiveBackendType::PCIE_BAR) {
        backend_ = createNCCLBackend(devices_);
    } else if (all_rocm && backend_type_ != CollectiveBackendType::PCIE_BAR) {
        backend_ = createRCCLBackend(devices_);
    } else {
        backend_ = createPCIeBarBackend(devices_);
    }
    
    return backend_.get();
}
```

### 4.3 Tensor Device Buffer Management

The tricky part is that `TensorBase` has **single-device coherence**, but for LOCAL TP we need **per-device buffers**. Two approaches:

#### Option A: Per-Device Tensors (Simpler)

Each `DeviceGraphOrchestrator` owns its own tensors. The `LocalTPAllreduceStage` operates on tensors that are already on the correct device.

**Advantage**: No changes to `TensorBase` coherence model.  
**Disadvantage**: Requires tensor copies between stages on different devices.

#### Option B: Multi-Device Tensors (More Complex)

Extend `TensorBase` to support multiple device buffers:
```cpp
std::unordered_map<DeviceId, void*> device_buffers_;
```

**Advantage**: Zero-copy between devices.  
**Disadvantage**: Significant changes to coherence model.

**Recommendation**: Start with **Option A** (simpler) and optimize later if needed.

---

## 5. Phase 4: Test Infrastructure

**Goal**: Create comprehensive tests for RankOrchestrator and real LOCAL TP.

### 5.1 Unit Tests

**Location**: `tests/v2/unit/Test__RankOrchestrator.cpp`

```cpp
/**
 * @file Test__RankOrchestrator.cpp
 * @brief Unit tests for RankOrchestrator
 */

#include <gtest/gtest.h>
#include "execution/local_execution/orchestrators/RankOrchestrator.h"
#include "utils/MockModelContext.h"
#include "utils/TestTensorFactory.h"

namespace llaminar2::test {

class Test__RankOrchestrator : public ::testing::Test {
protected:
    void SetUp() override {
        // Create mock model context
        model_ctx_ = MockModelContext::Builder()
            .withVocabSize(32000)
            .withHiddenDim(896)
            .withNumLayers(24)
            .withNumHeads(14)
            .withNumKVHeads(2)
            .withFFNDim(4864)
            .build();
    }
    
    std::shared_ptr<IModelContext> model_ctx_;
};

TEST_F(Test__RankOrchestrator, RejectsSingleDevice)
{
    auto tp_ctx = createLocalTPContext(
        {GlobalDeviceAddress::cpu()},
        {1.0f},
        CollectiveBackendType::HOST);
    
    RankOrchestrator::Config config;
    config.devices = tp_ctx->devices();
    
    EXPECT_THROW(
        RankOrchestrator(model_ctx_, std::move(tp_ctx), config),
        std::invalid_argument);
}

TEST_F(Test__RankOrchestrator, InitializesWithTwoDevices)
{
    auto tp_ctx = createLocalTPContext(
        {GlobalDeviceAddress::cpu(), GlobalDeviceAddress::cpu()},  // Use CPU for unit test
        {0.5f, 0.5f},
        CollectiveBackendType::HOST);
    
    RankOrchestrator::Config config;
    config.devices = tp_ctx->devices();
    
    RankOrchestrator orchestrator(model_ctx_, std::move(tp_ctx), config);
    
    EXPECT_EQ(orchestrator.device_count(), 2);
    EXPECT_NE(orchestrator.deviceRunner(0), nullptr);
    EXPECT_NE(orchestrator.deviceRunner(1), nullptr);
}

TEST_F(Test__RankOrchestrator, ClearCacheAffectsAllDevices)
{
    // ... test that clear_cache() is called on all device runners
}

} // namespace llaminar2::test
```

### 5.2 Integration Tests

**Location**: `tests/v2/integration/Test__LocalTPMultiDevice.cpp`

```cpp
/**
 * @file Test__LocalTPMultiDevice.cpp
 * @brief Integration tests for real LOCAL TP with RankOrchestrator
 */

class Test__LocalTPMultiDevice : public ::testing::Test {
protected:
    void SetUp() override {
        auto& dm = DeviceManager::instance();
        dm.initialize(-1);
        
        cuda_count_ = dm.cuda_device_count();
        rocm_count_ = dm.rocm_device_count();
    }
    
    bool hasTwoGPUs() const {
        return cuda_count_ >= 2 || rocm_count_ >= 2 || 
               (cuda_count_ >= 1 && rocm_count_ >= 1);
    }
    
    int cuda_count_ = 0;
    int rocm_count_ = 0;
};

TEST_F(Test__LocalTPMultiDevice, CUDA_TwoGPU_ForwardProducesSameLogits)
{
    if (cuda_count_ < 2) GTEST_SKIP() << "Requires 2 CUDA GPUs";
    
    // Create LOCAL TP context
    auto tp_ctx = createLocalTPContext(
        {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)},
        {0.5f, 0.5f},
        CollectiveBackendType::NCCL);
    
    // Load real model
    auto model_ctx = loadModel("models/qwen2.5-0.5b-instruct-q4_0.gguf");
    
    // Create multi-device orchestrator
    RankOrchestrator::Config config;
    config.devices = tp_ctx->devices();
    config.max_seq_len = 512;
    
    RankOrchestrator multi_orch(model_ctx, std::move(tp_ctx), config);
    
    // Create single-device baseline
    auto single_runner = createInferenceRunner(
        model_ctx, nullptr, DeviceId::cuda(0), {});
    
    // Compare outputs
    std::vector<int> tokens = {1, 2, 3, 4, 5};
    
    ASSERT_TRUE(multi_orch.forward(tokens.data(), tokens.size()));
    ASSERT_TRUE(single_runner->forward(tokens.data(), tokens.size()));
    
    const float* multi_logits = multi_orch.logits();
    const float* single_logits = single_runner->logits();
    
    // Verify logits are close
    float mse = TestTensorFactory::computeMSE(
        multi_logits, single_logits, model_ctx->n_vocab());
    EXPECT_LT(mse, 1e-4f) << "LOCAL TP logits diverged from single-device";
}
```

### 5.3 Parity Tests

**Location**: `tests/v2/integration/parity/Test__Qwen2_LocalTP_NCCL_vs_PyTorch.cpp`

Use the existing `ParityTestBase` infrastructure with modifications:

```cpp
class Test__Qwen2_LocalTP_NCCL_vs_PyTorch : public Qwen2ParityTestBase {
protected:
    GlobalDeviceAddress getDevice() override {
        return GlobalDeviceAddress::cuda(0);  // Primary device
    }
    
    std::string getBackendName() const override {
        return "LOCAL_TP_NCCL";
    }
    
    WeightStrategy getWeightStrategy() const override {
        return WeightStrategy::TENSOR_PARALLEL;
    }
    
    std::unique_ptr<IInferenceRunner> createRunner() override {
        // Create LOCAL TP runner instead of single-device
        auto tp_ctx = createLocalTPContext(
            {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)},
            {0.5f, 0.5f},
            CollectiveBackendType::NCCL);
        
        RankOrchestrator::Config config;
        config.devices = tp_ctx->devices();
        
        return std::make_unique<RankOrchestrator>(
            model_ctx_, std::move(tp_ctx), config);
    }
};

REGISTER_PARITY_TESTS(Test__Qwen2_LocalTP_NCCL_vs_PyTorch)
```

---

## 6. Phase 5: Integration and Validation

### 6.1 OrchestrationRunner Integration

Update `OrchestrationRunner` to use `RankOrchestrator` when LOCAL TP is configured:

**Location**: `src/v2/execution/runner/OrchestrationRunner.cpp`

```cpp
std::unique_ptr<IInferenceRunner> OrchestrationRunner::createRunner()
{
    if (hasLocalTP()) {
        return createRankOrchestrator(
            model_ctx_,
            std::move(local_tp_ctx_),
            buildMultiDeviceConfig());
    }
    
    // Existing single-device path
    return createInferenceRunner(model_ctx_, mpi_ctx_, device_, config_);
}
```

### 6.2 CLI Integration

The existing CLI flags should work with `RankOrchestrator`:

```bash
# 2-way LOCAL TP with NCCL
./build_v2_release/llaminar2 \
  --tp 2 --tp-scope local \
  -m models/qwen2.5-0.5b-instruct-q4_0.gguf \
  -p "Hello, world!" -n 50

# Mixed CUDA + ROCm with PCIeBAR
./build_v2_release/llaminar2 \
  --tp-devices "cuda:0,rocm:0" \
  --tp-weights "0.73,0.27" \
  -m models/llama-7b.gguf \
  -p "Test prompt"
```

### 6.3 Validation Checklist

- [ ] **Rename verification**: All unit tests pass after GraphOrchestrator rename
- [ ] **Single-device parity**: RankOrchestrator with 1 device matches baseline
- [ ] **Two-device parity**: 2-GPU LOCAL TP matches single-device output
- [ ] **Proportional TP**: Unequal weights (0.73/0.27) produce correct results
- [ ] **Cross-vendor**: CUDA + ROCm LOCAL TP works via PCIeBAR
- [ ] **Benchmarks**: Performance scales with additional devices
- [ ] **Memory**: Each device uses ~1/N of baseline memory for sharded buffers

---

## 7. File Change Summary

### 7.1 Phase 1: Rename (17+ files)

| File | Change |
|------|--------|
| `src/v2/execution/GraphOrchestrator.h` | **Rename** → `DeviceGraphOrchestrator.h`, class rename |
| `src/v2/execution/GraphOrchestrator.cpp` | **Rename** → `DeviceGraphOrchestrator.cpp`, scope operators |
| `src/v2/execution/InferenceRunnerFactory.cpp` | Include, instantiations |
| `src/v2/execution/InferenceRunnerFactory.h` | Forward declaration |
| `src/v2/execution/MultiDomainOrchestrator.h` | Forward decl, return type |
| `src/v2/execution/MultiDomainOrchestrator.cpp` | Include, dynamic_cast |
| `src/v2/interfaces/IInferenceRunner.h` | Comments |
| `src/v2/CMakeLists.txt` | Source path |
| `tests/v2/CMakeLists.txt` | Test targets |
| `tests/v2/unit/Test__GraphOrchestrator*.cpp` | **Rename** (6 files), all references |
| `tests/v2/utils/MockGraphOrchestrator.h` | Class rename |

### 7.2 Phase 2: New Files (3 files)

| File | Description |
|------|-------------|
| `src/v2/execution/RankOrchestrator.h` | Class definition |
| `src/v2/execution/RankOrchestrator.cpp` | Implementation |
| `tests/v2/unit/Test__RankOrchestrator.cpp` | Unit tests |

### 7.3 Phase 3: ILocalTPContext Wiring (2-3 files)

| File | Change |
|------|--------|
| `src/v2/collectives/LocalTPContext.cpp` | Wire to real backends |
| `src/v2/collectives/LocalTPContext.h` | Add backend member |

### 7.4 Phase 4: Test Files (3-5 files)

| File | Description |
|------|-------------|
| `tests/v2/integration/Test__LocalTPMultiDevice.cpp` | Integration tests |
| `tests/v2/integration/parity/Test__Qwen2_LocalTP_NCCL_vs_PyTorch.cpp` | NCCL parity |
| `tests/v2/integration/parity/Test__Qwen2_LocalTP_RCCL_vs_PyTorch.cpp` | RCCL parity |
| `tests/v2/integration/parity/Test__Qwen2_LocalTP_PCIeBAR_vs_PyTorch.cpp` | PCIeBAR parity |

### 7.5 Phase 5: Integration (2 files)

| File | Change |
|------|--------|
| `src/v2/execution/runner/OrchestrationRunner.cpp` | Use RankOrchestrator |
| `src/v2/execution/InferenceRunnerFactory.cpp` | Add factory function |

---

## 8. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Rename breaks compilation | Low | Medium | Comprehensive grep + sed script, CI verification |
| Thread synchronization issues | Medium | High | Extensive testing, use std::future for simplicity |
| NCCL initialization failures | Medium | Medium | Graceful fallback to PCIeBAR |
| Memory allocation failures | Low | High | Pre-check device memory, fail fast |
| Numerical divergence | Medium | Medium | Extensive parity tests with relaxed tolerances |
| Performance regression | Low | Medium | Benchmark suite, compare against single-device |

---

## 9. Timeline Estimate

| Phase | Duration | Depends On |
|-------|----------|------------|
| Phase 1: Rename | 2-3 hours | - |
| Phase 2: RankOrchestrator | 4-6 hours | Phase 1 |
| Phase 3: ILocalTPContext Wiring | 3-4 hours | Phase 2 |
| Phase 4: Test Infrastructure | 4-5 hours | Phase 2, 3 |
| Phase 5: Integration | 2-3 hours | Phase 2, 3 |
| **Total** | **15-21 hours** | |

---

## Appendix A: Buffer Sharding Reference

### Buffers that are SHARDED (local per device)

| Buffer | Full Size | Local Size with TP=N |
|--------|-----------|----------------------|
| Q, Q_rope | `[seq, n_heads * head_dim]` | `[seq, local_n_heads * head_dim]` |
| K, K_rope, V, V_dequant | `[seq, n_kv_heads * head_dim]` | `[seq, local_n_kv_heads * head_dim]` |
| attn_output | `[seq, n_heads * head_dim]` | `[seq, local_n_heads * head_dim]` |
| gate, up | `[seq, d_ff]` | `[seq, d_ff / N]` |
| KV Cache | `[layers, 2, max_seq, n_kv_heads, head_dim]` | `[layers, 2, max_seq, local_n_kv_heads, head_dim]` |
| logits_local | `[batch, vocab_size]` | `[batch, vocab_size / N]` |

### Buffers that are REPLICATED (full copy on each device)

| Buffer | Size |
|--------|------|
| hidden | `[seq, hidden_dim]` |
| normalized | `[seq, hidden_dim]` |
| residual | `[seq, hidden_dim]` |
| attn_proj | `[seq, hidden_dim]` |
| ffn_output | `[seq, hidden_dim]` |
| logits (after AllGather) | `[batch, vocab_size]` |

---

## Appendix B: Collective Operation Points

| Stage | Collective | Purpose |
|-------|------------|---------|
| After Wo projection | **AllReduce** | Sum partial attention outputs |
| After FFN down projection | **AllReduce** | Sum partial FFN outputs |
| After LM head | **AllGather** | Gather vocab shards into full logits |

---

*End of Document*
