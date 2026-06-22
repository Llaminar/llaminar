# LayerExecutor Migration Plan: Pipeline Replacement Architecture

**Author:** David Sanftenberg  
**Date:** December 2025  
**Status:** DRAFT - Under Review  
**Target:** Llaminar V2.1

---

## Executive Summary

This document outlines a phased migration strategy to replace the imperative `PipelineBase`/`Qwen2Pipeline` system with a declarative `LayerExecutor`-based architecture. The goal is to achieve a purely declarative compute graph with metadata-driven execution while maintaining full feature parity.

### Key Objectives

1. **Declarative Execution**: Replace imperative `forward_batch()` with graph-based model specification
2. **Centralized Resource Management**: Consolidate buffer, cache, and weight lifecycle into dedicated managers
3. **MPI-Aware Graph Building**: Automatic allreduce insertion based on parallelism configuration
4. **Zero Runtime Overhead**: Graph construction at model load time, not per-forward
5. **Gradual Migration**: Parallel execution paths until parity is proven

---

## Table of Contents

1. [Current Architecture Analysis](#current-architecture-analysis)
2. [Existing Infrastructure (REUSE)](#existing-infrastructure-reuse)
3. [Target Architecture](#target-architecture)
4. [Phase 1: Buffer Management](#phase-1-buffer-management)
5. [Phase 2: KV Cache Integration](#phase-2-kv-cache-integration)
6. [Phase 3: Weight Integration (NOT New!)](#phase-3-weight-integration-not-new)
7. [Phase 4: Embedding & LM Head Stages](#phase-4-embedding--lm-head-stages)
8. [Phase 5: Model Graph Abstraction](#phase-5-model-graph-abstraction)
9. [Phase 6: MPI Strategy Configuration](#phase-6-mpi-strategy-configuration)
10. [Migration Timeline](#migration-timeline)
11. [Risk Assessment](#risk-assessment)
12. [Success Criteria](#success-criteria)

---

## Current Architecture Analysis

### PipelineBase (1,363 lines)

`PipelineBase` serves as the foundation for all model pipelines, providing:

| Capability | Implementation | Lines |
|------------|----------------|-------|
| Model Context | `model_ctx_`, `mpi_ctx_`, `device_idx_`, `config_` | ~50 |
| Weight Placement | `placement_map_`, `active_devices_`, `buffers_per_device_` | ~150 |
| Buffer Management | `ActivationBuffers` struct, `createBuffersForDevice()` | ~200 |
| KV Cache | `kv_cache_`, `current_positions_`, `initializeKVCache()` | ~100 |
| Attention Workspace | `attention_workspace_scores_`, `_qkv_buffer_`, `_context_`, `_mask_` | ~80 |
| Declarative Ops | `rms_norm()`, `project()`, `project_row_parallel()`, `add_residual()`, `rope()`, `swiglu()` | ~400 |
| MPI Strategy | `mpi_config_`, `selectOptimalStrategy()`, distribution helpers | ~200 |
| Snapshots | `captureSnapshot()`, `CAPTURE_SNAPSHOT` macro | ~50 |

### Qwen2Pipeline (1,562 lines)

Architecture-specific implementation adding:

| Capability | Implementation | Lines |
|------------|----------------|-------|
| LayerWeights | wq/wk/wv/wo/attn_norm/gate_proj/up_proj/down_proj/ffn_norm + biases | ~150 |
| FusedGEMM | `qkv_fused`, `gate_up_fused` kernels per layer | ~100 |
| Embedding/LM Head | `embedding_batch()`, `lm_head_batch()`, lazy weight loading | ~200 |
| Batch Support | `forward_batch()`, `padded_seq_len_`, `sequence_lengths_` | ~300 |
| Fused Attention+Wo | `fused_attn_wo_kernel_` (optional Q8_1 path) | ~100 |
| Transformer Layer | `transformer_layer()`, `attention_block()`, `ffn_block()` | ~500 |

### Qwen2LayerExecutor (836 lines)

Current executor implementation with:

| Capability | Implementation | Status |
|------------|----------------|--------|
| Config Mirroring | `Qwen2ExecutorConfig` | ✅ Complete |
| Graph Building | `buildAttentionGraph()`, `buildFFNGraph()` | ✅ Complete |
| Stage Wiring | RMSNorm → QKV → RoPE → Attention → Wo → Residual | ✅ Complete |
| Buffer Mapping | `Qwen2ActivationBuffers` (borrows, doesn't own) | ⚠️ Needs ownership |
| KV Cache | Uses passed `kv_cache` pointer | ⚠️ Needs ownership |
| Embedding/LMHead | Not implemented | ❌ Missing |
| MPI Auto-wiring | Manual allreduce insertion | ⚠️ Needs automation |

### Identified Gaps

| Feature | Pipeline | Executor | Gap |
|---------|----------|----------|-----|
| Buffer ownership | ✅ Owns | ❌ Borrows | Executor needs BufferManager |
| KV cache lifecycle | ✅ Manages | ❌ Uses pointer | Executor needs CacheManager |
| Weight loading | ✅ Lazy load | ❌ Receives pointers | Wire to existing ModelContext/WeightManager |
| Embedding lookup | ✅ `embedding_batch()` | ❌ None | Need EmbeddingStage |
| LM head projection | ✅ `lm_head_batch()` | ❌ None | Need LMHeadStage |
| Full model graph | ❌ Imperative loop | ❌ Per-layer only | Need ModelGraph |
| MPI auto-allreduce | ❌ Manual checks | ❌ Manual checks | Need config-driven |

---

## Existing Infrastructure (REUSE)

**CRITICAL**: We already have robust infrastructure for weight loading, placement, and device orchestration. The executor migration should **reuse these components**, not replace them.

### ModelLoader (`src/v2/loaders/ModelLoader.h`)

**What it does:**
- Parses GGUF files (magic, version, metadata, tensor info)
- Supports quantization formats: F32, F16, Q4_0, Q6_K, Q8_0, IQ4_NL
- Provides `loadTensor()`, `loadTensorRowSlice()`, `loadTensorColumnSlice()`
- Device-aware tensor creation via TensorFactory

**Executor integration:** ModelLoader stays as-is. The executor gets weights through ModelContext.

### WeightManager (`src/v2/loaders/WeightManager.h`)

**What it does:**
- Sits between ModelContext and pipelines
- Applies distribution strategies: REPLICATED, SHARDED, INTERLEAVED
- Supports sharding modes: REPLICATE, COLUMN_PARALLEL, ROW_PARALLEL, INPUT_PARALLEL
- Caches loaded tensors for reuse
- Knows if weight `isGemmWeight()` (for packing optimization)

**Executor integration:** WeightManager stays as-is. The executor calls `weight_manager_->getWeight()` exactly like the pipeline does.

### WeightPlacementMap (`src/v2/loaders/WeightPlacementMap.h`)

**What it does:**
- Encodes fine-grained weight→device mapping
- Supports per-tensor, per-layer, pattern-based, and default placement
- Block-level convenience: `setAttentionDevice()`, `setFFNDevice()`, `getAttentionDevice()`, `getFFNDevice()`

**Executor integration:** Graph builder queries WeightPlacementMap to determine device for each stage. No changes needed.

### DeviceOrchestrator (`src/v2/loaders/DeviceOrchestrator.h`)

**What it does:**
- High-level orchestration of placement strategies
- Strategies: ALL_GPU, ALL_CPU, LAYER_SPLIT, AUTO, MEMORY_AWARE, MOE_OPTIMIZED, CUSTOM, MULTI_GPU
- Creates WeightPlacementMap based on strategy + model metadata + hardware

**Executor integration:** DeviceOrchestrator is called at model load time to produce placement_map. Executor receives the map through ModelContext.

### ModelContext (`src/v2/loaders/ModelContext.h`)

**What it does:**
- Wraps model path, GGUFModel metadata, ModelLoader, WeightManager, WeightPlacementMap
- Single entry point: `ModelContext::create(path, mpi_ctx, placement_map, factory, strategy)`
- Exposes `getWeight()`, `model()`, `architecture()`, `weightManager()`, `placementMap()`

**Executor integration:** Qwen2ModelExecutor receives `ModelContext` exactly like Qwen2Pipeline does. This is the key reuse point.

### What This Means for the Migration

| Originally Planned | Actual Approach |
|--------------------|-----------------|
| New `ExecutorWeightManager` | ❌ DELETE - Use existing `ModelContext::getWeight()` |
| Weight declaration API | ❌ DELETE - WeightManager already handles lazy loading |
| Lazy loading refactor | ❌ DELETE - Already exists in WeightManager |
| Device staging | ❌ DELETE - WeightPlacementMap + TensorBase::ensureOnDevice() |

**Revised Phase 3:** Instead of building new weight infrastructure, we wire the existing `ModelContext` into the executor. This reduces Phase 3 from ~400 lines to ~50 lines of integration code.

---

## Target Architecture

### Component Hierarchy

```
Qwen2ModelExecutor (NEW - replaces Qwen2Pipeline)
│
├── ModelContext (EXISTING - REUSE)
│   ├── ModelLoader (GGUF parsing)
│   ├── WeightManager (distribution, caching)
│   └── WeightPlacementMap (device routing)
│
├── ExecutorBufferManager (NEW)
│   ├── Device-aware allocation
│   ├── Activation buffers per device
│   └── Shared attention workspace
│
├── ExecutorKVCacheManager (NEW)
│   ├── Cache lifecycle
│   ├── Position tracking
│   └── Clear/reset operations
│
├── ModelGraph (NEW)
│   ├── addEmbedding()
│   ├── addTransformerLayer() × n_layers
│   ├── addFinalNorm()
│   ├── addLMHead()
│   └── build() → ComputeGraph
│
└── LayerExecutor (EXISTING - enhanced)
    └── execute(ComputeGraph, DeviceContext)
```

### Execution Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    Qwen2ModelExecutor                           │
├─────────────────────────────────────────────────────────────────┤
│  1. Initialize managers (weights, buffers, cache)               │
│  2. Build ModelGraph declaratively                              │
│  3. Call executor.execute(graph.build(), ctx)                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       ComputeGraph                              │
├─────────────────────────────────────────────────────────────────┤
│  EmbeddingLookupStage                                           │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Layer 0-N (repeated)                                     │   │
│  │  RMSNormStage → FusedQKVGEMMStage → RoPEStage           │   │
│  │       → AttentionWithKVCacheStage → GEMMStage(Wo)       │   │
│  │       → [AllreduceStage?] → ResidualAddStage            │   │
│  │       → RMSNormStage → FusedGateUpGEMMStage             │   │
│  │       → SwiGLUStage → GEMMStage(Down)                   │   │
│  │       → [AllreduceStage?] → ResidualAddStage            │   │
│  └─────────────────────────────────────────────────────────┘   │
│       │                                                         │
│       ▼                                                         │
│  RMSNormStage (final_norm)                                      │
│       │                                                         │
│       ▼                                                         │
│  LMHeadStage                                                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Buffer Management

### Goal

Move buffer allocation from `PipelineBase` into a dedicated `ExecutorBufferManager` that the executor owns.

### New Component

**File:** `src/v2/execution/ExecutorBufferManager.h`

```cpp
#pragma once

#include "../tensors/TensorBase.h"
#include "../pipelines/PipelineConfig.h"
#include <map>
#include <memory>
#include <vector>

namespace llaminar2 {

/**
 * @brief Configuration for buffer allocation
 */
struct BufferConfig {
    int max_seq_len = 2048;
    int batch_size = 1;
    int d_model = 0;
    int n_heads = 0;
    int n_kv_heads = 0;
    int head_dim = 0;
    int d_ff = 0;
    ActivationPrecision precision = ActivationPrecision::FP32;
};

/**
 * @brief Activation buffers for a single device
 * 
 * Mirrors PipelineBase::ActivationBuffers but owned by executor.
 */
struct ExecutorActivationBuffers {
    std::shared_ptr<TensorBase> residual;      // [seq, d_model]
    std::shared_ptr<TensorBase> normalized;    // [seq, d_model]
    std::shared_ptr<TensorBase> Q;             // [seq, n_heads * head_dim]
    std::shared_ptr<TensorBase> K;             // [seq, n_kv_heads * head_dim]
    std::shared_ptr<TensorBase> V;             // [seq, n_kv_heads * head_dim]
    std::shared_ptr<TensorBase> attn_output;   // [seq, n_heads * head_dim]
    std::shared_ptr<TensorBase> attn_proj;     // [seq, d_model]
    std::shared_ptr<TensorBase> gate;          // [seq, d_ff]
    std::shared_ptr<TensorBase> up;            // [seq, d_ff]
    std::shared_ptr<TensorBase> ffn_output;    // [seq, d_model]
    std::shared_ptr<TensorBase> current_hidden; // [seq, d_model]
    int max_seq_len = 0;
};

/**
 * @brief Manages activation buffer allocation across devices
 * 
 * Replaces PipelineBase's buffer management with executor-owned lifecycle.
 */
class ExecutorBufferManager {
public:
    /**
     * @brief Construct with allocation configuration
     */
    explicit ExecutorBufferManager(const BufferConfig& config);
    ~ExecutorBufferManager() = default;
    
    // Non-copyable
    ExecutorBufferManager(const ExecutorBufferManager&) = delete;
    ExecutorBufferManager& operator=(const ExecutorBufferManager&) = delete;
    
    /**
     * @brief Allocate buffers for a specific device
     * @param device_idx Device index (-1 for CPU)
     */
    void allocateForDevice(int device_idx);
    
    /**
     * @brief Get activation buffers for a device
     * @param device_idx Device index
     * @return Reference to buffer pool
     * @throws std::runtime_error if device not allocated
     */
    ExecutorActivationBuffers& getBuffers(int device_idx);
    const ExecutorActivationBuffers& getBuffers(int device_idx) const;
    
    /**
     * @brief Check if device has allocated buffers
     */
    bool hasDevice(int device_idx) const;
    
    /**
     * @brief Get list of allocated devices
     */
    std::vector<int> allocatedDevices() const;
    
    // =========================================================================
    // Attention Workspace (shared across layers)
    // =========================================================================
    
    /**
     * @brief Allocate shared attention workspace
     * @param max_heads Maximum number of attention heads
     * @param max_seq Maximum sequence length
     * @param head_dim Dimension per head
     * @param device_idx Target device
     */
    void allocateAttentionWorkspace(int max_heads, int max_seq, int head_dim, int device_idx);
    
    /**
     * @brief Get workspace tensors (pre-allocated)
     */
    TensorBase* workspaceScores() const { return workspace_scores_.get(); }
    TensorBase* workspaceContext() const { return workspace_context_.get(); }
    TensorBase* workspaceMask() const { return workspace_mask_.get(); }
    
    /**
     * @brief Get configuration
     */
    const BufferConfig& config() const { return config_; }

private:
    BufferConfig config_;
    std::map<int, ExecutorActivationBuffers> device_buffers_;
    
    // Shared attention workspace
    std::shared_ptr<TensorBase> workspace_scores_;
    std::shared_ptr<TensorBase> workspace_context_;
    std::shared_ptr<TensorBase> workspace_mask_;
    
    /**
     * @brief Create buffers for a device with configured dimensions
     */
    ExecutorActivationBuffers createBuffers(int device_idx);
};

} // namespace llaminar2
```

### Integration Changes

1. **Qwen2LayerExecutor** gains `buffer_manager_` member
2. Constructor accepts `BufferConfig` and creates manager
3. `executeAttention()` / `executeFFN()` use owned buffers
4. External `Qwen2ActivationBuffers` parameter becomes optional (backward compat)

### Estimated Effort

- New code: ~300 lines
- Refactoring: ~100 lines in Qwen2LayerExecutor
- Tests: ~150 lines

---

## Phase 2: KV Cache Integration

### Goal

Move KV cache lifecycle management from `PipelineBase` into executor framework.

### New Component

**File:** `src/v2/execution/ExecutorKVCacheManager.h`

```cpp
#pragma once

#include "../tensors/UnifiedKVCache.h"
#include <memory>
#include <vector>

namespace llaminar2 {

/**
 * @brief Configuration for KV cache
 */
struct KVCacheConfig {
    int n_layers = 0;
    int max_seq_len = 2048;
    int batch_size = 1;
    int n_kv_heads = 0;
    int head_dim = 0;
    ActivationPrecision precision = ActivationPrecision::FP32;
};

/**
 * @brief Manages KV cache lifecycle for executor
 * 
 * Replaces PipelineBase's KV cache management with executor-owned lifecycle.
 */
class ExecutorKVCacheManager {
public:
    /**
     * @brief Construct with configuration
     */
    explicit ExecutorKVCacheManager(const KVCacheConfig& config);
    ~ExecutorKVCacheManager() = default;
    
    // Non-copyable
    ExecutorKVCacheManager(const ExecutorKVCacheManager&) = delete;
    ExecutorKVCacheManager& operator=(const ExecutorKVCacheManager&) = delete;
    
    /**
     * @brief Initialize cache with per-layer device placement
     * @param layer_devices Device index for each layer's KV cache
     */
    void initialize(const std::vector<int>& layer_devices);
    
    /**
     * @brief Get underlying KV cache
     */
    IUnifiedKVCache* cache() const { return cache_.get(); }
    
    // =========================================================================
    // Position Tracking
    // =========================================================================
    
    /**
     * @brief Update positions after processing tokens
     * @param actual_lengths Tokens processed per sequence in batch
     */
    void updatePositions(const std::vector<int>& actual_lengths);
    
    /**
     * @brief Get current positions for each sequence
     */
    const std::vector<int>& currentPositions() const { return positions_; }
    
    /**
     * @brief Get position for single sequence (batch_idx=0)
     */
    int currentPosition() const { return positions_.empty() ? 0 : positions_[0]; }
    
    /**
     * @brief Reset positions to zero (start new generation)
     */
    void resetPositions();
    
    /**
     * @brief Clear cache contents and reset positions
     */
    void clear();
    
    /**
     * @brief Get configuration
     */
    const KVCacheConfig& config() const { return config_; }

private:
    KVCacheConfig config_;
    std::unique_ptr<IUnifiedKVCache> cache_;
    std::vector<int> positions_;
};

} // namespace llaminar2
```

### Integration Changes

1. **Qwen2LayerExecutor** gains `cache_manager_` member
2. Remove `kv_cache` parameter from `executeLayer()`
3. `AttentionWithKVCacheStage` uses manager's cache internally
4. Position tracking automatic during graph execution

### Estimated Effort

- New code: ~200 lines
- Refactoring: ~80 lines in Qwen2LayerExecutor
- Tests: ~100 lines

---

## Phase 3: Weight Integration

### Goal

Wire the executor to use **existing** `ModelContext` and `WeightManager` infrastructure. No new weight management components needed.

### Key Insight

The existing infrastructure already provides everything we need:

| Need | Provided By |
|------|-------------|
| GGUF parsing | `ModelLoader::loadModel()` |
| Tensor loading | `ModelLoader::loadTensor()` / `loadTensorRowSlice()` |
| Sharding | `WeightManager::getShardedWeight()` |
| Device placement | `WeightPlacementMap::getDeviceForWeight()` |
| Lazy loading | `ModelContext::getWeight()` (caches in WeightManager) |
| Distribution strategies | `WeightManager::setDistributionStrategy()` |

### Existing Qwen2LayerWeights Pattern

The current `Qwen2Pipeline` already uses this pattern:

```cpp
// src/v2/pipelines/qwen/Qwen2Pipeline.h
struct LayerWeights {
    std::shared_ptr<TensorBase> wq, wk, wv, wo;
    std::shared_ptr<TensorBase> attn_norm;
    std::shared_ptr<TensorBase> gate_proj, up_proj, down_proj;
    std::shared_ptr<TensorBase> ffn_norm;
    // ...
};
```

**This pattern is correct.** The executor should use it unchanged.

### Integration Approach

**1. Qwen2ModelExecutor constructor takes ModelContext:**

```cpp
class Qwen2ModelExecutor {
public:
    Qwen2ModelExecutor(
        std::shared_ptr<ModelContext> model_ctx,
        std::shared_ptr<MPIContext> mpi_ctx,
        const Qwen2Config& config
    ) : model_ctx_(model_ctx),
        mpi_ctx_(mpi_ctx),
        config_(config) {}

private:
    std::shared_ptr<ModelContext> model_ctx_;
    std::shared_ptr<MPIContext> mpi_ctx_;
    Qwen2Config config_;
    std::vector<LayerWeights> layer_weights_;  // Populated by loadWeights()
};
```

**2. Weight loading uses existing API:**

```cpp
void Qwen2ModelExecutor::loadWeights() {
    layer_weights_.resize(config_.n_layers);
    
    for (int i = 0; i < config_.n_layers; ++i) {
        auto& w = layer_weights_[i];
        
        // Use existing ModelContext::getWeight() - handles caching, sharding, device
        w.wq = model_ctx_->getWeight(fmt::format("blk.{}.attn_q.weight", i));
        w.wk = model_ctx_->getWeight(fmt::format("blk.{}.attn_k.weight", i));
        w.wv = model_ctx_->getWeight(fmt::format("blk.{}.attn_v.weight", i));
        w.wo = model_ctx_->getWeight(fmt::format("blk.{}.attn_output.weight", i));
        
        w.attn_norm = model_ctx_->getWeight(fmt::format("blk.{}.attn_norm.weight", i));
        
        w.gate_proj = model_ctx_->getWeight(fmt::format("blk.{}.ffn_gate.weight", i));
        w.up_proj = model_ctx_->getWeight(fmt::format("blk.{}.ffn_up.weight", i));
        w.down_proj = model_ctx_->getWeight(fmt::format("blk.{}.ffn_down.weight", i));
        
        w.ffn_norm = model_ctx_->getWeight(fmt::format("blk.{}.ffn_norm.weight", i));
    }
    
    // Global weights
    embedding_ = model_ctx_->getWeight("token_embd.weight");
    final_norm_ = model_ctx_->getWeight("output_norm.weight");
    lm_head_ = model_ctx_->getWeight("output.weight");
}
```

**3. Device placement comes from WeightPlacementMap:**

```cpp
int Qwen2ModelExecutor::getDeviceForLayer(int layer_idx) const {
    // WeightPlacementMap already handles this
    return model_ctx_->placementMap()->getAttentionDevice(layer_idx);
}
```

### What's NOT Needed

| Originally Planned | Why Not Needed |
|--------------------|----------------|
| `WeightDescriptor` struct | Weight names already known (GGUF convention) |
| `declareQwen2Weights()` | Weight requirements implicit in architecture |
| `loadAllWeights()` | Lazy loading via `getWeight()` is sufficient |
| New caching layer | `WeightManager::cache_` already exists |
| Device staging | `TensorBase::ensureOnDevice()` handles this |

### Estimated Effort

- New code: ~50 lines (just the `loadWeights()` method)
- Refactoring: ~20 lines to use `model_ctx_` instead of per-weight loading
- Tests: Existing `ModelContext` tests sufficient

### Migration Note

This phase is dramatically simplified because the existing infrastructure was well-designed. The executor becomes a **consumer** of `ModelContext`, not a replacement for it.

---

## Phase 4: Embedding & LM Head Stages

### Goal

Convert embedding lookup and LM head projection into proper compute stages.

### New Stages

**Added to:** `src/v2/execution/ComputeStage.h`

```cpp
/**
 * @brief Embedding lookup stage
 * 
 * Converts token IDs to dense embeddings via table lookup.
 */
class EmbeddingLookupStage : public IComputeStage {
public:
    struct Params {
        const int* token_ids = nullptr;        // Input tokens [total_tokens]
        TensorBase* output = nullptr;          // Output embeddings [total_tokens, d_model]
        const TensorBase* embedding_table = nullptr;  // [vocab_size, d_model]
        int total_tokens = 0;                  // Total tokens (batch_size * seq_len)
    };
    
    explicit EmbeddingLookupStage(Params params);
    
    bool execute(IDeviceContext* ctx) override;
    ComputeStageType type() const override { return ComputeStageType::EMBEDDING; }
    size_t estimatedFlops() const override;
    size_t estimatedMemoryBytes() const override;
    bool supportsBackend(ComputeBackendType backend) const override;
    StageDumpInfo getDumpInfo() const override;

private:
    Params params_;
};

/**
 * @brief LM head projection stage
 * 
 * Projects hidden states to vocabulary logits.
 */
class LMHeadStage : public IComputeStage {
public:
    struct Params {
        TensorBase* hidden = nullptr;      // Input [total_tokens, d_model]
        TensorBase* logits = nullptr;      // Output [total_tokens, vocab_size]
        const TensorBase* lm_head = nullptr;  // Weight [vocab_size, d_model]
        int total_tokens = 0;
        int vocab_size = 0;
        int d_model = 0;
    };
    
    explicit LMHeadStage(Params params);
    
    bool execute(IDeviceContext* ctx) override;
    ComputeStageType type() const override { return ComputeStageType::LM_HEAD; }
    size_t estimatedFlops() const override;
    size_t estimatedMemoryBytes() const override;
    bool supportsBackend(ComputeBackendType backend) const override;
    StageDumpInfo getDumpInfo() const override;

private:
    Params params_;
};
```

### ComputeStageType Additions

```cpp
enum class ComputeStageType {
    // ... existing ...
    
    // New stages
    EMBEDDING,  ///< Token embedding lookup
    LM_HEAD,    ///< Language model head projection
};
```

### Estimated Effort

- New code: ~200 lines
- Factory methods: ~50 lines
- Tests: ~100 lines

---

## Phase 5: Model Graph Abstraction

### Goal

Create a higher-level abstraction for building complete model graphs declaratively.

### New Component

**File:** `src/v2/execution/ModelGraph.h`

```cpp
#pragma once

#include "ComputeStage.h"
#include "LayerExecutor.h"
#include "ExecutorWeightManager.h"
#include "ExecutorBufferManager.h"
#include "ExecutorKVCacheManager.h"
#include <memory>
#include <vector>

namespace llaminar2 {

/**
 * @brief Model-level configuration for graph building
 */
struct ModelGraphConfig {
    int n_layers = 0;
    int d_model = 0;
    int n_heads = 0;
    int n_kv_heads = 0;
    int head_dim = 0;
    int d_ff = 0;
    int vocab_size = 0;
    float rms_norm_eps = 1e-6f;
    float rope_theta = 10000.0f;
    
    // MPI configuration
    bool tensor_parallel = false;
    bool ffn_column_parallel = false;
    int world_size = 1;
    int rank = 0;
};

/**
 * @brief High-level model graph builder
 * 
 * Provides declarative API for constructing complete model execution graphs.
 * Automatically handles:
 * - Stage parameter wiring
 * - Dependency specification
 * - MPI allreduce insertion
 * - Device placement
 */
class ModelGraph {
public:
    /**
     * @brief Construct with configuration
     */
    explicit ModelGraph(const ModelGraphConfig& config);
    ~ModelGraph() = default;
    
    // Non-copyable, movable
    ModelGraph(const ModelGraph&) = delete;
    ModelGraph& operator=(const ModelGraph&) = delete;
    ModelGraph(ModelGraph&&) = default;
    ModelGraph& operator=(ModelGraph&&) = default;
    
    // =========================================================================
    // Graph Building API
    // =========================================================================
    
    /**
     * @brief Add embedding lookup stage
     * @param token_ids Input token IDs
     * @param output Output tensor
     * @param embedding_table Embedding weights
     * @param total_tokens Total tokens in batch
     */
    ModelGraph& addEmbedding(
        const int* token_ids,
        TensorBase* output,
        const TensorBase* embedding_table,
        int total_tokens);
    
    /**
     * @brief Add a complete transformer layer
     * @param layer_idx Layer index
     * @param weights Layer weights
     * @param buffers Activation buffers
     * @param kv_cache KV cache
     * @param seq_len Sequence length
     * @param position_offset Position for RoPE
     * @param device_idx Target device
     */
    ModelGraph& addTransformerLayer(
        int layer_idx,
        const Qwen2LayerWeightsOwned& weights,
        ExecutorActivationBuffers& buffers,
        IUnifiedKVCache* kv_cache,
        int seq_len,
        int position_offset,
        int device_idx);
    
    /**
     * @brief Add final RMS normalization
     */
    ModelGraph& addFinalNorm(
        TensorBase* input,
        TensorBase* output,
        const TensorBase* gamma,
        int seq_len);
    
    /**
     * @brief Add LM head projection
     */
    ModelGraph& addLMHead(
        TensorBase* hidden,
        TensorBase* logits,
        const TensorBase* lm_head,
        int total_tokens);
    
    // =========================================================================
    // Graph Compilation
    // =========================================================================
    
    /**
     * @brief Build the compute graph
     * @return Compiled ComputeGraph ready for execution
     */
    ComputeGraph build();
    
    /**
     * @brief Get estimated total FLOPs
     */
    size_t estimatedFlops() const;
    
    /**
     * @brief Get number of stages
     */
    size_t numStages() const { return stages_.size(); }

private:
    ModelGraphConfig config_;
    
    struct StageDef {
        std::string name;
        std::unique_ptr<IComputeStage> stage;
        std::vector<std::string> dependencies;
        int device_idx;
    };
    
    std::vector<StageDef> stages_;
    std::string last_stage_name_;
    
    /**
     * @brief Add attention block stages
     */
    void addAttentionBlock(
        int layer_idx,
        const Qwen2LayerWeightsOwned& weights,
        ExecutorActivationBuffers& buffers,
        IUnifiedKVCache* kv_cache,
        int seq_len,
        int position_offset,
        int device_idx);
    
    /**
     * @brief Add FFN block stages
     */
    void addFFNBlock(
        int layer_idx,
        const Qwen2LayerWeightsOwned& weights,
        ExecutorActivationBuffers& buffers,
        int seq_len,
        int device_idx);
    
    /**
     * @brief Insert allreduce if needed
     */
    void maybeAddAllreduce(
        const std::string& after_stage,
        TensorBase* tensor,
        size_t count,
        int device_idx);
};

} // namespace llaminar2
```

### New Model Executor

**File:** `src/v2/pipelines/qwen/Qwen2ModelExecutor.h`

```cpp
#pragma once

#include "../../execution/ModelGraph.h"
#include "../../execution/ExecutorWeightManager.h"
#include "../../execution/ExecutorBufferManager.h"
#include "../../execution/ExecutorKVCacheManager.h"
#include "../../execution/LayerExecutor.h"
#include "../../execution/DeviceContext.h"
#include "../../loaders/ModelContext.h"
#include "../../utils/MPIContext.h"
#include <memory>
#include <vector>

namespace llaminar2 {

/**
 * @brief Configuration for Qwen2ModelExecutor
 */
struct Qwen2ModelExecutorConfig {
    int max_seq_len = 2048;
    int batch_size = 1;
    ActivationPrecision precision = ActivationPrecision::FP32;
    ExecutionMode mode = ExecutionMode::SEQUENTIAL;
    bool enable_profiling = false;
    bool enable_snapshots = false;
};

/**
 * @brief Complete model executor for Qwen2 architecture
 * 
 * Replaces Qwen2Pipeline with declarative graph-based execution.
 * Owns all resources: buffers, cache, weights.
 */
class Qwen2ModelExecutor {
public:
    /**
     * @brief Construct from model context
     */
    Qwen2ModelExecutor(
        std::shared_ptr<ModelContext> model_ctx,
        std::shared_ptr<MPIContext> mpi_ctx,
        const Qwen2ModelExecutorConfig& config);
    
    ~Qwen2ModelExecutor() = default;
    
    // Non-copyable
    Qwen2ModelExecutor(const Qwen2ModelExecutor&) = delete;
    Qwen2ModelExecutor& operator=(const Qwen2ModelExecutor&) = delete;
    
    // =========================================================================
    // Inference API
    // =========================================================================
    
    /**
     * @brief Forward pass for batched input
     * @param token_batches Token sequences [batch_size][seq_len]
     * @return true on success
     */
    bool forward(const std::vector<std::vector<int>>& token_batches);
    
    /**
     * @brief Get output logits after forward pass
     * @return Logits tensor [total_tokens, vocab_size]
     */
    const TensorBase* logits() const;
    
    /**
     * @brief Clear KV cache and reset positions
     */
    void clear();
    
    // =========================================================================
    // Introspection
    // =========================================================================
    
    int n_layers() const { return n_layers_; }
    int d_model() const { return d_model_; }
    int vocab_size() const { return vocab_size_; }
    
    const LayerExecutorStats& stats() const { return executor_.stats(); }

private:
    // Configuration
    Qwen2ModelExecutorConfig config_;
    int n_layers_, d_model_, n_heads_, n_kv_heads_, head_dim_, d_ff_, vocab_size_;
    float rms_norm_eps_, rope_theta_;
    
    // Owned resources
    std::unique_ptr<ExecutorWeightManager> weight_manager_;
    std::unique_ptr<ExecutorBufferManager> buffer_manager_;
    std::unique_ptr<ExecutorKVCacheManager> cache_manager_;
    
    // Execution infrastructure
    LayerExecutor executor_;
    std::unique_ptr<IDeviceContext> device_ctx_;
    std::shared_ptr<MPIContext> mpi_ctx_;
    
    // Output buffers
    std::shared_ptr<TensorBase> logits_buffer_;
    
    /**
     * @brief Build model graph for current batch
     */
    ModelGraph buildGraph(
        const int* tokens,
        int total_tokens,
        int seq_len);
};

} // namespace llaminar2
```

### Estimated Effort

- ModelGraph: ~500 lines
- Qwen2ModelExecutor: ~400 lines
- Tests: ~300 lines

---

## Phase 6: MPI Strategy Configuration

### Goal

Make MPI parallelism configuration-driven rather than scattered checks.

### Enhanced Configuration

```cpp
/**
 * @brief MPI execution configuration
 */
struct ExecutorMPIConfig {
    // Strategy selection
    MPIStrategy strategy = MPIStrategy::None;
    
    // Topology
    int world_size = 1;
    int rank = 0;
    MPI_Comm comm = MPI_COMM_WORLD;
    
    // Tensor parallelism options
    struct TensorParallelConfig {
        bool enabled = false;
        bool shard_attention_heads = false;  // Distribute Q/K/V heads
        bool shard_ffn_columns = false;      // Distribute gate/up columns
        
        // Derived (auto-calculated from enabled + world_size)
        int local_heads = 0;
        int local_kv_heads = 0;
        int local_d_ff = 0;
    } tensor_parallel;
    
    // Pipeline parallelism options
    struct PipelineParallelConfig {
        bool enabled = false;
        int layers_per_rank = 0;
        int start_layer = 0;
        int end_layer = 0;
    } pipeline_parallel;
    
    /**
     * @brief Auto-configure based on model architecture
     */
    static ExecutorMPIConfig autoSelect(
        int n_heads, int n_kv_heads, int d_ff, int n_layers,
        int world_size, int rank);
};
```

### Integration with ModelGraph

```cpp
// In ModelGraph::addTransformerLayer():
if (config_.tensor_parallel && isRowParallelWeight(weights.wo)) {
    // Automatically insert allreduce after Wo projection
    addAllreduce(wo_output, seq_len * d_model, device_idx);
}

if (config_.ffn_column_parallel) {
    // Automatically insert allreduce after Down projection
    addAllreduce(down_output, seq_len * d_model, device_idx);
}
```

### Estimated Effort

- Config refactoring: ~200 lines
- Auto-insertion logic: ~100 lines
- Tests: ~100 lines

---

## Migration Timeline

### Phase Schedule

| Phase | Description | Duration | Dependencies |
|-------|-------------|----------|--------------|
| **Phase 1** | Buffer Management | 2-3 days | None |
| **Phase 2** | KV Cache Integration | 1-2 days | Phase 1 |
| **Phase 3** | Weight Integration | 0.5 days | Phase 1 (trivial - reuse existing) |
| **Phase 4** | Embedding & LM Head Stages | 1-2 days | None |
| **Phase 5** | Model Graph Abstraction | 3-4 days | Phases 1-4 |
| **Phase 6** | MPI Strategy Configuration | 1-2 days | Phase 5 |

**Total Estimated Duration:** 8-14 days (reduced from 10-16)

**NOTE:** Phase 3 dramatically reduced because existing `ModelContext`/`WeightManager`/`WeightPlacementMap` infrastructure is reused as-is. No new weight management code needed.

### Milestones

1. **M1 (Phase 2 complete):** Executor owns buffer + cache lifecycle
2. **M2 (Phase 4 complete):** Full model can be represented as stages
3. **M3 (Phase 5 complete):** `Qwen2ModelExecutor` functional behind flag
4. **M4 (Phase 6 complete):** MPI parallelism automatic, parity proven
5. **M5 (Post-migration):** Deprecate old pipeline, remove dead code

---

## Risk Assessment

### Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Performance regression | Medium | High | Benchmark each phase, A/B comparison |
| Memory overhead from dual paths | Low | Medium | Lazy cleanup after parity proven |
| KV cache incompatibility | Low | High | Maintain interface compatibility |
| MPI correctness issues | Medium | High | Extensive allreduce testing |

### Process Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Scope creep | Medium | Medium | Strict phase boundaries |
| Breaking existing tests | Low | Medium | Parallel paths, feature flags |
| Documentation lag | Medium | Low | Update docs with each phase |

---

## Success Criteria

### Functional Requirements

- [ ] All unit tests pass with executor path
- [ ] Integration tests pass with executor path
- [ ] E2E parity tests show identical output (within FP tolerance)
- [ ] Benchmark performance within 5% of pipeline path
- [ ] Memory usage comparable (±10%)

### Quality Requirements

- [ ] No new runtime allocations in hot path
- [ ] Graph construction < 1ms for typical models
- [ ] Stage execution timing matches kernel profiler
- [ ] Snapshot capture works with executor path

### Migration Completion

- [ ] `Qwen2ModelExecutor` is default path
- [ ] `Qwen2Pipeline` marked deprecated
- [ ] Old pipeline code removed (Phase M5)
- [ ] Documentation updated
- [ ] Changelog entry written

---

## Code Metrics

### Estimated Changes

| Category | New Lines | Refactored | Deleted (after) |
|----------|-----------|------------|-----------------|
| Buffer Manager | ~300 | ~100 | - |
| Cache Manager | ~200 | ~80 | - |
| Weight Manager | ~400 | ~150 | - |
| New Stages | ~200 | - | - |
| ModelGraph | ~500 | - | - |
| ModelExecutor | ~400 | - | - |
| MPI Config | ~200 | ~100 | - |
| **Subtotal (new)** | **~2,200** | **~430** | - |
| **Post-migration cleanup** | - | - | **~1,200** |

**Net Change:** +1,000 to +1,500 lines (cleaner, more maintainable)

---

## Appendix A: File Inventory

### New Files

```
src/v2/execution/
├── ExecutorBufferManager.h      (Phase 1)
├── ExecutorBufferManager.cpp    (Phase 1)
├── ExecutorKVCacheManager.h     (Phase 2)
├── ExecutorKVCacheManager.cpp   (Phase 2)
├── ModelGraph.h                 (Phase 5)
└── ModelGraph.cpp               (Phase 5)

src/v2/pipelines/qwen/
├── Qwen2ModelExecutor.h         (Phase 5)
└── Qwen2ModelExecutor.cpp       (Phase 5)

tests/v2/unit/
├── Test__ExecutorBufferManager.cpp   (Phase 1)
├── Test__ExecutorKVCacheManager.cpp  (Phase 2)
├── Test__EmbeddingStage.cpp          (Phase 4)
├── Test__LMHeadStage.cpp             (Phase 4)
└── Test__ModelGraph.cpp              (Phase 5)

tests/v2/integration/
└── Test__Qwen2ModelExecutor.cpp      (Phase 5)
```

### Existing Files (REUSED - No Changes Needed)

```
src/v2/loaders/
├── ModelLoader.h                (GGUF parsing - unchanged)
├── ModelLoader.cpp
├── WeightManager.h              (distribution, caching - unchanged)
├── WeightManager.cpp
├── WeightPlacementMap.h         (device routing - unchanged)
├── WeightPlacementMap.cpp
├── DeviceOrchestrator.h         (placement strategies - unchanged)
├── DeviceOrchestrator.cpp
├── ModelContext.h               (facade API - unchanged)
└── ModelContext.cpp
```

### Modified Files

```
src/v2/execution/
├── ComputeStage.h               (Phase 4: add EMBEDDING, LM_HEAD)
├── ComputeStage.cpp             (Phase 4: stage implementations)
└── LayerExecutor.h              (Phase 5: minor interface additions)

src/v2/pipelines/qwen/
├── Qwen2LayerExecutor.h         (Phases 1-2: use managers)
└── Qwen2LayerExecutor.cpp       (Phases 1-2: use managers)
```

### Deprecated (Post M4)

```
src/v2/pipelines/
├── PipelineBase.h               (buffer/cache management only)
└── PipelineBase.cpp             (buffer/cache management only)

src/v2/pipelines/qwen/
├── Qwen2Pipeline.h              (entire file)
└── Qwen2Pipeline.cpp            (entire file)
```

---

## Appendix B: Environment Flags

### Development Flags

```bash
# Enable new executor path (Phase 5+)
export LLAMINAR_USE_MODEL_EXECUTOR=1

# Force old pipeline path (testing)
export LLAMINAR_USE_LEGACY_PIPELINE=1

# Enable executor profiling
export LLAMINAR_EXECUTOR_PROFILING=1

# Enable executor validation (check tensor shapes)
export LLAMINAR_EXECUTOR_VALIDATION=1
```

### Existing Flags (Preserved)

```bash
# Existing LayerExecutor flag (per-layer)
export LLAMINAR_USE_LAYER_EXECUTOR=1

# Execution mode
export LLAMINAR_EXECUTION_MODE=sequential|parallel|pipelined
```

---

*Document Version: 1.0 DRAFT*  
*Last Updated: December 2025*
