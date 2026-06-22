# WeightManager LOCAL Parallelism Support

## Project Overview

This project extends Llaminar's `WeightManager` to support **device-level parallelism** (LOCAL TP, LOCAL PP, and compositions) in addition to the existing MPI-rank-level parallelism (GLOBAL TP/PP).

**Problem Statement**: Currently, `WeightManager::getShardedWeight()` uses only MPI `rank` and `world_size` to determine weight slicing. For LOCAL parallelism scenarios (multiple devices in a single MPI rank), this causes all devices to receive the same **replicated** weight instead of their appropriate shards.

**Root Cause** (from `WeightManager.cpp:339-343`):
```cpp
if (!mpi_ctx_ || mpi_ctx_->world_size() == 1)
{
    // Single rank: no sharding needed
    return getReplicatedWeight(name, device);  // BUG FOR LOCAL TP!
}
```

**Solution**: Extend `WeightManager` to use `TensorParallelConfig` with `DeviceShardingAssignment` for device-level sharding, enabling:
- **Phase 1**: LOCAL TP weight sharding (multiple devices share TP work)
- **Phase 2**: LOCAL PP weight filtering (devices handle different layers)
- **Phase 3**: Compositions (TP within PP stages)

---

## Architecture Context

### Existing Infrastructure (What We Have)

| Component | Purpose | Status |
|-----------|---------|--------|
| `TensorParallelConfig` | Container for per-device sharding assignments | ✅ Complete |
| `DeviceShardingAssignment` | Per-device head/FFN/vocab slice info | ✅ Complete |
| `TensorParallelConfig::forDevice(DeviceId)` | Look up assignment by device | ✅ Complete |
| `ILocalTPContext` | LOCAL TP collective operations | ✅ Complete |
| `WeightManager::setTensorParallelConfig()` | Set TP config on weight manager | ✅ Complete |
| `PartialWeightLoader` | Load weights for layer range (PP) | ✅ Complete |
| `WeightPlacementMap` | Layer→device mapping | ✅ Complete |
| `PPStageDefinition` + CLI parsing | PP config from CLI | ✅ Complete |
| `TPDomain` / `MultiDomainTPConfig` | Multi-group domain support | ✅ Complete |

### What's Missing

| Component | Gap | Phase |
|-----------|-----|-------|
| WeightManager device-aware slicing | Uses MPI rank, not DeviceId | Phase 1 |
| TensorParallelConfig from ILocalTPContext | No factory method | Phase 1 |
| DeviceShardingAssignment layer fields | No `first_layer`/`last_layer` | Phase 2 |
| Graph builders respect layer bounds | Build all layers unconditionally | Phase 2 |
| Local PP communication stages | Only MPI-based P2P exists | Phase 2 |
| Per-PP-stage LocalTPContext | Single global context | Phase 3 |
| TPContextManager | No multi-context orchestration | Phase 3 |

---

## Phase 1: LOCAL TP Weight Sharding

### Goal
Fix the immediate bug: each device in LOCAL TP receives its correct weight shard instead of replicated weights.

### Success Criteria
- `DecodeParity_LocalTP` test passes (currently fails with cosine=0.023)
- Device 0 (CUDA) gets vocab rows 0-75967
- Device 1 (ROCm) gets vocab rows 75968-151935
- Both devices produce reasonable logit values (not inflated/garbage)

### Changes Required

#### 1.1 Add Helper: Build TensorParallelConfig from ILocalTPContext

**File**: `src/v2/config/TensorParallelConfig.h`

```cpp
/**
 * @brief Create TensorParallelConfig from LOCAL TP context
 *
 * Converts ILocalTPContext device/weight info into DeviceShardingAssignments
 * for use by WeightManager during weight loading.
 *
 * @param local_tp_ctx LOCAL TP context with devices and weights
 * @param n_heads Total Q attention heads
 * @param n_kv_heads Total KV heads (for GQA)
 * @param d_ff FFN intermediate dimension
 * @param vocab_size Vocabulary size
 * @return TensorParallelConfig with one assignment per device
 */
static TensorParallelConfig fromLocalTPContext(
    const ILocalTPContext& local_tp_ctx,
    int n_heads,
    int n_kv_heads,
    int d_ff,
    int vocab_size);
```

**File**: `src/v2/config/TensorParallelConfig.cpp`

Implementation mirrors the calculation in `InferenceRunnerFactory.cpp:294-369`:
- Iterate devices in `local_tp_ctx.devices()`
- Use weights from `local_tp_ctx.weights()` (or 1/degree if empty)
- Compute cumulative `head_start`, `kv_head_start`, `d_ff_start`, `vocab_start`
- Last device gets remainder for exact totals

#### 1.2 Set TensorParallelConfig in RankOrchestrator

**File**: `src/v2/execution/RankOrchestrator.cpp`

In `initializeDeviceRunners()`, before the device loop:

```cpp
void RankOrchestrator::initializeDeviceRunners()
{
    // ... existing validation ...

    // =====================================================================
    // BUILD TENSORPARALLELCONFIG FOR LOCAL TP WEIGHT SHARDING
    // =====================================================================
    // This enables WeightManager to slice weights by DeviceId instead of
    // falling back to REPLICATED mode for world_size==1.
    // =====================================================================
    {
        auto weight_mgr = model_ctx_->weightManager();
        if (weight_mgr && tp_ctx_)
        {
            // Get model dimensions from first device runner config (or model_ctx)
            int n_heads = model_ctx_->headCount();
            int n_kv_heads = model_ctx_->headCountKV();
            int d_ff = model_ctx_->feedForwardLength();
            int vocab_size = model_ctx_->vocabSize();
            
            if (d_ff == 0) d_ff = model_ctx_->embeddingLength() * 4; // Estimate
            
            auto tp_config = std::make_shared<TensorParallelConfig>(
                TensorParallelConfig::fromLocalTPContext(
                    *tp_ctx_, n_heads, n_kv_heads, d_ff, vocab_size));
            
            weight_mgr->setTensorParallelConfig(tp_config);
            
            LOG_INFO("RankOrchestrator: Set TensorParallelConfig for LOCAL TP ("
                     << tp_ctx_->degree() << " devices)");
        }
    }

    // ... existing preload + runner creation code ...
}
```

#### 1.3 Modify WeightManager::getWeightForDevice() for Device-Aware Slicing

**File**: `src/v2/loaders/WeightManager.cpp`

The key insight: When `tp_config_` is set and has an assignment for the requested `DeviceId`, use that assignment for slicing **regardless of MPI world_size**.

```cpp
std::shared_ptr<TensorBase> WeightManager::getWeightForDevice(
    const std::string &name,
    DeviceId device,
    int layer_idx)
{
    std::lock_guard<std::mutex> lock(cache_mutex_);

    // =====================================================================
    // CHECK FOR DEVICE-SPECIFIC SHARDING VIA TensorParallelConfig
    // =====================================================================
    // This handles LOCAL TP where world_size==1 but we still need per-device shards.
    // =====================================================================
    if (tp_config_ && strategy_ == WeightDistributionStrategy::SHARDED)
    {
        try
        {
            const auto& assignment = tp_config_->forDevice(device);
            // Device has an assignment - use device-aware sharded loading
            return getShardedWeightForAssignment(name, device, assignment, layer_idx);
        }
        catch (const std::out_of_range&)
        {
            // Device not in TensorParallelConfig - fall through to standard path
            LOG_DEBUG("[WeightManager] Device " << device.to_string() 
                      << " not in TensorParallelConfig, using standard path");
        }
    }

    // ... existing logic (first device, clone for subsequent) ...
}
```

#### 1.4 Add New Method: getShardedWeightForAssignment()

**File**: `src/v2/loaders/WeightManager.h`

```cpp
private:
    /**
     * @brief Load weight with device-specific sharding from TensorParallelConfig
     *
     * Uses DeviceShardingAssignment to determine slice bounds instead of MPI rank.
     * Enables LOCAL TP where multiple devices in one rank need different slices.
     *
     * @param name Weight tensor name
     * @param device Target device
     * @param assignment Device's sharding assignment (heads, FFN, vocab bounds)
     * @param layer_idx Layer index for sharding mode lookup
     * @return Sliced tensor for this device, or nullptr on error
     */
    std::shared_ptr<TensorBase> getShardedWeightForAssignment(
        const std::string &name,
        DeviceId device,
        const DeviceShardingAssignment& assignment,
        int layer_idx);
```

**File**: `src/v2/loaders/WeightManager.cpp`

Implementation logic:
1. Determine `ShardingMode` from weight name (COLUMN_PARALLEL, ROW_PARALLEL, INPUT_PARALLEL, REPLICATE)
2. For REPLICATE: call `getReplicatedWeight()` 
3. For COLUMN_PARALLEL (Q, K, V, Gate, Up, LM Head): slice rows based on assignment
   - `attn_q/k/v.weight`: use `head_start`, `head_count` (scaled by head_dim)
   - `ffn_gate/up.weight`: use `d_ff_start`, `d_ff_count`
   - `output.weight` (LM head): use `vocab_start`, `vocab_count`
4. For ROW_PARALLEL (Wo): slice rows based on `head_start`, `head_count`
5. For INPUT_PARALLEL (Down): slice columns based on `d_ff_start`, `d_ff_count`
6. Cache result with device-specific key

```cpp
std::shared_ptr<TensorBase> WeightManager::getShardedWeightForAssignment(
    const std::string &name,
    DeviceId device,
    const DeviceShardingAssignment& assignment,
    int layer_idx)
{
    // Check per-device cache first
    std::string cache_key = device.to_string() + ":" + name;
    auto it = per_device_cache_.find(cache_key);
    if (it != per_device_cache_.end())
    {
        return it->second;
    }

    ShardingMode mode = getShardingMode(name);

    if (mode == ShardingMode::REPLICATE)
    {
        // Norms, biases, embeddings - full copy
        auto tensor = getReplicatedWeight(name, device);
        per_device_cache_[cache_key] = tensor;
        return tensor;
    }

    // Load full tensor (will be sliced)
    auto full_tensor = loader_.loadTensor(name, device, weight_precision_);
    if (!full_tensor)
    {
        LOG_ERROR("[WeightManager] Failed to load tensor for slicing: " << name);
        return nullptr;
    }

    std::shared_ptr<TensorBase> sliced;
    
    if (mode == ShardingMode::COLUMN_PARALLEL)
    {
        // Determine slice based on weight category
        size_t row_start, row_count;
        
        if (isQKVWeight(name))
        {
            // Q/K/V: slice by head assignment (head_start * head_dim)
            int head_dim = full_tensor->shape()[1] / assignment.totalHeads();  // Approximate
            // For Q: use head_start, head_count
            // For K/V: use kv_head_start, kv_head_count
            if (name.find("attn_q") != std::string::npos)
            {
                row_start = assignment.head_start * head_dim;
                row_count = assignment.head_count * head_dim;
            }
            else // K or V
            {
                row_start = assignment.kv_head_start * head_dim;
                row_count = assignment.kv_head_count * head_dim;
            }
        }
        else if (isFFNGateUpWeight(name))
        {
            // Gate/Up: slice by d_ff assignment
            row_start = assignment.d_ff_start;
            row_count = assignment.d_ff_count;
        }
        else if (isLMHeadWeight(name))
        {
            // LM Head: slice by vocab assignment
            row_start = assignment.vocab_start;
            row_count = assignment.vocab_count;
        }
        else
        {
            LOG_WARN("[WeightManager] Unknown COLUMN_PARALLEL weight: " << name);
            row_start = 0;
            row_count = full_tensor->shape()[0];
        }

        sliced = sliceRowRange(full_tensor, row_start, row_count);
    }
    else if (mode == ShardingMode::ROW_PARALLEL)
    {
        // Wo: slice rows by head assignment
        int head_dim = /* compute from tensor shape */;
        size_t row_start = assignment.head_start * head_dim;
        size_t row_count = assignment.head_count * head_dim;
        sliced = sliceRowRange(full_tensor, row_start, row_count);
    }
    else if (mode == ShardingMode::INPUT_PARALLEL)
    {
        // Down: slice columns by d_ff assignment
        sliced = sliceColumnRange(full_tensor, assignment.d_ff_start, assignment.d_ff_count);
    }

    if (sliced)
    {
        sliced->setDebugName(name + "@" + device.to_string());
        per_device_cache_[cache_key] = sliced;
        
        LOG_DEBUG("[WeightManager] Device-sharded " << name << " for " << device.to_string()
                  << " [" << sliced->shape()[0] << "x" 
                  << (sliced->shape().size() > 1 ? sliced->shape()[1] : 1) << "]"
                  << " (assignment: local_rank=" << assignment.local_rank << ")");
    }

    return sliced;
}
```

#### 1.5 Add Helper Methods for Weight Category Detection

**File**: `src/v2/loaders/WeightManager.h` (private section)

```cpp
// Weight category detection helpers
static bool isQKVWeight(const std::string& name);
static bool isFFNGateUpWeight(const std::string& name);
static bool isFFNDownWeight(const std::string& name);
static bool isLMHeadWeight(const std::string& name);
static bool isWoWeight(const std::string& name);
```

#### 1.6 Add Static Slice Methods

**File**: `src/v2/loaders/WeightManager.h`

```cpp
/**
 * @brief Slice a range of rows from a tensor
 * @param tensor Source tensor [rows, cols]
 * @param row_start Starting row index
 * @param row_count Number of rows to extract
 * @return New tensor with shape [row_count, cols]
 */
static std::shared_ptr<TensorBase> sliceRowRange(
    const std::shared_ptr<TensorBase>& tensor,
    size_t row_start,
    size_t row_count);

/**
 * @brief Slice a range of columns from a tensor
 * @param tensor Source tensor [rows, cols]
 * @param col_start Starting column index
 * @param col_count Number of columns to extract
 * @return New tensor with shape [rows, col_count]
 */
static std::shared_ptr<TensorBase> sliceColumnRange(
    const std::shared_ptr<TensorBase>& tensor,
    size_t col_start,
    size_t col_count);
```

### Phase 1 Test Plan

1. **Unit Test**: `Test__WeightManager_LocalTPSlicing`
   - Create mock ILocalTPContext with 2 devices
   - Build TensorParallelConfig from it
   - Verify `getWeightForDevice()` returns correct slice sizes
   
2. **Integration Test**: Run existing `DecodeParity_LocalTP`
   - Should now pass with cosine > 0.8
   - Both devices should produce similar max logit values

### Phase 1 Files Modified

| File | Changes |
|------|---------|
| `src/v2/config/TensorParallelConfig.h` | Add `fromLocalTPContext()` factory |
| `src/v2/config/TensorParallelConfig.cpp` | Implement factory method |
| `src/v2/loaders/WeightManager.h` | Add `getShardedWeightForAssignment()`, helpers |
| `src/v2/loaders/WeightManager.cpp` | Implement device-aware slicing |
| `src/v2/execution/RankOrchestrator.cpp` | Build and set TensorParallelConfig |

---

## Phase 2: LOCAL PP Weight Filtering

### Goal
Support Pipeline Parallelism within a single MPI rank, where different devices handle different layer ranges.

### Success Criteria
- Device 0 loads only layers 0-13 (+ embedding)
- Device 1 loads only layers 14-27 (+ output_norm, LM head)
- Graph builders only construct nodes for assigned layers
- Activation tensors passed between PP stages via local P2P

### Existing Infrastructure to Leverage

| Component | Location | How to Use |
|-----------|----------|------------|
| `PartialWeightLoader` | `src/v2/loaders/PartialWeightLoader.cpp` | Already filters weights by layer range |
| `LayerDevicePlacement` | `src/v2/config/LayerDevicePlacement.h` | Has `shouldBuildLayer()`, `firstLayer()`, `lastLayer()` |
| `PPStageDefinition` | `src/v2/orchestration/OrchestrationConfig.h` | CLI parsing for `--pp-stage` |
| `RankExecutionPlan` | `src/v2/orchestration/RankExecutionPlan.h` | Has `first_layer`, `last_layer`, `has_embedding`, `has_lm_head` |

### Changes Required

#### 2.1 Extend DeviceShardingAssignment for PP

**File**: `src/v2/config/TensorParallelConfig.h`

```cpp
struct DeviceShardingAssignment
{
    // ... existing TP fields ...

    // Pipeline Parallelism fields (Phase 2)
    int first_layer = -1;   // First layer this device handles (-1 = all)
    int last_layer = -1;    // Last layer (inclusive) this device handles (-1 = all)
    
    // Convenience methods
    bool hasAllLayers() const { return first_layer == -1 && last_layer == -1; }
    bool hasLayer(int layer_idx) const {
        if (hasAllLayers()) return true;
        return layer_idx >= first_layer && layer_idx <= last_layer;
    }
    bool hasEmbedding() const { return hasAllLayers() || first_layer == 0; }
    bool hasLMHead(int n_layers) const { return hasAllLayers() || last_layer == n_layers - 1; }
};
```

#### 2.2 Build TensorParallelConfig with PP Info

**File**: `src/v2/config/TensorParallelConfig.h`

```cpp
/**
 * @brief Create TensorParallelConfig for LOCAL PP + optional TP
 *
 * Each device gets a layer range and optionally a TP slice within those layers.
 *
 * @param pp_stages PP stage definitions with layer ranges per device
 * @param n_heads, n_kv_heads, d_ff, vocab_size Model dimensions
 * @param tp_within_stage If true, also apply TP slicing within each PP stage
 */
static TensorParallelConfig fromPPStageDefinitions(
    const std::vector<PPStageDefinition>& pp_stages,
    int n_layers,
    int n_heads,
    int n_kv_heads,
    int d_ff,
    int vocab_size,
    bool tp_within_stage = false);
```

#### 2.3 Modify WeightManager for Layer-Aware Loading

**File**: `src/v2/loaders/WeightManager.cpp`

In `getShardedWeightForAssignment()`, add layer filtering:

```cpp
std::shared_ptr<TensorBase> WeightManager::getShardedWeightForAssignment(
    const std::string &name,
    DeviceId device,
    const DeviceShardingAssignment& assignment,
    int layer_idx)
{
    // =====================================================================
    // PHASE 2: CHECK IF THIS DEVICE HANDLES THIS LAYER
    // =====================================================================
    if (!assignment.hasAllLayers() && layer_idx >= 0)
    {
        if (!assignment.hasLayer(layer_idx))
        {
            // This device doesn't handle this layer - return nullptr
            LOG_DEBUG("[WeightManager] Device " << device.to_string() 
                      << " skipping layer " << layer_idx << " weight " << name
                      << " (assigned layers " << assignment.first_layer 
                      << "-" << assignment.last_layer << ")");
            return nullptr;
        }
    }
    
    // Special handling for non-layer weights
    if (layer_idx < 0)
    {
        // Embedding weight: only first PP stage
        if (isEmbeddingWeight(name) && !assignment.hasEmbedding())
        {
            return nullptr;
        }
        // LM head / output_norm: only last PP stage
        if ((isLMHeadWeight(name) || isOutputNormWeight(name)) && 
            !assignment.hasLMHead(n_layers_))
        {
            return nullptr;
        }
    }

    // ... existing slicing logic ...
}
```

#### 2.4 Modify Graph Builders to Respect Layer Bounds

**File**: `src/v2/models/qwen/Qwen2Graph.cpp`

```cpp
void Qwen2Graph::build()
{
    // ... existing embedding/input handling ...

    // Build transformer layers (RESPECTING PP BOUNDS)
    int first_layer = config_.first_layer >= 0 ? config_.first_layer : 0;
    int last_layer = config_.last_layer >= 0 ? config_.last_layer : config_.n_layers - 1;
    
    for (int layer = first_layer; layer <= last_layer; ++layer)
    {
        buildTransformerLayer(layer);
    }

    // Only build LM head if this PP stage includes it
    if (config_.last_layer < 0 || config_.last_layer == config_.n_layers - 1)
    {
        buildLMHead();
    }
}
```

**File**: `src/v2/execution/DeviceGraphOrchestrator.cpp`

Similar changes to respect `config_.first_layer` / `config_.last_layer`.

#### 2.5 Add Qwen2GraphConfig PP Fields

**File**: `src/v2/models/qwen/Qwen2GraphConfig.h`

```cpp
struct Qwen2GraphConfig
{
    // ... existing fields ...

    // Pipeline Parallelism (Phase 2)
    int first_layer = -1;  // First layer to build (-1 = layer 0)
    int last_layer = -1;   // Last layer to build (-1 = n_layers-1)
    
    // PP stage communication
    bool needs_recv_from_prev = false;  // Receive activations from previous stage
    bool needs_send_to_next = false;    // Send activations to next stage
    int prev_pp_rank = -1;              // Rank to receive from (-1 = none)
    int next_pp_rank = -1;              // Rank to send to (-1 = none)
};
```

#### 2.6 Add Local PP Communication Stages

**Files**: 
- `src/v2/execution/compute_stages/stages/LocalSendActivationsStage.h/cpp`
- `src/v2/execution/compute_stages/stages/LocalReceiveActivationsStage.h/cpp`

These mirror the existing MPI-based `SendActivationsStage` / `ReceiveActivationsStage` but use:
- NCCL `ncclSend/ncclRecv` for CUDA-CUDA
- RCCL for ROCm-ROCm
- PCIeBAR direct copy for CUDA-ROCm
- Host staging as fallback

```cpp
class LocalSendActivationsStage : public ComputeStageBase
{
public:
    LocalSendActivationsStage(
        TensorBase* activations,
        DeviceId target_device,
        ILocalTPContext* local_ctx);  // Uses local_ctx for P2P backend

    void execute() override;
    
private:
    TensorBase* activations_;
    DeviceId target_device_;
    ILocalTPContext* local_ctx_;
};
```

#### 2.7 Integrate PP Stages in Graph Building

**File**: `src/v2/models/qwen/Qwen2Graph.cpp`

```cpp
void Qwen2Graph::build()
{
    // Receive activations from previous PP stage (if not first stage)
    if (config_.needs_recv_from_prev && config_.local_tp_ctx)
    {
        auto recv_stage = std::make_unique<LocalReceiveActivationsStage>(
            input_tensor_, config_.prev_pp_device, config_.local_tp_ctx);
        graph_->addStage(std::move(recv_stage));
    }

    // ... build layers ...

    // Send activations to next PP stage (if not last stage)
    if (config_.needs_send_to_next && config_.local_tp_ctx)
    {
        auto send_stage = std::make_unique<LocalSendActivationsStage>(
            output_tensor_, config_.next_pp_device, config_.local_tp_ctx);
        graph_->addStage(std::move(send_stage));
    }
}
```

### Phase 2 Test Plan

1. **Unit Test**: `Test__WeightManager_LocalPPFiltering`
   - Create config with 2 PP stages (layers 0-13, 14-27)
   - Verify device 0 only loads layers 0-13 weights
   - Verify device 1 only loads layers 14-27 + LM head
   
2. **Integration Test**: `Test__LocalPP_TwoStage_Inference`
   - Run 2-device LOCAL PP inference
   - Verify activations transfer correctly between stages
   - Compare output to single-device reference

### Phase 2 Files Modified

| File | Changes |
|------|---------|
| `src/v2/config/TensorParallelConfig.h` | Add PP fields to DeviceShardingAssignment |
| `src/v2/config/TensorParallelConfig.cpp` | Add `fromPPStageDefinitions()` |
| `src/v2/loaders/WeightManager.cpp` | Layer filtering in `getShardedWeightForAssignment()` |
| `src/v2/models/qwen/Qwen2GraphConfig.h` | Add PP config fields |
| `src/v2/models/qwen/Qwen2Graph.cpp` | Respect layer bounds, add P2P stages |
| `src/v2/execution/DeviceGraphOrchestrator.cpp` | Respect layer bounds |
| NEW: `LocalSendActivationsStage.h/cpp` | Local P2P send |
| NEW: `LocalReceiveActivationsStage.h/cpp` | Local P2P receive |

---

## Phase 3: TP + PP Compositions

### Goal
Support tensor parallelism **within** each pipeline parallel stage. For example:
- PP Stage 0 (layers 0-13): 2-way TP across devices 0,1
- PP Stage 1 (layers 14-27): 2-way TP across devices 2,3

### Success Criteria
- Each PP stage has its own TP communicator group
- TP collectives (AllReduce/AllGather) stay within PP stage
- Activations pass between PP stages at stage boundaries
- WeightManager correctly slices for both TP and PP dimensions

### Existing Infrastructure to Leverage

| Component | Location | How to Use |
|-----------|----------|------------|
| `TPDomain` | `src/v2/config/TPDomain.h` | Defines named device groups |
| `MultiDomainTPConfig` | `src/v2/config/MultiDomainTPConfig.h` | Multiple named domains |
| `BackendRouter` | `src/v2/collective/BackendRouter.h` | Domain-aware backend selection |
| `CollectiveContext` | `src/v2/collective/CollectiveContext.h` | Domain-aware collective execution |
| `--define-domain` CLI | `OrchestrationConfig.h` | Domain definition syntax |
| `--pp-stage` CLI | `OrchestrationConfig.h` | PP stage → domain binding |

### Changes Required

#### 3.1 TPContextManager for Per-Stage Contexts

**File**: `src/v2/config/TPContextManager.h`

```cpp
/**
 * @brief Manages multiple ILocalTPContext instances for TP+PP compositions
 *
 * Each PP stage can have its own TP domain with separate collective
 * communicators. This manager creates and caches LocalTPContext instances
 * per PP stage.
 */
class TPContextManager
{
public:
    /**
     * @brief Create manager from orchestration config
     *
     * Parses domain definitions and PP stage bindings to create
     * one LocalTPContext per PP stage that has TP enabled.
     */
    static std::unique_ptr<TPContextManager> createFromConfig(
        const OrchestrationConfig& config);

    /**
     * @brief Get LocalTPContext for a specific PP stage
     * @param pp_stage_id PP stage index
     * @return Context for that stage's TP domain, or nullptr if no TP
     */
    ILocalTPContext* getContextForStage(int pp_stage_id);

    /**
     * @brief Get all PP stages managed
     */
    const std::vector<int>& ppStages() const { return pp_stages_; }

    /**
     * @brief Get TensorParallelConfig combining all stages
     *
     * Returns a config where each device has both:
     * - TP slice info (heads, FFN, vocab within its PP stage)
     * - PP layer range info
     */
    std::shared_ptr<TensorParallelConfig> buildCombinedTPConfig(
        int n_layers, int n_heads, int n_kv_heads, int d_ff, int vocab_size) const;

private:
    std::unordered_map<int, std::unique_ptr<LocalTPContext>> stage_contexts_;
    std::unordered_map<std::string, TPDomain> domains_;
    std::vector<int> pp_stages_;
    std::vector<PPStageDefinition> stage_definitions_;
};
```

#### 3.2 Extend DeviceShardingAssignment for Combined TP+PP

The `DeviceShardingAssignment` already has both TP fields (heads, FFN, vocab) and PP fields (first_layer, last_layer). For compositions, both are populated:

```cpp
// Example: Device 1 in PP stage 0 (layers 0-13) with 2-way TP
DeviceShardingAssignment {
    .device = DeviceId::cuda(1),
    
    // TP slice (second half of heads/FFN/vocab for this PP stage's portion)
    .head_start = 7,        // heads 7-13 (of 14 total in model)
    .head_count = 7,
    .kv_head_start = 1,     // kv_heads 1-1 (of 2 total)
    .kv_head_count = 1,
    .d_ff_start = 2432,     // FFN second half
    .d_ff_count = 2432,
    .vocab_start = 75968,   // vocab second half (for LM head if last stage)
    .vocab_count = 75968,
    
    // PP layer range
    .first_layer = 0,
    .last_layer = 13,
    
    .local_rank = 1,        // Second device in this TP group
    .work_fraction = 0.5f,
};
```

#### 3.3 Modify RankOrchestrator for Compositions

**File**: `src/v2/execution/RankOrchestrator.h`

```cpp
class RankOrchestrator : public IRankOrchestrator
{
    // ... existing members ...

private:
    // Phase 3: Per-PP-stage contexts
    std::unique_ptr<TPContextManager> tp_manager_;
    
    // PP stage → device indices mapping
    std::unordered_map<int, std::vector<int>> pp_stage_devices_;
};
```

**File**: `src/v2/execution/RankOrchestrator.cpp`

```cpp
void RankOrchestrator::initializeDeviceRunners()
{
    // If we have PP+TP composition, use TPContextManager
    if (hasComposedParallelism())
    {
        tp_manager_ = TPContextManager::createFromConfig(config_);
        
        // Build combined TensorParallelConfig
        auto combined_config = tp_manager_->buildCombinedTPConfig(
            n_layers, n_heads, n_kv_heads, d_ff, vocab_size);
        model_ctx_->weightManager()->setTensorParallelConfig(combined_config);
        
        // Create runners per PP stage
        for (int pp_stage : tp_manager_->ppStages())
        {
            auto* stage_ctx = tp_manager_->getContextForStage(pp_stage);
            createRunnersForPPStage(pp_stage, stage_ctx);
        }
    }
    else
    {
        // Existing simple LOCAL TP path
        // ...
    }
}

void RankOrchestrator::createRunnersForPPStage(
    int pp_stage_id, 
    ILocalTPContext* stage_tp_ctx)
{
    const auto& stage_def = tp_manager_->stageDefinition(pp_stage_id);
    
    for (int device_idx = 0; device_idx < stage_tp_ctx->degree(); ++device_idx)
    {
        InferenceRunnerConfig runner_config;
        runner_config.local_tp_ctx = stage_tp_ctx;
        runner_config.local_tp_device_index = device_idx;
        
        // PP config
        runner_config.first_layer = stage_def.first_layer;
        runner_config.last_layer = stage_def.last_layer;
        runner_config.pp_stage_id = pp_stage_id;
        runner_config.needs_recv_from_prev = (pp_stage_id > 0);
        runner_config.needs_send_to_next = (pp_stage_id < num_pp_stages_ - 1);
        
        auto runner = createTestableInferenceRunner(model_ctx_, device_id, runner_config);
        device_runners_.push_back(std::move(runner));
        pp_stage_devices_[pp_stage_id].push_back(device_runners_.size() - 1);
    }
}
```

#### 3.4 Pipeline Scheduler (Optional Enhancement)

For production-quality PP, implement micro-batch scheduling to reduce pipeline bubbles:

**File**: `src/v2/execution/PipelineScheduler.h`

```cpp
/**
 * @brief Schedules micro-batches across PP stages
 *
 * Implements 1F1B (one-forward-one-backward) scheduling or similar
 * to maximize pipeline utilization.
 */
class PipelineScheduler
{
public:
    struct MicroBatch {
        int batch_id;
        int pp_stage;
        enum Phase { FORWARD, BACKWARD } phase;
    };

    /**
     * @brief Generate schedule for given batch size and PP stages
     */
    std::vector<MicroBatch> generateSchedule(
        int total_batch_size,
        int micro_batch_size,
        int num_pp_stages);

    /**
     * @brief Execute schedule using RankOrchestrator
     */
    void execute(
        RankOrchestrator& orchestrator,
        const std::vector<MicroBatch>& schedule);
};
```

> **Note**: Micro-batch scheduling is a significant undertaking and may be deferred to a later phase. Basic compositions can work without it using synchronous execution.

### Phase 3 Test Plan

1. **Unit Test**: `Test__TPContextManager_Creation`
   - Parse config with 2 PP stages, each 2-way TP
   - Verify 2 LocalTPContext instances created
   - Verify each context has correct devices
   
2. **Unit Test**: `Test__CombinedTPConfig_TPWithinPP`
   - Create combined config for 4 devices (2 PP stages × 2 TP)
   - Verify each DeviceShardingAssignment has both TP and PP info
   - Verify weight loading produces correct slices

3. **Integration Test**: `Test__TPWithinPP_FourDevice_Inference`
   - Run with 4 devices: PP(0)=devices 0,1 (TP), PP(1)=devices 2,3 (TP)
   - Verify TP collectives stay within PP stage
   - Verify PP boundaries transfer activations correctly
   - Compare output to single-device reference

### Phase 3 Files Created/Modified

| File | Changes |
|------|---------|
| NEW: `src/v2/config/TPContextManager.h` | Per-PP-stage context manager |
| NEW: `src/v2/config/TPContextManager.cpp` | Implementation |
| `src/v2/execution/RankOrchestrator.h` | Add `tp_manager_`, composition support |
| `src/v2/execution/RankOrchestrator.cpp` | Composed initialization |
| `src/v2/execution/InferenceRunnerConfig.h` | Add PP config fields |
| OPTIONAL: `src/v2/execution/PipelineScheduler.h/cpp` | Micro-batch scheduling |

---

## Implementation Order

### Recommended Sequence

```
Phase 1 (LOCAL TP) ─────────────────────────────────────────────────────────────►
                   [2-3 days]
    ├── 1.1 TensorParallelConfig::fromLocalTPContext()
    ├── 1.2 RankOrchestrator sets config
    ├── 1.3-1.4 WeightManager device-aware slicing
    ├── 1.5-1.6 Helper methods
    └── Tests

Phase 2 (LOCAL PP) ─────────────────────────────────────────────────────────────►
                   [1-2 weeks]
    ├── 2.1 DeviceShardingAssignment PP fields
    ├── 2.2 TensorParallelConfig::fromPPStageDefinitions()
    ├── 2.3 WeightManager layer filtering
    ├── 2.4-2.5 Graph builder PP awareness
    ├── 2.6-2.7 Local P2P stages
    └── Tests

Phase 3 (Compositions) ─────────────────────────────────────────────────────────►
                       [2-3 weeks]
    ├── 3.1 TPContextManager
    ├── 3.2 Combined TP+PP assignments
    ├── 3.3 RankOrchestrator composition mode
    ├── 3.4 PipelineScheduler (optional)
    └── Tests
```

### Dependencies

```
Phase 1 ──► Phase 2 ──► Phase 3
   │            │
   │            └── Requires: PP fields in DeviceShardingAssignment
   │                          Graph builders respect layer bounds
   │
   └── Requires: TensorParallelConfig on WeightManager
                 Device-aware weight slicing
```

### Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Phase 1 breaks existing GLOBAL TP | Add unit tests for GLOBAL TP before changes |
| Phase 2 P2P stages complex | Start with host-staging fallback, optimize later |
| Phase 3 micro-batching scope creep | Defer to Phase 3b, basic sync works first |
| Performance regression | Benchmark each phase against baseline |

---

## Success Metrics

### Phase 1
- [ ] `DecodeParity_LocalTP` test passes (cosine > 0.8)
- [ ] No regression in GLOBAL TP tests
- [ ] Weight shapes logged correctly per device

### Phase 2
- [ ] 2-device LOCAL PP produces correct output
- [ ] Memory usage reduced proportionally to layer split
- [ ] P2P latency < 100μs for activation transfer

### Phase 3
- [ ] 4-device TP-within-PP produces correct output
- [ ] TP collectives isolated to PP stage
- [ ] (Optional) Micro-batch scheduling improves throughput > 1.5x

---

## Appendix: Code Snippets

### A. Weight Category Detection Helpers

```cpp
// src/v2/loaders/WeightManager.cpp

bool WeightManager::isQKVWeight(const std::string& name)
{
    return name.find("attn_q.weight") != std::string::npos ||
           name.find("attn_k.weight") != std::string::npos ||
           name.find("attn_v.weight") != std::string::npos;
}

bool WeightManager::isFFNGateUpWeight(const std::string& name)
{
    return name.find("ffn_gate.weight") != std::string::npos ||
           name.find("ffn_up.weight") != std::string::npos;
}

bool WeightManager::isFFNDownWeight(const std::string& name)
{
    return name.find("ffn_down.weight") != std::string::npos;
}

bool WeightManager::isLMHeadWeight(const std::string& name)
{
    return name == "output.weight";
}

bool WeightManager::isWoWeight(const std::string& name)
{
    return name.find("attn_output.weight") != std::string::npos;
}

bool WeightManager::isEmbeddingWeight(const std::string& name)
{
    return name == "token_embd.weight";
}

bool WeightManager::isOutputNormWeight(const std::string& name)
{
    return name == "output_norm.weight";
}
```

### B. TensorParallelConfig::fromLocalTPContext()

```cpp
// src/v2/config/TensorParallelConfig.cpp

TensorParallelConfig TensorParallelConfig::fromLocalTPContext(
    const ILocalTPContext& local_tp_ctx,
    int n_heads,
    int n_kv_heads,
    int d_ff,
    int vocab_size)
{
    const auto& devices = local_tp_ctx.devices();
    const auto& weights = local_tp_ctx.weights();
    const int degree = local_tp_ctx.degree();
    
    std::vector<DeviceShardingAssignment> assignments;
    assignments.reserve(degree);
    
    int head_start = 0;
    int kv_head_start = 0;
    int d_ff_start = 0;
    int vocab_start = 0;
    
    for (int i = 0; i < degree; ++i)
    {
        DeviceShardingAssignment assignment;
        assignment.device = devices[i].toLocalDeviceId();
        assignment.local_rank = i;
        
        float w = weights.empty() ? (1.0f / degree) : weights[i];
        assignment.work_fraction = w;
        
        // Compute counts (last device gets remainder)
        int local_heads, local_kv_heads, local_d_ff, local_vocab;
        
        if (i == degree - 1)
        {
            local_heads = n_heads - head_start;
            local_kv_heads = n_kv_heads - kv_head_start;
            local_d_ff = d_ff - d_ff_start;
            local_vocab = vocab_size - vocab_start;
        }
        else
        {
            local_heads = static_cast<int>(std::round(w * n_heads));
            local_kv_heads = static_cast<int>(std::round(w * n_kv_heads));
            local_d_ff = static_cast<int>(std::round(w * d_ff));
            local_vocab = static_cast<int>(std::round(w * vocab_size));
        }
        
        assignment.head_start = head_start;
        assignment.head_count = local_heads;
        assignment.kv_head_start = kv_head_start;
        assignment.kv_head_count = local_kv_heads;
        assignment.d_ff_start = d_ff_start;
        assignment.d_ff_count = local_d_ff;
        assignment.vocab_start = vocab_start;
        assignment.vocab_count = local_vocab;
        
        // PP fields default to all layers
        assignment.first_layer = -1;
        assignment.last_layer = -1;
        
        assignments.push_back(assignment);
        
        head_start += local_heads;
        kv_head_start += local_kv_heads;
        d_ff_start += local_d_ff;
        vocab_start += local_vocab;
    }
    
    return TensorParallelConfig(std::move(assignments));
}
```

### C. Slice Row Range (Quantized-Aware)

```cpp
// src/v2/loaders/WeightManager.cpp

std::shared_ptr<TensorBase> WeightManager::sliceRowRange(
    const std::shared_ptr<TensorBase>& tensor,
    size_t row_start,
    size_t row_count)
{
    if (!tensor || row_count == 0)
    {
        return nullptr;
    }
    
    const auto& shape = tensor->shape();
    if (shape.size() != 2)
    {
        LOG_ERROR("[WeightManager] sliceRowRange requires 2D tensor");
        return nullptr;
    }
    
    size_t total_rows = shape[0];
    size_t cols = shape[1];
    
    if (row_start + row_count > total_rows)
    {
        LOG_ERROR("[WeightManager] Row range [" << row_start << ", " 
                  << row_start + row_count << ") exceeds tensor rows " << total_rows);
        return nullptr;
    }
    
    // Use TensorSlice for efficient view without copy (when possible)
    // For quantized tensors, we may need to copy if block boundaries don't align
    
    auto meta = SliceMetadata::forRowRange(total_rows, cols, row_start, row_count);
    
    // Check if we can use a view or need to copy
    if (tensor->supportsRowSliceView(row_start, row_count))
    {
        return std::make_shared<TensorSlice>(tensor, meta);
    }
    else
    {
        // Need to copy - create new tensor of appropriate type
        return copyRowRange(tensor, row_start, row_count);
    }
}
```

---

## References

- [TensorParallelConfig.h](../../../../src/v2/config/TensorParallelConfig.h) - Existing TP config
- [WeightManager.cpp](../../../../src/v2/loaders/WeightManager.cpp) - Current weight loading
- [RankOrchestrator.cpp](../src/v2/execution/RankOrchestrator.cpp) - Multi-device setup
- [HYBRID_PARALLELISM_PROJECT_PLAN.md](HYBRID_PARALLELISM_PROJECT_PLAN.md) - Full hybrid plan
- [MULTI_DEVICE_ORCHESTRATOR_PLAN.md](MULTI_DEVICE_ORCHESTRATOR_PLAN.md) - Orchestrator design
