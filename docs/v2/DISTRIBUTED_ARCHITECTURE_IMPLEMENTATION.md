# Distributed Architecture Implementation Plan

**Author**: David Sanftenberg  
**Date**: December 2025  
**Reference**: [DISTRIBUTED_ARCHITECTURE_PROPOSAL.md](./DISTRIBUTED_ARCHITECTURE_PROPOSAL.md)

---

## Master Todo List

### Phase 1: MPITopology Foundation ✅ COMPLETE
- [x] Create `MPITopology` class (`src/v2/utils/MPITopology.h`)
- [x] Implement `WorkRange` struct with equal/weighted distribution
- [x] Implement `DeviceCapability` struct
- [x] Implement `RankPlacement` struct
- [x] Add work distribution methods (`get_head_range`, `get_column_range`, etc.)
- [x] Device capability exchange via AllGather

### Phase 2: Stage-Level TP Integration ✅ COMPLETE
- [x] **2.1** Add `KernelFactory::getOrCreateGemmSliced()` ✅
- [x] **2.2** Extend `GEMMStage::Params` with `WorkRange output_range` ✅
- [x] **2.3** Modify `GEMMStage::execute()` to use sliced kernel ✅
- [x] **2.4** Add `head_start/local_n_heads` to `AttentionWithKVCacheStage::Params` ✅
- [x] **2.5** Modify `ITensorAttention::compute_tensor()` interface ✅
- [x] **2.6** Update `CPUAttentionKernelTyped::compute_tensor()` (stub with validation) ✅
- [x] **2.7** Unit tests for sliced GEMM (11 tests in `Test__KernelFactorySliced.cpp`) ✅

### Phase 3: Column-Parallel QKV ✅ COMPLETE (December 21, 2025)
- [x] **3.1** Add column-parallel weight loading to `WeightManager` ✅
  - `attn_q.weight`, `attn_k.weight`, `attn_v.weight` → `COLUMN_PARALLEL`
  - `attn_q.bias`, `attn_k.bias`, `attn_v.bias` → `COLUMN_PARALLEL`
- [x] **3.2** Update `Qwen2Graph::buildAttentionGraph()` to pass head ranges ✅
  - `Qwen2GraphConfig` extended with `head_start`, `local_n_heads`, `local_n_kv_heads`, `qkv_column_parallel`
  - RoPE and Attention stages use `local_n_heads` when TP enabled
- [x] **3.3** Modify Q/K/V buffer allocation for local dimensions ✅
  - `Qwen2BufferSpecBuilder` uses `buf_n_heads` = `local_n_heads` when `qkv_column_parallel=true`
  - Unit tests: 6 tests in `Test__Qwen2BufferSpec.cpp` for local head scenarios
- [x] **3.4** Integration test: 2-rank QKV sharding ✅
  - `Test__MPI_ColumnParallelQKV.cpp` with 5 tests (all passing)
  - Allreduce after Wo projection for row-parallel output

### Phase 4: Column-Parallel FFN ✅ COMPLETE (December 21, 2025)
- [x] **4.1** Add column-parallel Gate/Up weight loading ✅
  - `ffn_gate.weight`, `ffn_up.weight` → `COLUMN_PARALLEL`
  - `ffn_down.weight` → `INPUT_PARALLEL` (matches Gate/Up output dimension)
- [x] **4.2** Update `Qwen2Graph::buildFFNGraph()` for local FFN dim ✅
  - `Qwen2GraphConfig` extended with `d_ff_local`, `ffn_column_parallel`
  - FFN stages use weight shapes directly (auto-adapts to sharded dimensions)
- [x] **4.3** Add `AllreduceStage` after Down projection ✅
  - Conditional allreduce when `ffn_column_parallel=true` OR `down_is_row_sharded`
  - Uses `getMPIComm()` helper for MPI communicator
- [x] **4.4** Integration test: 2-rank FFN sharding ✅
  - `Test__MPI_ColumnParallelFFN.cpp` with 7 tests (all passing)

### Phase 5: Column-Parallel LM Head ✅ COMPLETE (December 22, 2025)
- [x] **5.1** Shard `lm_head` by vocab dimension ✅
  - `output.weight` → `COLUMN_PARALLEL` in `WeightManager::determineShardingMode()`
  - `vocab_local = vocab_size / world_size` (e.g., 151936/2 = 75968)
- [x] **5.2** Add `AllGatherStage` before sampling ✅
  - `AllGatherStage` class with MPI_Allgather for collecting distributed logits
  - Factory method `ComputeStageFactory::createAllGather()`
  - Updated `buildLMHeadGraph()` and `buildFullForwardGraph()` to add AllGather after LM head GEMM
- [x] **5.3** Unit tests: 8 tests in `Test__AllGatherStage.cpp` ✅
- [x] **5.4** Integration test: 5 tests in `Test__MPI_ColumnParallelLMHead.cpp` ✅

### Phase 6: Sharded KV Cache
- [x] **6.1** Modify `UnifiedKVCache` for local head storage ✅
  - Added `IUnifiedKVCache` sharding interface: `is_sharded()`, `n_kv_heads()`, `local_n_kv_heads()`, `kv_head_start()`, `local_kv_dim()`
  - Added sharded constructors accepting `local_n_kv_heads` and `kv_head_start`
  - Added `createShardedKVCache()` factory functions
- [x] **6.2** `KVCacheAppendStage` already works - derives kv_dim from tensor shapes ✅
- [x] **6.3** `KVCacheGatherStage` already works - uses cache's internal kv_dim_ ✅
- [x] **6.4** Updated `PipelineBase::initializeKVCache` for sharded cache ✅
- [x] **6.4b** Updated `GraphOrchestrator::initializeInferenceState` for sharded cache ✅
- [x] **6.5** Unit tests: 13 tests in `Test__ShardedKVCache.cpp` ✅
- [x] **6.6** Integration test: 7 tests in `Test__MPI_ShardedKVCache.cpp` ✅

### Phase 7: Cleanup ⏳ IN PROGRESS
- [x] **7.1** Mark `MpiAttentionOrchestrator` as deprecated ✅ (class has `[[deprecated]]` attribute)
- [x] **7.2** Mark `GQAAttention` as deprecated ✅ (class has `[[deprecated]]` attribute)
- [x] **7.3** Deprecate `PipelineBase` TP methods ✅ (`attention_gqa`, `attention_gqa_batch`, `attention_gqa_mpi`, `attention_gqa_tensor_parallel` all have `[[deprecated]]`)
- [x] **7.4** Decouple `AttentionWithKVCacheStage` from deprecated infrastructure ✅ (now uses `KernelFactory::createAttention()`)
- [ ] **7.5** Update tests to use graph-based execution (future work)
- [ ] **7.6** Full removal of deprecated classes (future work after all callers migrated)

---

### GPU Integration Phases

> **Note**: Phases 2-7 above are for **CPU-only tensor parallelism** (MPI across nodes/sockets).
> The GPU phases below add **heterogeneous execution** (CPU + GPU within a rank).
> Phase G0 is a prerequisite for all other GPU phases.

#### Phase G0: Placement Infrastructure (Foundation for GPU) ✅ COMPLETE
> **Depends on**: Phase 1 (MPITopology)  
> **Enables**: G1-G6 (all GPU phases need placement decisions)

- [x] **G0.1** Create `PlacementStrategy` base class (`src/v2/execution/PlacementStrategy.h`)
- [x] **G0.2** Implement `CPUOnlyStrategy` (default) + `GPUFirstStrategy` placeholder
- [x] **G0.3** Create `PlacementPlan` struct (`src/v2/execution/PlacementPlan.h`)
- [x] **G0.4** Integrate with `MPITopology::all_placements()` for capability input
- [x] **G0.5** Add `WeightPlacementMap::applyPlan()` to populate from plan
- [x] **G0.6** Add `MPITopology::computePlacement()` integration
- [x] **G0.7** Unit tests for placement strategy (34 tests in `Test__PlacementStrategy.cpp`)

#### Phase G1: Single-GPU per Rank
> **Depends on**: G0 (placement decisions)

- [ ] **G1.1** `DeviceManager` filters GPUs by NUMA affinity (already implemented)
- [ ] **G1.2** Default placement strategy: all layers → GPU (if fits in memory)
- [ ] **G1.3** Integration test: single-GPU inference matches CPU

#### Phase G2: GPU Attention Kernels
> **Depends on**: G1

- [ ] **G2.1** Integrate FlashAttention-2 for CUDA
- [ ] **G2.2** Add hipFlashAttention for ROCm
- [ ] **G2.3** `KernelFactory::createAttention()` dispatches to GPU kernel

#### Phase G3: Cross-Device Data Transfer
> **Depends on**: G1

- [ ] **G3.1** Add `DataTransferStage` for explicit GPU↔CPU copies
- [ ] **G3.2** Automatic transfer detection in `GraphExecutor`
- [ ] **G3.3** Async transfers with CUDA streams / HIP streams

#### Phase G4: Heterogeneous Work Distribution
> **Depends on**: G0, G2, G3

- [ ] **G4.1** `LayerBalancedStrategy` (equal layers per device)
- [ ] **G4.2** `ComputeWeightedStrategy` (by `relative_compute` from capabilities)
- [ ] **G4.3** Block-level placement: Attention→GPU, FFN→CPU
- [ ] **G4.4** Benchmark-based calibration of `relative_compute`

#### Phase G5: Device-Specific Weight Packing
> **Depends on**: G1

- [ ] **G5.1** CUDA packed weights (COL32 / COL32_2R_4R4 layout)
- [ ] **G5.2** ROCm packed weights (Matrix Core layout)
- [ ] **G5.3** `KernelFactory` dispatches to device-specific packing

#### Phase G6: Multi-GPU per Rank (Future)
> **Depends on**: G4

- [ ] **G6.1** Extend `RankPlacement::devices` for multiple GPUs
- [ ] **G6.2** Data parallelism within rank
- [ ] **G6.3** NVLink/xGMI-aware placement

---

## Key Architecture Concepts

### Work Distribution Decision Flow (Phase G0) ✅ IMPLEMENTED

**Summary**: How the placement decision flows to all ranks:

```
1. MPITopology::exchangeCapabilities()    ← AllGather device info (Phase 1)
         ↓
2. MPITopology::computePlacement(input)   ← Auto-selects strategy, computes plan
         ↓
3. PlacementStrategy::compute()           ← Algorithm (runs on ALL ranks, deterministic)
         ↓
4. PlacementPlan                          ← Struct with layer→device mappings
         ↓
5. WeightPlacementMap::applyPlan()        ← Populate local map from plan
         ↓
6. WeightManager uses placement_map       ← Loads weights to correct devices
```

**Key Design Choice**: All ranks compute the same plan locally (no broadcast needed) because:
- `PlacementStrategy::compute()` is deterministic
- All ranks have identical `all_placements_` after AllGather
- Avoids serialization/broadcast complexity

**Implemented Files**:
- `src/v2/execution/PlacementPlan.h/.cpp` - `PlacementDevice` enum, `LayerPlacement`, `GlobalPlacement`, `PlacementPlan`
- `src/v2/execution/PlacementStrategy.h/.cpp` - `PlacementInput`, `PlacementStrategy` base, `CPUOnlyStrategy`, `GPUFirstStrategy`, `PlacementStrategyFactory`

**Available Strategies**:
| Strategy | Description | Status |
|----------|-------------|--------|
| `CPUOnlyStrategy` | All layers → CPU (default for CPU-only TP) | ✅ Working |
| `GPUFirstStrategy` | Fill GPU memory first, overflow to CPU | Placeholder (falls back to CPU) |

**Example API**:
```cpp
// In main() or pipeline construction
auto topology = std::make_shared<MPITopology>(...);
topology->exchangeCapabilities();  // AllGather device info

// Each rank computes the same plan (deterministic)
PlacementInput input;
input.architecture = "Qwen2";
input.n_layers = model_loader.getBlockCount();
input.d_model = model_loader.getEmbeddingDim();
input.n_heads = model_loader.getHeadCount();
input.vocab_size = model_loader.getVocabSize();
input.quant_type = model_loader.getGGMLType();

PlacementPlan plan = topology->computePlacement(input);

// Apply plan to local WeightPlacementMap
auto placement_map = std::make_shared<WeightPlacementMap>();
placement_map->applyPlan(plan);

// Pass to WeightManager
WeightManager wm(loader, mpi_ctx, placement_map, WeightDistributionStrategy::SHARDED);
```

### Device-Specific Weight Packing (Phase G5)

**Principle**: GPU kernels pack weights into their own optimal formats, just as CPU kernels do for VNNI.

**Current CPU Packing** (`KernelFactory::getOrCreateGemm()`):
- VNNI layout: `[N/64][K/4][64][4]` for AVX512-VNNI instructions
- Packed weights stored in `tensor->cache_` (type: `TensorPackedWeightsCache*`)
- Pack once on first kernel creation, reuse thereafter

**GPU Packing Strategy** (Phase G5):
- CUDA: COL32 or COL32_2R_4R4 layout for INT8 Tensor Cores
- ROCm: Custom layout for Matrix Cores
- Each backend packs into `tensor->cache_` on first kernel creation

**Why This Works** (enabled by Phase G0):
1. `WeightPlacementMap` assigns each tensor to a **single device** (layer/block granularity)
2. Only one device type's kernel will ever be created for a given tensor
3. No cross-device packing conflict because tensor is device-bound

**Implementation Pattern**:
```cpp
// In KernelFactory::getOrCreateGemm()
auto dev_type = getDeviceType(tensor->device_index());

if (dev_type == DeviceType::CPU) {
    // Pack into VNNI format (existing code)
    packWeightsInto(tensor, new_cache->packed, VNNI_LAYOUT);
    return std::make_unique<QuantisedGemmKernel>(&new_cache->packed);
}
#ifdef HAVE_CUDA
else if (dev_type == DeviceType::CUDA) {
    // Pack into CUDA-optimal format
    auto* cuda_packed = new CudaPackedWeights();
    packForCuda(tensor, cuda_packed);
    tensor->cache_ = cuda_packed;
    return std::make_unique<CudaQuantisedGemmKernel>(cuda_packed);
}
#endif
```

### Intra-Node Work Division (CPU vs GPU)

**Problem**: How does a single MPI rank divide work between its CPU and GPU devices?

**Answer**: Through `WeightPlacementMap` + `GraphOrchestrator` cooperation:

| Component | Role | Scope |
|-----------|------|-------|
| `DeviceManager` | Enumerate CPU sockets and GPUs, NUMA filtering | Startup |
| `WeightPlacementMap` | Assign layers/blocks to device indices | Model load |
| `KernelFactory` | Create device-specific kernels with proper packing | First execution |
| `GraphOrchestrator` | Execute stages on their assigned devices | Forward pass |
| `IDeviceContext` | Actual parallel execution (OpenMP/CUDA/HIP) | Kernel execution |

**Assignment Granularity**:
```cpp
// Layer-level (simple)
placement_map->setLayerDevice(0, GPU_0);   // Layer 0 → GPU
placement_map->setLayerDevice(1, CPU_0);   // Layer 1 → CPU

// Block-level (fine-grained)
placement_map->setAttentionDevice(layer, GPU_0);  // Attention → GPU
placement_map->setFFNDevice(layer, CPU_0);        // FFN → CPU (AVX512 efficient)

// MoE expert-level (future)
placement_map->setLocalExpertDevice(layer, expert_id, device_idx);
```

**Relationship to MPI**:
- **Inter-node**: MPI handles tensor parallelism (AllReduce, AllGather)
- **Intra-node**: WeightPlacementMap handles heterogeneous device execution
- These are orthogonal and composable

---

## Phase 2: Stage-Level TP Integration ✅ COMPLETE

**Proposal Reference**: Section "Migration Path → Phase 2"  
**Status**: ✅ Implemented December 2025

**Goal**: Extend existing `ComputeStage` classes to accept tensor parallelism parameters from `MPITopology`, enabling work division without changing the graph structure.

### Implementation Summary

| Task | Status | Files Modified |
|------|--------|----------------|
| 2.1 Sliced GEMM Factory | ✅ | `KernelFactory.h/.cpp` |
| 2.2 GEMMStage WorkRange | ✅ | `ComputeStage.h` |
| 2.3 GEMMStage execute() | ✅ | `ComputeStage.h` |
| 2.4 Attention TP Params | ✅ | `ComputeStage.h` |
| 2.5 ITensorAttention | ✅ | `TensorKernels.h` |
| 2.6 CPUAttention impl | ✅ (stub) | `CPUAttentionKernelTyped.h` |
| 2.7 Unit tests | ✅ | `Test__KernelFactorySliced.cpp` (11 tests) |

**Key Implementation Notes**:
- All new parameters use defaults for backward compatibility
- Attention head slicing is stubbed with validation; inner loops still use full head range
- Sliced GEMM cache uses composite key: `(tensor_ptr, row_start, row_end)`

### 2.1 Add KernelFactory Sliced GEMM API ✅ IMPLEMENTED

**Files Modified**:
- `src/v2/kernels/KernelFactory.h`
- `src/v2/kernels/KernelFactory.cpp`

**Rationale**: The `QuantisedGemmKernel` already has a row-sliced constructor `QuantisedGemmKernel(weights, row_start, row_end)`. We need to expose this through `KernelFactory` with caching support.

**Actual Implementation** (December 2025):
- Added `SlicedCacheKey` struct with tensor pointer + row range
- Static `sliced_gemm_cache_` with mutex protection
- `clearSlicedCacheFor()` called from tensor destructor to prevent stale entries
- Supports all quantized tensor types (Q8_0, Q4_0, IQ4_NL, etc.)

**Code Changes**:

```cpp
// src/v2/kernels/KernelFactory.h - Add to KernelFactory class

/**
 * @brief Get or create a row-sliced GEMM kernel for tensor parallelism
 *
 * Creates a kernel that only packs weights for rows [row_start, row_end).
 * Used for row-parallel GEMM where output dimension is partitioned.
 *
 * @param tensor Weight tensor (quantized)
 * @param row_start First row to include (0-indexed)
 * @param row_end One past the last row
 * @return Cached GEMM kernel pointer (owned by factory)
 *
 * @note Cache key includes row range, so different slices get different kernels
 */
static ITensorGemm* getOrCreateGemmSliced(
    const TensorBase* tensor, 
    size_t row_start, 
    size_t row_end);
```

```cpp
// src/v2/kernels/KernelFactory.cpp - Implementation

ITensorGemm* KernelFactory::getOrCreateGemmSliced(
    const TensorBase* tensor, size_t row_start, size_t row_end)
{
    if (!tensor) return nullptr;
    
    // Create cache key that includes slice range
    // Format: tensor_ptr | row_start | row_end
    struct SlicedCacheKey {
        const TensorBase* tensor;
        size_t row_start;
        size_t row_end;
        
        bool operator==(const SlicedCacheKey& other) const {
            return tensor == other.tensor && 
                   row_start == other.row_start && 
                   row_end == other.row_end;
        }
    };
    
    // Hash function for SlicedCacheKey
    struct SlicedKeyHash {
        size_t operator()(const SlicedCacheKey& k) const {
            return std::hash<const void*>()(k.tensor) ^ 
                   (std::hash<size_t>()(k.row_start) << 1) ^
                   (std::hash<size_t>()(k.row_end) << 2);
        }
    };
    
    static std::unordered_map<SlicedCacheKey, std::unique_ptr<ITensorGemm>, SlicedKeyHash> sliced_cache_;
    static std::mutex sliced_cache_mutex_;
    
    SlicedCacheKey key{tensor, row_start, row_end};
    
    std::lock_guard<std::mutex> lock(sliced_cache_mutex_);
    auto it = sliced_cache_.find(key);
    if (it != sliced_cache_.end()) {
        return it->second.get();
    }
    
    // Create sliced kernel using row-range constructor
    auto kernel = std::make_unique<gemm_v4::QuantisedGemmKernel>(
        tensor, static_cast<int>(row_start), static_cast<int>(row_end));
    
    ITensorGemm* raw_ptr = kernel.get();
    sliced_cache_[key] = std::move(kernel);
    
    LOG_DEBUG("[KernelFactory] Created sliced GEMM kernel for rows [" 
              << row_start << ", " << row_end << ")");
    return raw_ptr;
}
```

**Testing**:
```cpp
// tests/v2/unit/Test__KernelFactorySliced.cpp
#include "utils/TestTensorFactory.h"

using namespace llaminar2::test;

TEST(KernelFactorySliced, CreatesSlicedKernel) {
    // Create Q8_0 weight tensor with random data [1024, 896]
    auto weights = TestTensorFactory::createQ8_0Random({1024, 896});
    
    // Create sliced kernel for first half of rows
    auto* kernel = KernelFactory::getOrCreateGemmSliced(weights.get(), 0, 512);
    ASSERT_NE(kernel, nullptr);
    EXPECT_EQ(kernel->get_n(), 512);  // Output dim is sliced
    EXPECT_EQ(kernel->get_k(), 896);  // Input dim unchanged
}

TEST(KernelFactorySliced, CachesSlicedKernels) {
    auto weights = TestTensorFactory::createQ8_0Random({1024, 896});
    auto* k1 = KernelFactory::getOrCreateGemmSliced(weights.get(), 0, 512);
    auto* k2 = KernelFactory::getOrCreateGemmSliced(weights.get(), 0, 512);
    EXPECT_EQ(k1, k2);  // Same slice = same kernel
    
    auto* k3 = KernelFactory::getOrCreateGemmSliced(weights.get(), 512, 1024);
    EXPECT_NE(k1, k3);  // Different slice = different kernel
}
```

---

### 2.2-2.3 Extend GEMMStage for TP ✅ IMPLEMENTED

**Files Modified**:
- `src/v2/execution/ComputeStage.h`

**Rationale**: `GEMMStage` currently has `mpi_ctx` but doesn't use it for sharding. We add optional `WorkRange` parameters that, when set, trigger sliced kernel usage.

**Actual Implementation** (December 2025):
- Added `output_range` (optional `WorkRange`) to `GEMMStage::Params`
- Added `needs_allreduce` flag for post-GEMM collective
- `execute()` checks for `output_range` and calls `KernelFactory::getOrCreateGemmSliced()`
- Local GEMM produces `[m, local_n]` output; caller responsible for AllGather

**Code Changes**:

```cpp
// src/v2/execution/ComputeStage.h - Extend GEMMStage::Params

struct Params {
    // ... existing fields ...
    TensorBase* A = nullptr;      // Input activation [m, k]
    TensorBase* B = nullptr;      // Weight tensor [n, k] or [k, n]
    TensorBase* C = nullptr;      // Output [m, n]
    int m = 0, n = 0, k = 0;
    float alpha = 1.0f, beta = 0.0f;
    bool transpose_B = true;
    
    // Existing MPI context (for collective ops)
    std::shared_ptr<MPIContext> mpi_ctx;
    int device_idx = 0;
    
    // === NEW: Tensor Parallelism Parameters ===
    
    /**
     * @brief Output dimension range for row-parallel GEMM
     * 
     * When set, the kernel only computes output columns [start, end).
     * Requires weight tensor to be full (not pre-sliced) OR TensorSlice.
     * 
     * After local GEMM: output has shape [m, range.size()]
     * Set needs_allgather=true to reconstruct full output.
     */
    std::optional<WorkRange> output_range;
    
    /**
     * @brief Input dimension range for column-parallel GEMM
     * 
     * When set, input is already sliced to [m, range.size()].
     * Weight should be sliced to [n, range.size()].
     * 
     * After local GEMM: output is partial sum, needs allreduce.
     * Set needs_allreduce=true.
     */
    std::optional<WorkRange> input_range;
    
    /**
     * @brief Whether to AllReduce output after GEMM
     * 
     * Used for column-parallel GEMM where each rank computes
     * partial products that sum to the final result.
     */
    bool needs_allreduce = false;
    
    /**
     * @brief Whether to AllGather output after GEMM
     * 
     * Used for row-parallel GEMM where each rank computes
     * a slice of the output that needs concatenation.
     */
    bool needs_allgather = false;
};
```

```cpp
// src/v2/execution/ComputeStage.cpp - Modify GEMMStage::execute()

bool GEMMStage::execute(IDeviceContext* ctx) {
    // ... existing validation ...
    
    ITensorGemm* gemm = nullptr;
    int local_n = params_.n;
    
    // Check for row-parallel (output dimension sharded)
    if (params_.output_range && params_.mpi_ctx && params_.mpi_ctx->world_size() > 1) {
        const auto& range = *params_.output_range;
        local_n = static_cast<int>(range.size());
        
        // Use sliced kernel
        gemm = KernelFactory::getOrCreateGemmSliced(
            params_.B, range.start, range.end);
        
        LOG_DEBUG("[GEMMStage] Row-parallel: output_range=[" << range.start 
                  << ", " << range.end << "), local_n=" << local_n);
    } else {
        // Full kernel (existing path)
        gemm = KernelFactory::getOrCreateGemm(params_.B);
    }
    
    if (!gemm) {
        LOG_ERROR("[GEMMStage] Failed to get GEMM kernel");
        return false;
    }
    
    // Execute local GEMM
    // For row-parallel: A[m,k] @ B_slice[local_n,k]^T = C_local[m,local_n]
    bool success = gemm->multiply_tensor(
        params_.A, params_.C,
        params_.m, local_n, params_.k,
        params_.transpose_B,
        params_.alpha, params_.beta,
        params_.mpi_ctx.get(), params_.device_idx);
    
    if (!success) {
        LOG_ERROR("[GEMMStage] GEMM execution failed");
        return false;
    }
    
    // Post-GEMM collective operations
    if (params_.needs_allreduce && params_.mpi_ctx) {
        // Column-parallel: sum partial products
        float* data = params_.C->mutable_data();
        size_t count = static_cast<size_t>(params_.m) * params_.n;
        params_.mpi_ctx->allreduce_sum(data, count);
        LOG_DEBUG("[GEMMStage] AllReduce completed");
    }
    
    if (params_.needs_allgather && params_.mpi_ctx) {
        // Row-parallel: gather output slices
        // Note: This requires output buffer to be full size
        // The local result is in the first local_n columns
        // AllGatherv to assemble full output
        
        // Implementation depends on output tensor layout
        // For now, assume separate AllGatherStage handles this
        LOG_WARN("[GEMMStage] needs_allgather=true but inline gather not implemented. "
                 "Use AllGatherStage after this stage.");
    }
    
    return true;
}
```

**Testing**:
```cpp
// tests/v2/unit/Test__GEMMStageTP.cpp
#include "utils/TestTensorFactory.h"

using namespace llaminar2::test;

TEST(GEMMStageTP, RowParallelExecutesSlicedKernel) {
    // Simulate 2-rank environment
    int world_size = 2, rank = 0;
    auto mpi_ctx = std::make_shared<MPIContext>(rank, world_size, MPI_COMM_WORLD);
    
    // Create tensors using TestTensorFactory
    auto input = TestTensorFactory::createFP32Random({32, 896});    // [m, k]
    auto weights = TestTensorFactory::createQ8_0Random({1024, 896}); // [n, k]
    auto output = TestTensorFactory::createFP32Zeros({32, 512});     // [m, local_n]
    
    GEMMStage::Params params;
    params.A = input.get();
    params.B = weights.get();
    params.C = output.get();
    params.m = 32;
    params.n = 1024;  // Full N
    params.k = 896;
    params.mpi_ctx = mpi_ctx;
    params.output_range = WorkRange{0, 512};  // Rank 0 gets first half
    
    GEMMStage stage(params);
    auto* ctx = getCPUContext();
    
    ASSERT_TRUE(stage.execute(ctx));
    
    // Verify output dimensions
    EXPECT_EQ(output->shape()[0], 32);
    EXPECT_EQ(output->shape()[1], 512);
    
    // Verify output has values and no NaN/Inf
    EXPECT_FALSE(TestTensorFactory::hasNaNOrInf(output.get()));
    
    // Verify output is not all zeros
    float sum = 0;
    for (size_t i = 0; i < output->numel(); ++i) {
        sum += std::abs(output->data()[i]);
    }
    EXPECT_GT(sum, 0.0f);
}
```

---

### 2.4-2.6 Extend AttentionComputeStage for TP ✅ IMPLEMENTED

**Files Modified**:
- `src/v2/execution/ComputeStage.h` (AttentionWithKVCacheStage::Params)
- `src/v2/tensors/TensorKernels.h` (ITensorAttention::compute_tensor)
- `src/v2/kernels/cpu/attention/CPUAttentionKernelTyped.h`

**Rationale**: Attention is embarrassingly parallel across heads. Each rank should only compute attention for its assigned heads.

**Actual Implementation** (December 2025):
- Added `head_start`, `local_n_heads`, `local_n_kv_heads` to `AttentionWithKVCacheStage::Params`
- Extended `ITensorAttention::compute_tensor()` interface with head range parameters (defaults for compatibility)
- `CPUAttentionKernelTyped::compute_tensor()` updated with validation and stub (logs warning if TP requested)
- Inner head loops still use full range; full implementation deferred to Phase 3 when end-to-end TP is tested

**Code Changes**:

```cpp
// src/v2/execution/ComputeStage.h - Extend AttentionComputeStage::Params

struct Params {
    // ... existing fields ...
    TensorBase* Q = nullptr;
    TensorBase* K = nullptr;
    TensorBase* V = nullptr;
    TensorBase* output = nullptr;
    int batch_size = 1;
    int seq_len = 0;
    int kv_len = 0;
    int n_heads = 0;
    int n_kv_heads = 0;
    int head_dim = 0;
    bool causal = true;
    
    // Existing
    std::shared_ptr<MPIContext> mpi_ctx;
    int device_idx = 0;
    
    // === NEW: Tensor Parallelism Parameters ===
    
    /**
     * @brief First query head for this rank (0-indexed)
     * 
     * Default 0 means start from head 0.
     * With TP: each rank gets a contiguous range of heads.
     */
    int head_start = 0;
    
    /**
     * @brief Number of query heads for this rank
     * 
     * Default -1 means use full n_heads.
     * With TP: set to n_heads / world_size (approximately).
     */
    int local_n_heads = -1;
    
    /**
     * @brief Number of KV heads for this rank
     * 
     * Default -1 means use full n_kv_heads.
     * For GQA: may equal local_n_heads / gqa_ratio.
     * 
     * Note: If n_kv_heads < world_size, some ranks may have 0 KV heads
     * and must skip attention entirely or participate in collective only.
     */
    int local_n_kv_heads = -1;
};
```

```cpp
// src/v2/tensors/TensorKernels.h - Update ITensorAttention interface

class ITensorAttention {
public:
    virtual ~ITensorAttention() = default;
    
    /**
     * @brief Compute attention with tensor inputs
     *
     * @param Q Query tensor [batch, seq_len, n_heads, head_dim]
     * @param K Key tensor [batch, kv_len, n_kv_heads, head_dim]
     * @param V Value tensor [batch, kv_len, n_kv_heads, head_dim]
     * @param output Output tensor [batch, seq_len, n_heads, head_dim]
     * @param batch_size Batch size
     * @param seq_len Query sequence length
     * @param kv_len Key/Value sequence length
     * @param n_heads Total number of query heads (for stride calculation)
     * @param n_kv_heads Total number of KV heads
     * @param head_dim Head dimension
     * @param causal Apply causal masking
     * @param window_size Sliding window size (-1 for full attention)
     * @param workspace_scores Workspace for attention scores
     * @param mask Optional attention mask
     * @param mpi_ctx MPI context (optional)
     * @param device_idx Target device
     * @param head_start First head to compute (default 0)
     * @param local_n_heads Number of heads to compute (-1 = all)
     * @param local_n_kv_heads Number of KV heads for this slice (-1 = all)
     */
    virtual bool compute_tensor(
        const TensorBase* Q, const TensorBase* K, const TensorBase* V,
        TensorBase* output,
        int batch_size, int seq_len, int kv_len,
        int n_heads, int n_kv_heads, int head_dim,
        bool causal, int window_size,
        TensorBase* workspace_scores,
        TensorBase* mask,
        const MPIContext* mpi_ctx,
        int device_idx,
        int head_start = 0,          // NEW
        int local_n_heads = -1,      // NEW
        int local_n_kv_heads = -1    // NEW
    ) = 0;
};
```

```cpp
// src/v2/kernels/cpu/attention/CPUAttentionKernelTyped.h
// Modify the compute loop to use head range

template<ActivationPrecision PREC>
bool CPUAttentionKernelTyped<PREC>::compute_tensor(
    const TensorBase* Q, const TensorBase* K, const TensorBase* V,
    TensorBase* output,
    int batch_size, int seq_len, int kv_len,
    int n_heads, int n_kv_heads, int head_dim,
    bool causal, int window_size,
    TensorBase* workspace_scores,
    TensorBase* mask,
    const MPIContext* mpi_ctx,
    int device_idx,
    int head_start,
    int local_n_heads,
    int local_n_kv_heads)
{
    // Resolve defaults
    const int actual_local_n_heads = (local_n_heads < 0) ? n_heads : local_n_heads;
    const int actual_local_n_kv_heads = (local_n_kv_heads < 0) ? n_kv_heads : local_n_kv_heads;
    const int head_end = head_start + actual_local_n_heads;
    
    // GQA ratio (how many Q heads per KV head)
    const int gqa_ratio = n_heads / n_kv_heads;
    
    LOG_DEBUG("[CPUAttention] head_range=[" << head_start << ", " << head_end 
              << "), local_n_kv_heads=" << actual_local_n_kv_heads);
    
    // Get typed data pointers
    const DataT* q_data = getTypedData<DataT>(Q);
    const DataT* k_data = getTypedData<DataT>(K);
    const DataT* v_data = getTypedData<DataT>(V);
    DataT* out_data = getMutableTypedData<DataT>(output);
    
    // === MODIFIED: Loop only over local heads ===
    #pragma omp parallel for collapse(2) schedule(static)
    for (int b = 0; b < batch_size; ++b) {
        for (int h = head_start; h < head_end; ++h) {
            // Map query head to KV head (GQA)
            // KV head index relative to this rank's slice
            int global_kv_head = h / gqa_ratio;
            int local_kv_head = global_kv_head - (head_start / gqa_ratio);
            
            // Bounds check for edge cases
            if (local_kv_head < 0 || local_kv_head >= actual_local_n_kv_heads) {
                LOG_ERROR("[CPUAttention] KV head out of range: local_kv_head=" 
                          << local_kv_head << " for query head " << h);
                continue;
            }
            
            // Compute attention for this head
            compute_single_head(
                q_data, k_data, v_data, out_data,
                b, h, local_kv_head,
                seq_len, kv_len, head_dim,
                n_heads, actual_local_n_kv_heads,  // Strides based on local dims
                causal, window_size,
                workspace_scores, mask);
        }
    }
    
    return true;
}
```

**Testing**:
```cpp
// tests/v2/unit/Test__AttentionStageTP.cpp
#include "utils/TestTensorFactory.h"

using namespace llaminar2::test;

TEST(AttentionStageTP, ComputesLocalHeadsOnly) {
    // 14 heads, 2 ranks -> 7 heads each
    constexpr int N_HEADS = 14, N_KV_HEADS = 2, HEAD_DIM = 64, SEQ_LEN = 32;
    constexpr int head_start = 0, local_n_heads = 7;
    
    int q_dim = N_HEADS * HEAD_DIM;
    int kv_dim = N_KV_HEADS * HEAD_DIM;
    int local_out_dim = local_n_heads * HEAD_DIM;
    
    // Create tensors - Q/K/V with full dimensions, output only for local heads
    auto Q = TestTensorFactory::createFP32Random({1, SEQ_LEN, static_cast<size_t>(q_dim)});
    auto K = TestTensorFactory::createFP32Random({1, SEQ_LEN, static_cast<size_t>(kv_dim)});
    auto V = TestTensorFactory::createFP32Random({1, SEQ_LEN, static_cast<size_t>(kv_dim)});
    auto output = TestTensorFactory::createFP32Zeros({1, SEQ_LEN, static_cast<size_t>(local_out_dim)});
    
    AttentionComputeStage::Params params;
    params.Q = Q.get();
    params.K = K.get();
    params.V = V.get();
    params.output = output.get();
    params.batch_size = 1;
    params.seq_len = SEQ_LEN;
    params.kv_len = SEQ_LEN;
    params.n_heads = N_HEADS;
    params.n_kv_heads = N_KV_HEADS;
    params.head_dim = HEAD_DIM;
    params.causal = true;
    params.head_start = head_start;
    params.local_n_heads = local_n_heads;
    params.local_n_kv_heads = 1;  // GQA: 7 Q heads share 1 KV head
    
    AttentionComputeStage stage(params);
    ASSERT_TRUE(stage.execute(getCPUContext()));
    
    // Verify output shape and validity
    EXPECT_EQ(output->shape()[2], local_out_dim);
    EXPECT_FALSE(TestTensorFactory::hasNaNOrInf(output.get()));
}
```

### 2.7 Unit Tests ✅ IMPLEMENTED

**Test File**: `tests/v2/unit/Test__KernelFactorySliced.cpp`

**Implemented Tests** (11 total):
1. `CreateSlicedKernel` - Basic sliced kernel creation
2. `CachesSlicedKernels` - Same slice returns same kernel
3. `DifferentSlicesDifferentKernels` - Different ranges cache separately
4. `FullRangeMatchesUnsliced` - Full slice equivalent to regular kernel
5. `SlicedGemmProducesCorrectOutput` - Numerical correctness
6. `MultipleSlicesConsistent` - Concatenated slices match full output
7. `EdgeCasesSingleRow` - Single-row slices work
8. `ClearCacheForTensor` - Cache invalidation on tensor destruction
9. `ThreadSafeAccess` - Concurrent cache access
10. `Q4_0SlicedKernel` - Q4_0 format support
11. `IQ4_NLSlicedKernel` - IQ4_NL format support

**Attention TP Tests**: Deferred until full head-sliced attention implementation (Phase 3 end-to-end testing)

---

## Phase 3: Column-Parallel QKV

**Proposal Reference**: Section "Migration Path → Phase 3"

**Goal**: Each rank computes Q/K/V only for its assigned heads, reducing redundant computation by `world_size`.

### 3.1 Column-Parallel Weight Loading

**Files to Modify**:
- `src/v2/loaders/WeightManager.h`
- `src/v2/loaders/WeightManager.cpp`

**Approach**: When loading attention weights (Wq, Wk, Wv), slice by output dimension based on rank's head range.

```cpp
// src/v2/loaders/WeightManager.h - Add method

/**
 * @brief Load attention weight with column-parallel sharding
 *
 * @param layer_idx Layer index
 * @param weight_name "wq", "wk", or "wv"
 * @param head_range Range of heads for this rank
 * @param head_dim Dimension per head
 * @return Sliced weight tensor [d_model, local_heads * head_dim]
 */
std::unique_ptr<TensorBase> getColumnShardedAttentionWeight(
    int layer_idx,
    const std::string& weight_name,
    const WorkRange& head_range,
    int head_dim);
```

```cpp
// src/v2/loaders/WeightManager.cpp

std::unique_ptr<TensorBase> WeightManager::getColumnShardedAttentionWeight(
    int layer_idx, const std::string& weight_name,
    const WorkRange& head_range, int head_dim)
{
    // Load full weight
    auto full_weight = getLayerWeight(layer_idx, weight_name);
    if (!full_weight) return nullptr;
    
    // Full shape: [n_heads * head_dim, d_model] for Wq/Wk/Wv
    // We want columns [head_range.start * head_dim, head_range.end * head_dim)
    size_t col_start = head_range.start * head_dim;
    size_t col_end = head_range.end * head_dim;
    
    // Create TensorSlice with column-parallel metadata
    auto meta = SliceMetadata::forColumnParallel(
        full_weight->shape()[0],  // rows (d_model)
        full_weight->shape()[1],  // cols (n_heads * head_dim)
        col_start, col_end,
        mpi_ctx_->rank(), mpi_ctx_->world_size());
    
    return std::make_unique<TensorSlice>(std::move(full_weight), meta);
}
```

### 3.2 Update Qwen2Graph for Head Sharding

**Files to Modify**:
- `src/v2/pipelines/qwen/Qwen2Graph.cpp`

```cpp
// src/v2/pipelines/qwen/Qwen2Graph.cpp - In buildAttentionGraph()

void Qwen2Graph::buildAttentionGraph(int layer_idx, ComputeGraph& graph) {
    // Get work range for this rank
    WorkRange head_range = mpi_topology_->get_head_range(config_.n_heads);
    WorkRange kv_head_range = mpi_topology_->get_kv_head_range(config_.n_kv_heads);
    
    int local_n_heads = static_cast<int>(head_range.size());
    int local_n_kv_heads = static_cast<int>(kv_head_range.size());
    
    // Q projection: [m, d_model] @ [d_model, local_n_heads * head_dim]
    auto q_params = GEMMStage::Params{};
    q_params.A = buffers_.hidden;
    q_params.B = weights_.layers[layer_idx].wq;  // Should be column-sharded
    q_params.C = buffers_.q_local;               // [m, local_n_heads * head_dim]
    q_params.m = seq_len_;
    q_params.n = local_n_heads * config_.head_dim;
    q_params.k = config_.d_model;
    q_params.mpi_ctx = mpi_ctx_;
    // No output_range needed - weight is already column-sliced
    
    graph.addStage(std::make_unique<GEMMStage>(q_params), "q_proj");
    
    // K/V projections similar...
    
    // Attention with local heads
    auto attn_params = AttentionComputeStage::Params{};
    attn_params.Q = buffers_.q_local;
    attn_params.K = buffers_.k_local;
    attn_params.V = buffers_.v_local;
    attn_params.output = buffers_.attn_output_local;
    attn_params.n_heads = config_.n_heads;      // Global for stride calculation
    attn_params.n_kv_heads = config_.n_kv_heads;
    attn_params.head_start = static_cast<int>(head_range.start);
    attn_params.local_n_heads = local_n_heads;
    attn_params.local_n_kv_heads = local_n_kv_heads;
    // ...
    
    graph.addStage(std::make_unique<AttentionComputeStage>(attn_params), "attention");
    
    // Wo projection: [m, local_n_heads * head_dim] @ [d_model, n_heads * head_dim]^T
    // This is row-parallel on Wo (input is sharded)
    auto wo_params = GEMMStage::Params{};
    wo_params.A = buffers_.attn_output_local;
    wo_params.B = weights_.layers[layer_idx].wo;  // Full Wo, row-sliced access
    wo_params.C = buffers_.attn_proj;
    wo_params.m = seq_len_;
    wo_params.n = config_.d_model;
    wo_params.k = local_n_heads * config_.head_dim;  // Local K
    wo_params.input_range = WorkRange{
        head_range.start * config_.head_dim,
        head_range.end * config_.head_dim
    };
    wo_params.needs_allreduce = true;  // Sum partial products
    
    graph.addStage(std::make_unique<GEMMStage>(wo_params), "wo_proj");
}
```

---

## Phase 4: Column-Parallel FFN

**Proposal Reference**: Section "Migration Path → Phase 4"

**Goal**: Shard Gate/Up weights by output dimension (d_ff), reducing per-rank memory.

### 4.1-4.3 FFN Graph with TP

```cpp
// src/v2/pipelines/qwen/Qwen2Graph.cpp - In buildFFNGraph()

void Qwen2Graph::buildFFNGraph(int layer_idx, ComputeGraph& graph) {
    // Get FFN dimension range
    WorkRange ffn_range = mpi_topology_->get_column_range(config_.d_ff);
    int local_d_ff = static_cast<int>(ffn_range.size());
    
    // Gate: [m, d_model] @ [d_model, local_d_ff]
    auto gate_params = GEMMStage::Params{};
    gate_params.A = buffers_.ffn_norm_output;
    gate_params.B = weights_.layers[layer_idx].gate;  // Column-sharded
    gate_params.C = buffers_.gate_local;              // [m, local_d_ff]
    gate_params.m = seq_len_;
    gate_params.n = local_d_ff;
    gate_params.k = config_.d_model;
    gate_params.mpi_ctx = mpi_ctx_;
    
    graph.addStage(std::make_unique<GEMMStage>(gate_params), "ffn_gate");
    
    // Up: [m, d_model] @ [d_model, local_d_ff]
    auto up_params = GEMMStage::Params{};
    up_params.A = buffers_.ffn_norm_output;
    up_params.B = weights_.layers[layer_idx].up;
    up_params.C = buffers_.up_local;
    up_params.m = seq_len_;
    up_params.n = local_d_ff;
    up_params.k = config_.d_model;
    up_params.mpi_ctx = mpi_ctx_;
    
    graph.addStage(std::make_unique<GEMMStage>(up_params), "ffn_up");
    
    // SwiGLU: local computation, no communication
    auto swiglu_params = SwiGLUStage::Params{};
    swiglu_params.gate = buffers_.gate_local;
    swiglu_params.up = buffers_.up_local;
    swiglu_params.output = buffers_.swiglu_local;
    swiglu_params.rows = seq_len_;
    swiglu_params.cols = local_d_ff;
    
    graph.addStage(std::make_unique<SwiGLUStage>(swiglu_params), "swiglu");
    
    // Down: [m, local_d_ff] @ [d_model, d_ff]^T with input_range
    // This is column-parallel: each rank has partial input, needs allreduce
    auto down_params = GEMMStage::Params{};
    down_params.A = buffers_.swiglu_local;
    down_params.B = weights_.layers[layer_idx].down;  // Full Down weight
    down_params.C = buffers_.ffn_output;
    down_params.m = seq_len_;
    down_params.n = config_.d_model;
    down_params.k = local_d_ff;
    down_params.input_range = ffn_range;
    down_params.needs_allreduce = true;  // Sum partial products across ranks
    down_params.mpi_ctx = mpi_ctx_;
    
    graph.addStage(std::make_unique<GEMMStage>(down_params), "ffn_down");
}
```

---

## Phase 5: Column-Parallel LM Head ✅ COMPLETE

**Proposal Reference**: Section "Migration Path → Phase 5"  
**Status**: ✅ Implemented December 22, 2025

**Goal**: Avoid computing full 151K vocab logits on every rank. Each rank computes `vocab_local = vocab_size / world_size` logits, then AllGather combines them.

### Implementation Summary

| Task | Status | Files Modified |
|------|--------|----------------|
| 5.1 LM Head Sharding | ✅ | `WeightManager.cpp` |
| 5.2 AllGatherStage | ✅ | `ComputeStage.h/.cpp` |
| 5.3 Graph Integration | ✅ | `Qwen2Graph.h/.cpp`, `GraphOrchestrator.cpp`, `InferenceRunner.cpp` |
| 5.4 Unit Tests | ✅ | `Test__AllGatherStage.cpp` (8 tests) |
| 5.5 Integration Tests | ✅ | `Test__MPI_ColumnParallelLMHead.cpp` (5 tests) |

### Key Changes

**WeightManager.cpp** - Added column-parallel sharding for LM head:
```cpp
if (name == "output.weight") {
    return ShardingMode::COLUMN_PARALLEL;  // Split vocab dimension
}
```

**ComputeStage.h/.cpp** - Created AllGatherStage:
```cpp
class AllGatherStage : public IComputeStage {
public:
    struct Params {
        TensorBase* local_input = nullptr;   // [seq_len, vocab_local]
        TensorBase* full_output = nullptr;   // [seq_len, vocab_size]
        void* mpi_comm = nullptr;
        int world_size = 1;
    };
    bool execute(IDeviceContext* ctx) override;  // Uses MPI_Allgather
};
```

**Qwen2Graph.h/.cpp** - Added config and buffer support:
```cpp
struct Qwen2GraphConfig {
    // Phase 5: Column-parallel LM head
    int vocab_local = 0;              // vocab_size / world_size when sharded
    bool lm_head_column_parallel = false;
};

struct Qwen2ModelBuffers {
    std::shared_ptr<TensorBase> logits_local;  // For column-parallel LM head
};
```

**buildLMHeadGraph()** now:
1. Uses `vocab_local` dimension when `lm_head_column_parallel=true`
2. Adds `AllGatherStage` after LM head GEMM to collect distributed logits

### Test Coverage

**Unit Tests** (`Test__AllGatherStage.cpp`):
- StageTypeIsAllGather
- SupportsAllBackends (CPU, CUDA, ROCm, Vulkan)
- FailsWithNullLocalInput, FailsWithNullFullOutput, FailsWithNullMpiComm
- FailsWithInvalidWorldSize
- BufferRequirementsAreCorrect
- FactoryCreatesStage

**Integration Tests** (`Test__MPI_ColumnParallelLMHead.cpp`):
- ShardingModeDetection - Verifies output.weight uses COLUMN_PARALLEL
- LocalVocabDimensionCalculation - Verifies vocab_local = vocab_size / world_size
- WeightShapeValidation - Verifies sharded weight shape [vocab_local, d_model]
- AllGatherStageIntegration - Verifies AllGather collects distributed logits correctly
- FullForwardDataFlow - Full pipeline with distributed LM head and argmax consistency

```cpp
// src/v2/pipelines/qwen/Qwen2Graph.cpp - In buildLMHeadGraph()

void Qwen2Graph::buildLMHeadGraph(ComputeGraph& graph) {
    // Get vocab range for this rank
    WorkRange vocab_range = mpi_topology_->get_vocab_range(config_.vocab_size);
    int local_vocab = static_cast<int>(vocab_range.size());
    
    // LM Head: [m, d_model] @ [d_model, local_vocab]
    auto lm_params = GEMMStage::Params{};
    lm_params.A = buffers_.final_norm_output;
    lm_params.B = weights_.lm_head;  // Column-sharded by vocab
    lm_params.C = buffers_.logits_local;
    lm_params.m = seq_len_;
    lm_params.n = local_vocab;
    lm_params.k = config_.d_model;
    lm_params.mpi_ctx = mpi_ctx_;
    
    graph.addStage(std::make_unique<GEMMStage>(lm_params), "lm_head");
    
    // AllGather logits for sampling
    auto gather_params = AllGatherStage::Params{};
    gather_params.input = buffers_.logits_local;
    gather_params.output = buffers_.logits_full;
    gather_params.local_size = seq_len_ * local_vocab;
    gather_params.mpi_ctx = mpi_ctx_;
    
    graph.addStage(std::make_unique<AllGatherStage>(gather_params), "gather_logits");
}
```

---

## Phase 6: Sharded KV Cache

**Proposal Reference**: Section "Migration Path → Phase 6"  
**Status**: ✅ COMPLETE (January 2025)

**Goal**: When using column-parallel QKV (Phase 3), each rank only computes local K/V heads. The KV cache should only store these local heads rather than full heads, reducing memory usage proportionally.

### Implementation Progress

#### 6.1 UnifiedKVCache Sharding Support ✅ COMPLETE

**Files Modified**:
- `src/v2/tensors/UnifiedKVCache.h`
- `src/v2/tensors/UnifiedKVCache.cpp`

**Changes**:
1. Extended `IUnifiedKVCache` interface with sharding methods:
   - `is_sharded()` - Returns true if using local KV heads
   - `n_kv_heads()` - Total KV heads across all ranks
   - `local_n_kv_heads()` - KV heads on this rank
   - `kv_head_start()` - Starting head index for this rank
   - `local_kv_dim()` - local_n_kv_heads * head_dim

2. Added sharded constructors to `UnifiedKVCache<Precision>`:
   ```cpp
   // Sharded with default device
   UnifiedKVCache(const MPIContext& mpi_ctx, int n_layers, int batch_size, int max_seq_len,
                  int n_kv_heads, int local_n_kv_heads, int kv_head_start,
                  int head_dim, int device_idx = -1);
   
   // Sharded with per-layer devices
   UnifiedKVCache(const MPIContext& mpi_ctx, int n_layers, int batch_size, int max_seq_len,
                  int n_kv_heads, int local_n_kv_heads, int kv_head_start,
                  int head_dim, const std::vector<int>& attention_devices);
   ```

3. Added factory functions:
   ```cpp
   std::unique_ptr<IUnifiedKVCache> createShardedKVCache(
       ActivationPrecision precision,
       const MPIContext& mpi_ctx,
       int n_layers, int batch_size, int max_seq_len,
       int n_kv_heads, int local_n_kv_heads, int kv_head_start,
       int head_dim, int device_idx = -1);
   
   std::unique_ptr<IUnifiedKVCache> createShardedKVCache(
       ActivationPrecision precision,
       const MPIContext& mpi_ctx,
       int n_layers, int batch_size, int max_seq_len,
       int n_kv_heads, int local_n_kv_heads, int kv_head_start,
       int head_dim, const std::vector<int>& attention_devices);
   ```

4. Updated non-sharded constructors to initialize sharding fields with defaults:
   - `is_sharded_ = false`
   - `local_n_kv_heads_ = n_kv_heads_`
   - `kv_head_start_ = 0`

#### 6.5 Unit Tests ✅ COMPLETE

**File**: `tests/v2/unit/Test__ShardedKVCache.cpp`

13 tests covering:
- Non-sharded cache backward compatibility
- Sharded cache metadata (is_sharded, n_kv_heads, local_n_kv_heads, kv_head_start)
- Tensor shape verification (local KV dim)
- Memory savings (50% reduction with 2-way sharding)
- Append/clear operations
- All precisions (FP32, BF16, FP16, Q8_1)
- Per-layer device placement

#### 6.6 MPI Integration Tests ✅ COMPLETE

**File**: `tests/v2/integration/Test__MPI_ShardedKVCache.cpp`

7 tests with 2 MPI ranks covering:
- Each rank creates independent local sharded cache
- Tensor shapes are local on each rank
- Append/clear work independently (no MPI communication)
- Memory usage reduced proportionally across ranks
- All precisions work in MPI environment
- Sequential decode simulation (prefill + decode)

### Design Overview

With column-parallel QKV:
- Each rank computes `local_n_kv_heads = n_kv_heads / world_size` KV heads
- K/V projections output `[seq_len, local_n_kv_heads * head_dim]`
- KV cache should store `[max_seq_len, local_n_kv_heads * head_dim]` per layer

**Memory Savings** (Qwen 2.5 0.5B with 2 ranks):
- Full KV cache: `[4096, 2 * 64]` = 512KB per layer per sequence
- Sharded: `[4096, 1 * 64]` = 256KB per layer per sequence (50% reduction)

### Completed Work

**6.2 KVCacheAppendStage** ✅ COMPLETE (No Changes Needed)
- Stage derives `kv_dim` from input tensor shape (`params_.K->shape()[1]`)
- Works automatically with sharded K/V tensors

**6.3 KVCacheGatherStage** ✅ COMPLETE (No Changes Needed)
- Delegates to `gather_kv_batched()` which validates against cache's internal `kv_dim_`
- Output uses local dimensions automatically

**6.4 Pipeline Integration** ✅ COMPLETE
- `PipelineBase::initializeKVCache()` creates sharded cache when `mpi_strategy_ == TensorParallel`
- `GraphOrchestrator::initializeInferenceState()` creates sharded cache when `local_n_kv_heads` set

### Key Insight: No MPI Communication Needed

Unlike LM head (which requires AllGather), sharded KV cache does NOT need MPI communication:
- Each rank computes its local Q/K/V heads
- Each rank stores its local K/V in cache
- Attention computation is local (Q_local @ K_local^T → scores → softmax → @ V_local)
- Only the attention output projection (Wo) needs AllReduce (already handled in Phase 3)

```cpp
// After Phase 6, the flow is:
// Rank 0:
//   Q0, K0, V0 = local_qkv_proj(hidden)  // [seq, local_kv_heads * head_dim]
//   kv_cache.append(K0, V0)               // Local cache only
//   attn0 = attention(Q0, K0, V0)         // Local computation
//   output = Wo @ attn0                    // Sharded Wo, needs AllReduce
// Rank 1:
//   Q1, K1, V1 = local_qkv_proj(hidden)  // Different heads
//   kv_cache.append(K1, V1)               // Local cache only
//   attn1 = attention(Q1, K1, V1)         // Local computation
//   output = Wo @ attn1                    // AllReduce combines
```

---

## GPU Integration Phases

### Phase G1: Single-GPU Per Rank

**Proposal Reference**: Section "GPU Integration Phases → Phase G1"

**Goal**: Each MPI rank uses its NUMA-local GPU for GEMM.

**Files to Modify**:
- `src/v2/pipelines/qwen/GraphOrchestrator.cpp`

```cpp
// GraphOrchestrator initialization - select GPU device

void GraphOrchestrator::initializeDevices() {
    auto& dm = DeviceManager::instance();
    
    // Find GPU device for this rank (NUMA-filtered)
    int gpu_idx = dm.find_device(ComputeBackendType::GPU_CUDA, 0);  // First CUDA
    if (gpu_idx < 0) {
        gpu_idx = dm.find_device(ComputeBackendType::GPU_ROCM, 0);  // Try ROCm
    }
    
    if (gpu_idx >= 0) {
        device_idx_ = gpu_idx;
        LOG_INFO("[GraphOrchestrator] Using GPU device " << gpu_idx 
                 << " (" << dm.devices()[gpu_idx].name << ")");
    } else {
        device_idx_ = DeviceManager::cpuDeviceIndex();
        LOG_INFO("[GraphOrchestrator] No GPU found, using CPU");
    }
}
```

### Phase G2: GPU Attention Kernels

**Files to Create**:
- `src/v2/kernels/cuda/CudaFlashAttentionKernel.h`
- `src/v2/kernels/cuda/CudaFlashAttentionKernel.cu`

```cpp
// src/v2/kernels/cuda/CudaFlashAttentionKernel.h

#pragma once

#ifdef HAVE_CUDA

#include "../../tensors/TensorKernels.h"

namespace llaminar::v2::kernels::cuda {

/**
 * @brief FlashAttention-2 based CUDA attention kernel
 *
 * Implements efficient attention using tiled computation
 * to minimize HBM reads/writes.
 */
class CudaFlashAttentionKernel : public llaminar2::ITensorAttention {
public:
    bool compute_tensor(
        const TensorBase* Q, const TensorBase* K, const TensorBase* V,
        TensorBase* output,
        int batch_size, int seq_len, int kv_len,
        int n_heads, int n_kv_heads, int head_dim,
        bool causal, int window_size,
        TensorBase* workspace_scores,
        TensorBase* mask,
        const MPIContext* mpi_ctx,
        int device_idx,
        int head_start = 0,
        int local_n_heads = -1,
        int local_n_kv_heads = -1) override;
        
private:
    // FlashAttention parameters
    float softmax_scale_ = 0.0f;  // Computed from head_dim
};

} // namespace llaminar::v2::kernels::cuda

#endif // HAVE_CUDA
```

**Update KernelFactory**:
```cpp
// src/v2/kernels/KernelFactory.cpp

#ifdef HAVE_CUDA
#include "cuda/CudaAttentionKernelTyped.h"
#endif

#ifdef HAVE_ROCM
#include "rocm/RocmAttentionKernelTyped.h"
#endif

/**
 * @brief Create attention kernel with automatic type dispatch
 *
 * All backends use the same ActivationPrecision enum pattern:
 * - CPU: CPUAttentionKernelTyped<ActivationPrecision::XXX>
 * - CUDA: CudaAttentionKernelTyped<ActivationPrecision::XXX>
 * - ROCm: RocmAttentionKernelTyped<ActivationPrecision::XXX>
 *
 * This ensures consistent type handling across all devices.
 *
 * @param tensor The activation tensor (Q/K/V) - used to determine precision
 * @param dev_type Target device (CPU, CUDA, ROCm)
 */
std::unique_ptr<ITensorAttention> KernelFactory::createAttention(
    const TensorBase* tensor, DeviceType dev_type)
{
    // Map TensorType → ActivationPrecision
    ActivationPrecision prec = tensorTypeToActivationPrecision(tensor->dtype());
    
    switch (dev_type) {
#ifdef HAVE_CUDA
    case DeviceType::CUDA:
        return createCudaAttention(prec);
#endif

#ifdef HAVE_ROCM
    case DeviceType::ROCm:
        return createRocmAttention(prec);
#endif

    case DeviceType::CPU:
    default:
        return createCpuAttention(prec);
    }
}

// Helper: TensorType enum → ActivationPrecision enum
static ActivationPrecision tensorTypeToActivationPrecision(TensorType dtype) {
    switch (dtype) {
    case TensorType::FP32: return ActivationPrecision::FP32;
    case TensorType::FP16: return ActivationPrecision::FP16;
    case TensorType::BF16: return ActivationPrecision::BF16;
    case TensorType::Q8_1: return ActivationPrecision::Q8_1;
    default:
        LOG_WARN("Unsupported dtype, falling back to FP32");
        return ActivationPrecision::FP32;
    }
}

// CPU dispatch
static std::unique_ptr<ITensorAttention> createCpuAttention(ActivationPrecision prec) {
    switch (prec) {
    case ActivationPrecision::FP32: return std::make_unique<CPUAttentionKernelTyped<ActivationPrecision::FP32>>();
    case ActivationPrecision::FP16: return std::make_unique<CPUAttentionKernelTyped<ActivationPrecision::FP16>>();
    case ActivationPrecision::BF16: return std::make_unique<CPUAttentionKernelTyped<ActivationPrecision::BF16>>();
    case ActivationPrecision::Q8_1: return std::make_unique<CPUAttentionKernelTyped<ActivationPrecision::Q8_1>>();
    default: return std::make_unique<CPUAttentionKernelTyped<ActivationPrecision::FP32>>();
    }
}

#ifdef HAVE_CUDA
// CUDA dispatch - same ActivationPrecision pattern
static std::unique_ptr<ITensorAttention> createCudaAttention(ActivationPrecision prec) {
    switch (prec) {
    case ActivationPrecision::FP32: return std::make_unique<cuda::CudaAttentionKernelTyped<ActivationPrecision::FP32>>();
    case ActivationPrecision::FP16: return std::make_unique<cuda::CudaAttentionKernelTyped<ActivationPrecision::FP16>>();
    case ActivationPrecision::BF16: return std::make_unique<cuda::CudaAttentionKernelTyped<ActivationPrecision::BF16>>();
    case ActivationPrecision::Q8_1: return std::make_unique<cuda::CudaAttentionKernelTyped<ActivationPrecision::Q8_1>>();
    default: return std::make_unique<cuda::CudaAttentionKernelTyped<ActivationPrecision::FP16>>();  // FP16 default for GPU
    }
}
#endif

#ifdef HAVE_ROCM
// ROCm dispatch - same ActivationPrecision pattern
static std::unique_ptr<ITensorAttention> createRocmAttention(ActivationPrecision prec) {
    switch (prec) {
    case ActivationPrecision::FP32: return std::make_unique<rocm::RocmAttentionKernelTyped<ActivationPrecision::FP32>>();
    case ActivationPrecision::FP16: return std::make_unique<rocm::RocmAttentionKernelTyped<ActivationPrecision::FP16>>();
    case ActivationPrecision::BF16: return std::make_unique<rocm::RocmAttentionKernelTyped<ActivationPrecision::BF16>>();
    case ActivationPrecision::Q8_1: return std::make_unique<rocm::RocmAttentionKernelTyped<ActivationPrecision::Q8_1>>();
    default: return std::make_unique<rocm::RocmAttentionKernelTyped<ActivationPrecision::FP16>>();  // FP16 default for GPU
    }
}
#endif
```

**CUDA Kernel Template** (mirrors CPU pattern):
```cpp
// src/v2/kernels/cuda/CudaAttentionKernelTyped.h

#pragma once

#ifdef HAVE_CUDA

#include "../../tensors/TensorKernels.h"
#include "../../pipelines/PipelineConfig.h"  // ActivationPrecision

namespace llaminar::v2::kernels::cuda {

namespace detail {
    // CUDA type mapping (similar to CPU but uses CUDA types)
    template <ActivationPrecision P> struct CudaPrecisionTraits;
    
    template <> struct CudaPrecisionTraits<ActivationPrecision::FP32> {
        using ComputeType = float;
        using StorageType = float;
        static constexpr bool use_tensor_cores = false;
    };
    
    template <> struct CudaPrecisionTraits<ActivationPrecision::FP16> {
        using ComputeType = half;
        using StorageType = half;
        static constexpr bool use_tensor_cores = true;  // FP16 Tensor Cores
    };
    
    template <> struct CudaPrecisionTraits<ActivationPrecision::BF16> {
        using ComputeType = nv_bfloat16;
        using StorageType = nv_bfloat16;
        static constexpr bool use_tensor_cores = true;  // BF16 Tensor Cores (Ampere+)
    };
    
    template <> struct CudaPrecisionTraits<ActivationPrecision::Q8_1> {
        using ComputeType = float;      // Accumulate in FP32
        using StorageType = int8_t;     // INT8 storage
        static constexpr bool use_tensor_cores = true;  // INT8 Tensor Cores
    };
}

/**
 * @brief CUDA attention kernel with precision template
 *
 * Uses same ActivationPrecision enum as CPU for consistent factory dispatch.
 * Internally uses CudaPrecisionTraits for CUDA-specific type mappings.
 */
template <ActivationPrecision Precision>
class CudaAttentionKernelTyped : public llaminar2::ITensorAttention {
public:
    using Traits = detail::CudaPrecisionTraits<Precision>;
    using ComputeType = typename Traits::ComputeType;
    using StorageType = typename Traits::StorageType;
    
    bool compute_tensor(
        const TensorBase* Q, const TensorBase* K, const TensorBase* V,
        TensorBase* output,
        int batch_size, int seq_len, int kv_len,
        int n_heads, int n_kv_heads, int head_dim,
        bool causal, int window_size,
        TensorBase* workspace_scores,
        TensorBase* mask,
        const MPIContext* mpi_ctx,
        int device_idx,
        int head_start = 0,
        int local_n_heads = -1,
        int local_n_kv_heads = -1) override
    {
        // Launch precision-specific CUDA kernel
        if constexpr (Traits::use_tensor_cores) {
            return launch_flash_attention_tc<ComputeType, StorageType>(
                Q, K, V, output, batch_size, seq_len, kv_len,
                n_heads, n_kv_heads, head_dim, causal, device_idx,
                head_start, local_n_heads);
        } else {
            return launch_flash_attention_cuda<ComputeType>(
                Q, K, V, output, batch_size, seq_len, kv_len,
                n_heads, n_kv_heads, head_dim, causal, device_idx,
                head_start, local_n_heads);
        }
    }
    
private:
    // FlashAttention with Tensor Cores
    template <typename Compute, typename Storage>
    bool launch_flash_attention_tc(...);
    
    // FlashAttention without Tensor Cores (FP32 fallback)
    template <typename Compute>
    bool launch_flash_attention_cuda(...);
};

} // namespace llaminar::v2::kernels::cuda

#endif // HAVE_CUDA
```

**ROCm Kernel Template** (same pattern for AMD):
```cpp
// src/v2/kernels/rocm/RocmAttentionKernelTyped.h

#pragma once

#ifdef HAVE_ROCM

#include "../../tensors/TensorKernels.h"
#include "../../pipelines/PipelineConfig.h"

namespace llaminar::v2::kernels::rocm {

namespace detail {
    template <ActivationPrecision P> struct RocmPrecisionTraits;
    
    template <> struct RocmPrecisionTraits<ActivationPrecision::FP32> {
        using ComputeType = float;
        using StorageType = float;
        static constexpr bool use_matrix_cores = false;
    };
    
    template <> struct RocmPrecisionTraits<ActivationPrecision::FP16> {
        using ComputeType = _Float16;
        using StorageType = _Float16;
        static constexpr bool use_matrix_cores = true;  // CDNA Matrix Cores
    };
    
    template <> struct RocmPrecisionTraits<ActivationPrecision::BF16> {
        using ComputeType = hip_bfloat16;
        using StorageType = hip_bfloat16;
        static constexpr bool use_matrix_cores = true;  // MI200+ Matrix Cores
    };
    
    template <> struct RocmPrecisionTraits<ActivationPrecision::Q8_1> {
        using ComputeType = float;
        using StorageType = int8_t;
        static constexpr bool use_matrix_cores = true;  // INT8 Matrix Cores
    };
}

template <ActivationPrecision Precision>
class RocmAttentionKernelTyped : public llaminar2::ITensorAttention {
    // Same structure as CUDA, uses RocmPrecisionTraits
    // ...
};

} // namespace llaminar::v2::kernels::rocm

#endif // HAVE_ROCM
```
```

**Template Design** (from `CPUAttentionKernelTyped.h`):

```cpp
// ActivationPrecision enum (from PipelineConfig.h)
enum class ActivationPrecision { FP32, BF16, FP16, Q8_1 };

// Template is on the ENUM, not primitive types
template <ActivationPrecision Precision>
class CPUAttentionKernelTyped : public ITensorAttention {
public:
    // Two-level type resolution:
    // 1. Enum → Tensor type
    using TensorT = typename detail::PrecisionToTensor<Precision>::Type;
    // 2. Tensor type → Element type (via ActivationTraits)
    using ElementType = typename primitives::ActivationTraits<TensorT>::ElementType;
    using Traits = primitives::ActivationTraits<TensorT>;
    
    // ...
};

// Mapping specializations:
template<> struct PrecisionToTensor<ActivationPrecision::FP32> { using Type = FP32Tensor; };
template<> struct PrecisionToTensor<ActivationPrecision::BF16> { using Type = BF16Tensor; };
template<> struct PrecisionToTensor<ActivationPrecision::FP16> { using Type = FP16Tensor; };
template<> struct PrecisionToTensor<ActivationPrecision::Q8_1> { using Type = Q8_1Tensor; };

// ActivationTraits then provides ElementType:
// - FP32Tensor → float
// - BF16Tensor → uint16_t (bf16 storage)
// - FP16Tensor → uint16_t (fp16 storage)
// - Q8_1Tensor → int8_t (quantized storage)
```

**GEMM Kernel Factory** (similar pattern):
```cpp
std::unique_ptr<ITensorGemm> KernelFactory::createGemm(
    const TensorBase* weight_tensor, DeviceType dev_type)
{
#ifdef HAVE_CUDA
    if (dev_type == DeviceType::CUDA) {
        return std::make_unique<cuda::CudaGemmKernel>();
    }
#endif

    // CPU quantized kernels - dispatch based on weight tensor type
    switch (weight_tensor->dtype()) {
    case TensorType::Q8_0:
        return std::make_unique<QuantizedGemmKernel<Q8_0Tensor>>(
            static_cast<const Q8_0Tensor*>(weight_tensor));
            
    case TensorType::Q4_0:
        return std::make_unique<QuantizedGemmKernel<Q4_0Tensor>>(
            static_cast<const Q4_0Tensor*>(weight_tensor));
            
    case TensorType::IQ4_NL:
        return std::make_unique<QuantizedGemmKernel<IQ4_NLTensor>>(
            static_cast<const IQ4_NLTensor*>(weight_tensor));
            
    case TensorType::FP32:
        return std::make_unique<FP32GemmKernel>();
        
    default:
        throw std::runtime_error("Unsupported weight tensor type for GEMM");
    }
}
```

**Key Design Pattern**: Runtime dtype dispatch → compile-time template instantiation

```cpp
// The bridge pattern: polymorphic input -> templated output
//
// TensorBase* tensor (runtime)  →  tensor->dtype() (runtime enum)
//                                         ↓
//                               switch (dtype) { ... }
//                                         ↓
//           CPUAttentionKernelTyped<ActivationPrecision::XXX>() (compile-time)
//                                         ↓
//                      detail::PrecisionToTensor<P>::Type (tensor type)
//                                         ↓
//                      ActivationTraits<TensorT>::ElementType (primitive type)
//
// This is idiomatic C++ for mixing polymorphism with templates.
```

---

## Collective Operation Stages

These new stages wrap MPI collectives for use in the compute graph.

### AllReduceStage

**Location**: `src/v2/execution/CollectiveStages.h`

```cpp
// src/v2/execution/CollectiveStages.h

#pragma once

#include "ComputeStage.h"
#include "../utils/MPIContext.h"

namespace llaminar::v2::execution {

/**
 * @brief Stage that performs MPI AllReduce (sum) on a tensor
 *
 * Used after column-parallel GEMM where each rank has a partial sum
 * that must be combined to produce the final result.
 *
 * Input: Partial tensor [m, n] (local partial product)
 * Output: Same tensor, in-place reduced across all ranks
 */
class AllReduceStage : public ComputeStage {
public:
    struct Params {
        TensorBase* tensor = nullptr;   // In-place reduce
        size_t count = 0;               // Number of elements (0 = infer from tensor)
        std::shared_ptr<MPIContext> mpi_ctx;
    };
    
    explicit AllReduceStage(Params params) : params_(std::move(params)) {}
    
    bool execute(IDeviceContext* ctx) override {
        if (!params_.tensor || !params_.mpi_ctx) {
            LOG_ERROR("[AllReduceStage] Missing tensor or MPI context");
            return false;
        }
        
        float* data = params_.tensor->mutable_data();
        size_t count = params_.count > 0 ? params_.count : params_.tensor->numel();
        
        LOG_DEBUG("[AllReduceStage] AllReduce sum on " << count << " elements");
        params_.mpi_ctx->allreduce_sum(data, count);
        
        return true;
    }
    
    const char* name() const override { return "AllReduceStage"; }
    
private:
    Params params_;
};

/**
 * @brief Stage that performs MPI AllGather to reconstruct sharded tensor
 *
 * Used after row-parallel GEMM or vocab-sharded LM head where each rank
 * has a slice of the output dimension that must be concatenated.
 *
 * Input: Local tensor slice [m, local_n]
 * Output: Full tensor [m, n] with all slices gathered
 */
class AllGatherStage : public ComputeStage {
public:
    struct Params {
        TensorBase* input = nullptr;    // Local slice [m, local_n]
        TensorBase* output = nullptr;   // Full output [m, n]
        size_t local_count = 0;         // Elements per rank (0 = infer)
        std::shared_ptr<MPIContext> mpi_ctx;
        
        // Optional: slice info for variable-size gathers
        std::vector<int> recv_counts;   // Per-rank element counts
        std::vector<int> displacements; // Per-rank offsets
    };
    
    explicit AllGatherStage(Params params) : params_(std::move(params)) {}
    
    bool execute(IDeviceContext* ctx) override {
        if (!params_.input || !params_.output || !params_.mpi_ctx) {
            LOG_ERROR("[AllGatherStage] Missing tensor(s) or MPI context");
            return false;
        }
        
        const float* send = params_.input->data();
        float* recv = params_.output->mutable_data();
        size_t local_count = params_.local_count > 0 
                           ? params_.local_count 
                           : params_.input->numel();
        
        if (params_.recv_counts.empty()) {
            // Equal-size gather
            LOG_DEBUG("[AllGatherStage] AllGather " << local_count 
                      << " elements from each of " << params_.mpi_ctx->world_size() << " ranks");
            
            MPI_Allgather(
                send, static_cast<int>(local_count), MPI_FLOAT,
                recv, static_cast<int>(local_count), MPI_FLOAT,
                params_.mpi_ctx->comm());
        } else {
            // Variable-size gather
            LOG_DEBUG("[AllGatherStage] AllGatherv with variable sizes");
            
            MPI_Allgatherv(
                send, static_cast<int>(local_count), MPI_FLOAT,
                recv, params_.recv_counts.data(), params_.displacements.data(), MPI_FLOAT,
                params_.mpi_ctx->comm());
        }
        
        return true;
    }
    
    const char* name() const override { return "AllGatherStage"; }
    
private:
    Params params_;
};

/**
 * @brief Stage that broadcasts a tensor from rank 0 to all ranks
 *
 * Used for token IDs at the start of inference.
 */
class BroadcastStage : public ComputeStage {
public:
    struct Params {
        TensorBase* tensor = nullptr;
        size_t count = 0;
        int root_rank = 0;
        std::shared_ptr<MPIContext> mpi_ctx;
    };
    
    explicit BroadcastStage(Params params) : params_(std::move(params)) {}
    
    bool execute(IDeviceContext* ctx) override {
        if (!params_.tensor || !params_.mpi_ctx) {
            return false;
        }
        
        float* data = params_.tensor->mutable_data();
        size_t count = params_.count > 0 ? params_.count : params_.tensor->numel();
        
        MPI_Bcast(data, static_cast<int>(count), MPI_FLOAT, 
                  params_.root_rank, params_.mpi_ctx->comm());
        
        return true;
    }
    
    const char* name() const override { return "BroadcastStage"; }
    
private:
    Params params_;
};

} // namespace llaminar::v2::execution
```

### Usage in Graph Construction

```cpp
// Example: Adding AllReduce after column-parallel Down projection

void Qwen2Graph::buildFFNGraph(int layer_idx, ComputeGraph& graph) {
    // ... Gate/Up/SwiGLU stages ...
    
    // Down projection (column-parallel: partial input)
    auto down_params = GEMMStage::Params{};
    down_params.A = buffers_.swiglu_local;
    down_params.B = weights_.layers[layer_idx].down;
    down_params.C = buffers_.ffn_output;
    down_params.m = seq_len_;
    down_params.n = config_.d_model;
    down_params.k = local_d_ff;  // Local input dimension
    down_params.mpi_ctx = mpi_ctx_;
    // Note: Set needs_allreduce=false, use separate stage instead
    
    graph.addStage(std::make_unique<GEMMStage>(down_params), "ffn_down");
    
    // AllReduce to sum partial products
    auto reduce_params = AllReduceStage::Params{};
    reduce_params.tensor = buffers_.ffn_output;
    reduce_params.mpi_ctx = mpi_ctx_;
    
    graph.addStage(std::make_unique<AllReduceStage>(reduce_params), "ffn_allreduce");
}
```

---

## Data Transfer Stage (CPU ↔ GPU)

For heterogeneous execution, data must move between devices.

```cpp
// src/v2/execution/DataTransferStage.h

#pragma once

#include "ComputeStage.h"

namespace llaminar::v2::execution {

/**
 * @brief Stage that copies tensor data between devices
 *
 * Handles:
 * - CPU → GPU (upload before GPU kernel)
 * - GPU → CPU (download for CPU kernel or collective)
 * - GPU → GPU (future: peer-to-peer or via CPU staging)
 */
class DataTransferStage : public ComputeStage {
public:
    enum class Direction {
        HOST_TO_DEVICE,
        DEVICE_TO_HOST,
        DEVICE_TO_DEVICE
    };
    
    struct Params {
        TensorBase* src = nullptr;
        TensorBase* dst = nullptr;
        Direction direction = Direction::HOST_TO_DEVICE;
        int src_device = 0;
        int dst_device = 0;
        bool async = false;  // Future: async copy with events
    };
    
    explicit DataTransferStage(Params params) : params_(std::move(params)) {}
    
    bool execute(IDeviceContext* ctx) override {
        if (!params_.src || !params_.dst) {
            LOG_ERROR("[DataTransferStage] Missing source or destination tensor");
            return false;
        }
        
        size_t bytes = params_.src->numel() * sizeof(float);
        
        switch (params_.direction) {
        case Direction::HOST_TO_DEVICE:
            LOG_DEBUG("[DataTransferStage] Upload " << bytes << " bytes to GPU");
#ifdef HAVE_CUDA
            cudaMemcpy(params_.dst->mutable_data(), params_.src->data(), 
                       bytes, cudaMemcpyHostToDevice);
#endif
            break;
            
        case Direction::DEVICE_TO_HOST:
            LOG_DEBUG("[DataTransferStage] Download " << bytes << " bytes from GPU");
#ifdef HAVE_CUDA
            cudaMemcpy(params_.dst->mutable_data(), params_.src->data(),
                       bytes, cudaMemcpyDeviceToHost);
#endif
            break;
            
        case Direction::DEVICE_TO_DEVICE:
            LOG_WARN("[DataTransferStage] D2D not yet implemented");
            return false;
        }
        
        return true;
    }
    
    const char* name() const override { return "DataTransferStage"; }
    
private:
    Params params_;
};

} // namespace llaminar::v2::execution
```

---

## Testing Strategy

### Unit Tests (Per Phase)

| Phase | Test File | Coverage |
|-------|-----------|----------|
| 2.1 | `Test__KernelFactorySliced.cpp` | Sliced kernel creation/caching |
| 2.2-2.3 | `Test__GEMMStageTP.cpp` | Row-parallel GEMM execution |
| 2.4-2.6 | `Test__AttentionStageTP.cpp` | Head-sharded attention |
| 3 | `Test__Qwen2GraphColumnParallelQKV.cpp` | QKV weight sharding |
| 4 | `Test__Qwen2GraphColumnParallelFFN.cpp` | FFN sharding + allreduce |
| 5 | `Test__Qwen2GraphColumnParallelLMHead.cpp` | Vocab sharding + allgather |

### Integration Tests

```cpp
// tests/v2/integration/Test__DistributedInference.cpp

TEST(DistributedInference, TwoRankTPProducesSameOutputAsFullCompute) {
    // Run with mpirun -np 2
    auto mpi_ctx = std::make_shared<MPIContext>();
    
    // Load model with TP enabled
    ModelContext model_ctx = ModelContext::create(
        "models/qwen2.5-0.5b-instruct-q4_0.gguf",
        mpi_ctx);
    
    // Create TP-enabled orchestrator
    GraphOrchestrator orchestrator(model_ctx, mpi_ctx);
    
    // Run inference
    std::vector<int> tokens = {785, 3974, 13876};  // "The quick brown"
    auto output = orchestrator.forward(tokens.data(), tokens.size());
    
    // Rank 0 gathers and validates
    if (mpi_ctx->rank() == 0) {
        // Compare logits against single-rank baseline
        auto baseline = loadBaselineLogits("baseline_logits.bin");
        float cosine_sim = computeCosineSimilarity(output, baseline);
        EXPECT_GT(cosine_sim, 0.999f);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
}
```

---

## Success Criteria

### Phase 2 Complete When:
- [ ] `GEMMStage` with `output_range` produces correct sliced output
- [ ] `AttentionComputeStage` with head range computes only local heads
- [ ] Unit tests pass for sliced operations
- [ ] Memory usage per rank reduced for tested operations

### Phase 3-4 Complete When:
- [ ] QKV projection uses column-parallel weights
- [ ] FFN uses column-parallel Gate/Up + allreduce for Down
- [ ] 2-rank inference produces same logits as 1-rank (cosine sim > 0.999)
- [ ] Per-rank memory reduced by ~50% for sharded weights

### Phase 5 Complete When:
- [ ] LM head sharded by vocab
- [ ] AllGather reconstructs full logits
- [ ] Sampling produces identical tokens as single-rank

### Full TP Complete When:
- [ ] All deprecated components removed
- [ ] All tests migrated to graph-based execution
- [ ] Documentation updated
- [ ] Performance benchmarks show scaling benefit (>1.5x for 2 ranks)

---

## Appendix A: Proposal Cross-Reference

| Implementation Phase | Proposal Section |
|----------------------|------------------|
| Phase 1 (MPITopology) | "Phase 1: MPITopology Class", "Decision 1: Work Division" |
| Phase 2 (Stage TP) | "Migration Path → Phase 2", "Kernel Support Analysis" |
| Phase 3 (QKV) | "Decision 2: QKV Parallelism Strategy" |
| Phase 4 (FFN) | "Decision 3: Collective Operations Placement" |
| Phase 5 (LM Head) | "Migration Path → Phase 5" |
| Phase 6 (KV Cache) | "Migration Path → Phase 6" |
| Phase 7 (Cleanup) | "Decision 4: Deprecated Components" |
| GPU Phases G1-G5 | "Device Support Status", "GPU Integration Phases" |

---

## Appendix B: File Inventory

### Files to Create

| File | Phase | Purpose |
|------|-------|---------|
| `src/v2/execution/CollectiveStages.h` | 2-4 | AllReduce, AllGather, Broadcast stages |
| `src/v2/execution/DataTransferStage.h` | G3 | CPU↔GPU data movement |
| `src/v2/kernels/cuda/CudaFlashAttentionKernel.h` | G2 | CUDA attention implementation |
| `src/v2/kernels/cuda/CudaFlashAttentionKernel.cu` | G2 | CUDA attention kernels |
| `tests/v2/unit/Test__KernelFactorySliced.cpp` | 2.1 | Sliced kernel tests |
| `tests/v2/unit/Test__GEMMStageTP.cpp` | 2.2-2.3 | TP GEMM tests |
| `tests/v2/unit/Test__AttentionStageTP.cpp` | 2.4-2.6 | TP attention tests |
| `tests/v2/integration/Test__DistributedInference.cpp` | 3-5 | End-to-end TP tests |

### Files to Modify

| File | Phase | Changes |
|------|-------|---------|
| `src/v2/kernels/KernelFactory.h` | 2.1 | Add `getOrCreateGemmSliced()` |
| `src/v2/kernels/KernelFactory.cpp` | 2.1, G2 | Implement sliced cache, add CUDA attention |
| `src/v2/execution/ComputeStage.h` | 2.2, 2.4 | Add TP params to GEMM/Attention stages |
| `src/v2/execution/ComputeStage.cpp` | 2.3, 2.5 | Implement TP logic in execute() |
| `src/v2/tensors/TensorKernels.h` | 2.5 | Add head_start/local_n_heads to ITensorAttention |
| `src/v2/kernels/cpu/attention/CPUAttentionKernelTyped.h` | 2.6 | Use head range in loop |
| `src/v2/loaders/WeightManager.h` | 3.1, 4.1, 5.1 | Add column-shard methods |
| `src/v2/pipelines/qwen/Qwen2Graph.cpp` | 3-5 | Pass TP params to stages |
| `src/v2/pipelines/qwen/GraphOrchestrator.cpp` | G1 | GPU device selection |

### Files to Delete (Phase 7)

| File | Reason |
|------|--------|
| `src/v2/orchestrators/MpiAttentionOrchestrator.h` | Replaced by TP attention stages |
| `src/v2/orchestrators/MpiAttentionOrchestrator.cpp` | " |
| `src/v2/kernels/cpu/attention/GQAAttention.h` | Replaced by CPUAttentionKernelTyped |
| `src/v2/kernels/cpu/attention/GQAAttention.cpp` | " |

---

## Appendix C: Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Head count not divisible by world_size | Incorrect output | WorkRange already handles uneven division |
| KV cache sync across ranks | Divergent decode | Phase 6 ensures local KV storage per rank |
| GPU memory fragmentation | OOM on large models | Phase G1 uses pre-allocated buffers |
| MPI deadlocks in collective stages | Hang | Barrier before AllReduce/AllGather, timeout tests |
| Quantized weight slicing | Alignment issues | Slice at block boundaries (32-element granularity) |

---

## Appendix D: Performance Targets

| Configuration | Metric | Target |
|---------------|--------|--------|
| 2-rank TP (CPU) | Speedup | >1.5x vs 1-rank |
| 2-rank TP (CPU) | Memory/rank | <55% of 1-rank |
| 1-rank GPU | Speedup | >3x vs 1-rank CPU |
| 2-rank GPU | Speedup | >5x vs 1-rank CPU |
| LM head gather | Overhead | <5% of forward pass |
| AllReduce (1KB) | Latency | <100μs |
| AllReduce (1MB) | Latency | <10ms |
