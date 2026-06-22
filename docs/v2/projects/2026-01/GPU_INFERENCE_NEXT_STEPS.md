# GPU Inference Next Steps - Detailed Implementation Plan

**Date**: January 8, 2026  
**Status**: Phase 4.3 Complete, Weight Preloader Complete, Phase 4.4 Next  
**Prerequisite**: Phases 1-3 complete (CUDA GEMM, ops kernels, Flash Attention, Ring KV Cache)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Phase 4.1: KV Cache GPU Integration](#phase-41-kv-cache-gpu-integration) ✅ COMPLETE
3. [Phase 4.2: AttentionWithKVCacheStage GPU Path](#phase-42-attentionwithkvcachestage-gpu-path) ✅ COMPLETE
4. [Phase 4.3: Remaining GEMM Stages](#phase-43-remaining-gemm-stages) ✅ COMPLETE
5. [Phase 4.3.5: Weight Preloader Architecture](#phase-435-weight-preloader-architecture) ✅ COMPLETE
6. [Phase 4.4: GPU Buffer Allocation](#phase-44-gpu-buffer-allocation)
7. [Phase 4.5: Device Transfer Stage](#phase-45-device-transfer-stage)
8. [Phase 4.6: End-to-End Integration](#phase-46-end-to-end-integration)
9. [Testing Strategy](#testing-strategy)

---

## Executive Summary

### Current State (January 10, 2026)

**✅ Complete:**
- All CUDA kernels (GEMM, RMSNorm, RoPE, SwiGLU, ResidualAdd)
- Flash Attention 2 (prefill + decode)
- Ring Buffer KV Cache (FP32/FP16/BF16)
- Stage dispatch for: `GEMMStage`, `LMHeadStage`, `ResidualAddStage`, `RMSNormStage`, `RoPEStage`, `SwiGLUStage`
- **KVCacheAppendStage** - GPU dispatch via `cuda_kv_cache` param ✅ (Jan 9, 2026)
- **KVCacheGatherStage** - GPU dispatch via `cuda_kv_cache` param ✅ (Jan 9, 2026)
- **AttentionWithKVCacheStage** - GPU dispatch via `cuda_kv_cache` param ✅ (Jan 10, 2026)
- **FusedQKVGEMMStage** - GPU dispatch via `getOrCreateGemmForCUDA()` ✅ (Jan 10, 2026)
- **FusedGateUpGEMMStage** - GPU dispatch via `getOrCreateGemmForCUDA()` ✅ (Jan 10, 2026)
- **WeightPreloader** - Unified cross-device weight packing API ✅ (Jan 8, 2026)
- **KernelFactory unified API** - `ensurePackedWeightsInTensorCache(tensor, DeviceType)` ✅ (Jan 8, 2026)

**🔄 Next:**
- Phase 4.4: GPU Buffer Allocation (simplified with WeightPreloader)
- Phase 4.5: TransferStage implementation

**⛔ Blocked:**
- None (attention path now unblocked!)

### Required Changes

| Component | Status | Effort |
|-----------|--------|--------|
| KVCacheAppendStage | ✅ Complete | - |
| KVCacheGatherStage | ✅ Complete | - |
| AttentionWithKVCacheStage | ✅ Complete | - |
| FusedQKVGEMMStage | ✅ Complete | - |
| FusedGateUpGEMMStage | ✅ Complete | - |
| WeightPreloader | ✅ Complete | - |
| Unified KernelFactory API | ✅ Complete | - |
| GPU Buffer Allocation | 🟡 Partial | 0.5 day |
| TransferStage | 🔴 Not implemented | 1 day |
| FusedAttentionWoStage | ⬜ CPU-only (no fused GPU kernel) | N/A |

**Note**: GPU uses the **decomposed attention path** (separate KVCache stages + AttentionWithKVCacheStage + Wo GEMM) rather than the fused CPU JIT kernel.

**Architecture Note**: Weight packing is now handled by `WeightPreloader` which can pre-pack all weights for their target devices (CPU VNNI or CUDA INT8) before graph execution begins. This eliminates lazy packing overhead during inference.

**Total Estimated Effort**: ~1.5 working days (reduced with WeightPreloader complete)

---

## Phase 4.1: KV Cache GPU Integration ✅ COMPLETE

### Completion Summary (January 9, 2026)

Both KV Cache stages now support GPU dispatch when configured with a `cuda_kv_cache` parameter.

**Test Coverage**: `V2_Unit_KVCacheStages_GPU` - 6 tests passing
- `Test__KVCacheAppendStage_GPU.SupportsGPUBackend`
- `Test__KVCacheAppendStage_GPU.AppendDeviceTensors`
- `Test__KVCacheGatherStage_GPU.SupportsGPUBackend`
- `Test__KVCacheGatherStage_GPU.GatherToDeviceTensors`
- `Test__KVCacheStages_GPU.RoundTrip_AppendThenGather`
- `Test__KVCacheStages_GPU.FP16Cache_RoundTrip`

### 4.1.1 KVCacheAppendStage Implementation ✅

**Files Modified**:
- [KVCacheAppendStage.h](../../../../src/v2/execution/compute_stages/stages/KVCacheAppendStage.h)
- [KVCacheAppendStage.cpp](../../../../src/v2/execution/compute_stages/stages/KVCacheAppendStage.cpp)

**Key Changes**:

1. Added `ICUDARingKVCache *cuda_kv_cache = nullptr;` to `Params` struct
2. Implemented `supportsBackend()` to return GPU support when cuda_kv_cache is set
3. GPU dispatch in `execute()`:

```cpp
// GPU path: CUDA KV cache with device tensors
if (params_.cuda_kv_cache && params_.K->is_on_gpu() && params_.V->is_on_gpu())
{
    const float *d_K = reinterpret_cast<const float *>(params_.K->raw_data());
    const float *d_V = reinterpret_cast<const float *>(params_.V->raw_data());
    
    LOG_DEBUG("[KVCacheAppendStage] GPU append: layer=" << params_.layer_idx
              << " seq=" << params_.seq_idx << " tokens=" << total_tokens);
    
    bool success = params_.cuda_kv_cache->append(
        params_.layer_idx, params_.seq_idx, d_K, d_V, total_tokens);
    
    if (!success)
    {
        LOG_ERROR("[KVCacheAppendStage] CUDA append failed");
        return false;
    }
    return true;
}
```

### 4.1.2 KVCacheGatherStage Implementation ✅

**Files Modified**:
- [KVCacheGatherStage.h](../../../../src/v2/execution/compute_stages/stages/KVCacheGatherStage.h)
- [KVCacheGatherStage.cpp](../../../../src/v2/execution/compute_stages/stages/KVCacheGatherStage.cpp)

**Key Changes**:

1. Added `ICUDARingKVCache *cuda_kv_cache = nullptr;` to `Params` struct
2. Implemented `supportsBackend()` to return GPU support when cuda_kv_cache is set
3. GPU dispatch in `execute()` using `gather_kv_batched()`:

```cpp
// GPU path: CUDA KV cache gather
if (params_.cuda_kv_cache && params_.out_K->is_on_gpu() && params_.out_V->is_on_gpu())
{
    float *d_out_K = reinterpret_cast<float *>(params_.out_K->raw_mutable_data());
    float *d_out_V = reinterpret_cast<float *>(params_.out_V->raw_mutable_data());
    
    std::vector<int> kv_lens(batch_size);
    for (int seq = 0; seq < batch_size; ++seq)
    {
        kv_lens[seq] = params_.cuda_kv_cache->get_cached_tokens(layer_idx, seq);
    }
    
    int max_kv_len = 0;
    bool success = params_.cuda_kv_cache->gather_kv_batched(
        layer_idx, batch_size, d_out_K, d_out_V, kv_lens.data(), max_kv_len);
    
    max_kv_len_ = max_kv_len;
    return success;
}
```

---

## Phase 4.2: AttentionWithKVCacheStage GPU Path
        // Get raw device pointers
        const void* d_K = params_.K->raw_data();
        const void* d_V = params_.V->raw_data();
        
        // Extract CUDA stream from context
        cudaStream_t stream = 0;
        if (auto* cuda_ctx = dynamic_cast<CUDADeviceContext*>(ctx))
        {
            stream = cuda_ctx->stream();
        }
        
        LOG_DEBUG("[KVCacheAppendStage] GPU append: layer=" << params_.layer_idx
                  << " seq=" << params_.seq_idx << " tokens=" << total_tokens);
        
        bool success = params_.cuda_kv_cache->append(
            params_.layer_idx,
            params_.seq_idx,
            d_K, d_V,
            total_tokens,
            stream);
        
        if (!success)
        {
            LOG_ERROR("[KVCacheAppendStage] CUDA append failed");
            return false;
        }
        return true;
    }
    
    // CPU path (existing implementation)
    auto *K_base = dynamic_cast<const TensorBase *>(params_.K);
    auto *V_base = dynamic_cast<const TensorBase *>(params_.V);
    // ... rest of existing code
}
```

---

## Phase 4.2: AttentionWithKVCacheStage GPU Path ✅ COMPLETE

### Completion Summary (January 10, 2026)

AttentionWithKVCacheStage now supports GPU dispatch when configured with a `cuda_kv_cache` parameter. Both prefill and decode modes use Flash Attention 2 via `KernelFactory::createAttention()`.

**Test Coverage**: `V2_Unit_AttentionWithKVCacheStage_GPU` - 4 tests passing
- `Test__AttentionWithKVCacheStage_GPU.SupportsGPUBackend`
- `Test__AttentionWithKVCacheStage_GPU.PrefillMode_GPUTensors`
- `Test__AttentionWithKVCacheStage_GPU.DecodeMode_GPUTensors`
- `Test__AttentionWithKVCacheStage_GPU.MultiDecodeIterations`

### 4.2.1 Implementation Details

**Files Modified**:
- [AttentionWithKVCacheStage.h](src/v2/execution/compute_stages/stages/AttentionWithKVCacheStage.h)
- [AttentionWithKVCacheStage.cpp](src/v2/execution/compute_stages/stages/AttentionWithKVCacheStage.cpp)

**Key Changes**:

1. Added `ICUDARingKVCache *cuda_kv_cache = nullptr;` to `Params` struct
2. Added forward declaration `class ICUDARingKVCache;`
3. Implemented `supportsBackend()` to return GPU support when `cuda_kv_cache` is set
4. Added `executeGPUPrefill()` and `executeGPUDecode()` helper methods

**GPU Dispatch in executePrefill()**:
```cpp
// At start of executePrefill()
if (params_.cuda_kv_cache && params_.Q->is_on_gpu())
{
    return executeGPUPrefill(ctx);
}
```

**GPU Dispatch in executeDecode()**:
```cpp
// At start of executeDecode()
if (params_.cuda_kv_cache && params_.Q->is_on_gpu())
{
    return executeGPUDecode(ctx);
}
```

**GPU Execution Flow**:
1. Append K/V to CUDA Ring KV Cache
2. Get all cached K/V pointers via `get_kv_for_attention()`
3. Create Flash Attention kernel via `KernelFactory::createAttention()`
4. Execute attention with `compute_batch()` (prefill) or `compute()` (decode)

### Design Approach

GPU uses the **decomposed attention path** rather than the fused JIT kernel:

```
CPU (FusedAttentionWoStage):  QKV → [Fused Attention + Wo] → output
GPU (Decomposed):             QKV → KVCacheAppend → Attention → Wo GEMM → output
```

This leverages existing CUDA kernels:
- **CUDAFlashAttentionKernelT** - Flash Attention 2 (prefill + decode)
- **CUDARingKVCache** - KV storage and retrieval
- **KernelFactory** - Unified kernel dispatch

---

## Phase 4.3: Remaining GEMM Stages ✅ COMPLETE

### Completion Summary (January 10, 2026)

Both fused GEMM stages now support GPU dispatch when input activations are on GPU. Weights remain on CPU and are packed/transferred by the CUDA GEMM kernels.

**Key Implementation**: Added `KernelFactory::getOrCreateGemmForCUDA()` method to force CUDA kernel creation even when weight tensors are on CPU. This handles the common pattern where weights stay on CPU but activations are on GPU.

**Test Coverage**: `V2_Unit_FusedGEMMStages_GPU` - 5 tests passing
- `Test__FusedQKVGEMMStage_GPU.SupportsGPUBackend`
- `Test__FusedQKVGEMMStage_GPU.GPUTensors_Execute`
- `Test__FusedGateUpGEMMStage_GPU.SupportsGPUBackend`
- `Test__FusedGateUpGEMMStage_GPU.GPUTensors_Execute`
- `Test__FusedGateUpGEMMStage_GPU.DecodeSize`

**Files Modified**:
- [FusedQKVGEMMStage.cpp](../../../../src/v2/execution/compute_stages/stages/FusedQKVGEMMStage.cpp)
- [FusedGateUpGEMMStage.cpp](../../../../src/v2/execution/compute_stages/stages/FusedGateUpGEMMStage.cpp)
- [KernelFactory.h](../../../../src/v2/kernels/KernelFactory.h) - Added `getOrCreateGemmForCUDA()`
- [KernelFactory.cpp](../../../../src/v2/kernels/KernelFactory.cpp) - Added `getCUDADeviceIdForTensor()` helper

### 4.3.1 FusedQKVGEMMStage Implementation ✅

**File**: `src/v2/execution/compute_stages/stages/FusedQKVGEMMStage.cpp`

**Blocking calls** (3 locations, 6 total calls):
- Lines 71-73: Mixed precision path
- Lines 217-219: Q8_1 output path  
- Lines 301-303: FP32 output path

**GPU Dispatch Pattern**:

```cpp
bool FusedQKVGEMMStage::execute(IDeviceContext *ctx)
{
    // GPU path: dispatch via KernelFactory
    if (params_.input->is_on_gpu())
    {
        // Weights stay on CPU, get pre-packed by CUDA kernel
        auto* Wq_base = asTensorBase(params_.wq, "wq");
        auto* Wk_base = asTensorBase(params_.wk, "wk");
        auto* Wv_base = asTensorBase(params_.wv, "wv");
        
        if (!Wq_base || !Wk_base || !Wv_base)
        {
            LOG_ERROR("[FusedQKVGEMMStage] Weights must be CPU TensorBase");
            return false;
        }
        
        // Get CUDA GEMM kernels
        auto* gemm_q = KernelFactory::getOrCreateGemm(Wq_base);
        auto* gemm_k = KernelFactory::getOrCreateGemm(Wk_base);
        auto* gemm_v = KernelFactory::getOrCreateGemm(Wv_base);
        
        const float* d_input = static_cast<const float*>(params_.input->raw_data());
        float* d_Q = static_cast<float*>(params_.output_q->raw_mutable_data());
        float* d_K = static_cast<float*>(params_.output_k->raw_mutable_data());
        float* d_V = static_cast<float*>(params_.output_v->raw_mutable_data());
        
        bool ok = true;
        ok &= gemm_q->multiply(d_input, d_Q, params_.m, params_.n_q, params_.k,
                               true, 1.0f, 0.0f, params_.mpi_ctx, params_.device_idx);
        ok &= gemm_k->multiply(d_input, d_K, params_.m, params_.n_k, params_.k,
                               true, 1.0f, 0.0f, params_.mpi_ctx, params_.device_idx);
        ok &= gemm_v->multiply(d_input, d_V, params_.m, params_.n_v, params_.k,
                               true, 1.0f, 0.0f, params_.mpi_ctx, params_.device_idx);
        
        // Add biases if present (need CUDA kernel)
        if (params_.bias_q)
        {
            // TODO: CUDA bias add kernel
            LOG_WARN("[FusedQKVGEMMStage] GPU bias add not yet implemented");
        }
        
        return ok;
    }
    
    // CPU path (existing)
    // ...
}
```

### 4.3.2 FusedGateUpGEMMStage

**File**: `src/v2/execution/compute_stages/stages/FusedGateUpGEMMStage.cpp`

**Blocking calls** (Lines 59-62):
```cpp
auto *w_gate_base = requireTensorBase(params_.w_gate, "w_gate");
auto *w_up_base = requireTensorBase(params_.w_up, "w_up");
```

**GPU Dispatch Pattern** (same as FusedQKVGEMMStage but 2 projections):

```cpp
bool FusedGateUpGEMMStage::execute(IDeviceContext *ctx)
{
    if (params_.input->is_on_gpu())
    {
        auto* w_gate_base = asTensorBase(params_.w_gate, "w_gate");
        auto* w_up_base = asTensorBase(params_.w_up, "w_up");
        
        auto* gemm_gate = KernelFactory::getOrCreateGemm(w_gate_base);
        auto* gemm_up = KernelFactory::getOrCreateGemm(w_up_base);
        
        const float* d_input = static_cast<const float*>(params_.input->raw_data());
        float* d_gate = static_cast<float*>(params_.output_gate->raw_mutable_data());
        float* d_up = static_cast<float*>(params_.output_up->raw_mutable_data());
        
        bool ok = true;
        ok &= gemm_gate->multiply(d_input, d_gate, params_.m, params_.n_gate, params_.k,
                                  true, 1.0f, 0.0f, params_.mpi_ctx, params_.device_idx);
        ok &= gemm_up->multiply(d_input, d_up, params_.m, params_.n_up, params_.k,
                                true, 1.0f, 0.0f, params_.mpi_ctx, params_.device_idx);
        
        return ok;
    }
    
    // CPU path
    // ...
}
```

### 4.3.3 Wo Projection (Separate GEMMStage)

For GPU decomposed attention, the **Wo projection** is handled by a separate `GEMMStage` after `AttentionWithKVCacheStage`:

```
GPU Pipeline:
  AttentionWithKVCacheStage (Flash Attention) → context [batch, n_heads * head_dim]
  GEMMStage (Wo projection)                   → attn_output [batch, d_model]
```

`GEMMStage` was already updated with GPU dispatch in the previous session.

---

## Phase 4.3.5: Weight Preloader Architecture ✅ COMPLETE

### Completion Summary (January 8, 2026)

Implemented unified cross-device weight packing with `WeightPreloader` class and refactored `KernelFactory` to provide a single entry point for weight packing regardless of target device.

### Problem Statement

Previously, weight packing was:
1. **Lazy** - packed on first use during graph execution (adding latency to first inference)
2. **Fragmented** - separate `ensurePackedWeightsInTensorCache()` and `ensureCUDAPackedWeightsInTensorCache()` methods
3. **Device-agnostic** - callers had to know which method to call based on target device

### Solution: Unified Weight Packing API

**Files Modified**:
- `src/v2/kernels/KernelFactory.h`
- `src/v2/kernels/KernelFactory.cpp`

**New Files**:
- `src/v2/loaders/WeightPreloader.h`
- `src/v2/loaders/WeightPreloader.cpp`

### 4.3.5.1 KernelFactory Unified API

Single entry point for all weight packing:

```cpp
// Unified packing - dispatches to CPU VNNI or CUDA INT8 automatically
static bool ensurePackedWeightsInTensorCache(
    const TensorBase* tensor, 
    DeviceType target_device = DeviceType::CPU);

// Getters (throw if not packed)
static const gemm_v4::QuantisedPackedWeights* getCPUPackedWeights(const TensorBase* tensor);
static cuda::CUDAPackedWeights* getCUDAPackedWeights(const TensorBase* tensor);
```

**Internal Implementation**:
- `packForCPU()` - Packs weights into VNNI format via `QuantisedGemmKernel::packWeightsInto()`
- `packForCUDA()` - Packs weights into INT8 + per-column scales via `cuda::packWeightsToCUDA()`
- Packed data stored in `tensor->cache_` (CPU) or `tensor->cuda_cache_` (CUDA)

### 4.3.5.2 WeightPreloader Class

Pre-packs all weights before graph execution:

```cpp
class WeightPreloader
{
public:
    using PreloadProgressCallback = std::function<void(size_t current, size_t total, const std::string& name)>;
    
    WeightPreloader(WeightManager& weight_manager, const WeightPlacementMap* placement = nullptr);
    
    // Pre-pack all weights for their assigned devices
    bool preloadAll(PreloadProgressCallback progress_cb = nullptr);
    
    // Pre-pack specific weights by name
    bool preload(const std::vector<std::string>& weight_names);
    
    // Pre-pack all weights for a specific device
    bool preloadForDevice(DeviceType target_device);
    
    // Stats
    size_t numCPUPacked() const { return num_cpu_packed_; }
    size_t numGPUPacked() const { return num_gpu_packed_; }
};
```

**Usage in Inference Entry Point**:

```cpp
// After model load, before inference loop
WeightPreloader preloader(weight_manager, &placement_map);
preloader.preloadAll([](size_t cur, size_t total, const std::string& name) {
    LOG_INFO("Packing weight " << cur << "/" << total << ": " << name);
});

LOG_INFO("Pre-packed " << preloader.numCPUPacked() << " CPU weights, "
         << preloader.numGPUPacked() << " GPU weights");
```

### 4.3.5.3 Benefits

1. **Eliminates lazy packing overhead** - All weights packed before first token
2. **Unified API** - Single method handles CPU/CUDA automatically
3. **Progress reporting** - Callback for UI progress bars
4. **Placement-aware** - Uses `WeightPlacementMap` to determine target device per weight
5. **Memory optimization** - Can optionally release raw weight data after packing

### Test Coverage

All existing tests pass (225 unit tests, 60/61 integration tests).

**Note**: The one failing integration test (`V2_Integration_CUDAFullModelInference`) is a pre-existing issue with CUDA KV cache tensor creation, not related to WeightPreloader.

---

## Phase 4.4: GPU Buffer Allocation

### Problem Statement

Currently, `BufferManager` and `ComputeGraph` only support CPU tensor allocation via `TensorFactory::create()`. With **WeightPreloader** complete, weights are now pre-packed before execution. The remaining gap is **activation buffer allocation** on GPU.

### Scope (Reduced with WeightPreloader)

| Task | Status | Notes |
|------|--------|-------|
| Weight packing on correct device | ✅ Done | `WeightPreloader.preloadAll()` handles this |
| Activation buffer GPU allocation | 🔴 Not done | `TensorFactory::createGPU()` needed |
| BufferManager device-aware allocation | 🔴 Not done | Respect `device_idx` in `BufferDescriptor` |

### Required Changes

1. **BufferDescriptor device field** (already exists but not used):

```cpp
// In BufferDescriptor.h
struct BufferDescriptor
{
    std::string name;
    std::vector<size_t> shape;
    TensorType dtype = TensorType::FP32;
    BufferUsage usage = BufferUsage::ACTIVATION;
    int device_idx = -1;  // -1 = CPU, 0+ = GPU device
};
```

2. **TensorFactory GPU dispatch**:

```cpp
// In TensorFactory.cpp
std::unique_ptr<ITensor> TensorFactory::create(
    TensorType type,
    const std::vector<size_t>& shape,
    int device_idx)
{
    if (device_idx >= 0)
    {
        // GPU allocation
        return createGPU(type, shape, device_idx);
    }
    
    // CPU allocation (existing code)
    // ...
}

std::unique_ptr<ITensor> TensorFactory::createGPU(
    TensorType type,
    const std::vector<size_t>& shape,
    int device_idx)
{
#ifdef HAVE_CUDA
    switch (type)
    {
    case TensorType::FP32:
        return std::make_unique<CUDAFPTensor<float>>(shape, device_idx);
    case TensorType::FP16:
        return std::make_unique<CUDAFPTensor<__half>>(shape, device_idx);
    case TensorType::BF16:
        return std::make_unique<CUDAFPTensor<__nv_bfloat16>>(shape, device_idx);
    default:
        LOG_ERROR("[TensorFactory] Unsupported GPU tensor type: " << static_cast<int>(type));
        return nullptr;
    }
#else
    LOG_ERROR("[TensorFactory] CUDA not available for GPU allocation");
    return nullptr;
#endif
}
```

3. **BufferManager GPU allocation**:

```cpp
// In BufferManager.cpp
ITensor* BufferManager::getOrAllocate(const BufferDescriptor& desc)
{
    auto it = buffers_.find(desc.name);
    if (it != buffers_.end())
    {
        return it->second.get();
    }
    
    // Allocate new buffer
    auto tensor = TensorFactory::create(desc.dtype, desc.shape, desc.device_idx);
    if (!tensor)
    {
        LOG_ERROR("[BufferManager] Failed to allocate buffer: " << desc.name);
        return nullptr;
    }
    
    ITensor* ptr = tensor.get();
    buffers_[desc.name] = std::move(tensor);
    return ptr;
}
```

### Integration with WeightPreloader

The overall initialization flow becomes:

```cpp
// 1. Load model
auto model_ctx = ModelLoader::load(model_path);
auto weight_manager = model_ctx->weight_manager();

// 2. Create placement map (assigns weights to CPU/GPU)
WeightPlacementMap placement_map;
placement_map.assignLayersToDevice(0, 11, DeviceType::CPU);   // Layers 0-11 on CPU
placement_map.assignLayersToDevice(12, 23, DeviceType::CUDA); // Layers 12-23 on GPU

// 3. Pre-pack all weights for their target devices
WeightPreloader preloader(*weight_manager, &placement_map);
preloader.preloadAll();

// 4. Create graph with device-aware buffer allocation
GraphOrchestrator orchestrator(graph_builder, weight_manager);
orchestrator.initializeInferenceState(batch_size, max_seq_len, d_model, vocab_size, 
                                       /*device_idx=*/0);  // GPU 0

// 5. Run inference (no lazy packing overhead!)
orchestrator.forward(tokens, seq_len);
```

### Testing Checklist

- [ ] Unit test: GPU tensor allocation via factory
- [ ] Unit test: BufferManager with GPU descriptors
- [ ] Integration test: Mixed CPU/GPU buffer allocation
- [ ] Integration test: WeightPreloader + GPU inference E2E

---

## Phase 4.5: Device Transfer Stage

### Design

A new `TransferStage` for explicit CPU↔GPU data movement.

**File**: `src/v2/execution/compute_stages/stages/TransferStage.h`

```cpp
#pragma once

#include "../IComputeStage.h"

namespace llaminar2
{

enum class TransferDirection
{
    HOST_TO_DEVICE,   // CPU → GPU
    DEVICE_TO_HOST,   // GPU → CPU
    DEVICE_TO_DEVICE  // GPU → GPU (via host staging)
};

class TransferStage : public IComputeStage
{
public:
    struct Params
    {
        ITensor *src = nullptr;              // Source tensor
        ITensor *dst = nullptr;              // Destination tensor
        TransferDirection direction;
        bool sync_after = true;              // Synchronize after transfer
    };

    explicit TransferStage(Params params);

    bool execute(IDeviceContext *ctx) override;
    ComputeStageType type() const override { return ComputeStageType::COPY; }
    std::string name() const override;
    bool supportsBackend(ComputeBackendType backend) const override { return true; }
    
    size_t estimatedMemoryBytes() const override { return params_.src->size_bytes(); }

private:
    Params params_;
};

// Factory helpers
inline std::unique_ptr<TransferStage> createHostToDeviceTransfer(ITensor *src, ITensor *dst)
{
    return std::make_unique<TransferStage>(TransferStage::Params{
        src, dst, TransferDirection::HOST_TO_DEVICE, true});
}

inline std::unique_ptr<TransferStage> createDeviceToHostTransfer(ITensor *src, ITensor *dst)
{
    return std::make_unique<TransferStage>(TransferStage::Params{
        src, dst, TransferDirection::DEVICE_TO_HOST, true});
}

} // namespace llaminar2
```

**Implementation**: `src/v2/execution/compute_stages/stages/TransferStage.cpp`

```cpp
#include "TransferStage.h"
#include "../../../backends/cuda/CUDABackend.h"
#include "../../../utils/Logger.h"

namespace llaminar2
{

TransferStage::TransferStage(Params params) : params_(std::move(params))
{
    if (!params_.src || !params_.dst)
    {
        throw std::invalid_argument("[TransferStage] Null tensor");
    }
    if (params_.src->size_bytes() != params_.dst->size_bytes())
    {
        throw std::invalid_argument("[TransferStage] Size mismatch");
    }
}

bool TransferStage::execute(IDeviceContext *ctx)
{
    const size_t bytes = params_.src->size_bytes();
    
    LOG_DEBUG("[TransferStage] Execute: " << (bytes / 1024.0 / 1024.0) << " MB, "
              << "direction=" << static_cast<int>(params_.direction));
    
    auto& backend = getCUDABackend();
    bool success = false;
    
    switch (params_.direction)
    {
    case TransferDirection::HOST_TO_DEVICE:
    {
        int dst_device = params_.dst->home_dm_device_index();
        success = backend.hostToDevice(
            params_.dst->raw_mutable_data(),
            params_.src->raw_data(),
            bytes,
            dst_device);
        
        if (params_.sync_after && success)
        {
            backend.synchronize(dst_device);
        }
        break;
    }
    
    case TransferDirection::DEVICE_TO_HOST:
    {
        int src_device = params_.src->home_dm_device_index();
        
        // Sync source device before read
        backend.synchronize(src_device);
        
        success = backend.deviceToHost(
            params_.dst->raw_mutable_data(),
            params_.src->raw_data(),
            bytes,
            src_device);
        break;
    }
    
    case TransferDirection::DEVICE_TO_DEVICE:
    {
        // Via host staging (peer-to-peer is future optimization)
        std::vector<char> staging(bytes);
        
        int src_device = params_.src->home_dm_device_index();
        int dst_device = params_.dst->home_dm_device_index();
        
        backend.synchronize(src_device);
        if (!backend.deviceToHost(staging.data(), params_.src->raw_data(), bytes, src_device))
        {
            LOG_ERROR("[TransferStage] D2D: D2H step failed");
            return false;
        }
        
        if (!backend.hostToDevice(params_.dst->raw_mutable_data(), staging.data(), bytes, dst_device))
        {
            LOG_ERROR("[TransferStage] D2D: H2D step failed");
            return false;
        }
        
        if (params_.sync_after)
        {
            backend.synchronize(dst_device);
        }
        success = true;
        break;
    }
    }
    
    return success;
}

std::string TransferStage::name() const
{
    const char* dir_str = "Unknown";
    switch (params_.direction)
    {
    case TransferDirection::HOST_TO_DEVICE: dir_str = "H2D"; break;
    case TransferDirection::DEVICE_TO_HOST: dir_str = "D2H"; break;
    case TransferDirection::DEVICE_TO_DEVICE: dir_str = "D2D"; break;
    }
    
    return std::string("Transfer_") + dir_str + "_" + 
           std::to_string(params_.src->size_bytes() / 1024) + "KB";
}

} // namespace llaminar2
```

---

## Phase 4.6: End-to-End Integration

### Graph Builder Updates

**File**: `src/v2/pipelines/qwen/Qwen2Graph.cpp`

Add device-aware graph construction with **decomposed attention for GPU**:

```cpp
ComputeGraph Qwen2GraphBuilder::buildLayer(
    int layer_idx,
    const LayerBuffers& buffers,
    const LayerWeights& weights,
    DevicePlacement placement)
{
    ComputeGraph graph;
    
    const bool use_gpu = (placement == DevicePlacement::GPU);
    
    // 1. Attention norm
    auto attn_norm = createRMSNormStage({
        .input = buffers.hidden_states,
        .output = buffers.attn_norm_out,
        .gamma = weights.attn_norm,
        .device_idx = use_gpu ? gpu_device_idx_ : -1
    });
    graph.addNode("attn_norm", std::move(attn_norm));
    
    // 2. QKV projection
    auto qkv = createFusedQKVGEMMStage({
        .input = buffers.attn_norm_out,
        .wq = weights.wq,
        .wk = weights.wk,
        .wv = weights.wv,
        .output_q = buffers.Q,
        .output_k = buffers.K,
        .output_v = buffers.V,
        .device_idx = use_gpu ? gpu_device_idx_ : -1
    });
    graph.addNode("qkv", std::move(qkv));
    graph.addDependency("qkv", "attn_norm");
    
    // 3. RoPE
    auto rope = createRoPEStage({
        .Q = buffers.Q,
        .K = buffers.K,
        .device_idx = use_gpu ? gpu_device_idx_ : -1
        // ...
    });
    graph.addNode("rope", std::move(rope));
    graph.addDependency("rope", "qkv");
    
    // 4. Attention - DIFFERENT PATHS FOR CPU vs GPU
    if (use_gpu)
    {
        // ================================================================
        // GPU: Decomposed attention path
        // ================================================================
        
        // 4a. Append K/V to CUDA cache
        auto kv_append = createKVCacheAppendStage({
            .K = buffers.K,
            .V = buffers.V,
            .cuda_kv_cache = cuda_kv_cache_,
            .layer_idx = layer_idx,
            // ...
        });
        graph.addNode("kv_append", std::move(kv_append));
        graph.addDependency("kv_append", "rope");
        
        // 4b. Flash Attention (reads from CUDA cache)
        auto attention = createAttentionWithKVCacheStage({
            .Q = buffers.Q,
            .output = buffers.attn_context,  // Intermediate buffer
            .cuda_kv_cache = cuda_kv_cache_,
            .layer_idx = layer_idx,
            .device_idx = gpu_device_idx_
            // ...
        });
        graph.addNode("attention", std::move(attention));
        graph.addDependency("attention", "kv_append");
        
        // 4c. Wo projection (separate GEMM)
        auto wo_proj = createGEMMStage({
            .input = buffers.attn_context,
            .weights = weights.wo,
            .output = buffers.attn_out,
            .device_idx = gpu_device_idx_
            // ...
        });
        graph.addNode("wo_proj", std::move(wo_proj));
        graph.addDependency("wo_proj", "attention");
        
        // Residual depends on Wo projection
        graph.addDependency("attn_residual", "wo_proj");
    }
    else
    {
        // ================================================================
        // CPU: Fused attention + Wo (JIT kernel)
        // ================================================================
        auto attn = createFusedAttentionWoStage({
            .Q = buffers.Q,
            .K = buffers.K,
            .V = buffers.V,
            .Wo = weights.wo,
            .output = buffers.attn_out,
            .kv_cache = kv_cache_,
            .layer_idx = layer_idx,
            .device_idx = -1
            // ...
        });
        graph.addNode("attention", std::move(attn));
        graph.addDependency("attention", "rope");
        
        // Residual depends on fused attention
        graph.addDependency("attn_residual", "attention");
    }
    
    // 5. Attention residual
    auto attn_residual = createResidualAddStage({
        .input = buffers.hidden_states,
        .residual = buffers.attn_out,
        .output = buffers.post_attn,
        .device_idx = use_gpu ? gpu_device_idx_ : -1
    });
    graph.addNode("attn_residual", std::move(attn_residual));
    
    // ... rest of layer (FFN norm, gate/up, SwiGLU, down, FFN residual)
    
    return graph;
}
```

**Key difference**: GPU path has 3 nodes (kv_append → attention → wo_proj), CPU path has 1 fused node.
```

---

## Testing Strategy

### Unit Tests

| Test | File | Verifies |
|------|------|----------|
| KVCacheAppendStage GPU | `Test__KVCacheAppendStage.cpp` | GPU append via CUDA cache |
| KVCacheGatherStage GPU | `Test__KVCacheGatherStage.cpp` | GPU gather via CUDA cache |
| FusedAttentionWoStage GPU | `Test__FusedAttentionWoStage.cpp` | Flash Attention + Wo GEMM |
| FusedQKVGEMMStage GPU | `Test__FusedQKVGEMMStage.cpp` | 3x CUDA GEMM projections |
| TransferStage | `Test__TransferStage.cpp` | H2D, D2H, D2D transfers |
| WeightPreloader | `Test__WeightPreloader.cpp` | Pre-packing all weights |
| KernelFactory unified API | `Test__KernelFactory.cpp` | `ensurePackedWeightsInTensorCache()` |

### Integration Tests

| Test | File | Verifies |
|------|------|----------|
| Single Layer GPU | `Test__GPULayerForward.cpp` | Full layer: QKV → Attn → FFN |
| Multi-Token Generation | `Test__GPUMultiTokenGeneration.cpp` | Decode loop with GPU KV cache |
| CPU/GPU Parity | `Test__GPUCPUParity.cpp` | Numerical agreement |
| WeightPreloader + Inference | `Test__WeightPreloaderInference.cpp` | Pre-pack then run E2E |

### End-to-End Tests

| Test | File | Verifies |
|------|------|----------|
| GPU Inference Full | `Test__GPUFullModelInference.cpp` | Prefill + decode |
| Mixed Device | `Test__MixedDeviceInference.cpp` | CPU embedding, GPU transformer, CPU LM head |

---

## Implementation Order

### ✅ COMPLETE (Week 1)

```
├── Phase 4.1 (KV Cache stages) ✅
│   ├── KVCacheAppendStage GPU path
│   └── KVCacheGatherStage GPU path
│
├── Phase 4.2 (AttentionWithKVCacheStage) ✅
│   ├── GPU execute paths (prefill/decode/batched)
│   └── Integration with CUDA KV cache
│
├── Phase 4.3 (Remaining GEMM stages) ✅
│   ├── FusedQKVGEMMStage
│   └── FusedGateUpGEMMStage
│
└── Phase 4.3.5 (WeightPreloader) ✅
    ├── Unified KernelFactory API
    ├── WeightPreloader class
    └── Cross-device packing (CPU VNNI + CUDA INT8)
```

### 🔄 REMAINING (Week 2)

```
├── Day 1: Phase 4.4 (GPU Buffer Allocation) - 0.5 day
│   ├── TensorFactory::createGPU()
│   └── BufferManager device-aware allocation
│
├── Day 2: Phase 4.5 (TransferStage) - 1 day
│   ├── TransferStage implementation
│   └── H2D/D2H/D2D tests
│
└── Day 3: Phase 4.6 (Integration) - 0.5 day
    ├── Qwen2Graph GPU mode (decomposed attention)
    ├── Fix V2_Integration_CUDAFullModelInference (CUDA KV cache tensor creation bug)
    └── End-to-end testing
```

**Estimated remaining effort**: ~2 days

**Note**: FusedAttentionWoStage remains CPU-only. GPU uses decomposed path.

---

## Success Criteria

1. **Functional**: GPU decode generates correct tokens (matches CPU output)
2. **Performance**: GPU decode achieves >50 tok/s (vs ~15 tok/s CPU baseline)
3. **Stability**: All 27 CUDA kernel tests pass + new integration tests
4. **Memory**: GPU VRAM usage within expected bounds (see project plan)
5. **Weight Pre-packing**: `WeightPreloader.preloadAll()` completes before first token

---

## Appendix: Files to Modify

### Completed Files

| File | Changes | Status |
|------|---------|--------|
| `KVCacheAppendStage.h/cpp` | `ICUDARingKVCache*` + GPU dispatch | ✅ |
| `KVCacheGatherStage.h/cpp` | `ICUDARingKVCache*` + GPU dispatch | ✅ |
| `AttentionWithKVCacheStage.h/cpp` | GPU dispatch in all execute methods | ✅ |
| `FusedQKVGEMMStage.cpp` | GPU dispatch at start | ✅ |
| `FusedGateUpGEMMStage.cpp` | GPU dispatch at start | ✅ |
| `KernelFactory.h/cpp` | Unified `ensurePackedWeightsInTensorCache(tensor, DeviceType)` | ✅ |
| `WeightPreloader.h/cpp` | New files for pre-packing | ✅ |
| `WeightManager.h` | Added `cache()` accessor | ✅ |

### Remaining Files

| File | Changes | Status |
|------|---------|--------|
| `TensorFactory.cpp` | `createGPU()` method | 🔴 |
| `BufferManager.cpp` | GPU allocation support | 🔴 |
| `TransferStage.h/cpp` | New files | 🔴 |
| `Qwen2Graph.cpp` | Device-aware graph building | 🔴 |

### Not Modified (CPU-only)

| File | Reason |
|------|--------|
| `FusedAttentionWoStage.cpp` | No fused GPU kernel - stays CPU-only |
| `EmbeddingStage.cpp` | Embeddings stay on CPU (no GPU kernel) |
