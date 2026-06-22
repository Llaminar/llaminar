# CUDA Kernels Project Plan

**Branch**: `feature/cuda-kernels`  
**Created**: January 6, 2026  
**Updated**: January 7, 2026  
**Goal**: Enable GPU inference in Llaminar V2 via CUDA kernels

---

## Executive Summary

This project adds CUDA kernel support to Llaminar V2, enabling GPU-accelerated inference. The existing architecture has been designed with GPU support in mind—device detection, per-tensor device affinity, and graph-level device routing infrastructure already exist. This project implements the missing GPU execution components.

**Key Architecture Decision (January 7, 2026)**: The codebase uses a **two-tier tensor abstraction**:
- **`ITensor*`**: Device-agnostic interface used in all stage `Params` structs
- **`TensorBase*`**: CPU-specific implementation (subclass of `ITensor`)
- **`CUDATensorBase*`**: GPU-specific implementation (also subclass of `ITensor`)

Stages must be updated to work with `ITensor*` directly (via polymorphism) rather than casting to `TensorBase*`.

---

## Current State Assessment

### ✅ Infrastructure Already Complete

| Component | Status | Location |
|-----------|--------|----------|
| Device enumeration (CUDA/ROCm) | ✅ Working | `src/v2/devices/CUDAEnumeration.cu` |
| NUMA-aware GPU filtering | ✅ Working | `src/v2/utils/NUMATopology.cpp` |
| `DeviceManager` singleton | ✅ Working | `src/v2/devices/DeviceManager.cpp` |
| `DeviceId` type-safe device identification | ✅ Working | `src/v2/backends/DeviceId.h` |
| `DeviceType` enum | ✅ Working | `src/v2/devices/DeviceContext.h` |
| `IDeviceContext` interface | ✅ Working | `src/v2/devices/DeviceContext.h` |
| `ComputeGraph` with `device_idx` | ✅ Working | `src/v2/execution/ComputeGraph.h` |
| `DeviceGraphExecutor::executeMultiDevice()` | ✅ Working | `src/v2/execution/DeviceGraphExecutor.cpp` |
| `KernelFactory` device routing | ✅ Working | `src/v2/kernels/KernelFactory.h` |
| `LayerPlacement` per-layer devices | ✅ Exists | `src/v2/pipelines/qwen/PlacementPlan.h` |
| `PlacementStrategy` interface | ✅ Exists | `src/v2/pipelines/qwen/PlacementStrategy.h` |
| `ITensor` interface with `home_device()` | ✅ Working | `src/v2/tensors/ITensor.h` |
| Stage Params structs use `ITensor*` | ✅ Done | All stage headers updated |

### ✅ Completed CUDA Kernels (January 6-7, 2026)

| Component | Status | Location |
|-----------|--------|----------|
| `CUDADeviceContext` | ✅ Done | `src/v2/execution/DeviceContext.cpp` |
| GPU memory allocation/transfer | ✅ Done | `src/v2/backends/cuda/CUDABackend.cu` |
| cuBLAS GEMM (FP32/FP16/BF16) | ✅ Done | `src/v2/kernels/cuda/CuBLASGemmKernel.cu` |
| CUDA Quantized GEMM (CUTLASS) | ✅ Done | `src/v2/kernels/cuda/CUDAQuantisedGemmKernel_CUTLASS.cu` |
| CUDA RMSNorm | ✅ Done | `src/v2/kernels/cuda/ops/CUDARMSNormKernelT.h` |
| CUDA RoPE | ✅ Done | `src/v2/kernels/cuda/ops/CUDARoPEKernelT.h` |
| CUDA SwiGLU | ✅ Done | `src/v2/kernels/cuda/ops/CUDASwiGLUKernelT.h` |
| CUDA ResidualAdd | ✅ Done | `src/v2/kernels/cuda/ops/CUDAResidualAddKernelT.h` |
| CUDA Flash Attention | ✅ Done | `src/v2/kernels/cuda/attention/CUDAFlashAttentionKernels.cu` |
| CUDA Ring Buffer KV Cache | ✅ Done | `src/v2/kernels/cuda/CUDARingKVCache.cu` |

### ⚠️ Blocking Issues for End-to-End GPU Inference

| Issue | Impact | Fix Required |
|-------|--------|--------------|
| Stage `.cpp` files use `requireTensorBase()` | ❌ Blocks GPU tensors | Update to dispatch via `ITensor*` |
| `KernelFactory::createAttention()` takes `TensorBase*` | ❌ Blocks GPU attention | Add `ITensor*` overload |
| `FusedAttentionWoStage` uses `TensorBase*` casts | ❌ Decode pipeline blocked | Use decomposed attention for GPU |
| No decomposed attention CUDA path | ❌ GPU decode blocked | Wire `AttentionComputeStage` to GPU |

### ❌ Components Remaining

| Component | Priority | Complexity | Blocker |
|-----------|----------|------------|---------|
| Update `AttentionComputeStage` for GPU | **P0** | Medium | GPU decode |
| Update `KVCacheAppendStage` for GPU | **P0** | Medium | GPU decode |
| Wire KernelFactory with `ITensor*` | **P1** | Low | All stages |
| `GPUFirstStrategy` implementation | P2 | Medium | Multi-device |
| Per-layer device routing in Qwen2Graph | P3 | Medium | Hybrid CPU/GPU |

---

## Integration Hooks Reference

This section documents the exact code locations where CUDA kernels integrate with the existing system.

### Hook 1: Device Context Factory

**File**: `src/v2/execution/DeviceContext.cpp` line 35-60

```cpp
std::unique_ptr<IDeviceContext> IDeviceContext::create(int device_idx, int num_threads) {
    auto& dm = DeviceManager::instance();
    const auto& devices = dm.devices();
    const auto& device = devices[device_idx];
    
    switch (device.type) {
    case ComputeBackendType::CPU:
        return std::make_unique<CPUDeviceContext>(device_idx, num_threads);
    case ComputeBackendType::GPU_CUDA:
#ifdef HAVE_CUDA
        return std::make_unique<CUDADeviceContext>(device_idx, device.device_id);
#endif
    // ...
    }
}
```

**Status**: ✅ Already wired - creating `IDeviceContext::create(1)` returns `CUDADeviceContext`.

---

### Hook 2: KernelFactory GEMM Dispatch

**File**: `src/v2/kernels/KernelFactory.cpp` 

Each tensor type has a `createGemm()` overload with device dispatch:

```cpp
// Example: IQ4_NL (line ~420)
std::unique_ptr<ITensorGemm> KernelFactory::createGemm(
    const IQ4_NLTensor* tensor, DeviceType dev_type) 
{
    switch (dev_type) {
    case DeviceType::CPU:
        return createCPUIQ4NLGemm(tensor);
#ifdef HAVE_CUDA
    case DeviceType::CUDA:
        return cuda::createCudaGemm(tensor);  // ← CUDA hook
#endif
    default:
        throwUnsupportedBackend(dev_type, "IQ4_NL");
    }
}
```

**Integration Points** (add CUDA dispatch to each):

| Tensor Type | Function | Line (approx) | Status |
|-------------|----------|---------------|--------|
| `IQ4_NLTensor` | `createGemm(const IQ4_NLTensor*)` | 420 | ✅ Has CUDA |
| `Q8_0Tensor` | `createGemm(const Q8_0Tensor*)` | 300 | ❌ CPU only |
| `Q4_0Tensor` | `createGemm(const Q4_0Tensor*)` | 250 | ❌ CPU only |
| `Q4_1Tensor` | `createGemm(const Q4_1Tensor*)` | 270 | ❌ CPU only |
| `Q5_0Tensor` | `createGemm(const Q5_0Tensor*)` | 280 | ❌ CPU only |
| `Q5_1Tensor` | `createGemm(const Q5_1Tensor*)` | 290 | ❌ CPU only |
| `Q8_1Tensor` | `createGemm(const Q8_1Tensor*)` | 310 | ❌ CPU only |
| `FP32Tensor` | `createGemm(const FP32Tensor*)` | 180 | ❌ CPU only |
| `FP16Tensor` | `createGemm(const FP16Tensor*)` | 200 | ❌ CPU only |
| `BF16Tensor` | `createGemm(const BF16Tensor*)` | 220 | ❌ CPU only |

---

### Hook 3: ComputeStage Device Index - ITensor* Scheme

**File**: `src/v2/execution/compute_stages/stages/*.h`

All stage Params structs now use `ITensor*` (device-agnostic interface):

```cpp
// Example: GEMMStage (UPDATED - uses ITensor*)
struct GEMMStage::Params
{
    const ITensor *A = nullptr;       ///< Input activation tensor [m, k]
    const ITensor *B = nullptr;       ///< Weight tensor [k, n] (may be quantized)
    ITensor *C = nullptr;             ///< Output tensor [m, n]
    int device_idx = -1;              ///< Device routing
    const ITensor *bias_tensor = nullptr;
    // ...
};

// Example: AttentionComputeStage (UPDATED - uses ITensor*)
struct AttentionComputeStage::Params
{
    ITensor *Q = nullptr;             ///< Query tensor [seq_len, n_heads*head_dim]
    ITensor *K = nullptr;             ///< Key tensor [kv_len, n_kv_heads*head_dim]  
    ITensor *V = nullptr;             ///< Value tensor [kv_len, n_kv_heads*head_dim]
    ITensor *output = nullptr;        ///< Output tensor [seq_len, n_heads*head_dim]
    int device_idx = -1;
    // ...
};
```

**Status**: ✅ Headers updated to `ITensor*` - implementations need GPU dispatch.

**BLOCKING ISSUE**: Stage `.cpp` implementations still cast to `TensorBase*`:

```cpp
// AttentionComputeStage.cpp - CURRENT (blocks GPU)
auto *Q_base = requireTensorBase(params_.Q, "Q");  // ❌ Fails for CUDATensorBase
if (!Q_base) {
    LOG_ERROR("[AttentionComputeStage] GPU tensors not yet supported");
    return false;
}
```

**FIX REQUIRED**: Dispatch via `ITensor*` polymorphism:

```cpp
// AttentionComputeStage.cpp - FIXED (supports GPU)
if (params_.Q->is_on_gpu()) {
    // GPU path: use CUDAFlashAttentionKernelT directly
    auto kernel = KernelFactory::createAttention(params_.Q, DeviceType::CUDA);
    kernel->compute_tensor(params_.Q, params_.K, params_.V, params_.output, ...);
} else {
    // CPU path: existing TensorBase* logic
    auto *Q_base = dynamic_cast<TensorBase*>(params_.Q);
    // ...
}
```

---

### Hook 4: Tensor ensureOnDevice() 

**File**: `src/v2/tensors/TensorBase.cpp`

```cpp
bool TensorBase::ensureOnDevice(int target_device) {
    if (is_on_device(target_device)) return true;
    
    // Get backend for target device
    auto* backend = BackendManager::getBackendForDeviceIdx(target_device);
    if (!backend) return false;
    
    // Allocate device memory
    size_t bytes = byte_size();
    device_data_ptr_ = backend->allocate(bytes, getBackendDeviceId(target_device));
    
    // Upload via sync_to_device() - VIRTUAL, per-tensor implementation
    return sync_to_device(target_device);
}
```

**Per-Tensor Hooks** (implement `sync_to_device()`):

| File | Method | Status |
|------|--------|--------|
| `FP32Tensor.cpp` | `sync_to_device()` | ❌ Stub |
| `BF16Tensor.cpp` | `sync_to_device()` | ❌ Stub |
| `FP16Tensor.cpp` | `sync_to_device()` | ❌ Stub |
| `Q8_0Tensor.cpp` | `sync_to_device()` | ❌ Stub |
| `Q4_0Tensor.cpp` | `sync_to_device()` | ❌ Stub |

---

### Hook 5: DeviceGraphExecutor Multi-Device Dispatch

**File**: `src/v2/execution/DeviceGraphExecutor.cpp`

```cpp
bool DeviceGraphExecutor::executeMultiDevice(
    ComputeGraph& graph,
    const std::unordered_map<int, IDeviceContext*>& contexts) 
{
    for (auto* node : sorted_nodes) {
        // Find context for this node's device
        IDeviceContext* ctx = default_ctx_;
        if (node->device_idx >= 0) {
            auto it = contexts.find(node->device_idx);
            if (it != contexts.end()) {
                ctx = it->second;
            }
        }
        
        // Execute on selected device
        node->stage->execute(ctx);
    }
}
```

**Status**: ✅ Already wired - nodes with `device_idx >= 1` route to GPU context.

---

### Hook 6: PlacementPlan Per-Layer Device

**File**: `src/v2/pipelines/qwen/PlacementPlan.h`

```cpp
struct LayerPlacement {
    int layer_idx = -1;
    PlacementDevice device = PlacementDevice::CPU;
    
    // Fine-grained: attention vs FFN can be on different devices
    PlacementDevice attention_device = PlacementDevice::CPU;
    PlacementDevice ffn_device = PlacementDevice::CPU;
    bool split_attention_ffn = false;
    
    int getAttentionDeviceIdx() const;
    int getFFNDeviceIdx() const;
};

struct PlacementPlan {
    std::vector<LayerPlacement> layer_placements;
    int getAttentionDevice(int layer_idx) const;
    int getFFNDevice(int layer_idx) const;
};
```

**Status**: ⚠️ Data structures exist, not wired to Qwen2Graph.

**Qwen2Graph Integration** (needed):
```cpp
// In Qwen2Graph::buildLayerGraphs()
for (int layer = 0; layer < n_layers; ++layer) {
    int attn_device = placement_plan_.getAttentionDevice(layer);  // NEW
    int ffn_device = placement_plan_.getFFNDevice(layer);          // NEW
    
    buildAttentionGraph(..., attn_device);
    buildFFNGraph(..., ffn_device);
}
```

---

## Prototypes

### Prototype 1: CuBLAS GEMM Wrapper (FP32/FP16/BF16 weights only)

**New File**: `src/v2/kernels/cuda/CuBLASGemmKernel.cuh`

```cpp
#pragma once
#include "../interfaces/ITensorGemm.h"
#include <cublas_v2.h>

namespace llaminar2::cuda {

/**
 * @brief cuBLAS-based GEMM kernel for FP32/FP16/BF16 **weights only**
 * 
 * For quantized weights, use CudaQuantizedGemmKernel instead (INT8 Tensor Cores).
 */
class CuBLASGemmKernel : public ITensorGemm {
public:
    enum class Precision { FP32, FP16, BF16 };
    
    CuBLASGemmKernel(int device_id, Precision precision);
    ~CuBLASGemmKernel();
    
    // ITensorGemm interface
    void multiply(const void* A, const void* B, void* C,
                  size_t M, size_t N, size_t K,
                  bool transA = false, bool transB = false) override;
    
private:
    cublasHandle_t handle_;
    int device_id_;
    Precision precision_;
};

// Factory function
std::unique_ptr<ITensorGemm> createCuBLASGemm(int device_id, CuBLASGemmKernel::Precision prec);

} // namespace llaminar2::cuda
```

---

### Prototype 2: CUDA Quantized GEMM Adapter (INT8 Tensor Cores)

**New File**: `src/v2/kernels/cuda/CudaQuantizedGemmKernel.h`

```cpp
#pragma once
#include "../interfaces/ITensorGemm.h"
#include "CudaGemmKernelPhase7_CUTLASS.h"
#include <memory>

namespace llaminar2::cuda {

/**
 * @brief INT8 Tensor Core GEMM for quantized weights
 * 
 * Wraps CudaGemmKernelPhase7_CUTLASS with ITensorGemm interface.
 * 
 * Strategy (matches CPU QuantisedGemmKernel):
 * 1. Quantize FP32 activations → INT8 (per-row symmetric)
 * 2. Use pre-converted INT8 weights (from quantized format)
 * 3. CUTLASS int8×int8→int32 GEMM (Tensor Cores)
 * 4. Dequantize: result × scale_A × scale_B
 * 
 * @note Requires SM 8.0+ (Ampere) for Tensor Core int8 instructions
 */
class CudaQuantizedGemmKernel : public ITensorGemm {
public:
    /**
     * @brief Construct from pre-converted INT8 weights
     * 
     * @param B_int8 Pre-converted INT8 weights [K×N] (column-major for Tensor Cores)
     * @param scales_B Per-column scale factors [N]
     * @param N Number of output features
     * @param K Number of input features
     * @param device_id CUDA device ID
     */
    CudaQuantizedGemmKernel(const int8_t* B_int8, const float* scales_B,
                            int N, int K, int device_id);
    ~CudaQuantizedGemmKernel();
    
    /**
     * @brief Quantized GEMM: C = A × B (with quantization/dequantization)
     * 
     * @param A FP32 activations [M×K] (will be quantized on-the-fly)
     * @param B Ignored (uses pre-converted B_int8 from constructor)
     * @param C FP32 output [M×N]
     * @param M Batch/sequence dimension
     * @param N Output features
     * @param K Input features
     */
    void multiply(const void* A, const void* B, void* C,
                  size_t M, size_t N, size_t K,
                  bool transA = false, bool transB = false) override;
    
private:
    std::unique_ptr<llaminar::v2::CudaGemmKernelPhase7_CUTLASS> cutlass_gemm_;
    const int8_t* B_int8_;      // Pre-converted weights [K×N]
    const float* scales_B_;     // Per-column scales [N]
    int N_, K_;
    int device_id_;
    
    // Device buffers for activation quantization
    int8_t* d_A_int8_ = nullptr;
    float* d_scales_A_ = nullptr;
    int allocated_M_ = 0;
};

/**
 * @brief Pre-convert quantized tensor to INT8 format for GPU GEMM
 * 
 * Call this once per weight tensor during model loading or ensureOnDevice().
 * 
 * @param tensor Quantized weight tensor (Q4_0, Q8_0, IQ4_NL, etc.)
 * @param B_int8 Output: INT8 weights [K×N] column-major
 * @param scales_B Output: Per-column scales [N]
 * @return true on success
 */
bool preconvertQuantizedToInt8(const TensorBase* tensor, 
                               int8_t* B_int8, float* scales_B);

/**
 * @brief Factory: Create CUDA quantized GEMM from weight tensor
 * 
 * Handles pre-conversion and kernel creation.
 */
std::unique_ptr<ITensorGemm> createCudaQuantizedGemm(const TensorBase* weights, 
                                                      int device_id);

} // namespace llaminar2::cuda
```

---

### Prototype 3: Weight Pre-Conversion for Q8_0Tensor

**File**: `src/v2/tensors/Q8_0Tensor.cpp` (add to existing)

```cpp
/**
 * @brief Pre-convert Q8_0 weights to INT8 format for CUDA Tensor Cores
 * 
 * Q8_0 format: 32 elements per block, each block has:
 * - scale (FP16): max_abs / 127
 * - quants[32] (INT8): quantized values
 * 
 * Output format for CUTLASS:
 * - B_int8 [K×N] column-major INT8 values
 * - scales_B [N] per-column (or per-block) FP32 scales
 */
bool Q8_0Tensor::preconvert_to_int8_cuda(int8_t* B_int8, float* scales_B) const {
    const int N = rows();  // Output features (weight rows)
    const int K = cols();  // Input features (weight columns)
    const int K_blocks = (K + 31) / 32;
    
    // Process each output row
    for (int n = 0; n < N; ++n) {
        // Compute aggregate scale for this row (max of block scales)
        float row_max_scale = 0.0f;
        for (int k_blk = 0; k_blk < K_blocks; ++k_blk) {
            const Q8_0Block& block = blocks_[n * K_blocks + k_blk];
            float block_scale = fp16_to_fp32(block.d);  // FP16 → FP32
            row_max_scale = std::max(row_max_scale, block_scale);
        }
        
        // Rescale each block's INT8 values to common scale and pack
        float inv_row_scale = (row_max_scale > 0) ? (1.0f / row_max_scale) : 1.0f;
        
        for (int k_blk = 0; k_blk < K_blocks; ++k_blk) {
            const Q8_0Block& block = blocks_[n * K_blocks + k_blk];
            float block_scale = fp16_to_fp32(block.d);
            float rescale = block_scale * inv_row_scale;
            
            for (int i = 0; i < 32; ++i) {
                int k = k_blk * 32 + i;
                if (k < K) {
                    // Column-major layout for CUTLASS: [K×N]
                    int8_t rescaled = (int8_t)std::round(block.qs[i] * rescale);
                    B_int8[k * N + n] = rescaled;
                }
            }
        }
        
        scales_B[n] = row_max_scale;  // Per-column scale
    }
    
    return true;
}
```

---

### Prototype 4: KernelFactory CUDA Dispatch for Quantized Tensors

**File**: `src/v2/kernels/KernelFactory.cpp` (modify existing)

```cpp
#ifdef HAVE_CUDA
#include "cuda/CuBLASGemmKernel.cuh"
#include "cuda/CudaQuantizedGemmKernel.h"
#endif

std::unique_ptr<ITensorGemm> KernelFactory::createGemm(
    const Q8_0Tensor* tensor, DeviceType dev_type) 
{
    switch (dev_type) {
    case DeviceType::CPU:
        // Existing CPU path: QuantisedGemmKernel with VNNI
        auto* packed = ensurePackedWeightsInTensorCache(tensor);
        return std::make_unique<QuantisedGemmKernel>(packed);
        
#ifdef HAVE_CUDA
    case DeviceType::CUDA: {
        // INT8 Tensor Core path: pre-convert weights, use CUTLASS
        int device_idx = tensor->device_index();
        if (device_idx < 0) {
            device_idx = DeviceManager::instance().find_device(ComputeBackendType::GPU_CUDA);
        }
        int gpu_id = BackendManager::getBackendDeviceId(device_idx);
        
        // Check if pre-converted INT8 weights are cached
        auto& cache = tensor->cache_;
        if (!cache.has_value()) {
            // Pre-convert quantized weights to INT8 format
            // This is done ONCE per tensor (amortized cost)
            cache = preconvertAndCacheInt8Weights(tensor, gpu_id);
        }
        
        auto& cached = std::any_cast<CudaInt8WeightsCache&>(cache);
        return std::make_unique<cuda::CudaQuantizedGemmKernel>(
            cached.B_int8, cached.scales_B, 
            tensor->rows(), tensor->cols(), gpu_id);
    }
#endif

    default:
        throwUnsupportedBackend(dev_type, "Q8_0");
        return nullptr;
    }
}

// Similar pattern for Q4_0Tensor, IQ4_NLTensor, etc.
```

---


### Directory Structure

```
src/v2/
├── devices/
│   ├── DeviceContext.h           # IDeviceContext interface (existing)
│   ├── CPUDeviceContext.cpp      # CPU implementation (existing)
│   ├── CUDADeviceContext.h       # NEW: CUDA context header
│   └── CUDADeviceContext.cu      # NEW: CUDA context implementation
├── kernels/
│   ├── cpu/                      # Existing CPU kernels
│   └── cuda/                     # NEW: CUDA kernels
│       ├── CUDAKernelCommon.cuh  # Shared utilities
│       ├── gemm/
│       │   ├── CUDAGemmKernel.cuh
│       │   └── CUDAGemmKernel.cu
│       ├── attention/
│       │   ├── CUDAAttentionKernel.cuh
│       │   └── CUDAAttentionKernel.cu
│       ├── normalization/
│       │   ├── CUDARMSNormKernel.cuh
│       │   └── CUDARMSNormKernel.cu
│       ├── activation/
│       │   ├── CUDASwiGLUKernel.cuh
│       │   └── CUDASwiGLUKernel.cu
│       └── rope/
│           ├── CUDARoPEKernel.cuh
│           └── CUDARoPEKernel.cu
└── tensors/
    └── gpu/                      # NEW: GPU tensor utilities
        ├── GPUMemoryPool.cuh     # Memory pooling
        └── TensorTransfer.cu     # Host↔Device transfers
```

### Execution Flow (Target State)

```
GraphOrchestrator
    │
    ├─> PlacementStrategy::compute() → PlacementPlan
    │       └─> Determines which layers go on which device
    │
    ├─> Qwen2Graph::build*Graph(placement_plan)
    │       └─> Creates stages with per-layer device_idx
    │
    └─> DeviceGraphExecutor::executeMultiDevice(graph, {cpu_ctx, cuda_ctx})
            │
            ├─> CPU nodes → CPUDeviceContext → CPU kernels
            └─> GPU nodes → CUDADeviceContext → CUDA kernels
```

---

## Implementation Phases

### Phase 1: Foundation (Week 1-2)
**Goal**: Establish CUDA execution infrastructure

> **Key Finding**: Much of Phase 1 is already implemented! The infrastructure is more complete than initially assessed.

#### 1.1 CUDADeviceContext ✅ ALREADY IMPLEMENTED

**Location**: [src/v2/execution/DeviceContext.cpp](../../src/v2/execution/DeviceContext.cpp) lines 344-450

The `CUDADeviceContext` class is fully implemented and functional:

| Method | Status | Implementation |
|--------|--------|----------------|
| `deviceIndex()` | ✅ | Returns logical device index |
| `backendType()` | ✅ | Returns `GPU_CUDA` |
| `deviceType()` | ✅ | Returns `DeviceType::CUDA` |
| `deviceName()` | ✅ | Delegates to `CUDABackend::deviceName()` |
| `synchronize()` | ✅ | `cudaDeviceSynchronize()` via CUDABackend |
| `barrier()` | ✅ | Same as `synchronize()` |
| `allocate()` | ✅ | `cudaMalloc()` via CUDABackend |
| `free()` | ✅ | `cudaFree()` via CUDABackend |
| `getWorkspace()` | ✅ | Growable device workspace (inherited from IGPUDeviceContext) |
| `availableMemory()` | ✅ | `cudaMemGetInfo()` free |
| `totalMemory()` | ✅ | `cudaMemGetInfo()` total |
| `copyToDevice()` | ✅ | `cudaMemcpy()` H2D via CUDABackend |
| `copyToHost()` | ✅ | `cudaMemcpy()` D2H via CUDABackend |
| `copyFromDevice()` | ✅ | Cross-device copy (via host staging) |

**Factory Integration** (DeviceContext.cpp line 49):
```cpp
case ComputeBackendType::GPU_CUDA:
    return std::make_unique<CUDADeviceContext>(device_idx, device.device_id);
```

#### 1.2 CUDABackend ✅ ALREADY IMPLEMENTED

**Location**: [src/v2/backends/cuda/CUDABackend.cu](../../../../src/v2/backends/cuda/CUDABackend.cu)

Full CUDA runtime wrapper implementing `IBackend`:

| Method | Status | CUDA API |
|--------|--------|----------|
| `deviceToHost()` | ✅ | `cudaMemcpy(..., cudaMemcpyDeviceToHost)` |
| `hostToDevice()` | ✅ | `cudaMemcpy(..., cudaMemcpyHostToDevice)` |
| `allocate()` | ✅ | `cudaMalloc()` |
| `free()` | ✅ | `cudaFree()` |
| `synchronize()` | ✅ | `cudaDeviceSynchronize()` |
| `setDevice()` | ✅ | `cudaSetDevice()` |
| `deviceMemoryFree()` | ✅ | `cudaMemGetInfo()` |
| `deviceMemoryTotal()` | ✅ | `cudaMemGetInfo()` |
| `deviceName()` | ✅ | Device name string |

#### 1.3 Tensor GPU Transfer ✅ PARTIALLY IMPLEMENTED

**Location**: [src/v2/tensors/Tensors.h](../../../../src/v2/tensors/Tensors.h) lines 750-830

**TensorBase** provides lazy transfer infrastructure:

| Method | Status | Description |
|--------|--------|-------------|
| `ensureOnDevice(target)` | ✅ | Lazy upload to GPU |
| `ensureOnHost()` | ✅ | Lazy download from GPU |
| `releaseDeviceMemory()` | ✅ | Free GPU buffer, keep host |
| `isOnGPU()` | ✅ | Check GPU residency |
| `isOnCPU()` | ✅ | Check host validity |
| `is_on_device(idx)` | ✅ | Check specific device |

**State Tracking**:
```cpp
void* device_data_ptr_ = nullptr;  // GPU buffer
bool host_invalid_ = false;        // True if GPU has newer data
int gpu_device_idx_ = -1;          // Which GPU
```

**Per-Tensor Stubs** (need implementation):

| Tensor Type | `sync_to_device()` | `sync_from_device()` | Status |
|-------------|-------------------|---------------------|--------|
| FP32Tensor | ❌ Stub | ❌ Stub | Logs error, returns false |
| BF16Tensor | ❌ Stub | ❌ Stub | Needs implementation |
| FP16Tensor | ❌ Stub | ❌ Stub | Needs implementation |
| Q8_0Tensor | ❌ Stub | ❌ Stub | Upload blocks as-is |
| Q4_0Tensor | ❌ Stub | ❌ Stub | Upload blocks as-is |
| IQ4_NLTensor | ❌ Stub | ❌ Stub | Upload blocks as-is |

#### 1.4 Build System ✅ ALREADY COMPLETE

**Location**: [src/v2/CMakeLists.txt](../../../../src/v2/CMakeLists.txt)

CUDA build configuration is complete:

```cmake
# CUDA enabled by default
option(HAVE_CUDA "Enable CUDA backend" ON)

# Compute capabilities (Turing → Hopper)
-gencode=arch=compute_75,code=sm_75  # Turing (RTX 20xx)
-gencode=arch=compute_80,code=sm_80  # Ampere (A100)
-gencode=arch=compute_86,code=sm_86  # Ampere (RTX 30xx)
-gencode=arch=compute_89,code=sm_89  # Ada (RTX 40xx)
-gencode=arch=compute_90,code=sm_90  # Hopper (H100)

# Libraries linked
CUDA::cudart      # Runtime API
CUDA::nvrtc       # JIT compilation
CUDA::cuda_driver # Driver API
CUDA::cublas      # GEMM
CUDA::nvml        # GPU NUMA detection
```

**Existing CUDA Files** (19 total in `kernels/cuda/`):
- `CudaGemmVariantsBaseline.cu` - GEMM implementations
- `CudaGemmAutoTuner.cu` - Auto-tuning
- `CudaGemmJIT*.cu` - JIT kernel infrastructure
- `IQ4_NL_BlockDecoder.cu` - Quantization decode
- `CUDABackend.cu` - Backend implementation

---

### Phase 1 Revised: Remaining Work

Given the existing infrastructure, Phase 1 actual work is:

#### 1.1 Implement Per-Tensor `sync_to_device()` / `sync_from_device()`

**What**: Enable FP32Tensor, BF16Tensor, and quantized tensors to upload/download to GPU.

**Files to modify**:
- `src/v2/tensors/FP32Tensor.cpp` - Replace stub with real implementation
- `src/v2/tensors/BF16Tensor.cpp` - Add implementation
- `src/v2/tensors/FP16Tensor.cpp` - Add implementation
- `src/v2/tensors/Quantized*.cpp` - Upload quantized blocks

**Pattern** (FP32Tensor example):
```cpp
bool FP32Tensor::sync_to_device(int target_device) {
    if (device_data_ptr_ && gpu_device_idx_ == target_device) {
        return true;  // Already on target device
    }
    
    auto* backend = BackendManager::getBackendForDeviceIdx(target_device);
    if (!backend) return false;
    
    int gpu_id = getBackendDeviceId(target_device);
    size_t bytes = byte_size();
    
    // Allocate if needed
    if (!device_data_ptr_) {
        device_data_ptr_ = backend->allocate(bytes, gpu_id);
        if (!device_data_ptr_) return false;
    }
    
    // Upload
    if (!backend->hostToDevice(device_data_ptr_, data_.data(), bytes, gpu_id)) {
        return false;
    }
    
    gpu_device_idx_ = target_device;
    return true;
}
```

#### 1.2 Wire Up KernelFactory GPU Dispatch

**What**: Enable `KernelFactory::createGemm()` to return CUDA kernels for all tensor types.

**Current State**: Only `IQ4_NL` has CUDA GEMM dispatch. Other types fall through to CPU.

**Files to modify**:
- `src/v2/kernels/KernelFactory.cpp` - Add CUDA dispatch for:
  - `Q8_0Tensor` → CUDA Q8_0 GEMM
  - `Q4_0Tensor` → CUDA Q4_0 GEMM
  - `FP32Tensor` → cuBLAS SGEMM
  - `FP16Tensor` → cuBLAS HGEMM / Tensor Cores
  - `BF16Tensor` → cuBLAS BFGEMM

**Pattern** (already exists for IQ4_NL):
```cpp
case DeviceType::CUDA:
#ifdef HAVE_CUDA
    return cuda::createCudaGemm(tensor);
#endif
    break;
```

#### 1.3 Add cuBLAS GEMM Wrapper

**What**: Create CUDA GEMM kernel using cuBLAS for FP32/FP16 tensors.

**New files**:
- `src/v2/kernels/cuda/CuBLASGemmKernel.cuh` - Interface
- `src/v2/kernels/cuda/CuBLASGemmKernel.cu` - Implementation

**Implementation sketch**:
```cpp
class CuBLASGemmKernel : public ITensorGemm {
public:
    CuBLASGemmKernel(cublasHandle_t handle, int device_id);
    
    void multiply(const float* A, const float* B, float* C,
                  size_t M, size_t N, size_t K,
                  bool transA, bool transB) override;
                  
private:
    cublasHandle_t handle_;
    int device_id_;
};
```

#### 1.4 Integration Test

**What**: End-to-end test that runs a single GEMM on GPU.

**New file**: `tests/v2/integration/Test__CUDAGemmBasic.cpp`

```cpp
TEST(Test__CUDAGemm, BasicFP32Multiply) {
    // Skip if no CUDA
    auto& dm = DeviceManager::instance();
    if (!dm.has_gpu()) GTEST_SKIP();
    
    int gpu_idx = dm.find_device(ComputeBackendType::GPU_CUDA);
    
    // Create FP32 tensors
    auto A = TensorFactory::createFP32({128, 256}, gpu_idx);
    auto B = TensorFactory::createFP32({256, 64}, gpu_idx);
    auto C = TensorFactory::createFP32({128, 64}, gpu_idx);
    
    // Upload to GPU
    ASSERT_TRUE(A->ensureOnDevice(gpu_idx));
    ASSERT_TRUE(B->ensureOnDevice(gpu_idx));
    
    // Create CUDA GEMM kernel
    auto gemm = KernelFactory::createGemm(A.get(), DeviceType::CUDA);
    ASSERT_NE(gemm, nullptr);
    
    // Execute
    gemm->multiply(A->device_data(), B->device_data(), C->mutable_device_data(),
                   128, 64, 256, false, false);
    
    // Download and verify
    ASSERT_TRUE(C->ensureOnHost());
    // ... verify results against CPU reference
}
```

### Phase 2: Core Kernels (Week 3-4)
**Goal**: Implement performance-critical kernels

> **Design Principle**: Quantized weights use INT8 Tensor Core GEMM (like CPU `QuantisedGemmKernel`), NOT cuBLAS float GEMM. This matches the CPU pattern: quantize activations to Q8_1 → INT8×INT8 GEMM → dequantize result.

#### 2.0 CUDA GEMM Strategy by Weight Type

| Weight Type | Strategy | Implementation |
|-------------|----------|----------------|
| **FP32/FP16/BF16** | cuBLAS | `cublasSgemm` / `cublasHgemm` |
| **Quantized (Q4_0, Q8_0, IQ4_NL, etc.)** | INT8 Tensor Cores | Quantize activations → CUTLASS int8×int8 GEMM |

**Why INT8 Tensor Cores for Quantized Weights:**
- INT8 Tensor Cores (IMMA instructions) are **2-4× faster** than FP16 Tensor Cores
- Matches CPU architecture: `QuantisedGemmKernel` uses AVX-512 VNNI (INT8×INT8)
- Preserves quantization benefits (smaller memory footprint, faster loads)
- Existing infrastructure: Phase 6 (DP4A) and Phase 7 (CUTLASS) already implemented

#### 2.1 Quantized CUDA GEMM Architecture

**CPU QuantisedGemmKernel Flow** (what we replicate on GPU):
```
FP32 activations [M×K] → Quantize to Q8_1 → INT8 activations [M×K] + scales[M]
                                              ↓
                            INT8×INT8 GEMM (VNNI vpdpbusd)
                                              ↓
                           Dequantize: INT32 result × scale_A × scale_B → FP32 [M×N]
```

**CUDA Quantized GEMM Flow** (Phase 7 CUTLASS):
```
FP32 activations [M×K] → CUDA kernel: quantize_A_kernel() → INT8 activations + scales_A
                                              ↓
                         CUTLASS int8×int8→int32 GEMM (Tensor Cores)
                            - Uses mma.sync.m16n8k32 (Ampere+)
                            - 128×128×64 thread block tiles
                                              ↓
                         CUDA kernel: apply_scaling_kernel() → FP32 [M×N]
                            - C_fp32[i,j] = C_int32[i,j] × scales_A[i] × scales_B[j]
```

#### 2.2 Weight Pre-Conversion (Critical Optimization)

**Key Insight from Phase 7**: Converting IQ4_NL → INT8 at runtime is a bottleneck. Solution: **pre-convert weights at model load time**.

**Pre-conversion during model loading**:
```cpp
// At model load (once per weight tensor)
void preconvertWeightsToInt8(const IQ4_NLTensor* weights, 
                              int8_t* B_int8,      // [K×N] pre-converted
                              float* scales_B) {   // [N] per-column scales
    for (int n = 0; n < N; ++n) {
        for (int k_blk = 0; k_blk < K_blocks; ++k_blk) {
            int8_t vals[32];
            float scale = weights->decode_block(n, k_blk, vals);
            memcpy(B_int8 + ..., vals, 32);
            scales_B[n] = scale;  // Per-column (or per-block)
        }
    }
}
```

**Benefits**:
- Conversion overhead removed from inference critical path
- Pure CUTLASS Tensor Core GEMM achieves 50-90 TFLOPS (vs 3.84 with runtime conversion)
- INT8 weights are **2× smaller** than FP16 dequantized weights

#### 2.3 Existing CUDA INT8 GEMM Infrastructure

| File | Description | Status |
|------|-------------|--------|
| `CudaGemmKernelPhase6_Int8.h` | DP4A-based INT8 GEMM (SM 6.1+) | ✅ Implemented |
| `CudaGemmKernelPhase7_CUTLASS.h/cu` | CUTLASS Tensor Core INT8 GEMM (SM 8.0+) | ✅ Implemented |
| `IQ4_NL_BlockDecoder.cu` | IQ4_NL lookup table decode | ✅ Implemented |

**CUTLASS Configuration** (from Phase7):
```cpp
using CutlassGemm = cutlass::gemm::device::Gemm<
    int8_t,                                 // ElementA (activations)
    cutlass::layout::RowMajor,              // LayoutA
    int8_t,                                 // ElementB (weights)
    cutlass::layout::ColumnMajor,           // LayoutB (required for Tensor Cores)
    int32_t,                                // ElementOutput
    cutlass::layout::RowMajor,              // LayoutC
    int32_t,                                // Accumulator
    cutlass::arch::OpClassTensorOp,         // Tensor Cores!
    cutlass::arch::Sm80,                    // Ampere SM 8.0+
    cutlass::gemm::GemmShape<128, 128, 64>, // ThreadblockShape
    cutlass::gemm::GemmShape<64, 64, 64>,   // WarpShape
    cutlass::gemm::GemmShape<16, 8, 32>     // InstructionShape (mma.sync.m16n8k32)
>;
```

---

### 📦 Weight Packing Format (CUDA vs CPU)

This section documents the **definitive weight packing format** for CUDA INT8 Tensor Core GEMM.

#### Hardware Requirement: Column-Major Weights

CUTLASS requires the B matrix (weights) in **Column-Major** layout for Tensor Core INT8 instructions. This is a hardware constraint - the `mma.sync.m16n8k32` instruction expects this memory access pattern for optimal coalescing.

From [CudaGemmKernelPhase7_CUTLASS.cu#L16-21](../../src/v2/kernels/cuda/CudaGemmKernelPhase7_CUTLASS.cu#L16-21):
```cpp
using CutlassGemm = cutlass::gemm::device::Gemm<
    int8_t, cutlass::layout::RowMajor,      // A: activations (row-major)
    int8_t, cutlass::layout::ColumnMajor,   // B: weights (MUST be column-major!)
    ...
>;
```

From the `execute()` method [line 322](../../src/v2/kernels/cuda/CudaGemmKernelPhase7_CUTLASS.cu#L322):
```cpp
{impl_->d_B_int8, N},  // TensorRef B: leading dimension = N means column-major
```

#### CUDA vs CPU Weight Format Comparison

| Property | CPU (VNNI) | CUDA (CUTLASS) |
|----------|------------|----------------|
| **Memory Layout** | `[N/64][K/4][64][4]` (VNNI-optimal) | `[K × N]` column-major |
| **Block Size** | 32 elements | Per-column (all K elements) |
| **Scale Storage** | `[K_blocks × N]` per-block FP32 | `[N]` per-column FP32 |
| **Compensation** | INT32 row sums (for INT8→UINT8) | Not needed (symmetric quant) |
| **Min Values** | FP32 per-block | Not needed (symmetric quant) |
| **Quantization** | Asymmetric (min/max per block) | Symmetric (max_abs per column) |

#### Data Structures

**CUDA Packed Weights**:
```cpp
struct CudaPackedWeights {
    int8_t* B_int8;        // Column-major INT8 weights [K × N]
                           // Access: B_int8[k * N + n] for element (k, n)
    float* scales_B;       // Per-column scale factors [N]
    int K;                 // Input dimension (rows in column-major)
    int N;                 // Output dimension (columns)
};
```

**CPU Packed Weights** (for comparison):
```cpp
struct QuantisedPackedWeights {
    int8_t* packed_data;   // VNNI layout [N/64][K/4][64][4]
    int32_t* compensation; // Row sums [K_blocks × N]
    float* scales;         // Per-block scales [K_blocks × N]
    float* mins;           // Per-block mins [K_blocks × N] (if asymmetric)
    int K, N, K_blocks;
};
```

#### Why Symmetric Quantization for CUDA?

CPU `QuantisedGemmKernel` uses **asymmetric** quantization (stores both scale and min per block) because VNNI instructions operate on unsigned INT8 (UINT8). The compensation term corrects for the signed→unsigned conversion.

CUDA INT8 Tensor Cores work directly with **signed INT8**, so:
- No need for min values (symmetric around zero)
- No need for compensation (no signed→unsigned conversion)
- Simpler per-column quantization (vs per-block)

#### Pre-Conversion Algorithm

**Input**: Quantized tensor (Q4_0, Q8_0, IQ4_NL, etc.) with dimensions [N, K]  
**Output**: Column-major INT8 [K × N] + per-column scales [N]

```cpp
void preconvert_to_cuda_int8(const TensorBase* tensor, 
                              int8_t* B_int8, float* scales_B) {
    int N = tensor->rows();      // Output features
    int K = tensor->cols();      // Input features
    int K_blocks = (K + 31) / 32;
    
    // Temporary buffer for dequantization
    std::vector<float> col_fp32(K);
    
    for (int n = 0; n < N; ++n) {
        // Step 1: Dequantize this row to FP32
        float max_abs = 0.0f;
        for (int k_blk = 0; k_blk < K_blocks; ++k_blk) {
            float temp[32];
            tensor->dequantize_block(n, k_blk, temp);  // Tensor-specific dequant
            
            for (int i = 0; i < 32 && (k_blk * 32 + i) < K; ++i) {
                int k = k_blk * 32 + i;
                col_fp32[k] = temp[i];
                max_abs = std::max(max_abs, std::abs(temp[i]));
            }
        }
        
        // Step 2: Compute per-column scale (symmetric)
        scales_B[n] = (max_abs > 0.0f) ? (max_abs / 127.0f) : 1.0f;
        float inv_scale = (max_abs > 0.0f) ? (127.0f / max_abs) : 1.0f;
        
        // Step 3: Quantize to INT8 and pack column-major
        for (int k = 0; k < K; ++k) {
            int8_t q = (int8_t)std::roundf(std::clamp(col_fp32[k] * inv_scale, -127.0f, 127.0f));
            B_int8[k * N + n] = q;  // Column-major: stride is N
        }
    }
}
```

#### Integration with KernelFactory

The `KernelFactory::ensureCudaPackedWeights()` function pre-converts weights:

```cpp
// In KernelFactory.cpp

struct CudaInt8WeightsCache {
    std::unique_ptr<int8_t[]> B_int8;     // Host buffer [K×N]
    std::unique_ptr<float[]> scales_B;    // Host buffer [N]
    int8_t* d_B_int8 = nullptr;           // Device buffer
    float* d_scales_B = nullptr;          // Device buffer
    int K, N;
};

const CudaInt8WeightsCache* ensureCudaPackedWeights(const TensorBase* tensor, int device_id) {
    // Check cache (stored in tensor's auxiliary data)
    auto* cache = tensor->getCudaCache<CudaInt8WeightsCache>();
    if (cache) return cache;
    
    // Pre-convert
    int K = tensor->cols(), N = tensor->rows();
    auto new_cache = std::make_unique<CudaInt8WeightsCache>();
    new_cache->B_int8 = std::make_unique<int8_t[]>(K * N);
    new_cache->scales_B = std::make_unique<float[]>(N);
    new_cache->K = K;
    new_cache->N = N;
    
    preconvert_to_cuda_int8(tensor, new_cache->B_int8.get(), new_cache->scales_B.get());
    
    // Upload to GPU
    cudaSetDevice(device_id);
    cudaMalloc(&new_cache->d_B_int8, K * N * sizeof(int8_t));
    cudaMalloc(&new_cache->d_scales_B, N * sizeof(float));
    cudaMemcpy(new_cache->d_B_int8, new_cache->B_int8.get(), K * N, cudaMemcpyHostToDevice);
    cudaMemcpy(new_cache->d_scales_B, new_cache->scales_B.get(), N * sizeof(float), cudaMemcpyHostToDevice);
    
    return tensor->setCudaCache(std::move(new_cache));
}
```

#### Memory Requirements

| Format | Size for [4096 × 4096] Weight |
|--------|------------------------------|
| Original Q4_0 | 8.5 MB (4 bits/element + overhead) |
| CPU VNNI Packed | 16.8 MB (INT8 + scales + comp) |
| CUDA Column-Major INT8 | 16.0 MB (INT8 [K×N]) + 16 KB (scales [N]) |

CUDA format is slightly more compact than CPU VNNI format because it doesn't store compensation or mins.

---

#### 2.4 Work Items for Phase 2

- [x] **2.4.1 cuBLAS GEMM for FP32/FP16/BF16 weights** ✅ (Completed in Phase 1)
  - [x] Create `CuBLASGemmKernel` class
  - [x] Wire to `KernelFactory::createGemm(FP32Tensor*, DeviceType::CUDA)`
  - [x] Test: FP32 GEMM correctness vs CPU (10/10 tests passing)

- [x] **2.4.2 CUDAQuantisedGemmKernel - CUTLASS INT8 Tensor Core GEMM** ✅ (Completed Jan 2026)

  **Overview**: Create a unified CUDA quantized GEMM kernel that works with any quantized weight
  tensor type. The kernel implements `multiply_tensor()` as the primary entry point, inspecting
  input/output tensor types at runtime to route to optimal CUTLASS kernel paths.

  **Architecture**:
  ```
  ┌─────────────────────────────────────────────────────────────────────┐
  │                      KernelFactory                                   │
  │  createGemm(TensorBase* weights, DeviceType::CUDA)                  │
  │  - Supports: IQ4_NL, Q8_0, Q4_0, Q4_K, Q5_K, etc.                  │
  └─────────────────────────┬───────────────────────────────────────────┘
                            │
                            ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │                  CUDAQuantisedGemmKernel                             │
  │  - Stores weight tensor pointer (any quantized type)                │
  │  - Implements multiply_tensor(A, C, ...)                           │
  │                                                                     │
  │  multiply_tensor() dispatches based on:                            │
  │  ┌─────────────────────────────────────────────────────────────┐   │
  │  │ A->native_type()    C->native_type()    Path                │   │
  │  ├─────────────────────────────────────────────────────────────┤   │
  │  │ Q8_1               Q8_1                 INT8×INT8→INT8      │   │
  │  │ Q8_1               FP32                 INT8×INT8→FP32      │   │
  │  │ FP32               FP32                 quant→INT8×INT8→FP32│   │
  │  │ FP32               Q8_1                 quant→INT8×INT8→Q8_1│   │
  │  └─────────────────────────────────────────────────────────────┘   │
  └─────────────────────────────────────────────────────────────────────┘
  ```

  **Key Design Decisions**:
  - `multiply_tensor()` is primary entry point (not raw `multiply()`)
  - Kernel inspects `A->native_type()` and `C->native_type()` internally
  - Weight-type agnostic: works with any quantized tensor that can provide INT8 + scales
  - Weights pre-converted to INT8 + per-column scales on first use (cached)
  - Uses CUTLASS INT8 Tensor Core GEMM (SM 8.0+ Ampere)

  **Files to Create/Modify**:
  | File | Purpose |
  |------|---------|
  | `src/v2/kernels/cuda/CUDAQuantisedGemmKernel.h` | **NEW**: Header with class declaration |
  | `src/v2/kernels/cuda/CUDAQuantisedGemmKernel.cu` | **NEW**: Implementation using CUTLASS |
  | `src/v2/kernels/cuda/CuBLASGemmFactory.h` | Add factory function |
  | `src/v2/kernels/cuda/CuBLASGemmFactory.cu` | Add adapter (no tensor pointers in CUDA!) |
  | `src/v2/kernels/KernelFactory.cpp` | Update quantized CUDA paths |
  | `tests/v2/unit/Test__CUDAQuantisedGemm.cpp` | **NEW**: Unit tests |

  **Class Interface**:
  ```cpp
  class CUDAQuantisedGemmKernel : public ITensorGemm {
  public:
      CUDAQuantisedGemmKernel(const TensorBase* weights, int cuda_device_id);
      
      // Primary entry point - dispatches based on tensor types
      bool multiply_tensor(const TensorBase* A, TensorBase* C,
                           bool transpose_B, float alpha, float beta,
                           const MPIContext*, int device_idx) override;
      
      // With explicit dimensions
      bool multiply_tensor(const TensorBase* A, TensorBase* C,
                           int m, int n, int k,
                           bool transpose_B, float alpha, float beta,
                           const MPIContext*, int device_idx) override;
      
      // Raw pointer fallback (FP32→INT8→FP32 path)
      bool multiply(const float* A, float* C, int m, int n, int k,
                    bool transpose_B, float alpha, float beta,
                    const MPIContext*, int device_idx) override;
      
  private:
      // Type-specific dispatch paths
      bool multiply_q8_to_fp32(const Q8_1Block* A, float* C, int m, int n, int k, ...);
      bool multiply_q8_to_q8(const Q8_1Block* A, Q8_1Block* C, int m, int n, int k);
      bool multiply_fp32_to_fp32(const float* A, float* C, int m, int n, int k, ...);
      
      void ensureWeightsConverted();  // Lazy INT8 conversion
      
      const TensorBase* weights_;
      int cuda_device_id_;
      size_t K_, N_;
      
      // Cached INT8 weight representation
      int8_t* d_weights_int8_ = nullptr;
      float* d_scales_ = nullptr;
      bool weights_converted_ = false;
      
      // Work buffers
      int8_t* d_A_int8_ = nullptr;
      float* d_scales_A_ = nullptr;
      int32_t* d_C_int32_ = nullptr;
  };
  ```

  **Implementation Phases**:
  - [x] **2.4.2a** Create `CUDAQuantisedGemmKernel.h` header ✅
  - [x] **2.4.2b** Implement `multiply()` (FP32→INT8→FP32 path) using CUTLASS ✅
  - [x] **2.4.2c** Implement `multiply_tensor()` with type dispatch ✅
  - [x] **2.4.2d** Add weight conversion for quantized types (extract INT8 + scales) ✅
  - [x] **2.4.2e** ~~Add factory function~~ → Refactored: `KernelFactory` directly instantiates kernel classes ✅
  - [x] **2.4.2f** Delete old `CudaGemmFactory.h/cu` (ODR violation source) ✅
  - [x] **2.4.2g** Unit tests (220/220 unit, 55/55 integration passing) ✅

  **Additional work completed**:
  - [x] Created `CUDAFloatingPointGemmKernel` class (companion for FP32/FP16/BF16)
  - [x] Device tracking API cleanup (`home_dm_device_index()`, `current_dm_device_index()`, `gpu_data_ptr()`, `NOT_ON_GPU`)
  - [x] CUDA test portability audit (all tests use dynamic device detection)

- [x] **2.4.3 Weight pre-conversion infrastructure** ✅ (Inherited from Phase 7 prototype)
  - [x] Add `preconvert_to_int8()` method to quantized tensor classes
  - [x] Store pre-converted INT8 weights + scales in tensor's GPU buffer
  - [x] Pre-convert during `ensureOnDevice()` (lazy, once per tensor)

- [x] **2.4.4 Activation quantization kernel** ✅ (Inherited from Phase 7 prototype)
  - [x] Verify `quantize_A_kernel` from Phase 7 is production-ready
  - [x] Add batched quantization for prefill (M > 1)
  - [x] Optimize for decode (M = 1) case

#### 2.5 CUDA RMSNorm Kernel ✅
- [x] Parallel reduction for variance computation
- [x] Fused normalize + scale operation
- [x] Support for FP32 (FP16/BF16 type stubs ready)

#### 2.6 CUDA RoPE Kernel ✅
- [x] Rotary position embedding application
- [x] Support for FP32 precision

#### 2.7 CUDA SwiGLU Kernel ✅
- [x] Fused SiLU(gate) * up computation
- [x] Elementwise operation
- [ ] Residual add (stubbed, not yet implemented)

### Phase 3: Attention ✅ COMPLETE (January 6, 2026)
**Goal**: GPU attention implementation

#### 3.1 CUDA Flash Attention Kernel ✅
- [x] Flash Attention 2 implementation (prefill, seq_len > 1)
- [x] Flash Decoding implementation (decode, seq_len = 1)
- [x] Causal and non-causal masking
- [x] Head dimension 64 and 128 support
- [x] FP32 precision (FP16/BF16 type stubs ready)
- [x] Batch decoding support
- [x] **9/9 tests passing**

**Files Created**:
- `src/v2/kernels/cuda/attention/CUDAFlashAttentionKernelT.h` (~18KB, template header)
- `src/v2/kernels/cuda/attention/CUDAFlashAttentionKernelT.cpp` (~25KB, implementation)
- `src/v2/kernels/cuda/attention/CUDAFlashAttentionKernels.cu` (~27KB, CUDA kernels)
- `tests/v2/integration/Test__CUDAFlashAttentionParity.cpp` (~33KB, 9 tests)

**Test Results**:
| Test | Status |
|------|--------|
| FlashAttn2_FP32_Small | ✅ |
| FlashAttn2_FP32_Medium | ✅ |
| FlashAttn2_FP32_Large | ✅ |
| FlashDecode_FP32_Short | ✅ |
| FlashDecode_FP32_Long | ✅ |
| FlashAttn2_HeadDim128 | ✅ |
| FlashAttn2_NonCausal | ✅ |
| FlashAttn2_CausalMasking | ✅ |
| FlashDecode_BatchDecoding | ✅ |

#### 3.2 CUDA Ring Buffer KV Cache ✅
- [x] Ring buffer with O(1) append and eviction
- [x] Contiguous optimization (no linearization when not wrapped)
- [x] Per-sequence scratch buffers for linearization
- [x] Multi-precision: FP32, FP16, BF16
- [x] Batched gather for multi-sequence inference
- [x] **9/9 tests passing**

**Files Created**:
- `src/v2/kernels/cuda/CUDARingKVCache.h` (~20KB, template header)
- `src/v2/kernels/cuda/CUDARingKVCache.cu` (~26KB, CUDA kernels)
- `tests/v2/integration/Test__CUDARingKVCacheParity.cpp` (~20KB, 9 tests)

**Test Results**:
| Test | Status |
|------|--------|
| BasicAppendRetrieve_FP32 | ✅ |
| WrapAround_FP32 | ✅ |
| Eviction_O1 | ✅ |
| SlidingWindow | ✅ |
| BatchedGather | ✅ |
| ContiguousOptimization | ✅ |
| ClearOperations | ✅ |
| MultiPrecision_FP16 | ✅ |
| MultiPrecision_BF16 | ✅ |

### Phase 4: Integration (Week 7-8)
**Goal**: End-to-end GPU inference

**Status Update (January 7, 2026)**: Phase 4 is BLOCKED on stage implementation updates.

#### 4.0 CRITICAL BLOCKER: Stage Implementations Use `requireTensorBase()` ⛔

The root cause of GPU inference failure is that stage `.cpp` files cast `ITensor*` → `TensorBase*`:

```cpp
// ComputeStageUtils.h - the blocker
inline TensorBase *requireTensorBase(ITensor *tensor, const char *name)
{
    auto *base = dynamic_cast<TensorBase *>(tensor);
    LLAMINAR_ASSERTF(base, name << " must be a CPU TensorBase (GPU not yet supported)");
    return base;
}
```

**Stages with `requireTensorBase()` calls** (all block GPU execution):

| Stage | Impact |
|-------|--------|
| `AttentionComputeStage` | Blocks GPU attention (Q, K, V, output) |
| `GEMMStage` | Blocks GPU GEMM (A, B, C) |
| `RoPEStage` | Blocks GPU RoPE (Q, K, Q_out, K_out) |
| `RMSNormStage` | Blocks GPU normalization |
| `SwiGLUStage` | Blocks GPU FFN activation |
| `ResidualAddStage` | Blocks GPU residual connections |
| `EmbeddingStage` | Blocks GPU embedding lookup |
| `LMHeadStage` | Blocks GPU logit projection |
| `FusedAttentionWoStage` | Blocks GPU fused attention |
| `FusedGateUpGEMMStage` | Blocks GPU FFN projections |

**Fix Strategy**:
1. Add `ITensor*` overloads to KernelFactory (dispatch via `native_type()` + `is_on_gpu()`)
2. Update each stage to check `ITensor::is_on_gpu()` and dispatch accordingly
3. For GPU path: pass `ITensor*` directly to CUDA kernels (they work with device pointers)

#### 4.1 Device Routing
- [ ] Implement `GPUFirstStrategy` with memory-based layer assignment
- [ ] Modify `Qwen2Graph` to query `PlacementPlan` per-layer
- [ ] Add automatic fallback to CPU when GPU OOM

#### 4.2 Stage Implementation Updates (NEW - P0)
- [ ] Add `KernelFactory::createAttention(ITensor*, DeviceType)` overload
- [ ] Update `AttentionComputeStage` to dispatch via `ITensor::is_on_gpu()`
- [ ] Update `KVCacheAppendStage` for GPU KV cache
- [ ] Test decomposed attention path on GPU (prefill + decode)

#### 4.3 Tensor Transfers
- [ ] Automatic device migration at stage boundaries
- [ ] Overlap compute with transfers using streams
- [ ] Profile and optimize transfer overhead

#### 4.4 Testing
- [x] Unit tests for each CUDA kernel (**27 tests passing**)
- [x] Numerical parity tests vs CPU kernels
- [ ] Integration tests with real models (prefill works, **decode blocked**)
- [ ] Benchmark suite for GPU performance

**Test Status** (`Test__CUDAFullModelInference.cpp`):

| Test | Status | Notes |
|------|--------|-------|
| `CanCreateCPURunner` | ✅ Pass | |
| `CanCreateGPURunner` | ✅ Pass | |
| `SingleTokenPrediction_MatchesCPU` | ✅ Pass | Prefill only |
| `LongerPrompt_MatchesCPU` | ✅ Pass | Prefill only |
| `MultiTokenGeneration_MatchesCPU` | ⛔ **SKIP** | GPU decode blocked |
| `GPUMemoryUsage` | ✅ Pass | |
| `WeightsAreOnGPU` | ✅ Pass | |

**Skip Reason**:
```cpp
GTEST_SKIP() << "GPU decode not yet supported - FusedAttentionWoStage requires TensorBase* "
             << "but GPU KV cache returns CUDATensorBase*";
```

---

## Path Forward: Enabling GPU Decode (January 7, 2026)

The GPU prefill pipeline works, but **decode is blocked** because stages can't handle GPU tensors from the KV cache. Here's the recommended path to fix it:

### Option A: Use Decomposed Attention for GPU (Recommended) ⭐

**Rationale**: The decomposed attention path (`KVCacheAppendStage` + `AttentionComputeStage`) is designed for flexibility. Unlike `FusedAttentionWoStage`, it separates KV cache management from attention computation, making it easier to support heterogeneous tensor types.

**Steps**:
1. **Update `AttentionComputeStage` for GPU dispatch**
   ```cpp
   // In AttentionComputeStage::execute()
   if (params_.Q->is_on_gpu()) {
       // Use CUDAFlashAttentionKernelT
       auto kernel = KernelFactory::createAttention(DeviceType::CUDA);
       kernel->compute_tensor(
           params_.Q->raw_data(),  // device pointer
           params_.K->raw_data(),
           params_.V->raw_data(),
           params_.output->raw_mutable_data(),
           ...);
   } else {
       // Existing CPU path
   }
   ```

2. **Add `ITensor*` overload to KernelFactory**
   ```cpp
   std::unique_ptr<ITensorAttention> KernelFactory::createAttention(
       const ITensor* tensor, DeviceType dev_type)
   {
       // Dispatch by tensor location
       if (tensor->is_on_gpu()) {
           return createCUDAAttention(tensor);
       }
       // Fall back to TensorBase path for CPU
       return createAttention(requireTensorBase(tensor, "Q"), dev_type);
   }
   ```

3. **Ensure `KVCacheAppendStage` works with GPU cache**
   - The CUDA Ring KV Cache (`CUDARingKVCache`) already exists
   - Need to wire it into the stage's execute path

4. **Configure Qwen2Graph to use decomposed attention for GPU**
   ```cpp
   // In Qwen2Graph - when building for GPU inference
   bool use_fused_attention = !config_.use_gpu;  // Disable fused for GPU
   ```

**Estimated Effort**: 1-2 days

### Option B: Create CUDAFusedAttentionWoStage

**Rationale**: Keep the fused stage architecture but create a GPU-native version.

**Pros**: 
- Potentially faster (fewer stage transitions)
- Matches CPU architecture

**Cons**:
- More code to maintain
- Duplicates logic between CPU and GPU

**Estimated Effort**: 3-5 days

### Recommended Next Steps

1. **Immediate**: Enable decomposed attention for GPU (Option A)
   - Lower risk, reuses existing CUDA Flash Attention kernel
   - Unblocks GPU decode testing

2. **Short-term**: Add `ITensor*` overloads to KernelFactory
   - Makes all stages GPU-compatible
   - Enables gradual migration

3. **Medium-term**: Profile and optimize
   - Benchmark decomposed vs fused attention
   - Consider fused stage if perf gap is significant

---

### Phase 5: Optimization (Week 9+)
**Goal**: Production-ready performance

#### 5.1 Quantized Kernels
- [ ] Q4_0 dequantize + GEMM fused kernel
- [ ] Q8_0 GEMM kernel
- [ ] INT8 Tensor Core utilization

#### 5.2 Kernel Fusion
- [ ] RMSNorm + QKV projection fusion
- [ ] Attention + RoPE fusion
- [ ] FFN fusion (gate/up → SwiGLU → down)

#### 5.3 Multi-GPU
- [ ] Tensor parallelism across GPUs
- [ ] Pipeline parallelism for large models
- [ ] NVLink-aware communication

---

## Technical Specifications

### Supported GPUs

| Architecture | Compute Capability | Notes |
|--------------|-------------------|-------|
| Volta | sm_70 | V100, Tensor Cores |
| Turing | sm_75 | RTX 20xx, T4 |
| Ampere | sm_80, sm_86 | A100, RTX 30xx |
| Ada Lovelace | sm_89 | RTX 40xx, L4 |
| Hopper | sm_90 | H100 |

### Memory Requirements

| Model | FP16 Weights | KV Cache (4K ctx) | Total VRAM |
|-------|--------------|-------------------|------------|
| Qwen2.5-0.5B | ~1 GB | ~0.2 GB | ~1.5 GB |
| Qwen2.5-3B | ~6 GB | ~0.8 GB | ~8 GB |
| Qwen2.5-7B | ~14 GB | ~1.5 GB | ~18 GB |

### Performance Targets

| Operation | Target (RTX 4090) | Notes |
|-----------|-------------------|-------|
| **INT8 GEMM (quantized)** | 50-90 TFLOPS | CUTLASS Tensor Cores (mma.sync.m16n8k32) |
| FP16 GEMM | >200 TFLOPS | cuBLAS with Tensor Cores |
| FP32 GEMM | >80 TFLOPS | cuBLAS |
| Attention (4K seq) | <5ms per layer | Flash Attention |
| Decode throughput | >100 tok/s | Single batch, quantized weights |
| Prefill throughput | >2000 tok/s | Batch size 1, quantized weights |

**INT8 Tensor Core Performance Notes:**
- RTX 4090 INT8 peak: ~660 TOPS (vs 165 TFLOPS FP16)
- CUTLASS achieves 50-90% of peak with good tile sizes
- Memory-bound for small M (decode): activation quantization overhead matters
- Compute-bound for large M (prefill): Tensor Cores dominate

---

## Dependencies

### External Libraries

| Library | Purpose | Integration | Status |
|---------|---------|-------------|--------|
| cuBLAS | FP32/FP16/BF16 GEMM | Required | ✅ Linked |
| CUTLASS | INT8 Tensor Core GEMM | Required for quantized | ✅ Integrated |
| cuDNN | Optional convolutions | Optional | Not used |
| Flash Attention | Fast attention | Recommended | Phase 3 |
| NVRTC | JIT kernel compilation | Required | ✅ Linked |

### CUTLASS Configuration

CUTLASS is already integrated for INT8 Tensor Core GEMM (Phase 7):

```cmake
# In CMakeLists.txt
if(EXISTS "${CUTLASS_DIR}/include")
    target_include_directories(cuda_backend PUBLIC "${CUTLASS_DIR}/include")
    target_compile_definitions(cuda_backend PUBLIC HAVE_CUTLASS)
endif()
```

**CUTLASS GEMM Configuration for Quantized Weights:**
```cpp
// SM 8.0+ (Ampere, Ada, Hopper)
using GemmKernel = cutlass::gemm::device::Gemm<
    int8_t, RowMajor,      // A: quantized activations
    int8_t, ColumnMajor,   // B: pre-converted weights (col-major required!)
    int32_t, RowMajor,     // C: accumulator output
    int32_t,               // Accumulator type
    OpClassTensorOp,       // Tensor Cores
    Sm80,                  // Architecture
    GemmShape<128,128,64>, // ThreadBlock
    GemmShape<64,64,64>,   // Warp
    GemmShape<16,8,32>     // Instruction (mma.sync.m16n8k32)
>;
```

### Build Requirements

- CUDA Toolkit 11.8+ (12.x recommended)
- CMake 3.18+ (for CUDA language support)
- GCC 9+ or Clang 10+ (host compiler)

---

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Flash Attention licensing | Medium | Fallback to custom kernel |
| Memory fragmentation | High | Implement memory pool |
| Numerical divergence GPU vs CPU | Medium | Parity test suite |
| Multi-GPU synchronization bugs | High | Extensive MPI+CUDA testing |
| Build complexity | Low | Docker with CUDA support |

---

## Success Criteria

### Phase 1 Complete ✅ (January 6, 2026)
- [x] CUDA device context working
- [x] Tensors can be transferred to/from GPU
- [x] Basic CUDA kernel compiles and runs
- [x] cuBLAS FP32 GEMM integrated via KernelFactory
- [x] All 27 Phase 1 tests passing:
  - Tensor Transfer: 11/11 ✅
  - cuBLAS GEMM: 10/10 ✅
  - Integration: 6/6 ✅

**Key Architectural Fix**: Resolved ODR violation in CuBLASGemmFactory by changing factory
functions to accept extracted primitive data (device_ptr, device_idx, rows, cols) instead of
tensor pointers. This prevents CUDA code from needing minimal tensor class definitions.

### Phase 2 Complete ✅ (January 6, 2026)
- [x] cuBLAS GEMM for FP32/FP16/BF16 (10/10 tests)
- [x] CUTLASS INT8 Tensor Core GEMM for quantized weights
- [x] Weight pre-conversion infrastructure
- [x] Activation quantization kernel
- [x] CUDA RMSNorm kernel (2/2 tests)
- [x] CUDA RoPE kernel (2/2 tests)
- [x] CUDA SwiGLU kernel (2/2 tests)
- [ ] CUDA Residual Add (stubbed in SwiGLU, not yet implemented)

### Phase 3 Complete ✅ (January 6, 2026)
- [x] Flash Attention 2 for prefill (9/9 tests)
- [x] Flash Decoding for decode
- [x] Ring buffer KV cache with O(1) eviction (9/9 tests)
- [x] Multi-precision support (FP32, FP16, BF16)
- [x] Causal and non-causal masking
- [x] Sliding window attention support

### Phase 4 Complete
- [ ] Heterogeneous CPU+GPU execution works
- [ ] Automatic layer placement functional
- [ ] Performance exceeds CPU-only baseline

### Phase 5 Complete
- [ ] Quantized inference (Q4_0) on GPU
- [ ] Multi-GPU tensor parallelism works
- [ ] Production-ready stability and performance

---

## Current Focus: Phase 4 Integration

With Phases 1-3 complete (GEMM, Flash Attention, KV Cache), the next priority is wiring everything together for end-to-end GPU inference.

### Phase 2 Elementwise Kernels ✅ COMPLETE

| Kernel | Status | Tests | Notes |
|--------|--------|-------|-------|
| **RMSNorm** | ✅ Done | 2/2 | `CUDARMSNormKernelT` in `ops/` |
| **RoPE** | ✅ Done | 2/2 | `CUDARoPEKernelT` in `ops/` |
| **SwiGLU** | ✅ Done | 2/2 | `CUDASwiGLUKernelT` in `ops/` |
| **Residual Add** | ✅ Done | 4/4 | Direct CUDA kernels in `CUDAOpsKernels.cu`, stage-level dispatch |

**Files**:
- `src/v2/kernels/cuda/ops/CUDARMSNormKernelT.h` (~280 lines)
- `src/v2/kernels/cuda/ops/CUDARoPEKernelT.h` (~280 lines)
- `src/v2/kernels/cuda/ops/CUDASwiGLUKernelT.h` (~300 lines)
- `src/v2/kernels/cuda/ops/CUDAOpsKernels.cu` (~19KB, CUDA kernels + Residual Add)
- `src/v2/kernels/cuda/ops/CUDAOpsKernels.cpp` (~16KB, C++ wrappers)
- `src/v2/execution/compute_stages/stages/ResidualAddStage.cpp` (CUDA dispatch integrated)
- `tests/v2/integration/Test__CUDAOpsParity.cpp` (~19KB, 10 tests)

### Phase 4 Immediate Tasks

#### 4.1 Residual Add Implementation (P2) ✅ COMPLETE
Implemented CUDA residual add kernels (FP32, BF16, FP16) in `CUDAOpsKernels.cu`.
`ResidualAddStage` now detects GPU context and dispatches to CUDA kernels automatically.

**Completed**: January 7, 2026
- Added CUDA kernels: `residual_add_fp32_kernel`, `residual_add_bf16_kernel`, `residual_add_fp16_kernel`
- Added extern "C" wrappers: `cudaOps_residual_add_fp32/bf16/fp16`
- Updated `ResidualAddStage::executeFP32/BF16/FP16` with CUDA dispatch
- Updated `supportsBackend()` to include `GPU_CUDA`
- Added 4 parity tests in `Test__CUDAOpsParity.cpp`

#### 4.2 Stage Integration (P1) ✅ COMPLETE

Wire CUDA kernels to `ComputeStage` execution via `KernelFactory`:

| Stage | CUDA Kernel | KernelFactory Status |
|-------|-------------|---------------------|
| `GEMMStage` | `CUDAQuantisedGemmKernel` | ✅ Wired |
| `GEMMStage` (FP32) | `CuBLASGemmKernel` | ✅ Wired |
| `FusedAttentionStage` | `CUDAFlashAttentionKernelT` | ✅ Wired |
| `RMSNormStage` | `CUDARMSNormKernelT` | ✅ Wired |
| `RoPEStage` | `CUDARoPEKernelT` | ✅ Wired |
| `SwiGLUStage` | `CUDASwiGLUKernelT` | ✅ Wired |
| `ResidualAddStage` | N/A (fused) | ⚠️ Stubbed |

**Completed**: January 7, 2026
- Modified `src/v2/kernels/KernelFactory.cpp` to dispatch CUDA kernels
- All `DeviceType::CUDA` cases now return actual CUDA kernels instead of CPU fallbacks
- Tests passing: 7/7 CUDA tests (100%)

#### 4.3 Device Routing (P2)
Implement `GPUFirstStrategy` to route layers to GPU:

```cpp
class GPUFirstStrategy : public IPlacementStrategy {
    PlacementPlan compute(const ModelConfig& config,
                          const DeviceManager& dm) override {
        // Place as many layers on GPU as memory allows
        // Fallback remaining layers to CPU
    }
};
```

### Recommended Next Steps

1. ~~**Integration test**: Single transformer layer on GPU (all kernels ready)~~ ✅ Done
2. **GPUFirstStrategy** for automatic device routing (P2)
3. ~~**Wire KernelFactory** to dispatch to CUDA ops kernels~~ ✅ Done (January 7, 2026)
4. ~~**Implement residual add** in SwiGLU kernel (if performance matters)~~ ✅ Done (January 7, 2026) - standalone kernel
5. **End-to-end test**: Full model on GPU (P1)

---

## Appendix: Key Code References

### Device Context Interface
```cpp
// src/v2/devices/DeviceContext.h
class IDeviceContext {
public:
    virtual void* allocate(size_t bytes) = 0;
    virtual void deallocate(void* ptr) = 0;
    virtual bool copyToDevice(void* dst, const void* src, size_t bytes) = 0;
    virtual bool copyToHost(void* dst, const void* src, size_t bytes) = 0;
    virtual void synchronize() = 0;
};
```

### Kernel Factory Dispatch
```cpp
// src/v2/kernels/KernelFactory.h
static ITensorGemm* getOrCreateGemm(ITensor* weights, int device_idx) {
    if (device_idx >= 0 && DeviceManager::instance().isGPU(device_idx)) {
        return createCUDAGemm(weights);  // NEW
    }
    return createCPUGemm(weights);  // Existing
}
```

### PlacementPlan Query
```cpp
// src/v2/pipelines/qwen/PlacementPlan.h
struct PlacementPlan {
    int getAttentionDevice(int layer_idx) const;
    int getFFNDevice(int layer_idx) const;
};
```

---

## Phase 1 Test Strategy

This section outlines the testing approach for Phase 1 (Foundation) work. Tests should be written incrementally as each component is implemented.

### Test Categories

| Category | Location | Purpose | When to Run |
|----------|----------|---------|-------------|
| **Unit Tests** | `tests/v2/unit/Test__CUDA*.cpp` | Isolated component testing | During development, CI |
| **Integration Tests** | `tests/v2/integration/Test__CUDA*.cpp` | Multi-component workflows | Before PR merge |
| **Parity Tests** | `tests/v2/parity/Test__CUDAVsCPU*.cpp` | Numerical correctness vs CPU | Before PR merge |

### Test Naming Convention

```
Test__CUDA<Component>_<Scenario>
```

Examples:
- `Test__CUDATensorTransfer_FP32UploadDownload`
- `Test__CUDAGemm_SmallMatrixCorrectness`
- `Test__CUDAKernelFactory_DispatchesCorrectly`

### Test Infrastructure Requirements

#### 1. GPU Skip Guards

All CUDA tests MUST gracefully skip when no GPU is available:

```cpp
class CUDATestBase : public ::testing::Test {
protected:
    void SetUp() override {
        auto& dm = DeviceManager::instance();
        if (!dm.has_gpu()) {
            GTEST_SKIP() << "No CUDA GPU available";
        }
        gpu_idx_ = dm.find_device(ComputeBackendType::GPU_CUDA);
        ctx_ = IDeviceContext::create(gpu_idx_);
    }
    
    int gpu_idx_ = -1;
    std::unique_ptr<IDeviceContext> ctx_;
};
```

#### 2. Parity Comparison Utilities

Reuse from existing `Test__CUDAGemm.cpp`:

```cpp
// Already exists in Test__CUDAGemm.cpp
bool compareFloatArrays(const float* a, const float* b, size_t count,
                        float abs_tol = 1e-3f, float rel_tol = 1e-2f);
```

Add to `tests/v2/utils/CUDATestUtils.h`:

```cpp
namespace llaminar2::test::cuda {

// Skip guard for test fixtures
#define SKIP_IF_NO_CUDA() \
    do { \
        auto& dm = DeviceManager::instance(); \
        if (!dm.has_gpu()) { \
            GTEST_SKIP() << "No CUDA GPU available"; \
        } \
    } while(0)

// Tolerance constants for parity tests
constexpr float FP32_ABS_TOL = 1e-5f;   // FP32 → FP32 comparison
constexpr float FP32_REL_TOL = 1e-4f;
constexpr float QUANT_ABS_TOL = 1e-2f;  // Quantized → FP32 comparison (looser)
constexpr float QUANT_REL_TOL = 5e-2f;

// Compare arrays with detailed mismatch reporting
bool compareArrays(const float* cuda, const float* cpu, size_t count,
                   float abs_tol, float rel_tol, 
                   size_t max_print_mismatches = 5);

// Generate random test data
std::vector<float> generateRandomFP32(size_t count, float min = -1.0f, float max = 1.0f, 
                                       unsigned seed = 12345);

} // namespace
```

---

### Test 1.1: Tensor GPU Transfer

**File**: `tests/v2/unit/Test__CUDATensorTransfer.cpp`

Tests `FP32Tensor::sync_to_device()` and `sync_from_device()`.

```cpp
#include <gtest/gtest.h>
#include "tensors/FP32Tensor.h"
#include "devices/DeviceManager.h"
#include "utils/CUDATestUtils.h"

using namespace llaminar2;
using namespace llaminar2::test::cuda;

class Test__CUDATensorTransfer : public CUDATestBase {};

TEST_F(Test__CUDATensorTransfer, FP32Upload) {
    // Create FP32 tensor with known values
    auto tensor = std::make_unique<FP32Tensor>(1024, 256);
    auto host_data = generateRandomFP32(1024 * 256);
    std::memcpy(tensor->mutable_data(), host_data.data(), 
                host_data.size() * sizeof(float));
    
    // Upload to GPU
    ASSERT_TRUE(tensor->ensureOnDevice(gpu_idx_));
    EXPECT_TRUE(tensor->isOnGPU());
    EXPECT_NE(tensor->device_data(), nullptr);
    
    // Verify state tracking
    EXPECT_EQ(tensor->gpu_device_idx(), gpu_idx_);
}

TEST_F(Test__CUDATensorTransfer, FP32RoundTrip) {
    // Create tensor with known values
    auto tensor = std::make_unique<FP32Tensor>(512, 128);
    std::vector<float> original(512 * 128);
    for (size_t i = 0; i < original.size(); ++i) {
        original[i] = static_cast<float>(i) * 0.001f;
    }
    std::memcpy(tensor->mutable_data(), original.data(), 
                original.size() * sizeof(float));
    
    // Upload to GPU
    ASSERT_TRUE(tensor->ensureOnDevice(gpu_idx_));
    
    // Corrupt host data to ensure we're really testing download
    std::memset(tensor->mutable_data(), 0, tensor->byte_size());
    
    // Mark host as invalid, download from GPU
    tensor->invalidate_host();  // Force re-download
    ASSERT_TRUE(tensor->ensureOnHost());
    
    // Verify data integrity (should be bit-exact for FP32)
    EXPECT_TRUE(compareArrays(tensor->data(), original.data(), original.size(),
                              0.0f, 0.0f));  // Exact match
}

TEST_F(Test__CUDATensorTransfer, BF16Upload) {
    auto tensor = std::make_unique<BF16Tensor>(256, 64);
    // ... similar pattern
}

TEST_F(Test__CUDATensorTransfer, QuantizedBlockUpload) {
    // For Q8_0, Q4_0, IQ4_NL: upload quantized blocks, not dequantized FP32
    auto loader = ModelLoader();
    auto weights = loader.loadTensor("test_q8_0_weights.bin");  // Test fixture
    
    ASSERT_TRUE(weights->ensureOnDevice(gpu_idx_));
    EXPECT_NE(weights->device_data(), nullptr);
    
    // Verify size matches host (quantized blocks, not expanded FP32)
    // Q8_0: 32 bytes per block (1 FP16 scale + 32 INT8 values) - wait, that's 34
    // Actually verify the implementation...
}
```

**Test Matrix**:

| Test | Tensor Type | Size | Validation |
|------|-------------|------|------------|
| `FP32Upload` | FP32Tensor | 1024×256 | Device pointer non-null |
| `FP32RoundTrip` | FP32Tensor | 512×128 | Bit-exact after upload→download |
| `BF16Upload` | BF16Tensor | 256×64 | Device pointer non-null |
| `BF16RoundTrip` | BF16Tensor | 256×64 | Tolerance: 1e-3 (BF16 precision) |
| `FP16Upload` | FP16Tensor | 256×64 | Device pointer non-null |
| `QuantizedUpload` | Q8_0Tensor | 64×128 | Block data matches |
| `MultipleUploads` | FP32Tensor | Various | Memory not leaked |
| `LargeTensor` | FP32Tensor | 4096×4096 | 64MB upload succeeds |

---

### Test 1.2: cuBLAS GEMM Wrapper

**File**: `tests/v2/unit/Test__CUBLASGemm.cpp`

Tests `CuBLASGemmKernel` correctness and performance.

```cpp
class Test__CUBLASGemm : public CUDATestBase {
protected:
    void SetUp() override {
        CUDATestBase::SetUp();
        kernel_ = std::make_unique<cuda::CuBLASGemmKernel>(gpu_idx_);
    }
    
    // CPU reference GEMM for comparison
    void cpuGemm(const float* A, const float* B, float* C, 
                 int M, int N, int K, bool transA, bool transB) {
        // Simple triple-loop reference
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (int p = 0; p < K; ++p) {
                    float a = transA ? A[p * M + i] : A[i * K + p];
                    float b = transB ? B[j * K + p] : B[p * N + j];
                    sum += a * b;
                }
                C[i * N + j] = sum;
            }
        }
    }
    
    std::unique_ptr<cuda::CuBLASGemmKernel> kernel_;
};

TEST_F(Test__CUBLASGemm, SmallMatrixCorrectness) {
    const int M = 64, N = 32, K = 128;
    
    auto A = generateRandomFP32(M * K);
    auto B = generateRandomFP32(K * N);
    std::vector<float> C_cuda(M * N, 0.0f);
    std::vector<float> C_cpu(M * N, 0.0f);
    
    // Allocate and upload
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));
    cudaMemcpy(d_A, A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);
    
    // CUDA GEMM
    kernel_->multiply(d_A, d_B, d_C, M, N, K, false, false);
    cudaMemcpy(C_cuda.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // CPU reference
    cpuGemm(A.data(), B.data(), C_cpu.data(), M, N, K, false, false);
    
    // Compare
    EXPECT_TRUE(compareArrays(C_cuda.data(), C_cpu.data(), M * N,
                              FP32_ABS_TOL, FP32_REL_TOL));
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

TEST_F(Test__CUBLASGemm, TransposeVariants) {
    // Test all 4 transpose combinations: NN, NT, TN, TT
    const int M = 32, N = 48, K = 64;
    // ... 
}

TEST_F(Test__CUBLASGemm, LargeMatrixPerformance) {
    // Benchmark cuBLAS vs threshold
    const int M = 4096, N = 4096, K = 4096;
    // Should achieve >50% of cuBLAS peak TFLOPS
}
```

**Test Matrix**:

| Test | Dims (M×N×K) | Transpose | Tolerance | Notes |
|------|--------------|-----------|-----------|-------|
| `SmallCorrectness` | 64×32×128 | NN | 1e-5 | Basic sanity |
| `SquareMatrix` | 256×256×256 | NN | 1e-5 | Typical tile sizes |
| `TallSkinny` | 4096×64×256 | NN | 1e-5 | Prefill workload |
| `SingleRow` | 1×4096×896 | NN | 1e-5 | Decode workload |
| `TransposeA` | 128×128×128 | TN | 1e-5 | |
| `TransposeB` | 128×128×128 | NT | 1e-5 | |
| `TransposeBoth` | 128×128×128 | TT | 1e-5 | |
| `LargePerf` | 4096×4096×4096 | NN | - | >10 TFLOPS |

---

### Test 1.3: KernelFactory CUDA Dispatch

**File**: `tests/v2/unit/Test__CUDAKernelFactoryDispatch.cpp`

Tests that `KernelFactory::createGemm()` returns correct kernel types.

```cpp
TEST_F(Test__CUDAKernelFactoryDispatch, FP32ReturnsCorrectType) {
    auto tensor = std::make_unique<FP32Tensor>(128, 256);
    
    // Request CUDA kernel
    auto kernel = KernelFactory::createGemm(tensor.get(), DeviceType::CUDA);
    
    ASSERT_NE(kernel, nullptr);
    
    // Verify it's the CUDA implementation (not CPU fallback)
    auto* cuda_kernel = dynamic_cast<cuda::CuBLASGemmKernel*>(kernel.get());
    EXPECT_NE(cuda_kernel, nullptr) << "Expected CuBLASGemmKernel, got different type";
}

TEST_F(Test__CUDAKernelFactoryDispatch, Q8_0ReturnsQuantizedKernel) {
    // Load or create Q8_0 tensor
    auto tensor = createTestQ8_0Tensor(256, 128);
    
    auto kernel = KernelFactory::createGemm(tensor.get(), DeviceType::CUDA);
    
    ASSERT_NE(kernel, nullptr);
    
    // Should be INT8 Tensor Core kernel, not cuBLAS
    auto* quant_kernel = dynamic_cast<cuda::CudaQuantizedGemmKernel*>(kernel.get());
    EXPECT_NE(quant_kernel, nullptr);
}

TEST_F(Test__CUDAKernelFactoryDispatch, CPUFallbackWhenNoGPU) {
    // This test runs even without GPU
    // Force CPU device type
    auto tensor = std::make_unique<FP32Tensor>(64, 64);
    
    auto kernel = KernelFactory::createGemm(tensor.get(), DeviceType::CPU);
    
    ASSERT_NE(kernel, nullptr);
    // Should NOT be CUDA kernel
    auto* cuda_kernel = dynamic_cast<cuda::CuBLASGemmKernel*>(kernel.get());
    EXPECT_EQ(cuda_kernel, nullptr);
}

TEST_F(Test__CUDAKernelFactoryDispatch, CacheReusesKernels) {
    auto tensor = std::make_unique<FP32Tensor>(128, 128);
    
    auto kernel1 = KernelFactory::getOrCreateGemm(tensor.get(), DeviceType::CUDA);
    auto kernel2 = KernelFactory::getOrCreateGemm(tensor.get(), DeviceType::CUDA);
    
    // Should return same cached kernel
    EXPECT_EQ(kernel1, kernel2);
}
```

---

### Test 1.4: End-to-End Integration

**File**: `tests/v2/integration/Test__CUDABasicPipeline.cpp`

Tests complete workflow: tensor creation → upload → GEMM → download → verify.

```cpp
TEST_F(Test__CUDABasicPipeline, SingleGEMMWorkflow) {
    // 1. Create tensors on host
    auto A = TensorFactory::createFP32({128, 256});
    auto B = TensorFactory::createFP32({256, 64});
    auto C = TensorFactory::createFP32({128, 64});
    
    // Fill with test data
    fillRandom(A.get());
    fillRandom(B.get());
    
    // 2. Upload to GPU
    ASSERT_TRUE(A->ensureOnDevice(gpu_idx_));
    ASSERT_TRUE(B->ensureOnDevice(gpu_idx_));
    ASSERT_TRUE(C->ensureOnDevice(gpu_idx_));  // Allocate output on GPU
    
    // 3. Get CUDA kernel via factory
    auto kernel = KernelFactory::createGemm(A.get(), DeviceType::CUDA);
    ASSERT_NE(kernel, nullptr);
    
    // 4. Execute GEMM on GPU
    kernel->multiply(A->device_data<float>(), 
                     B->device_data<float>(),
                     C->mutable_device_data<float>(),
                     128, 64, 256, false, false);
    
    // 5. Download result
    C->invalidate_host();  // Mark host copy stale
    ASSERT_TRUE(C->ensureOnHost());
    
    // 6. Verify against CPU reference
    auto C_ref = computeCPUReference(A.get(), B.get());
    EXPECT_TRUE(compareArrays(C->data(), C_ref.data(), 128 * 64,
                              FP32_ABS_TOL, FP32_REL_TOL));
}

TEST_F(Test__CUDABasicPipeline, QuantizedGEMMWorkflow) {
    // Similar but with Q8_0 weights
    // Verifies INT8 Tensor Core path
}
```

---

### CI Integration

Add to `.github/workflows/test.yml`:

```yaml
cuda-tests:
  runs-on: [self-hosted, gpu]  # GPU runner required
  if: github.event_name == 'push' || contains(github.event.pull_request.labels.*.name, 'cuda')
  steps:
    - uses: actions/checkout@v4
    
    - name: Configure CUDA build
      run: |
        cmake -B build_v2_cuda -S src/v2 \
          -DCMAKE_BUILD_TYPE=Debug \
          -DHAVE_CUDA=ON
    
    - name: Build
      run: cmake --build build_v2_cuda --parallel
    
    - name: Run CUDA tests
      run: |
        ctest --test-dir build_v2_cuda \
          -R "^V2_Unit_CUDA|^V2_Integration_CUDA" \
          --output-on-failure
```

For CPU-only CI, CUDA tests auto-skip via `GTEST_SKIP()`.

---

### Test Execution Commands

```bash
# Run all Phase 1 CUDA tests (requires GPU)
ctest --test-dir build_v2 -R "CUDA" --output-on-failure

# Run only tensor transfer tests
ctest --test-dir build_v2 -R "CUDATensorTransfer" -V

# Run with verbose output for debugging
ctest --test-dir build_v2 -R "CUBLASGemm" -V --output-on-failure

# Run parity tests (CPU vs CUDA comparison)
ctest --test-dir build_v2 -R "CUDAVsCPU" --output-on-failure
```

---

### Phase 1 Test Acceptance Criteria

| Component | Tests Pass | Parity Tolerance | Notes |
|-----------|------------|------------------|-------|
| FP32 Tensor Upload | ✅ | Bit-exact | No conversion |
| FP32 Tensor Round-trip | ✅ | Bit-exact | No conversion |
| BF16 Tensor Upload | ✅ | 1e-3 | BF16 precision |
| cuBLAS FP32 GEMM | ✅ | 1e-5 | vs CPU reference |
| cuBLAS FP16 GEMM | ✅ | 1e-3 | vs FP32 CPU |
| KernelFactory dispatch | ✅ | N/A | Type checking |
| End-to-end workflow | ✅ | 1e-5 | Full pipeline |

---

## Phase 1 Implementation Checklist

### Week 1: Core Infrastructure

- [ ] **1.1 Tensor GPU Transfer**
  - [ ] Implement `FP32Tensor::sync_to_device()` and `sync_from_device()`
  - [ ] Implement `BF16Tensor::sync_to_device()` and `sync_from_device()`
  - [ ] Implement `FP16Tensor::sync_to_device()` and `sync_from_device()`
  - [ ] Test: Upload FP32 tensor, verify data matches
  - [ ] Test: Round-trip upload → download, verify no data loss

- [ ] **1.2 cuBLAS GEMM Wrapper**
  - [ ] Create `CuBLASGemmKernel.cuh` header
  - [ ] Create `CuBLASGemmKernel.cu` implementation
  - [ ] Wire cuBLAS handle lifecycle (create in constructor, destroy in destructor)
  - [ ] Implement FP32 `cublasSgemm()` path
  - [ ] Implement FP16 `cublasHgemm()` path with Tensor Core math mode
  - [ ] Test: GEMM correctness vs CPU reference

### Week 2: Integration

- [ ] **1.3 KernelFactory Integration**
  - [ ] Add CUDA dispatch to `createGemm(const FP32Tensor*, DeviceType)`
  - [ ] Add CUDA dispatch to `createGemm(const FP16Tensor*, DeviceType)`
  - [ ] Add CUDA dispatch to `createGemm(const BF16Tensor*, DeviceType)`
  - [ ] Update `getOrCreateGemm()` cache to handle device type
  - [ ] Test: `KernelFactory::createGemm(tensor, DeviceType::CUDA)` returns CuBLAS kernel

- [ ] **1.4 End-to-End Test**
  - [ ] Create `Test__CUDAGemmBasic.cpp` integration test
  - [ ] Test: Create FP32 tensors, upload, GEMM, download, verify
  - [ ] Test: Benchmark GEMM throughput vs CPU
  - [ ] Add to CI (skip if no GPU available)

### Acceptance Criteria

✅ Phase 1 is complete when:
1. `FP32Tensor::ensureOnDevice(gpu_idx)` successfully uploads data
2. `KernelFactory::createGemm(fp32_tensor, DeviceType::CUDA)` returns working kernel
3. GPU GEMM produces correct results (MSE < 1e-5 vs CPU)
4. All tests pass on GPU-equipped machine
5. Tests gracefully skip on CPU-only machines

---

## Phase 4: CUDA Flash Attention Kernel

**Priority**: P1 (High)  
**Complexity**: High  
**Target**: After Phase 1-3 (GEMM + Ops kernels)

This section details implementing a CUDA attention kernel using **Flash Attention 2** for prefill and **Flash Decoding** for decode. These algorithms provide O(1) memory complexity in sequence length and significantly better performance than naive attention.

### 4.1 Background: Why Flash Attention?

#### Standard Attention Algorithm (Naive)

```python
# Naive attention - O(n²) memory
scores = Q @ K.T / sqrt(d)      # [seq_len, kv_len] - materialized!
scores = softmax(scores, dim=-1)
output = scores @ V              # [seq_len, head_dim]
```

**Problems**:
1. **Memory**: Must materialize `[seq_len, kv_len]` score matrix → O(n²) HBM
2. **Bandwidth**: Score matrix reads/writes to HBM dominate runtime
3. **Sequence length limit**: 8K sequence @ FP16 = 128MB per head per layer

#### Flash Attention 2 Algorithm (Tiled + Online Softmax)

```
Key Innovation: Never materialize the full score matrix.
Instead, compute attention in tiles using online softmax correction.

For each tile of Q (Br rows):
  For each tile of K/V (Bc columns):
    1. Load Q tile from HBM → SRAM (Br × d)
    2. Load K/V tile from HBM → SRAM (Bc × d)
    3. Compute tile scores: S_tile = Q_tile @ K_tile.T / sqrt(d)
    4. Update online softmax: max_new = max(max_old, rowmax(S_tile))
    5. Rescale previous output: O = O * exp(max_old - max_new)
    6. Add contribution: O += softmax(S_tile) @ V_tile
    7. Update normalization sum
```

**Benefits**:
- **Memory**: O(n) - only tile buffers in SRAM, no full score matrix
- **Speed**: 2-4x faster by reducing HBM traffic
- **Sequence length**: Unlimited (stream through tiles)

#### Flash Decoding Algorithm (Single Query Token)

For decode (seq_len=1), Flash Attention 2 is suboptimal because:
- Single query doesn't saturate GPU parallelism
- Better to parallelize across KV length

**Flash Decoding** splits the KV cache across thread blocks:
```
For each block b ∈ [0, num_splits):
  kv_start = b * (kv_len / num_splits)
  kv_end = (b+1) * (kv_len / num_splits)
  
  # Compute partial attention over this KV range
  partial_out[b], partial_lse[b] = flash_attn(Q, K[kv_start:kv_end], V[kv_start:kv_end])

# Final reduction: combine partial outputs with LSE correction
output = reduce_with_lse_correction(partial_out, partial_lse)
```

**Benefits for decode**:
- Full GPU utilization even with seq_len=1
- Parallelizes over KV cache length instead of batch
- Critical for single-sequence inference

---

### 4.2 Interface Design

The CUDA attention kernel will implement `ITensorAttention` to match the CPU kernel interface:

```cpp
// File: src/v2/kernels/cuda/attention/CUDAFlashAttentionKernel.h

#pragma once
#include "../../../tensors/TensorKernels.h"
#include "../../../tensors/Tensors.h"

namespace llaminar2::cuda {

/**
 * @brief CUDA Flash Attention kernel
 *
 * Implements Flash Attention 2 for prefill (seq_len > 1) and
 * Flash Decoding for decode (seq_len = 1).
 *
 * Automatically selects algorithm based on seq_len:
 * - seq_len > 1: Flash Attention 2 (tile over Q and K/V)
 * - seq_len = 1: Flash Decoding (split-K parallelism)
 *
 * Supports:
 * - FP32, FP16, BF16 precision
 * - Grouped Query Attention (GQA) with n_heads != n_kv_heads
 * - Causal masking
 * - Sliding window attention
 */
template <ActivationPrecision Precision>
class CUDAFlashAttentionKernelT : public ITensorAttention {
public:
    using ElementType = typename detail::PrecisionElement<Precision>::Type;

    explicit CUDAFlashAttentionKernelT(int device_idx = 0);
    ~CUDAFlashAttentionKernelT() override;

    // ITensorAttention interface
    bool supports_device(int device_idx) const override {
        return device_idx >= 0;  // GPU only (device_idx >= 0)
    }

    bool compute(
        const float* Q, const float* K, const float* V, float* output,
        int seq_len, int n_heads, int n_kv_heads, int head_dim,
        bool causal = false,
        int window_size = -1,
        TensorBase* workspace_scores = nullptr,
        TensorBase* workspace_buffer = nullptr,
        TensorBase* workspace_context = nullptr,
        TensorBase* workspace_mask = nullptr,
        bool use_bf16 = false,
        const MPIContext* mpi_ctx = nullptr,
        int device_idx = -1) override;

    bool compute_batch(
        const float* Q, const float* K, const float* V, float* output,
        int batch_size, int seq_len, int n_heads, int n_kv_heads, int head_dim,
        bool causal = false,
        int window_size = -1,
        TensorBase* workspace_scores = nullptr,
        TensorBase* workspace_buffer = nullptr,
        TensorBase* workspace_context = nullptr,
        TensorBase* workspace_mask = nullptr,
        bool use_bf16 = false,
        const MPIContext* mpi_ctx = nullptr,
        int device_idx = -1) override;

    /**
     * @brief Compute attention with KV cache (decode mode)
     *
     * Optimized for single-token decode with large KV cache.
     * Uses Flash Decoding algorithm with split-K parallelism.
     *
     * @param Q Query [1, n_heads, head_dim]
     * @param K_cache Full K cache [kv_len, n_kv_heads, head_dim]
     * @param V_cache Full V cache [kv_len, n_kv_heads, head_dim]
     * @param output Output [1, n_heads, head_dim]
     * @param kv_len Current KV cache length
     * @param n_heads Number of query heads
     * @param n_kv_heads Number of KV heads
     * @param head_dim Dimension per head
     * @param position Current position (for causal masking)
     */
    bool compute_decode(
        const float* Q, const float* K_cache, const float* V_cache, float* output,
        int kv_len, int n_heads, int n_kv_heads, int head_dim,
        int position);

private:
    int device_idx_;
    cudaStream_t stream_;
    
    // Workspace for Flash Decoding partial outputs/LSE
    void* partial_output_buf_ = nullptr;
    void* partial_lse_buf_ = nullptr;
    size_t workspace_size_ = 0;

    // Algorithm dispatch
    bool flash_attention_2_prefill(
        const ElementType* Q, const ElementType* K, const ElementType* V,
        ElementType* output,
        int batch_size, int seq_len, int n_heads, int n_kv_heads, int head_dim,
        bool causal, int window_size);

    bool flash_decoding(
        const ElementType* Q, const ElementType* K_cache, const ElementType* V_cache,
        ElementType* output,
        int kv_len, int n_heads, int n_kv_heads, int head_dim,
        int position);
};

// Type aliases
using CUDAFlashAttentionKernelFP32 = CUDAFlashAttentionKernelT<ActivationPrecision::FP32>;
using CUDAFlashAttentionKernelFP16 = CUDAFlashAttentionKernelT<ActivationPrecision::FP16>;
using CUDAFlashAttentionKernelBF16 = CUDAFlashAttentionKernelT<ActivationPrecision::BF16>;

} // namespace llaminar2::cuda
```

---

### 4.3 CUDA Kernel Design

#### 4.3.1 Flash Attention 2 Prefill Kernel

```cuda
// File: src/v2/kernels/cuda/attention/FlashAttention2.cu

/**
 * Flash Attention 2 Forward Kernel
 *
 * Template parameters:
 *   Br: Tile size for Q (rows) - typically 64 or 128
 *   Bc: Tile size for K/V (cols) - typically 64 or 128
 *   d:  Head dimension (compile-time for register allocation)
 *
 * Thread block: (Br, d/4) for vectorized loads
 * Grid: (n_heads, ceil(seq_len/Br), batch_size)
 *
 * Shared memory layout:
 *   Q_tile: [Br, d]
 *   K_tile: [Bc, d]
 *   V_tile: [Bc, d]
 *   S_tile: [Br, Bc] (attention scores)
 */
template <int Br, int Bc, int d, typename T>
__global__ void flash_attention_2_kernel(
    const T* __restrict__ Q,      // [batch, seq_len, n_heads, d]
    const T* __restrict__ K,      // [batch, kv_len, n_kv_heads, d]
    const T* __restrict__ V,      // [batch, kv_len, n_kv_heads, d]
    T* __restrict__ O,            // [batch, seq_len, n_heads, d]
    float* __restrict__ L,        // [batch, n_heads, seq_len] - logsumexp for backward
    int seq_len,
    int kv_len,
    int n_heads,
    int n_kv_heads,
    float softmax_scale,
    bool causal,
    int window_size)
{
    // Determine which head and batch this block processes
    const int head_idx = blockIdx.x;
    const int tile_row = blockIdx.y;
    const int batch_idx = blockIdx.z;
    
    // GQA: map query head to KV head
    const int kv_head_idx = head_idx / (n_heads / n_kv_heads);
    
    // Shared memory allocation
    extern __shared__ char smem[];
    T* Q_tile = reinterpret_cast<T*>(smem);
    T* K_tile = Q_tile + Br * d;
    T* V_tile = K_tile + Bc * d;
    float* S_tile = reinterpret_cast<float*>(V_tile + Bc * d);
    
    // Per-thread accumulator and online softmax state
    float O_acc[d / 4][4];  // Register tiling
    float m_i = -INFINITY;  // Running max
    float l_i = 0.0f;       // Running sum
    
    // Initialize accumulators to zero
    #pragma unroll
    for (int i = 0; i < d / 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            O_acc[i][j] = 0.0f;
        }
    }
    
    // Load Q tile (this block's rows)
    const int q_row_start = tile_row * Br;
    load_tile_Q(Q, Q_tile, batch_idx, q_row_start, head_idx, seq_len, n_heads, d);
    __syncthreads();
    
    // Iterate over K/V tiles
    const int num_kv_tiles = (kv_len + Bc - 1) / Bc;
    
    for (int kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++) {
        const int kv_start = kv_tile * Bc;
        const int kv_end = min(kv_start + Bc, kv_len);
        const int tile_kv_len = kv_end - kv_start;
        
        // Causal masking: skip tiles that are entirely masked
        if (causal && kv_start > q_row_start + Br) {
            continue;
        }
        
        // Sliding window: skip tiles outside window
        if (window_size > 0 && kv_start > q_row_start + window_size) {
            continue;
        }
        
        // Load K/V tiles
        load_tile_KV(K, K_tile, batch_idx, kv_start, kv_head_idx, kv_len, n_kv_heads, d);
        load_tile_KV(V, V_tile, batch_idx, kv_start, kv_head_idx, kv_len, n_kv_heads, d);
        __syncthreads();
        
        // Compute S = Q @ K^T * scale
        compute_attention_scores<Br, Bc, d>(Q_tile, K_tile, S_tile, softmax_scale);
        __syncthreads();
        
        // Apply causal mask to S_tile
        if (causal) {
            apply_causal_mask<Br, Bc>(S_tile, q_row_start, kv_start);
            __syncthreads();
        }
        
        // Apply sliding window mask
        if (window_size > 0) {
            apply_window_mask<Br, Bc>(S_tile, q_row_start, kv_start, window_size);
            __syncthreads();
        }
        
        // Online softmax update
        float m_ij = -INFINITY;  // Max in this tile
        
        // Find row max
        #pragma unroll
        for (int j = 0; j < tile_kv_len; j++) {
            m_ij = fmaxf(m_ij, S_tile[threadIdx.x * Bc + j]);
        }
        
        // New global max
        float m_i_new = fmaxf(m_i, m_ij);
        
        // Rescale previous accumulator
        float scale_old = expf(m_i - m_i_new);
        #pragma unroll
        for (int i = 0; i < d / 4; i++) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                O_acc[i][j] *= scale_old;
            }
        }
        l_i *= scale_old;
        
        // Compute softmax of this tile and accumulate
        float l_ij = 0.0f;
        #pragma unroll
        for (int j = 0; j < tile_kv_len; j++) {
            float p_ij = expf(S_tile[threadIdx.x * Bc + j] - m_i_new);
            l_ij += p_ij;
            
            // Accumulate P @ V
            #pragma unroll
            for (int di = 0; di < d / 4; di++) {
                #pragma unroll
                for (int dj = 0; dj < 4; dj++) {
                    O_acc[di][dj] += p_ij * V_tile[j * d + di * 4 + dj];
                }
            }
        }
        
        l_i += l_ij;
        m_i = m_i_new;
        
        __syncthreads();
    }
    
    // Final normalization: O = O / l
    float inv_l = 1.0f / l_i;
    
    // Write output
    const int out_row = q_row_start + threadIdx.x;
    if (out_row < seq_len) {
        T* O_row = O + ((batch_idx * seq_len + out_row) * n_heads + head_idx) * d;
        #pragma unroll
        for (int i = 0; i < d / 4; i++) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                O_row[i * 4 + j] = static_cast<T>(O_acc[i][j] * inv_l);
            }
        }
        
        // Store logsumexp for potential backward pass
        L[(batch_idx * n_heads + head_idx) * seq_len + out_row] = m_i + logf(l_i);
    }
}
```

#### 4.3.2 Flash Decoding Kernel (Single Query)

```cuda
// File: src/v2/kernels/cuda/attention/FlashDecoding.cu

/**
 * Flash Decoding Kernel - Split-K parallel attention for decode
 *
 * For seq_len=1, parallelizes over KV cache instead of sequence.
 * Each thread block computes attention over a KV range, then reduce.
 *
 * Grid: (n_heads, num_splits, batch_size)
 * Thread block: (128,) or (256,) depending on head_dim
 */
template <int d, typename T>
__global__ void flash_decoding_kernel(
    const T* __restrict__ Q,         // [batch, 1, n_heads, d]
    const T* __restrict__ K_cache,   // [batch, kv_len, n_kv_heads, d]
    const T* __restrict__ V_cache,   // [batch, kv_len, n_kv_heads, d]
    T* __restrict__ O_partial,       // [batch, n_heads, num_splits, d]
    float* __restrict__ L_partial,   // [batch, n_heads, num_splits] (logsumexp)
    int kv_len,
    int n_heads,
    int n_kv_heads,
    int num_splits,
    float softmax_scale)
{
    const int head_idx = blockIdx.x;
    const int split_idx = blockIdx.y;
    const int batch_idx = blockIdx.z;
    
    const int kv_head_idx = head_idx / (n_heads / n_kv_heads);
    
    // Determine this split's KV range
    const int split_size = (kv_len + num_splits - 1) / num_splits;
    const int kv_start = split_idx * split_size;
    const int kv_end = min(kv_start + split_size, kv_len);
    
    if (kv_start >= kv_len) return;
    
    // Load Q into shared memory (all threads load different parts)
    __shared__ float Q_shared[d];
    const T* Q_ptr = Q + (batch_idx * n_heads + head_idx) * d;
    
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        Q_shared[i] = static_cast<float>(Q_ptr[i]);
    }
    __syncthreads();
    
    // Each thread processes one or more KV positions
    float O_local[d] = {0};
    float m_local = -INFINITY;
    float l_local = 0.0f;
    
    for (int kv_pos = kv_start + threadIdx.x; kv_pos < kv_end; kv_pos += blockDim.x) {
        // Compute Q @ K^T for this position
        const T* K_ptr = K_cache + ((batch_idx * kv_len + kv_pos) * n_kv_heads + kv_head_idx) * d;
        float score = 0.0f;
        
        #pragma unroll
        for (int i = 0; i < d; i++) {
            score += Q_shared[i] * static_cast<float>(K_ptr[i]);
        }
        score *= softmax_scale;
        
        // Online softmax update
        float m_new = fmaxf(m_local, score);
        float scale = expf(m_local - m_new);
        
        l_local = l_local * scale + expf(score - m_new);
        
        #pragma unroll
        for (int i = 0; i < d; i++) {
            O_local[i] *= scale;
        }
        
        // Accumulate V
        const T* V_ptr = V_cache + ((batch_idx * kv_len + kv_pos) * n_kv_heads + kv_head_idx) * d;
        float p = expf(score - m_new);
        
        #pragma unroll
        for (int i = 0; i < d; i++) {
            O_local[i] += p * static_cast<float>(V_ptr[i]);
        }
        
        m_local = m_new;
    }
    
    // Warp reduction for partial results
    // ... (warp shuffle reduction to combine thread results)
    
    // Block reduction
    // ... (shared memory reduction)
    
    // Thread 0 writes partial output
    if (threadIdx.x == 0) {
        T* O_out = O_partial + ((batch_idx * n_heads + head_idx) * num_splits + split_idx) * d;
        for (int i = 0; i < d; i++) {
            O_out[i] = static_cast<T>(O_local[i]);
        }
        L_partial[(batch_idx * n_heads + head_idx) * num_splits + split_idx] = m_local + logf(l_local);
    }
}

/**
 * Flash Decoding Reduction Kernel
 * Combines partial outputs from all splits using LSE correction.
 */
template <int d, typename T>
__global__ void flash_decoding_reduce_kernel(
    const T* __restrict__ O_partial,   // [batch, n_heads, num_splits, d]
    const float* __restrict__ L_partial, // [batch, n_heads, num_splits]
    T* __restrict__ O,                   // [batch, 1, n_heads, d]
    int n_heads,
    int num_splits)
{
    const int head_idx = blockIdx.x;
    const int batch_idx = blockIdx.y;
    
    // Find global max LSE
    float lse_max = -INFINITY;
    for (int s = 0; s < num_splits; s++) {
        float lse = L_partial[(batch_idx * n_heads + head_idx) * num_splits + s];
        lse_max = fmaxf(lse_max, lse);
    }
    
    // Combine with LSE correction
    float O_combined[d] = {0};
    float l_combined = 0.0f;
    
    for (int s = 0; s < num_splits; s++) {
        float lse = L_partial[(batch_idx * n_heads + head_idx) * num_splits + s];
        float scale = expf(lse - lse_max);
        l_combined += scale;
        
        const T* O_s = O_partial + ((batch_idx * n_heads + head_idx) * num_splits + s) * d;
        for (int i = threadIdx.x; i < d; i += blockDim.x) {
            O_combined[i] += scale * static_cast<float>(O_s[i]);
        }
    }
    
    // Normalize and write output
    float inv_l = 1.0f / l_combined;
    T* O_out = O + (batch_idx * n_heads + head_idx) * d;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        O_out[i] = static_cast<T>(O_combined[i] * inv_l);
    }
}
```

---

### 4.4 Tile Size and Performance Tuning

#### Recommended Tile Sizes by GPU Architecture

| GPU Architecture | SM | Shared Memory | Br × Bc | Head Dim | Notes |
|-----------------|-----|---------------|---------|----------|-------|
| Ampere (A100) | 8.0 | 164 KB | 128×64 | 64, 128 | Maximum occupancy |
| Ampere (RTX 3090) | 8.6 | 100 KB | 64×64 | 64, 128 | Consumer GPU |
| Ada (RTX 4090) | 8.9 | 100 KB | 64×64 | 64, 128 | Higher clocks |
| Hopper (H100) | 9.0 | 228 KB | 128×128 | 64, 128 | TMA + wgmma |

#### Shared Memory Requirements

For Flash Attention 2 with Br=64, Bc=64, d=64, FP16:
```
Q_tile:  Br × d × sizeof(T) = 64 × 64 × 2 = 8 KB
K_tile:  Bc × d × sizeof(T) = 64 × 64 × 2 = 8 KB
V_tile:  Bc × d × sizeof(T) = 64 × 64 × 2 = 8 KB
S_tile:  Br × Bc × sizeof(float) = 64 × 64 × 4 = 16 KB
Total: 40 KB per block
```

For head_dim=128:
```
Q_tile:  64 × 128 × 2 = 16 KB
K_tile:  64 × 128 × 2 = 16 KB
V_tile:  64 × 128 × 2 = 16 KB
S_tile:  64 × 64 × 4 = 16 KB
Total: 64 KB per block
```

---

### 4.5 Implementation Plan

#### Files to Create

```
src/v2/kernels/cuda/attention/
├── CUDAFlashAttentionKernelT.h      # Header with template declarations
├── CUDAFlashAttentionKernelT.cpp    # ITensorAttention method implementations
├── FlashAttention2.cu               # FA2 prefill CUDA kernel
├── FlashDecoding.cu                 # Flash Decoding CUDA kernel
├── FlashAttentionCommon.cuh         # Shared device functions
└── CUDAAttentionOpsKernels.cu       # extern "C" wrapper functions

tests/v2/integration/
├── Test__CUDAFlashAttentionParity.cpp  # CPU vs CUDA parity tests
```

#### 4.5.1 Phase 4.1: Flash Attention 2 Prefill (Week 1-2)

- [ ] **4.1.1 Core Kernel**
  - [ ] Create `FlashAttention2.cu` with basic kernel
  - [ ] Implement tiled Q@K^T score computation
  - [ ] Implement online softmax with max tracking
  - [ ] Implement V accumulation with rescaling
  - [ ] Test: Single head, single batch, no masking

- [ ] **4.1.2 Masking Support**
  - [ ] Implement causal masking (lower triangular)
  - [ ] Implement sliding window masking
  - [ ] Test: Causal mask correctness vs CPU

- [ ] **4.1.3 GQA Support**
  - [ ] Add KV head index mapping (head_idx / group_size)
  - [ ] Test: GQA with n_heads=32, n_kv_heads=8

- [ ] **4.1.4 Multi-Precision**
  - [ ] Implement FP16 path with Tensor Core accumulation
  - [ ] Implement BF16 path
  - [ ] Implement FP32 path (fallback)
  - [ ] Test: Precision parity vs CPU

#### 4.5.2 Phase 4.2: Flash Decoding (Week 3)

- [ ] **4.2.1 Split-K Kernel**
  - [ ] Create `FlashDecoding.cu` with split-K kernel
  - [ ] Implement per-thread online softmax
  - [ ] Implement warp reduction for partial outputs
  - [ ] Implement block reduction

- [ ] **4.2.2 LSE Reduction**
  - [ ] Create reduction kernel for combining splits
  - [ ] Implement LSE correction for accurate output
  - [ ] Test: Single-query decode correctness

- [ ] **4.2.3 Workspace Management**
  - [ ] Allocate partial output buffer in kernel constructor
  - [ ] Size buffer for max expected num_splits
  - [ ] Reuse buffer across calls

#### 4.5.3 Phase 4.3: Integration (Week 4)

- [ ] **4.3.1 ITensorAttention Implementation**
  - [ ] Create `CUDAFlashAttentionKernelT.cpp`
  - [ ] Implement `compute()` dispatching to FA2
  - [ ] Implement `compute_batch()` with batch dimension
  - [ ] Implement `compute_decode()` using Flash Decoding
  - [ ] Auto-detect prefill vs decode based on seq_len

- [ ] **4.3.2 KernelFactory Integration**
  - [ ] Add `createAttention(DeviceType::CUDA)` dispatch
  - [ ] Add caching for CUDA attention kernels
  - [ ] Wire into `FusedAttentionWoStage` device routing

- [ ] **4.3.3 Parity Tests**
  - [ ] Create `Test__CUDAFlashAttentionParity.cpp`
  - [ ] Test: Prefill parity (cosine > 0.9999)
  - [ ] Test: Decode parity (cosine > 0.9999)
  - [ ] Test: Large sequence (8K tokens)
  - [ ] Test: GQA configurations

---

### 4.6 Test Plan

#### Parity Test Structure

```cpp
// tests/v2/integration/Test__CUDAFlashAttentionParity.cpp

class Test__CUDAFlashAttentionParity : public ::testing::Test {
protected:
    void SetUp() override {
        auto& dm = DeviceManager::instance();
        gpu_idx_ = dm.firstGPU();
        if (gpu_idx_ < 0) {
            GTEST_SKIP() << "No GPU available";
        }
    }
    
    int gpu_idx_ = -1;
};

TEST_F(Test__CUDAFlashAttentionParity, Prefill_FP32_Small) {
    const int seq_len = 128;
    const int n_heads = 12;
    const int n_kv_heads = 12;
    const int head_dim = 64;
    
    // Create random input tensors
    auto Q = createRandomFP32({seq_len, n_heads, head_dim});
    auto K = createRandomFP32({seq_len, n_kv_heads, head_dim});
    auto V = createRandomFP32({seq_len, n_kv_heads, head_dim});
    auto output_cpu = createFP32({seq_len, n_heads, head_dim});
    auto output_cuda = createFP32({seq_len, n_heads, head_dim});
    
    // CPU reference
    CPUAttentionKernelT<ActivationPrecision::FP32> cpu_kernel;
    cpu_kernel.compute(
        Q->data(), K->data(), V->data(), output_cpu->mutable_data(),
        seq_len, n_heads, n_kv_heads, head_dim, true);
    
    // CUDA Flash Attention
    CUDAFlashAttentionKernelFP32 cuda_kernel(gpu_idx_);
    cuda_kernel.compute(
        Q->data(), K->data(), V->data(), output_cuda->mutable_data(),
        seq_len, n_heads, n_kv_heads, head_dim, true,
        -1, nullptr, nullptr, nullptr, nullptr, false, nullptr, gpu_idx_);
    
    // Compare
    float cosine = computeCosineSimilarity(
        output_cpu->data(), output_cuda->data(), seq_len * n_heads * head_dim);
    
    std::cout << "Prefill FP32 Small: cosine=" << cosine << std::endl;
    EXPECT_GE(cosine, 0.9999f);
}

TEST_F(Test__CUDAFlashAttentionParity, Decode_FP32_LargeKV) {
    const int kv_len = 4096;  // Large KV cache
    const int n_heads = 32;
    const int n_kv_heads = 8;  // GQA
    const int head_dim = 128;
    
    auto Q = createRandomFP32({1, n_heads, head_dim});
    auto K_cache = createRandomFP32({kv_len, n_kv_heads, head_dim});
    auto V_cache = createRandomFP32({kv_len, n_kv_heads, head_dim});
    auto output_cpu = createFP32({1, n_heads, head_dim});
    auto output_cuda = createFP32({1, n_heads, head_dim});
    
    // CPU decode (compute with seq_len=1, kv_len=4096)
    CPUAttentionKernelT<ActivationPrecision::FP32> cpu_kernel;
    cpu_kernel.compute_decode(
        Q->data(), K_cache->data(), V_cache->data(), output_cpu->mutable_data(),
        1, kv_len, n_heads, n_kv_heads, head_dim, true);
    
    // CUDA Flash Decoding
    CUDAFlashAttentionKernelFP32 cuda_kernel(gpu_idx_);
    cuda_kernel.compute_decode(
        Q->data(), K_cache->data(), V_cache->data(), output_cuda->mutable_data(),
        kv_len, n_heads, n_kv_heads, head_dim, kv_len - 1);
    
    float cosine = computeCosineSimilarity(
        output_cpu->data(), output_cuda->data(), n_heads * head_dim);
    
    std::cout << "Decode FP32 Large KV: cosine=" << cosine << std::endl;
    EXPECT_GE(cosine, 0.9999f);
}
```

#### Performance Benchmarks

```cpp
TEST_F(Test__CUDAFlashAttentionPerf, Prefill_Throughput) {
    const int seq_len = 2048;
    const int n_heads = 32;
    const int head_dim = 128;
    
    // Warm up
    for (int i = 0; i < 5; i++) {
        cuda_kernel.compute(...);
    }
    cudaDeviceSynchronize();
    
    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    const int iterations = 100;
    for (int i = 0; i < iterations; i++) {
        cuda_kernel.compute(...);
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    double tflops = (4.0 * seq_len * seq_len * n_heads * head_dim * iterations) / (ms * 1e9);
    
    std::cout << "Flash Attention 2 Prefill: " << tflops << " TFLOP/s" << std::endl;
    // Expected: 150+ TFLOP/s on RTX 3090
}
```

---

### 4.7 Acceptance Criteria

| Test | Requirement | Notes |
|------|-------------|-------|
| Prefill FP32 parity | cosine ≥ 0.9999 | vs CPU reference |
| Prefill FP16 parity | cosine ≥ 0.999 | Some precision loss expected |
| Prefill BF16 parity | cosine ≥ 0.999 | BF16 precision |
| Decode FP32 parity | cosine ≥ 0.9999 | Flash Decoding |
| Decode FP16 parity | cosine ≥ 0.999 | Flash Decoding |
| GQA correctness | cosine ≥ 0.9999 | n_heads=32, n_kv_heads=8 |
| Causal masking | exact | No leakage of future tokens |
| Sliding window | exact | Correct mask application |
| Large sequence (8K) | passes | Memory-efficient |
| Prefill throughput | ≥100 TFLOP/s | RTX 3090 baseline |
| Decode throughput | ≥50 TFLOP/s | kv_len=4096 |

---

### 4.8 Future Enhancements (Post-MVP)

1. **Quantized Q/K/V**: INT8 Flash Attention using Tensor Cores
2. **Paged Attention**: Support for vLLM-style paged KV cache
3. **Speculative Decoding**: Batch verification with variable-length queries
4. **Multi-GPU**: Tensor parallel attention with NCCL
5. **Backward Pass**: Training support with gradient computation
6. **TMA (Hopper)**: Use Tensor Memory Access for H100+
7. **WGMMA (Hopper)**: Use warp group matrix multiply for H100+
