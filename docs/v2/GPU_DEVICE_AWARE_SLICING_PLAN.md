# GPU Device-Aware Tensor Slicing Plan

**Author:** David Sanftenberg  
**Date:** December 3, 2025  
**Status:** Phase 5 Complete

## Executive Summary

This document outlines the design and implementation plan for GPU device-aware tensor slicing in Llaminar V2. The goal is to extend the existing MPI-aware and NUMA-aware tensor slicing infrastructure to support GPU placement with lazy transfers and quantized GPU execution.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Transfer Strategy** | **Lazy** | Transfer to GPU on first use, not at load time. Flexible for CPU/GPU hybrid execution. |
| **Quantized on GPU** | **Yes** | Keep weights quantized in GPU memory. Implement GPU dequant kernels later. Memory efficient. |
| **Slice Location** | **CPU** | Slice rows on CPU before GPU transfer. Minimal GPU memory footprint per rank. |
| **Implementation in TensorBase** | **Yes** | Single transfer implementation with per-tensor accessors. Avoids duplication across 20+ tensor types. |

## Current State (Completed)

### What's Working

1. **MPI-Aware Slicing** ✅
   - `WeightManager` uses `mpi_ctx_->rank()` and `world_size` to compute row bounds
   - Each rank loads only its slice: `[rows_per_rank * rank, rows_per_rank * (rank + 1))`

2. **NUMA-Aware Allocation** ✅
   - `ModelLoader::loadTensorRowSlice()` now uses `TensorFactory` when available
   - Factory binds allocations to local NUMA node based on MPI rank
   - Fix implemented: December 3, 2025

3. **Memory-Efficient Loading** ✅
   - `loadTensorRowSlice()` reads only slice bytes from GGUF file
   - No full tensor materialization required
   - Preserves quantized format (no dequantization)

4. **TensorSlice Wrapper** ✅
   - Wraps sliced tensor with `SliceMetadata`
   - Tracks original dimensions, slice bounds, MPI rank
   - `inner_is_presliced=true` indicates memory-efficient mode

5. **Lazy Transfer Infrastructure (Phase 1)** ✅ - Implemented December 3, 2025
   - `TensorBase::ensureOnDevice(int)` - lazy upload to GPU
   - `TensorBase::ensureOnHost()` - lazy download to CPU
   - `TensorBase::releaseDeviceMemory()` - free GPU buffer
   - `TensorBase::isOnGPU()` / `isOnCPU()` - query residency
   - `BackendManager` - global GPU backend accessor
   - Single implementation in `TensorBase::ensureOnDevice()` with per-tensor accessors

6. **Tensor Accessor Implementation (Phase 3)** ✅ - Implemented December 3, 2025
   - All 26 tensor types implement `raw_host_data_ptr()` and `byte_size()` accessors
   - FP32/FP16/BF16: Uses `AlignedVector<T>` with view offset handling
   - INT8/INT32: Simple `AlignedVector` accessors
   - Quantized: Uses `raw_data_.data()` for owned tensors
   - View handling: Returns correct pointer for views, 0 size for non-owned data

7. **KernelFactory for Device Dispatch (Phase 4)** ✅ - Implemented December 3, 2025
   - `KernelFactory::getDeviceType(device_idx)` - convert device index to DeviceType enum
   - `KernelFactory::createGemm(tensor, DeviceType)` - centralized kernel dispatch
   - All 22 tensor types refactored to use KernelFactory
   - Eliminates duplicate switch statements across tensor files
   - Unit tests: 37 passing tests covering all tensor types and device types

8. **Pipeline Device Orchestration (Phase 5)** ✅ - Implemented December 3, 2025
   - `Qwen2Pipeline::ensureAttentionWeightsOnDevice()` - lazy transfer Q/K/V/O weights
   - `Qwen2Pipeline::ensureFFNWeightsOnDevice()` - lazy transfer gate/up/down weights
   - Called at start of `attention_block()` and `ffn_block()` before kernels execute
   - No-op when target device is CPU or weights already on target
   - Integrates with `WeightPlacementMap` for per-layer device decisions
   - All 113 unit tests + 30 integration tests pass

## Data Flow Architecture

```
GGUF File
    │
    ▼ (CPU read, only slice bytes)
┌─────────────────────────────┐
│ loadTensorRowSlice()        │
│ - Read [row_start:row_end]  │
│ - NUMA-aware allocation     │
│ - Returns quantized slice   │
└─────────────────────────────┘
    │
    ▼ (stays on CPU initially)
┌─────────────────────────────┐
│ TensorSlice (CPU)           │
│ - Quantized data            │
│ - SliceMetadata             │
│ - device_idx_ = -1 (CPU)    │
└─────────────────────────────┘
    │
    ▼ (lazy: on first createGemm() with GPU target)
┌─────────────────────────────┐
│ ensureOnDevice(gpu_idx)     │
│ - Allocate GPU memory       │
│ - cudaMemcpyAsync()         │
│ - Keep CPU copy (optional)  │
│ - device_idx_ = gpu_idx     │
└─────────────────────────────┘
    │
    ▼ (GPU GEMM with on-the-fly dequant)
┌─────────────────────────────┐
│ GPU Quantized GEMM Kernel   │
│ - Dequant + matmul fused    │
│ - No full FP32 expansion    │
└─────────────────────────────┘
```

## Implementation Phases

### Phase 1: Lazy Transfer Infrastructure (Foundation)

**Goal:** Add device tracking and lazy transfer API to `TensorBase` with a **single implementation** that works for all tensor types.

**Design Rationale:** GPU transfers are byte-level operations - the backend doesn't care if it's FP32, BF16, or quantized blocks. The only tensor-specific parts are:
1. Getting the host data pointer
2. Calculating byte size

By implementing transfers in `TensorBase` with two small abstract accessors, we avoid duplicating transfer logic in every tensor class.

**Files to Modify:**
- `src/v2/tensors/Tensors.h` - Add device tracking members and abstract accessors
- `src/v2/tensors/TensorBase.cpp` - **Single implementation** of transfer logic
- `src/v2/backends/BackendManager.{h,cpp}` - Global GPU backend accessor

**API Design:**
```cpp
class TensorBase {
protected:
    // Device state (maintained by TensorBase transfer methods)
    void* device_data_ptr_ = nullptr;  // GPU buffer (nullptr = not on GPU)
    bool host_invalid_ = false;        // true if GPU has newer data
    int gpu_device_idx_ = -1;          // Which GPU device, -1 = none
    
    // ==== Abstract accessors (each tensor type implements) ====
    virtual void* raw_host_data_ptr() = 0;        // Pointer to host data
    virtual const void* raw_host_data_ptr() const = 0;
    virtual size_t byte_size() const = 0;         // Total bytes to transfer
    
public:
    // ==== Transfer API (single implementation in TensorBase) ====
    bool ensureOnDevice(int target_device);  // Non-virtual, implemented in TensorBase
    bool ensureOnHost();                      // Non-virtual, implemented in TensorBase
    bool releaseDeviceMemory();               // Non-virtual, implemented in TensorBase
    
    // Query current location
    bool isOnGPU() const { return device_data_ptr_ != nullptr; }
    bool isOnCPU() const { return !host_invalid_; }
};

// Example: FP32Tensor just implements accessors
class FP32Tensor : public TensorBase {
protected:
    void* raw_host_data_ptr() override { return host_data_.data(); }
    const void* raw_host_data_ptr() const override { return host_data_.data(); }
    size_t byte_size() const override { return element_count() * sizeof(float); }
};

// Example: IQ4_NLTensor just implements accessors
class IQ4_NLTensor : public TensorBase {
protected:
    void* raw_host_data_ptr() override { return blocks_.data(); }
    const void* raw_host_data_ptr() const override { return blocks_.data(); }
    size_t byte_size() const override { return blocks_.size() * sizeof(IQ4_NLBlock); }
};
```

**Benefits:**
- Single transfer implementation (~100 lines) vs duplicated in 20+ tensor types
- Each tensor class adds only 3 trivial one-liner methods
- Automatically supports all current and future tensor types
- Backend abstraction (`IBackend`) already exists

**Tests:**
- Unit test for device state transitions
- Test lazy transfer triggers correctly
- Test no-op when already on target device
- Round-trip test: CPU→GPU→CPU data integrity

---

### Phase 2: GPU Memory Management

**Goal:** Abstract GPU memory allocation across CUDA/ROCm backends.

**Files to Create:**
- `src/v2/backends/gpu/GPUAllocator.h`
- `src/v2/backends/gpu/GPUAllocator.cpp`
- `src/v2/backends/gpu/CUDAAllocator.cpp` (CUDA implementation)
- `src/v2/backends/gpu/ROCmAllocator.cpp` (ROCm implementation)

**API:**
```cpp
// Simple GPU allocator (CUDA/ROCm abstracted)
class GPUAllocator {
public:
    static void* allocate(size_t bytes, int device_idx);
    static void free(void* ptr, int device_idx);
    static void copyToDevice(void* dst, const void* src, size_t bytes, int device_idx);
    static void copyToHost(void* dst, const void* src, size_t bytes, int device_idx);
    
    // Async variants for pipelining
    static void copyToDeviceAsync(void* dst, const void* src, size_t bytes, 
                                   int device_idx, void* stream);
    static void synchronize(int device_idx);
    static void synchronizeStream(void* stream);
};
```

**Tests:**
- Allocate/free round-trip
- Copy to device and back, verify data integrity
- Multi-device allocation (if available)

---

### Phase 3: Quantized Tensor Accessor Implementation ✅ COMPLETE

**Goal:** Implement `raw_host_data_ptr()` and `byte_size()` for all quantized tensor types.

**Status:** All 26 tensor types now have accessor implementations.

**Files Modified:**
- `src/v2/tensors/Tensors.h` - Added protected accessor declarations to all tensor classes
- `src/v2/tensors/FP16Tensor.cpp` - Implemented accessors
- `src/v2/tensors/BF16Tensor.cpp` - Implemented accessors

**Implementation Notes:**
1. FP32Tensor, FP16Tensor, BF16Tensor: Use `AlignedVector<T>` with view offset handling
2. INT8Tensor, INT32Tensor: Simple `AlignedVector` accessors (no view support)
3. Quantized tensors (IQ4_NL, Q8_0, Q4_0, etc.): Use `raw_data_.data()` for owned tensors

**View Handling:**
- Views return correct pointer: `is_view_ ? (raw_data_ptr_ + view_byte_offset_) : raw_data_.data()`
- `byte_size()` returns `raw_data_.size()` - for views this is 0 since views don't own data
- GPU transfers should operate on parent tensors, not views

**Tensor Types Implemented:**
- [x] FP32Tensor, FP16Tensor, BF16Tensor (simple memcpy)
- [x] INT8Tensor, INT32Tensor
- [x] Q4_0Tensor, Q4_1Tensor
- [x] Q5_0Tensor, Q5_1Tensor
- [x] Q8_0Tensor, Q8_1Tensor
- [x] Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K
- [x] IQ4_NL, IQ4_XS
- [x] IQ3_S, IQ3_XXS
- [x] IQ2_S, IQ2_XS, IQ2_XXS
- [x] IQ1_S, IQ1_M

**Tests:** All 112 unit tests pass.

---

### Phase 4: Kernel Dispatch by Device

**Goal:** `createGemm()` returns appropriate kernel based on tensor location.

**Files to Modify:**
- `src/v2/tensors/Tensors.h` - Update `createGemm()` implementations

**Files to Create:**
- `src/v2/kernels/cuda/` - CUDA kernel directory
- `src/v2/kernels/cuda/CUDAQuantizedGemm.h`
- `src/v2/kernels/cuda/CUDAQuantizedGemm.cu`

**Kernel Dispatch Logic:**
```cpp
std::unique_ptr<ITensorGemm> Q4_0Tensor::createGemm() override {
    if (device_idx_ >= 0 && gpu_blocks_ != nullptr) {
        // Return GPU kernel
        return std::make_unique<CUDAQuantizedGemmKernel<Q4_0Block>>(
            gpu_blocks_, shape_, device_idx_);
    }
    // CPU kernel (existing)
    return std::make_unique<gemm_v4::QuantisedGemmKernel>(this);
}
```

**GPU Kernel Interface:**
```cpp
template<typename BlockType>
class CUDAQuantizedGemmKernel : public ITensorGemm {
public:
    CUDAQuantizedGemmKernel(void* gpu_blocks, 
                            const std::vector<size_t>& shape,
                            int device_idx);
    
    bool multiply(const float* A, float* C,
                  int m, int n, int k,
                  float alpha, float beta) override;
    
private:
    void* gpu_blocks_;
    std::vector<size_t> shape_;
    int device_idx_;
};
```

**Tests:**
- CPU kernel still works when tensor on CPU
- GPU kernel selected when tensor on GPU
- Parity test: CPU vs GPU results match (within tolerance)

---

### Phase 5: Pipeline Device Orchestration ✅ COMPLETE

**Goal:** Pipelines trigger lazy transfers based on `WeightPlacementMap`.

**Status:** Implemented December 3, 2025

**Files Modified:**
- `src/v2/pipelines/qwen/Qwen2Pipeline.h` - Added method declarations
- `src/v2/pipelines/qwen/Qwen2Pipeline.cpp` - Added implementations

**Implementation:**
```cpp
// Helper methods for lazy weight transfer
bool Qwen2Pipeline::ensureAttentionWeightsOnDevice(const LayerWeights &layer, int target_device)
{
    if (target_device < 0) return true;  // CPU path: no transfer needed
    
    // Lazy transfer attention weights (no-op if already on device)
    if (layer.wq && !layer.wq->ensureOnDevice(target_device)) return false;
    if (layer.wk && !layer.wk->ensureOnDevice(target_device)) return false;
    if (layer.wv && !layer.wv->ensureOnDevice(target_device)) return false;
    if (layer.wo && !layer.wo->ensureOnDevice(target_device)) return false;
    // ... biases, norms
    return true;
}

bool Qwen2Pipeline::ensureFFNWeightsOnDevice(const LayerWeights &layer, int target_device)
{
    if (target_device < 0) return true;  // CPU path: no transfer needed
    
    // Lazy transfer FFN weights (no-op if already on device)
    if (layer.gate_proj && !layer.gate_proj->ensureOnDevice(target_device)) return false;
    if (layer.up_proj && !layer.up_proj->ensureOnDevice(target_device)) return false;
    if (layer.down_proj && !layer.down_proj->ensureOnDevice(target_device)) return false;
    // ... norms
    return true;
}

// Called at start of attention_block():
bool Qwen2Pipeline::attention_block(const LayerWeights &layer, int layer_idx, ...) {
    int attn_device = placement_map_ ? getWeightDevice("attn_q", -1) : device_idx_;
    
    // Phase 5: Lazy transfer weights to target device
    if (!ensureAttentionWeightsOnDevice(layer, attn_device)) return false;
    
    // ... rest of attention computation
}

// Called at start of ffn_block():
bool Qwen2Pipeline::ffn_block(const LayerWeights &layer, int layer_idx, ...) {
    int ffn_device = placement_map_ ? getWeightDevice("ffn_gate", -1) : device_idx_;
    
    // Phase 5: Lazy transfer weights to target device  
    if (!ensureFFNWeightsOnDevice(layer, ffn_device)) return false;
    
    // ... rest of FFN computation
}
```

**Tests:**
- 113 unit tests pass (including KernelFactory tests)
- 30 integration tests pass (full pipeline execution)
- Transfers are no-op for CPU path (target_device < 0)
- Designed for GPU: will trigger actual transfers when GPU backend available

---

### Phase 6: GPU Quantized GEMM Kernels

**Goal:** Implement fused dequant+GEMM kernels for each quantized format.

**Files to Create:**
- `src/v2/kernels/cuda/dequant/Q4_0Dequant.cuh`
- `src/v2/kernels/cuda/dequant/IQ4_NLDequant.cuh`
- `src/v2/kernels/cuda/gemm/QuantizedGemm.cu`

**Kernel Design:**
```cuda
// Fused dequant + GEMM for Q4_0
// Each thread block handles a tile of output
// Dequantization happens in shared memory or registers
__global__ void q4_0_gemm_kernel(
    const Q4_0Block* __restrict__ B,  // Quantized weights
    const float* __restrict__ A,       // Activations (FP32)
    float* __restrict__ C,             // Output
    int M, int N, int K,
    float alpha, float beta
) {
    // Load Q4_0 block to shared memory
    // Dequantize to FP32 in registers
    // Compute partial dot product
    // Accumulate and write output
}
```

**Priority Order:**
1. Q4_0 (most common quantization)
2. Q8_0 (simple dequant, good baseline)
3. IQ4_NL (used in many small models)
4. Q6_K, Q4_K (K-quants for larger models)
5. Remaining formats

**Tests:**
- Parity with CPU dequant (bit-exact or within epsilon)
- Performance benchmarks vs cuBLAS with FP32 weights
- Memory bandwidth utilization

---

## Testing Strategy

### Unit Tests
- Device state machine transitions
- GPU allocator correctness
- Individual tensor type transfers

### Integration Tests
- Full pipeline with mixed CPU/GPU layers
- Multi-GPU distribution
- MPI + GPU combined

### Parity Tests
- CPU vs GPU numerical accuracy
- Quantized GPU vs dequantized cuBLAS

### Performance Tests
- Transfer overhead measurement
- Kernel throughput (TFLOPS)
- Memory bandwidth utilization

---

## Dependencies

### Required Libraries
- **CUDA Toolkit 12.x** or **ROCm 6.x**
- cuBLAS/rocBLAS (for FP32/FP16 fallback)

### Build System Changes
- CMake: Add CUDA/HIP language support
- Conditional compilation for GPU backends
- Separate CPU-only and GPU-enabled builds

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| GPU memory fragmentation | Use memory pools, pre-allocate based on model size |
| PCIe bottleneck | Async transfers, overlap with computation |
| Numerical divergence | Extensive parity testing, document expected differences |
| Multi-GPU complexity | Start single-GPU, add multi-GPU in Phase 7 |

---

## Success Criteria

1. **Functional:** Lazy transfer works for all quantized formats
2. **Correct:** GPU results match CPU within 1e-5 tolerance
3. **Performant:** GPU GEMM faster than CPU for batch size > 1
4. **Memory Efficient:** GPU memory < 1.1x quantized weight size

---

## Timeline Estimate

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Phase 1: Lazy Transfer API | 2-3 days | None |
| Phase 2: GPU Allocator | 2-3 days | Phase 1 |
| Phase 3: Tensor GPU Support | 3-5 days | Phase 2 |
| Phase 4: Kernel Dispatch | 2-3 days | Phase 3 |
| Phase 5: Pipeline Integration | 2-3 days | Phase 4 |
| Phase 6: GPU GEMM Kernels | 5-10 days | Phase 4 |

**Total Estimate:** 3-4 weeks for full GPU support

---

## References

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [ROCm Documentation](https://rocm.docs.amd.com/)
- llama.cpp GPU implementation (reference for quantized kernels)
- [Tensor Parallelism Paper](https://arxiv.org/abs/1909.08053) (Megatron-LM)
