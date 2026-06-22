# CUDA Tensor Design Analysis and Fix

**Date**: January 2026  
**Status**: ✅ Fixed (January 8, 2026)

## 1. Current Architecture

### 1.1 How Tensors Get to GPU

There are **two** different patterns for GPU tensor usage:

#### Pattern A: CPU Tensor + Lazy Transfer (Current Working Pattern)
```cpp
// Create on CPU
auto weights = std::make_unique<IQ4_NLTensor>(shape, data_ptr);

// Transfer to GPU lazily
weights->ensureOnDevice(gpu_idx);

// Access GPU pointer via existing API
void* d_ptr = weights->gpu_data_ptr();  // Returns device pointer
```

**Key Methods in CPUTensorBase** (defined in `CPUTensors.h`):
- `ensureOnDevice(int target_device)` - Uploads raw bytes to GPU
- `ensureOnHost()` - Downloads from GPU back to host
- `gpu_data_ptr()` - Returns device pointer (or nullptr if not on GPU)
- `isOnGPU()` - Returns `gpu_data_ptr_ != nullptr`
- `raw_data()` / `raw_mutable_data()` - Returns **host** pointer (always)
- `active_data_ptr()` / `active_mutable_data_ptr()` - Returns GPU pointer if on GPU, else host pointer ✅ **Added**

#### Pattern B: GPU-Native Tensor (Broken Infrastructure)
```cpp
// Attempt to create directly on GPU
auto cuda_tensor = std::make_unique<CUDAFp32Tensor>(shape, device_idx);

// Access GPU data
float* d_ptr = cuda_tensor->typed_data();  // Device pointer
```

**Issues with Pattern B** (defined in `CUDATensorBase.h`, `CUDATypedTensor.h`):
1. `CUDATensorBase` has `override` markers on methods that don't exist in `ITensor`
2. `CUDATypedTensor<T>` tries to override `data()` with return type `T*`, but base returns `float*`
3. Only `CUDAFp32Tensor` compiles (because T=float matches float* return type)
4. `CUDAINT8Tensor`, `CUDAINT32Tensor` fail to compile

### 1.2 How Stages Use GPU Tensors

From `AttentionComputeStage.cpp`:
```cpp
const bool is_gpu_tensor = params_.Q->is_on_gpu();

if (is_gpu_tensor) {
    // GPU path: use raw_data() which returns device pointer when on GPU
    const float *Q_ptr = static_cast<const float *>(params_.Q->raw_data());
    // ... kernel launch with device pointers
}
```

**IMPORTANT DISCOVERY**: The current GPU path uses `raw_data()` expecting it to return a **device** pointer when the tensor is on GPU. But `CPUTensorBase::raw_data()` **always returns the host pointer**:

```cpp
// CPUTensors.h line 606
const void *raw_data() const override { return raw_host_data_ptr(); }
```

This means stages should be using `gpu_data_ptr()` instead!

### 1.3 What Kernels Need

From `CUDAQuantisedGemmKernel.h`:
```cpp
// Weight conversion: Takes TensorBase*, dequantizes to FP32, requantizes to INT8
bool packWeightsToCUDA(const TensorBase *tensor, CUDAPackedWeights &out);

// The kernel takes TensorBase* and extracts pointers internally
bool multiply_tensor(const TensorBase *A, TensorBase *C, ...);
```

From `CUDAResidualAddKernelT.h`:
```cpp
// Assumes raw_data() returns device pointer!
static_cast<const uint16_t *>(input->raw_data())  // BUG: This is host pointer!
```

## 2. Use Cases for CUDA Tensors

### 2.1 Activation Tensors (FP32, BF16, FP16)

**Lifecycle**:
1. Created on CPU during buffer allocation
2. Uploaded to GPU via `ensureOnDevice()` at start of GPU pipeline
3. Used throughout GPU pipeline (attention, GEMM, norms)
4. Downloaded back to host via `ensureOnHost()` for logits/output

**Needed Operations**:
- FP32/BF16/FP16 pointwise ops (RMSNorm, RoPE, SwiGLU, residual add)
- GEMM (as A matrix)
- Attention (Q, K, V, output)

### 2.2 Quantized Weight Tensors (IQ4_NL, Q4_0, Q6_K, etc.)

**Lifecycle**:
1. Loaded from GGUF on CPU (quantized blocks)
2. Uploaded to GPU via `ensureOnDevice()` (raw blocks)
3. Converted to INT8 on first GEMM kernel use (cached)
4. Stay on GPU for duration of inference

**Needed Operations**:
- GEMM (as B/weight matrix) - converted to INT8 column-major
- Dequantization for debugging/snapshots

### 2.3 Q8_1 Activation Tensors (Quantized Activations)

**Lifecycle**:
1. CPU path: FP32 → Q8_1 quantization before GEMM
2. GPU path: Should use FP32 activations (GPU does activation quantization in kernel)

**Needed Operations**:
- GEMM (as A matrix with embedded scales)
- CPU path only - GPU uses FP32 activations

### 2.4 INT32 Accumulator Tensors

**Lifecycle**:
- Temporary buffers for INT8×INT8→INT32 GEMM
- GPU-only, don't need host copy

**Needed Operations**:
- GEMM output buffer
- Scale and convert to FP32/Q8_1

### 2.5 KV Cache

**Lifecycle**:
1. Allocated on GPU at initialization
2. Updated during decode (K/V writes)
3. Read during attention (K/V reads)
4. Never transferred to CPU during normal inference

**Needed Operations**:
- Attention K/V access
- Cache update (append new K/V)
- Ring buffer management

## 3. Design Problems

### 3.1 `raw_data()` Inconsistency

**Problem**: Stages call `raw_data()` expecting device pointer when `is_on_gpu()` is true, but `CPUTensorBase::raw_data()` always returns host pointer.

**Evidence** from AttentionComputeStage.cpp:
```cpp
if (is_gpu_tensor) {
    const float *Q_ptr = static_cast<const float *>(params_.Q->raw_data());
    // BUG: Q_ptr is HOST pointer, not device!
}
```

### 3.2 CUDATypedTensor Design Flaw

**Problem**: The template tries to override `data()` with different return types.

```cpp
// ITensor (and CUDATensorBase)
virtual const float *data() const = 0;

// CUDATypedTensor<int8_t, INT8>
const int8_t* data() const { ... }  // COMPILE ERROR: conflicting return type
```

### 3.3 Missing Device-Aware Data Access

**Problem**: No clean way to get "the right pointer for where the tensor is".

Currently need:
```cpp
void* ptr;
if (tensor->is_on_gpu()) {
    ptr = tensor->gpu_data_ptr();  // Device pointer
} else {
    ptr = tensor->raw_mutable_data();  // Host pointer
}
```

## 4. Proposed Fix

### Option A: Fix CPUTensorBase (Minimal Change)

Add a unified data access method that returns the "active" pointer:

```cpp
// CPUTensorBase (add to CPUTensors.h)
/**
 * @brief Get pointer to data where it currently resides
 * @return GPU pointer if isOnGPU(), else host pointer
 */
const void* active_data_ptr() const {
    return gpu_data_ptr_ ? gpu_data_ptr_ : raw_host_data_ptr();
}
void* active_mutable_data_ptr() {
    return gpu_data_ptr_ ? gpu_data_ptr_ : raw_host_data_ptr();
}
```

Then update stages to use `active_data_ptr()` instead of `raw_data()`.

### Option B: Virtual Dispatch on raw_data() (Breaking Change)

Override `raw_data()` in CPUTensorBase to be device-aware:

```cpp
const void* raw_data() const override {
    return gpu_data_ptr_ ? gpu_data_ptr_ : raw_host_data_ptr();
}
```

**Risk**: Breaks code that expects `raw_data()` to always return host pointer.

### Option C: Remove CUDATypedTensor (Simplification)

Delete the broken `CUDATypedTensor` infrastructure. Instead:

1. All tensors stay as `CPUTensorBase` derived types
2. Use `ensureOnDevice()` / `gpu_data_ptr()` for GPU access
3. Kernels always take `ITensor*` or `TensorBase*`

This matches how the code **actually works today**.

### Option D: Fix CUDATypedTensor Properly

If we want GPU-native tensors:

```cpp
// CUDATensorBase changes:
// 1. Don't override data() - leave it pure virtual from ITensor
// 2. Provide typed accessors without overriding

class CUDATensorBase : public ITensor {
    // DON'T override data() from ITensor
    // const float* data() const override { return nullptr; }  // REMOVE

    // Device data access - new API
    const void* device_data() const { return device_ptr_; }
    void* mutable_device_data() { return device_ptr_; }
};

// CUDATypedTensor changes:
template<typename T, TensorType DType>
class CUDATypedTensor : public CUDATensorBase {
    // Type-safe device accessors (don't override ITensor::data)
    const T* typed_device_data() const { return static_cast<T*>(device_ptr_); }
    T* mutable_typed_device_data() { return static_cast<T*>(device_ptr_); }
    
    // ITensor::data() returns nullptr for GPU tensors (data is on device)
    const float* data() const override { return nullptr; }
    float* mutable_data() override { return nullptr; }
};
```

## 5. Recommended Solution

**Option A + C**: Minimal fix with simplification.

1. **Add `active_data_ptr()`** to CPUTensorBase for clean device-aware access
2. **Fix stages** to use `active_data_ptr()` or `gpu_data_ptr()`  
3. **Keep CUDATypedTensor** for `CUDAFp32Tensor` only (it works)
4. **Don't expand CUDATypedTensor** to other types
5. **Document** the pattern: All weight/activation tensors use CPUTensorBase + ensureOnDevice()

This preserves backward compatibility while fixing the actual bugs.

## 6. Action Items

### Priority 1: Fix the Stage Bug
1. [x] Audit stages using `raw_data()` for GPU tensors  
2. [x] Fix `AttentionComputeStage` to use `active_data_ptr()` when `is_on_gpu()` ✅
3. [x] Fix `CUDAResidualAddKernelT` to use `active_data_ptr()` ✅
4. [x] Add helper method `active_data_ptr()` to ITensor and CPUTensorBase ✅

### Priority 2: Clean Up CUDATensorBase
1. [x] Remove invalid `override` markers from `CUDATensorBase` → **Deleted entire file**
2. [x] Remove broken `CUDAINT8Tensor`, `CUDAINT32Tensor` type aliases → **Deleted CUDATypedTensor.h**

### Priority 3: Testing
1. [ ] Add integration test for CPU tensor → GPU transfer → kernel execution
2. [ ] Test with real GPU (run existing V2_Integration_CUDABasicPipeline)
3. [x] Document the GPU tensor pattern (this file)

## 7. Fix Implementation (January 8, 2026)

The fix adds `active_data_ptr()` to the ITensor interface with a device-aware override in CPUTensorBase:

```cpp
// ITensor.h - Default implementation (returns host pointer)
virtual const void *active_data_ptr() const { return raw_data(); }
virtual void *active_mutable_data_ptr() { return raw_mutable_data(); }

// CPUTensors.h - Device-aware override
const void *active_data_ptr() const override {
    return gpu_data_ptr_ ? gpu_data_ptr_ : raw_host_data_ptr();
}
void *active_mutable_data_ptr() override {
    return gpu_data_ptr_ ? gpu_data_ptr_ : raw_host_data_ptr();
}
```

Stages now use `active_data_ptr()` instead of `raw_data()`:

```cpp
// BEFORE (BUGGY):
if (is_gpu_tensor) {
    const float *Q_ptr = static_cast<const float *>(params_.Q->raw_data());
    // BUG: Q_ptr is HOST pointer!
}

// AFTER (FIXED):
if (is_gpu_tensor) {
    // active_data_ptr() returns GPU pointer when tensor is on GPU
    const float *Q_ptr = static_cast<const float *>(params_.Q->active_data_ptr());
    // Q_ptr is now correctly a DEVICE pointer
}
```

## 8. Summary

**Root Cause**: Stages incorrectly used `raw_data()` (always returns host) when they should use the active device pointer.

**Fix**: Added `active_data_ptr()` / `active_mutable_data_ptr()` to ITensor interface:
- Default implementation returns `raw_data()` (host pointer)
- CPUTensorBase override returns GPU pointer if `gpu_data_ptr_` is set, else host pointer
- No breaking changes - existing code using `raw_data()` still works
- Stages updated to use `active_data_ptr()` for kernel dispatch

**Why It Wasn't Caught**: 
- CUDA kernels use `gpu_data_ptr()` correctly
- Stages used `raw_data()` incorrectly  
- If CUDA is unavailable, stages take CPU path (works)
- If CUDA is available but tensors not on GPU, stages take CPU path (works)
- Only fails when tensors ARE on GPU → passes host pointer to CUDA kernel → segfault/garbage

**Correct Pattern Going Forward**:
```cpp
// Use active_data_ptr() for kernel dispatch - works for both CPU and GPU
void* data_ptr = tensor->active_mutable_data_ptr();

// Use raw_data() only when you explicitly need the HOST pointer
void* host_ptr = tensor->raw_mutable_data();

// Use gpu_data_ptr() only when you explicitly need the GPU pointer
void* gpu_ptr = dynamic_cast<CPUTensorBase*>(tensor)->gpu_data_ptr();
```

