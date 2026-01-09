# CUDA Full Model Inference SEGFAULT - Debug Handover

**Date**: January 9, 2026  
**Status**: In Progress  
**Priority**: High  

## Executive Summary

A SEGFAULT occurs during GPU execution in the `V2_Integration_CUDAFullModelInference` test. The crash happens after successful weight upload to GPU, during the actual GEMM kernel execution. The `ensureOnDevice(DeviceId)` API migration was completed successfully, but the underlying GPU execution issue remains.

---

## Test Under Investigation

**Test Name**: `V2_Integration_CUDAFullModelInference`  
**Test File**: [tests/v2/integration/Test__CUDABasicPipeline.cpp](../../tests/v2/integration/Test__CUDABasicPipeline.cpp)  
**Build**: Integration build (`build_v2_integration`)  

### How to Run

```bash
# Build integration (has snapshots + debug symbols)
cmake -B build_v2_integration -S src/v2 -DCMAKE_BUILD_TYPE=Integration
cmake --build build_v2_integration --parallel

# Run the failing test
ctest --test-dir build_v2_integration -R "V2_Integration_CUDAFullModelInference" --output-on-failure
```

### Test Purpose

The test verifies end-to-end GPU inference by:
1. Loading a real GGUF model (qwen2.5-0.5b-instruct-q4_0.gguf)
2. Creating an inference runner with GPU device target
3. Running prefill on a prompt
4. Comparing output against CPU reference

---

## Crash Location

The SEGFAULT occurs in `FusedQKVGEMMStage::execute()` after:
1. ✅ Weight tensors successfully uploaded to GPU via `ensureOnDevice()`
2. ✅ GPU device pointers obtained via `gpu_data_ptr()`
3. ❌ CRASH during actual GEMM kernel execution

### Last Successful Log Lines

```
[FusedQKVGEMMStage] Execute: m=2 k=896 n_q=896 n_k=128 n_v=128
[CPUTensorBase::ensureOnDevice] Uploaded 1835008 bytes to device CUDA:0 (backend device ID: 0)
[CPUTensorBase::ensureOnDevice] Uploaded 1835008 bytes to device CUDA:0 (backend device ID: 0)
[CPUTensorBase::ensureOnDevice] Uploaded 262144 bytes to device CUDA:0 (backend device ID: 0)
[CPUTensorBase::ensureOnDevice] Uploaded 262144 bytes to device CUDA:0 (backend device ID: 0)
[FusedQKVGEMMStage] GPU execution using device pointers (CUDA:0)
--- SEGFAULT ---
```

---

## Working Hypotheses

### Hypothesis 1: Input/Output Buffer Location Mismatch (MOST LIKELY)

The weight tensors are uploaded to GPU, but the **input activations** and **output buffers** may still be on CPU. The CUDA kernel would then receive:
- Weight pointers: GPU memory ✅
- Input pointers: CPU memory ❌ (causes SEGFAULT when CUDA tries to read)
- Output pointers: CPU memory ❌ (causes SEGFAULT when CUDA tries to write)

**Evidence**: Only weight uploads are logged. No uploads for input/output buffers.

**Fix Direction**: 
1. Upload input tensor to GPU before GEMM
2. Allocate output buffer on GPU
3. Download output back to CPU after GEMM (or keep on GPU for next stage)

### Hypothesis 2: Kernel Receives Wrong Device Type

The `KernelFactory::getOrCreateGemm(tensor, DeviceType::GPU)` was added to force GPU kernel selection. However, the kernel may still be checking the tensor's native device and getting confused.

**Fix Direction**: Verify the CUDA GEMM kernel is actually being created and called.

### Hypothesis 3: cuBLAS Handle Not Initialized

CUDA GEMM kernels typically require a cuBLAS handle. If the handle isn't properly initialized for the current CUDA context, the kernel will crash.

**Fix Direction**: Check `CUDAFloatingPointGemmKernel` or `CUDAQuantisedGemmKernel` initialization.

---

## Key Source Files

### Stage Implementation
- [src/v2/execution/compute_stages/FusedQKVGEMMStage.cpp](../../src/v2/execution/compute_stages/FusedQKVGEMMStage.cpp) - Where crash occurs
- [src/v2/execution/compute_stages/FusedQKVGEMMStage.h](../../src/v2/execution/compute_stages/FusedQKVGEMMStage.h)

### GPU Data Transfer
- [src/v2/tensors/cpu/CPUTensorBase.cpp](../../src/v2/tensors/cpu/CPUTensorBase.cpp) - `ensureOnDevice()`, `gpu_data_ptr()`, `mark_device_dirty()`
- [src/v2/tensors/cpu/CPUTensors.h](../../src/v2/tensors/cpu/CPUTensors.h) - API declarations

### Kernel Factory
- [src/v2/kernels/KernelFactory.cpp](../../src/v2/kernels/KernelFactory.cpp) - `getOrCreateGemm(tensor, DeviceType)`
- [src/v2/kernels/KernelFactory.h](../../src/v2/kernels/KernelFactory.h)

### CUDA Kernels
- [src/v2/kernels/cuda/CUDAFloatingPointGemmKernel.cpp](../../src/v2/kernels/cuda/CUDAFloatingPointGemmKernel.cpp)
- [src/v2/kernels/cuda/CUDAQuantisedGemmKernel.cpp](../../src/v2/kernels/cuda/CUDAQuantisedGemmKernel.cpp)

### Device Management
- [src/v2/backends/DeviceId.h](../../src/v2/backends/DeviceId.h) - `DeviceId` type
- [src/v2/backends/DeviceContext.cpp](../../src/v2/backends/DeviceContext.cpp) - GPU context creation

### Test File
- [tests/v2/integration/Test__CUDABasicPipeline.cpp](../../tests/v2/integration/Test__CUDABasicPipeline.cpp)

---

## Recent Changes (This Session)

### 1. ensureOnDevice(DeviceId) API Migration ✅

Changed `ensureOnDevice(int legacy_idx)` to `ensureOnDevice(DeviceId target_device)` for type safety.

**Files Modified**:
- `CPUTensors.h` - Signature change
- `CPUTensorBase.cpp` - Implementation change
- `FusedQKVGEMMStage.cpp` - Caller updated
- All test files - Updated to use `DeviceId`

### 2. Device-Targeted Kernel Selection ✅

Added `KernelFactory::getOrCreateGemm(tensor, DeviceType target_device)` overload to explicitly request GPU kernels even when the weight tensor's native device is CPU.

### 3. GPU Data Handling in FusedQKVGEMMStage ✅

Added logic to:
1. Upload weights to GPU via `ensureOnDevice(params_.device_id)`
2. Get GPU pointers via `gpu_data_ptr()`
3. Mark output dirty via `mark_device_dirty()`

**Current Code** (lines ~300-340 in FusedQKVGEMMStage.cpp):
```cpp
if (params_.device_id.is_gpu())
{
    // Upload weights to GPU
    wq->ensureOnDevice(params_.device_id);
    wk->ensureOnDevice(params_.device_id);
    wv->ensureOnDevice(params_.device_id);
    // ... get gpu_data_ptr() and call kernel ...
}
```

---

## What's Missing (Likely Fix)

The input activations (`input_` tensor) and output buffers (`q_output_`, `k_output_`, `v_output_`) need GPU handling:

```cpp
// MISSING: Upload input to GPU
input_->ensureOnDevice(params_.device_id);
const float* input_gpu = static_cast<const float*>(input_->gpu_data_ptr());

// MISSING: Allocate outputs on GPU (or ensure they're there)
q_output_->ensureOnDevice(params_.device_id);
k_output_->ensureOnDevice(params_.device_id);
v_output_->ensureOnDevice(params_.device_id);

// After GEMM:
q_output_->mark_device_dirty();
k_output_->mark_device_dirty();
v_output_->mark_device_dirty();
```

---

## Debugging Commands

### Run with GDB

```bash
cd /workspaces/llaminar

# Create GDB command file
cat > /tmp/gdbcommands.txt << 'EOF'
set debuginfod enabled off
set pagination off
handle SIGSEGV stop print
run
thread apply all bt full
quit
EOF

# Run under GDB (single rank for simplicity)
gdb -x /tmp/gdbcommands.txt --args \
  ./build_v2_integration/tests/v2/v2_integration_cuda_full_model_inference
```

### Run with CUDA-MEMCHECK

```bash
compute-sanitizer --tool memcheck \
  ./build_v2_integration/tests/v2/v2_integration_cuda_full_model_inference
```

### Enable Verbose Logging

```bash
LLAMINAR_LOG_LEVEL=TRACE \
  ./build_v2_integration/tests/v2/v2_integration_cuda_full_model_inference
```

---

## Other Stages That May Need Similar Fixes

If Hypothesis 1 is correct (input/output buffer location mismatch), these stages also need GPU data handling:

1. **GEMMStage** - [src/v2/execution/compute_stages/GEMMStage.cpp](../../src/v2/execution/compute_stages/GEMMStage.cpp)
2. **LMHeadStage** - [src/v2/execution/compute_stages/LMHeadStage.cpp](../../src/v2/execution/compute_stages/LMHeadStage.cpp)
3. **FusedGateUpGEMMStage** - [src/v2/execution/compute_stages/FusedGateUpGEMMStage.cpp](../../src/v2/execution/compute_stages/FusedGateUpGEMMStage.cpp)
4. **Attention stages** - Various files in `compute_stages/`

---

## Architecture Notes

### DeviceId Type

```cpp
// DeviceId provides type-safe device identification
DeviceId::cpu()           // CPU device
DeviceId::cuda(ordinal)   // CUDA GPU with ordinal 0, 1, 2...
DeviceId::rocm(ordinal)   // ROCm GPU with ordinal 0, 1, 2...

// Key methods:
device_id.is_gpu()        // true for CUDA/ROCm
device_id.is_cuda()       // true for CUDA only
device_id.toLegacyIndex() // Convert to DeviceManager index (CPU=-1, GPUs=0,1,2...)
device_id.toKernelDeviceIndex() // Convert to CUDA ordinal (0,1,2...)
device_id.toString()      // "CPU", "CUDA:0", "ROCm:1", etc.
```

### Tensor GPU Transfer API

```cpp
// Upload to GPU
tensor->ensureOnDevice(DeviceId::cuda(0));

// Get GPU pointer (for CUDA kernels)
const void* gpu_ptr = tensor->gpu_data_ptr();

// Mark as modified on GPU (needs sync before CPU access)
tensor->mark_device_dirty();

// Sync back to CPU
tensor->sync_from_device();

// Download to CPU
tensor->ensureOnHost();
```

---

## Success Criteria

The test should:
1. Run without SEGFAULT
2. Produce non-zero logits from GPU execution
3. Match CPU reference output within tolerance (MSE < 1e-4 or similar)

---

## Contact

This document was created during a debugging session. The `ensureOnDevice(DeviceId)` API migration is complete - the remaining work is fixing the GPU execution path to handle input/output buffers correctly.
