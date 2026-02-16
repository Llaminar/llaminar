# CUDA Full Model Inference Debugging - Handover Document

**Date**: January 9, 2026  
**Status**: In Progress  
**Branch**: `feature/cuda-kernels`

## Executive Summary

We are debugging why CUDA full model inference produces divergent results compared to CPU inference, despite isolated kernel parity tests passing with high accuracy.

### Key Finding
**Isolated CUDA kernels work correctly** (GEMM cosine ~0.9998, Flash Attention now passing), but **full pipeline inference diverges** starting at the first projection layer (Q_PROJECTION cosine = 0.098).

## Recent Fixes (This Session)

### 1. DeviceId Refactoring
- **Problem**: Ambiguous `device_idx` integer references throughout codebase - unclear whether they referred to global device index, backend-specific index, or kernel device index.
- **Solution**: Refactored call sites to use `DeviceId` objects directly instead of raw integers.
- **Impact**: Cleaner device targeting, reduced confusion in multi-GPU and heterogeneous (CUDA+ROCm) scenarios.

### 2. CUDA Flash Attention Decode Path Fix
- **Problem**: The decode path in `CUDAFlashAttentionKernels.cu` was completely broken.
- **Solution**: Fixed the decode kernel implementation.
- **Verification**: New parity tests confirm the fix works correctly.
- **Key File**: [src/v2/kernels/cuda/attention/CUDAFlashAttentionKernels.cu](../../src/v2/kernels/cuda/attention/CUDAFlashAttentionKernels.cu)

## Current Debugging State

### Symptoms

| Test | Result | Notes |
|------|--------|-------|
| GEMM Parity (real weights) | ✅ cosine ~0.9998 | Isolated kernel works |
| Flash Attention Parity | ✅ Passing | After decode fix |
| RMSNorm Parity | ✅ Passing | Isolated kernel works |
| **Full Model Inference** | ❌ Diverges | Q_PROJECTION cosine = 0.098 |

### Divergence Location

```
╔══════════════════════════════════════════════════════════════════════════════╗
║ Full Inference Snapshot Comparison (CPU vs GPU)                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ EMBEDDING                           │   1.000000 │ ✓ Perfect                 ║
║ layer0_ATTENTION_NORM               │   1.000000 │ ✓ Perfect                 ║
║ layer0_Q_PROJECTION                 │   0.098653 │ ❌ Nearly orthogonal!      ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

The divergence starts at the **first GEMM operation** (Q projection) despite isolated GEMM tests passing.

### Critical Debug Log Finding

When running with `LLAMINAR_LOG_LEVEL=DEBUG`, we observed:

```
# CPU Pass:
[EmbeddingStage] output[0:8]=-0,0.00720215,0.00720215,0.0288086,-0,-0.0360107,...  device_id=CPU
[RMSNormStage] output[0:8]=0,0.0293105,-0.0236825,0.177124,0,-0.14644,...  device_id=CPU

# GPU Pass:
[EmbeddingStage] output[0:8]=-0,0.00720215,0.00720215,0.0288086,-0,-0.0360107,...  device_id=CUDA:0  ✓ SAME
[RMSNormStage] input[0:8]=1.06415,1.87386,1.86659,0.574819,-1.42822,6.62286,...  device_id=CUDA:0   ❌ DIFFERENT!
```

**Key Observation**: 
- EmbeddingStage produces **identical** output on CPU and GPU ✓
- RMSNormStage on GPU receives **completely different input** than what EmbeddingStage wrote ❌
- The values RMSNorm sees (`1.06415,1.87386,...`) don't match embedding output (`-0,0.00720215,...`)

## Suspected Root Cause: Tensor Coherence Issue

### Hypothesis

The tensor coherence model (host_valid_/device_valid_ flags) may not be correctly maintaining state between stages, causing GPU stages to read stale or wrong data.

### Coherence Model (Two-Flag System)

From [CPUTensors.h](../../src/v2/tensors/cpu/CPUTensors.h):

```cpp
bool host_valid_ = true;    // Host data is current
bool device_valid_ = false; // Device data is current

void mark_device_dirty() { 
    device_valid_ = true;   // Device just got written to
    host_valid_ = false;    // Host is now stale
}
```

### Flow Analysis

1. **EmbeddingStage** (GPU):
   - Writes to `current_hidden` buffer via GPU kernel
   - Calls `mark_device_dirty()` → `device_valid_=true, host_valid_=false`
   
2. **RMSNormStage** (GPU):
   - Reads from `current_hidden` buffer
   - Kernel calls `input_fp32->ensureOnDevice(target_device)`
   - `ensureOnDevice()` should see `device_valid_=true` and return immediately
   - **But**: Debug logs show it's reading different data

### Potential Issues to Investigate

1. **Buffer Aliasing**: Are `current_hidden` buffers actually the same object between stages?
   - Check [Qwen2Graph.cpp](../../src/v2/models/qwen/Qwen2Graph.cpp) buffer wiring

2. **Debug Log Reading Host Data**: The debug logging may be calling `data()` which syncs from device to host, but the printed values suggest GPU has wrong data.

3. **Hardcoded CPU Devices** (Partially Fixed):
   - [BufferRole.h](../../src/v2/execution/BufferRole.h) lines 372-400: Static builders hardcode `DeviceId::cpu()`
   - [DeviceGraphBufferManager.cpp](../../src/v2/execution/DeviceGraphBufferManager.cpp) line 289: `group_desc.device = DeviceId::cpu();`
   - Note: `GraphOrchestrator::initializeInferenceState()` does pass device correctly to TensorFactory

4. **Multiple Tensor Instances**: Could there be separate tensor objects for the same logical buffer?

## Key Files for Investigation

| File | Purpose | Notes |
|------|---------|-------|
| [CPUTensorBase.cpp](../../src/v2/tensors/cpu/CPUTensorBase.cpp) | Coherence implementation | `ensureOnDevice()`, `mark_device_dirty()` |
| [CPUTensors.h](../../src/v2/tensors/cpu/CPUTensors.h) | FP32Tensor with coherence | Lines 1469-1472: `mark_device_dirty()` |
| [CUDAOpsKernels.cpp](../../src/v2/kernels/cuda/ops/CUDAOpsKernels.cpp) | CUDA kernel implementations | Embedding line 909, RMSNorm line 152 |
| [Qwen2Graph.cpp](../../src/v2/models/qwen/Qwen2Graph.cpp) | Buffer wiring | Lines 700-770: Stage buffer connections |
| [GraphOrchestrator.cpp](../../src/v2/execution/GraphOrchestrator.cpp) | Buffer initialization | Lines 730-900: `initializeInferenceState()` |
| [DeviceGraphBufferManager.cpp](../../src/v2/execution/DeviceGraphBufferManager.cpp) | Buffer allocation | Line 289: Hardcoded CPU device |
| [BufferRole.h](../../src/v2/execution/BufferRole.h) | Buffer descriptors | Lines 372-400: Static builders hardcode CPU |

## Test Commands

```bash
# Run the failing test with debug logging
LLAMINAR_LOG_LEVEL=DEBUG ./build_v2/tests/v2/v2_integration_cuda_full_model_inference

# Filter for coherence-related logs
LLAMINAR_LOG_LEVEL=DEBUG ./build_v2/tests/v2/v2_integration_cuda_full_model_inference 2>&1 | \
  grep -E "(device_valid|host_valid|ensureOnDevice|mark_device_dirty)"

# Filter for stage data flow
LLAMINAR_LOG_LEVEL=DEBUG ./build_v2/tests/v2/v2_integration_cuda_full_model_inference 2>&1 | \
  grep -E "(EmbeddingStage|RMSNormStage).*output|input"

# Run isolated GEMM parity (should pass)
./build_v2/tests/v2/v2_integration_cuda_quantised_gemm_parity

# Run Flash Attention parity (should now pass)
./build_v2/tests/v2/v2_integration_cuda_flash_attention_parity
```

## Recommended Next Steps

### Priority 1: Verify Buffer Identity
Add logging to confirm the same tensor object is used:
```cpp
// In EmbeddingStage after write:
LOG_DEBUG("EmbeddingStage wrote to current_hidden @ " << (void*)current_hidden_ptr 
          << " device_valid=" << current_hidden->isDeviceValid());

// In RMSNormStage before read:
LOG_DEBUG("RMSNormStage reading from input @ " << (void*)input_ptr 
          << " device_valid=" << input->isDeviceValid());
```

### Priority 2: Trace Coherence State
Add state logging to `ensureOnDevice()`:
```cpp
LOG_DEBUG("[ensureOnDevice] tensor=" << (void*)this 
          << " host_valid=" << host_valid_ 
          << " device_valid=" << device_valid_
          << " gpu_ptr=" << gpu_data_ptr_);
```

### Priority 3: Check Buffer Allocation Path
Trace whether `DeviceGraphBufferManager::allocateAliasingGroups()` is creating buffers with wrong device affinity.

### Priority 4: Validate Debug Log Accuracy
The debug output may be reading from host after device write. Verify by:
1. Adding `cudaDeviceSynchronize()` before debug reads
2. Using `cudaMemcpy` to read GPU data directly for debug output

## Environment

- **Model**: qwen2.5-0.5b-instruct-q4_0.gguf (Q4_0 quantized)
- **Build**: Debug build with assertions enabled
- **CUDA**: Confirmed CUDA device available and functional

## Summary

The core issue is **data flow between pipeline stages on GPU** - the embedding stage writes correct data, but the next stage (RMSNorm) reads incorrect data. This suggests either:
1. A tensor coherence bug where the wrong memory is being read
2. Buffer aliasing/identity issue where stages use different tensor objects
3. A race condition or missing synchronization

The individual CUDA kernels are verified working. The bug is in the pipeline data flow machinery.
