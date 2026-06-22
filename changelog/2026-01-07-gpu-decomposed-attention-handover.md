# Handover: Enable GPU Decode via Decomposed Attention

**Date**: January 7, 2026  
**Branch**: `feature/cuda-kernels`  
**Status**: Ready for Implementation  
**Estimated Effort**: 1-2 days

---

## Executive Summary

GPU prefill works, but **GPU decode is blocked** because `AttentionComputeStage` (and other stages) cast `ITensor*` to `TensorBase*`, which fails for GPU tensors (`CUDATensorBase*`).

**Goal**: Enable GPU decode by updating `AttentionComputeStage` to dispatch attention computation directly via `ITensor*` interface, using `CUDAFlashAttentionKernelT` for GPU tensors.

---

## Problem Statement

### Root Cause

All stage implementations use `requireTensorBase()` from [ComputeStageUtils.h](../src/v2/execution/compute_stages/ComputeStageUtils.h):

```cpp
// This blocks GPU tensors
inline TensorBase *requireTensorBase(ITensor *tensor, const char *name)
{
    auto *base = dynamic_cast<TensorBase *>(tensor);
    LLAMINAR_ASSERTF(base, name << " must be a CPU TensorBase (GPU not yet supported)");
    return base;
}
```

When KV cache returns `CUDATensorBase*` (which inherits from `ITensor`, NOT `TensorBase`), the cast fails.

### Current Test Status

From [Test__CUDAFullModelInference.cpp](tests/v2/integration/Test__CUDAFullModelInference.cpp):

| Test | Status |
|------|--------|
| `SingleTokenPrediction_MatchesCPU` | ✅ Pass (prefill) |
| `LongerPrompt_MatchesCPU` | ✅ Pass (prefill) |
| `MultiTokenGeneration_MatchesCPU` | ⛔ **SKIP** (decode blocked) |

The skip message (line 436-443):
```cpp
GTEST_SKIP() << "GPU decode not yet supported - FusedAttentionWoStage requires TensorBase* "
             << "but GPU KV cache returns CUDATensorBase*";
```

---

## Solution: Update AttentionComputeStage for GPU Dispatch

### Files to Modify

1. **[AttentionComputeStage.cpp](../src/v2/execution/compute_stages/stages/AttentionComputeStage.cpp)** - Add GPU tensor dispatch
2. **[KernelFactory.h](../src/v2/kernels/KernelFactory.h)** - Add `ITensor*` overload for `createAttention()`
3. **[KernelFactory.cpp](../src/v2/kernels/KernelFactory.cpp)** - Implement `ITensor*` dispatch
4. **[Test__CUDAFullModelInference.cpp](tests/v2/integration/Test__CUDAFullModelInference.cpp)** - Remove GTEST_SKIP

### Optional: Configure Qwen2Graph for GPU
5. **[Qwen2Graph.cpp](src/v2/models/qwen/Qwen2Graph.cpp)** - Force decomposed attention for GPU path

---

## Implementation Guide

### Step 1: Add ITensor* Overload to KernelFactory

**File**: `src/v2/kernels/KernelFactory.h`

Add after the existing `TensorBase*` overload (around line 674):

```cpp
/**
 * @brief Create Attention kernel for any ITensor via device-aware dispatch
 *
 * Unlike TensorBase* overload, this handles both CPU (TensorBase*) and
 * GPU (CUDATensorBase*) tensors via ITensor interface.
 *
 * @param tensor Input tensor (ITensor interface)
 * @param dev_type Target device type
 * @return ITensorAttention implementation
 */
static std::unique_ptr<llaminar2::ITensorAttention> createAttention(
    const llaminar2::ITensor *tensor, DeviceType dev_type);
```

**File**: `src/v2/kernels/KernelFactory.cpp`

Add the implementation (after the existing `createAttention(TensorBase*)` around line 1960):

```cpp
std::unique_ptr<llaminar2::ITensorAttention> KernelFactory::createAttention(
    const llaminar2::ITensor *tensor, DeviceType dev_type)
{
    if (!tensor)
    {
        throw std::runtime_error("KernelFactory::createAttention: null tensor");
    }

    // For GPU tensors, dispatch to CUDA Flash Attention directly
    if (tensor->is_on_gpu())
    {
#ifdef HAVE_CUDA
        // GPU tensors use CUDAFlashAttentionKernelT based on native_type
        switch (tensor->native_type())
        {
        case llaminar2::TensorType::FP32:
            return std::make_unique<llaminar2::cuda::CUDAFlashAttentionKernelT<llaminar2::ActivationPrecision::FP32>>();
        case llaminar2::TensorType::FP16:
            return std::make_unique<llaminar2::cuda::CUDAFlashAttentionKernelT<llaminar2::ActivationPrecision::FP16>>();
        case llaminar2::TensorType::BF16:
            return std::make_unique<llaminar2::cuda::CUDAFlashAttentionKernelT<llaminar2::ActivationPrecision::BF16>>();
        default:
            throw std::runtime_error(
                "KernelFactory::createAttention: unsupported GPU tensor type " +
                std::string(tensor->dtype_name()));
        }
#else
        throw std::runtime_error("KernelFactory::createAttention: GPU tensor but CUDA not available");
#endif
    }

    // For CPU tensors, use existing TensorBase* dispatch
    auto *cpu_tensor = dynamic_cast<const TensorBase *>(tensor);
    if (!cpu_tensor)
    {
        throw std::runtime_error("KernelFactory::createAttention: non-GPU ITensor must be TensorBase");
    }
    return createAttention(cpu_tensor, dev_type);
}
```

### Step 2: Update AttentionComputeStage for GPU Dispatch

**File**: `src/v2/execution/compute_stages/stages/AttentionComputeStage.cpp`

Replace the current CPU-only block (lines 83-114) with GPU-aware dispatch:

**BEFORE** (lines 83-114):
```cpp
// Cast ITensor* to TensorBase* for CPU operations
auto *Q_base = requireTensorBase(params_.Q, "Q");
auto *K_base = requireTensorBase(params_.K, "K");
auto *V_base = requireTensorBase(params_.V, "V");
auto *output_base = requireTensorBase(params_.output, "output");
if (!Q_base || !K_base || !V_base || !output_base)
{
    LOG_ERROR("[AttentionComputeStage] GPU tensors not yet supported");
    return false;
}

// Determine device type from context
using DeviceType = llaminar::v2::kernels::DeviceType;
DeviceType dev_type = DeviceType::CPU;

if (ctx)
{
    auto *gpu_ctx = dynamic_cast<IGPUDeviceContext *>(ctx);
    if (gpu_ctx)
    {
#if defined(HAVE_CUDA)
        dev_type = DeviceType::CUDA;
#elif defined(HAVE_ROCM)
        dev_type = DeviceType::ROCm;
#endif
    }
}

// Create attention kernel via KernelFactory
auto kernel = llaminar::v2::kernels::KernelFactory::createAttention(Q_base, dev_type);
```

**AFTER**:
```cpp
// Determine device type from tensor location
using DeviceType = llaminar::v2::kernels::DeviceType;
DeviceType dev_type = DeviceType::CPU;

if (params_.Q->is_on_gpu())
{
#if defined(HAVE_CUDA)
    dev_type = DeviceType::CUDA;
    LOG_DEBUG("[AttentionComputeStage] Using CUDA path for GPU tensors");
#elif defined(HAVE_ROCM)
    dev_type = DeviceType::ROCm;
    LOG_DEBUG("[AttentionComputeStage] Using ROCm path for GPU tensors");
#else
    LOG_ERROR("[AttentionComputeStage] GPU tensors but no GPU backend compiled");
    return false;
#endif
}
else if (ctx)
{
    // Check context for GPU preference (for CPU tensors that should run on GPU)
    auto *gpu_ctx = dynamic_cast<IGPUDeviceContext *>(ctx);
    if (gpu_ctx)
    {
#if defined(HAVE_CUDA)
        dev_type = DeviceType::CUDA;
#elif defined(HAVE_ROCM)
        dev_type = DeviceType::ROCm;
#endif
    }
}

// Create attention kernel via KernelFactory (ITensor* overload handles GPU/CPU dispatch)
auto kernel = llaminar::v2::kernels::KernelFactory::createAttention(params_.Q, dev_type);
```

### Step 3: Update Kernel Invocation for GPU

The kernel call needs to handle both CPU (`TensorBase*`) and GPU (`ITensor*` with device pointers).

**For GPU path**, the `CUDAFlashAttentionKernelT::compute()` takes raw `float*` pointers. Use `ITensor::raw_data()` which returns device pointers for GPU tensors:

Replace the kernel invocation block (around lines 175-195) with:

```cpp
bool success = false;

if (dev_type == DeviceType::CUDA || dev_type == DeviceType::ROCm)
{
    // GPU path: use raw device pointers via ITensor interface
    const float *Q_ptr = static_cast<const float *>(params_.Q->raw_data());
    const float *K_ptr = static_cast<const float *>(params_.K->raw_data());
    const float *V_ptr = static_cast<const float *>(params_.V->raw_data());
    float *output_ptr = static_cast<float *>(params_.output->raw_mutable_data());

    // GPU kernels use compute() with raw pointers
    success = kernel->compute(
        Q_ptr, K_ptr, V_ptr, output_ptr,
        params_.seq_len,
        params_.n_heads,
        params_.n_kv_heads,
        params_.head_dim,
        kernel_causal,
        params_.window_size,
        nullptr, // workspace_scores (Flash Attention manages internally)
        nullptr, // workspace_buffer
        nullptr, // workspace_context
        nullptr, // mask (TODO: support decode mask on GPU)
        false,   // use_bf16
        params_.mpi_ctx,
        device_idx);
}
else
{
    // CPU path: use existing TensorBase* logic
    auto *Q_base = dynamic_cast<TensorBase *>(params_.Q);
    auto *K_base = dynamic_cast<TensorBase *>(params_.K);
    auto *V_base = dynamic_cast<TensorBase *>(params_.V);
    auto *output_base = dynamic_cast<TensorBase *>(params_.output);

    if (!Q_base || !K_base || !V_base || !output_base)
    {
        LOG_ERROR("[AttentionComputeStage] CPU path requires TensorBase*");
        return false;
    }

    auto *workspace_scores_base = asTensorBase(params_.workspace_scores, "workspace_scores");

    success = kernel->compute_tensor(
        Q_base, K_base, V_base, output_base,
        params_.batch_size,
        params_.seq_len,
        effective_kv_len,
        params_.n_heads,
        params_.n_kv_heads,
        params_.head_dim,
        kernel_causal,
        params_.window_size,
        workspace_scores_base,
        mask_to_use,
        params_.mpi_ctx,
        device_idx);
}
```

### Step 4: Enable the Test

**File**: `tests/v2/integration/Test__CUDAFullModelInference.cpp`

Remove the `GTEST_SKIP()` on lines 436-443:

```cpp
// REMOVE THIS BLOCK:
GTEST_SKIP() << "GPU decode not yet supported - FusedAttentionWoStage requires TensorBase* "
             << "but GPU KV cache returns CUDATensorBase*";
```

---

## Key Files Reference

### Tensor Hierarchy

```
ITensor (interface)
├── TensorBase (CPU tensors)
│   ├── FP32Tensor
│   ├── BF16Tensor
│   ├── Q8_1Tensor
│   └── ... (all quantized types)
└── CUDATensorBase (GPU tensors)
    └── CUDATypedTensor<T>
```

### Device Detection on ITensor

```cpp
// ITensor methods (from src/v2/tensors/ITensor.h)
bool is_on_cpu() const { return home_device().is_cpu(); }
bool is_on_gpu() const { return home_device().is_gpu(); }
DeviceId home_device() const;  // Returns DeviceId::cpu() or DeviceId::cuda(n)

// Raw data access
const void* raw_data() const;        // Returns HOST ptr for CPU, DEVICE ptr for GPU
void* raw_mutable_data();            // Same but mutable
```

### CUDA Flash Attention Kernel

**File**: `src/v2/kernels/cuda/attention/CUDAFlashAttentionKernelT.h`

Key methods:
- `compute()` - Single sequence attention (takes raw `float*` device pointers)
- `compute_batch()` - Batched attention
- `compute_decode()` - Optimized for decode (seq_len=1 with large kv_len)

The kernel automatically selects:
- **Flash Attention 2** for prefill (seq_len > 1)
- **Flash Decoding** for decode (seq_len = 1)

---

## Testing Strategy

1. **Build Integration config** (has CUDA + assertions):
   ```bash
   cmake -B build_v2_integration -S src/v2 -DCMAKE_BUILD_TYPE=Integration -DHAVE_CUDA=ON
   cmake --build build_v2_integration --parallel
   ```

2. **Run CUDA integration tests**:
   ```bash
   ctest --test-dir build_v2_integration -R "CUDA" --output-on-failure
   ```

3. **Specifically run the full model inference test**:
   ```bash
   ./build_v2_integration/tests/v2/v2_integration_cuda_full_model_inference
   ```

4. **Expected Result**: `MultiTokenGeneration_MatchesCPU` should now pass (was skipped before)

---

## Potential Issues to Watch For

1. **Decode mask handling**: The CPU path builds explicit causal masks for decode. GPU path may need similar handling or rely on kernel's internal masking.

2. **kv_len vs seq_len**: Ensure `effective_kv_len` is correctly passed to GPU kernel. The decode case has `seq_len=1` but `kv_len` = full cache length.

3. **Memory layout**: GPU tensors use device pointers. Ensure no accidental host memory access.

4. **Precision mismatch**: Current implementation is FP32. The kernel templates support FP16/BF16 but need proper tensor type detection.

---

## Success Criteria

- [ ] `MultiTokenGeneration_MatchesCPU` test passes
- [ ] GPU decode generates same tokens as CPU (top-1 match)
- [ ] No memory errors (run with CUDA memcheck if available)
- [ ] All other CUDA tests still pass

---

## Related Documentation

- [CUDA_KERNELS_PROJECT_PLAN.md](../docs/v2/projects/2026-01/CUDA_KERNELS_PROJECT_PLAN.md) - Full project plan with Phase 4 blockers
- [copilot-instructions.md](../.github/copilot-instructions.md) - Build and test instructions
- [CUDAFlashAttentionKernelT.h](../src/v2/kernels/cuda/attention/CUDAFlashAttentionKernelT.h) - CUDA attention kernel interface

---

## Commands Quick Reference

```bash
# Build Integration config (CUDA + debug symbols + assertions)
cmake -B build_v2_integration -S src/v2 -DCMAKE_BUILD_TYPE=Integration -DHAVE_CUDA=ON
cmake --build build_v2_integration --parallel

# Run all CUDA tests
ctest --test-dir build_v2_integration -R "CUDA" --output-on-failure --parallel

# Run specific test with verbose output
LLAMINAR_LOG_LEVEL=DEBUG ./build_v2_integration/tests/v2/v2_integration_cuda_full_model_inference --gtest_filter=*MultiToken*

# Check for CUDA errors
CUDA_LAUNCH_BLOCKING=1 ./build_v2_integration/tests/v2/v2_integration_cuda_full_model_inference
```
