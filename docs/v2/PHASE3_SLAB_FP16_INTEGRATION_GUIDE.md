/**
 * @file ROCmQuantisedGemmKernel_SlabFP16_Integration.md
 * @brief Proposed modifications to integrate slab-based FP16 GEMM
 *
 * This document shows the required changes to ROCmQuantisedGemmKernel.cpp
 * to integrate the slab-based FP16 GEMM path.
 */

# Integration Changes for ROCmQuantisedGemmKernel.cpp

## 1. Add Include for Slab FP16 Header

```cpp
// At the top of ROCmQuantisedGemmKernel.cpp, add:
#include "ROCmQuantisedGemmKernel_SlabFP16.h"
#include "kernels/SlabGemmConfig.h"
```

## 2. Add GemmPath Enum Value

```cpp
// In ROCmQuantisedGemmKernel.h or at the top of .cpp:
enum class GemmPath
{
    CK_FUSED,        // CK with fused scaling (legacy)
    CK_TWO_KERNEL,   // CK + separate scaling kernel
    HIPBLAS_INT8,    // hipBLAS INT8 GEMM
    FP16_HIPBLAS,    // Full FP16 conversion + hipBLAS hgemm
    FP16_SLAB        // NEW: Slab-based FP16 (memory-bounded)
};
```

## 3. Add Slab Workspace Buffers to Impl

```cpp
// In struct ROCmQuantisedGemmKernel::Impl, add:
struct Impl
{
    // ... existing members ...

    // Slab FP16 workspace buffers (for memory-bounded FP16 GEMM)
    void *d_slab_a_fp16 = nullptr;     // [slab_m × slab_k] FP16
    void *d_slab_b_fp16 = nullptr;     // [slab_k × slab_n] FP16
    void *d_slab_c_fp16 = nullptr;     // [slab_m × slab_n] FP16
    size_t slab_a_capacity = 0;        // Current allocation size (bytes)
    size_t slab_b_capacity = 0;
    size_t slab_c_capacity = 0;
    SlabGemmConfig current_slab_config; // Current slab configuration

    ~Impl()
    {
        // ... existing cleanup ...

        // Free slab workspace buffers
        if (d_slab_a_fp16)
            rocmQuantGemm_freeDevice(d_slab_a_fp16);
        if (d_slab_b_fp16)
            rocmQuantGemm_freeDevice(d_slab_b_fp16);
        if (d_slab_c_fp16)
            rocmQuantGemm_freeDevice(d_slab_c_fp16);
    }
};
```

## 4. Modify selectGemmPath() to Consider Slab Mode

```cpp
GemmPath selectGemmPath(int M, int N, int K, size_t workspace_budget = 0)
{
    // Environment variable overrides
    const char *force_hipblas_env = std::getenv("LLAMINAR_ROCM_GEMM_FORCE_HIPBLAS");
    const bool force_hipblas = (force_hipblas_env && force_hipblas_env[0] == '1');

    const char *force_fp16_env = std::getenv("LLAMINAR_ROCM_GEMM_FORCE_FP16");
    const bool force_fp16 = (force_fp16_env && force_fp16_env[0] == '1');

    const char *force_slab_env = std::getenv("LLAMINAR_ROCM_GEMM_FORCE_SLAB");
    const bool force_slab = (force_slab_env && force_slab_env[0] == '1');

    const char *disable_fp16_env = std::getenv("LLAMINAR_ROCM_GEMM_DISABLE_FP16");
    const bool disable_fp16 = (disable_fp16_env && disable_fp16_env[0] == '1');

    // Force overrides
    if (force_hipblas)
    {
        return GemmPath::HIPBLAS_INT8;
    }
    if (force_slab && !disable_fp16)
    {
        return GemmPath::FP16_SLAB;
    }
    if (force_fp16 && !disable_fp16)
    {
        return GemmPath::FP16_HIPBLAS;
    }

    // Check if CK supports these dimensions
    const bool ck_supported = rocmQuantGemm_areDimensionsSupported(M, N, K);

    if (!ck_supported)
    {
        return GemmPath::HIPBLAS_INT8;
    }

    // Heuristic selection for FP16 paths
    constexpr int FP16_THRESHOLD_M = 128;

    if (!disable_fp16 && M > FP16_THRESHOLD_M)
    {
        // Check if full FP16 fits in workspace budget
        if (workspace_budget > 0 && rocmQuantGemm_shouldUseSlabFP16(M, N, K, workspace_budget))
        {
            return GemmPath::FP16_SLAB;
        }
        return GemmPath::FP16_HIPBLAS;
    }

    return GemmPath::CK_TWO_KERNEL;
}
```

## 5. Add Slab FP16 Execution Path in multiply_tensor()

```cpp
// In multiply_tensor(), add new case for FP16_SLAB:

case GemmPath::FP16_SLAB:
{
    // Slab-based FP16: Memory-bounded chunked GEMM
    LOG_DEBUG("[ROCmQuantisedGemmKernel] Using SLAB FP16 (M=" << m << ", N=" << n << ", K=" << k << ")");

    // Get optimal slab configuration for current dimensions
    SlabGemmConfig slab_config;
    size_t workspace_budget = 64 * 1024 * 1024; // 64MB default
    
    // TODO: Get budget from DeviceWorkspaceManager when available
    // if (workspace_) {
    //     workspace_budget = workspace_->availableBytes();
    // }
    
    rocmQuantGemm_getSlabConfig(m, n, k, workspace_budget, &slab_config);

    // Ensure slab workspace buffers are allocated
    const size_t slab_a_bytes = slab_config.slabABytes(SlabDataType::FP16);
    const size_t slab_b_bytes = slab_config.slabBBytes(SlabDataType::FP16);
    const size_t slab_c_bytes = slab_config.slabCBytes(SlabDataType::FP16);

    if (slab_a_bytes > impl_->slab_a_capacity)
    {
        if (impl_->d_slab_a_fp16)
            rocmQuantGemm_freeDevice(impl_->d_slab_a_fp16);
        impl_->d_slab_a_fp16 = nullptr;
        impl_->slab_a_capacity = 0;

        // Allocate as bytes (FP16 = 2 bytes per element)
        size_t count = slab_a_bytes / 2;
        if (!rocmQuantGemm_allocFP16(&impl_->d_slab_a_fp16, count, rocm_device_id_))
        {
            LOG_ERROR("[ROCmQuantisedGemmKernel] Failed to allocate slab A workspace");
            return false;
        }
        impl_->slab_a_capacity = slab_a_bytes;
    }

    if (slab_b_bytes > impl_->slab_b_capacity)
    {
        if (impl_->d_slab_b_fp16)
            rocmQuantGemm_freeDevice(impl_->d_slab_b_fp16);
        impl_->d_slab_b_fp16 = nullptr;
        impl_->slab_b_capacity = 0;

        size_t count = slab_b_bytes / 2;
        if (!rocmQuantGemm_allocFP16(&impl_->d_slab_b_fp16, count, rocm_device_id_))
        {
            LOG_ERROR("[ROCmQuantisedGemmKernel] Failed to allocate slab B workspace");
            return false;
        }
        impl_->slab_b_capacity = slab_b_bytes;
    }

    if (slab_c_bytes > impl_->slab_c_capacity)
    {
        if (impl_->d_slab_c_fp16)
            rocmQuantGemm_freeDevice(impl_->d_slab_c_fp16);
        impl_->d_slab_c_fp16 = nullptr;
        impl_->slab_c_capacity = 0;

        size_t count = slab_c_bytes / 2;
        if (!rocmQuantGemm_allocFP16(&impl_->d_slab_c_fp16, count, rocm_device_id_))
        {
            LOG_ERROR("[ROCmQuantisedGemmKernel] Failed to allocate slab C workspace");
            return false;
        }
        impl_->slab_c_capacity = slab_c_bytes;
    }

    impl_->current_slab_config = slab_config;

    // Execute slab-based FP16 GEMM
    success = rocmQuantGemm_executeSlabFP16(
        d_A_int8, d_weights_int8, d_C_fp32,
        d_scales_A, d_scales_B,
        m, n, k,
        impl_->d_slab_a_fp16,
        impl_->d_slab_b_fp16,
        impl_->d_slab_c_fp16,
        &slab_config,
        rocm_device_id_,
        nullptr);

    if (!success)
    {
        // Fallback to full FP16 if slab fails
        LOG_WARN("[ROCmQuantisedGemmKernel] Slab FP16 failed, trying full FP16 fallback");
        
        // [existing FP16_HIPBLAS code here as fallback]
    }
    break;
}
```

## 6. Add CMakeLists.txt Entry

```cmake
# In src/v2/kernels/rocm/CMakeLists.txt, add:

if(HAVE_ROCM)
    set(ROCM_SOURCES
        ROCmQuantisedGemmKernel.cpp
        ROCmQuantisedGemmKernel_CK.hip
        ROCmQuantisedGemmKernel_FP16.hip
        ROCmQuantisedGemmKernel_SlabFP16.hip  # NEW
    )
    
    # ... rest of CMake config ...
endif()

# Also add SlabGemmConfig to CPU sources (it's not HIP-specific):
set(KERNEL_SOURCES
    KernelFactory.cpp
    SlabGemmConfig.cpp  # NEW
    # ...
)
```

## 7. Environment Variables Summary

| Variable | Description | Default |
|----------|-------------|---------|
| `LLAMINAR_ROCM_GEMM_FORCE_SLAB` | Force slab FP16 for all FP16-eligible GEMMs | 0 |
| `LLAMINAR_GPU_WORKSPACE_MB` | Workspace budget per GPU (MB) | 64 |
| `LLAMINAR_GPU_SLAB_M` | Override slab M dimension | auto |
| `LLAMINAR_GPU_SLAB_N` | Override slab N dimension | auto |
| `LLAMINAR_GPU_SLAB_K` | Override slab K dimension | auto |

## 8. Integration with DeviceWorkspaceManager (Future)

When Phase 1-2 are complete, the slab buffers should come from `DeviceWorkspaceManager`:

```cpp
// Future integration pattern:
case GemmPath::FP16_SLAB:
{
    if (workspace_)
    {
        // Use managed workspace buffers
        void* slab_a = workspace_->getBuffer("slab_a_fp16");
        void* slab_b = workspace_->getBuffer("slab_b_fp16");
        void* slab_c = workspace_->getBuffer("slab_c_fp16");
        
        success = rocmQuantGemm_executeSlabFP16(
            d_A_int8, d_weights_int8, d_C_fp32,
            d_scales_A, d_scales_B,
            m, n, k,
            slab_a, slab_b, slab_c,
            &impl_->current_slab_config,
            rocm_device_id_,
            nullptr);
    }
    else
    {
        // Fallback to internal allocation (current pattern)
        // ... allocate and execute as shown above ...
    }
}
```
