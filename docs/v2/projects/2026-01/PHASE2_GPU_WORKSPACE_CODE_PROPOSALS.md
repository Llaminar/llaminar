# Phase 2: GPU Workspace Buffer Management - Code Proposals

**Author:** David Sanftenberg  
**Created:** January 14, 2026  
**Status:** Proposal  
**Context:** Implementation plan for Phase 2 of GPU Workspace Buffer Management

This document contains complete, production-ready code proposals for modifying GPU kernels to receive workspace buffers instead of allocating internally.

---

## Table of Contents

1. [IWorkspaceConsumer.h - New Interface](#1-igpuworkspaceconsumerh---new-interface)
2. [ROCmQuantisedGemmKernel.h Modifications](#2-rocmquantisedgemmkernelh-modifications)
3. [ROCmQuantisedGemmKernel.cpp Modifications](#3-rocmquantisedgemmkernelcpp-modifications)
4. [CUDAQuantisedGemmKernel.h Modifications](#4-cudaquantisedgemmkernelh-modifications)
5. [CUDAQuantisedGemmKernel.cpp Modifications](#5-cudaquantisedgemmkernelcpp-modifications)
6. [Backward Compatibility Patterns](#6-backward-compatibility-patterns)

---

## 1. IWorkspaceConsumer.h - New Interface

**File:** `src/v2/interfaces/IWorkspaceConsumer.h`

```cpp
/**
 * @file IWorkspaceConsumer.h
 * @brief Interface for GPU kernels that consume centralized workspace buffers
 *
 * This interface enables kernels to declare their workspace requirements and
 * receive pre-allocated buffers from DeviceWorkspaceManager instead of managing
 * their own ad-hoc allocations.
 *
 * ## Design Goals
 *
 * 1. **Memory Budgeting**: Centralized allocation allows global VRAM budget control
 * 2. **Hot-Path Efficiency**: No allocations during GEMM execution
 * 3. **Backward Compatibility**: Kernels work in both legacy and managed modes
 *
 * ## Usage Pattern
 *
 * ```cpp
 * // At graph construction time:
 * auto requirements = kernel->getWorkspaceRequirements(max_m, max_n, max_k);
 * workspaceManager->allocate(requirements);
 *
 * // Bind workspace to kernel:
 * kernel->bindWorkspace(workspaceManager);
 *
 * // During execution (no allocations):
 * kernel->multiply_tensor(A, C, m, n, k);
 * ```
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "../execution/WorkspaceDescriptor.h"
#include <string>
#include <vector>

namespace llaminar2
{

// Forward declaration (defined in execution/DeviceWorkspaceManager.h)
class DeviceWorkspaceManager;

/**
 * @brief Interface for GPU kernels that consume centralized workspace buffers
 *
 * Kernels implementing this interface:
 * 1. Declare workspace requirements via getWorkspaceRequirements()
 * 2. Receive pre-allocated buffers via bindWorkspace()
 * 3. Use bound workspace instead of internal allocations when available
 * 4. Fall back to legacy internal buffers when workspace not bound
 */
class IWorkspaceConsumer
{
public:
    virtual ~IWorkspaceConsumer() = default;

    // =========================================================================
    // Workspace Requirements Declaration
    // =========================================================================

    /**
     * @brief Get workspace buffer requirements for this kernel
     *
     * Returns the set of buffers this kernel needs for execution at the given
     * dimensions. The caller allocates these buffers in DeviceWorkspaceManager
     * and then calls bindWorkspace().
     *
     * @param m Number of rows (batch size / sequence length)
     * @param n Number of output features (may be 0 if kernel-specific)
     * @param k Number of input features (may be 0 if kernel-specific)
     * @return WorkspaceRequirements describing all needed buffers
     *
     * @note Dimensions are typically the MAXIMUM expected values to avoid
     *       re-allocation during inference. For variable-length sequences,
     *       pass the maximum sequence length used in KV cache.
     *
     * @note If n and k are 0, kernel uses its internal N_ and K_ dimensions.
     */
    virtual WorkspaceRequirements getWorkspaceRequirements(
        int m, int n = 0, int k = 0) const = 0;

    // =========================================================================
    // Workspace Binding
    // =========================================================================

    /**
     * @brief Bind a workspace manager to this kernel
     *
     * After binding, the kernel uses buffers from the workspace manager
     * instead of its internal ad-hoc allocations. The workspace manager
     * must have allocated all buffers returned by getWorkspaceRequirements()
     * for the maximum expected dimensions.
     *
     * @param workspace Pointer to workspace manager (NOT owned, must outlive kernel)
     *                  Pass nullptr to unbind and return to legacy mode.
     *
     * @note Thread Safety: This method should only be called during setup,
     *       not during concurrent kernel execution.
     */
    virtual void bindWorkspace(DeviceWorkspaceManager* workspace) = 0;

    /**
     * @brief Unbind workspace and return to legacy mode
     *
     * Equivalent to bindWorkspace(nullptr). Kernel will use internal
     * buffer management after this call.
     */
    virtual void unbindWorkspace()
    {
        bindWorkspace(nullptr);
    }

    // =========================================================================
    // Workspace Query
    // =========================================================================

    /**
     * @brief Check if a workspace is currently bound
     *
     * @return true if bindWorkspace() was called with non-null workspace
     */
    virtual bool hasWorkspace() const = 0;

    /**
     * @brief Get the currently bound workspace manager
     *
     * @return Pointer to bound workspace manager, or nullptr if not bound
     */
    virtual DeviceWorkspaceManager* getWorkspace() const = 0;
};

// =============================================================================
// Standard Buffer Names for GEMM Kernels
// =============================================================================

/**
 * Standard buffer names used by GEMM workspace consumers.
 *
 * Using consistent names across CUDA/ROCm kernels enables the workspace
 * manager to share buffers between different kernel types on the same device.
 */
namespace GemmWorkspaceBuffers
{
    // INT8 quantization buffers
    constexpr const char* QUANT_A = "gemm_quant_a";       ///< [M × K] INT8 quantized activations
    constexpr const char* SCALES_A = "gemm_scales_a";    ///< [M] FP32 per-row activation scales
    constexpr const char* ACC_INT32 = "gemm_acc_int32";  ///< [M × N] INT32 accumulator

    // FP32 temporary buffers
    constexpr const char* TEMP_A_FP32 = "gemm_temp_a_fp32"; ///< [M × K] FP32 activation copy
    constexpr const char* TEMP_C_FP32 = "gemm_temp_c_fp32"; ///< [M × N] FP32 output copy

    // FP16 slab buffers (for slab-based FP16 GEMM)
    constexpr const char* SLAB_A_FP16 = "gemm_slab_a_fp16"; ///< [slab_m × slab_k] FP16 A slab
    constexpr const char* SLAB_B_FP16 = "gemm_slab_b_fp16"; ///< [slab_k × slab_n] FP16 B slab
    constexpr const char* SLAB_C_FP16 = "gemm_slab_c_fp16"; ///< [slab_m × slab_n] FP16 C slab

    // Full-matrix FP16 buffers (legacy path, deprecated in managed mode)
    constexpr const char* FULL_A_FP16 = "gemm_full_a_fp16"; ///< [M × K] FP16 full matrix
    constexpr const char* FULL_B_FP16 = "gemm_full_b_fp16"; ///< [K × N] FP16 full matrix
    constexpr const char* FULL_C_FP16 = "gemm_full_c_fp16"; ///< [M × N] FP16 full matrix
}

} // namespace llaminar2
```

---

## 2. ROCmQuantisedGemmKernel.h Modifications

**File:** `src/v2/kernels/rocm/ROCmQuantisedGemmKernel.h`

### 2.1 Add Include and Forward Declaration

Add after existing includes (around line 70):

```cpp
// EXISTING:
#include "../../tensors/TensorKernels.h"
#include "../../tensors/BlockStructures.h"
#include <memory>
#include <cstdint>
#include <vector>

// ADD THIS:
#include "../../interfaces/IWorkspaceConsumer.h"
```

### 2.2 Modify Class Declaration

Change the class declaration (around line 205) from:

```cpp
// BEFORE:
class ROCmQuantisedGemmKernel : public ITensorGemm
{
public:
```

To:

```cpp
// AFTER:
class ROCmQuantisedGemmKernel : public ITensorGemm, public IWorkspaceConsumer
{
public:
```

### 2.3 Add IWorkspaceConsumer Methods

Add after the existing public methods section (after `getKernelSnapshotInfo()` around line 435):

```cpp
            // =========================================================================
            // IWorkspaceConsumer interface
            // =========================================================================

            /**
             * @brief Get workspace buffer requirements for this kernel
             *
             * Returns requirements for INT8 quantization buffers and optional FP16
             * slab buffers. Buffer sizes depend on:
             * - m: sequence length (determines quantization buffer sizes)
             * - n: output features (uses N_ if not specified)
             * - k: input features (uses K_ if not specified)
             *
             * @param m Maximum expected sequence length
             * @param n Output features (0 = use kernel's N_)
             * @param k Input features (0 = use kernel's K_)
             * @return Workspace requirements for centralized allocation
             */
            WorkspaceRequirements getWorkspaceRequirements(
                int m, int n = 0, int k = 0) const override;

            /**
             * @brief Bind workspace manager for centralized buffer allocation
             *
             * After binding, multiply_tensor() uses workspace buffers instead of
             * internal Impl allocations. Pass nullptr to return to legacy mode.
             *
             * @param workspace Workspace manager (NOT owned, must outlive kernel)
             */
            void bindWorkspace(DeviceWorkspaceManager* workspace) override;

            /**
             * @brief Check if workspace is currently bound
             */
            bool hasWorkspace() const override;

            /**
             * @brief Get bound workspace manager
             */
            DeviceWorkspaceManager* getWorkspace() const override;
```

### 2.4 Add Private Member

Add to the private section (around line 525, after `std::unique_ptr<Impl> impl_`):

```cpp
            // PIMPL for CK implementation (avoids CK headers in this header)
            struct Impl;
            std::unique_ptr<Impl> impl_;

            // ADD THIS:
            // Workspace management (Phase 2)
            DeviceWorkspaceManager* workspace_ = nullptr;  ///< Bound workspace (NOT owned)
```

---

## 3. ROCmQuantisedGemmKernel.cpp Modifications

**File:** `src/v2/kernels/rocm/ROCmQuantisedGemmKernel.cpp`

### 3.1 Add Include

Add at the top with other includes (around line 55):

```cpp
// EXISTING:
#include "ROCmQuantisedGemmKernel.h"
#include "backends/ComputeBackend.h"
#include "backends/DeviceId.h"
#include "tensors/Tensors.h"
#include "tensors/BlockStructures.h"
#include "tensors/KernelSnapshotInfo.h"
#include "utils/Logger.h"

// ADD THIS:
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
```

### 3.2 Implement IWorkspaceConsumer Methods

Add after `getKernelSnapshotInfo()` implementation (around line 980):

```cpp
        KernelSnapshotInfo ROCmQuantisedGemmKernel::getKernelSnapshotInfo() const
        {
            // ... existing implementation ...
        }

        // =====================================================================
        // IWorkspaceConsumer Implementation
        // =====================================================================

        WorkspaceRequirements ROCmQuantisedGemmKernel::getWorkspaceRequirements(
            int m, int n_arg, int k_arg) const
        {
            // Use kernel's dimensions if not specified
            const int n = (n_arg > 0) ? n_arg : static_cast<int>(N_);
            const int k = (k_arg > 0) ? k_arg : static_cast<int>(K_);

            WorkspaceRequirements requirements;

            // INT8 quantization buffers (required for all paths)
            requirements.buffers.push_back({
                GemmWorkspaceBuffers::QUANT_A,
                static_cast<size_t>(m) * k * sizeof(int8_t),
                256,  // alignment
                true  // required
            });

            requirements.buffers.push_back({
                GemmWorkspaceBuffers::SCALES_A,
                static_cast<size_t>(m) * sizeof(float),
                256,
                true
            });

            requirements.buffers.push_back({
                GemmWorkspaceBuffers::ACC_INT32,
                static_cast<size_t>(m) * n * sizeof(int32_t),
                256,
                true
            });

            // FP32 temporary buffers
            requirements.buffers.push_back({
                GemmWorkspaceBuffers::TEMP_A_FP32,
                static_cast<size_t>(m) * k * sizeof(float),
                256,
                true
            });

            requirements.buffers.push_back({
                GemmWorkspaceBuffers::TEMP_C_FP32,
                static_cast<size_t>(m) * n * sizeof(float),
                256,
                true
            });

            // FP16 buffers (for FP16 hipBLAS path on gfx906)
            // Only required if M > FP16_THRESHOLD_M (128)
            requirements.buffers.push_back({
                GemmWorkspaceBuffers::FULL_A_FP16,
                static_cast<size_t>(m) * k * sizeof(uint16_t),  // __half = 2 bytes
                256,
                false  // optional - only used for FP16 path
            });

            requirements.buffers.push_back({
                GemmWorkspaceBuffers::FULL_B_FP16,
                static_cast<size_t>(k) * n * sizeof(uint16_t),
                256,
                false
            });

            requirements.buffers.push_back({
                GemmWorkspaceBuffers::FULL_C_FP16,
                static_cast<size_t>(m) * n * sizeof(uint16_t),
                256,
                false
            });

            LOG_DEBUG("[ROCmQuantisedGemmKernel::getWorkspaceRequirements] "
                      << "m=" << m << " n=" << n << " k=" << k
                      << " total=" << requirements.total_bytes() << " bytes");

            return requirements;
        }

        void ROCmQuantisedGemmKernel::bindWorkspace(DeviceWorkspaceManager* workspace)
        {
            workspace_ = workspace;

            if (workspace_)
            {
                LOG_DEBUG("[ROCmQuantisedGemmKernel::bindWorkspace] Bound workspace manager"
                          << " (device=" << workspace_->device().device_index
                          << ", budget=" << workspace_->budgetBytes() << " bytes)");
            }
            else
            {
                LOG_DEBUG("[ROCmQuantisedGemmKernel::bindWorkspace] Unbound workspace, using legacy mode");
            }
        }

        bool ROCmQuantisedGemmKernel::hasWorkspace() const
        {
            return workspace_ != nullptr;
        }

        DeviceWorkspaceManager* ROCmQuantisedGemmKernel::getWorkspace() const
        {
            return workspace_;
        }
```

### 3.3 Modify multiply_tensor() for Dual-Mode Operation

Modify the `multiply_tensor()` method (around line 550-650) to use workspace when available.

**Before** (current implementation snippet):

```cpp
            // Ensure work buffers are allocated
            ensureWorkBuffers(m);

            int8_t *d_A_int8 = impl_->d_A_int8;
            float *d_scales_A = impl_->d_scales_A;
            int32_t *d_C_int32 = impl_->d_C_int32;
```

**After** (modified for dual-mode):

```cpp
            // =========================================================================
            // Buffer Acquisition: Managed vs Legacy Mode
            // =========================================================================
            int8_t *d_A_int8 = nullptr;
            float *d_scales_A = nullptr;
            int32_t *d_C_int32 = nullptr;
            float *d_A_fp32 = nullptr;
            float *d_C_fp32 = nullptr;

            if (workspace_)
            {
                // MANAGED MODE: Use pre-allocated workspace buffers
                d_A_int8 = static_cast<int8_t*>(workspace_->getBuffer(GemmWorkspaceBuffers::QUANT_A));
                d_scales_A = static_cast<float*>(workspace_->getBuffer(GemmWorkspaceBuffers::SCALES_A));
                d_C_int32 = static_cast<int32_t*>(workspace_->getBuffer(GemmWorkspaceBuffers::ACC_INT32));
                d_A_fp32 = static_cast<float*>(workspace_->getBuffer(GemmWorkspaceBuffers::TEMP_A_FP32));
                d_C_fp32 = static_cast<float*>(workspace_->getBuffer(GemmWorkspaceBuffers::TEMP_C_FP32));

                // Validate all required buffers are available
                if (!d_A_int8 || !d_scales_A || !d_C_int32 || !d_A_fp32 || !d_C_fp32)
                {
                    LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] Workspace missing required buffers, "
                              "falling back to legacy mode");
                    workspace_ = nullptr;  // Fall through to legacy path
                }
                else
                {
                    LOG_TRACE("[ROCmQuantisedGemmKernel::multiply_tensor] Using managed workspace buffers");
                }
            }

            if (!workspace_)
            {
                // LEGACY MODE: Use internal Impl buffers (allocate if needed)
                ensureWorkBuffers(m);

                d_A_int8 = impl_->d_A_int8;
                d_scales_A = impl_->d_scales_A;
                d_C_int32 = impl_->d_C_int32;

                // Ensure cached d_A_fp32 buffer is large enough (reuse across calls)
                const size_t a_fp32_size = static_cast<size_t>(m) * k;
                if (a_fp32_size > impl_->d_A_fp32_capacity)
                {
                    if (impl_->d_A_fp32)
                        rocmQuantGemm_freeDevice(impl_->d_A_fp32);
                    impl_->d_A_fp32 = nullptr;
                    impl_->d_A_fp32_capacity = 0;

                    if (!rocmQuantGemm_allocFloat(&impl_->d_A_fp32, a_fp32_size, rocm_device_id_))
                    {
                        LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] Failed to allocate activation buffer");
                        return false;
                    }
                    impl_->d_A_fp32_capacity = a_fp32_size;
                }
                d_A_fp32 = impl_->d_A_fp32;

                // Ensure cached d_C_fp32 buffer is large enough
                const size_t c_fp32_size = static_cast<size_t>(m) * n;
                if (c_fp32_size > impl_->d_C_fp32_capacity)
                {
                    if (impl_->d_C_fp32)
                        rocmQuantGemm_freeDevice(impl_->d_C_fp32);
                    impl_->d_C_fp32 = nullptr;
                    impl_->d_C_fp32_capacity = 0;

                    if (!rocmQuantGemm_allocFloat(&impl_->d_C_fp32, c_fp32_size, rocm_device_id_))
                    {
                        LOG_ERROR("[ROCmQuantisedGemmKernel::multiply_tensor] Failed to allocate output buffer");
                        return false;
                    }
                    impl_->d_C_fp32_capacity = c_fp32_size;
                }
                d_C_fp32 = impl_->d_C_fp32;

                LOG_TRACE("[ROCmQuantisedGemmKernel::multiply_tensor] Using legacy Impl buffers");
            }

            LOG_TRACE("[ROCmQuantisedGemmKernel::multiply_tensor] Work buffers: A_int8=" << (void*)d_A_int8
                      << " scales_A=" << (void*)d_scales_A << " C_int32=" << (void*)d_C_int32
                      << " A_fp32=" << (void*)d_A_fp32 << " C_fp32=" << (void*)d_C_fp32);
```

### 3.4 Modify FP16 Path for Managed Mode

Update the FP16 path section (around line 780-850) to use workspace buffers when available:

**Before** (current FP16 buffer allocation):

```cpp
            case GemmPath::FP16_HIPBLAS:
            {
                // Ensure FP16 workspace buffers are large enough (cached for reuse)
                const size_t a_fp16_size = static_cast<size_t>(m) * k;
                const size_t b_fp16_size = static_cast<size_t>(k) * n;
                const size_t c_fp16_size = static_cast<size_t>(m) * n;

                if (a_fp16_size > impl_->d_A_fp16_capacity)
                {
                    // ... allocation code ...
                }
```

**After** (dual-mode FP16 buffer acquisition):

```cpp
            case GemmPath::FP16_HIPBLAS:
            {
                LOG_DEBUG("[ROCmQuantisedGemmKernel] Using FP16 hipBLAS (M=" << m << ", N=" << n << ", K=" << k << ")");

                const size_t a_fp16_size = static_cast<size_t>(m) * k;
                const size_t b_fp16_size = static_cast<size_t>(k) * n;
                const size_t c_fp16_size = static_cast<size_t>(m) * n;

                void *d_A_fp16 = nullptr;
                void *d_B_fp16 = nullptr;
                void *d_C_fp16 = nullptr;

                if (workspace_)
                {
                    // MANAGED MODE: Try to get FP16 buffers from workspace
                    d_A_fp16 = workspace_->getBuffer(GemmWorkspaceBuffers::FULL_A_FP16);
                    d_B_fp16 = workspace_->getBuffer(GemmWorkspaceBuffers::FULL_B_FP16);
                    d_C_fp16 = workspace_->getBuffer(GemmWorkspaceBuffers::FULL_C_FP16);

                    // Validate workspace has sufficient capacity
                    if (d_A_fp16 && d_B_fp16 && d_C_fp16)
                    {
                        size_t avail_a = workspace_->getBufferSize(GemmWorkspaceBuffers::FULL_A_FP16);
                        size_t avail_b = workspace_->getBufferSize(GemmWorkspaceBuffers::FULL_B_FP16);
                        size_t avail_c = workspace_->getBufferSize(GemmWorkspaceBuffers::FULL_C_FP16);

                        if (avail_a < a_fp16_size * 2 ||
                            avail_b < b_fp16_size * 2 ||
                            avail_c < c_fp16_size * 2)
                        {
                            LOG_WARN("[ROCmQuantisedGemmKernel] Workspace FP16 buffers too small, "
                                     "falling back to legacy allocation");
                            d_A_fp16 = d_B_fp16 = d_C_fp16 = nullptr;
                        }
                        else
                        {
                            LOG_TRACE("[ROCmQuantisedGemmKernel] Using workspace FP16 buffers");
                        }
                    }
                    else
                    {
                        LOG_DEBUG("[ROCmQuantisedGemmKernel] FP16 buffers not in workspace, using legacy");
                    }
                }

                if (!d_A_fp16 || !d_B_fp16 || !d_C_fp16)
                {
                    // LEGACY MODE: Use/allocate internal Impl FP16 buffers
                    if (a_fp16_size > impl_->d_A_fp16_capacity)
                    {
                        if (impl_->d_A_fp16)
                            rocmQuantGemm_freeDevice(impl_->d_A_fp16);
                        impl_->d_A_fp16 = nullptr;
                        impl_->d_A_fp16_capacity = 0;

                        if (!rocmQuantGemm_allocFP16(&impl_->d_A_fp16, a_fp16_size, rocm_device_id_))
                        {
                            LOG_ERROR("[ROCmQuantisedGemmKernel] Failed to allocate FP16 A buffer");
                            return false;
                        }
                        impl_->d_A_fp16_capacity = a_fp16_size;
                    }

                    if (b_fp16_size > impl_->d_B_fp16_capacity)
                    {
                        if (impl_->d_B_fp16)
                            rocmQuantGemm_freeDevice(impl_->d_B_fp16);
                        impl_->d_B_fp16 = nullptr;
                        impl_->d_B_fp16_capacity = 0;

                        if (!rocmQuantGemm_allocFP16(&impl_->d_B_fp16, b_fp16_size, rocm_device_id_))
                        {
                            LOG_ERROR("[ROCmQuantisedGemmKernel] Failed to allocate FP16 B buffer");
                            return false;
                        }
                        impl_->d_B_fp16_capacity = b_fp16_size;
                    }

                    if (c_fp16_size > impl_->d_C_fp16_capacity)
                    {
                        if (impl_->d_C_fp16)
                            rocmQuantGemm_freeDevice(impl_->d_C_fp16);
                        impl_->d_C_fp16 = nullptr;
                        impl_->d_C_fp16_capacity = 0;

                        if (!rocmQuantGemm_allocFP16(&impl_->d_C_fp16, c_fp16_size, rocm_device_id_))
                        {
                            LOG_ERROR("[ROCmQuantisedGemmKernel] Failed to allocate FP16 C buffer");
                            return false;
                        }
                        impl_->d_C_fp16_capacity = c_fp16_size;
                    }

                    d_A_fp16 = impl_->d_A_fp16;
                    d_B_fp16 = impl_->d_B_fp16;
                    d_C_fp16 = impl_->d_C_fp16;
                }

                // Execute with resolved buffers (no allocations in this path)
                success = rocmQuantGemm_executeFP16_cached(
                    d_A_int8, d_weights_int8, d_C_fp32,
                    d_scales_A, d_scales_B,
                    d_A_fp16, d_B_fp16, d_C_fp16,
                    m, n, k, rocm_device_id_, nullptr);

                // ... rest of FP16 path with fallback ...
                break;
            }
```

---

## 4. CUDAQuantisedGemmKernel.h Modifications

**File:** `src/v2/kernels/cuda/CUDAQuantisedGemmKernel.h`

### 4.1 Add Include

Add after existing includes (around line 37):

```cpp
// EXISTING:
#include "../../tensors/TensorKernels.h"
#include "../../tensors/BlockStructures.h"
#include <memory>
#include <cstdint>
#include <vector>

// ADD THIS:
#include "../../interfaces/IWorkspaceConsumer.h"
```

### 4.2 Modify Class Declaration

Change the class declaration (around line 102) from:

```cpp
// BEFORE:
class CUDAQuantisedGemmKernel : public ITensorGemm
{
public:
```

To:

```cpp
// AFTER:
class CUDAQuantisedGemmKernel : public ITensorGemm, public IWorkspaceConsumer
{
public:
```

### 4.3 Add IWorkspaceConsumer Methods

Add after the `IKernelSnapshotCapable` section (around line 300):

```cpp
            // =========================================================================
            // IKernelSnapshotCapable interface
            // =========================================================================

            KernelSnapshotInfo getKernelSnapshotInfo() const override;

            // =========================================================================
            // IWorkspaceConsumer interface (NEW)
            // =========================================================================

            /**
             * @brief Get workspace buffer requirements for this kernel
             *
             * Returns requirements for INT8 quantization buffers.
             * CUDA kernel uses CUTLASS which has different workspace patterns than ROCm.
             *
             * @param m Maximum expected sequence length
             * @param n Output features (0 = use kernel's N_)
             * @param k Input features (0 = use kernel's K_)
             * @return Workspace requirements for centralized allocation
             */
            WorkspaceRequirements getWorkspaceRequirements(
                int m, int n = 0, int k = 0) const override;

            /**
             * @brief Bind workspace manager for centralized buffer allocation
             *
             * @param workspace Workspace manager (NOT owned, must outlive kernel)
             */
            void bindWorkspace(DeviceWorkspaceManager* workspace) override;

            /**
             * @brief Check if workspace is currently bound
             */
            bool hasWorkspace() const override;

            /**
             * @brief Get bound workspace manager
             */
            DeviceWorkspaceManager* getWorkspace() const override;
```

### 4.4 Add Private Member

Add to the private section (around line 395, after `std::unique_ptr<Impl> impl_`):

```cpp
            // PIMPL for CUTLASS implementation (avoids CUTLASS in header)
            struct Impl;
            std::unique_ptr<Impl> impl_;

            // ADD THIS:
            // Workspace management (Phase 2)
            DeviceWorkspaceManager* workspace_ = nullptr;  ///< Bound workspace (NOT owned)
```

---

## 5. CUDAQuantisedGemmKernel.cpp Modifications

**File:** `src/v2/kernels/cuda/CUDAQuantisedGemmKernel.cpp`

### 5.1 Add Include

Add at the top with other includes (around line 35):

```cpp
// EXISTING:
#include "CUDAQuantisedGemmKernel.h"
#include "backends/ComputeBackend.h"
#include "backends/DeviceId.h"
#include "tensors/Tensors.h"
#include "tensors/BlockStructures.h"
#include "tensors/KernelSnapshotInfo.h"
#include "utils/Logger.h"

// ADD THIS:
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
```

### 5.2 Implement IWorkspaceConsumer Methods

Add after `getKernelSnapshotInfo()` implementation (find appropriate location):

```cpp
        // =====================================================================
        // IWorkspaceConsumer Implementation
        // =====================================================================

        WorkspaceRequirements CUDAQuantisedGemmKernel::getWorkspaceRequirements(
            int m, int n_arg, int k_arg) const
        {
            // Use kernel's dimensions if not specified
            const int n = (n_arg > 0) ? n_arg : static_cast<int>(N_);
            const int k = (k_arg > 0) ? k_arg : static_cast<int>(K_);

            WorkspaceRequirements requirements;

            // INT8 quantization buffers (required for all CUDA GEMM paths)
            requirements.buffers.push_back({
                GemmWorkspaceBuffers::QUANT_A,
                static_cast<size_t>(m) * k * sizeof(int8_t),
                256,  // alignment for CUTLASS
                true  // required
            });

            requirements.buffers.push_back({
                GemmWorkspaceBuffers::SCALES_A,
                static_cast<size_t>(m) * sizeof(float),
                256,
                true
            });

            requirements.buffers.push_back({
                GemmWorkspaceBuffers::ACC_INT32,
                static_cast<size_t>(m) * n * sizeof(int32_t),
                256,
                true
            });

            LOG_DEBUG("[CUDAQuantisedGemmKernel::getWorkspaceRequirements] "
                      << "m=" << m << " n=" << n << " k=" << k
                      << " total=" << requirements.total_bytes() << " bytes");

            return requirements;
        }

        void CUDAQuantisedGemmKernel::bindWorkspace(DeviceWorkspaceManager* workspace)
        {
            workspace_ = workspace;

            if (workspace_)
            {
                LOG_DEBUG("[CUDAQuantisedGemmKernel::bindWorkspace] Bound workspace manager"
                          << " (device=" << workspace_->device().device_index
                          << ", budget=" << workspace_->budgetBytes() << " bytes)");
            }
            else
            {
                LOG_DEBUG("[CUDAQuantisedGemmKernel::bindWorkspace] Unbound workspace, using legacy mode");
            }
        }

        bool CUDAQuantisedGemmKernel::hasWorkspace() const
        {
            return workspace_ != nullptr;
        }

        DeviceWorkspaceManager* CUDAQuantisedGemmKernel::getWorkspace() const
        {
            return workspace_;
        }
```

### 5.3 Modify ensureWorkBuffers() for Managed Mode

Modify the `ensureWorkBuffers()` method (around line 470) to check workspace first:

**Before:**

```cpp
        void CUDAQuantisedGemmKernel::ensureWorkBuffers(int m)
        {
            if (m <= impl_->work_buffer_M)
            {
                return; // Already have enough capacity
            }

            if (!cudaQuantGemm_ensureWorkBuffers(
                    &impl_->d_A_int8,
                    &impl_->d_scales_A,
                    &impl_->d_C_int32,
                    &impl_->work_buffer_M,
                    m,
                    static_cast<int>(K_),
                    static_cast<int>(N_),
                    cuda_device_id_))
            {
                throw std::runtime_error("[CUDAQuantisedGemmKernel] Failed to allocate work buffers");
            }
        }
```

**After:**

```cpp
        void CUDAQuantisedGemmKernel::ensureWorkBuffers(int m)
        {
            // In managed mode, this method becomes a no-op
            // Workspace buffers are acquired directly in multiply_tensor()
            if (workspace_)
            {
                LOG_TRACE("[CUDAQuantisedGemmKernel::ensureWorkBuffers] Skipped - using workspace manager");
                return;
            }

            // LEGACY MODE: Allocate internal buffers if needed
            if (m <= impl_->work_buffer_M)
            {
                return; // Already have enough capacity
            }

            if (!cudaQuantGemm_ensureWorkBuffers(
                    &impl_->d_A_int8,
                    &impl_->d_scales_A,
                    &impl_->d_C_int32,
                    &impl_->work_buffer_M,
                    m,
                    static_cast<int>(K_),
                    static_cast<int>(N_),
                    cuda_device_id_))
            {
                throw std::runtime_error("[CUDAQuantisedGemmKernel] Failed to allocate work buffers");
            }
        }
```

### 5.4 Modify multiply_fp32_to_fp32() for Dual-Mode

The CUDA kernel's `multiply_fp32_to_fp32()` method (which is called from `multiply_tensor()`) should acquire buffers from workspace when available. Find the method and modify the buffer acquisition pattern:

```cpp
        bool CUDAQuantisedGemmKernel::multiply_fp32_to_fp32(
            const float *d_A, float *d_C,
            int m, int n, int k,
            float alpha, float beta)
        {
            // =========================================================================
            // Buffer Acquisition: Managed vs Legacy Mode
            // =========================================================================
            int8_t *d_A_int8 = nullptr;
            float *d_scales_A = nullptr;
            int32_t *d_C_int32 = nullptr;

            if (workspace_)
            {
                // MANAGED MODE: Use pre-allocated workspace buffers
                d_A_int8 = static_cast<int8_t*>(workspace_->getBuffer(GemmWorkspaceBuffers::QUANT_A));
                d_scales_A = static_cast<float*>(workspace_->getBuffer(GemmWorkspaceBuffers::SCALES_A));
                d_C_int32 = static_cast<int32_t*>(workspace_->getBuffer(GemmWorkspaceBuffers::ACC_INT32));

                if (!d_A_int8 || !d_scales_A || !d_C_int32)
                {
                    LOG_WARN("[CUDAQuantisedGemmKernel::multiply_fp32_to_fp32] "
                             "Workspace missing buffers, falling back to legacy");
                }
                else
                {
                    LOG_TRACE("[CUDAQuantisedGemmKernel::multiply_fp32_to_fp32] Using workspace buffers");
                }
            }

            if (!d_A_int8 || !d_scales_A || !d_C_int32)
            {
                // LEGACY MODE: Use internal Impl buffers
                ensureWorkBuffers(m);
                d_A_int8 = impl_->d_A_int8;
                d_scales_A = impl_->d_scales_A;
                d_C_int32 = impl_->d_C_int32;
            }

            // ... rest of implementation using resolved buffers ...
```

---

## 6. Backward Compatibility Patterns

### 6.1 General Pattern for Dual-Mode Buffer Acquisition

```cpp
/**
 * Pattern: Dual-mode buffer acquisition
 *
 * Used at the start of any method that needs workspace buffers.
 * Provides seamless fallback from managed to legacy mode.
 */
void* d_buffer = nullptr;

if (workspace_)
{
    // MANAGED MODE: Try workspace first
    d_buffer = workspace_->getBuffer("buffer_name");
    
    if (!d_buffer)
    {
        // Buffer not in workspace - log and fall back
        LOG_WARN("[Kernel] Buffer 'buffer_name' not in workspace, using legacy mode");
    }
    else
    {
        // Optionally validate size
        size_t available = workspace_->getBufferSize("buffer_name");
        if (available < required_size)
        {
            LOG_WARN("[Kernel] Workspace buffer too small, using legacy mode");
            d_buffer = nullptr;
        }
    }
}

if (!d_buffer)
{
    // LEGACY MODE: Use internal buffers (allocate if needed)
    ensureInternalBuffers(required_size);
    d_buffer = impl_->internal_buffer;
}
```

### 6.2 Complete Kernel Execution Pattern

```cpp
bool GpuKernel::execute(const TensorBase* input, TensorBase* output, int m, int n, int k)
{
    // =========================================================================
    // 1. Acquire all buffers upfront (managed or legacy)
    // =========================================================================
    
    void* buf_a = nullptr;
    void* buf_b = nullptr;
    void* buf_c = nullptr;
    bool using_workspace = false;
    
    if (workspace_)
    {
        buf_a = workspace_->getBuffer(GemmWorkspaceBuffers::QUANT_A);
        buf_b = workspace_->getBuffer(GemmWorkspaceBuffers::SCALES_A);
        buf_c = workspace_->getBuffer(GemmWorkspaceBuffers::ACC_INT32);
        
        using_workspace = (buf_a && buf_b && buf_c);
        
        if (!using_workspace)
        {
            LOG_WARN("[GpuKernel::execute] Incomplete workspace, falling back to legacy");
        }
    }
    
    if (!using_workspace)
    {
        // Legacy path: allocate internal buffers
        ensureInternalBuffers(m, n, k);
        buf_a = impl_->d_A_int8;
        buf_b = impl_->d_scales_A;
        buf_c = impl_->d_C_int32;
    }
    
    // =========================================================================
    // 2. Execute kernel (identical code path for both modes)
    // =========================================================================
    
    // ... kernel execution using buf_a, buf_b, buf_c ...
    
    return true;
}
```

### 6.3 Test Pattern for Dual-Mode Verification

```cpp
// Test: Verify kernel works in both modes with identical results

TEST(Test__GpuKernel, DualModeProducesIdenticalResults)
{
    auto kernel = createKernel(weights, device_id);
    auto input = createTestInput(m, k);
    auto output_legacy = createOutput(m, n);
    auto output_managed = createOutput(m, n);
    
    // Run in LEGACY mode (no workspace bound)
    ASSERT_FALSE(kernel->hasWorkspace());
    ASSERT_TRUE(kernel->multiply_tensor(input.get(), output_legacy.get(), m, n, k));
    
    // Create and bind workspace
    auto workspace = std::make_unique<DeviceWorkspaceManager>(device, budget);
    auto requirements = kernel->getWorkspaceRequirements(m, n, k);
    ASSERT_TRUE(workspace->allocate(requirements));
    kernel->bindWorkspace(workspace.get());
    
    // Run in MANAGED mode
    ASSERT_TRUE(kernel->hasWorkspace());
    ASSERT_TRUE(kernel->multiply_tensor(input.get(), output_managed.get(), m, n, k));
    
    // Verify results are identical (bitwise or within tolerance)
    EXPECT_TRUE(tensorsEqual(output_legacy.get(), output_managed.get()));
    
    // Verify unbinding returns to legacy mode
    kernel->unbindWorkspace();
    ASSERT_FALSE(kernel->hasWorkspace());
}
```

---

## Summary

This document provides complete code proposals for Phase 2 of GPU Workspace Buffer Management:

| Deliverable | Status | Notes |
|-------------|--------|-------|
| `IWorkspaceConsumer.h` | Complete | New interface header |
| `ROCmQuantisedGemmKernel.h` | Complete | Add interface inheritance + methods |
| `ROCmQuantisedGemmKernel.cpp` | Complete | Implement interface + dual-mode GEMM |
| `CUDAQuantisedGemmKernel.h` | Complete | Add interface inheritance + methods |
| `CUDAQuantisedGemmKernel.cpp` | Complete | Implement interface + dual-mode GEMM |
| Backward Compatibility | Complete | Pattern documentation + test example |

**Key Design Decisions:**

1. **Non-owning workspace pointer**: Kernel stores `DeviceWorkspaceManager*` but doesn't own it
2. **Graceful fallback**: If workspace missing buffers, fall back to legacy without error
3. **Standard buffer names**: Shared constants in `GemmWorkspaceBuffers` namespace
4. **Minimal interface**: Only 4 methods in `IWorkspaceConsumer`
5. **Buffer validation**: Check both presence and size before using workspace buffers

**Dependencies:**

- Phase 1 must be complete (`DeviceWorkspaceManager`, `WorkspaceDescriptor`, `WorkspaceRequirements`)
- Existing kernel functionality must remain unchanged when workspace not bound
