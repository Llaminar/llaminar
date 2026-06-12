/**
 * @file ComputeStageUtils.h
 * @brief Shared utilities for compute stage implementations
 */
#pragma once

#include "../../backends/DeviceId.h"
#include "../../execution/local_execution/device/WorkspaceDescriptor.h"
#include "../../interfaces/IWorkspaceConsumer.h"
#include "../../tensors/ITensor.h"
#include "../../tensors/Tensors.h"
#include "../../utils/Logger.h"
#include "../../utils/Assertions.h"

#include <algorithm>

namespace llaminar2
{

    // =========================================================================
    // ITensor to TensorBase casting utilities
    // =========================================================================
    // These helpers provide safe downcasting from device-agnostic ITensor*
    // to CPU-specific TensorBase*. Used during the transition period where
    // stage interfaces use ITensor* but implementations still need TensorBase*.

    /**
     * @brief Cast ITensor* to TensorBase* (mutable)
     * @param tensor ITensor pointer to cast
     * @param name Name for error messages (optional)
     * @return TensorBase pointer, or nullptr if cast fails
     */
    inline TensorBase *asTensorBase(ITensor *tensor, const char *name = nullptr)
    {
        if (!tensor)
            return nullptr;
        auto *base = dynamic_cast<TensorBase *>(tensor);
        if (!base && name)
        {
            LOG_WARN("asTensorBase: " << name << " is not a TensorBase (likely GPU tensor)");
        }
        return base;
    }

    /**
     * @brief Cast const ITensor* to const TensorBase*
     * @param tensor ITensor pointer to cast
     * @param name Name for error messages (optional)
     * @return const TensorBase pointer, or nullptr if cast fails
     */
    inline const TensorBase *asTensorBase(const ITensor *tensor, const char *name = nullptr)
    {
        if (!tensor)
            return nullptr;
        auto *base = dynamic_cast<const TensorBase *>(tensor);
        if (!base && name)
        {
            LOG_WARN("asTensorBase: " << name << " is not a TensorBase (likely GPU tensor)");
        }
        return base;
    }

    /**
     * @brief Cast ITensor* to TensorBase* with assertion
     * @param tensor ITensor pointer to cast
     * @param name Name for error messages
     * @return TensorBase pointer (never nullptr if input is non-null)
     * @throws Assertion failure if cast fails
     */
    inline TensorBase *requireTensorBase(ITensor *tensor, const char *name)
    {
        if (!tensor)
            return nullptr;
        auto *base = dynamic_cast<TensorBase *>(tensor);
        LLAMINAR_ASSERTF(base, name << " must derive from TensorBase");
        return base;
    }

    /**
     * @brief Cast const ITensor* to const TensorBase* with assertion
     * @param tensor ITensor pointer to cast
     * @param name Name for error messages
     * @return const TensorBase pointer (never nullptr if input is non-null)
     * @throws Assertion failure if cast fails
     */
    inline const TensorBase *requireTensorBase(const ITensor *tensor, const char *name)
    {
        if (!tensor)
            return nullptr;
        auto *base = dynamic_cast<const TensorBase *>(tensor);
        LLAMINAR_ASSERTF(base, name << " must derive from TensorBase");
        return base;
    }

    /**
     * @brief Add CUDA decode side-stream GEMV partials for fused projection stages.
     *
     * CUDA NativeVNNI decode can launch several M=1 projections from one fused stage
     * on separate streams. Slot 0 uses the normal `GEMV_KPAR_PARTIALS` buffer; the
     * remaining projections need one disjoint side-stream slot each. Single-output
     * GEMV stages such as LM head cannot consume these slots, so the declaration
     * belongs at the fused-stage layer where the projection fan-out is known.
     *
     * @param reqs Merged per-projection workspace requirements to augment.
     * @param device Stage device; only CUDA receives this CUDA-specific buffer.
     * @param m Declared row count. Only exact decode M=1 needs side-stream slots.
     * @param projection_count Number of fused projections the stage may launch.
     * @param max_concurrent_streams Maximum active projection streams used by
     *                               the backend's fused decode pool.
     */
    inline void addCudaConcurrentDecodeGemvSideStreamWorkspace(
        WorkspaceRequirements &reqs,
        const DeviceId &device,
        int m,
        size_t projection_count,
        size_t max_concurrent_streams = 8)
    {
        if (!device.is_cuda() || m != 1 || projection_count <= 1 || max_concurrent_streams <= 1)
            return;

        const WorkspaceDescriptor *serial =
            reqs.find(GemmWorkspaceBuffers::GEMV_KPAR_PARTIALS);
        if (!serial || serial->size_bytes == 0)
            return;

        const size_t active_streams =
            std::min(projection_count, max_concurrent_streams);

        // The merged serial descriptor is already sized for the largest
        // projection. Give every active side stream a slot of that largest size
        // so mixed-width fused groups cannot alias or underflow the partial
        // arena. Projection fan-out may exceed active_streams; those later
        // projections reuse stream slots only after the previous launch on that
        // slot has completed.
        WorkspaceRequirements extra;
        extra.buffers.push_back({
            GemmWorkspaceBuffers::CUDA_CONCURRENT_DECODE_GEMV_KPAR_PARTIALS,
            (active_streams - 1) * serial->size_bytes,
            serial->alignment,
            true});
        reqs.merge(extra);
    }

    // =========================================================================
    // FP32 Data Access
    // =========================================================================

    /**
     * @brief Safe FP32 data access for getDumpInfo()
     *
     * For Q8_1 tensors, uses fp32_data() (explicit dequant for debugging only)
     * For other tensors, uses data()
     *
     * For GPU tensors: calls ensureOnHost() to sync GPU data to CPU before access.
     * This enables snapshot capture for CUDA inference.
     *
     * @note This function modifies tensor state (via ensureOnHost) for GPU tensors.
     *       The const_cast is intentional - snapshot capture is a debugging/testing
     *       facility that should not affect normal operation semantics.
     */
    inline const float *getSafeFp32Data(const ITensor *tensor)
    {
        if (!tensor)
            return nullptr;

        // Try to get as TensorBase for CPU/GPU tensor operations
        auto *cpu_tensor = dynamic_cast<const TensorBase *>(tensor);
        if (!cpu_tensor)
            return nullptr;

        // For GPU tensors, sync data to host first
        // This enables snapshot capture for CUDA inference
        if (!tensor->is_on_cpu())
        {
            // const_cast is safe here - ensureOnHost() only copies GPU->CPU without
            // affecting the logical tensor data, and snapshot capture is a debugging facility
            auto *mutable_tensor = const_cast<TensorBase *>(cpu_tensor);
            if (!mutable_tensor->ensureOnHost())
            {
                LOG_WARN("[getSafeFp32Data] Failed to sync GPU tensor to host for snapshot");
                return nullptr;
            }
        }

        if (cpu_tensor->native_type() == TensorType::Q8_1)
        {
            // Q8_1 tensors throw on data() - use explicit fp32_data() for debugging
            auto *q8_1 = dynamic_cast<const Q8_1Tensor *>(cpu_tensor);
            return q8_1 ? q8_1->fp32_data() : nullptr;
        }

        if (cpu_tensor->native_type() == TensorType::Q16_1)
        {
            // Q16_1 tensors also need special handling
            auto *q16_1 = dynamic_cast<const Q16_1Tensor *>(cpu_tensor);
            return q16_1 ? q16_1->fp32_data() : nullptr;
        }

        // Standard FP32/BF16/FP16 tensors
        return cpu_tensor->data();
    }

} // namespace llaminar2
