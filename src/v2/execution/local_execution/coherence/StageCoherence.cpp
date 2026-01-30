/**
 * @file StageCoherence.cpp
 * @brief Implementation of automatic device coherence management
 * @author David Sanftenberg
 * @date January 2026
 */

#include "StageCoherence.h"
#include "../../compute_stages/IComputeStage.h"
#include "../../../tensors/TensorClasses.h"
#include "../../../tensors/TensorSlice.h"
#include "../../../utils/Logger.h"
#include "../../../utils/DebugEnv.h"
#include <chrono>

namespace llaminar2
{

    // =========================================================================
    // Coherence Operations
    // =========================================================================

    bool cohereInputs(const std::vector<CoherenceBuffer> &inputs, DeviceId target_device)
    {
        if (inputs.empty())
        {
            return true;
        }

        const bool trace_coherence = debugEnv().rocm.trace_coherence;
        LOG_DEBUG("[StageCoherence] Cohering " << inputs.size() << " inputs to " << target_device.to_string());

        for (const auto &buf : inputs)
        {
            if (!buf.tensor)
            {
                // No tensor pointer available - skip silently
                continue;
            }

            // IMPORTANT: Check for TensorSlice FIRST before TensorBase!
            // TensorSlice inherits from TensorBase (which IS TensorBase), so the
            // TensorBase cast would succeed, but we need to cohere the INNER tensor,
            // not the slice wrapper itself.
            TensorBase *tensor_base = nullptr;
            const void *slice_ptr_for_log = nullptr;

            auto *slice = dynamic_cast<TensorSlice *>(buf.tensor);
            if (slice)
            {
                slice_ptr_for_log = slice;
                // It's a TensorSlice - delegate to inner tensor
                tensor_base = dynamic_cast<TensorBase *>(slice->inner());
                if (!tensor_base)
                {
                    LOG_DEBUG("[StageCoherence] Input '" << (buf.name ? buf.name : "unknown")
                                                         << "' is TensorSlice but inner doesn't support coherence");
                    continue;
                }
                LOG_DEBUG("[StageCoherence] Input '" << (buf.name ? buf.name : "unknown")
                                                     << "' unwrapped from TensorSlice " << slice_ptr_for_log
                                                     << " to " << tensor_base->dtype_name()
                                                     << " inner_ptr=" << static_cast<void *>(tensor_base));
            }
            else
            {
                // Not a TensorSlice - try direct TensorBase cast
                tensor_base = dynamic_cast<TensorBase *>(buf.tensor);
                if (!tensor_base)
                {
                    LOG_DEBUG("[StageCoherence] Input '" << (buf.name ? buf.name : "unknown")
                                                         << "' does not support device coherence (not TensorBase or TensorSlice)");
                    continue;
                }
            }

            // Check if tensor is already on device
            if (tensor_base->is_on_device(target_device))
            {
                if (trace_coherence)
                {
                    LOG_INFO("[StageCoherence] Input '" << (buf.name ? buf.name : "unknown")
                                                        << "' already on " << target_device.to_string());
                }
                continue;
            }

            // Check if tensor has cached device data (e.g., packed INT8 weights for CUDA GEMM)
            // These tensors don't need raw data upload - the kernel uses its own representation
            if (tensor_base->hasCachedDeviceData(target_device.type))
            {
                if (trace_coherence)
                {
                    LOG_INFO("[StageCoherence] Input '" << (buf.name ? buf.name : "unknown")
                                                        << "' has cached device data for " << target_device.to_string()
                                                        << " (skipping raw upload)");
                }
                continue;
            }

            // DEBUG: Log why is_on_device returned false for weights
            if (trace_coherence && (std::string(buf.name ? buf.name : "").find("w") == 0))
            {
                auto current_dev = tensor_base->current_device();
                LOG_WARN("[StageCoherence] WEIGHT '" << (buf.name ? buf.name : "unknown")
                                                     << "' is_on_device(" << target_device.to_string() << ") = false"
                                                     << " | gpu_device=" << (current_dev.has_value() ? current_dev->to_string() : "none")
                                                     << " device_valid=" << (tensor_base->is_on_device(target_device) ? "true" : "false (rechecked)")
                                                     << " ptr=" << tensor_base);
            }

            // Ensure tensor is on target device
            // For CPU targets, use ensureOnHost() to sync data from GPU if needed
            // For GPU targets, use ensureOnDevice() to upload data if needed
            if (target_device.is_cpu())
            {
                LOG_DEBUG("[StageCoherence] Syncing input '" << (buf.name ? buf.name : "unknown")
                                                             << "' to host (CPU)");
                if (!tensor_base->ensureOnHost())
                {
                    LOG_ERROR("[StageCoherence] Failed to sync input '"
                              << (buf.name ? buf.name : "unknown")
                              << "' to host");
                    return false;
                }
            }
            else
            {
                auto start = std::chrono::high_resolution_clock::now();
                bool success = tensor_base->ensureOnDevice(target_device);
                auto end = std::chrono::high_resolution_clock::now();
                auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

                if (trace_coherence)
                {
                    LOG_INFO("[StageCoherence] ensureOnDevice('" << (buf.name ? buf.name : "unknown")
                                                                 << "') to " << target_device.to_string()
                                                                 << " took " << elapsed_us << " us"
                                                                 << " (numel=" << tensor_base->numel() << ")"
                                                                 << " ptr=" << static_cast<void *>(tensor_base)
                                                                 << " gpu_ptr=" << tensor_base->gpu_data_ptr());
                }

                if (!success)
                {
                    LOG_ERROR("[StageCoherence] Failed to upload input '"
                              << (buf.name ? buf.name : "unknown")
                              << "' to " << target_device.to_string());
                    return false;
                }
            }
        }

        return true;
    }

    bool cohereOutputs(const std::vector<CoherenceBuffer> &outputs, DeviceId target_device)
    {
        // Only needed for GPU targets - CPU buffers don't need pre-allocation
        if (target_device.is_cpu())
        {
            return true;
        }

        if (outputs.empty())
        {
            return true;
        }

        const bool trace_coherence = debugEnv().rocm.trace_coherence;
        LOG_DEBUG("[StageCoherence] Allocating GPU buffers for " << outputs.size() << " outputs on " << target_device.to_string());

        for (const auto &buf : outputs)
        {
            if (!buf.tensor)
            {
                // No tensor pointer available - skip silently
                continue;
            }

            // Try to cast to TensorBase to access coherence methods
            auto *tensor_base = dynamic_cast<TensorBase *>(buf.tensor);
            if (!tensor_base)
            {
                // Tensor doesn't support coherence - skip
                LOG_DEBUG("[StageCoherence] Output '" << (buf.name ? buf.name : "unknown")
                                                      << "' does not support device coherence");
                continue;
            }

            // Check if tensor is already on device
            bool already_on_device = tensor_base->is_on_device(target_device);
            if (trace_coherence && !already_on_device)
            {
                // Debug: why is_on_device returned false?
                auto current_dev = tensor_base->current_device();
                LOG_INFO("[StageCoherence] OUTPUT '" << (buf.name ? buf.name : "unknown")
                                                     << "' NOT on " << target_device.to_string()
                                                     << " | gpu_device=" << (current_dev.has_value() ? current_dev->to_string() : "none")
                                                     << " ptr=" << static_cast<void *>(tensor_base));
            }
            if (already_on_device)
            {
                if (trace_coherence)
                {
                    LOG_INFO("[StageCoherence] Output '" << (buf.name ? buf.name : "unknown")
                                                         << "' already on " << target_device.to_string());
                }
                continue;
            }

            // Ensure output buffer is allocated on device
            // Use allocateOnDevice() - allocates GPU memory but does NOT upload host data
            // (the kernel will write to the GPU buffer, so H2D upload would be wasteful)
            auto start = std::chrono::high_resolution_clock::now();
            bool success = tensor_base->allocateOnDevice(target_device);
            auto end = std::chrono::high_resolution_clock::now();
            auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

            if (trace_coherence)
            {
                LOG_INFO("[StageCoherence] allocateOnDevice OUTPUT('" << (buf.name ? buf.name : "unknown")
                                                                      << "') to " << target_device.to_string()
                                                                      << " took " << elapsed_us << " us"
                                                                      << " (numel=" << tensor_base->numel() << ")");
            }

            if (!success)
            {
                LOG_ERROR("[StageCoherence] Failed to allocate output buffer '"
                          << (buf.name ? buf.name : "unknown")
                          << "' on " << target_device.to_string());
                return false;
            }
        }

        return true;
    }

    void markOutputsDirty(const std::vector<CoherenceBuffer> &outputs)
    {
        if (outputs.empty())
        {
            return;
        }

        LOG_DEBUG("[StageCoherence] Marking " << outputs.size() << " outputs as device-dirty (with events)");

        for (const auto &buf : outputs)
        {
            if (!buf.tensor)
            {
                // No tensor pointer available - skip silently
                continue;
            }

            // Try to cast to TensorBase to access coherence methods
            auto *tensor_base = dynamic_cast<TensorBase *>(buf.tensor);
            if (!tensor_base)
            {
                // Tensor doesn't support coherence - skip
                LOG_DEBUG("[StageCoherence] Output '" << (buf.name ? buf.name : "unknown")
                                                      << "' does not support device coherence");
                continue;
            }

            // Mark as device-dirty WITH EVENT for fine-grained synchronization
            // This records a completion event so ensureOnHost() can wait on just
            // this kernel rather than doing a full device sync
            LOG_DEBUG("[StageCoherence] Marking output '" << (buf.name ? buf.name : "unknown")
                                                          << "' as device-dirty (with event)");
            tensor_base->mark_device_dirty_with_event();
        }
    }

    // =========================================================================
    // Buffer Extraction
    // =========================================================================

    std::vector<CoherenceBuffer> extractInputBuffers(const StageDumpInfo &info)
    {
        std::vector<CoherenceBuffer> result;
        result.reserve(info.inputs.size());

        for (const auto &input : info.inputs)
        {
            CoherenceBuffer buf;
            buf.tensor = input.tensor;
            buf.name = input.name;
            buf.data = input.data;
            buf.rows = input.rows;
            buf.cols = input.cols;
            buf.dtype = input.dtype;
            buf.is_inout = false;

            result.push_back(buf);
        }

        return result;
    }

    std::vector<CoherenceBuffer> extractWeightBuffers(const StageDumpInfo &info)
    {
        std::vector<CoherenceBuffer> result;
        result.reserve(info.weights.size());

        for (const auto &weight : info.weights)
        {
            CoherenceBuffer buf;
            // Need to cast away const for coherence operations (upload is non-mutating semantically)
            buf.tensor = const_cast<ITensor *>(weight.tensor);
            buf.name = weight.name;
            buf.data = weight.raw_data;
            buf.rows = weight.rows;
            buf.cols = weight.cols;
            buf.dtype = weight.dtype;
            buf.is_inout = false;

            result.push_back(buf);
        }

        return result;
    }

    std::vector<CoherenceBuffer> extractOutputBuffers(const StageDumpInfo &info)
    {
        std::vector<CoherenceBuffer> result;
        result.reserve(info.outputs.size());

        for (const auto &output : info.outputs)
        {
            CoherenceBuffer buf;
            buf.tensor = output.tensor;
            buf.name = output.name;
            buf.data = output.data;
            buf.rows = output.rows;
            buf.cols = output.cols;
            buf.dtype = output.dtype;
            buf.is_inout = false;

            result.push_back(buf);
        }

        return result;
    }

    // =========================================================================
    // Utility Functions
    // =========================================================================

    const char *toString(CoherencePolicy policy)
    {
        switch (policy)
        {
        case CoherencePolicy::NONE:
            return "NONE";
        case CoherencePolicy::INPUT:
            return "INPUT";
        case CoherencePolicy::OUTPUT:
            return "OUTPUT";
        case CoherencePolicy::FULL:
            return "FULL";
        default:
            return "UNKNOWN";
        }
    }

} // namespace llaminar2
