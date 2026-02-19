/**
 * @file TPAllreduceStage.cpp
 * @brief Implementation of all-reduce stage for tensor parallelism
 * @author David Sanftenberg
 * @date February 2026
 */

#include "TPAllreduceStage.h"
#include "../../../tensors/TensorClasses.h"
#include "../../../utils/Logger.h"
#include "../../../utils/KernelProfiler.h"

#ifdef HAVE_ROCM
#include <hip/hip_runtime.h>
#endif

namespace llaminar2
{

    // =========================================================================
    // Construction
    // =========================================================================

    TPAllreduceStage::TPAllreduceStage(Params params)
        : IComputeStage(params.device_id), params_(std::move(params))
    {
        // Validation is done at execute time to allow late binding
    }

    // =========================================================================
    // IComputeStage Implementation
    // =========================================================================

    bool TPAllreduceStage::execute(IDeviceContext *ctx)
    {
        KERNEL_PROFILE_SCOPE(KernelType::ALLREDUCE);

        (void)ctx; // Device context not directly used - TP context handles devices

        // Validate parameters
        if (!params_.tp_ctx)
        {
            LOG_ERROR("TPAllreduceStage: null tp_ctx");
            return false;
        }

        if (!params_.tensor)
        {
            LOG_ERROR("TPAllreduceStage: null tensor");
            return false;
        }

        // Single-device context - no-op
        if (params_.tp_ctx->degree() == 1)
        {
            LOG_DEBUG("TPAllreduceStage: single device, no-op");
            return true;
        }

        // Resolve count: 0 means use tensor->numel()
        const size_t effective_count = (params_.count > 0) ? params_.count : params_.tensor->numel();

        const auto tensor_current_device = params_.tensor->current_device();
#ifdef HAVE_ROCM
        int current_hip_device = -1;
        if (hipGetDevice(&current_hip_device) != hipSuccess)
        {
            current_hip_device = -1;
        }
#else
        const int current_hip_device = -1;
#endif
        LOG_DEBUG("TPAllreduceStage: tensor diagnostics"
                  << " stage_name=" << (params_.stage_name.empty() ? "(none)" : params_.stage_name)
                  << " tensor=" << static_cast<void *>(params_.tensor)
                  << " tensor_name=" << (params_.tensor->debugName().empty() ? "(unnamed)" : params_.tensor->debugName())
                  << " home_device=" << params_.tensor->home_device().toString()
                  << " current_device=" << (tensor_current_device.has_value() ? tensor_current_device->toString() : "none")
                  << " stage_stream=" << gpuStream()
                  << " hip_current_device=" << current_hip_device
                  << " gpu_ptr=" << params_.tensor->gpu_data_ptr()
                  << " count=" << effective_count
                  << " tensor_numel=" << params_.tensor->numel());

        // Log scope-aware message
        const char *scope_str = params_.tp_ctx->isGlobal() ? "GLOBAL" : "LOCAL";
        LOG_DEBUG("TPAllreduceStage (" << scope_str << "): all-reduce across " << params_.tp_ctx->degree()
                                       << " devices using " << collectiveBackendTypeToString(params_.tp_ctx->backend())
                                       << " stage_name=" << (params_.stage_name.empty() ? "(none)" : params_.stage_name)
                                       << " count=" << effective_count
                                       << " (params_.count=" << params_.count
                                       << ", tensor numel=" << params_.tensor->numel() << ")");

        // Use stage_name overload with count parameter
        // CRITICAL: Pass actual count for decode (seq_len * hidden_dim, not buffer size)
        bool success;
        if (!params_.stage_name.empty())
        {
            success = params_.tp_ctx->allreduce(params_.tensor, params_.stage_name, effective_count);
        }
        else
        {
            // Fall back to no-stage-name overload (uses tensor->numel())
            success = params_.tp_ctx->allreduce(params_.tensor);
        }

        if (!success)
        {
            LOG_ERROR("TPAllreduceStage (" << scope_str << "): allreduce failed");
        }

        return success;
    }

    bool TPAllreduceStage::supportsBackend(ComputeBackendType backend) const
    {
        // TP can work with any backend - the tp_ctx handles routing
        (void)backend;
        return true;
    }

    StageBufferRequirements TPAllreduceStage::getBufferRequirements() const
    {
        StageBufferRequirements reqs;

        if (params_.tensor)
        {
            // In-place all-reduce: buffer is both input and output
            BufferTensorType buf_type = toBufferTensorType(params_.tensor->native_type());
            reqs.addInout("tensor", params_.tensor->shape(), buf_type);
        }

        return reqs;
    }

    StageDumpInfo TPAllreduceStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;

        if (params_.tensor)
        {
            // All-reduce is in-place, so tensor is both input and output
            // Use actual count being reduced, not buffer size
            // For decode: count = seq_len * hidden_dim (not max_seq_len * hidden_dim)
            const size_t effective_count = (params_.count > 0) ? params_.count : params_.tensor->numel();
            const size_t cols = params_.tensor->cols();
            const size_t rows = (cols > 0) ? effective_count / cols : effective_count;

            info.addInput("tensor", params_.tensor, rows, cols);
            info.addOutput("tensor", params_.tensor, rows, cols);
        }

        if (params_.tp_ctx)
        {
            info.addScalarInt("tp_degree", params_.tp_ctx->degree());
            info.addScalarInt("backend", static_cast<int>(params_.tp_ctx->backend()));
            info.addScalarInt("is_global", params_.tp_ctx->isGlobal() ? 1 : 0);
        }

        return info;
    }

    void TPAllreduceStage::setParams(const Params &params)
    {
        params_ = params;
        // Update base class device
        // Note: IComputeStage doesn't expose setDevice() publicly, so device
        // is fixed at construction. For reuse, create a new stage.
    }

} // namespace llaminar2
