/**
 * @file LocalTPAllreduceStage.cpp
 * @brief Implementation of all-reduce stage for LOCAL tensor parallelism
 * @author David Sanftenberg
 * @date January 2026
 */

#include "LocalTPAllreduceStage.h"
#include "../../../tensors/TensorClasses.h"
#include "../../../utils/Logger.h"

namespace llaminar2
{

    // =========================================================================
    // Construction
    // =========================================================================

    LocalTPAllreduceStage::LocalTPAllreduceStage(Params params)
        : IComputeStage(params.device_id), params_(std::move(params))
    {
        // Validation is done at execute time to allow late binding
    }

    // =========================================================================
    // IComputeStage Implementation
    // =========================================================================

    bool LocalTPAllreduceStage::execute(IDeviceContext *ctx)
    {
        (void)ctx; // Device context not directly used - LOCAL TP context handles devices

        // Validate parameters
        if (!params_.tp_ctx)
        {
            LOG_ERROR("LocalTPAllreduceStage: null tp_ctx");
            return false;
        }

        if (!params_.tensor)
        {
            LOG_ERROR("LocalTPAllreduceStage: null tensor");
            return false;
        }

        // Single-device context - no-op
        if (params_.tp_ctx->degree() == 1)
        {
            LOG_DEBUG("LocalTPAllreduceStage: single device, no-op");
            return true;
        }

        // Resolve count: 0 means use tensor->numel()
        const size_t effective_count = (params_.count > 0) ? params_.count : params_.tensor->numel();

        // Delegate to LOCAL TP context
        LOG_DEBUG("LocalTPAllreduceStage: all-reduce across " << params_.tp_ctx->degree()
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
            LOG_ERROR("LocalTPAllreduceStage: allreduce failed");
        }

        return success;
    }

    bool LocalTPAllreduceStage::supportsBackend(ComputeBackendType backend) const
    {
        // LOCAL TP can work with any backend - the tp_ctx handles routing
        (void)backend;
        return true;
    }

    StageBufferRequirements LocalTPAllreduceStage::getBufferRequirements() const
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

    StageDumpInfo LocalTPAllreduceStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;

        if (params_.tensor)
        {
            // All-reduce is in-place, so tensor is both input and output
            info.addInput("tensor", params_.tensor, params_.tensor->rows(), params_.tensor->cols());
            info.addOutput("tensor", params_.tensor, params_.tensor->rows(), params_.tensor->cols());
        }

        if (params_.tp_ctx)
        {
            info.addScalarInt("tp_degree", params_.tp_ctx->degree());
            info.addScalarInt("backend", static_cast<int>(params_.tp_ctx->backend()));
        }

        return info;
    }

    void LocalTPAllreduceStage::setParams(const Params &params)
    {
        params_ = params;
        // Update base class device
        // Note: IComputeStage doesn't expose setDevice() publicly, so device
        // is fixed at construction. For reuse, create a new stage.
    }

} // namespace llaminar2
