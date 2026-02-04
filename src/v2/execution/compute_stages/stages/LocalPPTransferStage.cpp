/**
 * @file LocalPPTransferStage.cpp
 * @brief Implementation of activation transfer stage for LOCAL pipeline parallelism
 * @author David Sanftenberg
 * @date February 2026
 */

#include "LocalPPTransferStage.h"
#include "../../../tensors/TensorClasses.h"
#include "../../../utils/Logger.h"

namespace llaminar2
{

    // =========================================================================
    // Construction
    // =========================================================================

    LocalPPTransferStage::LocalPPTransferStage(Params params)
        : IComputeStage(params.device_id), params_(std::move(params))
    {
        // Validation is done at execute time to allow late binding
    }

    // =========================================================================
    // IComputeStage Implementation
    // =========================================================================

    bool LocalPPTransferStage::execute(IDeviceContext *ctx)
    {
        (void)ctx; // Device context not directly used - LOCAL PP context handles devices

        // Validate parameters
        if (!params_.pp_ctx)
        {
            LOG_ERROR("LocalPPTransferStage: null pp_ctx");
            return false;
        }

        if (!params_.tensor)
        {
            LOG_ERROR("LocalPPTransferStage: null tensor");
            return false;
        }

        if (params_.stage_from < 0 || params_.stage_from >= params_.pp_ctx->numStages())
        {
            LOG_ERROR("LocalPPTransferStage: invalid stage_from: " << params_.stage_from
                                                                   << " (numStages=" << params_.pp_ctx->numStages() << ")");
            return false;
        }

        if (params_.stage_to < 0 || params_.stage_to >= params_.pp_ctx->numStages())
        {
            LOG_ERROR("LocalPPTransferStage: invalid stage_to: " << params_.stage_to
                                                                 << " (numStages=" << params_.pp_ctx->numStages() << ")");
            return false;
        }

        // Same-device transfer is a no-op
        if (params_.stage_from == params_.stage_to)
        {
            LOG_DEBUG("LocalPPTransferStage: same stage, no-op");
            return true;
        }

        // Check if devices are the same (different stages can share a device)
        const auto &src_device = params_.pp_ctx->deviceForStage(params_.stage_from);
        const auto &dst_device = params_.pp_ctx->deviceForStage(params_.stage_to);
        if (src_device == dst_device)
        {
            LOG_DEBUG("LocalPPTransferStage: same device for stages " << params_.stage_from
                                                                      << " and " << params_.stage_to << ", no-op");
            return true;
        }

        // Log transfer details
        LOG_DEBUG("LocalPPTransferStage: transferring " << params_.tensor->numel() << " elements"
                                                        << " from stage " << params_.stage_from << " (" << src_device.toString() << ")"
                                                        << " to stage " << params_.stage_to << " (" << dst_device.toString() << ")"
                                                        << " using backend " << collectiveBackendTypeToString(
                                                                                    params_.pp_ctx->backendForTransfer(params_.stage_from, params_.stage_to)));

        // Delegate to LOCAL PP context
        bool success = params_.pp_ctx->transfer(params_.tensor, params_.stage_from, params_.stage_to);

        if (!success)
        {
            LOG_ERROR("LocalPPTransferStage: transfer failed from stage " << params_.stage_from
                                                                          << " to stage " << params_.stage_to);
        }

        return success;
    }

    std::string LocalPPTransferStage::name() const
    {
        if (!params_.stage_name.empty())
        {
            return params_.stage_name;
        }
        return "LocalPPTransfer_" + std::to_string(params_.stage_from) + "_to_" + std::to_string(params_.stage_to);
    }

    bool LocalPPTransferStage::supportsBackend(ComputeBackendType backend) const
    {
        // LOCAL PP can work with any backend - the pp_ctx handles routing
        (void)backend;
        return true;
    }

    StageBufferRequirements LocalPPTransferStage::getBufferRequirements() const
    {
        StageBufferRequirements reqs;

        if (params_.tensor)
        {
            // Transfer: tensor is input from source device, output on dest device
            BufferTensorType buf_type = toBufferTensorType(params_.tensor->native_type());
            reqs.addInput("tensor", params_.tensor->shape(), buf_type);
            reqs.addOutput("tensor", params_.tensor->shape(), buf_type);
        }

        return reqs;
    }

    StageDumpInfo LocalPPTransferStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;

        if (params_.tensor)
        {
            const size_t rows = params_.tensor->rows();
            const size_t cols = params_.tensor->cols();

            info.addInput("tensor", params_.tensor, rows, cols);
            info.addOutput("tensor", params_.tensor, rows, cols);
        }

        // Add transfer metadata
        info.addScalarInt("stage_from", params_.stage_from);
        info.addScalarInt("stage_to", params_.stage_to);

        if (params_.pp_ctx)
        {
            info.addScalarInt("num_stages", params_.pp_ctx->numStages());
            info.addScalarInt("backend", static_cast<int>(
                                             params_.pp_ctx->backendForTransfer(params_.stage_from, params_.stage_to)));
        }

        return info;
    }

    void LocalPPTransferStage::setParams(const Params &params)
    {
        params_ = params;
        // Note: IComputeStage doesn't expose setDevice() publicly, so device
        // is fixed at construction. For reuse with different device, create a new stage.
    }

} // namespace llaminar2
