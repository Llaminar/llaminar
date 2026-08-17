/**
 * @file MoEOverlayEpochBoundaryStage.cpp
 * @brief Implementation of captured ExpertOverlay epoch reader boundaries.
 */

#include "MoEOverlayEpochBoundaryStage.h"

#include "../../../kernels/KernelFactory.h"
#include "../../../utils/Logger.h"

#include <stdexcept>
#include <utility>

namespace llaminar2
{
    MoEOverlayEpochBoundaryStage::MoEOverlayEpochBoundaryStage(Params params)
        : IComputeStage(params.device_id),
          params_(std::move(params)),
          moe_kernel_(
              llaminar::v2::kernels::KernelFactory::createMoEKernel(
                  params_.device_id))
    {
        if (!validate())
        {
            throw std::invalid_argument(
                "MoEOverlayEpochBoundaryStage requires one complete GPU epoch binding");
        }
        if (!moe_kernel_)
        {
            throw std::runtime_error(
                "MoEOverlayEpochBoundaryStage could not create its MoE kernel");
        }
    }

    bool MoEOverlayEpochBoundaryStage::validate() const noexcept
    {
        return params_.device_id.is_gpu() && params_.arena &&
               params_.arena->deviceId() == params_.device_id &&
               params_.request_slot < params_.arena->requestSlotCapacity() &&
               (params_.operation == Operation::Acquire ||
                params_.operation == Operation::Release);
    }

    bool MoEOverlayEpochBoundaryStage::execute(IDeviceContext *ctx)
    {
        if (!ensureContext(ctx, "MoEOverlayEpochBoundaryStage") ||
            !validate() || !moe_kernel_)
        {
            return false;
        }

        void *const stream = requireGPUStream();
        if (!stream)
            return false;

        /*
         * The ticket and status are stable device addresses. The kernel reads
         * the live publication selector and reader counters in stream order; no
         * host copy or shadow value participates in this transition.
         */
        const MoEKernelLaunchContext launch{
            .stream = stream,
            .workspace = nullptr,
        };
        DeviceMoEOverlayEpochTicket *const ticket =
            params_.arena->requestTicket(params_.request_slot);
        DeviceMoEOverlayEpochStatus *const status =
            params_.arena->requestStatus(params_.request_slot);
        if (params_.operation == Operation::Acquire)
        {
            return moe_kernel_->acquireMoEOverlayEpoch(
                launch,
                params_.arena->control(),
                ticket,
                status);
        }
        return moe_kernel_->releaseMoEOverlayEpoch(
            launch,
            params_.arena->control(),
            ticket,
            status);
    }

    size_t MoEOverlayEpochBoundaryStage::estimatedMemoryBytes() const
    {
        return sizeof(DeviceMoEOverlayEpochControl) +
               sizeof(DeviceMoEOverlayEpochTicket) +
               sizeof(DeviceMoEOverlayEpochStatus);
    }

    bool MoEOverlayEpochBoundaryStage::supportsBackend(
        ComputeBackendType backend) const
    {
        return backend == ComputeBackendType::GPU_CUDA ||
               backend == ComputeBackendType::GPU_ROCM;
    }

    StageDumpInfo MoEOverlayEpochBoundaryStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;
        info.addScalarInt(
            "operation",
            static_cast<int>(params_.operation));
        info.addScalarInt(
            "request_slot",
            static_cast<int>(params_.request_slot));
        return info;
    }

    bool MoEOverlayEpochBoundaryStage::hasSameCaptureIdentity(
        const Params &other) const noexcept
    {
        return params_.device_id == other.device_id &&
               params_.arena == other.arena &&
               params_.request_slot == other.request_slot &&
               params_.operation == other.operation;
    }
} // namespace llaminar2
