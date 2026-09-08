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
                params_.operation == Operation::Release) &&
               (!params_.peer_epoch_source.has_value() ||
                (params_.operation == Operation::Acquire &&
                 params_.peer_epoch_source->valid())) &&
               (!params_.retirement_readiness_controller.has_value() ||
                (params_.operation == Operation::Release &&
                 params_.retirement_readiness_controller->valid() &&
                 params_.retirement_readiness_controller->device ==
                     params_.device_id));
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
            const std::uint64_t *required_epoch =
                params_.arena->externalAdmissionEpoch();
            auto admission_barrier = params_.arena->admissionBarrier();
            MoEOverlayPeerPlacementEpochBinding peer_placement_epoch;
            if (params_.peer_epoch_source)
            {
                const auto &peer = *params_.peer_epoch_source;
                /* The kernel waits on activation generation plus descriptor
                 * digest, not the reusable leased timeline value alone. This
                 * makes the epoch selection and reader installation one
                 * captured device operation. */
                required_epoch = nullptr;
                admission_barrier = {};
                peer_placement_epoch = peer.binding;
            }
            return moe_kernel_->acquireMoEOverlayEpoch(
                launch,
                params_.arena->control(),
                ticket,
                status,
                required_epoch,
                admission_barrier,
                peer_placement_epoch);
        }
        if (!moe_kernel_->releaseMoEOverlayEpoch(
            launch,
            params_.arena->control(),
            ticket,
            status))
        {
            return false;
        }
        if (!params_.retirement_readiness_controller)
            return true;
        return moe_kernel_->runMoEOverlayDeviceControllerAction(
            launch,
            {
                .binding = params_.retirement_readiness_controller
                               ->deviceBinding(),
                .action = MoEOverlayDeviceControllerAction::
                    PublishRuntimeRetirementReadiness,
                .retirement_readiness = {
                    .epoch_control = params_.arena->control(),
                },
            });
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
        info.addScalarInt(
            "peer_selected_epoch",
            params_.peer_epoch_source.has_value() ? 1 : 0);
        info.addScalarInt(
            "transaction_epoch_barrier",
            params_.arena->admissionBarrier().valid() ? 1 : 0);
        info.addScalarInt(
            "retirement_readiness_receipt",
            params_.retirement_readiness_controller.has_value() ? 1 : 0);
        return info;
    }

    bool MoEOverlayEpochBoundaryStage::hasSameCaptureIdentity(
        const Params &other) const noexcept
    {
        const bool same_peer_source =
            params_.peer_epoch_source.has_value() ==
                other.peer_epoch_source.has_value() &&
            (!params_.peer_epoch_source ||
             (params_.peer_epoch_source->mapped_region ==
                  other.peer_epoch_source->mapped_region &&
              params_.peer_epoch_source->binding ==
                  other.peer_epoch_source->binding));
        const auto same_retirement_controller = [&]
        {
            if (params_.retirement_readiness_controller.has_value() !=
                other.retirement_readiness_controller.has_value())
            {
                return false;
            }
            if (!params_.retirement_readiness_controller)
                return true;
            const auto &left = *params_.retirement_readiness_controller;
            const auto &right = *other.retirement_readiness_controller;
            return left.device == right.device &&
                   left.participant_id == right.participant_id &&
                   left.group_id == right.group_id &&
                   left.controller == right.controller &&
                   left.command == right.command &&
                   left.local_participant_record ==
                       right.local_participant_record &&
                   left.lifetime == right.lifetime;
        }();
        return params_.device_id == other.device_id &&
               params_.arena == other.arena &&
               params_.request_slot == other.request_slot &&
               params_.operation == other.operation && same_peer_source &&
               same_retirement_controller;
    }
} // namespace llaminar2
