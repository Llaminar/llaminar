/**
 * @file MoEDeviceDecodeCommitBoundaryStage.cpp
 * @brief Implementation of captured ROCm serial-decode MoE cadence publication.
 */

#include "MoEDeviceDecodeCommitBoundaryStage.h"

#include "MoEDeviceRebalanceStage.h"

#include "../../../backends/IBackend.h"
#include "../../../execution/moe/DeviceMoERebalanceController.h"
#include "../../../execution/local_execution/device/DeviceWorkspaceManager.h"
#include "../../../execution/local_execution/device/WorkspaceDescriptor.h"
#include "../../../utils/Logger.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace llaminar2
{
    MoEDeviceDecodeCommitBoundaryStage::
        MoEDeviceDecodeCommitBoundaryStage(Params params)
        : IComputeStage(params.device_id), params_(std::move(params))
    {
        if (!validate())
        {
            throw std::invalid_argument(
                "MoEDeviceDecodeCommitBoundaryStage requires an explicit "
                "ROCm backend and maintenance-workspace identity");
        }
    }

    bool MoEDeviceDecodeCommitBoundaryStage::validate() const noexcept
    {
        return params_.device_id.is_rocm() && params_.backend != nullptr &&
               !params_.workspace_name.empty() &&
               !params_.stage_name.empty();
    }

    std::string
    MoEDeviceDecodeCommitBoundaryStage::controllerBufferName() const
    {
        return MoEDeviceRebalanceStage::workspaceBufferName(
            MoEDeviceRebalanceStage::WS_CONTROLLER_STATE,
            params_.workspace_name);
    }

    bool MoEDeviceDecodeCommitBoundaryStage::execute(IDeviceContext *ctx)
    {
        if (!ensureContext(ctx, "MoEDeviceDecodeCommitBoundaryStage") ||
            !validate() || !bound_workspace_)
        {
            LOG_ERROR(
                "[MoEDeviceDecodeCommitBoundaryStage] Complete ROCm context "
                "and persistent workspace are required");
            return false;
        }

        auto *const controller =
            static_cast<DeviceMoERebalanceGraphControllerState *>(
                bound_workspace_->getBuffer(controllerBufferName()));
        if (!controller)
        {
            LOG_ERROR(
                "[MoEDeviceDecodeCommitBoundaryStage] Missing shared controller buffer '"
                << controllerBufferName() << "'");
            return false;
        }

        void *const stream = requireGPUStream();
        if (!stream)
            return false;

        /*
         * These are device addresses, never host-dereferenced state.  Offsets
         * are computed once while the C++ stage is captured; native graph nodes
         * retain the stable model-lifetime pointers for every later replay.
         */
        auto *const bytes = reinterpret_cast<std::byte *>(controller);
        auto *const committed = reinterpret_cast<std::uint32_t *>(
            bytes + offsetof(
                        DeviceMoERebalanceGraphControllerState,
                        decode_rounds_committed));
        auto *const remaining = reinterpret_cast<std::uint32_t *>(
            bytes + offsetof(
                        DeviceMoERebalanceGraphControllerState,
                        decode_rounds_until_maintenance));
        auto *const due = reinterpret_cast<std::uint32_t *>(
            bytes + offsetof(
                        DeviceMoERebalanceGraphControllerState,
                        maintenance_due));
        auto *const advanced = reinterpret_cast<std::uint32_t *>(
            bytes + offsetof(
                        DeviceMoERebalanceGraphControllerState,
                        decode_boundary_advanced));

        if (!params_.backend->enqueuePublishSerialDecodeCommitBoundary(
                committed,
                remaining,
                due,
                advanced,
                params_.device_id.gpu_ordinal(),
                stream))
        {
            LOG_ERROR(
                "[MoEDeviceDecodeCommitBoundaryStage] Backend rejected serial "
                "decode cadence publication");
            return false;
        }

        /*
         * A non-due edge is retired before this complete decode graph returns.
         * A due edge remains armed because the acknowledgement kernel observes
         * `maintenance_due != 0`; the later authenticated ticket transaction
         * is therefore the only host-visible branch in the HIP protocol.
         */
        if (!params_.backend->enqueueAcknowledgeDecodeCommitBoundary(
                remaining,
                due,
                advanced,
                params_.device_id.gpu_ordinal(),
                stream))
        {
            LOG_ERROR(
                "[MoEDeviceDecodeCommitBoundaryStage] Backend rejected serial "
                "decode cadence acknowledgement");
            return false;
        }
        return true;
    }

    size_t MoEDeviceDecodeCommitBoundaryStage::estimatedMemoryBytes() const
    {
        // Two kernels each touch a bounded subset of four 32-bit fields.
        return 2u * 4u * sizeof(std::uint32_t);
    }

    bool MoEDeviceDecodeCommitBoundaryStage::supportsBackend(
        ComputeBackendType backend) const
    {
        return backend == ComputeBackendType::GPU_ROCM;
    }

    WorkspaceRequirements
    MoEDeviceDecodeCommitBoundaryStage::getWorkspaceRequirements(
        int m,
        int n,
        int k) const
    {
        (void)m;
        (void)n;
        (void)k;
        WorkspaceRequirements requirements;
        requirements.buffers.push_back(
            {controllerBufferName(),
             sizeof(DeviceMoERebalanceGraphControllerState),
             256u,
             true});
        return requirements;
    }

    void MoEDeviceDecodeCommitBoundaryStage::bindWorkspace(
        DeviceWorkspaceManager *workspace)
    {
        bound_workspace_ = workspace;
    }

    void MoEDeviceDecodeCommitBoundaryStage::unbindWorkspace()
    {
        bound_workspace_ = nullptr;
    }

    StageDumpInfo
    MoEDeviceDecodeCommitBoundaryStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;
        info.addScalarBool("publishes_serial_round", true);
        info.addScalarBool("acknowledges_only_non_due", true);
        return info;
    }
} // namespace llaminar2
