/**
 * @file MoEOverlayDeviceControllerStage.cpp
 * @brief Captured-stage implementation for mapped ExpertOverlay control.
 */

#include "MoEOverlayDeviceControllerStage.h"

#include "../../../execution/local_execution/device/DeviceWorkspaceManager.h"
#include "../../../execution/local_execution/device/WorkspaceDescriptor.h"
#include "../../../kernels/KernelFactory.h"
#include "../../../utils/Logger.h"

#include <stdexcept>
#include <utility>

namespace llaminar2
{
    namespace
    {
        /** @return Whether the action mutates topology-wide leader state. */
        bool requiresLeader(
            MoEOverlayDeviceControllerAction action) noexcept
        {
            switch (action)
            {
            case MoEOverlayDeviceControllerAction::BeginTransaction:
            case MoEOverlayDeviceControllerAction::PublishCommand:
            case MoEOverlayDeviceControllerAction::BeginCommit:
            case MoEOverlayDeviceControllerAction::PublishAdmission:
            case MoEOverlayDeviceControllerAction::BeginDynamicRetirement:
            case MoEOverlayDeviceControllerAction::CompleteDynamicRetirement:
            case MoEOverlayDeviceControllerAction::BeginLLEPRestore:
            case MoEOverlayDeviceControllerAction::CompleteLLEPRestore:
            case MoEOverlayDeviceControllerAction::AuthorStaticPolicy:
            case MoEOverlayDeviceControllerAction::AuthorDynamicPolicy:
                return true;
            default:
                return false;
            }
        }

        /** @return Whether the action publishes one group-owned record. */
        bool requiresGroupRoot(
            MoEOverlayDeviceControllerAction action) noexcept
        {
            switch (action)
            {
            case MoEOverlayDeviceControllerAction::PublishGroupSnapshot:
            case MoEOverlayDeviceControllerAction::AcknowledgePrepared:
            case MoEOverlayDeviceControllerAction::AcknowledgePublished:
            case MoEOverlayDeviceControllerAction::AcknowledgeRetired:
            case MoEOverlayDeviceControllerAction::AcknowledgeLLEPRestored:
                return true;
            default:
                return false;
            }
        }
    } // namespace

    MoEOverlayDeviceControllerStage::MoEOverlayDeviceControllerStage(
        Params params)
        : IComputeStage(params.device_id),
          params_(std::move(params)),
          moe_kernel_(
              llaminar::v2::kernels::KernelFactory::createMoEKernel(
                  params_.device_id))
    {
        if (!validate())
        {
            throw std::invalid_argument(
                "MoEOverlayDeviceControllerStage requires an exact GPU role/action binding");
        }
        if (!moe_kernel_)
        {
            throw std::runtime_error(
                "MoEOverlayDeviceControllerStage could not create its backend kernel");
        }
    }

    bool MoEOverlayDeviceControllerStage::validate() const noexcept
    {
        if (!params_.device_id.is_gpu() || !params_.binding.valid() ||
            params_.binding.device != params_.device_id ||
            params_.action == MoEOverlayDeviceControllerAction::Invalid ||
            params_.workspace_name.empty() || params_.stage_name.empty() ||
            (requiresLeader(params_.action) &&
             !params_.binding.authority_leader) ||
            (requiresGroupRoot(params_.action) &&
             !params_.binding.group_root))
        {
            return false;
        }
        const bool begin =
            params_.action ==
            MoEOverlayDeviceControllerAction::BeginTransaction;
        const bool valid_kind =
            params_.transaction_kind >=
                MoEOverlayDeviceControllerTransactionKind::StaticCheck &&
            params_.transaction_kind <=
                MoEOverlayDeviceControllerTransactionKind::CurrentBatchLLEP;
        const bool valid_phase =
            params_.action ==
                    MoEOverlayDeviceControllerAction::AuthorDynamicPolicy
                ? params_.demand_phase ==
                          MoEOverlayDeviceDemandPhase::Prefill ||
                      params_.demand_phase ==
                          MoEOverlayDeviceDemandPhase::Decode
                : params_.demand_phase ==
                      MoEOverlayDeviceDemandPhase::Invalid;
        return valid_phase &&
               (begin ? valid_kind
                      : params_.transaction_kind ==
                            MoEOverlayDeviceControllerTransactionKind::Invalid);
    }

    bool MoEOverlayDeviceControllerStage::execute(IDeviceContext *ctx)
    {
        if (!ensureContext(ctx, "MoEOverlayDeviceControllerStage") ||
            !validate() || !moe_kernel_)
        {
            return false;
        }
        void *const stream = requireGPUStream();
        if (!stream)
            return false;

        MoEOverlayDeviceControllerPolicyResult *policy_result = nullptr;
        if (params_.action ==
                MoEOverlayDeviceControllerAction::PublishCommand ||
            params_.action ==
                MoEOverlayDeviceControllerAction::AuthorStaticPolicy ||
            params_.action ==
                MoEOverlayDeviceControllerAction::AuthorDynamicPolicy)
        {
            if (!workspace_)
            {
                LOG_ERROR("[MoEOverlayDeviceControllerStage] PublishCommand has no persistent device workspace");
                return false;
            }
            policy_result = static_cast<
                MoEOverlayDeviceControllerPolicyResult *>(
                workspace_->getBuffer(policyResultBufferName()));
            if (!policy_result)
            {
                LOG_ERROR("[MoEOverlayDeviceControllerStage] PublishCommand cannot resolve its device-authored policy result");
                return false;
            }
        }

        return moe_kernel_->runMoEOverlayDeviceControllerAction(
            {.stream = stream, .workspace = workspace_},
            {
                .binding = params_.binding.deviceBinding(),
                .action = params_.action,
                .transaction_kind = params_.transaction_kind,
                .demand_phase = params_.demand_phase,
                .policy_result = policy_result,
            });
    }

    size_t MoEOverlayDeviceControllerStage::estimatedMemoryBytes() const
    {
        size_t bytes = sizeof(MoEOverlayDeviceControllerSharedHeader) +
                       sizeof(MoEOverlayDeviceControllerCommandHeader) +
                       sizeof(MoEOverlayDeviceControllerGroupRecord);
        if (params_.action ==
            MoEOverlayDeviceControllerAction::PublishGroupSnapshot)
        {
            const auto &group = params_.binding.groups[
                static_cast<std::size_t>(params_.binding.group_id)];
            bytes += static_cast<size_t>(group.collected_state_words) *
                     sizeof(std::uint64_t);
        }
        return bytes;
    }

    bool MoEOverlayDeviceControllerStage::supportsBackend(
        ComputeBackendType backend) const
    {
        return backend == ComputeBackendType::GPU_CUDA ||
               backend == ComputeBackendType::GPU_ROCM;
    }

    WorkspaceRequirements
    MoEOverlayDeviceControllerStage::getWorkspaceRequirements(
        int m,
        int n,
        int k) const
    {
        (void)m;
        (void)n;
        (void)k;
        WorkspaceRequirements requirements;
        if (params_.action ==
                MoEOverlayDeviceControllerAction::PublishCommand ||
            params_.action ==
                MoEOverlayDeviceControllerAction::AuthorStaticPolicy ||
            params_.action ==
                MoEOverlayDeviceControllerAction::AuthorDynamicPolicy)
        {
            requirements.buffers.push_back({
                policyResultBufferName(),
                sizeof(MoEOverlayDeviceControllerPolicyResult),
                alignof(MoEOverlayDeviceControllerPolicyResult),
                true,
            });
        }
        return requirements;
    }

    void MoEOverlayDeviceControllerStage::bindWorkspace(
        DeviceWorkspaceManager *workspace)
    {
        workspace_ = workspace;
    }

    void MoEOverlayDeviceControllerStage::unbindWorkspace()
    {
        workspace_ = nullptr;
    }

    std::string MoEOverlayDeviceControllerStage::policyResultBufferName()
        const
    {
        return params_.workspace_name + "_" + kPolicyResultBuffer;
    }

    bool MoEOverlayDeviceControllerStage::hasSameCaptureIdentity(
        const Params &other) const noexcept
    {
        const auto &left = params_.binding;
        const auto &right = other.binding;
        return params_.device_id == other.device_id &&
               params_.action == other.action &&
               params_.transaction_kind == other.transaction_kind &&
               params_.demand_phase == other.demand_phase &&
               params_.workspace_name == other.workspace_name &&
               left.device == right.device &&
               left.participant_id == right.participant_id &&
               left.group_id == right.group_id &&
               left.authority_leader == right.authority_leader &&
               left.group_root == right.group_root &&
               left.mapped_base_device == right.mapped_base_device &&
               left.mapped_bytes == right.mapped_bytes &&
               left.layout == right.layout &&
               left.controller == right.controller &&
               left.command == right.command &&
               left.command_entries == right.command_entries &&
               left.payload_bytes_per_layer ==
                   right.payload_bytes_per_layer &&
               left.demand_history == right.demand_history &&
               left.economy == right.economy &&
               left.economy_service_costs ==
                   right.economy_service_costs &&
               left.economy_migration_costs ==
                   right.economy_migration_costs &&
               left.economy_last_moved == right.economy_last_moved &&
               left.local_group == right.local_group &&
               left.local_transport == right.local_transport &&
               left.group_participant_records ==
                   right.group_participant_records &&
               left.local_participant_record ==
                   right.local_participant_record &&
               left.group_collected_state == right.group_collected_state &&
               left.participant_collected_state ==
                   right.participant_collected_state;
    }

    StageDumpInfo MoEOverlayDeviceControllerStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;
        info.addScalarInt(
                "action", static_cast<int>(params_.action))
            .addScalarInt(
                "transaction_kind",
                static_cast<int>(params_.transaction_kind))
            .addScalarInt(
                "demand_phase", static_cast<int>(params_.demand_phase))
            .addScalarInt(
                "participant_id", params_.binding.participant_id)
            .addScalarInt("group_id", params_.binding.group_id)
            .addScalarBool(
                "authority_leader", params_.binding.authority_leader)
            .addScalarBool("group_root", params_.binding.group_root);
        return info;
    }
} // namespace llaminar2
