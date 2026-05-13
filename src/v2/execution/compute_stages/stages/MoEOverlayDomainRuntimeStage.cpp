/**
 * @file MoEOverlayDomainRuntimeStage.cpp
 * @brief Implementation of the MoE overlay domain runtime stage.
 */

#include "MoEOverlayDomainRuntimeStage.h"

#include "../../../interfaces/IWorkspaceConsumer.h"
#include "../../../utils/DebugEnv.h"
#include "../../../utils/Logger.h"

#include <utility>

namespace llaminar2
{
    namespace
    {
        void forEachParticipantPreparedGemm(
            const MoEExpertOverlayLocalTPPreparedParticipant &participant,
            const std::function<void(ITensorGemm *)> &visitor)
        {
            for (auto *gemm : participant.gate_gemm)
                visitor(gemm);
            for (auto *gemm : participant.up_gemm)
                visitor(gemm);
            for (auto *gemm : participant.down_gemm)
                visitor(gemm);
        }

        void forEachPreparedGemm(
            const MoEExpertOverlayLocalTPStage::Params &params,
            const std::function<void(ITensorGemm *)> &visitor)
        {
            for (const auto &participant : params.prepared_participants)
                forEachParticipantPreparedGemm(participant, visitor);
        }

        WorkspaceRequirements firstParticipantWorkspaceRequirements(
            const MoEExpertOverlayLocalTPPreparedParticipant &participant,
            int m,
            int n,
            int k)
        {
            WorkspaceRequirements requirements;
            forEachParticipantPreparedGemm(participant, [&](ITensorGemm *gemm) {
                if (!gemm || !requirements.buffers.empty())
                    return;
                auto *consumer = dynamic_cast<IWorkspaceConsumer *>(gemm);
                if (consumer)
                    requirements = consumer->getWorkspaceRequirements(m, n, k);
            });
            return requirements;
        }

        void bindParticipantPreparedGemm(
            const MoEExpertOverlayLocalTPPreparedParticipant &participant,
            DeviceWorkspaceManager *workspace)
        {
            forEachParticipantPreparedGemm(participant, [workspace](ITensorGemm *gemm) {
                if (!gemm)
                    return;
                auto *consumer = dynamic_cast<IWorkspaceConsumer *>(gemm);
                if (consumer)
                    consumer->bindWorkspace(workspace);
            });
        }
    } // namespace

    MoEOverlayDomainRuntimeStage::MoEOverlayDomainRuntimeStage(Params params)
        : IComputeStage(params.device_id), params_(std::move(params))
    {
        if (!params_.result && params_.result_lifetime)
            params_.result = params_.result_lifetime.get();
    }

    bool MoEOverlayDomainRuntimeStage::execute(IDeviceContext *ctx)
    {
        if (!ctx)
        {
            LOG_ERROR("[MoEOverlayDomainRuntimeStage] Null device context");
            return false;
        }
        if (!params_.runtime)
        {
            LOG_ERROR("[MoEOverlayDomainRuntimeStage] Missing overlay domain runtime service");
            return false;
        }

        const bool trace = debugEnv().moe_expert_overlay.trace;
        if (trace)
        {
            LOG_INFO("[MoEOverlayDomainRuntimeStage][trace] layer=" << params_.request.layer_idx
                     << " tier='" << params_.request.tier_name
                     << "' domain='" << params_.request.runtime_domain.name
                     << "' submit begin");
        }
        auto result = params_.runtime->submit(params_.request, ctx);
        if (trace)
        {
            LOG_INFO("[MoEOverlayDomainRuntimeStage][trace] layer=" << params_.request.layer_idx
                     << " tier='" << params_.request.tier_name
                     << "' domain='" << params_.request.runtime_domain.name
                     << "' submit done ok=" << (result.ok ? "true" : "false")
                     << " kind=" << toString(result.dispatch_kind));
        }
        if (params_.result)
            *params_.result = result;
        if (!result.ok)
        {
            LOG_ERROR("[MoEOverlayDomainRuntimeStage] Domain '" << params_.request.runtime_domain.name
                      << "' tier='" << params_.request.tier_name
                      << "' failed through runtime kind=" << toString(result.dispatch_kind)
                      << ": " << result.error);
        }
        return result.ok;
    }

    size_t MoEOverlayDomainRuntimeStage::estimatedFlops() const
    {
        if (params_.request.has_local_tp_params)
            return MoEExpertOverlayLocalTPStage(params_.request.local_tp_params).estimatedFlops();
        if (params_.request.has_cpu_fallback_params)
            return MoEExpertOverlayCPUFallbackStage(params_.request.cpu_fallback_params).estimatedFlops();
        if (params_.request.has_compute_params)
            return MoEExpertComputeStage(params_.request.compute_params).estimatedFlops();
        return 0;
    }

    size_t MoEOverlayDomainRuntimeStage::estimatedMemoryBytes() const
    {
        if (params_.request.has_local_tp_params)
            return MoEExpertOverlayLocalTPStage(params_.request.local_tp_params).estimatedMemoryBytes();
        if (params_.request.has_cpu_fallback_params)
            return MoEExpertOverlayCPUFallbackStage(params_.request.cpu_fallback_params).estimatedMemoryBytes();
        if (params_.request.has_compute_params)
            return MoEExpertComputeStage(params_.request.compute_params).estimatedMemoryBytes();
        return 0;
    }

    bool MoEOverlayDomainRuntimeStage::supportsBackend(ComputeBackendType backend) const
    {
        if (params_.request.has_local_tp_params)
            return MoEExpertOverlayLocalTPStage(params_.request.local_tp_params).supportsBackend(backend);
        if (params_.request.has_cpu_fallback_params)
            return MoEExpertOverlayCPUFallbackStage(params_.request.cpu_fallback_params).supportsBackend(backend);
        if (params_.request.has_compute_params)
            return MoEExpertComputeStage(params_.request.compute_params).supportsBackend(backend);
        return backend == ComputeBackendType::CPU || backend == ComputeBackendType::GPU_CUDA || backend == ComputeBackendType::GPU_ROCM;
    }

    StageBufferRequirements MoEOverlayDomainRuntimeStage::getBufferRequirements() const
    {
        if (params_.request.has_local_tp_params)
            return MoEExpertOverlayLocalTPStage(params_.request.local_tp_params).getBufferRequirements();
        if (params_.request.has_cpu_fallback_params)
            return MoEExpertOverlayCPUFallbackStage(params_.request.cpu_fallback_params).getBufferRequirements();
        if (params_.request.has_compute_params)
            return MoEExpertComputeStage(params_.request.compute_params).getBufferRequirements();
        return {};
    }

    StageDumpInfo MoEOverlayDomainRuntimeStage::buildDumpInfoImpl() const
    {
        if (params_.request.has_local_tp_params)
            return MoEExpertOverlayLocalTPStage(params_.request.local_tp_params).getDumpInfo();
        if (params_.request.has_cpu_fallback_params)
            return MoEExpertOverlayCPUFallbackStage(params_.request.cpu_fallback_params).getDumpInfo();
        if (params_.request.has_compute_params)
            return MoEExpertComputeStage(params_.request.compute_params).getDumpInfo();
        return {};
    }

    WorkspaceRequirements MoEOverlayDomainRuntimeStage::getWorkspaceRequirements(int m, int n, int k) const
    {
        if (params_.request.has_local_tp_params)
        {
            for (const auto &participant : params_.request.local_tp_params.prepared_participants)
            {
                auto requirements = firstParticipantWorkspaceRequirements(participant, m, n, k);
                if (!requirements.buffers.empty())
                    return requirements;
            }
        }
        if (params_.request.has_compute_params)
        {
            MoEExpertComputeStage stage(params_.request.compute_params);
            if (auto *consumer = dynamic_cast<IWorkspaceConsumer *>(&stage))
                return consumer->getWorkspaceRequirements(m, n, k);
        }
        return WorkspaceRequirements{};
    }

    void MoEOverlayDomainRuntimeStage::bindWorkspace(DeviceWorkspaceManager *workspace)
    {
        bound_workspace_ = workspace;
        owned_participant_workspaces_.clear();

        if (!workspace)
        {
            unbindWorkspace();
            return;
        }

        if (!params_.request.has_local_tp_params)
        {
            if (params_.request.has_compute_params)
            {
                MoEExpertComputeStage stage(params_.request.compute_params);
                if (auto *consumer = dynamic_cast<IWorkspaceConsumer *>(&stage))
                    consumer->bindWorkspace(workspace);
            }
            return;
        }

        const size_t budget = workspace->budget();
        auto &local_tp_params = params_.request.local_tp_params;
        size_t participant_workspace_count = 0;
        for (const auto &participant : local_tp_params.prepared_participants)
        {
            DeviceWorkspaceManager *participant_workspace = workspace;
            if (participant.device != workspace->device())
            {
                auto requirements = firstParticipantWorkspaceRequirements(participant, 4096, 0, 0);
                auto owned_workspace = std::make_unique<DeviceWorkspaceManager>(participant.device, budget);
                if (!owned_workspace->allocate(requirements))
                {
                    LOG_ERROR("[MoEOverlayDomainRuntimeStage] Failed to allocate participant workspace for domain '"
                              << local_tp_params.domain.name << "' participant=" << participant.participant_index
                              << " device=" << participant.device.to_string());
                    continue;
                }
                participant_workspace = owned_workspace.get();
                owned_participant_workspaces_.push_back(std::move(owned_workspace));
                ++participant_workspace_count;
            }

            bindParticipantPreparedGemm(participant, participant_workspace);
        }

        LOG_DEBUG("[MoEOverlayDomainRuntimeStage] Bound workspaces for "
                  << local_tp_params.prepared_participants.size()
                  << " prepared LocalTP participants in domain '"
                  << local_tp_params.domain.name
                  << "' owned_extra_workspaces=" << participant_workspace_count);
    }

    void MoEOverlayDomainRuntimeStage::unbindWorkspace()
    {
        if (params_.request.has_local_tp_params)
        {
            for (const auto &participant : params_.request.local_tp_params.prepared_participants)
                bindParticipantPreparedGemm(participant, nullptr);
        }
        owned_participant_workspaces_.clear();
        bound_workspace_ = nullptr;
    }

    bool MoEOverlayDomainRuntimeStage::hasWorkspace() const
    {
        if (!bound_workspace_)
            return false;

        if (params_.request.has_local_tp_params)
        {
            bool any_bound = false;
            forEachPreparedGemm(params_.request.local_tp_params, [&](ITensorGemm *gemm) {
                if (any_bound || !gemm)
                    return;
                auto *consumer = dynamic_cast<IWorkspaceConsumer *>(gemm);
                any_bound = consumer && consumer->hasWorkspace();
            });
            return any_bound;
        }
        return false;
    }

} // namespace llaminar2
