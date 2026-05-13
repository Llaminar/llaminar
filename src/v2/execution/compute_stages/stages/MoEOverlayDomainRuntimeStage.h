/**
 * @file MoEOverlayDomainRuntimeStage.h
 * @brief Graph stage that submits MoE overlay tier work through IOverlayDomainRuntime.
 */

#pragma once

#include "../IComputeStage.h"
#include "../StageParamsBase.h"
#include "../../local_execution/device/DeviceWorkspaceManager.h"
#include "../../moe/IOverlayDomainRuntime.h"
#include "../../../interfaces/IWorkspaceConsumer.h"

#include <memory>
#include <vector>

namespace llaminar2
{
    class MoEOverlayDomainRuntimeStage : public IComputeStage, public IWorkspaceConsumer
    {
    public:
        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;

            std::shared_ptr<IOverlayDomainRuntime> runtime;
            MoEOverlayDomainWorkRequest request;
            MoEOverlayDomainWorkResult *result = nullptr;
            std::shared_ptr<MoEOverlayDomainWorkResult> result_lifetime;
        };

        static_assert(StageParamsRequired<Params>);

        explicit MoEOverlayDomainRuntimeStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::MOE_EXPERT_FFN; }
        std::string name() const override { return "moe_overlay_domain_runtime"; }
        size_t estimatedFlops() const override;
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        bool isGraphCapturable() const override { return false; }
        bool allowsZeroOutput() const override { return true; }
        CoherencePolicy coherencePolicy() const override { return CoherencePolicy::NONE; }
        StageBufferRequirements getBufferRequirements() const override;
        StageDumpInfo buildDumpInfoImpl() const override;
        WorkspaceRequirements getWorkspaceRequirements(int m, int n = 0, int k = 0) const override;
        void bindWorkspace(DeviceWorkspaceManager *workspace) override;
        void unbindWorkspace() override;
        bool hasWorkspace() const override;
        DeviceWorkspaceManager *getWorkspace() const override { return bound_workspace_; }

        const Params &params() const { return params_; }

    private:
        Params params_;
        DeviceWorkspaceManager *bound_workspace_ = nullptr;
        std::vector<std::unique_ptr<DeviceWorkspaceManager>> owned_participant_workspaces_;
    };

} // namespace llaminar2
