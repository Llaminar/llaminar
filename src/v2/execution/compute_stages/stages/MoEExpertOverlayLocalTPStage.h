/**
 * @file MoEExpertOverlayLocalTPStage.h
 * @brief Graph stage wrapper for accelerator LocalTP MoE expert-overlay tiers.
 */

#pragma once

#include "../IComputeStage.h"
#include "../StageParamsBase.h"
#include "MoEExpertDispatchStage.h"
#include "../../moe/MoEExpertOverlayLocalTPExecutor.h"
#include "../../local_execution/device/DeviceWorkspaceManager.h"
#include "../../../interfaces/IWorkspaceConsumer.h"

#include <memory>

namespace llaminar2
{

    class MoEExpertOverlayLocalTPStage : public IComputeStage, public IWorkspaceConsumer
    {
    public:
        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;

            MoEOverlayRuntimeDomain domain;
            ILocalTPContext *local_tp_context = nullptr;

            TensorBase *input = nullptr;
            TensorBase *routing_indices = nullptr;
            TensorBase *routing_weights = nullptr;
            TensorBase *gate_exps = nullptr;
            TensorBase *up_exps = nullptr;
            TensorBase *down_exps = nullptr;
            TensorBase *output = nullptr;

            int seq_len = 0;
            int d_model = 0;
            int num_experts = 0;
            int top_k = 0;
            int expert_intermediate = 0;
            int layer_idx = -1;

            std::vector<bool> expert_mask;
            std::string stage_name_prefix = "moe_overlay_localtp";
            bool upload_partials_to_participant_devices = true;
            std::vector<MoEExpertOverlayLocalTPPreparedParticipant> prepared_participants;
            std::vector<FP32Tensor *> prepared_partial_views;
            const MoEExpertDispatchOutput *dispatch_output = nullptr;
            std::shared_ptr<MoEExpertDispatchOutput> dispatch_output_lifetime;
            int dispatch_tier_index = -1;
            MoEExpertOverlayLocalTPDiagnostics *diagnostics = nullptr;
            std::shared_ptr<MoEExpertOverlayLocalTPDiagnostics> diagnostics_lifetime;
            MoEExpertSparseTransferBuffers sparse_transfer_buffers;
        };

        static_assert(StageParamsRequired<Params>);

        explicit MoEExpertOverlayLocalTPStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::MOE_EXPERT_FFN; }
        std::string name() const override { return "moe_expert_overlay_localtp"; }
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