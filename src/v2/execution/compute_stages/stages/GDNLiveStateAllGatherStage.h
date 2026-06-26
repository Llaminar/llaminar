/**
 * @file GDNLiveStateAllGatherStage.h
 * @brief LocalTP handoff from TP-local GDN live state to mirrored decode state.
 */

#pragma once

#include "../IComputeStage.h"
#include "../StageParamsBase.h"
#include "../../../interfaces/IWorkspaceConsumer.h"

#include <string>

namespace llaminar2
{
    class ILocalTPContext;
    class ITensorGatedDeltaNet;
    class ITensorShortConvolution;

    /**
     * @brief Gather TP-local GDN kernel state into full mirrored state.
     *
     * Prefill can keep GDN projections/state TP-local. Phase-split decode with
     * replicated dense weights needs every participant to own the full GDN
     * short-conv and recurrence state so it can compute full dense attention
     * output without a tiny output allreduce. This stage is the explicit
     * production-graph handoff between those two modes.
     */
    class GDNLiveStateAllGatherStage : public IComputeStage, public IWorkspaceConsumer
    {
    public:
        static constexpr const char *WS_LOCAL_CONV_STATE = "gdn_live_state_local_conv";
        static constexpr const char *WS_GATHERED_CONV_STATE = "gdn_live_state_gathered_conv";
        static constexpr const char *WS_FULL_CONV_STATE = "gdn_live_state_full_conv";
        static constexpr const char *WS_LOCAL_RECURRENCE_STATE = "gdn_live_state_local_recurrence";
        static constexpr const char *WS_FULL_RECURRENCE_STATE = "gdn_live_state_full_recurrence";

        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;

            ILocalTPContext *tp_ctx = nullptr;
            ITensorShortConvolution *conv_kernel = nullptr;
            ITensorGatedDeltaNet *recurrence_kernel = nullptr;

            int layer_idx = -1;
            int tp_device_idx = -1;
            int local_conv_state_floats = 0;
            int full_conv_state_floats = 0;
            bool modular_conv_state = false;
            int conv_history_len = 0;
            int conv_qk_channels = 0;
            int conv_local_v_channels = 0;
            int conv_full_v_channels = 0;
            int local_recurrence_state_floats = 0;
            int full_recurrence_state_floats = 0;
            std::string stage_name;
        };

        static_assert(StageParamsRequired<Params>);

        explicit GDNLiveStateAllGatherStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::GDN_LIVE_STATE_ALLGATHER; }
        size_t estimatedFlops() const override { return 0; }
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        StageDumpInfo buildDumpInfoImpl() const override;
        StageBufferRequirements getBufferRequirements() const override;
        StageBufferContract bufferContract() const override;
        CoherencePolicy coherencePolicy() const override { return CoherencePolicy::NONE; }
        bool isGraphCapturable() const override;

        WorkspaceRequirements getWorkspaceRequirements(int m, int n = 0, int k = 0) const override;
        void bindWorkspace(DeviceWorkspaceManager *workspace) override;
        void unbindWorkspace() override;
        bool hasWorkspace() const override { return bound_workspace_ != nullptr; }
        DeviceWorkspaceManager *getWorkspace() const override { return bound_workspace_; }

        const Params &getParams() const { return params_; }

    private:
        Params params_;
        DeviceWorkspaceManager *bound_workspace_ = nullptr;

        std::string localConvBufferName() const;
        std::string gatheredConvBufferName() const;
        std::string fullConvBufferName() const;
        std::string localRecurrenceBufferName() const;
        std::string fullRecurrenceBufferName() const;
    };

} // namespace llaminar2
