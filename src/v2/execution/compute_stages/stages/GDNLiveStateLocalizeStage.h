/**
 * @file GDNLiveStateLocalizeStage.h
 * @brief LocalTP handoff from mirrored GDN decode state to TP-local verifier state.
 *
 * Dense-replicated LocalTP decode keeps a full mirrored GDN live-state bank on
 * every participant so ordinary one-token decode can avoid tiny recurrent-state
 * collectives.  The MTP all-position verifier deliberately runs the economical
 * TP-local GDN graph, so each participant must first slice its local portion out
 * of that mirrored bank.  This stage performs that slice on the graph stream and
 * imports the TP-local state back into the backend kernels before short-conv and
 * recurrence read live state.
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
     * @brief Slice full mirrored GDN state into the current LocalTP participant.
     *
     * The stage is intentionally GPU-only: CPU LocalTP has no mirrored device
     * state bank to slice, and unsupported lanes should fail loudly rather than
     * silently switching verifier semantics.  When the graph inserts this stage,
     * execute() expects the backend kernels to own the full mirrored bank:
     * accepting an already-local bank would let graph capture record a no-op and
     * replay stale TP-local state later.  The stage exports the full bank into
     * workspace, slices the current participant's portion, and imports the local
     * bank with importStateForSize().
     */
    class GDNLiveStateLocalizeStage : public IComputeStage, public IWorkspaceConsumer
    {
    public:
        static constexpr const char *WS_FULL_CONV_STATE = "gdn_live_state_localize_full_conv";
        static constexpr const char *WS_LOCAL_CONV_STATE = "gdn_live_state_localize_local_conv";
        static constexpr const char *WS_FULL_RECURRENCE_STATE = "gdn_live_state_localize_full_recurrence";
        static constexpr const char *WS_LOCAL_RECURRENCE_STATE = "gdn_live_state_localize_local_recurrence";

        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;

            ILocalTPContext *tp_ctx = nullptr;                 ///< LocalTP participant group.
            ITensorShortConvolution *conv_kernel = nullptr;    ///< Backend short-conv state owner.
            ITensorGatedDeltaNet *recurrence_kernel = nullptr; ///< Backend recurrence state owner.

            int layer_idx = -1;      ///< Logical layer, used for stable workspace names.
            int tp_device_idx = -1;  ///< Participant index in the LocalTP group.

            int local_conv_state_floats = 0; ///< TP-local conv-state size.
            int full_conv_state_floats = 0;  ///< Mirrored full conv-state size.
            bool modular_conv_state = false; ///< True when Q/K are mirrored and V is sharded.
            int conv_history_len = 0;        ///< Short-conv history length.
            int conv_qk_channels = 0;        ///< Full mirrored Q+K conv channels.
            int conv_local_v_channels = 0;   ///< V channels owned by one participant.
            int conv_full_v_channels = 0;    ///< Full mirrored V channels.

            int local_recurrence_state_floats = 0; ///< TP-local recurrence state size.
            int full_recurrence_state_floats = 0;  ///< Mirrored full recurrence state size.
            std::string stage_name;                ///< Stable diagnostic/collective name.
        };

        static_assert(StageParamsRequired<Params>);

        explicit GDNLiveStateLocalizeStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::GDN_LIVE_STATE_LOCALIZE; }
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

        std::string fullConvBufferName() const;
        std::string localConvBufferName() const;
        std::string fullRecurrenceBufferName() const;
        std::string localRecurrenceBufferName() const;
        bool runLiveStateLocalize(const char *context);
    };

} // namespace llaminar2
