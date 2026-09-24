/**
 * @file GDNLiveStateAllGatherStage.h
 * @brief LocalTP handoff from linked local GDN state to mirrored decode state.
 *
 * Production GDN weights and state use dependency-closed modulo-linked head
 * ownership. This stage exports participant-local state, performs a captured
 * equal-count raw allgather, reassembles the result into global semantic-group
 * order on device, and imports the full bank used by replicated decode.
 */

#pragma once

#include "../IComputeStage.h"
#include "../StageParamsBase.h"
#include "GDNLinkedLiveStateGeometry.h"
#include "../../../interfaces/IWorkspaceConsumer.h"

#include <string>

namespace llaminar2
{
    class ILocalTPContext;
    class ITensorGatedDeltaNet;
    class ITensorShortConvolution;

    /**
     * @brief Publish TP-local GDN kernel state as full mirrored state.
     *
     * The collective and both layout-conversion kernels run on the exact graph
     * stream. No host mirror, synchronization, allocation, or eager replay is
     * permitted. The typed geometry is the sole size/layout authority.
     */
    class GDNLiveStateAllGatherStage : public IComputeStage,
                                       public IWorkspaceConsumer
    {
    public:
        static constexpr const char *WS_LOCAL_CONV_STATE =
            "gdn_live_state_local_conv";
        static constexpr const char *WS_GATHERED_CONV_STATE =
            "gdn_live_state_gathered_conv";
        static constexpr const char *WS_FULL_CONV_STATE =
            "gdn_live_state_full_conv";
        static constexpr const char *WS_LOCAL_RECURRENCE_STATE =
            "gdn_live_state_local_recurrence";
        static constexpr const char *WS_GATHERED_RECURRENCE_STATE =
            "gdn_live_state_gathered_recurrence";
        static constexpr const char *WS_FULL_RECURRENCE_STATE =
            "gdn_live_state_full_recurrence";

        /** @brief Complete construction contract for one graph-local handoff. */
        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;

            ILocalTPContext *tp_ctx = nullptr; ///< Exact LocalTP collective owner.
            ITensorShortConvolution *conv_kernel = nullptr; ///< Conv state owner.
            ITensorGatedDeltaNet *recurrence_kernel = nullptr; ///< Recurrence state owner.
            int layer_idx = -1; ///< Logical model layer for workspace identity.
            int tp_device_idx = -1; ///< Participant index within `tp_ctx`.
            GDNLinkedLiveStateGeometry geometry; ///< Sole state layout authority.
            std::string stage_name; ///< Stable collective/diagnostic namespace.
        };

        static_assert(StageParamsRequired<Params>);

        /** @brief Construct a live-state handoff from its complete typed contract. */
        explicit GDNLiveStateAllGatherStage(Params params);

        /** @brief Enqueue both state gathers and device-side reassemblies. */
        bool execute(IDeviceContext *ctx) override;

        /** @brief Return the stable stage type used by graph diagnostics. */
        ComputeStageType type() const override
        {
            return ComputeStageType::GDN_LIVE_STATE_ALLGATHER;
        }

        /** @brief Layout conversion has no model FLOPs. */
        size_t estimatedFlops() const override { return 0; }

        /** @brief Return persistent workspace bytes derived from typed geometry. */
        size_t estimatedMemoryBytes() const override;

        /** @brief Report CUDA/ROCm support compiled into this binary. */
        bool supportsBackend(ComputeBackendType backend) const override;

        /** @brief Describe resolved TP geometry for stage-dump diagnostics. */
        StageDumpInfo buildDumpInfoImpl() const override;

        /** @brief This state-only stage has no tensor-buffer inputs or outputs. */
        StageBufferRequirements getBufferRequirements() const override;

        /** @brief Return the empty tensor-buffer contract for state-only work. */
        StageBufferContract bufferContract() const override;

        /** @brief Kernel-owned state uses explicit stream order, not tensor coherence. */
        CoherencePolicy coherencePolicy() const override
        {
            return CoherencePolicy::NONE;
        }

        /** @brief Validate that the complete allgather/reassembly path is capturable. */
        bool isGraphCapturable() const override;

        /** @brief Declare persistent local, gathered, and full state workspaces. */
        WorkspaceRequirements getWorkspaceRequirements(
            int m,
            int n = 0,
            int k = 0) const override;

        /** @brief Bind the graph-owned persistent workspace before capture. */
        void bindWorkspace(DeviceWorkspaceManager *workspace) override;

        /** @brief Release the non-owning workspace binding after graph retirement. */
        void unbindWorkspace() override;

        /** @brief Return whether persistent workspace has been bound. */
        bool hasWorkspace() const override { return bound_workspace_ != nullptr; }

        /** @brief Return the currently bound non-owning workspace pointer. */
        DeviceWorkspaceManager *getWorkspace() const override
        {
            return bound_workspace_;
        }

        /** @brief Expose the immutable construction contract to graph tests. */
        const Params &getParams() const { return params_; }

    private:
        Params params_;
        DeviceWorkspaceManager *bound_workspace_ = nullptr;

        /** @brief Return a layer-unique local conv workspace name. */
        std::string localConvBufferName() const;

        /** @brief Return a layer-unique raw gathered conv workspace name. */
        std::string gatheredConvBufferName() const;

        /** @brief Return a layer-unique full conv workspace name. */
        std::string fullConvBufferName() const;

        /** @brief Return a layer-unique local recurrence workspace name. */
        std::string localRecurrenceBufferName() const;

        /** @brief Return a layer-unique raw gathered recurrence workspace name. */
        std::string gatheredRecurrenceBufferName() const;

        /** @brief Return a layer-unique full recurrence workspace name. */
        std::string fullRecurrenceBufferName() const;

        /** @brief Execute the complete state publication on the active stream. */
        bool runLiveStateAllGather(const char *context);

        /**
         * @brief Launch the backend-specific group-major reassembly kernel.
         * @param gathered Rank-major raw allgather result.
         * @param full Group-major destination bank.
         * @param shape Fully resolved layout for this state kind.
         * @param stream Exact producer/consumer stream.
         */
        bool reassembleOnDevice(
            const float *gathered,
            float *full,
            const GDNLinkedLiveStateShape &shape,
            void *stream) const;
    };
} // namespace llaminar2
