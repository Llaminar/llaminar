/**
 * @file MoEOverlayTicketPublishStage.h
 * @brief Captured fixed-capacity GPU publication for heterogeneous MoE rows.
 *
 * The stage is the final producer in a native GPU graph segment. It copies the
 * router outputs, normalized hidden rows, and device-owned logical row count to
 * one model-lifetime pinned ticket and then system-release publishes its
 * isolated mapped timeline word. The following explicitly declared manual
 * participant boundary acquires only that word, never the graph terminal.
 */

#pragma once

#include "../IComputeStage.h"
#include "../StageParamsBase.h"
#include "../../../execution/moe/MoEOverlaySparseCollective.h"
#include "../../../memory/BufferId.h"

#include <memory>
#include <optional>

namespace llaminar2
{
    class TensorBase;

    /**
     * @brief Graph-capturable publisher for a fixed-capacity overlay ticket.
     */
    class MoEOverlayTicketPublishStage final : public IComputeStage
    {
    public:
        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;

            TensorBase *hidden = nullptr;
            TensorBase *routing_indices = nullptr;
            TensorBase *routing_weights = nullptr;
            std::optional<BufferId> hidden_buffer_id;
            std::optional<BufferId> routing_indices_buffer_id;
            std::optional<BufferId> routing_weights_buffer_id;
            const int32_t *active_row_count_device = nullptr;
            int layer_idx = -1;
            int bucket_rows = 0;
            int top_k = 0;
            int d_model = 0;
            std::shared_ptr<MoEOverlayDispatchTicketStorage> ticket_storage;
        };

        static_assert(StageParamsRequired<Params>);

        explicit MoEOverlayTicketPublishStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override
        {
            return ComputeStageType::MOE_OVERLAY_TICKET_PUBLISH;
        }
        std::string name() const override
        {
            return "moe_overlay_ticket_publish";
        }
        bool supportsBackend(ComputeBackendType backend) const override;
        bool isGraphCapturable() const override;
        bool supportsGraphCaptureAfterLaunchPreparation() const override;
        bool supportsLazyPrefillGraphCapturePreflight() const override;
        bool supportsPaddedPrefillGraphCapturePreflight() const override;
        bool supportsPaddedPrefillRealLengthContract() const override
        {
            return params_.active_row_count_device != nullptr;
        }
        std::string graphCaptureReadinessDebugString() const override;
        bool prepareGraphLaunch(IDeviceContext *ctx, void *stream) override;
        GraphLaunchPreparationPolicy graphLaunchPreparationPolicy() const override
        {
            return supportsGraphCaptureAfterLaunchPreparation()
                       ? GraphLaunchPreparationPolicy::CaptureAndReplay
                       : GraphLaunchPreparationPolicy::None;
        }
        StageBufferRequirements getBufferRequirements() const override;
        StageBufferContract bufferContract() const override;
        StageDumpInfo buildDumpInfoImpl() const override;

        const Params &params() const noexcept { return params_; }

    private:
        /**
         * @brief Validate immutable geometry and pinned-ticket identity only.
         *
         * Cold prefill preflight runs before BufferArena has published device
         * addresses. It may certify this static contract, but capture readiness
         * still requires @ref hasFixedTicketContract and therefore every exact
         * device pointer.
         */
        bool hasStaticTicketContract() const noexcept;

        /** @brief Validate the complete capture-ready ticket and device bindings. */
        bool hasFixedTicketContract() const noexcept;

        Params params_;
    };

} // namespace llaminar2
