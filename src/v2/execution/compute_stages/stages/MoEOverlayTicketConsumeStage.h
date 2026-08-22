/**
 * @file MoEOverlayTicketConsumeStage.h
 * @brief Captured fixed-capacity GPU ingress for heterogeneous MoE returns.
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
     * @brief Copy one CPU-produced pinned return ticket into a stable GPU output.
     *
     * The H2D node is captured once. Replay observes bytes written by the
     * preceding manual participant segment at the same immutable host address.
     * Cold setup records that node before a request exists; mutable payload
     * readiness is consequently validated only by live direct execution and by
     * the typed manual-segment publication contract, never while recording.
     */
    class MoEOverlayTicketConsumeStage final : public IComputeStage
    {
    public:
        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;

            TensorBase *output = nullptr;
            std::optional<BufferId> output_buffer_id;
            int layer_idx = -1;
            int bucket_rows = 0;
            int d_model = 0;
            std::shared_ptr<MoEOverlayDispatchTicketStorage> ticket_storage;
        };

        static_assert(StageParamsRequired<Params>);

        explicit MoEOverlayTicketConsumeStage(Params params);

        /** @brief Record or enqueue the fixed H2D ingress after its typed boundary. */
        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override
        {
            return ComputeStageType::MOE_OVERLAY_TICKET_CONSUME;
        }
        std::string name() const override
        {
            return "moe_overlay_ticket_consume";
        }
        bool supportsBackend(ComputeBackendType backend) const override;
        bool isGraphCapturable() const override;
        bool supportsGraphCaptureAfterLaunchPreparation() const override;
        bool supportsLazyPrefillGraphCapturePreflight() const override;
        bool supportsPaddedPrefillGraphCapturePreflight() const override;
        bool supportsPaddedPrefillRealLengthContract() const override
        {
            return true;
        }
        std::string graphCaptureReadinessDebugString() const override;
        bool prepareGraphLaunch(IDeviceContext *ctx, void *stream) override;
        GraphLaunchPreparationPolicy graphLaunchPreparationPolicy() const override
        {
            return supportsGraphCaptureAfterLaunchPreparation()
                       ? GraphLaunchPreparationPolicy::CaptureOnly
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
         * This is the allocation-free cold-preflight contract. The output's
         * exact GPU address remains mandatory in @ref hasFixedTicketContract
         * before capture or execution.
         */
        bool hasStaticTicketContract() const noexcept;

        /** @brief Validate the complete capture-ready ticket and device binding. */
        bool hasFixedTicketContract() const noexcept;

        Params params_;
    };

} // namespace llaminar2
