/**
 * @file MoESparseReturnReduceStage.h
 * @brief Graph-native sparse MoE return/reduce payload stage.
 */

#pragma once

#include "../IComputeStage.h"
#include "../StageParamsBase.h"
#include "../../../memory/BufferId.h"
#include "../../moe/MoEOverlaySparseCollective.h"
#include "MoEExpertDispatchStage.h"

#include <optional>

namespace llaminar2
{
    class ITPContext;
    class TensorBase;

    /**
     * @brief Return compact expert output rows to the continuation participant.
     *
     * This is the paired manual boundary for @ref MoESparseDispatchStage. It
     * uses the same runner-stamped transaction identity so independently
     * materialized root and participant graphs agree on the exact sparse wire
     * operation after graph capture and replay.
     */
    class MoESparseReturnReduceStage : public IComputeStage
    {
    public:
        /** @brief Authority that may consume rows received by this boundary. */
        enum class InboundConsumerRole : uint8_t
        {
            ContinuationAccumulator, ///< Scatter rows into the continuation tensor or ticket.
            ProtocolParticipant,     ///< Enter the collective but own no returned dense state.
            /**
             * Colocated CPU producer already published compact canonical rows;
             * this boundary certifies that publication without dense scatter.
             */
            CanonicalRouteTicketCompletion,
        };

        /** @brief Immutable construction and ownership contract for return/reduce. */
        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;

            IMoEOverlaySparseCollectiveContext *collective_context = nullptr;
            std::shared_ptr<IMoEOverlaySparseCollectiveContext> collective_context_lifetime;
            std::shared_ptr<MoEOverlayCollectiveWorkspace> workspace_lifetime;
            MoEOverlayCollectiveKey key;
            int source_participant = -1;
            int target_participant = -1;
            const MoEOverlayReturnRows *outbound_rows = nullptr;
            std::shared_ptr<const MoEOverlayReturnRows> outbound_rows_lifetime;
            MoEOverlayReturnRows *inbound_rows = nullptr;
            std::shared_ptr<MoEOverlayReturnRows> inbound_rows_lifetime;
            TensorBase *dense_output = nullptr;
            /**
             * @brief Whether this rank owns the returned dense activation.
             *
             * Auxiliary ExpertOverlay ranks select ProtocolParticipant. They
             * must still enter the matched return collective, but allocating
             * and clearing a capacity-wide dummy tensor on every layer would
             * create fake state and needless hot-path work.
             */
            InboundConsumerRole inbound_consumer_role =
                InboundConsumerRole::ContinuationAccumulator;
            /** Fixed pinned return destination for a following captured H2D ingress. */
            std::shared_ptr<MoEOverlayDispatchTicketStorage> ticket_storage;
            /** Typed colocated CPU canonical-route publication to certify. */
            std::shared_ptr<MoEOverlayCanonicalRouteReturnTicketStorage>
                canonical_route_ticket_storage;
            /**
             * @brief Publish completion of the one fully accumulated ticket.
             *
             * A distributed return has one source participant per sparse
             * target, so source-id equality cannot identify the final packet.
             * Graph lowering sets this flag on exactly the last ordered return
             * after every preceding scatter has completed.
             */
            bool publish_ticket_completion = false;
            std::optional<BufferId> dense_output_buffer_id;
            int seq_len = 0;
            int d_model = 0;
            bool clear_output_before_scatter = false;
            bool manual_boundary_requires_collective_completion = true;

            /** Require a runner-stamped identity for graph-native multi-rank use. */
            bool require_explicit_transaction_identity = false;
            /** Require typed decode/prefill semantics for this production boundary. */
            bool require_explicit_execution_semantics = false;

            ITPContext *continuation_tp_context = nullptr;
            bool broadcast_after_scatter = false;
            int continuation_root_tp_index = 0;

            /**
             * Exact dispatch descriptor whose residency lease may be released
             * after this stage completes the final continuation-root return.
             */
            std::shared_ptr<MoEExpertDispatchOutput>
                dispatch_output_lifetime;
            /** Typed terminal authority for the host dispatch epoch lease. */
            MoEOverlayHostDispatchLeaseTerminal residency_lease_terminal =
                MoEOverlayHostDispatchLeaseTerminal::Retain;
        };

        static_assert(StageParamsRequired<Params>);

        /** @brief Construct one return/reduce boundary with stable graph-owned views. */
        explicit MoESparseReturnReduceStage(Params params);

        /**
         * @brief Enter the paired sparse return collective and consume rows only
         *        on the declared continuation authority.
         */
        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::MOE_SPARSE_RETURN_REDUCE; }
        std::string name() const override { return "moe_sparse_return_reduce"; }
        bool supportsBackend(ComputeBackendType backend) const override;
        bool isGraphCapturable() const override { return false; }
        bool isManualGraphBoundary() const override { return true; }
        /**
         * @brief Admit overlap only for the mapped canonical-ticket terminal.
         *
         * Dense return scattering and portable collectives require host launch
         * sequencing. The canonical role merely validates the publication
         * already produced by its colocated CPU expert stage.
         */
        ManualGraphBoundaryScheduling
        manualGraphBoundaryScheduling() const noexcept override
        {
            return params_.canonical_route_ticket_storage &&
                           params_.inbound_consumer_role ==
                               InboundConsumerRole::CanonicalRouteTicketCompletion
                       ? ManualGraphBoundaryScheduling::ConcurrentTicketService
                       : ManualGraphBoundaryScheduling::BetweenExecutableLaunches;
        }
        bool supportsPaddedPrefillGraphCapturePreflight() const override
        {
            return true;
        }
        bool supportsPaddedPrefillRealLengthContract() const override
        {
            return true;
        }
        bool hasMoEOverlayCollectiveRuntimeParams() const override
        {
            return params_.require_explicit_transaction_identity ||
                   params_.require_explicit_execution_semantics;
        }
        /** @brief Store the root-authoritative identity for the next execution. */
        void updateMoEOverlayCollectiveRuntimeParams(
            const MoEOverlayCollectiveRuntimeParams &params) override;
        bool manualGraphBoundaryComplete() const override
        {
            const bool collective_complete =
                last_collective_result_.ok &&
                (!params_.manual_boundary_requires_collective_completion ||
                 last_collective_result_.collective_complete);
            if (!collective_complete)
                return false;
            if (params_.ticket_storage &&
                params_.publish_ticket_completion &&
                params_.manual_boundary_requires_collective_completion &&
                !params_.ticket_storage->ticket().returnPayloadReady())
            {
                return false;
            }
            return !params_.canonical_route_ticket_storage ||
                   (params_.outbound_rows &&
                    params_.canonical_route_ticket_storage
                        ->publicationSucceededFor(
                        params_.outbound_rows->residency_epoch));
        }
        bool allowsZeroOutput() const override { return true; }
        CoherencePolicy coherencePolicy() const override { return CoherencePolicy::NONE; }
        StageBufferRequirements getBufferRequirements() const override;
        StageBufferContract bufferContract() const override;
        StageDumpInfo buildDumpInfoImpl() const override;

        /** @brief Return the immutable construction contract for diagnostics. */
        const Params &params() const { return params_; }

    private:
        Params params_;
        MoEOverlayCollectiveResult last_collective_result_{};
        /** Runner-stamped identity used only by required distributed graphs. */
        MoEOverlayCollectiveRuntimeParams runtime_params_{};
        /** Legacy local-fixture counter; never used by required distributed overlays. */
        uint64_t execution_count_ = 0;
    };

} // namespace llaminar2
