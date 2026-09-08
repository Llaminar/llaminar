/**
 * @file MoESparseDispatchStage.h
 * @brief Graph-native sparse MoE dispatch payload stage.
 */

#pragma once

#include "../IComputeStage.h"
#include "../StageParamsBase.h"
#include "../../../memory/BufferId.h"
#include "MoEExpertDispatchStage.h"
#include "../../moe/MoEOverlaySparseCollective.h"

#include <memory>
#include <optional>

namespace llaminar2
{
    class TensorBase;

    /**
     * @brief Publish compact routed rows to one graph-native MoE participant.
     *
     * This manual graph-boundary stage owns the dispatch half of the sparse
     * collective protocol. It is intentionally CPU-orchestrated because the
     * packet header and variable-length sparse row list cross heterogeneous
     * participants, while the following local-expert stage remains
     * participant-local GPU or CPU work. Distributed production graphs must
     * receive an explicit runtime transaction identity before execution;
     * local unit fixtures may continue to use their isolated stage counter.
     */
    class MoESparseDispatchStage : public IComputeStage
    {
    public:
        /**
         * @brief Whether this graph instance may materialize dispatch bytes.
         *
         * Distributed graphs preserve one logical source participant in the
         * wire key on every MPI rank.  Only the graph that owns that logical
         * source may bind routing tensors or a dispatch descriptor; symmetric
         * peer graphs enter the collective with an explicitly empty packet.
         */
        enum class PayloadPublicationRole : uint8_t
        {
            PayloadAuthority,
            EmptyCollectiveParticipant,
        };

        /**
         * @brief Authority that made a captured ticket host-readable.
         *
         * `DirectCapturedProducer` means this stage is the first host observer
         * and must fence the immediately preceding captured producer.
         * `MaterializedHostDispatch` means an earlier
         * @ref MoEExpertDispatchStage already crossed that exact boundary and
         * published the immutable dispatch output consumed here. Re-fencing a
         * later, independent GPU segment would serialize CPU expert work behind
         * continuation compute and is therefore forbidden.
         */
        enum class TicketObservationRole : uint8_t
        {
            DirectCapturedProducer,
            MaterializedHostDispatch,
        };

        /** @brief Immutable construction and ownership contract for dispatch. */
        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;

            IMoEOverlaySparseCollectiveContext *collective_context = nullptr;
            std::shared_ptr<IMoEOverlaySparseCollectiveContext> collective_context_lifetime;
            MoEOverlayCollectiveWorkspace *workspace = nullptr;
            std::shared_ptr<MoEOverlayCollectiveWorkspace> workspace_lifetime;
            MoEOverlayCollectiveKey key;
            int source_participant = -1;
            int target_participant = -1;

            const TensorBase *hidden = nullptr;
            const TensorBase *routing_indices = nullptr;
            const TensorBase *routing_weights = nullptr;
            std::optional<BufferId> hidden_buffer_id;
            std::optional<BufferId> routing_indices_buffer_id;
            std::optional<BufferId> routing_weights_buffer_id;
            int seq_len = 0;
            int top_k = 0;
            int d_model = 0;

            const MoEExpertTierDispatch *tier_dispatch = nullptr;
            const MoEExpertDispatchOutput *dispatch_output = nullptr;
            std::shared_ptr<MoEExpertDispatchOutput> dispatch_output_lifetime;
            /**
             * Epoch for a directly supplied static tier descriptor.
             *
             * Production root graphs obtain the epoch from dispatch_output;
             * this scalar exists for isolated protocol fixtures whose immutable
             * tier descriptor does not have a full dispatch-output owner.
             */
            uint64_t fixed_residency_epoch = 0;
            std::shared_ptr<MoEOverlayDispatchTicketStorage> ticket_storage;
            TicketObservationRole ticket_observation_role =
                TicketObservationRole::DirectCapturedProducer;
            int tier_index = -1;

            /** Typed authority for payload materialization on this graph. */
            PayloadPublicationRole payload_publication_role =
                PayloadPublicationRole::PayloadAuthority;
            bool replicated_hidden_export = false;
            int logical_continuation_root_participant = -1;
            bool manual_boundary_requires_collective_completion = true;

            /**
             * Require a runner-stamped distributed transaction identity.
             *
             * Graph-native multi-rank overlays set this to true so a captured
             * root graph cannot silently fall back to a local stage counter.
             */
            bool require_explicit_transaction_identity = false;
            /**
             * Require runner-stamped decode/prefill semantics even for a
             * process-local collective. Production overlay graphs enable this
             * so a one-row prefill tail retains its mathematical phase.
             */
            bool require_explicit_execution_semantics = false;

            MoEOverlaySparseRows *inbound_rows = nullptr;
            std::shared_ptr<MoEOverlaySparseRows> inbound_rows_lifetime;
        };

        static_assert(StageParamsRequired<Params>);

        /** @brief Construct one dispatch boundary with stable graph-owned views. */
        explicit MoESparseDispatchStage(Params params);

        /** @brief Validate, pack, and enter this stage's sparse dispatch collective. */
        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::MOE_SPARSE_DISPATCH; }
        std::string name() const override { return "moe_sparse_dispatch"; }
        bool supportsBackend(ComputeBackendType backend) const override;
        bool isGraphCapturable() const override { return false; }
        bool isManualGraphBoundary() const override { return true; }
        /**
         * @brief Admit parent-overlapped service only for host-materialized tickets.
         *
         * This role consumes the descriptor already authenticated by
         * MoEExpertDispatchStage. Direct captured-producer and tensor-backed
         * forms retain ordinary between-executable sequencing.
         */
        ManualGraphBoundaryScheduling
        manualGraphBoundaryScheduling() const noexcept override
        {
            return params_.ticket_storage && params_.dispatch_output &&
                           params_.ticket_observation_role ==
                               TicketObservationRole::MaterializedHostDispatch
                       ? ManualGraphBoundaryScheduling::ConcurrentTicketService
                       : ManualGraphBoundaryScheduling::BetweenExecutableLaunches;
        }
        bool supportsPaddedPrefillGraphCapturePreflight() const override
        {
            return params_.ticket_storage != nullptr;
        }
        bool supportsPaddedPrefillRealLengthContract() const override
        {
            return params_.ticket_storage != nullptr;
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
            return last_collective_result_.ok &&
                   (!params_.manual_boundary_requires_collective_completion ||
                    last_collective_result_.collective_complete);
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
