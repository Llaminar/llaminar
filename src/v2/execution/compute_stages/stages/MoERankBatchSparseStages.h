/**
 * @file MoERankBatchSparseStages.h
 * @brief Graph stages for one rank-batched heterogeneous MoE round trip.
 *
 * These stages are the graph-facing counterpart of the typed rank-batch
 * transport. One dispatch boundary packs every logical participant owned by a
 * remote rank, and one return boundary receives and accumulates those results
 * in ascending participant-id order. Node-local pairs bind the stages directly
 * to shared rows; inter-node pairs retain the explicit MPI wire workspace. The
 * participant-local expert stages remain ordinary per-device graph nodes, so
 * their exact device streams and prepared-weight ownership stay explicit.
 */

#pragma once

#include "../IComputeStage.h"
#include "../StageParamsBase.h"
#include "../../../memory/BufferId.h"
#include "../../moe/MoEOverlayRankBatchTransport.h"
#include "MoEExpertDispatchStage.h"

#include <memory>
#include <optional>
#include <vector>

namespace llaminar2
{
    class ITPContext;
    class TensorBase;

    /** @brief Which side of a direct rank-batch protocol a graph stage owns. */
    enum class MoERankBatchEndpointRole : uint8_t
    {
        ContinuationSource, ///< Dense continuation rank sends dispatch and receives return.
        RemoteTarget,       ///< Remote expert rank receives dispatch and sends return.
    };

    /**
     * @brief Pack or receive one complete remote-rank sparse dispatch batch.
     *
     * A continuation-source instance filters the root dispatch descriptor once
     * per target participant into fixed transport views, then publishes one
     * rank batch. A remote-target instance consumes the exact transport-owned
     * views used by its device-local expert nodes.
     */
    class MoERankBatchDispatchStage final : public IComputeStage
    {
    public:
        /** @brief Immutable graph-construction contract for one dispatch batch. */
        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;

            MoERankBatchEndpointRole endpoint_role =
                MoERankBatchEndpointRole::ContinuationSource;
            std::shared_ptr<IMoEOverlayRankBatchTransport> transport;
            MoEOverlayRankBatchKey key;
            int source_participant = -1;
            std::vector<int> participant_ids;
            std::vector<std::shared_ptr<MoEOverlayCollectiveWorkspace>>
                participant_workspaces;
            std::vector<std::shared_ptr<MoEOverlaySparseRows>> inbound_rows;

            const TensorBase *hidden = nullptr;
            const TensorBase *routing_indices = nullptr;
            const TensorBase *routing_weights = nullptr;
            std::optional<BufferId> hidden_buffer_id;
            std::optional<BufferId> routing_indices_buffer_id;
            std::optional<BufferId> routing_weights_buffer_id;
            std::shared_ptr<MoEExpertDispatchOutput> dispatch_output;
            std::shared_ptr<MoEOverlayDispatchTicketStorage> ticket_storage;

            int seq_len = 0;
            int top_k = 0;
            int d_model = 0;
            int tier_index = -1;
            bool require_explicit_transaction_identity = true;
            bool require_explicit_execution_semantics = true;
        };

        static_assert(StageParamsRequired<Params>);

        /**
         * @brief Bind fixed participant views and validate endpoint ownership.
         * @throws std::invalid_argument for inconsistent rank, participant, or payload roles.
         */
        explicit MoERankBatchDispatchStage(Params params);

        /** @brief Pack/send or receive/publish one rank-batched dispatch envelope. */
        bool execute(IDeviceContext *ctx) override;

        /** @brief Identify the typed rank-batched dispatch operation. */
        ComputeStageType type() const override
        {
            return ComputeStageType::MOE_RANK_BATCH_DISPATCH;
        }

        /** @brief Return the stable diagnostic stage name. */
        std::string name() const override
        {
            return "moe_rank_batch_dispatch";
        }

        /** @brief This host boundary executes only under the CPU executor. */
        bool supportsBackend(ComputeBackendType backend) const override;

        /** @brief Rank-to-rank exchange is an explicit heterogeneous boundary. */
        bool isGraphCapturable() const override { return false; }

        /** @brief Mark the stage as an intentional segmented graph boundary. */
        bool isManualGraphBoundary() const override { return true; }

        /**
         * @brief Return whether this stage must fence a captured ticket producer.
         *
         * The continuation-source rank batch never owns that first host
         * observation. @ref MoEExpertDispatchStage is the single ticket
         * materializer and publishes `dispatch_output` before this dependent
         * stage can run. The rank batch validates that descriptor's exact
         * ticket lifetime and may then pack while an independent captured
         * continuation-local expert branch remains in flight.
         *
         * Returning false here is therefore an overlap contract, not an
         * unfenced device read: the immutable host ticket has already crossed
         * its one declared captured-to-host boundary.
         */
        bool requiresHostGraphTicketFence() const override
        {
            return false;
        }

        /** @brief Fixed ticket storage supports padded prefill capture preflight. */
        bool supportsPaddedPrefillGraphCapturePreflight() const override
        {
            return params_.endpoint_role ==
                       MoERankBatchEndpointRole::RemoteTarget ||
                   params_.ticket_storage != nullptr;
        }

        /** @brief Live ticket counts preserve the real padded-prefill row contract. */
        bool supportsPaddedPrefillRealLengthContract() const override
        {
            return supportsPaddedPrefillGraphCapturePreflight();
        }

        /** @brief Production rank batches always consume runner-stamped identity. */
        bool hasMoEOverlayCollectiveRuntimeParams() const override
        {
            return params_.require_explicit_transaction_identity ||
                   params_.require_explicit_execution_semantics;
        }

        /** @brief Publish request identity and exact decode/prefill/MTP semantics. */
        void updateMoEOverlayCollectiveRuntimeParams(
            const MoEOverlayCollectiveRuntimeParams &params) override;

        /**
         * @brief Report completion of this stage's exact transport handoff.
         *
         * A receive stage completes after authenticated bytes are decoded. A
         * send stage completes once bytes live in a transport-owned fixed slot
         * and the non-blocking request is submitted; the transport retains that
         * slot until `MPI_Test` reports completion. Thus graph progress never
         * depends on recovering ownership of the stage's source views.
         */
        bool manualGraphBoundaryComplete() const override
        {
            return last_result_.ok && last_result_.collective_complete;
        }

        /** @brief Empty participant subpackets are valid outputs. */
        bool allowsZeroOutput() const override { return true; }

        /** @brief Host packet views own their own explicit publication boundary. */
        CoherencePolicy coherencePolicy() const override
        {
            return CoherencePolicy::NONE;
        }

        /** @brief Describe source activation inputs when no captured ticket is bound. */
        StageBufferRequirements getBufferRequirements() const override;

        /** @brief Declare source arena reads for executor dependency validation. */
        StageBufferContract bufferContract() const override;

        /** @brief Expose immutable topology and source tensors to diagnostics. */
        StageDumpInfo buildDumpInfoImpl() const override;

        /** @return Immutable stage construction contract for lowering tests. */
        const Params &params() const noexcept { return params_; }

    private:
        /** @brief Resolve and validate the tier descriptor owned by the root dispatch stage. */
        const MoEExpertTierDispatch *resolveTierDispatch() const;

        /** @brief Pack one target's descriptor entries into its fixed outbound view. */
        bool packParticipant(
            const MoEOverlayRankBatchKey &runtime_key,
            const MoEExpertTierDispatch &tier,
            const float *hidden,
            int logical_seq_len,
            size_t participant_index,
            uint64_t residency_epoch);

        Params params_;
        std::vector<MoEOverlaySparseRows> outbound_rows_;
        std::vector<const MoEOverlaySparseRows *> outbound_views_;
        std::vector<MoEOverlaySparseRows *> inbound_views_;
        MoEOverlayCollectiveResult last_result_{};
        MoEOverlayCollectiveRuntimeParams runtime_params_{};
        uint64_t local_execution_count_ = 0;
    };

    /**
     * @brief Send or receive one complete remote-rank sparse return batch.
     *
     * Remote-target instances publish all completed participant rows in one
     * batch. Continuation-source instances consume those rows and add them to
     * the dense/ticket output strictly in canonical participant order.
     */
    class MoERankBatchReturnReduceStage final : public IComputeStage
    {
    public:
        /** @brief Immutable graph-construction contract for one return batch. */
        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;

            MoERankBatchEndpointRole endpoint_role =
                MoERankBatchEndpointRole::ContinuationSource;
            std::shared_ptr<IMoEOverlayRankBatchTransport> transport;
            MoEOverlayRankBatchKey key;
            int continuation_participant = -1;
            std::vector<int> participant_ids;
            std::vector<std::shared_ptr<const MoEOverlayReturnRows>>
                outbound_rows;
            std::vector<std::shared_ptr<MoEOverlayReturnRows>> inbound_rows;

            TensorBase *dense_output = nullptr;
            std::optional<BufferId> dense_output_buffer_id;
            std::shared_ptr<MoEOverlayDispatchTicketStorage> ticket_storage;
            int seq_len = 0;
            int d_model = 0;
            bool clear_output_before_scatter = false;
            bool publish_ticket_completion = false;
            bool require_explicit_transaction_identity = true;
            bool require_explicit_execution_semantics = true;

            ITPContext *continuation_tp_context = nullptr;
            bool broadcast_after_scatter = false;
            int continuation_root_tp_index = 0;

            std::shared_ptr<MoEExpertDispatchOutput> dispatch_output;
            /** Typed terminal authority for the host dispatch epoch lease. */
            MoEOverlayHostDispatchLeaseTerminal residency_lease_terminal =
                MoEOverlayHostDispatchLeaseTerminal::Retain;
        };

        static_assert(StageParamsRequired<Params>);

        /**
         * @brief Bind fixed return views and validate exclusive endpoint ownership.
         * @throws std::invalid_argument for an ambiguous source/target contract.
         */
        explicit MoERankBatchReturnReduceStage(Params params);

        /** @brief Send participant results or receive and canonically accumulate them. */
        bool execute(IDeviceContext *ctx) override;

        /** @brief Identify the typed rank-batched return operation. */
        ComputeStageType type() const override
        {
            return ComputeStageType::MOE_RANK_BATCH_RETURN_REDUCE;
        }

        /** @brief Return the stable diagnostic stage name. */
        std::string name() const override
        {
            return "moe_rank_batch_return_reduce";
        }

        /** @brief This host boundary executes only under the CPU executor. */
        bool supportsBackend(ComputeBackendType backend) const override;

        /** @brief Rank-to-rank exchange is an explicit heterogeneous boundary. */
        bool isGraphCapturable() const override { return false; }

        /** @brief Mark the stage as an intentional segmented graph boundary. */
        bool isManualGraphBoundary() const override { return true; }

        /** @brief Fixed row views support padded prefill capture preflight. */
        bool supportsPaddedPrefillGraphCapturePreflight() const override
        {
            return true;
        }

        /** @brief Live row counts retain exact padded-prefill semantics. */
        bool supportsPaddedPrefillRealLengthContract() const override
        {
            return true;
        }

        /** @brief Production rank batches always consume runner-stamped identity. */
        bool hasMoEOverlayCollectiveRuntimeParams() const override
        {
            return params_.require_explicit_transaction_identity ||
                   params_.require_explicit_execution_semantics;
        }

        /** @brief Publish request identity and exact decode/prefill/MTP semantics. */
        void updateMoEOverlayCollectiveRuntimeParams(
            const MoEOverlayCollectiveRuntimeParams &params) override;

        /**
         * @brief Completion includes exact receive or durable async-send handoff.
         *
         * Continuation receive completion includes canonical accumulation and
         * optional ticket publication. Remote send completion means immutable
         * bytes are retained by the transport, not that the peer has already
         * consumed them.
         */
        bool manualGraphBoundaryComplete() const override;

        /** @brief Empty participant results are valid protocol contributions. */
        bool allowsZeroOutput() const override { return true; }

        /** @brief Host packet views own their explicit coherence boundary. */
        CoherencePolicy coherencePolicy() const override
        {
            return CoherencePolicy::NONE;
        }

        /** @brief Describe the continuation dense output when one is bound. */
        StageBufferRequirements getBufferRequirements() const override;

        /** @brief Declare dense output initialization or accumulation semantics. */
        StageBufferContract bufferContract() const override;

        /** @brief Expose immutable topology and output policy to diagnostics. */
        StageDumpInfo buildDumpInfoImpl() const override;

        /** @return Immutable stage construction contract for lowering tests. */
        const Params &params() const noexcept { return params_; }

    private:
        /** @brief Validate and add decoded rows in canonical participant order. */
        bool scatterReceivedRows();

        Params params_;
        std::vector<const MoEOverlayReturnRows *> outbound_views_;
        std::vector<MoEOverlayReturnRows *> inbound_views_;
        MoEOverlayCollectiveResult last_result_{};
        MoEOverlayCollectiveRuntimeParams runtime_params_{};
        uint64_t local_execution_count_ = 0;
    };

} // namespace llaminar2
