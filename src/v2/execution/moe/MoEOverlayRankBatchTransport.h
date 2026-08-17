/**
 * @file MoEOverlayRankBatchTransport.h
 * @brief Fixed-capacity point-to-point transport for rank-batched MoE packets.
 *
 * A heterogeneous ExpertOverlay rank can own several logical participants
 * (for example, four ROCm devices on one socket).  Sending one MPI collective
 * per logical participant serializes otherwise independent device work and
 * makes latency scale with the number of devices.  This file defines the
 * production protocol boundary that sends every participant for one
 * `(layer, tier, remote rank)` as one authenticated envelope.
 *
 * All wire storage is allocated when the model graph is built.  Encoding and
 * decoding operate directly on caller-owned sparse row views and never grow a
 * container in the inference path.  Participants are always serialized in
 * ascending global participant-id order so return accumulation can retain the
 * model's canonical arithmetic order regardless of device completion order.
 */

#pragma once

#include "MoEOverlaySparseCollective.h"

#include <mpi.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace llaminar2
{
    class IMPIContext;

    /** @brief Physical transport selected for one rank-batched MoE boundary. */
    enum class MoEOverlayRankBatchTransportKind : uint8_t
    {
        MPI,                   ///< Portable inter-node or explicitly remote transport.
        NodeLocalSharedRows,   ///< Same-host shared activation/result pages.
    };

    /**
     * @brief Identity of one rank-to-rank ExpertOverlay protocol transaction.
     *
     * Unlike @ref MoEOverlayCollectiveKey, this key identifies a complete
     * domain/rank batch rather than one logical participant.  Participant ids
     * live inside the authenticated envelope and are checked against immutable
     * graph topology by the receiver.
     */
    struct MoEOverlayRankBatchKey
    {
        uint64_t generation_id = 0; ///< Model-instance generation published by the runner.
        uint64_t step_id = 0;       ///< Decode step or prefill chunk identity.
        MoEOverlayCollectiveNamespace key_namespace =
            MoEOverlayCollectiveNamespace::Main; ///< Main or MTP transaction family.
        ExpertHistogramSource histogram_source =
            ExpertHistogramSource::DecodeToken; ///< Exact mathematical inference phase.
        int32_t mtp_depth = -1;       ///< MTP depth, or -1 for the main graph.
        int32_t layer_idx = -1;       ///< Transformer layer owning this exchange.
        int32_t tier_idx = -1;        ///< Integer-priority tier ordinal in the placement plan.
        int32_t domain_ordinal = -1;  ///< Stable placement-plan domain ordinal.
        int32_t source_world_rank = -1; ///< Continuation authority rank for dispatch.
        int32_t target_world_rank = -1; ///< Rank owning the batched expert endpoints.
        MoEOverlayCollectiveDirection direction =
            MoEOverlayCollectiveDirection::Dispatch; ///< Dispatch or reverse return.
        uint64_t sequence = 0; ///< Stable graph-construction sequence used by the replay ledger.

        /** @return Whether every identity field and phase combination is valid. */
        bool isValid() const noexcept;

        /** @return Stable diagnostic representation containing every identity field. */
        std::string toString() const;
    };

    /** @brief Compare complete rank-batch transaction identities. */
    bool operator==(
        const MoEOverlayRankBatchKey &lhs,
        const MoEOverlayRankBatchKey &rhs) noexcept;

    /** @brief Compare complete rank-batch transaction identities. */
    bool operator!=(
        const MoEOverlayRankBatchKey &lhs,
        const MoEOverlayRankBatchKey &rhs) noexcept;

    /**
     * @brief Build a main-graph rank-batch key with a deterministic sequence.
     */
    MoEOverlayRankBatchKey makeMoEOverlayRankBatchKey(
        uint64_t generation_id,
        uint64_t step_id,
        ExpertHistogramSource histogram_source,
        int layer_idx,
        int tier_idx,
        int domain_ordinal,
        int source_world_rank,
        int target_world_rank,
        MoEOverlayCollectiveDirection direction);

    /**
     * @brief Build an MTP verifier rank-batch key with a deterministic sequence.
     */
    MoEOverlayRankBatchKey makeMTPMoEOverlayRankBatchKey(
        uint64_t generation_id,
        uint64_t decode_step_id,
        int mtp_depth,
        int layer_idx,
        int tier_idx,
        int domain_ordinal,
        int source_world_rank,
        int target_world_rank,
        MoEOverlayCollectiveDirection direction);

    /**
     * @brief Model-lifetime wire buffers and codec for one remote rank group.
     *
     * The workspace owns exactly two byte buffers: one transmit buffer and one
     * receive buffer.  Dispatch and return are graph-ordered and therefore
     * reuse those addresses.  `max_total_rows` includes row duplication when a
     * token routes to experts on more than one participant; `max_total_entries`
     * is the corresponding routed-entry budget.
     */
    class MoEOverlayRankBatchWireWorkspace final
    {
    public:
        /** @brief Immutable capacity and topology bound at graph construction. */
        struct Config
        {
            std::vector<int> participant_ids; ///< Global ids, normalized to ascending order.
            size_t max_total_rows = 0;         ///< Sum of compact rows across all subpackets.
            size_t max_total_entries = 0;      ///< Sum of routed entries across all subpackets.
            int d_model = 0;                   ///< Hidden width of every compact row.
            int top_k = 0;                     ///< Maximum routes represented by one token row.
        };

        /**
         * @brief Allocate immutable buffers and validate participant topology.
         * @throws std::invalid_argument for empty/duplicate/negative topology or invalid geometry.
         * @throws std::overflow_error if the fixed wire capacity cannot be represented.
         */
        explicit MoEOverlayRankBatchWireWorkspace(Config config);

        MoEOverlayRankBatchWireWorkspace(
            const MoEOverlayRankBatchWireWorkspace &) = delete;
        MoEOverlayRankBatchWireWorkspace &operator=(
            const MoEOverlayRankBatchWireWorkspace &) = delete;
        MoEOverlayRankBatchWireWorkspace(
            MoEOverlayRankBatchWireWorkspace &&) = delete;
        MoEOverlayRankBatchWireWorkspace &operator=(
            MoEOverlayRankBatchWireWorkspace &&) = delete;

        /** @return Immutable canonical participant order encoded by every envelope. */
        const std::vector<int> &participantIds() const noexcept
        {
            return participant_ids_;
        }

        /** @return Maximum encoded byte count accepted by the receive buffer. */
        size_t wireCapacityBytes() const noexcept
        {
            return receive_buffer_.size();
        }

        /** @return Address of the immutable transmit allocation for stability tests/transport. */
        const std::byte *sendBufferData() const noexcept
        {
            return send_buffer_.data();
        }

        /** @return Address of the immutable receive allocation for stability tests/transport. */
        const std::byte *receiveBufferData() const noexcept
        {
            return receive_buffer_.data();
        }

        /**
         * @brief Encode all dispatch subpackets directly into the fixed send buffer.
         *
         * `rows` must have exactly one view for each @ref participantIds entry,
         * in that order.  No container allocation occurs.
         */
        bool encodeDispatch(
            const MoEOverlayRankBatchKey &key,
            std::span<const MoEOverlaySparseRows *const> rows,
            size_t *encoded_bytes,
            std::string *error);

        /**
         * @brief Encode dispatch rows directly into one transport-owned ring slot.
         *
         * The destination must remain stable until its non-blocking MPI request
         * completes. Encoding directly into that registered/fixed allocation
         * avoids a second whole-envelope memcpy after packet construction.
         */
        bool encodeDispatchInto(
            const MoEOverlayRankBatchKey &key,
            std::span<const MoEOverlaySparseRows *const> rows,
            std::span<std::byte> destination,
            size_t *encoded_bytes,
            std::string *error);

        /**
         * @brief Decode a dispatch envelope into fixed participant-owned views.
         *
         * The envelope key and complete participant list must exactly match the
         * graph-bound expectation before any payload is published.
         */
        bool decodeDispatch(
            const MoEOverlayRankBatchKey &expected_key,
            std::span<const std::byte> payload,
            std::span<MoEOverlaySparseRows *const> rows,
            std::string *error);

        /** @brief Encode participant-local result rows in canonical participant order. */
        bool encodeReturn(
            const MoEOverlayRankBatchKey &key,
            std::span<const MoEOverlayReturnRows *const> rows,
            size_t *encoded_bytes,
            std::string *error);

        /** @brief Encode return rows directly into one transport-owned ring slot. */
        bool encodeReturnInto(
            const MoEOverlayRankBatchKey &key,
            std::span<const MoEOverlayReturnRows *const> rows,
            std::span<std::byte> destination,
            size_t *encoded_bytes,
            std::string *error);

        /** @brief Decode returned subpackets into fixed continuation-owned views. */
        bool decodeReturn(
            const MoEOverlayRankBatchKey &expected_key,
            std::span<const std::byte> payload,
            std::span<MoEOverlayReturnRows *const> rows,
            std::string *error);

        /** @return Read-only prefix most recently encoded for MPI transmission. */
        std::span<const std::byte> encodedPayload(size_t encoded_bytes) const;

        /** @return Complete writable receive allocation passed to MPI_Irecv. */
        std::span<std::byte> receiveCapacity() noexcept
        {
            return receive_buffer_;
        }

    private:
        std::vector<int> participant_ids_;
        size_t max_total_rows_ = 0;
        size_t max_total_entries_ = 0;
        int d_model_ = 0;
        int top_k_ = 0;
        std::vector<std::byte> send_buffer_;
        std::vector<std::byte> receive_buffer_;
    };

    /**
     * @brief Typed graph-facing authority for one rank-batched sparse exchange.
     *
     * Implementations have identical authenticated transaction semantics but
     * different physical locality contracts.  The node-local implementation
     * exposes fixed shared row views so graph builders can bind endpoint stages
     * directly to the mapped payload instead of serializing it.  MPI never
     * advertises those views.
     */
    class IMoEOverlayRankBatchTransport
    {
    public:
        virtual ~IMoEOverlayRankBatchTransport() = default;

        /** @return Exact physical transport kind resolved from topology. */
        virtual MoEOverlayRankBatchTransportKind kind() const noexcept = 0;

        /** @brief Publish or consume one authenticated dispatch transaction. */
        virtual MoEOverlayCollectiveResult exchangeDispatch(
            const MoEOverlayRankBatchKey &key,
            std::span<const MoEOverlaySparseRows *const> outbound,
            std::span<MoEOverlaySparseRows *const> inbound) = 0;

        /** @brief Publish or consume one authenticated return transaction. */
        virtual MoEOverlayCollectiveResult exchangeReturn(
            const MoEOverlayRankBatchKey &key,
            std::span<const MoEOverlayReturnRows *const> outbound,
            std::span<MoEOverlayReturnRows *const> inbound) = 0;

        /** @return Continuation authority rank fixed by graph topology. */
        virtual int sourceWorldRank() const noexcept = 0;
        /** @return Remote endpoint-owner rank fixed by graph topology. */
        virtual int targetWorldRank() const noexcept = 0;
        /** @return MPI world rank owning this process-local object. */
        virtual int localWorldRank() const noexcept = 0;
        /** @return Immutable canonical participant ordering. */
        virtual const std::vector<int> &participantIds() const noexcept = 0;

        /** @return Whether bulk dispatch/return arrays reside in shared pages. */
        virtual bool hasSharedRowStorage() const noexcept { return false; }

        /**
         * @brief Return a process-local view onto one shared dispatch slot.
         * @throws std::logic_error when this transport has no shared row storage.
         */
        virtual MoEOverlaySparseRows sharedDispatchRows(int participant_id) const;

        /**
         * @brief Return a process-local view onto one shared result slot.
         * @throws std::logic_error when this transport has no shared row storage.
         */
        virtual MoEOverlayReturnRows sharedReturnRows(int participant_id) const;
    };

    /**
     * @brief Direct MPI transport for one continuation-rank/endpoint-rank pair.
     *
     * Exactly the two named ranks may call this object. Dispatch and return
     * sends are submitted from transport-owned fixed rings and are retired by
     * `MPI_Test`; a graph stage never waits merely to recover ownership of a
     * caller buffer. The continuation rank also preposts the return receive
     * before publishing dispatch so the reverse packet can match immediately.
     * Fixed-size receives admit any live prefix up to the graph-bound capacity,
     * while the authenticated envelope supplies exact live counts. No count
     * all-gather or variable receive allocation occurs in the hot path.
     */
    class MoEOverlayMPIRankBatchTransport final
        : public IMoEOverlayRankBatchTransport
    {
    public:
        /** @brief Immutable rank pair, workspace, and replay-ledger capacity. */
        struct Config
        {
            std::shared_ptr<IMPIContext> mpi_ctx; ///< MPI communicator authority.
            int source_world_rank = -1;           ///< Continuation authority rank.
            int target_world_rank = -1;           ///< Remote endpoint owner rank.
            std::shared_ptr<MoEOverlayRankBatchWireWorkspace> workspace;
            size_t transaction_slot_count = 4096; ///< Fixed stale-key ledger slots.
            /** Number of stable send buffers retained until MPI_Test completion. */
            size_t asynchronous_send_slot_count = 4;
        };

        /**
         * @brief Bind one immutable rank-pair transport.
         * @throws std::invalid_argument for invalid ranks, workspace, or ledger capacity.
         */
        explicit MoEOverlayMPIRankBatchTransport(Config config);

        /**
         * @brief Drain every matched non-blocking send before freeing its slot.
         *
         * Destruction with an unmatched preposted return is a fatal lifecycle
         * violation: destroying that receive buffer would let MPI write into
         * released storage.
         */
        ~MoEOverlayMPIRankBatchTransport();

        /**
         * @brief Send or receive one complete dispatch envelope for this rank pair.
         *
         * On the source rank, `outbound` must contain every participant and
         * `inbound` must be empty. The target rank supplies the inverse. Source
         * completion means the encoded bytes have been handed to a stable
         * transport slot and the matching return receive has been posted; it
         * does not mean the peer has consumed the packet.
         */
        MoEOverlayCollectiveResult exchangeDispatch(
            const MoEOverlayRankBatchKey &key,
            std::span<const MoEOverlaySparseRows *const> outbound,
            std::span<MoEOverlaySparseRows *const> inbound) override;

        /**
         * @brief Send or receive one complete return envelope for this rank pair.
         *
         * The target rank publishes every participant result; the continuation
         * source rank receives all results into canonical participant views.
         * Target completion means the immutable return bytes are retained by a
         * transport slot until MPI reports completion.
         */
        MoEOverlayCollectiveResult exchangeReturn(
            const MoEOverlayRankBatchKey &key,
            std::span<const MoEOverlayReturnRows *const> outbound,
            std::span<MoEOverlayReturnRows *const> inbound) override;

        /** @return MPI because this implementation owns MPI requests. */
        MoEOverlayRankBatchTransportKind kind() const noexcept override
        {
            return MoEOverlayRankBatchTransportKind::MPI;
        }

        /** @return Continuation authority rank fixed by graph topology. */
        int sourceWorldRank() const noexcept override { return config_.source_world_rank; }

        /** @return Remote endpoint-owner rank fixed by graph topology. */
        int targetWorldRank() const noexcept override { return config_.target_world_rank; }

        /** @return MPI world rank that owns this process-local transport instance. */
        int localWorldRank() const noexcept override;

        /** @return Canonical participant order authenticated by the codec. */
        const std::vector<int> &participantIds() const noexcept override
        {
            return config_.workspace->participantIds();
        }

        /** @return Shared immutable-capacity codec workspace. */
        const std::shared_ptr<MoEOverlayRankBatchWireWorkspace> &workspace() const noexcept
        {
            return config_.workspace;
        }

    private:
        /** @brief One fixed envelope allocation and its exact live MPI request. */
        struct AsynchronousSendSlot
        {
            std::vector<std::byte> storage;
            MPI_Request request = MPI_REQUEST_NULL;
            bool in_flight = false;
        };

        /** @brief Claim one fixed replay-ledger slot or reject a stale/colliding key. */
        bool claimTransaction(
            const MoEOverlayRankBatchKey &key,
            std::vector<std::optional<MoEOverlayRankBatchKey>> &ledger,
            std::string *error);

        /** @brief Progress completed sends and return one reusable fixed slot. */
        AsynchronousSendSlot *acquireSendSlot(
            std::vector<AsynchronousSendSlot> &slots,
            size_t *cursor,
            const char *direction,
            std::string *error);

        /** @brief Active-progress one dependency request with the canonical timeout. */
        bool progressRequestToCompletion(
            MPI_Request *request,
            MPI_Status *status,
            const char *operation,
            std::string *error) const;

        /** @brief Retire every already-completed request in one send ring. */
        void progressSendSlots(
            std::vector<AsynchronousSendSlot> &slots) const;

        /** @brief Teardown-only bounded drain for all fixed send slots. */
        void drainSendSlotsNoexcept(
            std::vector<AsynchronousSendSlot> &slots,
            const char *direction) noexcept;

        Config config_;
        std::vector<std::optional<MoEOverlayRankBatchKey>> dispatch_ledger_;
        std::vector<std::optional<MoEOverlayRankBatchKey>> return_ledger_;
        std::vector<AsynchronousSendSlot> dispatch_send_slots_;
        std::vector<AsynchronousSendSlot> return_send_slots_;
        size_t next_dispatch_send_slot_ = 0;
        size_t next_return_send_slot_ = 0;
        /** Source posts the matching return receive before dispatch leaves. */
        MPI_Request pending_return_receive_ = MPI_REQUEST_NULL;
        std::optional<MoEOverlayRankBatchKey> pending_return_dispatch_key_;
        /** Reject accidental concurrent reuse of the one shared wire buffer. */
        std::atomic_flag in_flight_ = ATOMIC_FLAG_INIT;
    };

} // namespace llaminar2
