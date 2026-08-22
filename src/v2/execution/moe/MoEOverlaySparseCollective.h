/**
 * @file MoEOverlaySparseCollective.h
 * @brief Compact sparse payload transport for graph-native MoE overlay collectives.
 */

#pragma once

#include "backends/DeviceId.h"
#include "DecodeExpertHistogram.h"
#include "MoEOverlayActivationPayloadLayout.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <vector>

namespace llaminar2
{
    class IBackend;
    class IDeviceContext;
    class IMPIContext;

    /**
     * @brief Fixed ABI header for one captured heterogeneous MoE dispatch.
     *
     * The fixed geometry and source-device fields are immutable capture
     * identity established when the model graph is built.  Live publication
     * fields are @ref logical_row_count, the exact @ref residency_epoch used by
     * that replay, and @ref return_logical_row_count.  The current segmented
     * producer copies the logical count first and the host admission boundary
     * fills the epoch before any sparse packet is emitted.  The device-local
     * continuation path copies both values from its captured epoch ticket.
     * Consumers may inspect the record only after the producer's exact event;
     * they never infer an epoch by consulting a newer global publication.
     */
    struct MoEOverlayDispatchTicketHeader
    {
        static constexpr uint32_t kMagic = 0x54454F4Du; // "MOET"
        static constexpr uint32_t kABIVersion = 3u;

        uint32_t magic = kMagic;
        uint32_t abi_version = kABIVersion;
        uint64_t workspace_generation = 0;
        /** Exact immutable owner/weight generation used by this packet. */
        uint64_t residency_epoch = 0;
        int32_t layer_idx = -1;
        int32_t bucket_row_capacity = 0;
        int32_t route_capacity = 0;
        int32_t top_k = 0;
        int32_t d_model = 0;
        int32_t logical_row_count = 0;
        int32_t return_logical_row_count = 0;
        int32_t source_device_kind = -1;
        int32_t source_device_ordinal = -1;

        /** @brief Validate ABI identity, fixed geometry, and live row counts. */
        bool isValid() const noexcept;
    };

    static_assert(std::is_trivially_copyable_v<MoEOverlayDispatchTicketHeader>);

    /**
     * @brief Stable host-visible view populated by a captured GPU segment.
     *
     * Arrays always have physical bucket capacity.  Consumers process only the
     * leading `header->logical_row_count` rows, so two prompt lengths can reuse
     * one captured bucket without routing, transferring, or executing padding.
     */
    struct MoEOverlayDispatchTicket
    {
        MoEOverlayDispatchTicketHeader *header = nullptr;
        float *routing_indices_fp32 = nullptr;
        float *routing_weights_fp32 = nullptr;
        float *hidden_rows_fp32 = nullptr;
        float *return_rows_fp32 = nullptr;

        bool isValid() const noexcept;
        bool returnPayloadReady() const noexcept;
    };

    /**
     * @brief Model-lifetime owner for one immutable-address dispatch ticket.
     *
     * GPU sources require backend-pinned storage because native graph replay
     * records fixed asynchronous D2H destinations.  CPU sources use ordinary
     * aligned storage as their first-class implementation.  Capacity is bound
     * exactly once; rebinding a live ticket is a fatal topology error.
     */
    class MoEOverlayDispatchTicketStorage final
    {
    public:
        MoEOverlayDispatchTicketStorage() = default;
        ~MoEOverlayDispatchTicketStorage();

        MoEOverlayDispatchTicketStorage(
            const MoEOverlayDispatchTicketStorage &) = delete;
        MoEOverlayDispatchTicketStorage &operator=(
            const MoEOverlayDispatchTicketStorage &) = delete;
        MoEOverlayDispatchTicketStorage(
            MoEOverlayDispatchTicketStorage &&) = delete;
        MoEOverlayDispatchTicketStorage &operator=(
            MoEOverlayDispatchTicketStorage &&) = delete;

        /**
         * @brief Bind the ticket's complete fixed-capacity capture identity.
         * @throws std::invalid_argument for invalid geometry.
         * @throws std::logic_error when an existing ticket is rebound.
         * @throws std::runtime_error when pinned allocation fails.
         */
        void bindFixedCapacity(
            int layer_idx,
            int bucket_rows,
            int top_k,
            int d_model,
            DeviceId source_device,
            uint64_t workspace_generation);

        bool isBound() const noexcept { return allocation_ != nullptr; }
        const DeviceId &sourceDevice() const noexcept { return source_device_; }
        size_t allocationBytes() const noexcept { return allocation_bytes_; }
        MoEOverlayDispatchTicket &ticket() noexcept { return ticket_; }
        const MoEOverlayDispatchTicket &ticket() const noexcept { return ticket_; }
        /** @brief Validate mutable host memory against the separately retained binding identity. */
        bool hasValidBoundIdentity() const noexcept;

    private:
        void release() noexcept;

        void *allocation_ = nullptr;
        size_t allocation_bytes_ = 0;
        IBackend *backend_ = nullptr;
        bool backend_pinned_ = false;
        DeviceId source_device_ = DeviceId::cpu();
        int layer_idx_ = -1;
        int bucket_rows_ = 0;
        int top_k_ = 0;
        int d_model_ = 0;
        uint64_t workspace_generation_ = 0;
        std::vector<std::max_align_t> cpu_storage_;
        MoEOverlayDispatchTicket ticket_;
    };

    enum class MoEOverlayCollectiveDirection : uint8_t
    {
        Dispatch = 0,
        ReturnReduce = 1,
    };

    enum class MoEOverlayCollectiveNamespace : uint8_t
    {
        Main = 0,
        MTP = 1,
    };

    const char *toString(MoEOverlayCollectiveDirection direction);
    const char *toString(MoEOverlayCollectiveNamespace key_namespace);

    struct MoEOverlayCollectiveKey
    {
        uint64_t generation_id = 0;
        uint64_t step_id = 0;
        MoEOverlayCollectiveNamespace key_namespace = MoEOverlayCollectiveNamespace::Main;
        /** Exact mathematical phase; part of cross-rank protocol identity. */
        ExpertHistogramSource histogram_source =
            ExpertHistogramSource::DecodeToken;
        int32_t mtp_depth = -1;
        int32_t layer_idx = -1;
        int32_t tier_idx = -1;
        int32_t domain_id = -1;
        int32_t participant_id = -1;
        MoEOverlayCollectiveDirection direction = MoEOverlayCollectiveDirection::Dispatch;
        uint64_t sequence = 0;

        bool isValid() const;
        std::string toString() const;
    };

    bool operator==(const MoEOverlayCollectiveKey &lhs, const MoEOverlayCollectiveKey &rhs);
    bool operator!=(const MoEOverlayCollectiveKey &lhs, const MoEOverlayCollectiveKey &rhs);
    bool operator<(const MoEOverlayCollectiveKey &lhs, const MoEOverlayCollectiveKey &rhs);

    MoEOverlayCollectiveKey makeMoEOverlayCollectiveKey(
        uint64_t generation_id,
        uint64_t step_id,
        int layer_idx,
        int tier_idx,
        int domain_id,
        int participant_id,
        MoEOverlayCollectiveDirection direction);

    MoEOverlayCollectiveKey makeMTPMoEOverlayCollectiveKey(
        uint64_t generation_id,
        uint64_t decode_step_id,
        int mtp_depth,
        int layer_idx,
        int tier_idx,
        int domain_id,
        int participant_id,
        MoEOverlayCollectiveDirection direction);

    struct MoEOverlaySparseRows
    {
        MoEOverlayCollectiveKey key;
        /**
         * Residency snapshot that routed this packet.
         *
         * The value is copied from the root dispatch lease and remains
         * unchanged through participant-local execution and return/reduce.
         * Epoch zero is reserved for an empty protocol contribution that does
         * not carry routed work; every non-empty packet must name an epoch.
         */
        uint64_t residency_epoch = 0;
        int32_t source_participant = -1;
        int32_t target_participant = -1;
        int32_t d_model = 0;
        int32_t top_k = 0;
        size_t live_row_count = 0;
        size_t live_entry_count = 0;
        size_t row_capacity = 0;
        size_t entry_capacity = 0;
        /**
         * Number of addressable rows in @ref hidden_rows_fp32.
         *
         * This is deliberately independent of compact @ref row_capacity. A
         * node-local multi-row activation packet stores compact CSR metadata
         * but references the source transaction's shared physical-row matrix.
         */
        size_t hidden_row_capacity = 0;
        /** Immutable interpretation of @ref hidden_rows_fp32 for this view. */
        MoEOverlayActivationHiddenPayloadLayout hidden_payload_layout =
            MoEOverlayActivationHiddenPayloadLayout::CompactRows;

        int32_t *row_ids_host = nullptr;
        int32_t *entry_offsets_host = nullptr;
        int32_t *expert_ids_host = nullptr;
        float *route_weights_host = nullptr;
        float *hidden_rows_fp32 = nullptr;

        /**
         * @brief Resolve one compact packet row to its authoritative hidden row.
         *
         * Compact payloads use @p compact_row directly. Shared physical
         * payloads use the authenticated packet's logical row id. Returning a
         * null pointer is a structural protocol failure: callers must reject
         * it rather than reading another matrix or copying a fallback payload.
         *
         * @param compact_row Row ordinal in the compact CSR packet.
         * @return Exact hidden-row address, or null for invalid geometry.
         */
        [[nodiscard]] const float *hiddenRowForCompactIndex(
            size_t compact_row) const noexcept
        {
            if (!hidden_rows_fp32 || !row_ids_host || d_model <= 0 ||
                compact_row >= live_row_count ||
                !isValidMoEOverlayActivationHiddenPayloadLayout(
                    hidden_payload_layout))
            {
                return nullptr;
            }
            size_t hidden_row = compact_row;
            if (hidden_payload_layout ==
                MoEOverlayActivationHiddenPayloadLayout::SharedPhysicalRows)
            {
                const int32_t physical_row = row_ids_host[compact_row];
                if (physical_row < 0)
                    return nullptr;
                hidden_row = static_cast<size_t>(physical_row);
            }
            if (hidden_row >= hidden_row_capacity)
                return nullptr;
            return hidden_rows_fp32 +
                   hidden_row * static_cast<size_t>(d_model);
        }

        /**
         * @copydoc hiddenRowForCompactIndex(size_t) const
         */
        [[nodiscard]] float *hiddenRowForCompactIndex(
            size_t compact_row) noexcept
        {
            return const_cast<float *>(
                static_cast<const MoEOverlaySparseRows &>(*this)
                    .hiddenRowForCompactIndex(compact_row));
        }
    };

    struct MoEOverlayReturnRows
    {
        MoEOverlayCollectiveKey key;
        /** Exact residency snapshot used to compute these returned rows. */
        uint64_t residency_epoch = 0;
        int32_t source_participant = -1;
        int32_t target_participant = -1;
        int32_t d_model = 0;
        size_t live_row_count = 0;
        size_t row_capacity = 0;

        int32_t *row_ids_host = nullptr;
        float *output_rows_fp32 = nullptr;
    };

    struct MoEOverlaySparseTransferCounters
    {
        size_t dense_dispatch_bytes = 0;
        size_t dense_return_bytes = 0;
        size_t compact_dispatch_bytes = 0;
        size_t compact_return_bytes = 0;
        size_t compact_row_count = 0;
        size_t compact_entry_count = 0;

        size_t denseTotalBytes() const { return dense_dispatch_bytes + dense_return_bytes; }
        size_t compactTotalBytes() const { return compact_dispatch_bytes + compact_return_bytes; }
        size_t denseBytesAvoided() const
        {
            const size_t dense = denseTotalBytes();
            const size_t compact = compactTotalBytes();
            return dense > compact ? dense - compact : 0;
        }
    };

    size_t denseMoEOverlayDispatchBytes(int seq_len, int top_k, int d_model);
    size_t denseMoEOverlayReturnBytes(int seq_len, int d_model);
    /**
     * @brief Return exact live dispatch bytes under the device packet ABI.
     *
     * Empty participant contributions publish no payload bytes; their
     * setup-owned CSR sentinel remains outside the live-byte accounting.
     */
    size_t compactMoEOverlayDispatchBytes(const MoEOverlaySparseRows &rows);
    size_t compactMoEOverlayReturnBytes(const MoEOverlayReturnRows &rows);
    MoEOverlaySparseTransferCounters measureMoEOverlaySparseTransferCounters(
        int seq_len,
        int top_k,
        int d_model,
        const MoEOverlaySparseRows *dispatch_rows,
        const MoEOverlayReturnRows *return_rows);

    /**
     * @brief Host-visible packet storage for one sparse collective graph family.
     *
     * The default workspace keeps separate backing arrays for every
     * `(layer, tier)` key and remains useful for graphs whose protocol nodes
     * can overlap. A participant-only graph is strictly serial, so it may
     * explicitly select @ref StorageReusePolicy::SerialGraphFamily and bind
     * every key to one fixed packet slot. That policy is what makes a
     * capacity-wide decode/prefill graph economical without weakening pointer
     * stability or allocating a packet per transformer layer.
     */
    class MoEOverlayCollectiveWorkspace
    {
    public:
        /** @brief Backing-storage lifetime selected when the workspace is built. */
        enum class StorageReusePolicy : uint8_t
        {
            DistinctLayerTier, ///< Independent arrays for potentially overlapping protocol keys.
            SerialGraphFamily, ///< One array family reused by graph-ordered protocol keys.
        };

        /** @brief Complete immutable-capacity contract for a production workspace. */
        struct FixedCapacityConfig
        {
            size_t max_rows = 0;    ///< Maximum live rows in one sparse packet.
            size_t max_entries = 0; ///< Maximum routed entries in one packet.
            int d_model = 0;        ///< Hidden width copied per live row.
            int top_k = 0;          ///< Maximum routing entries per token row.
            DeviceId device = DeviceId::invalid(); ///< Host/device placement identity.
            StorageReusePolicy reuse_policy =
                StorageReusePolicy::DistinctLayerTier; ///< Array aliasing contract.
        };

        /** @brief Construct a growable workspace for isolated fixtures and builders. */
        MoEOverlayCollectiveWorkspace() = default;

        /**
         * @brief Construct a workspace whose capacity can never be rebound.
         *
         * Arrays are materialized lazily when graph nodes request their views,
         * but every resulting size and address is derived from this setup-time
         * contract. Calls to @ref ensureCapacity may verify the exact geometry
         * but cannot grow or change it.
         *
         * @throws std::invalid_argument when geometry or placement is invalid.
         */
        explicit MoEOverlayCollectiveWorkspace(FixedCapacityConfig config);

        /**
         * @brief Grow a non-fixed workspace or verify an exact fixed contract.
         * @throws std::logic_error if a fixed workspace would be rebound.
         */
        void ensureCapacity(size_t max_rows,
                            size_t max_entries,
                            int d_model,
                            int top_k,
                            DeviceId device);

        /** @brief Clear per-step live metadata without changing any allocation. */
        void resetForStep(uint64_t generation_id, uint64_t step_id);

        /** @brief Return the dispatch-receive view for one ordered protocol key. */
        MoEOverlaySparseRows dispatchReceive(int layer_idx, int tier_idx);
        /** @brief Return the locally produced dispatch view for one protocol key. */
        MoEOverlaySparseRows localExpertInput(int layer_idx, int tier_idx);
        /** @brief Return the participant-local expert result view. */
        MoEOverlayReturnRows localExpertOutput(int layer_idx, int tier_idx);
        /** @brief Return the continuation-bound collective result view. */
        MoEOverlayReturnRows returnReceive(int layer_idx, int tier_idx);

        /** @return Maximum live sparse rows retained by this workspace. */
        size_t maxRows() const noexcept { return max_rows_; }
        /** @return Maximum routed entries retained by this workspace. */
        size_t maxEntries() const noexcept { return max_entries_; }
        /** @return Hidden width of every sparse row. */
        int dModel() const noexcept { return d_model_; }
        /** @return Maximum routing width represented by one token row. */
        int topK() const noexcept { return top_k_; }
        /** @return Whether construction froze the complete capacity contract. */
        bool hasFixedCapacity() const noexcept { return fixed_capacity_; }
        /** @return Backing-array reuse policy selected at construction. */
        StorageReusePolicy storageReusePolicy() const noexcept
        {
            return reuse_policy_;
        }

    private:
        struct SparseStorage
        {
            std::vector<int32_t> row_ids_host;
            std::vector<int32_t> entry_offsets_host;
            std::vector<int32_t> expert_ids_host;
            std::vector<float> route_weights_host;
            std::vector<float> hidden_rows_fp32;
        };

        struct ReturnStorage
        {
            std::vector<int32_t> row_ids_host;
            std::vector<float> output_rows_fp32;
        };

        struct LayerTierBuffers
        {
            SparseStorage dispatch_receive;
            SparseStorage local_expert_input;
            ReturnStorage local_expert_output;
            ReturnStorage return_receive;
        };

        LayerTierBuffers &buffersFor(int layer_idx, int tier_idx);
        void ensureSparseStorage(SparseStorage &storage);
        void ensureReturnStorage(ReturnStorage &storage);

        size_t max_rows_ = 0;
        size_t max_entries_ = 0;
        int d_model_ = 0;
        int top_k_ = 0;
        DeviceId device_ = DeviceId::cpu();
        StorageReusePolicy reuse_policy_ =
            StorageReusePolicy::DistinctLayerTier;
        bool fixed_capacity_ = false;
        uint64_t generation_id_ = 0;
        uint64_t step_id_ = 0;
        std::map<std::pair<int, int>, LayerTierBuffers> buffers_by_layer_tier_;
    };

    struct MoEOverlayCollectiveResult
    {
        bool ok = true;
        bool collective_complete = false;
        int error_code = 0;
        std::string error;
    };

    class IMoEOverlaySparseCollectiveContext
    {
    public:
        virtual ~IMoEOverlaySparseCollectiveContext() = default;

        virtual MoEOverlayCollectiveResult dispatch(const MoEOverlayCollectiveKey &key,
                                                    const MoEOverlaySparseRows &outbound,
                                                    MoEOverlaySparseRows *inbound,
                                                    IDeviceContext *ctx) = 0;

        virtual MoEOverlayCollectiveResult returnReduce(const MoEOverlayCollectiveKey &key,
                                                        const MoEOverlayReturnRows &outbound,
                                                        MoEOverlayReturnRows *inbound,
                                                        IDeviceContext *ctx) = 0;

        virtual void abort(const MoEOverlayCollectiveKey &key, int reason_code) = 0;
    };

    /**
     * @brief Allocation-free sparse transport between endpoints on one rank.
     *
     * A distributed rank graph may contain the continuation endpoint itself
     * and/or colocated participants from another tier. Neither relation needs
     * an MPI collective: a rank-local packet has one exact consumer and its
     * return has one exact destination. The endpoints may share a logical
     * participant ID (continuation loopback) or use distinct participant IDs
     * (colocated cross-participant execution). Both are the same ownership and
     * ordering lifecycle, so one context handles them without manufacturing a
     * second loopback protocol or empty contributions for remote ranks.
     *
     * The context copies between setup-owned sparse views immediately and
     * retains a fixed replay ledger. It never allocates during execution.
     */
    class MoEOverlayRankLocalSparseCollectiveContext final
        : public IMoEOverlaySparseCollectiveContext
    {
    public:
        /** @brief Immutable stale-key ledger capacity. */
        struct Config
        {
            size_t slot_count = 0;
        };

        /**
         * @brief Allocate the fixed replay ledger.
         * @param config Positive model-lifetime slot capacity.
         * @throws std::invalid_argument when no slots are provided.
         */
        explicit MoEOverlayRankLocalSparseCollectiveContext(Config config);
        ~MoEOverlayRankLocalSparseCollectiveContext() override = default;

        /** @brief Publish one compact packet directly into its local consumer view. */
        MoEOverlayCollectiveResult dispatch(
            const MoEOverlayCollectiveKey &key,
            const MoEOverlaySparseRows &outbound,
            MoEOverlaySparseRows *inbound,
            IDeviceContext *ctx) override;

        /** @brief Publish one compact result directly into its local continuation view. */
        MoEOverlayCollectiveResult returnReduce(
            const MoEOverlayCollectiveKey &key,
            const MoEOverlayReturnRows &outbound,
            MoEOverlayReturnRows *inbound,
            IDeviceContext *ctx) override;

        /** @brief Mark one exact key terminal so a later publication is rejected. */
        void abort(
            const MoEOverlayCollectiveKey &key,
            int reason_code) override;

    private:
        /** @brief One bounded replay slot for each protocol direction. */
        struct ReplaySlot
        {
            std::optional<MoEOverlayCollectiveKey> completed;
            std::optional<MoEOverlayCollectiveKey> aborted;
            int abort_reason = 0;
        };

        /** @return Direction-qualified fixed ledger containing @p key. */
        std::vector<ReplaySlot> &ledgerFor(
            const MoEOverlayCollectiveKey &key) noexcept;

        std::vector<ReplaySlot> dispatch_slots_; ///< Dispatch replay ledger.
        std::vector<ReplaySlot> return_slots_;   ///< Return replay ledger.
        std::mutex mutex_; ///< Protects direct edges executed by concurrent graph segments.
    };

    class MoEOverlayLocalSparseCollectiveContext final : public IMoEOverlaySparseCollectiveContext
    {
    public:
        struct Config
        {
            int participant_count = 0;
            size_t slot_count = 0;
        };

        explicit MoEOverlayLocalSparseCollectiveContext(Config config);
        ~MoEOverlayLocalSparseCollectiveContext() override;

        MoEOverlayCollectiveResult dispatch(const MoEOverlayCollectiveKey &key,
                                            const MoEOverlaySparseRows &outbound,
                                            MoEOverlaySparseRows *inbound,
                                            IDeviceContext *ctx) override;

        MoEOverlayCollectiveResult returnReduce(const MoEOverlayCollectiveKey &key,
                                                const MoEOverlayReturnRows &outbound,
                                                MoEOverlayReturnRows *inbound,
                                                IDeviceContext *ctx) override;

        void abort(const MoEOverlayCollectiveKey &key, int reason_code) override;

    private:
        struct DispatchPayload;
        struct ReturnPayload;
        struct Slot;

        MoEOverlayCollectiveResult publishDispatch(const MoEOverlayCollectiveKey &key,
                                                   const MoEOverlaySparseRows &outbound,
                                                   MoEOverlaySparseRows *inbound);

        MoEOverlayCollectiveResult publishReturn(const MoEOverlayCollectiveKey &key,
                                                 const MoEOverlayReturnRows &outbound,
                                                 MoEOverlayReturnRows *inbound);

        int participant_count_ = 0;
        std::vector<std::unique_ptr<Slot>> slots_;
        std::unordered_set<std::string> completed_keys_;
        std::map<std::string, int> aborted_keys_;
    };

    class MoEOverlayMPISparseCollectiveContext final : public IMoEOverlaySparseCollectiveContext
    {
    public:
        /**
         * @brief Rank-local membership for one MPI sparse transport endpoint.
         *
         * MPI collectives execute once per rank for each protocol key, while a
         * rank may own any number of CUDA, ROCm, or CPU expert participants.
         * The membership set determines which addressed packets this rank may
         * consume; it is deliberately independent of MPI rank numbering.
         */
        struct Config
        {
            /** Rank communicator used by every matched protocol boundary. */
            std::shared_ptr<IMPIContext> mpi_ctx;
            /** Stable global participant ids physically owned by this rank. */
            std::vector<int> local_participant_ids;
        };

        /**
         * @brief Construct a rank transport with immutable local membership.
         * @throws std::invalid_argument for a null communicator, negative id,
         *         or duplicate local participant id.
         */
        explicit MoEOverlayMPISparseCollectiveContext(Config config);
        ~MoEOverlayMPISparseCollectiveContext() override;

        MoEOverlayCollectiveResult dispatch(const MoEOverlayCollectiveKey &key,
                                            const MoEOverlaySparseRows &outbound,
                                            MoEOverlaySparseRows *inbound,
                                            IDeviceContext *ctx) override;

        MoEOverlayCollectiveResult returnReduce(const MoEOverlayCollectiveKey &key,
                                                const MoEOverlayReturnRows &outbound,
                                                MoEOverlayReturnRows *inbound,
                                                IDeviceContext *ctx) override;

        void abort(const MoEOverlayCollectiveKey &key, int reason_code) override;

        /** @return Whether this MPI rank owns @p participant_id. */
        bool ownsLocalParticipant(int participant_id) const noexcept;

        /** @return Immutable rank-local participant membership. */
        const std::vector<int> &localParticipantIds() const noexcept
        {
            return config_.local_participant_ids;
        }

    private:
        struct DispatchPacket;
        struct ReturnPacket;

        /** Perform one compact host-staged dispatch all-gather. */
        MoEOverlayCollectiveResult dispatchHostStaged(const MoEOverlayCollectiveKey &key,
                                                      const MoEOverlaySparseRows &outbound,
                                                      MoEOverlaySparseRows *inbound);

        /** Perform one compact host-staged return all-gather. */
        MoEOverlayCollectiveResult returnHostStaged(const MoEOverlayCollectiveKey &key,
                                                    const MoEOverlayReturnRows &outbound,
                                                    MoEOverlayReturnRows *inbound);

        Config config_;
        std::unordered_set<std::string> completed_keys_;
        std::map<std::string, int> aborted_keys_;
    };

} // namespace llaminar2
