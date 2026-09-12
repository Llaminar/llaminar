/**
 * @file MoEOverlaySparseCollective.h
 * @brief Compact sparse payload transport for graph-native MoE overlay collectives.
 *
 * Graph-owned packet views retain exact residency epochs, live counts and
 * arithmetic layouts across rank-local, shared-memory and MPI transports.
 * A canonical return is a collection of raw expert rows, never a partial sum;
 * only the final continuation reducer may apply original router ordering.
 */

#pragma once

#include "backends/DeviceId.h"
#include "DecodeExpertHistogram.h"
#include "MoEOverlayActivationPayloadLayout.h"
#include "MoEOverlayReturnLayout.h"

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
    class TensorBase;
    class MappedHostTransferRegion;
    class MappedHostTransferArena;
    struct MoEOverlayCanonicalRouteTicketControl;
    struct DeviceMoEOverlayEpochTicket;

    /** @brief Sole owner of the placement epoch in a dispatch publication. */
    enum class MoEOverlayDispatchEpochAuthority : uint8_t
    {
        HostAdmission, ///< CPU/host-scheduled transaction admits an immutable snapshot.
        CapturedRequest, ///< GPU request ticket selected the epoch before any layer ran.
    };

    /**
     * @brief Fixed ABI header for one captured heterogeneous MoE dispatch.
     *
     * The fixed geometry and source-device fields are immutable capture
     * identity established when the model graph is built.  Live publication
     * fields are @ref logical_row_count, the exact @ref residency_epoch used by
     * that replay, and @ref return_logical_row_count.  The current segmented
     * producer copies the logical count and, for device-admitted execution,
     * the epoch from its captured request ticket. Only an explicitly host-
     * admitted transaction fills the epoch at the host admission boundary.
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
     * GPU sources own mapped payload pages and one isolated mapped timeline
     * word. Captured producer kernels write the payload through the exact
     * device alias, then system-release publish the timeline; retained graphs
     * therefore contain no D2H memcpy nodes or host synchronization. The CPU
     * endpoint acquires the same host pages without waiting for unrelated work
     * at the graph terminal. CPU sources use ordinary aligned storage as their
     * first-class implementation. Capacity is bound exactly once; rebinding a
     * live ticket is a fatal topology error.
     */
    class MoEOverlayDispatchTicketStorage final
    {
    public:
        /**
         * @brief Stable device sources copied into one mapped dispatch ticket.
         *
         * The logical-row scalar is optional for exact-width decode graphs.
         * Every other pointer and byte count is mandatory and must describe
         * immutable graph-capture storage on the ticket's source device.
         */
        struct CapturedDevicePayload
        {
            const int32_t *logical_row_count = nullptr;
            const void *routing_indices = nullptr;
            const void *routing_weights = nullptr;
            size_t route_bytes = 0u;
            const void *hidden_rows = nullptr;
            size_t hidden_bytes = 0u;
        };

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
         * @param layer_idx Model layer owning this publication.
         * @param bucket_rows Captured physical row capacity.
         * @param top_k Maximum routed experts per row.
         * @param d_model Width of each activation row.
         * @param source_device Exact CPU or GPU publisher.
         * @param workspace_generation Positive immutable workspace identity.
         * @param mapped_arena Model-owned mapped storage for a GPU publisher.
         * @param captured_request_epoch Borrowed GPU request ticket owned by
         *        the model's runtime table. When present, its address is frozen
         *        and its live epoch is published before the ticket release edge;
         *        it must outlive every graph using this storage.
         * @throws std::invalid_argument for invalid geometry.
         * @throws std::logic_error when an existing ticket is rebound.
         * @throws std::runtime_error when mapped arena allocation fails.
         */
        void bindFixedCapacity(
            int layer_idx,
            int bucket_rows,
            int top_k,
            int d_model,
            DeviceId source_device,
            uint64_t workspace_generation,
            std::shared_ptr<MappedHostTransferArena> mapped_arena = nullptr,
            const DeviceMoEOverlayEpochTicket *captured_request_epoch = nullptr);

        /** @return Immutable epoch authority selected with this capture identity. */
        [[nodiscard]] MoEOverlayDispatchEpochAuthority epochAuthority() const noexcept
        {
            return captured_request_epoch_
                ? MoEOverlayDispatchEpochAuthority::CapturedRequest
                : MoEOverlayDispatchEpochAuthority::HostAdmission;
        }

        bool isBound() const noexcept { return allocation_ != nullptr; }
        const DeviceId &sourceDevice() const noexcept { return source_device_; }
        size_t allocationBytes() const noexcept { return allocation_bytes_; }
        MoEOverlayDispatchTicket &ticket() noexcept { return ticket_; }
        const MoEOverlayDispatchTicket &ticket() const noexcept { return ticket_; }
        /** @brief Validate mutable host memory against the separately retained binding identity. */
        bool hasValidBoundIdentity() const noexcept;

        /**
         * @brief Reset the GPU-to-host publication edge before one graph launch.
         *
         * A ticket is consumed within its layer before the serial transaction
         * can launch the same graph again. Resetting this data word therefore
         * performs no device work, synchronization, or topology change.
         *
         * @param error Optional exact contract diagnostic.
         * @return True for a valid GPU publication contract or a CPU no-op.
         */
        bool armCapturedPublication(std::string *error = nullptr) noexcept;

        /**
         * @brief Enqueue all device-owned ticket bytes through mapped aliases.
         *
         * TransferEngine validates each destination offset and launches bounded
         * backend kernels on @p stream. The method never allocates, waits,
         * synchronizes, or records a host-facing memcpy node. Call
         * @ref enqueueCapturedPublication afterward on the same stream to
         * release-publish the complete payload.
         *
         * @param payload Exact device pointers and immutable byte geometry.
         * @param stream Exact non-null captured producer stream.
         * @param error Optional exact contract or launch diagnostic.
         * @return True only after every mapped copy kernel was enqueued.
         */
        bool enqueueCapturedPayload(
            const CapturedDevicePayload &payload,
            void *stream,
            std::string *error = nullptr) noexcept;

        /**
         * @brief Enqueue the ticket's system-release publication on @p stream.
         *
         * Publication follows every preceding mapped payload kernel on the
         * same exact stream and is graph-capturable on CUDA and ROCm.
         *
         * @param stream Exact non-null producer stream.
         * @param error Optional exact enqueue diagnostic.
         * @return True after enqueue, or for a CPU ticket that needs no edge.
         */
        bool enqueueCapturedPublication(
            void *stream,
            std::string *error = nullptr) noexcept;

        /**
         * @brief Acquire the exact captured payload publication on the host.
         *
         * This bounded wait observes only the isolated timeline word. It never
         * synchronizes a GPU stream or device, so later GPU work remains
         * concurrent with CPU expert execution.
         *
         * @param error Optional timeout or protocol diagnostic.
         * @return True when the payload is visible, or for a CPU ticket.
         */
        bool awaitCapturedPublication(std::string *error = nullptr) noexcept;

        /** @return Whether this GPU ticket owns a complete mapped publication edge. */
        bool hasCapturedPublicationContract() const noexcept;

    private:
        /** Borrowed from the model-owned runtime table; it outlives every captured replay. */
        const DeviceMoEOverlayEpochTicket *captured_request_epoch_ = nullptr;
        void release() noexcept;

        void *allocation_ = nullptr;
        size_t allocation_bytes_ = 0;
        /** GPU ticket payload pages; null for a host-owned CPU ticket. */
        std::shared_ptr<MappedHostTransferRegion> payload_region_;
        /** Isolated mapped cache line used only for GPU-to-host readiness. */
        std::shared_ptr<MappedHostTransferRegion> publication_region_;
        /** Stable host alias at byte zero of @ref publication_region_. */
        std::uint64_t *publication_timeline_ = nullptr;
        DeviceId source_device_ = DeviceId::cpu();
        int layer_idx_ = -1;
        int bucket_rows_ = 0;
        int top_k_ = 0;
        int d_model_ = 0;
        uint64_t workspace_generation_ = 0;
        std::vector<std::max_align_t> cpu_storage_;
        MoEOverlayDispatchTicket ticket_;
    };

    /**
     * @brief Setup-owned sparse return ticket from one colocated CPU participant.
     *
     * The route-slot arrays and completion record live in a small mapped region.
     * Preweighted rows remain in the participant's serial CPU expert arena and
     * are allocated as one native mapped region through TransferEngine, so every layer graph
     * can reuse the maximum route workspace without allocating a top-k-sized
     * return matrix per layer. One ticket has one producer and one continuation
     * GPU; capacity and every mapped alias are immutable after binding.
     */
    class MoEOverlayCanonicalRouteReturnTicketStorage final
    {
    public:
        /**
         * @brief Move-only CPU ownership of one unpublished ticket payload.
         *
         * The lease is created only after the prior GPU acknowledgement is
         * visible. Destroying an unpublished lease cancels the arm transition,
         * so a failed expert computation cannot strand the reusable ticket.
         * Calling @ref publish transfers ownership to the GPU consumer and
         * makes the mapped payload immutable until its device acknowledgement.
         */
        class Publication final
        {
        public:
            /** @brief Construct an invalid lease for conditional ownership. */
            Publication() = default;
            /** @brief Cancel an armed but unpublished payload, if one remains. */
            ~Publication();

            Publication(const Publication &) = delete;
            Publication &operator=(const Publication &) = delete;
            /** @brief Transfer unique producer ownership from @p other. */
            Publication(Publication &&other) noexcept;
            /** @brief Cancel current ownership, then acquire it from @p other. */
            Publication &operator=(Publication &&other) noexcept;

            /** @return Whether this lease owns one armed CPU publication. */
            [[nodiscard]] explicit operator bool() const noexcept
            {
                return owner_ != nullptr && sequence_ != 0u;
            }

            /**
             * @brief Release-publish the initialized compact route prefix.
             * @param live_entry_count Number of initialized route identities/rows.
             * @return False for invalid ownership, capacity, or sequence state.
             */
            bool publish(size_t live_entry_count) noexcept;

            /** @return Whether this lease belongs to @p storage. */
            [[nodiscard]] bool belongsTo(
                const MoEOverlayCanonicalRouteReturnTicketStorage &storage)
                const noexcept
            {
                return owner_ == &storage;
            }

        private:
            friend class MoEOverlayCanonicalRouteReturnTicketStorage;

            /** @brief Construct the unique lease for one armed sequence. */
            Publication(
                MoEOverlayCanonicalRouteReturnTicketStorage *owner,
                uint64_t sequence) noexcept
                : owner_(owner), sequence_(sequence)
            {
            }

            /** @brief Cancel an unpublished arm before dropping ownership. */
            void reset() noexcept;

            MoEOverlayCanonicalRouteReturnTicketStorage *owner_ = nullptr;
            uint64_t sequence_ = 0u;
        };

        MoEOverlayCanonicalRouteReturnTicketStorage() = default;
        ~MoEOverlayCanonicalRouteReturnTicketStorage() = default;

        MoEOverlayCanonicalRouteReturnTicketStorage(
            const MoEOverlayCanonicalRouteReturnTicketStorage &) = delete;
        MoEOverlayCanonicalRouteReturnTicketStorage &operator=(
            const MoEOverlayCanonicalRouteReturnTicketStorage &) = delete;
        MoEOverlayCanonicalRouteReturnTicketStorage(
            MoEOverlayCanonicalRouteReturnTicketStorage &&) = delete;
        MoEOverlayCanonicalRouteReturnTicketStorage &operator=(
            MoEOverlayCanonicalRouteReturnTicketStorage &&) = delete;

        /**
         * @brief Bind immutable layer geometry and the serial CPU contribution arena.
         * @param layer_idx Exact model layer represented by this ticket.
         * @param route_capacity Maximum compact/original route slots.
         * @param d_model Width of one FP32 contribution row.
         * @param continuation_device Exact local CUDA/ROCm consumer.
         * @param workspace_generation Positive setup-owned graph identity.
         * @param contribution_region Mapped CPU canonical-route arena, with at
         *        least `route_capacity * d_model * sizeof(float)` bytes.
         * @param metadata_arena Model-owned mapped arena for this ticket's
         *        captured control/route metadata. It must name the exact
         *        continuation GPU; per-ticket native registration is forbidden.
         * @throws std::invalid_argument for incomplete geometry or mapping.
         * @throws std::logic_error when a live ticket is rebound.
         */
        void bindFixedCapacity(
            int layer_idx,
            size_t route_capacity,
            int d_model,
            DeviceId continuation_device,
            uint64_t workspace_generation,
            std::shared_ptr<MappedHostTransferRegion> contribution_region,
            std::shared_ptr<MappedHostTransferArena> metadata_arena);

        /**
         * @brief Arm one CPU publication before it writes route rows.
         * @param residency_epoch Exact non-zero placement epoch of the packet.
         * @return A valid move-only publication lease, or an invalid lease when
         *         identity is incomplete, this producer is already armed, or
         *         the GPU has not acknowledged the prior publication.
         */
        [[nodiscard]] Publication arm(uint64_t residency_epoch) noexcept;

        /**
         * @brief Release a captured consumer after CPU ticket service fails.
         *
         * The method publishes one monotonic @c Aborted sequence when no
         * success payload is pending. A pending current publication is already
         * sufficient to release the consumer and is accepted idempotently.
         * It never overwrites an armed producer lease or an unconsumed payload.
         *
         * @return True when the GPU consumer is guaranteed to observe either a
         *         success or abort publication; false for an unsafe lifecycle.
         */
        [[nodiscard]] bool publishAbort() noexcept;

        /** @return Whether a complete current publication is host-visible. */
        [[nodiscard]] bool payloadReady() const noexcept;
        /**
         * @brief Test readiness for one exact immutable residency epoch.
         * @param residency_epoch Epoch named by the sparse dispatch packet.
         */
        [[nodiscard]] bool payloadReadyFor(
            uint64_t residency_epoch) const noexcept;
        /**
         * @brief Test whether one exact epoch was successfully published.
         *
         * Unlike @ref payloadReadyFor, this certificate remains true after the
         * GPU has acknowledged the publication. It is intended for a CPU
         * protocol terminal that follows the producer while a retained GPU
         * parent consumes the same ticket concurrently. It does not grant the
         * caller permission to overwrite payload storage; only @ref arm owns
         * that producer transition.
         *
         * @param residency_epoch Epoch named by the sparse dispatch packet.
         * @return True for a successful pending or already-acknowledged
         *         publication of exactly @p residency_epoch.
         */
        [[nodiscard]] bool publicationSucceededFor(
            uint64_t residency_epoch) const noexcept;
        /** @return Whether all immutable mapped identities remain valid. */
        [[nodiscard]] bool hasValidBoundIdentity() const noexcept;
        /** @return Exact continuation GPU consuming this ticket. */
        [[nodiscard]] DeviceId continuationDevice() const noexcept
        {
            return continuation_device_;
        }
        /** @return Immutable compact/original route capacity. */
        [[nodiscard]] size_t routeCapacity() const noexcept
        {
            return route_capacity_;
        }
        /** @return Immutable contribution width. */
        [[nodiscard]] int dModel() const noexcept { return d_model_; }
        /** @return Exact model layer represented by the ticket. */
        [[nodiscard]] int layerIndex() const noexcept { return layer_idx_; }

        /** @return Host route-slot destination written by the CPU producer. */
        [[nodiscard]] int32_t *originalRouteSlotsHost() const noexcept;
        /** @return Host compact-slot destination written by the CPU producer. */
        [[nodiscard]] int32_t *compactRouteSlotsHost() const noexcept;
        /** @return Host contribution arena written by the CPU expert stage. */
        [[nodiscard]] float *contributionRowsHost() const noexcept;
        /** @return Device-visible completion record embedded in captured ingress. */
        [[nodiscard]] MoEOverlayCanonicalRouteTicketControl *
        controlDeviceAlias() const noexcept;
        /** @return Device-visible original-slot array embedded in captured ingress. */
        [[nodiscard]] const int32_t *originalRouteSlotsDeviceAlias() const noexcept;
        /** @return Device-visible compact-slot array embedded in captured ingress. */
        [[nodiscard]] const int32_t *compactRouteSlotsDeviceAlias() const noexcept;
        /** @return Device-visible CPU contribution arena embedded in ingress. */
        [[nodiscard]] const float *contributionRowsDeviceAlias() const noexcept;

    private:
        /** Host-side producer transition; GPU acknowledgement lives in the ABI. */
        enum class ProducerLifecycle : uint8_t
        {
            Unbound,   ///< No immutable mapped ticket has been installed.
            Quiescent, ///< Producer is idle; sequences may be checked for reuse.
            Armed,     ///< One unpublished payload is owned by the CPU producer.
        };

        /**
         * @brief Complete the exact lease and transfer payload ownership to GPU.
         * @param sequence Sequence authenticated by the unique producer lease.
         * @param live_entry_count Number of initialized compact route rows.
         * @return Whether the exact armed payload was release-published.
         */
        bool publishArmed(
            uint64_t sequence, size_t live_entry_count) noexcept;
        /**
         * @brief Roll back one exact unpublished lease after CPU failure.
         * @param sequence Sequence authenticated by the expiring producer lease.
         */
        void cancelArmed(uint64_t sequence) noexcept;

        std::shared_ptr<MappedHostTransferRegion> metadata_region_;
        std::shared_ptr<MappedHostTransferRegion> contribution_region_;
        MoEOverlayCanonicalRouteTicketControl *control_host_ = nullptr;
        int32_t *original_route_slots_host_ = nullptr;
        int32_t *compact_route_slots_host_ = nullptr;
        size_t original_route_slots_offset_ = 0u;
        size_t compact_route_slots_offset_ = 0u;
        DeviceId continuation_device_ = DeviceId::invalid();
        int layer_idx_ = -1;
        size_t route_capacity_ = 0u;
        int d_model_ = 0;
        uint64_t workspace_generation_ = 0u;
        uint64_t armed_sequence_ = 0u;
        ProducerLifecycle producer_lifecycle_ = ProducerLifecycle::Unbound;
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
        /** Original `logical_row * top_k + router_slot` for every live entry. */
        int32_t *original_route_slots_host = nullptr;
        /** Participant-local `compact_row * top_k + local_slot` per entry. */
        int32_t *compact_route_slots_host = nullptr;
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

    /**
     * @brief Borrowed live return payload with an explicit arithmetic contract.
     *
     * Canonical expert routes preserve the router's original slot through
     * movement and transport. They must be gathered, then weighted/reduced in
     * router order; summing participant partials changes FP32 parenthesization
     * whenever an expert changes owner. Capacity is storage, never wire length.
     */
    struct MoEOverlayReturnRows
    {
        MoEOverlayCollectiveKey key;
        /** Exact residency snapshot used to compute these returned rows. */
        uint64_t residency_epoch = 0;
        int32_t source_participant = -1;
        int32_t target_participant = -1;
        int32_t d_model = 0;
        MoEOverlayReturnLayout layout = MoEOverlayReturnLayout::ParticipantTokenPartials;
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
    /**
     * @brief Return exact live bytes read through one canonical CPU ticket.
     *
     * The fixed mapped allocation is setup-owned capacity and is deliberately
     * excluded. One live publication exposes a cache-line control record, two
     * route-slot integers, and one FP32 contribution row per route entry.
     * Returning zero denotes invalid geometry or arithmetic overflow.
     *
     * @param live_entry_count Number of published compact route entries.
     * @param d_model Width of one FP32 contribution row.
     * @return Exact live mapped payload bytes, or zero when unrepresentable.
     */
    size_t canonicalMoEOverlayTicketReturnBytes(
        size_t live_entry_count,
        int d_model) noexcept;
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
            MoEOverlayReturnLayout return_layout = MoEOverlayReturnLayout::ParticipantTokenPartials;
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
            std::vector<int32_t> original_route_slots_host;
            std::vector<int32_t> compact_route_slots_host;
            std::vector<float> hidden_rows_fp32;
        };

        struct ReturnStorage
        {
            std::vector<int32_t> row_ids_host;
            /** Bulk payload uses the canonical tensor/physical-memory authority. */
            std::shared_ptr<TensorBase> output_rows_fp32;
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
        MoEOverlayReturnLayout return_layout_ = MoEOverlayReturnLayout::ParticipantTokenPartials;
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
