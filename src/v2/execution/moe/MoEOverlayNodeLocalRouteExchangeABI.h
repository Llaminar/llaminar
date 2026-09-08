/**
 * @file MoEOverlayNodeLocalRouteExchangeABI.h
 * @brief Fixed-width ABI for sparse canonical-route exchange within one node.
 *
 * A continuation domain may contain several accelerators while routed experts
 * are assigned whole to exactly one participant.  A dense reduce of every
 * `[row, route, hidden]` slot therefore transports mostly zeroes.  This ABI
 * describes a single-producer/single-consumer lane that publishes only the
 * original router slots assigned to one non-root participant through mapped
 * node-local memory.  The root consumes those slots in original router order,
 * selecting local device memory for its own routes and mapped memory only for
 * peer routes.
 *
 * The control words are monotonic and device-owned.  The producer is the sole
 * writer of `produced_epoch`; the root is the sole writer of `consumed_epoch`.
 * A producer never overwrites the one reusable payload bank until the prior
 * epoch has been consumed.  This is safe because the following dense
 * continuation publication prevents a participant from entering the next MoE
 * layer before the root has acknowledged the current layer.  No host shadow,
 * callback, stream synchronization, or replay-time topology decision exists.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define LLAMINAR_MOE_NODE_LOCAL_ROUTE_HD __host__ __device__
#else
#define LLAMINAR_MOE_NODE_LOCAL_ROUTE_HD
#endif

namespace llaminar2
{
    /** Binary identity of a node-local canonical-route lane (`NLRE`). */
    inline constexpr std::uint32_t kMoENodeLocalRouteExchangeMagic =
        0x45524c4eu;

    /** Current binary layout version. */
    inline constexpr std::uint32_t kMoENodeLocalRouteExchangeVersion = 1u;

    /** Terminal monotonic value that releases a peer wait after a fatal fault. */
    inline constexpr std::uint64_t kMoENodeLocalRouteExchangeAbortEpoch =
        std::numeric_limits<std::uint64_t>::max();

    /** Device-visible semantic state of one producer-to-root lane. */
    enum class MoENodeLocalRouteExchangeState : std::uint32_t
    {
        Ready = 1u, ///< Immutable topology is installed and no fault is present.
        Aborted = 2u, ///< The lane cannot be reused; peers must drain and fail.
    };

    /** Device-visible failure classification for one lane. */
    enum class MoENodeLocalRouteExchangeCode : std::uint32_t
    {
        Success = 0u,
        InvalidControl = 1u,
        EpochOverflow = 2u,
        RouteAssignmentMismatch = 3u,
        PeerAborted = 4u,
    };

    /**
     * @brief Shared cache-line-isolated ownership and progress record.
     *
     * Immutable fields are written once before either endpoint graph is built.
     * Mutable ownership is explicit: only the producer writes `produced_epoch`,
     * only the root writes `consumed_epoch`, and either endpoint may atomically
     * publish the first terminal failure into `state`/`code`.
     */
    struct alignas(64) MoENodeLocalRouteLaneControl
    {
        std::uint32_t magic = kMoENodeLocalRouteExchangeMagic;
        std::uint32_t version = kMoENodeLocalRouteExchangeVersion;
        std::int32_t producer_participant = -1;
        std::int32_t root_participant = -1;
        std::uint32_t route_capacity = 0u;
        std::uint32_t d_model = 0u;
        std::uint32_t state = static_cast<std::uint32_t>(
            MoENodeLocalRouteExchangeState::Ready);
        std::uint32_t code = static_cast<std::uint32_t>(
            MoENodeLocalRouteExchangeCode::Success);
        std::uint64_t produced_epoch = 0u;
        std::uint64_t consumed_epoch = 0u;
        std::uint64_t failed_epoch = 0u;
        std::uint64_t reserved = 0u;
    };

    static_assert(
        sizeof(MoENodeLocalRouteLaneControl) == 64u,
        "node-local route control must occupy one cache line");
    static_assert(
        std::is_trivially_copyable_v<MoENodeLocalRouteLaneControl>);

    /**
     * @brief One peer lane as addressed from an exact GPU endpoint.
     *
     * The same host allocation can have a different device virtual address on
     * every CUDA/HIP ordinal.  The fabric resolves those aliases once and puts
     * only endpoint-correct pointers in captured kernel descriptors.
     */
    struct MoENodeLocalRoutePeerDeviceBinding
    {
        MoENodeLocalRouteLaneControl *control = nullptr;
        std::uint64_t *slot_epochs = nullptr; ///< `[route_capacity]` publication tags.
        /** Producer-mapped rows or root-device staged rows, by endpoint role. */
        float *route_payload = nullptr;
        /** Exact mapped source alias used only by the root staging kernel. */
        const float *mapped_route_payload = nullptr;
        std::int32_t producer_participant = -1;
        std::uint32_t route_capacity = 0u;
        std::uint32_t d_model = 0u;

        /** @return Whether every endpoint pointer and immutable scalar is valid. */
        [[nodiscard]] LLAMINAR_MOE_NODE_LOCAL_ROUTE_HD constexpr bool
        valid() const noexcept
        {
            return control != nullptr && slot_epochs != nullptr &&
                   route_payload != nullptr && mapped_route_payload != nullptr &&
                   producer_participant >= 0 &&
                   route_capacity > 0u && d_model > 0u;
        }
    };

    /**
     * @brief Capture-stable final route-to-participant publication.
     *
     * Participant values are always local to the homogeneous continuation
     * domain represented by this channel. `-1` denotes an expert in another
     * ExpertOverlay domain; its contribution is intentionally folded by the
     * heterogeneous return protocol instead of this node-local exchange.
     *
     * Static, Dynamic, and LLEP differ only in how the upstream router/expert
     * transaction chooses these values. Every continuation-domain publisher
     * and reducer consumes this final local ledger verbatim. Overlay-wide
     * domain selection remains in the request-pinned placement bank; keeping
     * the two ID spaces explicit prevents reconstruction from stale setup-time
     * owner metadata.
     */
    struct MoEDomainRouteAssignmentLedger
    {
        /** Final continuation-domain participant, or `-1`, for every route slot. */
        const std::int32_t *participant_ids = nullptr;
        /** Number of route slots backed by @ref participant_ids. */
        std::uint32_t capacity = 0u;

        /**
         * @brief Validate this publication for one concrete graph invocation.
         * @param live_route_slots Number of route slots the consumer will read.
         * @return Whether the device pointer and declared capacity are complete.
         */
        [[nodiscard]] LLAMINAR_MOE_NODE_LOCAL_ROUTE_HD constexpr bool
        validFor(std::uint32_t live_route_slots) const noexcept
        {
            return participant_ids != nullptr && live_route_slots > 0u &&
                   live_route_slots <= capacity;
        }
    };

    /**
     * @brief Authoritative source for routes outside the continuation domain.
     *
     * A negative entry in @ref MoEDomainRouteAssignmentLedger is meaningful,
     * not absent state. Portable inter-node returns are merged into the dense
     * output after the continuation-domain fold, whereas node-local mapped
     * returns first materialize the same original route slot in the root's
     * canonical bank. Encoding that distinction prevents the reducer from
     * guessing based on topology or pointer presence.
     */
    enum class MoEExternalCanonicalRouteSource : std::uint32_t
    {
        Unspecified = 0u, ///< Invalid graph construction state.
        DeferredDenseMerge = 1u, ///< Contribute exact zero; a later dense merge owns it.
        RootCanonicalRouteBank = 2u, ///< Read the materialized root canonical slot.
    };

    /**
     * @brief Producer launch for sparse mapped route publication.
     *
     * `domain_assignment` is the final domain-local device publication used by
     * expert execution. A negative entry names a route owned by another overlay
     * domain; it is not an invalid global placement.
     * The publication kernel copies a route slot only when that ledger selects
     * this producer.
     */
    struct MoENodeLocalRoutePublishLaunch
    {
        MoENodeLocalRoutePeerDeviceBinding lane{};
        const float *canonical_route_contributions = nullptr;
        MoEDomainRouteAssignmentLedger domain_assignment{};
        std::uint32_t live_route_slots = 0u;

        /** @return Whether the fixed-capacity producer launch is complete. */
        [[nodiscard]] LLAMINAR_MOE_NODE_LOCAL_ROUTE_HD constexpr bool
        valid() const noexcept
        {
            return lane.valid() && canonical_route_contributions != nullptr &&
                   domain_assignment.validFor(live_route_slots) &&
                   live_route_slots <= lane.route_capacity;
        }
    };

    /**
     * @brief Root launch for peer acquisition and canonical router-order fold.
     *
     * `peers` resides in root-device memory in ascending topology order.  Route
     * assignment, rather than completion timing, selects exactly one source
     * for each original slot.  Each output element has one writer and folds
     * routes `0..top_k-1` with explicit FP32 rounding.
     */
    struct MoENodeLocalRouteConsumeLaunch
    {
        const MoENodeLocalRoutePeerDeviceBinding *peers = nullptr;
        std::uint32_t peer_count = 0u;
        const float *root_canonical_route_contributions = nullptr;
        MoEDomainRouteAssignmentLedger domain_assignment{};
        MoEExternalCanonicalRouteSource external_route_source =
            MoEExternalCanonicalRouteSource::Unspecified;
        float *dense_output = nullptr;
        std::int32_t *validation_status = nullptr; ///< Root-device terminal status scalar.
        std::int32_t root_participant = -1;
        std::uint32_t physical_rows = 0u;
        std::uint32_t top_k = 0u;
        std::uint32_t d_model = 0u;

        /** @return Whether the root launch has a complete immutable geometry. */
        [[nodiscard]] LLAMINAR_MOE_NODE_LOCAL_ROUTE_HD constexpr bool
        valid() const noexcept
        {
            if (peers == nullptr || peer_count == 0u ||
                root_canonical_route_contributions == nullptr ||
                dense_output == nullptr ||
                validation_status == nullptr ||
                external_route_source ==
                    MoEExternalCanonicalRouteSource::Unspecified ||
                root_participant < 0 || physical_rows == 0u || top_k == 0u ||
                d_model == 0u)
            {
                return false;
            }
            const std::uint64_t slots =
                static_cast<std::uint64_t>(physical_rows) *
                static_cast<std::uint64_t>(top_k);
            return slots <= static_cast<std::uint64_t>(UINT32_MAX) &&
                   domain_assignment.validFor(
                       static_cast<std::uint32_t>(slots));
        }
    };

    /** Binary identity of the dense root-publication channel (`NLDP`). */
    inline constexpr std::uint32_t kMoENodeLocalDensePublicationMagic =
        0x50444c4eu;

    /** Current binary layout version of the dense publication records. */
    inline constexpr std::uint32_t kMoENodeLocalDensePublicationVersion = 1u;

    /** Device-visible lifecycle of the shared dense publication channel. */
    enum class MoENodeLocalDensePublicationState : std::uint32_t
    {
        Ready = 1u, ///< Immutable topology is installed and publication may run.
        Aborted = 2u, ///< A terminal peer/setup failure released every waiter.
    };

    /** Device-visible terminal result for dense publication progress. */
    enum class MoENodeLocalDensePublicationCode : std::uint32_t
    {
        Success = 0u,
        InvalidControl = 1u,
        EpochOverflow = 2u,
        PeerAborted = 3u,
    };

    /**
     * @brief Root-owned progress record for one reusable dense payload bank.
     *
     * The root is the sole writer of `produced_epoch`. A publication may reuse
     * the payload only after every peer acknowledgement equals that epoch. The
     * record occupies its own cache line so peer acknowledgement writes cannot
     * invalidate the root's hot progress word.
     */
    struct alignas(64) MoENodeLocalDensePublicationControl
    {
        std::uint32_t magic = kMoENodeLocalDensePublicationMagic;
        std::uint32_t version = kMoENodeLocalDensePublicationVersion;
        std::uint32_t state = static_cast<std::uint32_t>(
            MoENodeLocalDensePublicationState::Ready);
        std::uint32_t code = static_cast<std::uint32_t>(
            MoENodeLocalDensePublicationCode::Success);
        std::uint32_t peer_count = 0u;
        std::uint32_t d_model = 0u;
        std::uint64_t element_capacity = 0u;
        std::uint64_t produced_epoch = 0u;
        std::uint64_t failed_epoch = 0u;
        std::uint64_t reserved[2] = {};
    };

    static_assert(sizeof(MoENodeLocalDensePublicationControl) == 64u);
    static_assert(
        std::is_trivially_copyable_v<MoENodeLocalDensePublicationControl>);

    /**
     * @brief One peer-owned acknowledgement isolated on a cache line.
     *
     * `participant_id` is immutable setup identity. The named participant is
     * the sole live writer of `consumed_epoch`; the root only acquires it before
     * overwriting the shared payload bank.
     */
    struct alignas(64) MoENodeLocalDensePublicationPeer
    {
        std::int32_t participant_id = -1;
        std::uint32_t state = static_cast<std::uint32_t>(
            MoENodeLocalDensePublicationState::Ready);
        std::uint64_t consumed_epoch = 0u;
        std::uint64_t reserved[6] = {};
    };

    static_assert(sizeof(MoENodeLocalDensePublicationPeer) == 64u);
    static_assert(
        std::is_trivially_copyable_v<MoENodeLocalDensePublicationPeer>);

    /** Role embedded in one endpoint's captured dense-publication binding. */
    enum class MoENodeLocalDensePublicationRole : std::uint32_t
    {
        RootProducer = 1u, ///< Publishes one dense matrix for every peer.
        PeerConsumer = 2u, ///< Imports that matrix and acknowledges its epoch.
    };

    /**
     * @brief Exact endpoint aliases for a captured root-to-peers publication.
     *
     * The root binding addresses the entire sorted peer array. A peer binding
     * additionally addresses only its own acknowledgement cache line. All
     * pointers are resolved through the endpoint's registration and may differ
     * numerically even though they refer to the same physical host pages.
     */
    struct MoENodeLocalDensePublicationDeviceBinding
    {
        MoENodeLocalDensePublicationControl *control = nullptr;
        MoENodeLocalDensePublicationPeer *peers = nullptr;
        MoENodeLocalDensePublicationPeer *local_peer = nullptr;
        float *payload = nullptr;
        std::int32_t participant_id = -1;
        std::int32_t root_participant = -1;
        std::uint32_t peer_count = 0u;
        std::uint64_t element_capacity = 0u;
        MoENodeLocalDensePublicationRole role =
            MoENodeLocalDensePublicationRole::PeerConsumer;

        /** @return Whether every pointer, role, and immutable bound is valid. */
        [[nodiscard]] LLAMINAR_MOE_NODE_LOCAL_ROUTE_HD constexpr bool
        valid() const noexcept
        {
            const bool role_valid =
                role == MoENodeLocalDensePublicationRole::RootProducer ||
                role == MoENodeLocalDensePublicationRole::PeerConsumer;
            const bool peer_pointer_valid =
                role == MoENodeLocalDensePublicationRole::RootProducer
                    ? local_peer == nullptr
                    : local_peer != nullptr;
            return control && peers && payload && participant_id >= 0 &&
                   root_participant >= 0 && peer_count > 0u &&
                   element_capacity > 0u && role_valid && peer_pointer_valid;
        }

        /** @return Whether this endpoint is the sole payload producer. */
        [[nodiscard]] LLAMINAR_MOE_NODE_LOCAL_ROUTE_HD constexpr bool
        isRoot() const noexcept
        {
            return role ==
                   MoENodeLocalDensePublicationRole::RootProducer;
        }
    };

    /**
     * @brief Fixed captured geometry for one dense publication invocation.
     *
     * Payload movement itself is lowered through TransferEngine between the
     * begin/finish kernels. Keeping element count here lets both endpoints
     * authenticate that the captured copy remains inside the setup envelope.
     */
    struct MoENodeLocalDensePublicationLaunch
    {
        MoENodeLocalDensePublicationDeviceBinding binding{};
        std::uint64_t element_count = 0u;

        /** @return Whether the invocation fits the immutable channel capacity. */
        [[nodiscard]] LLAMINAR_MOE_NODE_LOCAL_ROUTE_HD constexpr bool
        valid() const noexcept
        {
            return binding.valid() && element_count > 0u &&
                   element_count <= binding.element_capacity;
        }
    };
} // namespace llaminar2

#undef LLAMINAR_MOE_NODE_LOCAL_ROUTE_HD
