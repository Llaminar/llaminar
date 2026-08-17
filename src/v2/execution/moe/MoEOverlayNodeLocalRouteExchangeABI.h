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
     * @brief Producer launch for sparse mapped route publication.
     *
     * `route_participant_ids` is the device-owned runtime assignment after
     * static-owner, Dynamic, or LLEP policy has run.  The publication kernel
     * copies a route slot only when that array selects this producer.
     */
    struct MoENodeLocalRoutePublishLaunch
    {
        MoENodeLocalRoutePeerDeviceBinding lane{};
        const float *canonical_route_contributions = nullptr;
        const std::int32_t *route_participant_ids = nullptr;
        std::uint32_t live_route_slots = 0u;

        /** @return Whether the fixed-capacity producer launch is complete. */
        [[nodiscard]] LLAMINAR_MOE_NODE_LOCAL_ROUTE_HD constexpr bool
        valid() const noexcept
        {
            return lane.valid() && canonical_route_contributions != nullptr &&
                   route_participant_ids != nullptr && live_route_slots > 0u &&
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
        const std::int32_t *route_participant_ids = nullptr;
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
                route_participant_ids == nullptr || dense_output == nullptr ||
                validation_status == nullptr ||
                root_participant < 0 || physical_rows == 0u || top_k == 0u ||
                d_model == 0u)
            {
                return false;
            }
            const std::uint64_t slots =
                static_cast<std::uint64_t>(physical_rows) *
                static_cast<std::uint64_t>(top_k);
            return slots <= static_cast<std::uint64_t>(UINT32_MAX);
        }
    };
} // namespace llaminar2

#undef LLAMINAR_MOE_NODE_LOCAL_ROUTE_HD
