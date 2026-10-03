/**
 * @file MoERouterRowPacket.h
 * @brief Lossless selected-route packets for device-owned token-row routing.
 *
 * A packet contains only live (expert ID, probability) pairs in original top-k
 * order. Its immutable capacity is a storage/stride contract, never the wire
 * extent. The existing router writes its count with top-k publication; a graph
 * collective acquires every peer count before the local publication kernel
 * restores complete route arrays. No router logits or hidden rows cross devices.
 */
#pragma once

#include "DeviceRowPartition.h"
#include <cstddef>
#include <limits>

namespace llaminar2
{
    /** @brief Unchanged FP32 route representation, copied without arithmetic. */
    struct alignas(8) MoERouterSelectedRoute
    {
        float expert; ///< Existing exact FP32 expert ID, not a precision conversion.
        float weight; ///< Original normalized or unnormalized probability bits.
    };
    static_assert(sizeof(MoERouterSelectedRoute) == 2 * sizeof(float));
    static_assert(std::is_trivially_copyable_v<MoERouterSelectedRoute>);

    /** @brief Frozen packet storage contract shared by production and tests. */
    struct MoERouterRowPacketLayout
    {
        DeviceRowPartition partition;
        int capacity = 0;
        int top_k = 0;

        /** @return Whether counts and every peer-major storage offset fit size_t. */
        [[nodiscard]] bool valid() const noexcept
        {
            if (capacity < 2 || top_k <= 0 || partition.participants() < 2)
                return false;
            const auto rows = DeviceRowPartition::resolveMember(
                capacity, capacity, 0, partition.participants()).count;
            const auto limit = std::numeric_limits<std::size_t>::max();
            return std::size_t(rows) <= limit / sizeof(MoERouterSelectedRoute) / std::size_t(top_k) /
                std::size_t(partition.participants());
        }

        /** @return Uniform peer slot capacity, including the largest owner only.
         * @throws std::invalid_argument For invalid or overflowing geometry. */
        [[nodiscard]] std::size_t packetBytes() const
        {
            if (!valid()) throw std::invalid_argument("Invalid owned-router packet geometry");
            return std::size_t(DeviceRowPartition::resolveMember(
                capacity, capacity, 0, partition.participants()).count) * top_k * sizeof(MoERouterSelectedRoute);
        }
    };

    /** @brief Complete local publication after the explicit counted collective.
     *
     * Peer banks exclude this participant, ordered by original participant ID.
     * Counts are one contiguous acquired uint64 per original source. No count
     * is copied to the host. Output arrays must be disjoint from both packets.
     */
    struct MoERouterRowPublication
    {
        MoERouterRowPacketLayout layout;
        const std::int32_t *live_rows = nullptr;
        const MoERouterSelectedRoute *local = nullptr;
        const void *peers = nullptr;
        const std::uint64_t *acquired_bytes = nullptr;
        float *indices = nullptr;
        float *weights = nullptr;

        /** @return Whether immutable launch geometry and mandatory bindings exist. */
        [[nodiscard]] bool valid() const noexcept
        { return layout.valid() && live_rows && local && peers && acquired_bytes && indices && weights; }
    };

    namespace cuda
    {
        /** @brief Copy exact acquired selections into complete original row order.
         * @param publication Immutable banks, counts and device-owned live prefix.
         * @param stream Exact non-null producer/consumer graph stream.
         * @return False for malformed launch bindings or a native enqueue error. */
        bool publishRouterOwnedRows(const MoERouterRowPublication &publication, void *stream);
    }
    namespace rocm
    {
        /** @brief Publish the same byte-exact route and inactive-suffix contract on HIP.
         * @param publication Immutable banks, counts and device-owned live prefix.
         * @param stream Exact non-null producer/consumer graph stream.
         * @return False for malformed launch bindings or a native enqueue error. */
        bool publishRouterOwnedRows(const MoERouterRowPublication &publication, void *stream);
    }
}
