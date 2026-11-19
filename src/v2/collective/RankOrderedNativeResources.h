/**
 * @file RankOrderedNativeResources.h
 * @brief Cold peer-push BOM and measured row/payload reduction selection.
 *
 * The original collective owner quotes its additional physical device backing.
 * This declaration carries that quote into the existing PMA; it is neither a
 * free-memory observation nor a second live allocation ledger. Opaque native
 * bookkeeping retains a generous typed host bound until collective retirement.
 */
#pragma once

#include "backends/DeviceId.h"
#include <cstddef>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>

namespace llaminar2
{
    /** @brief Whether this LocalTP owner can admit the measured optional peer transport to its model ledger. */
    enum class RankOrderedQuotePolicy { ExplicitOnly, MeasuredDomain };

    /** @brief Additional storage owned by one original communicator participant. */
    struct RankOrderedNativeResources
    {
        DeviceId device; ///< Original rank-local GPU, never a logical TP coordinate.
        std::size_t device_bytes = 0; ///< Exact inbox, progress and pointer-table allocation sizes.
        std::size_t metadata_units = 0; ///< Original participant resource owners requiring host bookkeeping.

        /**
         * @return Bound for native host bookkeeping created with these connections.
         * @throws std::overflow_error For an unrepresentable owner inventory.
         *
         * This covers opaque proxy/transport metadata, not GPU payload storage.
         * The bound deliberately allows allocator/ABI variation instead of
         * treating a measured host struct size as an artificial OOM threshold.
         */
        [[nodiscard]] std::size_t hostBytes() const
        {
            constexpr std::size_t per_owner = 64u * 1024u;
            if (metadata_units > std::numeric_limits<std::size_t>::max() / per_owner)
                throw std::overflow_error("Native rank-ordered owner host BOM overflow");
            return metadata_units * per_owner;
        }

        bool operator==(const RankOrderedNativeResources &) const = default;
    };

    /** @brief Largest economical row regime; M64 and above retain native gather/fold. */
    inline constexpr std::size_t rankOrderedPushRowsExclusive = 64;
    /** @brief Bounded resident inbox envelope; row width also participates in dispatch. */
    inline constexpr std::size_t rankOrderedPushMaxElements = 256u * 1024u;

    /**
     * @brief Project the original native inventory into one physical owner.
     * @param resources Complete original communicator quotation, never live counters.
     * @param device Exact physical rank-local endpoint being admitted.
     * @return Its quote, or no contribution when this endpoint is not a member.
     * @throws std::invalid_argument For duplicate physical owner declarations.
     */
    [[nodiscard]] inline std::optional<RankOrderedNativeResources> rankOrderedResourcesForDevice(
        std::span<const RankOrderedNativeResources> resources, DeviceId device)
    {
        std::optional<RankOrderedNativeResources> result;
        for (const auto &resource : resources)
            if (resource.device == device)
            {
                if (result) throw std::invalid_argument("Duplicate native rank-ordered physical owner");
                result = resource;
            }
        return result;
    }

    /**
     * @brief Select only the bank sizes with qualified native-push economics.
     * @param enabled Whether the bound original communicator has quoted native push resources.
     * @param rows Actual retained transaction rows, not arena allocation capacity.
     * @param columns Exact FP32 row width; together these determine useful payload.
     * @return Whether the small-bank native implementation owns this operation.
     *
     * The native push and gather/fold implementations share ascending-rank
     * FP32 arithmetic. Larger banks retain their separately measured path.
     */
    [[nodiscard]] constexpr bool useNativeRankOrderedPush(bool enabled, std::size_t rows,
                                                         std::size_t columns) noexcept
    { return enabled && rows > 0 && rows < rankOrderedPushRowsExclusive && columns > 0 &&
             rows <= rankOrderedPushMaxElements / columns; }
}
