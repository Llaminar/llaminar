/**
 * @file PrefixPlacementEpochSpan.h
 * @brief Lossless placement-publication interval for coordinated prefix lookup.
 *
 * Participant lookups are independently admitted while expert maintenance may
 * publish a newer placement. Keep both ends of that immutable observation:
 * MAX alone hides a stale child, while MIN alone hides publication during the
 * next lookup. This is request evidence, never a live placement authority.
 */
#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace llaminar2
{
    /**
     * @brief Validated inclusive span of participant admission epochs.
     *
     * Epoch zero is the ordinary non-moving placement. Accumulators must start
     * with their first participant rather than merging an artificial zero.
     */
    class PrefixPlacementEpochSpan
    {
    public:
        /** @brief Construct the non-moving epoch-zero admission. */
        constexpr PrefixPlacementEpochSpan() noexcept = default;

        /**
         * @param epoch Placement observed by one participant's lookup.
         * @return One participant's exact, immutable admission epoch.
         */
        [[nodiscard]] static constexpr PrefixPlacementEpochSpan at(
            uint64_t epoch) noexcept
        {
            return PrefixPlacementEpochSpan(epoch, epoch);
        }

        /**
         * @brief Authenticate the endpoints returned by a collective reduction.
         * @param earliest Minimum epoch admitted by any required participant.
         * @param latest Maximum epoch admitted by any required participant.
         * @return Ordered admission interval.
         * @throws std::invalid_argument If the reduction produced reversed ends.
         */
        [[nodiscard]] static constexpr PrefixPlacementEpochSpan covering(
            uint64_t earliest, uint64_t latest)
        {
            if (earliest > latest)
                throw std::invalid_argument("Prefix admission epoch span is reversed");
            return PrefixPlacementEpochSpan(earliest, latest);
        }

        /** @return Oldest participant admission, including nested aggregates. */
        [[nodiscard]] constexpr uint64_t earliest() const noexcept { return earliest_; }
        /** @return Newest participant admission, including nested aggregates. */
        [[nodiscard]] constexpr uint64_t latest() const noexcept { return latest_; }

        /**
         * @brief Union independently admitted participants without losing either end.
         * @param other Another leaf or nested coordination result.
         * @return Smallest span containing both observations.
         */
        [[nodiscard]] constexpr PrefixPlacementEpochSpan mergedWith(
            PrefixPlacementEpochSpan other) const noexcept
        {
            return PrefixPlacementEpochSpan(
                std::min(earliest_, other.earliest_),
                std::max(latest_, other.latest_));
        }

        /** @return Whether both immutable admission endpoints match. */
        constexpr bool operator==(const PrefixPlacementEpochSpan &) const noexcept = default;

    private:
        /** @brief Construct endpoints already proven ordered by a public operation. */
        constexpr PrefixPlacementEpochSpan(uint64_t earliest, uint64_t latest) noexcept
            : earliest_(earliest), latest_(latest) {}

        uint64_t earliest_ = 0;
        uint64_t latest_ = 0;
    };
} // namespace llaminar2
