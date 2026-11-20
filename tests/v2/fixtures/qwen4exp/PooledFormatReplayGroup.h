/**
 * @file PooledFormatReplayGroup.h
 * @brief Partition original GGUF format replays across process-owned graph budgets.
 *
 * Each supported format has exactly one group. The unsupported descriptor is
 * checked independently and never advances the ordinal. Each process can thus
 * retire its real graph population before the next group's admission.
 */
#pragma once

#include <cstddef>
#include <stdexcept>

namespace llaminar2::test
{
    /** @brief One of the disjoint subsets of the canonical embedding inventory. */
    class PooledFormatReplayGroup
    {
    public:
        static constexpr unsigned Count = 6; ///< Default group count for provider replay.

        /** @brief Bind an explicit inventory group, rejecting indices outside the sweep. */
        explicit PooledFormatReplayGroup(unsigned group, unsigned count = Count) : group_(group), count_(count)
        {
            if (group >= count) throw std::out_of_range("Invalid pooled format replay group");
        }

        /** @return Whether this process owns the format at its original supported ordinal. */
        [[nodiscard]] bool includes(std::size_t ordinal) const noexcept { return ordinal % count_ == group_; }

    private:
        unsigned group_; ///< A group names a process partition, independent of device and cache capacity.
        unsigned count_; ///< Number of processes required for this source inventory.
    };
}
