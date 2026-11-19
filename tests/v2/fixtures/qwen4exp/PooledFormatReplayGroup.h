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
    /** @brief One of six disjoint subsets of the canonical embedding inventory. */
    class PooledFormatReplayGroup
    {
    public:
        static constexpr unsigned Count = 6; ///< At most four original formats per process.

        /** @brief Bind an explicit inventory group, rejecting indices outside the sweep. */
        explicit PooledFormatReplayGroup(unsigned group) : group_(group)
        {
            if (group >= Count) throw std::out_of_range("Invalid pooled format replay group");
        }

        /** @return Whether this process owns the format at its original supported ordinal. */
        [[nodiscard]] bool includes(std::size_t ordinal) const noexcept { return ordinal % Count == group_; }

    private:
        unsigned group_; ///< A group names a process partition, independent of device and cache capacity.
    };
}
