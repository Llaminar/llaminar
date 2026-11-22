/**
 * @file RaggedRowCompaction.h
 * @brief Device-authored ragged-bank compaction validation (design §12.4).
 *
 * Ragged request work uses device-authored compaction/scatter to original
 * coordinates or a proved native ragged protocol. Passing the first request's
 * live count for a whole ragged bank is INVALID: the compaction must name the
 * exact original row for every candidate and the candidate count must equal
 * the number of nonempty rows. This host-side validator runs on the
 * device-authored artifact only; it never fabricates a participation set
 * (no host-selected oracle).
 */

#pragma once

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace llaminar2
{
    /** @brief One compacted candidate mapped back to its original row. */
    struct CompactedRow
    {
        std::uint32_t original_row = 0;
        std::uint64_t live_count = 0; ///< Exact live length of that original row.
    };

    /** @brief Device-authored compaction artifact (never host-fabricated). */
    struct RaggedCompaction
    {
        std::vector<CompactedRow> candidates;
        bool device_authored = true;
    };

    /**
     * @brief Validate a ragged bank compaction against per-row live counts.
     *
     * @param live_counts Per-original-row live lengths (empty entry = inactive).
     * @param compaction Device-authored candidate mapping.
     * @throws std::logic_error On any mismatch, or when the compaction is not
     *         device authored, or when a single live count would have to be
     *         reused for the whole bank (callers express the whole bank below).
     */
    inline void validateRaggedBank(const std::vector<std::uint64_t> &live_counts,
                                   const RaggedCompaction &compaction)
    {
        if (!compaction.device_authored)
            throw std::logic_error("Ragged compaction must be device authored");
        std::uint64_t nonempty = 0;
        for (const auto c : live_counts)
            if (c > 0)
                ++nonempty;
        if (compaction.candidates.size() != nonempty)
            throw std::logic_error("Compaction candidate count must equal nonempty row count");
        std::vector<bool> seen(live_counts.size(), false);
        for (const auto &row : compaction.candidates)
        {
            if (row.original_row >= live_counts.size())
                throw std::logic_error("Compaction names an out-of-range original row");
            if (seen[row.original_row])
                throw std::logic_error("Compaction maps an original row more than once");
            if (live_counts[row.original_row] == 0)
                throw std::logic_error("Compaction maps an inactive row");
            if (live_counts[row.original_row] != row.live_count)
                throw std::logic_error("Compaction live count mismatch for an original row");
            seen[row.original_row] = true;
        }
    }

    /**
     * @brief Reject a whole-bank single live count (the invalid shortcut).
     */
    inline void rejectWholeBankSingleLiveCount(std::uint64_t, const RaggedCompaction &)
    {
        // The validator above already enforces per-row exactness; this helper
        // exists to make the banned pattern unrepresentable at the call site.
        throw std::logic_error("Whole-bank single live count is forbidden for ragged banks");
    }
} // namespace llaminar2
