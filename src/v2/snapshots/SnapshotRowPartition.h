/**
 * @file SnapshotRowPartition.h
 * @brief Explicit row ownership for immutable diagnostic snapshots.
 *
 * A device producer writes a compact balanced slice. Only after diagnostic
 * observation may it be projected into the logical host tensor with declared
 * owned intervals. Unowned host cells are never added or interpreted as data.
 * Chunk concatenation offsets intervals; TP assembly copies each row exactly
 * once. This does not gather probabilities on the inference wire or own live
 * execution counts.
 */
#pragma once
#include "../kernels/common/DeviceRowPartition.h"
#include <algorithm>
#include <cstring>
#include <limits>
#include <span>
#include <variant>
#include <vector>

namespace llaminar2
{
    /** @brief A nonempty half-open interval owned by one diagnostic producer. */
    struct SnapshotRowInterval
    {
        std::size_t begin = 0, count = 0;
        bool operator==(const SnapshotRowInterval &) const = default;
    };
    /** @brief Physical producer layout, independent of logical diagnostic ownership. */
    enum class SnapshotRowStorage { CompactOwned, CompleteReplicated };
    /** @brief Frozen row-owned diagnostic geometry, before terminal observation.
     * Small replicated buckets can expose the same disjoint diagnostic partition
     * as bulk compact buckets. This changes observation only, never inference math. */
    struct SnapshotCompactRows
    {
        DeviceRowPartition partition;
        int capacity = 0;
        SnapshotRowStorage storage = SnapshotRowStorage::CompactOwned;
    };
    /** @brief Resolved host ownership; an empty list is a valid empty participant. */
    struct SnapshotOwnedRows
    {
        std::vector<SnapshotRowInterval> intervals;
    };
    /** @brief Diagnostic lifecycle: ordinary, captured compact, or observed ownership. */
    using SnapshotRowLayout = std::variant<std::monostate, SnapshotCompactRows, SnapshotOwnedRows>;

    /** @brief Validate ordered disjoint intervals against their logical shape.
     * @throws std::invalid_argument On empty, overlapping, unsorted or out-of-bounds intervals. */
    inline void validateSnapshotOwnedRows(const SnapshotOwnedRows &owned, std::size_t rows)
    {
        std::size_t end = 0;
        for (const auto &range : owned.intervals)
        {
            if (!range.count || range.begin < end || range.begin > rows || range.count > rows - range.begin)
                throw std::invalid_argument("Invalid diagnostic row ownership interval");
            end = range.begin + range.count;
        }
    }

    /** @brief Project acquired compact FP32 rows into their diagnostic-only host coordinates.
     * @param source Already observed compact capacity, never a device pointer.
     * @param layout Frozen producer membership and bucket size.
     * @param live_rows Terminal diagnostic logical count, not an inference input.
     * @param cols Values per row.
     * @param output Host-only full logical tensor, with unowned cells explicitly irrelevant.
     * @return The exact rows whose bits were copied, including empty ownership.
     * @throws std::invalid_argument For missing capacity or impossible geometry. */
    inline SnapshotOwnedRows projectSnapshotOwnedRows(std::span<const float> source,
        SnapshotCompactRows layout, int live_rows, std::size_t cols, std::vector<float> &output)
    {
        if (layout.capacity < 1 || live_rows < 0 || live_rows > layout.capacity || !cols ||
            std::size_t(layout.capacity) > std::numeric_limits<std::size_t>::max() / cols)
            throw std::invalid_argument("Invalid compact diagnostic row geometry");
        const auto admitted = layout.partition.resolveFor(layout.capacity, layout.capacity);
        const auto live = layout.partition.resolveFor(layout.capacity, live_rows);
        const auto source_rows = layout.storage == SnapshotRowStorage::CompleteReplicated
            ? layout.capacity : admitted.count;
        if (source.size() != std::size_t(source_rows) * cols)
            throw std::invalid_argument("Compact diagnostic storage does not match its producer");
        output.assign(std::size_t(live_rows) * cols, 0.0f);
        SnapshotOwnedRows owned;
        if (live.count)
        {
            std::memcpy(output.data() + std::size_t(live.first) * cols,
                        source.data() + (layout.storage == SnapshotRowStorage::CompleteReplicated
                            ? std::size_t(live.first) * cols : 0),
                        std::size_t(live.count) * cols * sizeof(float));
            owned.intervals.push_back({std::size_t(live.first), std::size_t(live.count)});
        }
        return owned;
    }

    /** @brief Immutable observed source for pure host diagnostic assembly. */
    struct SnapshotRowSource
    {
        std::span<const float> data;
        std::size_t rows = 0, cols = 0;
        const SnapshotOwnedRows *owned = nullptr;
    };

    /** @brief Copy each logical row exactly once, without arithmetic or rank-order assumptions.
     * @throws std::invalid_argument For mixed geometry, missing coverage or duplicate ownership.
     * @return Complete byte-identical diagnostic data. */
    inline std::vector<float> assembleSnapshotOwnedRows(std::span<const SnapshotRowSource> sources)
    {
        if (sources.empty() || !sources.front().rows || !sources.front().cols)
            throw std::invalid_argument("Missing row-partitioned diagnostic sources");
        const auto rows = sources.front().rows, cols = sources.front().cols;
        if (rows > std::numeric_limits<std::size_t>::max() / cols)
            throw std::invalid_argument("Overflowed row-partitioned diagnostic shape");
        std::vector<float> result(rows * cols);
        std::vector<bool> covered(rows, false);
        for (const auto &source : sources)
        {
            if (source.rows != rows || source.cols != cols || source.data.size() != result.size() || !source.owned)
                throw std::invalid_argument("Mixed row-partitioned diagnostic sources");
            validateSnapshotOwnedRows(*source.owned, rows);
            for (const auto &range : source.owned->intervals)
            {
                for (auto row = range.begin; row < range.begin + range.count; ++row)
                {
                    if (covered[row]) throw std::invalid_argument("Duplicate diagnostic row ownership");
                    covered[row] = true;
                }
                std::memcpy(result.data() + range.begin * cols, source.data.data() + range.begin * cols,
                            range.count * cols * sizeof(float));
            }
        }
        if (std::find(covered.begin(), covered.end(), false) != covered.end())
            throw std::invalid_argument("Incomplete diagnostic row ownership");
        return result;
    }
}
