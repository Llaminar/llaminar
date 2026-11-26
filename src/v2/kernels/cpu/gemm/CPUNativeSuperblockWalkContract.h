/**
 * @file CPUNativeSuperblockWalkContract.h
 * @brief Host admission for bounded, strictly progressing superblock walks and
 *        the single authority for ordered K-partition intervals.
 *
 * Native CPU projection kernels walk a K range one (possibly ragged) physical
 * superblock at a time. The walk advances only while its range lies inside the
 * row extent, so every range must be admitted before the kernel's hot programs
 * run. Ordered-partition kernels obtain each partition's range from
 * orderedPartitionInterval(); a plan admitted once per call by
 * requireOrderedPartitionPlan() makes every such interval a valid walk by
 * construction, so the per-partition hot programs carry no check.
 */
#pragma once
#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstdint>
#include "utils/Assertions.h"
namespace llaminar2::cpu::native_quantised {
/** @brief Reject ranges whose clipped superblock end cannot advance an int cursor.
 * @param first Inclusive partition start.
 * @param last Exclusive partition end.
 * @param blocks Physical row extent.
 * @param phase Wire superblock phase.
 */
inline void requireSuperblockWalk(int first, int last, std::size_t blocks, std::size_t phase = 0) noexcept {
    if (!(blocks <= INT_MAX && phase < 8 && first >= 0 && first <= last &&
          static_cast<std::size_t>(last) <= blocks))
        LLAMINAR_UNREACHABLE("Invalid CPU superblock walk range");
}

/** @brief One ordered K partition's block range [first, last). */
struct OrderedPartitionInterval {
    int first; ///< Inclusive first block.
    int last;  ///< Exclusive end block.
};

/** @brief The partition'th interval of an ordered plan over [begin, end).
 * @param begin Inclusive start of the partitioned range.
 * @param end Exclusive end of the partitioned range.
 * @param partition Zero-based partition index.
 * @param width Blocks per partition.
 * @return [begin + partition * width, + width), both ends saturated at end.
 * @pre requireOrderedPartitionPlan() admitted (begin, end, width) and
 *      0 <= partition < partitions.
 * @details The start and end are formed in int64 (|partition * width| < 2^62,
 * so nothing overflows) and saturate at end, so begin <= first <= last <= end
 * for every partition: an excess partition is an explicit empty interval
 * instead of a reversed range. Every nonempty interval equals the unsaturated
 * form, so the partition count, fold order and arithmetic are unchanged.
 */
[[nodiscard, gnu::always_inline]] constexpr OrderedPartitionInterval orderedPartitionInterval(
    int begin, int end, int partition, int width) noexcept {
    const std::int64_t first = std::min<std::int64_t>(std::int64_t(begin) + std::int64_t(partition) * width, end);
    const std::int64_t last = std::min<std::int64_t>(first + width, end);
    return {static_cast<int>(first), static_cast<int>(last)};
}

/** @brief Admit a whole ordered partition plan once per call, before any hot program.
 * @param begin Inclusive start of the partitioned range.
 * @param end Exclusive end of the partitioned range.
 * @param blocks Physical row extent.
 * @param partitions Partition count.
 * @param width Blocks per partition.
 * @param phase Wire superblock phase.
 * @details Admits [begin, end) as a superblock walk, at least one partition and
 * a nonnegative width whose partitions cover the range. Every interval
 * orderedPartitionInterval() then produces lies inside [begin, end) and is
 * itself an admitted walk, so kernels need no per-partition or per-block check.
 */
inline void requireOrderedPartitionPlan(int begin, int end, std::size_t blocks, int partitions, int width,
                                        std::size_t phase = 0) noexcept {
    requireSuperblockWalk(begin, end, blocks, phase);
    if (!(partitions > 0 && width >= 0 &&
          std::int64_t(partitions) * width >= std::int64_t(end) - begin))
        LLAMINAR_UNREACHABLE("Invalid CPU ordered partition plan");
}
}
