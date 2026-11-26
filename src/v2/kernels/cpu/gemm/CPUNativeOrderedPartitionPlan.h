/**
 * @file CPUNativeOrderedPartitionPlan.h
 * @brief The single authority for ordered K-partition intervals and the
 *        noexcept predicates that admit superblock walks and partition plans.
 *
 * Native CPU projection kernels walk a K range one (possibly ragged) physical
 * superblock at a time. The walk advances only while its range lies inside the
 * row extent, so every range must be admitted before the kernel's hot programs
 * run. Ordered-partition kernels obtain each partition's range from
 * orderedPartitionInterval(); a plan admitted once per call makes every such
 * interval an admitted walk by construction, so the per-partition hot programs
 * carry no check.
 *
 * This header is deliberately free of logging and stream headers: the NEON
 * tile programs include it directly, and their register/stack proof binds the
 * complete dependency closure, which must not acquire iostream initialization
 * assembly. CPUNativeSuperblockWalkContract.h adds the fatal admission wrappers.
 */
#pragma once
#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstdint>
namespace llaminar2::cpu::native_quantised {
/** @brief Whether a walk over [first, last) of a row can always advance an int cursor.
 * @param first Inclusive partition start.
 * @param last Exclusive partition end.
 * @param blocks Physical row extent.
 * @param phase Wire superblock phase.
 * @return 0 <= first <= last <= blocks <= INT_MAX and phase < 8.
 */
[[nodiscard]] constexpr bool admitsSuperblockWalk(int first, int last, std::size_t blocks, std::size_t phase = 0) noexcept {
    return blocks <= INT_MAX && phase < 8 && first >= 0 && first <= last && static_cast<std::size_t>(last) <= blocks;
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
 * @pre admitsOrderedPartitionPlan() holds for (begin, end, width) and
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

/** @brief Whether a whole ordered partition plan forms only admitted walks.
 * @param begin Inclusive start of the partitioned range.
 * @param end Exclusive end of the partitioned range.
 * @param blocks Physical row extent.
 * @param partitions Partition count.
 * @param width Blocks per partition.
 * @param phase Wire superblock phase.
 * @return [begin, end) is an admitted walk, there is at least one partition and
 *         the nonnegative width's partitions cover the range.
 * @details Every interval orderedPartitionInterval() then produces lies inside
 * [begin, end) and is itself an admitted walk, so kernels admit the plan once
 * per call and need no per-partition or per-block check.
 */
[[nodiscard]] constexpr bool admitsOrderedPartitionPlan(int begin, int end, std::size_t blocks, int partitions, int width,
                                                        std::size_t phase = 0) noexcept {
    return admitsSuperblockWalk(begin, end, blocks, phase) && partitions > 0 && width >= 0 &&
           std::int64_t(partitions) * width >= std::int64_t(end) - begin;
}
}
