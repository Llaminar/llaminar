/**
 * @file CPUNativeOrderedPartitionPlan.h
 * @brief The single admission authority for native CPU superblock walks and
 *        ordered K-partition plans.
 *
 * Native CPU projection kernels walk a K range one (possibly ragged) physical
 * superblock at a time, and ordered-partition kernels form each partition's
 * range with the int expressions
 *
 *     first = begin + partition * width,   last = min(first + width, end)
 *
 * in their hot programs. Those expressions are deliberately left unchanged;
 * this header proves them safe once, in cold code, instead:
 *
 * - A walk over [first, last) advances whenever 0 <= first <= last <= blocks
 *   <= INT_MAX and phase < 8. Every entered iteration has block < last, the
 *   untruncated superblock end (at most INT_MAX + 15 in size_t) is greater than
 *   block, and clipping it to last therefore still advances and fits int.
 * - For every partition p < partitions, the widest int intermediate is
 *   begin + p * width + width <= begin + partitions * width. The plan is
 *   therefore representable in the kernels' int arithmetic exactly when
 *   int64(begin) + int64(partitions) * width <= INT_MAX. This is the precise
 *   API representability constraint, not a capacity or topology cap: a row of
 *   INT_MAX blocks in one partition is admitted, while INT_MAX blocks in two
 *   partitions (width 2^30, product 2^31) genuinely exceeds the int arithmetic.
 * - An excess partition then has first >= last: an empty (possibly reversed)
 *   interval whose `block < last` walks load nothing and terminate at once.
 *
 * The header has no logging or stream dependency: the NEON tile programs
 * include it, and their register/stack proof binds the probe's complete
 * dependency closure, which must not acquire iostream initialization assembly.
 * Rejections call cold, out-of-line, noreturn reporters defined in
 * CPUNativeOrderedPartitionPlan.cpp; they are fatal in every build type.
 */
#pragma once
#include <climits>
#include <cstddef>
#include <cstdint>
namespace llaminar2::cpu::native_quantised {
/** @brief Report an inadmissible superblock walk and terminate.
 * @param first Inclusive start that was requested.
 * @param last Exclusive end that was requested.
 * @param blocks Physical row extent.
 * @param phase Wire superblock phase.
 */
[[noreturn, gnu::cold, gnu::noinline]] void failSuperblockWalk(int first, int last, std::size_t blocks,
                                                              std::size_t phase) noexcept;

/** @brief Report an ordered partition plan outside the kernels' int arithmetic and terminate.
 * @param begin Inclusive start of the partitioned range.
 * @param end Exclusive end of the partitioned range.
 * @param partitions Partition count.
 * @param width Blocks per partition, exactly as computed (int64).
 */
[[noreturn, gnu::cold, gnu::noinline]] void failOrderedPartitionPlan(int begin, int end, int partitions,
                                                                    std::int64_t width) noexcept;

/** @brief Whether a walk over [first, last) of a row can always advance an int cursor.
 * @param first Inclusive start.
 * @param last Exclusive end.
 * @param blocks Physical row extent.
 * @param phase Wire superblock phase.
 * @return 0 <= first <= last <= blocks <= INT_MAX and phase < 8.
 */
[[nodiscard]] constexpr bool admitsSuperblockWalk(int first, int last, std::size_t blocks,
                                                  std::size_t phase = 0) noexcept {
    return blocks <= INT_MAX && phase < 8 && first >= 0 && first <= last && static_cast<std::size_t>(last) <= blocks;
}

/** @brief Whether a plan's partitions cover [begin, end) within the kernels' int arithmetic.
 * @param begin Inclusive start of the partitioned range (nonnegative).
 * @param end Exclusive end of the partitioned range.
 * @param partitions Partition count.
 * @param width Blocks per partition.
 * @return partitions >= 1, width >= 0, partitions * width >= end - begin and
 *         begin + partitions * width <= INT_MAX, all evaluated exactly in int64.
 */
[[nodiscard]] constexpr bool admitsOrderedPartitionArithmetic(int begin, int end, int partitions,
                                                              std::int64_t width) noexcept {
    if (begin < 0 || partitions < 1 || width < 0 || width > INT_MAX)
        return false;
    const std::int64_t extent = std::int64_t(partitions) * width; // < 2^62: exact.
    return extent >= std::int64_t(end) - begin && std::int64_t(begin) + extent <= INT_MAX;
}

/** @brief Whether a whole ordered plan over [begin, end) of a row is admitted.
 * @return admitsSuperblockWalk(begin, end, blocks, phase) and admitsOrderedPartitionArithmetic().
 * @details Every interval the kernels then form lies in [begin, end] or is
 * empty, and none of their int intermediates overflows.
 */
[[nodiscard]] constexpr bool admitsOrderedPartitionPlan(int begin, int end, std::size_t blocks, int partitions,
                                                        std::int64_t width, std::size_t phase = 0) noexcept {
    return admitsSuperblockWalk(begin, end, blocks, phase) &&
           admitsOrderedPartitionArithmetic(begin, end, partitions, width);
}

/** @brief Admit one walk range, once per kernel entry; fatal otherwise.
 * @param first Inclusive start.
 * @param last Exclusive end.
 * @param blocks Physical row extent.
 * @param phase Wire superblock phase.
 */
inline void requireSuperblockWalk(int first, int last, std::size_t blocks, std::size_t phase = 0) noexcept {
    if (!admitsSuperblockWalk(first, last, blocks, phase)) [[unlikely]]
        failSuperblockWalk(first, last, blocks, phase);
}

/** @brief Compute and admit the canonical width of an ordered plan; fatal otherwise.
 * @param begin Inclusive start of the partitioned range.
 * @param end Exclusive end of the partitioned range.
 * @param blocks Physical row extent.
 * @param partitions Partition count.
 * @param phase Wire superblock phase.
 * @return ceil((end - begin) / partitions), computed in int64 and admitted to fit
 *         the kernels' int arithmetic, for storage in an existing int field.
 * @details Every producer of a canonical plan (the NEON tile constructor, its
 * once-per-call publication admission and the NEON packed-prefill chunk) calls
 * this instead of an int ceil, which could itself overflow near INT_MAX.
 */
[[nodiscard]] inline int requireOrderedPartitionWidth(int begin, int end, std::size_t blocks, int partitions,
                                                      std::size_t phase = 0) noexcept {
    requireSuperblockWalk(begin, end, blocks, phase);
    if (partitions < 1) [[unlikely]]
        failOrderedPartitionPlan(begin, end, partitions, 0);
    const std::int64_t width = (std::int64_t(end) - begin + partitions - 1) / partitions;
    if (!admitsOrderedPartitionArithmetic(begin, end, partitions, width)) [[unlikely]]
        failOrderedPartitionPlan(begin, end, partitions, width);
    return static_cast<int>(width);
}
}
