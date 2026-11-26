/**
 * @file CPUNativeSuperblockWalkContract.h
 * @brief Fatal host admission for bounded, strictly progressing superblock
 *        walks and ordered K-partition plans.
 *
 * The predicates and the ordered interval authority live in
 * CPUNativeOrderedPartitionPlan.h; this header turns a rejected walk or plan
 * into an always-active fatal error (Release as well as Integration), so an
 * invalid range can never reach a walk that would stop advancing.
 */
#pragma once
#include <cstddef>
#include "CPUNativeOrderedPartitionPlan.h"
#include "utils/Assertions.h"
namespace llaminar2::cpu::native_quantised {
/** @brief Reject ranges whose clipped superblock end cannot advance an int cursor.
 * @param first Inclusive partition start.
 * @param last Exclusive partition end.
 * @param blocks Physical row extent.
 * @param phase Wire superblock phase.
 */
inline void requireSuperblockWalk(int first, int last, std::size_t blocks, std::size_t phase = 0) noexcept {
    if (!admitsSuperblockWalk(first, last, blocks, phase))
        LLAMINAR_UNREACHABLE("Invalid CPU superblock walk range");
}

/** @brief Admit a whole ordered partition plan once per call, before any hot program.
 * @param begin Inclusive start of the partitioned range.
 * @param end Exclusive end of the partitioned range.
 * @param blocks Physical row extent.
 * @param partitions Partition count.
 * @param width Blocks per partition.
 * @param phase Wire superblock phase.
 * @details Fatal unless admitsOrderedPartitionPlan() holds, so every interval
 * orderedPartitionInterval() forms is an admitted walk by construction.
 */
inline void requireOrderedPartitionPlan(int begin, int end, std::size_t blocks, int partitions, int width,
                                        std::size_t phase = 0) noexcept {
    requireSuperblockWalk(begin, end, blocks, phase);
    if (!admitsOrderedPartitionPlan(begin, end, blocks, partitions, width, phase))
        LLAMINAR_UNREACHABLE("Invalid CPU ordered partition plan");
}
}
