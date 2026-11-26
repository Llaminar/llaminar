/**
 * @file CPUNativeOrderedPartitionPlan.cpp
 * @brief Cold, fatal reporters for rejected superblock walks and ordered plans.
 *
 * Kept out of line so the kernel headers that admit walks and plans carry no
 * logging or stream dependency. LLAMINAR_UNREACHABLE is always active; it
 * throws, and these functions are noexcept, so a rejection terminates the
 * process in Release as well as Integration. An inadmissible range is a
 * broken invariant, never a recoverable request: entering the walk could
 * stop advancing or overflow the kernels' int partition arithmetic.
 */
#include "CPUNativeOrderedPartitionPlan.h"

#include "utils/Assertions.h"

namespace llaminar2::cpu::native_quantised {
void failSuperblockWalk(int first, int last, std::size_t blocks, std::size_t phase) noexcept {
    LLAMINAR_UNREACHABLE("Invalid CPU superblock walk range: first=" << first << " last=" << last
                         << " blocks=" << blocks << " phase=" << phase
                         << " (requires 0 <= first <= last <= blocks <= INT_MAX and phase < 8)");
}

void failOrderedPartitionPlan(int begin, int end, int partitions, std::int64_t width) noexcept {
    // Report the exact widest intermediate the kernels would form, in int64.
    const std::int64_t extent = partitions > 0 && width >= 0 ? std::int64_t(partitions) * width : -1;
    LLAMINAR_UNREACHABLE("Invalid CPU ordered partition plan: ordered partition geometry exceeds int partition "
                         "arithmetic or does not cover its range: begin=" << begin << " end=" << end
                         << " partitions=" << partitions << " width=" << width << " partitions*width=" << extent
                         << " begin+partitions*width=" << (extent < 0 ? extent : std::int64_t(begin) + extent)
                         << " (requires partitions >= 1, width >= 0, partitions*width >= end-begin and "
                            "begin+partitions*width <= INT_MAX)");
}
}
