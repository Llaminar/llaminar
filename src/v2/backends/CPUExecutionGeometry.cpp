/**
 * @file CPUExecutionGeometry.cpp
 * @brief Shares one real CPU policy observation between inventory and kernels.
 *
 * Observe the same process that will execute the work. MPI publication carries
 * this value to planning root, so no remote estimate calls local detection.
 * Missing CPUID fields remain missing; guessed historical cache defaults are
 * not admissible evidence for an exact workspace allocation.
 */
#include "CPUExecutionGeometry.h"
#include "utils/CPUFeatures.h"

namespace llaminar2
{
    const CPUExecutionGeometry &CPUExecutionGeometry::local()
    {
        static const CPUExecutionGeometry observation = [] {
            CPUExecutionGeometry result;
            std::uint32_t l2 = 0, l3 = 0, l2_ways = 0, l3_ways = 0;
            if (detail::detect_cache_by_level(2, false, l2, nullptr, &l2_ways) &&
                detail::detect_cache_by_level(3, false, l3, nullptr, &l3_ways))
                result.cache = {l2, l3, l2_ways, l3_ways};
            const auto isa = activeISALevel();
            result.maximum_native_row_tile = isa >= ISALevel::AVX512 ? 4 :
                (isa >= ISALevel::AVX2 ? 2 : 0);
            return result;
        }();
        return observation;
    }
}
