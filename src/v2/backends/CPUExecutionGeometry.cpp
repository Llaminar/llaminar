/**
 * @file CPUExecutionGeometry.cpp
 * @brief Shares one real CPU policy observation between inventory and kernels.
 *
 * Observe the same process that will execute the work. MPI publication carries
 * this value to planning root, so no remote estimate calls local detection.
 * Missing cache fields remain missing; guessed historical cache defaults are
 * not admissible evidence for an exact workspace allocation. Native ARM's
 * register-resident K implementation does not consume x86 cache-derived
 * partial geometry and therefore publishes its distinct reduction-storage ABI.
 */
#include "CPUExecutionGeometry.h"
#include "utils/CPUFeatures.h"

namespace llaminar2
{
    const CPUExecutionGeometry &CPUExecutionGeometry::local()
    {
        static const CPUExecutionGeometry observation = [] {
            CPUExecutionGeometry result;
            const auto isa = activeISALevel();
            result.compiled_isa = compiledCPUISALevel();
            result.active_isa = isa;
#if defined(__aarch64__) && defined(__ARM_NEON)
            if (isa == ISALevel::NEON || isa == ISALevel::NEONDotProd)
            {
                result.maximum_native_row_tile = 4;
                result.projection_abi = CPUProjectionScheduleABI::ARMRegisterK;
            }
#else
            const auto &cache = cpu_cache_observation();
            result.cache = {cache.private_l2_bytes, cache.shared_l3_bytes,
                cache.private_l2_ways, cache.shared_l3_ways};
            result.maximum_native_row_tile = isa == ISALevel::AVX512 ? 4 :
                (isa == ISALevel::AVX2 ? 2 : 0);
            if (result.maximum_native_row_tile)
                result.projection_abi = CPUProjectionScheduleABI::X86SharedK;
#endif
            return result;
        }();
        return observation;
    }
}
