/**
 * @file ROCmFullNativeHCFFNBody.hip.inc
 * @brief Tuned canonical HC and FFN phase composition in the existing native entry.
 *
 * Include after the original reusable numerical bodies. The third HC edge
 * publishes mixed hidden/injection; the prefix then retires its local public
 * stores before lending both static and dynamic LDS to the suffix. The same
 * prepared owner retains original outputs, arithmetic and runtime publication.
 */
#pragma once
#include "ROCmFullNativeHCFFN.h"
#include "ROCmResidentGridBarrier.cuh"
#include "ROCmFullNativeHCFFNPrefixBody.hip.inc"
#include "ROCmFullNativeHCFFNSuffixBody.hip.inc"

namespace llaminar2::rocm {
/** @brief Phase-exclusive static LDS, reused only after the HC mixed-ready join. */
union FullNativeHCFFNShared {
    FullNativeHCFFNPrefixScratch hc; ///< Live activated/Q8 prefix banks, retired before suffix reuse.
    NativeLayoutFFNPhaseScratch ffn; ///< Original selector, acquired bank token and down metadata.
};
static_assert(sizeof(FullNativeHCFFNShared)==
    std::max(sizeof(FullNativeHCFFNPrefixScratch),sizeof(NativeLayoutFFNPhaseScratch)));
/**
 * @brief Complete original participant-local write/HC/router/shared/routed FFN.
 * @tparam Mask Complete original physical decoder family, resolved before capture.
 * @param suffix FIRST by-value kernarg, preserving canonical phase field offsets.
 * @param prefix SECOND by-value kernarg, with independently retained HC bindings.
 * @param prefix_bytes THIRD immutable cold-derived prefix extent; no tensor state is mirrored.
 * @details Every admitted CTA reaches all three prefix and two suffix joins.
 * Terminal down publication is ordered by native completion; the explicit TP
 * collective remains outside this compute entry. No row replay or host decision
 * is introduced. Actual residency is admitted against this full function.
 */
template<std::uint32_t Mask>
__global__ __launch_bounds__(1024,1) void fullNativeHCFFNKernel(
    NativeLayoutRouterFFNArgs suffix,FullNativeHCFFNPrefixArgs prefix,unsigned prefix_bytes)
{
    __shared__ FullNativeHCFFNShared shared;
    extern __shared__ __align__(16) unsigned char dynamic_storage[];
    ResidentGridBarrierFor<FullNativeHCFFNBarrierLayout::prefix_phases> prefix_grid{prefix.barrier_state};
    // Cold admission bounds every positive grid and captures that exact byte
    // extent. Only live ownership and data-dependent work execute on the device.
    fullNativeHCFFNPrefix(prefix,prefix_grid,shared.hc,dynamic_storage,prefix_bytes);
    // Prefix publication and local reader retirement have completed. Both LDS
    // regions are now phase-exclusive; no additive per-phase storage is needed.
    ResidentGridBarrier suffix_grid{suffix.barrier_state};
    tunedFullNativeHCFFNSuffix<Mask>(suffix,suffix_grid,shared.ffn,dynamic_storage,prefix_bytes,prefix.output_demand);
}
}
