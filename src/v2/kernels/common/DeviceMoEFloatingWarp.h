/**
 * @file DeviceMoEFloatingWarp.h
 * @brief Register-owned mapping of the canonical floating expert reduction.
 *
 * A native warp/wave owns one output column and each lane holds its logical
 * K partitions. This is the existing 256-partition arithmetic, not a new dot
 * product: partition p still visits p, p+256, p+512, ... in that order. The
 * upper levels of the fixed tree fold registers; the remaining levels
 * exchange registers within the subgroup. No shared scratch or block barrier
 * is necessary. CUDA uses eight partitions per lane; supported HIP wave64
 * targets use four so all lanes issue contiguous weight transactions.
 */
#pragma once

#include "kernels/common/DeviceFP32NumericalContract.h"
#include "kernels/common/MoEProjectionNumericalContract.h"

namespace llaminar2::device_moe_floating_warp
{
#if defined(__HIPCC__)
    constexpr int width = 64;
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__GFX9__)
#error "Floating expert launch mapping requires the supported HIP wave64 ISA"
#endif
#else
    constexpr int width = 32;
#endif
    constexpr int threads = 256;
    constexpr int columns_per_block = threads / width;
    constexpr int partitions = MoEProjectionNumericalContract::floating_ordered_k_partitions;
    constexpr int partitions_per_lane = partitions / width;
    static_assert(partitions == 256 && partitions % width == 0);

    /**
     * @brief Fold register partitions and then the native warp/wave subtree.
     * @param partials Ordered FP32 partials for logical lanes lane + width*p.
     * @return The complete canonical sum in subgroup lane zero only.
     *
     * All subgroup lanes participate. A caller may retire whole subgroups for column
     * tails, but not individual lanes. Explicit rounded adds preserve the
     * serial oracle's 128,64,32,16,8,4,2,1 reduction parenthesization.
     */
    __device__ __forceinline__ float reduce(float (&partials)[partitions_per_lane])
    {
#pragma unroll
        for (int stride = partitions_per_lane / 2; stride > 0; stride >>= 1)
        {
#pragma unroll
            for (int index = 0; index < stride; ++index)
                partials[index] = device_fp32_contract::add(partials[index], partials[index + stride]);
        }
        float sum = partials[0];
#pragma unroll
        for (int stride = width / 2; stride > 0; stride >>= 1)
        {
#if defined(__HIPCC__)
            const float peer = __shfl_down(sum, stride, width);
#else
            const float peer = __shfl_down_sync(0xffffffffu, sum, stride, width);
#endif
            sum = device_fp32_contract::add(sum, peer);
        }
        return sum;
    }
}
