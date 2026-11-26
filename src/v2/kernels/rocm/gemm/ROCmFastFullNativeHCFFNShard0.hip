/**
 * @file ROCmFastFullNativeHCFFNShard0.hip
 * @brief Shard 0 of fast full native HC/FFN megakernel: primary Q4_K/Q5_K/Q6_K/Q8_0 pairs.
 */
#include "ROCmFastFullNativeHCFFNBody.hip.inc"

namespace llaminar2::rocm {

const void* fastFullNativeHCFFNShard0Function(const ResidentMoEFFNExecutionMasks& execution) noexcept
{
    using namespace fast_full_hc_ffn;
    std::uint8_t g = 0u, d = 0u, s = 0u;
    bool exp = false;
    if (!decodeFastKey(execution, g, d, s, exp) || s != 19u)
        return nullptr;

#define LLAMINAR_FAST_PAIR(G, D, EXP) \
    if (g == (G) && d == (D) && exp == (EXP)) \
        return reinterpret_cast<const void*>(&fastFullNativeHCFFNKernel<(G), (D), 19u, (EXP)>);

    // Production mixed and homogeneous Q4_1(Q4_K), Q5_1(Q5_K), Q6_K, Q8_0 pairs (Shared = Q8_0).
    LLAMINAR_FAST_PAIR(5u, 7u, false)
    LLAMINAR_FAST_PAIR(5u, 7u, true)
    LLAMINAR_FAST_PAIR(5u, 19u, false)
    LLAMINAR_FAST_PAIR(5u, 19u, true)
    LLAMINAR_FAST_PAIR(7u, 19u, false)
    LLAMINAR_FAST_PAIR(7u, 19u, true)
    LLAMINAR_FAST_PAIR(5u, 8u, false)
    LLAMINAR_FAST_PAIR(5u, 8u, true)
    LLAMINAR_FAST_PAIR(7u, 8u, false)
    LLAMINAR_FAST_PAIR(7u, 8u, true)
    LLAMINAR_FAST_PAIR(8u, 19u, false)
    LLAMINAR_FAST_PAIR(5u, 5u, false)
    LLAMINAR_FAST_PAIR(5u, 5u, true)
    LLAMINAR_FAST_PAIR(7u, 7u, false)
    LLAMINAR_FAST_PAIR(7u, 7u, true)

#undef LLAMINAR_FAST_PAIR
    return nullptr;
}

const void* fastMappedFFNSuffixShard0Function(const ResidentMoEFFNExecutionMasks& execution) noexcept
{
    using namespace fast_full_hc_ffn;
    std::uint8_t g = 0u, d = 0u, s = 0u;
    bool exp = false;
    if (!decodeFastKey(execution, g, d, s, exp) || s != 19u)
        return nullptr;

#define LLAMINAR_FAST_MAPPED_PAIR(G, D, EXP) \
    if (g == (G) && d == (D) && exp == (EXP)) \
        return reinterpret_cast<const void*>(&fastMappedFFNSuffixKernel<(G), (D), 19u, (EXP)>);

    LLAMINAR_FAST_MAPPED_PAIR(5u, 7u, false)
    LLAMINAR_FAST_MAPPED_PAIR(5u, 7u, true)
    LLAMINAR_FAST_MAPPED_PAIR(5u, 19u, false)
    LLAMINAR_FAST_MAPPED_PAIR(5u, 19u, true)
    LLAMINAR_FAST_MAPPED_PAIR(7u, 19u, false)
    LLAMINAR_FAST_MAPPED_PAIR(7u, 19u, true)
    LLAMINAR_FAST_MAPPED_PAIR(5u, 8u, false)
    LLAMINAR_FAST_MAPPED_PAIR(5u, 8u, true)
    LLAMINAR_FAST_MAPPED_PAIR(7u, 8u, false)
    LLAMINAR_FAST_MAPPED_PAIR(7u, 8u, true)
    LLAMINAR_FAST_MAPPED_PAIR(8u, 19u, false)
    LLAMINAR_FAST_MAPPED_PAIR(5u, 5u, false)
    LLAMINAR_FAST_MAPPED_PAIR(5u, 5u, true)
    LLAMINAR_FAST_MAPPED_PAIR(7u, 7u, false)
    LLAMINAR_FAST_MAPPED_PAIR(7u, 7u, true)

#undef LLAMINAR_FAST_MAPPED_PAIR
    return nullptr;
}

} // namespace llaminar2::rocm
