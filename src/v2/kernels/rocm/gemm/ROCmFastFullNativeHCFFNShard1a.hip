/**
 * @file ROCmFastFullNativeHCFFNShard1a.hip
 * @brief Shard 1a of fast full native HC/FFN megakernel: homogeneous standard K-quant & legacy codebooks.
 */
#include "ROCmFastFullNativeHCFFNBody.hip.inc"

namespace llaminar2::rocm {

const void* fastFullNativeHCFFNShard1aFunction(const ResidentMoEFFNExecutionMasks& execution) noexcept
{
    using namespace fast_full_hc_ffn;
    std::uint8_t g = 0u, d = 0u, s = 0u;
    bool exp = false;
    if (!decodeFastKey(execution, g, d, s, exp) || exp)
        return nullptr;

#define LLAMINAR_FAST_TRIPLE(G, D, S) \
    if (g == (G) && d == (D) && s == (S)) \
        return reinterpret_cast<const void*>(&fastFullNativeHCFFNKernel<(G), (D), (S), false>);

    LLAMINAR_FAST_TRIPLE(0u, 0u, 19u)   LLAMINAR_FAST_TRIPLE(0u, 0u, 0u)
    LLAMINAR_FAST_TRIPLE(4u, 4u, 19u)   LLAMINAR_FAST_TRIPLE(4u, 4u, 4u)
    LLAMINAR_FAST_TRIPLE(24u, 24u, 19u) LLAMINAR_FAST_TRIPLE(24u, 24u, 24u)
    LLAMINAR_FAST_TRIPLE(5u, 5u, 5u)
    LLAMINAR_FAST_TRIPLE(6u, 6u, 19u)   LLAMINAR_FAST_TRIPLE(6u, 6u, 6u)
    LLAMINAR_FAST_TRIPLE(7u, 7u, 7u)
    LLAMINAR_FAST_TRIPLE(8u, 8u, 19u)   LLAMINAR_FAST_TRIPLE(8u, 8u, 8u)
    LLAMINAR_FAST_TRIPLE(9u, 9u, 19u)   LLAMINAR_FAST_TRIPLE(9u, 9u, 9u)

#undef LLAMINAR_FAST_TRIPLE
    return nullptr;
}

const void* fastMappedFFNSuffixShard1aFunction(const ResidentMoEFFNExecutionMasks& execution) noexcept
{
    using namespace fast_full_hc_ffn;
    std::uint8_t g = 0u, d = 0u, s = 0u;
    bool exp = false;
    if (!decodeFastKey(execution, g, d, s, exp) || exp)
        return nullptr;

#define LLAMINAR_FAST_MAPPED_TRIPLE(G, D, S) \
    if (g == (G) && d == (D) && s == (S)) \
        return reinterpret_cast<const void*>(&fastMappedFFNSuffixKernel<(G), (D), (S), false>);

    LLAMINAR_FAST_MAPPED_TRIPLE(0u, 0u, 19u)   LLAMINAR_FAST_MAPPED_TRIPLE(0u, 0u, 0u)
    LLAMINAR_FAST_MAPPED_TRIPLE(4u, 4u, 19u)   LLAMINAR_FAST_MAPPED_TRIPLE(4u, 4u, 4u)
    LLAMINAR_FAST_MAPPED_TRIPLE(24u, 24u, 19u) LLAMINAR_FAST_MAPPED_TRIPLE(24u, 24u, 24u)
    LLAMINAR_FAST_MAPPED_TRIPLE(5u, 5u, 5u)
    LLAMINAR_FAST_MAPPED_TRIPLE(6u, 6u, 19u)   LLAMINAR_FAST_MAPPED_TRIPLE(6u, 6u, 6u)
    LLAMINAR_FAST_MAPPED_TRIPLE(7u, 7u, 7u)
    LLAMINAR_FAST_MAPPED_TRIPLE(8u, 8u, 19u)   LLAMINAR_FAST_MAPPED_TRIPLE(8u, 8u, 8u)
    LLAMINAR_FAST_MAPPED_TRIPLE(9u, 9u, 19u)   LLAMINAR_FAST_MAPPED_TRIPLE(9u, 9u, 9u)

#undef LLAMINAR_FAST_MAPPED_TRIPLE
    return nullptr;
}

} // namespace llaminar2::rocm
