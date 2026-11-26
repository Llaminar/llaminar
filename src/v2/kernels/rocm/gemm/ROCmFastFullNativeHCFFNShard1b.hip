/**
 * @file ROCmFastFullNativeHCFFNShard1b.hip
 * @brief Shard 1b of fast full native HC/FFN megakernel: Q2_K/Q8_0 & mixed GGUF quantization pairs.
 */
#include "ROCmFastFullNativeHCFFNBody.hip.inc"

namespace llaminar2::rocm {

const void* fastFullNativeHCFFNShard1bFunction(const ResidentMoEFFNExecutionMasks& execution) noexcept
{
    using namespace fast_full_hc_ffn;
    std::uint8_t g = 0u, d = 0u, s = 0u;
    bool exp = false;
    if (!decodeFastKey(execution, g, d, s, exp) || exp)
        return nullptr;

#define LLAMINAR_FAST_TRIPLE(G, D, S) \
    if (g == (G) && d == (D) && s == (S)) \
        return reinterpret_cast<const void*>(&fastFullNativeHCFFNKernel<(G), (D), (S), false>);

    LLAMINAR_FAST_TRIPLE(10u, 10u, 19u) LLAMINAR_FAST_TRIPLE(10u, 10u, 10u)
    LLAMINAR_FAST_TRIPLE(19u, 19u, 19u)
    LLAMINAR_FAST_TRIPLE(9u, 5u, 19u)
    LLAMINAR_FAST_TRIPLE(9u, 7u, 19u)
    LLAMINAR_FAST_TRIPLE(9u, 8u, 19u)
    LLAMINAR_FAST_TRIPLE(10u, 5u, 19u)
    LLAMINAR_FAST_TRIPLE(10u, 8u, 19u)
    LLAMINAR_FAST_TRIPLE(4u, 7u, 19u)
    LLAMINAR_FAST_TRIPLE(4u, 8u, 19u)
    LLAMINAR_FAST_TRIPLE(4u, 19u, 19u)
    LLAMINAR_FAST_TRIPLE(24u, 7u, 19u)
    LLAMINAR_FAST_TRIPLE(24u, 8u, 19u)
    LLAMINAR_FAST_TRIPLE(24u, 19u, 19u)

#undef LLAMINAR_FAST_TRIPLE
    return nullptr;
}

const void* fastMappedFFNSuffixShard1bFunction(const ResidentMoEFFNExecutionMasks& execution) noexcept
{
    using namespace fast_full_hc_ffn;
    std::uint8_t g = 0u, d = 0u, s = 0u;
    bool exp = false;
    if (!decodeFastKey(execution, g, d, s, exp) || exp)
        return nullptr;

#define LLAMINAR_FAST_MAPPED_TRIPLE(G, D, S) \
    if (g == (G) && d == (D) && s == (S)) \
        return reinterpret_cast<const void*>(&fastMappedFFNSuffixKernel<(G), (D), (S), false>);

    LLAMINAR_FAST_MAPPED_TRIPLE(10u, 10u, 19u) LLAMINAR_FAST_MAPPED_TRIPLE(10u, 10u, 10u)
    LLAMINAR_FAST_MAPPED_TRIPLE(19u, 19u, 19u)
    LLAMINAR_FAST_MAPPED_TRIPLE(9u, 5u, 19u)
    LLAMINAR_FAST_MAPPED_TRIPLE(9u, 7u, 19u)
    LLAMINAR_FAST_MAPPED_TRIPLE(9u, 8u, 19u)
    LLAMINAR_FAST_MAPPED_TRIPLE(10u, 5u, 19u)
    LLAMINAR_FAST_MAPPED_TRIPLE(10u, 8u, 19u)
    LLAMINAR_FAST_MAPPED_TRIPLE(4u, 7u, 19u)
    LLAMINAR_FAST_MAPPED_TRIPLE(4u, 8u, 19u)
    LLAMINAR_FAST_MAPPED_TRIPLE(4u, 19u, 19u)
    LLAMINAR_FAST_MAPPED_TRIPLE(24u, 7u, 19u)
    LLAMINAR_FAST_MAPPED_TRIPLE(24u, 8u, 19u)
    LLAMINAR_FAST_MAPPED_TRIPLE(24u, 19u, 19u)

#undef LLAMINAR_FAST_MAPPED_TRIPLE
    return nullptr;
}

} // namespace llaminar2::rocm
