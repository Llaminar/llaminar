/**
 * @file ROCmFastFullNativeHCFFNShard2.hip
 * @brief Shard 2 of fast full native HC/FFN megakernel: all IQ-grid codebooks (IQ1/IQ2/IQ3).
 */
#include "ROCmFastFullNativeHCFFNBody.hip.inc"

namespace llaminar2::rocm {

const void* fastFullNativeHCFFNShard2Function(const ResidentMoEFFNExecutionMasks& execution) noexcept
{
    using namespace fast_full_hc_ffn;
    std::uint8_t g = 0u, d = 0u, s = 0u;
    bool exp = false;
    if (decodeFastKey(execution, g, d, s, exp)) {
        if (exp) {
            if (g == 16u && d == 16u && s == 19u)
                return reinterpret_cast<const void*>(&fastFullNativeHCFFNKernel<16u, 16u, 19u, true>);
        } else {
#define LLAMINAR_FAST_IQ(G, D, S) \
    if (g == (G) && d == (D) && s == (S)) \
        return reinterpret_cast<const void*>(&fastFullNativeHCFFNKernel<(G), (D), (S), false>);

            // Homogeneous across all 7 IQ-grid codebooks (both Shared=Q8_0 and Shared=CB).
            LLAMINAR_FAST_IQ(11u, 11u, 19u) LLAMINAR_FAST_IQ(11u, 11u, 11u)
            LLAMINAR_FAST_IQ(12u, 12u, 19u) LLAMINAR_FAST_IQ(12u, 12u, 12u)
            LLAMINAR_FAST_IQ(13u, 13u, 19u) LLAMINAR_FAST_IQ(13u, 13u, 13u)
            LLAMINAR_FAST_IQ(14u, 14u, 19u) LLAMINAR_FAST_IQ(14u, 14u, 14u)
            LLAMINAR_FAST_IQ(15u, 15u, 19u) LLAMINAR_FAST_IQ(15u, 15u, 15u)
            LLAMINAR_FAST_IQ(16u, 16u, 19u) LLAMINAR_FAST_IQ(16u, 16u, 16u)
            LLAMINAR_FAST_IQ(17u, 17u, 19u) LLAMINAR_FAST_IQ(17u, 17u, 17u)

            // Common mixed IQ GGUF quantizations.
            LLAMINAR_FAST_IQ(11u, 5u, 19u)
            LLAMINAR_FAST_IQ(11u, 8u, 19u)
            LLAMINAR_FAST_IQ(13u, 5u, 19u)
            LLAMINAR_FAST_IQ(13u, 8u, 19u)

#undef LLAMINAR_FAST_IQ
        }
    }

    // Complete runtime-dispatched all-18-codebook general entry covering every arbitrary
    // (Gate, Up, Down, Shared) GGUF quantization mix and dynamic multi-codebook union.
    if (execution.routed_gate != 0u && execution.routed_down != 0u &&
        execution.shared_gate != 0u && execution.shared_down != 0u) {
        return reinterpret_cast<const void*>(
            &fastFullNativeHCFFNKernel<kFastDynamicCodebook, kFastDynamicCodebook, kFastDynamicCodebook, true>);
    }
    return nullptr;
}

const void* fastMappedFFNSuffixShard2Function(const ResidentMoEFFNExecutionMasks& execution) noexcept
{
    using namespace fast_full_hc_ffn;
    std::uint8_t g = 0u, d = 0u, s = 0u;
    bool exp = false;
    if (decodeFastKey(execution, g, d, s, exp)) {
        if (exp) {
            if (g == 16u && d == 16u && s == 19u)
                return reinterpret_cast<const void*>(&fastMappedFFNSuffixKernel<16u, 16u, 19u, true>);
        } else {
#define LLAMINAR_FAST_MAPPED_IQ(G, D, S) \
    if (g == (G) && d == (D) && s == (S)) \
        return reinterpret_cast<const void*>(&fastMappedFFNSuffixKernel<(G), (D), (S), false>);

            LLAMINAR_FAST_MAPPED_IQ(11u, 11u, 19u) LLAMINAR_FAST_MAPPED_IQ(11u, 11u, 11u)
            LLAMINAR_FAST_MAPPED_IQ(12u, 12u, 19u) LLAMINAR_FAST_MAPPED_IQ(12u, 12u, 12u)
            LLAMINAR_FAST_MAPPED_IQ(13u, 13u, 19u) LLAMINAR_FAST_MAPPED_IQ(13u, 13u, 13u)
            LLAMINAR_FAST_MAPPED_IQ(14u, 14u, 19u) LLAMINAR_FAST_MAPPED_IQ(14u, 14u, 14u)
            LLAMINAR_FAST_MAPPED_IQ(15u, 15u, 19u) LLAMINAR_FAST_MAPPED_IQ(15u, 15u, 15u)
            LLAMINAR_FAST_MAPPED_IQ(16u, 16u, 19u) LLAMINAR_FAST_MAPPED_IQ(16u, 16u, 16u)
            LLAMINAR_FAST_MAPPED_IQ(17u, 17u, 19u) LLAMINAR_FAST_MAPPED_IQ(17u, 17u, 17u)

            LLAMINAR_FAST_MAPPED_IQ(11u, 5u, 19u)
            LLAMINAR_FAST_MAPPED_IQ(11u, 8u, 19u)
            LLAMINAR_FAST_MAPPED_IQ(13u, 5u, 19u)
            LLAMINAR_FAST_MAPPED_IQ(13u, 8u, 19u)

#undef LLAMINAR_FAST_MAPPED_IQ
        }
    }

    if (execution.routed_gate != 0u && execution.routed_down != 0u &&
        execution.shared_gate != 0u && execution.shared_down != 0u) {
        return reinterpret_cast<const void*>(
            &fastMappedFFNSuffixKernel<kFastDynamicCodebook, kFastDynamicCodebook, kFastDynamicCodebook, true>);
    }
    return nullptr;
}

} // namespace llaminar2::rocm
