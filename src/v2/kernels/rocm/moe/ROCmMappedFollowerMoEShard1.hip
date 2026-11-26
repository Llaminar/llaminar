/**
 * @file ROCmMappedFollowerMoEShard1.hip
 * @brief Shard 1 of 1-kernel resident ROCm mapped follower megakernel: all IQ1/IQ2/IQ3/IQ4 and mixed pairs.
 */
#include "ROCmMappedFollowerMoEBody.hip.inc"

namespace llaminar2::rocm
{
    const void *fastMappedFollowerMoEShard1Function(
        std::uint32_t gate_mask, std::uint32_t down_mask) noexcept
    {
        using namespace mapped_follower_moe;
        std::uint8_t g = 0u, d = 0u;
        bool exp = false;
        if (decodeFollowerKey(gate_mask, down_mask, g, d, exp))
        {
#define LLAMINAR_FOLLOWER_IQ(G, D, EXP) \
    if (g == (G) && d == (D) && exp == (EXP)) \
        return reinterpret_cast<const void *>(&fastMappedFollowerMoEKernel<(G), (D), (EXP)>);

            // IQ4_NL (4) and IQ4_XS source-scale (24) homogeneous & mixed pairs.
            LLAMINAR_FOLLOWER_IQ(4u, 4u, false)
            LLAMINAR_FOLLOWER_IQ(4u, 4u, true)
            LLAMINAR_FOLLOWER_IQ(4u, 7u, false)
            LLAMINAR_FOLLOWER_IQ(4u, 8u, false)
            LLAMINAR_FOLLOWER_IQ(4u, 19u, false)
            LLAMINAR_FOLLOWER_IQ(24u, 24u, false)
            LLAMINAR_FOLLOWER_IQ(24u, 24u, true)
            LLAMINAR_FOLLOWER_IQ(24u, 7u, false)
            LLAMINAR_FOLLOWER_IQ(24u, 8u, false)
            LLAMINAR_FOLLOWER_IQ(24u, 19u, false)

            // All 7 IQ1/IQ2/IQ3 grid codebooks (homogeneous + common mixed GGUF pairs).
            LLAMINAR_FOLLOWER_IQ(11u, 11u, false)
            LLAMINAR_FOLLOWER_IQ(12u, 12u, false)
            LLAMINAR_FOLLOWER_IQ(13u, 13u, false)
            LLAMINAR_FOLLOWER_IQ(14u, 14u, false)
            LLAMINAR_FOLLOWER_IQ(15u, 15u, false)
            LLAMINAR_FOLLOWER_IQ(16u, 16u, false)
            LLAMINAR_FOLLOWER_IQ(16u, 16u, true)
            LLAMINAR_FOLLOWER_IQ(17u, 17u, false)
            LLAMINAR_FOLLOWER_IQ(11u, 5u, false)
            LLAMINAR_FOLLOWER_IQ(11u, 8u, false)
            LLAMINAR_FOLLOWER_IQ(13u, 5u, false)
            LLAMINAR_FOLLOWER_IQ(13u, 8u, false)

#undef LLAMINAR_FOLLOWER_IQ
        }

        // Complete runtime-dispatched all-18-codebook general follower entry covering every
        // arbitrary (Gate, Up, Down) GGUF quantization mix and multi-codebook dynamic union.
        if (gate_mask != 0u && down_mask != 0u)
            return reinterpret_cast<const void *>(&fastMappedFollowerMoEKernel<kFastDynamicCodebook, kFastDynamicCodebook, true>);
        return nullptr;
    }

} // namespace llaminar2::rocm
