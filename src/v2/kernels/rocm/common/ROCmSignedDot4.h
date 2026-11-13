/**
 * @file ROCmSignedDot4.h
 * @brief Native signed-byte dot products with identical modulo-32-bit arithmetic.
 *
 * The compiler selects signed INT8 dot4 directly when available, or encodes it
 * through two explicit signedness operands. Both forms execute one native dot4 with
 * saturation disabled. Packing, accumulator bits and every caller's reduction
 * order remain unchanged; neither architecture executes a scalar substitute.
 */
#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>

namespace llaminar2::rocm_integer_contract
{
    /**
     * @brief Add four signed-byte products to the accumulator modulo 2^32.
     * @param a Four packed two's-complement signed INT8 operands.
     * @param b Four packed two's-complement signed INT8 operands.
     * @param accumulator Initial 32-bit accumulator bit pattern.
     * @return Unsaturated native sum, retaining the original accumulator bits.
     */
    __device__ __forceinline__ std::int32_t signedDot4(
        std::int32_t a, std::int32_t b, std::int32_t accumulator)
    {
#if defined(__HIP_DEVICE_COMPILE__)
        static_assert(__builtin_amdgcn_is_invocable(__builtin_amdgcn_sdot4) ||
                      __builtin_amdgcn_is_invocable(__builtin_amdgcn_sudot4),
                      "ROCm signed INT8 requires a native unsaturated dot4 instruction");
#endif
        if constexpr (__builtin_amdgcn_is_invocable(__builtin_amdgcn_sdot4))
            return __builtin_amdgcn_sdot4(a, b, accumulator, false);
        else
            return __builtin_amdgcn_sudot4(true, a, true, b, accumulator, false);
    }
}
