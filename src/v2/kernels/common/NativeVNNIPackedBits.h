/**
 * @file NativeVNNIPackedBits.h
 * @brief Exact integer bit-plane primitives for compact NativeVNNI payloads.
 *
 * These operations change neither stored formats nor numerical reductions.
 * Host/device compilation lets device-free tests exhaust the complete input
 * domain while production kernels inline the same integer expression.
 */
#pragma once

#include <cstdint>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define LLAMINAR_PACKED_BITS_INLINE __host__ __device__ __forceinline__
#else
#define LLAMINAR_PACKED_BITS_INLINE inline
#endif

namespace llaminar2::native_vnni
{
    /**
     * @brief Deposit four Q5 high bits into bit four of four packed bytes.
     * @param nibble Low four bits, one high bit for each output byte.
     * @return Packed high plane; all other output bits are zero.
     *
     * Multiplication replicates the nibble at bit offsets 4,11,18,25. The
     * four-bit groups cannot overlap, so no carries cross them. Masking keeps
     * source bit i at destination bit 8*i+4, replacing four individual
     * shift/mask/or sequences without a lookup, extra storage, or branches.
     */
    [[nodiscard]] LLAMINAR_PACKED_BITS_INLINE constexpr std::uint32_t
    q5HighBitsToPackedBytes(std::uint32_t nibble) noexcept
    {
        return ((nibble & 0xfu) * 0x02040810u) & 0x10101010u;
    }
}

#undef LLAMINAR_PACKED_BITS_INLINE
