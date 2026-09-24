/**
 * @file FNV1a.h
 * @brief Shared 64-bit FNV-1a hashing for byte-exact runtime diagnostics.
 *
 * Runtime parity diagnostics frequently compare payloads emitted by different
 * subsystems.  Those producers must use one canonical offset basis and update
 * rule; otherwise identical bytes can appear different solely because two
 * diagnostic helpers implemented different hash variants.
 */

#pragma once

#include <cstddef>
#include <cstdint>

namespace llaminar2
{
    /**
     * @brief Canonical 64-bit FNV-1a offset basis.
     */
    inline constexpr uint64_t kFNV1a64OffsetBasis =
        14695981039346656037ull;

    /**
     * @brief Canonical 64-bit FNV-1a prime.
     */
    inline constexpr uint64_t kFNV1a64Prime = 1099511628211ull;

    /**
     * @brief Hash a contiguous byte range with canonical 64-bit FNV-1a.
     *
     * @param data Start of the byte range. A null pointer is valid only when
     *             @p byte_count is zero.
     * @param byte_count Number of bytes to hash.
     * @param seed Initial hash state, normally @ref kFNV1a64OffsetBasis.
     * @return Deterministic 64-bit digest of the supplied bytes.
     */
    inline uint64_t fnv1a64(
        const void *data,
        size_t byte_count,
        uint64_t seed = kFNV1a64OffsetBasis)
    {
        if (!data || byte_count == 0)
            return seed;

        uint64_t hash = seed;
        const auto *bytes = static_cast<const unsigned char *>(data);
        for (size_t i = 0; i < byte_count; ++i)
        {
            hash ^= static_cast<uint64_t>(bytes[i]);
            hash *= kFNV1a64Prime;
        }
        return hash;
    }
} // namespace llaminar2
