/**
 * @file KVCacheLogicalBlockCodec.h
 * @brief Canonical host serialization helpers for logical KV cache blocks.
 *
 * Logical KV payloads cross ownership boundaries: live cache to prefix cache,
 * live cache to diagnostics, and prefix payload back to live cache.  Those
 * boundaries need a stable byte representation so graph shape, backend, or
 * RoPE implementation details do not leak incidental floating-point encodings
 * into prefix-cache keys and parity probes.
 */

#pragma once

#include "execution/config/RuntimeConfig.h"

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace llaminar2::kv_cache_codec
{
    /**
     * @brief Canonicalize floating zero encodings in a host logical KV payload.
     *
     * IEEE floating types have both positive and negative zero.  RoPE-heavy K
     * rows can legitimately produce either sign while remaining numerically
     * identical, but prefix-cache payloads and state probes compare byte hashes.
     * This helper rewrites all signed zero encodings to positive zero while
     * preserving every non-zero, infinity, and NaN payload bit exactly.
     *
     * Quantized formats are already integer payloads and are left untouched.
     * Malformed byte sizes for floating formats return false so callers can
     * fail loudly instead of hashing a partially canonicalized payload.
     *
     * @param payload Host payload bytes to canonicalize in place.
     * @param bytes Number of bytes in @p payload.
     * @param precision Native KV cache precision represented by @p payload.
     * @return True when canonicalization completed or was not needed.
     */
    inline bool canonicalizeFloatingZeros(void *payload, size_t bytes, ActivationPrecision precision)
    {
        if (!payload || bytes == 0)
        {
            return true;
        }

        auto *raw = static_cast<unsigned char *>(payload);
        switch (precision)
        {
        case ActivationPrecision::FP16:
        case ActivationPrecision::BF16:
        {
            if ((bytes % sizeof(uint16_t)) != 0)
            {
                return false;
            }
            for (size_t offset = 0; offset < bytes; offset += sizeof(uint16_t))
            {
                uint16_t word = 0;
                std::memcpy(&word, raw + offset, sizeof(word));
                if ((word & 0x7fffu) == 0u)
                {
                    word = 0;
                    std::memcpy(raw + offset, &word, sizeof(word));
                }
            }
            return true;
        }
        case ActivationPrecision::FP32:
        {
            if ((bytes % sizeof(uint32_t)) != 0)
            {
                return false;
            }
            for (size_t offset = 0; offset < bytes; offset += sizeof(uint32_t))
            {
                uint32_t word = 0;
                std::memcpy(&word, raw + offset, sizeof(word));
                if ((word & 0x7fffffffu) == 0u)
                {
                    word = 0;
                    std::memcpy(raw + offset, &word, sizeof(word));
                }
            }
            return true;
        }
        default:
            return true;
        }
    }
} // namespace llaminar2::kv_cache_codec
