/**
 * @file Test__KVCacheLogicalBlockCodec.cpp
 * @brief Unit tests for canonical logical KV block host serialization.
 *
 * Prefix-cache payloads and runtime probes compare exported KV blocks by byte
 * hash.  These tests pin the access-layer contract that floating zero signs are
 * normalized at export time while non-zero payload bits remain exact.
 */

#include "kernels/kvcache/KVCacheLogicalBlockCodec.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>

namespace llaminar2::test
{
    /**
     * @brief Verify FP16/BF16-style 16-bit zero canonicalization.
     */
    TEST(Test__KVCacheLogicalBlockCodec, CanonicalizesSixteenBitFloatingZerosOnly)
    {
        std::array<uint16_t, 6> words{
            0x0000u, // +0
            0x8000u, // -0
            0x3c00u, // +1
            0xbc00u, // -1
            0x7e01u, // NaN payload
            0xfe01u, // negative NaN payload
        };

        ASSERT_TRUE(kv_cache_codec::canonicalizeFloatingZeros(
            words.data(),
            words.size() * sizeof(uint16_t),
            ActivationPrecision::FP16));

        EXPECT_EQ(words[0], 0x0000u);
        EXPECT_EQ(words[1], 0x0000u);
        EXPECT_EQ(words[2], 0x3c00u);
        EXPECT_EQ(words[3], 0xbc00u);
        EXPECT_EQ(words[4], 0x7e01u);
        EXPECT_EQ(words[5], 0xfe01u);
    }

    /**
     * @brief Verify FP32 zero canonicalization without changing other values.
     */
    TEST(Test__KVCacheLogicalBlockCodec, CanonicalizesThirtyTwoBitFloatingZerosOnly)
    {
        std::array<uint32_t, 6> words{
            0x00000000u, // +0
            0x80000000u, // -0
            0x3f800000u, // +1
            0xbf800000u, // -1
            0x7fc00001u, // NaN payload
            0xffc00001u, // negative NaN payload
        };

        ASSERT_TRUE(kv_cache_codec::canonicalizeFloatingZeros(
            words.data(),
            words.size() * sizeof(uint32_t),
            ActivationPrecision::FP32));

        EXPECT_EQ(words[0], 0x00000000u);
        EXPECT_EQ(words[1], 0x00000000u);
        EXPECT_EQ(words[2], 0x3f800000u);
        EXPECT_EQ(words[3], 0xbf800000u);
        EXPECT_EQ(words[4], 0x7fc00001u);
        EXPECT_EQ(words[5], 0xffc00001u);
    }

    /**
     * @brief Verify quantized payloads and malformed floating sizes are explicit.
     */
    TEST(Test__KVCacheLogicalBlockCodec, LeavesQuantizedPayloadsAndRejectsMalformedFloatingPayloads)
    {
        std::array<uint8_t, 3> quantized{0x80u, 0x00u, 0xffu};
        ASSERT_TRUE(kv_cache_codec::canonicalizeFloatingZeros(
            quantized.data(),
            quantized.size(),
            ActivationPrecision::Q8_1));
        EXPECT_EQ(quantized[0], 0x80u);
        EXPECT_EQ(quantized[1], 0x00u);
        EXPECT_EQ(quantized[2], 0xffu);

        std::array<uint8_t, 3> malformed_fp16{0u, 0u, 0u};
        EXPECT_FALSE(kv_cache_codec::canonicalizeFloatingZeros(
            malformed_fp16.data(),
            malformed_fp16.size(),
            ActivationPrecision::FP16));

        std::array<uint8_t, 6> malformed_fp32{0u, 0u, 0u, 0u, 0u, 0u};
        EXPECT_FALSE(kv_cache_codec::canonicalizeFloatingZeros(
            malformed_fp32.data(),
            malformed_fp32.size(),
            ActivationPrecision::FP32));
    }
} // namespace llaminar2::test
