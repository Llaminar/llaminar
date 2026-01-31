/**
 * @file Test__Q6_K_DecodeVectorization.cpp
 * @brief Unit tests for Q6_K → Q8_0 SIMD decode correctness
 * @author David Sanftenberg
 *
 * Validates Q6_K sub-block decode across scalar/AVX2/AVX-512 implementations.
 */

#include <gtest/gtest.h>
#include "tensors/Tensors.h"
#include "tensors/SIMDHelpers.h"
#include "utils/CPUFeatures.h"
#include <vector>
#include <cmath>
#include <random>

using namespace llaminar2;
using namespace llaminar2::simd;

/**
 * @brief Test fixture for Q6_K decode vectorization tests
 */
class Q6_K_DecodeVectorizationTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Create a representative Q6_K super-block (256 elements)
        // Layout: ql[128], qh[64], scales[16], d (FP16)

        // Set super-block scale (FP16)
        q6k_block_.d = fp32_to_fp16(0.5f);

        // Initialize scales (16 int8 values)
        // Scales pattern: -8, -6, -4, -2, 2, 4, 6, 8, -7, -5, -3, -1, 1, 3, 5, 7
        for (size_t i = 0; i < 16; ++i)
        {
            q6k_block_.scales[i] = static_cast<int8_t>((i < 8) ? (-8 + i * 2) : (-7 + (i - 8) * 2));
        }

        // Initialize ql (low 4 bits of 6-bit values) - 128 bytes
        // Each byte contains 2×4-bit values (low nibble, high nibble)
        for (size_t i = 0; i < 128; ++i)
        {
            q6k_block_.ql[i] = static_cast<uint8_t>((i % 16) | ((15 - i % 16) << 4));
        }

        // Initialize qh (high 2 bits of 6-bit values) - 64 bytes
        // Each byte contains 4×2-bit values (bits 0-1, 2-3, 4-5, 6-7)
        for (size_t i = 0; i < 64; ++i)
        {
            q6k_block_.qh[i] = static_cast<uint8_t>((i % 4) | ((i % 4) << 2) | ((i % 4) << 4) | ((i % 4) << 6));
        }
    }

    Q6_KBlock q6k_block_;
};

// ========================================================================
// Scalar vs AVX2 Correctness Tests
// ========================================================================

#if defined(__AVX2__)
TEST_F(Q6_K_DecodeVectorizationTest, ScalarVsAVX2_SubBlock0)
{
    alignas(32) int8_t q8_scalar[32];
    alignas(32) int8_t q8_avx2[32];
    uint16_t scale_scalar, scale_avx2;

    decode_q6_k_to_q8_0_scalar(q6k_block_, 0, q8_scalar, &scale_scalar);
    decode_q6_k_to_q8_0_avx2(q6k_block_, 0, q8_avx2, &scale_avx2);

    // Compare scales
    EXPECT_EQ(scale_scalar, scale_avx2) << "Scales should match";

    // Compare quantized values
    for (size_t i = 0; i < 32; ++i)
    {
        EXPECT_EQ(q8_scalar[i], q8_avx2[i])
            << "Mismatch at index " << i << " in sub-block 0";
    }
}

TEST_F(Q6_K_DecodeVectorizationTest, ScalarVsAVX2_AllSubBlocks)
{
    // Test all 8 sub-blocks (0-7)
    for (size_t sub_idx = 0; sub_idx < 8; ++sub_idx)
    {
        alignas(32) int8_t q8_scalar[32];
        alignas(32) int8_t q8_avx2[32];
        uint16_t scale_scalar, scale_avx2;

        decode_q6_k_to_q8_0_scalar(q6k_block_, sub_idx, q8_scalar, &scale_scalar);
        decode_q6_k_to_q8_0_avx2(q6k_block_, sub_idx, q8_avx2, &scale_avx2);

        EXPECT_EQ(scale_scalar, scale_avx2)
            << "Scale mismatch in sub-block " << sub_idx;

        for (size_t i = 0; i < 32; ++i)
        {
            EXPECT_EQ(q8_scalar[i], q8_avx2[i])
                << "Value mismatch at index " << i << " in sub-block " << sub_idx;
        }
    }
}
#endif

// ========================================================================
// Scalar vs AVX-512 Correctness Tests
// ========================================================================

#if defined(__AVX512F__) && defined(__AVX512BW__)
TEST_F(Q6_K_DecodeVectorizationTest, ScalarVsAVX512_SubBlock0)
{
    alignas(64) int8_t q8_scalar[32];
    alignas(64) int8_t q8_avx512[32];
    uint16_t scale_scalar, scale_avx512;

    decode_q6_k_to_q8_0_scalar(q6k_block_, 0, q8_scalar, &scale_scalar);
    decode_q6_k_to_q8_0_avx512(q6k_block_, 0, q8_avx512, &scale_avx512);

    // Compare scales
    EXPECT_EQ(scale_scalar, scale_avx512) << "Scales should match";

    // Compare quantized values
    for (size_t i = 0; i < 32; ++i)
    {
        EXPECT_EQ(q8_scalar[i], q8_avx512[i])
            << "Mismatch at index " << i << " in sub-block 0";
    }
}

TEST_F(Q6_K_DecodeVectorizationTest, ScalarVsAVX512_AllSubBlocks)
{
    // Test all 8 sub-blocks (0-7)
    for (size_t sub_idx = 0; sub_idx < 8; ++sub_idx)
    {
        alignas(64) int8_t q8_scalar[32];
        alignas(64) int8_t q8_avx512[32];
        uint16_t scale_scalar, scale_avx512;

        decode_q6_k_to_q8_0_scalar(q6k_block_, sub_idx, q8_scalar, &scale_scalar);
        decode_q6_k_to_q8_0_avx512(q6k_block_, sub_idx, q8_avx512, &scale_avx512);

        EXPECT_EQ(scale_scalar, scale_avx512)
            << "Scale mismatch in sub-block " << sub_idx;

        for (size_t i = 0; i < 32; ++i)
        {
            EXPECT_EQ(q8_scalar[i], q8_avx512[i])
                << "Value mismatch at index " << i << " in sub-block " << sub_idx;
        }
    }
}
#endif

// ========================================================================
// Auto-Dispatch Correctness Tests
// ========================================================================

TEST_F(Q6_K_DecodeVectorizationTest, AutoDispatch_MatchesScalar_SubBlock0)
{
    alignas(64) int8_t q8_scalar[32];
    alignas(64) int8_t q8_auto[32];
    uint16_t scale_scalar, scale_auto;

    decode_q6_k_to_q8_0_scalar(q6k_block_, 0, q8_scalar, &scale_scalar);
    decode_q6_k_to_q8_0(q6k_block_, 0, q8_auto, &scale_auto);

    EXPECT_EQ(scale_scalar, scale_auto) << "Auto-dispatch scale should match scalar";

    for (size_t i = 0; i < 32; ++i)
    {
        EXPECT_EQ(q8_scalar[i], q8_auto[i])
            << "Auto-dispatch value mismatch at index " << i;
    }
}

TEST_F(Q6_K_DecodeVectorizationTest, AutoDispatch_AllSubBlocks)
{
    for (size_t sub_idx = 0; sub_idx < 8; ++sub_idx)
    {
        alignas(64) int8_t q8_scalar[32];
        alignas(64) int8_t q8_auto[32];
        uint16_t scale_scalar, scale_auto;

        decode_q6_k_to_q8_0_scalar(q6k_block_, sub_idx, q8_scalar, &scale_scalar);
        decode_q6_k_to_q8_0(q6k_block_, sub_idx, q8_auto, &scale_auto);

        EXPECT_EQ(scale_scalar, scale_auto)
            << "Auto-dispatch scale mismatch in sub-block " << sub_idx;

        for (size_t i = 0; i < 32; ++i)
        {
            EXPECT_EQ(q8_scalar[i], q8_auto[i])
                << "Auto-dispatch value mismatch at index " << i << " in sub-block " << sub_idx;
        }
    }
}

// ========================================================================
// Edge Case Tests
// ========================================================================

TEST_F(Q6_K_DecodeVectorizationTest, ZeroScale_AllSubBlocks)
{
    // Test with zero super-block scale
    Q6_KBlock zero_block = q6k_block_;
    zero_block.d = fp32_to_fp16(0.0f);

    for (size_t sub_idx = 0; sub_idx < 8; ++sub_idx)
    {
        alignas(64) int8_t q8_out[32];
        uint16_t scale_out;

        decode_q6_k_to_q8_0_scalar(zero_block, sub_idx, q8_out, &scale_out);

        // Should produce zero scale and zero values
        float scale_fp32 = fp16_to_fp32(scale_out);
        EXPECT_FLOAT_EQ(scale_fp32, 0.0f) << "Sub-block " << sub_idx << " should have zero scale";
    }
}

TEST_F(Q6_K_DecodeVectorizationTest, LargeScale_NoOverflow)
{
    // Test with large super-block scale (near FP16 max)
    Q6_KBlock large_block = q6k_block_;
    large_block.d = fp32_to_fp16(65000.0f);

    for (size_t sub_idx = 0; sub_idx < 8; ++sub_idx)
    {
        alignas(64) int8_t q8_out[32];
        uint16_t scale_out;

        EXPECT_NO_THROW({
            decode_q6_k_to_q8_0_scalar(large_block, sub_idx, q8_out, &scale_out);
        }) << "Large scale should not cause overflow in sub-block "
           << sub_idx;

        // Verify scale is finite
        float scale_fp32 = fp16_to_fp32(scale_out);
        EXPECT_TRUE(std::isfinite(scale_fp32))
            << "Output scale should be finite in sub-block " << sub_idx;
    }
}

TEST_F(Q6_K_DecodeVectorizationTest, UniformValues_Correctness)
{
    // Create block with uniform quantized values
    Q6_KBlock uniform_block = {};
    uniform_block.d = fp32_to_fp16(1.0f);

    // Set all ql bytes to 0x77 (both nibbles = 7)
    std::memset(uniform_block.ql, 0x77, 128);

    // Set all qh bytes to 0x55 (all 2-bit pairs = 1)
    std::memset(uniform_block.qh, 0x55, 64);

    // Set all scales to 4
    for (size_t i = 0; i < 16; ++i)
    {
        uniform_block.scales[i] = 4;
    }

    // Test sub-block 0
    alignas(64) int8_t q8_scalar[32];
    alignas(64) int8_t q8_auto[32];
    uint16_t scale_scalar, scale_auto;

    decode_q6_k_to_q8_0_scalar(uniform_block, 0, q8_scalar, &scale_scalar);
    decode_q6_k_to_q8_0(uniform_block, 0, q8_auto, &scale_auto);

    // Verify auto-dispatch matches scalar
    EXPECT_EQ(scale_scalar, scale_auto);
    for (size_t i = 0; i < 32; ++i)
    {
        EXPECT_EQ(q8_scalar[i], q8_auto[i])
            << "Uniform values mismatch at index " << i;
    }
}

// ========================================================================
// Randomized Fuzz Testing
// ========================================================================

TEST_F(Q6_K_DecodeVectorizationTest, RandomizedFuzz_100Blocks)
{
    // Test 100 random Q6_K blocks across all sub-blocks
    for (size_t iter = 0; iter < 100; ++iter)
    {
        Q6_KBlock rand_block;

        // Random super-block scale (0.01 to 100.0)
        float rand_scale = 0.01f + static_cast<float>(rand() % 10000) / 100.0f;
        rand_block.d = fp32_to_fp16(rand_scale);

        // Random scales (-127 to 127)
        for (size_t i = 0; i < 16; ++i)
        {
            rand_block.scales[i] = static_cast<int8_t>((rand() % 255) - 127);
        }

        // Random ql and qh
        for (size_t i = 0; i < 128; ++i)
        {
            rand_block.ql[i] = static_cast<uint8_t>(rand() % 256);
        }
        for (size_t i = 0; i < 64; ++i)
        {
            rand_block.qh[i] = static_cast<uint8_t>(rand() % 256);
        }

        // Test all 8 sub-blocks
        for (size_t sub_idx = 0; sub_idx < 8; ++sub_idx)
        {
            alignas(64) int8_t q8_scalar[32];
            alignas(64) int8_t q8_auto[32];
            uint16_t scale_scalar, scale_auto;

            decode_q6_k_to_q8_0_scalar(rand_block, sub_idx, q8_scalar, &scale_scalar);
            decode_q6_k_to_q8_0(rand_block, sub_idx, q8_auto, &scale_auto);

            EXPECT_EQ(scale_scalar, scale_auto)
                << "Fuzz test " << iter << ", sub-block " << sub_idx << " scale mismatch";

            for (size_t i = 0; i < 32; ++i)
            {
                EXPECT_EQ(q8_scalar[i], q8_auto[i])
                    << "Fuzz test " << iter << ", sub-block " << sub_idx
                    << ", value mismatch at index " << i;
            }
        }
    }
}

// ========================================================================
// Performance Characterization Tests
// ========================================================================

TEST_F(Q6_K_DecodeVectorizationTest, ScalarPerformance_1000Iterations)
{
    alignas(64) int8_t q8_out[32];
    uint16_t scale_out;

    auto start = std::chrono::high_resolution_clock::now();

    for (size_t iter = 0; iter < 1000; ++iter)
    {
        for (size_t sub_idx = 0; sub_idx < 8; ++sub_idx)
        {
            decode_q6_k_to_q8_0_scalar(q6k_block_, sub_idx, q8_out, &scale_out);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "Scalar: 1000 iterations × 8 sub-blocks = "
              << duration.count() << " µs ("
              << (duration.count() / 8000.0) << " µs/decode)"
              << std::endl;
}

#if defined(__AVX2__)
TEST_F(Q6_K_DecodeVectorizationTest, AVX2_Performance_1000Iterations)
{
    alignas(32) int8_t q8_out[32];
    uint16_t scale_out;

    auto start = std::chrono::high_resolution_clock::now();

    for (size_t iter = 0; iter < 1000; ++iter)
    {
        for (size_t sub_idx = 0; sub_idx < 8; ++sub_idx)
        {
            decode_q6_k_to_q8_0_avx2(q6k_block_, sub_idx, q8_out, &scale_out);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "AVX2: 1000 iterations × 8 sub-blocks = "
              << duration.count() << " µs ("
              << (duration.count() / 8000.0) << " µs/decode)"
              << std::endl;
}
#endif

#if defined(__AVX512F__) && defined(__AVX512BW__)
TEST_F(Q6_K_DecodeVectorizationTest, AVX512_Performance_1000Iterations)
{
    alignas(64) int8_t q8_out[32];
    uint16_t scale_out;

    auto start = std::chrono::high_resolution_clock::now();

    for (size_t iter = 0; iter < 1000; ++iter)
    {
        for (size_t sub_idx = 0; sub_idx < 8; ++sub_idx)
        {
            decode_q6_k_to_q8_0_avx512(q6k_block_, sub_idx, q8_out, &scale_out);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "AVX-512: 1000 iterations × 8 sub-blocks = "
              << duration.count() << " µs ("
              << (duration.count() / 8000.0) << " µs/decode)"
              << std::endl;
}
#endif
