/**
 * @file Test__Q4_1_DecodeVectorization.cpp
 * @brief Tests for Q4_1 decode vectorization (SIMD optimizations)
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include <random>
#include <chrono>
#include <cstring>
#include "tensors/SIMDHelpers.h"

using namespace llaminar2::simd;

class Q4_1_DecodeVectorizationTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Initialize test data
        std::memset(input_qs, 0, sizeof(input_qs));
        d_fp16 = fp32_to_fp16(1.0f);
        m_fp16 = fp32_to_fp16(0.0f);
    }

    uint8_t input_qs[16]; // 16 bytes = 32 nibbles
    uint16_t d_fp16;
    uint16_t m_fp16;

    int8_t output_scalar[32];
    int8_t output_avx2[32];
    int8_t output_avx512[32];

    uint16_t d_scalar_fp16;
    uint16_t d_avx2_fp16;
    uint16_t d_avx512_fp16;
};

/**
 * @brief Test scalar vs AVX-512 with all possible 4-bit values
 */
TEST_F(Q4_1_DecodeVectorizationTest, ScalarVsAVX512_AllValues)
{
#if defined(__AVX512F__) && defined(__AVX512BW__)
    // Test all 16 possible 4-bit values (0-15)
    for (int i = 0; i < 16; ++i)
    {
        input_qs[i] = static_cast<uint8_t>((i << 4) | i); // Both nibbles same value
    }

    d_fp16 = fp32_to_fp16(0.5f);
    m_fp16 = fp32_to_fp16(-1.0f);

    decode_q4_1_to_q8_0_scalar(input_qs, d_fp16, m_fp16, output_scalar, &d_scalar_fp16);
    decode_q4_1_to_q8_0_avx512(input_qs, d_fp16, m_fp16, output_avx512, &d_avx512_fp16);

    for (int i = 0; i < 32; ++i)
    {
        EXPECT_EQ(output_scalar[i], output_avx512[i])
            << "Mismatch at index " << i
            << ": scalar=" << static_cast<int>(output_scalar[i])
            << ", avx512=" << static_cast<int>(output_avx512[i]);
    }

    EXPECT_EQ(d_scalar_fp16, d_avx512_fp16) << "Scale mismatch";
#else
    GTEST_SKIP() << "AVX-512 not available";
#endif
}

/**
 * @brief Test scalar vs AVX2 with all possible 4-bit values
 */
TEST_F(Q4_1_DecodeVectorizationTest, ScalarVsAVX2_AllValues)
{
#ifdef __AVX2__
    // Test all 16 possible 4-bit values (0-15)
    for (int i = 0; i < 16; ++i)
    {
        input_qs[i] = static_cast<uint8_t>((i << 4) | i);
    }

    d_fp16 = fp32_to_fp16(0.5f);
    m_fp16 = fp32_to_fp16(-1.0f);

    decode_q4_1_to_q8_0_scalar(input_qs, d_fp16, m_fp16, output_scalar, &d_scalar_fp16);
    decode_q4_1_to_q8_0_avx2(input_qs, d_fp16, m_fp16, output_avx2, &d_avx2_fp16);

    for (int i = 0; i < 32; ++i)
    {
        EXPECT_EQ(output_scalar[i], output_avx2[i])
            << "Mismatch at index " << i
            << ": scalar=" << static_cast<int>(output_scalar[i])
            << ", avx2=" << static_cast<int>(output_avx2[i]);
    }

    EXPECT_EQ(d_scalar_fp16, d_avx2_fp16) << "Scale mismatch";
#else
    GTEST_SKIP() << "AVX2 not available";
#endif
}

/**
 * @brief Test auto-dispatch function matches scalar
 */
TEST_F(Q4_1_DecodeVectorizationTest, AutoDispatch_MatchesScalar)
{
    std::mt19937 gen(12345);
    std::uniform_int_distribution<> dist(0, 255);

    for (int i = 0; i < 16; ++i)
    {
        input_qs[i] = static_cast<uint8_t>(dist(gen));
    }

    d_fp16 = fp32_to_fp16(1.5f);
    m_fp16 = fp32_to_fp16(0.25f);

    int8_t output_auto[32];
    uint16_t d_auto_fp16;

    decode_q4_1_to_q8_0_scalar(input_qs, d_fp16, m_fp16, output_scalar, &d_scalar_fp16);
    decode_q4_1_to_q8_0(input_qs, d_fp16, m_fp16, output_auto, &d_auto_fp16);

    for (int i = 0; i < 32; ++i)
    {
        EXPECT_EQ(output_scalar[i], output_auto[i])
            << "Auto-dispatch mismatch at index " << i;
    }

    EXPECT_EQ(d_scalar_fp16, d_auto_fp16) << "Auto-dispatch scale mismatch";
}

/**
 * @brief Test zero input values
 */
TEST_F(Q4_1_DecodeVectorizationTest, ZeroValues)
{
    std::memset(input_qs, 0, sizeof(input_qs));
    d_fp16 = fp32_to_fp16(1.0f);
    m_fp16 = fp32_to_fp16(0.0f);

    int8_t output[32];
    uint16_t d_output_fp16;

    decode_q4_1_to_q8_0(input_qs, d_fp16, m_fp16, output, &d_output_fp16);

    // All values should be near zero (quantized)
    for (int i = 0; i < 32; ++i)
    {
        EXPECT_GE(output[i], -5) << "Zero test: value too negative at " << i;
        EXPECT_LE(output[i], 5) << "Zero test: value too positive at " << i;
    }
}

/**
 * @brief Test uniform positive values (all nibbles = 15)
 */
TEST_F(Q4_1_DecodeVectorizationTest, UniformPositive)
{
    std::memset(input_qs, 0xFF, sizeof(input_qs)); // All nibbles = 15
    d_fp16 = fp32_to_fp16(1.0f);
    m_fp16 = fp32_to_fp16(0.0f);

    decode_q4_1_to_q8_0_scalar(input_qs, d_fp16, m_fp16, output_scalar, &d_scalar_fp16);

#ifdef __AVX2__
    decode_q4_1_to_q8_0_avx2(input_qs, d_fp16, m_fp16, output_avx2, &d_avx2_fp16);

    for (int i = 0; i < 32; ++i)
    {
        EXPECT_EQ(output_scalar[i], output_avx2[i])
            << "Uniform positive AVX2 mismatch at " << i;
    }
#endif

#if defined(__AVX512F__) && defined(__AVX512BW__)
    decode_q4_1_to_q8_0_avx512(input_qs, d_fp16, m_fp16, output_avx512, &d_avx512_fp16);

    for (int i = 0; i < 32; ++i)
    {
        EXPECT_EQ(output_scalar[i], output_avx512[i])
            << "Uniform positive AVX512 mismatch at " << i;
    }
#endif
}

/**
 * @brief Test with non-zero offset (min value)
 */
TEST_F(Q4_1_DecodeVectorizationTest, NonZeroOffset)
{
    std::memset(input_qs, 0x00, sizeof(input_qs)); // All nibbles = 0
    d_fp16 = fp32_to_fp16(0.5f);
    m_fp16 = fp32_to_fp16(-2.0f); // Non-zero offset

    decode_q4_1_to_q8_0_scalar(input_qs, d_fp16, m_fp16, output_scalar, &d_scalar_fp16);

#ifdef __AVX2__
    decode_q4_1_to_q8_0_avx2(input_qs, d_fp16, m_fp16, output_avx2, &d_avx2_fp16);

    for (int i = 0; i < 32; ++i)
    {
        EXPECT_EQ(output_scalar[i], output_avx2[i])
            << "Non-zero offset AVX2 mismatch at " << i;
    }
#endif

#if defined(__AVX512F__) && defined(__AVX512BW__)
    decode_q4_1_to_q8_0_avx512(input_qs, d_fp16, m_fp16, output_avx512, &d_avx512_fp16);

    for (int i = 0; i < 32; ++i)
    {
        EXPECT_EQ(output_scalar[i], output_avx512[i])
            << "Non-zero offset AVX512 mismatch at " << i;
    }
#endif
}

/**
 * @brief Randomized fuzz test (100 blocks)
 */
TEST_F(Q4_1_DecodeVectorizationTest, RandomizedFuzz_100Blocks)
{
    std::mt19937 gen(42);
    std::uniform_int_distribution<> byte_dist(0, 255);
    std::uniform_real_distribution<float> scale_dist(0.1f, 2.0f);
    std::uniform_real_distribution<float> offset_dist(-2.0f, 2.0f);

    for (int block = 0; block < 100; ++block)
    {
        for (int i = 0; i < 16; ++i)
        {
            input_qs[i] = static_cast<uint8_t>(byte_dist(gen));
        }

        d_fp16 = fp32_to_fp16(scale_dist(gen));
        m_fp16 = fp32_to_fp16(offset_dist(gen));

        decode_q4_1_to_q8_0_scalar(input_qs, d_fp16, m_fp16, output_scalar, &d_scalar_fp16);

#ifdef __AVX2__
        decode_q4_1_to_q8_0_avx2(input_qs, d_fp16, m_fp16, output_avx2, &d_avx2_fp16);

        for (int i = 0; i < 32; ++i)
        {
            EXPECT_EQ(output_scalar[i], output_avx2[i])
                << "Fuzz block " << block << " AVX2 mismatch at " << i;
        }
#endif

#if defined(__AVX512F__) && defined(__AVX512BW__)
        decode_q4_1_to_q8_0_avx512(input_qs, d_fp16, m_fp16, output_avx512, &d_avx512_fp16);

        for (int i = 0; i < 32; ++i)
        {
            EXPECT_EQ(output_scalar[i], output_avx512[i])
                << "Fuzz block " << block << " AVX512 mismatch at " << i;
        }
#endif
    }
}

/**
 * @brief Performance test: Scalar baseline
 */
TEST_F(Q4_1_DecodeVectorizationTest, ScalarPerformance_1000Iterations)
{
    std::memset(input_qs, 0xAA, sizeof(input_qs));
    d_fp16 = fp32_to_fp16(1.0f);
    m_fp16 = fp32_to_fp16(0.0f);

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; ++i)
    {
        decode_q4_1_to_q8_0_scalar(input_qs, d_fp16, m_fp16, output_scalar, &d_scalar_fp16);
    }
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Scalar: 1000 iterations = " << duration << " µs ("
              << (duration / 1000.0) << " µs/decode)\n";
}

/**
 * @brief Performance test: AVX2
 */
TEST_F(Q4_1_DecodeVectorizationTest, AVX2_Performance_1000Iterations)
{
#ifdef __AVX2__
    std::memset(input_qs, 0xAA, sizeof(input_qs));
    d_fp16 = fp32_to_fp16(1.0f);
    m_fp16 = fp32_to_fp16(0.0f);

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; ++i)
    {
        decode_q4_1_to_q8_0_avx2(input_qs, d_fp16, m_fp16, output_avx2, &d_avx2_fp16);
    }
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "AVX2: 1000 iterations = " << duration << " µs ("
              << (duration / 1000.0) << " µs/decode)\n";
#else
    GTEST_SKIP() << "AVX2 not available";
#endif
}

/**
 * @brief Performance test: AVX-512
 */
TEST_F(Q4_1_DecodeVectorizationTest, AVX512_Performance_1000Iterations)
{
#if defined(__AVX512F__) && defined(__AVX512BW__)
    std::memset(input_qs, 0xAA, sizeof(input_qs));
    d_fp16 = fp32_to_fp16(1.0f);
    m_fp16 = fp32_to_fp16(0.0f);

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; ++i)
    {
        decode_q4_1_to_q8_0_avx512(input_qs, d_fp16, m_fp16, output_avx512, &d_avx512_fp16);
    }
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "AVX-512: 1000 iterations = " << duration << " µs ("
              << (duration / 1000.0) << " µs/decode)\n";
#else
    GTEST_SKIP() << "AVX-512 not available";
#endif
}
