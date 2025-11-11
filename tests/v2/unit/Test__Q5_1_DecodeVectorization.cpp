/**
 * @file Test__Q5_1_DecodeVectorization.cpp
 * @brief Tests for Q5_1 decode vectorization (SIMD optimizations)
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include <random>
#include <chrono>
#include <cstring>
#include "tensors/SIMDHelpers.h"

using namespace llaminar2::simd;

class Q5_1_DecodeVectorizationTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Initialize test data
        std::memset(input_qs, 0, sizeof(input_qs));
        std::memset(input_qh, 0, sizeof(input_qh));
        d_fp16 = fp32_to_fp16(1.0f);
        m_fp16 = fp32_to_fp16(0.0f);
    }

    uint8_t input_qs[16]; // 16 bytes = 32 nibbles (4 low bits)
    uint8_t input_qh[4];  // 4 bytes = 32 bits (1 high bit per element)
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
 * @brief Test scalar vs AVX-512 with all possible 5-bit values
 */
TEST_F(Q5_1_DecodeVectorizationTest, ScalarVsAVX512_AllValues)
{
#if defined(__AVX512F__) && defined(__AVX512BW__)
    // Test pattern: nibbles 0-15, high bits alternating
    for (int i = 0; i < 16; ++i)
    {
        input_qs[i] = static_cast<uint8_t>((i << 4) | i);
    }

    // Set high bits: alternating pattern
    input_qh[0] = 0xAA; // 10101010
    input_qh[1] = 0x55; // 01010101
    input_qh[2] = 0xCC; // 11001100
    input_qh[3] = 0x33; // 00110011

    d_fp16 = fp32_to_fp16(0.5f);
    m_fp16 = fp32_to_fp16(-1.0f);

    decode_q5_1_to_q8_0_scalar(input_qs, input_qh, d_fp16, m_fp16, output_scalar, &d_scalar_fp16);
    decode_q5_1_to_q8_0_avx512(input_qs, input_qh, d_fp16, m_fp16, output_avx512, &d_avx512_fp16);

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
 * @brief Test scalar vs AVX2 with all possible 5-bit values
 */
TEST_F(Q5_1_DecodeVectorizationTest, ScalarVsAVX2_AllValues)
{
#ifdef __AVX2__
    for (int i = 0; i < 16; ++i)
    {
        input_qs[i] = static_cast<uint8_t>((i << 4) | i);
    }

    input_qh[0] = 0xAA;
    input_qh[1] = 0x55;
    input_qh[2] = 0xCC;
    input_qh[3] = 0x33;

    d_fp16 = fp32_to_fp16(0.5f);
    m_fp16 = fp32_to_fp16(-1.0f);

    decode_q5_1_to_q8_0_scalar(input_qs, input_qh, d_fp16, m_fp16, output_scalar, &d_scalar_fp16);
    decode_q5_1_to_q8_0_avx2(input_qs, input_qh, d_fp16, m_fp16, output_avx2, &d_avx2_fp16);

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
TEST_F(Q5_1_DecodeVectorizationTest, AutoDispatch_MatchesScalar)
{
    std::mt19937 gen(12345);
    std::uniform_int_distribution<> dist(0, 255);

    for (int i = 0; i < 16; ++i)
    {
        input_qs[i] = static_cast<uint8_t>(dist(gen));
    }
    for (int i = 0; i < 4; ++i)
    {
        input_qh[i] = static_cast<uint8_t>(dist(gen));
    }

    d_fp16 = fp32_to_fp16(1.5f);
    m_fp16 = fp32_to_fp16(0.25f);

    int8_t output_auto[32];
    uint16_t d_auto_fp16;

    decode_q5_1_to_q8_0_scalar(input_qs, input_qh, d_fp16, m_fp16, output_scalar, &d_scalar_fp16);
    decode_q5_1_to_q8_0(input_qs, input_qh, d_fp16, m_fp16, output_auto, &d_auto_fp16);

    for (int i = 0; i < 32; ++i)
    {
        EXPECT_EQ(output_scalar[i], output_auto[i])
            << "Auto-dispatch mismatch at index " << i;
    }

    EXPECT_EQ(d_scalar_fp16, d_auto_fp16) << "Auto-dispatch scale mismatch";
}

/**
 * @brief Test zero input values (nibbles=0, high_bit=0, but with offset)
 */
TEST_F(Q5_1_DecodeVectorizationTest, ZeroValuesWithOffset)
{
    std::memset(input_qs, 0, sizeof(input_qs));
    std::memset(input_qh, 0, sizeof(input_qh));
    d_fp16 = fp32_to_fp16(1.0f);
    m_fp16 = fp32_to_fp16(-2.0f); // Offset

    int8_t output[32];
    uint16_t d_output_fp16;

    decode_q5_1_to_q8_0(input_qs, input_qh, d_fp16, m_fp16, output, &d_output_fp16);

    // Q5_1: value 0, offset -2.0 → FP32 value = 0 * 1.0 + (-2.0) = -2.0
    // After quantization to Q8_0, should be near -127 (clamped)
    for (int i = 0; i < 32; ++i)
    {
        EXPECT_GE(output[i], -128) << "Zero test: value too negative at " << i;
        EXPECT_LE(output[i], -120) << "Zero test: value too positive at " << i;
    }
}

/**
 * @brief Test maximum positive values (nibbles=15, high_bit=1 → value=31)
 */
TEST_F(Q5_1_DecodeVectorizationTest, MaximumPositive)
{
    std::memset(input_qs, 0xFF, sizeof(input_qs)); // All nibbles = 15
    std::memset(input_qh, 0xFF, sizeof(input_qh)); // All high bits = 1
    d_fp16 = fp32_to_fp16(1.0f);
    m_fp16 = fp32_to_fp16(0.0f);

    decode_q5_1_to_q8_0_scalar(input_qs, input_qh, d_fp16, m_fp16, output_scalar, &d_scalar_fp16);

#ifdef __AVX2__
    decode_q5_1_to_q8_0_avx2(input_qs, input_qh, d_fp16, m_fp16, output_avx2, &d_avx2_fp16);

    for (int i = 0; i < 32; ++i)
    {
        EXPECT_EQ(output_scalar[i], output_avx2[i])
            << "Maximum positive AVX2 mismatch at " << i;
    }
#endif

#if defined(__AVX512F__) && defined(__AVX512BW__)
    decode_q5_1_to_q8_0_avx512(input_qs, input_qh, d_fp16, m_fp16, output_avx512, &d_avx512_fp16);

    for (int i = 0; i < 32; ++i)
    {
        EXPECT_EQ(output_scalar[i], output_avx512[i])
            << "Maximum positive AVX512 mismatch at " << i;
    }
#endif
}

/**
 * @brief Test high bit extraction with non-zero offset
 */
TEST_F(Q5_1_DecodeVectorizationTest, HighBitExtractionWithOffset)
{
    // Set nibbles to 0, vary high bits
    std::memset(input_qs, 0, sizeof(input_qs));

    // Set specific high bit pattern: first 8 bits = 10101010
    input_qh[0] = 0xAA;
    input_qh[1] = 0x00;
    input_qh[2] = 0x00;
    input_qh[3] = 0x00;

    d_fp16 = fp32_to_fp16(1.0f);
    m_fp16 = fp32_to_fp16(0.5f);

    decode_q5_1_to_q8_0_scalar(input_qs, input_qh, d_fp16, m_fp16, output_scalar, &d_scalar_fp16);

#ifdef __AVX2__
    decode_q5_1_to_q8_0_avx2(input_qs, input_qh, d_fp16, m_fp16, output_avx2, &d_avx2_fp16);

    for (int i = 0; i < 32; ++i)
    {
        EXPECT_EQ(output_scalar[i], output_avx2[i])
            << "High bit AVX2 mismatch at " << i;
    }
#endif

#if defined(__AVX512F__) && defined(__AVX512BW__)
    decode_q5_1_to_q8_0_avx512(input_qs, input_qh, d_fp16, m_fp16, output_avx512, &d_avx512_fp16);

    for (int i = 0; i < 32; ++i)
    {
        EXPECT_EQ(output_scalar[i], output_avx512[i])
            << "High bit AVX512 mismatch at " << i;
    }
#endif
}

/**
 * @brief Test with large positive offset
 */
TEST_F(Q5_1_DecodeVectorizationTest, LargePositiveOffset)
{
    std::memset(input_qs, 0, sizeof(input_qs));
    std::memset(input_qh, 0, sizeof(input_qh));

    d_fp16 = fp32_to_fp16(0.5f);
    m_fp16 = fp32_to_fp16(5.0f); // Large offset

    decode_q5_1_to_q8_0_scalar(input_qs, input_qh, d_fp16, m_fp16, output_scalar, &d_scalar_fp16);

#ifdef __AVX2__
    decode_q5_1_to_q8_0_avx2(input_qs, input_qh, d_fp16, m_fp16, output_avx2, &d_avx2_fp16);

    for (int i = 0; i < 32; ++i)
    {
        EXPECT_EQ(output_scalar[i], output_avx2[i])
            << "Large offset AVX2 mismatch at " << i;
    }
#endif

#if defined(__AVX512F__) && defined(__AVX512BW__)
    decode_q5_1_to_q8_0_avx512(input_qs, input_qh, d_fp16, m_fp16, output_avx512, &d_avx512_fp16);

    for (int i = 0; i < 32; ++i)
    {
        EXPECT_EQ(output_scalar[i], output_avx512[i])
            << "Large offset AVX512 mismatch at " << i;
    }
#endif
}

/**
 * @brief Randomized fuzz test (100 blocks)
 */
TEST_F(Q5_1_DecodeVectorizationTest, RandomizedFuzz_100Blocks)
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
        for (int i = 0; i < 4; ++i)
        {
            input_qh[i] = static_cast<uint8_t>(byte_dist(gen));
        }

        d_fp16 = fp32_to_fp16(scale_dist(gen));
        m_fp16 = fp32_to_fp16(offset_dist(gen));

        decode_q5_1_to_q8_0_scalar(input_qs, input_qh, d_fp16, m_fp16, output_scalar, &d_scalar_fp16);

#ifdef __AVX2__
        decode_q5_1_to_q8_0_avx2(input_qs, input_qh, d_fp16, m_fp16, output_avx2, &d_avx2_fp16);

        for (int i = 0; i < 32; ++i)
        {
            EXPECT_EQ(output_scalar[i], output_avx2[i])
                << "Fuzz block " << block << " AVX2 mismatch at " << i;
        }
#endif

#if defined(__AVX512F__) && defined(__AVX512BW__)
        decode_q5_1_to_q8_0_avx512(input_qs, input_qh, d_fp16, m_fp16, output_avx512, &d_avx512_fp16);

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
TEST_F(Q5_1_DecodeVectorizationTest, ScalarPerformance_1000Iterations)
{
    std::memset(input_qs, 0xAA, sizeof(input_qs));
    std::memset(input_qh, 0x55, sizeof(input_qh));
    d_fp16 = fp32_to_fp16(1.0f);
    m_fp16 = fp32_to_fp16(0.0f);

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; ++i)
    {
        decode_q5_1_to_q8_0_scalar(input_qs, input_qh, d_fp16, m_fp16, output_scalar, &d_scalar_fp16);
    }
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Scalar: 1000 iterations = " << duration << " µs ("
              << (duration / 1000.0) << " µs/decode)\n";
}

/**
 * @brief Performance test: AVX2
 */
TEST_F(Q5_1_DecodeVectorizationTest, AVX2_Performance_1000Iterations)
{
#ifdef __AVX2__
    std::memset(input_qs, 0xAA, sizeof(input_qs));
    std::memset(input_qh, 0x55, sizeof(input_qh));
    d_fp16 = fp32_to_fp16(1.0f);
    m_fp16 = fp32_to_fp16(0.0f);

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; ++i)
    {
        decode_q5_1_to_q8_0_avx2(input_qs, input_qh, d_fp16, m_fp16, output_avx2, &d_avx2_fp16);
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
TEST_F(Q5_1_DecodeVectorizationTest, AVX512_Performance_1000Iterations)
{
#if defined(__AVX512F__) && defined(__AVX512BW__)
    std::memset(input_qs, 0xAA, sizeof(input_qs));
    std::memset(input_qh, 0x55, sizeof(input_qh));
    d_fp16 = fp32_to_fp16(1.0f);
    m_fp16 = fp32_to_fp16(0.0f);

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; ++i)
    {
        decode_q5_1_to_q8_0_avx512(input_qs, input_qh, d_fp16, m_fp16, output_avx512, &d_avx512_fp16);
    }
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "AVX-512: 1000 iterations = " << duration << " µs ("
              << (duration / 1000.0) << " µs/decode)\n";
#else
    GTEST_SKIP() << "AVX-512 not available";
#endif
}
