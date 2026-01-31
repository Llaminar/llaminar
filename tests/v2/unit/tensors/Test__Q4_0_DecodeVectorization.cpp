/**
 * @file Test__Q4_0_DecodeVectorization.cpp
 * @brief Unit tests for Q4_0 decode SIMD vectorization
 */

#include <gtest/gtest.h>
#include "tensors/SIMDHelpers.h"
#include "tensors/BlockStructures.h"
#include <vector>
#include <random>
#include <cmath>
#include <chrono>

using namespace llaminar2;
using namespace llaminar2::simd;

class Q4_0_DecodeVectorizationTest : public ::testing::Test
{
protected:
    static constexpr float TOLERANCE = 1e-5f;

    // Create test block with known values
    void create_test_block(uint8_t *qs, uint16_t *d, const std::vector<int8_t> &vals)
    {
        ASSERT_EQ(vals.size(), 32) << "Q4_0 block has 32 elements";

        float max_abs = 0.0f;
        for (auto v : vals)
        {
            max_abs = std::max(max_abs, std::abs(static_cast<float>(v)));
        }

        float scale = (max_abs > 0.0f) ? (max_abs / 7.0f) : 0.01f; // Q4_0 range: [-8, 7]
        *d = fp32_to_fp16(scale);

        for (size_t i = 0; i < 16; ++i)
        {
            // Clamp and pack
            int8_t v0 = std::max(static_cast<int8_t>(-8), std::min(static_cast<int8_t>(7), vals[2 * i]));
            int8_t v1 = std::max(static_cast<int8_t>(-8), std::min(static_cast<int8_t>(7), vals[2 * i + 1]));
            uint8_t nibble0 = (v0 + 8) & 0x0F;
            uint8_t nibble1 = (v1 + 8) & 0x0F;
            qs[i] = nibble0 | (nibble1 << 4);
        }
    }
};

TEST_F(Q4_0_DecodeVectorizationTest, ScalarVsAVX512_AllValues)
{
    uint8_t qs[16];
    uint16_t d;

    // Create block with pattern: -8, -7, -6, ..., 6, 7, -8, -7, ...
    std::vector<int8_t> vals(32);
    for (int i = 0; i < 32; ++i)
    {
        vals[i] = ((i % 16) - 8);
    }
    create_test_block(qs, &d, vals);

    // Decode with scalar
    int8_t scalar_qs[32];
    uint16_t scalar_d;
    decode_q4_0_to_q8_0_scalar(qs, d, scalar_qs, &scalar_d);

    // Decode with AVX512
    int8_t avx512_qs[32];
    uint16_t avx512_d;
    decode_q4_0_to_q8_0_avx512(qs, d, avx512_qs, &avx512_d);

    // Compare
    for (int i = 0; i < 32; ++i)
    {
        EXPECT_EQ(scalar_qs[i], avx512_qs[i])
            << "Mismatch at index " << i;
    }
    EXPECT_EQ(scalar_d, avx512_d) << "Scale mismatch";
}

TEST_F(Q4_0_DecodeVectorizationTest, ScalarVsAVX2_AllValues)
{
    uint8_t qs[16];
    uint16_t d;

    std::vector<int8_t> vals(32);
    for (int i = 0; i < 32; ++i)
    {
        vals[i] = ((i % 16) - 8);
    }
    create_test_block(qs, &d, vals);

    int8_t scalar_qs[32];
    uint16_t scalar_d;
    decode_q4_0_to_q8_0_scalar(qs, d, scalar_qs, &scalar_d);

    int8_t avx2_qs[32];
    uint16_t avx2_d;
    decode_q4_0_to_q8_0_avx2(qs, d, avx2_qs, &avx2_d);

    for (int i = 0; i < 32; ++i)
    {
        EXPECT_EQ(scalar_qs[i], avx2_qs[i])
            << "Mismatch at index " << i;
    }
    EXPECT_EQ(scalar_d, avx2_d);
}

TEST_F(Q4_0_DecodeVectorizationTest, AutoDispatch_MatchesScalar)
{
    uint8_t qs[16];
    uint16_t d;

    std::vector<int8_t> vals(32);
    for (int i = 0; i < 32; ++i)
    {
        vals[i] = (i % 16) - 8;
    }
    create_test_block(qs, &d, vals);

    int8_t scalar_qs[32];
    uint16_t scalar_d;
    decode_q4_0_to_q8_0_scalar(qs, d, scalar_qs, &scalar_d);

    int8_t auto_qs[32];
    uint16_t auto_d;
    decode_q4_0_to_q8_0(qs, d, auto_qs, &auto_d);

    for (int i = 0; i < 32; ++i)
    {
        EXPECT_EQ(scalar_qs[i], auto_qs[i]);
    }
    EXPECT_EQ(scalar_d, auto_d);
}

TEST_F(Q4_0_DecodeVectorizationTest, ZeroValues)
{
    uint8_t qs[16];
    uint16_t d;

    std::vector<int8_t> vals(32, 0);
    create_test_block(qs, &d, vals);

    int8_t output_qs[32];
    uint16_t output_d;
    decode_q4_0_to_q8_0(qs, d, output_qs, &output_d);

    // All zeros should quantize to zeros
    for (int i = 0; i < 32; ++i)
    {
        EXPECT_EQ(output_qs[i], 0);
    }
}

TEST_F(Q4_0_DecodeVectorizationTest, UniformPositive)
{
    uint8_t qs[16];
    uint16_t d;

    std::vector<int8_t> vals(32, 5);
    create_test_block(qs, &d, vals);

    int8_t scalar_qs[32], avx512_qs[32];
    uint16_t scalar_d, avx512_d;

    decode_q4_0_to_q8_0_scalar(qs, d, scalar_qs, &scalar_d);
    decode_q4_0_to_q8_0_avx512(qs, d, avx512_qs, &avx512_d);

    for (int i = 0; i < 32; ++i)
    {
        EXPECT_EQ(scalar_qs[i], avx512_qs[i]);
    }
}

TEST_F(Q4_0_DecodeVectorizationTest, UniformNegative)
{
    uint8_t qs[16];
    uint16_t d;

    std::vector<int8_t> vals(32, -5);
    create_test_block(qs, &d, vals);

    int8_t scalar_qs[32], avx512_qs[32];
    uint16_t scalar_d, avx512_d;

    decode_q4_0_to_q8_0_scalar(qs, d, scalar_qs, &scalar_d);
    decode_q4_0_to_q8_0_avx512(qs, d, avx512_qs, &avx512_d);

    for (int i = 0; i < 32; ++i)
    {
        EXPECT_EQ(scalar_qs[i], avx512_qs[i]);
    }
}

TEST_F(Q4_0_DecodeVectorizationTest, RandomizedFuzz_100Blocks)
{
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(-8, 7);

    for (int trial = 0; trial < 100; ++trial)
    {
        std::vector<int8_t> vals(32);
        for (auto &v : vals)
        {
            v = dist(rng);
        }

        uint8_t qs[16];
        uint16_t d;
        create_test_block(qs, &d, vals);

        int8_t scalar_qs[32], avx512_qs[32], avx2_qs[32];
        uint16_t scalar_d, avx512_d, avx2_d;

        decode_q4_0_to_q8_0_scalar(qs, d, scalar_qs, &scalar_d);
        decode_q4_0_to_q8_0_avx512(qs, d, avx512_qs, &avx512_d);
        decode_q4_0_to_q8_0_avx2(qs, d, avx2_qs, &avx2_d);

        for (int i = 0; i < 32; ++i)
        {
            EXPECT_EQ(scalar_qs[i], avx512_qs[i])
                << "AVX512 mismatch at trial " << trial << ", index " << i;
            EXPECT_EQ(scalar_qs[i], avx2_qs[i])
                << "AVX2 mismatch at trial " << trial << ", index " << i;
        }
        EXPECT_EQ(scalar_d, avx512_d) << "AVX512 scale mismatch at trial " << trial;
        EXPECT_EQ(scalar_d, avx2_d) << "AVX2 scale mismatch at trial " << trial;
    }
}

TEST_F(Q4_0_DecodeVectorizationTest, ScalarPerformance_1000Iterations)
{
    uint8_t qs[16];
    uint16_t d = fp32_to_fp16(0.5f);
    std::fill_n(qs, 16, 0x88); // All nibbles = 8 (represents 0 after -8 offset)

    int8_t output[32];
    uint16_t output_d;

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; ++i)
    {
        decode_q4_0_to_q8_0_scalar(qs, d, output, &output_d);
    }
    auto end = std::chrono::high_resolution_clock::now();

    auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Scalar: 1000 iterations = " << us << " µs ("
              << (us / 1000.0) << " µs/decode)\n";
}

TEST_F(Q4_0_DecodeVectorizationTest, AVX2_Performance_1000Iterations)
{
    uint8_t qs[16];
    uint16_t d = fp32_to_fp16(0.5f);
    std::fill_n(qs, 16, 0x88);

    int8_t output[32];
    uint16_t output_d;

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; ++i)
    {
        decode_q4_0_to_q8_0_avx2(qs, d, output, &output_d);
    }
    auto end = std::chrono::high_resolution_clock::now();

    auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "AVX2: 1000 iterations = " << us << " µs ("
              << (us / 1000.0) << " µs/decode)\n";
}

TEST_F(Q4_0_DecodeVectorizationTest, AVX512_Performance_1000Iterations)
{
    uint8_t qs[16];
    uint16_t d = fp32_to_fp16(0.5f);
    std::fill_n(qs, 16, 0x88);

    int8_t output[32];
    uint16_t output_d;

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; ++i)
    {
        decode_q4_0_to_q8_0_avx512(qs, d, output, &output_d);
    }
    auto end = std::chrono::high_resolution_clock::now();

    auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "AVX-512: 1000 iterations = " << us << " µs ("
              << (us / 1000.0) << " µs/decode)\n";
}
