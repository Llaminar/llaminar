/**
 * @file Test__Q8_0DecodeVectorization.cpp
 * @brief Unit tests for vectorized decode_to_q8_0 implementations
 * @author David Sanftenberg
 * @date November 2025
 */

#include <gtest/gtest.h>
#include "tensors/SIMDHelpers.h"
#include "tensors/SIMDKQuantHelpers.h"
#include "tensors/FP16Utils.h"
#include "tensors/Tensors.h"
#include "kernels/cpu/gemm/Q8_0WeightAccessor.h"
#include <random>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <vector>

using namespace llaminar2;
using namespace llaminar2::simd;
using namespace llaminar2::kernels::gemm;

// ============================================================================
// Test Fixture
// ============================================================================

class Q8_0DecodeVectorizationTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Seed for reproducible tests
        rng_.seed(42);
    }

    std::mt19937 rng_;

    // Helper: Compare two Q8_0 outputs with tolerance
    struct ComparisonResult
    {
        bool passed;
        int max_diff;
        int num_mismatches;
        float scale_diff;
    };

    ComparisonResult compare_q8_0_blocks(
        const int8_t *a_qs,
        uint16_t a_scale,
        const int8_t *b_qs,
        uint16_t b_scale,
        int tolerance = 0) // Tolerance in quantized int8 domain
    {
        ComparisonResult result{true, 0, 0, 0.0f};

        // Compare scales
        float scale_a = fp16_to_fp32(a_scale);
        float scale_b = fp16_to_fp32(b_scale);
        result.scale_diff = std::fabs(scale_a - scale_b);

        // Compare quantized values
        for (int i = 0; i < 32; ++i)
        {
            int diff = std::abs(static_cast<int>(a_qs[i]) - static_cast<int>(b_qs[i]));
            result.max_diff = std::max(result.max_diff, diff);
            if (diff > tolerance)
            {
                result.num_mismatches++;
                result.passed = false;
            }
        }

        return result;
    }

    // Helper: Generate random 4-bit packed data (IQ4_NL format)
    void generate_random_iq4nl(uint8_t *qs, uint16_t *scale)
    {
        std::uniform_int_distribution<int> nibble_dist(0, 15);
        std::uniform_real_distribution<float> scale_dist(0.001f, 1.0f);

        // Generate 16 bytes of packed 4-bit indices
        for (int i = 0; i < 16; ++i)
        {
            uint8_t low = nibble_dist(rng_);
            uint8_t high = nibble_dist(rng_);
            qs[i] = (high << 4) | low;
        }

        // Generate scale
        *scale = fp32_to_fp16(scale_dist(rng_));
    }

    // Helper: Generate random FP32 data
    void generate_random_fp32(float *data, size_t count, float min_val = -1.0f, float max_val = 1.0f)
    {
        std::uniform_real_distribution<float> dist(min_val, max_val);
        for (size_t i = 0; i < count; ++i)
        {
            data[i] = dist(rng_);
        }
    }

    void generate_random_q4_0(uint8_t *qs, uint16_t *scale)
    {
        std::uniform_int_distribution<int> nibble_dist(0, 15);
        std::uniform_real_distribution<float> scale_dist(0.01f, 3.0f);

        for (int i = 0; i < 16; ++i)
        {
            const uint8_t low = static_cast<uint8_t>(nibble_dist(rng_));
            const uint8_t high = static_cast<uint8_t>(nibble_dist(rng_));
            qs[i] = static_cast<uint8_t>((high << 4) | low);
        }

        *scale = fp32_to_fp16(scale_dist(rng_));
    }

    void generate_random_q4_1(uint8_t *qs, uint16_t *scale, uint16_t *min_val)
    {
        generate_random_q4_0(qs, scale);
        std::uniform_real_distribution<float> min_dist(-1.0f, 1.0f);
        *min_val = fp32_to_fp16(min_dist(rng_));
    }

    void generate_random_q5_0(uint8_t *qs, uint8_t *qh, uint16_t *scale)
    {
        std::uniform_int_distribution<int> val_dist(0, 31);
        std::uniform_real_distribution<float> scale_dist(0.01f, 3.0f);

        uint8_t qh_bytes[4] = {0, 0, 0, 0};

        for (int i = 0; i < 16; ++i)
        {
            const int v0 = val_dist(rng_);
            const int v1 = val_dist(rng_);

            qs[i] = static_cast<uint8_t>(((v1 & 0x0F) << 4) | (v0 & 0x0F));

            if (v0 & 0x10)
            {
                qh_bytes[i / 8] |= static_cast<uint8_t>(1u << (i % 8));
            }
            if (v1 & 0x10)
            {
                const int idx = i + 16;
                qh_bytes[idx / 8] |= static_cast<uint8_t>(1u << (idx % 8));
            }
        }

        std::memcpy(qh, qh_bytes, sizeof(qh_bytes));
        *scale = fp32_to_fp16(scale_dist(rng_));
    }

    void generate_random_q5_1(uint8_t *qs, uint8_t *qh, uint16_t *scale, uint16_t *min_val)
    {
        generate_random_q5_0(qs, qh, scale);
        std::uniform_real_distribution<float> min_dist(-1.0f, 1.0f);
        *min_val = fp32_to_fp16(min_dist(rng_));
    }

    void generate_random_q2_k(Q2_KBlock &block)
    {
        std::uniform_int_distribution<int> scale_nibble_dist(0, 15);
        std::uniform_int_distribution<int> value_dist(0, 3);
        std::uniform_real_distribution<float> d_dist(0.01f, 3.0f);
        std::uniform_real_distribution<float> dmin_dist(0.01f, 3.0f);

        for (auto &s : block.scales)
        {
            s = static_cast<uint8_t>(scale_nibble_dist(rng_));
        }

        for (auto &packed : block.qs)
        {
            uint8_t byte = 0;
            byte |= static_cast<uint8_t>(value_dist(rng_) & 0x3);
            byte |= static_cast<uint8_t>((value_dist(rng_) & 0x3) << 2);
            byte |= static_cast<uint8_t>((value_dist(rng_) & 0x3) << 4);
            byte |= static_cast<uint8_t>((value_dist(rng_) & 0x3) << 6);
            packed = byte;
        }

        block.d = fp32_to_fp16(d_dist(rng_));
        block.dmin = fp32_to_fp16(dmin_dist(rng_));
    }

    void generate_random_q6_k(Q6_KBlock &block)
    {
        std::uniform_int_distribution<int> q_dist(-32, 31);
        std::uniform_int_distribution<int> scale_dist(-16, 16);
        std::uniform_real_distribution<float> d_dist(0.01f, 3.0f);

        for (int half = 0; half < 2; ++half)
        {
            for (int l = 0; l < 32; ++l)
            {
                const int q1 = q_dist(rng_) + 32;
                const int q2 = q_dist(rng_) + 32;
                const int q3 = q_dist(rng_) + 32;
                const int q4 = q_dist(rng_) + 32;

                block.ql[half * 64 + l] = static_cast<uint8_t>(((q3 & 0x0F) << 4) | (q1 & 0x0F));
                block.ql[half * 64 + l + 32] = static_cast<uint8_t>(((q4 & 0x0F) << 4) | (q2 & 0x0F));

                const uint8_t qh_val = static_cast<uint8_t>(((q1 >> 4) & 0x03) |
                                                            (((q2 >> 4) & 0x03) << 2) |
                                                            (((q3 >> 4) & 0x03) << 4) |
                                                            (((q4 >> 4) & 0x03) << 6));
                block.qh[half * 32 + l] = qh_val;
            }

            for (int s = 0; s < 8; ++s)
            {
                int val = scale_dist(rng_);
                val = std::clamp(val, -32, 31);
                block.scales[half * 8 + s] = static_cast<int8_t>(val);
            }
        }

        block.d = fp32_to_fp16(d_dist(rng_));
    }

    void generate_random_q8_k(Q8_KBlock &block)
    {
        std::uniform_int_distribution<int> q_dist(-127, 127);
        std::uniform_int_distribution<int> sum_dist(-4096, 4096);

        for (auto &v : block.qs)
        {
            v = static_cast<int8_t>(q_dist(rng_));
        }

        for (auto &s : block.bsums)
        {
            s = static_cast<int16_t>(sum_dist(rng_));
        }
    }
};

// ============================================================================
// IQ4_NL → Q8_0 Decode Tests
// ============================================================================

TEST_F(Q8_0DecodeVectorizationTest, IQ4NL_ScalarVsAVX2)
{
#if defined(__AVX2__)
    for (int trial = 0; trial < 100; ++trial)
    {
        // Generate random IQ4_NL block
        alignas(64) uint8_t iq4_qs[16];
        uint16_t iq4_scale;
        generate_random_iq4nl(iq4_qs, &iq4_scale);

        // Decode with scalar
        alignas(64) int8_t scalar_qs[32];
        uint16_t scalar_scale;
        decode_iq4nl_to_q8_0_scalar(iq4_qs, iq4_scale, scalar_qs, &scalar_scale);

        // Decode with AVX2
        alignas(64) int8_t avx2_qs[32];
        uint16_t avx2_scale;
        decode_iq4nl_to_q8_0_avx2(iq4_qs, iq4_scale, avx2_qs, &avx2_scale);

        // Compare results (should be identical)
        auto result = compare_q8_0_blocks(scalar_qs, scalar_scale, avx2_qs, avx2_scale, 0);

        EXPECT_TRUE(result.passed)
            << "Trial " << trial << ": AVX2 mismatch"
            << " max_diff=" << result.max_diff
            << " mismatches=" << result.num_mismatches;

        EXPECT_FLOAT_EQ(fp16_to_fp32(scalar_scale), fp16_to_fp32(avx2_scale))
            << "Trial " << trial << ": Scale mismatch";
    }
#else
    GTEST_SKIP() << "AVX2 not available";
#endif
}

TEST_F(Q8_0DecodeVectorizationTest, IQ4NL_ScalarVsAVX512)
{
#if defined(__AVX512F__) && defined(__AVX512BW__)
    for (int trial = 0; trial < 100; ++trial)
    {
        // Generate random IQ4_NL block
        alignas(64) uint8_t iq4_qs[16];
        uint16_t iq4_scale;
        generate_random_iq4nl(iq4_qs, &iq4_scale);

        // Decode with scalar
        alignas(64) int8_t scalar_qs[32];
        uint16_t scalar_scale;
        decode_iq4nl_to_q8_0_scalar(iq4_qs, iq4_scale, scalar_qs, &scalar_scale);

        // Decode with AVX-512
        alignas(64) int8_t avx512_qs[32];
        uint16_t avx512_scale;
        decode_iq4nl_to_q8_0_avx512(iq4_qs, iq4_scale, avx512_qs, &avx512_scale);

        // Compare results (should be identical)
        auto result = compare_q8_0_blocks(scalar_qs, scalar_scale, avx512_qs, avx512_scale, 0);

        EXPECT_TRUE(result.passed)
            << "Trial " << trial << ": AVX-512 mismatch"
            << " max_diff=" << result.max_diff
            << " mismatches=" << result.num_mismatches;

        EXPECT_FLOAT_EQ(fp16_to_fp32(scalar_scale), fp16_to_fp32(avx512_scale))
            << "Trial " << trial << ": Scale mismatch";
    }
#else
    GTEST_SKIP() << "AVX-512 not available";
#endif
}

TEST_F(Q8_0DecodeVectorizationTest, IQ4NL_AutoDispatch)
{
    // Test auto-dispatched function matches scalar
    for (int trial = 0; trial < 100; ++trial)
    {
        alignas(64) uint8_t iq4_qs[16];
        uint16_t iq4_scale;
        generate_random_iq4nl(iq4_qs, &iq4_scale);

        alignas(64) int8_t scalar_qs[32];
        uint16_t scalar_scale;
        decode_iq4nl_to_q8_0_scalar(iq4_qs, iq4_scale, scalar_qs, &scalar_scale);

        alignas(64) int8_t auto_qs[32];
        uint16_t auto_scale;
        decode_iq4nl_to_q8_0(iq4_qs, iq4_scale, auto_qs, &auto_scale);

        auto result = compare_q8_0_blocks(scalar_qs, scalar_scale, auto_qs, auto_scale, 0);

        EXPECT_TRUE(result.passed)
            << "Trial " << trial << ": Auto-dispatch mismatch";
    }
}

// ============================================================================
// Q4_0 → Q8_0 Decode Tests
// ============================================================================

TEST_F(Q8_0DecodeVectorizationTest, Q4_0_ScalarVsAVX2)
{
#if defined(__AVX2__)
    for (int trial = 0; trial < 100; ++trial)
    {
        alignas(32) uint8_t qs[16];
        uint16_t scale;
        generate_random_q4_0(qs, &scale);

        alignas(32) int8_t scalar_qs[32];
        uint16_t scalar_scale;
        decode_q4_0_to_q8_0_scalar(qs, scale, scalar_qs, &scalar_scale);

        alignas(32) int8_t avx2_qs[32];
        uint16_t avx2_scale;
        decode_q4_0_to_q8_0_avx2(qs, scale, avx2_qs, &avx2_scale);

        auto result = compare_q8_0_blocks(scalar_qs, scalar_scale, avx2_qs, avx2_scale, 0);
        EXPECT_TRUE(result.passed) << "Trial " << trial << ": Q4_0 AVX2 mismatch";
        EXPECT_FLOAT_EQ(fp16_to_fp32(scalar_scale), fp16_to_fp32(avx2_scale));
    }
#else
    GTEST_SKIP() << "AVX2 not available";
#endif
}

TEST_F(Q8_0DecodeVectorizationTest, Q4_0_ScalarVsAVX512)
{
#if defined(__AVX512F__) && defined(__AVX512BW__)
    for (int trial = 0; trial < 100; ++trial)
    {
        alignas(64) uint8_t qs[16];
        uint16_t scale;
        generate_random_q4_0(qs, &scale);

        alignas(32) int8_t scalar_qs[32];
        uint16_t scalar_scale;
        decode_q4_0_to_q8_0_scalar(qs, scale, scalar_qs, &scalar_scale);

        alignas(32) int8_t avx512_qs[32];
        uint16_t avx512_scale;
        decode_q4_0_to_q8_0_avx512(qs, scale, avx512_qs, &avx512_scale);

        auto result = compare_q8_0_blocks(scalar_qs, scalar_scale, avx512_qs, avx512_scale, 0);
        EXPECT_TRUE(result.passed) << "Trial " << trial << ": Q4_0 AVX512 mismatch";
        EXPECT_FLOAT_EQ(fp16_to_fp32(scalar_scale), fp16_to_fp32(avx512_scale));
    }
#else
    GTEST_SKIP() << "AVX-512 not available";
#endif
}

TEST_F(Q8_0DecodeVectorizationTest, Q4_0_AutoDispatch)
{
    for (int trial = 0; trial < 100; ++trial)
    {
        alignas(32) uint8_t qs[16];
        uint16_t scale;
        generate_random_q4_0(qs, &scale);

        alignas(32) int8_t scalar_qs[32];
        uint16_t scalar_scale;
        decode_q4_0_to_q8_0_scalar(qs, scale, scalar_qs, &scalar_scale);

        alignas(32) int8_t auto_qs[32];
        uint16_t auto_scale;
        decode_q4_0_to_q8_0(qs, scale, auto_qs, &auto_scale);

        auto result = compare_q8_0_blocks(scalar_qs, scalar_scale, auto_qs, auto_scale, 0);
        EXPECT_TRUE(result.passed) << "Trial " << trial << ": Q4_0 auto mismatch";
    }
}

// ============================================================================
// Q4_1 → Q8_0 Decode Tests
// ============================================================================

TEST_F(Q8_0DecodeVectorizationTest, Q4_1_ScalarVsAVX2)
{
#if defined(__AVX2__)
    for (int trial = 0; trial < 100; ++trial)
    {
        alignas(32) uint8_t qs[16];
        uint16_t scale;
        uint16_t min_val;
        generate_random_q4_1(qs, &scale, &min_val);

        alignas(32) int8_t scalar_qs[32];
        uint16_t scalar_scale;
        decode_q4_1_to_q8_0_scalar(qs, scale, min_val, scalar_qs, &scalar_scale);

        alignas(32) int8_t avx2_qs[32];
        uint16_t avx2_scale;
        decode_q4_1_to_q8_0_avx2(qs, scale, min_val, avx2_qs, &avx2_scale);

        auto result = compare_q8_0_blocks(scalar_qs, scalar_scale, avx2_qs, avx2_scale, 1);
        EXPECT_TRUE(result.passed) << "Trial " << trial << ": Q4_1 AVX2 mismatch";
        EXPECT_NEAR(fp16_to_fp32(scalar_scale), fp16_to_fp32(avx2_scale), 0.001f);
    }
#else
    GTEST_SKIP() << "AVX2 not available";
#endif
}

TEST_F(Q8_0DecodeVectorizationTest, Q4_1_ScalarVsAVX512)
{
#if defined(__AVX512F__) && defined(__AVX512BW__)
    for (int trial = 0; trial < 100; ++trial)
    {
        alignas(64) uint8_t qs[16];
        uint16_t scale;
        uint16_t min_val;
        generate_random_q4_1(qs, &scale, &min_val);

        alignas(32) int8_t scalar_qs[32];
        uint16_t scalar_scale;
        decode_q4_1_to_q8_0_scalar(qs, scale, min_val, scalar_qs, &scalar_scale);

        alignas(32) int8_t avx512_qs[32];
        uint16_t avx512_scale;
        decode_q4_1_to_q8_0_avx512(qs, scale, min_val, avx512_qs, &avx512_scale);

        auto result = compare_q8_0_blocks(scalar_qs, scalar_scale, avx512_qs, avx512_scale, 1);
        EXPECT_TRUE(result.passed) << "Trial " << trial << ": Q4_1 AVX512 mismatch";
        EXPECT_NEAR(fp16_to_fp32(scalar_scale), fp16_to_fp32(avx512_scale), 0.001f);
    }
#else
    GTEST_SKIP() << "AVX-512 not available";
#endif
}

TEST_F(Q8_0DecodeVectorizationTest, Q4_1_AutoDispatch)
{
    for (int trial = 0; trial < 100; ++trial)
    {
        alignas(32) uint8_t qs[16];
        uint16_t scale;
        uint16_t min_val;
        generate_random_q4_1(qs, &scale, &min_val);

        alignas(32) int8_t scalar_qs[32];
        uint16_t scalar_scale;
        decode_q4_1_to_q8_0_scalar(qs, scale, min_val, scalar_qs, &scalar_scale);

        alignas(32) int8_t auto_qs[32];
        uint16_t auto_scale;
        decode_q4_1_to_q8_0(qs, scale, min_val, auto_qs, &auto_scale);

        auto result = compare_q8_0_blocks(scalar_qs, scalar_scale, auto_qs, auto_scale, 1);
        EXPECT_TRUE(result.passed) << "Trial " << trial << ": Q4_1 auto mismatch";
    }
}

// ============================================================================
// Q5_0 → Q8_0 Decode Tests
// ============================================================================

TEST_F(Q8_0DecodeVectorizationTest, Q5_0_ScalarVsAVX2)
{
#if defined(__AVX2__)
    for (int trial = 0; trial < 100; ++trial)
    {
        alignas(32) uint8_t qs[16];
        alignas(16) uint8_t qh[4];
        uint16_t scale;
        generate_random_q5_0(qs, qh, &scale);

        alignas(32) int8_t scalar_qs[32];
        uint16_t scalar_scale;
        decode_q5_0_to_q8_0_scalar(qs, qh, scale, scalar_qs, &scalar_scale);

        alignas(32) int8_t avx2_qs[32];
        uint16_t avx2_scale;
        decode_q5_0_to_q8_0_avx2(qs, qh, scale, avx2_qs, &avx2_scale);

        auto result = compare_q8_0_blocks(scalar_qs, scalar_scale, avx2_qs, avx2_scale, 0);
        EXPECT_TRUE(result.passed) << "Trial " << trial << ": Q5_0 AVX2 mismatch";
        EXPECT_FLOAT_EQ(fp16_to_fp32(scalar_scale), fp16_to_fp32(avx2_scale));
    }
#else
    GTEST_SKIP() << "AVX2 not available";
#endif
}

TEST_F(Q8_0DecodeVectorizationTest, Q5_0_ScalarVsAVX512)
{
#if defined(__AVX512F__) && defined(__AVX512BW__)
    for (int trial = 0; trial < 100; ++trial)
    {
        alignas(64) uint8_t qs[16];
        alignas(16) uint8_t qh[4];
        uint16_t scale;
        generate_random_q5_0(qs, qh, &scale);

        alignas(32) int8_t scalar_qs[32];
        uint16_t scalar_scale;
        decode_q5_0_to_q8_0_scalar(qs, qh, scale, scalar_qs, &scalar_scale);

        alignas(32) int8_t avx512_qs[32];
        uint16_t avx512_scale;
        decode_q5_0_to_q8_0_avx512(qs, qh, scale, avx512_qs, &avx512_scale);

        auto result = compare_q8_0_blocks(scalar_qs, scalar_scale, avx512_qs, avx512_scale, 0);
        EXPECT_TRUE(result.passed) << "Trial " << trial << ": Q5_0 AVX512 mismatch";
        EXPECT_FLOAT_EQ(fp16_to_fp32(scalar_scale), fp16_to_fp32(avx512_scale));
    }
#else
    GTEST_SKIP() << "AVX-512 not available";
#endif
}

TEST_F(Q8_0DecodeVectorizationTest, Q5_0_AutoDispatch)
{
    for (int trial = 0; trial < 100; ++trial)
    {
        alignas(32) uint8_t qs[16];
        alignas(16) uint8_t qh[4];
        uint16_t scale;
        generate_random_q5_0(qs, qh, &scale);

        alignas(32) int8_t scalar_qs[32];
        uint16_t scalar_scale;
        decode_q5_0_to_q8_0_scalar(qs, qh, scale, scalar_qs, &scalar_scale);

        alignas(32) int8_t auto_qs[32];
        uint16_t auto_scale;
        decode_q5_0_to_q8_0(qs, qh, scale, auto_qs, &auto_scale);

        auto result = compare_q8_0_blocks(scalar_qs, scalar_scale, auto_qs, auto_scale, 0);
        EXPECT_TRUE(result.passed) << "Trial " << trial << ": Q5_0 auto mismatch";
    }
}

// ============================================================================
// Q5_1 → Q8_0 Decode Tests
// ============================================================================

TEST_F(Q8_0DecodeVectorizationTest, Q5_1_ScalarVsAVX2)
{
#if defined(__AVX2__)
    for (int trial = 0; trial < 100; ++trial)
    {
        alignas(32) uint8_t qs[16];
        alignas(16) uint8_t qh[4];
        uint16_t scale;
        uint16_t min_val;
        generate_random_q5_1(qs, qh, &scale, &min_val);

        alignas(32) int8_t scalar_qs[32];
        uint16_t scalar_scale;
        decode_q5_1_to_q8_0_scalar(qs, qh, scale, min_val, scalar_qs, &scalar_scale);

        alignas(32) int8_t avx2_qs[32];
        uint16_t avx2_scale;
        decode_q5_1_to_q8_0_avx2(qs, qh, scale, min_val, avx2_qs, &avx2_scale);

        auto result = compare_q8_0_blocks(scalar_qs, scalar_scale, avx2_qs, avx2_scale, 1);
        EXPECT_TRUE(result.passed) << "Trial " << trial << ": Q5_1 AVX2 mismatch";
        EXPECT_NEAR(fp16_to_fp32(scalar_scale), fp16_to_fp32(avx2_scale), 0.001f);
    }
#else
    GTEST_SKIP() << "AVX2 not available";
#endif
}

// ============================================================================
// Q2_K → Q8_0 Decode Tests
// ============================================================================

TEST_F(Q8_0DecodeVectorizationTest, Q2_K_BlockDecodeMatchesScalarQuantizer)
{
    for (int trial = 0; trial < 50; ++trial)
    {
        Q2_KBlock q2_block{};
        generate_random_q2_k(q2_block);

        // Reference decode using scalar path followed by scalar quantization
        alignas(64) float full_decoded[Q2_KBlock::BLOCK_SIZE];
        Q2_KTensor::decodeBlockScalar(q2_block, full_decoded);

        std::vector<uint8_t> raw(sizeof(Q2_KBlock));
        std::memcpy(raw.data(), &q2_block, raw.size());
        Q2_KTensor tensor({1, Q2_KBlock::BLOCK_SIZE}, raw);

        const size_t total_blocks = Q2_KBlock::BLOCK_SIZE / Q8_0Block::BLOCK_SIZE;
        for (size_t block_idx = 0; block_idx < total_blocks; ++block_idx)
        {
            Q8_0Block decoded{};
            tensor.decode_to_q8_0(0, block_idx, &decoded);

            alignas(64) int8_t ref_qs[Q8_0Block::BLOCK_SIZE];
            uint16_t ref_scale = 0;
            simd::quantize_fp32_to_q8_0_scalar(full_decoded + block_idx * Q8_0Block::BLOCK_SIZE,
                                               Q8_0Block::BLOCK_SIZE,
                                               ref_qs,
                                               &ref_scale);

            auto result = compare_q8_0_blocks(decoded.qs, decoded.d, ref_qs, ref_scale, 0);

            EXPECT_TRUE(result.passed)
                << "Trial " << trial << ", sub-block " << block_idx << ": mismatch detected";
            EXPECT_FLOAT_EQ(fp16_to_fp32(decoded.d), fp16_to_fp32(ref_scale));
        }
    }
}

TEST_F(Q8_0DecodeVectorizationTest, Q2_K_ScalarVsAVX2_Helper)
{
#if defined(__AVX2__)
    for (int trial = 0; trial < 50; ++trial)
    {
        Q2_KBlock block{};
        generate_random_q2_k(block);

        for (size_t sub = 0; sub < 8; ++sub)
        {
            alignas(64) int8_t scalar_qs[Q8_0Block::BLOCK_SIZE];
            uint16_t scalar_scale;
            decode_q2_k_to_q8_0_scalar(block, sub, scalar_qs, &scalar_scale);

            alignas(64) int8_t avx2_qs[Q8_0Block::BLOCK_SIZE];
            uint16_t avx2_scale;
            decode_q2_k_to_q8_0_avx2(block, sub, avx2_qs, &avx2_scale);

            auto result = compare_q8_0_blocks(scalar_qs, scalar_scale, avx2_qs, avx2_scale, 0);
            EXPECT_TRUE(result.passed) << "Trial " << trial << ", sub " << sub << ": AVX2 mismatch";
        }
    }
#else
    GTEST_SKIP() << "AVX2 not available";
#endif
}

TEST_F(Q8_0DecodeVectorizationTest, Q2_K_ScalarVsAVX512_Helper)
{
#if defined(__AVX512F__) && defined(__AVX512BW__)
    for (int trial = 0; trial < 50; ++trial)
    {
        Q2_KBlock block{};
        generate_random_q2_k(block);

        for (size_t sub = 0; sub < 8; ++sub)
        {
            alignas(64) int8_t scalar_qs[Q8_0Block::BLOCK_SIZE];
            uint16_t scalar_scale;
            decode_q2_k_to_q8_0_scalar(block, sub, scalar_qs, &scalar_scale);

            alignas(64) int8_t avx512_qs[Q8_0Block::BLOCK_SIZE];
            uint16_t avx512_scale;
            decode_q2_k_to_q8_0_avx512(block, sub, avx512_qs, &avx512_scale);

            auto result = compare_q8_0_blocks(scalar_qs, scalar_scale, avx512_qs, avx512_scale, 0);
            EXPECT_TRUE(result.passed) << "Trial " << trial << ", sub " << sub << ": AVX512 mismatch";
        }
    }
#else
    GTEST_SKIP() << "AVX-512 not available";
#endif
}

TEST_F(Q8_0DecodeVectorizationTest, Q2_K_AutoDispatch_Helper)
{
    for (int trial = 0; trial < 50; ++trial)
    {
        Q2_KBlock block{};
        generate_random_q2_k(block);

        for (size_t sub = 0; sub < 8; ++sub)
        {
            alignas(64) int8_t scalar_qs[Q8_0Block::BLOCK_SIZE];
            uint16_t scalar_scale;
            decode_q2_k_to_q8_0_scalar(block, sub, scalar_qs, &scalar_scale);

            alignas(64) int8_t auto_qs[Q8_0Block::BLOCK_SIZE];
            uint16_t auto_scale;
            decode_q2_k_to_q8_0(block, sub, auto_qs, &auto_scale);

            auto result = compare_q8_0_blocks(scalar_qs, scalar_scale, auto_qs, auto_scale, 0);
            EXPECT_TRUE(result.passed) << "Trial " << trial << ": auto-dispatch mismatch";
        }
    }
}

// ============================================================================
// Q6_K → Q8_0 Helper Tests
// ============================================================================

TEST_F(Q8_0DecodeVectorizationTest, Q6_K_ScalarVsAVX2_Helper)
{
#if defined(__AVX2__)
    for (int trial = 0; trial < 50; ++trial)
    {
        Q6_KBlock block{};
        generate_random_q6_k(block);

        for (size_t sub = 0; sub < 8; ++sub)
        {
            alignas(64) int8_t scalar_qs[Q8_0Block::BLOCK_SIZE];
            uint16_t scalar_scale;
            decode_q6_k_to_q8_0_scalar(block, sub, scalar_qs, &scalar_scale);

            alignas(64) int8_t avx2_qs[Q8_0Block::BLOCK_SIZE];
            uint16_t avx2_scale;
            decode_q6_k_to_q8_0_avx2(block, sub, avx2_qs, &avx2_scale);

            auto result = compare_q8_0_blocks(scalar_qs, scalar_scale, avx2_qs, avx2_scale, 0);
            EXPECT_TRUE(result.passed) << "Trial " << trial << ", sub " << sub << ": AVX2 mismatch";
        }
    }
#else
    GTEST_SKIP() << "AVX2 not available";
#endif
}

TEST_F(Q8_0DecodeVectorizationTest, Q6_K_ScalarVsAVX512_Helper)
{
#if defined(__AVX512F__) && defined(__AVX512BW__)
    for (int trial = 0; trial < 50; ++trial)
    {
        Q6_KBlock block{};
        generate_random_q6_k(block);

        for (size_t sub = 0; sub < 8; ++sub)
        {
            alignas(64) int8_t scalar_qs[Q8_0Block::BLOCK_SIZE];
            uint16_t scalar_scale;
            decode_q6_k_to_q8_0_scalar(block, sub, scalar_qs, &scalar_scale);

            alignas(64) int8_t avx512_qs[Q8_0Block::BLOCK_SIZE];
            uint16_t avx512_scale;
            decode_q6_k_to_q8_0_avx512(block, sub, avx512_qs, &avx512_scale);

            auto result = compare_q8_0_blocks(scalar_qs, scalar_scale, avx512_qs, avx512_scale, 0);
            EXPECT_TRUE(result.passed) << "Trial " << trial << ", sub " << sub << ": AVX512 mismatch";
        }
    }
#else
    GTEST_SKIP() << "AVX-512 not available";
#endif
}

TEST_F(Q8_0DecodeVectorizationTest, Q6_K_AutoDispatch_Helper)
{
    for (int trial = 0; trial < 50; ++trial)
    {
        Q6_KBlock block{};
        generate_random_q6_k(block);

        for (size_t sub = 0; sub < 8; ++sub)
        {
            alignas(64) int8_t scalar_qs[Q8_0Block::BLOCK_SIZE];
            uint16_t scalar_scale;
            decode_q6_k_to_q8_0_scalar(block, sub, scalar_qs, &scalar_scale);

            alignas(64) int8_t auto_qs[Q8_0Block::BLOCK_SIZE];
            uint16_t auto_scale;
            decode_q6_k_to_q8_0(block, sub, auto_qs, &auto_scale);

            auto result = compare_q8_0_blocks(scalar_qs, scalar_scale, auto_qs, auto_scale, 0);
            EXPECT_TRUE(result.passed) << "Trial " << trial << ": auto-dispatch mismatch";
        }
    }
}

// ============================================================================
// Q8_K → Q8_0 Helper Tests
// ============================================================================

TEST_F(Q8_0DecodeVectorizationTest, Q8_K_ScalarVsAVX2_Helper)
{
#if defined(__AVX2__)
    for (int trial = 0; trial < 50; ++trial)
    {
        Q8_KBlock block{};
        generate_random_q8_k(block);

        for (size_t sub = 0; sub < 8; ++sub)
        {
            alignas(64) int8_t scalar_qs[Q8_0Block::BLOCK_SIZE];
            uint16_t scalar_scale;
            decode_q8_k_to_q8_0_scalar(block, sub, scalar_qs, &scalar_scale);

            alignas(64) int8_t avx2_qs[Q8_0Block::BLOCK_SIZE];
            uint16_t avx2_scale;
            decode_q8_k_to_q8_0_avx2(block, sub, avx2_qs, &avx2_scale);

            auto result = compare_q8_0_blocks(scalar_qs, scalar_scale, avx2_qs, avx2_scale, 0);
            EXPECT_TRUE(result.passed) << "Trial " << trial << ", sub " << sub << ": AVX2 mismatch";
        }
    }
#else
    GTEST_SKIP() << "AVX2 not available";
#endif
}

TEST_F(Q8_0DecodeVectorizationTest, Q8_K_ScalarVsAVX512_Helper)
{
#if defined(__AVX512F__) && defined(__AVX512BW__)
    for (int trial = 0; trial < 50; ++trial)
    {
        Q8_KBlock block{};
        generate_random_q8_k(block);

        for (size_t sub = 0; sub < 8; ++sub)
        {
            alignas(64) int8_t scalar_qs[Q8_0Block::BLOCK_SIZE];
            uint16_t scalar_scale;
            decode_q8_k_to_q8_0_scalar(block, sub, scalar_qs, &scalar_scale);

            alignas(64) int8_t avx512_qs[Q8_0Block::BLOCK_SIZE];
            uint16_t avx512_scale;
            decode_q8_k_to_q8_0_avx512(block, sub, avx512_qs, &avx512_scale);

            auto result = compare_q8_0_blocks(scalar_qs, scalar_scale, avx512_qs, avx512_scale, 0);
            EXPECT_TRUE(result.passed) << "Trial " << trial << ", sub " << sub << ": AVX512 mismatch";
        }
    }
#else
    GTEST_SKIP() << "AVX-512 not available";
#endif
}

TEST_F(Q8_0DecodeVectorizationTest, Q8_K_AutoDispatch_Helper)
{
    for (int trial = 0; trial < 50; ++trial)
    {
        Q8_KBlock block{};
        generate_random_q8_k(block);

        for (size_t sub = 0; sub < 8; ++sub)
        {
            alignas(64) int8_t scalar_qs[Q8_0Block::BLOCK_SIZE];
            uint16_t scalar_scale;
            decode_q8_k_to_q8_0_scalar(block, sub, scalar_qs, &scalar_scale);

            alignas(64) int8_t auto_qs[Q8_0Block::BLOCK_SIZE];
            uint16_t auto_scale;
            decode_q8_k_to_q8_0(block, sub, auto_qs, &auto_scale);

            auto result = compare_q8_0_blocks(scalar_qs, scalar_scale, auto_qs, auto_scale, 0);
            EXPECT_TRUE(result.passed) << "Trial " << trial << ": auto-dispatch mismatch";
        }
    }
}

TEST_F(Q8_0DecodeVectorizationTest, Q5_1_ScalarVsAVX512)
{
#if defined(__AVX512F__) && defined(__AVX512BW__)
    for (int trial = 0; trial < 100; ++trial)
    {
        alignas(64) uint8_t qs[16];
        alignas(16) uint8_t qh[4];
        uint16_t scale;
        uint16_t min_val;
        generate_random_q5_1(qs, qh, &scale, &min_val);

        alignas(32) int8_t scalar_qs[32];
        uint16_t scalar_scale;
        decode_q5_1_to_q8_0_scalar(qs, qh, scale, min_val, scalar_qs, &scalar_scale);

        alignas(32) int8_t avx512_qs[32];
        uint16_t avx512_scale;
        decode_q5_1_to_q8_0_avx512(qs, qh, scale, min_val, avx512_qs, &avx512_scale);

        auto result = compare_q8_0_blocks(scalar_qs, scalar_scale, avx512_qs, avx512_scale, 1);
        EXPECT_TRUE(result.passed) << "Trial " << trial << ": Q5_1 AVX512 mismatch";
        EXPECT_NEAR(fp16_to_fp32(scalar_scale), fp16_to_fp32(avx512_scale), 0.001f);
    }
#else
    GTEST_SKIP() << "AVX-512 not available";
#endif
}

TEST_F(Q8_0DecodeVectorizationTest, Q5_1_AutoDispatch)
{
    for (int trial = 0; trial < 100; ++trial)
    {
        alignas(32) uint8_t qs[16];
        alignas(16) uint8_t qh[4];
        uint16_t scale;
        uint16_t min_val;
        generate_random_q5_1(qs, qh, &scale, &min_val);

        alignas(32) int8_t scalar_qs[32];
        uint16_t scalar_scale;
        decode_q5_1_to_q8_0_scalar(qs, qh, scale, min_val, scalar_qs, &scalar_scale);

        alignas(32) int8_t auto_qs[32];
        uint16_t auto_scale;
        decode_q5_1_to_q8_0(qs, qh, scale, min_val, auto_qs, &auto_scale);

        auto result = compare_q8_0_blocks(scalar_qs, scalar_scale, auto_qs, auto_scale, 1);
        EXPECT_TRUE(result.passed) << "Trial " << trial << ": Q5_1 auto mismatch";
    }
}

// ============================================================================
// FP32 → Q8_0 Quantize Tests
// ============================================================================

TEST_F(Q8_0DecodeVectorizationTest, FP32_ScalarVsAVX2)
{
#if defined(__AVX2__)
    for (int trial = 0; trial < 100; ++trial)
    {
        // Generate random FP32 data (full 32 elements)
        alignas(32) float fp32_vals[32];
        generate_random_fp32(fp32_vals, 32, -100.0f, 100.0f);

        // Quantize with scalar
        alignas(32) int8_t scalar_qs[32];
        uint16_t scalar_scale;
        quantize_fp32_to_q8_0_scalar(fp32_vals, 32, scalar_qs, &scalar_scale);

        // Quantize with AVX2
        alignas(32) int8_t avx2_qs[32];
        uint16_t avx2_scale;
        quantize_fp32_to_q8_0_avx2(fp32_vals, 32, avx2_qs, &avx2_scale);

        // Compare results (allow small tolerance due to rounding)
        auto result = compare_q8_0_blocks(scalar_qs, scalar_scale, avx2_qs, avx2_scale, 1);

        EXPECT_TRUE(result.passed)
            << "Trial " << trial << ": AVX2 mismatch"
            << " max_diff=" << result.max_diff
            << " mismatches=" << result.num_mismatches;

        // Scale should be nearly identical (FP16 quantization may differ slightly)
        EXPECT_NEAR(fp16_to_fp32(scalar_scale), fp16_to_fp32(avx2_scale), 0.001f)
            << "Trial " << trial << ": Scale mismatch";
    }
#else
    GTEST_SKIP() << "AVX2 not available";
#endif
}

TEST_F(Q8_0DecodeVectorizationTest, FP32_ScalarVsAVX512)
{
#if defined(__AVX512F__)
    for (int trial = 0; trial < 100; ++trial)
    {
        // Generate random FP32 data (full 32 elements)
        alignas(64) float fp32_vals[32];
        generate_random_fp32(fp32_vals, 32, -100.0f, 100.0f);

        // Quantize with scalar
        alignas(32) int8_t scalar_qs[32];
        uint16_t scalar_scale;
        quantize_fp32_to_q8_0_scalar(fp32_vals, 32, scalar_qs, &scalar_scale);

        // Quantize with AVX-512
        alignas(64) int8_t avx512_qs[32];
        uint16_t avx512_scale;
        quantize_fp32_to_q8_0_avx512(fp32_vals, 32, avx512_qs, &avx512_scale);

        // Compare results (allow small tolerance due to rounding)
        auto result = compare_q8_0_blocks(scalar_qs, scalar_scale, avx512_qs, avx512_scale, 1);

        EXPECT_TRUE(result.passed)
            << "Trial " << trial << ": AVX-512 mismatch"
            << " max_diff=" << result.max_diff
            << " mismatches=" << result.num_mismatches;

        EXPECT_NEAR(fp16_to_fp32(scalar_scale), fp16_to_fp32(avx512_scale), 0.001f)
            << "Trial " << trial << ": Scale mismatch";
    }
#else
    GTEST_SKIP() << "AVX-512 not available";
#endif
}

TEST_F(Q8_0DecodeVectorizationTest, FP32_PartialBlocks)
{
    // Test with various partial block sizes
    std::vector<size_t> sizes = {1, 7, 15, 16, 23, 31, 32};

    for (size_t count : sizes)
    {
        alignas(64) float fp32_vals[32];
        generate_random_fp32(fp32_vals, count);

        // Fill remainder with zeros
        for (size_t i = count; i < 32; ++i)
        {
            fp32_vals[i] = 0.0f;
        }

        alignas(32) int8_t scalar_qs[32];
        uint16_t scalar_scale;
        quantize_fp32_to_q8_0_scalar(fp32_vals, count, scalar_qs, &scalar_scale);

        alignas(64) int8_t auto_qs[32];
        uint16_t auto_scale;
        quantize_fp32_to_q8_0(fp32_vals, count, auto_qs, &auto_scale);

        // Compare
        auto result = compare_q8_0_blocks(scalar_qs, scalar_scale, auto_qs, auto_scale, 1);

        EXPECT_TRUE(result.passed)
            << "Count " << count << ": Mismatch in partial block";

        // Verify zero-padding
        for (size_t i = count; i < 32; ++i)
        {
            EXPECT_EQ(auto_qs[i], 0)
                << "Count " << count << ": Non-zero padding at index " << i;
        }
    }
}

TEST_F(Q8_0DecodeVectorizationTest, FP32_EdgeCases)
{
    // Test all-zero block
    {
        alignas(32) float fp32_vals[32] = {0};
        alignas(32) int8_t q8_qs[32];
        uint16_t q8_scale;

        quantize_fp32_to_q8_0(fp32_vals, 32, q8_qs, &q8_scale);

        EXPECT_FLOAT_EQ(fp16_to_fp32(q8_scale), 0.0f) << "All-zero scale should be 0";
        for (int i = 0; i < 32; ++i)
        {
            EXPECT_EQ(q8_qs[i], 0) << "All-zero block should have all 0 values";
        }
    }

    // Test very small values
    {
        alignas(32) float fp32_vals[32];
        std::fill_n(fp32_vals, 32, 1e-8f);

        alignas(32) int8_t q8_qs[32];
        uint16_t q8_scale;

        quantize_fp32_to_q8_0(fp32_vals, 32, q8_qs, &q8_scale);

        // Should quantize to ±1 or 0 depending on rounding
        for (int i = 0; i < 32; ++i)
        {
            EXPECT_LE(std::abs(q8_qs[i]), 1) << "Small values should quantize near 0";
        }
    }

    // Test large values
    {
        alignas(32) float fp32_vals[32];
        std::fill_n(fp32_vals, 32, 1000.0f);

        alignas(32) int8_t q8_qs[32];
        uint16_t q8_scale;

        quantize_fp32_to_q8_0(fp32_vals, 32, q8_qs, &q8_scale);

        // Should clamp to 127
        for (int i = 0; i < 32; ++i)
        {
            EXPECT_EQ(q8_qs[i], 127) << "Large positive values should clamp to 127";
        }
    }

    // Test mixed signs
    {
        alignas(32) float fp32_vals[32];
        for (int i = 0; i < 16; ++i)
            fp32_vals[i] = 50.0f;
        for (int i = 16; i < 32; ++i)
            fp32_vals[i] = -50.0f;

        alignas(32) int8_t q8_qs[32];
        uint16_t q8_scale;

        quantize_fp32_to_q8_0(fp32_vals, 32, q8_qs, &q8_scale);

        // First half should be positive, second half negative
        for (int i = 0; i < 16; ++i)
        {
            EXPECT_GT(q8_qs[i], 0) << "Positive values should stay positive";
        }
        for (int i = 16; i < 32; ++i)
        {
            EXPECT_LT(q8_qs[i], 0) << "Negative values should stay negative";
        }
    }
}

// ============================================================================
// Correctness: Decode + Dequantize Round-Trip
// ============================================================================

TEST_F(Q8_0DecodeVectorizationTest, IQ4NL_DequantizeRoundTrip)
{
    // Verify that IQ4_NL → Q8_0 → FP32 produces reasonable values
    // (not exact due to quantization, but should be in similar range)

    for (int trial = 0; trial < 10; ++trial)
    {
        alignas(16) uint8_t iq4_qs[16];
        uint16_t iq4_scale;
        generate_random_iq4nl(iq4_qs, &iq4_scale);

        alignas(32) int8_t q8_qs[32];
        uint16_t q8_scale;
        decode_iq4nl_to_q8_0(iq4_qs, iq4_scale, q8_qs, &q8_scale);

        // Dequantize Q8_0 back to FP32
        float scale = fp16_to_fp32(q8_scale);
        for (int i = 0; i < 32; ++i)
        {
            float dequant = q8_qs[i] * scale;

            // Should be in reasonable range (IQ4_NL values are -127 to 113)
            EXPECT_GE(dequant, -127.0f * 2.0f) << "Dequant value too negative";
            EXPECT_LE(dequant, 127.0f * 2.0f) << "Dequant value too positive";
        }
    }
}

TEST_F(Q8_0DecodeVectorizationTest, FP32_QuantizeDequantizeError)
{
    // Measure quantization error: FP32 → Q8_0 → FP32
    // Should be within expected quantization error bounds

    for (int trial = 0; trial < 10; ++trial)
    {
        alignas(32) float fp32_vals[32];
        generate_random_fp32(fp32_vals, 32, -10.0f, 10.0f);

        alignas(32) int8_t q8_qs[32];
        uint16_t q8_scale;
        quantize_fp32_to_q8_0(fp32_vals, 32, q8_qs, &q8_scale);

        // Dequantize
        float scale = fp16_to_fp32(q8_scale);
        float max_error = 0.0f;
        float avg_error = 0.0f;

        for (int i = 0; i < 32; ++i)
        {
            float dequant = q8_qs[i] * scale;
            float error = std::fabs(dequant - fp32_vals[i]);
            max_error = std::max(max_error, error);
            avg_error += error;
        }
        avg_error /= 32.0f;

        // Expected quantization error: ≈ scale (one quantization step)
        EXPECT_LT(max_error, scale * 2.0f)
            << "Trial " << trial << ": Max error exceeds 2× quantization step";

        EXPECT_LT(avg_error, scale)
            << "Trial " << trial << ": Average error exceeds quantization step";
    }
}

// ============================================================================
// Accessor Warmup Cache Tests
// ============================================================================

TEST_F(Q8_0DecodeVectorizationTest, AccessorWarmupCache_IQ4_NL)
{
    // Test that warmup_cache pre-decodes blocks correctly for IQ4_NL

    // Create a small IQ4_NL tensor (4 rows × 64 elements = 2 Q8_0 blocks per row)
    const size_t rows = 4;
    const size_t cols = 64;
    const size_t blocks_per_row = 2;

    std::vector<IQ4_NLBlock> blocks(rows * blocks_per_row);

    // Fill with random data
    for (auto &block : blocks)
    {
        generate_random_iq4nl(block.qs, &block.d);
    }

    // Create tensor
    std::vector<uint8_t> raw_data(blocks.size() * sizeof(IQ4_NLBlock));
    std::memcpy(raw_data.data(), blocks.data(), raw_data.size());

    auto tensor = std::make_shared<IQ4_NLTensor>(
        std::vector<size_t>{rows, cols},
        raw_data);

    // Create accessor with small cache
    auto accessor = createQ8_0Accessor(tensor.get(), 16);
    ASSERT_NE(accessor, nullptr);

    // Warm up cache for first 2 rows
    accessor->warmup_cache(0, 2, 0, blocks_per_row);

    // Verify cache was populated by checking that subsequent accesses
    // return the same pointer (cache hit)
    const Q8_0Block *block_0_0_first = accessor->get_q8_block(0, 0);
    const Q8_0Block *block_0_0_second = accessor->get_q8_block(0, 0);
    EXPECT_EQ(block_0_0_first, block_0_0_second) << "Cache should return same pointer";

    const Q8_0Block *block_1_1_first = accessor->get_q8_block(1, 1);
    const Q8_0Block *block_1_1_second = accessor->get_q8_block(1, 1);
    EXPECT_EQ(block_1_1_first, block_1_1_second) << "Cache should return same pointer";
}

TEST_F(Q8_0DecodeVectorizationTest, AccessorWarmupCache_Q8_0_NoOp)
{
    // Test that warmup_cache is a no-op for Q8_0 (zero-copy accessor)

    const size_t rows = 4;
    const size_t cols = 64;
    const size_t blocks_per_row = 2;

    std::vector<Q8_0Block> blocks(rows * blocks_per_row);

    // Fill with random data
    std::uniform_int_distribution<int> int8_dist(-127, 127);
    for (auto &block : blocks)
    {
        for (int i = 0; i < 32; ++i)
        {
            block.qs[i] = static_cast<int8_t>(int8_dist(rng_));
        }
        block.d = fp32_to_fp16(0.5f);
    }

    // Create tensor
    std::vector<uint8_t> raw_data(blocks.size() * sizeof(Q8_0Block));
    std::memcpy(raw_data.data(), blocks.data(), raw_data.size());

    auto tensor = std::make_shared<Q8_0Tensor>(
        std::vector<size_t>{rows, cols},
        raw_data);

    // Create accessor
    auto accessor = createQ8_0Accessor(tensor.get());
    ASSERT_NE(accessor, nullptr);
    EXPECT_TRUE(accessor->is_zero_copy()) << "Q8_0 accessor should be zero-copy";

    // Warmup should be no-op (doesn't crash)
    accessor->warmup_cache(0, rows, 0, blocks_per_row);

    // Verify direct access still works
    const Q8_0Block *block = accessor->get_q8_block(0, 0);
    ASSERT_NE(block, nullptr);
    EXPECT_EQ(block->d, blocks[0].d);
}

TEST_F(Q8_0DecodeVectorizationTest, AccessorWarmupCache_Parallel)
{
    // Test that parallel warmup doesn't cause race conditions

    const size_t rows = 64;
    const size_t cols = 256;
    const size_t blocks_per_row = 8;

    std::vector<IQ4_NLBlock> blocks(rows * blocks_per_row);

    // Fill with random data
    for (auto &block : blocks)
    {
        generate_random_iq4nl(block.qs, &block.d);
    }

    // Create tensor
    std::vector<uint8_t> raw_data(blocks.size() * sizeof(IQ4_NLBlock));
    std::memcpy(raw_data.data(), blocks.data(), raw_data.size());

    auto tensor = std::make_shared<IQ4_NLTensor>(
        std::vector<size_t>{rows, cols},
        raw_data);

    // Create accessor
    auto accessor = createQ8_0Accessor(tensor.get(), 512);
    ASSERT_NE(accessor, nullptr);

    // Warm up entire cache (should trigger OpenMP parallelization)
    accessor->warmup_cache(0, rows, 0, blocks_per_row);

    // Verify all blocks are cached correctly
    for (size_t r = 0; r < rows; ++r)
    {
        for (size_t kb = 0; kb < blocks_per_row; ++kb)
        {
            const Q8_0Block *cached_block = accessor->get_q8_block(r, kb);
            ASSERT_NE(cached_block, nullptr) << "Block (" << r << ", " << kb << ") should be cached";

            // Verify it's the same pointer (cache hit)
            const Q8_0Block *second_access = accessor->get_q8_block(r, kb);
            EXPECT_EQ(cached_block, second_access) << "Cache should return same pointer for (" << r << ", " << kb << ")";
        }
    }
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
