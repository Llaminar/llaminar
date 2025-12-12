/**
 * @file Test__Q8DotProduct.cpp
 * @brief Unit tests for Q8_1 dot product microkernel
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include "kernels/cpu/microkernels/q8_1/Q8DotProduct.h"
#include "tensors/FP16Utils.h"

#include <cmath>
#include <random>
#include <vector>

using namespace llaminar::v2::kernels::microkernels;

class Test__Q8DotProduct : public ::testing::Test
{
protected:
    std::mt19937 rng_{42}; // Fixed seed for reproducibility

    // Helper to create a Q8_1 block with known values
    Q8_1Block createBlock(float scale, const std::vector<int8_t> &values)
    {
        Q8_1Block block;
        block.d = llaminar2::fp32_to_fp16(scale);

        int16_t sum = 0;
        for (int i = 0; i < 32; ++i)
        {
            int8_t v = (i < static_cast<int>(values.size())) ? values[i] : 0;
            block.qs[i] = v;
            sum += v;
        }
        block.sum_qs = sum;
        return block;
    }

    // Helper to create a Q8_1 block with random values
    Q8_1Block createRandomBlock(float scale)
    {
        std::uniform_int_distribution<int> dist(-127, 127);

        Q8_1Block block;
        block.d = llaminar2::fp32_to_fp16(scale);

        int16_t sum = 0;
        for (int i = 0; i < 32; ++i)
        {
            int8_t v = static_cast<int8_t>(dist(rng_));
            block.qs[i] = v;
            sum += v;
        }
        block.sum_qs = sum;
        return block;
    }

    // Helper to compute expected dot product in FP32 (ground truth)
    float computeExpectedDot(const Q8_1Block *q_blocks, const Q8_1Block *k_blocks,
                             int num_blocks, float global_scale)
    {
        float total = 0.0f;
        for (int b = 0; b < num_blocks; ++b)
        {
            float d_q = llaminar2::fp16_to_fp32(q_blocks[b].d);
            float d_k = llaminar2::fp16_to_fp32(k_blocks[b].d);

            float block_dot = 0.0f;
            for (int i = 0; i < 32; ++i)
            {
                block_dot += static_cast<float>(q_blocks[b].qs[i]) *
                             static_cast<float>(k_blocks[b].qs[i]);
            }
            total += block_dot * d_q * d_k;
        }
        return total * global_scale;
    }
};

TEST_F(Test__Q8DotProduct, ZeroVectors)
{
    // Two zero vectors should produce 0
    std::vector<int8_t> zeros(32, 0);
    Q8_1Block q = createBlock(1.0f, zeros);
    Q8_1Block k = createBlock(1.0f, zeros);

    Q8DotProductParams params = {&q, &k, 1, 1.0f};
    auto result = q8_dot_product_ref(params);

    EXPECT_FLOAT_EQ(result.score, 0.0f);
}

TEST_F(Test__Q8DotProduct, IdentityVector_SingleBlock)
{
    // Same vector dotted with itself should produce ||v||²
    std::vector<int8_t> values = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                  11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                  21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32};
    Q8_1Block v = createBlock(1.0f, values);

    Q8DotProductParams params = {&v, &v, 1, 1.0f};
    auto result = q8_dot_product_ref(params);

    // Expected: sum of squares = 1² + 2² + ... + 32² = 32*33*65/6 = 11440
    float expected = 0.0f;
    for (int i = 1; i <= 32; ++i)
    {
        expected += i * i;
    }

    EXPECT_NEAR(result.score, expected, 1e-3f);
}

TEST_F(Test__Q8DotProduct, OrthogonalVectors)
{
    // Vectors where sum(q*k) = 0
    std::vector<int8_t> q_vals(32, 0);
    std::vector<int8_t> k_vals(32, 0);

    // q = [1, -1, 0, 0, ...]
    // k = [1, 1, 0, 0, ...]
    // dot = 1*1 + (-1)*1 = 0
    q_vals[0] = 1;
    q_vals[1] = -1;
    k_vals[0] = 1;
    k_vals[1] = 1;

    Q8_1Block q = createBlock(1.0f, q_vals);
    Q8_1Block k = createBlock(1.0f, k_vals);

    Q8DotProductParams params = {&q, &k, 1, 1.0f};
    auto result = q8_dot_product_ref(params);

    EXPECT_NEAR(result.score, 0.0f, 1e-6f);
}

TEST_F(Test__Q8DotProduct, ScaleFactors)
{
    // Test that scale factors are applied correctly
    std::vector<int8_t> ones(32, 1);
    Q8_1Block q = createBlock(2.0f, ones); // scale = 2
    Q8_1Block k = createBlock(3.0f, ones); // scale = 3

    Q8DotProductParams params = {&q, &k, 1, 1.0f};
    auto result = q8_dot_product_ref(params);

    // Each element is 1*1 = 1, 32 elements, scales 2*3 = 6
    float expected = 32.0f * 2.0f * 3.0f;
    EXPECT_NEAR(result.score, expected, 1e-3f);
}

TEST_F(Test__Q8DotProduct, GlobalScale)
{
    // Test that global scale (1/sqrt(d)) is applied
    std::vector<int8_t> ones(32, 1);
    Q8_1Block q = createBlock(1.0f, ones);
    Q8_1Block k = createBlock(1.0f, ones);

    float global_scale = 0.125f; // 1/8
    Q8DotProductParams params = {&q, &k, 1, global_scale};
    auto result = q8_dot_product_ref(params);

    // 32 * 1 * 1 * 0.125 = 4
    EXPECT_NEAR(result.score, 4.0f, 1e-5f);
}

TEST_F(Test__Q8DotProduct, MultipleBlocks)
{
    // Test with multiple Q8_1 blocks (head_dim = 64 → 2 blocks)
    Q8_1Block q_blocks[2];
    Q8_1Block k_blocks[2];

    std::vector<int8_t> vals1(32, 2);
    std::vector<int8_t> vals2(32, 3);

    q_blocks[0] = createBlock(1.0f, vals1);
    q_blocks[1] = createBlock(1.0f, vals2);
    k_blocks[0] = createBlock(1.0f, vals1);
    k_blocks[1] = createBlock(1.0f, vals2);

    Q8DotProductParams params = {q_blocks, k_blocks, 2, 1.0f};
    auto result = q8_dot_product_ref(params);

    // Block 0: 32 * 2 * 2 = 128
    // Block 1: 32 * 3 * 3 = 288
    // Total: 416
    EXPECT_NEAR(result.score, 416.0f, 1e-3f);
}

TEST_F(Test__Q8DotProduct, RandomVectors_MatchesFP32)
{
    // Test random vectors against FP32 ground truth
    const int num_blocks = 2; // head_dim = 64

    Q8_1Block q_blocks[2];
    Q8_1Block k_blocks[2];

    for (int b = 0; b < num_blocks; ++b)
    {
        std::uniform_real_distribution<float> scale_dist(0.01f, 0.1f);
        q_blocks[b] = createRandomBlock(scale_dist(rng_));
        k_blocks[b] = createRandomBlock(scale_dist(rng_));
    }

    float global_scale = 1.0f / std::sqrt(64.0f); // 1/sqrt(head_dim)

    Q8DotProductParams params = {q_blocks, k_blocks, num_blocks, global_scale};
    auto result = q8_dot_product_ref(params);

    float expected = computeExpectedDot(q_blocks, k_blocks, num_blocks, global_scale);

    // Allow small tolerance due to FP16 scale factors
    EXPECT_NEAR(result.score, expected, std::abs(expected) * 0.01f + 1e-5f);
}

TEST_F(Test__Q8DotProduct, EdgeCase_MaxValues)
{
    // Test with maximum int8 values
    std::vector<int8_t> max_vals(32, 127);
    Q8_1Block q = createBlock(1.0f, max_vals);
    Q8_1Block k = createBlock(1.0f, max_vals);

    Q8DotProductParams params = {&q, &k, 1, 1.0f};
    auto result = q8_dot_product_ref(params);

    // 32 * 127 * 127 = 516128
    float expected = 32.0f * 127.0f * 127.0f;
    EXPECT_NEAR(result.score, expected, 1.0f);
}

TEST_F(Test__Q8DotProduct, EdgeCase_MinValues)
{
    // Test with minimum int8 values (-127, not -128 to stay in valid Q8_1 range)
    std::vector<int8_t> min_vals(32, -127);
    Q8_1Block q = createBlock(1.0f, min_vals);
    Q8_1Block k = createBlock(1.0f, min_vals);

    Q8DotProductParams params = {&q, &k, 1, 1.0f};
    auto result = q8_dot_product_ref(params);

    // 32 * (-127) * (-127) = 516128
    float expected = 32.0f * 127.0f * 127.0f;
    EXPECT_NEAR(result.score, expected, 1.0f);
}

TEST_F(Test__Q8DotProduct, AVX512_MatchesReference)
{
    // Test that AVX-512 implementation matches reference
    const int num_blocks = 4; // head_dim = 128

    Q8_1Block q_blocks[4];
    Q8_1Block k_blocks[4];

    for (int b = 0; b < num_blocks; ++b)
    {
        std::uniform_real_distribution<float> scale_dist(0.01f, 0.1f);
        q_blocks[b] = createRandomBlock(scale_dist(rng_));
        k_blocks[b] = createRandomBlock(scale_dist(rng_));
    }

    float global_scale = 1.0f / std::sqrt(128.0f);

    Q8DotProductParams params = {q_blocks, k_blocks, num_blocks, global_scale};

    auto ref_result = q8_dot_product_ref(params);
    auto avx_result = q8_dot_product_avx512(params);

    // Should match within floating point tolerance
    EXPECT_NEAR(avx_result.score, ref_result.score, std::abs(ref_result.score) * 0.001f + 1e-5f);
}
