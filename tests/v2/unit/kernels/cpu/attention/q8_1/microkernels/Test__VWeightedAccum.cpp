/**
 * @file Test__VWeightedAccum.cpp
 * @brief Unit tests for weighted V accumulation microkernel
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include "kernels/cpu/attention/q8_1/microkernels/VWeightedAccum.h"
#include "tensors/FP16Utils.h"

#include <cmath>
#include <random>
#include <vector>

using namespace llaminar::v2::kernels::microkernels;

class Test__VWeightedAccum : public ::testing::Test
{
protected:
    std::mt19937 rng_{42};

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

    // Dequantize V block to FP32 for reference
    void dequantize(const Q8_1Block &block, float *out)
    {
        float scale = llaminar2::fp16_to_fp32(block.d);
        for (int i = 0; i < 32; ++i)
        {
            out[i] = static_cast<float>(block.qs[i]) * scale;
        }
    }
};

TEST_F(Test__VWeightedAccum, ZeroWeight)
{
    // Zero weight should not change context (except correction)
    Q8_1Block v = createBlock(1.0f, std::vector<int8_t>(32, 100));

    std::vector<float> context(32, 5.0f); // Initial values

    VWeightedAccumParams params = {
        .v_blocks = &v,
        .weight = 0.0f,
        .correction = 1.0f, // No correction
        .context = context.data(),
        .num_blocks = 1};

    v_weighted_accum_ref(params);

    // Context should be unchanged
    for (int i = 0; i < 32; ++i)
    {
        EXPECT_FLOAT_EQ(context[i], 5.0f);
    }
}

TEST_F(Test__VWeightedAccum, UnitWeight_ZeroContext)
{
    // Unit weight + zero context = V
    std::vector<int8_t> values(32);
    for (int i = 0; i < 32; ++i)
        values[i] = static_cast<int8_t>(i - 16);

    Q8_1Block v = createBlock(0.1f, values);

    std::vector<float> context(32, 0.0f);

    VWeightedAccumParams params = {
        .v_blocks = &v,
        .weight = 1.0f,
        .correction = 1.0f,
        .context = context.data(),
        .num_blocks = 1};

    v_weighted_accum_ref(params);

    // context = 0 + 1.0 * V = V
    // Note: FP16 scale has ~0.1% precision loss, so use relaxed tolerance
    float scale = llaminar2::fp16_to_fp32(v.d); // Get actual FP16-converted scale
    for (int i = 0; i < 32; ++i)
    {
        float expected = static_cast<float>(i - 16) * scale;
        EXPECT_NEAR(context[i], expected, std::abs(expected) * 0.01f + 1e-5f);
    }
}

TEST_F(Test__VWeightedAccum, CorrectionFactor)
{
    // Test that correction factor is applied
    std::vector<int8_t> zeros(32, 0);
    Q8_1Block v = createBlock(1.0f, zeros); // Zero V

    std::vector<float> context(32, 10.0f); // Initial context

    VWeightedAccumParams params = {
        .v_blocks = &v,
        .weight = 0.0f,
        .correction = 0.5f, // Should halve context
        .context = context.data(),
        .num_blocks = 1};

    v_weighted_accum_ref(params);

    // Context should be halved
    for (int i = 0; i < 32; ++i)
    {
        EXPECT_NEAR(context[i], 5.0f, 1e-6f);
    }
}

TEST_F(Test__VWeightedAccum, WeightedAccumulation)
{
    // Test weighted accumulation: context = context * correction + weight * V
    std::vector<int8_t> v_vals(32, 20);      // All 20s
    Q8_1Block v = createBlock(0.5f, v_vals); // scale = 0.5, so dequant = 10.0

    std::vector<float> context(32, 8.0f); // Initial context

    VWeightedAccumParams params = {
        .v_blocks = &v,
        .weight = 0.3f,
        .correction = 0.7f,
        .context = context.data(),
        .num_blocks = 1};

    v_weighted_accum_ref(params);

    // context = 8.0 * 0.7 + 0.3 * (20 * 0.5) = 5.6 + 3.0 = 8.6
    for (int i = 0; i < 32; ++i)
    {
        EXPECT_NEAR(context[i], 8.6f, 1e-5f);
    }
}

TEST_F(Test__VWeightedAccum, MultipleBlocks)
{
    // Test with head_dim = 64 (2 blocks)
    Q8_1Block v_blocks[2];

    std::vector<int8_t> vals1(32, 10);
    std::vector<int8_t> vals2(32, -10);
    v_blocks[0] = createBlock(1.0f, vals1);
    v_blocks[1] = createBlock(1.0f, vals2);

    std::vector<float> context(64, 0.0f);

    VWeightedAccumParams params = {
        .v_blocks = v_blocks,
        .weight = 1.0f,
        .correction = 1.0f,
        .context = context.data(),
        .num_blocks = 2};

    v_weighted_accum_ref(params);

    // First 32: 10.0, Second 32: -10.0
    for (int i = 0; i < 32; ++i)
    {
        EXPECT_NEAR(context[i], 10.0f, 1e-5f);
    }
    for (int i = 32; i < 64; ++i)
    {
        EXPECT_NEAR(context[i], -10.0f, 1e-5f);
    }
}

TEST_F(Test__VWeightedAccum, MultipleAccumulations)
{
    // Simulate attention: accumulate multiple weighted V vectors
    const int seq_len = 4;
    const int head_dim = 32;

    // Create V blocks for 4 positions
    std::vector<Q8_1Block> V(seq_len);
    for (int n = 0; n < seq_len; ++n)
    {
        std::vector<int8_t> vals(32, static_cast<int8_t>(n * 10)); // 0, 10, 20, 30
        V[n] = createBlock(1.0f, vals);
    }

    // Softmax weights (sum to 1)
    std::vector<float> weights = {0.1f, 0.2f, 0.3f, 0.4f};

    std::vector<float> context(head_dim, 0.0f);

    for (int n = 0; n < seq_len; ++n)
    {
        VWeightedAccumParams params = {
            .v_blocks = &V[n],
            .weight = weights[n],
            .correction = 1.0f,
            .context = context.data(),
            .num_blocks = 1};
        v_weighted_accum_ref(params);
    }

    // Expected: 0.1*0 + 0.2*10 + 0.3*20 + 0.4*30 = 0 + 2 + 6 + 12 = 20
    for (int i = 0; i < head_dim; ++i)
    {
        EXPECT_NEAR(context[i], 20.0f, 1e-5f);
    }
}

TEST_F(Test__VWeightedAccum, AVX512_MatchesReference)
{
    // Test that AVX-512 matches reference implementation
    const int num_blocks = 4; // head_dim = 128
    const int head_dim = num_blocks * 32;

    std::vector<Q8_1Block> v_blocks(num_blocks);
    for (int b = 0; b < num_blocks; ++b)
    {
        std::uniform_real_distribution<float> scale_dist(0.01f, 0.1f);
        v_blocks[b] = createRandomBlock(scale_dist(rng_));
    }

    // Test both implementations
    std::vector<float> context_ref(head_dim);
    std::vector<float> context_avx(head_dim);

    // Fill with same random initial values
    std::uniform_real_distribution<float> init_dist(-10.0f, 10.0f);
    for (int i = 0; i < head_dim; ++i)
    {
        float v = init_dist(rng_);
        context_ref[i] = v;
        context_avx[i] = v;
    }

    float weight = 0.35f;
    float correction = 0.85f;

    VWeightedAccumParams params_ref = {
        .v_blocks = v_blocks.data(),
        .weight = weight,
        .correction = correction,
        .context = context_ref.data(),
        .num_blocks = num_blocks};

    VWeightedAccumParams params_avx = {
        .v_blocks = v_blocks.data(),
        .weight = weight,
        .correction = correction,
        .context = context_avx.data(),
        .num_blocks = num_blocks};

    v_weighted_accum_ref(params_ref);
    v_weighted_accum_avx512(params_avx);

    for (int i = 0; i < head_dim; ++i)
    {
        EXPECT_NEAR(context_avx[i], context_ref[i], 1e-4f)
            << "Mismatch at index " << i;
    }
}

TEST_F(Test__VWeightedAccum, CorrectionOnly)
{
    // Test apply_softmax_correction functions
    const int head_dim = 64;

    std::vector<float> context_ref(head_dim);
    std::vector<float> context_avx(head_dim);

    std::uniform_real_distribution<float> dist(-100.0f, 100.0f);
    for (int i = 0; i < head_dim; ++i)
    {
        float v = dist(rng_);
        context_ref[i] = v;
        context_avx[i] = v;
    }

    float correction = 0.42f;

    apply_softmax_correction_ref(context_ref.data(), correction, head_dim);
    apply_softmax_correction_avx512(context_avx.data(), correction, head_dim);

    for (int i = 0; i < head_dim; ++i)
    {
        EXPECT_NEAR(context_avx[i], context_ref[i], 1e-5f);
    }
}
