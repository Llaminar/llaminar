/**
 * @file Test__WoProjection.cpp
 * @brief Unit tests for Wo output projection microkernel
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include "kernels/cpu/attention/q8_1/microkernels/WoProjection.h"
#include "tensors/FP16Utils.h"
#include "tensors/SIMDHelpers.h"

#include <cmath>
#include <random>
#include <vector>

using namespace llaminar::v2::kernels::microkernels;

class Test__WoProjection : public ::testing::Test
{
protected:
    std::mt19937 rng_{42};

    // Create FP32 Wo weights with known pattern
    std::vector<float> createIdentityLikeWo(int d_model, int n_heads, int head_dim)
    {
        // Each row of Wo is mostly zeros except for a specific head slice
        // This creates a "routing" pattern for testing
        std::vector<float> wo(d_model * n_heads * head_dim, 0.0f);

        // Simple pattern: output[i] = input[i % head_dim] for the first head
        for (int o = 0; o < d_model; ++o)
        {
            int src_idx = o % head_dim;
            wo[o * n_heads * head_dim + src_idx] = 1.0f;
        }

        return wo;
    }

    // Create random FP32 Wo weights
    std::vector<float> createRandomWo(int d_model, int n_heads, int head_dim)
    {
        std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
        std::vector<float> wo(d_model * n_heads * head_dim);
        for (auto &w : wo)
        {
            w = dist(rng_);
        }
        return wo;
    }

    // Reference implementation: naive GEMV
    void referenceGemv(const float *context, const float *wo, float *output,
                       int d_model, int n_heads, int head_dim, int head_idx,
                       bool accumulate)
    {
        int head_offset = head_idx * head_dim;
        int wo_row_stride = n_heads * head_dim;

        for (int o = 0; o < d_model; ++o)
        {
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d)
            {
                dot += context[d] * wo[o * wo_row_stride + head_offset + d];
            }
            if (accumulate)
            {
                output[o] += dot;
            }
            else
            {
                output[o] = dot;
            }
        }
    }
};

TEST_F(Test__WoProjection, ZeroContext)
{
    const int d_model = 64;
    const int n_heads = 4;
    const int head_dim = 16;

    std::vector<float> context(head_dim, 0.0f);
    std::vector<float> wo = createRandomWo(d_model, n_heads, head_dim);
    std::vector<float> output(d_model, 999.0f); // Initialize to non-zero

    WoProjectionParams params = {
        .context = context.data(),
        .wo_weights = wo.data(),
        .wo_type = WoWeightType::FP32,
        .head_dim = head_dim,
        .d_model = d_model,
        .head_idx = 0,
        .n_heads = n_heads,
        .output = output.data(),
        .accumulate = false};

    wo_projection_fp32_ref(params);

    // Zero context should produce zero output
    for (int i = 0; i < d_model; ++i)
    {
        EXPECT_FLOAT_EQ(output[i], 0.0f);
    }
}

TEST_F(Test__WoProjection, IdentityContext_SingleHead)
{
    const int d_model = 32;
    const int n_heads = 2;
    const int head_dim = 16;

    // Context = [1, 0, 0, ..., 0]
    std::vector<float> context(head_dim, 0.0f);
    context[0] = 1.0f;

    // Wo where each row's first element of head 0 slice is the output index
    std::vector<float> wo(d_model * n_heads * head_dim, 0.0f);
    for (int o = 0; o < d_model; ++o)
    {
        wo[o * n_heads * head_dim + 0] = static_cast<float>(o); // First element of head 0
    }

    std::vector<float> output(d_model);

    WoProjectionParams params = {
        .context = context.data(),
        .wo_weights = wo.data(),
        .wo_type = WoWeightType::FP32,
        .head_dim = head_dim,
        .d_model = d_model,
        .head_idx = 0,
        .n_heads = n_heads,
        .output = output.data(),
        .accumulate = false};

    wo_projection_fp32_ref(params);

    // output[o] = context[0] * wo[o, 0] = 1.0 * o = o
    for (int o = 0; o < d_model; ++o)
    {
        EXPECT_FLOAT_EQ(output[o], static_cast<float>(o));
    }
}

TEST_F(Test__WoProjection, DifferentHeads)
{
    const int d_model = 16;
    const int n_heads = 4;
    const int head_dim = 8;

    // Context = all ones
    std::vector<float> context(head_dim, 1.0f);

    // Wo: each head has a different constant value
    std::vector<float> wo(d_model * n_heads * head_dim, 0.0f);
    for (int h = 0; h < n_heads; ++h)
    {
        float head_val = static_cast<float>(h + 1); // 1, 2, 3, 4
        for (int o = 0; o < d_model; ++o)
        {
            for (int d = 0; d < head_dim; ++d)
            {
                wo[o * n_heads * head_dim + h * head_dim + d] = head_val;
            }
        }
    }

    // Test each head
    for (int h = 0; h < n_heads; ++h)
    {
        std::vector<float> output(d_model);

        WoProjectionParams params = {
            .context = context.data(),
            .wo_weights = wo.data(),
            .wo_type = WoWeightType::FP32,
            .head_dim = head_dim,
            .d_model = d_model,
            .head_idx = h,
            .n_heads = n_heads,
            .output = output.data(),
            .accumulate = false};

        wo_projection_fp32_ref(params);

        // sum(context) * head_val = head_dim * (h+1)
        float expected = static_cast<float>(head_dim * (h + 1));
        for (int o = 0; o < d_model; ++o)
        {
            EXPECT_FLOAT_EQ(output[o], expected)
                << "Failed for head " << h << ", output " << o;
        }
    }
}

TEST_F(Test__WoProjection, Accumulation)
{
    const int d_model = 16;
    const int n_heads = 2;
    const int head_dim = 8;

    std::vector<float> context(head_dim, 1.0f);
    std::vector<float> wo(d_model * n_heads * head_dim, 1.0f); // All ones
    std::vector<float> output(d_model, 10.0f);                 // Pre-initialize

    WoProjectionParams params = {
        .context = context.data(),
        .wo_weights = wo.data(),
        .wo_type = WoWeightType::FP32,
        .head_dim = head_dim,
        .d_model = d_model,
        .head_idx = 0,
        .n_heads = n_heads,
        .output = output.data(),
        .accumulate = true // Add to existing
    };

    wo_projection_fp32_ref(params);

    // output = 10 + head_dim * 1.0 = 10 + 8 = 18
    for (int o = 0; o < d_model; ++o)
    {
        EXPECT_FLOAT_EQ(output[o], 18.0f);
    }
}

TEST_F(Test__WoProjection, MultiHeadAccumulation)
{
    // Simulate full attention output: accumulate all heads
    const int d_model = 32;
    const int n_heads = 4;
    const int head_dim = 8;

    std::vector<float> wo = createRandomWo(d_model, n_heads, head_dim);
    std::vector<float> output(d_model, 0.0f);

    // Create different context for each head
    std::vector<std::vector<float>> contexts(n_heads);
    for (int h = 0; h < n_heads; ++h)
    {
        contexts[h].resize(head_dim);
        for (int d = 0; d < head_dim; ++d)
        {
            contexts[h][d] = static_cast<float>(h * head_dim + d);
        }
    }

    // Accumulate all heads
    for (int h = 0; h < n_heads; ++h)
    {
        WoProjectionParams params = {
            .context = contexts[h].data(),
            .wo_weights = wo.data(),
            .wo_type = WoWeightType::FP32,
            .head_dim = head_dim,
            .d_model = d_model,
            .head_idx = h,
            .n_heads = n_heads,
            .output = output.data(),
            .accumulate = (h > 0) // First head overwrites, rest accumulate
        };

        wo_projection_fp32_ref(params);
    }

    // Compute reference
    std::vector<float> ref_output(d_model, 0.0f);
    for (int h = 0; h < n_heads; ++h)
    {
        referenceGemv(contexts[h].data(), wo.data(), ref_output.data(),
                      d_model, n_heads, head_dim, h, true);
    }

    for (int o = 0; o < d_model; ++o)
    {
        EXPECT_NEAR(output[o], ref_output[o], 1e-4f);
    }
}

TEST_F(Test__WoProjection, RandomValues_MatchesReference)
{
    const int d_model = 128;
    const int n_heads = 8;
    const int head_dim = 16;

    // Random context
    std::vector<float> context(head_dim);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto &c : context)
        c = dist(rng_);

    std::vector<float> wo = createRandomWo(d_model, n_heads, head_dim);

    for (int h = 0; h < n_heads; ++h)
    {
        std::vector<float> output(d_model);
        std::vector<float> ref_output(d_model);

        WoProjectionParams params = {
            .context = context.data(),
            .wo_weights = wo.data(),
            .wo_type = WoWeightType::FP32,
            .head_dim = head_dim,
            .d_model = d_model,
            .head_idx = h,
            .n_heads = n_heads,
            .output = output.data(),
            .accumulate = false};

        wo_projection_fp32_ref(params);
        referenceGemv(context.data(), wo.data(), ref_output.data(),
                      d_model, n_heads, head_dim, h, false);

        for (int o = 0; o < d_model; ++o)
        {
            EXPECT_NEAR(output[o], ref_output[o], 1e-5f)
                << "Mismatch at head " << h << ", output " << o;
        }
    }
}

TEST_F(Test__WoProjection, AVX512_MatchesReference)
{
    const int d_model = 256;
    const int n_heads = 8;
    const int head_dim = 32;

    std::vector<float> context(head_dim);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto &c : context)
        c = dist(rng_);

    std::vector<float> wo = createRandomWo(d_model, n_heads, head_dim);

    std::vector<float> output_ref(d_model);
    std::vector<float> output_avx(d_model);

    int h = 3; // Test arbitrary head

    WoProjectionParams params_ref = {
        .context = context.data(),
        .wo_weights = wo.data(),
        .wo_type = WoWeightType::FP32,
        .head_dim = head_dim,
        .d_model = d_model,
        .head_idx = h,
        .n_heads = n_heads,
        .output = output_ref.data(),
        .accumulate = false};

    WoProjectionParams params_avx = {
        .context = context.data(),
        .wo_weights = wo.data(),
        .wo_type = WoWeightType::FP32,
        .head_dim = head_dim,
        .d_model = d_model,
        .head_idx = h,
        .n_heads = n_heads,
        .output = output_avx.data(),
        .accumulate = false};

    wo_projection_fp32_ref(params_ref);
    wo_projection_fp32_avx512(params_avx);

    for (int o = 0; o < d_model; ++o)
    {
        EXPECT_NEAR(output_avx[o], output_ref[o], 1e-4f)
            << "Mismatch at output " << o;
    }
}

TEST_F(Test__WoProjection, Qwen2Dimensions)
{
    // Test with actual Qwen2.5-0.5B dimensions
    const int d_model = 896;
    const int n_heads = 14;
    const int head_dim = 64;

    std::vector<float> context(head_dim);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    for (auto &c : context)
        c = dist(rng_);

    std::vector<float> wo = createRandomWo(d_model, n_heads, head_dim);
    std::vector<float> output(d_model);

    WoProjectionParams params = {
        .context = context.data(),
        .wo_weights = wo.data(),
        .wo_type = WoWeightType::FP32,
        .head_dim = head_dim,
        .d_model = d_model,
        .head_idx = 7, // Middle head
        .n_heads = n_heads,
        .output = output.data(),
        .accumulate = false};

    // Should complete without error
    wo_projection_fp32_ref(params);

    // Verify output is not all zeros (sanity check)
    float sum = 0.0f;
    for (float o : output)
        sum += std::abs(o);
    EXPECT_GT(sum, 0.0f);
}

// ============================================================================
// FP16 Wo Weight Tests
// ============================================================================

TEST_F(Test__WoProjection, FP16_Basic)
{
    const int d_model = 32;
    const int n_heads = 2;
    const int head_dim = 16;

    // Create FP32 reference weights
    std::vector<float> wo_fp32 = createRandomWo(d_model, n_heads, head_dim);

    // Convert to FP16
    std::vector<uint16_t> wo_fp16(wo_fp32.size());
    for (size_t i = 0; i < wo_fp32.size(); ++i)
    {
        wo_fp16[i] = llaminar2::fp32_to_fp16(wo_fp32[i]);
    }

    std::vector<float> context(head_dim);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto &c : context)
        c = dist(rng_);

    std::vector<float> output_fp32(d_model);
    std::vector<float> output_fp16(d_model);

    // Run FP32 reference
    WoProjectionParams params_fp32 = {
        .context = context.data(),
        .wo_weights = wo_fp32.data(),
        .wo_type = WoWeightType::FP32,
        .head_dim = head_dim,
        .d_model = d_model,
        .head_idx = 0,
        .n_heads = n_heads,
        .output = output_fp32.data(),
        .accumulate = false};
    wo_projection_fp32_ref(params_fp32);

    // Run FP16 version
    WoProjectionParams params_fp16 = {
        .context = context.data(),
        .wo_weights = wo_fp16.data(),
        .wo_type = WoWeightType::FP16,
        .head_dim = head_dim,
        .d_model = d_model,
        .head_idx = 0,
        .n_heads = n_heads,
        .output = output_fp16.data(),
        .accumulate = false};
    wo_projection_fp16_ref(params_fp16);

    // Compare with FP16 tolerance (about 0.1% relative error typical)
    for (int o = 0; o < d_model; ++o)
    {
        EXPECT_NEAR(output_fp16[o], output_fp32[o], std::abs(output_fp32[o]) * 0.05f + 1e-4f)
            << "Mismatch at output " << o;
    }
}

TEST_F(Test__WoProjection, FP16_AllHeads)
{
    const int d_model = 64;
    const int n_heads = 4;
    const int head_dim = 16;

    std::vector<float> wo_fp32 = createRandomWo(d_model, n_heads, head_dim);
    std::vector<uint16_t> wo_fp16(wo_fp32.size());
    for (size_t i = 0; i < wo_fp32.size(); ++i)
    {
        wo_fp16[i] = llaminar2::fp32_to_fp16(wo_fp32[i]);
    }

    std::vector<float> context(head_dim);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto &c : context)
        c = dist(rng_);

    for (int h = 0; h < n_heads; ++h)
    {
        std::vector<float> output_fp32(d_model);
        std::vector<float> output_fp16(d_model);

        WoProjectionParams params_fp32 = {
            .context = context.data(),
            .wo_weights = wo_fp32.data(),
            .wo_type = WoWeightType::FP32,
            .head_dim = head_dim,
            .d_model = d_model,
            .head_idx = h,
            .n_heads = n_heads,
            .output = output_fp32.data(),
            .accumulate = false};
        wo_projection_fp32_ref(params_fp32);

        WoProjectionParams params_fp16 = {
            .context = context.data(),
            .wo_weights = wo_fp16.data(),
            .wo_type = WoWeightType::FP16,
            .head_dim = head_dim,
            .d_model = d_model,
            .head_idx = h,
            .n_heads = n_heads,
            .output = output_fp16.data(),
            .accumulate = false};
        wo_projection_fp16_ref(params_fp16);

        for (int o = 0; o < d_model; ++o)
        {
            EXPECT_NEAR(output_fp16[o], output_fp32[o], std::abs(output_fp32[o]) * 0.05f + 1e-4f)
                << "Mismatch at head " << h << ", output " << o;
        }
    }
}

TEST_F(Test__WoProjection, FP16_Accumulation)
{
    const int d_model = 32;
    const int n_heads = 2;
    const int head_dim = 16;

    std::vector<float> wo_fp32 = createRandomWo(d_model, n_heads, head_dim);
    std::vector<uint16_t> wo_fp16(wo_fp32.size());
    for (size_t i = 0; i < wo_fp32.size(); ++i)
    {
        wo_fp16[i] = llaminar2::fp32_to_fp16(wo_fp32[i]);
    }

    std::vector<float> context(head_dim, 1.0f);
    std::vector<float> output(d_model, 5.0f);

    WoProjectionParams params = {
        .context = context.data(),
        .wo_weights = wo_fp16.data(),
        .wo_type = WoWeightType::FP16,
        .head_dim = head_dim,
        .d_model = d_model,
        .head_idx = 0,
        .n_heads = n_heads,
        .output = output.data(),
        .accumulate = true};
    wo_projection_fp16_ref(params);

    // Verify accumulation happened (values should not be exactly 5.0)
    bool any_changed = false;
    for (int o = 0; o < d_model; ++o)
    {
        if (std::abs(output[o] - 5.0f) > 1e-6f)
        {
            any_changed = true;
            break;
        }
    }
    EXPECT_TRUE(any_changed);
}

TEST_F(Test__WoProjection, FP16_Dispatch)
{
    const int d_model = 32;
    const int n_heads = 2;
    const int head_dim = 16;

    std::vector<float> wo_fp32 = createRandomWo(d_model, n_heads, head_dim);
    std::vector<uint16_t> wo_fp16(wo_fp32.size());
    for (size_t i = 0; i < wo_fp32.size(); ++i)
    {
        wo_fp16[i] = llaminar2::fp32_to_fp16(wo_fp32[i]);
    }

    std::vector<float> context(head_dim);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto &c : context)
        c = dist(rng_);

    std::vector<float> output_direct(d_model);
    std::vector<float> output_dispatch(d_model);

    // Direct call
    WoProjectionParams params1 = {
        .context = context.data(),
        .wo_weights = wo_fp16.data(),
        .wo_type = WoWeightType::FP16,
        .head_dim = head_dim,
        .d_model = d_model,
        .head_idx = 0,
        .n_heads = n_heads,
        .output = output_direct.data(),
        .accumulate = false};
    wo_projection_fp16_ref(params1);

    // Dispatch call
    WoProjectionParams params2 = {
        .context = context.data(),
        .wo_weights = wo_fp16.data(),
        .wo_type = WoWeightType::FP16,
        .head_dim = head_dim,
        .d_model = d_model,
        .head_idx = 0,
        .n_heads = n_heads,
        .output = output_dispatch.data(),
        .accumulate = false};
    wo_projection(params2);

    for (int o = 0; o < d_model; ++o)
    {
        EXPECT_FLOAT_EQ(output_dispatch[o], output_direct[o]);
    }
}

// ============================================================================
// BF16 Wo Weight Tests
// ============================================================================

TEST_F(Test__WoProjection, BF16_Basic)
{
    const int d_model = 32;
    const int n_heads = 2;
    const int head_dim = 16;

    // Create FP32 reference weights
    std::vector<float> wo_fp32 = createRandomWo(d_model, n_heads, head_dim);

    // Convert to BF16
    std::vector<uint16_t> wo_bf16(wo_fp32.size());
    for (size_t i = 0; i < wo_fp32.size(); ++i)
    {
        wo_bf16[i] = llaminar2::simd::fp32_to_bf16(wo_fp32[i]);
    }

    std::vector<float> context(head_dim);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto &c : context)
        c = dist(rng_);

    std::vector<float> output_fp32(d_model);
    std::vector<float> output_bf16(d_model);

    // Run FP32 reference
    WoProjectionParams params_fp32 = {
        .context = context.data(),
        .wo_weights = wo_fp32.data(),
        .wo_type = WoWeightType::FP32,
        .head_dim = head_dim,
        .d_model = d_model,
        .head_idx = 0,
        .n_heads = n_heads,
        .output = output_fp32.data(),
        .accumulate = false};
    wo_projection_fp32_ref(params_fp32);

    // Run BF16 version
    WoProjectionParams params_bf16 = {
        .context = context.data(),
        .wo_weights = wo_bf16.data(),
        .wo_type = WoWeightType::BF16,
        .head_dim = head_dim,
        .d_model = d_model,
        .head_idx = 0,
        .n_heads = n_heads,
        .output = output_bf16.data(),
        .accumulate = false};
    wo_projection_bf16_ref(params_bf16);

    // BF16 has less precision than FP16, but still good for our use case
    // Expect ~0.5-1% relative error due to truncation
    for (int o = 0; o < d_model; ++o)
    {
        EXPECT_NEAR(output_bf16[o], output_fp32[o], std::abs(output_fp32[o]) * 0.1f + 1e-3f)
            << "Mismatch at output " << o;
    }
}

TEST_F(Test__WoProjection, BF16_AllHeads)
{
    const int d_model = 64;
    const int n_heads = 4;
    const int head_dim = 16;

    std::vector<float> wo_fp32 = createRandomWo(d_model, n_heads, head_dim);
    std::vector<uint16_t> wo_bf16(wo_fp32.size());
    for (size_t i = 0; i < wo_fp32.size(); ++i)
    {
        wo_bf16[i] = llaminar2::simd::fp32_to_bf16(wo_fp32[i]);
    }

    std::vector<float> context(head_dim);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto &c : context)
        c = dist(rng_);

    for (int h = 0; h < n_heads; ++h)
    {
        std::vector<float> output_fp32(d_model);
        std::vector<float> output_bf16(d_model);

        WoProjectionParams params_fp32 = {
            .context = context.data(),
            .wo_weights = wo_fp32.data(),
            .wo_type = WoWeightType::FP32,
            .head_dim = head_dim,
            .d_model = d_model,
            .head_idx = h,
            .n_heads = n_heads,
            .output = output_fp32.data(),
            .accumulate = false};
        wo_projection_fp32_ref(params_fp32);

        WoProjectionParams params_bf16 = {
            .context = context.data(),
            .wo_weights = wo_bf16.data(),
            .wo_type = WoWeightType::BF16,
            .head_dim = head_dim,
            .d_model = d_model,
            .head_idx = h,
            .n_heads = n_heads,
            .output = output_bf16.data(),
            .accumulate = false};
        wo_projection_bf16_ref(params_bf16);

        for (int o = 0; o < d_model; ++o)
        {
            EXPECT_NEAR(output_bf16[o], output_fp32[o], std::abs(output_fp32[o]) * 0.1f + 1e-3f)
                << "Mismatch at head " << h << ", output " << o;
        }
    }
}

TEST_F(Test__WoProjection, BF16_Accumulation)
{
    const int d_model = 32;
    const int n_heads = 2;
    const int head_dim = 16;

    std::vector<float> wo_fp32 = createRandomWo(d_model, n_heads, head_dim);
    std::vector<uint16_t> wo_bf16(wo_fp32.size());
    for (size_t i = 0; i < wo_fp32.size(); ++i)
    {
        wo_bf16[i] = llaminar2::simd::fp32_to_bf16(wo_fp32[i]);
    }

    std::vector<float> context(head_dim, 1.0f);
    std::vector<float> output(d_model, 5.0f);

    WoProjectionParams params = {
        .context = context.data(),
        .wo_weights = wo_bf16.data(),
        .wo_type = WoWeightType::BF16,
        .head_dim = head_dim,
        .d_model = d_model,
        .head_idx = 0,
        .n_heads = n_heads,
        .output = output.data(),
        .accumulate = true};
    wo_projection_bf16_ref(params);

    // Verify accumulation happened
    bool any_changed = false;
    for (int o = 0; o < d_model; ++o)
    {
        if (std::abs(output[o] - 5.0f) > 1e-6f)
        {
            any_changed = true;
            break;
        }
    }
    EXPECT_TRUE(any_changed);
}

TEST_F(Test__WoProjection, BF16_Dispatch)
{
    const int d_model = 32;
    const int n_heads = 2;
    const int head_dim = 16;

    std::vector<float> wo_fp32 = createRandomWo(d_model, n_heads, head_dim);
    std::vector<uint16_t> wo_bf16(wo_fp32.size());
    for (size_t i = 0; i < wo_fp32.size(); ++i)
    {
        wo_bf16[i] = llaminar2::simd::fp32_to_bf16(wo_fp32[i]);
    }

    std::vector<float> context(head_dim);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto &c : context)
        c = dist(rng_);

    std::vector<float> output_direct(d_model);
    std::vector<float> output_dispatch(d_model);

    // Direct call
    WoProjectionParams params1 = {
        .context = context.data(),
        .wo_weights = wo_bf16.data(),
        .wo_type = WoWeightType::BF16,
        .head_dim = head_dim,
        .d_model = d_model,
        .head_idx = 0,
        .n_heads = n_heads,
        .output = output_direct.data(),
        .accumulate = false};
    wo_projection_bf16_ref(params1);

    // Dispatch call
    WoProjectionParams params2 = {
        .context = context.data(),
        .wo_weights = wo_bf16.data(),
        .wo_type = WoWeightType::BF16,
        .head_dim = head_dim,
        .d_model = d_model,
        .head_idx = 0,
        .n_heads = n_heads,
        .output = output_dispatch.data(),
        .accumulate = false};
    wo_projection(params2);

    for (int o = 0; o < d_model; ++o)
    {
        EXPECT_FLOAT_EQ(output_dispatch[o], output_direct[o]);
    }
}

TEST_F(Test__WoProjection, BF16_Qwen2Dimensions)
{
    // Test with actual Qwen2.5-0.5B dimensions
    const int d_model = 896;
    const int n_heads = 14;
    const int head_dim = 64;

    std::vector<float> wo_fp32 = createRandomWo(d_model, n_heads, head_dim);
    std::vector<uint16_t> wo_bf16(wo_fp32.size());
    for (size_t i = 0; i < wo_fp32.size(); ++i)
    {
        wo_bf16[i] = llaminar2::simd::fp32_to_bf16(wo_fp32[i]);
    }

    std::vector<float> context(head_dim);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    for (auto &c : context)
        c = dist(rng_);

    std::vector<float> output_fp32(d_model);
    std::vector<float> output_bf16(d_model);

    WoProjectionParams params_fp32 = {
        .context = context.data(),
        .wo_weights = wo_fp32.data(),
        .wo_type = WoWeightType::FP32,
        .head_dim = head_dim,
        .d_model = d_model,
        .head_idx = 7,
        .n_heads = n_heads,
        .output = output_fp32.data(),
        .accumulate = false};
    wo_projection_fp32_ref(params_fp32);

    WoProjectionParams params_bf16 = {
        .context = context.data(),
        .wo_weights = wo_bf16.data(),
        .wo_type = WoWeightType::BF16,
        .head_dim = head_dim,
        .d_model = d_model,
        .head_idx = 7,
        .n_heads = n_heads,
        .output = output_bf16.data(),
        .accumulate = false};
    wo_projection_bf16_ref(params_bf16);

    // Compute max absolute error and mean absolute error
    float max_abs_err = 0.0f;
    float sum_abs_err = 0.0f;
    for (int o = 0; o < d_model; ++o)
    {
        float abs_err = std::abs(output_bf16[o] - output_fp32[o]);
        max_abs_err = std::max(max_abs_err, abs_err);
        sum_abs_err += abs_err;
    }
    float mean_abs_err = sum_abs_err / d_model;

    // BF16 should have reasonable absolute error for typical values in range [-0.1, 0.1]
    // With d_model=896, n_heads=14, head_dim=64, random [-0.1, 0.1] inputs,
    // outputs are small (order of 1e-4 to 1e-2), so max error of 0.01 is reasonable
    EXPECT_LT(max_abs_err, 0.01f) << "Max absolute error too high: " << max_abs_err;
    EXPECT_LT(mean_abs_err, 0.001f) << "Mean absolute error too high: " << mean_abs_err;

    std::cout << "BF16 Qwen2 - Max abs error: " << max_abs_err << ", Mean abs error: " << mean_abs_err << std::endl;
}

TEST_F(Test__WoProjection, BF16_VsFP16_Precision)
{
    // Compare BF16 vs FP16 precision for same weights
    const int d_model = 128;
    const int n_heads = 4;
    const int head_dim = 32;

    std::vector<float> wo_fp32 = createRandomWo(d_model, n_heads, head_dim);
    std::vector<uint16_t> wo_fp16(wo_fp32.size());
    std::vector<uint16_t> wo_bf16(wo_fp32.size());
    for (size_t i = 0; i < wo_fp32.size(); ++i)
    {
        wo_fp16[i] = llaminar2::fp32_to_fp16(wo_fp32[i]);
        wo_bf16[i] = llaminar2::simd::fp32_to_bf16(wo_fp32[i]);
    }

    std::vector<float> context(head_dim);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto &c : context)
        c = dist(rng_);

    std::vector<float> output_fp32(d_model);
    std::vector<float> output_fp16(d_model);
    std::vector<float> output_bf16(d_model);

    WoProjectionParams params_fp32 = {
        .context = context.data(),
        .wo_weights = wo_fp32.data(),
        .wo_type = WoWeightType::FP32,
        .head_dim = head_dim,
        .d_model = d_model,
        .head_idx = 0,
        .n_heads = n_heads,
        .output = output_fp32.data(),
        .accumulate = false};
    wo_projection_fp32_ref(params_fp32);

    WoProjectionParams params_fp16 = {
        .context = context.data(),
        .wo_weights = wo_fp16.data(),
        .wo_type = WoWeightType::FP16,
        .head_dim = head_dim,
        .d_model = d_model,
        .head_idx = 0,
        .n_heads = n_heads,
        .output = output_fp16.data(),
        .accumulate = false};
    wo_projection_fp16_ref(params_fp16);

    WoProjectionParams params_bf16 = {
        .context = context.data(),
        .wo_weights = wo_bf16.data(),
        .wo_type = WoWeightType::BF16,
        .head_dim = head_dim,
        .d_model = d_model,
        .head_idx = 0,
        .n_heads = n_heads,
        .output = output_bf16.data(),
        .accumulate = false};
    wo_projection_bf16_ref(params_bf16);

    // Compute errors
    float fp16_max_err = 0.0f, bf16_max_err = 0.0f;
    for (int o = 0; o < d_model; ++o)
    {
        fp16_max_err = std::max(fp16_max_err, std::abs(output_fp16[o] - output_fp32[o]));
        bf16_max_err = std::max(bf16_max_err, std::abs(output_bf16[o] - output_fp32[o]));
    }

    // FP16 should generally be more precise than BF16 for same input range
    // But both should be reasonable
    EXPECT_LT(fp16_max_err, 0.01f) << "FP16 max error too high";
    EXPECT_LT(bf16_max_err, 0.05f) << "BF16 max error too high";

    // Print for informational purposes
    std::cout << "FP16 max error: " << fp16_max_err << std::endl;
    std::cout << "BF16 max error: " << bf16_max_err << std::endl;
}
