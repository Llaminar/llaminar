/**
 * @file Test__VectorizedPrimitives.cpp
 * @brief Tests for vectorized primitives (RoPE, RMSNorm, Softmax)
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include "kernels/cpu/primitives/RoPEPrimitives.h"
#include "kernels/cpu/primitives/RMSNormPrimitives.h"
#include "kernels/cpu/primitives/SoftmaxPrimitives_New.h"
#include <cmath>
#include <vector>
#include <algorithm>

using namespace llaminar2::primitives;

// ============================================================================
// RoPE Tests
// ============================================================================

TEST(Test__VectorizedPrimitives, RoPE_InvFreqCache)
{
    // Test that inverse frequency cache works correctly
    const int head_dim = 128;
    const float freq_base = 10000.0f;

    auto &inv_freq1 = get_inv_freq_cached(head_dim, freq_base);
    auto &inv_freq2 = get_inv_freq_cached(head_dim, freq_base);

    // Should return same reference (cached)
    EXPECT_EQ(&inv_freq1, &inv_freq2);
    EXPECT_EQ(inv_freq1.size(), 64); // head_dim / 2

    // Verify first few values
    const float log_base = std::log(freq_base);
    for (int i = 0; i < 5; ++i)
    {
        float exponent = (2.0f * i) / head_dim;
        float expected = std::exp(-log_base * exponent);
        EXPECT_NEAR(inv_freq1[i], expected, 1e-6f);
    }
}

TEST(Test__VectorizedPrimitives, RoPE_BasicRotation)
{
    // Test basic RoPE rotation on small tensor
    const int seq_len = 4;
    const int head_dim = 64;
    const int q_heads = 8;
    const int k_heads = 8;
    const int n_past = 0;
    const float freq_base = 10000.0f;

    std::vector<float> q(seq_len * q_heads * head_dim, 1.0f);
    std::vector<float> k(seq_len * k_heads * head_dim, 1.0f);

    apply_rope_vectorized(q.data(), k.data(), seq_len, head_dim, q_heads, k_heads, n_past, freq_base);

    // After rotation, values should not all be 1.0
    bool has_changed = false;
    for (float val : q)
    {
        if (std::abs(val - 1.0f) > 1e-5f)
        {
            has_changed = true;
            break;
        }
    }
    EXPECT_TRUE(has_changed);
}

TEST(Test__VectorizedPrimitives, RoPE_PersistentState)
{
    // Test single-token decode with persistent state
    const int head_dim = 128;
    const int q_heads = 8;
    const int k_heads = 8;
    const float freq_base = 10000.0f;

    RoPEPersistentState state;

    // First token at position 0
    std::vector<float> q1(q_heads * head_dim, 1.0f);
    std::vector<float> k1(k_heads * head_dim, 1.0f);
    apply_rope_vectorized(q1.data(), k1.data(), 1, head_dim, q_heads, k_heads, 0, freq_base, &state);

    EXPECT_EQ(state.last_pos, 0);

    // Second token at position 1 (should use recurrence)
    std::vector<float> q2(q_heads * head_dim, 1.0f);
    std::vector<float> k2(k_heads * head_dim, 1.0f);
    apply_rope_vectorized(q2.data(), k2.data(), 1, head_dim, q_heads, k_heads, 1, freq_base, &state);

    EXPECT_EQ(state.last_pos, 1);

    // Rotations should be different
    bool different = false;
    for (size_t i = 0; i < q1.size(); ++i)
    {
        if (std::abs(q1[i] - q2[i]) > 1e-5f)
        {
            different = true;
            break;
        }
    }
    EXPECT_TRUE(different);
}

// ============================================================================
// RMSNorm Tests
// ============================================================================

TEST(Test__VectorizedPrimitives, RMSNorm_BasicComputation)
{
    // Test basic RMSNorm computation
    const int rows = 4;
    const int cols = 128;
    const float epsilon = 1e-6f;

    std::vector<float> input(rows * cols);
    std::vector<float> gamma(cols, 1.0f);
    std::vector<float> output(rows * cols);

    // Initialize input with known values
    for (int r = 0; r < rows; ++r)
    {
        for (int c = 0; c < cols; ++c)
        {
            input[r * cols + c] = (r + 1) * 0.1f; // Each row has constant values
        }
    }

    RMSNormExecOptions opts;
    opts.force_scalar = false; // Use vectorized path

    rmsnorm_fused_vectorized(input.data(), gamma.data(), output.data(), rows, cols, epsilon, opts);

    // Verify output rows have unit RMS
    for (int r = 0; r < rows; ++r)
    {
        double sum_sq = 0.0;
        for (int c = 0; c < cols; ++c)
        {
            float val = output[r * cols + c];
            sum_sq += val * val;
        }
        float rms = std::sqrt(sum_sq / cols);
        EXPECT_NEAR(rms, 1.0f, 1e-4f); // Should be normalized to unit RMS
    }
}

TEST(Test__VectorizedPrimitives, RMSNorm_T5CompatMode)
{
    // Test T5 compatibility mode (float32 accumulation)
    const int rows = 2;
    const int cols = 64;
    const float epsilon = 1e-6f;

    std::vector<float> input(rows * cols, 0.5f);
    std::vector<float> gamma(cols, 1.0f);
    std::vector<float> output_t5(rows * cols);
    std::vector<float> output_normal(rows * cols);

    RMSNormExecOptions opts_t5;
    opts_t5.t5_compat_mode = true;

    RMSNormExecOptions opts_normal;
    opts_normal.t5_compat_mode = false;

    rmsnorm_fused_vectorized(input.data(), gamma.data(), output_t5.data(), rows, cols, epsilon, opts_t5);
    rmsnorm_fused_vectorized(input.data(), gamma.data(), output_normal.data(), rows, cols, epsilon, opts_normal);

    // Results should be very similar (T5 uses float32, normal uses double accumulation)
    for (size_t i = 0; i < output_t5.size(); ++i)
    {
        EXPECT_NEAR(output_t5[i], output_normal[i], 1e-5f);
    }
}

// ============================================================================
// Softmax Tests
// ============================================================================

TEST(Test__VectorizedPrimitives, Softmax_BasicComputation)
{
    // Test basic softmax computation
    const int rows = 4;
    const int cols = 16;

    std::vector<float> scores(rows * cols);

    // Initialize with simple pattern
    for (int r = 0; r < rows; ++r)
    {
        for (int c = 0; c < cols; ++c)
        {
            scores[r * cols + c] = static_cast<float>(c); // Increasing values
        }
    }

    softmax_row_major_fp32(scores.data(), rows, cols, false, 1.0f, true);

    // Verify each row sums to 1.0
    for (int r = 0; r < rows; ++r)
    {
        float sum = 0.0f;
        for (int c = 0; c < cols; ++c)
        {
            float val = scores[r * cols + c];
            EXPECT_GE(val, 0.0f); // All probabilities non-negative
            EXPECT_LE(val, 1.0f); // All probabilities <= 1
            sum += val;
        }
        EXPECT_NEAR(sum, 1.0f, 1e-5f); // Row sums to 1
    }

    // Higher indices should have higher probabilities (increasing input)
    for (int r = 0; r < rows; ++r)
    {
        EXPECT_LT(scores[r * cols + 0], scores[r * cols + cols - 1]);
    }
}

TEST(Test__VectorizedPrimitives, Softmax_CausalMasking)
{
    // Test causal masking (upper triangle = 0)
    const int rows = 4;
    const int cols = 4;

    std::vector<float> scores(rows * cols, 1.0f); // All ones initially

    softmax_row_major_fp32(scores.data(), rows, cols, true, 1.0f, true);

    // Verify causal mask: position j > i should be 0
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            if (j > i)
            {
                EXPECT_FLOAT_EQ(scores[i * cols + j], 0.0f);
            }
            else
            {
                EXPECT_GT(scores[i * cols + j], 0.0f); // Lower triangle should be > 0
            }
        }

        // Each row should still sum to 1.0
        float sum = 0.0f;
        for (int j = 0; j < cols; ++j)
        {
            sum += scores[i * cols + j];
        }
        EXPECT_NEAR(sum, 1.0f, 1e-5f);
    }
}

TEST(Test__VectorizedPrimitives, Softmax_WithScaling)
{
    // Test softmax with scaling factor
    const int rows = 2;
    const int cols = 8;
    const float scale = 0.125f; // 1/sqrt(64) for head_dim=64

    std::vector<float> scores(rows * cols, 2.0f);

    softmax_row_major_fp32(scores.data(), rows, cols, false, scale, true);

    // With uniform input, softmax should produce uniform output
    for (int r = 0; r < rows; ++r)
    {
        for (int c = 0; c < cols; ++c)
        {
            float expected = 1.0f / cols; // Uniform distribution
            EXPECT_NEAR(scores[r * cols + c], expected, 1e-5f);
        }
    }
}
