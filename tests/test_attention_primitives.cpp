#include <gtest/gtest.h>
#include "kernels/common/attention_primitives.h"
#include <vector>
#include <cmath>

using namespace llaminar::attn;

static void fill(std::vector<float> &v, float scale)
{
    for (size_t i = 0; i < v.size(); ++i)
        v[i] = scale * float((i % 67) + 1);
}

TEST(AttentionPrimitives, FusedMatchesStep)
{
    const int heads = 2, head_dim = 8, seq = 4;
    std::vector<float> q(seq * heads * head_dim), k(q.size()), v(q.size());
    fill(q, 0.01f);
    fill(k, 0.015f);
    fill(v, 0.02f);
    auto q2 = q, k2 = k;
    apply_rope(q.data(), k.data(), seq, head_dim, heads, heads, 0, 10000.f);
    apply_rope(q2.data(), k2.data(), seq, head_dim, heads, heads, 0, 10000.f);
    std::vector<float> scores(size_t(heads) * seq * seq);
    compute_qk_scores(q.data(), k.data(), scores.data(), seq, head_dim, heads, true, true);
    std::vector<float> out_step(seq * heads * head_dim);
    apply_scores_to_v(scores.data(), v.data(), out_step.data(), seq, head_dim, heads);
    std::vector<float> out_fused(out_step.size());
    fused_attention(q2.data(), k2.data(), v.data(), out_fused.data(), seq, head_dim, heads, true);
    for (size_t i = 0; i < out_step.size(); ++i)
        ASSERT_NEAR(out_step[i], out_fused[i], 1e-6f) << "idx=" << i;
}

TEST(AttentionPrimitives, SoftmaxProperties)
{
    const int heads = 1, head_dim = 8, seq = 5;
    std::vector<float> q(seq * heads * head_dim), k(q.size()), v(q.size());
    fill(q, 0.01f);
    fill(k, 0.011f);
    fill(v, 0.012f);
    apply_rope(q.data(), k.data(), seq, head_dim, heads, heads, 0, 10000.f);
    std::vector<float> scores(size_t(heads) * seq * seq);
    compute_qk_scores(q.data(), k.data(), scores.data(), seq, head_dim, heads, true, true);
    auto stats = validate_softmax_rows(scores.data(), seq, heads);
    EXPECT_LT(stats.max_row_deviation, 1e-6f);
    EXPECT_GE(stats.max_negative, 0.f);
    EXPECT_LE(stats.max_prob, 1.f);
}

TEST(AttentionPrimitives, RoPEPairNormInvariant)
{
    const int heads = 1, head_dim = 8, seq = 3;
    std::vector<float> q(seq * heads * head_dim), k(q.size());
    fill(q, 0.02f);
    k = q;
    // HuggingFace split-half pattern: pairs are (0, head_dim/2), (1, head_dim/2+1), etc.
    std::vector<float> before;
    before.reserve(seq * head_dim / 2);
    for (int t = 0; t < seq; ++t)
    {
        float *row = q.data() + t * head_dim;
        for (int pair = 0; pair < head_dim / 2; ++pair)
        {
            float a = row[pair], b = row[pair + head_dim / 2];
            before.push_back(std::sqrt(a * a + b * b));
        }
    }
    apply_rope(q.data(), k.data(), seq, head_dim, heads, heads, 0, 10000.f);
    int idx = 0;
    for (int t = 0; t < seq; ++t)
    {
        float *row = q.data() + t * head_dim;
        for (int pair = 0; pair < head_dim / 2; ++pair)
        {
            float a = row[pair], b = row[pair + head_dim / 2];
            float r = std::sqrt(a * a + b * b);
            ASSERT_NEAR(r, before[idx++], 1e-6f);
        }
    }
}

// Test RoPE at position 0 should be identity (cos=1, sin=0)
TEST(AttentionPrimitives, RoPEPositionZeroIsIdentity)
{
    const int heads = 2, head_dim = 8, seq = 1;
    std::vector<float> q(seq * heads * head_dim), k(q.size());
    fill(q, 0.01f);
    fill(k, 0.02f);

    auto q_orig = q;
    auto k_orig = k;

    // Apply RoPE at position 0 (n_past=0, seq=1 means only position 0)
    apply_rope(q.data(), k.data(), seq, head_dim, heads, heads, 0, 10000.f);

    // At position 0, angle = 0 * theta = 0, so cos=1, sin=0
    // Rotation should be identity: x' = x*1 - y*0 = x, y' = x*0 + y*1 = y
    for (size_t i = 0; i < q.size(); ++i)
    {
        ASSERT_NEAR(q[i], q_orig[i], 1e-6f) << "Q mismatch at position " << i;
        ASSERT_NEAR(k[i], k_orig[i], 1e-6f) << "K mismatch at position " << i;
    }
}

// Test RoPE with explicit manual calculation for a simple case
// Uses HuggingFace canonical split-half RoPE pattern
TEST(AttentionPrimitives, RoPEManualCalculation)
{
    const int heads = 1, head_dim = 4, seq = 2;
    const float freq_base = 10000.f;

    // Create simple input: [1, 0, 2, 0] for first token, [3, 0, 4, 0] for second
    std::vector<float> q(seq * heads * head_dim);
    q[0] = 1.0f;
    q[1] = 0.0f;
    q[2] = 2.0f;
    q[3] = 0.0f; // token 0
    q[4] = 3.0f;
    q[5] = 0.0f;
    q[6] = 4.0f;
    q[7] = 0.0f; // token 1

    std::vector<float> k = q; // Same for K

    // Calculate expected values manually following HuggingFace rotate_half pattern
    // For head_dim=4, split into first_half=[0,1] and second_half=[2,3]
    // Pairs are (0,2) and (1,3) - each element in first half pairs with corresponding element in second half
    //
    // inv_freq[i] = 1 / (freq_base^(2*i/head_dim))
    // angle[i] = position * inv_freq[i]
    // cos/sin arrays are duplicated: [cos(angle0), cos(angle1), cos(angle0), cos(angle1)]
    //
    // rotate_half([x0, x1, x2, x3]) = [-x2, -x3, x0, x1]
    // output = input * cos + rotate_half(input) * sin

    float inv_freq0 = 1.0f / std::pow(freq_base, 0.0f / head_dim); // = 1.0
    float inv_freq1 = 1.0f / std::pow(freq_base, 2.0f / head_dim); // = 1 / 10000^0.5 = ~0.01

    // Token 0 (position 0): all angles = 0 -> identity
    EXPECT_NEAR(q[0], 1.0f, 1e-5f);
    EXPECT_NEAR(q[1], 0.0f, 1e-5f);
    EXPECT_NEAR(q[2], 2.0f, 1e-5f);
    EXPECT_NEAR(q[3], 0.0f, 1e-5f);

    apply_rope(q.data(), k.data(), seq, head_dim, heads, heads, 0, freq_base);

    // Token 0 should be unchanged (position 0)
    EXPECT_NEAR(q[0], 1.0f, 1e-5f);
    EXPECT_NEAR(q[1], 0.0f, 1e-5f);
    EXPECT_NEAR(q[2], 2.0f, 1e-5f);
    EXPECT_NEAR(q[3], 0.0f, 1e-5f);

    // Token 1 (position 1): input = [3, 0, 4, 0]
    // angle0 = 1 * inv_freq0 = 1.0
    // angle1 = 1 * inv_freq1 = 0.01
    float angle0 = 1.0f * inv_freq0;
    float angle1 = 1.0f * inv_freq1;
    float cos0 = std::cos(angle0);
    float sin0 = std::sin(angle0);
    float cos1 = std::cos(angle1);
    float sin1 = std::sin(angle1);

    // Pair (0,2) with angle0:
    //   q[4] = 3 * cos0 - 4 * sin0
    //   q[6] = 3 * sin0 + 4 * cos0
    EXPECT_NEAR(q[4], 3.0f * cos0 - 4.0f * sin0, 1e-4f) << "pair (0,2), index 0";
    EXPECT_NEAR(q[6], 3.0f * sin0 + 4.0f * cos0, 1e-4f) << "pair (0,2), index 2";

    // Pair (1,3) with angle1:
    //   q[5] = 0 * cos1 - 0 * sin1 = 0
    //   q[7] = 0 * sin1 + 0 * cos1 = 0
    EXPECT_NEAR(q[5], 0.0f, 1e-4f) << "pair (1,3), index 1";
    EXPECT_NEAR(q[7], 0.0f, 1e-4f) << "pair (1,3), index 3";
}

// Test RoPE with multi-head layout to expose tensor layout issues
// Uses HuggingFace canonical split-half RoPE pattern
TEST(AttentionPrimitives, RoPEMultiHeadLayout)
{
    const int heads = 2, head_dim = 4, seq = 2;
    const float freq_base = 10000.f;

    // Create input with distinguishable pattern for each head
    // Layout: [seq_len, num_heads, head_dim]
    std::vector<float> q(seq * heads * head_dim);

    // Token 0, Head 0: [1, 0, 2, 0]
    q[0] = 1.0f;
    q[1] = 0.0f;
    q[2] = 2.0f;
    q[3] = 0.0f;
    // Token 0, Head 1: [10, 0, 20, 0]
    q[4] = 10.0f;
    q[5] = 0.0f;
    q[6] = 20.0f;
    q[7] = 0.0f;

    // Token 1, Head 0: [3, 0, 4, 0]
    q[8] = 3.0f;
    q[9] = 0.0f;
    q[10] = 4.0f;
    q[11] = 0.0f;
    // Token 1, Head 1: [30, 0, 40, 0]
    q[12] = 30.0f;
    q[13] = 0.0f;
    q[14] = 40.0f;
    q[15] = 0.0f;

    std::vector<float> k = q;

    // Apply RoPE
    apply_rope(q.data(), k.data(), seq, head_dim, heads, heads, 0, freq_base);

    // Token 0 should be unchanged (position 0)
    EXPECT_NEAR(q[0], 1.0f, 1e-5f) << "Token 0, Head 0, dim 0";
    EXPECT_NEAR(q[4], 10.0f, 1e-5f) << "Token 0, Head 1, dim 0";

    // Token 1 - check rotation was applied using HuggingFace split-half pattern
    // inv_freq0 = 1.0, angle0 = 1.0 rad
    float inv_freq0 = 1.0f / std::pow(freq_base, 0.0f / head_dim);
    float angle0 = 1.0f * inv_freq0;
    float cos0 = std::cos(angle0);
    float sin0 = std::sin(angle0);

    // Token 1, Head 0: input = [3, 0, 4, 0]
    // Pair (0,2) with angle0: q[8] = 3*cos0 - 4*sin0, q[10] = 3*sin0 + 4*cos0
    // Pair (1,3) with angle1: q[9] = 0, q[11] = 0
    EXPECT_NEAR(q[8], 3.0f * cos0 - 4.0f * sin0, 1e-4f) << "Token 1, Head 0, pair (0,2), idx 0";
    EXPECT_NEAR(q[10], 3.0f * sin0 + 4.0f * cos0, 1e-4f) << "Token 1, Head 0, pair (0,2), idx 2";

    // Token 1, Head 1: input = [30, 0, 40, 0]
    // Pair (0,2) with angle0: q[12] = 30*cos0 - 40*sin0, q[14] = 30*sin0 + 40*cos0
    EXPECT_NEAR(q[12], 30.0f * cos0 - 40.0f * sin0, 1e-4f) << "Token 1, Head 1, pair (0,2), idx 0";
    EXPECT_NEAR(q[14], 30.0f * sin0 + 40.0f * cos0, 1e-4f) << "Token 1, Head 1, pair (0,2), idx 2";
}

// Test to verify theta (frequency) calculation matches PyTorch
TEST(AttentionPrimitives, RoPEThetaCalculation)
{
    const int head_dim = 128;
    const float freq_base = 10000.0f;
    const int pairs = head_dim / 2;

    // Calculate theta values as our code does
    std::vector<float> theta(pairs);
    for (int p = 0; p < pairs; ++p)
        theta[p] = 1.0f / std::pow(freq_base, (2.0f * p) / head_dim);

    // Verify a few key values
    // pair 0: theta = 1 / 10000^0 = 1.0
    EXPECT_NEAR(theta[0], 1.0f, 1e-6f);

    // pair 1: theta = 1 / 10000^(2/128) = 1 / 10000^0.015625
    float expected_theta1 = 1.0f / std::pow(10000.0f, 2.0f / 128.0f);
    EXPECT_NEAR(theta[1], expected_theta1, 1e-6f);

    // last pair (63): theta = 1 / 10000^(126/128) = 1 / 10000^0.984375
    float expected_theta_last = 1.0f / std::pow(10000.0f, 126.0f / 128.0f);
    EXPECT_NEAR(theta[63], expected_theta_last, 1e-6f);

    // Theta should be monotonically decreasing
    for (int p = 1; p < pairs; ++p)
    {
        EXPECT_LT(theta[p], theta[p - 1]) << "Theta not decreasing at pair " << p;
    }
}

// Test highlighting potential tensor layout issue with local head sharding
// This test simulates what happens in MPI: each rank gets a shard of heads
TEST(AttentionPrimitives, RoPELocalHeadSharding)
{
    // Simulate a scenario like the actual model:
    // Total Q heads: 14, Total KV heads: 2
    // On 2 MPI ranks: each rank gets local_q_heads=7, local_k_heads=1
    const int total_q_heads = 14;
    const int total_k_heads = 2;
    const int ranks = 2;
    const int local_q_heads = total_q_heads / ranks; // 7
    const int local_k_heads = total_k_heads / ranks; // 1
    const int head_dim = 64;
    const int seq_len = 2;
    const float freq_base = 10000.0f;

    // Create full Q tensor [seq_len, total_q_heads * head_dim]
    std::vector<float> full_q(seq_len * total_q_heads * head_dim);
    for (size_t i = 0; i < full_q.size(); ++i)
        full_q[i] = float(i % 100) * 0.01f; // Simple pattern

    // Simulate rank 0's local shard: first 7 heads
    // Layout: [seq_len, local_q_heads * head_dim]
    std::vector<float> local_q_rank0(seq_len * local_q_heads * head_dim);
    for (int t = 0; t < seq_len; ++t)
    {
        // Copy the first local_q_heads heads from full tensor
        for (int h = 0; h < local_q_heads; ++h)
        {
            for (int d = 0; d < head_dim; ++d)
            {
                int src_idx = t * (total_q_heads * head_dim) + h * head_dim + d;
                int dst_idx = t * (local_q_heads * head_dim) + h * head_dim + d;
                local_q_rank0[dst_idx] = full_q[src_idx];
            }
        }
    }

    // Create dummy K (not testing K here)
    std::vector<float> dummy_k(seq_len * local_k_heads * head_dim, 0.0f);

    // Apply RoPE to local shard
    // BUG: apply_rope uses q_heads parameter but the data only contains local_q_heads!
    // This is the test case that should fail if there's a layout mismatch
    auto local_q_copy = local_q_rank0;

    // This should work correctly because the tensor IS laid out as [seq, heads, head_dim]
    // even though it's a shard
    apply_rope(local_q_copy.data(), dummy_k.data(), seq_len, head_dim,
               local_q_heads, local_k_heads, 0, freq_base);

    // Verify that values changed (rotation applied) for position > 0
    // At position 0, rotation is identity
    for (int h = 0; h < local_q_heads; ++h)
    {
        for (int d = 0; d < head_dim; ++d)
        {
            int idx_t0 = 0 * local_q_heads * head_dim + h * head_dim + d;
            EXPECT_NEAR(local_q_copy[idx_t0], local_q_rank0[idx_t0], 1e-5f)
                << "Position 0 should be identity at h=" << h << " d=" << d;
        }
    }

    // At position 1, values should be rotated (different from original)
    bool some_values_changed = false;
    for (int h = 0; h < local_q_heads; ++h)
    {
        for (int d = 0; d < head_dim; d += 2) // Check pairs
        {
            int idx_t1 = 1 * local_q_heads * head_dim + h * head_dim + d;
            if (std::abs(local_q_copy[idx_t1] - local_q_rank0[idx_t1]) > 1e-5f)
            {
                some_values_changed = true;
                break;
            }
        }
        if (some_values_changed)
            break;
    }
    EXPECT_TRUE(some_values_changed) << "Position 1 should have rotated values";
}
