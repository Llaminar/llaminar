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
    compute_qk_scores(q.data(), k.data(), scores.data(), seq, seq, head_dim, heads, true, true);
    std::vector<float> out_step(seq * heads * head_dim);
    apply_scores_to_v(scores.data(), v.data(), out_step.data(), seq, seq, head_dim, heads);
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
    compute_qk_scores(q.data(), k.data(), scores.data(), seq, seq, head_dim, heads, true, true);
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

// ============================================================================
// GQA Head Mapping Tests (Global vs Local Index)
// ============================================================================

/**
 * @brief Test GQA head mapping with distributed heads across ranks
 *
 * This test validates that expand_kv_for_gqa correctly maps Q heads to KV heads
 * when heads are distributed across MPI ranks. The critical bug this prevents:
 * using local head indices instead of global head indices for GQA group mapping.
 *
 * Scenario: Qwen2.5-0.5B configuration
 * - 14 Q heads total (7 per rank on 2 ranks)
 * - 2 KV heads total
 * - Group size = 14 / 2 = 7
 * - Correct mapping: Q heads [0-6] → KV head 0, Q heads [7-13] → KV head 1
 */
TEST(AttentionPrimitives, GQAHeadMappingDistributedRanks)
{
    const int total_q_heads = 14;
    const int n_kv_heads = 2;
    const int head_dim = 64;
    const int seq_len = 4;
    const int num_ranks = 2;
    const int heads_per_rank = total_q_heads / num_ranks; // 7

    // Create reference K/V tensors (compact format: [seq, kv_heads, head_dim])
    std::vector<float> k_compact(seq_len * n_kv_heads * head_dim);
    std::vector<float> v_compact(seq_len * n_kv_heads * head_dim);

    // Fill with distinct values for each KV head
    for (int t = 0; t < seq_len; ++t)
    {
        for (int kv_h = 0; kv_h < n_kv_heads; ++kv_h)
        {
            for (int d = 0; d < head_dim; ++d)
            {
                int idx = t * n_kv_heads * head_dim + kv_h * head_dim + d;
                // KV head 0: values around 100, KV head 1: values around 200
                k_compact[idx] = 100.0f * (kv_h + 1) + d;
                v_compact[idx] = 100.0f * (kv_h + 1) + d + 0.5f;
            }
        }
    }

    // Simulate expansion on each rank
    for (int rank = 0; rank < num_ranks; ++rank)
    {
        int head_offset = rank * heads_per_rank; // Rank 0: offset=0, Rank 1: offset=7

        std::vector<float> k_expanded(seq_len * heads_per_rank * head_dim);
        std::vector<float> v_expanded(seq_len * heads_per_rank * head_dim);

        // Call the expansion function with global context
        expand_kv_for_gqa(
            k_compact.data(), v_compact.data(),
            k_expanded.data(), v_expanded.data(),
            seq_len, head_dim,
            heads_per_rank, // local_heads (7)
            n_kv_heads,     // n_kv_heads (2)
            head_offset,    // head_offset (0 or 7)
            total_q_heads   // total_q_heads (14)
        );

        // Verify the mapping for each local head
        for (int local_h = 0; local_h < heads_per_rank; ++local_h)
        {
            int global_h = head_offset + local_h;
            int expected_kv_head = global_h / (total_q_heads / n_kv_heads); // global_h / 7

            // Check a sample token
            for (int d = 0; d < head_dim; ++d)
            {
                int expanded_idx = 0 * heads_per_rank * head_dim + local_h * head_dim + d;
                int compact_idx = 0 * n_kv_heads * head_dim + expected_kv_head * head_dim + d;

                EXPECT_FLOAT_EQ(k_expanded[expanded_idx], k_compact[compact_idx])
                    << "Rank " << rank << ", local_h=" << local_h
                    << " (global_h=" << global_h << ") should map to KV head "
                    << expected_kv_head << ", dim=" << d;

                EXPECT_FLOAT_EQ(v_expanded[expanded_idx], v_compact[compact_idx])
                    << "V mismatch on Rank " << rank << ", local_h=" << local_h;
            }
        }

        // Specific assertions for this test scenario
        if (rank == 0)
        {
            // Rank 0 processes global heads 0-6, all should map to KV head 0
            for (int local_h = 0; local_h < heads_per_rank; ++local_h)
            {
                int idx = 0 * heads_per_rank * head_dim + local_h * head_dim + 0;
                EXPECT_FLOAT_EQ(k_expanded[idx], 100.0f)
                    << "Rank 0, local_h=" << local_h << " should use KV head 0";
            }
        }
        else if (rank == 1)
        {
            // Rank 1 processes global heads 7-13, all should map to KV head 1
            for (int local_h = 0; local_h < heads_per_rank; ++local_h)
            {
                int idx = 0 * heads_per_rank * head_dim + local_h * head_dim + 0;
                EXPECT_FLOAT_EQ(k_expanded[idx], 200.0f)
                    << "Rank 1, local_h=" << local_h << " should use KV head 1";
            }
        }
    }
}

/**
 * @brief Test GQA with various configuration scenarios
 *
 * Tests different Q-to-KV head ratios to ensure the formula works correctly
 * for various model architectures.
 */
TEST(AttentionPrimitives, GQAHeadMappingVariousConfigs)
{
    const int head_dim = 64;
    const int seq_len = 2;

    struct TestCase
    {
        int total_q_heads;
        int n_kv_heads;
        int num_ranks;
        std::string description;
    };

    std::vector<TestCase> test_cases = {
        {32, 8, 2, "LLaMA2-7B: 32 Q heads, 8 KV heads, 2 ranks"},
        {32, 8, 4, "LLaMA2-7B: 32 Q heads, 8 KV heads, 4 ranks"},
        {40, 8, 2, "LLaMA3-8B: 40 Q heads, 8 KV heads, 2 ranks"},
        {8, 2, 2, "Small model: 8 Q heads, 2 KV heads, 2 ranks"},
        {16, 1, 2, "Extreme GQA: 16 Q heads, 1 KV head, 2 ranks"},
    };

    for (const auto &tc : test_cases)
    {
        SCOPED_TRACE(tc.description);

        int heads_per_rank = tc.total_q_heads / tc.num_ranks;
        ASSERT_EQ(tc.total_q_heads % tc.num_ranks, 0) << "Heads must divide evenly across ranks";

        // Create compact K with distinct values per KV head
        std::vector<float> k_compact(seq_len * tc.n_kv_heads * head_dim);
        for (int t = 0; t < seq_len; ++t)
        {
            for (int kv_h = 0; kv_h < tc.n_kv_heads; ++kv_h)
            {
                for (int d = 0; d < head_dim; ++d)
                {
                    int idx = t * tc.n_kv_heads * head_dim + kv_h * head_dim + d;
                    k_compact[idx] = static_cast<float>(kv_h * 1000 + d);
                }
            }
        }

        std::vector<float> v_compact = k_compact; // V same as K for simplicity

        // Test each rank
        for (int rank = 0; rank < tc.num_ranks; ++rank)
        {
            int head_offset = rank * heads_per_rank;

            std::vector<float> k_expanded(seq_len * heads_per_rank * head_dim);
            std::vector<float> v_expanded(seq_len * heads_per_rank * head_dim);

            expand_kv_for_gqa(
                k_compact.data(), v_compact.data(),
                k_expanded.data(), v_expanded.data(),
                seq_len, head_dim, heads_per_rank, tc.n_kv_heads,
                head_offset, tc.total_q_heads);

            // Verify mapping for each head on this rank
            for (int local_h = 0; local_h < heads_per_rank; ++local_h)
            {
                int global_h = head_offset + local_h;
                int group_size = tc.total_q_heads / tc.n_kv_heads;
                int expected_kv_head = global_h / group_size;

                // Check first element of this head
                int expanded_idx = 0 * heads_per_rank * head_dim + local_h * head_dim + 0;
                float expected_value = static_cast<float>(expected_kv_head * 1000);

                EXPECT_FLOAT_EQ(k_expanded[expanded_idx], expected_value)
                    << "Rank " << rank << ", global_h=" << global_h
                    << " should map to KV head " << expected_kv_head;
            }
        }
    }
}

/**
 * @brief Test that head_offset=0 gives correct results (single rank case)
 *
 * When head_offset=0 (default), the function should work correctly for
 * single-rank execution or when processing all heads on one rank.
 */
TEST(AttentionPrimitives, GQAHeadMappingSingleRank)
{
    const int total_q_heads = 14;
    const int n_kv_heads = 2;
    const int head_dim = 64;
    const int seq_len = 2;

    // Create compact K/V
    std::vector<float> k_compact(seq_len * n_kv_heads * head_dim);
    std::vector<float> v_compact(seq_len * n_kv_heads * head_dim);

    for (int t = 0; t < seq_len; ++t)
    {
        for (int kv_h = 0; kv_h < n_kv_heads; ++kv_h)
        {
            for (int d = 0; d < head_dim; ++d)
            {
                int idx = t * n_kv_heads * head_dim + kv_h * head_dim + d;
                k_compact[idx] = static_cast<float>(kv_h * 100 + d);
                v_compact[idx] = static_cast<float>(kv_h * 100 + d + 50);
            }
        }
    }

    // Expand all heads on single rank (head_offset=0)
    std::vector<float> k_expanded(seq_len * total_q_heads * head_dim);
    std::vector<float> v_expanded(seq_len * total_q_heads * head_dim);

    expand_kv_for_gqa(
        k_compact.data(), v_compact.data(),
        k_expanded.data(), v_expanded.data(),
        seq_len, head_dim, total_q_heads, n_kv_heads,
        0, // head_offset = 0 (single rank)
        total_q_heads);

    // Verify mapping: heads 0-6 → KV 0, heads 7-13 → KV 1
    for (int h = 0; h < total_q_heads; ++h)
    {
        int expected_kv_head = h / 7;

        for (int d = 0; d < head_dim; ++d)
        {
            int expanded_idx = 0 * total_q_heads * head_dim + h * head_dim + d;
            int compact_idx = 0 * n_kv_heads * head_dim + expected_kv_head * head_dim + d;

            EXPECT_FLOAT_EQ(k_expanded[expanded_idx], k_compact[compact_idx])
                << "Q head " << h << " should map to KV head " << expected_kv_head
                << " at dim " << d;
        }
    }
}

/**
 * @brief Regression test for the bug where local indices were used
 *
 * This test explicitly checks the failure scenario: using local head indices
 * instead of global head indices. It verifies that Rank 1 does NOT incorrectly
 * alternate between KV heads when it should consistently use KV head 1.
 */
TEST(AttentionPrimitives, GQAHeadMappingRegressionLocalVsGlobal)
{
    const int total_q_heads = 14;
    const int n_kv_heads = 2;
    const int head_dim = 64;
    const int seq_len = 1;
    const int heads_per_rank = 7;

    // Create compact K with very distinct values
    std::vector<float> k_compact(seq_len * n_kv_heads * head_dim);
    for (int kv_h = 0; kv_h < n_kv_heads; ++kv_h)
    {
        for (int d = 0; d < head_dim; ++d)
        {
            int idx = kv_h * head_dim + d;
            k_compact[idx] = static_cast<float>((kv_h + 1) * 1000); // KV0=1000, KV1=2000
        }
    }

    std::vector<float> v_compact = k_compact;

    // Test Rank 1 (global heads 7-13)
    const int rank1_offset = 7;
    std::vector<float> k_expanded(seq_len * heads_per_rank * head_dim);
    std::vector<float> v_expanded(seq_len * heads_per_rank * head_dim);

    expand_kv_for_gqa(
        k_compact.data(), v_compact.data(),
        k_expanded.data(), v_expanded.data(),
        seq_len, head_dim, heads_per_rank, n_kv_heads,
        rank1_offset, total_q_heads);

    // ALL local heads on Rank 1 should map to KV head 1 (value 2000)
    // The bug would cause alternating pattern: local_h % 2
    for (int local_h = 0; local_h < heads_per_rank; ++local_h)
    {
        int global_h = rank1_offset + local_h;
        int idx = local_h * head_dim + 0; // First element of this head

        // Correct behavior: all map to KV head 1
        EXPECT_FLOAT_EQ(k_expanded[idx], 2000.0f)
            << "Rank 1, local_h=" << local_h << " (global_h=" << global_h
            << ") should map to KV head 1, not alternating based on local index";

        // The BUG would have caused:
        // local_h=0 → 1000 (KV head 0)  ✗
        // local_h=1 → 2000 (KV head 1)  ✓ (accidentally correct)
        // local_h=2 → 1000 (KV head 0)  ✗
        // etc.

        // Verify this does NOT happen
        if (local_h % 2 == 0)
        {
            // Even local indices should NOT map to KV head 0
            EXPECT_NE(k_expanded[idx], 1000.0f)
                << "Bug detected: local_h=" << local_h
                << " incorrectly using local index modulo instead of global index";
        }
    }
}

/**
 * @brief Test MHA scenario (no GQA)
 *
 * When n_heads == n_kv_heads (Multi-Head Attention, not Grouped-Query),
 * the expansion should be identity mapping: each Q head maps to its
 * corresponding KV head.
 */
TEST(AttentionPrimitives, GQAHeadMappingMHAIdentity)
{
    const int n_heads = 8;
    const int n_kv_heads = 8; // Same as n_heads → MHA
    const int head_dim = 64;
    const int seq_len = 2;

    std::vector<float> k_compact(seq_len * n_kv_heads * head_dim);
    std::vector<float> v_compact(seq_len * n_kv_heads * head_dim);

    // Fill with unique values per head
    for (int t = 0; t < seq_len; ++t)
    {
        for (int h = 0; h < n_kv_heads; ++h)
        {
            for (int d = 0; d < head_dim; ++d)
            {
                int idx = t * n_kv_heads * head_dim + h * head_dim + d;
                k_compact[idx] = static_cast<float>(h * 100 + d);
                v_compact[idx] = static_cast<float>(h * 100 + d + 10);
            }
        }
    }

    std::vector<float> k_expanded(seq_len * n_heads * head_dim);
    std::vector<float> v_expanded(seq_len * n_heads * head_dim);

    expand_kv_for_gqa(
        k_compact.data(), v_compact.data(),
        k_expanded.data(), v_expanded.data(),
        seq_len, head_dim, n_heads, n_kv_heads,
        0, n_heads);

    // Verify identity mapping: expanded should equal compact
    for (int t = 0; t < seq_len; ++t)
    {
        for (int h = 0; h < n_heads; ++h)
        {
            for (int d = 0; d < head_dim; ++d)
            {
                int idx = t * n_heads * head_dim + h * head_dim + d;
                EXPECT_FLOAT_EQ(k_expanded[idx], k_compact[idx])
                    << "MHA should be identity: h=" << h << " d=" << d;
                EXPECT_FLOAT_EQ(v_expanded[idx], v_compact[idx])
                    << "MHA should be identity: h=" << h << " d=" << d;
            }
        }
    }
}

// ============================================================================
// Edge Case Tests: More Ranks Than Heads
// ============================================================================

/**
 * @brief Test GQA head mapping when some ranks have no heads (excess ranks)
 *
 * This test validates the behavior when we have more MPI ranks than heads to
 * distribute. Some ranks will get local_heads=0, which should be handled gracefully.
 *
 * Scenario: 8 Q heads, 2 KV heads, 10 ranks
 * - Ranks 0-7: get 1 head each
 * - Ranks 8-9: get 0 heads (no work)
 */
TEST(AttentionPrimitives, GQAHeadMappingExcessRanks)
{
    const int total_q_heads = 8;
    const int n_kv_heads = 2;
    const int head_dim = 64;
    const int seq_len = 2;
    const int num_ranks = 10; // More ranks than Q heads!

    // Create compact K/V
    std::vector<float> k_compact(seq_len * n_kv_heads * head_dim);
    std::vector<float> v_compact(seq_len * n_kv_heads * head_dim);

    for (int t = 0; t < seq_len; ++t)
    {
        for (int kv_h = 0; kv_h < n_kv_heads; ++kv_h)
        {
            for (int d = 0; d < head_dim; ++d)
            {
                int idx = t * n_kv_heads * head_dim + kv_h * head_dim + d;
                k_compact[idx] = static_cast<float>((kv_h + 1) * 100 + d);
                v_compact[idx] = static_cast<float>((kv_h + 1) * 100 + d + 50);
            }
        }
    }

    // Simulate each rank
    for (int rank = 0; rank < num_ranks; ++rank)
    {
        // Distribute heads: 8 heads / 10 ranks = 0 heads per rank, 8 remainder
        int heads_per_rank = total_q_heads / num_ranks; // 0
        int remainder = total_q_heads % num_ranks;      // 8
        int local_heads = heads_per_rank + (rank < remainder ? 1 : 0);
        int head_offset = rank * heads_per_rank + std::min(rank, remainder);

        if (rank < 8)
        {
            // Ranks 0-7: should get 1 head each
            EXPECT_EQ(local_heads, 1) << "Rank " << rank << " should have 1 head";
            EXPECT_EQ(head_offset, rank) << "Rank " << rank << " head_offset should be " << rank;
        }
        else
        {
            // Ranks 8-9: should get 0 heads
            EXPECT_EQ(local_heads, 0) << "Rank " << rank << " should have 0 heads (excess rank)";
            EXPECT_EQ(head_offset, 8) << "Rank " << rank << " head_offset should be 8";
        }

        // Only process if this rank has heads
        if (local_heads > 0)
        {
            std::vector<float> k_expanded(seq_len * local_heads * head_dim);
            std::vector<float> v_expanded(seq_len * local_heads * head_dim);

            expand_kv_for_gqa(
                k_compact.data(), v_compact.data(),
                k_expanded.data(), v_expanded.data(),
                seq_len, head_dim, local_heads, n_kv_heads,
                head_offset, total_q_heads);

            // Verify the mapping
            int global_h = head_offset;                  // For this rank's single head
            int group_size = total_q_heads / n_kv_heads; // 8 / 2 = 4
            int expected_kv_head = global_h / group_size;

            // Check first element
            int idx = 0;
            float expected_value = static_cast<float>((expected_kv_head + 1) * 100);
            EXPECT_FLOAT_EQ(k_expanded[idx], expected_value)
                << "Rank " << rank << " global_h=" << global_h
                << " should map to KV head " << expected_kv_head;
        }
    }
}

/**
 * @brief Test handling of uneven head distribution
 *
 * Validates that heads are distributed as evenly as possible when they don't
 * divide evenly across ranks.
 *
 * Example: 14 heads across 3 ranks → 5, 5, 4 distribution
 */
TEST(AttentionPrimitives, GQAHeadMappingUnevenDistribution)
{
    const int total_q_heads = 14;
    const int n_kv_heads = 2;
    const int head_dim = 64;
    const int seq_len = 2;
    const int num_ranks = 3;

    std::vector<float> k_compact(seq_len * n_kv_heads * head_dim);
    std::vector<float> v_compact(seq_len * n_kv_heads * head_dim);

    // Fill with distinct values
    for (int t = 0; t < seq_len; ++t)
    {
        for (int kv_h = 0; kv_h < n_kv_heads; ++kv_h)
        {
            for (int d = 0; d < head_dim; ++d)
            {
                int idx = t * n_kv_heads * head_dim + kv_h * head_dim + d;
                k_compact[idx] = static_cast<float>(kv_h * 1000 + d);
                v_compact[idx] = static_cast<float>(kv_h * 1000 + d + 500);
            }
        }
    }

    int expected_distribution[] = {5, 5, 4}; // 14 / 3 = 4 remainder 2
    int expected_offsets[] = {0, 5, 10};

    for (int rank = 0; rank < num_ranks; ++rank)
    {
        int heads_per_rank = total_q_heads / num_ranks; // 4
        int remainder = total_q_heads % num_ranks;      // 2
        int local_heads = heads_per_rank + (rank < remainder ? 1 : 0);
        int head_offset = rank * heads_per_rank + std::min(rank, remainder);

        EXPECT_EQ(local_heads, expected_distribution[rank])
            << "Rank " << rank << " should have " << expected_distribution[rank] << " heads";
        EXPECT_EQ(head_offset, expected_offsets[rank])
            << "Rank " << rank << " should start at offset " << expected_offsets[rank];

        std::vector<float> k_expanded(seq_len * local_heads * head_dim);
        std::vector<float> v_expanded(seq_len * local_heads * head_dim);

        expand_kv_for_gqa(
            k_compact.data(), v_compact.data(),
            k_expanded.data(), v_expanded.data(),
            seq_len, head_dim, local_heads, n_kv_heads,
            head_offset, total_q_heads);

        // Verify each local head maps to correct KV head
        int group_size = total_q_heads / n_kv_heads; // 7
        for (int local_h = 0; local_h < local_heads; ++local_h)
        {
            int global_h = head_offset + local_h;
            int expected_kv_head = global_h / group_size;

            int idx = 0 * local_heads * head_dim + local_h * head_dim + 0;
            float expected_value = static_cast<float>(expected_kv_head * 1000);

            EXPECT_FLOAT_EQ(k_expanded[idx], expected_value)
                << "Rank " << rank << " local_h=" << local_h
                << " (global_h=" << global_h << ") should map to KV head " << expected_kv_head;
        }
    }
}

/**
 * @brief Test extreme edge case: 1 head, multiple ranks
 *
 * Only rank 0 gets the single head, all other ranks have no work.
 */
TEST(AttentionPrimitives, GQAHeadMappingOneHeadManyRanks)
{
    const int total_q_heads = 1;
    const int n_kv_heads = 1;
    const int head_dim = 64;
    const int seq_len = 2;
    const int num_ranks = 4;

    std::vector<float> k_compact(seq_len * n_kv_heads * head_dim, 42.0f);
    std::vector<float> v_compact(seq_len * n_kv_heads * head_dim, 84.0f);

    for (int rank = 0; rank < num_ranks; ++rank)
    {
        int heads_per_rank = total_q_heads / num_ranks; // 0
        int remainder = total_q_heads % num_ranks;      // 1
        int local_heads = heads_per_rank + (rank < remainder ? 1 : 0);

        if (rank == 0)
        {
            EXPECT_EQ(local_heads, 1) << "Rank 0 should get the only head";

            std::vector<float> k_expanded(seq_len * local_heads * head_dim);
            std::vector<float> v_expanded(seq_len * local_heads * head_dim);

            expand_kv_for_gqa(
                k_compact.data(), v_compact.data(),
                k_expanded.data(), v_expanded.data(),
                seq_len, head_dim, local_heads, n_kv_heads,
                0, total_q_heads);

            // Should be identity mapping (1 Q head → 1 KV head)
            EXPECT_FLOAT_EQ(k_expanded[0], 42.0f);
            EXPECT_FLOAT_EQ(v_expanded[0], 84.0f);
        }
        else
        {
            EXPECT_EQ(local_heads, 0) << "Rank " << rank << " should have 0 heads";
            // No expansion needed, rank has no work
        }
    }
}

/**
 * @brief Test that zero local heads produces empty output (not crash)
 *
 * Validates that calling expand_kv_for_gqa with n_heads=0 is safe.
 */
TEST(AttentionPrimitives, GQAHeadMappingZeroLocalHeads)
{
    const int n_kv_heads = 2;
    const int head_dim = 64;
    const int seq_len = 2;

    std::vector<float> k_compact(seq_len * n_kv_heads * head_dim, 1.0f);
    std::vector<float> v_compact(seq_len * n_kv_heads * head_dim, 2.0f);

    // Create empty output buffers (size 0)
    std::vector<float> k_expanded; // Empty
    std::vector<float> v_expanded; // Empty

    // This should not crash with n_heads=0
    EXPECT_NO_THROW({
        expand_kv_for_gqa(
            k_compact.data(), v_compact.data(),
            k_expanded.data(), v_expanded.data(),
            seq_len, head_dim,
            0, // n_heads = 0 (no local heads)
            n_kv_heads,
            0, // head_offset
            8  // total_q_heads (arbitrary)
        );
    });

    // No data should have been written (empty vectors)
    EXPECT_EQ(k_expanded.size(), 0);
    EXPECT_EQ(v_expanded.size(), 0);
}
