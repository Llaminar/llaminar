/**
 * @file test_attention_golden.cpp
 * @brief Golden oracle tests for attention primitives
 *
 * This test suite validates individual attention operations against known-good
 * PyTorch reference implementations. Each primitive is tested independently to
 * identify mathematical correctness issues.
 *
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include "kernels/common/attention_primitives.h"
#include "kernels/common/softmax_core.h"

namespace
{

    // Tolerance constants
    constexpr float ABS_TOL = 1e-5f;
    constexpr float REL_TOL = 1e-4f;

    /**
     * @brief Check if two floats are within absolute and relative tolerance
     */
    bool within_tolerance(float expected, float actual,
                          float abs_tol = ABS_TOL, float rel_tol = REL_TOL)
    {
        float abs_diff = std::abs(expected - actual);
        if (abs_diff <= abs_tol)
            return true;

        float rel_diff = abs_diff / std::max(std::abs(expected), std::abs(actual));
        return rel_diff <= rel_tol;
    }

    // ==================== Softmax Golden Tests ====================

    TEST(AttentionGolden, SoftmaxBasic)
    {
        // PyTorch: torch.nn.functional.softmax([1,2,3], dim=0)
        // Expected: [0.090, 0.245, 0.665]

        std::vector<float> input = {1.0f, 2.0f, 3.0f};
        std::vector<float> output = input; // Copy for in-place operation
        std::vector<float> expected = {0.09003057f, 0.24472848f, 0.66524094f};

        llaminar::kernels::SoftmaxRowArgs args;
        args.scores = output.data();
        args.rows = 1;
        args.cols = 3;
        args.causal = false;
        args.scale = 1.0f;

        llaminar::kernels::softmax_row_major(args);

        for (size_t i = 0; i < 3; ++i)
        {
            EXPECT_TRUE(within_tolerance(expected[i], output[i]));
        }
    }

    TEST(AttentionGolden, SoftmaxNegative)
    {
        // Test softmax with negative values
        std::vector<float> input = {-1.0f, 0.0f, 1.0f};
        std::vector<float> output = input;
        std::vector<float> expected = {0.09003057f, 0.24472848f, 0.66524094f};

        llaminar::kernels::SoftmaxRowArgs args;
        args.scores = output.data();
        args.rows = 1;
        args.cols = 3;
        args.causal = false;
        args.scale = 1.0f;

        llaminar::kernels::softmax_row_major(args);

        for (size_t i = 0; i < 3; ++i)
        {
            EXPECT_TRUE(within_tolerance(expected[i], output[i]));
        }
    }

    TEST(AttentionGolden, SoftmaxLargeRange)
    {
        // Test numerical stability with large range
        std::vector<float> input = {0.0f, 10.0f, 20.0f};
        std::vector<float> output = input;
        std::vector<float> expected = {2.0612e-09f, 4.5398e-05f, 0.99995f};

        llaminar::kernels::SoftmaxRowArgs args;
        args.scores = output.data();
        args.rows = 1;
        args.cols = 3;
        args.causal = false;
        args.scale = 1.0f;

        llaminar::kernels::softmax_row_major(args);

        for (size_t i = 0; i < 3; ++i)
        {
            EXPECT_TRUE(within_tolerance(expected[i], output[i]));
        }
    }

    TEST(AttentionGolden, SoftmaxAllEqual)
    {
        // Uniform distribution test
        std::vector<float> input = {1.0f, 1.0f, 1.0f, 1.0f};
        std::vector<float> output = input;
        std::vector<float> expected = {0.25f, 0.25f, 0.25f, 0.25f};

        llaminar::kernels::SoftmaxRowArgs args;
        args.scores = output.data();
        args.rows = 1;
        args.cols = 4;
        args.causal = false;
        args.scale = 1.0f;

        llaminar::kernels::softmax_row_major(args);

        for (size_t i = 0; i < 4; ++i)
        {
            EXPECT_TRUE(within_tolerance(expected[i], output[i]));
        }
    }

    // ==================== RoPE Golden Tests ====================

    TEST(AttentionGolden, RoPEBasic)
    {
        // Position 0 should be identity (no rotation)
        std::vector<float> q = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<float> k = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<float> q_expected = q;
        std::vector<float> k_expected = k;

        llaminar::attn::apply_rope(q.data(), k.data(), 1, 4, 1, 1, 0, 10000.0f);

        for (size_t i = 0; i < 4; ++i)
        {
            EXPECT_NEAR(q[i], q_expected[i], 1e-4f);
            EXPECT_NEAR(k[i], k_expected[i], 1e-4f);
        }
    }

    TEST(AttentionGolden, RoPENonZeroPosition)
    {
        // Position 1 should apply rotation
        std::vector<float> q = {1.0f, 0.0f, 1.0f, 0.0f};
        std::vector<float> k = {1.0f, 0.0f, 1.0f, 0.0f};

        llaminar::attn::apply_rope(q.data(), k.data(), 1, 4, 1, 1, 1, 10000.0f);

        // Verify values have changed (rotated)
        EXPECT_NE(q[0], 1.0f); // First element should be rotated
        EXPECT_NE(k[0], 1.0f);
    }

    // ==================== QK Scores Golden Tests ====================

    TEST(AttentionGolden, QKScoresSimple)
    {
        // Test basic QK^T computation with 1/√d scaling
        // Q=[1,0], K=[1,0], head_dim=2
        // Raw dot product = 1.0, scaled by 1/√2 = 0.7071

        std::vector<float> q = {1.0f, 0.0f};
        std::vector<float> k = {1.0f, 0.0f};
        std::vector<float> scores(1);

        llaminar::attn::compute_qk_scores(q.data(), k.data(), scores.data(),
                                          1, 1, 2, 1, false, false);

        float expected = 1.0f / std::sqrt(2.0f); // 1/√2 scaling
        EXPECT_NEAR(scores[0], expected, 1e-5f);
    }

    TEST(AttentionGolden, QKScoresTwoTokens)
    {
        // Test 2x2 attention matrix with 1/√d scaling
        // Q=[[1,0],[0,1]], K=[[1,0],[0,1]], head_dim=2
        // Expected (with 1/√2 scaling): [[0.707,0],[0,0.707]]

        std::vector<float> q = {1.0f, 0.0f, 0.0f, 1.0f};
        std::vector<float> k = {1.0f, 0.0f, 0.0f, 1.0f};
        std::vector<float> scores(4);
        float scale = 1.0f / std::sqrt(2.0f);
        std::vector<float> expected = {scale, 0.0f, 0.0f, scale};

        llaminar::attn::compute_qk_scores(q.data(), k.data(), scores.data(),
                                          2, 2, 2, 1, false, false);

        for (size_t i = 0; i < 4; ++i)
        {
            EXPECT_NEAR(scores[i], expected[i], 1e-5f);
        }
    }

    TEST(AttentionGolden, QKScoresWithSoftmax)
    {
        // Test QK^T with softmax - 2 tokens, head_dim=2
        // Check softmax properties: rows sum to 1, diagonal dominance

        std::vector<float> q = {2.0f, 0.0f, 0.0f, 2.0f};
        std::vector<float> k = {1.0f, 0.0f, 0.0f, 1.0f};
        std::vector<float> scores(4);

        llaminar::attn::compute_qk_scores(q.data(), k.data(), scores.data(),
                                          2, 2, 2, 1, false, true); // with softmax

        // Check that rows sum to 1.0 (softmax property)
        EXPECT_NEAR(scores[0] + scores[1], 1.0f, 0.001f);
        EXPECT_NEAR(scores[2] + scores[3], 1.0f, 0.001f);

        // Check diagonal dominance (same vectors have higher scores)
        EXPECT_GT(scores[0], scores[1]);
        EXPECT_GT(scores[3], scores[2]);
    }

    // ==================== Attention Output Golden Tests ====================

    TEST(AttentionGolden, ApplyScoresToV)
    {
        // Test scores @ V computation
        // scores = [[1.0, 0.0]], V = [[1, 2], [3, 4]], head_dim=2
        // Expected: output = [1.0*[1,2] + 0.0*[3,4]] = [1, 2]

        std::vector<float> scores = {1.0f, 0.0f};
        std::vector<float> v = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<float> output(2);
        std::vector<float> expected = {1.0f, 2.0f};

        llaminar::attn::apply_scores_to_v(scores.data(), v.data(), output.data(),
                                          1, 1, 2, 1);

        for (size_t i = 0; i < 2; ++i)
        {
            EXPECT_NEAR(output[i], expected[i], 1e-5f);
        }
    }

    TEST(AttentionGolden, ApplyScoresToVWeighted)
    {
        // Test weighted combination with 2 tokens
        // scores = [[0.6, 0.4], [0.3, 0.7]] (2x2 attention matrix)
        // V = [[1, 0], [0, 1]] (2 tokens, dim=2)
        // Token 0: 0.6*[1,0] + 0.4*[0,1] = [0.6, 0.4]
        // Token 1: 0.3*[1,0] + 0.7*[0,1] = [0.3, 0.7]

        std::vector<float> scores = {0.6f, 0.4f, 0.3f, 0.7f};
        std::vector<float> v = {1.0f, 0.0f, 0.0f, 1.0f};
        std::vector<float> output(4); // 2 tokens * 2 dim
        std::vector<float> expected = {0.6f, 0.4f, 0.3f, 0.7f};

        llaminar::attn::apply_scores_to_v(scores.data(), v.data(), output.data(),
                                          2, 2, 2, 1); // seq_len=2

        for (size_t i = 0; i < 4; ++i)
        {
            EXPECT_NEAR(output[i], expected[i], 1e-5f);
        }
    }

    // ==================== End-to-End Attention Golden Test ====================

    TEST(AttentionGolden, FusedAttentionSimple)
    {
        // End-to-end fused attention test with causal masking
        // Q=[[1,0],[0,1]], K=[[1,0],[0,1]], V=[[2,0],[0,3]], head_dim=2

        std::vector<float> q = {1.0f, 0.0f, 0.0f, 1.0f}; // 2 tokens
        std::vector<float> k = {1.0f, 0.0f, 0.0f, 1.0f}; // 2 tokens
        std::vector<float> v = {2.0f, 0.0f, 0.0f, 3.0f}; // 2 tokens
        std::vector<float> output(4);                    // 2 tokens * 2 dim

        llaminar::attn::fused_attention(q.data(), k.data(), v.data(), output.data(),
                                        2, 2, 1, true); // seq_len=2, head_dim=2, heads=1, causal=true

        // Token 0 (causal): can only attend to itself -> should output V[0] = [2,0]
        EXPECT_NEAR(output[0], 2.0f, 0.1f);
        EXPECT_NEAR(output[1], 0.0f, 0.1f);

        // Token 1 (causal): attends to tokens 0 and 1 (weighted combination)
        EXPECT_GT(output[2], 0.0f); // Should have some contribution
        EXPECT_GT(output[3], 0.0f); // Should have some contribution
    }

} // anonymous namespace

// TODO: Add more comprehensive tests:
// - Multi-head attention
// - Longer sequences (seq_len > 2)
// - Numerical stability edge cases
