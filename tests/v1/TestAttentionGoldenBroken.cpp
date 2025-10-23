/**
 * @file TestAttentionGolden.cpp
 * @brief Golden oracle tests for attention primitive operations
 * @author David Sanftenberg
 * 
 * This test suite validates individual attention operations against known-good
 * reference implementations. Each primitive (RoPE, softmax, attention scores, etc.)
 * is tested in isolation with PyTorch-generated golden reference data.
 * 
 * Test Strategy:
 * 1. Generate reference outputs using PyTorch (or other trusted implementation)
 * 2. Run same inputs through Llaminar primitives
 * 3. Compare outputs with strict tolerances
 * 4. Test edge cases (zeros, large values, boundary conditions)
 */

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <random>
#include <iostream>
#include <iomanip>
#include "operators/common/AttentionPrimitives.h"
#include "operators/common/SoftmaxCore.h"

namespace {

// Tolerance constants
constexpr float ABS_TOL = 1e-5f;
constexpr float REL_TOL = 1e-4f;

// Utility: Check if two values are within tolerance
bool within_tolerance(float expected, float actual, float abs_tol = ABS_TOL, float rel_tol = REL_TOL) {
    float abs_diff = std::fabs(expected - actual);
    if (abs_diff <= abs_tol) return true;
    
    float denom = std::max(std::fabs(expected), 1e-6f);
    float rel_diff = abs_diff / denom;
    return rel_diff <= rel_tol;
}

// Utility: Print top differences for debugging
void print_top_diffs(const std::vector<float>& expected, const std::vector<float>& actual, int top_k = 5) {
    struct Diff {
        size_t idx;
        float abs_diff;
        float expected_val;
        float actual_val;
    };
    
    std::vector<Diff> diffs;
    for (size_t i = 0; i < expected.size(); ++i) {
        diffs.push_back({i, std::fabs(expected[i] - actual[i]), expected[i], actual[i]});
    }
    
    std::partial_sort(diffs.begin(), diffs.begin() + std::min((size_t)top_k, diffs.size()), diffs.end(),
                     [](const Diff& a, const Diff& b) { return a.abs_diff > b.abs_diff; });
    
    std::cout << "Top " << top_k << " differences:\n";
    for (int i = 0; i < std::min(top_k, (int)diffs.size()); ++i) {
        std::cout << "  [" << diffs[i].idx << "] diff=" << diffs[i].abs_diff 
                  << " expected=" << diffs[i].expected_val 
                  << " actual=" << diffs[i].actual_val << "\n";
    }
}

// ==================== Softmax Golden Tests ====================

TEST(AttentionGolden, SoftmaxBasic) {
    // Test case: [1, 2, 3] -> softmax
    // PyTorch reference:
    // >>> import torch
    // >>> x = torch.tensor([1.0, 2.0, 3.0])
    // >>> torch.nn.functional.softmax(x, dim=0)
    // tensor([0.0900, 0.2447, 0.6652])
    
    std::vector<float> input = {1.0f, 2.0f, 3.0f};
    std::vector<float> expected = {0.09003057f, 0.24472848f, 0.66524094f};
    std::vector<float> output = input;  // Copy input since softmax_row_major is in-place
    
    llaminar::kernels::SoftmaxRowArgs args;
    args.scores = output.data();
    args.rows = 1;
    args.cols = 3;
    args.causal = false;
    args.scale = 1.0f;
    llaminar::kernels::softmax_row_major(args);
    
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_TRUE(within_tolerance(expected[i], output[i])) 
            << "Mismatch at index " << i << ": expected=" << expected[i] << " actual=" << output[i];
    }
}

TEST(AttentionGolden, SoftmaxNegative) {
    // Test with negative values
    // >>> x = torch.tensor([-1.0, -2.0, -3.0])
    // >>> torch.nn.functional.softmax(x, dim=0)
    // tensor([0.6652, 0.2447, 0.0900])
    
    std::vector<float> input = {-1.0f, -2.0f, -3.0f};
    std::vector<float> expected = {0.66524094f, 0.24472848f, 0.09003057f};
    std::vector<float> output = input;
    
    llaminar::kernels::SoftmaxRowArgs args;
    args.scores = output.data();
    args.rows = 1;
    args.cols = 3;
    args.causal = false;
    args.scale = 1.0f;
    llaminar::kernels::softmax_row_major(args);
    
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_TRUE(within_tolerance(expected[i], output[i]));
    }
}

TEST(AttentionGolden, SoftmaxLargeRange) {
    // Test numerical stability with large range
    // >>> x = torch.tensor([0.0, 10.0, 20.0])
    // >>> torch.nn.functional.softmax(x, dim=0)
    // tensor([2.0612e-09, 4.5398e-05, 9.9995e-01])
    
    std::vector<float> input = {0.0f, 10.0f, 20.0f};
    std::vector<float> expected = {2.0612e-09f, 4.5398e-05f, 0.99995460f};
    std::vector<float> output = input;
    
    llaminar::kernels::SoftmaxRowArgs args;
    args.scores = output.data();
    args.rows = 1;
    args.cols = 3;
    args.causal = false;
    args.scale = 1.0f;
    llaminar::kernels::softmax_row_major(args);
    
    // Use looser tolerance for very small values
    EXPECT_TRUE(within_tolerance(expected[0], output[0], 1e-8f, 0.1f));
    EXPECT_TRUE(within_tolerance(expected[1], output[1], 1e-5f, 0.01f));
    EXPECT_TRUE(within_tolerance(expected[2], output[2]));
}

TEST(AttentionGolden, SoftmaxAllEqual) {
    // All equal values should give uniform distribution
    // >>> x = torch.tensor([1.0, 1.0, 1.0, 1.0])
    // >>> torch.nn.functional.softmax(x, dim=0)
    // tensor([0.2500, 0.2500, 0.2500, 0.2500])
    
    std::vector<float> input = {1.0f, 1.0f, 1.0f, 1.0f};
    std::vector<float> expected = {0.25f, 0.25f, 0.25f, 0.25f};
    std::vector<float> output = input;
    
    llaminar::kernels::SoftmaxRowArgs args;
    args.scores = output.data();
    args.rows = 1;
    args.cols = 4;
    args.causal = false;
    args.scale = 1.0f;
    llaminar::kernels::softmax_row_major(args);
    
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_TRUE(within_tolerance(expected[i], output[i]));
    }
}

// ==================== RoPE Golden Tests ====================

TEST(AttentionGolden, RoPEBasic) {
    // Test RoPE on small vectors
    // Reference: PyTorch implementation of RoPE
    // Input: q=[1,0,1,0], k=[0,1,0,1], head_dim=4, seq_len=1, n_past=0
    
    // For theta=10000, freq_0 = 1/10000^(0/4) = 1.0, freq_1 = 1/10000^(2/4) = 0.01
    // pos=0: cos_0=1.0, sin_0=0.0, cos_1=1.0, sin_1=0.0
    // RoPE rotation: [q0, q1] -> [q0*cos - q1*sin, q0*sin + q1*cos]
    //                [1, 0] -> [1*1 - 0*0, 1*0 + 0*1] = [1, 0]
    //                [1, 0] -> [1*1 - 0*0, 1*0 + 0*1] = [1, 0]
    
    std::vector<float> q = {1.0f, 0.0f, 1.0f, 0.0f};
    std::vector<float> k = {0.0f, 1.0f, 0.0f, 1.0f};
    std::vector<float> q_expected = {1.0f, 0.0f, 1.0f, 0.0f};  // No change at position 0
    std::vector<float> k_expected = {0.0f, 1.0f, 0.0f, 1.0f};
    
    llaminar::attn::apply_rope(q.data(), k.data(), 1, 4, 1, 1, 0, 10000.0f);
    
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(q[i], q_expected[i], 1e-5f);
        EXPECT_NEAR(k[i], k_expected[i], 1e-5f);
    }
}

TEST(AttentionGolden, RoPENonZeroPosition) {
    // Test RoPE with n_past > 0
    // TODO: Add PyTorch reference values for specific position
    
    std::vector<float> q = {1.0f, 0.0f, 0.0f, 1.0f};
    std::vector<float> k = {1.0f, 0.0f, 0.0f, 1.0f};
    
    // At position 1 with freq_base=10000:
    // freq[0] = 1.0, cos(1*1.0)=0.5403, sin(1*1.0)=0.8415
    // freq[1] = 0.01, cos(1*0.01)≈1.0, sin(1*0.01)≈0.01
    
    // Expected (approximate, needs exact PyTorch values):
    // q[0:2] rotation: [1,0] -> [1*0.5403 - 0*0.8415, 1*0.8415 + 0*0.5403] = [0.5403, 0.8415]
    // q[2:4] rotation: [0,1] -> [0*1.0 - 1*0.01, 0*0.01 + 1*1.0] = [-0.01, 1.0]
    
    llaminar::attn::apply_rope(q.data(), k.data(), 1, 4, 1, 1, 1, 10000.0f);
    
    // These are approximate - need exact PyTorch reference
    EXPECT_NEAR(q[0], 0.5403f, 0.001f);
    EXPECT_NEAR(q[1], 0.8415f, 0.001f);
    EXPECT_NEAR(q[2], -0.01f, 0.001f);
    EXPECT_NEAR(q[3], 1.0f, 0.001f);
}

// ==================== Attention Scores Golden Tests ====================

TEST(AttentionGolden, QKScoresSimple) {
    // Test basic QK^T computation with 1/√d scaling
    // Q=[1,0], K=[1,0], head_dim=2
    // Raw dot product = 1.0, scaled by 1/√2 = 0.7071
    
    std::vector<float> q = {1.0f, 0.0f};
    std::vector<float> k = {1.0f, 0.0f};
    std::vector<float> scores(1);
    
    llaminar::attn::compute_qk_scores(q.data(), k.data(), scores.data(), 
                                      1, 2, 1, false, false);
    
    float expected = 1.0f / std::sqrt(2.0f);  // 1/√2 scaling
    EXPECT_NEAR(scores[0], expected, 1e-5f);
}

TEST(AttentionGolden, QKScoresTwoTokens) {
    // Test 2x2 attention matrix with 1/√d scaling
    // Q=[[1,0],[0,1]], K=[[1,0],[0,1]], head_dim=2
    // Expected (with 1/√2 scaling): [[0.707,0],[0,0.707]]
    
    std::vector<float> q = {1.0f, 0.0f, 0.0f, 1.0f};
    std::vector<float> k = {1.0f, 0.0f, 0.0f, 1.0f};
    std::vector<float> scores(4);
    float scale = 1.0f / std::sqrt(2.0f);
    std::vector<float> expected = {scale, 0.0f, 0.0f, scale};
    
    llaminar::attn::compute_qk_scores(q.data(), k.data(), scores.data(), 
                                      2, 2, 1, false, false);
    
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(scores[i], expected[i], 1e-5f);
    }
}

TEST(AttentionGolden, QKScoresWithSoftmax) {
    // Test QK^T with softmax - 2 tokens, head_dim=2
    // Q=[[2,0],[0,2]], K=[[1,0],[0,1]]
    // Raw scores (with 1/√2): [[√2,0],[0,√2]]
    // After softmax: [[0.731,0.269],[0.269,0.731]]
    
    std::vector<float> q = {2.0f, 0.0f, 0.0f, 2.0f};
    std::vector<float> k = {1.0f, 0.0f, 0.0f, 1.0f};
    std::vector<float> scores(4);
    
    llaminar::attn::compute_qk_scores(q.data(), k.data(), scores.data(), 
                                      2, 2, 1, false, true);  // with softmax
    
    // Check that rows sum to 1.0 (softmax property)
    EXPECT_NEAR(scores[0] + scores[1], 1.0f, 0.001f);
    EXPECT_NEAR(scores[2] + scores[3], 1.0f, 0.001f);
    
    // Check diagonal dominance (same vectors have higher scores)
    EXPECT_GT(scores[0], scores[1]);
    EXPECT_GT(scores[3], scores[2]);
}

// ==================== Attention Output Golden Tests ====================

TEST(AttentionGolden, ApplyScoresToV) {
    // Test scores @ V computation
    // scores = [[1.0, 0.0]], V = [[1, 2], [3, 4]], head_dim=2
    // Expected: output = [1.0*[1,2] + 0.0*[3,4]] = [1, 2]
    
    std::vector<float> scores = {1.0f, 0.0f};
    std::vector<float> v = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> output(2);
    std::vector<float> expected = {1.0f, 2.0f};
    
    llaminar::attn::apply_scores_to_v(scores.data(), v.data(), output.data(), 
                                      1, 2, 1);
    
    for (size_t i = 0; i < 2; ++i) {
        EXPECT_NEAR(output[i], expected[i], 1e-5f);
    }
}

TEST(AttentionGolden, ApplyScoresToVWeighted) {
    // Test weighted combination with 2 tokens
    // scores = [[0.6, 0.4], [0.3, 0.7]] (2x2 attention matrix)
    // V = [[1, 0], [0, 1]] (2 tokens, dim=2)
    // Token 0: 0.6*[1,0] + 0.4*[0,1] = [0.6, 0.4]
    // Token 1: 0.3*[1,0] + 0.7*[0,1] = [0.3, 0.7]
    
    std::vector<float> scores = {0.6f, 0.4f, 0.3f, 0.7f};
    std::vector<float> v = {1.0f, 0.0f, 0.0f, 1.0f};
    std::vector<float> output(4);  // 2 tokens * 2 dim
    std::vector<float> expected = {0.6f, 0.4f, 0.3f, 0.7f};
    
    llaminar::attn::apply_scores_to_v(scores.data(), v.data(), output.data(), 
                                      2, 2, 1);  // seq_len=2
    
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(output[i], expected[i], 1e-5f);
    }
}

// ==================== End-to-End Attention Golden Test ====================

TEST(AttentionGolden, FusedAttentionSimple) {
    // Test full attention: softmax(Q @ K^T) @ V
    // Q = [[1, 0]], K = [[1, 0], [0, 1]], V = [[2, 0], [0, 3]]
    
    std::vector<float> q = {1.0f, 0.0f, 0.0f, 1.0f};  // 2 tokens
    std::vector<float> k = {1.0f, 0.0f, 0.0f, 1.0f};  // 2 tokens
    std::vector<float> v = {2.0f, 0.0f, 0.0f, 3.0f};  // 2 tokens
    std::vector<float> output(4);  // 2 tokens * 2 dim
    
    llaminar::attn::fused_attention(q.data(), k.data(), v.data(), output.data(), 
                                    2, 2, 1, true);  // seq_len=2, head_dim=2, heads=1, causal=true
    
    // Token 0 (causal): can only attend to itself -> should output V[0] = [2,0]
    EXPECT_NEAR(output[0], 2.0f, 0.1f);
    EXPECT_NEAR(output[1], 0.0f, 0.1f);
    
    // Token 1 (causal): attends to tokens 0 and 1 (weighted combination)
    EXPECT_GT(output[2], 0.0f);  // Should have some contribution
    EXPECT_GT(output[3], 0.0f);  // Should have some contribution
}

} // anonymous namespace

// TODO: Add more comprehensive tests:
// - Multi-head attention
// - Longer sequences (seq_len > 2)
// - Causal masking validation
// - Numerical stability edge cases
// - Compare against full PyTorch implementation
