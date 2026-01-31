/**
 * @file Test__AttentionMaskApplication.cpp
 * @brief Unit tests for attention mask application in CPUAttentionKernelT
 * @author David Sanftenberg
 *
 * Tests that verify attention masks are correctly applied to attention scores
 * before softmax, ensuring masked positions get zero probability after softmax.
 */

#include <gtest/gtest.h>
#include "v2/kernels/cpu/attention/CPUAttentionKernelT.h"
#include "v2/tensors/Tensors.h"
#include "v2/kernels/cpu/attention/AttentionUtils.h"
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

using namespace llaminar2;

namespace
{
    constexpr float NEG_INF = -std::numeric_limits<float>::infinity();
    constexpr float EPSILON = 1e-5f;

    // Helper: Check if value is close to zero (after softmax of masked position)
    bool is_near_zero(float value, float epsilon = EPSILON)
    {
        return std::abs(value) < epsilon;
    }

    // Helper: Check if mask value is masked (close to -inf)
    bool is_masked(float value)
    {
        return value < -1e10f;
    }

    // Helper: Initialize array with sequential pattern
    void init_sequential(float *data, size_t size, float offset = 0.0f)
    {
        for (size_t i = 0; i < size; ++i)
        {
            data[i] = offset + static_cast<float>(i % 100) / 100.0f;
        }
    }
}

class Test__AttentionMaskApplication : public ::testing::Test
{
protected:
    void SetUp() override
    {
    }
};

// ============================================================================
// Test 1: Single Head, Simple Padding Mask
// ============================================================================

TEST_F(Test__AttentionMaskApplication, SingleHead_SimplePaddingMask)
{
    // Setup: 1 head, seq_len=4, head_dim=8
    // Sequence: [token0, token1, PAD, PAD]
    // Expected: Attention weights for PAD positions should be ~0

    const int n_heads = 1;
    const int seq_len = 4;
    const int head_dim = 8;
    const int valid_tokens = 2; // First 2 tokens are valid

    // Create Q, K, V tensors
    std::vector<float> Q(seq_len * n_heads * head_dim);
    std::vector<float> K(seq_len * n_heads * head_dim);
    std::vector<float> V(seq_len * n_heads * head_dim);
    std::vector<float> output(seq_len * n_heads * head_dim, 0.0f);

    // Initialize Q, K, V with distinct patterns
    init_sequential(Q.data(), Q.size(), 0.0f);
    init_sequential(K.data(), K.size(), 10.0f);
    init_sequential(V.data(), V.size(), 20.0f);

    // Create padding mask: [4, 4] where rows 2,3 and cols 2,3 are masked
    FP32Tensor mask_tensor(std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(seq_len)});
    float *mask_data = mask_tensor.mutable_data();

    // Mask out padding positions (tokens 2, 3)
    for (int i = 0; i < seq_len; ++i)
    {
        // Can't attend TO padding tokens
        mask_data[i * seq_len + 2] = NEG_INF;
        mask_data[i * seq_len + 3] = NEG_INF;

        // Padding tokens can't attend FROM anything
        if (i >= valid_tokens)
        {
            for (int j = 0; j < seq_len; ++j)
            {
                mask_data[i * seq_len + j] = NEG_INF;
            }
        }
    }

    // Create attention kernel
    CPUAttentionKernelT<ActivationPrecision::FP32> attention;

    // Compute attention
    bool success = attention.compute(
        Q.data(), K.data(), V.data(), output.data(),
        seq_len, n_heads, n_heads, head_dim,
        false,                                  // causal
        -1,                                     // window_size
        nullptr, nullptr, nullptr, &mask_tensor // workspace_mask
    );

    ASSERT_TRUE(success) << "Attention computation failed";

    // Verify: Output for padding tokens (rows 2,3) should be all zeros or near-zero
    for (int i = valid_tokens; i < seq_len; ++i)
    {
        for (int j = 0; j < head_dim; ++j)
        {
            EXPECT_TRUE(is_near_zero(output[i * head_dim + j]))
                << "Padding token " << i << " dim " << j
                << " should be ~0, got " << output[i * head_dim + j];
        }
    }

    // Verify: Valid tokens (rows 0,1) should have non-zero output
    bool has_nonzero = false;
    for (int i = 0; i < valid_tokens; ++i)
    {
        for (int j = 0; j < head_dim; ++j)
        {
            if (std::abs(output[i * head_dim + j]) > EPSILON)
            {
                has_nonzero = true;
                break;
            }
        }
    }
    EXPECT_TRUE(has_nonzero) << "Valid tokens should have non-zero output";
}

// ============================================================================
// Test 2: Mask Application Utility Function
// ============================================================================

TEST_F(Test__AttentionMaskApplication, MaskApplicationUtility)
{
    // Test the attention_utils::apply_attention_mask function directly

    const int rows = 3;
    const int cols = 3;

    // Create scores (attention logits before softmax)
    std::vector<float> scores(rows * cols, 2.0f); // All 2.0 initially

    // Create mask with some positions masked
    std::vector<float> mask(rows * cols, 0.0f);
    mask[2] = NEG_INF; // Row 0, col 2
    mask[5] = NEG_INF; // Row 1, col 2
    mask[6] = NEG_INF; // Row 2, col 0
    mask[7] = NEG_INF; // Row 2, col 1
    mask[8] = NEG_INF; // Row 2, col 2

    // Apply mask
    attention_utils::apply_attention_mask(scores.data(), mask.data(), rows, cols);

    // Verify masked positions are now -inf
    EXPECT_TRUE(is_masked(scores[2])) << "Position (0,2) should be masked";
    EXPECT_TRUE(is_masked(scores[5])) << "Position (1,2) should be masked";
    EXPECT_TRUE(is_masked(scores[6])) << "Position (2,0) should be masked";
    EXPECT_TRUE(is_masked(scores[7])) << "Position (2,1) should be masked";
    EXPECT_TRUE(is_masked(scores[8])) << "Position (2,2) should be masked";

    // Verify unmasked positions are still 2.0
    EXPECT_NEAR(scores[0], 2.0f, EPSILON);
    EXPECT_NEAR(scores[1], 2.0f, EPSILON);
    EXPECT_NEAR(scores[3], 2.0f, EPSILON);
    EXPECT_NEAR(scores[4], 2.0f, EPSILON);
}

// ============================================================================
// Test 3: Batch Attention with Per-Sequence Masks
// ============================================================================

TEST_F(Test__AttentionMaskApplication, BatchAttention_PerSequenceMasks)
{
    // Batch of 2 sequences: [{token0, PAD}, {token0, token1}]
    // Combined batch has 4 tokens total: [seq0_tok0, seq0_PAD, seq1_tok0, seq1_tok1]

    const int batch_size = 2;
    const int seq_len = 2;
    const int n_heads = 1;
    const int head_dim = 4;
    const int total_tokens = batch_size * seq_len;
    std::vector<int> actual_lengths = {1, 2}; // Seq0: 1 token, Seq1: 2 tokens

    std::vector<float> Q(total_tokens * n_heads * head_dim);
    std::vector<float> K(total_tokens * n_heads * head_dim);
    std::vector<float> V(total_tokens * n_heads * head_dim);
    std::vector<float> output(total_tokens * n_heads * head_dim, 0.0f);

    init_sequential(Q.data(), Q.size(), 0.0f);
    init_sequential(K.data(), K.size(), 10.0f);
    init_sequential(V.data(), V.size(), 20.0f);

    // Create combined batch mask
    FP32Tensor mask_tensor(std::vector<size_t>{static_cast<size_t>(total_tokens), static_cast<size_t>(total_tokens)});
    float *mask_data = mask_tensor.mutable_data();

    attention_utils::create_combined_batch_mask(
        mask_data, batch_size, seq_len, actual_lengths.data(),
        /*causal=*/false, /*window_size=*/-1);

    // Create attention kernel and compute
    CPUAttentionKernelT<ActivationPrecision::FP32> attention;

    bool success = attention.compute_batch(
        Q.data(), K.data(), V.data(), output.data(),
        batch_size, seq_len, n_heads, n_heads, head_dim,
        false,                                  // causal
        -1,                                     // window_size
        nullptr, nullptr, nullptr, &mask_tensor // workspace_mask
    );

    ASSERT_TRUE(success);

    // Verify: Token 1 (seq0 padding) should be all zeros
    for (int j = 0; j < head_dim; ++j)
    {
        EXPECT_TRUE(is_near_zero(output[1 * head_dim + j]))
            << "Seq0 padding token dim " << j << " should be ~0, got " << output[1 * head_dim + j];
    }

    // Verify: Tokens 0, 2, 3 should have non-zero values
    bool token0_nonzero = false;
    bool token2_nonzero = false;
    bool token3_nonzero = false;

    for (int j = 0; j < head_dim; ++j)
    {
        if (std::abs(output[0 * head_dim + j]) > EPSILON)
            token0_nonzero = true;
        if (std::abs(output[2 * head_dim + j]) > EPSILON)
            token2_nonzero = true;
        if (std::abs(output[3 * head_dim + j]) > EPSILON)
            token3_nonzero = true;
    }

    EXPECT_TRUE(token0_nonzero) << "Token 0 (valid) should have non-zero output";
    EXPECT_TRUE(token2_nonzero) << "Token 2 (valid) should have non-zero output";
    EXPECT_TRUE(token3_nonzero) << "Token 3 (valid) should have non-zero output";
}
