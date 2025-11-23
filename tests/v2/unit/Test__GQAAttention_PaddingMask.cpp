/**
 * @file Test__GQAAttention_PaddingMask.cpp
 * @brief Comprehensive unit tests for GQAAttention padding mask behavior
 * @author David Sanftenberg
 *
 * These tests focus on verifying that padding masks are correctly applied
 * in batched attention, ensuring padded positions do not contaminate
 * attention scores for real tokens.
 *
 * Test Categories:
 * 1. Padding mask construction and values
 * 2. Padding mask application in attention scores
 * 3. Isolated sequence attention (no cross-contamination)
 * 4. Variable-length batch processing
 * 5. Padding + causal mask interaction
 */

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <memory>
#include <numeric>
#include <algorithm>
#include <cstring>

#include "v2/pipelines/attention/GQAAttention.h"
#include "v2/pipelines/AttentionUtils.h"
#include "v2/tensors/Tensors.h"
#include "v2/utils/MPIContext.h"
#include "v2/utils/Logger.h"

using namespace llaminar2;

namespace
{
    constexpr float FP32_TOLERANCE = 1e-4f;
    constexpr float MASK_NEG_INF = -1e10f;

    // Helper: Create FP32 tensor
    std::unique_ptr<FP32Tensor> create_fp32_tensor(size_t rows, size_t cols)
    {
        return std::make_unique<FP32Tensor>(std::vector<size_t>{rows, cols}, 0);
    }

    // Helper: Initialize tensor with constant value
    void fill_constant(float *data, int count, float value)
    {
        std::fill(data, data + count, value);
    }

    // Helper: Initialize with small random-like values (deterministic)
    void init_random_like(float *data, int count, int seed = 42)
    {
        for (int i = 0; i < count; ++i)
        {
            int val = (seed * 1103515245 + i * 12345) & 0x7fffffff;
            data[i] = static_cast<float>(val % 1000) / 1000.0f - 0.5f;
        }
    }

    // Helper: Check if value is effectively -inf (masked)
    bool is_masked(float value)
    {
        return value < -1e9f;
    }

    // Helper: Print first N values of tensor for debugging
    void print_tensor_sample(const char *label, const float *data, int count, int max_print = 10)
    {
        std::stringstream ss;
        ss << label << ": [";
        for (int i = 0; i < std::min(count, max_print); ++i)
        {
            if (i > 0)
                ss << ", ";
            ss << data[i];
        }
        if (count > max_print)
            ss << ", ...";
        ss << "]";
        LOG_ERROR(ss.str());
    }

    // Helper: Compare tensors with detailed error reporting
    struct ComparisonResult
    {
        bool equal;
        float max_abs_diff;
        float mean_abs_diff;
        int first_mismatch_idx;
        float first_expected;
        float first_actual;
        int mismatch_count;
    };

    ComparisonResult compare_tensors(const float *expected, const float *actual,
                                     int count, float tolerance)
    {
        ComparisonResult result;
        result.equal = true;
        result.max_abs_diff = 0.0f;
        result.mean_abs_diff = 0.0f;
        result.first_mismatch_idx = -1;
        result.mismatch_count = 0;

        double sum_abs_diff = 0.0;
        for (int i = 0; i < count; ++i)
        {
            float diff = std::abs(expected[i] - actual[i]);
            sum_abs_diff += diff;

            if (diff > result.max_abs_diff)
            {
                result.max_abs_diff = diff;
            }

            if (diff > tolerance)
            {
                result.mismatch_count++;
                if (result.first_mismatch_idx == -1)
                {
                    result.first_mismatch_idx = i;
                    result.first_expected = expected[i];
                    result.first_actual = actual[i];
                    result.equal = false;
                }
            }
        }

        result.mean_abs_diff = static_cast<float>(sum_abs_diff / count);
        return result;
    }
}

//==============================================================================
// Test 1: Verify mask tensor construction for variable-length sequences
//==============================================================================
TEST(Test__GQAAttention_PaddingMask, MaskConstruction_VariableLengths)
{
    // Setup: Batch of 2 sequences, seq_len=4
    // Seq0: length=4 (no padding)
    // Seq1: length=2 (2 padding tokens)
    const int batch_size = 2;
    const int seq_len = 4;
    const int n_heads = 4;
    const int n_kv_heads = 4;
    const int head_dim = 64;

    std::vector<int> actual_lengths = {4, 2};

    // Create mask workspace - must be [total_len, total_len] for batch attention
    const int total_len = batch_size * seq_len;
    auto mask_tensor = create_fp32_tensor(total_len, total_len);
    float *mask_data = mask_tensor->mutable_data();

    // Build padding mask (non-causal)
    GQAAttentionConfig config;
    config.n_heads = n_heads;
    config.n_kv_heads = n_kv_heads;
    config.head_dim = head_dim;
    config.causal = false; // Non-causal to isolate padding mask
    config.window_size = 0;
    config.workspace_mask = std::move(mask_tensor);

    // Manually build mask using attention_utils
    attention_utils::create_batch_padding_mask(
        config.workspace_mask->mutable_data(),
        batch_size,
        seq_len,
        actual_lengths.data(),
        config.window_size);

    // Verify mask structure
    const float *mask = config.workspace_mask->data();

    // Seq0 (length=4, positions [0-3]): All positions should be unmasked (0.0) within seq0
    // But masked from attending to seq1 (block diagonal)
    for (int i = 0; i < 4; ++i) // Query positions in seq0
    {
        for (int j = 0; j < 4; ++j) // Key positions in seq0
        {
            float mask_val = mask[i * total_len + j];
            EXPECT_FALSE(is_masked(mask_val))
                << "Seq0 position (" << i << "," << j << ") should be unmasked, got " << mask_val;
        }
    }

    // Seq1 (length=2, positions [4-7]):
    // - Positions [4-5] (real tokens) should attend to [4-5] only
    // - Positions [6-7] (padding) should be fully masked
    const int seq1_offset = seq_len; // First token of seq1 is at position 4

    // Real token at position 4 (first token of Seq1)
    EXPECT_FALSE(is_masked(mask[seq1_offset * total_len + seq1_offset]))
        << "Seq1[0] should attend to position 0 (itself)";
    EXPECT_FALSE(is_masked(mask[seq1_offset * total_len + seq1_offset + 1]))
        << "Seq1[0] should attend to position 1";
    EXPECT_TRUE(is_masked(mask[seq1_offset * total_len + seq1_offset + 2]))
        << "Seq1[0] should NOT attend to position 2 (padding)";
    EXPECT_TRUE(is_masked(mask[seq1_offset * total_len + seq1_offset + 3]))
        << "Seq1[0] should NOT attend to position 3 (padding)";

    // Real token at position 5 (second token of Seq1)
    EXPECT_FALSE(is_masked(mask[(seq1_offset + 1) * total_len + seq1_offset]))
        << "Seq1[1] should attend to position 0";
    EXPECT_FALSE(is_masked(mask[(seq1_offset + 1) * total_len + seq1_offset + 1]))
        << "Seq1[1] should attend to position 1 (itself)";
    EXPECT_TRUE(is_masked(mask[(seq1_offset + 1) * total_len + seq1_offset + 2]))
        << "Seq1[1] should NOT attend to position 2 (padding)";
    EXPECT_TRUE(is_masked(mask[(seq1_offset + 1) * total_len + seq1_offset + 3]))
        << "Seq1[1] should NOT attend to position 3 (padding)";

    // Padding tokens at positions 6-7 should have all-masked rows
    for (int i = 2; i < 4; ++i) // Padding positions in Seq1
    {
        for (int j = 0; j < total_len; ++j) // All columns
        {
            EXPECT_TRUE(is_masked(mask[(seq1_offset + i) * total_len + j]))
                << "Padding position " << (seq1_offset + i) << " row should be fully masked";
        }
    }
}

//==============================================================================
// Test 2: Verify causal + padding mask interaction
//==============================================================================
TEST(Test__GQAAttention_PaddingMask, CausalPaddingMaskInteraction)
{
    // Setup: Same as above but with causal masking enabled
    const int batch_size = 2;
    const int seq_len = 4;
    const int n_heads = 4;
    const int n_kv_heads = 4;
    const int head_dim = 64;

    std::vector<int> actual_lengths = {4, 2};

    const int total_len = batch_size * seq_len;
    auto mask_tensor = create_fp32_tensor(total_len, total_len);

    GQAAttentionConfig config;
    config.n_heads = n_heads;
    config.n_kv_heads = n_kv_heads;
    config.head_dim = head_dim;
    config.causal = true; // Causal + padding
    config.window_size = 0;
    config.workspace_mask = std::move(mask_tensor);

    attention_utils::create_combined_batch_mask(
        config.workspace_mask->mutable_data(),
        batch_size,
        seq_len,
        actual_lengths.data(),
        config.causal,
        config.window_size);

    const float *mask = config.workspace_mask->data();
    const int seq1_offset = seq_len;

    // Seq1 position 4 (first real token): Should only attend to itself (causal) within seq1
    EXPECT_FALSE(is_masked(mask[seq1_offset * total_len + seq1_offset]))
        << "Seq1[0] should attend to position 0 (itself)";
    EXPECT_TRUE(is_masked(mask[seq1_offset * total_len + seq1_offset + 1]))
        << "Seq1[0] should NOT attend to position 1 (future, causal mask)";
    EXPECT_TRUE(is_masked(mask[seq1_offset * total_len + seq1_offset + 2]))
        << "Seq1[0] should NOT attend to position 2 (padding)";
    EXPECT_TRUE(is_masked(mask[seq1_offset * total_len + seq1_offset + 3]))
        << "Seq1[0] should NOT attend to position 3 (padding)";

    // Seq1 position 5 (second real token): Should attend to [0-1] (causal), not [2-3] (padding)
    EXPECT_FALSE(is_masked(mask[(seq1_offset + 1) * total_len + seq1_offset]))
        << "Seq1[1] should attend to position 0 (past)";
    EXPECT_FALSE(is_masked(mask[(seq1_offset + 1) * total_len + seq1_offset + 1]))
        << "Seq1[1] should attend to position 1 (itself)";
    EXPECT_TRUE(is_masked(mask[(seq1_offset + 1) * total_len + seq1_offset + 2]))
        << "Seq1[1] should NOT attend to position 2 (padding)";
    EXPECT_TRUE(is_masked(mask[(seq1_offset + 1) * total_len + seq1_offset + 3]))
        << "Seq1[1] should NOT attend to position 3 (padding)";
}

//==============================================================================
// Test 3: Verify padded positions produce zero output
//==============================================================================
TEST(Test__GQAAttention_PaddingMask, PaddedPositionsProduceZeroOutput)
{
    // Setup: Batch with padding, verify padded positions don't affect output
    const int batch_size = 2;
    const int seq_len = 4;
    const int n_heads = 2;
    const int n_kv_heads = 2;
    const int head_dim = 16;
    const int d_model = n_heads * head_dim;
    const int total_len = batch_size * seq_len;

    std::vector<int> actual_lengths = {4, 2}; // Seq1 has 2 padding tokens

    // Create input tensors
    auto Q = create_fp32_tensor(total_len, d_model);
    auto K = create_fp32_tensor(total_len, d_model);
    auto V = create_fp32_tensor(total_len, d_model);
    auto output = create_fp32_tensor(total_len, d_model);

    // Initialize with small random values
    init_random_like(Q->mutable_data(), total_len * d_model, 42);
    init_random_like(K->mutable_data(), total_len * d_model, 43);
    init_random_like(V->mutable_data(), total_len * d_model, 44);

    // Zero out the output
    fill_constant(output->mutable_data(), total_len * d_model, 0.0f);

    // Create workspace tensors - must be square [total_len, total_len] for batch attention
    auto scores_ws = create_fp32_tensor(total_len, total_len);
    auto qkv_ws = create_fp32_tensor(total_len, d_model);
    auto context_ws = create_fp32_tensor(total_len, d_model);
    auto mask_ws = create_fp32_tensor(total_len, total_len);

    // Configure attention
    GQAAttentionConfig config;
    config.n_heads = n_heads;
    config.n_kv_heads = n_kv_heads;
    config.head_dim = head_dim;
    config.causal = false;
    config.window_size = 0;
    config.workspace_scores = std::move(scores_ws);
    config.workspace_qkv_buffer = std::move(qkv_ws);
    config.workspace_context = std::move(context_ws);
    config.workspace_mask = std::move(mask_ws);

    // Run batched attention
    bool success = GQAAttention::compute_batch(
        Q.get(), K.get(), V.get(), output.get(),
        actual_lengths,
        batch_size, seq_len,
        config);

    ASSERT_TRUE(success) << "Attention computation failed";

    // Verify: Padded positions [6-7] in output should be zero or very small
    const float *out_data = output->data();
    const int seq1_start = seq_len * d_model;       // Start of Seq1
    const int pad_start = seq1_start + 2 * d_model; // Start of padding (positions 6-7)

    for (int pos = 2; pos < seq_len; ++pos) // Padding positions in Seq1
    {
        for (int dim = 0; dim < d_model; ++dim)
        {
            int idx = seq1_start + pos * d_model + dim;
            // Padded outputs should be near-zero (may have small numerical errors)
            EXPECT_NEAR(out_data[idx], 0.0f, 1e-6f)
                << "Padding position " << pos << " dim " << dim << " should be ~0";
        }
    }

    // Verify: Real positions [4-5] should have non-zero output
    bool seq1_has_nonzero = false;
    for (int pos = 0; pos < 2; ++pos) // Real positions in Seq1
    {
        for (int dim = 0; dim < d_model; ++dim)
        {
            int idx = seq1_start + pos * d_model + dim;
            if (std::abs(out_data[idx]) > 1e-6f)
            {
                seq1_has_nonzero = true;
                break;
            }
        }
        if (seq1_has_nonzero)
            break;
    }

    EXPECT_TRUE(seq1_has_nonzero)
        << "Real positions in Seq1 should produce non-zero output";
}

//==============================================================================
// Test 4: Verify isolated sequence processing (no cross-contamination)
//==============================================================================
TEST(Test__GQAAttention_PaddingMask, IsolatedSequenceProcessing)
{
    // This test verifies that sequences in a batch are processed independently
    // and padding in one sequence doesn't affect another sequence
    const int batch_size = 2;
    const int seq_len = 4;
    const int n_heads = 2;
    const int n_kv_heads = 2;
    const int head_dim = 16;
    const int d_model = n_heads * head_dim;
    const int total_len = batch_size * seq_len;

    std::vector<int> actual_lengths = {4, 2};

    // Run 1: Full batch with padding
    auto Q_batch = create_fp32_tensor(total_len, d_model);
    auto K_batch = create_fp32_tensor(total_len, d_model);
    auto V_batch = create_fp32_tensor(total_len, d_model);
    auto output_batch = create_fp32_tensor(total_len, d_model);

    init_random_like(Q_batch->mutable_data(), total_len * d_model, 100);
    init_random_like(K_batch->mutable_data(), total_len * d_model, 101);
    init_random_like(V_batch->mutable_data(), total_len * d_model, 102);
    fill_constant(output_batch->mutable_data(), total_len * d_model, 0.0f);

    auto scores_ws1 = create_fp32_tensor(total_len, total_len);
    auto qkv_ws1 = create_fp32_tensor(total_len, d_model);
    auto context_ws1 = create_fp32_tensor(total_len, d_model);
    auto mask_ws1 = create_fp32_tensor(total_len, total_len);

    GQAAttentionConfig config_batch;
    config_batch.n_heads = n_heads;
    config_batch.n_kv_heads = n_kv_heads;
    config_batch.head_dim = head_dim;
    config_batch.causal = false;
    config_batch.window_size = 0;
    config_batch.workspace_scores = std::move(scores_ws1);
    config_batch.workspace_qkv_buffer = std::move(qkv_ws1);
    config_batch.workspace_context = std::move(context_ws1);
    config_batch.workspace_mask = std::move(mask_ws1);

    bool success_batch = GQAAttention::compute_batch(
        Q_batch.get(), K_batch.get(), V_batch.get(), output_batch.get(),
        actual_lengths,
        batch_size, seq_len,
        config_batch);

    ASSERT_TRUE(success_batch) << "Batch attention failed";

    // Run 2: Seq0 only (isolated)
    auto Q_seq0 = create_fp32_tensor(4, d_model);
    auto K_seq0 = create_fp32_tensor(4, d_model);
    auto V_seq0 = create_fp32_tensor(4, d_model);
    auto output_seq0 = create_fp32_tensor(4, d_model);

    // Copy Seq0 data from batch
    std::memcpy(Q_seq0->mutable_data(), Q_batch->data(), 4 * d_model * sizeof(float));
    std::memcpy(K_seq0->mutable_data(), K_batch->data(), 4 * d_model * sizeof(float));
    std::memcpy(V_seq0->mutable_data(), V_batch->data(), 4 * d_model * sizeof(float));
    fill_constant(output_seq0->mutable_data(), 4 * d_model, 0.0f);

    // Workspace for single-sequence attention: scores must be [n_heads * seq_len, seq_len]
    auto scores_ws2 = create_fp32_tensor(n_heads * 4, 4);
    auto qkv_ws2 = create_fp32_tensor(4, d_model);
    auto context_ws2 = create_fp32_tensor(4, d_model);
    auto mask_ws2 = create_fp32_tensor(4, 4);

    GQAAttentionConfig config_seq0;
    config_seq0.n_heads = n_heads;
    config_seq0.n_kv_heads = n_kv_heads;
    config_seq0.head_dim = head_dim;
    config_seq0.causal = false;
    config_seq0.window_size = 0;
    config_seq0.workspace_scores = std::move(scores_ws2);
    config_seq0.workspace_qkv_buffer = std::move(qkv_ws2);
    config_seq0.workspace_context = std::move(context_ws2);
    config_seq0.workspace_mask = std::move(mask_ws2);

    bool success_seq0 = GQAAttention::compute(
        Q_seq0.get(), K_seq0.get(), V_seq0.get(), output_seq0.get(),
        config_seq0,
        1,      // batch_size=1
        nullptr // No padding
    );

    ASSERT_TRUE(success_seq0) << "Isolated Seq0 attention failed";

    // Compare: Seq0 in batch should match isolated Seq0
    const float *batch_seq0 = output_batch->data();
    const float *isolated_seq0 = output_seq0->data();

    auto result = compare_tensors(isolated_seq0, batch_seq0, 4 * d_model, FP32_TOLERANCE);

    if (!result.equal)
    {
        LOG_ERROR("Seq0 mismatch: max_diff=" << result.max_abs_diff
                                             << " mean_diff=" << result.mean_abs_diff
                                             << " mismatches=" << result.mismatch_count);
        LOG_ERROR("First mismatch at idx=" << result.first_mismatch_idx
                                           << " expected=" << result.first_expected
                                           << " actual=" << result.first_actual);
    }

    EXPECT_TRUE(result.equal)
        << "Seq0 in batch should match isolated execution (no contamination from Seq1 padding)";
}

//==============================================================================
// Test 5: Verify attention scores for padded vs unpadded sequences
//==============================================================================
TEST(Test__GQAAttention_PaddingMask, AttentionScoresMasking)
{
    // This test directly inspects attention scores to verify padding is masked
    const int batch_size = 2;
    const int seq_len = 4;
    const int n_heads = 1; // Single head for simpler inspection
    const int n_kv_heads = 1;
    const int head_dim = 8;
    const int d_model = n_heads * head_dim;
    const int total_len = batch_size * seq_len;

    std::vector<int> actual_lengths = {4, 2};

    // Create input tensors with known values
    auto Q = create_fp32_tensor(total_len, d_model);
    auto K = create_fp32_tensor(total_len, d_model);
    auto V = create_fp32_tensor(total_len, d_model);
    auto output = create_fp32_tensor(total_len, d_model);

    // Simple initialization: Q[i] = i, K[i] = i, V[i] = 1.0
    for (int i = 0; i < total_len * d_model; ++i)
    {
        Q->mutable_data()[i] = static_cast<float>(i % d_model) * 0.1f;
        K->mutable_data()[i] = static_cast<float>(i % d_model) * 0.1f;
        V->mutable_data()[i] = 1.0f;
    }

    // Create workspace tensors (scores will be inspected) - must be square [total_len, total_len]
    auto scores_ws = create_fp32_tensor(total_len, total_len);
    auto qkv_ws = create_fp32_tensor(total_len, d_model);
    auto context_ws = create_fp32_tensor(total_len, d_model);
    auto mask_ws = create_fp32_tensor(total_len, total_len);

    GQAAttentionConfig config;
    config.n_heads = n_heads;
    config.n_kv_heads = n_kv_heads;
    config.head_dim = head_dim;
    config.causal = false;
    config.window_size = 0;
    config.workspace_scores = std::move(scores_ws);
    config.workspace_qkv_buffer = std::move(qkv_ws);
    config.workspace_context = std::move(context_ws);
    config.workspace_mask = std::move(mask_ws);

    bool success = GQAAttention::compute_batch(
        Q.get(), K.get(), V.get(), output.get(),
        actual_lengths,
        batch_size, seq_len,
        config);

    ASSERT_TRUE(success) << "Attention computation failed";

    // Inspect scores workspace after softmax
    // NOTE: This assumes the scores workspace contains post-softmax values
    // If it's pre-softmax, we'd expect -inf for masked positions
    const float *scores = config.workspace_scores->data();

    // Seq1 position 4 (first real token): Should have scores only for positions [0-1]
    const int seq1_offset = seq_len;
    float sum_unmasked = 0.0f;
    float sum_masked = 0.0f;

    for (int j = 0; j < 2; ++j) // Real positions [0-1]
    {
        float score = scores[seq1_offset * seq_len + j];
        sum_unmasked += score;
        EXPECT_GT(score, 0.0f)
            << "Seq1[0] should have positive attention score for real position " << j;
    }

    for (int j = 2; j < seq_len; ++j) // Padded positions [2-3]
    {
        float score = scores[seq1_offset * seq_len + j];
        sum_masked += score;
        // Post-softmax, masked positions should be near-zero
        EXPECT_NEAR(score, 0.0f, 1e-6f)
            << "Seq1[0] should have ~0 attention score for padded position " << j;
    }

    // Verify attention scores sum to ~1.0 (softmax property)
    EXPECT_NEAR(sum_unmasked + sum_masked, 1.0f, 1e-4f)
        << "Attention scores should sum to 1.0";

    // Most importantly: masked scores should contribute negligibly
    EXPECT_LT(sum_masked, 1e-5f)
        << "Masked positions should have negligible attention weight";
}

//==============================================================================
// Test 6: Verify batch with all padding (edge case)
//==============================================================================
TEST(Test__GQAAttention_PaddingMask, AllPaddingSequence)
{
    // Edge case: One sequence is all padding (length=0)
    const int batch_size = 2;
    const int seq_len = 4;
    const int n_heads = 2;
    const int n_kv_heads = 2;
    const int head_dim = 16;
    const int d_model = n_heads * head_dim;
    const int total_len = batch_size * seq_len;

    std::vector<int> actual_lengths = {4, 0}; // Seq1 is all padding!

    auto Q = create_fp32_tensor(total_len, d_model);
    auto K = create_fp32_tensor(total_len, d_model);
    auto V = create_fp32_tensor(total_len, d_model);
    auto output = create_fp32_tensor(total_len, d_model);

    init_random_like(Q->mutable_data(), total_len * d_model, 200);
    init_random_like(K->mutable_data(), total_len * d_model, 201);
    init_random_like(V->mutable_data(), total_len * d_model, 202);
    fill_constant(output->mutable_data(), total_len * d_model, 0.0f);

    auto scores_ws = create_fp32_tensor(total_len, total_len);
    auto qkv_ws = create_fp32_tensor(total_len, d_model);
    auto context_ws = create_fp32_tensor(total_len, d_model);
    auto mask_ws = create_fp32_tensor(total_len, total_len);

    GQAAttentionConfig config;
    config.n_heads = n_heads;
    config.n_kv_heads = n_kv_heads;
    config.head_dim = head_dim;
    config.causal = false;
    config.window_size = 0;
    config.workspace_scores = std::move(scores_ws);
    config.workspace_qkv_buffer = std::move(qkv_ws);
    config.workspace_context = std::move(context_ws);
    config.workspace_mask = std::move(mask_ws);

    bool success = GQAAttention::compute_batch(
        Q.get(), K.get(), V.get(), output.get(),
        actual_lengths,
        batch_size, seq_len,
        config);

    ASSERT_TRUE(success) << "Attention should handle all-padding sequence gracefully";

    // Verify: All of Seq1's output should be zero
    const float *out_data = output->data();
    const int seq1_start = seq_len * d_model;

    for (int pos = 0; pos < seq_len; ++pos)
    {
        for (int dim = 0; dim < d_model; ++dim)
        {
            int idx = seq1_start + pos * d_model + dim;
            EXPECT_NEAR(out_data[idx], 0.0f, 1e-6f)
                << "All-padding sequence should produce zero output";
        }
    }

    // Verify: Seq0 should still work normally (non-zero)
    bool seq0_has_nonzero = false;
    for (int i = 0; i < seq_len * d_model; ++i)
    {
        if (std::abs(out_data[i]) > 1e-6f)
        {
            seq0_has_nonzero = true;
            break;
        }
    }

    EXPECT_TRUE(seq0_has_nonzero)
        << "Seq0 (no padding) should produce non-zero output";
}
