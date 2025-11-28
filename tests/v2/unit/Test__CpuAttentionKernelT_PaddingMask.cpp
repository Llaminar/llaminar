/**
 * @file Test__CpuAttentionKernelT_PaddingMask.cpp
 * @brief Unit tests for CpuAttentionKernelT batch attention with padding masks
 * @author David Sanftenberg
 *
 * Tests specifically targeting the batch padding divergence bug:
 * 1. Mask application verification (scores should be -inf for padding)
 * 2. Padding output zeroing (output should be 0 for padded positions)
 * 3. Sequential vs batch with padding (should match exactly)
 * 4. Attention score inspection (verify mask is applied before softmax)
 * 5. K/V cache boundary verification (no cross-sequence contamination)
 */

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <memory>
#include <iostream>
#include <iomanip>

#include "v2/kernels/cpu/attention/CpuAttentionKernelT.h"
#include "v2/tensors/Tensors.h"
#include "v2/pipelines/AttentionUtils.h"

using namespace llaminar2;

namespace
{
    constexpr float FP32_TOLERANCE = 1e-4f;
    constexpr float MASK_NEGINF = -1e9f;

    // Helper: Initialize with sequential values
    void init_sequential(float *data, int count, float start = 0.0f)
    {
        for (int i = 0; i < count; ++i)
        {
            data[i] = start + static_cast<float>(i) / 10.0f;
        }
    }

    // Helper: Compare two float arrays with tolerance
    bool arrays_equal(const float *a, const float *b, int count, float tolerance = FP32_TOLERANCE)
    {
        for (int i = 0; i < count; ++i)
        {
            if (std::abs(a[i] - b[i]) > tolerance)
            {
                return false;
            }
        }
        return true;
    }

    // Helper: Print first N values
    void print_values(const char *label, const float *data, int count, int max_print = 10)
    {
        std::cout << label << ": [";
        for (int i = 0; i < std::min(count, max_print); ++i)
        {
            std::cout << std::fixed << std::setprecision(5) << data[i];
            if (i < std::min(count, max_print) - 1)
                std::cout << ", ";
        }
        if (count > max_print)
            std::cout << ", ...";
        std::cout << "]" << std::endl;
    }

    // Helper: Check if all values in range are zero
    bool all_zero(const float *data, int start, int count, float tolerance = 1e-6f)
    {
        for (int i = start; i < start + count; ++i)
        {
            if (std::abs(data[i]) > tolerance)
            {
                return false;
            }
        }
        return true;
    }

    // Helper: Count non-zero values
    int count_nonzero(const float *data, int count, float tolerance = 1e-6f)
    {
        int nonzero = 0;
        for (int i = 0; i < count; ++i)
        {
            if (std::abs(data[i]) > tolerance)
            {
                ++nonzero;
            }
        }
        return nonzero;
    }

} // anonymous namespace

// ============================================================================
// Test 1: Padding Mask Construction and Application
// ============================================================================

TEST(CpuAttentionKernelT_PaddingMask, MaskIsAppliedToScores)
{
    /**
     * CRITICAL TEST: Verify that the attention mask is actually being applied
     * to attention scores in the batch compute path.
     *
     * Setup:
     * - 2 sequences: Seq0=[4 tokens], Seq1=[2 tokens + 2 padding]
     * - Construct padding mask: tokens 4-5 should attend to themselves, 6-7 should be masked
     * - After mask application, scores for padded positions should be -inf
     *
     * This test directly addresses the hypothesis that masks are constructed correctly
     * but not applied in compute_batch().
     */

    const int batch_size = 2;
    const int seq_len = 4; // Max sequence length
    const int n_heads = 2;
    const int n_kv_heads = 2;
    const int head_dim = 4;

    // Actual sequence lengths: Seq0=4, Seq1=2 (padded to 4)
    std::vector<int> actual_lengths = {4, 2};

    const int total_tokens = batch_size * seq_len; // 8 tokens total

    // Allocate inputs
    std::vector<float> Q(total_tokens * n_heads * head_dim);
    std::vector<float> K(total_tokens * n_kv_heads * head_dim);
    std::vector<float> V(total_tokens * n_kv_heads * head_dim);
    std::vector<float> output(total_tokens * n_heads * head_dim, 0.0f);

    // Initialize with distinct values
    init_sequential(Q.data(), Q.size(), 1.0f);
    init_sequential(K.data(), K.size(), 10.0f);
    init_sequential(V.data(), V.size(), 20.0f);

    // Zero out padding positions in Q, K, V (positions 6-7 = batch 1, tokens 2-3)
    const int padding_start = 4 + 2; // Batch 1, after 2 real tokens
    const int padding_count = 2;
    for (int i = 0; i < padding_count; ++i)
    {
        int pos = padding_start + i;
        for (int h = 0; h < n_heads; ++h)
        {
            for (int d = 0; d < head_dim; ++d)
            {
                Q[pos * n_heads * head_dim + h * head_dim + d] = 0.0f;
            }
        }
        for (int h = 0; h < n_kv_heads; ++h)
        {
            for (int d = 0; d < head_dim; ++d)
            {
                K[pos * n_kv_heads * head_dim + h * head_dim + d] = 0.0f;
                V[pos * n_kv_heads * head_dim + h * head_dim + d] = 0.0f;
            }
        }
    }

    // Construct padding mask using AttentionUtils
    auto mask_tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{
        static_cast<size_t>(total_tokens * total_tokens)});

    attention_utils::create_batch_padding_mask(
        mask_tensor->mutable_data(),
        batch_size,
        seq_len,
        actual_lengths.data(),
        -1 // window_size
    );

    const float *mask_data = mask_tensor->data();

    // Verify mask structure (diagnostic)
    std::cout << "\n[MASK VERIFICATION]" << std::endl;
    std::cout << "Mask for positions 4-7 (Seq1):" << std::endl;
    for (int i = 4; i < 8; ++i)
    {
        std::cout << "  Row " << i << ": [";
        for (int j = 0; j < 8; ++j)
        {
            float val = mask_data[i * total_tokens + j];
            if (val < -1e8f)
            {
                std::cout << "-INF";
            }
            else
            {
                std::cout << std::setw(5) << val;
            }
            if (j < 7)
                std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }

    // Allocate workspace for scores inspection
    auto scores_workspace = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(batch_size * n_heads * seq_len * seq_len)});

    // Run batch attention with mask
    CpuAttentionKernelT<FP32Tensor> attention;
    bool success = attention.compute_batch(
        Q.data(), K.data(), V.data(), output.data(),
        batch_size, seq_len, n_heads, n_kv_heads, head_dim,
        false, -1,
        scores_workspace.get(), // Capture scores
        nullptr, nullptr,
        mask_tensor.get(), // Pass mask
        false, nullptr, -1);

    ASSERT_TRUE(success) << "Batch attention should succeed";

    // CRITICAL CHECK: Inspect attention scores for Seq1 (batch item 1)
    // Scores layout: [batch_size, n_heads, seq_len, seq_len]
    // Seq1 scores: batch=1, heads=2, seq_len=4, seq_len=4
    const float *scores_data = scores_workspace->data();
    const int scores_per_head = seq_len * seq_len;
    const int scores_per_batch = n_heads * scores_per_head;

    std::cout << "\n[ATTENTION SCORES - SEQ1]" << std::endl;
    for (int h = 0; h < n_heads; ++h)
    {
        std::cout << "Head " << h << ":" << std::endl;
        const float *scores_h = scores_data + (1 * scores_per_batch) + (h * scores_per_head);

        // Check scores for padding positions (rows 2-3 in Seq1's view)
        for (int i = 2; i < 4; ++i)
        {
            std::cout << "  Row " << i << " (PAD): [";
            for (int j = 0; j < 4; ++j)
            {
                float score = scores_h[i * seq_len + j];
                std::cout << std::setw(8) << std::fixed << std::setprecision(5) << score;
                if (j < 3)
                    std::cout << ", ";
            }
            std::cout << "]" << std::endl;

            // After softmax, scores for padding rows should be ~0 (not -inf)
            // But they should have been masked before softmax
            // Check if ALL scores in this row are near zero
            bool all_near_zero = true;
            for (int j = 0; j < 4; ++j)
            {
                if (std::abs(scores_h[i * seq_len + j]) > 1e-5f)
                {
                    all_near_zero = false;
                    break;
                }
            }

            if (!all_near_zero)
            {
                std::cout << "  ❌ PADDING ROW HAS NON-ZERO SCORES!" << std::endl;
            }
        }
    }

    // CRITICAL ASSERTION: Padding positions should produce near-zero output
    std::cout << "\n[OUTPUT VERIFICATION]" << std::endl;
    for (int i = padding_start; i < padding_start + padding_count; ++i)
    {
        std::cout << "Position " << i << " (PAD): ";
        float sum = 0.0f;
        for (int h = 0; h < n_heads; ++h)
        {
            for (int d = 0; d < head_dim; ++d)
            {
                sum += std::abs(output[i * n_heads * head_dim + h * head_dim + d]);
            }
        }
        std::cout << "sum(abs) = " << sum << std::endl;

        // **THIS IS THE KEY TEST**: Padding output should be ~0
        EXPECT_LT(sum, 1e-3f) << "Padding position " << i << " should have near-zero output";
    }
}

// ============================================================================
// Test 2: Sequential vs Batch with Padding (Detailed Comparison)
// ============================================================================

TEST(CpuAttentionKernelT_PaddingMask, SequentialVsBatchWithPadding)
{
    /**
     * CRITICAL TEST: Compare sequential and batch execution with padding.
     * This directly replicates the E2E test failure scenario at the kernel level.
     *
     * Setup:
     * - Seq0: 4 tokens (no padding)
     * - Seq1: 2 tokens + 2 padding
     * - Run sequential: process Seq0 and Seq1 separately
     * - Run batch: process both together with padding mask
     * - Compare: outputs should match exactly
     */

    const int seq0_len = 4;
    const int seq1_len = 2;
    const int max_seq_len = 4; // Padded length
    const int n_heads = 2;
    const int n_kv_heads = 2;
    const int head_dim = 4;

    // Allocate per-sequence inputs
    std::vector<float> Q_seq0(seq0_len * n_heads * head_dim);
    std::vector<float> K_seq0(seq0_len * n_kv_heads * head_dim);
    std::vector<float> V_seq0(seq0_len * n_kv_heads * head_dim);
    std::vector<float> output_seq0(seq0_len * n_heads * head_dim, 0.0f);

    std::vector<float> Q_seq1(seq1_len * n_heads * head_dim);
    std::vector<float> K_seq1(seq1_len * n_kv_heads * head_dim);
    std::vector<float> V_seq1(seq1_len * n_kv_heads * head_dim);
    std::vector<float> output_seq1(seq1_len * n_heads * head_dim, 0.0f);

    // Initialize with distinct values
    init_sequential(Q_seq0.data(), Q_seq0.size(), 1.0f);
    init_sequential(K_seq0.data(), K_seq0.size(), 10.0f);
    init_sequential(V_seq0.data(), V_seq0.size(), 20.0f);

    init_sequential(Q_seq1.data(), Q_seq1.size(), 100.0f);
    init_sequential(K_seq1.data(), K_seq1.size(), 200.0f);
    init_sequential(V_seq1.data(), V_seq1.size(), 300.0f);

    // Sequential execution
    CpuAttentionKernelT<FP32Tensor> attention_seq;

    bool success_seq0 = attention_seq.compute(
        Q_seq0.data(), K_seq0.data(), V_seq0.data(), output_seq0.data(),
        seq0_len, n_heads, n_kv_heads, head_dim,
        false, -1, nullptr, nullptr, nullptr, nullptr, false, nullptr, -1);
    ASSERT_TRUE(success_seq0) << "Sequential Seq0 should succeed";

    bool success_seq1 = attention_seq.compute(
        Q_seq1.data(), K_seq1.data(), V_seq1.data(), output_seq1.data(),
        seq1_len, n_heads, n_kv_heads, head_dim,
        false, -1, nullptr, nullptr, nullptr, nullptr, false, nullptr, -1);
    ASSERT_TRUE(success_seq1) << "Sequential Seq1 should succeed";

    // Batch execution with padding
    const int batch_size = 2;
    const int total_tokens = batch_size * max_seq_len;

    std::vector<float> Q_batch(total_tokens * n_heads * head_dim, 0.0f);
    std::vector<float> K_batch(total_tokens * n_kv_heads * head_dim, 0.0f);
    std::vector<float> V_batch(total_tokens * n_kv_heads * head_dim, 0.0f);
    std::vector<float> output_batch(total_tokens * n_heads * head_dim, 0.0f);

    // Copy Seq0 to batch positions 0-3
    std::memcpy(Q_batch.data(), Q_seq0.data(), Q_seq0.size() * sizeof(float));
    std::memcpy(K_batch.data(), K_seq0.data(), K_seq0.size() * sizeof(float));
    std::memcpy(V_batch.data(), V_seq0.data(), V_seq0.size() * sizeof(float));

    // Copy Seq1 to batch positions 4-5 (leave 6-7 as padding zeros)
    const int seq1_start = max_seq_len;
    for (int i = 0; i < seq1_len; ++i)
    {
        for (int h = 0; h < n_heads; ++h)
        {
            for (int d = 0; d < head_dim; ++d)
            {
                Q_batch[(seq1_start + i) * n_heads * head_dim + h * head_dim + d] =
                    Q_seq1[i * n_heads * head_dim + h * head_dim + d];
            }
        }
        for (int h = 0; h < n_kv_heads; ++h)
        {
            for (int d = 0; d < head_dim; ++d)
            {
                K_batch[(seq1_start + i) * n_kv_heads * head_dim + h * head_dim + d] =
                    K_seq1[i * n_kv_heads * head_dim + h * head_dim + d];
                V_batch[(seq1_start + i) * n_kv_heads * head_dim + h * head_dim + d] =
                    V_seq1[i * n_kv_heads * head_dim + h * head_dim + d];
            }
        }
    }

    // Construct padding mask
    std::vector<int> actual_lengths = {seq0_len, seq1_len};
    auto mask_tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{
        static_cast<size_t>(batch_size * max_seq_len * batch_size * max_seq_len)});

    attention_utils::create_batch_padding_mask(
        mask_tensor->mutable_data(),
        batch_size,
        max_seq_len,
        actual_lengths.data(),
        -1 // window_size
    );

    // Run batch attention
    CpuAttentionKernelT<FP32Tensor> attention_batch;
    bool success_batch = attention_batch.compute_batch(
        Q_batch.data(), K_batch.data(), V_batch.data(), output_batch.data(),
        batch_size, max_seq_len, n_heads, n_kv_heads, head_dim,
        false, -1, nullptr, nullptr, nullptr,
        mask_tensor.get(),
        false, nullptr, -1);

    ASSERT_TRUE(success_batch) << "Batch attention should succeed";

    // CRITICAL COMPARISON: Extract batch results and compare to sequential
    std::cout << "\n[COMPARISON: SEQUENTIAL VS BATCH]" << std::endl;

    // Compare Seq0 (positions 0-3)
    bool seq0_match = arrays_equal(
        output_seq0.data(),
        output_batch.data(),
        output_seq0.size(),
        FP32_TOLERANCE);

    if (!seq0_match)
    {
        std::cout << "❌ Seq0 MISMATCH!" << std::endl;
        print_values("  Sequential Seq0", output_seq0.data(), n_heads * head_dim);
        print_values("  Batch Seq0", output_batch.data(), n_heads * head_dim);
    }
    else
    {
        std::cout << "✅ Seq0 matches perfectly" << std::endl;
    }

    // Compare Seq1 (positions 4-5, ignore padding 6-7)
    bool seq1_match = true;
    for (int i = 0; i < seq1_len; ++i)
    {
        const float *seq_ptr = output_seq1.data() + i * n_heads * head_dim;
        const float *batch_ptr = output_batch.data() + (seq1_start + i) * n_heads * head_dim;

        if (!arrays_equal(seq_ptr, batch_ptr, n_heads * head_dim, FP32_TOLERANCE))
        {
            seq1_match = false;
            std::cout << "❌ Seq1 Position " << i << " MISMATCH!" << std::endl;
            print_values("  Sequential", seq_ptr, n_heads * head_dim);
            print_values("  Batch", batch_ptr, n_heads * head_dim);
        }
    }

    if (seq1_match)
    {
        std::cout << "✅ Seq1 matches perfectly" << std::endl;
    }

    // Check padding output is zero
    bool padding_zero = all_zero(
        output_batch.data() + (seq1_start + seq1_len) * n_heads * head_dim,
        0,
        (max_seq_len - seq1_len) * n_heads * head_dim,
        1e-5f);

    if (!padding_zero)
    {
        std::cout << "❌ Padding positions have NON-ZERO output!" << std::endl;
        print_values("  Padding output", output_batch.data() + (seq1_start + seq1_len) * n_heads * head_dim,
                     (max_seq_len - seq1_len) * n_heads * head_dim);
    }
    else
    {
        std::cout << "✅ Padding output is zero" << std::endl;
    }

    EXPECT_TRUE(seq0_match) << "Seq0 should match between sequential and batch";
    EXPECT_TRUE(seq1_match) << "Seq1 should match between sequential and batch";
    EXPECT_TRUE(padding_zero) << "Padding output should be zero";
}

// ============================================================================
// Test 3: Cross-Sequence Contamination Check
// ============================================================================

TEST(CpuAttentionKernelT_PaddingMask, NoCrossSequenceContamination)
{
    /**
     * CRITICAL TEST: Verify that sequences in a batch don't contaminate each other.
     * Each sequence should only attend to its own tokens, not to other sequences.
     *
     * Setup:
     * - Seq0: All 1.0 values
     * - Seq1: All 10.0 values
     * - With proper masking, Seq0 output should not contain any 10.0 influence
     * - Seq1 output should not contain any 1.0 influence
     */

    const int batch_size = 2;
    const int seq_len = 3;
    const int n_heads = 2;
    const int n_kv_heads = 2;
    const int head_dim = 4;
    const int total_tokens = batch_size * seq_len;

    // Allocate inputs with distinct values per sequence
    std::vector<float> Q(total_tokens * n_heads * head_dim);
    std::vector<float> K(total_tokens * n_kv_heads * head_dim);
    std::vector<float> V(total_tokens * n_kv_heads * head_dim);
    std::vector<float> output(total_tokens * n_heads * head_dim, 0.0f);

    // Seq0: all 1.0
    for (int i = 0; i < seq_len * n_heads * head_dim; ++i)
    {
        Q[i] = 1.0f;
    }
    for (int i = 0; i < seq_len * n_kv_heads * head_dim; ++i)
    {
        K[i] = 1.0f;
        V[i] = 1.0f;
    }

    // Seq1: all 10.0
    for (int i = 0; i < seq_len * n_heads * head_dim; ++i)
    {
        Q[seq_len * n_heads * head_dim + i] = 10.0f;
    }
    for (int i = 0; i < seq_len * n_kv_heads * head_dim; ++i)
    {
        K[seq_len * n_kv_heads * head_dim + i] = 10.0f;
        V[seq_len * n_kv_heads * head_dim + i] = 10.0f;
    }

    // Construct padding mask (no actual padding, just sequence boundaries)
    std::vector<int> actual_lengths = {seq_len, seq_len};
    auto mask_tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{
        static_cast<size_t>(total_tokens * total_tokens)});

    attention_utils::create_batch_padding_mask(
        mask_tensor->mutable_data(),
        batch_size,
        seq_len,
        actual_lengths.data(),
        -1 // window_size
    );

    // Run batch attention
    CpuAttentionKernelT<FP32Tensor> attention;
    bool success = attention.compute_batch(
        Q.data(), K.data(), V.data(), output.data(),
        batch_size, seq_len, n_heads, n_kv_heads, head_dim,
        false, -1, nullptr, nullptr, nullptr,
        mask_tensor.get(),
        false, nullptr, -1);

    ASSERT_TRUE(success);

    // Check Seq0 output (should be ~1.0, not influenced by 10.0)
    // After attention with uniform Q/K/V=1.0, output should be close to V=1.0
    std::cout << "\n[CROSS-SEQUENCE CONTAMINATION CHECK]" << std::endl;
    std::cout << "Seq0 output (expect ~1.0): ";
    print_values("", output.data(), n_heads * head_dim);

    std::cout << "Seq1 output (expect ~10.0): ";
    print_values("", output.data() + seq_len * n_heads * head_dim, n_heads * head_dim);

    // Verify Seq0 output is close to 1.0 (not contaminated by 10.0)
    for (int i = 0; i < seq_len * n_heads * head_dim; ++i)
    {
        EXPECT_NEAR(output[i], 1.0f, 0.5f) << "Seq0 contaminated at index " << i;
    }

    // Verify Seq1 output is close to 10.0 (not contaminated by 1.0)
    for (int i = 0; i < seq_len * n_heads * head_dim; ++i)
    {
        EXPECT_NEAR(output[seq_len * n_heads * head_dim + i], 10.0f, 1.0f)
            << "Seq1 contaminated at index " << i;
    }
}

// ============================================================================
// Test 4: Attention Score Inspection (Before/After Softmax)
// ============================================================================

TEST(CpuAttentionKernelT_PaddingMask, AttentionScoreInspection)
{
    /**
     * ADVANCED TEST: Inspect raw attention scores to verify masking happens BEFORE softmax.
     *
     * This test cannot directly observe pre-softmax scores without modifying the kernel,
     * but we can verify post-softmax scores show the expected pattern:
     * - Masked positions should have ~0 attention weight
     * - Unmasked positions should sum to ~1.0 per row
     */

    const int batch_size = 1;
    const int seq_len = 4;
    const int n_heads = 1;
    const int n_kv_heads = 1;
    const int head_dim = 4;

    // Actual length: 2 (positions 2-3 are padding)
    std::vector<int> actual_lengths = {2};

    std::vector<float> Q(seq_len * n_heads * head_dim);
    std::vector<float> K(seq_len * n_kv_heads * head_dim);
    std::vector<float> V(seq_len * n_kv_heads * head_dim);
    std::vector<float> output(seq_len * n_heads * head_dim, 0.0f);

    init_sequential(Q.data(), Q.size(), 1.0f);
    init_sequential(K.data(), K.size(), 10.0f);
    init_sequential(V.data(), V.size(), 20.0f);

    // Zero out padding
    for (int i = 2; i < 4; ++i)
    {
        for (int d = 0; d < head_dim; ++d)
        {
            Q[i * head_dim + d] = 0.0f;
            K[i * head_dim + d] = 0.0f;
            V[i * head_dim + d] = 0.0f;
        }
    }

    // Construct mask
    const int total_tokens = batch_size * seq_len;
    auto mask_tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{
        static_cast<size_t>(total_tokens * total_tokens)});

    attention_utils::create_batch_padding_mask(
        mask_tensor->mutable_data(),
        batch_size,
        seq_len,
        actual_lengths.data(),
        -1 // window_size
    );

    // Allocate workspace to capture scores
    auto scores_workspace = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(batch_size * n_heads * seq_len * seq_len)});

    CpuAttentionKernelT<FP32Tensor> attention;
    bool success = attention.compute_batch(
        Q.data(), K.data(), V.data(), output.data(),
        batch_size, seq_len, n_heads, n_kv_heads, head_dim,
        false, -1,
        scores_workspace.get(), // Capture scores
        nullptr, nullptr,
        mask_tensor.get(),
        false, nullptr, -1);

    ASSERT_TRUE(success);

    // Inspect attention scores (post-softmax)
    const float *scores = scores_workspace->data();

    std::cout << "\n[ATTENTION SCORES AFTER SOFTMAX]" << std::endl;
    std::cout << "Rows 0-1 (real tokens) should attend to positions 0-1 only" << std::endl;
    std::cout << "Rows 2-3 (padding) should have all ~0 weights" << std::endl;

    for (int i = 0; i < seq_len; ++i)
    {
        std::cout << "Row " << i << ": [";
        float row_sum = 0.0f;
        for (int j = 0; j < seq_len; ++j)
        {
            float score = scores[i * seq_len + j];
            row_sum += score;
            std::cout << std::setw(8) << std::fixed << std::setprecision(5) << score;
            if (j < seq_len - 1)
                std::cout << ", ";
        }
        std::cout << "] sum=" << row_sum << std::endl;

        if (i < 2)
        {
            // Real tokens: should sum to ~1.0, attend only to [0,1]
            EXPECT_NEAR(row_sum, 1.0f, 0.01f) << "Row " << i << " sum should be ~1.0";
            EXPECT_LT(std::abs(scores[i * seq_len + 2]), 1e-5f) << "Should not attend to padding";
            EXPECT_LT(std::abs(scores[i * seq_len + 3]), 1e-5f) << "Should not attend to padding";
        }
        else
        {
            // Padding rows: should have all ~0 (or sum to 0)
            EXPECT_LT(row_sum, 1e-3f) << "Padding row " << i << " should have near-zero sum";
        }
    }
}
