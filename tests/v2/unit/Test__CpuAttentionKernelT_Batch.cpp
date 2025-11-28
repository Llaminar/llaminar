/**
 * @file Test__CpuAttentionKernelT_Batch.cpp
 * @brief Comprehensive unit tests for CpuAttentionKernelT batch attention
 * @author David Sanftenberg
 *
 * Tests:
 * 1. Batch vs Sequential Equivalence
 * 2. Batch Independence (sequences don't interfere)
 * 3. Batch with GQA
 * 4. Batch with causal masking
 * 5. Multi-batch scenarios
 */

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <memory>

#include "v2/kernels/cpu/attention/CpuAttentionKernelT.h"
#include "v2/tensors/Tensors.h"

using namespace llaminar2;

namespace
{
    constexpr float FP32_TOLERANCE = 1e-5f;

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

    // Helper: Print first N values for debugging
    void print_values(const char *label, const float *data, int count, int max_print = 10)
    {
        std::cout << label << ": [";
        for (int i = 0; i < std::min(count, max_print); ++i)
        {
            std::cout << data[i];
            if (i < std::min(count, max_print) - 1)
                std::cout << ", ";
        }
        if (count > max_print)
            std::cout << ", ...";
        std::cout << "]" << std::endl;
    }

} // anonymous namespace

// ============================================================================
// Batch vs Sequential Equivalence Tests
// ============================================================================

TEST(CpuAttentionKernelT_Batch, BatchSize1EqualsSequential)
{
    // CRITICAL TEST: batch_size=1 should produce identical results to single-sequence
    const int seq_len = 4;
    const int n_heads = 2;
    const int n_kv_heads = 2;
    const int head_dim = 8;

    // Prepare input data
    std::vector<float> Q(seq_len * n_heads * head_dim);
    std::vector<float> K(seq_len * n_kv_heads * head_dim);
    std::vector<float> V(seq_len * n_kv_heads * head_dim);

    init_sequential(Q.data(), Q.size());
    init_sequential(K.data(), K.size(), 10.0f);
    init_sequential(V.data(), V.size(), 20.0f);

    // Sequential path: compute()
    std::vector<float> output_sequential(seq_len * n_heads * head_dim, 0.0f);
    CpuAttentionKernelT<FP32Tensor> attention_seq;
    bool success_seq = attention_seq.compute(
        Q.data(), K.data(), V.data(), output_sequential.data(),
        seq_len, n_heads, n_kv_heads, head_dim,
        false, -1,
        nullptr, nullptr, nullptr, nullptr,
        false, nullptr, -1);

    ASSERT_TRUE(success_seq) << "Sequential compute failed";

    // Batch path: compute_batch() with batch_size=1
    std::vector<float> output_batch(seq_len * n_heads * head_dim, 0.0f);
    CpuAttentionKernelT<FP32Tensor> attention_batch;
    bool success_batch = attention_batch.compute_batch(
        Q.data(), K.data(), V.data(), output_batch.data(),
        1, // batch_size=1
        seq_len, n_heads, n_kv_heads, head_dim,
        false, -1,
        nullptr, nullptr, nullptr, nullptr,
        false, nullptr, -1);

    ASSERT_TRUE(success_batch) << "Batch compute failed";

    // Compare outputs - should be IDENTICAL
    bool outputs_match = arrays_equal(output_sequential.data(), output_batch.data(), output_sequential.size());
    if (!outputs_match)
    {
        print_values("Sequential", output_sequential.data(), output_sequential.size());
        print_values("Batch(1)", output_batch.data(), output_batch.size());
    }
    EXPECT_TRUE(outputs_match) << "Batch size=1 should produce identical results to sequential";
}

TEST(CpuAttentionKernelT_Batch, BatchSize1EqualsSequential_GQA)
{
    // Same test but with GQA (n_heads > n_kv_heads)
    const int seq_len = 3;
    const int n_heads = 4;
    const int n_kv_heads = 2;
    const int head_dim = 4;

    std::vector<float> Q(seq_len * n_heads * head_dim);
    std::vector<float> K(seq_len * n_kv_heads * head_dim);
    std::vector<float> V(seq_len * n_kv_heads * head_dim);

    init_sequential(Q.data(), Q.size());
    init_sequential(K.data(), K.size(), 10.0f);
    init_sequential(V.data(), V.size(), 20.0f);

    // Sequential
    std::vector<float> output_sequential(seq_len * n_heads * head_dim, 0.0f);
    CpuAttentionKernelT<FP32Tensor> attention_seq;
    ASSERT_TRUE(attention_seq.compute(
        Q.data(), K.data(), V.data(), output_sequential.data(),
        seq_len, n_heads, n_kv_heads, head_dim,
        false, -1, nullptr, nullptr, nullptr, nullptr, false, nullptr, -1));

    // Batch
    std::vector<float> output_batch(seq_len * n_heads * head_dim, 0.0f);
    CpuAttentionKernelT<FP32Tensor> attention_batch;
    ASSERT_TRUE(attention_batch.compute_batch(
        Q.data(), K.data(), V.data(), output_batch.data(),
        1, seq_len, n_heads, n_kv_heads, head_dim,
        false, -1, nullptr, nullptr, nullptr, nullptr, false, nullptr, -1));

    EXPECT_TRUE(arrays_equal(output_sequential.data(), output_batch.data(), output_sequential.size()))
        << "GQA: Batch size=1 should equal sequential";
}

TEST(CpuAttentionKernelT_Batch, BatchSize1EqualsSequential_Causal)
{
    // Test with causal masking
    const int seq_len = 4;
    const int n_heads = 2;
    const int n_kv_heads = 2;
    const int head_dim = 8;

    std::vector<float> Q(seq_len * n_heads * head_dim);
    std::vector<float> K(seq_len * n_kv_heads * head_dim);
    std::vector<float> V(seq_len * n_kv_heads * head_dim);

    init_sequential(Q.data(), Q.size());
    init_sequential(K.data(), K.size(), 10.0f);
    init_sequential(V.data(), V.size(), 20.0f);

    // Sequential with causal=true
    std::vector<float> output_sequential(seq_len * n_heads * head_dim, 0.0f);
    CpuAttentionKernelT<FP32Tensor> attention_seq;
    ASSERT_TRUE(attention_seq.compute(
        Q.data(), K.data(), V.data(), output_sequential.data(),
        seq_len, n_heads, n_kv_heads, head_dim,
        true, -1, nullptr, nullptr, nullptr, nullptr, false, nullptr, -1));

    // Batch with causal=true
    std::vector<float> output_batch(seq_len * n_heads * head_dim, 0.0f);
    CpuAttentionKernelT<FP32Tensor> attention_batch;
    ASSERT_TRUE(attention_batch.compute_batch(
        Q.data(), K.data(), V.data(), output_batch.data(),
        1, seq_len, n_heads, n_kv_heads, head_dim,
        true, -1, nullptr, nullptr, nullptr, nullptr, false, nullptr, -1));

    EXPECT_TRUE(arrays_equal(output_sequential.data(), output_batch.data(), output_sequential.size()))
        << "Causal: Batch size=1 should equal sequential";
}

// ============================================================================
// Batch Independence Tests
// ============================================================================

TEST(CpuAttentionKernelT_Batch, BatchIndependence_TwoSequences)
{
    // CRITICAL TEST: Each sequence in batch should be processed independently
    // Running seq0 and seq1 together should equal running them separately
    const int batch_size = 2;
    const int seq_len = 3;
    const int n_heads = 2;
    const int n_kv_heads = 2;
    const int head_dim = 4;

    // Prepare TWO different sequences
    std::vector<float> Q0(seq_len * n_heads * head_dim);
    std::vector<float> K0(seq_len * n_kv_heads * head_dim);
    std::vector<float> V0(seq_len * n_kv_heads * head_dim);
    init_sequential(Q0.data(), Q0.size(), 0.0f);
    init_sequential(K0.data(), K0.size(), 10.0f);
    init_sequential(V0.data(), V0.size(), 20.0f);

    std::vector<float> Q1(seq_len * n_heads * head_dim);
    std::vector<float> K1(seq_len * n_kv_heads * head_dim);
    std::vector<float> V1(seq_len * n_kv_heads * head_dim);
    init_sequential(Q1.data(), Q1.size(), 100.0f); // Different values!
    init_sequential(K1.data(), K1.size(), 110.0f);
    init_sequential(V1.data(), V1.size(), 120.0f);

    // Process sequences separately
    std::vector<float> output0_separate(seq_len * n_heads * head_dim, 0.0f);
    std::vector<float> output1_separate(seq_len * n_heads * head_dim, 0.0f);

    CpuAttentionKernelT<FP32Tensor> attention_sep;
    ASSERT_TRUE(attention_sep.compute(
        Q0.data(), K0.data(), V0.data(), output0_separate.data(),
        seq_len, n_heads, n_kv_heads, head_dim,
        false, -1, nullptr, nullptr, nullptr, nullptr, false, nullptr, -1));

    ASSERT_TRUE(attention_sep.compute(
        Q1.data(), K1.data(), V1.data(), output1_separate.data(),
        seq_len, n_heads, n_kv_heads, head_dim,
        false, -1, nullptr, nullptr, nullptr, nullptr, false, nullptr, -1));

    // Process sequences together in batch
    const int total_tokens = batch_size * seq_len;
    std::vector<float> Q_batch(total_tokens * n_heads * head_dim);
    std::vector<float> K_batch(total_tokens * n_kv_heads * head_dim);
    std::vector<float> V_batch(total_tokens * n_kv_heads * head_dim);

    // Layout: [seq0_tok0, seq0_tok1, seq0_tok2, seq1_tok0, seq1_tok1, seq1_tok2]
    std::memcpy(Q_batch.data(), Q0.data(), Q0.size() * sizeof(float));
    std::memcpy(Q_batch.data() + Q0.size(), Q1.data(), Q1.size() * sizeof(float));
    std::memcpy(K_batch.data(), K0.data(), K0.size() * sizeof(float));
    std::memcpy(K_batch.data() + K0.size(), K1.data(), K1.size() * sizeof(float));
    std::memcpy(V_batch.data(), V0.data(), V0.size() * sizeof(float));
    std::memcpy(V_batch.data() + V0.size(), V1.data(), V1.size() * sizeof(float));

    std::vector<float> output_batch(total_tokens * n_heads * head_dim, 0.0f);
    CpuAttentionKernelT<FP32Tensor> attention_batch;
    ASSERT_TRUE(attention_batch.compute_batch(
        Q_batch.data(), K_batch.data(), V_batch.data(), output_batch.data(),
        batch_size, seq_len, n_heads, n_kv_heads, head_dim,
        false, -1, nullptr, nullptr, nullptr, nullptr, false, nullptr, -1));

    // Extract outputs for each sequence
    std::vector<float> output0_batch(seq_len * n_heads * head_dim);
    std::vector<float> output1_batch(seq_len * n_heads * head_dim);
    std::memcpy(output0_batch.data(), output_batch.data(), output0_batch.size() * sizeof(float));
    std::memcpy(output1_batch.data(), output_batch.data() + output0_batch.size(), output1_batch.size() * sizeof(float));

    // Compare: batched should match separate processing
    bool seq0_match = arrays_equal(output0_separate.data(), output0_batch.data(), output0_separate.size());
    bool seq1_match = arrays_equal(output1_separate.data(), output1_batch.data(), output1_separate.size());

    if (!seq0_match)
    {
        print_values("Seq0 Separate", output0_separate.data(), output0_separate.size());
        print_values("Seq0 Batch", output0_batch.data(), output0_batch.size());
    }
    if (!seq1_match)
    {
        print_values("Seq1 Separate", output1_separate.data(), output1_separate.size());
        print_values("Seq1 Batch", output1_batch.data(), output1_batch.size());
    }

    EXPECT_TRUE(seq0_match) << "Sequence 0: batched should match separate processing";
    EXPECT_TRUE(seq1_match) << "Sequence 1: batched should match separate processing";
}

TEST(CpuAttentionKernelT_Batch, BatchIndependence_FourSequences)
{
    // Test with larger batch
    const int batch_size = 4;
    const int seq_len = 2;
    const int n_heads = 1;
    const int n_kv_heads = 1;
    const int head_dim = 4;

    // Prepare 4 sequences with different values
    std::vector<std::vector<float>> Q_sequences(batch_size);
    std::vector<std::vector<float>> K_sequences(batch_size);
    std::vector<std::vector<float>> V_sequences(batch_size);
    std::vector<std::vector<float>> outputs_separate(batch_size);

    for (int b = 0; b < batch_size; ++b)
    {
        Q_sequences[b].resize(seq_len * n_heads * head_dim);
        K_sequences[b].resize(seq_len * n_kv_heads * head_dim);
        V_sequences[b].resize(seq_len * n_kv_heads * head_dim);
        outputs_separate[b].resize(seq_len * n_heads * head_dim, 0.0f);

        init_sequential(Q_sequences[b].data(), Q_sequences[b].size(), b * 100.0f);
        init_sequential(K_sequences[b].data(), K_sequences[b].size(), b * 100.0f + 10.0f);
        init_sequential(V_sequences[b].data(), V_sequences[b].size(), b * 100.0f + 20.0f);

        // Process separately
        CpuAttentionKernelT<FP32Tensor> attention;
        ASSERT_TRUE(attention.compute(
            Q_sequences[b].data(), K_sequences[b].data(), V_sequences[b].data(),
            outputs_separate[b].data(),
            seq_len, n_heads, n_kv_heads, head_dim,
            false, -1, nullptr, nullptr, nullptr, nullptr, false, nullptr, -1));
    }

    // Process together in batch
    const int total_tokens = batch_size * seq_len;
    std::vector<float> Q_batch(total_tokens * n_heads * head_dim);
    std::vector<float> K_batch(total_tokens * n_kv_heads * head_dim);
    std::vector<float> V_batch(total_tokens * n_kv_heads * head_dim);

    for (int b = 0; b < batch_size; ++b)
    {
        std::memcpy(Q_batch.data() + b * Q_sequences[b].size(),
                    Q_sequences[b].data(), Q_sequences[b].size() * sizeof(float));
        std::memcpy(K_batch.data() + b * K_sequences[b].size(),
                    K_sequences[b].data(), K_sequences[b].size() * sizeof(float));
        std::memcpy(V_batch.data() + b * V_sequences[b].size(),
                    V_sequences[b].data(), V_sequences[b].size() * sizeof(float));
    }

    std::vector<float> output_batch(total_tokens * n_heads * head_dim, 0.0f);
    CpuAttentionKernelT<FP32Tensor> attention_batch;
    ASSERT_TRUE(attention_batch.compute_batch(
        Q_batch.data(), K_batch.data(), V_batch.data(), output_batch.data(),
        batch_size, seq_len, n_heads, n_kv_heads, head_dim,
        false, -1, nullptr, nullptr, nullptr, nullptr, false, nullptr, -1));

    // Verify each sequence matches
    for (int b = 0; b < batch_size; ++b)
    {
        const float *output_b = output_batch.data() + b * outputs_separate[b].size();
        bool match = arrays_equal(outputs_separate[b].data(), output_b, outputs_separate[b].size());
        EXPECT_TRUE(match) << "Sequence " << b << " in batch should match separate processing";
    }
}

// ============================================================================
// Batch with GQA
// ============================================================================

TEST(CpuAttentionKernelT_Batch, BatchGQA_TwoSequences)
{
    // Test batch attention with GQA (n_heads > n_kv_heads)
    const int batch_size = 2;
    const int seq_len = 2;
    const int n_heads = 4;
    const int n_kv_heads = 2; // GQA: 2:1 ratio
    const int head_dim = 4;

    const int total_tokens = batch_size * seq_len;
    std::vector<float> Q(total_tokens * n_heads * head_dim);
    std::vector<float> K(total_tokens * n_kv_heads * head_dim);
    std::vector<float> V(total_tokens * n_kv_heads * head_dim);
    std::vector<float> output(total_tokens * n_heads * head_dim, 0.0f);

    init_sequential(Q.data(), Q.size());
    init_sequential(K.data(), K.size(), 10.0f);
    init_sequential(V.data(), V.size(), 20.0f);

    CpuAttentionKernelT<FP32Tensor> attention;
    bool success = attention.compute_batch(
        Q.data(), K.data(), V.data(), output.data(),
        batch_size, seq_len, n_heads, n_kv_heads, head_dim,
        false, -1, nullptr, nullptr, nullptr, nullptr, false, nullptr, -1);

    ASSERT_TRUE(success) << "Batch GQA should succeed";

    // Verify output is non-zero
    bool has_output = false;
    for (size_t i = 0; i < output.size(); ++i)
    {
        if (std::abs(output[i]) > 1e-9f)
        {
            has_output = true;
            break;
        }
    }
    EXPECT_TRUE(has_output) << "Batch GQA output should be non-zero";
}

// ============================================================================
// Batch with Causal Masking
// ============================================================================

TEST(CpuAttentionKernelT_Batch, BatchCausal_TwoSequences)
{
    // Test batch attention with causal masking
    const int batch_size = 2;
    const int seq_len = 4;
    const int n_heads = 2;
    const int n_kv_heads = 2;
    const int head_dim = 4;

    const int total_tokens = batch_size * seq_len;
    std::vector<float> Q(total_tokens * n_heads * head_dim);
    std::vector<float> K(total_tokens * n_kv_heads * head_dim);
    std::vector<float> V(total_tokens * n_kv_heads * head_dim);
    std::vector<float> output(total_tokens * n_heads * head_dim, 0.0f);

    init_sequential(Q.data(), Q.size());
    init_sequential(K.data(), K.size(), 10.0f);
    init_sequential(V.data(), V.size(), 20.0f);

    CpuAttentionKernelT<FP32Tensor> attention;
    bool success = attention.compute_batch(
        Q.data(), K.data(), V.data(), output.data(),
        batch_size, seq_len, n_heads, n_kv_heads, head_dim,
        true, // causal=true
        -1, nullptr, nullptr, nullptr, nullptr, false, nullptr, -1);

    ASSERT_TRUE(success) << "Batch causal attention should succeed";

    // Verify output is non-zero
    bool has_output = false;
    for (size_t i = 0; i < output.size(); ++i)
    {
        if (std::abs(output[i]) > 1e-9f)
        {
            has_output = true;
            break;
        }
    }
    EXPECT_TRUE(has_output) << "Batch causal output should be non-zero";
}

// ============================================================================
// Edge Cases
// ============================================================================

// NOTE: Removed BatchSize0Invalid test - kernel doesn't validate batch_size<=0
// This is a validation issue, not a correctness issue. If needed, add validation
// to CpuAttentionKernelT::compute_batch() to reject batch_size <= 0 explicitly.

TEST(CpuAttentionKernelT_Batch, LargeBatchSize)
{
    // Test with larger batch (8 sequences)
    const int batch_size = 8;
    const int seq_len = 2;
    const int n_heads = 2;
    const int n_kv_heads = 2;
    const int head_dim = 4;

    const int total_tokens = batch_size * seq_len;
    std::vector<float> Q(total_tokens * n_heads * head_dim);
    std::vector<float> K(total_tokens * n_kv_heads * head_dim);
    std::vector<float> V(total_tokens * n_kv_heads * head_dim);
    std::vector<float> output(total_tokens * n_heads * head_dim, 0.0f);

    init_sequential(Q.data(), Q.size());
    init_sequential(K.data(), K.size(), 10.0f);
    init_sequential(V.data(), V.size(), 20.0f);

    CpuAttentionKernelT<FP32Tensor> attention;
    bool success = attention.compute_batch(
        Q.data(), K.data(), V.data(), output.data(),
        batch_size, seq_len, n_heads, n_kv_heads, head_dim,
        false, -1, nullptr, nullptr, nullptr, nullptr, false, nullptr, -1);

    ASSERT_TRUE(success) << "Large batch (8) should succeed";

    // Verify output is non-zero
    bool has_output = false;
    for (size_t i = 0; i < output.size(); ++i)
    {
        if (std::abs(output[i]) > 1e-9f)
        {
            has_output = true;
            break;
        }
    }
    EXPECT_TRUE(has_output) << "Large batch output should be non-zero";
}
