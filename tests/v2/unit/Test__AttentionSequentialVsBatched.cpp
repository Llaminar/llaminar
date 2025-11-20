/**
 * @file Test__AttentionSequentialVsBatched.cpp
 * @brief Unit test comparing sequential vs batched attention execution
 * @author David Sanftenberg
 *
 * This test isolates the divergence seen in E2E tests by comparing:
 * - Sequential: Run each sequence separately with batch_size=1
 * - Batched: Run both sequences together with batch_size=2
 *
 * If these diverge significantly, it indicates a batched-specific bug.
 */

#include <gtest/gtest.h>
#include "v2/kernels/cpu/CpuAttentionKernelT.h"
#include "v2/tensors/Tensors.h"
#include "v2/pipelines/AttentionUtils.h"
#include <vector>
#include <cmath>
#include <iostream>

using namespace llaminar2;

class AttentionSequentialVsBatched : public ::testing::Test
{
protected:
    const int n_heads = 4;
    const int n_kv_heads = 4;
    const int head_dim = 8;
    const float tolerance = 1e-4f;

    /**
     * @brief Compare two output tensors element-wise
     */
    void compareOutputs(const std::vector<float> &expected, const std::vector<float> &actual,
                        const std::string &test_name)
    {
        ASSERT_EQ(expected.size(), actual.size()) << test_name << ": Size mismatch";

        float max_abs_diff = 0.0f;
        float mean_abs_diff = 0.0f;
        size_t mismatches = 0;

        for (size_t i = 0; i < expected.size(); ++i)
        {
            float diff = std::abs(expected[i] - actual[i]);
            max_abs_diff = std::max(max_abs_diff, diff);
            mean_abs_diff += diff;

            if (diff > tolerance)
            {
                ++mismatches;
                if (mismatches <= 10) // Print first 10 mismatches
                {
                    std::cout << test_name << ": Mismatch at index " << i
                              << ": expected=" << expected[i]
                              << ", actual=" << actual[i]
                              << ", diff=" << diff << std::endl;
                }
            }
        }

        mean_abs_diff /= expected.size();

        std::cout << test_name << " Results:" << std::endl;
        std::cout << "  Max abs diff:   " << max_abs_diff << std::endl;
        std::cout << "  Mean abs diff:  " << mean_abs_diff << std::endl;
        std::cout << "  Mismatches:     " << mismatches << " / " << expected.size() << std::endl;

        EXPECT_LT(max_abs_diff, tolerance) << test_name << ": Max diff exceeds tolerance";
        EXPECT_EQ(mismatches, 0) << test_name << ": Found mismatches";
    }
};

/**
 * @brief Test: Sequential vs batched attention with padding
 *
 * Setup:
 * - Sequence 0: 4 valid tokens
 * - Sequence 1: 2 valid tokens (padded to 4)
 *
 * Compare:
 * - Sequential: Run seq0 alone, then seq1 alone (each with batch_size=1)
 * - Batched: Run both together (batch_size=2, padded_seq_len=4)
 *
 * Expected: Results should be identical (within tolerance)
 */
TEST_F(AttentionSequentialVsBatched, PaddedBatchComparison)
{
    const int seq0_len = 4;
    const int seq1_len = 2;
    const int padded_seq_len = 4; // Max of seq0_len, seq1_len
    const int batch_size = 2;
    const int total_tokens = batch_size * padded_seq_len;

    const int qkv_dim = n_heads * head_dim;
    const int output_dim = n_heads * head_dim;

    // Initialize Q/K/V with small values for numerical stability
    std::vector<float> Q_seq0(seq0_len * qkv_dim, 0.1f);
    std::vector<float> K_seq0(seq0_len * qkv_dim, 0.1f);
    std::vector<float> V_seq0(seq0_len * qkv_dim, 0.1f);

    std::vector<float> Q_seq1(seq1_len * qkv_dim, 0.2f);
    std::vector<float> K_seq1(seq1_len * qkv_dim, 0.2f);
    std::vector<float> V_seq1(seq1_len * qkv_dim, 0.2f);

    // Add some variation
    for (int i = 0; i < seq0_len * qkv_dim; ++i)
    {
        Q_seq0[i] += 0.01f * (i % 10);
        K_seq0[i] += 0.01f * ((i + 1) % 10);
        V_seq0[i] += 0.01f * ((i + 2) % 10);
    }

    for (int i = 0; i < seq1_len * qkv_dim; ++i)
    {
        Q_seq1[i] += 0.01f * (i % 10);
        K_seq1[i] += 0.01f * ((i + 1) % 10);
        V_seq1[i] += 0.01f * ((i + 2) % 10);
    }

    // ======== Sequential Execution ========
    std::vector<float> output_seq0(seq0_len * output_dim, 0.0f);
    std::vector<float> output_seq1(seq1_len * output_dim, 0.0f);

    CpuAttentionKernelT<FP32Tensor> kernel;

    // Sequence 0: No padding, but create identity mask for consistency
    // This ensures sequential and batched paths use identical masking logic
    {
        std::vector<float> scores(n_heads * seq0_len * seq0_len);
        std::vector<float> buffer(seq0_len * qkv_dim * 2);
        std::vector<float> context(seq0_len * qkv_dim);

        auto scores_tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{scores.size()});
        auto buffer_tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{buffer.size()});
        auto context_tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{context.size()});

        // Create identity mask (all zeros = no masking, matches E2E causal=false behavior)
        std::vector<float> mask_seq0(seq0_len * seq0_len, 0.0f);
        auto mask_tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{mask_seq0.size()});
        std::memcpy(mask_tensor->mutable_data(), mask_seq0.data(), mask_seq0.size() * sizeof(float));

        std::memcpy(scores_tensor->mutable_data(), scores.data(), scores.size() * sizeof(float));

        bool success = kernel.compute(
            Q_seq0.data(), K_seq0.data(), V_seq0.data(), output_seq0.data(),
            seq0_len, n_heads, n_kv_heads, head_dim,
            /*causal=*/false, /*window_size=*/-1,
            scores_tensor.get(), buffer_tensor.get(), context_tensor.get(),
            mask_tensor.get(), /*use_bf16=*/false, /*mpi_ctx=*/nullptr, /*device_idx=*/-1);

        ASSERT_TRUE(success) << "Sequential seq0 attention failed";

        std::cout << "Sequential seq0 output[0-9]: ";
        for (int i = 0; i < 10; ++i)
        {
            std::cout << output_seq0[i] << " ";
        }
        std::cout << std::endl;
    }

    // Sequence 1: Has padding, use proper mask (only mask padding tokens)
    {
        std::vector<float> scores(n_heads * seq1_len * seq1_len);
        std::vector<float> buffer(seq1_len * qkv_dim * 2);
        std::vector<float> context(seq1_len * qkv_dim);

        auto scores_tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{scores.size()});
        auto buffer_tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{buffer.size()});
        auto context_tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{context.size()});

        // Create mask for seq1 (2 valid tokens, no padding in this view)
        // Since we're only processing 2 tokens here, no padding mask needed
        std::vector<float> mask_seq1(seq1_len * seq1_len, 0.0f);
        auto mask_tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{mask_seq1.size()});
        std::memcpy(mask_tensor->mutable_data(), mask_seq1.data(), mask_seq1.size() * sizeof(float));

        std::memcpy(scores_tensor->mutable_data(), scores.data(), scores.size() * sizeof(float));

        bool success = kernel.compute(
            Q_seq1.data(), K_seq1.data(), V_seq1.data(), output_seq1.data(),
            seq1_len, n_heads, n_kv_heads, head_dim,
            /*causal=*/false, /*window_size=*/-1,
            scores_tensor.get(), buffer_tensor.get(), context_tensor.get(),
            mask_tensor.get(), /*use_bf16=*/false, /*mpi_ctx=*/nullptr, /*device_idx=*/-1);

        ASSERT_TRUE(success) << "Sequential seq1 attention failed";
    }

    // ======== Batched Execution ========
    // Create padded Q/K/V (batch_size=2, padded_seq_len=4)
    // Layout: [batch_size, seq_len, n_heads, head_dim]
    // So batch 1 starts at offset: 1 * padded_seq_len * qkv_dim
    std::vector<float> Q_batched(total_tokens * qkv_dim, 0.0f);
    std::vector<float> K_batched(total_tokens * qkv_dim, 0.0f);
    std::vector<float> V_batched(total_tokens * qkv_dim, 0.0f);
    std::vector<float> output_batched(total_tokens * output_dim, 0.0f);

    // Copy seq0 (batch 0, tokens 0-3)
    for (int t = 0; t < seq0_len; ++t)
    {
        std::memcpy(&Q_batched[t * qkv_dim], &Q_seq0[t * qkv_dim], qkv_dim * sizeof(float));
        std::memcpy(&K_batched[t * qkv_dim], &K_seq0[t * qkv_dim], qkv_dim * sizeof(float));
        std::memcpy(&V_batched[t * qkv_dim], &V_seq0[t * qkv_dim], qkv_dim * sizeof(float));
    }

    // Copy seq1 (batch 1, tokens 0-1, leave tokens 2-3 as padding)
    // Batch 1 starts at offset: 1 * padded_seq_len * qkv_dim
    const int batch1_offset = padded_seq_len * qkv_dim;
    for (int t = 0; t < seq1_len; ++t)
    {
        std::memcpy(&Q_batched[batch1_offset + t * qkv_dim], &Q_seq1[t * qkv_dim], qkv_dim * sizeof(float));
        std::memcpy(&K_batched[batch1_offset + t * qkv_dim], &K_seq1[t * qkv_dim], qkv_dim * sizeof(float));
        std::memcpy(&V_batched[batch1_offset + t * qkv_dim], &V_seq1[t * qkv_dim], qkv_dim * sizeof(float));
    }

    // Create NON-CAUSAL padding mask (block-diagonal, seq1 has padding)
    // For non-causal attention, only mask padding tokens (not future tokens)
    std::vector<float> mask(total_tokens * total_tokens, 0.0f); // Start with all zeros
    const float neg_inf = -std::numeric_limits<float>::infinity();

    // Mask out padding tokens in seq1 (batch 1, tokens 2-3)
    // Seq1 occupies rows/cols [4,5,6,7] in the combined mask
    // Valid tokens are [4,5], padding tokens are [6,7]
    for (int i = 0; i < total_tokens; ++i)
    {
        // Mask column 6 (seq1 token 2 - padding)
        mask[i * total_tokens + 6] = neg_inf;
        // Mask column 7 (seq1 token 3 - padding)
        mask[i * total_tokens + 7] = neg_inf;
    }

    // Also mask rows 6-7 (padding tokens can't attend to anything)
    for (int j = 0; j < total_tokens; ++j)
    {
        mask[6 * total_tokens + j] = neg_inf;
        mask[7 * total_tokens + j] = neg_inf;
    }

    // DEBUG: Print mask for batch 0
    std::cout << "NON-CAUSAL Mask for batch 0 (first 4x4):" << std::endl;
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            float val = mask[i * total_tokens + j];
            std::cout << (std::isinf(val) ? "-inf" : std::to_string(val)) << " ";
        }
        std::cout << std::endl;
    }

    auto mask_tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{mask.size()});
    std::memcpy(mask_tensor->mutable_data(), mask.data(), mask.size() * sizeof(float));

    // Workspaces for batched attention
    std::vector<float> scores_batched(batch_size * n_heads * padded_seq_len * padded_seq_len);
    std::vector<float> buffer_batched(batch_size * padded_seq_len * qkv_dim * 2);
    std::vector<float> context_batched(batch_size * padded_seq_len * qkv_dim);

    auto scores_batched_tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{scores_batched.size()});
    auto buffer_batched_tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{buffer_batched.size()});
    auto context_batched_tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{context_batched.size()});

    std::memcpy(scores_batched_tensor->mutable_data(), scores_batched.data(), scores_batched.size() * sizeof(float));

    bool success_batched = kernel.compute_batch(
        Q_batched.data(), K_batched.data(), V_batched.data(), output_batched.data(),
        batch_size, padded_seq_len, n_heads, n_kv_heads, head_dim,
        /*causal=*/false, /*window_size=*/-1,
        scores_batched_tensor.get(), buffer_batched_tensor.get(), context_batched_tensor.get(),
        mask_tensor.get(), /*use_bf16=*/false, /*mpi_ctx=*/nullptr, /*device_idx=*/-1);

    ASSERT_TRUE(success_batched) << "Batched attention failed";

    // ======== Compare Results ========
    // Extract seq0 output from batched result (batch 0, tokens 0-3)
    std::vector<float> output_seq0_from_batch(seq0_len * output_dim);
    for (int t = 0; t < seq0_len; ++t)
    {
        std::memcpy(&output_seq0_from_batch[t * output_dim],
                    &output_batched[t * output_dim],
                    output_dim * sizeof(float));
    }

    // Extract seq1 output from batched result (batch 1, tokens 0-1)
    // Batch 1 starts at offset: 1 * padded_seq_len * output_dim
    std::vector<float> output_seq1_from_batch(seq1_len * output_dim);
    const int batch1_out_offset = padded_seq_len * output_dim;
    for (int t = 0; t < seq1_len; ++t)
    {
        std::memcpy(&output_seq1_from_batch[t * output_dim],
                    &output_batched[batch1_out_offset + t * output_dim],
                    output_dim * sizeof(float));
    }

    // Compare
    compareOutputs(output_seq0, output_seq0_from_batch, "Sequence 0");
    compareOutputs(output_seq1, output_seq1_from_batch, "Sequence 1");

    std::cout << "\n✓ Sequential vs Batched: PASS\n"
              << std::endl;
}
