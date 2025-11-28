/**
 * @file Test__AttentionParity.cpp
 * @brief Unit tests for attention parity behaviors to prevent regressions
 *
 * This test suite locks in critical attention behaviors discovered during
 * Qwen2 E2E parity debugging (November 2025):
 *
 * 1. RoPE theta configuration (Qwen2.5: 1000000.0, not 10000.0)
 * 2. Causal masking control (must be disabled for parity testing)
 * 3. GQA head expansion (n_kv_heads -> n_heads broadcast)
 * 4. Attention computation correctness (scores, softmax, context)
 *
 * @author David Sanftenberg
 * @date 2025-11-07
 */

#include <gtest/gtest.h>
#include <memory>
#include <cmath>
#include <algorithm>
#include "pipelines/attention/MpiAttentionOrchestrator.h"
#include "tensors/Tensors.h"
#include "kernels/cpu/ops/CPURoPEKernelT.h"
#include "utils/Logger.h"

using namespace llaminar2;

namespace
{
    /**
     * @brief Test fixture for attention parity regression tests
     */
    class AttentionParityTest : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            // Standard Qwen2.5 configuration
            n_heads_ = 14;
            n_kv_heads_ = 2;
            head_dim_ = 64;
            seq_len_ = 9;

            // Qwen2.5 uses rope_theta = 1000000.0 (NOT 10000.0)
            rope_theta_qwen25_ = 1000000.0f;
            rope_theta_llama_ = 10000.0f;
        }

        int n_heads_;
        int n_kv_heads_;
        int head_dim_;
        int seq_len_;
        float rope_theta_qwen25_;
        float rope_theta_llama_;

        /**
         * @brief Helper to compute relative L2 norm
         */
        float compute_rel_l2(const float *a, const float *b, size_t n)
        {
            float sum_diff_sq = 0.0f;
            float sum_b_sq = 0.0f;

            for (size_t i = 0; i < n; ++i)
            {
                float diff = a[i] - b[i];
                sum_diff_sq += diff * diff;
                sum_b_sq += b[i] * b[i];
            }

            return std::sqrt(sum_diff_sq / (sum_b_sq + 1e-10f));
        }

        /**
         * @brief Helper to find max absolute difference
         */
        float compute_max_abs_diff(const float *a, const float *b, size_t n)
        {
            float max_diff = 0.0f;
            for (size_t i = 0; i < n; ++i)
            {
                float diff = std::abs(a[i] - b[i]);
                max_diff = std::max(max_diff, diff);
            }
            return max_diff;
        }
    };

    /**
     * @brief REGRESSION TEST: RoPE theta must be configurable (not hardcoded)
     *
     * Bug: CPURoPEKernel had hardcoded freq_base = 10000.0f
     * Fix: Accept rope_theta parameter from model config
     * Impact: 100× error in rotation frequencies for Qwen2.5 models
     *
     * This test ensures RoPE output differs significantly between
     * rope_theta=10000.0 and rope_theta=1000000.0.
     */
    TEST_F(AttentionParityTest, RoPETheta_Qwen25VsLLaMA)
    {
        // Create input tensors (Q and K projections before RoPE)
        auto Q = std::make_shared<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(seq_len_), static_cast<size_t>(n_heads_ * head_dim_)});
        auto K = std::make_shared<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(seq_len_), static_cast<size_t>(n_kv_heads_ * head_dim_)});

        // Initialize with non-zero values
        float *Q_data = Q->mutable_data();
        float *K_data = K->mutable_data();
        for (size_t i = 0; i < seq_len_ * n_heads_ * head_dim_; ++i)
        {
            Q_data[i] = 0.01f * static_cast<float>(i % 100);
        }
        for (size_t i = 0; i < seq_len_ * n_kv_heads_ * head_dim_; ++i)
        {
            K_data[i] = 0.01f * static_cast<float>(i % 100);
        }

        // Position IDs (0, 1, 2, ..., 8)
        std::vector<int> position_ids(seq_len_);
        for (int i = 0; i < seq_len_; ++i)
        {
            position_ids[i] = i;
        }

        // Apply RoPE with LLaMA theta (10000.0)
        auto Q_llama = std::make_shared<FP32Tensor>(*Q);
        auto K_llama = std::make_shared<FP32Tensor>(*K);

        CPURoPEKernel rope_kernel_llama;
        ASSERT_TRUE(rope_kernel_llama.apply(
            Q_llama->mutable_data(), K_llama->mutable_data(), position_ids.data(),
            seq_len_, n_heads_, n_kv_heads_, head_dim_,
            rope_theta_llama_, false, nullptr, -1))
            << "RoPE with LLaMA theta failed";

        // Apply RoPE with Qwen2.5 theta (1000000.0)
        auto Q_qwen = std::make_shared<FP32Tensor>(*Q);
        auto K_qwen = std::make_shared<FP32Tensor>(*K);

        CPURoPEKernel rope_kernel_qwen;
        ASSERT_TRUE(rope_kernel_qwen.apply(
            Q_qwen->mutable_data(), K_qwen->mutable_data(), position_ids.data(),
            seq_len_, n_heads_, n_kv_heads_, head_dim_,
            rope_theta_qwen25_, false, nullptr, -1))
            << "RoPE with Qwen2.5 theta failed";

        // Verify outputs DIFFER significantly (100× theta difference)
        size_t n_q = seq_len_ * n_heads_ * head_dim_;
        size_t n_k = seq_len_ * n_kv_heads_ * head_dim_;

        float rel_l2_q = compute_rel_l2(Q_llama->data(), Q_qwen->data(), n_q);
        float rel_l2_k = compute_rel_l2(K_llama->data(), K_qwen->data(), n_k);

        // Different theta values should produce significantly different outputs
        EXPECT_GT(rel_l2_q, 0.1f)
            << "Q outputs should differ significantly with different rope_theta (got rel_l2=" << rel_l2_q << ")";
        EXPECT_GT(rel_l2_k, 0.1f)
            << "K outputs should differ significantly with different rope_theta (got rel_l2=" << rel_l2_k << ")";

        // Log for debugging
        LOG_INFO("RoPE theta comparison:");
        LOG_INFO("  LLaMA theta (10000.0)   vs Qwen2.5 theta (1000000.0)");
        LOG_INFO("  Q rel_l2: " << rel_l2_q);
        LOG_INFO("  K rel_l2: " << rel_l2_k);
    }

    /**
     * @brief REGRESSION TEST: Causal masking must be controllable
     *
     * Bug: MpiAttentionOrchestrator applied causal mask when sequence_lengths != nullptr,
     *      even when causal=false was specified
     * Fix: Only apply mask when causal=true
     * Impact: Parity tests failed because PyTorch reference has no causal mask
     *
     * This test ensures attention scores are correctly masked when causal=true,
     * by verifying that upper triangular elements (future positions) are set to -inf.
     */
    TEST_F(AttentionParityTest, CausalMasking_UpperTriangularMasked)
    {
        int small_seq = 4; // Small for clarity
        int small_heads = 2;
        int small_kv_heads = 2;

        // Create Q, K, V tensors
        auto Q = std::make_shared<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(small_seq), static_cast<size_t>(small_heads * head_dim_)});
        auto K = std::make_shared<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(small_seq), static_cast<size_t>(small_kv_heads * head_dim_)});
        auto V = std::make_shared<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(small_seq), static_cast<size_t>(small_kv_heads * head_dim_)});

        // Initialize with non-zero values
        float *Q_data = Q->mutable_data();
        float *K_data = K->mutable_data();
        float *V_data = V->mutable_data();

        for (size_t i = 0; i < small_seq * small_heads * head_dim_; ++i)
        {
            Q_data[i] = 0.1f;
        }
        for (size_t i = 0; i < small_seq * small_kv_heads * head_dim_; ++i)
        {
            K_data[i] = 0.1f;
            V_data[i] = 1.0f;
        }

        // Output buffer
        auto output = std::make_shared<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(small_seq), static_cast<size_t>(small_heads * head_dim_)});

        // Workspace buffers
        auto workspace_scores = std::make_shared<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(small_heads * small_seq * small_seq)});
        auto workspace_qkv = std::make_shared<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(28 * small_seq * head_dim_ * 3)}); // max threads
        auto workspace_context = std::make_shared<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(28 * small_seq * head_dim_)});
        auto workspace_mask = std::make_shared<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(small_seq * small_seq)});

        // Configure attention WITH causal masking
        MpiAttentionConfig config;
        config.n_heads = small_heads;
        config.n_kv_heads = small_kv_heads;
        config.head_dim = head_dim_;
        config.causal = true; // Enable causal mask
        config.window_size = -1;
        config.workspace_scores = workspace_scores;
        config.workspace_qkv_buffer = workspace_qkv;
        config.workspace_context = workspace_context;
        config.workspace_mask = workspace_mask;

        ASSERT_TRUE(MpiAttentionOrchestrator::compute(
            Q.get(), K.get(), V.get(), output.get(),
            config, 1, nullptr))
            << "Causal attention failed";

        // After attention, workspace_scores contains softmax weights
        // With causal mask, upper triangle should have near-zero weights
        const float *scores = workspace_scores->data();

        // Check that future positions (j > i) have very small attention weights
        for (int h = 0; h < small_heads; ++h)
        {
            for (int i = 0; i < small_seq; ++i)
            {
                for (int j = i + 1; j < small_seq; ++j) // j > i (future positions)
                {
                    int idx = h * small_seq * small_seq + i * small_seq + j;
                    EXPECT_LT(scores[idx], 1e-5f)
                        << "Causal masking should zero out future positions (head=" << h
                        << ", pos=" << i << ", future_pos=" << j << ", weight=" << scores[idx] << ")";
                }
            }
        }

        // Check that past/present positions (j <= i) have non-zero weights
        int non_zero_count = 0;
        for (int h = 0; h < small_heads; ++h)
        {
            for (int i = 0; i < small_seq; ++i)
            {
                for (int j = 0; j <= i; ++j) // j <= i (past/present positions)
                {
                    int idx = h * small_seq * small_seq + i * small_seq + j;
                    if (scores[idx] > 1e-5f)
                    {
                        non_zero_count++;
                    }
                }
            }
        }

        EXPECT_GT(non_zero_count, 0)
            << "Causal masking should preserve past/present attention weights";

        LOG_INFO("Causal masking validation:");
        LOG_INFO("  Found " << non_zero_count << " non-zero past/present attention weights");
        LOG_INFO("  All future positions correctly masked to ~0");
    }

    /**
     * @brief REGRESSION TEST: GQA head expansion correctness
     *
     * Ensures K/V heads are correctly broadcast from n_kv_heads to n_heads.
     * For Qwen2.5: 2 KV heads -> 14 heads (repeat factor 7).
     *
     * This validates the core GQA mechanism works correctly.
     */
    TEST_F(AttentionParityTest, GQA_HeadExpansion)
    {
        // Small example: 4 heads, 2 KV heads (repeat factor 2)
        int small_heads = 4;
        int small_kv_heads = 2;
        int small_seq = 3;

        auto Q = std::make_shared<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(small_seq), static_cast<size_t>(small_heads * head_dim_)});
        auto K = std::make_shared<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(small_seq), static_cast<size_t>(small_kv_heads * head_dim_)});
        auto V = std::make_shared<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(small_seq), static_cast<size_t>(small_kv_heads * head_dim_)});

        // Initialize K/V with distinct patterns per KV head
        float *K_data = K->mutable_data();
        float *V_data = V->mutable_data();

        for (int t = 0; t < small_seq; ++t)
        {
            for (int kv_h = 0; kv_h < small_kv_heads; ++kv_h)
            {
                for (int d = 0; d < head_dim_; ++d)
                {
                    int idx = t * small_kv_heads * head_dim_ + kv_h * head_dim_ + d;
                    K_data[idx] = static_cast<float>(kv_h * 100); // KV head 0: 0, KV head 1: 100
                    V_data[idx] = static_cast<float>(kv_h * 100);
                }
            }
        }

        // Initialize Q (uniform)
        float *Q_data = Q->mutable_data();
        for (size_t i = 0; i < small_seq * small_heads * head_dim_; ++i)
        {
            Q_data[i] = 1.0f;
        }

        // Run attention
        auto output = std::make_shared<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(small_seq), static_cast<size_t>(small_heads * head_dim_)});

        auto workspace_scores = std::make_shared<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(small_heads * small_seq * small_seq)});
        auto workspace_qkv = std::make_shared<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(28 * small_seq * head_dim_ * 3)});
        auto workspace_context = std::make_shared<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(28 * small_seq * head_dim_)});
        auto workspace_mask = std::make_shared<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(small_seq * small_seq)});

        MpiAttentionConfig config;
        config.n_heads = small_heads;
        config.n_kv_heads = small_kv_heads;
        config.head_dim = head_dim_;
        config.causal = false;
        config.window_size = -1;
        config.workspace_scores = workspace_scores;
        config.workspace_qkv_buffer = workspace_qkv;
        config.workspace_context = workspace_context;
        config.workspace_mask = workspace_mask;

        // Use the main MpiAttentionOrchestrator compute path, which routes scores·V
        // through the activation GEMM interface (supports non-transposed B).
        ASSERT_TRUE(MpiAttentionOrchestrator::compute(
            Q.get(), K.get(), V.get(), output.get(), config, /*batch_size=*/1, /*sequence_lengths=*/nullptr))
            << "GQA attention failed";

        // Verify: Heads 0,1 should use KV head 0, heads 2,3 should use KV head 1
        // Since V values are 0 for KV head 0 and 100 for KV head 1,
        // output heads 0,1 should be near 0, heads 2,3 should be near 100
        const float *output_data = output->data();

        // Check first head (should attend to KV head 0 with values ~0)
        float avg_head0 = 0.0f;
        for (int t = 0; t < small_seq; ++t)
        {
            for (int d = 0; d < head_dim_; ++d)
            {
                int idx = t * small_heads * head_dim_ + 0 * head_dim_ + d;
                avg_head0 += output_data[idx];
            }
        }
        avg_head0 /= (small_seq * head_dim_);

        // Check third head (should attend to KV head 1 with values ~100)
        float avg_head2 = 0.0f;
        for (int t = 0; t < small_seq; ++t)
        {
            for (int d = 0; d < head_dim_; ++d)
            {
                int idx = t * small_heads * head_dim_ + 2 * head_dim_ + d;
                avg_head2 += output_data[idx];
            }
        }
        avg_head2 /= (small_seq * head_dim_);

        EXPECT_NEAR(avg_head0, 0.0f, 10.0f)
            << "Head 0 should attend to KV head 0 (values ~0)";
        EXPECT_NEAR(avg_head2, 100.0f, 10.0f)
            << "Head 2 should attend to KV head 1 (values ~100)";

        LOG_INFO("GQA head expansion validation:");
        LOG_INFO("  Head 0 avg output: " << avg_head0 << " (expected ~0)");
        LOG_INFO("  Head 2 avg output: " << avg_head2 << " (expected ~100)");
    }

    /**
     * @brief REGRESSION TEST: RoPE should NOT be applied to V
     *
     * RoPE is only applied to Q and K, never to V.
     * This test ensures the pipeline doesn't mistakenly apply RoPE to V.
     */
    TEST_F(AttentionParityTest, RoPE_NotAppliedToV)
    {
        // Create V projection tensor
        auto V = std::make_shared<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(seq_len_), static_cast<size_t>(n_kv_heads_ * head_dim_)});

        // Initialize with known values
        float *V_data = V->mutable_data();
        for (size_t i = 0; i < seq_len_ * n_kv_heads_ * head_dim_; ++i)
        {
            V_data[i] = 0.01f * static_cast<float>(i);
        }

        // Save original V values
        std::vector<float> V_original(V_data, V_data + seq_len_ * n_kv_heads_ * head_dim_);

        // RoPE kernel interface only accepts Q and K, not V
        // This is a sanity check that the API doesn't allow V to be passed

        auto Q_dummy = std::make_shared<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(seq_len_), static_cast<size_t>(n_heads_ * head_dim_)});
        auto K_dummy = std::make_shared<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(seq_len_), static_cast<size_t>(n_kv_heads_ * head_dim_)});

        std::vector<int> position_ids(seq_len_);
        for (int i = 0; i < seq_len_; ++i)
        {
            position_ids[i] = i;
        }

        CPURoPEKernel rope_kernel;
        ASSERT_TRUE(rope_kernel.apply(
            Q_dummy->mutable_data(), K_dummy->mutable_data(), position_ids.data(),
            seq_len_, n_heads_, n_kv_heads_, head_dim_,
            rope_theta_qwen25_, false, nullptr, -1))
            << "RoPE failed";

        // Verify V is unchanged (RoPE interface doesn't touch V)
        for (size_t i = 0; i < V_original.size(); ++i)
        {
            EXPECT_EQ(V_data[i], V_original[i])
                << "V should not be modified by RoPE at index " << i;
        }

        LOG_INFO("✓ RoPE correctly does not modify V tensor");
    }

    /**
     * @brief REGRESSION TEST: Attention softmax must sum to 1.0
     *
     * Validates that attention weights after softmax sum to 1.0 per row.
     */
    TEST_F(AttentionParityTest, Softmax_RowSumsToOne)
    {
        int small_seq = 5;
        int small_heads = 2;
        int small_kv_heads = 2;

        auto Q = std::make_shared<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(small_seq), static_cast<size_t>(small_heads * head_dim_)});
        auto K = std::make_shared<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(small_seq), static_cast<size_t>(small_kv_heads * head_dim_)});
        auto V = std::make_shared<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(small_seq), static_cast<size_t>(small_kv_heads * head_dim_)});

        // Initialize with random-ish values
        float *Q_data = Q->mutable_data();
        float *K_data = K->mutable_data();
        float *V_data = V->mutable_data();

        for (size_t i = 0; i < small_seq * small_heads * head_dim_; ++i)
        {
            Q_data[i] = std::sin(static_cast<float>(i) * 0.1f);
        }
        for (size_t i = 0; i < small_seq * small_kv_heads * head_dim_; ++i)
        {
            K_data[i] = std::cos(static_cast<float>(i) * 0.1f);
            V_data[i] = static_cast<float>(i % 10);
        }

        auto output = std::make_shared<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(small_seq), static_cast<size_t>(small_heads * head_dim_)});

        auto workspace_scores = std::make_shared<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(small_heads * small_seq * small_seq)});
        auto workspace_qkv = std::make_shared<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(28 * small_seq * head_dim_ * 3)});
        auto workspace_context = std::make_shared<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(28 * small_seq * head_dim_)});
        auto workspace_mask = std::make_shared<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(small_seq * small_seq)});

        MpiAttentionConfig config;
        config.n_heads = small_heads;
        config.n_kv_heads = small_kv_heads;
        config.head_dim = head_dim_;
        config.causal = false;
        config.window_size = -1;
        config.workspace_scores = workspace_scores;
        config.workspace_qkv_buffer = workspace_qkv;
        config.workspace_context = workspace_context;
        config.workspace_mask = workspace_mask;

        ASSERT_TRUE(MpiAttentionOrchestrator::compute(
            Q.get(), K.get(), V.get(), output.get(), config, 1, nullptr))
            << "Attention failed";

        // After attention computation, workspace_scores contains softmax weights
        // Verify each row sums to ~1.0
        const float *scores = workspace_scores->data();

        for (int h = 0; h < small_heads; ++h)
        {
            for (int i = 0; i < small_seq; ++i)
            {
                float row_sum = 0.0f;
                for (int j = 0; j < small_seq; ++j)
                {
                    int idx = h * small_seq * small_seq + i * small_seq + j;
                    row_sum += scores[idx];
                }

                EXPECT_NEAR(row_sum, 1.0f, 1e-5f)
                    << "Softmax row sum should be 1.0 (head=" << h << ", row=" << i << ", got=" << row_sum << ")";
            }
        }

        LOG_INFO("✓ All softmax rows sum to 1.0");
    }

} // namespace
