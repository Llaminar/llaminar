/**
 * @file TestKGatheringAndRopeGqa.cpp
 * @brief Tests for K tensor MPI gathering order and RoPE GQA head indexing
 * @author David Sanftenberg
 *
 * These tests verify two critical hypotheses from the ROPE_APPLICATION failure investigation:
 *
 * Hypothesis 1: MPI_Allgather concatenates K heads in wrong order
 *   - Expected: [KV-head0_all_dims | KV-head1_all_dims] (sequential heads)
 *   - Bug would be: [rank0_KV-head_dims | rank1_KV-head_dims] (rank-based)
 *
 * Hypothesis 2: RoPE applies wrong head indices for GQA (q_heads ≠ k_heads)
 *   - Expected: K heads use indices 0, 1 with proper theta calculation
 *   - Bug would be: K heads use Q head indices or wrong theta values
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <cmath>
#include <vector>
#include <memory>
#include <iostream>
#include <iomanip>

#include "tensors/TensorBase.h"
#include "tensors/SimpleTensor.h"
#include "operators/common/AttentionPrimitives.h"
#include "Logger.h"

using namespace llaminar;

namespace
{

    /**
     * @brief Test fixture for K gathering and RoPE GQA tests
     */
    class KGatheringAndRopeGQATest : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
            MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
        }

        void TearDown() override
        {
            // Nothing to clean up
        }

        int rank_ = 0;
        int world_size_ = 1;
    };

    /**
     * @brief Test 1: Verify MPI_Allgather produces sequential head order for K tensor
     *
     * Test Setup:
     * - 2 ranks, each with 1 KV head (64 dims)
     * - Initialize with rank-specific patterns: rank0=[1,1,...], rank1=[2,2,...]
     *
     * Expected Result:
     * - After Allgather: [1,1,...1,1 (64 times) | 2,2,...2,2 (64 times)]
     * - This is sequential head order: [head0_all_dims | head1_all_dims]
     *
     * Bug Detection:
     * - If gathering is rank-based instead of head-based, pattern would be different
     */
    TEST_F(KGatheringAndRopeGQATest, AllgatherProducesSequentialHeadOrder)
    {
        if (world_size_ != 2)
        {
            GTEST_SKIP() << "This test requires exactly 2 MPI ranks";
        }

        const int n_kv_heads = 2;                          // Total KV heads across all ranks
        const int head_dim = 64;                           // Dimension per head
        const int seq_len = 5;                             // Sequence length
        const int local_kv_heads = 1;                      // Each rank has 1 KV head
        const int local_k_dim = local_kv_heads * head_dim; // 64

        // Create local K tensor with rank-specific pattern
        std::vector<float> local_k_data(seq_len * local_k_dim);
        float rank_marker = static_cast<float>(rank_ + 1); // rank0=1.0, rank1=2.0

        for (int s = 0; s < seq_len; ++s)
        {
            for (int d = 0; d < local_k_dim; ++d)
            {
                // Add small position-based offset for easier debugging
                local_k_data[s * local_k_dim + d] = rank_marker + 0.01f * d;
            }
        }

        if (rank_ == 0)
        {
            LOG_INFO("[K_GATHER_TEST] Local K pattern on rank 0: " << local_k_data[0]
                                                                   << ", " << local_k_data[1] << ", ..., " << local_k_data[63]);
        }

        // Allocate buffer for gathered K tensor
        const int global_k_dim = n_kv_heads * head_dim; // 128
        std::vector<float> global_k_data(seq_len * global_k_dim);

        // Perform MPI_Allgather for each sequence position
        for (int s = 0; s < seq_len; ++s)
        {
            const float *local_row = local_k_data.data() + s * local_k_dim;
            float *global_row = global_k_data.data() + s * global_k_dim;

            MPI_Allgather(
                local_row, local_k_dim, MPI_FLOAT,
                global_row, local_k_dim, MPI_FLOAT,
                MPI_COMM_WORLD);
        }

        // Verify gathered tensor has sequential head order
        if (rank_ == 0)
        {
            LOG_INFO("[K_GATHER_TEST] Verifying sequential head order in gathered K tensor");

            // Check first sequence position
            const float *first_row = global_k_data.data();

            // Head 0 should be all 1.0 (+ small offset)
            bool head0_correct = true;
            for (int d = 0; d < head_dim; ++d)
            {
                float expected = 1.0f + 0.01f * d;
                float actual = first_row[d];
                if (std::abs(actual - expected) > 1e-5f)
                {
                    LOG_ERROR("Head 0, dim " << d << ": expected " << expected << ", got " << actual);
                    head0_correct = false;
                }
            }
            EXPECT_TRUE(head0_correct) << "KV-Head 0 should have rank0 pattern (1.0 + offset)";

            // Head 1 should be all 2.0 (+ small offset)
            bool head1_correct = true;
            for (int d = 0; d < head_dim; ++d)
            {
                float expected = 2.0f + 0.01f * d;
                float actual = first_row[head_dim + d];
                if (std::abs(actual - expected) > 1e-5f)
                {
                    LOG_ERROR("Head 1, dim " << d << ": expected " << expected << ", got " << actual);
                    head1_correct = false;
                }
            }
            EXPECT_TRUE(head1_correct) << "KV-Head 1 should have rank1 pattern (2.0 + offset)";

            // Log sample values
            LOG_INFO("[K_GATHER_TEST] Gathered K[0, 0:70]: "
                     << first_row[0] << ", " << first_row[1] << ", ..., "
                     << first_row[63] << " | " << first_row[64] << ", " << first_row[65]
                     << ", ..., " << first_row[127]);

            if (head0_correct && head1_correct)
            {
                LOG_INFO("[K_GATHER_TEST] ✓ PASS: K tensor has sequential head order [head0 | head1]");
            }
            else
            {
                LOG_ERROR("[K_GATHER_TEST] ✗ FAIL: K tensor does NOT have sequential head order");
            }
        }
    }

    /**
     * @brief Test 2: Verify RoPE uses correct head indices for GQA K tensor
     *
     * Test Setup:
     * - Q: 4 heads × 8 dims = 32 total (simpler than real 14×64)
     * - K: 2 heads × 8 dims = 16 total (GQA with fewer K heads)
     * - Single rank for simplicity (tests RoPE logic, not gathering)
     *
     * Expected Behavior:
     * - K head 0 should use theta based on head_index=0
     * - K head 1 should use theta based on head_index=1
     * - NOT head_index from Q heads (0-3)
     *
     * Verification:
     * - Apply RoPE and check that results are independent of q_heads count
     * - Compare with reference implementation using explicit K head indexing
     */
    TEST_F(KGatheringAndRopeGQATest, RopeUsesCorrectKVHeadIndices)
    {
        const int q_heads = 4;
        const int k_heads = 2; // GQA: fewer K heads
        const int head_dim = 8;
        const int seq_len = 3;
        const int freq_base = 10000;
        const int n_past = 0;

        // Create Q and K tensors with known patterns
        auto q = std::make_shared<SimpleTensor>(std::vector<int>{seq_len, q_heads * head_dim});
        auto k = std::make_shared<SimpleTensor>(std::vector<int>{seq_len, k_heads * head_dim});

        // Initialize with simple incremental pattern
        for (int s = 0; s < seq_len; ++s)
        {
            for (int h = 0; h < q_heads; ++h)
            {
                for (int d = 0; d < head_dim; ++d)
                {
                    int idx = s * (q_heads * head_dim) + h * head_dim + d;
                    q->data()[idx] = static_cast<float>(idx);
                }
            }
            for (int h = 0; h < k_heads; ++h)
            {
                for (int d = 0; d < head_dim; ++d)
                {
                    int idx = s * (k_heads * head_dim) + h * head_dim + d;
                    k->data()[idx] = static_cast<float>(idx + 1000); // Offset to distinguish from Q
                }
            }
        }

        // Save original K values for comparison
        std::vector<float> k_original(k->data(), k->data() + k->size());

        // Apply RoPE using llaminar::attn namespace
        llaminar::attn::apply_rope(q->data(), k->data(),
                                   seq_len, head_dim, q_heads, k_heads, n_past, freq_base);

        // Manually compute expected RoPE for K to verify head indexing
        // For each K head, we should use theta_i = freq_base^(-2i/head_dim) for i in [0, head_dim/2)

        if (rank_ == 0)
        {
            LOG_INFO("[ROPE_GQA_TEST] Verifying K head indices in RoPE calculation");

            bool all_correct = true;

            // Implementation note: llaminar currently uses split-half pairing
            // (real[0:half], imag[0:half]) with pair i mapped to (i, i+half), not adjacent (2i,2i+1).
            int half = head_dim / 2;
            for (int kv_head = 0; kv_head < k_heads; ++kv_head) {
                LOG_INFO("[ROPE_GQA_TEST] Checking KV-head " << kv_head);
                for (int pos = 0; pos < seq_len; ++pos) {
                    for (int i = 0; i < half; ++i) {
                        float theta = 1.0f / std::pow(static_cast<float>(freq_base), (2.0f * i) / head_dim);
                        float angle = (pos + n_past) * theta;
                        float c = std::cos(angle);
                        float s = std::sin(angle);
                        int base_idx = pos * (k_heads * head_dim) + kv_head * head_dim;
                        float x_orig = k_original[base_idx + i];
                        float y_orig = k_original[base_idx + i + half];
                        float expected_x = x_orig * c - y_orig * s;
                        float expected_y = x_orig * s + y_orig * c;
                        float actual_x = k->data()[base_idx + i];
                        float actual_y = k->data()[base_idx + i + half];
                        float tol = 1e-4f;
                        if (std::abs(actual_x - expected_x) > tol || std::abs(actual_y - expected_y) > tol) {
                            LOG_ERROR("KV-head " << kv_head << ", pos " << pos << ", pair " << i
                                      << ": expected (" << expected_x << ", " << expected_y
                                      << "), got (" << actual_x << ", " << actual_y << ")");
                            all_correct = false;
                        }
                    }
                }
            }

            EXPECT_TRUE(all_correct) << "RoPE should use correct K head indices (0, 1), not Q head indices";

            if (all_correct)
            {
                LOG_INFO("[ROPE_GQA_TEST] ✓ PASS: RoPE uses correct KV-head indices for GQA");
            }
            else
            {
                LOG_ERROR("[ROPE_GQA_TEST] ✗ FAIL: RoPE uses WRONG head indices for K tensor");
            }
        }
    }

    /**
     * @brief Test 3: Verify RoPE GQA with different q_heads/k_heads ratios
     *
     * This test ensures that K head rotation is independent of Q head count
     * by comparing results with different q_heads but same k_heads.
     */
    TEST_F(KGatheringAndRopeGQATest, RopeKIndependentOfQHeadCount)
    {
        const int k_heads = 2;
        const int head_dim = 8;
        const int seq_len = 3;
        const int freq_base = 10000;
        const int n_past = 0;

        // Test with different Q head counts but same K
        std::vector<int> q_heads_variants = {2, 4, 8, 14}; // Including real Qwen config
        std::vector<std::vector<float>> k_results;

        for (int q_heads : q_heads_variants)
        {
            // Create fresh K tensor with same initial values
            auto k = std::make_shared<SimpleTensor>(std::vector<int>{seq_len, k_heads * head_dim});
            for (int i = 0; i < k->size(); ++i)
            {
                k->data()[i] = static_cast<float>(i + 100);
            }

            // Create Q tensor (values don't matter for this test)
            auto q = std::make_shared<SimpleTensor>(std::vector<int>{seq_len, q_heads * head_dim});
            for (int i = 0; i < q->size(); ++i)
            {
                q->data()[i] = static_cast<float>(i);
            }

            // Apply RoPE
            llaminar::attn::apply_rope(q->data(), k->data(),
                                       seq_len, head_dim, q_heads, k_heads, n_past, freq_base);

            // Store K result
            k_results.push_back(std::vector<float>(k->data(), k->data() + k->size()));
        }

        // Verify all K results are identical
        if (rank_ == 0)
        {
            LOG_INFO("[ROPE_INDEPENDENCE_TEST] Comparing K results across q_heads variants");

            bool all_identical = true;
            const auto &reference = k_results[0];

            for (size_t variant = 1; variant < k_results.size(); ++variant)
            {
                const auto &current = k_results[variant];

                for (size_t i = 0; i < reference.size(); ++i)
                {
                    if (std::abs(reference[i] - current[i]) > 1e-5f)
                    {
                        LOG_ERROR("K differs at index " << i
                                                        << " between q_heads=" << q_heads_variants[0]
                                                        << " and q_heads=" << q_heads_variants[variant]
                                                        << ": " << reference[i] << " vs " << current[i]);
                        all_identical = false;
                        break;
                    }
                }

                if (!all_identical)
                    break;
            }

            EXPECT_TRUE(all_identical)
                << "K tensor RoPE result should be independent of Q head count";

            if (all_identical)
            {
                LOG_INFO("[ROPE_INDEPENDENCE_TEST] ✓ PASS: K RoPE is independent of q_heads count");
            }
            else
            {
                LOG_ERROR("[ROPE_INDEPENDENCE_TEST] ✗ FAIL: K RoPE depends on q_heads (BUG!)");
            }
        }
    }

    /**
     * @brief Test 4: Multi-rank K gathering with RoPE applied
     *
     * This combines both hypotheses: verify that after local RoPE computation,
     * MPI_Allgather produces the same result as sequential RoPE on full K tensor.
     */
    TEST_F(KGatheringAndRopeGQATest, MultiRankKGatheringAfterRope)
    {
        if (world_size_ != 2)
        {
            GTEST_SKIP() << "This test requires exactly 2 MPI ranks";
        }

        const int n_kv_heads = 2;
        const int head_dim = 64;
        const int seq_len = 5;
        const int freq_base = 10000;
        const int n_past = 0;
        const int local_kv_heads = 1; // Each rank has 1 KV head
        const int local_k_dim = local_kv_heads * head_dim;
        const int global_k_dim = n_kv_heads * head_dim;

        // Create reference K tensor on rank 0 (full sequential)
        std::vector<float> reference_k;
        if (rank_ == 0)
        {
            reference_k.resize(seq_len * global_k_dim);
            // Initialize with known pattern: each head has distinct values
            for (int s = 0; s < seq_len; ++s)
            {
                for (int h = 0; h < n_kv_heads; ++h)
                {
                    for (int d = 0; d < head_dim; ++d)
                    {
                        reference_k[s * global_k_dim + h * head_dim + d] =
                            static_cast<float>(h * 100 + d); // head0: 0-63, head1: 100-163
                    }
                }
            }

            // Apply RoPE to full K tensor (sequential, non-distributed)
            // Note: For K-only RoPE, we pass nullptr for Q
            auto k_tensor = std::make_shared<SimpleTensor>(std::vector<int>{seq_len, global_k_dim});
            std::copy(reference_k.begin(), reference_k.end(), k_tensor->data());

            // We need a dummy Q for the API, but it won't affect K
            auto dummy_q = std::make_shared<SimpleTensor>(std::vector<int>{seq_len, global_k_dim});

            llaminar::attn::apply_rope(dummy_q->data(), k_tensor->data(),
                                       seq_len, head_dim, n_kv_heads, n_kv_heads, n_past, freq_base);

            std::copy(k_tensor->data(), k_tensor->data() + k_tensor->size(), reference_k.begin());

            LOG_INFO("[MULTI_RANK_ROPE_TEST] Reference K after sequential RoPE computed");
        }

        // Create local K tensor on each rank
        std::vector<float> local_k(seq_len * local_k_dim);
        int my_kv_head = rank_; // rank0=head0, rank1=head1

        for (int s = 0; s < seq_len; ++s)
        {
            for (int d = 0; d < head_dim; ++d)
            {
                local_k[s * local_k_dim + d] =
                    static_cast<float>(my_kv_head * 100 + d);
            }
        }

        // Apply RoPE to local K tensor
        auto local_k_tensor = std::make_shared<SimpleTensor>(std::vector<int>{seq_len, local_k_dim});
        std::copy(local_k.begin(), local_k.end(), local_k_tensor->data());

        auto dummy_q = std::make_shared<SimpleTensor>(std::vector<int>{seq_len, local_k_dim});

        llaminar::attn::apply_rope(dummy_q->data(), local_k_tensor->data(),
                                   seq_len, head_dim, local_kv_heads, local_kv_heads, n_past, freq_base);

        std::copy(local_k_tensor->data(), local_k_tensor->data() + local_k_tensor->size(),
                  local_k.begin());

        // Gather K tensors using MPI_Allgather
        std::vector<float> gathered_k(seq_len * global_k_dim);

        for (int s = 0; s < seq_len; ++s)
        {
            const float *local_row = local_k.data() + s * local_k_dim;
            float *global_row = gathered_k.data() + s * global_k_dim;

            MPI_Allgather(
                local_row, local_k_dim, MPI_FLOAT,
                global_row, local_k_dim, MPI_FLOAT,
                MPI_COMM_WORLD);
        }

        // Compare gathered K with reference K
        if (rank_ == 0)
        {
            LOG_INFO("[MULTI_RANK_ROPE_TEST] Comparing gathered K with reference");

            bool all_match = true;
            float max_diff = 0.0f;
            int max_diff_idx = 0;

            for (size_t i = 0; i < reference_k.size(); ++i)
            {
                float diff = std::abs(gathered_k[i] - reference_k[i]);
                if (diff > max_diff)
                {
                    max_diff = diff;
                    max_diff_idx = i;
                }
                if (diff > 1e-4f)
                {
                    int s = i / global_k_dim;
                    int remainder = i % global_k_dim;
                    int h = remainder / head_dim;
                    int d = remainder % head_dim;
                    LOG_ERROR("Mismatch at [" << s << ", head=" << h << ", dim=" << d
                                              << "]: reference=" << reference_k[i]
                                              << ", gathered=" << gathered_k[i]
                                              << ", diff=" << diff);
                    all_match = false;
                }
            }

            LOG_INFO("[MULTI_RANK_ROPE_TEST] Max difference: " << max_diff
                                                               << " at index " << max_diff_idx);

            EXPECT_TRUE(all_match)
                << "Distributed RoPE + Allgather should match sequential RoPE";

            if (all_match)
            {
                LOG_INFO("[MULTI_RANK_ROPE_TEST] ✓ PASS: Multi-rank K gathering after RoPE is correct");
            }
            else
            {
                LOG_ERROR("[MULTI_RANK_ROPE_TEST] ✗ FAIL: Distributed RoPE+gather differs from sequential");
            }
        }
    }

} // anonymous namespace

int main(int argc, char **argv)
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();

    MPI_Finalize();
    return result;
}
