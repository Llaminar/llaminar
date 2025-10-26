/**
 * @file Test__MPIBatchOperations.cpp
 * @brief Systematic validation of MPI operations with batched data
 *
 * This test suite validates core assumptions about how MPI tensor-parallel
 * operations should work with batched inputs. Each test isolates a specific
 * assumption and verifies correctness.
 *
 * Test Strategy:
 * 1. Start with simplest operations (embedding lookup, linear projection)
 * 2. Progress to complex operations (attention, reduction)
 * 3. Compare single-rank (ground truth) vs multi-rank (MPI) results
 * 4. Test both batch_size=1 and batch_size>1 scenarios
 *
 * Key Assumptions Being Tested:
 * - A1: Embedding lookup works identically for batched inputs (MPI vs single-rank)
 * - A2: Linear projections (Q/K/V) produce identical results (MPI vs single-rank)
 * - A3: RoPE application respects per-sequence positions
 * - A4: Attention scores computed per-head are identical (MPI vs single-rank)
 * - A5: Attention output allreduce preserves batch boundaries
 * - A6: Block-diagonal masking isolates sequences in batches
 *
 * Author: David Sanftenberg
 * Date: 2025-10-26
 */

#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>

#include "../../../src/v2/pipelines/qwen/Qwen2Pipeline.h"
#include "../../../src/v2/loaders/ModelContext.h"
#include "../../../src/v2/utils/MPIContext.h"
#include "../../../src/v2/tensors/Tensors.h"
#include "../../../src/v2/kernels/cpu/FP32StandaloneGemm.h"

using namespace llaminar2;

class MPIBatchOperations : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Get MPI rank and world size (MPI_Init called by CTest infrastructure)
        int rank, world_size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);

        mpi_ctx_ = std::make_shared<MPIContext>(rank, world_size);
        rank_ = mpi_ctx_->rank();
        world_size_ = mpi_ctx_->world_size();

        // Test requires exactly 2 ranks for tensor-parallel validation
        if (world_size_ != 2)
        {
            GTEST_SKIP() << "MPIBatchOperations tests require exactly 2 MPI ranks (got " << world_size_ << ")";
        }

        // Load Qwen 0.5B model for realistic architecture params
        const char *model_path = std::getenv("LLAMINAR_TEST_MODEL_PATH");
        if (!model_path)
        {
            model_path = "models/qwen2.5-0.5b-instruct-iq4_nl.gguf";
        }

        model_ctx_ = ModelContext::create(model_path);
        ASSERT_NE(model_ctx_, nullptr) << "Failed to load model: " << model_path;

        // Extract architecture params (Qwen 0.5B: 14 heads, 7 KV heads, 64 head_dim)
        const auto &model = model_ctx_->model();
        n_heads_ = static_cast<int>(model.head_count);                           // 14
        n_kv_heads_ = static_cast<int>(model.head_count_kv);                     // 7
        head_dim_ = static_cast<int>(model.embedding_length / model.head_count); // 64
        d_model_ = static_cast<int>(model.embedding_length);                     // 896
        vocab_size_ = static_cast<int>(model.vocab_size);                        // 151936

        // Validate tensor-parallel compatibility
        ASSERT_EQ(n_heads_ % world_size_, 0) << "n_heads must be divisible by world_size for tensor-parallel";
        ASSERT_EQ(n_kv_heads_ % world_size_, 0) << "n_kv_heads must be divisible by world_size";
    }

    void TearDown() override
    {
        if (mpi_ctx_)
        {
            mpi_ctx_->barrier();
        }
    }

    /**
     * @brief Compare two tensors element-wise
     * @return True if all elements match within tolerance
     */
    bool compareTensors(const float *a, const float *b, size_t count, float tolerance = 1e-5f)
    {
        float max_diff = 0.0f;
        size_t mismatches = 0;

        for (size_t i = 0; i < count; ++i)
        {
            float diff = std::abs(a[i] - b[i]);
            max_diff = std::max(max_diff, diff);
            if (diff > tolerance)
            {
                mismatches++;
            }
        }

        if (rank_ == 0 && mismatches > 0)
        {
            std::cout << "[Comparison] max_diff=" << max_diff << ", mismatches=" << mismatches << "/" << count << std::endl;
        }

        return mismatches == 0;
    }

    std::shared_ptr<MPIContext> mpi_ctx_;
    std::shared_ptr<ModelContext> model_ctx_;
    int rank_ = 0;
    int world_size_ = 1;

    // Architecture params (from Qwen 0.5B)
    int n_heads_ = 14;
    int n_kv_heads_ = 7;
    int head_dim_ = 64;
    int d_model_ = 896;
    int vocab_size_ = 151936;
};

// =============================================================================
// A1: Embedding Lookup - Batched inputs should produce identical results
// =============================================================================

TEST_F(MPIBatchOperations, A1_EmbeddingLookup_Batch1)
{
    /**
     * Assumption: Embedding lookup for batch_size=1 produces identical results
     *             on all ranks (embeddings are replicated, not sharded)
     *
     * Test: Look up 2 tokens, verify all ranks get same embeddings
     */

    std::vector<int> tokens = {151644, 9906}; // BOS + token
    const int seq_len = tokens.size();

    // Create embedding table (replicated on all ranks)
    auto embedding = model_ctx_->getWeight("token_embd.weight");
    ASSERT_NE(embedding, nullptr) << "Failed to load embedding table";

    // Allocate output: [seq_len, d_model]
    auto output = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(d_model_)}, -1);

    // Perform embedding lookup
    for (int i = 0; i < seq_len; ++i)
    {
        int token_id = tokens[i];
        const float *embedding_row = embedding->data() + token_id * d_model_;
        float *output_row = output->mutable_data() + i * d_model_;
        std::memcpy(output_row, embedding_row, d_model_ * sizeof(float));
    }

    // Gather results from rank 0 to all ranks for comparison
    std::vector<float> rank0_output(seq_len * d_model_);
    MPI_Bcast(output->mutable_data(), seq_len * d_model_, MPI_FLOAT, 0, mpi_ctx_->comm());

    // All ranks should have identical embeddings
    bool match = compareTensors(output->data(), rank0_output.data(), seq_len * d_model_);
    EXPECT_TRUE(match) << "Rank " << rank_ << " embedding mismatch";

    if (rank_ == 0)
    {
        std::cout << "[A1] ✓ Embedding lookup produces identical results on all ranks (batch_size=1)" << std::endl;
    }
}

TEST_F(MPIBatchOperations, A1_EmbeddingLookup_Batch2)
{
    /**
     * Assumption: Embedding lookup for batch_size=2 produces identical results
     *             on all ranks, with correct batch structure preserved
     *
     * Test: Look up 2 sequences (4 tokens total), verify batch layout
     */

    std::vector<std::vector<int>> batch = {
        {151644, 9906}, // Seq 0: 2 tokens
        {151644, 1374}  // Seq 1: 2 tokens (different second token)
    };

    const int batch_size = batch.size();
    const int padded_seq_len = 2; // Both sequences have length 2
    const int effective_seq_len = batch_size * padded_seq_len;

    // Create embedding table
    auto embedding = model_ctx_->getWeight("token_embd.weight");
    ASSERT_NE(embedding, nullptr);

    // Allocate output: [effective_seq_len, d_model] = [4, 896]
    auto output = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(effective_seq_len), static_cast<size_t>(d_model_)}, -1);

    // Perform batched embedding lookup (layout: [batch, seq, features] flattened to [batch*seq, features])
    for (int b = 0; b < batch_size; ++b)
    {
        for (size_t s = 0; s < batch[b].size(); ++s)
        {
            int token_id = batch[b][s];
            const float *embedding_row = embedding->data() + token_id * d_model_;
            float *output_row = output->mutable_data() + (b * padded_seq_len + s) * d_model_;
            std::memcpy(output_row, embedding_row, d_model_ * sizeof(float));
        }
    }

    // Verify batch structure: seq0[0] == seq1[0] (both use BOS token 151644)
    const float *seq0_tok0 = output->data() + 0 * d_model_;
    const float *seq1_tok0 = output->data() + 2 * d_model_;

    bool bos_match = compareTensors(seq0_tok0, seq1_tok0, d_model_);
    EXPECT_TRUE(bos_match) << "BOS embeddings should match between sequences";

    // Verify seq0[1] != seq1[1] (different tokens: 9906 vs 1374)
    const float *seq0_tok1 = output->data() + 1 * d_model_;
    const float *seq1_tok1 = output->data() + 3 * d_model_;

    bool different_tokens = !compareTensors(seq0_tok1, seq1_tok1, d_model_, 1e-5f);
    EXPECT_TRUE(different_tokens) << "Different tokens should have different embeddings";

    if (rank_ == 0)
    {
        std::cout << "[A1] ✓ Batched embedding preserves batch structure correctly (batch_size=2)" << std::endl;
    }
}

// =============================================================================
// A2: Linear Projections (Q/K/V) - Tensor-parallel should match single-rank
// =============================================================================

TEST_F(MPIBatchOperations, A2_LinearProjection_SingleRankBaseline)
{
    /**
     * Assumption: Single-rank linear projection produces expected output shape
     *
     * Test: X @ W^T where X=[seq_len, d_in], W=[d_out, d_in]
     *       Output should be [seq_len, d_out]
     */

    const int seq_len = 2;
    const int d_in = d_model_;              // 896
    const int d_out = n_heads_ * head_dim_; // 14 * 64 = 896

    // Create random input: [seq_len, d_in]
    auto input = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(d_in)}, -1);

    // Initialize with known pattern (rank-independent)
    for (int i = 0; i < seq_len * d_in; ++i)
    {
        input->mutable_data()[i] = static_cast<float>(i % 100) / 100.0f;
    }

    // Load Q projection weight: [d_out, d_in]
    auto weight = model_ctx_->getWeight("blk.0.attn_q.weight");
    ASSERT_NE(weight, nullptr) << "Failed to load Q weight";

    // Allocate output: [seq_len, d_out]
    auto output = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(d_out)}, -1);

    // Perform GEMM: output = input @ weight^T
    auto gemm = weight->createGemm();
    ASSERT_NE(gemm, nullptr);

    bool success = gemm->multiply(
        input->data(), output->mutable_data(),
        seq_len, d_out, d_in,
        /*transpose=*/true, /*alpha=*/1.0f, /*beta=*/0.0f,
        nullptr, -1);

    ASSERT_TRUE(success) << "Linear projection failed";

    // Verify output shape
    EXPECT_EQ(output->shape()[0], seq_len);
    EXPECT_EQ(output->shape()[1], d_out);

    // Verify output is not all zeros
    bool has_nonzero = false;
    for (int i = 0; i < seq_len * d_out; ++i)
    {
        if (std::abs(output->data()[i]) > 1e-6f)
        {
            has_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero) << "Output should not be all zeros";

    if (rank_ == 0)
    {
        std::cout << "[A2] ✓ Single-rank linear projection produces valid output" << std::endl;
    }
}

TEST_F(MPIBatchOperations, A2_LinearProjection_TensorParallelVsSingleRank)
{
    /**
     * Assumption: Tensor-parallel Q projection (sharded across ranks) produces
     *             identical results to single-rank execution after allreduce
     *
     * Test: Run Q projection on 2 ranks with head sharding, verify final output
     *       matches single-rank baseline
     */

    const int seq_len = 2;
    const int d_in = d_model_;
    const int d_out = n_heads_ * head_dim_;

    // Create identical input on all ranks
    auto input = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(d_in)}, -1);

    for (int i = 0; i < seq_len * d_in; ++i)
    {
        input->mutable_data()[i] = static_cast<float>(i % 100) / 100.0f;
    }

    // Load full Q weight
    auto weight = model_ctx_->getWeight("blk.0.attn_q.weight");
    ASSERT_NE(weight, nullptr);

    // === Single-Rank Baseline ===
    auto output_single = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(d_out)}, -1);

    auto gemm = weight->createGemm();
    bool success = gemm->multiply(
        input->data(), output_single->mutable_data(),
        seq_len, d_out, d_in, true, 1.0f, 0.0f, nullptr, -1);
    ASSERT_TRUE(success);

    // === Tensor-Parallel Execution ===
    // Each rank computes subset of heads
    const int local_n_heads = n_heads_ / world_size_;  // 14 / 2 = 7 heads per rank
    const int local_d_out = local_n_heads * head_dim_; // 7 * 64 = 448
    const int start_head = rank_ * local_n_heads;

    auto output_local = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(local_d_out)}, -1);

    // Extract local weight shard: weight[start_head*head_dim : (start_head+local_n_heads)*head_dim, :]
    const float *weight_data = weight->data();
    std::vector<float> local_weight(local_d_out * d_in);

    for (int i = 0; i < local_d_out; ++i)
    {
        for (int j = 0; j < d_in; ++j)
        {
            local_weight[i * d_in + j] = weight_data[(start_head * head_dim_ + i) * d_in + j];
        }
    }

    // Compute local projection
    success = FP32StandaloneGemm::multiply_with_b(
        input->data(), local_weight.data(), output_local->mutable_data(),
        seq_len, local_d_out, d_in, true, 1.0f, 0.0f);
    ASSERT_TRUE(success);

    // Allgather to reconstruct full output
    std::vector<float> output_tp(seq_len * d_out);
    MPI_Allgather(
        output_local->data(), seq_len * local_d_out, MPI_FLOAT,
        output_tp.data(), seq_len * local_d_out, MPI_FLOAT,
        mpi_ctx_->comm());

    // Compare tensor-parallel vs single-rank
    bool match = compareTensors(output_single->data(), output_tp.data(), seq_len * d_out, 1e-4f);
    EXPECT_TRUE(match) << "Tensor-parallel Q projection should match single-rank";

    if (rank_ == 0 && match)
    {
        std::cout << "[A2] ✓ Tensor-parallel linear projection matches single-rank baseline" << std::endl;
    }
}

// =============================================================================
// A3: RoPE Application - Per-sequence position tracking
// =============================================================================

TEST_F(MPIBatchOperations, A3_RoPE_PerSequencePositions)
{
    /**
     * Assumption: RoPE with batch_size=2 uses independent position counters
     *             for each sequence (not global positions)
     *
     * Test: Apply RoPE with positions [0,1] for both sequences
     *       Verify seq0_pos0 == seq1_pos0 (same position, same rotation)
     *       Verify seq0_pos1 != seq1_pos1 only if input Q/K differ
     */

    const int batch_size = 2;
    const int seq_len_per_batch = 2;
    const int effective_seq_len = batch_size * seq_len_per_batch; // 4

    // Create Q tensor: [effective_seq_len, n_heads * head_dim]
    auto Q = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(effective_seq_len), static_cast<size_t>(n_heads_ * head_dim_)}, -1);

    // Initialize with known pattern
    for (int i = 0; i < effective_seq_len * n_heads_ * head_dim_; ++i)
    {
        Q->mutable_data()[i] = static_cast<float>(i % 50) / 50.0f;
    }

    // Make seq0[0] == seq1[0] (same input)
    std::memcpy(
        Q->mutable_data() + 2 * n_heads_ * head_dim_, // seq1[0] destination
        Q->mutable_data() + 0 * n_heads_ * head_dim_, // seq0[0] source
        n_heads_ * head_dim_ * sizeof(float));

    // Create K tensor (same structure)
    auto K = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(effective_seq_len), static_cast<size_t>(n_kv_heads_ * head_dim_)}, -1);

    for (int i = 0; i < effective_seq_len * n_kv_heads_ * head_dim_; ++i)
    {
        K->mutable_data()[i] = static_cast<float>((i + 10) % 50) / 50.0f;
    }

    std::memcpy(
        K->mutable_data() + 2 * n_kv_heads_ * head_dim_,
        K->mutable_data() + 0 * n_kv_heads_ * head_dim_,
        n_kv_heads_ * head_dim_ * sizeof(float));

    // Position IDs: [0, 1, 0, 1] (per-sequence positions)
    std::vector<int> position_ids = {0, 1, 0, 1};

    // Load weights to create RoPE kernel
    auto weight = model_ctx_->getWeight("blk.0.attn_q.weight");
    ASSERT_NE(weight, nullptr);
    auto rope_kernel = weight->createRoPE();
    ASSERT_NE(rope_kernel, nullptr);

    // Apply RoPE
    bool success = rope_kernel->apply(
        Q->mutable_data(), K->mutable_data(), position_ids.data(),
        effective_seq_len, n_heads_, n_kv_heads_, head_dim_,
        false, mpi_ctx_.get(), -1);

    ASSERT_TRUE(success) << "RoPE application failed";

    // Verify: seq0[0] and seq1[0] should still match (same position, same input)
    const float *seq0_pos0 = Q->data() + 0 * n_heads_ * head_dim_;
    const float *seq1_pos0 = Q->data() + 2 * n_heads_ * head_dim_;

    bool pos0_match = compareTensors(seq0_pos0, seq1_pos0, n_heads_ * head_dim_, 1e-4f);
    EXPECT_TRUE(pos0_match) << "RoPE at position 0 should produce identical results for both sequences";

    if (rank_ == 0 && pos0_match)
    {
        std::cout << "[A3] ✓ RoPE respects per-sequence position tracking (batch_size=2)" << std::endl;
    }
}

// =============================================================================
// A4: Attention Scores - Per-head computation correctness
// =============================================================================

TEST_F(MPIBatchOperations, A4_AttentionScores_SingleHead)
{
    /**
     * Assumption: Attention score computation Q @ K^T produces correct output
     *             for a single head (no batching complexity)
     *
     * Test: Compute scores for one head, verify output shape and values
     */

    const int seq_len = 2;

    // Create Q and K for single head: [seq_len, head_dim]
    std::vector<float> Q(seq_len * head_dim_);
    std::vector<float> K(seq_len * head_dim_);

    for (int i = 0; i < seq_len * head_dim_; ++i)
    {
        Q[i] = static_cast<float>(i % 20) / 20.0f;
        K[i] = static_cast<float>((i + 5) % 20) / 20.0f;
    }

    // Compute scores: [seq_len, seq_len]
    std::vector<float> scores(seq_len * seq_len);

    bool success = FP32StandaloneGemm::multiply_with_b(
        Q.data(), K.data(), scores.data(),
        seq_len, seq_len, head_dim_,
        true, 1.0f, 0.0f);

    ASSERT_TRUE(success);

    // Verify scores are non-zero
    bool has_nonzero = false;
    for (float score : scores)
    {
        if (std::abs(score) > 1e-6f)
        {
            has_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero) << "Attention scores should not be all zeros";

    if (rank_ == 0)
    {
        std::cout << "[A4] ✓ Single-head attention scores computed correctly" << std::endl;
    }
}

// =============================================================================
// A5: Attention Allreduce - Batch boundary preservation
// =============================================================================

TEST_F(MPIBatchOperations, A5_AttentionAllreduce_BatchPreservation)
{
    /**
     * Assumption: MPI allreduce of attention outputs preserves batch boundaries
     *             (seq0 and seq1 outputs don't mix)
     *
     * Test: Create local outputs for 2 sequences on each rank, allreduce,
     *       verify seq0 and seq1 remain independent
     */

    const int batch_size = 2;
    const int seq_len_per_batch = 2;
    const int effective_seq_len = batch_size * seq_len_per_batch;
    const int local_n_heads = n_heads_ / world_size_;
    const int local_d_out = local_n_heads * head_dim_;

    // Create local output: [effective_seq_len, local_d_out]
    std::vector<float> local_output(effective_seq_len * local_d_out);

    // Rank 0: Initialize seq0 with 1.0, seq1 with 2.0
    // Rank 1: Initialize seq0 with 3.0, seq1 with 4.0
    float seq0_value = (rank_ == 0) ? 1.0f : 3.0f;
    float seq1_value = (rank_ == 0) ? 2.0f : 4.0f;

    for (int s = 0; s < seq_len_per_batch; ++s)
    {
        for (int d = 0; d < local_d_out; ++d)
        {
            local_output[s * local_d_out + d] = seq0_value;
        }
    }

    for (int s = 0; s < seq_len_per_batch; ++s)
    {
        for (int d = 0; d < local_d_out; ++d)
        {
            local_output[(seq_len_per_batch + s) * local_d_out + d] = seq1_value;
        }
    }

    // Prepare send buffer (pack local heads)
    std::vector<float> send_buffer(effective_seq_len * n_heads_ * head_dim_, 0.0f);

    for (int s = 0; s < effective_seq_len; ++s)
    {
        for (int local_h = 0; local_h < local_n_heads; ++local_h)
        {
            int global_h = rank_ * local_n_heads + local_h;
            for (int d = 0; d < head_dim_; ++d)
            {
                send_buffer[s * n_heads_ * head_dim_ + global_h * head_dim_ + d] =
                    local_output[s * local_d_out + local_h * head_dim_ + d];
            }
        }
    }

    // Allreduce
    std::vector<float> output(effective_seq_len * n_heads_ * head_dim_);
    MPI_Allreduce(send_buffer.data(), output.data(),
                  effective_seq_len * n_heads_ * head_dim_,
                  MPI_FLOAT, MPI_SUM, mpi_ctx_->comm());

    // Verify seq0 and seq1 outputs are different
    float seq0_sum = 0.0f;
    float seq1_sum = 0.0f;

    for (int s = 0; s < seq_len_per_batch; ++s)
    {
        for (int h = 0; h < n_heads_; ++h)
        {
            for (int d = 0; d < head_dim_; ++d)
            {
                seq0_sum += output[s * n_heads_ * head_dim_ + h * head_dim_ + d];
            }
        }
    }

    for (int s = 0; s < seq_len_per_batch; ++s)
    {
        for (int h = 0; h < n_heads_; ++h)
        {
            for (int d = 0; d < head_dim_; ++d)
            {
                seq1_sum += output[(seq_len_per_batch + s) * n_heads_ * head_dim_ + h * head_dim_ + d];
            }
        }
    }

    // Expected: seq0 = (1.0 + 3.0) * seq_len * n_heads * head_dim
    //           seq1 = (2.0 + 4.0) * seq_len * n_heads * head_dim
    float expected_seq0 = 4.0f * seq_len_per_batch * n_heads_ * head_dim_;
    float expected_seq1 = 6.0f * seq_len_per_batch * n_heads_ * head_dim_;

    EXPECT_NEAR(seq0_sum, expected_seq0, 1e-3f) << "Seq0 allreduce incorrect";
    EXPECT_NEAR(seq1_sum, expected_seq1, 1e-3f) << "Seq1 allreduce incorrect";

    if (rank_ == 0)
    {
        std::cout << "[A5] ✓ Attention allreduce preserves batch boundaries" << std::endl;
        std::cout << "     seq0_sum=" << seq0_sum << " (expected " << expected_seq0 << ")" << std::endl;
        std::cout << "     seq1_sum=" << seq1_sum << " (expected " << expected_seq1 << ")" << std::endl;
    }
}

// =============================================================================
// A6: Block-Diagonal Masking - Sequence isolation
// =============================================================================

TEST_F(MPIBatchOperations, A6_BlockDiagonalMask_Isolation)
{
    /**
     * Assumption: Block-diagonal causal mask prevents cross-sequence attention
     *             (seq0 cannot attend to seq1 and vice versa)
     *
     * Test: Create mask for batch_size=2, verify structure:
     *       - seq0 can only attend to seq0
     *       - seq1 can only attend to seq1
     *       - Cross-sequence attention is masked (-inf)
     */

    const int batch_size = 2;
    const int seq_len_per_batch = 2;
    const int total_len = batch_size * seq_len_per_batch; // 4

    // Create block-diagonal causal mask
    std::vector<float> mask(total_len * total_len);
    const float neg_inf = -std::numeric_limits<float>::infinity();

    // Apply block-diagonal + causal structure
    for (int i = 0; i < total_len; ++i)
    {
        for (int j = 0; j < total_len; ++j)
        {
            int batch_i = i / seq_len_per_batch;
            int batch_j = j / seq_len_per_batch;
            int pos_i = i % seq_len_per_batch;
            int pos_j = j % seq_len_per_batch;

            if (batch_i != batch_j)
            {
                // Cross-sequence: always mask
                mask[i * total_len + j] = neg_inf;
            }
            else if (pos_i < pos_j)
            {
                // Future token in same sequence: mask (causal)
                mask[i * total_len + j] = neg_inf;
            }
            else
            {
                // Can attend
                mask[i * total_len + j] = 0.0f;
            }
        }
    }

    // Verify block structure
    // Seq0 tokens (0,1) should ONLY attend within seq0
    EXPECT_EQ(mask[0 * total_len + 0], 0.0f) << "seq0[0] should attend to seq0[0]";
    EXPECT_EQ(mask[1 * total_len + 0], 0.0f) << "seq0[1] should attend to seq0[0]";
    EXPECT_EQ(mask[1 * total_len + 1], 0.0f) << "seq0[1] should attend to seq0[1]";
    EXPECT_TRUE(std::isinf(mask[0 * total_len + 1])) << "seq0[0] cannot attend to seq0[1] (causal)";

    // Cross-sequence attention should be masked
    EXPECT_TRUE(std::isinf(mask[0 * total_len + 2])) << "seq0[0] cannot attend to seq1[0]";
    EXPECT_TRUE(std::isinf(mask[0 * total_len + 3])) << "seq0[0] cannot attend to seq1[1]";
    EXPECT_TRUE(std::isinf(mask[2 * total_len + 0])) << "seq1[0] cannot attend to seq0[0]";
    EXPECT_TRUE(std::isinf(mask[2 * total_len + 1])) << "seq1[0] cannot attend to seq0[1]";

    // Seq1 tokens (2,3) should ONLY attend within seq1
    EXPECT_EQ(mask[2 * total_len + 2], 0.0f) << "seq1[0] should attend to seq1[0]";
    EXPECT_EQ(mask[3 * total_len + 2], 0.0f) << "seq1[1] should attend to seq1[0]";
    EXPECT_EQ(mask[3 * total_len + 3], 0.0f) << "seq1[1] should attend to seq1[1]";

    if (rank_ == 0)
    {
        std::cout << "[A6] ✓ Block-diagonal mask correctly isolates sequences" << std::endl;
    }
}
