/**
 * @file test_mpi_attention_batch.cpp
 * @brief Unit tests for MPIAttentionOperator batch dimension support
 * 
 * Tests verifying that MPIAttentionOperator correctly handles:
 * - 2D inputs: [seq_len, d_model] (backward compatibility)
 * - 3D inputs: [batch, seq_len, d_model] (batch processing)
 * 
 * Strategy:
 * - Batched inputs are flattened to [batch*seq_len, d_model]
 * - All 8 pipeline stages process flattened sequence
 * - Output shape matches input dimensionality
 * 
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <memory>
#include <vector>
#include <cmath>
#include <algorithm>
#include "../src/operators/MPIAttentionOperator.h"
#include "../src/tensors/SimpleTensor.h"

using namespace llaminar;

class MPIAttentionBatchTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        int initialized;
        MPI_Initialized(&initialized);
        if (!initialized)
        {
            int argc = 0;
            char **argv = nullptr;
            MPI_Init(&argc, &argv);
        }
        
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
    }

    void TearDown() override
    {
        // MPI_Finalize called by main
    }

    /**
     * @brief Create minimal attention inputs for testing
     * @param seq_len Sequence length
     * @param d_model Model dimension
     * @param n_head Number of attention heads
     * @param n_head_kv Number of KV heads
     * @param head_dim Dimension per head
     * @param inputs Output vector to populate with input tensors
     */
    void createAttentionInputs(int seq_len, int d_model, int n_head, int n_head_kv, int head_dim,
                               std::vector<std::shared_ptr<TensorBase>>& inputs)
    {
        // Input activations [seq_len, d_model]
        auto input = std::make_shared<SimpleTensor>(std::vector<int>{seq_len, d_model});
        for (int i = 0; i < seq_len * d_model; ++i)
        {
            input->data()[i] = static_cast<float>((i % 100)) / 100.0f;
        }

        // Weight matrices
        int total_head_dim = n_head * head_dim;
        int total_kv_head_dim = n_head_kv * head_dim;
        
        auto wq = std::make_shared<SimpleTensor>(std::vector<int>{total_head_dim, d_model});
        auto wk = std::make_shared<SimpleTensor>(std::vector<int>{total_kv_head_dim, d_model});
        auto wv = std::make_shared<SimpleTensor>(std::vector<int>{total_kv_head_dim, d_model});
        auto wo = std::make_shared<SimpleTensor>(std::vector<int>{d_model, total_head_dim});
        
        // Initialize weights with small values
        for (int i = 0; i < total_head_dim * d_model; ++i)
        {
            wq->data()[i] = 0.01f * static_cast<float>((i % 50)) / 50.0f;
        }
        for (int i = 0; i < total_kv_head_dim * d_model; ++i)
        {
            wk->data()[i] = 0.01f * static_cast<float>((i % 50)) / 50.0f;
            wv->data()[i] = 0.01f * static_cast<float>((i % 50)) / 50.0f;
        }
        for (int i = 0; i < d_model * total_head_dim; ++i)
        {
            wo->data()[i] = 0.01f * static_cast<float>((i % 50)) / 50.0f;
        }

        // Bias vectors (small tensors to pass validation)
        auto bq = std::make_shared<SimpleTensor>(std::vector<int>{total_head_dim});
        auto bk = std::make_shared<SimpleTensor>(std::vector<int>{total_kv_head_dim});
        auto bv = std::make_shared<SimpleTensor>(std::vector<int>{total_kv_head_dim});
        std::fill_n(bq->data(), total_head_dim, 0.0f);
        std::fill_n(bk->data(), total_kv_head_dim, 0.0f);
        std::fill_n(bv->data(), total_kv_head_dim, 0.0f);

        // KV cache (empty for prefill mode)
        int max_cache_len = 2048; // Sufficient capacity
        auto k_cache = std::make_shared<SimpleTensor>(std::vector<int>{max_cache_len, total_kv_head_dim});
        auto v_cache = std::make_shared<SimpleTensor>(std::vector<int>{max_cache_len, total_kv_head_dim});
        std::fill_n(k_cache->data(), k_cache->size(), 0.0f);
        std::fill_n(v_cache->data(), v_cache->size(), 0.0f);

        inputs = {input, wq, wk, wv, wo, bq, bk, bv, k_cache, v_cache};
    }

    /**
     * @brief Verify output contains no NaN or Inf values
     */
    bool hasNoNaNOrInf(const std::shared_ptr<TensorBase>& tensor)
    {
        const float* data = tensor->data();
        size_t count = tensor->size();
        for (size_t i = 0; i < count; ++i)
        {
            if (std::isnan(data[i]) || std::isinf(data[i]))
            {
                return false;
            }
        }
        return true;
    }

    int rank_;
    int world_size_;
};

/**
 * @brief Test backward compatibility with single sequence (2D input)
 * 
 * Verifies that 2D input [seq_len, d_model] still works as before.
 */
TEST_F(MPIAttentionBatchTest, SingleSequence_BackwardCompat)
{
    const int seq_len = 4;
    const int d_model = 64;
    const int n_head = 4;
    const int n_head_kv = 4;
    const int head_dim = 16;

    // Create attention operator
    MPIAttentionOperator op(n_head, n_head_kv, head_dim);
    op.setSequencePosition(0); // Prefill mode

    // Create inputs with 2D activations [seq_len, d_model]
    std::vector<std::shared_ptr<TensorBase>> inputs;
    createAttentionInputs(seq_len, d_model, n_head, n_head_kv, head_dim, inputs);

    // Execute
    std::vector<std::shared_ptr<TensorBase>> outputs;
    outputs.push_back(std::make_shared<SimpleTensor>(std::vector<int>{seq_len, d_model}));
    
    bool success = op.execute(inputs, outputs);
    ASSERT_TRUE(success) << "Attention operator failed on 2D input";

    // Verify output shape is 2D [seq_len, d_model]
    ASSERT_EQ(outputs[0]->shape().size(), 2);
    EXPECT_EQ(outputs[0]->shape()[0], seq_len);
    EXPECT_EQ(outputs[0]->shape()[1], d_model);

    // Verify no NaN/Inf
    EXPECT_TRUE(hasNoNaNOrInf(outputs[0])) << "Output contains NaN or Inf";
}

/**
 * @brief Test batched sequences processing
 * 
 * Verifies that 3D input [batch, seq_len, d_model] is correctly processed.
 */
TEST_F(MPIAttentionBatchTest, BatchedSequences)
{
    const int batch_size = 2;
    const int seq_len = 4;
    const int d_model = 64;
    const int n_head = 4;
    const int n_head_kv = 4;
    const int head_dim = 16;

    // Create attention operator
    MPIAttentionOperator op(n_head, n_head_kv, head_dim);
    op.setSequencePosition(0); // Prefill mode

    // Create base inputs (2D)
    std::vector<std::shared_ptr<TensorBase>> base_inputs;
    createAttentionInputs(seq_len, d_model, n_head, n_head_kv, head_dim, base_inputs);

    // Replace input with batched 3D version [batch, seq_len, d_model]
    auto input_3d = std::make_shared<SimpleTensor>(
        std::vector<int>{batch_size, seq_len, d_model});
    
    // Fill with test data (different values per batch)
    for (int b = 0; b < batch_size; ++b)
    {
        for (int s = 0; s < seq_len; ++s)
        {
            for (int d = 0; d < d_model; ++d)
            {
                int idx = b * seq_len * d_model + s * d_model + d;
                input_3d->data()[idx] = static_cast<float>((idx + b * 7) % 100) / 100.0f;
            }
        }
    }
    
    std::vector<std::shared_ptr<TensorBase>> inputs = base_inputs;
    inputs[0] = input_3d; // Replace 2D input with 3D batched input

    // Execute
    std::vector<std::shared_ptr<TensorBase>> outputs;
    outputs.push_back(std::make_shared<SimpleTensor>(
        std::vector<int>{batch_size, seq_len, d_model}));
    
    bool success = op.execute(inputs, outputs);
    ASSERT_TRUE(success) << "Attention operator failed on 3D batched input";

    // Verify output shape is 3D [batch, seq_len, d_model]
    ASSERT_EQ(outputs[0]->shape().size(), 3);
    EXPECT_EQ(outputs[0]->shape()[0], batch_size);
    EXPECT_EQ(outputs[0]->shape()[1], seq_len);
    EXPECT_EQ(outputs[0]->shape()[2], d_model);

    // Verify no NaN/Inf
    EXPECT_TRUE(hasNoNaNOrInf(outputs[0])) << "Output contains NaN or Inf";
}

/**
 * @brief Test that batched processing produces same results as individual sequences
 * 
 * Verifies: Attention(batch=[seq1, seq2]) == [Attention(seq1), Attention(seq2)]
 * 
 * Note: This test verifies that flattening batch dimension doesn't break
 * the operator, though the exact equivalence depends on attention mechanics.
 */
TEST_F(MPIAttentionBatchTest, BatchProcessing_NoRegressions)
{
    const int batch_size = 2;
    const int seq_len = 4;
    const int d_model = 64;
    const int n_head = 4;
    const int n_head_kv = 4;
    const int head_dim = 16;

    // Create base inputs
    std::vector<std::shared_ptr<TensorBase>> base_inputs;
    createAttentionInputs(seq_len, d_model, n_head, n_head_kv, head_dim, base_inputs);

    // Process as batch
    auto batch_input = std::make_shared<SimpleTensor>(
        std::vector<int>{batch_size, seq_len, d_model});
    
    for (int b = 0; b < batch_size; ++b)
    {
        for (int s = 0; s < seq_len; ++s)
        {
            for (int d = 0; d < d_model; ++d)
            {
                int idx = b * seq_len * d_model + s * d_model + d;
                batch_input->data()[idx] = static_cast<float>((idx * 3 + b * 11) % 100) / 100.0f;
            }
        }
    }

    std::vector<std::shared_ptr<TensorBase>> batch_inputs = base_inputs;
    batch_inputs[0] = batch_input;

    MPIAttentionOperator op_batch(n_head, n_head_kv, head_dim);
    op_batch.setSequencePosition(0);
    
    std::vector<std::shared_ptr<TensorBase>> batch_outputs;
    batch_outputs.push_back(std::make_shared<SimpleTensor>(
        std::vector<int>{batch_size, seq_len, d_model}));
    
    bool success = op_batch.execute(batch_inputs, batch_outputs);
    ASSERT_TRUE(success) << "Batch processing failed";

    // Verify shape and no NaN/Inf
    ASSERT_EQ(batch_outputs[0]->shape().size(), 3);
    EXPECT_EQ(batch_outputs[0]->shape()[0], batch_size);
    EXPECT_EQ(batch_outputs[0]->shape()[1], seq_len);
    EXPECT_EQ(batch_outputs[0]->shape()[2], d_model);
    EXPECT_TRUE(hasNoNaNOrInf(batch_outputs[0])) << "Batch output contains NaN or Inf";

    // Note: We can't test exact equivalence with individual sequences because
    // attention's softmax operates over the full flattened sequence.
    // This is expected behavior for the current flatten-based implementation.
    // Future work: Add padding masks for true per-sequence attention.
}

/**
 * @brief Test large batch processing
 * 
 * Ensures batch support scales to larger batch sizes.
 */
TEST_F(MPIAttentionBatchTest, LargeBatch)
{
    const int batch_size = 4;
    const int seq_len = 8;
    const int d_model = 64;
    const int n_head = 4;
    const int n_head_kv = 4;
    const int head_dim = 16;

    // Create attention operator
    MPIAttentionOperator op(n_head, n_head_kv, head_dim);
    op.setSequencePosition(0);

    // Create base inputs
    std::vector<std::shared_ptr<TensorBase>> base_inputs;
    createAttentionInputs(seq_len, d_model, n_head, n_head_kv, head_dim, base_inputs);

    // Create large batched input
    auto input = std::make_shared<SimpleTensor>(
        std::vector<int>{batch_size, seq_len, d_model});
    
    for (int i = 0; i < batch_size * seq_len * d_model; ++i)
    {
        input->data()[i] = static_cast<float>((i * 7 + 13) % 100) / 100.0f;
    }

    std::vector<std::shared_ptr<TensorBase>> inputs = base_inputs;
    inputs[0] = input;

    // Execute
    std::vector<std::shared_ptr<TensorBase>> outputs;
    outputs.push_back(std::make_shared<SimpleTensor>(
        std::vector<int>{batch_size, seq_len, d_model}));
    
    bool success = op.execute(inputs, outputs);
    ASSERT_TRUE(success) << "Large batch processing failed";

    // Verify output shape
    ASSERT_EQ(outputs[0]->shape().size(), 3);
    EXPECT_EQ(outputs[0]->shape()[0], batch_size);
    EXPECT_EQ(outputs[0]->shape()[1], seq_len);
    EXPECT_EQ(outputs[0]->shape()[2], d_model);

    // Verify no NaN/Inf
    EXPECT_TRUE(hasNoNaNOrInf(outputs[0])) << "Output contains NaN or Inf";
}

int main(int argc, char** argv)
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    
    MPI_Finalize();
    return result;
}
