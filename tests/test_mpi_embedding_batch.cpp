/**
 * @file test_mpi_embedding_batch.cpp
 * @brief Test MPIEmbeddingOperator batch dimension support
 * @author David Sanftenberg
 */

#include "operators/MPIEmbeddingOperator.h"
#include "tensors/SimpleTensor.h"
#include "tensors/TensorFactory.h"
#include <gtest/gtest.h>
#include <mpi.h>
#include <memory>
#include <vector>

using namespace llaminar;

class MPIEmbeddingBatchTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
    }

    int rank;
    int size;
};

TEST_F(MPIEmbeddingBatchTest, SingleSequence_BackwardCompat)
{
    // Test 1D [seq_len] input (backward compatibility)
    const size_t vocab_size = 100;
    const size_t embed_dim = 32;
    const size_t seq_len = 4;

    MPIEmbeddingOperator op(vocab_size, embed_dim);

    // Create 1D token IDs [seq_len]
    auto token_ids = TensorFactory::create_simple({static_cast<int>(seq_len)});
    float* tid_data = token_ids->data();
    tid_data[0] = 10.0f;
    tid_data[1] = 20.0f;
    tid_data[2] = 30.0f;
    tid_data[3] = 40.0f;

    // Create full embedding table [vocab_size, embed_dim]
    auto embedding_table = TensorFactory::create_simple({static_cast<int>(vocab_size), static_cast<int>(embed_dim)});
    float* emb_data = embedding_table->data();
    for (size_t i = 0; i < vocab_size * embed_dim; ++i)
    {
        emb_data[i] = static_cast<float>(i) * 0.01f;
    }

    // Create output [seq_len, embed_dim]
    auto output = TensorFactory::create_simple({static_cast<int>(seq_len), static_cast<int>(embed_dim)});

    std::vector<std::shared_ptr<TensorBase>> inputs = {token_ids, embedding_table};
    std::vector<std::shared_ptr<TensorBase>> outputs = {output};

    // Execute
    bool success = op.execute(inputs, outputs);
    ASSERT_TRUE(success);

    // Verify output shape
    EXPECT_EQ(output->shape().size(), 2);
    EXPECT_EQ(output->shape()[0], seq_len);
    EXPECT_EQ(output->shape()[1], embed_dim);

    // Verify embedding lookup correctness for token 10
    const float* out_data = output->data();
    for (size_t d = 0; d < embed_dim; ++d)
    {
        float expected = emb_data[10 * embed_dim + d];
        EXPECT_FLOAT_EQ(out_data[0 * embed_dim + d], expected) << "Token 0, dim " << d;
    }
}

TEST_F(MPIEmbeddingBatchTest, BatchedSequences)
{
    // Test 2D [batch, seq_len] input
    const size_t vocab_size = 100;
    const size_t embed_dim = 16;
    const size_t batch_size = 3;
    const size_t seq_len = 4;

    MPIEmbeddingOperator op(vocab_size, embed_dim);

    // Create 2D token IDs [batch, seq_len]
    auto token_ids = TensorFactory::create_simple({static_cast<int>(batch_size), static_cast<int>(seq_len)});
    float* tid_data = token_ids->data();
    
    // Batch 0: [10, 11, 12, 13]
    tid_data[0] = 10.0f; tid_data[1] = 11.0f; tid_data[2] = 12.0f; tid_data[3] = 13.0f;
    // Batch 1: [20, 21, 22, 23]
    tid_data[4] = 20.0f; tid_data[5] = 21.0f; tid_data[6] = 22.0f; tid_data[7] = 23.0f;
    // Batch 2: [30, 31, 32, 33]
    tid_data[8] = 30.0f; tid_data[9] = 31.0f; tid_data[10] = 32.0f; tid_data[11] = 33.0f;

    // Create full embedding table [vocab_size, embed_dim]
    auto embedding_table = TensorFactory::create_simple({static_cast<int>(vocab_size), static_cast<int>(embed_dim)});
    float* emb_data = embedding_table->data();
    for (size_t i = 0; i < vocab_size * embed_dim; ++i)
    {
        emb_data[i] = static_cast<float>(i) * 0.01f;
    }

    // Create output [batch, seq_len, embed_dim]
    auto output = TensorFactory::create_simple({static_cast<int>(batch_size), static_cast<int>(seq_len), static_cast<int>(embed_dim)});

    std::vector<std::shared_ptr<TensorBase>> inputs = {token_ids, embedding_table};
    std::vector<std::shared_ptr<TensorBase>> outputs = {output};

    // Execute
    bool success = op.execute(inputs, outputs);
    ASSERT_TRUE(success);

    // Verify output shape
    EXPECT_EQ(output->shape().size(), 3);
    EXPECT_EQ(output->shape()[0], batch_size);
    EXPECT_EQ(output->shape()[1], seq_len);
    EXPECT_EQ(output->shape()[2], embed_dim);

    // Verify embedding lookups for each batch
    const float* out_data = output->data();
    
    // Batch 0, Token 0 (ID=10)
    for (size_t d = 0; d < embed_dim; ++d)
    {
        float expected = emb_data[10 * embed_dim + d];
        size_t idx = 0 * seq_len * embed_dim + 0 * embed_dim + d;
        EXPECT_FLOAT_EQ(out_data[idx], expected) << "Batch 0, Token 0, dim " << d;
    }
    
    // Batch 1, Token 2 (ID=22)
    for (size_t d = 0; d < embed_dim; ++d)
    {
        float expected = emb_data[22 * embed_dim + d];
        size_t idx = 1 * seq_len * embed_dim + 2 * embed_dim + d;
        EXPECT_FLOAT_EQ(out_data[idx], expected) << "Batch 1, Token 2, dim " << d;
    }
    
    // Batch 2, Token 3 (ID=33)
    for (size_t d = 0; d < embed_dim; ++d)
    {
        float expected = emb_data[33 * embed_dim + d];
        size_t idx = 2 * seq_len * embed_dim + 3 * embed_dim + d;
        EXPECT_FLOAT_EQ(out_data[idx], expected) << "Batch 2, Token 3, dim " << d;
    }
}

TEST_F(MPIEmbeddingBatchTest, LargeBatch)
{
    // Test with larger batch to verify scalability
    const size_t vocab_size = 1000;
    const size_t embed_dim = 64;
    const size_t batch_size = 8;
    const size_t seq_len = 16;

    MPIEmbeddingOperator op(vocab_size, embed_dim);

    // Create 2D token IDs [batch, seq_len]
    auto token_ids = TensorFactory::create_simple({static_cast<int>(batch_size), static_cast<int>(seq_len)});
    float* tid_data = token_ids->data();
    
    // Fill with sequential token IDs
    for (size_t b = 0; b < batch_size; ++b)
    {
        for (size_t s = 0; s < seq_len; ++s)
        {
            tid_data[b * seq_len + s] = static_cast<float>((b * 10 + s) % vocab_size);
        }
    }

    // Create full embedding table
    auto embedding_table = TensorFactory::create_simple({static_cast<int>(vocab_size), static_cast<int>(embed_dim)});
    float* emb_data = embedding_table->data();
    for (size_t i = 0; i < vocab_size * embed_dim; ++i)
    {
        emb_data[i] = static_cast<float>(i) * 0.001f;
    }

    // Create output
    auto output = TensorFactory::create_simple({static_cast<int>(batch_size), static_cast<int>(seq_len), static_cast<int>(embed_dim)});

    std::vector<std::shared_ptr<TensorBase>> inputs = {token_ids, embedding_table};
    std::vector<std::shared_ptr<TensorBase>> outputs = {output};

    // Execute
    bool success = op.execute(inputs, outputs);
    ASSERT_TRUE(success);

    // Verify output shape
    EXPECT_EQ(output->shape().size(), 3);
    EXPECT_EQ(output->shape()[0], batch_size);
    EXPECT_EQ(output->shape()[1], seq_len);
    EXPECT_EQ(output->shape()[2], embed_dim);

    // Spot check a few embeddings
    const float* out_data = output->data();
    
    // Check batch 0, token 0
    int token_id_0_0 = static_cast<int>(tid_data[0]);
    for (size_t d = 0; d < 4; ++d)  // Check first 4 dims
    {
        float expected = emb_data[token_id_0_0 * embed_dim + d];
        size_t idx = 0 * seq_len * embed_dim + 0 * embed_dim + d;
        EXPECT_FLOAT_EQ(out_data[idx], expected) << "Batch 0, Token 0, dim " << d;
    }
    
    // Check batch 3, token 7
    int token_id_3_7 = static_cast<int>(tid_data[3 * seq_len + 7]);
    for (size_t d = 0; d < 4; ++d)
    {
        float expected = emb_data[token_id_3_7 * embed_dim + d];
        size_t idx = 3 * seq_len * embed_dim + 7 * embed_dim + d;
        EXPECT_FLOAT_EQ(out_data[idx], expected) << "Batch 3, Token 7, dim " << d;
    }
}

TEST_F(MPIEmbeddingBatchTest, BatchSizeOne_MatchesSingleSequence)
{
    // Verify batch=1 produces same result as 1D input
    const size_t vocab_size = 50;
    const size_t embed_dim = 24;
    const size_t seq_len = 5;

    MPIEmbeddingOperator op(vocab_size, embed_dim);

    // Shared token IDs
    std::vector<int> token_vals = {5, 10, 15, 20, 25};

    // Test 1: 1D input [seq_len]
    auto token_ids_1d = TensorFactory::create_simple({static_cast<int>(seq_len)});
    float* tid_1d = token_ids_1d->data();
    for (size_t i = 0; i < seq_len; ++i)
    {
        tid_1d[i] = static_cast<float>(token_vals[i]);
    }

    // Test 2: 2D input [1, seq_len]
    auto token_ids_2d = TensorFactory::create_simple({1, static_cast<int>(seq_len)});
    float* tid_2d = token_ids_2d->data();
    for (size_t i = 0; i < seq_len; ++i)
    {
        tid_2d[i] = static_cast<float>(token_vals[i]);
    }

    // Shared embedding table
    auto embedding_table = TensorFactory::create_simple({static_cast<int>(vocab_size), static_cast<int>(embed_dim)});
    float* emb_data = embedding_table->data();
    for (size_t i = 0; i < vocab_size * embed_dim; ++i)
    {
        emb_data[i] = static_cast<float>(i) * 0.02f;
    }

    // Output 1: [seq_len, embed_dim]
    auto output_1d = TensorFactory::create_simple({static_cast<int>(seq_len), static_cast<int>(embed_dim)});
    
    // Output 2: [1, seq_len, embed_dim]
    auto output_2d = TensorFactory::create_simple({1, static_cast<int>(seq_len), static_cast<int>(embed_dim)});

    // Execute both
    std::vector<std::shared_ptr<TensorBase>> inputs_1d = {token_ids_1d, embedding_table};
    std::vector<std::shared_ptr<TensorBase>> outputs_1d = {output_1d};
    bool success_1d = op.execute(inputs_1d, outputs_1d);
    ASSERT_TRUE(success_1d);

    std::vector<std::shared_ptr<TensorBase>> inputs_2d = {token_ids_2d, embedding_table};
    std::vector<std::shared_ptr<TensorBase>> outputs_2d = {output_2d};
    bool success_2d = op.execute(inputs_2d, outputs_2d);
    ASSERT_TRUE(success_2d);

    // Compare outputs - should be identical
    const float* out_1d_data = output_1d->data();
    const float* out_2d_data = output_2d->data();
    
    for (size_t i = 0; i < seq_len * embed_dim; ++i)
    {
        EXPECT_FLOAT_EQ(out_1d_data[i], out_2d_data[i]) << "Element " << i;
    }
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    
    int result = RUN_ALL_TESTS();
    
    MPI_Finalize();
    return result;
}
