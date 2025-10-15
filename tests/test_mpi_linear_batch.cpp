/**
 * @file test_mpi_linear_batch.cpp
 * @brief Test MPILinearOperator batch dimension support
 * @author David Sanftenberg
 */

#include "operators/MPILinearOperator.h"
#include "tensors/SimpleTensor.h"
#include "tensors/TensorFactory.h"
#include <gtest/gtest.h>
#include <mpi.h>
#include <memory>
#include <vector>

using namespace llaminar;

class MPILinearBatchTest : public ::testing::Test
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

TEST_F(MPILinearBatchTest, SingleSequence_BackwardCompat)
{
    // Test 2D [seq_len, in_dim] input (backward compatibility)
    const size_t seq_len = 4;
    const size_t in_dim = 32;
    const size_t out_dim = 64;

    MPILinearOperator op(MPI_COMM_WORLD);

    // Create input [seq_len, in_dim]
    auto input = TensorFactory::create_simple({static_cast<int>(seq_len), static_cast<int>(in_dim)});
    float* in_data = input->data();
    for (size_t i = 0; i < seq_len * in_dim; ++i)
    {
        in_data[i] = static_cast<float>(i) * 0.01f;
    }

    // Create weight [out_dim, in_dim] (PyTorch convention)
    auto weight = TensorFactory::create_simple({static_cast<int>(out_dim), static_cast<int>(in_dim)});
    float* w_data = weight->data();
    for (size_t i = 0; i < out_dim * in_dim; ++i)
    {
        w_data[i] = static_cast<float>(i) * 0.001f;
    }

    // Create output [seq_len, out_dim]
    auto output = TensorFactory::create_simple({static_cast<int>(seq_len), static_cast<int>(out_dim)});

    std::vector<std::shared_ptr<TensorBase>> inputs = {input, weight};
    std::vector<std::shared_ptr<TensorBase>> outputs = {output};

    // Execute
    bool success = op.execute(inputs, outputs);
    ASSERT_TRUE(success);

    // Verify output shape
    EXPECT_EQ(output->shape().size(), 2);
    EXPECT_EQ(output->shape()[0], seq_len);
    EXPECT_EQ(output->shape()[1], out_dim);

    // Verify no NaNs in output
    const float* out_data = output->data();
    for (size_t i = 0; i < seq_len * out_dim; ++i)
    {
        EXPECT_FALSE(std::isnan(out_data[i])) << "NaN at position " << i;
        EXPECT_FALSE(std::isinf(out_data[i])) << "Inf at position " << i;
    }
}

TEST_F(MPILinearBatchTest, BatchedSequences_ViaReshape)
{
    // Test 3D [batch, seq_len, in_dim] input by reshaping to 2D
    const size_t batch_size = 3;
    const size_t seq_len = 4;
    const size_t in_dim = 16;
    const size_t out_dim = 32;

    MPILinearOperator op(MPI_COMM_WORLD);

    // Create batched input [batch, seq_len, in_dim] - use SimpleTensor directly for reshape_copy
    auto input_3d = std::make_shared<SimpleTensor>(std::vector<int>{static_cast<int>(batch_size), static_cast<int>(seq_len), static_cast<int>(in_dim)});
    float* in_data_3d = input_3d->data();
    for (size_t i = 0; i < batch_size * seq_len * in_dim; ++i)
    {
        in_data_3d[i] = static_cast<float>(i) * 0.01f;
    }

    // Reshape to 2D [batch*seq_len, in_dim] - this is what the pipeline will do
    auto input_2d = input_3d->reshape_copy({static_cast<int>(batch_size * seq_len), static_cast<int>(in_dim)});

    // Create weight [out_dim, in_dim]
    auto weight = TensorFactory::create_simple({static_cast<int>(out_dim), static_cast<int>(in_dim)});
    float* w_data = weight->data();
    for (size_t i = 0; i < out_dim * in_dim; ++i)
    {
        w_data[i] = static_cast<float>(i) * 0.001f;
    }

    // Create output [batch*seq_len, out_dim]
    auto output_2d = std::make_shared<SimpleTensor>(std::vector<int>{static_cast<int>(batch_size * seq_len), static_cast<int>(out_dim)});

    std::vector<std::shared_ptr<TensorBase>> inputs = {input_2d, weight};
    std::vector<std::shared_ptr<TensorBase>> outputs = {output_2d};

    // Execute
    bool success = op.execute(inputs, outputs);
    ASSERT_TRUE(success);

    // Verify output shape
    EXPECT_EQ(output_2d->shape().size(), 2);
    EXPECT_EQ(output_2d->shape()[0], batch_size * seq_len);
    EXPECT_EQ(output_2d->shape()[1], out_dim);

    // Reshape back to 3D [batch, seq_len, out_dim]
    auto output_3d = output_2d->reshape_copy({static_cast<int>(batch_size), static_cast<int>(seq_len), static_cast<int>(out_dim)});
    
    EXPECT_EQ(output_3d->shape().size(), 3);
    EXPECT_EQ(output_3d->shape()[0], batch_size);
    EXPECT_EQ(output_3d->shape()[1], seq_len);
    EXPECT_EQ(output_3d->shape()[2], out_dim);

    // Verify no NaNs
    const float* out_data = output_3d->data();
    for (size_t i = 0; i < batch_size * seq_len * out_dim; ++i)
    {
        EXPECT_FALSE(std::isnan(out_data[i])) << "NaN at position " << i;
    }
}

TEST_F(MPILinearBatchTest, BatchProcessing_Correctness)
{
    // Verify batch processing gives same results as processing sequences individually
    const size_t batch_size = 2;
    const size_t seq_len = 3;
    const size_t in_dim = 8;
    const size_t out_dim = 12;

    MPILinearOperator op(MPI_COMM_WORLD);

    // Shared weight [out_dim, in_dim]
    auto weight = TensorFactory::create_simple({static_cast<int>(out_dim), static_cast<int>(in_dim)});
    float* w_data = weight->data();
    for (size_t i = 0; i < out_dim * in_dim; ++i)
    {
        w_data[i] = static_cast<float>(i) * 0.01f;
    }

    // Create individual sequence inputs
    std::vector<std::shared_ptr<SimpleTensor>> seq_inputs;
    std::vector<std::shared_ptr<SimpleTensor>> seq_outputs;
    
    for (size_t b = 0; b < batch_size; ++b)
    {
        auto seq_input = std::make_shared<SimpleTensor>(std::vector<int>{static_cast<int>(seq_len), static_cast<int>(in_dim)});
        float* seq_in_data = seq_input->data();
        for (size_t i = 0; i < seq_len * in_dim; ++i)
        {
            seq_in_data[i] = static_cast<float>(b * 100 + i) * 0.001f;
        }
        seq_inputs.push_back(seq_input);

        auto seq_output = std::make_shared<SimpleTensor>(std::vector<int>{static_cast<int>(seq_len), static_cast<int>(out_dim)});
        seq_outputs.push_back(seq_output);
    }

    // Process each sequence individually
    for (size_t b = 0; b < batch_size; ++b)
    {
        std::vector<std::shared_ptr<TensorBase>> inputs = {seq_inputs[b], weight};
        std::vector<std::shared_ptr<TensorBase>> outputs = {seq_outputs[b]};
        bool success = op.execute(inputs, outputs);
        ASSERT_TRUE(success) << "Sequence " << b << " failed";
    }

    // Stack sequences into batch
    auto batch_input = SimpleTensor::stack_batch(seq_inputs);
    auto batch_input_2d = batch_input->reshape_copy({static_cast<int>(batch_size * seq_len), static_cast<int>(in_dim)});
    
    auto batch_output_2d = std::make_shared<SimpleTensor>(std::vector<int>{static_cast<int>(batch_size * seq_len), static_cast<int>(out_dim)});
    
    std::vector<std::shared_ptr<TensorBase>> batch_inputs = {batch_input_2d, weight};
    std::vector<std::shared_ptr<TensorBase>> batch_outputs = {batch_output_2d};
    bool success = op.execute(batch_inputs, batch_outputs);
    ASSERT_TRUE(success);

    // Reshape batch output back to 3D
    auto batch_output_3d = batch_output_2d->reshape_copy({static_cast<int>(batch_size), static_cast<int>(seq_len), static_cast<int>(out_dim)});

    // Compare: batch output should match concatenated individual outputs
    for (size_t b = 0; b < batch_size; ++b)
    {
        const float* seq_data = seq_outputs[b]->data();
        const float* batch_data = batch_output_3d->data() + b * seq_len * out_dim;
        
        for (size_t i = 0; i < seq_len * out_dim; ++i)
        {
            EXPECT_NEAR(seq_data[i], batch_data[i], 1e-4f) 
                << "Mismatch at batch " << b << ", element " << i
                << " (individual=" << seq_data[i] << ", batched=" << batch_data[i] << ")";
        }
    }
}

TEST_F(MPILinearBatchTest, LargeBatch)
{
    // Test with larger batch to verify scalability
    const size_t batch_size = 8;
    const size_t seq_len = 16;
    const size_t in_dim = 64;
    const size_t out_dim = 128;

    MPILinearOperator op(MPI_COMM_WORLD);

    // Create batched input via reshape
    auto input_2d = TensorFactory::create_simple({static_cast<int>(batch_size * seq_len), static_cast<int>(in_dim)});
    float* in_data = input_2d->data();
    for (size_t i = 0; i < batch_size * seq_len * in_dim; ++i)
    {
        in_data[i] = static_cast<float>(i) * 0.0001f;
    }

    // Create weight
    auto weight = TensorFactory::create_simple({static_cast<int>(out_dim), static_cast<int>(in_dim)});
    float* w_data = weight->data();
    for (size_t i = 0; i < out_dim * in_dim; ++i)
    {
        w_data[i] = static_cast<float>(i) * 0.00001f;
    }

    // Create output
    auto output_2d = TensorFactory::create_simple({static_cast<int>(batch_size * seq_len), static_cast<int>(out_dim)});

    std::vector<std::shared_ptr<TensorBase>> inputs = {input_2d, weight};
    std::vector<std::shared_ptr<TensorBase>> outputs = {output_2d};

    // Execute
    bool success = op.execute(inputs, outputs);
    ASSERT_TRUE(success);

    // Verify output
    EXPECT_EQ(output_2d->shape()[0], batch_size * seq_len);
    EXPECT_EQ(output_2d->shape()[1], out_dim);

    // Verify no NaNs
    const float* out_data = output_2d->data();
    size_t nan_count = 0;
    for (size_t i = 0; i < batch_size * seq_len * out_dim; ++i)
    {
        if (std::isnan(out_data[i])) ++nan_count;
    }
    EXPECT_EQ(nan_count, 0) << "Found " << nan_count << " NaNs in output";
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    
    int result = RUN_ALL_TESTS();
    
    MPI_Finalize();
    return result;
}
