/**
 * @file test_mpi_rmsnorm_batch.cpp
 * @brief Unit tests for MPIRMSNormOperator batch dimension support
 * 
 * Tests verifying that MPIRMSNormOperator correctly handles:
 * - 2D inputs: [seq_len, hidden_size] (backward compatibility)
 * - 3D inputs: [batch, seq_len, hidden_size] (batch processing)
 * 
 * Strategy:
 * - Batched inputs are processed as flattened [batch*seq_len, hidden_size]
 * - Each row is normalized independently: RMS = sqrt(mean(x^2) + eps)
 * - Output: (x / RMS) * gamma (element-wise scale)
 * 
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <memory>
#include <vector>
#include <cmath>
#include <algorithm>
#include "../src/operators/MPIRMSNormOperator.h"
#include "../src/tensors/SimpleTensor.h"

using namespace llaminar;

class MPIRMSNormBatchTest : public ::testing::Test
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
     * @brief Compute reference RMSNorm on CPU for validation
     * @param input Input data [rows, cols]
     * @param gamma Scale parameters [cols]
     * @param output Output buffer [rows, cols]
     * @param rows Number of rows
     * @param cols Number of columns (hidden dimension)
     * @param epsilon Small constant for numerical stability
     */
    void computeReferenceRMSNorm(const float *input, const float *gamma,
                                 float *output, int rows, int cols,
                                 float epsilon = 1e-6f)
    {
        for (int r = 0; r < rows; ++r)
        {
            const float *row_in = input + r * cols;
            float *row_out = output + r * cols;
            
            // Compute RMS: sqrt(mean(x^2) + eps)
            double sum_sq = 0.0;
            for (int c = 0; c < cols; ++c)
            {
                double val = static_cast<double>(row_in[c]);
                sum_sq += val * val;
            }
            double mean_sq = sum_sq / static_cast<double>(cols);
            double rms = std::sqrt(mean_sq + static_cast<double>(epsilon));
            float inv_rms = static_cast<float>(1.0 / rms);
            
            // Apply normalization and scaling: (x / RMS) * gamma
            for (int c = 0; c < cols; ++c)
            {
                row_out[c] = row_in[c] * inv_rms * gamma[c];
            }
        }
    }

    /**
     * @brief Verify output contains no NaN or Inf values
     */
    bool hasNoNaNOrInf(const std::shared_ptr<SimpleTensor> &tensor)
    {
        const float *data = tensor->data();
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
 * Verifies that 2D input [seq_len, hidden_size] still works as before.
 */
TEST_F(MPIRMSNormBatchTest, SingleSequence_BackwardCompat)
{
    const int seq_len = 4;
    const int hidden_size = 8;
    const float epsilon = 1e-6f;

    // Create input tensor [seq_len, hidden_size]
    auto input = std::make_shared<SimpleTensor>(std::vector<int>{seq_len, hidden_size});
    auto gamma = std::make_shared<SimpleTensor>(std::vector<int>{hidden_size});
    auto output = std::make_shared<SimpleTensor>(std::vector<int>{seq_len, hidden_size});

    // Fill with test data
    for (int i = 0; i < seq_len * hidden_size; ++i)
    {
        input->data()[i] = static_cast<float>(i % 10) / 10.0f + 0.1f;
    }
    for (int i = 0; i < hidden_size; ++i)
    {
        gamma->data()[i] = 1.0f; // Unit scaling
    }

    // Execute MPIRMSNormOperator
    MPIRMSNormOperator op;
    op.setEpsilon(epsilon);
    
    std::vector<std::shared_ptr<TensorBase>> inputs = {input, gamma};
    std::vector<std::shared_ptr<TensorBase>> outputs = {output};
    
    bool success = op.execute(inputs, outputs);
    ASSERT_TRUE(success) << "MPIRMSNormOperator failed on 2D input";

    // Verify no NaN/Inf
    EXPECT_TRUE(hasNoNaNOrInf(output)) << "Output contains NaN or Inf";

    // Verify with reference implementation
    std::vector<float> expected(seq_len * hidden_size);
    computeReferenceRMSNorm(input->data(), gamma->data(), expected.data(),
                           seq_len, hidden_size, epsilon);

    // Compare outputs
    for (int i = 0; i < seq_len * hidden_size; ++i)
    {
        EXPECT_NEAR(output->data()[i], expected[i], 1e-5f)
            << "Mismatch at index " << i;
    }
}

/**
 * @brief Test batched sequences processing
 * 
 * Verifies that 3D input [batch, seq_len, hidden_size] is correctly processed.
 */
TEST_F(MPIRMSNormBatchTest, BatchedSequences)
{
    const int batch_size = 3;
    const int seq_len = 4;
    const int hidden_size = 8;
    const float epsilon = 1e-6f;

    // Create batched input tensor [batch, seq_len, hidden_size]
    auto input_3d = std::make_shared<SimpleTensor>(
        std::vector<int>{batch_size, seq_len, hidden_size});
    auto gamma = std::make_shared<SimpleTensor>(std::vector<int>{hidden_size});
    auto output_3d = std::make_shared<SimpleTensor>(
        std::vector<int>{batch_size, seq_len, hidden_size});

    // Fill with test data (different values per batch)
    for (int b = 0; b < batch_size; ++b)
    {
        for (int s = 0; s < seq_len; ++s)
        {
            for (int h = 0; h < hidden_size; ++h)
            {
                int idx = b * seq_len * hidden_size + s * hidden_size + h;
                input_3d->data()[idx] = static_cast<float>((idx + b * 7) % 13) / 10.0f;
            }
        }
    }
    for (int i = 0; i < hidden_size; ++i)
    {
        gamma->data()[i] = 1.0f + static_cast<float>(i) / 100.0f;
    }

    // Execute MPIRMSNormOperator with batched input
    MPIRMSNormOperator op;
    op.setEpsilon(epsilon);
    
    std::vector<std::shared_ptr<TensorBase>> inputs = {input_3d, gamma};
    std::vector<std::shared_ptr<TensorBase>> outputs = {output_3d};
    
    bool success = op.execute(inputs, outputs);
    ASSERT_TRUE(success) << "MPIRMSNormOperator failed on 3D batched input";

    // Verify output shape is correct
    ASSERT_EQ(output_3d->shape().size(), 3);
    EXPECT_EQ(output_3d->shape()[0], batch_size);
    EXPECT_EQ(output_3d->shape()[1], seq_len);
    EXPECT_EQ(output_3d->shape()[2], hidden_size);

    // Verify no NaN/Inf
    EXPECT_TRUE(hasNoNaNOrInf(output_3d)) << "Output contains NaN or Inf";

    // Verify with reference implementation (treat as flattened batch*seq_len rows)
    std::vector<float> expected(batch_size * seq_len * hidden_size);
    computeReferenceRMSNorm(input_3d->data(), gamma->data(), expected.data(),
                           batch_size * seq_len, hidden_size, epsilon);

    // Compare outputs
    for (int i = 0; i < batch_size * seq_len * hidden_size; ++i)
    {
        EXPECT_NEAR(output_3d->data()[i], expected[i], 1e-5f)
            << "Mismatch at index " << i;
    }
}

/**
 * @brief Test that batched processing produces same results as individual sequences
 * 
 * Verifies: RMSNorm(batch=[seq1, seq2]) == [RMSNorm(seq1), RMSNorm(seq2)]
 */
TEST_F(MPIRMSNormBatchTest, BatchProcessing_Correctness)
{
    const int batch_size = 2;
    const int seq_len = 4;
    const int hidden_size = 16;
    const float epsilon = 1e-6f;

    // Create batched input [batch_size, seq_len, hidden_size]
    auto batch_input = std::make_shared<SimpleTensor>(
        std::vector<int>{batch_size, seq_len, hidden_size});
    auto gamma = std::make_shared<SimpleTensor>(std::vector<int>{hidden_size});
    auto batch_output = std::make_shared<SimpleTensor>(
        std::vector<int>{batch_size, seq_len, hidden_size});

    // Fill with distinct test data per batch element
    for (int b = 0; b < batch_size; ++b)
    {
        for (int s = 0; s < seq_len; ++s)
        {
            for (int h = 0; h < hidden_size; ++h)
            {
                int idx = b * seq_len * hidden_size + s * hidden_size + h;
                batch_input->data()[idx] = static_cast<float>((idx * 3 + b * 11) % 17) / 10.0f + 0.1f;
            }
        }
    }
    for (int i = 0; i < hidden_size; ++i)
    {
        gamma->data()[i] = 0.5f + static_cast<float>(i % 5) / 10.0f;
    }

    // Process batch
    MPIRMSNormOperator op_batch;
    op_batch.setEpsilon(epsilon);
    
    std::vector<std::shared_ptr<TensorBase>> batch_inputs = {batch_input, gamma};
    std::vector<std::shared_ptr<TensorBase>> batch_outputs = {batch_output};
    
    bool success = op_batch.execute(batch_inputs, batch_outputs);
    ASSERT_TRUE(success) << "Batch processing failed";

    // Process each sequence individually
    std::vector<std::shared_ptr<SimpleTensor>> seq_outputs;
    for (int b = 0; b < batch_size; ++b)
    {
        auto seq_input = std::make_shared<SimpleTensor>(std::vector<int>{seq_len, hidden_size});
        auto seq_output = std::make_shared<SimpleTensor>(std::vector<int>{seq_len, hidden_size});
        
        // Extract sequence from batch
        for (int s = 0; s < seq_len; ++s)
        {
            for (int h = 0; h < hidden_size; ++h)
            {
                int batch_idx = b * seq_len * hidden_size + s * hidden_size + h;
                int seq_idx = s * hidden_size + h;
                seq_input->data()[seq_idx] = batch_input->data()[batch_idx];
            }
        }
        
        // Process individual sequence
        MPIRMSNormOperator op_seq;
        op_seq.setEpsilon(epsilon);
        
        std::vector<std::shared_ptr<TensorBase>> seq_inputs = {seq_input, gamma};
        std::vector<std::shared_ptr<TensorBase>> seq_out_vec = {seq_output};
        
        bool seq_success = op_seq.execute(seq_inputs, seq_out_vec);
        ASSERT_TRUE(seq_success) << "Individual sequence " << b << " processing failed";
        
        seq_outputs.push_back(seq_output);
    }

    // Compare batch output vs individual sequence outputs
    for (int b = 0; b < batch_size; ++b)
    {
        for (int s = 0; s < seq_len; ++s)
        {
            for (int h = 0; h < hidden_size; ++h)
            {
                int batch_idx = b * seq_len * hidden_size + s * hidden_size + h;
                int seq_idx = s * hidden_size + h;
                
                EXPECT_NEAR(batch_output->data()[batch_idx],
                           seq_outputs[b]->data()[seq_idx],
                           1e-5f)
                    << "Mismatch at batch=" << b << ", seq=" << s << ", hidden=" << h;
            }
        }
    }
}

/**
 * @brief Test large batch processing
 * 
 * Ensures batch support scales to larger batch sizes.
 */
TEST_F(MPIRMSNormBatchTest, LargeBatch)
{
    const int batch_size = 8;
    const int seq_len = 16;
    const int hidden_size = 32;
    const float epsilon = 1e-6f;

    // Create large batched input
    auto input = std::make_shared<SimpleTensor>(
        std::vector<int>{batch_size, seq_len, hidden_size});
    auto gamma = std::make_shared<SimpleTensor>(std::vector<int>{hidden_size});
    auto output = std::make_shared<SimpleTensor>(
        std::vector<int>{batch_size, seq_len, hidden_size});

    // Fill with randomized test data
    for (int i = 0; i < batch_size * seq_len * hidden_size; ++i)
    {
        input->data()[i] = static_cast<float>((i * 7 + 13) % 23) / 20.0f;
    }
    for (int i = 0; i < hidden_size; ++i)
    {
        gamma->data()[i] = 0.8f + static_cast<float>(i % 7) / 10.0f;
    }

    // Execute on large batch
    MPIRMSNormOperator op;
    op.setEpsilon(epsilon);
    
    std::vector<std::shared_ptr<TensorBase>> inputs = {input, gamma};
    std::vector<std::shared_ptr<TensorBase>> outputs = {output};
    
    bool success = op.execute(inputs, outputs);
    ASSERT_TRUE(success) << "Large batch processing failed";

    // Verify output shape
    ASSERT_EQ(output->shape().size(), 3);
    EXPECT_EQ(output->shape()[0], batch_size);
    EXPECT_EQ(output->shape()[1], seq_len);
    EXPECT_EQ(output->shape()[2], hidden_size);

    // Verify no NaN/Inf
    EXPECT_TRUE(hasNoNaNOrInf(output)) << "Output contains NaN or Inf";

    // Spot check a few rows against reference
    std::vector<float> expected(batch_size * seq_len * hidden_size);
    computeReferenceRMSNorm(input->data(), gamma->data(), expected.data(),
                           batch_size * seq_len, hidden_size, epsilon);

    // Sample check (first row, middle row, last row)
    std::vector<int> check_rows = {0, (batch_size * seq_len) / 2, batch_size * seq_len - 1};
    for (int row : check_rows)
    {
        for (int h = 0; h < hidden_size; ++h)
        {
            int idx = row * hidden_size + h;
            EXPECT_NEAR(output->data()[idx], expected[idx], 1e-5f)
                << "Mismatch at row=" << row << ", hidden=" << h;
        }
    }
}

int main(int argc, char **argv)
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    
    MPI_Finalize();
    return result;
}
