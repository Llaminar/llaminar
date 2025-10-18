/**
 * @file test_linear_batch_operator.cpp
 * @brief Comprehensive parity and validation tests for MPILinearBatchOperator
 *
 * Test Strategy:
 * 1. Parity tests: Verify batch=1 produces identical results to MPILinearOperator
 * 2. Equivalence tests: Verify batch=N equals N× single sequence operations
 * 3. Shape validation: Test various batch sizes and sequence lengths
 * 4. Edge cases: Empty batches, large batches, single tokens, long sequences
 *
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include "operators/MPILinearOperator.h"
#include "operators/MPILinearBatchOperator.h"
#include "tensors/SimpleTensor.h"
#include "Logger.h"
#include <random>
#include <cmath>

using namespace llaminar;

namespace
{
    // Test fixture with common setup
    class LinearBatchOperatorTest : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
            MPI_Comm_size(MPI_COMM_WORLD, &world_size_);

            // Initialize random number generator with fixed seed for reproducibility
            rng_.seed(42);
        }

        void TearDown() override
        {
            // Cleanup
        }

        // Helper: Create random tensor with given shape
        std::shared_ptr<SimpleTensor> createRandomTensor(const std::vector<size_t> &shape, float mean = 0.0f, float stddev = 1.0f)
        {
            // Convert size_t to int for SimpleTensor
            std::vector<int> int_shape(shape.begin(), shape.end());
            auto tensor = std::make_shared<SimpleTensor>(int_shape);
            std::normal_distribution<float> dist(mean, stddev);

            for (size_t i = 0; i < tensor->size(); ++i)
            {
                tensor->data()[i] = dist(rng_);
            }

            return tensor;
        }

        // Helper: Create zero tensor
        std::shared_ptr<SimpleTensor> createZeroTensor(const std::vector<size_t> &shape)
        {
            // Convert size_t to int for SimpleTensor
            std::vector<int> int_shape(shape.begin(), shape.end());
            auto tensor = std::make_shared<SimpleTensor>(int_shape);
            std::fill_n(tensor->data(), tensor->size(), 0.0f);
            return tensor;
        }

        // Helper: Compare two tensors with relative tolerance
        void expectTensorsNear(const std::shared_ptr<TensorBase> &a,
                               const std::shared_ptr<TensorBase> &b,
                               float rel_tol = 1e-5f,
                               float abs_tol = 1e-6f)
        {
            ASSERT_EQ(a->shape(), b->shape()) << "Tensor shapes must match";

            size_t mismatch_count = 0;
            float max_diff = 0.0f;
            size_t max_diff_idx = 0;

            for (size_t i = 0; i < a->size(); ++i)
            {
                float diff = std::abs(a->data()[i] - b->data()[i]);
                float threshold = abs_tol + rel_tol * std::abs(a->data()[i]);

                if (diff > threshold)
                {
                    mismatch_count++;
                    if (diff > max_diff)
                    {
                        max_diff = diff;
                        max_diff_idx = i;
                    }
                }
            }

            if (mismatch_count > 0)
            {
                LOG_ERROR("Tensor mismatch: " << mismatch_count << " / " << a->size() << " elements differ");
                LOG_ERROR("Max diff: " << max_diff << " at index " << max_diff_idx
                                       << " (a=" << a->data()[max_diff_idx] << ", b=" << b->data()[max_diff_idx] << ")");
            }

            EXPECT_EQ(mismatch_count, 0) << "Tensors should be equal within tolerance";
        }

        // Helper: Extract single sequence from batch tensor
        std::shared_ptr<SimpleTensor> extractSequence(const std::shared_ptr<TensorBase> &batch_tensor, size_t batch_idx)
        {
            EXPECT_EQ(batch_tensor->shape().size(), 3) << "Must be 3D batch tensor";

            size_t batch_size = batch_tensor->shape()[0];
            size_t seq_len = batch_tensor->shape()[1];
            size_t hidden = batch_tensor->shape()[2];

            EXPECT_LT(batch_idx, batch_size) << "Batch index out of range";

            auto sequence = std::make_shared<SimpleTensor>(std::vector<int>{static_cast<int>(seq_len), static_cast<int>(hidden)});

            // Copy data for this sequence
            size_t src_offset = batch_idx * seq_len * hidden;
            std::memcpy(sequence->data(), batch_tensor->data() + src_offset, seq_len * hidden * sizeof(float));

            return sequence;
        }

        int rank_;
        int world_size_;
        std::mt19937 rng_;
    };

    // ====================================================================================
    // PARITY TESTS: Verify batch=1 matches MPILinearOperator
    // ====================================================================================

    TEST_F(LinearBatchOperatorTest, ParityWithSingleSequence_SmallDims)
    {
        // Test with small dimensions: batch=1, seq=8, in=64, out=32
        const size_t seq_len = 8;
        const size_t in_dim = 64;
        const size_t out_dim = 32;

        // Create test data
        auto input_2d = createRandomTensor({seq_len, in_dim});
        auto input_3d = createRandomTensor({1, seq_len, in_dim});
        auto weight = createRandomTensor({out_dim, in_dim});

        // Copy same data to 3D input
        std::memcpy(input_3d->data(), input_2d->data(), seq_len * in_dim * sizeof(float));

        // Create operators
        auto old_op = std::make_unique<MPILinearOperator>();
        auto new_op = std::make_unique<MPILinearBatchOperator>();

        // Prepare outputs
        auto output_2d = createZeroTensor({seq_len, out_dim});
        auto output_3d = createZeroTensor({1, seq_len, out_dim});

        // Execute both operators
        std::vector<std::shared_ptr<TensorBase>> inputs_2d = {input_2d, weight};
        std::vector<std::shared_ptr<TensorBase>> outputs_2d = {output_2d};

        std::vector<std::shared_ptr<TensorBase>> inputs_3d = {input_3d, weight};
        std::vector<std::shared_ptr<TensorBase>> outputs_3d = {output_3d};

        ASSERT_TRUE(old_op->execute(inputs_2d, outputs_2d)) << "Old operator execution failed";
        ASSERT_TRUE(new_op->execute(inputs_3d, outputs_3d)) << "New operator execution failed";

        // Extract sequence from batch output
        auto extracted = extractSequence(output_3d, 0);

        // Compare results
        if (rank_ == 0)
        {
            LOG_INFO("Testing parity: batch=1, seq=" << seq_len << ", in=" << in_dim << ", out=" << out_dim);
        }

        expectTensorsNear(output_2d, extracted, 1e-5f, 1e-6f);
    }

    TEST_F(LinearBatchOperatorTest, ParityWithSingleSequence_LargeDims)
    {
        // Test with larger dimensions: batch=1, seq=128, in=512, out=256
        const size_t seq_len = 128;
        const size_t in_dim = 512;
        const size_t out_dim = 256;

        // Create test data
        auto input_2d = createRandomTensor({seq_len, in_dim});
        auto input_3d = createRandomTensor({1, seq_len, in_dim});
        auto weight = createRandomTensor({out_dim, in_dim});

        // Copy same data
        std::memcpy(input_3d->data(), input_2d->data(), seq_len * in_dim * sizeof(float));

        // Create operators
        auto old_op = std::make_unique<MPILinearOperator>();
        auto new_op = std::make_unique<MPILinearBatchOperator>();

        // Prepare outputs
        auto output_2d = createZeroTensor({seq_len, out_dim});
        auto output_3d = createZeroTensor({1, seq_len, out_dim});

        // Execute
        std::vector<std::shared_ptr<TensorBase>> inputs_2d = {input_2d, weight};
        std::vector<std::shared_ptr<TensorBase>> outputs_2d = {output_2d};

        std::vector<std::shared_ptr<TensorBase>> inputs_3d = {input_3d, weight};
        std::vector<std::shared_ptr<TensorBase>> outputs_3d = {output_3d};

        ASSERT_TRUE(old_op->execute(inputs_2d, outputs_2d));
        ASSERT_TRUE(new_op->execute(inputs_3d, outputs_3d));

        // Compare
        auto extracted = extractSequence(output_3d, 0);
        expectTensorsNear(output_2d, extracted, 1e-5f, 1e-6f);
    }

    TEST_F(LinearBatchOperatorTest, ParityWithBias)
    {
        // Test parity including bias term
        const size_t seq_len = 32;
        const size_t in_dim = 128;
        const size_t out_dim = 64;

        auto input_2d = createRandomTensor({seq_len, in_dim});
        auto input_3d = createRandomTensor({1, seq_len, in_dim});
        auto weight = createRandomTensor({out_dim, in_dim});
        auto bias = createRandomTensor({out_dim});

        std::memcpy(input_3d->data(), input_2d->data(), seq_len * in_dim * sizeof(float));

        auto old_op = std::make_unique<MPILinearOperator>();
        auto new_op = std::make_unique<MPILinearBatchOperator>();

        auto output_2d = createZeroTensor({seq_len, out_dim});
        auto output_3d = createZeroTensor({1, seq_len, out_dim});

        std::vector<std::shared_ptr<TensorBase>> inputs_2d = {input_2d, weight, bias};
        std::vector<std::shared_ptr<TensorBase>> outputs_2d = {output_2d};

        std::vector<std::shared_ptr<TensorBase>> inputs_3d = {input_3d, weight, bias};
        std::vector<std::shared_ptr<TensorBase>> outputs_3d = {output_3d};

        ASSERT_TRUE(old_op->execute(inputs_2d, outputs_2d));
        ASSERT_TRUE(new_op->execute(inputs_3d, outputs_3d));

        auto extracted = extractSequence(output_3d, 0);
        expectTensorsNear(output_2d, extracted, 1e-5f, 1e-6f);
    }

    // ====================================================================================
    // EQUIVALENCE TESTS: Verify batch=N equals N× single sequence
    // ====================================================================================

    TEST_F(LinearBatchOperatorTest, BatchEquivalence_MultipleSequences)
    {
        // Test that processing 3 sequences in batch gives same result as processing separately
        const size_t batch_size = 3;
        const size_t seq_len = 16;
        const size_t in_dim = 64;
        const size_t out_dim = 32;

        auto weight = createRandomTensor({out_dim, in_dim});
        auto new_op = std::make_unique<MPILinearBatchOperator>();

        // Create batch input
        auto batch_input = createRandomTensor({batch_size, seq_len, in_dim});
        auto batch_output = createZeroTensor({batch_size, seq_len, out_dim});

        // Execute batch operation
        std::vector<std::shared_ptr<TensorBase>> batch_inputs = {batch_input, weight};
        std::vector<std::shared_ptr<TensorBase>> batch_outputs = {batch_output};

        ASSERT_TRUE(new_op->execute(batch_inputs, batch_outputs));

        // Process each sequence individually
        for (size_t b = 0; b < batch_size; ++b)
        {
            // Create single-sequence batch
            auto single_input = createZeroTensor({1, seq_len, in_dim});
            auto single_output = createZeroTensor({1, seq_len, out_dim});

            // Copy data for this sequence
            size_t offset = b * seq_len * in_dim;
            std::memcpy(single_input->data(), batch_input->data() + offset, seq_len * in_dim * sizeof(float));

            // Execute
            std::vector<std::shared_ptr<TensorBase>> single_inputs = {single_input, weight};
            std::vector<std::shared_ptr<TensorBase>> single_outputs = {single_output};

            ASSERT_TRUE(new_op->execute(single_inputs, single_outputs));

            // Extract corresponding sequence from batch result
            auto batch_sequence = extractSequence(batch_output, b);
            auto single_sequence = extractSequence(single_output, 0);

            // Compare
            if (rank_ == 0)
            {
                LOG_INFO("Comparing batch sequence " << b << " with individual execution");
            }
            expectTensorsNear(batch_sequence, single_sequence, 1e-5f, 1e-6f);
        }
    }

    TEST_F(LinearBatchOperatorTest, BatchEquivalence_LargeBatch)
    {
        // Test with larger batch size
        const size_t batch_size = 16;
        const size_t seq_len = 8;
        const size_t in_dim = 32;
        const size_t out_dim = 16;

        auto weight = createRandomTensor({out_dim, in_dim});
        auto new_op = std::make_unique<MPILinearBatchOperator>();

        auto batch_input = createRandomTensor({batch_size, seq_len, in_dim});
        auto batch_output = createZeroTensor({batch_size, seq_len, out_dim});

        std::vector<std::shared_ptr<TensorBase>> batch_inputs = {batch_input, weight};
        std::vector<std::shared_ptr<TensorBase>> batch_outputs = {batch_output};

        ASSERT_TRUE(new_op->execute(batch_inputs, batch_outputs));

        // Spot check a few sequences (not all to keep test time reasonable)
        for (size_t b : {0, 5, 10, 15})
        {
            auto single_input = createZeroTensor({1, seq_len, in_dim});
            auto single_output = createZeroTensor({1, seq_len, out_dim});

            size_t offset = b * seq_len * in_dim;
            std::memcpy(single_input->data(), batch_input->data() + offset, seq_len * in_dim * sizeof(float));

            std::vector<std::shared_ptr<TensorBase>> single_inputs = {single_input, weight};
            std::vector<std::shared_ptr<TensorBase>> single_outputs = {single_output};

            ASSERT_TRUE(new_op->execute(single_inputs, single_outputs));

            auto batch_sequence = extractSequence(batch_output, b);
            auto single_sequence = extractSequence(single_output, 0);

            expectTensorsNear(batch_sequence, single_sequence, 1e-5f, 1e-6f);
        }
    }

    // ====================================================================================
    // SHAPE VALIDATION TESTS
    // ====================================================================================

    TEST_F(LinearBatchOperatorTest, VariousBatchSizes)
    {
        // Test different batch sizes work correctly
        const size_t seq_len = 16;
        const size_t in_dim = 64;
        const size_t out_dim = 32;

        auto weight = createRandomTensor({out_dim, in_dim});
        auto op = std::make_unique<MPILinearBatchOperator>();

        for (size_t batch_size : {1, 2, 4, 8, 16, 32})
        {
            auto input = createRandomTensor({batch_size, seq_len, in_dim});
            auto output = createZeroTensor({batch_size, seq_len, out_dim});

            std::vector<std::shared_ptr<TensorBase>> inputs = {input, weight};
            std::vector<std::shared_ptr<TensorBase>> outputs = {output};

            ASSERT_TRUE(op->execute(inputs, outputs)) << "Failed for batch_size=" << batch_size;

            // Verify output shape
            EXPECT_EQ(output->shape()[0], batch_size);
            EXPECT_EQ(output->shape()[1], seq_len);
            EXPECT_EQ(output->shape()[2], out_dim);

            // Verify output is not all zeros
            bool has_nonzero = false;
            for (size_t i = 0; i < output->size(); ++i)
            {
                if (std::abs(output->data()[i]) > 1e-6f)
                {
                    has_nonzero = true;
                    break;
                }
            }
            EXPECT_TRUE(has_nonzero) << "Output should not be all zeros for batch_size=" << batch_size;
        }
    }

    TEST_F(LinearBatchOperatorTest, VariousSequenceLengths)
    {
        // Test different sequence lengths
        const size_t batch_size = 4;
        const size_t in_dim = 64;
        const size_t out_dim = 32;

        auto weight = createRandomTensor({out_dim, in_dim});
        auto op = std::make_unique<MPILinearBatchOperator>();

        for (size_t seq_len : {1, 8, 32, 128, 512})
        {
            auto input = createRandomTensor({batch_size, seq_len, in_dim});
            auto output = createZeroTensor({batch_size, seq_len, out_dim});

            std::vector<std::shared_ptr<TensorBase>> inputs = {input, weight};
            std::vector<std::shared_ptr<TensorBase>> outputs = {output};

            ASSERT_TRUE(op->execute(inputs, outputs)) << "Failed for seq_len=" << seq_len;

            EXPECT_EQ(output->shape()[0], batch_size);
            EXPECT_EQ(output->shape()[1], seq_len);
            EXPECT_EQ(output->shape()[2], out_dim);
        }
    }

    // ====================================================================================
    // ERROR HANDLING TESTS
    // ====================================================================================

    TEST_F(LinearBatchOperatorTest, DimensionMismatch)
    {
        auto op = std::make_unique<MPILinearBatchOperator>();

        // Input: [4, 16, 64], Weight: [32, 128] - should fail (64 != 128)
        auto input = createRandomTensor({4, 16, 64});
        auto weight = createRandomTensor({32, 128});
        auto output = createZeroTensor({4, 16, 32});

        std::vector<std::shared_ptr<TensorBase>> inputs = {input, weight};
        std::vector<std::shared_ptr<TensorBase>> outputs = {output};

        EXPECT_FALSE(op->execute(inputs, outputs)) << "Should fail with dimension mismatch";
    }

    TEST_F(LinearBatchOperatorTest, InvalidInputDimensions)
    {
        auto op = std::make_unique<MPILinearBatchOperator>();

        // 2D input should fail (expects 3D)
        auto input_2d = createRandomTensor({16, 64});
        auto weight = createRandomTensor({32, 64});
        auto output = createZeroTensor({16, 32});

        std::vector<std::shared_ptr<TensorBase>> inputs = {input_2d, weight};
        std::vector<std::shared_ptr<TensorBase>> outputs = {output};

        EXPECT_FALSE(op->execute(inputs, outputs)) << "Should fail with 2D input";
    }

} // anonymous namespace

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Run tests
    int result = RUN_ALL_TESTS();

    MPI_Finalize();
    return result;
}
