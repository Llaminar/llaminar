/**
 * @file test_swiglu_batch_operator.cpp
 * @brief Comprehensive test suite for MPISwiGLUBatchOperator
 *
 * Test Categories:
 * 1. Parity Tests: batch=1 matches MPISwiGLUOperator
 * 2. Equivalence Tests: batch=N matches N×single operations
 * 3. Shape Tests: Various batch sizes and sequence lengths
 * 4. Error Tests: Invalid inputs, shape mismatches
 *
 * @author David Sanftenberg
 */
#include <gtest/gtest.h>
#include <mpi.h>
#include "operators/MPISwiGLUBatchOperator.h"
#include "operators/MPISwiGLUOperator.h"
#include "tensors/SimpleTensor.h"
#include "Logger.h"
#include <cmath>
#include <random>

namespace llaminar
{

    class SwiGLUBatchOperatorTest : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
            MPI_Comm_size(MPI_COMM_WORLD, &world_size_);

            // Initialize random seed consistently across ranks
            std::srand(42);
        }

        // Helper: Create SimpleTensor with size_t shape (converts to int internally)
        std::shared_ptr<SimpleTensor> createTensor(const std::vector<size_t> &shape)
        {
            std::vector<int> int_shape(shape.begin(), shape.end());
            return std::make_shared<SimpleTensor>(int_shape);
        }

        // Helper: Create random tensor with specified shape
        std::shared_ptr<SimpleTensor> createRandomTensor(const std::vector<size_t> &shape)
        {
            std::vector<int> int_shape(shape.begin(), shape.end());
            auto tensor = std::make_shared<SimpleTensor>(int_shape);
            std::random_device rd;
            std::mt19937 gen(42); // Fixed seed for reproducibility
            std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

            for (size_t i = 0; i < tensor->size(); ++i)
            {
                tensor->data()[i] = dist(gen);
            }
            return tensor;
        }

        // Helper: Compare two tensors for approximate equality
        bool tensorsApproximatelyEqual(const std::shared_ptr<SimpleTensor> &a,
                                      const std::shared_ptr<SimpleTensor> &b,
                                      float rtol = 1e-5f,
                                      float atol = 1e-6f)
        {
            if (a->shape() != b->shape())
            {
                LOG_ERROR("Shape mismatch: a=" << a->shape().size() << "D, b=" << b->shape().size() << "D");
                return false;
            }

            const float *a_data = a->data();
            const float *b_data = b->data();
            size_t n = a->size();

            float max_diff = 0.0f;
            size_t mismatches = 0;

            for (size_t i = 0; i < n; ++i)
            {
                float diff = std::abs(a_data[i] - b_data[i]);
                float threshold = atol + rtol * std::max(std::abs(a_data[i]), std::abs(b_data[i]));

                if (diff > threshold)
                {
                    ++mismatches;
                    max_diff = std::max(max_diff, diff);
                }
            }

            if (mismatches > 0)
            {
                LOG_ERROR("Found " << mismatches << " mismatches out of " << n
                                  << " elements, max diff: " << max_diff);
            }

            return mismatches == 0;
        }

        // Helper: Extract single sequence from batch tensor
        std::shared_ptr<SimpleTensor> extractSequence(const std::shared_ptr<TensorBase>& batch_tensor, size_t batch_idx)
        {
            if (batch_tensor->shape().size() != 3) {
                LOG_ERROR("extractSequence requires 3D tensor, got " << batch_tensor->shape().size() << "D");
                return nullptr;
            }

            size_t batch_size = batch_tensor->shape()[0];
            size_t seq_len = batch_tensor->shape()[1];
            size_t hidden = batch_tensor->shape()[2];

            if (batch_idx >= batch_size) {
                LOG_ERROR("Batch index " << batch_idx << " out of range (batch_size=" << batch_size << ")");
                return nullptr;
            }

            auto sequence = std::make_shared<SimpleTensor>(std::vector<int>{static_cast<int>(seq_len), static_cast<int>(hidden)});
            size_t offset = batch_idx * seq_len * hidden;
            std::memcpy(sequence->data(), batch_tensor->data() + offset, seq_len * hidden * sizeof(float));
            return sequence;
        }

        int rank_;
        int world_size_;
    };

    // ========== PARITY TESTS: batch=1 vs MPISwiGLUOperator ==========

    TEST_F(SwiGLUBatchOperatorTest, ParitySmallDims)
    {
        // Test batch=1 produces identical results to MPISwiGLUOperator
        size_t batch_size = 1;
        size_t seq_len = 4;
        size_t hidden_ff = 8;

        // Create inputs [batch=1, seq_len, hidden_ff]
        auto gate_batch = createRandomTensor({batch_size, seq_len, hidden_ff});
        auto up_batch = createRandomTensor({batch_size, seq_len, hidden_ff});
        auto output_batch = createTensor({batch_size, seq_len, hidden_ff});

        // Batch operator execution
        MPISwiGLUBatchOperator batch_op;
        std::vector<std::shared_ptr<TensorBase>> inputs_batch = {gate_batch, up_batch};
        std::vector<std::shared_ptr<TensorBase>> outputs_batch = {output_batch};
        ASSERT_TRUE(batch_op.execute(inputs_batch, outputs_batch));

        // Create flattened [seq_len, hidden_ff] for reference operator
        auto gate_flat = createTensor({seq_len, hidden_ff});
        auto up_flat = createTensor({seq_len, hidden_ff});
        auto output_flat = createTensor({seq_len, hidden_ff});

        std::memcpy(gate_flat->data(), gate_batch->data(), seq_len * hidden_ff * sizeof(float));
        std::memcpy(up_flat->data(), up_batch->data(), seq_len * hidden_ff * sizeof(float));

        // Reference operator execution
        MPISwiGLUOperator ref_op;
        std::vector<std::shared_ptr<TensorBase>> inputs_ref = {gate_flat, up_flat};
        std::vector<std::shared_ptr<TensorBase>> outputs_ref = {output_flat};
        ASSERT_TRUE(ref_op.execute(inputs_ref, outputs_ref));

        // Extract sequence from batch output [1, seq, hidden] -> [seq, hidden]
        auto output_batch_extracted = extractSequence(output_batch, 0);
        ASSERT_NE(output_batch_extracted, nullptr);

        // Compare results (extracted batch output should match flattened output)
        EXPECT_TRUE(tensorsApproximatelyEqual(output_batch_extracted, output_flat, 1e-5f, 1e-6f))
            << "batch=1 batch operator should match reference MPISwiGLUOperator";
    }

    TEST_F(SwiGLUBatchOperatorTest, ParityLargeDims)
    {
        // Larger dimensions to stress test parity
        size_t batch_size = 1;
        size_t seq_len = 32;
        size_t hidden_ff = 128;

        auto gate_batch = createRandomTensor({batch_size, seq_len, hidden_ff});
        auto up_batch = createRandomTensor({batch_size, seq_len, hidden_ff});
        auto output_batch = createTensor({batch_size, seq_len, hidden_ff});

        MPISwiGLUBatchOperator batch_op;
        std::vector<std::shared_ptr<TensorBase>> inputs_batch = {gate_batch, up_batch};
        std::vector<std::shared_ptr<TensorBase>> outputs_batch = {output_batch};
        ASSERT_TRUE(batch_op.execute(inputs_batch, outputs_batch));

        auto gate_flat = createTensor({seq_len, hidden_ff});
        auto up_flat = createTensor({seq_len, hidden_ff});
        auto output_flat = createTensor({seq_len, hidden_ff});

        std::memcpy(gate_flat->data(), gate_batch->data(), seq_len * hidden_ff * sizeof(float));
        std::memcpy(up_flat->data(), up_batch->data(), seq_len * hidden_ff * sizeof(float));

        MPISwiGLUOperator ref_op;
        std::vector<std::shared_ptr<TensorBase>> inputs_ref = {gate_flat, up_flat};
        std::vector<std::shared_ptr<TensorBase>> outputs_ref = {output_flat};
        ASSERT_TRUE(ref_op.execute(inputs_ref, outputs_ref));

        // Extract sequence from batch output
        auto output_batch_extracted = extractSequence(output_batch, 0);
        ASSERT_NE(output_batch_extracted, nullptr);

        // Compare results
        EXPECT_TRUE(tensorsApproximatelyEqual(output_batch_extracted, output_flat, 1e-5f, 1e-6f))
            << "batch=1 batch operator should match reference for larger dimensions";
    }

    // ========== EQUIVALENCE TESTS: batch=N vs N×single ==========

    TEST_F(SwiGLUBatchOperatorTest, EquivalenceBatch3)
    {
        // Test batch=3 execution vs 3 separate single-batch executions
        size_t batch_size = 3;
        size_t seq_len = 8;
        size_t hidden_ff = 16;

        auto gate_batch = createRandomTensor({batch_size, seq_len, hidden_ff});
        auto up_batch = createRandomTensor({batch_size, seq_len, hidden_ff});
        auto output_batch = createTensor({batch_size, seq_len, hidden_ff});

        // Batch execution
        MPISwiGLUBatchOperator batch_op;
        std::vector<std::shared_ptr<TensorBase>> inputs_batch = {gate_batch, up_batch};
        std::vector<std::shared_ptr<TensorBase>> outputs_batch = {output_batch};
        ASSERT_TRUE(batch_op.execute(inputs_batch, outputs_batch));

        // Execute each batch element separately
        for (size_t b = 0; b < batch_size; ++b)
        {
            // Extract single batch element [1, seq_len, hidden_ff]
            auto gate_single = createTensor({1, seq_len, hidden_ff});
            auto up_single = createTensor({1, seq_len, hidden_ff});
            auto output_single = createTensor({1, seq_len, hidden_ff});

            size_t offset = b * seq_len * hidden_ff;
            std::memcpy(gate_single->data(), gate_batch->data() + offset, seq_len * hidden_ff * sizeof(float));
            std::memcpy(up_single->data(), up_batch->data() + offset, seq_len * hidden_ff * sizeof(float));

            MPISwiGLUBatchOperator single_op;
            std::vector<std::shared_ptr<TensorBase>> inputs_single = {gate_single, up_single};
            std::vector<std::shared_ptr<TensorBase>> outputs_single = {output_single};
            ASSERT_TRUE(single_op.execute(inputs_single, outputs_single));

            // Compare this batch element
            const float *batch_result = output_batch->data() + offset;
            const float *single_result = output_single->data();

            for (size_t i = 0; i < seq_len * hidden_ff; ++i)
            {
                EXPECT_NEAR(batch_result[i], single_result[i], 1e-6f)
                    << "Batch element " << b << " position " << i << " mismatch";
            }
        }
    }

    TEST_F(SwiGLUBatchOperatorTest, EquivalenceBatch16)
    {
        // Larger batch size
        size_t batch_size = 16;
        size_t seq_len = 4;
        size_t hidden_ff = 32;

        auto gate_batch = createRandomTensor({batch_size, seq_len, hidden_ff});
        auto up_batch = createRandomTensor({batch_size, seq_len, hidden_ff});
        auto output_batch = createTensor({batch_size, seq_len, hidden_ff});

        MPISwiGLUBatchOperator batch_op;
        std::vector<std::shared_ptr<TensorBase>> inputs_batch = {gate_batch, up_batch};
        std::vector<std::shared_ptr<TensorBase>> outputs_batch = {output_batch};
        ASSERT_TRUE(batch_op.execute(inputs_batch, outputs_batch));

        // Spot check first and last batch elements
        for (size_t b : {size_t(0), batch_size - 1})
        {
            auto gate_single = createTensor({1, seq_len, hidden_ff});
            auto up_single = createTensor({1, seq_len, hidden_ff});
            auto output_single = createTensor({1, seq_len, hidden_ff});

            size_t offset = b * seq_len * hidden_ff;
            std::memcpy(gate_single->data(), gate_batch->data() + offset, seq_len * hidden_ff * sizeof(float));
            std::memcpy(up_single->data(), up_batch->data() + offset, seq_len * hidden_ff * sizeof(float));

            MPISwiGLUBatchOperator single_op;
            std::vector<std::shared_ptr<TensorBase>> inputs_single = {gate_single, up_single};
            std::vector<std::shared_ptr<TensorBase>> outputs_single = {output_single};
            ASSERT_TRUE(single_op.execute(inputs_single, outputs_single));

            const float *batch_result = output_batch->data() + offset;
            const float *single_result = output_single->data();

            for (size_t i = 0; i < seq_len * hidden_ff; ++i)
            {
                EXPECT_NEAR(batch_result[i], single_result[i], 1e-6f)
                    << "Batch=" << batch_size << " element " << b << " position " << i << " mismatch";
            }
        }
    }

    // ========== SHAPE VALIDATION TESTS ==========

    TEST_F(SwiGLUBatchOperatorTest, VariousBatchSizesAndSeqLengths)
    {
        // Test various batch sizes and sequence lengths
        std::vector<std::tuple<size_t, size_t, size_t>> test_cases = {
            {1, 1, 8},      // Minimal
            {2, 4, 16},     // Small batch
            {4, 16, 32},    // Medium batch
            {8, 64, 64},    // Larger batch
            {16, 128, 128}, // Large batch
            {32, 512, 256}  // Very large
        };

        for (const auto &[batch, seq, hidden] : test_cases)
        {
            auto gate = createRandomTensor({batch, seq, hidden});
            auto up = createRandomTensor({batch, seq, hidden});
            auto output = createTensor({batch, seq, hidden});

            MPISwiGLUBatchOperator op;
            std::vector<std::shared_ptr<TensorBase>> inputs = {gate, up};
            std::vector<std::shared_ptr<TensorBase>> outputs = {output};

            EXPECT_TRUE(op.execute(inputs, outputs))
                << "Failed for batch=" << batch << ", seq=" << seq << ", hidden=" << hidden;

            // Verify output shape
            EXPECT_EQ(output->shape()[0], batch);
            EXPECT_EQ(output->shape()[1], seq);
            EXPECT_EQ(output->shape()[2], hidden);
        }
    }

    // ========== ERROR HANDLING TESTS ==========

    TEST_F(SwiGLUBatchOperatorTest, ShapeMismatchGateUp)
    {
        // Gate and up have different shapes
        auto gate = createRandomTensor({2, 4, 8});
        auto up = createRandomTensor({2, 4, 16}); // Different hidden_ff
        auto output = createTensor({2, 4, 8});

        MPISwiGLUBatchOperator op;
        std::vector<std::shared_ptr<TensorBase>> inputs = {gate, up};
        std::vector<std::shared_ptr<TensorBase>> outputs = {output};

        EXPECT_FALSE(op.execute(inputs, outputs)) << "Should reject gate/up shape mismatch";
    }

    TEST_F(SwiGLUBatchOperatorTest, InvalidInputDimensions)
    {
        // Gate is 2D instead of 3D
        auto gate = createRandomTensor({4, 8}); // Missing batch dimension
        auto up = createRandomTensor({2, 4, 8});
        auto output = createTensor({2, 4, 8});

        MPISwiGLUBatchOperator op;
        std::vector<std::shared_ptr<TensorBase>> inputs = {gate, up};
        std::vector<std::shared_ptr<TensorBase>> outputs = {output};

        EXPECT_FALSE(op.execute(inputs, outputs)) << "Should reject 2D gate tensor";
    }

} // namespace llaminar

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
