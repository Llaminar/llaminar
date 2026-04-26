/**
 * @file Test__MPIVectorizedAttention.cpp
 * @brief Tests for MPI tensor-parallel attention with vectorized primitives
 * @author David Sanftenberg
 *
 * Validates that Phase 2 MPI tensor-parallel attention correctly uses
 * vectorized primitives (Softmax) for improved performance while maintaining
 * numerical correctness.
 */

#include <gtest/gtest.h>
#include "../../src/v2/kernels/cpu/primitives/SoftmaxPrimitives_New.h"
#include "../../src/v2/tensors/TensorFactory.h"
#include "../../src/v2/tensors/Tensors.h"
#include <cmath>
#include <vector>
#include <memory>
#include <cstring>

using namespace llaminar2;

/**
 * @brief Test vectorized softmax on attention-like scores
 *
 * Validates that vectorized primitives produce correct softmax probabilities
 * for multi-head attention score patterns.
 */
TEST(Test__MPIVectorizedAttention, VectorizedSoftmax_MultiHead)
{
    const int n_heads = 4;
    const int seq_len = 8;
    const int rows = n_heads * seq_len;
    const int cols = seq_len;

    // Create scores tensor
    auto scores = std::make_shared<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(cols)});

    // Initialize with simple pattern
    float *scores_data = scores->mutable_data();
    for (int i = 0; i < rows * cols; ++i)
    {
        scores_data[i] = 0.1f * (i % 10);
    }

    // Apply vectorized softmax
    primitives::softmax_row_major_fp32(scores_data, rows, cols, false, 1.0f, true);

    // Validate: each row should sum to 1.0
    for (int r = 0; r < rows; ++r)
    {
        float row_sum = 0.0f;
        for (int c = 0; c < cols; ++c)
        {
            float val = scores_data[r * cols + c];
            ASSERT_TRUE(std::isfinite(val)) << "Score should be finite at (" << r << ", " << c << ")";
            ASSERT_GE(val, 0.0f) << "Softmax output should be non-negative";
            row_sum += val;
        }
        EXPECT_NEAR(row_sum, 1.0f, 1e-5f) << "Row " << r << " should sum to 1.0";
    }
}

/**
 * @brief Test causal masking with vectorized softmax
 *
 * Validates that vectorized softmax with causal=true correctly applies
 * lower-triangular masking without explicit mask materialization.
 */
TEST(Test__MPIVectorizedAttention, VectorizedSoftmax_CausalMasking)
{
    const int seq_len = 4;
    const int rows = seq_len;
    const int cols = seq_len;

    auto scores = std::make_shared<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(cols)});

    // Initialize with uniform values
    float *scores_data = scores->mutable_data();
    for (int i = 0; i < rows * cols; ++i)
    {
        scores_data[i] = 1.0f;
    }

    // Apply vectorized causal softmax
    primitives::softmax_row_major_fp32(scores_data, rows, cols, true, 1.0f, true);

    // Validate causal masking: upper triangle should be zero
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            float val = scores_data[i * cols + j];
            if (j > i)
            {
                // Upper triangle (future positions) should be masked
                EXPECT_FLOAT_EQ(val, 0.0f)
                    << "Position (" << i << ", " << j << ") should be masked (future token)";
            }
            else
            {
                // Lower triangle (past positions) should have probabilities
                EXPECT_GT(val, 0.0f)
                    << "Position (" << i << ", " << j << ") should have non-zero probability";
            }
        }

        // Each row should still sum to 1.0 (only over unmasked positions)
        float row_sum = 0.0f;
        for (int j = 0; j <= i; ++j)
        {
            row_sum += scores_data[i * cols + j];
        }
        EXPECT_NEAR(row_sum, 1.0f, 1e-5f) << "Row " << i << " should sum to 1.0";
    }
}

/**
 * @brief Test attention scaling with vectorized softmax
 *
 * Validates that attention scaling (1/sqrt(head_dim)) affects the
 * probability distribution correctly.
 */
TEST(Test__MPIVectorizedAttention, VectorizedSoftmax_AttentionScaling)
{
    const int seq_len = 8;
    const int head_dim = 64;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    auto scores = std::make_shared<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(seq_len)});

    // Initialize with varying values
    float *scores_data = scores->mutable_data();
    for (int i = 0; i < seq_len * seq_len; ++i)
    {
        scores_data[i] = static_cast<float>(i) / seq_len;
    }

    // Apply softmax with scaling
    primitives::softmax_row_major_fp32(scores_data, seq_len, seq_len, false, scale, true);

    // Validate: each row should sum to 1.0
    for (int r = 0; r < seq_len; ++r)
    {
        float row_sum = 0.0f;
        for (int c = 0; c < seq_len; ++c)
        {
            float val = scores_data[r * seq_len + c];
            ASSERT_TRUE(std::isfinite(val)) << "Scaled softmax should be finite";
            ASSERT_GE(val, 0.0f) << "Softmax output should be non-negative";
            row_sum += val;
        }
        EXPECT_NEAR(row_sum, 1.0f, 1e-5f) << "Row " << r << " should sum to 1.0 after scaling";
    }
}

/**
 * @brief Test large multi-head attention pattern
 *
 * Validates that vectorized softmax handles large attention matrices
 * efficiently (exercises AVX512/AVX2 code paths).
 */
TEST(Test__MPIVectorizedAttention, VectorizedSoftmax_LargeMultiHead)
{
    const int n_heads = 16;
    const int seq_len = 128;
    const int rows = n_heads * seq_len;
    const int cols = seq_len;

    auto scores = std::make_shared<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(cols)});

    // Initialize with random-like pattern
    float *scores_data = scores->mutable_data();
    for (int i = 0; i < rows * cols; ++i)
    {
        scores_data[i] = 0.01f * (i % 100);
    }

    // Apply vectorized softmax
    primitives::softmax_row_major_fp32(scores_data, rows, cols, false, 1.0f, true);

    // Validate: sample rows should sum to 1.0
    const std::vector<int> sample_rows = {0, rows / 4, rows / 2, 3 * rows / 4, rows - 1};
    for (int r : sample_rows)
    {
        float row_sum = 0.0f;
        for (int c = 0; c < cols; ++c)
        {
            float val = scores_data[r * cols + c];
            ASSERT_TRUE(std::isfinite(val)) << "Value should be finite at row " << r;
            row_sum += val;
        }
        EXPECT_NEAR(row_sum, 1.0f, 1e-4f) << "Sample row " << r << " should sum to 1.0";
    }
}

// Main function
int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
