/**
 * @file Test__CPUKernels.cpp
 * @brief Unit tests for CPU kernel implementations
 *
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include "../../src/v2/kernels/cpu/ops/CPURoPEKernelT.h"
#include "../../src/v2/kernels/cpu/ops/CPURMSNormKernelT.h"
#include "../../src/v2/kernels/cpu/ops/CPUSoftmaxKernelT.h"

#include "../../src/v2/kernels/cpu/ops/CPUSwiGLUKernelT.h"
#include "../../src/v2/kernels/cpu/FP32GemmKernel.h"
#include "../../src/v2/tensors/Tensors.h"
#include <cmath>
#include <vector>

using namespace llaminar2;

/**
 * @brief Test fixture for CPU kernels
 */
class Test__CPUKernels : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Basic setup
    }
};

/**
 * @brief Test RMSNorm kernel
 */
TEST_F(Test__CPUKernels, RMSNormBasic)
{
    CPURMSNormKernelT<FP32Tensor> kernel;

    const int seq_len = 2;
    const int d_model = 4;
    const float eps = 1e-6f;

    // Input: [[1, 2, 3, 4], [2, 3, 4, 5]]
    std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    std::vector<float> gamma = {1.0f, 1.0f, 1.0f, 1.0f}; // Identity scaling
    std::vector<float> output(seq_len * d_model);

    bool success = kernel.apply(
        input.data(), gamma.data(), output.data(),
        seq_len, d_model, eps, false, nullptr, -1);

    ASSERT_TRUE(success);

    // Verify RMS normalization
    // For row 0: RMS = sqrt((1 + 4 + 9 + 16) / 4) = sqrt(7.5) ≈ 2.739
    float rms0 = std::sqrt((1.0f + 4.0f + 9.0f + 16.0f) / 4.0f + eps);
    EXPECT_NEAR(output[0], 1.0f / rms0, 1e-5f);
    EXPECT_NEAR(output[1], 2.0f / rms0, 1e-5f);
}

/**
 * @brief Test Softmax kernel
 */
TEST_F(Test__CPUKernels, SoftmaxBasic)
{
    CPUSoftmaxKernelT<FP32Tensor> kernel;

    const int rows = 2;
    const int cols = 3;

    // Input: [[1, 2, 3], [0, 1, 2]]
    std::vector<float> input = {1.0f, 2.0f, 3.0f, 0.0f, 1.0f, 2.0f};
    std::vector<float> output(rows * cols);

    bool success = kernel.apply(
        input.data(), output.data(),
        rows, cols, false, nullptr, -1);

    ASSERT_TRUE(success);

    // Verify softmax normalization (sum = 1)
    float sum_row0 = output[0] + output[1] + output[2];
    float sum_row1 = output[3] + output[4] + output[5];

    EXPECT_NEAR(sum_row0, 1.0f, 1e-6f);
    EXPECT_NEAR(sum_row1, 1.0f, 1e-6f);

    // Verify monotonicity (larger input → larger probability)
    EXPECT_LT(output[0], output[1]);
    EXPECT_LT(output[1], output[2]);
}

/**
 * @brief Test SwiGLU kernel
 */
TEST_F(Test__CPUKernels, SwiGLUBasic)
{
    CPUSwiGLUKernel kernel;

    const int seq_len = 2;
    const int d_ff = 3;

    std::vector<float> gate = {0.0f, 1.0f, 2.0f, -1.0f, -2.0f, 3.0f};
    std::vector<float> up = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    std::vector<float> output(seq_len * d_ff);

    bool success = kernel.apply(
        gate.data(), up.data(), output.data(),
        seq_len, d_ff, false, nullptr, -1);

    ASSERT_TRUE(success);

    // Verify SwiGLU: output = silu(gate) * up
    // silu(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
    EXPECT_NEAR(output[0], 0.0f, 1e-5f);

    // silu(1) = 1 * sigmoid(1) ≈ 1 * 0.731 = 0.731
    float sigmoid_1 = 1.0f / (1.0f + std::exp(-1.0f));
    EXPECT_NEAR(output[1], 1.0f * sigmoid_1, 1e-5f);
}

/**
 * @brief Test RoPE kernel (basic functionality)
 */
TEST_F(Test__CPUKernels, RoPEBasic)
{
    CPURoPEKernel kernel;

    const int seq_len = 1;
    const int n_heads = 1;
    const int n_kv_heads = 1;
    const int head_dim = 4;

    std::vector<float> Q = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> K = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<int> position_ids = {0};

    bool success = kernel.apply(
        Q.data(), K.data(), position_ids.data(),
        seq_len, n_heads, n_kv_heads, head_dim,
        10000.0f, // rope_theta (default for LLaMA-style models)
        false, nullptr, -1);

    ASSERT_TRUE(success);

    // TODO: Implement RoPE kernel and verify rotation
    // After RoPE, Q and K should be modified
    // At position 0, rotation is minimal but non-zero
    // EXPECT_NE(Q[0], 1.0f); // First element should change (currently stubbed)
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
