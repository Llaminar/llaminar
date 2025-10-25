/**
 * @file Test__FP32Tensor.cpp
 * @brief Unit tests for FP32Tensor class
 * @author David Sanftenberg
 *
 * This is an example unit test demonstrating the V2 test infrastructure.
 * It tests the FP32Tensor class without requiring model loading.
 *
 * Naming convention: Test file and test suite are named after the class under test.
 * File: Test__FP32Tensor.cpp → Testing: FP32Tensor class
 */

#include <gtest/gtest.h>
#include "tensors/Tensors.h"
#include "tensors/TensorFactory.h"
#include <memory>
#include <cmath>

using namespace llaminar2;

/**
 * @brief Test FP32 tensor creation and basic properties
 */
TEST(Test__FP32Tensor, FP32Creation)
{
    std::vector<size_t> shape = {2, 3}; // 2x3 matrix

    auto tensor = std::make_shared<FP32Tensor>(shape);

    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->shape().size(), 2);
    EXPECT_EQ(tensor->shape()[0], 2);
    EXPECT_EQ(tensor->shape()[1], 3);
    EXPECT_EQ(tensor->native_type(), TensorType::FP32);

    // Calculate expected sizes
    size_t expected_elements = 2 * 3;                          // 6 elements
    size_t expected_bytes = expected_elements * sizeof(float); // 24 bytes

    // V2 TensorBase doesn't expose size() directly, but we can verify via data pointer
    ASSERT_NE(tensor->data(), nullptr);
}

/**
 * @brief Test FP16 tensor creation and basic properties
 */
TEST(Test__FP32Tensor, FP16Creation)
{
    std::vector<size_t> shape = {4, 4};     // 4x4 matrix
    std::vector<uint16_t> data(16, 0x3C00); // FP16 value for 1.0

    auto tensor = std::make_shared<FP16Tensor>(shape, data);

    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->shape().size(), 2);
    EXPECT_EQ(tensor->shape()[0], 4);
    EXPECT_EQ(tensor->shape()[1], 4);
    EXPECT_EQ(tensor->native_type(), TensorType::FP16);

    ASSERT_NE(tensor->data(), nullptr);
}

/**
 * @brief Test BF16 tensor creation and basic properties
 */
TEST(Test__FP32Tensor, BF16Creation)
{
    std::vector<size_t> shape = {3, 5};     // 3x5 matrix
    std::vector<uint16_t> data(15, 0x3F80); // BF16 value for 1.0

    auto tensor = std::make_shared<BF16Tensor>(shape, data);

    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->shape().size(), 2);
    EXPECT_EQ(tensor->shape()[0], 3);
    EXPECT_EQ(tensor->shape()[1], 5);
    EXPECT_EQ(tensor->native_type(), TensorType::BF16);

    ASSERT_NE(tensor->data(), nullptr);
}

/**
 * @brief Test various tensor shapes (1D, 2D, 3D)
 */
TEST(Test__FP32Tensor, ShapeValidation)
{
    // 1D tensor (vector)
    auto tensor1d = std::make_shared<FP32Tensor>(std::vector<size_t>{10});
    EXPECT_EQ(tensor1d->shape().size(), 1);
    EXPECT_EQ(tensor1d->shape()[0], 10);

    // 2D tensor (matrix)
    auto tensor2d = std::make_shared<FP32Tensor>(std::vector<size_t>{5, 8});
    EXPECT_EQ(tensor2d->shape().size(), 2);
    EXPECT_EQ(tensor2d->shape()[0], 5);
    EXPECT_EQ(tensor2d->shape()[1], 8);

    // 3D tensor (batch of matrices)
    auto tensor3d = std::make_shared<FP32Tensor>(std::vector<size_t>{2, 3, 4});
    EXPECT_EQ(tensor3d->shape().size(), 3);
    EXPECT_EQ(tensor3d->shape()[0], 2);
    EXPECT_EQ(tensor3d->shape()[1], 3);
    EXPECT_EQ(tensor3d->shape()[2], 4);
}

/**
 * @brief Test device affinity defaults (CPU = device -1)
 */
TEST(Test__FP32Tensor, DeviceAffinity)
{
    auto tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{10, 10});

    // Default device should be host (CPU = -1)
    EXPECT_EQ(tensor->device_index(), -1);
    EXPECT_TRUE(tensor->is_on_device(-1));
}

/**
 * @brief Test FP32 GEMM correctness with small known matrix
 *
 * Tests C = A @ B^T where:
 * A = [[1, 2, 3],    B^T = [[1, 4],     C = [[22, 49],
 *      [4, 5, 6]]            [2, 5],          [28, 64]]
 *                            [3, 6]]
 */
TEST(Test__FP32Tensor, GemmCorrectnessTranspose)
{
    // Create activation matrix A [2, 3]
    std::vector<float> A_data = {
        1.0f, 2.0f, 3.0f, // Row 0
        4.0f, 5.0f, 6.0f  // Row 1
    };

    // Create weight matrix B stored in transposed layout [2, 3]
    // When transpose_B=true, this represents B^T where B is [3, 2]
    auto B_tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{2, 3});
    float *B_data = B_tensor->mutable_data();
    B_data[0] = 1.0f;
    B_data[1] = 2.0f;
    B_data[2] = 3.0f; // Column 0 of B
    B_data[3] = 4.0f;
    B_data[4] = 5.0f;
    B_data[5] = 6.0f; // Column 1 of B

    // Create output matrix C [2, 2]
    auto C_tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{2, 2});
    float *C_data = C_tensor->mutable_data();

    // Create GEMM kernel
    auto gemm = B_tensor->createGemm();
    ASSERT_NE(gemm, nullptr);

    // Execute: C = A @ B^T (with transpose_B=true)
    bool success = gemm->multiply(
        A_data.data(), // A [2, 3]
        C_data,        // C [2, 2]
        2,             // m = 2 rows
        2,             // n = 2 cols
        3,             // k = 3 (inner dimension)
        true,          // transpose_B = true
        1.0f,          // alpha = 1.0
        0.0f,          // beta = 0.0
        nullptr,       // no MPI
        -1             // CPU device
    );

    ASSERT_TRUE(success);

    // Expected results:
    // C[0,0] = 1*1 + 2*2 + 3*3 = 14
    // C[0,1] = 1*4 + 2*5 + 3*6 = 32
    // C[1,0] = 4*1 + 5*2 + 6*3 = 32
    // C[1,1] = 4*4 + 5*5 + 6*6 = 77
    EXPECT_NEAR(C_data[0], 14.0f, 1e-5f);
    EXPECT_NEAR(C_data[1], 32.0f, 1e-5f);
    EXPECT_NEAR(C_data[2], 32.0f, 1e-5f);
    EXPECT_NEAR(C_data[3], 77.0f, 1e-5f);
}

/**
 * @brief Test FP32 GEMM with alpha and beta parameters
 *
 * Tests C = alpha * A @ B^T + beta * C
 */
TEST(Test__FP32Tensor, GemmAlphaBeta)
{
    // Simple 2x2 matrices
    std::vector<float> A_data = {1.0f, 2.0f, 3.0f, 4.0f}; // [2, 2]

    auto B_tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{2, 2});
    float *B_data = B_tensor->mutable_data();
    B_data[0] = 1.0f;
    B_data[1] = 0.0f; // Identity matrix
    B_data[2] = 0.0f;
    B_data[3] = 1.0f;

    auto gemm = B_tensor->createGemm();

    // Initialize C with non-zero values
    auto C_tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{2, 2});
    float *C_data = C_tensor->mutable_data();
    C_data[0] = 10.0f;
    C_data[1] = 20.0f;
    C_data[2] = 30.0f;
    C_data[3] = 40.0f;

    // Execute: C = 2.0 * A @ I + 0.5 * C
    bool success = gemm->multiply(
        A_data.data(), C_data,
        2, 2, 2,
        true, 2.0f, 0.5f, // alpha=2.0, beta=0.5
        nullptr, -1);

    ASSERT_TRUE(success);

    // Expected: C = 2.0 * A + 0.5 * C_old
    // C[0,0] = 2.0*1.0 + 0.5*10.0 = 7.0
    // C[0,1] = 2.0*2.0 + 0.5*20.0 = 14.0
    // C[1,0] = 2.0*3.0 + 0.5*30.0 = 21.0
    // C[1,1] = 2.0*4.0 + 0.5*40.0 = 28.0
    EXPECT_NEAR(C_data[0], 7.0f, 1e-5f);
    EXPECT_NEAR(C_data[1], 14.0f, 1e-5f);
    EXPECT_NEAR(C_data[2], 21.0f, 1e-5f);
    EXPECT_NEAR(C_data[3], 28.0f, 1e-5f);
}

/**
 * @brief Test FP32 GEMM with non-transposed weight matrix
 */
TEST(Test__FP32Tensor, GemmNoTranspose)
{
    // A = [2, 3], B = [3, 2] (stored non-transposed)
    std::vector<float> A_data = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f};

    // B stored in row-major non-transposed layout [3, 2]
    auto B_tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{3, 2});
    float *B_data = B_tensor->mutable_data();
    B_data[0] = 1.0f;
    B_data[1] = 4.0f; // Row 0
    B_data[2] = 2.0f;
    B_data[3] = 5.0f; // Row 1
    B_data[4] = 3.0f;
    B_data[5] = 6.0f; // Row 2

    auto gemm = B_tensor->createGemm();

    auto C_tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{2, 2});
    float *C_data = C_tensor->mutable_data();

    // Execute with transpose_B=false
    bool success = gemm->multiply(
        A_data.data(), C_data,
        2, 2, 3,
        false, // transpose_B = false
        1.0f, 0.0f,
        nullptr, -1);

    ASSERT_TRUE(success);

    // Expected: Same result as transposed case
    // C[0,0] = 1*1 + 2*2 + 3*3 = 14
    // C[0,1] = 1*4 + 2*5 + 3*6 = 32
    // C[1,0] = 4*1 + 5*2 + 6*3 = 32
    // C[1,1] = 4*4 + 5*5 + 6*6 = 77
    EXPECT_NEAR(C_data[0], 14.0f, 1e-5f);
    EXPECT_NEAR(C_data[1], 32.0f, 1e-5f);
    EXPECT_NEAR(C_data[2], 32.0f, 1e-5f);
    EXPECT_NEAR(C_data[3], 77.0f, 1e-5f);
}

/**
 * @brief Test FP32 GEMM with larger matrix (stress test)
 */
TEST(Test__FP32Tensor, GemmLargerMatrix)
{
    const int m = 16, n = 32, k = 24;

    // Create random-like input (deterministic)
    std::vector<float> A_data(m * k);
    for (int i = 0; i < m * k; ++i)
    {
        A_data[i] = static_cast<float>((i * 7 + 3) % 100) / 10.0f;
    }

    auto B_tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(n), static_cast<size_t>(k)});
    float *B_data = B_tensor->mutable_data();
    for (int i = 0; i < n * k; ++i)
    {
        B_data[i] = static_cast<float>((i * 11 + 5) % 100) / 10.0f;
    }

    auto gemm = B_tensor->createGemm();

    auto C_tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(m), static_cast<size_t>(n)});
    float *C_data = C_tensor->mutable_data();

    bool success = gemm->multiply(
        A_data.data(), C_data,
        m, n, k,
        true, 1.0f, 0.0f,
        nullptr, -1);

    ASSERT_TRUE(success);

    // Spot check: verify at least some non-zero results
    bool has_nonzero = false;
    for (int i = 0; i < m * n; ++i)
    {
        if (std::abs(C_data[i]) > 1e-6f)
        {
            has_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero);

    // Verify no NaN or Inf values
    for (int i = 0; i < m * n; ++i)
    {
        EXPECT_FALSE(std::isnan(C_data[i]));
        EXPECT_FALSE(std::isinf(C_data[i]));
    }
}
