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

/**
 * @brief Test FP32 to INT8 block quantization
 *
 * Validates that FP32Tensor::to_int8_blocked() produces correct INT8
 * quantization with reasonable accuracy.
 */
TEST(Test__FP32Tensor, ToINT8BlockedConversion)
{
    const size_t rows = 8;
    const size_t cols = 256;
    const size_t block_size = 32;

    // Create FP32 tensor with known values
    auto tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{rows, cols});
    float *data = tensor->mutable_data();

    // Fill with predictable pattern (range: -10.0 to +10.0)
    for (size_t i = 0; i < rows * cols; ++i)
    {
        data[i] = -10.0f + (20.0f * i) / (rows * cols);
    }

    // Quantize to INT8 with block scales
    const size_t total_elements = rows * cols;
    const size_t num_blocks = (total_elements + block_size - 1) / block_size;
    std::vector<int8_t> int8_data(total_elements);
    std::vector<float> scales(num_blocks);

    tensor->to_int8_blocked(int8_data.data(), scales.data(), block_size);

    // Verify all int8 values are in valid range [-127, 127]
    for (size_t i = 0; i < total_elements; ++i)
    {
        EXPECT_GE(int8_data[i], -127);
        EXPECT_LE(int8_data[i], 127);
    }

    // Verify all scales are positive and reasonable
    for (size_t i = 0; i < num_blocks; ++i)
    {
        EXPECT_GT(scales[i], 0.0f) << "Scale at block " << i << " should be positive";
        EXPECT_LT(scales[i], 1e6f) << "Scale at block " << i << " should be reasonable";
    }

    // Dequantize and verify accuracy
    std::vector<float> dequantized(total_elements);
    for (size_t block_idx = 0; block_idx < num_blocks; ++block_idx)
    {
        const size_t offset = block_idx * block_size;
        const size_t count = std::min(block_size, total_elements - offset);
        const float scale = scales[block_idx];

        for (size_t i = 0; i < count; ++i)
        {
            dequantized[offset + i] = static_cast<float>(int8_data[offset + i]) * scale;
        }
    }

    // Calculate relative L2 error
    float sum_sq_diff = 0.0f;
    float sum_sq_orig = 0.0f;
    for (size_t i = 0; i < total_elements; ++i)
    {
        float diff = data[i] - dequantized[i];
        sum_sq_diff += diff * diff;
        sum_sq_orig += data[i] * data[i];
    }
    float rel_l2_error = std::sqrt(sum_sq_diff / (sum_sq_orig + 1e-10f));

    // INT8 quantization should have <5% relative error for this range
    EXPECT_LT(rel_l2_error, 0.05f) << "Relative L2 error too high: " << rel_l2_error;

    // Calculate max absolute difference
    float max_abs_diff = 0.0f;
    for (size_t i = 0; i < total_elements; ++i)
    {
        max_abs_diff = std::max(max_abs_diff, std::abs(data[i] - dequantized[i]));
    }

    // Max error should be reasonable (< 0.5 for range -10 to +10)
    EXPECT_LT(max_abs_diff, 0.5f) << "Max absolute difference too high: " << max_abs_diff;
}

/**
 * @brief Test to<float>() template method (FP32 conversion)
 */
TEST(Test__FP32Tensor, ToFloat_TemplateMethod)
{
    const size_t rows = 4;
    const size_t cols = 8;
    std::vector<size_t> shape = {rows, cols};

    // Create FP32 tensor with known values
    auto tensor = std::make_shared<FP32Tensor>(shape);
    float *data = tensor->mutable_data();
    for (size_t i = 0; i < rows * cols; ++i)
    {
        data[i] = static_cast<float>(i) * 0.5f - 5.0f; // Range: -5.0 to +10.5
    }

    // Convert using to<float>() template method
    std::vector<float> fp32_output(rows * cols);
    tensor->to<float>(fp32_output.data());

    // Verify exact match (FP32 -> FP32 should be identity)
    for (size_t i = 0; i < rows * cols; ++i)
    {
        EXPECT_FLOAT_EQ(fp32_output[i], data[i]) << "Mismatch at index " << i;
    }

    // Verify equivalence with legacy to_fp32() method
    std::vector<float> fp32_legacy(rows * cols);
    tensor->to_fp32(fp32_legacy.data());

    for (size_t i = 0; i < rows * cols; ++i)
    {
        EXPECT_FLOAT_EQ(fp32_output[i], fp32_legacy[i])
            << "to<float>() and to_fp32() differ at index " << i;
    }
}

/**
 * @brief Test to<uint16_t>() template method for BF16 conversion
 */
TEST(Test__FP32Tensor, ToBF16_TemplateMethod)
{
    const size_t rows = 4;
    const size_t cols = 8;
    std::vector<size_t> shape = {rows, cols};

    // Create FP32 tensor with known values
    auto tensor = std::make_shared<FP32Tensor>(shape);
    float *data = tensor->mutable_data();
    for (size_t i = 0; i < rows * cols; ++i)
    {
        data[i] = static_cast<float>(i) * 0.25f; // Range: 0.0 to 7.75
    }

    // Convert using to<uint16_t>() with BF16 format
    std::vector<uint16_t> bf16_output(rows * cols);
    tensor->to<uint16_t>(bf16_output.data(), TensorType::BF16);

    // Verify equivalence with legacy to_bf16() method
    std::vector<uint16_t> bf16_legacy(rows * cols);
    tensor->to_bf16(bf16_legacy.data());

    for (size_t i = 0; i < rows * cols; ++i)
    {
        EXPECT_EQ(bf16_output[i], bf16_legacy[i])
            << "to<uint16_t>(BF16) and to_bf16() differ at index " << i;
    }

    // Verify BF16 values are reasonable (not all zeros)
    bool has_nonzero = false;
    for (size_t i = 0; i < rows * cols; ++i)
    {
        if (bf16_output[i] != 0)
        {
            has_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero) << "BF16 output should contain non-zero values";
}

/**
 * @brief Test to<uint16_t>() template method for FP16 conversion
 */
TEST(Test__FP32Tensor, ToFP16_TemplateMethod)
{
    const size_t rows = 3;
    const size_t cols = 5;
    std::vector<size_t> shape = {rows, cols};

    // Create FP32 tensor with known values
    auto tensor = std::make_shared<FP32Tensor>(shape);
    float *data = tensor->mutable_data();
    for (size_t i = 0; i < rows * cols; ++i)
    {
        data[i] = static_cast<float>(i) * 0.1f; // Range: 0.0 to 1.4
    }

    // Convert using to<uint16_t>() with FP16 format
    std::vector<uint16_t> fp16_output(rows * cols);
    tensor->to<uint16_t>(fp16_output.data(), TensorType::FP16);

    // Verify equivalence with legacy to_fp16() method
    std::vector<uint16_t> fp16_legacy(rows * cols);
    tensor->to_fp16(fp16_legacy.data());

    for (size_t i = 0; i < rows * cols; ++i)
    {
        EXPECT_EQ(fp16_output[i], fp16_legacy[i])
            << "to<uint16_t>(FP16) and to_fp16() differ at index " << i;
    }
}

/**
 * @brief Test to<int8_t>() template method (INT8 blocked quantization)
 */
TEST(Test__FP32Tensor, ToINT8_TemplateMethod)
{
    const size_t rows = 8;
    const size_t cols = 16;
    std::vector<size_t> shape = {rows, cols};

    // Create FP32 tensor with known values
    auto tensor = std::make_shared<FP32Tensor>(shape);
    float *data = tensor->mutable_data();
    for (size_t i = 0; i < rows * cols; ++i)
    {
        data[i] = static_cast<float>(i % 64) - 32.0f; // Range: -32 to +31
    }

    // Convert using to<int8_t>() template method
    std::vector<int8_t> int8_output(rows * cols);
    tensor->to<int8_t>(int8_output.data());

    // Verify all values are in valid INT8 range
    for (size_t i = 0; i < rows * cols; ++i)
    {
        EXPECT_GE(int8_output[i], -127);
        EXPECT_LE(int8_output[i], 127);
    }

    // Verify at least some non-zero values (not all quantized to zero)
    bool has_nonzero = false;
    for (size_t i = 0; i < rows * cols; ++i)
    {
        if (int8_output[i] != 0)
        {
            has_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero) << "INT8 output should contain non-zero values";
}

/**
 * @brief Test to<int32_t>() template method (INT32 scaled conversion)
 */
TEST(Test__FP32Tensor, ToINT32_TemplateMethod)
{
    const size_t rows = 4;
    const size_t cols = 6;
    std::vector<size_t> shape = {rows, cols};

    // Create FP32 tensor with known values
    auto tensor = std::make_shared<FP32Tensor>(shape);
    float *data = tensor->mutable_data();
    for (size_t i = 0; i < rows * cols; ++i)
    {
        data[i] = static_cast<float>(i) * 0.01f - 0.1f; // Range: -0.1 to +0.13
    }

    // Convert using to<int32_t>() template method
    std::vector<int32_t> int32_output(rows * cols);
    tensor->to<int32_t>(int32_output.data());

    // Verify values are in INT32 range (no overflow)
    for (size_t i = 0; i < rows * cols; ++i)
    {
        // Values should be scaled to ~2^30 range
        EXPECT_NE(int32_output[i], 0) << "INT32 values should not all be zero";
    }

    // Verify at least some positive and negative values
    bool has_positive = false, has_negative = false;
    for (size_t i = 0; i < rows * cols; ++i)
    {
        if (int32_output[i] > 0)
            has_positive = true;
        if (int32_output[i] < 0)
            has_negative = true;
    }
    EXPECT_TRUE(has_positive) << "INT32 output should contain positive values";
    EXPECT_TRUE(has_negative) << "INT32 output should contain negative values";
}

/**
 * @brief Test round-trip conversion: FP32 -> BF16 -> FP32
 */
TEST(Test__FP32Tensor, RoundTripFP32_BF16_FP32)
{
    const size_t rows = 2;
    const size_t cols = 4;
    std::vector<size_t> shape = {rows, cols};

    // Create FP32 tensor with known values
    auto tensor_fp32 = std::make_shared<FP32Tensor>(shape);
    float *data = tensor_fp32->mutable_data();
    for (size_t i = 0; i < rows * cols; ++i)
    {
        data[i] = static_cast<float>(i) * 1.5f; // 0.0, 1.5, 3.0, 4.5, ...
    }

    // Convert FP32 -> BF16
    std::vector<uint16_t> bf16_data(rows * cols);
    tensor_fp32->to<uint16_t>(bf16_data.data(), TensorType::BF16);

    // Create BF16 tensor
    auto tensor_bf16 = std::make_shared<BF16Tensor>(shape, bf16_data);

    // Convert BF16 -> FP32
    std::vector<float> fp32_roundtrip(rows * cols);
    tensor_bf16->to<float>(fp32_roundtrip.data());

    // Verify round-trip accuracy (BF16 has ~3 decimal digits of precision)
    for (size_t i = 0; i < rows * cols; ++i)
    {
        float original = data[i];
        float roundtrip = fp32_roundtrip[i];
        float rel_error = std::abs(original - roundtrip) / (std::abs(original) + 1e-6f);

        // BF16 should preserve ~0.8% relative accuracy
        EXPECT_LT(rel_error, 0.01f)
            << "Round-trip error too high at index " << i
            << ": original=" << original << ", roundtrip=" << roundtrip;
    }
}
