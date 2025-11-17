/**
 * @file Test__INT32Tensor.cpp
 * @brief Unit tests for INT32Tensor class
 * @author David Sanftenberg
 * @date 2025-11-05
 *
 * This test suite validates the INT32Tensor class, which is a critical component
 * for full INT8 pipelines. INT32Tensor stores intermediate accumulator results
 * from INT8 GEMM operations and supports requantization back to INT8 for the
 * next layer.
 *
 * Test Coverage:
 * 1. Basic tensor creation and properties
 * 2. Constructor variants (from shape, INT32 data, FP32 data)
 * 3. INT32→INT8 requantization with per-row scaling
 * 4. INT32→FP32 dequantization
 * 5. Reconstruction accuracy after requant/dequant roundtrip
 * 6. Edge cases (zero values, extreme ranges, single row/column)
 * 7. Per-row vs per-tensor quantization comparison
 *
 * Naming convention: Test file and test suite are named after the class under test.
 * File: Test__INT32Tensor.cpp → Testing: INT32Tensor class
 */

#include <gtest/gtest.h>
#include "tensors/Tensors.h"
#include "tensors/TensorFactory.h"
#include <memory>
#include <cmath>
#include <vector>
#include <algorithm>
#include <random>

using namespace llaminar2;

// ============================================================================
// Test Fixture
// ============================================================================

class Test__INT32Tensor : public ::testing::Test
{
protected:
    // Random number generator
    std::mt19937 rng_{42}; // Fixed seed for reproducibility

    void SetUp() override
    {
        // Setup code if needed
    }

    // Helper: Compute relative L2 error
    float compute_relative_l2(const float *expected, const float *actual, size_t count)
    {
        float sum_sq_error = 0.0f;
        float sum_sq_expected = 0.0f;

        for (size_t i = 0; i < count; ++i)
        {
            float error = actual[i] - expected[i];
            sum_sq_error += error * error;
            sum_sq_expected += expected[i] * expected[i];
        }

        if (sum_sq_expected < 1e-12f)
        {
            return std::sqrt(sum_sq_error); // Absolute error if expected is ~0
        }

        return std::sqrt(sum_sq_error / sum_sq_expected);
    }

    // Helper: Compute max absolute error
    float compute_max_abs_error(const float *expected, const float *actual, size_t count)
    {
        float max_error = 0.0f;
        for (size_t i = 0; i < count; ++i)
        {
            max_error = std::max(max_error, std::fabs(actual[i] - expected[i]));
        }
        return max_error;
    }

    // Helper: Fill vector with random INT32 values
    void fill_random_int32(std::vector<int32_t> &data, int32_t min_val, int32_t max_val)
    {
        std::uniform_int_distribution<int32_t> dist(min_val, max_val);
        for (auto &val : data)
        {
            val = dist(rng_);
        }
    }

    // Helper: Fill vector with random FP32 values
    void fill_random_fp32(std::vector<float> &data, float min_val, float max_val)
    {
        std::uniform_real_distribution<float> dist(min_val, max_val);
        for (auto &val : data)
        {
            val = dist(rng_);
        }
    }
};

// ============================================================================
// Basic Creation and Properties Tests
// ============================================================================

/**
 * @brief Test INT32 tensor creation from shape
 */
TEST_F(Test__INT32Tensor, CreationFromShape)
{
    std::vector<size_t> shape = {4, 6}; // 4x6 matrix

    auto tensor = std::make_shared<INT32Tensor>(shape);

    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->shape().size(), 2);
    EXPECT_EQ(tensor->shape()[0], 4);
    EXPECT_EQ(tensor->shape()[1], 6);
    EXPECT_EQ(tensor->native_type(), TensorType::INT32);
    ASSERT_NE(tensor->int32_data(), nullptr);

    // Default scale should be 1.0
    EXPECT_FLOAT_EQ(tensor->scale(), 1.0f);
}

/**
 * @brief Test INT32 tensor creation from INT32 data
 */
TEST_F(Test__INT32Tensor, CreationFromINT32Data)
{
    std::vector<size_t> shape = {2, 3};
    std::vector<int32_t> data = {
        100, 200, 300,
        400, 500, 600};

    auto tensor = std::make_shared<INT32Tensor>(shape, data);

    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->shape()[0], 2);
    EXPECT_EQ(tensor->shape()[1], 3);
    // Default scale should be 1.0
    EXPECT_FLOAT_EQ(tensor->scale(), 1.0f);

    const int32_t *tensor_data = tensor->int32_data();
    ASSERT_NE(tensor_data, nullptr);

    // Verify data was copied correctly
    for (size_t i = 0; i < 6; ++i)
    {
        EXPECT_EQ(tensor_data[i], data[i]);
    }

    // Test set_scale
    tensor->set_scale(2.5f);
    EXPECT_FLOAT_EQ(tensor->scale(), 2.5f);
}

/**
 * @brief Test INT32 tensor creation from FP32 data with quantization
 */
TEST_F(Test__INT32Tensor, CreationFromFP32Data)
{
    std::vector<size_t> shape = {3, 4};
    std::vector<float> fp32_data = {
        1.5f, 2.5f, 3.5f, 4.5f,
        5.5f, 6.5f, 7.5f, 8.5f,
        9.5f, 10.5f, 11.5f, 12.5f};

    auto tensor = std::make_shared<INT32Tensor>(shape, fp32_data, 0.1f);

    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->shape()[0], 3);
    EXPECT_EQ(tensor->shape()[1], 4);
    EXPECT_FLOAT_EQ(tensor->scale(), 0.1f);

    const int32_t *tensor_data = tensor->int32_data();
    ASSERT_NE(tensor_data, nullptr);

    // Verify quantization: int32 = round(fp32 / scale)
    // scale = 0.1, so fp32=1.5 → int32=15
    EXPECT_EQ(tensor_data[0], 15);   // 1.5 / 0.1 = 15
    EXPECT_EQ(tensor_data[1], 25);   // 2.5 / 0.1 = 25
    EXPECT_EQ(tensor_data[11], 125); // 12.5 / 0.1 = 125
}

/**
 * @brief Verify INT32Tensor::from_int32_with_scales copies accumulators and persists row scales.
 */
TEST_F(Test__INT32Tensor, FromInt32WithScalesCopiesDataAndStoresRowScales)
{
    const int rows = 2;
    const int cols = 3;
    auto tensor = std::make_shared<INT32Tensor>(std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(cols)});

    const std::vector<int32_t> accum = {
        11, -22, 33,
        -44, 55, -66};
    const std::vector<float> row_scales = {0.25f, 0.5f};
    const std::vector<float> col_scales = {10.0f, -3.0f, 0.25f}; // Should be ignored for INT32 tensors
    const std::vector<float> bias = {5.0f, -7.0f, 9.0f};         // Should be ignored for INT32 tensors

    ASSERT_TRUE(tensor->from_int32_with_scales(
        accum.data(),
        rows,
        cols,
        row_scales.data(),
        col_scales.data(),
        bias.data()));

    const int32_t *int_data = tensor->int32_data();
    ASSERT_NE(int_data, nullptr);
    for (size_t i = 0; i < accum.size(); ++i)
    {
        EXPECT_EQ(int_data[i], accum[i]) << "Accumulator mismatch at index " << i;
    }

    ASSERT_TRUE(tensor->has_row_scales());
    EXPECT_EQ(tensor->num_row_scales(), row_scales.size());
    const float *stored_scales = tensor->row_scales();
    ASSERT_NE(stored_scales, nullptr);
    for (size_t i = 0; i < row_scales.size(); ++i)
    {
        EXPECT_FLOAT_EQ(stored_scales[i], row_scales[i]) << "Row scale mismatch at index " << i;
    }

    // A second call without row scales should clear the cache
    ASSERT_TRUE(tensor->from_int32_with_scales(
        accum.data(),
        rows,
        cols,
        nullptr,
        nullptr,
        nullptr));
    EXPECT_FALSE(tensor->has_row_scales());
}

/**
 * @brief Test various tensor shapes (1D, 2D, 3D)
 */
TEST_F(Test__INT32Tensor, ShapeValidation)
{
    // 1D tensor (vector)
    auto tensor1d = std::make_shared<INT32Tensor>(std::vector<size_t>{10});
    EXPECT_EQ(tensor1d->shape().size(), 1);
    EXPECT_EQ(tensor1d->shape()[0], 10);

    // 2D tensor (matrix)
    auto tensor2d = std::make_shared<INT32Tensor>(std::vector<size_t>{5, 8});
    EXPECT_EQ(tensor2d->shape().size(), 2);
    EXPECT_EQ(tensor2d->shape()[0], 5);
    EXPECT_EQ(tensor2d->shape()[1], 8);

    // 3D tensor (batch of matrices)
    auto tensor3d = std::make_shared<INT32Tensor>(std::vector<size_t>{2, 3, 4});
    EXPECT_EQ(tensor3d->shape().size(), 3);
    EXPECT_EQ(tensor3d->shape()[0], 2);
    EXPECT_EQ(tensor3d->shape()[1], 3);
    EXPECT_EQ(tensor3d->shape()[2], 4);
}

/**
 * @brief Test device affinity defaults (CPU = device -1)
 */
TEST_F(Test__INT32Tensor, DeviceAffinity)
{
    auto tensor = std::make_shared<INT32Tensor>(std::vector<size_t>{10, 10});

    // Default device should be host (CPU = -1)
    EXPECT_EQ(tensor->device_index(), -1);
    EXPECT_TRUE(tensor->is_on_device(-1));
}

// ============================================================================
// INT32→INT8 Requantization Tests
// ============================================================================

/**
 * @brief Test basic INT32→INT8 requantization with per-row scaling
 */
TEST_F(Test__INT32Tensor, RequantizeToINT8Basic)
{
    // Create tensor with different dynamic ranges per row
    std::vector<size_t> shape = {3, 4};
    std::vector<int32_t> data = {
        127, 64, 32, 16,     // Row 0: max = 127
        1000, 500, 250, 125, // Row 1: max = 1000
        50, 25, 12, 6        // Row 2: max = 50
    };

    auto int32_tensor = std::make_shared<INT32Tensor>(shape, data);

    // Requantize to INT8
    std::vector<int8_t> int8_data(12);
    std::vector<float> row_scales(3);
    bool success = int32_tensor->requantize_to_int8(int8_data.data(), row_scales.data());

    ASSERT_TRUE(success) << "Requantization should succeed";

    // Row 0: scale = 127/127 = 1.0, so values should be unchanged
    EXPECT_EQ(int8_data[0], 127);
    EXPECT_EQ(int8_data[1], 64);

    // Row 1: scale = 1000/127 ≈ 7.87, so 1000→127, 500→63, etc.
    EXPECT_EQ(int8_data[4], 127);     // 1000 / 7.87 ≈ 127
    EXPECT_NEAR(int8_data[5], 63, 1); // 500 / 7.87 ≈ 63

    // Row 2: scale = 50/127 ≈ 0.39, so 50→127, 25→63, etc.
    EXPECT_EQ(int8_data[8], 127); // Max value always maps to 127
}

/**
 * @brief Test INT32→INT8 requantization preserves sign
 */
TEST_F(Test__INT32Tensor, RequantizePreservesSign)
{
    std::vector<size_t> shape = {2, 4};
    std::vector<int32_t> data = {
        1000, -500, 250, -125,   // Row 0: mixed signs
        -2000, -1000, -500, -250 // Row 1: all negative
    };

    auto int32_tensor = std::make_shared<INT32Tensor>(shape, data);
    std::vector<int8_t> int8_data(8);
    std::vector<float> row_scales(2);
    bool success = int32_tensor->requantize_to_int8(int8_data.data(), row_scales.data());

    ASSERT_TRUE(success);

    // Row 0: scale = 1000/127 ≈ 7.87
    EXPECT_GT(int8_data[0], 0); // 1000 → positive
    EXPECT_LT(int8_data[1], 0); // -500 → negative
    EXPECT_GT(int8_data[2], 0); // 250 → positive
    EXPECT_LT(int8_data[3], 0); // -125 → negative

    // Row 1: all negative
    EXPECT_LT(int8_data[4], 0); // -2000 → -127
    EXPECT_LT(int8_data[5], 0); // -1000 → negative
    EXPECT_LT(int8_data[6], 0); // -500 → negative
    EXPECT_LT(int8_data[7], 0); // -250 → negative
}

/**
 * @brief Test INT32→INT8 requantization with extreme values
 */
TEST_F(Test__INT32Tensor, RequantizeExtremeValues)
{
    std::vector<size_t> shape = {4, 1}; // Single column, 4 rows
    std::vector<int32_t> data = {
        60000,  // Row 0: very large
        -50000, // Row 1: very large negative
        10,     // Row 2: small positive
        -5      // Row 3: small negative
    };

    auto int32_tensor = std::make_shared<INT32Tensor>(shape, data);
    std::vector<int8_t> int8_data(4);
    std::vector<float> row_scales(4);
    bool success = int32_tensor->requantize_to_int8(int8_data.data(), row_scales.data());

    ASSERT_TRUE(success);

    // Each row has one value, so it should map to ±127
    EXPECT_EQ(int8_data[0], 127);  // Max positive
    EXPECT_EQ(int8_data[1], -127); // Max negative
    EXPECT_EQ(int8_data[2], 127);  // Only value in row
    EXPECT_EQ(int8_data[3], -127); // Only value in row
}

/**
 * @brief Test INT32→INT8 requantization accuracy via reconstruction
 */
TEST_F(Test__INT32Tensor, RequantizeReconstructionAccuracy)
{
    std::vector<size_t> shape = {4, 8};
    std::vector<int32_t> original_data(32);
    fill_random_int32(original_data, -5000, 5000);

    auto int32_tensor = std::make_shared<INT32Tensor>(shape, original_data);

    // Requantize to INT8
    std::vector<int8_t> int8_data(32);
    std::vector<float> row_scales(4);
    bool success = int32_tensor->requantize_to_int8(int8_data.data(), row_scales.data());
    ASSERT_TRUE(success);

    // Reconstruct FP32 from INT8 using the row scales
    std::vector<float> reconstructed(32);
    for (size_t i = 0; i < 4; ++i) // For each row
    {
        for (size_t j = 0; j < 8; ++j) // For each column
        {
            reconstructed[i * 8 + j] = static_cast<float>(int8_data[i * 8 + j]) * row_scales[i];
        }
    }

    // Compute relative error
    std::vector<float> original_fp32(32);
    for (size_t i = 0; i < 32; ++i)
    {
        original_fp32[i] = static_cast<float>(original_data[i]);
    }

    float rel_error = compute_relative_l2(original_fp32.data(), reconstructed.data(), 32);

    // INT32→INT8→FP32 should have <2% relative error
    EXPECT_LT(rel_error, 0.02f) << "Reconstruction error should be <2%";
}

// ============================================================================
// INT32→FP32 Dequantization Tests
// ============================================================================

/**
 * @brief Test basic INT32→FP32 dequantization
 */
TEST_F(Test__INT32Tensor, DequantizeToFP32Basic)
{
    std::vector<size_t> shape = {2, 3};
    std::vector<int32_t> data = {100, 200, 300, 400, 500, 600};

    auto int32_tensor = std::make_shared<INT32Tensor>(shape, data);
    int32_tensor->set_scale(0.5f);

    // Allocate FP32 buffer
    std::vector<float> fp32_data(6);
    int32_tensor->to_fp32(fp32_data.data());

    // Dequantization: fp32 = int32 * scale
    EXPECT_FLOAT_EQ(fp32_data[0], 100 * 0.5f); // 50.0
    EXPECT_FLOAT_EQ(fp32_data[1], 200 * 0.5f); // 100.0
    EXPECT_FLOAT_EQ(fp32_data[5], 600 * 0.5f); // 300.0
}

/**
 * @brief Test INT32→FP32 dequantization with negative values
 */
TEST_F(Test__INT32Tensor, DequantizeNegativeValues)
{
    std::vector<size_t> shape = {3, 2};
    std::vector<int32_t> data = {-100, 200, -300, 400, -500, 600};

    auto int32_tensor = std::make_shared<INT32Tensor>(shape, data);
    int32_tensor->set_scale(2.0f);

    std::vector<float> fp32_data(6);
    int32_tensor->to_fp32(fp32_data.data());

    EXPECT_FLOAT_EQ(fp32_data[0], -100 * 2.0f); // -200.0
    EXPECT_FLOAT_EQ(fp32_data[1], 200 * 2.0f);  // 400.0
    EXPECT_FLOAT_EQ(fp32_data[2], -300 * 2.0f); // -600.0
    EXPECT_FLOAT_EQ(fp32_data[5], 600 * 2.0f);  // 1200.0
}

/**
 * @brief Test INT32→FP32 roundtrip (FP32→INT32→FP32)
 */
TEST_F(Test__INT32Tensor, RoundtripFP32ToINT32ToFP32)
{
    std::vector<size_t> shape = {4, 5};
    std::vector<float> original_data(20);
    fill_random_fp32(original_data, -1000.0f, 1000.0f);

    float scale = 0.1f;

    // FP32 → INT32
    auto int32_tensor = std::make_shared<INT32Tensor>(shape, original_data, scale);

    // INT32 → FP32
    std::vector<float> fp32_data(20);
    int32_tensor->to_fp32(fp32_data.data());

    // Compute error (should be small due to rounding in quantization)
    float max_error = compute_max_abs_error(original_data.data(), fp32_data.data(), 20);

    // Max error should be ≤ scale/2 (quantization rounding error)
    EXPECT_LE(max_error, scale / 2.0f + 1e-5f);
}

// ============================================================================
// Edge Cases and Robustness Tests
// ============================================================================

/**
 * @brief Test requantization with all-zero row
 */
TEST_F(Test__INT32Tensor, RequantizeZeroRow)
{
    std::vector<size_t> shape = {3, 4};
    std::vector<int32_t> data = {
        100, 200, 300, 400,    // Row 0: normal values
        0, 0, 0, 0,            // Row 1: all zeros
        -100, -200, -300, -400 // Row 2: normal negative
    };

    auto int32_tensor = std::make_shared<INT32Tensor>(shape, data);
    std::vector<int8_t> int8_data(12);
    std::vector<float> row_scales(3);
    bool success = int32_tensor->requantize_to_int8(int8_data.data(), row_scales.data());

    ASSERT_TRUE(success);

    // Row 1 (all zeros) should remain zeros
    EXPECT_EQ(int8_data[4], 0);
    EXPECT_EQ(int8_data[5], 0);
    EXPECT_EQ(int8_data[6], 0);
    EXPECT_EQ(int8_data[7], 0);
}

/**
 * @brief Test requantization with single-element rows
 */
TEST_F(Test__INT32Tensor, RequantizeSingleColumn)
{
    std::vector<size_t> shape = {5, 1}; // Single column
    std::vector<int32_t> data = {1000, -2000, 500, -100, 50};

    auto int32_tensor = std::make_shared<INT32Tensor>(shape, data);
    std::vector<int8_t> int8_data(5);
    std::vector<float> row_scales(5);
    bool success = int32_tensor->requantize_to_int8(int8_data.data(), row_scales.data());

    ASSERT_TRUE(success);

    // Each row has one value, so it should map to ±127
    EXPECT_EQ(int8_data[0], 127);  // 1000 → 127
    EXPECT_EQ(int8_data[1], -127); // -2000 → -127
    EXPECT_EQ(int8_data[2], 127);  // 500 → 127
    EXPECT_EQ(int8_data[3], -127); // -100 → -127
    EXPECT_EQ(int8_data[4], 127);  // 50 → 127
}

/**
 * @brief Test requantization with single row (vector)
 */
TEST_F(Test__INT32Tensor, RequantizeSingleRow)
{
    std::vector<size_t> shape = {1, 8}; // Single row
    std::vector<int32_t> data = {127, 64, 32, 16, -16, -32, -64, -127};

    auto int32_tensor = std::make_shared<INT32Tensor>(shape, data);
    std::vector<int8_t> int8_data(8);
    std::vector<float> row_scales(1);
    bool success = int32_tensor->requantize_to_int8(int8_data.data(), row_scales.data());

    ASSERT_TRUE(success);

    // Max abs value is 127, so scale = 1.0, values unchanged
    EXPECT_EQ(int8_data[0], 127);
    EXPECT_EQ(int8_data[1], 64);
    EXPECT_EQ(int8_data[7], -127);
}

/**
 * @brief Test scale factor is correctly propagated
 */
TEST_F(Test__INT32Tensor, ScaleFactorPropagation)
{
    std::vector<size_t> shape = {2, 2};
    std::vector<int32_t> data = {1000, 2000, 3000, 4000};

    auto int32_tensor = std::make_shared<INT32Tensor>(shape, data);
    int32_tensor->set_scale(3.14159f);

    EXPECT_FLOAT_EQ(int32_tensor->scale(), 3.14159f);

    // Dequantize and verify scale is applied
    std::vector<float> fp32_data(4);
    int32_tensor->to_fp32(fp32_data.data());

    EXPECT_FLOAT_EQ(fp32_data[0], 1000.0f * 3.14159f);
    EXPECT_FLOAT_EQ(fp32_data[3], 4000.0f * 3.14159f);
}

/**
 * @brief Test large tensor (stress test)
 */
TEST_F(Test__INT32Tensor, LargeTensorStressTest)
{
    std::vector<size_t> shape = {128, 256}; // ~32K elements
    std::vector<int32_t> data(128 * 256);
    fill_random_int32(data, -10000, 10000);

    auto int32_tensor = std::make_shared<INT32Tensor>(shape, data);

    // Requantize
    std::vector<int8_t> int8_data(128 * 256);
    std::vector<float> row_scales(128);
    bool success = int32_tensor->requantize_to_int8(int8_data.data(), row_scales.data());
    ASSERT_TRUE(success);

    // Dequantize
    std::vector<float> fp32_data(128 * 256);
    int32_tensor->to_fp32(fp32_data.data());

    // Both operations should succeed without crashes
    SUCCEED();
}

// ============================================================================
// Per-Row vs Per-Tensor Quantization Comparison
// ============================================================================

/**
 * @brief Demonstrate why per-row quantization is better than per-tensor
 *
 * This test shows that per-row quantization maintains better precision
 * when different rows have vastly different dynamic ranges.
 */
TEST_F(Test__INT32Tensor, PerRowVsPerTensorComparison)
{
    // Create tensor with vastly different row ranges
    std::vector<size_t> shape = {3, 4};
    std::vector<int32_t> data = {
        10, 20, 30, 40,             // Row 0: small values (max=40)
        10000, 20000, 30000, 40000, // Row 1: large values (max=40000)
        100, 200, 300, 400          // Row 2: medium values (max=400)
    };

    auto int32_tensor = std::make_shared<INT32Tensor>(shape, data);

    // Per-row quantization (INT32Tensor default)
    std::vector<int8_t> int8_data(12);
    std::vector<float> row_scales(3);
    bool success = int32_tensor->requantize_to_int8(int8_data.data(), row_scales.data());
    ASSERT_TRUE(success);

    // Row 0: scale ≈ 40/127 ≈ 0.31, good precision
    // Row 1: scale ≈ 40000/127 ≈ 315, large scale
    // Row 2: scale ≈ 400/127 ≈ 3.15, medium scale

    // Each row's max value should map to 127
    EXPECT_EQ(int8_data[3], 127);  // Row 0: 40 → 127
    EXPECT_EQ(int8_data[7], 127);  // Row 1: 40000 → 127
    EXPECT_EQ(int8_data[11], 127); // Row 2: 400 → 127

    // Row 0 small values maintain precision (10 → ~32)
    EXPECT_GE(int8_data[0], 30);
    EXPECT_LE(int8_data[0], 35);

    // If we used per-tensor quantization, Row 0 would be crushed:
    // Per-tensor scale = 40000/127 ≈ 315
    // Row 0: 10/315 ≈ 0.03 → 0 (precision lost!)
    // This test demonstrates the superiority of per-row quantization.
}

// ============================================================================
// Main Test Entry Point
// ============================================================================

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
