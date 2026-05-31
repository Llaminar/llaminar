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
TEST(Test__FP32Tensor, BasicCreation)
{
    FP32Tensor tensor({2, 3});
    EXPECT_EQ(tensor.shape().size(), 2);
    EXPECT_EQ(tensor.shape()[0], 2);
    EXPECT_EQ(tensor.shape()[1], 3);
}

TEST(Test__FP32Tensor, DataInitialization)
{
    FP32Tensor tensor({2, 2});
    float *data = tensor.mutable_data();
    data[0] = 1.0f;
    data[1] = 2.0f;
    data[2] = 3.0f;
    data[3] = 4.0f;

    // Simple sanity check that values round-trip through the tensor buffer.
    EXPECT_FLOAT_EQ(data[0], 1.0f);
    EXPECT_FLOAT_EQ(data[1], 2.0f);
    EXPECT_FLOAT_EQ(data[2], 3.0f);
    EXPECT_FLOAT_EQ(data[3], 4.0f);
}

TEST(Test__FP32Tensor, MarkHostDirtyRestoresHostAuthorityAfterDirectWrite)
{
    FP32Tensor tensor({1, 2});
    float *data = static_cast<float *>(tensor.raw_mutable_data());
    data[0] = 1.0f;
    data[1] = 2.0f;

    tensor.transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE, DeviceId::cuda(0));
    EXPECT_FALSE(tensor.hostValid());
    EXPECT_TRUE(tensor.needsDownload());
    EXPECT_TRUE(tensor.getAuthoritativeDevice().has_value());

    data = static_cast<float *>(tensor.raw_mutable_data());
    data[0] = 3.0f;
    data[1] = 4.0f;
    tensor.mark_host_dirty();

    EXPECT_TRUE(tensor.hostValid());
    EXPECT_FALSE(tensor.needsDownload());
    EXPECT_FALSE(tensor.getAuthoritativeDevice().has_value());
    EXPECT_FLOAT_EQ(tensor.data()[0], 3.0f);
    EXPECT_FLOAT_EQ(tensor.data()[1], 4.0f);
}
// NOTE: Legacy GEMM-related checks removed; FP32Tensor no longer owns GEMM tests.

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
    // NOTE: This behavior is now covered by dedicated BF16 tensor tests.
}

/**
 * @brief Validate that FP32Tensor::from_int32_with_scales applies row/column scales and bias.
 */
TEST(Test__FP32Tensor, FromInt32WithScalesAppliesAllFactors)
{
    const int rows = 2;
    const int cols = 3;
    auto tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(cols)});

    const std::vector<int32_t> accum = {
        10, -20, 30,
        -40, 50, -60};

    const std::vector<float> row_scales = {0.1f, 0.2f};
    const std::vector<float> col_scales = {1.0f, 2.0f, 0.5f};
    const std::vector<float> bias = {0.5f, -1.5f, 0.0f};

    bool ok = tensor->from_int32_with_scales(
        accum.data(),
        rows,
        cols,
        row_scales.data(),
        col_scales.data(),
        bias.data());

    ASSERT_TRUE(ok);

    const float *data = tensor->data();
    const std::vector<float> expected = {
        1.5f,  // 10 * 0.1 * 1.0 + 0.5
        -5.5f, // -20 * 0.1 * 2.0 - 1.5
        1.5f,  // 30 * 0.1 * 0.5 + 0.0
        -7.5f, // -40 * 0.2 * 1.0 + 0.5
        18.5f, // 50 * 0.2 * 2.0 - 1.5
        -6.0f  // -60 * 0.2 * 0.5 + 0.0
    };

    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_FLOAT_EQ(data[i], expected[i]) << "Mismatch at index " << i;
    }
}

/**
 * @brief Ensure FP32Tensor::from_int32_with_scales gracefully handles nullptr scale/bias arrays.
 */
TEST(Test__FP32Tensor, FromInt32WithScalesDefaultsWhenPointersNull)
{
    const int rows = 2;
    const int cols = 2;
    auto tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(cols)});

    const std::vector<int32_t> accum = {
        3, -6,
        9, -12};

    ASSERT_TRUE(tensor->from_int32_with_scales(
        accum.data(),
        rows,
        cols,
        nullptr,
        nullptr,
        nullptr));

    const float *data = tensor->data();
    for (size_t i = 0; i < accum.size(); ++i)
    {
        EXPECT_FLOAT_EQ(data[i], static_cast<float>(accum[i])) << "Index " << i << " should remain unchanged";
    }
}

// ============================================================================
// quantize_to_q8_1 Tests
// ============================================================================

TEST(Test__FP32Tensor, QuantizeToQ8_1_BasicRoundTrip)
{
    // Create a small FP32 tensor with known values
    auto tensor = std::make_unique<llaminar2::FP32Tensor>(std::vector<size_t>{2, 64});
    float *data = tensor->mutable_data();

    // Initialize with varied values across dynamic range
    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 64; ++j)
        {
            // Mix of positive and negative values
            data[i * 64 + j] = (j % 2 == 0) ? (j * 0.5f) : (-j * 0.3f);
        }
    }

    // Allocate Q8_1 buffer using static helper
    const int m = 2, k = 64;
    size_t buffer_size = llaminar2::IActivationTensor::get_q8_1_buffer_size(m, k);
    std::vector<uint8_t> q8_buffer(buffer_size);

    // Quantize
    bool success = tensor->quantize_to_q8_1(q8_buffer.data(), m, k);
    EXPECT_TRUE(success);

    // Verify blocks are valid (scale should be non-zero, sum should be computed)
    const llaminar2::Q8_1Block *blocks = reinterpret_cast<const llaminar2::Q8_1Block *>(q8_buffer.data());
    int k_blocks = (k + 31) / 32;

    for (int i = 0; i < m; ++i)
    {
        for (int kb = 0; kb < k_blocks; ++kb)
        {
            const auto &block = blocks[i * k_blocks + kb];

            // Scale should be positive
            float scale = llaminar2::fp16_to_fp32(block.d);
            EXPECT_GT(scale, 0.0f) << "Block [" << i << "," << kb << "] has zero scale";

            // Verify sum_qs matches actual sum
            int32_t actual_sum = 0;
            for (int j = 0; j < 32; ++j)
            {
                actual_sum += block.qs[j];
            }
            EXPECT_EQ(block.sum_qs, static_cast<int16_t>(actual_sum))
                << "Block [" << i << "," << kb << "] sum mismatch";
        }
    }
}

TEST(Test__FP32Tensor, QuantizeToQ8_1_DequantAccuracy)
{
    // Test that quantize -> dequant has acceptable error
    // Use dimensions that are multiples of 32 to avoid padding edge cases
    auto tensor = std::make_unique<llaminar2::FP32Tensor>(std::vector<size_t>{4, 128});

    // Initialize with realistic activation-like values
    // Use data() consistently since quantize_to_q8_1 uses data()
    for (size_t i = 0; i < 4 * 128; ++i)
    {
        tensor->mutable_data()[i] = ((static_cast<int>(i) % 100) - 50) * 0.1f; // Range: -5 to +5
    }

    // Now use const access
    const float *data = tensor->data();

    // Verify initial data is set
    ASSERT_NE(data, nullptr);
    EXPECT_FLOAT_EQ(data[0], -5.0f); // First value: ((0 % 100) - 50) * 0.1 = -5
    EXPECT_FLOAT_EQ(data[1], -4.9f); // Second value: ((1 % 100) - 50) * 0.1 = -4.9

    const int m = 4, k = 128;
    size_t buffer_size = llaminar2::IActivationTensor::get_q8_1_buffer_size(m, k);
    std::vector<uint8_t> q8_buffer(buffer_size);

    ASSERT_TRUE(tensor->quantize_to_q8_1(q8_buffer.data(), m, k));

    // Dequantize and check max error
    const llaminar2::Q8_1Block *blocks = reinterpret_cast<const llaminar2::Q8_1Block *>(q8_buffer.data());
    int k_blocks = (k + 31) / 32;

    // First, check the raw block values
    const auto &first_block = blocks[0];
    std::cout << "First block: d_bits=0x" << std::hex << first_block.d << std::dec
              << " d_float=" << llaminar2::fp16_to_fp32(first_block.d)
              << " sum_qs=" << first_block.sum_qs
              << " qs[0]=" << (int)first_block.qs[0]
              << std::endl;

    float max_error = 0.0f;
    for (int i = 0; i < m; ++i)
    {
        for (int kb = 0; kb < k_blocks; ++kb)
        {
            const auto &block = blocks[i * k_blocks + kb];
            float scale = llaminar2::fp16_to_fp32(block.d);

            // Skip if scale is somehow invalid
            if (!std::isfinite(scale) || scale <= 0.0f)
            {
                FAIL() << "Invalid scale at block [" << i << "," << kb << "]: " << scale
                       << " d_bits=0x" << std::hex << block.d << std::dec;
            }

            for (int j = 0; j < 32; ++j)
            {
                int col_idx = kb * 32 + j;
                if (col_idx >= k)
                    continue;

                float original = data[i * k + col_idx];
                float dequant = static_cast<float>(block.qs[j]) * scale;
                float error = std::abs(original - dequant);
                if (!std::isfinite(error))
                {
                    FAIL() << "Non-finite error at [" << i << "," << kb << "," << j
                           << "]: original=" << original << ", dequant=" << dequant
                           << ", scale=" << scale << ", qs=" << (int)block.qs[j];
                }
                max_error = std::max(max_error, error);
            }
        }
    }

    // Q8_1 should have ~1% max error for typical activation ranges
    EXPECT_LT(max_error, 0.1f) << "Max quantization error too high: " << max_error;
}

TEST(Test__FP32Tensor, QuantizeToQ8_1_InvalidParams)
{
    auto tensor = std::make_unique<llaminar2::FP32Tensor>(std::vector<size_t>{4, 64});
    std::vector<uint8_t> buffer(1024);

    // Null buffer
    EXPECT_FALSE(tensor->quantize_to_q8_1(nullptr, 4, 64));

    // Invalid dimensions
    EXPECT_FALSE(tensor->quantize_to_q8_1(buffer.data(), 0, 64));
    EXPECT_FALSE(tensor->quantize_to_q8_1(buffer.data(), 4, 0));
    EXPECT_FALSE(tensor->quantize_to_q8_1(buffer.data(), -1, 64));

    // Dimensions exceeding tensor shape
    EXPECT_FALSE(tensor->quantize_to_q8_1(buffer.data(), 10, 64)); // m > rows
    EXPECT_FALSE(tensor->quantize_to_q8_1(buffer.data(), 4, 128)); // k > cols
}
