/**
 * @file Test__TensorConversion.cpp
 * @brief Conversion correctness tests for TensorBase interface
 * @author David Sanftenberg
 *
 * Tests the tensor format conversion interface across all tensor types:
 * - Round-trip conversions (FP32 ↔ BF16 ↔ FP16)
 * - INT8 block quantization accuracy
 * - Row and span conversion consistency
 * - Cross-format conversion equivalence
 *
 * Validates that conversion methods maintain expected precision and
 * produce consistent results across different conversion paths.
 */

#include <gtest/gtest.h>
#include "tensors/Tensors.h"
#include "tensors/TensorFactory.h"
#include "utils/BFloat16.h"
#include <memory>
#include <cmath>
#include <random>
#include <algorithm>

using namespace llaminar2;

namespace
{

    /**
     * @brief Helper to generate random FP32 data
     */
    std::vector<float> generateRandomFP32(size_t count, float min = -10.0f, float max = 10.0f)
    {
        std::vector<float> data(count);
        std::random_device rd;
        std::mt19937 gen(42); // Fixed seed for reproducibility
        std::uniform_real_distribution<float> dist(min, max);

        for (size_t i = 0; i < count; ++i)
        {
            data[i] = dist(gen);
        }
        return data;
    }

    /**
     * @brief Calculate relative L2 error between two arrays
     */
    float relativeL2Error(const float *a, const float *b, size_t count)
    {
        float sum_sq_diff = 0.0f;
        float sum_sq_ref = 0.0f;

        for (size_t i = 0; i < count; ++i)
        {
            float diff = a[i] - b[i];
            sum_sq_diff += diff * diff;
            sum_sq_ref += a[i] * a[i];
        }

        if (sum_sq_ref < 1e-10f)
        {
            return 0.0f; // Both near zero
        }

        return std::sqrt(sum_sq_diff / sum_sq_ref);
    }

    /**
     * @brief Calculate max absolute difference
     */
    float maxAbsDiff(const float *a, const float *b, size_t count)
    {
        float max_diff = 0.0f;
        for (size_t i = 0; i < count; ++i)
        {
            float diff = std::abs(a[i] - b[i]);
            max_diff = std::max(max_diff, diff);
        }
        return max_diff;
    }

} // anonymous namespace

/**
 * @brief Test FP32 → FP32 conversion (identity)
 */
TEST(Test__TensorConversion, FP32ToFP32Identity)
{
    const size_t rows = 8;
    const size_t cols = 256;
    auto original_data = generateRandomFP32(rows * cols);

    // Create FP32 tensor
    auto tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{rows, cols});
    std::memcpy(tensor->mutable_data(), original_data.data(), original_data.size() * sizeof(float));

    // Convert to FP32 (should be identity)
    std::vector<float> converted(rows * cols);
    tensor->to_fp32(converted.data());

    // Should be exact match
    for (size_t i = 0; i < rows * cols; ++i)
    {
        EXPECT_FLOAT_EQ(converted[i], original_data[i]);
    }
}

/**
 * @brief Test FP32 → BF16 → FP32 round-trip conversion
 *
 * BF16 has 8 exponent bits (same as FP32) but only 7 mantissa bits
 * (vs 23 for FP32). Expected precision loss: ~1e-3 relative error.
 */
TEST(Test__TensorConversion, FP32ToBF16RoundTrip)
{
    const size_t rows = 8;
    const size_t cols = 256;
    auto original_data = generateRandomFP32(rows * cols, -10.0f, 10.0f);

    // Create FP32 tensor
    auto fp32_tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{rows, cols});
    std::memcpy(fp32_tensor->mutable_data(), original_data.data(), original_data.size() * sizeof(float));

    // Convert FP32 → BF16
    std::vector<uint16_t> bf16_data(rows * cols);
    fp32_tensor->to_bf16(bf16_data.data());

    // Create BF16 tensor
    auto bf16_tensor = std::make_shared<BF16Tensor>(std::vector<size_t>{rows, cols}, bf16_data);

    // Convert BF16 → FP32
    std::vector<float> recovered(rows * cols);
    bf16_tensor->to_fp32(recovered.data());

    // Check precision loss is within expected bounds
    float rel_error = relativeL2Error(original_data.data(), recovered.data(), rows * cols);
    float max_diff = maxAbsDiff(original_data.data(), recovered.data(), rows * cols);

    EXPECT_LT(rel_error, 1e-2) << "Relative L2 error too high for BF16 round-trip";
    EXPECT_LT(max_diff, 0.1f) << "Max absolute difference too high for BF16 round-trip";
}

/**
 * @brief Test BF16 → BF16 conversion (identity)
 */
TEST(Test__TensorConversion, BF16ToBF16Identity)
{
    const size_t rows = 8;
    const size_t cols = 256;

    // Generate BF16 data
    auto fp32_data = generateRandomFP32(rows * cols);
    std::vector<uint16_t> bf16_data(rows * cols);
    for (size_t i = 0; i < rows * cols; ++i)
    {
        bf16_data[i] = bfloat16::from_float(fp32_data[i]).data;
    }

    // Create BF16 tensor
    auto tensor = std::make_shared<BF16Tensor>(std::vector<size_t>{rows, cols}, bf16_data);

    // Convert to BF16 (should be identity)
    std::vector<uint16_t> converted(rows * cols);
    tensor->to_bf16(converted.data());

    // Should be exact match
    for (size_t i = 0; i < rows * cols; ++i)
    {
        EXPECT_EQ(converted[i], bf16_data[i]);
    }
}

/**
 * @brief Test FP32 → FP16 → FP32 round-trip conversion
 *
 * FP16 has 5 exponent bits and 10 mantissa bits (vs 8/23 for FP32).
 * Expected precision loss: ~1e-3 to 1e-4 relative error, but smaller
 * dynamic range (can overflow/underflow more easily).
 */
TEST(Test__TensorConversion, FP32ToFP16RoundTrip)
{
    const size_t rows = 8;
    const size_t cols = 256;

    // Use smaller range for FP16 (max value ~65504)
    auto original_data = generateRandomFP32(rows * cols, -10.0f, 10.0f);

    // Create FP32 tensor
    auto fp32_tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{rows, cols});
    std::memcpy(fp32_tensor->mutable_data(), original_data.data(), original_data.size() * sizeof(float));

    // Convert FP32 → FP16
    std::vector<uint16_t> fp16_data(rows * cols);
    fp32_tensor->to_fp16(fp16_data.data());

    // Create FP16 tensor
    auto fp16_tensor = std::make_shared<FP16Tensor>(std::vector<size_t>{rows, cols}, fp16_data);

    // Convert FP16 → FP32
    std::vector<float> recovered(rows * cols);
    fp16_tensor->to_fp32(recovered.data());

    // Check precision loss is within expected bounds
    float rel_error = relativeL2Error(original_data.data(), recovered.data(), rows * cols);
    float max_diff = maxAbsDiff(original_data.data(), recovered.data(), rows * cols);

    EXPECT_LT(rel_error, 1e-3) << "Relative L2 error too high for FP16 round-trip";
    EXPECT_LT(max_diff, 0.01f) << "Max absolute difference too high for FP16 round-trip";
}

/**
 * @brief Test FP32 → INT8 block quantization accuracy
 *
 * Block quantization should preserve relative magnitudes within each block.
 * Check that dequantization recovers approximate values.
 */
TEST(Test__TensorConversion, FP32ToINT8BlockQuantization)
{
    const size_t rows = 8;
    const size_t cols = 256;
    const size_t block_size = 32;
    auto original_data = generateRandomFP32(rows * cols, -10.0f, 10.0f);

    // Create FP32 tensor
    auto tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{rows, cols});
    std::memcpy(tensor->mutable_data(), original_data.data(), original_data.size() * sizeof(float));

    // Quantize to INT8 with block scales
    const size_t total_elements = rows * cols;
    const size_t num_blocks = (total_elements + block_size - 1) / block_size;
    std::vector<int8_t> int8_data(total_elements);
    std::vector<float> scales(num_blocks);

    tensor->to_int8_blocked(int8_data.data(), scales.data(), block_size);

    // Dequantize and check accuracy
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

    // Check quantization error
    float rel_error = relativeL2Error(original_data.data(), dequantized.data(), total_elements);
    float max_diff = maxAbsDiff(original_data.data(), dequantized.data(), total_elements);

    // INT8 quantization has ~1/127 precision per block
    EXPECT_LT(rel_error, 0.05f) << "Relative L2 error too high for INT8 block quantization";
    EXPECT_LT(max_diff, 0.5f) << "Max absolute difference too high for INT8 block quantization";
}

/**
 * @brief Test row conversion consistency
 *
 * Converting individual rows should produce same results as converting
 * entire tensor and extracting rows.
 */
TEST(Test__TensorConversion, RowConversionConsistency)
{
    const size_t rows = 8;
    const size_t cols = 256;
    auto original_data = generateRandomFP32(rows * cols);

    // Create FP32 tensor
    auto tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{rows, cols});
    std::memcpy(tensor->mutable_data(), original_data.data(), original_data.size() * sizeof(float));

    // Convert entire tensor
    std::vector<float> full_conversion(rows * cols);
    tensor->to_fp32(full_conversion.data());

    // Convert row-by-row
    std::vector<float> row_conversion(rows * cols);
    for (size_t row = 0; row < rows; ++row)
    {
        tensor->to_fp32_row(row, row_conversion.data() + row * cols);
    }

    // Should be identical
    for (size_t i = 0; i < rows * cols; ++i)
    {
        EXPECT_FLOAT_EQ(full_conversion[i], row_conversion[i])
            << "Mismatch at element " << i;
    }
}

/**
 * @brief Test span conversion consistency
 *
 * Converting spans should produce same results as converting entire
 * tensor and extracting spans.
 */
TEST(Test__TensorConversion, SpanConversionConsistency)
{
    const size_t rows = 8;
    const size_t cols = 256;
    auto original_data = generateRandomFP32(rows * cols);

    // Create FP32 tensor
    auto tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{rows, cols});
    std::memcpy(tensor->mutable_data(), original_data.data(), original_data.size() * sizeof(float));

    // Convert entire tensor
    std::vector<float> full_conversion(rows * cols);
    tensor->to_fp32(full_conversion.data());

    // Test various span conversions
    const size_t span_size = 64;
    for (size_t offset = 0; offset + span_size <= rows * cols; offset += span_size)
    {
        std::vector<float> span_conversion(span_size);
        tensor->to_fp32_span(offset, span_size, span_conversion.data());

        // Compare with corresponding section of full conversion
        for (size_t i = 0; i < span_size; ++i)
        {
            EXPECT_FLOAT_EQ(span_conversion[i], full_conversion[offset + i])
                << "Mismatch at offset " << offset << ", index " << i;
        }
    }
}

/**
 * @brief Test cross-format conversion equivalence
 *
 * All conversion paths to FP32 should produce equivalent results
 * (within expected precision bounds).
 */
TEST(Test__TensorConversion, CrossFormatEquivalence)
{
    const size_t rows = 8;
    const size_t cols = 256;
    auto original_data = generateRandomFP32(rows * cols, -10.0f, 10.0f);

    // Create FP32 tensor
    auto fp32_tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{rows, cols});
    std::memcpy(fp32_tensor->mutable_data(), original_data.data(), original_data.size() * sizeof(float));

    // Path 1: FP32 → FP32 (direct)
    std::vector<float> path1(rows * cols);
    fp32_tensor->to_fp32(path1.data());

    // Path 2: FP32 → BF16 → FP32
    std::vector<uint16_t> bf16_temp(rows * cols);
    fp32_tensor->to_bf16(bf16_temp.data());
    auto bf16_tensor = std::make_shared<BF16Tensor>(std::vector<size_t>{rows, cols}, bf16_temp);
    std::vector<float> path2(rows * cols);
    bf16_tensor->to_fp32(path2.data());

    // Path 3: FP32 → FP16 → FP32
    std::vector<uint16_t> fp16_temp(rows * cols);
    fp32_tensor->to_fp16(fp16_temp.data());
    auto fp16_tensor = std::make_shared<FP16Tensor>(std::vector<size_t>{rows, cols}, fp16_temp);
    std::vector<float> path3(rows * cols);
    fp16_tensor->to_fp32(path3.data());

    // Path 1 (direct) should be exact
    for (size_t i = 0; i < rows * cols; ++i)
    {
        EXPECT_FLOAT_EQ(path1[i], original_data[i]);
    }

    // Path 2 and 3 should be close to original (with expected precision loss)
    float rel_error_bf16 = relativeL2Error(original_data.data(), path2.data(), rows * cols);
    float rel_error_fp16 = relativeL2Error(original_data.data(), path3.data(), rows * cols);

    EXPECT_LT(rel_error_bf16, 1e-2) << "BF16 path error too high";
    EXPECT_LT(rel_error_fp16, 1e-3) << "FP16 path error too high";
}

/**
 * @brief Test BF16 → FP16 cross-conversion
 *
 * Converting between BF16 and FP16 should go through FP32 intermediate.
 */
TEST(Test__TensorConversion, BF16ToFP16CrossConversion)
{
    const size_t rows = 8;
    const size_t cols = 256;
    auto original_data = generateRandomFP32(rows * cols, -10.0f, 10.0f);

    // Create BF16 tensor
    std::vector<uint16_t> bf16_data(rows * cols);
    for (size_t i = 0; i < rows * cols; ++i)
    {
        bf16_data[i] = bfloat16::from_float(original_data[i]).data;
    }
    auto bf16_tensor = std::make_shared<BF16Tensor>(std::vector<size_t>{rows, cols}, bf16_data);

    // Convert BF16 → FP16
    std::vector<uint16_t> fp16_data(rows * cols);
    bf16_tensor->to_fp16(fp16_data.data());

    // Convert FP16 → FP32 for validation
    auto fp16_tensor = std::make_shared<FP16Tensor>(std::vector<size_t>{rows, cols}, fp16_data);
    std::vector<float> recovered(rows * cols);
    fp16_tensor->to_fp32(recovered.data());

    // Should be close to original (accumulated precision loss from BF16 and FP16)
    float rel_error = relativeL2Error(original_data.data(), recovered.data(), rows * cols);
    EXPECT_LT(rel_error, 2e-2) << "Cross-conversion error too high";
}

/**
 * @brief Test INT8 block size variations
 *
 * Different block sizes should all work correctly.
 */
TEST(Test__TensorConversion, INT8VariableBlockSize)
{
    const size_t rows = 8;
    const size_t cols = 256;
    auto original_data = generateRandomFP32(rows * cols, -10.0f, 10.0f);

    auto tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{rows, cols});
    std::memcpy(tensor->mutable_data(), original_data.data(), original_data.size() * sizeof(float));

    // Test different block sizes
    for (size_t block_size : {16, 32, 64, 128})
    {
        const size_t total_elements = rows * cols;
        const size_t num_blocks = (total_elements + block_size - 1) / block_size;

        std::vector<int8_t> int8_data(total_elements);
        std::vector<float> scales(num_blocks);

        tensor->to_int8_blocked(int8_data.data(), scales.data(), block_size);

        // Verify all blocks have valid scales
        for (size_t i = 0; i < num_blocks; ++i)
        {
            EXPECT_GE(scales[i], 0.0f) << "Scale should be non-negative";
            EXPECT_LT(scales[i], 1e6f) << "Scale should be reasonable";
        }

        // Verify all int8 values are in valid range
        for (size_t i = 0; i < total_elements; ++i)
        {
            EXPECT_GE(int8_data[i], -127);
            EXPECT_LE(int8_data[i], 127);
        }
    }
}

/**
 * @brief Test conversion with zero and near-zero values
 *
 * Edge case: tensors with very small values should convert correctly.
 */
TEST(Test__TensorConversion, ZeroAndNearZeroValues)
{
    const size_t rows = 8;
    const size_t cols = 256;

    // Create tensor with mix of zeros and small values
    std::vector<float> data(rows * cols);
    for (size_t i = 0; i < rows * cols; ++i)
    {
        data[i] = (i % 3 == 0) ? 0.0f : 1e-6f;
    }

    auto tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{rows, cols});
    std::memcpy(tensor->mutable_data(), data.data(), data.size() * sizeof(float));

    // Convert to BF16 and back
    std::vector<uint16_t> bf16_data(rows * cols);
    tensor->to_bf16(bf16_data.data());

    auto bf16_tensor = std::make_shared<BF16Tensor>(std::vector<size_t>{rows, cols}, bf16_data);
    std::vector<float> recovered(rows * cols);
    bf16_tensor->to_fp32(recovered.data());

    // Zeros should remain exact zeros
    for (size_t i = 0; i < rows * cols; ++i)
    {
        if (data[i] == 0.0f)
        {
            EXPECT_FLOAT_EQ(recovered[i], 0.0f);
        }
    }
}

/**
 * @brief Test conversion with extreme values
 *
 * Edge case: large positive/negative values should convert correctly.
 */
TEST(Test__TensorConversion, ExtremeValues)
{
    const size_t rows = 8;
    const size_t cols = 256;

    auto tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{rows, cols});
    float *data = tensor->mutable_data();

    // Fill with extreme values
    for (size_t i = 0; i < rows * cols; ++i)
    {
        if (i % 4 == 0)
        {
            data[i] = 1000.0f;
        }
        else if (i % 4 == 1)
        {
            data[i] = -1000.0f;
        }
        else if (i % 4 == 2)
        {
            data[i] = 0.001f;
        }
        else
        {
            data[i] = -0.001f;
        }
    }

    // Convert to INT8 and verify scales handle extreme values
    const size_t block_size = 32;
    const size_t num_blocks = (rows * cols + block_size - 1) / block_size;
    std::vector<int8_t> int8_data(rows * cols);
    std::vector<float> scales(num_blocks);

    tensor->to_int8_blocked(int8_data.data(), scales.data(), block_size);

    // All scales should be valid
    for (size_t i = 0; i < num_blocks; ++i)
    {
        EXPECT_TRUE(std::isfinite(scales[i])) << "Scale should be finite";
        EXPECT_GT(scales[i], 0.0f) << "Scale should be positive for non-zero data";
    }
}
