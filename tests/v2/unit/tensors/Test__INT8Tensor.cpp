/**
 * @file Test__INT8Tensor.cpp
 * @brief Unit tests for INT8Tensor class, focusing on dequantization and per-row scale handling
 * @author David Sanftenberg
 * @date 2025-01-31
 *
 * Tests cover:
 * - Basic construction and data access
 * - Per-row scale registration and dequantization
 * - Partial buffer dequantization (critical for batched execution)
 * - Global scale dequantization (weight tensors)
 * - Buffer size vs data size handling
 */

#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <cmath>
#include <algorithm>

#include "tensors/Tensors.h"
#include "tensors/TensorFactory.h"

using namespace llaminar2;

class Test__INT8Tensor : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Ensure clean state for each test
    }

    void TearDown() override
    {
        // Cleanup if needed
    }

    // Helper: Create FP32 test data
    std::vector<float> create_test_data(size_t rows, size_t cols, float base_value = 1.0f)
    {
        std::vector<float> data(rows * cols);
        for (size_t i = 0; i < data.size(); ++i)
        {
            data[i] = base_value + static_cast<float>(i) * 0.1f;
        }
        return data;
    }

    // Helper: Quantize FP32 to INT8 with per-row scales
    void quantize_per_row(const float *input, int8_t *output, float *scales,
                          size_t rows, size_t cols)
    {
        for (size_t r = 0; r < rows; ++r)
        {
            // Find max absolute value in row
            float max_val = 0.0f;
            for (size_t c = 0; c < cols; ++c)
            {
                max_val = std::max(max_val, std::abs(input[r * cols + c]));
            }

            // Compute scale
            scales[r] = max_val > 1e-8f ? max_val / 127.0f : 1.0f;
            float inv_scale = 1.0f / scales[r];

            // Quantize row
            for (size_t c = 0; c < cols; ++c)
            {
                const size_t idx = r * cols + c;
                float quantized = std::round(input[idx] * inv_scale);
                quantized = std::max(-127.0f, std::min(127.0f, quantized));
                output[idx] = static_cast<int8_t>(quantized);
            }
        }
    }

    // Helper: Compute max absolute difference
    float max_abs_diff(const float *a, const float *b, size_t n)
    {
        float max_diff = 0.0f;
        for (size_t i = 0; i < n; ++i)
        {
            max_diff = std::max(max_diff, std::abs(a[i] - b[i]));
        }
        return max_diff;
    }
};

/**
 * @brief Test basic INT8Tensor construction from FP32 data
 */
TEST_F(Test__INT8Tensor, BasicConstruction)
{
    const size_t rows = 4;
    const size_t cols = 8;
    auto fp32_data = create_test_data(rows, cols);

    // Create INT8Tensor from FP32 (uses per-column quantization by default)
    auto tensor = std::make_unique<INT8Tensor>(
        std::vector<size_t>{rows, cols},
        std::vector<float>(fp32_data.begin(), fp32_data.end()));

    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->shape()[0], rows);
    EXPECT_EQ(tensor->shape()[1], cols);
    EXPECT_EQ(tensor->shape()[0] * tensor->shape()[1], rows * cols);
}

/**
 * @brief Test global scale dequantization (weight tensor path)
 */
TEST_F(Test__INT8Tensor, GlobalScaleDequantization)
{
    const size_t rows = 4;
    const size_t cols = 8;
    const float scale = 0.05f;

    // Create INT8 data manually
    std::vector<int8_t> int8_data(rows * cols);
    for (size_t i = 0; i < int8_data.size(); ++i)
    {
        int8_data[i] = static_cast<int8_t>((i % 127) - 63);
    }

    // Create tensor with global scale
    auto tensor = std::make_unique<INT8Tensor>(
        std::vector<size_t>{rows, cols},
        std::vector<int8_t>(int8_data.begin(), int8_data.end()),
        scale);

    ASSERT_NE(tensor, nullptr);

    // Dequantize
    const float *dequant = tensor->data();
    ASSERT_NE(dequant, nullptr);

    // Verify dequantization
    for (size_t i = 0; i < rows * cols; ++i)
    {
        float expected = static_cast<float>(int8_data[i]) * scale;
        EXPECT_NEAR(dequant[i], expected, 1e-6f)
            << "Mismatch at index " << i;
    }
}

/**
 * @brief Test per-row scale registration and dequantization
 *
 * This tests the FusedRMSNormQuantize output path where each row
 * has its own quantization scale.
 */
TEST_F(Test__INT8Tensor, PerRowScaleDequantization)
{
    const size_t rows = 8;
    const size_t cols = 16;

    // Create test data
    auto fp32_data = create_test_data(rows, cols, 10.0f);

    // Quantize with per-row scales
    std::vector<int8_t> int8_data(rows * cols);
    std::vector<float> row_scales(rows);
    quantize_per_row(fp32_data.data(), int8_data.data(), row_scales.data(),
                     rows, cols);

    // Create INT8Tensor from pre-quantized data
    auto tensor = std::make_unique<INT8Tensor>(
        std::vector<size_t>{rows, cols},
        std::vector<int8_t>(int8_data.begin(), int8_data.end()),
        1.0f);

    // Register per-row scales (simulates FusedRMSNormQuantize output)
    tensor->set_row_scales(row_scales.data(), rows);

    // Dequantize
    const float *dequant = tensor->data();
    ASSERT_NE(dequant, nullptr);

    // Verify dequantization matches original (within quantization error)
    float max_diff = max_abs_diff(dequant, fp32_data.data(), rows * cols);

    // With symmetric per-row quantization, error should be < 2 * scale
    float max_expected_error = 0.0f;
    for (size_t r = 0; r < rows; ++r)
    {
        max_expected_error = std::max(max_expected_error, row_scales[r] * 2.0f);
    }

    EXPECT_LT(max_diff, max_expected_error)
        << "Dequantization error too large: " << max_diff
        << " (expected < " << max_expected_error << ")";
}

/**
 * @brief Test partial buffer dequantization
 *
 * CRITICAL TEST: This tests the bug we just fixed.
 * Buffer allocated for effective_max rows, but only effective_seq_len
 * rows have valid data and scales.
 */
TEST_F(Test__INT8Tensor, PartialBufferDequantization)
{
    const size_t buffer_capacity = 4096; // effective_max
    const size_t valid_rows = 4;         // effective_seq_len
    const size_t cols = 16;

    // Create test data (only for valid rows)
    auto fp32_data = create_test_data(valid_rows, cols, 5.0f);

    // Quantize with per-row scales (only valid rows)
    std::vector<int8_t> int8_data(buffer_capacity * cols, 0); // Zero-initialized full buffer
    std::vector<float> row_scales(valid_rows);
    quantize_per_row(fp32_data.data(), int8_data.data(), row_scales.data(),
                     valid_rows, cols);

    // Create INT8Tensor with FULL BUFFER SIZE (mimics batched execution)
    auto tensor = std::make_unique<INT8Tensor>(
        std::vector<size_t>{buffer_capacity, cols},
        std::vector<int8_t>(int8_data.begin(), int8_data.end()),
        1.0f);

    // Register scales ONLY for valid rows (mimics effective_seq_len)
    tensor->set_row_scales(row_scales.data(), valid_rows);

    // Dequantize - should handle partial buffer gracefully
    const float *dequant = tensor->data();
    ASSERT_NE(dequant, nullptr) << "Dequantization should succeed even with partial scales";

    // Verify ONLY valid rows are dequantized correctly
    for (size_t r = 0; r < valid_rows; ++r)
    {
        for (size_t c = 0; c < cols; ++c)
        {
            const size_t idx = r * cols + c;
            float expected = static_cast<float>(int8_data[idx]) * row_scales[r];
            EXPECT_NEAR(dequant[idx], expected, row_scales[r] * 2.0f)
                << "Mismatch at valid row " << r << ", col " << c;
        }
    }

    // Verify remaining rows are zeroed out
    for (size_t r = valid_rows; r < buffer_capacity; ++r)
    {
        for (size_t c = 0; c < cols; ++c)
        {
            const size_t idx = r * cols + c;
            EXPECT_EQ(dequant[idx], 0.0f)
                << "Row " << r << " (beyond valid data) should be zero";
        }
    }
}

/**
 * @brief Test INT8 direct data access
 */
TEST_F(Test__INT8Tensor, INT8DataAccess)
{
    const size_t rows = 4;
    const size_t cols = 8;

    std::vector<int8_t> int8_data(rows * cols);
    for (size_t i = 0; i < int8_data.size(); ++i)
    {
        int8_data[i] = static_cast<int8_t>(i - 16);
    }

    auto tensor = std::make_unique<INT8Tensor>(
        std::vector<size_t>{rows, cols},
        std::vector<int8_t>(int8_data.begin(), int8_data.end()),
        1.0f);

    const int8_t *retrieved = tensor->int8_data();
    ASSERT_NE(retrieved, nullptr);

    // Verify data integrity
    for (size_t i = 0; i < rows * cols; ++i)
    {
        EXPECT_EQ(retrieved[i], int8_data[i]);
    }
}

/**
 * @brief Test scale update after construction
 */
TEST_F(Test__INT8Tensor, ScaleUpdate)
{
    const size_t rows = 2;
    const size_t cols = 4;
    const float initial_scale = 1.0f;
    const float new_scale = 2.0f;

    std::vector<int8_t> int8_data = {10, 20, 30, 40, 50, 60, 70, 80};
    auto tensor = std::make_unique<INT8Tensor>(
        std::vector<size_t>{rows, cols},
        std::vector<int8_t>(int8_data.begin(), int8_data.end()),
        initial_scale);

    // Initial dequantization
    const float *dequant1 = tensor->data();
    float first_value = dequant1[0];
    EXPECT_NEAR(first_value, 10.0f * initial_scale, 1e-6f);

    // Update scale (note: INT8Tensor doesn't have set_scale method,
    // but we can test reconstruction)
    auto tensor2 = std::make_unique<INT8Tensor>(
        std::vector<size_t>{rows, cols},
        std::vector<int8_t>(int8_data.begin(), int8_data.end()),
        new_scale);

    const float *dequant2 = tensor2->data();
    float second_value = dequant2[0];
    EXPECT_NEAR(second_value, 10.0f * new_scale, 1e-6f);
}

/**
 * @brief Test empty row scales (should fall back to global scale)
 */
TEST_F(Test__INT8Tensor, EmptyRowScalesFallback)
{
    const size_t rows = 4;
    const size_t cols = 8;
    const float global_scale = 0.1f;

    std::vector<int8_t> int8_data(rows * cols);
    for (size_t i = 0; i < int8_data.size(); ++i)
    {
        int8_data[i] = static_cast<int8_t>((i % 50) - 25);
    }

    auto tensor = std::make_unique<INT8Tensor>(
        std::vector<size_t>{rows, cols},
        std::vector<int8_t>(int8_data.begin(), int8_data.end()),
        global_scale);

    // Don't set row scales - should use global scale
    const float *dequant = tensor->data();
    ASSERT_NE(dequant, nullptr);

    // Verify uses global scale
    for (size_t i = 0; i < rows * cols; ++i)
    {
        float expected = static_cast<float>(int8_data[i]) * global_scale;
        EXPECT_NEAR(dequant[i], expected, 1e-6f);
    }
}

/**
 * @brief Test zero scales handling
 */
TEST_F(Test__INT8Tensor, ZeroScalesHandling)
{
    const size_t rows = 4;
    const size_t cols = 8;

    auto fp32_data = create_test_data(rows, cols);

    // Create row with all zeros (will have near-zero scale)
    for (size_t c = 0; c < cols; ++c)
    {
        fp32_data[1 * cols + c] = 0.0f;
    }

    std::vector<int8_t> int8_data(rows * cols);
    std::vector<float> row_scales(rows);
    quantize_per_row(fp32_data.data(), int8_data.data(), row_scales.data(),
                     rows, cols);

    auto tensor = std::make_unique<INT8Tensor>(
        std::vector<size_t>{rows, cols},
        std::vector<int8_t>(int8_data.begin(), int8_data.end()),
        1.0f);
    tensor->set_row_scales(row_scales.data(), rows);

    const float *dequant = tensor->data();
    ASSERT_NE(dequant, nullptr);

    // Row 1 should be all zeros
    for (size_t c = 0; c < cols; ++c)
    {
        EXPECT_NEAR(dequant[1 * cols + c], 0.0f, 1e-6f);
    }

    // Other rows should dequantize correctly
    for (size_t r : {0, 2, 3})
    {
        for (size_t c = 0; c < cols; ++c)
        {
            float diff = std::abs(dequant[r * cols + c] - fp32_data[r * cols + c]);
            EXPECT_LT(diff, row_scales[r] * 2.0f);
        }
    }
}

/**
 * @brief Test repeated dequantization (caching behavior)
 */
TEST_F(Test__INT8Tensor, RepeatedDequantization)
{
    const size_t rows = 4;
    const size_t cols = 8;
    auto fp32_data = create_test_data(rows, cols);

    auto tensor = std::make_unique<INT8Tensor>(
        std::vector<size_t>{rows, cols},
        std::vector<float>(fp32_data.begin(), fp32_data.end()));

    // First call
    const float *dequant1 = tensor->data();
    ASSERT_NE(dequant1, nullptr);

    // Second call (should return cached result)
    const float *dequant2 = tensor->data();
    EXPECT_EQ(dequant1, dequant2) << "Should return same pointer (cached)";

    // Verify data is identical
    for (size_t i = 0; i < rows * cols; ++i)
    {
        EXPECT_EQ(dequant1[i], dequant2[i]);
    }
}

/**
 * @brief Test large buffer partial dequantization (realistic batch scenario)
 *
 * Simulates batch_size=8 with seq_len=2 in a buffer sized for max_seq_len=512
 */
TEST_F(Test__INT8Tensor, RealisticBatchScenario)
{
    const size_t max_seq_len = 512; // Buffer capacity
    const size_t batch_size = 8;
    const size_t seq_len_per_seq = 2;
    const size_t effective_seq_len = batch_size * seq_len_per_seq; // 16
    const size_t d_model = 896;

    // Create test data for effective sequence length only
    auto fp32_data = create_test_data(effective_seq_len, d_model, 8.0f);

    // Quantize
    std::vector<int8_t> int8_data(max_seq_len * d_model, 0);
    std::vector<float> row_scales(effective_seq_len);
    quantize_per_row(fp32_data.data(), int8_data.data(), row_scales.data(),
                     effective_seq_len, d_model);

    // Create tensor with full buffer
    auto tensor = std::make_unique<INT8Tensor>(
        std::vector<size_t>{max_seq_len, d_model},
        std::vector<int8_t>(int8_data.begin(), int8_data.end()),
        1.0f);

    // Register scales for effective_seq_len only
    tensor->set_row_scales(row_scales.data(), effective_seq_len);

    // Dequantize - MUST NOT FAIL
    const float *dequant = tensor->data();
    ASSERT_NE(dequant, nullptr)
        << "Dequantization must succeed with partial buffer";

    // Verify valid rows
    for (size_t r = 0; r < effective_seq_len; ++r)
    {
        float row_max_diff = 0.0f;
        for (size_t c = 0; c < d_model; ++c)
        {
            const size_t idx = r * d_model + c;
            float diff = std::abs(dequant[idx] - fp32_data[r * d_model + c]);
            row_max_diff = std::max(row_max_diff, diff);
        }
        EXPECT_LT(row_max_diff, row_scales[r] * 2.0f)
            << "Row " << r << " dequantization error too large";
    }

    // Verify padding rows are zeroed
    for (size_t r = effective_seq_len; r < max_seq_len; ++r)
    {
        for (size_t c = 0; c < d_model; ++c)
        {
            EXPECT_EQ(dequant[r * d_model + c], 0.0f)
                << "Padding row " << r << " should be zero";
        }
    }
}

/**
 * @brief Test shape validation
 */
TEST_F(Test__INT8Tensor, ShapeValidation)
{
    const size_t rows = 4;
    const size_t cols = 8;
    auto fp32_data = create_test_data(rows, cols);

    auto tensor = std::make_unique<INT8Tensor>(
        std::vector<size_t>{rows, cols},
        std::vector<float>(fp32_data.begin(), fp32_data.end()));

    EXPECT_EQ(tensor->shape().size(), 2);
    EXPECT_EQ(tensor->shape()[0], rows);
    EXPECT_EQ(tensor->shape()[1], cols);
    EXPECT_EQ(tensor->shape()[0] * tensor->shape()[1], rows * cols);
    EXPECT_EQ(tensor->native_type(), TensorType::INT8);
}
