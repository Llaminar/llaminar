/**
 * @file Test__INT32RMSNorm.cpp
 * @brief Unit tests for INT32 RMSNorm kernel
 * @author David Sanftenberg
 * @date 2025-11-05
 *
 * This test suite validates the INT32→INT8 RMSNorm pipeline, which is
 * critical for full INT8 transformer inference. The kernel performs:
 * 1. RMS normalization on INT32 accumulator tensors
 * 2. Application of FP32 gamma weights
 * 3. Requantization to INT8 with per-row dynamic scaling
 *
 * Test Coverage:
 * 1. Basic correctness against FP32 reference
 * 2. Per-row quantization accuracy
 * 3. Gamma weight application
 * 4. Edge cases (zero rows, extreme values)
 * 5. Large tensor stress tests
 * 6. Numerical precision validation
 */

#include <gtest/gtest.h>
#include "../../../src/v2/kernels/cpu/ops/CPURMSNormKernelT.h"
#include "../../../src/v2/kernels/cpu/primitives/RMSNormPrimitives.h"
#include <memory>
#include <cmath>
#include <vector>
#include <algorithm>
#include <random>

using namespace llaminar2;

// ============================================================================
// Test Fixture
// ============================================================================

class Test__INT32RMSNorm : public ::testing::Test
{
protected:
    // Random number generator
    std::mt19937 rng_{42}; // Fixed seed for reproducibility

    void SetUp() override
    {
        // Setup code if needed
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

    // Helper: Compute RMSNorm reference (FP32)
    void rmsnorm_reference_fp32(
        const int32_t *input_int32,
        const float *gamma,
        float *output_fp32,
        int seq_len, int d_model,
        float eps)
    {
        for (int i = 0; i < seq_len; ++i)
        {
            const int32_t *row_in = input_int32 + i * d_model;
            float *row_out = output_fp32 + i * d_model;

            // Compute sum of squares
            double sum_sq = 0.0;
            for (int j = 0; j < d_model; ++j)
            {
                double val = (double)row_in[j];
                sum_sq += val * val;
            }

            // Compute RMS inverse
            double rms = std::sqrt(sum_sq / d_model + eps);
            float inv_rms = (rms > 0.0) ? (1.0f / rms) : 0.0f;

            // Apply normalization and gamma
            if (gamma)
            {
                for (int j = 0; j < d_model; ++j)
                {
                    row_out[j] = (float)row_in[j] * inv_rms * gamma[j];
                }
            }
            else
            {
                for (int j = 0; j < d_model; ++j)
                {
                    row_out[j] = (float)row_in[j] * inv_rms;
                }
            }
        }
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
};

// ============================================================================
// Basic Functionality Tests
// ============================================================================

/**
 * @brief Test basic INT32→INT8 RMSNorm without gamma
 */
TEST_F(Test__INT32RMSNorm, BasicWithoutGamma)
{
    const int seq_len = 4;
    const int d_model = 8;
    const float eps = 1e-6f;

    // Create input INT32 data
    std::vector<int32_t> input(seq_len * d_model);
    fill_random_int32(input, -1000, 1000);

    // Allocate outputs
    std::vector<int8_t> output_int8(seq_len * d_model);
    std::vector<float> output_row_scales(seq_len);

    // Run INT32→INT8 RMSNorm
    CPURMSNormKernelT<INT32Tensor> kernel;
    bool success = kernel.apply_int32_to_int8(
        input.data(),
        nullptr, // No gamma
        output_int8.data(),
        output_row_scales.data(),
        seq_len, d_model,
        eps);

    ASSERT_TRUE(success) << "INT32 RMSNorm should succeed";

    // Verify output is not all zeros
    bool has_nonzero = false;
    for (int i = 0; i < seq_len * d_model; ++i)
    {
        if (output_int8[i] != 0)
        {
            has_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero) << "Output should not be all zeros";

    // Verify scales are positive
    for (int i = 0; i < seq_len; ++i)
    {
        EXPECT_GT(output_row_scales[i], 0.0f) << "Row scales should be positive";
    }
}

/**
 * @brief Test INT32→INT8 RMSNorm with gamma weights
 */
TEST_F(Test__INT32RMSNorm, BasicWithGamma)
{
    const int seq_len = 4;
    const int d_model = 8;
    const float eps = 1e-6f;

    // Create input INT32 data and gamma weights
    std::vector<int32_t> input(seq_len * d_model);
    std::vector<float> gamma(d_model);
    fill_random_int32(input, -1000, 1000);
    fill_random_fp32(gamma, 0.8f, 1.2f);

    // Allocate outputs
    std::vector<int8_t> output_int8(seq_len * d_model);
    std::vector<float> output_row_scales(seq_len);

    // Run INT32→INT8 RMSNorm
    CPURMSNormKernelT<INT32Tensor> kernel;
    bool success = kernel.apply_int32_to_int8(
        input.data(),
        gamma.data(),
        output_int8.data(),
        output_row_scales.data(),
        seq_len, d_model,
        eps);

    ASSERT_TRUE(success) << "INT32 RMSNorm with gamma should succeed";

    // Verify output is not all zeros
    bool has_nonzero = false;
    for (int i = 0; i < seq_len * d_model; ++i)
    {
        if (output_int8[i] != 0)
        {
            has_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero) << "Output should not be all zeros";
}

/**
 * @brief Test INT32 vs FP32 RMSNorm accuracy (before quantization)
 */
TEST_F(Test__INT32RMSNorm, AccuracyVsFP32Reference)
{
    const int seq_len = 8;
    const int d_model = 64;
    const float eps = 1e-6f;

    // Create input INT32 data
    std::vector<int32_t> input(seq_len * d_model);
    fill_random_int32(input, -5000, 5000);

    // Create gamma weights
    std::vector<float> gamma(d_model);
    fill_random_fp32(gamma, 0.9f, 1.1f);

    // Compute FP32 reference
    std::vector<float> reference_fp32(seq_len * d_model);
    rmsnorm_reference_fp32(input.data(), gamma.data(), reference_fp32.data(),
                           seq_len, d_model, eps);

    // Run INT32→INT8 RMSNorm
    std::vector<int8_t> output_int8(seq_len * d_model);
    std::vector<float> output_row_scales(seq_len);

    CPURMSNormKernelT<INT32Tensor> kernel;
    bool success = kernel.apply_int32_to_int8(
        input.data(),
        gamma.data(),
        output_int8.data(),
        output_row_scales.data(),
        seq_len, d_model,
        eps);

    ASSERT_TRUE(success);

    // Dequantize INT8 output for comparison
    std::vector<float> dequantized(seq_len * d_model);
    for (int i = 0; i < seq_len; ++i)
    {
        float scale = output_row_scales[i];
        for (int j = 0; j < d_model; ++j)
        {
            dequantized[i * d_model + j] = (float)output_int8[i * d_model + j] * scale;
        }
    }

    // Compare against FP32 reference
    float rel_error = compute_relative_l2(reference_fp32.data(), dequantized.data(),
                                          seq_len * d_model);

    // INT32→INT8→FP32 should have <2% error due to requantization
    EXPECT_LT(rel_error, 0.02f) << "Dequantized output should match FP32 reference";
}

// ============================================================================
// Edge Cases and Robustness Tests
// ============================================================================

/**
 * @brief Test with all-zero row
 */
TEST_F(Test__INT32RMSNorm, ZeroRow)
{
    const int seq_len = 3;
    const int d_model = 8;
    const float eps = 1e-6f;

    std::vector<int32_t> input(seq_len * d_model);
    fill_random_int32(input, -1000, 1000);

    // Make second row all zeros
    std::fill(input.begin() + d_model, input.begin() + 2 * d_model, 0);

    std::vector<int8_t> output_int8(seq_len * d_model);
    std::vector<float> output_row_scales(seq_len);

    CPURMSNormKernelT<INT32Tensor> kernel;
    bool success = kernel.apply_int32_to_int8(
        input.data(), nullptr,
        output_int8.data(), output_row_scales.data(),
        seq_len, d_model, eps);

    ASSERT_TRUE(success);

    // Zero row should produce zero output
    for (int j = 0; j < d_model; ++j)
    {
        EXPECT_EQ(output_int8[d_model + j], 0)
            << "Zero input row should produce zero output";
    }
}

/**
 * @brief Test with extreme values
 */
TEST_F(Test__INT32RMSNorm, ExtremeValues)
{
    const int seq_len = 4;
    const int d_model = 8;
    const float eps = 1e-6f;

    std::vector<int32_t> input = {
        // Row 0: Very large positive
        100000, 90000, 80000, 70000, 60000, 50000, 40000, 30000,
        // Row 1: Very large negative
        -100000, -90000, -80000, -70000, -60000, -50000, -40000, -30000,
        // Row 2: Mixed large
        100000, -90000, 80000, -70000, 60000, -50000, 40000, -30000,
        // Row 3: Small values
        10, 20, 30, 40, 50, 60, 70, 80};

    std::vector<int8_t> output_int8(seq_len * d_model);
    std::vector<float> output_row_scales(seq_len);

    CPURMSNormKernelT<INT32Tensor> kernel;
    bool success = kernel.apply_int32_to_int8(
        input.data(), nullptr,
        output_int8.data(), output_row_scales.data(),
        seq_len, d_model, eps);

    ASSERT_TRUE(success);

    // All rows should have valid INT8 output
    for (int i = 0; i < seq_len * d_model; ++i)
    {
        EXPECT_GE(output_int8[i], -127);
        EXPECT_LE(output_int8[i], 127);
    }

    // All scales should be positive and reasonable
    for (int i = 0; i < seq_len; ++i)
    {
        EXPECT_GT(output_row_scales[i], 0.0f) << "Row " << i << " scale should be positive";
        EXPECT_LT(output_row_scales[i], 1.0f) << "Row " << i << " scale should be reasonable";
    }

    // After normalization, all rows should have similar output magnitude
    // (RMSNorm normalizes to unit RMS, so input magnitude doesn't affect output scale)
    // Verify that each row uses a good portion of INT8 range
    for (int row = 0; row < seq_len; ++row)
    {
        int max_val = 0;
        for (int col = 0; col < d_model; ++col)
        {
            max_val = std::max(max_val, (int)std::abs(output_int8[row * d_model + col]));
        }
        EXPECT_GE(max_val, 50) << "Row " << row << " should use INT8 range reasonably";
    }
}

/**
 * @brief Test single row (vector normalization)
 */
TEST_F(Test__INT32RMSNorm, SingleRow)
{
    const int seq_len = 1;
    const int d_model = 64;
    const float eps = 1e-6f;

    std::vector<int32_t> input(d_model);
    fill_random_int32(input, -1000, 1000);

    std::vector<int8_t> output_int8(d_model);
    std::vector<float> output_row_scales(1);

    CPURMSNormKernelT<INT32Tensor> kernel;
    bool success = kernel.apply_int32_to_int8(
        input.data(), nullptr,
        output_int8.data(), output_row_scales.data(),
        seq_len, d_model, eps);

    ASSERT_TRUE(success);
    EXPECT_GT(output_row_scales[0], 0.0f);
}

/**
 * @brief Test large tensor (stress test)
 */
TEST_F(Test__INT32RMSNorm, LargeTensorStressTest)
{
    const int seq_len = 128;
    const int d_model = 512;
    const float eps = 1e-6f;

    std::vector<int32_t> input(seq_len * d_model);
    fill_random_int32(input, -10000, 10000);

    std::vector<float> gamma(d_model);
    fill_random_fp32(gamma, 0.8f, 1.2f);

    std::vector<int8_t> output_int8(seq_len * d_model);
    std::vector<float> output_row_scales(seq_len);

    CPURMSNormKernelT<INT32Tensor> kernel;
    bool success = kernel.apply_int32_to_int8(
        input.data(), gamma.data(),
        output_int8.data(), output_row_scales.data(),
        seq_len, d_model, eps);

    ASSERT_TRUE(success);

    // Verify all scales are positive
    for (int i = 0; i < seq_len; ++i)
    {
        EXPECT_GT(output_row_scales[i], 0.0f);
    }
}

/**
 * @brief Test per-row quantization maintains precision
 */
TEST_F(Test__INT32RMSNorm, PerRowQuantizationPrecision)
{
    const int seq_len = 3;
    const int d_model = 8;
    const float eps = 1e-6f;

    // Create rows with vastly different input magnitudes
    // Note: After RMSNorm, magnitudes will be normalized, but distributions differ
    std::vector<int32_t> input = {
        // Row 0: Small values, narrow distribution
        10, 10, 11, 11, 12, 12, 13, 13,
        // Row 1: Large values, wide distribution
        10000, 5000, -3000, 8000, -6000, 4000, -2000, 7000,
        // Row 2: Medium values, medium distribution
        100, 50, -30, 80, -60, 40, -20, 70};

    std::vector<int8_t> output_int8(seq_len * d_model);
    std::vector<float> output_row_scales(seq_len);

    CPURMSNormKernelT<INT32Tensor> kernel;
    bool success = kernel.apply_int32_to_int8(
        input.data(), nullptr,
        output_int8.data(), output_row_scales.data(),
        seq_len, d_model, eps);

    ASSERT_TRUE(success);

    // Each row should have positive scales
    for (int i = 0; i < seq_len; ++i)
    {
        EXPECT_GT(output_row_scales[i], 0.0f) << "Row " << i << " scale should be positive";
    }

    // All rows should use the full INT8 range (max value ≈ 127)
    // Per-row quantization adapts to each row's post-normalization distribution
    for (int row = 0; row < seq_len; ++row)
    {
        int max_val = 0;
        for (int col = 0; col < d_model; ++col)
        {
            max_val = std::max(max_val, (int)std::abs(output_int8[row * d_model + col]));
        }
        EXPECT_GE(max_val, 80) << "Row " << row << " should use INT8 range effectively";
    }

    // Compute reference FP32 normalization to verify INT8 accuracy
    std::vector<float> reference_fp32(seq_len * d_model);
    rmsnorm_reference_fp32(input.data(), nullptr, reference_fp32.data(),
                           seq_len, d_model, eps);

    // Dequantize INT8 output
    std::vector<float> dequantized(seq_len * d_model);
    for (int i = 0; i < seq_len; ++i)
    {
        float scale = output_row_scales[i];
        for (int j = 0; j < d_model; ++j)
        {
            dequantized[i * d_model + j] = (float)output_int8[i * d_model + j] * scale;
        }
    }

    // Verify accuracy against reference
    float rel_error = compute_relative_l2(reference_fp32.data(), dequantized.data(),
                                          seq_len * d_model);
    EXPECT_LT(rel_error, 0.02f) << "Per-row quantization should maintain accuracy";
}

// ============================================================================
// Invalid Input Tests
// ============================================================================

/**
 * @brief Test null pointer handling
 */
TEST_F(Test__INT32RMSNorm, NullPointerHandling)
{
    const int seq_len = 4;
    const int d_model = 8;
    const float eps = 1e-6f;

    std::vector<int32_t> input(seq_len * d_model, 100);
    std::vector<int8_t> output_int8(seq_len * d_model);
    std::vector<float> output_row_scales(seq_len);

    CPURMSNormKernelT<INT32Tensor> kernel;

    // Null input
    EXPECT_FALSE(kernel.apply_int32_to_int8(
        nullptr, nullptr,
        output_int8.data(), output_row_scales.data(),
        seq_len, d_model, eps));

    // Null output
    EXPECT_FALSE(kernel.apply_int32_to_int8(
        input.data(), nullptr,
        nullptr, output_row_scales.data(),
        seq_len, d_model, eps));

    // Null scales
    EXPECT_FALSE(kernel.apply_int32_to_int8(
        input.data(), nullptr,
        output_int8.data(), nullptr,
        seq_len, d_model, eps));
}

/**
 * @brief Test invalid dimensions
 */
TEST_F(Test__INT32RMSNorm, InvalidDimensions)
{
    std::vector<int32_t> input(32);
    std::vector<int8_t> output_int8(32);
    std::vector<float> output_row_scales(4);

    CPURMSNormKernelT<INT32Tensor> kernel;

    // Zero seq_len
    EXPECT_FALSE(kernel.apply_int32_to_int8(
        input.data(), nullptr,
        output_int8.data(), output_row_scales.data(),
        0, 8, 1e-6f));

    // Zero d_model
    EXPECT_FALSE(kernel.apply_int32_to_int8(
        input.data(), nullptr,
        output_int8.data(), output_row_scales.data(),
        4, 0, 1e-6f));

    // Negative dimensions
    EXPECT_FALSE(kernel.apply_int32_to_int8(
        input.data(), nullptr,
        output_int8.data(), output_row_scales.data(),
        -1, 8, 1e-6f));
}

// ============================================================================
// Main Test Entry Point
// ============================================================================

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
