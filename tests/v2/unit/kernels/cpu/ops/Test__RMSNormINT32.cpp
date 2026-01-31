/**
 * @file Test__RMSNormINT32.cpp
 * @brief Unit tests for INT32→INT8 RMSNorm with dynamic requantization
 *
 * Tests the full INT8 pipeline RMSNorm path:
 * - INT32 activations (accumulated from INT8×INT8 matmul)
 * - RMSNorm computation
 * - Dynamic per-row requantization to INT8
 * - Scale factor verification
 *
 * Validates:
 * - Numerical accuracy vs FP32 reference
 * - Dynamic range preservation
 * - Quantization error bounds
 * - SIMD correctness (AVX512/AVX2 vs scalar)
 *
 * @author David Sanftenberg
 * @date 2025-11-07
 */

#include <gtest/gtest.h>
#include "kernels/cpu/primitives/RMSNormPrimitives.h"
#include "utils/Logger.h"
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <limits>
#include <cstring>

using namespace llaminar2::primitives;

namespace
{
    constexpr float EPSILON = 1e-6f;

    /// Test fixture for INT32→INT8 RMSNorm tests
    class Test__RMSNormINT32 : public ::testing::Test
    {
    protected:
        std::mt19937 rng_{42}; // Fixed seed for reproducibility

        /// Generate random INT32 activations (typical range from INT8×INT8 accumulation)
        std::vector<int32_t> generateRandomINT32(size_t rows, size_t cols, int32_t min_val = -100000, int32_t max_val = 100000)
        {
            std::uniform_int_distribution<int32_t> dist(min_val, max_val);
            std::vector<int32_t> data(rows * cols);
            for (auto &val : data)
            {
                val = dist(rng_);
            }
            return data;
        }

        /// Generate gamma weights (FP32)
        std::vector<float> generateGamma(size_t cols)
        {
            std::uniform_real_distribution<float> dist(0.5f, 1.5f);
            std::vector<float> gamma(cols);
            for (auto &val : gamma)
            {
                val = dist(rng_);
            }
            return gamma;
        }

        /// Compute FP32 reference (INT32→FP32→normalize→FP32)
        std::vector<float> computeFP32Reference(
            const std::vector<int32_t> &input_int32,
            const std::vector<float> &gamma,
            size_t rows,
            size_t cols,
            float epsilon)
        {
            std::vector<float> output(rows * cols);

            for (size_t r = 0; r < rows; ++r)
            {
                const int32_t *row_in = input_int32.data() + r * cols;
                float *row_out = output.data() + r * cols;

                // Compute RMS
                double sum_sq = 0.0;
                for (size_t c = 0; c < cols; ++c)
                {
                    double val = static_cast<double>(row_in[c]);
                    sum_sq += val * val;
                }
                float rms = std::sqrt(static_cast<float>(sum_sq / cols) + epsilon);
                float inv_rms = 1.0f / rms;

                // Normalize
                for (size_t c = 0; c < cols; ++c)
                {
                    row_out[c] = static_cast<float>(row_in[c]) * inv_rms * gamma[c];
                }
            }

            return output;
        }

        /// Dequantize INT8 to FP32 using row scales
        std::vector<float> dequantizeINT8(
            const std::vector<int8_t> &int8_data,
            const std::vector<float> &row_scales,
            size_t rows,
            size_t cols)
        {
            std::vector<float> fp32_data(rows * cols);

            for (size_t r = 0; r < rows; ++r)
            {
                const int8_t *row_in = int8_data.data() + r * cols;
                float *row_out = fp32_data.data() + r * cols;
                float scale = row_scales[r];

                for (size_t c = 0; c < cols; ++c)
                {
                    row_out[c] = static_cast<float>(row_in[c]) * scale;
                }
            }

            return fp32_data;
        }

        /// Compute relative L2 error
        double computeRelativeL2(const std::vector<float> &ref, const std::vector<float> &test)
        {
            double sum_diff_sq = 0.0;
            double sum_ref_sq = 0.0;

            for (size_t i = 0; i < ref.size(); ++i)
            {
                double diff = static_cast<double>(ref[i]) - static_cast<double>(test[i]);
                sum_diff_sq += diff * diff;
                sum_ref_sq += static_cast<double>(ref[i]) * static_cast<double>(ref[i]);
            }

            return std::sqrt(sum_diff_sq / (sum_ref_sq + 1e-12));
        }

        /// Compute max absolute error
        float computeMaxAbsError(const std::vector<float> &ref, const std::vector<float> &test)
        {
            float max_err = 0.0f;
            for (size_t i = 0; i < ref.size(); ++i)
            {
                max_err = std::max(max_err, std::fabs(ref[i] - test[i]));
            }
            return max_err;
        }

        /// Check dynamic range utilization (what % of [-127, 127] is used)
        float computeDynamicRangeUtilization(const std::vector<int8_t> &int8_data)
        {
            int8_t min_val = 127;
            int8_t max_val = -127;

            for (auto val : int8_data)
            {
                min_val = std::min(min_val, val);
                max_val = std::max(max_val, val);
            }

            return (std::abs(max_val) > std::abs(min_val) ? std::abs(max_val) : std::abs(min_val)) / 127.0f;
        }
    };

    // ========================================================================
    // Test Cases
    // ========================================================================

    TEST_F(Test__RMSNormINT32, INT32ToINT8_ScalarCorrectness)
    {
        const size_t rows = 16;
        const size_t cols = 128;
        const float epsilon = EPSILON;

        auto input_int32 = generateRandomINT32(rows, cols);
        auto gamma = generateGamma(cols);

        // Compute FP32 reference
        auto ref_fp32 = computeFP32Reference(input_int32, gamma, rows, cols, epsilon);

        // Compute INT32→INT8 with scalar path
        std::vector<int8_t> output_int8(rows * cols);
        std::vector<float> row_scales(rows);

        RMSNormExecOptions opts_scalar;
        opts_scalar.force_scalar = true;

        rmsnorm_fused_int32_to_int8_vectorized(
            input_int32.data(), gamma.data(),
            output_int8.data(), row_scales.data(),
            rows, cols, epsilon, opts_scalar);

        // Dequantize INT8 back to FP32
        auto output_fp32 = dequantizeINT8(output_int8, row_scales, rows, cols);

        // Compare against FP32 reference
        double rel_l2 = computeRelativeL2(ref_fp32, output_fp32);
        float max_abs = computeMaxAbsError(ref_fp32, output_fp32);

        LOG_INFO("INT32→INT8 Scalar vs FP32 Reference: rel_l2=" << rel_l2 << ", max_abs=" << max_abs);

        // INT8 quantization introduces ~0.5% error (127 quantization levels)
        EXPECT_LT(rel_l2, 0.01) << "INT8 quantization error should be <1%";
        EXPECT_LT(max_abs, 1.0f) << "Max absolute error should be within 1 scale unit";
    }

    TEST_F(Test__RMSNormINT32, INT32ToINT8_ScalarVsVectorized)
    {
        const size_t rows = 16;
        const size_t cols = 128;
        const float epsilon = EPSILON;

        auto input_int32 = generateRandomINT32(rows, cols);
        auto gamma = generateGamma(cols);

        // Scalar path
        std::vector<int8_t> output_scalar(rows * cols);
        std::vector<float> scales_scalar(rows);

        RMSNormExecOptions opts_scalar;
        opts_scalar.force_scalar = true;

        rmsnorm_fused_int32_to_int8_vectorized(
            input_int32.data(), gamma.data(),
            output_scalar.data(), scales_scalar.data(),
            rows, cols, epsilon, opts_scalar);

        // Vectorized path
        std::vector<int8_t> output_vec(rows * cols);
        std::vector<float> scales_vec(rows);

        RMSNormExecOptions opts_vec;
        opts_vec.force_scalar = false;

        rmsnorm_fused_int32_to_int8_vectorized(
            input_int32.data(), gamma.data(),
            output_vec.data(), scales_vec.data(),
            rows, cols, epsilon, opts_vec);

        // Dequantize both
        auto scalar_fp32 = dequantizeINT8(output_scalar, scales_scalar, rows, cols);
        auto vec_fp32 = dequantizeINT8(output_vec, scales_vec, rows, cols);

        double rel_l2 = computeRelativeL2(scalar_fp32, vec_fp32);
        float max_abs = computeMaxAbsError(scalar_fp32, vec_fp32);

        LOG_INFO("INT32→INT8 Scalar vs Vectorized: rel_l2=" << rel_l2 << ", max_abs=" << max_abs);

        // Vectorized should match scalar exactly (same quantization logic)
        EXPECT_LT(rel_l2, 1e-5) << "Vectorized should closely match scalar";
        EXPECT_LT(max_abs, 0.01f) << "Max difference should be negligible";

        // Check scales match
        for (size_t r = 0; r < rows; ++r)
        {
            EXPECT_NEAR(scales_scalar[r], scales_vec[r], 1e-6f) << "Row " << r << " scales should match";
        }
    }

    TEST_F(Test__RMSNormINT32, DynamicRangeUtilization)
    {
        const size_t rows = 32;
        const size_t cols = 256;
        const float epsilon = EPSILON;

        auto input_int32 = generateRandomINT32(rows, cols);
        auto gamma = generateGamma(cols);

        std::vector<int8_t> output_int8(rows * cols);
        std::vector<float> row_scales(rows);

        RMSNormExecOptions opts;
        opts.force_scalar = false;

        rmsnorm_fused_int32_to_int8_vectorized(
            input_int32.data(), gamma.data(),
            output_int8.data(), row_scales.data(),
            rows, cols, epsilon, opts);

        // Check dynamic range utilization per row
        for (size_t r = 0; r < rows; ++r)
        {
            const int8_t *row_data = output_int8.data() + r * cols;
            std::vector<int8_t> row_vec(row_data, row_data + cols);

            float utilization = computeDynamicRangeUtilization(row_vec);

            // Dynamic scaling should use most of [-127, 127] range
            EXPECT_GT(utilization, 0.5f) << "Row " << r << " should use >50% of INT8 range";

            LOG_INFO("Row " << r << " dynamic range utilization: " << (utilization * 100.0f) << "%");
        }
    }

    TEST_F(Test__RMSNormINT32, NoGamma)
    {
        const size_t rows = 8;
        const size_t cols = 64;
        const float epsilon = EPSILON;

        auto input_int32 = generateRandomINT32(rows, cols);

        std::vector<int8_t> output_int8(rows * cols);
        std::vector<float> row_scales(rows);

        RMSNormExecOptions opts;
        opts.force_scalar = false;

        // No gamma (nullptr)
        rmsnorm_fused_int32_to_int8_vectorized(
            input_int32.data(), nullptr,
            output_int8.data(), row_scales.data(),
            rows, cols, epsilon, opts);

        // Check no crashes, output is non-zero
        bool has_nonzero = false;
        for (auto val : output_int8)
        {
            if (val != 0)
            {
                has_nonzero = true;
                break;
            }
        }

        EXPECT_TRUE(has_nonzero) << "Output should have non-zero values";
    }

    TEST_F(Test__RMSNormINT32, LargeSequence)
    {
        const size_t rows = 128;
        const size_t cols = 512;
        const float epsilon = EPSILON;

        auto input_int32 = generateRandomINT32(rows, cols);
        auto gamma = generateGamma(cols);

        std::vector<int8_t> output_int8(rows * cols);
        std::vector<float> row_scales(rows);

        RMSNormExecOptions opts;
        opts.force_scalar = false;

        rmsnorm_fused_int32_to_int8_vectorized(
            input_int32.data(), gamma.data(),
            output_int8.data(), row_scales.data(),
            rows, cols, epsilon, opts);

        // Check for NaN/Inf in scales
        for (size_t r = 0; r < rows; ++r)
        {
            EXPECT_FALSE(std::isnan(row_scales[r])) << "Row " << r << " scale is NaN";
            EXPECT_FALSE(std::isinf(row_scales[r])) << "Row " << r << " scale is Inf";
            EXPECT_GT(row_scales[r], 0.0f) << "Row " << r << " scale should be positive";
        }

        // Compute FP32 reference and compare
        auto ref_fp32 = computeFP32Reference(input_int32, gamma, rows, cols, epsilon);
        auto output_fp32 = dequantizeINT8(output_int8, row_scales, rows, cols);

        double rel_l2 = computeRelativeL2(ref_fp32, output_fp32);
        LOG_INFO("Large sequence INT32→INT8: rel_l2=" << rel_l2);

        EXPECT_LT(rel_l2, 0.01) << "Large sequence should maintain <1% error";
    }

    TEST_F(Test__RMSNormINT32, ExtremeDynamicRange)
    {
        const size_t rows = 8;
        const size_t cols = 64;
        const float epsilon = EPSILON;

        // Create data with extreme dynamic range
        std::vector<int32_t> input_int32(rows * cols);
        for (size_t r = 0; r < rows; ++r)
        {
            for (size_t c = 0; c < cols; ++c)
            {
                // Alternate between very large and very small values
                if (c % 2 == 0)
                {
                    input_int32[r * cols + c] = 1000000;
                }
                else
                {
                    input_int32[r * cols + c] = 10;
                }
            }
        }

        auto gamma = generateGamma(cols);

        std::vector<int8_t> output_int8(rows * cols);
        std::vector<float> row_scales(rows);

        RMSNormExecOptions opts;
        opts.force_scalar = false;

        rmsnorm_fused_int32_to_int8_vectorized(
            input_int32.data(), gamma.data(),
            output_int8.data(), row_scales.data(),
            rows, cols, epsilon, opts);

        // Verify dynamic scaling handles extreme range
        for (size_t r = 0; r < rows; ++r)
        {
            EXPECT_GT(row_scales[r], 0.0f);
            EXPECT_FALSE(std::isnan(row_scales[r]));
            EXPECT_FALSE(std::isinf(row_scales[r]));
        }

        // Check quantized values are within [-127, 127]
        for (auto val : output_int8)
        {
            EXPECT_GE(val, -127);
            EXPECT_LE(val, 127);
        }
    }

    TEST_F(Test__RMSNormINT32, EdgeCase_ZeroInput)
    {
        const size_t rows = 4;
        const size_t cols = 32;
        const float epsilon = EPSILON;

        std::vector<int32_t> input_int32(rows * cols, 0); // All zeros
        auto gamma = generateGamma(cols);

        std::vector<int8_t> output_int8(rows * cols);
        std::vector<float> row_scales(rows);

        RMSNormExecOptions opts;
        opts.force_scalar = false;

        rmsnorm_fused_int32_to_int8_vectorized(
            input_int32.data(), gamma.data(),
            output_int8.data(), row_scales.data(),
            rows, cols, epsilon, opts);

        // All zeros → RMS = epsilon → inv_rms → still zero after scale
        for (auto val : output_int8)
        {
            EXPECT_EQ(val, 0) << "Zero input should produce zero output";
        }
    }

    TEST_F(Test__RMSNormINT32, EdgeCase_SingleRow)
    {
        const size_t rows = 1;
        const size_t cols = 64;
        const float epsilon = EPSILON;

        auto input_int32 = generateRandomINT32(rows, cols);
        auto gamma = generateGamma(cols);

        std::vector<int8_t> output_int8(rows * cols);
        std::vector<float> row_scales(rows);

        RMSNormExecOptions opts;
        opts.force_scalar = false;

        rmsnorm_fused_int32_to_int8_vectorized(
            input_int32.data(), gamma.data(),
            output_int8.data(), row_scales.data(),
            rows, cols, epsilon, opts);

        EXPECT_GT(row_scales[0], 0.0f);
        EXPECT_FALSE(std::isnan(row_scales[0]));
    }

    TEST_F(Test__RMSNormINT32, QuantizationErrorBounds)
    {
        const size_t rows = 16;
        const size_t cols = 128;
        const float epsilon = EPSILON;

        auto input_int32 = generateRandomINT32(rows, cols);
        auto gamma = generateGamma(cols);

        // Compute FP32 reference
        auto ref_fp32 = computeFP32Reference(input_int32, gamma, rows, cols, epsilon);

        // Quantize to INT8
        std::vector<int8_t> output_int8(rows * cols);
        std::vector<float> row_scales(rows);

        RMSNormExecOptions opts;
        opts.force_scalar = false;

        rmsnorm_fused_int32_to_int8_vectorized(
            input_int32.data(), gamma.data(),
            output_int8.data(), row_scales.data(),
            rows, cols, epsilon, opts);

        // Dequantize
        auto output_fp32 = dequantizeINT8(output_int8, row_scales, rows, cols);

        // Check per-element quantization error
        for (size_t i = 0; i < ref_fp32.size(); ++i)
        {
            float ref = ref_fp32[i];
            float quantized = output_fp32[i];
            float error = std::fabs(ref - quantized);

            size_t r = i / cols;
            float scale = row_scales[r];

            // Error should be within ±0.5 scale units (rounding error)
            EXPECT_LT(error, scale * 0.6f) << "Element " << i << " quantization error exceeds bounds";
        }
    }

} // anonymous namespace
