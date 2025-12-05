/**
 * @file Test__CPURMSNormTypedKernel.cpp
 * @brief Unit tests for CPURMSNormTypedKernel with typed precision conversion
 *
 * Tests the "kernel black-box model" where:
 * - External interface uses configured ActivationPrecision
 * - Internal computation uses FP32 for numerical stability
 * - Fused dequant/requant at precision boundaries
 *
 * @author David Sanftenberg
 * @date 2025-12-04
 */

#include <gtest/gtest.h>

#include "v2/kernels/cpu/ops/CPURMSNormTypedKernel.h"
#include "v2/kernels/cpu/primitives/RMSNormPrimitives.h"
#include "v2/tensors/SIMDHelpers.h"
#include "v2/tensors/BlockStructures.h"

#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <random>

namespace llaminar2
{
    namespace
    {

        /**
         * @brief Reference RMSNorm implementation for testing
         *
         * Computes: output[i] = (input[i] / rms(input)) * gamma[i]
         * where rms(x) = sqrt(mean(x^2) + eps)
         */
        void reference_rmsnorm(
            const float *input,
            const float *gamma,
            float *output,
            int rows,
            int cols,
            float epsilon)
        {
            for (int row = 0; row < rows; ++row)
            {
                const float *row_in = input + row * cols;
                float *row_out = output + row * cols;

                // Compute sum of squares
                float sum_sq = 0.0f;
                for (int i = 0; i < cols; ++i)
                {
                    sum_sq += row_in[i] * row_in[i];
                }

                // Compute RMS
                float rms = std::sqrt(sum_sq / cols + epsilon);
                float inv_rms = 1.0f / rms;

                // Normalize and scale
                for (int i = 0; i < cols; ++i)
                {
                    row_out[i] = row_in[i] * inv_rms * gamma[i];
                }
            }
        }

        /**
         * @brief Generate random FP32 data
         */
        std::vector<float> generate_random_fp32(size_t count, float min_val = -2.0f, float max_val = 2.0f)
        {
            std::mt19937 rng(42); // Fixed seed for reproducibility
            std::uniform_real_distribution<float> dist(min_val, max_val);

            std::vector<float> data(count);
            for (auto &v : data)
            {
                v = dist(rng);
            }
            return data;
        }

        /**
         * @brief Compute max absolute difference between two vectors
         */
        float max_abs_diff(const float *a, const float *b, size_t count)
        {
            float max_diff = 0.0f;
            for (size_t i = 0; i < count; ++i)
            {
                float diff = std::abs(a[i] - b[i]);
                if (diff > max_diff)
                    max_diff = diff;
            }
            return max_diff;
        }

        /**
         * @brief Compute mean absolute error between two vectors
         */
        float mean_abs_error(const float *a, const float *b, size_t count)
        {
            float sum = 0.0f;
            for (size_t i = 0; i < count; ++i)
            {
                sum += std::abs(a[i] - b[i]);
            }
            return sum / count;
        }

    } // anonymous namespace

    // =========================================================================
    // Test Fixture
    // =========================================================================

    class CPURMSNormTypedKernelTest : public ::testing::Test
    {
    protected:
        static constexpr float EPSILON = 1e-6f;

        // Tolerances for different precisions
        static constexpr float FP32_TOLERANCE = 1e-5f;
        static constexpr float BF16_TOLERANCE = 2e-2f; // BF16 has ~3 decimal digits precision (8-bit mantissa)
        static constexpr float FP16_TOLERANCE = 1e-2f; // FP16 has ~3.3 decimal digits precision (10-bit mantissa)
        static constexpr float Q8_1_TOLERANCE = 0.1f;  // Q8_1 is lossy (8-bit quantization)

        void SetUp() override
        {
            // Default test dimensions
            rows_ = 4;
            cols_ = 128; // Must be multiple of 32 for Q8_1
        }

        int rows_;
        int cols_;
    };

    // =========================================================================
    // FP32 Specialization Tests
    // =========================================================================

    TEST_F(CPURMSNormTypedKernelTest, FP32_apply_typed_matches_reference)
    {
        // Generate test data
        auto input = generate_random_fp32(rows_ * cols_);
        auto gamma = generate_random_fp32(cols_, 0.5f, 1.5f);
        std::vector<float> output(rows_ * cols_);
        std::vector<float> expected(rows_ * cols_);

        // Compute reference
        reference_rmsnorm(input.data(), gamma.data(), expected.data(), rows_, cols_, EPSILON);

        // Test typed kernel
        CPURMSNormTypedKernel<ActivationPrecision::FP32> kernel;
        ASSERT_TRUE(kernel.apply_typed(
            input.data(), gamma.data(), output.data(),
            rows_, cols_, EPSILON));

        // Verify
        float max_diff = max_abs_diff(output.data(), expected.data(), rows_ * cols_);
        EXPECT_LT(max_diff, FP32_TOLERANCE)
            << "FP32 RMSNorm max diff: " << max_diff;
    }

    TEST_F(CPURMSNormTypedKernelTest, FP32_apply_with_residual_add)
    {
        // Generate test data
        auto residual = generate_random_fp32(rows_ * cols_);
        auto input = generate_random_fp32(rows_ * cols_);
        auto gamma = generate_random_fp32(cols_, 0.5f, 1.5f);
        std::vector<float> output(rows_ * cols_);
        std::vector<float> expected(rows_ * cols_);

        // Compute reference: RMSNorm(residual + input)
        std::vector<float> fused(rows_ * cols_);
        for (size_t i = 0; i < fused.size(); ++i)
        {
            fused[i] = residual[i] + input[i];
        }
        reference_rmsnorm(fused.data(), gamma.data(), expected.data(), rows_, cols_, EPSILON);

        // Test typed kernel
        CPURMSNormTypedKernel<ActivationPrecision::FP32> kernel;
        ASSERT_TRUE(kernel.apply_with_residual_add(
            residual.data(), input.data(), gamma.data(), output.data(),
            rows_, cols_, EPSILON));

        // Verify
        float max_diff = max_abs_diff(output.data(), expected.data(), rows_ * cols_);
        EXPECT_LT(max_diff, FP32_TOLERANCE)
            << "FP32 RMSNorm with residual max diff: " << max_diff;
    }

    TEST_F(CPURMSNormTypedKernelTest, FP32_precision_metadata)
    {
        CPURMSNormTypedKernel<ActivationPrecision::FP32> kernel;
        EXPECT_EQ(kernel.precision(), ActivationPrecision::FP32);
        EXPECT_STREQ(kernel.precision_name(), "FP32");
        EXPECT_FLOAT_EQ(kernel.compression_ratio(), 1.0f);
    }

    // =========================================================================
    // BF16 Specialization Tests
    // =========================================================================

    TEST_F(CPURMSNormTypedKernelTest, BF16_apply_typed_with_conversion)
    {
        // Generate FP32 test data and convert to BF16
        auto fp32_input = generate_random_fp32(rows_ * cols_);
        auto gamma = generate_random_fp32(cols_, 0.5f, 1.5f);

        // Convert input to BF16
        std::vector<uint16_t> bf16_input(rows_ * cols_);
        simd::convert_fp32_to_bf16(
            fp32_input.data(), bf16_input.data(), rows_ * cols_);

        // Output buffers
        std::vector<uint16_t> bf16_output(rows_ * cols_);
        std::vector<float> fp32_output(rows_ * cols_);

        // Test typed kernel
        CPURMSNormTypedKernel<ActivationPrecision::BF16> kernel;
        ASSERT_TRUE(kernel.apply_typed(
            bf16_input.data(), gamma.data(), bf16_output.data(),
            rows_, cols_, EPSILON));

        // Dequantize output for comparison
        simd::convert_bf16_to_fp32(
            bf16_output.data(), fp32_output.data(), rows_ * cols_);

        // Compute reference with dequantized input
        std::vector<float> dequant_input(rows_ * cols_);
        simd::convert_bf16_to_fp32(
            bf16_input.data(), dequant_input.data(), rows_ * cols_);

        std::vector<float> expected(rows_ * cols_);
        reference_rmsnorm(dequant_input.data(), gamma.data(), expected.data(), rows_, cols_, EPSILON);

        // Verify (with BF16 tolerance for quantization noise)
        float max_diff = max_abs_diff(fp32_output.data(), expected.data(), rows_ * cols_);
        EXPECT_LT(max_diff, BF16_TOLERANCE)
            << "BF16 RMSNorm max diff: " << max_diff;
    }

    TEST_F(CPURMSNormTypedKernelTest, BF16_apply_with_residual_add)
    {
        // Generate test data
        auto fp32_residual = generate_random_fp32(rows_ * cols_);
        auto fp32_input = generate_random_fp32(rows_ * cols_);
        auto gamma = generate_random_fp32(cols_, 0.5f, 1.5f);

        // Convert residual to BF16
        std::vector<uint16_t> bf16_residual(rows_ * cols_);
        simd::convert_fp32_to_bf16(
            fp32_residual.data(), bf16_residual.data(), rows_ * cols_);

        // Output buffers
        std::vector<uint16_t> bf16_output(rows_ * cols_);
        std::vector<float> fp32_output(rows_ * cols_);

        // Test typed kernel
        CPURMSNormTypedKernel<ActivationPrecision::BF16> kernel;
        ASSERT_TRUE(kernel.apply_with_residual_add(
            bf16_residual.data(), fp32_input.data(), gamma.data(), bf16_output.data(),
            rows_, cols_, EPSILON));

        // Dequantize output for comparison
        simd::convert_bf16_to_fp32(
            bf16_output.data(), fp32_output.data(), rows_ * cols_);

        // Compute reference
        std::vector<float> dequant_residual(rows_ * cols_);
        simd::convert_bf16_to_fp32(
            bf16_residual.data(), dequant_residual.data(), rows_ * cols_);

        std::vector<float> fused(rows_ * cols_);
        for (size_t i = 0; i < fused.size(); ++i)
        {
            fused[i] = dequant_residual[i] + fp32_input[i];
        }

        std::vector<float> expected(rows_ * cols_);
        reference_rmsnorm(fused.data(), gamma.data(), expected.data(), rows_, cols_, EPSILON);

        // Verify
        float max_diff = max_abs_diff(fp32_output.data(), expected.data(), rows_ * cols_);
        EXPECT_LT(max_diff, BF16_TOLERANCE)
            << "BF16 RMSNorm with residual max diff: " << max_diff;
    }

    TEST_F(CPURMSNormTypedKernelTest, BF16_precision_metadata)
    {
        CPURMSNormTypedKernel<ActivationPrecision::BF16> kernel;
        EXPECT_EQ(kernel.precision(), ActivationPrecision::BF16);
        EXPECT_STREQ(kernel.precision_name(), "BF16");
        EXPECT_FLOAT_EQ(kernel.compression_ratio(), 2.0f);
    }

    // =========================================================================
    // FP16 Specialization Tests
    // =========================================================================

    TEST_F(CPURMSNormTypedKernelTest, FP16_apply_typed_with_conversion)
    {
        // Generate FP32 test data and convert to FP16
        auto fp32_input = generate_random_fp32(rows_ * cols_);
        auto gamma = generate_random_fp32(cols_, 0.5f, 1.5f);

        // Convert input to FP16
        std::vector<uint16_t> fp16_input(rows_ * cols_);
        simd::convert_fp32_to_fp16(
            fp32_input.data(), fp16_input.data(), rows_ * cols_);

        // Output buffers
        std::vector<uint16_t> fp16_output(rows_ * cols_);
        std::vector<float> fp32_output(rows_ * cols_);

        // Test typed kernel
        CPURMSNormTypedKernel<ActivationPrecision::FP16> kernel;
        ASSERT_TRUE(kernel.apply_typed(
            fp16_input.data(), gamma.data(), fp16_output.data(),
            rows_, cols_, EPSILON));

        // Dequantize output for comparison
        simd::convert_fp16_to_fp32(
            fp16_output.data(), fp32_output.data(), rows_ * cols_);

        // Compute reference with dequantized input
        std::vector<float> dequant_input(rows_ * cols_);
        simd::convert_fp16_to_fp32(
            fp16_input.data(), dequant_input.data(), rows_ * cols_);

        std::vector<float> expected(rows_ * cols_);
        reference_rmsnorm(dequant_input.data(), gamma.data(), expected.data(), rows_, cols_, EPSILON);

        // Verify
        float max_diff = max_abs_diff(fp32_output.data(), expected.data(), rows_ * cols_);
        EXPECT_LT(max_diff, FP16_TOLERANCE)
            << "FP16 RMSNorm max diff: " << max_diff;
    }

    TEST_F(CPURMSNormTypedKernelTest, FP16_precision_metadata)
    {
        CPURMSNormTypedKernel<ActivationPrecision::FP16> kernel;
        EXPECT_EQ(kernel.precision(), ActivationPrecision::FP16);
        EXPECT_STREQ(kernel.precision_name(), "FP16");
        EXPECT_FLOAT_EQ(kernel.compression_ratio(), 2.0f);
    }

    // =========================================================================
    // Q8_1 Specialization Tests
    // =========================================================================

    TEST_F(CPURMSNormTypedKernelTest, Q8_1_apply_typed_with_conversion)
    {
        // Generate FP32 test data and convert to Q8_1
        auto fp32_input = generate_random_fp32(rows_ * cols_);
        auto gamma = generate_random_fp32(cols_, 0.5f, 1.5f);

        // Calculate number of Q8_1 blocks needed
        size_t n_blocks = (rows_ * cols_ + 31) / 32;
        std::vector<Q8_1Block> q8_input(n_blocks);
        std::vector<Q8_1Block> q8_output(n_blocks);

        // Convert input to Q8_1
        simd::quantize_fp32_to_q8_1_blocks(
            fp32_input.data(), q8_input.data(), rows_ * cols_);

        // Test typed kernel
        CPURMSNormTypedKernel<ActivationPrecision::Q8_1> kernel;
        ASSERT_TRUE(kernel.apply_typed(
            q8_input.data(), gamma.data(), q8_output.data(),
            rows_, cols_, EPSILON));

        // Dequantize output for comparison
        std::vector<float> fp32_output(rows_ * cols_);
        simd::dequantize_q8_1_to_fp32(
            q8_output.data(), fp32_output.data(), rows_ * cols_);

        // Compute reference with dequantized input
        std::vector<float> dequant_input(rows_ * cols_);
        simd::dequantize_q8_1_to_fp32(
            q8_input.data(), dequant_input.data(), rows_ * cols_);

        std::vector<float> expected(rows_ * cols_);
        reference_rmsnorm(dequant_input.data(), gamma.data(), expected.data(), rows_, cols_, EPSILON);

        // Verify (with Q8_1 tolerance for quantization noise)
        float max_diff = max_abs_diff(fp32_output.data(), expected.data(), rows_ * cols_);
        EXPECT_LT(max_diff, Q8_1_TOLERANCE)
            << "Q8_1 RMSNorm max diff: " << max_diff;
    }

    TEST_F(CPURMSNormTypedKernelTest, Q8_1_apply_with_residual_add)
    {
        // Generate test data
        auto fp32_residual = generate_random_fp32(rows_ * cols_);
        auto fp32_input = generate_random_fp32(rows_ * cols_);
        auto gamma = generate_random_fp32(cols_, 0.5f, 1.5f);

        // Calculate number of Q8_1 blocks needed
        size_t n_blocks = (rows_ * cols_ + 31) / 32;
        std::vector<Q8_1Block> q8_residual(n_blocks);
        std::vector<Q8_1Block> q8_output(n_blocks);

        // Convert residual to Q8_1
        simd::quantize_fp32_to_q8_1_blocks(
            fp32_residual.data(), q8_residual.data(), rows_ * cols_);

        // Test typed kernel
        CPURMSNormTypedKernel<ActivationPrecision::Q8_1> kernel;
        ASSERT_TRUE(kernel.apply_with_residual_add(
            q8_residual.data(), fp32_input.data(), gamma.data(), q8_output.data(),
            rows_, cols_, EPSILON));

        // Dequantize output for comparison
        std::vector<float> fp32_output(rows_ * cols_);
        simd::dequantize_q8_1_to_fp32(
            q8_output.data(), fp32_output.data(), rows_ * cols_);

        // Compute reference
        std::vector<float> dequant_residual(rows_ * cols_);
        simd::dequantize_q8_1_to_fp32(
            q8_residual.data(), dequant_residual.data(), rows_ * cols_);

        std::vector<float> fused(rows_ * cols_);
        for (size_t i = 0; i < fused.size(); ++i)
        {
            fused[i] = dequant_residual[i] + fp32_input[i];
        }

        std::vector<float> expected(rows_ * cols_);
        reference_rmsnorm(fused.data(), gamma.data(), expected.data(), rows_, cols_, EPSILON);

        // Verify
        float max_diff = max_abs_diff(fp32_output.data(), expected.data(), rows_ * cols_);
        EXPECT_LT(max_diff, Q8_1_TOLERANCE)
            << "Q8_1 RMSNorm with residual max diff: " << max_diff;
    }

    TEST_F(CPURMSNormTypedKernelTest, Q8_1_requires_multiple_of_32_cols)
    {
        // Test with cols not multiple of 32
        int bad_cols = 100; // Not multiple of 32
        auto input = generate_random_fp32(rows_ * bad_cols);
        auto gamma = generate_random_fp32(bad_cols);

        size_t n_blocks = (rows_ * bad_cols + 31) / 32;
        std::vector<Q8_1Block> q8_input(n_blocks);
        std::vector<Q8_1Block> q8_output(n_blocks);

        CPURMSNormTypedKernel<ActivationPrecision::Q8_1> kernel;
        EXPECT_FALSE(kernel.apply_typed(
            q8_input.data(), gamma.data(), q8_output.data(),
            rows_, bad_cols, EPSILON))
            << "Q8_1 kernel should reject cols not multiple of 32";
    }

    TEST_F(CPURMSNormTypedKernelTest, Q8_1_precision_metadata)
    {
        CPURMSNormTypedKernel<ActivationPrecision::Q8_1> kernel;
        EXPECT_EQ(kernel.precision(), ActivationPrecision::Q8_1);
        EXPECT_STREQ(kernel.precision_name(), "Q8_1");
        EXPECT_NEAR(kernel.compression_ratio(), 3.556f, 0.01f);
    }

    // =========================================================================
    // Edge Cases and Error Handling
    // =========================================================================

    TEST_F(CPURMSNormTypedKernelTest, FP32_rejects_null_input)
    {
        auto gamma = generate_random_fp32(cols_);
        std::vector<float> output(rows_ * cols_);

        CPURMSNormTypedKernel<ActivationPrecision::FP32> kernel;
        EXPECT_FALSE(kernel.apply_typed(
            nullptr, gamma.data(), output.data(),
            rows_, cols_, EPSILON));
    }

    TEST_F(CPURMSNormTypedKernelTest, FP32_rejects_null_gamma)
    {
        auto input = generate_random_fp32(rows_ * cols_);
        std::vector<float> output(rows_ * cols_);

        CPURMSNormTypedKernel<ActivationPrecision::FP32> kernel;
        EXPECT_FALSE(kernel.apply_typed(
            input.data(), nullptr, output.data(),
            rows_, cols_, EPSILON));
    }

    TEST_F(CPURMSNormTypedKernelTest, FP32_rejects_null_output)
    {
        auto input = generate_random_fp32(rows_ * cols_);
        auto gamma = generate_random_fp32(cols_);

        CPURMSNormTypedKernel<ActivationPrecision::FP32> kernel;
        EXPECT_FALSE(kernel.apply_typed(
            input.data(), gamma.data(), nullptr,
            rows_, cols_, EPSILON));
    }

    TEST_F(CPURMSNormTypedKernelTest, FP32_rejects_invalid_dimensions)
    {
        auto input = generate_random_fp32(rows_ * cols_);
        auto gamma = generate_random_fp32(cols_);
        std::vector<float> output(rows_ * cols_);

        CPURMSNormTypedKernel<ActivationPrecision::FP32> kernel;

        // Zero rows
        EXPECT_FALSE(kernel.apply_typed(
            input.data(), gamma.data(), output.data(),
            0, cols_, EPSILON));

        // Zero cols
        EXPECT_FALSE(kernel.apply_typed(
            input.data(), gamma.data(), output.data(),
            rows_, 0, EPSILON));

        // Negative rows
        EXPECT_FALSE(kernel.apply_typed(
            input.data(), gamma.data(), output.data(),
            -1, cols_, EPSILON));

        // Negative cols
        EXPECT_FALSE(kernel.apply_typed(
            input.data(), gamma.data(), output.data(),
            rows_, -1, EPSILON));
    }

    // =========================================================================
    // Large Scale Tests
    // =========================================================================

    TEST_F(CPURMSNormTypedKernelTest, FP32_large_batch)
    {
        int large_rows = 64;
        int large_cols = 4096;

        auto input = generate_random_fp32(large_rows * large_cols);
        auto gamma = generate_random_fp32(large_cols, 0.5f, 1.5f);
        std::vector<float> output(large_rows * large_cols);
        std::vector<float> expected(large_rows * large_cols);

        // Compute reference
        reference_rmsnorm(input.data(), gamma.data(), expected.data(),
                          large_rows, large_cols, EPSILON);

        // Test kernel
        CPURMSNormTypedKernel<ActivationPrecision::FP32> kernel;
        ASSERT_TRUE(kernel.apply_typed(
            input.data(), gamma.data(), output.data(),
            large_rows, large_cols, EPSILON));

        // Verify
        float max_diff = max_abs_diff(output.data(), expected.data(), large_rows * large_cols);
        EXPECT_LT(max_diff, FP32_TOLERANCE)
            << "Large batch FP32 RMSNorm max diff: " << max_diff;
    }

    TEST_F(CPURMSNormTypedKernelTest, BF16_large_batch)
    {
        int large_rows = 32;
        int large_cols = 2048;

        auto fp32_input = generate_random_fp32(large_rows * large_cols);
        auto gamma = generate_random_fp32(large_cols, 0.5f, 1.5f);

        // Convert to BF16
        std::vector<uint16_t> bf16_input(large_rows * large_cols);
        std::vector<uint16_t> bf16_output(large_rows * large_cols);
        simd::convert_fp32_to_bf16(
            fp32_input.data(), bf16_input.data(), large_rows * large_cols);

        // Test kernel
        CPURMSNormTypedKernel<ActivationPrecision::BF16> kernel;
        ASSERT_TRUE(kernel.apply_typed(
            bf16_input.data(), gamma.data(), bf16_output.data(),
            large_rows, large_cols, EPSILON));

        // Dequantize and compare
        std::vector<float> fp32_output(large_rows * large_cols);
        simd::convert_bf16_to_fp32(
            bf16_output.data(), fp32_output.data(), large_rows * large_cols);

        std::vector<float> dequant_input(large_rows * large_cols);
        simd::convert_bf16_to_fp32(
            bf16_input.data(), dequant_input.data(), large_rows * large_cols);

        std::vector<float> expected(large_rows * large_cols);
        reference_rmsnorm(dequant_input.data(), gamma.data(), expected.data(),
                          large_rows, large_cols, EPSILON);

        float max_diff = max_abs_diff(fp32_output.data(), expected.data(), large_rows * large_cols);
        EXPECT_LT(max_diff, BF16_TOLERANCE)
            << "Large batch BF16 RMSNorm max diff: " << max_diff;
    }

    TEST_F(CPURMSNormTypedKernelTest, Q8_1_large_batch)
    {
        int large_rows = 32;
        int large_cols = 2048;

        auto fp32_input = generate_random_fp32(large_rows * large_cols);
        auto gamma = generate_random_fp32(large_cols, 0.5f, 1.5f);

        // Calculate number of Q8_1 blocks needed
        size_t n_blocks = (large_rows * large_cols + 31) / 32;
        std::vector<Q8_1Block> q8_input(n_blocks);
        std::vector<Q8_1Block> q8_output(n_blocks);

        // Convert input to Q8_1
        simd::quantize_fp32_to_q8_1_blocks(
            fp32_input.data(), q8_input.data(), large_rows * large_cols);

        // Test typed kernel
        CPURMSNormTypedKernel<ActivationPrecision::Q8_1> kernel;
        ASSERT_TRUE(kernel.apply_typed(
            q8_input.data(), gamma.data(), q8_output.data(),
            large_rows, large_cols, EPSILON));

        // Dequantize output for comparison
        std::vector<float> fp32_output(large_rows * large_cols);
        simd::dequantize_q8_1_to_fp32(
            q8_output.data(), fp32_output.data(), large_rows * large_cols);

        // Compute reference with dequantized input
        std::vector<float> dequant_input(large_rows * large_cols);
        simd::dequantize_q8_1_to_fp32(
            q8_input.data(), dequant_input.data(), large_rows * large_cols);

        std::vector<float> expected(large_rows * large_cols);
        reference_rmsnorm(dequant_input.data(), gamma.data(), expected.data(),
                          large_rows, large_cols, EPSILON);

        // Verify
        float max_diff = max_abs_diff(fp32_output.data(), expected.data(), large_rows * large_cols);
        EXPECT_LT(max_diff, Q8_1_TOLERANCE)
            << "Large batch Q8_1 RMSNorm max diff: " << max_diff;
    }

    TEST_F(CPURMSNormTypedKernelTest, Q8_1_minimum_size)
    {
        int min_rows = 1;
        int min_cols = 32;

        auto fp32_input = generate_random_fp32(min_rows * min_cols);
        auto gamma = generate_random_fp32(min_cols, 0.5f, 1.5f);

        size_t n_blocks = (min_rows * min_cols + 31) / 32;
        std::vector<Q8_1Block> q8_input(n_blocks);
        std::vector<Q8_1Block> q8_output(n_blocks);

        simd::quantize_fp32_to_q8_1_blocks(
            fp32_input.data(), q8_input.data(), min_rows * min_cols);

        CPURMSNormTypedKernel<ActivationPrecision::Q8_1> kernel;
        ASSERT_TRUE(kernel.apply_typed(
            q8_input.data(), gamma.data(), q8_output.data(),
            min_rows, min_cols, EPSILON));

        std::vector<float> fp32_output(min_rows * min_cols);
        simd::dequantize_q8_1_to_fp32(
            q8_output.data(), fp32_output.data(), min_rows * min_cols);

        std::vector<float> dequant_input(min_rows * min_cols);
        simd::dequantize_q8_1_to_fp32(
            q8_input.data(), dequant_input.data(), min_rows * min_cols);

        std::vector<float> expected(min_rows * min_cols);
        reference_rmsnorm(dequant_input.data(), gamma.data(), expected.data(),
                          min_rows, min_cols, EPSILON);

        float max_diff = max_abs_diff(fp32_output.data(), expected.data(), min_rows * min_cols);
        EXPECT_LT(max_diff, Q8_1_TOLERANCE)
            << "Minimum size Q8_1 RMSNorm max diff: " << max_diff;
    }

    TEST_F(CPURMSNormTypedKernelTest, Q8_1_odd_rows)
    {
        int odd_rows = 7;
        int test_cols = 128;

        auto fp32_input = generate_random_fp32(odd_rows * test_cols);
        auto gamma = generate_random_fp32(test_cols, 0.5f, 1.5f);

        size_t n_blocks = (odd_rows * test_cols + 31) / 32;
        std::vector<Q8_1Block> q8_input(n_blocks);
        std::vector<Q8_1Block> q8_output(n_blocks);

        simd::quantize_fp32_to_q8_1_blocks(
            fp32_input.data(), q8_input.data(), odd_rows * test_cols);

        CPURMSNormTypedKernel<ActivationPrecision::Q8_1> kernel;
        ASSERT_TRUE(kernel.apply_typed(
            q8_input.data(), gamma.data(), q8_output.data(),
            odd_rows, test_cols, EPSILON));

        std::vector<float> fp32_output(odd_rows * test_cols);
        simd::dequantize_q8_1_to_fp32(
            q8_output.data(), fp32_output.data(), odd_rows * test_cols);

        std::vector<float> dequant_input(odd_rows * test_cols);
        simd::dequantize_q8_1_to_fp32(
            q8_input.data(), dequant_input.data(), odd_rows * test_cols);

        std::vector<float> expected(odd_rows * test_cols);
        reference_rmsnorm(dequant_input.data(), gamma.data(), expected.data(),
                          odd_rows, test_cols, EPSILON);

        float max_diff = max_abs_diff(fp32_output.data(), expected.data(), odd_rows * test_cols);
        EXPECT_LT(max_diff, Q8_1_TOLERANCE)
            << "Odd rows Q8_1 RMSNorm max diff: " << max_diff;
    }

    TEST_F(CPURMSNormTypedKernelTest, Q8_1_zero_input)
    {
        // All zeros input
        std::vector<float> fp32_input(rows_ * cols_, 0.0f);
        auto gamma = generate_random_fp32(cols_, 0.5f, 1.5f);

        size_t n_blocks = (rows_ * cols_ + 31) / 32;
        std::vector<Q8_1Block> q8_input(n_blocks);
        std::vector<Q8_1Block> q8_output(n_blocks);

        simd::quantize_fp32_to_q8_1_blocks(
            fp32_input.data(), q8_input.data(), rows_ * cols_);

        CPURMSNormTypedKernel<ActivationPrecision::Q8_1> kernel;
        ASSERT_TRUE(kernel.apply_typed(
            q8_input.data(), gamma.data(), q8_output.data(),
            rows_, cols_, EPSILON));

        std::vector<float> fp32_output(rows_ * cols_);
        simd::dequantize_q8_1_to_fp32(
            q8_output.data(), fp32_output.data(), rows_ * cols_);

        // Expected output should be all zeros (0 * anything = 0)
        // RMSNorm of 0 vector is 0 because of epsilon in denominator but 0 in numerator
        for (float val : fp32_output)
        {
            EXPECT_NEAR(val, 0.0f, Q8_1_TOLERANCE);
        }
    }

} // namespace llaminar2
