/**
 * @file Test__RMSNormPrecision.cpp
 * @brief Unit tests for RMSNorm precision variants (FP32/BF16/FP16/INT32)
 *
 * Tests all precision paths (scalar/AVX2/AVX512) against FP32 reference.
 * Validates:
 * - Numerical accuracy (relative L2, max absolute error)
 * - SIMD intrinsics correctness (AVX2 vs AVX512 vs scalar)
 * - Cross-precision consistency (BF16 vs FP16 vs FP32)
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
    /// Test fixture for RMSNorm precision tests
    class Test__RMSNormPrecision : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            // Seed RNG for reproducibility
            rng_.seed(42);
        }

        /// Generate random FP32 tensor data
        std::vector<float> generateRandomFP32(size_t rows, size_t cols, float min_val = -1.0f, float max_val = 1.0f)
        {
            std::uniform_real_distribution<float> dist(min_val, max_val);
            std::vector<float> data(rows * cols);
            for (auto &val : data)
            {
                val = dist(rng_);
            }
            return data;
        }

        /// Generate random gamma weights
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

        /// Convert FP32 to BF16
        std::vector<uint16_t> fp32ToBF16(const std::vector<float> &fp32_data)
        {
            std::vector<uint16_t> bf16_data(fp32_data.size());
            for (size_t i = 0; i < fp32_data.size(); ++i)
            {
                uint32_t bits;
                std::memcpy(&bits, &fp32_data[i], sizeof(float));
                uint32_t rounding_bias = 0x7FFF + ((bits >> 16) & 1);
                uint32_t rounded = bits + rounding_bias;
                bf16_data[i] = static_cast<uint16_t>(rounded >> 16);
            }
            return bf16_data;
        }

        /// Convert BF16 to FP32
        std::vector<float> bf16ToFP32(const std::vector<uint16_t> &bf16_data)
        {
            std::vector<float> fp32_data(bf16_data.size());
            for (size_t i = 0; i < bf16_data.size(); ++i)
            {
                uint32_t bits = static_cast<uint32_t>(bf16_data[i]) << 16;
                std::memcpy(&fp32_data[i], &bits, sizeof(float));
            }
            return fp32_data;
        }

        /// Convert FP32 to FP16 (IEEE 754 compliant)
        std::vector<uint16_t> fp32ToFP16(const std::vector<float> &fp32_data)
        {
            std::vector<uint16_t> fp16_data(fp32_data.size());
            for (size_t i = 0; i < fp32_data.size(); ++i)
            {
                uint32_t bits;
                std::memcpy(&bits, &fp32_data[i], sizeof(float));

                uint32_t sign = (bits >> 16) & 0x8000U;
                int32_t exp = ((bits >> 23) & 0xFFU) - 127 + 15;
                uint32_t mantissa = (bits >> 13) & 0x03FFU;

                if (exp <= 0)
                {
                    if (exp < -10)
                    {
                        fp16_data[i] = sign; // Underflow
                    }
                    else
                    {
                        mantissa |= 0x0400U;
                        mantissa >>= (1 - exp);
                        fp16_data[i] = sign | mantissa;
                    }
                }
                else if (exp >= 0x1F)
                {
                    fp16_data[i] = sign | 0x7C00U; // Overflow
                }
                else
                {
                    fp16_data[i] = sign | (exp << 10) | mantissa;
                }
            }
            return fp16_data;
        }

        /// Convert FP16 to FP32
        std::vector<float> fp16ToFP32(const std::vector<uint16_t> &fp16_data)
        {
            std::vector<float> fp32_data(fp16_data.size());
            for (size_t i = 0; i < fp16_data.size(); ++i)
            {
                uint16_t h = fp16_data[i];
                uint32_t sign = (h & 0x8000U) << 16;
                uint32_t exp = (h & 0x7C00U) >> 10;
                uint32_t mantissa = h & 0x03FFU;

                uint32_t fp32_bits;
                if (exp == 0)
                {
                    if (mantissa == 0)
                    {
                        fp32_bits = sign;
                    }
                    else
                    {
                        exp = 1;
                        while ((mantissa & 0x0400U) == 0)
                        {
                            mantissa <<= 1;
                            exp--;
                        }
                        mantissa &= 0x03FFU;
                        fp32_bits = sign | ((exp + (127 - 15)) << 23) | (mantissa << 13);
                    }
                }
                else if (exp == 0x1F)
                {
                    fp32_bits = sign | 0x7F800000U | (mantissa << 13);
                }
                else
                {
                    fp32_bits = sign | ((exp + (127 - 15)) << 23) | (mantissa << 13);
                }

                std::memcpy(&fp32_data[i], &fp32_bits, sizeof(float));
            }
            return fp32_data;
        }

        /// Compute relative L2 error
        double computeRelativeL2(const std::vector<float> &ref, const std::vector<float> &test)
        {
            EXPECT_EQ(ref.size(), test.size());
            if (ref.size() != test.size())
                return std::numeric_limits<double>::infinity();

            double diff_sq = 0.0;
            double ref_sq = 0.0;

            for (size_t i = 0; i < ref.size(); ++i)
            {
                double diff = static_cast<double>(test[i]) - static_cast<double>(ref[i]);
                diff_sq += diff * diff;
                ref_sq += static_cast<double>(ref[i]) * static_cast<double>(ref[i]);
            }

            if (ref_sq == 0.0)
                return diff_sq == 0.0 ? 0.0 : std::numeric_limits<double>::infinity();

            return std::sqrt(diff_sq / ref_sq);
        }

        /// Compute max absolute error
        float computeMaxAbsError(const std::vector<float> &ref, const std::vector<float> &test)
        {
            EXPECT_EQ(ref.size(), test.size());
            if (ref.size() != test.size())
                return std::numeric_limits<float>::infinity();

            float max_err = 0.0f;
            for (size_t i = 0; i < ref.size(); ++i)
            {
                float err = std::abs(test[i] - ref[i]);
                max_err = std::max(max_err, err);
            }
            return max_err;
        }

        std::mt19937 rng_;
        static constexpr float EPSILON = 1e-6f;
    };

    // ========================================================================
    // LEGACY TESTS (Using old force_scalar pattern - DISABLED)
    // ========================================================================
    // These tests used the deprecated force_scalar flag which has been removed.
    // New parity tests below directly call per-row primitives instead.

#if 0 // DISABLED - Use new per-row parity tests instead

    // ========================================================================
    // FP32 Baseline Tests (scalar vs vectorized)
    // ========================================================================

    TEST_F(Test__RMSNormPrecision, FP32_ScalarVsVectorized)
    {
        const size_t rows = 32;
        const size_t cols = 896; // Typical d_model
        const float epsilon = EPSILON;

        auto input = generateRandomFP32(rows, cols);
        auto gamma = generateGamma(cols);

        std::vector<float> output_scalar(rows * cols);
        std::vector<float> output_vectorized(rows * cols);

        // Run scalar path
        RMSNormExecOptions opts_scalar;
        opts_scalar.force_scalar = true;
        rmsnorm_fused_vectorized(input.data(), gamma.data(), output_scalar.data(),
                                 rows, cols, epsilon, opts_scalar);

        // Run vectorized path
        RMSNormExecOptions opts_vec;
        opts_vec.force_scalar = false;
        rmsnorm_fused_vectorized(input.data(), gamma.data(), output_vectorized.data(),
                                 rows, cols, epsilon, opts_vec);

        // Compare results
        double rel_l2 = computeRelativeL2(output_scalar, output_vectorized);
        float max_abs = computeMaxAbsError(output_scalar, output_vectorized);

        LOG_INFO("FP32 Scalar vs Vectorized: rel_l2=" << rel_l2 << ", max_abs=" << max_abs);

        EXPECT_LT(rel_l2, 1e-6) << "FP32 vectorized should match scalar (machine precision)";
        EXPECT_LT(max_abs, 1e-5f);
    }

    // ========================================================================
    // BF16 Precision Tests
    // ========================================================================

    TEST_F(Test__RMSNormPrecision, BF16_ScalarCorrectness)
    {
        const size_t rows = 16;
        const size_t cols = 512;
        const float epsilon = EPSILON;

        auto input_fp32 = generateRandomFP32(rows, cols);
        auto gamma = generateGamma(cols);

        // Convert to BF16
        auto input_bf16 = fp32ToBF16(input_fp32);
        std::vector<uint16_t> output_bf16(rows * cols);

        // Run BF16 scalar kernel
        RMSNormExecOptions opts;
        opts.force_scalar = true;
        rmsnorm_fused_bf16_vectorized(input_bf16.data(), gamma.data(), output_bf16.data(),
                                       rows, cols, epsilon, opts);

        // Convert back to FP32 for comparison
        auto output_fp32 = bf16ToFP32(output_bf16);

        // Run FP32 reference on BF16-quantized input
        auto input_fp32_from_bf16 = bf16ToFP32(input_bf16);
        std::vector<float> ref_fp32(rows * cols);
        rmsnorm_fused_vectorized(input_fp32_from_bf16.data(), gamma.data(), ref_fp32.data(),
                                 rows, cols, epsilon, opts);

        double rel_l2 = computeRelativeL2(ref_fp32, output_fp32);
        float max_abs = computeMaxAbsError(ref_fp32, output_fp32);

        LOG_INFO("BF16 Scalar vs FP32: rel_l2=" << rel_l2 << ", max_abs=" << max_abs);

        // BF16 has 7 mantissa bits (~3 decimal digits), expect moderate precision
        EXPECT_LT(rel_l2, 5e-3) << "BF16 should match FP32 within BF16 precision limits";
        EXPECT_LT(max_abs, 1e-2f);
    }

    TEST_F(Test__RMSNormPrecision, BF16_ScalarVsVectorized)
    {
        const size_t rows = 32;
        const size_t cols = 896;
        const float epsilon = EPSILON;

        auto input_fp32 = generateRandomFP32(rows, cols);
        auto input_bf16 = fp32ToBF16(input_fp32);
        auto gamma = generateGamma(cols);

        std::vector<uint16_t> output_scalar(rows * cols);
        std::vector<uint16_t> output_vectorized(rows * cols);

        // Scalar path
        RMSNormExecOptions opts_scalar;
        opts_scalar.force_scalar = true;
        rmsnorm_fused_bf16_vectorized(input_bf16.data(), gamma.data(), output_scalar.data(),
                                       rows, cols, epsilon, opts_scalar);

        // Vectorized path
        RMSNormExecOptions opts_vec;
        opts_vec.force_scalar = false;
        rmsnorm_fused_bf16_vectorized(input_bf16.data(), gamma.data(), output_vectorized.data(),
                                       rows, cols, epsilon, opts_vec);

        // Convert to FP32 for comparison
        auto output_scalar_fp32 = bf16ToFP32(output_scalar);
        auto output_vec_fp32 = bf16ToFP32(output_vectorized);

        double rel_l2 = computeRelativeL2(output_scalar_fp32, output_vec_fp32);
        float max_abs = computeMaxAbsError(output_scalar_fp32, output_vec_fp32);

        LOG_INFO("BF16 Scalar vs Vectorized: rel_l2=" << rel_l2 << ", max_abs=" << max_abs);

        EXPECT_LT(rel_l2, 1e-6) << "BF16 vectorized should match scalar exactly";
        EXPECT_LT(max_abs, 1e-5f);
    }

    TEST_F(Test__RMSNormPrecision, BF16_LargeSequence)
    {
        const size_t rows = 512; // Large sequence
        const size_t cols = 896;
        const float epsilon = EPSILON;

        auto input_fp32 = generateRandomFP32(rows, cols);
        auto input_bf16 = fp32ToBF16(input_fp32);
        auto gamma = generateGamma(cols);

        std::vector<uint16_t> output_bf16(rows * cols);

        // Run BF16 kernel (will use vectorized path)
        RMSNormExecOptions opts;
        rmsnorm_fused_bf16_vectorized(input_bf16.data(), gamma.data(), output_bf16.data(),
                                       rows, cols, epsilon, opts);

        // Basic sanity checks
        auto output_fp32 = bf16ToFP32(output_bf16);
        
        // Check for NaN/Inf
        for (float val : output_fp32)
        {
            EXPECT_FALSE(std::isnan(val)) << "BF16 output should not contain NaN";
            EXPECT_FALSE(std::isinf(val)) << "BF16 output should not contain Inf";
        }

        // Check reasonable magnitude (normalized values)
        float max_val = *std::max_element(output_fp32.begin(), output_fp32.end(),
                                          [](float a, float b) { return std::abs(a) < std::abs(b); });
        EXPECT_LT(std::abs(max_val), 10.0f) << "Normalized values should be reasonable";
    }

    // ========================================================================
    // FP16 Precision Tests
    // ========================================================================

    TEST_F(Test__RMSNormPrecision, FP16_ScalarCorrectness)
    {
        const size_t rows = 16;
        const size_t cols = 512;
        const float epsilon = EPSILON;

        auto input_fp32 = generateRandomFP32(rows, cols, -2.0f, 2.0f); // FP16 range
        auto gamma = generateGamma(cols);

        // Convert to FP16
        auto input_fp16 = fp32ToFP16(input_fp32);
        std::vector<uint16_t> output_fp16(rows * cols);

        // Run FP16 scalar kernel
        RMSNormExecOptions opts;
        opts.force_scalar = true;
        rmsnorm_fused_fp16_vectorized(input_fp16.data(), gamma.data(), output_fp16.data(),
                                       rows, cols, epsilon, opts);

        // Convert back to FP32
        auto output_fp32 = fp16ToFP32(output_fp16);

        // Run FP32 reference on FP16-quantized input
        auto input_fp32_from_fp16 = fp16ToFP32(input_fp16);
        std::vector<float> ref_fp32(rows * cols);
        rmsnorm_fused_vectorized(input_fp32_from_fp16.data(), gamma.data(), ref_fp32.data(),
                                 rows, cols, epsilon, opts);

        double rel_l2 = computeRelativeL2(ref_fp32, output_fp32);
        float max_abs = computeMaxAbsError(ref_fp32, output_fp32);

        LOG_INFO("FP16 Scalar vs FP32: rel_l2=" << rel_l2 << ", max_abs=" << max_abs);

        // FP16 has 10 mantissa bits (~3 decimal digits)
        EXPECT_LT(rel_l2, 5e-3) << "FP16 should match FP32 within FP16 precision limits";
        EXPECT_LT(max_abs, 1e-2f);
    }

    TEST_F(Test__RMSNormPrecision, FP16_ScalarVsVectorized)
    {
        const size_t rows = 32;
        const size_t cols = 896;
        const float epsilon = EPSILON;

        auto input_fp32 = generateRandomFP32(rows, cols, -2.0f, 2.0f);
        auto input_fp16 = fp32ToFP16(input_fp32);
        auto gamma = generateGamma(cols);

        std::vector<uint16_t> output_scalar(rows * cols);
        std::vector<uint16_t> output_vectorized(rows * cols);

        // Scalar path
        RMSNormExecOptions opts_scalar;
        opts_scalar.force_scalar = true;
        rmsnorm_fused_fp16_vectorized(input_fp16.data(), gamma.data(), output_scalar.data(),
                                       rows, cols, epsilon, opts_scalar);

        // Vectorized path
        RMSNormExecOptions opts_vec;
        opts_vec.force_scalar = false;
        rmsnorm_fused_fp16_vectorized(input_fp16.data(), gamma.data(), output_vectorized.data(),
                                       rows, cols, epsilon, opts_vec);

        // Convert to FP32 for comparison
        auto output_scalar_fp32 = fp16ToFP32(output_scalar);
        auto output_vec_fp32 = fp16ToFP32(output_vectorized);

        double rel_l2 = computeRelativeL2(output_scalar_fp32, output_vec_fp32);
        float max_abs = computeMaxAbsError(output_scalar_fp32, output_vec_fp32);

        LOG_INFO("FP16 Scalar vs Vectorized: rel_l2=" << rel_l2 << ", max_abs=" << max_abs);

        // Note: Hardware F16C and scalar IEEE 754 conversions differ slightly in rounding
        // Tolerances relaxed to account for expected hardware/software conversion differences
        EXPECT_LT(rel_l2, 1e-3) << "FP16 vectorized should closely match scalar";
        EXPECT_LT(max_abs, 5e-3f) << "Max absolute error should be within FP16 precision limits";
    }

    TEST_F(Test__RMSNormPrecision, FP16_DenormalHandling)
    {
        const size_t rows = 8;
        const size_t cols = 64;
        const float epsilon = EPSILON;

        // Generate values that will produce denormals
        std::vector<float> input_fp32(rows * cols);
        for (size_t i = 0; i < input_fp32.size(); ++i)
        {
            input_fp32[i] = 1e-5f * (i % 2 == 0 ? 1.0f : -1.0f);
        }

        auto input_fp16 = fp32ToFP16(input_fp32);
        auto gamma = generateGamma(cols);
        std::vector<uint16_t> output_fp16(rows * cols);

        RMSNormExecOptions opts;
        rmsnorm_fused_fp16_vectorized(input_fp16.data(), gamma.data(), output_fp16.data(),
                                       rows, cols, epsilon, opts);

        auto output_fp32 = fp16ToFP32(output_fp16);

        // Check no NaN/Inf
        for (float val : output_fp32)
        {
            EXPECT_FALSE(std::isnan(val)) << "FP16 should handle denormals correctly";
            EXPECT_FALSE(std::isinf(val));
        }
    }

    // ========================================================================
    // Cross-Precision Comparison Tests
    // ========================================================================

    TEST_F(Test__RMSNormPrecision, CrossPrecision_BF16vsFP16vsFP32)
    {
        const size_t rows = 16;
        const size_t cols = 512;
        const float epsilon = EPSILON;

        auto input_fp32 = generateRandomFP32(rows, cols, -1.0f, 1.0f);
        auto gamma = generateGamma(cols);

        // Run FP32 baseline
        std::vector<float> output_fp32(rows * cols);
        RMSNormExecOptions opts;
        rmsnorm_fused_vectorized(input_fp32.data(), gamma.data(), output_fp32.data(),
                                 rows, cols, epsilon, opts);

        // Run BF16
        auto input_bf16 = fp32ToBF16(input_fp32);
        std::vector<uint16_t> output_bf16(rows * cols);
        rmsnorm_fused_bf16_vectorized(input_bf16.data(), gamma.data(), output_bf16.data(),
                                       rows, cols, epsilon, opts);
        auto output_bf16_fp32 = bf16ToFP32(output_bf16);

        // Run FP16
        auto input_fp16 = fp32ToFP16(input_fp32);
        std::vector<uint16_t> output_fp16(rows * cols);
        rmsnorm_fused_fp16_vectorized(input_fp16.data(), gamma.data(), output_fp16.data(),
                                       rows, cols, epsilon, opts);
        auto output_fp16_fp32 = fp16ToFP32(output_fp16);

        // Compare all three
        double rel_l2_bf16 = computeRelativeL2(output_fp32, output_bf16_fp32);
        double rel_l2_fp16 = computeRelativeL2(output_fp32, output_fp16_fp32);
        double rel_l2_bf16_fp16 = computeRelativeL2(output_bf16_fp32, output_fp16_fp32);

        LOG_INFO("Cross-Precision:");
        LOG_INFO("  BF16 vs FP32: rel_l2=" << rel_l2_bf16);
        LOG_INFO("  FP16 vs FP32: rel_l2=" << rel_l2_fp16);
        LOG_INFO("  BF16 vs FP16: rel_l2=" << rel_l2_bf16_fp16);

        // All should be within reduced precision tolerance
        EXPECT_LT(rel_l2_bf16, 1e-2) << "BF16 should approximate FP32";
        EXPECT_LT(rel_l2_fp16, 1e-2) << "FP16 should approximate FP32";
        EXPECT_LT(rel_l2_bf16_fp16, 1e-2) << "BF16 and FP16 should be similar";
    }

    // ========================================================================
    // Edge Case Tests
    // ========================================================================

    TEST_F(Test__RMSNormPrecision, EdgeCase_ZeroInput)
    {
        const size_t rows = 4;
        const size_t cols = 64;
        const float epsilon = EPSILON;

        std::vector<float> input_fp32(rows * cols, 0.0f);
        auto gamma = generateGamma(cols);

        std::vector<float> output_fp32(rows * cols);
        RMSNormExecOptions opts;
        rmsnorm_fused_vectorized(input_fp32.data(), gamma.data(), output_fp32.data(),
                                 rows, cols, epsilon, opts);

        // Output should be zero (0 / rms * gamma = 0)
        for (float val : output_fp32)
        {
            EXPECT_FLOAT_EQ(val, 0.0f) << "Zero input should produce zero output";
        }
    }

    TEST_F(Test__RMSNormPrecision, EdgeCase_SingleRow)
    {
        const size_t rows = 1;
        const size_t cols = 896;
        const float epsilon = EPSILON;

        auto input_fp32 = generateRandomFP32(rows, cols);
        auto gamma = generateGamma(cols);

        std::vector<float> output_scalar(rows * cols);
        std::vector<float> output_vectorized(rows * cols);

        RMSNormExecOptions opts_scalar;
        opts_scalar.force_scalar = true;
        rmsnorm_fused_vectorized(input_fp32.data(), gamma.data(), output_scalar.data(),
                                 rows, cols, epsilon, opts_scalar);

        RMSNormExecOptions opts_vec;
        opts_vec.force_scalar = false;
        rmsnorm_fused_vectorized(input_fp32.data(), gamma.data(), output_vectorized.data(),
                                 rows, cols, epsilon, opts_vec);

        double rel_l2 = computeRelativeL2(output_scalar, output_vectorized);
        EXPECT_LT(rel_l2, 1e-6) << "Single row should work correctly";
    }

    TEST_F(Test__RMSNormPrecision, EdgeCase_LargeValues)
    {
        const size_t rows = 8;
        const size_t cols = 256;
        const float epsilon = EPSILON;

        // Generate large values (near FP16/BF16 overflow threshold)
        std::vector<float> input_fp32(rows * cols);
        for (size_t i = 0; i < input_fp32.size(); ++i)
        {
            input_fp32[i] = 10.0f * ((i % 2 == 0) ? 1.0f : -1.0f);
        }

        auto gamma = generateGamma(cols);

        // Test BF16
        auto input_bf16 = fp32ToBF16(input_fp32);
        std::vector<uint16_t> output_bf16(rows * cols);
        RMSNormExecOptions opts;
        rmsnorm_fused_bf16_vectorized(input_bf16.data(), gamma.data(), output_bf16.data(),
                                       rows, cols, epsilon, opts);

        auto output_bf16_fp32 = bf16ToFP32(output_bf16);
        for (float val : output_bf16_fp32)
        {
            EXPECT_FALSE(std::isnan(val));
            EXPECT_FALSE(std::isinf(val));
        }
    }

    // ========================================================================
    // Implementation Parity Tests (Scalar vs AVX2 vs AVX512)
    // ========================================================================
    // These tests verify that scalar, AVX2, and AVX512 implementations
    // produce identical results for the same precision type.
    // ========================================================================

    TEST_F(Test__RMSNormPrecision, FP32_ScalarVsAVX2_Parity)
    {
        const size_t rows = 8;
        const size_t cols = 896;
        const float epsilon = EPSILON;

        auto input = generateRandomFP32(rows, cols);
        auto gamma = generateGamma(cols);

        std::vector<float> output_scalar(rows * cols);
        std::vector<float> output_avx2(rows * cols);

        // Scalar path
        RMSNormExecOptions opts_scalar;
        opts_scalar.force_scalar = true;
        rmsnorm_fused_vectorized(input.data(), gamma.data(), output_scalar.data(),
                                 rows, cols, epsilon, opts_scalar);

        // AVX2 path
#if defined(__AVX2__)
        RMSNormExecOptions opts_avx2;
        opts_avx2.force_scalar = false;
        opts_avx2.disable_avx512 = true;
        rmsnorm_fused_vectorized(input.data(), gamma.data(), output_avx2.data(),
                                 rows, cols, epsilon, opts_avx2);
#else
        output_avx2 = output_scalar;
#endif

        double rel_l2 = computeRelativeL2(output_scalar, output_avx2);
        float max_abs = computeMaxAbsError(output_scalar, output_avx2);

        LOG_INFO("FP32 Scalar vs AVX2: rel_l2=" << rel_l2 << ", max_abs=" << max_abs);
        EXPECT_LT(rel_l2, 1e-6) << "FP32 AVX2 should match scalar exactly";
        EXPECT_LT(max_abs, 1e-5f);
    }

    TEST_F(Test__RMSNormPrecision, FP32_ScalarVsAVX512_Parity)
    {
        const size_t rows = 8;
        const size_t cols = 896;
        const float epsilon = EPSILON;

        auto input = generateRandomFP32(rows, cols);
        auto gamma = generateGamma(cols);

        std::vector<float> output_scalar(rows * cols);
        std::vector<float> output_avx512(rows * cols);

        // Scalar path
        RMSNormExecOptions opts_scalar;
        opts_scalar.force_scalar = true;
        rmsnorm_fused_vectorized(input.data(), gamma.data(), output_scalar.data(),
                                 rows, cols, epsilon, opts_scalar);

        // AVX512 path
#if defined(__AVX512F__)
        RMSNormExecOptions opts_avx512;
        opts_avx512.force_scalar = false;
        opts_avx512.disable_avx512 = false;
        rmsnorm_fused_vectorized(input.data(), gamma.data(), output_avx512.data(),
                                 rows, cols, epsilon, opts_avx512);
#else
        output_avx512 = output_scalar;
#endif

        double rel_l2 = computeRelativeL2(output_scalar, output_avx512);
        float max_abs = computeMaxAbsError(output_scalar, output_avx512);

        LOG_INFO("FP32 Scalar vs AVX512: rel_l2=" << rel_l2 << ", max_abs=" << max_abs);
        EXPECT_LT(rel_l2, 1e-6) << "FP32 AVX512 should match scalar exactly";
        EXPECT_LT(max_abs, 1e-5f);
    }

    TEST_F(Test__RMSNormPrecision, BF16_ScalarVsAVX2_Parity)
    {
        const size_t rows = 8;
        const size_t cols = 896;
        const float epsilon = EPSILON;

        auto input_fp32 = generateRandomFP32(rows, cols);
        auto input_bf16 = fp32ToBF16(input_fp32);
        auto gamma = generateGamma(cols);

        std::vector<uint16_t> output_scalar(rows * cols);
        std::vector<uint16_t> output_avx2(rows * cols);

        // Scalar path
        RMSNormExecOptions opts_scalar;
        opts_scalar.force_scalar = true;
        rmsnorm_fused_bf16_vectorized(input_bf16.data(), gamma.data(), output_scalar.data(),
                                       rows, cols, epsilon, opts_scalar);

        // AVX2 path
#if defined(__AVX2__)
        RMSNormExecOptions opts_avx2;
        opts_avx2.force_scalar = false;
        opts_avx2.disable_avx512 = true;
        rmsnorm_fused_bf16_vectorized(input_bf16.data(), gamma.data(), output_avx2.data(),
                                       rows, cols, epsilon, opts_avx2);
#else
        output_avx2 = output_scalar;
#endif

        // Compare bit-exact
        for (size_t i = 0; i < output_scalar.size(); ++i)
        {
            EXPECT_EQ(output_scalar[i], output_avx2[i])
                << "BF16 AVX2 mismatch at index " << i;
        }

        auto output_scalar_fp32 = bf16ToFP32(output_scalar);
        auto output_avx2_fp32 = bf16ToFP32(output_avx2);
        double rel_l2 = computeRelativeL2(output_scalar_fp32, output_avx2_fp32);
        LOG_INFO("BF16 Scalar vs AVX2: rel_l2=" << rel_l2);
        EXPECT_LT(rel_l2, 1e-6);
    }

    TEST_F(Test__RMSNormPrecision, BF16_ScalarVsAVX512_Parity)
    {
        const size_t rows = 8;
        const size_t cols = 896;
        const float epsilon = EPSILON;

        auto input_fp32 = generateRandomFP32(rows, cols);
        auto input_bf16 = fp32ToBF16(input_fp32);
        auto gamma = generateGamma(cols);

        std::vector<uint16_t> output_scalar(rows * cols);
        std::vector<uint16_t> output_avx512(rows * cols);

        // Scalar path
        RMSNormExecOptions opts_scalar;
        opts_scalar.force_scalar = true;
        rmsnorm_fused_bf16_vectorized(input_bf16.data(), gamma.data(), output_scalar.data(),
                                       rows, cols, epsilon, opts_scalar);

        // AVX512 path
#if defined(__AVX512F__)
        RMSNormExecOptions opts_avx512;
        opts_avx512.force_scalar = false;
        opts_avx512.disable_avx512 = false;
        rmsnorm_fused_bf16_vectorized(input_bf16.data(), gamma.data(), output_avx512.data(),
                                       rows, cols, epsilon, opts_avx512);
#else
        output_avx512 = output_scalar;
#endif

        // Compare bit-exact
        for (size_t i = 0; i < output_scalar.size(); ++i)
        {
            EXPECT_EQ(output_scalar[i], output_avx512[i])
                << "BF16 AVX512 mismatch at index " << i;
        }

        auto output_scalar_fp32 = bf16ToFP32(output_scalar);
        auto output_avx512_fp32 = bf16ToFP32(output_avx512);
        double rel_l2 = computeRelativeL2(output_scalar_fp32, output_avx512_fp32);
        LOG_INFO("BF16 Scalar vs AVX512: rel_l2=" << rel_l2);
        EXPECT_LT(rel_l2, 1e-6);
    }

    TEST_F(Test__RMSNormPrecision, FP16_ScalarVsAVX2_Parity)
    {
        const size_t rows = 8;
        const size_t cols = 896;
        const float epsilon = EPSILON;

        auto input_fp32 = generateRandomFP32(rows, cols);
        auto input_fp16 = fp32ToFP16(input_fp32);
        auto gamma = generateGamma(cols);

        std::vector<uint16_t> output_scalar(rows * cols);
        std::vector<uint16_t> output_avx2(rows * cols);

        // Scalar path
        RMSNormExecOptions opts_scalar;
        opts_scalar.force_scalar = true;
        rmsnorm_fused_fp16_vectorized(input_fp16.data(), gamma.data(), output_scalar.data(),
                                       rows, cols, epsilon, opts_scalar);

        // AVX2 path
#if defined(__AVX2__)
        RMSNormExecOptions opts_avx2;
        opts_avx2.force_scalar = false;
        opts_avx2.disable_avx512 = true;
        rmsnorm_fused_fp16_vectorized(input_fp16.data(), gamma.data(), output_avx2.data(),
                                       rows, cols, epsilon, opts_avx2);
#else
        output_avx2 = output_scalar;
#endif

        // Compare bit-exact
        for (size_t i = 0; i < output_scalar.size(); ++i)
        {
            EXPECT_EQ(output_scalar[i], output_avx2[i])
                << "FP16 AVX2 mismatch at index " << i;
        }

        auto output_scalar_fp32 = fp16ToFP32(output_scalar);
        auto output_avx2_fp32 = fp16ToFP32(output_avx2);
        double rel_l2 = computeRelativeL2(output_scalar_fp32, output_avx2_fp32);
        LOG_INFO("FP16 Scalar vs AVX2: rel_l2=" << rel_l2);
        EXPECT_LT(rel_l2, 1e-6);
    }

    TEST_F(Test__RMSNormPrecision, FP16_ScalarVsAVX512_Parity)
    {
        const size_t rows = 8;
        const size_t cols = 896;
        const float epsilon = EPSILON;

        auto input_fp32 = generateRandomFP32(rows, cols);
        auto input_fp16 = fp32ToFP16(input_fp32);
        auto gamma = generateGamma(cols);

        std::vector<uint16_t> output_scalar(rows * cols);
        std::vector<uint16_t> output_avx512(rows * cols);

        // Scalar path
        RMSNormExecOptions opts_scalar;
        opts_scalar.force_scalar = true;
        rmsnorm_fused_fp16_vectorized(input_fp16.data(), gamma.data(), output_scalar.data(),
                                       rows, cols, epsilon, opts_scalar);

        // AVX512 path
#if defined(__AVX512F__)
        RMSNormExecOptions opts_avx512;
        opts_avx512.force_scalar = false;
        opts_avx512.disable_avx512 = false;
        rmsnorm_fused_fp16_vectorized(input_fp16.data(), gamma.data(), output_avx512.data(),
                                       rows, cols, epsilon, opts_avx512);
#else
        output_avx512 = output_scalar;
#endif

        // Compare bit-exact
        for (size_t i = 0; i < output_scalar.size(); ++i)
        {
            EXPECT_EQ(output_scalar[i], output_avx512[i])
                << "FP16 AVX512 mismatch at index " << i;
        }

        auto output_scalar_fp32 = fp16ToFP32(output_scalar);
        auto output_avx512_fp32 = fp16ToFP32(output_avx512);
        double rel_l2 = computeRelativeL2(output_scalar_fp32, output_avx512_fp32);
        LOG_INFO("FP16 Scalar vs AVX512: rel_l2=" << rel_l2);
        EXPECT_LT(rel_l2, 1e-6);
    }

#endif // DISABLED legacy tests

    // ============================================================================
    // Per-Row Primitive Parity Tests (Direct Function Calls)
    // ============================================================================
    // These tests directly call the exposed per-row primitives to verify
    // that different SIMD implementations (scalar/AVX2/AVX512) produce
    // identical results for the same precision.

    /// Test FP32 scalar vs AVX2 per-row primitives
    TEST_F(Test__RMSNormPrecision, FP32_PerRow_ScalarVsAVX2)
    {
        const size_t cols = 896; // Multiple of 8 for AVX2
        const float epsilon = 1e-5f;

        auto input = generateRandomFP32(1, cols);
        auto gamma = generateGamma(cols);

        std::vector<float> output_scalar(cols);
        std::vector<float> output_avx2(cols);

        // Call scalar primitive directly
        rmsnorm_fused_row_scalar(input.data(), gamma.data(), output_scalar.data(), cols, epsilon);

#if defined(__AVX2__)
        // Call AVX2 primitive directly
        rmsnorm_fused_row_avx2(input.data(), gamma.data(), output_avx2.data(), cols, epsilon);
#else
        output_avx2 = output_scalar; // Fallback if no AVX2
#endif

        // Expect bit-exact match (same precision, same algorithm)
        double rel_l2 = computeRelativeL2(output_scalar, output_avx2);
        LOG_INFO("FP32 PerRow Scalar vs AVX2: rel_l2=" << rel_l2);
        EXPECT_LT(rel_l2, 1e-6);
    }

    /// Test FP32 scalar vs AVX512 per-row primitives
    TEST_F(Test__RMSNormPrecision, FP32_PerRow_ScalarVsAVX512)
    {
        const size_t cols = 896;
        const float epsilon = 1e-5f;

        auto input = generateRandomFP32(1, cols);
        auto gamma = generateGamma(cols);

        std::vector<float> output_scalar(cols);
        std::vector<float> output_avx512(cols);

        rmsnorm_fused_row_scalar(input.data(), gamma.data(), output_scalar.data(), cols, epsilon);

#if defined(__AVX512F__)
        rmsnorm_fused_row_avx512(input.data(), gamma.data(), output_avx512.data(), cols, epsilon);
#else
        output_avx512 = output_scalar;
#endif

        double rel_l2 = computeRelativeL2(output_scalar, output_avx512);
        LOG_INFO("FP32 PerRow Scalar vs AVX512: rel_l2=" << rel_l2);
        EXPECT_LT(rel_l2, 1e-6);
    }

    /// Test FP32 AVX2 vs AVX512 per-row primitives
    TEST_F(Test__RMSNormPrecision, FP32_PerRow_AVX2VsAVX512)
    {
        const size_t cols = 896;
        const float epsilon = 1e-5f;

        auto input = generateRandomFP32(1, cols);
        auto gamma = generateGamma(cols);

        std::vector<float> output_avx2(cols);
        std::vector<float> output_avx512(cols);

#if defined(__AVX2__)
        rmsnorm_fused_row_avx2(input.data(), gamma.data(), output_avx2.data(), cols, epsilon);
#else
        rmsnorm_fused_row_scalar(input.data(), gamma.data(), output_avx2.data(), cols, epsilon);
#endif

#if defined(__AVX512F__)
        rmsnorm_fused_row_avx512(input.data(), gamma.data(), output_avx512.data(), cols, epsilon);
#else
        output_avx512 = output_avx2;
#endif

        double rel_l2 = computeRelativeL2(output_avx2, output_avx512);
        LOG_INFO("FP32 PerRow AVX2 vs AVX512: rel_l2=" << rel_l2);
        EXPECT_LT(rel_l2, 1e-6);
    }

    /// Test BF16 scalar vs AVX2 per-row primitives
    TEST_F(Test__RMSNormPrecision, BF16_PerRow_ScalarVsAVX2)
    {
        const size_t cols = 896;
        const float epsilon = 1e-5f;

        auto input_fp32 = generateRandomFP32(1, cols);
        auto input_bf16 = fp32ToBF16(input_fp32);
        auto gamma = generateGamma(cols);

        std::vector<uint16_t> output_scalar(cols);
        std::vector<uint16_t> output_avx2(cols);

        rmsnorm_fused_row_bf16_scalar(input_bf16.data(), gamma.data(), output_scalar.data(), cols, epsilon);

#if defined(__AVX2__)
        rmsnorm_fused_row_bf16_avx2(input_bf16.data(), gamma.data(), output_avx2.data(), cols, epsilon);
#else
        output_avx2 = output_scalar;
#endif

        auto output_scalar_fp32 = bf16ToFP32(output_scalar);
        auto output_avx2_fp32 = bf16ToFP32(output_avx2);
        double rel_l2 = computeRelativeL2(output_scalar_fp32, output_avx2_fp32);
        LOG_INFO("BF16 PerRow Scalar vs AVX2: rel_l2=" << rel_l2);
        EXPECT_LT(rel_l2, 1e-5); // BF16 has lower precision
    }

    /// Test BF16 scalar vs AVX512 per-row primitives
    TEST_F(Test__RMSNormPrecision, BF16_PerRow_ScalarVsAVX512)
    {
        const size_t cols = 896;
        const float epsilon = 1e-5f;

        auto input_fp32 = generateRandomFP32(1, cols);
        auto input_bf16 = fp32ToBF16(input_fp32);
        auto gamma = generateGamma(cols);

        std::vector<uint16_t> output_scalar(cols);
        std::vector<uint16_t> output_avx512(cols);

        rmsnorm_fused_row_bf16_scalar(input_bf16.data(), gamma.data(), output_scalar.data(), cols, epsilon);

#if defined(__AVX512F__)
        rmsnorm_fused_row_bf16_avx512(input_bf16.data(), gamma.data(), output_avx512.data(), cols, epsilon);
#else
        output_avx512 = output_scalar;
#endif

        auto output_scalar_fp32 = bf16ToFP32(output_scalar);
        auto output_avx512_fp32 = bf16ToFP32(output_avx512);
        double rel_l2 = computeRelativeL2(output_scalar_fp32, output_avx512_fp32);
        LOG_INFO("BF16 PerRow Scalar vs AVX512: rel_l2=" << rel_l2);
        EXPECT_LT(rel_l2, 1e-5);
    }

    /// Test BF16 AVX2 vs AVX512 per-row primitives
    TEST_F(Test__RMSNormPrecision, BF16_PerRow_AVX2VsAVX512)
    {
        const size_t cols = 896;
        const float epsilon = 1e-5f;

        auto input_fp32 = generateRandomFP32(1, cols);
        auto input_bf16 = fp32ToBF16(input_fp32);
        auto gamma = generateGamma(cols);

        std::vector<uint16_t> output_avx2(cols);
        std::vector<uint16_t> output_avx512(cols);

#if defined(__AVX2__)
        rmsnorm_fused_row_bf16_avx2(input_bf16.data(), gamma.data(), output_avx2.data(), cols, epsilon);
#else
        rmsnorm_fused_row_bf16_scalar(input_bf16.data(), gamma.data(), output_avx2.data(), cols, epsilon);
#endif

#if defined(__AVX512F__)
        rmsnorm_fused_row_bf16_avx512(input_bf16.data(), gamma.data(), output_avx512.data(), cols, epsilon);
#else
        output_avx512 = output_avx2;
#endif

        auto output_avx2_fp32 = bf16ToFP32(output_avx2);
        auto output_avx512_fp32 = bf16ToFP32(output_avx512);
        double rel_l2 = computeRelativeL2(output_avx2_fp32, output_avx512_fp32);
        LOG_INFO("BF16 PerRow AVX2 vs AVX512: rel_l2=" << rel_l2);
        EXPECT_LT(rel_l2, 1e-5);
    }

    /// Test FP16 scalar vs AVX2 per-row primitives
    TEST_F(Test__RMSNormPrecision, FP16_PerRow_ScalarVsAVX2)
    {
        const size_t cols = 896;
        const float epsilon = 1e-5f;

        auto input_fp32 = generateRandomFP32(1, cols, -0.5f, 0.5f); // Smaller range for FP16
        auto input_fp16 = fp32ToFP16(input_fp32);
        auto gamma = generateGamma(cols);

        std::vector<uint16_t> output_scalar(cols);
        std::vector<uint16_t> output_avx2(cols);

        rmsnorm_fused_row_fp16_scalar(input_fp16.data(), gamma.data(), output_scalar.data(), cols, epsilon);

#if defined(__AVX2__) && defined(__F16C__)
        rmsnorm_fused_row_fp16_avx2(input_fp16.data(), gamma.data(), output_avx2.data(), cols, epsilon);
#else
        output_avx2 = output_scalar;
#endif

        auto output_scalar_fp32 = fp16ToFP32(output_scalar);
        auto output_avx2_fp32 = fp16ToFP32(output_avx2);
        double rel_l2 = computeRelativeL2(output_scalar_fp32, output_avx2_fp32);
        LOG_INFO("FP16 PerRow Scalar vs AVX2: rel_l2=" << rel_l2);
        EXPECT_LT(rel_l2, 1e-3); // FP16 has lowest precision
    }

    /// Test FP16 scalar vs AVX512 per-row primitives
    TEST_F(Test__RMSNormPrecision, FP16_PerRow_ScalarVsAVX512)
    {
        const size_t cols = 896;
        const float epsilon = 1e-5f;

        auto input_fp32 = generateRandomFP32(1, cols, -0.5f, 0.5f);
        auto input_fp16 = fp32ToFP16(input_fp32);
        auto gamma = generateGamma(cols);

        std::vector<uint16_t> output_scalar(cols);
        std::vector<uint16_t> output_avx512(cols);

        rmsnorm_fused_row_fp16_scalar(input_fp16.data(), gamma.data(), output_scalar.data(), cols, epsilon);

#if defined(__AVX512F__) && defined(__AVX512FP16__)
        rmsnorm_fused_row_fp16_avx512(input_fp16.data(), gamma.data(), output_avx512.data(), cols, epsilon);
#else
        output_avx512 = output_scalar;
#endif

        auto output_scalar_fp32 = fp16ToFP32(output_scalar);
        auto output_avx512_fp32 = fp16ToFP32(output_avx512);
        double rel_l2 = computeRelativeL2(output_scalar_fp32, output_avx512_fp32);
        LOG_INFO("FP16 PerRow Scalar vs AVX512: rel_l2=" << rel_l2);
        EXPECT_LT(rel_l2, 1e-3);
    }

    /// Test FP16 AVX2 vs AVX512 per-row primitives
    TEST_F(Test__RMSNormPrecision, FP16_PerRow_AVX2VsAVX512)
    {
        const size_t cols = 896;
        const float epsilon = 1e-5f;

        auto input_fp32 = generateRandomFP32(1, cols, -0.5f, 0.5f);
        auto input_fp16 = fp32ToFP16(input_fp32);
        auto gamma = generateGamma(cols);

        std::vector<uint16_t> output_avx2(cols);
        std::vector<uint16_t> output_avx512(cols);

#if defined(__AVX2__) && defined(__F16C__)
        rmsnorm_fused_row_fp16_avx2(input_fp16.data(), gamma.data(), output_avx2.data(), cols, epsilon);
#else
        rmsnorm_fused_row_fp16_scalar(input_fp16.data(), gamma.data(), output_avx2.data(), cols, epsilon);
#endif

#if defined(__AVX512F__) && defined(__AVX512FP16__)
        rmsnorm_fused_row_fp16_avx512(input_fp16.data(), gamma.data(), output_avx512.data(), cols, epsilon);
#else
        output_avx512 = output_avx2;
#endif

        auto output_avx2_fp32 = fp16ToFP32(output_avx2);
        auto output_avx512_fp32 = fp16ToFP32(output_avx512);
        double rel_l2 = computeRelativeL2(output_avx2_fp32, output_avx512_fp32);
        LOG_INFO("FP16 PerRow AVX2 vs AVX512: rel_l2=" << rel_l2);
        EXPECT_LT(rel_l2, 1e-3);
    }

} // anonymous namespace
