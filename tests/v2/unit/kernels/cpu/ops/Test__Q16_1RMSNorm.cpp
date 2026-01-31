/**
 * @file Test__Q16_1RMSNorm.cpp
 * @brief Unit tests for Q16_1 → FP32 RMSNorm implementation
 * @author David Sanftenberg
 * @date 2025-06
 *
 * Tests the Q16_1 RMSNorm kernel that reads Q16_1 quantized input,
 * dequantizes on-the-fly during normalization, and outputs FP32.
 * This is part of the typed residual stream optimization (Phase 4).
 */

#include <gtest/gtest.h>
#include "kernels/cpu/primitives/RMSNormPrimitives.h"
#include "tensors/BlockStructures.h"
#include <vector>
#include <cmath>
#include <cstring>
#include <random>

using namespace llaminar2;
using namespace llaminar2::primitives;

class Q16_1RMSNormTest : public ::testing::Test
{
protected:
    std::mt19937 rng_{42};

    // Helper: Quantize FP32 to Q16_1 blocks
    void quantize_fp32_to_q16_1(const float *fp32, Q16_1Block *blocks, size_t count)
    {
        size_t n_blocks = count / 32;
        for (size_t b = 0; b < n_blocks; ++b)
        {
            const float *src = fp32 + b * 32;
            Q16_1Block &blk = blocks[b];

            // Find max for scale
            float max_abs = 0.0f;
            for (int i = 0; i < 32; ++i)
            {
                max_abs = std::max(max_abs, std::abs(src[i]));
            }

            float scale = (max_abs > 0.0f) ? max_abs / 32767.0f : 1.0f / 32767.0f;
            float inv_scale = 32767.0f / std::max(max_abs, 1e-8f);

            blk.d = scale;

            int32_t sum = 0;
            for (int i = 0; i < 32; ++i)
            {
                int32_t q = static_cast<int32_t>(std::round(src[i] * inv_scale));
                q = std::max(-32767, std::min(32767, q));
                blk.qs[i] = static_cast<int16_t>(q);
                sum += blk.qs[i];
            }
            blk.sum_qs = sum;
        }
    }

    // Helper: Dequantize Q16_1 to FP32
    void dequantize_q16_1_to_fp32(const Q16_1Block *blocks, float *fp32, size_t count)
    {
        size_t n_blocks = count / 32;
        for (size_t b = 0; b < n_blocks; ++b)
        {
            const Q16_1Block &blk = blocks[b];
            float *dst = fp32 + b * 32;
            float scale = blk.d;

            for (int i = 0; i < 32; ++i)
            {
                dst[i] = scale * static_cast<float>(blk.qs[i]);
            }
        }
    }

    // Reference FP32 RMSNorm
    void reference_rmsnorm_fp32(
        const float *input,
        const float *gamma,
        float *output,
        size_t cols,
        float eps)
    {
        // Compute sum of squares
        double sum_sq = 0.0;
        for (size_t i = 0; i < cols; ++i)
        {
            sum_sq += static_cast<double>(input[i]) * input[i];
        }

        // Compute RMS
        float rms = std::sqrt(static_cast<float>(sum_sq / cols) + eps);
        float inv_rms = 1.0f / rms;

        // Apply normalization and gamma
        for (size_t i = 0; i < cols; ++i)
        {
            output[i] = input[i] * inv_rms * gamma[i];
        }
    }

    // Compute accuracy metrics
    struct AccuracyMetrics
    {
        double max_diff;
        double mean_diff;
        double rel_rmse;
        double cosine_similarity;
    };

    AccuracyMetrics compute_metrics(const float *a, const float *b, size_t count)
    {
        AccuracyMetrics m{};
        double sum_diff = 0.0;
        double sum_diff_sq = 0.0;
        double sum_a_sq = 0.0;
        double sum_b_sq = 0.0;
        double dot = 0.0;

        for (size_t i = 0; i < count; ++i)
        {
            double diff = std::abs(a[i] - b[i]);
            m.max_diff = std::max(m.max_diff, diff);
            sum_diff += diff;
            sum_diff_sq += diff * diff;
            sum_a_sq += static_cast<double>(a[i]) * a[i];
            sum_b_sq += static_cast<double>(b[i]) * b[i];
            dot += static_cast<double>(a[i]) * b[i];
        }

        m.mean_diff = sum_diff / count;
        m.rel_rmse = std::sqrt(sum_diff_sq / sum_a_sq);
        double norm_a = std::sqrt(sum_a_sq);
        double norm_b = std::sqrt(sum_b_sq);
        m.cosine_similarity = (norm_a > 0 && norm_b > 0) ? dot / (norm_a * norm_b) : 0.0;

        return m;
    }
};

// =============================================================================
// Basic Functionality Tests
// =============================================================================

TEST_F(Q16_1RMSNormTest, ScalarBasic)
{
    const size_t cols = 64; // 2 blocks
    const float eps = 1e-5f;

    // Create input data
    std::vector<float> input_fp32(cols);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < cols; ++i)
    {
        input_fp32[i] = dist(rng_);
    }

    // Create gamma weights
    std::vector<float> gamma(cols);
    std::uniform_real_distribution<float> gamma_dist(0.5f, 1.5f);
    for (size_t i = 0; i < cols; ++i)
    {
        gamma[i] = gamma_dist(rng_);
    }

    // Quantize input to Q16_1
    std::vector<Q16_1Block> input_q16(cols / 32);
    quantize_fp32_to_q16_1(input_fp32.data(), input_q16.data(), cols);

    // Run Q16_1 RMSNorm (scalar)
    std::vector<float> output_q16(cols);
    RMSNormExecOptions opts;
    opts.force_scalar = true;
    rmsnorm_q16_1_fp32_row_scalar(input_q16.data(), gamma.data(), output_q16.data(), cols, eps);

    // Run FP32 reference (using dequantized input for fair comparison)
    std::vector<float> dequant_fp32(cols);
    dequantize_q16_1_to_fp32(input_q16.data(), dequant_fp32.data(), cols);
    std::vector<float> output_ref(cols);
    reference_rmsnorm_fp32(dequant_fp32.data(), gamma.data(), output_ref.data(), cols, eps);

    // Compare outputs
    auto metrics = compute_metrics(output_q16.data(), output_ref.data(), cols);

    EXPECT_LT(metrics.max_diff, 1e-5) << "Max diff too large: " << metrics.max_diff;
    EXPECT_GT(metrics.cosine_similarity, 0.999999) << "Cosine similarity too low: " << metrics.cosine_similarity;
}

TEST_F(Q16_1RMSNormTest, AVX512Basic)
{
    const size_t cols = 64; // 2 blocks
    const float eps = 1e-5f;

    // Create input data
    std::vector<float> input_fp32(cols);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < cols; ++i)
    {
        input_fp32[i] = dist(rng_);
    }

    // Create gamma weights
    std::vector<float> gamma(cols);
    std::uniform_real_distribution<float> gamma_dist(0.5f, 1.5f);
    for (size_t i = 0; i < cols; ++i)
    {
        gamma[i] = gamma_dist(rng_);
    }

    // Quantize input to Q16_1
    std::vector<Q16_1Block> input_q16(cols / 32);
    quantize_fp32_to_q16_1(input_fp32.data(), input_q16.data(), cols);

    // Run Q16_1 RMSNorm (AVX512)
    std::vector<float> output_q16(cols);
    rmsnorm_q16_1_fp32_row_avx512(input_q16.data(), gamma.data(), output_q16.data(), cols, eps);

    // Run FP32 reference (using dequantized input for fair comparison)
    std::vector<float> dequant_fp32(cols);
    dequantize_q16_1_to_fp32(input_q16.data(), dequant_fp32.data(), cols);
    std::vector<float> output_ref(cols);
    reference_rmsnorm_fp32(dequant_fp32.data(), gamma.data(), output_ref.data(), cols, eps);

    // Compare outputs
    auto metrics = compute_metrics(output_q16.data(), output_ref.data(), cols);

    EXPECT_LT(metrics.max_diff, 1e-5) << "Max diff too large: " << metrics.max_diff;
    EXPECT_GT(metrics.cosine_similarity, 0.999999) << "Cosine similarity too low: " << metrics.cosine_similarity;
}

TEST_F(Q16_1RMSNormTest, ScalarVsAVX512Parity)
{
    const size_t cols = 896; // Qwen 0.5B hidden dim
    const float eps = 1e-5f;

    // Create input data
    std::vector<float> input_fp32(cols);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    for (size_t i = 0; i < cols; ++i)
    {
        input_fp32[i] = dist(rng_);
    }

    // Create gamma weights
    std::vector<float> gamma(cols);
    std::uniform_real_distribution<float> gamma_dist(0.5f, 1.5f);
    for (size_t i = 0; i < cols; ++i)
    {
        gamma[i] = gamma_dist(rng_);
    }

    // Quantize input to Q16_1
    std::vector<Q16_1Block> input_q16(cols / 32);
    quantize_fp32_to_q16_1(input_fp32.data(), input_q16.data(), cols);

    // Run scalar
    std::vector<float> output_scalar(cols);
    rmsnorm_q16_1_fp32_row_scalar(input_q16.data(), gamma.data(), output_scalar.data(), cols, eps);

    // Run AVX512
    std::vector<float> output_avx512(cols);
    rmsnorm_q16_1_fp32_row_avx512(input_q16.data(), gamma.data(), output_avx512.data(), cols, eps);

    // Compare: should be bitwise identical or very close
    auto metrics = compute_metrics(output_scalar.data(), output_avx512.data(), cols);

    EXPECT_LT(metrics.max_diff, 1e-6) << "Scalar vs AVX512 max diff: " << metrics.max_diff;
    EXPECT_GT(metrics.cosine_similarity, 0.9999999) << "Scalar vs AVX512 cosine: " << metrics.cosine_similarity;
}

// =============================================================================
// Multi-Row Fused Tests
// =============================================================================

TEST_F(Q16_1RMSNormTest, FusedMultiRow)
{
    const size_t rows = 4;
    const size_t cols = 896;
    const float eps = 1e-5f;

    // Create input data
    std::vector<float> input_fp32(rows * cols);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    for (size_t i = 0; i < rows * cols; ++i)
    {
        input_fp32[i] = dist(rng_);
    }

    // Create gamma weights (shared across rows)
    std::vector<float> gamma(cols);
    std::uniform_real_distribution<float> gamma_dist(0.5f, 1.5f);
    for (size_t i = 0; i < cols; ++i)
    {
        gamma[i] = gamma_dist(rng_);
    }

    // Quantize input to Q16_1
    std::vector<Q16_1Block> input_q16(rows * cols / 32);
    quantize_fp32_to_q16_1(input_fp32.data(), input_q16.data(), rows * cols);

    // Run fused
    std::vector<float> output_fused(rows * cols);
    RMSNormExecOptions opts;
    rmsnorm_q16_1_fp32_fused(input_q16.data(), gamma.data(), output_fused.data(), rows, cols, eps, opts);

    // Compute reference per row
    std::vector<float> output_ref(rows * cols);
    std::vector<float> dequant_fp32(rows * cols);
    dequantize_q16_1_to_fp32(input_q16.data(), dequant_fp32.data(), rows * cols);
    for (size_t r = 0; r < rows; ++r)
    {
        reference_rmsnorm_fp32(
            dequant_fp32.data() + r * cols,
            gamma.data(),
            output_ref.data() + r * cols,
            cols, eps);
    }

    // Compare
    auto metrics = compute_metrics(output_fused.data(), output_ref.data(), rows * cols);

    EXPECT_LT(metrics.max_diff, 1e-5) << "Fused max diff: " << metrics.max_diff;
    EXPECT_GT(metrics.cosine_similarity, 0.999999) << "Fused cosine: " << metrics.cosine_similarity;
}

// =============================================================================
// Accuracy vs Pure FP32 (quantization impact)
// =============================================================================

TEST_F(Q16_1RMSNormTest, Q16_1_vs_FP32_RMSNorm_Accuracy)
{
    // This test measures the accuracy loss due to Q16_1 quantization
    // when compared to pure FP32 RMSNorm on the same (unquantized) input
    const size_t cols = 896;
    const float eps = 1e-5f;

    // Create input data
    std::vector<float> input_fp32(cols);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    for (size_t i = 0; i < cols; ++i)
    {
        input_fp32[i] = dist(rng_);
    }

    // Create gamma weights
    std::vector<float> gamma(cols);
    std::uniform_real_distribution<float> gamma_dist(0.5f, 1.5f);
    for (size_t i = 0; i < cols; ++i)
    {
        gamma[i] = gamma_dist(rng_);
    }

    // Pure FP32 reference
    std::vector<float> output_fp32_ref(cols);
    reference_rmsnorm_fp32(input_fp32.data(), gamma.data(), output_fp32_ref.data(), cols, eps);

    // Quantize input to Q16_1 and run Q16_1 RMSNorm
    std::vector<Q16_1Block> input_q16(cols / 32);
    quantize_fp32_to_q16_1(input_fp32.data(), input_q16.data(), cols);
    std::vector<float> output_q16(cols);
    rmsnorm_q16_1_fp32_row_avx512(input_q16.data(), gamma.data(), output_q16.data(), cols, eps);

    // Compare
    auto metrics = compute_metrics(output_q16.data(), output_fp32_ref.data(), cols);

    // Print metrics for debugging
    std::cout << "\nQ16_1 RMSNorm vs FP32 RMSNorm accuracy:" << std::endl;
    std::cout << "  Max diff:    " << metrics.max_diff << std::endl;
    std::cout << "  Mean diff:   " << metrics.mean_diff << std::endl;
    std::cout << "  Rel RMSE:    " << metrics.rel_rmse << std::endl;
    std::cout << "  Cosine sim:  " << metrics.cosine_similarity << std::endl;

    // Thresholds: 10× margin over expected values for Q16_1
    // Q16_1 has 16-bit precision (1 part in 32767 ≈ 3e-5 quantization error)
    EXPECT_LT(metrics.max_diff, 5e-4) << "Max diff exceeds threshold";
    EXPECT_GT(metrics.cosine_similarity, 0.99999) << "Cosine similarity below threshold";
    EXPECT_LT(metrics.rel_rmse, 5e-4) << "Relative RMSE exceeds threshold";
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST_F(Q16_1RMSNormTest, ZeroInput)
{
    const size_t cols = 64;
    const float eps = 1e-5f;

    // Zero input
    std::vector<float> input_fp32(cols, 0.0f);
    std::vector<float> gamma(cols, 1.0f);

    // Quantize (will have scale near zero)
    std::vector<Q16_1Block> input_q16(cols / 32);
    quantize_fp32_to_q16_1(input_fp32.data(), input_q16.data(), cols);

    // Run RMSNorm
    std::vector<float> output(cols);
    rmsnorm_q16_1_fp32_row_avx512(input_q16.data(), gamma.data(), output.data(), cols, eps);

    // Output should be all zeros (or very close due to epsilon)
    for (size_t i = 0; i < cols; ++i)
    {
        EXPECT_NEAR(output[i], 0.0f, 1e-3) << "Non-zero output at index " << i;
    }
}

TEST_F(Q16_1RMSNormTest, ConstantInput)
{
    const size_t cols = 64;
    const float eps = 1e-5f;
    const float const_val = 1.5f;

    // Constant input
    std::vector<float> input_fp32(cols, const_val);
    std::vector<float> gamma(cols, 1.0f);

    // Quantize
    std::vector<Q16_1Block> input_q16(cols / 32);
    quantize_fp32_to_q16_1(input_fp32.data(), input_q16.data(), cols);

    // Run RMSNorm
    std::vector<float> output(cols);
    rmsnorm_q16_1_fp32_row_avx512(input_q16.data(), gamma.data(), output.data(), cols, eps);

    // With gamma=1, constant input should normalize to ~1.0 (or close)
    // RMS of constant = constant, so normalized = constant / constant = 1.0
    for (size_t i = 0; i < cols; ++i)
    {
        EXPECT_NEAR(output[i], 1.0f, 1e-3) << "Unexpected output at index " << i;
    }
}

TEST_F(Q16_1RMSNormTest, LargeHiddenDim)
{
    const size_t cols = 3584; // Qwen 7B hidden dim
    const float eps = 1e-5f;

    // Create input data
    std::vector<float> input_fp32(cols);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    for (size_t i = 0; i < cols; ++i)
    {
        input_fp32[i] = dist(rng_);
    }

    // Create gamma weights
    std::vector<float> gamma(cols);
    std::uniform_real_distribution<float> gamma_dist(0.5f, 1.5f);
    for (size_t i = 0; i < cols; ++i)
    {
        gamma[i] = gamma_dist(rng_);
    }

    // Quantize
    std::vector<Q16_1Block> input_q16(cols / 32);
    quantize_fp32_to_q16_1(input_fp32.data(), input_q16.data(), cols);

    // Run
    std::vector<float> output_q16(cols);
    rmsnorm_q16_1_fp32_row_avx512(input_q16.data(), gamma.data(), output_q16.data(), cols, eps);

    // Reference
    std::vector<float> dequant_fp32(cols);
    dequantize_q16_1_to_fp32(input_q16.data(), dequant_fp32.data(), cols);
    std::vector<float> output_ref(cols);
    reference_rmsnorm_fp32(dequant_fp32.data(), gamma.data(), output_ref.data(), cols, eps);

    // Compare
    auto metrics = compute_metrics(output_q16.data(), output_ref.data(), cols);

    EXPECT_LT(metrics.max_diff, 1e-5) << "Large hidden dim max diff: " << metrics.max_diff;
    EXPECT_GT(metrics.cosine_similarity, 0.999999) << "Large hidden dim cosine: " << metrics.cosine_similarity;
}
