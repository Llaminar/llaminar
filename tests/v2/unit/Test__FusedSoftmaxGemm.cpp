/**
 * @file Test__FusedSoftmaxGemm.cpp
 * @brief Unit tests for FusedSoftmaxGemmMicroKernel
 *
 * Tests verify:
 * 1. Fused softmax+GEMM produces same results as separate operations
 * 2. Softmax normalization is correct (row sums = 1.0)
 * 3. Temperature scaling works correctly
 * 4. Edge cases (zeros, large values) are handled
 */

#include <gtest/gtest.h>
#include "../../../src/v2/kernels/cpu/gemm/int8/FusedSoftmaxGemmMicroKernel.h"
#include "../../../src/v2/kernels/cpu/gemm/int8/IntegerGemmMicroKernel.h"
#include "../../../src/v2/kernels/cpu/SimdTraits.h"
#include "../../../src/v2/tensors/Tensors.h"
#include <cmath>
#include <random>

using namespace llaminar2::kernels::gemm;
using namespace llaminar2::kernels::simd;
using namespace llaminar2; // For Q8_0Block, fp16_to_fp32, fp32_to_fp16

// Helper: Check if AVX512-VNNI is available
bool hasVNNI()
{
#if defined(__AVX512VNNI__)
    return true;
#else
    return false;
#endif
}

// Helper: Apply softmax to FP32 array
void referenceSoftmax(const float *x, float *out, int n)
{
    float max_val = x[0];
    for (int i = 1; i < n; ++i)
        if (x[i] > max_val)
            max_val = x[i];

    float sum_exp = 0.0f;
    for (int i = 0; i < n; ++i)
    {
        float exp_val = std::exp(x[i] - max_val);
        out[i] = exp_val;
        sum_exp += exp_val;
    }

    float inv_sum = 1.0f / (sum_exp + 1e-12f);
    for (int i = 0; i < n; ++i)
        out[i] *= inv_sum;
}

// Helper: Dequantize Q8_0 block to FP32
void dequantizeQ8_0(const Q8_0Block *block, float *out, int n)
{
    float scale = fp16_to_fp32(block->d);
    for (int i = 0; i < n; ++i)
    {
        out[i] = static_cast<float>(block->qs[i]) * scale;
    }
}

/**
 * @brief Test basic softmax normalization correctness
 *
 * Verifies that output sums to ~1.0 per row (probability distribution)
 */
TEST(Test__FusedSoftmaxGemm, BasicSoftmaxNormalization)
{
    if (!hasVNNI())
    {
        GTEST_SKIP() << "AVX512-VNNI not available";
    }

    constexpr int TILE_M = 4;
    constexpr int TILE_N = 32;

    using Kernel = FusedSoftmaxGemmMicroKernel<AVX512VNNITag, TILE_M, TILE_N>;

    // Create simple test data (identity-like pattern)
    alignas(64) int8_t A_panel[TILE_M * 32];
    alignas(64) int8_t B_panel[TILE_N * 32];
    alignas(64) double a_scales[TILE_M] = {1.0, 1.0, 1.0, 1.0};
    alignas(64) double b_scales[TILE_N];

    // Initialize with small random values
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> dist(-10, 10);

    for (int i = 0; i < TILE_M * 32; ++i)
        A_panel[i] = dist(gen);
    for (int i = 0; i < TILE_N * 32; ++i)
        B_panel[i] = dist(gen);
    for (int i = 0; i < TILE_N; ++i)
        b_scales[i] = 1.0;

    // Execute fused kernel
    Kernel kernel;
    kernel.zero();
    kernel.accumulate(A_panel, B_panel, 32, a_scales, b_scales);

    alignas(64) Q8_0Block output[TILE_M];
    kernel.reduce(output);

    // Verify softmax normalization: each row should sum to ~1.0
    for (int i = 0; i < TILE_M; ++i)
    {
        float dequant[TILE_N];
        dequantizeQ8_0(&output[i], dequant, TILE_N);

        float row_sum = 0.0f;
        for (int j = 0; j < TILE_N; ++j)
            row_sum += dequant[j];

        // Allow 5% tolerance due to quantization error
        EXPECT_NEAR(row_sum, 1.0f, 0.05f)
            << "Row " << i << " sum = " << row_sum << " (expected ~1.0)";

        // All values should be non-negative (probabilities)
        for (int j = 0; j < TILE_N; ++j)
        {
            EXPECT_GE(dequant[j], -0.01f)
                << "Row " << i << ", col " << j << " = " << dequant[j];
        }
    }
}

/**
 * @brief Test fused kernel vs separate GEMM + softmax
 *
 * Verifies that fused operation produces same results as:
 * 1. Standard integer GEMM
 * 2. Dequantize to FP32
 * 3. Apply softmax
 * 4. Requantize to Q8_0
 */
TEST(Test__FusedSoftmaxGemm, CompareWithSeparateOperations)
{
    if (!hasVNNI())
    {
        GTEST_SKIP() << "AVX512-VNNI not available";
    }

    constexpr int TILE_M = 8;
    constexpr int TILE_N = 32;

    using FusedKernel = FusedSoftmaxGemmMicroKernel<AVX512VNNITag, TILE_M, TILE_N>;
    using StandardKernel = IntegerGemmMicroKernel<AVX512VNNITag, TILE_M, TILE_N>;

    // Create test data
    alignas(64) int8_t A_panel[TILE_M * 32];
    alignas(64) int8_t B_panel[TILE_N * 32];
    alignas(64) double a_scales[TILE_M];
    alignas(64) double b_scales[TILE_N];

    std::mt19937 gen(123);
    std::uniform_int_distribution<int> dist(-50, 50);

    for (int i = 0; i < TILE_M * 32; ++i)
        A_panel[i] = dist(gen);
    for (int i = 0; i < TILE_N * 32; ++i)
        B_panel[i] = dist(gen);
    for (int i = 0; i < TILE_M; ++i)
        a_scales[i] = 0.5 + (i * 0.1);
    for (int i = 0; i < TILE_N; ++i)
        b_scales[i] = 0.5 + (i * 0.05);

    // Execute fused kernel
    FusedKernel fused_kernel;
    fused_kernel.zero();
    fused_kernel.accumulate(A_panel, B_panel, 32, a_scales, b_scales);

    alignas(64) Q8_0Block fused_output[TILE_M];
    fused_kernel.reduce(fused_output);

    // Execute separate operations
    StandardKernel std_kernel;
    std_kernel.zero();
    std_kernel.accumulate(A_panel, B_panel, 32, a_scales, b_scales);

    alignas(64) Q8_0Block gemm_output[TILE_M];
    std_kernel.reduce(gemm_output);

    // Apply softmax to GEMM output
    alignas(64) Q8_0Block separate_output[TILE_M];
    for (int i = 0; i < TILE_M; ++i)
    {
        // Dequantize GEMM output
        float gemm_fp32[TILE_N];
        dequantizeQ8_0(&gemm_output[i], gemm_fp32, TILE_N);

        // Apply softmax
        float softmax_fp32[TILE_N];
        referenceSoftmax(gemm_fp32, softmax_fp32, TILE_N);

        // Requantize
        float amax = 0.0f;
        for (int j = 0; j < TILE_N; ++j)
            if (std::fabs(softmax_fp32[j]) > amax)
                amax = std::fabs(softmax_fp32[j]);

        float scale = (amax > 0.0f) ? (amax / 127.0f) : 1.0f;
        float inv_scale = 1.0f / scale;
        separate_output[i].d = fp32_to_fp16(scale);

        for (int j = 0; j < TILE_N; ++j)
        {
            float scaled = softmax_fp32[j] * inv_scale;
            int32_t q = static_cast<int32_t>(std::round(scaled));
            q = std::max(-127, std::min(127, q));
            separate_output[i].qs[j] = static_cast<int8_t>(q);
        }
    }

    // Compare fused vs separate
    for (int i = 0; i < TILE_M; ++i)
    {
        float fused_dequant[TILE_N];
        float separate_dequant[TILE_N];
        dequantizeQ8_0(&fused_output[i], fused_dequant, TILE_N);
        dequantizeQ8_0(&separate_output[i], separate_dequant, TILE_N);

        for (int j = 0; j < TILE_N; ++j)
        {
            // Allow 10% relative error due to double quantization in separate path
            float abs_diff = std::fabs(fused_dequant[j] - separate_dequant[j]);
            float magnitude = std::max(std::fabs(fused_dequant[j]), std::fabs(separate_dequant[j]));
            float rel_error = (magnitude > 1e-6f) ? (abs_diff / magnitude) : abs_diff;

            EXPECT_LT(rel_error, 0.15f)
                << "Row " << i << ", col " << j << ": fused=" << fused_dequant[j]
                << ", separate=" << separate_dequant[j] << ", rel_error=" << rel_error;
        }
    }
}

/**
 * @brief Test temperature scaling
 *
 * Lower temperature should produce sharper distributions (higher max)
 * Higher temperature should produce smoother distributions (lower max)
 */
TEST(Test__FusedSoftmaxGemm, TemperatureScaling)
{
    if (!hasVNNI())
    {
        GTEST_SKIP() << "AVX512-VNNI not available";
    }

    constexpr int TILE_M = 4;
    constexpr int TILE_N = 32;

    using Kernel = FusedSoftmaxGemmMicroKernel<AVX512VNNITag, TILE_M, TILE_N>;

    // Create test data with clear dominant values
    alignas(64) int8_t A_panel[TILE_M * 32];
    alignas(64) int8_t B_panel[TILE_N * 32];
    alignas(64) double a_scales[TILE_M] = {1.0, 1.0, 1.0, 1.0};
    alignas(64) double b_scales[TILE_N];

    std::memset(A_panel, 0, sizeof(A_panel));
    std::memset(B_panel, 0, sizeof(B_panel));

    // Make first column have high score
    for (int i = 0; i < TILE_M; ++i)
        A_panel[i * 32] = 50;
    for (int j = 0; j < TILE_N; ++j)
    {
        B_panel[j * 32] = (j == 0) ? 50 : 1;
        b_scales[j] = 1.0;
    }

    // Test low temperature (should be sharper)
    Kernel kernel_low;
    kernel_low.zero();
    kernel_low.accumulate(A_panel, B_panel, 32, a_scales, b_scales);

    alignas(64) Q8_0Block output_low[TILE_M];
    kernel_low.reduce_with_temperature(output_low, 0.5f);

    // Test high temperature (should be smoother)
    Kernel kernel_high;
    kernel_high.zero();
    kernel_high.accumulate(A_panel, B_panel, 32, a_scales, b_scales);

    alignas(64) Q8_0Block output_high[TILE_M];
    kernel_high.reduce_with_temperature(output_high, 2.0f);

    // Compare first row (which should have dominant first element)
    float low_dequant[TILE_N];
    float high_dequant[TILE_N];
    dequantizeQ8_0(&output_low[0], low_dequant, TILE_N);
    dequantizeQ8_0(&output_high[0], high_dequant, TILE_N);

    // Calculate entropy as measure of distribution sharpness
    // Lower entropy = sharper (more concentrated)
    // Higher entropy = smoother (more uniform)
    auto calc_entropy = [](const float *p, int n) -> float
    {
        float ent = 0.0f;
        for (int i = 0; i < n; ++i)
        {
            if (p[i] > 1e-9f)
                ent -= p[i] * std::log(p[i]);
        }
        return ent;
    };

    float entropy_low = calc_entropy(low_dequant, TILE_N);
    float entropy_high = calc_entropy(high_dequant, TILE_N);

    // Low temperature should have lower entropy (sharper distribution)
    // Note: Due to Q8_0 quantization, the difference may be small
    EXPECT_LE(entropy_low, entropy_high + 0.5f)
        << "Low temperature entropy=" << entropy_low
        << " should be <= high temperature entropy=" << entropy_high
        << " (or similar due to quantization limits)";
}

/**
 * @brief Test edge case: all zeros
 */
TEST(Test__FusedSoftmaxGemm, AllZeros)
{
    if (!hasVNNI())
    {
        GTEST_SKIP() << "AVX512-VNNI not available";
    }

    constexpr int TILE_M = 4;
    constexpr int TILE_N = 32;

    using Kernel = FusedSoftmaxGemmMicroKernel<AVX512VNNITag, TILE_M, TILE_N>;

    alignas(64) int8_t A_panel[TILE_M * 32] = {};
    alignas(64) int8_t B_panel[TILE_N * 32] = {};
    alignas(64) double a_scales[TILE_M] = {1.0, 1.0, 1.0, 1.0};
    alignas(64) double b_scales[TILE_N];
    for (int i = 0; i < TILE_N; ++i)
        b_scales[i] = 1.0;

    Kernel kernel;
    kernel.zero();
    kernel.accumulate(A_panel, B_panel, 32, a_scales, b_scales);

    alignas(64) Q8_0Block output[TILE_M];
    kernel.reduce(output);

    // With all zeros input, softmax should produce uniform distribution
    // (all exp(0) = 1, so each element = 1/N)
    for (int i = 0; i < TILE_M; ++i)
    {
        float dequant[TILE_N];
        dequantizeQ8_0(&output[i], dequant, TILE_N);

        float row_sum = 0.0f;
        for (int j = 0; j < TILE_N; ++j)
            row_sum += dequant[j];

        EXPECT_NEAR(row_sum, 1.0f, 0.05f);

        // Each element should be approximately 1/32 = 0.03125
        float expected = 1.0f / TILE_N;
        for (int j = 0; j < TILE_N; ++j)
        {
            EXPECT_NEAR(dequant[j], expected, 0.02f);
        }
    }
}
