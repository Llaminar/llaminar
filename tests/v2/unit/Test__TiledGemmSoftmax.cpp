/**
 * @file Test__TiledGemmSoftmax.cpp
 * @brief Unit tests for TiledGemmSoftmax kernel
 *
 * Tests:
 * 1. Correctness: Compare against sequential GEMM+Softmax
 * 2. Causal masking: Verify future tokens masked correctly
 * 3. Numerical stability: Test with extreme values
 * 4. Edge cases: Small/large dimensions, single token
 *
 * @author David Sanftenberg
 * @date January 2025
 */

#include "../../../src/v2/kernels/cpu/TiledGemmSoftmax.h"
#include "../../../src/v2/kernels/cpu/gemm/int8/FusedSoftmaxGemmMicroKernel.h"
#include "../../../src/v2/kernels/cpu/SimdTraits.h"
#include <gtest/gtest.h>
#include <random>
#include <cmath>
#include <vector>

using namespace llaminar2;
using namespace llaminar2::kernels;
using namespace llaminar2::kernels::gemm;
using namespace llaminar2::kernels::simd; // For AVX2Tag, AVX512Tag

namespace
{

    /**
     * @brief Helper to generate random FP32 data
     */
    void fill_random(float *data, size_t count, float mean = 0.0f, float stddev = 1.0f)
    {
        static std::mt19937 gen(42); // Fixed seed for reproducibility
        std::normal_distribution<float> dist(mean, stddev);
        for (size_t i = 0; i < count; ++i)
        {
            data[i] = dist(gen);
        }
    }

    /**
     * @brief Reference implementation: Sequential GEMM + Softmax
     */
    void reference_gemm_softmax(
        const float *Q, const float *K, float *weights,
        int m, int n, int d, float scale, bool causal)
    {
        // Step 1: Compute scores = Q @ K^T
        for (int i = 0; i < m; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                float sum = 0.0f;
                for (int k = 0; k < d; ++k)
                {
                    sum += Q[i * d + k] * K[j * d + k];
                }
                weights[i * n + j] = sum * scale;
            }
        }

        // Step 2: Apply causal mask (if needed)
        if (causal)
        {
            for (int i = 0; i < m; ++i)
            {
                for (int j = i + 1; j < n; ++j)
                {
                    weights[i * n + j] = -INFINITY;
                }
            }
        }

        // Step 3: Softmax
        for (int i = 0; i < m; ++i)
        {
            float *row = weights + i * n;

            // Find max
            float max_val = row[0];
            for (int j = 1; j < n; ++j)
            {
                max_val = std::max(max_val, row[j]);
            }

            // Exp and sum
            float sum = 0.0f;
            for (int j = 0; j < n; ++j)
            {
                row[j] = std::exp(row[j] - max_val);
                sum += row[j];
            }

            // Normalize
            float inv_sum = 1.0f / sum;
            for (int j = 0; j < n; ++j)
            {
                row[j] *= inv_sum;
            }
        }
    }

    /**
     * @brief Compare two FP32 arrays with relative tolerance
     */
    void compare_arrays(
        const float *expected, const float *actual, size_t count,
        float rel_tol = 1e-5f, float abs_tol = 1e-6f)
    {
        size_t mismatches = 0;
        float max_rel_error = 0.0f;

        for (size_t i = 0; i < count; ++i)
        {
            float exp_val = expected[i];
            float act_val = actual[i];

            float abs_error = std::abs(exp_val - act_val);
            float rel_error = abs_error / (std::abs(exp_val) + 1e-8f);

            max_rel_error = std::max(max_rel_error, rel_error);

            if (abs_error > abs_tol && rel_error > rel_tol)
            {
                mismatches++;
                if (mismatches <= 5)
                { // Print first 5 mismatches
                    std::cerr << "Mismatch at index " << i << ": "
                              << "expected=" << exp_val << ", "
                              << "actual=" << act_val << ", "
                              << "rel_error=" << rel_error << std::endl;
                }
            }
        }

        EXPECT_EQ(mismatches, 0) << "Found " << mismatches << " mismatches, "
                                 << "max_rel_error=" << max_rel_error;
    }

} // anonymous namespace

/**
 * @brief Test suite for TiledGemmSoftmax
 */
class TiledGemmSoftmaxTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Allocate test buffers (reused across tests)
        Q_.resize(512 * 128);        // Max: 512 tokens × 128 dim
        K_.resize(512 * 128);        // Max: 512 tokens × 128 dim
        expected_.resize(512 * 512); // Max: 512 × 512 attention matrix
        actual_.resize(512 * 512);
    }

    std::vector<float> Q_;
    std::vector<float> K_;
    std::vector<float> expected_;
    std::vector<float> actual_;
};

/**
 * @brief Basic correctness: Small matrix without causal masking
 */
TEST_F(TiledGemmSoftmaxTest, BasicCorrectness)
{
    constexpr int m = 32;  // 32 query tokens
    constexpr int n = 64;  // 64 key tokens (seq_len)
    constexpr int d = 128; // 128 model dimension
    constexpr float scale = 1.0f / std::sqrt(128.0f);

    // Generate random data
    fill_random(Q_.data(), m * d);
    fill_random(K_.data(), n * d);

    // Reference implementation
    reference_gemm_softmax(Q_.data(), K_.data(), expected_.data(),
                           m, n, d, scale, false);

    // Tiled implementation
    bool success = TiledGemmSoftmax<>::execute_fp32(
        Q_.data(), K_.data(), actual_.data(),
        m, n, d, scale, false, GemmOutputPrecision::FP32);

    ASSERT_TRUE(success);
    compare_arrays(expected_.data(), actual_.data(), m * n, 1e-4f, 1e-5f);
}

/**
 * @brief Causal masking: Verify future tokens masked to zero
 */
TEST_F(TiledGemmSoftmaxTest, CausalMasking)
{
    constexpr int m = 64; // 64 query tokens
    constexpr int n = 64; // 64 key tokens (square for causal)
    constexpr int d = 64; // 64 model dimension
    constexpr float scale = 1.0f / std::sqrt(64.0f);

    fill_random(Q_.data(), m * d);
    fill_random(K_.data(), n * d);

    // Reference with causal masking
    reference_gemm_softmax(Q_.data(), K_.data(), expected_.data(),
                           m, n, d, scale, true);

    // Tiled with causal masking
    bool success = TiledGemmSoftmax<>::execute_fp32(
        Q_.data(), K_.data(), actual_.data(),
        m, n, d, scale, true, GemmOutputPrecision::FP32);

    ASSERT_TRUE(success);
    compare_arrays(expected_.data(), actual_.data(), m * n, 1e-4f, 1e-5f);

    // Verify causal property: weights[i,j] ≈ 0 for j > i
    for (int i = 0; i < m; ++i)
    {
        for (int j = i + 1; j < n; ++j)
        {
            EXPECT_NEAR(actual_[i * n + j], 0.0f, 1e-6f)
                << "Causal mask violation at (" << i << "," << j << ")";
        }
    }
}

/**
 * @brief Large sequence: Test with realistic transformer size
 *
 * This test previously failed with heap-buffer-overflow when the micro-kernel
 * performed SIMD loads past the end of packed buffers. Fixed by adding SIMD
 * padding (kWidth elements) to A_packed and B_packed allocations.
 */
TEST_F(TiledGemmSoftmaxTest, LargeSequence)
{
    constexpr int m = 256; // 256 query tokens
    constexpr int n = 512; // 512 key tokens (long context)
    constexpr int d = 128; // 128 model dimension
    constexpr float scale = 1.0f / std::sqrt(128.0f);

    fill_random(Q_.data(), m * d, 0.0f, 0.1f); // Small variance for stability
    fill_random(K_.data(), n * d, 0.0f, 0.1f);

    reference_gemm_softmax(Q_.data(), K_.data(), expected_.data(),
                           m, n, d, scale, false);

    bool success = TiledGemmSoftmax<>::execute_fp32(
        Q_.data(), K_.data(), actual_.data(),
        m, n, d, scale, false, GemmOutputPrecision::FP32);

    ASSERT_TRUE(success);
    compare_arrays(expected_.data(), actual_.data(), m * n, 5e-4f, 1e-5f);
}

/**
 * @brief Edge case: Single query token
 */
TEST_F(TiledGemmSoftmaxTest, SingleToken)
{
    constexpr int m = 1;   // Single token
    constexpr int n = 128; // 128 key tokens
    constexpr int d = 64;  // 64 model dimension
    constexpr float scale = 1.0f / std::sqrt(64.0f);

    fill_random(Q_.data(), m * d);
    fill_random(K_.data(), n * d);

    reference_gemm_softmax(Q_.data(), K_.data(), expected_.data(),
                           m, n, d, scale, false);

    bool success = TiledGemmSoftmax<>::execute_fp32(
        Q_.data(), K_.data(), actual_.data(),
        m, n, d, scale, false, GemmOutputPrecision::FP32);

    ASSERT_TRUE(success);
    compare_arrays(expected_.data(), actual_.data(), m * n, 1e-5f, 1e-6f);
}

/**
 * @brief Numerical stability: Test with extreme values
 */
TEST_F(TiledGemmSoftmaxTest, NumericalStability)
{
    constexpr int m = 32;
    constexpr int n = 64;
    constexpr int d = 64;
    constexpr float scale = 1.0f / std::sqrt(64.0f);

    // Create data with large positive/negative values
    fill_random(Q_.data(), m * d, 0.0f, 10.0f); // Large variance
    fill_random(K_.data(), n * d, 0.0f, 10.0f);

    reference_gemm_softmax(Q_.data(), K_.data(), expected_.data(),
                           m, n, d, scale, false);

    bool success = TiledGemmSoftmax<>::execute_fp32(
        Q_.data(), K_.data(), actual_.data(),
        m, n, d, scale, false, GemmOutputPrecision::FP32);

    ASSERT_TRUE(success);
    compare_arrays(expected_.data(), actual_.data(), m * n, 1e-3f, 1e-5f);

    // Verify softmax properties: sum of each row ≈ 1
    for (int i = 0; i < m; ++i)
    {
        float row_sum = 0.0f;
        for (int j = 0; j < n; ++j)
        {
            row_sum += actual_[i * n + j];
        }
        EXPECT_NEAR(row_sum, 1.0f, 1e-5f)
            << "Softmax sum violation at row " << i;
    }
}

/**
 * @brief Tile boundary test: Verify correct handling at tile edges
 */
TEST_F(TiledGemmSoftmaxTest, TileBoundary)
{
    // Choose m to not be multiple of TILE_M (default 32)
    constexpr int m = 50;  // Not divisible by 32
    constexpr int n = 100; // Not divisible by 32
    constexpr int d = 64;
    constexpr float scale = 1.0f / std::sqrt(64.0f);

    fill_random(Q_.data(), m * d);
    fill_random(K_.data(), n * d);

    reference_gemm_softmax(Q_.data(), K_.data(), expected_.data(),
                           m, n, d, scale, false);

    bool success = TiledGemmSoftmax<>::execute_fp32(
        Q_.data(), K_.data(), actual_.data(),
        m, n, d, scale, false, GemmOutputPrecision::FP32);

    ASSERT_TRUE(success);
    compare_arrays(expected_.data(), actual_.data(), m * n, 1e-4f, 1e-5f);
}

/**
 * @brief Invalid input test: Ensure proper error handling
 */
TEST_F(TiledGemmSoftmaxTest, InvalidInput)
{
    constexpr int m = 32;
    constexpr int n = 64;
    constexpr int d = 64;
    constexpr float scale = 1.0f / std::sqrt(64.0f);

    // Test nullptr inputs
    bool success = TiledGemmSoftmax<>::execute_fp32(
        nullptr, K_.data(), actual_.data(),
        m, n, d, scale, false, GemmOutputPrecision::FP32);
    EXPECT_FALSE(success);

    success = TiledGemmSoftmax<>::execute_fp32(
        Q_.data(), nullptr, actual_.data(),
        m, n, d, scale, false, GemmOutputPrecision::FP32);
    EXPECT_FALSE(success);

    success = TiledGemmSoftmax<>::execute_fp32(
        Q_.data(), K_.data(), nullptr,
        m, n, d, scale, false, GemmOutputPrecision::FP32);
    EXPECT_FALSE(success);

    // Test invalid dimensions
    success = TiledGemmSoftmax<>::execute_fp32(
        Q_.data(), K_.data(), actual_.data(),
        0, n, d, scale, false, GemmOutputPrecision::FP32);
    EXPECT_FALSE(success);

    success = TiledGemmSoftmax<>::execute_fp32(
        Q_.data(), K_.data(), actual_.data(),
        m, 0, d, scale, false, GemmOutputPrecision::FP32);
    EXPECT_FALSE(success);

    success = TiledGemmSoftmax<>::execute_fp32(
        Q_.data(), K_.data(), actual_.data(),
        m, n, 0, scale, false, GemmOutputPrecision::FP32);
    EXPECT_FALSE(success);
}

/**
 * @brief AVX2 variant test: Ensure AVX2 path also works correctly
 */
TEST_F(TiledGemmSoftmaxTest, AVX2Variant)
{
    constexpr int m = 32;
    constexpr int n = 64;
    constexpr int d = 128;
    constexpr float scale = 1.0f / std::sqrt(128.0f);

    fill_random(Q_.data(), m * d);
    fill_random(K_.data(), n * d);

    reference_gemm_softmax(Q_.data(), K_.data(), expected_.data(),
                           m, n, d, scale, false);

    // Explicitly test AVX2 variant (even on AVX-512 systems)
    bool success = TiledGemmSoftmax<llaminar2::kernels::simd::AVX2Tag>::execute_fp32(
        Q_.data(), K_.data(), actual_.data(),
        m, n, d, scale, false, GemmOutputPrecision::FP32);

    ASSERT_TRUE(success);
    compare_arrays(expected_.data(), actual_.data(), m * n, 1e-4f, 1e-5f);
}

/**
 * @brief SIMD alignment test: Verify micro-kernel doesn't overrun buffers
 *
 * This test uses dimensions that are NOT multiples of SIMD width (16 for AVX-512)
 * to ensure the padding added to A_packed/B_packed prevents buffer overruns.
 *
 * The micro-kernel loads full SIMD vectors even at the end of k_panel, so buffers
 * must have SIMD_PADDING extra elements to prevent heap-buffer-overflow.
 */
TEST_F(TiledGemmSoftmaxTest, SimdAlignmentPadding)
{
    // Use d=130 (not multiple of 16) to test SIMD padding
    constexpr int m = 33;  // 33 query tokens (not multiple of TILE_M=32)
    constexpr int n = 65;  // 65 key tokens (not multiple of NR=6)
    constexpr int d = 130; // 130 dimension (not multiple of kWidth=16)
    constexpr float scale = 1.0f / std::sqrt(130.0f);

    // Resize buffers to accommodate larger d
    Q_.resize(m * d);
    K_.resize(n * d);
    expected_.resize(m * n);
    actual_.resize(m * n);

    fill_random(Q_.data(), m * d, 0.0f, 0.1f);
    fill_random(K_.data(), n * d, 0.0f, 0.1f);

    reference_gemm_softmax(Q_.data(), K_.data(), expected_.data(),
                           m, n, d, scale, false);

    // This should NOT crash with heap-buffer-overflow thanks to SIMD padding
    bool success = TiledGemmSoftmax<>::execute_fp32(
        Q_.data(), K_.data(), actual_.data(),
        m, n, d, scale, false, GemmOutputPrecision::FP32);

    ASSERT_TRUE(success);
    compare_arrays(expected_.data(), actual_.data(), m * n, 1e-4f, 1e-5f);
}

// ============================================================================
// Multi-Precision Tests (INT8, BF16)
// ============================================================================

/**
 * @brief Test BF16 input path with FP32 output
 */
TEST_F(TiledGemmSoftmaxTest, BF16Input_FP32Output)
{
    const int m = 8, n = 8, d = 16;
    const float scale = 1.0f / std::sqrt(static_cast<float>(d));

    std::vector<float> Q_fp32(m * d);
    std::vector<float> K_fp32(n * d);
    std::vector<uint16_t> Q_bf16(m * d);
    std::vector<uint16_t> K_bf16(n * d);
    std::vector<float> weights_tiled(m * n);
    std::vector<float> weights_ref(m * n);

    // Generate FP32 data
    fill_random(Q_fp32.data(), m * d);
    fill_random(K_fp32.data(), n * d);

    // Convert to BF16
    for (int i = 0; i < m * d; ++i)
        Q_bf16[i] = llaminar2::simd::fp32_to_bf16(Q_fp32[i]);
    for (int i = 0; i < n * d; ++i)
        K_bf16[i] = llaminar2::simd::fp32_to_bf16(K_fp32[i]);

    // Convert back to FP32 for reference (simulates BF16 precision loss)
    for (int i = 0; i < m * d; ++i)
        Q_fp32[i] = llaminar2::simd::bf16_to_fp32(Q_bf16[i]);
    for (int i = 0; i < n * d; ++i)
        K_fp32[i] = llaminar2::simd::bf16_to_fp32(K_bf16[i]);

    // Compute reference
    reference_gemm_softmax(Q_fp32.data(), K_fp32.data(), weights_ref.data(), m, n, d, scale, false);

    // Compute with TiledGemmSoftmax BF16 path
    auto kernel = createTiledGemmSoftmaxBF16();
    ASSERT_TRUE(kernel(Q_bf16.data(), K_bf16.data(), weights_tiled.data(), m, n, d, scale, false, GemmOutputPrecision::FP32));

    // Compare
    compare_arrays(weights_tiled.data(), weights_ref.data(), m * n, 1e-4f, 1e-5f);
}

/**
 * @brief Test FP32 input with BF16 output
 */
TEST_F(TiledGemmSoftmaxTest, FP32Input_BF16Output)
{
    const int m = 8, n = 8, d = 16;
    const float scale = 1.0f / std::sqrt(static_cast<float>(d));

    std::vector<float> Q(m * d);
    std::vector<float> K(n * d);
    std::vector<uint16_t> weights_bf16(m * n);
    std::vector<float> weights_ref(m * n);

    fill_random(Q.data(), m * d);
    fill_random(K.data(), n * d);

    // Compute reference
    reference_gemm_softmax(Q.data(), K.data(), weights_ref.data(), m, n, d, scale, false);

    // Compute with BF16 output
    auto kernel = createTiledGemmSoftmaxFP32();
    ASSERT_TRUE(kernel(Q.data(), K.data(), weights_bf16.data(), m, n, d, scale, false, GemmOutputPrecision::BF16));

    // Convert BF16 → FP32 and compare
    std::vector<float> weights_fp32(m * n);
    for (int i = 0; i < m * n; ++i)
    {
        weights_fp32[i] = llaminar2::simd::bf16_to_fp32(weights_bf16[i]);
    }

    // BF16 has ~3 decimal digits of precision, so use relaxed tolerance
    // Allow up to 0.4% relative error for BF16 output quantization
    compare_arrays(weights_fp32.data(), weights_ref.data(), m * n, 4e-3f, 2e-3f);
}

/**
 * @brief Test INT8 quantized attention (FP32 output)
 */
TEST_F(TiledGemmSoftmaxTest, INT8Input_FP32Output)
{
    const int m = 8, n = 8, d = 16;
    const float scale = 1.0f / std::sqrt(static_cast<float>(d));
    const float q_scale = 0.01f; // Query dequant scale
    const float k_scale = 0.01f; // Key dequant scale

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(-127, 127);

    std::vector<int8_t> Q_int8(m * d);
    std::vector<int8_t> K_int8(n * d);
    std::vector<float> Q_fp32(m * d);
    std::vector<float> K_fp32(n * d);
    std::vector<float> weights_tiled(m * n);
    std::vector<float> weights_ref(m * n);

    // Generate INT8 data
    for (auto &v : Q_int8)
        v = static_cast<int8_t>(dist(rng));
    for (auto &v : K_int8)
        v = static_cast<int8_t>(dist(rng));

    // Dequantize for reference
    for (int i = 0; i < m * d; ++i)
        Q_fp32[i] = static_cast<float>(Q_int8[i]) * q_scale;
    for (int i = 0; i < n * d; ++i)
        K_fp32[i] = static_cast<float>(K_int8[i]) * k_scale;

    // Compute reference
    reference_gemm_softmax(Q_fp32.data(), K_fp32.data(), weights_ref.data(), m, n, d, scale, false);

    // Compute with TiledGemmSoftmax INT8 path
    auto kernel = createTiledGemmSoftmaxINT8();
    ASSERT_TRUE(kernel(Q_int8.data(), K_int8.data(), weights_tiled.data(), m, n, d, scale, q_scale, k_scale, false, GemmOutputPrecision::FP32));

    // INT8 has lower precision due to quantization
    compare_arrays(weights_tiled.data(), weights_ref.data(), m * n, 1e-3f, 1e-4f);
}

/**
 * @brief Test INT8 with BF16 output (full quantization pipeline)
 */
TEST_F(TiledGemmSoftmaxTest, INT8Input_BF16Output)
{
    const int m = 8, n = 8, d = 16;
    const float scale = 1.0f / std::sqrt(static_cast<float>(d));
    const float q_scale = 0.01f;
    const float k_scale = 0.01f;

    std::mt19937 rng(123);
    std::uniform_int_distribution<int> dist(-127, 127);

    std::vector<int8_t> Q_int8(m * d);
    std::vector<int8_t> K_int8(n * d);
    std::vector<float> Q_fp32(m * d);
    std::vector<float> K_fp32(n * d);
    std::vector<uint16_t> weights_bf16(m * n);
    std::vector<float> weights_ref(m * n);

    // Generate INT8 data
    for (auto &v : Q_int8)
        v = static_cast<int8_t>(dist(rng));
    for (auto &v : K_int8)
        v = static_cast<int8_t>(dist(rng));

    // Dequantize for reference
    for (int i = 0; i < m * d; ++i)
        Q_fp32[i] = static_cast<float>(Q_int8[i]) * q_scale;
    for (int i = 0; i < n * d; ++i)
        K_fp32[i] = static_cast<float>(K_int8[i]) * k_scale;

    // Compute reference
    reference_gemm_softmax(Q_fp32.data(), K_fp32.data(), weights_ref.data(), m, n, d, scale, false);

    // Compute with INT8 → BF16 path
    auto kernel = createTiledGemmSoftmaxINT8();
    ASSERT_TRUE(kernel(Q_int8.data(), K_int8.data(), weights_bf16.data(), m, n, d, scale, q_scale, k_scale, false, GemmOutputPrecision::BF16));

    // Convert BF16 → FP32 for comparison
    std::vector<float> weights_fp32(m * n);
    for (int i = 0; i < m * n; ++i)
    {
        weights_fp32[i] = llaminar2::simd::bf16_to_fp32(weights_bf16[i]);
    }

    // Relaxed tolerance for INT8 + BF16 quantization
    compare_arrays(weights_fp32.data(), weights_ref.data(), m * n, 5e-3f, 1e-3f);
}

/**
 * @brief Test INT8 with causal masking
 */
TEST_F(TiledGemmSoftmaxTest, INT8_CausalMasking)
{
    const int m = 8, n = 8, d = 16;
    const float scale = 1.0f / std::sqrt(static_cast<float>(d));
    const float q_scale = 0.01f;
    const float k_scale = 0.01f;

    std::mt19937 rng(456);
    std::uniform_int_distribution<int> dist(-127, 127);

    std::vector<int8_t> Q_int8(m * d);
    std::vector<int8_t> K_int8(n * d);
    std::vector<float> weights(m * n);

    for (auto &v : Q_int8)
        v = static_cast<int8_t>(dist(rng));
    for (auto &v : K_int8)
        v = static_cast<int8_t>(dist(rng));

    // Compute with causal masking
    auto kernel = createTiledGemmSoftmaxINT8();
    ASSERT_TRUE(kernel(Q_int8.data(), K_int8.data(), weights.data(), m, n, d, scale, q_scale, k_scale, true, GemmOutputPrecision::FP32));

    // Verify causal structure: weights[i,j] should be 0 for j > i
    for (int i = 0; i < m; ++i)
    {
        for (int j = i + 1; j < n; ++j)
        {
            EXPECT_NEAR(weights[i * n + j], 0.0f, 1e-6f) << "INT8 causal mask failed at [" << i << "," << j << "]";
        }
    }
}

/**
 * @brief Test softmax sum = 1.0 invariant for all precision modes
 */
TEST_F(TiledGemmSoftmaxTest, SoftmaxSumInvariant_AllPrecisions)
{
    const int m = 8, n = 8, d = 16;
    const float scale = 1.0f;

    std::mt19937 rng(789);
    std::uniform_int_distribution<int> int_dist(-127, 127);

    // Test FP32
    {
        std::vector<float> Q(m * d), K(n * d), weights(m * n);
        fill_random(Q.data(), m * d);
        fill_random(K.data(), n * d);

        auto kernel = createTiledGemmSoftmaxFP32();
        ASSERT_TRUE(kernel(Q.data(), K.data(), weights.data(), m, n, d, scale, false, GemmOutputPrecision::FP32));

        for (int i = 0; i < m; ++i)
        {
            float row_sum = 0.0f;
            for (int j = 0; j < n; ++j)
                row_sum += weights[i * n + j];
            EXPECT_NEAR(row_sum, 1.0f, 1e-5f) << "FP32: Row " << i << " sum != 1.0";
        }
    }

    // Test INT8
    {
        std::vector<int8_t> Q(m * d), K(n * d);
        std::vector<float> weights(m * n);

        for (auto &v : Q)
            v = static_cast<int8_t>(int_dist(rng));
        for (auto &v : K)
            v = static_cast<int8_t>(int_dist(rng));

        auto kernel = createTiledGemmSoftmaxINT8();
        ASSERT_TRUE(kernel(Q.data(), K.data(), weights.data(), m, n, d, scale, 0.01f, 0.01f, false, GemmOutputPrecision::FP32));

        for (int i = 0; i < m; ++i)
        {
            float row_sum = 0.0f;
            for (int j = 0; j < n; ++j)
                row_sum += weights[i * n + j];
            EXPECT_NEAR(row_sum, 1.0f, 1e-5f) << "INT8: Row " << i << " sum != 1.0";
        }
    }
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
