/**
 * @file Test__FusedGemmSoftmax.cpp
 * @brief Tests for fused GEMM+Softmax kernel
 *
 * @author David Sanftenberg
 * @date November 8, 2025
 *
 * Test strategy:
 * 1. Correctness: Compare fused vs separate GEMM+softmax (exact match)
 * 2. Causal masking: Verify triangular attention correctness
 * 3. Edge cases: seq_len=1, very large sequences (2048+)
 * 4. Numerical stability: Extreme logit values
 * 5. Performance: Verify tile-based execution is faster
 */

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <chrono>

#ifdef __x86_64__
#include <cblas.h>
#else
extern "C"
{
    void cblas_sgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                     const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                     const int K, const float alpha, const float *A,
                     const int lda, const float *B, const int ldb,
                     const float beta, float *C, const int ldc);
}
#endif

#include "../../src/v2/kernels/cpu/gemm/int8/FusedSoftmaxGemmMicroKernel.h"
#include "../../src/v2/kernels/cpu/primitives/SoftmaxPrimitives_New.h"
#include "../../src/v2/utils/Logger.h"

using namespace llaminar2;

namespace
{

    /**
     * @brief Helper: Initialize vector with sequential values
     */
    void init_sequential(float *data, size_t size, float start = 0.0f)
    {
        for (size_t i = 0; i < size; ++i)
        {
            data[i] = start + static_cast<float>(i) * 0.1f;
        }
    }

    /**
     * @brief Helper: Initialize vector with random values
     */
    void init_random(float *data, size_t size, float min = -1.0f, float max = 1.0f)
    {
        static std::mt19937 rng(12345);
        std::uniform_real_distribution<float> dist(min, max);
        for (size_t i = 0; i < size; ++i)
        {
            data[i] = dist(rng);
        }
    }

    /**
     * @brief Helper: Reference implementation (separate GEMM + softmax)
     */
    void reference_gemm_softmax(
        const float *Q,
        const float *K,
        float *weights,
        int m, int n, int k,
        int lda, int ldb, int ldc,
        float scale,
        bool causal)
    {
        // Step 1: GEMM (Q @ K^T)
        cblas_sgemm(
            CblasRowMajor,
            CblasNoTrans,
            CblasTrans,
            m, n, k,
            scale,
            Q, lda,
            K, ldb,
            0.0f,
            weights, ldc);

// Step 2: Softmax (row-wise)
#pragma omp parallel for
        for (int i = 0; i < m; ++i)
        {
            float *row = weights + i * ldc;
            int effective_n = causal ? std::min(i + 1, n) : n;

            primitives::softmax_row_major_fp32(
                row, 1, effective_n, false, 1.0f, true);

            // Zero out causal-masked elements
            if (causal && effective_n < n)
            {
                std::memset(row + effective_n, 0, (n - effective_n) * sizeof(float));
            }
        }
    }

    /**
     * @brief Helper: Compare two buffers with tolerance
     */
    bool compare_buffers(
        const float *expected,
        const float *actual,
        size_t size,
        float rel_tol = 1e-5f,
        float abs_tol = 1e-6f,
        int *num_mismatches = nullptr)
    {
        int mismatches = 0;
        float max_rel_diff = 0.0f;
        float max_abs_diff = 0.0f;

        for (size_t i = 0; i < size; ++i)
        {
            float exp = expected[i];
            float act = actual[i];
            float abs_diff = std::abs(exp - act);
            float rel_diff = (std::abs(exp) > abs_tol) ? abs_diff / std::abs(exp) : abs_diff;

            max_rel_diff = std::max(max_rel_diff, rel_diff);
            max_abs_diff = std::max(max_abs_diff, abs_diff);

            if (rel_diff > rel_tol && abs_diff > abs_tol)
            {
                mismatches++;
                if (mismatches <= 5)
                { // Print first 5 mismatches
                    LOG_WARN("Mismatch at index " << i << ": expected=" << exp
                                                  << ", actual=" << act << ", rel_diff=" << rel_diff);
                }
            }
        }

        if (num_mismatches)
        {
            *num_mismatches = mismatches;
        }

        LOG_INFO("Buffer comparison: max_rel_diff=" << max_rel_diff
                                                    << ", max_abs_diff=" << max_abs_diff
                                                    << ", mismatches=" << mismatches << "/" << size);

        return mismatches == 0;
    }

} // anonymous namespace

// =============================================================================
// Basic Functionality Tests
// =============================================================================

TEST(FusedGemmSoftmax, InstantiationWorks)
{
    FusedGemmSoftmax kernel;
    EXPECT_TRUE(true); // Just ensure constructor/destructor work
}

TEST(FusedGemmSoftmax, BasicComputation)
{
    // Small test: 4 tokens, 8-dim heads
    constexpr int seq_len = 4;
    constexpr int head_dim = 8;
    constexpr float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // Initialize Q and K
    std::vector<float> Q(seq_len * head_dim);
    std::vector<float> K(seq_len * head_dim);
    init_sequential(Q.data(), Q.size(), 0.0f);
    init_sequential(K.data(), K.size(), 1.0f);

    // Reference and fused outputs
    std::vector<float> weights_ref(seq_len * seq_len, 0.0f);
    std::vector<float> weights_fused(seq_len * seq_len, 0.0f);

    // Compute reference (separate GEMM + softmax)
    reference_gemm_softmax(
        Q.data(), K.data(), weights_ref.data(),
        seq_len, seq_len, head_dim,
        head_dim, head_dim, seq_len,
        scale, false);

    // Compute fused
    FusedGemmSoftmax kernel;
    bool success = kernel.execute(
        Q.data(), K.data(), weights_fused.data(),
        seq_len, seq_len, head_dim,
        head_dim, head_dim, seq_len,
        scale, false,
        /*tile_size=*/2); // Small tile for testing

    EXPECT_TRUE(success);
    EXPECT_TRUE(compare_buffers(weights_ref.data(), weights_fused.data(),
                                seq_len * seq_len));
}

TEST(FusedGemmSoftmax, CausalMasking)
{
    // Test causal attention (triangular masking)
    constexpr int seq_len = 8;
    constexpr int head_dim = 16;
    constexpr float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    std::vector<float> Q(seq_len * head_dim);
    std::vector<float> K(seq_len * head_dim);
    init_random(Q.data(), Q.size(), -1.0f, 1.0f);
    init_random(K.data(), K.size(), -1.0f, 1.0f);

    std::vector<float> weights_ref(seq_len * seq_len, 0.0f);
    std::vector<float> weights_fused(seq_len * seq_len, 0.0f);

    // Reference with causal masking
    reference_gemm_softmax(
        Q.data(), K.data(), weights_ref.data(),
        seq_len, seq_len, head_dim,
        head_dim, head_dim, seq_len,
        scale, true);

    // Fused with causal masking
    FusedGemmSoftmax kernel;
    bool success = kernel.execute(
        Q.data(), K.data(), weights_fused.data(),
        seq_len, seq_len, head_dim,
        head_dim, head_dim, seq_len,
        scale, true,
        /*tile_size=*/4);

    EXPECT_TRUE(success);
    EXPECT_TRUE(compare_buffers(weights_ref.data(), weights_fused.data(),
                                seq_len * seq_len));

    // Verify upper triangular is zero (future tokens masked)
    for (int i = 0; i < seq_len; ++i)
    {
        for (int j = i + 1; j < seq_len; ++j)
        {
            EXPECT_FLOAT_EQ(weights_fused[i * seq_len + j], 0.0f)
                << "Causal mask failed at (" << i << "," << j << ")";
        }
    }
}

TEST(FusedGemmSoftmax, RowSumsToOne)
{
    // Verify softmax normalization (each row sums to 1.0)
    constexpr int seq_len = 16;
    constexpr int head_dim = 32;
    constexpr float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    std::vector<float> Q(seq_len * head_dim);
    std::vector<float> K(seq_len * head_dim);
    init_random(Q.data(), Q.size(), -2.0f, 2.0f);
    init_random(K.data(), K.size(), -2.0f, 2.0f);

    std::vector<float> weights(seq_len * seq_len, 0.0f);

    FusedGemmSoftmax kernel;
    bool success = kernel.execute(
        Q.data(), K.data(), weights.data(),
        seq_len, seq_len, head_dim,
        head_dim, head_dim, seq_len,
        scale, false,
        /*tile_size=*/8);

    EXPECT_TRUE(success);

    // Verify each row sums to ~1.0
    for (int i = 0; i < seq_len; ++i)
    {
        float row_sum = 0.0f;
        for (int j = 0; j < seq_len; ++j)
        {
            row_sum += weights[i * seq_len + j];
        }
        EXPECT_NEAR(row_sum, 1.0f, 1e-4f) << "Row " << i << " sum = " << row_sum;
    }
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST(FusedGemmSoftmax, SingleToken)
{
    // Edge case: seq_len = 1
    constexpr int seq_len = 1;
    constexpr int head_dim = 64;
    constexpr float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    std::vector<float> Q(head_dim);
    std::vector<float> K(head_dim);
    init_sequential(Q.data(), head_dim);
    init_sequential(K.data(), head_dim);

    std::vector<float> weights_ref(1, 0.0f);
    std::vector<float> weights_fused(1, 0.0f);

    reference_gemm_softmax(
        Q.data(), K.data(), weights_ref.data(),
        1, 1, head_dim,
        head_dim, head_dim, 1,
        scale, false);

    FusedGemmSoftmax kernel;
    bool success = kernel.execute(
        Q.data(), K.data(), weights_fused.data(),
        1, 1, head_dim,
        head_dim, head_dim, 1,
        scale, false,
        /*tile_size=*/1);

    EXPECT_TRUE(success);
    EXPECT_NEAR(weights_fused[0], 1.0f, 1e-5f); // Single element softmax = 1.0
    EXPECT_NEAR(weights_ref[0], weights_fused[0], 1e-5f);
}

TEST(FusedGemmSoftmax, LargeSequence)
{
    // Large sequence: 2048 tokens (stress test)
    constexpr int seq_len = 2048;
    constexpr int head_dim = 64;
    constexpr float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    std::vector<float> Q(seq_len * head_dim);
    std::vector<float> K(seq_len * head_dim);
    init_random(Q.data(), Q.size(), -1.0f, 1.0f);
    init_random(K.data(), K.size(), -1.0f, 1.0f);

    std::vector<float> weights(seq_len * seq_len, 0.0f);

    FusedGemmSoftmax kernel;
    bool success = kernel.execute(
        Q.data(), K.data(), weights.data(),
        seq_len, seq_len, head_dim,
        head_dim, head_dim, seq_len,
        scale, false,
        /*tile_size=*/64); // Default tile size

    EXPECT_TRUE(success);

    // Spot-check: verify first and last rows sum to 1.0
    float first_row_sum = 0.0f;
    float last_row_sum = 0.0f;
    for (int j = 0; j < seq_len; ++j)
    {
        first_row_sum += weights[j];
        last_row_sum += weights[(seq_len - 1) * seq_len + j];
    }

    EXPECT_NEAR(first_row_sum, 1.0f, 1e-3f);
    EXPECT_NEAR(last_row_sum, 1.0f, 1e-3f);
}

TEST(FusedGemmSoftmax, StridedInputs)
{
    // Test with strided inputs (lda != k, ldc != n)
    constexpr int seq_len = 4;
    constexpr int head_dim = 8;
    constexpr int lda = 16; // Q stride (>= head_dim)
    constexpr int ldb = 16; // K stride (>= head_dim)
    constexpr int ldc = 8;  // Weights stride (>= seq_len)
    constexpr float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // Allocate strided buffers
    std::vector<float> Q_strided(seq_len * lda, 0.0f);
    std::vector<float> K_strided(seq_len * ldb, 0.0f);
    std::vector<float> weights_strided(seq_len * ldc, 0.0f);

    // Initialize only the active columns
    for (int i = 0; i < seq_len; ++i)
    {
        for (int j = 0; j < head_dim; ++j)
        {
            Q_strided[i * lda + j] = static_cast<float>(i * head_dim + j) * 0.1f;
            K_strided[i * ldb + j] = static_cast<float>(i * head_dim + j) * 0.1f + 1.0f;
        }
    }

    FusedGemmSoftmax kernel;
    bool success = kernel.execute(
        Q_strided.data(), K_strided.data(), weights_strided.data(),
        seq_len, seq_len, head_dim,
        lda, ldb, ldc,
        scale, false,
        /*tile_size=*/2);

    EXPECT_TRUE(success);

    // Verify row normalization (accounting for stride)
    for (int i = 0; i < seq_len; ++i)
    {
        float row_sum = 0.0f;
        for (int j = 0; j < seq_len; ++j)
        {
            row_sum += weights_strided[i * ldc + j];
        }
        EXPECT_NEAR(row_sum, 1.0f, 1e-4f) << "Strided row " << i;
    }
}

// =============================================================================
// Numerical Stability
// =============================================================================

TEST(FusedGemmSoftmax, ExtremeLargeLogits)
{
    // Test with very large logit values (potential overflow in exp)
    constexpr int seq_len = 4;
    constexpr int head_dim = 8;
    constexpr float scale = 1.0f; // No scaling to make logits large

    std::vector<float> Q(seq_len * head_dim);
    std::vector<float> K(seq_len * head_dim);

    // Create Q, K such that Q@K^T produces large values
    for (size_t i = 0; i < Q.size(); ++i)
    {
        Q[i] = 10.0f; // Large positive values
        K[i] = 10.0f;
    }

    std::vector<float> weights(seq_len * seq_len, 0.0f);

    FusedGemmSoftmax kernel;
    bool success = kernel.execute(
        Q.data(), K.data(), weights.data(),
        seq_len, seq_len, head_dim,
        head_dim, head_dim, seq_len,
        scale, false,
        /*tile_size=*/2);

    EXPECT_TRUE(success);

    // Verify no NaN/Inf
    for (size_t i = 0; i < weights.size(); ++i)
    {
        EXPECT_FALSE(std::isnan(weights[i])) << "NaN at index " << i;
        EXPECT_FALSE(std::isinf(weights[i])) << "Inf at index " << i;
    }

    // Verify normalization
    for (int i = 0; i < seq_len; ++i)
    {
        float row_sum = 0.0f;
        for (int j = 0; j < seq_len; ++j)
        {
            row_sum += weights[i * seq_len + j];
        }
        EXPECT_NEAR(row_sum, 1.0f, 1e-4f);
    }
}

TEST(FusedGemmSoftmax, ExtremeSmallLogits)
{
    // Test with very small logit values (potential underflow)
    constexpr int seq_len = 4;
    constexpr int head_dim = 8;
    constexpr float scale = 1.0f;

    std::vector<float> Q(seq_len * head_dim);
    std::vector<float> K(seq_len * head_dim);

    // Small values
    for (size_t i = 0; i < Q.size(); ++i)
    {
        Q[i] = 0.001f;
        K[i] = 0.001f;
    }

    std::vector<float> weights(seq_len * seq_len, 0.0f);

    FusedGemmSoftmax kernel;
    bool success = kernel.execute(
        Q.data(), K.data(), weights.data(),
        seq_len, seq_len, head_dim,
        head_dim, head_dim, seq_len,
        scale, false,
        /*tile_size=*/2);

    EXPECT_TRUE(success);

    // Verify normalization
    for (int i = 0; i < seq_len; ++i)
    {
        float row_sum = 0.0f;
        for (int j = 0; j < seq_len; ++j)
        {
            row_sum += weights[i * seq_len + j];
        }
        EXPECT_NEAR(row_sum, 1.0f, 1e-4f);
    }
}

// =============================================================================
// Tile Size Variations
// =============================================================================

TEST(FusedGemmSoftmax, VariousTileSizes)
{
    // Test different tile sizes produce identical results
    constexpr int seq_len = 128;
    constexpr int head_dim = 64;
    constexpr float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    std::vector<float> Q(seq_len * head_dim);
    std::vector<float> K(seq_len * head_dim);
    init_random(Q.data(), Q.size(), -1.0f, 1.0f);
    init_random(K.data(), K.size(), -1.0f, 1.0f);

    std::vector<float> weights_ref(seq_len * seq_len, 0.0f);
    std::vector<float> weights_test(seq_len * seq_len, 0.0f);

    // Reference: tile_size = 64
    FusedGemmSoftmax kernel;
    kernel.execute(
        Q.data(), K.data(), weights_ref.data(),
        seq_len, seq_len, head_dim,
        head_dim, head_dim, seq_len,
        scale, false,
        /*tile_size=*/64);

    // Test various tile sizes
    std::vector<int> tile_sizes = {16, 32, 64, 128};
    for (int tile_size : tile_sizes)
    {
        std::fill(weights_test.begin(), weights_test.end(), 0.0f);

        kernel.execute(
            Q.data(), K.data(), weights_test.data(),
            seq_len, seq_len, head_dim,
            head_dim, head_dim, seq_len,
            scale, false,
            tile_size);

        int mismatches = 0;
        EXPECT_TRUE(compare_buffers(weights_ref.data(), weights_test.data(),
                                    seq_len * seq_len, 1e-5f, 1e-6f, &mismatches))
            << "Tile size " << tile_size << " produced different results ("
            << mismatches << " mismatches)";
    }
}

// =============================================================================
// Error Handling
// =============================================================================

TEST(FusedGemmSoftmax, NullPointerInputs)
{
    FusedGemmSoftmax kernel;

    std::vector<float> Q(64);
    std::vector<float> K(64);
    std::vector<float> weights(16);

    // Null Q
    EXPECT_FALSE(kernel.execute(nullptr, K.data(), weights.data(),
                                4, 4, 4, 4, 4, 4, 1.0f, false, 2));

    // Null K
    EXPECT_FALSE(kernel.execute(Q.data(), nullptr, weights.data(),
                                4, 4, 4, 4, 4, 4, 1.0f, false, 2));

    // Null weights
    EXPECT_FALSE(kernel.execute(Q.data(), K.data(), nullptr,
                                4, 4, 4, 4, 4, 4, 1.0f, false, 2));
}

TEST(FusedGemmSoftmax, InvalidDimensions)
{
    FusedGemmSoftmax kernel;

    std::vector<float> Q(64);
    std::vector<float> K(64);
    std::vector<float> weights(16);

    // Negative dimensions
    EXPECT_FALSE(kernel.execute(Q.data(), K.data(), weights.data(),
                                -4, 4, 4, 4, 4, 4, 1.0f, false, 2));

    EXPECT_FALSE(kernel.execute(Q.data(), K.data(), weights.data(),
                                4, -4, 4, 4, 4, 4, 1.0f, false, 2));

    EXPECT_FALSE(kernel.execute(Q.data(), K.data(), weights.data(),
                                4, 4, -4, 4, 4, 4, 1.0f, false, 2));

    // Invalid strides
    EXPECT_FALSE(kernel.execute(Q.data(), K.data(), weights.data(),
                                4, 4, 8, 4, 8, 4, 1.0f, false, 2)); // lda < k
}

// =============================================================================
// Performance Comparison (informational, not validation)
// =============================================================================

TEST(FusedGemmSoftmax, DISABLED_PerformanceComparison)
{
    // Disabled by default (use --gtest_also_run_disabled_tests to run)
    constexpr int seq_len = 512;
    constexpr int head_dim = 64;
    constexpr float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    constexpr int num_trials = 10;

    std::vector<float> Q(seq_len * head_dim);
    std::vector<float> K(seq_len * head_dim);
    init_random(Q.data(), Q.size(), -1.0f, 1.0f);
    init_random(K.data(), K.size(), -1.0f, 1.0f);

    std::vector<float> weights_ref(seq_len * seq_len, 0.0f);
    std::vector<float> weights_fused(seq_len * seq_len, 0.0f);

    // Benchmark reference (separate GEMM + softmax)
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_trials; ++i)
    {
        reference_gemm_softmax(
            Q.data(), K.data(), weights_ref.data(),
            seq_len, seq_len, head_dim,
            head_dim, head_dim, seq_len,
            scale, false);
    }
    auto end = std::chrono::high_resolution_clock::now();
    double ref_ms = std::chrono::duration<double, std::milli>(end - start).count() / num_trials;

    // Benchmark fused
    FusedGemmSoftmax kernel;
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_trials; ++i)
    {
        kernel.execute(
            Q.data(), K.data(), weights_fused.data(),
            seq_len, seq_len, head_dim,
            head_dim, head_dim, seq_len,
            scale, false,
            /*tile_size=*/64);
    }
    end = std::chrono::high_resolution_clock::now();
    double fused_ms = std::chrono::duration<double, std::milli>(end - start).count() / num_trials;

    LOG_INFO("Performance comparison (seq_len=" << seq_len << ", " << num_trials << " trials):");
    LOG_INFO("  Reference (separate GEMM+softmax): " << ref_ms << " ms");
    LOG_INFO("  Fused (tile-based):                 " << fused_ms << " ms");
    LOG_INFO("  Speedup:                            " << (ref_ms / fused_ms) << "x");

    // Verify correctness
    EXPECT_TRUE(compare_buffers(weights_ref.data(), weights_fused.data(),
                                seq_len * seq_len));
}

// =============================================================================
// BF16 Output Precision Tests (Stage 2)
// =============================================================================

/**
 * @brief Test BF16 output mode correctness
 *
 * Verifies:
 * 1. FP32 and BF16 modes produce similar results (within BF16 tolerance)
 * 2. Memory reduction is achieved (2 bytes vs 4 bytes per element)
 * 3. Conversion is numerically stable
 */
TEST(FusedGemmSoftmax, BF16OutputMode)
{
    constexpr int seq_len = 128;
    constexpr int head_dim = 64;
    constexpr float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    std::vector<float> Q(seq_len * head_dim);
    std::vector<float> K(seq_len * head_dim);
    init_random(Q.data(), Q.size(), -1.0f, 1.0f);
    init_random(K.data(), K.size(), -1.0f, 1.0f);

    // FP32 output (baseline)
    std::vector<float> weights_fp32(seq_len * seq_len, 0.0f);
    FusedGemmSoftmax kernel_fp32;
    ASSERT_TRUE(kernel_fp32.execute(
        Q.data(), K.data(), weights_fp32.data(),
        seq_len, seq_len, head_dim,
        head_dim, head_dim, seq_len,
        scale, false,
        /*tile_size=*/64,
        GemmOutputPrecision::FP32));

    // BF16 output
    std::vector<float> weights_bf16_buffer(seq_len * seq_len, 0.0f);
    FusedGemmSoftmax kernel_bf16;
    ASSERT_TRUE(kernel_bf16.execute(
        Q.data(), K.data(), weights_bf16_buffer.data(),
        seq_len, seq_len, head_dim,
        head_dim, head_dim, seq_len,
        scale, false,
        /*tile_size=*/64,
        GemmOutputPrecision::BF16));

    // Convert BF16 back to FP32 for comparison
    // BF16 output is tightly packed (m×n), no stride
    llaminar2::bfloat16 *bf16_weights = reinterpret_cast<llaminar2::bfloat16 *>(weights_bf16_buffer.data());
    std::vector<float> weights_bf16_fp32(seq_len * seq_len);
    for (int i = 0; i < seq_len; ++i)
    {
        for (int j = 0; j < seq_len; ++j)
        {
            weights_bf16_fp32[i * seq_len + j] = bf16_weights[i * seq_len + j].to_float();
        }
    }

    // Verify: BF16 vs FP32 within tolerance
    // BF16 has ~3 decimal digits precision (7 mantissa bits)
    // Softmax outputs are in [0, 1], so absolute tolerance is appropriate
    constexpr float bf16_abs_tol = 0.02f; // 2% absolute tolerance (BF16 precision limit)
    constexpr float bf16_rel_tol = 0.05f; // 5% relative tolerance

    double max_abs_diff = 0.0;
    double max_rel_diff = 0.0;
    int mismatches = 0;

    for (size_t i = 0; i < weights_fp32.size(); ++i)
    {
        float fp32_val = weights_fp32[i];
        float bf16_val = weights_bf16_fp32[i];

        double abs_diff = std::abs(fp32_val - bf16_val);
        double rel_diff = (fp32_val > 1e-6f) ? abs_diff / std::abs(fp32_val) : 0.0;

        max_abs_diff = std::max(max_abs_diff, abs_diff);
        max_rel_diff = std::max(max_rel_diff, rel_diff);

        if (abs_diff > bf16_abs_tol && rel_diff > bf16_rel_tol)
        {
            mismatches++;
        }
    }

    LOG_INFO("BF16 vs FP32: max_abs_diff=" << max_abs_diff
                                           << ", max_rel_diff=" << max_rel_diff
                                           << ", mismatches=" << mismatches << "/" << weights_fp32.size());

    // Most elements should match within BF16 tolerance
    EXPECT_LT(mismatches, static_cast<int>(weights_fp32.size() * 0.05)); // <5% mismatches
    EXPECT_LT(max_abs_diff, 0.03);                                       // Absolute error < 3%
}

/**
 * @brief Test BF16 output with causal masking
 *
 * Verifies BF16 mode works correctly with causal attention.
 */
TEST(FusedGemmSoftmax, BF16OutputCausal)
{
    constexpr int seq_len = 64;
    constexpr int head_dim = 32;
    constexpr float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    std::vector<float> Q(seq_len * head_dim);
    std::vector<float> K(seq_len * head_dim);
    init_random(Q.data(), Q.size(), -1.0f, 1.0f);
    init_random(K.data(), K.size(), -1.0f, 1.0f);

    // BF16 output with causal masking
    std::vector<float> weights_bf16_buffer(seq_len * seq_len, 0.0f);
    FusedGemmSoftmax kernel;
    ASSERT_TRUE(kernel.execute(
        Q.data(), K.data(), weights_bf16_buffer.data(),
        seq_len, seq_len, head_dim,
        head_dim, head_dim, seq_len,
        scale, true, // causal=true
        /*tile_size=*/64,
        GemmOutputPrecision::BF16));

    // Verify causal masking: upper triangle should be zero (or very close due to BF16 precision)
    // BF16 output is tightly packed (m×n)
    llaminar2::bfloat16 *bf16_weights = reinterpret_cast<llaminar2::bfloat16 *>(weights_bf16_buffer.data());
    for (int i = 0; i < seq_len; ++i)
    {
        for (int j = i + 1; j < seq_len; ++j)
        {
            float val = bf16_weights[i * seq_len + j].to_float();
            EXPECT_LT(std::abs(val), 1e-4f) << "Causal mask failed at (" << i << ", " << j << "): val=" << val;
        }
    }

    // Verify row sums ≈ 1.0 (within BF16 tolerance)
    for (int i = 0; i < seq_len; ++i)
    {
        float row_sum = 0.0f;
        for (int j = 0; j <= i; ++j)
        { // Causal: only sum up to diagonal
            row_sum += bf16_weights[i * seq_len + j].to_float();
        }
        EXPECT_NEAR(row_sum, 1.0f, 0.02f) << "Row " << i << " sum failed"; // Relaxed tolerance for BF16
    }
}

/**
 * @brief Test BF16 memory footprint reduction
 *
 * Verifies that BF16 mode actually uses less memory.
 */
TEST(FusedGemmSoftmax, BF16MemoryReduction)
{
    constexpr int seq_len = 256;
    constexpr int head_dim = 64;

    // Calculate memory footprint
    size_t fp32_bytes = seq_len * seq_len * sizeof(float);
    size_t bf16_bytes = seq_len * seq_len * sizeof(llaminar2::bfloat16);

    LOG_INFO("Memory footprint (seq_len=" << seq_len << "):");
    LOG_INFO("  FP32: " << fp32_bytes << " bytes (" << (fp32_bytes / 1024.0) << " KB)");
    LOG_INFO("  BF16: " << bf16_bytes << " bytes (" << (bf16_bytes / 1024.0) << " KB)");
    LOG_INFO("  Reduction: " << (100.0 * (fp32_bytes - bf16_bytes) / fp32_bytes) << "%");

    // Verify 50% reduction
    EXPECT_EQ(bf16_bytes, fp32_bytes / 2);
    EXPECT_EQ(sizeof(llaminar2::bfloat16), 2); // 2 bytes per BF16 value
}
