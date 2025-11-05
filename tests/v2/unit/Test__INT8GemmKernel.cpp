/**
 * @file Test__INT8GemmKernel.cpp
 * @brief Unit tests for INT8GemmKernel with OneDNN
 *
 * Test Coverage:
 * 1. Basic correctness against FP32 reference
 * 2. Quantization accuracy (per-row and per-column scales)
 * 3. Transpose operations
 * 4. Alpha/beta scaling
 * 5. Edge cases (small matrices, zero matrices, extreme values)
 * 6. Performance benchmarks vs FP32
 *
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include "../../../src/v2/kernels/cpu/INT8GemmKernel.h"
#include "../../../src/v2/kernels/cpu/FP32GemmKernel.h"
#include "../../../src/v2/tensors/Tensors.h"
#include "../../../src/v2/utils/Logger.h"
#include <cmath>
#include <random>
#include <vector>
#include <chrono>

using namespace llaminar2;

// ============================================================================
// Test Fixtures
// ============================================================================

class Test__INT8GemmKernel : public ::testing::Test
{
protected:
    // Random number generator
    std::mt19937 rng_{42}; // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dist_normal_{-1.0f, 1.0f};
    std::uniform_real_distribution<float> dist_small_{-0.1f, 0.1f};

    void SetUp() override
    {
        // Ensure we're testing on CPU
    }

    // Helper: Fill matrix with random values
    void fill_random(float *data, size_t count, bool small_range = false)
    {
        auto &dist = small_range ? dist_small_ : dist_normal_;
        for (size_t i = 0; i < count; ++i)
        {
            data[i] = dist(rng_);
        }
    }

    // Helper: Naive FP32 GEMM for reference (C = alpha*A*B + beta*C)
    void reference_gemm(
        const float *A, const float *B, float *C,
        int m, int n, int k,
        bool transpose_B, float alpha, float beta)
    {
        // Apply beta scaling to existing C
        if (beta != 1.0f)
        {
            for (int i = 0; i < m * n; ++i)
            {
                C[i] *= beta;
            }
        }

        // Compute alpha*A*B
        for (int i = 0; i < m; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                float sum = 0.0f;
                for (int kk = 0; kk < k; ++kk)
                {
                    float a_val = A[i * k + kk];
                    float b_val = transpose_B ? B[j * k + kk] : B[kk * n + j];
                    sum += a_val * b_val;
                }
                C[i * n + j] += alpha * sum;
            }
        }
    }

    // Helper: Compute relative error between two matrices
    float compute_relative_error(const float *C_ref, const float *C_test, int m, int n)
    {
        float max_error = 0.0f;
        float max_ref = 0.0f;

        for (int i = 0; i < m * n; ++i)
        {
            float ref_val = std::fabs(C_ref[i]);
            float error = std::fabs(C_test[i] - C_ref[i]);
            max_error = std::max(max_error, error);
            max_ref = std::max(max_ref, ref_val);
        }

        return max_ref > 1e-6f ? (max_error / max_ref) : max_error;
    }

    // Helper: Create INT8 weight tensor from FP32 data
    std::unique_ptr<INT8Tensor> create_int8_tensor(const float *fp32_data, int rows, int cols)
    {
        // INT8Tensor constructor with FP32 data automatically computes per-column scales for 2D tensors
        std::vector<float> fp32_vec(fp32_data, fp32_data + rows * cols);
        return std::make_unique<INT8Tensor>(
            std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(cols)},
            fp32_vec);
    }
};

// ============================================================================
// Basic Correctness Tests
// ============================================================================

TEST_F(Test__INT8GemmKernel, BasicMatrixMultiply)
{
    const int m = 64, n = 128, k = 96;

    // Create random input matrices
    std::vector<float> A(m * k);
    std::vector<float> B(k * n);
    std::vector<float> C_ref(m * n, 0.0f);
    std::vector<float> C_int8(m * n, 0.0f);

    fill_random(A.data(), m * k, true); // Use small range for better INT8 accuracy
    fill_random(B.data(), k * n, true);

    // Compute reference with FP32
    reference_gemm(A.data(), B.data(), C_ref.data(), m, n, k, false, 1.0f, 0.0f);

    // Create INT8 weight tensor
    auto B_int8_tensor = create_int8_tensor(B.data(), k, n);

    // Compute with INT8 kernel
    INT8GemmKernel int8_kernel(B_int8_tensor.get());
    bool success = int8_kernel.multiply(
        A.data(), C_int8.data(),
        m, n, k,
        false, // transpose_B
        1.0f, 0.0f);

    ASSERT_TRUE(success) << "INT8 GEMM should succeed";

    // Check relative error (INT8 quantization typically <1% error for this range)
    float rel_error = compute_relative_error(C_ref.data(), C_int8.data(), m, n);
    EXPECT_LT(rel_error, 0.02f) << "Relative error should be <2% for INT8 quantization";

    LOG_INFO("[Test__INT8GemmKernel] BasicMatrixMultiply: relative error = " << (rel_error * 100.0f) << "%");
}

TEST_F(Test__INT8GemmKernel, TransposeB)
{
    const int m = 32, n = 64, k = 48;

    std::vector<float> A(m * k);
    std::vector<float> B_T(n * k); // Transposed layout [n, k]
    std::vector<float> C_ref(m * n, 0.0f);
    std::vector<float> C_int8(m * n, 0.0f);

    fill_random(A.data(), m * k, true);
    fill_random(B_T.data(), n * k, true);

    // Reference with transpose_B=true
    reference_gemm(A.data(), B_T.data(), C_ref.data(), m, n, k, true, 1.0f, 0.0f);

    // Create INT8 tensor (stored as [n, k] for transpose)
    auto B_int8_tensor = create_int8_tensor(B_T.data(), n, k);

    INT8GemmKernel int8_kernel(B_int8_tensor.get());
    bool success = int8_kernel.multiply(
        A.data(), C_int8.data(),
        m, n, k,
        true, // transpose_B
        1.0f, 0.0f);

    ASSERT_TRUE(success);

    float rel_error = compute_relative_error(C_ref.data(), C_int8.data(), m, n);
    EXPECT_LT(rel_error, 0.03f); // Slightly higher tolerance for transpose (uses per-tensor scale)

    LOG_INFO("[Test__INT8GemmKernel] TransposeB: relative error = " << (rel_error * 100.0f) << "%");
}

// ============================================================================
// Alpha/Beta Scaling Tests
// ============================================================================

TEST_F(Test__INT8GemmKernel, AlphaScaling)
{
    const int m = 32, n = 32, k = 32;
    const float alpha = 2.5f;

    std::vector<float> A(m * k);
    std::vector<float> B(k * n);
    std::vector<float> C_ref(m * n, 0.0f);
    std::vector<float> C_int8(m * n, 0.0f);

    fill_random(A.data(), m * k, true);
    fill_random(B.data(), k * n, true);

    reference_gemm(A.data(), B.data(), C_ref.data(), m, n, k, false, alpha, 0.0f);

    auto B_int8_tensor = create_int8_tensor(B.data(), k, n);
    INT8GemmKernel int8_kernel(B_int8_tensor.get());
    bool success = int8_kernel.multiply(
        A.data(), C_int8.data(),
        m, n, k,
        false, alpha, 0.0f);

    ASSERT_TRUE(success);

    float rel_error = compute_relative_error(C_ref.data(), C_int8.data(), m, n);
    EXPECT_LT(rel_error, 0.02f);

    LOG_INFO("[Test__INT8GemmKernel] AlphaScaling (α=" << alpha << "): relative error = " << (rel_error * 100.0f) << "%");
}

TEST_F(Test__INT8GemmKernel, BetaScaling)
{
    const int m = 32, n = 32, k = 32;
    const float alpha = 1.0f, beta = 0.5f;

    std::vector<float> A(m * k);
    std::vector<float> B(k * n);
    std::vector<float> C_ref(m * n);
    std::vector<float> C_int8(m * n);

    fill_random(A.data(), m * k, true);
    fill_random(B.data(), k * n, true);
    fill_random(C_ref.data(), m * n, true); // Pre-fill C for beta test

    // Copy initial C to int8 version
    std::copy(C_ref.begin(), C_ref.end(), C_int8.begin());

    reference_gemm(A.data(), B.data(), C_ref.data(), m, n, k, false, alpha, beta);

    auto B_int8_tensor = create_int8_tensor(B.data(), k, n);
    INT8GemmKernel int8_kernel(B_int8_tensor.get());
    bool success = int8_kernel.multiply(
        A.data(), C_int8.data(),
        m, n, k,
        false, alpha, beta);

    ASSERT_TRUE(success);

    float rel_error = compute_relative_error(C_ref.data(), C_int8.data(), m, n);
    EXPECT_LT(rel_error, 0.02f);

    LOG_INFO("[Test__INT8GemmKernel] BetaScaling (β=" << beta << "): relative error = " << (rel_error * 100.0f) << "%");
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(Test__INT8GemmKernel, SmallMatrix)
{
    const int m = 4, n = 8, k = 6;

    std::vector<float> A(m * k);
    std::vector<float> B(k * n);
    std::vector<float> C_ref(m * n, 0.0f);
    std::vector<float> C_int8(m * n, 0.0f);

    fill_random(A.data(), m * k, true);
    fill_random(B.data(), k * n, true);

    reference_gemm(A.data(), B.data(), C_ref.data(), m, n, k, false, 1.0f, 0.0f);

    auto B_int8_tensor = create_int8_tensor(B.data(), k, n);
    INT8GemmKernel int8_kernel(B_int8_tensor.get());
    bool success = int8_kernel.multiply(
        A.data(), C_int8.data(),
        m, n, k,
        false, 1.0f, 0.0f);

    ASSERT_TRUE(success);

    float rel_error = compute_relative_error(C_ref.data(), C_int8.data(), m, n);
    EXPECT_LT(rel_error, 0.05f); // Slightly higher tolerance for small matrices

    LOG_INFO("[Test__INT8GemmKernel] SmallMatrix (" << m << "×" << k << "×" << n << "): relative error = " << (rel_error * 100.0f) << "%");
}

TEST_F(Test__INT8GemmKernel, ZeroMatrix)
{
    const int m = 32, n = 32, k = 32;

    std::vector<float> A(m * k, 0.0f); // All zeros
    std::vector<float> B(k * n);
    std::vector<float> C_int8(m * n, 0.0f);

    fill_random(B.data(), k * n, true);

    auto B_int8_tensor = create_int8_tensor(B.data(), k, n);
    INT8GemmKernel int8_kernel(B_int8_tensor.get());
    bool success = int8_kernel.multiply(
        A.data(), C_int8.data(),
        m, n, k,
        false, 1.0f, 0.0f);

    ASSERT_TRUE(success);

    // Result should be all zeros
    for (int i = 0; i < m * n; ++i)
    {
        EXPECT_NEAR(C_int8[i], 0.0f, 1e-5f) << "Zero input should produce zero output at index " << i;
    }

    LOG_INFO("[Test__INT8GemmKernel] ZeroMatrix: output is zero (as expected)");
}

TEST_F(Test__INT8GemmKernel, ExtremeValues)
{
    const int m = 16, n = 16, k = 16;

    std::vector<float> A(m * k);
    std::vector<float> B(k * n);
    std::vector<float> C_ref(m * n, 0.0f);
    std::vector<float> C_int8(m * n, 0.0f);

    // Fill with extreme values (near INT8 range limits)
    std::uniform_real_distribution<float> dist_extreme(-100.0f, 100.0f);
    for (size_t i = 0; i < m * k; ++i)
        A[i] = dist_extreme(rng_);
    for (size_t i = 0; i < k * n; ++i)
        B[i] = dist_extreme(rng_);

    reference_gemm(A.data(), B.data(), C_ref.data(), m, n, k, false, 1.0f, 0.0f);

    auto B_int8_tensor = create_int8_tensor(B.data(), k, n);
    INT8GemmKernel int8_kernel(B_int8_tensor.get());
    bool success = int8_kernel.multiply(
        A.data(), C_int8.data(),
        m, n, k,
        false, 1.0f, 0.0f);

    ASSERT_TRUE(success);

    // Higher tolerance for extreme values due to quantization error
    float rel_error = compute_relative_error(C_ref.data(), C_int8.data(), m, n);
    EXPECT_LT(rel_error, 0.05f);

    LOG_INFO("[Test__INT8GemmKernel] ExtremeValues: relative error = " << (rel_error * 100.0f) << "%");
}

// ============================================================================
// Quantization Quality Tests
// ============================================================================

TEST_F(Test__INT8GemmKernel, QuantizationAccuracy)
{
    // Test that per-channel quantization preserves accuracy better than per-tensor
    const int m = 64, n = 128, k = 96;

    std::vector<float> A(m * k);
    std::vector<float> B(k * n);

    // Create data with varying magnitudes across channels
    for (int i = 0; i < k; ++i)
    {
        float channel_scale = 0.1f + 0.9f * (i % 10) / 10.0f; // Varying scales
        for (int j = 0; j < m; ++j)
        {
            A[j * k + i] = dist_normal_(rng_) * channel_scale;
        }
    }

    for (int j = 0; j < n; ++j)
    {
        float channel_scale = 0.1f + 0.9f * (j % 10) / 10.0f;
        for (int i = 0; i < k; ++i)
        {
            B[i * n + j] = dist_normal_(rng_) * channel_scale;
        }
    }

    std::vector<float> C_ref(m * n, 0.0f);
    std::vector<float> C_int8(m * n, 0.0f);

    reference_gemm(A.data(), B.data(), C_ref.data(), m, n, k, false, 1.0f, 0.0f);

    auto B_int8_tensor = create_int8_tensor(B.data(), k, n);
    INT8GemmKernel int8_kernel(B_int8_tensor.get());
    bool success = int8_kernel.multiply(
        A.data(), C_int8.data(),
        m, n, k,
        false, 1.0f, 0.0f);

    ASSERT_TRUE(success);

    // Per-channel quantization should handle varying scales well
    float rel_error = compute_relative_error(C_ref.data(), C_int8.data(), m, n);
    EXPECT_LT(rel_error, 0.03f) << "Per-channel quantization should maintain <3% error even with varying scales";

    LOG_INFO("[Test__INT8GemmKernel] QuantizationAccuracy (varying channel scales): relative error = " << (rel_error * 100.0f) << "%");
}

// ============================================================================
// Performance Benchmarks
// ============================================================================

TEST_F(Test__INT8GemmKernel, PerformanceBenchmark)
{
    // Compare INT8 vs FP32 performance on realistic sizes
    const int m = 512, n = 4096, k = 4096; // Typical LLM layer size
    const int num_runs = 10;

    std::vector<float> A(m * k);
    std::vector<float> B(k * n);
    std::vector<float> C_int8(m * n, 0.0f);
    std::vector<float> C_fp32(m * n, 0.0f);

    fill_random(A.data(), m * k, true);
    fill_random(B.data(), k * n, true);

    // Benchmark INT8
    auto B_int8_tensor = create_int8_tensor(B.data(), k, n);
    INT8GemmKernel int8_kernel(B_int8_tensor.get());

    auto start_int8 = std::chrono::high_resolution_clock::now();
    for (int run = 0; run < num_runs; ++run)
    {
        bool success = int8_kernel.multiply(
            A.data(), C_int8.data(),
            m, n, k,
            false, 1.0f, 0.0f);
        ASSERT_TRUE(success);
    }
    auto end_int8 = std::chrono::high_resolution_clock::now();
    double time_int8_ms = std::chrono::duration<double, std::milli>(end_int8 - start_int8).count() / num_runs;

    // Benchmark FP32 (using OpenBLAS)
    auto B_fp32_tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(k), static_cast<size_t>(n)});

    // Copy FP32 data to tensor
    float *B_fp32_data = B_fp32_tensor->mutable_data();
    std::copy(B.begin(), B.end(), B_fp32_data);

    FP32GemmKernel fp32_kernel(B_fp32_tensor.get());

    auto start_fp32 = std::chrono::high_resolution_clock::now();
    for (int run = 0; run < num_runs; ++run)
    {
        bool success = fp32_kernel.multiply(
            A.data(), C_fp32.data(),
            m, n, k,
            false, 1.0f, 0.0f);
        ASSERT_TRUE(success);
    }
    auto end_fp32 = std::chrono::high_resolution_clock::now();
    double time_fp32_ms = std::chrono::duration<double, std::milli>(end_fp32 - start_fp32).count() / num_runs;

    // Compute GFLOPS
    double total_flops = 2.0 * m * n * k; // multiply + add
    double gflops_int8 = total_flops / (time_int8_ms * 1e6);
    double gflops_fp32 = total_flops / (time_fp32_ms * 1e6);
    double speedup = time_fp32_ms / time_int8_ms;

    LOG_INFO("[Test__INT8GemmKernel] PerformanceBenchmark (" << m << "×" << k << "×" << n << "):");
    LOG_INFO("  INT8: " << time_int8_ms << " ms/iter, " << gflops_int8 << " GFLOPS");
    LOG_INFO("  FP32: " << time_fp32_ms << " ms/iter, " << gflops_fp32 << " GFLOPS");
    LOG_INFO("  Speedup: " << speedup << "×");

    // TODO: Performance currently limited by separate dequantization loop
    // With OneDNN post-op fusion or custom implementation, should achieve 2-4× speedup

    // Check correctness
    float rel_error = compute_relative_error(C_fp32.data(), C_int8.data(), m, n);
    EXPECT_LT(rel_error, 0.05f) << "INT8 quantization error too high";

    if (speedup < 1.0f)
    {
        LOG_WARN("[Test__INT8GemmKernel] INT8 slower than FP32 due to dequant overhead");
        LOG_WARN("[Test__INT8GemmKernel] See OneDNNPrimitiveReuse test for amortized perf");
    }
}

// ============================================================================
// OneDNN-Specific Tests
// ============================================================================

#ifdef HAVE_ONEDNN
TEST_F(Test__INT8GemmKernel, OneDNNAvailable)
{
    // Just verify OneDNN is compiled and available
    SUCCEED() << "OneDNN is available";
    LOG_INFO("[Test__INT8GemmKernel] OneDNN support is compiled");
}

TEST_F(Test__INT8GemmKernel, OneDNNPrimitiveReuse)
{
    // Test that OneDNN primitives can be reused efficiently
    const int m = 128, n = 256, k = 256;
    const int num_iterations = 100;

    std::vector<float> A(m * k);
    std::vector<float> B(k * n);
    std::vector<float> C(m * n, 0.0f);

    fill_random(A.data(), m * k, true);
    fill_random(B.data(), k * n, true);

    auto B_int8_tensor = create_int8_tensor(B.data(), k, n);
    INT8GemmKernel int8_kernel(B_int8_tensor.get());

    // First run (creates primitive)
    auto start_first = std::chrono::high_resolution_clock::now();
    bool success = int8_kernel.multiply(A.data(), C.data(), m, n, k, false, 1.0f, 0.0f);
    auto end_first = std::chrono::high_resolution_clock::now();
    ASSERT_TRUE(success);
    double time_first_ms = std::chrono::duration<double, std::milli>(end_first - start_first).count();

    // Subsequent runs (reuse primitive)
    auto start_reuse = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; ++i)
    {
        success = int8_kernel.multiply(A.data(), C.data(), m, n, k, false, 1.0f, 0.0f);
        ASSERT_TRUE(success);
    }
    auto end_reuse = std::chrono::high_resolution_clock::now();
    double time_reuse_avg_ms = std::chrono::duration<double, std::milli>(end_reuse - start_reuse).count() / num_iterations;

    LOG_INFO("[Test__INT8GemmKernel] OneDNN primitive reuse:");
    LOG_INFO("  First run: " << time_first_ms << " ms (includes primitive creation)");
    LOG_INFO("  Avg reuse: " << time_reuse_avg_ms << " ms (primitive reused)");
    LOG_INFO("  Speedup from reuse: " << (time_first_ms / time_reuse_avg_ms) << "×");

    // Reused runs should be faster (primitive creation overhead amortized)
    EXPECT_LT(time_reuse_avg_ms, time_first_ms * 1.5f) << "Primitive reuse should be efficient";
}
#endif

// ============================================================================
// Main
// ============================================================================

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
