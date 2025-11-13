/**
 * @file Test__ParameterizedInt8Gemm.cpp
 * @brief Tests for template-based parameterized INT8 GEMM microkernel
 *
 * Tests different MR×NR configurations to find optimal size:
 * - 48×8 (OneDNN primary)
 * - 32×8 (large, fits registers)
 * - 16×8 (medium)
 * - 16×16 (original size, rounded up from 6×16)
 *
 * @author David Sanftenberg
 * @date November 12, 2025
 */

#include <gtest/gtest.h>
#include "v2/kernels/cpu/gemm_v2/ParameterizedInt8Gemm.h"
#include <random>
#include <chrono>
#include <cmath>

using namespace llaminar::v2::kernels::cpu;

class Test__ParameterizedInt8Gemm : public ::testing::Test
{
protected:
    void SetUp() override
    {
        rng_.seed(42);
    }

    std::mt19937 rng_;

    /**
     * @brief Generate random INT8 matrix
     */
    void fillRandomInt8(int8_t *data, size_t count)
    {
        std::uniform_int_distribution<int> dist(-127, 127);
        for (size_t i = 0; i < count; ++i)
        {
            data[i] = static_cast<int8_t>(dist(rng_));
        }
    }

    /**
     * @brief Reference INT8 GEMM (scalar)
     */
    void referenceGemm(int M, int N, int K,
                       const int8_t *A, int lda,
                       const int8_t *B, int ldb,
                       int32_t *C, int ldc)
    {
        for (int i = 0; i < M; ++i)
        {
            for (int j = 0; j < N; ++j)
            {
                int32_t sum = 0;
                for (int k = 0; k < K; ++k)
                {
                    sum += static_cast<int32_t>(A[i * lda + k]) *
                           static_cast<int32_t>(B[j * ldb + k]);
                }
                C[i * ldc + j] = sum;
            }
        }
    }

    /**
     * @brief Verify correctness against reference
     */
    void verifyCorrectness(const int32_t *C_actual, const int32_t *C_expected,
                           int M, int N, int ldc, const char *test_name)
    {
        int max_error = 0;
        int error_count = 0;

        for (int i = 0; i < M; ++i)
        {
            for (int j = 0; j < N; ++j)
            {
                int32_t actual = C_actual[i * ldc + j];
                int32_t expected = C_expected[i * ldc + j];
                int error = std::abs(actual - expected);

                if (error > max_error)
                {
                    max_error = error;
                }
                if (error > 0)
                {
                    error_count++;
                    if (error_count <= 5)
                    { // Print first few errors
                        std::cout << test_name << " Error at [" << i << "," << j << "]: "
                                  << "actual=" << actual << ", expected=" << expected
                                  << ", diff=" << error << std::endl;
                    }
                }
            }
        }

        EXPECT_EQ(error_count, 0) << test_name << " had " << error_count
                                  << " errors, max_error=" << max_error;
    }

    /**
     * @brief Benchmark helper
     */
    template <typename GemmKernel>
    double benchmarkGemm(int M, int N, int K,
                         const int8_t *A, int lda,
                         const int8_t *B, int ldb,
                         int32_t *C, int ldc,
                         int iterations = 100)
    {
        auto start = std::chrono::high_resolution_clock::now();

        for (int iter = 0; iter < iterations; ++iter)
        {
            GemmKernel::gemm(M, N, K, A, lda, B, ldb, C, ldc);
        }

        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        return ms / iterations;
    }
};

// ============================================================================
// Correctness Tests
// ============================================================================

TEST_F(Test__ParameterizedInt8Gemm, SmallMatrix_48x8)
{
    const int M = 48, N = 8, K = 256;
    const int lda = K, ldb = K, ldc = N;

    std::vector<int8_t> A(M * K);
    std::vector<int8_t> B(N * K);
    std::vector<int32_t> C_actual(M * N, 0);
    std::vector<int32_t> C_expected(M * N, 0);

    fillRandomInt8(A.data(), A.size());
    fillRandomInt8(B.data(), B.size());

    Int8Gemm_48x8::gemm(M, N, K, A.data(), lda, B.data(), ldb, C_actual.data(), ldc);
    referenceGemm(M, N, K, A.data(), lda, B.data(), ldb, C_expected.data(), ldc);

    verifyCorrectness(C_actual.data(), C_expected.data(), M, N, ldc, "48x8");
}

TEST_F(Test__ParameterizedInt8Gemm, SmallMatrix_32x8)
{
    const int M = 32, N = 8, K = 256;
    const int lda = K, ldb = K, ldc = N;

    std::vector<int8_t> A(M * K);
    std::vector<int8_t> B(N * K);
    std::vector<int32_t> C_actual(M * N, 0);
    std::vector<int32_t> C_expected(M * N, 0);

    fillRandomInt8(A.data(), A.size());
    fillRandomInt8(B.data(), B.size());

    Int8Gemm_32x8::gemm(M, N, K, A.data(), lda, B.data(), ldb, C_actual.data(), ldc);
    referenceGemm(M, N, K, A.data(), lda, B.data(), ldb, C_expected.data(), ldc);

    verifyCorrectness(C_actual.data(), C_expected.data(), M, N, ldc, "32x8");
}

TEST_F(Test__ParameterizedInt8Gemm, SmallMatrix_16x8)
{
    const int M = 16, N = 8, K = 256;
    const int lda = K, ldb = K, ldc = N;

    std::vector<int8_t> A(M * K);
    std::vector<int8_t> B(N * K);
    std::vector<int32_t> C_actual(M * N, 0);
    std::vector<int32_t> C_expected(M * N, 0);

    fillRandomInt8(A.data(), A.size());
    fillRandomInt8(B.data(), B.size());

    Int8Gemm_16x8::gemm(M, N, K, A.data(), lda, B.data(), ldb, C_actual.data(), ldc);
    referenceGemm(M, N, K, A.data(), lda, B.data(), ldb, C_expected.data(), ldc);

    verifyCorrectness(C_actual.data(), C_expected.data(), M, N, ldc, "16x8");
}

TEST_F(Test__ParameterizedInt8Gemm, SmallMatrix_16x16)
{
    const int M = 16, N = 16, K = 256;
    const int lda = K, ldb = K, ldc = N;

    std::vector<int8_t> A(M * K);
    std::vector<int8_t> B(N * K);
    std::vector<int32_t> C_actual(M * N, 0);
    std::vector<int32_t> C_expected(M * N, 0);

    fillRandomInt8(A.data(), A.size());
    fillRandomInt8(B.data(), B.size());

    Int8Gemm_6x16::gemm(M, N, K, A.data(), lda, B.data(), ldb, C_actual.data(), ldc);
    referenceGemm(M, N, K, A.data(), lda, B.data(), ldb, C_expected.data(), ldc);

    verifyCorrectness(C_actual.data(), C_expected.data(), M, N, ldc, "16x16");
}

// ============================================================================
// Non-aligned dimension tests (remainder handling)
// ============================================================================

TEST_F(Test__ParameterizedInt8Gemm, NonAligned_47x7_48x8)
{
    const int M = 47, N = 7, K = 253; // Odd sizes to test edge cases
    const int lda = K, ldb = K, ldc = N;

    std::vector<int8_t> A(M * K);
    std::vector<int8_t> B(N * K);
    std::vector<int32_t> C_actual(M * N, 0);
    std::vector<int32_t> C_expected(M * N, 0);

    fillRandomInt8(A.data(), A.size());
    fillRandomInt8(B.data(), B.size());

    Int8Gemm_48x8::gemm(M, N, K, A.data(), lda, B.data(), ldb, C_actual.data(), ldc);
    referenceGemm(M, N, K, A.data(), lda, B.data(), ldb, C_expected.data(), ldc);

    verifyCorrectness(C_actual.data(), C_expected.data(), M, N, ldc, "47x7 (48x8 kernel)");
}

// ============================================================================
// Performance Benchmarks
// ============================================================================

/**
 * @brief FFN Down workload: [1, 896] × [896, 3584]^T (after transpose: [1, 896] × [3584, 896])
 *
 * This is the critical decode path - single token inference.
 */
TEST_F(Test__ParameterizedInt8Gemm, DISABLED_Benchmark_FFNDown_SingleToken)
{
    const int M = 1, N = 3584, K = 896;
    const int lda = K, ldb = K, ldc = N;

    std::vector<int8_t> A(M * K);
    std::vector<int8_t> B(N * K);
    std::vector<int32_t> C(M * N, 0);

    fillRandomInt8(A.data(), A.size());
    fillRandomInt8(B.data(), B.size());

    std::cout << "\n=== FFN Down Single Token Benchmark ===\n";
    std::cout << "Problem: M=" << M << ", N=" << N << ", K=" << K << "\n";
    std::cout << "FLOPs: " << (2.0 * M * N * K / 1e9) << " GFLOPS\n\n";

    double time_48x8 = benchmarkGemm<Int8Gemm_48x8>(M, N, K, A.data(), lda, B.data(), ldb, C.data(), ldc, 1000);
    double gops_48x8 = (2.0 * M * N * K) / (time_48x8 * 1e6);
    std::cout << "48×8:  " << time_48x8 << " ms → " << gops_48x8 << " GOPS\n";

    double time_32x8 = benchmarkGemm<Int8Gemm_32x8>(M, N, K, A.data(), lda, B.data(), ldb, C.data(), ldc, 1000);
    double gops_32x8 = (2.0 * M * N * K) / (time_32x8 * 1e6);
    std::cout << "32×8:  " << time_32x8 << " ms → " << gops_32x8 << " GOPS\n";

    double time_16x8 = benchmarkGemm<Int8Gemm_16x8>(M, N, K, A.data(), lda, B.data(), ldb, C.data(), ldc, 1000);
    double gops_16x8 = (2.0 * M * N * K) / (time_16x8 * 1e6);
    std::cout << "16×8:  " << time_16x8 << " ms → " << gops_16x8 << " GOPS\n";

    double time_16x16 = benchmarkGemm<Int8Gemm_6x16>(M, N, K, A.data(), lda, B.data(), ldb, C.data(), ldc, 1000);
    double gops_16x16 = (2.0 * M * N * K) / (time_16x16 * 1e6);
    std::cout << "16×16: " << time_16x16 << " ms → " << gops_16x16 << " GOPS\n";

    std::cout << "\nSpeedup over 16×16 (baseline):\n";
    std::cout << "48×8:  " << (time_16x16 / time_48x8) << "×\n";
    std::cout << "32×8:  " << (time_16x16 / time_32x8) << "×\n";
    std::cout << "16×8:  " << (time_16x16 / time_16x8) << "×\n";
}

/**
 * @brief Large matrix benchmark: [128, 3584] × [3584, 896]^T
 *
 * This simulates larger batch sizes or prefill.
 */
TEST_F(Test__ParameterizedInt8Gemm, DISABLED_Benchmark_LargeMatrix)
{
    const int M = 128, N = 3584, K = 896;
    const int lda = K, ldb = K, ldc = N;

    std::vector<int8_t> A(M * K);
    std::vector<int8_t> B(N * K);
    std::vector<int32_t> C(M * N, 0);

    fillRandomInt8(A.data(), A.size());
    fillRandomInt8(B.data(), B.size());

    std::cout << "\n=== Large Matrix Benchmark ===\n";
    std::cout << "Problem: M=" << M << ", N=" << N << ", K=" << K << "\n";
    std::cout << "FLOPs: " << (2.0 * M * N * K / 1e9) << " GFLOPS\n\n";

    double time_48x8 = benchmarkGemm<Int8Gemm_48x8>(M, N, K, A.data(), lda, B.data(), ldb, C.data(), ldc, 100);
    double gops_48x8 = (2.0 * M * N * K) / (time_48x8 * 1e6);
    std::cout << "48×8:  " << time_48x8 << " ms → " << gops_48x8 << " GOPS\n";

    double time_32x8 = benchmarkGemm<Int8Gemm_32x8>(M, N, K, A.data(), lda, B.data(), ldb, C.data(), ldc, 100);
    double gops_32x8 = (2.0 * M * N * K) / (time_32x8 * 1e6);
    std::cout << "32×8:  " << time_32x8 << " ms → " << gops_32x8 << " GOPS\n";

    double time_16x8 = benchmarkGemm<Int8Gemm_16x8>(M, N, K, A.data(), lda, B.data(), ldb, C.data(), ldc, 100);
    double gops_16x8 = (2.0 * M * N * K) / (time_16x8 * 1e6);
    std::cout << "16×8:  " << time_16x8 << " ms → " << gops_16x8 << " GOPS\n";

    double time_16x16 = benchmarkGemm<Int8Gemm_6x16>(M, N, K, A.data(), lda, B.data(), ldb, C.data(), ldc, 100);
    double gops_16x16 = (2.0 * M * N * K) / (time_16x16 * 1e6);
    std::cout << "16×16: " << time_16x16 << " ms → " << gops_16x16 << " GOPS\n";

    std::cout << "\nSpeedup over 16×16 (baseline):\n";
    std::cout << "48×8:  " << (time_16x16 / time_48x8) << "×\n";
    std::cout << "32×8:  " << (time_16x16 / time_32x8) << "×\n";
    std::cout << "16×8:  " << (time_16x16 / time_16x8) << "×\n";
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
