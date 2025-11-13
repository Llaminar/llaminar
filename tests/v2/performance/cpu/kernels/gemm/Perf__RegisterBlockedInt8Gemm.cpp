/**
 * @file Perf__RegisterBlockedInt8Gemm.cpp
 * @brief Performance test for register-blocked INT8 GEMM (Phase 1)
 *
 * Target: 2,000-3,000 GOPS (4-6× speedup over baseline 472 GOPS)
 *
 * @author David Sanftenberg
 * @date November 12, 2025
 */

#include <gtest/gtest.h>
#include "kernels/cpu/gemm_v2/RegisterBlockedInt8Gemm.h"
#include <vector>
#include <random>
#include <chrono>
#include <iostream>
#include <iomanip>

using namespace llaminar::v2::kernels::cpu;

class RegisterBlockedInt8GemmTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        std::random_device rd;
        gen_ = std::mt19937(rd());
        dist_int8_ = std::uniform_int_distribution<int>(-127, 127);
    }

    std::mt19937 gen_;
    std::uniform_int_distribution<int> dist_int8_;

    // Helper: Fill matrix with random INT8 values
    void fill_random_int8(std::vector<int8_t> &mat)
    {
        for (auto &val : mat)
        {
            val = static_cast<int8_t>(dist_int8_(gen_));
        }
    }

    // Helper: Naive GEMM for correctness verification
    void naive_gemm(
        int M, int N, int K,
        const int8_t *A, int lda,
        const int8_t *B, int ldb,
        int32_t *C, int ldc)
    {
        for (int i = 0; i < M; ++i)
        {
            for (int j = 0; j < N; ++j)
            {
                int32_t sum = C[i * ldc + j];
                for (int k = 0; k < K; ++k)
                {
                    sum += static_cast<int32_t>(A[i * lda + k]) *
                           static_cast<int32_t>(B[j * ldb + k]);
                }
                C[i * ldc + j] = sum;
            }
        }
    }

    // Helper: Verify results match within tolerance
    bool verify_results(
        const std::vector<int32_t> &result,
        const std::vector<int32_t> &reference,
        int M, int N,
        double rel_tol = 1e-5)
    {
        int mismatches = 0;
        int32_t max_diff = 0;

        for (int i = 0; i < M; ++i)
        {
            for (int j = 0; j < N; ++j)
            {
                int idx = i * N + j;
                int32_t diff = std::abs(result[idx] - reference[idx]);
                if (diff > max_diff)
                    max_diff = diff;

                if (diff > 0)
                {
                    double rel_error = static_cast<double>(diff) /
                                       std::max(std::abs(reference[idx]), 1);
                    if (rel_error > rel_tol)
                    {
                        if (mismatches < 10)
                        {
                            std::cout << "Mismatch at [" << i << "," << j << "]: "
                                      << "got " << result[idx]
                                      << ", expected " << reference[idx]
                                      << ", diff=" << diff << std::endl;
                        }
                        ++mismatches;
                    }
                }
            }
        }

        std::cout << "Max absolute difference: " << max_diff << std::endl;
        if (mismatches > 0)
        {
            std::cout << "Total mismatches: " << mismatches << " / " << (M * N) << std::endl;
        }

        return mismatches == 0;
    }
};

// Test 1: Correctness on small 6×16 tile (exact microkernel size)
TEST_F(RegisterBlockedInt8GemmTest, Correctness_SingleTile_6x16)
{
    const int M = 6, N = 16, K = 128;

    std::vector<int8_t> A(M * K);
    std::vector<int8_t> B(N * K); // Column-major: N columns × K rows
    std::vector<int32_t> C_result(M * N, 0);
    std::vector<int32_t> C_reference(M * N, 0);

    fill_random_int8(A);
    fill_random_int8(B);

    // Optimized kernel
    RegisterBlockedInt8Gemm::gemm(M, N, K, A.data(), K, B.data(), K, C_result.data(), N);

    // Naive reference
    naive_gemm(M, N, K, A.data(), K, B.data(), K, C_reference.data(), N);

    EXPECT_TRUE(verify_results(C_result, C_reference, M, N));
}

// Test 2: Correctness on multiple tiles
TEST_F(RegisterBlockedInt8GemmTest, Correctness_MultipleTiles_18x48)
{
    const int M = 18, N = 48, K = 256; // 3×3 tiles of 6×16

    std::vector<int8_t> A(M * K);
    std::vector<int8_t> B(N * K);
    std::vector<int32_t> C_result(M * N, 0);
    std::vector<int32_t> C_reference(M * N, 0);

    fill_random_int8(A);
    fill_random_int8(B);

    RegisterBlockedInt8Gemm::gemm(M, N, K, A.data(), K, B.data(), K, C_result.data(), N);
    naive_gemm(M, N, K, A.data(), K, B.data(), K, C_reference.data(), N);

    EXPECT_TRUE(verify_results(C_result, C_reference, M, N));
}

// Test 3: Edge case - non-aligned dimensions
TEST_F(RegisterBlockedInt8GemmTest, Correctness_EdgeCase_7x17)
{
    const int M = 7, N = 17, K = 128;

    std::vector<int8_t> A(M * K);
    std::vector<int8_t> B(N * K);
    std::vector<int32_t> C_result(M * N, 0);
    std::vector<int32_t> C_reference(M * N, 0);

    fill_random_int8(A);
    fill_random_int8(B);

    RegisterBlockedInt8Gemm::gemm(M, N, K, A.data(), K, B.data(), K, C_result.data(), N);
    naive_gemm(M, N, K, A.data(), K, B.data(), K, C_reference.data(), N);

    EXPECT_TRUE(verify_results(C_result, C_reference, M, N));
}

// Test 4: Performance - Single token (small case)
TEST_F(RegisterBlockedInt8GemmTest, Performance_SingleToken_1x896x4096)
{
    const int M = 1, N = 896, K = 4096;

    std::vector<int8_t> A(M * K);
    std::vector<int8_t> B(N * K);
    std::vector<int32_t> C(M * N, 0);

    fill_random_int8(A);
    fill_random_int8(B);

    // Warmup
    for (int i = 0; i < 3; ++i)
    {
        RegisterBlockedInt8Gemm::gemm(M, N, K, A.data(), K, B.data(), K, C.data(), N);
    }

    // Timed run
    auto start = std::chrono::high_resolution_clock::now();
    const int iterations = 100;
    for (int i = 0; i < iterations; ++i)
    {
        RegisterBlockedInt8Gemm::gemm(M, N, K, A.data(), K, B.data(), K, C.data(), N);
    }
    auto end = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(end - start).count() / iterations;
    double ops = 2.0 * M * N * K; // Multiply-accumulate = 2 ops
    double gops = ops / (ms * 1e6);

    std::cout << "\n[Single Token 1×896×4096]" << std::endl;
    std::cout << "  Time: " << std::fixed << std::setprecision(3) << ms << " ms" << std::endl;
    std::cout << "  Performance: " << std::setprecision(1) << gops << " GOPS" << std::endl;
    std::cout << "  Target: 2000-3000 GOPS (4-6× speedup)" << std::endl;
}

// Test 5: Performance - Medium batch
TEST_F(RegisterBlockedInt8GemmTest, Performance_MediumBatch_128x896x4096)
{
    const int M = 128, N = 896, K = 4096;

    std::vector<int8_t> A(M * K);
    std::vector<int8_t> B(N * K);
    std::vector<int32_t> C(M * N, 0);

    fill_random_int8(A);
    fill_random_int8(B);

    // Warmup
    for (int i = 0; i < 3; ++i)
    {
        RegisterBlockedInt8Gemm::gemm(M, N, K, A.data(), K, B.data(), K, C.data(), N);
    }

    // Timed run
    auto start = std::chrono::high_resolution_clock::now();
    const int iterations = 10;
    for (int i = 0; i < iterations; ++i)
    {
        RegisterBlockedInt8Gemm::gemm(M, N, K, A.data(), K, B.data(), K, C.data(), N);
    }
    auto end = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(end - start).count() / iterations;
    double ops = 2.0 * M * N * K;
    double gops = ops / (ms * 1e6);

    std::cout << "\n[Medium Batch 128×896×4096]" << std::endl;
    std::cout << "  Time: " << std::fixed << std::setprecision(3) << ms << " ms" << std::endl;
    std::cout << "  Performance: " << std::setprecision(1) << gops << " GOPS" << std::endl;
    std::cout << "  Target: 2000-3000 GOPS" << std::endl;
}

// Test 6: Performance - FFN down projection (Qwen profile)
TEST_F(RegisterBlockedInt8GemmTest, Performance_FFN_Down_4096x896x4864)
{
    const int M = 4096, N = 896, K = 4864;

    std::vector<int8_t> A(M * K);
    std::vector<int8_t> B(N * K);
    std::vector<int32_t> C(M * N, 0);

    fill_random_int8(A);
    fill_random_int8(B);

    // Warmup
    for (int i = 0; i < 3; ++i)
    {
        RegisterBlockedInt8Gemm::gemm(M, N, K, A.data(), K, B.data(), K, C.data(), N);
    }

    // Timed run
    auto start = std::chrono::high_resolution_clock::now();
    const int iterations = 5;
    for (int i = 0; i < iterations; ++i)
    {
        RegisterBlockedInt8Gemm::gemm(M, N, K, A.data(), K, B.data(), K, C.data(), N);
    }
    auto end = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(end - start).count() / iterations;
    double ops = 2.0 * M * N * K;
    double gops = ops / (ms * 1e6);

    std::cout << "\n[FFN Down 4096×896×4864 - Primary Bottleneck]" << std::endl;
    std::cout << "  Time: " << std::fixed << std::setprecision(3) << ms << " ms" << std::endl;
    std::cout << "  Performance: " << std::setprecision(1) << gops << " GOPS" << std::endl;
    std::cout << "  Baseline: 472 GOPS" << std::endl;
    std::cout << "  Target: 2000-3000 GOPS (4-6× speedup)" << std::endl;
    std::cout << "  OneDNN: 6610 GOPS (final goal)" << std::endl;

    // Success criteria
    EXPECT_GT(gops, 1000.0) << "Phase 1 should achieve at least 1000 GOPS (2× speedup)";
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    std::cout << "\n========================================" << std::endl;
    std::cout << "Register-Blocked INT8 GEMM - Phase 1" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Architecture: 6×16 microkernel" << std::endl;
    std::cout << "Target: 2,000-3,000 GOPS (4-6× speedup)" << std::endl;
    std::cout << "Baseline: 472 GOPS" << std::endl;
    std::cout << "OneDNN Goal: 6,610 GOPS" << std::endl;
    std::cout << "========================================\n"
              << std::endl;

    return RUN_ALL_TESTS();
}
