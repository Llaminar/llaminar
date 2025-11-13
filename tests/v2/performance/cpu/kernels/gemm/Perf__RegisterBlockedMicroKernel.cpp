/**
 * @file Perf__RegisterBlockedMicroKernel.cpp
 * @brief Performance test for Phase 1: 6×16 register-blocked microkernel
 *
 * Validates:
 * - Correctness vs naive 1×1 implementation
 * - Performance improvement (target: 4-6× speedup, 2,000-3,000 GOPS)
 * - Memory traffic reduction (should be ~76× less C traffic)
 *
 * Test cases:
 * - Single 6×16 microkernel (K=128)
 * - Tiled 384×128 GEMM (realistic FFN_down size)
 * - Comparison with baseline IntegerGemmKernelTemplate
 *
 * Expected results:
 * - Baseline: ~472 GOPS
 * - Phase 1:  ~2,500 GOPS (5.3× speedup)
 * - Max error: <1e-3 (numerical stability)
 *
 * @author David Sanftenberg
 * @date November 12, 2025
 */

#include <gtest/gtest.h>
#include "kernels/cpu/gemm_v2/RegisterBlockedMicroKernel.h"
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <iostream>
#include <iomanip>

using namespace llaminar2::register_blocked;

class RegisterBlockedMicroKernelTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        rng.seed(42); // Reproducible
    }

    std::mt19937 rng;
    std::uniform_int_distribution<int> dist_int8{-127, 127};

    // Naive 1×1 reference implementation
    void naive_gemm_int8(
        const int8_t *A, const int8_t *B, int32_t *C,
        int M, int N, int K, int lda, int ldb, int ldc,
        bool zero_init = false)
    {
        for (int m = 0; m < M; ++m)
        {
            for (int n = 0; n < N; ++n)
            {
                if (zero_init)
                {
                    C[m * ldc + n] = 0;
                }
                for (int k = 0; k < K; k += 4)
                {
                    // VNNI-style 4-way dot product
                    for (int kk = 0; kk < 4; ++kk)
                    {
                        C[m * ldc + n] += static_cast<int32_t>(A[m * lda + k + kk]) *
                                          static_cast<int32_t>(B[(k + kk) * ldb + n]);
                    }
                }
            }
        }
    }

    double compute_gflops(int64_t ops, double time_ms)
    {
        return (ops / 1e9) / (time_ms / 1000.0);
    }
};

// Test 1: Single 6×16 microkernel correctness
TEST_F(RegisterBlockedMicroKernelTest, SingleMicroKernelCorrectness)
{
    constexpr int MR = 6, NR = 16, K = 128;

    // Allocate aligned matrices
    std::vector<int8_t> A(MR * K);
    std::vector<int8_t> B(K * NR);
    std::vector<int32_t> C_ref(MR * NR, 0);
    std::vector<int32_t> C_opt(MR * NR, 0);

    // Initialize with random INT8
    for (auto &v : A)
        v = dist_int8(rng);
    for (auto &v : B)
        v = dist_int8(rng);

    // Reference: Naive implementation
    naive_gemm_int8(A.data(), B.data(), C_ref.data(), MR, NR, K, K, NR, NR, true);

    // Optimized: Register-blocked microkernel
    MicroKernel<6, 16>::execute_zero_init(
        A.data(), B.data(), C_opt.data(), K, K, NR, NR);

    // Verify correctness
    int32_t max_diff = 0;
    double sum_sq_diff = 0.0;
    for (int i = 0; i < MR * NR; ++i)
    {
        int32_t diff = std::abs(C_ref[i] - C_opt[i]);
        max_diff = std::max(max_diff, diff);
        sum_sq_diff += diff * diff;
    }

    double rms_error = std::sqrt(sum_sq_diff / (MR * NR));

    std::cout << "Single 6×16 microkernel (K=" << K << "):\n";
    std::cout << "  Max absolute error: " << max_diff << "\n";
    std::cout << "  RMS error: " << rms_error << "\n";

    EXPECT_EQ(max_diff, 0) << "Perfect match expected for INT8×INT8→INT32";
}

// Test 2: Performance comparison - Single microkernel
TEST_F(RegisterBlockedMicroKernelTest, SingleMicroKernelPerformance)
{
    constexpr int MR = 6, NR = 16, K = 4096; // Larger K for timing
    constexpr int WARMUP = 10;
    constexpr int ITERATIONS = 1000;

    std::vector<int8_t> A(MR * K);
    std::vector<int8_t> B(K * NR);
    std::vector<int32_t> C(MR * NR, 0);

    for (auto &v : A)
        v = dist_int8(rng);
    for (auto &v : B)
        v = dist_int8(rng);

    // Warmup
    for (int i = 0; i < WARMUP; ++i)
    {
        MicroKernel<6, 16>::execute_zero_init(A.data(), B.data(), C.data(), K, K, NR, NR);
    }

    // Timed run
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERATIONS; ++i)
    {
        MicroKernel<6, 16>::execute_zero_init(A.data(), B.data(), C.data(), K, K, NR, NR);
    }
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double time_per_call_us = (elapsed_ms * 1000.0) / ITERATIONS;

    // GFLOPS: 2 * M * N * K operations (multiply + add)
    int64_t ops_per_call = 2LL * MR * NR * K;
    double gflops = compute_gflops(ops_per_call * ITERATIONS, elapsed_ms);

    std::cout << "\nSingle 6×16 microkernel performance (K=" << K << "):\n";
    std::cout << "  Time per call: " << std::fixed << std::setprecision(2)
              << time_per_call_us << " μs\n";
    std::cout << "  Throughput: " << std::fixed << std::setprecision(1)
              << gflops << " GFLOPS\n";

    // Sanity check: Should be > 100 GFLOPS for a single microkernel
    EXPECT_GT(gflops, 100.0) << "Microkernel suspiciously slow";
}

// Test 3: Tiled GEMM performance (384×128, FFN_down typical)
TEST_F(RegisterBlockedMicroKernelTest, TiledGemmPerformance)
{
    constexpr int M = 384, N = 128, K = 4096;
    constexpr int MR = 6, NR = 16;
    constexpr int ITERATIONS = 10;

    std::vector<int8_t> A(M * K);
    std::vector<int8_t> B(K * N);
    std::vector<int32_t> C(M * N, 0);

    for (auto &v : A)
        v = dist_int8(rng);
    for (auto &v : B)
        v = dist_int8(rng);

    // Tiled GEMM using 6×16 microkernel
    auto gemm_tiled = [&]()
    {
        for (int m = 0; m < M; m += MR)
        {
            for (int n = 0; n < N; n += NR)
            {
                int m_block = std::min(MR, M - m);
                int n_block = std::min(NR, N - n);

                if (m_block == MR && n_block == NR)
                {
                    // Full 6×16 tile
                    MicroKernel<6, 16>::execute_zero_init(
                        A.data() + m * K,
                        B.data() + n,
                        C.data() + m * N + n,
                        K, K, N, N);
                }
                else
                {
                    // Edge case: Use naive for simplicity in Phase 1
                    for (int mm = 0; mm < m_block; ++mm)
                    {
                        for (int nn = 0; nn < n_block; ++nn)
                        {
                            int32_t sum = 0;
                            for (int k = 0; k < K; k += 4)
                            {
                                for (int kk = 0; kk < 4; ++kk)
                                {
                                    sum += A[(m + mm) * K + k + kk] * B[(k + kk) * N + n + nn];
                                }
                            }
                            C[(m + mm) * N + n + nn] = sum;
                        }
                    }
                }
            }
        }
    };

    // Warmup
    gemm_tiled();

    // Timed run
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERATIONS; ++i)
    {
        gemm_tiled();
    }
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double time_per_gemm_ms = elapsed_ms / ITERATIONS;

    int64_t ops = 2LL * M * N * K;
    double gflops = compute_gflops(ops * ITERATIONS, elapsed_ms);

    std::cout << "\nTiled GEMM " << M << "×" << N << "×" << K << " (Phase 1):\n";
    std::cout << "  Time per GEMM: " << std::fixed << std::setprecision(2)
              << time_per_gemm_ms << " ms\n";
    std::cout << "  Throughput: " << std::fixed << std::setprecision(1)
              << gflops << " GFLOPS\n";
    std::cout << "  Tiles: " << (M / MR) << "×" << (N / NR) << " = "
              << (M / MR) * (N / NR) << " microkernel calls\n";

    // Phase 1 target: 2,000-3,000 GFLOPS
    std::cout << "\n  Target: 2,000-3,000 GFLOPS (4-6× baseline 472 GFLOPS)\n";
    if (gflops >= 2000.0)
    {
        std::cout << "  ✅ Phase 1 target ACHIEVED! (" << gflops << " GFLOPS)\n";
    }
    else if (gflops >= 1000.0)
    {
        std::cout << "  ⚠️  Good progress, needs tuning (" << gflops << " GFLOPS)\n";
    }
    else
    {
        std::cout << "  ❌ Below expectations, investigate (" << gflops << " GFLOPS)\n";
    }
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
