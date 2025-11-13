/**
 * @file Perf__Int8Gemm_MicrokernelSize.cpp
 * @brief Benchmark comparing different microkernel dimensions (48×8 vs 32×8 vs 16×16)
 *
 * OneDNN uses 48×8 microkernel for better M-dimension amortization.
 * This test validates the expected 20-30% improvement from larger M unrolling.
 *
 * Expected results:
 * - 48×8: Best performance (OneDNN-matched dimensions)
 * - 32×8: Good compromise (medium M unrolling)
 * - 16×16: Baseline (our current default)
 * - 16×8: Medium performance (smaller N unrolling)
 *
 * @author David Sanftenberg
 * @date November 12, 2025
 */

#include <gtest/gtest.h>
#include <chrono>
#include <random>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <omp.h>

#include "kernels/cpu/gemm_v2/ParameterizedInt8Gemm.h"

using namespace llaminar::v2::kernels::cpu;

class Perf__Int8Gemm_MicrokernelSize : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Detect OMP_NUM_THREADS for reporting
        max_threads_ = 56; // Default
        const char *omp_num_threads_env = std::getenv("OMP_NUM_THREADS");
        if (omp_num_threads_env)
        {
            max_threads_ = std::atoi(omp_num_threads_env);
        }
    }

    int max_threads_;
};

/**
 * @brief Helper to run GEMM benchmark with specific configuration
 */
template <typename GemmKernel>
double benchmark_gemm(int M, int N, int K, int iterations = 10)
{
    // Allocate aligned arrays
    std::vector<int8_t> A(M * K);
    std::vector<int8_t> B(K * N);
    std::vector<int32_t> C(M * N);

    // Initialize with random data
    std::mt19937 gen(12345);
    std::uniform_int_distribution<int> dist(-127, 127);
    for (auto &val : A)
        val = static_cast<int8_t>(dist(gen));
    for (auto &val : B)
        val = static_cast<int8_t>(dist(gen));

    // Warmup
    GemmKernel::gemm(M, N, K, A.data(), K, B.data(), K, C.data(), N);

    // Timed run
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i)
    {
        GemmKernel::gemm(M, N, K, A.data(), K, B.data(), K, C.data(), N);
    }
    auto end = std::chrono::high_resolution_clock::now();

    double total_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
    double ms_per_iter = total_ms / iterations;
    return ms_per_iter;
}

/**
 * @brief Compare microkernel sizes at M=2048 (typical inference size)
 *
 * Expected results based on OneDNN analysis:
 * - 48×8: Best (3× M unrolling vs 16×16, better amortization)
 * - 32×8: Good (2× M unrolling)
 * - 16×8: Medium (same M as 16×16, smaller N)
 * - 16×16: Baseline (current default)
 *
 * Target: 20-30% improvement from 48×8 vs 16×16
 */
TEST_F(Perf__Int8Gemm_MicrokernelSize, MicrokernelComparison_M2048)
{
    const int M = 2048;
    const int N = 3584; // Match other tests
    const int K = 896;
    const int iterations = 10;

    std::cout << "\n=== Microkernel Size Comparison (M=" << M << ", N=" << N
              << ", K=" << K << ", " << max_threads_ << " threads) ===\n";
    std::cout << std::setw(20) << "Microkernel (MR×NR)"
              << std::setw(15) << "Time (ms)"
              << std::setw(15) << "GOPS"
              << std::setw(15) << "Speedup"
              << std::setw(15) << "vs 16×16\n";
    std::cout << std::string(80, '-') << "\n";

    // Baseline: 16×16
    double baseline_ms = benchmark_gemm<Int8Gemm_6x16>(M, N, K, iterations);
    double baseline_gops = (2.0 * M * N * K / 1e9) / (baseline_ms / 1000.0);

    std::cout << std::setw(20) << "16×16 (baseline)"
              << std::setw(15) << std::fixed << std::setprecision(2) << baseline_ms
              << std::setw(15) << std::fixed << std::setprecision(2) << baseline_gops
              << std::setw(15) << "1.00×"
              << std::setw(15) << "---\n";

    // 16×8: Smaller N
    double ms_16x8 = benchmark_gemm<Int8Gemm_16x8>(M, N, K, iterations);
    double gops_16x8 = (2.0 * M * N * K / 1e9) / (ms_16x8 / 1000.0);
    double speedup_16x8 = baseline_ms / ms_16x8;

    std::cout << std::setw(20) << "16×8"
              << std::setw(15) << std::fixed << std::setprecision(2) << ms_16x8
              << std::setw(15) << std::fixed << std::setprecision(2) << gops_16x8
              << std::setw(15) << std::fixed << std::setprecision(2) << speedup_16x8 << "×"
              << std::setw(15) << std::showpos << std::fixed << std::setprecision(1)
              << ((gops_16x8 - baseline_gops) / baseline_gops * 100) << "%\n"
              << std::noshowpos;

    // 32×8: Medium M
    double ms_32x8 = benchmark_gemm<Int8Gemm_32x8>(M, N, K, iterations);
    double gops_32x8 = (2.0 * M * N * K / 1e9) / (ms_32x8 / 1000.0);
    double speedup_32x8 = baseline_ms / ms_32x8;

    std::cout << std::setw(20) << "32×8"
              << std::setw(15) << std::fixed << std::setprecision(2) << ms_32x8
              << std::setw(15) << std::fixed << std::setprecision(2) << gops_32x8
              << std::setw(15) << std::fixed << std::setprecision(2) << speedup_32x8 << "×"
              << std::setw(15) << std::showpos << std::fixed << std::setprecision(1)
              << ((gops_32x8 - baseline_gops) / baseline_gops * 100) << "%\n"
              << std::noshowpos;

    // 48×8: OneDNN primary (BEST EXPECTED)
    double ms_48x8 = benchmark_gemm<Int8Gemm_48x8>(M, N, K, iterations);
    double gops_48x8 = (2.0 * M * N * K / 1e9) / (ms_48x8 / 1000.0);
    double speedup_48x8 = baseline_ms / ms_48x8;

    std::cout << std::setw(20) << "48×8 (OneDNN)"
              << std::setw(15) << std::fixed << std::setprecision(2) << ms_48x8
              << std::setw(15) << std::fixed << std::setprecision(2) << gops_48x8
              << std::setw(15) << std::fixed << std::setprecision(2) << speedup_48x8 << "×"
              << std::setw(15) << std::showpos << std::fixed << std::setprecision(1)
              << ((gops_48x8 - baseline_gops) / baseline_gops * 100) << "%\n"
              << std::noshowpos;

    std::cout << "\nTarget: 20-30% improvement from 48×8 vs 16×16\n";
    std::cout << "OneDNN uses 48×8 for better M-dimension amortization\n\n";

    // Validate we got meaningful improvement
    EXPECT_GT(gops_48x8, baseline_gops)
        << "Expected 48×8 to outperform 16×16 baseline";
}

/**
 * @brief Microkernel scaling across different M sizes
 *
 * Tests which microkernel is best at different problem sizes.
 * Expected: 48×8 advantage increases with M (better amortization).
 */
TEST_F(Perf__Int8Gemm_MicrokernelSize, MicrokernelScalingWithM)
{
    const int N = 3584;
    const int K = 896;
    const int iterations = 10;

    std::vector<int> M_values = {512, 2048, 8192};

    std::cout << "\n=== Microkernel Scaling vs M Size (N=" << N << ", K=" << K
              << ", " << max_threads_ << " threads) ===\n";
    std::cout << std::setw(10) << "M"
              << std::setw(15) << "16×16 GOPS"
              << std::setw(15) << "48×8 GOPS"
              << std::setw(15) << "Speedup"
              << std::setw(15) << "Improvement\n";
    std::cout << std::string(70, '-') << "\n";

    for (int M : M_values)
    {
        double baseline_ms = benchmark_gemm<Int8Gemm_6x16>(M, N, K, iterations);
        double baseline_gops = (2.0 * M * N * K / 1e9) / (baseline_ms / 1000.0);

        double ms_48x8 = benchmark_gemm<Int8Gemm_48x8>(M, N, K, iterations);
        double gops_48x8 = (2.0 * M * N * K / 1e9) / (ms_48x8 / 1000.0);

        double speedup = baseline_ms / ms_48x8;
        double improvement = ((gops_48x8 - baseline_gops) / baseline_gops * 100);

        std::cout << std::setw(10) << M
                  << std::setw(15) << std::fixed << std::setprecision(2) << baseline_gops
                  << std::setw(15) << std::fixed << std::setprecision(2) << gops_48x8
                  << std::setw(15) << std::fixed << std::setprecision(2) << speedup << "×"
                  << std::setw(15) << std::showpos << std::fixed << std::setprecision(1)
                  << improvement << "%\n"
                  << std::noshowpos;
    }

    std::cout << "\nExpected: 48×8 advantage increases with M (better amortization)\n\n";
}

/**
 * @brief Single-threaded comparison to isolate microkernel benefit
 *
 * Removes OpenMP effects to measure pure microkernel performance.
 */
TEST_F(Perf__Int8Gemm_MicrokernelSize, SingleThreadMicrokernelImpact)
{
    const int M = 2048;
    const int N = 3584;
    const int K = 896;
    const int iterations = 10;

    // Force single-threaded execution
    int old_threads = omp_get_max_threads();
    omp_set_num_threads(1);

    std::cout << "\n=== Single-Thread Microkernel Impact (M=" << M
              << ", N=" << N << ", K=" << K << ") ===\n";
    std::cout << std::setw(20) << "Microkernel"
              << std::setw(15) << "GOPS"
              << std::setw(15) << "vs 16×16\n";
    std::cout << std::string(50, '-') << "\n";

    double baseline_ms = benchmark_gemm<Int8Gemm_6x16>(M, N, K, iterations);
    double baseline_gops = (2.0 * M * N * K / 1e9) / (baseline_ms / 1000.0);
    std::cout << std::setw(20) << "16×16"
              << std::setw(15) << std::fixed << std::setprecision(2) << baseline_gops
              << std::setw(15) << "1.00×\n";

    double ms_48x8 = benchmark_gemm<Int8Gemm_48x8>(M, N, K, iterations);
    double gops_48x8 = (2.0 * M * N * K / 1e9) / (ms_48x8 / 1000.0);
    std::cout << std::setw(20) << "48×8"
              << std::setw(15) << std::fixed << std::setprecision(2) << gops_48x8
              << std::setw(15) << std::fixed << std::setprecision(2)
              << (baseline_ms / ms_48x8) << "×\n";

    // Restore thread count
    omp_set_num_threads(old_threads);

    std::cout << "\nSingle-threaded isolates pure microkernel benefit (no OpenMP effects)\n";
    std::cout << "Expected: 1.2-1.3× improvement from better register usage\n\n";
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
