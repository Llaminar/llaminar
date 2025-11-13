/**
 * @file Perf__Int8Gemm_Prefetch.cpp
 * @brief Benchmark comparing different prefetch distance configurations
 *
 * Tests the impact of OneDNN-style aggressive prefetching vs conservative vs none.
 * Expected: Significant performance improvement from prefetching (2-4× speedup).
 *
 * Prefetch configurations tested:
 * - OneDNN default: A=160, B=128, C=64 (5, 4, 2 cache lines ahead)
 * - Aggressive: A=256, B=192, C=96 (8, 6, 3 cache lines ahead)
 * - Conservative: A=64, B=64, C=32 (2, 2, 1 cache lines ahead)
 * - None: A=0, B=0, C=0 (baseline, no prefetching)
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

class Perf__Int8Gemm_Prefetch : public ::testing::Test
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
{ // Match M-scaling test: 10 iterations
    // Allocate aligned arrays (match M-scaling dimensions)
    std::vector<int8_t> A(M * K);
    std::vector<int8_t> B(K * N); // Fixed: K×N, not N×K
    std::vector<int32_t> C(M * N);

    // Initialize with random data
    std::mt19937 gen(12345);
    std::uniform_int_distribution<int> dist(-127, 127);
    for (auto &val : A)
        val = static_cast<int8_t>(dist(gen));
    for (auto &val : B)
        val = static_cast<int8_t>(dist(gen));

    // Warmup (match M-scaling test: 1 iteration)
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
 * @brief Compare prefetch configurations at M=2048 (plateau region)
 *
 * NOTE: Using same problem size as Perf__Int8Gemm_MScaling for fair comparison
 * - N=3584 (typical FFN width), K=896 (typical d_model)
 *
 * Expected results based on OneDNN analysis:
 * - No prefetch: ~100 GOPS (baseline, single-threaded equivalent)
 * - Conservative: ~200-300 GOPS (some improvement)
 * - OneDNN default: ~400-600 GOPS (2-4× improvement)
 * - Aggressive: ~450-650 GOPS (may overshoot, thrash cache)
 *
 * With 28 threads:
 * - No prefetch: ~1600 GOPS (current performance from M-scaling test)
 * - OneDNN default: ~3200-4800 GOPS (target: 2-3× improvement)
 * - Aggressive: ~3000-5000 GOPS (may not help if too far ahead)
 */
TEST_F(Perf__Int8Gemm_Prefetch, PrefetchDistanceComparison)
{
    const int M = 2048;
    const int N = 3584; // Match Perf__Int8Gemm_MScaling configuration
    const int K = 896;
    const int iterations = 10;

    std::cout << "\n=== Prefetch Distance Comparison (M=" << M << ", N=" << N
              << ", K=" << K << ", " << max_threads_ << " threads) ===\n";
    std::cout << std::setw(25) << "Configuration"
              << std::setw(15) << "Time (ms)"
              << std::setw(15) << "GOPS"
              << std::setw(15) << "Speedup"
              << std::setw(15) << "vs Baseline\n";
    std::cout << std::string(85, '-') << "\n";

    // Baseline: No prefetching
    double baseline_ms = benchmark_gemm<Int8Gemm_16x16_NoPrefetch>(M, N, K, iterations);
    double baseline_gops = (2.0 * M * N * K / 1e9) / (baseline_ms / 1000.0);

    std::cout << std::setw(25) << "No Prefetch (0,0,0)"
              << std::setw(15) << std::fixed << std::setprecision(2) << baseline_ms
              << std::setw(15) << std::fixed << std::setprecision(2) << baseline_gops
              << std::setw(15) << "1.00×"
              << std::setw(15) << "---\n";

    // Conservative: 64, 64, 32
    double cons_ms = benchmark_gemm<Int8Gemm_16x16_Prefetch64>(M, N, K, iterations);
    double cons_gops = (2.0 * M * N * K / 1e9) / (cons_ms / 1000.0);
    double cons_speedup = baseline_ms / cons_ms;

    std::cout << std::setw(25) << "Conservative (64,64,32)"
              << std::setw(15) << std::fixed << std::setprecision(2) << cons_ms
              << std::setw(15) << std::fixed << std::setprecision(2) << cons_gops
              << std::setw(15) << std::fixed << std::setprecision(2) << cons_speedup << "×"
              << std::setw(15) << std::showpos << std::fixed << std::setprecision(1)
              << ((cons_gops - baseline_gops) / baseline_gops * 100) << "%\n"
              << std::noshowpos;

    // OneDNN default: 160, 128, 64
    double onednn_ms = benchmark_gemm<Int8Gemm_6x16>(M, N, K, iterations); // Uses default 160,128,64
    double onednn_gops = (2.0 * M * N * K / 1e9) / (onednn_ms / 1000.0);
    double onednn_speedup = baseline_ms / onednn_ms;

    std::cout << std::setw(25) << "OneDNN (160,128,64)"
              << std::setw(15) << std::fixed << std::setprecision(2) << onednn_ms
              << std::setw(15) << std::fixed << std::setprecision(2) << onednn_gops
              << std::setw(15) << std::fixed << std::setprecision(2) << onednn_speedup << "×"
              << std::setw(15) << std::showpos << std::fixed << std::setprecision(1)
              << ((onednn_gops - baseline_gops) / baseline_gops * 100) << "%\n"
              << std::noshowpos;

    // Aggressive: 256, 192, 96
    double aggr_ms = benchmark_gemm<Int8Gemm_16x16_Prefetch256>(M, N, K, iterations);
    double aggr_gops = (2.0 * M * N * K / 1e9) / (aggr_ms / 1000.0);
    double aggr_speedup = baseline_ms / aggr_ms;

    std::cout << std::setw(25) << "Aggressive (256,192,96)"
              << std::setw(15) << std::fixed << std::setprecision(2) << aggr_ms
              << std::setw(15) << std::fixed << std::setprecision(2) << aggr_gops
              << std::setw(15) << std::fixed << std::setprecision(2) << aggr_speedup << "×"
              << std::setw(15) << std::showpos << std::fixed << std::setprecision(1)
              << ((aggr_gops - baseline_gops) / baseline_gops * 100) << "%\n"
              << std::noshowpos;

    std::cout << "\nTarget: 2-4× improvement with OneDNN prefetch distances\n";
    std::cout << "Goal: Achieve 3200-4800 GOPS (current baseline ~1600 GOPS)\n\n";

    // Validate we got meaningful improvement
    EXPECT_GT(onednn_gops, baseline_gops * 1.2)
        << "Expected at least 20% improvement from prefetching";
}

/**
 * @brief Prefetch impact across different M sizes
 *
 * Tests how prefetching affects performance at various problem sizes.
 * Expected: Larger M benefits more from prefetching (more data in flight).
 *
 * NOTE: Using N=3584, K=896 to match Perf__Int8Gemm_MScaling configuration
 */
TEST_F(Perf__Int8Gemm_Prefetch, PrefetchScalingWithM)
{
    const int N = 3584; // Match M-scaling test
    const int K = 896;
    const int iterations = 10;

    std::vector<int> M_values = {128, 512, 2048, 8192};

    std::cout << "\n=== Prefetch Impact vs M Size (N=" << N << ", K=" << K
              << ", " << max_threads_ << " threads) ===\n";
    std::cout << std::setw(10) << "M"
              << std::setw(20) << "No Prefetch GOPS"
              << std::setw(20) << "OneDNN GOPS"
              << std::setw(15) << "Speedup"
              << std::setw(15) << "Improvement\n";
    std::cout << std::string(80, '-') << "\n";

    for (int M : M_values)
    {
        double baseline_ms = benchmark_gemm<Int8Gemm_16x16_NoPrefetch>(M, N, K, iterations);
        double baseline_gops = (2.0 * M * N * K / 1e9) / (baseline_ms / 1000.0);

        double onednn_ms = benchmark_gemm<Int8Gemm_6x16>(M, N, K, iterations);
        double onednn_gops = (2.0 * M * N * K / 1e9) / (onednn_ms / 1000.0);

        double speedup = baseline_ms / onednn_ms;
        double improvement = ((onednn_gops - baseline_gops) / baseline_gops * 100);

        std::cout << std::setw(10) << M
                  << std::setw(20) << std::fixed << std::setprecision(2) << baseline_gops
                  << std::setw(20) << std::fixed << std::setprecision(2) << onednn_gops
                  << std::setw(15) << std::fixed << std::setprecision(2) << speedup << "×"
                  << std::setw(15) << std::showpos << std::fixed << std::setprecision(1)
                  << improvement << "%\n"
                  << std::noshowpos;
    }

    std::cout << "\nExpected: Speedup increases with M (more parallel work to hide latency)\n\n";
}

/**
 * @brief Single-threaded comparison to isolate prefetch benefit
 *
 * Removes OpenMP parallelization effects to measure pure prefetch impact.
 *
 * NOTE: Using N=3584, K=896 to match Perf__Int8Gemm_MScaling configuration
 */
TEST_F(Perf__Int8Gemm_Prefetch, SingleThreadPrefetchImpact)
{
    const int M = 2048;
    const int N = 3584; // Match M-scaling test
    const int K = 896;
    const int iterations = 10;

    // Force single-threaded execution
    int old_threads = omp_get_max_threads();
    omp_set_num_threads(1);

    std::cout << "\n=== Single-Thread Prefetch Impact (M=" << M
              << ", N=" << N << ", K=" << K << ") ===\n";
    std::cout << std::setw(25) << "Configuration"
              << std::setw(15) << "GOPS"
              << std::setw(15) << "vs No Prefetch\n";
    std::cout << std::string(55, '-') << "\n";

    double baseline_ms = benchmark_gemm<Int8Gemm_16x16_NoPrefetch>(M, N, K, iterations);
    double baseline_gops = (2.0 * M * N * K / 1e9) / (baseline_ms / 1000.0);
    std::cout << std::setw(25) << "No Prefetch"
              << std::setw(15) << std::fixed << std::setprecision(2) << baseline_gops
              << std::setw(15) << "1.00×\n";

    double onednn_ms = benchmark_gemm<Int8Gemm_6x16>(M, N, K, iterations);
    double onednn_gops = (2.0 * M * N * K / 1e9) / (onednn_ms / 1000.0);
    std::cout << std::setw(25) << "OneDNN Prefetch"
              << std::setw(15) << std::fixed << std::setprecision(2) << onednn_gops
              << std::setw(15) << std::fixed << std::setprecision(2)
              << (baseline_ms / onednn_ms) << "×\n";

    // Restore thread count
    omp_set_num_threads(old_threads);

    std::cout << "\nSingle-threaded isolates pure prefetch benefit (no OpenMP effects)\n";
    std::cout << "Expected: 1.5-3× improvement even at 1 thread\n\n";
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
