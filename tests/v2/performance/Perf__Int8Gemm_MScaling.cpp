/**
 * @file Perf__Int8Gemm_MScaling.cpp
 * @brief Benchmark INT8 GEMM performance scaling with increasing M
 *
 * Tests how performance scales as M increases from 128 to 4096,
 * comparing single-threaded vs multi-threaded execution to identify
 * the point where parallelization becomes beneficial.
 */

#include "kernels/cpu/gemm_v2/ParameterizedInt8Gemm.h"
#include <gtest/gtest.h>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <random>
#include <omp.h>

using namespace llaminar::v2::kernels::cpu;

class Int8GemmMScalingTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        gen_.seed(42);
    }

    // Generate random INT8 matrix
    std::vector<int8_t> generateMatrix(size_t rows, size_t cols)
    {
        std::vector<int8_t> mat(rows * cols);
        std::uniform_int_distribution<> dis(-128, 127);
        for (auto &v : mat)
        {
            v = dis(gen_);
        }
        return mat;
    }

    // Run benchmark for given M size
    struct BenchmarkResult
    {
        int M;
        int num_threads;
        double time_ms;
        double gops;
        double throughput_mb_s; // Memory bandwidth utilization
    };

    BenchmarkResult runBenchmark(int M, int N, int K, int num_threads)
    {
        // Set thread count
        omp_set_num_threads(num_threads);

        // Generate test data
        auto A = generateMatrix(M, K);
        auto B = generateMatrix(K, N);
        std::vector<int32_t> C(M * N);

        // Leading dimensions
        const int lda = K;
        const int ldb = K;
        const int ldc = N;

        // Warmup
        Int8Gemm_6x16::gemm(M, N, K, A.data(), lda, B.data(), ldb, C.data(), ldc);

        // Benchmark (10 iterations)
        const int iterations = 10;
        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < iterations; ++i)
        {
            Int8Gemm_6x16::gemm(M, N, K, A.data(), lda, B.data(), ldb, C.data(), ldc);
        }

        auto end = std::chrono::high_resolution_clock::now();

        // Calculate metrics
        double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double avg_ms = total_ms / iterations;
        double gflops = (2.0 * M * N * K) / (avg_ms * 1e6);

        // Memory bandwidth (read A, B, write C)
        size_t bytes = M * K * sizeof(int8_t) + // Read A
                       K * N * sizeof(int8_t) + // Read B
                       M * N * sizeof(int32_t); // Write C
        double throughput_mb_s = (bytes / 1e6) / (avg_ms / 1000.0);

        return {M, num_threads, avg_ms, gflops, throughput_mb_s};
    }

    std::mt19937 gen_;
};

TEST_F(Int8GemmMScalingTest, MScaling_SingleVsMultiThread)
{
    const int N = 3584; // Typical FFN width
    const int K = 896;  // Typical FFN depth

    // Test M from 128 to 16384 (powers of 2) to find scaling plateau
    std::vector<int> M_values = {128, 256, 512, 1024, 2048, 4096, 8192, 16384};

    std::cout << "\n=== INT8 GEMM M-Scaling Analysis ===" << std::endl;
    std::cout << "Problem: M varies, N=" << N << ", K=" << K << std::endl;
    std::cout << "FLOPs per iteration: varies with M\n"
              << std::endl;

    // Table header
    std::cout << std::setw(6) << "M"
              << std::setw(10) << "Threads"
              << std::setw(12) << "Time (ms)"
              << std::setw(12) << "GOPS"
              << std::setw(14) << "BW (MB/s)"
              << std::setw(12) << "Speedup"
              << std::setw(14) << "Efficiency"
              << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    for (int M : M_values)
    {
        // Run single-threaded
        auto result_1t = runBenchmark(M, N, K, 1);

        // Run multi-threaded (get from environment, not omp_get_max_threads which returns 1 before parallel region)
        int max_threads = 28; // Default
        const char *omp_num_threads_env = std::getenv("OMP_NUM_THREADS");
        if (omp_num_threads_env)
        {
            max_threads = std::atoi(omp_num_threads_env);
        }
        auto result_mt = runBenchmark(M, N, K, max_threads);

        // Calculate speedup and efficiency
        double speedup = result_1t.time_ms / result_mt.time_ms;
        double efficiency = (speedup / max_threads) * 100.0;

        // Print single-threaded results
        std::cout << std::setw(6) << M
                  << std::setw(10) << 1
                  << std::setw(12) << std::fixed << std::setprecision(3) << result_1t.time_ms
                  << std::setw(12) << std::fixed << std::setprecision(2) << result_1t.gops
                  << std::setw(14) << std::fixed << std::setprecision(1) << result_1t.throughput_mb_s
                  << std::setw(12) << "-"
                  << std::setw(14) << "-"
                  << std::endl;

        // Print multi-threaded results
        std::cout << std::setw(6) << M
                  << std::setw(10) << max_threads
                  << std::setw(12) << std::fixed << std::setprecision(3) << result_mt.time_ms
                  << std::setw(12) << std::fixed << std::setprecision(2) << result_mt.gops
                  << std::setw(14) << std::fixed << std::setprecision(1) << result_mt.throughput_mb_s
                  << std::setw(12) << std::fixed << std::setprecision(2) << speedup << "×"
                  << std::setw(13) << std::fixed << std::setprecision(1) << efficiency << "%"
                  << std::endl;

        std::cout << std::endl;
    }

    std::cout << "\nKey Metrics:" << std::endl;
    std::cout << "- GOPS: Giga-operations per second (higher is better)" << std::endl;
    std::cout << "- BW: Memory bandwidth in MB/s" << std::endl;
    std::cout << "- Speedup: Multi-thread time / Single-thread time" << std::endl;
    std::cout << "- Efficiency: (Speedup / Num_Threads) × 100%" << std::endl;
    std::cout << "\nGood scaling indicators:" << std::endl;
    std::cout << "- Speedup should increase with M (more work to parallelize)" << std::endl;
    std::cout << "- Efficiency >70% indicates good parallelization" << std::endl;
    std::cout << "- Efficiency <30% indicates parallelization overhead dominates" << std::endl;
}

TEST_F(Int8GemmMScalingTest, MScaling_DetailedThreadCount)
{
    const int N = 3584;
    const int K = 896;
    const int M = 2048; // Focus on large M

    // Get max threads from environment (omp_get_max_threads() returns 1 before first parallel region)
    int max_threads = 56; // Default to max hyperthreads
    const char *omp_num_threads_env = std::getenv("OMP_NUM_THREADS");
    if (omp_num_threads_env)
    {
        max_threads = std::atoi(omp_num_threads_env);
    }

    std::vector<int> thread_counts = {1, 2, 4, 8, 16};
    if (max_threads >= 28)
    {
        thread_counts.push_back(28);
    }
    if (max_threads >= 56)
    {
        thread_counts.push_back(56);
    }

    std::cout << "\n=== Thread Scaling Analysis (M=" << M << ") ===" << std::endl;
    std::cout << "Problem: M=" << M << ", N=" << N << ", K=" << K << std::endl;
    std::cout << "FLOPs: " << (2.0 * M * N * K / 1e9) << " GFLOPS\n"
              << std::endl;

    std::cout << std::setw(10) << "Threads"
              << std::setw(12) << "Time (ms)"
              << std::setw(12) << "GOPS"
              << std::setw(12) << "Speedup"
              << std::setw(14) << "Efficiency"
              << std::endl;
    std::cout << std::string(60, '-') << std::endl;

    double baseline_time = 0.0;

    for (int threads : thread_counts)
    {
        if (threads > max_threads)
            continue;

        auto result = runBenchmark(M, N, K, threads);

        if (threads == 1)
        {
            baseline_time = result.time_ms;
        }

        double speedup = baseline_time / result.time_ms;
        double efficiency = (speedup / threads) * 100.0;

        std::cout << std::setw(10) << threads
                  << std::setw(12) << std::fixed << std::setprecision(3) << result.time_ms
                  << std::setw(12) << std::fixed << std::setprecision(2) << result.gops
                  << std::setw(12) << std::fixed << std::setprecision(2) << speedup << "×"
                  << std::setw(13) << std::fixed << std::setprecision(1) << efficiency << "%"
                  << std::endl;
    }

    std::cout << "\nInterpretation:" << std::endl;
    std::cout << "- Linear scaling: Speedup ≈ Num_Threads (100% efficiency)" << std::endl;
    std::cout << "- Good scaling: Efficiency >70%" << std::endl;
    std::cout << "- Poor scaling: Efficiency <50% (overhead dominates)" << std::endl;
}

TEST_F(Int8GemmMScalingTest, MScaling_IdentifyBreakpoint)
{
    const int N = 3584;
    const int K = 896;

    // Fine-grained M sweep to find scaling breakpoint and plateau
    std::vector<int> M_values;
    for (int M = 64; M <= 512; M += 64)
    {
        M_values.push_back(M);
    }
    for (int M = 512; M <= 2048; M += 256)
    {
        if (M > 512)
            M_values.push_back(M);
    }
    // Add larger M values to find plateau (go up to 16384)
    for (int M = 2048; M <= 16384; M += 1024)
    {
        if (M > 2048)
            M_values.push_back(M);
    }

    // Get max threads from environment (omp_get_max_threads() returns 1 before first parallel region)
    int max_threads = 28; // Default
    const char *omp_num_threads_env = std::getenv("OMP_NUM_THREADS");
    if (omp_num_threads_env)
    {
        max_threads = std::atoi(omp_num_threads_env);
    }

    std::cout << "\n=== Scaling Breakpoint Analysis ===" << std::endl;
    std::cout << "Finding M where multi-threading becomes beneficial" << std::endl;
    std::cout << "Threads: " << max_threads << "\n"
              << std::endl;

    std::cout << std::setw(6) << "M"
              << std::setw(10) << "1T GOPS"
              << std::setw(12) << max_threads << "T GOPS"
              << std::setw(12) << "Speedup"
              << std::setw(14) << "Efficiency"
              << std::setw(14) << "Beneficial?"
              << std::endl;
    std::cout << std::string(70, '-') << std::endl;

    int first_beneficial_M = -1;

    for (int M : M_values)
    {
        auto result_1t = runBenchmark(M, N, K, 1);
        auto result_mt = runBenchmark(M, N, K, max_threads);

        double speedup = result_1t.time_ms / result_mt.time_ms;
        double efficiency = (speedup / max_threads) * 100.0;
        bool beneficial = speedup > 1.5; // At least 1.5× speedup to be worthwhile

        if (beneficial && first_beneficial_M == -1)
        {
            first_beneficial_M = M;
        }

        std::cout << std::setw(6) << M
                  << std::setw(10) << std::fixed << std::setprecision(2) << result_1t.gops
                  << std::setw(12) << std::fixed << std::setprecision(2) << result_mt.gops
                  << std::setw(12) << std::fixed << std::setprecision(2) << speedup << "×"
                  << std::setw(13) << std::fixed << std::setprecision(1) << efficiency << "%"
                  << std::setw(14) << (beneficial ? "✓ YES" : "✗ NO")
                  << std::endl;
    }

    std::cout << "\n=== Conclusion ===" << std::endl;
    if (first_beneficial_M > 0)
    {
        std::cout << "Multi-threading becomes beneficial at M ≥ " << first_beneficial_M << std::endl;
        std::cout << "For M < " << first_beneficial_M << ", single-threaded is competitive." << std::endl;
    }
    else
    {
        std::cout << "Multi-threading not beneficial even at largest M tested." << std::endl;
        std::cout << "OpenMP overhead exceeds parallelization benefit." << std::endl;
    }
}
