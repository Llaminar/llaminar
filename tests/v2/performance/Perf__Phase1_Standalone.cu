/**
 * @file Perf__Phase1_Standalone.cu
 * @brief Standalone Phase 1 CUDA kernel performance test
 *
 * This test directly benchmarks baseline vs Phase 1 optimized CUDA kernels
 * without going through the full pipeline. It:
 *   - Allocates CUDA memory directly
 *   - Calls kernel launchers directly (no autotuner)
 *   - Measures pure kernel performance (no device management overhead)
 *
 * This is a minimal test to validate Phase 1 optimizations work as expected
 * before full pipeline integration.
 *
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>
#include <cmath>

// CUDA kernel launchers
#include "kernels/cuda/CudaGemmVariantsBaseline.h"
#include "kernels/cuda/CudaGemmVariantsMemoryOpt.h"
#include "kernels/cuda/IQ4_NL_BlockDecoder.h"

using namespace llaminar2::cuda;

/**
 * @brief Statistics for a benchmark run
 */
struct BenchmarkStats
{
    double mean_ms = 0.0;
    double stddev_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;
    double mean_gflops = 0.0;
    double cv_percent = 0.0; // Coefficient of variation
};

/**
 * @brief Phase 1 standalone kernel test fixture
 */
class Phase1_Standalone : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Check CUDA device availability
        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err != cudaSuccess || device_count == 0)
        {
            GTEST_SKIP() << "No CUDA device available";
        }

        // Get device properties
        cudaGetDeviceProperties(&device_props_, 0);
        std::cout << "[CUDA Device] " << device_props_.name << "\n";
        std::cout << "  Compute capability: " << device_props_.major << "." << device_props_.minor << "\n";
        std::cout << "  Global memory: " << (device_props_.totalGlobalMem / (1024 * 1024 * 1024)) << " GB\n";
        std::cout << "  Shared memory per block: " << (device_props_.sharedMemPerBlock / 1024) << " KB\n";
        std::cout << std::endl;

        // Create CUDA events for timing
        cudaEventCreate(&start_event_);
        cudaEventCreate(&stop_event_);
    }

    void TearDown() override
    {
        cudaEventDestroy(start_event_);
        cudaEventDestroy(stop_event_);
    }

    /**
     * @brief Allocate and initialize test data on CUDA device
     */
    void allocateTestData(int m, int n, int k)
    {
        // Allocate activation (A) - FP32, m×k
        size_t A_size = m * k * sizeof(float);
        cudaMalloc(&d_A_, A_size);

        // Initialize with random-ish data on host then copy
        std::vector<float> h_A(m * k);
        for (int i = 0; i < m * k; ++i)
        {
            h_A[i] = (static_cast<float>(i % 1000) / 1000.0f) - 0.5f; // [-0.5, 0.5]
        }
        cudaMemcpy(d_A_, h_A.data(), A_size, cudaMemcpyHostToDevice);

        // Allocate weight (B) - IQ4_NL blocks, k×n
        // IQ4_NL: 32 elements per block, 2 bytes per block + 2 bytes scale
        int n_blocks = (k / 32) * n;
        size_t B_size = n_blocks * sizeof(IQ4_NLBlock);
        cudaMalloc(&d_B_, B_size);

        // Initialize with valid IQ4_NL data
        std::vector<IQ4_NLBlock> h_B(n_blocks);
        for (int i = 0; i < n_blocks; ++i)
        {
            // Simple pattern: alternating indices
            for (int j = 0; j < 16; ++j)
            {
                h_B[i].qs[j] = (j % 16); // Valid IQ4_NL indices [0-15]
            }
            h_B[i].d = 1.0f; // Scale factor
        }
        cudaMemcpy(d_B_, h_B.data(), B_size, cudaMemcpyHostToDevice);

        // Allocate output (C) - FP32, m×n
        size_t C_size = m * n * sizeof(float);
        cudaMalloc(&d_C_, C_size);
        cudaMemset(d_C_, 0, C_size); // Zero initialize
    }

    void freeTestData()
    {
        if (d_A_)
            cudaFree(d_A_);
        if (d_B_)
            cudaFree(d_B_);
        if (d_C_)
            cudaFree(d_C_);
        d_A_ = nullptr;
        d_B_ = nullptr;
        d_C_ = nullptr;
    }

    /**
     * @brief Benchmark a kernel configuration
     */
    BenchmarkStats benchmarkKernel(
        int m, int n, int k,
        CudaGemmConfig config,
        bool use_optimized,
        int warmup_iters,
        int bench_iters)
    {
        std::vector<float> times_ms;
        times_ms.reserve(bench_iters);

        // Warmup iterations
        for (int i = 0; i < warmup_iters; ++i)
        {
            cudaError_t err;
            if (use_optimized)
            {
                err = launchIQ4NLGemmVariantOptimized(d_A_, d_B_, d_C_, m, n, k, config, nullptr);
            }
            else
            {
                err = launchIQ4NLGemmVariant(d_A_, d_B_, d_C_, m, n, k, config, nullptr);
            }
            EXPECT_EQ(err, cudaSuccess) << "Warmup kernel launch failed";
        }
        cudaDeviceSynchronize();

        // Benchmark iterations
        for (int i = 0; i < bench_iters; ++i)
        {
            cudaEventRecord(start_event_);

            cudaError_t err;
            if (use_optimized)
            {
                if (i == 0)
                    std::cout << "[DEBUG] Calling launchIQ4NLGemmVariantOptimized" << std::endl;
                err = launchIQ4NLGemmVariantOptimized(d_A_, d_B_, d_C_, m, n, k, config, nullptr);
            }
            else
            {
                if (i == 0)
                    std::cout << "[DEBUG] Calling launchIQ4NLGemmVariant (baseline)" << std::endl;
                err = launchIQ4NLGemmVariant(d_A_, d_B_, d_C_, m, n, k, config, nullptr);
            }
            EXPECT_EQ(err, cudaSuccess) << "Benchmark kernel launch failed";

            cudaEventRecord(stop_event_);
            cudaEventSynchronize(stop_event_);

            float time_ms = 0.0f;
            cudaEventElapsedTime(&time_ms, start_event_, stop_event_);
            times_ms.push_back(time_ms);
        }

        // Compute statistics
        BenchmarkStats stats;
        double sum = 0.0;
        stats.min_ms = times_ms[0];
        stats.max_ms = times_ms[0];

        for (float t : times_ms)
        {
            sum += t;
            stats.min_ms = std::min(stats.min_ms, static_cast<double>(t));
            stats.max_ms = std::max(stats.max_ms, static_cast<double>(t));
        }

        stats.mean_ms = sum / bench_iters;

        // Compute standard deviation
        double var_sum = 0.0;
        for (float t : times_ms)
        {
            double diff = t - stats.mean_ms;
            var_sum += diff * diff;
        }
        stats.stddev_ms = std::sqrt(var_sum / bench_iters);
        stats.cv_percent = (stats.stddev_ms / stats.mean_ms) * 100.0;

        // Compute GFLOPS: 2*m*n*k operations (multiply + add)
        double flops = 2.0 * m * n * k;
        stats.mean_gflops = flops / (stats.mean_ms * 1e6);

        return stats;
    }

    void printResults(const std::string &name, int m, int n, int k, const BenchmarkStats &stats)
    {
        std::cout << "\n╔════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║ " << std::left << std::setw(62) << name << " ║\n";
        std::cout << "╠════════════════════════════════════════════════════════════════╣\n";
        std::cout << "║ Shape: " << m << "×" << n << "×" << k << std::string(53 - (std::to_string(m).length() + std::to_string(n).length() + std::to_string(k).length()), ' ') << "║\n";
        std::cout << "║ Time:      " << std::fixed << std::setprecision(2) << std::setw(8) << stats.mean_ms
                  << " ± " << std::setw(6) << stats.stddev_ms << " ms"
                  << " (CV: " << std::setw(4) << std::setprecision(1) << stats.cv_percent << "%)"
                  << std::string(14, ' ') << "║\n";
        std::cout << "║ Throughput:" << std::fixed << std::setprecision(2) << std::setw(10) << stats.mean_gflops
                  << " GFLOPS" << std::string(35, ' ') << "║\n";
        std::cout << "║ Range:     " << std::fixed << std::setprecision(2) << std::setw(8) << stats.min_ms
                  << " - " << std::setw(8) << stats.max_ms << " ms"
                  << std::string(25, ' ') << "║\n";
        std::cout << "╚════════════════════════════════════════════════════════════════╝\n";
    }

    cudaDeviceProp device_props_;
    cudaEvent_t start_event_ = nullptr;
    cudaEvent_t stop_event_ = nullptr;

    // Device pointers
    float *d_A_ = nullptr;
    IQ4_NLBlock *d_B_ = nullptr;
    float *d_C_ = nullptr;
};

/**
 * @brief Test: Small batch (32×896) - Baseline vs Optimized
 *
 * Expected baseline: ~585 GFLOPS (from benchmark data)
 * Expected optimized: ~1200-1800 GFLOPS (2-3× speedup)
 */
TEST_F(Phase1_Standalone, SmallBatch_32x896)
{
    const int m = 32;
    const int n = 896;
    const int k = 896;

    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║         PHASE 1 STANDALONE: SMALL BATCH (32×896)              ║\n";
    std::cout << "╠════════════════════════════════════════════════════════════════╣\n";
    std::cout << "║ Baseline expected:  ~585 GFLOPS                                ║\n";
    std::cout << "║ Optimized target:   1200-1800 GFLOPS (2-3× speedup)           ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n";

    // Allocate test data
    allocateTestData(m, n, k);

    // Use exact instantiated configuration from CudaGemmVariants.cu (BASELINE)
    // LAUNCH_VARIANT(64, 64, 32, 16, 16, 4, 4, 1, true, true);
    CudaGemmConfig config;
    config.tile_m = 64;
    config.tile_n = 64;
    config.tile_k = 32;
    config.threads_m = 16; // Different from optimized! Baseline uses 16×16
    config.threads_n = 16;
    config.work_per_thread_m = 4; // work = tile / threads
    config.work_per_thread_n = 4;
    config.prefetch_stages = 1;   // Baseline uses pipelining
    config.transpose_smem = true; // Baseline transposes
    config.vectorize_load = 1;    // Baseline uses scalar loads (VEC param means something else)
    config.transpose_smem = false;
    config.vectorize_load = 4;

    // Benchmark baseline
    auto baseline_stats = benchmarkKernel(m, n, k, config, false, 5, 100);
    printResults("BASELINE KERNEL", m, n, k, baseline_stats);

    // Benchmark optimized
    auto optimized_stats = benchmarkKernel(m, n, k, config, true, 5, 100);
    printResults("OPTIMIZED KERNEL (Phase 1)", m, n, k, optimized_stats);

    // Check correctness: Copy results back to CPU and compare first few elements
    std::vector<float> h_C_baseline(m * n);
    std::vector<float> h_C_optimized(m * n);

    // Run baseline and copy result
    cudaMemset(d_C_, 0, m * n * sizeof(float));
    launchIQ4NLGemmVariant(d_A_, d_B_, d_C_, m, n, k, config, nullptr);
    cudaMemcpy(h_C_baseline.data(), d_C_, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    // Clear output and run optimized
    cudaMemset(d_C_, 0, m * n * sizeof(float));
    launchIQ4NLGemmVariantOptimized(d_A_, d_B_, d_C_, m, n, k, config, nullptr);
    cudaMemcpy(h_C_optimized.data(), d_C_, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    // Compare first 10 elements
    std::cout << "\n[CORRECTNESS CHECK]" << std::endl;
    std::cout << "First 10 elements comparison:" << std::endl;
    std::cout << "  Baseline:  ";
    for (int i = 0; i < std::min(10, m * n); ++i)
    {
        std::cout << h_C_baseline[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "  Optimized: ";
    for (int i = 0; i < std::min(10, m * n); ++i)
    {
        std::cout << h_C_optimized[i] << " ";
    }
    std::cout << std::endl;

    // Calculate max diff
    float max_diff = 0.0f;
    for (int i = 0; i < m * n; ++i)
    {
        max_diff = std::max(max_diff, std::abs(h_C_baseline[i] - h_C_optimized[i]));
    }
    std::cout << "  Max difference: " << max_diff << std::endl;

    // Compute speedup
    double speedup = baseline_stats.mean_gflops / optimized_stats.mean_gflops;
    if (speedup < 1.0)
        speedup = 1.0 / speedup; // Handle if optimized is slower

    std::cout << "\n╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                        COMPARISON                              ║\n";
    std::cout << "╠════════════════════════════════════════════════════════════════╣\n";
    std::cout << "║ Baseline:   " << std::fixed << std::setprecision(2) << std::setw(10) << baseline_stats.mean_gflops << " GFLOPS" << std::string(34, ' ') << "║\n";
    std::cout << "║ Optimized:  " << std::fixed << std::setprecision(2) << std::setw(10) << optimized_stats.mean_gflops << " GFLOPS" << std::string(34, ' ') << "║\n";
    std::cout << "║ Speedup:    " << std::fixed << std::setprecision(2) << std::setw(10) << speedup << "×" << std::string(39, ' ') << "║\n";
    std::cout << "╠════════════════════════════════════════════════════════════════╣\n";

    if (speedup >= 2.0)
    {
        std::cout << "║ Status:     ✓ TARGET MET (≥2× speedup)                        ║\n";
    }
    else if (speedup >= 1.5)
    {
        std::cout << "║ Status:     ~ PARTIAL (1.5-2× speedup)                        ║\n";
    }
    else
    {
        std::cout << "║ Status:     ✗ BELOW TARGET (<1.5× speedup)                    ║\n";
    }

    std::cout << "╚════════════════════════════════════════════════════════════════╝\n";

    // Validate that optimized is faster
    EXPECT_GT(optimized_stats.mean_gflops, baseline_stats.mean_gflops * 1.2)
        << "Optimized kernel should be at least 1.2× faster than baseline";

    freeTestData();
}

/**
 * @brief Test: Single token (1×896) - Baseline vs Optimized
 *
 * Expected baseline: ~22.7 GFLOPS
 * Expected optimized: ~50-100 GFLOPS (2-4× speedup)
 */
TEST_F(Phase1_Standalone, SingleToken_1x896)
{
    const int m = 1;
    const int n = 896;
    const int k = 896;

    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║        PHASE 1 STANDALONE: SINGLE TOKEN (1×896)               ║\n";
    std::cout << "╠════════════════════════════════════════════════════════════════╣\n";
    std::cout << "║ Baseline expected:  ~22.7 GFLOPS                               ║\n";
    std::cout << "║ Optimized target:   50-100 GFLOPS (2-4× speedup)              ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n";

    // Allocate test data
    allocateTestData(m, n, k);

    // Use exact instantiated configuration from CudaGemmVariantsOptimized.cu
    // LAUNCH_VARIANT(16, 16, 32, 1, 1, 16, 16, 0, false, 4);
    CudaGemmConfig config;
    config.tile_m = 16;
    config.tile_n = 16;
    config.tile_k = 32;
    config.threads_m = 1;
    config.threads_n = 1;
    config.work_per_thread_m = 16;
    config.work_per_thread_n = 16;
    config.prefetch_stages = 0;
    config.transpose_smem = false;
    config.vectorize_load = 4;

    // Benchmark baseline
    auto baseline_stats = benchmarkKernel(m, n, k, config, false, 10, 1000);
    printResults("BASELINE KERNEL", m, n, k, baseline_stats);

    // Benchmark optimized
    auto optimized_stats = benchmarkKernel(m, n, k, config, true, 10, 1000);
    printResults("OPTIMIZED KERNEL (Phase 1)", m, n, k, optimized_stats);

    // Compute speedup
    double speedup = baseline_stats.mean_gflops / optimized_stats.mean_gflops;
    if (speedup < 1.0)
        speedup = 1.0 / speedup;

    std::cout << "\n╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                        COMPARISON                              ║\n";
    std::cout << "╠════════════════════════════════════════════════════════════════╣\n";
    std::cout << "║ Baseline:   " << std::fixed << std::setprecision(2) << std::setw(10) << baseline_stats.mean_gflops << " GFLOPS" << std::string(34, ' ') << "║\n";
    std::cout << "║ Optimized:  " << std::fixed << std::setprecision(2) << std::setw(10) << optimized_stats.mean_gflops << " GFLOPS" << std::string(34, ' ') << "║\n";
    std::cout << "║ Speedup:    " << std::fixed << std::setprecision(2) << std::setw(10) << speedup << "×" << std::string(39, ' ') << "║\n";
    std::cout << "╠════════════════════════════════════════════════════════════════╣\n";

    if (speedup >= 2.0)
    {
        std::cout << "║ Status:     ✓ TARGET MET (≥2× speedup)                        ║\n";
    }
    else if (speedup >= 1.5)
    {
        std::cout << "║ Status:     ~ PARTIAL (1.5-2× speedup)                        ║\n";
    }
    else
    {
        std::cout << "║ Status:     ✗ BELOW TARGET (<1.5× speedup)                    ║\n";
    }

    std::cout << "╚════════════════════════════════════════════════════════════════╝\n";

    EXPECT_GT(optimized_stats.mean_gflops, baseline_stats.mean_gflops * 1.2)
        << "Optimized kernel should be at least 1.2× faster than baseline";

    freeTestData();
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
