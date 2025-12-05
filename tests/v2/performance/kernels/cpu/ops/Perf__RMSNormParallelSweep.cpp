/**
 * @file Perf__RMSNormParallelSweep.cpp
 * @brief Parallelization threshold sweep for Q8_1 RMSNorm
 *
 * This test empirically determines optimal parallelization thresholds by
 * sweeping different configurations and comparing against single-threaded
 * performance.
 *
 * Usage:
 *   OMP_NUM_THREADS=1 ./v2_perf_rmsnorm_parallel_sweep  # Single-threaded baseline
 *   ./v2_perf_rmsnorm_parallel_sweep                     # Multi-threaded sweep
 *
 * Environment Variables:
 *   LLAMINAR_RMSNORM_PARALLEL_MIN_ROWS - Min rows for parallelism (default: 8)
 *   LLAMINAR_RMSNORM_PARALLEL_MIN_ELEMS - Min elements for parallelism (default: 65536)
 *   LLAMINAR_RMSNORM_MIN_ELEMS_PER_THREAD - Min elements per thread (default: 8192)
 *   LLAMINAR_RMSNORM_MAX_THREADS - Max threads to use (0=unlimited)
 *
 * @author David Sanftenberg
 * @date 2025-12-05
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <omp.h>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <random>
#include <cstdlib>

#include "tensors/BlockStructures.h"
#include "kernels/cpu/primitives/RMSNormPrimitives.h"
#include "tensors/SIMDHelpers.h"
#include "utils/DebugEnv.h"

using namespace llaminar2;
using namespace llaminar2::primitives;

// ============================================================================
// Test Configuration
// ============================================================================

struct SweepConfig
{
    int seq_len;
    int d_model;
    std::string description;
};

// Representative workloads from real inference
static const std::vector<SweepConfig> SWEEP_CONFIGS = {
    // Decode: Single token cases (where OpenMP overhead is most problematic)
    {1, 896, "Decode 1×896 (Qwen 0.5B)"},
    {1, 1536, "Decode 1×1536 (Qwen 1.5B)"},
    {1, 2048, "Decode 1×2048 (Qwen 3B)"},
    {1, 3584, "Decode 1×3584 (Qwen 7B)"},
    {1, 5120, "Decode 1×5120 (Qwen 14B)"},
    {1, 8192, "Decode 1×8192 (Qwen 72B)"},

    // Small prefill
    {8, 896, "Prefill 8×896"},
    {8, 3584, "Prefill 8×3584"},

    // Medium prefill (crossover region)
    {16, 896, "Prefill 16×896"},
    {16, 3584, "Prefill 16×3584"},
    {32, 896, "Prefill 32×896"},
    {32, 3584, "Prefill 32×3584"},
    {64, 896, "Prefill 64×896"},
    {64, 3584, "Prefill 64×3584"},

    // Large prefill
    {128, 896, "Prefill 128×896"},
    {128, 3584, "Prefill 128×3584"},
    {256, 3584, "Prefill 256×3584"},
    {512, 3584, "Prefill 512×3584"},
    {1024, 3584, "Prefill 1024×3584"},
    {2048, 3584, "Prefill 2048×3584"},
};

// ============================================================================
// Test Fixture
// ============================================================================

class RMSNormParallelSweep : public ::testing::Test
{
protected:
    int rank_ = 0;
    int world_size_ = 1;
    std::mt19937 rng_{42};

    void SetUp() override
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
    }

    // Benchmark single configuration with given thread count
    double bench_q8_1(int seq_len, int d_model, int warmup, int iters)
    {
        size_t blocks_per_row = (d_model + 31) / 32;
        size_t total_blocks = seq_len * blocks_per_row;

        std::vector<Q8_1Block> input(total_blocks);
        std::vector<float> gamma(d_model);
        std::vector<Q8_1Block> output(total_blocks);

        // Initialize with realistic random data
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::uniform_real_distribution<float> gamma_dist(0.5f, 2.0f);

        size_t fp32_size = seq_len * d_model;
        std::vector<float> fp32_temp(fp32_size);
        for (auto &v : fp32_temp)
            v = dist(rng_);
        simd::quantize_fp32_to_q8_1_blocks(fp32_temp.data(), input.data(), fp32_size);

        for (auto &g : gamma)
            g = gamma_dist(rng_);

        RMSNormExecOptions opts;
        opts.allow_parallel = true;

        // Warmup
        for (int i = 0; i < warmup; ++i)
        {
            rmsnorm_q8_1_pure_integer(input.data(), gamma.data(), output.data(),
                                      seq_len, blocks_per_row, 1e-6f, opts);
        }

        // Benchmark
        std::vector<double> times_us;
        times_us.reserve(iters);

        for (int i = 0; i < iters; ++i)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            auto start = std::chrono::high_resolution_clock::now();

            rmsnorm_q8_1_pure_integer(input.data(), gamma.data(), output.data(),
                                      seq_len, blocks_per_row, 1e-6f, opts);

            auto end = std::chrono::high_resolution_clock::now();
            double us = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1000.0;
            times_us.push_back(us);
        }

        // Return median to reduce noise
        std::sort(times_us.begin(), times_us.end());
        return times_us[times_us.size() / 2];
    }
};

// ============================================================================
// Baseline Test: Single-Threaded Reference
// ============================================================================

TEST_F(RMSNormParallelSweep, SingleThreadBaseline)
{
    if (rank_ != 0)
        return;

    std::cout << "\n";
    std::cout << "================================================================\n";
    std::cout << "Q8_1 RMSNorm Single-Thread Baseline\n";
    std::cout << "OMP_NUM_THREADS=" << omp_get_max_threads() << "\n";
    std::cout << "================================================================\n";
    std::cout << std::left << std::setw(28) << "Configuration"
              << std::right << std::setw(12) << "Elements"
              << std::setw(12) << "Time (µs)"
              << std::setw(14) << "Throughput"
              << "\n";
    std::cout << std::string(66, '-') << "\n";

    // Force sequential execution
    setenv("LLAMINAR_RMSNORM_PARALLEL_MIN_ELEMS", "999999999", 1);
    mutableDebugEnv().rmsnorm.reload();

    for (const auto &cfg : SWEEP_CONFIGS)
    {
        size_t elems = cfg.seq_len * cfg.d_model;
        int warmup = std::max(10, 1000 / cfg.seq_len);
        int iters = std::max(50, 5000 / cfg.seq_len);

        double us = bench_q8_1(cfg.seq_len, cfg.d_model, warmup, iters);
        double melem_per_sec = (elems / us); // elements per µs = Melem/s

        std::cout << std::left << std::setw(28) << cfg.description
                  << std::right << std::setw(12) << elems
                  << std::setw(12) << std::fixed << std::setprecision(2) << us
                  << std::setw(12) << std::fixed << std::setprecision(1) << melem_per_sec
                  << " Melem/s\n";
    }

    // Reset
    unsetenv("LLAMINAR_RMSNORM_PARALLEL_MIN_ELEMS");
    mutableDebugEnv().rmsnorm.reload();
}

// ============================================================================
// Sweep Test: Find Optimal Thresholds
// ============================================================================

TEST_F(RMSNormParallelSweep, ThreadCountSweep)
{
    if (rank_ != 0)
        return;

    int max_threads = omp_get_max_threads();

    std::cout << "\n";
    std::cout << "================================================================\n";
    std::cout << "Q8_1 RMSNorm Thread Count Sweep\n";
    std::cout << "System max threads: " << max_threads << "\n";
    std::cout << "================================================================\n";

    // Test thread counts: 1, 2, 4, 8, 16, ...
    std::vector<int> thread_counts = {1, 2, 4};
    for (int t = 8; t <= max_threads; t *= 2)
        thread_counts.push_back(t);
    if (thread_counts.back() != max_threads)
        thread_counts.push_back(max_threads);

    // Header
    std::cout << std::left << std::setw(28) << "Configuration";
    for (int t : thread_counts)
        std::cout << std::right << std::setw(8) << ("T=" + std::to_string(t));
    std::cout << std::setw(10) << "Best";
    std::cout << "\n";
    std::cout << std::string(28 + thread_counts.size() * 8 + 10, '-') << "\n";

    for (const auto &cfg : SWEEP_CONFIGS)
    {
        int warmup = std::max(5, 500 / cfg.seq_len);
        int iters = std::max(20, 2000 / cfg.seq_len);

        std::cout << std::left << std::setw(28) << cfg.description;

        double best_time = 1e9;
        int best_threads = 1;

        for (int t : thread_counts)
        {
            // Set max threads
            char buf[32];
            snprintf(buf, sizeof(buf), "%d", t);
            setenv("LLAMINAR_RMSNORM_MAX_THREADS", buf, 1);

            // Enable parallelism (low threshold)
            setenv("LLAMINAR_RMSNORM_PARALLEL_MIN_ELEMS", "0", 1);
            setenv("LLAMINAR_RMSNORM_PARALLEL_MIN_ROWS", "1", 1);
            setenv("LLAMINAR_RMSNORM_MIN_ELEMS_PER_THREAD", "0", 1);
            mutableDebugEnv().rmsnorm.reload();

            double us = bench_q8_1(cfg.seq_len, cfg.d_model, warmup, iters);

            std::cout << std::right << std::setw(8) << std::fixed << std::setprecision(1) << us;

            if (us < best_time)
            {
                best_time = us;
                best_threads = t;
            }
        }

        std::cout << std::setw(8) << best_threads << "T";
        std::cout << "\n";
    }

    // Reset
    unsetenv("LLAMINAR_RMSNORM_MAX_THREADS");
    unsetenv("LLAMINAR_RMSNORM_PARALLEL_MIN_ELEMS");
    unsetenv("LLAMINAR_RMSNORM_PARALLEL_MIN_ROWS");
    unsetenv("LLAMINAR_RMSNORM_MIN_ELEMS_PER_THREAD");
    mutableDebugEnv().rmsnorm.reload();
}

// ============================================================================
// Sweep Test: Elements Per Thread Threshold
// ============================================================================

TEST_F(RMSNormParallelSweep, ElemsPerThreadSweep)
{
    if (rank_ != 0)
        return;

    std::cout << "\n";
    std::cout << "================================================================\n";
    std::cout << "Q8_1 RMSNorm Elements-Per-Thread Threshold Sweep\n";
    std::cout << "================================================================\n";

    // Test different min_elems_per_thread values
    std::vector<int> thresholds = {0, 1024, 2048, 4096, 8192, 16384, 32768, 65536};

    // Header
    std::cout << std::left << std::setw(28) << "Configuration";
    for (int th : thresholds)
        std::cout << std::right << std::setw(8) << (th == 0 ? "NoLim" : std::to_string(th / 1024) + "K");
    std::cout << std::setw(10) << "Best";
    std::cout << "\n";
    std::cout << std::string(28 + thresholds.size() * 8 + 10, '-') << "\n";

    for (const auto &cfg : SWEEP_CONFIGS)
    {
        int warmup = std::max(5, 500 / cfg.seq_len);
        int iters = std::max(20, 2000 / cfg.seq_len);

        std::cout << std::left << std::setw(28) << cfg.description;

        double best_time = 1e9;
        int best_threshold = 0;

        for (int th : thresholds)
        {
            // Enable parallelism, vary elements per thread
            setenv("LLAMINAR_RMSNORM_PARALLEL_MIN_ELEMS", "0", 1);
            setenv("LLAMINAR_RMSNORM_PARALLEL_MIN_ROWS", "1", 1);
            setenv("LLAMINAR_RMSNORM_MAX_THREADS", "0", 1);

            char buf[32];
            snprintf(buf, sizeof(buf), "%d", th);
            setenv("LLAMINAR_RMSNORM_MIN_ELEMS_PER_THREAD", buf, 1);
            mutableDebugEnv().rmsnorm.reload();

            double us = bench_q8_1(cfg.seq_len, cfg.d_model, warmup, iters);

            std::cout << std::right << std::setw(8) << std::fixed << std::setprecision(1) << us;

            if (us < best_time)
            {
                best_time = us;
                best_threshold = th;
            }
        }

        std::cout << std::setw(8) << (best_threshold == 0 ? "NoLim" : std::to_string(best_threshold / 1024) + "K");
        std::cout << "\n";
    }

    // Reset
    unsetenv("LLAMINAR_RMSNORM_MAX_THREADS");
    unsetenv("LLAMINAR_RMSNORM_PARALLEL_MIN_ELEMS");
    unsetenv("LLAMINAR_RMSNORM_PARALLEL_MIN_ROWS");
    unsetenv("LLAMINAR_RMSNORM_MIN_ELEMS_PER_THREAD");
    mutableDebugEnv().rmsnorm.reload();
}

// ============================================================================
// Sweep Test: Minimum Elements Threshold
// ============================================================================

TEST_F(RMSNormParallelSweep, MinElemsSweep)
{
    if (rank_ != 0)
        return;

    std::cout << "\n";
    std::cout << "================================================================\n";
    std::cout << "Q8_1 RMSNorm Minimum Elements Threshold Sweep\n";
    std::cout << "================================================================\n";

    // Test different parallel_min_elems values
    std::vector<int> thresholds = {0, 8192, 16384, 32768, 65536, 131072, 262144, 524288};

    // Header
    std::cout << std::left << std::setw(28) << "Configuration";
    for (int th : thresholds)
        std::cout << std::right << std::setw(8) << (th == 0 ? "0" : std::to_string(th / 1024) + "K");
    std::cout << std::setw(10) << "Best";
    std::cout << "\n";
    std::cout << std::string(28 + thresholds.size() * 8 + 10, '-') << "\n";

    for (const auto &cfg : SWEEP_CONFIGS)
    {
        int warmup = std::max(5, 500 / cfg.seq_len);
        int iters = std::max(20, 2000 / cfg.seq_len);

        std::cout << std::left << std::setw(28) << cfg.description;

        double best_time = 1e9;
        int best_threshold = 0;

        for (int th : thresholds)
        {
            // Vary min_elems threshold
            char buf[32];
            snprintf(buf, sizeof(buf), "%d", th);
            setenv("LLAMINAR_RMSNORM_PARALLEL_MIN_ELEMS", buf, 1);
            setenv("LLAMINAR_RMSNORM_PARALLEL_MIN_ROWS", "1", 1);
            setenv("LLAMINAR_RMSNORM_MIN_ELEMS_PER_THREAD", "8192", 1); // Reasonable default
            setenv("LLAMINAR_RMSNORM_MAX_THREADS", "0", 1);
            mutableDebugEnv().rmsnorm.reload();

            double us = bench_q8_1(cfg.seq_len, cfg.d_model, warmup, iters);

            std::cout << std::right << std::setw(8) << std::fixed << std::setprecision(1) << us;

            if (us < best_time)
            {
                best_time = us;
                best_threshold = th;
            }
        }

        std::cout << std::setw(8) << (best_threshold == 0 ? "0" : std::to_string(best_threshold / 1024) + "K");
        std::cout << "\n";
    }

    // Reset
    unsetenv("LLAMINAR_RMSNORM_MAX_THREADS");
    unsetenv("LLAMINAR_RMSNORM_PARALLEL_MIN_ELEMS");
    unsetenv("LLAMINAR_RMSNORM_PARALLEL_MIN_ROWS");
    unsetenv("LLAMINAR_RMSNORM_MIN_ELEMS_PER_THREAD");
    mutableDebugEnv().rmsnorm.reload();
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
