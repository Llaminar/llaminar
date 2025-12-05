/**
 * @file Perf__RMSNormThreadSweep.cpp
 * @brief Empirical thread count sweep for Q8_1 RMSNorm optimization
 * @author David Sanftenberg
 *
 * This test sweeps thread counts across various workload sizes to determine
 * optimal parallelization thresholds for Q8_1 RMSNorm.
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <omp.h>
#include <chrono>
#include <vector>
#include <random>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <numeric>

// V2 includes
#include "tensors/Tensors.h"
#include "tensors/BlockStructures.h"
#include "kernels/cpu/ops/CPURMSNormTypedKernel.h"
#include "kernels/cpu/primitives/RMSNormPrimitives.h"
#include "tensors/SIMDHelpers.h"

using namespace llaminar2;

class RMSNormThreadSweep : public ::testing::Test
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

    struct SweepResult
    {
        int threads;
        double mean_ms;
        double stddev_ms;
        double speedup_vs_seq; // vs 1 thread
        double efficiency;     // speedup / threads
    };

    std::vector<SweepResult> sweep_thread_counts(
        int seq_len, int d_model, int warmup, int iters, int max_threads)
    {
        size_t size = static_cast<size_t>(seq_len) * d_model;
        size_t blocks_per_row = (d_model + 31) / 32;
        size_t total_blocks = static_cast<size_t>(seq_len) * blocks_per_row;

        // Prepare data
        std::vector<float> fp32_input(size);
        std::vector<float> gamma(d_model, 1.0f);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto &v : fp32_input)
            v = dist(rng_);

        std::vector<Q8_1Block> input(total_blocks);
        std::vector<Q8_1Block> output(total_blocks);
        simd::quantize_fp32_to_q8_1_blocks(fp32_input.data(), input.data(), size);

        std::vector<SweepResult> results;
        double baseline_ms = 0.0;

        // Sweep from 1 to max_threads
        for (int nthreads = 1; nthreads <= max_threads; ++nthreads)
        {
            omp_set_num_threads(nthreads);

            // Warmup
            for (int i = 0; i < warmup; ++i)
            {
                // Direct call to primitive with forced parallelization
                primitives::rmsnorm_q8_1_pure_integer(
                    input.data(), gamma.data(), output.data(),
                    seq_len, blocks_per_row, 1e-6f,
                    {.allow_parallel = (nthreads > 1)});
            }

            // Benchmark
            std::vector<double> times_ms;
            times_ms.reserve(iters);

            for (int i = 0; i < iters; ++i)
            {
                MPI_Barrier(MPI_COMM_WORLD);
                auto start = std::chrono::high_resolution_clock::now();

                primitives::rmsnorm_q8_1_pure_integer(
                    input.data(), gamma.data(), output.data(),
                    seq_len, blocks_per_row, 1e-6f,
                    {.allow_parallel = (nthreads > 1)});

                auto end = std::chrono::high_resolution_clock::now();
                double ms = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1e6;
                times_ms.push_back(ms);
            }

            // Calculate stats
            double sum = std::accumulate(times_ms.begin(), times_ms.end(), 0.0);
            double mean = sum / times_ms.size();
            double sq_sum = std::inner_product(times_ms.begin(), times_ms.end(), times_ms.begin(), 0.0);
            double stddev = std::sqrt(sq_sum / times_ms.size() - mean * mean);

            if (nthreads == 1)
                baseline_ms = mean;

            double speedup = baseline_ms / mean;
            double efficiency = speedup / nthreads;

            results.push_back({nthreads, mean, stddev, speedup, efficiency});
        }

        // Restore original thread count
        omp_set_num_threads(max_threads);

        return results;
    }

    void print_sweep_results(const std::string &label, int seq_len, int d_model,
                             const std::vector<SweepResult> &results)
    {
        if (rank_ != 0)
            return;

        size_t total_elems = static_cast<size_t>(seq_len) * d_model;
        size_t total_bytes = (total_elems * 36) / 32; // Q8_1: 36 bytes per 32 elements

        std::cout << "\n"
                  << std::string(80, '=') << "\n";
        std::cout << label << "\n";
        std::cout << "  SeqLen=" << seq_len << ", D_Model=" << d_model
                  << ", TotalElems=" << total_elems << ", Bytes=" << total_bytes << "\n";
        std::cout << std::string(80, '-') << "\n";
        std::cout << std::setw(8) << "Threads"
                  << std::setw(12) << "Mean(ms)"
                  << std::setw(12) << "Stddev"
                  << std::setw(10) << "Speedup"
                  << std::setw(12) << "Efficiency"
                  << std::setw(15) << "Status"
                  << "\n";
        std::cout << std::string(80, '-') << "\n";

        double best_time = results[0].mean_ms;
        int best_threads = 1;
        for (const auto &r : results)
        {
            if (r.mean_ms < best_time)
            {
                best_time = r.mean_ms;
                best_threads = r.threads;
            }
        }

        for (const auto &r : results)
        {
            std::string status;
            if (r.threads == best_threads)
                status = "<-- BEST";
            else if (r.efficiency < 0.5 && r.threads > 1)
                status = "inefficient";
            else if (r.speedup_vs_seq < 1.0)
                status = "SLOWER";

            std::cout << std::setw(8) << r.threads
                      << std::setw(12) << std::fixed << std::setprecision(4) << r.mean_ms
                      << std::setw(12) << std::fixed << std::setprecision(4) << r.stddev_ms
                      << std::setw(10) << std::fixed << std::setprecision(2) << r.speedup_vs_seq
                      << std::setw(12) << std::fixed << std::setprecision(2) << r.efficiency
                      << std::setw(15) << status
                      << "\n";
        }

        // Summary
        std::cout << std::string(80, '-') << "\n";
        std::cout << "RECOMMENDATION: Use " << best_threads << " threads for this workload\n";
        std::cout << "  Sequential time: " << std::fixed << std::setprecision(4)
                  << results[0].mean_ms << " ms\n";
        std::cout << "  Best time: " << std::fixed << std::setprecision(4)
                  << best_time << " ms (speedup: "
                  << std::fixed << std::setprecision(2) << results[0].mean_ms / best_time << "x)\n";
    }
};

// ============================================================================
// Qwen 7B (d_model=3584) Sweep Tests
// ============================================================================

TEST_F(RMSNormThreadSweep, Qwen7B_SingleToken)
{
    auto results = sweep_thread_counts(1, 3584, 50, 500, 28);
    print_sweep_results("Q8_1 RMSNorm Thread Sweep: Qwen 7B Single Token", 1, 3584, results);
}

TEST_F(RMSNormThreadSweep, Qwen7B_Prefill_32)
{
    auto results = sweep_thread_counts(32, 3584, 30, 300, 28);
    print_sweep_results("Q8_1 RMSNorm Thread Sweep: Qwen 7B Prefill 32", 32, 3584, results);
}

TEST_F(RMSNormThreadSweep, Qwen7B_Prefill_64)
{
    auto results = sweep_thread_counts(64, 3584, 30, 300, 28);
    print_sweep_results("Q8_1 RMSNorm Thread Sweep: Qwen 7B Prefill 64", 64, 3584, results);
}

TEST_F(RMSNormThreadSweep, Qwen7B_Prefill_128)
{
    auto results = sweep_thread_counts(128, 3584, 20, 200, 28);
    print_sweep_results("Q8_1 RMSNorm Thread Sweep: Qwen 7B Prefill 128", 128, 3584, results);
}

TEST_F(RMSNormThreadSweep, Qwen7B_Prefill_256)
{
    auto results = sweep_thread_counts(256, 3584, 20, 200, 28);
    print_sweep_results("Q8_1 RMSNorm Thread Sweep: Qwen 7B Prefill 256", 256, 3584, results);
}

TEST_F(RMSNormThreadSweep, Qwen7B_Prefill_512)
{
    auto results = sweep_thread_counts(512, 3584, 10, 100, 28);
    print_sweep_results("Q8_1 RMSNorm Thread Sweep: Qwen 7B Prefill 512", 512, 3584, results);
}

TEST_F(RMSNormThreadSweep, Qwen7B_Prefill_1024)
{
    auto results = sweep_thread_counts(1024, 3584, 10, 100, 28);
    print_sweep_results("Q8_1 RMSNorm Thread Sweep: Qwen 7B Prefill 1024", 1024, 3584, results);
}

TEST_F(RMSNormThreadSweep, Qwen7B_Prefill_2048)
{
    auto results = sweep_thread_counts(2048, 3584, 5, 50, 28);
    print_sweep_results("Q8_1 RMSNorm Thread Sweep: Qwen 7B Prefill 2048", 2048, 3584, results);
}

// ============================================================================
// Qwen 32B (d_model=5120) Sweep Tests
// ============================================================================

TEST_F(RMSNormThreadSweep, Qwen32B_SingleToken)
{
    auto results = sweep_thread_counts(1, 5120, 50, 500, 28);
    print_sweep_results("Q8_1 RMSNorm Thread Sweep: Qwen 32B Single Token", 1, 5120, results);
}

TEST_F(RMSNormThreadSweep, Qwen32B_Prefill_32)
{
    auto results = sweep_thread_counts(32, 5120, 30, 300, 28);
    print_sweep_results("Q8_1 RMSNorm Thread Sweep: Qwen 32B Prefill 32", 32, 5120, results);
}

TEST_F(RMSNormThreadSweep, Qwen32B_Prefill_64)
{
    auto results = sweep_thread_counts(64, 5120, 30, 300, 28);
    print_sweep_results("Q8_1 RMSNorm Thread Sweep: Qwen 32B Prefill 64", 64, 5120, results);
}

TEST_F(RMSNormThreadSweep, Qwen32B_Prefill_128)
{
    auto results = sweep_thread_counts(128, 5120, 20, 200, 28);
    print_sweep_results("Q8_1 RMSNorm Thread Sweep: Qwen 32B Prefill 128", 128, 5120, results);
}

TEST_F(RMSNormThreadSweep, Qwen32B_Prefill_256)
{
    auto results = sweep_thread_counts(256, 5120, 20, 200, 28);
    print_sweep_results("Q8_1 RMSNorm Thread Sweep: Qwen 32B Prefill 256", 256, 5120, results);
}

TEST_F(RMSNormThreadSweep, Qwen32B_Prefill_512)
{
    auto results = sweep_thread_counts(512, 5120, 10, 100, 28);
    print_sweep_results("Q8_1 RMSNorm Thread Sweep: Qwen 32B Prefill 512", 512, 5120, results);
}

TEST_F(RMSNormThreadSweep, Qwen32B_Prefill_1024)
{
    auto results = sweep_thread_counts(1024, 5120, 10, 100, 28);
    print_sweep_results("Q8_1 RMSNorm Thread Sweep: Qwen 32B Prefill 1024", 1024, 5120, results);
}

TEST_F(RMSNormThreadSweep, Qwen32B_Prefill_2048)
{
    auto results = sweep_thread_counts(2048, 5120, 5, 50, 28);
    print_sweep_results("Q8_1 RMSNorm Thread Sweep: Qwen 32B Prefill 2048", 2048, 5120, results);
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    MPI_Init(&argc, &argv);
    int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
