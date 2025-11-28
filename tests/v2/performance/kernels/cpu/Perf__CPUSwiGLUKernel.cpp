/**
 * @file Perf__CPUSwiGLUKernel.cpp
 * @brief Performance benchmark for CPUSwiGLUKernel
 *
 * This test benchmarks the CPU SwiGLU kernel performance.
 * It measures:
 *   - Latency (ms)
 *   - Throughput (elements/s)
 *
 * @author GitHub Copilot
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <omp.h>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>

// V2 includes
#include "tensors/Tensors.h"
#include "kernels/cpu/ops/CPUSwiGLUKernelT.h"
#include "utils/Logger.h"

using namespace llaminar2;

struct BenchmarkConfig
{
    int seq_len;
    int d_ff;
    int warmup_iters;
    int bench_iters;
    std::string description;
};

struct BenchmarkStats
{
    double mean_ms;
    double stddev_ms;
    double min_ms;
    double max_ms;
};

class CPUSwiGLUKernel_Perf : public ::testing::Test
{
protected:
    int rank_ = 0;
    int world_size_ = 1;

    void SetUp() override
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
    }

    BenchmarkStats run_benchmark(const BenchmarkConfig &config)
    {
        if (rank_ == 0)
        {
            std::cout << "\n----------------------------------------------------------------" << std::endl;
            std::cout << "Running Benchmark: " << config.description << std::endl;
            std::cout << "  Seq Len:    " << config.seq_len << std::endl;
            std::cout << "  d_ff:       " << config.d_ff << std::endl;
            std::cout << "----------------------------------------------------------------" << std::endl;
        }

        // Create kernel
        CPUSwiGLUKernel kernel;

        // Allocate tensors
        size_t size = config.seq_len * config.d_ff;
        std::vector<float> gate(size, 0.5f);
        std::vector<float> up(size, 0.5f);
        std::vector<float> output(size);

        // Warmup
        for (int i = 0; i < config.warmup_iters; ++i)
        {
            kernel.apply(
                gate.data(), up.data(), output.data(),
                config.seq_len, config.d_ff,
                false,   // use_bf16
                nullptr, // mpi_ctx
                -1       // device_idx
            );
        }

        // Benchmark
        std::vector<double> latencies;
        latencies.reserve(config.bench_iters);

        for (int i = 0; i < config.bench_iters; ++i)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            auto start = std::chrono::high_resolution_clock::now();

            kernel.apply(
                gate.data(), up.data(), output.data(),
                config.seq_len, config.d_ff,
                false,   // use_bf16
                nullptr, // mpi_ctx
                -1       // device_idx
            );

            auto end = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1e6;
            latencies.push_back(ms);
        }

        // Calculate stats
        double sum = std::accumulate(latencies.begin(), latencies.end(), 0.0);
        double mean = sum / latencies.size();
        double sq_sum = std::inner_product(latencies.begin(), latencies.end(), latencies.begin(), 0.0);
        double stddev = std::sqrt(sq_sum / latencies.size() - mean * mean);
        double min_val = *std::min_element(latencies.begin(), latencies.end());
        double max_val = *std::max_element(latencies.begin(), latencies.end());

        if (rank_ == 0)
        {
            std::cout << "  Mean:   " << std::fixed << std::setprecision(3) << mean << " ms" << std::endl;
            std::cout << "  StdDev: " << stddev << " ms" << std::endl;
            std::cout << "  Min:    " << min_val << " ms" << std::endl;
            std::cout << "  Max:    " << max_val << " ms" << std::endl;
        }

        return {mean, stddev, min_val, max_val};
    }
};

TEST_F(CPUSwiGLUKernel_Perf, SingleToken_Latency)
{
    BenchmarkConfig config;
    config.seq_len = 1;
    config.d_ff = 4864; // Qwen 2.5 0.5B
    config.warmup_iters = 100;
    config.bench_iters = 1000;
    config.description = "Qwen 2.5 0.5B Single Token (1->1)";

    run_benchmark(config);
}

TEST_F(CPUSwiGLUKernel_Perf, Prefill_128)
{
    BenchmarkConfig config;
    config.seq_len = 128;
    config.d_ff = 4864; // Qwen 2.5 0.5B
    config.warmup_iters = 10;
    config.bench_iters = 100;
    config.description = "Qwen 2.5 0.5B Prefill (128)";

    run_benchmark(config);
}

TEST_F(CPUSwiGLUKernel_Perf, Prefill_1024)
{
    BenchmarkConfig config;
    config.seq_len = 1024;
    config.d_ff = 4864; // Qwen 2.5 0.5B
    config.warmup_iters = 5;
    config.bench_iters = 50;
    config.description = "Qwen 2.5 0.5B Prefill (1024)";

    run_benchmark(config);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    MPI_Init(&argc, &argv);
    int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
