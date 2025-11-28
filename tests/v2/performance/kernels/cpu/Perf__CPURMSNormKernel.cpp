/**
 * @file Perf__CPURMSNormKernel.cpp
 * @brief Performance benchmark for CPURMSNormKernel
 *
 * This test benchmarks the CPU RMSNorm kernel performance.
 * It measures:
 *   - Latency (ms)
 *   - Throughput (tokens/s)
 *   - Bandwidth (GB/s)
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
#include "kernels/cpu/ops/CPURMSNormKernelT.h"
#include "utils/Logger.h"

using namespace llaminar2;

struct BenchmarkConfig
{
    int seq_len;
    int d_model;
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
    double bandwidth_gbps;
};

class CPURMSNormKernel_Perf : public ::testing::Test
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
            std::cout << "  D Model:    " << config.d_model << std::endl;
            std::cout << "----------------------------------------------------------------" << std::endl;
        }

        // Create kernel
        CPURMSNormKernelT<FP32Tensor> kernel;

        // Allocate tensors
        size_t size = config.seq_len * config.d_model;

        std::vector<float> input(size, 0.1f);
        std::vector<float> gamma(config.d_model, 1.0f);
        std::vector<float> output(size);

        // Warmup
        for (int i = 0; i < config.warmup_iters; ++i)
        {
            kernel.apply(
                input.data(), gamma.data(), output.data(),
                config.seq_len, config.d_model,
                1e-6f // eps
            );
        }

        // Benchmark
        std::vector<double> times_ms;
        times_ms.reserve(config.bench_iters);

        for (int i = 0; i < config.bench_iters; ++i)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            auto start = std::chrono::high_resolution_clock::now();

            kernel.apply(
                input.data(), gamma.data(), output.data(),
                config.seq_len, config.d_model,
                1e-6f // eps
            );

            auto end = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1e6;
            times_ms.push_back(ms);
        }

        // Calculate stats
        double sum = std::accumulate(times_ms.begin(), times_ms.end(), 0.0);
        double mean = sum / times_ms.size();
        double sq_sum = std::inner_product(times_ms.begin(), times_ms.end(), times_ms.begin(), 0.0);
        double stddev = std::sqrt(sq_sum / times_ms.size() - mean * mean);
        double min_val = *std::min_element(times_ms.begin(), times_ms.end());
        double max_val = *std::max_element(times_ms.begin(), times_ms.end());

        // Calculate bandwidth (Read Input + Read Gamma + Write Output)
        // Gamma is read once per row, but effectively it's small.
        // Main traffic is Input (Read) + Output (Write).
        // Bytes = (seq_len * d_model * 4) * 2
        double total_bytes = (double)config.seq_len * config.d_model * sizeof(float) * 2.0;
        double bandwidth_gbps = (total_bytes / (mean * 1e-3)) / 1e9;

        if (rank_ == 0)
        {
            std::cout << std::fixed << std::setprecision(3);
            std::cout << "  Mean:      " << mean << " ms" << std::endl;
            std::cout << "  StdDev:    " << stddev << " ms" << std::endl;
            std::cout << "  Min:       " << min_val << " ms" << std::endl;
            std::cout << "  Max:       " << max_val << " ms" << std::endl;
            std::cout << "  Bandwidth: " << bandwidth_gbps << " GB/s" << std::endl;
        }

        return {mean, stddev, min_val, max_val, bandwidth_gbps};
    }
};

TEST_F(CPURMSNormKernel_Perf, SingleToken_Latency)
{
    BenchmarkConfig config;
    config.seq_len = 1;
    config.d_model = 896; // Qwen 2.5 0.5B d_model
    config.warmup_iters = 100;
    config.bench_iters = 1000;
    config.description = "Qwen 2.5 0.5B Single Token (1x896)";

    run_benchmark(config);
}

TEST_F(CPURMSNormKernel_Perf, Prefill_128)
{
    BenchmarkConfig config;
    config.seq_len = 128;
    config.d_model = 896;
    config.warmup_iters = 10;
    config.bench_iters = 100;
    config.description = "Qwen 2.5 0.5B Prefill (128x896)";

    run_benchmark(config);
}

TEST_F(CPURMSNormKernel_Perf, Prefill_1024)
{
    BenchmarkConfig config;
    config.seq_len = 1024;
    config.d_model = 896;
    config.warmup_iters = 5;
    config.bench_iters = 50;
    config.description = "Qwen 2.5 0.5B Prefill (1024x896)";

    run_benchmark(config);
}

TEST_F(CPURMSNormKernel_Perf, Large_SingleToken_Latency)
{
    BenchmarkConfig config;
    config.seq_len = 1;
    config.d_model = 3584; // Qwen 2.5 7B d_model
    config.warmup_iters = 100;
    config.bench_iters = 1000;
    config.description = "Qwen 2.5 7B Single Token (1x3584)";

    run_benchmark(config);
}

TEST_F(CPURMSNormKernel_Perf, Large_Prefill_128)
{
    BenchmarkConfig config;
    config.seq_len = 128;
    config.d_model = 3584;
    config.warmup_iters = 10;
    config.bench_iters = 100;
    config.description = "Qwen 2.5 7B Prefill (128x3584)";

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
