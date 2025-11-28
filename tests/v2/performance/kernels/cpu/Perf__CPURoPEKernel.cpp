/**
 * @file Perf__CPURoPEKernel.cpp
 * @brief Performance benchmark for CPURoPEKernel
 *
 * This test benchmarks the CPU RoPE kernel performance.
 * It measures:
 *   - Latency (ms)
 *   - Throughput (tokens/s)
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
#include "tensors/BlockStructures.h"
#include "kernels/cpu/ops/CPURoPEKernelT.h"
#include "utils/Logger.h"

using namespace llaminar2;

struct BenchmarkConfig
{
    int seq_len;
    int n_heads;
    int n_kv_heads;
    int head_dim;
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

class CPURoPEKernel_Perf : public ::testing::Test
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
            std::cout << "  Heads:      " << config.n_heads << " (KV: " << config.n_kv_heads << ")" << std::endl;
            std::cout << "  Head Dim:   " << config.head_dim << std::endl;
            std::cout << "----------------------------------------------------------------" << std::endl;
        }

        // Create kernel
        CPURoPEKernel kernel;

        // Allocate tensors
        size_t q_size = config.seq_len * config.n_heads * config.head_dim;
        size_t k_size = config.seq_len * config.n_kv_heads * config.head_dim;

        std::vector<float> Q(q_size, 0.1f);
        std::vector<float> K(k_size, 0.1f);
        std::vector<int> position_ids(config.seq_len);
        for (int i = 0; i < config.seq_len; ++i)
            position_ids[i] = i;

        // Warmup
        for (int i = 0; i < config.warmup_iters; ++i)
        {
            kernel.apply(
                Q.data(), K.data(),
                position_ids.data(),
                config.seq_len, config.n_heads, config.n_kv_heads, config.head_dim,
                10000.0f, // rope_theta
                false,    // use_bf16
                nullptr,  // mpi_ctx
                -1        // device_idx
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
                Q.data(), K.data(),
                position_ids.data(),
                config.seq_len, config.n_heads, config.n_kv_heads, config.head_dim,
                10000.0f, // rope_theta
                false,    // use_bf16
                nullptr,  // mpi_ctx
                -1        // device_idx
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

        if (rank_ == 0)
        {
            std::cout << std::fixed << std::setprecision(3);
            std::cout << "  Mean:   " << mean << " ms" << std::endl;
            std::cout << "  StdDev: " << stddev << " ms" << std::endl;
            std::cout << "  Min:    " << min_val << " ms" << std::endl;
            std::cout << "  Max:    " << max_val << " ms" << std::endl;
        }

        return {mean, stddev, min_val, max_val};
    }
};

TEST_F(CPURoPEKernel_Perf, SingleToken_Latency)
{
    BenchmarkConfig config;
    config.seq_len = 1;
    config.n_heads = 14;
    config.n_kv_heads = 2;
    config.head_dim = 64;
    config.warmup_iters = 100;
    config.bench_iters = 1000;
    config.description = "Qwen 2.5 0.5B Single Token (1->1)";

    run_benchmark(config);
}

TEST_F(CPURoPEKernel_Perf, Prefill_128)
{
    BenchmarkConfig config;
    config.seq_len = 128;
    config.n_heads = 14;
    config.n_kv_heads = 2;
    config.head_dim = 64;
    config.warmup_iters = 10;
    config.bench_iters = 100;
    config.description = "Qwen 2.5 0.5B Prefill (128)";

    run_benchmark(config);
}

TEST_F(CPURoPEKernel_Perf, Prefill_1024)
{
    BenchmarkConfig config;
    config.seq_len = 1024;
    config.n_heads = 14;
    config.n_kv_heads = 2;
    config.head_dim = 64;
    config.warmup_iters = 5;
    config.bench_iters = 50;
    config.description = "Qwen 2.5 0.5B Prefill (1024)";

    run_benchmark(config);
}

TEST_F(CPURoPEKernel_Perf, Large_SingleToken_Latency)
{
    BenchmarkConfig config;
    config.seq_len = 1;
    config.n_heads = 28;
    config.n_kv_heads = 4;
    config.head_dim = 128;
    config.warmup_iters = 100;
    config.bench_iters = 1000;
    config.description = "Qwen 2.5 7B Single Token (1->1)";

    run_benchmark(config);
}

TEST_F(CPURoPEKernel_Perf, Q8_1_SingleToken_Latency)
{
    if (rank_ == 0)
    {
        std::cout << "\n----------------------------------------------------------------" << std::endl;
        std::cout << "Running Benchmark: Q8_1 Single Token (1->1)" << std::endl;
        std::cout << "----------------------------------------------------------------" << std::endl;
    }

    // Create kernel for Q8_1
    CPURoPEKernelT<Q8_1Tensor> kernel;

    int seq_len = 1;
    int n_heads = 14;
    int n_kv_heads = 2;
    int head_dim = 64;

    // Allocate buffers
    size_t q_blocks = seq_len * n_heads * head_dim / Q8_1Block::BLOCK_SIZE;
    size_t k_blocks = seq_len * n_kv_heads * head_dim / Q8_1Block::BLOCK_SIZE;

    std::vector<Q8_1Block> Q(q_blocks);
    std::vector<Q8_1Block> K(k_blocks);
    std::vector<int> position_ids(seq_len);
    for (int i = 0; i < seq_len; ++i)
        position_ids[i] = i;

    // Initialize with dummy data
    for (auto &b : Q)
    {
        b.d = 1;
        b.sum_qs = 0;
        for (int i = 0; i < 32; ++i)
            b.qs[i] = 1;
    }
    for (auto &b : K)
    {
        b.d = 1;
        b.sum_qs = 0;
        for (int i = 0; i < 32; ++i)
            b.qs[i] = 1;
    }

    // Warmup
    for (int i = 0; i < 10; ++i)
    {
        kernel.apply_q8_1(
            Q.data(), K.data(),
            position_ids.data(),
            seq_len, n_heads, n_kv_heads, head_dim,
            10000.0f, // rope_theta
            -1        // device_idx
        );
    }

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; ++i)
    {
        kernel.apply_q8_1(
            Q.data(), K.data(),
            position_ids.data(),
            seq_len, n_heads, n_kv_heads, head_dim,
            10000.0f, // rope_theta
            -1        // device_idx
        );
    }
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1000000.0 / 1000.0;

    if (rank_ == 0)
    {
        std::cout << "  Mean:   " << std::fixed << std::setprecision(3) << ms << " ms" << std::endl;
    }
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    MPI_Init(&argc, &argv);
    int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
