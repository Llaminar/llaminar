/**
 * @file Perf__CpuAttentionKernelT.cpp
 * @brief Performance benchmark for CpuAttentionKernelT (FP32)
 *
 * This test benchmarks the CPU attention kernel performance.
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
#include "kernels/cpu/attention/CpuAttentionKernelT.h"
#include "utils/Logger.h"

using namespace llaminar2;

struct BenchmarkConfig
{
    int seq_len;
    int kv_len = -1; // -1 means equal to seq_len
    int n_heads;
    int n_kv_heads;
    int head_dim;
    int batch_size;
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

class CpuAttentionKernelT_Perf : public ::testing::Test
{
protected:
    int rank_ = 0;
    int world_size_ = 1;

    void SetUp() override
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
    }

    BenchmarkStats run_benchmark(const BenchmarkConfig& config)
    {
        int kv_len = (config.kv_len == -1) ? config.seq_len : config.kv_len;

        if (rank_ == 0)
        {
            std::cout << "\n----------------------------------------------------------------" << std::endl;
            std::cout << "Running Benchmark: " << config.description << std::endl;
            std::cout << "  Batch Size: " << config.batch_size << std::endl;
            std::cout << "  Seq Len:    " << config.seq_len << std::endl;
            std::cout << "  KV Len:     " << kv_len << std::endl;
            std::cout << "  Heads:      " << config.n_heads << " (KV: " << config.n_kv_heads << ")" << std::endl;
            std::cout << "  Head Dim:   " << config.head_dim << std::endl;
            std::cout << "----------------------------------------------------------------" << std::endl;
        }

        // Create kernel
        CpuAttentionKernelT<FP32Tensor> kernel;

        // Allocate tensors
        size_t q_size = config.batch_size * config.seq_len * config.n_heads * config.head_dim;
        size_t k_size = config.batch_size * kv_len * config.n_kv_heads * config.head_dim;
        size_t v_size = config.batch_size * kv_len * config.n_kv_heads * config.head_dim;
        size_t out_size = config.batch_size * config.seq_len * config.n_heads * config.head_dim;

        std::vector<float> Q(q_size, 0.1f);
        std::vector<float> K(k_size, 0.1f);
        std::vector<float> V(v_size, 0.1f);
        std::vector<float> output(out_size);

        // Warmup
        for (int i = 0; i < config.warmup_iters; ++i)
        {
            if (config.kv_len != -1)
            {
                kernel.compute_decode(
                    Q.data(), K.data(), V.data(), output.data(),
                    config.seq_len, kv_len, config.n_heads, config.n_kv_heads, config.head_dim,
                    false, -1, nullptr, nullptr, nullptr, nullptr, false, nullptr, -1
                );
            }
            else
            {
                kernel.compute_batch(
                    Q.data(), K.data(), V.data(), output.data(),
                    config.batch_size, config.seq_len, config.n_heads, config.n_kv_heads, config.head_dim,
                    false, -1, nullptr, nullptr, nullptr, nullptr, false, nullptr, -1
                );
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);

        // Benchmark
        std::vector<double> times_ms;
        times_ms.reserve(config.bench_iters);

        for (int i = 0; i < config.bench_iters; ++i)
        {
            auto start = std::chrono::high_resolution_clock::now();

            if (config.kv_len != -1)
            {
                kernel.compute_decode(
                    Q.data(), K.data(), V.data(), output.data(),
                    config.seq_len, kv_len, config.n_heads, config.n_kv_heads, config.head_dim,
                    false, -1, nullptr, nullptr, nullptr, nullptr, false, nullptr, -1
                );
            }
            else
            {
                kernel.compute_batch(
                    Q.data(), K.data(), V.data(), output.data(),
                    config.batch_size, config.seq_len, config.n_heads, config.n_kv_heads, config.head_dim,
                    false, -1, nullptr, nullptr, nullptr, nullptr, false, nullptr, -1
                );
            }

            auto end = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(end - start).count();
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
            std::cout << "  Mean:   " << std::fixed << std::setprecision(3) << mean << " ms" << std::endl;
            std::cout << "  StdDev: " << stddev << " ms" << std::endl;
            std::cout << "  Min:    " << min_val << " ms" << std::endl;
            std::cout << "  Max:    " << max_val << " ms" << std::endl;
        }

        return {mean, stddev, min_val, max_val};
    }
};

TEST_F(CpuAttentionKernelT_Perf, SingleToken_Latency_Prefill)
{
    BenchmarkConfig config;
    config.seq_len = 1;
    config.kv_len = -1; // Implies kv_len = seq_len = 1
    config.n_heads = 14;
    config.n_kv_heads = 2;
    config.head_dim = 64;
    config.batch_size = 1;
    config.warmup_iters = 100;
    config.bench_iters = 1000;
    config.description = "Qwen 2.5 0.5B Single Token Prefill (1->1)";

    run_benchmark(config);
}

TEST_F(CpuAttentionKernelT_Perf, SingleToken_Decode_128)
{
    BenchmarkConfig config;
    config.seq_len = 1;
    config.kv_len = 128;
    config.n_heads = 14;
    config.n_kv_heads = 2;
    config.head_dim = 64;
    config.batch_size = 1;
    config.warmup_iters = 100;
    config.bench_iters = 1000;
    config.description = "Qwen 2.5 0.5B Decode (1->128)";

    run_benchmark(config);
}

TEST_F(CpuAttentionKernelT_Perf, SingleToken_Decode_1024)
{
    BenchmarkConfig config;
    config.seq_len = 1;
    config.kv_len = 1024;
    config.n_heads = 14;
    config.n_kv_heads = 2;
    config.head_dim = 64;
    config.batch_size = 1;
    config.warmup_iters = 100;
    config.bench_iters = 1000;
    config.description = "Qwen 2.5 0.5B Decode (1->1024)";

    run_benchmark(config);
}

TEST_F(CpuAttentionKernelT_Perf, SingleToken_Decode_Large_1024)
{
    BenchmarkConfig config;
    config.seq_len = 1;
    config.kv_len = 1024;
    config.n_heads = 28;
    config.n_kv_heads = 4;
    config.head_dim = 128;
    config.batch_size = 1;
    config.warmup_iters = 100;
    config.bench_iters = 1000;
    config.description = "Qwen 2.5 7B Decode (1->1024)";

    run_benchmark(config);
}


int main(int argc, char **argv)
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
