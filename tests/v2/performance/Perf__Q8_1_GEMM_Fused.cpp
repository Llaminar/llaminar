/**
 * @file Perf__Q8_1_GEMM_Fused.cpp
 * @brief Performance benchmark for Q8_1 GEMM Fused with Online Softmax
 *
 * This test benchmarks the performance impact of fusing Softmax into the Q8_1 GEMM kernel.
 * It compares:
 *   1. Baseline: GEMM + Standalone Softmax
 *   2. Fused: GEMM with Online Softmax
 *
 * @author David Sanftenberg
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
#include <random>
#include <numeric>
#include <algorithm>
#include <cstring>

// V2 includes
#include "tensors/Tensors.h"
#include "kernels/cpu/gemm_v4/Q8_1GemmKernel.h"

using namespace llaminar2;
using namespace llaminar2::gemm_v4;

// Standalone Softmax implementation for baseline comparison
// Computes Softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
// We only need the max and sum stats for this benchmark to match the kernel output
void standalone_softmax_stats(const float *input, int M, int N, float *local_max, float *local_sum)
{
    int blocks_per_row = N / 64;

#pragma omp parallel for collapse(2)
    for (int m = 0; m < M; ++m)
    {
        for (int blk = 0; blk < blocks_per_row; ++blk)
        {
            // 1. Compute Max
            float max_val = -1e9f;
            for (int i = 0; i < 64; ++i)
            {
                float val = input[m * N + blk * 64 + i];
                if (val > max_val)
                    max_val = val;
            }
            local_max[m * blocks_per_row + blk] = max_val;

            // 2. Compute Sum of Exp
            float sum_val = 0.0f;
            for (int i = 0; i < 64; ++i)
            {
                float val = input[m * N + blk * 64 + i];
                sum_val += std::exp(val - max_val);
            }
            local_sum[m * blocks_per_row + blk] = sum_val;
        }
    }
}

struct BenchmarkConfig
{
    int M;                   ///< Sequence length (m dimension)
    int K;                   ///< Input feature dimension (k dimension)
    int N;                   ///< Output feature dimension (n dimension)
    int warmup_iters;        ///< Number of warmup iterations
    int bench_iters;         ///< Number of timed benchmark iterations per trial
    std::string description; ///< Human-readable description
};

class Q8_1_GEMM_Fused_Perf : public ::testing::Test
{
protected:
    int rank_ = 0;
    int world_size_ = 1;

    void SetUp() override
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
    }

    void run_benchmark(const BenchmarkConfig &config)
    {
        if (rank_ != 0)
            return;

        int M = config.M;
        int K = config.K;
        int N = config.N;

        std::cout << "\n----------------------------------------------------------------" << std::endl;
        std::cout << "Benchmarking: " << config.description << std::endl;
        std::cout << "Dimensions: M=" << M << ", N=" << N << ", K=" << K << std::endl;
        std::cout << "----------------------------------------------------------------" << std::endl;

        // 1. Setup Data
        std::vector<float> weights_fp32(N * K);
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
        for (auto &x : weights_fp32)
            x = dist(gen);

        auto weights_tensor = Q8_1Tensor::quantize_from_fp32(weights_fp32.data(), {static_cast<size_t>(N), static_cast<size_t>(K)});

        MPIContext ctx(0, 1, MPI_COMM_WORLD);
        auto generic_kernel = weights_tensor->createGemm();
        auto kernel = dynamic_cast<Q8_1GemmKernel *>(generic_kernel.get());
        ASSERT_NE(kernel, nullptr);

        std::vector<float> A(M * K);
        for (auto &x : A)
            x = dist(gen);

        std::vector<float> C(M * N, 0.0f);
        int blocks_per_row = N / 64;
        std::vector<float> local_max(M * blocks_per_row);
        std::vector<float> local_sum(M * blocks_per_row);

        // Warmup
        for (int i = 0; i < config.warmup_iters; ++i)
        {
            kernel->multiply_fused(A.data(), C.data(), M, N, K, nullptr, nullptr, true, local_max.data(), local_sum.data(), false, 1.0f, 0.0f, &ctx, -1);
        }

        // --- Benchmark Baseline (Unfused) ---
        auto start_unfused = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < config.bench_iters; ++i)
        {
            // 1. GEMM
            kernel->multiply_fused(A.data(), C.data(), M, N, K, nullptr, nullptr, false, nullptr, nullptr, false, 1.0f, 0.0f, &ctx, -1);
            // 2. Softmax
            standalone_softmax_stats(C.data(), M, N, local_max.data(), local_sum.data());
        }
        auto end_unfused = std::chrono::high_resolution_clock::now();
        double time_unfused = std::chrono::duration<double, std::milli>(end_unfused - start_unfused).count() / config.bench_iters;

        // --- Benchmark Fused ---
        auto start_fused = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < config.bench_iters; ++i)
        {
            kernel->multiply_fused(A.data(), C.data(), M, N, K, nullptr, nullptr, true, local_max.data(), local_sum.data(), false, 1.0f, 0.0f, &ctx, -1);
        }
        auto end_fused = std::chrono::high_resolution_clock::now();
        double time_fused = std::chrono::duration<double, std::milli>(end_fused - start_fused).count() / config.bench_iters;

        // Report
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "Baseline (Unfused): " << time_unfused << " ms" << std::endl;
        std::cout << "Fused             : " << time_fused << " ms" << std::endl;
        std::cout << "Speedup           : " << (time_unfused / time_fused) << "x" << std::endl;

        // Verify correctness (sanity check)
        // Run one unfused and one fused and compare local_max/sum
        std::vector<float> max_unfused(local_max.size());
        std::vector<float> sum_unfused(local_sum.size());

        kernel->multiply_fused(A.data(), C.data(), M, N, K, nullptr, nullptr, false, nullptr, nullptr, false, 1.0f, 0.0f, &ctx, -1);
        standalone_softmax_stats(C.data(), M, N, max_unfused.data(), sum_unfused.data());

        std::vector<float> max_fused(local_max.size());
        std::vector<float> sum_fused(local_sum.size());
        kernel->multiply_fused(A.data(), C.data(), M, N, K, nullptr, nullptr, true, max_fused.data(), sum_fused.data(), false, 1.0f, 0.0f, &ctx, -1);

        // Check first block
        std::cout << "Sanity Check (Block 0):" << std::endl;
        std::cout << "  Max: Unfused=" << max_unfused[0] << ", Fused=" << max_fused[0] << std::endl;
        std::cout << "  Sum: Unfused=" << sum_unfused[0] << ", Fused=" << sum_fused[0] << std::endl;

        // Expect max to be identical (it's just max)
        // Expect sum to be close (fused uses fast exp approximation)
        if (std::abs(max_unfused[0] - max_fused[0]) > 1e-4)
        {
            std::cout << "WARNING: Max mismatch!" << std::endl;
        }
        // 5% tolerance for fast exp
        if (std::abs(sum_unfused[0] - sum_fused[0]) / sum_unfused[0] > 0.05)
        {
            std::cout << "WARNING: Sum mismatch > 5%!" << std::endl;
        }
        else
        {
            std::cout << "Sanity Check Passed." << std::endl;
        }
    }
};

TEST_F(Q8_1_GEMM_Fused_Perf, M1_SingleToken)
{
    BenchmarkConfig config;
    config.M = 1;
    config.N = 4096;
    config.K = 4096;
    config.warmup_iters = 10;
    config.bench_iters = 100;
    config.description = "Single Token Decode (M=1, N=4096, K=4096)";
    run_benchmark(config);
}

TEST_F(Q8_1_GEMM_Fused_Perf, M2_SmallBatch)
{
    BenchmarkConfig config;
    config.M = 2;
    config.N = 4096;
    config.K = 4096;
    config.warmup_iters = 10;
    config.bench_iters = 100;
    config.description = "Small Batch Decode (M=2, N=4096, K=4096)";
    run_benchmark(config);
}

TEST_F(Q8_1_GEMM_Fused_Perf, M32_Batch)
{
    BenchmarkConfig config;
    config.M = 32;
    config.N = 4096;
    config.K = 4096;
    config.warmup_iters = 5;
    config.bench_iters = 20;
    config.description = "Batch Prefill/Decode (M=32, N=4096, K=4096)";
    run_benchmark(config);
}

int main(int argc, char **argv)
{
    // Initialize MPI
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    // Initialize GTest
    ::testing::InitGoogleTest(&argc, argv);

    // Run tests
    int result = RUN_ALL_TESTS();

    // Finalize MPI
    MPI_Finalize();

    return result;
}
