/**
 * @file Perf__Q8_1_GEMM.cpp
 * @brief Performance benchmark for Q8_1 quantized GEMM operations
 *
 * This test benchmarks Q8_1 quantized matrix multiplication performance
 * (Q8_1 weights x Q8_1 activations). It measures:
 *   - Throughput (GFLOPS)
 *   - Time per iteration (ms)
 *
 * Test configuration is optimized for consistent performance measurement:
 *   - Runs on Release builds
 *   - Uses optimal MPI/OpenMP settings
 *   - Includes warmup iterations
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
#include <oneapi/dnnl/dnnl.hpp>

// V2 includes
#include "tensors/Tensors.h"
#include "kernels/cpu/gemm_v4/Q8_1GemmKernel.h"

using namespace llaminar2;

/**
 * @brief Configuration for a single benchmark run
 */
struct BenchmarkConfig
{
    int seq_len;             ///< Sequence length (m dimension)
    int in_features;         ///< Input feature dimension (k dimension)
    int out_features;        ///< Output feature dimension (n dimension)
    int warmup_iters;        ///< Number of warmup iterations
    int bench_iters;         ///< Number of timed benchmark iterations per trial
    int num_trials;          ///< Number of independent trials for statistics
    std::string description; ///< Human-readable description
};

/**
 * @brief Statistics for multiple benchmark trials
 */
struct BenchmarkStats
{
    double mean_ms;       ///< Mean time per iteration (ms)
    double stddev_ms;     ///< Standard deviation (ms)
    double min_ms;        ///< Minimum time (ms)
    double max_ms;        ///< Maximum time (ms)
    double mean_gflops;   ///< Mean throughput (GFLOPS)
    double stddev_gflops; ///< Standard deviation of throughput
    double l2_error;      ///< L2 error from verification
};

class Q8_1_GEMM_Perf : public ::testing::Test
{
protected:
    int rank_ = 0;
    int world_size_ = 1;

    void SetUp() override
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size_);

        int max_threads = omp_get_max_threads();
        if (rank_ == 0)
        {
            std::cout << "[Performance Test] OpenMP max threads: " << max_threads << std::endl;
        }
    }

    void compute_reference_gemm(int M, int N, int K, const float *A, const float *B, float *C)
    {
        using namespace dnnl;
        engine eng(engine::kind::cpu, 0);
        stream s(eng);

        // A: M x K, row-major
        memory::dims a_dims = {M, K};
        memory::dims a_strides = {K, 1};
        auto a_md = memory::desc(a_dims, memory::data_type::f32, a_strides);
        auto a_mem = memory(a_md, eng, (void *)A);

        // B: K x N, column-major (because B is passed as N x K row-major)
        // We want to compute A * B^T.
        // If B is N x K row-major, it is effectively K x N column-major.
        // So we treat it as K x N matrix.
        memory::dims b_dims = {K, N};
        memory::dims b_strides = {1, K};
        auto b_md = memory::desc(b_dims, memory::data_type::f32, b_strides);
        auto b_mem = memory(b_md, eng, (void *)B);

        // C: M x N, row-major
        memory::dims c_dims = {M, N};
        memory::dims c_strides = {N, 1};
        auto c_md = memory::desc(c_dims, memory::data_type::f32, c_strides);
        auto c_mem = memory(c_md, eng, (void *)C);

        // Matmul primitive
        auto matmul_pd = matmul::primitive_desc(eng, a_md, b_md, c_md);
        auto matmul_prim = matmul(matmul_pd);

        matmul_prim.execute(s, {{DNNL_ARG_SRC, a_mem},
                                {DNNL_ARG_WEIGHTS, b_mem},
                                {DNNL_ARG_DST, c_mem}});
        s.wait();
    }

    double verify_correctness(int M, int N, int K, const float *A, const float *weights_fp32, const float *C_actual)
    {
        if (rank_ != 0)
            return 0.0;

        std::vector<float> C_ref(M * N);
        compute_reference_gemm(M, N, K, A, weights_fp32, C_ref.data());

        double max_diff = 0.0;
        double mean_diff = 0.0;
        double max_rel_diff = 0.0;
        double sum_sq_diff = 0.0;
        double sum_sq_ref = 0.0;

        for (size_t i = 0; i < C_ref.size(); ++i)
        {
            double diff = std::abs(C_actual[i] - C_ref[i]);
            max_diff = std::max(max_diff, diff);
            mean_diff += diff;

            if (std::abs(C_ref[i]) > 1e-5)
            {
                max_rel_diff = std::max(max_rel_diff, diff / std::abs(C_ref[i]));
            }

            sum_sq_diff += diff * diff;
            sum_sq_ref += C_ref[i] * C_ref[i];
        }
        mean_diff /= C_ref.size();

        double l2_error = (sum_sq_ref > 0.0) ? std::sqrt(sum_sq_diff) / std::sqrt(sum_sq_ref) : 0.0;

        // Tolerance for Q8_1 Kernel vs Dequantized Reference
        // Since we are comparing against dequantized weights, error should be very low (arithmetic only).
        EXPECT_LT(l2_error, 0.01) << "Large divergence detected! L2 Error > 1%";

        return l2_error;
    }

    BenchmarkStats run_benchmark(const BenchmarkConfig &config)
    {
        int M = config.seq_len;
        int K = config.in_features;
        int N = config.out_features;

        // 1. Create random weights (N x K)
        std::vector<float> weights_fp32(N * K);
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto &x : weights_fp32)
            x = dist(gen);

        // 2. Quantize weights to Q8_1Tensor
        auto weights_tensor = Q8_1Tensor::quantize_from_fp32(weights_fp32.data(), {static_cast<size_t>(N), static_cast<size_t>(K)});

        // 2b. Dequantize weights for reference calculation (to exclude quantization error)
        // We want to verify the kernel's math, not the quantization loss.
        std::vector<float> weights_dequantized(N * K);
        weights_tensor->to_fp32(weights_dequantized.data());

        // 3. Create kernel
        auto kernel = weights_tensor->createGemm();
        if (!kernel)
        {
            throw std::runtime_error("Failed to create GEMM kernel");
        }

        // 4. Create random input A (M x K)
        std::vector<float> A(M * K);
        for (auto &x : A)
            x = dist(gen);

        // 5. Output buffer C (M x N)
        std::vector<float> C(M * N);

        // Verify correctness (once)
        kernel->multiply(A.data(), C.data(), M, N, K);
        double l2_error = verify_correctness(M, N, K, A.data(), weights_dequantized.data(), C.data());

        // Warmup
        for (int i = 0; i < config.warmup_iters; ++i)
        {
            kernel->multiply(A.data(), C.data(), M, N, K);
        }

        // Benchmark trials
        std::vector<double> trial_times_ms;
        trial_times_ms.reserve(config.num_trials);

        for (int t = 0; t < config.num_trials; ++t)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            auto start = std::chrono::high_resolution_clock::now();

            for (int i = 0; i < config.bench_iters; ++i)
            {
                kernel->multiply(A.data(), C.data(), M, N, K);
            }

            MPI_Barrier(MPI_COMM_WORLD);
            auto end = std::chrono::high_resolution_clock::now();

            double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
            trial_times_ms.push_back(total_ms / config.bench_iters);
        }

        // Calculate stats
        double sum = std::accumulate(trial_times_ms.begin(), trial_times_ms.end(), 0.0);
        double mean_ms = sum / config.num_trials;

        double sq_sum = std::inner_product(trial_times_ms.begin(), trial_times_ms.end(), trial_times_ms.begin(), 0.0);
        double stddev_ms = std::sqrt(sq_sum / config.num_trials - mean_ms * mean_ms);

        double min_ms = *std::min_element(trial_times_ms.begin(), trial_times_ms.end());
        double max_ms = *std::max_element(trial_times_ms.begin(), trial_times_ms.end());

        // GFLOPS = 2 * M * N * K / (time_s * 1e9)
        double ops = 2.0 * M * N * K;
        double mean_gflops = (ops / (mean_ms * 1e-3)) / 1e9;

        // Propagate error for GFLOPS stddev (approximate)
        double stddev_gflops = mean_gflops * (stddev_ms / mean_ms);

        return {mean_ms, stddev_ms, min_ms, max_ms, mean_gflops, stddev_gflops, l2_error};
    }

    void print_results(const BenchmarkConfig &config, const BenchmarkStats &stats)
    {
        if (rank_ != 0)
            return;

        std::cout << std::left << std::setw(40) << config.description
                  << " | M=" << std::setw(4) << config.seq_len
                  << " N=" << std::setw(6) << config.out_features
                  << " K=" << std::setw(6) << config.in_features
                  << " | Time: " << std::fixed << std::setprecision(3) << stats.mean_ms << " ms"
                  << " (+/- " << stats.stddev_ms << ")"
                  << " | Perf: " << std::setprecision(2) << stats.mean_gflops << " GFLOPS"
                  << " | L2 Err: " << std::scientific << std::setprecision(2) << stats.l2_error
                  << std::endl;
    }
};

// Main function for standalone execution
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

// Qwen 7B Sizes
// Hidden: 4096
// Intermediate: 11008
// Layers: 32

TEST_F(Q8_1_GEMM_Perf, Qwen7B_Attention_QKV)
{
    // QKV Projection: [M, 4096] -> [M, 12288]
    std::vector<int> batch_sizes = {1, 32, 128, 512};

    for (int m : batch_sizes)
    {
        BenchmarkConfig config;
        config.seq_len = m;
        config.in_features = 4096;
        config.out_features = 12288; // 4096 * 3
        config.warmup_iters = 5;
        config.bench_iters = 20;
        config.num_trials = 5;
        config.description = "Qwen7B Attn QKV";

        auto stats = run_benchmark(config);
        print_results(config, stats);
    }
}

TEST_F(Q8_1_GEMM_Perf, Qwen7B_Attention_Output)
{
    // Output Projection: [M, 4096] -> [M, 4096]
    std::vector<int> batch_sizes = {1, 32, 128, 512};

    for (int m : batch_sizes)
    {
        BenchmarkConfig config;
        config.seq_len = m;
        config.in_features = 4096;
        config.out_features = 4096;
        config.warmup_iters = 5;
        config.bench_iters = 20;
        config.num_trials = 5;
        config.description = "Qwen7B Attn Output";

        auto stats = run_benchmark(config);
        print_results(config, stats);
    }
}

TEST_F(Q8_1_GEMM_Perf, Qwen7B_FFN_GateUp)
{
    // FFN Gate/Up: [M, 4096] -> [M, 11008]
    std::vector<int> batch_sizes = {1, 32, 128, 512};

    for (int m : batch_sizes)
    {
        BenchmarkConfig config;
        config.seq_len = m;
        config.in_features = 4096;
        config.out_features = 11008;
        config.warmup_iters = 5;
        config.bench_iters = 20;
        config.num_trials = 5;
        config.description = "Qwen7B FFN GateUp";

        auto stats = run_benchmark(config);
        print_results(config, stats);
    }
}

TEST_F(Q8_1_GEMM_Perf, Qwen7B_FFN_Down)
{
    // FFN Down: [M, 11008] -> [M, 4096]
    std::vector<int> batch_sizes = {1, 32, 128, 512};

    for (int m : batch_sizes)
    {
        BenchmarkConfig config;
        config.seq_len = m;
        config.in_features = 11008;
        config.out_features = 4096;
        config.warmup_iters = 5;
        config.bench_iters = 20;
        config.num_trials = 5;
        config.description = "Qwen7B FFN Down";

        auto stats = run_benchmark(config);
        print_results(config, stats);
    }
}

// --- Qwen 0.5B Tests ---

TEST_F(Q8_1_GEMM_Perf, Qwen0_5B_Attention_Output)
{
    // Output Projection: [M, 896] -> [M, 896]
    std::vector<int> batch_sizes = {1, 32, 128, 512};

    for (int m : batch_sizes)
    {
        BenchmarkConfig config;
        config.seq_len = m;
        config.in_features = 896;
        config.out_features = 896;
        config.warmup_iters = 5;
        config.bench_iters = 20;
        config.num_trials = 5;
        config.description = "Qwen0.5B Attn Output";

        auto stats = run_benchmark(config);
        print_results(config, stats);
    }
}

TEST_F(Q8_1_GEMM_Perf, Qwen0_5B_FFN_Down)
{
    // FFN Down: [M, 4864] -> [M, 896]
    std::vector<int> batch_sizes = {1, 32, 128, 512};

    for (int m : batch_sizes)
    {
        BenchmarkConfig config;
        config.seq_len = m;
        config.in_features = 4864;
        config.out_features = 896;
        config.warmup_iters = 5;
        config.bench_iters = 20;
        config.num_trials = 5;
        config.description = "Qwen0.5B FFN Down";

        auto stats = run_benchmark(config);
        print_results(config, stats);
    }
}

// --- Qwen 32B Tests ---

TEST_F(Q8_1_GEMM_Perf, Qwen32B_Attention_Output)
{
    // Output Projection: [M, 5120] -> [M, 5120]
    std::vector<int> batch_sizes = {1, 32, 128, 512};

    for (int m : batch_sizes)
    {
        BenchmarkConfig config;
        config.seq_len = m;
        config.in_features = 5120;
        config.out_features = 5120;
        config.warmup_iters = 5;
        config.bench_iters = 20;
        config.num_trials = 5;
        config.description = "Qwen32B Attn Output";

        auto stats = run_benchmark(config);
        print_results(config, stats);
    }
}

TEST_F(Q8_1_GEMM_Perf, Qwen32B_FFN_Down)
{
    // FFN Down: [M, 27392] -> [M, 5120]
    std::vector<int> batch_sizes = {1, 2, 32, 128, 512};

    for (int m : batch_sizes)
    {
        BenchmarkConfig config;
        config.seq_len = m;
        config.in_features = 27392;
        config.out_features = 5120;
        config.warmup_iters = 5;
        config.bench_iters = 20;
        config.num_trials = 5;
        config.description = "Qwen32B FFN Down";

        auto stats = run_benchmark(config);
        print_results(config, stats);
    }
}
