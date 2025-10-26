/**
 * @file Perf__L1OptimizationComparison.cpp
 * @brief Compare original vs L1-optimized GEMM kernels
 *
 * Measures performance and L1 cache behavior for both implementations.
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <cmath>

#include "loaders/ModelLoader.h"
#include "tensors/Tensors.h"
#include "kernels/cpu/QuantizedGemm.h"
#include "kernels/cpu/QuantizedGemmL1Opt.h"

using namespace llaminar2;

class L1OptimizationComparison : public ::testing::Test
{
protected:
    void SetUp() override
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size_);

        // Load model
        std::string model_path = "models/qwen2.5-0.5b-instruct-iq4_nl.gguf";
        loader_ = std::make_unique<ModelLoader>();

        if (rank_ == 0)
        {
            std::cout << "[L1 Opt Comparison] Loading model: " << model_path << std::endl;
        }

        if (!loader_->loadModel(model_path))
        {
            if (rank_ == 0)
            {
                std::cerr << "[L1 Opt Comparison] Failed to load model" << std::endl;
            }
            GTEST_SKIP() << "Model not available";
        }

        if (rank_ == 0)
        {
            std::cout << "[L1 Opt Comparison] Model loaded successfully" << std::endl;
        }
    }

    void TearDown() override
    {
        MPI_Barrier(MPI_COMM_WORLD);
        loader_.reset();
    }

    std::shared_ptr<IQ4_NLTensor> getWeightTensor()
    {
        auto weight_base = loader_->loadTensor("blk.0.attn_q.weight", -1);
        EXPECT_NE(weight_base, nullptr) << "Failed to load weight";

        auto weight = std::dynamic_pointer_cast<IQ4_NLTensor>(weight_base);
        EXPECT_NE(weight, nullptr) << "Weight is not IQ4_NL tensor";

        return weight;
    }

    std::shared_ptr<FP32Tensor> createFP32Activation(int seq_len, int features)
    {
        auto activation = std::make_shared<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(features)});

        float *data = activation->mutable_data();
        size_t total = seq_len * features;

        for (size_t i = 0; i < total; ++i)
        {
            data[i] = (static_cast<float>(i % 1000) / 1000.0f) - 0.5f;
        }

        return activation;
    }

    struct BenchmarkResult
    {
        double mean_ms;
        double stddev_ms;
        double mean_gflops;
        double cv_percent;
    };

    BenchmarkResult benchmarkKernel(
        ITensorGemm *gemm,
        const float *A, float *C,
        int m, int n, int k,
        int num_trials = 5, int num_iters = 50)
    {
        std::vector<double> times_ms;

        // Warmup
        for (int i = 0; i < 3; ++i)
        {
            gemm->multiply(A, C, m, n, k, true, 1.0f, 0.0f, nullptr, -1);
        }

        MPI_Barrier(MPI_COMM_WORLD);

        // Benchmark
        for (int trial = 0; trial < num_trials; ++trial)
        {
            auto start = std::chrono::high_resolution_clock::now();

            for (int iter = 0; iter < num_iters; ++iter)
            {
                gemm->multiply(A, C, m, n, k, true, 1.0f, 0.0f, nullptr, -1);
            }

            auto end = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
            times_ms.push_back(elapsed / num_iters);
        }

        // Compute statistics
        double sum = 0.0, sum_sq = 0.0;
        for (double t : times_ms)
        {
            sum += t;
            sum_sq += t * t;
        }

        double mean_ms = sum / num_trials;
        double variance = (sum_sq / num_trials) - (mean_ms * mean_ms);
        double stddev_ms = std::sqrt(std::max(0.0, variance));

        double flops = 2.0 * m * n * k; // multiply + add
        double mean_gflops = flops / (mean_ms * 1e6);
        double cv_percent = (stddev_ms / mean_ms) * 100.0;

        return {mean_ms, stddev_ms, mean_gflops, cv_percent};
    }

    int rank_;
    int world_size_;
    std::unique_ptr<ModelLoader> loader_;
};

TEST_F(L1OptimizationComparison, LargeBatch_OriginalVsL1Opt)
{
    // Configuration: 512×896×896 (LargeBatch_Prefill workload)
    const int m = 512;
    const int n = 896;
    const int k = 896;

    auto weight = getWeightTensor();
    auto activation = createFP32Activation(m, k);
    auto output = std::make_shared<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(m), static_cast<size_t>(n)});

    const float *A = activation->data();
    float *C = output->mutable_data();

    // Create both kernel implementations
    QuantizedGemmKernel original_kernel(weight.get());
    QuantizedGemmL1Opt l1opt_kernel(weight.get());

    if (rank_ == 0)
    {
        std::cout << "\n╔════════════════════════════════════════════════════════════════╗" << std::endl;
        std::cout << "║ L1 Cache Optimization Comparison - LargeBatch (512×896×896)   ║" << std::endl;
        std::cout << "╚════════════════════════════════════════════════════════════════╝\n" << std::endl;
    }

    // Benchmark original kernel
    if (rank_ == 0)
    {
        std::cout << "Testing ORIGINAL kernel..." << std::endl;
    }
    auto original_result = benchmarkKernel(&original_kernel, A, C, m, n, k);

    // Benchmark L1-optimized kernel
    if (rank_ == 0)
    {
        std::cout << "Testing L1-OPTIMIZED kernel..." << std::endl;
    }
    auto l1opt_result = benchmarkKernel(&l1opt_kernel, A, C, m, n, k);

    // Print results
    if (rank_ == 0)
    {
        std::cout << "\n╔════════════════════════════════════════════════════════════════╗" << std::endl;
        std::cout << "║ Performance Comparison Results                                 ║" << std::endl;
        std::cout << "╠════════════════════════════════════════════════════════════════╣" << std::endl;
        std::cout << "║ Metric               │  Original  │  L1-Opt    │  Speedup      ║" << std::endl;
        std::cout << "╟──────────────────────┼────────────┼────────────┼───────────────╢" << std::endl;

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "║ Time (ms)            │ " << std::setw(10) << original_result.mean_ms
                  << " │ " << std::setw(10) << l1opt_result.mean_ms
                  << " │ " << std::setw(13) << (original_result.mean_ms / l1opt_result.mean_ms) << "× ║" << std::endl;

        std::cout << "║ Throughput (GFLOPS)  │ " << std::setw(10) << original_result.mean_gflops
                  << " │ " << std::setw(10) << l1opt_result.mean_gflops
                  << " │ " << std::setw(13) << (l1opt_result.mean_gflops / original_result.mean_gflops) << "× ║" << std::endl;

        std::cout << "║ Consistency (CV%)    │ " << std::setw(10) << original_result.cv_percent
                  << " │ " << std::setw(10) << l1opt_result.cv_percent
                  << " │ " << std::setw(13) << (original_result.cv_percent / l1opt_result.cv_percent) << "× ║" << std::endl;

        std::cout << "╚════════════════════════════════════════════════════════════════╝" << std::endl;

        double speedup = l1opt_result.mean_gflops / original_result.mean_gflops;
        if (speedup > 1.05)
        {
            std::cout << "\n✅ L1 optimization SUCCESSFUL: " << std::setprecision(1) << ((speedup - 1.0) * 100) << "% faster" << std::endl;
        }
        else if (speedup > 0.95)
        {
            std::cout << "\n⚠️ L1 optimization NEUTRAL: Performance within 5% margin" << std::endl;
        }
        else
        {
            std::cout << "\n❌ L1 optimization REGRESSED: " << std::setprecision(1) << ((1.0 - speedup) * 100) << "% slower" << std::endl;
        }

        std::cout << "\nTo measure L1 cache misses, run with:" << std::endl;
        std::cout << "  perf stat -e L1-dcache-load-misses,L1-dcache-loads \\" << std::endl;
        std::cout << "    ./build_v2_release/performance/v2_perf_l1_opt_comparison" << std::endl;
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
