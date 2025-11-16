/**
 * @file Perf__Q8_1_KC_Tuning.cpp
 * @brief Empirical KC blocking parameter tuning
 * @author David Sanftenberg
 *
 * This test suite empirically measures optimal KC values by:
 * 1. Testing multiple KC values (128, 256, 512, 1024, 2048)
 * 2. Measuring GFLOPS and working set size for each
 * 3. Identifying sweet spot where performance plateaus
 * 4. Validating that working set fits in L2 cache
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <vector>
#include <algorithm>

#include "kernels/cpu/gemm_v2/Q8_1GemmKernel.h"
#include "loaders/ModelLoader.h"
#include "tensors/Tensors.h"
#include "utils/CPUFeatures.h"

using namespace llaminar2;

class Q8_1_KC_Tuning : public ::testing::Test
{
protected:
    void SetUp() override
    {
        model_path_ = "models/qwen2.5-0.5b-instruct-q8_0.gguf";
        loader_ = std::make_unique<ModelLoader>();

        if (!loader_->loadModel(model_path_))
        {
            GTEST_SKIP() << "Model not found: " << model_path_;
        }

        // Get L2 cache info
        l2_total_ = cpu_l2_cache_total();
    }

    std::string model_path_;
    std::unique_ptr<ModelLoader> loader_;
    uint32_t l2_total_;

    /**
     * @brief Benchmark a specific KC configuration
     */
    struct BenchmarkResult
    {
        int kc_value;
        double gflops;
        size_t working_set_bytes;
        double l2_usage_pct;
        double avg_ms;
    };

    template <int KC_VALUE>
    BenchmarkResult benchmark_kc(int M, int N, int K,
                                 const Q8_1Tensor &A, const Q8_0Tensor &B)
    {
        using KernelConfig = Q8_1GemmKernelTemplate<32, 128, 1, 0, KC_VALUE, 2, 18>;

        std::vector<float> C(M * N, 0.0f);

        // Warmup
        for (int i = 0; i < 3; ++i)
        {
            std::fill(C.begin(), C.end(), 0.0f);
            KernelConfig::gemm(M, N, K, A, B, C.data(), N);
        }

        // Benchmark
        MPI_Barrier(MPI_COMM_WORLD);
        const int num_iters = 20;
        auto t0 = std::chrono::high_resolution_clock::now();

        for (int iter = 0; iter < num_iters; ++iter)
        {
            std::fill(C.begin(), C.end(), 0.0f);
            KernelConfig::gemm(M, N, K, A, B, C.data(), N);
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        MPI_Barrier(MPI_COMM_WORLD);

        double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double avg_ms = total_ms / num_iters;
        double flops = 2.0 * M * N * K;
        double gflops = flops / (avg_ms * 1e6);

        // Calculate working set size
        const int K_blocks = K / 32;
        const int kc_blocks = std::min(KC_VALUE, K_blocks);
        const int nc_blocks = 128; // Default NC estimate
        size_t working_set = KernelConfig::compute_working_set_size(kc_blocks, nc_blocks);

        BenchmarkResult result;
        result.kc_value = KC_VALUE;
        result.gflops = gflops;
        result.working_set_bytes = working_set;
        result.l2_usage_pct = 100.0 * working_set / l2_total_;
        result.avg_ms = avg_ms;

        return result;
    }
};

/**
 * @brief Sweep KC values for prefill workload (M=4096)
 *
 * Tests: KC ∈ {128, 256, 512, 1024, 2048}
 * Measures: GFLOPS, working set, L2 usage
 * Goal: Find KC where performance plateaus without exceeding L2 capacity
 */
TEST_F(Q8_1_KC_Tuning, PrefillWorkload_KC_Sweep)
{
    // Load Q8_0 weights
    auto wq_template = loader_->loadTensor("blk.0.attn_q.weight", 0, WeightPrecision::NATIVE);
    ASSERT_NE(wq_template, nullptr);
    auto q8_0_template = std::dynamic_pointer_cast<Q8_0Tensor>(wq_template);
    ASSERT_NE(q8_0_template, nullptr);

    const int M = 4096; // Large prefill
    const int N = 896;
    const int K = 896;

    // Create Q8_1 activation tensor
    std::vector<float> A_fp32(M * K);
    for (size_t i = 0; i < A_fp32.size(); ++i)
    {
        A_fp32[i] = 0.01f * ((i % 128) - 64);
    }
    auto q8_1_A = Q8_1Tensor::quantize_from_fp32(A_fp32.data(), {M, K});

    std::cout << "\n=== KC BLOCKING EMPIRICAL TUNING ===" << std::endl;
    std::cout << "Problem size: M=" << M << ", N=" << N << ", K=" << K << std::endl;
    std::cout << "L2 cache: " << (l2_total_ / 1024 / 1024) << " MB" << std::endl;
    std::cout << "\nTesting KC configurations..." << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    std::cout << std::setw(8) << "KC"
              << std::setw(12) << "GFLOPS"
              << std::setw(15) << "Working Set"
              << std::setw(12) << "L2 Usage"
              << std::setw(12) << "Time (ms)"
              << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    std::vector<BenchmarkResult> results;

    // Test KC=128
    {
        auto result = benchmark_kc<128>(M, N, K, *q8_1_A, *q8_0_template);
        results.push_back(result);
        std::cout << std::setw(8) << result.kc_value
                  << std::setw(12) << std::fixed << std::setprecision(1) << result.gflops
                  << std::setw(13) << (result.working_set_bytes / 1024) << " KB"
                  << std::setw(11) << std::setprecision(1) << result.l2_usage_pct << "%"
                  << std::setw(12) << std::setprecision(2) << result.avg_ms
                  << std::endl;
    }

    // Test KC=256
    {
        auto result = benchmark_kc<256>(M, N, K, *q8_1_A, *q8_0_template);
        results.push_back(result);
        std::cout << std::setw(8) << result.kc_value
                  << std::setw(12) << std::fixed << std::setprecision(1) << result.gflops
                  << std::setw(13) << (result.working_set_bytes / 1024) << " KB"
                  << std::setw(11) << std::setprecision(1) << result.l2_usage_pct << "%"
                  << std::setw(12) << std::setprecision(2) << result.avg_ms
                  << std::endl;
    }

    // Test KC=512
    {
        auto result = benchmark_kc<512>(M, N, K, *q8_1_A, *q8_0_template);
        results.push_back(result);
        std::cout << std::setw(8) << result.kc_value
                  << std::setw(12) << std::fixed << std::setprecision(1) << result.gflops
                  << std::setw(13) << (result.working_set_bytes / 1024) << " KB"
                  << std::setw(11) << std::setprecision(1) << result.l2_usage_pct << "%"
                  << std::setw(12) << std::setprecision(2) << result.avg_ms
                  << std::endl;
    }

    // Test KC=896 (K_blocks exactly)
    {
        auto result = benchmark_kc<896 / 32>(M, N, K, *q8_1_A, *q8_0_template);
        results.push_back(result);
        std::cout << std::setw(8) << result.kc_value
                  << std::setw(12) << std::fixed << std::setprecision(1) << result.gflops
                  << std::setw(13) << (result.working_set_bytes / 1024) << " KB"
                  << std::setw(11) << std::setprecision(1) << result.l2_usage_pct << "%"
                  << std::setw(12) << std::setprecision(2) << result.avg_ms
                  << std::endl;
    }

    std::cout << std::string(80, '-') << std::endl;

    // Find best performance
    auto best_result = *std::max_element(results.begin(), results.end(),
                                         [](const BenchmarkResult &a, const BenchmarkResult &b)
                                         {
                                             return a.gflops < b.gflops;
                                         });

    std::cout << "\n🏆 BEST PERFORMANCE: KC=" << best_result.kc_value
              << " (" << std::fixed << std::setprecision(1) << best_result.gflops << " GFLOPS)" << std::endl;

    // Find largest KC that fits in 40% of L2
    size_t l2_budget = static_cast<size_t>(l2_total_ * 0.4);
    auto best_fit = results.front();
    for (const auto &r : results)
    {
        if (r.working_set_bytes <= l2_budget)
        {
            best_fit = r;
        }
    }

    std::cout << "📊 BEST FIT (≤40% L2): KC=" << best_fit.kc_value
              << " (" << (best_fit.working_set_bytes / 1024) << " KB, "
              << std::setprecision(1) << best_fit.l2_usage_pct << "% L2)" << std::endl;

    // Analyze performance plateau
    double max_gflops = best_result.gflops;
    std::cout << "\n📈 PERFORMANCE PLATEAU ANALYSIS:" << std::endl;
    for (const auto &r : results)
    {
        double pct_of_max = 100.0 * r.gflops / max_gflops;
        std::string verdict = (pct_of_max >= 98.0) ? "✅ PLATEAU" : (pct_of_max >= 95.0) ? "⚠️  CLOSE"
                                                                                         : "❌ SLOW";
        std::cout << "  KC=" << std::setw(4) << r.kc_value
                  << ": " << std::setw(6) << std::fixed << std::setprecision(1) << pct_of_max
                  << "% of max  " << verdict << std::endl;
    }

    std::cout << "\n💡 RECOMMENDATION:" << std::endl;
    if (best_result.kc_value == best_fit.kc_value)
    {
        std::cout << "  Current choice (KC=" << best_fit.kc_value
                  << ") is optimal - achieves best GFLOPS within L2 budget" << std::endl;
    }
    else if (best_result.l2_usage_pct <= 50.0)
    {
        std::cout << "  Consider increasing KC to " << best_result.kc_value
                  << " (only " << std::setprecision(1) << best_result.l2_usage_pct
                  << "% L2, +" << std::setprecision(1)
                  << (best_result.gflops - best_fit.gflops) << " GFLOPS)" << std::endl;
    }
    else
    {
        std::cout << "  Trade-off: KC=" << best_result.kc_value
                  << " faster but uses " << std::setprecision(1) << best_result.l2_usage_pct
                  << "% L2" << std::endl;
        std::cout << "  Conservative: KC=" << best_fit.kc_value
                  << " (safer, only " << std::setprecision(1) << best_fit.l2_usage_pct
                  << "% L2)" << std::endl;
    }
}

/**
 * @brief Test KC scaling with problem size
 *
 * Measures whether optimal KC changes with M (batch size)
 */
TEST_F(Q8_1_KC_Tuning, KC_Scaling_With_M)
{
    auto wq_template = loader_->loadTensor("blk.0.attn_q.weight", 0, WeightPrecision::NATIVE);
    ASSERT_NE(wq_template, nullptr);
    auto q8_0_template = std::dynamic_pointer_cast<Q8_0Tensor>(wq_template);

    const int N = 896;
    const int K = 896;
    std::vector<int> M_values = {128, 512, 2048, 4096};

    std::cout << "\n=== KC SCALING WITH BATCH SIZE ===" << std::endl;
    std::cout << "Problem: N=" << N << ", K=" << K << std::endl;

    for (int M : M_values)
    {
        std::vector<float> A_fp32(M * K);
        for (size_t i = 0; i < A_fp32.size(); ++i)
        {
            A_fp32[i] = 0.01f * ((i % 128) - 64);
        }
        auto q8_1_A = Q8_1Tensor::quantize_from_fp32(A_fp32.data(), {M, K});

        std::cout << "\nM=" << M << ":" << std::endl;

        auto r256 = benchmark_kc<256>(M, N, K, *q8_1_A, *q8_0_template);
        auto r512 = benchmark_kc<512>(M, N, K, *q8_1_A, *q8_0_template);

        std::cout << "  KC=256: " << std::fixed << std::setprecision(1) << r256.gflops << " GFLOPS" << std::endl;
        std::cout << "  KC=512: " << std::fixed << std::setprecision(1) << r512.gflops << " GFLOPS"
                  << " (" << std::showpos << std::setprecision(1)
                  << (100.0 * (r512.gflops - r256.gflops) / r256.gflops) << "%)" << std::noshowpos << std::endl;
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
