/**
 * @file Perf__CudaGemmHeuristicCanary.cpp
 * @brief Canary tests for ML heuristic generalization
 *
 * These tests use matrix shapes NOT included in the training data to validate
 * that the ML model generalizes well to unseen configurations. They should
 * never generate training data (no CSV export).
 *
 * @author David Sanftenberg
 */

#include "../../src/v2/kernels/cuda/CudaGemmAutoTuner.h"
#include "../../src/v2/kernels/cuda/CudaGemmVariantsBaseline.h"
#include "../../src/v2/kernels/cuda/IQ4_NL_BlockDecoder.h"
#include "../../src/v2/tensors/FP16Utils.h"
#include <gtest/gtest.h>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <iostream>

using namespace llaminar2::cuda;

/**
 * @brief Canary test fixture for heuristic generalization validation
 *
 * Key differences from main validation suite:
 * - Uses UNTRAINED matrix shapes (intermediate sizes, odd dimensions)
 * - Does NOT export to CSV (no training data collection)
 * - Focuses on top-K hit rate rather than correlation
 * - Tests generalization, not memorization
 */
class CudaGemmHeuristicCanary : public ::testing::Test
{
protected:
    cudaStream_t stream_;
    cudaEvent_t start_event_, stop_event_;
    float *test_A_device_;
    IQ4_NLBlock *test_B_device_;
    float *test_C_device_;
    int allocated_m_ = 0, allocated_n_ = 0, allocated_k_ = 0;
    int warmup_iterations_ = 3;
    int benchmark_iterations_ = 10;

    void SetUp() override
    {
        int device_count;
        cudaGetDeviceCount(&device_count);
        if (device_count == 0)
        {
            GTEST_SKIP() << "No CUDA devices available";
        }

        cudaSetDevice(0);
        cudaStreamCreate(&stream_);
        cudaEventCreate(&start_event_);
        cudaEventCreate(&stop_event_);

        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        std::cout << "[Canary] Device: " << prop.name << std::endl;
        std::cout << "[Canary] Testing ML heuristic generalization (NO training data export)" << std::endl;
    }

    void TearDown() override
    {
        if (test_A_device_)
            cudaFree(test_A_device_);
        if (test_B_device_)
            cudaFree(test_B_device_);
        if (test_C_device_)
            cudaFree(test_C_device_);
        if (start_event_)
            cudaEventDestroy(start_event_);
        if (stop_event_)
            cudaEventDestroy(stop_event_);
        if (stream_)
            cudaStreamDestroy(stream_);
    }

    void ensureAllocation(int m, int n, int k)
    {
        if (m <= allocated_m_ && n <= allocated_n_ && k <= allocated_k_)
        {
            return;
        }

        if (test_A_device_)
            cudaFree(test_A_device_);
        if (test_B_device_)
            cudaFree(test_B_device_);
        if (test_C_device_)
            cudaFree(test_C_device_);

        cudaMalloc(&test_A_device_, m * k * sizeof(float));
        cudaMalloc(&test_B_device_, n * (k / 32) * sizeof(IQ4_NLBlock));
        cudaMalloc(&test_C_device_, m * n * sizeof(float));

        allocated_m_ = m;
        allocated_n_ = n;
        allocated_k_ = k;

        // Initialize with random data
        std::vector<float> A_host(m * k);
        std::vector<IQ4_NLBlock> B_host(n * (k / 32));

        for (auto &val : A_host)
            val = static_cast<float>(rand()) / RAND_MAX;
        for (auto &block : B_host)
        {
            block.d = llaminar2::fp32_to_fp16(1.0f);
            for (int i = 0; i < 16; ++i)
                block.qs[i] = rand() % 256;
        }

        cudaMemcpy(test_A_device_, A_host.data(), A_host.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(test_B_device_, B_host.data(), B_host.size() * sizeof(IQ4_NLBlock), cudaMemcpyHostToDevice);
    }

    CudaBenchmarkResult benchmarkConfig(const CudaGemmConfig &config, int m, int n, int k)
    {
        CudaBenchmarkResult result;
        result.config = config;

        // Warmup
        for (int i = 0; i < warmup_iterations_; ++i)
        {
            auto err = launchIQ4NLGemmVariant(test_A_device_, test_B_device_, test_C_device_,
                                              m, n, k, config, stream_);
            if (err != cudaSuccess)
            {
                result.gflops = 0.0;
                result.time_ms = 1e9;
                result.iterations = 0;
                return result;
            }
        }
        cudaStreamSynchronize(stream_);

        // Timed runs
        cudaEventRecord(start_event_, stream_);
        for (int i = 0; i < benchmark_iterations_; ++i)
        {
            launchIQ4NLGemmVariant(test_A_device_, test_B_device_, test_C_device_,
                                   m, n, k, config, stream_);
        }
        cudaEventRecord(stop_event_, stream_);
        cudaEventSynchronize(stop_event_);

        float elapsed_ms;
        cudaEventElapsedTime(&elapsed_ms, start_event_, stop_event_);

        result.time_ms = elapsed_ms / benchmark_iterations_;
        result.iterations = benchmark_iterations_;

        double flops = 2.0 * m * n * k;
        result.gflops = (flops / 1e9) / (result.time_ms / 1000.0);

        return result;
    }

    /**
     * @brief Evaluate heuristic top-K hit rate (generalization metric)
     */
    std::tuple<bool, double, int, double> evaluateTopKHitRate(
        int m, int n, int k, const std::string &test_name, int top_k = 20)
    {

        ensureAllocation(m, n, k);

        auto &tuner = CudaGemmAutoTuner::instance();
        auto all_configs = tuner.getAvailableConfigs();

        std::cout << "[Canary] " << test_name << " [" << m << "x" << n << "x" << k << "] - "
                  << all_configs.size() << " configs" << std::endl;

        // Benchmark all configs
        std::vector<CudaBenchmarkResult> results;
        int valid_count = 0;
        for (const auto &config : all_configs)
        {
            auto result = benchmarkConfig(config, m, n, k);
            if (result.gflops > 0.0)
            {
                results.push_back(result);
                valid_count++;
            }
        }

        if (results.empty())
        {
            std::cout << "[Canary]   ERROR: No valid configurations!" << std::endl;
            return {false, 0.0, -1, 0.0};
        }

        // Sort by GFLOPS
        std::sort(results.begin(), results.end(),
                  [](const CudaBenchmarkResult &a, const CudaBenchmarkResult &b)
                  {
                      return a.gflops > b.gflops;
                  });

        double best_gflops = results[0].gflops;

        // Get ML heuristic ranking
        auto heuristic_ranked = tuner.rankByPerformanceModel(all_configs, m, n, k);

        if (heuristic_ranked.empty())
        {
            std::cout << "[Canary]   ERROR: Heuristic returned no configs!" << std::endl;
            return {false, best_gflops, -1, 0.0};
        }

        // Find where heuristic #1 ranks empirically
        auto heuristic_best = heuristic_ranked[0];
        int rank = -1;
        double heuristic_gflops = 0.0;

        for (size_t i = 0; i < results.size(); ++i)
        {
            if (results[i].config == heuristic_best)
            {
                rank = static_cast<int>(i) + 1;
                heuristic_gflops = results[i].gflops;
                break;
            }
        }

        bool in_top_k = (rank > 0 && rank <= top_k);

        std::cout << "[Canary]   Best empirical: " << std::fixed << std::setprecision(1)
                  << best_gflops << " GFLOPS" << std::endl;
        std::cout << "[Canary]   ML heuristic: Rank #" << rank << " (" << heuristic_gflops << " GFLOPS)" << std::endl;
        std::cout << "[Canary]   Top-" << top_k << " hit: " << (in_top_k ? "YES ✓" : "NO ✗") << std::endl;

        if (!in_top_k)
        {
            std::cout << "[Canary]   ⚠️  Heuristic missed top-" << top_k << " (may indicate overfitting)" << std::endl;
            std::cout << "[Canary]   Performance gap: " << (best_gflops - heuristic_gflops)
                      << " GFLOPS (" << std::setprecision(1)
                      << (100.0 * (best_gflops - heuristic_gflops) / best_gflops) << "% slower)" << std::endl;
        }

        return {in_top_k, best_gflops, rank, heuristic_gflops};
    }
};

// ============================================================================
// Canary Tests: Intermediate Model Sizes (NOT in training data)
// ============================================================================

/**
 * @brief Test 1.5B model size (between 0.5B and 4B)
 *
 * Training data has 896 (0.5B) and 2560 (4B).
 * This tests 1280 - right in the middle.
 */
TEST_F(CudaGemmHeuristicCanary, Qwen_1_5B_SingleToken_QKV)
{
    const int m = 1;    // Single token
    const int n = 1280; // 1.5B model dimension (UNTRAINED)
    const int k = 1280;

    auto [in_top_k, best_gflops, rank, heuristic_gflops] =
        evaluateTopKHitRate(m, n, k, "Qwen 1.5B Single Token QKV");

    // Expect at least top-30 (looser than training data)
    EXPECT_LE(rank, 30) << "Heuristic should generalize to intermediate sizes";
    EXPECT_GT(heuristic_gflops, 0.0) << "Heuristic should find a valid config";
}

/**
 * @brief Test odd batch size
 *
 * Training data uses batch 32, 128, 256.
 * This tests batch=17 (prime, unusual).
 */
TEST_F(CudaGemmHeuristicCanary, SmallOddBatch_17x2048x2048)
{
    const int m = 17; // Odd batch (UNTRAINED)
    const int n = 2048;
    const int k = 2048;

    auto [in_top_k, best_gflops, rank, heuristic_gflops] =
        evaluateTopKHitRate(m, n, k, "Batch=17 Odd Batch Size");

    EXPECT_LE(rank, 30) << "Heuristic should handle odd batch sizes";
    EXPECT_GT(heuristic_gflops, best_gflops * 0.7) << "Should be within 30% of optimal";
}

/**
 * @brief Test non-power-of-2 dimensions
 *
 * Most training data uses nice round numbers (896, 2048, 4096, etc.).
 * Real-world models may have odd dimensions.
 */
TEST_F(CudaGemmHeuristicCanary, OddDimension_1537x2049)
{
    const int m = 1;
    const int n = 1537; // Prime-ish number
    const int k = 2048; // Standard dimension

    auto [in_top_k, best_gflops, rank, heuristic_gflops] =
        evaluateTopKHitRate(m, n, k, "Odd Dimensions 1x1537x2048");

    EXPECT_LE(rank, 40) << "Heuristic should handle non-aligned dimensions";
}

/**
 * @brief Summary test: Aggregate generalization score
 */
TEST_F(CudaGemmHeuristicCanary, OverallGeneralizationScore)
{
    struct CanaryConfig
    {
        int m, n, k;
        std::string name;
    };

    std::vector<CanaryConfig> canaries = {
        {1, 1280, 1280, "1.5B interpolation"},
        {17, 2048, 2048, "Odd batch"},
        {1, 1537, 2048, "Odd dimension"}};

    int top_20_hits = 0;
    int top_30_hits = 0;
    int total = canaries.size();

    for (const auto &canary : canaries)
    {
        auto [in_top_k, best, rank, heur] =
            evaluateTopKHitRate(canary.m, canary.n, canary.k, canary.name, 30);

        if (rank > 0 && rank <= 20)
            top_20_hits++;
        if (rank > 0 && rank <= 30)
            top_30_hits++;
    }

    double top_20_rate = static_cast<double>(top_20_hits) / total * 100.0;
    double top_30_rate = static_cast<double>(top_30_hits) / total * 100.0;

    std::cout << "═══════════════════════════════════════════════════════" << std::endl;
    std::cout << "Canary Generalization Summary" << std::endl;
    std::cout << "═══════════════════════════════════════════════════════" << std::endl;
    std::cout << "Top-20 hit rate: " << std::fixed << std::setprecision(1)
              << top_20_rate << "% (" << top_20_hits << "/" << total << ")" << std::endl;
    std::cout << "Top-30 hit rate: " << top_30_rate << "% (" << top_30_hits << "/" << total << ")" << std::endl;
    std::cout << "═══════════════════════════════════════════════════════" << std::endl;

    // Good generalization: >33% top-20 (at least 1/3), >66% top-30 (at least 2/3)
    EXPECT_GE(top_20_rate, 33.0) << "Model should hit top-20 for ≥33% of unseen configs";
    EXPECT_GE(top_30_rate, 66.0) << "Model should hit top-30 for ≥66% of unseen configs";
}
