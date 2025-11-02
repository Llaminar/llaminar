/**
 * @file Perf__CudaGemmHeuristicValidation.cpp
 * @brief Validates CUDA GEMM auto-selection heuristics against empirical performance
 *
 * This test benchmarks ALL kernel variants for matrix sizes common in 0.5B and 4B models,
 * then compares the heuristic's top-ranked kernels against actual measured performance.
 *
 * Matrix dimensions for Qwen models:
 * - 0.5B: d_model=896, n_heads=14, d_ff=4864, vocab=151936
 *   - Q/K/V projections: [1, 896, 896] or [batch, 896, 896]
 *   - Attention output: [1, 896, 896]
 *   - FFN gate/up: [1, 4864, 896]
 *   - FFN down: [1, 896, 4864]
 *   - LM head: [1, 151936, 896] (huge!)
 *
 * - 4B: d_model=2560, n_heads=20, d_ff=13824, vocab=152064
 *   - Q/K/V projections: [1, 2560, 2560]
 *   - FFN gate/up: [1, 13824, 2560]
 *   - FFN down: [1, 2560, 13824]
 *   - LM head: [1, 152064, 2560]
 *
 * - 7B: d_model=4096, n_heads=32, d_ff=22016, vocab=152064
 *   - Q/K/V projections: [1, 4096, 4096] or [batch, 4096, 4096]
 *   - FFN gate/up: [1, 22016, 4096]
 *   - FFN down: [1, 4096, 22016]
 *   - LM head: [1, 152064, 4096]
 *
 * - 14B: d_model=5120, n_heads=40, d_ff=27648, vocab=152064
 *   - Q/K/V projections: [1, 5120, 5120] or [batch, 5120, 5120]
 *   - FFN gate/up: [1, 27648, 5120]
 *   - FFN down: [1, 5120, 27648]
 *   - LM head: [1, 152064, 5120]
 *
 * Batch prefill scenarios (e.g., batch=32, batch=128):
 * - 0.5B: [32, 896, 896], [32, 4864, 896], etc.
 * - 4B: [32, 2560, 2560], [32, 13824, 2560], etc.
 * - 7B: [128, 4096, 4096], [128, 22016, 4096], etc.
 * - 14B: [128, 5120, 5120], [128, 27648, 5120], etc.
 *
 * @author David Sanftenberg
 * @date November 1, 2025
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
#include <fstream>
#include <sstream>

using namespace llaminar2::cuda;

/**
 * @brief Test fixture for heuristic validation
 */
class CudaGemmHeuristicValidation : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Get device properties
        cudaGetDevice(&device_id_);
        cudaGetDeviceProperties(&device_props_, device_id_);

        // Create CUDA resources
        cudaStreamCreate(&stream_);
        cudaEventCreate(&start_event_);
        cudaEventCreate(&stop_event_);

        // Use fewer iterations for faster testing (still statistically valid)
        warmup_iterations_ = 2;
        benchmark_iterations_ = 5;
    }

    void TearDown() override
    {
        if (stream_)
            cudaStreamDestroy(stream_);
        if (start_event_)
            cudaEventDestroy(start_event_);
        if (stop_event_)
            cudaEventDestroy(stop_event_);

        if (test_A_device_)
            cudaFree(test_A_device_);
        if (test_B_device_)
            cudaFree(test_B_device_);
        if (test_C_device_)
            cudaFree(test_C_device_);
    }

    /**
     * @brief Allocate test data for specific matrix size
     */
    void allocateTestData(int m, int n, int k)
    {
        if (m > allocated_m_ || n > allocated_n_ || k > allocated_k_)
        {
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
    }

    /**
     * @brief Benchmark a single configuration
     */
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
                // Most configs fail with CONFIG_NOT_FOUND (not compiled)
                // This is expected - only 648 valid configs exist
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

        // Compute GFLOPS: 2*m*n*k FLOPs per GEMM
        double flops = 2.0 * m * n * k;
        result.gflops = (flops / 1e9) / (result.time_ms / 1000.0);

        return result;
    }

    /**
     * @brief Rank all configurations using heuristic
     *
     * Delegates to CudaGemmAutoTuner::rankByPerformanceModel() which supports:
     * - Manual heuristic (default, DEPRECATED - has -12,000 correlation)
     * - ML heuristic (enabled via LLAMINAR_USE_ML_HEURISTIC=1 - data-driven)
     */
    std::vector<CudaGemmConfig> rankByHeuristic(const std::vector<CudaGemmConfig> &configs,
                                                int m, int n, int k)
    {
        // Use the AutoTuner's ranking method (respects LLAMINAR_USE_ML_HEURISTIC)
        auto &tuner = CudaGemmAutoTuner::instance();
        return tuner.rankByPerformanceModel(configs, m, n, k);
    }

    /**
     * @brief Print detailed comparison table
     */
    void printComparisonTable(const std::string &shape_desc,
                              const std::vector<CudaBenchmarkResult> &empirical_top10,
                              const std::vector<CudaGemmConfig> &heuristic_top10,
                              int m, int n, int k)
    {
        std::cout << "\n╔══════════════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║ " << std::left << std::setw(72) << shape_desc << " ║\n";
        std::cout << "╠══════════════════════════════════════════════════════════════════════════╣\n";
        std::cout << "║ Shape: [" << m << " × " << n << " × " << k << "]" << std::setw(72 - 20 - std::to_string(m).length() - std::to_string(n).length() - std::to_string(k).length()) << "" << " ║\n";
        std::cout << "╠════════╦══════════════════════════════╦═════════╦═════════════════════════╣\n";
        std::cout << "║ Rank   ║ Empirical Best (Measured)    ║ GFLOPS  ║ Heuristic Prediction    ║\n";
        std::cout << "╠════════╬══════════════════════════════╬═════════╬═════════════════════════╣\n";

        for (int i = 0; i < 10 && i < static_cast<int>(empirical_top10.size()); ++i)
        {
            const auto &emp = empirical_top10[i];
            const auto &heur = heuristic_top10[i];

            std::cout << "║ " << std::setw(6) << (i + 1) << " ║ ";
            std::cout << std::setw(28) << emp.config.id() << " ║ ";
            std::cout << std::setw(7) << std::fixed << std::setprecision(1) << emp.gflops << " ║ ";
            std::cout << std::setw(23) << heur.id() << " ║\n";
        }

        std::cout << "╚════════╩══════════════════════════════╩═════════╩═════════════════════════╝\n";
    }

    /**
     * @brief Compute ranking correlation (Spearman's rho approximation)
     */
    double computeRankCorrelation(const std::vector<CudaBenchmarkResult> &empirical,
                                  const std::vector<CudaGemmConfig> &heuristic)
    {
        // Build empirical ranking map: config.id() -> rank
        std::unordered_map<std::string, int> empirical_ranks;
        for (size_t i = 0; i < empirical.size(); ++i)
        {
            empirical_ranks[empirical[i].config.id()] = i;
        }

        // Compute rank differences for configs in heuristic top-10
        double sum_squared_diff = 0.0;
        int count = 0;

        for (size_t h_rank = 0; h_rank < std::min(size_t(10), heuristic.size()); ++h_rank)
        {
            auto it = empirical_ranks.find(heuristic[h_rank].id());
            if (it != empirical_ranks.end())
            {
                int e_rank = it->second;
                int diff = static_cast<int>(h_rank) - e_rank;
                sum_squared_diff += diff * diff;
                count++;
            }
        }

        if (count == 0)
            return 0.0;

        // Spearman's rho: 1 - (6 * sum_d^2) / (n * (n^2 - 1))
        double n = count;
        double rho = 1.0 - (6.0 * sum_squared_diff) / (n * (n * n - 1.0));
        return rho;
    }

    /**
     * @brief Export benchmark results to CSV for regression analysis
     */
    void exportToCSV(const std::string &filename,
                     const std::string &test_name,
                     int m, int n, int k,
                     const std::vector<CudaBenchmarkResult> &results)
    {
        std::ofstream csv;

        // Check if file exists to determine if we need header
        bool file_exists = std::ifstream(filename).good();
        csv.open(filename, std::ios::app);

        if (!file_exists)
        {
            // Write header
            csv << "test_name,m,n,k,"
                << "tile_m,tile_n,tile_k,"
                << "threads_m,threads_n,"
                << "work_m,work_n,"
                << "prefetch_stages,transpose_smem,vectorize_load,"
                << "gflops,time_ms,iterations\n";
        }

        // Write data rows
        for (const auto &result : results)
        {
            const auto &cfg = result.config;
            csv << test_name << ","
                << m << "," << n << "," << k << ","
                << cfg.tile_m << "," << cfg.tile_n << "," << cfg.tile_k << ","
                << cfg.threads_m << "," << cfg.threads_n << ","
                << cfg.work_per_thread_m << "," << cfg.work_per_thread_n << ","
                << cfg.prefetch_stages << "," << (cfg.transpose_smem ? 1 : 0) << "," << cfg.vectorize_load << ","
                << std::fixed << std::setprecision(4) << result.gflops << ","
                << std::fixed << std::setprecision(6) << result.time_ms << ","
                << result.iterations << "\n";
        }

        csv.close();
        std::cout << "[CSV] Exported " << results.size() << " results to " << filename << "\n";
    }

    int device_id_ = 0;
    cudaDeviceProp device_props_;
    cudaStream_t stream_ = nullptr;
    cudaEvent_t start_event_ = nullptr;
    cudaEvent_t stop_event_ = nullptr;

    float *test_A_device_ = nullptr;
    IQ4_NLBlock *test_B_device_ = nullptr;
    float *test_C_device_ = nullptr;

    int allocated_m_ = 0;
    int allocated_n_ = 0;
    int allocated_k_ = 0;

    int warmup_iterations_ = 2;
    int benchmark_iterations_ = 5;
};

// ============================================================================
// Test Cases: Common Matrix Shapes
// ============================================================================

/**
 * @brief Test: 0.5B model single-token decode (Q/K/V projections)
 * Shape: [1, 896, 896] - Most common decode operation
 */
TEST_F(CudaGemmHeuristicValidation, Qwen_0_5B_SingleToken_QKV)
{
    const int m = 1;
    const int n = 896;
    const int k = 896;

    allocateTestData(m, n, k);

    // Get all valid configurations
    auto &tuner = CudaGemmAutoTuner::instance();
    auto all_configs = tuner.getAvailableConfigs();

    std::cout << "[INFO] Benchmarking " << all_configs.size() << " configurations for shape ["
              << m << ", " << n << ", " << k << "]...\n";

    // Benchmark ALL configs
    std::vector<CudaBenchmarkResult> all_results;
    int successful = 0;
    int failed = 0;
    for (size_t i = 0; i < all_configs.size(); ++i)
    {
        if (i % 100 == 0)
        {
            std::cout << "[PROGRESS] " << i << "/" << all_configs.size() << " configs tested ("
                      << successful << " successful, " << failed << " failed)\n";
        }

        auto result = benchmarkConfig(all_configs[i], m, n, k);
        if (result.gflops > 0.0)
        {
            all_results.push_back(result);
            successful++;
        }
        else
        {
            failed++;
        }
    }

    std::cout << "[SUMMARY] Tested " << all_configs.size() << " configs: "
              << successful << " successful, " << failed << " failed\n";

    // Export to CSV
    exportToCSV("cuda_gemm_benchmark_data.csv", "Qwen_0_5B_SingleToken_QKV", m, n, k, all_results);

    // Sort by performance
    std::sort(all_results.begin(), all_results.end());

    // Rank by heuristic
    auto heuristic_ranking = rankByHeuristic(all_configs, m, n, k);

    // Print comparison
    printComparisonTable("0.5B Model - Single Token Decode (Q/K/V)",
                         all_results, heuristic_ranking, m, n, k);

    // Compute correlation
    double correlation = computeRankCorrelation(all_results, heuristic_ranking);
    std::cout << "\n[METRIC] Rank correlation (Spearman's rho): " << std::fixed << std::setprecision(3)
              << correlation << "\n";
    std::cout << "[METRIC] Best empirical: " << all_results[0].gflops << " GFLOPS ("
              << all_results[0].config.id() << ")\n";
    std::cout << "[METRIC] Heuristic #1: " << heuristic_ranking[0].id() << "\n";

    // Validation: Heuristic #1 should be in empirical top-10
    bool found_in_top10 = false;
    for (int i = 0; i < 10 && i < static_cast<int>(all_results.size()); ++i)
    {
        if (all_results[i].config.id() == heuristic_ranking[0].id())
        {
            found_in_top10 = true;
            std::cout << "[RESULT] ✅ Heuristic #1 found at empirical rank " << (i + 1) << "\n\n";
            break;
        }
    }

    if (!found_in_top10)
    {
        std::cout << "[RESULT] ❌ Heuristic #1 NOT in empirical top-10\n\n";
    }

    EXPECT_TRUE(found_in_top10) << "Heuristic should select config in empirical top-10";
    EXPECT_GT(correlation, 0.3) << "Correlation should be positive (better than random)";
}

/**
 * @brief Test: 0.5B model batch prefill (batch=32)
 * Shape: [32, 896, 896] - Batch inference scenario
 */
TEST_F(CudaGemmHeuristicValidation, Qwen_0_5B_Batch32_QKV)
{
    const int m = 32;
    const int n = 896;
    const int k = 896;

    allocateTestData(m, n, k);

    auto &tuner = CudaGemmAutoTuner::instance();
    auto all_configs = tuner.getAvailableConfigs();

    std::cout << "[INFO] Benchmarking " << all_configs.size() << " configurations...\n";

    std::vector<CudaBenchmarkResult> all_results;
    for (const auto &config : all_configs)
    {
        auto result = benchmarkConfig(config, m, n, k);
        if (result.gflops > 0.0)
            all_results.push_back(result);
    }

    // Export to CSV
    exportToCSV("cuda_gemm_benchmark_data.csv", "Qwen_0_5B_Batch32_QKV", m, n, k, all_results);

    std::sort(all_results.begin(), all_results.end());

    auto heuristic_ranking = rankByHeuristic(all_configs, m, n, k);

    printComparisonTable("0.5B Model - Batch=32 Prefill (Q/K/V)",
                         all_results, heuristic_ranking, m, n, k);

    double correlation = computeRankCorrelation(all_results, heuristic_ranking);
    std::cout << "\n[METRIC] Rank correlation: " << correlation << "\n";
    std::cout << "[METRIC] Best empirical: " << all_results[0].gflops << " GFLOPS\n\n";

    EXPECT_GT(correlation, 0.3);
}

/**
 * @brief Test: 0.5B model FFN gate projection
 * Shape: [1, 4864, 896] - Wide matrix (d_ff > d_model)
 */
TEST_F(CudaGemmHeuristicValidation, Qwen_0_5B_FFN_Gate)
{
    const int m = 1;
    const int n = 4864;
    const int k = 896;

    allocateTestData(m, n, k);

    auto &tuner = CudaGemmAutoTuner::instance();
    auto all_configs = tuner.getAvailableConfigs();

    std::vector<CudaBenchmarkResult> all_results;
    for (const auto &config : all_configs)
    {
        auto result = benchmarkConfig(config, m, n, k);
        if (result.gflops > 0.0)
            all_results.push_back(result);
    }
    exportToCSV("cuda_gemm_benchmark_data.csv", "Qwen_0_5B_FFN_Gate", m, n, k, all_results);
    std::sort(all_results.begin(), all_results.end());

    auto heuristic_ranking = rankByHeuristic(all_configs, m, n, k);

    printComparisonTable("0.5B Model - FFN Gate Projection",
                         all_results, heuristic_ranking, m, n, k);

    double correlation = computeRankCorrelation(all_results, heuristic_ranking);
    std::cout << "\n[METRIC] Rank correlation: " << correlation << "\n";
    std::cout << "[METRIC] Best empirical: " << all_results[0].gflops << " GFLOPS\n\n";

    EXPECT_GT(correlation, 0.3);
}

/**
 * @brief Test: 4B model single-token decode
 * Shape: [1, 2560, 2560] - Larger model
 */
TEST_F(CudaGemmHeuristicValidation, Qwen_4B_SingleToken_QKV)
{
    const int m = 1;
    const int n = 2560;
    const int k = 2560;

    allocateTestData(m, n, k);

    auto &tuner = CudaGemmAutoTuner::instance();
    auto all_configs = tuner.getAvailableConfigs();

    std::vector<CudaBenchmarkResult> all_results;
    for (const auto &config : all_configs)
    {
        auto result = benchmarkConfig(config, m, n, k);
        if (result.gflops > 0.0)
            all_results.push_back(result);
    }
    exportToCSV("cuda_gemm_benchmark_data.csv", "Qwen_4B_SingleToken_QKV", m, n, k, all_results);
    std::sort(all_results.begin(), all_results.end());

    auto heuristic_ranking = rankByHeuristic(all_configs, m, n, k);

    printComparisonTable("4B Model - Single Token Decode (Q/K/V)",
                         all_results, heuristic_ranking, m, n, k);

    double correlation = computeRankCorrelation(all_results, heuristic_ranking);
    std::cout << "\n[METRIC] Rank correlation: " << correlation << "\n";
    std::cout << "[METRIC] Best empirical: " << all_results[0].gflops << " GFLOPS\n\n";

    EXPECT_GT(correlation, 0.3);
}

/**
 * @brief Test: 4B model batch prefill
 * Shape: [128, 2560, 2560] - Large batch
 */
TEST_F(CudaGemmHeuristicValidation, Qwen_4B_Batch128_QKV)
{
    const int m = 128;
    const int n = 2560;
    const int k = 2560;

    allocateTestData(m, n, k);

    auto &tuner = CudaGemmAutoTuner::instance();
    auto all_configs = tuner.getAvailableConfigs();

    std::vector<CudaBenchmarkResult> all_results;
    for (const auto &config : all_configs)
    {
        auto result = benchmarkConfig(config, m, n, k);
        if (result.gflops > 0.0)
            all_results.push_back(result);
    }
    exportToCSV("cuda_gemm_benchmark_data.csv", "Qwen_4B_Batch128_QKV", m, n, k, all_results);
    std::sort(all_results.begin(), all_results.end());

    auto heuristic_ranking = rankByHeuristic(all_configs, m, n, k);

    printComparisonTable("4B Model - Batch=128 Prefill (Q/K/V)",
                         all_results, heuristic_ranking, m, n, k);

    double correlation = computeRankCorrelation(all_results, heuristic_ranking);
    std::cout << "\n[METRIC] Rank correlation: " << correlation << "\n";
    std::cout << "[METRIC] Best empirical: " << all_results[0].gflops << " GFLOPS\n\n";

    EXPECT_GT(correlation, 0.3);
}

/**
 * @brief Test: 4B model FFN down projection
 * Shape: [1, 2560, 13824] - Tall matrix (d_ff >> d_model)
 */
TEST_F(CudaGemmHeuristicValidation, Qwen_4B_FFN_Down)
{
    const int m = 1;
    const int n = 2560;
    const int k = 13824;

    allocateTestData(m, n, k);

    auto &tuner = CudaGemmAutoTuner::instance();
    auto all_configs = tuner.getAvailableConfigs();

    std::vector<CudaBenchmarkResult> all_results;
    int successful = 0, failed = 0;
    for (size_t i = 0; i < all_configs.size(); ++i)
    {
        if (i % 100 == 0)
        {
            std::cout << "[PROGRESS] " << i << "/" << all_configs.size() << " configs\n";
        }
        auto result = benchmarkConfig(all_configs[i], m, n, k);
        if (result.gflops > 0.0)
        {
            all_results.push_back(result);
            successful++;
        }
        else
        {
            failed++;
        }
    }
    std::cout << "[SUMMARY] " << successful << " successful, " << failed << " failed\n";

    exportToCSV("cuda_gemm_benchmark_data.csv", "Qwen_4B_FFN_Down", m, n, k, all_results);
    std::sort(all_results.begin(), all_results.end());

    auto heuristic_ranking = rankByHeuristic(all_configs, m, n, k);

    printComparisonTable("4B Model - FFN Down Projection",
                         all_results, heuristic_ranking, m, n, k);

    double correlation = computeRankCorrelation(all_results, heuristic_ranking);
    std::cout << "\n[METRIC] Rank correlation: " << correlation << "\n";
    std::cout << "[METRIC] Best empirical: " << all_results[0].gflops << " GFLOPS\n\n";

    EXPECT_GT(correlation, 0.3);
}

/**
 * @brief Test: 7B model single-token decode
 * Shape: [1, 4096, 4096] - Large model Q/K/V
 */
TEST_F(CudaGemmHeuristicValidation, Qwen_7B_SingleToken_QKV)
{
    const int m = 1;
    const int n = 4096;
    const int k = 4096;

    allocateTestData(m, n, k);

    auto &tuner = CudaGemmAutoTuner::instance();
    auto all_configs = tuner.getAvailableConfigs();

    std::cout << "[INFO] Testing 7B model single-token decode [" << m << ", " << n << ", " << k << "]\n";

    std::vector<CudaBenchmarkResult> all_results;
    int successful = 0, failed = 0;
    for (size_t i = 0; i < all_configs.size(); ++i)
    {
        if (i % 100 == 0)
        {
            std::cout << "[PROGRESS] " << i << "/" << all_configs.size() << " configs\n";
        }
        auto result = benchmarkConfig(all_configs[i], m, n, k);
        if (result.gflops > 0.0)
        {
            all_results.push_back(result);
            successful++;
        }
        else
        {
            failed++;
        }
    }
    std::cout << "[SUMMARY] " << successful << " successful, " << failed << " failed\n";

    exportToCSV("cuda_gemm_benchmark_data.csv", "Qwen_7B_SingleToken_QKV", m, n, k, all_results);
    std::sort(all_results.begin(), all_results.end());

    auto heuristic_ranking = rankByHeuristic(all_configs, m, n, k);

    printComparisonTable("7B Model - Single Token Decode (Q/K/V)",
                         all_results, heuristic_ranking, m, n, k);

    double correlation = computeRankCorrelation(all_results, heuristic_ranking);
    std::cout << "\n[METRIC] Rank correlation: " << correlation << "\n";
    std::cout << "[METRIC] Best empirical: " << all_results[0].gflops << " GFLOPS\n\n";

    EXPECT_GT(correlation, 0.3);
}

/**
 * @brief Test: 7B model batch prefill
 * Shape: [128, 4096, 4096] - Large batch, large model
 */
TEST_F(CudaGemmHeuristicValidation, Qwen_7B_Batch128_QKV)
{
    const int m = 128;
    const int n = 4096;
    const int k = 4096;

    allocateTestData(m, n, k);

    auto &tuner = CudaGemmAutoTuner::instance();
    auto all_configs = tuner.getAvailableConfigs();

    std::cout << "[INFO] Testing 7B model batch=128 prefill [" << m << ", " << n << ", " << k << "]\n";

    std::vector<CudaBenchmarkResult> all_results;
    int successful = 0, failed = 0;
    for (size_t i = 0; i < all_configs.size(); ++i)
    {
        if (i % 100 == 0)
        {
            std::cout << "[PROGRESS] " << i << "/" << all_configs.size() << " configs\n";
        }
        auto result = benchmarkConfig(all_configs[i], m, n, k);
        if (result.gflops > 0.0)
        {
            all_results.push_back(result);
            successful++;
        }
        else
        {
            failed++;
        }
    }
    std::cout << "[SUMMARY] " << successful << " successful, " << failed << " failed\n";

    exportToCSV("cuda_gemm_benchmark_data.csv", "Qwen_7B_Batch128_QKV", m, n, k, all_results);
    std::sort(all_results.begin(), all_results.end());

    auto heuristic_ranking = rankByHeuristic(all_configs, m, n, k);

    printComparisonTable("7B Model - Batch=128 Prefill (Q/K/V)",
                         all_results, heuristic_ranking, m, n, k);

    double correlation = computeRankCorrelation(all_results, heuristic_ranking);
    std::cout << "\n[METRIC] Rank correlation: " << correlation << "\n";
    std::cout << "[METRIC] Best empirical: " << all_results[0].gflops << " GFLOPS\n\n";

    EXPECT_GT(correlation, 0.3);
}

/**
 * @brief Test: 7B model FFN gate projection
 * Shape: [1, 22016, 4096] - Wide matrix
 */
TEST_F(CudaGemmHeuristicValidation, Qwen_7B_FFN_Gate)
{
    const int m = 1;
    const int n = 22016;
    const int k = 4096;

    allocateTestData(m, n, k);

    auto &tuner = CudaGemmAutoTuner::instance();
    auto all_configs = tuner.getAvailableConfigs();

    std::cout << "[INFO] Testing 7B model FFN gate [" << m << ", " << n << ", " << k << "]\n";

    std::vector<CudaBenchmarkResult> all_results;
    int successful = 0, failed = 0;
    for (size_t i = 0; i < all_configs.size(); ++i)
    {
        if (i % 100 == 0)
        {
            std::cout << "[PROGRESS] " << i << "/" << all_configs.size() << " configs\n";
        }
        auto result = benchmarkConfig(all_configs[i], m, n, k);
        if (result.gflops > 0.0)
        {
            all_results.push_back(result);
            successful++;
        }
        else
        {
            failed++;
        }
    }
    std::cout << "[SUMMARY] " << successful << " successful, " << failed << " failed\n";

    exportToCSV("cuda_gemm_benchmark_data.csv", "Qwen_7B_FFN_Gate", m, n, k, all_results);
    std::sort(all_results.begin(), all_results.end());

    auto heuristic_ranking = rankByHeuristic(all_configs, m, n, k);

    printComparisonTable("7B Model - FFN Gate Projection",
                         all_results, heuristic_ranking, m, n, k);

    double correlation = computeRankCorrelation(all_results, heuristic_ranking);
    std::cout << "\n[METRIC] Rank correlation: " << correlation << "\n";
    std::cout << "[METRIC] Best empirical: " << all_results[0].gflops << " GFLOPS\n\n";

    EXPECT_GT(correlation, 0.3);
}

/**
 * @brief Test: 14B model single-token decode
 * Shape: [1, 5120, 5120] - Largest model Q/K/V
 */
TEST_F(CudaGemmHeuristicValidation, Qwen_14B_SingleToken_QKV)
{
    const int m = 1;
    const int n = 5120;
    const int k = 5120;

    allocateTestData(m, n, k);

    auto &tuner = CudaGemmAutoTuner::instance();
    auto all_configs = tuner.getAvailableConfigs();

    std::cout << "[INFO] Testing 14B model single-token decode [" << m << ", " << n << ", " << k << "]\n";

    std::vector<CudaBenchmarkResult> all_results;
    int successful = 0, failed = 0;
    for (size_t i = 0; i < all_configs.size(); ++i)
    {
        if (i % 100 == 0)
        {
            std::cout << "[PROGRESS] " << i << "/" << all_configs.size() << " configs\n";
        }
        auto result = benchmarkConfig(all_configs[i], m, n, k);
        if (result.gflops > 0.0)
        {
            all_results.push_back(result);
            successful++;
        }
        else
        {
            failed++;
        }
    }
    std::cout << "[SUMMARY] " << successful << " successful, " << failed << " failed\n";

    exportToCSV("cuda_gemm_benchmark_data.csv", "Qwen_14B_SingleToken_QKV", m, n, k, all_results);
    std::sort(all_results.begin(), all_results.end());

    auto heuristic_ranking = rankByHeuristic(all_configs, m, n, k);

    printComparisonTable("14B Model - Single Token Decode (Q/K/V)",
                         all_results, heuristic_ranking, m, n, k);

    double correlation = computeRankCorrelation(all_results, heuristic_ranking);
    std::cout << "\n[METRIC] Rank correlation: " << correlation << "\n";
    std::cout << "[METRIC] Best empirical: " << all_results[0].gflops << " GFLOPS\n\n";

    EXPECT_GT(correlation, 0.3);
}

/**
 * @brief Test: 14B model batch prefill
 * Shape: [256, 5120, 5120] - Very large batch
 */
TEST_F(CudaGemmHeuristicValidation, Qwen_14B_Batch256_QKV)
{
    const int m = 256;
    const int n = 5120;
    const int k = 5120;

    allocateTestData(m, n, k);

    auto &tuner = CudaGemmAutoTuner::instance();
    auto all_configs = tuner.getAvailableConfigs();

    std::cout << "[INFO] Testing 14B model batch=256 prefill [" << m << ", " << n << ", " << k << "]\n";

    std::vector<CudaBenchmarkResult> all_results;
    int successful = 0, failed = 0;
    for (size_t i = 0; i < all_configs.size(); ++i)
    {
        if (i % 100 == 0)
        {
            std::cout << "[PROGRESS] " << i << "/" << all_configs.size() << " configs\n";
        }
        auto result = benchmarkConfig(all_configs[i], m, n, k);
        if (result.gflops > 0.0)
        {
            all_results.push_back(result);
            successful++;
        }
        else
        {
            failed++;
        }
    }
    std::cout << "[SUMMARY] " << successful << " successful, " << failed << " failed\n";

    exportToCSV("cuda_gemm_benchmark_data.csv", "Qwen_14B_Batch256_QKV", m, n, k, all_results);
    std::sort(all_results.begin(), all_results.end());

    auto heuristic_ranking = rankByHeuristic(all_configs, m, n, k);

    printComparisonTable("14B Model - Batch=256 Prefill (Q/K/V)",
                         all_results, heuristic_ranking, m, n, k);

    double correlation = computeRankCorrelation(all_results, heuristic_ranking);
    std::cout << "\n[METRIC] Rank correlation: " << correlation << "\n";
    std::cout << "[METRIC] Best empirical: " << all_results[0].gflops << " GFLOPS\n\n";

    EXPECT_GT(correlation, 0.3);
}

/**
 * @brief Test: 14B model FFN down projection
 * Shape: [1, 5120, 27648] - Very tall matrix
 */
TEST_F(CudaGemmHeuristicValidation, Qwen_14B_FFN_Down)
{
    const int m = 1;
    const int n = 5120;
    const int k = 27648;

    allocateTestData(m, n, k);

    auto &tuner = CudaGemmAutoTuner::instance();
    auto all_configs = tuner.getAvailableConfigs();

    std::cout << "[INFO] Testing 14B model FFN down [" << m << ", " << n << ", " << k << "]\n";

    std::vector<CudaBenchmarkResult> all_results;
    int successful = 0, failed = 0;
    for (size_t i = 0; i < all_configs.size(); ++i)
    {
        if (i % 100 == 0)
        {
            std::cout << "[PROGRESS] " << i << "/" << all_configs.size() << " configs\n";
        }
        auto result = benchmarkConfig(all_configs[i], m, n, k);
        if (result.gflops > 0.0)
        {
            all_results.push_back(result);
            successful++;
        }
        else
        {
            failed++;
        }
    }
    std::cout << "[SUMMARY] " << successful << " successful, " << failed << " failed\n";

    exportToCSV("cuda_gemm_benchmark_data.csv", "Qwen_14B_FFN_Down", m, n, k, all_results);
    std::sort(all_results.begin(), all_results.end());

    auto heuristic_ranking = rankByHeuristic(all_configs, m, n, k);

    printComparisonTable("14B Model - FFN Down Projection",
                         all_results, heuristic_ranking, m, n, k);

    double correlation = computeRankCorrelation(all_results, heuristic_ranking);
    std::cout << "\n[METRIC] Rank correlation: " << correlation << "\n";
    std::cout << "[METRIC] Best empirical: " << all_results[0].gflops << " GFLOPS\n\n";

    EXPECT_GT(correlation, 0.3);
}
