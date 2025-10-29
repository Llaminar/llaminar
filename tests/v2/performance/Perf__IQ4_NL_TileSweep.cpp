/**
 * @file Perf__IQ4_NL_TileSweep.cpp
 * @brief Tile size sweep benchmark for IQ4_NL GEMM optimization
 *
 * This test sweeps through different tile sizes to find optimal configuration
 * for cache utilization. Measures:
 *   - L1/L2/L3 cache miss rates (via perf counters)
 *   - Throughput (GFLOPS)
 *   - Memory bandwidth
 *   - IPC (instructions per cycle)
 *
 * Usage:
 *   perf stat -e L1-dcache-load-misses,L1-dcache-loads,LLC-load-misses,LLC-loads \
 *     ./build_v2_release/performance/v2_perf_iq4nl_tile_sweep
 *
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <vector>
#include <cmath>
#include <algorithm>

// V2 includes
#include "tensors/Tensors.h"
#include "loaders/ModelLoader.h"
// Note: No longer needs QuantizedGemm.h - uses ITensorGemm interface

using namespace llaminar2;

/**
 * @brief Tile configuration for cache optimization experiments
 */
struct TileConfig
{
    int mc; ///< Tile size for M dimension (rows, outer product)
    int kc; ///< Tile size for K dimension (reduction)
    int nc; ///< Tile size for N dimension (columns)
    std::string description;

    // Estimate working set size in KB
    double working_set_kb() const
    {
        // A tile: mc x kc (FP32 = 4 bytes)
        // B tile: kc x nc (IQ4_NL ~0.56 bytes per element)
        // C tile: mc x nc (FP32 = 4 bytes)
        double a_kb = (mc * kc * 4.0) / 1024.0;
        double b_kb = (kc * nc * 0.56) / 1024.0;
        double c_kb = (mc * nc * 4.0) / 1024.0;
        return a_kb + b_kb + c_kb;
    }
};

/**
 * @brief Performance metrics for a single tile configuration
 */
struct TilePerformance
{
    TileConfig config;
    double mean_ms;
    double stddev_ms;
    double mean_gflops;
    double stddev_gflops;
    double cv_percent; // Coefficient of variation
};

/**
 * @brief IQ4_NL Tile Sweep Test Fixture
 */
class IQ4_NL_TileSweep : public ::testing::Test
{
protected:
    int rank_ = 0;
    int world_size_ = 1;
    std::unique_ptr<ModelLoader> loader_;

    void SetUp() override
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size_);

        // Load model
        std::string model_path = "models/qwen2.5-0.5b-instruct-iq4_nl.gguf";

        try
        {
            loader_ = std::make_unique<ModelLoader>();

            if (rank_ == 0)
            {
                std::cout << "[Tile Sweep] Loading model: " << model_path << std::endl;
            }

            if (!loader_->loadModel(model_path))
            {
                throw std::runtime_error("Failed to load model");
            }

            if (rank_ == 0)
            {
                std::cout << "[Tile Sweep] Model loaded successfully" << std::endl;
            }
        }
        catch (const std::exception &e)
        {
            FAIL() << "Model loading failed: " << e.what();
        }
    }

    /**
     * @brief Get a real IQ4_NL weight tensor for testing
     */
    std::shared_ptr<IQ4_NLTensor> getWeightTensor()
    {
        // Get blk.0.attn_q.weight (896 x 896)
        auto weight_base = loader_->loadTensor("blk.0.attn_q.weight", -1); // -1 = CPU
        EXPECT_NE(weight_base, nullptr) << "Failed to load weight tensor";

        auto weight = std::dynamic_pointer_cast<IQ4_NLTensor>(weight_base);
        EXPECT_NE(weight, nullptr) << "Weight is not IQ4_NL tensor";

        return weight;
    }

    /**
     * @brief Create FP32 activation tensor with random data
     */
    std::shared_ptr<FP32Tensor> createFP32Activation(int seq_len, int features)
    {
        auto activation = std::make_shared<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(features)});

        // Initialize with random data in [-1, 1]
        auto data = activation->mutable_data();
        for (size_t i = 0; i < seq_len * features; ++i)
        {
            data[i] = (static_cast<float>(rand()) / RAND_MAX) * 2.0f - 1.0f;
        }

        return activation;
    }

    /**
     * @brief Benchmark a specific tile configuration
     */
    TilePerformance benchmarkTileConfig(const TileConfig &config, int seq_len, int features, int trials = 3)
    {
        auto weight = getWeightTensor();
        auto activation = createFP32Activation(seq_len, features);
        auto output = std::make_shared<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(features)});

        auto gemm = weight->createGemm();

        // Warmup
        for (int i = 0; i < 3; ++i)
        {
            gemm->multiply(activation->data(), output->mutable_data(),
                           seq_len, features, features,
                           true, 1.0f, 0.0f, nullptr, -1);
        }

        // Benchmark trials
        std::vector<double> trial_times_ms;
        trial_times_ms.reserve(trials);

        for (int trial = 0; trial < trials; ++trial)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            auto start = std::chrono::high_resolution_clock::now();

            // Run 50 iterations per trial
            for (int i = 0; i < 50; ++i)
            {
                gemm->multiply(activation->data(), output->mutable_data(),
                               seq_len, features, features,
                               true, 1.0f, 0.0f, nullptr, -1);
            }

            MPI_Barrier(MPI_COMM_WORLD);
            auto end = std::chrono::high_resolution_clock::now();

            double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
            trial_times_ms.push_back(elapsed_ms / 50.0);
        }

        // Calculate statistics
        TilePerformance perf;
        perf.config = config;

        double sum = 0.0;
        for (double t : trial_times_ms)
            sum += t;
        perf.mean_ms = sum / trial_times_ms.size();

        double sq_diff_sum = 0.0;
        for (double t : trial_times_ms)
        {
            double diff = t - perf.mean_ms;
            sq_diff_sum += diff * diff;
        }
        perf.stddev_ms = std::sqrt(sq_diff_sum / trial_times_ms.size());

        double flops = 2.0 * seq_len * features * features;
        perf.mean_gflops = (flops / perf.mean_ms) / 1e6;
        perf.stddev_gflops = (flops / 1e6) * perf.stddev_ms / (perf.mean_ms * perf.mean_ms);
        perf.cv_percent = (perf.stddev_ms / perf.mean_ms) * 100.0;

        return perf;
    }

    /**
     * @brief Print results table
     */
    void printResults(const std::vector<TilePerformance> &results)
    {
        if (rank_ != 0)
            return;

        std::cout << "\n";
        std::cout << "╔════════════════════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║ IQ4_NL GEMM Tile Size Sweep Results (512x896x896)                             ║\n";
        std::cout << "╠════════════════════════════════════════════════════════════════════════════════╣\n";
        std::cout << "║  MC   KC   NC  │ WS(KB) │  Time(ms)  │ GFLOPS  │  CV%  │ Description         ║\n";
        std::cout << "╟────────────────┼────────┼────────────┼─────────┼───────┼─────────────────────╢\n";

        for (const auto &perf : results)
        {
            printf("║ %4d %4d %4d │ %6.1f │ %5.2f±%-4.2f │ %7.2f │ %5.1f │ %-19s ║\n",
                   perf.config.mc, perf.config.kc, perf.config.nc,
                   perf.config.working_set_kb(),
                   perf.mean_ms, perf.stddev_ms,
                   perf.mean_gflops,
                   perf.cv_percent,
                   perf.config.description.c_str());
        }

        std::cout << "╚════════════════════════════════════════════════════════════════════════════════╝\n";

        // Find best configuration
        auto best = std::max_element(results.begin(), results.end(),
                                     [](const TilePerformance &a, const TilePerformance &b)
                                     {
                                         return a.mean_gflops < b.mean_gflops;
                                     });

        if (best != results.end())
        {
            std::cout << "\n🏆 Best Configuration:\n";
            std::cout << "   MC=" << best->config.mc << ", KC=" << best->config.kc
                      << ", NC=" << best->config.nc << "\n";
            std::cout << "   Performance: " << std::fixed << std::setprecision(2)
                      << best->mean_gflops << " GFLOPS\n";
            std::cout << "   Working Set: " << std::fixed << std::setprecision(1)
                      << best->config.working_set_kb() << " KB\n\n";
        }
    }

    /**
     * @brief Export results to CSV for analysis
     */
    void exportCSV(const std::vector<TilePerformance> &results, const std::string &filename)
    {
        if (rank_ != 0)
            return;

        std::ofstream csv(filename);
        csv << "mc,kc,nc,working_set_kb,mean_ms,stddev_ms,mean_gflops,stddev_gflops,cv_percent,description\n";

        for (const auto &perf : results)
        {
            csv << perf.config.mc << ","
                << perf.config.kc << ","
                << perf.config.nc << ","
                << perf.config.working_set_kb() << ","
                << perf.mean_ms << ","
                << perf.stddev_ms << ","
                << perf.mean_gflops << ","
                << perf.stddev_gflops << ","
                << perf.cv_percent << ","
                << perf.config.description << "\n";
        }

        std::cout << "Results exported to: " << filename << "\n";
    }
};

/**
 * @brief L1 Cache Optimization Sweep (32KB typical)
 *
 * Test small tiles that fit in L1 data cache.
 * Typical L1D: 32KB per core
 */
TEST_F(IQ4_NL_TileSweep, L1_CacheOptimization)
{
    std::vector<TileConfig> configs = {
        {32, 32, 32, "Tiny (3KB)"},
        {64, 64, 64, "Small (18KB)"},
        {96, 96, 96, "Medium (40KB - L1 spill)"},
        {128, 64, 64, "M-biased (28KB)"},
        {64, 128, 64, "K-biased (26KB)"},
        {64, 64, 128, "N-biased (26KB)"},
    };

    std::vector<TilePerformance> results;
    for (const auto &config : configs)
    {
        if (rank_ == 0)
        {
            std::cout << "\nTesting: " << config.description
                      << " (MC=" << config.mc << ", KC=" << config.kc
                      << ", NC=" << config.nc << ", WS=" << std::fixed
                      << std::setprecision(1) << config.working_set_kb() << "KB)\n";
        }

        auto perf = benchmarkTileConfig(config, 512, 896);
        results.push_back(perf);
    }

    printResults(results);
    exportCSV(results, "tile_sweep_l1.csv");

    MPI_Barrier(MPI_COMM_WORLD);
}

/**
 * @brief L2 Cache Optimization Sweep (256KB-512KB typical)
 *
 * Test medium tiles that fit in L2 cache.
 * Typical L2: 256-512KB per core
 */
TEST_F(IQ4_NL_TileSweep, L2_CacheOptimization)
{
    std::vector<TileConfig> configs = {
        {128, 128, 128, "Base (82KB)"},
        {192, 192, 192, "Medium (184KB)"},
        {256, 256, 128, "Large M (263KB)"},
        {128, 256, 256, "Large K/N (263KB)"},
        {256, 128, 256, "Balanced (263KB)"},
        {384, 192, 192, "Extra Large (369KB)"},
        {512, 256, 128, "L2 Max (525KB - may spill)"},
    };

    std::vector<TilePerformance> results;
    for (const auto &config : configs)
    {
        if (rank_ == 0)
        {
            std::cout << "\nTesting: " << config.description
                      << " (MC=" << config.mc << ", KC=" << config.kc
                      << ", NC=" << config.nc << ", WS=" << std::fixed
                      << std::setprecision(1) << config.working_set_kb() << "KB)\n";
        }

        auto perf = benchmarkTileConfig(config, 512, 896);
        results.push_back(perf);
    }

    printResults(results);
    exportCSV(results, "tile_sweep_l2.csv");

    MPI_Barrier(MPI_COMM_WORLD);
}

/**
 * @brief L3/LLC Optimization Sweep (Large tiles, 8MB-32MB typical)
 *
 * Test large tiles for L3/LLC optimization.
 * May show diminishing returns or performance degradation.
 */
TEST_F(IQ4_NL_TileSweep, L3_CacheOptimization)
{
    std::vector<TileConfig> configs = {
        {512, 512, 256, "L3 Small (1.05MB)"},
        {512, 512, 512, "L3 Medium (2.10MB)"},
        {896, 448, 448, "Model-aligned (2.53MB)"},
        {1024, 512, 512, "L3 Large (4.19MB)"},
        {512, 896, 896, "Full-K (6.46MB)"},
        {896, 896, 512, "Full-M/K (6.46MB)"},
    };

    std::vector<TilePerformance> results;
    for (const auto &config : configs)
    {
        if (rank_ == 0)
        {
            std::cout << "\nTesting: " << config.description
                      << " (MC=" << config.mc << ", KC=" << config.kc
                      << ", NC=" << config.nc << ", WS=" << std::fixed
                      << std::setprecision(1) << config.working_set_kb() << "KB)\n";
        }

        auto perf = benchmarkTileConfig(config, 512, 896);
        results.push_back(perf);
    }

    printResults(results);
    exportCSV(results, "tile_sweep_l3.csv");

    MPI_Barrier(MPI_COMM_WORLD);
}

/**
 * @brief Comprehensive sweep across all cache levels
 *
 * This test runs a comprehensive sweep to identify the optimal tile size
 * for the 512x896x896 workload (LargeBatch_Prefill configuration).
 */
TEST_F(IQ4_NL_TileSweep, ComprehensiveSweep)
{
    if (rank_ == 0)
    {
        std::cout << "\n";
        std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║ Comprehensive Tile Size Sweep                                  ║\n";
        std::cout << "║ Workload: 512x896x896 (LargeBatch_Prefill)                     ║\n";
        std::cout << "╚════════════════════════════════════════════════════════════════╝\n";
    }

    std::vector<TileConfig> configs = {
        // L1 candidates
        {64, 64, 64, "L1-Tiny"},
        {96, 96, 96, "L1-Small"},

        // L2 candidates (sweet spot expected here)
        {128, 128, 128, "L2-Base"},
        {192, 192, 192, "L2-Medium"},
        {256, 128, 256, "L2-Balanced"},
        {256, 256, 128, "L2-M-heavy"},
        {128, 256, 256, "L2-K/N-heavy"},

        // L3 candidates
        {384, 384, 192, "L3-Small"},
        {512, 512, 256, "L3-Medium"},
        {512, 512, 512, "L3-Large"},
        {896, 448, 448, "L3-Model-aligned"},

        // Edge cases
        {1024, 256, 256, "Large-M"},
        {256, 896, 256, "Large-K"},
        {256, 256, 896, "Large-N"},
    };

    std::vector<TilePerformance> results;
    for (const auto &config : configs)
    {
        if (rank_ == 0)
        {
            std::cout << "\nTesting: " << config.description
                      << " (MC=" << config.mc << ", KC=" << config.kc
                      << ", NC=" << config.nc << ", WS=" << std::fixed
                      << std::setprecision(1) << config.working_set_kb() << "KB)\n";
        }

        auto perf = benchmarkTileConfig(config, 512, 896);
        results.push_back(perf);
    }

    printResults(results);
    exportCSV(results, "tile_sweep_comprehensive.csv");

    MPI_Barrier(MPI_COMM_WORLD);
}

// Main function
int main(int argc, char **argv)
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();

    MPI_Finalize();
    return result;
}
