/**
 * @file Perf__Tile64x32Comparison.cpp
 * @brief Compare 64×32 tiles vs auto-selected tiles on 1024 tokens
 *
 * Tests whether V1's optimal 64×32 configuration performs better than
 * the smart search's auto-selected tile size for 1024 token workload.
 *
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include "tensors/Tensors.h"
#include "loaders/ModelLoader.h"
#include "kernels/cpu/gemm/GemmAutoTuner.h"
#include "utils/Logger.h"
#include <chrono>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>

using namespace llaminar2;
using namespace llaminar::v2::kernels;

/**
 * @brief Benchmark a specific GEMM configuration
 */
double benchmark_config(
    const std::shared_ptr<IQ4_NLTensor> &weight,
    const GemmKernelConfig &config,
    int m, int n, int k,
    int warmup_iters = 3,
    int benchmark_iters = 100,
    int trials = 10)
{
    // Allocate test data
    auto activation = std::make_shared<llaminar2::FP32Tensor>(std::vector<size_t>{
        static_cast<size_t>(m),
        static_cast<size_t>(k)});
    auto output = std::make_shared<llaminar2::FP32Tensor>(std::vector<size_t>{
        static_cast<size_t>(m),
        static_cast<size_t>(n)});

    // Fill activation with random data
    float *act_data = activation->mutable_data();
    for (size_t i = 0; i < m * k; ++i)
    {
        act_data[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Create GEMM with specific variant
    auto &tuner = GemmAutoTuner::instance();
    auto variant = tuner.createVariant(config, weight.get());

    if (!variant)
    {
        LOG_ERROR("Failed to create variant: " << config.id());
        return 0.0;
    }

    // Warmup
    for (int i = 0; i < warmup_iters; ++i)
    {
        variant->multiply(act_data, output->mutable_data(), m, n, k, weight.get(), false, 1.0f, 0.0f);
    }

    // Timed runs
    std::vector<double> times;
    times.reserve(trials);

    for (int trial = 0; trial < trials; ++trial)
    {
        auto start = std::chrono::high_resolution_clock::now();

        for (int iter = 0; iter < benchmark_iters; ++iter)
        {
            variant->multiply(act_data, output->mutable_data(), m, n, k, weight.get(), false, 1.0f, 0.0f);
        }

        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count() / benchmark_iters;
        times.push_back(ms);
    }

    // Return best time (GFLOPS)
    double min_time = *std::min_element(times.begin(), times.end());
    double flops = 2.0 * m * n * k;
    return flops / (min_time * 1e6);
}

TEST(Tile64x32Comparison, Compare1024Tokens)
{
    // Load model to get real IQ4_NL weights
    ModelLoader loader;
    ASSERT_TRUE(loader.loadModel("models/qwen2.5-0.5b-instruct-iq4_nl.gguf"))
        << "Failed to load model";

    // Get Q projection weights (896×896, IQ4_NL quantized)
    auto q_proj_weight = loader.loadTensor("blk.0.attn_q.weight", -1); // -1 = CPU
    ASSERT_NE(q_proj_weight, nullptr) << "Failed to get Q projection weights";

    auto iq4nl_tensor = std::dynamic_pointer_cast<IQ4_NLTensor>(q_proj_weight);
    ASSERT_NE(iq4nl_tensor, nullptr) << "Q projection is not IQ4_NL format";

    const int m = 1024; // 1024 tokens
    const int n = 896;  // Output features
    const int k = 896;  // Input features

    std::cout << "\n╔═══════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  64×32 Tile Size Comparison (1024 tokens, 896×896 matrix)        ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════════╝\n\n";

    std::cout << "Matrix dimensions: " << m << " × " << n << " × " << k << "\n";
    std::cout << "Total FLOPs per multiply: " << (2.0 * m * n * k / 1e9) << " billion\n\n";

    // Test 1: Auto-selected configuration (from benchmark output: tile16×64)
    std::cout << "Testing AUTO-SELECTED configuration (16×64)...\n";
    GemmKernelConfig auto_config;
    auto_config.unroll_factor = 16;
    auto_config.prefetch_blocks = 5;
    auto_config.tile_m = 16;
    auto_config.tile_n = 64;

    double auto_gflops = benchmark_config(iq4nl_tensor, auto_config, m, n, k);
    std::cout << "  Result: " << std::fixed << std::setprecision(2) << auto_gflops << " GFLOPS\n\n";

    // Test 2: Try all 64×32 variants
    std::cout << "Testing all 64×32 variants...\n";
    std::vector<std::tuple<int, int, double>> results; // unroll, prefetch, gflops

    for (int unroll : {4, 8, 16})
    {
        for (int prefetch : {3, 5})
        {
            GemmKernelConfig cfg;
            cfg.unroll_factor = unroll;
            cfg.prefetch_blocks = prefetch;
            cfg.tile_m = 64;
            cfg.tile_n = 32;

            double gflops = benchmark_config(iq4nl_tensor, cfg, m, n, k);
            results.push_back({unroll, prefetch, gflops});

            std::cout << "  64×32 (unroll=" << unroll << ", prefetch=" << prefetch << "): "
                      << std::fixed << std::setprecision(2) << gflops << " GFLOPS\n";
        }
    }

    // Find best 64×32
    auto best_64x32 = std::max_element(results.begin(), results.end(),
                                       [](const auto &a, const auto &b)
                                       { return std::get<2>(a) < std::get<2>(b); });

    double best_64x32_gflops = std::get<2>(*best_64x32);
    int best_unroll = std::get<0>(*best_64x32);
    int best_prefetch = std::get<1>(*best_64x32);

    std::cout << "\n╔═══════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                          RESULTS SUMMARY                          ║\n";
    std::cout << "╠═══════════════════════════════════════════════════════════════════╣\n";
    std::cout << "║ Auto-Selected (16×64, u16, p5):                                   ║\n";
    std::cout << "║   " << std::fixed << std::setprecision(2) << std::setw(6) << auto_gflops
              << " GFLOPS" << std::setw(47) << " " << " ║\n";
    std::cout << "╠═══════════════════════════════════════════════════════════════════╣\n";
    std::cout << "║ Best 64×32 (u" << best_unroll << ", p" << best_prefetch << "):                                         ║\n";
    std::cout << "║   " << std::fixed << std::setprecision(2) << std::setw(6) << best_64x32_gflops
              << " GFLOPS" << std::setw(47) << " " << " ║\n";
    std::cout << "╠═══════════════════════════════════════════════════════════════════╣\n";

    if (best_64x32_gflops > auto_gflops)
    {
        double improvement = (best_64x32_gflops / auto_gflops - 1.0) * 100.0;
        std::cout << "║ WINNER: 64×32 is " << std::fixed << std::setprecision(1) << improvement
                  << "% FASTER!";
        std::cout << std::setw(35 - std::to_string((int)improvement).length()) << " " << " ║\n";
    }
    else
    {
        double improvement = (auto_gflops / best_64x32_gflops - 1.0) * 100.0;
        std::cout << "║ WINNER: Auto-selected (16×64) is " << std::fixed << std::setprecision(1)
                  << improvement << "% FASTER!";
        std::cout << std::setw(21 - std::to_string((int)improvement).length()) << " " << " ║\n";
    }
    std::cout << "╚═══════════════════════════════════════════════════════════════════╝\n\n";

    // Test passes regardless (this is for information)
    SUCCEED();
}
