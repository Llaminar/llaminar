/**
 * @file Perf__SimpleInt8Gemm_FFNDown4096.cpp
 * @brief Focused benchmark for FFN_down prefill-4096 optimization
 *
 * Single high-throughput test case:
 * - Operation: FFN_down (Qwen 0.5B)
 * - Dimensions: M=4096, N=896, K=4864
 * - Total ops: 35.6 billion INT8 MACs
 * - Target: 6,600 GOPS (OneDNN baseline, 84% efficiency)
 *
 * This is our optimization target for cache blocking parameters.
 *
 * @author David Sanftenberg
 * @date 2025-11-12
 */

#include <gtest/gtest.h>
#include <random>
#include <chrono>
#include <iomanip>
#include <iostream>

#include "kernels/cpu/gemm/int8/SimpleInt8Gemm.h"

using namespace llaminar2::kernels::gemm;

// CPU theoretical peak for INT8 GEMM (28-core @ 2.2GHz base)
constexpr double THEORETICAL_PEAK_GIOPS = 7884.8;
constexpr double TARGET_GIOPS = 6610.0; // OneDNN baseline (84% efficiency)

/**
 * @brief Benchmark FFN_down prefill-4096 operation
 */
TEST(SimpleInt8GemmFFNDown4096, Throughput)
{
    // FFN_down dimensions (Qwen 0.5B)
    const int M = 4096;
    const int N = 896;  // d_model
    const int K = 4864; // ffn_dim

    const int k_blocks = (K + 31) / 32;

    std::cout << "\n";
    std::cout << "╔═══════════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║ FFN_DOWN PREFILL-4096 OPTIMIZATION BENCHMARK                                  ║\n";
    std::cout << "╠═══════════════════════════════════════════════════════════════════════════════╣\n";
    std::cout << "║ Dimensions:  M=" << M << ", N=" << N << ", K=" << K << "                                   ║\n";
    std::cout << "║ Total ops:   35.6 billion INT8 MACs                                           ║\n";
    std::cout << "║ Target:      " << std::fixed << std::setprecision(0) << TARGET_GIOPS << " GOPS (OneDNN baseline, 84% efficiency)                ║\n";
    std::cout << "║ Current:     ~173 GOPS (2.2% efficiency) - naive implementation              ║\n";
    std::cout << "║ Goal:        Achieve target through cache blocking optimization              ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════════════════════╝\n\n";

    // Allocate matrices
    std::vector<SimpleQ8Block> A(M * k_blocks);
    std::vector<SimpleQ8Block> B_row(k_blocks * N);
    std::vector<SimpleQ8Block> B_col(N * k_blocks);
    std::vector<float> C(M * N, 0.0f);

    // Initialize with random data
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(-127, 127);
    std::uniform_int_distribution<uint16_t> scale_dist(0x3000, 0x4000);

    std::cout << "Initializing matrices...\n";
    for (auto &block : A)
    {
        block.d = scale_dist(rng);
        for (int i = 0; i < 32; ++i)
        {
            block.qs[i] = static_cast<int8_t>(dist(rng));
        }
    }

    for (auto &block : B_row)
    {
        block.d = scale_dist(rng);
        for (int i = 0; i < 32; ++i)
        {
            block.qs[i] = static_cast<int8_t>(dist(rng));
        }
    }

    // Transpose B to column-major
    std::cout << "Transposing B matrix...\n";
    transpose_q8_blocks(B_row.data(), B_col.data(), N, k_blocks);

    // Warmup (3 iterations)
    std::cout << "Warming up (3 iterations)...\n";
    for (int i = 0; i < 3; ++i)
    {
        SimpleInt8Gemm::multiply(A.data(), B_col.data(), C.data(), M, N, K);
    }

    // Benchmark (10 iterations for stable measurement)
    std::cout << "Running benchmark (10 iterations)...\n";
    const int iterations = 10;

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i)
    {
        SimpleInt8Gemm::multiply(A.data(), B_col.data(), C.data(), M, N, K);
    }
    auto end = std::chrono::high_resolution_clock::now();

    double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double avg_ms = total_ms / iterations;

    // Calculate performance metrics
    double ops = 2.0 * M * N * K; // MACs = 2 ops each
    double giops = (ops / 1e9) / (avg_ms / 1000.0);
    double efficiency = (giops / THEORETICAL_PEAK_GIOPS) * 100.0;
    double target_pct = (giops / TARGET_GIOPS) * 100.0;

    std::cout << "\n";
    std::cout << "╔═══════════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║ RESULTS                                                                       ║\n";
    std::cout << "╠═══════════════════════════════════════════════════════════════════════════════╣\n";
    std::cout << "║ Time per iteration:     " << std::setw(10) << std::fixed << std::setprecision(2)
              << avg_ms << " ms                                             ║\n";
    std::cout << "║ Throughput:             " << std::setw(10) << std::fixed << std::setprecision(2)
              << giops << " GOPS                                           ║\n";
    std::cout << "║ Hardware efficiency:    " << std::setw(10) << std::fixed << std::setprecision(2)
              << efficiency << " %                                              ║\n";
    std::cout << "║ vs OneDNN target:       " << std::setw(10) << std::fixed << std::setprecision(2)
              << target_pct << " %                                              ║\n";
    std::cout << "╠═══════════════════════════════════════════════════════════════════════════════╣\n";

    if (giops >= TARGET_GIOPS)
    {
        std::cout << "║ ✓ TARGET ACHIEVED!                                                            ║\n";
    }
    else
    {
        double speedup_needed = TARGET_GIOPS / giops;
        std::cout << "║ Speedup needed: " << std::setw(10) << std::fixed << std::setprecision(2)
                  << speedup_needed << "×                                                    ║\n";
    }

    std::cout << "╚═══════════════════════════════════════════════════════════════════════════════╝\n\n";

    // Sanity check: result should not be all zeros
    double sum = 0.0;
    for (int i = 0; i < std::min(1000, M * N); ++i)
    {
        sum += std::abs(C[i]);
    }
    EXPECT_GT(sum, 0.0) << "Output is all zeros - computation failed!";

    // Print summary for easy tracking
    std::cout << "SUMMARY: " << std::fixed << std::setprecision(2)
              << giops << " GOPS (" << efficiency << "% efficiency, "
              << target_pct << "% of target)\n\n";
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
