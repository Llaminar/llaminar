/**
 * @file Perf__SimpleInt8Gemm.cpp
 * @brief Benchmark for simple INT8 GEMM baseline (no abstractions)
 *
 * This tests our clean-sheet SimpleInt8Gemm design to establish
 * baseline performance before adding complexity.
 *
 * @author David Sanftenberg
 * @date 2025-11-12
 */

#include <gtest/gtest.h>
#include <random>
#include <chrono>
#include <iomanip>
#include <cmath>

#include "kernels/cpu/gemm/int8/SimpleInt8Gemm.h"

using namespace llaminar2::kernels::gemm;

// CPU theoretical peak for INT8 GEMM (28-core @ 2.2GHz base)
// AVX512VNNI: 64 INT8 MACs per cycle per core
// Peak = 28 cores × 2.2 GHz × 2 FMA units × 64 INT8 ops/cycle = 7,884.8 GIOPS
constexpr double THEORETICAL_PEAK_GIOPS = 7884.8; // 28-core @ 2.2GHz base

/**
 * @brief Test SimpleInt8Gemm on Qwen 0.5B operations
 */
TEST(SimpleInt8GemmTest, QwenOperations)
{
    // Qwen 0.5B dimensions
    const int d_model = 896;
    const int ffn_dim = 4864;

    struct TestCase
    {
        std::string name;
        int m, n, k;
    };

    std::vector<TestCase> cases = {
        {"Q_proj (decode)", 1, d_model, d_model},
        {"FFN_gate (decode)", 1, ffn_dim, d_model},
        {"FFN_down (decode)", 1, d_model, ffn_dim},
        {"Q_proj (prefill-32)", 32, d_model, d_model},
        {"FFN_gate (prefill-32)", 32, ffn_dim, d_model},
        {"FFN_down (prefill-32)", 32, d_model, ffn_dim},
        {"Q_proj (prefill-512)", 512, d_model, d_model},
        {"FFN_gate (prefill-512)", 512, ffn_dim, d_model},
        {"FFN_down (prefill-512)", 512, d_model, ffn_dim},
    };

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(-127, 127);
    std::uniform_int_distribution<uint16_t> scale_dist(0x3000, 0x4000); // FP16 range

    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║ SIMPLE INT8 GEMM BASELINE BENCHMARK                                        ║\n";
    std::cout << "╠════════════════════════════════════════════════════════════════════════════╣\n";
    std::cout << "║ No microkernels, no panels, no abstractions - just pure VNNI              ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════════════════╝\n\n";

    for (const auto &tc : cases)
    {
        const int k_blocks = (tc.k + 31) / 32;
        const int n_blocks = (tc.n + 31) / 32;

        // Allocate A (row-major), B (row-major initially), C
        std::vector<SimpleQ8Block> A(tc.m * k_blocks);
        std::vector<SimpleQ8Block> B_row(k_blocks * tc.n);
        std::vector<SimpleQ8Block> B_col(tc.n * k_blocks);
        std::vector<float> C(tc.m * tc.n, 0.0f);

        // Initialize with random data
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
        transpose_q8_blocks(B_row.data(), B_col.data(), tc.n, k_blocks);

        // Warmup
        for (int i = 0; i < 3; ++i)
        {
            SimpleInt8Gemm::multiply(A.data(), B_col.data(), C.data(), tc.m, tc.n, tc.k);
        }

        // Benchmark
        const int iterations = (tc.m >= 128) ? 10 : 50;
        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < iterations; ++i)
        {
            SimpleInt8Gemm::multiply(A.data(), B_col.data(), C.data(), tc.m, tc.n, tc.k);
        }

        auto end = std::chrono::high_resolution_clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double avg_ms = total_ms / iterations;

        // Calculate performance
        // For INT8 GEMM: 2 ops per MAC (multiply + accumulate)
        double ops = 2.0 * tc.m * tc.n * tc.k;
        double giops = (ops / 1e9) / (avg_ms / 1000.0);
        double efficiency = (giops / THEORETICAL_PEAK_GIOPS) * 100.0;

        std::cout << std::left << std::setw(25) << tc.name
                  << " [" << std::setw(4) << tc.m << "×" << std::setw(4) << tc.n << "×" << std::setw(4) << tc.k << "] "
                  << std::fixed << std::setprecision(2)
                  << std::setw(10) << giops << " GOPS   "
                  << std::setw(8) << avg_ms << " ms   "
                  << std::setw(6) << efficiency << "% eff\n";
    }

    std::cout << "\nTarget: 6,610 GOPS (OneDNN baseline, 84% efficiency)\n";
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
