/**
 * @file Perf__PackedInt8Gemm.cpp
 * @brief Performance test for packed INT8 GEMM
 * @author David Sanftenberg
 *
 * Tests whether pre-packing B matrix improves performance vs
 * decoding blocks on-the-fly.
 */

#include <gtest/gtest.h>
#include "v2/kernels/cpu/gemm/int8/PackedInt8Gemm.h"
#include "v2/kernels/cpu/gemm/int8/SimpleInt8Gemm.h"
#include <random>
#include <chrono>
#include <iomanip>

using namespace llaminar2::kernels::gemm;

class PackedInt8GemmPerf : public ::testing::Test
{
protected:
    // FFN_down dimensions for Qwen 0.5B
    static constexpr int M = 4096;
    static constexpr int N = 896;
    static constexpr int K = 4864;
    static constexpr int K_BLOCKS = (K + 31) / 32;

    std::unique_ptr<SimpleQ8Block[]> A_blocks;
    std::unique_ptr<SimpleQ8Block[]> B_blocks;
    std::unique_ptr<float[]> C_naive;
    std::unique_ptr<float[]> C_packed;

    void SetUp() override
    {
        // Allocate matrices
        A_blocks = std::make_unique<SimpleQ8Block[]>(M * K_BLOCKS);
        B_blocks = std::make_unique<SimpleQ8Block[]>(N * K_BLOCKS);
        C_naive = std::make_unique<float[]>(M * N);
        C_packed = std::make_unique<float[]>(M * N);

        // Initialize with random data
        std::mt19937 gen(42);
        std::uniform_int_distribution<int8_t> dist_int8(-127, 127);
        std::uniform_real_distribution<float> dist_scale(0.001f, 0.1f);

        // Fill A matrix blocks
        for (int i = 0; i < M * K_BLOCKS; ++i)
        {
            // Random FP16 scale
            float scale = dist_scale(gen);
            uint32_t f;
            std::memcpy(&f, &scale, sizeof(float));
            uint16_t h = ((f >> 16) & 0x8000) | ((((f & 0x7f800000) - 0x38000000) >> 13) & 0x7c00) | ((f >> 13) & 0x03ff);
            A_blocks[i].d = h;

            // Random INT8 values
            for (int j = 0; j < 32; ++j)
            {
                A_blocks[i].qs[j] = dist_int8(gen);
            }
        }

        // Fill B matrix blocks (column-major: [N, K_BLOCKS])
        for (int j = 0; j < N; ++j)
        {
            for (int kb = 0; kb < K_BLOCKS; ++kb)
            {
                int idx = j * K_BLOCKS + kb;

                // Random FP16 scale
                float scale = dist_scale(gen);
                uint32_t f;
                std::memcpy(&f, &scale, sizeof(float));
                uint16_t h = ((f >> 16) & 0x8000) | ((((f & 0x7f800000) - 0x38000000) >> 13) & 0x7c00) | ((f >> 13) & 0x03ff);
                B_blocks[idx].d = h;

                // Random INT8 values
                for (int ki = 0; ki < 32; ++ki)
                {
                    B_blocks[idx].qs[ki] = dist_int8(gen);
                }
            }
        }
    }
};

TEST_F(PackedInt8GemmPerf, ComparePackedVsNaive)
{
    std::cout << "\n╔═══════════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║ PACKED INT8 GEMM PERFORMANCE COMPARISON                                       ║\n";
    std::cout << "╠═══════════════════════════════════════════════════════════════════════════════╣\n";
    std::cout << "║ Dimensions:  M=" << M << ", N=" << N << ", K=" << K << "                                   ║\n";
    std::cout << "║ Strategy:    Pre-pack B matrix once, then run optimized GEMM                 ║\n";
    std::cout << "║ Hypothesis:  Eliminates repeated block decoding overhead                     ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════════════════════╝\n\n";

    // BASELINE: Naive implementation (decodes blocks on-the-fly)
    std::cout << "Running baseline (naive block-based GEMM)...\n";

    constexpr int WARMUP = 3;
    constexpr int ITERS = 10;

    // Warmup
    for (int i = 0; i < WARMUP; ++i)
    {
        SimpleInt8Gemm::multiply_naive(A_blocks.get(), B_blocks.get(), C_naive.get(), M, N, K);
    }

    // Benchmark
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERS; ++i)
    {
        SimpleInt8Gemm::multiply_naive(A_blocks.get(), B_blocks.get(), C_naive.get(), M, N, K);
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    double naive_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / ITERS;
    double ops = 2.0 * M * N * K; // Multiply-add = 2 ops
    double naive_gops = (ops / (naive_ms / 1000.0)) / 1e9;

    std::cout << "  Time per iteration: " << std::fixed << std::setprecision(2) << naive_ms << " ms\n";
    std::cout << "  Throughput:         " << std::fixed << std::setprecision(2) << naive_gops << " GOPS\n\n";

    // OPTIMIZED: Pre-pack B matrix, then run GEMM
    std::cout << "Pre-packing B matrix...\n";
    auto pack_t0 = std::chrono::high_resolution_clock::now();
    auto B_packed = PackedInt8Gemm::pack_b_matrix(B_blocks.get(), N, K);
    auto pack_t1 = std::chrono::high_resolution_clock::now();
    double pack_ms = std::chrono::duration<double, std::milli>(pack_t1 - pack_t0).count();
    std::cout << "  Packing time: " << pack_ms << " ms (done once)\n\n";

    std::cout << "Running packed GEMM...\n";

    // Warmup
    for (int i = 0; i < WARMUP; ++i)
    {
        PackedInt8Gemm::multiply_packed(A_blocks.get(), B_packed.get(), C_packed.get(), M, N, K);
    }

    // Benchmark
    auto t2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERS; ++i)
    {
        PackedInt8Gemm::multiply_packed(A_blocks.get(), B_packed.get(), C_packed.get(), M, N, K);
    }
    auto t3 = std::chrono::high_resolution_clock::now();

    double packed_ms = std::chrono::duration<double, std::milli>(t3 - t2).count() / ITERS;
    double packed_gops = (ops / (packed_ms / 1000.0)) / 1e9;

    std::cout << "  Time per iteration: " << std::fixed << std::setprecision(2) << packed_ms << " ms\n";
    std::cout << "  Throughput:         " << std::fixed << std::setprecision(2) << packed_gops << " GOPS\n\n";

    // Comparison
    double speedup = packed_gops / naive_gops;

    std::cout << "╔═══════════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║ RESULTS                                                                       ║\n";
    std::cout << "╠═══════════════════════════════════════════════════════════════════════════════╣\n";
    std::cout << "║ Naive (block-based):    " << std::setw(8) << naive_gops << " GOPS                                       ║\n";
    std::cout << "║ Packed (pre-expanded):  " << std::setw(8) << packed_gops << " GOPS                                       ║\n";
    std::cout << "║ Speedup:                " << std::setw(8) << std::setprecision(2) << speedup << "×                                            ║\n";
    std::cout << "╠═══════════════════════════════════════════════════════════════════════════════╣\n";

    if (speedup > 1.1)
    {
        std::cout << "║ ✅ Pre-packing helps! Speedup: " << speedup << "×                                       ║\n";
    }
    else if (speedup < 0.9)
    {
        std::cout << "║ ❌ Pre-packing hurts! Slowdown: " << (1.0 / speedup) << "×                                      ║\n";
    }
    else
    {
        std::cout << "║ ⚠️  Pre-packing neutral (within 10%)                                         ║\n";
    }
    std::cout << "╚═══════════════════════════════════════════════════════════════════════════════╝\n";

    // Verify correctness (spot check)
    double max_diff = 0.0;
    for (int i = 0; i < std::min(1000, M * N); ++i)
    {
        double diff = std::abs(C_naive[i] - C_packed[i]);
        max_diff = std::max(max_diff, diff);
    }

    std::cout << "\nCorrectness check (first 1000 elements): max_diff = " << max_diff << "\n";
    EXPECT_LT(max_diff, 1e-3) << "Packed and naive results differ!";
}
