/**
 * @file Perf__IQ4_NL_GEMM_Diagnostics.cpp
 * @brief Diagnostic test to identify IQ4_NL GEMM performance regression
 *
 * Historical performance: ~1100 GFLOPS @ large batch
 * Current performance:    ~392 GFLOPS @ M=2048
 * Regression factor:      2.8× slower
 *
 * This test measures:
 * - Memory access patterns
 * - Block size impact
 * - Tile configuration effectiveness
 * - SIMD vectorization status
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <omp.h>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>

#include "../../src/v2/loaders/ModelLoader.h"
#include "../../src/v2/tensors/Tensors.h"
#include "../../src/v2/kernels/cpu/gemm/GemmAutoTuner.h"
#include "../../src/v2/utils/CPUFeatures.h"

using namespace llaminar2;

class IQ4_NL_Diagnostics : public ::testing::Test
{
protected:
    void SetUp() override
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &size_);

        // Load model once
        if (rank_ == 0)
        {
            std::cout << "Loading model for diagnostics..." << std::endl;
        }

        loader_ = std::make_unique<ModelLoader>();
        bool loaded = loader_->loadModel("models/qwen2.5-0.5b-instruct-iq4_nl.gguf");
        ASSERT_TRUE(loaded) << "Failed to load model";
    }

    void TearDown() override {}

    int rank_ = 0;
    int size_ = 1;
    std::unique_ptr<ModelLoader> loader_;
};

/**
 * @brief Diagnostic: Check SIMD capability and usage
 */
TEST_F(IQ4_NL_Diagnostics, SIMD_Capability)
{
    if (rank_ != 0)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        return;
    }

    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                   SIMD CAPABILITY CHECK                        ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n";

    const auto &cpu = CPUFeatures::getInstance();

    std::cout << "CPU Features:\n";
    std::cout << "  AVX2:       " << (cpu.hasAVX2() ? "✓ YES" : "✗ NO") << "\n";
    std::cout << "  AVX512F:    " << (cpu.hasAVX512F() ? "✓ YES" : "✗ NO") << "\n";
    std::cout << "  AVX512BW:   " << (cpu.hasAVX512BW() ? "✓ YES" : "✗ NO") << "\n";
    std::cout << "  AVX512VNNI: " << (cpu.hasAVX512VNNI() ? "✓ YES" : "✗ NO") << "\n";
    std::cout << "  FMA:        " << (cpu.hasFMA() ? "✓ YES" : "✗ NO") << "\n";

    std::cout << "\nExpected for optimal IQ4_NL performance:\n";
    std::cout << "  - AVX2 (minimum):  " << (cpu.hasAVX2() ? "PASS" : "FAIL") << "\n";
    std::cout << "  - AVX512 (best):   " << (cpu.hasAVX512F() ? "PASS" : "not available") << "\n";
    std::cout << "  - FMA (required):  " << (cpu.hasFMA() ? "PASS" : "FAIL") << "\n";

    if (!cpu.hasAVX2())
    {
        std::cout << "\n⚠️  WARNING: AVX2 not available - falling back to scalar code\n";
        std::cout << "   This could explain 2-3× performance regression!\n";
    }

    std::cout << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
}

/**
 * @brief Diagnostic: Measure block decode overhead
 */
TEST_F(IQ4_NL_Diagnostics, Block_Decode_Overhead)
{
    if (rank_ != 0)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        return;
    }

    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║              BLOCK DECODE OVERHEAD ANALYSIS                    ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n";

    // Load IQ4_NL weight
    auto weight = loader_->loadTensor("blk.0.attn_q.weight", 0);
    auto iq4nl_weight = std::dynamic_pointer_cast<IQ4_NLTensor>(weight);
    ASSERT_TRUE(iq4nl_weight) << "Failed to load IQ4_NL tensor";

    const size_t num_blocks = iq4nl_weight->shape()[0] * (iq4nl_weight->shape()[1] / 32);
    std::cout << "Weight shape: " << iq4nl_weight->shape()[0] << " × " << iq4nl_weight->shape()[1] << "\n";
    std::cout << "Total blocks: " << num_blocks << "\n";
    std::cout << "Block size: 32 elements\n\n";

    // Measure decode time for single block
    float output[32];
    const int warmup = 100;
    const int iterations = 10000;

    // Warmup
    for (int i = 0; i < warmup; ++i)
    {
        iq4nl_weight->decode_block_at(0, 0, output);
    }

    // Time block decode
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i)
    {
        iq4nl_weight->decode_block_at(0, i % (num_blocks / iq4nl_weight->shape()[0]), output);
    }
    auto end = std::chrono::high_resolution_clock::now();

    double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double ns_per_block = (total_ms * 1e6) / iterations;
    double blocks_per_sec = iterations / (total_ms / 1000.0);

    std::cout << "Block decode performance:\n";
    std::cout << "  Time per block:     " << std::fixed << std::setprecision(2) << ns_per_block << " ns\n";
    std::cout << "  Blocks per second:  " << std::scientific << std::setprecision(2) << blocks_per_sec << "\n";
    std::cout << "  Elements per second: " << std::scientific << (blocks_per_sec * 32) << "\n\n";

    // For GEMM at 2048×896×896
    const size_t M = 2048;
    const size_t N = 896;
    const size_t K = 896;
    const size_t total_ops = 2 * M * N * K;        // FMA counts as 2 ops
    const size_t blocks_needed = M * (K / 32) * N; // Approximate

    double decode_time_s = (blocks_needed * ns_per_block) / 1e9;
    double gemm_time_budget_s = total_ops / (1100e9); // Historical 1100 GFLOPS

    std::cout << "For GEMM (2048×896×896):\n";
    std::cout << "  Blocks needed:      " << blocks_needed << "\n";
    std::cout << "  Decode overhead:    " << std::fixed << std::setprecision(3) << decode_time_s << " seconds\n";
    std::cout << "  GEMM budget (1100 GFLOPS): " << gemm_time_budget_s << " seconds\n";
    std::cout << "  Decode fraction:    " << std::setprecision(1) << (decode_time_s / gemm_time_budget_s * 100) << "%\n\n";

    if (decode_time_s / gemm_time_budget_s > 0.1)
    {
        std::cout << "⚠️  Decode overhead is >10% of compute time - potential bottleneck!\n";
    }

    std::cout << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
}

/**
 * @brief Diagnostic: Tile size sensitivity analysis
 */
TEST_F(IQ4_NL_Diagnostics, Tile_Size_Sensitivity)
{
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║              TILE SIZE SENSITIVITY ANALYSIS                    ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n";

    // Check what tiles the autotuner selected
    std::cout << "AutoTuner tile selection for IQ4_NL GEMM:\n\n";

    struct TestConfig
    {
        int M;
        int N;
        int K;
        std::string description;
    };

    std::vector<TestConfig> configs = {
        {32, 896, 896, "SmallBatch"},
        {512, 896, 896, "LargeBatch (near peak)"},
        {2048, 896, 896, "XXLargeBatch (current test)"},
        {8192, 896, 896, "MassiveBatch (plateau)"}};

    for (const auto &config : configs)
    {
        if (rank_ == 0)
        {
            std::cout << config.description << " [" << config.M << "×" << config.N << "×" << config.K << "]:\n";

            GemmAutoTuner tuner;
            auto best_variant = tuner.selectBest(config.M, config.N, config.K);

            std::cout << "  Tile: " << best_variant.tile_m << "×" << best_variant.tile_n << "\n";
            std::cout << "  Unroll: " << best_variant.unroll_factor << "\n";
            std::cout << "  Prefetch: " << best_variant.prefetch_distance << "\n";
            std::cout << "  Estimated GFLOPS: " << best_variant.estimated_gflops << "\n\n";
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

/**
 * @brief Diagnostic: Memory access pattern analysis
 */
TEST_F(IQ4_NL_Diagnostics, Memory_Access_Pattern)
{
    if (rank_ != 0)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        return;
    }

    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║           MEMORY ACCESS PATTERN ANALYSIS                       ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n";

    // Load weight tensor
    auto weight = loader_->loadTensor("blk.0.attn_q.weight", 0);
    auto iq4nl_weight = std::dynamic_pointer_cast<IQ4_NLTensor>(weight);

    const size_t rows = iq4nl_weight->shape()[0];
    const size_t cols = iq4nl_weight->shape()[1];
    const size_t block_size = 32;
    const size_t blocks_per_row = cols / block_size;

    std::cout << "IQ4_NL Tensor Layout:\n";
    std::cout << "  Dimensions:      " << rows << " × " << cols << "\n";
    std::cout << "  Block size:      " << block_size << " elements\n";
    std::cout << "  Blocks per row:  " << blocks_per_row << "\n";
    std::cout << "  Total blocks:    " << (rows * blocks_per_row) << "\n\n";

    // Calculate memory footprint
    const size_t bytes_per_block = sizeof(IQ4_NLBlock);
    const size_t total_bytes = rows * blocks_per_row * bytes_per_block;
    const size_t total_mb = total_bytes / (1024 * 1024);

    std::cout << "Memory footprint:\n";
    std::cout << "  Bytes per block: " << bytes_per_block << "\n";
    std::cout << "  Total size:      " << total_mb << " MB\n\n";

    // Cache analysis (assumes 32KB L1, 256KB L2, 35MB LLC per core)
    const size_t L1_size = 32 * 1024;
    const size_t L2_size = 256 * 1024;
    const size_t LLC_size = 35 * 1024 * 1024;

    std::cout << "Cache residency:\n";
    std::cout << "  L1 (32 KB):      " << (total_bytes <= L1_size ? "✓ FITS" : "✗ SPILLS") << "\n";
    std::cout << "  L2 (256 KB):     " << (total_bytes <= L2_size ? "✓ FITS" : "✗ SPILLS") << "\n";
    std::cout << "  LLC (35 MB):     " << (total_bytes <= LLC_size ? "✓ FITS" : "✗ SPILLS") << "\n\n";

    if (total_bytes > L2_size)
    {
        std::cout << "⚠️  Weight tensor doesn't fit in L2 cache!\n";
        std::cout << "   This causes repeated DRAM accesses during GEMM.\n";
        std::cout << "   Consider: blocking strategy, better tiling, or weight caching.\n\n";
    }

    std::cout << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
}

int main(int argc, char **argv)
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();

    MPI_Finalize();
    return result;
}
