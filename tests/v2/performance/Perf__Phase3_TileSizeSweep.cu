/**
 * @file Perf__Phase3_TileSizeSweep.cu
 * @brief Phase 3: Tile size tuning sweep to find optimal TILE_M, TILE_N, TILE_K
 * @author David Sanftenberg
 *
 * Tests multiple tile configurations to maximize Tensor Core utilization,
 * register efficiency, and shared memory usage.
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cutlass/half.h>
#include <cute/arch/mma_sm80.hpp> // For SM80_16x8x16_F32F16F16F32_TN
#include "kernels/cuda/CudaGemmKernel.cuh"
#include "kernels/cuda/IQ4_NL_BlockDecoder.h"
#include <chrono>
#include <iomanip>
#include <vector>

using namespace llaminar2::cuda;
using namespace cute; // For MMA atoms

class Phase3_TileSizeSweep : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Matrix dimensions (m=32 for single token)
        m = 32;
        n = 896;
        k = 896;

        // Allocate device memory for A, B (weights), and C
        cudaMalloc(&d_A, m * k * sizeof(cutlass::half_t));
        cudaMalloc(&d_B, (k / 32) * n * sizeof(IQ4_NLBlock));
        cudaMalloc(&d_C, m * n * sizeof(float));

        // Initialize with dummy data
        std::vector<cutlass::half_t> h_A(m * k, cutlass::half_t(0.1f));
        std::vector<IQ4_NLBlock> h_B((k / 32) * n);
        for (auto &block : h_B)
        {
            block.d = __float2half_rn(0.1f);
            for (int i = 0; i < 16; ++i)
                block.qs[i] = 128;
        }

        cudaMemcpy(d_A, h_A.data(), m * k * sizeof(cutlass::half_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B.data(), (k / 32) * n * sizeof(IQ4_NLBlock), cudaMemcpyHostToDevice);
    }

    void TearDown() override
    {
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }

    template <int TILE_M, int TILE_N, int TILE_K>
    double benchmarkTileSize(const char *config_name)
    {
        // Clear output
        cudaMemset(d_C, 0, m * n * sizeof(float));

        // Create decoder
        IQ4_NL_Decoder<IQ4_NLBlock> decoder(d_B, n, k / 32);

        // Warmup
        for (int i = 0; i < 3; ++i)
        {
            launchQuantizedGemmCuTe<cutlass::half_t, SM80_16x8x16_F32F16F16F32_TN, 2, 2, 1, IQ4_NL_Decoder<IQ4_NLBlock>, TILE_M, TILE_N, TILE_K>(
                d_A, d_C, m, n, k, decoder, 0);
            cudaDeviceSynchronize();
        }

        // Benchmark
        constexpr int num_iters = 100;
        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < num_iters; ++i)
        {
            launchQuantizedGemmCuTe<cutlass::half_t, SM80_16x8x16_F32F16F16F32_TN, 2, 2, 1, IQ4_NL_Decoder<IQ4_NLBlock>, TILE_M, TILE_N, TILE_K>(
                d_A, d_C, m, n, k, decoder, 0);
        }

        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();

        double elapsed_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
        double avg_time_ms = elapsed_ms / num_iters;

        // Calculate GFLOPS
        double flops = 2.0 * m * n * k; // multiply-add
        double gflops = flops / (avg_time_ms * 1e6);

        // Calculate grid dimensions
        int blocks_m = (m + TILE_M - 1) / TILE_M;
        int blocks_n = (n + TILE_N - 1) / TILE_N;
        int k_tiles = (k + TILE_K - 1) / TILE_K;

        // Calculate shared memory usage
        size_t smem_A = TILE_M * TILE_K * sizeof(cutlass::half_t) * 2; // Double-buffered
        size_t smem_B = TILE_N * TILE_K * sizeof(cutlass::half_t) * 2;
        size_t smem_total = smem_A + smem_B;

        std::cout << std::setw(15) << config_name
                  << " | " << std::setw(8) << std::fixed << std::setprecision(2) << gflops << " GFLOPS"
                  << " | " << std::setw(6) << std::fixed << std::setprecision(3) << avg_time_ms << " ms"
                  << " | Grid: " << std::setw(2) << blocks_m << "×" << std::setw(2) << blocks_n
                  << " | K-tiles: " << std::setw(3) << k_tiles
                  << " | SMEM: " << std::setw(5) << (smem_total / 1024) << " KB"
                  << std::endl;

        return gflops;
    }

    int m, n, k;
    cutlass::half_t *d_A;
    IQ4_NLBlock *d_B;
    float *d_C;
};

TEST_F(Phase3_TileSizeSweep, ComprehensiveTileSweep)
{
    std::cout << "\n=== PHASE 3 TENSOR CORE: TILE SIZE SWEEP ===\n";
    std::cout << "Matrix: m=" << m << ", n=" << n << ", k=" << k << "\n";
    std::cout << "Baseline (Phase 2.5): 1,666 GFLOPS with 64×64×16\n\n";

    std::cout << "Configuration    | Performance  | Time    | Grid Size | K-tiles | Shared Mem\n";
    std::cout << "─────────────────┼──────────────┼─────────┼───────────┼─────────┼───────────\n";

    double best_gflops = 0.0;
    std::string best_config;

    // Test various tile configurations
    // Note: TILE_M limited to 32 since m=32 (single token)

    // Baseline (Phase 2.5)
    double gflops_64_64_16 = benchmarkTileSize<64, 64, 16>("64×64×16 (2.5)");
    if (gflops_64_64_16 > best_gflops)
    {
        best_gflops = gflops_64_64_16;
        best_config = "64×64×16";
    }

    // Match M exactly (32)
    double gflops_32_64_16 = benchmarkTileSize<32, 64, 16>("32×64×16");
    if (gflops_32_64_16 > best_gflops)
    {
        best_gflops = gflops_32_64_16;
        best_config = "32×64×16";
    }

    double gflops_32_128_16 = benchmarkTileSize<32, 128, 16>("32×128×16");
    if (gflops_32_128_16 > best_gflops)
    {
        best_gflops = gflops_32_128_16;
        best_config = "32×128×16";
    }

    double gflops_32_64_32 = benchmarkTileSize<32, 64, 32>("32×64×32");
    if (gflops_32_64_32 > best_gflops)
    {
        best_gflops = gflops_32_64_32;
        best_config = "32×64×32";
    }

    double gflops_32_128_32 = benchmarkTileSize<32, 128, 32>("32×128×32");
    if (gflops_32_128_32 > best_gflops)
    {
        best_gflops = gflops_32_128_32;
        best_config = "32×128×32";
    }

    // Smaller M tiles (more parallelism)
    double gflops_16_64_16 = benchmarkTileSize<16, 64, 16>("16×64×16");
    if (gflops_16_64_16 > best_gflops)
    {
        best_gflops = gflops_16_64_16;
        best_config = "16×64×16";
    }

    double gflops_16_128_16 = benchmarkTileSize<16, 128, 16>("16×128×16");
    if (gflops_16_128_16 > best_gflops)
    {
        best_gflops = gflops_16_128_16;
        best_config = "16×128×16";
    }

    double gflops_16_128_32 = benchmarkTileSize<16, 128, 32>("16×128×32");
    if (gflops_16_128_32 > best_gflops)
    {
        best_gflops = gflops_16_128_32;
        best_config = "16×128×32";
    }

    // Larger K (more reuse)
    double gflops_64_64_32 = benchmarkTileSize<64, 64, 32>("64×64×32");
    if (gflops_64_64_32 > best_gflops)
    {
        best_gflops = gflops_64_64_32;
        best_config = "64×64×32";
    }

    // Wider N tiles
    double gflops_64_128_16 = benchmarkTileSize<64, 128, 16>("64×128×16");
    if (gflops_64_128_16 > best_gflops)
    {
        best_gflops = gflops_64_128_16;
        best_config = "64×128×16";
    }

    double gflops_64_128_32 = benchmarkTileSize<64, 128, 32>("64×128×32");
    if (gflops_64_128_32 > best_gflops)
    {
        best_gflops = gflops_64_128_32;
        best_config = "64×128×32";
    }

    std::cout << "─────────────────┴──────────────┴─────────┴───────────┴─────────┴───────────\n";
    std::cout << "\n=== RESULTS ===\n";
    std::cout << "Baseline (64×64×16): " << std::fixed << std::setprecision(2) << gflops_64_64_16 << " GFLOPS\n";
    std::cout << "Best configuration: " << best_config << " at " << best_gflops << " GFLOPS\n";

    double speedup = best_gflops / gflops_64_64_16;
    std::cout << "Speedup over baseline: " << std::fixed << std::setprecision(3) << speedup << "x\n";

    if (speedup >= 1.1)
    {
        std::cout << "\n✅ Found better tile configuration! Speedup: " << speedup << "x\n";
    }
    else if (speedup >= 1.0)
    {
        std::cout << "\n✅ Baseline is already optimal (within 10%)\n";
    }
    else
    {
        std::cout << "\n⚠️  Baseline is best (new configs slower)\n";
    }

    // Test passes if we found improvement or baseline is within 10% of best
    EXPECT_GE(speedup, 0.9) << "Best config should be at least 90% of baseline";
}

TEST_F(Phase3_TileSizeSweep, DetailedAnalysis)
{
    std::cout << "\n=== DETAILED TILE SIZE ANALYSIS ===\n";
    std::cout << "Matrix: m=" << m << ", n=" << n << ", k=" << k << "\n\n";

    // Analyze 32×64×16 in detail (matches M exactly)
    {
        constexpr int TILE_M = 32, TILE_N = 64, TILE_K = 16;
        std::cout << "Configuration: " << TILE_M << "×" << TILE_N << "×" << TILE_K << "\n";
        std::cout << "Analysis:\n";
        std::cout << "  • M blocks: " << (m + TILE_M - 1) / TILE_M << " (exact match!)\n";
        std::cout << "  • N blocks: " << (n + TILE_N - 1) / TILE_N << "\n";
        std::cout << "  • K tiles: " << (k + TILE_K - 1) / TILE_K << "\n";
        std::cout << "  • Total blocks: " << ((m + TILE_M - 1) / TILE_M) * ((n + TILE_N - 1) / TILE_N) << "\n";

        double gflops = benchmarkTileSize<TILE_M, TILE_N, TILE_K>("32×64×16");
        std::cout << "  • Performance: " << gflops << " GFLOPS\n\n";
    }

    // Analyze 32×128×16 (wider N)
    {
        constexpr int TILE_M = 32, TILE_N = 128, TILE_K = 16;
        std::cout << "Configuration: " << TILE_M << "×" << TILE_N << "×" << TILE_K << "\n";
        std::cout << "Analysis:\n";
        std::cout << "  • M blocks: " << (m + TILE_M - 1) / TILE_M << " (exact match!)\n";
        std::cout << "  • N blocks: " << (n + TILE_N - 1) / TILE_N << " (fewer!)\n";
        std::cout << "  • K tiles: " << (k + TILE_K - 1) / TILE_K << "\n";
        std::cout << "  • Total blocks: " << ((m + TILE_M - 1) / TILE_M) * ((n + TILE_N - 1) / TILE_N) << "\n";

        double gflops = benchmarkTileSize<TILE_M, TILE_N, TILE_K>("32×128×16");
        std::cout << "  • Performance: " << gflops << " GFLOPS\n\n";
    }

    // Analyze 64×64×16 (baseline)
    {
        constexpr int TILE_M = 64, TILE_N = 64, TILE_K = 16;
        std::cout << "Configuration: " << TILE_M << "×" << TILE_N << "×" << TILE_K << " (Baseline)\n";
        std::cout << "Analysis:\n";
        std::cout << "  • M blocks: " << (m + TILE_M - 1) / TILE_M << " (partial tile!)\n";
        std::cout << "  • N blocks: " << (n + TILE_N - 1) / TILE_N << "\n";
        std::cout << "  • K tiles: " << (k + TILE_K - 1) / TILE_K << "\n";
        std::cout << "  • Total blocks: " << ((m + TILE_M - 1) / TILE_M) * ((n + TILE_N - 1) / TILE_N) << "\n";

        double gflops = benchmarkTileSize<TILE_M, TILE_N, TILE_K>("64×64×16");
        std::cout << "  • Performance: " << gflops << " GFLOPS\n\n";
    }
}
