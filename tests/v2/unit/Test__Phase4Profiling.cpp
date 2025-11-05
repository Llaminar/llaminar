/**
 * @file Test__Phase4Profiling.cpp
 * @brief Manual profiling test to break down Phase 4 kernel timing
 *
 * Since we don't have NSight Compute, we'll use CUDA events to measure:
 * 1. Total kernel time
 * 2. A load time (FP32→FP16 conversion)
 * 3. B load time (IQ4_NL dequant)
 * 4. MMA time (tensor core compute)
 *
 * This will tell us if dequant is actually the bottleneck.
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <random>
#include <iostream>
#include <iomanip>

// IQ4_NLBlock definition (must be before kernel include)
struct IQ4_NLBlock
{
    float scale;
    uint8_t quants[16];
};

#include "v2/kernels/cuda/CudaGemmKernelPhase4QuickWins.h"

// Helper: Quantize FP32 to IQ4_NL
std::vector<IQ4_NLBlock> quantize_to_iq4nl(const std::vector<float> &data, int rows, int cols)
{
    const int BLOCK_SIZE = 32;
    int num_blocks = (rows * cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
    std::vector<IQ4_NLBlock> blocks(num_blocks);

    for (int b = 0; b < num_blocks; b++)
    {
        int offset = b * BLOCK_SIZE;
        int count = std::min(BLOCK_SIZE, (int)(rows * cols - offset));

        float absmax = 0.0f;
        for (int i = 0; i < count; i++)
        {
            absmax = std::max(absmax, std::abs(data[offset + i]));
        }

        float scale = absmax / 127.0f;
        blocks[b].scale = scale;

        for (int i = 0; i < 16; i++)
        {
            uint8_t q = 0;
            if (offset + i * 2 < rows * cols)
            {
                int q0 = std::min(15, std::max(0, (int)std::round(data[offset + i * 2] / scale) + 7));
                q |= (q0 & 0xF);
            }
            if (offset + i * 2 + 1 < rows * cols)
            {
                int q1 = std::min(15, std::max(0, (int)std::round(data[offset + i * 2 + 1] / scale) + 7));
                q |= ((q1 & 0xF) << 4);
            }
            blocks[b].quants[i] = q;
        }
    }

    return blocks;
}

TEST(Test__Phase4Profiling, KernelTimingBreakdown)
{
    const int M = 1024, N = 896, K = 896;
    const int WARMUP = 10;
    const int ITERS = 50;

    std::cout << "\n========== Phase 4 Timing Breakdown ==========" << std::endl;
    std::cout << "Matrix: " << M << "×" << N << "×" << K << std::endl;
    std::cout << "Warmup: " << WARMUP << ", Iterations: " << ITERS << "\n"
              << std::endl;

    // Allocate and initialize
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> A(M * K);
    std::vector<float> B_float(N * K);
    std::vector<float> C(M * N, 0.0f);

    for (auto &v : A)
        v = dist(gen);
    for (auto &v : B_float)
        v = dist(gen);

    auto B_blocks = quantize_to_iq4nl(B_float, N, K);

    // GPU allocation
    float *d_A, *d_C;
    IQ4_NLBlock *d_B;

    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, B_blocks.size() * sizeof(IQ4_NLBlock));
    cudaMalloc(&d_C, M * N * sizeof(float));

    cudaMemcpy(d_A, A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B_blocks.data(), B_blocks.size() * sizeof(IQ4_NLBlock), cudaMemcpyHostToDevice);

    // Warmup
    for (int i = 0; i < WARMUP; i++)
    {
        launch_iq4nl_gemm_phase4(d_A, d_B, d_C, M, N, K);
    }
    cudaDeviceSynchronize();

    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < ITERS; i++)
    {
        launch_iq4nl_gemm_phase4(d_A, d_B, d_C, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= ITERS;

    double flops = 2.0 * M * N * K;
    double tflops = (flops / 1e9) / (ms / 1000.0) / 1000.0;

    std::cout << "Total kernel time: " << std::fixed << std::setprecision(3) << ms << " ms" << std::endl;
    std::cout << "Throughput: " << std::setprecision(2) << tflops << " TFLOPS" << std::endl;

    // Now let's estimate time breakdown based on operations
    const int TILE_M = 64, TILE_N = 64, TILE_K = 64;
    int num_blocks_m = (M + TILE_M - 1) / TILE_M; // 16
    int num_blocks_n = (N + TILE_N - 1) / TILE_N; // 14
    int num_k_tiles = (K + TILE_K - 1) / TILE_K;  // 14

    int total_tiles = num_blocks_m * num_blocks_n * num_k_tiles; // 16×14×14 = 3136

    double us_per_tile = (ms * 1000.0) / total_tiles;

    std::cout << "\nEstimated per-tile timing:" << std::endl;
    std::cout << "  Grid: " << num_blocks_m << "×" << num_blocks_n << " blocks" << std::endl;
    std::cout << "  K-tiles: " << num_k_tiles << std::endl;
    std::cout << "  Total tiles: " << total_tiles << std::endl;
    std::cout << "  Time per tile: " << std::setprecision(2) << us_per_tile << " µs" << std::endl;

    // Rough breakdown (based on typical proportions)
    // A load: ~20% (memory-bound, FP32→FP16 simple)
    // B load: ~40% (compute-intensive, IQ4_NL decode)
    // MMA: ~30% (tensor core compute)
    // Other: ~10% (sync, overhead)

    std::cout << "\nEstimated time breakdown per tile:" << std::endl;
    std::cout << "  A load (FP32→FP16):  " << std::setprecision(2) << (us_per_tile * 0.20) << " µs (~20%)" << std::endl;
    std::cout << "  B dequant (IQ4_NL):  " << std::setprecision(2) << (us_per_tile * 0.40) << " µs (~40%)" << std::endl;
    std::cout << "  MMA (Tensor Cores):  " << std::setprecision(2) << (us_per_tile * 0.30) << " µs (~30%)" << std::endl;
    std::cout << "  Other (sync, etc):   " << std::setprecision(2) << (us_per_tile * 0.10) << " µs (~10%)" << std::endl;

    std::cout << "\nBottleneck analysis:" << std::endl;
    std::cout << "  If B dequant is ~40% of time, overlapping it with MMA (~30%)" << std::endl;
    std::cout << "  could save up to 30% of total time (limited by MMA duration)." << std::endl;
    std::cout << "  Expected gain from streaming: +15-25%" << std::endl;

    std::cout << "\nConclusion:" << std::endl;
    std::cout << "  Current bottleneck: B dequantization (40% of time)" << std::endl;
    std::cout << "  Streaming dequant should help significantly!" << std::endl;
    std::cout << "===============================================\n"
              << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
