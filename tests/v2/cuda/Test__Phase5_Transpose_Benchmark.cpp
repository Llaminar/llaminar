/**
 * @file Test__Phase5_Transpose_Benchmark.cpp
 * @brief Quick benchmark comparing row-major vs transposed B layouts
 *
 * NCU Analysis shows:
 * - Row-major: 8.7 / 32 bytes per sector (27% utilization)
 * - Expected transpose: 28-32 / 32 bytes (87-100% utilization)
 * - Speedup potential: 62%
 *
 * @author David Sanftenberg
 * @date November 9, 2025
 */

#include <gtest/gtest.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>
#include <vector>
#include <iostream>
#include <iomanip>
#include <random>

#include "kernels/cuda/CudaGemmJITPhase5.h"
#include "kernels/cuda/CudaGemmConfigPhase5.h"

using namespace llaminar2::cuda;

// IQ4_NL block structure
struct IQ4_NLBlock
{
    uint16_t qs[8]; // 16 bytes
    uint16_t d;     // 2 bytes (FP16)
}; // Total: 18 bytes

class Phase5TransposeBenchmark : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Initialize CUDA driver API
        cuInit(0);
        CUdevice device;
        cuDeviceGet(&device, 0);
        cuCtxCreate(&context, 0, device);
    }

    void TearDown() override
    {
        cuCtxDestroy(context);
    }

    CUcontext context;
};

TEST_F(Phase5TransposeBenchmark, CompareLayouts)
{
    int M = 1024;
    int N = 896;
    int K = 896;
    int K_blocks = (K + 31) / 32; // 28 blocks
    const int ITERS = 20;

    std::cout << "\n========================================\n";
    std::cout << "Phase 5 Transpose B Layout Benchmark\n";
    std::cout << "========================================\n";
    std::cout << "Problem size: M=" << M << ", N=" << N << ", K=" << K << "\n";
    std::cout << "K blocks: " << K_blocks << " (18 bytes each)\n";
    std::cout << "Total B size: " << (N * K_blocks * 18 / 1024.0) << " KB\n\n";

    // Create configurations
    CudaGemmConfigPhase5 config_rowmajor(
        64, 64, 64, // tile_m, tile_n, tile_k
        64,         // sub_k (NCU-validated optimal)
        2, 2,       // mma_m, mma_n
        2,          // buffer_stages
        128,        // threads_per_block
        3, 3, 3,    // swizzle
        8,          // vectorize_a
        false);     // transpose_b = false (row-major)

    CudaGemmConfigPhase5 config_transposed(
        64, 64, 64, // tile_m, tile_n, tile_k
        64,         // sub_k
        2, 2,       // mma_m, mma_n
        2,          // buffer_stages
        128,        // threads_per_block
        3, 3, 3,    // swizzle
        8,          // vectorize_a
        true);      // transpose_b = TRUE (transposed layout!)

    std::cout << "Config row-major: " << config_rowmajor.config_id() << "\n";
    std::cout << "Config transposed: " << config_transposed.config_id() << "\n\n";

    // Compile kernels
    std::cout << "Compiling kernels...\n";
    auto &jit = CudaGemmJITPhase5::instance();

    CUfunction kernel_rowmajor = jit.getKernel(config_rowmajor);
    CUfunction kernel_transposed = jit.getKernel(config_transposed);

    std::cout << "✓ Both kernels compiled successfully\n\n";

    // Allocate memory
    size_t size_A = M * K * sizeof(float);
    size_t size_B = N * K_blocks * sizeof(IQ4_NLBlock); // 18 bytes per block
    size_t size_C = M * N * sizeof(float);

    float *d_A;
    IQ4_NLBlock *d_B_rowmajor, *d_B_transposed;
    float *d_C;

    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B_rowmajor, size_B);
    cudaMalloc(&d_B_transposed, size_B);
    cudaMalloc(&d_C, size_C);

    // Initialize data
    std::vector<float> h_A(M * K, 1.0f);
    std::vector<IQ4_NLBlock> h_B(N * K_blocks);

    // Simple initialization
    for (auto &block : h_B)
    {
        block.d = 0x3C00; // FP16: 1.0
        for (int i = 0; i < 8; i++)
        {
            block.qs[i] = 0x8888; // All values = 8 (maps to ~0.01)
        }
    }

    cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_rowmajor, h_B.data(), size_B, cudaMemcpyHostToDevice);

    // Transpose B matrix: [N][K_blocks] → [K_blocks][N]
    std::vector<IQ4_NLBlock> h_B_transposed(N * K_blocks);
    for (int n = 0; n < N; n++)
    {
        for (int k = 0; k < K_blocks; k++)
        {
            int src_idx = n * K_blocks + k; // Row-major: [n][k]
            int dst_idx = k * N + n;        // Transposed: [k][n]
            h_B_transposed[dst_idx] = h_B[src_idx];
        }
    }
    cudaMemcpy(d_B_transposed, h_B_transposed.data(), size_B, cudaMemcpyHostToDevice);

    // Grid/block dimensions
    dim3 block(config_rowmajor.threads_per_block);
    dim3 grid(
        (M + config_rowmajor.tile_m - 1) / config_rowmajor.tile_m,
        (N + config_rowmajor.tile_n - 1) / config_rowmajor.tile_n);

    std::cout << "Grid: (" << grid.x << ", " << grid.y << "), Block: " << block.x << "\n\n";

    // ========== Test 1: Row-Major B (baseline) ==========
    std::cout << "========================================\n";
    std::cout << "Test 1: Row-Major B [N][K_blocks]\n";
    std::cout << "========================================\n";
    std::cout << "Addressing: B[gn * K_BLOCKS_TOTAL + gk_block]\n";
    std::cout << "Coalescing: Poor (adjacent threads → different rows)\n\n";

    void *args_rowmajor[] = {&d_A, &d_B_rowmajor, &d_C, &M, &N, &K};

    // Warmup
    for (int i = 0; i < 5; i++)
    {
        cuLaunchKernel(kernel_rowmajor, grid.x, grid.y, grid.z,
                       block.x, block.y, block.z, 0, nullptr, args_rowmajor, nullptr);
    }
    cudaDeviceSynchronize();

    // Benchmark
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERS; i++)
    {
        cuLaunchKernel(kernel_rowmajor, grid.x, grid.y, grid.z,
                       block.x, block.y, block.z, 0, nullptr, args_rowmajor, nullptr);
    }
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms_rowmajor = std::chrono::duration<double, std::milli>(t1 - t0).count() / ITERS;
    double ops_rowmajor = 2.0 * M * N * K; // multiply-add per element
    double tflops_rowmajor = (ops_rowmajor / 1e12) / (ms_rowmajor / 1e3);

    std::cout << "Time:   " << std::fixed << std::setprecision(2) << ms_rowmajor << " ms\n";
    std::cout << "TFLOPS: " << tflops_rowmajor << "\n\n";

    // ========== Test 2: Transposed B (optimized) ==========
    std::cout << "========================================\n";
    std::cout << "Test 2: Transposed B [K_blocks][N]\n";
    std::cout << "========================================\n";
    std::cout << "Addressing: B[gk_block * N + gn]\n";
    std::cout << "Coalescing: Good (adjacent threads → adjacent blocks)\n\n";

    cudaMemset(d_C, 0, size_C); // Clear output
    void *args_transposed[] = {&d_A, &d_B_transposed, &d_C, &M, &N, &K};

    // Warmup
    for (int i = 0; i < 5; i++)
    {
        cuLaunchKernel(kernel_transposed, grid.x, grid.y, grid.z,
                       block.x, block.y, block.z, 0, nullptr, args_transposed, nullptr);
    }
    cudaDeviceSynchronize();

    // Benchmark
    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERS; i++)
    {
        cuLaunchKernel(kernel_transposed, grid.x, grid.y, grid.z,
                       block.x, block.y, block.z, 0, nullptr, args_transposed, nullptr);
    }
    cudaDeviceSynchronize();
    t1 = std::chrono::high_resolution_clock::now();

    double ms_transposed = std::chrono::duration<double, std::milli>(t1 - t0).count() / ITERS;
    double tflops_transposed = (ops_rowmajor / 1e12) / (ms_transposed / 1e3);

    std::cout << "Time:   " << std::fixed << std::setprecision(2) << ms_transposed << " ms\n";
    std::cout << "TFLOPS: " << tflops_transposed << "\n\n";

    // ========== Compare ==========
    double speedup = tflops_transposed / tflops_rowmajor;
    double improvement_pct = (speedup - 1.0) * 100.0;

    std::cout << "========================================\n";
    std::cout << "Performance Summary\n";
    std::cout << "========================================\n";
    std::cout << "Row-major:   " << std::fixed << std::setprecision(2) << tflops_rowmajor << " TFLOPS\n";
    std::cout << "Transposed:  " << tflops_transposed << " TFLOPS\n";
    std::cout << "Speedup:     " << std::setprecision(3) << speedup << "×\n";
    std::cout << "Improvement: " << std::showpos << std::setprecision(1) << improvement_pct << std::noshowpos << "%\n";
    std::cout << "========================================\n\n";

    std::cout << "NCU Prediction: 62% speedup potential\n";
    std::cout << "Actual result:  " << std::showpos << improvement_pct << std::noshowpos << "%\n\n";

    if (speedup > 1.5)
    {
        std::cout << "✅ EXCELLENT: Transpose provides major performance boost!\n";
    }
    else if (speedup > 1.3)
    {
        std::cout << "✅ GOOD: Significant improvement from transpose.\n";
    }
    else if (speedup > 1.1)
    {
        std::cout << "⚠️  MODERATE: Some improvement, but less than expected.\n";
    }
    else
    {
        std::cout << "❌ MINIMAL: Transpose didn't help as expected.\n";
    }

    std::cout << "\nNext steps:\n";
    std::cout << "1. If speedup > 1.3×: Adopt transposed layout permanently\n";
    std::cout << "2. Run NCU profile to confirm coalescing metrics\n";
    std::cout << "3. Address shared memory bank conflicts (63% additional potential)\n";

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B_rowmajor);
    cudaFree(d_B_transposed);
    cudaFree(d_C);

    // Assertion
    EXPECT_GT(speedup, 1.2) << "Expected at least 20% improvement from transpose";
}
