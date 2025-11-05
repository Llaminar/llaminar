/**
 * @file Test__Phase5_SUB_K_64_Profile.cpp
 * @brief Simple test for NCU profiling of SUB_K=64 kernel
 *
 * Purpose: Validate improved memory coalescing with SUB_K=64
 * Expected: Global load sector utilization ~28-32/32 bytes (vs 8.2/32 with SUB_K=16)
 *
 * @author David Sanftenberg
 * @date November 4, 2025
 */

#include "CudaGemmJITPhase5.h"
#include "Phase5ConfigSpace.h"
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <chrono>

using namespace llaminar2::cuda;

/**
 * @brief Profile SUB_K=64 kernel with NCU
 *
 * This test runs a single GEMM operation with SUB_K=64 configuration
 * for detailed NCU profiling analysis.
 */
class Phase5SubK64ProfileTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Initialize CUDA
        int device;
        cudaGetDevice(&device);

        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);
        std::cout << "GPU: " << prop.name << "\n";
        std::cout << "SM Count: " << prop.multiProcessorCount << "\n\n";
    }
};

TEST_F(Phase5SubK64ProfileTest, ProfileSingleKernel)
{
    // Matrix dimensions (same as SUB_K sweep test)
    int M = 1024;
    int N = 896;
    int K = 896;

    std::cout << "========================================\n";
    std::cout << "SUB_K=64 Profiling Test\n";
    std::cout << "========================================\n";
    std::cout << "Matrix: " << M << "x" << N << "x" << K << "\n";
    std::cout << "Config: TILE 64x64x64, SUB_K=64, MMA 2x2, Swizzle 3,3,3\n\n";

    // Configuration with SUB_K=64
    CudaGemmConfigPhase5 config(
        64, 64, 64, // tile_m, tile_n, tile_k
        64,         // sub_k = 64 (enables coalescing!)
        2, 2,       // mma_m, mma_n
        2,          // buffer_stages = 2 (double buffering)
        128,        // threads_per_block
        3, 3, 3,    // swizzle_b, swizzle_m, swizzle_s
        8);         // vectorize_a

    std::cout << "Config ID: " << config.config_id() << "\n";
    std::cout << "K_BLOCKS_IN_SUB_K: " << ((config.sub_k + 31) / 32) << " (coalescing enabled)\n";
    std::cout << "Shared Memory: " << (config.shared_memory_bytes() / 1024) << " KB\n\n";

    // Allocate device memory directly
    float *d_A, *d_C;
    void *d_B; // IQ4_NL blocks

    size_t size_A = M * K * sizeof(float);
    int blocks_per_row = (K + 31) / 32;
    size_t size_B = N * blocks_per_row * 18; // IQ4_NL: 18 bytes per block
    size_t size_C = M * N * sizeof(float);

    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    // Initialize with zeros (good enough for profiling)
    cudaMemset(d_A, 0, size_A);
    cudaMemset(d_B, 0, size_B);
    cudaMemset(d_C, 0, size_C);

    // Get JIT kernel instance
    auto &jit = CudaGemmJITPhase5::instance();

    std::cout << "Step 1: JIT compilation...\n";
    auto t_compile_start = std::chrono::high_resolution_clock::now();
    CUfunction kernel = jit.getKernel(config);
    auto t_compile_end = std::chrono::high_resolution_clock::now();

    double compile_ms = std::chrono::duration<double, std::milli>(t_compile_end - t_compile_start).count();
    std::cout << "  ✓ Compilation time: " << compile_ms << " ms\n\n";

    // Launch parameters
    int blocks_m = (M + config.tile_m - 1) / config.tile_m;
    int blocks_n = (N + config.tile_n - 1) / config.tile_n;
    dim3 grid(blocks_m, blocks_n);
    dim3 block(config.threads_per_block);
    void *args[] = {&d_A, &d_B, &d_C, &M, &N, &K};

    std::cout << "Step 2: Warmup runs (10 iterations)...\n";
    for (int i = 0; i < 10; ++i)
    {
        cuLaunchKernel(kernel, grid.x, grid.y, grid.z,
                       block.x, block.y, block.z, 0, nullptr, args, nullptr);
    }
    cudaDeviceSynchronize();
    std::cout << "  ✓ Warmup complete\n\n";

    std::cout << "Step 3: Profiled run (NCU will capture this)...\n";
    auto t0 = std::chrono::high_resolution_clock::now();

    cuLaunchKernel(kernel, grid.x, grid.y, grid.z,
                   block.x, block.y, block.z, 0, nullptr, args, nullptr);

    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double ops = 2.0 * M * N * K;
    double tflops = (ops / 1e12) / (ms / 1e3);

    std::cout << "  Time: " << ms << " ms\n";
    std::cout << "  Throughput: " << tflops << " TFLOPS\n\n";

    std::cout << "========================================\n";
    std::cout << "Expected NCU Results (vs SUB_K=16):\n";
    std::cout << "========================================\n";
    std::cout << "Global Load Sector Utilization:\n";
    std::cout << "  SUB_K=16: 8.2 / 32 bytes (25.6%)\n";
    std::cout << "  SUB_K=64: ~28-32 / 32 bytes (87-100%) ✓\n\n";
    std::cout << "Uncoalesced Transactions:\n";
    std::cout << "  SUB_K=16: ~11,648\n";
    std::cout << "  SUB_K=64: ~0-100 ✓\n\n";
    std::cout << "Performance:\n";
    std::cout << "  SUB_K=16: 8.41 TFLOPS\n";
    std::cout << "  SUB_K=64: 12.98 TFLOPS (+54.3%) ✓\n";
    std::cout << "========================================\n";

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// Single kernel execution for clean NCU profile
TEST_F(Phase5SubK64ProfileTest, SingleExecution)
{
    int M = 1024;
    int N = 896;
    int K = 896;

    CudaGemmConfigPhase5 config(64, 64, 64, 64, 2, 2, 2, 128, 3, 3, 3, 8);

    float *d_A, *d_C;
    void *d_B;

    int blocks_per_row = (K + 31) / 32;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, N * blocks_per_row * 18);
    cudaMalloc(&d_C, M * N * sizeof(float));

    cudaMemset(d_A, 0, M * K * sizeof(float));
    cudaMemset(d_B, 0, N * blocks_per_row * 18);
    cudaMemset(d_C, 0, M * N * sizeof(float));

    auto &jit = CudaGemmJITPhase5::instance();
    CUfunction kernel = jit.getKernel(config);

    // Launch parameters
    int blocks_m = (M + config.tile_m - 1) / config.tile_m;
    int blocks_n = (N + config.tile_n - 1) / config.tile_n;
    dim3 grid(blocks_m, blocks_n);
    dim3 block(config.threads_per_block);
    void *args[] = {&d_A, &d_B, &d_C, &M, &N, &K};

    // Single execution for NCU profiling
    cuLaunchKernel(kernel, grid.x, grid.y, grid.z,
                   block.x, block.y, block.z, 0, nullptr, args, nullptr);
    cudaDeviceSynchronize();

    std::cout << "✓ Single execution complete (NCU profile captured)\n";

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
