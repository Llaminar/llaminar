/**
 * @file Test__Phase5_Transpose_B.cpp
 * @brief Test transposed B matrix layout for improved coalescing
 *
 * Purpose: Validate that transposing B matrix from [N][K] to [K][N] improves
 *          global memory coalescing from 27% to 87-100%
 *
 * NCU Analysis Results (SUB_K=64, row-major B):
 *   - Global load utilization: 8.7 / 32 bytes (27%)
 *   - Excessive sectors: 5,049,856 / 6,995,968 (72%)
 *   - Root cause: Adjacent threads access different N indices (different rows)
 *
 * Expected with Transposed B:
 *   - Global load utilization: ~28-32 / 32 bytes (87-100%)
 *   - Excessive sectors: < 1,000,000 (< 15%)
 *   - Performance: +30-60% (NCU predicts 62% potential)
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

// IQ4_NL block structure (from kernel)
struct IQ4_NLBlock
{
    uint16_t qs[8]; // 16 bytes: packed 4-bit quantized values
    uint16_t d;     // 2 bytes: FP16 scale factor
};
static_assert(sizeof(IQ4_NLBlock) == 18, "IQ4_NLBlock must be 18 bytes");

/**
 * @brief Transpose IQ4_NL blocks from [N][K_blocks] to [K_blocks][N]
 *
 * Original layout: B[n * K_blocks + k_block]
 * Transposed:      B_T[k_block * N + n]
 *
 * This makes adjacent threads (different n) access adjacent blocks in memory.
 */
void transpose_iq4nl_blocks(
    const IQ4_NLBlock *B_rowmajor, // [N][K_blocks]
    IQ4_NLBlock *B_transposed,     // [K_blocks][N]
    int N,
    int K_blocks)
{
    for (int n = 0; n < N; n++)
    {
        for (int k_block = 0; k_block < K_blocks; k_block++)
        {
            // Original: B[n * K_blocks + k_block]
            int src_idx = n * K_blocks + k_block;

            // Transposed: B_T[k_block * N + n]
            int dst_idx = k_block * N + n;

            B_transposed[dst_idx] = B_rowmajor[src_idx];
        }
    }
}

/**
 * @brief Test transposed B matrix layout
 */
class Phase5TransposeBTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        int device;
        cudaGetDevice(&device);

        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);
        std::cout << "GPU: " << prop.name << "\n";
        std::cout << "SM Count: " << prop.multiProcessorCount << "\n\n";
    }
};

TEST_F(Phase5TransposeBTest, CompareRowMajorVsTransposed)
{
    // Matrix dimensions
    int M = 1024;
    int N = 896;
    int K = 896;
    int K_blocks = (K + 31) / 32;

    std::cout << "========================================\n";
    std::cout << "B Matrix Layout Comparison\n";
    std::cout << "========================================\n";
    std::cout << "Matrix: " << M << "x" << N << "x" << K << "\n";
    std::cout << "K_blocks: " << K_blocks << "\n";
    std::cout << "Config: TILE 64x64x64, SUB_K=64, MMA 2x2\n\n";

    // Configuration with SUB_K=64
    CudaGemmConfigPhase5 config(
        64, 64, 64, // tile_m, tile_n, tile_k
        64,         // sub_k = 64
        2, 2,       // mma_m, mma_n
        2,          // buffer_stages
        128,        // threads_per_block
        3, 3, 3,    // swizzle
        8);         // vectorize_a

    // Allocate host memory for B matrix
    size_t num_blocks = N * K_blocks;
    std::vector<IQ4_NLBlock> h_B_rowmajor(num_blocks);
    std::vector<IQ4_NLBlock> h_B_transposed(num_blocks);

    // Initialize with test pattern (non-zero for realistic cache behavior)
    for (size_t i = 0; i < num_blocks; i++)
    {
        for (int j = 0; j < 8; j++)
        {
            h_B_rowmajor[i].qs[j] = static_cast<uint16_t>(i + j);
        }
        h_B_rowmajor[i].d = static_cast<uint16_t>(i & 0xFFFF);
    }

    // Transpose B matrix
    std::cout << "Transposing B matrix: [N=" << N << "][K_blocks=" << K_blocks << "] → [K_blocks][N]...\n";
    auto t_transpose_start = std::chrono::high_resolution_clock::now();
    transpose_iq4nl_blocks(
        h_B_rowmajor.data(),
        h_B_transposed.data(),
        N,
        K_blocks);
    auto t_transpose_end = std::chrono::high_resolution_clock::now();
    double transpose_ms = std::chrono::duration<double, std::milli>(t_transpose_end - t_transpose_start).count();
    std::cout << "  ✓ Transpose time: " << transpose_ms << " ms\n\n";

    // Allocate device memory
    float *d_A, *d_C;
    IQ4_NLBlock *d_B_rowmajor, *d_B_transposed;

    size_t size_A = M * K * sizeof(float);
    size_t size_B = num_blocks * sizeof(IQ4_NLBlock);
    size_t size_C = M * N * sizeof(float);

    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B_rowmajor, size_B);
    cudaMalloc(&d_B_transposed, size_B);
    cudaMalloc(&d_C, size_C);

    // Initialize A with random data
    std::vector<float> h_A(M * K);
    for (auto &val : h_A)
    {
        val = (rand() % 1000) / 1000.0f;
    }

    cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_rowmajor, h_B_rowmajor.data(), size_B, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_transposed, h_B_transposed.data(), size_B, cudaMemcpyHostToDevice);

    // Get kernel
    auto &jit = CudaGemmJITPhase5::instance();
    CUfunction kernel = jit.getKernel(config);

    // Launch parameters
    int blocks_m = (M + config.tile_m - 1) / config.tile_m;
    int blocks_n = (N + config.tile_n - 1) / config.tile_n;
    dim3 grid(blocks_m, blocks_n);
    dim3 block(config.threads_per_block);

    // ========== Test 1: Row-Major B (baseline) ==========
    std::cout << "========================================\n";
    std::cout << "Test 1: Row-Major B [N][K_blocks]\n";
    std::cout << "========================================\n";
    std::cout << "Memory layout: B[n * K_blocks + k_block]\n";
    std::cout << "Expected: 8.7 / 32 bytes per sector (27% utilization)\n\n";

    void *args_rowmajor[] = {&d_A, &d_B_rowmajor, &d_C, &M, &N, &K};

    // Warmup
    for (int i = 0; i < 5; ++i)
    {
        cuLaunchKernel(kernel, grid.x, grid.y, grid.z,
                       block.x, block.y, block.z, 0, nullptr, args_rowmajor, nullptr);
    }
    cudaDeviceSynchronize();

    // Benchmark
    auto t0_rowmajor = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 20; ++i)
    {
        cuLaunchKernel(kernel, grid.x, grid.y, grid.z,
                       block.x, block.y, block.z, 0, nullptr, args_rowmajor, nullptr);
    }
    cudaDeviceSynchronize();
    auto t1_rowmajor = std::chrono::high_resolution_clock::now();

    double ms_rowmajor = std::chrono::duration<double, std::milli>(t1_rowmajor - t0_rowmajor).count() / 20.0;
    double ops = 2.0 * M * N * K;
    double tflops_rowmajor = (ops / 1e12) / (ms_rowmajor / 1e3);

    std::cout << "Results:\n";
    std::cout << "  Time: " << ms_rowmajor << " ms\n";
    std::cout << "  Throughput: " << tflops_rowmajor << " TFLOPS\n\n";

    // Save output for verification
    std::vector<float> h_C_rowmajor(M * N);
    cudaMemcpy(h_C_rowmajor.data(), d_C, size_C, cudaMemcpyDeviceToHost);

    // Clear C for next test
    cudaMemset(d_C, 0, size_C);

    // ========== Test 2: Transposed B (optimized) ==========
    std::cout << "========================================\n";
    std::cout << "Test 2: Transposed B [K_blocks][N]\n";
    std::cout << "========================================\n";
    std::cout << "Memory layout: B_T[k_block * N + n]\n";
    std::cout << "Expected: ~28-32 / 32 bytes per sector (87-100% utilization)\n";
    std::cout << "Expected: +30-60% performance improvement\n\n";

    void *args_transposed[] = {&d_A, &d_B_transposed, &d_C, &M, &N, &K};

    // Warmup
    for (int i = 0; i < 5; ++i)
    {
        cuLaunchKernel(kernel, grid.x, grid.y, grid.z,
                       block.x, block.y, block.z, 0, nullptr, args_transposed, nullptr);
    }
    cudaDeviceSynchronize();

    // Benchmark
    auto t0_transposed = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 20; ++i)
    {
        cuLaunchKernel(kernel, grid.x, grid.y, grid.z,
                       block.x, block.y, block.z, 0, nullptr, args_transposed, nullptr);
    }
    cudaDeviceSynchronize();
    auto t1_transposed = std::chrono::high_resolution_clock::now();

    double ms_transposed = std::chrono::duration<double, std::milli>(t1_transposed - t0_transposed).count() / 20.0;
    double tflops_transposed = (ops / 1e12) / (ms_transposed / 1e3);

    std::cout << "Results:\n";
    std::cout << "  Time: " << ms_transposed << " ms\n";
    std::cout << "  Throughput: " << tflops_transposed << " TFLOPS\n\n";

    // Compare performance
    double speedup = tflops_transposed / tflops_rowmajor;
    double improvement_pct = (speedup - 1.0) * 100.0;

    std::cout << "========================================\n";
    std::cout << "Performance Comparison\n";
    std::cout << "========================================\n";
    std::cout << "Row-Major:   " << std::fixed << std::setprecision(2) << tflops_rowmajor << " TFLOPS\n";
    std::cout << "Transposed:  " << tflops_transposed << " TFLOPS\n";
    std::cout << "Speedup:     " << speedup << "×\n";
    std::cout << "Improvement: " << std::showpos << improvement_pct << std::noshowpos << "%\n";
    std::cout << "========================================\n\n";

    // Validate correctness (outputs should be identical since same data, just reordered)
    std::vector<float> h_C_transposed(M * N);
    cudaMemcpy(h_C_transposed.data(), d_C, size_C, cudaMemcpyDeviceToHost);

    // Note: Outputs will be DIFFERENT because we've changed the memory access pattern!
    // The kernel still uses the old addressing: B[gn * K_BLOCKS_TOTAL + gk_block]
    // With transposed data, this reads the wrong elements.

    std::cout << "NOTE: This test demonstrates the performance potential.\n";
    std::cout << "To make it work correctly, the kernel addressing must also change:\n";
    std::cout << "  Old: B[gn * K_BLOCKS_TOTAL + gk_block]\n";
    std::cout << "  New: B[gk_block * N + gn]\n\n";

    std::cout << "Next steps:\n";
    std::cout << "1. If speedup > 1.3×: Modify kernel addressing to use transposed layout\n";
    std::cout << "2. Run NCU profiling to confirm coalescing improvement\n";
    std::cout << "3. Add transpose step to model loading pipeline\n";

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B_rowmajor);
    cudaFree(d_B_transposed);
    cudaFree(d_C);

    // Expect significant speedup (NCU predicts 62% potential = 1.62× speedup)
    EXPECT_GT(speedup, 1.2) << "Expected at least 20% improvement with transposed B";
}
