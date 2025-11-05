/**
 * @file Test__Phase5_Transpose_Large.cpp
 * @brief Large-scale benchmark to test transpose + N-major indexing
 *
 * Uses larger problem size for more accurate timing
 *
 * @author David Sanftenberg
 * @date November 5, 2025
 */

#include <gtest/gtest.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>

#include "kernels/cuda/CudaGemmJITPhase5.h"
#include "kernels/cuda/CudaGemmConfigPhase5.h"

using namespace llaminar2::cuda;

struct IQ4_NLBlock
{
    uint16_t qs[8];
    uint16_t d;
};

class Phase5TransposeLarge : public ::testing::Test
{
protected:
    void SetUp() override
    {
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

TEST_F(Phase5TransposeLarge, LargerProblemSize)
{
    // Larger problem for better timing accuracy
    int M = 2048;
    int N = 2048;
    int K = 2048;
    int K_blocks = (K + 31) / 32; // 64 blocks
    const int ITERS = 50;         // More iterations

    std::cout << "\n========================================\n";
    std::cout << "Large Problem Transpose Benchmark\n";
    std::cout << "========================================\n";
    std::cout << "Size: M=" << M << ", N=" << N << ", K=" << K << "\n";
    std::cout << "K blocks: " << K_blocks << "\n";
    std::cout << "Total B size: " << (N * K_blocks * 18 / 1024.0 / 1024.0) << " MB\n";
    std::cout << "Iterations: " << ITERS << "\n\n";

    CudaGemmConfigPhase5 config_rowmajor(
        64, 64, 64, 64, 2, 2, 2, 128, 3, 3, 3, 8, false);
    CudaGemmConfigPhase5 config_transposed(
        64, 64, 64, 64, 2, 2, 2, 128, 3, 3, 3, 8, true);

    auto &jit = CudaGemmJITPhase5::instance();
    CUfunction kernel_rowmajor = jit.getKernel(config_rowmajor);
    CUfunction kernel_transposed = jit.getKernel(config_transposed);

    // Allocate
    size_t size_A = M * K * sizeof(float);
    size_t size_B = N * K_blocks * sizeof(IQ4_NLBlock);
    size_t size_C = M * N * sizeof(float);

    float *d_A;
    IQ4_NLBlock *d_B_rowmajor, *d_B_transposed;
    float *d_C;

    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B_rowmajor, size_B);
    cudaMalloc(&d_B_transposed, size_B);
    cudaMalloc(&d_C, size_C);

    // Initialize
    std::vector<float> h_A(M * K, 0.01f);
    std::vector<IQ4_NLBlock> h_B(N * K_blocks);
    for (auto &block : h_B)
    {
        block.d = 0x3C00; // FP16: 1.0
        for (int i = 0; i < 8; i++)
        {
            block.qs[i] = 0x8888;
        }
    }

    cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_rowmajor, h_B.data(), size_B, cudaMemcpyHostToDevice);

    // Transpose
    std::vector<IQ4_NLBlock> h_B_transposed(N * K_blocks);
    for (int n = 0; n < N; n++)
    {
        for (int k = 0; k < K_blocks; k++)
        {
            h_B_transposed[k * N + n] = h_B[n * K_blocks + k];
        }
    }
    cudaMemcpy(d_B_transposed, h_B_transposed.data(), size_B, cudaMemcpyHostToDevice);

    dim3 block(128);
    dim3 grid((M + 63) / 64, (N + 63) / 64);

    std::cout << "Grid: (" << grid.x << ", " << grid.y << "), Block: " << block.x << "\n\n";

    // Test 1: Row-major
    std::cout << "Test 1: Row-Major (K-major indexing)\n";
    void *args_row[] = {&d_A, &d_B_rowmajor, &d_C, &M, &N, &K};

    for (int i = 0; i < 5; i++)
    {
        cuLaunchKernel(kernel_rowmajor, grid.x, grid.y, grid.z,
                       block.x, block.y, block.z, 0, nullptr, args_row, nullptr);
    }
    cudaDeviceSynchronize();

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERS; i++)
    {
        cuLaunchKernel(kernel_rowmajor, grid.x, grid.y, grid.z,
                       block.x, block.y, block.z, 0, nullptr, args_row, nullptr);
    }
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms_row = std::chrono::duration<double, std::milli>(t1 - t0).count() / ITERS;
    double ops = 2.0 * M * N * K;
    double tflops_row = (ops / 1e12) / (ms_row / 1e3);

    std::cout << "Time:   " << std::fixed << std::setprecision(3) << ms_row << " ms\n";
    std::cout << "TFLOPS: " << std::setprecision(2) << tflops_row << "\n\n";

    // Save output for validation
    std::vector<float> h_C_row(M * N);
    cudaMemcpy(h_C_row.data(), d_C, size_C, cudaMemcpyDeviceToHost);

    // Test 2: Transposed (with N-major indexing)
    std::cout << "Test 2: Transposed (N-major indexing)\n";
    cudaMemset(d_C, 0, size_C);
    void *args_trans[] = {&d_A, &d_B_transposed, &d_C, &M, &N, &K};

    for (int i = 0; i < 5; i++)
    {
        cuLaunchKernel(kernel_transposed, grid.x, grid.y, grid.z,
                       block.x, block.y, block.z, 0, nullptr, args_trans, nullptr);
    }
    cudaDeviceSynchronize();

    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERS; i++)
    {
        cuLaunchKernel(kernel_transposed, grid.x, grid.y, grid.z,
                       block.x, block.y, block.z, 0, nullptr, args_trans, nullptr);
    }
    cudaDeviceSynchronize();
    t1 = std::chrono::high_resolution_clock::now();

    double ms_trans = std::chrono::duration<double, std::milli>(t1 - t0).count() / ITERS;
    double tflops_trans = (ops / 1e12) / (ms_trans / 1e3);

    std::cout << "Time:   " << std::fixed << std::setprecision(3) << ms_trans << " ms\n";
    std::cout << "TFLOPS: " << std::setprecision(2) << tflops_trans << "\n\n";

    // Validate correctness
    std::vector<float> h_C_trans(M * N);
    cudaMemcpy(h_C_trans.data(), d_C, size_C, cudaMemcpyDeviceToHost);

    double max_diff = 0.0;
    int mismatches = 0;
    for (size_t i = 0; i < h_C_row.size(); i++)
    {
        double diff = std::abs(h_C_row[i] - h_C_trans[i]);
        max_diff = std::max(max_diff, diff);
        if (diff > 1e-3)
            mismatches++;
    }

    std::cout << "Correctness validation:\n";
    std::cout << "  Max difference: " << std::scientific << max_diff << "\n";
    std::cout << "  Mismatches (>1e-3): " << mismatches << "/" << h_C_row.size() << "\n";

    if (mismatches == 0)
    {
        std::cout << "  ✓ Results match!\n\n";
    }
    else
    {
        std::cout << "  ✗ Results differ!\n\n";
    }

    // Performance comparison
    double speedup = tflops_trans / tflops_row;
    double improvement = (speedup - 1.0) * 100.0;

    std::cout << "========================================\n";
    std::cout << "Performance Summary\n";
    std::cout << "========================================\n";
    std::cout << "Row-major:   " << std::fixed << std::setprecision(2) << tflops_row << " TFLOPS\n";
    std::cout << "Transposed:  " << tflops_trans << " TFLOPS\n";
    std::cout << "Speedup:     " << std::setprecision(3) << speedup << "×\n";
    std::cout << "Improvement: " << std::showpos << std::setprecision(1) << improvement << std::noshowpos << "%\n";
    std::cout << "========================================\n\n";

    if (speedup > 1.4)
    {
        std::cout << "✅ EXCELLENT: Major improvement from transpose!\n";
    }
    else if (speedup > 1.2)
    {
        std::cout << "✅ GOOD: Significant improvement.\n";
    }
    else if (speedup > 1.05)
    {
        std::cout << "⚠️  MODERATE: Some improvement.\n";
    }
    else
    {
        std::cout << "❌ MINIMAL: No significant benefit.\n";
    }

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B_rowmajor);
    cudaFree(d_B_transposed);
    cudaFree(d_C);

    // Assertions
    EXPECT_EQ(mismatches, 0) << "Results should match exactly";
    EXPECT_GT(speedup, 1.1) << "Expected at least 10% improvement";
}
