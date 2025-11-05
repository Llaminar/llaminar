/**
 * @file Test__Phase6_Int8_Debug.cpp
 * @brief Debug test for Phase 6 kernel - prints intermediate values
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <iomanip>
#include <vector>

#include "kernels/cuda/CudaGemmJITPhase6.h"
#include "kernels/cuda/CudaGemmConfigPhase6.h"

using namespace llaminar2::cuda;

struct IQ4_NLBlock
{
    uint8_t quants[16];
    __half scale;
};

TEST(Phase6Int8Debug, PrintIntermediateValues)
{
    const int M = 32, N = 32, K = 32;
    constexpr int BLOCK_SIZE = 32;
    const int K_BLOCKS = K / BLOCK_SIZE;

    // A = all ones
    std::vector<float> A_host(M * K, 1.0f);

    // B = simple pattern (all blocks with scale=1.0, quants=0x88 -> kvalues[8]=1)
    std::vector<IQ4_NLBlock> B_host(N * K_BLOCKS);
    for (auto &block : B_host)
    {
        block.scale = __float2half(1.0f);
        for (int i = 0; i < 16; i++)
        {
            block.quants[i] = 0x88; // Both nibbles = 8
        }
    }

    // Verify host-side memory layout
    std::cout << "Host-side block verification:\n";
    const uint8_t *block_bytes = (const uint8_t *)&B_host[0];
    std::cout << "  sizeof(IQ4_NLBlock) = " << sizeof(IQ4_NLBlock) << " bytes\n";
    std::cout << "  quants[0] = 0x" << std::hex << (int)block_bytes[0] << std::dec << "\n";
    std::cout << "  quants[15] = 0x" << std::hex << (int)block_bytes[15] << std::dec << "\n";
    std::cout << "  scale bytes [16-17] = 0x" << std::hex << (int)block_bytes[17] << (int)block_bytes[16] << std::dec << "\n";
    std::cout << "  scale (FP32) = " << __half2float(B_host[0].scale) << "\n\n";

    std::cout << "Test setup:\n";
    std::cout << "  M=" << M << ", N=" << N << ", K=" << K << "\n";
    std::cout << "  A: all 1.0\n";
    std::cout << "  B: all blocks with scale=1.0, quants=0x88 (kvalues[8]=1)\n";
    std::cout << "  Expected C[0,0]: ~0.252 (after int8 quantization)\n\n";

    // Allocate GPU memory
    float *d_A, *d_C;
    IQ4_NLBlock *d_B;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, N * K_BLOCKS * sizeof(IQ4_NLBlock));
    cudaMalloc(&d_C, M * N * sizeof(float));

    cudaMemcpy(d_A, A_host.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B_host.data(), N * K_BLOCKS * sizeof(IQ4_NLBlock), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, M * N * sizeof(float));

    // Compile kernel
    auto config = get_default_phase6_config();
    std::cout << "Kernel config:\n";
    std::cout << "  tile_m=" << config.tile_m << ", tile_n=" << config.tile_n
              << ", tile_k=" << config.tile_k << ", threads=" << config.threads_per_block << "\n\n";

    auto kernel = CudaGemmJITPhase6::compile(config);

    // Launch kernel
    dim3 grid((M + config.tile_m - 1) / config.tile_m,
              (N + config.tile_n - 1) / config.tile_n);
    dim3 block(config.threads_per_block);

    std::cout << "Grid: (" << grid.x << ", " << grid.y << "), Block: (" << block.x << ")\n\n";

    void *args[] = {&d_A, &d_B, &d_C, (void *)&M, (void *)&N, (void *)&K};
    cuLaunchKernel(kernel.function, grid.x, grid.y, 1, block.x, block.y, block.z, 0, 0, args, nullptr);

    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess) << "Kernel error: " << cudaGetErrorString(err);

    // Copy result
    std::vector<float> C_gpu(M * N);
    cudaMemcpy(C_gpu.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Results:\n";
    std::cout << "  GPU C[0,0] = " << C_gpu[0] << "\n";
    std::cout << "  GPU C[0,1] = " << C_gpu[1] << "\n";
    std::cout << "  GPU C[1,0] = " << C_gpu[N] << "\n";
    std::cout << "  GPU C[1,1] = " << C_gpu[N + 1] << "\n\n";

    // Print first few rows
    std::cout << "First 4×4 block:\n";
    for (int i = 0; i < 4; i++)
    {
        std::cout << "  [" << i << "]: ";
        for (int j = 0; j < 4; j++)
        {
            std::cout << std::setw(10) << C_gpu[i * N + j] << " ";
        }
        std::cout << "\n";
    }

    std::cout << "\nExpected: all values ~0.252\n";
    std::cout << "Actual: " << C_gpu[0] << " (error: " << (C_gpu[0] - 0.252) / 0.252 * 100 << "%)\n";

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// Test with actual quantization to see what values we get
TEST(Phase6Int8Debug, TestQuantization)
{
    std::cout << "\n=== Testing quantization function ===\n";

    // Test the quantization that the kernel uses
    const int K = 64;
    std::vector<float> input(K, 1.0f);

    std::cout << "Input: " << K << " elements, all 1.0\n";
    std::cout << "Max value: 1.0\n";
    std::cout << "Expected scale: 1.0 / 127 = " << (1.0f / 127.0f) << "\n";
    std::cout << "Expected quantized value: 127\n";
    std::cout << "Expected dequantized: 127 × " << (1.0f / 127.0f) << " = 1.0\n\n";

    // Simulate 32 × 1 dot product with IQ4_NL
    // A: all 1.0 → quantized to 127 with scale 1/127
    // B: kvalues[8] = 1 for all 32 elements, scale = 1.0
    // Result: (127 × 1) × (1/127) × (1.0/127) × 32

    float scale_a = 1.0f / 127.0f;
    int8_t a_val = 127;
    int8_t b_val = 1;              // kvalues[8]
    float scale_b = 1.0f / 127.0f; // B scale normalized

    int32_t acc = 0;
    for (int i = 0; i < 32; i++)
    {
        acc += static_cast<int32_t>(a_val) * static_cast<int32_t>(b_val);
    }

    float result = scale_a * scale_b * static_cast<float>(acc);

    std::cout << "Simulation:\n";
    std::cout << "  scale_a = " << scale_a << "\n";
    std::cout << "  scale_b = " << scale_b << "\n";
    std::cout << "  acc = " << acc << " (32 × 127 × 1)\n";
    std::cout << "  result = " << result << "\n";
    std::cout << "  expected = 0.251969\n";
}
