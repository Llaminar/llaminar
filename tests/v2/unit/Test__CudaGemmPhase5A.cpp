/**
 * @file Test__CudaGemmPhase5A.cpp
 * @brief Tests for Phase 5A streaming dequantization
 *
 * Expected: 10.0+ TFLOPS at M=1024 (+15% over Phase 4's 8.69 TFLOPS)
 *
 * Validates:
 * 1. Correctness vs CPU reference
 * 2. Correctness vs Phase 4 (should be identical)
 * 3. Performance gain from streaming (+15-25%)
 *
 * @author David Sanftenberg
 * @date November 4, 2025
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <random>
#include <iostream>
#include <iomanip>

// IQ4_NLBlock definition (must be before kernel includes)
struct IQ4_NLBlock
{
    float scale;
    uint8_t quants[16];
};

#include "v2/kernels/cuda/CudaGemmKernelPhase4QuickWins.h"
#include "v2/kernels/cuda/CudaGemmKernelPhase5ASimple.h"

// Helper: CPU reference GEMM
void cpu_gemm_reference(
    const std::vector<float> &A,
    const std::vector<float> &B,
    std::vector<float> &C,
    int M, int N, int K)
{
    for (int m = 0; m < M; m++)
    {
        for (int n = 0; n < N; n++)
        {
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
            {
                sum += A[m * K + k] * B[n * K + k];
            }
            C[m * N + n] = sum;
        }
    }
}

// IQ4_NL quantization values (same as kernel)
const float iq4nl_values[16] = {
    -127.0f / 127.0f, -104.0f / 127.0f, -83.0f / 127.0f, -65.0f / 127.0f,
    -49.0f / 127.0f, -35.0f / 127.0f, -22.0f / 127.0f, -10.0f / 127.0f,
    0.0f, 10.0f / 127.0f, 22.0f / 127.0f, 35.0f / 127.0f,
    49.0f / 127.0f, 65.0f / 127.0f, 83.0f / 127.0f, 104.0f / 127.0f};

// Helper: Quantize float matrix to IQ4_NL blocks
std::vector<IQ4_NLBlock> quantize_to_iq4nl(
    const std::vector<float> &B,
    int N, int K)
{
    int blocks_per_row = K / 32;
    std::vector<IQ4_NLBlock> blocks(N * blocks_per_row);

    for (int n = 0; n < N; n++)
    {
        for (int kb = 0; kb < blocks_per_row; kb++)
        {
            float absmax = 0.0f;
            for (int i = 0; i < 32; i++)
            {
                int k = kb * 32 + i;
                absmax = std::max(absmax, std::abs(B[n * K + k]));
            }

            float scale = absmax / 127.0f;
            blocks[n * blocks_per_row + kb].scale = scale;

            for (int i = 0; i < 16; i++)
            {
                int k0 = kb * 32 + i * 2;
                int k1 = kb * 32 + i * 2 + 1;

                float val0 = B[n * K + k0];
                float val1 = B[n * K + k1];

                int q0 = std::min(15, std::max(0,
                                               (int)std::round((val0 / scale) * 127.0f / 127.0f * 7.0f) + 7));
                int q1 = std::min(15, std::max(0,
                                               (int)std::round((val1 / scale) * 127.0f / 127.0f * 7.0f) + 7));

                blocks[n * blocks_per_row + kb].quants[i] = (q1 << 4) | q0;
            }
        }
    }

    return blocks;
}

// Helper: Dequantize IQ4_NL blocks back to float
std::vector<float> dequantize_iq4nl(
    const std::vector<IQ4_NLBlock> &blocks,
    int N, int K)
{
    std::vector<float> B(N * K);
    int blocks_per_row = K / 32;

    for (int n = 0; n < N; n++)
    {
        for (int kb = 0; kb < blocks_per_row; kb++)
        {
            const IQ4_NLBlock &block = blocks[n * blocks_per_row + kb];
            float scale = block.scale;

            for (int i = 0; i < 16; i++)
            {
                uint8_t q = block.quants[i];
                int q0 = q & 0xF;
                int q1 = q >> 4;

                int k0 = kb * 32 + i * 2;
                int k1 = kb * 32 + i * 2 + 1;

                B[n * K + k0] = scale * iq4nl_values[q0];
                B[n * K + k1] = scale * iq4nl_values[q1];
            }
        }
    }

    return B;
}

// Test: Correctness vs CPU reference
TEST(Test__CudaGemmPhase5A, CorrectnessVsCPU)
{
    const int M = 256;
    const int N = 256;
    const int K = 896;

    // Generate random matrices
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> A(M * K);
    std::vector<float> B_fp32(N * K);
    for (auto &x : A)
        x = dist(gen);
    for (auto &x : B_fp32)
        x = dist(gen);

    // Quantize B to IQ4_NL
    auto B_blocks = quantize_to_iq4nl(B_fp32, N, K);
    auto B_dequant = dequantize_iq4nl(B_blocks, N, K);

    // CPU reference
    std::vector<float> C_ref(M * N, 0.0f);
    cpu_gemm_reference(A, B_dequant, C_ref, M, N, K);

    // Allocate GPU memory
    float *d_A, *d_C;
    IQ4_NLBlock *d_B_blocks;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B_blocks, B_blocks.size() * sizeof(IQ4_NLBlock));
    cudaMalloc(&d_C, M * N * sizeof(float));

    cudaMemcpy(d_A, A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_blocks, B_blocks.data(),
               B_blocks.size() * sizeof(IQ4_NLBlock), cudaMemcpyHostToDevice);

    // Launch Phase 5A kernel
    launch_iq4nl_gemm_phase5a(d_A, d_B_blocks, d_C, M, N, K);
    cudaDeviceSynchronize();

    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess) << "Kernel launch failed: " << cudaGetErrorString(err);

    // Copy result back
    std::vector<float> C_gpu(M * N);
    cudaMemcpy(C_gpu.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify correctness (allow small numerical error)
    int mismatches = 0;
    float max_error = 0.0f;
    for (int i = 0; i < M * N; i++)
    {
        float error = std::abs(C_gpu[i] - C_ref[i]);
        float rel_error = error / (std::abs(C_ref[i]) + 1e-6f);
        max_error = std::max(max_error, rel_error);

        if (rel_error > 0.01f)
        { // 1% tolerance
            mismatches++;
            if (mismatches <= 5)
            {
                std::cout << "Mismatch at [" << i / N << "," << i % N << "]: "
                          << "GPU=" << C_gpu[i] << ", CPU=" << C_ref[i]
                          << ", error=" << rel_error << std::endl;
            }
        }
    }

    std::cout << "Max relative error: " << max_error << std::endl;
    std::cout << "Mismatches: " << mismatches << " / " << M * N << std::endl;

    EXPECT_LT(mismatches, M * N * 0.001); // <0.1% mismatches
    EXPECT_LT(max_error, 0.05f);          // <5% max error

    cudaFree(d_A);
    cudaFree(d_B_blocks);
    cudaFree(d_C);
}

// Test: Phase 5A vs Phase 4 (should be identical)
TEST(Test__CudaGemmPhase5A, ParityVsPhase4)
{
    const int M = 512;
    const int N = 512;
    const int K = 896;

    std::mt19937 gen(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> A(M * K);
    std::vector<float> B_fp32(N * K);
    for (auto &x : A)
        x = dist(gen);
    for (auto &x : B_fp32)
        x = dist(gen);

    auto B_blocks = quantize_to_iq4nl(B_fp32, N, K);

    // Allocate GPU memory
    float *d_A, *d_C_p4, *d_C_p5a;
    IQ4_NLBlock *d_B_blocks;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B_blocks, B_blocks.size() * sizeof(IQ4_NLBlock));
    cudaMalloc(&d_C_p4, M * N * sizeof(float));
    cudaMalloc(&d_C_p5a, M * N * sizeof(float));

    cudaMemcpy(d_A, A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_blocks, B_blocks.data(),
               B_blocks.size() * sizeof(IQ4_NLBlock), cudaMemcpyHostToDevice);

    // Launch Phase 4
    launch_iq4nl_gemm_phase4(d_A, d_B_blocks, d_C_p4, M, N, K);

    // Launch Phase 5A
    launch_iq4nl_gemm_phase5a(d_A, d_B_blocks, d_C_p5a, M, N, K);

    cudaDeviceSynchronize();

    // Compare results
    std::vector<float> C_p4(M * N);
    std::vector<float> C_p5a(M * N);
    cudaMemcpy(C_p4.data(), d_C_p4, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(C_p5a.data(), d_C_p5a, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    int mismatches = 0;
    float max_diff = 0.0f;
    for (int i = 0; i < M * N; i++)
    {
        float diff = std::abs(C_p5a[i] - C_p4[i]);
        max_diff = std::max(max_diff, diff);

        if (diff > 1e-4f)
        {
            mismatches++;
            if (mismatches <= 5)
            {
                std::cout << "Difference at [" << i / N << "," << i % N << "]: "
                          << "P4=" << C_p4[i] << ", P5A=" << C_p5a[i]
                          << ", diff=" << diff << std::endl;
            }
        }
    }

    std::cout << "Phase 5A vs Phase 4 max diff: " << max_diff << std::endl;
    std::cout << "Mismatches: " << mismatches << " / " << M * N << std::endl;

    EXPECT_EQ(mismatches, 0); // Should be bit-exact!
    EXPECT_LT(max_diff, 1e-4f);

    cudaFree(d_A);
    cudaFree(d_B_blocks);
    cudaFree(d_C_p4);
    cudaFree(d_C_p5a);
}

// Test: Performance gain vs Phase 4
TEST(Test__CudaGemmPhase5A, PerformanceVsPhase4)
{
    const int M = 1024;
    const int N = 896;
    const int K = 896;
    const int warmup = 10;
    const int iterations = 50;

    std::mt19937 gen(456);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> A(M * K);
    std::vector<float> B_fp32(N * K);
    for (auto &x : A)
        x = dist(gen);
    for (auto &x : B_fp32)
        x = dist(gen);

    auto B_blocks = quantize_to_iq4nl(B_fp32, N, K);

    // Allocate GPU memory
    float *d_A, *d_C;
    IQ4_NLBlock *d_B_blocks;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B_blocks, B_blocks.size() * sizeof(IQ4_NLBlock));
    cudaMalloc(&d_C, M * N * sizeof(float));

    cudaMemcpy(d_A, A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_blocks, B_blocks.data(),
               B_blocks.size() * sizeof(IQ4_NLBlock), cudaMemcpyHostToDevice);

    // ===== Benchmark Phase 4 =====
    for (int i = 0; i < warmup; i++)
    {
        launch_iq4nl_gemm_phase4(d_A, d_B_blocks, d_C, M, N, K);
    }
    cudaDeviceSynchronize();

    cudaEvent_t start_p4, stop_p4;
    cudaEventCreate(&start_p4);
    cudaEventCreate(&stop_p4);

    cudaEventRecord(start_p4);
    for (int i = 0; i < iterations; i++)
    {
        launch_iq4nl_gemm_phase4(d_A, d_B_blocks, d_C, M, N, K);
    }
    cudaEventRecord(stop_p4);
    cudaEventSynchronize(stop_p4);

    float time_p4_ms;
    cudaEventElapsedTime(&time_p4_ms, start_p4, stop_p4);
    float avg_time_p4 = time_p4_ms / iterations;

    // ===== Benchmark Phase 5A =====
    for (int i = 0; i < warmup; i++)
    {
        launch_iq4nl_gemm_phase5a(d_A, d_B_blocks, d_C, M, N, K);
    }
    cudaDeviceSynchronize();

    cudaEvent_t start_p5a, stop_p5a;
    cudaEventCreate(&start_p5a);
    cudaEventCreate(&stop_p5a);

    cudaEventRecord(start_p5a);
    for (int i = 0; i < iterations; i++)
    {
        launch_iq4nl_gemm_phase5a(d_A, d_B_blocks, d_C, M, N, K);
    }
    cudaEventRecord(stop_p5a);
    cudaEventSynchronize(stop_p5a);

    float time_p5a_ms;
    cudaEventElapsedTime(&time_p5a_ms, start_p5a, stop_p5a);
    float avg_time_p5a = time_p5a_ms / iterations;

    // ===== Calculate Performance =====
    double ops = 2.0 * M * N * K; // FMA = 2 ops
    double tflops_p4 = (ops / (avg_time_p4 * 1e-3)) / 1e12;
    double tflops_p5a = (ops / (avg_time_p5a * 1e-3)) / 1e12;
    double speedup = tflops_p5a / tflops_p4;

    std::cout << "\n========== Performance Comparison ==========\n";
    std::cout << "Matrix: " << M << "×" << N << "×" << K << "\n";
    std::cout << "Iterations: " << iterations << "\n\n";

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Phase 4 (swizzle):\n";
    std::cout << "  Time:       " << avg_time_p4 << " ms\n";
    std::cout << "  Throughput: " << tflops_p4 << " TFLOPS\n\n";

    std::cout << "Phase 5A (streaming):\n";
    std::cout << "  Time:       " << avg_time_p5a << " ms\n";
    std::cout << "  Throughput: " << tflops_p5a << " TFLOPS\n\n";

    std::cout << "Speedup: " << std::setprecision(2) << speedup << "× ";
    std::cout << "(" << std::setprecision(1) << (speedup - 1.0) * 100 << "% gain)\n";
    std::cout << "===========================================\n";

    // Expected: +15-25% gain (1.15-1.25× speedup)
    EXPECT_GT(speedup, 1.10);             // At least +10% (conservative)
    EXPECT_LT(avg_time_p5a, avg_time_p4); // Should be faster

    cudaEventDestroy(start_p4);
    cudaEventDestroy(stop_p4);
    cudaEventDestroy(start_p5a);
    cudaEventDestroy(stop_p5a);

    cudaFree(d_A);
    cudaFree(d_B_blocks);
    cudaFree(d_C);
}
