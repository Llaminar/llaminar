/**
 * @file Test__CudaGemmPhase3Pipelined.cpp
 * @brief Tests for Phase 3 Part 2 pipelined GEMM kernel
 *
 * Compares pipelined vs non-pipelined performance
 * Target: 900-1,000 GFLOPS (1.3-1.4× over Phase 3 Part 1's 695 GFLOPS)
 *
 * @author David Sanftenberg
 * @date November 4, 2025
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <random>

// IQ4_NLBlock definition
struct IQ4_NLBlock
{
    float scale;
    uint8_t quants[16];
};

#include "v2/kernels/cuda/CudaGemmKernelPhase3.h"
#include "v2/kernels/cuda/CudaGemmKernelPhase3Pipelined.h"

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

// Helper: Quantize float matrix to IQ4_NL blocks
std::vector<IQ4_NLBlock> quantize_to_iq4nl(
    const std::vector<float> &B,
    int N, int K)
{
    int blocks_per_row = K / 32;
    std::vector<IQ4_NLBlock> blocks(N * blocks_per_row);

    const float iq4nl_values[16] = {
        -127.0f / 127.0f, -104.0f / 127.0f, -83.0f / 127.0f, -65.0f / 127.0f,
        -49.0f / 127.0f, -35.0f / 127.0f, -22.0f / 127.0f, -10.0f / 127.0f,
        0.0f, 10.0f / 127.0f, 22.0f / 127.0f, 35.0f / 127.0f,
        49.0f / 127.0f, 65.0f / 127.0f, 83.0f / 127.0f, 104.0f / 127.0f};

    for (int n = 0; n < N; n++)
    {
        for (int kb = 0; kb < blocks_per_row; kb++)
        {
            IQ4_NLBlock &block = blocks[n * blocks_per_row + kb];
            int k_start = kb * 32;

            float max_abs = 0.0f;
            for (int i = 0; i < 32; i++)
            {
                float val = B[n * K + k_start + i];
                max_abs = std::max(max_abs, std::abs(val));
            }

            block.scale = (max_abs > 0.0f) ? max_abs : 1.0f;

            for (int i = 0; i < 16; i++)
            {
                float val0 = B[n * K + k_start + i * 2];
                float val1 = B[n * K + k_start + i * 2 + 1];

                float norm0 = val0 / block.scale;
                float norm1 = val1 / block.scale;

                int q0 = 0, q1 = 0;
                float min_err0 = 1e9, min_err1 = 1e9;

                for (int j = 0; j < 16; j++)
                {
                    float err0 = std::abs(norm0 - iq4nl_values[j]);
                    float err1 = std::abs(norm1 - iq4nl_values[j]);
                    if (err0 < min_err0)
                    {
                        min_err0 = err0;
                        q0 = j;
                    }
                    if (err1 < min_err1)
                    {
                        min_err1 = err1;
                        q1 = j;
                    }
                }

                block.quants[i] = (q1 << 4) | q0;
            }
        }
    }

    return blocks;
}

// ============================================================
// TEST: Correctness (Pipelined vs Reference)
// ============================================================

TEST(Test__CudaGemmPhase3Pipelined, CorrectnessVsCPU)
{
    const int M = 128, N = 128, K = 128;

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> A(M * K);
    std::vector<float> B_float(N * K);
    std::vector<float> C_gpu(M * N, 0.0f);
    std::vector<float> C_cpu(M * N, 0.0f);

    for (auto &v : A)
        v = dist(gen);
    for (auto &v : B_float)
        v = dist(gen);

    auto B_blocks = quantize_to_iq4nl(B_float, N, K);

    // Dequantize for CPU reference
    std::vector<float> B_dequant(N * K);
    const float iq4nl_values[16] = {
        -127.0f / 127.0f, -104.0f / 127.0f, -83.0f / 127.0f, -65.0f / 127.0f,
        -49.0f / 127.0f, -35.0f / 127.0f, -22.0f / 127.0f, -10.0f / 127.0f,
        0.0f, 10.0f / 127.0f, 22.0f / 127.0f, 35.0f / 127.0f,
        49.0f / 127.0f, 65.0f / 127.0f, 83.0f / 127.0f, 104.0f / 127.0f};

    for (int n = 0; n < N; n++)
    {
        for (int kb = 0; kb < K / 32; kb++)
        {
            const auto &block = B_blocks[n * (K / 32) + kb];
            for (int i = 0; i < 16; i++)
            {
                uint8_t q = block.quants[i];
                B_dequant[n * K + kb * 32 + i * 2] = block.scale * iq4nl_values[q & 0xF];
                B_dequant[n * K + kb * 32 + i * 2 + 1] = block.scale * iq4nl_values[q >> 4];
            }
        }
    }

    cpu_gemm_reference(A, B_dequant, C_cpu, M, N, K);

    // GPU computation (pipelined)
    float *d_A, *d_C;
    IQ4_NLBlock *d_B;

    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, B_blocks.size() * sizeof(IQ4_NLBlock));
    cudaMalloc(&d_C, M * N * sizeof(float));

    cudaMemcpy(d_A, A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B_blocks.data(), B_blocks.size() * sizeof(IQ4_NLBlock), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, M * N * sizeof(float));

    launch_iq4nl_gemm_pipelined(d_A, d_B, d_C, M, N, K);

    cudaMemcpy(C_gpu.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Compare results
    float max_diff = 0.0f;
    for (int i = 0; i < M * N; i++)
    {
        float diff = std::abs(C_gpu[i] - C_cpu[i]);
        max_diff = std::max(max_diff, diff);
    }

    std::cout << "Pipelined Correctness (128×128×128):" << std::endl;
    std::cout << "  Max difference vs CPU: " << max_diff << std::endl;

    EXPECT_LT(max_diff, 5e-3f) << "Pipelined kernel numerical error too large";

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// ============================================================
// TEST: Performance Comparison (Pipelined vs Non-Pipelined)
// ============================================================

TEST(Test__CudaGemmPhase3Pipelined, SpeedupVsNonPipelined)
{
    const int M = 128, N = 896, K = 896;
    const int WARMUP = 5;
    const int ITERS = 100;

    std::mt19937 gen(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> A(M * K);
    std::vector<float> B_float(N * K);

    for (auto &v : A)
        v = dist(gen);
    for (auto &v : B_float)
        v = dist(gen);

    auto B_blocks = quantize_to_iq4nl(B_float, N, K);

    // Allocate GPU memory
    float *d_A, *d_C;
    IQ4_NLBlock *d_B;

    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, B_blocks.size() * sizeof(IQ4_NLBlock));
    cudaMalloc(&d_C, M * N * sizeof(float));

    cudaMemcpy(d_A, A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B_blocks.data(), B_blocks.size() * sizeof(IQ4_NLBlock), cudaMemcpyHostToDevice);

    // ===== BASELINE: Non-Pipelined (Phase 3 Part 1) =====
    for (int i = 0; i < WARMUP; i++)
    {
        launch_iq4nl_gemm_phase3(d_A, d_B, d_C, M, N, K);
    }
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < ITERS; i++)
    {
        launch_iq4nl_gemm_phase3(d_A, d_B, d_C, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms_baseline = 0.0f;
    cudaEventElapsedTime(&ms_baseline, start, stop);
    ms_baseline /= ITERS;

    double flops = 2.0 * M * N * K;
    double gflops_baseline = (flops / 1e9) / (ms_baseline / 1000.0);

    // ===== OPTIMIZED: Pipelined (Phase 3 Part 2) =====
    for (int i = 0; i < WARMUP; i++)
    {
        launch_iq4nl_gemm_pipelined(d_A, d_B, d_C, M, N, K);
    }
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < ITERS; i++)
    {
        launch_iq4nl_gemm_pipelined(d_A, d_B, d_C, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms_pipelined = 0.0f;
    cudaEventElapsedTime(&ms_pipelined, start, stop);
    ms_pipelined /= ITERS;

    double gflops_pipelined = (flops / 1e9) / (ms_pipelined / 1000.0);

    double speedup = gflops_pipelined / gflops_baseline;

    std::cout << "\n========== Phase 3 Pipelined vs Non-Pipelined ==========" << std::endl;
    std::cout << "Matrix: " << M << "×" << N << "×" << K << " (Batch=128)" << std::endl;
    std::cout << "\nBaseline (Phase 3 Part 1 - No Pipeline):" << std::endl;
    std::cout << "  Time: " << ms_baseline << " ms" << std::endl;
    std::cout << "  Performance: " << gflops_baseline << " GFLOPS" << std::endl;
    std::cout << "\nOptimized (Phase 3 Part 2 - Pipelined):" << std::endl;
    std::cout << "  Time: " << ms_pipelined << " ms" << std::endl;
    std::cout << "  Performance: " << gflops_pipelined << " GFLOPS" << std::endl;
    std::cout << "\nSpeedup: " << speedup << "×" << std::endl;
    std::cout << "Expected: 1.3-1.4× (target 900-1,000 GFLOPS)" << std::endl;
    std::cout << "=========================================================\n"
              << std::endl;

    EXPECT_GT(speedup, 1.0) << "Pipelined kernel should be faster!";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// ============================================================
// TEST: Batch Size Scaling (M = 128, 256, 512, 1024, 2048, 4096)
// ============================================================
TEST(Test__CudaGemmPhase3Pipelined, BatchSizeScaling)
{
    const int N = 896;
    const int K = 896;
    const int WARMUP = 3;
    const int ITERS = 10;

    // RTX 3090 theoretical peak: 35.58 TFLOPS
    const double GPU_PEAK_TFLOPS = 35.58;

    std::cout << "\n========== Batch Size Scaling Analysis ==========" << std::endl;
    std::cout << "GPU: RTX 3090 (Compute 8.6, 82 SMs)" << std::endl;
    std::cout << "Theoretical FP16 Peak: " << GPU_PEAK_TFLOPS << " TFLOPS" << std::endl;
    std::cout << "Matrix: M×896×896 (varying M)" << std::endl;
    std::cout << "Tile: 64×64×64 (pipelined)" << std::endl;
    std::cout << "===================================================\n"
              << std::endl;

    std::vector<int> batch_sizes = {128, 256, 512, 1024, 2048, 4096};

    for (int M : batch_sizes)
    {
        // Generate random test data
        std::vector<float> A(M * K);
        std::vector<float> B_fp32(N * K);
        std::vector<float> C(M * N, 0.0f);

        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
        for (auto &v : A)
            v = dis(gen);
        for (auto &v : B_fp32)
            v = dis(gen);

        auto B_blocks = quantize_to_iq4nl(B_fp32, N, K);

        // Allocate device memory
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
            launch_iq4nl_gemm_pipelined(d_A, d_B, d_C, M, N, K);
        }
        cudaDeviceSynchronize();

        // Benchmark
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        for (int i = 0; i < ITERS; i++)
        {
            launch_iq4nl_gemm_pipelined(d_A, d_B, d_C, M, N, K);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        ms /= ITERS;

        double flops = 2.0 * M * N * K;
        double gflops = (flops / 1e9) / (ms / 1000.0);
        double tflops = gflops / 1000.0;
        double utilization = (tflops / GPU_PEAK_TFLOPS) * 100.0;

        // Calculate grid configuration
        int blocks_m = (M + 63) / 64;
        int blocks_n = (N + 63) / 64;
        int total_blocks = blocks_m * blocks_n;

        std::cout << "M=" << M << ":" << std::endl;
        std::cout << "  Time:        " << ms << " ms" << std::endl;
        std::cout << "  Throughput:  " << gflops << " GFLOPS (" << tflops << " TFLOPS)" << std::endl;
        std::cout << "  Utilization: " << utilization << "% of peak" << std::endl;
        std::cout << "  Grid:        " << blocks_m << "×" << blocks_n << " = " << total_blocks << " blocks";
        if (total_blocks < 82)
        {
            std::cout << " (⚠️ " << (82 - total_blocks) << " SMs idle)";
        }
        std::cout << std::endl
                  << std::endl;

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }

    std::cout << "===================================================\n"
              << std::endl;
}
