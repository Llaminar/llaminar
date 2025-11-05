/**
 * @file Test__CudaGemmPhase4QuickWins.cpp
 * @brief Tests for Phase 4 quick wins (swizzled + cp.async)
 *
 * Expected: 7.6 TFLOPS at M=1024 (+16% over Phase 3 Part 2's 6.56 TFLOPS)
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

#include "v2/kernels/cuda/CudaGemmKernelPhase3Pipelined.h"
#include "v2/kernels/cuda/CudaGemmKernelPhase4QuickWins.h"

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
// TEST: Correctness (Phase 4 vs Reference)
// ============================================================
TEST(Test__CudaGemmPhase4QuickWins, CorrectnessVsCPU)
{
    const int M = 128, N = 128, K = 128;

    std::vector<float> A(M * K);
    std::vector<float> B_fp32(N * K);
    std::vector<float> C_ref(M * N, 0.0f);
    std::vector<float> C_gpu(M * N, 0.0f);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (auto &v : A)
        v = dis(gen);
    for (auto &v : B_fp32)
        v = dis(gen);

    auto B_blocks = quantize_to_iq4nl(B_fp32, N, K);

    // CRITICAL: Dequantize B_blocks for CPU reference (quantization is lossy!)
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

    // CPU reference with DEQUANTIZED B matrix
    cpu_gemm_reference(A, B_dequant, C_ref, M, N, K);

    // GPU Phase 4
    float *d_A, *d_C;
    IQ4_NLBlock *d_B;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, B_blocks.size() * sizeof(IQ4_NLBlock));
    cudaMalloc(&d_C, M * N * sizeof(float));

    cudaMemcpy(d_A, A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B_blocks.data(), B_blocks.size() * sizeof(IQ4_NLBlock), cudaMemcpyHostToDevice);

    launch_iq4nl_gemm_phase4(d_A, d_B, d_C, M, N, K);

    cudaMemcpy(C_gpu.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Compare
    float max_diff = 0.0f;
    for (int i = 0; i < M * N; i++)
    {
        max_diff = std::max(max_diff, std::abs(C_gpu[i] - C_ref[i]));
    }

    std::cout << "Phase 4 Correctness (128×128×128):" << std::endl;
    std::cout << "  Max difference vs CPU: " << max_diff << std::endl;

    EXPECT_LT(max_diff, 5e-3) << "Phase 4 kernel should match CPU reference";

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// ============================================================
// TEST: Performance (Phase 4 vs Phase 3 Part 2)
// ============================================================
TEST(Test__CudaGemmPhase4QuickWins, SpeedupVsPhase3)
{
    const int M = 1024; // Sweet spot batch size
    const int N = 896;
    const int K = 896;
    const int WARMUP = 5;
    const int ITERS = 20;

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

    float *d_A, *d_C;
    IQ4_NLBlock *d_B;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, B_blocks.size() * sizeof(IQ4_NLBlock));
    cudaMalloc(&d_C, M * N * sizeof(float));

    cudaMemcpy(d_A, A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B_blocks.data(), B_blocks.size() * sizeof(IQ4_NLBlock), cudaMemcpyHostToDevice);

    // ===== BASELINE: Phase 3 Part 2 (Pipelined) =====
    for (int i = 0; i < WARMUP; i++)
    {
        launch_iq4nl_gemm_pipelined(d_A, d_B, d_C, M, N, K);
    }
    cudaDeviceSynchronize();

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

    float ms_phase3 = 0.0f;
    cudaEventElapsedTime(&ms_phase3, start, stop);
    ms_phase3 /= ITERS;

    double flops = 2.0 * M * N * K;
    double gflops_phase3 = (flops / 1e9) / (ms_phase3 / 1000.0);
    double tflops_phase3 = gflops_phase3 / 1000.0;

    // ===== OPTIMIZED: Phase 4 (Swizzled + cp.async) =====
    for (int i = 0; i < WARMUP; i++)
    {
        launch_iq4nl_gemm_phase4(d_A, d_B, d_C, M, N, K);
    }
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < ITERS; i++)
    {
        launch_iq4nl_gemm_phase4(d_A, d_B, d_C, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms_phase4 = 0.0f;
    cudaEventElapsedTime(&ms_phase4, start, stop);
    ms_phase4 /= ITERS;

    double gflops_phase4 = (flops / 1e9) / (ms_phase4 / 1000.0);
    double tflops_phase4 = gflops_phase4 / 1000.0;

    double speedup = gflops_phase4 / gflops_phase3;
    double improvement = ((gflops_phase4 - gflops_phase3) / gflops_phase3) * 100.0;

    // RTX 3090 theoretical peak
    const double GPU_PEAK_TFLOPS = 35.58;
    double util_phase3 = (tflops_phase3 / GPU_PEAK_TFLOPS) * 100.0;
    double util_phase4 = (tflops_phase4 / GPU_PEAK_TFLOPS) * 100.0;

    std::cout << "\n========== Phase 4 Quick Wins vs Phase 3 Part 2 ==========" << std::endl;
    std::cout << "Matrix: " << M << "×" << N << "×" << K << " (Sweet spot batch)" << std::endl;
    std::cout << "GPU: RTX 3090 (35.58 TFLOPS peak)" << std::endl;
    std::cout << "\nBaseline (Phase 3 Part 2 - Pipelined):" << std::endl;
    std::cout << "  Time:        " << ms_phase3 << " ms" << std::endl;
    std::cout << "  Throughput:  " << gflops_phase3 << " GFLOPS (" << tflops_phase3 << " TFLOPS)" << std::endl;
    std::cout << "  Utilization: " << util_phase3 << "% of peak" << std::endl;
    std::cout << "\nOptimized (Phase 4 - Swizzled + cp.async):" << std::endl;
    std::cout << "  Time:        " << ms_phase4 << " ms" << std::endl;
    std::cout << "  Throughput:  " << gflops_phase4 << " GFLOPS (" << tflops_phase4 << " TFLOPS)" << std::endl;
    std::cout << "  Utilization: " << util_phase4 << "% of peak" << std::endl;
    std::cout << "\nImprovement:" << std::endl;
    std::cout << "  Speedup:     " << speedup << "×" << std::endl;
    std::cout << "  Relative:    +" << improvement << "%" << std::endl;
    std::cout << "  Target:      +10-16% (7.2-7.6 TFLOPS)" << std::endl;
    std::cout << "===========================================================\n"
              << std::endl;

    EXPECT_GT(speedup, 1.0) << "Phase 4 should be faster than Phase 3 Part 2!";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// ============================================================
// TEST: Batch Scaling (Phase 4 across M=128 to M=4096)
// ============================================================
TEST(Test__CudaGemmPhase4QuickWins, BatchScaling)
{
    const int N = 896, K = 896; // Fixed dimensions
    const int WARMUP = 5;
    const int ITERS = 20;

    // RTX 3090 theoretical peak
    const double GPU_PEAK_TFLOPS = 35.58;

    std::cout << "\n========== Phase 4 Swizzle: Batch Scaling Analysis ==========" << std::endl;
    std::cout << "GPU: RTX 3090 (35.58 TFLOPS FP16 peak)" << std::endl;
    std::cout << "Matrix: M×896×896 (varying M)" << std::endl;
    std::cout << "Backend: Phase 4 with Swizzle<3,3,3>\n"
              << std::endl;

    std::cout << "| Batch M | Time (ms) | TFLOPS | % of Peak | Speedup vs M=128 |" << std::endl;
    std::cout << "|---------|-----------|--------|-----------|------------------|" << std::endl;

    std::vector<int> batch_sizes = {128, 256, 512, 1024, 2048, 4096};
    double baseline_tflops = 0.0;

    for (int M : batch_sizes)
    {
        // Allocate matrices
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        std::vector<float> A(M * K);
        std::vector<float> B_float(N * K);

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
        double util = (tflops / GPU_PEAK_TFLOPS) * 100.0;

        if (M == 128)
        {
            baseline_tflops = tflops;
        }
        double speedup = tflops / baseline_tflops;

        std::cout << "| " << std::setw(7) << M
                  << " | " << std::setw(9) << std::fixed << std::setprecision(3) << ms
                  << " | " << std::setw(6) << std::setprecision(2) << tflops
                  << " | " << std::setw(9) << std::setprecision(2) << util << "%"
                  << " | " << std::setw(16) << std::setprecision(2) << speedup << "× |"
                  << std::endl;

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    std::cout << "=============================================================\n"
              << std::endl;
}

// ============================================================
// TEST: Phase 4 vs Phase 3 Batch Scaling Comparison
// ============================================================
TEST(Test__CudaGemmPhase4QuickWins, BatchScalingComparison)
{
    const int N = 896, K = 896;
    const int WARMUP = 5;
    const int ITERS = 20;

    const double GPU_PEAK_TFLOPS = 35.58;

    std::cout << "\n========== Phase 4 (Swizzle) vs Phase 3 (No Swizzle): Batch Scaling ==========" << std::endl;
    std::cout << "GPU: RTX 3090 (35.58 TFLOPS FP16 peak)" << std::endl;
    std::cout << "Matrix: M×896×896 (varying M)\n"
              << std::endl;

    std::cout << "| Batch M | Phase 3 TFLOPS | Phase 4 TFLOPS | Speedup | Improvement |" << std::endl;
    std::cout << "|---------|----------------|----------------|---------|-------------|" << std::endl;

    std::vector<int> batch_sizes = {128, 256, 512, 1024, 2048, 4096};

    for (int M : batch_sizes)
    {
        // Allocate matrices
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        std::vector<float> A(M * K);
        std::vector<float> B_float(N * K);

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

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // ===== PHASE 3 BASELINE =====
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

        float ms_phase3 = 0.0f;
        cudaEventElapsedTime(&ms_phase3, start, stop);
        ms_phase3 /= ITERS;

        double flops = 2.0 * M * N * K;
        double tflops_phase3 = (flops / 1e9) / (ms_phase3 / 1000.0) / 1000.0;

        // ===== PHASE 4 OPTIMIZED =====
        for (int i = 0; i < WARMUP; i++)
        {
            launch_iq4nl_gemm_phase4(d_A, d_B, d_C, M, N, K);
        }
        cudaDeviceSynchronize();

        cudaEventRecord(start);
        for (int i = 0; i < ITERS; i++)
        {
            launch_iq4nl_gemm_phase4(d_A, d_B, d_C, M, N, K);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms_phase4 = 0.0f;
        cudaEventElapsedTime(&ms_phase4, start, stop);
        ms_phase4 /= ITERS;

        double tflops_phase4 = (flops / 1e9) / (ms_phase4 / 1000.0) / 1000.0;

        double speedup = tflops_phase4 / tflops_phase3;
        double improvement = ((tflops_phase4 - tflops_phase3) / tflops_phase3) * 100.0;

        std::cout << "| " << std::setw(7) << M
                  << " | " << std::setw(14) << std::fixed << std::setprecision(2) << tflops_phase3
                  << " | " << std::setw(14) << std::setprecision(2) << tflops_phase4
                  << " | " << std::setw(7) << std::setprecision(2) << speedup << "×"
                  << " | " << std::setw(11) << std::setprecision(1) << improvement << "% |"
                  << std::endl;

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    std::cout << "===============================================================================\n"
              << std::endl;
}
