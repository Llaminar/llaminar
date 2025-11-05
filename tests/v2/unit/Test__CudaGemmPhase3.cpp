/**
 * @file Test__CudaGemmPhase3.cpp
 * @brief Tests for Phase 3 large-tile GEMM kernel
 *
 * Validates correctness and measures performance of Phase 3 kernel:
 * - Larger tiles (128×128×64)
 * - 2×2 MMA atom layout
 * - K-dimension blocking
 *
 * Expected: 2-3× speedup over Phase 2 (800-1,000 GFLOPS target)
 *
 * @author David Sanftenberg
 * @date November 3, 2025
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <random>

// IQ4_NL block definition (must match kernel)
struct IQ4_NLBlock
{
    float scale;        // FP32 scale factor
    uint8_t quants[16]; // 16 bytes for 32 4-bit values
};

#include "v2/kernels/cuda/CudaGemmKernelPhase3.h"

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

            // Compute scale (max abs value in block)
            float max_abs = 0.0f;
            for (int i = 0; i < 32; i++)
            {
                float val = B[n * K + k_start + i];
                max_abs = std::max(max_abs, std::abs(val));
            }

            block.scale = (max_abs > 0.0f) ? max_abs : 1.0f;

            // Quantize each value
            for (int i = 0; i < 16; i++)
            {
                float val0 = B[n * K + k_start + i * 2];
                float val1 = B[n * K + k_start + i * 2 + 1];

                // Normalize and find closest quantization level
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
// TEST: Basic Correctness (Small Matrix)
// ============================================================

TEST(Test__CudaGemmPhase3, SmallMatrixCorrectness)
{
    const int M = 128, N = 128, K = 128;

    // Generate random input
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

    // Quantize B
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

    // CPU reference
    cpu_gemm_reference(A, B_dequant, C_cpu, M, N, K);

    // GPU computation
    float *d_A, *d_C;
    IQ4_NLBlock *d_B;

    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, B_blocks.size() * sizeof(IQ4_NLBlock));
    cudaMalloc(&d_C, M * N * sizeof(float));

    cudaMemcpy(d_A, A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B_blocks.data(), B_blocks.size() * sizeof(IQ4_NLBlock), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, M * N * sizeof(float));

    launch_iq4nl_gemm_phase3(d_A, d_B, d_C, M, N, K);

    cudaMemcpy(C_gpu.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Compare results
    float max_diff = 0.0f;
    for (int i = 0; i < M * N; i++)
    {
        float diff = std::abs(C_gpu[i] - C_cpu[i]);
        max_diff = std::max(max_diff, diff);
    }

    std::cout << "Phase 3 Small Matrix (128×128×128):" << std::endl;
    std::cout << "  Max difference: " << max_diff << std::endl;

    EXPECT_LT(max_diff, 5e-3f) << "Phase 3 numerical error too large";

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// ============================================================
// TEST: Qwen 0.5B QKV Performance (BATCH mode - realistic workload)
// ============================================================

TEST(Test__CudaGemmPhase3, Qwen05B_Batch128_QKV)
{
    const int M = 128, N = 896, K = 896; // Batch of 128 tokens
    const int WARMUP = 5;
    const int ITERS = 100;

    // Generate random input
    std::mt19937 gen(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> A(M * K);
    std::vector<float> B_float(N * K);
    std::vector<float> C_gpu(M * N, 0.0f);

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

    // Warmup
    for (int i = 0; i < WARMUP; i++)
    {
        launch_iq4nl_gemm_phase3(d_A, d_B, d_C, M, N, K);
    }
    cudaDeviceSynchronize();

    // Benchmark
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

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= ITERS;

    // Calculate GFLOPS
    double flops = 2.0 * M * N * K;
    double gflops = (flops / 1e9) / (ms / 1000.0);

    std::cout << "\nPhase 3 Performance (Qwen 0.5B Batch=128 QKV " << M << "×" << N << "×" << K << "):" << std::endl;
    std::cout << "  Time: " << ms << " ms" << std::endl;
    std::cout << "  Performance: " << gflops << " GFLOPS" << std::endl;
    std::cout << "  Grid size: (" << (M + 127) / 128 << "×" << (N + 127) / 128 << ") blocks" << std::endl;
    std::cout << "  Expected: 800-1,000 GFLOPS (NOW with proper parallelism!)" << std::endl;

    // Verify correctness
    cudaMemcpy(C_gpu.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    bool has_nonzero = false;
    for (int i = 0; i < M * N; i++)
    {
        if (std::abs(C_gpu[i]) > 1e-6f)
        {
            has_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero) << "Phase 3 output is all zeros";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

TEST(Test__CudaGemmPhase3, Qwen05B_Batch32_QKV)
{
    const int M = 32, N = 896, K = 896; // Batch of 32 tokens
    const int WARMUP = 5;
    const int ITERS = 100;

    // Generate random input
    std::mt19937 gen(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> A(M * K);
    std::vector<float> B_float(N * K);
    std::vector<float> C_gpu(M * N, 0.0f);

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

    // Warmup
    for (int i = 0; i < WARMUP; i++)
    {
        launch_iq4nl_gemm_phase3(d_A, d_B, d_C, M, N, K);
    }
    cudaDeviceSynchronize();

    // Benchmark
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

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= ITERS;

    // Calculate GFLOPS
    double flops = 2.0 * M * N * K;
    double gflops = (flops / 1e9) / (ms / 1000.0);

    std::cout << "\nPhase 3 Performance (Qwen 0.5B Batch=32 QKV " << M << "×" << N << "×" << K << "):" << std::endl;
    std::cout << "  Time: " << ms << " ms" << std::endl;
    std::cout << "  Performance: " << gflops << " GFLOPS" << std::endl;
    std::cout << "  Grid size: (" << (M + 127) / 128 << "×" << (N + 127) / 128 << ") blocks" << std::endl;
    std::cout << "  Expected: 800-1,000 GFLOPS (large tiles benefit from batch)" << std::endl;

    // Verify correctness
    cudaMemcpy(C_gpu.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    bool has_nonzero = false;
    for (int i = 0; i < M * N; i++)
    {
        if (std::abs(C_gpu[i]) > 1e-6f)
        {
            has_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero) << "Phase 3 output is all zeros";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// ============================================================
// TEST: Qwen 0.5B QKV Performance (SINGLE token - for comparison)
// ============================================================

TEST(Test__CudaGemmPhase3, Qwen05B_SingleToken_QKV)
{
    const int M = 1, N = 896, K = 896;
    const int WARMUP = 5;
    const int ITERS = 100;

    // Generate random input
    std::mt19937 gen(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> A(M * K);
    std::vector<float> B_float(N * K);
    std::vector<float> C_gpu(M * N, 0.0f);

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

    // Warmup
    for (int i = 0; i < WARMUP; i++)
    {
        launch_iq4nl_gemm_phase3(d_A, d_B, d_C, M, N, K);
    }
    cudaDeviceSynchronize();

    // Benchmark
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

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= ITERS;

    // Calculate GFLOPS
    // Operations: 2*M*N*K (multiply-add)
    double flops = 2.0 * M * N * K;
    double gflops = (flops / 1e9) / (ms / 1000.0);

    std::cout << "\nPhase 3 Performance (Qwen 0.5B SingleToken QKV " << M << "×" << N << "×" << K << "):" << std::endl;
    std::cout << "  Time: " << ms << " ms" << std::endl;
    std::cout << "  Performance: " << gflops << " GFLOPS" << std::endl;
    std::cout << "  Grid size: (" << (M + 127) / 128 << "×" << (N + 127) / 128 << ") blocks" << std::endl;
    std::cout << "  Note: Single token decode NOT optimal for large tiles!" << std::endl;
    std::cout << "  Expected: Poor performance (large tiles need M≥32)" << std::endl;

    // Verify correctness (spot check)
    cudaMemcpy(C_gpu.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    bool has_nonzero = false;
    for (int i = 0; i < M * N; i++)
    {
        if (std::abs(C_gpu[i]) > 1e-6f)
        {
            has_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero) << "Phase 3 output is all zeros";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// ============================================================
// TEST: Comparison with Phase 2
// ============================================================

TEST(Test__CudaGemmPhase3, SpeedupVsPhase2)
{
    // This test requires Phase 2 implementation to be available
    // Just document expected speedup for now
    std::cout << "\n=== Phase 3 vs Phase 2 Speedup Analysis ===" << std::endl;
    std::cout << "Phase 2 (32×32×32 tiles):  ~363 GFLOPS" << std::endl;
    std::cout << "Phase 3 (128×128×64 tiles): Expected 800-1,000 GFLOPS" << std::endl;
    std::cout << "Target speedup: 2.2-2.8×" << std::endl;
    std::cout << "\nKey improvements:" << std::endl;
    std::cout << "  - 16× larger output tiles (better reuse)" << std::endl;
    std::cout << "  - 2× larger K tiles (better arithmetic intensity)" << std::endl;
    std::cout << "  - 4× more threads (better parallelism)" << std::endl;
    std::cout << "  - 2×2 MMA atom layout (better warp cooperation)" << std::endl;
}
