/**
 * @file Test__Phase5_SUB_K_Sweep.cpp
 * @brief Sweep SUB_K values to find optimal configuration for memory coalescing
 *
 * @author David Sanftenberg
 * @date November 2025
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

using namespace llaminar2::cuda;

class Phase5SubKSweepTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        if (device_count == 0)
        {
            GTEST_SKIP() << "No CUDA device available";
        }

        // Initialize CUDA context
        CUdevice device;
        cuDeviceGet(&device, 0);
        CUcontext ctx;
        cuDevicePrimaryCtxRetain(&ctx, device);
        cuCtxSetCurrent(ctx);

        // Same size as original Phase 5A benchmark
        M_ = 1024;
        N_ = 896;
        K_ = 896;

        // Allocate device memory
        cudaMalloc(&d_A_, M_ * K_ * sizeof(float));
        int blocks_per_row = (K_ + 31) / 32;
        cudaMalloc(&d_B_iq4nl_, N_ * blocks_per_row * 18); // IQ4_NL
        cudaMalloc(&d_C_, M_ * N_ * sizeof(float));

        // Initialize with random data
        std::vector<float> h_A(M_ * K_);
        std::vector<uint8_t> h_B_iq4nl(N_ * blocks_per_row * 18);

        for (auto &val : h_A)
            val = (float)(rand() % 100) / 100.0f;
        for (auto &val : h_B_iq4nl)
            val = rand() % 256;

        cudaMemcpy(d_A_, h_A.data(), M_ * K_ * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B_iq4nl_, h_B_iq4nl.data(), N_ * blocks_per_row * 18, cudaMemcpyHostToDevice);
        cudaMemset(d_C_, 0, M_ * N_ * sizeof(float));
    }

    void TearDown() override
    {
        if (d_A_)
            cudaFree(d_A_);
        if (d_B_iq4nl_)
            cudaFree(d_B_iq4nl_);
        if (d_C_)
            cudaFree(d_C_);
    }

    int M_, N_, K_;
    float *d_A_ = nullptr;
    void *d_B_iq4nl_ = nullptr;
    float *d_C_ = nullptr;
};

TEST_F(Phase5SubKSweepTest, ComprehensiveSweep)
{
    auto &jit = CudaGemmJITPhase5::instance();

    std::cout << "\n========================================\n";
    std::cout << "Phase 5 SUB_K Performance Sweep\n";
    std::cout << "========================================\n";
    std::cout << "Matrix: " << M_ << "x" << N_ << "x" << K_ << "\n";
    std::cout << "Config: TILE 64x64x64, MMA 2x2, Swizzle 3,3,3\n";
    std::cout << "Testing: SUB_K = 16, 32, 64\n\n";

    struct BenchmarkResult
    {
        int sub_k;
        double time_ms;
        double tflops;
    };

    std::vector<int> sub_k_values = {16, 32, 64};
    std::vector<BenchmarkResult> results;

    for (int sub_k : sub_k_values)
    {
        std::cout << "Testing SUB_K = " << sub_k << "..." << std::flush;

        Phase5GemmConfig config{
            .tile_m = 64,
            .tile_n = 64,
            .tile_k = 64,
            .sub_k = sub_k,
            .mma_m = 2,
            .mma_n = 2,
            .buffer_stages = 2,
            .threads_per_block = 128,
            .swizzle_b = 3,
            .swizzle_m = 3,
            .swizzle_s = 3};

        CudaGemmConfigPhase5 full_config(
            config.tile_m, config.tile_n, config.tile_k,
            config.sub_k, config.mma_m, config.mma_n,
            config.buffer_stages);

        CUfunction kernel = jit.getKernel(full_config);

        int blocks_m = (M_ + full_config.tile_m - 1) / full_config.tile_m;
        int blocks_n = (N_ + full_config.tile_n - 1) / full_config.tile_n;
        dim3 grid(blocks_m, blocks_n);
        dim3 block(full_config.threads_per_block);
        void *args[] = {&d_A_, &d_B_iq4nl_, &d_C_, &M_, &N_, &K_};

        // Warmup
        for (int i = 0; i < 20; i++)
        {
            cuLaunchKernel(kernel, grid.x, grid.y, grid.z,
                           block.x, block.y, block.z, 0, nullptr, args, nullptr);
        }
        cudaDeviceSynchronize();

        // Benchmark
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        for (int i = 0; i < 50; i++)
        {
            cuLaunchKernel(kernel, grid.x, grid.y, grid.z,
                           block.x, block.y, block.z, 0, nullptr, args, nullptr);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float elapsed_ms;
        cudaEventElapsedTime(&elapsed_ms, start, stop);
        double avg_ms = elapsed_ms / 50.0;

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        double flops = 2.0 * M_ * N_ * K_;
        double tflops = flops / (avg_ms * 1e9);

        results.push_back({sub_k, avg_ms, tflops});
        std::cout << " " << std::fixed << std::setprecision(2) << tflops << " TFLOPS\n";
    }

    // Results table
    std::cout << "\n========================================\n";
    std::cout << "Results Summary\n";
    std::cout << "========================================\n";
    std::cout << std::setw(10) << "SUB_K"
              << std::setw(15) << "Time (ms)"
              << std::setw(15) << "TFLOPS"
              << std::setw(20) << "vs Baseline\n";
    std::cout << "----------------------------------------\n";

    double baseline_tflops = results[0].tflops;

    for (const auto &r : results)
    {
        double speedup = (r.tflops / baseline_tflops - 1.0) * 100.0;
        std::cout << std::setw(10) << r.sub_k
                  << std::setw(15) << std::fixed << std::setprecision(4) << r.time_ms
                  << std::setw(15) << std::fixed << std::setprecision(2) << r.tflops
                  << std::setw(15) << std::fixed << std::setprecision(2) << std::showpos << speedup << "%" << std::noshowpos << "\n";
    }

    auto best = std::max_element(results.begin(), results.end(),
                                 [](const BenchmarkResult &a, const BenchmarkResult &b)
                                 { return a.tflops < b.tflops; });

    std::cout << "\n*** OPTIMAL: SUB_K = " << best->sub_k
              << " (" << std::fixed << std::setprecision(2) << best->tflops << " TFLOPS) ***\n";
    std::cout << "========================================\n\n";

    if (best->sub_k == 64)
    {
        std::cout << "✓ Hypothesis CONFIRMED: SUB_K=64 enables coalescing\n";
        double actual = (best->tflops / baseline_tflops - 1.0) * 100.0;
        std::cout << "  Expected improvement: ~42%\n";
        std::cout << "  Actual improvement:   " << std::fixed << std::setprecision(1) << actual << "%\n";
    }
    else
    {
        std::cout << "✗ Hypothesis REJECTED: SUB_K=" << best->sub_k << " performed best\n";
    }
}
