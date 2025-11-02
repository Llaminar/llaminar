/**
 * @file Perf__Phase1_Debug.cu
 * @brief Debug version of Phase 1 performance test with separate configs
 *
 * @author David Sanftenberg
 * @date November 1, 2025
 *
 * PURPOSE: Test baseline and optimized kernels with THEIR OWN optimal configs
 * - Baseline: Uses its own instantiated configurations
 * - Optimized: Uses its own instantiated configurations
 * - Compares performance even though configs differ
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>

#include "kernels/cuda/CudaGemmVariantsBaseline.h"
#include "kernels/cuda/CudaGemmVariantsMemoryOpt.h"
#include "kernels/cuda/CudaGemmConfig.h"
#include "kernels/cuda/IQ4_NL_BlockDecoder.h"

using namespace llaminar2::cuda;

class Phase1_Debug : public ::testing::Test
{
protected:
    float *d_A_;
    IQ4_NLBlock *d_B_;
    float *d_C_;

    cudaEvent_t start_event_, stop_event_;

    void SetUp() override
    {
        cudaEventCreate(&start_event_);
        cudaEventCreate(&stop_event_);

        std::cout << "\n[CUDA Device] ";
        int device_id;
        cudaGetDevice(&device_id);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device_id);
        std::cout << prop.name << std::endl;
        std::cout << "  Compute capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Global memory: " << prop.totalGlobalMem / (1024 * 1024 * 1024) << " GB" << std::endl;
        std::cout << "  Shared memory per block: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
        std::cout << std::endl;
    }

    void TearDown() override
    {
        if (d_A_)
            cudaFree(d_A_);
        if (d_B_)
            cudaFree(d_B_);
        if (d_C_)
            cudaFree(d_C_);
        cudaEventDestroy(start_event_);
        cudaEventDestroy(stop_event_);
    }

    void allocateTestData(int m, int n, int k)
    {
        // Allocate device memory
        cudaMalloc(&d_A_, m * k * sizeof(float));
        cudaMalloc(&d_B_, (k / 32) * n * sizeof(IQ4_NLBlock));
        cudaMalloc(&d_C_, m * n * sizeof(float));

        // Initialize with random data on CPU, then copy
        std::vector<float> h_A(m * k);
        std::vector<IQ4_NLBlock> h_B((k / 32) * n);

        for (auto &val : h_A)
            val = static_cast<float>(rand()) / RAND_MAX;
        for (auto &block : h_B)
        {
            block.d = 1.0f; // Scale factor
            for (int i = 0; i < 16; ++i)
            {
                block.qs[i] = rand() % 256; // Random quantized values
            }
        }

        cudaMemcpy(d_A_, h_A.data(), m * k * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B_, h_B.data(), (k / 32) * n * sizeof(IQ4_NLBlock), cudaMemcpyHostToDevice);
        cudaMemset(d_C_, 0, m * n * sizeof(float));
    }

    float benchmarkKernel(
        int m, int n, int k,
        const CudaGemmConfig &config,
        bool use_optimized,
        int iters = 100)
    {
        std::vector<float> times;
        times.reserve(iters);

        for (int i = 0; i < iters; ++i)
        {
            cudaMemset(d_C_, 0, m * n * sizeof(float));

            cudaEventRecord(start_event_);

            cudaError_t err;
            if (use_optimized)
            {
                err = launchIQ4NLGemmVariantOptimized(d_A_, d_B_, d_C_, m, n, k, config, nullptr);
            }
            else
            {
                err = launchIQ4NLGemmVariant(d_A_, d_B_, d_C_, m, n, k, config, nullptr);
            }

            cudaEventRecord(stop_event_);
            cudaEventSynchronize(stop_event_);

            if (err != cudaSuccess)
            {
                std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
                return 0.0f;
            }

            float time_ms = 0.0f;
            cudaEventElapsedTime(&time_ms, start_event_, stop_event_);
            times.push_back(time_ms);
        }

        // Compute median time
        std::sort(times.begin(), times.end());
        float median_time_ms = times[times.size() / 2];

        // Compute GFLOPS
        double flops = 2.0 * m * n * k; // Each output element: k multiplies and k-1 adds
        double gflops = flops / (median_time_ms * 1e6);

        return gflops;
    }
};

/**
 * Test baseline vs optimized with THEIR OWN optimal configurations
 * Baseline: 64×64×32, threads 16×16, work 4×4, prefetch=1
 * Optimized: 64×64×32, threads 4×4, work 16×16, prefetch=0, vec=4
 */
TEST_F(Phase1_Debug, SmallBatch_Separate_Configs)
{
    const int m = 32;
    const int n = 896;
    const int k = 896;

    allocateTestData(m, n, k);

    // Baseline config (from CudaGemmVariants.cu)
    CudaGemmConfig baseline_config;
    baseline_config.tile_m = 64;
    baseline_config.tile_n = 64;
    baseline_config.tile_k = 32;
    baseline_config.threads_m = 16;
    baseline_config.threads_n = 16;
    baseline_config.work_per_thread_m = 4;
    baseline_config.work_per_thread_n = 4;
    baseline_config.prefetch_stages = 1;
    baseline_config.transpose_smem = true;
    baseline_config.vectorize_load = 1; // Note: VEC means something different in baseline

    // Optimized config (from CudaGemmVariantsOptimized.cu)
    // FIXED (2025-11-01): Changed to threads(16,16) work(4,4) to reduce bank conflicts
    CudaGemmConfig optimized_config;
    optimized_config.tile_m = 64;
    optimized_config.tile_n = 64;
    optimized_config.tile_k = 32;
    optimized_config.threads_m = 16;        // FIXED: Was 4, now 16 (matches baseline)
    optimized_config.threads_n = 16;        // FIXED: Was 4, now 16
    optimized_config.work_per_thread_m = 4; // FIXED: Was 16, now 4
    optimized_config.work_per_thread_n = 4; // FIXED: Was 16, now 4
    optimized_config.prefetch_stages = 0;
    optimized_config.transpose_smem = false;
    optimized_config.vectorize_load = 4; // float4 vectorization (advantage over baseline)

    std::cout << "\n=== BASELINE KERNEL (own config) ===" << std::endl;
    std::cout << "Config: tile(64,64,32) threads(16,16) work(4,4) prefetch=1 transpose=true vec=1" << std::endl;
    float baseline_gflops = benchmarkKernel(m, n, k, baseline_config, false, 100);
    std::cout << "Performance: " << baseline_gflops << " GFLOPS" << std::endl;

    std::cout << "\n=== OPTIMIZED KERNEL (own config) ===" << std::endl;
    std::cout << "Config: tile(64,64,32) threads(4,4) work(16,16) prefetch=0 transpose=false vec=4" << std::endl;
    float optimized_gflops = benchmarkKernel(m, n, k, optimized_config, true, 100);
    std::cout << "Performance: " << optimized_gflops << " GFLOPS" << std::endl;

    std::cout << "\n=== COMPARISON ===" << std::endl;
    if (baseline_gflops > 0 && optimized_gflops > 0)
    {
        float speedup = optimized_gflops / baseline_gflops;
        std::cout << "Speedup: " << speedup << "× ";
        if (speedup > 1.2)
        {
            std::cout << "(✓ OPTIMIZED FASTER)" << std::endl;
        }
        else if (speedup < 0.8)
        {
            std::cout << "(✗ OPTIMIZED SLOWER!)" << std::endl;
        }
        else
        {
            std::cout << "(≈ SIMILAR PERFORMANCE)" << std::endl;
        }
    }

    // Check correctness
    std::vector<float> h_C_baseline(m * n);
    std::vector<float> h_C_optimized(m * n);

    cudaMemset(d_C_, 0, m * n * sizeof(float));
    launchIQ4NLGemmVariant(d_A_, d_B_, d_C_, m, n, k, baseline_config, nullptr);
    cudaMemcpy(h_C_baseline.data(), d_C_, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaMemset(d_C_, 0, m * n * sizeof(float));
    launchIQ4NLGemmVariantOptimized(d_A_, d_B_, d_C_, m, n, k, optimized_config, nullptr);
    cudaMemcpy(h_C_optimized.data(), d_C_, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "\n=== CORRECTNESS CHECK ===" << std::endl;
    float max_diff = 0.0f;
    for (int i = 0; i < m * n; ++i)
    {
        max_diff = std::max(max_diff, std::abs(h_C_baseline[i] - h_C_optimized[i]));
    }
    std::cout << "Max absolute difference: " << max_diff << std::endl;

    if (max_diff < 1e-3)
    {
        std::cout << "✓ Results match!" << std::endl;
    }
    else
    {
        std::cout << "✗ Results DIFFER - possible correctness bug!" << std::endl;
        std::cout << "First 10 baseline:  ";
        for (int i = 0; i < 10; ++i)
            std::cout << h_C_baseline[i] << " ";
        std::cout << std::endl;
        std::cout << "First 10 optimized: ";
        for (int i = 0; i < 10; ++i)
            std::cout << h_C_optimized[i] << " ";
        std::cout << std::endl;
    }
}
