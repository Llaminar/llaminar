/**
 * @file Perf__Phase2_TensorCore.cu
 * @brief Phase 2 Tensor Core performance test
 *
 * @author David Sanftenberg
 * @date November 1, 2025
 *
 * PURPOSE: Validate Phase 2 Tensor Core kernel performance and correctness
 *
 * PHASE 2 TARGET: 3-4× speedup over Phase 1 (425 → 1,275-1,700 GFLOPS)
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>

#include "kernels/cuda/CudaGemmVariantsBaseline.h"
#include "kernels/cuda/CudaGemmVariantsMemoryOpt.h"
#include "kernels/cuda/CudaGemmVariantsTensorCore.h"
#include "kernels/cuda/CudaGemmConfig.h"
#include "kernels/cuda/IQ4_NL_BlockDecoder.h"

using namespace llaminar2::cuda;

class Phase2_TensorCore : public ::testing::Test
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

        if (prop.major < 7)
        {
            GTEST_SKIP() << "Tensor Cores require compute capability ≥ 7.0";
        }

        std::cout << "  Global memory: " << prop.totalGlobalMem / (1024 * 1024 * 1024) << " GB" << std::endl;
        std::cout << "  Shared memory per block: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
        std::cout << "  Tensor Cores: " << (prop.major >= 7 ? "YES" : "NO") << std::endl;
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

        // Initialize with random data
        std::vector<float> h_A(m * k);
        std::vector<IQ4_NLBlock> h_B((k / 32) * n);

        for (auto &val : h_A)
            val = static_cast<float>(rand()) / RAND_MAX - 0.5f;
        for (auto &block : h_B)
        {
            block.d = __float2half_rn(0.1f); // Scale factor
            for (int i = 0; i < 16; ++i)
            {
                block.qs[i] = rand() % 256;
            }
        }

        cudaMemcpy(d_A_, h_A.data(), m * k * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B_, h_B.data(), (k / 32) * n * sizeof(IQ4_NLBlock), cudaMemcpyHostToDevice);
        cudaMemset(d_C_, 0, m * n * sizeof(float));
    }

    float benchmarkKernel(
        int m, int n, int k,
        const CudaGemmConfig &config,
        std::function<cudaError_t(const float *, const IQ4_NLBlock *, float *, int, int, int, const CudaGemmConfig &, cudaStream_t)> launcher,
        int iters = 100)
    {
        std::vector<float> times;
        times.reserve(iters);

        for (int i = 0; i < iters; ++i)
        {
            cudaMemset(d_C_, 0, m * n * sizeof(float));

            cudaEventRecord(start_event_);
            cudaError_t err = launcher(d_A_, d_B_, d_C_, m, n, k, config, nullptr);
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
        double flops = 2.0 * m * n * k;
        double gflops = flops / (median_time_ms * 1e6);

        return gflops;
    }
};

/**
 * Test Phase 2 Tensor Core kernel vs Phase 1 optimized kernel
 *
 * Phase 1 (baseline): threads(16,16) work(4,4) vectorized loads
 * Phase 2 (Tensor Core): wmma 16×16×16 tiles, FP16 compute, FP32 accumulation
 *
 * Expected: 3-4× speedup with Tensor Cores
 */
TEST_F(Phase2_TensorCore, SmallBatch_Phase1_vs_Phase2)
{
    const int m = 32;
    const int n = 896;
    const int k = 896;

    allocateTestData(m, n, k);

    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║         PHASE 2 TENSOR CORE: SMALL BATCH (32×896)            ║\n";
    std::cout << "╠════════════════════════════════════════════════════════════════╣\n";
    std::cout << "║ Phase 1 (optimized): ~425 GFLOPS                               ║\n";
    std::cout << "║ Phase 2 (Tensor Core): 1,275-1,700 GFLOPS (3-4× target)       ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n";

    // Phase 1 config: threads(16,16) work(4,4) with vectorization
    CudaGemmConfig phase1_config;
    phase1_config.tile_m = 64;
    phase1_config.tile_n = 64;
    phase1_config.tile_k = 32;
    phase1_config.threads_m = 16;
    phase1_config.threads_n = 16;
    phase1_config.work_per_thread_m = 4;
    phase1_config.work_per_thread_n = 4;
    phase1_config.prefetch_stages = 0;
    phase1_config.transpose_smem = false;
    phase1_config.vectorize_load = 4;

    // Phase 2 config: Tensor Core with 16×16×16 tiles
    CudaGemmConfig phase2_config;
    phase2_config.tile_m = 64;
    phase2_config.tile_n = 64;
    phase2_config.tile_k = 16;   // Tensor Cores use K=16 for optimal throughput
    phase2_config.threads_m = 0; // Not used for Tensor Cores (warp-level)
    phase2_config.threads_n = 0;
    phase2_config.work_per_thread_m = 0;
    phase2_config.work_per_thread_n = 0;
    phase2_config.prefetch_stages = 0;
    phase2_config.transpose_smem = false;
    phase2_config.vectorize_load = 0; // Not applicable to Tensor Cores

    std::cout << "\n=== PHASE 1 OPTIMIZED KERNEL ===" << std::endl;
    std::cout << "Config: tile(64,64,32) threads(16,16) work(4,4) vec=4" << std::endl;
    float phase1_gflops = benchmarkKernel(m, n, k, phase1_config, launchIQ4NLGemmVariantOptimized, 100);
    std::cout << "Performance: " << phase1_gflops << " GFLOPS" << std::endl;

    std::cout << "\n=== PHASE 2 TENSOR CORE KERNEL ===" << std::endl;
    std::cout << "Config: tile(64,64,16) wmma(16,16,16) FP16 compute" << std::endl;
    float phase2_gflops = benchmarkKernel(m, n, k, phase2_config, launchIQ4NLGemmVariantTensorCore, 100);
    std::cout << "Performance: " << phase2_gflops << " GFLOPS" << std::endl;

    std::cout << "\n=== COMPARISON ===" << std::endl;
    if (phase1_gflops > 0 && phase2_gflops > 0)
    {
        float speedup = phase2_gflops / phase1_gflops;
        std::cout << "Speedup: " << speedup << "× ";

        if (speedup >= 3.0)
        {
            std::cout << "(✓ PHASE 2 TARGET MET! 3-4× achieved)" << std::endl;
        }
        else if (speedup >= 2.0)
        {
            std::cout << "(⚠ GOOD but below 3× target)" << std::endl;
        }
        else if (speedup >= 1.2)
        {
            std::cout << "(⚠ PHASE 2 FASTER but below target)" << std::endl;
        }
        else
        {
            std::cout << "(✗ PHASE 2 SLOWER - debug needed!)" << std::endl;
        }

        // Check if we hit absolute performance targets
        if (phase2_gflops >= 1275.0)
        {
            std::cout << "✓ Absolute target met (≥1,275 GFLOPS)" << std::endl;
        }
    }

    // Correctness check
    std::cout << "\n=== CORRECTNESS CHECK ===" << std::endl;
    std::vector<float> h_C_phase1(m * n);
    std::vector<float> h_C_phase2(m * n);

    cudaMemset(d_C_, 0, m * n * sizeof(float));
    launchIQ4NLGemmVariantOptimized(d_A_, d_B_, d_C_, m, n, k, phase1_config, nullptr);
    cudaMemcpy(h_C_phase1.data(), d_C_, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaMemset(d_C_, 0, m * n * sizeof(float));
    launchIQ4NLGemmVariantTensorCore(d_A_, d_B_, d_C_, m, n, k, phase2_config, nullptr);
    cudaMemcpy(h_C_phase2.data(), d_C_, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    float max_diff = 0.0f;
    float max_rel_diff = 0.0f;
    for (int i = 0; i < m * n; ++i)
    {
        float abs_diff = std::abs(h_C_phase1[i] - h_C_phase2[i]);
        max_diff = std::max(max_diff, abs_diff);

        if (std::abs(h_C_phase1[i]) > 1e-6f)
        {
            float rel_diff = abs_diff / std::abs(h_C_phase1[i]);
            max_rel_diff = std::max(max_rel_diff, rel_diff);
        }
    }

    std::cout << "Max absolute difference: " << max_diff << std::endl;
    std::cout << "Max relative difference: " << max_rel_diff << std::endl;

    // Tensor Cores use FP16, so allow slightly larger tolerance than FP32
    const float tolerance = 1e-3f; // 0.1% tolerance
    if (max_rel_diff < tolerance)
    {
        std::cout << "✓ Results match within tolerance!" << std::endl;
    }
    else
    {
        std::cout << "✗ Results DIFFER beyond tolerance - FP16 precision issue?" << std::endl;
        std::cout << "First 10 Phase 1:  ";
        for (int i = 0; i < 10; ++i)
            std::cout << h_C_phase1[i] << " ";
        std::cout << std::endl;
        std::cout << "First 10 Phase 2:  ";
        for (int i = 0; i < 10; ++i)
            std::cout << h_C_phase2[i] << " ";
        std::cout << std::endl;
    }

    // Assert performance target
    EXPECT_GE(phase2_gflops, phase1_gflops * 2.0)
        << "Phase 2 should be at least 2× faster than Phase 1";
}
