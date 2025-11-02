/**
 * @file Perf__Phase2_7_TensorCore_Pipeline.cu
 * @brief Phase 2.7 Tensor Core multi-stage pipeline performance test
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cutlass/half.h>
#include <vector>
#include <iostream>

#include "kernels/cuda/CudaGemmKernelTensorCorePipeline.cuh"
#include "kernels/cuda/IQ4_NL_BlockDecoder.h"

using namespace llaminar2::cuda;

TEST(Phase2_7_TensorCore_Pipeline, MultiStagePipelinePerformance)
{
    const int m = 32, n = 896, k = 896;

    std::cout << "\n=== PHASE 2.7 TENSOR CORE: MULTI-STAGE PIPELINE ===\n";
    std::cout << "Target: 1.5-2x speedup over Phase 2.5 (2,500-3,300 GFLOPS)\n\n";

    cutlass::half_t *d_A;
    IQ4_NLBlock *d_B;
    float *d_C;

    cudaMalloc(&d_A, m * k * sizeof(cutlass::half_t));
    cudaMalloc(&d_B, (k / 32) * n * sizeof(IQ4_NLBlock));
    cudaMalloc(&d_C, m * n * sizeof(float));

    std::vector<cutlass::half_t> h_A(m * k);
    std::vector<IQ4_NLBlock> h_B((k / 32) * n);

    for (auto &val : h_A)
        val = cutlass::half_t(0.1f);
    for (auto &block : h_B)
    {
        block.d = __float2half_rn(0.1f);
        for (int i = 0; i < 16; ++i)
            block.qs[i] = 128;
    }

    cudaMemcpy(d_A, h_A.data(), m * k * sizeof(cutlass::half_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), (k / 32) * n * sizeof(IQ4_NLBlock), cudaMemcpyHostToDevice);

    IQ4_NL_Decoder<IQ4_NLBlock> decoder(d_B, n, k / 32);

    // Warmup
    for (int i = 0; i < 10; ++i)
    {
        launchQuantizedGemmPipeline<cutlass::half_t, IQ4_NL_Decoder<IQ4_NLBlock>>(
            d_A, d_C, m, n, k, decoder, 0);
    }
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int iters = 100;
    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i)
    {
        launchQuantizedGemmPipeline<cutlass::half_t, IQ4_NL_Decoder<IQ4_NLBlock>>(
            d_A, d_C, m, n, k, decoder, 0);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= iters;

    double gflops = (2.0 * m * n * k / 1e9) / (ms / 1000.0);

    std::cout << "=== PHASE 2.7 MULTI-STAGE PIPELINE ===\n";
    std::cout << "Performance: " << gflops << " GFLOPS\n";
    std::cout << "Time: " << ms << " ms\n\n";

    const double phase2_5_gflops = 1666.0;
    double speedup = gflops / phase2_5_gflops;

    std::cout << "=== VS PHASE 2.5 ===\n";
    std::cout << "Phase 2.5 (async copy): " << phase2_5_gflops << " GFLOPS\n";
    std::cout << "Phase 2.7 (pipeline): " << gflops << " GFLOPS\n";
    std::cout << "Speedup: " << speedup << "x\n\n";

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    EXPECT_GE(speedup, 1.2) << "Phase 2.7 should be faster than Phase 2.5";
}
