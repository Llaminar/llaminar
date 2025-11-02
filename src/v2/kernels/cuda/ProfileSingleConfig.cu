/**
 * @file ProfileSingleConfig.cu
 * @brief Profile a single CUDA GEMM config with NVIDIA Nsight Compute
 *
 * This executable runs a single GEMM configuration so that `ncu` can profile it.
 * Used by collect_profiling_data.py to gather real profiling metrics.
 *
 * Usage:
 *   ncu --metrics <metrics> ./profile_cuda_config m n k tile_m tile_n tile_k threads_m threads_n work_m work_n prefetch transpose
 *
 * @author David Sanftenberg
 * @date November 2, 2025
 */

#include <iostream>
#include <cuda_runtime.h>
#include "CudaGemmVariantsBaseline.h"
#include "CudaGemmConfig.h"
#include "IQ4_NL_BlockDecoder.h"

using namespace llaminar2::cuda;

int main(int argc, char **argv)
{
    if (argc < 13)
    {
        std::cerr << "Usage: " << argv[0] << " m n k tile_m tile_n tile_k "
                  << "threads_m threads_n work_m work_n prefetch transpose\n";
        return 1;
    }

    // Parse arguments
    int m = std::atoi(argv[1]);
    int n = std::atoi(argv[2]);
    int k = std::atoi(argv[3]);

    CudaGemmConfig config;
    config.tile_m = std::atoi(argv[4]);
    config.tile_n = std::atoi(argv[5]);
    config.tile_k = std::atoi(argv[6]);
    config.threads_m = std::atoi(argv[7]);
    config.threads_n = std::atoi(argv[8]);
    config.work_per_thread_m = std::atoi(argv[9]);
    config.work_per_thread_n = std::atoi(argv[10]);
    config.prefetch_stages = std::atoi(argv[11]);
    config.transpose_smem = (std::atoi(argv[12]) != 0);
    config.vectorize_load = (argc > 13) ? std::atoi(argv[13]) : 1;
    config.atom_type = 0; // Default to 16x8x16
    config.atom_layout_m = 1;
    config.atom_layout_n = 1;
    config.atom_layout_k = 1;

    // Allocate device memory
    float *d_A, *d_C;
    IQ4_NLBlock *d_B; // IQ4_NL quantized weights
    size_t size_A = m * k * sizeof(float);
    size_t size_B = (k / 32) * n * sizeof(IQ4_NLBlock); // IQ4_NL: 32 elements per block
    size_t size_C = m * n * sizeof(float);

    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    // Initialize to avoid profiling uninitialized memory warnings
    cudaMemset(d_A, 0, size_A);
    cudaMemset(d_B, 0, size_B);
    cudaMemset(d_C, 0, size_C);

    // Warmup (1 iteration)
    launchIQ4NLGemmVariant(d_A, d_B, d_C, m, n, k, config);
    cudaDeviceSynchronize();

    // Run kernel (ncu will profile this)
    cudaError_t err = launchIQ4NLGemmVariant(d_A, d_B, d_C, m, n, k, config);
    cudaDeviceSynchronize(); // Check for errors
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA kernel error: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
