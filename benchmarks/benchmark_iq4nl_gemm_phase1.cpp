/**
 * @file benchmark_iq4nl_gemm_phase1.cpp
 * @brief Benchmark Phase 1 optimizations vs baseline
 *
 * @author David Sanftenberg
 * @date November 1, 2025
 *
 * Compares baseline kernel vs Phase 1 optimized kernel:
 * - Baseline: Original CudaGemmVariants.cu
 * - Optimized: CudaGemmVariantsOptimized.cu (coalesced + vectorized + padding)
 *
 * Expected: 2-3× speedup for large batches
 */

#include "../../src/v2/kernels/cuda/CudaGemmVariants.h"
#include "../../src/v2/kernels/cuda/CudaGemmVariantsOptimized.h"
#include "../../src/v2/tensors/Tensors.h"
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <cmath>
#include <nlohmann/json.hpp>

using json = nlohmann::json;
using namespace llaminar::v2;
using namespace llaminar::v2::kernels;

/**
 * @brief CUDA error checking macro
 */
#define CUDA_CHECK(call)                                                         \
    do                                                                           \
    {                                                                            \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess)                                                  \
        {                                                                        \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl;                   \
            exit(1);                                                             \
        }                                                                        \
    } while (0)

/**
 * @brief Generate random IQ4_NL quantized matrix
 */
std::vector<IQ4_NLBlock> generateRandomIQ4NL(int n, int k, int block_size = 32)
{
    const int k_blocks = (k + block_size - 1) / block_size;
    std::vector<IQ4_NLBlock> blocks(n * k_blocks);

    // Initialize with random values
    for (auto &block : blocks)
    {
        block.d = 0.1f + static_cast<float>(rand()) / RAND_MAX * 0.9f; // Scale [0.1, 1.0]
        for (int i = 0; i < 16; ++i)
        {
            block.qs[i] = rand() % 256; // Random quantized values
        }
    }

    return blocks;
}

/**
 * @brief Benchmark single kernel configuration
 */
struct BenchmarkResult
{
    double time_ms;
    double gflops;
    double bandwidth_gb_s;
    int iterations;
};

BenchmarkResult benchmarkKernel(
    const std::string &kernel_type,
    const float *d_A,
    const IQ4_NLBlock *d_B_blocks,
    float *d_C,
    int m, int n, int k,
    const CudaGemmConfig &config,
    int warmup = 10,
    int iterations = 100)
{
    BenchmarkResult result = {0, 0, 0, iterations};

    // Warmup
    for (int i = 0; i < warmup; ++i)
    {
        if (kernel_type == "baseline")
        {
            CUDA_CHECK(launchIQ4NLGemmVariant(d_A, d_B_blocks, d_C, m, n, k, config));
        }
        else if (kernel_type == "optimized")
        {
            CUDA_CHECK(launchIQ4NLGemmVariantOptimized(d_A, d_B_blocks, d_C, m, n, k, config));
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed iterations
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; ++i)
    {
        if (kernel_type == "baseline")
        {
            CUDA_CHECK(launchIQ4NLGemmVariant(d_A, d_B_blocks, d_C, m, n, k, config));
        }
        else if (kernel_type == "optimized")
        {
            CUDA_CHECK(launchIQ4NLGemmVariantOptimized(d_A, d_B_blocks, d_C, m, n, k, config));
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

    // Calculate metrics
    result.time_ms = elapsed_ms / iterations;

    // FLOPs = 2 * m * n * k (multiply + add)
    double flops = 2.0 * m * n * k;
    result.gflops = (flops / 1e9) / (result.time_ms / 1000.0);

    // Bytes read = m*k (A) + k*n/2 (B quantized, ~4 bits/element) + m*n (C read for beta)
    // Bytes write = m*n (C)
    // Simplified: (m*k + k*n/2 + 2*m*n) * 4 bytes
    double bytes = (static_cast<double>(m) * k + static_cast<double>(k) * n / 2.0 + 2.0 * m * n) * 4.0;
    result.bandwidth_gb_s = (bytes / 1e9) / (result.time_ms / 1000.0);

    return result;
}

/**
 * @brief Parse command line arguments
 */
struct BenchmarkConfig
{
    std::string kernel_type = "baseline";
    int m = 256;
    int n = 5120;
    int k = 5120;
    int tile_m = 128;
    int tile_n = 128;
    int tile_k = 64;
    int warmup = 10;
    int iterations = 100;
};

BenchmarkConfig parseArgs(int argc, char **argv)
{
    BenchmarkConfig config;

    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];

        if (arg.find("--kernel=") == 0)
        {
            config.kernel_type = arg.substr(9);
        }
        else if (arg.find("--m=") == 0)
        {
            config.m = std::stoi(arg.substr(4));
        }
        else if (arg.find("--n=") == 0)
        {
            config.n = std::stoi(arg.substr(4));
        }
        else if (arg.find("--k=") == 0)
        {
            config.k = std::stoi(arg.substr(4));
        }
        else if (arg.find("--tile_m=") == 0)
        {
            config.tile_m = std::stoi(arg.substr(9));
        }
        else if (arg.find("--tile_n=") == 0)
        {
            config.tile_n = std::stoi(arg.substr(9));
        }
        else if (arg.find("--tile_k=") == 0)
        {
            config.tile_k = std::stoi(arg.substr(9));
        }
        else if (arg.find("--warmup=") == 0)
        {
            config.warmup = std::stoi(arg.substr(9));
        }
        else if (arg.find("--iterations=") == 0)
        {
            config.iterations = std::stoi(arg.substr(13));
        }
    }

    return config;
}

int main(int argc, char **argv)
{
    // Parse arguments
    BenchmarkConfig bench_config = parseArgs(argc, argv);

    const int m = bench_config.m;
    const int n = bench_config.n;
    const int k = bench_config.k;

    // Create kernel configuration
    CudaGemmConfig kernel_config;
    kernel_config.tile_m = bench_config.tile_m;
    kernel_config.tile_n = bench_config.tile_n;
    kernel_config.tile_k = bench_config.tile_k;
    kernel_config.threads_m = bench_config.tile_m / 16; // Assuming WORK_M = 16
    kernel_config.threads_n = bench_config.tile_n / 16; // Assuming WORK_N = 16
    kernel_config.work_m = 16;
    kernel_config.work_n = 16;
    kernel_config.prefetch_stages = 0;
    kernel_config.transpose_smem = false;
    kernel_config.vectorize_load = 4; // Phase 1: Enable vectorized loads

    // Allocate host memory
    std::vector<float> h_A(m * k);
    auto h_B_blocks = generateRandomIQ4NL(n, k);
    std::vector<float> h_C(m * n, 0.0f);

    // Initialize A with random values
    for (auto &val : h_A)
    {
        val = static_cast<float>(rand()) / RAND_MAX - 0.5f;
    }

    // Allocate device memory
    float *d_A, *d_C;
    IQ4_NLBlock *d_B_blocks;

    CUDA_CHECK(cudaMalloc(&d_A, m * k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B_blocks, h_B_blocks.size() * sizeof(IQ4_NLBlock)));
    CUDA_CHECK(cudaMalloc(&d_C, m * n * sizeof(float)));

    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), m * k * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B_blocks, h_B_blocks.data(), h_B_blocks.size() * sizeof(IQ4_NLBlock), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C, 0, m * n * sizeof(float)));

    // Run benchmark
    auto result = benchmarkKernel(
        bench_config.kernel_type,
        d_A, d_B_blocks, d_C,
        m, n, k,
        kernel_config,
        bench_config.warmup,
        bench_config.iterations);

    // Output results as JSON
    json output;
    output["kernel_type"] = bench_config.kernel_type;
    output["m"] = m;
    output["n"] = n;
    output["k"] = k;
    output["tile_m"] = kernel_config.tile_m;
    output["tile_n"] = kernel_config.tile_n;
    output["tile_k"] = kernel_config.tile_k;
    output["time_ms"] = result.time_ms;
    output["gflops"] = result.gflops;
    output["bandwidth_gb_s"] = result.bandwidth_gb_s;
    output["iterations"] = result.iterations;

    std::cout << output.dump(2) << std::endl;

    // Cleanup
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B_blocks));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
