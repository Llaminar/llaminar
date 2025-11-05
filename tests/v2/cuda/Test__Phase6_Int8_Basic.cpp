/**
 * @file Test__Phase6_Int8_Basic.cpp
 * @brief Basic correctness and performance test for Phase 6 Int8 DP4A kernel
 *
 * @author David Sanftenberg
 * @date November 5, 2025
 */

#include <gtest/gtest.h>
#include "kernels/cuda/CudaGemmJITPhase6.h"
#include "kernels/cuda/CudaGemmConfigPhase6.h"
#include "tensors/Tensors.h"
#include <cuda_runtime.h>
#include <chrono>
#include <iostream>
#include <iomanip>

using namespace llaminar2;
using namespace llaminar2::cuda;

// CUDA error checking macro
#define CUDA_CHECK(call)                                                         \
    {                                                                            \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess)                                                  \
        {                                                                        \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl;                   \
            abort();                                                             \
        }                                                                        \
    }

class Phase6Int8Basic : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Check CUDA availability
        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        if (device_count == 0)
        {
            GTEST_SKIP() << "No CUDA devices available";
        }
    }
};

/**
 * @brief Test small problem: 256×256×256
 */
TEST_F(Phase6Int8Basic, SmallProblem)
{
    const int M = 256;
    const int N = 256;
    const int K = 256;
    const int K_blocks = K / 32;

    std::cout << "\n========================================\n";
    std::cout << "Phase 6 Int8 DP4A - Small Problem Test\n";
    std::cout << "========================================\n";
    std::cout << "Size: M=" << M << ", N=" << N << ", K=" << K << "\n";
    std::cout << "K blocks: " << K_blocks << "\n\n";

    // Create tensors
    auto A_fp32 = std::make_shared<FP32Tensor>(std::vector<size_t>{(size_t)M, (size_t)K});

    // Initialize A
    A_fp32->randomize(-1.0f, 1.0f);

    // Create B tensor
    size_t num_blocks = (N * K) / 32;
    std::vector<uint8_t> b_raw_data(num_blocks * (16 + 2));
    for (size_t i = 0; i < b_raw_data.size(); i++)
    {
        b_raw_data[i] = rand() % 256;
    }

    auto B_iq4nl = std::make_shared<IQ4_NLTensor>(std::vector<size_t>{(size_t)N, (size_t)K}, b_raw_data);
    auto C_result = std::make_shared<FP32Tensor>(std::vector<size_t>{(size_t)M, (size_t)N});
    C_result->zero();

    // Get device pointers
    const float *d_A = A_fp32->device_data();
    const void *d_B = B_iq4nl->device_data();
    float *d_C = C_result->device_mutable_data();

    // Get default config
    auto config = get_default_phase6_config();
    ASSERT_TRUE(config.is_valid()) << "Invalid configuration";

    std::cout << "Config: tile_m=" << config.tile_m
              << ", tile_n=" << config.tile_n
              << ", tile_k=" << config.tile_k
              << ", threads=" << config.threads_per_block << "\n\n";

    // JIT compile
    std::cout << "JIT compiling kernel...\n";
    auto kernel = CudaGemmJITPhase6::compile(config);
    std::cout << "✓ Compilation successful\n\n";

    // Launch kernel
    dim3 grid(
        (M + config.tile_m - 1) / config.tile_m,
        (N + config.tile_n - 1) / config.tile_n);
    dim3 block(config.threads_per_block);

    std::cout << "Grid: (" << grid.x << ", " << grid.y << "), Block: " << block.x << "\n";

    // Warmup
    kernel.launch(grid, block, d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    const int iterations = 10;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; i++)
    {
        kernel.launch(grid, block, d_A, d_B, d_C, M, N, K);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count() / iterations;

    // Calculate TFLOPS
    double flops = 2.0 * M * N * K; // FMA = 2 ops
    double tflops = (flops / elapsed_ms) / 1e9;

    std::cout << "\nPerformance:\n";
    std::cout << "  Time:   " << std::fixed << std::setprecision(3) << elapsed_ms << " ms\n";
    std::cout << "  TFLOPS: " << std::fixed << std::setprecision(2) << tflops << "\n\n";

    // Basic correctness check - verify no NaNs or Infs
    C_result->to_host();
    const float *h_C = C_result->host_data();

    int nan_count = 0;
    int inf_count = 0;
    for (size_t i = 0; i < M * N; i++)
    {
        if (std::isnan(h_C[i]))
            nan_count++;
        if (std::isinf(h_C[i]))
            inf_count++;
    }

    std::cout << "Correctness check:\n";
    std::cout << "  NaN count: " << nan_count << "/" << M * N << "\n";
    std::cout << "  Inf count: " << inf_count << "/" << M * N << "\n";

    EXPECT_EQ(nan_count, 0) << "Output contains NaN values";
    EXPECT_EQ(inf_count, 0) << "Output contains Inf values";

    // Basic sanity check - output should be non-zero
    double sum = 0.0;
    for (size_t i = 0; i < M * N; i++)
    {
        sum += std::abs(h_C[i]);
    }
    double avg_abs = sum / (M * N);

    std::cout << "  Avg abs value: " << avg_abs << "\n";
    EXPECT_GT(avg_abs, 1e-6) << "Output is suspiciously close to zero";

    std::cout << "\n✓ Test passed!\n";
    std::cout << "========================================\n\n";
}

/**
 * @brief Test larger problem: 1024×1024×1024
 */
TEST_F(Phase6Int8Basic, LargeProblem)
{
    const int M = 1024;
    const int N = 1024;
    const int K = 1024;
    const int K_blocks = K / 32;

    std::cout << "\n========================================\n";
    std::cout << "Phase 6 Int8 DP4A - Large Problem Test\n";
    std::cout << "========================================\n";
    std::cout << "Size: M=" << M << ", N=" << N << ", K=" << K << "\n";
    std::cout << "K blocks: " << K_blocks << "\n\n";

    // Create tensors
    auto A_fp32 = std::make_shared<FP32Tensor>(std::vector<size_t>{(size_t)M, (size_t)K});
    auto B_iq4nl = std::make_shared<IQ4_NLTensor>(std::vector<size_t>{(size_t)N, (size_t)K});
    auto C_result = std::make_shared<FP32Tensor>(std::vector<size_t>{(size_t)M, (size_t)N});

    // Initialize
    A_fp32->randomize(-1.0f, 1.0f);
    B_iq4nl->randomize(-1.0f, 1.0f);
    C_result->zero();

    // Get device pointers
    const float *d_A = A_fp32->device_data();
    const void *d_B = B_iq4nl->device_data();
    float *d_C = C_result->device_mutable_data();

    // Get config
    auto config = get_phase6_config_for_size(M, N, K);
    ASSERT_TRUE(config.is_valid());

    // JIT compile
    std::cout << "JIT compiling...\n";
    auto kernel = CudaGemmJITPhase6::compile(config);
    std::cout << "✓ Compiled\n\n";

    // Launch
    dim3 grid(
        (M + config.tile_m - 1) / config.tile_m,
        (N + config.tile_n - 1) / config.tile_n);
    dim3 block(config.threads_per_block);

    std::cout << "Grid: (" << grid.x << ", " << grid.y << "), Block: " << block.x << "\n";

    // Warmup
    kernel.launch(grid, block, d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    const int iterations = 20;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; i++)
    {
        kernel.launch(grid, block, d_A, d_B, d_C, M, N, K);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count() / iterations;

    // Calculate TFLOPS
    double flops = 2.0 * M * N * K;
    double tflops = (flops / elapsed_ms) / 1e9;

    std::cout << "\nPerformance:\n";
    std::cout << "  Time:   " << std::fixed << std::setprecision(3) << elapsed_ms << " ms\n";
    std::cout << "  TFLOPS: " << std::fixed << std::setprecision(2) << tflops << "\n";

    // Expected: 50-90 TFLOPS on RTX 3090
    std::cout << "\nExpected range: 50-90 TFLOPS (RTX 3090)\n";

    if (tflops >= 50.0)
    {
        std::cout << "✅ EXCELLENT: Achieved target performance!\n";
    }
    else if (tflops >= 30.0)
    {
        std::cout << "✓ GOOD: Approaching target performance\n";
    }
    else
    {
        std::cout << "⚠️ BELOW TARGET: Needs optimization\n";
    }

    std::cout << "========================================\n\n";
}
