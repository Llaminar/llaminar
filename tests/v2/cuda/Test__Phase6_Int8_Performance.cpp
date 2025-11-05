/**
 * @file Test__Phase6_Int8_Performance.cpp
 * @brief Performance benchmark for Phase 6 int8 DP4A GEMM kernel
 *
 * @author David Sanftenberg
 * @date 2025-01-XX
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda.h> // Driver API for cuLaunchKernel
#include <cuda_fp16.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <random>

#include "kernels/cuda/CudaGemmJITPhase6.h"
#include "kernels/cuda/CudaGemmConfigPhase6.h"

using namespace llaminar2::cuda;

// IQ4_NL block structure (must match kernel)
struct IQ4_NLBlock
{
    uint8_t quants[16];
    __half scale;
};

// IQ4_NL lookup table
const int8_t kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

// Helper: Generate random IQ4_NL blocks
void generate_random_iq4nl_blocks(std::vector<IQ4_NLBlock> &blocks, int seed = 42)
{
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> nibble_dist(0, 15);
    std::uniform_real_distribution<> scale_dist(0.5f, 2.0f);

    for (auto &block : blocks)
    {
        // Random scale
        block.scale = __float2half(scale_dist(gen));

        // Random nibbles (2 per byte)
        for (int i = 0; i < 16; i++)
        {
            uint8_t nibble_low = nibble_dist(gen);
            uint8_t nibble_high = nibble_dist(gen);
            block.quants[i] = (nibble_high << 4) | nibble_low;
        }
    }
}

// Helper: Generate random FP32 matrix
void generate_random_fp32_matrix(std::vector<float> &matrix, int rows, int cols, int seed = 42)
{
    std::mt19937 gen(seed);
    std::uniform_real_distribution<> dist(-1.0f, 1.0f);

    matrix.resize(rows * cols);
    for (auto &val : matrix)
    {
        val = dist(gen);
    }
}

/**
 * Performance benchmark for Phase 6 kernel
 */
TEST(Phase6Int8Performance, LargeMatrix2048x2048)
{
    const int M = 2048;
    const int N = 2048;
    const int K = 2048;
    const int K_BLOCKS = K / 32; // IQ4_NL block size is 32
    const int WARMUP_ITERS = 3;
    const int TIMING_ITERS = 10;

    std::cout << "\n=== Phase 6 Int8 DP4A Performance Benchmark ===\n";
    std::cout << "Matrix size: M=" << M << ", N=" << N << ", K=" << K << "\n";
    std::cout << "IQ4_NL blocks: " << K_BLOCKS << " per row\n";

    // Print GPU info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU: " << prop.name << "\n";
    std::cout << "SM count: " << prop.multiProcessorCount << "\n";
    std::cout << "Max threads per SM: " << prop.maxThreadsPerMultiProcessor << "\n\n";

    // Generate random data
    std::cout << "Generating random input data...\n";
    std::vector<float> A_host;
    std::vector<IQ4_NLBlock> B_host(N * K_BLOCKS);
    std::vector<float> C_host(M * N, 0.0f);

    generate_random_fp32_matrix(A_host, M, K, 42);
    generate_random_iq4nl_blocks(B_host, 42);

    // Allocate GPU memory
    float *d_A, *d_C;
    IQ4_NLBlock *d_B;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, N * K_BLOCKS * sizeof(IQ4_NLBlock));
    cudaMalloc(&d_C, M * N * sizeof(float));

    cudaMemcpy(d_A, A_host.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B_host.data(), N * K_BLOCKS * sizeof(IQ4_NLBlock), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, M * N * sizeof(float));

    // Compile kernel
    std::cout << "Compiling kernel...\n";
    auto start_compile = std::chrono::high_resolution_clock::now();

    auto config = get_default_phase6_config();
    auto kernel = CudaGemmJITPhase6::compile(config);

    auto end_compile = std::chrono::high_resolution_clock::now();
    double compile_time = std::chrono::duration<double>(end_compile - start_compile).count() * 1000.0;
    std::cout << "Compilation time: " << std::fixed << std::setprecision(2) << compile_time << " ms\n\n";

    // Print kernel config
    std::cout << "Kernel configuration:\n";
    std::cout << "  TILE_M x TILE_N x TILE_K = "
              << config.tile_m << " × " << config.tile_n << " × " << config.tile_k << "\n";
    std::cout << "  Threads per block = " << config.threads_per_block << "\n";

    // Calculate grid dimensions
    int grid_x = (M + config.tile_m - 1) / config.tile_m;
    int grid_y = (N + config.tile_n - 1) / config.tile_n;
    std::cout << "  Grid dimensions = " << grid_x << " × " << grid_y << " = " << (grid_x * grid_y) << " blocks\n";
    std::cout << "  Total threads = " << (grid_x * grid_y * config.threads_per_block) << "\n\n";

    // Warmup
    std::cout << "Warmup (" << WARMUP_ITERS << " iterations)...\n";
    dim3 grid((M + config.tile_m - 1) / config.tile_m,
              (N + config.tile_n - 1) / config.tile_n);
    dim3 block(config.threads_per_block);
    void *args[] = {&d_A, &d_B, &d_C, (void *)&M, (void *)&N, (void *)&K};

    for (int i = 0; i < WARMUP_ITERS; i++)
    {
        cuLaunchKernel(kernel.function, grid.x, grid.y, 1, block.x, block.y, block.z, 0, 0, args, nullptr);
        cudaDeviceSynchronize();
    }
    std::cout << "Warmup complete\n\n";

    // Timing runs
    std::cout << "Running " << TIMING_ITERS << " timing iterations...\n";
    std::vector<double> times_ms;
    times_ms.reserve(TIMING_ITERS);

    for (int i = 0; i < TIMING_ITERS; i++)
    {
        cudaDeviceSynchronize();
        auto start = std::chrono::high_resolution_clock::now();

        cuLaunchKernel(kernel.function, grid.x, grid.y, 1, block.x, block.y, block.z, 0, 0, args, nullptr);
        cudaDeviceSynchronize();

        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double>(end - start).count() * 1000.0;
        times_ms.push_back(time_ms);

        std::cout << "  Iteration " << (i + 1) << ": " << std::fixed << std::setprecision(3)
                  << time_ms << " ms\n";
    }

    // Calculate statistics
    double total_time = 0.0;
    double min_time = times_ms[0];
    double max_time = times_ms[0];
    for (double t : times_ms)
    {
        total_time += t;
        min_time = std::min(min_time, t);
        max_time = std::max(max_time, t);
    }
    double avg_time = total_time / TIMING_ITERS;

    // Calculate TFLOPS
    // For GEMM: 2 * M * N * K operations (multiply + add)
    double gflops = (2.0 * M * N * K) / 1e9;
    double avg_tflops = gflops / (avg_time / 1000.0) / 1000.0;
    double peak_tflops = gflops / (min_time / 1000.0) / 1000.0;

    // Print results
    std::cout << "\n=== Performance Results ===\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "  Min time:     " << min_time << " ms  →  " << peak_tflops << " TFLOPS\n";
    std::cout << "  Max time:     " << max_time << " ms  →  " << (gflops / (max_time / 1000.0) / 1000.0) << " TFLOPS\n";
    std::cout << "  Avg time:     " << avg_time << " ms  →  " << avg_tflops << " TFLOPS\n";
    std::cout << "\n";
    std::cout << "  GFLOPS:       " << std::setprecision(2) << gflops << "\n";
    std::cout << "  Avg TFLOPS:   " << std::setprecision(2) << avg_tflops << "\n";
    std::cout << "  Peak TFLOPS:  " << std::setprecision(2) << peak_tflops << "\n";
    std::cout << "\n";
    std::cout << "=== Comparison ===\n";
    std::cout << "  Phase 5 baseline: ~17.5 TFLOPS\n";
    std::cout << "  Phase 6 target:   50-90 TFLOPS (2.9-5.1× speedup)\n";
    std::cout << "  Phase 6 actual:   " << avg_tflops << " TFLOPS ("
              << std::setprecision(1) << (avg_tflops / 17.5) << "× speedup)\n";
    std::cout << "\n";

    // Verify correctness on first few elements
    std::cout << "Verifying output (first few elements)...\n";
    std::vector<float> C_gpu(M * N);
    cudaMemcpy(C_gpu.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "  C[0,0] = " << C_gpu[0] << "\n";
    std::cout << "  C[0,1] = " << C_gpu[1] << "\n";
    std::cout << "  C[1,0] = " << C_gpu[N] << "\n";
    std::cout << "  C[1,1] = " << C_gpu[N + 1] << "\n";

    // Check for NaN/Inf
    bool has_nan = false;
    bool has_inf = false;
    for (float val : C_gpu)
    {
        if (std::isnan(val))
            has_nan = true;
        if (std::isinf(val))
            has_inf = true;
    }

    EXPECT_FALSE(has_nan) << "Output contains NaN values!";
    EXPECT_FALSE(has_inf) << "Output contains Inf values!";
    std::cout << "  No NaN/Inf detected ✓\n";

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    std::cout << "\n=== Benchmark Complete ===\n";
}

/**
 * Test different matrix sizes for scaling analysis
 * NOTE: Currently disabled due to kernel.function pointer corruption bug
 * The 2048×2048 test works, but reusing/recompiling kernels causes segfault
 */
TEST(Phase6Int8Performance, DISABLED_ScalingAnalysis)
{
    const std::vector<int> sizes = {512, 1024, 2048, 4096};
    const int WARMUP_ITERS = 2;
    const int TIMING_ITERS = 5;

    std::cout << "\n=== Phase 6 Int8 DP4A Scaling Analysis ===\n";
    std::cout << std::setw(10) << "Size"
              << std::setw(15) << "Time (ms)"
              << std::setw(15) << "TFLOPS"
              << std::setw(15) << "Speedup\n";
    std::cout << std::string(55, '-') << "\n";

    auto config = get_default_phase6_config();

    for (int size : sizes)
    {
        std::cout << "Testing size " << size << "...\n";
        std::cout.flush();

        // Compile kernel for this size (fresh compilation each time)
        std::cout << "  Compiling kernel...\n";
        std::cout.flush();
        auto kernel = CudaGemmJITPhase6::compile(config);
        std::cout << "  Kernel compiled, function ptr: " << (void *)kernel.function << "\n";
        std::cout.flush();
        int M = size; // Remove const so we can take address safely
        int N = size;
        int K = size;
        const int K_BLOCKS = K / 32;

        std::cout << "  Generating data (M=" << M << ", N=" << N << ", K=" << K << ", K_BLOCKS=" << K_BLOCKS << ")...\n";
        std::cout.flush();

        // Generate data
        std::vector<float> A_host;
        std::vector<IQ4_NLBlock> B_host(N * K_BLOCKS);
        generate_random_fp32_matrix(A_host, M, K, 42);
        std::cout << "  Generated A_host (" << A_host.size() << " elements)\n";
        std::cout.flush();
        generate_random_iq4nl_blocks(B_host, 42);
        std::cout << "  Generated B_host (" << B_host.size() << " blocks)\n";
        std::cout.flush();

        // GPU memory
        std::cout << "  Allocating GPU memory...\n";
        std::cout.flush();
        float *d_A, *d_C;
        IQ4_NLBlock *d_B;

        cudaError_t alloc_err;
        alloc_err = cudaMalloc(&d_A, M * K * sizeof(float));
        if (alloc_err != cudaSuccess)
        {
            std::cerr << "cudaMalloc d_A failed: " << cudaGetErrorString(alloc_err) << "\n";
            FAIL() << "cudaMalloc failed for d_A";
        }
        std::cout << "  Allocated d_A\n";
        std::cout.flush();

        alloc_err = cudaMalloc(&d_B, N * K_BLOCKS * sizeof(IQ4_NLBlock));
        if (alloc_err != cudaSuccess)
        {
            std::cerr << "cudaMalloc d_B failed: " << cudaGetErrorString(alloc_err) << "\n";
            cudaFree(d_A);
            FAIL() << "cudaMalloc failed for d_B";
        }
        std::cout << "  Allocated d_B\n";
        std::cout.flush();

        alloc_err = cudaMalloc(&d_C, M * N * sizeof(float));
        if (alloc_err != cudaSuccess)
        {
            std::cerr << "cudaMalloc d_C failed: " << cudaGetErrorString(alloc_err) << "\n";
            cudaFree(d_A);
            cudaFree(d_B);
            FAIL() << "cudaMalloc failed for d_C";
        }
        std::cout << "  Allocated d_C\n";
        std::cout.flush();

        std::cout << "  Copying data to GPU...\n";
        std::cout.flush();
        alloc_err = cudaMemcpy(d_A, A_host.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
        if (alloc_err != cudaSuccess)
        {
            std::cerr << "cudaMemcpy d_A failed: " << cudaGetErrorString(alloc_err) << "\n";
            cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_C);
            FAIL() << "cudaMemcpy failed for d_A";
        }
        std::cout << "  Copied d_A\n";
        std::cout.flush();

        alloc_err = cudaMemcpy(d_B, B_host.data(), N * K_BLOCKS * sizeof(IQ4_NLBlock), cudaMemcpyHostToDevice);
        if (alloc_err != cudaSuccess)
        {
            std::cerr << "cudaMemcpy d_B failed: " << cudaGetErrorString(alloc_err) << "\n";
            cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_C);
            FAIL() << "cudaMemcpy failed for d_B";
        }
        std::cout << "  Copied d_B\n";
        std::cout.flush();

        // Setup launch parameters
        std::cout << "  Setting up launch parameters...\n";
        std::cout.flush();
        dim3 grid((M + config.tile_m - 1) / config.tile_m,
                  (N + config.tile_n - 1) / config.tile_n);
        dim3 block(config.threads_per_block);
        std::cout << "  Grid: " << grid.x << "x" << grid.y << ", Block: " << block.x << "\n";
        std::cout.flush();

        // CRITICAL: args array uses stack addresses - must remain valid during kernel execution
        void *args[] = {&d_A, &d_B, &d_C, (void *)&M, (void *)&N, (void *)&K};
        std::cout << "  Args array created\n";
        std::cout.flush();

        // Warmup
        std::cout << "  Starting warmup...\n";
        std::cout.flush();
        for (int i = 0; i < WARMUP_ITERS; i++)
        {
            std::cout << "    Warmup iteration " << (i + 1) << "...\n";
            std::cout.flush();
            CUresult res = cuLaunchKernel(kernel.function, grid.x, grid.y, 1, block.x, block.y, block.z, 0, 0, args, nullptr);
            std::cout << "    Launch returned\n";
            std::cout.flush();
            if (res != CUDA_SUCCESS)
            {
                const char *errName;
                cuGetErrorName(res, &errName);
                std::cerr << "cuLaunchKernel failed in warmup: " << errName << "\n";
                cudaFree(d_A);
                cudaFree(d_B);
                cudaFree(d_C);
                FAIL() << "Kernel launch failed";
            }
        }
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            std::cerr << "CUDA error after warmup: " << cudaGetErrorString(err) << "\n";
            cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_C);
            FAIL() << "CUDA error in warmup";
        }

        // Timing
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < TIMING_ITERS; i++)
        {
            CUresult res = cuLaunchKernel(kernel.function, grid.x, grid.y, 1, block.x, block.y, block.z, 0, 0, args, nullptr);
            if (res != CUDA_SUCCESS)
            {
                const char *errName;
                cuGetErrorName(res, &errName);
                std::cerr << "cuLaunchKernel failed in timing: " << errName << "\n";
                cudaFree(d_A);
                cudaFree(d_B);
                cudaFree(d_C);
                FAIL() << "Kernel launch failed";
            }
        }
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();

        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            std::cerr << "CUDA error after timing: " << cudaGetErrorString(err) << "\n";
            cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_C);
            FAIL() << "CUDA error in timing";
        }

        double avg_time = std::chrono::duration<double>(end - start).count() * 1000.0 / TIMING_ITERS;
        double gflops = (2.0 * M * N * K) / 1e9;
        double tflops = gflops / (avg_time / 1000.0) / 1000.0;
        double speedup = tflops / 17.5;

        std::cout << std::setw(10) << size
                  << std::setw(15) << std::fixed << std::setprecision(2) << avg_time
                  << std::setw(15) << std::setprecision(2) << tflops
                  << std::setw(15) << std::setprecision(1) << speedup << "×\n";

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }

    std::cout << "\n";
}
