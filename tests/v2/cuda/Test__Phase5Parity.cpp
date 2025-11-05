/**
 * @file Test__Phase5Parity.cpp
 * @brief Phase 5 JIT kernel parity verification against baseline
 *
 * Validates that JIT-compiled Phase 5A kernel achieves same performance
 * as the original nvcc-compiled Phase 5A implementation (8.86 TFLOPS).
 *
 * @author David Sanftenberg
 * @date November 4, 2025
 */

#include "CudaGemmJITPhase5.h"
#include "Phase5ConfigSpace.h"
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <iomanip>

using namespace llaminar2::cuda;

class Phase5ParityTest : public ::testing::Test
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

        // Initialize CUDA context and ensure primary context is active
        CUdevice device;
        cuDeviceGet(&device, 0);
        CUcontext ctx;
        cuDevicePrimaryCtxRetain(&ctx, device);
        cuCtxSetCurrent(ctx);

        // Same size as original Phase 5A benchmark
        M_ = 1024;
        N_ = 896;
        K_ = 896;

        // Allocate device memory (Phase 5 uses FP32 input, not FP16!)
        cudaMalloc(&d_A_, M_ * K_ * sizeof(float));
        int blocks_per_row = (K_ + 31) / 32;
        cudaMalloc(&d_B_iq4nl_, N_ * blocks_per_row * 18); // IQ4_NL
        cudaMalloc(&d_C_, M_ * N_ * sizeof(float));

        // Initialize activations in COLUMN-MAJOR format for coalesced global loads
        // Store as A[K][M] so consecutive threads access consecutive elements
        std::vector<float> h_A(M_ * K_);
        std::vector<uint8_t> h_B(N_ * blocks_per_row * 18);

        // Fill column-major: A[k * M + m] = value
        // This makes consecutive elements in memory come from same K but different M
        for (int k = 0; k < K_; ++k)
        {
            for (int m = 0; m < M_; ++m)
            {
                h_A[k * M_ + m] = 0.1f;
            }
        }

        for (auto &v : h_B)
            v = 0x42;

        cudaMemcpy(d_A_, h_A.data(), M_ * K_ * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B_iq4nl_, h_B.data(), N_ * blocks_per_row * 18, cudaMemcpyHostToDevice);

        // Force GPU initialization with a dummy kernel
        cudaDeviceSynchronize();
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

    float benchmarkKernel(CUfunction kernel, const CudaGemmConfigPhase5 &config, int num_iters = 100)
    {
        int blocks_m = (M_ + config.tile_m - 1) / config.tile_m;
        int blocks_n = (N_ + config.tile_n - 1) / config.tile_n;
        dim3 grid(blocks_m, blocks_n); // FIX: grid.x = M blocks, grid.y = N blocks (match Phase 5A)
        dim3 block(config.threads_per_block);

        void *args[] = {&d_A_, &d_B_iq4nl_, &d_C_, &M_, &N_, &K_};

        // Warmup (20 iterations to ensure steady state) - NO PRINTF
        for (int i = 0; i < 20; ++i)
        {
            CUresult res = cuLaunchKernel(kernel,
                                          grid.x, grid.y, grid.z,
                                          block.x, block.y, block.z,
                                          0, nullptr, args, nullptr);
            if (res != CUDA_SUCCESS)
            {
                const char *err_str;
                cuGetErrorString(res, &err_str);
                throw std::runtime_error(std::string("cuLaunchKernel warmup failed: ") + err_str);
            }
        }
        cudaDeviceSynchronize();

        // Timed run - MINIMAL overhead
        CUevent start, stop;
        cuEventCreate(&start, CU_EVENT_DEFAULT);
        cuEventCreate(&stop, CU_EVENT_DEFAULT);

        cuEventRecord(start, nullptr);
        for (int i = 0; i < num_iters; ++i)
        {
            cuLaunchKernel(kernel,
                           grid.x, grid.y, grid.z,
                           block.x, block.y, block.z,
                           0, nullptr, args, nullptr);
        }
        cuEventRecord(stop, nullptr);
        cuEventSynchronize(stop);

        float total_ms = 0.0f;
        cuEventElapsedTime(&total_ms, start, stop);

        cuEventDestroy(start);
        cuEventDestroy(stop);

        return total_ms / num_iters;
    }

    int M_, N_, K_;
    float *d_A_ = nullptr;
    void *d_B_iq4nl_ = nullptr;
    float *d_C_ = nullptr;
};

TEST_F(Phase5ParityTest, Phase5A_Baseline_Config)
{
    // Original Phase 5A configuration that achieved 8.86 TFLOPS
    // IMPORTANT: Updated to valid configuration (64x64x64 with streaming)
    // The 128x128x64 config with double buffer exceeds 48KB shared memory limit!
    // From Phase5ConfigSpace.h focused configs:
    // - TILE_M=64, TILE_N=64, TILE_K=64 (baseline Phase 4 size)
    // - SUB_K=16 (streaming 16-elem sub-tiles for Phase 5A)
    // - MMA 2x2 (16x16x16 tensor cores)
    // - BUFFER_STAGES=2 (double buffering - current Phase 5A)
    // - THREADS_PER_BLOCK=128 (32 * 2 * 2)
    // - Swizzle 3,3,3 (shared memory bank conflict avoidance)

    Phase5GemmConfig baseline_config{
        .tile_m = 64,
        .tile_n = 64,
        .tile_k = 64,
        .sub_k = 16, // Streaming sub-tiles (actual Phase 5A feature)
        .mma_m = 2,
        .mma_n = 2,
        .buffer_stages = 2,
        .threads_per_block = 128,
        .swizzle_b = 3,
        .swizzle_m = 3,
        .swizzle_s = 3};

    CudaGemmConfigPhase5 full_config(
        baseline_config.tile_m, baseline_config.tile_n, baseline_config.tile_k,
        baseline_config.sub_k,
        baseline_config.mma_m, baseline_config.mma_n,
        baseline_config.buffer_stages);

    std::cout << "\n========================================\n";
    std::cout << "Phase 5A Baseline Configuration\n";
    std::cout << "========================================\n";
    std::cout << "Config: " << full_config.config_id() << "\n";
    std::cout << "Shared Memory: " << full_config.shared_memory_bytes() / 1024 << " KB\n";
    std::cout << "Occupancy: " << full_config.estimate_occupancy_blocks_per_sm() << " blocks/SM\n";
    std::cout << "Matrix: " << M_ << "x" << N_ << "x" << K_ << "\n\n";

    // Get JIT-compiled kernel (first call compiles, subsequent calls are cached)
    auto &jit = CudaGemmJITPhase5::instance();

    std::cout << "Step 1: JIT compilation (first call)...\n";
    auto start_compile = std::chrono::high_resolution_clock::now();
    CUfunction kernel = jit.getKernel(full_config);
    auto end_compile = std::chrono::high_resolution_clock::now();
    double compile_ms = std::chrono::duration<double, std::milli>(end_compile - start_compile).count();
    std::cout << "  Compilation time: " << std::fixed << std::setprecision(2)
              << compile_ms << " ms\n\n";

    // Verify kernel is cached for subsequent calls
    std::cout << "Step 2: Verify caching (second call)...\n";
    auto start_cached = std::chrono::high_resolution_clock::now();
    CUfunction kernel_cached = jit.getKernel(full_config);
    auto end_cached = std::chrono::high_resolution_clock::now();
    double cached_ms = std::chrono::duration<double, std::milli>(end_cached - start_cached).count();
    std::cout << "  Cached lookup time: " << std::fixed << std::setprecision(6)
              << cached_ms << " ms\n";
    std::cout << "  Speedup: " << (compile_ms / cached_ms) << "x\n\n";

    EXPECT_EQ(kernel, kernel_cached) << "Cached kernel should be identical";

    // Benchmark performance - RUNTIME API (match Phase 5A exactly)
    std::cout << "Step 3: Runtime API timing (match Phase 5A methodology)...\n";

    // Zero output
    cudaMemset(d_C_, 0, M_ * N_ * sizeof(float));
    cudaDeviceSynchronize();

    int blocks_m = (M_ + full_config.tile_m - 1) / full_config.tile_m;
    int blocks_n = (N_ + full_config.tile_n - 1) / full_config.tile_n;
    dim3 grid(blocks_m, blocks_n);
    dim3 block(full_config.threads_per_block);
    void *args[] = {&d_A_, &d_B_iq4nl_, &d_C_, &M_, &N_, &K_};

    // Warmup
    for (int i = 0; i < 20; i++)
    {
        cuLaunchKernel(kernel, grid.x, grid.y, grid.z,
                       block.x, block.y, block.z,
                       0, nullptr, args, nullptr);
    }
    cudaDeviceSynchronize();

    // Runtime API events (like Phase 5A)
    cudaEvent_t start_rt, stop_rt;
    cudaEventCreate(&start_rt);
    cudaEventCreate(&stop_rt);

    cudaEventRecord(start_rt);
    for (int i = 0; i < 50; i++)
    { // Match Phase 5A's 50 iterations
        cuLaunchKernel(kernel, grid.x, grid.y, grid.z,
                       block.x, block.y, block.z,
                       0, nullptr, args, nullptr);
    }
    cudaEventRecord(stop_rt);
    cudaEventSynchronize(stop_rt);

    float time_rt_ms;
    cudaEventElapsedTime(&time_rt_ms, start_rt, stop_rt);
    float avg_rt = time_rt_ms / 50.0f;

    cudaEventDestroy(start_rt);
    cudaEventDestroy(stop_rt);

    double flops_rt = 2.0 * M_ * N_ * K_;
    double tflops_rt = flops_rt / (avg_rt * 1e9);

    std::cout << "  Runtime API (50 iters): " << std::fixed << std::setprecision(4) << avg_rt << " ms\n";
    std::cout << "  Throughput: " << std::fixed << std::setprecision(2) << tflops_rt << " TFLOPS\n\n";

    float avg_ms = avg_rt;

    // Calculate TFLOPS
    double flops = 2.0 * M_ * N_ * K_; // MAC = multiply + add
    double tflops = flops / (avg_ms * 1e9);

    std::cout << "  Average time: " << std::fixed << std::setprecision(4) << avg_ms << " ms\n";
    std::cout << "  Throughput: " << std::fixed << std::setprecision(2) << tflops << " TFLOPS\n\n";

    // Parity check: JIT kernel should match nvcc-compiled Phase 5A
    constexpr float PHASE_5A_BASELINE = 8.86f; // From Phase 5A analysis
    constexpr float TOLERANCE = 0.15f;         // ±15% tolerance (accounting for JIT vs nvcc)

    std::cout << "========================================\n";
    std::cout << "Parity Analysis\n";
    std::cout << "========================================\n";
    std::cout << "JIT kernel:        " << std::fixed << std::setprecision(2) << tflops << " TFLOPS\n";
    std::cout << "Phase 5A baseline: " << PHASE_5A_BASELINE << " TFLOPS\n";
    std::cout << "Difference:        " << std::showpos << (tflops - PHASE_5A_BASELINE) << " TFLOPS ("
              << std::noshowpos << ((tflops / PHASE_5A_BASELINE - 1.0) * 100.0) << "%)\n";
    std::cout << "Tolerance:         ±" << (TOLERANCE * 100.0) << "%\n";

    if (tflops >= PHASE_5A_BASELINE * (1.0f - TOLERANCE) &&
        tflops <= PHASE_5A_BASELINE * (1.0f + TOLERANCE))
    {
        std::cout << "\n✓ PASS: JIT kernel within tolerance of Phase 5A baseline\n";
    }
    else
    {
        std::cout << "\n✗ FAIL: JIT kernel outside tolerance range\n";
    }
    std::cout << "========================================\n\n";

    // Test assertion
    EXPECT_GE(tflops, PHASE_5A_BASELINE * (1.0f - TOLERANCE))
        << "JIT kernel significantly slower than baseline";
    EXPECT_LE(tflops, PHASE_5A_BASELINE * (1.0f + TOLERANCE))
        << "JIT kernel suspiciously faster than baseline (check correctness)";
}

TEST_F(Phase5ParityTest, SingleBuffer_vs_DoubleBuffer)
{
    // Compare single buffer (buffer_stages=1) vs double buffer (buffer_stages=2)
    // Hypothesis: Single buffer may achieve higher occupancy → better performance
    // Using valid 64x64x64 config with SUB_K=16 streaming

    Phase5GemmConfig config_template{
        .tile_m = 64,
        .tile_n = 64,
        .tile_k = 64,
        .sub_k = 16, // Streaming sub-tiles (Phase 5A feature)
        .mma_m = 2,
        .mma_n = 2,
        .buffer_stages = 1, // Will vary
        .threads_per_block = 128,
        .swizzle_b = 3,
        .swizzle_m = 3,
        .swizzle_s = 3};

    auto &jit = CudaGemmJITPhase5::instance();

    std::cout << "\n========================================\n";
    std::cout << "Single vs Double Buffer Comparison\n";
    std::cout << "========================================\n";
    std::cout << "Config: 64x64x64, SUB_K=16 (streaming), MMA 2x2, Swizzle 3,3,3\n\n";

    struct Result
    {
        int buffer_stages;
        size_t smem_bytes;
        int occupancy;
        float time_ms;
        float tflops;
    };

    std::vector<Result> results;

    for (int buffer_stages : {1, 2})
    {
        config_template.buffer_stages = buffer_stages;
        CudaGemmConfigPhase5 full_config(
            config_template.tile_m, config_template.tile_n, config_template.tile_k,
            config_template.sub_k,
            config_template.mma_m, config_template.mma_n,
            config_template.buffer_stages);

        CUfunction kernel = jit.getKernel(full_config);
        float avg_ms = benchmarkKernel(kernel, full_config, 100);

        double flops = 2.0 * M_ * N_ * K_;
        float tflops = flops / (avg_ms * 1e9);

        results.push_back({buffer_stages,
                           full_config.shared_memory_bytes(),
                           full_config.estimate_occupancy_blocks_per_sm(),
                           avg_ms,
                           tflops});
    }

    // Print results
    std::cout << std::left << std::setw(15) << "Buffer Stages"
              << std::setw(12) << "Smem (KB)"
              << std::setw(12) << "Occupancy"
              << std::setw(12) << "Time (ms)"
              << std::setw(12) << "TFLOPS" << "\n";
    std::cout << std::string(63, '-') << "\n";

    for (const auto &r : results)
    {
        std::cout << std::left << std::setw(15) << r.buffer_stages
                  << std::setw(12) << (r.smem_bytes / 1024)
                  << std::setw(12) << r.occupancy
                  << std::fixed << std::setprecision(4) << std::setw(12) << r.time_ms
                  << std::fixed << std::setprecision(2) << std::setw(12) << r.tflops << "\n";
    }

    float speedup = results[1].tflops / results[0].tflops;
    std::cout << "\nDouble buffer speedup: " << std::fixed << std::setprecision(2)
              << speedup << "x\n";
    std::cout << "========================================\n\n";

    // Verify both compile and run successfully
    EXPECT_GT(results[0].tflops, 0.0f) << "Single buffer should produce valid results";
    EXPECT_GT(results[1].tflops, 0.0f) << "Double buffer should produce valid results";
}

/**
 * @brief Sweep all 7 valid swizzle configurations for TILE_K=64
 *
 * For TILE_K=64 (log2(64)=6), valid swizzle configurations satisfy B+M=6 and S=B:
 * - Swizzle<0,6,0>: No swizzle (baseline)
 * - Swizzle<1,5,1>: 2-element XOR
 * - Swizzle<2,4,2>: 4-element XOR
 * - Swizzle<3,3,3>: 8-element XOR (current default)
 * - Swizzle<4,2,4>: 16-element XOR
 * - Swizzle<5,1,5>: 32-element XOR
 * - Swizzle<6,0,6>: 64-element XOR (maximum)
 */
TEST_F(Phase5ParityTest, SwizzleSweep_64x64x64)
{
    auto &jit = CudaGemmJITPhase5::instance();

    std::cout << "\n========================================\n";
    std::cout << "CuTe Swizzle Parameter Sweep\n";
    std::cout << "========================================\n";
    std::cout << "Base Config: 64x64x64, SUB_K=16, MMA 2x2, Double Buffer\n";
    std::cout << "Testing all 7 valid swizzle configurations for TILE_K=64\n\n";

    struct SwizzleResult
    {
        int swizzle_b;
        int swizzle_m;
        int swizzle_s;
        int vectorize_a; // Optimal vectorization width for this MBase
        float time_ms;
        float tflops;
        std::string description;
    };

    std::vector<SwizzleResult> results;

    // Generate all 7 valid swizzle configurations
    // For TILE_K=64 (log2(64)=6): B+M=6, S=B
    for (int M = 0; M <= 6; ++M)
    {
        int B = 6 - M;
        int S = B;

        // Optimal vectorization width = 2^MBase (consecutive elements before XOR)
        // MBase controls the "base" stride: higher MBase → more consecutive elements
        int vectorize_a = 1 << M; // 2^M elements

        // Cap at 8 (2×float4 is reasonable max for global loads)
        if (vectorize_a > 8)
            vectorize_a = 8;

        CudaGemmConfigPhase5 config(
            64, 64, 64, // tile_m, tile_n, tile_k
            16,         // sub_k (streaming)
            2, 2,       // mma_m, mma_n
            2,          // buffer_stages (double buffer)
            128,        // threads_per_block
            B, M, S,    // swizzle parameters
            vectorize_a // vectorization width
        );

        std::cout << "Compiling Swizzle<" << B << "," << M << "," << S << "> "
                  << "(vectorize_a=" << vectorize_a << ")...\n";

        CUfunction kernel = jit.getKernel(config);
        float avg_ms = benchmarkKernel(kernel, config, 100);

        double flops = 2.0 * M_ * N_ * K_;
        float tflops = flops / (avg_ms * 1e9);

        // Generate description
        std::string desc;
        if (B == 0)
            desc = "No swizzle (baseline)";
        else if (B == 6)
            desc = "Maximum XOR (64-elem)";
        else if (B == 3 && M == 3)
            desc = "Current default (8-elem XOR)";
        else
            desc = std::to_string(1 << B) + "-element XOR";

        results.push_back({B, M, S, vectorize_a, avg_ms, tflops, desc});

        std::cout << "  Result: " << std::fixed << std::setprecision(2)
                  << tflops << " TFLOPS (" << std::setprecision(4) << avg_ms << " ms)\n";
    }

    // Print summary table
    std::cout << "\n"
              << std::string(90, '=') << "\n";
    std::cout << "SWIZZLE SWEEP RESULTS\n";
    std::cout << std::string(90, '=') << "\n";
    std::cout << std::left << std::setw(20) << "Swizzle<B,M,S>"
              << std::setw(10) << "VecWidth"
              << std::setw(12) << "Time (ms)"
              << std::setw(12) << "TFLOPS"
              << std::setw(10) << "vs Best"
              << "Description\n";
    std::cout << std::string(90, '-') << "\n";

    // Find best result
    auto best_it = std::max_element(results.begin(), results.end(),
                                    [](const SwizzleResult &a, const SwizzleResult &b)
                                    {
                                        return a.tflops < b.tflops;
                                    });
    float best_tflops = best_it->tflops;

    for (const auto &r : results)
    {
        std::ostringstream swizzle_str;
        swizzle_str << "<" << r.swizzle_b << "," << r.swizzle_m << "," << r.swizzle_s << ">";

        float vs_best_pct = 100.0f * (r.tflops / best_tflops);

        std::cout << std::left << std::setw(20) << swizzle_str.str()
                  << std::setw(10) << r.vectorize_a
                  << std::fixed << std::setprecision(4) << std::setw(12) << r.time_ms
                  << std::fixed << std::setprecision(2) << std::setw(12) << r.tflops
                  << std::setprecision(1) << std::setw(10) << vs_best_pct << "%  "
                  << r.description << "\n";
    }

    std::cout << std::string(90, '=') << "\n";
    std::cout << "Best: Swizzle<" << best_it->swizzle_b << "," << best_it->swizzle_m
              << "," << best_it->swizzle_s << "> - " << best_it->tflops << " TFLOPS\n";
    std::cout << std::string(90, '=') << "\n\n";

    // Verify all configurations compile and run
    for (const auto &r : results)
    {
        EXPECT_GT(r.tflops, 0.0f) << "Swizzle<" << r.swizzle_b << "," << r.swizzle_m
                                  << "," << r.swizzle_s << "> should produce valid results";
    }
}
