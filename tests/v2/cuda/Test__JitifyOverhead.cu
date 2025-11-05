/**
 * @file Test__JitifyOverhead.cu
 * @brief Test harness to measure Jitify-specific overhead in JIT workflow
 *
 * This test measures overhead at each stage:
 * 1. Pure Driver API launch (baseline)
 * 2. Raw NVRTC compilation + Driver API launch
 * 3. Jitify compilation + Driver API launch
 * 4. Cached Jitify (memory hit) + Driver API launch
 *
 * Goal: Identify where the 900 μs overhead is coming from
 *
 * @author David Sanftenberg
 * @date November 4, 2025
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <nvrtc.h>
#include "jitify.hpp"
#include <chrono>
#include <iostream>
#include <iomanip>
#include <vector>

// Error checking macros
#define CU_CHECK(call)                                                                    \
    do                                                                                    \
    {                                                                                     \
        CUresult result = call;                                                           \
        if (result != CUDA_SUCCESS)                                                       \
        {                                                                                 \
            const char *error_str;                                                        \
            cuGetErrorString(result, &error_str);                                         \
            throw std::runtime_error(std::string("CUDA Driver API error: ") + error_str); \
        }                                                                                 \
    } while (0)

#define CUDA_CHECK(call)                                                                                 \
    do                                                                                                   \
    {                                                                                                    \
        cudaError_t err = call;                                                                          \
        if (err != cudaSuccess)                                                                          \
        {                                                                                                \
            throw std::runtime_error(std::string("CUDA Runtime API error: ") + cudaGetErrorString(err)); \
        }                                                                                                \
    } while (0)

#define NVRTC_CHECK(call)                                                                      \
    do                                                                                         \
    {                                                                                          \
        nvrtcResult res = call;                                                                \
        if (res != NVRTC_SUCCESS)                                                              \
        {                                                                                      \
            throw std::runtime_error(std::string("NVRTC error: ") + nvrtcGetErrorString(res)); \
        }                                                                                      \
    } while (0)

// Simple kernel source
const char *KERNEL_SOURCE = R"(
extern "C" __global__ void test_kernel(float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = (float)idx * 2.0f;
    }
}
)";

class JitifyOverheadTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Initialize CUDA
        cudaSetDevice(0);
        cudaDeviceSynchronize();

        CU_CHECK(cuInit(0));

        CUdevice device;
        CU_CHECK(cuDeviceGet(&device, 0));
        CU_CHECK(cuDevicePrimaryCtxRetain(&ctx_, device));
        CU_CHECK(cuCtxSetCurrent(ctx_));

        N_ = 1024 * 1024;
        CUDA_CHECK(cudaMalloc(&d_output_, N_ * sizeof(float)));

        std::cout << "\n[Setup] Test environment initialized\n";
        std::cout << "[Setup] Matrix size: " << N_ << " elements\n\n";
    }

    void TearDown() override
    {
        cudaFree(d_output_);
        if (nvrtc_module_)
            cuModuleUnload(nvrtc_module_);
        if (jitify_module_)
            cuModuleUnload(jitify_module_);
        if (ctx_)
            cuCtxDestroy(ctx_);
    }

    // Compile with raw NVRTC
    CUfunction compileWithNVRTC()
    {
        nvrtcProgram prog;
        NVRTC_CHECK(nvrtcCreateProgram(&prog, KERNEL_SOURCE, "test.cu", 0, nullptr, nullptr));

        const char *opts[] = {
            "--gpu-architecture=compute_86",
            "-std=c++17"};

        auto compile_start = std::chrono::high_resolution_clock::now();
        NVRTC_CHECK(nvrtcCompileProgram(prog, 2, opts));
        auto compile_end = std::chrono::high_resolution_clock::now();
        nvrtc_compile_time_ms_ = std::chrono::duration<double, std::milli>(compile_end - compile_start).count();

        size_t ptx_size;
        NVRTC_CHECK(nvrtcGetPTXSize(prog, &ptx_size));
        std::vector<char> ptx(ptx_size);
        NVRTC_CHECK(nvrtcGetPTX(prog, ptx.data()));
        nvrtcDestroyProgram(&prog);

        CU_CHECK(cuModuleLoadData(&nvrtc_module_, ptx.data()));

        CUfunction func;
        CU_CHECK(cuModuleGetFunction(&func, nvrtc_module_, "test_kernel"));

        return func;
    }

    // Compile with Jitify
    CUfunction compileWithJitify()
    {
        static jitify::JitCache cache;

        std::vector<std::string> opts = {
            "--gpu-architecture=compute_86",
            "-std=c++17"};

        auto compile_start = std::chrono::high_resolution_clock::now();
        jitify::Program program = cache.program(KERNEL_SOURCE, {}, opts);
        auto kernel_inst = program.kernel("test_kernel").instantiate();
        auto compile_end = std::chrono::high_resolution_clock::now();
        jitify_compile_time_ms_ = std::chrono::duration<double, std::milli>(compile_end - compile_start).count();

        // Get PTX and load
        const std::string &ptx = kernel_inst.ptx();
        CU_CHECK(cuModuleLoadData(&jitify_module_, ptx.data()));

        const std::string &mangled_name = kernel_inst.mangled_name();

        CUfunction func;
        CU_CHECK(cuModuleGetFunction(&func, jitify_module_, mangled_name.c_str()));

        return func;
    }

    // Measure kernel launch time
    double measureLaunchTime(CUfunction kernel, int iterations)
    {
        void *args[] = {&d_output_, &N_};
        dim3 grid((N_ + 255) / 256);
        dim3 block(256);

        // Warmup
        for (int i = 0; i < 10; i++)
        {
            CU_CHECK(cuLaunchKernel(kernel, grid.x, grid.y, grid.z,
                                    block.x, block.y, block.z,
                                    0, nullptr, args, nullptr));
        }
        cudaDeviceSynchronize();

        // Benchmark
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        for (int i = 0; i < iterations; i++)
        {
            CU_CHECK(cuLaunchKernel(kernel, grid.x, grid.y, grid.z,
                                    block.x, block.y, block.z,
                                    0, nullptr, args, nullptr));
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float time_ms;
        cudaEventElapsedTime(&time_ms, start, stop);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        return (time_ms * 1000.0) / iterations; // Return μs per launch
    }

    int N_;
    float *d_output_ = nullptr;
    CUcontext ctx_ = nullptr;
    CUmodule nvrtc_module_ = nullptr;
    CUmodule jitify_module_ = nullptr;
    double nvrtc_compile_time_ms_ = 0.0;
    double jitify_compile_time_ms_ = 0.0;
};

TEST_F(JitifyOverheadTest, CompareNVRTCvsJitify)
{
    std::cout << "========================================\n";
    std::cout << "NVRTC vs Jitify Compilation & Launch\n";
    std::cout << "========================================\n\n";

    const int bench_iters = 100;

    // Test 1: Raw NVRTC
    std::cout << "Test 1: Raw NVRTC\n";
    std::cout << "------------------------------------\n";

    auto nvrtc_start = std::chrono::high_resolution_clock::now();
    CUfunction nvrtc_kernel = compileWithNVRTC();
    auto nvrtc_end = std::chrono::high_resolution_clock::now();
    double nvrtc_total_ms = std::chrono::duration<double, std::milli>(nvrtc_end - nvrtc_start).count();

    double nvrtc_launch_us = measureLaunchTime(nvrtc_kernel, bench_iters);

    std::cout << "  Compilation time:   " << std::fixed << std::setprecision(2) << nvrtc_compile_time_ms_ << " ms\n";
    std::cout << "  Module load time:   " << std::fixed << std::setprecision(2) << (nvrtc_total_ms - nvrtc_compile_time_ms_) << " ms\n";
    std::cout << "  Total setup time:   " << std::fixed << std::setprecision(2) << nvrtc_total_ms << " ms\n";
    std::cout << "  Launch time:        " << std::fixed << std::setprecision(3) << nvrtc_launch_us << " μs\n\n";

    // Test 2: Jitify
    std::cout << "Test 2: Jitify\n";
    std::cout << "------------------------------------\n";

    auto jitify_start = std::chrono::high_resolution_clock::now();
    CUfunction jitify_kernel = compileWithJitify();
    auto jitify_end = std::chrono::high_resolution_clock::now();
    double jitify_total_ms = std::chrono::duration<double, std::milli>(jitify_end - jitify_start).count();

    double jitify_launch_us = measureLaunchTime(jitify_kernel, bench_iters);

    std::cout << "  Jitify compile:     " << std::fixed << std::setprecision(2) << jitify_compile_time_ms_ << " ms\n";
    std::cout << "  Module load time:   " << std::fixed << std::setprecision(2) << (jitify_total_ms - jitify_compile_time_ms_) << " ms\n";
    std::cout << "  Total setup time:   " << std::fixed << std::setprecision(2) << jitify_total_ms << " ms\n";
    std::cout << "  Launch time:        " << std::fixed << std::setprecision(3) << jitify_launch_us << " μs\n\n";

    // Comparison
    std::cout << "========================================\n";
    std::cout << "Overhead Analysis\n";
    std::cout << "========================================\n";
    std::cout << "  Setup overhead:     " << std::fixed << std::setprecision(2)
              << (jitify_total_ms - nvrtc_total_ms) << " ms\n";
    std::cout << "  Launch overhead:    " << std::fixed << std::setprecision(3)
              << (jitify_launch_us - nvrtc_launch_us) << " μs\n";
    std::cout << "  Launch slowdown:    " << std::fixed << std::setprecision(2)
              << (jitify_launch_us / nvrtc_launch_us) << "x\n";
    std::cout << "========================================\n\n";

    if (jitify_launch_us > nvrtc_launch_us + 100.0)
    {
        std::cout << "✗ WARNING: Jitify launches are significantly slower!\n";
        std::cout << "  Extra overhead: " << (jitify_launch_us - nvrtc_launch_us) << " μs\n\n";
    }
    else
    {
        std::cout << "✓ Jitify launch overhead is minimal\n\n";
    }
}

TEST_F(JitifyOverheadTest, MeasureCachedJitifyLaunch)
{
    std::cout << "========================================\n";
    std::cout << "Jitify Cached Launch Performance\n";
    std::cout << "========================================\n\n";

    // Compile once
    std::cout << "Step 1: Initial compilation\n";
    auto compile_start = std::chrono::high_resolution_clock::now();
    CUfunction kernel = compileWithJitify();
    auto compile_end = std::chrono::high_resolution_clock::now();
    double compile_ms = std::chrono::duration<double, std::milli>(compile_end - compile_start).count();
    std::cout << "  Compilation: " << std::fixed << std::setprecision(2) << compile_ms << " ms\n\n";

    // Measure launch overhead (cached kernel)
    std::cout << "Step 2: Cached kernel launch performance\n";
    const int iters = 1000;

    void *args[] = {&d_output_, &N_};
    dim3 grid((N_ + 255) / 256);
    dim3 block(256);

    // Warmup
    for (int i = 0; i < 50; i++)
    {
        CU_CHECK(cuLaunchKernel(kernel, grid.x, grid.y, grid.z,
                                block.x, block.y, block.z,
                                0, nullptr, args, nullptr));
    }
    cudaDeviceSynchronize();

    // Time just the launches (no sync between)
    auto launch_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++)
    {
        CU_CHECK(cuLaunchKernel(kernel, grid.x, grid.y, grid.z,
                                block.x, block.y, block.z,
                                0, nullptr, args, nullptr));
    }
    cudaDeviceSynchronize();
    auto launch_end = std::chrono::high_resolution_clock::now();

    auto total_us = std::chrono::duration_cast<std::chrono::microseconds>(launch_end - launch_start).count();
    double per_launch_us = total_us / (double)iters;

    std::cout << "  Iterations:       " << iters << "\n";
    std::cout << "  Total time:       " << std::fixed << std::setprecision(2) << (total_us / 1000.0) << " ms\n";
    std::cout << "  Per launch:       " << std::fixed << std::setprecision(3) << per_launch_us << " μs\n";
    std::cout << "  Throughput:       " << std::fixed << std::setprecision(0) << (1000000.0 / per_launch_us) << " launches/sec\n\n";

    // Also measure with CUDA events (includes some kernel execution)
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < 100; i++)
    {
        CU_CHECK(cuLaunchKernel(kernel, grid.x, grid.y, grid.z,
                                block.x, block.y, block.z,
                                0, nullptr, args, nullptr));
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float event_time_ms;
    cudaEventElapsedTime(&event_time_ms, start, stop);
    double event_per_launch_us = (event_time_ms * 1000.0) / 100.0;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    std::cout << "With CUDA Events (includes kernel execution):\n";
    std::cout << "  Per launch+exec:  " << std::fixed << std::setprecision(3) << event_per_launch_us << " μs\n\n";
}

TEST_F(JitifyOverheadTest, MeasureParameterSetupOverhead)
{
    std::cout << "========================================\n";
    std::cout << "Parameter Setup Overhead\n";
    std::cout << "========================================\n\n";

    CUfunction kernel = compileWithNVRTC();
    dim3 grid((N_ + 255) / 256);
    dim3 block(256);

    const int iters = 10000;

    // Test 1: Just parameter array creation
    auto param_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++)
    {
        void *args[] = {&d_output_, &N_};
        // Don't optimize away
        volatile void *ptr = args;
        (void)ptr;
    }
    auto param_end = std::chrono::high_resolution_clock::now();
    auto param_us = std::chrono::duration_cast<std::chrono::microseconds>(param_end - param_start).count();

    std::cout << "  Parameter array creation: " << std::fixed << std::setprecision(3)
              << (param_us / (double)iters) << " μs per iteration\n";

    // Test 2: Parameter setup + launch (no sync)
    void *args[] = {&d_output_, &N_};
    auto launch_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++)
    {
        CU_CHECK(cuLaunchKernel(kernel, grid.x, grid.y, grid.z,
                                block.x, block.y, block.z,
                                0, nullptr, args, nullptr));
    }
    cudaDeviceSynchronize();
    auto launch_end = std::chrono::high_resolution_clock::now();
    auto launch_us = std::chrono::duration_cast<std::chrono::microseconds>(launch_end - launch_start).count();

    std::cout << "  Launch (batched):         " << std::fixed << std::setprecision(3)
              << (launch_us / (double)iters) << " μs per launch\n\n";
}

TEST_F(JitifyOverheadTest, ProfileModuleLookup)
{
    std::cout << "========================================\n";
    std::cout << "Module Function Lookup Overhead\n";
    std::cout << "========================================\n\n";

    // Compile module
    nvrtcProgram prog;
    NVRTC_CHECK(nvrtcCreateProgram(&prog, KERNEL_SOURCE, "test.cu", 0, nullptr, nullptr));

    const char *opts[] = {"--gpu-architecture=compute_86", "-std=c++17"};
    NVRTC_CHECK(nvrtcCompileProgram(prog, 2, opts));

    size_t ptx_size;
    NVRTC_CHECK(nvrtcGetPTXSize(prog, &ptx_size));
    std::vector<char> ptx(ptx_size);
    NVRTC_CHECK(nvrtcGetPTX(prog, ptx.data()));
    nvrtcDestroyProgram(&prog);

    // Load module
    CUmodule module;
    auto load_start = std::chrono::high_resolution_clock::now();
    CU_CHECK(cuModuleLoadData(&module, ptx.data()));
    auto load_end = std::chrono::high_resolution_clock::now();
    auto load_ms = std::chrono::duration<double, std::milli>(load_end - load_start).count();

    std::cout << "  Module load time:         " << std::fixed << std::setprecision(2) << load_ms << " ms\n";

    // Measure function lookup
    const int iters = 10000;
    auto lookup_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++)
    {
        CUfunction func;
        CU_CHECK(cuModuleGetFunction(&func, module, "test_kernel"));
    }
    auto lookup_end = std::chrono::high_resolution_clock::now();
    auto lookup_us = std::chrono::duration_cast<std::chrono::microseconds>(lookup_end - lookup_start).count();

    std::cout << "  Function lookup:          " << std::fixed << std::setprecision(3)
              << (lookup_us / (double)iters) << " μs per lookup\n";
    std::cout << "  (Averaged over " << iters << " lookups)\n\n";

    cuModuleUnload(module);
}
