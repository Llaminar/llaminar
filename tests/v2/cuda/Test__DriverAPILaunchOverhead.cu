/**
 * @file Test__DriverAPILaunchOverhead.cpp
 * @brief Test harness to measure Driver API vs Runtime API launch overhead
 *
 * This test explicitly measures the overhead of launching kernels with:
 * 1. Driver API (cuLaunchKernel) - Used by JIT kernels
 * 2. Runtime API (<<<>>>) - Used by pre-compiled kernels
 *
 * Goal: Confirm that Driver API has ~900 μs overhead per launch
 *
 * @author David Sanftenberg
 * @date November 4, 2025
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <nvrtc.h>
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

// Simple kernel that does minimal work (just to have something to launch)
extern "C" __global__ void dummy_kernel(float *output, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        output[idx] = static_cast<float>(idx);
    }
}

// Wrapper for Runtime API launch
void launchWithRuntimeAPI(float *d_output, int N)
{
    dim3 block(256);
    dim3 grid((N + 255) / 256);
    dummy_kernel<<<grid, block>>>(d_output, N);
}

class DriverAPILaunchOverheadTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Initialize CUDA
        cudaSetDevice(0);
        cudaDeviceSynchronize();

        // Initialize Driver API
        CU_CHECK(cuInit(0));

        // Get primary context for Driver API
        CUdevice device;
        CU_CHECK(cuDeviceGet(&device, 0));
        CU_CHECK(cuDevicePrimaryCtxRetain(&ctx_, device));
        CU_CHECK(cuCtxSetCurrent(ctx_));

        // Allocate device memory
        N_ = 1024 * 1024; // 1M elements
        CUDA_CHECK(cudaMalloc(&d_output_, N_ * sizeof(float)));

        // For Driver API, we need to get a CUfunction from the compiled kernel
        // We'll use cudaGetFuncBySymbol or extract from the current module
        // Simpler approach: Get the function address from the compiled binary

        void *kernel_ptr = (void *)dummy_kernel;

        // Actually, we need to compile some PTX or get a module
        // Let's use NVRTC to compile a simple kernel
        const char *kernel_source = R"(
extern "C" __global__ void dummy_kernel_driver(float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = (float)idx;
    }
}
)";

        // Compile with NVRTC
        nvrtcProgram prog;
        nvrtcResult res = nvrtcCreateProgram(&prog, kernel_source, "dummy.cu", 0, nullptr, nullptr);
        if (res != NVRTC_SUCCESS)
        {
            throw std::runtime_error(std::string("nvrtcCreateProgram failed: ") + nvrtcGetErrorString(res));
        }

        const char *opts[] = {
            "--gpu-architecture=compute_86",
            "-std=c++17"};

        res = nvrtcCompileProgram(prog, 2, opts);
        if (res != NVRTC_SUCCESS)
        {
            size_t log_size;
            nvrtcGetProgramLogSize(prog, &log_size);
            std::vector<char> log(log_size);
            nvrtcGetProgramLog(prog, log.data());
            std::cerr << "NVRTC compilation log:\n"
                      << log.data() << std::endl;
            nvrtcDestroyProgram(&prog);
            throw std::runtime_error("NVRTC compilation failed");
        }

        size_t ptx_size;
        res = nvrtcGetPTXSize(prog, &ptx_size);
        if (res != NVRTC_SUCCESS)
        {
            nvrtcDestroyProgram(&prog);
            throw std::runtime_error("nvrtcGetPTXSize failed");
        }

        std::vector<char> ptx(ptx_size);
        res = nvrtcGetPTX(prog, ptx.data());
        if (res != NVRTC_SUCCESS)
        {
            nvrtcDestroyProgram(&prog);
            throw std::runtime_error("nvrtcGetPTX failed");
        }

        nvrtcDestroyProgram(&prog);

        // Load module from PTX
        CU_CHECK(cuModuleLoadData(&driver_module_, ptx.data()));
        CU_CHECK(cuModuleGetFunction(&driver_kernel_, driver_module_, "dummy_kernel_driver"));

        std::cout << "\n[Setup] Initialized Driver and Runtime API contexts\n";
        std::cout << "[Setup] Allocated " << N_ << " elements (" << (N_ * sizeof(float) / 1024.0 / 1024.0) << " MB)\n\n";
    }

    void TearDown() override
    {
        cudaFree(d_output_);
        if (driver_module_)
        {
            cuModuleUnload(driver_module_);
        }
        if (ctx_)
        {
            cuCtxDestroy(ctx_);
        }
    }

    int N_;
    float *d_output_ = nullptr;
    CUcontext ctx_ = nullptr;
    CUmodule driver_module_ = nullptr;
    CUfunction driver_kernel_ = nullptr;
};

TEST_F(DriverAPILaunchOverheadTest, CompareDriverVsRuntimeAPI)
{
    std::cout << "========================================\n";
    std::cout << "Driver API vs Runtime API Launch Overhead\n";
    std::cout << "========================================\n\n";

    const int warmup_iters = 50;
    const int bench_iters = 100;

    dim3 grid((N_ + 255) / 256);
    dim3 block(256);

    // ===== Test 1: Runtime API (<<<>>>) =====
    std::cout << "Test 1: Runtime API (<<<>>>)\n";
    std::cout << "------------------------------------\n";

    // Warmup
    for (int i = 0; i < warmup_iters; i++)
    {
        launchWithRuntimeAPI(d_output_, N_);
    }
    cudaDeviceSynchronize();

    // Benchmark
    cudaEvent_t rt_start, rt_stop;
    cudaEventCreate(&rt_start);
    cudaEventCreate(&rt_stop);

    cudaEventRecord(rt_start);
    for (int i = 0; i < bench_iters; i++)
    {
        launchWithRuntimeAPI(d_output_, N_);
    }
    cudaEventRecord(rt_stop);
    cudaEventSynchronize(rt_stop);

    float rt_time_ms;
    cudaEventElapsedTime(&rt_time_ms, rt_start, rt_stop);
    float rt_avg_us = (rt_time_ms * 1000.0f) / bench_iters;

    cudaEventDestroy(rt_start);
    cudaEventDestroy(rt_stop);

    std::cout << "  Total time:     " << std::fixed << std::setprecision(3) << rt_time_ms << " ms\n";
    std::cout << "  Per launch:     " << std::fixed << std::setprecision(3) << rt_avg_us << " μs\n";
    std::cout << "  Iterations:     " << bench_iters << "\n\n";

    // ===== Test 2: Driver API (cuLaunchKernel) =====
    std::cout << "Test 2: Driver API (cuLaunchKernel)\n";
    std::cout << "------------------------------------\n";

    void *args[] = {&d_output_, &N_};

    // Warmup
    for (int i = 0; i < warmup_iters; i++)
    {
        CU_CHECK(cuLaunchKernel(driver_kernel_,
                                grid.x, grid.y, grid.z,
                                block.x, block.y, block.z,
                                0, nullptr, args, nullptr));
    }
    cudaDeviceSynchronize();

    // Benchmark
    cudaEvent_t dr_start, dr_stop;
    cudaEventCreate(&dr_start);
    cudaEventCreate(&dr_stop);

    cudaEventRecord(dr_start);
    for (int i = 0; i < bench_iters; i++)
    {
        CU_CHECK(cuLaunchKernel(driver_kernel_,
                                grid.x, grid.y, grid.z,
                                block.x, block.y, block.z,
                                0, nullptr, args, nullptr));
    }
    cudaEventRecord(dr_stop);
    cudaEventSynchronize(dr_stop);

    float dr_time_ms;
    cudaEventElapsedTime(&dr_time_ms, dr_start, dr_stop);
    float dr_avg_us = (dr_time_ms * 1000.0f) / bench_iters;

    cudaEventDestroy(dr_start);
    cudaEventDestroy(dr_stop);

    std::cout << "  Total time:     " << std::fixed << std::setprecision(3) << dr_time_ms << " ms\n";
    std::cout << "  Per launch:     " << std::fixed << std::setprecision(3) << dr_avg_us << " μs\n";
    std::cout << "  Iterations:     " << bench_iters << "\n\n";

    // ===== Test 3: Overhead Breakdown =====
    std::cout << "Test 3: Overhead Analysis\n";
    std::cout << "------------------------------------\n";

    // Measure actual kernel execution time (single launch with NCU would show this)
    // For now, estimate by launching once and measuring
    cudaEvent_t kernel_start, kernel_stop;
    cudaEventCreate(&kernel_start);
    cudaEventCreate(&kernel_stop);

    cudaEventRecord(kernel_start);
    launchWithRuntimeAPI(d_output_, N_);
    cudaEventRecord(kernel_stop);
    cudaEventSynchronize(kernel_stop);

    float kernel_time_us;
    cudaEventElapsedTime(&kernel_time_us, kernel_start, kernel_stop);
    kernel_time_us *= 1000.0f; // ms to μs

    cudaEventDestroy(kernel_start);
    cudaEventDestroy(kernel_stop);

    std::cout << "  Kernel execution:        " << std::fixed << std::setprecision(3) << kernel_time_us << " μs\n";
    std::cout << "  Runtime API overhead:    " << std::fixed << std::setprecision(3) << (rt_avg_us - kernel_time_us) << " μs\n";
    std::cout << "  Driver API overhead:     " << std::fixed << std::setprecision(3) << (dr_avg_us - kernel_time_us) << " μs\n";
    std::cout << "  **Overhead difference:   " << std::fixed << std::setprecision(3) << (dr_avg_us - rt_avg_us) << " μs**\n\n";

    // ===== Summary =====
    std::cout << "========================================\n";
    std::cout << "Summary\n";
    std::cout << "========================================\n";
    std::cout << "  Runtime API:   " << std::fixed << std::setprecision(3) << rt_avg_us << " μs/launch\n";
    std::cout << "  Driver API:    " << std::fixed << std::setprecision(3) << dr_avg_us << " μs/launch\n";
    std::cout << "  Slowdown:      " << std::fixed << std::setprecision(1) << (dr_avg_us / rt_avg_us) << "x\n";
    std::cout << "  Extra cost:    " << std::fixed << std::setprecision(3) << (dr_avg_us - rt_avg_us) << " μs\n";
    std::cout << "========================================\n\n";

    // Assertions
    EXPECT_GT(dr_avg_us, rt_avg_us) << "Driver API should be slower than Runtime API";

    // Check if overhead matches our hypothesis (~900 μs)
    float overhead_us = dr_avg_us - rt_avg_us;
    if (overhead_us > 500.0f)
    {
        std::cout << "✓ CONFIRMED: Driver API has significant overhead (" << overhead_us << " μs)\n";
        std::cout << "  This explains the JIT kernel slowdown!\n\n";
    }
    else
    {
        std::cout << "✗ UNEXPECTED: Driver API overhead is low (" << overhead_us << " μs)\n";
        std::cout << "  The JIT slowdown may have a different root cause.\n\n";
    }
}

TEST_F(DriverAPILaunchOverheadTest, MeasureLaunchOnlyOverhead)
{
    std::cout << "\n========================================\n";
    std::cout << "Launch-Only Overhead (Empty Loop)\n";
    std::cout << "========================================\n\n";

    const int iters = 1000;
    void *args[] = {&d_output_, &N_};
    dim3 grid(1);  // Minimal grid
    dim3 block(1); // Minimal block

    // Runtime API
    auto rt_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++)
    {
        launchWithRuntimeAPI(d_output_, N_);
    }
    cudaDeviceSynchronize();
    auto rt_end = std::chrono::high_resolution_clock::now();
    auto rt_us = std::chrono::duration_cast<std::chrono::microseconds>(rt_end - rt_start).count();

    // Driver API
    auto dr_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++)
    {
        cuLaunchKernel(driver_kernel_, grid.x, grid.y, grid.z,
                       block.x, block.y, block.z,
                       0, nullptr, args, nullptr);
    }
    cudaDeviceSynchronize();
    auto dr_end = std::chrono::high_resolution_clock::now();
    auto dr_us = std::chrono::duration_cast<std::chrono::microseconds>(dr_end - dr_start).count();

    std::cout << "  Runtime API: " << (rt_us / iters) << " μs/launch (avg over " << iters << " launches)\n";
    std::cout << "  Driver API:  " << (dr_us / iters) << " μs/launch (avg over " << iters << " launches)\n";
    std::cout << "  Difference:  " << ((dr_us - rt_us) / iters) << " μs/launch\n\n";
}

TEST_F(DriverAPILaunchOverheadTest, SingleLaunchLatency)
{
    std::cout << "\n========================================\n";
    std::cout << "Single Launch Latency (Cold Start)\n";
    std::cout << "========================================\n\n";

    void *args[] = {&d_output_, &N_};
    dim3 grid((N_ + 255) / 256);
    dim3 block(256);

    // Measure single Runtime API launch
    auto rt_start = std::chrono::high_resolution_clock::now();
    launchWithRuntimeAPI(d_output_, N_);
    cudaDeviceSynchronize();
    auto rt_end = std::chrono::high_resolution_clock::now();
    auto rt_us = std::chrono::duration_cast<std::chrono::microseconds>(rt_end - rt_start).count();

    // Measure single Driver API launch
    auto dr_start = std::chrono::high_resolution_clock::now();
    cuLaunchKernel(driver_kernel_, grid.x, grid.y, grid.z,
                   block.x, block.y, block.z,
                   0, nullptr, args, nullptr);
    cudaDeviceSynchronize();
    auto dr_end = std::chrono::high_resolution_clock::now();
    auto dr_us = std::chrono::duration_cast<std::chrono::microseconds>(dr_end - dr_start).count();

    std::cout << "  Runtime API: " << rt_us << " μs (single cold launch + sync)\n";
    std::cout << "  Driver API:  " << dr_us << " μs (single cold launch + sync)\n";
    std::cout << "  Difference:  " << (dr_us - rt_us) << " μs\n\n";
}
