/**
 * @file ROCmBenchmark.cpp
 * @brief ROCm/HIP GPU benchmark implementation
 *
 * Lightweight benchmarks designed to complete in < 500ms total
 * while providing accurate measurements for placement decisions.
 *
 * Uses HIP runtime API for AMD GPU benchmarking.
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include "ROCmBenchmark.h"
#include "utils/Logger.h"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <chrono>
#include <vector>

namespace llaminar2
{
    namespace
    {
/// Check HIP errors
#define HIP_CHECK(call)                                                                      \
    do                                                                                       \
    {                                                                                        \
        hipError_t err = (call);                                                             \
        if (err != hipSuccess)                                                               \
        {                                                                                    \
            LOG_ERROR("HIP error: " << hipGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__); \
            return 0.0;                                                                      \
        }                                                                                    \
    } while (0)

        /// High-resolution timing
        double getTimeMs()
        {
            using clock = std::chrono::high_resolution_clock;
            auto now = clock::now();
            auto duration = now.time_since_epoch();
            return std::chrono::duration<double, std::milli>(duration).count();
        }

        /// Simple FP32 FMA kernel for compute benchmark
        __global__ void computeKernelFP32(float *data, size_t n)
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

            float a = 1.0001f;
            float b = 0.9999f;
            float c = data[idx % n];

// Heavy FMA workload
#pragma unroll 8
            for (int i = 0; i < 256; ++i)
            {
                c = __fmaf_rn(a, b, c);
                a = __fmaf_rn(b, c, a);
                b = __fmaf_rn(c, a, b);
                c = __fmaf_rn(a, b, c);
            }

            if (idx < n)
            {
                data[idx] = c;
            }
        }

        /// Simple FP16 FMA kernel for compute benchmark
        __global__ void computeKernelFP16(__half *data, size_t n)
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

            __half a = __float2half(1.0001f);
            __half b = __float2half(0.9999f);
            __half c = data[idx % n];

// Heavy FMA workload
#pragma unroll 8
            for (int i = 0; i < 256; ++i)
            {
                c = __hfma(a, b, c);
                a = __hfma(b, c, a);
                b = __hfma(c, a, b);
                c = __hfma(a, b, c);
            }

            if (idx < n)
            {
                data[idx] = c;
            }
        }

        /// Memory copy kernel (for device memory bandwidth)
        __global__ void copyKernel(const float *__restrict__ src,
                                   float *__restrict__ dst,
                                   size_t n)
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            size_t stride = blockDim.x * gridDim.x;

            for (size_t i = idx; i < n; i += stride)
            {
                dst[i] = src[i];
            }
        }

    } // anonymous namespace

    ROCmBenchmark::ROCmBenchmark(int device_ordinal, const BenchmarkConfig &config)
        : device_ordinal_(device_ordinal), config_(config)
    {
    }

    ROCmBenchmark::~ROCmBenchmark() = default;

    double ROCmBenchmark::estimatedDurationMs() const
    {
        double iterations = config_.warmup_iterations + config_.iterations;
        return iterations * 20.0 * 7; // ~7 tests
    }

    double ROCmBenchmark::benchmarkDeviceMemory()
    {
        (void)hipSetDevice(device_ordinal_);

        size_t num_bytes = config_.memory_test_bytes;
        size_t num_elements = num_bytes / sizeof(float);

        float *d_src = nullptr;
        float *d_dst = nullptr;
        HIP_CHECK(hipMalloc(&d_src, num_bytes));
        HIP_CHECK(hipMalloc(&d_dst, num_bytes));

        HIP_CHECK(hipMemset(d_src, 0, num_bytes));
        HIP_CHECK(hipMemset(d_dst, 0, num_bytes));

        int block_size = 256;
        int num_blocks = static_cast<int>((num_elements + block_size - 1) / block_size);
        num_blocks = std::min(num_blocks, 65535);

        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));

        // Warmup
        for (int w = 0; w < config_.warmup_iterations; ++w)
        {
            hipLaunchKernelGGL(copyKernel, dim3(num_blocks), dim3(block_size), 0, 0,
                               d_src, d_dst, num_elements);
        }
        HIP_CHECK(hipDeviceSynchronize());

        // Timed runs
        HIP_CHECK(hipEventRecord(start));
        for (int iter = 0; iter < config_.iterations; ++iter)
        {
            hipLaunchKernelGGL(copyKernel, dim3(num_blocks), dim3(block_size), 0, 0,
                               d_src, d_dst, num_elements);
        }
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));

        float elapsed_ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));

        double bytes_transferred = 2.0 * num_bytes * config_.iterations;
        double gbps = (bytes_transferred / (elapsed_ms / 1000.0)) / 1e9;

        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));
        HIP_CHECK(hipFree(d_src));
        HIP_CHECK(hipFree(d_dst));

        return gbps;
    }

    double ROCmBenchmark::benchmarkH2DPinned()
    {
        (void)hipSetDevice(device_ordinal_);

        size_t num_bytes = config_.memory_test_bytes;

        void *h_pinned = nullptr;
        void *d_buffer = nullptr;
        HIP_CHECK(hipHostMalloc(&h_pinned, num_bytes));
        HIP_CHECK(hipMalloc(&d_buffer, num_bytes));

        memset(h_pinned, 0, num_bytes);

        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));

        // Warmup
        for (int w = 0; w < config_.warmup_iterations; ++w)
        {
            HIP_CHECK(hipMemcpy(d_buffer, h_pinned, num_bytes, hipMemcpyHostToDevice));
        }

        // Timed runs
        HIP_CHECK(hipEventRecord(start));
        for (int iter = 0; iter < config_.iterations; ++iter)
        {
            HIP_CHECK(hipMemcpy(d_buffer, h_pinned, num_bytes, hipMemcpyHostToDevice));
        }
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));

        float elapsed_ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));

        double gbps = (static_cast<double>(num_bytes) * config_.iterations / (elapsed_ms / 1000.0)) / 1e9;

        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));
        HIP_CHECK(hipHostFree(h_pinned));
        HIP_CHECK(hipFree(d_buffer));

        return gbps;
    }

    double ROCmBenchmark::benchmarkH2DPageable()
    {
        (void)hipSetDevice(device_ordinal_);

        size_t num_bytes = config_.memory_test_bytes;

        std::vector<char> h_pageable(num_bytes, 0);
        void *d_buffer = nullptr;
        HIP_CHECK(hipMalloc(&d_buffer, num_bytes));

        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));

        // Warmup
        for (int w = 0; w < config_.warmup_iterations; ++w)
        {
            HIP_CHECK(hipMemcpy(d_buffer, h_pageable.data(), num_bytes, hipMemcpyHostToDevice));
        }

        // Timed runs
        HIP_CHECK(hipEventRecord(start));
        for (int iter = 0; iter < config_.iterations; ++iter)
        {
            HIP_CHECK(hipMemcpy(d_buffer, h_pageable.data(), num_bytes, hipMemcpyHostToDevice));
        }
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));

        float elapsed_ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));

        double gbps = (static_cast<double>(num_bytes) * config_.iterations / (elapsed_ms / 1000.0)) / 1e9;

        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));
        HIP_CHECK(hipFree(d_buffer));

        return gbps;
    }

    double ROCmBenchmark::benchmarkD2HPinned()
    {
        (void)hipSetDevice(device_ordinal_);

        size_t num_bytes = config_.memory_test_bytes;

        void *h_pinned = nullptr;
        void *d_buffer = nullptr;
        HIP_CHECK(hipHostMalloc(&h_pinned, num_bytes));
        HIP_CHECK(hipMalloc(&d_buffer, num_bytes));

        HIP_CHECK(hipMemset(d_buffer, 1, num_bytes));

        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));

        // Warmup
        for (int w = 0; w < config_.warmup_iterations; ++w)
        {
            HIP_CHECK(hipMemcpy(h_pinned, d_buffer, num_bytes, hipMemcpyDeviceToHost));
        }

        // Timed runs
        HIP_CHECK(hipEventRecord(start));
        for (int iter = 0; iter < config_.iterations; ++iter)
        {
            HIP_CHECK(hipMemcpy(h_pinned, d_buffer, num_bytes, hipMemcpyDeviceToHost));
        }
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));

        float elapsed_ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));

        double gbps = (static_cast<double>(num_bytes) * config_.iterations / (elapsed_ms / 1000.0)) / 1e9;

        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));
        HIP_CHECK(hipHostFree(h_pinned));
        HIP_CHECK(hipFree(d_buffer));

        return gbps;
    }

    double ROCmBenchmark::benchmarkD2HPageable()
    {
        (void)hipSetDevice(device_ordinal_);

        size_t num_bytes = config_.memory_test_bytes;

        std::vector<char> h_pageable(num_bytes);
        void *d_buffer = nullptr;
        HIP_CHECK(hipMalloc(&d_buffer, num_bytes));

        HIP_CHECK(hipMemset(d_buffer, 1, num_bytes));

        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));

        // Warmup
        for (int w = 0; w < config_.warmup_iterations; ++w)
        {
            HIP_CHECK(hipMemcpy(h_pageable.data(), d_buffer, num_bytes, hipMemcpyDeviceToHost));
        }

        // Timed runs
        HIP_CHECK(hipEventRecord(start));
        for (int iter = 0; iter < config_.iterations; ++iter)
        {
            HIP_CHECK(hipMemcpy(h_pageable.data(), d_buffer, num_bytes, hipMemcpyDeviceToHost));
        }
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));

        float elapsed_ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));

        double gbps = (static_cast<double>(num_bytes) * config_.iterations / (elapsed_ms / 1000.0)) / 1e9;

        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));
        HIP_CHECK(hipFree(d_buffer));

        return gbps;
    }

    double ROCmBenchmark::benchmarkComputeFP32()
    {
        (void)hipSetDevice(device_ordinal_);

        size_t num_elements = 1024 * 1024; // 4MB
        float *d_buffer = nullptr;
        HIP_CHECK(hipMalloc(&d_buffer, num_elements * sizeof(float)));
        HIP_CHECK(hipMemset(d_buffer, 0, num_elements * sizeof(float)));

        int block_size = 256;
        int num_blocks = static_cast<int>((num_elements + block_size - 1) / block_size);

        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));

        // Warmup
        for (int w = 0; w < config_.warmup_iterations; ++w)
        {
            hipLaunchKernelGGL(computeKernelFP32, dim3(num_blocks), dim3(block_size), 0, 0,
                               d_buffer, num_elements);
        }
        HIP_CHECK(hipDeviceSynchronize());

        // Timed runs
        HIP_CHECK(hipEventRecord(start));
        for (int iter = 0; iter < config_.iterations; ++iter)
        {
            hipLaunchKernelGGL(computeKernelFP32, dim3(num_blocks), dim3(block_size), 0, 0,
                               d_buffer, num_elements);
        }
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));

        float elapsed_ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));

        double total_threads = static_cast<double>(num_blocks) * block_size;
        double flops_per_kernel = total_threads * 256 * 4 * 2;
        double total_flops = flops_per_kernel * config_.iterations;
        double gflops = (total_flops / (elapsed_ms / 1000.0)) / 1e9;

        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));
        HIP_CHECK(hipFree(d_buffer));

        return gflops;
    }

    double ROCmBenchmark::benchmarkComputeFP16()
    {
        (void)hipSetDevice(device_ordinal_);

        size_t num_elements = 1024 * 1024; // 2MB of half
        __half *d_buffer = nullptr;
        HIP_CHECK(hipMalloc(&d_buffer, num_elements * sizeof(__half)));
        HIP_CHECK(hipMemset(d_buffer, 0, num_elements * sizeof(__half)));

        int block_size = 256;
        int num_blocks = static_cast<int>((num_elements + block_size - 1) / block_size);

        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));

        // Warmup
        for (int w = 0; w < config_.warmup_iterations; ++w)
        {
            hipLaunchKernelGGL(computeKernelFP16, dim3(num_blocks), dim3(block_size), 0, 0,
                               d_buffer, num_elements);
        }
        HIP_CHECK(hipDeviceSynchronize());

        // Timed runs
        HIP_CHECK(hipEventRecord(start));
        for (int iter = 0; iter < config_.iterations; ++iter)
        {
            hipLaunchKernelGGL(computeKernelFP16, dim3(num_blocks), dim3(block_size), 0, 0,
                               d_buffer, num_elements);
        }
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));

        float elapsed_ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));

        double total_threads = static_cast<double>(num_blocks) * block_size;
        double flops_per_kernel = total_threads * 256 * 4 * 2;
        double total_flops = flops_per_kernel * config_.iterations;
        double gflops = (total_flops / (elapsed_ms / 1000.0)) / 1e9;

        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));
        HIP_CHECK(hipFree(d_buffer));

        return gflops;
    }

    double ROCmBenchmark::benchmarkPeerTransfer(int peer_ordinal)
    {
        if (peer_ordinal == device_ordinal_)
        {
            return 0.0;
        }

        // Check P2P availability (Infinity Fabric/xGMI or direct PCIe P2P)
        // This covers ALL direct GPU-to-GPU transfer methods without host staging
        int can_access = 0;
        hipError_t err = hipDeviceCanAccessPeer(&can_access, device_ordinal_, peer_ordinal);
        if (err != hipSuccess || !can_access)
        {
            LOG_DEBUG("Direct P2P not available between GPU " << device_ordinal_ << " and GPU " << peer_ordinal
                      << " (no xGMI/Infinity Fabric and no direct PCIe P2P)"
                      << " - transfers must stage through host memory");
            return 0.0;
        }

        // Enable P2P
        (void)hipSetDevice(device_ordinal_);
        err = hipDeviceEnablePeerAccess(peer_ordinal, 0);
        if (err != hipSuccess && err != hipErrorPeerAccessAlreadyEnabled)
        {
            LOG_DEBUG("Failed to enable P2P: " << hipGetErrorString(err));
            return 0.0;
        }

        size_t num_bytes = config_.memory_test_bytes;

        void *d_src = nullptr;
        void *d_dst = nullptr;

        (void)hipSetDevice(device_ordinal_);
        HIP_CHECK(hipMalloc(&d_src, num_bytes));
        HIP_CHECK(hipMemset(d_src, 1, num_bytes));

        (void)hipSetDevice(peer_ordinal);
        HIP_CHECK(hipMalloc(&d_dst, num_bytes));

        (void)hipSetDevice(device_ordinal_);

        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));

        // Warmup
        for (int w = 0; w < config_.warmup_iterations; ++w)
        {
            HIP_CHECK(hipMemcpyPeer(d_dst, peer_ordinal, d_src, device_ordinal_, num_bytes));
        }

        // Timed runs
        HIP_CHECK(hipEventRecord(start));
        for (int iter = 0; iter < config_.iterations; ++iter)
        {
            HIP_CHECK(hipMemcpyPeer(d_dst, peer_ordinal, d_src, device_ordinal_, num_bytes));
        }
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));

        float elapsed_ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));

        double gbps = (static_cast<double>(num_bytes) * config_.iterations / (elapsed_ms / 1000.0)) / 1e9;

        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));

        (void)hipSetDevice(peer_ordinal);
        (void)hipFree(d_dst);
        (void)hipSetDevice(device_ordinal_);
        (void)hipFree(d_src);

        return gbps;
    }

    DeviceBenchmarkResult ROCmBenchmark::run()
    {
        DeviceBenchmarkResult result;
        result.device = DeviceId::rocm(device_ordinal_);

        hipError_t err = hipSetDevice(device_ordinal_);
        if (err != hipSuccess)
        {
            LOG_ERROR("Failed to set ROCm device " << device_ordinal_ << ": " << hipGetErrorString(err));
            result.valid = false;
            return result;
        }

        hipDeviceProp_t props;
        (void)hipGetDeviceProperties(&props, device_ordinal_);
        LOG_DEBUG("Starting ROCm benchmark for GPU " << device_ordinal_ << ": " << props.name);

        double start_time = getTimeMs();

        // Device memory bandwidth
        LOG_DEBUG("Running device memory benchmark...");
        result.memory_copy_gbps = benchmarkDeviceMemory();

        // H2D transfers
        LOG_DEBUG("Running H2D pinned benchmark...");
        result.h2d_pinned_gbps = benchmarkH2DPinned();
        LOG_DEBUG("Running H2D pageable benchmark...");
        result.h2d_pageable_gbps = benchmarkH2DPageable();

        // D2H transfers
        LOG_DEBUG("Running D2H pinned benchmark...");
        result.d2h_pinned_gbps = benchmarkD2HPinned();
        LOG_DEBUG("Running D2H pageable benchmark...");
        result.d2h_pageable_gbps = benchmarkD2HPageable();

        // Compute
        LOG_DEBUG("Running FP32 compute benchmark...");
        result.compute_fp32_gflops = benchmarkComputeFP32();
        LOG_DEBUG("Running FP16 compute benchmark...");
        result.compute_fp16_gflops = benchmarkComputeFP16();

        // P2P transfers to all other GPUs
        int num_gpus = 0;
        (void)hipGetDeviceCount(&num_gpus);
        for (int peer = 0; peer < num_gpus; ++peer)
        {
            if (peer != device_ordinal_)
            {
                double p2p_gbps = benchmarkPeerTransfer(peer);
                if (p2p_gbps > 0.0)
                {
                    result.peer_transfer_gbps[DeviceId::rocm(peer)] = p2p_gbps;
                }
            }
        }

        result.benchmark_duration_ms = getTimeMs() - start_time;
        result.valid = true;

        LOG_INFO("ROCm GPU " << device_ordinal_ << " Benchmark completed in " << result.benchmark_duration_ms << " ms");
        LOG_INFO("  Device Memory:  " << result.memory_copy_gbps << " GB/s");
        LOG_INFO("  H2D (pinned):   " << result.h2d_pinned_gbps << " GB/s");
        LOG_INFO("  H2D (pageable): " << result.h2d_pageable_gbps << " GB/s");
        LOG_INFO("  D2H (pinned):   " << result.d2h_pinned_gbps << " GB/s");
        LOG_INFO("  D2H (pageable): " << result.d2h_pageable_gbps << " GB/s");
        LOG_INFO("  FP32 Compute:   " << result.compute_fp32_gflops << " GFLOPS");
        LOG_INFO("  FP16 Compute:   " << result.compute_fp16_gflops << " GFLOPS");
        for (const auto &[peer, gbps] : result.peer_transfer_gbps)
        {
            LOG_INFO("  P2P to GPU " << peer.ordinal << ": " << gbps << " GB/s");
        }

        return result;
    }

    std::vector<DeviceId> ROCmBenchmark::enumerateDevices()
    {
        std::vector<DeviceId> devices;

        int device_count = 0;
        hipError_t err = hipGetDeviceCount(&device_count);
        if (err == hipSuccess)
        {
            for (int i = 0; i < device_count; ++i)
            {
                devices.push_back(DeviceId::rocm(i));
            }
        }

        return devices;
    }

    double ROCmBenchmark::copyHostToDevice(int device_ordinal, const void *h_src, size_t num_bytes)
    {
        (void)hipSetDevice(device_ordinal);

        void *d_dst = nullptr;
        hipError_t err = hipMalloc(&d_dst, num_bytes);
        if (err != hipSuccess)
        {
            LOG_WARN("ROCm copyHostToDevice: hipMalloc failed for " << num_bytes << " bytes");
            return 0.0;
        }

        // Warm up
        (void)hipMemcpy(d_dst, h_src, num_bytes, hipMemcpyHostToDevice);
        (void)hipDeviceSynchronize();

        // Timed copy
        hipEvent_t start, stop;
        (void)hipEventCreate(&start);
        (void)hipEventCreate(&stop);

        (void)hipEventRecord(start);
        (void)hipMemcpy(d_dst, h_src, num_bytes, hipMemcpyHostToDevice);
        (void)hipEventRecord(stop);
        (void)hipEventSynchronize(stop);

        float ms = 0;
        (void)hipEventElapsedTime(&ms, start, stop);

        (void)hipEventDestroy(start);
        (void)hipEventDestroy(stop);
        (void)hipFree(d_dst);

        double seconds = ms / 1000.0;
        return (num_bytes / (1024.0 * 1024.0 * 1024.0)) / seconds;
    }

    double ROCmBenchmark::copyDeviceToHost(int device_ordinal, void *h_dst, size_t num_bytes)
    {
        (void)hipSetDevice(device_ordinal);

        void *d_src = nullptr;
        hipError_t err = hipMalloc(&d_src, num_bytes);
        if (err != hipSuccess)
        {
            LOG_WARN("ROCm copyDeviceToHost: hipMalloc failed for " << num_bytes << " bytes");
            return 0.0;
        }

        // Initialize device memory with pattern
        (void)hipMemset(d_src, 0xAB, num_bytes);
        (void)hipDeviceSynchronize();

        // Warm up
        (void)hipMemcpy(h_dst, d_src, num_bytes, hipMemcpyDeviceToHost);
        (void)hipDeviceSynchronize();

        // Timed copy
        hipEvent_t start, stop;
        (void)hipEventCreate(&start);
        (void)hipEventCreate(&stop);

        (void)hipEventRecord(start);
        (void)hipMemcpy(h_dst, d_src, num_bytes, hipMemcpyDeviceToHost);
        (void)hipEventRecord(stop);
        (void)hipEventSynchronize(stop);

        float ms = 0;
        (void)hipEventElapsedTime(&ms, start, stop);

        (void)hipEventDestroy(start);
        (void)hipEventDestroy(stop);
        (void)hipFree(d_src);

        double seconds = ms / 1000.0;
        return (num_bytes / (1024.0 * 1024.0 * 1024.0)) / seconds;
    }

} // namespace llaminar2
