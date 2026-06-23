/**
 * @file CUDABenchmark.cu
 * @brief CUDA GPU benchmark implementation
 *
 * Lightweight benchmarks designed to complete in < 500ms total
 * while providing accurate measurements for placement decisions.
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include "CUDABenchmark.h"
#include "utils/Logger.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <chrono>
#include <vector>

namespace llaminar2
{
    namespace
    {
        /// Check CUDA errors
#define CUDA_CHECK(call)                                           \
    do                                                             \
    {                                                              \
        cudaError_t err = (call);                                  \
        if (err != cudaSuccess)                                    \
        {                                                          \
            LOG_ERROR("CUDA error: " << cudaGetErrorString(err)    \
                                     << " at " << __FILE__ << ":" << __LINE__); \
            return 0.0;                                            \
        }                                                          \
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
        __global__ void computeKernelFP16(half *data, size_t n)
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

            half a = __float2half(1.0001f);
            half b = __float2half(0.9999f);
            half c = data[idx % n];

            // Heavy FMA workload using half2 for efficiency
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

    CUDABenchmark::CUDABenchmark(int device_ordinal, const BenchmarkConfig &config)
        : device_ordinal_(device_ordinal), config_(config)
    {
    }

    CUDABenchmark::~CUDABenchmark() = default;

    double CUDABenchmark::estimatedDurationMs() const
    {
        // Estimate based on config and number of tests
        double iterations = config_.warmup_iterations + config_.iterations;
        return iterations * 20.0 * 7; // ~7 tests
    }

    double CUDABenchmark::benchmarkDeviceMemory()
    {
        cudaSetDevice(device_ordinal_);

        size_t num_bytes = config_.memory_test_bytes;
        size_t num_elements = num_bytes / sizeof(float);

        float *d_src = nullptr;
        float *d_dst = nullptr;
        CUDA_CHECK(cudaMalloc(&d_src, num_bytes));
        CUDA_CHECK(cudaMalloc(&d_dst, num_bytes));

        // Initialize
        CUDA_CHECK(cudaMemset(d_src, 0, num_bytes));
        CUDA_CHECK(cudaMemset(d_dst, 0, num_bytes));

        int block_size = 256;
        int num_blocks = static_cast<int>((num_elements + block_size - 1) / block_size);
        num_blocks = std::min(num_blocks, 65535);

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        // Warmup
        for (int w = 0; w < config_.warmup_iterations; ++w)
        {
            copyKernel<<<num_blocks, block_size>>>(d_src, d_dst, num_elements);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // Timed runs
        CUDA_CHECK(cudaEventRecord(start));
        for (int iter = 0; iter < config_.iterations; ++iter)
        {
            copyKernel<<<num_blocks, block_size>>>(d_src, d_dst, num_elements);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

        // Copy reads and writes
        double bytes_transferred = 2.0 * num_bytes * config_.iterations;
        double gbps = (bytes_transferred / (elapsed_ms / 1000.0)) / 1e9;

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        CUDA_CHECK(cudaFree(d_src));
        CUDA_CHECK(cudaFree(d_dst));

        return gbps;
    }

    double CUDABenchmark::benchmarkH2DPinned()
    {
        cudaSetDevice(device_ordinal_);

        size_t num_bytes = config_.memory_test_bytes;

        void *h_pinned = nullptr;
        void *d_buffer = nullptr;
        CUDA_CHECK(cudaMallocHost(&h_pinned, num_bytes));
        CUDA_CHECK(cudaMalloc(&d_buffer, num_bytes));

        memset(h_pinned, 0, num_bytes);

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        // Warmup
        for (int w = 0; w < config_.warmup_iterations; ++w)
        {
            CUDA_CHECK(cudaMemcpy(d_buffer, h_pinned, num_bytes, cudaMemcpyHostToDevice));
        }

        // Timed runs
        CUDA_CHECK(cudaEventRecord(start));
        for (int iter = 0; iter < config_.iterations; ++iter)
        {
            CUDA_CHECK(cudaMemcpy(d_buffer, h_pinned, num_bytes, cudaMemcpyHostToDevice));
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

        double gbps = (static_cast<double>(num_bytes) * config_.iterations / (elapsed_ms / 1000.0)) / 1e9;

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        CUDA_CHECK(cudaFreeHost(h_pinned));
        CUDA_CHECK(cudaFree(d_buffer));

        return gbps;
    }

    double CUDABenchmark::benchmarkH2DPageable()
    {
        cudaSetDevice(device_ordinal_);

        size_t num_bytes = config_.memory_test_bytes;

        std::vector<char> h_pageable(num_bytes, 0);
        void *d_buffer = nullptr;
        CUDA_CHECK(cudaMalloc(&d_buffer, num_bytes));

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        // Warmup
        for (int w = 0; w < config_.warmup_iterations; ++w)
        {
            CUDA_CHECK(cudaMemcpy(d_buffer, h_pageable.data(), num_bytes, cudaMemcpyHostToDevice));
        }

        // Timed runs
        CUDA_CHECK(cudaEventRecord(start));
        for (int iter = 0; iter < config_.iterations; ++iter)
        {
            CUDA_CHECK(cudaMemcpy(d_buffer, h_pageable.data(), num_bytes, cudaMemcpyHostToDevice));
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

        double gbps = (static_cast<double>(num_bytes) * config_.iterations / (elapsed_ms / 1000.0)) / 1e9;

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        CUDA_CHECK(cudaFree(d_buffer));

        return gbps;
    }

    double CUDABenchmark::benchmarkD2HPinned()
    {
        cudaSetDevice(device_ordinal_);

        size_t num_bytes = config_.memory_test_bytes;

        void *h_pinned = nullptr;
        void *d_buffer = nullptr;
        CUDA_CHECK(cudaMallocHost(&h_pinned, num_bytes));
        CUDA_CHECK(cudaMalloc(&d_buffer, num_bytes));

        CUDA_CHECK(cudaMemset(d_buffer, 1, num_bytes));

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        // Warmup
        for (int w = 0; w < config_.warmup_iterations; ++w)
        {
            CUDA_CHECK(cudaMemcpy(h_pinned, d_buffer, num_bytes, cudaMemcpyDeviceToHost));
        }

        // Timed runs
        CUDA_CHECK(cudaEventRecord(start));
        for (int iter = 0; iter < config_.iterations; ++iter)
        {
            CUDA_CHECK(cudaMemcpy(h_pinned, d_buffer, num_bytes, cudaMemcpyDeviceToHost));
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

        double gbps = (static_cast<double>(num_bytes) * config_.iterations / (elapsed_ms / 1000.0)) / 1e9;

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        CUDA_CHECK(cudaFreeHost(h_pinned));
        CUDA_CHECK(cudaFree(d_buffer));

        return gbps;
    }

    double CUDABenchmark::benchmarkD2HPageable()
    {
        cudaSetDevice(device_ordinal_);

        size_t num_bytes = config_.memory_test_bytes;

        std::vector<char> h_pageable(num_bytes);
        void *d_buffer = nullptr;
        CUDA_CHECK(cudaMalloc(&d_buffer, num_bytes));

        CUDA_CHECK(cudaMemset(d_buffer, 1, num_bytes));

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        // Warmup
        for (int w = 0; w < config_.warmup_iterations; ++w)
        {
            CUDA_CHECK(cudaMemcpy(h_pageable.data(), d_buffer, num_bytes, cudaMemcpyDeviceToHost));
        }

        // Timed runs
        CUDA_CHECK(cudaEventRecord(start));
        for (int iter = 0; iter < config_.iterations; ++iter)
        {
            CUDA_CHECK(cudaMemcpy(h_pageable.data(), d_buffer, num_bytes, cudaMemcpyDeviceToHost));
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

        double gbps = (static_cast<double>(num_bytes) * config_.iterations / (elapsed_ms / 1000.0)) / 1e9;

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        CUDA_CHECK(cudaFree(d_buffer));

        return gbps;
    }

    double CUDABenchmark::benchmarkComputeFP32()
    {
        cudaSetDevice(device_ordinal_);

        // Allocate minimal buffer - compute bound, not memory bound
        size_t num_elements = 1024 * 1024; // 4MB
        float *d_buffer = nullptr;
        CUDA_CHECK(cudaMalloc(&d_buffer, num_elements * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_buffer, 0, num_elements * sizeof(float)));

        int block_size = 256;
        int num_blocks = static_cast<int>((num_elements + block_size - 1) / block_size);

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        // Warmup
        for (int w = 0; w < config_.warmup_iterations; ++w)
        {
            computeKernelFP32<<<num_blocks, block_size>>>(d_buffer, num_elements);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // Timed runs
        CUDA_CHECK(cudaEventRecord(start));
        for (int iter = 0; iter < config_.iterations; ++iter)
        {
            computeKernelFP32<<<num_blocks, block_size>>>(d_buffer, num_elements);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

        // Each thread does 256 iterations * 4 FMAs * 2 ops = 2048 FLOPS
        double total_threads = static_cast<double>(num_blocks) * block_size;
        double flops_per_kernel = total_threads * 256 * 4 * 2;
        double total_flops = flops_per_kernel * config_.iterations;
        double gflops = (total_flops / (elapsed_ms / 1000.0)) / 1e9;

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        CUDA_CHECK(cudaFree(d_buffer));

        return gflops;
    }

    double CUDABenchmark::benchmarkComputeFP16()
    {
        cudaSetDevice(device_ordinal_);

        size_t num_elements = 1024 * 1024; // 2MB of half
        half *d_buffer = nullptr;
        CUDA_CHECK(cudaMalloc(&d_buffer, num_elements * sizeof(half)));
        CUDA_CHECK(cudaMemset(d_buffer, 0, num_elements * sizeof(half)));

        int block_size = 256;
        int num_blocks = static_cast<int>((num_elements + block_size - 1) / block_size);

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        // Warmup
        for (int w = 0; w < config_.warmup_iterations; ++w)
        {
            computeKernelFP16<<<num_blocks, block_size>>>(d_buffer, num_elements);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // Timed runs
        CUDA_CHECK(cudaEventRecord(start));
        for (int iter = 0; iter < config_.iterations; ++iter)
        {
            computeKernelFP16<<<num_blocks, block_size>>>(d_buffer, num_elements);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

        // Each thread does 256 iterations * 4 FMAs * 2 ops = 2048 FLOPS
        double total_threads = static_cast<double>(num_blocks) * block_size;
        double flops_per_kernel = total_threads * 256 * 4 * 2;
        double total_flops = flops_per_kernel * config_.iterations;
        double gflops = (total_flops / (elapsed_ms / 1000.0)) / 1e9;

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        CUDA_CHECK(cudaFree(d_buffer));

        return gflops;
    }

    double CUDABenchmark::benchmarkPeerTransfer(int peer_ordinal)
    {
        if (peer_ordinal == device_ordinal_)
        {
            return 0.0;
        }

        // Check P2P availability (NVLink or direct PCIe P2P)
        // This covers ALL direct GPU-to-GPU transfer methods without host staging
        int can_access = 0;
        cudaError_t err = cudaDeviceCanAccessPeer(&can_access, device_ordinal_, peer_ordinal);
        if (err != cudaSuccess || !can_access)
        {
            LOG_DEBUG("Direct P2P not available between GPU " << device_ordinal_ << " and GPU " << peer_ordinal
                      << " (no NVLink and no direct PCIe P2P)"
                      << " - transfers must stage through host memory");
            return 0.0;
        }

        // Enable P2P
        cudaSetDevice(device_ordinal_);
        err = cudaDeviceEnablePeerAccess(peer_ordinal, 0);
        if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled)
        {
            LOG_DEBUG("Failed to enable P2P: " << cudaGetErrorString(err));
            return 0.0;
        }

        size_t num_bytes = config_.memory_test_bytes;

        void *d_src = nullptr;
        void *d_dst = nullptr;

        cudaSetDevice(device_ordinal_);
        CUDA_CHECK(cudaMalloc(&d_src, num_bytes));
        CUDA_CHECK(cudaMemset(d_src, 1, num_bytes));

        cudaSetDevice(peer_ordinal);
        CUDA_CHECK(cudaMalloc(&d_dst, num_bytes));

        cudaSetDevice(device_ordinal_);

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        // Warmup
        for (int w = 0; w < config_.warmup_iterations; ++w)
        {
            CUDA_CHECK(cudaMemcpyPeer(d_dst, peer_ordinal, d_src, device_ordinal_, num_bytes));
        }

        // Timed runs
        CUDA_CHECK(cudaEventRecord(start));
        for (int iter = 0; iter < config_.iterations; ++iter)
        {
            CUDA_CHECK(cudaMemcpyPeer(d_dst, peer_ordinal, d_src, device_ordinal_, num_bytes));
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

        double gbps = (static_cast<double>(num_bytes) * config_.iterations / (elapsed_ms / 1000.0)) / 1e9;

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));

        cudaSetDevice(peer_ordinal);
        cudaFree(d_dst);
        cudaSetDevice(device_ordinal_);
        cudaFree(d_src);

        return gbps;
    }

    DeviceBenchmarkResult CUDABenchmark::run()
    {
        DeviceBenchmarkResult result;
        result.device = DeviceId::cuda(device_ordinal_);

        cudaError_t err = cudaSetDevice(device_ordinal_);
        if (err != cudaSuccess)
        {
            LOG_ERROR("Failed to set CUDA device " << device_ordinal_ << ": " << cudaGetErrorString(err));
            result.valid = false;
            return result;
        }

        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, device_ordinal_);
        LOG_DEBUG("Starting CUDA benchmark for GPU " << device_ordinal_ << ": " << props.name);

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
        cudaGetDeviceCount(&num_gpus);
        for (int peer = 0; peer < num_gpus; ++peer)
        {
            if (peer != device_ordinal_)
            {
                double p2p_gbps = benchmarkPeerTransfer(peer);
                if (p2p_gbps > 0.0)
                {
                    result.peer_transfer_gbps[DeviceId::cuda(peer)] = p2p_gbps;
                }
            }
        }

        result.benchmark_duration_ms = getTimeMs() - start_time;
        result.valid = true;

        LOG_INFO("CUDA GPU " << device_ordinal_ << " Benchmark completed in " << result.benchmark_duration_ms << " ms");
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

    std::vector<DeviceId> CUDABenchmark::enumerateDevices()
    {
        std::vector<DeviceId> devices;

        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err == cudaSuccess)
        {
            for (int i = 0; i < device_count; ++i)
            {
                devices.push_back(DeviceId::cuda(i));
            }
        }

        return devices;
    }

    double CUDABenchmark::copyHostToDevice(int device_ordinal, const void *h_src, size_t num_bytes)
    {
        cudaSetDevice(device_ordinal);

        void *d_dst = nullptr;
        cudaError_t err = cudaMalloc(&d_dst, num_bytes);
        if (err != cudaSuccess)
        {
            LOG_WARN("CUDA copyHostToDevice: cudaMalloc failed for " << num_bytes << " bytes");
            return 0.0;
        }

        // Warm up
        cudaMemcpy(d_dst, h_src, num_bytes, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();

        // Timed copy
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        cudaMemcpy(d_dst, h_src, num_bytes, cudaMemcpyHostToDevice);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_dst);

        double seconds = ms / 1000.0;
        return (num_bytes / (1024.0 * 1024.0 * 1024.0)) / seconds;
    }

    double CUDABenchmark::copyDeviceToHost(int device_ordinal, void *h_dst, size_t num_bytes)
    {
        cudaSetDevice(device_ordinal);

        void *d_src = nullptr;
        cudaError_t err = cudaMalloc(&d_src, num_bytes);
        if (err != cudaSuccess)
        {
            LOG_WARN("CUDA copyDeviceToHost: cudaMalloc failed for " << num_bytes << " bytes");
            return 0.0;
        }

        // Initialize device memory with pattern
        cudaMemset(d_src, 0xAB, num_bytes);
        cudaDeviceSynchronize();

        // Warm up
        cudaMemcpy(h_dst, d_src, num_bytes, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        // Timed copy
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        cudaMemcpy(h_dst, d_src, num_bytes, cudaMemcpyDeviceToHost);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_src);

        double seconds = ms / 1000.0;
        return (num_bytes / (1024.0 * 1024.0 * 1024.0)) / seconds;
    }

} // namespace llaminar2
