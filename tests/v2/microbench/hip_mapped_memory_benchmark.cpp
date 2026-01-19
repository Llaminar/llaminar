/**
 * @file hip_mapped_memory_benchmark.cpp
 * @brief Benchmark mapped pinned memory (zero-copy) vs explicit D2H memcpy
 *
 * Tests whether GPU can write directly to host memory faster than
 * compute + explicit memcpy approach.
 *
 * Build: Part of build_v2_integration
 * Run: ./build_v2_integration/tests/v2/v2_microbench_hip_mapped_memory
 */

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <cmath>

#define HIP_CHECK(call)                                          \
    do                                                           \
    {                                                            \
        hipError_t err = call;                                   \
        if (err != hipSuccess)                                   \
        {                                                        \
            fprintf(stderr, "HIP error at %s:%d: %s\n",          \
                    __FILE__, __LINE__, hipGetErrorString(err)); \
            exit(1);                                             \
        }                                                        \
    } while (0)

// Simple kernel that writes output to a buffer
// Simulates the output phase of a GEMM kernel
__global__ void write_output_kernel(float *output, int n, float value)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        // Simulate some computation before writing
        float result = value + static_cast<float>(idx) * 0.001f;
        output[idx] = result;
    }
}

// Kernel that writes to mapped host memory
// The GPU writes directly to host RAM via PCIe
__global__ void write_to_host_kernel(float *__restrict__ host_mapped_ptr, int n, float value)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        float result = value + static_cast<float>(idx) * 0.001f;
        // This write goes directly to host memory via PCIe!
        host_mapped_ptr[idx] = result;
    }
}

struct BenchmarkResult
{
    const char *name;
    double avg_ms;
    double min_ms;
    double max_ms;
    double bandwidth_gbps;
};

// Benchmark 1: Traditional approach - compute on device, then memcpy D2H
BenchmarkResult benchmark_compute_then_memcpy(int n, int iterations)
{
    size_t bytes = n * sizeof(float);

    float *d_output;
    float *h_output;

    HIP_CHECK(hipMalloc(&d_output, bytes));
    HIP_CHECK(hipHostMalloc(&h_output, bytes, hipHostMallocDefault)); // Pinned for fast memcpy

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    // Warmup
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    write_output_kernel<<<blocks, threads, 0, stream>>>(d_output, n, 1.0f);
    HIP_CHECK(hipMemcpyAsync(h_output, d_output, bytes, hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    // Benchmark
    std::vector<double> times;
    for (int i = 0; i < iterations; i++)
    {
        auto start = std::chrono::high_resolution_clock::now();

        write_output_kernel<<<blocks, threads, 0, stream>>>(d_output, n, static_cast<float>(i));
        HIP_CHECK(hipMemcpyAsync(h_output, d_output, bytes, hipMemcpyDeviceToHost, stream));
        HIP_CHECK(hipStreamSynchronize(stream));

        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        times.push_back(ms);
    }

    // Verify
    if (std::abs(h_output[0] - static_cast<float>(iterations - 1)) > 0.01f)
    {
        fprintf(stderr, "Verification failed: expected %f, got %f\n",
                static_cast<float>(iterations - 1), h_output[0]);
    }

    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK(hipFree(d_output));
    HIP_CHECK(hipHostFree(h_output));

    double sum = 0, min_t = 1e9, max_t = 0;
    for (double t : times)
    {
        sum += t;
        min_t = std::min(min_t, t);
        max_t = std::max(max_t, t);
    }
    double avg = sum / iterations;
    double bandwidth = (bytes / 1e9) / (avg / 1000.0); // GB/s

    return {"Compute + D2H memcpy", avg, min_t, max_t, bandwidth};
}

// Benchmark 2: Zero-copy approach - GPU writes directly to mapped host memory
BenchmarkResult benchmark_mapped_host_memory(int n, int iterations)
{
    size_t bytes = n * sizeof(float);

    float *h_mapped;
    float *d_mapped_ptr;

    // Allocate pinned + mapped host memory
    // hipHostMallocMapped allows GPU to access this memory directly
    HIP_CHECK(hipHostMalloc(&h_mapped, bytes, hipHostMallocMapped));

    // Get the device-visible pointer for the mapped memory
    HIP_CHECK(hipHostGetDevicePointer((void **)&d_mapped_ptr, h_mapped, 0));

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    // Warmup
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    write_to_host_kernel<<<blocks, threads, 0, stream>>>(d_mapped_ptr, n, 1.0f);
    HIP_CHECK(hipStreamSynchronize(stream));

    // Benchmark
    std::vector<double> times;
    for (int i = 0; i < iterations; i++)
    {
        auto start = std::chrono::high_resolution_clock::now();

        // GPU writes directly to host memory - no explicit memcpy!
        write_to_host_kernel<<<blocks, threads, 0, stream>>>(d_mapped_ptr, n, static_cast<float>(i));
        HIP_CHECK(hipStreamSynchronize(stream));

        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        times.push_back(ms);
    }

    // Verify - data should be directly visible in h_mapped
    if (std::abs(h_mapped[0] - static_cast<float>(iterations - 1)) > 0.01f)
    {
        fprintf(stderr, "Verification failed: expected %f, got %f\n",
                static_cast<float>(iterations - 1), h_mapped[0]);
    }

    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK(hipHostFree(h_mapped));

    double sum = 0, min_t = 1e9, max_t = 0;
    for (double t : times)
    {
        sum += t;
        min_t = std::min(min_t, t);
        max_t = std::max(max_t, t);
    }
    double avg = sum / iterations;
    double bandwidth = (bytes / 1e9) / (avg / 1000.0); // GB/s

    return {"Zero-copy (mapped host)", avg, min_t, max_t, bandwidth};
}

// Benchmark 3: Async compute + memcpy overlap (for reference)
BenchmarkResult benchmark_async_overlap(int n, int iterations)
{
    size_t bytes = n * sizeof(float);

    // Use double buffering
    float *d_output[2];
    float *h_output[2];

    for (int i = 0; i < 2; i++)
    {
        HIP_CHECK(hipMalloc(&d_output[i], bytes));
        HIP_CHECK(hipHostMalloc(&h_output[i], bytes, hipHostMallocDefault));
    }

    hipStream_t compute_stream, copy_stream;
    HIP_CHECK(hipStreamCreate(&compute_stream));
    HIP_CHECK(hipStreamCreate(&copy_stream));

    hipEvent_t compute_done;
    HIP_CHECK(hipEventCreate(&compute_done));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    // Warmup
    write_output_kernel<<<blocks, threads, 0, compute_stream>>>(d_output[0], n, 1.0f);
    HIP_CHECK(hipEventRecord(compute_done, compute_stream));
    HIP_CHECK(hipStreamWaitEvent(copy_stream, compute_done, 0));
    HIP_CHECK(hipMemcpyAsync(h_output[0], d_output[0], bytes, hipMemcpyDeviceToHost, copy_stream));
    HIP_CHECK(hipStreamSynchronize(copy_stream));

    // Benchmark with double buffering
    std::vector<double> times;
    for (int i = 0; i < iterations; i++)
    {
        int curr = i % 2;
        int prev = (i + 1) % 2;

        auto start = std::chrono::high_resolution_clock::now();

        // Start compute for current iteration
        write_output_kernel<<<blocks, threads, 0, compute_stream>>>(d_output[curr], n, static_cast<float>(i));
        HIP_CHECK(hipEventRecord(compute_done, compute_stream));

        // Copy previous result while compute runs (if not first iteration)
        if (i > 0)
        {
            HIP_CHECK(hipStreamWaitEvent(copy_stream, compute_done, 0));
            HIP_CHECK(hipMemcpyAsync(h_output[prev], d_output[prev], bytes, hipMemcpyDeviceToHost, copy_stream));
        }

        // Wait for both streams
        HIP_CHECK(hipStreamSynchronize(compute_stream));
        HIP_CHECK(hipStreamSynchronize(copy_stream));

        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        times.push_back(ms);
    }

    // Final copy
    HIP_CHECK(hipMemcpyAsync(h_output[(iterations - 1) % 2], d_output[(iterations - 1) % 2],
                             bytes, hipMemcpyDeviceToHost, copy_stream));
    HIP_CHECK(hipStreamSynchronize(copy_stream));

    HIP_CHECK(hipEventDestroy(compute_done));
    HIP_CHECK(hipStreamDestroy(compute_stream));
    HIP_CHECK(hipStreamDestroy(copy_stream));
    for (int i = 0; i < 2; i++)
    {
        HIP_CHECK(hipFree(d_output[i]));
        HIP_CHECK(hipHostFree(h_output[i]));
    }

    double sum = 0, min_t = 1e9, max_t = 0;
    for (double t : times)
    {
        sum += t;
        min_t = std::min(min_t, t);
        max_t = std::max(max_t, t);
    }
    double avg = sum / iterations;
    double bandwidth = (bytes / 1e9) / (avg / 1000.0);

    return {"Async overlap (double-buffer)", avg, min_t, max_t, bandwidth};
}

void print_result(const BenchmarkResult &r, size_t bytes)
{
    printf("  %-30s: %8.3f ms avg (min=%.3f max=%.3f) | %.2f GB/s\n",
           r.name, r.avg_ms, r.min_ms, r.max_ms, r.bandwidth_gbps);
}

int main(int argc, char **argv)
{
    int device = 0;
    if (argc > 1)
    {
        device = atoi(argv[1]);
    }

    HIP_CHECK(hipSetDevice(device));

    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, device));
    printf("Device: %s\n\n", props.name);

    // Test different sizes
    std::vector<int> sizes = {
        1024 * 1024,      // 4 MB (small activation)
        4 * 1024 * 1024,  // 16 MB (typical GEMM output)
        16 * 1024 * 1024, // 64 MB (large GEMM output)
        64 * 1024 * 1024, // 256 MB (very large)
    };

    int iterations = 20;

    for (int n : sizes)
    {
        size_t bytes = n * sizeof(float);
        printf("=== Size: %d elements (%.2f MB) ===\n", n, bytes / (1024.0 * 1024.0));

        auto r1 = benchmark_compute_then_memcpy(n, iterations);
        auto r2 = benchmark_mapped_host_memory(n, iterations);
        auto r3 = benchmark_async_overlap(n, iterations);

        print_result(r1, bytes);
        print_result(r2, bytes);
        print_result(r3, bytes);

        printf("\n");
    }

    printf("=== Summary ===\n");
    printf("- 'Compute + D2H memcpy': Traditional approach, kernel runs then explicit copy\n");
    printf("- 'Zero-copy (mapped host)': GPU writes directly to host memory via PCIe\n");
    printf("- 'Async overlap': Double-buffered compute/copy overlap\n");
    printf("\n");
    printf("If zero-copy is faster, we can eliminate the D2H sync bottleneck!\n");

    return 0;
}
