/**
 * HIP Memory Copy Microbenchmark
 *
 * Standalone benchmark to measure HIP memcpy performance and identify
 * any abnormal latency in host-device transfers.
 *
 * Build:
 *   hipcc -O3 -o hip_memcpy_benchmark hip_memcpy_benchmark.cpp
 *
 * Run:
 *   ./hip_memcpy_benchmark
 */

#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define HIP_CHECK(call)                                                     \
    do                                                                      \
    {                                                                       \
        hipError_t err = call;                                              \
        if (err != hipSuccess)                                              \
        {                                                                   \
            fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__, \
                    hipGetErrorString(err));                                \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

using Clock = std::chrono::high_resolution_clock;

struct BenchmarkResult
{
    const char *name;
    size_t bytes;
    double avg_us;
    double min_us;
    double max_us;
    double bandwidth_gbps;
};

void print_result(const BenchmarkResult &r)
{
    printf("  %-40s %8zu bytes | avg: %8.2f us | min: %8.2f us | max: %8.2f us | %.2f GB/s\n",
           r.name, r.bytes, r.avg_us, r.min_us, r.max_us, r.bandwidth_gbps);
}

// Benchmark hipMemcpy H2D
BenchmarkResult bench_h2d(void *d_ptr, void *h_ptr, size_t bytes, int iterations)
{
    std::vector<double> times;
    times.reserve(iterations);

    // Warmup
    for (int i = 0; i < 3; i++)
    {
        HIP_CHECK(hipMemcpy(d_ptr, h_ptr, bytes, hipMemcpyHostToDevice));
        HIP_CHECK(hipDeviceSynchronize());
    }

    for (int i = 0; i < iterations; i++)
    {
        auto start = Clock::now();
        HIP_CHECK(hipMemcpy(d_ptr, h_ptr, bytes, hipMemcpyHostToDevice));
        HIP_CHECK(hipDeviceSynchronize());
        auto end = Clock::now();
        times.push_back(std::chrono::duration<double, std::micro>(end - start).count());
    }

    double sum = 0, min_t = times[0], max_t = times[0];
    for (double t : times)
    {
        sum += t;
        if (t < min_t)
            min_t = t;
        if (t > max_t)
            max_t = t;
    }
    double avg = sum / iterations;
    double bandwidth = (bytes / 1e9) / (avg / 1e6); // GB/s

    return {"hipMemcpy H2D", bytes, avg, min_t, max_t, bandwidth};
}

// Benchmark hipMemcpy D2H
BenchmarkResult bench_d2h(void *d_ptr, void *h_ptr, size_t bytes, int iterations)
{
    std::vector<double> times;
    times.reserve(iterations);

    // Warmup
    for (int i = 0; i < 3; i++)
    {
        HIP_CHECK(hipMemcpy(h_ptr, d_ptr, bytes, hipMemcpyDeviceToHost));
        HIP_CHECK(hipDeviceSynchronize());
    }

    for (int i = 0; i < iterations; i++)
    {
        auto start = Clock::now();
        HIP_CHECK(hipMemcpy(h_ptr, d_ptr, bytes, hipMemcpyDeviceToHost));
        HIP_CHECK(hipDeviceSynchronize());
        auto end = Clock::now();
        times.push_back(std::chrono::duration<double, std::micro>(end - start).count());
    }

    double sum = 0, min_t = times[0], max_t = times[0];
    for (double t : times)
    {
        sum += t;
        if (t < min_t)
            min_t = t;
        if (t > max_t)
            max_t = t;
    }
    double avg = sum / iterations;
    double bandwidth = (bytes / 1e9) / (avg / 1e6); // GB/s

    return {"hipMemcpy D2H", bytes, avg, min_t, max_t, bandwidth};
}

// Benchmark hipMemcpyAsync H2D
BenchmarkResult bench_h2d_async(void *d_ptr, void *h_ptr, size_t bytes, hipStream_t stream, int iterations)
{
    std::vector<double> times;
    times.reserve(iterations);

    // Warmup
    for (int i = 0; i < 3; i++)
    {
        HIP_CHECK(hipMemcpyAsync(d_ptr, h_ptr, bytes, hipMemcpyHostToDevice, stream));
        HIP_CHECK(hipStreamSynchronize(stream));
    }

    for (int i = 0; i < iterations; i++)
    {
        auto start = Clock::now();
        HIP_CHECK(hipMemcpyAsync(d_ptr, h_ptr, bytes, hipMemcpyHostToDevice, stream));
        HIP_CHECK(hipStreamSynchronize(stream));
        auto end = Clock::now();
        times.push_back(std::chrono::duration<double, std::micro>(end - start).count());
    }

    double sum = 0, min_t = times[0], max_t = times[0];
    for (double t : times)
    {
        sum += t;
        if (t < min_t)
            min_t = t;
        if (t > max_t)
            max_t = t;
    }
    double avg = sum / iterations;
    double bandwidth = (bytes / 1e9) / (avg / 1e6); // GB/s

    return {"hipMemcpyAsync H2D", bytes, avg, min_t, max_t, bandwidth};
}

// Benchmark hipMemcpyAsync D2H
BenchmarkResult bench_d2h_async(void *d_ptr, void *h_ptr, size_t bytes, hipStream_t stream, int iterations)
{
    std::vector<double> times;
    times.reserve(iterations);

    // Warmup
    for (int i = 0; i < 3; i++)
    {
        HIP_CHECK(hipMemcpyAsync(h_ptr, d_ptr, bytes, hipMemcpyDeviceToHost, stream));
        HIP_CHECK(hipStreamSynchronize(stream));
    }

    for (int i = 0; i < iterations; i++)
    {
        auto start = Clock::now();
        HIP_CHECK(hipMemcpyAsync(h_ptr, d_ptr, bytes, hipMemcpyDeviceToHost, stream));
        HIP_CHECK(hipStreamSynchronize(stream));
        auto end = Clock::now();
        times.push_back(std::chrono::duration<double, std::micro>(end - start).count());
    }

    double sum = 0, min_t = times[0], max_t = times[0];
    for (double t : times)
    {
        sum += t;
        if (t < min_t)
            min_t = t;
        if (t > max_t)
            max_t = t;
    }
    double avg = sum / iterations;
    double bandwidth = (bytes / 1e9) / (avg / 1e6); // GB/s

    return {"hipMemcpyAsync D2H", bytes, avg, min_t, max_t, bandwidth};
}

// Benchmark hipDeviceSynchronize overhead (no actual work)
BenchmarkResult bench_device_sync(int iterations)
{
    std::vector<double> times;
    times.reserve(iterations);

    // Warmup
    for (int i = 0; i < 3; i++)
    {
        HIP_CHECK(hipDeviceSynchronize());
    }

    for (int i = 0; i < iterations; i++)
    {
        auto start = Clock::now();
        HIP_CHECK(hipDeviceSynchronize());
        auto end = Clock::now();
        times.push_back(std::chrono::duration<double, std::micro>(end - start).count());
    }

    double sum = 0, min_t = times[0], max_t = times[0];
    for (double t : times)
    {
        sum += t;
        if (t < min_t)
            min_t = t;
        if (t > max_t)
            max_t = t;
    }
    double avg = sum / iterations;

    return {"hipDeviceSynchronize (idle)", 0, avg, min_t, max_t, 0};
}

// Benchmark hipStreamSynchronize overhead (no actual work)
BenchmarkResult bench_stream_sync(hipStream_t stream, int iterations)
{
    std::vector<double> times;
    times.reserve(iterations);

    // Warmup
    for (int i = 0; i < 3; i++)
    {
        HIP_CHECK(hipStreamSynchronize(stream));
    }

    for (int i = 0; i < iterations; i++)
    {
        auto start = Clock::now();
        HIP_CHECK(hipStreamSynchronize(stream));
        auto end = Clock::now();
        times.push_back(std::chrono::duration<double, std::micro>(end - start).count());
    }

    double sum = 0, min_t = times[0], max_t = times[0];
    for (double t : times)
    {
        sum += t;
        if (t < min_t)
            min_t = t;
        if (t > max_t)
            max_t = t;
    }
    double avg = sum / iterations;

    return {"hipStreamSynchronize (idle)", 0, avg, min_t, max_t, 0};
}

// Benchmark with pinned (page-locked) host memory
BenchmarkResult bench_h2d_pinned(void *d_ptr, void *h_pinned, size_t bytes, hipStream_t stream, int iterations)
{
    std::vector<double> times;
    times.reserve(iterations);

    // Warmup
    for (int i = 0; i < 3; i++)
    {
        HIP_CHECK(hipMemcpyAsync(d_ptr, h_pinned, bytes, hipMemcpyHostToDevice, stream));
        HIP_CHECK(hipStreamSynchronize(stream));
    }

    for (int i = 0; i < iterations; i++)
    {
        auto start = Clock::now();
        HIP_CHECK(hipMemcpyAsync(d_ptr, h_pinned, bytes, hipMemcpyHostToDevice, stream));
        HIP_CHECK(hipStreamSynchronize(stream));
        auto end = Clock::now();
        times.push_back(std::chrono::duration<double, std::micro>(end - start).count());
    }

    double sum = 0, min_t = times[0], max_t = times[0];
    for (double t : times)
    {
        sum += t;
        if (t < min_t)
            min_t = t;
        if (t > max_t)
            max_t = t;
    }
    double avg = sum / iterations;
    double bandwidth = (bytes / 1e9) / (avg / 1e6); // GB/s

    return {"hipMemcpyAsync H2D (pinned)", bytes, avg, min_t, max_t, bandwidth};
}

BenchmarkResult bench_d2h_pinned(void *d_ptr, void *h_pinned, size_t bytes, hipStream_t stream, int iterations)
{
    std::vector<double> times;
    times.reserve(iterations);

    // Warmup
    for (int i = 0; i < 3; i++)
    {
        HIP_CHECK(hipMemcpyAsync(h_pinned, d_ptr, bytes, hipMemcpyDeviceToHost, stream));
        HIP_CHECK(hipStreamSynchronize(stream));
    }

    for (int i = 0; i < iterations; i++)
    {
        auto start = Clock::now();
        HIP_CHECK(hipMemcpyAsync(h_pinned, d_ptr, bytes, hipMemcpyDeviceToHost, stream));
        HIP_CHECK(hipStreamSynchronize(stream));
        auto end = Clock::now();
        times.push_back(std::chrono::duration<double, std::micro>(end - start).count());
    }

    double sum = 0, min_t = times[0], max_t = times[0];
    for (double t : times)
    {
        sum += t;
        if (t < min_t)
            min_t = t;
        if (t > max_t)
            max_t = t;
    }
    double avg = sum / iterations;
    double bandwidth = (bytes / 1e9) / (avg / 1e6); // GB/s

    return {"hipMemcpyAsync D2H (pinned)", bytes, avg, min_t, max_t, bandwidth};
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

    printf("╔══════════════════════════════════════════════════════════════════════════════════════╗\n");
    printf("║                        HIP MEMORY COPY MICROBENCHMARK                                ║\n");
    printf("╠══════════════════════════════════════════════════════════════════════════════════════╣\n");
    printf("║ Device: %s (device %d)\n", props.name, device);
    printf("║ PCIe: Gen %d x %d\n", props.pciBusID, props.pciDeviceID);
    printf("║ Memory: %.2f GB\n", props.totalGlobalMem / 1e9);
    printf("╚══════════════════════════════════════════════════════════════════════════════════════╝\n\n");

    // Test sizes: typical tensor sizes in inference
    std::vector<size_t> sizes = {
        4 * 1024,          // 4 KB - small tensor
        64 * 1024,         // 64 KB - typical activation row
        256 * 1024,        // 256 KB
        1024 * 1024,       // 1 MB
        4 * 1024 * 1024,   // 4 MB
        16 * 1024 * 1024,  // 16 MB - typical layer weights
        64 * 1024 * 1024,  // 64 MB
        256 * 1024 * 1024, // 256 MB - large weight matrix
    };

    const int iterations = 100;

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    // First, test sync overhead
    printf("=== SYNCHRONIZATION OVERHEAD ===\n");
    print_result(bench_device_sync(iterations));
    print_result(bench_stream_sync(stream, iterations));
    printf("\n");

    for (size_t bytes : sizes)
    {
        printf("=== TRANSFER SIZE: ");
        if (bytes >= 1024 * 1024)
        {
            printf("%zu MB ===\n", bytes / (1024 * 1024));
        }
        else
        {
            printf("%zu KB ===\n", bytes / 1024);
        }

        // Allocate device memory
        void *d_ptr;
        HIP_CHECK(hipMalloc(&d_ptr, bytes));

        // Allocate pageable host memory
        void *h_pageable = malloc(bytes);
        memset(h_pageable, 0xAB, bytes);

        // Allocate pinned host memory
        void *h_pinned;
        HIP_CHECK(hipHostMalloc(&h_pinned, bytes, hipHostMallocDefault));
        memset(h_pinned, 0xCD, bytes);

        // Benchmark pageable memory
        print_result(bench_h2d(d_ptr, h_pageable, bytes, iterations));
        print_result(bench_d2h(d_ptr, h_pageable, bytes, iterations));
        print_result(bench_h2d_async(d_ptr, h_pageable, bytes, stream, iterations));
        print_result(bench_d2h_async(d_ptr, h_pageable, bytes, stream, iterations));

        // Benchmark pinned memory
        print_result(bench_h2d_pinned(d_ptr, h_pinned, bytes, stream, iterations));
        print_result(bench_d2h_pinned(d_ptr, h_pinned, bytes, stream, iterations));

        printf("\n");

        // Cleanup
        HIP_CHECK(hipFree(d_ptr));
        free(h_pageable);
        HIP_CHECK(hipHostFree(h_pinned));
    }

    HIP_CHECK(hipStreamDestroy(stream));

    printf("Benchmark complete.\n");
    return 0;
}
