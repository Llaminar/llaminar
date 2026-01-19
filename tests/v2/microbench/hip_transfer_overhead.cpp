/**
 * HIP Transfer Overhead Microbenchmark
 *
 * Measures individual components of the transfer lifecycle:
 * 1. hipHostMalloc (pinning)
 * 2. hipMalloc (device allocation)
 * 3. hipMemcpy H2D / D2H
 * 4. hipDeviceSynchronize
 * 5. Full round-trip with/without kernel execution
 *
 * Build:
 *   hipcc -O3 -o hip_transfer_overhead hip_transfer_overhead.cpp
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

double measure_us(std::function<void()> fn)
{
    auto start = Clock::now();
    fn();
    auto end = Clock::now();
    return std::chrono::duration<double, std::micro>(end - start).count();
}

// Simple kernel that just touches memory
__global__ void touch_memory_kernel(float *data, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        data[idx] = data[idx] * 1.0001f + 0.0001f;
    }
}

// Heavier compute kernel (simulate GEMM-like work)
__global__ void heavy_compute_kernel(float *data, size_t n, int iterations)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        float val = data[idx];
        for (int i = 0; i < iterations; i++)
        {
            val = val * 1.0001f + 0.0001f;
            val = __fsqrt_rn(val * val + 1.0f);
        }
        data[idx] = val;
    }
}

void benchmark_transfer_overhead(int device, size_t bytes)
{
    HIP_CHECK(hipSetDevice(device));

    printf("\n=== Transfer Overhead Analysis: %zu KB ===\n", bytes / 1024);

    const int iterations = 20;
    std::vector<double> times;
    times.reserve(iterations);

    // 1. Measure hipHostMalloc (pinning)
    {
        double total = 0;
        for (int i = 0; i < iterations; i++)
        {
            void *ptr;
            double t = measure_us([&]()
                                  { HIP_CHECK(hipHostMalloc(&ptr, bytes, hipHostMallocDefault)); });
            total += t;
            HIP_CHECK(hipHostFree(ptr));
        }
        printf("  hipHostMalloc (pinning):        %8.2f us avg\n", total / iterations);
    }

    // 2. Measure hipMalloc (device allocation)
    {
        double total = 0;
        for (int i = 0; i < iterations; i++)
        {
            void *ptr;
            double t = measure_us([&]()
                                  { HIP_CHECK(hipMalloc(&ptr, bytes)); });
            total += t;
            HIP_CHECK(hipFree(ptr));
        }
        printf("  hipMalloc (device alloc):       %8.2f us avg\n", total / iterations);
    }

    // Pre-allocate for remaining tests
    void *h_pinned;
    void *h_pageable = malloc(bytes);
    void *d_ptr;
    HIP_CHECK(hipHostMalloc(&h_pinned, bytes, hipHostMallocDefault));
    HIP_CHECK(hipMalloc(&d_ptr, bytes));
    memset(h_pinned, 0xAB, bytes);
    memset(h_pageable, 0xCD, bytes);

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    // Warmup
    HIP_CHECK(hipMemcpy(d_ptr, h_pinned, bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipDeviceSynchronize());

    // 3. Measure pure D2H memcpy (pinned, no prior kernel)
    {
        double total = 0;
        for (int i = 0; i < iterations; i++)
        {
            double t = measure_us([&]()
                                  { HIP_CHECK(hipMemcpy(h_pinned, d_ptr, bytes, hipMemcpyDeviceToHost)); });
            total += t;
        }
        printf("  hipMemcpy D2H (pinned, idle):   %8.2f us avg (%.2f GB/s)\n",
               total / iterations, (bytes / 1e9) / ((total / iterations) / 1e6));
    }

    // 4. Measure D2H memcpy AFTER light kernel (implicit sync)
    {
        double total = 0;
        size_t n = bytes / sizeof(float);
        int blocks = (n + 255) / 256;

        for (int i = 0; i < iterations; i++)
        {
            // Launch kernel first
            touch_memory_kernel<<<blocks, 256>>>((float *)d_ptr, n);

            double t = measure_us([&]()
                                  { HIP_CHECK(hipMemcpy(h_pinned, d_ptr, bytes, hipMemcpyDeviceToHost)); });
            total += t;
        }
        printf("  hipMemcpy D2H (after light kernel): %8.2f us avg\n", total / iterations);
    }

    // 5. Measure D2H memcpy AFTER heavy kernel (implicit sync)
    {
        double total = 0;
        size_t n = bytes / sizeof(float);
        int blocks = (n + 255) / 256;

        for (int i = 0; i < iterations; i++)
        {
            // Launch heavier kernel
            heavy_compute_kernel<<<blocks, 256>>>((float *)d_ptr, n, 1000);

            double t = measure_us([&]()
                                  { HIP_CHECK(hipMemcpy(h_pinned, d_ptr, bytes, hipMemcpyDeviceToHost)); });
            total += t;
        }
        printf("  hipMemcpy D2H (after heavy kernel): %8.2f us avg\n", total / iterations);
    }

    // 6. Measure hipDeviceSynchronize after kernel
    {
        double total = 0;
        size_t n = bytes / sizeof(float);
        int blocks = (n + 255) / 256;

        for (int i = 0; i < iterations; i++)
        {
            touch_memory_kernel<<<blocks, 256>>>((float *)d_ptr, n);

            double t = measure_us([&]()
                                  { HIP_CHECK(hipDeviceSynchronize()); });
            total += t;
        }
        printf("  hipDeviceSynchronize (after light): %8.2f us avg\n", total / iterations);
    }

    // 7. Separate kernel sync THEN memcpy
    {
        double sync_total = 0, memcpy_total = 0;
        size_t n = bytes / sizeof(float);
        int blocks = (n + 255) / 256;

        for (int i = 0; i < iterations; i++)
        {
            touch_memory_kernel<<<blocks, 256>>>((float *)d_ptr, n);

            double t_sync = measure_us([&]()
                                       { HIP_CHECK(hipDeviceSynchronize()); });
            sync_total += t_sync;

            double t_memcpy = measure_us([&]()
                                         { HIP_CHECK(hipMemcpy(h_pinned, d_ptr, bytes, hipMemcpyDeviceToHost)); });
            memcpy_total += t_memcpy;
        }
        printf("  Sync then memcpy (light kernel):\n");
        printf("    - sync:   %8.2f us avg\n", sync_total / iterations);
        printf("    - memcpy: %8.2f us avg\n", memcpy_total / iterations);
        printf("    - total:  %8.2f us avg\n", (sync_total + memcpy_total) / iterations);
    }

    // 8. Measure pin+memcpy combo (simulates ensureOnHost with fresh pin)
    {
        double total = 0;
        for (int i = 0; i < iterations; i++)
        {
            void *h_fresh_pinned;

            double t = measure_us([&]()
                                  {
                HIP_CHECK(hipHostMalloc(&h_fresh_pinned, bytes, hipHostMallocDefault));
                HIP_CHECK(hipMemcpy(h_fresh_pinned, d_ptr, bytes, hipMemcpyDeviceToHost)); });
            total += t;

            HIP_CHECK(hipHostFree(h_fresh_pinned));
        }
        printf("  Pin + D2H memcpy combo:         %8.2f us avg\n", total / iterations);
    }

    // 9. Test pageable vs pinned
    {
        double pinned_total = 0, pageable_total = 0;

        for (int i = 0; i < iterations; i++)
        {
            double t = measure_us([&]()
                                  { HIP_CHECK(hipMemcpy(h_pinned, d_ptr, bytes, hipMemcpyDeviceToHost)); });
            pinned_total += t;
        }

        for (int i = 0; i < iterations; i++)
        {
            double t = measure_us([&]()
                                  { HIP_CHECK(hipMemcpy(h_pageable, d_ptr, bytes, hipMemcpyDeviceToHost)); });
            pageable_total += t;
        }

        printf("  D2H pinned vs pageable:\n");
        printf("    - pinned:   %8.2f us avg\n", pinned_total / iterations);
        printf("    - pageable: %8.2f us avg\n", pageable_total / iterations);
    }

    // 10. Async memcpy with event wait
    {
        hipEvent_t event;
        HIP_CHECK(hipEventCreate(&event));

        double total = 0;
        for (int i = 0; i < iterations; i++)
        {
            double t = measure_us([&]()
                                  {
                HIP_CHECK(hipMemcpyAsync(h_pinned, d_ptr, bytes, hipMemcpyDeviceToHost, stream));
                HIP_CHECK(hipEventRecord(event, stream));
                HIP_CHECK(hipEventSynchronize(event)); });
            total += t;
        }
        printf("  Async D2H + event sync:         %8.2f us avg\n", total / iterations);

        HIP_CHECK(hipEventDestroy(event));
    }

    // Cleanup
    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK(hipFree(d_ptr));
    HIP_CHECK(hipHostFree(h_pinned));
    free(h_pageable);
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
    printf("║                    HIP TRANSFER OVERHEAD MICROBENCHMARK                              ║\n");
    printf("╠══════════════════════════════════════════════════════════════════════════════════════╣\n");
    printf("║ Device: %s (device %d)\n", props.name, device);
    printf("╚══════════════════════════════════════════════════════════════════════════════════════╝\n");

    // Test various sizes typical in Llaminar
    std::vector<size_t> sizes = {
        3584,            // Small norm weight (~3.5 KB)
        64 * 1024,       // 64 KB
        451584,          // ~440 KB (from logs)
        2 * 1024 * 1024, // 2 MB
        14680064,        // ~14 MB (common in logs)
        79691776,        // ~76 MB (large tensor from logs)
    };

    for (size_t bytes : sizes)
    {
        benchmark_transfer_overhead(device, bytes);
    }

    printf("\nBenchmark complete.\n");
    return 0;
}
