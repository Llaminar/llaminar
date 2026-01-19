/**
 * @file hip_async_writeback_benchmark.cpp
 * @brief Benchmark async output writeback strategies for LLM inference
 *
 * The problem: After a GEMM kernel, we want output data on host for:
 * - Verification (NaN checks)
 * - Snapshot capture (parity tests)
 * - Debugging
 *
 * Strategies tested:
 * 1. Sync: Wait for GEMM, then memcpy D2H (current approach - SLOW)
 * 2. Async stream: GEMM on compute stream, D2H on copy stream (better)
 * 3. Zero-copy: GEMM writes directly to mapped host memory (trades HBM BW for PCIe)
 * 4. Deferred: Don't sync immediately, batch syncs later
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

// Simulate a compute-intensive kernel (like GEMM)
// Uses shared memory and does many FMAs to simulate real workload
__global__ void fake_gemm_kernel(const float *A, const float *B, float *C, int M, int N, int K)
{
    __shared__ float As[16][16];
    __shared__ float Bs[16][16];

    int row = blockIdx.y * 16 + threadIdx.y;
    int col = blockIdx.x * 16 + threadIdx.x;

    float sum = 0.0f;

    for (int tile = 0; tile < (K + 15) / 16; tile++)
    {
        // Load tiles
        if (row < M && tile * 16 + threadIdx.x < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + tile * 16 + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && tile * 16 + threadIdx.y < K)
            Bs[threadIdx.y][threadIdx.x] = B[(tile * 16 + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

// Compute partial dot product
#pragma unroll
        for (int k = 0; k < 16; k++)
        {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N)
    {
        C[row * N + col] = sum;
    }
}

// Kernel that writes directly to mapped host memory
__global__ void fake_gemm_to_host_kernel(const float *A, const float *B, float *__restrict__ C_host, int M, int N, int K)
{
    __shared__ float As[16][16];
    __shared__ float Bs[16][16];

    int row = blockIdx.y * 16 + threadIdx.y;
    int col = blockIdx.x * 16 + threadIdx.x;

    float sum = 0.0f;

    for (int tile = 0; tile < (K + 15) / 16; tile++)
    {
        if (row < M && tile * 16 + threadIdx.x < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + tile * 16 + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && tile * 16 + threadIdx.y < K)
            Bs[threadIdx.y][threadIdx.x] = B[(tile * 16 + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

#pragma unroll
        for (int k = 0; k < 16; k++)
        {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N)
    {
        // Write directly to host memory via PCIe!
        C_host[row * N + col] = sum;
    }
}

struct BenchResult
{
    double compute_ms;   // Time for GEMM kernel
    double transfer_ms;  // Time for D2H transfer (or included in compute for zero-copy)
    double sync_ms;      // Time waiting for sync
    double total_ms;     // Total wall time
    bool data_available; // Is data available on host after total_ms?
};

// Strategy 1: Sync compute, then sync memcpy (current approach)
BenchResult benchmark_sync_compute_sync_memcpy(float *d_A, float *d_B, float *d_C, float *h_C,
                                               int M, int N, int K, size_t output_bytes)
{
    BenchResult r = {};

    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);

    auto t0 = std::chrono::high_resolution_clock::now();

    // Launch GEMM
    fake_gemm_kernel<<<blocks, threads>>>(d_A, d_B, d_C, M, N, K);

    auto t1 = std::chrono::high_resolution_clock::now();

    // Wait for GEMM to complete
    HIP_CHECK(hipDeviceSynchronize());

    auto t2 = std::chrono::high_resolution_clock::now();

    // Copy result to host
    HIP_CHECK(hipMemcpy(h_C, d_C, output_bytes, hipMemcpyDeviceToHost));

    auto t3 = std::chrono::high_resolution_clock::now();

    r.compute_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    r.sync_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    r.transfer_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    r.total_ms = std::chrono::duration<double, std::milli>(t3 - t0).count();
    r.data_available = true;

    return r;
}

// Strategy 2: Async compute, async memcpy on separate stream, sync at end
BenchResult benchmark_async_streams(float *d_A, float *d_B, float *d_C, float *h_C,
                                    int M, int N, int K, size_t output_bytes,
                                    hipStream_t compute_stream, hipStream_t copy_stream,
                                    hipEvent_t compute_done)
{
    BenchResult r = {};

    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);

    auto t0 = std::chrono::high_resolution_clock::now();

    // Launch GEMM on compute stream
    fake_gemm_kernel<<<blocks, threads, 0, compute_stream>>>(d_A, d_B, d_C, M, N, K);

    // Record when compute is done
    HIP_CHECK(hipEventRecord(compute_done, compute_stream));

    auto t1 = std::chrono::high_resolution_clock::now();

    // Make copy stream wait for compute
    HIP_CHECK(hipStreamWaitEvent(copy_stream, compute_done, 0));

    // Start async copy (will execute as soon as GEMM finishes)
    HIP_CHECK(hipMemcpyAsync(h_C, d_C, output_bytes, hipMemcpyDeviceToHost, copy_stream));

    auto t2 = std::chrono::high_resolution_clock::now();

    // Now we sync - this waits for both compute and copy
    HIP_CHECK(hipStreamSynchronize(copy_stream));

    auto t3 = std::chrono::high_resolution_clock::now();

    r.compute_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    r.transfer_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    r.sync_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    r.total_ms = std::chrono::duration<double, std::milli>(t3 - t0).count();
    r.data_available = true;

    return r;
}

// Strategy 3: Zero-copy - GEMM writes directly to mapped host memory
BenchResult benchmark_zero_copy(float *d_A, float *d_B, float *h_C_mapped, float *d_C_mapped_ptr,
                                int M, int N, int K)
{
    BenchResult r = {};

    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);

    auto t0 = std::chrono::high_resolution_clock::now();

    // Launch GEMM that writes directly to host
    fake_gemm_to_host_kernel<<<blocks, threads>>>(d_A, d_B, d_C_mapped_ptr, M, N, K);

    auto t1 = std::chrono::high_resolution_clock::now();

    // Wait for kernel to complete (data is then available on host)
    HIP_CHECK(hipDeviceSynchronize());

    auto t2 = std::chrono::high_resolution_clock::now();

    r.compute_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    r.sync_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    r.transfer_ms = 0; // Transfer happens during compute!
    r.total_ms = std::chrono::duration<double, std::milli>(t2 - t0).count();
    r.data_available = true;

    return r;
}

// Strategy 4: Deferred - Launch kernel, return immediately (data NOT available yet)
BenchResult benchmark_deferred(float *d_A, float *d_B, float *d_C,
                               int M, int N, int K, hipStream_t stream)
{
    BenchResult r = {};

    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);

    auto t0 = std::chrono::high_resolution_clock::now();

    // Launch GEMM and return immediately - no sync!
    fake_gemm_kernel<<<blocks, threads, 0, stream>>>(d_A, d_B, d_C, M, N, K);

    auto t1 = std::chrono::high_resolution_clock::now();

    r.compute_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    r.sync_ms = 0;
    r.transfer_ms = 0;
    r.total_ms = r.compute_ms;
    r.data_available = false; // Data NOT available on host!

    return r;
}

void print_result(const char *name, const BenchResult &r)
{
    printf("  %-25s: total=%7.2f ms (compute=%5.2f sync=%7.2f xfer=%5.2f) data=%s\n",
           name, r.total_ms, r.compute_ms, r.sync_ms, r.transfer_ms,
           r.data_available ? "YES" : "NO");
}

int main(int argc, char **argv)
{
    int device = 0;
    if (argc > 1)
        device = atoi(argv[1]);

    HIP_CHECK(hipSetDevice(device));

    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, device));
    printf("Device: %s\n\n", props.name);

    // GEMM dimensions similar to real LLM inference
    // M=128 (batch*seq), N=4096 (hidden), K=4096 (hidden)
    struct TestCase
    {
        int M, N, K;
        const char *name;
    };

    std::vector<TestCase> cases = {
        {9, 896, 896, "Prefill embed (9x896x896)"},
        {9, 4864, 896, "Prefill FFN gate (9x4864x896)"},
        {128, 4096, 4096, "Prefill 128x4096x4096"},
        {512, 4096, 4096, "Prefill 512x4096x4096"},
    };

    int iterations = 5;

    for (const auto &tc : cases)
    {
        int M = tc.M, N = tc.N, K = tc.K;
        size_t A_bytes = M * K * sizeof(float);
        size_t B_bytes = K * N * sizeof(float);
        size_t C_bytes = M * N * sizeof(float);

        printf("=== %s (output: %.2f MB) ===\n", tc.name, C_bytes / (1024.0 * 1024.0));

        // Allocate device memory
        float *d_A, *d_B, *d_C;
        HIP_CHECK(hipMalloc(&d_A, A_bytes));
        HIP_CHECK(hipMalloc(&d_B, B_bytes));
        HIP_CHECK(hipMalloc(&d_C, C_bytes));

        // Allocate pinned host memory
        float *h_C;
        HIP_CHECK(hipHostMalloc(&h_C, C_bytes, hipHostMallocDefault));

        // Allocate mapped host memory for zero-copy
        float *h_C_mapped;
        float *d_C_mapped_ptr;
        HIP_CHECK(hipHostMalloc(&h_C_mapped, C_bytes, hipHostMallocMapped));
        HIP_CHECK(hipHostGetDevicePointer((void **)&d_C_mapped_ptr, h_C_mapped, 0));

        // Initialize with random data
        std::vector<float> h_A(M * K), h_B(K * N);
        for (int i = 0; i < M * K; i++)
            h_A[i] = (float)(rand() % 100) / 100.0f;
        for (int i = 0; i < K * N; i++)
            h_B[i] = (float)(rand() % 100) / 100.0f;
        HIP_CHECK(hipMemcpy(d_A, h_A.data(), A_bytes, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_B, h_B.data(), B_bytes, hipMemcpyHostToDevice));

        // Create streams and events
        hipStream_t compute_stream, copy_stream;
        hipEvent_t compute_done;
        HIP_CHECK(hipStreamCreate(&compute_stream));
        HIP_CHECK(hipStreamCreate(&copy_stream));
        HIP_CHECK(hipEventCreate(&compute_done));

        // Warmup
        dim3 threads(16, 16);
        dim3 blocks((N + 15) / 16, (M + 15) / 16);
        fake_gemm_kernel<<<blocks, threads>>>(d_A, d_B, d_C, M, N, K);
        HIP_CHECK(hipDeviceSynchronize());

        // Run benchmarks
        BenchResult r1 = {}, r2 = {}, r3 = {}, r4 = {};

        for (int i = 0; i < iterations; i++)
        {
            auto t = benchmark_sync_compute_sync_memcpy(d_A, d_B, d_C, h_C, M, N, K, C_bytes);
            r1.total_ms += t.total_ms;
            r1.compute_ms += t.compute_ms;
            r1.sync_ms += t.sync_ms;
            r1.transfer_ms += t.transfer_ms;
        }
        r1.total_ms /= iterations;
        r1.compute_ms /= iterations;
        r1.sync_ms /= iterations;
        r1.transfer_ms /= iterations;
        r1.data_available = true;

        for (int i = 0; i < iterations; i++)
        {
            auto t = benchmark_async_streams(d_A, d_B, d_C, h_C, M, N, K, C_bytes,
                                             compute_stream, copy_stream, compute_done);
            r2.total_ms += t.total_ms;
            r2.compute_ms += t.compute_ms;
            r2.sync_ms += t.sync_ms;
            r2.transfer_ms += t.transfer_ms;
        }
        r2.total_ms /= iterations;
        r2.compute_ms /= iterations;
        r2.sync_ms /= iterations;
        r2.transfer_ms /= iterations;
        r2.data_available = true;

        for (int i = 0; i < iterations; i++)
        {
            auto t = benchmark_zero_copy(d_A, d_B, h_C_mapped, d_C_mapped_ptr, M, N, K);
            r3.total_ms += t.total_ms;
            r3.compute_ms += t.compute_ms;
            r3.sync_ms += t.sync_ms;
        }
        r3.total_ms /= iterations;
        r3.compute_ms /= iterations;
        r3.sync_ms /= iterations;
        r3.data_available = true;

        for (int i = 0; i < iterations; i++)
        {
            auto t = benchmark_deferred(d_A, d_B, d_C, M, N, K, compute_stream);
            r4.total_ms += t.total_ms;
            r4.compute_ms += t.compute_ms;
        }
        r4.total_ms /= iterations;
        r4.compute_ms /= iterations;
        r4.data_available = false;
        // Sync to ensure kernel is done before next benchmark
        HIP_CHECK(hipStreamSynchronize(compute_stream));

        print_result("1. Sync compute+memcpy", r1);
        print_result("2. Async streams", r2);
        print_result("3. Zero-copy to host", r3);
        print_result("4. Deferred (no sync)", r4);

        printf("  → Zero-copy overhead vs device mem: %.1fx slower\n", r3.total_ms / r1.sync_ms);
        printf("  → Deferred saves: %.2f ms per stage\n", r1.total_ms - r4.total_ms);
        printf("\n");

        // Cleanup
        HIP_CHECK(hipEventDestroy(compute_done));
        HIP_CHECK(hipStreamDestroy(compute_stream));
        HIP_CHECK(hipStreamDestroy(copy_stream));
        HIP_CHECK(hipHostFree(h_C));
        HIP_CHECK(hipHostFree(h_C_mapped));
        HIP_CHECK(hipFree(d_A));
        HIP_CHECK(hipFree(d_B));
        HIP_CHECK(hipFree(d_C));
    }

    printf("=== Conclusions ===\n");
    printf("1. Sync compute+memcpy: Current approach. Total = kernel_time + memcpy_time\n");
    printf("2. Async streams: Overlaps next kernel with D2H copy. Slightly better.\n");
    printf("3. Zero-copy: GEMM writes to host via PCIe. MUCH SLOWER due to PCIe bandwidth limit.\n");
    printf("4. Deferred: Just launch kernel, don't wait. FASTEST but data not immediately available.\n");
    printf("\n");
    printf("Recommendation: Use DEFERRED mode during inference, sync only when data actually needed.\n");

    return 0;
}
