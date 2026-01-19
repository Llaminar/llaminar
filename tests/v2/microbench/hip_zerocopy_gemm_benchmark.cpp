/**
 * @file hip_zerocopy_gemm_benchmark.cpp
 * @brief Benchmark zero-copy vs traditional GEMM output handling
 *
 * Compares two approaches for getting GEMM output to host:
 * 1. Traditional: GEMM writes to device memory -> hipMemcpy D2H -> read host
 * 2. Zero-copy: GEMM writes to mapped host memory -> hipDeviceSynchronize -> read host
 *
 * The goal is to eliminate the explicit D2H memcpy overhead.
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

// Simple tiled GEMM - writes to device memory
__global__ void tiled_gemm_kernel(const float *A, const float *B, float *C, int M, int N, int K)
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
        C[row * N + col] = sum;
    }
}

void run_traditional_gemm(const float *d_A, const float *d_B, float *d_C, float *h_C,
                          int M, int N, int K, size_t C_bytes, int warmup, int iterations,
                          double &avg_kernel_ms, double &avg_memcpy_ms, double &avg_total_ms)
{
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);

    // Warmup
    for (int i = 0; i < warmup; i++)
    {
        tiled_gemm_kernel<<<blocks, threads>>>(d_A, d_B, d_C, M, N, K);
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(h_C, d_C, C_bytes, hipMemcpyDeviceToHost));
    }

    avg_kernel_ms = 0;
    avg_memcpy_ms = 0;
    avg_total_ms = 0;

    for (int i = 0; i < iterations; i++)
    {
        auto t0 = std::chrono::high_resolution_clock::now();

        tiled_gemm_kernel<<<blocks, threads>>>(d_A, d_B, d_C, M, N, K);

        auto t1 = std::chrono::high_resolution_clock::now();

        HIP_CHECK(hipDeviceSynchronize());

        auto t2 = std::chrono::high_resolution_clock::now();

        HIP_CHECK(hipMemcpy(h_C, d_C, C_bytes, hipMemcpyDeviceToHost));

        auto t3 = std::chrono::high_resolution_clock::now();

        avg_kernel_ms += std::chrono::duration<double, std::milli>(t2 - t0).count();
        avg_memcpy_ms += std::chrono::duration<double, std::milli>(t3 - t2).count();
        avg_total_ms += std::chrono::duration<double, std::milli>(t3 - t0).count();
    }

    avg_kernel_ms /= iterations;
    avg_memcpy_ms /= iterations;
    avg_total_ms /= iterations;
}

void run_zerocopy_gemm(const float *d_A, const float *d_B, float *d_C_mapped, float *h_C_mapped,
                       int M, int N, int K, int warmup, int iterations,
                       double &avg_kernel_ms, double &avg_total_ms)
{
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);

    // Warmup
    for (int i = 0; i < warmup; i++)
    {
        tiled_gemm_kernel<<<blocks, threads>>>(d_A, d_B, d_C_mapped, M, N, K);
        HIP_CHECK(hipDeviceSynchronize());
        // Data is already on host - just touch it to ensure it's there
        volatile float x = h_C_mapped[0];
        (void)x;
    }

    avg_kernel_ms = 0;
    avg_total_ms = 0;

    for (int i = 0; i < iterations; i++)
    {
        auto t0 = std::chrono::high_resolution_clock::now();

        tiled_gemm_kernel<<<blocks, threads>>>(d_A, d_B, d_C_mapped, M, N, K);

        auto t1 = std::chrono::high_resolution_clock::now();

        HIP_CHECK(hipDeviceSynchronize());

        auto t2 = std::chrono::high_resolution_clock::now();

        // Data is already on host! Just sync was needed.

        avg_kernel_ms += std::chrono::duration<double, std::milli>(t2 - t1).count();
        avg_total_ms += std::chrono::duration<double, std::milli>(t2 - t0).count();
    }

    avg_kernel_ms /= iterations;
    avg_total_ms /= iterations;
}

int main(int argc, char **argv)
{
    int device = 0;
    if (argc > 1)
        device = atoi(argv[1]);

    HIP_CHECK(hipSetDevice(device));

    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, device));
    printf("Device: %s\n", props.name);
    printf("Testing zero-copy mapped memory for GEMM outputs\n\n");

    // Test cases representing real Qwen2 shapes
    struct TestCase
    {
        int M, N, K;
        const char *name;
    };

    std::vector<TestCase> cases = {
        {1, 896, 896, "Decode QKV (1x896x896)"},
        {9, 896, 896, "Prefill QKV (9x896x896)"},
        {1, 4864, 896, "Decode FFN gate (1x4864x896)"},
        {9, 4864, 896, "Prefill FFN gate (9x4864x896)"},
        {128, 4096, 4096, "Large prefill (128x4096x4096)"},
    };

    int warmup = 3;
    int iterations = 10;

    printf("%-30s | %-25s | %-25s | Speedup\n", "Test Case", "Traditional", "Zero-Copy");
    printf("%-30s | %-25s | %-25s | -------\n", "----------", "-----------", "---------");

    for (const auto &tc : cases)
    {
        int M = tc.M, N = tc.N, K = tc.K;
        size_t A_bytes = M * K * sizeof(float);
        size_t B_bytes = K * N * sizeof(float);
        size_t C_bytes = M * N * sizeof(float);

        // Allocate inputs on device
        float *d_A, *d_B;
        HIP_CHECK(hipMalloc(&d_A, A_bytes));
        HIP_CHECK(hipMalloc(&d_B, B_bytes));

        // Traditional: output on device + host copy
        float *d_C_trad;
        float *h_C_trad;
        HIP_CHECK(hipMalloc(&d_C_trad, C_bytes));
        HIP_CHECK(hipHostMalloc(&h_C_trad, C_bytes, hipHostMallocDefault));

        // Zero-copy: output is mapped host memory
        float *h_C_mapped;
        float *d_C_mapped;
        HIP_CHECK(hipHostMalloc(&h_C_mapped, C_bytes, hipHostMallocMapped | hipHostMallocWriteCombined));
        HIP_CHECK(hipHostGetDevicePointer((void **)&d_C_mapped, h_C_mapped, 0));

        // Initialize inputs
        std::vector<float> h_A(M * K), h_B(K * N);
        for (int i = 0; i < M * K; i++)
            h_A[i] = (float)(rand() % 100) / 100.0f;
        for (int i = 0; i < K * N; i++)
            h_B[i] = (float)(rand() % 100) / 100.0f;
        HIP_CHECK(hipMemcpy(d_A, h_A.data(), A_bytes, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_B, h_B.data(), B_bytes, hipMemcpyHostToDevice));

        // Run benchmarks
        double trad_kernel_ms, trad_memcpy_ms, trad_total_ms;
        run_traditional_gemm(d_A, d_B, d_C_trad, h_C_trad, M, N, K, C_bytes,
                             warmup, iterations, trad_kernel_ms, trad_memcpy_ms, trad_total_ms);

        double zc_kernel_ms, zc_total_ms;
        run_zerocopy_gemm(d_A, d_B, d_C_mapped, h_C_mapped, M, N, K,
                          warmup, iterations, zc_kernel_ms, zc_total_ms);

        // Verify results match
        HIP_CHECK(hipDeviceSynchronize());
        float max_diff = 0.0f;
        for (int i = 0; i < M * N; i++)
        {
            float diff = std::abs(h_C_trad[i] - h_C_mapped[i]);
            if (diff > max_diff)
                max_diff = diff;
        }

        double speedup = trad_total_ms / zc_total_ms;

        printf("%-30s | kernel=%5.2f memcpy=%5.2f | kernel=%5.2f total=%6.2f | %.2fx %s\n",
               tc.name, trad_kernel_ms, trad_memcpy_ms, zc_kernel_ms, zc_total_ms,
               speedup, speedup > 1.0 ? "(faster)" : "(slower)");

        if (max_diff > 0.01f)
        {
            printf("  WARNING: Max diff = %f\n", max_diff);
        }

        // Cleanup
        HIP_CHECK(hipFree(d_A));
        HIP_CHECK(hipFree(d_B));
        HIP_CHECK(hipFree(d_C_trad));
        HIP_CHECK(hipHostFree(h_C_trad));
        HIP_CHECK(hipHostFree(h_C_mapped));
    }

    printf("\n");
    printf("=== Analysis ===\n");
    printf("Traditional: GEMM to device mem -> hipDeviceSynchronize -> hipMemcpy D2H\n");
    printf("Zero-copy:   GEMM to mapped host mem -> hipDeviceSynchronize (data already on host!)\n");
    printf("\n");
    printf("Zero-copy eliminates the D2H memcpy but may be slower if kernel is memory-bound\n");
    printf("because writes go over PCIe (~15 GB/s) instead of HBM (~1 TB/s).\n");
    printf("\n");
    printf("For compute-bound kernels, zero-copy can be a win because:\n");
    printf("1. No explicit D2H memcpy needed\n");
    printf("2. Kernel writes overlap with computation (memory-level parallelism)\n");

    return 0;
}
