/**
 * @file Perf__GemvTemplateAB.hip
 * @brief A/B benchmark: old kernel template vs v2 kernel template at same KB
 *
 * Tests the hypothesis that the v2 kernel template is slower than the old one.
 * For each 7B grid_kpar shape (QKV, Wo, FFN Down), benchmarks:
 *   1. Old template with legacy KB (production legacy path)
 *   2. V2 template fast-path with same legacy KB (isolates template effect)
 *   3. V2 template fast-path with v2 KB (current v2 path)
 *   4. V2 template with various KB (find sweet spot)
 */

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>
#include <algorithm>
#include <random>
#include <string>
#include <numeric>

// ============================================================================
// OLD kernel template (exact copy from production)
// ============================================================================
template <int TILE_N, int CPT>
__global__ void gemv_old_kernel(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B_int8_vnni,
    int32_t* __restrict__ d_C_int32,
    int N, int K, int kblocks)
{
    const int n_base = blockIdx.x * TILE_N + threadIdx.x * CPT;
    const bool active = (n_base < N);
    const int k_block = blockIdx.y;

    const int k_per_raw = (K + kblocks - 1) / kblocks;
    const int k_per = ((k_per_raw + 3) / 4) * 4;
    const int k_start = k_block * k_per;
    const int k_end = min(k_start + k_per, K);

    int32_t acc[CPT];
    #pragma unroll
    for (int c = 0; c < CPT; ++c) acc[c] = 0;

    if (active)
    {
        const int kg_start = k_start / 4;
        const int kg_end = k_end / 4;
        const int32_t* __restrict__ b_row =
            reinterpret_cast<const int32_t*>(d_B_int8_vnni) + n_base;
        const size_t b_stride = static_cast<size_t>(N);

        #pragma unroll 4
        for (int kg = kg_start; kg < kg_end; ++kg)
        {
            const int32_t a_packed = *reinterpret_cast<const int32_t*>(d_A_int8 + kg * 4);
            if constexpr (CPT == 2)
            {
                const int2 b_pair = *reinterpret_cast<const int2*>(
                    b_row + kg * b_stride);
                acc[0] = __builtin_amdgcn_sdot4(a_packed, b_pair.x, acc[0], false);
                acc[1] = __builtin_amdgcn_sdot4(a_packed, b_pair.y, acc[1], false);
            }
            else
            {
                #pragma unroll
                for (int c = 0; c < CPT; ++c)
                {
                    const int32_t b_packed = *(b_row + kg * b_stride + c);
                    acc[c] = __builtin_amdgcn_sdot4(a_packed, b_packed, acc[c], false);
                }
            }
        }

        #pragma unroll
        for (int c = 0; c < CPT; ++c)
        {
            atomicAdd(&d_C_int32[n_base + c], acc[c]);
        }
    }
}

// ============================================================================
// V2 kernel template fast-path (WAVES=1, PERSISTENT=1, Atomic)
// Compile-time KB and UNROLL
// ============================================================================
template <int TILE_N, int CPT, int UNROLL, int KB>
__global__ void gemv_v2_fast_kernel(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B_int8_vnni,
    int32_t* __restrict__ d_C_int32,
    int N, int K, int kblocks_runtime)
{
    constexpr int THREADS_PER_WAVE = TILE_N / CPT;
    static_assert(THREADS_PER_WAVE <= 64, "THREADS_PER_WAVE must be <= 64");

    const int lane_n = threadIdx.x;
    const int kblocks = (KB > 0) ? KB : kblocks_runtime;

    const int k_per_raw = (K + kblocks - 1) / kblocks;
    const int k_per = ((k_per_raw + 3) / 4) * 4;
    const int k_block = blockIdx.y;
    const int k_start = k_block * k_per;
    const int k_end = min(k_start + k_per, K);

    const int n_base = blockIdx.x * TILE_N + lane_n * CPT;
    const bool active = (n_base < N);

    int32_t acc[CPT];
    #pragma unroll
    for (int c = 0; c < CPT; ++c) acc[c] = 0;

    if (active)
    {
        const int kg_start = k_start / 4;
        const int kg_end = k_end / 4;
        const int32_t* __restrict__ b_row =
            reinterpret_cast<const int32_t*>(d_B_int8_vnni) + n_base;
        const size_t b_stride = static_cast<size_t>(N);

        if constexpr (UNROLL == 4) {
            // Match old kernel: #pragma unroll 4
            #pragma unroll 4
            for (int kg = kg_start; kg < kg_end; ++kg)
            {
                const int32_t a_packed = *reinterpret_cast<const int32_t*>(d_A_int8 + kg * 4);
                if constexpr (CPT == 2)
                {
                    const int2 b_pair = *reinterpret_cast<const int2*>(
                        b_row + kg * b_stride);
                    acc[0] = __builtin_amdgcn_sdot4(a_packed, b_pair.x, acc[0], false);
                    acc[1] = __builtin_amdgcn_sdot4(a_packed, b_pair.y, acc[1], false);
                }
                else
                {
                    #pragma unroll
                    for (int c = 0; c < CPT; ++c)
                    {
                        const int32_t b_packed = *(b_row + kg * b_stride + c);
                        acc[c] = __builtin_amdgcn_sdot4(a_packed, b_packed, acc[c], false);
                    }
                }
            }
        } else {
            for (int kg = kg_start; kg < kg_end; kg += UNROLL) {
                #pragma unroll
                for (int u = 0; u < UNROLL; ++u) {
                    const int kg_u = kg + u;
                    if (kg_u >= kg_end) break;
                    const int32_t a_packed = *reinterpret_cast<const int32_t*>(d_A_int8 + kg_u * 4);
                    if constexpr (CPT == 2) {
                        const int2 b_pair = *reinterpret_cast<const int2*>(
                            b_row + kg_u * b_stride);
                        acc[0] = __builtin_amdgcn_sdot4(a_packed, b_pair.x, acc[0], false);
                        acc[1] = __builtin_amdgcn_sdot4(a_packed, b_pair.y, acc[1], false);
                    } else {
                        #pragma unroll
                        for (int c = 0; c < CPT; ++c) {
                            const int32_t b_packed = *(b_row + kg_u * b_stride + c);
                            acc[c] = __builtin_amdgcn_sdot4(a_packed, b_packed, acc[c], false);
                        }
                    }
                }
            }
        }

        // Store
        #pragma unroll
        for (int c = 0; c < CPT; ++c)
        {
            if (kblocks == 1)
                d_C_int32[n_base + c] = acc[c];
            else
                atomicAdd(&d_C_int32[n_base + c], acc[c]);
        }
    }
}

// ============================================================================
// Benchmarking infrastructure
// ============================================================================
struct BenchResult {
    double min_us;
    double mean_us;
    double gbps;
};

static BenchResult benchmarkOld(const int8_t* A, const int8_t* B, int32_t* C,
                                 int N, int K, int KB, int warmup, int iters)
{
    constexpr int TN = 128;
    constexpr int CPT = 2;
    const int grid_n = (N + TN - 1) / TN;

    // Warmup
    for (int i = 0; i < warmup; ++i) {
        hipMemsetAsync(C, 0, N * sizeof(int32_t), 0);
        hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_old_kernel<TN, CPT>),
            dim3(grid_n, KB), dim3(TN / CPT), 0, 0,
            A, B, C, N, K, KB);
    }
    hipDeviceSynchronize();

    std::vector<double> times(iters);
    for (int i = 0; i < iters; ++i) {
        hipMemsetAsync(C, 0, N * sizeof(int32_t), 0);
        hipDeviceSynchronize();

        auto t0 = std::chrono::high_resolution_clock::now();
        hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_old_kernel<TN, CPT>),
            dim3(grid_n, KB), dim3(TN / CPT), 0, 0,
            A, B, C, N, K, KB);
        hipDeviceSynchronize();
        auto t1 = std::chrono::high_resolution_clock::now();
        times[i] = std::chrono::duration<double, std::micro>(t1 - t0).count();
    }

    double min_t = *std::min_element(times.begin(), times.end());
    double mean_t = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    double bytes = static_cast<double>(N) * K;  // INT8 weight bytes
    double gbps = bytes / (min_t * 1e-6) / 1e9;

    return {min_t, mean_t, gbps};
}

template <int UNROLL, int KB_TEMPLATE>
static BenchResult benchmarkV2(const int8_t* A, const int8_t* B, int32_t* C,
                                int N, int K, int kb_runtime, int warmup, int iters)
{
    constexpr int TN = 128;
    constexpr int CPT = 2;
    const int grid_n = (N + TN - 1) / TN;
    const int kb = (KB_TEMPLATE > 0) ? KB_TEMPLATE : kb_runtime;

    // Warmup
    for (int i = 0; i < warmup; ++i) {
        if (kb > 1)
            hipMemsetAsync(C, 0, N * sizeof(int32_t), 0);
        hipLaunchKernelGGL(HIP_KERNEL_NAME((gemv_v2_fast_kernel<TN, CPT, UNROLL, KB_TEMPLATE>)),
            dim3(grid_n, kb), dim3(TN / CPT), 0, 0,
            A, B, C, N, K, kb_runtime);
    }
    hipDeviceSynchronize();

    std::vector<double> times(iters);
    for (int i = 0; i < iters; ++i) {
        if (kb > 1)
            hipMemsetAsync(C, 0, N * sizeof(int32_t), 0);
        hipDeviceSynchronize();

        auto t0 = std::chrono::high_resolution_clock::now();
        hipLaunchKernelGGL(HIP_KERNEL_NAME((gemv_v2_fast_kernel<TN, CPT, UNROLL, KB_TEMPLATE>)),
            dim3(grid_n, kb), dim3(TN / CPT), 0, 0,
            A, B, C, N, K, kb_runtime);
        hipDeviceSynchronize();
        auto t1 = std::chrono::high_resolution_clock::now();
        times[i] = std::chrono::duration<double, std::micro>(t1 - t0).count();
    }

    double min_t = *std::min_element(times.begin(), times.end());
    double mean_t = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    double bytes = static_cast<double>(N) * K;
    double gbps = bytes / (min_t * 1e-6) / 1e9;

    return {min_t, mean_t, gbps};
}

// V2 with runtime KB (KB_TEMPLATE=0), matching legacy code path
static BenchResult benchmarkV2RuntimeKB(const int8_t* A, const int8_t* B, int32_t* C,
                                int N, int K, int kb, int warmup, int iters)
{
    return benchmarkV2<4, 0>(A, B, C, N, K, kb, warmup, iters);
}

// Legacy KB heuristic
static int legacy_kb(int N, int K) {
    constexpr int TN = 128;
    constexpr int NUM_CUS = 60;
    const int grid_n = (N + TN - 1) / TN;
    const int k_groups = K / 4;

    if (K > 8192) {
        return std::max(4, std::min(32, K / 512));
    }
    constexpr int TARGET_KGROUPS = 64;
    constexpr int TARGET_MIN_WAVES_PER_CU = 10;
    constexpr int MIN_KGROUPS_FLOOR = 16;

    int kb_for_kgroups = std::max(2, k_groups / TARGET_KGROUPS);
    int min_blocks = TARGET_MIN_WAVES_PER_CU * NUM_CUS;
    int kb_for_occupancy = std::max(2, (min_blocks + grid_n - 1) / grid_n);
    int kb_max = std::max(2, k_groups / MIN_KGROUPS_FLOOR);
    return std::min(std::max(kb_for_kgroups, kb_for_occupancy), kb_max);
}

struct Shape {
    const char* name;
    int N;
    int K;
};

int main()
{
    int device_count = 0;
    hipGetDeviceCount(&device_count);
    if (device_count == 0) {
        fprintf(stderr, "No HIP device found\n");
        return 1;
    }

    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);
    hipSetDevice(0);
    printf("Device[0]: %s  (%d CUs)\n\n", prop.name, prop.multiProcessorCount);

    // 7B grid_kpar shapes only
    const Shape shapes[] = {
        {"QKV fused", 4608,  3584},
        {"Wo",        3584,  3584},
        {"FFN Down",  3584, 18944},
    };

    constexpr int WARMUP = 5;
    constexpr int ITERS = 50;

    // Allocate max-sized buffers
    int maxN = 0, maxK = 0;
    for (auto& s : shapes) {
        maxN = std::max(maxN, s.N);
        maxK = std::max(maxK, s.K);
    }

    std::mt19937 rng(42);
    std::vector<int8_t> h_A(maxK), h_B(static_cast<size_t>(maxK) * maxN);
    for (auto& v : h_A) v = static_cast<int8_t>(rng() % 255 - 127);
    for (auto& v : h_B) v = static_cast<int8_t>(rng() % 255 - 127);

    int8_t *d_A, *d_B;
    int32_t *d_C;
    hipMalloc(&d_A, maxK);
    hipMalloc(&d_B, static_cast<size_t>(maxK) * maxN);
    hipMalloc(&d_C, maxN * sizeof(int32_t));
    hipMemcpy(d_A, h_A.data(), maxK, hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B.data(), static_cast<size_t>(maxK) * maxN, hipMemcpyHostToDevice);

    // KB values to test
    const int kb_test[] = {8, 16, 24, 32, 48, 64, 96, 128};

    printf("╔══════════════════════════════════════════════════════════════════════════════════════════╗\n");
    printf("║  EXPERIMENT: Old vs V2 kernel template at same KB values                               ║\n");
    printf("║  TN=128, CPT=2, UNROLL=4, WAVES=1, Atomic                                             ║\n");
    printf("╚══════════════════════════════════════════════════════════════════════════════════════════╝\n\n");

    for (auto& shape : shapes) {
        int lkb = legacy_kb(shape.N, shape.K);

        printf("┌──────────────────────────────────────────────────────────────────────────────┐\n");
        printf("│ %-20s  N=%-6d K=%-6d  Legacy KB=%d\n", shape.name, shape.N, shape.K, lkb);
        printf("├──────────┬──────────────────────┬──────────────────────┬─────────────────────┤\n");
        printf("│    KB    │   Old Template (μs)  │   V2 Template (μs)   │   V2/Old Ratio      │\n");
        printf("├──────────┼──────────────────────┼──────────────────────┼─────────────────────┤\n");

        for (int kb : kb_test) {
            auto old_r = benchmarkOld(d_A, d_B, d_C, shape.N, shape.K, kb, WARMUP, ITERS);
            auto v2_r  = benchmarkV2RuntimeKB(d_A, d_B, d_C, shape.N, shape.K, kb, WARMUP, ITERS);
            double ratio = v2_r.min_us / old_r.min_us;

            const char* marker = "";
            if (kb == lkb) marker = " ← LEGACY";

            printf("│  %4d    │  %8.2f  %6.1f GB/s│  %8.2f  %6.1f GB/s│   %.3f%s\n",
                   kb, old_r.min_us, old_r.gbps, v2_r.min_us, v2_r.gbps, ratio, marker);
        }

        // Also test v2 with compile-time KB at the best sweep values
        printf("├──────────┼──────────────────────┼──────────────────────┼─────────────────────┤\n");
        printf("│ V2 compile-time KB at sweep-best configs:                                    │\n");

        // KB=32 compile-time, UNROLL=4
        {
            auto r = benchmarkV2<4, 32>(d_A, d_B, d_C, shape.N, shape.K, 0, WARMUP, ITERS);
            printf("│  KB=32ct │                      │  %8.2f  %6.1f GB/s│   (compile-time)    │\n",
                   r.min_us, r.gbps);
        }
        // KB=64 compile-time, UNROLL=4
        {
            auto r = benchmarkV2<4, 64>(d_A, d_B, d_C, shape.N, shape.K, 0, WARMUP, ITERS);
            printf("│  KB=64ct │                      │  %8.2f  %6.1f GB/s│   (compile-time)    │\n",
                   r.min_us, r.gbps);
        }
        // KB=128 compile-time, UNROLL=4
        {
            auto r = benchmarkV2<4, 128>(d_A, d_B, d_C, shape.N, shape.K, 0, WARMUP, ITERS);
            printf("│  KB=128ct│                      │  %8.2f  %6.1f GB/s│   (compile-time)    │\n",
                   r.min_us, r.gbps);
        }
        // KB=128 compile-time, UNROLL=8
        {
            auto r = benchmarkV2<8, 128>(d_A, d_B, d_C, shape.N, shape.K, 0, WARMUP, ITERS);
            printf("│  KB=128/8│                      │  %8.2f  %6.1f GB/s│  (ct,UNROLL=8)      │\n",
                   r.min_us, r.gbps);
        }
        // KB=128 compile-time, UNROLL=16
        {
            auto r = benchmarkV2<16, 128>(d_A, d_B, d_C, shape.N, shape.K, 0, WARMUP, ITERS);
            printf("│  KB=128/16│                     │  %8.2f  %6.1f GB/s│  (ct,UNROLL=16)     │\n",
                   r.min_us, r.gbps);
        }
        printf("└──────────┴──────────────────────┴──────────────────────┴─────────────────────┘\n\n");
    }

    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
    return 0;
}
