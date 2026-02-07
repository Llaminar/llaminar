/**
 * @file Perf__KparCeiling.hip
 * @brief Round 2b: Find the ceiling — test KB=48, 64, 96, 128
 *
 * Round 2 showed KB=32 (896 wavefronts) is dramatically better than KB=16.
 * Test whether KB=48-128 can push further, or if diminishing returns set in.
 */

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <functional>
#include <cstring>

#define HIP_CHECK(cmd) do {                                                    \
    hipError_t err = cmd;                                                      \
    if (err != hipSuccess) {                                                   \
        fprintf(stderr, "HIP error %s at %s:%d\n",                            \
                hipGetErrorString(err), __FILE__, __LINE__);                   \
        exit(1);                                                               \
    }                                                                          \
} while (0)

static constexpr double MI50_PEAK_BW = 1000.0;

// Multi-block kpar with atomicAdd
template <int TILE_N, int COLS_PER_THREAD>
__global__ void kpar_atomic(
    const int8_t* __restrict__ A, const int8_t* __restrict__ B,
    int32_t* __restrict__ out, int N, int K, int kblocks)
{
    const int CPT = COLS_PER_THREAD;
    const int n_base = blockIdx.x * TILE_N + threadIdx.x * CPT;
    const bool active = (n_base < N);
    const int k_block = blockIdx.y;

    const int k_per = (K + kblocks - 1) / kblocks;
    const int k_start = k_block * k_per;
    const int k_end = min(k_start + k_per, K);

    int32_t acc[COLS_PER_THREAD];
    for (int c = 0; c < CPT; ++c) acc[c] = 0;

    if (active) {
        const int kg_start = k_start / 4;
        const int kg_end = k_end / 4;
        for (int kg = kg_start; kg < kg_end; ++kg) {
            const int kb = kg * 4;
            const int32_t a0 = A[kb], a1 = A[kb+1], a2 = A[kb+2], a3 = A[kb+3];
            #pragma unroll
            for (int c = 0; c < CPT; ++c) {
                const uint32_t p = *reinterpret_cast<const uint32_t*>(
                    B + (static_cast<size_t>(kg) * N + n_base + c) * 4);
                acc[c] += a0 * (int32_t)(int8_t)(p & 0xFF)
                        + a1 * (int32_t)(int8_t)((p >> 8) & 0xFF)
                        + a2 * (int32_t)(int8_t)((p >> 16) & 0xFF)
                        + a3 * (int32_t)(int8_t)((p >> 24) & 0xFF);
            }
        }
        for (int c = 0; c < CPT; ++c) {
            atomicAdd(&out[n_base + c], acc[c]);
        }
    }
}

// Two-pass: compute partials
template <int TILE_N, int COLS_PER_THREAD>
__global__ void kpar_2p_compute(
    const int8_t* __restrict__ A, const int8_t* __restrict__ B,
    int32_t* __restrict__ partials, int N, int K, int kblocks)
{
    const int CPT = COLS_PER_THREAD;
    const int n_base = blockIdx.x * TILE_N + threadIdx.x * CPT;
    const bool active = (n_base < N);
    const int k_block = blockIdx.y;

    const int k_per = (K + kblocks - 1) / kblocks;
    const int k_start = k_block * k_per;
    const int k_end = min(k_start + k_per, K);

    int32_t acc[COLS_PER_THREAD];
    for (int c = 0; c < CPT; ++c) acc[c] = 0;

    if (active) {
        const int kg_start = k_start / 4;
        const int kg_end = k_end / 4;
        for (int kg = kg_start; kg < kg_end; ++kg) {
            const int kb = kg * 4;
            const int32_t a0 = A[kb], a1 = A[kb+1], a2 = A[kb+2], a3 = A[kb+3];
            #pragma unroll
            for (int c = 0; c < CPT; ++c) {
                const uint32_t p = *reinterpret_cast<const uint32_t*>(
                    B + (static_cast<size_t>(kg) * N + n_base + c) * 4);
                acc[c] += a0 * (int32_t)(int8_t)(p & 0xFF)
                        + a1 * (int32_t)(int8_t)((p >> 8) & 0xFF)
                        + a2 * (int32_t)(int8_t)((p >> 16) & 0xFF)
                        + a3 * (int32_t)(int8_t)((p >> 24) & 0xFF);
            }
        }
        for (int c = 0; c < CPT; ++c)
            partials[static_cast<size_t>(k_block) * N + n_base + c] = acc[c];
    }
}

// Two-pass: reduce
__global__ void kpar_2p_reduce(
    const int32_t* __restrict__ partials,
    int32_t* __restrict__ out, int N, int kblocks)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    int32_t sum = 0;
    for (int kb = 0; kb < kblocks; ++kb)
        sum += partials[static_cast<size_t>(kb) * N + n];
    out[n] = sum;
}

struct Result {
    const char* name;
    double us;
    double bw;
    double pct;
    int wavefronts;
};

static double bench(std::function<void()> fn, int warmup, int iters) {
    for (int i = 0; i < warmup; ++i) fn();
    HIP_CHECK(hipDeviceSynchronize());
    hipEvent_t t0, t1;
    HIP_CHECK(hipEventCreate(&t0)); HIP_CHECK(hipEventCreate(&t1));
    HIP_CHECK(hipEventRecord(t0));
    for (int i = 0; i < iters; ++i) fn();
    HIP_CHECK(hipEventRecord(t1));
    HIP_CHECK(hipEventSynchronize(t1));
    float ms = 0;
    HIP_CHECK(hipEventElapsedTime(&ms, t0, t1));
    HIP_CHECK(hipEventDestroy(t0)); HIP_CHECK(hipEventDestroy(t1));
    return (double)ms * 1000.0 / iters;
}

static void pack_vnni(const int8_t* src, int8_t* dst, int N, int K) {
    for (int kg = 0; kg < K/4; ++kg)
        for (int n = 0; n < N; ++n)
            for (int i = 0; i < 4; ++i)
                dst[(static_cast<size_t>(kg) * N + n) * 4 + i] = src[(kg*4+i)*N + n];
}

static void run_shape(const char* name, int N, int K, int iters)
{
    size_t sz = (size_t)N * K;
    double weight_mb = sz / 1e6;
    double theory_us = weight_mb / (MI50_PEAK_BW / 1000.0);

    printf("\n╔═══════════════════════════════════════════════════════════════════════════════════╗\n");
    printf("║  %-12s  N=%-6d K=%-6d  Weight=%.1fMB  Theory=%.1fus                      ║\n",
           name, N, K, weight_mb, theory_us);
    printf("╠═══════════════════════════════════════════════════════════════════════════════════╣\n");

    std::vector<int8_t> h_A(K), h_B(sz);
    for (int i = 0; i < K; ++i) h_A[i] = (i % 7) - 3;
    for (size_t i = 0; i < sz; ++i) h_B[i] = (i % 11) - 5;

    std::vector<int8_t> h_vnni(sz);
    pack_vnni(h_B.data(), h_vnni.data(), N, K);

    int8_t *d_A, *d_vnni;
    int32_t *d_out;
    HIP_CHECK(hipMalloc(&d_A, K));
    HIP_CHECK(hipMalloc(&d_vnni, sz));
    HIP_CHECK(hipMalloc(&d_out, N * sizeof(int32_t)));
    HIP_CHECK(hipMemcpy(d_A, h_A.data(), K, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_vnni, h_vnni.data(), sz, hipMemcpyHostToDevice));

    const int W = 20;
    std::vector<Result> results;

    auto add = [&](const char* n, double us, int wf) {
        double bw = weight_mb / (us / 1e6) / 1e3;
        double pct = bw / MI50_PEAK_BW * 100.0;
        results.push_back({n, us, bw, pct, wf});
    };

    constexpr int TN = 128;
    constexpr int CPT = 4;
    int bx = TN / CPT;
    int grid_n = (N + TN - 1) / TN;

    for (int kb : {16, 24, 32, 48, 64, 96, 128}) {
        // Check K is divisible enough
        if (K % (kb * 4) != 0 && K < kb * 4) continue;

        int total_blocks = grid_n * kb;
        int wf = total_blocks * ((bx + 63) / 64);

        // Atomic
        {
            char buf[64];
            snprintf(buf, sizeof(buf), "atomic KB=%d", kb);
            add(strdup(buf),
                bench([&]{
                    HIP_CHECK(hipMemset(d_out, 0, N * sizeof(int32_t)));
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(kpar_atomic<TN, CPT>),
                        dim3(grid_n, kb), dim3(bx), 0, 0,
                        d_A, d_vnni, d_out, N, K, kb);
                }, W, iters), wf);
        }

        // Two-pass
        {
            int32_t* d_partials;
            HIP_CHECK(hipMalloc(&d_partials, (size_t)kb * N * sizeof(int32_t)));

            char buf[64];
            snprintf(buf, sizeof(buf), "2pass KB=%d", kb);
            add(strdup(buf),
                bench([&]{
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(kpar_2p_compute<TN, CPT>),
                        dim3(grid_n, kb), dim3(bx), 0, 0,
                        d_A, d_vnni, d_partials, N, K, kb);
                    int rblk = 256, rgrid = (N + rblk - 1) / rblk;
                    hipLaunchKernelGGL(kpar_2p_reduce,
                        dim3(rgrid), dim3(rblk), 0, 0,
                        d_partials, d_out, N, kb);
                }, W, iters), wf);

            HIP_CHECK(hipFree(d_partials));
        }
    }

    // Sort and print
    std::sort(results.begin(), results.end(), [](auto& a, auto& b){ return a.us < b.us; });

    printf("║  %-30s %8s %9s %7s %6s ║\n", "Config", "Time(us)", "BW(GB/s)", "%Peak", "Waves");
    printf("║  %-30s %8s %9s %7s %6s ║\n", "──────────────────────────────", "────────", "─────────", "───────", "──────");
    for (auto& r : results) {
        const char* star = (r.us <= results[0].us * 1.05) ? " ★" : "";
        printf("║  %-30s %8.1f %9.1f %6.1f%% %6d%s ║\n",
               r.name, r.us, r.bw, r.pct, r.wavefronts, star);
    }
    printf("║  %-30s %8.1f %9s %6s%% %6s ║\n", "THEORETICAL MINIMUM", theory_us, "1000.0", "100.0", "-");
    printf("╚═══════════════════════════════════════════════════════════════════════════════════╝\n");

    HIP_CHECK(hipFree(d_A)); HIP_CHECK(hipFree(d_vnni)); HIP_CHECK(hipFree(d_out));
}

int main(int argc, char** argv)
{
    int dev = argc > 1 ? atoi(argv[1]) : 0;
    HIP_CHECK(hipSetDevice(dev));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, dev));
    printf("Device: %s | CUs: %d | Mem: %.1f GB\n",
           prop.name, prop.multiProcessorCount, prop.totalGlobalMem / 1e9);

    const int iters = 200;
    int shape = argc > 2 ? atoi(argv[2]) : -1;

    if (shape == -1 || shape == 0) run_shape("Q/Wo",     3584,  3584, iters);
    if (shape == -1 || shape == 1) run_shape("FFN Down", 3584, 18944, iters);

    return 0;
}
