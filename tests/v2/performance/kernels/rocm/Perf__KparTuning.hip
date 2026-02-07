/**
 * @file Perf__KparTuning.hip
 * @brief Round 2: Focused KSPLIT tuning for Q/Wo and FFN Down shapes on flat VNNI.
 *
 * Round 1 findings:
 *   Gate/Up:  Tiled KN VNNI 128×128 wins at 75.6% peak (solved)
 *   Q/Wo:    Flat VNNI kpar (TN=128, KSPLIT=16) wins at 32.9% peak
 *   FFN Down: Flat VNNI kpar (TN=128, KSPLIT=16) wins at 34.5% peak
 *
 * This benchmark explores whether we can push Q/Wo and FFN Down past 35%:
 *
 * Experiment 1: KSPLIT sweep {4, 8, 16, 32} × {TN=64, TN=128, TN=256}
 *   - More KSPLIT = more wavefronts = better latency hiding
 *   - But more reduction overhead + less work per thread
 *
 * Experiment 2: Multi-block kpar (grid-level K-split with atomicAdd)
 *   - grid = (N/TN, KBLOCKS), each block handles K/KBLOCKS fraction
 *   - atomicAdd accumulation instead of smem reduction
 *   - Tests whether distributing K across blocks is better than within
 *
 * Experiment 3: Warp-shuffle reduce (KSPLIT <= 64 within one wavefront)
 *   - Use __shfl_xor_b32 instead of shared memory
 *   - Lower latency reduction
 *
 * Experiment 4: Two-pass kpar (no atomics)
 *   - Pass 1: Each block writes partials to global buffer
 *   - Pass 2: Separate kernel reduces partials
 *   - Clean pipeline, no atomic contention
 */

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
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

// ============================================================================
// Experiment 1: KSPLIT sweep — smem reduction within block
// ============================================================================
template <int TILE_N, int KSPLIT>
__global__ void kpar_smem(
    const int8_t* __restrict__ A, const int8_t* __restrict__ B,
    int32_t* __restrict__ out, int N, int K)
{
    const int n4 = blockIdx.x * TILE_N + threadIdx.x * 4;
    const bool active = (n4 < N);
    const int lane_k = threadIdx.y;

    __shared__ int8_t sA[256];
    int4 acc = {0, 0, 0, 0};

    for (int kb = 0; kb < K; kb += 256) {
        const int tk = min(256, K - kb);
        const int ltid = lane_k * blockDim.x + threadIdx.x;
        for (int i = ltid; i < tk; i += blockDim.x * KSPLIT) sA[i] = A[kb + i];
        __syncthreads();

        const int tg = tk / 4;
        const int kg0 = kb / 4;
        for (int kg = lane_k; kg < tg; kg += KSPLIT) {
            if (active) {
                const int kl = kg * 4;
                const int32_t a0 = sA[kl], a1 = sA[kl+1], a2 = sA[kl+2], a3 = sA[kl+3];
                #pragma unroll
                for (int c = 0; c < 4; ++c) {
                    const uint32_t p = *reinterpret_cast<const uint32_t*>(
                        B + (static_cast<size_t>(kg0 + kg) * N + n4 + c) * 4);
                    int32_t* a_ptr = &acc.x + c;
                    *a_ptr += a0 * (int32_t)(int8_t)(p & 0xFF)
                            + a1 * (int32_t)(int8_t)((p >> 8) & 0xFF)
                            + a2 * (int32_t)(int8_t)((p >> 16) & 0xFF)
                            + a3 * (int32_t)(int8_t)((p >> 24) & 0xFF);
                }
            }
        }
        __syncthreads();
    }

    __shared__ int4 partial[KSPLIT][TILE_N / 4];
    partial[lane_k][threadIdx.x] = acc;
    __syncthreads();

    if (lane_k == 0 && active) {
        int4 s = {0,0,0,0};
        #pragma unroll
        for (int i = 0; i < KSPLIT; ++i) {
            s.x += partial[i][threadIdx.x].x;
            s.y += partial[i][threadIdx.x].y;
            s.z += partial[i][threadIdx.x].z;
            s.w += partial[i][threadIdx.x].w;
        }
        out[n4] = s.x; out[n4+1] = s.y; out[n4+2] = s.z; out[n4+3] = s.w;
    }
}

// ============================================================================
// Experiment 2: Multi-block kpar with atomicAdd (grid-level K-split)
// ============================================================================
template <int TILE_N, int COLS_PER_THREAD>
__global__ void kpar_atomic(
    const int8_t* __restrict__ A, const int8_t* __restrict__ B,
    int32_t* __restrict__ out, int N, int K, int kblocks)
{
    const int CPT = COLS_PER_THREAD;
    const int n_base = blockIdx.x * TILE_N + threadIdx.x * CPT;
    const bool active = (n_base < N);
    const int k_block = blockIdx.y;

    // Each block processes k_start..k_end
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

        // AtomicAdd to output
        for (int c = 0; c < CPT; ++c) {
            atomicAdd(&out[n_base + c], acc[c]);
        }
    }
}

// ============================================================================
// Experiment 3: Warp-shuffle reduce (K-parallel within wavefront)
// Each thread computes partial product, then wavefront-level reduce via shfl
// ============================================================================
template <int TILE_N, int KSPLIT>
__global__ void kpar_shfl(
    const int8_t* __restrict__ A, const int8_t* __restrict__ B,
    int32_t* __restrict__ out, int N, int K)
{
    // Thread layout: blockDim.x threads, each group of KSPLIT consecutive
    // threads cooperates on one output group of 4 columns
    const int group_id = threadIdx.x / KSPLIT;      // which 4-col group
    const int lane_k = threadIdx.x % KSPLIT;        // K-split lane
    const int groups_per_block = TILE_N / 4;
    const int n4 = blockIdx.x * TILE_N + group_id * 4;
    const bool active = (group_id < groups_per_block && n4 < N);

    int4 acc = {0, 0, 0, 0};
    const int k_groups = K / 4;

    if (active) {
        for (int kg = lane_k; kg < k_groups; kg += KSPLIT) {
            const int kb = kg * 4;
            const int32_t a0 = A[kb], a1 = A[kb+1], a2 = A[kb+2], a3 = A[kb+3];
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                const uint32_t p = *reinterpret_cast<const uint32_t*>(
                    B + (static_cast<size_t>(kg) * N + n4 + c) * 4);
                int32_t* a_ptr = &acc.x + c;
                *a_ptr += a0 * (int32_t)(int8_t)(p & 0xFF)
                        + a1 * (int32_t)(int8_t)((p >> 8) & 0xFF)
                        + a2 * (int32_t)(int8_t)((p >> 16) & 0xFF)
                        + a3 * (int32_t)(int8_t)((p >> 24) & 0xFF);
            }
        }
    }

    // Warp-shuffle reduce across KSPLIT lanes
    #pragma unroll
    for (int offset = KSPLIT / 2; offset >= 1; offset >>= 1) {
        acc.x += __shfl_xor(acc.x, offset);
        acc.y += __shfl_xor(acc.y, offset);
        acc.z += __shfl_xor(acc.z, offset);
        acc.w += __shfl_xor(acc.w, offset);
    }

    // Only lane 0 of each group writes
    if (lane_k == 0 && active) {
        out[n4] = acc.x; out[n4+1] = acc.y; out[n4+2] = acc.z; out[n4+3] = acc.w;
    }
}

// ============================================================================
// Experiment 4: Two-pass kpar (no atomics, no reduction)
// Pass 1: Write partials to intermediate buffer
// Pass 2: Reduce partials
// ============================================================================
template <int TILE_N, int COLS_PER_THREAD>
__global__ void kpar_twopass_compute(
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

        // Write partials: partials[k_block * N + n_base + c]
        for (int c = 0; c < CPT; ++c) {
            partials[static_cast<size_t>(k_block) * N + n_base + c] = acc[c];
        }
    }
}

template <int KBLOCKS>
__global__ void kpar_twopass_reduce(
    const int32_t* __restrict__ partials,
    int32_t* __restrict__ out, int N)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    int32_t sum = 0;
    #pragma unroll
    for (int kb = 0; kb < KBLOCKS; ++kb) {
        sum += partials[static_cast<size_t>(kb) * N + n];
    }
    out[n] = sum;
}

// ============================================================================
// Experiment 5: "Fused multi-col kpar" - 8 cols per thread with kpar
// Higher ILP per thread, fewer threads needed
// ============================================================================
template <int TILE_N, int KSPLIT>
__global__ void kpar_smem_8col(
    const int8_t* __restrict__ A, const int8_t* __restrict__ B,
    int32_t* __restrict__ out, int N, int K)
{
    // Each thread handles 8 columns (2 × int4)
    const int n8 = blockIdx.x * TILE_N + threadIdx.x * 8;
    const bool active = (n8 + 7 < N);
    const int lane_k = threadIdx.y;

    __shared__ int8_t sA[256];
    int4 acc0 = {0,0,0,0};
    int4 acc1 = {0,0,0,0};

    for (int kb = 0; kb < K; kb += 256) {
        const int tk = min(256, K - kb);
        const int ltid = lane_k * blockDim.x + threadIdx.x;
        for (int i = ltid; i < tk; i += blockDim.x * KSPLIT) sA[i] = A[kb + i];
        __syncthreads();

        const int tg = tk / 4;
        const int kg0 = kb / 4;
        for (int kg = lane_k; kg < tg; kg += KSPLIT) {
            if (active) {
                const int kl = kg * 4;
                const int32_t a0 = sA[kl], a1 = sA[kl+1], a2 = sA[kl+2], a3 = sA[kl+3];
                const size_t base = (static_cast<size_t>(kg0 + kg) * N + n8) * 4;

                // First 4 cols
                #pragma unroll
                for (int c = 0; c < 4; ++c) {
                    const uint32_t p = *reinterpret_cast<const uint32_t*>(B + base + c * 4);
                    int32_t* a_ptr = &acc0.x + c;
                    *a_ptr += a0 * (int32_t)(int8_t)(p & 0xFF)
                            + a1 * (int32_t)(int8_t)((p >> 8) & 0xFF)
                            + a2 * (int32_t)(int8_t)((p >> 16) & 0xFF)
                            + a3 * (int32_t)(int8_t)((p >> 24) & 0xFF);
                }
                // Next 4 cols
                #pragma unroll
                for (int c = 0; c < 4; ++c) {
                    const uint32_t p = *reinterpret_cast<const uint32_t*>(B + base + (4 + c) * 4);
                    int32_t* a_ptr = &acc1.x + c;
                    *a_ptr += a0 * (int32_t)(int8_t)(p & 0xFF)
                            + a1 * (int32_t)(int8_t)((p >> 8) & 0xFF)
                            + a2 * (int32_t)(int8_t)((p >> 16) & 0xFF)
                            + a3 * (int32_t)(int8_t)((p >> 24) & 0xFF);
                }
            }
        }
        __syncthreads();
    }

    // Smem reduce
    constexpr int OUT_PER_THREAD = TILE_N / 8;
    __shared__ int4 partial0[KSPLIT][OUT_PER_THREAD];
    __shared__ int4 partial1[KSPLIT][OUT_PER_THREAD];
    partial0[lane_k][threadIdx.x] = acc0;
    partial1[lane_k][threadIdx.x] = acc1;
    __syncthreads();

    if (lane_k == 0 && active) {
        int4 s0 = {0,0,0,0}, s1 = {0,0,0,0};
        #pragma unroll
        for (int i = 0; i < KSPLIT; ++i) {
            s0.x += partial0[i][threadIdx.x].x;
            s0.y += partial0[i][threadIdx.x].y;
            s0.z += partial0[i][threadIdx.x].z;
            s0.w += partial0[i][threadIdx.x].w;
            s1.x += partial1[i][threadIdx.x].x;
            s1.y += partial1[i][threadIdx.x].y;
            s1.z += partial1[i][threadIdx.x].z;
            s1.w += partial1[i][threadIdx.x].w;
        }
        out[n8]   = s0.x; out[n8+1] = s0.y; out[n8+2] = s0.z; out[n8+3] = s0.w;
        out[n8+4] = s1.x; out[n8+5] = s1.y; out[n8+6] = s1.z; out[n8+7] = s1.w;
    }
}

// ============================================================================
// Experiment 6: Large A-tile kpar — pack more A into smem to reduce
// K-loop iterations and amortize smem load cost
// ============================================================================
template <int TILE_N, int KSPLIT, int A_TILE>
__global__ void kpar_smem_bigtile(
    const int8_t* __restrict__ A, const int8_t* __restrict__ B,
    int32_t* __restrict__ out, int N, int K)
{
    const int n4 = blockIdx.x * TILE_N + threadIdx.x * 4;
    const bool active = (n4 < N);
    const int lane_k = threadIdx.y;

    __shared__ int8_t sA[A_TILE];
    int4 acc = {0, 0, 0, 0};

    for (int kb = 0; kb < K; kb += A_TILE) {
        const int tk = min(A_TILE, K - kb);
        // Cooperative load of A tile
        const int ltid = lane_k * blockDim.x + threadIdx.x;
        const int nthreads = blockDim.x * KSPLIT;
        for (int i = ltid; i < tk; i += nthreads) sA[i] = A[kb + i];
        __syncthreads();

        const int tg = tk / 4;
        const int kg0 = kb / 4;
        for (int kg = lane_k; kg < tg; kg += KSPLIT) {
            if (active) {
                const int kl = kg * 4;
                const int32_t a0 = sA[kl], a1 = sA[kl+1], a2 = sA[kl+2], a3 = sA[kl+3];
                #pragma unroll
                for (int c = 0; c < 4; ++c) {
                    const uint32_t p = *reinterpret_cast<const uint32_t*>(
                        B + (static_cast<size_t>(kg0 + kg) * N + n4 + c) * 4);
                    int32_t* a_ptr = &acc.x + c;
                    *a_ptr += a0 * (int32_t)(int8_t)(p & 0xFF)
                            + a1 * (int32_t)(int8_t)((p >> 8) & 0xFF)
                            + a2 * (int32_t)(int8_t)((p >> 16) & 0xFF)
                            + a3 * (int32_t)(int8_t)((p >> 24) & 0xFF);
                }
            }
        }
        __syncthreads();
    }

    __shared__ int4 partial[KSPLIT][TILE_N / 4];
    partial[lane_k][threadIdx.x] = acc;
    __syncthreads();

    if (lane_k == 0 && active) {
        int4 s = {0,0,0,0};
        #pragma unroll
        for (int i = 0; i < KSPLIT; ++i) {
            s.x += partial[i][threadIdx.x].x;
            s.y += partial[i][threadIdx.x].y;
            s.z += partial[i][threadIdx.x].z;
            s.w += partial[i][threadIdx.x].w;
        }
        out[n4] = s.x; out[n4+1] = s.y; out[n4+2] = s.z; out[n4+3] = s.w;
    }
}

// ============================================================================
// Benchmark harness
// ============================================================================
struct Result {
    const char* name;
    double us;
    double bw;
    double pct;
    int wavefronts;
};

static double bench(std::function<void()> fn, int warmup, int iters) {
    for (int i = 0; i < warmup; ++i) { fn(); }
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

    // Allocate host data
    std::vector<int8_t> h_A(K), h_B(sz);
    for (int i = 0; i < K; ++i) h_A[i] = (i % 7) - 3;
    for (size_t i = 0; i < sz; ++i) h_B[i] = (i % 11) - 5;

    // Pack VNNI
    std::vector<int8_t> h_vnni(sz);
    pack_vnni(h_B.data(), h_vnni.data(), N, K);

    // Allocate device
    int8_t *d_A, *d_vnni;
    int32_t *d_out;
    HIP_CHECK(hipMalloc(&d_A, K));
    HIP_CHECK(hipMalloc(&d_vnni, sz));
    HIP_CHECK(hipMalloc(&d_out, N * sizeof(int32_t)));
    HIP_CHECK(hipMemcpy(d_A, h_A.data(), K, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_vnni, h_vnni.data(), sz, hipMemcpyHostToDevice));

    const int W = 20;  // warmup
    std::vector<Result> results;

    auto add = [&](const char* n, double us, int wf) {
        double bw = weight_mb / (us / 1e6) / 1e3;
        double pct = bw / MI50_PEAK_BW * 100.0;
        results.push_back({n, us, bw, pct, wf});
    };

    // ── Experiment 1: KSPLIT sweep ──────────────────────────────────────────
    // TN=64
    {
        constexpr int TN = 64;
        int grid = (N + TN - 1) / TN;
        int bx = TN / 4;  // 16 threads on N (each handles 4 cols)

        // KSPLIT=4: block=(16,4)=64 threads=1 wavefront
        if (bx * 4 <= 1024) {
            add("smem TN=64 KS=4",
                bench([&]{ hipLaunchKernelGGL(HIP_KERNEL_NAME(kpar_smem<TN,4>),
                    dim3(grid), dim3(bx, 4), 0, 0, d_A, d_vnni, d_out, N, K); }, W, iters),
                grid * ((bx*4+63)/64));
        }
        // KSPLIT=8
        if (bx * 8 <= 1024) {
            add("smem TN=64 KS=8",
                bench([&]{ hipLaunchKernelGGL(HIP_KERNEL_NAME(kpar_smem<TN,8>),
                    dim3(grid), dim3(bx, 8), 0, 0, d_A, d_vnni, d_out, N, K); }, W, iters),
                grid * ((bx*8+63)/64));
        }
        // KSPLIT=16
        if (bx * 16 <= 1024) {
            add("smem TN=64 KS=16",
                bench([&]{ hipLaunchKernelGGL(HIP_KERNEL_NAME(kpar_smem<TN,16>),
                    dim3(grid), dim3(bx, 16), 0, 0, d_A, d_vnni, d_out, N, K); }, W, iters),
                grid * ((bx*16+63)/64));
        }
        // KSPLIT=32
        if (bx * 32 <= 1024) {
            add("smem TN=64 KS=32",
                bench([&]{ hipLaunchKernelGGL(HIP_KERNEL_NAME(kpar_smem<TN,32>),
                    dim3(grid), dim3(bx, 32), 0, 0, d_A, d_vnni, d_out, N, K); }, W, iters),
                grid * ((bx*32+63)/64));
        }
    }

    // TN=128
    {
        constexpr int TN = 128;
        int grid = (N + TN - 1) / TN;
        int bx = TN / 4;  // 32 threads on N

        if (bx * 4 <= 1024) {
            add("smem TN=128 KS=4",
                bench([&]{ hipLaunchKernelGGL(HIP_KERNEL_NAME(kpar_smem<TN,4>),
                    dim3(grid), dim3(bx, 4), 0, 0, d_A, d_vnni, d_out, N, K); }, W, iters),
                grid * ((bx*4+63)/64));
        }
        if (bx * 8 <= 1024) {
            add("smem TN=128 KS=8",
                bench([&]{ hipLaunchKernelGGL(HIP_KERNEL_NAME(kpar_smem<TN,8>),
                    dim3(grid), dim3(bx, 8), 0, 0, d_A, d_vnni, d_out, N, K); }, W, iters),
                grid * ((bx*8+63)/64));
        }
        if (bx * 16 <= 1024) {
            add("smem TN=128 KS=16",
                bench([&]{ hipLaunchKernelGGL(HIP_KERNEL_NAME(kpar_smem<TN,16>),
                    dim3(grid), dim3(bx, 16), 0, 0, d_A, d_vnni, d_out, N, K); }, W, iters),
                grid * ((bx*16+63)/64));
        }
        if (bx * 32 <= 1024) {
            add("smem TN=128 KS=32",
                bench([&]{ hipLaunchKernelGGL(HIP_KERNEL_NAME(kpar_smem<TN,32>),
                    dim3(grid), dim3(bx, 32), 0, 0, d_A, d_vnni, d_out, N, K); }, W, iters),
                grid * ((bx*32+63)/64));
        }
    }

    // TN=256
    {
        constexpr int TN = 256;
        int grid = (N + TN - 1) / TN;
        int bx = TN / 4;  // 64 threads on N

        if (bx * 4 <= 1024) {
            add("smem TN=256 KS=4",
                bench([&]{ hipLaunchKernelGGL(HIP_KERNEL_NAME(kpar_smem<TN,4>),
                    dim3(grid), dim3(bx, 4), 0, 0, d_A, d_vnni, d_out, N, K); }, W, iters),
                grid * ((bx*4+63)/64));
        }
        if (bx * 8 <= 1024) {
            add("smem TN=256 KS=8",
                bench([&]{ hipLaunchKernelGGL(HIP_KERNEL_NAME(kpar_smem<TN,8>),
                    dim3(grid), dim3(bx, 8), 0, 0, d_A, d_vnni, d_out, N, K); }, W, iters),
                grid * ((bx*8+63)/64));
        }
        if (bx * 16 <= 1024) {
            add("smem TN=256 KS=16",
                bench([&]{ hipLaunchKernelGGL(HIP_KERNEL_NAME(kpar_smem<TN,16>),
                    dim3(grid), dim3(bx, 16), 0, 0, d_A, d_vnni, d_out, N, K); }, W, iters),
                grid * ((bx*16+63)/64));
        }
    }

    // ── Experiment 2: Multi-block kpar (atomicAdd) ──────────────────────────
    for (int kblocks : {4, 8, 16, 32}) {
        // Zero output before each atomic test
        HIP_CHECK(hipMemset(d_out, 0, N * sizeof(int32_t)));

        // 4 cols per thread
        constexpr int TN = 128;
        constexpr int CPT = 4;
        int bx = TN / CPT;
        int grid_n = (N + TN - 1) / TN;

        char buf[64];
        snprintf(buf, sizeof(buf), "atomic TN=128 KB=%d", kblocks);

        add(strdup(buf),
            bench([&]{
                HIP_CHECK(hipMemset(d_out, 0, N * sizeof(int32_t)));
                hipLaunchKernelGGL(HIP_KERNEL_NAME(kpar_atomic<TN, CPT>),
                    dim3(grid_n, kblocks), dim3(bx), 0, 0,
                    d_A, d_vnni, d_out, N, K, kblocks);
            }, W, iters),
            grid_n * kblocks * ((bx+63)/64));
    }

    // ── Experiment 3: Warp-shuffle reduce ───────────────────────────────────
    // KSPLIT must be power-of-2 and <= 64 (wavefront size)
    // TN=128: 32 groups of 4 cols. With KSPLIT=2: 64 threads = 1 wavefront/block
    {
        constexpr int TN = 128;
        int grid = (N + TN - 1) / TN;
        int groups = TN / 4; // 32

        // KSPLIT=2: 32*2=64 threads (1 wavefront)
        add("shfl TN=128 KS=2",
            bench([&]{ hipLaunchKernelGGL(HIP_KERNEL_NAME(kpar_shfl<TN, 2>),
                dim3(grid), dim3(groups * 2), 0, 0, d_A, d_vnni, d_out, N, K); }, W, iters),
            grid * ((groups*2+63)/64));

        // KSPLIT=4: 32*4=128 threads (2 wavefronts)
        add("shfl TN=128 KS=4",
            bench([&]{ hipLaunchKernelGGL(HIP_KERNEL_NAME(kpar_shfl<TN, 4>),
                dim3(grid), dim3(groups * 4), 0, 0, d_A, d_vnni, d_out, N, K); }, W, iters),
            grid * ((groups*4+63)/64));

        // KSPLIT=8: 32*8=256 threads (4 wavefronts)
        add("shfl TN=128 KS=8",
            bench([&]{ hipLaunchKernelGGL(HIP_KERNEL_NAME(kpar_shfl<TN, 8>),
                dim3(grid), dim3(groups * 8), 0, 0, d_A, d_vnni, d_out, N, K); }, W, iters),
            grid * ((groups*8+63)/64));

        // KSPLIT=16: 32*16=512 threads (8 wavefronts)
        add("shfl TN=128 KS=16",
            bench([&]{ hipLaunchKernelGGL(HIP_KERNEL_NAME(kpar_shfl<TN, 16>),
                dim3(grid), dim3(groups * 16), 0, 0, d_A, d_vnni, d_out, N, K); }, W, iters),
            grid * ((groups*16+63)/64));
    }

    // ── Experiment 4: Two-pass kpar ─────────────────────────────────────────
    for (int kblocks : {4, 8, 16, 32}) {
        constexpr int TN = 128;
        constexpr int CPT = 4;
        int bx = TN / CPT;
        int grid_n = (N + TN - 1) / TN;

        // Allocate partials buffer
        int32_t* d_partials;
        HIP_CHECK(hipMalloc(&d_partials, (size_t)kblocks * N * sizeof(int32_t)));

        char buf[64];
        snprintf(buf, sizeof(buf), "2pass TN=128 KB=%d", kblocks);

        add(strdup(buf),
            bench([&]{
                hipLaunchKernelGGL(HIP_KERNEL_NAME(kpar_twopass_compute<TN, CPT>),
                    dim3(grid_n, kblocks), dim3(bx), 0, 0,
                    d_A, d_vnni, d_partials, N, K, kblocks);
                // Reduce
                int reduce_blk = 256;
                int reduce_grid = (N + reduce_blk - 1) / reduce_blk;
                if (kblocks == 4) hipLaunchKernelGGL(HIP_KERNEL_NAME(kpar_twopass_reduce<4>),
                    dim3(reduce_grid), dim3(reduce_blk), 0, 0, d_partials, d_out, N);
                else if (kblocks == 8) hipLaunchKernelGGL(HIP_KERNEL_NAME(kpar_twopass_reduce<8>),
                    dim3(reduce_grid), dim3(reduce_blk), 0, 0, d_partials, d_out, N);
                else if (kblocks == 16) hipLaunchKernelGGL(HIP_KERNEL_NAME(kpar_twopass_reduce<16>),
                    dim3(reduce_grid), dim3(reduce_blk), 0, 0, d_partials, d_out, N);
                else if (kblocks == 32) hipLaunchKernelGGL(HIP_KERNEL_NAME(kpar_twopass_reduce<32>),
                    dim3(reduce_grid), dim3(reduce_blk), 0, 0, d_partials, d_out, N);
            }, W, iters),
            grid_n * kblocks * ((bx+63)/64));

        HIP_CHECK(hipFree(d_partials));
    }

    // ── Experiment 5: 8 cols per thread + kpar ──────────────────────────────
    if (N % 8 == 0) {
        {
            constexpr int TN = 128;
            int grid = (N + TN - 1) / TN;
            int bx = TN / 8; // 16 threads on N

            if (bx * 8 <= 1024) {
                add("8col TN=128 KS=8",
                    bench([&]{ hipLaunchKernelGGL(HIP_KERNEL_NAME(kpar_smem_8col<TN,8>),
                        dim3(grid), dim3(bx, 8), 0, 0, d_A, d_vnni, d_out, N, K); }, W, iters),
                    grid * ((bx*8+63)/64));
            }
            if (bx * 16 <= 1024) {
                add("8col TN=128 KS=16",
                    bench([&]{ hipLaunchKernelGGL(HIP_KERNEL_NAME(kpar_smem_8col<TN,16>),
                        dim3(grid), dim3(bx, 16), 0, 0, d_A, d_vnni, d_out, N, K); }, W, iters),
                    grid * ((bx*16+63)/64));
            }
            if (bx * 32 <= 1024) {
                add("8col TN=128 KS=32",
                    bench([&]{ hipLaunchKernelGGL(HIP_KERNEL_NAME(kpar_smem_8col<TN,32>),
                        dim3(grid), dim3(bx, 32), 0, 0, d_A, d_vnni, d_out, N, K); }, W, iters),
                    grid * ((bx*32+63)/64));
            }
        }
        {
            constexpr int TN = 256;
            int grid = (N + TN - 1) / TN;
            int bx = TN / 8; // 32 threads on N

            if (bx * 8 <= 1024) {
                add("8col TN=256 KS=8",
                    bench([&]{ hipLaunchKernelGGL(HIP_KERNEL_NAME(kpar_smem_8col<TN,8>),
                        dim3(grid), dim3(bx, 8), 0, 0, d_A, d_vnni, d_out, N, K); }, W, iters),
                    grid * ((bx*8+63)/64));
            }
            if (bx * 16 <= 1024) {
                add("8col TN=256 KS=16",
                    bench([&]{ hipLaunchKernelGGL(HIP_KERNEL_NAME(kpar_smem_8col<TN,16>),
                        dim3(grid), dim3(bx, 16), 0, 0, d_A, d_vnni, d_out, N, K); }, W, iters),
                    grid * ((bx*16+63)/64));
            }
        }
    }

    // ── Experiment 6: Large A-tile kpar ─────────────────────────────────────
    {
        constexpr int TN = 128;
        int grid = (N + TN - 1) / TN;
        int bx = TN / 4;

        // A_TILE=512
        if (bx * 16 <= 1024) {
            add("bigtile TN=128 KS=16 A=512",
                bench([&]{ hipLaunchKernelGGL(HIP_KERNEL_NAME(kpar_smem_bigtile<TN,16,512>),
                    dim3(grid), dim3(bx, 16), 0, 0, d_A, d_vnni, d_out, N, K); }, W, iters),
                grid * ((bx*16+63)/64));
        }
        // A_TILE=1024
        if (bx * 16 <= 1024) {
            add("bigtile TN=128 KS=16 A=1024",
                bench([&]{ hipLaunchKernelGGL(HIP_KERNEL_NAME(kpar_smem_bigtile<TN,16,1024>),
                    dim3(grid), dim3(bx, 16), 0, 0, d_A, d_vnni, d_out, N, K); }, W, iters),
                grid * ((bx*16+63)/64));
        }
        // A_TILE=2048
        if (bx * 16 <= 1024) {
            add("bigtile TN=128 KS=16 A=2048",
                bench([&]{ hipLaunchKernelGGL(HIP_KERNEL_NAME(kpar_smem_bigtile<TN,16,2048>),
                    dim3(grid), dim3(bx, 16), 0, 0, d_A, d_vnni, d_out, N, K); }, W, iters),
                grid * ((bx*16+63)/64));
        }
    }

    // ── Print Results ───────────────────────────────────────────────────────
    std::sort(results.begin(), results.end(), [](auto& a, auto& b){ return a.us < b.us; });

    printf("║  %-34s %8s %9s %7s %6s ║\n", "Config", "Time(us)", "BW(GB/s)", "%Peak", "Waves");
    printf("║  %-34s %8s %9s %7s %6s ║\n", "──────────────────────────────────", "────────", "─────────", "───────", "──────");
    for (auto& r : results) {
        const char* star = (r.us <= results[0].us * 1.05) ? " ★" : "";
        printf("║  %-34s %8.1f %9.1f %6.1f%% %6d%s ║\n",
               r.name, r.us, r.bw, r.pct, r.wavefronts, star);
    }
    printf("║  %-34s %8.1f %9s %6s%% %6s ║\n", "THEORETICAL MINIMUM", theory_us, "1000.0", "100.0", "-");
    printf("╚═══════════════════════════════════════════════════════════════════════════════════╝\n");

    HIP_CHECK(hipFree(d_A)); HIP_CHECK(hipFree(d_vnni)); HIP_CHECK(hipFree(d_out));
}

int main(int argc, char** argv)
{
    int dev = argc > 1 ? atoi(argv[1]) : 0;
    HIP_CHECK(hipSetDevice(dev));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, dev));
    printf("Device: %s | CUs: %d | Mem: %.1f GB | LDS: %d KB/CU\n",
           prop.name, prop.multiProcessorCount,
           prop.totalGlobalMem / 1e9, (int)(prop.sharedMemPerBlock / 1024));
    printf("Peak BW: %.0f GB/s\n\n", MI50_PEAK_BW);

    const int iters = 100;
    int shape = argc > 2 ? atoi(argv[2]) : -1;

    if (shape == -1 || shape == 0) run_shape("Q/Wo",     3584,  3584, iters);
    if (shape == -1 || shape == 1) run_shape("FFN Down", 3584, 18944, iters);

    return 0;
}
