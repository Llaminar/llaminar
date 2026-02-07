/**
 * @file Perf__TiledGemvTuning.hip
 * @brief Round 2: Fine-tune the tiled KN VNNI GEMV layout.
 *
 * Based on Round 1 findings:
 *   - Tiled KN VNNI 64x64 is 4x faster for FFN Gate/Up (75% peak BW)
 *   - Tiled KN VNNI 128x128 is 3.5x faster for Q/Wo (19.5% peak BW)
 *   - FFN Down needs k-parallel variant of tiled layout
 *   - LM Head already ~84% peak, keep current flat VNNI
 *
 * This benchmark tests:
 *   1. Tile size sweep: TN×TK from 32x32 to 256x256
 *   2. Tiled + k-parallel for FFN Down
 *   3. Shared memory A tile vs global A reads
 *   4. Number of threads per block (1 vs 4 cols per thread)
 */

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <functional>

#define HIP_CHECK(cmd)                                                              \
    do {                                                                            \
        hipError_t err = cmd;                                                       \
        if (err != hipSuccess) {                                                    \
            fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(err),      \
                    __FILE__, __LINE__);                                             \
            exit(1);                                                                \
        }                                                                           \
    } while (0)

static constexpr double MI50_PEAK_BW = 1000.0;  // GB/s

// ============================================================================
// Tiled KN VNNI GEMV: B[K/TK][N/TN][(TK/4)][TN][4]
// One thread per output N position. TN threads per block. Shared mem for A.
// ============================================================================
template <int TN, int TK>
__global__ void gemv_tiled_kn_vnni(const int8_t* __restrict__ A, const int8_t* __restrict__ B,
                                    int32_t* __restrict__ out, int N, int K)
{
    const int tile_n = blockIdx.x;
    const int lane = threadIdx.x;
    const int n = tile_n * TN + lane;

    __shared__ int8_t sA[TK];
    int32_t acc = 0;

    const int k_tiles = K / TK;
    const int n_tiles = N / TN;
    const int groups_per_tile = TK / 4;

    for (int kt = 0; kt < k_tiles; ++kt) {
        for (int i = lane; i < TK; i += TN) {
            sA[i] = A[kt * TK + i];
        }
        __syncthreads();

        if (n < N) {
            const int8_t* tile = B + (static_cast<size_t>(kt) * n_tiles + tile_n) * TK * TN;
            for (int kg = 0; kg < groups_per_tile; ++kg) {
                const int k_local = kg * 4;
                const int32_t a0 = sA[k_local], a1 = sA[k_local+1], a2 = sA[k_local+2], a3 = sA[k_local+3];
                const uint32_t p = *reinterpret_cast<const uint32_t*>(tile + (kg * TN + lane) * 4);
                acc += a0 * (int32_t)(int8_t)(p & 0xFF)
                     + a1 * (int32_t)(int8_t)((p >> 8) & 0xFF)
                     + a2 * (int32_t)(int8_t)((p >> 16) & 0xFF)
                     + a3 * (int32_t)(int8_t)((p >> 24) & 0xFF);
            }
        }
        __syncthreads();
    }

    if (n < N) out[n] = acc;
}

// ============================================================================
// Tiled KN VNNI with K-parallel (for FFN Down where K >> N)
// Multiple k-splits per block, reduction in shared memory
// ============================================================================
template <int TN, int TK, int KSPLIT>
__global__ void gemv_tiled_kn_vnni_kpar(const int8_t* __restrict__ A, const int8_t* __restrict__ B,
                                         int32_t* __restrict__ out, int N, int K)
{
    const int tile_n = blockIdx.x;
    const int lane_n = threadIdx.x;         // 0..TN-1
    const int lane_k = threadIdx.y;         // 0..KSPLIT-1
    const int n = tile_n * TN + lane_n;

    __shared__ int8_t sA[TK];
    int32_t acc = 0;

    const int k_tiles = K / TK;
    const int n_tiles = N / TN;
    const int groups_per_tile = TK / 4;

    for (int kt = 0; kt < k_tiles; ++kt) {
        // Cooperative A load
        const int linear_tid = lane_k * TN + lane_n;
        for (int i = linear_tid; i < TK; i += TN * KSPLIT) {
            sA[i] = A[kt * TK + i];
        }
        __syncthreads();

        if (n < N) {
            const int8_t* tile = B + (static_cast<size_t>(kt) * n_tiles + tile_n) * TK * TN;
            // Each k-split lane handles every KSPLIT-th k-group
            for (int kg = lane_k; kg < groups_per_tile; kg += KSPLIT) {
                const int k_local = kg * 4;
                const int32_t a0 = sA[k_local], a1 = sA[k_local+1], a2 = sA[k_local+2], a3 = sA[k_local+3];
                const uint32_t p = *reinterpret_cast<const uint32_t*>(tile + (kg * TN + lane_n) * 4);
                acc += a0 * (int32_t)(int8_t)(p & 0xFF)
                     + a1 * (int32_t)(int8_t)((p >> 8) & 0xFF)
                     + a2 * (int32_t)(int8_t)((p >> 16) & 0xFF)
                     + a3 * (int32_t)(int8_t)((p >> 24) & 0xFF);
            }
        }
        __syncthreads();
    }

    // Reduce across k-splits
    __shared__ int32_t partial[KSPLIT][TN];
    partial[lane_k][lane_n] = acc;
    __syncthreads();

    if (lane_k == 0 && n < N) {
        int32_t sum = 0;
        #pragma unroll
        for (int i = 0; i < KSPLIT; ++i) {
            sum += partial[i][lane_n];
        }
        out[n] = sum;
    }
}

// ============================================================================
// Tiled KN VNNI with 2 cols per thread (wide variant for large N)
// ============================================================================
template <int TN, int TK>
__global__ void gemv_tiled_kn_vnni_wide2(const int8_t* __restrict__ A, const int8_t* __restrict__ B,
                                          int32_t* __restrict__ out, int N, int K)
{
    const int tile_n = blockIdx.x;
    const int lane = threadIdx.x;  // 0..TN/2-1
    const int n2 = tile_n * TN + lane * 2;

    __shared__ int8_t sA[TK];
    int32_t acc0 = 0, acc1 = 0;

    const int k_tiles = K / TK;
    const int n_tiles = N / TN;
    const int groups_per_tile = TK / 4;

    for (int kt = 0; kt < k_tiles; ++kt) {
        for (int i = lane; i < TK; i += blockDim.x) {
            sA[i] = A[kt * TK + i];
        }
        __syncthreads();

        if (n2 < N) {
            const int8_t* tile = B + (static_cast<size_t>(kt) * n_tiles + tile_n) * TK * TN;
            for (int kg = 0; kg < groups_per_tile; ++kg) {
                const int k_local = kg * 4;
                const int32_t a0 = sA[k_local], a1 = sA[k_local+1], a2 = sA[k_local+2], a3 = sA[k_local+3];

                const uint32_t p0 = *reinterpret_cast<const uint32_t*>(tile + (kg * TN + lane * 2) * 4);
                const uint32_t p1 = *reinterpret_cast<const uint32_t*>(tile + (kg * TN + lane * 2 + 1) * 4);

                acc0 += a0 * (int32_t)(int8_t)(p0 & 0xFF) + a1 * (int32_t)(int8_t)((p0 >> 8) & 0xFF)
                      + a2 * (int32_t)(int8_t)((p0 >> 16) & 0xFF) + a3 * (int32_t)(int8_t)((p0 >> 24) & 0xFF);
                acc1 += a0 * (int32_t)(int8_t)(p1 & 0xFF) + a1 * (int32_t)(int8_t)((p1 >> 8) & 0xFF)
                      + a2 * (int32_t)(int8_t)((p1 >> 16) & 0xFF) + a3 * (int32_t)(int8_t)((p1 >> 24) & 0xFF);
            }
        }
        __syncthreads();
    }

    if (n2 < N) {
        out[n2] = acc0;
        out[n2+1] = acc1;
    }
}


// ============================================================================
// Pack function
// ============================================================================
template <int TN, int TK>
void pack_tiled_kn_vnni(const int8_t* src_row_major, int8_t* dst, int N, int K)
{
    const int k_tiles = K / TK;
    const int n_tiles = N / TN;
    for (int kt = 0; kt < k_tiles; ++kt) {
        for (int nt = 0; nt < n_tiles; ++nt) {
            for (int kg = 0; kg < TK/4; ++kg) {
                for (int ln = 0; ln < TN; ++ln) {
                    for (int ki = 0; ki < 4; ++ki) {
                        int k = kt * TK + kg * 4 + ki;
                        int n = nt * TN + ln;
                        size_t dst_idx = ((static_cast<size_t>(kt) * n_tiles + nt) * TK * TN)
                                       + (kg * TN + ln) * 4 + ki;
                        dst[dst_idx] = src_row_major[static_cast<size_t>(k) * N + n];
                    }
                }
            }
        }
    }
}

double benchmark_kernel(std::function<void()> launcher, int warmup, int iters)
{
    for (int i = 0; i < warmup; ++i) { launcher(); }
    HIP_CHECK(hipDeviceSynchronize());

    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start));

    for (int i = 0; i < iters; ++i) { launcher(); }

    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    float ms;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));

    return (ms * 1000.0) / iters;
}

struct Result { const char* name; double us; double bw; double pct; };

void run_tile_sweep(int N, int K, int iters)
{
    const size_t wb = static_cast<size_t>(N) * K;
    printf("\n═══════════════════════════════════════════════════════════════════════════════\n");
    printf("  TILE SIZE SWEEP: N=%d  K=%d  Weight=%.2f MB\n", N, K, wb / 1e6);
    printf("═══════════════════════════════════════════════════════════════════════════════\n");

    std::vector<int8_t> h_B(wb);
    for (size_t i = 0; i < wb; ++i) h_B[i] = static_cast<int8_t>((i * 7 + 13) % 251 - 125);
    std::vector<int8_t> h_A(K);
    for (int i = 0; i < K; ++i) h_A[i] = static_cast<int8_t>((i * 3 + 7) % 127 - 63);

    int8_t *d_A; int32_t *d_out;
    HIP_CHECK(hipMalloc(&d_A, K));
    HIP_CHECK(hipMalloc(&d_out, N * sizeof(int32_t)));
    HIP_CHECK(hipMemcpy(d_A, h_A.data(), K, hipMemcpyHostToDevice));

    const int warmup = 5;
    const double total_bytes = wb + K + N * 4.0;
    std::vector<Result> results;

    auto test_tile = [&](const char* label, auto launcher_fn, const int8_t* h_packed) {
        int8_t* d_B;
        HIP_CHECK(hipMalloc(&d_B, wb));
        HIP_CHECK(hipMemcpy(d_B, h_packed, wb, hipMemcpyHostToDevice));
        double us = benchmark_kernel([&]{ launcher_fn(d_A, d_B, d_out, N, K); }, warmup, iters);
        double bw = (total_bytes / 1e9) / (us / 1e6);
        results.push_back({label, us, bw, 100.0 * bw / MI50_PEAK_BW});
        HIP_CHECK(hipFree(d_B));
    };

    // Macro to avoid repeating pack+test logic
    #define TEST_TILED(TN_VAL, TK_VAL, NAME)                                           \
    if ((N % TN_VAL == 0) && (K % TK_VAL == 0)) {                                      \
        std::vector<int8_t> packed(wb);                                                 \
        pack_tiled_kn_vnni<TN_VAL, TK_VAL>(h_B.data(), packed.data(), N, K);           \
        test_tile(NAME, [](int8_t* dA, int8_t* dB, int32_t* dO, int N, int K) {        \
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_tiled_kn_vnni<TN_VAL, TK_VAL>),    \
                dim3(N / TN_VAL), dim3(TN_VAL), 0, 0, dA, dB, dO, N, K);              \
        }, packed.data());                                                              \
    }

    #define TEST_TILED_W2(TN_VAL, TK_VAL, NAME)                                        \
    if ((N % TN_VAL == 0) && (K % TK_VAL == 0)) {                                      \
        std::vector<int8_t> packed(wb);                                                 \
        pack_tiled_kn_vnni<TN_VAL, TK_VAL>(h_B.data(), packed.data(), N, K);           \
        test_tile(NAME, [](int8_t* dA, int8_t* dB, int32_t* dO, int N, int K) {        \
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_tiled_kn_vnni_wide2<TN_VAL, TK_VAL>), \
                dim3(N / TN_VAL), dim3(TN_VAL / 2), 0, 0, dA, dB, dO, N, K);          \
        }, packed.data());                                                              \
    }

    #define TEST_TILED_KPAR(TN_VAL, TK_VAL, KS_VAL, NAME)                              \
    if ((N % TN_VAL == 0) && (K % TK_VAL == 0)) {                                      \
        std::vector<int8_t> packed(wb);                                                 \
        pack_tiled_kn_vnni<TN_VAL, TK_VAL>(h_B.data(), packed.data(), N, K);           \
        test_tile(NAME, [](int8_t* dA, int8_t* dB, int32_t* dO, int N, int K) {        \
            dim3 block(TN_VAL, KS_VAL);                                                \
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_tiled_kn_vnni_kpar<TN_VAL, TK_VAL, KS_VAL>), \
                dim3(N / TN_VAL), block, 0, 0, dA, dB, dO, N, K);                     \
        }, packed.data());                                                              \
    }

    // ── 1-col-per-thread sweep ──
    printf("\n  ── 1 col/thread (TN threads per block) ──\n");
    TEST_TILED(32, 32, "32x32");
    TEST_TILED(32, 64, "32x64");
    TEST_TILED(32, 128, "32x128");
    TEST_TILED(32, 256, "32x256");
    TEST_TILED(64, 32, "64x32");
    TEST_TILED(64, 64, "64x64");
    TEST_TILED(64, 128, "64x128");
    TEST_TILED(64, 256, "64x256");
    TEST_TILED(128, 32, "128x32");
    TEST_TILED(128, 64, "128x64");
    TEST_TILED(128, 128, "128x128");
    TEST_TILED(128, 256, "128x256");
    TEST_TILED(256, 32, "256x32");
    TEST_TILED(256, 64, "256x64");
    TEST_TILED(256, 128, "256x128");
    TEST_TILED(256, 256, "256x256");

    printf("  %-25s %10s %10s %8s\n", "TileNxTileK", "Time(us)", "BW(GB/s)", "%Peak");
    printf("  %-25s %10s %10s %8s\n", "─────────────────────────", "──────────", "──────────", "────────");
    for (auto& r : results) {
        printf("  %-25s %10.1f %10.1f %7.1f%%\n", r.name, r.us, r.bw, r.pct);
    }

    // ── 2-cols-per-thread sweep ──
    results.clear();
    printf("\n  ── 2 cols/thread (TN/2 threads per block) ──\n");
    TEST_TILED_W2(64, 64, "w2_64x64");
    TEST_TILED_W2(64, 128, "w2_64x128");
    TEST_TILED_W2(128, 64, "w2_128x64");
    TEST_TILED_W2(128, 128, "w2_128x128");
    TEST_TILED_W2(256, 64, "w2_256x64");
    TEST_TILED_W2(256, 128, "w2_256x128");
    TEST_TILED_W2(256, 256, "w2_256x256");

    printf("  %-25s %10s %10s %8s\n", "TileNxTileK", "Time(us)", "BW(GB/s)", "%Peak");
    printf("  %-25s %10s %10s %8s\n", "─────────────────────────", "──────────", "──────────", "────────");
    for (auto& r : results) {
        printf("  %-25s %10.1f %10.1f %7.1f%%\n", r.name, r.us, r.bw, r.pct);
    }

    // ── K-parallel sweep (useful when K > N) ──
    results.clear();
    if (K >= 2 * N) {
        printf("\n  ── K-parallel (k-split reduction, for K>N shapes) ──\n");
        TEST_TILED_KPAR(32, 32, 2, "kp_32x32_s2");
        TEST_TILED_KPAR(32, 32, 4, "kp_32x32_s4");
        TEST_TILED_KPAR(32, 32, 8, "kp_32x32_s8");
        TEST_TILED_KPAR(32, 64, 2, "kp_32x64_s2");
        TEST_TILED_KPAR(32, 64, 4, "kp_32x64_s4");
        TEST_TILED_KPAR(32, 64, 8, "kp_32x64_s8");
        TEST_TILED_KPAR(32, 128, 4, "kp_32x128_s4");
        TEST_TILED_KPAR(32, 128, 8, "kp_32x128_s8");
        TEST_TILED_KPAR(64, 64, 2, "kp_64x64_s2");
        TEST_TILED_KPAR(64, 64, 4, "kp_64x64_s4");
        TEST_TILED_KPAR(64, 128, 2, "kp_64x128_s2");
        TEST_TILED_KPAR(64, 128, 4, "kp_64x128_s4");
        TEST_TILED_KPAR(128, 128, 2, "kp_128x128_s2");
        TEST_TILED_KPAR(128, 128, 4, "kp_128x128_s4");
        TEST_TILED_KPAR(128, 256, 2, "kp_128x256_s2");
        TEST_TILED_KPAR(128, 256, 4, "kp_128x256_s4");

        printf("  %-25s %10s %10s %8s\n", "TileNxTileK_sS", "Time(us)", "BW(GB/s)", "%Peak");
        printf("  %-25s %10s %10s %8s\n", "─────────────────────────", "──────────", "──────────", "────────");
        for (auto& r : results) {
            printf("  %-25s %10.1f %10.1f %7.1f%%\n", r.name, r.us, r.bw, r.pct);
        }
    }

    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_out));
}

int main(int argc, char** argv)
{
    int device = 0;
    if (argc > 1) device = atoi(argv[1]);
    HIP_CHECK(hipSetDevice(device));

    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device));
    printf("Device: %s | CUs: %d | Mem: %.1f GB\n",
           prop.name, prop.multiProcessorCount, prop.totalGlobalMem / 1e9);

    const int iters = 100;

    // Test the shapes where tiled layout showed promise
    run_tile_sweep(3584, 3584, iters);      // Q/Wo proj (square) — 3.5x win
    run_tile_sweep(18944, 3584, iters);     // FFN Gate/Up (wide) — 4x win
    run_tile_sweep(3584, 18944, iters);     // FFN Down (tall, needs kpar)

    printf("\n═══════════════════════════════════════════════════════════════════════\n");
    printf("  Done. MI50 peak = 1000 GB/s HBM2\n");
    printf("═══════════════════════════════════════════════════════════════════════\n");

    return 0;
}
