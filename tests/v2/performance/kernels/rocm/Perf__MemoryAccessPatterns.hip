/**
 * @file Perf__MemoryAccessPatterns.hip
 * @brief Micro-benchmarks for INT8 GEMV memory access patterns on MI50.
 *
 * Tests multiple weight layouts and access strategies to determine the
 * optimal memory access pattern for each shape class:
 *
 * Layouts tested:
 *   1. ROW_MAJOR:  B[K][N] — each row of K is contiguous N-wide
 *   2. VNNI:       B[K/4][N][4] — 4 adjacent K values packed per N position
 *   3. COL_MAJOR:  B[N][K] — each column is contiguous K-wide
 *   4. TILED_NK:   B[N/TN][K/TK][TN][TK] — 2D tiled, TN×TK sub-blocks contiguous
 *   5. TILED_KN:   B[K/TK][N/TN][TK][TN] — K-major tiled
 *
 * For each layout, we test a pure memory-read kernel (no compute) and a
 * dot-product kernel, measuring achieved bandwidth vs the 1 TB/s MI50 peak.
 */

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <chrono>

#define HIP_CHECK(cmd)                                                              \
    do {                                                                            \
        hipError_t err = cmd;                                                       \
        if (err != hipSuccess) {                                                    \
            fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(err),      \
                    __FILE__, __LINE__);                                             \
            exit(1);                                                                \
        }                                                                           \
    } while (0)

// ============================================================================
// KERNEL 1: Pure memory read bandwidth test (no compute)
// Read every byte of the weight matrix, sum into output to prevent dead-code elimination
// ============================================================================

// Row-major read: B[K][N], threads sweep N, outer loop over K
__global__ void bw_test_row_major(const int8_t* __restrict__ B, int32_t* __restrict__ out, int N, int K)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    int32_t acc = 0;
    for (int k = 0; k < K; k += 4) {
        const int32_t v = *reinterpret_cast<const int32_t*>(B + k * N + n);  // unaligned for int32 but OK for test
        acc += v;
    }
    out[n] = acc;
}

// Row-major vec4 read: B[K][N], each thread reads 4 consecutive N, outer loop K
__global__ void bw_test_row_major_vec4(const int8_t* __restrict__ B, int32_t* __restrict__ out, int N, int K)
{
    const int n4 = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (n4 >= N) return;
    int4 acc = {0, 0, 0, 0};
    for (int k = 0; k < K; ++k) {
        const int32_t v = *reinterpret_cast<const int32_t*>(B + k * N + n4);
        acc.x += (v & 0xFF);
        acc.y += ((v >> 8) & 0xFF);
        acc.z += ((v >> 16) & 0xFF);
        acc.w += ((v >> 24) & 0xFF);
    }
    *reinterpret_cast<int4*>(out + n4) = acc;
}

// VNNI read: B[K/4][N][4], threads sweep N, outer loop K/4
__global__ void bw_test_vnni(const int8_t* __restrict__ B, int32_t* __restrict__ out, int N, int K)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    int32_t acc = 0;
    const int k_groups = K / 4;
    for (int kg = 0; kg < k_groups; ++kg) {
        const uint32_t v = *reinterpret_cast<const uint32_t*>(B + (static_cast<size_t>(kg) * N + n) * 4);
        acc += v;
    }
    out[n] = acc;
}

// VNNI vec4 read: B[K/4][N][4], each thread reads 4 consecutive N positions
__global__ void bw_test_vnni_vec4(const int8_t* __restrict__ B, int32_t* __restrict__ out, int N, int K)
{
    const int n4 = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (n4 >= N) return;
    int4 acc = {0, 0, 0, 0};
    const int k_groups = K / 4;
    for (int kg = 0; kg < k_groups; ++kg) {
        // Read 4 consecutive VNNI packs = 16 bytes = 4 × uint32_t
        const size_t base = (static_cast<size_t>(kg) * N + n4) * 4;
        const uint32_t v0 = *reinterpret_cast<const uint32_t*>(B + base);
        const uint32_t v1 = *reinterpret_cast<const uint32_t*>(B + base + 4);
        const uint32_t v2 = *reinterpret_cast<const uint32_t*>(B + base + 8);
        const uint32_t v3 = *reinterpret_cast<const uint32_t*>(B + base + 12);
        acc.x += v0;
        acc.y += v1;
        acc.z += v2;
        acc.w += v3;
    }
    *reinterpret_cast<int4*>(out + n4) = acc;
}

// Column-major read: B[N][K], each thread reads its own contiguous K row
__global__ void bw_test_col_major(const int8_t* __restrict__ B, int32_t* __restrict__ out, int N, int K)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    int32_t acc = 0;
    const int8_t* row = B + static_cast<size_t>(n) * K;
    for (int k = 0; k < K; k += 4) {
        const int32_t v = *reinterpret_cast<const int32_t*>(row + k);
        acc += v;
    }
    out[n] = acc;
}

// Tiled NK read: B[N/TN][K/TK][TN][TK] with TN=64, TK=64
// Each block handles one N-tile; threads iterate K-tiles reading contiguous sub-blocks
template <int TN, int TK>
__global__ void bw_test_tiled_nk(const int8_t* __restrict__ B, int32_t* __restrict__ out, int N, int K)
{
    const int tile_n = blockIdx.x;
    const int lane = threadIdx.x;  // 0..TN-1
    const int n = tile_n * TN + lane;
    if (n >= N) return;

    int32_t acc = 0;
    const int k_tiles = K / TK;
    for (int kt = 0; kt < k_tiles; ++kt) {
        // Tile at [tile_n][kt] starts at offset (tile_n * k_tiles + kt) * TN * TK
        const int8_t* tile = B + (static_cast<size_t>(tile_n) * k_tiles + kt) * TN * TK;
        // Within tile: element [lane][k] is at offset lane * TK + k
        const int8_t* row = tile + lane * TK;
        for (int k = 0; k < TK; k += 4) {
            const int32_t v = *reinterpret_cast<const int32_t*>(row + k);
            acc += v;
        }
    }
    out[n] = acc;
}

// Tiled KN read: B[K/TK][N/TN][TK][TN] — K-major outer, N-inner sub-blocks contiguous
// Adjacent threads read adjacent bytes within a TK×TN tile (coalesced reads on N dimension)
template <int TN, int TK>
__global__ void bw_test_tiled_kn(const int8_t* __restrict__ B, int32_t* __restrict__ out, int N, int K)
{
    const int tile_n = blockIdx.x;
    const int lane = threadIdx.x;
    const int n = tile_n * TN + lane;
    if (n >= N) return;

    int32_t acc = 0;
    const int k_tiles = K / TK;
    const int n_tiles = N / TN;
    for (int kt = 0; kt < k_tiles; ++kt) {
        // Tile at [kt][tile_n] starts at offset (kt * n_tiles + tile_n) * TK * TN
        const int8_t* tile = B + (static_cast<size_t>(kt) * n_tiles + tile_n) * TK * TN;
        // Within tile: element [k][lane] is at offset k * TN + lane
        for (int k = 0; k < TK; k += 4) {
            // Read 4 values for this lane across 4 consecutive k rows within the tile
            const int8_t b0 = tile[k * TN + lane];
            const int8_t b1 = tile[(k+1) * TN + lane];
            const int8_t b2 = tile[(k+2) * TN + lane];
            const int8_t b3 = tile[(k+3) * TN + lane];
            acc += b0 + b1 + b2 + b3;
        }
    }
    out[n] = acc;
}

// Tiled KN with VNNI-within-tile: B[K/TK][N/TN][(TK/4)][TN][4]
// Best of both: coalesced reads + compact k-packing within each tile
template <int TN, int TK>
__global__ void bw_test_tiled_kn_vnni(const int8_t* __restrict__ B, int32_t* __restrict__ out, int N, int K)
{
    const int tile_n = blockIdx.x;
    const int lane = threadIdx.x;
    const int n = tile_n * TN + lane;
    if (n >= N) return;

    int32_t acc = 0;
    const int k_tiles = K / TK;
    const int n_tiles = N / TN;
    const int groups_per_tile = TK / 4;
    for (int kt = 0; kt < k_tiles; ++kt) {
        const int8_t* tile = B + (static_cast<size_t>(kt) * n_tiles + tile_n) * TK * TN;
        // Inside tile: VNNI layout [(TK/4)][TN][4]
        for (int kg = 0; kg < groups_per_tile; ++kg) {
            const uint32_t v = *reinterpret_cast<const uint32_t*>(tile + (kg * TN + lane) * 4);
            acc += v;
        }
    }
    out[n] = acc;
}


// ============================================================================
// KERNEL 2: Actual GEMV dot-product kernels per layout
// ============================================================================

// Row-major GEMV: B[K][N]
__global__ void gemv_row_major(const int8_t* __restrict__ A, const int8_t* __restrict__ B,
                               int32_t* __restrict__ out, int N, int K)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    int32_t acc = 0;
    for (int k = 0; k < K; k += 4) {
        const int32_t a0 = A[k], a1 = A[k+1], a2 = A[k+2], a3 = A[k+3];
        acc += a0 * (int32_t)B[k*N + n] + a1 * (int32_t)B[(k+1)*N + n]
             + a2 * (int32_t)B[(k+2)*N + n] + a3 * (int32_t)B[(k+3)*N + n];
    }
    out[n] = acc;
}

// VNNI GEMV: B[K/4][N][4]  (current production kernel style)
__global__ void gemv_vnni(const int8_t* __restrict__ A, const int8_t* __restrict__ B,
                          int32_t* __restrict__ out, int N, int K)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    int32_t acc = 0;
    const int k_groups = K / 4;
    for (int kg = 0; kg < k_groups; ++kg) {
        const int k = kg * 4;
        const int32_t a0 = A[k], a1 = A[k+1], a2 = A[k+2], a3 = A[k+3];
        const uint32_t p = *reinterpret_cast<const uint32_t*>(B + (static_cast<size_t>(kg) * N + n) * 4);
        acc += a0 * (int32_t)(int8_t)(p & 0xFF)
             + a1 * (int32_t)(int8_t)((p >> 8) & 0xFF)
             + a2 * (int32_t)(int8_t)((p >> 16) & 0xFF)
             + a3 * (int32_t)(int8_t)((p >> 24) & 0xFF);
    }
    out[n] = acc;
}

// VNNI GEMV with shared memory for A: B[K/4][N][4]
__global__ void gemv_vnni_smem(const int8_t* __restrict__ A, const int8_t* __restrict__ B,
                               int32_t* __restrict__ out, int N, int K)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ int8_t sA[256];

    int32_t acc = 0;
    const int k_groups = K / 4;

    for (int k_base = 0; k_base < K; k_base += 256) {
        const int tile_k = min(256, K - k_base);
        for (int i = threadIdx.x; i < tile_k; i += blockDim.x) {
            sA[i] = A[k_base + i];
        }
        __syncthreads();

        if (n < N) {
            const int tile_groups = tile_k / 4;
            const int kg_base = k_base / 4;
            for (int kg = 0; kg < tile_groups; ++kg) {
                const int k_local = kg * 4;
                const int32_t a0 = sA[k_local], a1 = sA[k_local+1], a2 = sA[k_local+2], a3 = sA[k_local+3];
                const uint32_t p = *reinterpret_cast<const uint32_t*>(
                    B + (static_cast<size_t>(kg_base + kg) * N + n) * 4);
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

// Col-major GEMV: B[N][K], each thread reads its own row contiguously
__global__ void gemv_col_major(const int8_t* __restrict__ A, const int8_t* __restrict__ B,
                               int32_t* __restrict__ out, int N, int K)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    int32_t acc = 0;
    const int8_t* row = B + static_cast<size_t>(n) * K;
    for (int k = 0; k < K; k += 4) {
        acc += (int32_t)A[k]   * (int32_t)row[k]
             + (int32_t)A[k+1] * (int32_t)row[k+1]
             + (int32_t)A[k+2] * (int32_t)row[k+2]
             + (int32_t)A[k+3] * (int32_t)row[k+3];
    }
    out[n] = acc;
}

// Col-major GEMV with shared A: B[N][K]
__global__ void gemv_col_major_smem(const int8_t* __restrict__ A, const int8_t* __restrict__ B,
                                    int32_t* __restrict__ out, int N, int K)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ int8_t sA[256];

    int32_t acc = 0;
    for (int k_base = 0; k_base < K; k_base += 256) {
        const int tile_k = min(256, K - k_base);
        for (int i = threadIdx.x; i < tile_k; i += blockDim.x) {
            sA[i] = A[k_base + i];
        }
        __syncthreads();

        if (n < N) {
            const int8_t* row = B + static_cast<size_t>(n) * K + k_base;
            for (int k = 0; k < tile_k; k += 4) {
                acc += (int32_t)sA[k]   * (int32_t)row[k]
                     + (int32_t)sA[k+1] * (int32_t)row[k+1]
                     + (int32_t)sA[k+2] * (int32_t)row[k+2]
                     + (int32_t)sA[k+3] * (int32_t)row[k+3];
            }
        }
        __syncthreads();
    }

    if (n < N) out[n] = acc;
}

// Tiled KN VNNI GEMV: B[K/TK][N/TN][(TK/4)][TN][4] with shared A tile
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
        // Load A tile into shared memory
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

// Tiled KN VNNI GEMV with vec4: each thread handles 4 consecutive N within tile
template <int TN, int TK>
__global__ void gemv_tiled_kn_vnni_vec4(const int8_t* __restrict__ A, const int8_t* __restrict__ B,
                                         int32_t* __restrict__ out, int N, int K)
{
    const int tile_n = blockIdx.x;
    const int lane = threadIdx.x;  // 0..TN/4-1
    const int n4 = tile_n * TN + lane * 4;

    __shared__ int8_t sA[TK];
    int4 acc = {0, 0, 0, 0};

    const int k_tiles = K / TK;
    const int n_tiles = N / TN;
    const int groups_per_tile = TK / 4;

    for (int kt = 0; kt < k_tiles; ++kt) {
        for (int i = lane; i < TK; i += blockDim.x) {
            sA[i] = A[kt * TK + i];
        }
        __syncthreads();

        if (n4 < N) {
            const int8_t* tile = B + (static_cast<size_t>(kt) * n_tiles + tile_n) * TK * TN;
            for (int kg = 0; kg < groups_per_tile; ++kg) {
                const int k_local = kg * 4;
                const int32_t a0 = sA[k_local], a1 = sA[k_local+1], a2 = sA[k_local+2], a3 = sA[k_local+3];

                const size_t base = (kg * TN + lane * 4) * 4;
                const uint32_t p0 = *reinterpret_cast<const uint32_t*>(tile + base);
                const uint32_t p1 = *reinterpret_cast<const uint32_t*>(tile + base + 4);
                const uint32_t p2 = *reinterpret_cast<const uint32_t*>(tile + base + 8);
                const uint32_t p3 = *reinterpret_cast<const uint32_t*>(tile + base + 12);

                acc.x += a0*(int32_t)(int8_t)(p0&0xFF) + a1*(int32_t)(int8_t)((p0>>8)&0xFF)
                       + a2*(int32_t)(int8_t)((p0>>16)&0xFF) + a3*(int32_t)(int8_t)((p0>>24)&0xFF);
                acc.y += a0*(int32_t)(int8_t)(p1&0xFF) + a1*(int32_t)(int8_t)((p1>>8)&0xFF)
                       + a2*(int32_t)(int8_t)((p1>>16)&0xFF) + a3*(int32_t)(int8_t)((p1>>24)&0xFF);
                acc.z += a0*(int32_t)(int8_t)(p2&0xFF) + a1*(int32_t)(int8_t)((p2>>8)&0xFF)
                       + a2*(int32_t)(int8_t)((p2>>16)&0xFF) + a3*(int32_t)(int8_t)((p2>>24)&0xFF);
                acc.w += a0*(int32_t)(int8_t)(p3&0xFF) + a1*(int32_t)(int8_t)((p3>>8)&0xFF)
                       + a2*(int32_t)(int8_t)((p3>>16)&0xFF) + a3*(int32_t)(int8_t)((p3>>24)&0xFF);
            }
        }
        __syncthreads();
    }

    if (n4 < N) {
        out[n4] = acc.x; out[n4+1] = acc.y; out[n4+2] = acc.z; out[n4+3] = acc.w;
    }
}


// ============================================================================
// Benchmark harness
// ============================================================================

struct BenchResult {
    const char* name;
    double time_us;
    double bw_gbps;
    double pct_peak;
};

static constexpr double MI50_PEAK_BW = 1000.0;  // GB/s

void pack_vnni(const int8_t* src_row_major, int8_t* dst_vnni, int N, int K)
{
    // src: B[K][N] row-major → dst: B[K/4][N][4] VNNI
    for (int k = 0; k < K; k += 4) {
        int kg = k / 4;
        for (int n = 0; n < N; ++n) {
            for (int ki = 0; ki < 4; ++ki) {
                dst_vnni[(static_cast<size_t>(kg) * N + n) * 4 + ki] = src_row_major[(k + ki) * N + n];
            }
        }
    }
}

void pack_col_major(const int8_t* src_row_major, int8_t* dst_col, int N, int K)
{
    // src: B[K][N] → dst: B[N][K]
    for (int k = 0; k < K; ++k) {
        for (int n = 0; n < N; ++n) {
            dst_col[static_cast<size_t>(n) * K + k] = src_row_major[static_cast<size_t>(k) * N + n];
        }
    }
}

template <int TN, int TK>
void pack_tiled_kn_vnni(const int8_t* src_row_major, int8_t* dst, int N, int K)
{
    // src: B[K][N] → dst: B[K/TK][N/TN][(TK/4)][TN][4]
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

    return (ms * 1000.0) / iters;  // us per call
}

void run_suite(int N, int K, int iters)
{
    const size_t weight_bytes = static_cast<size_t>(N) * K;
    printf("\n");
    printf("═══════════════════════════════════════════════════════════════════════════════════════\n");
    printf("  Shape: N=%d  K=%d  Weight=%.2f MB   (iters=%d)\n", N, K, weight_bytes / 1e6, iters);
    printf("═══════════════════════════════════════════════════════════════════════════════════════\n");

    // Allocate and fill host weights (row-major)
    std::vector<int8_t> h_B(weight_bytes);
    for (size_t i = 0; i < weight_bytes; ++i) {
        h_B[i] = static_cast<int8_t>((i * 7 + 13) % 251 - 125);
    }
    std::vector<int8_t> h_A(K);
    for (int i = 0; i < K; ++i) {
        h_A[i] = static_cast<int8_t>((i * 3 + 7) % 127 - 63);
    }

    // Pack into different layouts on host
    std::vector<int8_t> h_vnni(weight_bytes);
    pack_vnni(h_B.data(), h_vnni.data(), N, K);

    std::vector<int8_t> h_col(weight_bytes);
    pack_col_major(h_B.data(), h_col.data(), N, K);

    // Tile sizes for tiled layouts
    constexpr int TN64 = 64, TK64 = 64;
    constexpr int TN128 = 128, TK128 = 128;
    constexpr int TN256 = 256, TK256 = 256;

    // Only pack tiled if dimensions are divisible
    const bool can_tile_64 = (N % TN64 == 0) && (K % TK64 == 0);
    const bool can_tile_128 = (N % TN128 == 0) && (K % TK128 == 0);
    const bool can_tile_256 = (N % TN256 == 0) && (K % TK256 == 0);

    std::vector<int8_t> h_tiled_kn_vnni_64, h_tiled_kn_vnni_128, h_tiled_kn_vnni_256;
    if (can_tile_64) {
        h_tiled_kn_vnni_64.resize(weight_bytes);
        pack_tiled_kn_vnni<TN64, TK64>(h_B.data(), h_tiled_kn_vnni_64.data(), N, K);
    }
    if (can_tile_128) {
        h_tiled_kn_vnni_128.resize(weight_bytes);
        pack_tiled_kn_vnni<TN128, TK128>(h_B.data(), h_tiled_kn_vnni_128.data(), N, K);
    }
    if (can_tile_256) {
        h_tiled_kn_vnni_256.resize(weight_bytes);
        pack_tiled_kn_vnni<TN256, TK256>(h_B.data(), h_tiled_kn_vnni_256.data(), N, K);
    }

    // Allocate device memory
    int8_t *d_B_row, *d_B_vnni, *d_B_col, *d_A;
    int8_t *d_B_tkv64 = nullptr, *d_B_tkv128 = nullptr, *d_B_tkv256 = nullptr;
    int32_t *d_out;

    HIP_CHECK(hipMalloc(&d_B_row, weight_bytes));
    HIP_CHECK(hipMalloc(&d_B_vnni, weight_bytes));
    HIP_CHECK(hipMalloc(&d_B_col, weight_bytes));
    HIP_CHECK(hipMalloc(&d_A, K));
    HIP_CHECK(hipMalloc(&d_out, N * sizeof(int32_t)));

    HIP_CHECK(hipMemcpy(d_B_row, h_B.data(), weight_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B_vnni, h_vnni.data(), weight_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B_col, h_col.data(), weight_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_A, h_A.data(), K, hipMemcpyHostToDevice));

    if (can_tile_64) {
        HIP_CHECK(hipMalloc(&d_B_tkv64, weight_bytes));
        HIP_CHECK(hipMemcpy(d_B_tkv64, h_tiled_kn_vnni_64.data(), weight_bytes, hipMemcpyHostToDevice));
    }
    if (can_tile_128) {
        HIP_CHECK(hipMalloc(&d_B_tkv128, weight_bytes));
        HIP_CHECK(hipMemcpy(d_B_tkv128, h_tiled_kn_vnni_128.data(), weight_bytes, hipMemcpyHostToDevice));
    }
    if (can_tile_256) {
        HIP_CHECK(hipMalloc(&d_B_tkv256, weight_bytes));
        HIP_CHECK(hipMemcpy(d_B_tkv256, h_tiled_kn_vnni_256.data(), weight_bytes, hipMemcpyHostToDevice));
    }

    std::vector<BenchResult> results;
    const int warmup = 5;

    auto add_result = [&](const char* name, double us) {
        // Total bytes: weights + input + output
        double total_bytes = weight_bytes + K + N * 4.0;
        double bw = (total_bytes / 1e9) / (us / 1e6);
        double pct = 100.0 * bw / MI50_PEAK_BW;
        results.push_back({name, us, bw, pct});
    };

    // ── BANDWIDTH TESTS (read-only, no compute) ──────────────────────────────
    printf("\n  ── PURE BANDWIDTH TESTS (read-only) ──────────────────────────────\n");

    {
        int block = 256;
        int grid = (N + block - 1) / block;
        add_result("bw_row_major",
            benchmark_kernel([&]{ hipLaunchKernelGGL(bw_test_row_major, dim3(grid), dim3(block), 0, 0,
                                                     d_B_row, d_out, N, K); }, warmup, iters));
    }
    {
        int block = 256;
        int grid = (N/4 + block - 1) / block;
        add_result("bw_row_major_vec4",
            benchmark_kernel([&]{ hipLaunchKernelGGL(bw_test_row_major_vec4, dim3(grid), dim3(block), 0, 0,
                                                     d_B_row, d_out, N, K); }, warmup, iters));
    }
    {
        int block = 256;
        int grid = (N + block - 1) / block;
        add_result("bw_vnni",
            benchmark_kernel([&]{ hipLaunchKernelGGL(bw_test_vnni, dim3(grid), dim3(block), 0, 0,
                                                     d_B_vnni, d_out, N, K); }, warmup, iters));
    }
    {
        int block = 256;
        int grid = (N/4 + block - 1) / block;
        add_result("bw_vnni_vec4",
            benchmark_kernel([&]{ hipLaunchKernelGGL(bw_test_vnni_vec4, dim3(grid), dim3(block), 0, 0,
                                                     d_B_vnni, d_out, N, K); }, warmup, iters));
    }
    {
        int block = 256;
        int grid = (N + block - 1) / block;
        add_result("bw_col_major",
            benchmark_kernel([&]{ hipLaunchKernelGGL(bw_test_col_major, dim3(grid), dim3(block), 0, 0,
                                                     d_B_col, d_out, N, K); }, warmup, iters));
    }
    if (can_tile_64) {
        add_result("bw_tiled_kn_vnni_64x64",
            benchmark_kernel([&]{ hipLaunchKernelGGL(HIP_KERNEL_NAME(bw_test_tiled_kn_vnni<64,64>),
                dim3(N/64), dim3(64), 0, 0, d_B_tkv64, d_out, N, K); }, warmup, iters));
    }
    if (can_tile_128) {
        add_result("bw_tiled_kn_vnni_128x128",
            benchmark_kernel([&]{ hipLaunchKernelGGL(HIP_KERNEL_NAME(bw_test_tiled_kn_vnni<128,128>),
                dim3(N/128), dim3(128), 0, 0, d_B_tkv128, d_out, N, K); }, warmup, iters));
    }
    if (can_tile_256) {
        add_result("bw_tiled_kn_vnni_256x256",
            benchmark_kernel([&]{ hipLaunchKernelGGL(HIP_KERNEL_NAME(bw_test_tiled_kn_vnni<256,256>),
                dim3(N/256), dim3(256), 0, 0, d_B_tkv256, d_out, N, K); }, warmup, iters));
    }

    // Print BW results
    printf("  %-35s %10s %10s %8s\n", "Kernel", "Time(us)", "BW(GB/s)", "%Peak");
    printf("  %-35s %10s %10s %8s\n", "───────────────────────────────────", "──────────", "──────────", "────────");
    for (auto& r : results) {
        printf("  %-35s %10.1f %10.1f %7.1f%%\n", r.name, r.time_us, r.bw_gbps, r.pct_peak);
    }

    // ── GEMV TESTS (dot-product, measures real throughput) ────────────────────
    results.clear();
    printf("\n  ── GEMV TESTS (dot-product) ───────────────────────────────────────\n");

    {
        int block = 256;
        int grid = (N + block - 1) / block;
        add_result("gemv_row_major",
            benchmark_kernel([&]{ hipLaunchKernelGGL(gemv_row_major, dim3(grid), dim3(block), 0, 0,
                                                     d_A, d_B_row, d_out, N, K); }, warmup, iters));
    }
    {
        int block = 256;
        int grid = (N + block - 1) / block;
        add_result("gemv_vnni",
            benchmark_kernel([&]{ hipLaunchKernelGGL(gemv_vnni, dim3(grid), dim3(block), 0, 0,
                                                     d_A, d_B_vnni, d_out, N, K); }, warmup, iters));
    }
    {
        int block = 256;
        int grid = (N + block - 1) / block;
        add_result("gemv_vnni_smem",
            benchmark_kernel([&]{ hipLaunchKernelGGL(gemv_vnni_smem, dim3(grid), dim3(block), 0, 0,
                                                     d_A, d_B_vnni, d_out, N, K); }, warmup, iters));
    }
    {
        int block = 256;
        int grid = (N + block - 1) / block;
        add_result("gemv_col_major",
            benchmark_kernel([&]{ hipLaunchKernelGGL(gemv_col_major, dim3(grid), dim3(block), 0, 0,
                                                     d_A, d_B_col, d_out, N, K); }, warmup, iters));
    }
    {
        int block = 256;
        int grid = (N + block - 1) / block;
        add_result("gemv_col_major_smem",
            benchmark_kernel([&]{ hipLaunchKernelGGL(gemv_col_major_smem, dim3(grid), dim3(block), 0, 0,
                                                     d_A, d_B_col, d_out, N, K); }, warmup, iters));
    }
    if (can_tile_64) {
        add_result("gemv_tiled_kn_vnni_64x64",
            benchmark_kernel([&]{ hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_tiled_kn_vnni<64,64>),
                dim3(N/64), dim3(64), 0, 0, d_A, d_B_tkv64, d_out, N, K); }, warmup, iters));
    }
    if (can_tile_128) {
        add_result("gemv_tiled_kn_vnni_128x128",
            benchmark_kernel([&]{ hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_tiled_kn_vnni<128,128>),
                dim3(N/128), dim3(128), 0, 0, d_A, d_B_tkv128, d_out, N, K); }, warmup, iters));
    }
    if (can_tile_256) {
        add_result("gemv_tiled_kn_vnni_256x256",
            benchmark_kernel([&]{ hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_tiled_kn_vnni<256,256>),
                dim3(N/256), dim3(256), 0, 0, d_A, d_B_tkv256, d_out, N, K); }, warmup, iters));
    }
    // vec4 variants for tiled
    if (can_tile_64) {
        add_result("gemv_tiled_kn_vnni_v4_64x64",
            benchmark_kernel([&]{ hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_tiled_kn_vnni_vec4<64,64>),
                dim3(N/64), dim3(64/4), 0, 0, d_A, d_B_tkv64, d_out, N, K); }, warmup, iters));
    }
    if (can_tile_128) {
        add_result("gemv_tiled_kn_vnni_v4_128x128",
            benchmark_kernel([&]{ hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_tiled_kn_vnni_vec4<128,128>),
                dim3(N/128), dim3(128/4), 0, 0, d_A, d_B_tkv128, d_out, N, K); }, warmup, iters));
    }
    if (can_tile_256) {
        add_result("gemv_tiled_kn_vnni_v4_256x256",
            benchmark_kernel([&]{ hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_tiled_kn_vnni_vec4<256,256>),
                dim3(N/256), dim3(256/4), 0, 0, d_A, d_B_tkv256, d_out, N, K); }, warmup, iters));
    }

    // Print GEMV results
    printf("  %-35s %10s %10s %8s\n", "Kernel", "Time(us)", "BW(GB/s)", "%Peak");
    printf("  %-35s %10s %10s %8s\n", "───────────────────────────────────", "──────────", "──────────", "────────");
    for (auto& r : results) {
        printf("  %-35s %10.1f %10.1f %7.1f%%\n", r.name, r.time_us, r.bw_gbps, r.pct_peak);
    }

    // Cleanup
    HIP_CHECK(hipFree(d_B_row));
    HIP_CHECK(hipFree(d_B_vnni));
    HIP_CHECK(hipFree(d_B_col));
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_out));
    if (d_B_tkv64) HIP_CHECK(hipFree(d_B_tkv64));
    if (d_B_tkv128) HIP_CHECK(hipFree(d_B_tkv128));
    if (d_B_tkv256) HIP_CHECK(hipFree(d_B_tkv256));
}

int main(int argc, char** argv)
{
    int device = 0;
    if (argc > 1) device = atoi(argv[1]);
    HIP_CHECK(hipSetDevice(device));

    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device));
    printf("Device: %s | CUs: %d | Mem: %.1f GB | Clock: %d MHz\n",
           prop.name, prop.multiProcessorCount,
           prop.totalGlobalMem / 1e9, prop.memoryClockRate / 1000);

    const int iters = 100;

    // Test all shapes from the 7B decode layer
    run_suite(3584, 3584, iters);      // Q proj / Wo proj (square)
    run_suite(512, 3584, iters);       // K proj / V proj (tall)
    run_suite(18944, 3584, iters);     // FFN Gate / FFN Up (wide)
    run_suite(3584, 18944, iters);     // FFN Down (tall, k-parallel)
    run_suite(152064, 3584, iters);    // LM Head (very wide)

    printf("\n══════════════════════════════════════════════════════════════════════════════════\n");
    printf("  Done. Peak BW reference: MI50 = 1000 GB/s (HBM2)\n");
    printf("══════════════════════════════════════════════════════════════════════════════════\n");

    return 0;
}
