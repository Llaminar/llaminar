/**
 * @file Perf__GemvStrategySearch.hip
 * @brief Comprehensive GEMV strategy search for MI50 (gfx906).
 *
 * Explores fundamentally different kernel designs:
 *
 * Strategy A: "Flat VNNI" (current production baseline)
 *   Layout: B[K/4][N][4]
 *   1 thread per output col, loops all K/4 groups
 *   Problem: 76KB stride between k-groups for Gate/Up
 *
 * Strategy B: "Tiled KN VNNI" (Round 1 winner for wide shapes)
 *   Layout: B[K/TK][N/TN][(TK/4)][TN][4]
 *   1 thread per output col within tile, loops k-groups within tile
 *   Stride reduced to TN*4 (256B for TN=64) within tile
 *
 * Strategy C: "Tiled KN VNNI + K-parallel" (new, for square/tall shapes)
 *   Same tiled layout, but multiple threads per output col split K
 *   More wavefronts → better latency hiding for small-N shapes
 *
 * Strategy D: "Column-major + LDS staging" (new, fundamentally different)
 *   Layout: B[N][K] (transposed)
 *   Block cooperatively loads [BN×TK] tile into LDS via coalesced K-reads
 *   Each thread reads its column from LDS sequentially
 *   Tests whether LDS staging can beat direct tiled HBM reads
 *
 * Strategy E: "Row-major wide-vec" (new, no repacking needed!)
 *   Layout: B[K][N] (original row-major, no packing)
 *   Each thread handles 4-16 adjacent N columns via wide loads
 *   Tests whether avoiding packing overhead is worth the stride cost
 *
 * Tests each strategy across 3 shape classes:
 *   - Gate/Up 18944×3584 (wide: N >> K, biggest time sink)
 *   - Q/Wo    3584×3584  (square: N = K)
 *   - FFN Down 3584×18944 (tall: K >> N)
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

// ============================================================================
// STRATEGY A: Flat VNNI baseline — B[K/4][N][4]
// Current production kernel, 1 col/thread, shared A
// ============================================================================
template <int BLOCK>
__global__ void strat_A_flat_vnni(
    const int8_t* __restrict__ A, const int8_t* __restrict__ B,
    int32_t* __restrict__ out, int N, int K)
{
    const int n = blockIdx.x * BLOCK + threadIdx.x;
    if (n >= N) return;

    __shared__ int8_t sA[256];
    int32_t acc = 0;
    const int k_groups = K / 4;

    for (int kb = 0; kb < K; kb += 256) {
        const int tk = min(256, K - kb);
        for (int i = threadIdx.x; i < tk; i += BLOCK) sA[i] = A[kb + i];
        __syncthreads();

        const int tg = tk / 4;
        const int kg0 = kb / 4;
        for (int kg = 0; kg < tg; ++kg) {
            const int kl = kg * 4;
            const int32_t a0 = sA[kl], a1 = sA[kl+1], a2 = sA[kl+2], a3 = sA[kl+3];
            const uint32_t p = *reinterpret_cast<const uint32_t*>(
                B + (static_cast<size_t>(kg0 + kg) * N + n) * 4);
            acc += a0 * (int32_t)(int8_t)(p & 0xFF)
                 + a1 * (int32_t)(int8_t)((p >> 8) & 0xFF)
                 + a2 * (int32_t)(int8_t)((p >> 16) & 0xFF)
                 + a3 * (int32_t)(int8_t)((p >> 24) & 0xFF);
        }
        __syncthreads();
    }
    out[n] = acc;
}

// Flat VNNI vec4 — 4 cols per thread (for wide shapes)
template <int BLOCK>
__global__ void strat_A_flat_vnni_vec4(
    const int8_t* __restrict__ A, const int8_t* __restrict__ B,
    int32_t* __restrict__ out, int N, int K)
{
    const int n4 = blockIdx.x * (BLOCK * 4) + threadIdx.x * 4;
    if (n4 >= N) return;

    int4 acc = {0, 0, 0, 0};
    const int k_groups = K / 4;
    for (int kg = 0; kg < k_groups; ++kg) {
        const int kb = kg * 4;
        const int32_t a0 = A[kb], a1 = A[kb+1], a2 = A[kb+2], a3 = A[kb+3];
        const size_t base = (static_cast<size_t>(kg) * N + n4) * 4;

        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            const uint32_t p = *reinterpret_cast<const uint32_t*>(B + base + c * 4);
            int32_t* a_ptr = &acc.x + c;
            *a_ptr += a0 * (int32_t)(int8_t)(p & 0xFF)
                    + a1 * (int32_t)(int8_t)((p >> 8) & 0xFF)
                    + a2 * (int32_t)(int8_t)((p >> 16) & 0xFF)
                    + a3 * (int32_t)(int8_t)((p >> 24) & 0xFF);
        }
    }
    out[n4] = acc.x; out[n4+1] = acc.y; out[n4+2] = acc.z; out[n4+3] = acc.w;
}

// Flat VNNI k-parallel — split K across threadIdx.y
template <int TILE_N, int KSPLIT>
__global__ void strat_A_flat_vnni_kpar(
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
// STRATEGY B: Tiled KN VNNI — B[K/TK][N/TN][(TK/4)][TN][4]
// 1 thread per output col within tile
// ============================================================================
template <int TN, int TK>
__global__ void strat_B_tiled(
    const int8_t* __restrict__ A, const int8_t* __restrict__ B,
    int32_t* __restrict__ out, int N, int K)
{
    const int tile_n = blockIdx.x;
    const int lane = threadIdx.x;
    const int n = tile_n * TN + lane;

    __shared__ int8_t sA[TK];
    int32_t acc = 0;

    const int k_tiles = K / TK;
    const int n_tiles = N / TN;
    const int groups = TK / 4;

    for (int kt = 0; kt < k_tiles; ++kt) {
        for (int i = lane; i < TK; i += TN) sA[i] = A[kt * TK + i];
        __syncthreads();

        if (n < N) {
            const int8_t* tile = B + (static_cast<size_t>(kt) * n_tiles + tile_n) * TK * TN;
            for (int kg = 0; kg < groups; ++kg) {
                const int kl = kg * 4;
                const int32_t a0 = sA[kl], a1 = sA[kl+1], a2 = sA[kl+2], a3 = sA[kl+3];
                const uint32_t p = *reinterpret_cast<const uint32_t*>(
                    tile + (kg * TN + lane) * 4);
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
// STRATEGY C: Tiled KN VNNI + K-parallel
// TN threads on N × KSPLIT threads on K, reduces in smem
// ============================================================================
template <int TN, int TK, int KSPLIT>
__global__ void strat_C_tiled_kpar(
    const int8_t* __restrict__ A, const int8_t* __restrict__ B,
    int32_t* __restrict__ out, int N, int K)
{
    const int tile_n = blockIdx.x;
    const int lane_n = threadIdx.x;
    const int lane_k = threadIdx.y;
    const int n = tile_n * TN + lane_n;

    __shared__ int8_t sA[TK];
    int32_t acc = 0;

    const int k_tiles = K / TK;
    const int n_tiles = N / TN;
    const int groups = TK / 4;

    for (int kt = 0; kt < k_tiles; ++kt) {
        const int ltid = lane_k * TN + lane_n;
        for (int i = ltid; i < TK; i += TN * KSPLIT) sA[i] = A[kt * TK + i];
        __syncthreads();

        if (n < N) {
            const int8_t* tile = B + (static_cast<size_t>(kt) * n_tiles + tile_n) * TK * TN;
            for (int kg = lane_k; kg < groups; kg += KSPLIT) {
                const int kl = kg * 4;
                const int32_t a0 = sA[kl], a1 = sA[kl+1], a2 = sA[kl+2], a3 = sA[kl+3];
                const uint32_t p = *reinterpret_cast<const uint32_t*>(
                    tile + (kg * TN + lane_n) * 4);
                acc += a0 * (int32_t)(int8_t)(p & 0xFF)
                     + a1 * (int32_t)(int8_t)((p >> 8) & 0xFF)
                     + a2 * (int32_t)(int8_t)((p >> 16) & 0xFF)
                     + a3 * (int32_t)(int8_t)((p >> 24) & 0xFF);
            }
        }
        __syncthreads();
    }

    __shared__ int32_t partial[KSPLIT][TN];
    partial[lane_k][lane_n] = acc;
    __syncthreads();

    if (lane_k == 0 && n < N) {
        int32_t s = 0;
        #pragma unroll
        for (int i = 0; i < KSPLIT; ++i) s += partial[i][lane_n];
        out[n] = s;
    }
}

// Strategy C variant: tiled kpar with vec4 (2 cols per thread to increase work)
template <int TN, int TK, int KSPLIT>
__global__ void strat_C_tiled_kpar_v2(
    const int8_t* __restrict__ A, const int8_t* __restrict__ B,
    int32_t* __restrict__ out, int N, int K)
{
    const int tile_n = blockIdx.x;
    const int lane_n = threadIdx.x;  // 0..TN/2-1
    const int lane_k = threadIdx.y;
    const int n2 = tile_n * TN + lane_n * 2;

    __shared__ int8_t sA[TK];
    int32_t acc0 = 0, acc1 = 0;

    const int k_tiles = K / TK;
    const int n_tiles = N / TN;
    const int groups = TK / 4;

    for (int kt = 0; kt < k_tiles; ++kt) {
        const int ltid = lane_k * (TN/2) + lane_n;
        for (int i = ltid; i < TK; i += (TN/2) * KSPLIT) sA[i] = A[kt * TK + i];
        __syncthreads();

        if (n2 < N) {
            const int8_t* tile = B + (static_cast<size_t>(kt) * n_tiles + tile_n) * TK * TN;
            for (int kg = lane_k; kg < groups; kg += KSPLIT) {
                const int kl = kg * 4;
                const int32_t a0 = sA[kl], a1 = sA[kl+1], a2 = sA[kl+2], a3 = sA[kl+3];
                const uint32_t p0 = *reinterpret_cast<const uint32_t*>(
                    tile + (kg * TN + lane_n * 2) * 4);
                const uint32_t p1 = *reinterpret_cast<const uint32_t*>(
                    tile + (kg * TN + lane_n * 2 + 1) * 4);
                acc0 += a0*(int32_t)(int8_t)(p0&0xFF) + a1*(int32_t)(int8_t)((p0>>8)&0xFF)
                      + a2*(int32_t)(int8_t)((p0>>16)&0xFF) + a3*(int32_t)(int8_t)((p0>>24)&0xFF);
                acc1 += a0*(int32_t)(int8_t)(p1&0xFF) + a1*(int32_t)(int8_t)((p1>>8)&0xFF)
                      + a2*(int32_t)(int8_t)((p1>>16)&0xFF) + a3*(int32_t)(int8_t)((p1>>24)&0xFF);
            }
        }
        __syncthreads();
    }

    // Reduce: store both accumulators interleaved
    __shared__ int32_t partial[KSPLIT][TN];
    partial[lane_k][lane_n * 2]     = acc0;
    partial[lane_k][lane_n * 2 + 1] = acc1;
    __syncthreads();

    if (lane_k == 0 && n2 < N) {
        int32_t s0 = 0, s1 = 0;
        #pragma unroll
        for (int i = 0; i < KSPLIT; ++i) {
            s0 += partial[i][lane_n * 2];
            s1 += partial[i][lane_n * 2 + 1];
        }
        out[n2] = s0;
        out[n2 + 1] = s1;
    }
}

// ============================================================================
// STRATEGY D: Column-major + LDS staging — B[N][K] transposed
// Block of BN threads cooperatively loads tiles from B into LDS
// Each thread reads its own column from LDS (sequential)
// ============================================================================
template <int BN, int TK>
__global__ void strat_D_colmajor_lds(
    const int8_t* __restrict__ A, const int8_t* __restrict__ B,
    int32_t* __restrict__ out, int N, int K)
{
    const int n = blockIdx.x * BN + threadIdx.x;
    const bool active = (n < N);

    // LDS tile: BN columns × TK rows
    // We load a [BN × TK] tile into LDS using coalesced reads along K
    __shared__ int8_t sB[BN][TK];  // BN * TK bytes of LDS
    __shared__ int8_t sA[TK];

    int32_t acc = 0;

    for (int kb = 0; kb < K; kb += TK) {
        const int tk = min(TK, K - kb);

        // Load A tile
        for (int i = threadIdx.x; i < tk; i += BN) sA[i] = A[kb + i];

        // Load B tile: each thread loads its own column's TK values
        // B[N][K]: B[n][k] = B[n * K + k]
        if (active) {
            const int8_t* col = B + static_cast<size_t>(n) * K + kb;
            // Vectorized load: read 4 bytes at a time from this thread's column
            int k = 0;
            for (; k + 3 < tk; k += 4) {
                *reinterpret_cast<int32_t*>(&sB[threadIdx.x][k]) =
                    *reinterpret_cast<const int32_t*>(col + k);
            }
            for (; k < tk; ++k) {
                sB[threadIdx.x][k] = col[k];
            }
        }
        __syncthreads();

        // Compute: each thread reads from its own LDS row (sequential access in LDS)
        if (active) {
            for (int k = 0; k < tk; k += 4) {
                const int32_t a0 = sA[k], a1 = sA[k+1], a2 = sA[k+2], a3 = sA[k+3];
                acc += a0 * (int32_t)sB[threadIdx.x][k]
                     + a1 * (int32_t)sB[threadIdx.x][k+1]
                     + a2 * (int32_t)sB[threadIdx.x][k+2]
                     + a3 * (int32_t)sB[threadIdx.x][k+3];
            }
        }
        __syncthreads();
    }
    if (active) out[n] = acc;
}

// Column-major + kpar: more threads per block via K-splitting
template <int BN, int TK, int KSPLIT>
__global__ void strat_D_colmajor_lds_kpar(
    const int8_t* __restrict__ A, const int8_t* __restrict__ B,
    int32_t* __restrict__ out, int N, int K)
{
    const int lane_n = threadIdx.x;
    const int lane_k = threadIdx.y;
    const int n = blockIdx.x * BN + lane_n;
    const bool active = (n < N);

    __shared__ int8_t sB[BN][TK];
    __shared__ int8_t sA[TK];

    int32_t acc = 0;

    for (int kb = 0; kb < K; kb += TK) {
        const int tk = min(TK, K - kb);

        // Cooperative A load
        const int ltid = lane_k * BN + lane_n;
        for (int i = ltid; i < tk; i += BN * KSPLIT) sA[i] = A[kb + i];

        // Each thread loads its own column portion
        if (active) {
            const int8_t* col = B + static_cast<size_t>(n) * K + kb;
            // Split load work across k-lanes
            for (int k = lane_k; k < tk; k += KSPLIT) {
                sB[lane_n][k] = col[k];
            }
        }
        __syncthreads();

        // Each k-lane processes every KSPLIT-th element
        if (active) {
            for (int k = lane_k * 4; k < tk; k += KSPLIT * 4) {
                if (k + 3 < tk) {
                    const int32_t a0 = sA[k], a1 = sA[k+1], a2 = sA[k+2], a3 = sA[k+3];
                    acc += a0 * (int32_t)sB[lane_n][k]
                         + a1 * (int32_t)sB[lane_n][k+1]
                         + a2 * (int32_t)sB[lane_n][k+2]
                         + a3 * (int32_t)sB[lane_n][k+3];
                }
            }
        }
        __syncthreads();
    }

    __shared__ int32_t partial[KSPLIT][BN];
    partial[lane_k][lane_n] = acc;
    __syncthreads();

    if (lane_k == 0 && active) {
        int32_t s = 0;
        #pragma unroll
        for (int i = 0; i < KSPLIT; ++i) s += partial[i][lane_n];
        out[n] = s;
    }
}

// ============================================================================
// STRATEGY E: Row-major wide-vec (no repacking!)
// B[K][N], each thread reads 4-16 bytes per K-row via vec loads
// ============================================================================
template <int BLOCK>
__global__ void strat_E_rowmajor_vec4(
    const int8_t* __restrict__ A, const int8_t* __restrict__ B,
    int32_t* __restrict__ out, int N, int K)
{
    const int n4 = blockIdx.x * (BLOCK * 4) + threadIdx.x * 4;
    if (n4 >= N) return;

    __shared__ int8_t sA[256];
    int4 acc = {0,0,0,0};

    for (int kb = 0; kb < K; kb += 256) {
        const int tk = min(256, K - kb);
        for (int i = threadIdx.x; i < tk; i += BLOCK) sA[i] = A[kb + i];
        __syncthreads();

        for (int k = 0; k < tk; k += 4) {
            // 4 K-rows, loading vec4 from each
            const int32_t a0 = sA[k], a1 = sA[k+1], a2 = sA[k+2], a3 = sA[k+3];
            const uint32_t p0 = *reinterpret_cast<const uint32_t*>(B + (kb+k)*N + n4);
            const uint32_t p1 = *reinterpret_cast<const uint32_t*>(B + (kb+k+1)*N + n4);
            const uint32_t p2 = *reinterpret_cast<const uint32_t*>(B + (kb+k+2)*N + n4);
            const uint32_t p3 = *reinterpret_cast<const uint32_t*>(B + (kb+k+3)*N + n4);

            acc.x += a0*(int32_t)(int8_t)(p0&0xFF)     + a1*(int32_t)(int8_t)(p1&0xFF)
                   + a2*(int32_t)(int8_t)(p2&0xFF)     + a3*(int32_t)(int8_t)(p3&0xFF);
            acc.y += a0*(int32_t)(int8_t)((p0>>8)&0xFF) + a1*(int32_t)(int8_t)((p1>>8)&0xFF)
                   + a2*(int32_t)(int8_t)((p2>>8)&0xFF) + a3*(int32_t)(int8_t)((p3>>8)&0xFF);
            acc.z += a0*(int32_t)(int8_t)((p0>>16)&0xFF)+ a1*(int32_t)(int8_t)((p1>>16)&0xFF)
                   + a2*(int32_t)(int8_t)((p2>>16)&0xFF)+ a3*(int32_t)(int8_t)((p3>>16)&0xFF);
            acc.w += a0*(int32_t)(int8_t)((p0>>24)&0xFF)+ a1*(int32_t)(int8_t)((p1>>24)&0xFF)
                   + a2*(int32_t)(int8_t)((p2>>24)&0xFF)+ a3*(int32_t)(int8_t)((p3>>24)&0xFF);
        }
        __syncthreads();
    }
    out[n4] = acc.x; out[n4+1] = acc.y; out[n4+2] = acc.z; out[n4+3] = acc.w;
}

// Row-major vec4 + kpar (for FFN Down where K >> N)
template <int TILE_N, int KSPLIT>
__global__ void strat_E_rowmajor_vec4_kpar(
    const int8_t* __restrict__ A, const int8_t* __restrict__ B,
    int32_t* __restrict__ out, int N, int K)
{
    const int lane_n = threadIdx.x;
    const int n4 = blockIdx.x * TILE_N + lane_n * 4;
    const bool active = (n4 < N);
    const int lane_k = threadIdx.y;

    __shared__ int8_t sA[256];
    int4 acc = {0,0,0,0};

    for (int kb = 0; kb < K; kb += 256) {
        const int tk = min(256, K - kb);
        const int ltid = lane_k * (TILE_N/4) + lane_n;
        for (int i = ltid; i < tk; i += (TILE_N/4) * KSPLIT) sA[i] = A[kb + i];
        __syncthreads();

        for (int k = lane_k * 4; k < tk; k += KSPLIT * 4) {
            if (active && k + 3 < tk) {
                const int32_t a0 = sA[k], a1 = sA[k+1], a2 = sA[k+2], a3 = sA[k+3];
                const uint32_t p0 = *reinterpret_cast<const uint32_t*>(B + (kb+k)*N + n4);
                const uint32_t p1 = *reinterpret_cast<const uint32_t*>(B + (kb+k+1)*N + n4);
                const uint32_t p2 = *reinterpret_cast<const uint32_t*>(B + (kb+k+2)*N + n4);
                const uint32_t p3 = *reinterpret_cast<const uint32_t*>(B + (kb+k+3)*N + n4);

                acc.x += a0*(int32_t)(int8_t)(p0&0xFF)     + a1*(int32_t)(int8_t)(p1&0xFF)
                       + a2*(int32_t)(int8_t)(p2&0xFF)     + a3*(int32_t)(int8_t)(p3&0xFF);
                acc.y += a0*(int32_t)(int8_t)((p0>>8)&0xFF) + a1*(int32_t)(int8_t)((p1>>8)&0xFF)
                       + a2*(int32_t)(int8_t)((p2>>8)&0xFF) + a3*(int32_t)(int8_t)((p3>>8)&0xFF);
                acc.z += a0*(int32_t)(int8_t)((p0>>16)&0xFF)+ a1*(int32_t)(int8_t)((p1>>16)&0xFF)
                       + a2*(int32_t)(int8_t)((p2>>16)&0xFF)+ a3*(int32_t)(int8_t)((p3>>16)&0xFF);
                acc.w += a0*(int32_t)(int8_t)((p0>>24)&0xFF)+ a1*(int32_t)(int8_t)((p1>>24)&0xFF)
                       + a2*(int32_t)(int8_t)((p2>>24)&0xFF)+ a3*(int32_t)(int8_t)((p3>>24)&0xFF);
            }
        }
        __syncthreads();
    }

    __shared__ int4 partial[KSPLIT][TILE_N / 4];
    partial[lane_k][lane_n] = acc;
    __syncthreads();

    if (lane_k == 0 && active) {
        int4 s = {0,0,0,0};
        #pragma unroll
        for (int i = 0; i < KSPLIT; ++i) {
            s.x += partial[i][lane_n].x;
            s.y += partial[i][lane_n].y;
            s.z += partial[i][lane_n].z;
            s.w += partial[i][lane_n].w;
        }
        out[n4] = s.x; out[n4+1] = s.y; out[n4+2] = s.z; out[n4+3] = s.w;
    }
}


// ============================================================================
// Packing helpers
// ============================================================================
void pack_vnni(const int8_t* src, int8_t* dst, int N, int K)
{
    for (int k = 0; k < K; k += 4)
        for (int n = 0; n < N; ++n)
            for (int ki = 0; ki < 4; ++ki)
                dst[((k/4) * N + n) * 4 + ki] = src[(k+ki) * N + n];
}

template <int TN, int TK>
void pack_tiled_kn_vnni(const int8_t* src, int8_t* dst, int N, int K)
{
    const int kt = K/TK, nt = N/TN;
    for (int ki = 0; ki < kt; ++ki)
        for (int ni = 0; ni < nt; ++ni)
            for (int kg = 0; kg < TK/4; ++kg)
                for (int ln = 0; ln < TN; ++ln)
                    for (int kk = 0; kk < 4; ++kk) {
                        int k = ki*TK + kg*4 + kk;
                        int n = ni*TN + ln;
                        size_t di = (static_cast<size_t>(ki)*nt + ni)*TK*TN + (kg*TN+ln)*4 + kk;
                        dst[di] = src[static_cast<size_t>(k)*N + n];
                    }
}

void pack_col_major(const int8_t* src, int8_t* dst, int N, int K)
{
    for (int k = 0; k < K; ++k)
        for (int n = 0; n < N; ++n)
            dst[static_cast<size_t>(n) * K + k] = src[static_cast<size_t>(k) * N + n];
}

// ============================================================================
// Benchmark harness
// ============================================================================
struct Result { const char* name; double us; double bw; double pct; int wavefronts; };

double bench(std::function<void()> f, int warmup, int iters) {
    for (int i = 0; i < warmup; ++i) f();
    HIP_CHECK(hipDeviceSynchronize());
    hipEvent_t t0, t1;
    HIP_CHECK(hipEventCreate(&t0)); HIP_CHECK(hipEventCreate(&t1));
    HIP_CHECK(hipEventRecord(t0));
    for (int i = 0; i < iters; ++i) f();
    HIP_CHECK(hipEventRecord(t1));
    HIP_CHECK(hipEventSynchronize(t1));
    float ms; HIP_CHECK(hipEventElapsedTime(&ms, t0, t1));
    HIP_CHECK(hipEventDestroy(t0)); HIP_CHECK(hipEventDestroy(t1));
    return (ms * 1000.0) / iters;
}

void run_shape(const char* label, int N, int K, int iters)
{
    const size_t wb = static_cast<size_t>(N) * K;
    const double total_bytes = wb + K + N * 4.0;
    const double theory_us = (total_bytes / 1e9) / (MI50_PEAK_BW / 1e6); // theoretical min

    printf("\n");
    printf("╔════════════════════════════════════════════════════════════════════════════════╗\n");
    printf("║  %-12s  N=%-6d K=%-6d  Weight=%.1fMB  Theory=%.1fus                     ║\n",
           label, N, K, wb/1e6, theory_us);
    printf("╠════════════════════════════════════════════════════════════════════════════════╣\n");

    std::vector<int8_t> h_B(wb), h_A(K);
    for (size_t i = 0; i < wb; ++i) h_B[i] = (int8_t)((i*7+13)%251-125);
    for (int i = 0; i < K; ++i) h_A[i] = (int8_t)((i*3+7)%127-63);

    // Prepare all layouts
    std::vector<int8_t> h_vnni(wb);
    pack_vnni(h_B.data(), h_vnni.data(), N, K);

    std::vector<int8_t> h_col(wb);
    pack_col_major(h_B.data(), h_col.data(), N, K);

    // Tiled layouts
    auto make_tiled = [&](auto packer) -> std::vector<int8_t> {
        std::vector<int8_t> v(wb);
        packer(h_B.data(), v.data(), N, K);
        return v;
    };

    // Device allocs
    int8_t *d_A, *d_row, *d_vnni, *d_col;
    int32_t *d_out;
    HIP_CHECK(hipMalloc(&d_A, K));
    HIP_CHECK(hipMalloc(&d_row, wb));
    HIP_CHECK(hipMalloc(&d_vnni, wb));
    HIP_CHECK(hipMalloc(&d_col, wb));
    HIP_CHECK(hipMalloc(&d_out, N * 4));
    HIP_CHECK(hipMemcpy(d_A, h_A.data(), K, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_row, h_B.data(), wb, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_vnni, h_vnni.data(), wb, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_col, h_col.data(), wb, hipMemcpyHostToDevice));

    auto alloc_tiled = [&](const std::vector<int8_t>& h) -> int8_t* {
        int8_t* d;
        HIP_CHECK(hipMalloc(&d, wb));
        HIP_CHECK(hipMemcpy(d, h.data(), wb, hipMemcpyHostToDevice));
        return d;
    };

    std::vector<Result> results;
    const int W = 5;

    auto add = [&](const char* name, double us, int wf) {
        double bw = (total_bytes/1e9)/(us/1e6);
        results.push_back({name, us, bw, 100.0*bw/MI50_PEAK_BW, wf});
    };

    // ── Strategy A: Flat VNNI baseline ──────────────────────────────────────
    {
        int blk = 256, grid = (N+blk-1)/blk;
        add("A: flat_vnni_256",
            bench([&]{ hipLaunchKernelGGL(HIP_KERNEL_NAME(strat_A_flat_vnni<256>),
                dim3(grid), dim3(blk), 0, 0, d_A, d_vnni, d_out, N, K); }, W, iters),
            grid * (blk/64));
    }
    if (N % 4 == 0) {
        int blk = 64, tile = blk*4;
        int grid = (N+tile-1)/tile;
        add("A: flat_vnni_v4_64",
            bench([&]{ hipLaunchKernelGGL(HIP_KERNEL_NAME(strat_A_flat_vnni_vec4<64>),
                dim3(grid), dim3(blk), 0, 0, d_A, d_vnni, d_out, N, K); }, W, iters),
            grid);
    }
    if (N % 4 == 0) {
        int blk = 128, tile = blk*4;
        int grid = (N+tile-1)/tile;
        add("A: flat_vnni_v4_128",
            bench([&]{ hipLaunchKernelGGL(HIP_KERNEL_NAME(strat_A_flat_vnni_vec4<128>),
                dim3(grid), dim3(blk), 0, 0, d_A, d_vnni, d_out, N, K); }, W, iters),
            grid * (blk/64));
    }
    // Flat VNNI kpar (for K >= N shapes)
    if (N % 4 == 0 && K >= N) {
        constexpr int TN = 64, KS = 4;
        int grid = (N + TN - 1) / TN;
        add("A: flat_vnni_kp64_s4",
            bench([&]{ hipLaunchKernelGGL(HIP_KERNEL_NAME(strat_A_flat_vnni_kpar<TN,KS>),
                dim3(grid), dim3(TN/4, KS), 0, 0, d_A, d_vnni, d_out, N, K); }, W, iters),
            grid * ((TN/4*KS+63)/64));
    }
    if (N % 4 == 0 && K >= N) {
        constexpr int TN = 128, KS = 16;
        int grid = (N + TN - 1) / TN;
        add("A: flat_vnni_kp128_s16",
            bench([&]{ hipLaunchKernelGGL(HIP_KERNEL_NAME(strat_A_flat_vnni_kpar<TN,KS>),
                dim3(grid), dim3(TN/4, KS), 0, 0, d_A, d_vnni, d_out, N, K); }, W, iters),
            grid * ((TN/4*KS+63)/64));
    }

    // ── Strategy B: Tiled KN VNNI ───────────────────────────────────────────
    auto test_tiled = [&](const char* name, int tn, int tk, int8_t* d_t, auto kernel) {
        int grid = N / tn;
        add(name, bench([&]{ hipLaunchKernelGGL(kernel,
            dim3(grid), dim3(tn), 0, 0, d_A, d_t, d_out, N, K); }, W, iters),
            grid * ((tn+63)/64));
    };

    // Test a variety of TN×TK combinations
    struct TileConfig { int tn; int tk; };
    TileConfig tiles[] = {{64,64},{64,128},{64,256},{128,64},{128,128},{128,256},{256,128},{256,256}};

    for (auto& t : tiles) {
        if (N % t.tn != 0 || K % t.tk != 0) continue;
        auto h = make_tiled([&](auto s, auto d, auto n, auto k){
            // Call the right template
            if (t.tn==64 && t.tk==64) pack_tiled_kn_vnni<64,64>(s,d,n,k);
            else if (t.tn==64 && t.tk==128) pack_tiled_kn_vnni<64,128>(s,d,n,k);
            else if (t.tn==64 && t.tk==256) pack_tiled_kn_vnni<64,256>(s,d,n,k);
            else if (t.tn==128 && t.tk==64) pack_tiled_kn_vnni<128,64>(s,d,n,k);
            else if (t.tn==128 && t.tk==128) pack_tiled_kn_vnni<128,128>(s,d,n,k);
            else if (t.tn==128 && t.tk==256) pack_tiled_kn_vnni<128,256>(s,d,n,k);
            else if (t.tn==256 && t.tk==128) pack_tiled_kn_vnni<256,128>(s,d,n,k);
            else if (t.tn==256 && t.tk==256) pack_tiled_kn_vnni<256,256>(s,d,n,k);
        });
        int8_t* d_t = alloc_tiled(h);

        char buf[64];
        snprintf(buf, sizeof(buf), "B: tiled_%dx%d", t.tn, t.tk);

        int grid = N / t.tn;
        int tn = t.tn;
        // Dispatch the right kernel template
        if (t.tn==64 && t.tk==64) add(strdup(buf), bench([&]{ hipLaunchKernelGGL(HIP_KERNEL_NAME(strat_B_tiled<64,64>), dim3(grid), dim3(tn), 0, 0, d_A, d_t, d_out, N, K); }, W, iters), grid*((tn+63)/64));
        else if (t.tn==64 && t.tk==128) add(strdup(buf), bench([&]{ hipLaunchKernelGGL(HIP_KERNEL_NAME(strat_B_tiled<64,128>), dim3(grid), dim3(tn), 0, 0, d_A, d_t, d_out, N, K); }, W, iters), grid*((tn+63)/64));
        else if (t.tn==64 && t.tk==256) add(strdup(buf), bench([&]{ hipLaunchKernelGGL(HIP_KERNEL_NAME(strat_B_tiled<64,256>), dim3(grid), dim3(tn), 0, 0, d_A, d_t, d_out, N, K); }, W, iters), grid*((tn+63)/64));
        else if (t.tn==128 && t.tk==64) add(strdup(buf), bench([&]{ hipLaunchKernelGGL(HIP_KERNEL_NAME(strat_B_tiled<128,64>), dim3(grid), dim3(tn), 0, 0, d_A, d_t, d_out, N, K); }, W, iters), grid*((tn+63)/64));
        else if (t.tn==128 && t.tk==128) add(strdup(buf), bench([&]{ hipLaunchKernelGGL(HIP_KERNEL_NAME(strat_B_tiled<128,128>), dim3(grid), dim3(tn), 0, 0, d_A, d_t, d_out, N, K); }, W, iters), grid*((tn+63)/64));
        else if (t.tn==128 && t.tk==256) add(strdup(buf), bench([&]{ hipLaunchKernelGGL(HIP_KERNEL_NAME(strat_B_tiled<128,256>), dim3(grid), dim3(tn), 0, 0, d_A, d_t, d_out, N, K); }, W, iters), grid*((tn+63)/64));
        else if (t.tn==256 && t.tk==128) add(strdup(buf), bench([&]{ hipLaunchKernelGGL(HIP_KERNEL_NAME(strat_B_tiled<256,128>), dim3(grid), dim3(tn), 0, 0, d_A, d_t, d_out, N, K); }, W, iters), grid*((tn+63)/64));
        else if (t.tn==256 && t.tk==256) add(strdup(buf), bench([&]{ hipLaunchKernelGGL(HIP_KERNEL_NAME(strat_B_tiled<256,256>), dim3(grid), dim3(tn), 0, 0, d_A, d_t, d_out, N, K); }, W, iters), grid*((tn+63)/64));

        HIP_CHECK(hipFree(d_t));
    }

    // ── Strategy C: Tiled + K-parallel ──────────────────────────────────────
    // Most promising combos per shape class
    struct TKPConfig { int tn; int tk; int ks; };
    TKPConfig tkp_configs[] = {
        {64,64,2}, {64,64,4}, {64,128,2}, {64,128,4},
        {64,256,4}, {64,256,8},
        {128,128,2}, {128,128,4}, {128,256,2}, {128,256,4},
        {256,256,2}, {256,256,4},
    };

    for (auto& c : tkp_configs) {
        if (N % c.tn != 0 || K % c.tk != 0) continue;
        if (c.tn * c.ks > 1024) continue; // block size limit

        // Reuse tiled layouts
        auto h = make_tiled([&](auto s, auto d, auto n, auto k){
            if (c.tn==64 && c.tk==64) pack_tiled_kn_vnni<64,64>(s,d,n,k);
            else if (c.tn==64 && c.tk==128) pack_tiled_kn_vnni<64,128>(s,d,n,k);
            else if (c.tn==64 && c.tk==256) pack_tiled_kn_vnni<64,256>(s,d,n,k);
            else if (c.tn==128 && c.tk==128) pack_tiled_kn_vnni<128,128>(s,d,n,k);
            else if (c.tn==128 && c.tk==256) pack_tiled_kn_vnni<128,256>(s,d,n,k);
            else if (c.tn==256 && c.tk==256) pack_tiled_kn_vnni<256,256>(s,d,n,k);
        });
        int8_t* d_t = alloc_tiled(h);

        char buf[64];
        snprintf(buf, sizeof(buf), "C: tkp_%dx%d_s%d", c.tn, c.tk, c.ks);

        int grid = N / c.tn;
        int wf = grid * ((c.tn * c.ks + 63) / 64);

        #define LAUNCH_C(TN,TK,KS) \
            add(strdup(buf), bench([&]{ hipLaunchKernelGGL( \
                HIP_KERNEL_NAME(strat_C_tiled_kpar<TN,TK,KS>), \
                dim3(grid), dim3(TN, KS), 0, 0, d_A, d_t, d_out, N, K); }, W, iters), wf)

        if      (c.tn==64  && c.tk==64  && c.ks==2) LAUNCH_C(64,64,2);
        else if (c.tn==64  && c.tk==64  && c.ks==4) LAUNCH_C(64,64,4);
        else if (c.tn==64  && c.tk==128 && c.ks==2) LAUNCH_C(64,128,2);
        else if (c.tn==64  && c.tk==128 && c.ks==4) LAUNCH_C(64,128,4);
        else if (c.tn==64  && c.tk==256 && c.ks==4) LAUNCH_C(64,256,4);
        else if (c.tn==64  && c.tk==256 && c.ks==8) LAUNCH_C(64,256,8);
        else if (c.tn==128 && c.tk==128 && c.ks==2) LAUNCH_C(128,128,2);
        else if (c.tn==128 && c.tk==128 && c.ks==4) LAUNCH_C(128,128,4);
        else if (c.tn==128 && c.tk==256 && c.ks==2) LAUNCH_C(128,256,2);
        else if (c.tn==128 && c.tk==256 && c.ks==4) LAUNCH_C(128,256,4);
        else if (c.tn==256 && c.tk==256 && c.ks==2) LAUNCH_C(256,256,2);
        else if (c.tn==256 && c.tk==256 && c.ks==4) LAUNCH_C(256,256,4);
        #undef LAUNCH_C

        HIP_CHECK(hipFree(d_t));
    }

    // Strategy C v2: tiled kpar with 2-cols-per-thread
    if (N % 128 == 0 && K % 128 == 0) {
        auto h = make_tiled([](auto s, auto d, auto n, auto k){ pack_tiled_kn_vnni<128,128>(s,d,n,k); });
        int8_t* d_t = alloc_tiled(h);
        int grid = N / 128;
        add("C: tkp_v2_128x128_s4",
            bench([&]{ hipLaunchKernelGGL(HIP_KERNEL_NAME(strat_C_tiled_kpar_v2<128,128,4>),
                dim3(grid), dim3(64, 4), 0, 0, d_A, d_t, d_out, N, K); }, W, iters),
            grid * 4);
        add("C: tkp_v2_128x128_s2",
            bench([&]{ hipLaunchKernelGGL(HIP_KERNEL_NAME(strat_C_tiled_kpar_v2<128,128,2>),
                dim3(grid), dim3(64, 2), 0, 0, d_A, d_t, d_out, N, K); }, W, iters),
            grid * 2);
        HIP_CHECK(hipFree(d_t));
    }
    if (N % 64 == 0 && K % 256 == 0) {
        auto h = make_tiled([](auto s, auto d, auto n, auto k){ pack_tiled_kn_vnni<64,256>(s,d,n,k); });
        int8_t* d_t = alloc_tiled(h);
        int grid = N / 64;
        add("C: tkp_v2_64x256_s4",
            bench([&]{ hipLaunchKernelGGL(HIP_KERNEL_NAME(strat_C_tiled_kpar_v2<64,256,4>),
                dim3(grid), dim3(32, 4), 0, 0, d_A, d_t, d_out, N, K); }, W, iters),
            grid * 2);
        HIP_CHECK(hipFree(d_t));
    }

    // ── Strategy D: Column-major + LDS ──────────────────────────────────────
    // LDS limit: ~64KB per block on MI50. BN*TK must fit.
    // BN=64, TK=256 → 16KB. BN=128, TK=128 → 16KB. BN=256, TK=64 → 16KB.
    {
        constexpr int BN = 64, TK = 256;
        int grid = (N + BN - 1) / BN;
        add("D: colmaj_lds_64x256",
            bench([&]{ hipLaunchKernelGGL(HIP_KERNEL_NAME(strat_D_colmajor_lds<BN,TK>),
                dim3(grid), dim3(BN), 0, 0, d_A, d_col, d_out, N, K); }, W, iters),
            grid);
    }
    {
        constexpr int BN = 128, TK = 128;
        int grid = (N + BN - 1) / BN;
        add("D: colmaj_lds_128x128",
            bench([&]{ hipLaunchKernelGGL(HIP_KERNEL_NAME(strat_D_colmajor_lds<BN,TK>),
                dim3(grid), dim3(BN), 0, 0, d_A, d_col, d_out, N, K); }, W, iters),
            grid * 2);
    }
    {
        constexpr int BN = 256, TK = 64;
        int grid = (N + BN - 1) / BN;
        add("D: colmaj_lds_256x64",
            bench([&]{ hipLaunchKernelGGL(HIP_KERNEL_NAME(strat_D_colmajor_lds<BN,TK>),
                dim3(grid), dim3(BN), 0, 0, d_A, d_col, d_out, N, K); }, W, iters),
            grid * 4);
    }
    // Col-major + kpar
    if (K >= N) {
        constexpr int BN = 64, TK = 256, KS = 4;
        int grid = (N + BN - 1) / BN;
        add("D: colmaj_kp_64x256_s4",
            bench([&]{ hipLaunchKernelGGL(HIP_KERNEL_NAME(strat_D_colmajor_lds_kpar<BN,TK,KS>),
                dim3(grid), dim3(BN, KS), 0, 0, d_A, d_col, d_out, N, K); }, W, iters),
            grid * 4);
    }

    // ── Strategy E: Row-major vec4 (no repacking) ───────────────────────────
    if (N % 4 == 0) {
        int blk = 64, tile = blk * 4;
        int grid = (N + tile - 1) / tile;
        add("E: rowmaj_v4_64",
            bench([&]{ hipLaunchKernelGGL(HIP_KERNEL_NAME(strat_E_rowmajor_vec4<64>),
                dim3(grid), dim3(blk), 0, 0, d_A, d_row, d_out, N, K); }, W, iters),
            grid);
    }
    if (N % 4 == 0) {
        int blk = 128, tile = blk * 4;
        int grid = (N + tile - 1) / tile;
        add("E: rowmaj_v4_128",
            bench([&]{ hipLaunchKernelGGL(HIP_KERNEL_NAME(strat_E_rowmajor_vec4<128>),
                dim3(grid), dim3(blk), 0, 0, d_A, d_row, d_out, N, K); }, W, iters),
            grid * 2);
    }
    // Row-major vec4 + kpar
    if (N % 4 == 0 && K >= N) {
        constexpr int TN = 128, KS = 16;
        int grid = (N + TN - 1) / TN;
        add("E: rowmaj_v4_kp128_s16",
            bench([&]{ hipLaunchKernelGGL(HIP_KERNEL_NAME(strat_E_rowmajor_vec4_kpar<TN,KS>),
                dim3(grid), dim3(TN/4, KS), 0, 0, d_A, d_row, d_out, N, K); }, W, iters),
            grid * ((TN/4*KS+63)/64));
    }

    // ── Print Results ───────────────────────────────────────────────────────
    // Sort by time
    std::sort(results.begin(), results.end(), [](auto& a, auto& b){ return a.us < b.us; });

    printf("║  %-30s %8s %9s %7s %6s ║\n", "Strategy", "Time(us)", "BW(GB/s)", "%Peak", "Waves");
    printf("║  %-30s %8s %9s %7s %6s ║\n", "──────────────────────────────", "────────", "─────────", "───────", "──────");
    for (auto& r : results) {
        const char* star = (r.us <= results[0].us * 1.05) ? " ★" : "";
        printf("║  %-30s %8.1f %9.1f %6.1f%% %6d%s ║\n",
               r.name, r.us, r.bw, r.pct, r.wavefronts, star);
    }
    printf("║  %-30s %8.1f %9s %6s%% %6s ║\n", "THEORETICAL MINIMUM", theory_us, "1000.0", "100.0", "-");
    printf("╚════════════════════════════════════════════════════════════════════════════════╝\n");

    HIP_CHECK(hipFree(d_A)); HIP_CHECK(hipFree(d_row));
    HIP_CHECK(hipFree(d_vnni)); HIP_CHECK(hipFree(d_col));
    HIP_CHECK(hipFree(d_out));
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

    const int iters = 100;

    // Run only the shapes specified, or all 3
    int shape = argc > 2 ? atoi(argv[2]) : -1;

    if (shape == -1 || shape == 0) run_shape("Gate/Up",  18944, 3584, iters);
    if (shape == -1 || shape == 1) run_shape("Q/Wo",     3584,  3584, iters);
    if (shape == -1 || shape == 2) run_shape("FFN Down", 3584, 18944, iters);

    return 0;
}
