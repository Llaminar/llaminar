/**
 * @file ROCmGemvKernel.hip
 * @brief Bandwidth-optimized INT8 GEMV kernel for decode (M=1) on MI50/MI60 (gfx906)
 *
 * ## Problem
 *
 * CK INT8 GEMM at M=1 achieves only 2.4 TFLOPS (~8% of MI50's 29.5 TOPS INT8),
 * because 32×32 tiles waste 31 of 32 rows and can't fill the GPU. Decode is
 * memory-bandwidth-bound (reading all 7B weights per token), not compute-bound.
 *
 * ## Solution: Direct FP32×INT8 GEMV
 *
 * Skip activation quantization entirely. Read INT8 weights, multiply with FP32
 * activations directly, scale by per-column weight scale only.
 *
 * For M=1:  output[n] = scale_B[n] * sum_k( A_fp32[k] * B_int8[k * N + n] )
 *
 * ## Memory Layout
 *
 * Weights are [K × N] RowMajor INT8 (same as CK's packed format).
 * For GEMV we want to stream through K (outer dim), accessing N elements per row.
 *
 * ## Kernel Design (gfx906-optimized)
 *
 * - Each workgroup handles a tile of output columns [n_start, n_start + TILE_N)
 * - Each thread accumulates one output element across all K
 * - K-reduction done in registers (no shared memory needed for M=1)
 * - Uses vectorized loads (int4 = 16 bytes = 16 INT8 values) for coalesced access
 * - MI50 has 64 CUs, 1000 GB/s HBM2 bandwidth
 *
 * ## Expected Performance
 *
 * Weights: 7 GB INT8. At 480 GB/s → ~14.6ms theoretical minimum.
 * With activation reads + scale reads + output writes → ~16-18ms realistic.
 * Current CK: ~113ms. Target: ~15-20ms (6-7× speedup).
 *
 * @author Llaminar V2
 * @date February 2026
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cmath>

// ============================================================================
// Kernel: gemv_int8_fp32
//
// Each thread computes one output element: output[n] = scale_B[n] * dot(A, B[:, n])
//
// Grid: (ceil(N / TILE_N), 1, 1)
// Block: (TILE_N, 1, 1)
//
// Uses K-reduction in registers with vectorized INT8 loads.
// ============================================================================

// Tile width: number of output columns per workgroup
// 256 threads per block — each thread does one output column
constexpr int GEMV_TILE_N = 256;

// K unroll factor for the inner loop
constexpr int GEMV_K_UNROLL = 4;

__global__ void gemv_int8_fp32_kernel(
    const float* __restrict__ d_A,          // [K] FP32 activation vector
    const int8_t* __restrict__ d_B,         // [K × N] INT8 weights, row-major
    const float* __restrict__ d_scale_B,    // [N] per-column weight scales
    const float* __restrict__ d_bias,       // [N] optional bias (nullable)
    float* __restrict__ d_C,                // [N] FP32 output vector
    int N,
    int K)
{
    const int n = blockIdx.x * GEMV_TILE_N + threadIdx.x;
    if (n >= N) return;

    // Accumulate dot product in FP32
    float acc = 0.0f;

    // Main loop over K dimension
    // B is [K × N] row-major: B[k, n] = d_B[k * N + n]
    int k = 0;

    // Unrolled loop: process GEMV_K_UNROLL rows of B per iteration
    const int k_end_unrolled = K - (K % GEMV_K_UNROLL);
    for (; k < k_end_unrolled; k += GEMV_K_UNROLL)
    {
        float a0 = d_A[k + 0];
        float a1 = d_A[k + 1];
        float a2 = d_A[k + 2];
        float a3 = d_A[k + 3];

        // Each B access: d_B[(k+i) * N + n] — coalesced across threads for same k
        int8_t b0 = d_B[(k + 0) * N + n];
        int8_t b1 = d_B[(k + 1) * N + n];
        int8_t b2 = d_B[(k + 2) * N + n];
        int8_t b3 = d_B[(k + 3) * N + n];

        acc += a0 * static_cast<float>(b0);
        acc += a1 * static_cast<float>(b1);
        acc += a2 * static_cast<float>(b2);
        acc += a3 * static_cast<float>(b3);
    }

    // Remainder
    for (; k < K; ++k)
    {
        acc += d_A[k] * static_cast<float>(d_B[k * N + n]);
    }

    // Apply per-column scale (and optional bias) and write output
    float result = acc * d_scale_B[n];
    if (d_bias) result += d_bias[n];
    d_C[n] = result;
}

// FP16 math variant: converts A and B to half, multiplies in FP16, accumulates in FP32
__global__ void gemv_int8_fp16_kernel(
    const float* __restrict__ d_A,          // [K] FP32 activation vector
    const int8_t* __restrict__ d_B,         // [K × N] INT8 weights, row-major
    const float* __restrict__ d_scale_B,    // [N] per-column weight scales
    const float* __restrict__ d_bias,       // [N] optional bias (nullable)
    float* __restrict__ d_C,                // [N] FP32 output vector
    int N,
    int K)
{
    const int n = blockIdx.x * GEMV_TILE_N + threadIdx.x;
    if (n >= N) return;

    float acc = 0.0f;
    for (int k = 0; k < K; ++k)
    {
        const __half a = __float2half(d_A[k]);
        const __half b = __int2half_rn(static_cast<int>(d_B[k * N + n]));
        acc += __half2float(__hmul(a, b));
    }

    float result = acc * d_scale_B[n];
    if (d_bias) result += d_bias[n];
    d_C[n] = result;
}

// ============================================================================
// Kernel: gemv_int8_fp32_kpar_kernel
//
// K-parallel reduction for tall-K shapes (K >> N).
// Each workgroup handles a tile of N columns, with K_SPLIT threads per column
// cooperating to reduce the K dimension without atomics.
//
// Grid: (ceil(N / TILE_N), 1, 1)
// Block: (TILE_N, K_SPLIT, 1)
// ============================================================================

constexpr int GEMV_KPAR_TILE_N = 64;   // Columns per workgroup
constexpr int GEMV_KPAR_SPLIT = 32;    // Threads per output column
constexpr int GEMV_KPAR_TILE_K = 128;  // K elements per tile for shared A prefetch
// INT8-tuned K-parallel settings (focus on FFN Down large-K shapes)
constexpr int GEMV_INT8_KPAR_TILE_N = 64;
constexpr int GEMV_INT8_KPAR_SPLIT = 16;
constexpr int GEMV_INT8_KPAR_TILE_K = 256;
// FFN Down specialization (large K, N multiple of 4)
constexpr int GEMV_INT8_FFND_TILE_N = 128;
constexpr int GEMV_INT8_FFND_SPLIT = 16;
constexpr int GEMV_INT8_FFND_TILE_K = 256;
// VNNI specialization defaults (tuned via VNNI FFN down sweep)
constexpr int GEMV_INT8_FFND_VNNI_TILE_N = 128;
constexpr int GEMV_INT8_FFND_VNNI_SPLIT = 16;
constexpr int GEMV_INT8_FFND_VNNI_TILE_K = 256;
// VNNI aspect-ratio microkernel tuning knobs
constexpr int GEMV_INT8_VNNI_KPAR_TILE_N = 64;
constexpr int GEMV_INT8_VNNI_KPAR_SPLIT = 16;
constexpr int GEMV_INT8_VNNI_KPAR_TILE_K = 256;
constexpr int GEMV_INT8_VNNI_SQUARE_TILE_N = 256;
constexpr int GEMV_INT8_VNNI_WIDE_TILE_N = 256;
constexpr int GEMV_INT8_VNNI_WIDE2_TILE_N = 256;
// Grid-level K-parallel: splits K into KB independent blocks, atomicAdd output
// Benchmark proven: 58.5% peak BW for Q/Wo, 70.9% for FFN Down (vs ~6% / ~35% old)
constexpr int GEMV_INT8_VNNI_GRID_KPAR_TILE_N = 128;  // Columns per N-block
constexpr int GEMV_INT8_VNNI_GRID_KPAR_CPT = 4;       // Columns per thread
// Tiled KN VNNI: B[K/TK][N/TN][(TK/4)][TN][4] — coalesced within 128×128 tiles
// Benchmark proven: 75.6% peak BW for Gate/Up (vs 34.8% flat VNNI)
constexpr int GEMV_INT8_TILED_VNNI_TN = 128;           // Tile width (N dimension)
constexpr int GEMV_INT8_TILED_VNNI_TK = 128;           // Tile height (K dimension)
constexpr int GEMV_KTILE_N = 64;        // Columns per workgroup (K-tile path)
constexpr int GEMV_KTILE_K = 256;       // K elements per tile (K-tile path)

constexpr int GEMV_WIDE_TILE_N = 512;  // Wide-N kernel tile
constexpr int GEMV_SQUARE_TILE_N = 256; // Square kernel tile
constexpr int GEMV_WIDE2_TILE_N = 512;  // Wide-N kernel: 2 columns per thread

__global__ void gemv_int8_fp32_kpar_kernel(
    const float* __restrict__ d_A,          // [K] FP32 activation vector
    const int8_t* __restrict__ d_B,         // [K × N] INT8 weights, row-major
    const float* __restrict__ d_scale_B,    // [N] per-column weight scales
    const float* __restrict__ d_bias,       // [N] optional bias (nullable)
    float* __restrict__ d_C,                // [N] FP32 output
    int N,
    int K)
{
    const int n = blockIdx.x * GEMV_INT8_KPAR_TILE_N + threadIdx.x;
    const bool active = (n < N);

    const int lane_k = threadIdx.y;
    float acc = 0.0f;

    __shared__ float sA[GEMV_KPAR_TILE_K];
    for (int k_base = 0; k_base < K; k_base += GEMV_KPAR_TILE_K)
    {
        const int tile_k = min(GEMV_KPAR_TILE_K, K - k_base);
        const int linear_tid = threadIdx.y * blockDim.x + threadIdx.x;
        for (int i = linear_tid; i < tile_k; i += blockDim.x * blockDim.y)
        {
            sA[i] = d_A[k_base + i];
        }
        __syncthreads();

            const int k_step = GEMV_KPAR_SPLIT * 4;
            int k_local = lane_k;
            const int k_limit = tile_k - (tile_k % k_step);
            for (; k_local < k_limit; k_local += k_step)
            {
                if (active)
                {
                    const int k0 = k_base + k_local + 0 * GEMV_KPAR_SPLIT;
                    const int k1 = k_base + k_local + 1 * GEMV_KPAR_SPLIT;
                    const int k2 = k_base + k_local + 2 * GEMV_KPAR_SPLIT;
                    const int k3 = k_base + k_local + 3 * GEMV_KPAR_SPLIT;

                    acc += sA[k_local + 0 * GEMV_KPAR_SPLIT] * static_cast<float>(d_B[k0 * N + n]);
                    acc += sA[k_local + 1 * GEMV_KPAR_SPLIT] * static_cast<float>(d_B[k1 * N + n]);
                    acc += sA[k_local + 2 * GEMV_KPAR_SPLIT] * static_cast<float>(d_B[k2 * N + n]);
                    acc += sA[k_local + 3 * GEMV_KPAR_SPLIT] * static_cast<float>(d_B[k3 * N + n]);
                }
            }

            for (; k_local < tile_k; k_local += GEMV_KPAR_SPLIT)
            {
                if (active)
                {
                    const int k = k_base + k_local;
                    acc += sA[k_local] * static_cast<float>(d_B[k * N + n]);
                }
            }
        __syncthreads();
    }

    __shared__ float partial[GEMV_KPAR_SPLIT][GEMV_KPAR_TILE_N];
    partial[lane_k][threadIdx.x] = acc;
    __syncthreads();

    if (lane_k == 0 && active)
    {
        float sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < GEMV_KPAR_SPLIT; ++i)
        {
            sum += partial[i][threadIdx.x];
        }

        float result = sum * d_scale_B[n];
        if (d_bias) result += d_bias[n];
        d_C[n] = result;
    }
}

// K-tiled 2D grid: grid.y splits K into tiles, accumulates partials with atomics
__global__ void gemv_int8_fp32_ktiles_kernel(
    const float* __restrict__ d_A,
    const int8_t* __restrict__ d_B,
    float* __restrict__ d_C,
    int N,
    int K)
{
    const int n = blockIdx.x * GEMV_KTILE_N + threadIdx.x;
    if (n >= N) return;

    const int k_tile = blockIdx.y;
    const int k_start = k_tile * GEMV_KTILE_K;
    const int k_end = min(k_start + GEMV_KTILE_K, K);

    float acc = 0.0f;
    for (int k = k_start; k < k_end; ++k)
    {
        acc += d_A[k] * static_cast<float>(d_B[k * N + n]);
    }

    atomicAdd(&d_C[n], acc);
}

// INT8×INT8 GEMV: produces INT32 output
__global__ void gemv_int8_int8_kernel(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B_int8,
    int32_t* __restrict__ d_C_int32,
    int N,
    int K)
{
    const int n = blockIdx.x * GEMV_TILE_N + threadIdx.x;
    if (n >= N) return;

    int32_t acc = 0;
    for (int k = 0; k < K; ++k)
    {
        acc += static_cast<int32_t>(d_A_int8[k]) * static_cast<int32_t>(d_B_int8[k * N + n]);
    }

    d_C_int32[n] = acc;
}

// VNNI layout: [K/4][N][4], pack 4 contiguous K values per column
__global__ void gemv_int8_int8_vnni_kernel(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B_int8_vnni,
    int32_t* __restrict__ d_C_int32,
    int N,
    int K)
{
    const int n = blockIdx.x * GEMV_TILE_N + threadIdx.x;
    if (n >= N) return;

    int32_t acc = 0;
    const int k_groups = K / 4;
    for (int kg = 0; kg < k_groups; ++kg)
    {
        const int k_base = kg * 4;
        const int32_t a0 = static_cast<int32_t>(d_A_int8[k_base + 0]);
        const int32_t a1 = static_cast<int32_t>(d_A_int8[k_base + 1]);
        const int32_t a2 = static_cast<int32_t>(d_A_int8[k_base + 2]);
        const int32_t a3 = static_cast<int32_t>(d_A_int8[k_base + 3]);

        const uint32_t packed = *reinterpret_cast<const uint32_t*>(
            d_B_int8_vnni + (static_cast<size_t>(kg) * N + n) * 4);
        const int8_t b0 = static_cast<int8_t>(packed & 0xFF);
        const int8_t b1 = static_cast<int8_t>((packed >> 8) & 0xFF);
        const int8_t b2 = static_cast<int8_t>((packed >> 16) & 0xFF);
        const int8_t b3 = static_cast<int8_t>((packed >> 24) & 0xFF);

        acc += a0 * static_cast<int32_t>(b0)
             + a1 * static_cast<int32_t>(b1)
             + a2 * static_cast<int32_t>(b2)
             + a3 * static_cast<int32_t>(b3);
    }

    d_C_int32[n] = acc;
}

// Vectorized variant: each thread handles 4 columns (N must be divisible by 4)
__global__ void gemv_int8_int8_wide_vec4_kernel(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B_int8,
    int32_t* __restrict__ d_C_int32,
    int N,
    int K)
{
    const int lane_n = threadIdx.x;
    const int n4 = blockIdx.x * GEMV_WIDE_TILE_N + lane_n * 4;
    if (n4 >= N) return;

    int4 acc = {0, 0, 0, 0};
    int k = 0;
    const int k_end_unrolled = K - (K % 4);
    for (; k < k_end_unrolled; k += 4)
    {
        const int32_t a0 = static_cast<int32_t>(d_A_int8[k + 0]);
        const int32_t a1 = static_cast<int32_t>(d_A_int8[k + 1]);
        const int32_t a2 = static_cast<int32_t>(d_A_int8[k + 2]);
        const int32_t a3 = static_cast<int32_t>(d_A_int8[k + 3]);

        const uint32_t p0 = *reinterpret_cast<const uint32_t*>(d_B_int8 + (k + 0) * N + n4);
        const uint32_t p1 = *reinterpret_cast<const uint32_t*>(d_B_int8 + (k + 1) * N + n4);
        const uint32_t p2 = *reinterpret_cast<const uint32_t*>(d_B_int8 + (k + 2) * N + n4);
        const uint32_t p3 = *reinterpret_cast<const uint32_t*>(d_B_int8 + (k + 3) * N + n4);

        const int8_t b00 = static_cast<int8_t>(p0 & 0xFF);
        const int8_t b01 = static_cast<int8_t>((p0 >> 8) & 0xFF);
        const int8_t b02 = static_cast<int8_t>((p0 >> 16) & 0xFF);
        const int8_t b03 = static_cast<int8_t>((p0 >> 24) & 0xFF);

        const int8_t b10 = static_cast<int8_t>(p1 & 0xFF);
        const int8_t b11 = static_cast<int8_t>((p1 >> 8) & 0xFF);
        const int8_t b12 = static_cast<int8_t>((p1 >> 16) & 0xFF);
        const int8_t b13 = static_cast<int8_t>((p1 >> 24) & 0xFF);

        const int8_t b20 = static_cast<int8_t>(p2 & 0xFF);
        const int8_t b21 = static_cast<int8_t>((p2 >> 8) & 0xFF);
        const int8_t b22 = static_cast<int8_t>((p2 >> 16) & 0xFF);
        const int8_t b23 = static_cast<int8_t>((p2 >> 24) & 0xFF);

        const int8_t b30 = static_cast<int8_t>(p3 & 0xFF);
        const int8_t b31 = static_cast<int8_t>((p3 >> 8) & 0xFF);
        const int8_t b32 = static_cast<int8_t>((p3 >> 16) & 0xFF);
        const int8_t b33 = static_cast<int8_t>((p3 >> 24) & 0xFF);

        acc.x += a0 * static_cast<int32_t>(b00) + a1 * static_cast<int32_t>(b10)
               + a2 * static_cast<int32_t>(b20) + a3 * static_cast<int32_t>(b30);
        acc.y += a0 * static_cast<int32_t>(b01) + a1 * static_cast<int32_t>(b11)
               + a2 * static_cast<int32_t>(b21) + a3 * static_cast<int32_t>(b31);
        acc.z += a0 * static_cast<int32_t>(b02) + a1 * static_cast<int32_t>(b12)
               + a2 * static_cast<int32_t>(b22) + a3 * static_cast<int32_t>(b32);
        acc.w += a0 * static_cast<int32_t>(b03) + a1 * static_cast<int32_t>(b13)
               + a2 * static_cast<int32_t>(b23) + a3 * static_cast<int32_t>(b33);
    }

    for (; k < K; ++k)
    {
        const uint32_t packed = *reinterpret_cast<const uint32_t*>(d_B_int8 + k * N + n4);
        const int8_t b0 = static_cast<int8_t>(packed & 0xFF);
        const int8_t b1 = static_cast<int8_t>((packed >> 8) & 0xFF);
        const int8_t b2 = static_cast<int8_t>((packed >> 16) & 0xFF);
        const int8_t b3 = static_cast<int8_t>((packed >> 24) & 0xFF);
        const int32_t a = static_cast<int32_t>(d_A_int8[k]);
        acc.x += a * static_cast<int32_t>(b0);
        acc.y += a * static_cast<int32_t>(b1);
        acc.z += a * static_cast<int32_t>(b2);
        acc.w += a * static_cast<int32_t>(b3);
    }

    d_C_int32[n4 + 0] = acc.x;
    d_C_int32[n4 + 1] = acc.y;
    d_C_int32[n4 + 2] = acc.z;
    d_C_int32[n4 + 3] = acc.w;
}

// Wide-N kernel: vec4 with VNNI-packed weights
__global__ void gemv_int8_int8_wide_vec4_vnni_kernel(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B_int8_vnni,
    int32_t* __restrict__ d_C_int32,
    int N,
    int K)
{
    const int lane_n = threadIdx.x;
    const int n4 = blockIdx.x * GEMV_INT8_VNNI_WIDE_TILE_N + lane_n * 4;
    if (n4 >= N) return;

    int4 acc = {0, 0, 0, 0};
    const int k_groups = K / 4;
    for (int kg = 0; kg < k_groups; ++kg)
    {
        const int k_base = kg * 4;
        const int32_t a0 = static_cast<int32_t>(d_A_int8[k_base + 0]);
        const int32_t a1 = static_cast<int32_t>(d_A_int8[k_base + 1]);
        const int32_t a2 = static_cast<int32_t>(d_A_int8[k_base + 2]);
        const int32_t a3 = static_cast<int32_t>(d_A_int8[k_base + 3]);

        const uint32_t p0 = *reinterpret_cast<const uint32_t*>(
            d_B_int8_vnni + (static_cast<size_t>(kg) * N + (n4 + 0)) * 4);
        const uint32_t p1 = *reinterpret_cast<const uint32_t*>(
            d_B_int8_vnni + (static_cast<size_t>(kg) * N + (n4 + 1)) * 4);
        const uint32_t p2 = *reinterpret_cast<const uint32_t*>(
            d_B_int8_vnni + (static_cast<size_t>(kg) * N + (n4 + 2)) * 4);
        const uint32_t p3 = *reinterpret_cast<const uint32_t*>(
            d_B_int8_vnni + (static_cast<size_t>(kg) * N + (n4 + 3)) * 4);

        const int8_t b00 = static_cast<int8_t>(p0 & 0xFF);
        const int8_t b01 = static_cast<int8_t>((p0 >> 8) & 0xFF);
        const int8_t b02 = static_cast<int8_t>((p0 >> 16) & 0xFF);
        const int8_t b03 = static_cast<int8_t>((p0 >> 24) & 0xFF);

        const int8_t b10 = static_cast<int8_t>(p1 & 0xFF);
        const int8_t b11 = static_cast<int8_t>((p1 >> 8) & 0xFF);
        const int8_t b12 = static_cast<int8_t>((p1 >> 16) & 0xFF);
        const int8_t b13 = static_cast<int8_t>((p1 >> 24) & 0xFF);

        const int8_t b20 = static_cast<int8_t>(p2 & 0xFF);
        const int8_t b21 = static_cast<int8_t>((p2 >> 8) & 0xFF);
        const int8_t b22 = static_cast<int8_t>((p2 >> 16) & 0xFF);
        const int8_t b23 = static_cast<int8_t>((p2 >> 24) & 0xFF);

        const int8_t b30 = static_cast<int8_t>(p3 & 0xFF);
        const int8_t b31 = static_cast<int8_t>((p3 >> 8) & 0xFF);
        const int8_t b32 = static_cast<int8_t>((p3 >> 16) & 0xFF);
        const int8_t b33 = static_cast<int8_t>((p3 >> 24) & 0xFF);

        acc.x += a0 * static_cast<int32_t>(b00) + a1 * static_cast<int32_t>(b01)
               + a2 * static_cast<int32_t>(b02) + a3 * static_cast<int32_t>(b03);
        acc.y += a0 * static_cast<int32_t>(b10) + a1 * static_cast<int32_t>(b11)
               + a2 * static_cast<int32_t>(b12) + a3 * static_cast<int32_t>(b13);
        acc.z += a0 * static_cast<int32_t>(b20) + a1 * static_cast<int32_t>(b21)
               + a2 * static_cast<int32_t>(b22) + a3 * static_cast<int32_t>(b23);
        acc.w += a0 * static_cast<int32_t>(b30) + a1 * static_cast<int32_t>(b31)
               + a2 * static_cast<int32_t>(b32) + a3 * static_cast<int32_t>(b33);
    }

    d_C_int32[n4 + 0] = acc.x;
    d_C_int32[n4 + 1] = acc.y;
    d_C_int32[n4 + 2] = acc.z;
    d_C_int32[n4 + 3] = acc.w;
}

// Wide-N kernel (templated): vec4 with VNNI-packed weights
template <int TILE_N>
__global__ void gemv_int8_int8_wide_vec4_vnni_kernel_t(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B_int8_vnni,
    int32_t* __restrict__ d_C_int32,
    int N,
    int K)
{
    const int lane_n = threadIdx.x;
    const int n4 = blockIdx.x * TILE_N + lane_n * 4;
    if (n4 >= N) return;

    int4 acc = {0, 0, 0, 0};
    const int k_groups = K / 4;
    for (int kg = 0; kg < k_groups; ++kg)
    {
        const int k_base = kg * 4;
        const int32_t a0 = static_cast<int32_t>(d_A_int8[k_base + 0]);
        const int32_t a1 = static_cast<int32_t>(d_A_int8[k_base + 1]);
        const int32_t a2 = static_cast<int32_t>(d_A_int8[k_base + 2]);
        const int32_t a3 = static_cast<int32_t>(d_A_int8[k_base + 3]);

        const uint32_t p0 = *reinterpret_cast<const uint32_t*>(
            d_B_int8_vnni + (static_cast<size_t>(kg) * N + (n4 + 0)) * 4);
        const uint32_t p1 = *reinterpret_cast<const uint32_t*>(
            d_B_int8_vnni + (static_cast<size_t>(kg) * N + (n4 + 1)) * 4);
        const uint32_t p2 = *reinterpret_cast<const uint32_t*>(
            d_B_int8_vnni + (static_cast<size_t>(kg) * N + (n4 + 2)) * 4);
        const uint32_t p3 = *reinterpret_cast<const uint32_t*>(
            d_B_int8_vnni + (static_cast<size_t>(kg) * N + (n4 + 3)) * 4);

        const int8_t b00 = static_cast<int8_t>(p0 & 0xFF);
        const int8_t b01 = static_cast<int8_t>((p0 >> 8) & 0xFF);
        const int8_t b02 = static_cast<int8_t>((p0 >> 16) & 0xFF);
        const int8_t b03 = static_cast<int8_t>((p0 >> 24) & 0xFF);

        const int8_t b10 = static_cast<int8_t>(p1 & 0xFF);
        const int8_t b11 = static_cast<int8_t>((p1 >> 8) & 0xFF);
        const int8_t b12 = static_cast<int8_t>((p1 >> 16) & 0xFF);
        const int8_t b13 = static_cast<int8_t>((p1 >> 24) & 0xFF);

        const int8_t b20 = static_cast<int8_t>(p2 & 0xFF);
        const int8_t b21 = static_cast<int8_t>((p2 >> 8) & 0xFF);
        const int8_t b22 = static_cast<int8_t>((p2 >> 16) & 0xFF);
        const int8_t b23 = static_cast<int8_t>((p2 >> 24) & 0xFF);

        const int8_t b30 = static_cast<int8_t>(p3 & 0xFF);
        const int8_t b31 = static_cast<int8_t>((p3 >> 8) & 0xFF);
        const int8_t b32 = static_cast<int8_t>((p3 >> 16) & 0xFF);
        const int8_t b33 = static_cast<int8_t>((p3 >> 24) & 0xFF);

        acc.x += a0 * static_cast<int32_t>(b00) + a1 * static_cast<int32_t>(b01)
               + a2 * static_cast<int32_t>(b02) + a3 * static_cast<int32_t>(b03);
        acc.y += a0 * static_cast<int32_t>(b10) + a1 * static_cast<int32_t>(b11)
               + a2 * static_cast<int32_t>(b12) + a3 * static_cast<int32_t>(b13);
        acc.z += a0 * static_cast<int32_t>(b20) + a1 * static_cast<int32_t>(b21)
               + a2 * static_cast<int32_t>(b22) + a3 * static_cast<int32_t>(b23);
        acc.w += a0 * static_cast<int32_t>(b30) + a1 * static_cast<int32_t>(b31)
               + a2 * static_cast<int32_t>(b32) + a3 * static_cast<int32_t>(b33);
    }

    d_C_int32[n4 + 0] = acc.x;
    d_C_int32[n4 + 1] = acc.y;
    d_C_int32[n4 + 2] = acc.z;
    d_C_int32[n4 + 3] = acc.w;
}

// Wide-N kernel: 2 columns per thread
__global__ void gemv_int8_int8_wide2_kernel(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B_int8,
    int32_t* __restrict__ d_C_int32,
    int N,
    int K)
{
    const int n2 = blockIdx.x * GEMV_WIDE2_TILE_N + threadIdx.x * 2;
    if (n2 >= N) return;

    int32_t acc0 = 0;
    int32_t acc1 = 0;

    int k = 0;
    const int k_end_unrolled = K - (K % 4);
    for (; k < k_end_unrolled; k += 4)
    {
        const int32_t a0 = static_cast<int32_t>(d_A_int8[k + 0]);
        const int32_t a1 = static_cast<int32_t>(d_A_int8[k + 1]);
        const int32_t a2 = static_cast<int32_t>(d_A_int8[k + 2]);
        const int32_t a3 = static_cast<int32_t>(d_A_int8[k + 3]);

        const uint16_t p0 = *reinterpret_cast<const uint16_t*>(d_B_int8 + (k + 0) * N + n2);
        const uint16_t p1 = *reinterpret_cast<const uint16_t*>(d_B_int8 + (k + 1) * N + n2);
        const uint16_t p2 = *reinterpret_cast<const uint16_t*>(d_B_int8 + (k + 2) * N + n2);
        const uint16_t p3 = *reinterpret_cast<const uint16_t*>(d_B_int8 + (k + 3) * N + n2);

        const int8_t b00 = static_cast<int8_t>(p0 & 0xFF);
        const int8_t b01 = static_cast<int8_t>((p0 >> 8) & 0xFF);
        const int8_t b10 = static_cast<int8_t>(p1 & 0xFF);
        const int8_t b11 = static_cast<int8_t>((p1 >> 8) & 0xFF);
        const int8_t b20 = static_cast<int8_t>(p2 & 0xFF);
        const int8_t b21 = static_cast<int8_t>((p2 >> 8) & 0xFF);
        const int8_t b30 = static_cast<int8_t>(p3 & 0xFF);
        const int8_t b31 = static_cast<int8_t>((p3 >> 8) & 0xFF);

        acc0 += a0 * static_cast<int32_t>(b00) + a1 * static_cast<int32_t>(b10)
              + a2 * static_cast<int32_t>(b20) + a3 * static_cast<int32_t>(b30);
        acc1 += a0 * static_cast<int32_t>(b01) + a1 * static_cast<int32_t>(b11)
              + a2 * static_cast<int32_t>(b21) + a3 * static_cast<int32_t>(b31);
    }

    for (; k < K; ++k)
    {
        const uint16_t packed = *reinterpret_cast<const uint16_t*>(d_B_int8 + k * N + n2);
        const int8_t b0 = static_cast<int8_t>(packed & 0xFF);
        const int8_t b1 = static_cast<int8_t>((packed >> 8) & 0xFF);
        const int32_t a = static_cast<int32_t>(d_A_int8[k]);
        acc0 += a * static_cast<int32_t>(b0);
        acc1 += a * static_cast<int32_t>(b1);
    }

    d_C_int32[n2 + 0] = acc0;
    d_C_int32[n2 + 1] = acc1;
}

// Wide-N kernel: 2 columns per thread with VNNI-packed weights
__global__ void gemv_int8_int8_wide2_vnni_kernel(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B_int8_vnni,
    int32_t* __restrict__ d_C_int32,
    int N,
    int K)
{
    const int n2 = blockIdx.x * GEMV_INT8_VNNI_WIDE2_TILE_N + threadIdx.x * 2;
    if (n2 >= N) return;

    int32_t acc0 = 0;
    int32_t acc1 = 0;
    const int k_groups = K / 4;
    for (int kg = 0; kg < k_groups; ++kg)
    {
        const int k_base = kg * 4;
        const int32_t a0 = static_cast<int32_t>(d_A_int8[k_base + 0]);
        const int32_t a1 = static_cast<int32_t>(d_A_int8[k_base + 1]);
        const int32_t a2 = static_cast<int32_t>(d_A_int8[k_base + 2]);
        const int32_t a3 = static_cast<int32_t>(d_A_int8[k_base + 3]);

        const uint32_t p0 = *reinterpret_cast<const uint32_t*>(
            d_B_int8_vnni + (static_cast<size_t>(kg) * N + (n2 + 0)) * 4);
        const uint32_t p1 = *reinterpret_cast<const uint32_t*>(
            d_B_int8_vnni + (static_cast<size_t>(kg) * N + (n2 + 1)) * 4);

        const int8_t b00 = static_cast<int8_t>(p0 & 0xFF);
        const int8_t b01 = static_cast<int8_t>((p0 >> 8) & 0xFF);
        const int8_t b02 = static_cast<int8_t>((p0 >> 16) & 0xFF);
        const int8_t b03 = static_cast<int8_t>((p0 >> 24) & 0xFF);

        const int8_t b10 = static_cast<int8_t>(p1 & 0xFF);
        const int8_t b11 = static_cast<int8_t>((p1 >> 8) & 0xFF);
        const int8_t b12 = static_cast<int8_t>((p1 >> 16) & 0xFF);
        const int8_t b13 = static_cast<int8_t>((p1 >> 24) & 0xFF);

        acc0 += a0 * static_cast<int32_t>(b00) + a1 * static_cast<int32_t>(b01)
              + a2 * static_cast<int32_t>(b02) + a3 * static_cast<int32_t>(b03);
        acc1 += a0 * static_cast<int32_t>(b10) + a1 * static_cast<int32_t>(b11)
              + a2 * static_cast<int32_t>(b12) + a3 * static_cast<int32_t>(b13);
    }

    d_C_int32[n2 + 0] = acc0;
    d_C_int32[n2 + 1] = acc1;
}

// Wide-N kernel (templated): 2 columns per thread with VNNI-packed weights
template <int TILE_N>
__global__ void gemv_int8_int8_wide2_vnni_kernel_t(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B_int8_vnni,
    int32_t* __restrict__ d_C_int32,
    int N,
    int K)
{
    const int n2 = blockIdx.x * TILE_N + threadIdx.x * 2;
    if (n2 >= N) return;

    int32_t acc0 = 0;
    int32_t acc1 = 0;
    const int k_groups = K / 4;
    for (int kg = 0; kg < k_groups; ++kg)
    {
        const int k_base = kg * 4;
        const int32_t a0 = static_cast<int32_t>(d_A_int8[k_base + 0]);
        const int32_t a1 = static_cast<int32_t>(d_A_int8[k_base + 1]);
        const int32_t a2 = static_cast<int32_t>(d_A_int8[k_base + 2]);
        const int32_t a3 = static_cast<int32_t>(d_A_int8[k_base + 3]);

        const uint32_t p0 = *reinterpret_cast<const uint32_t*>(
            d_B_int8_vnni + (static_cast<size_t>(kg) * N + (n2 + 0)) * 4);
        const uint32_t p1 = *reinterpret_cast<const uint32_t*>(
            d_B_int8_vnni + (static_cast<size_t>(kg) * N + (n2 + 1)) * 4);

        const int8_t b00 = static_cast<int8_t>(p0 & 0xFF);
        const int8_t b01 = static_cast<int8_t>((p0 >> 8) & 0xFF);
        const int8_t b02 = static_cast<int8_t>((p0 >> 16) & 0xFF);
        const int8_t b03 = static_cast<int8_t>((p0 >> 24) & 0xFF);

        const int8_t b10 = static_cast<int8_t>(p1 & 0xFF);
        const int8_t b11 = static_cast<int8_t>((p1 >> 8) & 0xFF);
        const int8_t b12 = static_cast<int8_t>((p1 >> 16) & 0xFF);
        const int8_t b13 = static_cast<int8_t>((p1 >> 24) & 0xFF);

        acc0 += a0 * static_cast<int32_t>(b00) + a1 * static_cast<int32_t>(b01)
              + a2 * static_cast<int32_t>(b02) + a3 * static_cast<int32_t>(b03);
        acc1 += a0 * static_cast<int32_t>(b10) + a1 * static_cast<int32_t>(b11)
              + a2 * static_cast<int32_t>(b12) + a3 * static_cast<int32_t>(b13);
    }

    d_C_int32[n2 + 0] = acc0;
    d_C_int32[n2 + 1] = acc1;
}

// K-parallel kernel: scalar threads, shared A tiles
__global__ void gemv_int8_int8_kpar_kernel(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B_int8,
    int32_t* __restrict__ d_C_int32,
    int N,
    int K)
{
    const int n = blockIdx.x * GEMV_INT8_KPAR_TILE_N + threadIdx.x;
    const bool active = (n < N);

    const int lane_k = threadIdx.y;
    int32_t acc = 0;

    __shared__ int8_t sA[GEMV_INT8_KPAR_TILE_K];
    for (int k_base = 0; k_base < K; k_base += GEMV_INT8_KPAR_TILE_K)
    {
        const int tile_k = min(GEMV_INT8_KPAR_TILE_K, K - k_base);
        const int linear_tid = threadIdx.y * blockDim.x + threadIdx.x;
        for (int i = linear_tid; i < tile_k; i += blockDim.x * blockDim.y)
        {
            sA[i] = d_A_int8[k_base + i];
        }
        __syncthreads();

        const int k_step = GEMV_INT8_KPAR_SPLIT * 4;
        int k_local = lane_k;
        const int k_limit = tile_k - (tile_k % k_step);
        for (; k_local < k_limit; k_local += k_step)
        {
            if (active)
            {
                const int k0 = k_base + k_local + 0 * GEMV_INT8_KPAR_SPLIT;
                const int k1 = k_base + k_local + 1 * GEMV_INT8_KPAR_SPLIT;
                const int k2 = k_base + k_local + 2 * GEMV_INT8_KPAR_SPLIT;
                const int k3 = k_base + k_local + 3 * GEMV_INT8_KPAR_SPLIT;

                acc += static_cast<int32_t>(sA[k_local + 0 * GEMV_INT8_KPAR_SPLIT])
                     * static_cast<int32_t>(d_B_int8[k0 * N + n]);
                acc += static_cast<int32_t>(sA[k_local + 1 * GEMV_INT8_KPAR_SPLIT])
                     * static_cast<int32_t>(d_B_int8[k1 * N + n]);
                acc += static_cast<int32_t>(sA[k_local + 2 * GEMV_INT8_KPAR_SPLIT])
                     * static_cast<int32_t>(d_B_int8[k2 * N + n]);
                acc += static_cast<int32_t>(sA[k_local + 3 * GEMV_INT8_KPAR_SPLIT])
                     * static_cast<int32_t>(d_B_int8[k3 * N + n]);
            }
        }

        for (; k_local < tile_k; k_local += GEMV_INT8_KPAR_SPLIT)
        {
            if (active)
            {
                const int k = k_base + k_local;
                acc += static_cast<int32_t>(sA[k_local])
                     * static_cast<int32_t>(d_B_int8[k * N + n]);
            }
        }
        __syncthreads();
    }

    __shared__ int32_t partial[GEMV_INT8_KPAR_SPLIT][GEMV_INT8_KPAR_TILE_N];
    partial[lane_k][threadIdx.x] = acc;
    __syncthreads();

    if (lane_k == 0 && active)
    {
        int32_t sum = 0;
        #pragma unroll
        for (int i = 0; i < GEMV_INT8_KPAR_SPLIT; ++i)
        {
            sum += partial[i][threadIdx.x];
        }

        d_C_int32[n] = sum;
    }
}

// K-parallel vec4 kernel: each thread handles 4 columns
__global__ void gemv_int8_int8_kpar_vec4_kernel(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B_int8,
    int32_t* __restrict__ d_C_int32,
    int N,
    int K)
{
    const int lane_n = threadIdx.x;
    const int n4 = blockIdx.x * GEMV_INT8_KPAR_TILE_N + lane_n * 4;
    const bool active = (n4 < N);

    const int lane_k = threadIdx.y;
    int4 acc = {0, 0, 0, 0};

    __shared__ int8_t sA[GEMV_INT8_KPAR_TILE_K];
    for (int k_base = 0; k_base < K; k_base += GEMV_INT8_KPAR_TILE_K)
    {
        const int tile_k = min(GEMV_INT8_KPAR_TILE_K, K - k_base);
        const int linear_tid = threadIdx.y * blockDim.x + threadIdx.x;
        for (int i = linear_tid; i < tile_k; i += blockDim.x * blockDim.y)
        {
            sA[i] = d_A_int8[k_base + i];
        }
        __syncthreads();

        const int k_step = GEMV_INT8_KPAR_SPLIT * 4;
        int k_local = lane_k;
        const int k_limit = tile_k - (tile_k % k_step);
        for (; k_local < k_limit; k_local += k_step)
        {
            if (active)
            {
                const int k0 = k_base + k_local + 0 * GEMV_INT8_KPAR_SPLIT;
                const int k1 = k_base + k_local + 1 * GEMV_INT8_KPAR_SPLIT;
                const int k2 = k_base + k_local + 2 * GEMV_INT8_KPAR_SPLIT;
                const int k3 = k_base + k_local + 3 * GEMV_INT8_KPAR_SPLIT;

                const uint32_t p0 = *reinterpret_cast<const uint32_t*>(d_B_int8 + k0 * N + n4);
                const uint32_t p1 = *reinterpret_cast<const uint32_t*>(d_B_int8 + k1 * N + n4);
                const uint32_t p2 = *reinterpret_cast<const uint32_t*>(d_B_int8 + k2 * N + n4);
                const uint32_t p3 = *reinterpret_cast<const uint32_t*>(d_B_int8 + k3 * N + n4);

                const int8_t b00 = static_cast<int8_t>(p0 & 0xFF);
                const int8_t b01 = static_cast<int8_t>((p0 >> 8) & 0xFF);
                const int8_t b02 = static_cast<int8_t>((p0 >> 16) & 0xFF);
                const int8_t b03 = static_cast<int8_t>((p0 >> 24) & 0xFF);

                const int8_t b10 = static_cast<int8_t>(p1 & 0xFF);
                const int8_t b11 = static_cast<int8_t>((p1 >> 8) & 0xFF);
                const int8_t b12 = static_cast<int8_t>((p1 >> 16) & 0xFF);
                const int8_t b13 = static_cast<int8_t>((p1 >> 24) & 0xFF);

                const int8_t b20 = static_cast<int8_t>(p2 & 0xFF);
                const int8_t b21 = static_cast<int8_t>((p2 >> 8) & 0xFF);
                const int8_t b22 = static_cast<int8_t>((p2 >> 16) & 0xFF);
                const int8_t b23 = static_cast<int8_t>((p2 >> 24) & 0xFF);

                const int8_t b30 = static_cast<int8_t>(p3 & 0xFF);
                const int8_t b31 = static_cast<int8_t>((p3 >> 8) & 0xFF);
                const int8_t b32 = static_cast<int8_t>((p3 >> 16) & 0xFF);
                const int8_t b33 = static_cast<int8_t>((p3 >> 24) & 0xFF);

                const int32_t a0 = static_cast<int32_t>(sA[k_local + 0 * GEMV_INT8_KPAR_SPLIT]);
                const int32_t a1 = static_cast<int32_t>(sA[k_local + 1 * GEMV_INT8_KPAR_SPLIT]);
                const int32_t a2 = static_cast<int32_t>(sA[k_local + 2 * GEMV_INT8_KPAR_SPLIT]);
                const int32_t a3 = static_cast<int32_t>(sA[k_local + 3 * GEMV_INT8_KPAR_SPLIT]);

                acc.x += a0 * static_cast<int32_t>(b00) + a1 * static_cast<int32_t>(b10)
                       + a2 * static_cast<int32_t>(b20) + a3 * static_cast<int32_t>(b30);
                acc.y += a0 * static_cast<int32_t>(b01) + a1 * static_cast<int32_t>(b11)
                       + a2 * static_cast<int32_t>(b21) + a3 * static_cast<int32_t>(b31);
                acc.z += a0 * static_cast<int32_t>(b02) + a1 * static_cast<int32_t>(b12)
                       + a2 * static_cast<int32_t>(b22) + a3 * static_cast<int32_t>(b32);
                acc.w += a0 * static_cast<int32_t>(b03) + a1 * static_cast<int32_t>(b13)
                       + a2 * static_cast<int32_t>(b23) + a3 * static_cast<int32_t>(b33);
            }
        }

        for (; k_local < tile_k; k_local += GEMV_INT8_KPAR_SPLIT)
        {
            const int k = k_base + k_local;
            if (active)
            {
                const uint32_t packed = *reinterpret_cast<const uint32_t*>(d_B_int8 + k * N + n4);
                const int8_t b0 = static_cast<int8_t>(packed & 0xFF);
                const int8_t b1 = static_cast<int8_t>((packed >> 8) & 0xFF);
                const int8_t b2 = static_cast<int8_t>((packed >> 16) & 0xFF);
                const int8_t b3 = static_cast<int8_t>((packed >> 24) & 0xFF);

                const int32_t a = static_cast<int32_t>(sA[k_local]);
                acc.x += a * static_cast<int32_t>(b0);
                acc.y += a * static_cast<int32_t>(b1);
                acc.z += a * static_cast<int32_t>(b2);
                acc.w += a * static_cast<int32_t>(b3);
            }
        }
        __syncthreads();
    }

    __shared__ int4 partial[GEMV_INT8_KPAR_SPLIT][GEMV_INT8_KPAR_TILE_N / 4];
    partial[lane_k][lane_n] = acc;
    __syncthreads();

    if (lane_k == 0 && active)
    {
        int4 sum = {0, 0, 0, 0};
        #pragma unroll
        for (int i = 0; i < GEMV_INT8_KPAR_SPLIT; ++i)
        {
            sum.x += partial[i][lane_n].x;
            sum.y += partial[i][lane_n].y;
            sum.z += partial[i][lane_n].z;
            sum.w += partial[i][lane_n].w;
        }

        d_C_int32[n4 + 0] = sum.x;
        d_C_int32[n4 + 1] = sum.y;
        d_C_int32[n4 + 2] = sum.z;
        d_C_int32[n4 + 3] = sum.w;
    }
}

// FFN Down specialization: vec4 + k-parallel reduction with fixed tiles
template <int TILE_N, int SPLIT>
__global__ void gemv_int8_int8_ffn_down_kernel_t(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B_int8,
    int32_t* __restrict__ d_C_int32,
    int N,
    int K)
{
    const int lane_n = threadIdx.x;
    const int n4 = blockIdx.x * TILE_N + lane_n * 4;
    const bool active = (n4 < N);

    const int lane_k = threadIdx.y;
    int4 acc = {0, 0, 0, 0};

    __shared__ int8_t sA[GEMV_INT8_FFND_TILE_K];
    for (int k_base = 0; k_base < K; k_base += GEMV_INT8_FFND_TILE_K)
    {
        const int tile_k = min(GEMV_INT8_FFND_TILE_K, K - k_base);
        const int linear_tid = threadIdx.y * blockDim.x + threadIdx.x;
        for (int i = linear_tid; i < tile_k; i += blockDim.x * blockDim.y)
        {
            sA[i] = d_A_int8[k_base + i];
        }
        __syncthreads();

        const int k_step = SPLIT * 4;
        int k_local = lane_k;
        const int k_limit = tile_k - (tile_k % k_step);
        for (; k_local < k_limit; k_local += k_step)
        {
            if (active)
            {
                const int k0 = k_base + k_local + 0 * SPLIT;
                const int k1 = k_base + k_local + 1 * SPLIT;
                const int k2 = k_base + k_local + 2 * SPLIT;
                const int k3 = k_base + k_local + 3 * SPLIT;

                const uint32_t p0 = *reinterpret_cast<const uint32_t*>(d_B_int8 + k0 * N + n4);
                const uint32_t p1 = *reinterpret_cast<const uint32_t*>(d_B_int8 + k1 * N + n4);
                const uint32_t p2 = *reinterpret_cast<const uint32_t*>(d_B_int8 + k2 * N + n4);
                const uint32_t p3 = *reinterpret_cast<const uint32_t*>(d_B_int8 + k3 * N + n4);

                const int8_t b00 = static_cast<int8_t>(p0 & 0xFF);
                const int8_t b01 = static_cast<int8_t>((p0 >> 8) & 0xFF);
                const int8_t b02 = static_cast<int8_t>((p0 >> 16) & 0xFF);
                const int8_t b03 = static_cast<int8_t>((p0 >> 24) & 0xFF);

                const int8_t b10 = static_cast<int8_t>(p1 & 0xFF);
                const int8_t b11 = static_cast<int8_t>((p1 >> 8) & 0xFF);
                const int8_t b12 = static_cast<int8_t>((p1 >> 16) & 0xFF);
                const int8_t b13 = static_cast<int8_t>((p1 >> 24) & 0xFF);

                const int8_t b20 = static_cast<int8_t>(p2 & 0xFF);
                const int8_t b21 = static_cast<int8_t>((p2 >> 8) & 0xFF);
                const int8_t b22 = static_cast<int8_t>((p2 >> 16) & 0xFF);
                const int8_t b23 = static_cast<int8_t>((p2 >> 24) & 0xFF);

                const int8_t b30 = static_cast<int8_t>(p3 & 0xFF);
                const int8_t b31 = static_cast<int8_t>((p3 >> 8) & 0xFF);
                const int8_t b32 = static_cast<int8_t>((p3 >> 16) & 0xFF);
                const int8_t b33 = static_cast<int8_t>((p3 >> 24) & 0xFF);

                const int32_t a0 = static_cast<int32_t>(sA[k_local + 0 * SPLIT]);
                const int32_t a1 = static_cast<int32_t>(sA[k_local + 1 * SPLIT]);
                const int32_t a2 = static_cast<int32_t>(sA[k_local + 2 * SPLIT]);
                const int32_t a3 = static_cast<int32_t>(sA[k_local + 3 * SPLIT]);

                acc.x += a0 * static_cast<int32_t>(b00) + a1 * static_cast<int32_t>(b10)
                       + a2 * static_cast<int32_t>(b20) + a3 * static_cast<int32_t>(b30);
                acc.y += a0 * static_cast<int32_t>(b01) + a1 * static_cast<int32_t>(b11)
                       + a2 * static_cast<int32_t>(b21) + a3 * static_cast<int32_t>(b31);
                acc.z += a0 * static_cast<int32_t>(b02) + a1 * static_cast<int32_t>(b12)
                       + a2 * static_cast<int32_t>(b22) + a3 * static_cast<int32_t>(b32);
                acc.w += a0 * static_cast<int32_t>(b03) + a1 * static_cast<int32_t>(b13)
                       + a2 * static_cast<int32_t>(b23) + a3 * static_cast<int32_t>(b33);
            }
        }

        for (; k_local < tile_k; k_local += SPLIT)
        {
            if (active)
            {
                const int k = k_base + k_local;
                const uint32_t packed = *reinterpret_cast<const uint32_t*>(d_B_int8 + k * N + n4);
                const int8_t b0 = static_cast<int8_t>(packed & 0xFF);
                const int8_t b1 = static_cast<int8_t>((packed >> 8) & 0xFF);
                const int8_t b2 = static_cast<int8_t>((packed >> 16) & 0xFF);
                const int8_t b3 = static_cast<int8_t>((packed >> 24) & 0xFF);
                const int32_t a = static_cast<int32_t>(sA[k_local]);
                acc.x += a * static_cast<int32_t>(b0);
                acc.y += a * static_cast<int32_t>(b1);
                acc.z += a * static_cast<int32_t>(b2);
                acc.w += a * static_cast<int32_t>(b3);
            }
        }
        __syncthreads();
    }

    __shared__ int4 partial[SPLIT][TILE_N / 4];
    partial[lane_k][lane_n] = acc;
    __syncthreads();

    if (lane_k == 0 && active)
    {
        int4 sum = {0, 0, 0, 0};
        #pragma unroll
        for (int i = 0; i < SPLIT; ++i)
        {
            sum.x += partial[i][lane_n].x;
            sum.y += partial[i][lane_n].y;
            sum.z += partial[i][lane_n].z;
            sum.w += partial[i][lane_n].w;
        }

        d_C_int32[n4 + 0] = sum.x;
        d_C_int32[n4 + 1] = sum.y;
        d_C_int32[n4 + 2] = sum.z;
        d_C_int32[n4 + 3] = sum.w;
    }
}

// FFN Down VNNI specialization: vec4, simple K/4 loop (no k-parallel)
template <int TILE_N>
__global__ void gemv_int8_int8_ffn_down_vnni_kernel_t(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B_int8_vnni,
    int32_t* __restrict__ d_C_int32,
    int N,
    int K)
{
    const int lane_n = threadIdx.x;
    const int n4 = blockIdx.x * TILE_N + lane_n * 4;
    if (n4 >= N) return;

    int4 acc = {0, 0, 0, 0};
    const int k_groups = K / 4;
    for (int kg = 0; kg < k_groups; ++kg)
    {
        const int k_base = kg * 4;
        const int32_t a0 = static_cast<int32_t>(d_A_int8[k_base + 0]);
        const int32_t a1 = static_cast<int32_t>(d_A_int8[k_base + 1]);
        const int32_t a2 = static_cast<int32_t>(d_A_int8[k_base + 2]);
        const int32_t a3 = static_cast<int32_t>(d_A_int8[k_base + 3]);

        const uint32_t p0 = *reinterpret_cast<const uint32_t*>(
            d_B_int8_vnni + (static_cast<size_t>(kg) * N + (n4 + 0)) * 4);
        const uint32_t p1 = *reinterpret_cast<const uint32_t*>(
            d_B_int8_vnni + (static_cast<size_t>(kg) * N + (n4 + 1)) * 4);
        const uint32_t p2 = *reinterpret_cast<const uint32_t*>(
            d_B_int8_vnni + (static_cast<size_t>(kg) * N + (n4 + 2)) * 4);
        const uint32_t p3 = *reinterpret_cast<const uint32_t*>(
            d_B_int8_vnni + (static_cast<size_t>(kg) * N + (n4 + 3)) * 4);

        const int8_t b00 = static_cast<int8_t>(p0 & 0xFF);
        const int8_t b01 = static_cast<int8_t>((p0 >> 8) & 0xFF);
        const int8_t b02 = static_cast<int8_t>((p0 >> 16) & 0xFF);
        const int8_t b03 = static_cast<int8_t>((p0 >> 24) & 0xFF);

        const int8_t b10 = static_cast<int8_t>(p1 & 0xFF);
        const int8_t b11 = static_cast<int8_t>((p1 >> 8) & 0xFF);
        const int8_t b12 = static_cast<int8_t>((p1 >> 16) & 0xFF);
        const int8_t b13 = static_cast<int8_t>((p1 >> 24) & 0xFF);

        const int8_t b20 = static_cast<int8_t>(p2 & 0xFF);
        const int8_t b21 = static_cast<int8_t>((p2 >> 8) & 0xFF);
        const int8_t b22 = static_cast<int8_t>((p2 >> 16) & 0xFF);
        const int8_t b23 = static_cast<int8_t>((p2 >> 24) & 0xFF);

        const int8_t b30 = static_cast<int8_t>(p3 & 0xFF);
        const int8_t b31 = static_cast<int8_t>((p3 >> 8) & 0xFF);
        const int8_t b32 = static_cast<int8_t>((p3 >> 16) & 0xFF);
        const int8_t b33 = static_cast<int8_t>((p3 >> 24) & 0xFF);

        acc.x += a0 * static_cast<int32_t>(b00) + a1 * static_cast<int32_t>(b01)
               + a2 * static_cast<int32_t>(b02) + a3 * static_cast<int32_t>(b03);
        acc.y += a0 * static_cast<int32_t>(b10) + a1 * static_cast<int32_t>(b11)
               + a2 * static_cast<int32_t>(b12) + a3 * static_cast<int32_t>(b13);
        acc.z += a0 * static_cast<int32_t>(b20) + a1 * static_cast<int32_t>(b21)
               + a2 * static_cast<int32_t>(b22) + a3 * static_cast<int32_t>(b23);
        acc.w += a0 * static_cast<int32_t>(b30) + a1 * static_cast<int32_t>(b31)
               + a2 * static_cast<int32_t>(b32) + a3 * static_cast<int32_t>(b33);
    }

    d_C_int32[n4 + 0] = acc.x;
    d_C_int32[n4 + 1] = acc.y;
    d_C_int32[n4 + 2] = acc.z;
    d_C_int32[n4 + 3] = acc.w;
}

// VNNI K-parallel specialization: vec4 + k-parallel reduction using VNNI-packed weights
template <int TILE_N, int SPLIT>
__global__ void gemv_int8_int8_ffn_down_vnni_kpar_kernel_t(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B_int8_vnni,
    int32_t* __restrict__ d_C_int32,
    int N,
    int K)
{
    const int lane_n = threadIdx.x;
    const int n4 = blockIdx.x * TILE_N + lane_n * 4;
    const bool active = (n4 < N);

    const int lane_k = threadIdx.y;
    int4 acc = {0, 0, 0, 0};

    __shared__ int8_t sA[GEMV_INT8_VNNI_KPAR_TILE_K];
    const int k_groups_per_tile = GEMV_INT8_VNNI_KPAR_TILE_K / 4;
    for (int k_base = 0; k_base < K; k_base += GEMV_INT8_VNNI_KPAR_TILE_K)
    {
        const int tile_k = min(GEMV_INT8_VNNI_KPAR_TILE_K, K - k_base);
        const int linear_tid = threadIdx.y * blockDim.x + threadIdx.x;
        for (int i = linear_tid; i < tile_k; i += blockDim.x * blockDim.y)
        {
            sA[i] = d_A_int8[k_base + i];
        }
        __syncthreads();

        const int tile_groups = tile_k / 4;
        const int k_group_base = k_base / 4;
        for (int kg = lane_k; kg < tile_groups; kg += SPLIT)
        {
            if (active)
            {
                const int k_local = kg * 4;
                const int32_t a0 = static_cast<int32_t>(sA[k_local + 0]);
                const int32_t a1 = static_cast<int32_t>(sA[k_local + 1]);
                const int32_t a2 = static_cast<int32_t>(sA[k_local + 2]);
                const int32_t a3 = static_cast<int32_t>(sA[k_local + 3]);

                const int global_kg = k_group_base + kg;
                const uint32_t p0 = *reinterpret_cast<const uint32_t*>(
                    d_B_int8_vnni + (static_cast<size_t>(global_kg) * N + (n4 + 0)) * 4);
                const uint32_t p1 = *reinterpret_cast<const uint32_t*>(
                    d_B_int8_vnni + (static_cast<size_t>(global_kg) * N + (n4 + 1)) * 4);
                const uint32_t p2 = *reinterpret_cast<const uint32_t*>(
                    d_B_int8_vnni + (static_cast<size_t>(global_kg) * N + (n4 + 2)) * 4);
                const uint32_t p3 = *reinterpret_cast<const uint32_t*>(
                    d_B_int8_vnni + (static_cast<size_t>(global_kg) * N + (n4 + 3)) * 4);

                const int8_t b00 = static_cast<int8_t>(p0 & 0xFF);
                const int8_t b01 = static_cast<int8_t>((p0 >> 8) & 0xFF);
                const int8_t b02 = static_cast<int8_t>((p0 >> 16) & 0xFF);
                const int8_t b03 = static_cast<int8_t>((p0 >> 24) & 0xFF);

                const int8_t b10 = static_cast<int8_t>(p1 & 0xFF);
                const int8_t b11 = static_cast<int8_t>((p1 >> 8) & 0xFF);
                const int8_t b12 = static_cast<int8_t>((p1 >> 16) & 0xFF);
                const int8_t b13 = static_cast<int8_t>((p1 >> 24) & 0xFF);

                const int8_t b20 = static_cast<int8_t>(p2 & 0xFF);
                const int8_t b21 = static_cast<int8_t>((p2 >> 8) & 0xFF);
                const int8_t b22 = static_cast<int8_t>((p2 >> 16) & 0xFF);
                const int8_t b23 = static_cast<int8_t>((p2 >> 24) & 0xFF);

                const int8_t b30 = static_cast<int8_t>(p3 & 0xFF);
                const int8_t b31 = static_cast<int8_t>((p3 >> 8) & 0xFF);
                const int8_t b32 = static_cast<int8_t>((p3 >> 16) & 0xFF);
                const int8_t b33 = static_cast<int8_t>((p3 >> 24) & 0xFF);

                acc.x += a0 * static_cast<int32_t>(b00) + a1 * static_cast<int32_t>(b01)
                       + a2 * static_cast<int32_t>(b02) + a3 * static_cast<int32_t>(b03);
                acc.y += a0 * static_cast<int32_t>(b10) + a1 * static_cast<int32_t>(b11)
                       + a2 * static_cast<int32_t>(b12) + a3 * static_cast<int32_t>(b13);
                acc.z += a0 * static_cast<int32_t>(b20) + a1 * static_cast<int32_t>(b21)
                       + a2 * static_cast<int32_t>(b22) + a3 * static_cast<int32_t>(b23);
                acc.w += a0 * static_cast<int32_t>(b30) + a1 * static_cast<int32_t>(b31)
                       + a2 * static_cast<int32_t>(b32) + a3 * static_cast<int32_t>(b33);
            }
        }
        __syncthreads();
    }

    __shared__ int4 partial[SPLIT][TILE_N / 4];
    partial[lane_k][lane_n] = acc;
    __syncthreads();

    if (lane_k == 0 && active)
    {
        int4 sum = {0, 0, 0, 0};
        #pragma unroll
        for (int i = 0; i < SPLIT; ++i)
        {
            sum.x += partial[i][lane_n].x;
            sum.y += partial[i][lane_n].y;
            sum.z += partial[i][lane_n].z;
            sum.w += partial[i][lane_n].w;
        }

        d_C_int32[n4 + 0] = sum.x;
        d_C_int32[n4 + 1] = sum.y;
        d_C_int32[n4 + 2] = sum.z;
        d_C_int32[n4 + 3] = sum.w;
    }
}


// Square-ish kernel: scalar threads, shared A tiles, heavier K unroll
__global__ void gemv_int8_int8_square_kernel(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B_int8,
    int32_t* __restrict__ d_C_int32,
    int N,
    int K)
{
    const int n = blockIdx.x * GEMV_SQUARE_TILE_N + threadIdx.x;
    const bool active = (n < N);

    int32_t acc = 0;
    __shared__ int8_t sA[GEMV_INT8_KPAR_TILE_K];

    for (int k_base = 0; k_base < K; k_base += GEMV_INT8_KPAR_TILE_K)
    {
        const int tile_k = min(GEMV_INT8_KPAR_TILE_K, K - k_base);
        for (int i = threadIdx.x; i < tile_k; i += blockDim.x)
        {
            sA[i] = d_A_int8[k_base + i];
        }
        __syncthreads();

        int k_local = 0;
        const int k_end_unrolled = tile_k - (tile_k % 8);
        for (; k_local < k_end_unrolled; k_local += 8)
        {
            if (active)
            {
                const int k0 = k_base + k_local + 0;
                const int k1 = k_base + k_local + 1;
                const int k2 = k_base + k_local + 2;
                const int k3 = k_base + k_local + 3;
                const int k4 = k_base + k_local + 4;
                const int k5 = k_base + k_local + 5;
                const int k6 = k_base + k_local + 6;
                const int k7 = k_base + k_local + 7;

                acc += static_cast<int32_t>(sA[k_local + 0])
                     * static_cast<int32_t>(d_B_int8[k0 * N + n]);
                acc += static_cast<int32_t>(sA[k_local + 1])
                     * static_cast<int32_t>(d_B_int8[k1 * N + n]);
                acc += static_cast<int32_t>(sA[k_local + 2])
                     * static_cast<int32_t>(d_B_int8[k2 * N + n]);
                acc += static_cast<int32_t>(sA[k_local + 3])
                     * static_cast<int32_t>(d_B_int8[k3 * N + n]);
                acc += static_cast<int32_t>(sA[k_local + 4])
                     * static_cast<int32_t>(d_B_int8[k4 * N + n]);
                acc += static_cast<int32_t>(sA[k_local + 5])
                     * static_cast<int32_t>(d_B_int8[k5 * N + n]);
                acc += static_cast<int32_t>(sA[k_local + 6])
                     * static_cast<int32_t>(d_B_int8[k6 * N + n]);
                acc += static_cast<int32_t>(sA[k_local + 7])
                     * static_cast<int32_t>(d_B_int8[k7 * N + n]);
            }
        }

        for (; k_local < tile_k; ++k_local)
        {
            if (active)
            {
                const int k = k_base + k_local;
                acc += static_cast<int32_t>(sA[k_local])
                     * static_cast<int32_t>(d_B_int8[k * N + n]);
            }
        }

        __syncthreads();
    }

    if (active)
    {
        d_C_int32[n] = acc;
    }
}

// Square-ish VNNI kernel: scalar threads, shared A tiles, K/4 loop
__global__ void gemv_int8_int8_square_vnni_kernel(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B_int8_vnni,
    int32_t* __restrict__ d_C_int32,
    int N,
    int K)
{
    const int n = blockIdx.x * GEMV_INT8_VNNI_SQUARE_TILE_N + threadIdx.x;
    if (n >= N) return;

    int32_t acc = 0;
    __shared__ int8_t sA[GEMV_INT8_VNNI_KPAR_TILE_K];

    for (int k_base = 0; k_base < K; k_base += GEMV_INT8_VNNI_KPAR_TILE_K)
    {
        const int tile_k = min(GEMV_INT8_VNNI_KPAR_TILE_K, K - k_base);
        for (int i = threadIdx.x; i < tile_k; i += blockDim.x)
        {
            sA[i] = d_A_int8[k_base + i];
        }
        __syncthreads();

        const int tile_groups = tile_k / 4;
        const int k_group_base = k_base / 4;
        for (int kg = 0; kg < tile_groups; ++kg)
        {
            const int k_local = kg * 4;
            const int32_t a0 = static_cast<int32_t>(sA[k_local + 0]);
            const int32_t a1 = static_cast<int32_t>(sA[k_local + 1]);
            const int32_t a2 = static_cast<int32_t>(sA[k_local + 2]);
            const int32_t a3 = static_cast<int32_t>(sA[k_local + 3]);

            const int global_kg = k_group_base + kg;
            const uint32_t packed = *reinterpret_cast<const uint32_t*>(
                d_B_int8_vnni + (static_cast<size_t>(global_kg) * N + n) * 4);
            const int8_t b0 = static_cast<int8_t>(packed & 0xFF);
            const int8_t b1 = static_cast<int8_t>((packed >> 8) & 0xFF);
            const int8_t b2 = static_cast<int8_t>((packed >> 16) & 0xFF);
            const int8_t b3 = static_cast<int8_t>((packed >> 24) & 0xFF);

            acc += a0 * static_cast<int32_t>(b0)
                 + a1 * static_cast<int32_t>(b1)
                 + a2 * static_cast<int32_t>(b2)
                 + a3 * static_cast<int32_t>(b3);
        }
        __syncthreads();
    }

    d_C_int32[n] = acc;
}

// ============================================================================
// Grid-level K-parallel VNNI kernel with atomicAdd reduction
//
// Each block processes a TILE_N/CPT-thread tile × one K-segment.
// Grid: (ceil(N/TILE_N), KB) where KB splits K across blockIdx.y.
// Output MUST be zeroed before launch (atomicAdd accumulation).
//
// Key insight: Grid-level K-splitting creates 28×48=1344 independent blocks
// (vs 56 with square_vnni), providing enough wavefronts to fully hide
// HBM2 latency on MI50 (60 CUs × 40 max concurrent waves = 2400).
//
// Benchmark results (MI50, gfx906):
//   Q/Wo (3584×3584):    22.0μs / 58.5% peak BW (9.6× vs square_vnni)
//   FFN Down (3584×18944): 95.7μs / 70.9% peak BW (2.0× vs smem kpar)
// ============================================================================
template <int TILE_N, int CPT>
__global__ void gemv_int8_int8_grid_kpar_vnni_kernel_t(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B_int8_vnni,   // [K/4][N][4] flat VNNI
    int32_t* __restrict__ d_C_int32,
    int N, int K, int kblocks)
{
    const int n_base = blockIdx.x * TILE_N + threadIdx.x * CPT;
    const bool active = (n_base < N);
    const int k_block = blockIdx.y;

    // Each block processes its K-segment: [k_start, k_end)
    const int k_per = (K + kblocks - 1) / kblocks;
    const int k_start = k_block * k_per;
    const int k_end = min(k_start + k_per, K);

    // Accumulate partial dot product for CPT output columns
    int32_t acc[CPT];
    #pragma unroll
    for (int c = 0; c < CPT; ++c) acc[c] = 0;

    if (active)
    {
        // Process VNNI k-groups within this block's K-segment
        const int kg_start = k_start / 4;
        const int kg_end = k_end / 4;

        for (int kg = kg_start; kg < kg_end; ++kg)
        {
            const int kb = kg * 4;
            const int32_t a0 = static_cast<int32_t>(d_A_int8[kb + 0]);
            const int32_t a1 = static_cast<int32_t>(d_A_int8[kb + 1]);
            const int32_t a2 = static_cast<int32_t>(d_A_int8[kb + 2]);
            const int32_t a3 = static_cast<int32_t>(d_A_int8[kb + 3]);

            #pragma unroll
            for (int c = 0; c < CPT; ++c)
            {
                const uint32_t packed = *reinterpret_cast<const uint32_t*>(
                    d_B_int8_vnni + (static_cast<size_t>(kg) * N + n_base + c) * 4);
                const int8_t b0 = static_cast<int8_t>(packed & 0xFF);
                const int8_t b1 = static_cast<int8_t>((packed >> 8) & 0xFF);
                const int8_t b2 = static_cast<int8_t>((packed >> 16) & 0xFF);
                const int8_t b3 = static_cast<int8_t>((packed >> 24) & 0xFF);

                acc[c] += a0 * static_cast<int32_t>(b0)
                        + a1 * static_cast<int32_t>(b1)
                        + a2 * static_cast<int32_t>(b2)
                        + a3 * static_cast<int32_t>(b3);
            }
        }

        // Accumulate partials to output via atomicAdd
        #pragma unroll
        for (int c = 0; c < CPT; ++c)
        {
            atomicAdd(&d_C_int32[n_base + c], acc[c]);
        }
    }
}

// ============================================================================
// Tiled KN VNNI kernel: B stored as [K/TK][N/TN][(TK/4)][TN][4]
//
// Each 128×128 tile has sequential memory access (256B stride vs 76KB flat).
// 1 thread per output column within the tile, blockDim = TN.
//
// Benchmark results (MI50, gfx906):
//   Gate/Up (18944×3584): 89.9μs / 75.6% peak BW (2.17× vs flat wide2)
// ============================================================================
template <int TN, int TK>
__global__ void gemv_int8_int8_tiled_vnni_kernel_t(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B_int8_tiled_vnni,  // [K/TK][N/TN][(TK/4)][TN][4]
    int32_t* __restrict__ d_C_int32,
    int N, int K)
{
    const int n_tile = blockIdx.x;
    const int n_local = threadIdx.x;
    const int n = n_tile * TN + n_local;
    if (n >= N) return;

    int32_t acc = 0;

    const int k_tiles = K / TK;
    const int n_tiles = N / TN;
    const int inner_groups = TK / 4;
    const size_t tile_stride = static_cast<size_t>(inner_groups) * TN * 4;

    for (int kt = 0; kt < k_tiles; ++kt)
    {
        const int8_t* tile_ptr = d_B_int8_tiled_vnni
            + (static_cast<size_t>(kt) * n_tiles + n_tile) * tile_stride;

        for (int ig = 0; ig < inner_groups; ++ig)
        {
            const int k = kt * TK + ig * 4;
            const int32_t a0 = static_cast<int32_t>(d_A_int8[k + 0]);
            const int32_t a1 = static_cast<int32_t>(d_A_int8[k + 1]);
            const int32_t a2 = static_cast<int32_t>(d_A_int8[k + 2]);
            const int32_t a3 = static_cast<int32_t>(d_A_int8[k + 3]);

            const uint32_t packed = *reinterpret_cast<const uint32_t*>(
                tile_ptr + (static_cast<size_t>(ig) * TN + n_local) * 4);
            const int8_t b0 = static_cast<int8_t>(packed & 0xFF);
            const int8_t b1 = static_cast<int8_t>((packed >> 8) & 0xFF);
            const int8_t b2 = static_cast<int8_t>((packed >> 16) & 0xFF);
            const int8_t b3 = static_cast<int8_t>((packed >> 24) & 0xFF);

            acc += a0 * static_cast<int32_t>(b0)
                 + a1 * static_cast<int32_t>(b1)
                 + a2 * static_cast<int32_t>(b2)
                 + a3 * static_cast<int32_t>(b3);
        }
    }

    d_C_int32[n] = acc;
}

// INT8×INT8 GEMV with K-tiles: accumulates partials into INT32 output
__global__ void gemv_int8_int8_ktiles_kernel(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B_int8,
    int32_t* __restrict__ d_C_int32,
    int N,
    int K)
{
    const int n = blockIdx.x * GEMV_KTILE_N + threadIdx.x;
    if (n >= N) return;

    const int k_tile = blockIdx.y;
    const int k_start = k_tile * GEMV_KTILE_K;
    const int k_end = min(k_start + GEMV_KTILE_K, K);

    int32_t acc = 0;
    for (int k = k_start; k < k_end; ++k)
    {
        acc += static_cast<int32_t>(d_A_int8[k]) * static_cast<int32_t>(d_B_int8[k * N + n]);
    }

    atomicAdd(&d_C_int32[n], acc);
}

__global__ void gemv_apply_scales_bias_kernel(
    float* __restrict__ d_C,
    const float* __restrict__ d_scale_B,
    const float* __restrict__ d_bias,
    int N)
{
    const int n = blockIdx.x * 256 + threadIdx.x;
    if (n >= N) return;

    float result = d_C[n] * d_scale_B[n];
    if (d_bias) result += d_bias[n];
    d_C[n] = result;
}
// Vectorized variant: each thread handles 4 columns (N must be divisible by 4)
__global__ void gemv_int8_fp32_kpar_vec4_kernel(
    const float* __restrict__ d_A,          // [K] FP32 activation vector
    const int8_t* __restrict__ d_B,         // [K × N] INT8 weights, row-major
    const float* __restrict__ d_scale_B,    // [N] per-column weight scales
    const float* __restrict__ d_bias,       // [N] optional bias (nullable)
    float* __restrict__ d_C,                // [N] FP32 output
    int N,
    int K)
{
    const int lane_n = threadIdx.x;
    const int n4 = blockIdx.x * GEMV_KPAR_TILE_N + lane_n * 4;
    const bool active = (n4 < N);

    const int lane_k = threadIdx.y;
    float4 acc = {0.0f, 0.0f, 0.0f, 0.0f};

    __shared__ float sA[GEMV_KPAR_TILE_K];
    for (int k_base = 0; k_base < K; k_base += GEMV_KPAR_TILE_K)
    {
        const int tile_k = min(GEMV_KPAR_TILE_K, K - k_base);
        const int linear_tid = threadIdx.y * blockDim.x + threadIdx.x;
        for (int i = linear_tid; i < tile_k; i += blockDim.x * blockDim.y)
        {
            sA[i] = d_A[k_base + i];
        }
        __syncthreads();

        for (int k_local = lane_k; k_local < tile_k; k_local += GEMV_KPAR_SPLIT)
        {
            const int k = k_base + k_local;
            if (active)
            {
                const uint32_t packed = *reinterpret_cast<const uint32_t*>(
                    d_B + k * N + n4);

                const int8_t b0 = static_cast<int8_t>(packed & 0xFF);
                const int8_t b1 = static_cast<int8_t>((packed >> 8) & 0xFF);
                const int8_t b2 = static_cast<int8_t>((packed >> 16) & 0xFF);
                const int8_t b3 = static_cast<int8_t>((packed >> 24) & 0xFF);

                const float a = sA[k_local];
                acc.x += a * static_cast<float>(b0);
                acc.y += a * static_cast<float>(b1);
                acc.z += a * static_cast<float>(b2);
                acc.w += a * static_cast<float>(b3);
            }
        }
        __syncthreads();
    }

    __shared__ float4 partial[GEMV_KPAR_SPLIT][GEMV_KPAR_TILE_N / 4];
    partial[lane_k][lane_n] = acc;
    __syncthreads();

    if (lane_k == 0 && active)
    {
        float4 sum = {0.0f, 0.0f, 0.0f, 0.0f};
        #pragma unroll
        for (int i = 0; i < GEMV_KPAR_SPLIT; ++i)
        {
            sum.x += partial[i][lane_n].x;
            sum.y += partial[i][lane_n].y;
            sum.z += partial[i][lane_n].z;
            sum.w += partial[i][lane_n].w;
        }

        float r0 = sum.x * d_scale_B[n4 + 0];
        float r1 = sum.y * d_scale_B[n4 + 1];
        float r2 = sum.z * d_scale_B[n4 + 2];
        float r3 = sum.w * d_scale_B[n4 + 3];
        if (d_bias)
        {
            r0 += d_bias[n4 + 0];
            r1 += d_bias[n4 + 1];
            r2 += d_bias[n4 + 2];
            r3 += d_bias[n4 + 3];
        }

        d_C[n4 + 0] = r0;
        d_C[n4 + 1] = r1;
        d_C[n4 + 2] = r2;
        d_C[n4 + 3] = r3;
    }
}

// Square-ish kernel: scalar threads, shared A tiles, heavier K unroll
__global__ void gemv_int8_fp32_square_kernel(
    const float* __restrict__ d_A,
    const int8_t* __restrict__ d_B,
    const float* __restrict__ d_scale_B,
    const float* __restrict__ d_bias,
    float* __restrict__ d_C,
    int N,
    int K)
{
    const int n = blockIdx.x * GEMV_SQUARE_TILE_N + threadIdx.x;
    const bool active = (n < N);

    float acc = 0.0f;
    __shared__ float sA[GEMV_KPAR_TILE_K];

    for (int k_base = 0; k_base < K; k_base += GEMV_KPAR_TILE_K)
    {
        const int tile_k = min(GEMV_KPAR_TILE_K, K - k_base);
        for (int i = threadIdx.x; i < tile_k; i += blockDim.x)
        {
            sA[i] = d_A[k_base + i];
        }
        __syncthreads();

        int k_local = 0;
        const int k_end_unrolled = tile_k - (tile_k % 8);
        for (; k_local < k_end_unrolled; k_local += 8)
        {
            if (active)
            {
                const int k0 = k_base + k_local + 0;
                const int k1 = k_base + k_local + 1;
                const int k2 = k_base + k_local + 2;
                const int k3 = k_base + k_local + 3;
                const int k4 = k_base + k_local + 4;
                const int k5 = k_base + k_local + 5;
                const int k6 = k_base + k_local + 6;
                const int k7 = k_base + k_local + 7;

                acc += sA[k_local + 0] * static_cast<float>(d_B[k0 * N + n]);
                acc += sA[k_local + 1] * static_cast<float>(d_B[k1 * N + n]);
                acc += sA[k_local + 2] * static_cast<float>(d_B[k2 * N + n]);
                acc += sA[k_local + 3] * static_cast<float>(d_B[k3 * N + n]);
                acc += sA[k_local + 4] * static_cast<float>(d_B[k4 * N + n]);
                acc += sA[k_local + 5] * static_cast<float>(d_B[k5 * N + n]);
                acc += sA[k_local + 6] * static_cast<float>(d_B[k6 * N + n]);
                acc += sA[k_local + 7] * static_cast<float>(d_B[k7 * N + n]);
            }
        }

        for (; k_local < tile_k; ++k_local)
        {
            if (active)
            {
                const int k = k_base + k_local;
                acc += sA[k_local] * static_cast<float>(d_B[k * N + n]);
            }
        }

        __syncthreads();
    }

    if (active)
    {
        float result = acc * d_scale_B[n];
        if (d_bias) result += d_bias[n];
        d_C[n] = result;
    }
}


// Wide-N kernel: vec4, simple K loop, larger N tile to improve occupancy
__global__ void gemv_int8_fp32_wide_vec4_kernel(
    const float* __restrict__ d_A,
    const int8_t* __restrict__ d_B,
    const float* __restrict__ d_scale_B,
    const float* __restrict__ d_bias,
    float* __restrict__ d_C,
    int N,
    int K)
{
    const int lane_n = threadIdx.x;
    const int n4 = blockIdx.x * GEMV_WIDE_TILE_N + lane_n * 4;
    if (n4 >= N) return;

    float4 acc = {0.0f, 0.0f, 0.0f, 0.0f};
    int k = 0;
    const int k_end_unrolled = K - (K % 4);
    for (; k < k_end_unrolled; k += 4)
    {
        const float a0 = d_A[k + 0];
        const float a1 = d_A[k + 1];
        const float a2 = d_A[k + 2];
        const float a3 = d_A[k + 3];

        const uint32_t p0 = *reinterpret_cast<const uint32_t*>(d_B + (k + 0) * N + n4);
        const uint32_t p1 = *reinterpret_cast<const uint32_t*>(d_B + (k + 1) * N + n4);
        const uint32_t p2 = *reinterpret_cast<const uint32_t*>(d_B + (k + 2) * N + n4);
        const uint32_t p3 = *reinterpret_cast<const uint32_t*>(d_B + (k + 3) * N + n4);

        const int8_t b00 = static_cast<int8_t>(p0 & 0xFF);
        const int8_t b01 = static_cast<int8_t>((p0 >> 8) & 0xFF);
        const int8_t b02 = static_cast<int8_t>((p0 >> 16) & 0xFF);
        const int8_t b03 = static_cast<int8_t>((p0 >> 24) & 0xFF);

        const int8_t b10 = static_cast<int8_t>(p1 & 0xFF);
        const int8_t b11 = static_cast<int8_t>((p1 >> 8) & 0xFF);
        const int8_t b12 = static_cast<int8_t>((p1 >> 16) & 0xFF);
        const int8_t b13 = static_cast<int8_t>((p1 >> 24) & 0xFF);

        const int8_t b20 = static_cast<int8_t>(p2 & 0xFF);
        const int8_t b21 = static_cast<int8_t>((p2 >> 8) & 0xFF);
        const int8_t b22 = static_cast<int8_t>((p2 >> 16) & 0xFF);
        const int8_t b23 = static_cast<int8_t>((p2 >> 24) & 0xFF);

        const int8_t b30 = static_cast<int8_t>(p3 & 0xFF);
        const int8_t b31 = static_cast<int8_t>((p3 >> 8) & 0xFF);
        const int8_t b32 = static_cast<int8_t>((p3 >> 16) & 0xFF);
        const int8_t b33 = static_cast<int8_t>((p3 >> 24) & 0xFF);

        acc.x += a0 * static_cast<float>(b00) + a1 * static_cast<float>(b10)
               + a2 * static_cast<float>(b20) + a3 * static_cast<float>(b30);
        acc.y += a0 * static_cast<float>(b01) + a1 * static_cast<float>(b11)
               + a2 * static_cast<float>(b21) + a3 * static_cast<float>(b31);
        acc.z += a0 * static_cast<float>(b02) + a1 * static_cast<float>(b12)
               + a2 * static_cast<float>(b22) + a3 * static_cast<float>(b32);
        acc.w += a0 * static_cast<float>(b03) + a1 * static_cast<float>(b13)
               + a2 * static_cast<float>(b23) + a3 * static_cast<float>(b33);
    }

    for (; k < K; ++k)
    {
        const uint32_t packed = *reinterpret_cast<const uint32_t*>(d_B + k * N + n4);
        const int8_t b0 = static_cast<int8_t>(packed & 0xFF);
        const int8_t b1 = static_cast<int8_t>((packed >> 8) & 0xFF);
        const int8_t b2 = static_cast<int8_t>((packed >> 16) & 0xFF);
        const int8_t b3 = static_cast<int8_t>((packed >> 24) & 0xFF);
        const float a = d_A[k];
        acc.x += a * static_cast<float>(b0);
        acc.y += a * static_cast<float>(b1);
        acc.z += a * static_cast<float>(b2);
        acc.w += a * static_cast<float>(b3);
    }

    float r0 = acc.x * d_scale_B[n4 + 0];
    float r1 = acc.y * d_scale_B[n4 + 1];
    float r2 = acc.z * d_scale_B[n4 + 2];
    float r3 = acc.w * d_scale_B[n4 + 3];
    if (d_bias)
    {
        r0 += d_bias[n4 + 0];
        r1 += d_bias[n4 + 1];
        r2 += d_bias[n4 + 2];
        r3 += d_bias[n4 + 3];
    }

    d_C[n4 + 0] = r0;
    d_C[n4 + 1] = r1;
    d_C[n4 + 2] = r2;
    d_C[n4 + 3] = r3;
}

// Wide-N kernel: 2 columns per thread, simple K loop, no shared A
__global__ void gemv_int8_fp32_wide2_kernel(
    const float* __restrict__ d_A,
    const int8_t* __restrict__ d_B,
    const float* __restrict__ d_scale_B,
    const float* __restrict__ d_bias,
    float* __restrict__ d_C,
    int N,
    int K)
{
    const int n2 = blockIdx.x * GEMV_WIDE2_TILE_N + threadIdx.x * 2;
    if (n2 >= N) return;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    int k = 0;
    const int k_end_unrolled = K - (K % 4);
    for (; k < k_end_unrolled; k += 4)
    {
        const float a0 = d_A[k + 0];
        const float a1 = d_A[k + 1];
        const float a2 = d_A[k + 2];
        const float a3 = d_A[k + 3];

        const uint16_t p0 = *reinterpret_cast<const uint16_t*>(d_B + (k + 0) * N + n2);
        const uint16_t p1 = *reinterpret_cast<const uint16_t*>(d_B + (k + 1) * N + n2);
        const uint16_t p2 = *reinterpret_cast<const uint16_t*>(d_B + (k + 2) * N + n2);
        const uint16_t p3 = *reinterpret_cast<const uint16_t*>(d_B + (k + 3) * N + n2);

        const int8_t b00 = static_cast<int8_t>(p0 & 0xFF);
        const int8_t b01 = static_cast<int8_t>((p0 >> 8) & 0xFF);
        const int8_t b10 = static_cast<int8_t>(p1 & 0xFF);
        const int8_t b11 = static_cast<int8_t>((p1 >> 8) & 0xFF);
        const int8_t b20 = static_cast<int8_t>(p2 & 0xFF);
        const int8_t b21 = static_cast<int8_t>((p2 >> 8) & 0xFF);
        const int8_t b30 = static_cast<int8_t>(p3 & 0xFF);
        const int8_t b31 = static_cast<int8_t>((p3 >> 8) & 0xFF);

        acc0 += a0 * static_cast<float>(b00) + a1 * static_cast<float>(b10)
              + a2 * static_cast<float>(b20) + a3 * static_cast<float>(b30);
        acc1 += a0 * static_cast<float>(b01) + a1 * static_cast<float>(b11)
              + a2 * static_cast<float>(b21) + a3 * static_cast<float>(b31);
    }

    for (; k < K; ++k)
    {
        const uint16_t packed = *reinterpret_cast<const uint16_t*>(d_B + k * N + n2);
        const int8_t b0 = static_cast<int8_t>(packed & 0xFF);
        const int8_t b1 = static_cast<int8_t>((packed >> 8) & 0xFF);
        const float a = d_A[k];
        acc0 += a * static_cast<float>(b0);
        acc1 += a * static_cast<float>(b1);
    }

    float r0 = acc0 * d_scale_B[n2 + 0];
    float r1 = acc1 * d_scale_B[n2 + 1];
    if (d_bias)
    {
        r0 += d_bias[n2 + 0];
        r1 += d_bias[n2 + 1];
    }

    d_C[n2 + 0] = r0;
    d_C[n2 + 1] = r1;
}


// ============================================================================
// C API: Launch wrappers
// ============================================================================

extern "C" {

static bool launch_ffn_down_variant(
    const int8_t* d_A_int8,
    const int8_t* d_B_int8,
    int32_t* d_C_int32,
    int N, int K,
    int tile_n, int split,
    int device_id)
{
    hipSetDevice(device_id);

    if ((tile_n % 4) != 0 || (N % 4) != 0 || split <= 0)
        return false;

    const int grid_n = (N + tile_n - 1) / tile_n;
    const dim3 block(tile_n / 4, split, 1);

    #define LAUNCH_FFND(TN, SPL) \
        hipLaunchKernelGGL( \
            HIP_KERNEL_NAME(gemv_int8_int8_ffn_down_kernel_t<TN, SPL>), \
            dim3(grid_n), block, 0, 0, \
            d_A_int8, d_B_int8, d_C_int32, N, K)

    if (tile_n == 128)
    {
        switch (split)
        {
        case 4:
            LAUNCH_FFND(128, 4);
            break;
        case 8:
            LAUNCH_FFND(128, 8);
            break;
        case 16:
            LAUNCH_FFND(128, 16);
            break;
        default:
            return false;
        }
    }
    else if (tile_n == 256)
    {
        switch (split)
        {
        case 4:
            LAUNCH_FFND(256, 4);
            break;
        case 8:
            LAUNCH_FFND(256, 8);
            break;
        case 16:
            LAUNCH_FFND(256, 16);
            break;
        default:
            return false;
        }
    }
    else
    {
        return false;
    }

    #undef LAUNCH_FFND

    hipError_t err = hipGetLastError();
    return err == hipSuccess;
}

static bool launch_vnni_kpar_variant(
    const int8_t* d_A_int8,
    const int8_t* d_B_int8_vnni,
    int32_t* d_C_int32,
    int N, int K,
    int tile_n, int split,
    int device_id)
{
    hipSetDevice(device_id);

    if ((tile_n % 4) != 0 || (N % 4) != 0 || split <= 0 || (K % 4) != 0)
        return false;

    const int grid_n = (N + tile_n - 1) / tile_n;
    const dim3 block(tile_n / 4, split, 1);

    #define LAUNCH_VNNI_KPAR(TN, SPL) \
        hipLaunchKernelGGL( \
            HIP_KERNEL_NAME(gemv_int8_int8_ffn_down_vnni_kpar_kernel_t<TN, SPL>), \
            dim3(grid_n), block, 0, 0, \
            d_A_int8, d_B_int8_vnni, d_C_int32, N, K)

    if (tile_n == 64)
    {
        switch (split)
        {
        case 4:
            LAUNCH_VNNI_KPAR(64, 4);
            break;
        case 8:
            LAUNCH_VNNI_KPAR(64, 8);
            break;
        case 16:
            LAUNCH_VNNI_KPAR(64, 16);
            break;
        default:
            return false;
        }
    }
    else if (tile_n == 128)
    {
        switch (split)
        {
        case 4:
            LAUNCH_VNNI_KPAR(128, 4);
            break;
        case 8:
            LAUNCH_VNNI_KPAR(128, 8);
            break;
        case 16:
            LAUNCH_VNNI_KPAR(128, 16);
            break;
        default:
            return false;
        }
    }
    else
    {
        return false;
    }

    #undef LAUNCH_VNNI_KPAR

    hipError_t err = hipGetLastError();
    return err == hipSuccess;
}


static bool launch_vnni_wide_vec4_variant(
    const int8_t* d_A_int8,
    const int8_t* d_B_int8_vnni,
    int32_t* d_C_int32,
    int N, int K,
    int tile_n,
    int device_id)
{
    hipSetDevice(device_id);

    if ((tile_n % 4) != 0 || (N % 4) != 0 || (K % 4) != 0)
        return false;

    const int grid_n = (N + tile_n - 1) / tile_n;
    const dim3 block(tile_n / 4, 1, 1);

    #define LAUNCH_VNNI_WIDE4(TN) \
        hipLaunchKernelGGL( \
            HIP_KERNEL_NAME(gemv_int8_int8_wide_vec4_vnni_kernel_t<TN>), \
            dim3(grid_n), block, 0, 0, \
            d_A_int8, d_B_int8_vnni, d_C_int32, N, K)

    if (tile_n == 256)
    {
        LAUNCH_VNNI_WIDE4(256);
    }
    else if (tile_n == 512)
    {
        LAUNCH_VNNI_WIDE4(512);
    }
    else if (tile_n == 1024)
    {
        LAUNCH_VNNI_WIDE4(1024);
    }
    else
    {
        return false;
    }

    #undef LAUNCH_VNNI_WIDE4

    hipError_t err = hipGetLastError();
    return err == hipSuccess;
}

static bool launch_vnni_wide2_variant(
    const int8_t* d_A_int8,
    const int8_t* d_B_int8_vnni,
    int32_t* d_C_int32,
    int N, int K,
    int tile_n,
    int device_id)
{
    hipSetDevice(device_id);

    if ((tile_n % 2) != 0 || (N % 2) != 0 || (K % 4) != 0)
        return false;

    const int grid_n = (N + tile_n - 1) / tile_n;
    const dim3 block(tile_n / 2, 1, 1);

    #define LAUNCH_VNNI_WIDE2(TN) \
        hipLaunchKernelGGL( \
            HIP_KERNEL_NAME(gemv_int8_int8_wide2_vnni_kernel_t<TN>), \
            dim3(grid_n), block, 0, 0, \
            d_A_int8, d_B_int8_vnni, d_C_int32, N, K)

    if (tile_n == 256)
    {
        LAUNCH_VNNI_WIDE2(256);
    }
    else if (tile_n == 512)
    {
        LAUNCH_VNNI_WIDE2(512);
    }
    else if (tile_n == 1024)
    {
        LAUNCH_VNNI_WIDE2(1024);
    }
    else
    {
        return false;
    }

    #undef LAUNCH_VNNI_WIDE2

    hipError_t err = hipGetLastError();
    return err == hipSuccess;
}

static bool launch_ffn_down_vnni_variant(
    const int8_t* d_A_int8,
    const int8_t* d_B_int8_vnni,
    int32_t* d_C_int32,
    int N, int K,
    int tile_n, int split,
    int device_id)
{
    hipSetDevice(device_id);

    if ((tile_n % 4) != 0 || (N % 4) != 0 || split <= 0 || (K % 4) != 0)
        return false;

    const int grid_n = (N + tile_n - 1) / tile_n;
    const dim3 block(tile_n / 4, split, 1);

    #define LAUNCH_FFND_VNNI(TN, SPL) \
        hipLaunchKernelGGL( \
            HIP_KERNEL_NAME(gemv_int8_int8_ffn_down_vnni_kpar_kernel_t<TN, SPL>), \
            dim3(grid_n), block, 0, 0, \
            d_A_int8, d_B_int8_vnni, d_C_int32, N, K)

    if (tile_n == 128)
    {
        switch (split)
        {
        case 4:
            LAUNCH_FFND_VNNI(128, 4);
            break;
        case 8:
            LAUNCH_FFND_VNNI(128, 8);
            break;
        case 16:
            LAUNCH_FFND_VNNI(128, 16);
            break;
        default:
            return false;
        }
    }
    else if (tile_n == 256)
    {
        switch (split)
        {
        case 4:
            LAUNCH_FFND_VNNI(256, 4);
            break;
        case 8:
            LAUNCH_FFND_VNNI(256, 8);
            break;
        case 16:
            LAUNCH_FFND_VNNI(256, 16);
            break;
        default:
            return false;
        }
    }
    else
    {
        return false;
    }

    #undef LAUNCH_FFND_VNNI

    hipError_t err = hipGetLastError();
    return err == hipSuccess;
}

bool rocmGemv_int8_int8_ffn_down_variant(
    const int8_t* d_A_int8,
    const int8_t* d_B_int8,
    int32_t* d_C_int32,
    int N, int K,
    int tile_n, int split,
    int device_id)
{
    return launch_ffn_down_variant(d_A_int8, d_B_int8, d_C_int32,
                                   N, K, tile_n, split, device_id);
}

bool rocmGemv_int8_int8_ffn_down_vnni_variant(
    const int8_t* d_A_int8,
    const int8_t* d_B_int8_vnni,
    int32_t* d_C_int32,
    int N, int K,
    int tile_n, int split,
    int device_id)
{
    return launch_ffn_down_vnni_variant(d_A_int8, d_B_int8_vnni, d_C_int32,
                                        N, K, tile_n, split, device_id);
}

bool rocmGemv_int8_int8_vnni_kpar_variant(
    const int8_t* d_A_int8,
    const int8_t* d_B_int8_vnni,
    int32_t* d_C_int32,
    int N, int K,
    int tile_n, int split,
    int device_id)
{
    return launch_vnni_kpar_variant(d_A_int8, d_B_int8_vnni, d_C_int32,
                                    N, K, tile_n, split, device_id);
}


bool rocmGemv_int8_int8_vnni_wide_vec4_variant(
    const int8_t* d_A_int8,
    const int8_t* d_B_int8_vnni,
    int32_t* d_C_int32,
    int N, int K,
    int tile_n,
    int device_id)
{
    return launch_vnni_wide_vec4_variant(d_A_int8, d_B_int8_vnni, d_C_int32,
                                         N, K, tile_n, device_id);
}

bool rocmGemv_int8_int8_vnni_wide2_variant(
    const int8_t* d_A_int8,
    const int8_t* d_B_int8_vnni,
    int32_t* d_C_int32,
    int N, int K,
    int tile_n,
    int device_id)
{
    return launch_vnni_wide2_variant(d_A_int8, d_B_int8_vnni, d_C_int32,
                                     N, K, tile_n, device_id);
}

/**
 * @brief Execute GEMV: output[N] = A[K] × B[K×N] × scale_B[N]
 *
 * All pointers are device pointers. No host-device copies.
 *
 * Uses the simple per-column kernel for most shapes.
 * Falls back to K-parallel reduction for very large K × N.
 *
 * @param d_A       [K] FP32 activation vector (device)
 * @param d_B       [K × N] INT8 weight matrix, row-major (device)
 * @param d_scale_B [N] per-column FP32 weight scales (device)
 * @param d_C       [N] FP32 output vector (device)
 * @param N         Number of output features (columns)
 * @param K         Number of input features (reduction dimension)
 * @param device_id HIP device ID
 * @return true on success
 */
bool rocmGemv_int8_fp32(
    const float* d_A,
    const int8_t* d_B,
    const float* d_scale_B,
    float* d_C,
    int N, int K,
    int device_id)
{
    hipSetDevice(device_id);

    const bool use_kpar = (K >= 3072) && (K >= 2 * N);
    // Favor k-parallel for large-N FFN Down; keep k-tiles for small-N, very tall K.
    const bool use_ktiles = (K >= 8192) && (K >= 2 * N) && (N <= 1024);
    const bool use_wide = (N >= 3 * K);
    const bool use_vec4 = ((N % 4) == 0);
    const bool use_vec2 = ((N % 2) == 0);
    const bool use_wide_vec4 = use_wide && (N >= 8 * K) && use_vec4;
    const bool use_wide2 = use_wide && (N < 8 * K) && use_vec2 && (N >= GEMV_WIDE2_TILE_N);

    if (use_ktiles)
    {
        const int grid_n = (N + GEMV_KTILE_N - 1) / GEMV_KTILE_N;
        const int grid_k = (K + GEMV_KTILE_K - 1) / GEMV_KTILE_K;
        hipMemset(d_C, 0, static_cast<size_t>(N) * sizeof(float));
        hipLaunchKernelGGL(
            gemv_int8_fp32_ktiles_kernel,
            dim3(grid_n, grid_k, 1), dim3(GEMV_KTILE_N, 1, 1), 0, 0,
            d_A, d_B, d_C, N, K);
        const int grid_scale = (N + 255) / 256;
        hipLaunchKernelGGL(
            gemv_apply_scales_bias_kernel,
            dim3(grid_scale), dim3(256, 1, 1), 0, 0,
            d_C, d_scale_B, nullptr, N);
    }
    else if (use_kpar && use_vec4 && (N >= GEMV_KPAR_TILE_N))
    {
        const int grid_n = (N + GEMV_KPAR_TILE_N - 1) / GEMV_KPAR_TILE_N;
        dim3 block(GEMV_KPAR_TILE_N / 4, GEMV_KPAR_SPLIT, 1);
        hipLaunchKernelGGL(
            gemv_int8_fp32_kpar_vec4_kernel,
            dim3(grid_n), block, 0, 0,
            d_A, d_B, d_scale_B, nullptr, d_C, N, K);
    }
    else if (use_kpar)
    {
        const int grid_n = (N + GEMV_KPAR_TILE_N - 1) / GEMV_KPAR_TILE_N;
        dim3 block(GEMV_KPAR_TILE_N, GEMV_KPAR_SPLIT, 1);
        hipLaunchKernelGGL(
            gemv_int8_fp32_kpar_kernel,
            dim3(grid_n), block, 0, 0,
            d_A, d_B, d_scale_B, nullptr, d_C, N, K);
    }
    else if (use_wide2)
    {
        const int grid_n = (N + GEMV_WIDE2_TILE_N - 1) / GEMV_WIDE2_TILE_N;
        dim3 block(GEMV_WIDE2_TILE_N / 2, 1, 1);
        hipLaunchKernelGGL(
            gemv_int8_fp32_wide2_kernel,
            dim3(grid_n), block, 0, 0,
            d_A, d_B, d_scale_B, nullptr, d_C, N, K);
    }
    else if (use_wide_vec4 && (N >= GEMV_WIDE_TILE_N))
    {
        const int grid_n = (N + GEMV_WIDE_TILE_N - 1) / GEMV_WIDE_TILE_N;
        dim3 block(GEMV_WIDE_TILE_N / 4, 1, 1);
        hipLaunchKernelGGL(
            gemv_int8_fp32_wide_vec4_kernel,
            dim3(grid_n), block, 0, 0,
            d_A, d_B, d_scale_B, nullptr, d_C, N, K);
    }
    else if (N >= GEMV_SQUARE_TILE_N)
    {
        const int grid_n = (N + GEMV_SQUARE_TILE_N - 1) / GEMV_SQUARE_TILE_N;
        dim3 block(GEMV_SQUARE_TILE_N, 1, 1);
        hipLaunchKernelGGL(
            gemv_int8_fp32_square_kernel,
            dim3(grid_n), block, 0, 0,
            d_A, d_B, d_scale_B, nullptr, d_C, N, K);
    }
    else
    {
        const int grid_n = (N + GEMV_TILE_N - 1) / GEMV_TILE_N;
        hipLaunchKernelGGL(
            gemv_int8_fp32_kernel,
            dim3(grid_n), dim3(GEMV_TILE_N), 0, 0,
            d_A, d_B, d_scale_B, nullptr, d_C, N, K);
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "[rocmGemv_int8_fp32] Kernel launch failed: %s (N=%d K=%d)\n",
                hipGetErrorString(err), N, K);
        return false;
    }

    return true;
}

/**
 * @brief Execute GEMV with FP16 math: output[N] = A[K] × B[K×N] × scale_B[N]
 */
bool rocmGemv_int8_fp16(
    const float* d_A,
    const int8_t* d_B,
    const float* d_scale_B,
    float* d_C,
    int N, int K,
    int device_id)
{
    hipSetDevice(device_id);

    const int grid_n = (N + GEMV_TILE_N - 1) / GEMV_TILE_N;
    hipLaunchKernelGGL(
        gemv_int8_fp16_kernel,
        dim3(grid_n), dim3(GEMV_TILE_N), 0, 0,
        d_A, d_B, d_scale_B, nullptr, d_C, N, K);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "[rocmGemv_int8_fp16] Kernel launch failed: %s (N=%d K=%d)\n",
                hipGetErrorString(err), N, K);
        return false;
    }

    return true;
}

/**
 * @brief Execute GEMV with bias: output[N] = A[K] × B[K×N] × scale_B[N] + bias[N]
 *
 * Bias is fused into the GEMV kernel — no extra kernel launch.
 */
bool rocmGemv_int8_fp32_bias(
    const float* d_A,
    const int8_t* d_B,
    const float* d_scale_B,
    const float* d_bias,
    float* d_C,
    int N, int K,
    int device_id)
{
    hipSetDevice(device_id);

    const bool use_kpar = (K >= 3072) && (K >= 2 * N);
    const bool use_ktiles = (K >= 8192) && (K >= 2 * N);
    const bool use_wide = (N >= 3 * K);
    const bool use_vec4 = ((N % 4) == 0);
    const bool use_vec2 = ((N % 2) == 0);
    const bool use_wide_vec4 = use_wide && (N >= 8 * K) && use_vec4;
    const bool use_wide2 = use_wide && (N < 8 * K) && use_vec2 && (N >= GEMV_WIDE2_TILE_N);

    if (use_ktiles)
    {
        const int grid_n = (N + GEMV_KTILE_N - 1) / GEMV_KTILE_N;
        const int grid_k = (K + GEMV_KTILE_K - 1) / GEMV_KTILE_K;
        hipMemset(d_C, 0, static_cast<size_t>(N) * sizeof(float));
        hipLaunchKernelGGL(
            gemv_int8_fp32_ktiles_kernel,
            dim3(grid_n, grid_k, 1), dim3(GEMV_KTILE_N, 1, 1), 0, 0,
            d_A, d_B, d_C, N, K);
        const int grid_scale = (N + 255) / 256;
        hipLaunchKernelGGL(
            gemv_apply_scales_bias_kernel,
            dim3(grid_scale), dim3(256, 1, 1), 0, 0,
            d_C, d_scale_B, d_bias, N);
    }
    else if (use_kpar && use_vec4 && (N >= GEMV_KPAR_TILE_N))
    {
        const int grid_n = (N + GEMV_KPAR_TILE_N - 1) / GEMV_KPAR_TILE_N;
        dim3 block(GEMV_KPAR_TILE_N / 4, GEMV_KPAR_SPLIT, 1);
        hipLaunchKernelGGL(
            gemv_int8_fp32_kpar_vec4_kernel,
            dim3(grid_n), block, 0, 0,
            d_A, d_B, d_scale_B, d_bias, d_C, N, K);
    }
    else if (use_kpar)
    {
        const int grid_n = (N + GEMV_KPAR_TILE_N - 1) / GEMV_KPAR_TILE_N;
        dim3 block(GEMV_KPAR_TILE_N, GEMV_KPAR_SPLIT, 1);
        hipLaunchKernelGGL(
            gemv_int8_fp32_kpar_kernel,
            dim3(grid_n), block, 0, 0,
            d_A, d_B, d_scale_B, d_bias, d_C, N, K);
    }
    else if (use_wide2)
    {
        const int grid_n = (N + GEMV_WIDE2_TILE_N - 1) / GEMV_WIDE2_TILE_N;
        dim3 block(GEMV_WIDE2_TILE_N / 2, 1, 1);
        hipLaunchKernelGGL(
            gemv_int8_fp32_wide2_kernel,
            dim3(grid_n), block, 0, 0,
            d_A, d_B, d_scale_B, d_bias, d_C, N, K);
    }
    else if (use_wide_vec4 && (N >= GEMV_WIDE_TILE_N))
    {
        const int grid_n = (N + GEMV_WIDE_TILE_N - 1) / GEMV_WIDE_TILE_N;
        dim3 block(GEMV_WIDE_TILE_N / 4, 1, 1);
        hipLaunchKernelGGL(
            gemv_int8_fp32_wide_vec4_kernel,
            dim3(grid_n), block, 0, 0,
            d_A, d_B, d_scale_B, d_bias, d_C, N, K);
    }
    else if (N >= GEMV_SQUARE_TILE_N)
    {
        const int grid_n = (N + GEMV_SQUARE_TILE_N - 1) / GEMV_SQUARE_TILE_N;
        dim3 block(GEMV_SQUARE_TILE_N, 1, 1);
        hipLaunchKernelGGL(
            gemv_int8_fp32_square_kernel,
            dim3(grid_n), block, 0, 0,
            d_A, d_B, d_scale_B, d_bias, d_C, N, K);
    }
    else
    {
        const int grid_n = (N + GEMV_TILE_N - 1) / GEMV_TILE_N;
        hipLaunchKernelGGL(
            gemv_int8_fp32_kernel,
            dim3(grid_n), dim3(GEMV_TILE_N), 0, 0,
            d_A, d_B, d_scale_B, d_bias, d_C, N, K);
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "[rocmGemv_int8_fp32_bias] Kernel launch failed: %s (N=%d K=%d)\n",
                hipGetErrorString(err), N, K);
        return false;
    }

    return true;
}

/**
 * @brief Execute GEMV with FP16 math and bias
 */
bool rocmGemv_int8_fp16_bias(
    const float* d_A,
    const int8_t* d_B,
    const float* d_scale_B,
    const float* d_bias,
    float* d_C,
    int N, int K,
    int device_id)
{
    hipSetDevice(device_id);

    const int grid_n = (N + GEMV_TILE_N - 1) / GEMV_TILE_N;
    hipLaunchKernelGGL(
        gemv_int8_fp16_kernel,
        dim3(grid_n), dim3(GEMV_TILE_N), 0, 0,
        d_A, d_B, d_scale_B, d_bias, d_C, N, K);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "[rocmGemv_int8_fp16_bias] Kernel launch failed: %s (N=%d K=%d)\n",
                hipGetErrorString(err), N, K);
        return false;
    }

    return true;
}

/**
 * @brief Execute INT8×INT8 GEMV: output_int32[N] = A_int8[K] × B_int8[K×N]
 */
bool rocmGemv_int8_int8_int32(
    const int8_t* d_A_int8,
    const int8_t* d_B_int8,
    int32_t* d_C_int32,
    int N, int K,
    int device_id)
{
    hipSetDevice(device_id);

    const bool use_kpar = (K >= 3072) && (K >= 2 * N);
    const bool use_ktiles = (K >= 8192) && (K >= 2 * N);
    const bool use_wide = (N >= 3 * K);
    const bool use_vec4 = ((N % 4) == 0);
    const bool use_vec2 = ((N % 2) == 0);
    const bool use_wide_vec4 = use_wide && (N >= 8 * K) && use_vec4;
    const bool use_wide2 = use_wide && (N < 8 * K) && use_vec2 && (N >= GEMV_WIDE2_TILE_N);

    const bool use_ffn_down = use_vec4 && (K >= 4096) && (N == 896 || N == 3584);

    if (use_ffn_down)
    {
        if (!launch_ffn_down_variant(d_A_int8, d_B_int8, d_C_int32,
                                     N, K,
                                     GEMV_INT8_FFND_TILE_N,
                                     GEMV_INT8_FFND_SPLIT,
                                     device_id))
        {
            return false;
        }
    }
    else if (use_ktiles)
    {
        const int grid_n = (N + GEMV_KTILE_N - 1) / GEMV_KTILE_N;
        const int grid_k = (K + GEMV_KTILE_K - 1) / GEMV_KTILE_K;
        hipMemset(d_C_int32, 0, static_cast<size_t>(N) * sizeof(int32_t));
        hipLaunchKernelGGL(
            gemv_int8_int8_ktiles_kernel,
            dim3(grid_n, grid_k, 1), dim3(GEMV_KTILE_N, 1, 1), 0, 0,
            d_A_int8, d_B_int8, d_C_int32, N, K);
    }
    else if (use_kpar && use_vec4 && (N >= GEMV_INT8_KPAR_TILE_N))
    {
        const int grid_n = (N + GEMV_INT8_KPAR_TILE_N - 1) / GEMV_INT8_KPAR_TILE_N;
        dim3 block(GEMV_INT8_KPAR_TILE_N / 4, GEMV_INT8_KPAR_SPLIT, 1);
        hipLaunchKernelGGL(
            gemv_int8_int8_kpar_vec4_kernel,
            dim3(grid_n), block, 0, 0,
            d_A_int8, d_B_int8, d_C_int32, N, K);
    }
    else if (use_kpar)
    {
        const int grid_n = (N + GEMV_INT8_KPAR_TILE_N - 1) / GEMV_INT8_KPAR_TILE_N;
        dim3 block(GEMV_INT8_KPAR_TILE_N, GEMV_INT8_KPAR_SPLIT, 1);
        hipLaunchKernelGGL(
            gemv_int8_int8_kpar_kernel,
            dim3(grid_n), block, 0, 0,
            d_A_int8, d_B_int8, d_C_int32, N, K);
    }
    else if (use_wide2)
    {
        const int grid_n = (N + GEMV_WIDE2_TILE_N - 1) / GEMV_WIDE2_TILE_N;
        dim3 block(GEMV_WIDE2_TILE_N / 2, 1, 1);
        hipLaunchKernelGGL(
            gemv_int8_int8_wide2_kernel,
            dim3(grid_n), block, 0, 0,
            d_A_int8, d_B_int8, d_C_int32, N, K);
    }
    else if (use_wide_vec4 && (N >= GEMV_WIDE_TILE_N))
    {
        const int grid_n = (N + GEMV_WIDE_TILE_N - 1) / GEMV_WIDE_TILE_N;
        dim3 block(GEMV_WIDE_TILE_N / 4, 1, 1);
        hipLaunchKernelGGL(
            gemv_int8_int8_wide_vec4_kernel,
            dim3(grid_n), block, 0, 0,
            d_A_int8, d_B_int8, d_C_int32, N, K);
    }
    else if (N >= GEMV_SQUARE_TILE_N)
    {
        const int grid_n = (N + GEMV_SQUARE_TILE_N - 1) / GEMV_SQUARE_TILE_N;
        dim3 block(GEMV_SQUARE_TILE_N, 1, 1);
        hipLaunchKernelGGL(
            gemv_int8_int8_square_kernel,
            dim3(grid_n), block, 0, 0,
            d_A_int8, d_B_int8, d_C_int32, N, K);
    }
    else
    {
        const int grid_n = (N + GEMV_TILE_N - 1) / GEMV_TILE_N;
        hipLaunchKernelGGL(
            gemv_int8_int8_kernel,
            dim3(grid_n), dim3(GEMV_TILE_N), 0, 0,
            d_A_int8, d_B_int8, d_C_int32, N, K);
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "[rocmGemv_int8_int8_int32] Kernel launch failed: %s (N=%d K=%d)\n",
                hipGetErrorString(err), N, K);
        return false;
    }

    return true;
}

/**
 * @brief Execute INT8×INT8 GEMV with VNNI-packed weights: output_int32[N] = A_int8[K] × B_vnni
 */
bool rocmGemv_int8_int8_int32_vnni(
    const int8_t* d_A_int8,
    const int8_t* d_B_int8_vnni,
    int32_t* d_C_int32,
    int N, int K,
    int device_id)
{
    hipSetDevice(device_id);

    if ((K % 4) != 0)
    {
        fprintf(stderr, "[rocmGemv_int8_int8_int32_vnni] K must be multiple of 4 (N=%d K=%d)\n", N, K);
        return false;
    }

    const bool use_wide = (N >= 3 * K);
    const bool use_vec4 = ((N % 4) == 0);
    const bool use_vec2 = ((N % 2) == 0);
    const bool use_wide_vec4 = use_wide && (N >= 8 * K) && use_vec4;
    const bool use_wide2 = use_wide && (N < 8 * K) && use_vec2 && (N >= GEMV_INT8_VNNI_WIDE2_TILE_N);
    // Grid-level K-parallel: replaces old ffn_down, kpar, and square paths for non-wide shapes
    // Uses blockIdx.y for K-splitting + atomicAdd → 1000+ wavefronts vs 56-224 old
    const bool use_grid_kpar = !use_wide && use_vec4 && (N >= GEMV_INT8_VNNI_GRID_KPAR_TILE_N);
    const bool use_square = (N >= GEMV_INT8_VNNI_SQUARE_TILE_N);

    if (use_wide2)
    {
        const int grid_n = (N + GEMV_INT8_VNNI_WIDE2_TILE_N - 1) / GEMV_INT8_VNNI_WIDE2_TILE_N;
        dim3 block(GEMV_INT8_VNNI_WIDE2_TILE_N / 2, 1, 1);
        hipLaunchKernelGGL(
            gemv_int8_int8_wide2_vnni_kernel,
            dim3(grid_n), block, 0, 0,
            d_A_int8, d_B_int8_vnni, d_C_int32, N, K);
    }
    else if (use_wide_vec4 && (N >= GEMV_INT8_VNNI_WIDE_TILE_N))
    {
        const int grid_n = (N + GEMV_INT8_VNNI_WIDE_TILE_N - 1) / GEMV_INT8_VNNI_WIDE_TILE_N;
        dim3 block(GEMV_INT8_VNNI_WIDE_TILE_N / 4, 1, 1);
        hipLaunchKernelGGL(
            gemv_int8_int8_wide_vec4_vnni_kernel,
            dim3(grid_n), block, 0, 0,
            d_A_int8, d_B_int8_vnni, d_C_int32, N, K);
    }
    else if (use_grid_kpar)
    {
        // Grid-level K-parallel with atomicAdd: KB independent blocks per output tile
        // Each block processes K/KB fraction; output must be zeroed before launch
        // KB heuristic: target ~1500 wavefronts to saturate MI50 (60 CUs × ~40 waves/CU)
        constexpr int TN = GEMV_INT8_VNNI_GRID_KPAR_TILE_N;
        constexpr int CPT = GEMV_INT8_VNNI_GRID_KPAR_CPT;
        const int grid_n = (N + TN - 1) / TN;
        const int kb = max(2, min(64, (K + 63) / 64));

        hipMemsetAsync(d_C_int32, 0, N * sizeof(int32_t), 0);
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(gemv_int8_int8_grid_kpar_vnni_kernel_t<TN, CPT>),
            dim3(grid_n, kb), dim3(TN / CPT), 0, 0,
            d_A_int8, d_B_int8_vnni, d_C_int32, N, K, kb);
    }
    else if (use_square)
    {
        const int grid_n = (N + GEMV_INT8_VNNI_SQUARE_TILE_N - 1) / GEMV_INT8_VNNI_SQUARE_TILE_N;
        dim3 block(GEMV_INT8_VNNI_SQUARE_TILE_N, 1, 1);
        hipLaunchKernelGGL(
            gemv_int8_int8_square_vnni_kernel,
            dim3(grid_n), block, 0, 0,
            d_A_int8, d_B_int8_vnni, d_C_int32, N, K);
    }
    else
    {
        const int grid_n = (N + GEMV_TILE_N - 1) / GEMV_TILE_N;
        hipLaunchKernelGGL(
            gemv_int8_int8_vnni_kernel,
            dim3(grid_n), dim3(GEMV_TILE_N), 0, 0,
            d_A_int8, d_B_int8_vnni, d_C_int32, N, K);
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "[rocmGemv_int8_int8_int32_vnni] Kernel launch failed: %s (N=%d K=%d)\n",
                hipGetErrorString(err), N, K);
        return false;
    }

    return true;
}

/**
 * @brief Execute INT8×INT8 GEMV with tiled KN VNNI-packed weights
 *
 * Uses B layout [K/TK][N/TN][(TK/4)][TN][4] for coalesced access within tiles.
 * Sequential stride within tile is TN*4=512 bytes vs 76KB for flat VNNI.
 *
 * Requirements:
 *   - N must be divisible by GEMV_INT8_TILED_VNNI_TN (128)
 *   - K must be divisible by GEMV_INT8_TILED_VNNI_TK (128)
 *
 * Best for wide shapes (N >> K) like Gate/Up projections.
 */
bool rocmGemv_int8_int8_int32_tiled_vnni(
    const int8_t* d_A_int8,
    const int8_t* d_B_int8_tiled_vnni,
    int32_t* d_C_int32,
    int N, int K,
    int device_id)
{
    hipSetDevice(device_id);

    constexpr int TN = GEMV_INT8_TILED_VNNI_TN;
    constexpr int TK = GEMV_INT8_TILED_VNNI_TK;

    if ((N % TN) != 0 || (K % TK) != 0)
    {
        fprintf(stderr, "[rocmGemv_int8_int8_int32_tiled_vnni] N must be div by %d, K by %d (N=%d K=%d)\n",
                TN, TK, N, K);
        return false;
    }

    const int grid_n = N / TN;
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(gemv_int8_int8_tiled_vnni_kernel_t<TN, TK>),
        dim3(grid_n), dim3(TN), 0, 0,
        d_A_int8, d_B_int8_tiled_vnni, d_C_int32, N, K);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "[rocmGemv_int8_int8_int32_tiled_vnni] Kernel launch failed: %s (N=%d K=%d)\n",
                hipGetErrorString(err), N, K);
        return false;
    }

    return true;
}

// =============================================================================
// VNNI → Row-Major Repack Kernel
// =============================================================================
//
// Repacks weights from flat VNNI layout [K/4][N][4] to row-major [N×K]
// (equivalently [K×N] column-major for CK).
//
// VNNI layout:  vnni[(kg*N + n)*4 + i]  where kg = k/4, i = k%4
// Row-major:    rowmajor[n*K + k]       where k = kg*4 + i
//
// Mapping:  rowmajor[n*K + kg*4 + i] = vnni[(kg*N + n)*4 + i]
//
// Each thread handles one (n, kg) pair = 4 elements, packed as int32_t.
// Grid: ceil(N * K/4 / blockDim.x) blocks, blockDim.x = 256 threads.
//
// Access pattern:
//   - VNNI reads:  consecutive tids (consecutive n) → stride 4 → coalesced
//   - Row-major writes: consecutive tids (consecutive n) → stride K → scattered
//     but each write is a 4-byte int32_t, so individual writes are efficient.
//
// Cost: ~65-200μs for 18944×3584 (Gate/Up shape). Amortized over prefill.
//
__global__ void
repack_vnni_to_rowmajor_kernel(
    const int8_t* __restrict__ d_B_vnni,     // [K/4][N][4] input
    int8_t* __restrict__ d_B_rowmajor,       // [N×K] output
    int N, int K)
{
    const int k_groups = K / 4;
    const int total = N * k_groups;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= total)
        return;

    const int n  = tid % N;
    const int kg = tid / N;

    // Read 4 bytes from VNNI as a single int32_t (coalesced for consecutive n)
    const int32_t packed = *reinterpret_cast<const int32_t*>(
        &d_B_vnni[(static_cast<int64_t>(kg) * N + n) * 4]);

    // Write 4 bytes to row-major (scattered for consecutive n, but 4B atomic)
    *reinterpret_cast<int32_t*>(
        &d_B_rowmajor[static_cast<int64_t>(n) * K + kg * 4]) = packed;
}


bool rocmGemv_repackVNNI_to_rowmajor(
    const int8_t* d_B_vnni,
    int8_t* d_B_rowmajor,
    int N, int K,
    int device_id)
{
    if ((K % 4) != 0)
    {
        fprintf(stderr, "[rocmGemv_repackVNNI_to_rowmajor] K must be multiple of 4 (K=%d)\n", K);
        return false;
    }

    hipSetDevice(device_id);

    const int k_groups = K / 4;
    const int total = N * k_groups;
    constexpr int BLOCK = 256;
    const int grid = (total + BLOCK - 1) / BLOCK;

    hipLaunchKernelGGL(
        repack_vnni_to_rowmajor_kernel,
        dim3(grid), dim3(BLOCK), 0, 0,
        d_B_vnni, d_B_rowmajor, N, K);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "[rocmGemv_repackVNNI_to_rowmajor] Kernel launch failed: %s (N=%d K=%d)\n",
                hipGetErrorString(err), N, K);
        return false;
    }

    return true;
}

} // extern "C"
