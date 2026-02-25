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
#include <cstdlib>
#include <atomic>

#include "../../backends/rocm/HipDeviceGuard.h"

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

// VNNI kernel tile parameters
constexpr int GEMV_INT8_VNNI_WIDE_TILE_N = 128;
constexpr int GEMV_INT8_VNNI_WIDE_DEFAULT_CPT = 2;
constexpr int GEMV_INT8_VNNI_WIDE2_TILE_N = 256;
constexpr int GEMV_INT8_VNNI_SQUARE_TILE_N = 256;
constexpr int GEMV_INT8_VNNI_KPAR_TILE_K = 256;
constexpr int GEMV_INT8_VNNI_GRID_KPAR_TILE_N = 128;
constexpr int GEMV_INT8_VNNI_GRID_KPAR_CPT = 2;  // 64 threads = 1 full wavefront (was 4 = 32 threads = half wavefront)

constexpr uint8_t CODEBOOK_IQ4 = 4;

static std::atomic<int> g_int8_vnni_tn_override{-1};
static std::atomic<int> g_int8_vnni_kb_override{-1};
static std::atomic<int> g_int8_vnni_wide_tn_override{-1};
static std::atomic<int> g_int8_vnni_wide_cpt_override{-1};
static std::atomic<int> g_int8_vnni_wide_vec_load_override{-1};

__device__ __constant__ int8_t k_iq4_codebook_i8[16] = {
    -127, -104, -83, -65,
    -49, -35, -22, -10,
    1, 13, 25, 38,
    53, 69, 89, 113};

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
        // Load 4 INT8 activations as packed int32 for v_dot4_i32_i8
        const int32_t a_packed = *reinterpret_cast<const int32_t*>(d_A_int8 + kg * 4);

        const int32_t b_packed = *reinterpret_cast<const int32_t*>(
            d_B_int8_vnni + (static_cast<size_t>(kg) * N + n) * 4);

        // Hardware INT8 dot product: 4 multiply-adds in 1 cycle (v_dot4_i32_i8)
        acc = __builtin_amdgcn_sdot4(a_packed, b_packed, acc, false);
    }

    d_C_int32[n] = acc;
}

/**
 * @brief 2D INT8 VNNI prefill kernel (M>1): computes C[M,N] in one launch.
 *
 * Thread mapping:
 * - threadIdx.y / blockIdx.y choose output row `m`
 * - threadIdx.x / blockIdx.x choose output column `n`
 *
 * For each output element C[m,n], we iterate over K in groups of 4 bytes and
 * use `v_dot4_i32_i8` (`__builtin_amdgcn_sdot4`) for efficient INT8 MAC.
 *
 * Layout reminder for junior developers:
 * - A is row-major [M x K]
 * - B is VNNI-packed [K/4 x N x 4]
 * - C is row-major [M x N]
 */
template <int BLOCK_X, int BLOCK_Y, int CPT>
__global__ void gemm_int8_int8_vnni_prefill_kernel_t(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B_int8_vnni,
    int32_t* __restrict__ d_C_int32,
    int M,
    int N,
    int K)
{
    const int n_base = blockIdx.x * (BLOCK_X * CPT) + threadIdx.x * CPT;
    const int m = blockIdx.y * BLOCK_Y + threadIdx.y;
    if (m >= M || n_base >= N)
    {
        return;
    }

    int32_t acc[CPT];
    #pragma unroll
    for (int c = 0; c < CPT; ++c)
    {
        acc[c] = 0;
    }

    const int k_groups = K / 4;

    // Compute one output element by reducing across K.
    for (int kg = 0; kg < k_groups; ++kg)
    {
        const int32_t a_packed = *reinterpret_cast<const int32_t*>(
            d_A_int8 + static_cast<int64_t>(m) * K + kg * 4);

        if constexpr (CPT == 4)
        {
            if ((n_base + 3) < N)
            {
                const uint4 b_pack4 = *reinterpret_cast<const uint4*>(
                    d_B_int8_vnni + (static_cast<int64_t>(kg) * N + n_base) * 4);
                acc[0] = __builtin_amdgcn_sdot4(a_packed, static_cast<int32_t>(b_pack4.x), acc[0], false);
                acc[1] = __builtin_amdgcn_sdot4(a_packed, static_cast<int32_t>(b_pack4.y), acc[1], false);
                acc[2] = __builtin_amdgcn_sdot4(a_packed, static_cast<int32_t>(b_pack4.z), acc[2], false);
                acc[3] = __builtin_amdgcn_sdot4(a_packed, static_cast<int32_t>(b_pack4.w), acc[3], false);
            }
            else
            {
                #pragma unroll
                for (int c = 0; c < CPT; ++c)
                {
                    const int n = n_base + c;
                    if (n < N)
                    {
                        const int32_t b_packed = *reinterpret_cast<const int32_t*>(
                            d_B_int8_vnni + (static_cast<int64_t>(kg) * N + n) * 4);
                        acc[c] = __builtin_amdgcn_sdot4(a_packed, b_packed, acc[c], false);
                    }
                }
            }
        }
        else if constexpr (CPT == 2)
        {
            if ((n_base + 1) < N)
            {
                const uint2 b_pack2 = *reinterpret_cast<const uint2*>(
                    d_B_int8_vnni + (static_cast<int64_t>(kg) * N + n_base) * 4);
                acc[0] = __builtin_amdgcn_sdot4(a_packed, static_cast<int32_t>(b_pack2.x), acc[0], false);
                acc[1] = __builtin_amdgcn_sdot4(a_packed, static_cast<int32_t>(b_pack2.y), acc[1], false);
            }
            else
            {
                #pragma unroll
                for (int c = 0; c < CPT; ++c)
                {
                    const int n = n_base + c;
                    if (n < N)
                    {
                        const int32_t b_packed = *reinterpret_cast<const int32_t*>(
                            d_B_int8_vnni + (static_cast<int64_t>(kg) * N + n) * 4);
                        acc[c] = __builtin_amdgcn_sdot4(a_packed, b_packed, acc[c], false);
                    }
                }
            }
        }
        else
        {
            const int32_t b_packed = *reinterpret_cast<const int32_t*>(
                d_B_int8_vnni + (static_cast<int64_t>(kg) * N + n_base) * 4);
            acc[0] = __builtin_amdgcn_sdot4(a_packed, b_packed, acc[0], false);
        }
    }

    if constexpr (CPT == 4)
    {
        if ((n_base + 3) < N)
        {
            *reinterpret_cast<uint4*>(d_C_int32 + static_cast<int64_t>(m) * N + n_base) =
                make_uint4(static_cast<uint32_t>(acc[0]), static_cast<uint32_t>(acc[1]), static_cast<uint32_t>(acc[2]), static_cast<uint32_t>(acc[3]));
        }
        else
        {
            #pragma unroll
            for (int c = 0; c < CPT; ++c)
            {
                const int n = n_base + c;
                if (n < N)
                {
                    d_C_int32[static_cast<int64_t>(m) * N + n] = acc[c];
                }
            }
        }
    }
    else if constexpr (CPT == 2)
    {
        if ((n_base + 1) < N)
        {
            *reinterpret_cast<uint2*>(d_C_int32 + static_cast<int64_t>(m) * N + n_base) =
                make_uint2(static_cast<uint32_t>(acc[0]), static_cast<uint32_t>(acc[1]));
        }
        else
        {
            #pragma unroll
            for (int c = 0; c < CPT; ++c)
            {
                const int n = n_base + c;
                if (n < N)
                {
                    d_C_int32[static_cast<int64_t>(m) * N + n] = acc[c];
                }
            }
        }
    }
    else
    {
        d_C_int32[static_cast<int64_t>(m) * N + n_base] = acc[0];
    }
}

/**
 * @brief 2D INT8 VNNI prefill kernel with grid-kpar split-K accumulation.
 *
 * grid.z partitions K into slices. Each z-slice computes a partial sum for
 * C[m,n] and atomically accumulates into INT32 output.
 */
template <int BLOCK_X, int BLOCK_Y, int CPT>
__global__ void gemm_int8_int8_vnni_prefill_grid_kpar_kernel_t(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B_int8_vnni,
    int32_t* __restrict__ d_C_int32,
    int M,
    int N,
    int K,
    int k_groups_per_slice)
{
    const int n_base = blockIdx.x * (BLOCK_X * CPT) + threadIdx.x * CPT;
    const int m = blockIdx.y * BLOCK_Y + threadIdx.y;
    const int split_idx = blockIdx.z;

    if (m >= M || n_base >= N)
    {
        return;
    }

    const int total_k_groups = K / 4;
    const int k_begin = split_idx * k_groups_per_slice;
    if (k_begin >= total_k_groups)
    {
        return;
    }

    const int k_end = min(total_k_groups, k_begin + k_groups_per_slice);
    int32_t acc[CPT];
    #pragma unroll
    for (int c = 0; c < CPT; ++c)
    {
        acc[c] = 0;
    }

    for (int kg = k_begin; kg < k_end; ++kg)
    {
        const int32_t a_packed = *reinterpret_cast<const int32_t*>(
            d_A_int8 + static_cast<int64_t>(m) * K + kg * 4);

        if constexpr (CPT == 4)
        {
            if ((n_base + 3) < N)
            {
                const uint4 b_pack4 = *reinterpret_cast<const uint4*>(
                    d_B_int8_vnni + (static_cast<int64_t>(kg) * N + n_base) * 4);
                acc[0] = __builtin_amdgcn_sdot4(a_packed, static_cast<int32_t>(b_pack4.x), acc[0], false);
                acc[1] = __builtin_amdgcn_sdot4(a_packed, static_cast<int32_t>(b_pack4.y), acc[1], false);
                acc[2] = __builtin_amdgcn_sdot4(a_packed, static_cast<int32_t>(b_pack4.z), acc[2], false);
                acc[3] = __builtin_amdgcn_sdot4(a_packed, static_cast<int32_t>(b_pack4.w), acc[3], false);
            }
            else
            {
                #pragma unroll
                for (int c = 0; c < CPT; ++c)
                {
                    const int n = n_base + c;
                    if (n < N)
                    {
                        const int32_t b_packed = *reinterpret_cast<const int32_t*>(
                            d_B_int8_vnni + (static_cast<int64_t>(kg) * N + n) * 4);
                        acc[c] = __builtin_amdgcn_sdot4(a_packed, b_packed, acc[c], false);
                    }
                }
            }
        }
        else if constexpr (CPT == 2)
        {
            if ((n_base + 1) < N)
            {
                const uint2 b_pack2 = *reinterpret_cast<const uint2*>(
                    d_B_int8_vnni + (static_cast<int64_t>(kg) * N + n_base) * 4);
                acc[0] = __builtin_amdgcn_sdot4(a_packed, static_cast<int32_t>(b_pack2.x), acc[0], false);
                acc[1] = __builtin_amdgcn_sdot4(a_packed, static_cast<int32_t>(b_pack2.y), acc[1], false);
            }
            else
            {
                #pragma unroll
                for (int c = 0; c < CPT; ++c)
                {
                    const int n = n_base + c;
                    if (n < N)
                    {
                        const int32_t b_packed = *reinterpret_cast<const int32_t*>(
                            d_B_int8_vnni + (static_cast<int64_t>(kg) * N + n) * 4);
                        acc[c] = __builtin_amdgcn_sdot4(a_packed, b_packed, acc[c], false);
                    }
                }
            }
        }
        else
        {
            const int32_t b_packed = *reinterpret_cast<const int32_t*>(
                d_B_int8_vnni + (static_cast<int64_t>(kg) * N + n_base) * 4);
            acc[0] = __builtin_amdgcn_sdot4(a_packed, b_packed, acc[0], false);
        }
    }

    #pragma unroll
    for (int c = 0; c < CPT; ++c)
    {
        const int n = n_base + c;
        if (n < N && acc[c] != 0)
        {
            atomicAdd(&d_C_int32[static_cast<int64_t>(m) * N + n], acc[c]);
        }
    }
}


/**
 * @brief 2D payload-v2 bitwidth-8 prefill kernel (M>1): computes C[M,N].
 *
 * Layout:
 * - A is row-major INT8 [M x K]
 * - Payload is [K/32 x N x 32 bytes] where each record stores 32 signed INT8
 *   values for one (block, output-column)
 * - C is row-major INT32 [M x N]
 */
__global__ void gemm_payload_v2_i8_prefill_kernel(
    const int8_t* __restrict__ d_A_int8,
    const uint8_t* __restrict__ d_payload,
    int32_t* __restrict__ d_C_int32,
    int M,
    int N,
    int K,
    int kblocks)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y * blockDim.y + threadIdx.y;
    const bool valid_n = (n < N);
    const bool valid_m = (m < M);

    const int blocks_per_row = K / 32;
    const int kb = max(1, kblocks);
    const int b_per = (blocks_per_row + kb - 1) / kb;
    const int b0 = min(blocks_per_row, static_cast<int>(blockIdx.z) * b_per);
    const int b1 = min(blocks_per_row, b0 + b_per);

    if (b0 >= b1)
    {
        return;
    }

    int32_t acc = 0;
    extern __shared__ int32_t shared_payload[]; // [blockDim.x * 8]

    for (int b = b0; b < b1; ++b)
    {
        if (threadIdx.y == 0)
        {
            int32_t* payload_dst = shared_payload + static_cast<int>(threadIdx.x) * 8;
            if (valid_n)
            {
                const size_t linear = static_cast<size_t>(b) * static_cast<size_t>(N) + static_cast<size_t>(n);
                const uint4 packed0 = *reinterpret_cast<const uint4*>(d_payload + linear * 32u);
                const uint4 packed1 = *reinterpret_cast<const uint4*>(d_payload + linear * 32u + 16u);
                payload_dst[0] = static_cast<int32_t>(packed0.x);
                payload_dst[1] = static_cast<int32_t>(packed0.y);
                payload_dst[2] = static_cast<int32_t>(packed0.z);
                payload_dst[3] = static_cast<int32_t>(packed0.w);
                payload_dst[4] = static_cast<int32_t>(packed1.x);
                payload_dst[5] = static_cast<int32_t>(packed1.y);
                payload_dst[6] = static_cast<int32_t>(packed1.z);
                payload_dst[7] = static_cast<int32_t>(packed1.w);
            }
            else
            {
                #pragma unroll
                for (int g = 0; g < 8; ++g)
                {
                    payload_dst[g] = 0;
                }
            }
        }

        __syncthreads();

        if (valid_m && valid_n)
        {
            const int32_t* a4 = reinterpret_cast<const int32_t*>(
                d_A_int8 + static_cast<int64_t>(m) * K + b * 32);
            const int32_t* payload_i32 = shared_payload + static_cast<int>(threadIdx.x) * 8;

            #pragma unroll
            for (int g = 0; g < 8; ++g)
            {
                acc = __builtin_amdgcn_sdot4(a4[g], payload_i32[g], acc, false);
            }
        }

        __syncthreads();
    }

    if (valid_m && valid_n)
    {
        const int64_t out_idx = static_cast<int64_t>(m) * N + n;
        if (kb == 1)
        {
            d_C_int32[out_idx] = acc;
        }
        else
        {
            atomicAdd(&d_C_int32[out_idx], acc);
        }
    }
}

__global__ void gemm_payload_v2_i8_prefill_kernel_direct(
    const int8_t* __restrict__ d_A_int8,
    const uint8_t* __restrict__ d_payload,
    int32_t* __restrict__ d_C_int32,
    int M,
    int N,
    int K,
    int kblocks)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= M || n >= N)
    {
        return;
    }

    const int blocks_per_row = K / 32;
    const int kb = max(1, kblocks);
    const int b_per = (blocks_per_row + kb - 1) / kb;
    const int b0 = min(blocks_per_row, static_cast<int>(blockIdx.z) * b_per);
    const int b1 = min(blocks_per_row, b0 + b_per);

    if (b0 >= b1)
    {
        return;
    }

    int32_t acc = 0;

    for (int b = b0; b < b1; ++b)
    {
        const int32_t* a4 = reinterpret_cast<const int32_t*>(
            d_A_int8 + static_cast<int64_t>(m) * K + b * 32);

        const size_t linear = static_cast<size_t>(b) * static_cast<size_t>(N) + static_cast<size_t>(n);
        const uint4 packed0 = *reinterpret_cast<const uint4*>(d_payload + linear * 32u);
        const uint4 packed1 = *reinterpret_cast<const uint4*>(d_payload + linear * 32u + 16u);

        acc = __builtin_amdgcn_sdot4(a4[0], static_cast<int32_t>(packed0.x), acc, false);
        acc = __builtin_amdgcn_sdot4(a4[1], static_cast<int32_t>(packed0.y), acc, false);
        acc = __builtin_amdgcn_sdot4(a4[2], static_cast<int32_t>(packed0.z), acc, false);
        acc = __builtin_amdgcn_sdot4(a4[3], static_cast<int32_t>(packed0.w), acc, false);
        acc = __builtin_amdgcn_sdot4(a4[4], static_cast<int32_t>(packed1.x), acc, false);
        acc = __builtin_amdgcn_sdot4(a4[5], static_cast<int32_t>(packed1.y), acc, false);
        acc = __builtin_amdgcn_sdot4(a4[6], static_cast<int32_t>(packed1.z), acc, false);
        acc = __builtin_amdgcn_sdot4(a4[7], static_cast<int32_t>(packed1.w), acc, false);
    }

    const int64_t out_idx = static_cast<int64_t>(m) * N + n;
    if (kb == 1)
    {
        d_C_int32[out_idx] = acc;
    }
    else
    {
        atomicAdd(&d_C_int32[out_idx], acc);
    }
}

template <int BLOCK_X, int BLOCK_Y, int CPT>
__global__ void gemm_payload_v2_i8_prefill_kernel_t(
    const int8_t* __restrict__ d_A_int8,
    const uint8_t* __restrict__ d_payload,
    int32_t* __restrict__ d_C_int32,
    int M,
    int N,
    int K,
    int kblocks)
{
    const int n_base = blockIdx.x * (BLOCK_X * CPT) + threadIdx.x * CPT;
    const int m = blockIdx.y * BLOCK_Y + threadIdx.y;
    if (m >= M || n_base >= N)
    {
        return;
    }

    const int blocks_per_row = K / 32;
    const int kb = max(1, kblocks);
    const int b_per = (blocks_per_row + kb - 1) / kb;
    const int b0 = min(blocks_per_row, static_cast<int>(blockIdx.z) * b_per);
    const int b1 = min(blocks_per_row, b0 + b_per);

    if (b0 >= b1)
    {
        return;
    }

    int32_t acc[CPT];
    #pragma unroll
    for (int c = 0; c < CPT; ++c)
    {
        acc[c] = 0;
    }

    for (int b = b0; b < b1; ++b)
    {
        const int32_t* a4 = reinterpret_cast<const int32_t*>(
            d_A_int8 + static_cast<int64_t>(m) * K + b * 32);

        const size_t linear0 = static_cast<size_t>(b) * static_cast<size_t>(N) + static_cast<size_t>(n_base);
        const int32_t* payload_i32_0 = reinterpret_cast<const int32_t*>(d_payload + linear0 * 32u);

        if constexpr (CPT == 4)
        {
            if ((n_base + 3) < N)
            {
                const int32_t* payload_i32_1 = payload_i32_0 + 8;
                const int32_t* payload_i32_2 = payload_i32_1 + 8;
                const int32_t* payload_i32_3 = payload_i32_2 + 8;

                #pragma unroll
                for (int g = 0; g < 8; ++g)
                {
                    const int32_t a = a4[g];
                    acc[0] = __builtin_amdgcn_sdot4(a, payload_i32_0[g], acc[0], false);
                    acc[1] = __builtin_amdgcn_sdot4(a, payload_i32_1[g], acc[1], false);
                    acc[2] = __builtin_amdgcn_sdot4(a, payload_i32_2[g], acc[2], false);
                    acc[3] = __builtin_amdgcn_sdot4(a, payload_i32_3[g], acc[3], false);
                }
            }
            else
            {
                #pragma unroll
                for (int c = 0; c < CPT; ++c)
                {
                    const int n = n_base + c;
                    if (n < N)
                    {
                        const size_t linear = static_cast<size_t>(b) * static_cast<size_t>(N) + static_cast<size_t>(n);
                        const int32_t* payload_i32 = reinterpret_cast<const int32_t*>(d_payload + linear * 32u);
                        #pragma unroll
                        for (int g = 0; g < 8; ++g)
                        {
                            acc[c] = __builtin_amdgcn_sdot4(a4[g], payload_i32[g], acc[c], false);
                        }
                    }
                }
            }
        }
        else if constexpr (CPT == 2)
        {
            if ((n_base + 1) < N)
            {
                const int32_t* payload_i32_1 = payload_i32_0 + 8;
                #pragma unroll
                for (int g = 0; g < 8; ++g)
                {
                    const int32_t a = a4[g];
                    acc[0] = __builtin_amdgcn_sdot4(a, payload_i32_0[g], acc[0], false);
                    acc[1] = __builtin_amdgcn_sdot4(a, payload_i32_1[g], acc[1], false);
                }
            }
            else
            {
                #pragma unroll
                for (int c = 0; c < CPT; ++c)
                {
                    const int n = n_base + c;
                    if (n < N)
                    {
                        const size_t linear = static_cast<size_t>(b) * static_cast<size_t>(N) + static_cast<size_t>(n);
                        const int32_t* payload_i32 = reinterpret_cast<const int32_t*>(d_payload + linear * 32u);
                        #pragma unroll
                        for (int g = 0; g < 8; ++g)
                        {
                            acc[c] = __builtin_amdgcn_sdot4(a4[g], payload_i32[g], acc[c], false);
                        }
                    }
                }
            }
        }
        else
        {
            #pragma unroll
            for (int g = 0; g < 8; ++g)
            {
                acc[0] = __builtin_amdgcn_sdot4(a4[g], payload_i32_0[g], acc[0], false);
            }
        }
    }

    const int64_t out_base = static_cast<int64_t>(m) * N + n_base;
    if (kb == 1)
    {
        if constexpr (CPT == 4)
        {
            if ((n_base + 3) < N)
            {
                *reinterpret_cast<uint4*>(d_C_int32 + out_base) = make_uint4(
                    static_cast<uint32_t>(acc[0]),
                    static_cast<uint32_t>(acc[1]),
                    static_cast<uint32_t>(acc[2]),
                    static_cast<uint32_t>(acc[3]));
            }
            else
            {
                #pragma unroll
                for (int c = 0; c < CPT; ++c)
                {
                    const int n = n_base + c;
                    if (n < N)
                    {
                        d_C_int32[out_base + c] = acc[c];
                    }
                }
            }
        }
        else if constexpr (CPT == 2)
        {
            if ((n_base + 1) < N)
            {
                *reinterpret_cast<uint2*>(d_C_int32 + out_base) = make_uint2(
                    static_cast<uint32_t>(acc[0]),
                    static_cast<uint32_t>(acc[1]));
            }
            else
            {
                d_C_int32[out_base] = acc[0];
                if ((n_base + 1) < N)
                {
                    d_C_int32[out_base + 1] = acc[1];
                }
            }
        }
        else
        {
            d_C_int32[out_base] = acc[0];
        }
    }
    else
    {
        #pragma unroll
        for (int c = 0; c < CPT; ++c)
        {
            const int n = n_base + c;
            if (n < N)
            {
                atomicAdd(&d_C_int32[out_base + c], acc[c]);
            }
        }
    }
}

template <int TILE_N, int CPT, bool VEC_PACKED_LOAD>
__global__ void gemv_int8_int8_wide_vnni_kernel_t(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B_int8_vnni,
    int32_t* __restrict__ d_C_int32,
    int N,
    int K)
{
    const int n_base = blockIdx.x * TILE_N + threadIdx.x * CPT;
    if (n_base >= N) return;

    int32_t acc[CPT];
    #pragma unroll
    for (int c = 0; c < CPT; ++c) acc[c] = 0;

    const bool full_cols = (n_base + (CPT - 1) < N);
    const int k_groups = K / 4;
    for (int kg = 0; kg < k_groups; ++kg)
    {
        // Load 4 INT8 activations as packed int32 for v_dot4_i32_i8
        const int32_t a_packed = *reinterpret_cast<const int32_t*>(d_A_int8 + kg * 4);

        const int32_t* __restrict__ b_row = reinterpret_cast<const int32_t*>(
            d_B_int8_vnni + (static_cast<size_t>(kg) * N + n_base) * 4);

        if constexpr (VEC_PACKED_LOAD && CPT == 4)
        {
            if (full_cols)
            {
                const int4 b = *reinterpret_cast<const int4*>(b_row);
                acc[0] = __builtin_amdgcn_sdot4(a_packed, b.x, acc[0], false);
                acc[1] = __builtin_amdgcn_sdot4(a_packed, b.y, acc[1], false);
                acc[2] = __builtin_amdgcn_sdot4(a_packed, b.z, acc[2], false);
                acc[3] = __builtin_amdgcn_sdot4(a_packed, b.w, acc[3], false);
            }
            else
            {
                #pragma unroll
                for (int c = 0; c < CPT; ++c)
                {
                    if (n_base + c < N)
                    {
                        const int32_t b = *(b_row + c);
                        acc[c] = __builtin_amdgcn_sdot4(a_packed, b, acc[c], false);
                    }
                }
            }
        }
        else if constexpr (VEC_PACKED_LOAD && CPT == 2)
        {
            if (full_cols)
            {
                const int2 b = *reinterpret_cast<const int2*>(b_row);
                acc[0] = __builtin_amdgcn_sdot4(a_packed, b.x, acc[0], false);
                acc[1] = __builtin_amdgcn_sdot4(a_packed, b.y, acc[1], false);
            }
            else
            {
                #pragma unroll
                for (int c = 0; c < CPT; ++c)
                {
                    if (n_base + c < N)
                    {
                        const int32_t b = *(b_row + c);
                        acc[c] = __builtin_amdgcn_sdot4(a_packed, b, acc[c], false);
                    }
                }
            }
        }
        else
        {
            #pragma unroll
            for (int c = 0; c < CPT; ++c)
            {
                if (n_base + c < N)
                {
                    const int32_t b = *(b_row + c);
                    acc[c] = __builtin_amdgcn_sdot4(a_packed, b, acc[c], false);
                }
            }
        }
    }

    #pragma unroll
    for (int c = 0; c < CPT; ++c)
    {
        if (n_base + c < N)
        {
            d_C_int32[n_base + c] = acc[c];
        }
    }
}

// Wide-N kernel: 2 columns per thread
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
        // Load 4 INT8 activations as packed int32 for v_dot4_i32_i8
        const int32_t a_packed = *reinterpret_cast<const int32_t*>(d_A_int8 + kg * 4);

        // Load 4 INT8 weights for each of 2 columns (VNNI = already packed)
        const int32_t p0 = *reinterpret_cast<const int32_t*>(
            d_B_int8_vnni + (static_cast<size_t>(kg) * N + (n2 + 0)) * 4);
        const int32_t p1 = *reinterpret_cast<const int32_t*>(
            d_B_int8_vnni + (static_cast<size_t>(kg) * N + (n2 + 1)) * 4);

        // Hardware INT8 dot product: 4 multiply-adds in 1 cycle (v_dot4_i32_i8)
        acc0 = __builtin_amdgcn_sdot4(a_packed, p0, acc0, false);
        acc1 = __builtin_amdgcn_sdot4(a_packed, p1, acc1, false);
    }

    d_C_int32[n2 + 0] = acc0;
    d_C_int32[n2 + 1] = acc1;
}

// K-parallel kernel: scalar threads, shared A tiles
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
            // Load 4 INT8 activations from shared memory as packed int32
            const int32_t a_packed = *reinterpret_cast<const int32_t*>(sA + kg * 4);

            const int global_kg = k_group_base + kg;
            const int32_t b_packed = *reinterpret_cast<const int32_t*>(
                d_B_int8_vnni + (static_cast<size_t>(global_kg) * N + n) * 4);

            // Hardware INT8 dot product: 4 multiply-adds in 1 cycle (v_dot4_i32_i8)
            acc = __builtin_amdgcn_sdot4(a_packed, b_packed, acc, false);
        }
        __syncthreads();
    }

    d_C_int32[n] = acc;
}

__global__ void gemv_int8_int8_vnni_scaled_kernel(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B_int8_vnni,
    float* __restrict__ d_C_fp32,
    const float* __restrict__ d_scale_A,
    const float* __restrict__ d_scale_B,
    int N,
    int K,
    float alpha,
    float beta,
    const float* __restrict__ d_C_existing,
    const float* __restrict__ d_bias)
{
    const int n = blockIdx.x * GEMV_TILE_N + threadIdx.x;
    if (n >= N) return;

    int32_t acc = 0;
    const int k_groups = K / 4;
    for (int kg = 0; kg < k_groups; ++kg)
    {
        const int32_t a_packed = *reinterpret_cast<const int32_t*>(d_A_int8 + kg * 4);
        const int32_t b_packed = *reinterpret_cast<const int32_t*>(
            d_B_int8_vnni + (static_cast<size_t>(kg) * N + n) * 4);
        acc = __builtin_amdgcn_sdot4(a_packed, b_packed, acc, false);
    }

    float result = static_cast<float>(acc) * d_scale_A[0] * d_scale_B[n];
    result *= alpha;
    if (beta != 0.0f)
    {
        result += beta * d_C_existing[n];
    }
    if (d_bias)
    {
        result += d_bias[n];
    }
    d_C_fp32[n] = result;
}

template <int TILE_N, int CPT, bool VEC_PACKED_LOAD>
__global__ void gemv_int8_int8_wide_vnni_scaled_kernel_t(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B_int8_vnni,
    float* __restrict__ d_C_fp32,
    const float* __restrict__ d_scale_A,
    const float* __restrict__ d_scale_B,
    int N,
    int K,
    float alpha,
    float beta,
    const float* __restrict__ d_C_existing,
    const float* __restrict__ d_bias)
{
    const int n_base = blockIdx.x * TILE_N + threadIdx.x * CPT;
    if (n_base >= N) return;

    int32_t acc[CPT];
    #pragma unroll
    for (int c = 0; c < CPT; ++c) acc[c] = 0;

    const bool full_cols = (n_base + (CPT - 1) < N);
    const int k_groups = K / 4;
    for (int kg = 0; kg < k_groups; ++kg)
    {
        const int32_t a_packed = *reinterpret_cast<const int32_t*>(d_A_int8 + kg * 4);

        const int32_t* __restrict__ b_row = reinterpret_cast<const int32_t*>(
            d_B_int8_vnni + (static_cast<size_t>(kg) * N + n_base) * 4);

        if constexpr (VEC_PACKED_LOAD && CPT == 4)
        {
            if (full_cols)
            {
                const int4 b = *reinterpret_cast<const int4*>(b_row);
                acc[0] = __builtin_amdgcn_sdot4(a_packed, b.x, acc[0], false);
                acc[1] = __builtin_amdgcn_sdot4(a_packed, b.y, acc[1], false);
                acc[2] = __builtin_amdgcn_sdot4(a_packed, b.z, acc[2], false);
                acc[3] = __builtin_amdgcn_sdot4(a_packed, b.w, acc[3], false);
            }
            else
            {
                #pragma unroll
                for (int c = 0; c < CPT; ++c)
                {
                    if (n_base + c < N)
                    {
                        const int32_t b = *(b_row + c);
                        acc[c] = __builtin_amdgcn_sdot4(a_packed, b, acc[c], false);
                    }
                }
            }
        }
        else if constexpr (VEC_PACKED_LOAD && CPT == 2)
        {
            if (full_cols)
            {
                const int2 b = *reinterpret_cast<const int2*>(b_row);
                acc[0] = __builtin_amdgcn_sdot4(a_packed, b.x, acc[0], false);
                acc[1] = __builtin_amdgcn_sdot4(a_packed, b.y, acc[1], false);
            }
            else
            {
                #pragma unroll
                for (int c = 0; c < CPT; ++c)
                {
                    if (n_base + c < N)
                    {
                        const int32_t b = *(b_row + c);
                        acc[c] = __builtin_amdgcn_sdot4(a_packed, b, acc[c], false);
                    }
                }
            }
        }
        else
        {
            #pragma unroll
            for (int c = 0; c < CPT; ++c)
            {
                if (n_base + c < N)
                {
                    const int32_t b = *(b_row + c);
                    acc[c] = __builtin_amdgcn_sdot4(a_packed, b, acc[c], false);
                }
            }
        }
    }

    const float scale_a = d_scale_A[0];
    #pragma unroll
    for (int c = 0; c < CPT; ++c)
    {
        const int n = n_base + c;
        if (n < N)
        {
            float out = static_cast<float>(acc[c]) * scale_a * d_scale_B[n];
            out *= alpha;
            if (beta != 0.0f)
            {
                out += beta * d_C_existing[n];
            }
            if (d_bias)
            {
                out += d_bias[n];
            }
            d_C_fp32[n] = out;
        }
    }
}

__global__ void gemv_int8_int8_square_vnni_scaled_kernel(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B_int8_vnni,
    float* __restrict__ d_C_fp32,
    const float* __restrict__ d_scale_A,
    const float* __restrict__ d_scale_B,
    int N,
    int K,
    float alpha,
    float beta,
    const float* __restrict__ d_C_existing,
    const float* __restrict__ d_bias)
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
            const int32_t a_packed = *reinterpret_cast<const int32_t*>(sA + kg * 4);
            const int global_kg = k_group_base + kg;
            const int32_t b_packed = *reinterpret_cast<const int32_t*>(
                d_B_int8_vnni + (static_cast<size_t>(global_kg) * N + n) * 4);
            acc = __builtin_amdgcn_sdot4(a_packed, b_packed, acc, false);
        }
        __syncthreads();
    }

    float result = static_cast<float>(acc) * d_scale_A[0] * d_scale_B[n];
    result *= alpha;
    if (beta != 0.0f)
    {
        result += beta * d_C_existing[n];
    }
    if (d_bias)
    {
        result += d_bias[n];
    }
    d_C_fp32[n] = result;
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
    // CRITICAL: k_per must be a multiple of 4 to ensure k_start aligns with
    // VNNI 4-byte groups. If k_start is not a multiple of 4, kg_offset would
    // misalign with the weight K-groups, producing garbage output.
    const int k_per_raw = (K + kblocks - 1) / kblocks;
    const int k_per = ((k_per_raw + 3) / 4) * 4;  // Round up to multiple of 4
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

        // Precompute B pointer: b_row[kg] points to the start of this thread's
        // columns for k-group kg. b_row = B_int32_ptr + n_base, stride = N (int32 units).
        const int32_t* __restrict__ b_row =
            reinterpret_cast<const int32_t*>(d_B_int8_vnni) + n_base;
        const size_t b_stride = static_cast<size_t>(N);

        // Partial unroll: expose 4 k-groups per iteration for better
        // memory-level parallelism and instruction scheduling
        #pragma unroll 4
        for (int kg = kg_start; kg < kg_end; ++kg)
        {
            // Load 4 INT8 activations as packed int32 for v_dot4_i32_i8
            const int32_t a_packed = *reinterpret_cast<const int32_t*>(d_A_int8 + kg * 4);

            if constexpr (CPT == 2)
            {
                // Vectorized 8-byte load: two adjacent VNNI columns in one
                // memory transaction. n_base is always even (threadIdx.x * 2),
                // so int2 alignment is guaranteed for N divisible by 2.
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

        // Accumulate partials to output via atomicAdd
        #pragma unroll
        for (int c = 0; c < CPT; ++c)
        {
            atomicAdd(&d_C_int32[n_base + c], acc[c]);
        }
    }
}


// ============================================================================
// LDS K-Reduction GEMV kernel: eliminates atomicAdd via intra-block reduction
//
// KEY INSIGHT: Instead of using 1-wavefront blocks with grid-level atomicAdd,
// use WK-wavefront blocks where each wavefront computes a K-slice and the
// partial sums are reduced via LDS within the block.
//
// Advantages over grid_kpar:
//   1. Eliminates atomicAdd when kb_outer==1 (large-N shapes like FFN)
//   2. Eliminates hipMemsetAsync when kb_outer==1
//   3. Reduces atomicAdd count by WK× when kb_outer>1 (small-N shapes)
//   4. Same total wavefront count → same occupancy and memory bandwidth
//
// Thread layout:
//   Block: WK × (TILE_N / CPT) threads = 4 × 64 = 256
//   wave_id = threadIdx.x / (TILE_N/CPT) → K-slice index (0..WK-1)
//   lane_id = threadIdx.x % (TILE_N/CPT) → column index within tile
//   Each wave handles TILE_N output columns across K/(WK*kb_outer) k-groups
//
// LDS usage: (WK-1) × TILE_N × sizeof(int32_t)
//   WK=4, TILE_N=128: 3 × 128 × 4 = 1536 bytes (trivial)
//
// Grid: (ceil(N/TILE_N), kb_outer)
//   When kb_outer==1: plain store (no atomicAdd, no memset)
//   When kb_outer>1:  atomicAdd (but WK× fewer than grid_kpar)
//
// Benchmark motivation (MI50, gfx906):
//   FFN Gate/Up (18944×3584): grid_kpar KB=4 → 756 GB/s (76% peak)
//     - 148 N-blocks × 4 K-blocks = 592 wavefronts = 9.9 waves/CU
//     - 18944 × 4 = 75776 atomicAdds per launch
//   NEW lds_kreduce kb_outer=1: same 592 wavefronts, zero atomicAdds
//   Q/Wo (3584×3584): grid_kpar KB=16 → 456 GB/s (46% peak)
//     - 28 × 16 = 448 wavefronts, 3584 × 16 = 57344 atomicAdds
//   NEW lds_kreduce kb_outer=4: 28×4×4 = 448 wavefronts, 3584 × 4 atomicAdds
// ============================================================================
template <int TILE_N, int CPT, int WK>
__global__ __launch_bounds__(TILE_N / CPT * WK)
void gemv_int8_int8_lds_kreduce_kernel_t(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B_int8_vnni,   // [K/4][N][4] flat VNNI
    int32_t* __restrict__ d_C_int32,
    int N, int K, int kb_outer)
{
    static_assert(CPT == 2 || CPT == 4, "CPT must be 2 or 4");
    static_assert(WK >= 2 && WK <= 8, "WK must be 2-8");
    constexpr int THREADS_PER_WAVE = TILE_N / CPT;
    static_assert(THREADS_PER_WAVE == 64, "TILE_N/CPT must equal 64 (one wavefront)");

    const int wave_id = threadIdx.x / THREADS_PER_WAVE;
    const int lane_id = threadIdx.x % THREADS_PER_WAVE;
    const int n_base = blockIdx.x * TILE_N + lane_id * CPT;
    const int outer_k_block = blockIdx.y;

    // --- Compute K-range for this block's outer K-slice ---
    const int total_k_groups = K / 4;

    // Evenly distribute k-groups across kb_outer grid blocks
    const int kgroups_per_outer = (total_k_groups + kb_outer - 1) / kb_outer;
    const int outer_kg_start = outer_k_block * kgroups_per_outer;
    const int outer_kg_end = min(outer_kg_start + kgroups_per_outer, total_k_groups);
    if (outer_kg_start >= total_k_groups) return;

    const int outer_kgroup_count = outer_kg_end - outer_kg_start;

    // --- Inner K-split: distribute outer range across WK wavefronts ---
    // Balanced distribution: waves 0..remainder-1 get one extra k-group
    const int kg_per_wave_base = outer_kgroup_count / WK;
    const int kg_remainder = outer_kgroup_count % WK;
    const int my_kg_start = outer_kg_start
        + wave_id * kg_per_wave_base
        + min(wave_id, kg_remainder);
    const int my_kg_count = kg_per_wave_base + (wave_id < kg_remainder ? 1 : 0);
    const int my_kg_end = my_kg_start + my_kg_count;

    // Accumulators — each thread accumulates CPT output column values
    int32_t acc[CPT];
    #pragma unroll
    for (int c = 0; c < CPT; ++c) acc[c] = 0;

    // Main K-reduction loop (innermost hot loop)
    if (n_base < N && my_kg_count > 0)
    {
        const int32_t* __restrict__ b_row =
            reinterpret_cast<const int32_t*>(d_B_int8_vnni) + n_base;
        const size_t b_stride = static_cast<size_t>(N);

        // #pragma unroll 4: expose 4 k-groups for memory-level parallelism.
        // Compiler generates 4× global_load + 4× s_load + 8× v_dot4 interleaved.
        #pragma unroll 4
        for (int kg = my_kg_start; kg < my_kg_end; ++kg)
        {
            // A is uniform across all threads in this wave (M=1)
            // → compiler uses scalar load (s_load_dword), broadcast to all lanes
            const int32_t a_packed = *reinterpret_cast<const int32_t*>(d_A_int8 + kg * 4);

            if constexpr (CPT == 2)
            {
                // 8-byte vectorized load: two adjacent VNNI columns
                const int2 b_pair = *reinterpret_cast<const int2*>(
                    b_row + kg * b_stride);
                acc[0] = __builtin_amdgcn_sdot4(a_packed, b_pair.x, acc[0], false);
                acc[1] = __builtin_amdgcn_sdot4(a_packed, b_pair.y, acc[1], false);
            }
            else // CPT == 4
            {
                // 16-byte vectorized load: four adjacent VNNI columns
                const int4 b_pack = *reinterpret_cast<const int4*>(
                    b_row + kg * b_stride);
                acc[0] = __builtin_amdgcn_sdot4(a_packed, b_pack.x, acc[0], false);
                acc[1] = __builtin_amdgcn_sdot4(a_packed, b_pack.y, acc[1], false);
                acc[2] = __builtin_amdgcn_sdot4(a_packed, b_pack.z, acc[2], false);
                acc[3] = __builtin_amdgcn_sdot4(a_packed, b_pack.w, acc[3], false);
            }
        }
    }

    // =========================================================================
    // LDS K-reduction: sum partial results across WK wavefronts
    //
    // Waves 1..WK-1 deposit their partial sums into LDS.
    // After barrier, wave 0 reads and sums all partials + its own accumulator.
    // This replaces global atomicAdd with fast LDS reads (5-10 cycles each).
    // =========================================================================
    __shared__ int32_t s_partial[WK - 1][TILE_N];

    // Waves 1..WK-1 write partials to LDS
    if (wave_id > 0 && n_base < N)
    {
        #pragma unroll
        for (int c = 0; c < CPT; ++c)
        {
            s_partial[wave_id - 1][lane_id * CPT + c] = acc[c];
        }
    }

    __syncthreads();

    // Wave 0 sums all partials and writes to global output
    if (wave_id == 0 && n_base < N)
    {
        // Sum own partial + (WK-1) LDS contributions
        #pragma unroll
        for (int w = 0; w < WK - 1; ++w)
        {
            #pragma unroll
            for (int c = 0; c < CPT; ++c)
            {
                acc[c] += s_partial[w][lane_id * CPT + c];
            }
        }

        // Store output
        if (kb_outer == 1)
        {
            // Plain store — no atomicAdd needed, no pre-zeroing required
            if constexpr (CPT == 2)
            {
                *reinterpret_cast<int2*>(&d_C_int32[n_base]) =
                    make_int2(acc[0], acc[1]);
            }
            else // CPT == 4
            {
                *reinterpret_cast<int4*>(&d_C_int32[n_base]) =
                    make_int4(acc[0], acc[1], acc[2], acc[3]);
            }
        }
        else
        {
            // Multi-block outer K-split: atomicAdd (but WK× fewer than grid_kpar)
            #pragma unroll
            for (int c = 0; c < CPT; ++c)
            {
                atomicAdd(&d_C_int32[n_base + c], acc[c]);
            }
        }
    }
}


// ============================================================================
// Native-VNNI GEMV kernel: lossless Q4/IQ4 decode with per-block FP16 scales
//
// Eliminates ratio-VNNI's (ratio * d + 64) >> 7 truncation by keeping
// per-block scales separate and decoding nibbles directly to INT8:
//   Q4_0:   w_int8 = q - 8     (exact)
//   IQ4_NL: w_int8 = lut[q]    (exact)
//
// Compute flow per block (32 elements):
//   int32_block = Σ sdot4(a_packed, w_packed)   // 8 sdot4 calls
//   f_acc += (float)int32_block * __half2float(block_scale)
//
// Output is FP32 with scale_A applied inline — no epilogue kernel needed.
// ============================================================================
template <int TILE_N, int CPT, bool IQ4>
__global__ void gemv_native_vnni_kernel_t(
    const int8_t*    __restrict__ d_A_int8,       // [K] quantized activations (INT8)
    const uint8_t*   __restrict__ d_payload,      // [blocks_per_row × N × 16] nibble payload
    const uint16_t*  __restrict__ d_block_scales,  // [blocks_per_row × N] FP16 as uint16_t
    float*           __restrict__ d_C_fp32,        // [N] output (FP32)
    const float*     __restrict__ d_scale_A,       // [1] activation scale
    int N, int K, int kblocks)
{
    static_assert(CPT == 1 || CPT == 2, "native-VNNI supports CPT in {1,2}");

    const int n_base = blockIdx.x * TILE_N + threadIdx.x * CPT;
    const bool active = (n_base < N);
    const int kb = blockIdx.y;

    const int blocks_per_row = K / 32;
    const int b_per = (blocks_per_row + kblocks - 1) / kblocks;
    const int b_start = kb * b_per;
    const int b_end = min(b_start + b_per, blocks_per_row);

    float f_acc[CPT];
    #pragma unroll
    for (int c = 0; c < CPT; ++c)
        f_acc[c] = 0.0f;

    if (active)
    {
        for (int b = b_start; b < b_end; ++b)
        {
            // A pointer for this block's 32 INT8 activation values
            const int32_t* a4 = reinterpret_cast<const int32_t*>(d_A_int8 + b * 32);

            #pragma unroll
            for (int c = 0; c < CPT; ++c)
            {
                const int n = n_base + c;
                if (n < N)
                {
                    const size_t linear = static_cast<size_t>(b) * static_cast<size_t>(N) + static_cast<size_t>(n);
                    const uint8_t* payload16 = d_payload + linear * 16;
                    const uint16_t scale_bits = d_block_scales[linear];

                    // Decode 32 nibbles to 8 packed INT32 words (4 INT8 each)
                    // Q4_0 nibble layout: qs[j] = lo_nibble(q_j) | hi_nibble(q_{j+16})
                    // We must decode in sequential element order [q0,q1,...,q31] to match
                    // the sequential activation layout [a0,a1,...,a31] for sdot4.
                    //
                    // Groups 0-3: elements [0..15] from low nibbles of qs[0..15]
                    // Groups 4-7: elements [16..31] from high nibbles of qs[0..15]
                    int32_t packed_groups[8];

                    if constexpr (IQ4)
                    {
                        // IQ4_NL: nibble → LUT lookup → INT8
                        #pragma unroll
                        for (int g = 0; g < 4; ++g)
                        {
                            const uint8_t b0 = payload16[g * 4 + 0];
                            const uint8_t b1 = payload16[g * 4 + 1];
                            const uint8_t b2 = payload16[g * 4 + 2];
                            const uint8_t b3 = payload16[g * 4 + 3];

                            // Group g: elements [4g..4g+3] via low nibbles
                            const int8_t lo0 = k_iq4_codebook_i8[b0 & 0x0F];
                            const int8_t lo1 = k_iq4_codebook_i8[b1 & 0x0F];
                            const int8_t lo2 = k_iq4_codebook_i8[b2 & 0x0F];
                            const int8_t lo3 = k_iq4_codebook_i8[b3 & 0x0F];
                            packed_groups[g] = static_cast<int32_t>(
                                (static_cast<uint32_t>(static_cast<uint8_t>(lo0))      ) |
                                (static_cast<uint32_t>(static_cast<uint8_t>(lo1)) <<  8) |
                                (static_cast<uint32_t>(static_cast<uint8_t>(lo2)) << 16) |
                                (static_cast<uint32_t>(static_cast<uint8_t>(lo3)) << 24));

                            // Group g+4: elements [4g+16..4g+19] via high nibbles
                            const int8_t hi0 = k_iq4_codebook_i8[b0 >> 4];
                            const int8_t hi1 = k_iq4_codebook_i8[b1 >> 4];
                            const int8_t hi2 = k_iq4_codebook_i8[b2 >> 4];
                            const int8_t hi3 = k_iq4_codebook_i8[b3 >> 4];
                            packed_groups[g + 4] = static_cast<int32_t>(
                                (static_cast<uint32_t>(static_cast<uint8_t>(hi0))      ) |
                                (static_cast<uint32_t>(static_cast<uint8_t>(hi1)) <<  8) |
                                (static_cast<uint32_t>(static_cast<uint8_t>(hi2)) << 16) |
                                (static_cast<uint32_t>(static_cast<uint8_t>(hi3)) << 24));
                        }
                    }
                    else
                    {
                        // Q4_0: nibble → (q - 8) → INT8 (exact, sequential order)
                        #pragma unroll
                        for (int g = 0; g < 4; ++g)
                        {
                            const uint8_t b0 = payload16[g * 4 + 0];
                            const uint8_t b1 = payload16[g * 4 + 1];
                            const uint8_t b2 = payload16[g * 4 + 2];
                            const uint8_t b3 = payload16[g * 4 + 3];

                            // Group g: elements [4g..4g+3] via low nibbles
                            const int8_t lo0 = static_cast<int8_t>((b0 & 0x0F) - 8);
                            const int8_t lo1 = static_cast<int8_t>((b1 & 0x0F) - 8);
                            const int8_t lo2 = static_cast<int8_t>((b2 & 0x0F) - 8);
                            const int8_t lo3 = static_cast<int8_t>((b3 & 0x0F) - 8);
                            packed_groups[g] = static_cast<int32_t>(
                                (static_cast<uint32_t>(static_cast<uint8_t>(lo0))      ) |
                                (static_cast<uint32_t>(static_cast<uint8_t>(lo1)) <<  8) |
                                (static_cast<uint32_t>(static_cast<uint8_t>(lo2)) << 16) |
                                (static_cast<uint32_t>(static_cast<uint8_t>(lo3)) << 24));

                            // Group g+4: elements [4g+16..4g+19] via high nibbles
                            const int8_t hi0 = static_cast<int8_t>((b0 >> 4) - 8);
                            const int8_t hi1 = static_cast<int8_t>((b1 >> 4) - 8);
                            const int8_t hi2 = static_cast<int8_t>((b2 >> 4) - 8);
                            const int8_t hi3 = static_cast<int8_t>((b3 >> 4) - 8);
                            packed_groups[g + 4] = static_cast<int32_t>(
                                (static_cast<uint32_t>(static_cast<uint8_t>(hi0))      ) |
                                (static_cast<uint32_t>(static_cast<uint8_t>(hi1)) <<  8) |
                                (static_cast<uint32_t>(static_cast<uint8_t>(hi2)) << 16) |
                                (static_cast<uint32_t>(static_cast<uint8_t>(hi3)) << 24));
                        }
                    }

                    // 8× sdot4: INT8 dot product of 32 activation × weight elements
                    int32_t block_acc = 0;
                    #pragma unroll
                    for (int g = 0; g < 8; ++g)
                    {
                        block_acc = __builtin_amdgcn_sdot4(a4[g], packed_groups[g], block_acc, false);
                    }

                    // Convert FP16 scale to FP32 and accumulate
                    const float block_scale = __half2float(*reinterpret_cast<const __half*>(&scale_bits));
                    f_acc[c] += static_cast<float>(block_acc) * block_scale;
                }
            }
        }

        // Apply activation scale and write output
        const float scale_a = d_scale_A[0];
        #pragma unroll
        for (int c = 0; c < CPT; ++c)
        {
            const int n = n_base + c;
            if (n < N)
            {
                const float result = f_acc[c] * scale_a;
                if (kblocks == 1)
                {
                    d_C_fp32[n] = result;
                }
                else
                {
                    atomicAdd(&d_C_fp32[n], result);
                }
            }
        }
    }
}


// ============================================================================
// Fused FP32→INT8 GEMV kernel: Quantize + GEMV + Scale in one launch
//
// Eliminates the 3-kernel pipeline (quantize → GEMV → scale) by:
//   1. Loading FP32 activations to shared memory per K-block
//   2. Computing block-local absmax for INT8 quantization
//   3. Quantizing in-register and using v_dot4_i32_i8 for the dot product
//   4. Converting INT32 accumulator to FP32 with (block_scale * scale_B[n])
//   5. atomicAdd FP32 partial results across K-blocks
//   6. Optional fused bias addition
//
// Memory traffic: Same weight reads as INT8 GEMV, but no separate
// activation INT8 buffer, no INT32 output buffer, no scale_A buffer.
// Activations (14KB for K=3584) stay in L2 after first projection.
//
// Grid: dim3(grid_n, KB), Block: dim3(TILE_N/CPT)
// Same grid structure as grid_kpar, but FP32 in → FP32 out.
// ============================================================================
template <int TILE_N, int CPT>
__global__ void gemv_fused_fp32_int8_vnni_kernel_t(
    const float* __restrict__ d_A_fp32,          // [K] FP32 activations
    const int8_t* __restrict__ d_B_int8_vnni,    // [K/4][N][4] VNNI weights
    float* __restrict__ d_C_fp32,                // [N] FP32 output (zeroed before launch)
    const float* __restrict__ d_scales_B,        // [N] per-column weight scales
    const float* __restrict__ d_bias,            // [N] optional bias (nullable)
    int N, int K, int kblocks)
{
    constexpr int THREADS = TILE_N / CPT;  // 64 threads = 1 wavefront
    // Process K in chunks of QUANT_TILE_K elements at a time for quantization.
    // Each chunk: load FP32 → find absmax → quantize → dot product.
    // 256 elements = 1KB FP32 in shared memory, fits easily.
    constexpr int QUANT_TILE_K = 256;

    const int n_base = blockIdx.x * TILE_N + threadIdx.x * CPT;
    const bool active = (n_base < N);
    const int k_block = blockIdx.y;
    const int tid = threadIdx.x;

    // Each block processes its K-segment: [k_start, k_end)
    // Round k_per UP to a multiple of 4 so every k_start aligns with
    // VNNI K-groups (4 consecutive K values packed together).
    // Without this, k_start for intermediate blocks can land mid-group,
    // causing activation/weight misalignment in sdot4.
    const int k_per_raw = (K + kblocks - 1) / kblocks;
    const int k_per = ((k_per_raw + 3) / 4) * 4;
    const int k_start = k_block * k_per;
    const int k_end = min(k_start + k_per, K);

    // After rounding k_per up, the last block(s) may have k_start >= K.
    // Those blocks have nothing to do — skip entirely.
    if (k_start >= K) return;

    // Shared memory: FP32 activations tile + reduction scratch
    __shared__ float s_A_fp32[QUANT_TILE_K];
    __shared__ float s_absmax;        // Block-wide absmax for this tile
    __shared__ int8_t s_A_int8[QUANT_TILE_K]; // Quantized activations

    // FP32 accumulators for partial results (will be atomicAdd'd)
    float f_acc[CPT];
    #pragma unroll
    for (int c = 0; c < CPT; ++c) f_acc[c] = 0.0f;

    // Process K-segment in QUANT_TILE_K-sized chunks
    for (int k_tile_start = k_start; k_tile_start < k_end; k_tile_start += QUANT_TILE_K)
    {
        const int tile_end = min(k_tile_start + QUANT_TILE_K, k_end);
        const int tile_len = tile_end - k_tile_start;
        const int tile_kgroups = tile_len / 4;

        // --- Phase 1: Cooperative load of FP32 activations to shared memory ---
        for (int i = tid; i < tile_len; i += THREADS)
        {
            s_A_fp32[i] = d_A_fp32[k_tile_start + i];
        }
        __syncthreads();

        // --- Phase 2: Find absmax via parallel reduction ---
        // Each thread finds local max over its portion
        float local_max = 0.0f;
        for (int i = tid; i < tile_len; i += THREADS)
        {
            float v = s_A_fp32[i];
            float a = (v >= 0.0f) ? v : -v;
            local_max = (a > local_max) ? a : local_max;
        }

        // Warp reduction (wavefront = 64 threads on gfx906)
        #pragma unroll
        for (int offset = 32; offset > 0; offset >>= 1)
        {
            float other = __shfl_down(local_max, offset);
            local_max = (other > local_max) ? other : local_max;
        }

        // Thread 0 writes the block absmax
        if (tid == 0)
        {
            s_absmax = local_max;
        }
        __syncthreads();

        const float absmax = s_absmax;
        const float scale = (absmax > 0.0f) ? (absmax / 127.0f) : 1.0f;
        const float inv_scale = (absmax > 0.0f) ? (127.0f / absmax) : 0.0f;

        // --- Phase 3: Cooperative quantize FP32 → INT8 in shared memory ---
        for (int i = tid; i < tile_len; i += THREADS)
        {
            float val = s_A_fp32[i] * inv_scale;
            int q = static_cast<int>(rintf(val));
            q = (q < -127) ? -127 : ((q > 127) ? 127 : q);
            s_A_int8[i] = static_cast<int8_t>(q);
        }
        __syncthreads();

        // --- Phase 4: INT8 dot product using v_dot4_i32_i8 ---
        if (active)
        {
            int32_t tile_acc[CPT];
            #pragma unroll
            for (int c = 0; c < CPT; ++c) tile_acc[c] = 0;

            const int32_t* __restrict__ b_row =
                reinterpret_cast<const int32_t*>(d_B_int8_vnni) + n_base;
            const size_t b_stride = static_cast<size_t>(N);
            const int kg_offset = k_tile_start / 4;

            #pragma unroll 4
            for (int tkg = 0; tkg < tile_kgroups; ++tkg)
            {
                const int32_t a_packed = *reinterpret_cast<const int32_t*>(
                    s_A_int8 + tkg * 4);

                if constexpr (CPT == 2)
                {
                    const int2 b_pair = *reinterpret_cast<const int2*>(
                        b_row + (kg_offset + tkg) * b_stride);
                    tile_acc[0] = __builtin_amdgcn_sdot4(a_packed, b_pair.x, tile_acc[0], false);
                    tile_acc[1] = __builtin_amdgcn_sdot4(a_packed, b_pair.y, tile_acc[1], false);
                }
                else
                {
                    #pragma unroll
                    for (int c = 0; c < CPT; ++c)
                    {
                        const int32_t b_packed = *(b_row + (kg_offset + tkg) * b_stride + c);
                        tile_acc[c] = __builtin_amdgcn_sdot4(a_packed, b_packed, tile_acc[c], false);
                    }
                }
            }

            // Convert tile INT32 → FP32 with tile-local scale
            #pragma unroll
            for (int c = 0; c < CPT; ++c)
            {
                f_acc[c] += static_cast<float>(tile_acc[c]) * scale;
            }
        }

        __syncthreads();  // Ensure shared memory is safe for next tile
    }

    // --- Phase 5: Apply weight scales, write to output ---
    if (active)
    {
        #pragma unroll
        for (int c = 0; c < CPT; ++c)
        {
            float result = f_acc[c] * d_scales_B[n_base + c];
            if (kblocks == 1)
                d_C_fp32[n_base + c] = result;      // Plain store — no contention
            else
                atomicAdd(&d_C_fp32[n_base + c], result);
        }
    }

    // Bias: only apply once across all K-blocks (last block does it)
    // We use k_block == kblocks - 1 as the "last" block.
    // Note: this has a data race if atomicAdds from other blocks haven't
    // completed yet. Instead, bias is applied as a separate follow-up.
    // See dispatch function for bias handling.
}


// ============================================================================
// Fused FP32→INT8 GEMV scatter kernel: Quantize + GEMV → partial buffer
//
// Same inner loop as gemv_fused_fp32_int8_vnni_kernel_t but writes unscaled
// FP32 partials to d_partial[k_block * N + n], enabling grid-level K-splitting
// WITHOUT atomicAdd or memset. Paired with gemv_reduce_scale_kernel_t to
// produce the final FP32 output.
//
// This replaces the 3-kernel pipeline (quantize → GEMV → scale) with a
// 2-kernel pipeline (fused_scatter → reduce+scale) that eliminates:
//   - The separate quantize kernel launch
//   - hipMemsetAsync for atomicAdd zeroing
//   - All atomicAdd contention
//
// Grid: dim3(grid_n, KB), Block: dim3(TILE_N / CPT) = dim3(64)
// ============================================================================
template <int TILE_N, int CPT>
__global__ void gemv_fused_scatter_int8_vnni_kernel_t(
    const float* __restrict__ d_A_fp32,          // [K] FP32 activations
    const int8_t* __restrict__ d_B_int8_vnni,    // [K/4][N][4] VNNI weights
    float* __restrict__ d_partial_fp32,           // [KB × N] FP32 partial buffer
    int N, int K, int kblocks)
{
    constexpr int THREADS = TILE_N / CPT;  // 64 threads = 1 wavefront
    constexpr int QUANT_TILE_K = 256;

    const int n_base = blockIdx.x * TILE_N + threadIdx.x * CPT;
    const bool active = (n_base < N);
    const int k_block = blockIdx.y;
    const int tid = threadIdx.x;

    const int k_per_raw = (K + kblocks - 1) / kblocks;
    const int k_per = ((k_per_raw + 3) / 4) * 4;
    const int k_start = k_block * k_per;
    const int k_end = min(k_start + k_per, K);

    if (k_start >= K) return;

    __shared__ float s_A_fp32[QUANT_TILE_K];
    __shared__ float s_absmax;
    __shared__ int8_t s_A_int8[QUANT_TILE_K];

    float f_acc[CPT];
    #pragma unroll
    for (int c = 0; c < CPT; ++c) f_acc[c] = 0.0f;

    for (int k_tile_start = k_start; k_tile_start < k_end; k_tile_start += QUANT_TILE_K)
    {
        const int tile_end = min(k_tile_start + QUANT_TILE_K, k_end);
        const int tile_len = tile_end - k_tile_start;
        const int tile_kgroups = tile_len / 4;

        // Phase 1: Cooperative load of FP32 activations to shared memory
        for (int i = tid; i < tile_len; i += THREADS)
            s_A_fp32[i] = d_A_fp32[k_tile_start + i];
        __syncthreads();

        // Phase 2: Find absmax via warp reduction
        float local_max = 0.0f;
        for (int i = tid; i < tile_len; i += THREADS)
        {
            float v = s_A_fp32[i];
            float a = (v >= 0.0f) ? v : -v;
            local_max = (a > local_max) ? a : local_max;
        }

        #pragma unroll
        for (int offset = 32; offset > 0; offset >>= 1)
        {
            float other = __shfl_down(local_max, offset);
            local_max = (other > local_max) ? other : local_max;
        }

        if (tid == 0) s_absmax = local_max;
        __syncthreads();

        const float absmax = s_absmax;
        const float scale = (absmax > 0.0f) ? (absmax / 127.0f) : 1.0f;
        const float inv_scale = (absmax > 0.0f) ? (127.0f / absmax) : 0.0f;

        // Phase 3: Cooperative quantize FP32 → INT8 in shared memory
        for (int i = tid; i < tile_len; i += THREADS)
        {
            float val = s_A_fp32[i] * inv_scale;
            int q = static_cast<int>(rintf(val));
            q = (q < -127) ? -127 : ((q > 127) ? 127 : q);
            s_A_int8[i] = static_cast<int8_t>(q);
        }
        __syncthreads();

        // Phase 4: INT8 dot product using v_dot4_i32_i8
        if (active)
        {
            int32_t tile_acc[CPT];
            #pragma unroll
            for (int c = 0; c < CPT; ++c) tile_acc[c] = 0;

            const int32_t* __restrict__ b_row =
                reinterpret_cast<const int32_t*>(d_B_int8_vnni) + n_base;
            const size_t b_stride = static_cast<size_t>(N);
            const int kg_offset = k_tile_start / 4;

            #pragma unroll 4
            for (int tkg = 0; tkg < tile_kgroups; ++tkg)
            {
                const int32_t a_packed = *reinterpret_cast<const int32_t*>(
                    s_A_int8 + tkg * 4);

                if constexpr (CPT == 2)
                {
                    const int2 b_pair = *reinterpret_cast<const int2*>(
                        b_row + (kg_offset + tkg) * b_stride);
                    tile_acc[0] = __builtin_amdgcn_sdot4(a_packed, b_pair.x, tile_acc[0], false);
                    tile_acc[1] = __builtin_amdgcn_sdot4(a_packed, b_pair.y, tile_acc[1], false);
                }
                else
                {
                    #pragma unroll
                    for (int c = 0; c < CPT; ++c)
                    {
                        const int32_t b_packed = *(b_row + (kg_offset + tkg) * b_stride + c);
                        tile_acc[c] = __builtin_amdgcn_sdot4(a_packed, b_packed, tile_acc[c], false);
                    }
                }
            }

            // Convert tile INT32 → FP32 with tile-local scale
            #pragma unroll
            for (int c = 0; c < CPT; ++c)
                f_acc[c] += static_cast<float>(tile_acc[c]) * scale;
        }

        __syncthreads();
    }

    // Phase 5: Write unscaled FP32 partials to buffer (plain store, no atomicAdd)
    if (active)
    {
        const size_t partial_offset = static_cast<size_t>(k_block) * N;
        #pragma unroll
        for (int c = 0; c < CPT; ++c)
            d_partial_fp32[partial_offset + n_base + c] = f_acc[c];
    }
}


// ============================================================================
// Reduce + scale kernel: Sum KB FP32 partials, apply scale_B + bias
//
// Used after gemv_fused_scatter_int8_vnni_kernel_t to combine K-block
// partial results into the final FP32 output with scaling and bias.
//
// Grid: dim3((N + TILE_N - 1) / TILE_N), Block: dim3(TILE_N / CPT)
// ============================================================================
template <int TILE_N, int CPT>
__global__ void gemv_reduce_scale_kernel_t(
    const float* __restrict__ d_partial_fp32,    // [KB × N] partial buffer
    float* __restrict__ d_C_fp32,                // [N] output
    const float* __restrict__ d_scales_B,        // [N] per-column weight scales
    const float* __restrict__ d_bias,            // [N] optional bias (nullable)
    int N, int KB,
    float alpha, float beta,
    const float* __restrict__ d_C_existing)      // [N] used when beta != 0
{
    const int n_base = blockIdx.x * TILE_N + threadIdx.x * CPT;

    #pragma unroll
    for (int c = 0; c < CPT; ++c)
    {
        const int n = n_base + c;
        if (n >= N) return;

        float sum = 0.0f;
        #pragma unroll 16
        for (int kb = 0; kb < KB; ++kb)
            sum += d_partial_fp32[static_cast<size_t>(kb) * N + n];

        float result = sum * d_scales_B[n] * alpha;
        if (beta != 0.0f && d_C_existing)
            result += beta * d_C_existing[n];
        if (d_bias)
            result += d_bias[n];

        d_C_fp32[n] = result;
    }
}


// ============================================================================
// C API: Launch wrappers
// ============================================================================

extern "C" {


void rocmGemv_int8_vnni_set_tuning_overrides(
    int tn,
    int kb)
{
    g_int8_vnni_tn_override.store(tn, std::memory_order_relaxed);
    g_int8_vnni_kb_override.store(kb, std::memory_order_relaxed);
}

void rocmGemv_int8_vnni_reset_tuning_overrides()
{
    rocmGemv_int8_vnni_set_tuning_overrides(-1, -1);
}

void rocmGemv_int8_vnni_set_wide_tuning_overrides(
    int wide_tn,
    int wide_cpt,
    int wide_vec_load)
{
    g_int8_vnni_wide_tn_override.store(wide_tn, std::memory_order_relaxed);
    g_int8_vnni_wide_cpt_override.store(wide_cpt, std::memory_order_relaxed);
    g_int8_vnni_wide_vec_load_override.store(wide_vec_load, std::memory_order_relaxed);
}

void rocmGemv_int8_vnni_reset_wide_tuning_overrides()
{
    rocmGemv_int8_vnni_set_wide_tuning_overrides(-1, -1, -1);
}

bool rocmGemv_int8_int8_int32_vnni(
    const int8_t* d_A_int8,
    const int8_t* d_B_int8_vnni,
    int32_t* d_C_int32,
    int N, int K,
    int device_id, void* stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_id);

    if ((K % 4) != 0)
    {
        fprintf(stderr, "[rocmGemv_int8_int8_int32_vnni] K must be multiple of 4 (N=%d K=%d)\n", N, K);
        return false;
    }

    const bool use_vec4 = ((N % 4) == 0);
    const int wide_tn_override = g_int8_vnni_wide_tn_override.load(std::memory_order_relaxed);
    const int wide_cpt_override = g_int8_vnni_wide_cpt_override.load(std::memory_order_relaxed);
    const int wide_vec_load_override = g_int8_vnni_wide_vec_load_override.load(std::memory_order_relaxed);

    int wide_tn = GEMV_INT8_VNNI_WIDE_TILE_N;
    if (wide_tn_override == 128 || wide_tn_override == 256 || wide_tn_override == 512)
    {
        wide_tn = wide_tn_override;
    }

    int wide_cpt = GEMV_INT8_VNNI_WIDE_DEFAULT_CPT;
    if (wide_cpt_override == 2 || wide_cpt_override == 4)
    {
        wide_cpt = wide_cpt_override;
    }

    bool wide_vec_load = true;
    if (wide_vec_load_override == 0)
    {
        wide_vec_load = false;
    }
    else if (wide_vec_load_override == 1)
    {
        wide_vec_load = true;
    }

    const bool use_wide_vec4 = (N >= 8 * K) && (N >= wide_tn);
    // Grid-level K-parallel: handles ALL non-wide_vec4 shapes with N%4==0.
    // Previous wide2 path (N>=3K, N<8K) only generated N/256 blocks → 148 waves
    // for FFN Gate/Up, achieving just 17% peak BW. Grid_kpar K-splits via
    // blockIdx.y to generate thousands of blocks, fully saturating the GPU.
    // Profiling: grid_kpar achieves 488-649 GB/s (48-63% peak) vs wide2's 176 GB/s.
    const bool use_grid_kpar = !use_wide_vec4 && use_vec4 && (N >= GEMV_INT8_VNNI_GRID_KPAR_TILE_N);
    const bool use_square = !use_wide_vec4 && !use_grid_kpar && (N >= GEMV_INT8_VNNI_SQUARE_TILE_N);
    const int int8_tn_override = g_int8_vnni_tn_override.load(std::memory_order_relaxed);
    const int int8_kb_override = g_int8_vnni_kb_override.load(std::memory_order_relaxed);

    if (use_wide_vec4)
    {
        const int grid_n = (N + wide_tn - 1) / wide_tn;
        dim3 block(wide_tn / wide_cpt, 1, 1);

        if (wide_tn == 128)
        {
            if (wide_cpt == 2)
            {
                if (wide_vec_load)
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_kernel_t<128, 2, true>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, N, K);
                }
                else
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_kernel_t<128, 2, false>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, N, K);
                }
            }
            else
            {
                if (wide_vec_load)
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_kernel_t<128, 4, true>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, N, K);
                }
                else
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_kernel_t<128, 4, false>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, N, K);
                }
            }
        }
        else if (wide_tn == 512)
        {
            if (wide_cpt == 2)
            {
                if (wide_vec_load)
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_kernel_t<512, 2, true>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, N, K);
                }
                else
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_kernel_t<512, 2, false>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, N, K);
                }
            }
            else
            {
                if (wide_vec_load)
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_kernel_t<512, 4, true>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, N, K);
                }
                else
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_kernel_t<512, 4, false>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, N, K);
                }
            }
        }
        else
        {
            if (wide_cpt == 2)
            {
                if (wide_vec_load)
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_kernel_t<256, 2, true>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, N, K);
                }
                else
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_kernel_t<256, 2, false>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, N, K);
                }
            }
            else
            {
                if (wide_vec_load)
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_kernel_t<256, 4, true>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, N, K);
                }
                else
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_kernel_t<256, 4, false>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, N, K);
                }
            }
        }
    }
    else if (use_grid_kpar)
    {
        // Grid-level K-parallel with atomicAdd: KB independent blocks per output tile
        // Each block processes K/KB fraction; output must be zeroed before launch
        //
        // KB heuristic: unified formula for all shapes.
        //
        // Key insight from exhaustive TN×KB sweep (MI50 gfx906, Qwen7B shapes):
        //   ~8 waves/CU is the sweet spot for ALL shapes. More waves just adds
        //   atomicAdd contention per N-tile without improving BW utilization.
        //   Additionally, KB values that evenly divide k_groups avoid tail-effect
        //   work imbalance (some wavefronts processing fewer k-groups than others).
        //
        // Sweep results (optimal KB per shape):
        //   Q/Wo (3584×3584):     KB=16, 7.5 w/CU, 56 kgrp/wave → 456 GB/s (46%)
        //   K/V (512×3584):       KB=56, 3.7 w/CU, 16 kgrp/wave → 129 GB/s (13%)
        //   FFN Gate/Up (18944×3584): KB=4, 9.9 w/CU, 224 kgrp/wave → 756 GB/s (76%)
        //   FFN Down (3584×18944):    KB=16, 7.5 w/CU, 296 kgrp/wave → 760 GB/s (76%)
        //
        // Strategy:
        //   1. Compute min KB for ~8 waves/CU (memory latency hiding)
        //   2. Cap so inner loop runs ≥16 k-groups (proven floor from sweep)
        //   3. Round to nearest factor of k_groups (avoid uneven work distribution)
        //      Bias toward lower factor on ties (fewer atomicAdds = less contention)
        constexpr int TN = GEMV_INT8_VNNI_GRID_KPAR_TILE_N;  // 128
        constexpr int CPT = GEMV_INT8_VNNI_GRID_KPAR_CPT;    // 2
        constexpr int TN_WIDE = GEMV_INT8_VNNI_WIDE_TILE_N;  // 256
        constexpr int NUM_CUS = 60;  // MI50

        int tn = TN;
        if (int8_tn_override == TN_WIDE)
        {
            tn = TN_WIDE;
        }
        else if (int8_tn_override == TN)
        {
            tn = TN;
        }

        const int grid_n = (N + tn - 1) / tn;
        const int k_groups = K / 4;

        constexpr int TARGET_WAVES_PER_CU = 8;
        constexpr int MIN_KGROUPS_PER_WAVE = 16;  // proven floor from sweep

        int target_blocks = TARGET_WAVES_PER_CU * NUM_CUS;  // 480
        int kb_raw = max(2, (target_blocks + grid_n - 1) / grid_n);

        // Cap: ensure minimum inner loop length
        int kb_max = max(2, k_groups / MIN_KGROUPS_PER_WAVE);
        kb_raw = min(kb_raw, kb_max);

        // Round to nearest factor of k_groups (prefer lower on tie → fewer atomicAdds)
        int kb = kb_raw;
        if (k_groups % kb_raw != 0)
        {
            for (int d = 1; d < kb_raw; ++d)
            {
                if (kb_raw - d >= 2 && k_groups % (kb_raw - d) == 0) { kb = kb_raw - d; break; }
                if (kb_raw + d <= kb_max && k_groups % (kb_raw + d) == 0) { kb = kb_raw + d; break; }
            }
        }

        if (int8_kb_override > 0)
        {
            kb = int8_kb_override;
        }

        hipMemsetAsync(d_C_int32, 0, N * sizeof(int32_t), static_cast<hipStream_t>(stream));  // NOLINT(clang-diagnostic-unused-result)

        if (tn == TN_WIDE)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(gemv_int8_int8_grid_kpar_vnni_kernel_t<TN_WIDE, CPT>),
                dim3(grid_n, kb), dim3(TN_WIDE / CPT), 0, static_cast<hipStream_t>(stream),
                d_A_int8, d_B_int8_vnni, d_C_int32, N, K, kb);
        }
        else
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(gemv_int8_int8_grid_kpar_vnni_kernel_t<TN, CPT>),
                dim3(grid_n, kb), dim3(TN / CPT), 0, static_cast<hipStream_t>(stream),
                d_A_int8, d_B_int8_vnni, d_C_int32, N, K, kb);
        }
    }
    else if (use_square)
    {
        const int grid_n = (N + GEMV_INT8_VNNI_SQUARE_TILE_N - 1) / GEMV_INT8_VNNI_SQUARE_TILE_N;
        dim3 block(GEMV_INT8_VNNI_SQUARE_TILE_N, 1, 1);
        hipLaunchKernelGGL(
            gemv_int8_int8_square_vnni_kernel,
            dim3(grid_n), block, 0, static_cast<hipStream_t>(stream),
            d_A_int8, d_B_int8_vnni, d_C_int32, N, K);
    }
    else
    {
        const int grid_n = (N + GEMV_TILE_N - 1) / GEMV_TILE_N;
        hipLaunchKernelGGL(
            gemv_int8_int8_vnni_kernel,
            dim3(grid_n), dim3(GEMV_TILE_N), 0, static_cast<hipStream_t>(stream),
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

bool rocmGemv_int8_int8_fp32_vnni_scaled(
    const int8_t* d_A_int8,
    const int8_t* d_B_int8_vnni,
    float* d_C_fp32,
    const float* d_scale_A,
    const float* d_scale_B,
    int N, int K,
    float alpha,
    float beta,
    const float* d_C_existing,
    const float* d_bias,
    int device_id, void* stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_id);

    if ((K % 4) != 0)
    {
        fprintf(stderr, "[rocmGemv_int8_int8_fp32_vnni_scaled] K must be multiple of 4 (N=%d K=%d)\n", N, K);
        return false;
    }

    const bool use_vec4 = ((N % 4) == 0);
    const int wide_tn_override = g_int8_vnni_wide_tn_override.load(std::memory_order_relaxed);
    const int wide_cpt_override = g_int8_vnni_wide_cpt_override.load(std::memory_order_relaxed);
    const int wide_vec_load_override = g_int8_vnni_wide_vec_load_override.load(std::memory_order_relaxed);

    int wide_tn = GEMV_INT8_VNNI_WIDE_TILE_N;
    if (wide_tn_override == 128 || wide_tn_override == 256 || wide_tn_override == 512)
    {
        wide_tn = wide_tn_override;
    }

    int wide_cpt = GEMV_INT8_VNNI_WIDE_DEFAULT_CPT;
    if (wide_cpt_override == 2 || wide_cpt_override == 4)
    {
        wide_cpt = wide_cpt_override;
    }

    bool wide_vec_load = true;
    if (wide_vec_load_override == 0)
    {
        wide_vec_load = false;
    }
    else if (wide_vec_load_override == 1)
    {
        wide_vec_load = true;
    }

    const bool use_wide_vec4 = (N >= 8 * K) && (N >= wide_tn);
    const bool use_grid_kpar = !use_wide_vec4 && use_vec4 && (N >= GEMV_INT8_VNNI_GRID_KPAR_TILE_N);
    const bool use_square = !use_wide_vec4 && !use_grid_kpar && (N >= GEMV_INT8_VNNI_SQUARE_TILE_N);

    if (beta != 0.0f && d_C_existing == nullptr)
    {
        fprintf(stderr, "[rocmGemv_int8_int8_fp32_vnni_scaled] d_C_existing required when beta != 0 (N=%d K=%d)\n", N, K);
        return false;
    }

    if (use_grid_kpar)
    {
        return false;
    }

    if (use_wide_vec4)
    {
        const int grid_n = (N + wide_tn - 1) / wide_tn;
        dim3 block(wide_tn / wide_cpt, 1, 1);

        if (wide_tn == 128)
        {
            if (wide_cpt == 2)
            {
                if (wide_vec_load)
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_scaled_kernel_t<128, 2, true>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_fp32, d_scale_A, d_scale_B, N, K, alpha, beta, d_C_existing, d_bias);
                }
                else
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_scaled_kernel_t<128, 2, false>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_fp32, d_scale_A, d_scale_B, N, K, alpha, beta, d_C_existing, d_bias);
                }
            }
            else
            {
                if (wide_vec_load)
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_scaled_kernel_t<128, 4, true>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_fp32, d_scale_A, d_scale_B, N, K, alpha, beta, d_C_existing, d_bias);
                }
                else
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_scaled_kernel_t<128, 4, false>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_fp32, d_scale_A, d_scale_B, N, K, alpha, beta, d_C_existing, d_bias);
                }
            }
        }
        else if (wide_tn == 512)
        {
            if (wide_cpt == 2)
            {
                if (wide_vec_load)
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_scaled_kernel_t<512, 2, true>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_fp32, d_scale_A, d_scale_B, N, K, alpha, beta, d_C_existing, d_bias);
                }
                else
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_scaled_kernel_t<512, 2, false>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_fp32, d_scale_A, d_scale_B, N, K, alpha, beta, d_C_existing, d_bias);
                }
            }
            else
            {
                if (wide_vec_load)
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_scaled_kernel_t<512, 4, true>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_fp32, d_scale_A, d_scale_B, N, K, alpha, beta, d_C_existing, d_bias);
                }
                else
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_scaled_kernel_t<512, 4, false>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_fp32, d_scale_A, d_scale_B, N, K, alpha, beta, d_C_existing, d_bias);
                }
            }
        }
        else
        {
            if (wide_cpt == 2)
            {
                if (wide_vec_load)
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_scaled_kernel_t<256, 2, true>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_fp32, d_scale_A, d_scale_B, N, K, alpha, beta, d_C_existing, d_bias);
                }
                else
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_scaled_kernel_t<256, 2, false>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_fp32, d_scale_A, d_scale_B, N, K, alpha, beta, d_C_existing, d_bias);
                }
            }
            else
            {
                if (wide_vec_load)
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_scaled_kernel_t<256, 4, true>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_fp32, d_scale_A, d_scale_B, N, K, alpha, beta, d_C_existing, d_bias);
                }
                else
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_scaled_kernel_t<256, 4, false>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_fp32, d_scale_A, d_scale_B, N, K, alpha, beta, d_C_existing, d_bias);
                }
            }
        }
    }
    else if (use_square)
    {
        const int grid_n = (N + GEMV_INT8_VNNI_SQUARE_TILE_N - 1) / GEMV_INT8_VNNI_SQUARE_TILE_N;
        dim3 block(GEMV_INT8_VNNI_SQUARE_TILE_N, 1, 1);
        hipLaunchKernelGGL(
            gemv_int8_int8_square_vnni_scaled_kernel,
            dim3(grid_n), block, 0, static_cast<hipStream_t>(stream),
            d_A_int8, d_B_int8_vnni, d_C_fp32, d_scale_A, d_scale_B,
            N, K, alpha, beta, d_C_existing, d_bias);
    }
    else
    {
        const int grid_n = (N + GEMV_TILE_N - 1) / GEMV_TILE_N;
        hipLaunchKernelGGL(
            gemv_int8_int8_vnni_scaled_kernel,
            dim3(grid_n), dim3(GEMV_TILE_N), 0, static_cast<hipStream_t>(stream),
            d_A_int8, d_B_int8_vnni, d_C_fp32, d_scale_A, d_scale_B,
            N, K, alpha, beta, d_C_existing, d_bias);
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "[rocmGemv_int8_int8_fp32_vnni_scaled] Kernel launch failed: %s (N=%d K=%d)\n",
                hipGetErrorString(err), N, K);
        return false;
    }

    return true;
}


// =============================================================================
// Fused FP32→FP32 GEMV: Quantize + GEMV + Scale in one kernel
// =============================================================================
//
// Replaces the 3-kernel decode pipeline:
//   OLD: quantizeActivations → rocmGemv_int8_int8_int32_vnni → applyScaling
//   NEW: rocmGemv_fused_fp32_int8_vnni (single launch, directly FP32→FP32)
//
// Eliminates:
//   - d_A_int8 buffer (activation quantization done in shared memory)
//   - d_scales_A buffer (scale folded into per-tile FP32 conversion)
//   - d_CK_int32 buffer (output is FP32 directly)
//   - 2 kernel launches per projection (quantize + scale)
//
bool rocmGemv_fused_fp32_int8_vnni(
    const float* d_A_fp32,           // [K] FP32 activations
    const int8_t* d_B_int8_vnni,     // [K/4 × N × 4] VNNI-packed weights
    float* d_C_fp32,                 // [N] FP32 output (will be zeroed)
    const float* d_scales_B,         // [N] per-column weight scales
    const float* d_bias,             // [N] optional bias (nullable)
    int N, int K,
    int device_id, void* stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_id);

    if ((K % 4) != 0)
    {
        fprintf(stderr, "[rocmGemv_fused_fp32_int8_vnni] K must be multiple of 4 (N=%d K=%d)\n", N, K);
        return false;
    }

    if ((N % 4) != 0)
    {
        fprintf(stderr, "[rocmGemv_fused_fp32_int8_vnni] N must be multiple of 4 (N=%d K=%d)\n", N, K);
        return false;
    }

    constexpr int TN = GEMV_INT8_VNNI_GRID_KPAR_TILE_N;  // 128
    constexpr int CPT = GEMV_INT8_VNNI_GRID_KPAR_CPT;    // 2
    constexpr int NUM_CUS = 60;  // MI50/MI60

    const int grid_n = (N + TN - 1) / TN;
    const int k_groups = K / 4;

    // KB heuristic for FUSED kernel: very different from INT8 grid_kpar!
    //
    // The fused kernel uses FP32 atomicAdd for cross-block K-accumulation.
    // On gfx906, FP32 atomicAdd is emulated via CAS loop (not hardware native),
    // so ANY K-splitting causes massive contention overhead.
    //
    // Force kb=1: each block processes the entire K dimension sequentially
    // via QUANT_TILE_K=256 tiles in a loop, then does a single plain store.
    // No atomicAdd, no hipMemsetAsync needed.
    const int kb = 1;

    // Zero output only if using atomicAdd accumulation across K-blocks
    if (kb > 1)
        hipMemsetAsync(d_C_fp32, 0, N * sizeof(float), static_cast<hipStream_t>(stream));

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(gemv_fused_fp32_int8_vnni_kernel_t<TN, CPT>),
        dim3(grid_n, kb), dim3(TN / CPT), 0, static_cast<hipStream_t>(stream),
        d_A_fp32, d_B_int8_vnni, d_C_fp32, d_scales_B, d_bias, N, K, kb);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "[rocmGemv_fused_fp32_int8_vnni] Kernel launch failed: %s (N=%d K=%d kb=%d)\n",
                hipGetErrorString(err), N, K, kb);
        return false;
    }

    // Apply bias separately if provided and using K-splitting.
    // With kb==1, bias could be fused but we keep the separate path for simplicity.
    if (d_bias)
    {
        // Reuse existing biasAdd from ROCmQuantisedGemmKernel_CK.hip
        extern bool rocmQuantGemm_biasAdd(
            float *d_output, const float *d_bias,
            int M, int N, int rocm_device_id, void *stream);

        if (!rocmQuantGemm_biasAdd(d_C_fp32, d_bias, 1, N, device_id, stream))
        {
            fprintf(stderr, "[rocmGemv_fused_fp32_int8_vnni] biasAdd failed (N=%d)\n", N);
            return false;
        }
    }

    return true;
}


// =============================================================================
// Fused Scatter FP32→FP32 GEMV: Quantize + GEMV → partials, then reduce+scale
// =============================================================================
//
// 2-kernel pipeline replacing the 3-kernel pipeline:
//   OLD: quantizeActivations → rocmGemv_int8_int8_int32_vnni → applyScaling
//   NEW: fused_scatter (quant+GEMV→partials) → reduce+scale
//
// Eliminates:
//   - Separate quant kernel launch (~10 μs)
//   - hipMemsetAsync (~3 μs)
//   - All atomicAdd contention
//   - One kernel launch overhead (~5-8 μs)
//
// When KB=1 (enough N-blocks for full occupancy), falls back to the
// single-kernel fused path for maximum launch elimination.
//
bool rocmGemv_fused_scatter_fp32_int8_vnni(
    const float* d_A_fp32,           // [K] FP32 activations
    const int8_t* d_B_int8_vnni,     // [K/4 × N × 4] VNNI-packed weights
    float* d_C_fp32,                 // [N] FP32 output
    const float* d_scales_B,         // [N] per-column weight scales
    const float* d_bias,             // [N] optional bias (nullable)
    float* d_partial_buf,            // [KB_MAX × N] pre-allocated partial buffer
    int N, int K,
    float alpha, float beta,
    const float* d_C_existing,       // [N] used when beta != 0
    int device_id, void* stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_id);

    if ((K % 4) != 0)
    {
        fprintf(stderr, "[rocmGemv_fused_scatter] K must be multiple of 4 (N=%d K=%d)\n", N, K);
        return false;
    }

    if ((N % 4) != 0)
    {
        fprintf(stderr, "[rocmGemv_fused_scatter] N must be multiple of 4 (N=%d K=%d)\n", N, K);
        return false;
    }

    constexpr int TN = GEMV_INT8_VNNI_GRID_KPAR_TILE_N;  // 128
    constexpr int CPT = GEMV_INT8_VNNI_GRID_KPAR_CPT;    // 2
    constexpr int NUM_CUS = 60;  // MI50/MI60

    const int grid_n = (N + TN - 1) / TN;
    const int k_groups = K / 4;

    // KB heuristic: same occupancy target as grid_kpar (~8 waves/CU),
    // but min KB=1 since scatter has no atomicAdd penalty for KB=1.
    constexpr int TARGET_WAVES_PER_CU = 8;
    constexpr int MIN_KGROUPS_PER_WAVE = 16;

    int target_blocks = TARGET_WAVES_PER_CU * NUM_CUS;  // 480
    int kb_raw = max(1, (target_blocks + grid_n - 1) / grid_n);

    int kb_max = max(1, k_groups / MIN_KGROUPS_PER_WAVE);
    kb_raw = min(kb_raw, kb_max);

    // Round to nearest factor of k_groups (avoid uneven work distribution)
    int kb = kb_raw;
    if (k_groups % kb_raw != 0)
    {
        for (int d = 1; d < kb_raw; ++d)
        {
            if (kb_raw - d >= 1 && k_groups % (kb_raw - d) == 0) { kb = kb_raw - d; break; }
            if (kb_raw + d <= kb_max && k_groups % (kb_raw + d) == 0) { kb = kb_raw + d; break; }
        }
    }

    if (kb <= 1)
    {
        // Single-kernel path: enough N-blocks for occupancy with KB=1.
        // Use the existing fused kernel directly (no partial buffer needed).
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(gemv_fused_fp32_int8_vnni_kernel_t<TN, CPT>),
            dim3(grid_n, 1), dim3(TN / CPT), 0, static_cast<hipStream_t>(stream),
            d_A_fp32, d_B_int8_vnni, d_C_fp32, d_scales_B, nullptr, N, K, 1);

        // Apply bias + alpha/beta if needed (the single-kernel path doesn't handle these)
        if (d_bias || beta != 0.0f)
        {
            // The fused kernel wrote: d_C_fp32[n] = sum * scale_B[n]
            // We need: result = alpha * sum * scale_B + beta * C_existing + bias
            // Since alpha and beta are not applied by the fused kernel for kb=1 path,
            // handle them via a quick fixup if non-trivial
            if (alpha != 1.0f || beta != 0.0f)
            {
                fprintf(stderr, "[rocmGemv_fused_scatter] alpha/beta not yet supported in kb=1 path\n");
                return false;
            }
            if (d_bias)
            {
                extern bool rocmQuantGemm_biasAdd(
                    float *d_output, const float *d_bias,
                    int M, int N, int rocm_device_id, void *stream);
                if (!rocmQuantGemm_biasAdd(d_C_fp32, d_bias, 1, N, device_id, stream))
                    return false;
            }
        }
    }
    else
    {
        // Two-kernel path: fused scatter → reduce+scale
        // Kernel 1: fused quantize + INT8 GEMV → FP32 partials
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(gemv_fused_scatter_int8_vnni_kernel_t<TN, CPT>),
            dim3(grid_n, kb), dim3(TN / CPT), 0, static_cast<hipStream_t>(stream),
            d_A_fp32, d_B_int8_vnni, d_partial_buf, N, K, kb);

        // Kernel 2: reduce KB partials + apply scale_B + bias
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(gemv_reduce_scale_kernel_t<TN, CPT>),
            dim3(grid_n), dim3(TN / CPT), 0, static_cast<hipStream_t>(stream),
            d_partial_buf, d_C_fp32, d_scales_B, d_bias, N, kb,
            alpha, beta, d_C_existing);
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "[rocmGemv_fused_scatter] Kernel launch failed: %s (N=%d K=%d kb=%d)\n",
                hipGetErrorString(err), N, K, kb);
        return false;
    }

    return true;
}


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
// Treat each 4-byte pack as one int32 element in matrix A[kg, n] and transpose
// to B[n, kg] using a tiled shared-memory transpose to keep both reads and
// writes coalesced.
//
// Cost: ~65-200μs for 18944×3584 (Gate/Up shape). Amortized over prefill.
//
__global__ void
repack_vnni_to_rowmajor_kernel(
    const int8_t* __restrict__ d_B_vnni,     // [K/4][N][4] input
    int8_t* __restrict__ d_B_rowmajor,       // [N×K] output
    int N, int K)
{
    constexpr int TILE_DIM = 32;
    constexpr int BLOCK_ROWS = 8;

    const int k_groups = K / 4; // number of int32 packs per output row

    const int x_in = blockIdx.x * TILE_DIM + threadIdx.x; // n index in source A[kg, n]
    const int y_in = blockIdx.y * TILE_DIM + threadIdx.y; // kg index in source A[kg, n]

    const int32_t* __restrict__ src32 = reinterpret_cast<const int32_t*>(d_B_vnni);
    int32_t* __restrict__ dst32 = reinterpret_cast<int32_t*>(d_B_rowmajor);

    __shared__ int32_t tile[TILE_DIM][TILE_DIM + 1];

    // Load tile from source (A = [k_groups, N], row-major in n)
    #pragma unroll
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    {
        const int y = y_in + j;
        if (x_in < N && y < k_groups)
        {
            tile[threadIdx.y + j][threadIdx.x] = src32[static_cast<int64_t>(y) * N + x_in];
        }
    }

    __syncthreads();

    // Store transposed tile to destination (B = [N, k_groups], row-major in kg)
    const int x_out = blockIdx.y * TILE_DIM + threadIdx.x; // kg
    const int y_out = blockIdx.x * TILE_DIM + threadIdx.y; // n

    #pragma unroll
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    {
        const int y = y_out + j;
        if (x_out < k_groups && y < N)
        {
            dst32[static_cast<int64_t>(y) * k_groups + x_out] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}


bool rocmGemv_repackVNNI_to_rowmajor(
    const int8_t* d_B_vnni,
    int8_t* d_B_rowmajor,
    int N, int K,
    int device_id, void* stream)
{
    if ((K % 4) != 0)
    {
        fprintf(stderr, "[rocmGemv_repackVNNI_to_rowmajor] K must be multiple of 4 (K=%d)\n", K);
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_id);

    const int k_groups = K / 4;
    constexpr int TILE_DIM = 32;
    constexpr int BLOCK_ROWS = 8;

    const int grid_x = (N + TILE_DIM - 1) / TILE_DIM;
    const int grid_y = (k_groups + TILE_DIM - 1) / TILE_DIM;

    hipLaunchKernelGGL(
        repack_vnni_to_rowmajor_kernel,
        dim3(grid_x, grid_y), dim3(TILE_DIM, BLOCK_ROWS), 0, static_cast<hipStream_t>(stream),
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

/**
 * @brief Entrypoint for INT8 VNNI prefill GEMM.
 *
 * This function dispatches to tiled INT8-x-INT8 GEMM kernels for M>1 prefill.
 * Returning false instructs the caller to keep using the CK fallback path.
 *
 * Junior-dev note:
 * - The decode path (M=1) already has specialized GEMV kernels.
 * - Prefill path (M>1) uses tiled INT8 VNNI GEMM kernels dispatched here.
 */
enum PrefillInt8TileVariant : int
{
    PREFILL_TILE_16x16 = 0,
    PREFILL_TILE_32x8 = 1,
    PREFILL_TILE_8x32 = 2,
    PREFILL_TILE_8x8 = 3,
};

static const char *prefillTileVariantName(int variant)
{
    switch (variant)
    {
    case PREFILL_TILE_16x16:
        return "16x16";
    case PREFILL_TILE_32x8:
        return "32x8";
    case PREFILL_TILE_8x32:
        return "8x32";
    case PREFILL_TILE_8x8:
        return "8x8";
    default:
        return "unknown";
    }
}

static int select_prefill_int8_tile_variant_auto(int M, int N, int K, bool grid_kpar)
{
    (void)K;
    if (M <= 12 && N <= 512)
    {
        return PREFILL_TILE_8x8;
    }

    if (N >= 4 * M && N >= 1024)
    {
        return PREFILL_TILE_32x8;
    }

    if (M >= 4 * N && M >= 128)
    {
        return PREFILL_TILE_8x32;
    }

    if (grid_kpar && N >= 2 * M && N >= 2048)
    {
        return PREFILL_TILE_32x8;
    }

    return PREFILL_TILE_16x16;
}

static int resolve_prefill_int8_tile_variant(int forced_variant, int M, int N, int K, bool grid_kpar)
{
    if (forced_variant >= PREFILL_TILE_16x16 && forced_variant <= PREFILL_TILE_8x8)
    {
        return forced_variant;
    }
    return select_prefill_int8_tile_variant_auto(M, N, K, grid_kpar);
}

bool rocmGemm_int8_int8_int32_vnni_prefill(
    const int8_t* d_A_int8,
    const int8_t* d_B_int8_vnni,
    int32_t* d_C_int32,
    int M, int N, int K,
    int tile_variant,
    int cpt,
    int device_id, void* stream)
{
    if (!d_A_int8 || !d_B_int8_vnni || !d_C_int32 || M <= 0 || N <= 0 || K <= 0)
    {
        fprintf(stderr, "[rocmGemm_int8_int8_int32_vnni_prefill] Invalid arguments (M=%d N=%d K=%d)\n", M, N, K);
        return false;
    }

    if ((K % 4) != 0)
    {
        fprintf(stderr, "[rocmGemm_int8_int8_int32_vnni_prefill] K must be multiple of 4 (M=%d N=%d K=%d)\n", M, N, K);
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_id);

    const int variant = resolve_prefill_int8_tile_variant(tile_variant, M, N, K, false);
    const int cpt_v = (cpt == 2 || cpt == 4) ? cpt : 1;

    if (variant == PREFILL_TILE_32x8)
    {
        constexpr int BX = 32;
        constexpr int BY = 8;
        const dim3 block(BX, BY, 1);
        if (cpt_v == 4)
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_kernel_t<BX, BY, 4>),
                               dim3(static_cast<unsigned>((N + BX * 4 - 1) / (BX * 4)), static_cast<unsigned>((M + BY - 1) / BY), 1),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K);
        }
        else if (cpt_v == 2)
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_kernel_t<BX, BY, 2>),
                               dim3(static_cast<unsigned>((N + BX * 2 - 1) / (BX * 2)), static_cast<unsigned>((M + BY - 1) / BY), 1),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K);
        }
        else
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_kernel_t<BX, BY, 1>),
                               dim3(static_cast<unsigned>((N + BX - 1) / BX), static_cast<unsigned>((M + BY - 1) / BY), 1),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K);
        }
    }
    else if (variant == PREFILL_TILE_8x32)
    {
        constexpr int BX = 8;
        constexpr int BY = 32;
        const dim3 block(BX, BY, 1);
        if (cpt_v == 4)
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_kernel_t<BX, BY, 4>),
                               dim3(static_cast<unsigned>((N + BX * 4 - 1) / (BX * 4)), static_cast<unsigned>((M + BY - 1) / BY), 1),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K);
        }
        else if (cpt_v == 2)
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_kernel_t<BX, BY, 2>),
                               dim3(static_cast<unsigned>((N + BX * 2 - 1) / (BX * 2)), static_cast<unsigned>((M + BY - 1) / BY), 1),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K);
        }
        else
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_kernel_t<BX, BY, 1>),
                               dim3(static_cast<unsigned>((N + BX - 1) / BX), static_cast<unsigned>((M + BY - 1) / BY), 1),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K);
        }
    }
    else if (variant == PREFILL_TILE_8x8)
    {
        constexpr int BX = 8;
        constexpr int BY = 8;
        const dim3 block(BX, BY, 1);
        if (cpt_v == 4)
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_kernel_t<BX, BY, 4>),
                               dim3(static_cast<unsigned>((N + BX * 4 - 1) / (BX * 4)), static_cast<unsigned>((M + BY - 1) / BY), 1),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K);
        }
        else if (cpt_v == 2)
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_kernel_t<BX, BY, 2>),
                               dim3(static_cast<unsigned>((N + BX * 2 - 1) / (BX * 2)), static_cast<unsigned>((M + BY - 1) / BY), 1),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K);
        }
        else
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_kernel_t<BX, BY, 1>),
                               dim3(static_cast<unsigned>((N + BX - 1) / BX), static_cast<unsigned>((M + BY - 1) / BY), 1),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K);
        }
    }
    else
    {
        constexpr int BX = 16;
        constexpr int BY = 16;
        const dim3 block(BX, BY, 1);
        if (cpt_v == 4)
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_kernel_t<BX, BY, 4>),
                               dim3(static_cast<unsigned>((N + BX * 4 - 1) / (BX * 4)), static_cast<unsigned>((M + BY - 1) / BY), 1),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K);
        }
        else if (cpt_v == 2)
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_kernel_t<BX, BY, 2>),
                               dim3(static_cast<unsigned>((N + BX * 2 - 1) / (BX * 2)), static_cast<unsigned>((M + BY - 1) / BY), 1),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K);
        }
        else
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_kernel_t<BX, BY, 1>),
                               dim3(static_cast<unsigned>((N + BX - 1) / BX), static_cast<unsigned>((M + BY - 1) / BY), 1),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K);
        }
    }

    const hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "[rocmGemm_int8_int8_int32_vnni_prefill] Kernel launch failed: %s (M=%d N=%d K=%d)\n",
                hipGetErrorString(err), M, N, K);
        return false;
    }

    return true;
}

/**
 * @brief INT8 VNNI prefill grid-kpar split-K variant.
 *
 * This variant uses grid.z to split K reduction into slices with INT32
 * atomic accumulation. It is opt-in from higher-level dispatch.
 */
bool rocmGemm_int8_int8_int32_vnni_prefill_grid_kpar(
    const int8_t* d_A_int8,
    const int8_t* d_B_int8_vnni,
    int32_t* d_C_int32,
    int M, int N, int K,
    int split_k_slices,
    int tile_variant,
    int cpt,
    int device_id, void* stream)
{
    if (!d_A_int8 || !d_B_int8_vnni || !d_C_int32 || M <= 0 || N <= 0 || K <= 0)
    {
        fprintf(stderr, "[rocmGemm_int8_int8_int32_vnni_prefill_grid_kpar] Invalid arguments (M=%d N=%d K=%d)\n", M, N, K);
        return false;
    }

    if ((K % 4) != 0)
    {
        fprintf(stderr, "[rocmGemm_int8_int8_int32_vnni_prefill_grid_kpar] K must be multiple of 4 (M=%d N=%d K=%d)\n", M, N, K);
        return false;
    }

    const int total_k_groups = K / 4;
    const int slices = max(1, min(split_k_slices, total_k_groups));
    const int k_groups_per_slice = (total_k_groups + slices - 1) / slices;

    llaminar2::HipDeviceGuard::setDevice(device_id);

    const size_t c_bytes = static_cast<size_t>(M) * static_cast<size_t>(N) * sizeof(int32_t);
    hipError_t memset_err = hipMemsetAsync(d_C_int32, 0, c_bytes, static_cast<hipStream_t>(stream));
    if (memset_err != hipSuccess)
    {
        fprintf(stderr, "[rocmGemm_int8_int8_int32_vnni_prefill_grid_kpar] hipMemsetAsync failed: %s\n", hipGetErrorString(memset_err));
        return false;
    }

    const int variant = resolve_prefill_int8_tile_variant(tile_variant, M, N, K, true);
    const int cpt_v = (cpt == 2 || cpt == 4) ? cpt : 1;

    if (variant == PREFILL_TILE_32x8)
    {
        constexpr int BX = 32;
        constexpr int BY = 8;
        const dim3 block(BX, BY, 1);
        if (cpt_v == 4)
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_grid_kpar_kernel_t<BX, BY, 4>),
                               dim3(static_cast<unsigned>((N + BX * 4 - 1) / (BX * 4)), static_cast<unsigned>((M + BY - 1) / BY), static_cast<unsigned>(slices)),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K, k_groups_per_slice);
        }
        else if (cpt_v == 2)
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_grid_kpar_kernel_t<BX, BY, 2>),
                               dim3(static_cast<unsigned>((N + BX * 2 - 1) / (BX * 2)), static_cast<unsigned>((M + BY - 1) / BY), static_cast<unsigned>(slices)),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K, k_groups_per_slice);
        }
        else
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_grid_kpar_kernel_t<BX, BY, 1>),
                               dim3(static_cast<unsigned>((N + BX - 1) / BX), static_cast<unsigned>((M + BY - 1) / BY), static_cast<unsigned>(slices)),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K, k_groups_per_slice);
        }
    }
    else if (variant == PREFILL_TILE_8x32)
    {
        constexpr int BX = 8;
        constexpr int BY = 32;
        const dim3 block(BX, BY, 1);
        if (cpt_v == 4)
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_grid_kpar_kernel_t<BX, BY, 4>),
                               dim3(static_cast<unsigned>((N + BX * 4 - 1) / (BX * 4)), static_cast<unsigned>((M + BY - 1) / BY), static_cast<unsigned>(slices)),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K, k_groups_per_slice);
        }
        else if (cpt_v == 2)
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_grid_kpar_kernel_t<BX, BY, 2>),
                               dim3(static_cast<unsigned>((N + BX * 2 - 1) / (BX * 2)), static_cast<unsigned>((M + BY - 1) / BY), static_cast<unsigned>(slices)),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K, k_groups_per_slice);
        }
        else
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_grid_kpar_kernel_t<BX, BY, 1>),
                               dim3(static_cast<unsigned>((N + BX - 1) / BX), static_cast<unsigned>((M + BY - 1) / BY), static_cast<unsigned>(slices)),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K, k_groups_per_slice);
        }
    }
    else if (variant == PREFILL_TILE_8x8)
    {
        constexpr int BX = 8;
        constexpr int BY = 8;
        const dim3 block(BX, BY, 1);
        if (cpt_v == 4)
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_grid_kpar_kernel_t<BX, BY, 4>),
                               dim3(static_cast<unsigned>((N + BX * 4 - 1) / (BX * 4)), static_cast<unsigned>((M + BY - 1) / BY), static_cast<unsigned>(slices)),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K, k_groups_per_slice);
        }
        else if (cpt_v == 2)
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_grid_kpar_kernel_t<BX, BY, 2>),
                               dim3(static_cast<unsigned>((N + BX * 2 - 1) / (BX * 2)), static_cast<unsigned>((M + BY - 1) / BY), static_cast<unsigned>(slices)),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K, k_groups_per_slice);
        }
        else
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_grid_kpar_kernel_t<BX, BY, 1>),
                               dim3(static_cast<unsigned>((N + BX - 1) / BX), static_cast<unsigned>((M + BY - 1) / BY), static_cast<unsigned>(slices)),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K, k_groups_per_slice);
        }
    }
    else
    {
        constexpr int BX = 16;
        constexpr int BY = 16;
        const dim3 block(BX, BY, 1);
        if (cpt_v == 4)
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_grid_kpar_kernel_t<BX, BY, 4>),
                               dim3(static_cast<unsigned>((N + BX * 4 - 1) / (BX * 4)), static_cast<unsigned>((M + BY - 1) / BY), static_cast<unsigned>(slices)),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K, k_groups_per_slice);
        }
        else if (cpt_v == 2)
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_grid_kpar_kernel_t<BX, BY, 2>),
                               dim3(static_cast<unsigned>((N + BX * 2 - 1) / (BX * 2)), static_cast<unsigned>((M + BY - 1) / BY), static_cast<unsigned>(slices)),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K, k_groups_per_slice);
        }
        else
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_grid_kpar_kernel_t<BX, BY, 1>),
                               dim3(static_cast<unsigned>((N + BX - 1) / BX), static_cast<unsigned>((M + BY - 1) / BY), static_cast<unsigned>(slices)),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K, k_groups_per_slice);
        }
    }

    const hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "[rocmGemm_int8_int8_int32_vnni_prefill_grid_kpar] Kernel launch failed: %s (M=%d N=%d K=%d slices=%d)\n",
                hipGetErrorString(err), M, N, K, slices);
        return false;
    }

    return true;
}

// ============================================================================
// Native-VNNI GEMV dispatch: lossless Q4/IQ4 decode with FP16 per-block scales
// ============================================================================
bool rocmGemv_native_vnni_fp32(
    const int8_t* d_A_int8,
    const uint8_t* d_payload,
    const void* d_block_scales,  // __half* (FP16)
    float* d_C_fp32,
    const float* d_scale_A,
    int N, int K,
    uint8_t codebook_id,
    int device_id, void* stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_id);

    if ((K % 32) != 0)
    {
        fprintf(stderr, "[rocmGemv_native_vnni_fp32] K must be multiple of 32 (N=%d K=%d)\n", N, K);
        return false;
    }

    const int blocks_per_row = K / 32;
    const bool is_iq4 = (codebook_id == CODEBOOK_IQ4);

    // Ensure IQ4 LUT is loaded (when needed)
    // Note: for native-VNNI we use the k_iq4_codebook_i8 device constant directly
    // which is statically initialized — no runtime init needed.

    constexpr int TN = 128;   // TILE_N: 128 output columns per block
    constexpr int CPT = 2;    // Columns per thread: 64 threads = 1 full wavefront (wavefront64)

    // K-partition heuristic: split K across grid.y for occupancy
    int kb;
    if (blocks_per_row > 256)
        kb = min(20, max(4, blocks_per_row / 16));
    else
        kb = min(12, max(2, blocks_per_row / 8));

    // Zero output for atomicAdd accumulation
    hipMemsetAsync(d_C_fp32, 0, static_cast<size_t>(N) * sizeof(float), static_cast<hipStream_t>(stream));

    const int logical_tn = TN * CPT;
    const int grid_n = (N + logical_tn - 1) / logical_tn;
    dim3 grid(grid_n, kb);
    dim3 block(TN, 1, 1);

    const uint16_t* d_scales_u16 = static_cast<const uint16_t*>(d_block_scales);

    if (is_iq4)
    {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(gemv_native_vnni_kernel_t<TN * CPT, CPT, true>),
            grid, block, 0, static_cast<hipStream_t>(stream),
            d_A_int8, d_payload, d_scales_u16, d_C_fp32, d_scale_A,
            N, K, kb);
    }
    else
    {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(gemv_native_vnni_kernel_t<TN * CPT, CPT, false>),
            grid, block, 0, static_cast<hipStream_t>(stream),
            d_A_int8, d_payload, d_scales_u16, d_C_fp32, d_scale_A,
            N, K, kb);
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "[rocmGemv_native_vnni_fp32] Kernel launch failed: %s (N=%d K=%d kb=%d codebook=%u)\n",
                hipGetErrorString(err), N, K, kb, static_cast<unsigned>(codebook_id));
        return false;
    }

    return true;
}

} // extern "C"
