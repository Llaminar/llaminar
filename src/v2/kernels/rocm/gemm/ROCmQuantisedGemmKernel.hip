/**
 * @file ROCmQuantisedGemmKernel.hip
 * @brief Common HIP kernels and utilities for INT8 quantized GEMM on AMD ROCm GPUs
 *
 * This file contains shared infrastructure used by all quantized GEMM backends:
 *
 *   GPU Kernels:
 *     - applyScales_kernel             : INT32 → FP32 with per-row/col scales
 *     - applyScaling_full_kernel       : Full epilogue (alpha, beta, bias)
 *     - applyScaling_m1_vec4_kernel    : M=1 decode-optimized scaling
 *     - fillConstantFloat_kernel       : Fill buffer with constant value
 *     - biasAdd_kernel                 : In-place bias addition
 *
 *   Dispatch Functions (extern "C"):
 *     - rocmQuantGemm_setDevice()              : HIP device selection
 *     - rocmQuantGemm_uploadWeights()          : Weight upload H2D
 *     - rocmQuantGemm_ensureWorkBuffers()      : Work buffer management
 *     - rocmQuantGemm_applyScaling()           : INT32 → FP32 scaling epilogue
 *     - rocmQuantGemm_biasAdd()                : In-place bias addition
 *     - rocmQuantGemm_alloc*(), copy*(), free* : Memory management utilities
 *
 * Backend-specific GEMM implementations live in separate files:
 *   - ROCmQuantisedGemmKernel_INT8_VNNI.hip : INT8 VNNI prefill kernels (gfx906)
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cfloat>
#include <cmath>
#include <chrono>

#include "../../../backends/rocm/HipDeviceGuard.h"
#include "../../../utils/DebugEnv.h"

// ============================================================================
// Apply Scales Kernel (for two-kernel approach)
// ============================================================================
//
// Applies per-row and per-column scales to INT32 GEMM output:
//   E_fp32[m,n] = C_int32[m,n] * scale_A[m] * scale_B[n]
//
// This is the second kernel in the two-kernel approach, which is more reliable
// than CK's fused D-tensor handling on gfx906.
//
// Grid: ((M*N + 255) / 256, 1, 1) blocks
// Block: (256, 1, 1) threads
//
__global__ void applyScales_kernel(
    const int32_t* __restrict__ d_C_int32,  // [M×N] INT32 GEMM output
    float* __restrict__ d_E_fp32,           // [M×N] FP32 scaled output
    const float* __restrict__ d_scale_A,    // [M] per-row scales
    const float* __restrict__ d_scale_B,    // [N] per-column scales
    int M, int N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = M * N;
    
    if (idx >= total) return;
    
    const int m = idx / N;
    const int n = idx % N;
    
    d_E_fp32[idx] = static_cast<float>(d_C_int32[idx]) * d_scale_A[m] * d_scale_B[n];
}

// ============================================================================
// Apply Scaling Kernel with Full Epilogue (alpha, beta, bias)
// ============================================================================
//
// Full-featured scaling epilogue matching CUDA's apply_scaling_kernel:
//   result = alpha * C_int32 * scale_A[m] * scale_B[n]
//   if (beta != 0) result += beta * C_existing[idx]
//   if (bias)      result += bias[n]
//
// Grid: ((N + 15) / 16, (M + 15) / 16, 1) blocks
// Block: (16, 16, 1) threads
//
__global__ void applyScaling_full_kernel(
    const int32_t* __restrict__ d_C_int32,  // [M×N] INT32 GEMM output
    float* __restrict__ d_C_fp32,           // [M×N] FP32 output
    const float* __restrict__ d_scale_A,    // [M] per-row scales
    const float* __restrict__ d_scale_B,    // [N] per-column scales
    int M, int N,
    float alpha, float beta,
    const float* __restrict__ d_C_existing, // [M×N] existing output (for beta != 0)
    const float* __restrict__ d_bias)       // [N] optional bias, broadcasted across rows
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N)
    {
        const int idx = row * N + col;
        const int32_t val_int32 = d_C_int32[idx];
        const float scale_combined = d_scale_A[row] * d_scale_B[col];
        float result = alpha * static_cast<float>(val_int32) * scale_combined;

        if (beta != 0.0f && d_C_existing != nullptr)
        {
            result += beta * d_C_existing[idx];
        }

        if (d_bias != nullptr)
        {
            result += d_bias[col];
        }

        d_C_fp32[idx] = result;
    }
}

__global__ void applyFp32Epilogue_kernel(
    float* __restrict__ d_output,
    const float* __restrict__ d_bias,
    int M, int N,
    float alpha)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = M * N;
    if (idx >= total) return;

    const int col = idx % N;
    float result = alpha * d_output[idx];
    if (d_bias != nullptr)
    {
        result += d_bias[col];
    }
    d_output[idx] = result;
}

__global__ void applyFp32CombineEpilogue_kernel(
    const float* d_computed,
    float* d_output,
    const float* d_existing,
    const float* __restrict__ d_bias,
    int M, int N,
    float alpha, float beta)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = M * N;
    if (idx >= total) return;

    const int col = idx % N;
    float result = alpha * d_computed[idx];
    if (beta != 0.0f && d_existing != nullptr)
    {
        result += beta * d_existing[idx];
    }
    if (d_bias != nullptr)
    {
        result += d_bias[col];
    }
    d_output[idx] = result;
}
// ============================================================================
// Vectorized Scaling Kernel with 1D Flat Grid (M > 1 fast path)
// ============================================================================
//
// Replaces the 16×16 applyScaling_full_kernel which creates catastrophic
// block counts for large N (LM_Head N=151936: 360K blocks, 254ms).
//
// Uses a 1D flat grid-stride loop over the entire M×N output:
//   - Sequential memory access across the full buffer (HBM-friendly)
//   - Vec4 loads/stores for 128-bit transactions
//   - Fixed ~480 blocks regardless of shape (vs 360K for LM_Head)
//
// N must be divisible by 4 for vec4 correctness (all Llaminar shapes are).
// When N % 4 != 0, falls back to the legacy applyScaling_full_kernel.
//
// Grid:  (min((M*N/4 + BLOCK_X-1)/BLOCK_X, MAX_BLOCKS))
// Block: (BLOCK_X) where BLOCK_X = 256
//
__global__ void applyScaling_vec4_flat_kernel(
    const int32_t* __restrict__ d_C_int32,  // [M×N] INT32 GEMM output
    float* __restrict__ d_C_fp32,           // [M×N] FP32 output
    const float* __restrict__ d_scale_A,    // [M] per-row scales
    const float* __restrict__ d_scale_B,    // [N] per-column scales
    int M, int N, int total_vec4,           // total_vec4 = M * N / 4
    float alpha, float beta,
    const float* __restrict__ d_C_existing, // [M×N] existing output (for beta != 0)
    const float* __restrict__ d_bias)       // [N] optional bias, broadcasted across rows
{
    const int N4 = N / 4;  // number of vec4 chunks per row

    for (int vi = blockIdx.x * blockDim.x + threadIdx.x;
         vi < total_vec4;
         vi += gridDim.x * blockDim.x)
    {
        const int row = vi / N4;
        const int col4 = (vi - row * N4) * 4;  // vi % N4 * 4
        const int idx = row * N + col4;

        const float row_scale = alpha * d_scale_A[row];
        const int4 vals = *reinterpret_cast<const int4*>(d_C_int32 + idx);
        const float4 sb = *reinterpret_cast<const float4*>(d_scale_B + col4);

        float4 out;
        out.x = static_cast<float>(vals.x) * row_scale * sb.x;
        out.y = static_cast<float>(vals.y) * row_scale * sb.y;
        out.z = static_cast<float>(vals.z) * row_scale * sb.z;
        out.w = static_cast<float>(vals.w) * row_scale * sb.w;

        if (beta != 0.0f && d_C_existing != nullptr)
        {
            const float4 ex = *reinterpret_cast<const float4*>(d_C_existing + idx);
            out.x += beta * ex.x;
            out.y += beta * ex.y;
            out.z += beta * ex.z;
            out.w += beta * ex.w;
        }

        if (d_bias != nullptr)
        {
            const float4 b = *reinterpret_cast<const float4*>(d_bias + col4);
            out.x += b.x;
            out.y += b.y;
            out.z += b.z;
            out.w += b.w;
        }

        *reinterpret_cast<float4*>(d_C_fp32 + idx) = out;
    }
}

// Decode-specialized scaling path (M=1, beta=0, no bias):
//   C_fp32[n] = alpha * C_int32[n] * scale_A[0] * scale_B[n]
// Uses 1D grid and vectorized x4 lanes to avoid the 16x16 underutilization
// of applyScaling_full_kernel when only a single row is active.
__global__ void applyScaling_m1_vec4_kernel(
    const int32_t* __restrict__ d_C_int32,
    float* __restrict__ d_C_fp32,
    const float* __restrict__ d_scale_A,
    const float* __restrict__ d_scale_B,
    int N,
    float alpha)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int n4 = tid * 4;

    const float row_scale = alpha * d_scale_A[0];

    if (n4 + 3 < N)
    {
        const int4 v = *reinterpret_cast<const int4*>(d_C_int32 + n4);
        const float4 sb = *reinterpret_cast<const float4*>(d_scale_B + n4);

        float4 out;
        out.x = static_cast<float>(v.x) * row_scale * sb.x;
        out.y = static_cast<float>(v.y) * row_scale * sb.y;
        out.z = static_cast<float>(v.z) * row_scale * sb.z;
        out.w = static_cast<float>(v.w) * row_scale * sb.w;

        *reinterpret_cast<float4*>(d_C_fp32 + n4) = out;
        return;
    }

    for (int n = n4; n < N && n < n4 + 4; ++n)
    {
        d_C_fp32[n] = static_cast<float>(d_C_int32[n]) * row_scale * d_scale_B[n];
    }
}

__global__ void fillConstantFloat_kernel(float* __restrict__ dst, int count, float value)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count)
    {
        dst[idx] = value;
    }
}

// ============================================================================
// BLOCKWISE ACTIVATION QUANTIZATION KERNEL (C++ linkage — template)
// ============================================================================

/**
 * @brief Per-block symmetric quantization: FP32 → INT8 with per-block scale
 *
 * Each row is divided into fixed-size blocks of BLOCK_SIZE elements.
 * Each block gets its own independent scale, providing finer quantization
 * granularity than whole-row quantization.
 *
 * Algorithm (per block):
 *   1. Find max_abs = max(|x_i|) within block
 *   2. scale = max_abs / 127.0 (or 1.0 if max_abs == 0)
 *   3. int8_i = round(x_i / scale) clamped to [-127, 127]
 *
 * Grid: (M × blocks_per_row, 1, 1)
 * Block: 64 threads (wave64 on gfx906, 32 do useful work)
 *
 * Scale output layout: d_scales[row * blocks_per_row + block_idx]
 */
template <int BLOCK_SIZE>
__global__ void quantizeActivationsQ8_blockwise_kernel(
    const float *__restrict__ d_A_fp32,
    int8_t *__restrict__ d_A_int8,
    float *__restrict__ d_scales,
    int M, int K, int blocks_per_row)
{
    const int linear_block = blockIdx.x;
    const int row = linear_block / blocks_per_row;
    const int block_idx = linear_block % blocks_per_row;

    if (row >= M)
        return;

    const int tid = threadIdx.x;
    const int k_start = block_idx * BLOCK_SIZE;
    const int k_end = min(k_start + BLOCK_SIZE, K);
    const int block_len = k_end - k_start;

    const float *src = d_A_fp32 + static_cast<size_t>(row) * K + k_start;
    int8_t *dst = d_A_int8 + static_cast<size_t>(row) * K + k_start;

    __shared__ float s_absmax;

    // Step 1: Find max absolute value in this block
    float local_max = 0.0f;
    if (tid < block_len)
    {
        local_max = fabsf(src[tid]);
    }

    // Wavefront reduction (wave64 on gfx906)
    #pragma unroll
    for (int offset = 32; offset > 0; offset >>= 1)
    {
        local_max = fmaxf(local_max, __shfl_down(local_max, offset));
    }

    if (tid == 0)
    {
        s_absmax = local_max;
    }
    __syncthreads();

    float max_abs = s_absmax;
    float scale = (max_abs > 0.0f) ? (max_abs / 127.0f) : 1.0f;
    float inv_scale = 1.0f / scale;

    // Step 2: Write scale for this block
    if (tid == 0)
    {
        d_scales[static_cast<size_t>(row) * blocks_per_row + block_idx] = scale;
    }

    // Step 3: Quantize elements in this block
    if (tid < block_len)
    {
        int q = static_cast<int>(rintf(src[tid] * inv_scale));
        q = (q < -127) ? -127 : ((q > 127) ? 127 : q);
        dst[tid] = static_cast<int8_t>(q);
    }
}

// ============================================================================
// BLOCKWISE ACTIVATION QUANTIZATION — OPTIMIZED VARIANTS
// ============================================================================

/**
 * Variant V2: Multi-block per workgroup, vectorized.
 *
 * Each workgroup processes BLOCKS_PER_WG consecutive 32-element blocks.
 * Uses 256 threads / wave64 = 4 waves for good occupancy.
 * Within each 32-element block, 32 threads each handle 1 element.
 * The remaining threads in the 256-thread WG handle other blocks in parallel.
 *
 * Advantages over V1 (original):
 *   - 8× fewer workgroup launches (BLOCKS_PER_WG=8)
 *   - Full thread utilization (256 threads, 8 blocks × 32 threads each)
 *   - No shared memory needed (pure wave shuffle within sub-groups)
 *
 * Grid:  (ceil(total_blocks / BLOCKS_PER_WG), 1, 1)
 * Block: (256, 1, 1) = 4 wave64s
 */
template <int BLOCKS_PER_WG = 8>
__global__ __launch_bounds__(256, 4)
void quantizeActivationsQ8_blockwise_v2_kernel(
    const float *__restrict__ d_A_fp32,
    int8_t *__restrict__ d_A_int8,
    float *__restrict__ d_scales,
    int M, int K, int blocks_per_row, int total_blocks)
{
    constexpr int QUANT_BLOCK = 32;  // elements per quantization block
    constexpr int WG_SIZE = 256;

    const int tid = threadIdx.x;
    // Each thread is assigned to one of BLOCKS_PER_WG sub-groups of 32 threads
    const int sub_group = tid / QUANT_BLOCK;      // which block within this WG (0..7)
    const int sub_tid = tid & (QUANT_BLOCK - 1);  // thread within the 32-element block

    const int global_block = blockIdx.x * BLOCKS_PER_WG + sub_group;
    if (global_block >= total_blocks)
        return;

    const int row = global_block / blocks_per_row;
    const int block_idx = global_block % blocks_per_row;
    if (row >= M)
        return;

    const int k_start = block_idx * QUANT_BLOCK;
    const int k_end = min(k_start + QUANT_BLOCK, K);
    const int block_len = k_end - k_start;

    const float *src = d_A_fp32 + static_cast<size_t>(row) * K + k_start;
    int8_t *dst = d_A_int8 + static_cast<size_t>(row) * K + k_start;

    // Step 1: Load and find max abs (each thread loads 1 element)
    float val = 0.0f;
    float local_max = 0.0f;
    if (sub_tid < block_len)
    {
        val = src[sub_tid];
        local_max = fabsf(val);
    }

    // Warp shuffle reduction within 32-thread sub-group
    // On wave64 hardware, threads 0-31 and 32-63 each form natural sub-groups
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        local_max = fmaxf(local_max, __shfl_down(local_max, offset));
    }
    // Broadcast max from sub-group lane 0 to all 32 threads
    local_max = __shfl(local_max, sub_group * QUANT_BLOCK);

    float scale = (local_max > 0.0f) ? (local_max / 127.0f) : 1.0f;
    float inv_scale = 1.0f / scale;

    // Step 2: Write scale
    if (sub_tid == 0)
    {
        d_scales[static_cast<size_t>(row) * blocks_per_row + block_idx] = scale;
    }

    // Step 3: Quantize
    if (sub_tid < block_len)
    {
        int q = static_cast<int>(rintf(val * inv_scale));
        q = max(-127, min(127, q));
        dst[sub_tid] = static_cast<int8_t>(q);
    }
}

template <int BLOCKS_PER_WG = 8>
__global__ __launch_bounds__(256, 4)
void quantizeActivationsQ8_blockwise_v2_with_sums_kernel(
    const float *__restrict__ d_A_fp32,
    int8_t *__restrict__ d_A_int8,
    float *__restrict__ d_scales,
    int32_t *__restrict__ d_sums,
    int M, int K, int blocks_per_row, int total_blocks)
{
    constexpr int QUANT_BLOCK = 32;
    const int tid = threadIdx.x;
    const int sub_group = tid / QUANT_BLOCK;
    const int sub_tid = tid & (QUANT_BLOCK - 1);

    const int global_block = blockIdx.x * BLOCKS_PER_WG + sub_group;
    if (global_block >= total_blocks)
        return;

    const int row = global_block / blocks_per_row;
    const int block_idx = global_block % blocks_per_row;
    if (row >= M)
        return;

    const int k_start = block_idx * QUANT_BLOCK;
    const int k_end = min(k_start + QUANT_BLOCK, K);
    const int block_len = k_end - k_start;

    const float *src = d_A_fp32 + static_cast<size_t>(row) * K + k_start;
    int8_t *dst = d_A_int8 + static_cast<size_t>(row) * K + k_start;

    float val = 0.0f;
    float local_max = 0.0f;
    if (sub_tid < block_len)
    {
        val = src[sub_tid];
        local_max = fabsf(val);
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        local_max = fmaxf(local_max, __shfl_down(local_max, offset));
    }
    local_max = __shfl(local_max, sub_group * QUANT_BLOCK);

    float scale = (local_max > 0.0f) ? (local_max / 127.0f) : 1.0f;
    float inv_scale = 1.0f / scale;

    int q = 0;
    if (sub_tid < block_len)
    {
        q = static_cast<int>(rintf(val * inv_scale));
        q = max(-127, min(127, q));
        dst[sub_tid] = static_cast<int8_t>(q);
    }

    int q_sum = q;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        q_sum += __shfl_down(q_sum, offset);
    }

    if (sub_tid == 0)
    {
        const size_t out_idx = static_cast<size_t>(row) * blocks_per_row + block_idx;
        d_scales[out_idx] = scale;
        d_sums[out_idx] = q_sum;
    }
}

/**
 * Variant V3: Coarsened — each workgroup processes an entire row (or chunk of rows).
 *
 * Per-block symmetric quantization producing per-block scales.
 * Uses vectorized float4 loads and packed int8x4 stores.
 *
 * Key insight: We still quantize in 32-element blocks, but the max-finding and
 * quantization are fused into a single pass per block, and multiple blocks are
 * handled by the same workgroup using a grid-stride pattern.
 *
 * Grid:  (M, 1, 1) — one workgroup per row
 * Block: (256, 1, 1) = 4 wave64s
 *
 * Each thread processes K/256 elements across multiple blocks, computing
 * the per-block max and quantizing in a single pass.
 */
__global__ __launch_bounds__(256, 4)
void quantizeActivationsQ8_blockwise_v3_kernel(
    const float *__restrict__ d_A_fp32,
    int8_t *__restrict__ d_A_int8,
    float *__restrict__ d_scales,
    int M, int K, int blocks_per_row)
{
    constexpr int QUANT_BLOCK = 32;
    constexpr int WG_SIZE = 256;

    const int row = blockIdx.x;
    if (row >= M)
        return;

    const int tid = threadIdx.x;
    const float *row_in = d_A_fp32 + static_cast<size_t>(row) * K;
    int8_t *row_out = d_A_int8 + static_cast<size_t>(row) * K;
    float *row_scales = d_scales + static_cast<size_t>(row) * blocks_per_row;

    // Shared memory for per-block absmax values
    // blocks_per_row can be up to K/32. For K=18944, that's 592 blocks.
    // We process in chunks that fit in LDS.
    extern __shared__ float s_block_max[];

    // Phase 1: Find per-block max abs values (cooperative across all threads)
    for (int b = tid; b < blocks_per_row; b += WG_SIZE)
    {
        const int k_start = b * QUANT_BLOCK;
        const int k_end = min(k_start + QUANT_BLOCK, K);
        float bmax = 0.0f;
        for (int k = k_start; k < k_end; ++k)
        {
            bmax = fmaxf(bmax, fabsf(row_in[k]));
        }
        s_block_max[b] = bmax;
    }
    __syncthreads();

    // Phase 2: Write scales and quantize (cooperative)
    for (int b = tid; b < blocks_per_row; b += WG_SIZE)
    {
        const float max_abs = s_block_max[b];
        const float scale = (max_abs > 0.0f) ? (max_abs / 127.0f) : 1.0f;
        const float inv_scale = 1.0f / scale;

        row_scales[b] = scale;

        const int k_start = b * QUANT_BLOCK;
        const int k_end = min(k_start + QUANT_BLOCK, K);

        // Vectorized: 8 iterations of float4 for a full 32-element block
        const int vec_end = k_start + ((k_end - k_start) / 4) * 4;
        const bool aligned = ((reinterpret_cast<uintptr_t>(row_in + k_start) & 0x0F) == 0) &&
                             ((reinterpret_cast<uintptr_t>(row_out + k_start) & 0x03) == 0);

        if (aligned && (k_end - k_start) == QUANT_BLOCK)
        {
            // Fast path: full aligned block
            #pragma unroll
            for (int k4 = k_start; k4 < vec_end; k4 += 4)
            {
                const float4 v = *reinterpret_cast<const float4 *>(row_in + k4);
                int q0 = max(-127, min(127, static_cast<int>(rintf(v.x * inv_scale))));
                int q1 = max(-127, min(127, static_cast<int>(rintf(v.y * inv_scale))));
                int q2 = max(-127, min(127, static_cast<int>(rintf(v.z * inv_scale))));
                int q3 = max(-127, min(127, static_cast<int>(rintf(v.w * inv_scale))));
                const uint32_t packed =
                    static_cast<uint32_t>(static_cast<uint8_t>(q0)) |
                    (static_cast<uint32_t>(static_cast<uint8_t>(q1)) << 8) |
                    (static_cast<uint32_t>(static_cast<uint8_t>(q2)) << 16) |
                    (static_cast<uint32_t>(static_cast<uint8_t>(q3)) << 24);
                *reinterpret_cast<uint32_t *>(row_out + k4) = packed;
            }
        }
        else
        {
            for (int k = k_start; k < k_end; ++k)
            {
                int q = max(-127, min(127, static_cast<int>(rintf(row_in[k] * inv_scale))));
                row_out[k] = static_cast<int8_t>(q);
            }
        }
    }
}

/**
 * Variant V4: Wave-parallel — exactly 32 threads per 32-element block.
 *
 * The minimal workgroup size that matches the data. No wasted threads.
 * Uses __shfl for reduction — no shared memory at all.
 *
 * Grid:  (total_blocks, 1, 1)
 * Block: (32, 1, 1) — half a wave64 on gfx906
 *
 * Note: On gfx906 (wave64), a 32-thread workgroup still schedules a full
 * wave64 — 32 threads are masked off. This variant measures whether the
 * reduced register pressure from smaller WG size helps.
 */
__global__ __launch_bounds__(32, 8)
void quantizeActivationsQ8_blockwise_v4_kernel(
    const float *__restrict__ d_A_fp32,
    int8_t *__restrict__ d_A_int8,
    float *__restrict__ d_scales,
    int M, int K, int blocks_per_row)
{
    constexpr int QUANT_BLOCK = 32;
    const int global_block = blockIdx.x;
    const int row = global_block / blocks_per_row;
    const int block_idx = global_block % blocks_per_row;

    if (row >= M)
        return;

    const int tid = threadIdx.x;
    const int k_start = block_idx * QUANT_BLOCK;
    const int k_end = min(k_start + QUANT_BLOCK, K);
    const int block_len = k_end - k_start;

    const float *src = d_A_fp32 + static_cast<size_t>(row) * K + k_start;
    int8_t *dst = d_A_int8 + static_cast<size_t>(row) * K + k_start;

    float val = 0.0f;
    float local_max = 0.0f;
    if (tid < block_len)
    {
        val = src[tid];
        local_max = fabsf(val);
    }

    // Wave shuffle reduction (32 threads)
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        local_max = fmaxf(local_max, __shfl_down(local_max, offset));
    }
    local_max = __shfl(local_max, 0);

    float scale = (local_max > 0.0f) ? (local_max / 127.0f) : 1.0f;
    float inv_scale = 1.0f / scale;

    if (tid == 0)
    {
        d_scales[static_cast<size_t>(row) * blocks_per_row + block_idx] = scale;
    }

    if (tid < block_len)
    {
        int q = max(-127, min(127, static_cast<int>(rintf(val * inv_scale))));
        dst[tid] = static_cast<int8_t>(q);
    }
}

/**
 * Generic blockwise quantization kernel for arbitrary block sizes (32, 64, 128).
 *
 * Uses a single wavefront (64 threads) per block. Each thread handles
 * BLOCK_SIZE/64 elements (rounded up), enabling block sizes larger than 64.
 * For BLOCK_SIZE=32: 1 element/thread (32 active, 32 masked).
 * For BLOCK_SIZE=64: 1 element/thread (all 64 active).
 * For BLOCK_SIZE=128: 2 elements/thread (all 64 active).
 *
 * Grid:  (total_blocks, 1, 1)
 * Block: (64, 1, 1) — one wave64 on gfx906
 */
template <int BLOCK_SIZE>
__global__ __launch_bounds__(64, 8)
void quantizeActivationsQ8_blockwise_generic_kernel(
    const float *__restrict__ d_A_fp32,
    int8_t *__restrict__ d_A_int8,
    float *__restrict__ d_scales,
    int M, int K, int blocks_per_row)
{
    constexpr int ELEMS_PER_THREAD = (BLOCK_SIZE + 63) / 64;

    const int global_block = blockIdx.x;
    const int row = global_block / blocks_per_row;
    const int block_idx = global_block % blocks_per_row;
    if (row >= M) return;

    const int tid = threadIdx.x;
    const int k_start = block_idx * BLOCK_SIZE;
    const int k_end = min(k_start + BLOCK_SIZE, K);
    const int block_len = k_end - k_start;

    const float *src = d_A_fp32 + static_cast<size_t>(row) * K + k_start;
    int8_t *dst = d_A_int8 + static_cast<size_t>(row) * K + k_start;

    // Load elements and find local max
    float vals[ELEMS_PER_THREAD];
    float local_max = 0.0f;

    #pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; ++e)
    {
        const int idx = tid + e * 64;
        if (idx < block_len)
        {
            vals[e] = src[idx];
            local_max = fmaxf(local_max, fabsf(vals[e]));
        }
        else
        {
            vals[e] = 0.0f;
        }
    }

    // Wave shuffle reduction (wave64)
    #pragma unroll
    for (int offset = 32; offset > 0; offset >>= 1)
    {
        local_max = fmaxf(local_max, __shfl_down(local_max, offset));
    }
    local_max = __shfl(local_max, 0);

    float scale = (local_max > 0.0f) ? (local_max / 127.0f) : 1.0f;
    float inv_scale = 1.0f / scale;

    if (tid == 0)
    {
        d_scales[static_cast<size_t>(row) * blocks_per_row + block_idx] = scale;
    }

    #pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; ++e)
    {
        const int idx = tid + e * 64;
        if (idx < block_len)
        {
            int q = max(-127, min(127, static_cast<int>(rintf(vals[e] * inv_scale))));
            dst[idx] = static_cast<int8_t>(q);
        }
    }
}

/**
 * Variant V5: Two-pass vectorized — process entire rows with vec4.
 *
 * Pass 1: Find per-block max abs using vec4 loads, store to shared memory.
 * Pass 2: Quantize using vec4 loads and packed uint32 stores.
 *
 * This variant is optimized for wide rows (K >= 896) where vectorized
 * memory access dominates over compute.
 *
 * Grid:  (M, 1, 1) — one workgroup per row
 * Block: (256, 1, 1)
 *
 * Shared memory: blocks_per_row × sizeof(float) for per-block max values
 */
__global__ __launch_bounds__(256, 4)
void quantizeActivationsQ8_blockwise_v5_kernel(
    const float *__restrict__ d_A_fp32,
    int8_t *__restrict__ d_A_int8,
    float *__restrict__ d_scales,
    int M, int K, int blocks_per_row)
{
    constexpr int QUANT_BLOCK = 32;
    constexpr int WG_SIZE = 256;
    // Each thread processes 4 elements at a time via float4
    constexpr int ELEMS_PER_THREAD = 4;

    const int row = blockIdx.x;
    if (row >= M)
        return;

    const int tid = threadIdx.x;
    const float *row_in = d_A_fp32 + static_cast<size_t>(row) * K;
    int8_t *row_out = d_A_int8 + static_cast<size_t>(row) * K;
    float *row_scales = d_scales + static_cast<size_t>(row) * blocks_per_row;

    extern __shared__ float s_block_max[];

    // Initialize block max values
    for (int b = tid; b < blocks_per_row; b += WG_SIZE)
        s_block_max[b] = 0.0f;
    __syncthreads();

    // Pass 1: Vectorized max-abs finding
    // Each thread loads float4 and atomicMax's into the appropriate block slot
    const int K4 = K / ELEMS_PER_THREAD;
    for (int i = tid; i < K4; i += WG_SIZE)
    {
        const int k = i * ELEMS_PER_THREAD;
        const float4 v = *reinterpret_cast<const float4 *>(row_in + k);

        // All 4 elements may span at most 2 blocks (since block=32, stride=4)
        // But typically they're in the same block
        const int b0 = k / QUANT_BLOCK;

        float bmax = fabsf(v.x);
        bmax = fmaxf(bmax, fabsf(v.y));
        bmax = fmaxf(bmax, fabsf(v.z));
        bmax = fmaxf(bmax, fabsf(v.w));

        // Atomic max in shared memory (float atomics via CAS)
        float old_max = s_block_max[b0];
        while (bmax > old_max)
        {
            float assumed = old_max;
            old_max = __uint_as_float(atomicCAS(
                reinterpret_cast<unsigned int *>(&s_block_max[b0]),
                __float_as_uint(assumed),
                __float_as_uint(bmax)));
            if (old_max == assumed)
                break;
        }
    }
    // Handle tail elements
    for (int k = K4 * ELEMS_PER_THREAD + tid; k < K; k += WG_SIZE)
    {
        const int b = k / QUANT_BLOCK;
        float absval = fabsf(row_in[k]);
        float old_max = s_block_max[b];
        while (absval > old_max)
        {
            float assumed = old_max;
            old_max = __uint_as_float(atomicCAS(
                reinterpret_cast<unsigned int *>(&s_block_max[b]),
                __float_as_uint(assumed),
                __float_as_uint(absval)));
            if (old_max == assumed)
                break;
        }
    }
    __syncthreads();

    // Write scales to global memory
    for (int b = tid; b < blocks_per_row; b += WG_SIZE)
    {
        float max_abs = s_block_max[b];
        float scale = (max_abs > 0.0f) ? (max_abs / 127.0f) : 1.0f;
        row_scales[b] = scale;
        // Overwrite s_block_max with inv_scale for pass 2
        s_block_max[b] = 1.0f / scale;
    }
    __syncthreads();

    // Pass 2: Vectorized quantization using precomputed inv_scale from LDS
    for (int i = tid; i < K4; i += WG_SIZE)
    {
        const int k = i * ELEMS_PER_THREAD;
        const float4 v = *reinterpret_cast<const float4 *>(row_in + k);

        const int b = k / QUANT_BLOCK;
        const float inv_scale = s_block_max[b];

        int q0 = max(-127, min(127, static_cast<int>(rintf(v.x * inv_scale))));
        int q1 = max(-127, min(127, static_cast<int>(rintf(v.y * inv_scale))));
        int q2 = max(-127, min(127, static_cast<int>(rintf(v.z * inv_scale))));
        int q3 = max(-127, min(127, static_cast<int>(rintf(v.w * inv_scale))));

        const uint32_t packed =
            static_cast<uint32_t>(static_cast<uint8_t>(q0)) |
            (static_cast<uint32_t>(static_cast<uint8_t>(q1)) << 8) |
            (static_cast<uint32_t>(static_cast<uint8_t>(q2)) << 16) |
            (static_cast<uint32_t>(static_cast<uint8_t>(q3)) << 24);

        *reinterpret_cast<uint32_t *>(row_out + k) = packed;
    }
    // Tail
    for (int k = K4 * ELEMS_PER_THREAD + tid; k < K; k += WG_SIZE)
    {
        const int b = k / QUANT_BLOCK;
        const float inv_scale = s_block_max[b];
        int q = max(-127, min(127, static_cast<int>(rintf(row_in[k] * inv_scale))));
        row_out[k] = static_cast<int8_t>(q);
    }
}

// ============================================================================
// DISPATCH FUNCTIONS
// ============================================================================

extern "C"
{
    // -------------------------------------------------------------------------
    // Device management
    // -------------------------------------------------------------------------

    bool rocmQuantGemm_setDevice(int rocm_device_id)
    {
        hipError_t err = static_cast<hipError_t>(llaminar2::HipDeviceGuard::setDevice(rocm_device_id));
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_setDevice] Failed to set device %d: %s\n",
                    rocm_device_id, hipGetErrorString(err));
            return false;
        }
        return true;
    }

    // -------------------------------------------------------------------------
    // Weight upload (Phase 1)
    // -------------------------------------------------------------------------

    bool rocmQuantGemm_uploadWeights(
        const int8_t *h_weights_int8,
        const float *h_scales_B,
        int8_t **d_weights_int8,
        float **d_scales_B,
        int K, int N,
        int rocm_device_id)
    {
        if (!rocmQuantGemm_setDevice(rocm_device_id))
            return false;

        // Allocate device memory for weights
        size_t weight_bytes = static_cast<size_t>(K) * N * sizeof(int8_t);
        hipError_t err = hipMalloc(d_weights_int8, weight_bytes);
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_uploadWeights] hipMalloc weights failed: %s\n",
                    hipGetErrorString(err));
            return false;
        }
        // CRITICAL: Verify hipMalloc actually allocated memory
        if (*d_weights_int8 == nullptr)
        {
            fprintf(stderr, "[rocmQuantGemm_uploadWeights] CRITICAL: hipMalloc returned success but pointer is NULL! "
                    "(bytes=%zu K=%d N=%d device=%d)\n",
                    weight_bytes, K, N, rocm_device_id);
            return false;
        }

        // Allocate device memory for scales
        size_t scale_bytes = static_cast<size_t>(N) * sizeof(float);
        err = hipMalloc(d_scales_B, scale_bytes);
        if (err != hipSuccess)
        {
            hipFree(*d_weights_int8);
            *d_weights_int8 = nullptr;
            fprintf(stderr, "[rocmQuantGemm_uploadWeights] hipMalloc scales failed: %s\n",
                    hipGetErrorString(err));
            return false;
        }

        // Copy weights H2D
        err = hipMemcpy(*d_weights_int8, h_weights_int8, weight_bytes, hipMemcpyHostToDevice);
        if (err != hipSuccess)
        {
            hipFree(*d_weights_int8);
            hipFree(*d_scales_B);
            *d_weights_int8 = nullptr;
            *d_scales_B = nullptr;
            fprintf(stderr, "[rocmQuantGemm_uploadWeights] hipMemcpy weights failed: %s "
                    "(src=%p dst=%p bytes=%zu K=%d N=%d device=%d)\n",
                    hipGetErrorString(err), (void*)h_weights_int8, (void*)*d_weights_int8,
                    weight_bytes, K, N, rocm_device_id);
            return false;
        }

        // Copy scales H2D
        err = hipMemcpy(*d_scales_B, h_scales_B, scale_bytes, hipMemcpyHostToDevice);
        if (err != hipSuccess)
        {
            hipFree(*d_weights_int8);
            hipFree(*d_scales_B);
            *d_weights_int8 = nullptr;
            *d_scales_B = nullptr;
            fprintf(stderr, "[rocmQuantGemm_uploadWeights] hipMemcpy scales failed: %s\n",
                    hipGetErrorString(err));
            return false;
        }

        return true;
    }

    // -------------------------------------------------------------------------
    // Work buffer allocation (Phase 2)
    // -------------------------------------------------------------------------

    bool rocmQuantGemm_ensureWorkBuffers(
        int8_t **d_A_int8,
        float **d_scales_A,
        int32_t **d_C_int32,
        int *work_buffer_M,
        int M, int K, int N,
        int rocm_device_id)
    {
        if (!rocmQuantGemm_setDevice(rocm_device_id))
            return false;

        // Only reallocate if M increased
        if (M <= *work_buffer_M)
        {
            return true;
        }

        // Free existing buffers
        if (*d_A_int8) {
            hipFree(*d_A_int8);
        }
        if (*d_scales_A) {
            hipFree(*d_scales_A);
        }
        if (*d_C_int32) {
            hipFree(*d_C_int32);
        }

        *d_A_int8 = nullptr;
        *d_scales_A = nullptr;
        *d_C_int32 = nullptr;

        // Allocate new buffers
        hipError_t err = hipMalloc(d_A_int8, static_cast<size_t>(M) * K * sizeof(int8_t));
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_ensureWorkBuffers] hipMalloc A_int8 failed: %s\n", hipGetErrorString(err));
            return false;
        }

        err = hipMalloc(d_scales_A, static_cast<size_t>(M) * sizeof(float));
        if (err != hipSuccess)
        {
            hipFree(*d_A_int8);
            *d_A_int8 = nullptr;
            fprintf(stderr, "[rocmQuantGemm_ensureWorkBuffers] hipMalloc scales_A failed: %s\n", hipGetErrorString(err));
            return false;
        }

        err = hipMalloc(d_C_int32, static_cast<size_t>(M) * N * sizeof(int32_t));
        if (err != hipSuccess)
        {
            hipFree(*d_A_int8);
            hipFree(*d_scales_A);
            *d_A_int8 = nullptr;
            *d_scales_A = nullptr;
            fprintf(stderr, "[rocmQuantGemm_ensureWorkBuffers] hipMalloc C_int32 failed: %s\n", hipGetErrorString(err));
            return false;
        }

        *work_buffer_M = M;
        return true;
    }

    // =========================================================================
    // INT8 ALLOCATION HELPER
    // =========================================================================
    
    bool rocmQuantGemm_allocInt8(int8_t **d_ptr, size_t count, int rocm_device_id)
    {
        if (!rocmQuantGemm_setDevice(rocm_device_id))
            return false;

        hipError_t err = hipMalloc(d_ptr, count * sizeof(int8_t));
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_allocInt8] hipMalloc failed: %s\n",
                    hipGetErrorString(err));
            return false;
        }
        return true;
    }

    // =========================================================================
    // BLOCKWISE ACTIVATION QUANTIZATION LAUNCHER
    // =========================================================================

    /**
     * @brief Launch blockwise activation quantization kernel
     *
     * Quantizes activations with per-block scales.
     * block_size: quantization block size (32, 64, or 128 elements).
     * For block_size=32: Variant selected by LLAMINAR_ROCM_BLOCKWISE_QUANT_VARIANT env var.
     * For block_size=64/128: Uses generic kernel (1 wave64 per block).
     */
    bool rocmQuantGemm_quantizeActivationsBlockwise(
        const float *d_A_fp32,
        int8_t *d_A_int8,
        float *d_scales_blockwise,
        int M, int K,
        int rocm_device_id, void *stream,
        int block_size)
    {
        if (!rocmQuantGemm_setDevice(rocm_device_id))
            return false;

        if (M <= 0 || K <= 0)
            return true;

        hipStream_t s = static_cast<hipStream_t>(stream);

        // For non-standard block sizes, use the generic kernel
        if (block_size == 64 || block_size == 128)
        {
            const int blocks_per_row = (K + block_size - 1) / block_size;
            const int total_blocks = M * blocks_per_row;
            dim3 grid(total_blocks, 1, 1);
            dim3 blk(64, 1, 1);

            if (block_size == 64)
            {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(quantizeActivationsQ8_blockwise_generic_kernel<64>),
                    grid, blk, 0, s,
                    d_A_fp32, d_A_int8, d_scales_blockwise, M, K, blocks_per_row);
            }
            else
            {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(quantizeActivationsQ8_blockwise_generic_kernel<128>),
                    grid, blk, 0, s,
                    d_A_fp32, d_A_int8, d_scales_blockwise, M, K, blocks_per_row);
            }

            hipError_t err = hipGetLastError();
            if (err != hipSuccess)
            {
                fprintf(stderr, "[rocmQuantGemm_quantizeActivationsBlockwise] Generic kernel (block_size=%d) launch failed: %s\n",
                        block_size, hipGetErrorString(err));
                return false;
            }
            return true;
        }

        // Standard block_size=32 path
        constexpr int QUANT_BLOCK = 32;
        const int blocks_per_row = (K + QUANT_BLOCK - 1) / QUANT_BLOCK;
        const int total_blocks = M * blocks_per_row;

        // variant=0 (auto): M=1 → V4 (best latency), M≤64 → V2 (multi-block), M≥128 → V5 (vec4)
        int variant = llaminar2::debugEnv().rocm.blockwise_quant_variant;
        if (variant == 0)
        {
            if (M <= 1)      variant = 4;  // V4: Minimal WG, best single-row latency
            else if (M <= 64) variant = 2;  // V2: Multi-block, amortized launch overhead
            else              variant = 5;  // V5: Two-pass vec4, bandwidth-dominated
        }

        switch (variant)
        {
        case 2:
        {
            // V2: Multi-block per workgroup (8 blocks × 32 threads = 256)
            constexpr int BLOCKS_PER_WG = 8;
            dim3 grid((total_blocks + BLOCKS_PER_WG - 1) / BLOCKS_PER_WG, 1, 1);
            dim3 block(256, 1, 1);
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(quantizeActivationsQ8_blockwise_v2_kernel<BLOCKS_PER_WG>),
                grid, block, 0, s,
                d_A_fp32, d_A_int8, d_scales_blockwise, M, K, blocks_per_row, total_blocks);
            break;
        }
        case 3:
        {
            // V3: Row-coarsened, sequential blocks with vec4
            dim3 grid(M, 1, 1);
            dim3 block(256, 1, 1);
            size_t shared_bytes = static_cast<size_t>(blocks_per_row) * sizeof(float);
            hipLaunchKernelGGL(
                quantizeActivationsQ8_blockwise_v3_kernel,
                grid, block, shared_bytes, s,
                d_A_fp32, d_A_int8, d_scales_blockwise, M, K, blocks_per_row);
            break;
        }
        case 4:
        {
            // V4: Minimal WG — 32 threads, 1 block/WG
            dim3 grid(total_blocks, 1, 1);
            dim3 block(32, 1, 1);
            hipLaunchKernelGGL(
                quantizeActivationsQ8_blockwise_v4_kernel,
                grid, block, 0, s,
                d_A_fp32, d_A_int8, d_scales_blockwise, M, K, blocks_per_row);
            break;
        }
        case 5:
        {
            // V5: Two-pass vectorized (vec4 loads, atomicCAS max, vec4 quant)
            dim3 grid(M, 1, 1);
            dim3 block(256, 1, 1);
            size_t shared_bytes = static_cast<size_t>(blocks_per_row) * sizeof(float);
            hipLaunchKernelGGL(
                quantizeActivationsQ8_blockwise_v5_kernel,
                grid, block, shared_bytes, s,
                d_A_fp32, d_A_int8, d_scales_blockwise, M, K, blocks_per_row);
            break;
        }
        default:
        {
            // V1 (default): Original — 64 threads, 1 block/WG
            dim3 grid(total_blocks, 1, 1);
            dim3 block(64, 1, 1);
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(quantizeActivationsQ8_blockwise_kernel<QUANT_BLOCK>),
                grid, block, 0, s,
                d_A_fp32, d_A_int8, d_scales_blockwise, M, K, blocks_per_row);
            break;
        }
        }

        hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_quantizeActivationsBlockwise] Kernel V%d launch failed: %s\n",
                    variant, hipGetErrorString(err));
            return false;
        }

        return true;
    }

    bool rocmQuantGemm_quantizeActivationsBlockwiseWithSums(
        const float *d_A_fp32,
        int8_t *d_A_int8,
        float *d_scales_blockwise,
        int32_t *d_sums_blockwise,
        int M, int K,
        int rocm_device_id, void *stream,
        int block_size)
    {
        if (!d_sums_blockwise)
        {
            return rocmQuantGemm_quantizeActivationsBlockwise(
                d_A_fp32, d_A_int8, d_scales_blockwise,
                M, K, rocm_device_id, stream, block_size);
        }

        if (!rocmQuantGemm_setDevice(rocm_device_id))
            return false;

        if (M <= 0 || K <= 0)
            return true;

        if (block_size != 32)
        {
            fprintf(stderr,
                    "[rocmQuantGemm_quantizeActivationsBlockwiseWithSums] block_size=%d is unsupported; expected 32 (M=%d K=%d)\n",
                    block_size, M, K);
            return false;
        }

        constexpr int QUANT_BLOCK = 32;
        constexpr int BLOCKS_PER_WG = 8;
        const int blocks_per_row = (K + QUANT_BLOCK - 1) / QUANT_BLOCK;
        const int total_blocks = M * blocks_per_row;
        dim3 grid((total_blocks + BLOCKS_PER_WG - 1) / BLOCKS_PER_WG, 1, 1);
        dim3 block(256, 1, 1);

        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(quantizeActivationsQ8_blockwise_v2_with_sums_kernel<BLOCKS_PER_WG>),
            grid, block, 0, static_cast<hipStream_t>(stream),
            d_A_fp32, d_A_int8, d_scales_blockwise, d_sums_blockwise,
            M, K, blocks_per_row, total_blocks);

        hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_quantizeActivationsBlockwiseWithSums] Kernel launch failed: %s\n",
                    hipGetErrorString(err));
            return false;
        }

        return true;
    }

    // =========================================================================
    // MEMORY MANAGEMENT UTILITIES
    // =========================================================================

    void rocmQuantGemm_freeDevice(void *d_ptr, int rocm_device_id)
    {
        if (d_ptr)
        {
            // Set device before freeing - required by HIP/CUDA
            hipError_t set_err = static_cast<hipError_t>(llaminar2::HipDeviceGuard::setDevice(rocm_device_id));
            if (set_err == hipErrorDeinitialized || set_err == hipErrorNoDevice)
            {
                // HIP runtime is shutting down, skip cleanup
                // This is normal during static destruction at program exit
                return;
            }
            hipError_t err = hipFree(d_ptr);
            if (err != hipSuccess && err != hipErrorDeinitialized && err != hipErrorNoDevice)
            {
                // Only log actual errors, not shutdown-related ones
                fprintf(stderr, "WARNING: hipFree failed: %s\n", hipGetErrorString(err));
            }
        }
    }

    bool rocmQuantGemm_allocFloat(float **d_ptr, size_t count, int rocm_device_id)
    {
        if (!rocmQuantGemm_setDevice(rocm_device_id))
            return false;

        hipError_t err = hipMalloc(d_ptr, count * sizeof(float));
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_allocFloat] hipMalloc failed: %s\n",
                    hipGetErrorString(err));
            return false;
        }
        return true;
    }

    bool rocmQuantGemm_allocInt32(int32_t **d_ptr, size_t count, int rocm_device_id)
    {
        if (!rocmQuantGemm_setDevice(rocm_device_id))
            return false;

        hipError_t err = hipMalloc(d_ptr, count * sizeof(int32_t));
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_allocInt32] hipMalloc failed: %s\n",
                    hipGetErrorString(err));
            return false;
        }
        return true;
    }

    bool rocmQuantGemm_copyHostToDevice(float *d_dst, const float *h_src, size_t count, int rocm_device_id)
    {
        if (!rocmQuantGemm_setDevice(rocm_device_id))
            return false;

        hipError_t err = hipMemcpy(d_dst, h_src, count * sizeof(float), hipMemcpyHostToDevice);
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_copyHostToDevice] hipMemcpy failed: %s\n",
                    hipGetErrorString(err));
            return false;
        }
        return true;
    }

    bool rocmQuantGemm_copyDeviceToHost(float *h_dst, const float *d_src, size_t count, int rocm_device_id)
    {
        if (!rocmQuantGemm_setDevice(rocm_device_id))
            return false;

        hipError_t err = hipMemcpy(h_dst, d_src, count * sizeof(float), hipMemcpyDeviceToHost);
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_copyDeviceToHost] hipMemcpy failed: %s\n",
                    hipGetErrorString(err));
            return false;
        }
        return true;
    }

    bool rocmQuantGemm_copyInt32DeviceToHost(int32_t *h_dst, const int32_t *d_src, size_t count, int rocm_device_id)
    {
        if (!rocmQuantGemm_setDevice(rocm_device_id))
            return false;

        hipError_t err = hipMemcpy(h_dst, d_src, count * sizeof(int32_t), hipMemcpyDeviceToHost);
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_copyInt32DeviceToHost] hipMemcpy failed: %s\n",
                    hipGetErrorString(err));
            return false;
        }
        return true;
    }

    // =========================================================================
    // APPLY SCALING WITH FULL EPILOGUE (alpha, beta, bias)
    // =========================================================================

    bool rocmQuantGemm_applyScaling(
        const int32_t *d_C_int32,    // [M×N] INT32 GEMM output
        float *d_C_fp32,             // [M×N] FP32 output
        const float *d_scales_A,     // [M] per-row scales
        const float *d_scales_B,     // [N] per-column scales
        int M, int N,
        float alpha, float beta,
        const float *d_C_existing,   // For beta != 0 (nullable)
        const float *d_bias,         // [N] optional bias (nullable)
        int rocm_device_id, void* stream)
    {
        // Validate required pointers
        if (!d_C_int32 || !d_C_fp32 || !d_scales_A || !d_scales_B)
        {
            fprintf(stderr, "[rocmQuantGemm_applyScaling] Null pointer: "
                    "d_C_int32=%p d_C_fp32=%p d_scales_A=%p d_scales_B=%p\n",
                    (void *)d_C_int32, (void *)d_C_fp32,
                    (void *)d_scales_A, (void *)d_scales_B);
            return false;
        }

        if (!rocmQuantGemm_setDevice(rocm_device_id))
            return false;

        // Decode-specialized fast path:
        // M=1 dominates INT8 VNNI decode and the generic 16x16 kernel wastes
        // 15/16 threads in this shape. Use a 1D vectorized kernel instead.
        if (M == 1 && beta == 0.0f && d_C_existing == nullptr && d_bias == nullptr)
        {
            const int block_size = 256;
            const int elems_per_thread = 4;
            const int grid_size = (N + block_size * elems_per_thread - 1) / (block_size * elems_per_thread);

            hipLaunchKernelGGL(
                applyScaling_m1_vec4_kernel,
                dim3(grid_size), dim3(block_size), 0, static_cast<hipStream_t>(stream),
                d_C_int32, d_C_fp32, d_scales_A, d_scales_B, N, alpha);

            hipError_t err = hipGetLastError();
            if (err != hipSuccess)
            {
                fprintf(stderr, "[rocmQuantGemm_applyScaling] applyScaling_m1_vec4_kernel launch failed: %s "
                        "(M=%d, N=%d)\n", hipGetErrorString(err), M, N);
                return false;
            }
            return true;
        }

        // Vectorized 1D flat path for M > 1:
        // Uses sequential access through the entire M×N buffer (HBM-friendly)
        // with a fixed block count regardless of shape dimensions.
        if (N % 4 == 0)
        {
            constexpr int BLOCK_X = 256;
            const int total_vec4 = M * (N / 4);
            // Cap grid to ~480 blocks for MI60 (60 CUs × 8 waves/CU occupancy target)
            constexpr int MAX_BLOCKS = 480;
            const int natural_blocks = (total_vec4 + BLOCK_X - 1) / BLOCK_X;
            const int grid_size = (natural_blocks < MAX_BLOCKS) ? natural_blocks : MAX_BLOCKS;

            hipLaunchKernelGGL(
                applyScaling_vec4_flat_kernel,
                dim3(grid_size), dim3(BLOCK_X), 0, static_cast<hipStream_t>(stream),
                d_C_int32, d_C_fp32, d_scales_A, d_scales_B,
                M, N, total_vec4, alpha, beta, d_C_existing, d_bias);
        }
        else
        {
            // Fallback for N not divisible by 4 (shouldn't happen in practice)
            dim3 block(16, 16);
            dim3 grid((N + 15) / 16, (M + 15) / 16);
            hipLaunchKernelGGL(
                applyScaling_full_kernel,
                grid, block, 0, static_cast<hipStream_t>(stream),
                d_C_int32, d_C_fp32, d_scales_A, d_scales_B,
                M, N, alpha, beta, d_C_existing, d_bias);
        }

        hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_applyScaling] applyScaling_vec4_2d_kernel launch failed: %s "
                    "(M=%d, N=%d)\n", hipGetErrorString(err), M, N);
            return false;
        }

        // No sync needed - coherence system handles sync when data is read
        return true;
    }

    // =========================================================================
    // Apply Scale-A Only Kernel (for grouped ratio-VNNI decode path)
    //
    // When sub-row grouping is used, the GEMV kernel already applies per-group
    // weight scales during FP32 partial accumulation. The remaining scaling is
    // just the activation scale:
    //   output[m,n] = alpha * C_fp32_grouped[m,n] * scale_A[m]
    //                 + beta * existing[m,n] + bias[n]
    // =========================================================================

    __global__ void applyScaleA_m1_vec4_kernel(
        const float* __restrict__ d_C_fp32_grouped,  // [N] FP32 with B-scales already applied
        float* __restrict__ d_output,                 // [N] FP32 output
        const float* __restrict__ d_scale_A,          // [1] activation scale
        int N,
        float alpha)
    {
        const int tid = blockIdx.x * blockDim.x + threadIdx.x;
        const int n4 = tid * 4;

        const float row_scale = alpha * d_scale_A[0];

        if (n4 + 3 < N)
        {
            const float4 v = *reinterpret_cast<const float4*>(d_C_fp32_grouped + n4);

            float4 out;
            out.x = v.x * row_scale;
            out.y = v.y * row_scale;
            out.z = v.z * row_scale;
            out.w = v.w * row_scale;

            *reinterpret_cast<float4*>(d_output + n4) = out;
            return;
        }

        for (int n = n4; n < N && n < n4 + 4; ++n)
        {
            d_output[n] = d_C_fp32_grouped[n] * row_scale;
        }
    }

    __global__ void applyScaleA_full_kernel(
        const float* __restrict__ d_C_fp32_grouped,  // [M×N] FP32 with B-scales applied
        float* __restrict__ d_output,                 // [M×N] FP32 output
        const float* __restrict__ d_scale_A,          // [M] activation scales
        int M, int N,
        float alpha, float beta,
        const float* __restrict__ d_C_existing,       // [M×N] existing (for beta != 0)
        const float* __restrict__ d_bias)             // [N] optional bias
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        const int total = M * N;
        if (idx >= total) return;

        const int m = idx / N;
        const int n = idx % N;

        float result = alpha * d_C_fp32_grouped[idx] * d_scale_A[m];

        if (beta != 0.0f && d_C_existing != nullptr)
        {
            result += beta * d_C_existing[idx];
        }
        if (d_bias != nullptr)
        {
            result += d_bias[n];
        }

        d_output[idx] = result;
    }

    bool rocmQuantGemm_applyScaleA_fp32(
        const float *d_C_fp32_grouped,   // [M×N] FP32 with B-scales already applied
        float *d_output,                  // [M×N] FP32 output
        const float *d_scales_A,          // [M] per-row activation scales
        int M, int N,
        float alpha, float beta,
        const float *d_C_existing,
        const float *d_bias,
        int rocm_device_id, void* stream)
    {
        if (!d_C_fp32_grouped || !d_output || !d_scales_A)
        {
            fprintf(stderr, "[rocmQuantGemm_applyScaleA_fp32] Null pointer\n");
            return false;
        }

        if (!rocmQuantGemm_setDevice(rocm_device_id))
            return false;

        // Decode-specialized fast path (M=1, no beta/bias)
        if (M == 1 && beta == 0.0f && d_C_existing == nullptr && d_bias == nullptr)
        {
            const int block_size = 256;
            const int elems_per_thread = 4;
            const int grid_size = (N + block_size * elems_per_thread - 1) / (block_size * elems_per_thread);

            hipLaunchKernelGGL(
                applyScaleA_m1_vec4_kernel,
                dim3(grid_size), dim3(block_size), 0, static_cast<hipStream_t>(stream),
                d_C_fp32_grouped, d_output, d_scales_A, N, alpha);
        }
        else
        {
            const int block_size = 256;
            const int total = M * N;
            const int grid_size = (total + block_size - 1) / block_size;

            hipLaunchKernelGGL(
                applyScaleA_full_kernel,
                dim3(grid_size), dim3(block_size), 0, static_cast<hipStream_t>(stream),
                d_C_fp32_grouped, d_output, d_scales_A,
                M, N, alpha, beta, d_C_existing, d_bias);
        }

        hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_applyScaleA_fp32] Kernel launch failed: %s (M=%d, N=%d)\n",
                    hipGetErrorString(err), M, N);
            return false;
        }

        return true;
    }

    bool rocmQuantGemm_applyFp32Epilogue(
        float *d_output,
        const float *d_bias,
        int M, int N,
        float alpha,
        int rocm_device_id, void* stream)
    {
        if (!d_output)
        {
            fprintf(stderr, "[rocmQuantGemm_applyFp32Epilogue] Null output pointer\n");
            return false;
        }

        if (!rocmQuantGemm_setDevice(rocm_device_id))
            return false;

        if (alpha == 1.0f && d_bias == nullptr)
            return true;

        const int block_size = 256;
        const int total = M * N;
        const int grid_size = (total + block_size - 1) / block_size;
        hipLaunchKernelGGL(
            applyFp32Epilogue_kernel,
            dim3(grid_size), dim3(block_size), 0, static_cast<hipStream_t>(stream),
            d_output, d_bias, M, N, alpha);

        hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_applyFp32Epilogue] Kernel launch failed: %s (M=%d, N=%d)\n",
                    hipGetErrorString(err), M, N);
            return false;
        }

        return true;
    }

    bool rocmQuantGemm_applyFp32CombineEpilogue(
        const float *d_computed,
        float *d_output,
        const float *d_existing,
        const float *d_bias,
        int M, int N,
        float alpha, float beta,
        int rocm_device_id, void *stream)
    {
        if (!d_computed || !d_output)
        {
            fprintf(stderr, "[rocmQuantGemm_applyFp32CombineEpilogue] Null pointer: d_computed=%p d_output=%p\n",
                    (const void *)d_computed, (void *)d_output);
            return false;
        }

        if (beta != 0.0f && d_existing == nullptr)
        {
            fprintf(stderr, "[rocmQuantGemm_applyFp32CombineEpilogue] d_existing required when beta != 0 (M=%d N=%d)\n",
                    M, N);
            return false;
        }

        if (!rocmQuantGemm_setDevice(rocm_device_id))
            return false;

        if (d_computed == d_output && beta == 0.0f)
        {
            return rocmQuantGemm_applyFp32Epilogue(
                d_output, d_bias, M, N, alpha, rocm_device_id, stream);
        }

        const int block_size = 256;
        const int total = M * N;
        const int grid_size = (total + block_size - 1) / block_size;
        hipLaunchKernelGGL(
            applyFp32CombineEpilogue_kernel,
            dim3(grid_size), dim3(block_size), 0, static_cast<hipStream_t>(stream),
            d_computed, d_output, d_existing, d_bias, M, N, alpha, beta);

        hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_applyFp32CombineEpilogue] Kernel launch failed: %s (M=%d, N=%d)\n",
                    hipGetErrorString(err), M, N);
            return false;
        }

        return true;
    }
    // =========================================================================
    // WEIGHT-ONLY SCALING EPILOGUE (for blockwise activation-scaled GEMM)
    // =========================================================================
    //
    // When the GEMM kernel has already applied per-block activation scales,
    // the epilogue only needs to apply weight scales:
    //   output[m,n] = alpha * C_fp32[m,n] * scale_B[n] + beta * existing + bias
    //
    // Input is FP32 (not INT32) because the blockwise kernel converts to float.

    __global__ void applyScaling_weightOnly_vec4_flat_kernel(
        const float* __restrict__ d_C_fp32_in,  // [M×N] FP32 with act-scales applied
        float* __restrict__ d_output,            // [M×N] FP32 final output
        const float* __restrict__ d_scale_B,     // [N] per-column weight scales
        int M, int N, int total_vec4,
        float alpha, float beta,
        const float* __restrict__ d_C_existing,
        const float* __restrict__ d_bias)
    {
        const int N4 = N / 4;

        for (int vi = blockIdx.x * blockDim.x + threadIdx.x;
             vi < total_vec4;
             vi += gridDim.x * blockDim.x)
        {
            const int row = vi / N4;
            const int col4 = (vi - row * N4) * 4;
            const int idx = row * N + col4;

            const float4 vals = *reinterpret_cast<const float4*>(d_C_fp32_in + idx);
            const float4 sb = *reinterpret_cast<const float4*>(d_scale_B + col4);

            float4 out;
            out.x = alpha * vals.x * sb.x;
            out.y = alpha * vals.y * sb.y;
            out.z = alpha * vals.z * sb.z;
            out.w = alpha * vals.w * sb.w;

            if (beta != 0.0f && d_C_existing != nullptr)
            {
                const float4 ex = *reinterpret_cast<const float4*>(d_C_existing + idx);
                out.x += beta * ex.x;
                out.y += beta * ex.y;
                out.z += beta * ex.z;
                out.w += beta * ex.w;
            }

            if (d_bias != nullptr)
            {
                const float4 b = *reinterpret_cast<const float4*>(d_bias + col4);
                out.x += b.x;
                out.y += b.y;
                out.z += b.z;
                out.w += b.w;
            }

            *reinterpret_cast<float4*>(d_output + idx) = out;
        }
    }

    bool rocmQuantGemm_applyScaling_weightOnly(
        const float *d_C_fp32_in,    // [M×N] FP32 with activation scales applied
        float *d_output,              // [M×N] FP32 final output
        const float *d_scales_B,      // [N] per-column weight scales
        int M, int N,
        float alpha, float beta,
        const float *d_C_existing,
        const float *d_bias,
        int rocm_device_id, void* stream)
    {
        if (!d_C_fp32_in || !d_output || !d_scales_B)
        {
            fprintf(stderr, "[rocmQuantGemm_applyScaling_weightOnly] Null pointer\n");
            return false;
        }

        if (!rocmQuantGemm_setDevice(rocm_device_id))
            return false;

        constexpr int BLOCK_X = 256;
        const int total_vec4 = M * (N / 4);
        constexpr int MAX_BLOCKS = 480;
        const int natural_blocks = (total_vec4 + BLOCK_X - 1) / BLOCK_X;
        const int grid_size = (natural_blocks < MAX_BLOCKS) ? natural_blocks : MAX_BLOCKS;

        hipLaunchKernelGGL(
            applyScaling_weightOnly_vec4_flat_kernel,
            dim3(grid_size), dim3(BLOCK_X), 0, static_cast<hipStream_t>(stream),
            d_C_fp32_in, d_output, d_scales_B,
            M, N, total_vec4, alpha, beta, d_C_existing, d_bias);

        hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_applyScaling_weightOnly] Launch failed: %s (M=%d, N=%d)\n",
                    hipGetErrorString(err), M, N);
            return false;
        }

        return true;
    }

    // =========================================================================
    // IN-PLACE BIAS ADDITION KERNEL
    // =========================================================================

    __global__ void biasAdd_kernel(float *output, const float *bias, int M, int N)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= M * N)
            return;
        
        const int col = idx % N;
        output[idx] += bias[col];
    }

    bool rocmQuantGemm_biasAdd(
        float *d_output,       // [M × N] FP32 output (modified in-place)
        const float *d_bias,   // [N] bias vector
        int M, int N,
        int rocm_device_id, void* stream)
    {
        if (!d_output || !d_bias)
        {
            fprintf(stderr, "[rocmQuantGemm_biasAdd] Null pointer: d_output=%p d_bias=%p\n",
                    (void *)d_output, (void *)d_bias);
            return false;
        }

        if (!rocmQuantGemm_setDevice(rocm_device_id))
            return false;

        const int total = M * N;
        const int block_size = 256;
        const int grid_size = (total + block_size - 1) / block_size;

        hipLaunchKernelGGL(
            biasAdd_kernel,
            dim3(grid_size), dim3(block_size), 0, static_cast<hipStream_t>(stream),
            d_output, d_bias, M, N);

        hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_biasAdd] biasAdd_kernel launch failed: %s\n",
                    hipGetErrorString(err));
            return false;
        }

        // No sync needed - coherence system handles sync when data is read
        return true;
    }

} // extern "C"
