/**
 * @file CUDARoPEKernels.cu
 * @brief CUDA RoPE (Rotary Position Embedding) kernel implementations
 * @author David Sanftenberg
 *
 * Contains FP32, BF16, and FP16 RoPE kernels with extern "C" wrapper functions.
 * Uses SPLIT-HALF layout matching the CPU implementation.
 *
 * OPTIMIZATIONS (v3 - CPU strategy adaptation):
 * - Pre-computed inverse frequency table cached in device memory
 * - Shared memory for sin/cos table per thread block
 * - Main rotation is pure FMA (no per-thread transcendentals)
 * - Single fused kernel for Q+K to reduce launch overhead
 *
 * OPTIMIZATION (v4 - Fused Q+K kernel):
 * - Process both Q and K in a single kernel launch to reduce overhead
 */

#include "CUDAHelpers.cuh"
#include "kernels/rope/RoPEDeviceParams.h"
#include <cmath>
#include <cstdio>
#include <vector>

// =========================================================================
// Inverse Frequency Cache (CPU-side, mirrors RoPEPrimitives.cpp)
// =========================================================================

namespace
{
    bool ropeLaunchOk(const char *name)
    {
        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            std::fprintf(stderr, "[%s] CUDA launch failed: %s\n",
                         name, cudaGetErrorString(err));
            return false;
        }
        return true;
    }

} // anonymous namespace

// =========================================================================
// Optimized RoPE CUDA Kernels (v3 - with precomputed inv_freq)
// =========================================================================

/**
 * @brief FP32 RoPE kernel with shared memory sin/cos table
 *
 * Strategy (adapted from CPU RoPEPrimitives.cpp):
 * 1. Each thread block loads inv_freq into shared memory once
 * 2. For each position in the block, compute sin/cos table in shared memory
 * 3. Apply rotation using cached sin/cos (pure FMA, no transcendentals)
 *
 * This kernel processes one (seq_idx, head_idx) pair per thread block.
 * Threads within the block cooperatively process the half_dim pairs.
 */
__global__ void rope_fp32_kernel_v3(
    float *__restrict__ data,
    const float *__restrict__ inv_freq, // Pre-computed inverse frequencies [half_rotary]
    const int *__restrict__ position_ids,
    int seq_len,
    int n_heads,
    int head_dim,
    int rotary_dim)
{
    const int half_rotary = rotary_dim / 2;

    // Shared memory for sin/cos cache (one set per position in this block)
    extern __shared__ float smem[];
    float *s_cos = smem;
    float *s_sin = smem + half_rotary;

    // Each block handles one (seq_idx, head_idx) pair
    int block_idx = blockIdx.x;
    int head_idx = block_idx % n_heads;
    int seq_idx = block_idx / n_heads;

    if (seq_idx >= seq_len)
        return;

    // Get actual position
    int pos = position_ids ? position_ids[seq_idx] : seq_idx;
    if (pos < 0)
        return;

    // Step 1: Cooperatively compute sin/cos table into shared memory
    // Each thread handles multiple pairs if half_rotary > blockDim.x
    for (int i = threadIdx.x; i < half_rotary; i += blockDim.x)
    {
        float angle = pos * inv_freq[i];
        __sincosf(angle, &s_sin[i], &s_cos[i]);
    }
    __syncthreads();

    // Step 2: Apply rotation using cached sin/cos (pure FMA)
    int base_idx = seq_idx * n_heads * head_dim + head_idx * head_dim;

    for (int i = threadIdx.x; i < half_rotary; i += blockDim.x)
    {
        int i0 = base_idx + i;
        int i1 = base_idx + i + half_rotary;

        float x0 = data[i0];
        float x1 = data[i1];
        float cos_val = s_cos[i];
        float sin_val = s_sin[i];

        data[i0] = x0 * cos_val - x1 * sin_val;
        data[i1] = x0 * sin_val + x1 * cos_val;
    }
}

/**
 * @brief Fused FP32 RoPE kernel for Q and K in single launch
 *
 * Processes both Q (first n_q_heads blocks) and K (next n_kv_heads blocks)
 * in a single kernel launch, reducing overhead by 50%.
 */
__global__ void rope_fp32_fused_qk_kernel(
    float *__restrict__ Q,
    float *__restrict__ K, // Can be nullptr
    const float *__restrict__ inv_freq,
    const int *__restrict__ position_ids,
    int seq_len,
    int n_q_heads,
    int n_kv_heads,
    int head_dim,
    int rotary_dim)
{
    const int half_rotary = rotary_dim / 2;

    extern __shared__ float smem[];
    float *s_cos = smem;
    float *s_sin = smem + half_rotary;

    // Total blocks: seq_len * n_q_heads (for Q) + seq_len * n_kv_heads (for K)
    int total_q_blocks = seq_len * n_q_heads;
    int block_idx = blockIdx.x;

    float *data;
    int n_heads;
    int local_block_idx;

    if (block_idx < total_q_blocks)
    {
        // This block processes Q
        data = Q;
        n_heads = n_q_heads;
        local_block_idx = block_idx;
    }
    else
    {
        // This block processes K
        if (K == nullptr)
            return;
        data = K;
        n_heads = n_kv_heads;
        local_block_idx = block_idx - total_q_blocks;
    }

    int head_idx = local_block_idx % n_heads;
    int seq_idx = local_block_idx / n_heads;

    if (seq_idx >= seq_len)
        return;

    int pos = position_ids ? position_ids[seq_idx] : seq_idx;
    if (pos < 0)
        return;

    // Compute sin/cos table
    for (int i = threadIdx.x; i < half_rotary; i += blockDim.x)
    {
        float angle = pos * inv_freq[i];
        __sincosf(angle, &s_sin[i], &s_cos[i]);
    }
    __syncthreads();

    // Apply rotation
    int base_idx = seq_idx * n_heads * head_dim + head_idx * head_dim;

    for (int i = threadIdx.x; i < half_rotary; i += blockDim.x)
    {
        int i0 = base_idx + i;
        int i1 = base_idx + i + half_rotary;

        float x0 = data[i0];
        float x1 = data[i1];

        data[i0] = x0 * s_cos[i] - x1 * s_sin[i];
        data[i1] = x0 * s_sin[i] + x1 * s_cos[i];
    }
}

/**
 * @brief BF16 RoPE kernel with shared memory sin/cos table
 */
__global__ void rope_bf16_kernel_v3(
    uint16_t *__restrict__ data,
    const float *__restrict__ inv_freq,
    const int *__restrict__ position_ids,
    int seq_len,
    int n_heads,
    int head_dim,
    int rotary_dim)
{
    const int half_rotary = rotary_dim / 2;

    extern __shared__ float smem[];
    float *s_cos = smem;
    float *s_sin = smem + half_rotary;

    int block_idx = blockIdx.x;
    int head_idx = block_idx % n_heads;
    int seq_idx = block_idx / n_heads;

    if (seq_idx >= seq_len)
        return;

    int pos = position_ids ? position_ids[seq_idx] : seq_idx;
    if (pos < 0)
        return;

    // Compute sin/cos table
    for (int i = threadIdx.x; i < half_rotary; i += blockDim.x)
    {
        float angle = pos * inv_freq[i];
        __sincosf(angle, &s_sin[i], &s_cos[i]);
    }
    __syncthreads();

    // Apply rotation
    int base_idx = seq_idx * n_heads * head_dim + head_idx * head_dim;

    for (int i = threadIdx.x; i < half_rotary; i += blockDim.x)
    {
        int i0 = base_idx + i;
        int i1 = base_idx + i + half_rotary;

        float x0 = bf16_to_float(data[i0]);
        float x1 = bf16_to_float(data[i1]);
        float cos_val = s_cos[i];
        float sin_val = s_sin[i];

        data[i0] = float_to_bf16(x0 * cos_val - x1 * sin_val);
        data[i1] = float_to_bf16(x0 * sin_val + x1 * cos_val);
    }
}

/**
 * @brief Fused BF16 RoPE kernel for Q and K in single launch
 */
__global__ void rope_bf16_fused_qk_kernel(
    uint16_t *__restrict__ Q,
    uint16_t *__restrict__ K,
    const float *__restrict__ inv_freq,
    const int *__restrict__ position_ids,
    int seq_len,
    int n_q_heads,
    int n_kv_heads,
    int head_dim,
    int rotary_dim)
{
    const int half_rotary = rotary_dim / 2;

    extern __shared__ float smem[];
    float *s_cos = smem;
    float *s_sin = smem + half_rotary;

    int total_q_blocks = seq_len * n_q_heads;
    int block_idx = blockIdx.x;

    uint16_t *data;
    int n_heads;
    int local_block_idx;

    if (block_idx < total_q_blocks)
    {
        data = Q;
        n_heads = n_q_heads;
        local_block_idx = block_idx;
    }
    else
    {
        if (K == nullptr)
            return;
        data = K;
        n_heads = n_kv_heads;
        local_block_idx = block_idx - total_q_blocks;
    }

    int head_idx = local_block_idx % n_heads;
    int seq_idx = local_block_idx / n_heads;

    if (seq_idx >= seq_len)
        return;

    int pos = position_ids ? position_ids[seq_idx] : seq_idx;
    if (pos < 0)
        return;

    for (int i = threadIdx.x; i < half_rotary; i += blockDim.x)
    {
        float angle = pos * inv_freq[i];
        __sincosf(angle, &s_sin[i], &s_cos[i]);
    }
    __syncthreads();

    int base_idx = seq_idx * n_heads * head_dim + head_idx * head_dim;

    for (int i = threadIdx.x; i < half_rotary; i += blockDim.x)
    {
        int i0 = base_idx + i;
        int i1 = base_idx + i + half_rotary;

        float x0 = bf16_to_float(data[i0]);
        float x1 = bf16_to_float(data[i1]);

        data[i0] = float_to_bf16(x0 * s_cos[i] - x1 * s_sin[i]);
        data[i1] = float_to_bf16(x0 * s_sin[i] + x1 * s_cos[i]);
    }
}

/**
 * @brief FP16 RoPE kernel with shared memory sin/cos table
 */
__global__ void rope_fp16_kernel_v3(
    uint16_t *__restrict__ data,
    const float *__restrict__ inv_freq,
    const int *__restrict__ position_ids,
    int seq_len,
    int n_heads,
    int head_dim,
    int rotary_dim)
{
    const int half_rotary = rotary_dim / 2;

    extern __shared__ float smem[];
    float *s_cos = smem;
    float *s_sin = smem + half_rotary;

    int block_idx = blockIdx.x;
    int head_idx = block_idx % n_heads;
    int seq_idx = block_idx / n_heads;

    if (seq_idx >= seq_len)
        return;

    int pos = position_ids ? position_ids[seq_idx] : seq_idx;
    if (pos < 0)
        return;

    // Compute sin/cos table
    for (int i = threadIdx.x; i < half_rotary; i += blockDim.x)
    {
        float angle = pos * inv_freq[i];
        __sincosf(angle, &s_sin[i], &s_cos[i]);
    }
    __syncthreads();

    // Apply rotation
    int base_idx = seq_idx * n_heads * head_dim + head_idx * head_dim;

    for (int i = threadIdx.x; i < half_rotary; i += blockDim.x)
    {
        int i0 = base_idx + i;
        int i1 = base_idx + i + half_rotary;

        float x0 = fp16_to_float(data[i0]);
        float x1 = fp16_to_float(data[i1]);
        float cos_val = s_cos[i];
        float sin_val = s_sin[i];

        data[i0] = float_to_fp16(x0 * cos_val - x1 * sin_val);
        data[i1] = float_to_fp16(x0 * sin_val + x1 * cos_val);
    }
}

/**
 * @brief Fused FP16 RoPE kernel for Q and K in single launch
 */
__global__ void rope_fp16_fused_qk_kernel(
    uint16_t *__restrict__ Q,
    uint16_t *__restrict__ K,
    const float *__restrict__ inv_freq,
    const int *__restrict__ position_ids,
    int seq_len,
    int n_q_heads,
    int n_kv_heads,
    int head_dim,
    int rotary_dim)
{
    const int half_rotary = rotary_dim / 2;

    extern __shared__ float smem[];
    float *s_cos = smem;
    float *s_sin = smem + half_rotary;

    int total_q_blocks = seq_len * n_q_heads;
    int block_idx = blockIdx.x;

    uint16_t *data;
    int n_heads;
    int local_block_idx;

    if (block_idx < total_q_blocks)
    {
        data = Q;
        n_heads = n_q_heads;
        local_block_idx = block_idx;
    }
    else
    {
        if (K == nullptr)
            return;
        data = K;
        n_heads = n_kv_heads;
        local_block_idx = block_idx - total_q_blocks;
    }

    int head_idx = local_block_idx % n_heads;
    int seq_idx = local_block_idx / n_heads;

    if (seq_idx >= seq_len)
        return;

    int pos = position_ids ? position_ids[seq_idx] : seq_idx;
    if (pos < 0)
        return;

    for (int i = threadIdx.x; i < half_rotary; i += blockDim.x)
    {
        float angle = pos * inv_freq[i];
        __sincosf(angle, &s_sin[i], &s_cos[i]);
    }
    __syncthreads();

    int base_idx = seq_idx * n_heads * head_dim + head_idx * head_dim;

    for (int i = threadIdx.x; i < half_rotary; i += blockDim.x)
    {
        int i0 = base_idx + i;
        int i1 = base_idx + i + half_rotary;

        float x0 = fp16_to_float(data[i0]);
        float x1 = fp16_to_float(data[i1]);

        data[i0] = float_to_fp16(x0 * s_cos[i] - x1 * s_sin[i]);
        data[i1] = float_to_fp16(x0 * s_sin[i] + x1 * s_cos[i]);
    }
}

// =========================================================================
// DECODE KERNELS (seq_len=1, scalar position - NO MEMCPY)
// =========================================================================

/**
 * @brief FP32 RoPE decode kernel - register-only sin/cos, no shared memory
 *
 * One block per head (Q + K). Each thread computes its own sin/cos in registers
 * and applies the rotation directly. No shared memory or __syncthreads needed
 * when half_rotary <= blockDim.x (each thread handles exactly one dim pair).
 */
__global__ void rope_fp32_decode_kernel(
    float *__restrict__ Q,
    float *__restrict__ K,
    const float *__restrict__ inv_freq,
    int pos,
    int n_q_heads,
    int n_kv_heads,
    int head_dim,
    int rotary_dim)
{
    const int half_rotary = rotary_dim / 2;
    int block_idx = blockIdx.x;

    float *data;
    int head_idx;
    if (block_idx < n_q_heads)
    {
        data = Q;
        head_idx = block_idx;
    }
    else
    {
        data = K;
        head_idx = block_idx - n_q_heads;
    }

    int base = head_idx * head_dim;

    for (int i = threadIdx.x; i < half_rotary; i += blockDim.x)
    {
        float angle = pos * inv_freq[i];
        float sin_val, cos_val;
        __sincosf(angle, &sin_val, &cos_val);

        float x0 = data[base + i];
        float x1 = data[base + i + half_rotary];
        data[base + i] = x0 * cos_val - x1 * sin_val;
        data[base + i + half_rotary] = x0 * sin_val + x1 * cos_val;
    }
}

/**
 * @brief BF16 RoPE decode kernel - register-only sin/cos, no shared memory
 */
__global__ void rope_bf16_decode_kernel(
    uint16_t *__restrict__ Q,
    uint16_t *__restrict__ K,
    const float *__restrict__ inv_freq,
    int pos,
    int n_q_heads,
    int n_kv_heads,
    int head_dim,
    int rotary_dim)
{
    const int half_rotary = rotary_dim / 2;
    int block_idx = blockIdx.x;

    uint16_t *data;
    int head_idx;
    if (block_idx < n_q_heads)
    {
        data = Q;
        head_idx = block_idx;
    }
    else
    {
        data = K;
        head_idx = block_idx - n_q_heads;
    }

    int base = head_idx * head_dim;

    for (int i = threadIdx.x; i < half_rotary; i += blockDim.x)
    {
        float angle = pos * inv_freq[i];
        float sin_val, cos_val;
        __sincosf(angle, &sin_val, &cos_val);

        float x0 = bf16_to_float(data[base + i]);
        float x1 = bf16_to_float(data[base + i + half_rotary]);
        data[base + i] = float_to_bf16(x0 * cos_val - x1 * sin_val);
        data[base + i + half_rotary] = float_to_bf16(x0 * sin_val + x1 * cos_val);
    }
}

/**
 * @brief FP16 RoPE decode kernel - register-only sin/cos, no shared memory
 */
__global__ void rope_fp16_decode_kernel(
    uint16_t *__restrict__ Q,
    uint16_t *__restrict__ K,
    const float *__restrict__ inv_freq,
    int pos,
    int n_q_heads,
    int n_kv_heads,
    int head_dim,
    int rotary_dim)
{
    const int half_rotary = rotary_dim / 2;
    int block_idx = blockIdx.x;

    uint16_t *data;
    int head_idx;
    if (block_idx < n_q_heads)
    {
        data = Q;
        head_idx = block_idx;
    }
    else
    {
        data = K;
        head_idx = block_idx - n_q_heads;
    }

    int base = head_idx * head_dim;

    for (int i = threadIdx.x; i < half_rotary; i += blockDim.x)
    {
        float angle = pos * inv_freq[i];
        float sin_val, cos_val;
        __sincosf(angle, &sin_val, &cos_val);

        float x0 = fp16_to_float(data[base + i]);
        float x1 = fp16_to_float(data[base + i + half_rotary]);
        data[base + i] = float_to_fp16(x0 * cos_val - x1 * sin_val);
        data[base + i + half_rotary] = float_to_fp16(x0 * sin_val + x1 * cos_val);
    }
}

// =========================================================================
// CONTIGUOUS KERNELS (positions computed on GPU - ZERO MEMCPY)
// =========================================================================

/**
 * @brief FP32 RoPE contiguous kernel - position computed from offset
 * Position is: pos_offset + seq_idx (no position_ids array needed)
 */
__global__ void rope_fp32_contiguous_kernel(
    float *__restrict__ Q,
    float *__restrict__ K,
    const float *__restrict__ inv_freq,
    int pos_offset,
    int seq_len,
    int n_q_heads,
    int n_kv_heads,
    int head_dim,
    int rotary_dim,
    const llaminar2::rope::RoPEDeviceParams *__restrict__ device_params)
{
    const int half_rotary = rotary_dim / 2;
    const int effective_pos_offset = (device_params) ? device_params->pos_offset : pos_offset;

    extern __shared__ float smem[];
    float *s_cos = smem;
    float *s_sin = smem + half_rotary;

    int total_q_blocks = seq_len * n_q_heads;
    int block_idx = blockIdx.x;

    float *data;
    int n_heads;
    int local_block_idx;

    if (block_idx < total_q_blocks)
    {
        data = Q;
        n_heads = n_q_heads;
        local_block_idx = block_idx;
    }
    else
    {
        if (K == nullptr)
            return;
        data = K;
        n_heads = n_kv_heads;
        local_block_idx = block_idx - total_q_blocks;
    }

    int head_idx = local_block_idx % n_heads;
    int seq_idx = local_block_idx / n_heads;

    if (seq_idx >= seq_len)
        return;

    // ZERO COPY: Position computed on GPU
    int pos = effective_pos_offset + seq_idx;

    for (int i = threadIdx.x; i < half_rotary; i += blockDim.x)
    {
        float angle = pos * inv_freq[i];
        __sincosf(angle, &s_sin[i], &s_cos[i]);
    }
    __syncthreads();

    int base_idx = seq_idx * n_heads * head_dim + head_idx * head_dim;

    for (int i = threadIdx.x; i < half_rotary; i += blockDim.x)
    {
        int i0 = base_idx + i;
        int i1 = base_idx + i + half_rotary;

        float x0 = data[i0];
        float x1 = data[i1];

        data[i0] = x0 * s_cos[i] - x1 * s_sin[i];
        data[i1] = x0 * s_sin[i] + x1 * s_cos[i];
    }
}

/**
 * @brief BF16 RoPE contiguous kernel - position computed from offset
 */
__global__ void rope_bf16_contiguous_kernel(
    uint16_t *__restrict__ Q,
    uint16_t *__restrict__ K,
    const float *__restrict__ inv_freq,
    int pos_offset,
    int seq_len,
    int n_q_heads,
    int n_kv_heads,
    int head_dim,
    int rotary_dim,
    const llaminar2::rope::RoPEDeviceParams *__restrict__ device_params)
{
    const int half_rotary = rotary_dim / 2;
    const int effective_pos_offset = (device_params) ? device_params->pos_offset : pos_offset;

    extern __shared__ float smem[];
    float *s_cos = smem;
    float *s_sin = smem + half_rotary;

    int total_q_blocks = seq_len * n_q_heads;
    int block_idx = blockIdx.x;

    uint16_t *data;
    int n_heads;
    int local_block_idx;

    if (block_idx < total_q_blocks)
    {
        data = Q;
        n_heads = n_q_heads;
        local_block_idx = block_idx;
    }
    else
    {
        if (K == nullptr)
            return;
        data = K;
        n_heads = n_kv_heads;
        local_block_idx = block_idx - total_q_blocks;
    }

    int head_idx = local_block_idx % n_heads;
    int seq_idx = local_block_idx / n_heads;

    if (seq_idx >= seq_len)
        return;

    int pos = effective_pos_offset + seq_idx;

    for (int i = threadIdx.x; i < half_rotary; i += blockDim.x)
    {
        float angle = pos * inv_freq[i];
        __sincosf(angle, &s_sin[i], &s_cos[i]);
    }
    __syncthreads();

    int base_idx = seq_idx * n_heads * head_dim + head_idx * head_dim;

    for (int i = threadIdx.x; i < half_rotary; i += blockDim.x)
    {
        int i0 = base_idx + i;
        int i1 = base_idx + i + half_rotary;

        float x0 = bf16_to_float(data[i0]);
        float x1 = bf16_to_float(data[i1]);

        data[i0] = float_to_bf16(x0 * s_cos[i] - x1 * s_sin[i]);
        data[i1] = float_to_bf16(x0 * s_sin[i] + x1 * s_cos[i]);
    }
}

/**
 * @brief FP16 RoPE contiguous kernel - position computed from offset
 */
__global__ void rope_fp16_contiguous_kernel(
    uint16_t *__restrict__ Q,
    uint16_t *__restrict__ K,
    const float *__restrict__ inv_freq,
    int pos_offset,
    int seq_len,
    int n_q_heads,
    int n_kv_heads,
    int head_dim,
    int rotary_dim,
    const llaminar2::rope::RoPEDeviceParams *__restrict__ device_params)
{
    const int half_rotary = rotary_dim / 2;
    const int effective_pos_offset = (device_params) ? device_params->pos_offset : pos_offset;

    extern __shared__ float smem[];
    float *s_cos = smem;
    float *s_sin = smem + half_rotary;

    int total_q_blocks = seq_len * n_q_heads;
    int block_idx = blockIdx.x;

    uint16_t *data;
    int n_heads;
    int local_block_idx;

    if (block_idx < total_q_blocks)
    {
        data = Q;
        n_heads = n_q_heads;
        local_block_idx = block_idx;
    }
    else
    {
        if (K == nullptr)
            return;
        data = K;
        n_heads = n_kv_heads;
        local_block_idx = block_idx - total_q_blocks;
    }

    int head_idx = local_block_idx % n_heads;
    int seq_idx = local_block_idx / n_heads;

    if (seq_idx >= seq_len)
        return;

    int pos = effective_pos_offset + seq_idx;

    for (int i = threadIdx.x; i < half_rotary; i += blockDim.x)
    {
        float angle = pos * inv_freq[i];
        __sincosf(angle, &s_sin[i], &s_cos[i]);
    }
    __syncthreads();

    int base_idx = seq_idx * n_heads * head_dim + head_idx * head_dim;

    for (int i = threadIdx.x; i < half_rotary; i += blockDim.x)
    {
        int i0 = base_idx + i;
        int i1 = base_idx + i + half_rotary;

        float x0 = fp16_to_float(data[i0]);
        float x1 = fp16_to_float(data[i1]);

        data[i0] = float_to_fp16(x0 * s_cos[i] - x1 * s_sin[i]);
        data[i1] = float_to_fp16(x0 * s_sin[i] + x1 * s_cos[i]);
    }
}

// =========================================================================
// Extern "C" Wrapper Functions
// =========================================================================

/**
 * @brief Publish one contiguous-position offset into graph-stable device state.
 *
 * The offset is passed to this kernel by value. CUDA copies kernel arguments
 * when the launch is enqueued, so a later host-side request update cannot
 * mutate an earlier publication that is still waiting in the same stream.
 */
__global__ void cuda_rope_publish_device_params_kernel(
    llaminar2::rope::RoPEDeviceParams *__restrict__ device_params,
    int pos_offset)
{
    device_params->pos_offset = pos_offset;
}

extern "C"
{
    // =========================================================================
    // WORKSPACE-AWARE API (v3)
    // These functions take external inv_freq buffer allocated from workspace
    // =========================================================================

    /**
     * @brief Enqueue a device-owned RoPE position publication.
     *
     * This deliberately uses a kernel instead of an asynchronous H2D copy from
     * mutable pinned host staging. The launch is ordered by @p stream and its
     * scalar argument has value semantics, which makes back-to-back graph
     * launches safe without a host wait or a staging-buffer lease.
     */
    bool cudaOps_rope_publish_device_params(
        llaminar2::rope::RoPEDeviceParams *device_params,
        int pos_offset,
        int device_idx,
        cudaStream_t stream)
    {
        if (!device_params || !stream)
            return false;

        if (cudaSetDevice(device_idx) != cudaSuccess)
            return false;

        cuda_rope_publish_device_params_kernel<<<1, 1, 0, stream>>>(
            device_params,
            pos_offset);
        return ropeLaunchOk("cudaOps_rope_publish_device_params");
    }

    /**
     * @brief Populate inverse frequency table in an external buffer
     * @param d_inv_freq Device buffer (must be at least half_dim * sizeof(float))
     * @param head_dim The head dimension
     * @param freq_base The frequency base (rope_theta)
     * @param device_idx CUDA device index
     * @return true on success
     *
     * Formula: inv_freq[i] = 1.0 / (freq_base^(2i/head_dim))
     */
    bool cudaOps_rope_populate_inv_freq(
        float *d_inv_freq,
        int head_dim,
        float freq_base,
        int device_idx,
        cudaStream_t stream)
    {
        if (!d_inv_freq)
            return false;

        const int half_dim = head_dim / 2;

        // Compute on host
        std::vector<float> h_inv_freq(half_dim);
        const float log_base = std::log(freq_base);
        for (int i = 0; i < half_dim; ++i)
        {
            float exponent = (2.0f * i) / head_dim;
            h_inv_freq[i] = std::exp(-log_base * exponent);
        }

        // Copy to device
        cudaSetDevice(device_idx);
        cudaError_t err = cudaMemcpyAsync(d_inv_freq, h_inv_freq.data(),
                                          half_dim * sizeof(float), cudaMemcpyHostToDevice, stream);
        return err == cudaSuccess;
    }

    /**
     * @brief FP32 RoPE with external inv_freq buffer (workspace-aware)
     */
    bool cudaOps_rope_fp32_v3(
        float *Q,
        float *K,
        const float *d_inv_freq,
        const int *position_ids,
        int seq_len,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        int rotary_dim,
        int device_idx,
        cudaStream_t stream)
    {
        if (!d_inv_freq)
            return false;

        cudaSetDevice(device_idx);

        const int half_rotary = rotary_dim / 2;
        const int threads_per_block = min(256, half_rotary);
        const size_t smem_size = 2 * half_rotary * sizeof(float);

        if (K != nullptr)
        {
            int total_blocks = seq_len * (n_heads + n_kv_heads);
            rope_fp32_fused_qk_kernel<<<total_blocks, threads_per_block, smem_size, stream>>>(
                Q, K, d_inv_freq, position_ids, seq_len, n_heads, n_kv_heads, head_dim, rotary_dim);
        }
        else
        {
            int num_blocks_q = seq_len * n_heads;
            rope_fp32_kernel_v3<<<num_blocks_q, threads_per_block, smem_size, stream>>>(
                Q, d_inv_freq, position_ids, seq_len, n_heads, head_dim, rotary_dim);
        }

        return ropeLaunchOk("cudaOps_rope_fp32_v3");
    }

    /**
     * @brief FP32 RoPE decode with external inv_freq buffer (workspace-aware)
     */
    bool cudaOps_rope_fp32_decode_v3(
        float *Q,
        float *K,
        const float *d_inv_freq,
        int pos,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        int rotary_dim,
        int device_idx,
        cudaStream_t stream)
    {
        if (!d_inv_freq)
            return false;

        cudaSetDevice(device_idx);

        const int half_rotary = rotary_dim / 2;

        // One block per head, register-only sin/cos (no shared memory)
        int total_blocks = n_heads + (K ? n_kv_heads : 0);
        int threads_per_block = min(256, half_rotary);
        rope_fp32_decode_kernel<<<total_blocks, threads_per_block, 0, stream>>>(
            Q, K, d_inv_freq, pos, n_heads, n_kv_heads, head_dim, rotary_dim);

        return ropeLaunchOk("cudaOps_rope_fp32_decode_v3");
    }

    /**
     * @brief FP32 RoPE contiguous with external inv_freq buffer (workspace-aware)
     */
    bool cudaOps_rope_fp32_contiguous_v3(
        float *Q,
        float *K,
        const float *d_inv_freq,
        int pos_offset,
        int seq_len,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        int rotary_dim,
        int device_idx,
        cudaStream_t stream,
        const llaminar2::rope::RoPEDeviceParams *device_params)
    {
        if (!d_inv_freq)
            return false;

        cudaSetDevice(device_idx);

        const int half_rotary = rotary_dim / 2;
        const int threads_per_block = min(256, half_rotary);
        const size_t smem_size = 2 * half_rotary * sizeof(float);

        int total_blocks = seq_len * (n_heads + (K ? n_kv_heads : 0));

        rope_fp32_contiguous_kernel<<<total_blocks, threads_per_block, smem_size, stream>>>(
            Q, K, d_inv_freq, pos_offset, seq_len, n_heads, n_kv_heads, head_dim, rotary_dim, device_params);

        return ropeLaunchOk("cudaOps_rope_fp32_contiguous_v3");
    }

    /**
     * @brief BF16 RoPE with external inv_freq buffer (workspace-aware)
     */
    bool cudaOps_rope_bf16_v3(
        uint16_t *Q,
        uint16_t *K,
        const float *d_inv_freq,
        const int *position_ids,
        int seq_len,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        int rotary_dim,
        int device_idx,
        cudaStream_t stream)
    {
        if (!d_inv_freq)
            return false;

        cudaSetDevice(device_idx);

        const int half_rotary = rotary_dim / 2;
        const int threads_per_block = min(256, half_rotary);
        const size_t smem_size = 2 * half_rotary * sizeof(float);

        if (K != nullptr)
        {
            int total_blocks = seq_len * (n_heads + n_kv_heads);
            rope_bf16_fused_qk_kernel<<<total_blocks, threads_per_block, smem_size, stream>>>(
                Q, K, d_inv_freq, position_ids, seq_len, n_heads, n_kv_heads, head_dim, rotary_dim);
        }
        else
        {
            int num_blocks = seq_len * n_heads;
            rope_bf16_kernel_v3<<<num_blocks, threads_per_block, smem_size, stream>>>(
                Q, d_inv_freq, position_ids, seq_len, n_heads, head_dim, rotary_dim);
        }
        return ropeLaunchOk("cudaOps_rope_bf16_v3");
    }

    /**
     * @brief BF16 RoPE decode with external inv_freq buffer (workspace-aware)
     */
    bool cudaOps_rope_bf16_decode_v3(
        uint16_t *Q,
        uint16_t *K,
        const float *d_inv_freq,
        int pos,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        int rotary_dim,
        int device_idx,
        cudaStream_t stream)
    {
        if (!d_inv_freq)
            return false;

        cudaSetDevice(device_idx);

        const int half_rotary = rotary_dim / 2;

        int total_blocks = n_heads + (K ? n_kv_heads : 0);
        int threads_per_block = min(256, half_rotary);
        rope_bf16_decode_kernel<<<total_blocks, threads_per_block, 0, stream>>>(
            Q, K, d_inv_freq, pos, n_heads, n_kv_heads, head_dim, rotary_dim);

        return ropeLaunchOk("cudaOps_rope_bf16_decode_v3");
    }

    /**
     * @brief BF16 RoPE contiguous with external inv_freq buffer (workspace-aware)
     */
    bool cudaOps_rope_bf16_contiguous_v3(
        uint16_t *Q,
        uint16_t *K,
        const float *d_inv_freq,
        int pos_offset,
        int seq_len,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        int rotary_dim,
        int device_idx,
        cudaStream_t stream,
        const llaminar2::rope::RoPEDeviceParams *device_params)
    {
        if (!d_inv_freq)
            return false;

        cudaSetDevice(device_idx);

        const int half_rotary = rotary_dim / 2;
        const int threads_per_block = min(256, half_rotary);
        const size_t smem_size = 2 * half_rotary * sizeof(float);

        int total_blocks = seq_len * (n_heads + (K ? n_kv_heads : 0));

        rope_bf16_contiguous_kernel<<<total_blocks, threads_per_block, smem_size, stream>>>(
            Q, K, d_inv_freq, pos_offset, seq_len, n_heads, n_kv_heads, head_dim, rotary_dim, device_params);

        return ropeLaunchOk("cudaOps_rope_bf16_contiguous_v3");
    }

    /**
     * @brief FP16 RoPE with external inv_freq buffer (workspace-aware)
     */
    bool cudaOps_rope_fp16_v3(
        uint16_t *Q,
        uint16_t *K,
        const float *d_inv_freq,
        const int *position_ids,
        int seq_len,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        int rotary_dim,
        int device_idx,
        cudaStream_t stream)
    {
        if (!d_inv_freq)
            return false;

        cudaSetDevice(device_idx);

        const int half_rotary = rotary_dim / 2;
        const int threads_per_block = min(256, half_rotary);
        const size_t smem_size = 2 * half_rotary * sizeof(float);

        if (K != nullptr)
        {
            int total_blocks = seq_len * (n_heads + n_kv_heads);
            rope_fp16_fused_qk_kernel<<<total_blocks, threads_per_block, smem_size, stream>>>(
                Q, K, d_inv_freq, position_ids, seq_len, n_heads, n_kv_heads, head_dim, rotary_dim);
        }
        else
        {
            int num_blocks = seq_len * n_heads;
            rope_fp16_kernel_v3<<<num_blocks, threads_per_block, smem_size, stream>>>(
                Q, d_inv_freq, position_ids, seq_len, n_heads, head_dim, rotary_dim);
        }
        return ropeLaunchOk("cudaOps_rope_fp16_v3");
    }

    /**
     * @brief FP16 RoPE decode with external inv_freq buffer (workspace-aware)
     */
    bool cudaOps_rope_fp16_decode_v3(
        uint16_t *Q,
        uint16_t *K,
        const float *d_inv_freq,
        int pos,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        int rotary_dim,
        int device_idx,
        cudaStream_t stream)
    {
        if (!d_inv_freq)
            return false;

        cudaSetDevice(device_idx);

        const int half_rotary = rotary_dim / 2;

        int total_blocks = n_heads + (K ? n_kv_heads : 0);
        int threads_per_block = min(256, half_rotary);
        rope_fp16_decode_kernel<<<total_blocks, threads_per_block, 0, stream>>>(
            Q, K, d_inv_freq, pos, n_heads, n_kv_heads, head_dim, rotary_dim);

        return ropeLaunchOk("cudaOps_rope_fp16_decode_v3");
    }

    /**
     * @brief FP16 RoPE contiguous with external inv_freq buffer (workspace-aware)
     */
    bool cudaOps_rope_fp16_contiguous_v3(
        uint16_t *Q,
        uint16_t *K,
        const float *d_inv_freq,
        int pos_offset,
        int seq_len,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        int rotary_dim,
        int device_idx,
        cudaStream_t stream,
        const llaminar2::rope::RoPEDeviceParams *device_params)
    {
        if (!d_inv_freq)
            return false;

        cudaSetDevice(device_idx);

        const int half_rotary = rotary_dim / 2;
        const int threads_per_block = min(256, half_rotary);
        const size_t smem_size = 2 * half_rotary * sizeof(float);

        int total_blocks = seq_len * (n_heads + (K ? n_kv_heads : 0));

        rope_fp16_contiguous_kernel<<<total_blocks, threads_per_block, smem_size, stream>>>(
            Q, K, d_inv_freq, pos_offset, seq_len, n_heads, n_kv_heads, head_dim, rotary_dim, device_params);

        return ropeLaunchOk("cudaOps_rope_fp16_contiguous_v3");
    }

} // extern "C"
