/**
 * @file ROCmRoPEKernels.hip
 * @brief HIP RoPE (Rotary Position Embedding) kernel implementations for ROCm
 * @author Llaminar Team
 *
 * Contains FP32, BF16, and FP16 RoPE kernels with extern "C" wrapper functions.
 * Uses SPLIT-HALF layout matching the CPU implementation.
 *
 * OPTIMIZATIONS (v2 - matching CUDA strategy):
 * - Pre-computed inverse frequency table passed from host (cached in caller)
 * - Shared memory for sin/cos table per thread block
 * - Main rotation is pure FMA (no per-thread transcendentals)
 * - Fused Q+K kernel to reduce launch overhead by 50%
 * - Scalar position path for decode (avoids memcpy for seq_len=1)
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <vector>

#include "../../../backends/rocm/HipDeviceGuard.h"

namespace
{

// =========================================================================
// BF16 Conversion Helpers (HIP compatible)
// =========================================================================

__device__ __forceinline__ float bf16_to_float(uint16_t bf16)
{
    uint32_t bits = static_cast<uint32_t>(bf16) << 16;
    float result;
    memcpy(&result, &bits, sizeof(float));
    return result;
}

__device__ __forceinline__ uint16_t float_to_bf16(float f)
{
    uint32_t bits;
    memcpy(&bits, &f, sizeof(float));
    // Round to nearest even
    bits += 0x7FFF + ((bits >> 16) & 1);
    return static_cast<uint16_t>(bits >> 16);
}

// =========================================================================
// FP16 Conversion Helpers (using HIP intrinsics)
// =========================================================================

__device__ __forceinline__ float fp16_to_float(uint16_t fp16)
{
    __half h;
    memcpy(&h, &fp16, sizeof(__half));
    return __half2float(h);
}

__device__ __forceinline__ uint16_t float_to_fp16(float f)
{
    __half h = __float2half(f);
    uint16_t result;
    memcpy(&result, &h, sizeof(uint16_t));
    return result;
}

} // anonymous namespace

// =========================================================================
// Decode-Optimized Kernels (scalar position, no position_ids array)
// =========================================================================

/**
 * @brief FP32 RoPE decode kernel - optimized for seq_len=1
 *
 * Takes a single position value instead of position_ids array, eliminating
 * the need for device memory allocation and H2D copy during decode.
 * All heads process the same position.
 */
__global__ void rocm_rope_fp32_decode_kernel(
    float *__restrict__ data,
    const float *__restrict__ inv_freq,
    int pos,        // Single position value (not array)
    int n_heads,
    int head_dim)
{
    const int half_dim = head_dim / 2;

    // Each block handles one head
    int head_idx = blockIdx.x;
    if (head_idx >= n_heads)
        return;

    // Shared memory for sin/cos cache
    extern __shared__ float smem[];
    float *s_cos = smem;
    float *s_sin = smem + half_dim;

    // Step 1: Cooperatively compute sin/cos table
    for (int i = threadIdx.x; i < half_dim; i += blockDim.x)
    {
        float angle = pos * inv_freq[i];
        s_cos[i] = cosf(angle);
        s_sin[i] = sinf(angle);
    }
    __syncthreads();

    // Step 2: Apply rotation (seq_idx=0 for decode)
    int base_idx = head_idx * head_dim;

    for (int i = threadIdx.x; i < half_dim; i += blockDim.x)
    {
        int i0 = base_idx + i;
        int i1 = base_idx + i + half_dim;

        float x0 = data[i0];
        float x1 = data[i1];

        data[i0] = x0 * s_cos[i] - x1 * s_sin[i];
        data[i1] = x0 * s_sin[i] + x1 * s_cos[i];
    }
}

/**
 * @brief BF16 RoPE decode kernel - optimized for seq_len=1
 */
__global__ void rocm_rope_bf16_decode_kernel(
    uint16_t *__restrict__ data,
    const float *__restrict__ inv_freq,
    int pos,
    int n_heads,
    int head_dim)
{
    const int half_dim = head_dim / 2;

    int head_idx = blockIdx.x;
    if (head_idx >= n_heads)
        return;

    extern __shared__ float smem[];
    float *s_cos = smem;
    float *s_sin = smem + half_dim;

    for (int i = threadIdx.x; i < half_dim; i += blockDim.x)
    {
        float angle = pos * inv_freq[i];
        s_cos[i] = cosf(angle);
        s_sin[i] = sinf(angle);
    }
    __syncthreads();

    int base_idx = head_idx * head_dim;

    for (int i = threadIdx.x; i < half_dim; i += blockDim.x)
    {
        int i0 = base_idx + i;
        int i1 = base_idx + i + half_dim;

        float x0 = bf16_to_float(data[i0]);
        float x1 = bf16_to_float(data[i1]);

        data[i0] = float_to_bf16(x0 * s_cos[i] - x1 * s_sin[i]);
        data[i1] = float_to_bf16(x0 * s_sin[i] + x1 * s_cos[i]);
    }
}

/**
 * @brief FP16 RoPE decode kernel - optimized for seq_len=1
 */
__global__ void rocm_rope_fp16_decode_kernel(
    uint16_t *__restrict__ data,
    const float *__restrict__ inv_freq,
    int pos,
    int n_heads,
    int head_dim)
{
    const int half_dim = head_dim / 2;

    int head_idx = blockIdx.x;
    if (head_idx >= n_heads)
        return;

    extern __shared__ float smem[];
    float *s_cos = smem;
    float *s_sin = smem + half_dim;

    for (int i = threadIdx.x; i < half_dim; i += blockDim.x)
    {
        float angle = pos * inv_freq[i];
        s_cos[i] = cosf(angle);
        s_sin[i] = sinf(angle);
    }
    __syncthreads();

    int base_idx = head_idx * head_dim;

    for (int i = threadIdx.x; i < half_dim; i += blockDim.x)
    {
        int i0 = base_idx + i;
        int i1 = base_idx + i + half_dim;

        float x0 = fp16_to_float(data[i0]);
        float x1 = fp16_to_float(data[i1]);

        data[i0] = float_to_fp16(x0 * s_cos[i] - x1 * s_sin[i]);
        data[i1] = float_to_fp16(x0 * s_sin[i] + x1 * s_cos[i]);
    }
}

// =========================================================================
// Contiguous Position Kernels (zero-copy: positions computed on GPU)
// =========================================================================

/**
 * @brief FP32 RoPE kernel with contiguous positions computed on-the-fly
 *
 * Eliminates host-to-device copy by computing pos = pos_offset + seq_idx on GPU.
 * This is the optimal path for most inference scenarios.
 */
__global__ void rocm_rope_fp32_contiguous_kernel(
    float *__restrict__ data,
    const float *__restrict__ inv_freq,
    int pos_offset,  // Starting position (positions are pos_offset + seq_idx)
    int seq_len,
    int n_heads,
    int head_dim)
{
    const int half_dim = head_dim / 2;

    extern __shared__ float smem[];
    float *s_cos = smem;
    float *s_sin = smem + half_dim;

    int block_idx = blockIdx.x;
    int head_idx = block_idx % n_heads;
    int seq_idx = block_idx / n_heads;

    if (seq_idx >= seq_len)
        return;

    // Compute position on-the-fly (no memory access!)
    int pos = pos_offset + seq_idx;

    // Compute sin/cos table
    for (int i = threadIdx.x; i < half_dim; i += blockDim.x)
    {
        float angle = pos * inv_freq[i];
        s_cos[i] = cosf(angle);
        s_sin[i] = sinf(angle);
    }
    __syncthreads();

    // Apply rotation
    int base_idx = seq_idx * n_heads * head_dim + head_idx * head_dim;

    for (int i = threadIdx.x; i < half_dim; i += blockDim.x)
    {
        int i0 = base_idx + i;
        int i1 = base_idx + i + half_dim;

        float x0 = data[i0];
        float x1 = data[i1];

        data[i0] = x0 * s_cos[i] - x1 * s_sin[i];
        data[i1] = x0 * s_sin[i] + x1 * s_cos[i];
    }
}

/**
 * @brief Fused FP32 Q+K RoPE kernel with contiguous positions
 */
__global__ void rocm_rope_fp32_fused_qk_contiguous_kernel(
    float *__restrict__ Q,
    float *__restrict__ K,
    const float *__restrict__ inv_freq,
    int pos_offset,
    int seq_len,
    int n_q_heads,
    int n_kv_heads,
    int head_dim)
{
    const int half_dim = head_dim / 2;

    extern __shared__ float smem[];
    float *s_cos = smem;
    float *s_sin = smem + half_dim;

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

    int pos = pos_offset + seq_idx;

    for (int i = threadIdx.x; i < half_dim; i += blockDim.x)
    {
        float angle = pos * inv_freq[i];
        s_cos[i] = cosf(angle);
        s_sin[i] = sinf(angle);
    }
    __syncthreads();

    int base_idx = seq_idx * n_heads * head_dim + head_idx * head_dim;

    for (int i = threadIdx.x; i < half_dim; i += blockDim.x)
    {
        int i0 = base_idx + i;
        int i1 = base_idx + i + half_dim;

        float x0 = data[i0];
        float x1 = data[i1];

        data[i0] = x0 * s_cos[i] - x1 * s_sin[i];
        data[i1] = x0 * s_sin[i] + x1 * s_cos[i];
    }
}

/**
 * @brief BF16 RoPE kernel with contiguous positions
 */
__global__ void rocm_rope_bf16_contiguous_kernel(
    uint16_t *__restrict__ data,
    const float *__restrict__ inv_freq,
    int pos_offset,
    int seq_len,
    int n_heads,
    int head_dim)
{
    const int half_dim = head_dim / 2;

    extern __shared__ float smem[];
    float *s_cos = smem;
    float *s_sin = smem + half_dim;

    int block_idx = blockIdx.x;
    int head_idx = block_idx % n_heads;
    int seq_idx = block_idx / n_heads;

    if (seq_idx >= seq_len)
        return;

    int pos = pos_offset + seq_idx;

    for (int i = threadIdx.x; i < half_dim; i += blockDim.x)
    {
        float angle = pos * inv_freq[i];
        s_cos[i] = cosf(angle);
        s_sin[i] = sinf(angle);
    }
    __syncthreads();

    int base_idx = seq_idx * n_heads * head_dim + head_idx * head_dim;

    for (int i = threadIdx.x; i < half_dim; i += blockDim.x)
    {
        int i0 = base_idx + i;
        int i1 = base_idx + i + half_dim;

        float x0 = bf16_to_float(data[i0]);
        float x1 = bf16_to_float(data[i1]);

        data[i0] = float_to_bf16(x0 * s_cos[i] - x1 * s_sin[i]);
        data[i1] = float_to_bf16(x0 * s_sin[i] + x1 * s_cos[i]);
    }
}

/**
 * @brief Fused BF16 Q+K RoPE kernel with contiguous positions
 */
__global__ void rocm_rope_bf16_fused_qk_contiguous_kernel(
    uint16_t *__restrict__ Q,
    uint16_t *__restrict__ K,
    const float *__restrict__ inv_freq,
    int pos_offset,
    int seq_len,
    int n_q_heads,
    int n_kv_heads,
    int head_dim)
{
    const int half_dim = head_dim / 2;

    extern __shared__ float smem[];
    float *s_cos = smem;
    float *s_sin = smem + half_dim;

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

    int pos = pos_offset + seq_idx;

    for (int i = threadIdx.x; i < half_dim; i += blockDim.x)
    {
        float angle = pos * inv_freq[i];
        s_cos[i] = cosf(angle);
        s_sin[i] = sinf(angle);
    }
    __syncthreads();

    int base_idx = seq_idx * n_heads * head_dim + head_idx * head_dim;

    for (int i = threadIdx.x; i < half_dim; i += blockDim.x)
    {
        int i0 = base_idx + i;
        int i1 = base_idx + i + half_dim;

        float x0 = bf16_to_float(data[i0]);
        float x1 = bf16_to_float(data[i1]);

        data[i0] = float_to_bf16(x0 * s_cos[i] - x1 * s_sin[i]);
        data[i1] = float_to_bf16(x0 * s_sin[i] + x1 * s_cos[i]);
    }
}

/**
 * @brief FP16 RoPE kernel with contiguous positions
 */
__global__ void rocm_rope_fp16_contiguous_kernel(
    uint16_t *__restrict__ data,
    const float *__restrict__ inv_freq,
    int pos_offset,
    int seq_len,
    int n_heads,
    int head_dim)
{
    const int half_dim = head_dim / 2;

    extern __shared__ float smem[];
    float *s_cos = smem;
    float *s_sin = smem + half_dim;

    int block_idx = blockIdx.x;
    int head_idx = block_idx % n_heads;
    int seq_idx = block_idx / n_heads;

    if (seq_idx >= seq_len)
        return;

    int pos = pos_offset + seq_idx;

    for (int i = threadIdx.x; i < half_dim; i += blockDim.x)
    {
        float angle = pos * inv_freq[i];
        s_cos[i] = cosf(angle);
        s_sin[i] = sinf(angle);
    }
    __syncthreads();

    int base_idx = seq_idx * n_heads * head_dim + head_idx * head_dim;

    for (int i = threadIdx.x; i < half_dim; i += blockDim.x)
    {
        int i0 = base_idx + i;
        int i1 = base_idx + i + half_dim;

        float x0 = fp16_to_float(data[i0]);
        float x1 = fp16_to_float(data[i1]);

        data[i0] = float_to_fp16(x0 * s_cos[i] - x1 * s_sin[i]);
        data[i1] = float_to_fp16(x0 * s_sin[i] + x1 * s_cos[i]);
    }
}

/**
 * @brief Fused FP16 Q+K RoPE kernel with contiguous positions
 */
__global__ void rocm_rope_fp16_fused_qk_contiguous_kernel(
    uint16_t *__restrict__ Q,
    uint16_t *__restrict__ K,
    const float *__restrict__ inv_freq,
    int pos_offset,
    int seq_len,
    int n_q_heads,
    int n_kv_heads,
    int head_dim)
{
    const int half_dim = head_dim / 2;

    extern __shared__ float smem[];
    float *s_cos = smem;
    float *s_sin = smem + half_dim;

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

    int pos = pos_offset + seq_idx;

    for (int i = threadIdx.x; i < half_dim; i += blockDim.x)
    {
        float angle = pos * inv_freq[i];
        s_cos[i] = cosf(angle);
        s_sin[i] = sinf(angle);
    }
    __syncthreads();

    int base_idx = seq_idx * n_heads * head_dim + head_idx * head_dim;

    for (int i = threadIdx.x; i < half_dim; i += blockDim.x)
    {
        int i0 = base_idx + i;
        int i1 = base_idx + i + half_dim;

        float x0 = fp16_to_float(data[i0]);
        float x1 = fp16_to_float(data[i1]);

        data[i0] = float_to_fp16(x0 * s_cos[i] - x1 * s_sin[i]);
        data[i1] = float_to_fp16(x0 * s_sin[i] + x1 * s_cos[i]);
    }
}

// =========================================================================
// Optimized RoPE HIP Kernels (v2 - with precomputed inv_freq and shared mem)
// =========================================================================

/**
 * @brief FP32 RoPE kernel with precomputed inv_freq and shared memory sin/cos
 *
 * Strategy (adapted from CUDA v3):
 * 1. Each thread block processes one (seq_idx, head_idx) pair
 * 2. Threads cooperatively compute sin/cos table into shared memory
 * 3. Apply rotation using cached sin/cos (pure FMA, no transcendentals per element)
 *
 * @param data Input/output tensor [seq_len, n_heads, head_dim]
 * @param inv_freq Pre-computed inverse frequencies [half_dim]
 * @param position_ids Position IDs [seq_len] (device pointer)
 * @param seq_len Sequence length
 * @param n_heads Number of heads
 * @param head_dim Head dimension
 */
__global__ void rocm_rope_fp32_kernel_v2(
    float *__restrict__ data,
    const float *__restrict__ inv_freq,
    const int *__restrict__ position_ids,
    int seq_len,
    int n_heads,
    int head_dim)
{
    const int half_dim = head_dim / 2;

    // Shared memory for sin/cos cache
    extern __shared__ float smem[];
    float *s_cos = smem;
    float *s_sin = smem + half_dim;

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
    for (int i = threadIdx.x; i < half_dim; i += blockDim.x)
    {
        float angle = pos * inv_freq[i];
        s_cos[i] = cosf(angle);
        s_sin[i] = sinf(angle);
    }
    __syncthreads();

    // Step 2: Apply rotation using cached sin/cos (pure FMA)
    int base_idx = seq_idx * n_heads * head_dim + head_idx * head_dim;

    for (int i = threadIdx.x; i < half_dim; i += blockDim.x)
    {
        int i0 = base_idx + i;
        int i1 = base_idx + i + half_dim;

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
__global__ void rocm_rope_fp32_fused_qk_kernel(
    float *__restrict__ Q,
    float *__restrict__ K,
    const float *__restrict__ inv_freq,
    const int *__restrict__ position_ids,
    int seq_len,
    int n_q_heads,
    int n_kv_heads,
    int head_dim)
{
    const int half_dim = head_dim / 2;

    extern __shared__ float smem[];
    float *s_cos = smem;
    float *s_sin = smem + half_dim;

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
    for (int i = threadIdx.x; i < half_dim; i += blockDim.x)
    {
        float angle = pos * inv_freq[i];
        s_cos[i] = cosf(angle);
        s_sin[i] = sinf(angle);
    }
    __syncthreads();

    // Apply rotation
    int base_idx = seq_idx * n_heads * head_dim + head_idx * head_dim;

    for (int i = threadIdx.x; i < half_dim; i += blockDim.x)
    {
        int i0 = base_idx + i;
        int i1 = base_idx + i + half_dim;

        float x0 = data[i0];
        float x1 = data[i1];

        data[i0] = x0 * s_cos[i] - x1 * s_sin[i];
        data[i1] = x0 * s_sin[i] + x1 * s_cos[i];
    }
}

/**
 * @brief BF16 RoPE kernel with precomputed inv_freq and shared memory
 */
__global__ void rocm_rope_bf16_kernel_v2(
    uint16_t *__restrict__ data,
    const float *__restrict__ inv_freq,
    const int *__restrict__ position_ids,
    int seq_len,
    int n_heads,
    int head_dim)
{
    const int half_dim = head_dim / 2;

    extern __shared__ float smem[];
    float *s_cos = smem;
    float *s_sin = smem + half_dim;

    int block_idx = blockIdx.x;
    int head_idx = block_idx % n_heads;
    int seq_idx = block_idx / n_heads;

    if (seq_idx >= seq_len)
        return;

    int pos = position_ids ? position_ids[seq_idx] : seq_idx;
    if (pos < 0)
        return;

    // Compute sin/cos table
    for (int i = threadIdx.x; i < half_dim; i += blockDim.x)
    {
        float angle = pos * inv_freq[i];
        s_cos[i] = cosf(angle);
        s_sin[i] = sinf(angle);
    }
    __syncthreads();

    // Apply rotation
    int base_idx = seq_idx * n_heads * head_dim + head_idx * head_dim;

    for (int i = threadIdx.x; i < half_dim; i += blockDim.x)
    {
        int i0 = base_idx + i;
        int i1 = base_idx + i + half_dim;

        float x0 = bf16_to_float(data[i0]);
        float x1 = bf16_to_float(data[i1]);
        float cos_val = s_cos[i];
        float sin_val = s_sin[i];

        data[i0] = float_to_bf16(x0 * cos_val - x1 * sin_val);
        data[i1] = float_to_bf16(x0 * sin_val + x1 * cos_val);
    }
}

/**
 * @brief Fused BF16 RoPE kernel for Q and K
 */
__global__ void rocm_rope_bf16_fused_qk_kernel(
    uint16_t *__restrict__ Q,
    uint16_t *__restrict__ K,
    const float *__restrict__ inv_freq,
    const int *__restrict__ position_ids,
    int seq_len,
    int n_q_heads,
    int n_kv_heads,
    int head_dim)
{
    const int half_dim = head_dim / 2;

    extern __shared__ float smem[];
    float *s_cos = smem;
    float *s_sin = smem + half_dim;

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

    for (int i = threadIdx.x; i < half_dim; i += blockDim.x)
    {
        float angle = pos * inv_freq[i];
        s_cos[i] = cosf(angle);
        s_sin[i] = sinf(angle);
    }
    __syncthreads();

    int base_idx = seq_idx * n_heads * head_dim + head_idx * head_dim;

    for (int i = threadIdx.x; i < half_dim; i += blockDim.x)
    {
        int i0 = base_idx + i;
        int i1 = base_idx + i + half_dim;

        float x0 = bf16_to_float(data[i0]);
        float x1 = bf16_to_float(data[i1]);

        data[i0] = float_to_bf16(x0 * s_cos[i] - x1 * s_sin[i]);
        data[i1] = float_to_bf16(x0 * s_sin[i] + x1 * s_cos[i]);
    }
}

/**
 * @brief FP16 RoPE kernel with precomputed inv_freq and shared memory
 */
__global__ void rocm_rope_fp16_kernel_v2(
    uint16_t *__restrict__ data,
    const float *__restrict__ inv_freq,
    const int *__restrict__ position_ids,
    int seq_len,
    int n_heads,
    int head_dim)
{
    const int half_dim = head_dim / 2;

    extern __shared__ float smem[];
    float *s_cos = smem;
    float *s_sin = smem + half_dim;

    int block_idx = blockIdx.x;
    int head_idx = block_idx % n_heads;
    int seq_idx = block_idx / n_heads;

    if (seq_idx >= seq_len)
        return;

    int pos = position_ids ? position_ids[seq_idx] : seq_idx;
    if (pos < 0)
        return;

    for (int i = threadIdx.x; i < half_dim; i += blockDim.x)
    {
        float angle = pos * inv_freq[i];
        s_cos[i] = cosf(angle);
        s_sin[i] = sinf(angle);
    }
    __syncthreads();

    int base_idx = seq_idx * n_heads * head_dim + head_idx * head_dim;

    for (int i = threadIdx.x; i < half_dim; i += blockDim.x)
    {
        int i0 = base_idx + i;
        int i1 = base_idx + i + half_dim;

        float x0 = fp16_to_float(data[i0]);
        float x1 = fp16_to_float(data[i1]);
        float cos_val = s_cos[i];
        float sin_val = s_sin[i];

        data[i0] = float_to_fp16(x0 * cos_val - x1 * sin_val);
        data[i1] = float_to_fp16(x0 * sin_val + x1 * cos_val);
    }
}

/**
 * @brief Fused FP16 RoPE kernel for Q and K
 */
__global__ void rocm_rope_fp16_fused_qk_kernel(
    uint16_t *__restrict__ Q,
    uint16_t *__restrict__ K,
    const float *__restrict__ inv_freq,
    const int *__restrict__ position_ids,
    int seq_len,
    int n_q_heads,
    int n_kv_heads,
    int head_dim)
{
    const int half_dim = head_dim / 2;

    extern __shared__ float smem[];
    float *s_cos = smem;
    float *s_sin = smem + half_dim;

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

    for (int i = threadIdx.x; i < half_dim; i += blockDim.x)
    {
        float angle = pos * inv_freq[i];
        s_cos[i] = cosf(angle);
        s_sin[i] = sinf(angle);
    }
    __syncthreads();

    int base_idx = seq_idx * n_heads * head_dim + head_idx * head_dim;

    for (int i = threadIdx.x; i < half_dim; i += blockDim.x)
    {
        int i0 = base_idx + i;
        int i1 = base_idx + i + half_dim;

        float x0 = fp16_to_float(data[i0]);
        float x1 = fp16_to_float(data[i1]);

        data[i0] = float_to_fp16(x0 * s_cos[i] - x1 * s_sin[i]);
        data[i1] = float_to_fp16(x0 * s_sin[i] + x1 * s_cos[i]);
    }
}

// =========================================================================
// Extern "C" Wrapper Functions (v2 - with inv_freq parameter)
// =========================================================================

extern "C"
{

    /**
     * @brief Optimized FP32 RoPE using precomputed inverse frequencies
     *
     * @param Q Q tensor [seq_len, n_heads, head_dim] (device pointer)
     * @param K K tensor [seq_len, n_kv_heads, head_dim] (device pointer, can be nullptr)
     * @param inv_freq Pre-computed inverse frequency table [head_dim/2] (device pointer)
     * @param position_ids Position IDs [seq_len] (device pointer)
     * @param seq_len Sequence length
     * @param n_heads Number of Q heads
     * @param n_kv_heads Number of KV heads
     * @param head_dim Head dimension
     * @param device_idx HIP device index
     */
    bool hipOps_rope_fp32_v2(
        float *Q,
        float *K,
        const float *inv_freq,
        const int *position_ids,
        int seq_len,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        int device_idx, void* stream)
    {
        llaminar2::HipDeviceGuard::setDevice(device_idx);

        hipStream_t hip_stream = static_cast<hipStream_t>(stream);
        const int half_dim = head_dim / 2;
        const int threads_per_block = min(256, half_dim);
        const size_t smem_size = 2 * half_dim * sizeof(float); // sin + cos tables

        if (K != nullptr)
        {
            // Fused Q+K kernel
            int total_blocks = seq_len * (n_heads + n_kv_heads);
            hipLaunchKernelGGL(rocm_rope_fp32_fused_qk_kernel, 
                               dim3(total_blocks), dim3(threads_per_block), smem_size, hip_stream,
                               Q, K, inv_freq, position_ids, seq_len, n_heads, n_kv_heads, head_dim);
        }
        else
        {
            // Q only
            int num_blocks = seq_len * n_heads;
            hipLaunchKernelGGL(rocm_rope_fp32_kernel_v2, 
                               dim3(num_blocks), dim3(threads_per_block), smem_size, hip_stream,
                               Q, inv_freq, position_ids, seq_len, n_heads, head_dim);
        }

        hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "HIP RoPE FP32 v2 kernel launch failed: %s\n", hipGetErrorString(err));
            return false;
        }
        return true;
    }

    bool hipOps_rope_bf16_v2(
        uint16_t *Q,
        uint16_t *K,
        const float *inv_freq,
        const int *position_ids,
        int seq_len,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        int device_idx, void* stream)
    {
        llaminar2::HipDeviceGuard::setDevice(device_idx);

        hipStream_t hip_stream = static_cast<hipStream_t>(stream);
        const int half_dim = head_dim / 2;
        const int threads_per_block = min(256, half_dim);
        const size_t smem_size = 2 * half_dim * sizeof(float);

        if (K != nullptr)
        {
            int total_blocks = seq_len * (n_heads + n_kv_heads);
            hipLaunchKernelGGL(rocm_rope_bf16_fused_qk_kernel, 
                               dim3(total_blocks), dim3(threads_per_block), smem_size, hip_stream,
                               Q, K, inv_freq, position_ids, seq_len, n_heads, n_kv_heads, head_dim);
        }
        else
        {
            int num_blocks = seq_len * n_heads;
            hipLaunchKernelGGL(rocm_rope_bf16_kernel_v2, 
                               dim3(num_blocks), dim3(threads_per_block), smem_size, hip_stream,
                               Q, inv_freq, position_ids, seq_len, n_heads, head_dim);
        }

        hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "HIP RoPE BF16 v2 kernel launch failed: %s\n", hipGetErrorString(err));
            return false;
        }
        return true;
    }

    bool hipOps_rope_fp16_v2(
        uint16_t *Q,
        uint16_t *K,
        const float *inv_freq,
        const int *position_ids,
        int seq_len,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        int device_idx, void* stream)
    {
        llaminar2::HipDeviceGuard::setDevice(device_idx);

        hipStream_t hip_stream = static_cast<hipStream_t>(stream);
        const int half_dim = head_dim / 2;
        const int threads_per_block = min(256, half_dim);
        const size_t smem_size = 2 * half_dim * sizeof(float);

        if (K != nullptr)
        {
            int total_blocks = seq_len * (n_heads + n_kv_heads);
            hipLaunchKernelGGL(rocm_rope_fp16_fused_qk_kernel, 
                               dim3(total_blocks), dim3(threads_per_block), smem_size, hip_stream,
                               Q, K, inv_freq, position_ids, seq_len, n_heads, n_kv_heads, head_dim);
        }
        else
        {
            int num_blocks = seq_len * n_heads;
            hipLaunchKernelGGL(rocm_rope_fp16_kernel_v2, 
                               dim3(num_blocks), dim3(threads_per_block), smem_size, hip_stream,
                               Q, inv_freq, position_ids, seq_len, n_heads, head_dim);
        }

        hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "HIP RoPE FP16 v2 kernel launch failed: %s\n", hipGetErrorString(err));
            return false;
        }
        return true;
    }

    // =========================================================================
    // Decode-Optimized Wrappers (scalar position, no array copy needed)
    // =========================================================================

    /**
     * @brief FP32 RoPE decode - takes scalar position (no H2D copy needed)
     *
     * Optimized for decode phase where seq_len=1 and all heads share the same position.
     * Eliminates the hipMemcpy overhead that dominates decode performance.
     */
    bool hipOps_rope_fp32_decode(
        float *Q,
        float *K,
        const float *inv_freq,
        int pos,           // Single position value (not array!)
        int n_q_heads,
        int n_kv_heads,
        int head_dim,
        int device_idx, void* stream)
    {
        llaminar2::HipDeviceGuard::setDevice(device_idx);

        hipStream_t hip_stream = static_cast<hipStream_t>(stream);
        const int half_dim = head_dim / 2;
        const int threads_per_block = min(256, half_dim);
        const size_t smem_size = 2 * half_dim * sizeof(float);

        // Launch decode kernel for Q
        hipLaunchKernelGGL(rocm_rope_fp32_decode_kernel, 
                           dim3(n_q_heads), dim3(threads_per_block), smem_size, hip_stream,
                           Q, inv_freq, pos, n_q_heads, head_dim);

        // Launch decode kernel for K if provided
        if (K != nullptr)
        {
            hipLaunchKernelGGL(rocm_rope_fp32_decode_kernel, 
                               dim3(n_kv_heads), dim3(threads_per_block), smem_size, hip_stream,
                               K, inv_freq, pos, n_kv_heads, head_dim);
        }

        hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "HIP RoPE FP32 decode kernel launch failed: %s\n", hipGetErrorString(err));
            return false;
        }
        return true;
    }

    bool hipOps_rope_bf16_decode(
        uint16_t *Q,
        uint16_t *K,
        const float *inv_freq,
        int pos,
        int n_q_heads,
        int n_kv_heads,
        int head_dim,
        int device_idx, void* stream)
    {
        llaminar2::HipDeviceGuard::setDevice(device_idx);

        hipStream_t hip_stream = static_cast<hipStream_t>(stream);
        const int half_dim = head_dim / 2;
        const int threads_per_block = min(256, half_dim);
        const size_t smem_size = 2 * half_dim * sizeof(float);

        hipLaunchKernelGGL(rocm_rope_bf16_decode_kernel, 
                           dim3(n_q_heads), dim3(threads_per_block), smem_size, hip_stream,
                           Q, inv_freq, pos, n_q_heads, head_dim);

        if (K != nullptr)
        {
            hipLaunchKernelGGL(rocm_rope_bf16_decode_kernel, 
                               dim3(n_kv_heads), dim3(threads_per_block), smem_size, hip_stream,
                               K, inv_freq, pos, n_kv_heads, head_dim);
        }

        hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "HIP RoPE BF16 decode kernel launch failed: %s\n", hipGetErrorString(err));
            return false;
        }
        return true;
    }

    bool hipOps_rope_fp16_decode(
        uint16_t *Q,
        uint16_t *K,
        const float *inv_freq,
        int pos,
        int n_q_heads,
        int n_kv_heads,
        int head_dim,
        int device_idx, void* stream)
    {
        llaminar2::HipDeviceGuard::setDevice(device_idx);

        hipStream_t hip_stream = static_cast<hipStream_t>(stream);
        const int half_dim = head_dim / 2;
        const int threads_per_block = min(256, half_dim);
        const size_t smem_size = 2 * half_dim * sizeof(float);

        hipLaunchKernelGGL(rocm_rope_fp16_decode_kernel, 
                           dim3(n_q_heads), dim3(threads_per_block), smem_size, hip_stream,
                           Q, inv_freq, pos, n_q_heads, head_dim);

        if (K != nullptr)
        {
            hipLaunchKernelGGL(rocm_rope_fp16_decode_kernel, 
                               dim3(n_kv_heads), dim3(threads_per_block), smem_size, hip_stream,
                               K, inv_freq, pos, n_kv_heads, head_dim);
        }

        hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "HIP RoPE FP16 decode kernel launch failed: %s\n", hipGetErrorString(err));
            return false;
        }
        return true;
    }

    // =========================================================================
    // Contiguous Position Wrappers (zero-copy: positions computed on GPU)
    // =========================================================================

    /**
     * @brief FP32 RoPE with contiguous positions - ZERO memory copy
     *
     * Optimal for inference where positions are sequential: [pos_offset, pos_offset+1, ...].
     * Positions computed on-the-fly by GPU threads (pos = pos_offset + seq_idx).
     */
    bool hipOps_rope_fp32_contiguous(
        float *Q,
        float *K,
        const float *inv_freq,
        int pos_offset,  // Starting position
        int seq_len,
        int n_q_heads,
        int n_kv_heads,
        int head_dim,
        int device_idx, void* stream)
    {
        llaminar2::HipDeviceGuard::setDevice(device_idx);

        hipStream_t hip_stream = static_cast<hipStream_t>(stream);
        const int half_dim = head_dim / 2;
        const int threads_per_block = min(256, half_dim);
        const size_t smem_size = 2 * half_dim * sizeof(float);

        if (K != nullptr)
        {
            int total_blocks = seq_len * (n_q_heads + n_kv_heads);
            hipLaunchKernelGGL(rocm_rope_fp32_fused_qk_contiguous_kernel, 
                               dim3(total_blocks), dim3(threads_per_block), smem_size, hip_stream,
                               Q, K, inv_freq, pos_offset, seq_len, n_q_heads, n_kv_heads, head_dim);
        }
        else
        {
            int num_blocks = seq_len * n_q_heads;
            hipLaunchKernelGGL(rocm_rope_fp32_contiguous_kernel, 
                               dim3(num_blocks), dim3(threads_per_block), smem_size, hip_stream,
                               Q, inv_freq, pos_offset, seq_len, n_q_heads, head_dim);
        }

        hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "HIP RoPE FP32 contiguous kernel launch failed: %s\n", hipGetErrorString(err));
            return false;
        }
        return true;
    }

    bool hipOps_rope_bf16_contiguous(
        uint16_t *Q,
        uint16_t *K,
        const float *inv_freq,
        int pos_offset,
        int seq_len,
        int n_q_heads,
        int n_kv_heads,
        int head_dim,
        int device_idx, void* stream)
    {
        llaminar2::HipDeviceGuard::setDevice(device_idx);

        hipStream_t hip_stream = static_cast<hipStream_t>(stream);
        const int half_dim = head_dim / 2;
        const int threads_per_block = min(256, half_dim);
        const size_t smem_size = 2 * half_dim * sizeof(float);

        if (K != nullptr)
        {
            int total_blocks = seq_len * (n_q_heads + n_kv_heads);
            hipLaunchKernelGGL(rocm_rope_bf16_fused_qk_contiguous_kernel, 
                               dim3(total_blocks), dim3(threads_per_block), smem_size, hip_stream,
                               Q, K, inv_freq, pos_offset, seq_len, n_q_heads, n_kv_heads, head_dim);
        }
        else
        {
            int num_blocks = seq_len * n_q_heads;
            hipLaunchKernelGGL(rocm_rope_bf16_contiguous_kernel, 
                               dim3(num_blocks), dim3(threads_per_block), smem_size, hip_stream,
                               Q, inv_freq, pos_offset, seq_len, n_q_heads, head_dim);
        }

        hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "HIP RoPE BF16 contiguous kernel launch failed: %s\n", hipGetErrorString(err));
            return false;
        }
        return true;
    }

    bool hipOps_rope_fp16_contiguous(
        uint16_t *Q,
        uint16_t *K,
        const float *inv_freq,
        int pos_offset,
        int seq_len,
        int n_q_heads,
        int n_kv_heads,
        int head_dim,
        int device_idx, void* stream)
    {
        llaminar2::HipDeviceGuard::setDevice(device_idx);

        hipStream_t hip_stream = static_cast<hipStream_t>(stream);
        const int half_dim = head_dim / 2;
        const int threads_per_block = min(256, half_dim);
        const size_t smem_size = 2 * half_dim * sizeof(float);

        if (K != nullptr)
        {
            int total_blocks = seq_len * (n_q_heads + n_kv_heads);
            hipLaunchKernelGGL(rocm_rope_fp16_fused_qk_contiguous_kernel, 
                               dim3(total_blocks), dim3(threads_per_block), smem_size, hip_stream,
                               Q, K, inv_freq, pos_offset, seq_len, n_q_heads, n_kv_heads, head_dim);
        }
        else
        {
            int num_blocks = seq_len * n_q_heads;
            hipLaunchKernelGGL(rocm_rope_fp16_contiguous_kernel, 
                               dim3(num_blocks), dim3(threads_per_block), smem_size, hip_stream,
                               Q, inv_freq, pos_offset, seq_len, n_q_heads, head_dim);
        }

        hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "HIP RoPE FP16 contiguous kernel launch failed: %s\n", hipGetErrorString(err));
            return false;
        }
        return true;
    }

    /**
     * @brief Populate inverse frequency table in device memory (workspace-aware)
     *
     * This function computes the inverse frequency table on the host and copies
     * it to the provided device buffer. Used for workspace-based allocation where
     * the caller owns the device memory.
     *
     * @param d_inv_freq Device pointer to store inverse frequencies [head_dim/2 floats]
     * @param head_dim Head dimension
     * @param freq_base RoPE frequency base (typically 10000.0)
     * @param device_idx HIP device index
     * @return true on success
     *
     * Formula: inv_freq[i] = 1.0 / (freq_base^(2i/head_dim))
     */
    /**
     * @brief GPU kernel to compute inverse frequency table directly on device.
     *
     * Replaces the host-compute + synchronous hipMemcpy path. By computing
     * inv_freq on the GPU, we avoid an implicit full pipeline drain that blocks
     * the host for ~11ms per layer during prefill (hipMemcpy synchronizes all
     * pending GPU work on the default stream before the copy can begin).
     *
     * Formula: inv_freq[i] = exp(-log(freq_base) * 2*i / head_dim)
     *                      = 1.0 / (freq_base^(2i/head_dim))
     */
    __global__ void rope_populate_inv_freq_kernel(
        float *d_inv_freq,
        int half_dim,
        float neg_log_base,
        int head_dim)
    {
        int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
        if (i < half_dim)
        {
            float exponent = (2.0f * i) / head_dim;
            d_inv_freq[i] = expf(neg_log_base * exponent);
        }
    }

    bool hipOps_rope_populate_inv_freq(
        float *d_inv_freq,
        int head_dim,
        float freq_base,
        int device_idx, void* stream)
    {
        if (!d_inv_freq)
            return false;

        const int half_dim = head_dim / 2;
        const float neg_log_base = -std::log(freq_base);

        llaminar2::HipDeviceGuard::setDevice(device_idx);

        hipStream_t hip_stream = static_cast<hipStream_t>(stream);
        // Compute inv_freq directly on GPU — fully async, no pipeline drain.
        // The kernel runs on the same default stream as subsequent RoPE kernels,
        // so ordering is guaranteed without explicit synchronization.
        constexpr int kBlockSize = 64;
        int grid_size = (half_dim + kBlockSize - 1) / kBlockSize;
        hipLaunchKernelGGL(rope_populate_inv_freq_kernel,
                           dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                           d_inv_freq, half_dim, neg_log_base, head_dim);

        hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "[ROCm RoPE] Failed to launch inv_freq kernel: %s\n",
                    hipGetErrorString(err));
            return false;
        }
        return true;
    }

} // extern "C"
