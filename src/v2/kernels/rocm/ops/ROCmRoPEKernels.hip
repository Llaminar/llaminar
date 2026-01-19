/**
 * @file ROCmRoPEKernels.hip
 * @brief HIP RoPE (Rotary Position Embedding) kernel implementations for ROCm
 * @author Llaminar Team
 *
 * Contains FP32, BF16, and FP16 RoPE kernels with extern "C" wrapper functions.
 * Uses SPLIT-HALF layout matching the CPU implementation.
 *
 * This follows the same pattern as the CUDA RoPE kernels, using HIP for
 * AMD GPU compatibility.
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cmath>

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
// RoPE HIP Kernels
// =========================================================================

/**
 * @brief FP32 RoPE kernel - applies rotary position embeddings
 *
 * Each thread handles one pair of dimensions for one head at one position.
 * Layout: [seq_len, n_heads, head_dim] with SPLIT-HALF pairing:
 *   - pair(i) = (data[i], data[i + half_dim]) for i in 0..half_dim-1
 * This matches the CPU implementation in RoPEPrimitives.cpp
 */
__global__ void rocm_rope_fp32_kernel(
    float *__restrict__ data,
    const int *__restrict__ position_ids,
    int seq_len,
    int n_heads,
    int head_dim,
    float rope_theta)
{
    // Each thread handles one (position, head, dim_pair) triplet
    const int half_dim = head_dim / 2;
    int total_pairs = seq_len * n_heads * half_dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total_pairs)
        return;

    // Decode indices
    int pair_idx = idx % half_dim;
    int head_idx = (idx / half_dim) % n_heads;
    int seq_idx = idx / (n_heads * half_dim);

    // Get actual position
    int pos = position_ids ? position_ids[seq_idx] : seq_idx;

    // Skip padding positions
    if (pos < 0)
        return;

    // Compute frequency: inv_freq[i] = pow(rope_theta, -2*i/head_dim)
    float freq = powf(rope_theta, -2.0f * pair_idx / head_dim);
    float angle = pos * freq;
    float cos_val = cosf(angle);
    float sin_val = sinf(angle);

    // Get data indices using SPLIT-HALF layout (matching CPU)
    // First element at pair_idx, second element at pair_idx + half_dim
    int base_idx = seq_idx * n_heads * head_dim + head_idx * head_dim;
    int i0 = base_idx + pair_idx;            // First half
    int i1 = base_idx + pair_idx + half_dim; // Second half

    // Apply rotation
    float x0 = data[i0];
    float x1 = data[i1];
    data[i0] = x0 * cos_val - x1 * sin_val;
    data[i1] = x0 * sin_val + x1 * cos_val;
}

/**
 * @brief BF16 RoPE kernel - FP32 computation with BF16 I/O
 * Uses SPLIT-HALF layout matching CPU implementation
 */
__global__ void rocm_rope_bf16_kernel(
    uint16_t *__restrict__ data,
    const int *__restrict__ position_ids,
    int seq_len,
    int n_heads,
    int head_dim,
    float rope_theta)
{
    const int half_dim = head_dim / 2;
    int total_pairs = seq_len * n_heads * half_dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total_pairs)
        return;

    int pair_idx = idx % half_dim;
    int head_idx = (idx / half_dim) % n_heads;
    int seq_idx = idx / (n_heads * half_dim);

    int pos = position_ids ? position_ids[seq_idx] : seq_idx;
    if (pos < 0)
        return;

    float freq = powf(rope_theta, -2.0f * pair_idx / head_dim);
    float angle = pos * freq;
    float cos_val = cosf(angle);
    float sin_val = sinf(angle);

    // SPLIT-HALF layout (matching CPU)
    int base_idx = seq_idx * n_heads * head_dim + head_idx * head_dim;
    int i0 = base_idx + pair_idx;            // First half
    int i1 = base_idx + pair_idx + half_dim; // Second half

    float x0 = bf16_to_float(data[i0]);
    float x1 = bf16_to_float(data[i1]);
    data[i0] = float_to_bf16(x0 * cos_val - x1 * sin_val);
    data[i1] = float_to_bf16(x0 * sin_val + x1 * cos_val);
}

/**
 * @brief FP16 RoPE kernel - FP32 computation with FP16 I/O
 * Uses SPLIT-HALF layout matching CPU implementation
 */
__global__ void rocm_rope_fp16_kernel(
    uint16_t *__restrict__ data,
    const int *__restrict__ position_ids,
    int seq_len,
    int n_heads,
    int head_dim,
    float rope_theta)
{
    const int half_dim = head_dim / 2;
    int total_pairs = seq_len * n_heads * half_dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total_pairs)
        return;

    int pair_idx = idx % half_dim;
    int head_idx = (idx / half_dim) % n_heads;
    int seq_idx = idx / (n_heads * half_dim);

    int pos = position_ids ? position_ids[seq_idx] : seq_idx;
    if (pos < 0)
        return;

    float freq = powf(rope_theta, -2.0f * pair_idx / head_dim);
    float angle = pos * freq;
    float cos_val = cosf(angle);
    float sin_val = sinf(angle);

    // SPLIT-HALF layout (matching CPU)
    int base_idx = seq_idx * n_heads * head_dim + head_idx * head_dim;
    int i0 = base_idx + pair_idx;            // First half
    int i1 = base_idx + pair_idx + half_dim; // Second half

    float x0 = fp16_to_float(data[i0]);
    float x1 = fp16_to_float(data[i1]);
    data[i0] = float_to_fp16(x0 * cos_val - x1 * sin_val);
    data[i1] = float_to_fp16(x0 * sin_val + x1 * cos_val);
}

// =========================================================================
// Extern "C" Wrapper Functions
// =========================================================================

extern "C"
{

    bool hipOps_rope_fp32(
        float *Q,
        float *K, // Can be nullptr
        const int *position_ids,
        int seq_len,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        float rope_theta,
        int device_idx)
    {
        hipSetDevice(device_idx);

        int threads_per_block = 256;

        // Process Q
        int total_pairs_q = seq_len * n_heads * (head_dim / 2);
        int num_blocks_q = (total_pairs_q + threads_per_block - 1) / threads_per_block;
        hipLaunchKernelGGL(rocm_rope_fp32_kernel, dim3(num_blocks_q), dim3(threads_per_block), 0, 0,
                           Q, position_ids, seq_len, n_heads, head_dim, rope_theta);

        // Process K if provided
        if (K != nullptr)
        {
            int total_pairs_k = seq_len * n_kv_heads * (head_dim / 2);
            int num_blocks_k = (total_pairs_k + threads_per_block - 1) / threads_per_block;
            hipLaunchKernelGGL(rocm_rope_fp32_kernel, dim3(num_blocks_k), dim3(threads_per_block), 0, 0,
                               K, position_ids, seq_len, n_kv_heads, head_dim, rope_theta);
        }

        hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "HIP RoPE FP32 kernel launch failed: %s\n", hipGetErrorString(err));
            return false;
        }
        return true;
    }

    bool hipOps_rope_bf16(
        uint16_t *Q,
        uint16_t *K, // Can be nullptr
        const int *position_ids,
        int seq_len,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        float rope_theta,
        int device_idx)
    {
        hipSetDevice(device_idx);

        int threads_per_block = 256;

        int total_pairs_q = seq_len * n_heads * (head_dim / 2);
        int num_blocks_q = (total_pairs_q + threads_per_block - 1) / threads_per_block;
        hipLaunchKernelGGL(rocm_rope_bf16_kernel, dim3(num_blocks_q), dim3(threads_per_block), 0, 0,
                           Q, position_ids, seq_len, n_heads, head_dim, rope_theta);

        if (K != nullptr)
        {
            int total_pairs_k = seq_len * n_kv_heads * (head_dim / 2);
            int num_blocks_k = (total_pairs_k + threads_per_block - 1) / threads_per_block;
            hipLaunchKernelGGL(rocm_rope_bf16_kernel, dim3(num_blocks_k), dim3(threads_per_block), 0, 0,
                               K, position_ids, seq_len, n_kv_heads, head_dim, rope_theta);
        }

        hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "HIP RoPE BF16 kernel launch failed: %s\n", hipGetErrorString(err));
            return false;
        }
        return true;
    }

    bool hipOps_rope_fp16(
        uint16_t *Q,
        uint16_t *K, // Can be nullptr
        const int *position_ids,
        int seq_len,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        float rope_theta,
        int device_idx)
    {
        hipSetDevice(device_idx);

        int threads_per_block = 256;

        int total_pairs_q = seq_len * n_heads * (head_dim / 2);
        int num_blocks_q = (total_pairs_q + threads_per_block - 1) / threads_per_block;
        hipLaunchKernelGGL(rocm_rope_fp16_kernel, dim3(num_blocks_q), dim3(threads_per_block), 0, 0,
                           Q, position_ids, seq_len, n_heads, head_dim, rope_theta);

        if (K != nullptr)
        {
            int total_pairs_k = seq_len * n_kv_heads * (head_dim / 2);
            int num_blocks_k = (total_pairs_k + threads_per_block - 1) / threads_per_block;
            hipLaunchKernelGGL(rocm_rope_fp16_kernel, dim3(num_blocks_k), dim3(threads_per_block), 0, 0,
                               K, position_ids, seq_len, n_kv_heads, head_dim, rope_theta);
        }

        hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "HIP RoPE FP16 kernel launch failed: %s\n", hipGetErrorString(err));
            return false;
        }
        return true;
    }

} // extern "C"
