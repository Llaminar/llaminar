/**
 * @file ROCmRingKVCacheKernels.hip
 * @brief HIP kernels for ROCm Ring Buffer KV Cache
 * @author Llaminar Team
 * @date January 2026
 *
 * HIP kernels for ring buffer KV cache operations on AMD GPUs.
 * Optimized for MI50 (gfx906 / Vega 20):
 * - 64 work-items per wavefront
 * - 256 threads per block (4 wavefronts)
 * - Coalesced memory access patterns
 *
 * Kernels:
 * 1. ring_append_kernel - Append tokens with wrap-around
 * 2. ring_linearize_kernel - Copy wrapped data to contiguous buffer
 * 3. ring_gather_batched_kernel - Gather multiple sequences for batched attention
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>

namespace llaminar2
{

    // =========================================================================
    // Constants
    // =========================================================================

    constexpr int BLOCK_SIZE = 256; // 4 wavefronts for MI50

    // =========================================================================
    // HIP Kernels
    // =========================================================================

    /**
     * @brief Append tokens to ring buffer with wrap-around
     *
     * Each thread handles one element of one token.
     * Grid: (num_tokens, ceil(kv_dim / BLOCK_SIZE))
     * Block: (BLOCK_SIZE)
     *
     * Memory layout: [max_seq_len, kv_dim] (position-major)
     *
     * @tparam T Data type (float, _Float16, hip_bfloat16)
     */
    template <typename T>
    __global__ void ring_append_kernel(
        T *__restrict__ d_K_cache,     // [max_seq_len, kv_dim]
        T *__restrict__ d_V_cache,     // [max_seq_len, kv_dim]
        const T *__restrict__ d_K_new, // [num_tokens, kv_dim]
        const T *__restrict__ d_V_new, // [num_tokens, kv_dim]
        int head,                      // Current head position
        int max_seq_len,               // Ring buffer capacity
        int kv_dim,                    // n_kv_heads * head_dim
        int num_tokens)                // Tokens to append
    {
        const int token_idx = blockIdx.x;
        const int elem_idx = blockIdx.y * blockDim.x + threadIdx.x;

        if (token_idx >= num_tokens || elem_idx >= kv_dim)
            return;

        // Calculate destination position with wrap-around
        const int dst_pos = (head + token_idx) % max_seq_len;
        const int dst_offset = dst_pos * kv_dim + elem_idx;
        const int src_offset = token_idx * kv_dim + elem_idx;

        d_K_cache[dst_offset] = d_K_new[src_offset];
        d_V_cache[dst_offset] = d_V_new[src_offset];
    }

    /**
     * @brief Linearize wrapped ring buffer to contiguous output
     *
     * Copies data from ring buffer [tail..end, 0..head) to linear [0..count)
     *
     * Grid: (count, ceil(kv_dim / BLOCK_SIZE))
     * Block: (BLOCK_SIZE)
     *
     * @tparam T Data type
     */
    template <typename T>
    __global__ void ring_linearize_kernel(
        T *__restrict__ d_out,         // [count, kv_dim]
        const T *__restrict__ d_cache, // [max_seq_len, kv_dim]
        int tail,                      // Start position (oldest token)
        int count,                     // Number of valid tokens
        int max_seq_len,               // Ring buffer capacity
        int kv_dim)                    // n_kv_heads * head_dim
    {
        const int token_idx = blockIdx.x;
        const int elem_idx = blockIdx.y * blockDim.x + threadIdx.x;

        if (token_idx >= count || elem_idx >= kv_dim)
            return;

        // Calculate source position with wrap-around
        const int src_pos = (tail + token_idx) % max_seq_len;
        const int src_offset = src_pos * kv_dim + elem_idx;
        const int dst_offset = token_idx * kv_dim + elem_idx;

        d_out[dst_offset] = d_cache[src_offset];
    }

    /**
     * @brief Gather multiple sequences into batched output
     *
     * Each z-block handles one sequence's linearization with padding.
     * Output: [num_seqs * max_kv_len, kv_dim]
     *
     * Grid: (max_kv_len, ceil(kv_dim / BLOCK_SIZE), num_seqs)
     * Block: (BLOCK_SIZE)
     *
     * @tparam T Data type
     */
    template <typename T>
    __global__ void ring_gather_batched_kernel(
        T *__restrict__ d_K_out,                 // [num_seqs * max_kv_len, kv_dim]
        T *__restrict__ d_V_out,                 // [num_seqs * max_kv_len, kv_dim]
        const T *const *__restrict__ d_K_caches, // Array of K cache pointers
        const T *const *__restrict__ d_V_caches, // Array of V cache pointers
        const int *__restrict__ tails,           // Per-sequence tail positions
        const int *__restrict__ counts,          // Per-sequence token counts
        int num_seqs,                            // Number of sequences
        int max_kv_len,                          // Output padding length
        int max_seq_len,                         // Ring buffer capacity
        int kv_dim)                              // n_kv_heads * head_dim
    {
        const int seq_idx = blockIdx.z;
        const int token_idx = blockIdx.x;
        const int elem_idx = blockIdx.y * blockDim.x + threadIdx.x;

        if (seq_idx >= num_seqs || elem_idx >= kv_dim)
            return;

        const int seq_count = counts[seq_idx];

        // Calculate output offset
        const int out_offset = (seq_idx * max_kv_len + token_idx) * kv_dim + elem_idx;

        if (token_idx >= seq_count)
        {
            // Padding: zero-fill beyond sequence length
            d_K_out[out_offset] = T(0);
            d_V_out[out_offset] = T(0);
            return;
        }

        // Calculate source position with wrap-around
        const int tail = tails[seq_idx];
        const int src_pos = (tail + token_idx) % max_seq_len;
        const int src_offset = src_pos * kv_dim + elem_idx;

        d_K_out[out_offset] = d_K_caches[seq_idx][src_offset];
        d_V_out[out_offset] = d_V_caches[seq_idx][src_offset];
    }

    // =========================================================================
    // Kernel Launch Wrappers (extern "C" for linking)
    // =========================================================================

    // ------ FP32 Variants ------

    extern "C" void hip_ring_append_fp32(
        float *d_K_cache, float *d_V_cache,
        const float *d_K_new, const float *d_V_new,
        int head, int max_seq_len, int kv_dim, int num_tokens,
        hipStream_t stream)
    {
        if (num_tokens <= 0 || kv_dim <= 0)
            return;

        dim3 grid(num_tokens, (kv_dim + BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 block(BLOCK_SIZE);

        hipLaunchKernelGGL(ring_append_kernel<float>,
                           grid, block, 0, stream,
                           d_K_cache, d_V_cache,
                           d_K_new, d_V_new,
                           head, max_seq_len, kv_dim, num_tokens);
    }

    extern "C" void hip_ring_linearize_fp32(
        float *d_out,
        const float *d_cache,
        int tail, int count, int max_seq_len, int kv_dim,
        hipStream_t stream)
    {
        if (count <= 0 || kv_dim <= 0)
            return;

        dim3 grid(count, (kv_dim + BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 block(BLOCK_SIZE);

        hipLaunchKernelGGL(ring_linearize_kernel<float>,
                           grid, block, 0, stream,
                           d_out, d_cache,
                           tail, count, max_seq_len, kv_dim);
    }

    extern "C" void hip_ring_gather_batched_fp32(
        float *d_K_out, float *d_V_out,
        const float *const *d_K_caches, const float *const *d_V_caches,
        const int *tails, const int *counts,
        int num_seqs, int max_kv_len, int max_seq_len, int kv_dim,
        hipStream_t stream)
    {
        if (num_seqs <= 0 || max_kv_len <= 0 || kv_dim <= 0)
            return;

        dim3 grid(max_kv_len, (kv_dim + BLOCK_SIZE - 1) / BLOCK_SIZE, num_seqs);
        dim3 block(BLOCK_SIZE);

        hipLaunchKernelGGL(ring_gather_batched_kernel<float>,
                           grid, block, 0, stream,
                           d_K_out, d_V_out,
                           d_K_caches, d_V_caches,
                           tails, counts,
                           num_seqs, max_kv_len, max_seq_len, kv_dim);
    }

    // ------ FP16 Variants ------

    extern "C" void hip_ring_append_fp16(
        _Float16 *d_K_cache, _Float16 *d_V_cache,
        const _Float16 *d_K_new, const _Float16 *d_V_new,
        int head, int max_seq_len, int kv_dim, int num_tokens,
        hipStream_t stream)
    {
        if (num_tokens <= 0 || kv_dim <= 0)
            return;

        dim3 grid(num_tokens, (kv_dim + BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 block(BLOCK_SIZE);

        hipLaunchKernelGGL(ring_append_kernel<_Float16>,
                           grid, block, 0, stream,
                           d_K_cache, d_V_cache,
                           d_K_new, d_V_new,
                           head, max_seq_len, kv_dim, num_tokens);
    }

    extern "C" void hip_ring_linearize_fp16(
        _Float16 *d_out,
        const _Float16 *d_cache,
        int tail, int count, int max_seq_len, int kv_dim,
        hipStream_t stream)
    {
        if (count <= 0 || kv_dim <= 0)
            return;

        dim3 grid(count, (kv_dim + BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 block(BLOCK_SIZE);

        hipLaunchKernelGGL(ring_linearize_kernel<_Float16>,
                           grid, block, 0, stream,
                           d_out, d_cache,
                           tail, count, max_seq_len, kv_dim);
    }

    extern "C" void hip_ring_gather_batched_fp16(
        _Float16 *d_K_out, _Float16 *d_V_out,
        const _Float16 *const *d_K_caches, const _Float16 *const *d_V_caches,
        const int *tails, const int *counts,
        int num_seqs, int max_kv_len, int max_seq_len, int kv_dim,
        hipStream_t stream)
    {
        if (num_seqs <= 0 || max_kv_len <= 0 || kv_dim <= 0)
            return;

        dim3 grid(max_kv_len, (kv_dim + BLOCK_SIZE - 1) / BLOCK_SIZE, num_seqs);
        dim3 block(BLOCK_SIZE);

        hipLaunchKernelGGL(ring_gather_batched_kernel<_Float16>,
                           grid, block, 0, stream,
                           d_K_out, d_V_out,
                           d_K_caches, d_V_caches,
                           tails, counts,
                           num_seqs, max_kv_len, max_seq_len, kv_dim);
    }

    // ------ BF16 Variants ------

    extern "C" void hip_ring_append_bf16(
        hip_bfloat16 *d_K_cache, hip_bfloat16 *d_V_cache,
        const hip_bfloat16 *d_K_new, const hip_bfloat16 *d_V_new,
        int head, int max_seq_len, int kv_dim, int num_tokens,
        hipStream_t stream)
    {
        if (num_tokens <= 0 || kv_dim <= 0)
            return;

        dim3 grid(num_tokens, (kv_dim + BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 block(BLOCK_SIZE);

        hipLaunchKernelGGL(ring_append_kernel<hip_bfloat16>,
                           grid, block, 0, stream,
                           d_K_cache, d_V_cache,
                           d_K_new, d_V_new,
                           head, max_seq_len, kv_dim, num_tokens);
    }

    extern "C" void hip_ring_linearize_bf16(
        hip_bfloat16 *d_out,
        const hip_bfloat16 *d_cache,
        int tail, int count, int max_seq_len, int kv_dim,
        hipStream_t stream)
    {
        if (count <= 0 || kv_dim <= 0)
            return;

        dim3 grid(count, (kv_dim + BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 block(BLOCK_SIZE);

        hipLaunchKernelGGL(ring_linearize_kernel<hip_bfloat16>,
                           grid, block, 0, stream,
                           d_out, d_cache,
                           tail, count, max_seq_len, kv_dim);
    }

    extern "C" void hip_ring_gather_batched_bf16(
        hip_bfloat16 *d_K_out, hip_bfloat16 *d_V_out,
        const hip_bfloat16 *const *d_K_caches, const hip_bfloat16 *const *d_V_caches,
        const int *tails, const int *counts,
        int num_seqs, int max_kv_len, int max_seq_len, int kv_dim,
        hipStream_t stream)
    {
        if (num_seqs <= 0 || max_kv_len <= 0 || kv_dim <= 0)
            return;

        dim3 grid(max_kv_len, (kv_dim + BLOCK_SIZE - 1) / BLOCK_SIZE, num_seqs);
        dim3 block(BLOCK_SIZE);

        hipLaunchKernelGGL(ring_gather_batched_kernel<hip_bfloat16>,
                           grid, block, 0, stream,
                           d_K_out, d_V_out,
                           d_K_caches, d_V_caches,
                           tails, counts,
                           num_seqs, max_kv_len, max_seq_len, kv_dim);
    }

} // namespace llaminar2
