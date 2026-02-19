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
#include "../../../tensors/BlockStructures.h"
#include "../../../tensors/TensorType.h"

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
     * @brief Append tokens to ring buffer (graph-capturable variant)
     *
     * Reads head position from a device pointer instead of a scalar argument.
     * This allows the kernel to be captured in a HIP graph while the head
     * position is updated between graph replays via H2D memcpy.
     *
     * @tparam T Data type (float, _Float16, hip_bfloat16)
     */
    template <typename T>
    __global__ void ring_append_kernel_dynamic(
        T *__restrict__ d_K_cache,     // [max_seq_len, kv_dim]
        T *__restrict__ d_V_cache,     // [max_seq_len, kv_dim]
        const T *__restrict__ d_K_new, // [num_tokens, kv_dim]
        const T *__restrict__ d_V_new, // [num_tokens, kv_dim]
        const int *__restrict__ d_head, // Head position (read from device memory)
        int max_seq_len,               // Ring buffer capacity
        int kv_dim,                    // n_kv_heads * head_dim
        int num_tokens)                // Tokens to append
    {
        const int token_idx = blockIdx.x;
        const int elem_idx = blockIdx.y * blockDim.x + threadIdx.x;

        if (token_idx >= num_tokens || elem_idx >= kv_dim)
            return;

        const int head = *d_head; // Read from device memory
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
            d_K_out[out_offset] = T{};
            d_V_out[out_offset] = T{};
            return;
        }

        // Calculate source position with wrap-around
        const int tail = tails[seq_idx];
        const int src_pos = (tail + token_idx) % max_seq_len;
        const int src_offset = src_pos * kv_dim + elem_idx;

        d_K_out[out_offset] = d_K_caches[seq_idx][src_offset];
        d_V_out[out_offset] = d_V_caches[seq_idx][src_offset];
    }

    template <typename T>
    __device__ inline float to_float_device(T v)
    {
        return static_cast<float>(v);
    }

    template <>
    __device__ inline float to_float_device<half>(half v)
    {
        return __half2float(v);
    }

    template <>
    __device__ inline float to_float_device<hip_bfloat16>(hip_bfloat16 v)
    {
        return static_cast<float>(v);
    }

    template <typename SrcT>
    __global__ void convert_to_fp16_kernel(
        const SrcT *__restrict__ src,
        half *__restrict__ dst,
        int count)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= count)
            return;
        dst[idx] = __float2half(to_float_device(src[idx]));
    }

    template <typename SrcT>
    __global__ void convert_to_q8_1_kernel(
        const SrcT *__restrict__ src,
        Q8_1Block *__restrict__ dst,
        int rows,
        int cols,
        int blocks_per_row)
    {
        const int row = blockIdx.y;
        const int block_col = blockIdx.x;
        const int lane = threadIdx.x;

        if (row >= rows || block_col >= blocks_per_row || lane >= Q8_1Block::BLOCK_SIZE)
            return;

        const int col = block_col * Q8_1Block::BLOCK_SIZE + lane;
        const bool in_bounds = (col < cols);
        const float x = in_bounds ? to_float_device(src[row * cols + col]) : 0.0f;

        __shared__ float s_absmax[Q8_1Block::BLOCK_SIZE];
        __shared__ int s_q[Q8_1Block::BLOCK_SIZE];
        __shared__ int s_sum[Q8_1Block::BLOCK_SIZE];

        s_absmax[lane] = fabsf(x);
        __syncthreads();

        for (int stride = Q8_1Block::BLOCK_SIZE / 2; stride > 0; stride >>= 1)
        {
            if (lane < stride)
            {
                s_absmax[lane] = fmaxf(s_absmax[lane], s_absmax[lane + stride]);
            }
            __syncthreads();
        }

        const float absmax = s_absmax[0];
        const float d = (absmax > 0.0f) ? (absmax / 127.0f) : 0.0f;
        int q = 0;
        if (d > 0.0f)
        {
            q = __float2int_rn(x / d);
            q = max(-127, min(127, q));
        }

        s_q[lane] = q;
        s_sum[lane] = q;
        __syncthreads();

        for (int stride = Q8_1Block::BLOCK_SIZE / 2; stride > 0; stride >>= 1)
        {
            if (lane < stride)
            {
                s_sum[lane] += s_sum[lane + stride];
            }
            __syncthreads();
        }

        if (lane == 0)
        {
            Q8_1Block &out = dst[row * blocks_per_row + block_col];
            out.d = __half_as_ushort(__float2half(d));
            out.sum_qs = static_cast<int16_t>(s_sum[0]);
        }

        dst[row * blocks_per_row + block_col].qs[lane] = static_cast<int8_t>(s_q[lane]);
    }

    // =========================================================================
    // FP16/BF16/Q8_1 → FP32 GPU conversion kernel
    // Used by flash attention to convert KV cache data on-device instead of
    // the catastrophically slow CPU D2H → convert → H2D round-trip.
    // =========================================================================
    template <typename SrcT>
    __global__ void convert_to_fp32_kernel(
        const SrcT *__restrict__ src,
        float *__restrict__ dst,
        int count)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= count)
            return;
        dst[idx] = to_float_device(src[idx]);
    }

    __global__ void convert_q8_1_to_fp32_kernel(
        const Q8_1Block *__restrict__ src,
        float *__restrict__ dst,
        int rows,
        int cols,
        int blocks_per_row)
    {
        const int row = blockIdx.y;
        const int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (row >= rows || col >= cols)
            return;

        const int block_col = col / Q8_1Block::BLOCK_SIZE;
        const int lane = col % Q8_1Block::BLOCK_SIZE;
        const Q8_1Block &blk = src[row * blocks_per_row + block_col];
        const float d = __half2float(__ushort_as_half(blk.d));
        dst[row * cols + col] = d * static_cast<float>(blk.qs[lane]);
    }

    extern "C" bool hip_convert_tensor_to_fp32(
        const void *d_src,
        TensorType src_type,
        float *d_dst,
        int count,
        int rows,
        int cols,
        hipStream_t stream)
    {
        if (!d_src || !d_dst || count <= 0)
            return false;

        dim3 block(256);
        dim3 grid((count + block.x - 1) / block.x);

        switch (src_type)
        {
        case TensorType::FP16:
            hipLaunchKernelGGL(convert_to_fp32_kernel<half>, grid, block, 0, stream,
                               reinterpret_cast<const half *>(d_src), d_dst, count);
            break;
        case TensorType::BF16:
            hipLaunchKernelGGL(convert_to_fp32_kernel<hip_bfloat16>, grid, block, 0, stream,
                               static_cast<const hip_bfloat16 *>(d_src), d_dst, count);
            break;
        case TensorType::FP32:
            hipMemcpyAsync(d_dst, d_src, static_cast<size_t>(count) * sizeof(float),
                           hipMemcpyDeviceToDevice, stream);
            break;
        case TensorType::Q8_1:
        {
            if (rows <= 0 || cols <= 0)
                return false;
            const int blocks_per_row = (cols + Q8_1Block::BLOCK_SIZE - 1) / Q8_1Block::BLOCK_SIZE;
            dim3 q_block(256);
            dim3 q_grid((cols + q_block.x - 1) / q_block.x, rows);
            hipLaunchKernelGGL(convert_q8_1_to_fp32_kernel, q_grid, q_block, 0, stream,
                               static_cast<const Q8_1Block *>(d_src), d_dst,
                               rows, cols, blocks_per_row);
            break;
        }
        default:
            return false;
        }

        return hipGetLastError() == hipSuccess;
    }

    extern "C" bool hip_convert_tensor_to_fp16(
        const void *d_src,
        TensorType src_type,
        uint16_t *d_dst,
        int count,
        hipStream_t stream)
    {
        if (!d_src || !d_dst || count <= 0)
        {
            return false;
        }

        dim3 block(256);
        dim3 grid((count + block.x - 1) / block.x);

        switch (src_type)
        {
        case TensorType::FP32:
            hipLaunchKernelGGL(convert_to_fp16_kernel<float>, grid, block, 0, stream,
                               static_cast<const float *>(d_src),
                               reinterpret_cast<half *>(d_dst),
                               count);
            break;
        case TensorType::FP16:
            hipMemcpyAsync(d_dst, d_src, static_cast<size_t>(count) * sizeof(uint16_t),
                           hipMemcpyDeviceToDevice, stream);
            break;
        case TensorType::BF16:
            hipLaunchKernelGGL(convert_to_fp16_kernel<hip_bfloat16>, grid, block, 0, stream,
                               static_cast<const hip_bfloat16 *>(d_src),
                               reinterpret_cast<half *>(d_dst),
                               count);
            break;
        case TensorType::Q8_1:
            return false;
        default:
            return false;
        }

        return hipGetLastError() == hipSuccess;
    }

    extern "C" bool hip_convert_tensor_to_q8_1(
        const void *d_src,
        TensorType src_type,
        Q8_1Block *d_dst,
        int rows,
        int cols,
        hipStream_t stream)
    {
        if (!d_src || !d_dst || rows <= 0 || cols <= 0)
        {
            return false;
        }

        const int blocks_per_row = (cols + Q8_1Block::BLOCK_SIZE - 1) / Q8_1Block::BLOCK_SIZE;
        dim3 block(Q8_1Block::BLOCK_SIZE);
        dim3 grid(blocks_per_row, rows);

        switch (src_type)
        {
        case TensorType::FP32:
            hipLaunchKernelGGL(convert_to_q8_1_kernel<float>, grid, block, 0, stream,
                               static_cast<const float *>(d_src), d_dst, rows, cols, blocks_per_row);
            break;
        case TensorType::FP16:
            hipLaunchKernelGGL(convert_to_q8_1_kernel<half>, grid, block, 0, stream,
                               static_cast<const half *>(d_src), d_dst, rows, cols, blocks_per_row);
            break;
        case TensorType::BF16:
            hipLaunchKernelGGL(convert_to_q8_1_kernel<hip_bfloat16>, grid, block, 0, stream,
                               static_cast<const hip_bfloat16 *>(d_src), d_dst, rows, cols, blocks_per_row);
            break;
        case TensorType::Q8_1:
            hipMemcpyAsync(d_dst, d_src,
                           static_cast<size_t>(rows) * static_cast<size_t>(blocks_per_row) * sizeof(Q8_1Block),
                           hipMemcpyDeviceToDevice, stream);
            break;
        default:
            return false;
        }

        return hipGetLastError() == hipSuccess;
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

    // ------ Q8_1 Variants (block-based) ------

    extern "C" void hip_ring_append_q8_1(
        Q8_1Block *d_K_cache, Q8_1Block *d_V_cache,
        const Q8_1Block *d_K_new, const Q8_1Block *d_V_new,
        int head, int max_seq_len, int kv_blocks, int num_tokens,
        hipStream_t stream)
    {
        if (num_tokens <= 0 || kv_blocks <= 0)
            return;

        dim3 grid(num_tokens, (kv_blocks + BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 block(BLOCK_SIZE);

        hipLaunchKernelGGL(ring_append_kernel<Q8_1Block>,
                           grid, block, 0, stream,
                           d_K_cache, d_V_cache,
                           d_K_new, d_V_new,
                           head, max_seq_len, kv_blocks, num_tokens);
    }

    extern "C" void hip_ring_linearize_q8_1(
        Q8_1Block *d_out,
        const Q8_1Block *d_cache,
        int tail, int count, int max_seq_len, int kv_blocks,
        hipStream_t stream)
    {
        if (count <= 0 || kv_blocks <= 0)
            return;

        dim3 grid(count, (kv_blocks + BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 block(BLOCK_SIZE);

        hipLaunchKernelGGL(ring_linearize_kernel<Q8_1Block>,
                           grid, block, 0, stream,
                           d_out, d_cache,
                           tail, count, max_seq_len, kv_blocks);
    }

    extern "C" void hip_ring_gather_batched_q8_1(
        Q8_1Block *d_K_out, Q8_1Block *d_V_out,
        const Q8_1Block *const *d_K_caches, const Q8_1Block *const *d_V_caches,
        const int *tails, const int *counts,
        int num_seqs, int max_kv_len, int max_seq_len, int kv_blocks,
        hipStream_t stream)
    {
        if (num_seqs <= 0 || max_kv_len <= 0 || kv_blocks <= 0)
            return;

        dim3 grid(max_kv_len, (kv_blocks + BLOCK_SIZE - 1) / BLOCK_SIZE, num_seqs);
        dim3 block(BLOCK_SIZE);

        hipLaunchKernelGGL(ring_gather_batched_kernel<Q8_1Block>,
                           grid, block, 0, stream,
                           d_K_out, d_V_out,
                           d_K_caches, d_V_caches,
                           tails, counts,
                           num_seqs, max_kv_len, max_seq_len, kv_blocks);
    }

    // =========================================================================
    // Dynamic Head Append Wrappers (graph-capturable)
    // =========================================================================

    extern "C" void hip_ring_append_dynamic_fp32(
        float *d_K_cache, float *d_V_cache,
        const float *d_K_new, const float *d_V_new,
        const int *d_head, int max_seq_len, int kv_dim, int num_tokens,
        hipStream_t stream)
    {
        if (num_tokens <= 0 || kv_dim <= 0)
            return;

        dim3 grid(num_tokens, (kv_dim + BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 block(BLOCK_SIZE);

        hipLaunchKernelGGL(ring_append_kernel_dynamic<float>,
                           grid, block, 0, stream,
                           d_K_cache, d_V_cache,
                           d_K_new, d_V_new,
                           d_head, max_seq_len, kv_dim, num_tokens);
    }

    extern "C" void hip_ring_append_dynamic_fp16(
        _Float16 *d_K_cache, _Float16 *d_V_cache,
        const _Float16 *d_K_new, const _Float16 *d_V_new,
        const int *d_head, int max_seq_len, int kv_dim, int num_tokens,
        hipStream_t stream)
    {
        if (num_tokens <= 0 || kv_dim <= 0)
            return;

        dim3 grid(num_tokens, (kv_dim + BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 block(BLOCK_SIZE);

        hipLaunchKernelGGL(ring_append_kernel_dynamic<_Float16>,
                           grid, block, 0, stream,
                           d_K_cache, d_V_cache,
                           d_K_new, d_V_new,
                           d_head, max_seq_len, kv_dim, num_tokens);
    }

    extern "C" void hip_ring_append_dynamic_bf16(
        hip_bfloat16 *d_K_cache, hip_bfloat16 *d_V_cache,
        const hip_bfloat16 *d_K_new, const hip_bfloat16 *d_V_new,
        const int *d_head, int max_seq_len, int kv_dim, int num_tokens,
        hipStream_t stream)
    {
        if (num_tokens <= 0 || kv_dim <= 0)
            return;

        dim3 grid(num_tokens, (kv_dim + BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 block(BLOCK_SIZE);

        hipLaunchKernelGGL(ring_append_kernel_dynamic<hip_bfloat16>,
                           grid, block, 0, stream,
                           d_K_cache, d_V_cache,
                           d_K_new, d_V_new,
                           d_head, max_seq_len, kv_dim, num_tokens);
    }

    extern "C" void hip_ring_append_dynamic_q8_1(
        Q8_1Block *d_K_cache, Q8_1Block *d_V_cache,
        const Q8_1Block *d_K_new, const Q8_1Block *d_V_new,
        const int *d_head, int max_seq_len, int kv_blocks, int num_tokens,
        hipStream_t stream)
    {
        if (num_tokens <= 0 || kv_blocks <= 0)
            return;

        dim3 grid(num_tokens, (kv_blocks + BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 block(BLOCK_SIZE);

        hipLaunchKernelGGL(ring_append_kernel_dynamic<Q8_1Block>,
                           grid, block, 0, stream,
                           d_K_cache, d_V_cache,
                           d_K_new, d_V_new,
                           d_head, max_seq_len, kv_blocks, num_tokens);
    }

} // namespace llaminar2
