/**
 * @file CUDARingKVCache.cu
 * @brief CUDA Ring Buffer KV Cache implementation
 * @author David Sanftenberg
 * @date January 2026
 *
 * CUDA kernels and implementation for ring buffer KV cache.
 *
 * Kernels:
 * 1. ring_append_kernel - Append tokens with wrap-around
 * 2. ring_linearize_kernel - Copy wrapped data to contiguous buffer
 * 3. ring_gather_batched_kernel - Gather multiple sequences
 */

#include "CUDARingKVCache.h"
#include "../../../execution/local_execution/graph/GraphCaptureGuard.h"
#include "../../../execution/local_execution/device/DeviceWorkspaceManager.h"
#include "../../../execution/local_execution/device/WorkspaceDescriptor.h"
#include "../../../backends/GPUDeviceContextPool.h"
#include "../../../utils/Logger.h"
#include "../../kvcache/KVCacheDeviceParams.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "../../../tensors/BlockStructures.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace llaminar2
{

    // =========================================================================
    // CUDA Kernels
    // =========================================================================

    /**
     * @brief Append tokens to ring buffer with wrap-around
     *
     * Each thread handles one element of one token.
     * Grid: (num_tokens, kv_dim / 256)
     * Block: (256)
     *
     * @tparam T Data type (float, __half, __nv_bfloat16)
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
        int token_idx = blockIdx.x;
        int elem_idx = blockIdx.y * blockDim.x + threadIdx.x;

        if (token_idx >= num_tokens || elem_idx >= kv_dim)
            return;

        // Calculate destination position with wrap-around
        int dst_pos = (head + token_idx) % max_seq_len;
        int dst_offset = dst_pos * kv_dim + elem_idx;
        int src_offset = token_idx * kv_dim + elem_idx;

        d_K_cache[dst_offset] = d_K_new[src_offset];
        d_V_cache[dst_offset] = d_V_new[src_offset];
    }

    /**
     * @brief Append tokens to ring buffer (graph-capturable variant)
     *
     * Reads head position from a device pointer instead of a scalar argument.
     * This allows the kernel to be captured in a GPU graph while the head
     * position is updated between graph replays via H2D memcpy to the
     * device params buffer.
     *
     * @tparam T Data type (float, __half, __nv_bfloat16)
     */
    template <typename T>
    __global__ void ring_append_kernel_dynamic(
        T *__restrict__ d_K_cache,      // [max_seq_len, kv_dim]
        T *__restrict__ d_V_cache,      // [max_seq_len, kv_dim]
        const T *__restrict__ d_K_new,  // [num_tokens, kv_dim]
        const T *__restrict__ d_V_new,  // [num_tokens, kv_dim]
        const int *__restrict__ d_head, // Head position (read from device memory)
        int max_seq_len,                // Ring buffer capacity
        int kv_dim,                     // n_kv_heads * head_dim
        int num_tokens)                 // Tokens to append
    {
        int token_idx = blockIdx.x;
        int elem_idx = blockIdx.y * blockDim.x + threadIdx.x;

        if (token_idx >= num_tokens || elem_idx >= kv_dim)
            return;

        int head = *d_head; // Read from device memory
        int dst_pos = (head + token_idx) % max_seq_len;
        int dst_offset = dst_pos * kv_dim + elem_idx;
        int src_offset = token_idx * kv_dim + elem_idx;

        d_K_cache[dst_offset] = d_K_new[src_offset];
        d_V_cache[dst_offset] = d_V_new[src_offset];
    }

    /**
     * @brief Append verifier rows from position-major or head-major source.
     *
     * MTP all-position verifier graphs may keep K/V grouped by KV head after
     * RoPE (`[head][row][dim]`) even though the ring cache is position-major.
     * This kernel performs the layout gather while writing rows at the same
     * next-write positions as serial decode.
     */
    template <typename T>
    __global__ void ring_append_verifier_rows_kernel(
        T *__restrict__ d_K_cache,
        T *__restrict__ d_V_cache,
        const T *__restrict__ d_K_new,
        const T *__restrict__ d_V_new,
        int head,
        int max_seq_len,
        int kv_storage_dim,
        int head_storage_dim,
        int source_verifier_rows,
        int source_start_row,
        int rows_to_write,
        bool k_source_head_major,
        bool v_source_head_major)
    {
        const int token_idx = blockIdx.x;
        const int elem_idx = blockIdx.y * blockDim.x + threadIdx.x;
        if (token_idx >= rows_to_write || elem_idx >= kv_storage_dim)
            return;

        const int source_row = source_start_row + token_idx;
        const int dst_pos = (head + token_idx) % max_seq_len;
        const int dst_offset = dst_pos * kv_storage_dim + elem_idx;

        const int head_idx = elem_idx / head_storage_dim;
        const int head_lane = elem_idx - head_idx * head_storage_dim;
        const int k_src_offset = k_source_head_major
                                     ? (head_idx * source_verifier_rows + source_row) * head_storage_dim + head_lane
                                     : source_row * kv_storage_dim + elem_idx;
        const int v_src_offset = v_source_head_major
                                     ? (head_idx * source_verifier_rows + source_row) * head_storage_dim + head_lane
                                     : source_row * kv_storage_dim + elem_idx;

        d_K_cache[dst_offset] = d_K_new[k_src_offset];
        d_V_cache[dst_offset] = d_V_new[v_src_offset];
    }

    template <typename T>
    __global__ void ring_append_verifier_rows_dynamic_kernel(
        T *__restrict__ d_K_cache,
        T *__restrict__ d_V_cache,
        const T *__restrict__ d_K_new,
        const T *__restrict__ d_V_new,
        const int *__restrict__ d_head,
        int max_seq_len,
        int kv_storage_dim,
        int head_storage_dim,
        int source_verifier_rows,
        int source_start_row,
        int rows_to_write,
        bool k_source_head_major,
        bool v_source_head_major)
    {
        const int token_idx = blockIdx.x;
        const int elem_idx = blockIdx.y * blockDim.x + threadIdx.x;
        if (token_idx >= rows_to_write || elem_idx >= kv_storage_dim)
            return;

        const int source_row = source_start_row + token_idx;
        const int head = *d_head;
        const int dst_pos = (head + token_idx) % max_seq_len;
        const int dst_offset = dst_pos * kv_storage_dim + elem_idx;

        const int head_idx = elem_idx / head_storage_dim;
        const int head_lane = elem_idx - head_idx * head_storage_dim;
        const int k_src_offset = k_source_head_major
                                     ? (head_idx * source_verifier_rows + source_row) * head_storage_dim + head_lane
                                     : source_row * kv_storage_dim + elem_idx;
        const int v_src_offset = v_source_head_major
                                     ? (head_idx * source_verifier_rows + source_row) * head_storage_dim + head_lane
                                     : source_row * kv_storage_dim + elem_idx;

        d_K_cache[dst_offset] = d_K_new[k_src_offset];
        d_V_cache[dst_offset] = d_V_new[v_src_offset];
    }

    /**
     * @brief Advance graph-captured KV sequence metadata on device.
     *
     * This runs after the append kernel on the same explicit stream. Keeping it
     * separate avoids racing with append blocks that still need to read the
     * pre-append head position.
     */
    __global__ void cuda_kv_sequence_state_advance_kernel(
        int *__restrict__ d_head,
        int *__restrict__ d_count,
        int num_tokens,
        int max_seq_len)
    {
        if (threadIdx.x != 0 || blockIdx.x != 0)
            return;
        const int old_head = *d_head;
        const int old_count = *d_count;
        *d_head = (old_head + num_tokens) % max_seq_len;
        const int next_count = old_count + num_tokens;
        *d_count = next_count > max_seq_len ? max_seq_len : next_count;
    }

    __global__ void cuda_kv_sequence_state_advance_dynamic_kernel(
        int *__restrict__ d_head,
        int *__restrict__ d_count,
        const int *__restrict__ d_append_count,
        int captured_num_tokens,
        int max_seq_len)
    {
        if (threadIdx.x != 0 || blockIdx.x != 0)
            return;

        int advance_tokens = d_append_count ? *d_append_count : captured_num_tokens;
        if (advance_tokens <= 0 || advance_tokens > captured_num_tokens)
            advance_tokens = captured_num_tokens;

        const int old_head = *d_head;
        const int old_count = *d_count;
        *d_head = (old_head + advance_tokens) % max_seq_len;
        const int next_count = old_count + advance_tokens;
        *d_count = next_count > max_seq_len ? max_seq_len : next_count;
    }

    /**
     * @brief Publish accepted verifier-row sequence metadata on device.
     *
     * Each block owns one [request, layer] pair.  The verifier graph may have
     * already advanced the live device head past rows that were later rejected,
     * so publication must keep the current ring tail and clamp the visible
     * window to the accepted target length.  This mirrors truncateSequence()
     * and prevents host/device KV metadata from disagreeing after rollback.
     */
    __global__ void cuda_kv_sequence_state_publish_kernel(
        int *__restrict__ d_heads,
        int *__restrict__ d_counts,
        const int32_t *__restrict__ target_cached_tokens,
        const int32_t *__restrict__ accepted_state_counts,
        const int32_t *__restrict__ publication_ok_flags,
        int batch_size,
        int first_seq_idx,
        int request_count,
        int max_seq_len)
    {
        const int request_idx = static_cast<int>(blockIdx.x);
        const int layer = static_cast<int>(blockIdx.y);
        if (threadIdx.x != 0 || request_idx >= request_count)
            return;

        if (publication_ok_flags[request_idx] == 0)
            return;

        const int seq_idx = first_seq_idx + request_idx;
        if (seq_idx < 0 || seq_idx >= batch_size)
            return;

        const int accepted_count = accepted_state_counts[request_idx];
        const int target_count = target_cached_tokens[request_idx];
        if (accepted_count < 0 ||
            target_count < 0 ||
            target_count > max_seq_len)
        {
            return;
        }

        const int entry_idx = layer * batch_size + seq_idx;
        const int old_head = d_heads[entry_idx];
        const int old_count = d_counts[entry_idx];
        if (old_count < 0 || old_count > max_seq_len)
        {
            return;
        }

        int tail = old_head - old_count;
        tail %= max_seq_len;
        if (tail < 0)
            tail += max_seq_len;
        d_heads[entry_idx] = (tail + target_count) % max_seq_len;
        d_counts[entry_idx] = target_count;
    }

    /**
     * @brief Linearize wrapped ring buffer to contiguous output
     *
     * Copies data from ring buffer [tail..end, 0..head) to linear [0..count)
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
        int token_idx = blockIdx.x;
        int elem_idx = blockIdx.y * blockDim.x + threadIdx.x;

        if (token_idx >= count || elem_idx >= kv_dim)
            return;

        // Calculate source position with wrap-around
        int src_pos = (tail + token_idx) % max_seq_len;
        int src_offset = src_pos * kv_dim + elem_idx;
        int dst_offset = token_idx * kv_dim + elem_idx;

        d_out[dst_offset] = d_cache[src_offset];
    }

    /**
     * @brief Gather multiple sequences into batched output
     *
     * Each block handles one sequence's linearization.
     * Output: [num_seqs * max_kv_len, kv_dim]
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
        int seq_idx = blockIdx.z;
        int token_idx = blockIdx.x;
        int elem_idx = blockIdx.y * blockDim.x + threadIdx.x;

        if (seq_idx >= num_seqs || elem_idx >= kv_dim)
            return;

        int seq_count = counts[seq_idx];

        // Calculate output offset
        int out_offset = (seq_idx * max_kv_len + token_idx) * kv_dim + elem_idx;

        if (token_idx >= seq_count)
        {
            // Padding: zero-fill beyond sequence length
            d_K_out[out_offset] = T{};
            d_V_out[out_offset] = T{};
            return;
        }

        // Calculate source position with wrap-around
        int tail = tails[seq_idx];
        int src_pos = (tail + token_idx) % max_seq_len;
        int src_offset = src_pos * kv_dim + elem_idx;

        d_K_out[out_offset] = d_K_caches[seq_idx][src_offset];
        d_V_out[out_offset] = d_V_caches[seq_idx][src_offset];
    }

    template <typename T>
    __device__ inline float to_float_device(T v)
    {
        return static_cast<float>(v);
    }

    template <>
    __device__ inline float to_float_device<__half>(__half v)
    {
        return __half2float(v);
    }

    template <>
    __device__ inline float to_float_device<__nv_bfloat16>(__nv_bfloat16 v)
    {
        return __bfloat162float(v);
    }

    template <typename SrcT>
    __global__ void convert_to_fp16_kernel(
        const SrcT *__restrict__ src,
        __half *__restrict__ dst,
        int count)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= count)
            return;
        dst[idx] = __float2half_rn(to_float_device(src[idx]));
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
            out.d = __half_as_ushort(__float2half_rn(d));
            out.sum_qs = static_cast<int16_t>(s_sum[0]);
        }

        if (lane < Q8_1Block::BLOCK_SIZE)
        {
            dst[row * blocks_per_row + block_col].qs[lane] = static_cast<int8_t>(s_q[lane]);
        }
    }

    // Q8_1 → FP16 dequantization kernel
    // Each thread processes one logical element
    __global__ void dequant_q8_1_to_fp16_kernel(
        const Q8_1Block *__restrict__ src,
        __half *__restrict__ dst,
        int count)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= count)
            return;

        const int block_idx = idx / Q8_1Block::BLOCK_SIZE;
        const int elem_idx = idx % Q8_1Block::BLOCK_SIZE;

        const Q8_1Block &block = src[block_idx];
        const float scale = __half2float(__ushort_as_half(block.d));
        const float val = scale * static_cast<float>(block.qs[elem_idx]);
        dst[idx] = __float2half_rn(val);
    }

    extern "C" bool cuda_convert_tensor_to_fp16(
        const void *d_src,
        TensorType src_type,
        uint16_t *d_dst,
        int count,
        cudaStream_t stream)
    {
        if (!d_src || !d_dst || count <= 0)
        {
            return false;
        }

        const dim3 block(256);
        const dim3 grid((count + block.x - 1) / block.x);

        switch (src_type)
        {
        case TensorType::FP32:
            convert_to_fp16_kernel<float><<<grid, block, 0, stream>>>(
                static_cast<const float *>(d_src),
                reinterpret_cast<__half *>(d_dst),
                count);
            break;
        case TensorType::FP16:
            cudaMemcpyAsync(d_dst, d_src, static_cast<size_t>(count) * sizeof(uint16_t),
                            cudaMemcpyDeviceToDevice, stream);
            break;
        case TensorType::BF16:
            convert_to_fp16_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
                static_cast<const __nv_bfloat16 *>(d_src),
                reinterpret_cast<__half *>(d_dst),
                count);
            break;
        case TensorType::Q8_1:
            dequant_q8_1_to_fp16_kernel<<<grid, block, 0, stream>>>(
                static_cast<const Q8_1Block *>(d_src),
                reinterpret_cast<__half *>(d_dst),
                count);
            break;
        default:
            return false;
        }

        return cudaGetLastError() == cudaSuccess;
    }

    extern "C" bool cuda_convert_tensor_to_q8_1(
        const void *d_src,
        TensorType src_type,
        Q8_1Block *d_dst,
        int rows,
        int cols,
        cudaStream_t stream)
    {
        if (!d_src || !d_dst || rows <= 0 || cols <= 0)
        {
            return false;
        }

        const int blocks_per_row = (cols + Q8_1Block::BLOCK_SIZE - 1) / Q8_1Block::BLOCK_SIZE;
        const dim3 block(Q8_1Block::BLOCK_SIZE);
        const dim3 grid(blocks_per_row, rows);

        switch (src_type)
        {
        case TensorType::FP32:
            convert_to_q8_1_kernel<float><<<grid, block, 0, stream>>>(
                static_cast<const float *>(d_src),
                d_dst,
                rows,
                cols,
                blocks_per_row);
            break;
        case TensorType::FP16:
            convert_to_q8_1_kernel<__half><<<grid, block, 0, stream>>>(
                static_cast<const __half *>(d_src),
                d_dst,
                rows,
                cols,
                blocks_per_row);
            break;
        case TensorType::BF16:
            convert_to_q8_1_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
                static_cast<const __nv_bfloat16 *>(d_src),
                d_dst,
                rows,
                cols,
                blocks_per_row);
            break;
        case TensorType::Q8_1:
            cudaMemcpyAsync(d_dst, d_src,
                            static_cast<size_t>(rows) * static_cast<size_t>(blocks_per_row) * sizeof(Q8_1Block),
                            cudaMemcpyDeviceToDevice, stream);
            break;
        default:
            return false;
        }

        return cudaGetLastError() == cudaSuccess;
    }

    // =========================================================================
    // Kernel Launch Helpers (extern "C" wrappers)
    // =========================================================================

    // FP32 variants
    extern "C" void cuda_ring_append_fp32(
        float *d_K_cache, float *d_V_cache,
        const float *d_K_new, const float *d_V_new,
        int head, int max_seq_len, int kv_dim, int num_tokens,
        cudaStream_t stream)
    {
        if (num_tokens == 0)
            return;

        dim3 block(256);
        dim3 grid(num_tokens, (kv_dim + 255) / 256);
        ring_append_kernel<float><<<grid, block, 0, stream>>>(
            d_K_cache, d_V_cache, d_K_new, d_V_new,
            head, max_seq_len, kv_dim, num_tokens);
    }

    extern "C" void cuda_ring_linearize_fp32(
        float *d_K_out, float *d_V_out,
        const float *d_K_cache, const float *d_V_cache,
        int tail, int count, int max_seq_len, int kv_dim,
        cudaStream_t stream)
    {
        if (count == 0)
            return;

        dim3 block(256);
        dim3 grid(count, (kv_dim + 255) / 256);

        ring_linearize_kernel<float><<<grid, block, 0, stream>>>(
            d_K_out, d_K_cache, tail, count, max_seq_len, kv_dim);
        ring_linearize_kernel<float><<<grid, block, 0, stream>>>(
            d_V_out, d_V_cache, tail, count, max_seq_len, kv_dim);
    }

    // FP16 variants
    extern "C" void cuda_ring_append_fp16(
        __half *d_K_cache, __half *d_V_cache,
        const __half *d_K_new, const __half *d_V_new,
        int head, int max_seq_len, int kv_dim, int num_tokens,
        cudaStream_t stream)
    {
        if (num_tokens == 0)
            return;

        dim3 block(256);
        dim3 grid(num_tokens, (kv_dim + 255) / 256);
        ring_append_kernel<__half><<<grid, block, 0, stream>>>(
            d_K_cache, d_V_cache, d_K_new, d_V_new,
            head, max_seq_len, kv_dim, num_tokens);
    }

    extern "C" void cuda_ring_linearize_fp16(
        __half *d_K_out, __half *d_V_out,
        const __half *d_K_cache, const __half *d_V_cache,
        int tail, int count, int max_seq_len, int kv_dim,
        cudaStream_t stream)
    {
        if (count == 0)
            return;

        dim3 block(256);
        dim3 grid(count, (kv_dim + 255) / 256);

        ring_linearize_kernel<__half><<<grid, block, 0, stream>>>(
            d_K_out, d_K_cache, tail, count, max_seq_len, kv_dim);
        ring_linearize_kernel<__half><<<grid, block, 0, stream>>>(
            d_V_out, d_V_cache, tail, count, max_seq_len, kv_dim);
    }

    // BF16 variants
    extern "C" void cuda_ring_append_bf16(
        __nv_bfloat16 *d_K_cache, __nv_bfloat16 *d_V_cache,
        const __nv_bfloat16 *d_K_new, const __nv_bfloat16 *d_V_new,
        int head, int max_seq_len, int kv_dim, int num_tokens,
        cudaStream_t stream)
    {
        if (num_tokens == 0)
            return;

        dim3 block(256);
        dim3 grid(num_tokens, (kv_dim + 255) / 256);
        ring_append_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
            d_K_cache, d_V_cache, d_K_new, d_V_new,
            head, max_seq_len, kv_dim, num_tokens);
    }

    extern "C" void cuda_ring_linearize_bf16(
        __nv_bfloat16 *d_K_out, __nv_bfloat16 *d_V_out,
        const __nv_bfloat16 *d_K_cache, const __nv_bfloat16 *d_V_cache,
        int tail, int count, int max_seq_len, int kv_dim,
        cudaStream_t stream)
    {
        if (count == 0)
            return;

        dim3 block(256);
        dim3 grid(count, (kv_dim + 255) / 256);

        ring_linearize_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
            d_K_out, d_K_cache, tail, count, max_seq_len, kv_dim);
        ring_linearize_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
            d_V_out, d_V_cache, tail, count, max_seq_len, kv_dim);
    }

    // Q8_1 variants (block-based)
    extern "C" void cuda_ring_append_q8_1(
        Q8_1Block *d_K_cache, Q8_1Block *d_V_cache,
        const Q8_1Block *d_K_new, const Q8_1Block *d_V_new,
        int head, int max_seq_len, int kv_blocks, int num_tokens,
        cudaStream_t stream)
    {
        if (num_tokens == 0)
            return;

        dim3 block(256);
        dim3 grid(num_tokens, (kv_blocks + 255) / 256);
        ring_append_kernel<Q8_1Block><<<grid, block, 0, stream>>>(
            d_K_cache, d_V_cache, d_K_new, d_V_new,
            head, max_seq_len, kv_blocks, num_tokens);
    }

    extern "C" void cuda_ring_linearize_q8_1(
        Q8_1Block *d_K_out, Q8_1Block *d_V_out,
        const Q8_1Block *d_K_cache, const Q8_1Block *d_V_cache,
        int tail, int count, int max_seq_len, int kv_blocks,
        cudaStream_t stream)
    {
        if (count == 0)
            return;

        dim3 block(256);
        dim3 grid(count, (kv_blocks + 255) / 256);

        ring_linearize_kernel<Q8_1Block><<<grid, block, 0, stream>>>(
            d_K_out, d_K_cache, tail, count, max_seq_len, kv_blocks);
        ring_linearize_kernel<Q8_1Block><<<grid, block, 0, stream>>>(
            d_V_out, d_V_cache, tail, count, max_seq_len, kv_blocks);
    }

    // =========================================================================
    // Dynamic Head Append Wrappers (graph-capturable)
    // =========================================================================

    extern "C" void cuda_ring_append_dynamic_fp32(
        float *d_K_cache, float *d_V_cache,
        const float *d_K_new, const float *d_V_new,
        const int *d_head, int max_seq_len, int kv_dim, int num_tokens,
        cudaStream_t stream)
    {
        if (num_tokens == 0)
            return;

        dim3 block(256);
        dim3 grid(num_tokens, (kv_dim + 255) / 256);
        ring_append_kernel_dynamic<float><<<grid, block, 0, stream>>>(
            d_K_cache, d_V_cache, d_K_new, d_V_new,
            d_head, max_seq_len, kv_dim, num_tokens);
    }

    extern "C" void cuda_ring_append_dynamic_fp16(
        __half *d_K_cache, __half *d_V_cache,
        const __half *d_K_new, const __half *d_V_new,
        const int *d_head, int max_seq_len, int kv_dim, int num_tokens,
        cudaStream_t stream)
    {
        if (num_tokens == 0)
            return;

        dim3 block(256);
        dim3 grid(num_tokens, (kv_dim + 255) / 256);
        ring_append_kernel_dynamic<__half><<<grid, block, 0, stream>>>(
            d_K_cache, d_V_cache, d_K_new, d_V_new,
            d_head, max_seq_len, kv_dim, num_tokens);
    }

    extern "C" void cuda_ring_append_dynamic_bf16(
        __nv_bfloat16 *d_K_cache, __nv_bfloat16 *d_V_cache,
        const __nv_bfloat16 *d_K_new, const __nv_bfloat16 *d_V_new,
        const int *d_head, int max_seq_len, int kv_dim, int num_tokens,
        cudaStream_t stream)
    {
        if (num_tokens == 0)
            return;

        dim3 block(256);
        dim3 grid(num_tokens, (kv_dim + 255) / 256);
        ring_append_kernel_dynamic<__nv_bfloat16><<<grid, block, 0, stream>>>(
            d_K_cache, d_V_cache, d_K_new, d_V_new,
            d_head, max_seq_len, kv_dim, num_tokens);
    }

    extern "C" void cuda_ring_append_dynamic_q8_1(
        Q8_1Block *d_K_cache, Q8_1Block *d_V_cache,
        const Q8_1Block *d_K_new, const Q8_1Block *d_V_new,
        const int *d_head, int max_seq_len, int kv_blocks, int num_tokens,
        cudaStream_t stream)
    {
        if (num_tokens == 0)
            return;

        dim3 block(256);
        dim3 grid(num_tokens, (kv_blocks + 255) / 256);
        ring_append_kernel_dynamic<Q8_1Block><<<grid, block, 0, stream>>>(
            d_K_cache, d_V_cache, d_K_new, d_V_new,
            d_head, max_seq_len, kv_blocks, num_tokens);
    }

    template <typename T>
    static void cuda_ring_append_verifier_rows_typed(
        T *d_K_cache, T *d_V_cache,
        const T *d_K_new, const T *d_V_new,
        int head, int max_seq_len, int kv_storage_dim, int head_storage_dim,
        int source_verifier_rows, int source_start_row, int rows_to_write,
        bool k_source_head_major, bool v_source_head_major,
        cudaStream_t stream)
    {
        if (rows_to_write <= 0)
            return;

        dim3 block(256);
        dim3 grid(rows_to_write, (kv_storage_dim + static_cast<int>(block.x) - 1) / static_cast<int>(block.x));
        ring_append_verifier_rows_kernel<T><<<grid, block, 0, stream>>>(
            d_K_cache, d_V_cache, d_K_new, d_V_new,
            head, max_seq_len, kv_storage_dim, head_storage_dim,
            source_verifier_rows, source_start_row, rows_to_write,
            k_source_head_major, v_source_head_major);
    }

    template <typename T>
    static void cuda_ring_append_verifier_rows_dynamic_typed(
        T *d_K_cache, T *d_V_cache,
        const T *d_K_new, const T *d_V_new,
        const int *d_head, int max_seq_len, int kv_storage_dim, int head_storage_dim,
        int source_verifier_rows, int source_start_row, int rows_to_write,
        bool k_source_head_major, bool v_source_head_major,
        cudaStream_t stream)
    {
        if (rows_to_write <= 0)
            return;

        dim3 block(256);
        dim3 grid(rows_to_write, (kv_storage_dim + static_cast<int>(block.x) - 1) / static_cast<int>(block.x));
        ring_append_verifier_rows_dynamic_kernel<T><<<grid, block, 0, stream>>>(
            d_K_cache, d_V_cache, d_K_new, d_V_new,
            d_head, max_seq_len, kv_storage_dim, head_storage_dim,
            source_verifier_rows, source_start_row, rows_to_write,
            k_source_head_major, v_source_head_major);
    }

    extern "C" void cuda_kv_sequence_state_advance(
        int *d_head, int *d_count, int num_tokens, int max_seq_len,
        cudaStream_t stream)
    {
        if (!d_head || !d_count || num_tokens <= 0 || max_seq_len <= 0)
            return;
        cuda_kv_sequence_state_advance_kernel<<<1, 1, 0, stream>>>(
            d_head, d_count, num_tokens, max_seq_len);
    }

    extern "C" void cuda_kv_sequence_state_advance_dynamic(
        int *d_head, int *d_count, const int *d_append_count,
        int captured_num_tokens, int max_seq_len,
        cudaStream_t stream)
    {
        if (!d_head || !d_count || captured_num_tokens <= 0 || max_seq_len <= 0)
            return;
        cuda_kv_sequence_state_advance_dynamic_kernel<<<1, 1, 0, stream>>>(
            d_head, d_count, d_append_count, captured_num_tokens, max_seq_len);
    }

    extern "C" bool cuda_kv_sequence_state_publish(
        int *d_heads,
        int *d_counts,
        const int32_t *target_cached_tokens,
        const int32_t *accepted_state_counts,
        const int32_t *publication_ok_flags,
        int n_layers,
        int batch_size,
        int first_seq_idx,
        int request_count,
        int max_seq_len,
        cudaStream_t stream)
    {
        if (!d_heads || !d_counts ||
            !target_cached_tokens || !accepted_state_counts ||
            !publication_ok_flags || !stream ||
            n_layers <= 0 || batch_size <= 0 ||
            first_seq_idx < 0 || request_count <= 0 ||
            first_seq_idx + request_count > batch_size ||
            max_seq_len <= 0)
        {
            return false;
        }

        cuda_kv_sequence_state_publish_kernel<<<
            dim3(request_count, n_layers), dim3(1), 0, stream>>>(
            d_heads,
            d_counts,
            target_cached_tokens,
            accepted_state_counts,
            publication_ok_flags,
            batch_size,
            first_seq_idx,
            request_count,
            max_seq_len);
        return cudaGetLastError() == cudaSuccess;
    }

    // =========================================================================
    // CUDARingKVCache Implementation
    // =========================================================================

    template <ActivationPrecision Precision>
    CUDARingKVCache<Precision>::CUDARingKVCache(
        int n_layers, int batch_size, int max_seq_len,
        int n_kv_heads, int head_dim, int device_id)
        : ICUDARingKVCache(n_layers, batch_size, max_seq_len,
                           n_kv_heads, head_dim, n_kv_heads * head_dim, device_id),
          local_n_kv_heads_(n_kv_heads), kv_head_start_(0),
          kv_storage_dim_((Precision == ActivationPrecision::Q8_1)
                              ? ((n_kv_heads * head_dim + Q8_1Block::BLOCK_SIZE - 1) / Q8_1Block::BLOCK_SIZE)
                              : (n_kv_heads * head_dim)),
          is_sharded_(false), device_ctx_(nullptr)
    {
        LOG_DEBUG("[CUDARingKVCache] Creating cache: "
                  << n_layers << " layers, batch=" << batch_size
                  << ", max_seq=" << max_seq_len << ", kv_dim=" << kv_dim_
                  << ", precision=" << static_cast<int>(Precision));

        cudaSetDevice(device_id_);

        // Allocate entries
        entries_.resize(n_layers_);
        for (int layer = 0; layer < n_layers_; ++layer)
        {
            entries_[layer].resize(batch_size_);
            for (int seq = 0; seq < batch_size_; ++seq)
            {
                allocate_entry(entries_[layer][seq]);
            }
        }

        // Initialize tensor_views_ storage for get_k()/get_v() wrappers
        tensor_views_.resize(n_layers_);
        for (int layer = 0; layer < n_layers_; ++layer)
        {
            tensor_views_[layer].resize(batch_size_);
            // Views are created lazily in get_k()/get_v()
        }

        LOG_DEBUG("[CUDARingKVCache] Allocated "
                  << (n_layers_ * batch_size_ * 4 * max_seq_len_ * kv_dim_ * sizeof(DataT)) / (1024 * 1024)
                  << " MB total (including scratch)");

        allocateDeviceParams();
    }

    template <ActivationPrecision Precision>
    CUDARingKVCache<Precision>::CUDARingKVCache(
        int n_layers, int batch_size, int max_seq_len,
        int n_kv_heads, int head_dim, IWorkerGPUContext *ctx)
        : ICUDARingKVCache(n_layers, batch_size, max_seq_len,
                           n_kv_heads, head_dim, n_kv_heads * head_dim, 0),
          local_n_kv_heads_(n_kv_heads), kv_head_start_(0),
          kv_storage_dim_((Precision == ActivationPrecision::Q8_1)
                              ? ((n_kv_heads * head_dim + Q8_1Block::BLOCK_SIZE - 1) / Q8_1Block::BLOCK_SIZE)
                              : (n_kv_heads * head_dim)),
          is_sharded_(false), device_ctx_(nullptr)
    {
        if (!ctx)
        {
            throw std::runtime_error("[CUDARingKVCache] Device context is null");
        }
        if (!ctx->isInitialized())
        {
            throw std::runtime_error("[CUDARingKVCache] Device context is not initialized");
        }

        device_ctx_ = ctx;
        device_id_ = ctx->deviceOrdinal();

        LOG_DEBUG("[CUDARingKVCache] Creating cache with device context: "
                  << n_layers << " layers, batch=" << batch_size
                  << ", max_seq=" << max_seq_len << ", kv_dim=" << kv_dim_
                  << ", device=" << device_id_
                  << ", precision=" << static_cast<int>(Precision));

        cudaSetDevice(device_id_);

        // Allocate entries
        entries_.resize(n_layers_);
        for (int layer = 0; layer < n_layers_; ++layer)
        {
            entries_[layer].resize(batch_size_);
            for (int seq = 0; seq < batch_size_; ++seq)
            {
                allocate_entry(entries_[layer][seq]);
            }
        }

        // Initialize tensor_views_ storage for get_k()/get_v() wrappers
        tensor_views_.resize(n_layers_);
        for (int layer = 0; layer < n_layers_; ++layer)
        {
            tensor_views_[layer].resize(batch_size_);
            // Views are created lazily in get_k()/get_v()
        }

        LOG_DEBUG("[CUDARingKVCache] Allocated "
                  << (n_layers_ * batch_size_ * 4 * max_seq_len_ * kv_dim_ * sizeof(DataT)) / (1024 * 1024)
                  << " MB total (including scratch)");

        allocateDeviceParams();
    }

    template <ActivationPrecision Precision>
    CUDARingKVCache<Precision>::CUDARingKVCache(
        int n_layers, int batch_size, int max_seq_len,
        int n_kv_heads, int local_n_kv_heads, int kv_head_start,
        int head_dim, int device_id)
        : ICUDARingKVCache(n_layers, batch_size, max_seq_len,
                           n_kv_heads, head_dim, local_n_kv_heads * head_dim, device_id),
          local_n_kv_heads_(local_n_kv_heads), kv_head_start_(kv_head_start),
          kv_storage_dim_((Precision == ActivationPrecision::Q8_1)
                              ? ((local_n_kv_heads * head_dim + Q8_1Block::BLOCK_SIZE - 1) / Q8_1Block::BLOCK_SIZE)
                              : (local_n_kv_heads * head_dim)),
          is_sharded_(local_n_kv_heads != n_kv_heads), device_ctx_(nullptr)
    {
        LOG_DEBUG("[CUDARingKVCache] Creating sharded cache: "
                  << n_layers << " layers, batch=" << batch_size
                  << ", max_seq=" << max_seq_len << ", total_kv_heads=" << n_kv_heads
                  << ", local_kv_heads=" << local_n_kv_heads << ", kv_head_start=" << kv_head_start
                  << ", local_kv_dim=" << kv_dim_
                  << ", precision=" << static_cast<int>(Precision));

        cudaSetDevice(device_id_);

        // Allocate entries
        entries_.resize(n_layers_);
        for (int layer = 0; layer < n_layers_; ++layer)
        {
            entries_[layer].resize(batch_size_);
            for (int seq = 0; seq < batch_size_; ++seq)
            {
                allocate_entry(entries_[layer][seq]);
            }
        }

        // Initialize tensor_views_ storage for get_k()/get_v() wrappers
        tensor_views_.resize(n_layers_);
        for (int layer = 0; layer < n_layers_; ++layer)
        {
            tensor_views_[layer].resize(batch_size_);
            // Views are created lazily in get_k()/get_v()
        }

        LOG_DEBUG("[CUDARingKVCache] Allocated "
                  << (n_layers_ * batch_size_ * 4 * max_seq_len_ * kv_dim_ * sizeof(DataT)) / (1024 * 1024)
                  << " MB total (including scratch)");

        allocateDeviceParams();
    }

    template <ActivationPrecision Precision>
    CUDARingKVCache<Precision>::CUDARingKVCache(
        int n_layers, int batch_size, int max_seq_len,
        int n_kv_heads, int local_n_kv_heads, int kv_head_start,
        int head_dim, IWorkerGPUContext *ctx)
        : ICUDARingKVCache(n_layers, batch_size, max_seq_len,
                           n_kv_heads, head_dim, local_n_kv_heads * head_dim, 0),
          local_n_kv_heads_(local_n_kv_heads), kv_head_start_(kv_head_start),
          kv_storage_dim_((Precision == ActivationPrecision::Q8_1)
                              ? ((local_n_kv_heads * head_dim + Q8_1Block::BLOCK_SIZE - 1) / Q8_1Block::BLOCK_SIZE)
                              : (local_n_kv_heads * head_dim)),
          is_sharded_(local_n_kv_heads != n_kv_heads), device_ctx_(nullptr)
    {
        if (!ctx)
        {
            throw std::runtime_error("[CUDARingKVCache] Device context is null");
        }
        if (!ctx->isInitialized())
        {
            throw std::runtime_error("[CUDARingKVCache] Device context is not initialized");
        }

        device_ctx_ = ctx;
        device_id_ = ctx->deviceOrdinal();

        LOG_DEBUG("[CUDARingKVCache] Creating sharded cache with device context: "
                  << n_layers << " layers, batch=" << batch_size
                  << ", max_seq=" << max_seq_len << ", total_kv_heads=" << n_kv_heads
                  << ", local_kv_heads=" << local_n_kv_heads << ", kv_head_start=" << kv_head_start
                  << ", local_kv_dim=" << kv_dim_
                  << ", device=" << device_id_
                  << ", precision=" << static_cast<int>(Precision));

        cudaSetDevice(device_id_);

        // Allocate entries
        entries_.resize(n_layers_);
        for (int layer = 0; layer < n_layers_; ++layer)
        {
            entries_[layer].resize(batch_size_);
            for (int seq = 0; seq < batch_size_; ++seq)
            {
                allocate_entry(entries_[layer][seq]);
            }
        }

        // Initialize tensor_views_ storage for get_k()/get_v() wrappers
        tensor_views_.resize(n_layers_);
        for (int layer = 0; layer < n_layers_; ++layer)
        {
            tensor_views_[layer].resize(batch_size_);
            // Views are created lazily in get_k()/get_v()
        }

        LOG_DEBUG("[CUDARingKVCache] Allocated "
                  << (n_layers_ * batch_size_ * 4 * max_seq_len_ * kv_dim_ * sizeof(DataT)) / (1024 * 1024)
                  << " MB total (including scratch)");

        allocateDeviceParams();
    }

    template <ActivationPrecision Precision>
    CUDARingKVCache<Precision>::~CUDARingKVCache()
    {
        // Check if CUDA runtime is shutting down
        cudaError_t set_err = cudaSetDevice(device_id_);
        if (set_err == cudaErrorCudartUnloading || set_err == cudaErrorNoDevice)
        {
            // Runtime is shutting down, skip cleanup
            return;
        }

        // Note: d_head_params_/h_head_params_ are freed by ~CUDARingKVCacheBase()

        for (auto &layer_entries : entries_)
        {
            for (auto &entry : layer_entries)
            {
                free_entry(entry);
            }
        }

        // Free RoPE shadow buffers
        for (auto &layer_shadows : rope_shadows_)
        {
            for (auto &shadow : layer_shadows)
            {
                if (shadow.d_K)
                    cudaFree(shadow.d_K);
                if (shadow.d_V)
                    cudaFree(shadow.d_V);
            }
        }
    }

    template <ActivationPrecision Precision>
    void CUDARingKVCache<Precision>::clear()
    {
        cudaSetDevice(device_id_);
        cudaStream_t clear_stream = getEffectiveStream(nullptr);

        // Drop conversion scratch so shorter follow-up requests cannot observe
        // stale lanes from a previous longer append/read conversion path.
        freeConvScratch();

        // Keep dynamic-head allocations stable for graph caches, but reset both
        // host and device values to match empty ring entries before next append.
        const int num_entries = n_layers_ * batch_size_;
        if (h_head_params_ && num_entries > 0)
        {
            std::memset(h_head_params_, 0, static_cast<size_t>(num_entries) * sizeof(int));
        }
        if (h_count_params_ && num_entries > 0)
        {
            std::memset(h_count_params_, 0, static_cast<size_t>(num_entries) * sizeof(int));
        }
        if (d_head_params_ && num_entries > 0)
        {
            cudaError_t head_err = cudaMemsetAsync(d_head_params_, 0,
                                                   static_cast<size_t>(num_entries) * sizeof(int),
                                                   clear_stream);
            if (head_err != cudaSuccess)
            {
                LOG_WARN("[CUDARingKVCache::clear] head params memset failed: "
                         << cudaGetErrorString(head_err));
            }
        }
        if (d_count_params_ && num_entries > 0)
        {
            cudaError_t count_err = cudaMemsetAsync(d_count_params_, 0,
                                                    static_cast<size_t>(num_entries) * sizeof(int),
                                                    clear_stream);
            if (count_err != cudaSuccess)
            {
                LOG_WARN("[CUDARingKVCache::clear] count params memset failed: "
                         << cudaGetErrorString(count_err));
            }
        }

        // RoPE-on-read shadows and tensor views are request-scoped wrappers over
        // cache contents. Reset them so no stale view can survive a cache clear.
        for (auto &layer_shadows : rope_shadows_)
        {
            for (auto &shadow : layer_shadows)
            {
                if (shadow.d_K)
                {
                    cudaFree(shadow.d_K);
                    shadow.d_K = nullptr;
                }
                if (shadow.d_V)
                {
                    cudaFree(shadow.d_V);
                    shadow.d_V = nullptr;
                }
                shadow.converted_count = 0;
                shadow.last_head = -1;
                shadow.rope_applied = false;
                shadow.k_view.reset();
                shadow.v_view.reset();
            }
        }
        rope_shadows_.clear();
        for (auto &layer_views : tensor_views_)
        {
            for (auto &views : layer_views)
            {
                views[0].reset();
                views[1].reset();
            }
        }

        // Scrub persistent device storage and linearization scratch. Metadata
        // reset alone should be sufficient logically, but zeroing matches a new
        // cache object and protects snapshot/debug read paths from stale rows.
        const size_t buffer_size = static_cast<size_t>(max_seq_len_) *
                                   static_cast<size_t>(kv_storage_dim_) *
                                   sizeof(DataT);
        for (auto &layer_entries : entries_)
        {
            for (auto &entry : layer_entries)
            {
                if (entry.d_K)
                    cudaMemsetAsync(entry.d_K, 0, buffer_size, clear_stream);
                if (entry.d_V)
                    cudaMemsetAsync(entry.d_V, 0, buffer_size, clear_stream);
                if (entry.d_K_scratch)
                    cudaMemsetAsync(entry.d_K_scratch, 0, buffer_size, clear_stream);
                if (entry.d_V_scratch)
                    cudaMemsetAsync(entry.d_V_scratch, 0, buffer_size, clear_stream);
            }
        }

        cudaError_t sync_err = cudaStreamSynchronize(clear_stream);
        if (sync_err != cudaSuccess)
        {
            LOG_WARN("[CUDARingKVCache::clear] KV buffer clear sync failed: "
                     << cudaGetErrorString(sync_err));
        }

        // Reset compressed entries directly. Calling CUDARingKVCacheBase::clear()
        // would dispatch through virtual clear_layer(), which hybrid caches map
        // through global model layer ids instead of compressed FA indices.
        for (int layer = 0; layer < n_layers_; ++layer)
        {
            for (int seq = 0; seq < batch_size_; ++seq)
            {
                CUDARingKVCacheBase::clear_sequence(layer, seq);
            }
        }
        wrap_warned_ = false;
    }

    template <ActivationPrecision Precision>
    void CUDARingKVCache<Precision>::allocate_entry(EntryT &entry)
    {
        size_t buffer_size = static_cast<size_t>(max_seq_len_) * static_cast<size_t>(kv_storage_dim_) * sizeof(DataT);

        // Main K/V buffers
        cudaMalloc(&entry.d_K, buffer_size);
        cudaMalloc(&entry.d_V, buffer_size);

        // Per-sequence scratch buffers for linearization
        cudaMalloc(&entry.d_K_scratch, buffer_size);
        cudaMalloc(&entry.d_V_scratch, buffer_size);

        // Initialize state
        entry.head = 0;
        entry.count = 0;
        entry.scratch_valid = false;
    }

    template <ActivationPrecision Precision>
    void CUDARingKVCache<Precision>::free_entry(EntryT &entry)
    {
        if (entry.d_K)
        {
            cudaError_t err = cudaFree(entry.d_K);
            if (err != cudaSuccess && err != cudaErrorCudartUnloading && err != cudaErrorNoDevice)
            {
                fprintf(stderr, "WARNING: cudaFree(d_K) failed: %s\n", cudaGetErrorString(err));
            }
        }
        if (entry.d_V)
        {
            cudaError_t err = cudaFree(entry.d_V);
            if (err != cudaSuccess && err != cudaErrorCudartUnloading && err != cudaErrorNoDevice)
            {
                fprintf(stderr, "WARNING: cudaFree(d_V) failed: %s\n", cudaGetErrorString(err));
            }
        }
        if (entry.d_K_scratch)
        {
            cudaError_t err = cudaFree(entry.d_K_scratch);
            if (err != cudaSuccess && err != cudaErrorCudartUnloading && err != cudaErrorNoDevice)
            {
                fprintf(stderr, "WARNING: cudaFree(d_K_scratch) failed: %s\n", cudaGetErrorString(err));
            }
        }
        if (entry.d_V_scratch)
        {
            cudaError_t err = cudaFree(entry.d_V_scratch);
            if (err != cudaSuccess && err != cudaErrorCudartUnloading && err != cudaErrorNoDevice)
            {
                fprintf(stderr, "WARNING: cudaFree(d_V_scratch) failed: %s\n", cudaGetErrorString(err));
            }
        }

        entry.d_K = nullptr;
        entry.d_V = nullptr;
        entry.d_K_scratch = nullptr;
        entry.d_V_scratch = nullptr;
    }

    // =========================================================================
    // Dynamic Head Kernel Launcher (graph-capturable)
    // =========================================================================

    // Forward declarations for dynamic wrappers
    extern "C" void cuda_ring_append_dynamic_fp32(
        float *, float *, const float *, const float *,
        const int *, int, int, int, cudaStream_t);
    extern "C" void cuda_ring_append_dynamic_fp16(
        __half *, __half *, const __half *, const __half *,
        const int *, int, int, int, cudaStream_t);
    extern "C" void cuda_ring_append_dynamic_bf16(
        __nv_bfloat16 *, __nv_bfloat16 *, const __nv_bfloat16 *, const __nv_bfloat16 *,
        const int *, int, int, int, cudaStream_t);
    extern "C" void cuda_ring_append_dynamic_q8_1(
        Q8_1Block *, Q8_1Block *, const Q8_1Block *, const Q8_1Block *,
        const int *, int, int, int, cudaStream_t);
    extern "C" void cuda_kv_sequence_state_advance(
        int *, int *, int, int, cudaStream_t);
    extern "C" void cuda_kv_sequence_state_advance_dynamic(
        int *, int *, const int *, int, int, cudaStream_t);

    template <ActivationPrecision Precision>
    void CUDARingKVCache<Precision>::launch_append_kernel_dynamic(
        EntryT &entry, const DataT *d_k, const DataT *d_v,
        const int *d_head, int num_tokens, cudaStream_t stream)
    {
        if constexpr (Precision == ActivationPrecision::FP32)
        {
            cuda_ring_append_dynamic_fp32(
                entry.d_K, entry.d_V, d_k, d_v,
                d_head, max_seq_len_, kv_dim_, num_tokens, stream);
        }
        else if constexpr (Precision == ActivationPrecision::FP16)
        {
            cuda_ring_append_dynamic_fp16(
                entry.d_K, entry.d_V, d_k, d_v,
                d_head, max_seq_len_, kv_dim_, num_tokens, stream);
        }
        else if constexpr (Precision == ActivationPrecision::BF16)
        {
            cuda_ring_append_dynamic_bf16(
                entry.d_K, entry.d_V, d_k, d_v,
                d_head, max_seq_len_, kv_storage_dim_, num_tokens, stream);
        }
        else if constexpr (Precision == ActivationPrecision::Q8_1)
        {
            cuda_ring_append_dynamic_q8_1(
                entry.d_K, entry.d_V, d_k, d_v,
                d_head, max_seq_len_, kv_storage_dim_, num_tokens, stream);
        }
    }

    template <ActivationPrecision Precision>
    bool CUDARingKVCache<Precision>::append(
        int layer, int seq_idx,
        const void *d_k, const void *d_v,
        int num_tokens, cudaStream_t stream)
    {
        return append_typed(layer, seq_idx,
                            static_cast<const DataT *>(d_k),
                            static_cast<const DataT *>(d_v),
                            num_tokens, stream);
    }

    template <ActivationPrecision Precision>
    bool CUDARingKVCache<Precision>::append_typed(
        int layer, int seq_idx,
        const DataT *d_k, const DataT *d_v,
        int num_tokens, cudaStream_t stream)
    {
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
        {
            LOG_ERROR("[CUDARingKVCache::append] Invalid layer=" << layer << " or seq_idx=" << seq_idx);
            return false;
        }

        EntryT &entry = entries_[layer][seq_idx];
        const bool capture_active = isGraphCaptureActive();
        cudaStream_t effective_stream = stream;
        if (!effective_stream)
        {
            if (capture_active)
            {
                LOG_ERROR("[CUDARingKVCache::append] Explicit CUDA stream required during graph capture");
                return false;
            }
            // Keep sequence-state ownership coherent on device even for legacy
            // callers that pass nullptr. This is an explicit worker stream, not
            // CUDA's device-default stream, so later metadata uploads are ordered
            // with the payload append instead of silently leaving stale counters.
            effective_stream = device_ctx_
                                   ? static_cast<cudaStream_t>(device_ctx_->defaultStream())
                                   : static_cast<cudaStream_t>(
                                         GPUDeviceContextPool::instance()
                                             .getNvidiaContext(device_id_)
                                             .defaultStream());
        }

        // Check if we would exceed capacity (ring buffer overwrites oldest)
        if (!capture_active && entry.count + num_tokens > max_seq_len_)
        {
            // Auto-evict oldest tokens to make room
            int to_evict = entry.count + num_tokens - max_seq_len_;
            entry.count -= to_evict;
            total_evicted_ += to_evict;
            if (!wrap_warned_)
            {
                LOG_WARN("Context window full (" << max_seq_len_
                                                 << " tokens). Sliding window is now overwriting oldest tokens. "
                                                 << "Use -c <size> to increase context length.");
                wrap_warned_ = true;
            }
            LOG_DEBUG("[CUDARingKVCache::append] Auto-evicted " << to_evict << " tokens");
        }

        // Captured graphs read the ring head from a stable device scalar that
        // updateDynamicParams()/setDynamicHead() uploads before capture/replay.
        // Do not record H2D copies in the captured graph.
        if (capture_active && d_head_params_ && h_head_params_)
        {
            int idx = layer * batch_size_ + seq_idx;
            launch_append_kernel_dynamic(entry, d_k, d_v, &d_head_params_[idx], num_tokens, effective_stream);
            if (d_count_params_)
            {
                cuda_kv_sequence_state_advance_dynamic(
                    &d_head_params_[idx], &d_count_params_[idx],
                    deviceDynamicAppendCountPtr(layer, seq_idx),
                    num_tokens, max_seq_len_, effective_stream);
            }
        }
        else
        {
            // Fallback: scalar head argument (non-graph path)
            launch_append_kernel(entry, d_k, d_v, num_tokens, effective_stream);
        }

        if (!capture_active)
        {
            // Update ring buffer state only for real execution. During graph
            // capture, kernels are recorded but not executed until launch, and
            // replay callbacks advance host metadata after that launch.
            entry.head = (entry.head + num_tokens) % max_seq_len_;
            entry.count += num_tokens;
            entry.scratch_valid = false; // Scratch is stale after append
            if (d_count_params_ && !uploadHostDeviceParamMirror(layer, seq_idx, effective_stream))
                return false;
        }

        return true;
    }

    template <ActivationPrecision Precision>
    bool CUDARingKVCache<Precision>::appendVerifierRowsDecodeEquivalent(
        int layer,
        int seq_idx,
        const ITensor *K,
        const ITensor *V,
        int verifier_rows,
        void *gpu_stream)
    {
        if (layer < 0 || layer >= n_layers_ ||
            seq_idx < 0 || seq_idx >= batch_size_ ||
            verifier_rows < 1 || verifier_rows > 4 ||
            !K || !V || !gpu_stream)
        {
            LOG_ERROR("[CUDARingKVCache] Invalid verifier append request: layer="
                      << layer << " n_layers=" << n_layers_
                      << " first_layer=" << first_layer_index()
                      << " seq_idx=" << seq_idx << " batch_size=" << batch_size_
                      << " rows=" << verifier_rows
                      << " K=" << (K ? "set" : "null")
                      << " V=" << (V ? "set" : "null")
                      << " stream=" << gpu_stream);
            return false;
        }

        const auto layout_for = [&](const ITensor *tensor,
                                    const char *label,
                                    bool *head_major,
                                    int *convert_rows,
                                    int *convert_cols) -> bool
        {
            if (!tensor || !head_major || !convert_rows || !convert_cols ||
                tensor->shape().size() < 2)
            {
                return false;
            }

            const size_t rows = tensor->shape()[0];
            const size_t cols = tensor->shape()[1];
            const bool position_major =
                rows >= static_cast<size_t>(verifier_rows) &&
                cols == static_cast<size_t>(kv_dim_);
            const bool verifier_head_major =
                rows == static_cast<size_t>(local_n_kv_heads_) *
                            static_cast<size_t>(verifier_rows) &&
                cols == static_cast<size_t>(head_dim_);

            if (!position_major && !verifier_head_major)
            {
                LOG_ERROR("[CUDARingKVCache] Unsupported verifier " << label
                                                                    << " source shape ["
                                                                    << rows << "," << cols
                                                                    << "] rows=" << verifier_rows
                                                                    << " local_heads="
                                                                    << local_n_kv_heads_
                                                                    << " head_dim=" << head_dim_
                                                                    << " kv_dim=" << kv_dim_);
                return false;
            }

            *head_major = verifier_head_major && !position_major;
            *convert_rows = *head_major ? local_n_kv_heads_ * verifier_rows : verifier_rows;
            *convert_cols = *head_major ? head_dim_ : kv_dim_;
            return true;
        };

        bool k_head_major = false;
        bool v_head_major = false;
        int k_convert_rows = 0;
        int k_convert_cols = 0;
        int v_convert_rows = 0;
        int v_convert_cols = 0;
        if (!layout_for(K, "K", &k_head_major, &k_convert_rows, &k_convert_cols) ||
            !layout_for(V, "V", &v_head_major, &v_convert_rows, &v_convert_cols))
        {
            return false;
        }

        const void *d_k_src = K->gpu_data_ptr();
        const void *d_v_src = V->gpu_data_ptr();
        if (!d_k_src || !d_v_src)
        {
            LOG_ERROR("[CUDARingKVCache] Verifier append requires device-resident K/V tensors");
            return false;
        }

        cudaStream_t stream = static_cast<cudaStream_t>(gpu_stream);
        const auto target_type = []() constexpr
        {
            if constexpr (Precision == ActivationPrecision::FP32)
                return TensorType::FP32;
            else if constexpr (Precision == ActivationPrecision::FP16)
                return TensorType::FP16;
            else if constexpr (Precision == ActivationPrecision::BF16)
                return TensorType::BF16;
            else
                return TensorType::Q8_1;
        }();

        const DataT *typed_k = static_cast<const DataT *>(d_k_src);
        const DataT *typed_v = static_cast<const DataT *>(d_v_src);
        if (K->native_type() != target_type || V->native_type() != target_type)
        {
            if constexpr (Precision == ActivationPrecision::FP16)
            {
                const size_t k_bytes = static_cast<size_t>(k_convert_rows) *
                                       static_cast<size_t>(k_convert_cols) * sizeof(uint16_t);
                const size_t v_bytes = static_cast<size_t>(v_convert_rows) *
                                       static_cast<size_t>(v_convert_cols) * sizeof(uint16_t);
                if (!ensureConvScratch(std::max(k_bytes, v_bytes)))
                    return false;
                if (!cuda_convert_tensor_to_fp16(d_k_src, K->native_type(),
                                                 static_cast<uint16_t *>(conv_scratch_k_),
                                                 k_convert_rows * k_convert_cols, stream) ||
                    !cuda_convert_tensor_to_fp16(d_v_src, V->native_type(),
                                                 static_cast<uint16_t *>(conv_scratch_v_),
                                                 v_convert_rows * v_convert_cols, stream))
                {
                    LOG_ERROR("[CUDARingKVCache] FP16 verifier append conversion failed");
                    return false;
                }
                typed_k = static_cast<const DataT *>(conv_scratch_k_);
                typed_v = static_cast<const DataT *>(conv_scratch_v_);
            }
            else if constexpr (Precision == ActivationPrecision::Q8_1)
            {
                const auto q8_bytes = [](int rows, int cols) -> size_t
                {
                    const int blocks = (cols + Q8_1Block::BLOCK_SIZE - 1) / Q8_1Block::BLOCK_SIZE;
                    return static_cast<size_t>(rows) * static_cast<size_t>(blocks) * sizeof(Q8_1Block);
                };
                if (!ensureConvScratch(std::max(q8_bytes(k_convert_rows, k_convert_cols),
                                                q8_bytes(v_convert_rows, v_convert_cols))))
                    return false;
                if (!cuda_convert_tensor_to_q8_1(d_k_src, K->native_type(),
                                                 static_cast<Q8_1Block *>(conv_scratch_k_),
                                                 k_convert_rows, k_convert_cols, stream) ||
                    !cuda_convert_tensor_to_q8_1(d_v_src, V->native_type(),
                                                 static_cast<Q8_1Block *>(conv_scratch_v_),
                                                 v_convert_rows, v_convert_cols, stream))
                {
                    LOG_ERROR("[CUDARingKVCache] Q8_1 verifier append conversion failed");
                    return false;
                }
                typed_k = static_cast<const DataT *>(conv_scratch_k_);
                typed_v = static_cast<const DataT *>(conv_scratch_v_);
            }
            else
            {
                LOG_ERROR("[CUDARingKVCache] Verifier append conversion is unsupported for cache precision "
                          << static_cast<int>(Precision));
                return false;
            }
        }

        EntryT &entry = entries_[layer][seq_idx];
        const bool capture_active = isGraphCaptureActive();
        int source_start = 0;
        int rows_to_write = verifier_rows;
        if (!capture_active && entry.count + verifier_rows > max_seq_len_)
        {
            const int to_evict = entry.count + verifier_rows - max_seq_len_;
            if (to_evict > entry.count)
            {
                source_start = to_evict - entry.count;
                rows_to_write = verifier_rows - source_start;
                total_evicted_ += entry.count + source_start;
                entry.count = 0;
            }
            else
            {
                entry.count -= to_evict;
                total_evicted_ += to_evict;
            }
            if (!wrap_warned_)
            {
                LOG_WARN("Context window full (" << max_seq_len_
                                                 << " tokens). Sliding window is now overwriting oldest tokens. "
                                                 << "Use -c <size> to increase context length.");
                wrap_warned_ = true;
            }
        }

        const int head_storage_dim =
            (Precision == ActivationPrecision::Q8_1)
                ? (head_dim_ + Q8_1Block::BLOCK_SIZE - 1) / Q8_1Block::BLOCK_SIZE
                : head_dim_;

        if (capture_active && d_head_params_ && h_head_params_)
        {
            const int idx = layer * batch_size_ + seq_idx;
            cuda_ring_append_verifier_rows_dynamic_typed<DataT>(
                entry.d_K, entry.d_V, typed_k, typed_v,
                &d_head_params_[idx], max_seq_len_, kv_storage_dim_, head_storage_dim,
                verifier_rows, source_start, rows_to_write,
                k_head_major, v_head_major, stream);
            if (d_count_params_)
            {
                cuda_kv_sequence_state_advance(
                    &d_head_params_[idx], &d_count_params_[idx],
                    rows_to_write, max_seq_len_, stream);
            }
        }
        else
        {
            cuda_ring_append_verifier_rows_typed<DataT>(
                entry.d_K, entry.d_V, typed_k, typed_v,
                entry.head, max_seq_len_, kv_storage_dim_, head_storage_dim,
                verifier_rows, source_start, rows_to_write,
                k_head_major, v_head_major, stream);
        }

        const cudaError_t launch_err = cudaGetLastError();
        if (launch_err != cudaSuccess)
        {
            LOG_ERROR("[CUDARingKVCache] Verifier append kernel launch failed: "
                      << cudaGetErrorString(launch_err));
            return false;
        }

        if (!capture_active)
        {
            entry.head = (entry.head + rows_to_write) % max_seq_len_;
            entry.count += rows_to_write;
            entry.scratch_valid = false;
            invalidateRoPEShadow(layer, seq_idx);
            if (d_count_params_ && !uploadHostDeviceParamMirror(layer, seq_idx, stream))
                return false;
        }

        return true;
    }

    template <ActivationPrecision Precision>
    void CUDARingKVCache<Precision>::launch_append_kernel(
        EntryT &entry, const DataT *d_k, const DataT *d_v,
        int num_tokens, cudaStream_t stream)
    {
        if constexpr (Precision == ActivationPrecision::FP32)
        {
            cuda_ring_append_fp32(
                entry.d_K, entry.d_V, d_k, d_v,
                entry.head, max_seq_len_, kv_storage_dim_, num_tokens, stream);
        }
        else if constexpr (Precision == ActivationPrecision::FP16)
        {
            cuda_ring_append_fp16(
                entry.d_K, entry.d_V, d_k, d_v,
                entry.head, max_seq_len_, kv_storage_dim_, num_tokens, stream);
        }
        else if constexpr (Precision == ActivationPrecision::BF16)
        {
            cuda_ring_append_bf16(
                entry.d_K, entry.d_V, d_k, d_v,
                entry.head, max_seq_len_, kv_storage_dim_, num_tokens, stream);
        }
        else if constexpr (Precision == ActivationPrecision::Q8_1)
        {
            cuda_ring_append_q8_1(
                entry.d_K, entry.d_V, d_k, d_v,
                entry.head, max_seq_len_, kv_storage_dim_, num_tokens, stream);
        }
    }

    template <ActivationPrecision Precision>
    bool CUDARingKVCache<Precision>::get_kv_for_attention(
        int layer, int seq_idx,
        const void **d_k_out, const void **d_v_out,
        int *kv_len, cudaStream_t stream)
    {
        const DataT *k_typed;
        const DataT *v_typed;
        bool result = get_kv_typed(layer, seq_idx, &k_typed, &v_typed, kv_len, stream);
        *d_k_out = k_typed;
        *d_v_out = v_typed;
        return result;
    }

    template <ActivationPrecision Precision>
    bool CUDARingKVCache<Precision>::get_kv_typed(
        int layer, int seq_idx,
        const DataT **d_k_out, const DataT **d_v_out,
        int *kv_len, cudaStream_t stream)
    {
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
        {
            LOG_ERROR("[CUDARingKVCache::get_kv] Invalid layer=" << layer << " or seq_idx=" << seq_idx);
            return false;
        }

        EntryT &entry = entries_[layer][seq_idx];
        *kv_len = entry.count;

        if (entry.count == 0)
        {
            *d_k_out = nullptr;
            *d_v_out = nullptr;
            return true;
        }

        // Optimization: if not wrapped, return direct pointers
        if (!entry.is_wrapped(max_seq_len_))
        {
            int tail = entry.tail(max_seq_len_);
            *d_k_out = entry.d_K + static_cast<size_t>(tail) * kv_storage_dim_;
            *d_v_out = entry.d_V + static_cast<size_t>(tail) * kv_storage_dim_;
            return true;
        }

        // Buffer is wrapped - need to linearize
        if (!entry.scratch_valid)
        {
            linearize_entry(entry, stream);
            entry.scratch_valid = true;
            ++linearization_count_;
        }

        *d_k_out = entry.d_K_scratch;
        *d_v_out = entry.d_V_scratch;
        return true;
    }

    template <ActivationPrecision Precision>
    void CUDARingKVCache<Precision>::linearize_entry(EntryT &entry, cudaStream_t stream)
    {
        launch_linearize_kernel(entry, entry.d_K_scratch, entry.d_V_scratch, stream);
    }

    template <ActivationPrecision Precision>
    void CUDARingKVCache<Precision>::launch_linearize_kernel(
        const EntryT &entry, DataT *d_k_out, DataT *d_v_out,
        cudaStream_t stream)
    {
        int tail = entry.tail(max_seq_len_);

        if constexpr (Precision == ActivationPrecision::FP32)
        {
            cuda_ring_linearize_fp32(
                d_k_out, d_v_out, entry.d_K, entry.d_V,
                tail, entry.count, max_seq_len_, kv_dim_, stream);
        }
        else if constexpr (Precision == ActivationPrecision::FP16)
        {
            cuda_ring_linearize_fp16(
                d_k_out, d_v_out, entry.d_K, entry.d_V,
                tail, entry.count, max_seq_len_, kv_dim_, stream);
        }
        else if constexpr (Precision == ActivationPrecision::BF16)
        {
            cuda_ring_linearize_bf16(
                d_k_out, d_v_out, entry.d_K, entry.d_V,
                tail, entry.count, max_seq_len_, kv_storage_dim_, stream);
        }
        else if constexpr (Precision == ActivationPrecision::Q8_1)
        {
            cuda_ring_linearize_q8_1(
                d_k_out, d_v_out, entry.d_K, entry.d_V,
                tail, entry.count, max_seq_len_, kv_storage_dim_, stream);
        }
    }

    template <ActivationPrecision Precision>
    bool CUDARingKVCache<Precision>::linearize_to(
        int layer, int seq_idx,
        void *d_k_out, void *d_v_out,
        int *kv_len, cudaStream_t stream)
    {
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
        {
            LOG_ERROR("[CUDARingKVCache::linearize_to] Invalid layer=" << layer << " or seq_idx=" << seq_idx);
            return false;
        }

        const EntryT &entry = entries_[layer][seq_idx];
        *kv_len = entry.count;

        if (entry.count == 0)
        {
            return true;
        }

        launch_linearize_kernel(entry,
                                static_cast<DataT *>(d_k_out),
                                static_cast<DataT *>(d_v_out),
                                stream);
        return true;
    }

    template <ActivationPrecision Precision>
    typename IKVCache::KVCacheLogicalBlockLayout
    CUDARingKVCache<Precision>::logicalBlockLayout(int global_layer, int token_count) const
    {
        KVCacheLogicalBlockLayout layout;
        layout.k_precision = Precision;
        layout.v_precision = Precision;
        layout.layout = TensorLayout::KV_POS_HEAD_DIM;
        layout.local_kv_heads = local_n_kv_heads_;
        layout.kv_head_start = kv_head_start_;
        layout.head_dim = head_dim_;
        layout.device_resident = true;

        const int local_layer = remapLayerIndex(global_layer);
        if (local_layer < 0 || local_layer >= n_layers_ || batch_size_ <= 0 || token_count <= 0)
        {
            return layout;
        }

        const size_t row_bytes = static_cast<size_t>(kv_storage_dim_) * sizeof(DataT);
        layout.k_bytes = static_cast<size_t>(token_count) * row_bytes;
        layout.v_bytes = static_cast<size_t>(token_count) * row_bytes;
        return layout;
    }

    template <ActivationPrecision Precision>
    typename IKVCache::KVCacheSequenceState
    CUDARingKVCache<Precision>::sequenceState(int global_layer, int seq_idx) const
    {
        const int local_layer = remapLayerIndex(global_layer);
        if (local_layer < 0 || local_layer >= n_layers_ ||
            seq_idx < 0 || seq_idx >= batch_size_)
        {
            return {};
        }

        const auto &entry = entries_[local_layer][seq_idx];
        KVCacheSequenceState state;
        state.cached_tokens = entry.count;
        state.implementation_head = entry.head;
        state.wrapped = entry.is_wrapped(max_seq_len_);
        return state;
    }

    template <ActivationPrecision Precision>
    bool CUDARingKVCache<Precision>::exportLogicalBlock(
        const KVCacheLogicalBlockDescriptor &desc, void *dst_k, void *dst_v) const
    {
        const int local_layer = remapLayerIndex(desc.layer);
        if (local_layer < 0 || local_layer >= n_layers_ ||
            desc.seq_idx < 0 || desc.seq_idx >= batch_size_ ||
            desc.logical_token_start < 0 || desc.token_count < 0)
        {
            return false;
        }

        const auto &entry = entries_[local_layer][desc.seq_idx];
        if (desc.logical_token_start > entry.count ||
            desc.token_count > entry.count - desc.logical_token_start)
        {
            return false;
        }
        if (desc.token_count == 0)
        {
            return true;
        }
        if (!dst_k || !dst_v || !entry.d_K || !entry.d_V)
        {
            return false;
        }

        cudaSetDevice(device_id_);
        cudaStream_t stream = desc.stream ? static_cast<cudaStream_t>(desc.stream)
                                          : getEffectiveStream(nullptr);
        const size_t row_bytes = static_cast<size_t>(kv_storage_dim_) * sizeof(DataT);
        const int tail = entry.tail(max_seq_len_);
        auto *out_k = static_cast<uint8_t *>(dst_k);
        auto *out_v = static_cast<uint8_t *>(dst_v);

        for (int i = 0; i < desc.token_count; ++i)
        {
            const int logical = desc.logical_token_start + i;
            const int phys = (tail + logical) % max_seq_len_;
            const size_t dst_offset = static_cast<size_t>(i) * row_bytes;
            const size_t src_offset = static_cast<size_t>(phys) * static_cast<size_t>(kv_storage_dim_);

            cudaError_t err = cudaMemcpyAsync(out_k + dst_offset,
                                              entry.d_K + src_offset,
                                              row_bytes,
                                              cudaMemcpyDeviceToHost,
                                              stream);
            if (err != cudaSuccess)
            {
                LOG_ERROR("[CUDARingKVCache::exportLogicalBlock] K copy failed: "
                          << cudaGetErrorString(err));
                return false;
            }
            err = cudaMemcpyAsync(out_v + dst_offset,
                                  entry.d_V + src_offset,
                                  row_bytes,
                                  cudaMemcpyDeviceToHost,
                                  stream);
            if (err != cudaSuccess)
            {
                LOG_ERROR("[CUDARingKVCache::exportLogicalBlock] V copy failed: "
                          << cudaGetErrorString(err));
                return false;
            }
        }

        const cudaError_t sync_err = cudaStreamSynchronize(stream);
        if (sync_err != cudaSuccess)
        {
            LOG_ERROR("[CUDARingKVCache::exportLogicalBlock] stream sync failed: "
                      << cudaGetErrorString(sync_err));
            return false;
        }
        return true;
    }

    template <ActivationPrecision Precision>
    bool CUDARingKVCache<Precision>::importLogicalBlock(
        const KVCacheLogicalBlockDescriptor &desc, const void *src_k, const void *src_v)
    {
        const int local_layer = remapLayerIndex(desc.layer);
        if (local_layer < 0 || local_layer >= n_layers_ ||
            desc.seq_idx < 0 || desc.seq_idx >= batch_size_ ||
            desc.logical_token_start < 0 || desc.token_count < 0 ||
            desc.logical_token_start > max_seq_len_ ||
            desc.token_count > max_seq_len_ - desc.logical_token_start)
        {
            return false;
        }

        auto &entry = entries_[local_layer][desc.seq_idx];
        cudaStream_t stream = desc.stream ? static_cast<cudaStream_t>(desc.stream)
                                          : getEffectiveStream(nullptr);
        if (desc.token_count == 0)
        {
            if (desc.logical_token_start == 0)
            {
                entry.head = 0;
                entry.count = 0;
                entry.scratch_valid = false;
                if (d_count_params_ && stream && !uploadHostDeviceParamMirror(local_layer, desc.seq_idx, stream))
                    return false;
                invalidateRoPEShadow(local_layer, desc.seq_idx);
            }
            return true;
        }
        if (!src_k || !src_v || !entry.d_K || !entry.d_V)
        {
            return false;
        }
        if (desc.logical_token_start != entry.count ||
            entry.head != (entry.count % max_seq_len_))
        {
            return false;
        }

        cudaSetDevice(device_id_);
        const size_t row_bytes = static_cast<size_t>(kv_storage_dim_) * sizeof(DataT);
        const size_t bytes = static_cast<size_t>(desc.token_count) * row_bytes;
        const size_t dst_offset = static_cast<size_t>(desc.logical_token_start) *
                                  static_cast<size_t>(kv_storage_dim_);

        cudaError_t err = cudaMemcpyAsync(entry.d_K + dst_offset,
                                          src_k,
                                          bytes,
                                          cudaMemcpyHostToDevice,
                                          stream);
        if (err != cudaSuccess)
        {
            LOG_ERROR("[CUDARingKVCache::importLogicalBlock] K copy failed: "
                      << cudaGetErrorString(err));
            return false;
        }
        err = cudaMemcpyAsync(entry.d_V + dst_offset,
                              src_v,
                              bytes,
                              cudaMemcpyHostToDevice,
                              stream);
        if (err != cudaSuccess)
        {
            LOG_ERROR("[CUDARingKVCache::importLogicalBlock] V copy failed: "
                      << cudaGetErrorString(err));
            return false;
        }

        entry.count = desc.logical_token_start + desc.token_count;
        entry.head = entry.count % max_seq_len_;
        entry.scratch_valid = false;
        if (d_count_params_ && !uploadHostDeviceParamMirror(local_layer, desc.seq_idx, stream))
            return false;

        const cudaError_t sync_err = cudaStreamSynchronize(stream);
        if (sync_err != cudaSuccess)
        {
            LOG_ERROR("[CUDARingKVCache::importLogicalBlock] stream sync failed: "
                      << cudaGetErrorString(sync_err));
            return false;
        }

        invalidateRoPEShadow(local_layer, desc.seq_idx);
        return true;
    }

    template <ActivationPrecision Precision>
    bool CUDARingKVCache<Precision>::truncateSequence(int seq_idx, int cached_tokens, void *stream)
    {
        if (seq_idx < 0 || seq_idx >= batch_size_ ||
            cached_tokens < 0 || cached_tokens > max_seq_len_)
        {
            return false;
        }

        for (int layer = 0; layer < n_layers_; ++layer)
        {
            if (cached_tokens > entries_[layer][seq_idx].count)
            {
                return false;
            }
        }

        for (int layer = 0; layer < n_layers_; ++layer)
        {
            auto &entry = entries_[layer][seq_idx];
            if (entry.count == cached_tokens)
            {
                if (d_count_params_ && stream && !uploadHostDeviceParamMirror(layer, seq_idx, stream))
                    return false;
                continue;
            }
            if (cached_tokens == 0)
            {
                entry.head = 0;
            }
            else
            {
                const int tail = entry.tail(max_seq_len_);
                entry.head = (tail + cached_tokens) % max_seq_len_;
            }
            entry.count = cached_tokens;
            entry.scratch_valid = false;
            if (d_count_params_ && stream && !uploadHostDeviceParamMirror(layer, seq_idx, stream))
                return false;
            invalidateRoPEShadow(layer, seq_idx);
        }
        return true;
    }

    template <ActivationPrecision Precision>
    void CUDARingKVCache<Precision>::evict_oldest(int layer, int seq_idx, int num_tokens)
    {
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
        {
            return;
        }

        EntryT &entry = entries_[layer][seq_idx];
        int to_evict = std::min(num_tokens, entry.count);

        // O(1) eviction - just update count (tail moves implicitly)
        entry.count -= to_evict;
        entry.scratch_valid = false;
        total_evicted_ += to_evict;
    }

    template <ActivationPrecision Precision>
    void CUDARingKVCache<Precision>::evict_oldest_layer(int layer, int num_tokens)
    {
        if (layer < 0 || layer >= n_layers_)
        {
            return;
        }

        for (int seq = 0; seq < batch_size_; ++seq)
        {
            evict_oldest(layer, seq, num_tokens);
        }
    }

    template <ActivationPrecision Precision>
    int CUDARingKVCache<Precision>::gather_kv_batched(
        int layer, int num_seqs,
        void *d_k_out, void *d_v_out,
        int *kv_lens, int max_kv_len,
        cudaStream_t stream)
    {
        if (layer < 0 || layer >= n_layers_ || num_seqs > batch_size_)
        {
            LOG_ERROR("[CUDARingKVCache::gather_kv_batched] Invalid layer=" << layer);
            return -1;
        }

        // Collect entry pointers and metadata
        std::vector<EntryT *> entry_ptrs(num_seqs);
        int actual_max_kv_len = 0;

        for (int seq = 0; seq < num_seqs; ++seq)
        {
            entry_ptrs[seq] = &entries_[layer][seq];
            kv_lens[seq] = entry_ptrs[seq]->count;
            actual_max_kv_len = std::max(actual_max_kv_len, kv_lens[seq]);
        }

        if (actual_max_kv_len == 0)
        {
            return 0;
        }

        // Use provided max_kv_len or actual
        int out_max_kv_len = (max_kv_len > 0) ? max_kv_len : actual_max_kv_len;

        if (!launch_gather_kernel(entry_ptrs,
                                  static_cast<DataT *>(d_k_out),
                                  static_cast<DataT *>(d_v_out),
                                  kv_lens, out_max_kv_len,
                                  num_seqs, stream))
        {
            return -1;
        }

        return actual_max_kv_len;
    }

    template <ActivationPrecision Precision>
    bool CUDARingKVCache<Precision>::launch_gather_kernel(
        const std::vector<EntryT *> &entries,
        DataT *d_k_out, DataT *d_v_out,
        int *kv_lens, int max_kv_len,
        int num_seqs, cudaStream_t stream)
    {
        // Device arrays for kernel
        DataT **d_k_caches = nullptr;
        DataT **d_v_caches = nullptr;
        int *d_tails = nullptr;
        int *d_counts = nullptr;
        if (!hasWorkspace() || !workspace_ || !workspace_->isAllocated())
        {
            LOG_ERROR("[CUDARingKVCache] Workspace is required for batched gather; "
                      << "raw cudaMalloc fallback is disabled");
            return false;
        }

        // Use pre-allocated workspace buffers. These are required so gather stays
        // visible to the memory planner and remains graph-capture compatible.
        d_k_caches = static_cast<DataT **>(
            workspace_->getBuffer(KVCacheWorkspaceBuffers::BATCH_K_PTRS));
        d_v_caches = static_cast<DataT **>(
            workspace_->getBuffer(KVCacheWorkspaceBuffers::BATCH_V_PTRS));
        d_tails = static_cast<int *>(
            workspace_->getBuffer(KVCacheWorkspaceBuffers::BATCH_TAILS));
        d_counts = static_cast<int *>(
            workspace_->getBuffer(KVCacheWorkspaceBuffers::BATCH_COUNTS));

        if (!d_k_caches || !d_v_caches || !d_tails || !d_counts)
        {
            LOG_ERROR("[CUDARingKVCache] Missing required workspace buffers for batched gather");
            return false;
        }

        LOG_TRACE("[CUDARingKVCache] Using workspace buffers for gather, num_seqs=" << num_seqs);

        // Prepare host arrays
        std::vector<DataT *> h_k_caches(num_seqs);
        std::vector<DataT *> h_v_caches(num_seqs);
        std::vector<int> h_tails(num_seqs);
        std::vector<int> h_counts(num_seqs);

        for (int i = 0; i < num_seqs; ++i)
        {
            h_k_caches[i] = entries[i]->d_K;
            h_v_caches[i] = entries[i]->d_V;
            h_tails[i] = entries[i]->tail(max_seq_len_);
            h_counts[i] = entries[i]->count;
        }

        // Copy to device
        cudaMemcpyAsync(d_k_caches, h_k_caches.data(), num_seqs * sizeof(DataT *),
                        cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_v_caches, h_v_caches.data(), num_seqs * sizeof(DataT *),
                        cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_tails, h_tails.data(), num_seqs * sizeof(int),
                        cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_counts, h_counts.data(), num_seqs * sizeof(int),
                        cudaMemcpyHostToDevice, stream);

        // Launch kernel
        dim3 block(256);
        dim3 grid(max_kv_len, (kv_storage_dim_ + 255) / 256, num_seqs);

        ring_gather_batched_kernel<DataT><<<grid, block, 0, stream>>>(
            d_k_out, d_v_out,
            d_k_caches, d_v_caches,
            d_tails, d_counts,
            num_seqs, max_kv_len, max_seq_len_, kv_storage_dim_);
        return cudaGetLastError() == cudaSuccess;
    }

    // =========================================================================
    // IWorkspaceConsumer Implementation
    // =========================================================================

    template <ActivationPrecision Precision>
    WorkspaceRequirements CUDARingKVCache<Precision>::getWorkspaceRequirements(
        int m, int n, int k) const
    {
        // New callers pass m=max graph tokens and n=batch size so conversion
        // scratch can follow bucket/chunk size. Legacy one-arg callers used
        // m=batch size; keep that behavior and size scratch to max_seq_len_.
        (void)k;

        const bool has_token_hint = n > 0;
        const int actual_batch_size = has_token_hint ? n : ((m > 0) ? m : batch_size_);
        const int scratch_tokens = has_token_hint ? m : max_seq_len_;
        const int bounded_batch_size = std::max(1, actual_batch_size);
        const int bounded_scratch_tokens = std::max(1, scratch_tokens);
        const size_t fp32_scratch_bytes =
            static_cast<size_t>(bounded_scratch_tokens) * static_cast<size_t>(kv_dim_) * sizeof(float);
        const size_t fp16_scratch_bytes =
            static_cast<size_t>(bounded_scratch_tokens) * static_cast<size_t>(kv_dim_) * sizeof(uint16_t);
        const size_t native_scratch_bytes =
            static_cast<size_t>(bounded_scratch_tokens) * static_cast<size_t>(kv_storage_dim_) * sizeof(DataT);
        const size_t conversion_scratch_bytes =
            std::max(fp32_scratch_bytes, std::max(fp16_scratch_bytes, native_scratch_bytes));

        WorkspaceRequirements reqs;

        // Buffer for K cache pointers: DataT* per sequence
        reqs.buffers.push_back({
            KVCacheWorkspaceBuffers::BATCH_K_PTRS,
            static_cast<size_t>(bounded_batch_size) * sizeof(DataT *),
            256, // Alignment
            true // Required: no raw cudaMalloc fallback in gather
        });

        // Buffer for V cache pointers: DataT* per sequence
        reqs.buffers.push_back({KVCacheWorkspaceBuffers::BATCH_V_PTRS,
                                static_cast<size_t>(bounded_batch_size) * sizeof(DataT *),
                                256,
                                true});

        // Buffer for tail indices: int per sequence
        reqs.buffers.push_back({KVCacheWorkspaceBuffers::BATCH_TAILS,
                                static_cast<size_t>(bounded_batch_size) * sizeof(int),
                                256,
                                true});

        // Buffer for count values: int per sequence
        reqs.buffers.push_back({KVCacheWorkspaceBuffers::BATCH_COUNTS,
                                static_cast<size_t>(bounded_batch_size) * sizeof(int),
                                256,
                                true});

        reqs.buffers.push_back({KVCacheWorkspaceBuffers::CONV_SCRATCH_K,
                                conversion_scratch_bytes,
                                256,
                                true});
        reqs.buffers.push_back({KVCacheWorkspaceBuffers::CONV_SCRATCH_V,
                                conversion_scratch_bytes,
                                256,
                                true});

        LOG_DEBUG("[CUDARingKVCache] Workspace requirements: batch_size="
                  << bounded_batch_size
                  << " scratch_tokens=" << bounded_scratch_tokens
                  << " BATCH_K_PTRS=" << bounded_batch_size * sizeof(DataT *)
                  << " BATCH_V_PTRS=" << bounded_batch_size * sizeof(DataT *)
                  << " BATCH_TAILS=" << bounded_batch_size * sizeof(int)
                  << " BATCH_COUNTS=" << bounded_batch_size * sizeof(int)
                  << " CONV_SCRATCH(each)=" << conversion_scratch_bytes);

        return reqs;
    }

    template <ActivationPrecision Precision>
    void CUDARingKVCache<Precision>::bindWorkspace(DeviceWorkspaceManager *workspace)
    {
        workspace_ = workspace;
        LOG_DEBUG("[CUDARingKVCache] Workspace bound: " << (workspace ? "yes" : "no"));
    }

    template <ActivationPrecision Precision>
    bool CUDARingKVCache<Precision>::hasWorkspace() const
    {
        return workspace_ != nullptr;
    }

    template <ActivationPrecision Precision>
    DeviceWorkspaceManager *CUDARingKVCache<Precision>::getWorkspace() const
    {
        return workspace_;
    }

    // =========================================================================
    // Explicit Template Instantiations
    // =========================================================================

    template class CUDARingKVCache<ActivationPrecision::FP32>;
    template class CUDARingKVCache<ActivationPrecision::FP16>;
    template class CUDARingKVCache<ActivationPrecision::BF16>;
    template class CUDARingKVCache<ActivationPrecision::Q8_1>;

    // =========================================================================
    // Factory Function
    // =========================================================================

    std::unique_ptr<ICUDARingKVCache> createCUDARingKVCache(
        ActivationPrecision precision,
        int n_layers, int batch_size, int max_seq_len,
        int n_kv_heads, int head_dim, int device_id)
    {
        switch (precision)
        {
        case ActivationPrecision::FP32:
            return std::make_unique<CUDARingKVCacheFP32>(
                n_layers, batch_size, max_seq_len, n_kv_heads, head_dim, device_id);

        case ActivationPrecision::FP16:
            return std::make_unique<CUDARingKVCacheFP16>(
                n_layers, batch_size, max_seq_len, n_kv_heads, head_dim, device_id);

        case ActivationPrecision::BF16:
            return std::make_unique<CUDARingKVCacheBF16>(
                n_layers, batch_size, max_seq_len, n_kv_heads, head_dim, device_id);

        case ActivationPrecision::Q8_1:
            return std::make_unique<CUDARingKVCacheQ8_1>(
                n_layers, batch_size, max_seq_len, n_kv_heads, head_dim, device_id);

        default:
            LOG_ERROR("[createCUDARingKVCache] Unsupported precision: "
                      << static_cast<int>(precision));
            return nullptr;
        }
    }

    // =========================================================================
    // Sharded Factory Function (for Tensor Parallelism)
    // =========================================================================

    std::unique_ptr<ICUDARingKVCache> createShardedCUDARingKVCache(
        ActivationPrecision precision,
        int n_layers, int batch_size, int max_seq_len,
        int n_kv_heads, int local_n_kv_heads, int kv_head_start,
        int head_dim, int device_id)
    {
        switch (precision)
        {
        case ActivationPrecision::FP32:
            return std::make_unique<CUDARingKVCacheFP32>(
                n_layers, batch_size, max_seq_len,
                n_kv_heads, local_n_kv_heads, kv_head_start,
                head_dim, device_id);

        case ActivationPrecision::FP16:
            return std::make_unique<CUDARingKVCacheFP16>(
                n_layers, batch_size, max_seq_len,
                n_kv_heads, local_n_kv_heads, kv_head_start,
                head_dim, device_id);

        case ActivationPrecision::BF16:
            return std::make_unique<CUDARingKVCacheBF16>(
                n_layers, batch_size, max_seq_len,
                n_kv_heads, local_n_kv_heads, kv_head_start,
                head_dim, device_id);

        case ActivationPrecision::Q8_1:
            return std::make_unique<CUDARingKVCacheQ8_1>(
                n_layers, batch_size, max_seq_len,
                n_kv_heads, local_n_kv_heads, kv_head_start,
                head_dim, device_id);

        default:
            LOG_ERROR("[createShardedCUDARingKVCache] Unsupported precision: "
                      << static_cast<int>(precision));
            return nullptr;
        }
    }

} // namespace llaminar2
