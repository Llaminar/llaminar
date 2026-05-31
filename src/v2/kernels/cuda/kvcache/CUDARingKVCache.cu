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
#include "../../../utils/Logger.h"
#include "../../kvcache/KVCacheDeviceParams.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "../../../tensors/BlockStructures.h"
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

        // Use dynamic head params when available and stream is provided.
        // This path is graph-capturable: the H2D copy and kernel launch are
        // recorded in the graph, and on replay setDynamicHead() updates the
        // pinned host buffer before the captured H2D re-reads it.
        if (d_head_params_ && h_head_params_ && stream)
        {
            int idx = layer * batch_size_ + seq_idx;
            h_head_params_[idx] = entry.head;
            cudaMemcpyAsync(&d_head_params_[idx], &h_head_params_[idx],
                            sizeof(int), cudaMemcpyHostToDevice, stream);
            launch_append_kernel_dynamic(entry, d_k, d_v, &d_head_params_[idx], num_tokens, stream);
        }
        else
        {
            // Fallback: scalar head argument (non-graph path)
            launch_append_kernel(entry, d_k, d_v, num_tokens, stream);
        }

        if (!capture_active)
        {
            // Update ring buffer state only for real execution. During graph
            // capture, kernels are recorded but not executed until launch, and
            // replay callbacks advance host metadata after that launch.
            entry.head = (entry.head + num_tokens) % max_seq_len_;
            entry.count += num_tokens;
            entry.scratch_valid = false; // Scratch is stale after append
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
        if (desc.token_count == 0)
        {
            if (desc.logical_token_start == 0)
            {
                entry.head = 0;
                entry.count = 0;
                entry.scratch_valid = false;
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
        cudaStream_t stream = desc.stream ? static_cast<cudaStream_t>(desc.stream)
                                          : getEffectiveStream(nullptr);
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

        const cudaError_t sync_err = cudaStreamSynchronize(stream);
        if (sync_err != cudaSuccess)
        {
            LOG_ERROR("[CUDARingKVCache::importLogicalBlock] stream sync failed: "
                      << cudaGetErrorString(sync_err));
            return false;
        }

        entry.count = desc.logical_token_start + desc.token_count;
        entry.head = entry.count % max_seq_len_;
        entry.scratch_valid = false;
        invalidateRoPEShadow(local_layer, desc.seq_idx);
        return true;
    }

    template <ActivationPrecision Precision>
    bool CUDARingKVCache<Precision>::truncateSequence(int seq_idx, int cached_tokens, void *stream)
    {
        (void)stream;
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

        launch_gather_kernel(entry_ptrs,
                             static_cast<DataT *>(d_k_out),
                             static_cast<DataT *>(d_v_out),
                             kv_lens, out_max_kv_len,
                             num_seqs, stream);

        return actual_max_kv_len;
    }

    template <ActivationPrecision Precision>
    void CUDARingKVCache<Precision>::launch_gather_kernel(
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
        bool using_workspace = false;

        if (hasWorkspace())
        {
            // Use pre-allocated workspace buffers (fast path)
            d_k_caches = static_cast<DataT **>(
                workspace_->getBuffer(KVCacheWorkspaceBuffers::BATCH_K_PTRS));
            d_v_caches = static_cast<DataT **>(
                workspace_->getBuffer(KVCacheWorkspaceBuffers::BATCH_V_PTRS));
            d_tails = static_cast<int *>(
                workspace_->getBuffer(KVCacheWorkspaceBuffers::BATCH_TAILS));
            d_counts = static_cast<int *>(
                workspace_->getBuffer(KVCacheWorkspaceBuffers::BATCH_COUNTS));

            if (d_k_caches && d_v_caches && d_tails && d_counts)
            {
                using_workspace = true;
                LOG_TRACE("[CUDARingKVCache] Using workspace buffers for gather, num_seqs=" << num_seqs);
            }
        }

        if (!using_workspace)
        {
            // Fallback to per-call allocation (backward compatibility)
            LOG_TRACE("[CUDARingKVCache] Fallback to per-call allocation for gather, num_seqs=" << num_seqs);
            cudaMalloc(&d_k_caches, num_seqs * sizeof(DataT *));
            cudaMalloc(&d_v_caches, num_seqs * sizeof(DataT *));
            cudaMalloc(&d_tails, num_seqs * sizeof(int));
            cudaMalloc(&d_counts, num_seqs * sizeof(int));
        }

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

        // Free temporary device arrays only if we allocated them
        if (!using_workspace)
        {
            cudaFreeAsync(d_k_caches, stream);
            cudaFreeAsync(d_v_caches, stream);
            cudaFreeAsync(d_tails, stream);
            cudaFreeAsync(d_counts, stream);
        }
    }

    // =========================================================================
    // IWorkspaceConsumer Implementation
    // =========================================================================

    template <ActivationPrecision Precision>
    WorkspaceRequirements CUDARingKVCache<Precision>::getWorkspaceRequirements(
        int m, int n, int k) const
    {
        // m = batch size (number of sequences in gather operation)
        // n, k unused for KV cache
        // Default to batch_size_ if m is 0
        (void)n;
        (void)k;

        const int actual_batch_size = (m > 0) ? m : batch_size_;

        WorkspaceRequirements reqs;

        // Buffer for K cache pointers: DataT* per sequence
        reqs.buffers.push_back({
            KVCacheWorkspaceBuffers::BATCH_K_PTRS,
            static_cast<size_t>(actual_batch_size) * sizeof(DataT *),
            256,  // Alignment
            false // Not required - fallback to per-call allocation
        });

        // Buffer for V cache pointers: DataT* per sequence
        reqs.buffers.push_back({KVCacheWorkspaceBuffers::BATCH_V_PTRS,
                                static_cast<size_t>(actual_batch_size) * sizeof(DataT *),
                                256,
                                false});

        // Buffer for tail indices: int per sequence
        reqs.buffers.push_back({KVCacheWorkspaceBuffers::BATCH_TAILS,
                                static_cast<size_t>(actual_batch_size) * sizeof(int),
                                256,
                                false});

        // Buffer for count values: int per sequence
        reqs.buffers.push_back({KVCacheWorkspaceBuffers::BATCH_COUNTS,
                                static_cast<size_t>(actual_batch_size) * sizeof(int),
                                256,
                                false});

        LOG_DEBUG("[CUDARingKVCache] Workspace requirements: batch_size="
                  << actual_batch_size
                  << " BATCH_K_PTRS=" << actual_batch_size * sizeof(DataT *)
                  << " BATCH_V_PTRS=" << actual_batch_size * sizeof(DataT *)
                  << " BATCH_TAILS=" << actual_batch_size * sizeof(int)
                  << " BATCH_COUNTS=" << actual_batch_size * sizeof(int));

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
