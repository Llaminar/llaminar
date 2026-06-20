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
#include <cstdint>

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
     * @brief Append verifier rows from position-major or head-major source.
     *
     * Qwen3.6 grouped verifier rows may keep K/V ordered as
     * `[head][row][dim]` after RoPE.  The ring cache is position-major, so this
     * kernel performs the small gather while preserving serial-decode write
     * positions.
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
     * This is launched after ring_append_kernel_dynamic on the same stream.
     * Keeping it separate prevents a write-after-read race on d_head while
     * append blocks are still using the pre-append ring position.
     */
    __global__ void hip_kv_sequence_state_advance_kernel(
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

    __global__ void hip_kv_sequence_state_advance_dynamic_kernel(
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
    __global__ void hip_kv_sequence_state_publish_kernel(
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

    // =========================================================================
    // convert_to_q8_1_kernel — wave-shuffle optimized
    //
    // Layout: one 32-thread workgroup per Q8_1 block. Each lane owns one
    // element, keeps it in a register, and uses cross-lane XOR butterflies
    // for absmax + sum reductions. This replaces the prior implementation,
    // which used 3 LDS arrays (s_absmax / s_q / s_sum), 2 LDS tree reductions,
    // and 10 __syncthreads() barriers — all avoidable when all participating
    // threads live in a single wavefront and the reduction width equals the
    // shuffle width.
    //
    // Benefits on gfx906/gfx90a:
    //   - 0 __syncthreads() (down from ~10)
    //   - 0 LDS usage (down from 3 × 32 × 4 B = 384 B)
    //   - 5 __shfl_xor butterflies replace each 5-level LDS tree reduction
    //   - Register-resident q value streams directly into coalesced qs[] store
    //   - Higher occupancy (no LDS pressure)
    // =========================================================================
    template <typename SrcT>
    __global__ void convert_to_q8_1_kernel(
        const SrcT *__restrict__ src,
        Q8_1Block *__restrict__ dst,
        int rows,
        int cols,
        int blocks_per_row)
    {
        constexpr int BS = Q8_1Block::BLOCK_SIZE;  // 32
        const int row = blockIdx.y;
        const int block_col = blockIdx.x;
        const int lane = threadIdx.x;

        if (row >= rows || block_col >= blocks_per_row || lane >= BS)
            return;

        const int col = block_col * BS + lane;
        const bool in_bounds = (col < cols);
        const float x = in_bounds ? to_float_device(src[row * cols + col]) : 0.0f;

        // -------- 1) Absmax via __shfl_xor butterfly (register-only) --------
        float absmax = fabsf(x);
        #pragma unroll
        for (int offset = BS / 2; offset > 0; offset >>= 1)
        {
            absmax = fmaxf(absmax, __shfl_xor(absmax, offset));
        }
        // After butterfly, every lane in the 32-thread group holds the full absmax.

        // -------- 2) Compute per-lane quantum --------
        const float d = (absmax > 0.0f) ? (absmax / 127.0f) : 0.0f;
        const float inv_d = (d > 0.0f) ? (127.0f / absmax) : 0.0f;
        int q = 0;
        if (d > 0.0f)
        {
            q = __float2int_rn(x * inv_d);
            q = max(-127, min(127, q));
        }

        // -------- 3) Sum via __shfl_xor butterfly (register-only) --------
        int sum_qs = q;
        #pragma unroll
        for (int offset = BS / 2; offset > 0; offset >>= 1)
        {
            sum_qs += __shfl_xor(sum_qs, offset);
        }
        // After butterfly, every lane holds the block sum.

        // -------- 4) Write qs[] in coalesced fashion + scalars from lane 0 --------
        Q8_1Block &out = dst[row * blocks_per_row + block_col];
        out.qs[lane] = static_cast<int8_t>(q);
        if (lane == 0)
        {
            out.d = __half_as_ushort(__float2half(d));
            out.sum_qs = static_cast<int16_t>(sum_qs);
        }
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

    // Q8_1 → FP16 dequantization kernel (anonymous namespace to avoid ODR conflict with CUDA)
    namespace
    {
        __global__ void hip_dequant_q8_1_to_fp16_kernel(
            const Q8_1Block *__restrict__ src,
            half *__restrict__ dst,
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
    } // anonymous namespace

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
            if (hipMemcpyAsync(d_dst, d_src, static_cast<size_t>(count) * sizeof(float),
                               hipMemcpyDeviceToDevice, stream) != hipSuccess)
            {
                return false;
            }
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
            if (hipMemcpyAsync(d_dst, d_src, static_cast<size_t>(count) * sizeof(uint16_t),
                               hipMemcpyDeviceToDevice, stream) != hipSuccess)
            {
                return false;
            }
            break;
        case TensorType::BF16:
            hipLaunchKernelGGL(convert_to_fp16_kernel<hip_bfloat16>, grid, block, 0, stream,
                               static_cast<const hip_bfloat16 *>(d_src),
                               reinterpret_cast<half *>(d_dst),
                               count);
            break;
        case TensorType::Q8_1:
            hipLaunchKernelGGL(hip_dequant_q8_1_to_fp16_kernel, grid, block, 0, stream,
                               static_cast<const Q8_1Block *>(d_src),
                               reinterpret_cast<half *>(d_dst),
                               count);
            break;
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
            if (hipMemcpyAsync(d_dst, d_src,
                               static_cast<size_t>(rows) * static_cast<size_t>(blocks_per_row) * sizeof(Q8_1Block),
                               hipMemcpyDeviceToDevice, stream) != hipSuccess)
            {
                return false;
            }
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

    extern "C" void hip_kv_sequence_state_advance(
        int *d_head, int *d_count, int num_tokens, int max_seq_len,
        hipStream_t stream)
    {
        if (!d_head || !d_count || num_tokens <= 0 || max_seq_len <= 0)
            return;

        hipLaunchKernelGGL(hip_kv_sequence_state_advance_kernel,
                           dim3(1), dim3(1), 0, stream,
                           d_head, d_count, num_tokens, max_seq_len);
    }

    extern "C" void hip_kv_sequence_state_advance_dynamic(
        int *d_head, int *d_count, const int *d_append_count,
        int captured_num_tokens, int max_seq_len,
        hipStream_t stream)
    {
        if (!d_head || !d_count || captured_num_tokens <= 0 || max_seq_len <= 0)
            return;

        hipLaunchKernelGGL(hip_kv_sequence_state_advance_dynamic_kernel,
                           dim3(1), dim3(1), 0, stream,
                           d_head, d_count, d_append_count,
                           captured_num_tokens, max_seq_len);
    }

    template <typename T>
    static void hip_ring_append_verifier_rows_typed(
        T *d_K_cache, T *d_V_cache,
        const T *d_K_new, const T *d_V_new,
        int head, int max_seq_len, int kv_storage_dim, int head_storage_dim,
        int source_verifier_rows, int source_start_row, int rows_to_write,
        bool k_source_head_major, bool v_source_head_major,
        hipStream_t stream)
    {
        if (rows_to_write <= 0 || kv_storage_dim <= 0)
            return;

        dim3 grid(rows_to_write, (kv_storage_dim + BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 block(BLOCK_SIZE);
        hipLaunchKernelGGL(ring_append_verifier_rows_kernel<T>,
                           grid, block, 0, stream,
                           d_K_cache, d_V_cache, d_K_new, d_V_new,
                           head, max_seq_len, kv_storage_dim, head_storage_dim,
                           source_verifier_rows, source_start_row, rows_to_write,
                           k_source_head_major, v_source_head_major);
    }

    template <typename T>
    static void hip_ring_append_verifier_rows_dynamic_typed(
        T *d_K_cache, T *d_V_cache,
        const T *d_K_new, const T *d_V_new,
        const int *d_head, int max_seq_len, int kv_storage_dim, int head_storage_dim,
        int source_verifier_rows, int source_start_row, int rows_to_write,
        bool k_source_head_major, bool v_source_head_major,
        hipStream_t stream)
    {
        if (rows_to_write <= 0 || kv_storage_dim <= 0)
            return;

        dim3 grid(rows_to_write, (kv_storage_dim + BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 block(BLOCK_SIZE);
        hipLaunchKernelGGL(ring_append_verifier_rows_dynamic_kernel<T>,
                           grid, block, 0, stream,
                           d_K_cache, d_V_cache, d_K_new, d_V_new,
                           d_head, max_seq_len, kv_storage_dim, head_storage_dim,
                           source_verifier_rows, source_start_row, rows_to_write,
                           k_source_head_major, v_source_head_major);
    }

    extern "C" void hip_ring_append_verifier_rows_fp32(
        float *d_K_cache, float *d_V_cache,
        const float *d_K_new, const float *d_V_new,
        int head, int max_seq_len, int kv_storage_dim, int head_storage_dim,
        int source_verifier_rows, int source_start_row, int rows_to_write,
        bool k_source_head_major, bool v_source_head_major,
        hipStream_t stream)
    {
        hip_ring_append_verifier_rows_typed<float>(
            d_K_cache, d_V_cache, d_K_new, d_V_new,
            head, max_seq_len, kv_storage_dim, head_storage_dim,
            source_verifier_rows, source_start_row, rows_to_write,
            k_source_head_major, v_source_head_major, stream);
    }

    extern "C" void hip_ring_append_verifier_rows_fp16(
        _Float16 *d_K_cache, _Float16 *d_V_cache,
        const _Float16 *d_K_new, const _Float16 *d_V_new,
        int head, int max_seq_len, int kv_storage_dim, int head_storage_dim,
        int source_verifier_rows, int source_start_row, int rows_to_write,
        bool k_source_head_major, bool v_source_head_major,
        hipStream_t stream)
    {
        hip_ring_append_verifier_rows_typed<_Float16>(
            d_K_cache, d_V_cache, d_K_new, d_V_new,
            head, max_seq_len, kv_storage_dim, head_storage_dim,
            source_verifier_rows, source_start_row, rows_to_write,
            k_source_head_major, v_source_head_major, stream);
    }

    extern "C" void hip_ring_append_verifier_rows_bf16(
        hip_bfloat16 *d_K_cache, hip_bfloat16 *d_V_cache,
        const hip_bfloat16 *d_K_new, const hip_bfloat16 *d_V_new,
        int head, int max_seq_len, int kv_storage_dim, int head_storage_dim,
        int source_verifier_rows, int source_start_row, int rows_to_write,
        bool k_source_head_major, bool v_source_head_major,
        hipStream_t stream)
    {
        hip_ring_append_verifier_rows_typed<hip_bfloat16>(
            d_K_cache, d_V_cache, d_K_new, d_V_new,
            head, max_seq_len, kv_storage_dim, head_storage_dim,
            source_verifier_rows, source_start_row, rows_to_write,
            k_source_head_major, v_source_head_major, stream);
    }

    extern "C" void hip_ring_append_verifier_rows_q8_1(
        Q8_1Block *d_K_cache, Q8_1Block *d_V_cache,
        const Q8_1Block *d_K_new, const Q8_1Block *d_V_new,
        int head, int max_seq_len, int kv_storage_dim, int head_storage_dim,
        int source_verifier_rows, int source_start_row, int rows_to_write,
        bool k_source_head_major, bool v_source_head_major,
        hipStream_t stream)
    {
        hip_ring_append_verifier_rows_typed<Q8_1Block>(
            d_K_cache, d_V_cache, d_K_new, d_V_new,
            head, max_seq_len, kv_storage_dim, head_storage_dim,
            source_verifier_rows, source_start_row, rows_to_write,
            k_source_head_major, v_source_head_major, stream);
    }

    extern "C" void hip_ring_append_verifier_rows_dynamic_fp32(
        float *d_K_cache, float *d_V_cache,
        const float *d_K_new, const float *d_V_new,
        const int *d_head, int max_seq_len, int kv_storage_dim, int head_storage_dim,
        int source_verifier_rows, int source_start_row, int rows_to_write,
        bool k_source_head_major, bool v_source_head_major,
        hipStream_t stream)
    {
        hip_ring_append_verifier_rows_dynamic_typed<float>(
            d_K_cache, d_V_cache, d_K_new, d_V_new,
            d_head, max_seq_len, kv_storage_dim, head_storage_dim,
            source_verifier_rows, source_start_row, rows_to_write,
            k_source_head_major, v_source_head_major, stream);
    }

    extern "C" void hip_ring_append_verifier_rows_dynamic_fp16(
        _Float16 *d_K_cache, _Float16 *d_V_cache,
        const _Float16 *d_K_new, const _Float16 *d_V_new,
        const int *d_head, int max_seq_len, int kv_storage_dim, int head_storage_dim,
        int source_verifier_rows, int source_start_row, int rows_to_write,
        bool k_source_head_major, bool v_source_head_major,
        hipStream_t stream)
    {
        hip_ring_append_verifier_rows_dynamic_typed<_Float16>(
            d_K_cache, d_V_cache, d_K_new, d_V_new,
            d_head, max_seq_len, kv_storage_dim, head_storage_dim,
            source_verifier_rows, source_start_row, rows_to_write,
            k_source_head_major, v_source_head_major, stream);
    }

    extern "C" void hip_ring_append_verifier_rows_dynamic_bf16(
        hip_bfloat16 *d_K_cache, hip_bfloat16 *d_V_cache,
        const hip_bfloat16 *d_K_new, const hip_bfloat16 *d_V_new,
        const int *d_head, int max_seq_len, int kv_storage_dim, int head_storage_dim,
        int source_verifier_rows, int source_start_row, int rows_to_write,
        bool k_source_head_major, bool v_source_head_major,
        hipStream_t stream)
    {
        hip_ring_append_verifier_rows_dynamic_typed<hip_bfloat16>(
            d_K_cache, d_V_cache, d_K_new, d_V_new,
            d_head, max_seq_len, kv_storage_dim, head_storage_dim,
            source_verifier_rows, source_start_row, rows_to_write,
            k_source_head_major, v_source_head_major, stream);
    }

    extern "C" void hip_ring_append_verifier_rows_dynamic_q8_1(
        Q8_1Block *d_K_cache, Q8_1Block *d_V_cache,
        const Q8_1Block *d_K_new, const Q8_1Block *d_V_new,
        const int *d_head, int max_seq_len, int kv_storage_dim, int head_storage_dim,
        int source_verifier_rows, int source_start_row, int rows_to_write,
        bool k_source_head_major, bool v_source_head_major,
        hipStream_t stream)
    {
        hip_ring_append_verifier_rows_dynamic_typed<Q8_1Block>(
            d_K_cache, d_V_cache, d_K_new, d_V_new,
            d_head, max_seq_len, kv_storage_dim, head_storage_dim,
            source_verifier_rows, source_start_row, rows_to_write,
            k_source_head_major, v_source_head_major, stream);
    }

    extern "C" bool hip_kv_sequence_state_publish(
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
        hipStream_t stream)
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

        hipLaunchKernelGGL(hip_kv_sequence_state_publish_kernel,
                           dim3(request_count, n_layers), dim3(1), 0, stream,
                           d_heads,
                           d_counts,
                           target_cached_tokens,
                           accepted_state_counts,
                           publication_ok_flags,
                           batch_size,
                           first_seq_idx,
                           request_count,
                           max_seq_len);
        return hipGetLastError() == hipSuccess;
    }

} // namespace llaminar2
