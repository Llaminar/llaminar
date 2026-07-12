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
 * 1. ring_append_kernel_dynamic - Append from canonical device state
 * 2. ring_linearize_kernel - Copy wrapped data to contiguous buffer
 * 3. device-state gather kernels - Gather request rings for batched attention
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include "ROCmTurboQuantKernels.h"
#include "../../../tensors/BlockStructures.h"
#include "../../../tensors/TensorType.h"
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <type_traits>

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
     * @brief Resolve the real append row count for a captured bucket launch.
     *
     * Captured prefill graphs keep bucket-shaped launch geometry so one graph
     * can serve several shorter prompt suffixes. The graph-owned request-length
     * row supplies the live count directly; copy kernels honor that value so
     * padded rows never mutate KV cache storage. Invalid metadata is treated as
     * zero rows because admission validation owns the fail-fast diagnostic.
     */
    __device__ __forceinline__ int effective_append_tokens(
        const int *__restrict__ d_append_count,
        int captured_num_tokens)
    {
        if (!d_append_count)
            return captured_num_tokens;
        const int real_tokens = *d_append_count;
        return (real_tokens > 0 && real_tokens <= captured_num_tokens)
                   ? real_tokens
                   : 0;
    }

    /**
     * @brief Append tokens to ring buffer (graph-capturable variant)
     *
     * Reads head position from a device pointer instead of a scalar argument.
     * This allows one HIP graph to replay against the canonical head advanced
     * by prior append/publication nodes, with no host adoption or H2D update.
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
        const int *__restrict__ d_append_count, // Real rows for padded bucket replay
        int max_seq_len,               // Ring buffer capacity
        int kv_dim,                    // n_kv_heads * head_dim
        int num_tokens)                // Tokens to append
    {
        const int token_idx = blockIdx.x;
        const int elem_idx = blockIdx.y * blockDim.x + threadIdx.x;

        const int rows_to_write = effective_append_tokens(d_append_count, num_tokens);
        const int source_start = rows_to_write > max_seq_len
                                     ? rows_to_write - max_seq_len
                                     : 0;
        if (token_idx < source_start || token_idx >= rows_to_write ||
            elem_idx >= kv_dim)
            return;

        const int head = *d_head; // Read from device memory
        const int dst_pos = (head + token_idx) % max_seq_len;
        const int dst_offset = dst_pos * kv_dim + elem_idx;
        const int src_offset = token_idx * kv_dim + elem_idx;

        d_K_cache[dst_offset] = d_K_new[src_offset];
        d_V_cache[dst_offset] = d_V_new[src_offset];
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
        const int dst_pos = (head + source_row) % max_seq_len;
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

        const int advance_tokens =
            effective_append_tokens(d_append_count, captured_num_tokens);
        if (advance_tokens <= 0)
            return;

        const int old_head = *d_head;
        const int old_count = *d_count;
        *d_head = (old_head + advance_tokens) % max_seq_len;
        const int next_count = old_count + advance_tokens;
        *d_count = next_count > max_seq_len ? max_seq_len : next_count;
    }

    /** Stream-ordered replacement of canonical device sequence metadata. */
    __global__ void hip_kv_sequence_state_set_kernel(
        int *__restrict__ d_head,
        int *__restrict__ d_count,
        int head,
        int count)
    {
        if (threadIdx.x == 0 && blockIdx.x == 0)
        {
            *d_head = head;
            *d_count = count;
        }
    }

    /**
     * @brief Publish accepted verifier-row sequence metadata on device.
     *
     * Each block owns one [request, layer] pair.  The verifier graph may have
     * already advanced the live device head past rows that were later rejected,
     * so publication must keep the current ring tail and clamp the visible
     * window to the accepted target length.  This mirrors truncateSequence()
     * and keeps accepted-state rollback ordered with subsequent device reads.
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
     * @brief Gather independent rings using only persistent device state.
     *
     * Captured append and accepted-state publication kernels update `d_heads`
     * and `d_counts` in graph order.  Reading those same scalars here avoids a
     * host metadata round trip between KV publication and grouped attention.
     * Immutable entry pointers are indexed as `[compressed_layer, request]`.
     *
     * @tparam T Native cache storage element or quantization-block type.
     */
    template <typename T, bool ZeroInactiveRows>
    __global__ void ring_gather_batched_device_state_kernel(
        T *__restrict__ d_K_out,
        T *__restrict__ d_V_out,
        const T *const *__restrict__ d_K_entry_table,
        const T *const *__restrict__ d_V_entry_table,
        const int *__restrict__ d_heads,
        const int *__restrict__ d_counts,
        int entry_offset,
        int request_count,
        int output_stride,
        int max_seq_len,
        int kv_dim)
    {
        const int request = static_cast<int>(blockIdx.z);
        const int element =
            static_cast<int>(blockIdx.y * blockDim.x + threadIdx.x);
        if (request >= request_count || element >= kv_dim)
            return;

        const int entry = entry_offset + request;
        int ring_count = d_counts[entry];
        ring_count = ring_count < 0
                         ? 0
                         : (ring_count > max_seq_len ? max_seq_len : ring_count);
        const int visible_count =
            ring_count < output_stride ? ring_count : output_stride;
        const int skipped_rows = ring_count - visible_count;
        int tail = d_heads[entry] - ring_count;
        tail %= max_seq_len;
        if (tail < 0)
            tail += max_seq_len;
        const int token_limit =
            ZeroInactiveRows ? output_stride : visible_count;

        for (int output_token = static_cast<int>(blockIdx.x);
             output_token < token_limit;
             output_token += static_cast<int>(gridDim.x))
        {
            const int output_offset =
                (request * output_stride + output_token) * kv_dim + element;
            if (output_token >= visible_count)
            {
                d_K_out[output_offset] = T{};
                d_V_out[output_offset] = T{};
                continue;
            }

            const int source_token =
                (tail + skipped_rows + output_token) % max_seq_len;
            const int source_offset = source_token * kv_dim + element;
            d_K_out[output_offset] = d_K_entry_table[entry][source_offset];
            d_V_out[output_offset] = d_V_entry_table[entry][source_offset];
        }
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

    template <typename StorageT>
    __device__ __forceinline__ float load_ring_logical_value(
        const StorageT *cache,
        int source_token,
        int logical_element,
        int logical_kv_dim)
    {
        return to_float_device(cache[
            static_cast<size_t>(source_token) * logical_kv_dim +
            logical_element]);
    }

    template <>
    __device__ __forceinline__ float load_ring_logical_value<Q8_1Block>(
        const Q8_1Block *cache,
        int source_token,
        int logical_element,
        int logical_kv_dim)
    {
        const int blocks_per_row =
            logical_kv_dim / static_cast<int>(Q8_1Block::BLOCK_SIZE);
        const Q8_1Block &block = cache[
            static_cast<size_t>(source_token) * blocks_per_row +
            logical_element / static_cast<int>(Q8_1Block::BLOCK_SIZE)];
        const float scale = __half2float(__ushort_as_half(block.d));
        return scale * static_cast<float>(
                           block.qs[logical_element %
                                    static_cast<int>(Q8_1Block::BLOCK_SIZE)]);
    }

    /**
     * @brief Gather canonical device rings directly into grouped FP16 output.
     *
     * This is the ROCm counterpart of the CUDA device-owned conversion path.
     * It reads immutable entry pointers plus canonical device head/count values,
     * retains the newest bounded window, and zero-fills inactive output rows.
     */
    template <typename StorageT, bool ZeroInactiveRows>
    __global__ void ring_gather_batched_device_state_fp16_kernel(
        _Float16 *__restrict__ k_output,
        _Float16 *__restrict__ v_output,
        const StorageT *const *__restrict__ k_entry_table,
        const StorageT *const *__restrict__ v_entry_table,
        const int *__restrict__ heads,
        const int *__restrict__ counts,
        int entry_offset,
        int request_count,
        int output_stride,
        int max_seq_len,
        int logical_kv_dim)
    {
        const int request = static_cast<int>(blockIdx.z);
        const int element =
            static_cast<int>(blockIdx.y * blockDim.x + threadIdx.x);
        if (request >= request_count || element >= logical_kv_dim)
            return;

        const int entry = entry_offset + request;
        int ring_count = counts[entry];
        ring_count = ring_count < 0
                         ? 0
                         : (ring_count > max_seq_len ? max_seq_len : ring_count);
        const int visible_count =
            ring_count < output_stride ? ring_count : output_stride;
        const int skipped_rows = ring_count - visible_count;
        int tail = heads[entry] - ring_count;
        tail %= max_seq_len;
        if (tail < 0)
            tail += max_seq_len;
        const int token_limit =
            ZeroInactiveRows ? output_stride : visible_count;
        for (int output_token = static_cast<int>(blockIdx.x);
             output_token < token_limit;
             output_token += static_cast<int>(gridDim.x))
        {
            const size_t output_index =
                (static_cast<size_t>(request) * output_stride + output_token) *
                    logical_kv_dim +
                element;
            if (output_token >= visible_count)
            {
                k_output[output_index] = static_cast<_Float16>(0.0f);
                v_output[output_index] = static_cast<_Float16>(0.0f);
                continue;
            }

            const int source_token =
                (tail + skipped_rows + output_token) % max_seq_len;
            k_output[output_index] = static_cast<_Float16>(
                load_ring_logical_value(
                    k_entry_table[entry], source_token, element,
                    logical_kv_dim));
            v_output[output_index] = static_cast<_Float16>(
                load_ring_logical_value(
                    v_entry_table[entry], source_token, element,
                    logical_kv_dim));
        }
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
    __global__ void ring_append_convert_to_fp16_kernel(
        _Float16 *__restrict__ d_K_cache,
        _Float16 *__restrict__ d_V_cache,
        const SrcT *__restrict__ d_K_new,
        const SrcT *__restrict__ d_V_new,
        const int *__restrict__ d_head,
        const int *__restrict__ d_append_count,
        int max_seq_len,
        int kv_dim,
        int num_tokens)
    {
        const int token_idx = blockIdx.x;
        const int elem_idx = blockIdx.y * blockDim.x + threadIdx.x;
        const int rows_to_write = effective_append_tokens(d_append_count, num_tokens);
        const int source_start = rows_to_write > max_seq_len
                                     ? rows_to_write - max_seq_len
                                     : 0;
        if (token_idx < source_start || token_idx >= rows_to_write ||
            elem_idx >= kv_dim)
            return;

        const int dst_pos = (*d_head + token_idx) % max_seq_len;
        const int dst_offset = dst_pos * kv_dim + elem_idx;
        const int src_offset = token_idx * kv_dim + elem_idx;

        d_K_cache[dst_offset] = static_cast<_Float16>(to_float_device(d_K_new[src_offset]));
        d_V_cache[dst_offset] = static_cast<_Float16>(to_float_device(d_V_new[src_offset]));
    }

    __global__ void ring_append_q8_1_to_fp16_kernel(
        _Float16 *__restrict__ d_K_cache,
        _Float16 *__restrict__ d_V_cache,
        const Q8_1Block *__restrict__ d_K_new,
        const Q8_1Block *__restrict__ d_V_new,
        const int *__restrict__ d_head,
        const int *__restrict__ d_append_count,
        int max_seq_len,
        int kv_dim,
        int num_tokens,
        int src_blocks_per_row)
    {
        const int token_idx = blockIdx.x;
        const int elem_idx = blockIdx.y * blockDim.x + threadIdx.x;
        const int rows_to_write = effective_append_tokens(d_append_count, num_tokens);
        const int source_start = rows_to_write > max_seq_len
                                     ? rows_to_write - max_seq_len
                                     : 0;
        if (token_idx < source_start || token_idx >= rows_to_write ||
            elem_idx >= kv_dim)
            return;

        const int dst_pos = (*d_head + token_idx) % max_seq_len;
        const int dst_offset = dst_pos * kv_dim + elem_idx;
        const int block_col = elem_idx / Q8_1Block::BLOCK_SIZE;
        const int lane = elem_idx % Q8_1Block::BLOCK_SIZE;
        const int src_offset = token_idx * src_blocks_per_row + block_col;

        const Q8_1Block &k_block = d_K_new[src_offset];
        const Q8_1Block &v_block = d_V_new[src_offset];
        const float k_scale = __half2float(__ushort_as_half(k_block.d));
        const float v_scale = __half2float(__ushort_as_half(v_block.d));
        d_K_cache[dst_offset] = static_cast<_Float16>(k_scale * static_cast<float>(k_block.qs[lane]));
        d_V_cache[dst_offset] = static_cast<_Float16>(v_scale * static_cast<float>(v_block.qs[lane]));
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

    extern "C" bool hip_ring_append_converted_fp16(
        _Float16 *d_K_cache, _Float16 *d_V_cache,
        const void *d_K_new, const void *d_V_new,
        TensorType src_type,
        const int *d_head,
        const int *d_append_count,
        int max_seq_len,
        int kv_dim,
        int num_tokens,
        hipStream_t stream)
    {
        if (!d_K_cache || !d_V_cache || !d_K_new || !d_V_new || !d_head ||
            max_seq_len <= 0 || kv_dim <= 0 || num_tokens <= 0)
        {
            return false;
        }

        dim3 block(BLOCK_SIZE);
        dim3 grid(num_tokens, (kv_dim + BLOCK_SIZE - 1) / BLOCK_SIZE);
        switch (src_type)
        {
        case TensorType::FP32:
            hipLaunchKernelGGL(HIP_KERNEL_NAME(ring_append_convert_to_fp16_kernel<float>),
                               grid, block, 0, stream,
                               d_K_cache, d_V_cache,
                               static_cast<const float *>(d_K_new),
                               static_cast<const float *>(d_V_new),
                               d_head, d_append_count, max_seq_len, kv_dim, num_tokens);
            break;
        case TensorType::FP16:
            hipLaunchKernelGGL(ring_append_kernel_dynamic<_Float16>,
                               grid, block, 0, stream,
                               d_K_cache, d_V_cache,
                               static_cast<const _Float16 *>(d_K_new),
                               static_cast<const _Float16 *>(d_V_new),
                               d_head, d_append_count, max_seq_len, kv_dim, num_tokens);
            break;
        case TensorType::BF16:
            hipLaunchKernelGGL(HIP_KERNEL_NAME(ring_append_convert_to_fp16_kernel<hip_bfloat16>),
                               grid, block, 0, stream,
                               d_K_cache, d_V_cache,
                               static_cast<const hip_bfloat16 *>(d_K_new),
                               static_cast<const hip_bfloat16 *>(d_V_new),
                               d_head, d_append_count, max_seq_len, kv_dim, num_tokens);
            break;
        case TensorType::Q8_1:
        {
            const int blocks_per_row = (kv_dim + Q8_1Block::BLOCK_SIZE - 1) / Q8_1Block::BLOCK_SIZE;
            hipLaunchKernelGGL(ring_append_q8_1_to_fp16_kernel,
                               grid, block, 0, stream,
                               d_K_cache, d_V_cache,
                               static_cast<const Q8_1Block *>(d_K_new),
                               static_cast<const Q8_1Block *>(d_V_new),
                               d_head, d_append_count, max_seq_len, kv_dim, num_tokens, blocks_per_row);
            break;
        }
        default:
            return false;
        }

        return hipGetLastError() == hipSuccess;
    }

    // =========================================================================
    // Kernel Launch Wrappers (extern "C" for linking)
    // =========================================================================

    /**
     * @brief Launch the native device-state gather for one storage type.
     *
     * Returning only the launch status keeps the operation asynchronous and
     * graph-capturable.  Stream completion remains owned by the graph executor.
     */
    template <typename T>
    static hipError_t launch_ring_gather_batched_device_state(
        T *d_K_out,
        T *d_V_out,
        const T *const *d_K_entry_table,
        const T *const *d_V_entry_table,
        const int *d_heads,
        const int *d_counts,
        int entry_offset,
        int request_count,
        int max_kv_len,
        int max_seq_len,
        int kv_dim,
        bool zero_inactive_rows,
        hipStream_t stream)
    {
        if (!d_K_out || !d_V_out || !d_K_entry_table || !d_V_entry_table ||
            !d_heads || !d_counts || !stream || entry_offset < 0 ||
            request_count <= 0 || max_kv_len <= 0 || max_seq_len <= 0 ||
            kv_dim <= 0)
        {
            return hipErrorInvalidValue;
        }

        constexpr unsigned int resident_token_blocks = 32;
        const dim3 grid(
            std::min(
                static_cast<unsigned int>(max_kv_len),
                resident_token_blocks),
            static_cast<unsigned int>((kv_dim + BLOCK_SIZE - 1) / BLOCK_SIZE),
            static_cast<unsigned int>(request_count));
        const dim3 block(BLOCK_SIZE);
        if (zero_inactive_rows)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(
                    ring_gather_batched_device_state_kernel<T, true>),
                grid, block, 0, stream,
                d_K_out, d_V_out,
                d_K_entry_table, d_V_entry_table,
                d_heads, d_counts,
                entry_offset, request_count, max_kv_len,
                max_seq_len, kv_dim);
        }
        else
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(
                    ring_gather_batched_device_state_kernel<T, false>),
                grid, block, 0, stream,
                d_K_out, d_V_out,
                d_K_entry_table, d_V_entry_table,
                d_heads, d_counts,
                entry_offset, request_count, max_kv_len,
                max_seq_len, kv_dim);
        }
        return hipGetLastError();
    }

    /**
     * @brief Launch grouped ring gather, native conversion, and optional RoPE.
     *
     * FP32 storage overwrites rotated dimensions directly from the source ring
     * so RoPE remains FP32-before-FP16-rounding. Other formats intentionally
     * rotate the gathered FP16 values, matching their serial conversion order.
     */
    template <typename T>
    static hipError_t launch_ring_gather_batched_converted_device_state(
        _Float16 *d_K_out,
        _Float16 *d_V_out,
        const T *const *d_K_entry_table,
        const T *const *d_V_entry_table,
        const int *d_heads,
        const int *d_counts,
        int entry_offset,
        int request_count,
        int max_kv_len,
        int max_seq_len,
        int logical_kv_dim,
        int n_kv_heads,
        int head_dim,
        float rope_theta,
        int position_start,
        int rope_dim,
        hipStream_t stream)
    {
        const int effective_rope_dim = rope_dim > 0 ? rope_dim : head_dim;
        if (!d_K_out || !d_V_out || !d_K_entry_table || !d_V_entry_table ||
            !d_heads || !d_counts || !stream || entry_offset < 0 ||
            request_count <= 0 || max_kv_len <= 0 || max_seq_len <= 0 ||
            logical_kv_dim <= 0 || n_kv_heads <= 0 || head_dim <= 0 ||
            n_kv_heads * head_dim != logical_kv_dim ||
            effective_rope_dim <= 0 || effective_rope_dim > head_dim ||
            (effective_rope_dim % 2) != 0)
        {
            return hipErrorInvalidValue;
        }

        constexpr unsigned int resident_token_blocks = 32;
        const dim3 gather_grid(
            std::min(
                static_cast<unsigned int>(max_kv_len),
                resident_token_blocks),
            static_cast<unsigned int>((logical_kv_dim + BLOCK_SIZE - 1) /
                                      BLOCK_SIZE),
            static_cast<unsigned int>(request_count));
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(
                ring_gather_batched_device_state_fp16_kernel<T, true>),
            gather_grid,
            dim3(BLOCK_SIZE),
            0,
            stream,
            d_K_out,
            d_V_out,
            d_K_entry_table,
            d_V_entry_table,
            d_heads,
            d_counts,
            entry_offset,
            request_count,
            max_kv_len,
            max_seq_len,
            logical_kv_dim);
        hipError_t status = hipGetLastError();
        if (status != hipSuccess || rope_theta <= 0.0f)
            return status;

        bool rope_ok = false;
        if constexpr (std::is_same_v<T, float>)
        {
            rope_ok = hip_rope_apply_batched_fp32_ring_to_fp16_device_state(
                d_K_out, d_K_entry_table, d_heads, d_counts, entry_offset,
                request_count, max_kv_len, max_seq_len, n_kv_heads, head_dim,
                rope_theta, position_start, effective_rope_dim, stream);
        }
        else
        {
            rope_ok = hip_rope_apply_batched_fp16_device_state(
                d_K_out, d_counts, entry_offset, request_count, max_kv_len,
                max_seq_len, n_kv_heads, head_dim, rope_theta,
                position_start, effective_rope_dim, stream);
        }
        if (rope_ok)
            return hipSuccess;
        const hipError_t rope_error = hipGetLastError();
        return rope_error == hipSuccess ? hipErrorInvalidValue : rope_error;
    }

    // ------ FP32 scalar diagnostic linearization ------
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

    // ------ FP16 scalar diagnostic linearization ------
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

    // ------ BF16 scalar diagnostic linearization ------
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

    // ------ Q8_1 scalar diagnostic linearization ------
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

    // ------ Device-resident metadata variants ------

    extern "C" hipError_t hip_ring_gather_batched_device_state_fp32(
        float *d_K_out, float *d_V_out,
        const float *const *d_K_entry_table,
        const float *const *d_V_entry_table,
        const int *d_heads, const int *d_counts,
        int entry_offset, int request_count, int max_kv_len,
        int max_seq_len, int kv_dim, bool zero_inactive_rows,
        hipStream_t stream)
    {
        return launch_ring_gather_batched_device_state(
            d_K_out, d_V_out, d_K_entry_table, d_V_entry_table,
            d_heads, d_counts, entry_offset, request_count, max_kv_len,
            max_seq_len, kv_dim, zero_inactive_rows, stream);
    }

    extern "C" hipError_t hip_ring_gather_batched_device_state_fp16(
        _Float16 *d_K_out, _Float16 *d_V_out,
        const _Float16 *const *d_K_entry_table,
        const _Float16 *const *d_V_entry_table,
        const int *d_heads, const int *d_counts,
        int entry_offset, int request_count, int max_kv_len,
        int max_seq_len, int kv_dim, bool zero_inactive_rows,
        hipStream_t stream)
    {
        return launch_ring_gather_batched_device_state(
            d_K_out, d_V_out, d_K_entry_table, d_V_entry_table,
            d_heads, d_counts, entry_offset, request_count, max_kv_len,
            max_seq_len, kv_dim, zero_inactive_rows, stream);
    }

    extern "C" hipError_t hip_ring_gather_batched_device_state_bf16(
        hip_bfloat16 *d_K_out, hip_bfloat16 *d_V_out,
        const hip_bfloat16 *const *d_K_entry_table,
        const hip_bfloat16 *const *d_V_entry_table,
        const int *d_heads, const int *d_counts,
        int entry_offset, int request_count, int max_kv_len,
        int max_seq_len, int kv_dim, bool zero_inactive_rows,
        hipStream_t stream)
    {
        return launch_ring_gather_batched_device_state(
            d_K_out, d_V_out, d_K_entry_table, d_V_entry_table,
            d_heads, d_counts, entry_offset, request_count, max_kv_len,
            max_seq_len, kv_dim, zero_inactive_rows, stream);
    }

    extern "C" hipError_t hip_ring_gather_batched_device_state_q8_1(
        Q8_1Block *d_K_out, Q8_1Block *d_V_out,
        const Q8_1Block *const *d_K_entry_table,
        const Q8_1Block *const *d_V_entry_table,
        const int *d_heads, const int *d_counts,
        int entry_offset, int request_count, int max_kv_len,
        int max_seq_len, int kv_blocks, bool zero_inactive_rows,
        hipStream_t stream)
    {
        return launch_ring_gather_batched_device_state(
            d_K_out, d_V_out, d_K_entry_table, d_V_entry_table,
            d_heads, d_counts, entry_offset, request_count, max_kv_len,
            max_seq_len, kv_blocks, zero_inactive_rows, stream);
    }

    extern "C" hipError_t hip_ring_gather_batched_converted_device_state_fp32(
        _Float16 *d_K_out, _Float16 *d_V_out,
        const float *const *d_K_entry_table,
        const float *const *d_V_entry_table,
        const int *d_heads, const int *d_counts,
        int entry_offset, int request_count, int max_kv_len,
        int max_seq_len, int logical_kv_dim,
        int n_kv_heads, int head_dim,
        float rope_theta, int position_start, int rope_dim,
        hipStream_t stream)
    {
        return launch_ring_gather_batched_converted_device_state(
            d_K_out, d_V_out, d_K_entry_table, d_V_entry_table,
            d_heads, d_counts, entry_offset, request_count, max_kv_len,
            max_seq_len, logical_kv_dim, n_kv_heads, head_dim,
            rope_theta, position_start, rope_dim, stream);
    }

    extern "C" hipError_t hip_ring_gather_batched_converted_device_state_fp16(
        _Float16 *d_K_out, _Float16 *d_V_out,
        const _Float16 *const *d_K_entry_table,
        const _Float16 *const *d_V_entry_table,
        const int *d_heads, const int *d_counts,
        int entry_offset, int request_count, int max_kv_len,
        int max_seq_len, int logical_kv_dim,
        int n_kv_heads, int head_dim,
        float rope_theta, int position_start, int rope_dim,
        hipStream_t stream)
    {
        return launch_ring_gather_batched_converted_device_state(
            d_K_out, d_V_out, d_K_entry_table, d_V_entry_table,
            d_heads, d_counts, entry_offset, request_count, max_kv_len,
            max_seq_len, logical_kv_dim, n_kv_heads, head_dim,
            rope_theta, position_start, rope_dim, stream);
    }

    extern "C" hipError_t hip_ring_gather_batched_converted_device_state_bf16(
        _Float16 *d_K_out, _Float16 *d_V_out,
        const hip_bfloat16 *const *d_K_entry_table,
        const hip_bfloat16 *const *d_V_entry_table,
        const int *d_heads, const int *d_counts,
        int entry_offset, int request_count, int max_kv_len,
        int max_seq_len, int logical_kv_dim,
        int n_kv_heads, int head_dim,
        float rope_theta, int position_start, int rope_dim,
        hipStream_t stream)
    {
        return launch_ring_gather_batched_converted_device_state(
            d_K_out, d_V_out, d_K_entry_table, d_V_entry_table,
            d_heads, d_counts, entry_offset, request_count, max_kv_len,
            max_seq_len, logical_kv_dim, n_kv_heads, head_dim,
            rope_theta, position_start, rope_dim, stream);
    }

    extern "C" hipError_t hip_ring_gather_batched_converted_device_state_q8_1(
        _Float16 *d_K_out, _Float16 *d_V_out,
        const Q8_1Block *const *d_K_entry_table,
        const Q8_1Block *const *d_V_entry_table,
        const int *d_heads, const int *d_counts,
        int entry_offset, int request_count, int max_kv_len,
        int max_seq_len, int logical_kv_dim,
        int n_kv_heads, int head_dim,
        float rope_theta, int position_start, int rope_dim,
        hipStream_t stream)
    {
        return launch_ring_gather_batched_converted_device_state(
            d_K_out, d_V_out, d_K_entry_table, d_V_entry_table,
            d_heads, d_counts, entry_offset, request_count, max_kv_len,
            max_seq_len, logical_kv_dim, n_kv_heads, head_dim,
            rope_theta, position_start, rope_dim, stream);
    }

    // =========================================================================
    // Dynamic Head Append Wrappers (graph-capturable)
    // =========================================================================

    extern "C" void hip_ring_append_dynamic_fp32(
        float *d_K_cache, float *d_V_cache,
        const float *d_K_new, const float *d_V_new,
        const int *d_head, const int *d_append_count, int max_seq_len, int kv_dim, int num_tokens,
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
                           d_head, d_append_count, max_seq_len, kv_dim, num_tokens);
    }

    extern "C" void hip_ring_append_dynamic_fp16(
        _Float16 *d_K_cache, _Float16 *d_V_cache,
        const _Float16 *d_K_new, const _Float16 *d_V_new,
        const int *d_head, const int *d_append_count, int max_seq_len, int kv_dim, int num_tokens,
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
                           d_head, d_append_count, max_seq_len, kv_dim, num_tokens);
    }

    extern "C" void hip_ring_append_dynamic_bf16(
        hip_bfloat16 *d_K_cache, hip_bfloat16 *d_V_cache,
        const hip_bfloat16 *d_K_new, const hip_bfloat16 *d_V_new,
        const int *d_head, const int *d_append_count, int max_seq_len, int kv_dim, int num_tokens,
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
                           d_head, d_append_count, max_seq_len, kv_dim, num_tokens);
    }

    extern "C" void hip_ring_append_dynamic_q8_1(
        Q8_1Block *d_K_cache, Q8_1Block *d_V_cache,
        const Q8_1Block *d_K_new, const Q8_1Block *d_V_new,
        const int *d_head, const int *d_append_count, int max_seq_len, int kv_blocks, int num_tokens,
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
                           d_head, d_append_count, max_seq_len, kv_blocks, num_tokens);
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

    extern "C" bool hip_kv_sequence_state_set(
        int *d_head,
        int *d_count,
        int head,
        int count,
        int max_seq_len,
        hipStream_t stream)
    {
        if (!d_head || !d_count || !stream ||
            head < 0 || head >= max_seq_len ||
            count < 0 || count > max_seq_len)
        {
            return false;
        }
        hipLaunchKernelGGL(
            hip_kv_sequence_state_set_kernel,
            dim3(1),
            dim3(1),
            0,
            stream,
            d_head,
            d_count,
            head,
            count);
        return hipGetLastError() == hipSuccess;
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
