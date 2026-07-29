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
 * 3. device-state batched gather kernels - Gather multiple request rings
 */

#include "CUDARingKVCache.h"
#include "CUDATurboQuantKernels.h"
#include "../../../execution/local_execution/graph/GraphCaptureGuard.h"
#include "../../../execution/local_execution/device/DeviceWorkspaceManager.h"
#include "../../../execution/local_execution/device/WorkspaceDescriptor.h"
#include "../../../backends/BackendManager.h"
#include "../../../backends/GPUDeviceContextPool.h"
#include "../../../utils/Logger.h"
#include "../../../utils/DebugEnv.h"
#include "../../../utils/PerfStatsCollector.h"
#include "../../../tensors/GpuTensorView.h"
#include "../../kvcache/KVCacheLogicalBlockCodec.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "../../../tensors/BlockStructures.h"
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <type_traits>

namespace llaminar2
{
    namespace
    {
        /**
         * @brief Check whether the opt-in MTP publication trace is enabled.
         *
         * Captured append kernels cannot perform host-side device probes while
         * CUDA graph capture is active, but they can report which dynamic
         * scalar path they record.  Keeping this behind the publication
         * diagnostic switch avoids noisy logs in normal inference.
         */
        bool mtpKVPublicationDiagnosticsEnabled()
        {
            return debugEnv().runtime_debug.mtp_publication_diagnostics;
        }
    } // namespace

    // =========================================================================
    // CUDA Kernels
    // =========================================================================

    /**
     * @brief Resolve the real append row count for captured bucket execution.
     *
     * CUDA graph replay uses fixed bucket launch geometry, while the request
     * may contain fewer real tokens. Persistent request metadata already owns
     * the real append count on device before replay; copy kernels use this guard
     * so padded rows never mutate KV cache storage. Invalid device metadata
     * produces a zero-row copy because admission validation owns the error.
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
     * @brief Canonicalize a native floating zero without changing other bits.
     *
     * Prefix payloads are byte-comparable. IEEE negative zero is numerically
     * equal to positive zero but would otherwise make an equivalent cache row
     * hash differently. Quantized block payloads bypass this helper unchanged.
     */
    template <typename T>
    __device__ __forceinline__ T canonical_logical_kv_value(T value)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            const uint32_t bits = __float_as_uint(value);
            return (bits & 0x7fffffffu) == 0u ? 0.0f : value;
        }
        else if constexpr (std::is_same_v<T, __half>)
        {
            const uint16_t bits = __half_as_ushort(value);
            return (bits & 0x7fffu) == 0u ? __ushort_as_half(0u) : value;
        }
        else if constexpr (std::is_same_v<T, __nv_bfloat16>)
        {
            const uint16_t bits = __bfloat16_as_ushort(value);
            return (bits & 0x7fffu) == 0u ? __ushort_as_bfloat16(0u) : value;
        }
        else
        {
            return value;
        }
    }

    /**
     * @brief Gather one logical ring slice into a device-resident prefix block.
     *
     * Every launch reads the authoritative device head/count pair. The host
     * supplies only immutable geometry and never adopts mutable sequence state.
     * K and V share one launch so their readiness is represented by one stream
     * edge and one prefix-block event.
     */
    template <typename T>
    __global__ void ring_logical_block_export_device_kernel(
        const T *__restrict__ ring_k,
        const T *__restrict__ ring_v,
        T *__restrict__ block_k,
        T *__restrict__ block_v,
        const int *__restrict__ ring_head,
        const int *__restrict__ cached_tokens,
        int logical_token_start,
        int token_count,
        int max_seq_len,
        int row_elements)
    {
        const int count = *cached_tokens;
        const int head = *ring_head;
        if (count < 0 || count > max_seq_len ||
            head < 0 || head >= max_seq_len ||
            logical_token_start < 0 ||
            logical_token_start > count ||
            token_count < 0 ||
            token_count > count - logical_token_start)
        {
            return;
        }

        int tail = (head - count) % max_seq_len;
        if (tail < 0)
            tail += max_seq_len;

        const size_t element_count =
            static_cast<size_t>(token_count) *
            static_cast<size_t>(row_elements);
        for (size_t linear =
                 static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
             linear < element_count;
             linear += static_cast<size_t>(gridDim.x) * blockDim.x)
        {
            const int token =
                static_cast<int>(linear / static_cast<size_t>(row_elements));
            const int element =
                static_cast<int>(linear % static_cast<size_t>(row_elements));
            const int physical =
                (tail + logical_token_start + token) % max_seq_len;
            const size_t source =
                static_cast<size_t>(physical) *
                    static_cast<size_t>(row_elements) +
                static_cast<size_t>(element);
            block_k[linear] = canonical_logical_kv_value(ring_k[source]);
            block_v[linear] = canonical_logical_kv_value(ring_v[source]);
        }
    }

    /**
     * @brief Scatter one device prefix block into the canonical live ring.
     *
     * Prefix restore imports blocks in monotonically increasing logical order.
     * The device head/count validation prevents an out-of-order block from
     * mutating payload storage without consulting a stale host mirror.
     */
    template <typename T>
    __global__ void ring_logical_block_import_device_kernel(
        T *__restrict__ ring_k,
        T *__restrict__ ring_v,
        const T *__restrict__ block_k,
        const T *__restrict__ block_v,
        const int *__restrict__ ring_head,
        const int *__restrict__ cached_tokens,
        int logical_token_start,
        int token_count,
        int max_seq_len,
        int row_elements)
    {
        const int count = *cached_tokens;
        const int head = *ring_head;
        if (count != logical_token_start ||
            head != (logical_token_start % max_seq_len))
        {
            return;
        }

        const size_t element_count =
            static_cast<size_t>(token_count) *
            static_cast<size_t>(row_elements);
        for (size_t linear =
                 static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
             linear < element_count;
             linear += static_cast<size_t>(gridDim.x) * blockDim.x)
        {
            const int token =
                static_cast<int>(linear / static_cast<size_t>(row_elements));
            const int element =
                static_cast<int>(linear % static_cast<size_t>(row_elements));
            const int physical = (head + token) % max_seq_len;
            const size_t destination =
                static_cast<size_t>(physical) *
                    static_cast<size_t>(row_elements) +
                static_cast<size_t>(element);
            ring_k[destination] = block_k[linear];
            ring_v[destination] = block_v[linear];
        }
    }

    /**
     * @brief Publish sequence metadata after the preceding import kernel.
     *
     * This is a separate one-thread launch so all payload blocks complete
     * before later stream work can observe the advanced count. It repeats the
     * device validation and leaves metadata unchanged on an out-of-order
     * import.
     */
    __global__ void ring_logical_block_import_publish_kernel(
        int *__restrict__ ring_head,
        int *__restrict__ cached_tokens,
        int logical_token_start,
        int token_count,
        int max_seq_len)
    {
        if (blockIdx.x != 0 || threadIdx.x != 0)
            return;
        const int count = *cached_tokens;
        const int head = *ring_head;
        if (count != logical_token_start ||
            head != (logical_token_start % max_seq_len))
        {
            return;
        }
        const int new_count = logical_token_start + token_count;
        *cached_tokens = new_count;
        *ring_head = new_count % max_seq_len;
    }

    /**
     * @brief Append tokens to ring buffer (graph-capturable variant)
     *
     * Reads head position from a device pointer instead of a scalar argument.
     * This allows one GPU graph to replay against the canonical head advanced
     * by prior append/publication nodes, with no host adoption or H2D update.
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
        const int *__restrict__ d_append_count, // Real rows for padded bucket replay
        int max_seq_len,                // Ring buffer capacity
        int kv_dim,                     // n_kv_heads * head_dim
        int num_tokens)                 // Tokens to append
    {
        int token_idx = blockIdx.x;
        int elem_idx = blockIdx.y * blockDim.x + threadIdx.x;

        const int rows_to_write = effective_append_tokens(d_append_count, num_tokens);
        const int source_start = rows_to_write > max_seq_len
                                     ? rows_to_write - max_seq_len
                                     : 0;
        if (token_idx < source_start || token_idx >= rows_to_write ||
            elem_idx >= kv_dim)
            return;

        int head = *d_head; // Read from device memory
        int dst_pos = (head + token_idx) % max_seq_len;
        int dst_offset = dst_pos * kv_dim + elem_idx;
        int src_offset = token_idx * kv_dim + elem_idx;

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

    /**
     * @brief Replace canonical sequence metadata at an explicit host boundary.
     *
     * Prefix import, truncation, and request reset may originate from a host
     * control decision, but the resulting live state is written by a stream-
     * ordered device kernel. No pinned mirror or H2D metadata payload exists.
     */
    __global__ void cuda_kv_sequence_state_set_kernel(
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
     * @brief Gather one request's canonical ring metadata into an opaque checkpoint.
     *
     * The cache stores metadata in layer-major [layer, request] rows. A live
     * MTP checkpoint needs one request across every layer, so this kernel
     * gathers the strided rows into contiguous [heads][counts] arrays without
     * involving the host. Each layer is independent and handled by one thread.
     */
    __global__ void cuda_kv_sequence_state_checkpoint_capture_kernel(
        const int *__restrict__ d_heads,
        const int *__restrict__ d_counts,
        int *__restrict__ checkpoint,
        int n_layers,
        int batch_size,
        int seq_idx)
    {
        const int layer =
            static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        if (layer >= n_layers)
            return;

        const int entry_idx = layer * batch_size + seq_idx;
        checkpoint[layer] = d_heads[entry_idx];
        checkpoint[n_layers + layer] = d_counts[entry_idx];
    }

    /**
     * @brief Restore one request's exact canonical ring metadata on device.
     *
     * Restoring metadata makes speculative rows beyond the saved logical
     * window unreachable. The KV payload itself is intentionally not replayed
     * or copied because speculative appends are admitted only with enough
     * headroom to avoid overwriting any row visible at checkpoint time.
     */
    __global__ void cuda_kv_sequence_state_checkpoint_restore_kernel(
        int *__restrict__ d_heads,
        int *__restrict__ d_counts,
        const int *__restrict__ checkpoint,
        int n_layers,
        int batch_size,
        int seq_idx)
    {
        const int layer =
            static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        if (layer >= n_layers)
            return;

        const int entry_idx = layer * batch_size + seq_idx;
        d_heads[entry_idx] = checkpoint[layer];
        d_counts[entry_idx] = checkpoint[n_layers + layer];
    }

    /**
     * @brief Truncate every layer for one request using canonical device state.
     *
     * The host supplies only the requested logical length. Each device thread
     * derives the old ring tail from its own canonical head/count pair and
     * publishes the new visible window without any D2H observation.
     */
    __global__ void cuda_kv_sequence_state_truncate_kernel(
        int *__restrict__ d_heads,
        int *__restrict__ d_counts,
        int n_layers,
        int batch_size,
        int seq_idx,
        int cached_tokens,
        int max_seq_len)
    {
        const int layer =
            static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        if (layer >= n_layers)
            return;

        const int entry_idx = layer * batch_size + seq_idx;
        const int old_head = d_heads[entry_idx];
        const int old_count = d_counts[entry_idx];
        if (old_head < 0 || old_head >= max_seq_len ||
            old_count < 0 || old_count > max_seq_len ||
            cached_tokens > old_count)
        {
            /*
             * Truncation is a monotonic rollback operation. Silently deriving
             * a larger head would expose stale ring payload as live KV state.
             * Keep the canonical pair untouched when a caller requests an
             * extension; production callers treat their target as an invariant.
             */
            return;
        }

        int tail = old_head - old_count;
        tail %= max_seq_len;
        if (tail < 0)
            tail += max_seq_len;
        d_heads[entry_idx] =
            cached_tokens == 0
                ? 0
                : (tail + cached_tokens) % max_seq_len;
        d_counts[entry_idx] = cached_tokens;
    }

    /**
     * @brief Evict oldest visible rows without materializing ring state on host.
     *
     * Eviction changes only the logical tail of the visible window. Because the
     * ring head names the position after the newest row, keeping it unchanged
     * and reducing the count is sufficient to drop the oldest rows. A corrupt
     * canonical pair is left untouched so the caller cannot manufacture a
     * plausible but incorrect sequence state.
     */
    __global__ void cuda_kv_sequence_state_evict_oldest_kernel(
        int *__restrict__ d_heads,
        int *__restrict__ d_counts,
        int batch_size,
        int layer,
        int seq_idx,
        int num_tokens,
        int max_seq_len)
    {
        if (blockIdx.x != 0 || threadIdx.x != 0)
            return;

        const int entry_idx = layer * batch_size + seq_idx;
        const int old_head = d_heads[entry_idx];
        const int old_count = d_counts[entry_idx];
        if (old_head < 0 || old_head >= max_seq_len ||
            old_count < 0 || old_count > max_seq_len)
        {
            return;
        }

        d_counts[entry_idx] =
            num_tokens >= old_count ? 0 : old_count - num_tokens;
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
        if (old_head < 0 || old_head >= max_seq_len ||
            old_count < 0 || old_count > max_seq_len ||
            target_count > old_count)
        {
            /*
             * Accepted-state publication can only remove verifier rows. A
             * larger target would manufacture history from stale ring bytes,
             * so refuse the entire head/count update atomically.
             */
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
     * @brief Gather request-local rings using only persistent device metadata.
     *
     * `d_heads` and `d_counts` are the same scalars updated by captured append
     * kernels.  Consequently this gather can follow append in a captured graph
     * without consulting a stale host `EntryT`.  Entry pointers are immutable
     * after cache construction and are uploaded as one all-layer table when the
     * cache workspace is bound.
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
        const int ring_count = min(max(d_counts[entry], 0), max_seq_len);
        const int visible_count = min(ring_count, output_stride);
        const int skipped_rows = ring_count - visible_count;
        const int tail =
            (d_heads[entry] - ring_count + max_seq_len) % max_seq_len;
        const int token_limit =
            ZeroInactiveRows ? output_stride : visible_count;

        /*
         * A small fixed token grid is graph-stable and avoids launching one
         * block per configured context row. Each block walks a disjoint token
         * stripe up to the live device count, so replay work grows with useful
         * history rather than with the maximum context allocation.
         */
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
    __device__ inline float to_float_device<__half>(__half v)
    {
        return __half2float(v);
    }

    template <>
    __device__ inline float to_float_device<__nv_bfloat16>(__nv_bfloat16 v)
    {
        return __bfloat162float(v);
    }

    /**
     * @brief Load one logical cache element as FP32.
     *
     * Floating-point cache rows are ordinary dense arrays. Q8_1 rows use one
     * scale and 32 signed values per block, so their specialization performs
     * exactly the same FP16-scale dequantization as the established conversion
     * kernel. Keeping this accessor in the grouped gather preserves conversion
     * arithmetic while avoiding an intermediate native-format materialization.
     */
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
     * @brief Gather independent device-owned rings directly into FP16.
     *
     * The output is request-major `[request, max_kv_len, kv_dim]`. Every block
     * reads canonical device head/count values after the captured append and
     * zero-fills rows outside the live count. No host-sized view or validity bit
     * participates in the operation.
     */
    template <typename StorageT, bool ZeroInactiveRows>
    __global__ void ring_gather_batched_device_state_fp16_kernel(
        __half *__restrict__ k_output,
        __half *__restrict__ v_output,
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
        const int ring_count = min(max(counts[entry], 0), max_seq_len);
        const int visible_count = min(ring_count, output_stride);
        const int skipped_rows = ring_count - visible_count;
        const int tail =
            (heads[entry] - ring_count + max_seq_len) % max_seq_len;
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
                k_output[output_index] = __float2half_rn(0.0f);
                v_output[output_index] = __float2half_rn(0.0f);
                continue;
            }

            const int source_token =
                (tail + skipped_rows + output_token) % max_seq_len;
            k_output[output_index] = __float2half_rn(
                load_ring_logical_value(
                    k_entry_table[entry], source_token, element,
                    logical_kv_dim));
            v_output[output_index] = __float2half_rn(
                load_ring_logical_value(
                    v_entry_table[entry], source_token, element,
                    logical_kv_dim));
        }
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

    /**
     * @brief Convert a producer value to the cache's native floating format.
     *
     * The ring cache owns this conversion and the device sequence metadata.
     * Fusing both responsibilities into one launch keeps publication resident
     * on the producer stream and legal inside a reusable CUDA graph.
     */
    template <typename DstT>
    __device__ __forceinline__ DstT kv_cache_from_float(float value);

    template <>
    __device__ __forceinline__ float kv_cache_from_float<float>(float value)
    {
        return value;
    }

    template <>
    __device__ __forceinline__ __half kv_cache_from_float<__half>(float value)
    {
        return __float2half_rn(value);
    }

    template <>
    __device__ __forceinline__ __nv_bfloat16
    kv_cache_from_float<__nv_bfloat16>(float value)
    {
        return __float2bfloat16_rn(value);
    }

    template <typename DstT, typename SrcT>
    __global__ void ring_append_convert_kernel(
        DstT *__restrict__ d_K_cache,
        DstT *__restrict__ d_V_cache,
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

        d_K_cache[dst_offset] =
            kv_cache_from_float<DstT>(to_float_device(d_K_new[src_offset]));
        d_V_cache[dst_offset] =
            kv_cache_from_float<DstT>(to_float_device(d_V_new[src_offset]));
    }

    template <typename DstT>
    __global__ void ring_append_q8_1_convert_kernel(
        DstT *__restrict__ d_K_cache,
        DstT *__restrict__ d_V_cache,
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
        d_K_cache[dst_offset] = kv_cache_from_float<DstT>(
            k_scale * static_cast<float>(k_block.qs[lane]));
        d_V_cache[dst_offset] = kv_cache_from_float<DstT>(
            v_scale * static_cast<float>(v_block.qs[lane]));
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

    template <typename DstT>
    bool launch_ring_append_converted(
        DstT *d_K_cache, DstT *d_V_cache,
        const void *d_K_new, const void *d_V_new,
        TensorType src_type,
        const int *d_head,
        const int *d_append_count,
        int max_seq_len,
        int kv_dim,
        int num_tokens,
        cudaStream_t stream)
    {
        if (!d_K_cache || !d_V_cache || !d_K_new || !d_V_new || !d_head ||
            max_seq_len <= 0 || kv_dim <= 0 || num_tokens <= 0)
        {
            return false;
        }

        const dim3 block(256);
        const dim3 grid(num_tokens, (kv_dim + static_cast<int>(block.x) - 1) / static_cast<int>(block.x));
        switch (src_type)
        {
        case TensorType::FP32:
            ring_append_convert_kernel<DstT, float><<<grid, block, 0, stream>>>(
                d_K_cache, d_V_cache,
                               static_cast<const float *>(d_K_new),
                               static_cast<const float *>(d_V_new),
                               d_head, d_append_count, max_seq_len, kv_dim, num_tokens);
            break;
        case TensorType::FP16:
            ring_append_convert_kernel<DstT, __half><<<grid, block, 0, stream>>>(
                d_K_cache, d_V_cache,
                static_cast<const __half *>(d_K_new),
                static_cast<const __half *>(d_V_new),
                d_head, d_append_count, max_seq_len, kv_dim, num_tokens);
            break;
        case TensorType::BF16:
            ring_append_convert_kernel<DstT, __nv_bfloat16><<<grid, block, 0, stream>>>(
                d_K_cache, d_V_cache,
                               static_cast<const __nv_bfloat16 *>(d_K_new),
                               static_cast<const __nv_bfloat16 *>(d_V_new),
                               d_head, d_append_count, max_seq_len, kv_dim, num_tokens);
            break;
        case TensorType::Q8_1:
        {
            const int blocks_per_row = (kv_dim + Q8_1Block::BLOCK_SIZE - 1) / Q8_1Block::BLOCK_SIZE;
            ring_append_q8_1_convert_kernel<DstT><<<grid, block, 0, stream>>>(
                d_K_cache, d_V_cache,
                               static_cast<const Q8_1Block *>(d_K_new),
                               static_cast<const Q8_1Block *>(d_V_new),
                               d_head, d_append_count, max_seq_len, kv_dim, num_tokens, blocks_per_row);
            break;
        }
        default:
            return false;
        }

        return cudaGetLastError() == cudaSuccess;
    }

    extern "C" bool cuda_ring_append_converted_fp32(
        float *d_K_cache, float *d_V_cache,
        const void *d_K_new, const void *d_V_new,
        TensorType src_type,
        const int *d_head,
        const int *d_append_count,
        int max_seq_len,
        int kv_dim,
        int num_tokens,
        cudaStream_t stream)
    {
        return launch_ring_append_converted(
            d_K_cache, d_V_cache, d_K_new, d_V_new, src_type,
            d_head, d_append_count, max_seq_len, kv_dim, num_tokens, stream);
    }

    extern "C" bool cuda_ring_append_converted_fp16(
        __half *d_K_cache, __half *d_V_cache,
        const void *d_K_new, const void *d_V_new,
        TensorType src_type,
        const int *d_head,
        const int *d_append_count,
        int max_seq_len,
        int kv_dim,
        int num_tokens,
        cudaStream_t stream)
    {
        return launch_ring_append_converted(
            d_K_cache, d_V_cache, d_K_new, d_V_new, src_type,
            d_head, d_append_count, max_seq_len, kv_dim, num_tokens, stream);
    }

    extern "C" bool cuda_ring_append_converted_bf16(
        __nv_bfloat16 *d_K_cache, __nv_bfloat16 *d_V_cache,
        const void *d_K_new, const void *d_V_new,
        TensorType src_type,
        const int *d_head,
        const int *d_append_count,
        int max_seq_len,
        int kv_dim,
        int num_tokens,
        cudaStream_t stream)
    {
        return launch_ring_append_converted(
            d_K_cache, d_V_cache, d_K_new, d_V_new, src_type,
            d_head, d_append_count, max_seq_len, kv_dim, num_tokens, stream);
    }

    // =========================================================================
    // Kernel Launch Helpers (extern "C" wrappers)
    // =========================================================================

    // FP32 scalar diagnostic linearization.
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

    // FP16 scalar diagnostic linearization.
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

    // BF16 scalar diagnostic linearization.
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

    // Q8_1 scalar diagnostic linearization.
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
        const int *d_head, const int *d_append_count, int max_seq_len, int kv_dim, int num_tokens,
        cudaStream_t stream)
    {
        if (num_tokens == 0)
            return;

        dim3 block(256);
        dim3 grid(num_tokens, (kv_dim + 255) / 256);
        ring_append_kernel_dynamic<float><<<grid, block, 0, stream>>>(
            d_K_cache, d_V_cache, d_K_new, d_V_new,
            d_head, d_append_count, max_seq_len, kv_dim, num_tokens);
    }

    extern "C" void cuda_ring_append_dynamic_fp16(
        __half *d_K_cache, __half *d_V_cache,
        const __half *d_K_new, const __half *d_V_new,
        const int *d_head, const int *d_append_count, int max_seq_len, int kv_dim, int num_tokens,
        cudaStream_t stream)
    {
        if (num_tokens == 0)
            return;

        dim3 block(256);
        dim3 grid(num_tokens, (kv_dim + 255) / 256);
        ring_append_kernel_dynamic<__half><<<grid, block, 0, stream>>>(
            d_K_cache, d_V_cache, d_K_new, d_V_new,
            d_head, d_append_count, max_seq_len, kv_dim, num_tokens);
    }

    extern "C" void cuda_ring_append_dynamic_bf16(
        __nv_bfloat16 *d_K_cache, __nv_bfloat16 *d_V_cache,
        const __nv_bfloat16 *d_K_new, const __nv_bfloat16 *d_V_new,
        const int *d_head, const int *d_append_count, int max_seq_len, int kv_dim, int num_tokens,
        cudaStream_t stream)
    {
        if (num_tokens == 0)
            return;

        dim3 block(256);
        dim3 grid(num_tokens, (kv_dim + 255) / 256);
        ring_append_kernel_dynamic<__nv_bfloat16><<<grid, block, 0, stream>>>(
            d_K_cache, d_V_cache, d_K_new, d_V_new,
            d_head, d_append_count, max_seq_len, kv_dim, num_tokens);
    }

    extern "C" void cuda_ring_append_dynamic_q8_1(
        Q8_1Block *d_K_cache, Q8_1Block *d_V_cache,
        const Q8_1Block *d_K_new, const Q8_1Block *d_V_new,
        const int *d_head, const int *d_append_count, int max_seq_len, int kv_blocks, int num_tokens,
        cudaStream_t stream)
    {
        if (num_tokens == 0)
            return;

        dim3 block(256);
        dim3 grid(num_tokens, (kv_blocks + 255) / 256);
        ring_append_kernel_dynamic<Q8_1Block><<<grid, block, 0, stream>>>(
            d_K_cache, d_V_cache, d_K_new, d_V_new,
            d_head, d_append_count, max_seq_len, kv_blocks, num_tokens);
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

    extern "C" bool cuda_kv_sequence_state_set(
        int *d_head,
        int *d_count,
        int head,
        int count,
        int max_seq_len,
        cudaStream_t stream)
    {
        if (!d_head || !d_count || !stream ||
            head < 0 || head >= max_seq_len ||
            count < 0 || count > max_seq_len)
        {
            return false;
        }
        cuda_kv_sequence_state_set_kernel<<<1, 1, 0, stream>>>(
            d_head, d_count, head, count);
        return cudaGetLastError() == cudaSuccess;
    }

    extern "C" bool cuda_kv_sequence_state_checkpoint_capture(
        const int *d_heads,
        const int *d_counts,
        int *checkpoint,
        int n_layers,
        int batch_size,
        int seq_idx,
        cudaStream_t stream)
    {
        if (!d_heads || !d_counts || !checkpoint || !stream ||
            n_layers <= 0 || batch_size <= 0 ||
            seq_idx < 0 || seq_idx >= batch_size)
        {
            return false;
        }

        constexpr int kThreads = 128;
        const int blocks = (n_layers + kThreads - 1) / kThreads;
        cuda_kv_sequence_state_checkpoint_capture_kernel<<<
            blocks, kThreads, 0, stream>>>(
            d_heads,
            d_counts,
            checkpoint,
            n_layers,
            batch_size,
            seq_idx);
        return cudaGetLastError() == cudaSuccess;
    }

    extern "C" bool cuda_kv_sequence_state_checkpoint_restore(
        int *d_heads,
        int *d_counts,
        const int *checkpoint,
        int n_layers,
        int batch_size,
        int seq_idx,
        cudaStream_t stream)
    {
        if (!d_heads || !d_counts || !checkpoint || !stream ||
            n_layers <= 0 || batch_size <= 0 ||
            seq_idx < 0 || seq_idx >= batch_size)
        {
            return false;
        }

        constexpr int kThreads = 128;
        const int blocks = (n_layers + kThreads - 1) / kThreads;
        cuda_kv_sequence_state_checkpoint_restore_kernel<<<
            blocks, kThreads, 0, stream>>>(
            d_heads,
            d_counts,
            checkpoint,
            n_layers,
            batch_size,
            seq_idx);
        return cudaGetLastError() == cudaSuccess;
    }

    extern "C" bool cuda_kv_sequence_state_truncate(
        int *d_heads,
        int *d_counts,
        int n_layers,
        int batch_size,
        int seq_idx,
        int cached_tokens,
        int max_seq_len,
        cudaStream_t stream)
    {
        if (!d_heads || !d_counts || !stream ||
            n_layers <= 0 || batch_size <= 0 ||
            seq_idx < 0 || seq_idx >= batch_size ||
            cached_tokens < 0 || cached_tokens > max_seq_len ||
            max_seq_len <= 0)
        {
            return false;
        }

        constexpr int kThreads = 128;
        const int blocks = (n_layers + kThreads - 1) / kThreads;
        cuda_kv_sequence_state_truncate_kernel<<<
            blocks, kThreads, 0, stream>>>(
            d_heads,
            d_counts,
            n_layers,
            batch_size,
            seq_idx,
            cached_tokens,
            max_seq_len);
        return cudaGetLastError() == cudaSuccess;
    }

    extern "C" bool cuda_kv_sequence_state_evict_oldest(
        int *d_heads,
        int *d_counts,
        int n_layers,
        int batch_size,
        int layer,
        int seq_idx,
        int num_tokens,
        int max_seq_len,
        cudaStream_t stream)
    {
        if (!d_heads || !d_counts || !stream ||
            n_layers <= 0 || batch_size <= 0 ||
            layer < 0 || layer >= n_layers ||
            seq_idx < 0 || seq_idx >= batch_size ||
            num_tokens < 0 || max_seq_len <= 0)
        {
            return false;
        }

        cuda_kv_sequence_state_evict_oldest_kernel<<<1, 1, 0, stream>>>(
            d_heads,
            d_counts,
            batch_size,
            layer,
            seq_idx,
            num_tokens,
            max_seq_len);
        return cudaGetLastError() == cudaSuccess;
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
        allocate_all_entries();
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
        allocate_all_entries();
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
        allocate_all_entries();
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
        allocate_all_entries();
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

        // Canonical device sequence metadata is freed by the common base.

        // Release pointer topology before the entry allocations it addresses.
        releaseBatchedEntryPointerTables();

        for (auto &layer_entries : entries_)
        {
            for (auto &entry : layer_entries)
            {
                free_entry(entry);
            }
        }

        // RoPE shadows are non-owning views over graph-planned conversion
        // workspace. Their vector destructors release only host-side wrappers.
    }

    template <ActivationPrecision Precision>
    void CUDARingKVCache<Precision>::allocate_entry(EntryT &entry)
    {
        auto *const backend = getCUDABackend();
        if (!backend)
        {
            throw std::runtime_error(
                "[CUDARingKVCache] CUDA backend unavailable during permanent entry allocation");
        }

        const size_t buffer_size =
            static_cast<size_t>(max_seq_len_) *
            static_cast<size_t>(kv_storage_dim_) *
            sizeof(DataT);
        auto allocate_buffer =
            [backend, buffer_size, this](auto **destination, const char *name)
        {
            using PointerT = std::remove_reference_t<decltype(*destination)>;
            *destination = static_cast<PointerT>(
                backend->allocate(buffer_size, device_id_));
            if (!*destination)
            {
                throw std::runtime_error(
                    std::string("[CUDARingKVCache] Failed to allocate ") +
                    name + " through the CUDA backend");
            }
        };

        // Every pointer is permanent cache topology. Any partial failure
        // escapes to allocate_all_entries(), which releases all completed and
        // partially completed entries before propagating the fatal error.
        allocate_buffer(&entry.d_K, "K entry");
        allocate_buffer(&entry.d_V, "V entry");
        allocate_buffer(&entry.d_K_scratch, "K linearization scratch");
        allocate_buffer(&entry.d_V_scratch, "V linearization scratch");
    }

    template <ActivationPrecision Precision>
    void CUDARingKVCache<Precision>::free_entry(EntryT &entry)
    {
        auto *const backend = getCUDABackend();
        if ((entry.d_K || entry.d_V ||
             entry.d_K_scratch || entry.d_V_scratch) &&
            !backend)
        {
            LOG_ERROR("[CUDARingKVCache] CUDA backend unavailable while releasing permanent entry storage");
            std::terminate();
        }

        if (entry.d_K)
            backend->free(entry.d_K, device_id_);
        if (entry.d_V)
            backend->free(entry.d_V, device_id_);
        if (entry.d_K_scratch)
            backend->free(entry.d_K_scratch, device_id_);
        if (entry.d_V_scratch)
            backend->free(entry.d_V_scratch, device_id_);

        entry.d_K = nullptr;
        entry.d_V = nullptr;
        entry.d_K_scratch = nullptr;
        entry.d_V_scratch = nullptr;
    }

    /**
     * @brief Allocate permanent CUDA entry topology as one constructor transaction.
     *
     * Raw CUDA pointers are stored inside EntryT, so the derived destructor
     * cannot clean them when construction throws. This transaction centralizes
     * all four constructors and explicitly unwinds partial ownership.
     */
    template <ActivationPrecision Precision>
    void CUDARingKVCache<Precision>::allocate_all_entries()
    {
        const cudaError_t device_status = cudaSetDevice(device_id_);
        if (device_status != cudaSuccess)
        {
            throw std::runtime_error(
                std::string("[CUDARingKVCache] Failed to select cache device: ") +
                cudaGetErrorString(device_status));
        }

        try
        {
            entries_.resize(n_layers_);
            for (int layer = 0; layer < n_layers_; ++layer)
            {
                entries_[layer].resize(batch_size_);
                for (int seq = 0; seq < batch_size_; ++seq)
                {
                    allocate_entry(entries_[layer][seq]);
                }
            }

            // Views contain no payload ownership and can be constructed after
            // every permanent device entry has been allocated successfully.
            tensor_views_.resize(n_layers_);
            for (int layer = 0; layer < n_layers_; ++layer)
            {
                tensor_views_[layer].resize(batch_size_);
            }

            initializeBatchedEntryPointerTables();
            allocateDeviceParams();
        }
        catch (...)
        {
            releaseBatchedEntryPointerTables();
            for (auto &layer_entries : entries_)
            {
                for (auto &entry : layer_entries)
                {
                    free_entry(entry);
                }
            }
            throw;
        }

        LOG_DEBUG("[CUDARingKVCache] Allocated "
                  << (n_layers_ * batch_size_ * 4 * max_seq_len_ * kv_dim_ * sizeof(DataT)) /
                         (1024 * 1024)
                  << " MB total (including scratch)");
    }

    /**
     * @brief Publish the permanent CUDA entry topology consumed by grouped gathers.
     */
    template <ActivationPrecision Precision>
    void CUDARingKVCache<Precision>::initializeBatchedEntryPointerTables()
    {
        if (batched_pointer_tables_ready_ ||
            d_batched_k_entry_table_ ||
            d_batched_v_entry_table_)
        {
            throw std::logic_error(
                "[CUDARingKVCache] Immutable entry pointer tables may only be initialized once");
        }

        const size_t entry_count =
            static_cast<size_t>(n_layers_) * static_cast<size_t>(batch_size_);
        if (entry_count == 0)
        {
            batched_pointer_tables_ready_ = true;
            return;
        }

        std::vector<DataT *> h_k_table(entry_count);
        std::vector<DataT *> h_v_table(entry_count);
        for (int layer = 0; layer < n_layers_; ++layer)
        {
            for (int seq = 0; seq < batch_size_; ++seq)
            {
                const size_t index =
                    static_cast<size_t>(layer) * static_cast<size_t>(batch_size_) +
                    static_cast<size_t>(seq);
                h_k_table[index] = entries_[layer][seq].d_K;
                h_v_table[index] = entries_[layer][seq].d_V;
                if (!h_k_table[index] || !h_v_table[index])
                {
                    throw std::runtime_error(
                        "[CUDARingKVCache] Cannot publish a null permanent KV entry");
                }
            }
        }

        auto *const backend = getCUDABackend();
        if (!backend)
        {
            throw std::runtime_error(
                "[CUDARingKVCache] CUDA backend unavailable while publishing immutable entry topology");
        }

        const size_t table_bytes = entry_count * sizeof(DataT *);
        d_batched_k_entry_table_ =
            static_cast<DataT **>(backend->allocate(table_bytes, device_id_));
        d_batched_v_entry_table_ =
            static_cast<DataT **>(backend->allocate(table_bytes, device_id_));
        if (!d_batched_k_entry_table_ ||
            !d_batched_v_entry_table_ ||
            !backend->hostToDevice(
                d_batched_k_entry_table_,
                h_k_table.data(),
                table_bytes,
                device_id_) ||
            !backend->hostToDevice(
                d_batched_v_entry_table_,
                h_v_table.data(),
                table_bytes,
                device_id_))
        {
            releaseBatchedEntryPointerTables();
            throw std::runtime_error(
                "[CUDARingKVCache] Failed to publish immutable batched entry pointer tables");
        }

        batched_pointer_tables_ready_ = true;
    }

    /**
     * @brief Release immutable grouped-gather topology without throwing.
     */
    template <ActivationPrecision Precision>
    void CUDARingKVCache<Precision>::releaseBatchedEntryPointerTables() noexcept
    {
        auto *const backend = getCUDABackend();
        if ((d_batched_k_entry_table_ || d_batched_v_entry_table_) && !backend)
        {
            LOG_ERROR("[CUDARingKVCache] CUDA backend unavailable while releasing immutable entry topology");
            std::terminate();
        }
        try
        {
            if (d_batched_k_entry_table_)
                backend->free(d_batched_k_entry_table_, device_id_);
            if (d_batched_v_entry_table_)
                backend->free(d_batched_v_entry_table_, device_id_);
        }
        catch (...)
        {
            LOG_ERROR("[CUDARingKVCache] CUDA backend threw while releasing immutable entry topology");
            std::terminate();
        }
        d_batched_k_entry_table_ = nullptr;
        d_batched_v_entry_table_ = nullptr;
        batched_pointer_tables_ready_ = false;
    }

    // =========================================================================
    // Dynamic Head Kernel Launcher (graph-capturable)
    // =========================================================================

    // Forward declarations for dynamic wrappers
    extern "C" void cuda_ring_append_dynamic_fp32(
        float *, float *, const float *, const float *,
        const int *, const int *, int, int, int, cudaStream_t);
    extern "C" void cuda_ring_append_dynamic_fp16(
        __half *, __half *, const __half *, const __half *,
        const int *, const int *, int, int, int, cudaStream_t);
    extern "C" void cuda_ring_append_dynamic_bf16(
        __nv_bfloat16 *, __nv_bfloat16 *, const __nv_bfloat16 *, const __nv_bfloat16 *,
        const int *, const int *, int, int, int, cudaStream_t);
    extern "C" void cuda_ring_append_dynamic_q8_1(
        Q8_1Block *, Q8_1Block *, const Q8_1Block *, const Q8_1Block *,
        const int *, const int *, int, int, int, cudaStream_t);
    extern "C" void cuda_kv_sequence_state_advance(
        int *, int *, int, int, cudaStream_t);
    extern "C" void cuda_kv_sequence_state_advance_dynamic(
        int *, int *, const int *, int, int, cudaStream_t);
    extern "C" bool cuda_ring_append_converted_fp32(
        float *, float *, const void *, const void *, TensorType,
        const int *, const int *, int, int, int, cudaStream_t);
    extern "C" bool cuda_ring_append_converted_fp16(
        __half *, __half *, const void *, const void *, TensorType,
        const int *, const int *, int, int, int, cudaStream_t);
    extern "C" bool cuda_ring_append_converted_bf16(
        __nv_bfloat16 *, __nv_bfloat16 *,
        const void *, const void *, TensorType,
        const int *, const int *, int, int, int, cudaStream_t);

    template <ActivationPrecision Precision>
    void CUDARingKVCache<Precision>::launch_append_kernel_dynamic(
        EntryT &entry, const DataT *d_k, const DataT *d_v,
        const int *d_head, const int *d_append_count, int num_tokens, cudaStream_t stream)
    {
        if constexpr (Precision == ActivationPrecision::FP32)
        {
            cuda_ring_append_dynamic_fp32(
                entry.d_K, entry.d_V, d_k, d_v,
                d_head, d_append_count, max_seq_len_, kv_dim_, num_tokens, stream);
        }
        else if constexpr (Precision == ActivationPrecision::FP16)
        {
            cuda_ring_append_dynamic_fp16(
                entry.d_K, entry.d_V, d_k, d_v,
                d_head, d_append_count, max_seq_len_, kv_dim_, num_tokens, stream);
        }
        else if constexpr (Precision == ActivationPrecision::BF16)
        {
            cuda_ring_append_dynamic_bf16(
                entry.d_K, entry.d_V, d_k, d_v,
                d_head, d_append_count, max_seq_len_, kv_storage_dim_, num_tokens, stream);
        }
        else if constexpr (Precision == ActivationPrecision::Q8_1)
        {
            cuda_ring_append_dynamic_q8_1(
                entry.d_K, entry.d_V, d_k, d_v,
                d_head, d_append_count, max_seq_len_, kv_storage_dim_, num_tokens, stream);
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

        // Every CUDA append reads and advances the canonical device sequence
        // state. Exact-shape graphs pass a null append-count source; padded
        // graphs bind the persistent request-length row directly.
        if (!d_head_params_ || !d_count_params_)
        {
            LOG_ERROR("[CUDARingKVCache::append] Canonical device sequence state is unavailable");
            return false;
        }
        const int idx = layer * batch_size_ + seq_idx;
        const int *d_append_count = deviceDynamicAppendCountPtr(layer, seq_idx);
        if (mtpKVPublicationDiagnosticsEnabled())
        {
            LOG_INFO("[MTPPublicationDiagnostics] phase=cuda_kv_append_capture_enqueue"
                     << " path=typed"
                     << " layer=" << layer
                     << " seq_idx=" << seq_idx
                     << " captured_num_tokens=" << num_tokens
                     << " d_head=" << static_cast<const void *>(&d_head_params_[idx])
                     << " d_count=" << static_cast<const void *>(&d_count_params_[idx])
                     << " d_append_count=" << static_cast<const void *>(d_append_count)
                     << " stream=" << static_cast<void *>(effective_stream));
        }
        launch_append_kernel_dynamic(entry, d_k, d_v,
                                     &d_head_params_[idx], d_append_count,
                                     num_tokens, effective_stream);
        cuda_kv_sequence_state_advance_dynamic(
            &d_head_params_[idx], &d_count_params_[idx],
            d_append_count,
            num_tokens, max_seq_len_, effective_stream);

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
            verifier_rows < 1 ||
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
        const int source_start = std::max(0, verifier_rows - max_seq_len_);
        const int rows_to_write = verifier_rows - source_start;

        const int head_storage_dim =
            (Precision == ActivationPrecision::Q8_1)
                ? (head_dim_ + Q8_1Block::BLOCK_SIZE - 1) / Q8_1Block::BLOCK_SIZE
                : head_dim_;

        if (!d_head_params_ || !d_count_params_)
        {
            LOG_ERROR("[CUDARingKVCache] Grouped verifier requires canonical device sequence state");
            return false;
        }
        const int idx = layer * batch_size_ + seq_idx;
        cuda_ring_append_verifier_rows_dynamic_typed<DataT>(
            entry.d_K, entry.d_V, typed_k, typed_v,
            &d_head_params_[idx], max_seq_len_, kv_storage_dim_, head_storage_dim,
            verifier_rows, source_start, rows_to_write,
            k_head_major, v_head_major, stream);
        cuda_kv_sequence_state_advance(
            &d_head_params_[idx], &d_count_params_[idx],
            verifier_rows, max_seq_len_, stream);

        const cudaError_t launch_err = cudaGetLastError();
        if (launch_err != cudaSuccess)
        {
            LOG_ERROR("[CUDARingKVCache] Verifier append kernel launch failed: "
                      << cudaGetErrorString(launch_err));
            return false;
        }

        // This counter is part of the grouped-verifier correctness contract.
        // A byte-equal result is not sufficient unless the test can also prove
        // that publication entered this one-launch grouped implementation rather
        // than an accidental row-at-a-time caller path.
        PerfStatsCollector::addCounter(
            "kernel",
            "cuda_kv_cache_grouped_verifier_append_calls",
            1.0,
            "verifier",
            "cuda",
            {{"cache_format", activationPrecisionToString(Precision)},
             {"source_k_format", K->dtype_name()},
             {"source_v_format", V->dtype_name()},
             {"verifier_rows", std::to_string(verifier_rows)},
             {"source_k_layout", k_head_major ? "head_major" : "position_major"},
             {"source_v_layout", v_head_major ? "head_major" : "position_major"},
             {"execution_mode", capture_active ? "graph_captured" : "eager_device_state"},
             {"topology", (local_n_kv_heads_ != n_kv_heads_ || kv_head_start_ != 0)
                              ? "local_tp_shard"
                              : "replicated"},
             {"commit_policy", "single_grouped_metadata_commit"},
             {"local_kv_heads", std::to_string(local_n_kv_heads_)},
             {"head_dim", std::to_string(head_dim_)},
             {"kv_head_start", std::to_string(kv_head_start_)}});

        return true;
    }

    template <ActivationPrecision Precision>
    bool CUDARingKVCache<Precision>::appendConvertedWithStream(
        int layer, int seq_idx,
        const void *d_k_src, const void *d_v_src,
        TensorType src_type,
        int num_tokens, cudaStream_t stream)
    {
        if constexpr (
            Precision != ActivationPrecision::FP32 &&
            Precision != ActivationPrecision::FP16 &&
            Precision != ActivationPrecision::BF16)
        {
            (void)layer;
            (void)seq_idx;
            (void)d_k_src;
            (void)d_v_src;
            (void)src_type;
            (void)num_tokens;
            (void)stream;
            LOG_ERROR("[CUDARingKVCache::appendConvertedWithStream] Converted append requires floating cache storage");
            return false;
        }
        else
        {
            if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
            {
                LOG_ERROR("[CUDARingKVCache::appendConvertedWithStream] Invalid layer=" << layer
                                                                                         << " or seq_idx=" << seq_idx);
                return false;
            }
            if (!d_k_src || !d_v_src || num_tokens < 0)
            {
                LOG_ERROR("[CUDARingKVCache::appendConvertedWithStream] Invalid source pointers or token count");
                return false;
            }
            if (!stream)
            {
                LOG_ERROR("[CUDARingKVCache::appendConvertedWithStream] Explicit CUDA stream is required");
                return false;
            }
            if (num_tokens == 0)
                return true;

            EntryT &entry = entries_[layer][seq_idx];
            const int idx = layer * batch_size_ + seq_idx;
            if (!d_head_params_ || !d_count_params_)
            {
                LOG_ERROR("[CUDARingKVCache::appendConvertedWithStream] Canonical device sequence state is unavailable");
                return false;
            }
            const int *d_head = &d_head_params_[idx];
            const int *d_append_count =
                deviceDynamicAppendCountPtr(layer, seq_idx);

            bool launch_ok = false;
            if constexpr (Precision == ActivationPrecision::FP32)
            {
                launch_ok = cuda_ring_append_converted_fp32(
                    entry.d_K, entry.d_V, d_k_src, d_v_src, src_type,
                    d_head, d_append_count,
                    max_seq_len_, kv_dim_, num_tokens, stream);
            }
            else if constexpr (Precision == ActivationPrecision::FP16)
            {
                launch_ok = cuda_ring_append_converted_fp16(
                    entry.d_K, entry.d_V, d_k_src, d_v_src, src_type,
                    d_head, d_append_count,
                    max_seq_len_, kv_dim_, num_tokens, stream);
            }
            else if constexpr (Precision == ActivationPrecision::BF16)
            {
                launch_ok = cuda_ring_append_converted_bf16(
                    entry.d_K, entry.d_V, d_k_src, d_v_src, src_type,
                    d_head, d_append_count,
                    max_seq_len_, kv_dim_, num_tokens, stream);
            }

            if (!launch_ok)
            {
                LOG_ERROR("[CUDARingKVCache::appendConvertedWithStream] Fused converted append launch failed");
                return false;
            }

            cuda_kv_sequence_state_advance_dynamic(
                &d_head_params_[idx], &d_count_params_[idx],
                d_append_count,
                num_tokens, max_seq_len_, stream);

            return true;
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
        KVCacheSequenceState state;
        if (!observeDeviceSequenceState(layer, seq_idx, &state))
        {
            LOG_ERROR("[CUDARingKVCache::get_kv] Could not observe canonical device sequence state");
            return false;
        }
        const int count = state.cached_tokens;
        const int head = state.implementation_head;
        *kv_len = count;

        if (count == 0)
        {
            *d_k_out = nullptr;
            *d_v_out = nullptr;
            return true;
        }

        // Optimization: if not wrapped, return direct pointers
        if (!state.wrapped)
        {
            const int tail = (head - count + max_seq_len_) % max_seq_len_;
            *d_k_out = entry.d_K + static_cast<size_t>(tail) * kv_storage_dim_;
            *d_v_out = entry.d_V + static_cast<size_t>(tail) * kv_storage_dim_;
            return true;
        }

        // Buffer is wrapped - need to linearize
        linearize_entry(entry, head, count, stream);
        ++linearization_count_;

        *d_k_out = entry.d_K_scratch;
        *d_v_out = entry.d_V_scratch;
        return true;
    }

    template <ActivationPrecision Precision>
    void CUDARingKVCache<Precision>::linearize_entry(
        EntryT &entry,
        int head,
        int count,
        cudaStream_t stream)
    {
        launch_linearize_kernel(
            entry, head, count,
            entry.d_K_scratch, entry.d_V_scratch, stream);
    }

    template <ActivationPrecision Precision>
    void CUDARingKVCache<Precision>::launch_linearize_kernel(
        const EntryT &entry,
        int head,
        int count,
        DataT *d_k_out,
        DataT *d_v_out,
        cudaStream_t stream)
    {
        const int tail = (head - count + max_seq_len_) % max_seq_len_;

        if constexpr (Precision == ActivationPrecision::FP32)
        {
            cuda_ring_linearize_fp32(
                d_k_out, d_v_out, entry.d_K, entry.d_V,
                tail, count, max_seq_len_, kv_dim_, stream);
        }
        else if constexpr (Precision == ActivationPrecision::FP16)
        {
            cuda_ring_linearize_fp16(
                d_k_out, d_v_out, entry.d_K, entry.d_V,
                tail, count, max_seq_len_, kv_dim_, stream);
        }
        else if constexpr (Precision == ActivationPrecision::BF16)
        {
            cuda_ring_linearize_bf16(
                d_k_out, d_v_out, entry.d_K, entry.d_V,
                tail, count, max_seq_len_, kv_storage_dim_, stream);
        }
        else if constexpr (Precision == ActivationPrecision::Q8_1)
        {
            cuda_ring_linearize_q8_1(
                d_k_out, d_v_out, entry.d_K, entry.d_V,
                tail, count, max_seq_len_, kv_storage_dim_, stream);
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
        KVCacheSequenceState state;
        if (!observeDeviceSequenceState(layer, seq_idx, &state))
            return false;
        *kv_len = state.cached_tokens;

        if (state.cached_tokens == 0)
        {
            return true;
        }

        launch_linearize_kernel(entry,
                                state.implementation_head,
                                state.cached_tokens,
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
        return CUDARingKVCacheBase::sequenceState(local_layer, seq_idx);
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

        if (!desc.stream)
        {
            if (desc.token_count == 0)
            {
                KVCacheSequenceState state;
                return observeDeviceSequenceState(
                           local_layer, desc.seq_idx, &state) &&
                       desc.logical_token_start <= state.cached_tokens;
            }
            LOG_ERROR("[CUDARingKVCache::exportLogicalBlock] non-empty GPU logical export requires an explicit stream");
            return false;
        }

        cudaSetDevice(device_id_);
        cudaStream_t stream = static_cast<cudaStream_t>(desc.stream);
        const auto &entry = entries_[local_layer][desc.seq_idx];
        if (desc.payload_domain ==
            KVCacheLogicalBlockPayloadDomain::Device)
        {
            if (desc.token_count == 0)
                return true;
            if (!dst_k || !dst_v || !entry.d_K || !entry.d_V ||
                !d_head_params_ || !d_count_params_)
            {
                LOG_ERROR("[CUDARingKVCache::exportLogicalBlock] device-domain logical export storage unavailable");
                return false;
            }

            const int entry_index =
                local_layer * batch_size_ + desc.seq_idx;
            constexpr int threads = 256;
            const size_t element_count =
                static_cast<size_t>(desc.token_count) *
                static_cast<size_t>(kv_storage_dim_);
            const int blocks = std::max(
                1,
                std::min<int>(
                    65535,
                    static_cast<int>(
                        (element_count + threads - 1) / threads)));
            ring_logical_block_export_device_kernel<DataT>
                <<<blocks, threads, 0, stream>>>(
                    entry.d_K,
                    entry.d_V,
                    static_cast<DataT *>(dst_k),
                    static_cast<DataT *>(dst_v),
                    &d_head_params_[entry_index],
                    &d_count_params_[entry_index],
                    desc.logical_token_start,
                    desc.token_count,
                    max_seq_len_,
                    kv_storage_dim_);
            const cudaError_t launch_error = cudaGetLastError();
            if (launch_error != cudaSuccess)
            {
                LOG_ERROR("[CUDARingKVCache::exportLogicalBlock] device-domain gather launch failed: "
                          << cudaGetErrorString(launch_error));
                return false;
            }
            PerfStatsCollector::addCounter(
                "prefix_cache",
                "cuda_device_logical_kv_exports",
                1.0,
                "harvest",
                "cuda:" + std::to_string(device_id_),
                {{"tokens", std::to_string(desc.token_count)},
                 {"precision", std::to_string(static_cast<int>(Precision))}});
            return true;
        }

        KVCacheSequenceState state;
        if (!observeDeviceSequenceState(local_layer, desc.seq_idx, &state))
            return false;
        const int export_head = state.implementation_head;
        const int export_count = state.cached_tokens;

        if (export_count < 0 || export_count > max_seq_len_ ||
            export_head < 0 || export_head >= max_seq_len_ ||
            desc.logical_token_start > export_count ||
            desc.token_count > export_count - desc.logical_token_start)
        {
            LOG_ERROR("[CUDARingKVCache::exportLogicalBlock] logical export bounds rejected"
                      << " global_layer=" << desc.layer
                      << " local_layer=" << local_layer
                      << " seq_idx=" << desc.seq_idx
                      << " logical_start=" << desc.logical_token_start
                      << " token_count=" << desc.token_count
                      << " device_head=" << export_head
                      << " device_count=" << export_count
                      << " max_seq_len=" << max_seq_len_);
            return false;
        }
        if (desc.token_count == 0)
        {
            return true;
        }
        if (!dst_k || !dst_v || !entry.d_K || !entry.d_V)
        {
            LOG_ERROR("[CUDARingKVCache::exportLogicalBlock] logical export storage unavailable"
                      << " global_layer=" << desc.layer
                      << " local_layer=" << local_layer
                      << " seq_idx=" << desc.seq_idx
                      << " token_count=" << desc.token_count
                      << " dst_k=" << dst_k
                      << " dst_v=" << dst_v
                      << " entry_k=" << entry.d_K
                      << " entry_v=" << entry.d_V);
            return false;
        }

        const size_t row_bytes = static_cast<size_t>(kv_storage_dim_) * sizeof(DataT);
        int tail = export_head - export_count;
        tail %= max_seq_len_;
        if (tail < 0)
        {
            tail += max_seq_len_;
        }
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
        const size_t payload_bytes = static_cast<size_t>(desc.token_count) * row_bytes;
        if (!kv_cache_codec::canonicalizeFloatingZeros(dst_k, payload_bytes, Precision) ||
            !kv_cache_codec::canonicalizeFloatingZeros(dst_v, payload_bytes, Precision))
        {
            LOG_ERROR("[CUDARingKVCache::exportLogicalBlock] logical payload canonicalization failed");
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
        if (!desc.stream)
        {
            LOG_ERROR("[CUDARingKVCache::importLogicalBlock] GPU logical import requires an explicit stream");
            return false;
        }
        cudaStream_t stream = static_cast<cudaStream_t>(desc.stream);
        if (desc.payload_domain ==
            KVCacheLogicalBlockPayloadDomain::Device)
        {
            if (desc.token_count == 0)
            {
                if (desc.logical_token_start != 0)
                    return true;
                if (!setDeviceSequenceState(
                        local_layer,
                        desc.seq_idx,
                        0,
                        0,
                        stream))
                {
                    return false;
                }
                invalidateRoPEShadow(local_layer, desc.seq_idx);
                return true;
            }
            if (!src_k || !src_v || !entry.d_K || !entry.d_V ||
                !d_head_params_ || !d_count_params_)
            {
                LOG_ERROR("[CUDARingKVCache::importLogicalBlock] device-domain logical import storage unavailable");
                return false;
            }

            cudaSetDevice(device_id_);
            const int entry_index =
                local_layer * batch_size_ + desc.seq_idx;
            constexpr int threads = 256;
            const size_t element_count =
                static_cast<size_t>(desc.token_count) *
                static_cast<size_t>(kv_storage_dim_);
            const int blocks = std::max(
                1,
                std::min<int>(
                    65535,
                    static_cast<int>(
                        (element_count + threads - 1) / threads)));
            ring_logical_block_import_device_kernel<DataT>
                <<<blocks, threads, 0, stream>>>(
                    entry.d_K,
                    entry.d_V,
                    static_cast<const DataT *>(src_k),
                    static_cast<const DataT *>(src_v),
                    &d_head_params_[entry_index],
                    &d_count_params_[entry_index],
                    desc.logical_token_start,
                    desc.token_count,
                    max_seq_len_,
                    kv_storage_dim_);
            ring_logical_block_import_publish_kernel<<<1, 1, 0, stream>>>(
                &d_head_params_[entry_index],
                &d_count_params_[entry_index],
                desc.logical_token_start,
                desc.token_count,
                max_seq_len_);
            const cudaError_t launch_error = cudaGetLastError();
            if (launch_error != cudaSuccess)
            {
                LOG_ERROR("[CUDARingKVCache::importLogicalBlock] device-domain scatter launch failed: "
                          << cudaGetErrorString(launch_error));
                return false;
            }
            invalidateRoPEShadow(local_layer, desc.seq_idx);
            PerfStatsCollector::addCounter(
                "prefix_cache",
                "cuda_device_logical_kv_imports",
                1.0,
                "restore",
                "cuda:" + std::to_string(device_id_),
                {{"tokens", std::to_string(desc.token_count)},
                 {"precision", std::to_string(static_cast<int>(Precision))}});
            return true;
        }

        if (desc.token_count == 0)
        {
            if (desc.logical_token_start == 0)
            {
                if (!setDeviceSequenceState(
                        local_layer, desc.seq_idx, 0, 0, stream))
                    return false;
                invalidateRoPEShadow(local_layer, desc.seq_idx);
            }
            return true;
        }
        if (!src_k || !src_v || !entry.d_K || !entry.d_V)
        {
            return false;
        }
        /*
         * Host-domain input is an explicit RAM/SSD archive boundary. The
         * restore planner has already cleared the sequence and supplies blocks
         * in logical order, so re-reading canonical metadata here would add a
         * D2H synchronization merely to rediscover planner-owned geometry.
         * Payload and metadata writes remain ordered on the caller's stream.
         */
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

        const int new_count = desc.logical_token_start + desc.token_count;
        const int new_head = new_count % max_seq_len_;
        if (!setDeviceSequenceState(
                local_layer, desc.seq_idx, new_head, new_count, stream))
            return false;

        invalidateRoPEShadow(local_layer, desc.seq_idx);
        return true;
    }

    template <ActivationPrecision Precision>
    bool CUDARingKVCache<Precision>::truncateSequence(int seq_idx, int cached_tokens, void *stream)
    {
        return CUDARingKVCacheBase::truncateSequence(
            seq_idx,
            cached_tokens,
            stream);
    }

    template <ActivationPrecision Precision>
    void CUDARingKVCache<Precision>::evict_oldest(int layer, int seq_idx, int num_tokens)
    {
        if (layer < 0 || layer >= n_layers_ ||
            seq_idx < 0 || seq_idx >= batch_size_ ||
            num_tokens < 0)
        {
            return;
        }

        cudaStream_t stream = static_cast<cudaStream_t>(
            GPUDeviceContextPool::instance()
                .getNvidiaContext(device_id_)
                .defaultStream());
        if (!evictOldestDeviceSequenceState(
                layer,
                seq_idx,
                num_tokens,
                stream))
        {
            LOG_ERROR("[CUDARingKVCache::evict_oldest] Device metadata eviction launch failed");
            return;
        }
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

        int actual_max_kv_len = 0;

        for (int seq = 0; seq < num_seqs; ++seq)
        {
            KVCacheSequenceState state;
            if (!observeDeviceSequenceState(layer, seq, &state))
                return -1;
            kv_lens[seq] = state.cached_tokens;
            actual_max_kv_len = std::max(actual_max_kv_len, kv_lens[seq]);
        }

        if (actual_max_kv_len == 0)
        {
            return 0;
        }

        // Use provided max_kv_len or actual
        int out_max_kv_len = (max_kv_len > 0) ? max_kv_len : actual_max_kv_len;

        if (!stream || !d_k_out || !d_v_out ||
            !d_batched_k_entry_table_ || !d_batched_v_entry_table_ ||
            !d_head_params_ || !d_count_params_ ||
            !batched_pointer_tables_ready_)
        {
            return -1;
        }

        const int entry_offset = layer * batch_size_;
        const dim3 block(256);
        const dim3 grid(
            static_cast<unsigned int>(out_max_kv_len),
            static_cast<unsigned int>((kv_storage_dim_ + 255) / 256),
            static_cast<unsigned int>(num_seqs));
        ring_gather_batched_device_state_kernel<DataT, true><<<grid, block, 0, stream>>>(
            static_cast<DataT *>(d_k_out),
            static_cast<DataT *>(d_v_out),
            d_batched_k_entry_table_,
            d_batched_v_entry_table_,
            d_head_params_,
            d_count_params_,
            entry_offset,
            num_seqs,
            out_max_kv_len,
            max_seq_len_,
            kv_storage_dim_);
        if (cudaGetLastError() != cudaSuccess)
            return -1;

        return actual_max_kv_len;
    }

    template <ActivationPrecision Precision>
    bool CUDARingKVCache<Precision>::get_kv_batched_device_view(
        int layer,
        int first_seq_idx,
        int request_count,
        ITensor **out_k,
        ITensor **out_v,
        void *gpu_stream)
    {
        if (out_k)
            *out_k = nullptr;
        if (out_v)
            *out_v = nullptr;

        if (layer < 0 || layer >= n_layers_ ||
            first_seq_idx < 0 || request_count <= 0 ||
            first_seq_idx + request_count > batch_size_ ||
            !gpu_stream || !workspace_ || !workspace_->isAllocated() ||
            !d_head_params_ || !d_count_params_ ||
            !batched_pointer_tables_ready_)
        {
            LOG_ERROR("[CUDARingKVCache::get_kv_batched_device_view] Invalid resident gather contract"
                      << " layer=" << layer
                      << " first_seq=" << first_seq_idx
                      << " requests=" << request_count
                      << " batch_capacity=" << batch_size_
                      << " max_seq_len=" << max_seq_len_
                      << " stream=" << gpu_stream
                      << " workspace=" << (workspace_ ? "bound" : "missing")
                      << " pointer_tables="
                      << (batched_pointer_tables_ready_ ? "ready" : "missing"));
            return false;
        }

        const size_t rows =
            static_cast<size_t>(request_count) *
            static_cast<size_t>(max_seq_len_);
        const size_t required_bytes =
            rows * static_cast<size_t>(kv_storage_dim_) * sizeof(DataT);
        if (!ensureConvScratch(required_bytes) ||
            !conv_scratch_k_ || !conv_scratch_v_)
        {
            LOG_ERROR("[CUDARingKVCache::get_kv_batched_device_view] Native gather scratch is unavailable"
                      << " required_bytes=" << required_bytes);
            return false;
        }

        if (!d_batched_k_entry_table_ || !d_batched_v_entry_table_)
        {
            LOG_ERROR("[CUDARingKVCache::get_kv_batched_device_view] Missing cache-owned immutable entry pointer tables");
            return false;
        }

        const int entry_offset = layer * batch_size_ + first_seq_idx;
        const dim3 block(256);
        constexpr unsigned int resident_token_blocks = 32;
        const dim3 grid(
            std::min(
                static_cast<unsigned int>(max_seq_len_),
                resident_token_blocks),
            static_cast<unsigned int>((kv_storage_dim_ + 255) / 256),
            static_cast<unsigned int>(request_count));
        auto stream = static_cast<cudaStream_t>(gpu_stream);
        ring_gather_batched_device_state_kernel<DataT, false><<<grid, block, 0, stream>>>(
            static_cast<DataT *>(conv_scratch_k_),
            static_cast<DataT *>(conv_scratch_v_),
            d_batched_k_entry_table_,
            d_batched_v_entry_table_,
            d_head_params_,
            d_count_params_,
            entry_offset,
            request_count,
            max_seq_len_,
            max_seq_len_,
            kv_storage_dim_);
        const cudaError_t launch_error = cudaGetLastError();
        if (launch_error != cudaSuccess)
        {
            LOG_ERROR("[CUDARingKVCache::get_kv_batched_device_view] Gather launch failed: "
                      << cudaGetErrorString(launch_error));
            return false;
        }

        const TensorType tensor_type = []() constexpr
        {
            if constexpr (Precision == ActivationPrecision::FP32)
                return TensorType::FP32;
            if constexpr (Precision == ActivationPrecision::FP16)
                return TensorType::FP16;
            if constexpr (Precision == ActivationPrecision::BF16)
                return TensorType::BF16;
            return TensorType::Q8_1;
        }();

        if (!batched_k_view_ ||
            batched_k_view_->gpu_data_ptr() != conv_scratch_k_ ||
            batched_k_view_->shape().size() < 2 ||
            batched_k_view_->shape()[0] != rows)
        {
            batched_k_view_ = std::make_unique<GpuTensorView>(
                conv_scratch_k_, rows, static_cast<size_t>(kv_storage_dim_),
                tensor_type, device_id_);
        }
        if (!batched_v_view_ ||
            batched_v_view_->gpu_data_ptr() != conv_scratch_v_ ||
            batched_v_view_->shape().size() < 2 ||
            batched_v_view_->shape()[0] != rows)
        {
            batched_v_view_ = std::make_unique<GpuTensorView>(
                conv_scratch_v_, rows, static_cast<size_t>(kv_storage_dim_),
                tensor_type, device_id_);
        }

        if (out_k)
            *out_k = batched_k_view_.get();
        if (out_v)
            *out_v = batched_v_view_.get();
        return true;
    }

    template <ActivationPrecision Precision>
    bool CUDARingKVCache<Precision>::get_kv_batched_converted_device_view(
        int layer,
        int first_seq_idx,
        int request_count,
        ActivationPrecision target,
        ITensor **out_k,
        ITensor **out_v,
        const KVReadParams &read)
    {
        if (out_k)
            *out_k = nullptr;
        if (out_v)
            *out_v = nullptr;

        const int requested_heads =
            read.n_kv_heads > 0 ? read.n_kv_heads : local_n_kv_heads_;
        const int requested_head_dim =
            read.head_dim > 0 ? read.head_dim : head_dim_;
        const int effective_rope_dim =
            read.rope_dim > 0 ? read.rope_dim : requested_head_dim;
        if (layer < 0 || layer >= n_layers_ || first_seq_idx < 0 ||
            request_count <= 0 ||
            first_seq_idx > batch_size_ - request_count ||
            target != ActivationPrecision::FP16 || !read.gpu_stream ||
            requested_heads != local_n_kv_heads_ ||
            requested_head_dim != head_dim_ ||
            effective_rope_dim <= 0 || effective_rope_dim > head_dim_ ||
            (effective_rope_dim % 2) != 0 ||
            !d_head_params_ || !d_count_params_ ||
            !batched_pointer_tables_ready_ ||
            !d_batched_k_entry_table_ || !d_batched_v_entry_table_)
        {
            LOG_ERROR("[CUDARingKVCache::get_kv_batched_converted_device_view] Invalid grouped conversion contract"
                      << " layer=" << layer
                      << " first_seq=" << first_seq_idx
                      << " requests=" << request_count
                      << " target=" << activationPrecisionToString(target)
                      << " heads=" << requested_heads
                      << " head_dim=" << requested_head_dim
                      << " rope_dim=" << effective_rope_dim
                      << " stream=" << read.gpu_stream);
            return false;
        }

        const size_t rows =
            static_cast<size_t>(request_count) * max_seq_len_;
        const size_t required_bytes =
            rows * static_cast<size_t>(kv_dim_) * sizeof(__half);
        if (!ensureConvScratch(required_bytes) ||
            !conv_scratch_k_ || !conv_scratch_v_)
        {
            LOG_ERROR("[CUDARingKVCache::get_kv_batched_converted_device_view] FP16 grouped workspace is unavailable"
                      << " required_bytes=" << required_bytes);
            return false;
        }

        auto *k_output = static_cast<__half *>(conv_scratch_k_);
        auto *v_output = static_cast<__half *>(conv_scratch_v_);
        auto stream = static_cast<cudaStream_t>(read.gpu_stream);
        const int entry_offset = layer * batch_size_ + first_seq_idx;
        const dim3 gather_block(256);
        constexpr unsigned int resident_token_blocks = 32;
        const dim3 gather_grid(
            std::min(
                static_cast<unsigned int>(max_seq_len_),
                resident_token_blocks),
            static_cast<unsigned int>((kv_dim_ + 255) / 256),
            static_cast<unsigned int>(request_count));
        ring_gather_batched_device_state_fp16_kernel<DataT, true>
            <<<gather_grid, gather_block, 0, stream>>>(
                k_output,
                v_output,
                d_batched_k_entry_table_,
                d_batched_v_entry_table_,
                d_head_params_,
                d_count_params_,
                entry_offset,
                request_count,
                max_seq_len_,
                max_seq_len_,
                kv_dim_);
        cudaError_t launch_error = cudaGetLastError();
        if (launch_error != cudaSuccess)
        {
            LOG_ERROR("[CUDARingKVCache::get_kv_batched_converted_device_view] Gather/convert launch failed: "
                      << cudaGetErrorString(launch_error));
            return false;
        }

        if (read.rope_theta > 0.0f)
        {
            bool rope_ok = false;
            if constexpr (Precision == ActivationPrecision::FP32)
            {
                rope_ok = cuda_rope_apply_batched_fp32_ring_to_fp16_device_state(
                    k_output,
                    reinterpret_cast<const float *const *>(
                    d_batched_k_entry_table_),
                    d_head_params_, d_count_params_, entry_offset,
                    request_count, max_seq_len_, max_seq_len_,
                    local_n_kv_heads_, head_dim_, read.rope_theta,
                    read.position_start, effective_rope_dim, stream);
            }
            else
            {
                rope_ok = cuda_rope_apply_batched_fp16_device_state(
                    k_output, d_count_params_, entry_offset, request_count,
                    max_seq_len_, max_seq_len_, local_n_kv_heads_, head_dim_,
                    read.rope_theta, read.position_start, effective_rope_dim,
                    stream);
            }
            if (!rope_ok)
            {
                LOG_ERROR("[CUDARingKVCache::get_kv_batched_converted_device_view] Grouped RoPE launch failed: "
                          << cudaGetErrorString(cudaGetLastError()));
                return false;
            }
        }

        if (!converted_batched_k_view_ ||
            converted_batched_k_view_->gpu_data_ptr() != k_output ||
            converted_batched_k_view_->shape().size() < 2 ||
            converted_batched_k_view_->shape()[0] != rows)
        {
            converted_batched_k_view_ = std::make_unique<GpuTensorView>(
                k_output,
                rows,
                static_cast<size_t>(kv_dim_),
                TensorType::FP16,
                device_id_);
        }
        if (!converted_batched_v_view_ ||
            converted_batched_v_view_->gpu_data_ptr() != v_output ||
            converted_batched_v_view_->shape().size() < 2 ||
            converted_batched_v_view_->shape()[0] != rows)
        {
            converted_batched_v_view_ = std::make_unique<GpuTensorView>(
                v_output,
                rows,
                static_cast<size_t>(kv_dim_),
                TensorType::FP16,
                device_id_);
        }
        if (out_k)
            *out_k = converted_batched_k_view_.get();
        if (out_v)
            *out_v = converted_batched_v_view_.get();
        return true;
    }

    // =========================================================================
    // IWorkspaceConsumer Implementation
    // =========================================================================

    template <ActivationPrecision Precision>
    WorkspaceRequirements CUDARingKVCache<Precision>::getWorkspaceRequirements(
        int m, int n, int k) const
    {
        /*
         * New callers pass m as the active graph bucket and n as the request
         * batch. That bucket is not a valid upper bound for resident KV reads:
         * a later prefill chunk can contain 256 new rows while attention gathers
         * 768 or more cached rows. Conversion scratch therefore follows the
         * cache's configured sequence horizon, never merely the current chunk.
         *
         * Legacy one-argument callers use m as a batch hint. Preserve that API,
         * while also reserving the cache's complete configured batch capacity so
         * a smaller first graph cannot under-size a later request batch.
         */
        (void)k;

        const bool has_token_hint = n > 0;
        const int actual_batch_size = has_token_hint ? n : ((m > 0) ? m : batch_size_);
        const int scratch_tokens = has_token_hint
                                       ? std::max(m, max_seq_len_)
                                       : max_seq_len_;
        const int bounded_batch_size =
            std::max(std::max(1, actual_batch_size), batch_size_);
        const int bounded_scratch_tokens = std::max(1, scratch_tokens);
        const size_t batched_scratch_tokens =
            static_cast<size_t>(bounded_scratch_tokens) *
            static_cast<size_t>(bounded_batch_size);
        const size_t fp32_scratch_bytes =
            batched_scratch_tokens * static_cast<size_t>(kv_dim_) * sizeof(float);
        const size_t fp16_scratch_bytes =
            batched_scratch_tokens * static_cast<size_t>(kv_dim_) * sizeof(uint16_t);
        const size_t native_scratch_bytes =
            batched_scratch_tokens * static_cast<size_t>(kv_storage_dim_) * sizeof(DataT);
        const size_t conversion_scratch_bytes =
            std::max(fp32_scratch_bytes, std::max(fp16_scratch_bytes, native_scratch_bytes));

        WorkspaceRequirements reqs;

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
                  << " CONV_SCRATCH(each)=" << conversion_scratch_bytes);

        return reqs;
    }

    template <ActivationPrecision Precision>
    void CUDARingKVCache<Precision>::bindWorkspace(DeviceWorkspaceManager *workspace)
    {
        const size_t entry_count =
            static_cast<size_t>(n_layers_) * static_cast<size_t>(batch_size_);
        if (!batched_pointer_tables_ready_ ||
            (entry_count > 0 &&
             (!d_batched_k_entry_table_ || !d_batched_v_entry_table_)))
        {
            throw std::logic_error(
                "[CUDARingKVCache] Cannot bind workspace before immutable entry topology is published");
        }

        // WorkspaceAllocator intentionally re-presents the current manager on
        // every graph execution. Preserve all wrappers and scratch ownership
        // when the identity is unchanged.
        if (workspace_ == workspace)
            return;

        if (isGraphCaptureActive())
        {
            throw std::runtime_error(
                "[CUDARingKVCache] Workspace ownership cannot change during CUDA graph capture");
        }
        if (workspace && !workspace->isAllocated())
        {
            throw std::invalid_argument(
                "[CUDARingKVCache] A bound workspace must be fully allocated");
        }

        void *new_scratch_k = nullptr;
        void *new_scratch_v = nullptr;
        size_t new_scratch_capacity = 0;
        if (workspace)
        {
            new_scratch_k =
                workspace->getBuffer(KVCacheWorkspaceBuffers::CONV_SCRATCH_K);
            new_scratch_v =
                workspace->getBuffer(KVCacheWorkspaceBuffers::CONV_SCRATCH_V);
            const size_t k_capacity =
                workspace->getBufferSize(KVCacheWorkspaceBuffers::CONV_SCRATCH_K);
            const size_t v_capacity =
                workspace->getBufferSize(KVCacheWorkspaceBuffers::CONV_SCRATCH_V);
            if (!new_scratch_k || !new_scratch_v ||
                k_capacity == 0 || v_capacity == 0)
            {
                throw std::runtime_error(
                    "[CUDARingKVCache] Bound workspace lacks mandatory grouped conversion scratch");
            }
            new_scratch_capacity = std::min(k_capacity, v_capacity);
        }

        freeConvScratch();
        workspace_ = workspace;
        conv_scratch_k_ = new_scratch_k;
        conv_scratch_v_ = new_scratch_v;
        conv_scratch_capacity_ = new_scratch_capacity;
        conv_scratch_workspace_backed_ = workspace != nullptr;
        batched_k_view_.reset();
        batched_v_view_.reset();
        converted_batched_k_view_.reset();
        converted_batched_v_view_.reset();

        LOG_DEBUG("[CUDARingKVCache] Workspace bound: "
                  << (workspace ? "yes" : "no"));
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
