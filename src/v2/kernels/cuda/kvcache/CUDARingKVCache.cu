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
#include "../../../execution/local_execution/device/DeviceWorkspaceManager.h"
#include "../../../execution/local_execution/device/WorkspaceDescriptor.h"
#include "../../../utils/Logger.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstring>

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
            d_K_out[out_offset] = T(0);
            d_V_out[out_offset] = T(0);
            return;
        }

        // Calculate source position with wrap-around
        int tail = tails[seq_idx];
        int src_pos = (tail + token_idx) % max_seq_len;
        int src_offset = src_pos * kv_dim + elem_idx;

        d_K_out[out_offset] = d_K_caches[seq_idx][src_offset];
        d_V_out[out_offset] = d_V_caches[seq_idx][src_offset];
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

    // =========================================================================
    // CUDARingKVCache Implementation
    // =========================================================================

    template <ActivationPrecision Precision>
    CUDARingKVCache<Precision>::CUDARingKVCache(
        int n_layers, int batch_size, int max_seq_len,
        int n_kv_heads, int head_dim, int device_id)
        : n_layers_(n_layers), batch_size_(batch_size), max_seq_len_(max_seq_len),
          n_kv_heads_(n_kv_heads), local_n_kv_heads_(n_kv_heads), kv_head_start_(0),
          head_dim_(head_dim), kv_dim_(n_kv_heads * head_dim), device_id_(device_id),
          is_sharded_(false)
    {
        LOG_INFO("[CUDARingKVCache] Creating cache: "
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

        LOG_INFO("[CUDARingKVCache] Allocated "
                 << (n_layers_ * batch_size_ * 4 * max_seq_len_ * kv_dim_ * sizeof(DataT)) / (1024 * 1024)
                 << " MB total (including scratch)");
    }

    template <ActivationPrecision Precision>
    CUDARingKVCache<Precision>::CUDARingKVCache(
        int n_layers, int batch_size, int max_seq_len,
        int n_kv_heads, int local_n_kv_heads, int kv_head_start,
        int head_dim, int device_id)
        : n_layers_(n_layers), batch_size_(batch_size), max_seq_len_(max_seq_len),
          n_kv_heads_(n_kv_heads), local_n_kv_heads_(local_n_kv_heads), kv_head_start_(kv_head_start),
          head_dim_(head_dim), kv_dim_(local_n_kv_heads * head_dim), device_id_(device_id),
          is_sharded_(local_n_kv_heads != n_kv_heads)
    {
        LOG_INFO("[CUDARingKVCache] Creating sharded cache: "
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

        LOG_INFO("[CUDARingKVCache] Allocated "
                 << (n_layers_ * batch_size_ * 4 * max_seq_len_ * kv_dim_ * sizeof(DataT)) / (1024 * 1024)
                 << " MB total (including scratch)");
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

        for (auto &layer_entries : entries_)
        {
            for (auto &entry : layer_entries)
            {
                free_entry(entry);
            }
        }
    }

    template <ActivationPrecision Precision>
    void CUDARingKVCache<Precision>::allocate_entry(EntryT &entry)
    {
        size_t buffer_size = max_seq_len_ * kv_dim_ * sizeof(DataT);

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

    template <ActivationPrecision Precision>
    int CUDARingKVCache<Precision>::get_cached_tokens(int layer, int seq_idx) const
    {
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
        {
            return 0;
        }
        return entries_[layer][seq_idx].count;
    }

    template <ActivationPrecision Precision>
    int CUDARingKVCache<Precision>::get_head_position(int layer, int seq_idx) const
    {
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
        {
            return 0;
        }
        return entries_[layer][seq_idx].head;
    }

    template <ActivationPrecision Precision>
    bool CUDARingKVCache<Precision>::is_wrapped(int layer, int seq_idx) const
    {
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
        {
            return false;
        }
        return entries_[layer][seq_idx].is_wrapped(max_seq_len_);
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

        // Check if we would exceed capacity (ring buffer overwrites oldest)
        if (entry.count + num_tokens > max_seq_len_)
        {
            // Auto-evict oldest tokens to make room
            int to_evict = entry.count + num_tokens - max_seq_len_;
            entry.count -= to_evict;
            total_evicted_ += to_evict;
            LOG_DEBUG("[CUDARingKVCache::append] Auto-evicted " << to_evict << " tokens");
        }

        // Launch append kernel
        launch_append_kernel(entry, d_k, d_v, num_tokens, stream);

        // Update ring buffer state
        entry.head = (entry.head + num_tokens) % max_seq_len_;
        entry.count += num_tokens;
        entry.scratch_valid = false; // Scratch is stale after append

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
                entry.head, max_seq_len_, kv_dim_, num_tokens, stream);
        }
        else if constexpr (Precision == ActivationPrecision::FP16)
        {
            cuda_ring_append_fp16(
                entry.d_K, entry.d_V, d_k, d_v,
                entry.head, max_seq_len_, kv_dim_, num_tokens, stream);
        }
        else if constexpr (Precision == ActivationPrecision::BF16)
        {
            cuda_ring_append_bf16(
                entry.d_K, entry.d_V, d_k, d_v,
                entry.head, max_seq_len_, kv_dim_, num_tokens, stream);
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
            *d_k_out = entry.d_K + tail * kv_dim_;
            *d_v_out = entry.d_V + tail * kv_dim_;
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
                tail, entry.count, max_seq_len_, kv_dim_, stream);
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
        dim3 grid(max_kv_len, (kv_dim_ + 255) / 256, num_seqs);

        ring_gather_batched_kernel<DataT><<<grid, block, 0, stream>>>(
            d_k_out, d_v_out,
            d_k_caches, d_v_caches,
            d_tails, d_counts,
            num_seqs, max_kv_len, max_seq_len_, kv_dim_);

        // Free temporary device arrays only if we allocated them
        if (!using_workspace)
        {
            cudaFreeAsync(d_k_caches, stream);
            cudaFreeAsync(d_v_caches, stream);
            cudaFreeAsync(d_tails, stream);
            cudaFreeAsync(d_counts, stream);
        }
    }

    template <ActivationPrecision Precision>
    void CUDARingKVCache<Precision>::clear()
    {
        for (int layer = 0; layer < n_layers_; ++layer)
        {
            clear_layer(layer);
        }
    }

    template <ActivationPrecision Precision>
    void CUDARingKVCache<Precision>::clear_sequence(int layer, int seq_idx)
    {
        if (layer < 0 || layer >= n_layers_ || seq_idx < 0 || seq_idx >= batch_size_)
        {
            return;
        }

        EntryT &entry = entries_[layer][seq_idx];
        entry.head = 0;
        entry.count = 0;
        entry.scratch_valid = false;
    }

    template <ActivationPrecision Precision>
    void CUDARingKVCache<Precision>::clear_layer(int layer)
    {
        if (layer < 0 || layer >= n_layers_)
        {
            return;
        }

        for (int seq = 0; seq < batch_size_; ++seq)
        {
            clear_sequence(layer, seq);
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

        default:
            LOG_ERROR("[createShardedCUDARingKVCache] Unsupported precision: "
                      << static_cast<int>(precision));
            return nullptr;
        }
    }

} // namespace llaminar2
