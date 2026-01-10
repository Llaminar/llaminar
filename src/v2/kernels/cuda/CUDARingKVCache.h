/**
 * @file CUDARingKVCache.h
 * @brief CUDA Ring Buffer KV Cache with O(1) eviction
 * @author David Sanftenberg
 * @date January 2026
 *
 * Ring buffer implementation for KV cache that enables:
 * - O(1) append (write to head position)
 * - O(1) eviction (pointer arithmetic only, no memory copies)
 * - Efficient sliding window attention
 * - Template support for FP32, BF16, FP16 precisions
 *
 * Ring Buffer Layout (max_seq_len=8, count=5):
 *
 *   Position:  [0] [1] [2] [3] [4] [5] [6] [7]
 *   Data:      [T3][T4][T5][ ][ ][ ][T1][T2]
 *                      ^           ^
 *                    head        tail
 *                  (write)     (oldest)
 *
 *   - head: next write position
 *   - count: number of valid tokens
 *   - tail = (head - count + max_seq_len) % max_seq_len
 *
 * Contiguous Optimization:
 *   When tail < head (no wrap-around), attention can read directly
 *   from the buffer without linearization. Only wrapped buffers
 *   require a linearize kernel call.
 */

#pragma once

// Minimal includes - avoid MPI headers for nvcc compatibility
#include "../IKVCache.h"                   // Unified KVCache interface
#include "../../execution/RuntimeConfig.h" // For ActivationPrecision
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <vector>
#include <memory>

namespace llaminar2
{

    // =========================================================================
    // ICUDARingKVCache Interface
    // =========================================================================

    /**
     * @brief Abstract interface for CUDA ring buffer KV cache
     *
     * Enables polymorphic use when precision is determined at runtime.
     * Inherits from IKVCache to provide unified CPU/GPU interface.
     *
     * NOTE: CUDA KV cache uses device pointers internally, so the ITensor-based
     * get_kv() returns CUDATensor wrappers around the device buffers.
     */
    class ICUDARingKVCache : public IKVCache
    {
    public:
        virtual ~ICUDARingKVCache() = default;

        // =====================================================================
        // Metadata
        // =====================================================================

        virtual ActivationPrecision precision() const = 0;
        int n_layers() const override { return num_layers(); }
        virtual int num_layers() const = 0;
        virtual int batch_size() const = 0;
        int max_seq_len() const override = 0;
        virtual int n_kv_heads() const = 0;
        virtual int head_dim() const = 0;
        virtual int kv_dim() const = 0; ///< n_kv_heads * head_dim

        // =====================================================================
        // Per-Sequence State
        // =====================================================================

        /**
         * @brief Get number of cached tokens for a sequence
         */
        int get_cached_tokens(int layer, int seq_idx = 0) const override = 0;

        /**
         * @brief Get head (write) position for a sequence
         */
        virtual int get_head_position(int layer, int seq_idx = 0) const = 0;

        /**
         * @brief Check if buffer is wrapped (tail > head)
         *
         * If not wrapped, attention can read directly without linearization.
         */
        virtual bool is_wrapped(int layer, int seq_idx = 0) const = 0;

        // =====================================================================
        // Append Operations
        // =====================================================================

        /**
         * @brief Append K/V tokens to cache (type-erased)
         *
         * @param layer Layer index
         * @param seq_idx Sequence index
         * @param d_k Device pointer to K data [num_tokens, kv_dim]
         * @param d_v Device pointer to V data [num_tokens, kv_dim]
         * @param num_tokens Number of tokens to append
         * @param stream CUDA stream (0 for default)
         * @return true on success, false if would exceed capacity
         */
        virtual bool append(int layer, int seq_idx,
                            const void *d_k, const void *d_v,
                            int num_tokens, cudaStream_t stream = 0) = 0;

        // Convenience for single-sequence mode
        bool append(int layer, const void *d_k, const void *d_v,
                    int num_tokens, cudaStream_t stream = 0)
        {
            return append(layer, 0, d_k, d_v, num_tokens, stream);
        }

        // =====================================================================
        // Read Operations
        // =====================================================================

        /**
         * @brief Get K/V pointers for attention computation
         *
         * If buffer is contiguous (not wrapped), returns direct pointers.
         * If wrapped, returns pointers to linearized scratch buffers.
         *
         * @param layer Layer index
         * @param seq_idx Sequence index
         * @param d_k_out Output: pointer to K data
         * @param d_v_out Output: pointer to V data
         * @param kv_len Output: number of valid tokens
         * @param stream CUDA stream for potential linearization
         * @return true on success
         */
        virtual bool get_kv_for_attention(int layer, int seq_idx,
                                          const void **d_k_out, const void **d_v_out,
                                          int *kv_len, cudaStream_t stream = 0) = 0;

        // Convenience for single-sequence mode
        bool get_kv_for_attention(int layer,
                                  const void **d_k_out, const void **d_v_out,
                                  int *kv_len, cudaStream_t stream = 0)
        {
            return get_kv_for_attention(layer, 0, d_k_out, d_v_out, kv_len, stream);
        }

        /**
         * @brief Force linearization to external buffer
         *
         * Always copies to provided buffer, regardless of wrap state.
         * Use when attention kernel requires specific memory layout.
         *
         * @param layer Layer index
         * @param seq_idx Sequence index
         * @param d_k_out Output K buffer [kv_len, kv_dim]
         * @param d_v_out Output V buffer [kv_len, kv_dim]
         * @param kv_len Output: number of tokens copied
         * @param stream CUDA stream
         * @return true on success
         */
        virtual bool linearize_to(int layer, int seq_idx,
                                  void *d_k_out, void *d_v_out,
                                  int *kv_len, cudaStream_t stream = 0) = 0;

        // =====================================================================
        // Eviction (O(1) - pointer arithmetic only)
        // =====================================================================

        /**
         * @brief Evict oldest tokens from a sequence
         *
         * O(1) operation - only updates tail pointer (via count decrement).
         * No memory copies or kernel launches.
         *
         * @param layer Layer index
         * @param seq_idx Sequence index
         * @param num_tokens Number of oldest tokens to evict
         */
        virtual void evict_oldest(int layer, int seq_idx, int num_tokens) = 0;

        // Convenience for single-sequence mode
        void evict_oldest(int layer, int num_tokens)
        {
            evict_oldest(layer, 0, num_tokens);
        }

        /**
         * @brief Evict oldest tokens from all sequences in a layer
         */
        virtual void evict_oldest_layer(int layer, int num_tokens) = 0;

        // =====================================================================
        // Batched Operations
        // =====================================================================

        /**
         * @brief Gather K/V from multiple sequences for batched attention
         *
         * Copies K/V from sequences [0..num_seqs-1] into contiguous output
         * tensors with padding to max_kv_len.
         *
         * Output layout: [num_seqs * max_kv_len, kv_dim]
         *
         * @param layer Layer index
         * @param num_seqs Number of sequences to gather
         * @param d_k_out Output K tensor
         * @param d_v_out Output V tensor
         * @param kv_lens Output: per-sequence kv_lens (size = num_seqs)
         * @param max_kv_len Maximum kv_len (for padding)
         * @param stream CUDA stream
         * @return Actual max kv_len found, or -1 on error
         */
        virtual int gather_kv_batched(int layer, int num_seqs,
                                      void *d_k_out, void *d_v_out,
                                      int *kv_lens, int max_kv_len,
                                      cudaStream_t stream = 0) = 0;

        // =====================================================================
        // Cache Management
        // =====================================================================

        /**
         * @brief Clear all cached tokens
         */
        virtual void clear() = 0;

        /**
         * @brief Clear a specific sequence
         */
        virtual void clear_sequence(int layer, int seq_idx) = 0;

        /**
         * @brief Clear all sequences in a layer
         */
        virtual void clear_layer(int layer) = 0;

        // =====================================================================
        // Statistics
        // =====================================================================

        /**
         * @brief Get total evicted token count (across all sequences)
         */
        virtual int get_total_evicted() const = 0;

        /**
         * @brief Reset eviction counter
         */
        virtual void reset_eviction_counter() = 0;

        /**
         * @brief Get number of linearizations performed (for profiling)
         */
        virtual int get_linearization_count() const = 0;

        /**
         * @brief Reset linearization counter
         */
        virtual void reset_linearization_counter() = 0;

        // =====================================================================
        // IKVCache Interface Implementation
        // =====================================================================
        // CUDA KV cache primarily uses device pointer APIs (get_kv_for_attention,
        // append with void*). These IKVCache methods provide the ITensor-based
        // interface for compatibility with GraphOrchestrator.

        /**
         * @brief Get K/V as ITensor pointers
         *
         * NOTE: For CUDA caches, this creates temporary CUDATensor wrappers
         * around device buffers. Prefer get_kv_for_attention() for performance.
         *
         * @return false - not yet implemented, use get_kv_for_attention() directly
         */
        bool get_kv(int layer, int seq_idx,
                    ITensor **out_k, ITensor **out_v,
                    int *out_kv_len = nullptr) override
        {
            // TODO: Implement with CUDATensor wrappers if needed
            // For now, CUDA attention kernels use get_kv_for_attention() directly
            (void)layer;
            (void)seq_idx;
            (void)out_k;
            (void)out_v;
            (void)out_kv_len;
            return false;
        }

        bool get_kv(int layer, int seq_idx,
                    const ITensor **out_k, const ITensor **out_v,
                    int *out_kv_len = nullptr) const override
        {
            // TODO: Implement with CUDATensor wrappers if needed
            (void)layer;
            (void)seq_idx;
            (void)out_k;
            (void)out_v;
            (void)out_kv_len;
            return false;
        }

        /**
         * @brief Append K/V from ITensor pointers
         *
         * Extracts device pointers from tensors and delegates to the
         * void* append method. Tensors must have GPU data available
         * (call ensureOnDevice() before if needed).
         *
         * @param layer Layer index
         * @param seq_idx Sequence index
         * @param K Key tensor with GPU data
         * @param V Value tensor with GPU data
         * @param num_tokens Number of tokens to append
         * @return true on success, false if tensors lack GPU data or append fails
         *
         * @note Implementation in CUDARingKVCacheTensorAdapter.cpp to avoid
         *       including heavy tensor headers in CUDA code.
         */
        bool append(int layer, int seq_idx,
                    const ITensor *K, const ITensor *V,
                    int num_tokens) override;

        // Bring in convenience overloads from IKVCache
        using IKVCache::append;
        using IKVCache::clear_sequence;
        using IKVCache::get_kv;
    };

    // =========================================================================
    // Type Mapping
    // =========================================================================

    namespace detail
    {
        /**
         * @brief Map ActivationPrecision to CUDA data type
         */
        template <ActivationPrecision P>
        struct CUDAKVCacheType;

        template <>
        struct CUDAKVCacheType<ActivationPrecision::FP32>
        {
            using Type = float;
            static constexpr size_t element_size = sizeof(float);
        };

        template <>
        struct CUDAKVCacheType<ActivationPrecision::FP16>
        {
            using Type = __half;
            static constexpr size_t element_size = sizeof(__half);
        };

        template <>
        struct CUDAKVCacheType<ActivationPrecision::BF16>
        {
            using Type = __nv_bfloat16;
            static constexpr size_t element_size = sizeof(__nv_bfloat16);
        };
    } // namespace detail

    // =========================================================================
    // CUDARingKVEntry
    // =========================================================================

    /**
     * @brief Per-layer, per-sequence ring buffer entry
     */
    template <ActivationPrecision Precision>
    struct CUDARingKVEntry
    {
        using DataT = typename detail::CUDAKVCacheType<Precision>::Type;

        DataT *d_K = nullptr; ///< Device memory: [max_seq_len, kv_dim]
        DataT *d_V = nullptr; ///< Device memory: [max_seq_len, kv_dim]

        // Ring buffer state
        int head = 0;  ///< Next write position
        int count = 0; ///< Number of valid tokens

        // Per-sequence scratch buffers for linearization
        DataT *d_K_scratch = nullptr; ///< Linearized K when wrapped
        DataT *d_V_scratch = nullptr; ///< Linearized V when wrapped
        bool scratch_valid = false;   ///< True if scratch contains current linearized data

        /**
         * @brief Get tail (oldest token) position
         */
        int tail(int max_seq_len) const
        {
            return (head - count + max_seq_len) % max_seq_len;
        }

        /**
         * @brief Check if buffer is wrapped (tail > head or wraps around)
         */
        bool is_wrapped(int max_seq_len) const
        {
            if (count == 0)
                return false;
            int t = tail(max_seq_len);
            // Wrapped if tail >= head (meaning data wraps around end of buffer)
            return t >= head && count > 0;
        }
    };

    // =========================================================================
    // CUDARingKVCache Template Class
    // =========================================================================

    /**
     * @brief CUDA Ring Buffer KV Cache implementation
     *
     * Template parameter Precision determines the data type:
     * - FP32: 32-bit float
     * - FP16: 16-bit half precision
     * - BF16: 16-bit brain float
     */
    template <ActivationPrecision Precision = ActivationPrecision::FP32>
    class CUDARingKVCache : public ICUDARingKVCache
    {
    public:
        using DataT = typename detail::CUDAKVCacheType<Precision>::Type;
        using EntryT = CUDARingKVEntry<Precision>;

        /**
         * @brief Construct CUDA ring buffer KV cache
         *
         * @param n_layers Number of transformer layers
         * @param batch_size Number of sequences (1 for single-sequence mode)
         * @param max_seq_len Maximum sequence length (ring buffer capacity)
         * @param n_kv_heads Number of KV heads (for GQA)
         * @param head_dim Dimension per head
         * @param device_id CUDA device ID
         */
        CUDARingKVCache(int n_layers, int batch_size, int max_seq_len,
                        int n_kv_heads, int head_dim, int device_id = 0);

        ~CUDARingKVCache();

        // Non-copyable, non-movable (owns GPU memory)
        CUDARingKVCache(const CUDARingKVCache &) = delete;
        CUDARingKVCache &operator=(const CUDARingKVCache &) = delete;
        CUDARingKVCache(CUDARingKVCache &&) = delete;
        CUDARingKVCache &operator=(CUDARingKVCache &&) = delete;

        // =====================================================================
        // ICUDARingKVCache Interface Implementation
        // =====================================================================

        ActivationPrecision precision() const override { return Precision; }
        int num_layers() const override { return n_layers_; }
        int batch_size() const override { return batch_size_; }
        int max_seq_len() const override { return max_seq_len_; }
        int n_kv_heads() const override { return n_kv_heads_; }
        int head_dim() const override { return head_dim_; }
        int kv_dim() const override { return kv_dim_; }

        int get_cached_tokens(int layer, int seq_idx = 0) const override;
        int get_head_position(int layer, int seq_idx = 0) const override;
        bool is_wrapped(int layer, int seq_idx = 0) const override;

        bool append(int layer, int seq_idx,
                    const void *d_k, const void *d_v,
                    int num_tokens, cudaStream_t stream = 0) override;

        bool get_kv_for_attention(int layer, int seq_idx,
                                  const void **d_k_out, const void **d_v_out,
                                  int *kv_len, cudaStream_t stream = 0) override;

        bool linearize_to(int layer, int seq_idx,
                          void *d_k_out, void *d_v_out,
                          int *kv_len, cudaStream_t stream = 0) override;

        void evict_oldest(int layer, int seq_idx, int num_tokens) override;
        void evict_oldest_layer(int layer, int num_tokens) override;

        int gather_kv_batched(int layer, int num_seqs,
                              void *d_k_out, void *d_v_out,
                              int *kv_lens, int max_kv_len,
                              cudaStream_t stream = 0) override;

        void clear() override;
        void clear_sequence(int layer, int seq_idx) override;
        void clear_layer(int layer) override;

        int get_total_evicted() const override { return total_evicted_; }
        void reset_eviction_counter() override { total_evicted_ = 0; }
        int get_linearization_count() const override { return linearization_count_; }
        void reset_linearization_counter() override { linearization_count_ = 0; }

        // =====================================================================
        // Typed Accessors (for direct use when precision is known)
        // =====================================================================

        /**
         * @brief Get typed K/V pointers for attention
         */
        bool get_kv_typed(int layer, int seq_idx,
                          const DataT **d_k_out, const DataT **d_v_out,
                          int *kv_len, cudaStream_t stream = 0);

        /**
         * @brief Append with typed pointers
         */
        bool append_typed(int layer, int seq_idx,
                          const DataT *d_k, const DataT *d_v,
                          int num_tokens, cudaStream_t stream = 0);

    private:
        int n_layers_;
        int batch_size_;
        int max_seq_len_;
        int n_kv_heads_;
        int head_dim_;
        int kv_dim_;
        int device_id_;

        // Entry storage: [n_layers][batch_size]
        std::vector<std::vector<EntryT>> entries_;

        // Statistics
        mutable int total_evicted_ = 0;
        mutable int linearization_count_ = 0;

        // Helper methods
        void allocate_entry(EntryT &entry);
        void free_entry(EntryT &entry);
        void linearize_entry(EntryT &entry, cudaStream_t stream);

        // Kernel launchers
        void launch_append_kernel(EntryT &entry, const DataT *d_k, const DataT *d_v,
                                  int num_tokens, cudaStream_t stream);
        void launch_linearize_kernel(const EntryT &entry, DataT *d_k_out, DataT *d_v_out,
                                     cudaStream_t stream);
        void launch_gather_kernel(const std::vector<EntryT *> &entries,
                                  DataT *d_k_out, DataT *d_v_out,
                                  int *kv_lens, int max_kv_len,
                                  int num_seqs, cudaStream_t stream);
    };

    // =========================================================================
    // Type Aliases
    // =========================================================================

    using CUDARingKVCacheFP32 = CUDARingKVCache<ActivationPrecision::FP32>;
    using CUDARingKVCacheFP16 = CUDARingKVCache<ActivationPrecision::FP16>;
    using CUDARingKVCacheBF16 = CUDARingKVCache<ActivationPrecision::BF16>;

    // =========================================================================
    // Factory Function
    // =========================================================================

    /**
     * @brief Create CUDA ring buffer KV cache with specified precision
     *
     * @param precision Activation precision (FP32, FP16, BF16)
     * @param n_layers Number of transformer layers
     * @param batch_size Number of sequences
     * @param max_seq_len Maximum sequence length
     * @param n_kv_heads Number of KV heads
     * @param head_dim Dimension per head
     * @param device_id CUDA device ID
     * @return Unique pointer to cache, or nullptr on failure
     */
    std::unique_ptr<ICUDARingKVCache> createCUDARingKVCache(
        ActivationPrecision precision,
        int n_layers, int batch_size, int max_seq_len,
        int n_kv_heads, int head_dim, int device_id = 0);

} // namespace llaminar2
