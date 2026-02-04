/**
 * @file ROCmRingKVCache.h
 * @brief ROCm/HIP Ring Buffer KV Cache with O(1) eviction
 * @author Llaminar Team
 * @date January 2026
 *
 * Ring buffer implementation for KV cache on AMD GPUs that enables:
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
 *
 * Target Hardware: AMD MI50 (gfx906 / Vega 20)
 */

#pragma once

// Minimal includes - avoid MPI headers for hipcc compatibility
#include "../../IKVCache.h"                          // Unified KVCache interface (src/v2/kernels/IKVCache.h)
#include "../../../execution/config/RuntimeConfig.h" // For ActivationPrecision
#include "../../../interfaces/IWorkspaceConsumer.h"  // Workspace management
#include "../../../backends/IWorkerGPUContext.h"     // Device context support
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <vector>
#include <memory>
#include <array>

namespace llaminar2
{
    // Forward declarations
    class DeviceWorkspaceManager;
    struct WorkspaceRequirements;

    // =========================================================================
    // KV Cache Workspace Buffer Names
    // =========================================================================

    /**
     * Standard buffer names for KV cache gather workspace.
     * Used by launch_gather_kernel() for batched attention.
     */
    namespace KVCacheWorkspaceBuffers
    {
        /// Array of K cache pointers [batch_size × sizeof(void*)]
        constexpr const char *K_CACHE_PTRS = "kvcache_k_ptrs";
        /// Array of V cache pointers [batch_size × sizeof(void*)]
        constexpr const char *V_CACHE_PTRS = "kvcache_v_ptrs";
        /// Array of tail indices [batch_size × sizeof(int)]
        constexpr const char *TAILS = "kvcache_tails";
        /// Array of count values [batch_size × sizeof(int)]
        constexpr const char *COUNTS = "kvcache_counts";
    }

    // =========================================================================
    // IROCmRingKVCache Interface
    // =========================================================================

    /**
     * @brief Abstract interface for ROCm ring buffer KV cache
     *
     * Enables polymorphic use when precision is determined at runtime.
     * Inherits from IKVCache to provide unified CPU/GPU interface.
     *
     * NOTE: ROCm KV cache uses device pointers internally, so the ITensor-based
     * get_kv() returns ROCmTensor wrappers around the device buffers.
     */
    class IROCmRingKVCache : public IKVCache
    {
    public:
        virtual ~IROCmRingKVCache() = default;

        // =====================================================================
        // IKVCache Interface (public API)
        // =====================================================================

        virtual ActivationPrecision precision() const = 0;
        int n_layers() const override { return num_layers(); }
        int max_seq_len() const override = 0;

        // Per-sequence token tracking (IKVCache)
        int get_cached_tokens(int layer, int seq_idx = 0) const override = 0;

        // Cache management (IKVCache)
        virtual void clear() = 0;
        virtual void clear_sequence(int layer, int seq_idx) = 0;
        virtual void clear_layer(int layer) = 0;

        // =====================================================================
        // ITensor Access (IKVCache interface via get_k/get_v)
        // =====================================================================

        /**
         * @brief Get K/V as ITensor pointers
         *
         * NOTE: For ROCm caches, this creates temporary GpuTensorView wrappers
         * around device buffers. Prefer get_kv_for_attention() for performance.
         */
        bool get_kv(int layer, int seq_idx,
                    ITensor **out_k, ITensor **out_v,
                    int *out_kv_len = nullptr) override
        {
            (void)layer;
            (void)seq_idx;
            (void)out_k;
            (void)out_v;
            (void)out_kv_len;
            return false; // Implemented by concrete class
        }

        bool get_kv(int layer, int seq_idx,
                    const ITensor **out_k, const ITensor **out_v,
                    int *out_kv_len = nullptr) const override
        {
            (void)layer;
            (void)seq_idx;
            (void)out_k;
            (void)out_v;
            (void)out_kv_len;
            return false; // Implemented by concrete class
        }

        /**
         * @brief Append K/V from ITensor pointers (IKVCache interface)
         *
         * Extracts device pointers from tensors and delegates to the
         * void* append method.
         */
        bool append(int layer, int seq_idx,
                    const ITensor *K, const ITensor *V,
                    int num_tokens) override;

        // Bring in convenience overloads from IKVCache
        using IKVCache::append;
        using IKVCache::clear_sequence;
        using IKVCache::get_kv;

        // =====================================================================
        // ROCm-Specific Methods (device pointer APIs)
        // These are exposed for ROCm attention kernels and testing.
        // =====================================================================

        virtual int num_layers() const = 0;
        virtual int batch_size() const = 0;
        virtual int n_kv_heads() const = 0;
        virtual int head_dim() const = 0;
        virtual int kv_dim() const = 0; ///< n_kv_heads * head_dim

        // Ring buffer state
        virtual int get_head_position(int layer, int seq_idx = 0) const = 0;
        virtual bool is_wrapped(int layer, int seq_idx = 0) const = 0;

        /**
         * @brief Append K/V tokens to cache (device pointer version)
         */
        virtual bool append(int layer, int seq_idx,
                            const void *d_k, const void *d_v,
                            int num_tokens, hipStream_t stream = 0) = 0;

        // Convenience for single-sequence mode
        bool append(int layer, const void *d_k, const void *d_v,
                    int num_tokens, hipStream_t stream = 0)
        {
            return append(layer, 0, d_k, d_v, num_tokens, stream);
        }

        /**
         * @brief Get K/V pointers for attention computation
         */
        virtual bool get_kv_for_attention(int layer, int seq_idx,
                                          const void **d_k_out, const void **d_v_out,
                                          int *kv_len, hipStream_t stream = 0) = 0;

        // Convenience for single-sequence mode
        bool get_kv_for_attention(int layer,
                                  const void **d_k_out, const void **d_v_out,
                                  int *kv_len, hipStream_t stream = 0)
        {
            return get_kv_for_attention(layer, 0, d_k_out, d_v_out, kv_len, stream);
        }

        /**
         * @brief Force linearization to external buffer
         */
        virtual bool linearize_to(int layer, int seq_idx,
                                  void *d_k_out, void *d_v_out,
                                  int *kv_len, hipStream_t stream = 0) = 0;

        // =====================================================================
        // Eviction (O(1) - pointer arithmetic only)
        // =====================================================================

        virtual void evict_oldest(int layer, int seq_idx, int num_tokens) = 0;

        void evict_oldest(int layer, int num_tokens)
        {
            evict_oldest(layer, 0, num_tokens);
        }

        virtual void evict_oldest_layer(int layer, int num_tokens) = 0;

        // =====================================================================
        // Batched Operations
        // =====================================================================

        virtual int gather_kv_batched(int layer, int num_seqs,
                                      void *d_k_out, void *d_v_out,
                                      int *kv_lens, int max_kv_len,
                                      hipStream_t stream = 0) = 0;

        // Bring in IKVCache::gather_kv_batched(ITensor*) to avoid hiding
        using IKVCache::gather_kv_batched;

        // =====================================================================
        // Statistics
        // =====================================================================

        virtual int get_total_evicted() const = 0;
        virtual void reset_eviction_counter() = 0;
        virtual int get_linearization_count() const = 0;
        virtual void reset_linearization_counter() = 0;
    };

    // =========================================================================
    // Type Mapping
    // =========================================================================

    namespace detail
    {
        /**
         * @brief Map ActivationPrecision to HIP data type
         */
        template <ActivationPrecision P>
        struct ROCmKVCacheType;

        template <>
        struct ROCmKVCacheType<ActivationPrecision::FP32>
        {
            using Type = float;
            static constexpr size_t element_size = sizeof(float);
        };

        template <>
        struct ROCmKVCacheType<ActivationPrecision::FP16>
        {
            using Type = _Float16;
            static constexpr size_t element_size = sizeof(_Float16);
        };

        template <>
        struct ROCmKVCacheType<ActivationPrecision::BF16>
        {
            using Type = hip_bfloat16;
            static constexpr size_t element_size = sizeof(hip_bfloat16);
        };
    } // namespace detail

    // =========================================================================
    // ROCmRingKVEntry
    // =========================================================================

    /**
     * @brief Per-layer, per-sequence ring buffer entry
     */
    template <ActivationPrecision Precision>
    struct ROCmRingKVEntry
    {
        using DataT = typename detail::ROCmKVCacheType<Precision>::Type;

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
    // ROCmRingKVCache Template Class
    // =========================================================================

    /**
     * @brief ROCm Ring Buffer KV Cache implementation
     *
     * Template parameter Precision determines the data type:
     * - FP32: 32-bit float
     * - FP16: 16-bit half precision
     * - BF16: 16-bit brain float
     *
     * Optimized for AMD MI50 (gfx906):
     * - Uses HIP memory APIs for device allocation
     * - Kernel launch with 256 threads (4 wavefronts)
     * - Coalesced memory access patterns
     *
     * Implements IWorkspaceConsumer for zero-alloc gather operations.
     * When workspace is bound, launch_gather_kernel() uses pre-allocated
     * buffers instead of hipMalloc/hipFree per call.
     */
    template <ActivationPrecision Precision = ActivationPrecision::FP32>
    class ROCmRingKVCache : public IROCmRingKVCache, public IWorkspaceConsumer
    {
    public:
        using DataT = typename detail::ROCmKVCacheType<Precision>::Type;
        using EntryT = ROCmRingKVEntry<Precision>;

        /**
         * @brief Construct ROCm ring buffer KV cache
         *
         * @param n_layers Number of transformer layers
         * @param batch_size Number of sequences (1 for single-sequence mode)
         * @param max_seq_len Maximum sequence length (ring buffer capacity)
         * @param n_kv_heads Number of KV heads (for GQA)
         * @param head_dim Dimension per head
         * @param device_id ROCm device ID
         */
        ROCmRingKVCache(int n_layers, int batch_size, int max_seq_len,
                        int n_kv_heads, int head_dim, int device_id = 0);

        /**
         * @brief Construct ROCm ring buffer KV cache with device context
         *
         * Uses the device context's ordinal for device selection. The context
         * provides the default stream when no explicit stream is passed to methods.
         *
         * @param n_layers Number of transformer layers
         * @param batch_size Number of sequences (1 for single-sequence mode)
         * @param max_seq_len Maximum sequence length (ring buffer capacity)
         * @param n_kv_heads Number of KV heads (for GQA)
         * @param head_dim Dimension per head
         * @param ctx Device context (NOT owned, must outlive cache)
         */
        ROCmRingKVCache(int n_layers, int batch_size, int max_seq_len,
                        int n_kv_heads, int head_dim, IWorkerGPUContext *ctx);

        /**
         * @brief Construct ROCm ring buffer KV cache with sharding (tensor parallelism)
         *
         * @param n_layers Number of transformer layers
         * @param batch_size Number of sequences (1 for single-sequence mode)
         * @param max_seq_len Maximum sequence length (ring buffer capacity)
         * @param n_kv_heads Total number of KV heads (for GQA, across all ranks)
         * @param local_n_kv_heads Number of KV heads stored on this rank
         * @param kv_head_start Starting KV head index for this rank
         * @param head_dim Dimension per head
         * @param device_id ROCm device ID
         */
        ROCmRingKVCache(int n_layers, int batch_size, int max_seq_len,
                        int n_kv_heads, int local_n_kv_heads, int kv_head_start,
                        int head_dim, int device_id = 0);

        /**
         * @brief Construct sharded ROCm ring buffer KV cache with device context
         *
         * Uses the device context's ordinal for device selection. The context
         * provides the default stream when no explicit stream is passed to methods.
         *
         * @param n_layers Number of transformer layers
         * @param batch_size Number of sequences (1 for single-sequence mode)
         * @param max_seq_len Maximum sequence length (ring buffer capacity)
         * @param n_kv_heads Total number of KV heads (for GQA, across all ranks)
         * @param local_n_kv_heads Number of KV heads stored on this rank
         * @param kv_head_start Starting KV head index for this rank
         * @param head_dim Dimension per head
         * @param ctx Device context (NOT owned, must outlive cache)
         */
        ROCmRingKVCache(int n_layers, int batch_size, int max_seq_len,
                        int n_kv_heads, int local_n_kv_heads, int kv_head_start,
                        int head_dim, IWorkerGPUContext *ctx);

        ~ROCmRingKVCache();

        // Non-copyable, non-movable (owns GPU memory)
        ROCmRingKVCache(const ROCmRingKVCache &) = delete;
        ROCmRingKVCache &operator=(const ROCmRingKVCache &) = delete;
        ROCmRingKVCache(ROCmRingKVCache &&) = delete;
        ROCmRingKVCache &operator=(ROCmRingKVCache &&) = delete;

        // =====================================================================
        // IKVCache Interface (public API)
        // =====================================================================

        ActivationPrecision precision() const override { return Precision; }
        int max_seq_len() const override { return max_seq_len_; }

        int get_cached_tokens(int layer, int seq_idx = 0) const override;

        void clear() override;
        void clear_sequence(int layer, int seq_idx) override;
        void clear_layer(int layer) override;

        // ITensor Access (IKVCache interface via get_k/get_v)
        ITensor *get_k(int layer, int seq_idx = 0) override;
        const ITensor *get_k(int layer, int seq_idx = 0) const override;
        ITensor *get_v(int layer, int seq_idx = 0) override;
        const ITensor *get_v(int layer, int seq_idx = 0) const override;

        // Bring in IROCmRingKVCache overloads to avoid hiding
        using IROCmRingKVCache::append;
        using IROCmRingKVCache::clear_sequence;
        using IROCmRingKVCache::gather_kv_batched;

        // =====================================================================
        // ROCm-Specific Methods (implementation)
        // =====================================================================

        int num_layers() const override { return n_layers_; }
        int batch_size() const override { return batch_size_; }
        int n_kv_heads() const override { return n_kv_heads_; }
        int head_dim() const override { return head_dim_; }
        int kv_dim() const override { return kv_dim_; }

        // =====================================================================
        // Sharding (Tensor Parallelism) Accessors (IKVCache interface)
        // =====================================================================

        bool is_sharded() const override { return is_sharded_; }
        int local_n_kv_heads() const override { return local_n_kv_heads_; }
        int kv_head_start() const override { return kv_head_start_; }
        int local_kv_dim() const override { return kv_dim_; }

        int get_head_position(int layer, int seq_idx = 0) const override;
        bool is_wrapped(int layer, int seq_idx = 0) const override;

        bool append(int layer, int seq_idx,
                    const void *d_k, const void *d_v,
                    int num_tokens, hipStream_t stream = 0) override;

        bool get_kv_for_attention(int layer, int seq_idx,
                                  const void **d_k_out, const void **d_v_out,
                                  int *kv_len, hipStream_t stream = 0) override;

        bool linearize_to(int layer, int seq_idx,
                          void *d_k_out, void *d_v_out,
                          int *kv_len, hipStream_t stream = 0) override;

        void evict_oldest(int layer, int seq_idx, int num_tokens) override;
        void evict_oldest_layer(int layer, int num_tokens) override;

        int gather_kv_batched(int layer, int num_seqs,
                              void *d_k_out, void *d_v_out,
                              int *kv_lens, int max_kv_len,
                              hipStream_t stream = 0) override;

        int get_total_evicted() const override { return total_evicted_; }
        void reset_eviction_counter() override { total_evicted_ = 0; }
        int get_linearization_count() const override { return linearization_count_; }
        void reset_linearization_counter() override { linearization_count_ = 0; }

        // =====================================================================
        // Typed Accessors (for direct use when precision is known)
        // =====================================================================

        bool get_kv_typed(int layer, int seq_idx,
                          const DataT **d_k_out, const DataT **d_v_out,
                          int *kv_len, hipStream_t stream = 0);

        bool append_typed(int layer, int seq_idx,
                          const DataT *d_k, const DataT *d_v,
                          int num_tokens, hipStream_t stream = 0);

        // =====================================================================
        // IWorkspaceConsumer Interface
        // =====================================================================

        /**
         * @brief Get workspace requirements for batched gather operations
         *
         * Returns buffer requirements for pointer arrays used in launch_gather_kernel().
         * When workspace is bound, gather operations use pre-allocated buffers
         * instead of hipMalloc/hipFree per call (which are slow on ROCm).
         *
         * @param m Batch size (number of sequences)
         * @param n Unused
         * @param k Unused
         * @return WorkspaceRequirements with buffer specifications
         */
        WorkspaceRequirements getWorkspaceRequirements(
            int m = 0, int n = 0, int k = 0) const override;

        /**
         * @brief Bind workspace manager for managed mode
         *
         * When bound, gather operations use pre-allocated buffers instead of
         * per-call hipMalloc/hipFree (which add ~100-500μs overhead each on ROCm).
         *
         * @param workspace Pointer to workspace manager (NOT owned, must outlive cache)
         */
        void bindWorkspace(DeviceWorkspaceManager *workspace) override;

        /**
         * @brief Check if workspace is currently bound
         */
        bool hasWorkspace() const override;

        /**
         * @brief Get the currently bound workspace manager
         */
        DeviceWorkspaceManager *getWorkspace() const override;

        // =====================================================================
        // Device Context Support
        // =====================================================================

        /**
         * @brief Set device context for kernel launches
         *
         * When set, methods that take an optional stream parameter will use the
         * context's default stream if no explicit stream is provided.
         *
         * @param ctx Device context (NOT owned, must outlive cache)
         */
        void setDeviceContext(IWorkerGPUContext *ctx) { device_ctx_ = ctx; }

        /**
         * @brief Check if device context is set
         */
        bool hasDeviceContext() const { return device_ctx_ != nullptr; }

        /**
         * @brief Get the current device context
         * @return Device context pointer, or nullptr if not set
         */
        IWorkerGPUContext *deviceContext() const { return device_ctx_; }

    private:
        int n_layers_;
        int batch_size_;
        int max_seq_len_;
        int n_kv_heads_;       // Total KV heads (across all ranks)
        int local_n_kv_heads_; // Local KV heads (this rank), == n_kv_heads_ if not sharded
        int kv_head_start_;    // Starting KV head index (0 if not sharded)
        int head_dim_;
        int kv_dim_; // local_n_kv_heads * head_dim (storage dimension)
        int device_id_;
        bool is_sharded_; // True if using local KV heads (TP enabled)

        // Device context for kernel launches (optional)
        // When set, provides default stream for methods that accept optional streams
        IWorkerGPUContext *device_ctx_ = nullptr;

        // Workspace manager for batched gather operations
        // When bound, launch_gather_kernel() uses pre-allocated buffers instead of hipMalloc/hipFree
        DeviceWorkspaceManager *workspace_ = nullptr;

        // Entry storage: [n_layers][batch_size]
        std::vector<std::vector<EntryT>> entries_;

        // GpuTensorView storage for get_k()/get_v(): [n_layers][batch_size][2]
        // Index 0 = K view, Index 1 = V view
        // Mutable because views are lazily created in const methods
        mutable std::vector<std::vector<std::array<std::unique_ptr<ITensor>, 2>>> tensor_views_;

        // Statistics
        mutable int total_evicted_ = 0;
        mutable int linearization_count_ = 0;

        // Helper methods
        void allocate_entry(EntryT &entry);
        void free_entry(EntryT &entry);
        void linearize_entry(EntryT &entry, hipStream_t stream);

        /**
         * @brief Get effective stream for kernel launches
         *
         * Returns the provided stream if non-null, otherwise returns the
         * device context's default stream if available, or nullptr.
         */
        hipStream_t getEffectiveStream(hipStream_t stream) const
        {
            if (stream != nullptr)
                return stream;
            if (device_ctx_ != nullptr)
                return static_cast<hipStream_t>(device_ctx_->defaultStream());
            return nullptr;
        }

        // Kernel launchers
        void launch_append_kernel(EntryT &entry, const DataT *d_k, const DataT *d_v,
                                  int num_tokens, hipStream_t stream);
        void launch_linearize_kernel(const EntryT &entry, DataT *d_k_out, DataT *d_v_out,
                                     hipStream_t stream);
        void launch_gather_kernel(const std::vector<EntryT *> &entries,
                                  DataT *d_k_out, DataT *d_v_out,
                                  int *kv_lens, int max_kv_len,
                                  int num_seqs, hipStream_t stream);
    };

    // =========================================================================
    // Type Aliases
    // =========================================================================

    using ROCmRingKVCacheFP32 = ROCmRingKVCache<ActivationPrecision::FP32>;
    using ROCmRingKVCacheFP16 = ROCmRingKVCache<ActivationPrecision::FP16>;
    using ROCmRingKVCacheBF16 = ROCmRingKVCache<ActivationPrecision::BF16>;

    // Note: Factory functions are declared in ROCmRingKVCacheFactory.h
    // to avoid HIP header pollution in non-HIP code.

} // namespace llaminar2
