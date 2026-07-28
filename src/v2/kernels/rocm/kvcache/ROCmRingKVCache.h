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
#include "ROCmRingKVCacheBase.h"                     // Common base for ROCm ring KV caches
#include "../../../execution/config/RuntimeConfig.h" // For ActivationPrecision
#include "../../../interfaces/IWorkspaceConsumer.h"  // Workspace management
#include "../../../backends/IWorkerGPUContext.h"     // Device context support
#include "../../../tensors/BlockStructures.h"        // Q8_1Block
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
     * Standard buffer names for cache-owned conversion workspace.
     */
    namespace KVCacheWorkspaceBuffers
    {
        /// K conversion scratch used by append/read precision adaptation.
        constexpr const char *CONV_SCRATCH_K = "kvcache_conv_scratch_k";
        /// V conversion scratch used by append/read precision adaptation.
        constexpr const char *CONV_SCRATCH_V = "kvcache_conv_scratch_v";
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
    class IROCmRingKVCache : public ROCmRingKVCacheBase
    {
    public:
        virtual ~IROCmRingKVCache();

        // =====================================================================
        // IKVCache Interface (public API)
        // =====================================================================

        virtual ActivationPrecision k_precision() const = 0;

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

        bool appendWithStream(int layer, int seq_idx, const ITensor *K, const ITensor *V,
                              int num_tokens, void *gpu_stream) override;

        // Bring in convenience overloads from IKVCache
        using IKVCache::append;
        using IKVCache::appendWithStream;
        using IKVCache::get_kv;

        // =====================================================================
        // ROCm-Specific Methods (device pointer APIs)
        // These are exposed for ROCm attention kernels and testing.
        // =====================================================================

        // Core accessors inherited from ROCmRingKVCacheBase:
        //   num_layers(), batch_size(), n_kv_heads(), head_dim(), kv_dim(),
        //   device_id(), get_head_position(), is_wrapped()

        /**
         * @brief Append K/V tokens to cache (device pointer version)
         */
        virtual bool append(int layer, int seq_idx,
                            const void *d_k, const void *d_v,
                            int num_tokens, hipStream_t stream) = 0;

        /**
         * @brief Capture-safe fused convert+append path.
         *
         * Used by appendWithStream() for precision-mismatch paths to avoid
         * intermediate scratch writes in graph-captured decode. Every concrete
         * cache must provide the operation; there is no conversion fallback.
         */
        virtual bool appendConvertedWithStream(int layer, int seq_idx,
                                               const void *d_k_src, const void *d_v_src,
                                               TensorType src_type,
                                               int num_tokens,
                                               hipStream_t stream) = 0;

        // Convenience for single-sequence mode
        bool append(int layer, const void *d_k, const void *d_v,
                    int num_tokens, hipStream_t stream)
        {
            return append(layer, 0, d_k, d_v, num_tokens, stream);
        }

        /**
         * @brief Get K/V pointers for attention computation
         */
        virtual bool get_kv_for_attention(int layer, int seq_idx,
                                          const void **d_k_out, const void **d_v_out,
                                          int *kv_len, hipStream_t stream) = 0;

        // Convenience for single-sequence mode
        bool get_kv_for_attention(int layer,
                                  const void **d_k_out, const void **d_v_out,
                                  int *kv_len, hipStream_t stream)
        {
            return get_kv_for_attention(layer, 0, d_k_out, d_v_out, kv_len, stream);
        }

        /**
         * @brief Force linearization to external buffer
         */
        virtual bool linearize_to(int layer, int seq_idx,
                                  void *d_k_out, void *d_v_out,
                                  int *kv_len, hipStream_t stream) = 0;

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
                                      hipStream_t stream) = 0;

        // Bring in IKVCache::gather_kv_batched(ITensor*) to avoid hiding
        using IKVCache::gather_kv_batched;

        // =====================================================================
        // Statistics
        // =====================================================================

        virtual int get_linearization_count() const = 0;
        virtual void reset_linearization_counter() = 0;

        // Graph capture and canonical sequence state are inherited from
        // ROCmRingKVCacheBase.

    protected:
        // Constructor forwards to ROCmRingKVCacheBase
        IROCmRingKVCache(int n_layers, int batch_size, int max_seq_len,
                         int n_kv_heads, int head_dim, int kv_dim, int device_id)
            : ROCmRingKVCacheBase(n_layers, batch_size, max_seq_len,
                                  n_kv_heads, head_dim, kv_dim, device_id)
        {
        }

        // =====================================================================
        // Pre-allocated conversion scratch buffers
        // =====================================================================
        // Avoids hipMalloc/hipFree per appendWithStream call.
        // Lazily allocated and grown as needed.

        /// Ensure scratch buffers have at least `bytes` capacity each.
        /// Returns true on success. Thread-safe via single-writer assumption
        /// (one decode step at a time).
        bool ensureConvScratch(size_t bytes);

        /// Free scratch buffers (called from destructor)
        void freeConvScratch();

        void *conv_scratch_k_ = nullptr;   ///< Scratch buffer for K conversion
        void *conv_scratch_v_ = nullptr;   ///< Scratch buffer for V conversion
        size_t conv_scratch_capacity_ = 0; ///< Current capacity in bytes (per buffer)
        bool conv_scratch_workspace_backed_ = false;
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

        template <>
        struct ROCmKVCacheType<ActivationPrecision::Q8_1>
        {
            using Type = Q8_1Block;
            static constexpr size_t element_size = sizeof(Q8_1Block);
        };
    } // namespace detail

    // =========================================================================
    // ROCmRingKVEntry
    // =========================================================================

    /**
     * @brief Immutable per-layer, per-sequence device payload topology.
     *
     * Mutable ring head/count state deliberately does not live in this host
     * object. ROCmRingKVCacheBase owns one canonical device row for each entry,
     * and every append/read kernel consumes that row directly.
     */
    template <ActivationPrecision Precision>
    struct ROCmRingKVEntry
    {
        using DataT = typename detail::ROCmKVCacheType<Precision>::Type;

        DataT *d_K = nullptr; ///< Device memory: [max_seq_len, kv_dim]
        DataT *d_V = nullptr; ///< Device memory: [max_seq_len, kv_dim]

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
     * - Q8_1: 8-bit block-quantized (Q8_1Block)
     *
     * Optimized for AMD MI50 (gfx906):
     * - Uses HIP memory APIs for device allocation
     * - Kernel launch with 256 threads (4 wavefronts)
     * - Coalesced memory access patterns
     *
     * Implements IWorkspaceConsumer for zero-allocation grouped request reads.
     * Workspace-backed conversion scratch is stable across graph capture and
     * replay; ring metadata and entry-pointer tables remain device owned.
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

        ActivationPrecision k_precision() const override { return Precision; }

        // ITensor Access (IKVCache interface via get_k/get_v)
        ITensor *get_k(int layer, int seq_idx = 0) override;
        const ITensor *get_k(int layer, int seq_idx = 0) const override;
        ITensor *get_v(int layer, int seq_idx = 0) override;
        const ITensor *get_v(int layer, int seq_idx = 0) const override;

        // IKVCache get_kv() override using get_k()/get_v()
        bool get_kv(int layer, int seq_idx,
                    ITensor **out_k, ITensor **out_v,
                    int *out_kv_len = nullptr) override;
        bool get_kv(int layer, int seq_idx,
                    const ITensor **out_k, const ITensor **out_v,
                    int *out_kv_len = nullptr) const override;
        bool get_kv_snapshot_view(int layer, int seq_idx,
                                  int token_count,
                                  ITensor **out_k, ITensor **out_v,
                                  int *out_kv_len = nullptr) override;
        bool get_kv_snapshot_view(int layer, int seq_idx,
                                  int token_count,
                                  const ITensor **out_k, const ITensor **out_v,
                                  int *out_kv_len = nullptr) const override;

        /**
         * @brief Gather request-local ROCm rings from device-owned metadata.
         *
         * The immutable entry-pointer tables are published when workspace is
         * bound.  Each invocation then reads live ring heads and counts from
         * the same device scalars updated by captured append/publication
         * kernels, so request-batched attention never adopts a host mirror.
         */
        bool get_kv_batched_device_view(
            int layer,
            int first_seq_idx,
            int request_count,
            ITensor **out_k,
            ITensor **out_v,
            void *gpu_stream) override;

        bool get_kv_batched_converted_device_view(
            int layer,
            int first_seq_idx,
            int request_count,
            ActivationPrecision target,
            ITensor **out_k,
            ITensor **out_v,
            const KVReadParams &read) override;

        // get_kv_converted with FP16 shadow buffers + optional RoPE (IKVCache override)
        bool get_kv_converted(int layer, int seq_idx,
                              ActivationPrecision target,
                              ITensor **out_k, ITensor **out_v,
                              int *out_kv_len = nullptr,
                              const KVReadParams *rope = nullptr) override;

        /**
         * @brief Append grouped verifier rows with serial-decode cache semantics.
         *
         * ROCm MTP verifier rows can arrive as position-major rows or as
         * head-major rows after grouped RoPE.  The grouped append kernel copies
         * either source layout into the ring positions serial decode would have
         * written, and it uses the caller-provided HIP stream for graph capture.
         */
        bool appendVerifierRowsDecodeEquivalent(int layer,
                                                int seq_idx,
                                                const ITensor *K,
                                                const ITensor *V,
                                                int verifier_rows,
                                                void *gpu_stream = nullptr) override;

        // Bring in IROCmRingKVCache overloads to avoid hiding
        using IROCmRingKVCache::append;
        using IROCmRingKVCache::gather_kv_batched;

        // =====================================================================
        // Sharding (Tensor Parallelism) Accessors (IKVCache interface)
        // =====================================================================

        bool is_sharded() const override { return is_sharded_; }
        int local_n_kv_heads() const override { return local_n_kv_heads_; }
        int kv_head_start() const override { return kv_head_start_; }
        int local_kv_dim() const override { return kv_dim_; }
        TensorLayout kv_layout() const override { return TensorLayout::KV_POS_HEAD_DIM; }

        KVCacheLogicalBlockLayout logicalBlockLayout(int global_layer, int token_count) const override;
        KVCacheSequenceState sequenceState(int global_layer, int seq_idx) const override;
        bool exportLogicalBlock(const KVCacheLogicalBlockDescriptor &desc, void *dst_k, void *dst_v) const override;
        bool importLogicalBlock(const KVCacheLogicalBlockDescriptor &desc, const void *src_k, const void *src_v) override;
        bool truncateSequence(int seq_idx, int cached_tokens, void *stream = nullptr) override;

        bool append(int layer, int seq_idx,
                    const void *d_k, const void *d_v,
                    int num_tokens, hipStream_t stream) override;

        bool get_kv_for_attention(int layer, int seq_idx,
                                  const void **d_k_out, const void **d_v_out,
                                  int *kv_len, hipStream_t stream) override;

        bool linearize_to(int layer, int seq_idx,
                          void *d_k_out, void *d_v_out,
                          int *kv_len, hipStream_t stream) override;

        void evict_oldest(int layer, int seq_idx, int num_tokens) override;
        void evict_oldest_layer(int layer, int num_tokens) override;

        int gather_kv_batched(int layer, int num_seqs,
                              void *d_k_out, void *d_v_out,
                              int *kv_lens, int max_kv_len,
                              hipStream_t stream) override;

        int get_linearization_count() const override { return linearization_count_; }
        void reset_linearization_counter() override { linearization_count_ = 0; }

        // =====================================================================
        // Typed Accessors (for direct use when precision is known)
        // =====================================================================

        bool get_kv_typed(int layer, int seq_idx,
                          const DataT **d_k_out, const DataT **d_v_out,
                          int *kv_len, hipStream_t stream);

        bool append_typed(int layer, int seq_idx,
                          const DataT *d_k, const DataT *d_v,
                          int num_tokens, hipStream_t stream);

        bool appendConvertedWithStream(int layer, int seq_idx,
                                       const void *d_k_src, const void *d_v_src,
                                       TensorType src_type,
                                       int num_tokens, hipStream_t stream) override;

        // =====================================================================
        // IWorkspaceConsumer Interface
        // =====================================================================

        /**
         * @brief Get workspace requirements for batched gather operations
         *
         * Returns conversion/materialization scratch for grouped request reads.
         * Immutable entry-pointer tables are cache owned, while mutable ring
         * state is read directly from canonical device head/count rows.
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

    protected:
        void onResetLayerSequenceState(int layer, int seq_idx) override
        {
            invalidateRoPEShadow(layer, seq_idx);
        }

    private:
        int local_n_kv_heads_; // Local KV heads (this rank), == n_kv_heads_ if not sharded
        int kv_head_start_;    // Starting KV head index (0 if not sharded)
        int kv_storage_dim_;   // per-token storage units (elements for fp/bf16, blocks for Q8_1)
        bool is_sharded_;      // True if using local KV heads (TP enabled)

        // Device context for kernel launches (optional)
        // When set, provides default stream for methods that accept optional streams
        IWorkerGPUContext *device_ctx_ = nullptr;

        // Workspace manager for grouped conversion/materialization scratch.
        DeviceWorkspaceManager *workspace_ = nullptr;

        // Entry storage: [n_layers][batch_size]
        std::vector<std::vector<EntryT>> entries_;

        // Pooled device memory for all KV cache entries.
        // Single hipMalloc replaces n_layers × batch_size × 2 individual calls.
        // Layout: [n_layers][batch_size][2_buffers][max_seq_len × kv_storage_dim]
        // Wrapped-ring linearization uses the cache-level conversion scratch on demand.
        void *pool_base_ = nullptr;
        size_t pool_size_ = 0;

        // GpuTensorView storage for get_k()/get_v(): [n_layers][batch_size][2]
        // Index 0 = K view, Index 1 = V view
        // Mutable because views are lazily created in const methods
        mutable std::vector<std::vector<std::array<std::unique_ptr<ITensor>, 2>>> tensor_views_;
        mutable std::vector<std::vector<std::array<std::unique_ptr<ITensor>, 2>>> snapshot_tensor_views_;

        /// Cache-owned tensor wrappers over workspace-backed grouped payloads.
        std::unique_ptr<ITensor> batched_k_view_;
        std::unique_ptr<ITensor> batched_v_view_;
        /// Stable FP16 wrappers for grouped device-owned conversion/RoPE reads.
        std::unique_ptr<ITensor> converted_batched_k_view_;
        std::unique_ptr<ITensor> converted_batched_v_view_;
        /**
         * Immutable device topology for grouped cache reads.
         *
         * These tables deliberately belong to this cache rather than the
         * shared workspace. A worker owns both its main decode cache and an
         * MTP sidecar cache; giving those objects the same named workspace
         * buffer would let the last binder silently replace the first cache's
         * topology. The pointed-to K/V allocations are stable for the cache's
         * lifetime, so publishing the tables once before graph capture is both
         * sufficient and capture-safe.
         */
        DataT **d_batched_k_entry_table_ = nullptr;
        DataT **d_batched_v_entry_table_ = nullptr;
        /// True after construction publishes every immutable entry pointer.
        bool batched_pointer_tables_ready_ = false;

        // Diagnostic scalar-read statistics. Canonical ring state remains on device.
        mutable int linearization_count_ = 0;

        // =====================================================================
        // RoPE-on-read Shadow Buffers (for get_kv_converted)
        // =====================================================================

        struct RoPEShadow
        {
            void *d_K = nullptr;       ///< [max_seq_len, kv_dim] FP16 K with RoPE
            void *d_V = nullptr;       ///< [max_seq_len, kv_dim] FP16 V

            std::unique_ptr<ITensor> k_view; ///< GpuTensorView for K
            std::unique_ptr<ITensor> v_view; ///< GpuTensorView for V
        };

        /// [n_layers][batch_size] — lazily initialized
        mutable std::vector<std::vector<RoPEShadow>> rope_shadows_;

        /// Bind a layer/sequence shadow view to graph-planned conversion output.
        void ensureRoPEShadow(int layer, int seq_idx);

        /// Invalidate shadow after append/evict
        void invalidateRoPEShadow(int layer, int seq_idx) const;

        // Helper methods
        void allocate_pool(); // One backend allocation for all entries
        void free_pool();     // Release the one backend allocation
        void assign_entry_from_pool(EntryT &entry, int linear_index);
        void allocate_all_entries();        // Pool + assign entries + tensor_views + device_params

        /**
         * @brief Publish immutable K/V entry topology before graph capture can begin.
         *
         * Grouped cache kernels consume one device array of K pointers and one
         * device array of V pointers. The entry addresses are fixed when the
         * cache pool is assigned, so construction publishes these arrays
         * exactly once. Keeping publication out of bindWorkspace() makes
         * repeated workspace binding a host-only operation and prevents an
         * implicit default-stream H2D copy from invalidating HIP graph capture.
         *
         * @throws std::runtime_error if either table cannot be allocated or
         *         copied to the owning device.
         */
        void initializeBatchedEntryPointerTables();

        /**
         * @brief Release cache-owned entry topology during teardown or failed construction.
         *
         * This method is noexcept so partially constructed caches can clean up
         * both tables before propagating their original initialization error.
         */
        void releaseBatchedEntryPointerTables() noexcept;

        bool linearize_entry(
            EntryT &entry,
            int head,
            int count,
            hipStream_t stream);

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
        /**
         * @brief Launch a graph-capturable append kernel with explicit dynamic row ownership.
         *
         * @param d_head Device scalar containing the ring head for this replay.
         * @param d_append_count Device scalar containing the real rows inside a captured bucket.
         */
        void launch_append_kernel_dynamic(EntryT &entry, const DataT *d_k, const DataT *d_v,
                                          const int *d_head, const int *d_append_count,
                                          int num_tokens, hipStream_t stream);
        void launch_linearize_kernel(
            const EntryT &entry,
            int head,
            int count,
            DataT *d_k_out,
            DataT *d_v_out,
            hipStream_t stream);
    };

    // =========================================================================
    // Type Aliases
    // =========================================================================

    using ROCmRingKVCacheFP32 = ROCmRingKVCache<ActivationPrecision::FP32>;
    using ROCmRingKVCacheFP16 = ROCmRingKVCache<ActivationPrecision::FP16>;
    using ROCmRingKVCacheBF16 = ROCmRingKVCache<ActivationPrecision::BF16>;
    using ROCmRingKVCacheQ8_1 = ROCmRingKVCache<ActivationPrecision::Q8_1>;

    // Note: Factory functions are declared in ROCmRingKVCacheFactory.h
    // to avoid HIP header pollution in non-HIP code.

} // namespace llaminar2
