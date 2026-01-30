/**
 * @file CPUKVCache.h
 * @brief Unified Key-Value cache supporting both single-sequence and batched modes
 * @author David Sanftenberg
 * @date December 2025
 *
 * Combines features of KVCache (typed precision) and BatchedKVCache (multi-sequence)
 * into a single implementation.
 *
 * Design Goals:
 * 1. Support FP32, BF16, FP16, Q8_1 precision for cache storage
 * 2. Support batch_size=1 (incremental decode) and batch_size>1 (batched inference)
 * 3. Efficient append without reallocation (pre-allocated to max_seq_len)
 * 4. Per-layer device placement for heterogeneous execution
 * 5. Sliding window eviction support
 *
 * Usage Patterns:
 * - Single-sequence decode: batch_size=1, append one token at a time
 * - Batched prefill: batch_size=N, append variable-length sequences
 * - Continuous batching: dynamic sequence completion with clear_sequence()
 */

#pragma once

#include "../IKVCache.h" // Unified KVCache interface
#include "../../tensors/Tensors.h"
#include "../../tensors/TensorFactory.h"
#include "../../tensors/TensorLayout.h"
#include "../../backends/DeviceId.h"
#include "../../utils/MPIContext.h"
#include "../../execution/config/RuntimeConfig.h"
#include <vector>
#include <memory>

namespace llaminar2
{

    // =========================================================================
    // KV Cache Layout Mode
    // =========================================================================

    /**
     * @brief Memory layout mode for KV cache storage
     *
     * Controls how K/V tensors are organized in memory:
     *
     * POSITION_MAJOR (default):
     *   Storage: [position][n_kv_heads][head_dim]
     *   Block indexing: block[p * n_kv_heads + h]
     *   Best for: Sequential cache append (new positions at end)
     *   Used by: FP32/BF16/FP16 attention backends
     *
     * HEAD_MAJOR:
     *   Storage: [n_kv_heads][position][head_dim]
     *   Block indexing: block[h * kv_len + p]
     *   Best for: Per-head attention computation (head-contiguous access)
     *   Used by: Q16_INTEGER attention kernel
     *
     * @note Choosing HEAD_MAJOR for Q16_1 caches eliminates the
     *       transpose workaround in FusedAttentionWoStage.
     */
    enum class KVCacheLayoutMode : uint8_t
    {
        POSITION_MAJOR, ///< [position][n_kv_heads][head_dim] - cache-append friendly
        HEAD_MAJOR      ///< [n_kv_heads][position][head_dim] - attention-compute friendly
    };

    // =========================================================================
    // ICPUKVCache Interface
    // =========================================================================

    /**
     * @brief Abstract interface for type-erased unified KV cache access
     *
     * Enables polymorphic use when precision is determined at runtime.
     * Supports both single-sequence (batch_size=1) and batched modes.
     *
     * Inherits from IKVCache to provide unified CPU/GPU interface.
     */
    class ICPUKVCache : public IKVCache
    {
    public:
        virtual ~ICPUKVCache() = default;

        // =====================================================================
        // IKVCache Interface (public API)
        // =====================================================================

        // Metadata (IKVCache)
        virtual ActivationPrecision precision() const = 0;
        int n_layers() const override { return num_layers(); }
        virtual int max_seq_len() const = 0;
        virtual TensorLayout kv_layout() const = 0;

        // Per-sequence token tracking (IKVCache)
        virtual int get_cached_tokens(int layer, int seq_idx = 0) const = 0;

        // Sharding (Tensor Parallelism) info (IKVCache)
        virtual bool is_sharded() const = 0;      ///< True if using local KV heads
        virtual int local_n_kv_heads() const = 0; ///< Local KV heads (this rank)
        virtual int local_kv_dim() const = 0;     ///< local_n_kv_heads * head_dim
        virtual int kv_head_start() const = 0;    ///< Starting KV head index

        // =====================================================================
        // CPU-Specific Methods (exposed for testing and internal use)
        // =====================================================================

        virtual int num_layers() const = 0;
        virtual int batch_size() const = 0;
        virtual KVCacheLayoutMode layout_mode() const = 0;
        virtual int n_kv_heads() const = 0; ///< Total KV heads (across all ranks)

    public:
        // =================================================================
        // Unified KV Access (preferred interface)
        // =================================================================

        /**
         * @brief Get both K and V cache tensors for attention computation
         *
         * This is the preferred interface - fetches both K and V in a single call.
         *
         * @param layer Layer index
         * @param seq_idx Sequence index (default 0)
         * @param out_k Output: pointer to K cache tensor
         * @param out_v Output: pointer to V cache tensor
         * @param out_kv_len Output: number of cached tokens (optional)
         * @return true on success
         */
        virtual bool get_kv(int layer, int seq_idx,
                            ITensor **out_k, ITensor **out_v,
                            int *out_kv_len = nullptr) = 0;

        virtual bool get_kv(int layer, int seq_idx,
                            const ITensor **out_k, const ITensor **out_v,
                            int *out_kv_len = nullptr) const = 0;

        // Convenience overloads for seq_idx=0
        bool get_kv(int layer, ITensor **out_k, ITensor **out_v, int *out_kv_len = nullptr)
        {
            return get_kv(layer, 0, out_k, out_v, out_kv_len);
        }

        bool get_kv(int layer, const ITensor **out_k, const ITensor **out_v, int *out_kv_len = nullptr) const
        {
            return get_kv(layer, 0, out_k, out_v, out_kv_len);
        }

        // =================================================================
        // Legacy tensor access (deprecated - use get_kv() instead)
        // =================================================================

        /// @deprecated Use get_kv() instead
        ITensor *get_k(int layer, int seq_idx = 0) override = 0;
        /// @deprecated Use get_kv() instead
        const ITensor *get_k(int layer, int seq_idx = 0) const override = 0;
        /// @deprecated Use get_kv() instead
        ITensor *get_v(int layer, int seq_idx = 0) override = 0;
        /// @deprecated Use get_kv() instead
        const ITensor *get_v(int layer, int seq_idx = 0) const override = 0;

        // =================================================================
        // IKVCache interface bridging
        // =================================================================

        // Bridge IKVCache::append(ITensor*) to ICPUKVCache::append_kv(TensorBase*)
        // dynamic_cast required due to virtual inheritance in ITensor hierarchy
        bool append(int layer, int seq_idx, const ITensor *K, const ITensor *V, int num_tokens) override
        {
            return append_kv(layer, seq_idx,
                             dynamic_cast<const TensorBase *>(K),
                             dynamic_cast<const TensorBase *>(V),
                             num_tokens);
        }

        // Bring in IKVCache convenience overloads
        using IKVCache::append;

        // IKVCache::gather_kv_batched bridge (uses ITensor*, delegates to TensorBase* version)
        int gather_kv_batched(
            int layer,
            int num_sequences,
            ITensor *out_k,
            ITensor *out_v,
            std::vector<int> &out_kv_lens) override
        {
            // dynamic_cast required due to virtual inheritance in ITensor hierarchy
            return gather_kv_batched(layer, num_sequences,
                                     dynamic_cast<TensorBase *>(out_k),
                                     dynamic_cast<TensorBase *>(out_v),
                                     out_kv_lens);
        }

        // Cache management (IKVCache)
        void clear() override = 0;
        void clear_sequence(int layer, int seq_idx) override = 0;
        void clear_layer(int layer) override = 0;

        // Bring in IKVCache::clear_sequence(seq_idx) default implementation
        using IKVCache::clear_sequence;

        // =================================================================
        // CPU-Specific Methods (for testing and internal use)
        // =================================================================

        /**
         * @brief Append K/V to cache for a sequence (TensorBase version)
         *
         * @param layer Layer index
         * @param seq_idx Sequence index (default 0 for single-sequence mode)
         * @param new_k K tensor to append
         * @param new_v V tensor to append
         * @return true on success, false if capacity exceeded
         */
        virtual bool append_kv(int layer, int seq_idx, const TensorBase *new_k, const TensorBase *new_v) = 0;

        /**
         * @brief Append K/V with explicit token count (partial tensor)
         */
        virtual bool append_kv(int layer, int seq_idx, const TensorBase *new_k, const TensorBase *new_v, int num_tokens) = 0;

        // Convenience for single-sequence mode (seq_idx = 0)
        bool append_kv(int layer, const TensorBase *new_k, const TensorBase *new_v)
        {
            return append_kv(layer, 0, new_k, new_v);
        }

        bool append_kv(int layer, const TensorBase *new_k, const TensorBase *new_v, int num_tokens)
        {
            return append_kv(layer, 0, new_k, new_v, num_tokens);
        }

        /**
         * @brief Gather K/V from multiple cache slots into batched output tensors (TensorBase version)
         *
         * @param layer Layer index
         * @param num_sequences Number of sequences to gather (typically batch_size)
         * @param out_k Output K tensor [num_sequences * max_kv_len, kv_dim]
         * @param out_v Output V tensor [num_sequences * max_kv_len, kv_dim]
         * @param out_kv_lens Per-sequence kv_lens (output, size = num_sequences)
         * @return Maximum kv_len across all sequences, or -1 on error
         */
        virtual int gather_kv_batched(
            int layer,
            int num_sequences,
            TensorBase *out_k,
            TensorBase *out_v,
            std::vector<int> &out_kv_lens) = 0;

        // Eviction operations
        virtual void evict_oldest(int tokens_to_evict) = 0;
        virtual void evict_oldest_from_sequence(int seq_idx, int tokens_to_evict) = 0;

        // Device placement
        virtual DeviceId get_layer_device(int layer) const = 0;

        // Eviction tracking
        virtual int get_total_evicted() const = 0;
        virtual void reset_eviction_counter() = 0;
    };

    // =========================================================================
    // Forward Declaration
    // =========================================================================

    namespace detail
    {
        /**
         * @brief Map ActivationPrecision to tensor type for KV cache
         */
        template <ActivationPrecision P>
        struct CPUKVCacheTensor;

        template <>
        struct CPUKVCacheTensor<ActivationPrecision::FP32>
        {
            using Type = FP32Tensor;
        };

        template <>
        struct CPUKVCacheTensor<ActivationPrecision::BF16>
        {
            using Type = BF16Tensor;
        };

        template <>
        struct CPUKVCacheTensor<ActivationPrecision::FP16>
        {
            using Type = FP16Tensor;
        };

        template <>
        struct CPUKVCacheTensor<ActivationPrecision::Q8_1>
        {
            using Type = Q8_1Tensor;
        };

        template <>
        struct CPUKVCacheTensor<ActivationPrecision::Q16_1>
        {
            using Type = Q16_1Tensor;
        };
    } // namespace detail

    // =========================================================================
    // CPUKVCacheEntry
    // =========================================================================

    /**
     * @brief Per-layer, per-sequence KV cache entry with typed tensors
     */
    template <ActivationPrecision Precision>
    struct CPUKVCacheEntry
    {
        using TensorT = typename detail::CPUKVCacheTensor<Precision>::Type;

        std::shared_ptr<TensorT> K; // [max_seq_len, n_kv_heads * head_dim] pre-allocated
        std::shared_ptr<TensorT> V; // [max_seq_len, n_kv_heads * head_dim] pre-allocated
        int cached_tokens = 0;      // Number of tokens currently cached
    };

    // =========================================================================
    // CPUKVCache Template Class
    // =========================================================================

    /**
     * @brief Unified typed KV cache for single-sequence and batched inference
     *
     * Stores K/V tensors for each layer and sequence to enable:
     * - Single-sequence incremental decode (batch_size=1)
     * - Batched prefill and decode (batch_size>1)
     *
     * Template parameter Precision determines the tensor type:
     * - FP32: Standard 32-bit float (default, highest accuracy)
     * - BF16: Brain float16 (Intel AMX optimization)
     * - FP16: Half precision (ARM/GPU optimization)
     * - Q8_1: Block-quantized int8 (memory bandwidth optimization)
     *
     * Design:
     * - Pre-allocates K/V buffers up to max_seq_len per sequence
     * - Supports efficient append without reallocation
     * - Per-sequence tracking for variable-length sequences
     * - Per-layer device placement for heterogeneous execution
     * - Configurable layout mode (POSITION_MAJOR or HEAD_MAJOR)
     */
    template <ActivationPrecision Precision = ActivationPrecision::FP32>
    class CPUKVCache : public ICPUKVCache
    {
    public:
        using TensorT = typename detail::CPUKVCacheTensor<Precision>::Type;
        using EntryT = CPUKVCacheEntry<Precision>;

        /**
         * @brief Construct unified KV cache
         *
         * @param mpi_ctx MPI context for NUMA-aware allocation
         * @param n_layers Number of transformer layers
         * @param batch_size Number of sequences (1 for single-sequence mode)
         * @param max_seq_len Maximum sequence length (cache capacity per sequence)
         * @param n_kv_heads Number of KV heads (GQA)
         * @param head_dim Dimension per head
         * @param device Default device for all layers (CPU by default)
         * @param layout_mode Memory layout mode (POSITION_MAJOR or HEAD_MAJOR)
         */
        CPUKVCache(const MPIContext &mpi_ctx, int n_layers, int batch_size, int max_seq_len,
                   int n_kv_heads, int head_dim, DeviceId device = DeviceId::cpu(),
                   KVCacheLayoutMode layout_mode = KVCacheLayoutMode::POSITION_MAJOR);

        /**
         * @brief Construct unified KV cache with per-layer device placement
         */
        CPUKVCache(const MPIContext &mpi_ctx, int n_layers, int batch_size, int max_seq_len,
                   int n_kv_heads, int head_dim, const std::vector<int> &attention_devices,
                   KVCacheLayoutMode layout_mode = KVCacheLayoutMode::POSITION_MAJOR);

        /**
         * @brief Construct unified KV cache with sharded (local) KV heads for tensor parallelism
         *
         * When using column-parallel QKV (Phase 3 TP), each rank computes only a subset
         * of KV heads. This constructor creates a cache that stores only local heads.
         *
         * @param mpi_ctx MPI context for NUMA-aware allocation
         * @param n_layers Number of transformer layers
         * @param batch_size Number of sequences (1 for single-sequence mode)
         * @param max_seq_len Maximum sequence length (cache capacity per sequence)
         * @param n_kv_heads Total number of KV heads (GQA) across all ranks
         * @param local_n_kv_heads Number of KV heads on this rank (n_kv_heads / world_size)
         * @param kv_head_start Starting KV head index for this rank
         * @param head_dim Dimension per head
         * @param device Default device for all layers (CPU by default)
         * @param layout_mode Memory layout mode (POSITION_MAJOR or HEAD_MAJOR)
         */
        CPUKVCache(const MPIContext &mpi_ctx, int n_layers, int batch_size, int max_seq_len,
                   int n_kv_heads, int local_n_kv_heads, int kv_head_start,
                   int head_dim, DeviceId device = DeviceId::cpu(),
                   KVCacheLayoutMode layout_mode = KVCacheLayoutMode::POSITION_MAJOR);

        /**
         * @brief Construct unified KV cache with sharded KV heads and per-layer device placement
         */
        CPUKVCache(const MPIContext &mpi_ctx, int n_layers, int batch_size, int max_seq_len,
                   int n_kv_heads, int local_n_kv_heads, int kv_head_start,
                   int head_dim, const std::vector<int> &attention_devices,
                   KVCacheLayoutMode layout_mode = KVCacheLayoutMode::POSITION_MAJOR);

        // ICPUKVCache interface implementation
        ActivationPrecision precision() const override { return Precision; }
        int max_seq_len() const override { return max_seq_len_; }

        // Layout mode accessors
        TensorLayout kv_layout() const override
        {
            return (layout_mode_ == KVCacheLayoutMode::HEAD_MAJOR)
                       ? TensorLayout::KV_HEAD_POS_DIM
                       : TensorLayout::KV_POS_HEAD_DIM;
        }

        int get_cached_tokens(int layer, int seq_idx = 0) const override;

        // Unified KV access (preferred interface)
        bool get_kv(int layer, int seq_idx,
                    ITensor **out_k, ITensor **out_v,
                    int *out_kv_len = nullptr) override;

        bool get_kv(int layer, int seq_idx,
                    const ITensor **out_k, const ITensor **out_v,
                    int *out_kv_len = nullptr) const override;

        // Bring in convenience methods from interface
        using ICPUKVCache::get_kv;

        // Legacy individual accessors (deprecated - use get_kv() instead)
        ITensor *get_k(int layer, int seq_idx = 0) override;
        const ITensor *get_k(int layer, int seq_idx = 0) const override;
        ITensor *get_v(int layer, int seq_idx = 0) override;
        const ITensor *get_v(int layer, int seq_idx = 0) const override;

        void clear() override;
        void clear_sequence(int layer, int seq_idx) override;
        void clear_layer(int layer) override;

        // Bring in IKVCache::clear_sequence(seq_idx) default implementation
        using ICPUKVCache::clear_sequence;

        // =====================================================================
        // Sharding (Tensor Parallelism) Accessors (IKVCache interface)
        // =====================================================================

        bool is_sharded() const override { return is_sharded_; }
        int local_n_kv_heads() const override { return local_n_kv_heads_; }
        int kv_head_start() const override { return kv_head_start_; }
        int local_kv_dim() const override { return kv_dim_; }

        // =====================================================================
        // CPU-Specific Methods (exposed for testing and internal use)
        // =====================================================================

        int num_layers() const override { return n_layers_; }
        int batch_size() const override { return batch_size_; }
        KVCacheLayoutMode layout_mode() const override { return layout_mode_; }
        int n_kv_heads() const override { return n_kv_heads_; }

        bool append_kv(int layer, int seq_idx, const TensorBase *new_k, const TensorBase *new_v) override;
        bool append_kv(int layer, int seq_idx, const TensorBase *new_k, const TensorBase *new_v, int num_tokens) override;

        // Bring in convenience methods from interface (would be hidden by overrides otherwise)
        using ICPUKVCache::append_kv;

        int gather_kv_batched(int layer, int num_sequences, TensorBase *out_k, TensorBase *out_v,
                              std::vector<int> &out_kv_lens) override;

        void evict_oldest(int tokens_to_evict) override;
        void evict_oldest_from_sequence(int seq_idx, int tokens_to_evict) override;

        DeviceId get_layer_device(int layer) const override;
        int get_total_evicted() const override { return total_evicted_; }
        void reset_eviction_counter() override { total_evicted_ = 0; }

        // =====================================================================
        // Typed accessors (for direct use when precision is known at compile time)
        // =====================================================================

        std::shared_ptr<TensorT> get_k_typed(int layer, int seq_idx = 0) const;
        std::shared_ptr<TensorT> get_v_typed(int layer, int seq_idx = 0) const;
        bool append_kv_typed(int layer, int seq_idx, const TensorT *new_k, const TensorT *new_v);
        bool append_kv_typed(int layer, int seq_idx, const TensorT *new_k, const TensorT *new_v, int num_tokens);

    private:
        int n_layers_;
        int batch_size_;
        int max_seq_len_;
        int n_kv_heads_;       // Total KV heads (across all ranks)
        int local_n_kv_heads_; // Local KV heads (this rank), == n_kv_heads_ if not sharded
        int kv_head_start_;    // Starting KV head index (0 if not sharded)
        int head_dim_;
        int kv_dim_;      // local_n_kv_heads * head_dim (storage dimension)
        bool is_sharded_; // True if using local KV heads (TP enabled)
        int total_evicted_ = 0;
        KVCacheLayoutMode layout_mode_; // Memory layout mode

        // Cache storage: [n_layers][batch_size]
        std::vector<std::vector<EntryT>> entries_;

        // Device placement per layer
        std::vector<DeviceId> layer_devices_;

        // TensorFactory for NUMA-aware allocation
        std::unique_ptr<TensorFactory> tensor_factory_;

        // Helpers
        std::shared_ptr<TensorT> allocate_tensor(size_t rows, size_t cols, DeviceId device);
        void copy_append_data(TensorT *dst, const TensorT *src, int offset_tokens, int new_tokens);
        void shift_evict_data(TensorT *tensor, int tokens_to_evict, int tokens_to_keep);
        bool append_kv_impl(int layer, int seq_idx, const TensorT *new_k, const TensorT *new_v, int num_tokens);

        // Helper to initialize all entries for a layer
        void initialize_layer(int layer, DeviceId device);
    };

    // =========================================================================
    // Type Aliases
    // =========================================================================

    using CPUKVCacheFP32 = CPUKVCache<ActivationPrecision::FP32>;
    using CPUKVCacheBF16 = CPUKVCache<ActivationPrecision::BF16>;
    using CPUKVCacheFP16 = CPUKVCache<ActivationPrecision::FP16>;
    using CPUKVCacheQ8_1 = CPUKVCache<ActivationPrecision::Q8_1>;
    using CPUKVCacheQ16_1 = CPUKVCache<ActivationPrecision::Q16_1>;

    // =========================================================================
    // Factory Functions
    // =========================================================================

    /**
     * @brief Create unified KV cache with specified activation precision
     *
     * @param precision Activation precision (FP32, BF16, FP16, Q8_1, Q16_1)
     * @param mpi_ctx MPI context for NUMA-aware allocation
     * @param n_layers Number of transformer layers
     * @param batch_size Maximum batch size
     * @param max_seq_len Maximum sequence length
     * @param n_kv_heads Number of KV heads (GQA)
     * @param head_dim Dimension per head
     * @param device Device identifier for all layers (CPU by default)
     * @param layout_mode Memory layout mode (POSITION_MAJOR or HEAD_MAJOR)
     */
    std::unique_ptr<ICPUKVCache> createCPUKVCache(
        ActivationPrecision precision,
        const MPIContext &mpi_ctx,
        int n_layers, int batch_size, int max_seq_len,
        int n_kv_heads, int head_dim,
        DeviceId device = DeviceId::cpu(),
        KVCacheLayoutMode layout_mode = KVCacheLayoutMode::POSITION_MAJOR);

    /**
     * @brief Create unified KV cache with per-layer device placement
     */
    std::unique_ptr<ICPUKVCache> createCPUKVCache(
        ActivationPrecision precision,
        const MPIContext &mpi_ctx,
        int n_layers, int batch_size, int max_seq_len,
        int n_kv_heads, int head_dim,
        const std::vector<int> &attention_devices,
        KVCacheLayoutMode layout_mode = KVCacheLayoutMode::POSITION_MAJOR);

    // ============================================================================
    // Sharded KV Cache Factory Functions (for Tensor Parallelism)
    // ============================================================================

    /**
     * @brief Create sharded unified KV cache for tensor parallelism
     *
     * Each MPI rank stores only its assigned KV heads, reducing memory
     * proportionally to the number of ranks.
     *
     * @param precision Activation precision (FP32, BF16, FP16, Q8_1)
     * @param mpi_ctx MPI context for device affinity
     * @param n_layers Number of transformer layers
     * @param batch_size Maximum batch size
     * @param max_seq_len Maximum sequence length
     * @param n_kv_heads Total number of KV heads (before sharding)
     * @param local_n_kv_heads Number of KV heads stored on this rank
     * @param kv_head_start Starting KV head index for this rank
     * @param head_dim Dimension per head
     * @param device Device identifier for all layers (CPU by default)
     * @param layout_mode Memory layout mode (POSITION_MAJOR or HEAD_MAJOR)
     */
    std::unique_ptr<ICPUKVCache> createShardedCPUKVCache(
        ActivationPrecision precision,
        const MPIContext &mpi_ctx,
        int n_layers, int batch_size, int max_seq_len,
        int n_kv_heads, int local_n_kv_heads, int kv_head_start,
        int head_dim, DeviceId device = DeviceId::cpu(),
        KVCacheLayoutMode layout_mode = KVCacheLayoutMode::POSITION_MAJOR);

    /**
     * @brief Create sharded unified KV cache with per-layer device placement
     */
    std::unique_ptr<ICPUKVCache> createShardedCPUKVCache(
        ActivationPrecision precision,
        const MPIContext &mpi_ctx,
        int n_layers, int batch_size, int max_seq_len,
        int n_kv_heads, int local_n_kv_heads, int kv_head_start,
        int head_dim,
        const std::vector<int> &attention_devices,
        KVCacheLayoutMode layout_mode = KVCacheLayoutMode::POSITION_MAJOR);

} // namespace llaminar2
