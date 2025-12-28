/**
 * @file UnifiedKVCache.h
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

#include "Tensors.h"
#include "TensorFactory.h"
#include "../utils/MPIContext.h"
#include "../execution/RuntimeConfig.h"
#include <vector>
#include <memory>

namespace llaminar2
{

    // =========================================================================
    // IUnifiedKVCache Interface
    // =========================================================================

    /**
     * @brief Abstract interface for type-erased unified KV cache access
     *
     * Enables polymorphic use when precision is determined at runtime.
     * Supports both single-sequence (batch_size=1) and batched modes.
     */
    class IUnifiedKVCache
    {
    public:
        virtual ~IUnifiedKVCache() = default;

        // Metadata
        virtual ActivationPrecision precision() const = 0;
        virtual int num_layers() const = 0;
        virtual int batch_size() const = 0;
        virtual int max_seq_len() const = 0;

        // Sharding (Tensor Parallelism) info
        virtual bool is_sharded() const = 0;      ///< True if using local KV heads
        virtual int n_kv_heads() const = 0;       ///< Total KV heads (across all ranks)
        virtual int local_n_kv_heads() const = 0; ///< Local KV heads (this rank)
        virtual int kv_head_start() const = 0;    ///< Starting KV head index
        virtual int local_kv_dim() const = 0;     ///< local_n_kv_heads * head_dim

        // Per-sequence token tracking
        virtual int get_cached_tokens(int layer, int seq_idx = 0) const = 0;

        // Polymorphic tensor access (returns base type)
        virtual TensorBase *get_k_base(int layer, int seq_idx = 0) = 0;
        virtual const TensorBase *get_k_base(int layer, int seq_idx = 0) const = 0;
        virtual TensorBase *get_v_base(int layer, int seq_idx = 0) = 0;
        virtual const TensorBase *get_v_base(int layer, int seq_idx = 0) const = 0;

        virtual std::shared_ptr<TensorBase> get_k(int layer, int seq_idx = 0) const = 0;
        virtual std::shared_ptr<TensorBase> get_v(int layer, int seq_idx = 0) const = 0;

        /**
         * @brief Append K/V to cache for a sequence
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
         * @brief Gather K/V from multiple cache slots into batched output tensors
         *
         * This method copies K/V from cache slots [0..batch_size-1] into contiguous
         * output tensors suitable for batched attention computation.
         *
         * For batched decode, each sequence may have different kv_len. This method
         * handles variable lengths by:
         * 1. Finding the maximum kv_len across all sequences
         * 2. Copying each sequence's K/V to its slot (with implicit zero-padding)
         * 3. Returning per-sequence kv_lens for masking
         *
         * Output tensor layout: [batch_size * max_kv_len, kv_dim]
         * where each sequence's K/V occupies [seq_idx * max_kv_len, (seq_idx+1) * max_kv_len)
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

        // Cache management
        virtual void clear() = 0;
        virtual void clear_sequence(int seq_idx) = 0;
        virtual void clear_layer(int layer) = 0;
        virtual void evict_oldest(int tokens_to_evict) = 0;
        virtual void evict_oldest_from_sequence(int seq_idx, int tokens_to_evict) = 0;

        // Device placement
        virtual int get_layer_device(int layer) const = 0;

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
        struct UnifiedKVCacheTensor;

        template <>
        struct UnifiedKVCacheTensor<ActivationPrecision::FP32>
        {
            using Type = FP32Tensor;
        };

        template <>
        struct UnifiedKVCacheTensor<ActivationPrecision::BF16>
        {
            using Type = BF16Tensor;
        };

        template <>
        struct UnifiedKVCacheTensor<ActivationPrecision::FP16>
        {
            using Type = FP16Tensor;
        };

        template <>
        struct UnifiedKVCacheTensor<ActivationPrecision::Q8_1>
        {
            using Type = Q8_1Tensor;
        };

        template <>
        struct UnifiedKVCacheTensor<ActivationPrecision::Q16_1>
        {
            using Type = Q16_1Tensor;
        };
    } // namespace detail

    // =========================================================================
    // UnifiedKVCacheEntry
    // =========================================================================

    /**
     * @brief Per-layer, per-sequence KV cache entry with typed tensors
     */
    template <ActivationPrecision Precision>
    struct UnifiedKVCacheEntry
    {
        using TensorT = typename detail::UnifiedKVCacheTensor<Precision>::Type;

        std::shared_ptr<TensorT> K; // [max_seq_len, n_kv_heads * head_dim] pre-allocated
        std::shared_ptr<TensorT> V; // [max_seq_len, n_kv_heads * head_dim] pre-allocated
        int cached_tokens = 0;      // Number of tokens currently cached
    };

    // =========================================================================
    // UnifiedKVCache Template Class
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
     */
    template <ActivationPrecision Precision = ActivationPrecision::FP32>
    class UnifiedKVCache : public IUnifiedKVCache
    {
    public:
        using TensorT = typename detail::UnifiedKVCacheTensor<Precision>::Type;
        using EntryT = UnifiedKVCacheEntry<Precision>;

        /**
         * @brief Construct unified KV cache
         *
         * @param mpi_ctx MPI context for NUMA-aware allocation
         * @param n_layers Number of transformer layers
         * @param batch_size Number of sequences (1 for single-sequence mode)
         * @param max_seq_len Maximum sequence length (cache capacity per sequence)
         * @param n_kv_heads Number of KV heads (GQA)
         * @param head_dim Dimension per head
         * @param device_idx Default device for all layers (-1 = CPU)
         */
        UnifiedKVCache(const MPIContext &mpi_ctx, int n_layers, int batch_size, int max_seq_len,
                       int n_kv_heads, int head_dim, int device_idx = -1);

        /**
         * @brief Construct unified KV cache with per-layer device placement
         */
        UnifiedKVCache(const MPIContext &mpi_ctx, int n_layers, int batch_size, int max_seq_len,
                       int n_kv_heads, int head_dim, const std::vector<int> &attention_devices);

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
         * @param device_idx Default device for all layers (-1 = CPU)
         */
        UnifiedKVCache(const MPIContext &mpi_ctx, int n_layers, int batch_size, int max_seq_len,
                       int n_kv_heads, int local_n_kv_heads, int kv_head_start,
                       int head_dim, int device_idx = -1);

        /**
         * @brief Construct unified KV cache with sharded KV heads and per-layer device placement
         */
        UnifiedKVCache(const MPIContext &mpi_ctx, int n_layers, int batch_size, int max_seq_len,
                       int n_kv_heads, int local_n_kv_heads, int kv_head_start,
                       int head_dim, const std::vector<int> &attention_devices);

        // IUnifiedKVCache interface implementation
        ActivationPrecision precision() const override { return Precision; }
        int num_layers() const override { return n_layers_; }
        int batch_size() const override { return batch_size_; }
        int max_seq_len() const override { return max_seq_len_; }

        int get_cached_tokens(int layer, int seq_idx = 0) const override;

        TensorBase *get_k_base(int layer, int seq_idx = 0) override;
        const TensorBase *get_k_base(int layer, int seq_idx = 0) const override;
        TensorBase *get_v_base(int layer, int seq_idx = 0) override;
        const TensorBase *get_v_base(int layer, int seq_idx = 0) const override;

        std::shared_ptr<TensorBase> get_k(int layer, int seq_idx = 0) const override;
        std::shared_ptr<TensorBase> get_v(int layer, int seq_idx = 0) const override;

        bool append_kv(int layer, int seq_idx, const TensorBase *new_k, const TensorBase *new_v) override;
        bool append_kv(int layer, int seq_idx, const TensorBase *new_k, const TensorBase *new_v, int num_tokens) override;

        // Bring in convenience methods from interface (would be hidden by overrides otherwise)
        using IUnifiedKVCache::append_kv;

        int gather_kv_batched(int layer, int num_sequences, TensorBase *out_k, TensorBase *out_v,
                              std::vector<int> &out_kv_lens) override;

        void clear() override;
        void clear_sequence(int seq_idx) override;
        void clear_layer(int layer) override;
        void evict_oldest(int tokens_to_evict) override;
        void evict_oldest_from_sequence(int seq_idx, int tokens_to_evict) override;

        int get_layer_device(int layer) const override;
        int get_total_evicted() const override { return total_evicted_; }
        void reset_eviction_counter() override { total_evicted_ = 0; }

        // =====================================================================
        // Sharding (Tensor Parallelism) Accessors
        // =====================================================================

        /**
         * @brief Check if cache is sharded (local heads only)
         */
        bool is_sharded() const override { return is_sharded_; }

        /**
         * @brief Get total KV heads (across all ranks)
         */
        int n_kv_heads() const override { return n_kv_heads_; }

        /**
         * @brief Get local KV heads (this rank's heads, == n_kv_heads if not sharded)
         */
        int local_n_kv_heads() const override { return local_n_kv_heads_; }

        /**
         * @brief Get starting KV head index for this rank (0 if not sharded)
         */
        int kv_head_start() const override { return kv_head_start_; }

        /**
         * @brief Get local KV dimension (local_n_kv_heads * head_dim)
         */
        int local_kv_dim() const override { return kv_dim_; }

        // =====================================================================
        // Typed accessors (for direct use when precision is known at compile time)
        // =====================================================================

        /**
         * @brief Get typed K cache for specific layer and sequence
         */
        std::shared_ptr<TensorT> get_k_typed(int layer, int seq_idx = 0) const;

        /**
         * @brief Get typed V cache for specific layer and sequence
         */
        std::shared_ptr<TensorT> get_v_typed(int layer, int seq_idx = 0) const;

        /**
         * @brief Append new K/V to cache (typed version - zero-copy for matching precision)
         */
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

        // Cache storage: [n_layers][batch_size]
        std::vector<std::vector<EntryT>> entries_;

        // Device placement per layer
        std::vector<int> layer_devices_;

        // TensorFactory for NUMA-aware allocation
        std::unique_ptr<TensorFactory> tensor_factory_;

        // Helpers
        std::shared_ptr<TensorT> allocate_tensor(size_t rows, size_t cols, int device_idx);
        void copy_append_data(TensorT *dst, const TensorT *src, int offset_tokens, int new_tokens);
        void shift_evict_data(TensorT *tensor, int tokens_to_evict, int tokens_to_keep);
        bool append_kv_impl(int layer, int seq_idx, const TensorT *new_k, const TensorT *new_v, int num_tokens);

        // Helper to initialize all entries for a layer
        void initialize_layer(int layer, int device_idx);
    };

    // =========================================================================
    // Type Aliases
    // =========================================================================

    using UnifiedKVCacheFP32 = UnifiedKVCache<ActivationPrecision::FP32>;
    using UnifiedKVCacheBF16 = UnifiedKVCache<ActivationPrecision::BF16>;
    using UnifiedKVCacheFP16 = UnifiedKVCache<ActivationPrecision::FP16>;
    using UnifiedKVCacheQ8_1 = UnifiedKVCache<ActivationPrecision::Q8_1>;
    using UnifiedKVCacheQ16_1 = UnifiedKVCache<ActivationPrecision::Q16_1>;

    // =========================================================================
    // Factory Functions
    // =========================================================================

    /**
     * @brief Create unified KV cache with specified activation precision
     */
    std::unique_ptr<IUnifiedKVCache> createUnifiedKVCache(
        ActivationPrecision precision,
        const MPIContext &mpi_ctx,
        int n_layers, int batch_size, int max_seq_len,
        int n_kv_heads, int head_dim,
        int device_idx = -1);

    /**
     * @brief Create unified KV cache with per-layer device placement
     */
    std::unique_ptr<IUnifiedKVCache> createUnifiedKVCache(
        ActivationPrecision precision,
        const MPIContext &mpi_ctx,
        int n_layers, int batch_size, int max_seq_len,
        int n_kv_heads, int head_dim,
        const std::vector<int> &attention_devices);

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
     * @param device_idx Device index for all layers (-1 for CPU)
     */
    std::unique_ptr<IUnifiedKVCache> createShardedKVCache(
        ActivationPrecision precision,
        const MPIContext &mpi_ctx,
        int n_layers, int batch_size, int max_seq_len,
        int n_kv_heads, int local_n_kv_heads, int kv_head_start,
        int head_dim, int device_idx = -1);

    /**
     * @brief Create sharded unified KV cache with per-layer device placement
     */
    std::unique_ptr<IUnifiedKVCache> createShardedKVCache(
        ActivationPrecision precision,
        const MPIContext &mpi_ctx,
        int n_layers, int batch_size, int max_seq_len,
        int n_kv_heads, int local_n_kv_heads, int kv_head_start,
        int head_dim,
        const std::vector<int> &attention_devices);

} // namespace llaminar2
