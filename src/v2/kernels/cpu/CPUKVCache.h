/**
 * @file CPUKVCache.h
 * @brief Shared CPU KV cache interface and precision→tensor mappings.
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
        virtual ActivationPrecision k_precision() const = 0;
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
         * @brief Map ActivationPrecision to tensor type for KV cache, with allocation.
         *
         * Each specialization provides:
         *   - Type: the concrete tensor class
         *   - allocate(): creates a shared_ptr<Type> for the given shape/device
         *   - row_bytes(): bytes per complete row (all heads, position-major stride)
         *   - head_bytes(): bytes per single head slice (head-major copy unit)
         *
         * row_bytes() and head_bytes() enable precision-agnostic byte-level copy
         * operations in the ring buffer, eliminating per-precision if-constexpr chains.
         *
         * The allocate() signature is uniform across all precisions. Parameters
         * that a given precision does not need (e.g. head_dim for FP32) are
         * simply ignored, keeping the call site a single line.
         */
        template <ActivationPrecision P>
        struct CPUKVCacheTensor;

        template <>
        struct CPUKVCacheTensor<ActivationPrecision::FP32>
        {
            using Type = FP32Tensor;
            static std::shared_ptr<Type> allocate(TensorFactory &factory, size_t rows, size_t cols, int /*head_dim*/, DeviceId device)
            {
                return factory.createFP32({rows, cols}, device);
            }
            static size_t row_bytes(const Type *, int kv_dim, int) { return static_cast<size_t>(kv_dim) * sizeof(float); }
            static size_t head_bytes(const Type *, int, int head_dim) { return static_cast<size_t>(head_dim) * sizeof(float); }
        };

        template <>
        struct CPUKVCacheTensor<ActivationPrecision::BF16>
        {
            using Type = BF16Tensor;
            static std::shared_ptr<Type> allocate(TensorFactory &factory, size_t rows, size_t cols, int /*head_dim*/, DeviceId /*device*/)
            {
                return factory.createBF16({rows, cols});
            }
            static size_t row_bytes(const Type *, int kv_dim, int) { return static_cast<size_t>(kv_dim) * sizeof(uint16_t); }
            static size_t head_bytes(const Type *, int, int head_dim) { return static_cast<size_t>(head_dim) * sizeof(uint16_t); }
        };

        template <>
        struct CPUKVCacheTensor<ActivationPrecision::FP16>
        {
            using Type = FP16Tensor;
            static std::shared_ptr<Type> allocate(TensorFactory &factory, size_t rows, size_t cols, int /*head_dim*/, DeviceId /*device*/)
            {
                return factory.createFP16({rows, cols});
            }
            static size_t row_bytes(const Type *, int kv_dim, int) { return static_cast<size_t>(kv_dim) * sizeof(uint16_t); }
            static size_t head_bytes(const Type *, int, int head_dim) { return static_cast<size_t>(head_dim) * sizeof(uint16_t); }
        };

        template <>
        struct CPUKVCacheTensor<ActivationPrecision::Q8_1>
        {
            using Type = Q8_1Tensor;
            static std::shared_ptr<Type> allocate(TensorFactory &factory, size_t rows, size_t cols, int /*head_dim*/, DeviceId device)
            {
                return factory.createQ8_1({rows, cols}, device);
            }
            static size_t row_bytes(const Type *, int kv_dim, int)
            {
                return ((static_cast<size_t>(kv_dim) + Q8_1Block::BLOCK_SIZE - 1) / Q8_1Block::BLOCK_SIZE) * sizeof(Q8_1Block);
            }
            static size_t head_bytes(const Type *, int, int head_dim)
            {
                return ((static_cast<size_t>(head_dim) + Q8_1Block::BLOCK_SIZE - 1) / Q8_1Block::BLOCK_SIZE) * sizeof(Q8_1Block);
            }
        };

        template <>
        struct CPUKVCacheTensor<ActivationPrecision::Q16_1>
        {
            using Type = Q16_1Tensor;
            static std::shared_ptr<Type> allocate(TensorFactory &factory, size_t rows, size_t cols, int head_dim, DeviceId device)
            {
                return factory.createQ16_1({rows, cols}, optimal_q16_block_size(head_dim), device);
            }
            static size_t row_bytes(const Type *t, int kv_dim, int)
            {
                const size_t be = q16_block_size_elements(t->q16_block_size());
                const size_t bb = q16_block_size_bytes(t->q16_block_size());
                return ((static_cast<size_t>(kv_dim) + be - 1) / be) * bb;
            }
            static size_t head_bytes(const Type *t, int, int head_dim)
            {
                const size_t be = q16_block_size_elements(t->q16_block_size());
                const size_t bb = q16_block_size_bytes(t->q16_block_size());
                return ((static_cast<size_t>(head_dim) + be - 1) / be) * bb;
            }
        };

        template <>
        struct CPUKVCacheTensor<ActivationPrecision::TQ4>
        {
            using Type = TQ4Tensor;
            static std::shared_ptr<Type> allocate(TensorFactory & /*factory*/, size_t rows, size_t cols, int head_dim, DeviceId device)
            {
                return std::make_shared<TQ4Tensor>(std::vector<size_t>{rows, cols}, head_dim, device);
            }
            static size_t row_bytes(const Type *t, int kv_dim, int head_dim)
            {
                // Use kv_dim parameter (not tensor's blocks_per_row) so the stride is always
                // kv_dim-wide.  HEAD_MAJOR internal tensors have cols=head_dim, but the
                // row stride for source/dest copies must span all heads.
                return static_cast<size_t>(kv_dim / head_dim) * t->block_bytes();
            }
            static size_t head_bytes(const Type *t, int, int) { return t->block_bytes(); } // 1 block per head
        };

        // ----- TQ8 (TurboQuant 8-bit for K-projection cache) -----
        template <>
        struct CPUKVCacheTensor<ActivationPrecision::TQ8>
        {
            using Type = TQ8Tensor;
            static std::shared_ptr<Type> allocate(TensorFactory & /*factory*/, size_t rows, size_t cols, int head_dim, DeviceId device)
            {
                return std::make_shared<TQ8Tensor>(std::vector<size_t>{rows, cols}, head_dim, device);
            }
            static size_t row_bytes(const Type *t, int kv_dim, int head_dim)
            {
                return static_cast<size_t>(kv_dim / head_dim) * t->block_bytes();
            }
            static size_t head_bytes(const Type *t, int, int) { return t->block_bytes(); }
        };

    } // namespace detail

} // namespace llaminar2
