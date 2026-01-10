/**
 * @file IKVCache.h
 * @brief Unified KV cache interface for CPU and GPU implementations
 *
 * This interface abstracts the common operations between CPU (CPUKVCache)
 * and GPU (CUDARingKVCache) implementations, allowing stages to work with
 * either cache type through a single pointer.
 */

#pragma once

#include "../execution/RuntimeConfig.h" // ActivationPrecision
#include "../tensors/ITensor.h"         // Lightweight interface (no MPI)
#include "../tensors/TensorLayout.h"
#include <vector>

namespace llaminar2
{

    /**
     * @brief Unified interface for KV cache implementations
     *
     * Both CPU (ICPUKVCache) and GPU (ICUDARingKVCache) caches inherit
     * from this interface. Stages that only need to query cache state or
     * append data can use IKVCache* without knowing the underlying implementation.
     */
    class IKVCache
    {
    public:
        virtual ~IKVCache() = default;

        // =================================================================
        // Query Operations
        // =================================================================

        /**
         * @brief Get the precision/data type of cached K/V tensors
         * @return ActivationPrecision (FP32, BF16, FP16, Q8_1, etc.)
         */
        virtual ActivationPrecision precision() const = 0;

        /**
         * @brief Get number of cached tokens for a layer/sequence
         * @param layer Layer index
         * @param seq_idx Sequence index (default 0 for single-sequence)
         * @return Number of tokens currently cached
         */
        virtual int get_cached_tokens(int layer, int seq_idx = 0) const = 0;

        /**
         * @brief Get maximum sequence length
         */
        virtual int max_seq_len() const = 0;

        /**
         * @brief Get number of layers
         */
        virtual int n_layers() const = 0;

        /**
         * @brief Get KV cache tensor layout
         *
         * @return TensorLayout indicating memory ordering
         *         Default is KV_POS_HEAD_DIM (position-major)
         */
        virtual TensorLayout kv_layout() const { return TensorLayout::KV_POS_HEAD_DIM; }

        // =================================================================
        // ITensor Access (unified CPU/GPU interface)
        // =================================================================

        /**
         * @brief Get both K and V cache tensors for attention computation
         *
         * This is the preferred interface for accessing KV cache - fetches both
         * K and V in a single call, which aligns with GPU batch operations and
         * enables potential optimizations.
         *
         * For CPU caches: returns TensorBase* pointers (which inherit ITensor)
         * For GPU caches: returns CUDATensorBase* pointers (which inherit ITensor)
         *
         * @param layer Layer index
         * @param seq_idx Sequence index (default 0)
         * @param out_k Output: pointer to K cache tensor
         * @param out_v Output: pointer to V cache tensor
         * @param out_kv_len Output: number of cached tokens (optional, can be nullptr)
         * @return true on success, false if layer/seq_idx invalid
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

        /**
         * @brief Get K cache tensor as ITensor for a layer/sequence
         *
         * @deprecated Use get_kv() instead for unified access to both K and V.
         *             This method will be removed in a future version.
         *
         * @param layer Layer index
         * @param seq_idx Sequence index (default 0)
         * @return ITensor* to K cache, or nullptr if not available
         */
        virtual ITensor *get_k(int layer, int seq_idx = 0)
        {
            (void)layer;
            (void)seq_idx;
            return nullptr;
        }
        virtual const ITensor *get_k(int layer, int seq_idx = 0) const
        {
            (void)layer;
            (void)seq_idx;
            return nullptr;
        }

        /**
         * @brief Get V cache tensor as ITensor for a layer/sequence
         *
         * @deprecated Use get_kv() instead for unified access to both K and V.
         *             This method will be removed in a future version.
         *
         * @param layer Layer index
         * @param seq_idx Sequence index (default 0)
         * @return ITensor* to V cache, or nullptr if not available
         */
        virtual ITensor *get_v(int layer, int seq_idx = 0)
        {
            (void)layer;
            (void)seq_idx;
            return nullptr;
        }
        virtual const ITensor *get_v(int layer, int seq_idx = 0) const
        {
            (void)layer;
            (void)seq_idx;
            return nullptr;
        }

        // =================================================================
        // Append Operations
        // =================================================================

        /**
         * @brief Append K/V tensors to cache
         *
         * For CPU caches: reads from tensor host data
         * For GPU caches: reads from tensor GPU data (must be on device)
         *
         * @param layer Layer index
         * @param seq_idx Sequence index
         * @param K Key tensor to append
         * @param V Value tensor to append
         * @param num_tokens Number of tokens to append
         * @return true on success
         */
        virtual bool append(int layer, int seq_idx, const ITensor *K, const ITensor *V, int num_tokens) = 0;

        // Convenience overload for seq_idx=0
        bool append(int layer, const ITensor *K, const ITensor *V, int num_tokens)
        {
            return append(layer, 0, K, V, num_tokens);
        }

        // =================================================================
        // Clear Operations
        // =================================================================

        /**
         * @brief Clear all cached tokens across all layers and sequences
         */
        virtual void clear() = 0;

        /**
         * @brief Clear a specific sequence across all layers
         * @param seq_idx Sequence index to clear
         *
         * Default implementation clears sequence in each layer.
         * Subclasses may override for more efficient implementation.
         */
        virtual void clear_sequence(int seq_idx)
        {
            for (int layer = 0; layer < n_layers(); ++layer)
            {
                clear_sequence(layer, seq_idx);
            }
        }

        /**
         * @brief Clear a specific sequence in a specific layer
         * @param layer Layer index
         * @param seq_idx Sequence index to clear
         *
         * This is the primitive operation that subclasses should implement.
         */
        virtual void clear_sequence(int layer, int seq_idx) = 0;

        /**
         * @brief Clear all sequences in a specific layer
         * @param layer Layer index
         */
        virtual void clear_layer(int layer) = 0;

        // =================================================================
        // Batched Operations
        // =================================================================

        /**
         * @brief Gather K/V from multiple sequences for batched attention
         *
         * Copies K/V from sequences [0..num_seqs-1] into contiguous output
         * tensors with padding to max_kv_len.
         *
         * Output layout: [num_seqs * max_kv_len, kv_dim]
         *
         * @param layer Layer index
         * @param num_sequences Number of sequences to gather
         * @param out_k Output K tensor
         * @param out_v Output V tensor
         * @param out_kv_lens Output: per-sequence kv_lens (size = num_seqs)
         * @return Maximum kv_len found, or -1 on error
         */
        virtual int gather_kv_batched(
            int layer,
            int num_sequences,
            ITensor *out_k,
            ITensor *out_v,
            std::vector<int> &out_kv_lens)
        {
            // Default: not supported (for caches that don't implement batched gather)
            (void)layer;
            (void)num_sequences;
            (void)out_k;
            (void)out_v;
            (void)out_kv_lens;
            return -1;
        }

        // =================================================================
        // Sharding Info (for tensor parallelism)
        // =================================================================

        /**
         * @brief Check if cache is sharded across ranks
         * @return true if sharded, false otherwise (default: not sharded)
         */
        virtual bool is_sharded() const { return false; }

        /**
         * @brief Get local number of KV heads (for sharding)
         * @return Number of KV heads on this rank (default: 0)
         */
        virtual int local_n_kv_heads() const { return 0; }

        /**
         * @brief Get local KV dimension (local_n_kv_heads * head_dim)
         * @return Local KV dimension (default: 0)
         */
        virtual int local_kv_dim() const { return 0; }

        /**
         * @brief Get starting KV head index for this rank
         * @return Starting KV head index (default: 0)
         */
        virtual int kv_head_start() const { return 0; }
    };

} // namespace llaminar2
