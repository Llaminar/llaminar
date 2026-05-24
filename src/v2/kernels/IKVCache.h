/**
 * @file IKVCache.h
 * @brief Unified KV cache interface for CPU and GPU implementations
 *
 * This interface abstracts the common operations between CPU (CPUKVCache)
 * and GPU (CUDARingKVCache) implementations, allowing stages to work with
 * either cache type through a single pointer.
 */

#pragma once

#include "../execution/config/RuntimeConfig.h" // ActivationPrecision
#include "../tensors/ITensor.h"                // Lightweight interface (no MPI)
#include "../tensors/TensorLayout.h"
#include <vector>

namespace llaminar2
{
    class TurboQuantContext; // Forward declaration for KVReadParams

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
         * @brief Get the precision/data type of cached K tensors
         * @return ActivationPrecision (FP32, BF16, FP16, Q8_1, TQ8, etc.)
         */
        virtual ActivationPrecision k_precision() const = 0;

        /**
         * @brief Get the precision/data type of cached V tensors
         *
         * Defaults to k_precision() for symmetric caches. Override for
         * asymmetric K/V storage (e.g., TQ8 for K, TQ4 for V).
         *
         * @return ActivationPrecision (FP32, BF16, FP16, Q8_1, TQ4, etc.)
         */
        virtual ActivationPrecision v_precision() const { return k_precision(); }

        /**
         * @brief Get the precision of cached K/V tensors (deprecated)
         *
         * @deprecated Use k_precision() and v_precision() instead.
         *             This returns k_precision() and is incorrect for asymmetric caches.
         * @return ActivationPrecision of K cache
         */
        [[deprecated("Use k_precision() and v_precision() instead")]]
        ActivationPrecision precision() const
        {
            return k_precision();
        }

        /**
         * @brief Get number of cached tokens for a layer/sequence
         * @param layer Layer index (global, will be remapped if first_layer_index > 0)
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
         * @brief Get the first layer index for this cache (Pipeline Parallelism)
         *
         * When > 0, this cache handles layers [first_layer_index, first_layer_index + n_layers()).
         * Incoming global layer indices are remapped: local_idx = global_idx - first_layer_index.
         *
         * @return First layer index (0 for non-PP caches)
         */
        virtual int first_layer_index() const { return 0; }

        /**
         * @brief Remap a global layer index to a local cache index
         *
         * Handles Pipeline Parallelism where each device's cache covers a subset of layers.
         * For example, PP stage 1 with layers [12, 23] uses cache with first_layer_index=12.
         * Global layer 15 maps to local cache index 3.
         *
         * @param global_layer The global layer index (0-based across all layers)
         * @return Local cache index (0-based within this cache), or -1 if out of range
         */
        int remapLayerIndex(int global_layer) const
        {
            int local_idx = global_layer - first_layer_index();
            if (local_idx < 0 || local_idx >= n_layers())
            {
                return -1; // Out of range for this cache
            }
            return local_idx;
        }

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

        /**
         * @brief Stream-aware append for GPU graph capture compatibility
         *
         * GPU KV cache implementations override this to dispatch the append
         * kernel on the specified stream instead of the default stream (0).
         * This is critical for GPU graph capture where all operations must
         * execute on the same stream.
         *
         * CPU caches ignore the stream and forward to the regular append.
         *
         * @param layer Layer index
         * @param seq_idx Sequence index
         * @param K Key tensor to append
         * @param V Value tensor to append
         * @param num_tokens Number of tokens to append
         * @param gpu_stream Opaque GPU stream pointer (hipStream_t or cudaStream_t)
         * @return true on success
         */
        virtual bool appendWithStream(int layer, int seq_idx, const ITensor *K, const ITensor *V,
                                      int num_tokens, void *gpu_stream)
        {
            (void)gpu_stream;
            return append(layer, seq_idx, K, V, num_tokens);
        }

        // =================================================================
        // Graph Capture Support
        // =================================================================

        /**
         * @brief Check if this cache supports GPU graph capture for append
         *
         * GPU caches that allocate device-side head parameter buffers return true.
         * When true, the KVCacheAppendStage can mark itself as graph-capturable.
         *
         * @return true if setDynamicHead() and advanceHead() are implemented
         */
        virtual bool isGraphCaptureReady() const { return false; }

        /**
         * @brief Copy current head position to device buffer for graph replay
         *
         * Updates the pinned host buffer with the current ring buffer head
         * position and issues an H2D copy on the given stream. This ensures
         * the device buffer has the correct value before the graph replays
         * the captured ring_append_kernel.
         *
         * For graph mode: the captured H2D in the graph also re-reads from
         * the pinned host buffer, making the explicit copy here a no-op
         * (but it's needed for stream-only mode without graph capture).
         *
         * @param layer Layer index
         * @param seq_idx Sequence index
         * @param gpu_stream GPU stream for the H2D copy
         */
        virtual void setDynamicHead(int layer, int seq_idx, void *gpu_stream)
        {
            (void)layer;
            (void)seq_idx;
            (void)gpu_stream;
        }

        /**
         * @brief Update device-side dequant params for graph-capturable read.
         *
         * TQ caches override this to write incremental dequant parameters
         * (ring_pos, out_offset, rope_position) to a pinned host buffer.
         * The captured H2D memcpy in the graph re-reads from pinned host
         * on replay, so the kernel receives updated values.
         *
         * Called from AttentionComputeStage::updateDynamicParams() before
         * graph replay. Not needed for non-TQ caches.
         */
        virtual void setDynamicDequantParams(int layer, int seq_idx,
                                             float rope_theta, int position_start,
                                             void *gpu_stream)
        {
            (void)layer;
            (void)seq_idx;
            (void)rope_theta;
            (void)position_start;
            (void)gpu_stream;
        }

        /**
         * @brief Advance ring buffer head position (host-side bookkeeping)
         *
         * During graph replay, execute() is not called on the KVCacheAppendStage.
         * This method performs the host-side state updates that append() would
         * normally do: advancing the head position, updating the token count,
         * and handling auto-eviction when the ring buffer is full.
         *
         * Must be called AFTER setDynamicHead() to maintain correct ordering:
         * setDynamicHead copies the CURRENT head to device, then advanceHead
         * moves the head forward for the NEXT iteration.
         *
         * @param layer Layer index
         * @param seq_idx Sequence index
         * @param num_tokens Number of tokens appended (typically 1 during decode)
         */
        virtual void advanceHead(int layer, int seq_idx, int num_tokens)
        {
            (void)layer;
            (void)seq_idx;
            (void)num_tokens;
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

        // =================================================================
        // Converted KV Access (dequant-on-read with optional fused RoPE)
        // =================================================================

        /**
         * @brief Parameters for fused RoPE during KV cache read.
         *
         * When rope_theta > 0, RoPE is applied to K inline during dequantization.
         * This is "RoPE-on-read": K is stored WITHOUT position embeddings in the cache,
         * and the embeddings are applied lazily when K is consumed by attention.
         *
         * Benefits:
         * - Fused dequant+RoPE eliminates a separate RoPE pass over K
         * - Position-free cache enables future speculative decoding
         * - For quantized caches, avoids dequant→RoPE→requant roundtrip
         */
        struct KVReadParams
        {
            float rope_theta = 0.0f;                           ///< RoPE frequency base (<=0 disables RoPE)
            int position_start = 0;                            ///< RoPE position of the first cached token
            int n_kv_heads = 0;                                ///< Number of KV heads
            int head_dim = 0;                                  ///< Dimension per attention head
            int rope_dim = 0;                                  ///< Number of dimensions to rotate per head (0 = full head_dim)
            const TurboQuantContext *turboquant_ctx = nullptr; ///< Required for TQ cache dequant (optional otherwise)
            void *gpu_stream = nullptr;                        ///< Optional GPU stream for fused conversion/read kernels
        };

        /**
         * @brief Get K/V converted to a target precision with optional fused RoPE.
         *
         * Returns FP32 (or other target precision) views of the cached K/V data.
         * The cache manages internal shadow buffers and performs incremental
         * conversion — only newly appended rows are processed each call.
         *
         * When rope.rope_theta > 0, RoPE is fused into K dequantization,
         * eliminating a separate RoPE computation pass.
         *
         * @param layer Layer index
         * @param seq_idx Sequence index
         * @param target Target output precision (e.g., FP32)
         * @param out_k Output: pointer to converted K tensor
         * @param out_v Output: pointer to converted V tensor
         * @param out_kv_len Output: number of cached tokens (optional)
         * @param rope Optional RoPE parameters for fused application on K
         * @return true on success
         */
        virtual bool get_kv_converted(int layer, int seq_idx,
                                      ActivationPrecision target,
                                      ITensor **out_k, ITensor **out_v,
                                      int *out_kv_len = nullptr,
                                      const KVReadParams *rope = nullptr)
        {
            // Default: ignore target precision and RoPE, return raw cache tensors
            (void)target;
            (void)rope;
            return get_kv(layer, seq_idx, out_k, out_v, out_kv_len);
        }

        /**
         * @brief Templated convenience for get_kv_converted().
         *
         * Usage:
         *   cache->get_kv<ActivationPrecision::FP32>(layer, seq, &k, &v, &len, &rope);
         *
         * @tparam Target Output precision (compile-time ActivationPrecision)
         */
        template <ActivationPrecision Target>
        bool get_kv(int layer, int seq_idx,
                    ITensor **out_k, ITensor **out_v,
                    int *out_kv_len = nullptr,
                    const KVReadParams *rope = nullptr)
        {
            return get_kv_converted(layer, seq_idx, Target, out_k, out_v, out_kv_len, rope);
        }

        /// Convenience: templated get_kv with seq_idx=0
        template <ActivationPrecision Target>
        bool get_kv(int layer,
                    ITensor **out_k, ITensor **out_v,
                    int *out_kv_len = nullptr,
                    const KVReadParams *rope = nullptr)
        {
            return get_kv_converted(layer, 0, Target, out_k, out_v, out_kv_len, rope);
        }
    };

} // namespace llaminar2
