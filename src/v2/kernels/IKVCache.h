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
#include "../utils/Logger.h"
#include <cstddef>
#include <cstdint>
#include <string>
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

        /**
         * @brief Descriptor for copying a logical KV block in oldest-to-newest order.
         *
         * The token range is logical within the sequence, not a physical ring row
         * range. Implementations may remap global_layer through first_layer_index().
         */
        struct KVCacheLogicalBlockDescriptor
        {
            int layer = 0;
            int seq_idx = 0;
            int logical_token_start = 0;
            int token_count = 0;
            void *stream = nullptr;
        };

        /**
         * @brief Packed byte layout for a logical KV block payload.
         *
         * Payloads preserve native K/V precision and KV cache layout. For
         * KV_POS_HEAD_DIM, rows are packed by token. For KV_HEAD_POS_DIM, rows
         * are packed by head, then logical token, then head dimension.
         */
        struct KVCacheLogicalBlockLayout
        {
            ActivationPrecision k_precision = ActivationPrecision::FP32;
            ActivationPrecision v_precision = ActivationPrecision::FP32;
            TensorLayout layout = TensorLayout::UNKNOWN;
            int local_kv_heads = 0;
            int kv_head_start = 0;
            int head_dim = 0;
            size_t k_bytes = 0;
            size_t v_bytes = 0;
            bool device_resident = false;
        };

        /**
         * @brief Logical sequence state for cache import/export planning.
         */
        struct KVCacheSequenceState
        {
            int cached_tokens = 0;
            int implementation_head = 0;
            bool wrapped = false;
        };

        /**
         * @brief Device-resident sequence-state publication request.
         *
         * vLLM-style MTP publication derives accepted verifier rows and target
         * cache lengths in GPU memory.  A cache implementation may consume this
         * request to update its device-side sequence metadata without first
         * copying accepted counts back to the CPU.
         *
         * Long-context ring caches must treat @p target_cached_tokens_device as
         * the target valid-token count and @p accepted_state_counts_device as
         * the number of verifier rows committed from the current device-visible
         * head.  The target count alone is not enough to reconstruct a wrapped
         * ring head, so implementations need a live base-head mirror or an
         * equivalent device-side state source.
         *
         * The stream is mandatory for GPU implementations.  Work enqueued by
         * this call must be ordered after the verifier outcome producer and
         * before any graph replay that consumes the updated KV state.
         */
        struct DeviceSequenceStatePublicationRequest
        {
            int request_count = 0;
            int first_seq_idx = 0;
            const int32_t *target_cached_tokens_device = nullptr;
            const int32_t *accepted_state_counts_device = nullptr;
            const int32_t *publication_ok_flags_device = nullptr;
            void *stream = nullptr;

            bool valid() const
            {
                return request_count > 0 &&
                       first_seq_idx >= 0 &&
                       target_cached_tokens_device != nullptr &&
                       accepted_state_counts_device != nullptr &&
                       publication_ok_flags_device != nullptr &&
                       stream != nullptr;
            }
        };

        /**
         * @brief Host mirror update paired with device-resident publication.
         *
         * DeviceSequenceStatePublicationRequest updates GPU-visible KV
         * head/count mirrors on an explicit stream.  Graph signatures and legacy
         * diagnostics still read host cache state through get_cached_tokens() and
         * ring_head(), so callers that later synchronize compact verifier metadata
         * can use this host-only request to make those mirrors match the already
         * enqueued device publication.  This method must not enqueue device work.
         */
        struct HostSequenceStatePublicationRequest
        {
            int request_count = 0;
            int first_seq_idx = 0;
            std::vector<int32_t> target_cached_tokens;
            std::vector<int32_t> accepted_state_counts;
            std::vector<int32_t> publication_ok_flags;

            bool valid() const
            {
                return request_count > 0 &&
                       first_seq_idx >= 0 &&
                       static_cast<int>(target_cached_tokens.size()) == request_count &&
                       static_cast<int>(accepted_state_counts.size()) == request_count &&
                       static_cast<int>(publication_ok_flags.size()) == request_count;
            }
        };

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

        /**
         * @brief Inspect the current physical ring head for diagnostics.
         *
         * For ring caches this is the physical row that represents the oldest
         * valid token. Non-ring implementations may return 0.
         */
        virtual int ring_head(int layer, int seq_idx = 0) const
        {
            (void)layer;
            (void)seq_idx;
            return 0;
        }

        /**
         * @brief Describe the byte layout for a packed logical KV block.
         */
        virtual KVCacheLogicalBlockLayout logicalBlockLayout(int global_layer, int token_count) const
        {
            (void)global_layer;
            (void)token_count;
            return {};
        }

        /**
         * @brief Inspect logical sequence state for a global layer/sequence.
         */
        virtual KVCacheSequenceState sequenceState(int global_layer, int seq_idx) const
        {
            (void)global_layer;
            (void)seq_idx;
            return {};
        }

        /**
         * @brief Export a logical KV block into packed native-precision buffers.
         */
        virtual bool exportLogicalBlock(const KVCacheLogicalBlockDescriptor &desc, void *dst_k, void *dst_v) const
        {
            (void)desc;
            (void)dst_k;
            (void)dst_v;
            return false;
        }

        /**
         * @brief Import a packed logical KV block into this cache.
         */
        virtual bool importLogicalBlock(const KVCacheLogicalBlockDescriptor &desc, const void *src_k, const void *src_v)
        {
            (void)desc;
            (void)src_k;
            (void)src_v;
            return false;
        }

        /**
         * @brief Truncate every layer for one sequence to cached_tokens.
         */
        virtual bool truncateSequence(int seq_idx, int cached_tokens, void *stream = nullptr)
        {
            (void)seq_idx;
            (void)cached_tokens;
            (void)stream;
            return false;
        }

        /**
         * @brief Device pointer to the live cached-token count mirror.
         *
         * GPU graph-captured stages can use this pointer once they are taught
         * to compute dynamic attention/KV parameters on device.  The pointer is
         * only a mirror unless supportsDeviceResidentSequenceStatePublication()
         * also returns true; callers must not treat it as authoritative for
         * host-visible state before that point.
         */
        virtual const int *deviceCachedTokenCountPtr(int layer, int seq_idx = 0) const
        {
            (void)layer;
            (void)seq_idx;
            return nullptr;
        }

        /**
         * @brief Device pointer to the canonical live cached-token count for a request.
         *
         * Verifier publication needs the cache length before the verifier appends
         * target rows.  GPU caches expose that value through per-layer device
         * mirrors; hybrid models may not have attention state at model layer zero.
         * This helper returns the first available per-layer count so callers can
         * consume a device-owned sequence length without knowing the model's
         * attention/GDN layout.
         */
        virtual const int *deviceSequenceCachedTokenCountPtr(int seq_idx = 0) const
        {
            const int first_layer = first_layer_index();
            const int layer_count = n_layers();
            for (int layer = first_layer; layer < first_layer + layer_count; ++layer)
            {
                if (const int *ptr = deviceCachedTokenCountPtr(layer, seq_idx))
                    return ptr;
            }
            return nullptr;
        }

        /**
         * @brief Device pointer to the live ring-head mirror.
         *
         * Wrapped ring-cache publication requires both the target cached-token
         * count and the current ring head.  This accessor gives future Phase 10
         * kernels a backend-neutral way to consume that head without reaching
         * into CUDA/HIP cache internals.
         */
        virtual const int *deviceRingHeadPtr(int layer, int seq_idx = 0) const
        {
            (void)layer;
            (void)seq_idx;
            return nullptr;
        }

        /**
         * @brief Whether this cache can publish sequence metadata from device buffers.
         *
         * Returning true means the cache can consume
         * DeviceSequenceStatePublicationRequest without a D2H sync and future
         * graph-dynamic KV consumers will observe the published state from
         * device-owned metadata.  Implementations must not return true if they
         * only update an auxiliary device buffer while get_cached_tokens(),
         * attention dynamic params, or graph signatures still depend on stale
         * host counters.
         */
        virtual bool supportsDeviceResidentSequenceStatePublication() const
        {
            return false;
        }

        /**
         * @brief Publish accepted verifier cache counts from device metadata.
         *
         * The default hard-fails so callers cannot accidentally hide the host
         * synchronization that Phase 10 is trying to remove.  GPU backends
         * should implement this only together with device-readable count/head
         * metadata for subsequent attention and append stages.
         */
        virtual bool publishSequenceStateFromDeviceMetadata(
            const DeviceSequenceStatePublicationRequest &request,
            std::string *error = nullptr)
        {
            (void)request;
            if (error)
            {
                *error =
                    "KV cache does not support device-resident sequence-state publication";
            }
            return false;
        }

        /**
         * @brief Adopt host KV head/count mirrors after direct device publication.
         *
         * The default hard-fails because only caches that understand their
         * wrapped-head transition can mirror device publication safely.
         */
        virtual bool adoptSequenceStateFromHostMetadata(
            const HostSequenceStatePublicationRequest &request,
            std::string *error = nullptr)
        {
            (void)request;
            if (error)
            {
                *error =
                    "KV cache does not support host mirror adoption after device-resident publication";
            }
            return false;
        }

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
         * Implementations that support stream-aware append must override this
         * method. The base implementation fails so GPU callers cannot silently
         * drop to append(), which may use the default stream or skip required
         * format conversion.
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
            (void)layer;
            (void)seq_idx;
            (void)K;
            (void)V;
            (void)num_tokens;
            (void)gpu_stream;
            LOG_ERROR("[IKVCache::appendWithStream] Stream-aware append is not implemented by this cache");
            return false;
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
         * @brief Prepare current head position in a device buffer for graph replay
         *
         * Implementations upload the current ring buffer head position to a
         * workspace-owned device buffer on the explicit graph stream. Captured
         * append execution must record the dynamic append kernel only, never an
         * H2D metadata copy.
         *
         * @param layer Layer index
         * @param seq_idx Sequence index
         * @param gpu_stream Explicit GPU stream for the pre-capture upload
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
         * TQ caches override this to upload incremental dequant parameters
         * (ring_pos, out_offset, rope_position) to a device buffer before graph
         * capture/replay. Captured dequant execution must record kernels only.
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
         * @brief Get total number of KV heads across all ranks.
         *
         * Most concrete caches already expose this. The default preserves
         * non-sharded semantics for older implementations.
         */
        virtual int n_kv_heads() const { return local_n_kv_heads(); }

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
            // No base fallback: callers that request converted/RoPE-applied KV
            // require an implementation that owns the necessary scratch buffers.
            (void)target;
            (void)rope;
            if (out_k)
                *out_k = nullptr;
            if (out_v)
                *out_v = nullptr;
            if (out_kv_len)
                *out_kv_len = 0;
            LOG_ERROR("[IKVCache::get_kv_converted] Converted KV read is not implemented by this cache");
            return false;
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
