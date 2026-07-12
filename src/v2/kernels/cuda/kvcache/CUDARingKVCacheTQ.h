/**
 * @file CUDARingKVCacheTQ.h
 * @brief CUDA Ring Buffer KV Cache with TurboQuant (TQ8 K + TQ4 V)
 * @author David Sanftenberg
 *
 * Asymmetric precision ring buffer cache:
 * - K projections stored as TQ8Block (8-bit Lloyd-Max, 256 centroids)
 * - V projections stored as TQ4Block (4-bit Lloyd-Max, 16 centroids)
 *
 * Quantization happens on-GPU during append (FP32 → TQ8/TQ4).
 * Dequantization happens on-GPU during read (TQ8/TQ4 → FP32) with
 * optional fused RoPE for K.
 *
 * Memory layout per position:
 *   K: [n_kv_heads] TQ8Block<D>  (72 bytes each for D=64)
 *   V: [n_kv_heads] TQ4Block<D>  (40 bytes each for D=64)
 *
 * vs FP16 cache:
 *   K: [n_kv_heads * D] __half   (128 bytes for D=64)
 *   V: [n_kv_heads * D] __half   (128 bytes for D=64)
 *
 * Memory savings per position (D=64, 2 KV heads):
 *   TQ: 2*72 + 2*40 = 224 bytes
 *   FP16: 2*128 + 2*128 = 512 bytes
 *   Ratio: 0.44× (56% savings)
 */

#pragma once

#include "CUDARingKVCacheBase.h"
#include "../../../execution/config/RuntimeConfig.h"
#include "../../../tensors/BlockStructures.h"
#include "CUDATurboQuantKernels.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <memory>

namespace llaminar2
{
    // Forward declarations
    class TurboQuantContext;

    // =========================================================================
    // CUDARingKVCacheTQ
    // =========================================================================

    /**
     * @brief CUDA ring buffer KV cache with TurboQuant asymmetric precision.
     *
     * K is stored as TQ8Block<D>, V as TQ4Block<D>.
     * Implements IKVCache for integration with the pipeline.
     */
    class CUDARingKVCacheTQ : public CUDARingKVCacheBase
    {
    public:
        /**
         * @brief Construct TQ KV cache on CUDA device.
         *
         * @param n_layers     Number of transformer layers
         * @param batch_size   Number of sequences
         * @param max_seq_len  Ring buffer capacity
         * @param n_kv_heads   Number of KV heads
         * @param head_dim     Head dimension (64 or 128)
         * @param tq_ctx       TurboQuant context (owns rotation matrices)
         * @param device_id    CUDA device ordinal
         */
        CUDARingKVCacheTQ(int n_layers, int batch_size, int max_seq_len,
                          int n_kv_heads, int head_dim,
                          const TurboQuantContext *tq_ctx,
                          int device_id = 0);

        /**
         * @brief Construct a LocalTP shard of the asymmetric TQ cache.
         *
         * Storage and verifier publication contain only the local KV heads,
         * while the base cache retains the global head count for graph policy
         * and diagnostics.  Rotation matrices are generated for global head
         * IDs `[kv_head_start, kv_head_start + local_n_kv_heads)` so a shard is
         * mathematically identical to slicing the replicated cache.
         *
         * @param n_layers Number of transformer layers owned by this cache.
         * @param batch_size Number of request sequences.
         * @param max_seq_len Ring capacity.
         * @param n_kv_heads Global number of KV heads.
         * @param local_n_kv_heads Number of KV heads stored on this device.
         * @param kv_head_start First global KV head stored on this device.
         * @param head_dim Width of one KV head, either 64 or 128.
         * @param tq_ctx TurboQuant codebook/rotation context.
         * @param device_id CUDA device ordinal.
         */
        CUDARingKVCacheTQ(int n_layers, int batch_size, int max_seq_len,
                          int n_kv_heads, int local_n_kv_heads, int kv_head_start,
                          int head_dim, const TurboQuantContext *tq_ctx,
                          int device_id = 0);

        ~CUDARingKVCacheTQ();

        // Non-copyable, non-movable
        CUDARingKVCacheTQ(const CUDARingKVCacheTQ &) = delete;
        CUDARingKVCacheTQ &operator=(const CUDARingKVCacheTQ &) = delete;

        // =====================================================================
        // IKVCache Interface
        // =====================================================================

        ActivationPrecision k_precision() const override { return ActivationPrecision::TQ8; }
        ActivationPrecision v_precision() const override { return ActivationPrecision::TQ4; }

        // ITensor-based access (returns FP32 shadow tensors)
        ITensor *get_k(int layer, int seq_idx = 0) override;
        const ITensor *get_k(int layer, int seq_idx = 0) const override;
        ITensor *get_v(int layer, int seq_idx = 0) override;
        const ITensor *get_v(int layer, int seq_idx = 0) const override;

        // Unified get_kv
        bool get_kv(int layer, int seq_idx,
                    ITensor **out_k, ITensor **out_v,
                    int *out_kv_len = nullptr) override;
        bool get_kv(int layer, int seq_idx,
                    const ITensor **out_k, const ITensor **out_v,
                    int *out_kv_len = nullptr) const override;

        /**
         * @brief Dequantize independent TQ rings from device-owned metadata.
         *
         * One grouped TQ8 kernel and one grouped TQ4 kernel read every
         * request's immutable ring pointer plus its live device head/count.
         * The result is a compact request-major FP16 view consumed directly by
         * request-batched attention. No host ring mirror or row replay is
         * involved, and both launches are safe to record in a CUDA graph.
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

        // Append (quantizes FP32 input to TQ8/TQ4 on GPU)
        bool append(int layer, int seq_idx,
                    const ITensor *K, const ITensor *V,
                    int num_tokens) override;

        bool appendWithStream(int layer, int seq_idx,
                              const ITensor *K, const ITensor *V,
                              int num_tokens, void *gpu_stream) override;

        /**
         * @brief Publish FP32 MTP verifier rows directly into TQ8/TQ4 storage.
         *
         * This is a true grouped implementation: one fused CUDA grid quantizes
         * all K and V rows and writes their wrapped ring destinations.  During
         * graph capture the destination head remains entirely device-owned.
         */
        bool appendVerifierRowsDecodeEquivalent(int layer,
                                                int seq_idx,
                                                const ITensor *K,
                                                const ITensor *V,
                                                int verifier_rows,
                                                void *gpu_stream) override;

        KVCacheLogicalBlockLayout logicalBlockLayout(int global_layer, int token_count) const override;
        KVCacheSequenceState sequenceState(int global_layer, int seq_idx) const override;
        bool exportLogicalBlock(const KVCacheLogicalBlockDescriptor &desc,
                                void *dst_k, void *dst_v) const override;
        bool importLogicalBlock(const KVCacheLogicalBlockDescriptor &desc,
                                const void *src_k, const void *src_v) override;

        /**
         * @brief Clear all TQ ring entries, scratch views, and device storage.
         *
         * The common CUDA base resets host ring metadata, but TQ also owns
         * compressed ring buffers and FP16 dequant scratch that must not carry
         * request-local rows across prompt boundaries.
         */
        void clear() override;

        /// @brief Clear one layer's TQ ring entries and per-layer scratch storage.
        void clear_layer(int layer) override;

        /// @brief Clear one sequence across all TQ cache layers.
        void clear_sequence(int seq_idx) override;

        /// @brief Clear one sequence entry and invalidate this layer's shared scratch.
        void clear_sequence(int layer, int seq_idx) override;

        // Converted read (dequant + optional RoPE)
        bool get_kv_converted(int layer, int seq_idx,
                              ActivationPrecision target,
                              ITensor **out_k, ITensor **out_v,
                              int *out_kv_len,
                              const KVReadParams *rope = nullptr) override;

        // LocalTP sharding metadata.
        bool is_sharded() const override { return local_n_kv_heads_ != n_kv_heads_; }
        int local_n_kv_heads() const override { return local_n_kv_heads_; }
        int kv_head_start() const override { return kv_head_start_; }
        int local_kv_dim() const override { return kv_dim_; }

        // Eviction
        void evict_oldest(int layer, int seq_idx, int num_tokens);
        void evict_oldest(int layer, int num_tokens)
        {
            evict_oldest(layer, 0, num_tokens);
        }

        // =====================================================================
        // TQ-Specific Accessors
        // =====================================================================

        /// Get the GPU rotation matrices
        const CUDATurboQuantRotations &rotations() const { return rotations_; }

        /// Get raw TQ8 K ring buffer for a layer/seq (for fused attention)
        const void *raw_k_cache(int layer, int seq_idx = 0) const { return entries_[layer][seq_idx].d_K; }
        /// Get raw TQ4 V ring buffer for a layer/seq (for fused attention)
        const void *raw_v_cache(int layer, int seq_idx = 0) const { return entries_[layer][seq_idx].d_V; }
        /**
         * @brief Observe the ring tail for opt-in diagnostics.
         *
         * This performs the same explicit device observation as sequenceState()
         * and must not be used by graph-captured production kernels.
         */
        int ring_tail(int layer, int seq_idx = 0) const
        {
            const KVCacheSequenceState state = sequenceState(layer, seq_idx);
            return state.cached_tokens > 0
                       ? (state.implementation_head - state.cached_tokens + max_seq_len_) % max_seq_len_
                       : state.implementation_head;
        }
        /// Get K block size (bytes per TQ8Block<D>)
        size_t k_block_size() const { return k_block_size_; }
        /// Get V block size (bytes per TQ4Block<D>)
        size_t v_block_size() const { return v_block_size_; }

    private:
        // TQ-specific members (core params are in CUDARingKVCacheBase)

        // Block sizes (depends on head_dim)
        size_t k_block_size_; ///< sizeof(TQ8Block<D>)
        size_t v_block_size_; ///< sizeof(TQ4Block<D>)

        // Per-position storage units
        size_t k_pos_bytes_; ///< n_kv_heads * k_block_size
        size_t v_pos_bytes_; ///< n_kv_heads * v_block_size

        int local_n_kv_heads_; ///< Heads physically stored and processed by this shard.
        int kv_head_start_;    ///< First global KV head represented by local head zero.

        // TurboQuant context (not owned)
        const TurboQuantContext *tq_ctx_;

        // GPU rotation matrices (owned)
        CUDATurboQuantRotations rotations_;

        // =====================================================================
        // Per-layer, per-sequence ring buffer entry
        // =====================================================================
        struct TQEntry
        {
            void *d_K = nullptr; ///< TQ8 blocks: [max_seq_len * n_kv_heads] TQ8Block<D>
            void *d_V = nullptr; ///< TQ4 blocks: [max_seq_len * n_kv_heads] TQ4Block<D>
        };

        // [n_layers][batch_size]
        std::vector<std::vector<TQEntry>> entries_;

        /// Cache-owned immutable entry topology for grouped device reads.
        void **d_batched_k_entry_table_ = nullptr;
        void **d_batched_v_entry_table_ = nullptr;
        /// Stable wrappers over the grouped FP16 payload in layer scratch.
        std::unique_ptr<ITensor> batched_k_view_;
        std::unique_ptr<ITensor> batched_v_view_;

        // =====================================================================
        // Per-Layer FP16 Scratch Buffers (one per layer, enables incremental dequant)
        // =====================================================================
        //
        // Each layer has its own scratch buffer pair so that incremental
        // dequant works during decode: only the newly appended position
        // needs to be dequantized, not the entire sequence.
        //
        // FP16 output enables the FP16 flash attention path (2× less bandwidth).
        // Internal dequant computation remains FP32; only the final write converts.
        //
        // VRAM cost: n_layers × 2 × max_seq_len × kv_dim × 2B (FP16)
        // For Qwen2.5-7B (28 layers, 4 heads, D=128, max_seq=4096):
        //   28 × 2 × 4096 × 512 × 2 = 224MB
        //
        struct ScratchBuffer
        {
            __half *d_K = nullptr; ///< [max_seq_len, kv_dim] FP16 scratch
            __half *d_V = nullptr; ///< [max_seq_len, kv_dim] FP16 scratch

            // ITensor views over the scratch (updated in-place)
            std::unique_ptr<ITensor> k_view;
            std::unique_ptr<ITensor> v_view;

            void invalidate()
            {
                k_view.reset();
                v_view.reset();
            }
        };

        mutable std::vector<ScratchBuffer> layer_scratch_; ///< Per-layer scratch buffers

        mutable cudaStream_t cached_stream_; ///< Last explicit stream used by append/read operations.

        void onClearSequence(int layer, int seq_idx) override
        {
            layer_scratch_[layer].invalidate();
        }

        // Helpers
        void allocate_entry(TQEntry &entry);
        void free_entry(TQEntry &entry);

        /**
         * @brief Publish every `[layer, request]` TQ ring pointer before capture.
         * @param stream Explicit initialization stream ordering the H2D publish.
         * @return true when both cache-owned device tables are ready.
         */
        bool publishBatchedEntryTables(cudaStream_t stream);

        /// @brief Return the stream used for clear-time memset operations.
        cudaStream_t clearStream() const;

        /// @brief Zero the compressed TQ ring storage for one layer/sequence entry.
        void clearEntryStorage(int layer, int seq_idx, cudaStream_t stream);

        /// @brief Zero and invalidate the FP16 dequant scratch owned by one layer.
        void clearScratchStorage(int layer, cudaStream_t stream);

        /// @brief Reset graph-capture sidecar params for one layer/sequence entry.
        void clearDynamicParams(int layer, int seq_idx, cudaStream_t stream);

        /**
         * @brief Materialize one scalar diagnostic view from canonical device state.
         *
         * Production attention uses the grouped device-state dequantizer. This
         * helper takes one temporary state observation and performs a complete
         * linearize/dequant so no host generation ledger can become stale.
         */
        bool dequant_to_scratch(int layer, int seq_idx,
                                float rope_theta,
                                int position_start,
                                int rope_dim,
                                cudaStream_t stream,
                                int *out_count) const;
    };

} // namespace llaminar2
