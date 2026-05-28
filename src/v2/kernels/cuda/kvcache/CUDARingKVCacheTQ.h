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
#include "../../kvcache/KVCacheDeviceParams.h"
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

        // Append (quantizes FP32 input to TQ8/TQ4 on GPU)
        bool append(int layer, int seq_idx,
                    const ITensor *K, const ITensor *V,
                    int num_tokens) override;

        bool appendWithStream(int layer, int seq_idx,
                              const ITensor *K, const ITensor *V,
                              int num_tokens, void *gpu_stream) override;

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

        // Sharding (basic — not yet implemented for TQ)
        bool is_sharded() const override { return false; }
        int local_n_kv_heads() const override { return n_kv_heads_; }
        int kv_head_start() const override { return 0; }
        int local_kv_dim() const override { return kv_dim_; }

        // TQ-specific graph capture support (dynamic dequant params)
        void setDynamicDequantParams(int layer, int seq_idx,
                                     float rope_theta, int position_start,
                                     void *gpu_stream) override;

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
        /// Get ring buffer tail position (start of valid data)
        int ring_tail(int layer, int seq_idx = 0) const { return entries_[layer][seq_idx].tail(max_seq_len_); }
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

            int head = 0;  ///< Next write position
            int count = 0; ///< Valid tokens

            int tail(int max_seq_len) const
            {
                return (head - count + max_seq_len) % max_seq_len;
            }

            bool is_wrapped(int max_seq_len) const
            {
                if (count == 0)
                    return false;
                int t = tail(max_seq_len);
                return t >= head && count > 0;
            }
        };

        // [n_layers][batch_size]
        std::vector<std::vector<TQEntry>> entries_;

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

            // Cache tracking — avoids redundant dequant when the same
            // (layer, seq, params) is accessed consecutively.
            int cached_layer = -1;
            int cached_seq = -1;
            int cached_count = 0;
            int cached_head = -1;
            int cached_tail = -1;
            float cached_rope_theta = 0.0f;
            int cached_position_start = -1;

            void invalidate()
            {
                cached_layer = -1;
                cached_seq = -1;
                cached_count = 0;
                cached_head = -1;
                cached_tail = -1;
                cached_rope_theta = 0.0f;
                cached_position_start = -1;
            }

            bool is_current_for(int layer, int seq, int count, int head,
                                float rope_theta, int pos_start) const
            {
                return cached_layer == layer && cached_seq == seq &&
                       cached_count == count && cached_head == head &&
                       cached_rope_theta == rope_theta &&
                       (rope_theta <= 0.0f || cached_position_start == pos_start);
            }

            /// Check if we can do incremental dequant (only 1 new position)
            bool can_incremental(int layer, int seq, int count, int tail,
                                 float rope_theta, int pos_start) const
            {
                return cached_layer == layer && cached_seq == seq &&
                       cached_count > 0 && count == cached_count + 1 &&
                       cached_tail == tail &&
                       cached_rope_theta == rope_theta &&
                       (rope_theta <= 0.0f || cached_position_start == pos_start);
            }
        };

        mutable std::vector<ScratchBuffer> layer_scratch_; ///< Per-layer scratch buffers

        // =====================================================================
        // Pre-allocated temp buffers for GPU quantization (avoid per-call malloc)
        // =====================================================================
        void *d_quantize_k_temp_ = nullptr; ///< Temp TQ8 blocks [max_seq_len * n_kv_heads]
        void *d_quantize_v_temp_ = nullptr; ///< Temp TQ4 blocks [max_seq_len * n_kv_heads]

        // =====================================================================
        // Device-side dynamic params for CUDA graph capture
        // =====================================================================
        // Note: d_head_params_ and h_head_params_ are in CUDARingKVCacheBase

        // Incremental dequant params (TQ-specific, for graph-capturable attention)
        TQDequantDynamicParams *d_dequant_params_ = nullptr; ///< [n_layers_] device
        TQDequantDynamicParams *h_dequant_params_ = nullptr; ///< [n_layers_] pinned host

        mutable cudaStream_t cached_stream_; ///< Last explicit stream used by append/read operations.

        // =====================================================================
        // CUDARingKVCacheBase entry accessors and hooks
        // =====================================================================

        int entryHead(int layer, int seq_idx) const override { return entries_[layer][seq_idx].head; }
        int entryCount(int layer, int seq_idx) const override { return entries_[layer][seq_idx].count; }
        void setEntryHead(int layer, int seq_idx, int value) override { entries_[layer][seq_idx].head = value; }
        void setEntryCount(int layer, int seq_idx, int value) override { entries_[layer][seq_idx].count = value; }

        void resetEntry(int layer, int seq_idx) override
        {
            entries_[layer][seq_idx].head = 0;
            entries_[layer][seq_idx].count = 0;
        }

        void onClearSequence(int layer, int seq_idx) override
        {
            layer_scratch_[layer].invalidate();
        }

        void onEviction(int layer, int seq_idx, int num_evicted) override
        {
            (void)num_evicted;
            if (layer_scratch_[layer].cached_layer == layer &&
                layer_scratch_[layer].cached_seq == seq_idx)
                layer_scratch_[layer].invalidate();
        }

        void onAdvanceComplete(int layer, int seq_idx) override
        {
            auto &scratch = layer_scratch_[layer];
            if (scratch.cached_layer == layer && scratch.cached_seq == seq_idx)
            {
                scratch.cached_count = entries_[layer][seq_idx].count;
                scratch.cached_head = entries_[layer][seq_idx].head;
            }
        }

        // Helpers
        void allocate_entry(TQEntry &entry);
        void free_entry(TQEntry &entry);

        /// @brief Return the stream used for clear-time memset operations.
        cudaStream_t clearStream() const;

        /// @brief Zero the compressed TQ ring storage for one layer/sequence entry.
        void clearEntryStorage(int layer, int seq_idx, cudaStream_t stream);

        /// @brief Zero and invalidate the FP16 dequant scratch owned by one layer.
        void clearScratchStorage(int layer, cudaStream_t stream);

        /// @brief Reset graph-capture sidecar params for one layer/sequence entry.
        void clearDynamicParams(int layer, int seq_idx, cudaStream_t stream);

        /// Dequantize a layer's TQ ring into the shared scratch buffer.
        /// Returns false on kernel launch failure.
        bool dequant_to_scratch(int layer, int seq_idx,
                                float rope_theta,
                                int position_start,
                                cudaStream_t stream) const;
    };

} // namespace llaminar2
