/**
 * @file ROCmRingKVCacheTQ.h
 * @brief ROCm/HIP ring buffer KV cache with TurboQuant compression
 * @author David Sanftenberg
 *
 * HIP mirror of CUDARingKVCacheTQ. Uses TQ8 for Keys and TQ4 for Values.
 * Memory savings: 56% vs FP16 (for D=64 with 2 KV heads).
 */

#pragma once

#include "ROCmRingKVCacheBase.h"
#include "ROCmTurboQuantKernels.h"
#include "../../../tensors/BlockStructures.h"
#include "../../../tensors/GpuTensorView.h"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <memory>
#include <vector>

namespace llaminar2
{
    class TurboQuantContext;

    class ROCmRingKVCacheTQ : public ROCmRingKVCacheBase
    {
    public:
        ROCmRingKVCacheTQ(int n_layers, int batch_size, int max_seq_len,
                          int n_kv_heads, int head_dim,
                          const TurboQuantContext *tq_ctx,
                          int device_id);

        ~ROCmRingKVCacheTQ() override;

        // Non-copyable, non-movable
        ROCmRingKVCacheTQ(const ROCmRingKVCacheTQ &) = delete;
        ROCmRingKVCacheTQ &operator=(const ROCmRingKVCacheTQ &) = delete;

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
         * The common ROCm base resets host ring metadata, but TQ owns compressed
         * ring buffers and FP16 dequant scratch that can otherwise retain rows
         * across request boundaries.
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

        // Sharding
        bool is_sharded() const override { return false; }
        int local_n_kv_heads() const override { return n_kv_heads_; }
        int kv_head_start() const override { return 0; }
        int local_kv_dim() const override { return kv_dim_; }

        // =====================================================================
        // Graph Capture Support
        // =====================================================================

        bool isGraphCaptureReady() const override { return d_dequant_params_ != nullptr; }

        /**
         * @brief Set dynamic dequant params for graph capture.
         * Kept for legacy append-position setup; dequant metadata is prepared
         * through setDynamicDequantParams().
         */
        void setDynamicHead(int layer, int seq_idx, void *gpu_stream) override;

        void setDynamicDequantParams(int layer, int seq_idx,
                                     float rope_theta, int position_start,
                                     void *gpu_stream) override;

        // =====================================================================
        // ROCm-Specific Accessors
        // =====================================================================

        const ROCmTurboQuantRotations &rotations() const { return rotations_; }

        // Eviction
        void evict_oldest(int layer, int seq_idx, int num_tokens);
        void evict_oldest(int layer, int num_tokens)
        {
            evict_oldest(layer, 0, num_tokens);
        }

    protected:
        // =====================================================================
        // ROCmRingKVCacheBase entry accessor overrides
        // =====================================================================

        int entryHead(int layer, int seq_idx) const override
        {
            return entries_[layer][seq_idx].head;
        }
        int entryCount(int layer, int seq_idx) const override
        {
            return entries_[layer][seq_idx].count;
        }
        void setEntryHead(int layer, int seq_idx, int value) override
        {
            entries_[layer][seq_idx].head = value;
        }
        void setEntryCount(int layer, int seq_idx, int value) override
        {
            entries_[layer][seq_idx].count = value;
        }
        void resetEntry(int layer, int seq_idx) override
        {
            entries_[layer][seq_idx].head = 0;
            entries_[layer][seq_idx].count = 0;
        }

        void onClearSequence(int layer, int seq_idx) override
        {
            (void)seq_idx;
            layer_scratch_[layer].invalidate();
            cached_stream_ = nullptr;
        }
        void onAdvanceComplete(int layer, int seq_idx) override
        {
            // TQ intentionally does NOT invalidate scratch here — the incremental
            // dequant path in dequant_to_scratch() handles count+1 efficiently.
            (void)layer;
            (void)seq_idx;
        }

    private:
        struct TQEntry
        {
            void *d_K = nullptr; // TQ8Block ring buffer
            void *d_V = nullptr; // TQ4Block ring buffer
            int head = 0;
            int count = 0;

            int tail(int max_seq_len) const
            {
                return (count >= max_seq_len) ? head : 0;
            }
        };

        struct ScratchBuffer
        {
            _Float16 *d_K = nullptr;
            _Float16 *d_V = nullptr;
            std::unique_ptr<ITensor> k_view;
            std::unique_ptr<ITensor> v_view;

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

        bool dequant_to_scratch(int layer, int seq_idx, float rope_theta, int position_start, hipStream_t stream) const;

        /// @brief Return the stream used for clear-time memset operations.
        hipStream_t clearStream() const;

        /// @brief Zero the compressed TQ ring storage for one layer/sequence entry.
        void clearEntryStorage(int layer, int seq_idx, hipStream_t stream);

        /// @brief Zero and invalidate the FP16 dequant scratch owned by one layer.
        void clearScratchStorage(int layer, hipStream_t stream);

        /// @brief Reset graph-capture sidecar params for one layer/sequence entry.
        void clearDynamicParams(int layer, int seq_idx, hipStream_t stream);

        size_t tq8_block_size_;
        size_t tq4_block_size_;

        ROCmTurboQuantRotations rotations_;

        std::vector<std::vector<TQEntry>> entries_;

        // Per-layer FP16 scratch buffers (eliminates cross-layer invalidation)
        mutable std::vector<ScratchBuffer> layer_scratch_;

        // Pre-allocated temp buffers for GPU-side quantization
        void *d_quantize_k_temp_ = nullptr;
        void *d_quantize_v_temp_ = nullptr;

        // Graph capture dynamic params (per-layer)
        HIPTQDequantDynamicParams *d_dequant_params_ = nullptr; ///< Device array [n_layers_]
        HIPTQDequantDynamicParams *h_dequant_params_ = nullptr; ///< Pinned host array [n_layers_]
        mutable std::vector<uint8_t> dequant_params_device_valid_; ///< Per-layer pre-upload readiness

        mutable hipStream_t cached_stream_; ///< Last explicit stream used by append/read operations.
    };

} // namespace llaminar2
