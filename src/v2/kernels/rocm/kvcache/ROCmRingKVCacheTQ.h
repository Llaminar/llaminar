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

        /**
         * @brief Construct a LocalTP TQ8-K/TQ4-V cache shard.
         *
         * The local cache stores only `local_n_kv_heads` and builds rotations
         * for their global head IDs, preserving replicated-cache mathematics
         * without allocating or publishing non-local heads.
         */
        ROCmRingKVCacheTQ(int n_layers, int batch_size, int max_seq_len,
                          int n_kv_heads, int local_n_kv_heads, int kv_head_start,
                          int head_dim, const TurboQuantContext *tq_ctx,
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

        /**
         * @brief Dequantize independent TQ rings from device-owned metadata.
         *
         * Grouped TQ8-K and TQ4-V HIP kernels read immutable cache-owned entry
         * pointers together with the live device head/count scalars. They emit
         * one compact request-major FP16 view for attention without host state,
         * compressed staging, or serial request replay.
         */
        bool get_kv_batched_device_view(
            int layer,
            int first_seq_idx,
            int request_count,
            int max_kv_len,
            ITensor **out_k,
            ITensor **out_v,
            void *gpu_stream) override;

        bool get_kv_batched_converted_device_view(
            int layer,
            int first_seq_idx,
            int request_count,
            int max_kv_len,
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
         * @brief Quantize and publish all verifier rows in one HIP grid.
         *
         * The grouped grid uses the same per-head reduction and rotation order
         * as serial decode and supports both position-major and verifier
         * head-major FP32 source tensors.
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

        // LocalTP sharding metadata.
        bool is_sharded() const override { return local_n_kv_heads_ != n_kv_heads_; }
        int local_n_kv_heads() const override { return local_n_kv_heads_; }
        int kv_head_start() const override { return kv_head_start_; }
        int local_kv_dim() const override { return kv_dim_; }

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
        void onClearSequence(int layer, int seq_idx) override
        {
            (void)seq_idx;
            layer_scratch_[layer].invalidate();
            cached_stream_ = nullptr;
        }
    private:
        struct TQEntry
        {
            void *d_K = nullptr; // TQ8Block ring buffer
            void *d_V = nullptr; // TQ4Block ring buffer
        };

        struct ScratchBuffer
        {
            _Float16 *d_K = nullptr;
            _Float16 *d_V = nullptr;
            std::unique_ptr<ITensor> k_view;
            std::unique_ptr<ITensor> v_view;

            void invalidate()
            {
                k_view.reset();
                v_view.reset();
            }
        };

        /** @brief Full scalar materialization from one temporary device-state observation. */
        bool dequant_to_scratch(
            int layer, int seq_idx, float rope_theta, int position_start,
            int rope_dim, hipStream_t stream, int *out_count) const;

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

        int local_n_kv_heads_; ///< Heads physically stored by this ROCm shard.
        int kv_head_start_;    ///< First global head represented by local head zero.

        ROCmTurboQuantRotations rotations_;

        std::vector<std::vector<TQEntry>> entries_;

        /// Cache-owned immutable `[layer, request]` compressed-ring topology.
        void **d_batched_k_entry_table_ = nullptr;
        void **d_batched_v_entry_table_ = nullptr;
        /// Stable wrappers over grouped FP16 layer scratch.
        std::unique_ptr<ITensor> batched_k_view_;
        std::unique_ptr<ITensor> batched_v_view_;

        // Per-layer FP16 scratch buffers (eliminates cross-layer invalidation)
        mutable std::vector<ScratchBuffer> layer_scratch_;

        mutable hipStream_t cached_stream_; ///< Last explicit stream used by append/read operations.

        /**
         * @brief Publish every compressed ring pointer on the initialization stream.
         * @return true when both cache-owned device tables are ready.
         */
        bool publishBatchedEntryTables(hipStream_t stream);
    };

} // namespace llaminar2
