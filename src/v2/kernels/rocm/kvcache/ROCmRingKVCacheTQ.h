/**
 * @file ROCmRingKVCacheTQ.h
 * @brief ROCm/HIP ring buffer KV cache with TurboQuant compression
 * @author David Sanftenberg
 *
 * HIP mirror of CUDARingKVCacheTQ with immutable AQ8-K/TQ4-V or
 * AQ8-K/TQ8-V storage selected before graph capture.
 */

#pragma once

#include "ROCmRingKVCacheBase.h"
#include "../../kvcache/KVCacheWorkspaceBuffers.h"
#include "../../kvcache/TurboQuantKVMode.h"
#include "ROCmTurboQuantKernels.h"
#include "../../../interfaces/IWorkspaceConsumer.h"
#include "../../../tensors/BlockStructures.h"
#include "../../../tensors/GpuTensorView.h"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <memory>
#include <vector>

namespace llaminar2
{
    class TurboQuantContext;

    class ROCmRingKVCacheTQ : public ROCmRingKVCacheBase,
                              public IWorkspaceConsumer
    {
    public:
        ROCmRingKVCacheTQ(int n_layers, int batch_size, int max_seq_len,
                          int n_kv_heads, int head_dim,
                          const TurboQuantContext *tq_ctx,
                          int device_id,
                           TurboQuantKVMode mode = TurboQuantKVMode::AQ8_K_TQ4_V);

        /**
         * @brief Construct a LocalTP AQ8-K/TQ4-or-TQ8-V cache shard.
         *
         * The local cache stores only `local_n_kv_heads` and builds rotations
         * for their global head IDs, preserving replicated-cache mathematics
         * without allocating or publishing non-local heads.
         */
        ROCmRingKVCacheTQ(int n_layers, int batch_size, int max_seq_len,
                          int n_kv_heads, int local_n_kv_heads, int kv_head_start,
                          int head_dim, const TurboQuantContext *tq_ctx,
                          int device_id,
                           TurboQuantKVMode mode = TurboQuantKVMode::AQ8_K_TQ4_V);

        ~ROCmRingKVCacheTQ() override;

        // Non-copyable, non-movable
        ROCmRingKVCacheTQ(const ROCmRingKVCacheTQ &) = delete;
        ROCmRingKVCacheTQ &operator=(const ROCmRingKVCacheTQ &) = delete;

        // =====================================================================
        // IKVCache Interface
        // =====================================================================

        ActivationPrecision k_precision() const override { return ActivationPrecision::AQ8; }
        ActivationPrecision v_precision() const override
        {
            return turboQuantValuePrecision(mode_);
        }

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

        /** @copydoc IKVCache::describeDeviceReadStorage */
        std::optional<DeviceReadStorage> describeDeviceReadStorage(
            const DeviceReadStorageRequest &request) const override;

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

        // Converted read (dequant + optional RoPE)
        bool get_kv_converted(int layer, int seq_idx,
                              ActivationPrecision target,
                              ITensor **out_k, ITensor **out_v,
                              int *out_kv_len,
                              const KVReadParams *rope = nullptr) override;

        // =====================================================================
        // IWorkspaceConsumer Interface
        // =====================================================================

        /**
         * @brief Declare full-horizon FP16 conversion buffers for graph replay.
         */
        WorkspaceRequirements getWorkspaceRequirements(
            int m, int n = 0, int k = 0) const override;

        /**
         * @brief Bind planner-owned conversion storage before HIP graph capture.
         *
         * No private scratch is allocated when the workspace is absent.
         */
        void bindWorkspace(DeviceWorkspaceManager *workspace) override;
        bool hasWorkspace() const override { return workspace_ != nullptr; }
        DeviceWorkspaceManager *getWorkspace() const override { return workspace_; }

        // LocalTP sharding metadata.
        bool is_sharded() const override { return local_n_kv_heads_ != n_kv_heads_; }
        int local_n_kv_heads() const override { return local_n_kv_heads_; }
        int kv_head_start() const override { return kv_head_start_; }
        int local_kv_dim() const override { return kv_dim_; }

        // =====================================================================
        // ROCm-Specific Accessors
        // =====================================================================

        /**
         * @brief Return the immutable physical AQ8 key ring for diagnostics and fused attention.
         * @param layer Local layer index.
         * @param seq_idx Request slot within the cache batch.
         * @return Device pointer to the first compressed key block.
         *
         * The pointer remains stable for the cache lifetime. Callers must use
         * the exact stream/event that published the corresponding cache data;
         * this accessor does not introduce an ordering edge.
         */
        const void *raw_k_cache(int layer, int seq_idx = 0) const
        {
            return entries_[layer][seq_idx].d_K;
        }

        /**
         * @brief Return the immutable physical compressed-value ring.
         * @param layer Local layer index.
         * @param seq_idx Request slot within the cache batch.
         * @return Device pointer to TQ4, TQ8, or Q8_1 blocks selected by `mode_`.
         *
         * This diagnostic/fused-kernel view exposes storage without
         * materializing a host shadow. The cache retains ownership.
         */
        const void *raw_v_cache(int layer, int seq_idx = 0) const
        {
            return entries_[layer][seq_idx].d_V;
        }

        const ROCmTurboQuantRotations &rotations() const { return rotations_; }

        // Eviction
        void evict_oldest(int layer, int seq_idx, int num_tokens);
        void evict_oldest(int layer, int num_tokens)
        {
            evict_oldest(layer, 0, num_tokens);
        }

    protected:
        void onResetLayerSequenceState(int layer, int seq_idx) override
        {
            (void)seq_idx;
            layer_scratch_[layer].invalidate();
            cached_stream_ = nullptr;
        }
    private:
        struct TQEntry
        {
            void *d_K = nullptr; // AttentionKeyQ8Block ring buffer
            void *d_V = nullptr; // TQ4Block ring buffer
            float *d_K_anchor = nullptr; ///< Exact first-key basis for AQ8 residuals.
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

        /// @brief Resolve the last explicit stream for diagnostic-only reads.
        hipStream_t clearStream() const;

        size_t k_block_size_; ///< sizeof(AttentionKeyQ8Block<head_dim>)
        size_t v_block_size_;

        int local_n_kv_heads_; ///< Heads physically stored by this ROCm shard.
        int kv_head_start_;    ///< First global head represented by local head zero.
        TurboQuantKVMode mode_; ///< Immutable storage policy resolved before capture.

        ROCmTurboQuantRotations rotations_;

        std::vector<std::vector<TQEntry>> entries_;

        /// Cache-owned immutable `[layer, request]` compressed-ring topology.
        void **d_batched_k_entry_table_ = nullptr;
        void **d_batched_v_entry_table_ = nullptr;
        void **d_batched_k_anchor_table_ = nullptr;
        /// Stable wrappers over grouped FP16 layer scratch.
        std::unique_ptr<ITensor> batched_k_view_;
        std::unique_ptr<ITensor> batched_v_view_;

        // Per-layer wrappers over one graph-planned producer/consumer buffer.
        mutable std::vector<ScratchBuffer> layer_scratch_;
        DeviceWorkspaceManager *workspace_ = nullptr; ///< Bound graph workspace, not owned.
        size_t scratch_capacity_bytes_ = 0;           ///< Capacity of each K/V buffer.

        mutable hipStream_t cached_stream_; ///< Last explicit stream used by append/read operations.

        /**
         * @brief Publish every compressed ring pointer on the initialization stream.
         * @return true when both cache-owned device tables are ready.
         */
        bool publishBatchedEntryTables(hipStream_t stream);

        /**
         * @brief Release all ROCm allocations owned by this cache instance.
         *
         * Both the constructor transaction guard and the destructor call this
         * method. That keeps partial initialization from leaking compressed
         * entries, grouped pointer tables, layer scratch, or rotations.
         */
        void releaseOwnedDeviceStorage() noexcept;
    };

} // namespace llaminar2
