/**
 * @file KVCacheAppendStage.h
 * @brief Explicit KV cache append stage
 */

#pragma once

#include "../IComputeStage.h"
#include "../StageParamsBase.h"
#include "kernels/IKVCache.h"
#include <memory>

namespace llaminar2
{

    class FP16Tensor;
    class Q8_1Tensor;

    /**
     * @brief Explicit KV cache append stage
     *
     * Separates cache operations from attention computation, enabling:
     * - Pipelined execution: Append on one device while attending on another
     * - Explicit control: Manual cache management for advanced use cases
     * - Cross-device caches: Cache on GPU while computing on CPU
     *
     * For most use cases, prefer AttentionWithKVCacheStage which handles
     * cache operations internally. Use this stage when you need fine-grained
     * control over cache timing.
     *
     * VNNI-Safe Quantization (Q16_1 cache):
     * When the cache is Q16_1, this stage uses FIXED-SCALE quantization with
     * VNNI-safe clipping to prevent INT32 overflow during attention computation.
     * Set kv_cache_scale and head_dim to enable proper clipping limits.
     *
     * See: kernels/cpu/attention/q16_1/VNNISafetyConstants.h for clipping limits
     * See: docs/v2/PROJECT_Q16_INTEGER_ATTENTION_V2.md "VNNI OVERFLOW PREVENTION CONTRACT"
     */
    class KVCacheAppendStage : public IComputeStage
    {
    public:
        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;

            const ITensor *K = nullptr; ///< Key to append
            const ITensor *V = nullptr; ///< Value to append
            IKVCache *kv_cache = nullptr;
            int layer_idx = 0;
            int seq_idx = 0;
            int num_tokens = 0;
            int batch_size = 1;
            int seq_len = 0;

            /// [Hybrid mode] Optional output for dequantized V (FP32)
            ITensor *V_dequant_out = nullptr;

            // =========================================================
            // VNNI-Safe Quantization Parameters (Q16_1 cache)
            // =========================================================

            /// Fixed scale for Q16_1 quantization (from GraphSchema::kv_cache_scale)
            /// Default 8.0 = ±8.0 FP32 range. Set to 0.0 to use legacy adaptive scaling.
            float kv_cache_scale = 256.0f; ///< Fixed Q16 scale. Must cover projection max (~130)

            /// Attention head dimension (for VNNI clipping limits)
            /// Required for proper MAX_SAFE_INT16 selection. Common values: 64, 96, 128, 192.
            int head_dim = 128;
        };

        explicit KVCacheAppendStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::COPY; }
        // KV cache append is graph-capturable when the KV cache supports
        // device-side head parameters. The H2D memcpy + dynamic kernel are
        // captured once; on replay, updateDynamicParams() writes the new head
        // to the same pinned host buffer, which the captured H2D re-reads.
        bool isGraphCapturable() const override
        {
            return params_.kv_cache && params_.kv_cache->isGraphCaptureReady();
        }
        bool hasDynamicParams() const override { return true; }
        void updateDynamicParams(int pos_offset, int seq_len) override
        {
            params_.seq_len = seq_len;
            if (params_.kv_cache)
            {
                // Write current head position to the pinned host buffer and issue
                // H2D async copy. The captured graph's H2D will re-read this value.
                // NOTE: Do NOT advanceHead here — that happens in onGraphReplayed()
                // after the graph launches. This preserves the invariant that
                // get_cached_tokens() returns the previous step's count when
                // AttentionComputeStage::updateDynamicParams() reads it.
                void *stream = gpuStream();
                params_.kv_cache->setDynamicHead(params_.layer_idx, params_.seq_idx, stream);
            }
        }
        void onGraphReplayed() override
        {
            // Advance the ring buffer head and count on the host side.
            // Called by GraphExecutor AFTER the captured graph segment replays.
            if (params_.kv_cache)
            {
                params_.kv_cache->advanceHead(params_.layer_idx, params_.seq_idx, params_.num_tokens);
            }
        }
        bool needsOnGraphReplayed() const override { return true; }
        bool supportsBackend(ComputeBackendType backend) const override { return true; }
        StageBufferRequirements getBufferRequirements() const override;
        std::vector<BufferDescriptor> getDeclaredOutputs() const override;
        StageDumpInfo buildDumpInfoImpl() const override;

        bool producesVDequant() const { return params_.V_dequant_out != nullptr; }

    private:
        Params params_;
        std::unique_ptr<FP16Tensor> fp16_k_scratch_;
        std::unique_ptr<FP16Tensor> fp16_v_scratch_;
        std::unique_ptr<Q8_1Tensor> q8_k_scratch_;
        std::unique_ptr<Q8_1Tensor> q8_v_scratch_;
    };

} // namespace llaminar2
