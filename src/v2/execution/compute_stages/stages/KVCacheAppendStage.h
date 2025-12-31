/**
 * @file KVCacheAppendStage.h
 * @brief Explicit KV cache append stage
 */

#pragma once

#include "../IComputeStage.h"

namespace llaminar2
{

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
            const TensorBase *K = nullptr; ///< Key to append
            const TensorBase *V = nullptr; ///< Value to append
            IUnifiedKVCache *kv_cache = nullptr;
            int layer_idx = 0;
            int seq_idx = 0;
            int num_tokens = 0;
            int batch_size = 1;
            int seq_len = 0;

            /// [Hybrid mode] Optional output for dequantized V (FP32)
            TensorBase *V_dequant_out = nullptr;

            // =========================================================
            // VNNI-Safe Quantization Parameters (Q16_1 cache)
            // =========================================================

            /// Fixed scale for Q16_1 quantization (from GraphSchema::kv_cache_scale)
            /// Default 8.0 = ±8.0 FP32 range. Set to 0.0 to use legacy adaptive scaling.
            float kv_cache_scale = 8.0f;

            /// Attention head dimension (for VNNI clipping limits)
            /// Required for proper MAX_SAFE_INT16 selection. Common values: 64, 96, 128, 192.
            int head_dim = 128;
        };

        explicit KVCacheAppendStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::COPY; }
        bool supportsBackend(ComputeBackendType backend) const override { return true; }
        StageBufferRequirements getBufferRequirements() const override;
        std::vector<BufferDescriptor> getDeclaredOutputs() const override;
        StageDumpInfo getDumpInfo() const override;

        bool producesVDequant() const { return params_.V_dequant_out != nullptr; }

    private:
        Params params_;
    };

} // namespace llaminar2
