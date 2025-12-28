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
