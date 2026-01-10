/**
 * @file KVCacheGatherStage.h
 * @brief Gather K/V from multiple cache slots into batched output tensors
 */

#pragma once

#include "../IComputeStage.h"

namespace llaminar2
{

    /**
     * @brief Gather K/V from multiple cache slots into batched output tensors
     *
     * For batched decode with KV cache history, each sequence has K/V stored
     * at a separate seq_idx in the cache. This stage gathers them into contiguous
     * tensors suitable for batched attention computation.
     */
    class KVCacheGatherStage : public IComputeStage
    {
    public:
        struct Params
        {
            IKVCache *kv_cache = nullptr;
            int layer_idx = 0;
            int batch_size = 1;

            ITensor *out_K = nullptr;
            ITensor *out_V = nullptr;

            int *out_max_kv_len = nullptr;
            std::vector<int> *out_per_seq_kv_lens = nullptr;
        };

        explicit KVCacheGatherStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::COPY; }
        bool supportsBackend(ComputeBackendType backend) const override { return true; }
        StageBufferRequirements getBufferRequirements() const override;
        StageDumpInfo getDumpInfo() const override;

        /**
         * @brief Allow zero output because cache may be empty on first token
         *
         * Before any tokens have been processed, the KV cache is empty and
         * gather produces a zero-length output. This is not a bug.
         */
        bool allowsZeroOutput() const override { return true; }

        int getMaxKVLen() const { return last_max_kv_len_; }
        const std::vector<int> &getPerSeqKVLens() const { return last_per_seq_kv_lens_; }

    private:
        Params params_;
        int last_max_kv_len_ = 0;
        std::vector<int> last_per_seq_kv_lens_;
    };

} // namespace llaminar2
