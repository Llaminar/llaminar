/**
 * @file AttentionComputeStage.h
 * @brief Pure attention compute stage (no KV cache management)
 */

#pragma once

#include "../IComputeStage.h"
#include "backends/DeviceId.h"

namespace llaminar2
{

    /**
     * @brief Pure attention compute stage (no KV cache management)
     *
     * Delegates to KernelFactory::createAttention() for device-appropriate kernel
     * selection. This stage handles ONLY the attention computation:
     *   output = softmax(Q @ K^T / sqrt(head_dim) + mask) @ V
     *
     * For KV cache management, use KVCacheAppendStage separately in the DAG.
     * For integrated cache+attention, use AttentionWithKVCacheStage.
     */
    class AttentionComputeStage : public IComputeStage
    {
    public:
        struct Params
        {
            // Input/output tensors
            ITensor *Q = nullptr;
            ITensor *K = nullptr;
            ITensor *V = nullptr;
            ITensor *output = nullptr;

            // Dimensions
            int batch_size = 1;
            int seq_len = 0;
            int kv_len = 0;
            int n_heads = 0;
            int n_kv_heads = 0;
            int head_dim = 0;

            // Attention configuration
            bool causal = true;
            int window_size = -1;

            /// Execution mode
            AttentionMode attention_mode = AttentionMode::PREFILL;
            bool auto_detect_mode = true;

            // Workspace buffers
            ITensor *workspace_scores = nullptr;
            ITensor *workspace_context = nullptr;
            ITensor *workspace_mask = nullptr;

            // KV cache for dynamic length query at execution time
            ICPUKVCache *kv_cache = nullptr;
            int layer_idx = -1;

            // Position offset for decode mode causal masking
            int position_offset = 0;

            // Optional MPI context
            const MPIContext *mpi_ctx = nullptr;
            DeviceId device_id = DeviceId::cpu();
        };

        explicit AttentionComputeStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::ATTENTION; }
        size_t estimatedFlops() const override;
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        StageDumpInfo getDumpInfo() const override;
        StageBufferRequirements getBufferRequirements() const override;

        void updateDynamicParams(int pos_offset, int seq_len) override
        {
            params_.position_offset = pos_offset;
            (void)seq_len;
        }

        const Params &getParams() const { return params_; }

    private:
        Params params_;
    };

} // namespace llaminar2
