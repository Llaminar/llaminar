/**
 * @file AttentionWithKVCacheStage.h
 * @brief Production attention stage with KV cache and MPI support
 */

#pragma once

#include "../IComputeStage.h"
#include "../StageParamsBase.h"

namespace llaminar2
{

    /**
     * @brief Production attention stage with KV cache and MPI support
     *
     * This is the full-featured attention implementation that integrates with:
     * - **KV Cache**: Automatic append during prefill, retrieval during decode
     * - **MPI Parallelism**: Tensor-parallel attention via KernelFactory + compute_tensor()
     * - **Batched Execution**: Multiple sequences with padding masks
     * - **Asymmetric Lengths**: Different Q and KV lengths (prefill vs decode)
     * - **GQA/MHA/MQA**: All attention variants supported
     *
     * Execution Modes:
     * 1. **Prefill**: Process all tokens, append to KV cache, full attention
     * 2. **Decode**: Single token query, use cached K/V, efficient O(n) per token
     * 3. **Batched**: Multiple sequences, padding-aware masking
     */
    class AttentionWithKVCacheStage : public IComputeStage
    {
    public:
        /**
         * @brief Legacy mode enum for backward compatibility
         * @deprecated Use AttentionMode from TensorKernels.h for new code
         */
        enum class Mode
        {
            AUTO,
            PREFILL,
            DECODE,
            BATCHED
        };

        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;

            // Input/output tensors (already projected)
            ITensor *Q = nullptr;
            ITensor *K = nullptr;
            ITensor *V = nullptr;
            ITensor *output = nullptr;

            // KV cache integration
            IKVCache *kv_cache = nullptr;
            int layer_idx = 0;

            // Execution mode (legacy - prefer attention_mode)
            Mode mode = Mode::AUTO;

            // Attention configuration
            int batch_size = 1;
            int seq_len = 0;
            int n_heads = 0;
            int n_kv_heads = 0;
            int head_dim = 0;
            bool causal = true;
            int window_size = -1;

            // MPI configuration - shared_ptr version for lifetime management
            std::shared_ptr<MPIContext> mpi_ctx_shared;
            int mpi_strategy = 0;

            // Workspace buffers
            ITensor *workspace_scores = nullptr;
            ITensor *workspace_context = nullptr;
            ITensor *workspace_mask = nullptr;

            // Per-sequence lengths for batched attention
            const std::vector<int> *sequence_lengths = nullptr;

            // Position offset for decode mode
            int position_offset = 0;

            // Tensor Parallelism Parameters
            int head_start = 0;
            int local_n_heads = -1;
            int local_n_kv_heads = -1;
        };

        explicit AttentionWithKVCacheStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::ATTENTION; }
        size_t estimatedFlops() const override;
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        bool isGraphCapturable() const override { return false; }
        StageDumpInfo buildDumpInfoImpl() const override;
        StageBufferRequirements getBufferRequirements() const override;

        Mode effectiveMode() const;

    private:
        Params params_;

        bool executePrefill(IDeviceContext *ctx);
        bool executeDecode(IDeviceContext *ctx);
        bool executeBatched(IDeviceContext *ctx);
    };

} // namespace llaminar2
