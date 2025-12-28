/**
 * @file LMHeadStage.h
 * @brief Language model head projection stage
 */

#pragma once

#include "../IComputeStage.h"

namespace llaminar2
{

    /**
     * @brief Language model head projection stage
     *
     * Projects hidden states to vocabulary logits for token prediction.
     * Typically the final stage in a forward pass before sampling.
     *
     * This stage wraps a GEMM operation but provides semantic clarity
     * in compute graphs and enables LM head-specific optimizations.
     */
    class LMHeadStage : public IComputeStage
    {
    public:
        struct Params
        {
            // Input/output tensors
            const TensorBase *hidden_states = nullptr;
            const TensorBase *lm_head_weight = nullptr;
            TensorBase *logits = nullptr;

            // Dimensions
            int seq_len = 0;
            int d_model = 0;
            int vocab_size = 0;

            // Optional bias
            const float *bias = nullptr;

            // Device placement
            const MPIContext *mpi_ctx = nullptr;
            int device_idx = -1;
        };

        explicit LMHeadStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::LM_HEAD; }
        size_t estimatedFlops() const override;
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        StageDumpInfo getDumpInfo() const override;
        StageBufferRequirements getBufferRequirements() const override;

    private:
        Params params_;
    };

} // namespace llaminar2
