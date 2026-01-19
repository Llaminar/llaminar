/**
 * @file LMHeadStage.h
 * @brief Language model head projection stage
 */

#pragma once

#include "../IComputeStage.h"
#include "../IWorkspaceConsumerStage.h"
#include "../StageParamsBase.h"

namespace llaminar2
{

    // Forward declaration
    class ITensorGemm;

    /**
     * @brief Language model head projection stage
     *
     * Projects hidden states to vocabulary logits for token prediction.
     * Typically the final stage in a forward pass before sampling.
     *
     * This stage wraps a GEMM operation but provides semantic clarity
     * in compute graphs and enables LM head-specific optimizations.
     *
     * **Workspace Management**:
     * Implements IWorkspaceConsumerStage to delegate workspace requirements to the
     * underlying GEMM kernel. This enables zero-allocation GPU execution by pre-binding
     * workspace buffers during graph setup.
     */
    class LMHeadStage : public IComputeStage, public IWorkspaceConsumerStage
    {
    public:
        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;

            // Input/output tensors
            const ITensor *hidden_states = nullptr;
            const ITensor *lm_head_weight = nullptr;
            ITensor *logits = nullptr;

            // Dimensions
            int seq_len = 0;
            int d_model = 0;
            int vocab_size = 0;

            // Optional bias
            const float *bias = nullptr;
        };

        explicit LMHeadStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::LM_HEAD; }
        size_t estimatedFlops() const override;
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        StageDumpInfo getDumpInfo() const override;
        StageBufferRequirements getBufferRequirements() const override;

        /**
         * @brief Return INPUT policy - kernel manages its own weight upload
         *
         * Quantized GEMM kernels (ROCm/CUDA) pack and upload weights internally
         * to their own INT8 buffers. The coherence system should NOT upload the
         * raw quantized weight tensor - it would be wasteful and the pointer
         * wouldn't be used anyway.
         */
        CoherencePolicy coherencePolicy() const override { return CoherencePolicy::INPUT; }

        // =================================================================
        // IWorkspaceConsumerStage Implementation
        // =================================================================

        /**
         * @brief Get the GEMM kernel as IWorkspaceConsumer for delegation
         *
         * Fetches the kernel from KernelFactory (which caches by tensor+device).
         * The same kernel is returned on every call for this stage.
         *
         * @return Kernel implementing IWorkspaceConsumer, or nullptr if not available
         */
        IWorkspaceConsumer *getKernelAsWorkspaceConsumer() override;

    private:
        Params params_;
    };

} // namespace llaminar2
