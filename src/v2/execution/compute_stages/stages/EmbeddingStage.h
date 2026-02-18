/**
 * @file EmbeddingStage.h
 * @brief Embedding lookup stage
 */

#pragma once

#include "../IComputeStage.h"
#include "../IWorkspaceConsumerStage.h"
#include "../StageParamsBase.h"
#include "../../../tensors/TensorKernels.h"

namespace llaminar2
{

    /**
     * @brief Embedding lookup stage
     *
     * Performs token → embedding lookup for transformer input processing.
     * Supports output to various precision formats (FP32, BF16, Q8_1).
     *
     * The embedding table is typically FP32 (stored in model weights).
     * Output format is determined by the output tensor's native_type().
     *
     * Implements IWorkspaceConsumerStage to delegate workspace requirements
     * to the underlying embedding kernel (required for ROCm kernels).
     */
    class EmbeddingStage : public IComputeStage, public IWorkspaceConsumerStage
    {
    public:
        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;

            // Input/output tensors
            const ITensor *embed_table = nullptr;
            const int *token_ids = nullptr;
            ITensor *output = nullptr;

            // Dimensions
            int num_tokens = 0;
            int d_model = 0;
            int vocab_size = 0;

            // Batched input (alternative to token_ids)
            const std::vector<std::vector<int>> *token_batches = nullptr;
            int padded_seq_len = 0;
        };

        explicit EmbeddingStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::EMBEDDING; }
        size_t estimatedFlops() const override;
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        bool isGraphCapturable() const override { return false; } // token_ids change each step
        StageDumpInfo buildDumpInfoImpl() const override;
        StageBufferRequirements getBufferRequirements() const override;

        // IWorkspaceConsumerStage Implementation
        IWorkspaceConsumer *getKernelAsWorkspaceConsumer() override;

    private:
        Params params_;
        ITensorEmbedding *cached_kernel_ = nullptr;
        int cached_kernel_tensor_type_ = -1;

        bool executeQ16_1Output();
        static void zero_output_row(ITensor *output, int row_idx, int d_model);

        // Get or create the embedding kernel (caches for workspace binding)
        ITensorEmbedding *getOrCreateKernel();
    };

} // namespace llaminar2
