/**
 * @file EmbeddingStage.h
 * @brief Embedding lookup stage
 */

#pragma once

#include "IComputeStage.h"

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
     */
    class EmbeddingStage : public IComputeStage
    {
    public:
        struct Params
        {
            // Input/output tensors
            const TensorBase *embed_table = nullptr;
            const int *token_ids = nullptr;
            TensorBase *output = nullptr;

            // Dimensions
            int num_tokens = 0;
            int d_model = 0;
            int vocab_size = 0;

            // Batched input (alternative to token_ids)
            const std::vector<std::vector<int>> *token_batches = nullptr;
            int padded_seq_len = 0;

            // Device placement
            const MPIContext *mpi_ctx = nullptr;
            int device_idx = -1;
        };

        explicit EmbeddingStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::EMBEDDING; }
        size_t estimatedFlops() const override;
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        StageDumpInfo getDumpInfo() const override;
        StageBufferRequirements getBufferRequirements() const override;

    private:
        Params params_;

        bool executeQ16_1Output();
        static void zero_output_row(TensorBase *output, int row_idx, int d_model);
    };

} // namespace llaminar2
