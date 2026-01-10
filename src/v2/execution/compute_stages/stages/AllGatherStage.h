/**
 * @file AllGatherStage.h
 * @brief MPI AllGather stage for collecting distributed tensor slices
 */

#pragma once

#include "../IComputeStage.h"

namespace llaminar2
{

    /**
     * @brief MPI AllGather stage for collecting distributed tensor slices
     *
     * Used after column-parallel GEMM to reconstruct full output tensor.
     * For example, after LM head projection where each rank computes logits
     * for a slice of the vocabulary, AllGather combines them into full logits.
     *
     * Input: local_input [seq_len, vocab_local] on each rank
     * Output: full_output [seq_len, vocab_size] on ALL ranks (same data)
     */
    class AllGatherStage : public IComputeStage
    {
    public:
        struct Params
        {
            ITensor *local_input = nullptr;
            ITensor *full_output = nullptr;
            const MPIContext *mpi_ctx = nullptr; ///< MPI context (required)
            size_t actual_seq_len = 0;
        };

        explicit AllGatherStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::ALLGATHER; }
        bool requiresAllreduce() const override { return true; }
        bool supportsBackend(ComputeBackendType backend) const override;
        StageBufferRequirements getBufferRequirements() const override;
        StageDumpInfo getDumpInfo() const override;

        /// MPI stages handle their own synchronization - no automatic coherence
        CoherencePolicy coherencePolicy() const override { return CoherencePolicy::NONE; }

    private:
        Params params_;
    };

} // namespace llaminar2
