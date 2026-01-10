/**
 * @file AllreduceStage.h
 * @brief MPI Allreduce stage
 */

#pragma once

#include "../IComputeStage.h"

namespace llaminar2
{
    // Forward declaration
    class MPIContext;

    /**
     * @brief MPI Allreduce stage
     */
    class AllreduceStage : public IComputeStage
    {
    public:
        struct Params
        {
            ITensor *buffer = nullptr;           ///< Buffer to allreduce (in-place)
            const MPIContext *mpi_ctx = nullptr; ///< MPI context (required)
            size_t count = 0;                    ///< Number of elements to reduce (0 = use buffer->numel())
        };

        explicit AllreduceStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::ALLREDUCE; }
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
