/**
 * @file AllreduceStage.h
 * @brief MPI Allreduce stage
 */

#pragma once

#include "IComputeStage.h"

namespace llaminar2
{

    /**
     * @brief MPI Allreduce stage
     */
    class AllreduceStage : public IComputeStage
    {
    public:
        struct Params
        {
            TensorBase *buffer = nullptr; ///< Buffer to allreduce (in-place)
            void *mpi_comm = nullptr;     ///< MPI communicator (cast to MPI_Comm)
            size_t count = 0;             ///< Number of elements to reduce (0 = use buffer->numel())
        };

        explicit AllreduceStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::ALLREDUCE; }
        bool requiresAllreduce() const override { return true; }
        bool supportsBackend(ComputeBackendType backend) const override;
        StageBufferRequirements getBufferRequirements() const override;
        StageDumpInfo getDumpInfo() const override;

    private:
        Params params_;
    };

} // namespace llaminar2
