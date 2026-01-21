/**
 * @file AllreduceStage.h
 * @brief MPI Allreduce stage with CollectiveContext support
 *
 * This stage now supports two execution modes:
 * 1. CollectiveContext (preferred): Delegates to the new collective infrastructure
 *    which can use MPI, NCCL, RCCL, or Host backends depending on configuration
 * 2. Direct MPI (legacy): Falls back to direct MPI calls for backward compatibility
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "../IComputeStage.h"
#include "../StageParamsBase.h"
#include "config/TPDomain.h"

namespace llaminar2
{
    // Forward declarations
    class CollectiveContext;

    /**
     * @brief MPI Allreduce stage with CollectiveContext support
     *
     * Performs in-place allreduce across all MPI ranks.
     * Can use either the new CollectiveContext infrastructure or direct MPI calls.
     */
    class AllreduceStage : public IComputeStage
    {
    public:
        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;

            ITensor *buffer = nullptr;                   ///< Buffer to allreduce (in-place)
            size_t count = 0;                            ///< Number of elements to reduce (0 = use buffer->numel())
            CollectiveContext *collective_ctx = nullptr; ///< Collective context (preferred over direct MPI)
            const TPDomain *domain = nullptr;            ///< Domain for routing (nullptr = use mpi_ctx legacy path)
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

        /// Check if this stage uses the new CollectiveContext
        bool usesCollectiveContext() const { return params_.collective_ctx != nullptr; }

        /// Get the TPDomain this stage belongs to (nullptr = legacy mpi_ctx path)
        const TPDomain *getDomain() const { return params_.domain; }

    private:
        Params params_;

        /// Execute using new CollectiveContext infrastructure
        bool executeViaCollectiveContext();

        /// Execute using direct MPI calls (legacy path)
        bool executeViaMPI();
    };

} // namespace llaminar2
