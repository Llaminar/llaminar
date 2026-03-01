/**
 * @file AllGatherStage.h
 * @brief MPI AllGather stage with CollectiveContext support
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
#include "../../../config/TPDomain.h"

namespace llaminar2
{

    // Forward declarations
    class CollectiveContext;

    /**
     * @brief MPI AllGather stage for collecting distributed tensor slices
     *
     * Used after column-parallel GEMM to reconstruct full output tensor.
     * For example, after LM head projection where each rank computes logits
     * for a slice of the vocabulary, AllGather combines them into full logits.
     *
     * Input: local_input [seq_len, vocab_local] on each rank
     * Output: full_output [seq_len, vocab_size] on ALL ranks (same data)
     *
     * Can use either the new CollectiveContext infrastructure or direct MPI calls.
     */
    class AllGatherStage : public IComputeStage
    {
    public:
        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;

            ITensor *local_input = nullptr;
            ITensor *full_output = nullptr;
            size_t actual_seq_len = 0;
            CollectiveContext *collective_ctx = nullptr; ///< Collective context (preferred over direct MPI)
            const TPDomain *domain = nullptr;            ///< Domain for routing (nullptr = use mpi_ctx legacy path)
        };

        explicit AllGatherStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::ALLGATHER; }
        bool requiresAllreduce() const override { return true; }
        bool supportsBackend(ComputeBackendType backend) const override;
        StageBufferRequirements getBufferRequirements() const override;
        StageDumpInfo buildDumpInfoImpl() const override;

        /// MPI stages handle their own synchronization - no automatic coherence
        CoherencePolicy coherencePolicy() const override { return CoherencePolicy::OUTPUT; }

        /// Check if this stage uses the new CollectiveContext
        bool usesCollectiveContext() const { return params_.collective_ctx != nullptr; }

        /// Get the TPDomain this stage belongs to (nullptr = legacy mpi_ctx path)
        const TPDomain *getDomain() const { return params_.domain; }

        /// Get the stage parameters (for DeviceGraphExecutor strided allgather path)
        const Params &getParams() const { return params_; }

    private:
        Params params_;

        /// Execute using new CollectiveContext infrastructure
        bool executeViaCollectiveContext();

        /// Execute using direct MPI calls (legacy path)
        bool executeViaMPI();
    };

} // namespace llaminar2
