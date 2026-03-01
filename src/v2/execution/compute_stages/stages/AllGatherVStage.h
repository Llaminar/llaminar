/**
 * @file AllGatherVStage.h
 * @brief Variable-sized AllGather stage for heterogeneous tensor parallelism
 *
 * This stage supports variable send counts per rank, needed when devices have
 * different head counts (e.g., 20 heads on NVIDIA vs 8 heads on AMD).
 *
 * Unlike regular AllGatherStage which assumes equal send counts, AllGatherVStage
 * uses MPI_Allgatherv semantics where each rank can send a different amount.
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
     * @brief Variable-sized AllGather stage for heterogeneous tensor parallelism
     *
     * Each rank can send a different amount of data (e.g., different head counts).
     * The recv_counts and displacements define the layout of the gathered output.
     *
     * Input: local_input [seq_len, local_dim] - this rank's data
     * Output: full_output [seq_len, sum(recv_counts)] - all gathered data
     *
     * The recv_counts array specifies how many elements each rank contributes.
     * The displacements array specifies the offset in full_output for each rank's data.
     */
    class AllGatherVStage : public IComputeStage
    {
    public:
        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;

            ITensor *local_input = nullptr; ///< This rank's data to send
            ITensor *full_output = nullptr; ///< Buffer to receive all gathered data

            /// Variable sizes per rank (size = world_size)
            std::vector<int> recv_counts;   ///< Elements per rank
            std::vector<int> displacements; ///< Offset in output per rank

            size_t actual_seq_len = 0;                   ///< Actual sequence length (0 = use tensor rows)
            CollectiveContext *collective_ctx = nullptr; ///< Collective context (preferred)
            const TPDomain *domain = nullptr;            ///< Domain for routing (nullptr = use mpi_ctx legacy path)
        };

        explicit AllGatherVStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::ALLGATHER_V; }
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

    private:
        Params params_;

        /// Execute using CollectiveContext infrastructure
        bool executeViaCollectiveContext();

        /// Execute using direct MPI calls (legacy/fallback path)
        bool executeViaMPI();
    };

} // namespace llaminar2
