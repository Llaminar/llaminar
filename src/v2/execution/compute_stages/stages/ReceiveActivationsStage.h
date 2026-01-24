/**
 * @file ReceiveActivationsStage.h
 * @brief Stage that receives activations from the previous pipeline stage (rank)
 *
 * Part of Pipeline Parallelism (Phase 2.2): Enables activation transfer between
 * pipeline stages running on different MPI ranks. Supports both synchronous and
 * asynchronous receive operations.
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "../IComputeStage.h"
#include "../StageParamsBase.h"
#include <mpi.h>

namespace llaminar2
{

    // Forward declarations
    class MPIContext;

    /**
     * @brief Stage that receives activations from the previous pipeline stage (rank)
     *
     * Used in pipeline parallelism to receive activations from the previous
     * pipeline stage. The source rank is typically the previous rank in the
     * pipeline (rank - 1), but can be any valid rank.
     *
     * Usage modes:
     * 1. Synchronous (async=false): Blocks until receive completes
     * 2. Asynchronous (async=true): Returns immediately, use wait() to complete
     *
     * Example:
     * @code
     *   ReceiveActivationsStage::Params params{
     *       .device_id = DeviceId::cpu(),
     *       .mpi_ctx = &mpi_ctx,
     *       .buffer = hidden_states,
     *       .src_rank = mpi_ctx.rank() - 1,
     *       .tag = mpi_tags::forwardTag(layer_idx),
     *       .async = false,
     *   };
     *   auto stage = ComputeStageFactory::createReceiveActivations(params);
     * @endcode
     */
    class ReceiveActivationsStage : public IComputeStage
    {
    public:
        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;

            ITensor *buffer = nullptr;   ///< Buffer to receive into
            int src_rank = -1;           ///< Source MPI rank (-1 for MPI_ANY_SOURCE)
            int tag = 0;                 ///< MPI tag (-1 for MPI_ANY_TAG)
            bool async = false;          ///< Non-blocking receive
            std::string stage_name = ""; ///< Optional custom stage name
        };

        static_assert(StageParamsRequired<Params>,
                      "ReceiveActivationsStage::Params must include device_id and mpi_ctx");

        explicit ReceiveActivationsStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::RECV_ACTIVATIONS; }
        std::string name() const override { return name_; }

        bool supportsBackend(ComputeBackendType backend) const override;
        StageDumpInfo buildDumpInfoImpl() const override;
        StageBufferRequirements getBufferRequirements() const override;

        /// MPI stages handle their own synchronization - no automatic coherence
        CoherencePolicy coherencePolicy() const override { return CoherencePolicy::NONE; }

        // =========================================================================
        // Async operation support
        // =========================================================================

        /**
         * @brief Check if async receive has completed
         *
         * Only valid after execute() with async=true.
         * Returns true if receive is complete or was synchronous.
         */
        bool isComplete() const;

        /**
         * @brief Wait for async receive to complete
         *
         * Blocks until the async receive operation finishes.
         * No-op if receive was synchronous or already complete.
         */
        void wait();

        /**
         * @brief Get pending request handle (for external waitAll)
         *
         * Returns MPI_REQUEST_NULL if no operation is pending.
         */
        MPI_Request getPendingRequest() const { return pending_request_; }

    private:
        Params params_;
        std::string name_;
        MPI_Request pending_request_ = MPI_REQUEST_NULL;
    };

} // namespace llaminar2
