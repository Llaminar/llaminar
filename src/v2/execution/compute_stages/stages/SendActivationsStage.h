/**
 * @file SendActivationsStage.h
 * @brief Stage that sends activations to the next pipeline stage (rank)
 *
 * Part of Pipeline Parallelism (Phase 2.2): Enables activation transfer between
 * pipeline stages running on different MPI ranks. Supports both synchronous and
 * asynchronous send operations.
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
     * @brief Stage that sends activations to the next pipeline stage (rank)
     *
     * Used in pipeline parallelism to transfer activations from one pipeline
     * stage to the next. The destination rank is typically the next rank in
     * the pipeline (rank + 1), but can be any valid rank.
     *
     * Usage modes:
     * 1. Synchronous (async=false): Blocks until send completes
     * 2. Asynchronous (async=true): Returns immediately, use wait() to complete
     *
     * Example:
     * @code
     *   SendActivationsStage::Params params{
     *       .device_id = DeviceId::cpu(),
     *       .mpi_ctx = &mpi_ctx,
     *       .buffer = hidden_states,
     *       .dest_rank = mpi_ctx.rank() + 1,
     *       .tag = mpi_tags::forwardTag(layer_idx),
     *       .async = false,
     *   };
     *   auto stage = ComputeStageFactory::createSendActivations(params);
     * @endcode
     */
    class SendActivationsStage : public IComputeStage
    {
    public:
        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;

            ITensor *buffer = nullptr;   ///< Activation tensor to send
            int dest_rank = -1;          ///< Destination MPI rank
            int tag = 0;                 ///< MPI tag (use mpi_tags::forwardTag(layer))
            bool async = false;          ///< Non-blocking send
            std::string stage_name = ""; ///< Optional custom stage name
        };

        static_assert(StageParamsRequired<Params>,
                      "SendActivationsStage::Params must include device_id and mpi_ctx");

        explicit SendActivationsStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::SEND_ACTIVATIONS; }
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
         * @brief Check if async send has completed
         *
         * Only valid after execute() with async=true.
         * Returns true if send is complete or was synchronous.
         */
        bool isComplete() const;

        /**
         * @brief Wait for async send to complete
         *
         * Blocks until the async send operation finishes.
         * No-op if send was synchronous or already complete.
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
