/**
 * @file LocalTPAllreduceStage.h
 * @brief All-reduce stage for LOCAL tensor parallelism
 *
 * Performs all-reduce across devices within a single MPI rank,
 * using ILocalTPContext for device management and backend selection.
 *
 * This stage is used after row-parallel GEMM operations (e.g., Wo projection,
 * FFN down projection) to sum partial results across LOCAL TP devices.
 *
 * Distinction from AllreduceStage:
 * - AllreduceStage: Cross-rank MPI all-reduce (GLOBAL TP)
 * - LocalTPAllreduceStage: Intra-rank device all-reduce (LOCAL TP)
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "../IComputeStage.h"
#include "../StageParamsBase.h"
#include "../../../collective/ILocalTPContext.h"
#include <string>

namespace llaminar2
{

    /**
     * @brief Parameters for LocalTPAllreduceStage
     */
    struct LocalTPAllreduceParams
    {
        STAGE_PARAMS_COMMON_FIELDS;

        ILocalTPContext *tp_ctx = nullptr; ///< LOCAL TP context (required)
        TensorBase *tensor = nullptr;      ///< Tensor to all-reduce (in-place)
        size_t count = 0;                  ///< Elements to reduce (0 = use tensor->numel())
        std::string stage_name;            ///< Stage identifier for registered tensor lookup (optional)
    };

    /**
     * @brief All-reduce stage for LOCAL tensor parallelism
     *
     * Performs in-place sum reduction across all devices in the LOCAL TP context.
     * Uses NCCL/RCCL/HOST based on backend configuration in tp_ctx.
     *
     * Thread safety: Execute must be called from appropriate device context.
     */
    class LocalTPAllreduceStage : public IComputeStage
    {
    public:
        using Params = LocalTPAllreduceParams;
        static_assert(StageParamsRequired<Params>, "Params must have device_id and mpi_ctx");

        /**
         * @brief Construct with parameters
         * @param params Stage parameters
         */
        explicit LocalTPAllreduceStage(Params params);

        ~LocalTPAllreduceStage() override = default;

        // =====================================================================
        // IComputeStage Interface
        // =====================================================================

        /**
         * @brief Execute the all-reduce operation
         *
         * Reduces tensor values across all LOCAL TP devices using sum reduction.
         * The operation is in-place: input tensor is modified with result.
         *
         * @param ctx Device context for execution
         * @return true on success, false on error
         */
        bool execute(IDeviceContext *ctx) override;

        /**
         * @brief Get stage type
         * @return ComputeStageType::ALLREDUCE (uses same type as MPI version)
         */
        ComputeStageType type() const override { return ComputeStageType::ALLREDUCE; }

        /**
         * @brief Get stage name
         * @return "LocalTPAllreduce"
         */
        std::string name() const override { return "LocalTPAllreduce"; }

        /**
         * @brief Check if stage requires all-reduce
         * @return true
         */
        bool requiresAllreduce() const override { return true; }

        /**
         * @brief Check if stage supports a backend type
         * @param backend Backend to check
         * @return true for all backends (LOCAL TP handles routing internally)
         */
        bool supportsBackend(ComputeBackendType backend) const override;

        /**
         * @brief Get buffer requirements for this stage
         * @return Buffer requirements (single in-place buffer)
         */
        StageBufferRequirements getBufferRequirements() const override;

        /**
         * @brief Get dump info for debugging
         * @return StageDumpInfo with tensor info
         */
        StageDumpInfo buildDumpInfoImpl() const override;

        /**
         * @brief Get coherence policy
         *
         * LOCAL TP stages handle their own synchronization across devices,
         * so we return NONE to avoid interfering with device-specific sync.
         *
         * @return CoherencePolicy::NONE
         */
        CoherencePolicy coherencePolicy() const override { return CoherencePolicy::NONE; }

        // =====================================================================
        // Accessors
        // =====================================================================

        /**
         * @brief Get the LOCAL TP context
         * @return Pointer to ILocalTPContext
         */
        ILocalTPContext *getTPContext() const { return params_.tp_ctx; }

        /**
         * @brief Get the tensor being reduced
         * @return Pointer to TensorBase
         */
        TensorBase *getTensor() const { return params_.tensor; }

        /**
         * @brief Update parameters (for stage reuse)
         * @param params New parameters
         */
        void setParams(const Params &params);

    private:
        Params params_;
    };

} // namespace llaminar2
