/**
 * @file TPAllreduceStage.h
 * @brief All-reduce stage for tensor parallelism (LOCAL and GLOBAL)
 *
 * Performs all-reduce across devices within a TP context, supporting both:
 * - LOCAL TP: Intra-rank device all-reduce (NCCL/RCCL/HOST)
 * - GLOBAL TP: Cross-rank MPI all-reduce (UPI/MPI backends)
 *
 * This stage uses ITPContext to abstract over both LOCAL and GLOBAL contexts,
 * enabling unified stage creation regardless of TP scope.
 *
 * This stage is used after row-parallel GEMM operations (e.g., Wo projection,
 * FFN down projection) to sum partial results across TP devices.
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#pragma once

#include "../IComputeStage.h"
#include "../StageParamsBase.h"
#include "../../../collective/ITPContext.h"
#include "../../../memory/BufferId.h"
#include <optional>
#include <string>

namespace llaminar2
{

    /**
     * @brief Parameters for TPAllreduceStage
     */
    struct TPAllreduceParams
    {
        STAGE_PARAMS_COMMON_FIELDS;

        ITPContext *tp_ctx = nullptr;             ///< TP context (LOCAL or GLOBAL, required)
        TensorBase *tensor = nullptr;             ///< Tensor to all-reduce (in-place)
        size_t count = 0;                         ///< Elements to reduce (0 = use tensor->numel())
        std::string stage_name;                   ///< Stage identifier for registered tensor lookup (optional)
        std::string precision;                    ///< Allreduce precision override ("fp32", "fp16", "bf16", "" = use global default)
        std::optional<BufferId> tensor_buffer_id; ///< Arena BufferId for the in-place tensor (enables contract-based coherence)
    };

    /**
     * @brief All-reduce stage for tensor parallelism
     *
     * Performs in-place sum reduction across all devices in the TP context.
     * The actual backend (NCCL/RCCL/HOST for LOCAL, UPI/MPI for GLOBAL)
     * is determined by the tp_ctx implementation.
     *
     * Thread safety: Execute must be called from appropriate device context.
     */
    class TPAllreduceStage : public IComputeStage
    {
    public:
        using Params = TPAllreduceParams;
        static_assert(StageParamsRequired<Params>, "Params must have device_id and mpi_ctx");

        /**
         * @brief Construct with parameters
         * @param params Stage parameters
         */
        explicit TPAllreduceStage(Params params);

        ~TPAllreduceStage() override = default;

        // =====================================================================
        // IComputeStage Interface
        // =====================================================================

        /**
         * @brief Execute the all-reduce operation
         *
         * Reduces tensor values across all TP devices using sum reduction.
         * The operation is in-place: input tensor is modified with result.
         *
         * @param ctx Device context for execution
         * @return true on success, false on error
         */
        bool execute(IDeviceContext *ctx) override;

        /**
         * @brief Get stage type
         * @return ComputeStageType::ALLREDUCE
         */
        ComputeStageType type() const override { return ComputeStageType::ALLREDUCE; }

        /**
         * @brief Get stage name
         * @return "TPAllreduce"
         */
        std::string name() const override { return "TPAllreduce"; }

        /**
         * @brief Check if stage requires all-reduce
         * @return true
         */
        bool requiresAllreduce() const override { return true; }

        /**
         * @brief Check if stage supports a backend type
         * @param backend Backend to check
         * @return true for all backends (TP context handles routing internally)
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
         * @brief Declarative buffer contract for arena-based coherence
         * @return Contract with single inout binding, or empty if no buffer_id set
         */
        StageBufferContract bufferContract() const override;

        /**
         * @brief Get coherence policy
         *
         * OUTPUT: The allreduce operates in-place on GPU buffers that are
         * already on-device (cohered by the preceding GEMM stage), so we
         * skip INPUT coherence. But we MUST mark outputs dirty so that
         * snapshot callbacks and subsequent host reads trigger a D2H copy
         * to get the post-allreduce data (not stale pre-allreduce data).
         *
         * @return CoherencePolicy::OUTPUT
         */
        CoherencePolicy coherencePolicy() const override { return CoherencePolicy::OUTPUT; }

        // =====================================================================
        // Accessors
        // =====================================================================

        /**
         * @brief Get the TP context (LOCAL or GLOBAL)
         * @return Pointer to ITPContext
         */
        ITPContext *getTPContext() const { return params_.tp_ctx; }

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
