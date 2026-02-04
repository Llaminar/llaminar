/**
 * @file LocalPPTransferStage.h
 * @brief Activation transfer stage for LOCAL pipeline parallelism
 *
 * Transfers activations between devices within a single MPI rank
 * (intra-node GPU-to-GPU). Uses ILocalPPContext for backend selection
 * and device management.
 *
 * This stage is analogous to:
 * - SendActivationsStage/ReceiveActivationsStage for MPI PP (cross-node)
 * - LocalTPAllreduceStage for LOCAL TP (intra-node collective)
 *
 * Design rationale:
 * - Explicit compute stage provides graph visibility (vs. direct LocalPPContext calls)
 * - Consistent with other collective/transfer stages
 * - Enables proper coherence policy declaration (CoherencePolicy::NONE)
 * - Facilitates debugging, profiling, and stage dump framework integration
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#pragma once

#include "../IComputeStage.h"
#include "../StageParamsBase.h"
#include "../../../collective/ILocalPPContext.h"
#include <string>

namespace llaminar2
{

    /**
     * @brief Parameters for LocalPPTransferStage
     */
    struct LocalPPTransferParams
    {
        STAGE_PARAMS_COMMON_FIELDS;

        ILocalPPContext *pp_ctx = nullptr; ///< LOCAL PP context (required)
        TensorBase *tensor = nullptr;      ///< Tensor to transfer
        int stage_from = -1;               ///< Source PP stage index
        int stage_to = -1;                 ///< Destination PP stage index
        std::string stage_name;            ///< Stage identifier for debugging/profiling
    };

    /**
     * @brief Activation transfer stage for LOCAL pipeline parallelism
     *
     * Transfers activations from one PP stage device to another within
     * the same MPI rank. Uses the LOCAL PP context's backend (NCCL/RCCL/PCIeBAR)
     * based on device types.
     *
     * Usage:
     * @code
     *   LocalPPTransferStage::Params params{
     *       .device_id = source_device,
     *       .mpi_ctx = &mpi_ctx,
     *       .pp_ctx = local_pp_ctx.get(),
     *       .tensor = activations.get(),
     *       .stage_from = 0,
     *       .stage_to = 1,
     *       .stage_name = "layer12_pp_transfer",
     *   };
     *   auto stage = std::make_unique<LocalPPTransferStage>(params);
     * @endcode
     *
     * Thread safety: Execute must be called with appropriate synchronization.
     */
    class LocalPPTransferStage : public IComputeStage
    {
    public:
        using Params = LocalPPTransferParams;
        static_assert(StageParamsRequired<Params>, "Params must have device_id and mpi_ctx");

        /**
         * @brief Construct with parameters
         * @param params Stage parameters
         */
        explicit LocalPPTransferStage(Params params);

        ~LocalPPTransferStage() override = default;

        // =====================================================================
        // IComputeStage Interface
        // =====================================================================

        /**
         * @brief Execute the transfer operation
         *
         * Transfers tensor data from stage_from's device to stage_to's device.
         * If both stages are on the same device, this is a no-op.
         *
         * @param ctx Device context for execution (not directly used - pp_ctx handles devices)
         * @return true on success, false on error
         */
        bool execute(IDeviceContext *ctx) override;

        /**
         * @brief Get stage type
         * @return ComputeStageType::LOCAL_PP_TRANSFER
         */
        ComputeStageType type() const override { return ComputeStageType::LOCAL_PP_TRANSFER; }

        /**
         * @brief Get stage name
         * @return Custom stage name or "LocalPPTransfer"
         */
        std::string name() const override;

        /**
         * @brief Check if stage supports a backend type
         * @param backend Backend to check
         * @return true for all backends (LOCAL PP handles routing internally)
         */
        bool supportsBackend(ComputeBackendType backend) const override;

        /**
         * @brief Get buffer requirements for this stage
         * @return Buffer requirements (input tensor on source, output on dest)
         */
        StageBufferRequirements getBufferRequirements() const override;

        /**
         * @brief Get dump info for debugging
         * @return StageDumpInfo with tensor and transfer info
         */
        StageDumpInfo buildDumpInfoImpl() const override;

        /**
         * @brief Get coherence policy
         *
         * LOCAL PP stages handle their own device synchronization,
         * so we return NONE to avoid interfering with transfer semantics.
         *
         * @return CoherencePolicy::NONE
         */
        CoherencePolicy coherencePolicy() const override { return CoherencePolicy::NONE; }

        // =====================================================================
        // Accessors
        // =====================================================================

        /**
         * @brief Get the LOCAL PP context
         * @return Pointer to ILocalPPContext
         */
        ILocalPPContext *getPPContext() const { return params_.pp_ctx; }

        /**
         * @brief Get the tensor being transferred
         * @return Pointer to TensorBase
         */
        TensorBase *getTensor() const { return params_.tensor; }

        /**
         * @brief Get source stage index
         * @return Source PP stage index
         */
        int getStageFrom() const { return params_.stage_from; }

        /**
         * @brief Get destination stage index
         * @return Destination PP stage index
         */
        int getStageTo() const { return params_.stage_to; }

        /**
         * @brief Update parameters (for stage reuse)
         * @param params New parameters
         */
        void setParams(const Params &params);

    private:
        Params params_;
    };

} // namespace llaminar2
