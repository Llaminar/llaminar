/**
 * @file GlobalPPTransferStage.h
 * @brief Activation transfer stage for GLOBAL pipeline parallelism (cross-rank MPI)
 *
 * Transfers activations between MPI ranks for Global Pipeline Parallel execution.
 * Uses MPI_Send/MPI_Recv for point-to-point communication.
 *
 * This is the cross-rank counterpart of LocalPPTransferStage (which handles
 * intra-node GPU-to-GPU transfers within a single MPI rank).
 *
 * Design:
 * - Each stage instance is either a SEND or RECV (not both)
 * - Uses synchronous MPI (blocking send/recv) for simplicity
 * - Handles GPU→host coherence for send (data() triggers sync)
 * - Marks tensor as host-authoritative after recv
 * - CoherencePolicy::NONE — we manage coherence explicitly
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#pragma once

#include "../IComputeStage.h"
#include "../StageParamsBase.h"
#include <string>

namespace llaminar2
{

    /**
     * @brief Parameters for GlobalPPTransferStage
     */
    struct GlobalPPTransferParams
    {
        STAGE_PARAMS_COMMON_FIELDS;

        /// Direction of transfer
        enum class Direction
        {
            SEND,
            RECV
        };

        Direction direction = Direction::SEND;   ///< Send or receive
        ITensor *tensor = nullptr;               ///< Activation tensor to transfer
        size_t count = 0;                        ///< Elements to transfer (0 = tensor->numel())
        int peer_rank = -1;                      ///< MPI rank to send to / receive from
        int tag = 0;                             ///< MPI tag for this transfer
        std::string stage_name;                  ///< Stage identifier for debugging/profiling
    };

    /**
     * @brief Activation transfer stage for Global Pipeline Parallelism
     *
     * Transfers activations from one MPI rank to another using synchronous
     * MPI point-to-point communication. Each instance handles either a send
     * or a receive operation.
     *
     * Usage:
     * @code
     *   // Sender side (rank 0)
     *   GlobalPPTransferStage::Params send_params{
     *       .device_id = DeviceId::cpu(),
     *       .mpi_ctx = &mpi_ctx,
     *       .direction = GlobalPPTransferParams::Direction::SEND,
     *       .tensor = hidden_states.get(),
     *       .peer_rank = 1,
     *       .tag = 100,
     *       .stage_name = "gpp_send_s0_to_s1",
     *   };
     *   auto send_stage = std::make_unique<GlobalPPTransferStage>(send_params);
     *
     *   // Receiver side (rank 1)
     *   GlobalPPTransferStage::Params recv_params{
     *       .device_id = DeviceId::cpu(),
     *       .mpi_ctx = &mpi_ctx,
     *       .direction = GlobalPPTransferParams::Direction::RECV,
     *       .tensor = hidden_states.get(),
     *       .peer_rank = 0,
     *       .tag = 100,
     *       .stage_name = "gpp_recv_s0_to_s1",
     *   };
     *   auto recv_stage = std::make_unique<GlobalPPTransferStage>(recv_params);
     * @endcode
     *
     * Thread safety: Execute must be called with appropriate synchronization.
     * Paired send/recv stages on different ranks must use matching tags.
     */
    class GlobalPPTransferStage : public IComputeStage
    {
    public:
        using Params = GlobalPPTransferParams;
        static_assert(StageParamsRequired<Params>, "Params must have device_id and mpi_ctx");

        /**
         * @brief Construct with parameters
         * @param params Stage parameters (direction, tensor, peer_rank, tag)
         */
        explicit GlobalPPTransferStage(Params params);

        ~GlobalPPTransferStage() override = default;

        // =====================================================================
        // IComputeStage Interface
        // =====================================================================

        /**
         * @brief Execute the MPI send or recv operation
         *
         * For SEND: Calls tensor->data() to ensure GPU→host sync, then MPI_Send.
         * For RECV: Calls tensor->mutable_data() to get host buffer, then MPI_Recv,
         *           then publishes the host write through TransferEngine.
         *
         * @param ctx Device context (not used — MPI operations are host-side)
         * @return true on success, false on error
         */
        bool execute(IDeviceContext *ctx) override;

        /**
         * @brief Get stage type
         * @return ComputeStageType::GLOBAL_PP_TRANSFER
         */
        ComputeStageType type() const override { return ComputeStageType::GLOBAL_PP_TRANSFER; }

        /**
         * @brief Get stage name
         * @return Custom stage name or auto-generated "gpp_send_to_rankN" / "gpp_recv_from_rankN"
         */
        std::string name() const override;

        /**
         * @brief Check if stage supports a backend type
         * @param backend Backend to check
         * @return true for all backends (MPI handles communication internally)
         */
        bool supportsBackend(ComputeBackendType backend) const override;

        /**
         * @brief Get buffer requirements for this stage
         * @return Buffer requirements (tensor as input for SEND, output for RECV)
         */
        StageBufferRequirements getBufferRequirements() const override;

        /**
         * @brief Get dump info for debugging
         * @return StageDumpInfo with tensor and transfer metadata
         */
        StageDumpInfo buildDumpInfoImpl() const override;

        /**
         * @brief Get coherence policy
         *
         * Global PP stages handle their own host/device coherence:
         * - SEND: data() triggers GPU→host sync
         * - RECV: TransferEngine::publishHostWrite() after receiving
         *
         * @return CoherencePolicy::NONE
         */
        CoherencePolicy coherencePolicy() const override { return CoherencePolicy::NONE; }

        // =====================================================================
        // Accessors
        // =====================================================================

        /** @brief Get transfer direction */
        GlobalPPTransferParams::Direction getDirection() const { return params_.direction; }

        /** @brief Get the tensor being transferred */
        ITensor *getTensor() const { return params_.tensor; }

        /** @brief Get peer MPI rank */
        int getPeerRank() const { return params_.peer_rank; }

        /** @brief Get MPI tag */
        int getTag() const { return params_.tag; }

        /** @brief Get element count (0 means use tensor->numel()) */
        size_t getCount() const { return params_.count; }

        /**
         * @brief Update parameters (for stage reuse across iterations)
         * @param params New parameters
         */
        void setParams(const Params &params);

    private:
        Params params_;
        std::string name_;

        /// Execute a send operation
        bool executeSend();

        /// Execute a receive operation
        bool executeRecv();
    };

} // namespace llaminar2
