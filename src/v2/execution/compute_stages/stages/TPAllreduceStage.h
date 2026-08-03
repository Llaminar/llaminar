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
#include "../../../collective/ILocalTPContext.h"
#include "../../../collective/ITPContext.h"
#include "../../../interfaces/IWorkspaceConsumer.h"
#include "../../../memory/BufferId.h"
#include <optional>
#include <string>
#include <vector>

namespace llaminar2
{
    class DeviceWorkspaceManager;
    struct WorkspaceRequirements;

    /**
     * @brief Native rooted LocalTP collective selected declaratively by a graph.
     */
    enum class TPLocalRootedCollectiveOperation
    {
        ReduceSum,
        Broadcast
    };

    /** @return Stable diagnostic name for @p operation. */
    const char *toString(TPLocalRootedCollectiveOperation operation) noexcept;

    /**
     * @brief Exact tensor ownership role of one rooted-collective participant.
     *
     * A rooted collective is not uniformly in-place on every participant.
     * Reduce contributors and the broadcast root only consume their local
     * tensor, while the reduce root and broadcast receivers produce bytes.
     * Keeping this distinction typed prevents the graph executor from
     * requiring nonexistent receiver input or publishing imaginary writes.
     */
    enum class TPLocalRootedCollectiveTensorRole
    {
        ReduceRootInOut,
        ReduceContributorInput,
        BroadcastRootInput,
        BroadcastReceiverOutput
    };

    /** @return Stable diagnostic name for @p role. */
    const char *toString(TPLocalRootedCollectiveTensorRole role) noexcept;

    /**
     * @brief Workspace-backed control sideband attached to a TP allreduce.
     *
     * The graph builder names persistent device workspace buffers here. At
     * execution/capture time TPAllreduceStage resolves them to raw device
     * pointers and passes the compact sideband to LocalTP. This keeps rebalance
     * metadata on the graph stream without baking allocator addresses into graph
     * construction.
     */
    struct TPAllreduceSidebandWorkspaceBinding
    {
        LocalTPCollectiveSidebandKind kind = LocalTPCollectiveSidebandKind::AllreduceSum;
        std::string send_buffer_name;
        std::string recv_buffer_name;
        size_t element_count = 0;
        CollectiveDataType dtype = CollectiveDataType::INT32;
        int root_device_index = 0;
        std::string name;
    };

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
        int sideband_device_index = -1;           ///< LocalTP participant index for optional sidebands.
        std::vector<LocalTPCollectiveSidebandBuffer> sidebands; ///< Optional same-stream control sidebands.
        std::vector<TPAllreduceSidebandWorkspaceBinding> sideband_workspace_bindings; ///< Workspace-resolved sidebands.
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
    class TPAllreduceStage : public IComputeStage, public IWorkspaceConsumer
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
         * @brief TP allreduce sidebands reuse buffers declared by producer stages.
         */
        WorkspaceRequirements getWorkspaceRequirements(
            int m, int n = 0, int k = 0) const override;
        void bindWorkspace(DeviceWorkspaceManager *workspace) override;
        void unbindWorkspace() override;
        bool hasWorkspace() const override { return bound_workspace_ != nullptr; }
        DeviceWorkspaceManager *getWorkspace() const override { return bound_workspace_; }

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
         * @brief Return the explicit collective element count.
         * @return Number of tensor elements passed to the collective, or zero
         *         when the full tensor size is selected at execution time.
         */
        [[nodiscard]] size_t getCount() const { return params_.count; }

        /**
         * @brief Return the graph-bound collective precision policy.
         * @return Empty string for the global policy, otherwise the explicit
         *         precision name supplied by the graph builder.
         */
        [[nodiscard]] const std::string &getPrecision() const
        {
            return params_.precision;
        }

        /**
         * @brief Return the arena identity of the in-place tensor.
         * @return Optional BufferId used by declarative coherence handling.
         */
        [[nodiscard]] std::optional<BufferId> getTensorBufferId() const
        {
            return params_.tensor_buffer_id;
        }

        /**
         * @brief Update parameters (for stage reuse)
         * @param params New parameters
         */
        void setParams(const Params &params);

    private:
        Params params_;
        DeviceWorkspaceManager *bound_workspace_ = nullptr;
    };

    /**
     * @brief Graph-capturable rooted collective with participant-specific ownership.
     *
     * This stage exists for protocols whose result is needed on one fixed root
     * before a compact publication. `ReduceSum` reduces the complete tensor to
     * that root; `Broadcast` publishes the root tensor to every participant.
     * The declarative contract distinguishes existing root/contributor inputs
     * from receiver outputs that the executor prepares before capture. Both
     * operations are issued directly on the executor-selected explicit stream.
     * There is no host rendezvous, transport emulation, hot-path allocation, or
     * synchronization path.
     */
    class TPLocalRootedCollectiveStage final
        : public IComputeStage,
          public IWorkspaceConsumer
    {
    public:
        /** @brief Immutable graph-bound operation parameters. */
        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;

            ILocalTPContext *tp_ctx = nullptr;
            TensorBase *tensor = nullptr;
            size_t count = 0;
            CollectiveDataType dtype = CollectiveDataType::FLOAT32;
            TPLocalRootedCollectiveOperation operation =
                TPLocalRootedCollectiveOperation::ReduceSum;
            int root_device_index = 0;
            int participant_device_index = -1;
            std::string stage_name;
            std::optional<BufferId> tensor_buffer_id;
            /**
             * @brief Persistent-workspace metadata collectives ordered after
             *        the rooted activation collective on the same stream.
             *
             * Rebalance metadata used to ride beside the routed activation
             * allreduce. Keeping these declarations on the replacement stage
             * preserves one symmetric collective order on every participant
             * without retaining a dummy activation allreduce.
             */
            std::vector<TPAllreduceSidebandWorkspaceBinding>
                sideband_workspace_bindings;
        };

        explicit TPLocalRootedCollectiveStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override
        {
            return ComputeStageType::ROOTED_COLLECTIVE;
        }
        std::string name() const override
        {
            return "tp_local_rooted_collective";
        }
        bool requiresAllreduce() const override { return true; }
        bool supportsBackend(ComputeBackendType backend) const override;
        bool isGraphCapturable() const override;
        StageBufferRequirements getBufferRequirements() const override;
        StageBufferContract bufferContract() const override;
        StageDumpInfo buildDumpInfoImpl() const override;
        WorkspaceRequirements getWorkspaceRequirements(
            int m,
            int n = 0,
            int k = 0) const override;
        void bindWorkspace(DeviceWorkspaceManager *workspace) override;
        void unbindWorkspace() override;
        bool hasWorkspace() const override
        {
            return bound_workspace_ != nullptr;
        }
        DeviceWorkspaceManager *getWorkspace() const override
        {
            return bound_workspace_;
        }
        CoherencePolicy coherencePolicy() const override
        {
            return CoherencePolicy::OUTPUT;
        }
        /**
         * @brief Return this participant's exact read/write ownership role.
         * @return Role derived solely from the graph-bound operation, root,
         *         and participant indices.
         */
        [[nodiscard]] TPLocalRootedCollectiveTensorRole tensorRole() const
            noexcept;
        /** @return Immutable operation parameters for graph regressions. */
        [[nodiscard]] const Params &params() const { return params_; }

    private:
        void recordBillOfMaterials() const;

        Params params_;
        DeviceWorkspaceManager *bound_workspace_ = nullptr;

        /**
         * @brief Prebound device descriptors for same-stream control collectives.
         *
         * Workspace names are resolved once while the graph is bound. Keeping
         * the resulting descriptors here makes execute() allocation-free and
         * guarantees that capture observes stable device addresses. The vector
         * reserves its complete capacity during stage construction and is only
         * populated outside graph capture by bindWorkspace().
         */
        std::vector<LocalTPCollectiveSidebandBuffer> bound_sidebands_;
    };

} // namespace llaminar2
