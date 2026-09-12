/**
 * @file MoEDeviceRebalanceStage.h
 * @brief Graph-capturable MoE runtime-table rebalance publish/apply stage.
 */

#pragma once

#include "../IComputeStage.h"
#include "../StageParamsBase.h"
#include "../../../execution/moe/DeviceMoERebalanceController.h"
#include "../../../execution/moe/DeviceMoERebalanceWorkspaceContract.h"
#include "../../../interfaces/IWorkspaceConsumer.h"

#include <memory>
#include <string>

namespace llaminar2
{
    class IBackend;
    class ILocalTPContext;
    class IMoEKernel;
    class IMoERuntimeTable;
    class DeviceMoEOverlayEpochArena;

    /**
     * @brief Declares which completion records one rebalance stage publishes.
     *
     * Rebalance graphs deliberately share persistent workspace across their
     * planning and payload transactions. Buffer allocation therefore cannot
     * prove that a particular graph replay wrote a completion record. This
     * contract makes publication ownership follow the typed stage phase
     * instead of the incidental presence of a shared workspace buffer.
     */
    struct DeviceMoERebalanceStatusPublicationContract
    {
        bool copy_status = false;
        bool apply_status = false;

        /**
         * @brief Merge another stage's publications into a graph-wide contract.
         *
         * @param other Publication contract for another stage in the same graph.
         */
        void merge(
            const DeviceMoERebalanceStatusPublicationContract &other) noexcept
        {
            copy_status = copy_status || other.copy_status;
            apply_status = apply_status || other.apply_status;
        }
    };

    /**
     * @brief Own one rebalance transaction's persistent GPU ordering events.
     *
     * A transfer state may reuse a context-owned auxiliary stream selected by
     * @p name_suffix, but its event handles must never be shared by unrelated
     * layers or graph transactions. CUDA and HIP graph nodes retain event
     * identities, so recording the same event repeatedly for several layers in
     * one captured graph makes producer/consumer ownership ambiguous. Graph
     * builders must therefore key this object by the layer-specific transfer
     * identity while separately selecting a bounded stream/workspace lane.
     *
     * The object contains no host mirror of transfer progress. Its events are
     * device-owned lifetime resources created before capture and destroyed only
     * when the owning graph lifetime ends.
     */
    class DeviceMoERebalanceTransferState
    {
    public:
        ~DeviceMoERebalanceTransferState();

        /**
         * @brief Materialize persistent resources when the stage workspace binds.
         *
         * This is a topology operation only: it may create the context-owned
         * auxiliary stream and this transaction's two event handles, but it must
         * never enqueue work or establish a dependency on either stream. Every
         * eager warmup or native graph capture begins. Every transaction records
         * compute-ready on its exact producer stream, performs auxiliary work,
         * and joins transfer-done back to its public graph stream. That complete
         * transaction edge makes graph completion the sole replay lifetime
         * boundary and keeps exported child fragments self-contained.
         *
         * Calling this method again is valid only for the same backend, device,
         * and named lane. A partially initialized or aliased state is a fatal
         * construction error rather than permission to rebind live graph events.
         *
         * @param device GPU that owns the stream and event resources.
         * @param name_suffix Stable name of the shared auxiliary stream lane.
         * @return true when all persistent resources are bound to this identity.
         */
        bool materializePersistentResources(
            DeviceId device,
            const std::string &name_suffix);

        /**
         * @brief Validate that capture preparation installed the exact resources.
         *
         * Execution calls this read-only predicate instead of attempting lazy
         * allocation. A false result therefore fails the graph transaction before
         * any kernel or collective is submitted.
         *
         * @param device GPU expected to own the resources.
         * @param name_suffix Stable lane identity expected by the stage.
         * @return true only for a complete, identity-matching resource set.
         */
        [[nodiscard]] bool isMaterializedFor(
            DeviceId device,
            const std::string &name_suffix) const;

        void release();

        void *transferStream() const { return transfer_stream_; }
        void *computeReadyEvent() const { return compute_ready_event_; }
        void *transferDoneEvent() const { return transfer_done_event_; }

    private:
        IBackend *event_backend_ = nullptr;
        int event_device_ordinal_ = -1;
        std::string stream_lane_name_;
        void *transfer_stream_ = nullptr;
        void *compute_ready_event_ = nullptr;
        void *transfer_done_event_ = nullptr;
    };

    /**
     * @brief Device-side MoE rebalance controller stage for homogeneous LocalTP domains.
     *
     * This stage keeps the hot-path publish/apply loop out of the host runner:
     * it packs per-layer decode histograms from the mirrored runtime table,
     * all-gathers those histograms across the LocalTP domain on the stage
     * stream, launches the backend device controller, copies planned arrivals
     * through graph-owned transfer slots on a context-owned auxiliary stream,
     * and applies completed arrivals back into the mirrored runtime table.
     *
     * The transfer stream dependency edges are recorded as GPU graph nodes.
     * Host code may allocate the persistent stream/event handles before
     * capture, but host code must not publish or apply expert placement after
     * this stage is active.
     */
    class MoEDeviceRebalanceStage : public IComputeStage, public IWorkspaceConsumer
    {
    public:
        static constexpr const char *WS_LOCAL_HISTOGRAM =
            DeviceMoERebalanceWorkspaceContract::WS_LOCAL_HISTOGRAM;
        static constexpr const char *WS_GATHERED_HISTOGRAM =
            DeviceMoERebalanceWorkspaceContract::WS_GATHERED_HISTOGRAM;
        static constexpr const char *WS_TRANSFER_PLAN =
            DeviceMoERebalanceWorkspaceContract::WS_TRANSFER_PLAN;
        static constexpr const char *WS_TRANSFER_PLAN_COUNT =
            DeviceMoERebalanceWorkspaceContract::WS_TRANSFER_PLAN_COUNT;
        static constexpr const char *WS_COMMAND_HEADER =
            DeviceMoERebalanceWorkspaceContract::WS_COMMAND_HEADER;
        static constexpr const char *WS_CONTROLLER_STATE =
            DeviceMoERebalanceWorkspaceContract::WS_CONTROLLER_STATE;
        static constexpr const char *WS_MOVEMENT_WAVES =
            DeviceMoERebalanceWorkspaceContract::WS_MOVEMENT_WAVES;
        static constexpr const char *WS_MOVEMENT_EDGES =
            DeviceMoERebalanceWorkspaceContract::WS_MOVEMENT_EDGES;
        static constexpr const char *WS_PLACEMENT_PLAN_SCRATCH =
            DeviceMoERebalanceWorkspaceContract::WS_PLACEMENT_PLAN_SCRATCH;
        static constexpr const char *WS_LLEP_LAYER_PLANS =
            DeviceMoERebalanceWorkspaceContract::WS_LLEP_LAYER_PLANS;
        static constexpr const char *WS_GATHERED_TRANSFER_PLAN =
            DeviceMoERebalanceWorkspaceContract::WS_GATHERED_TRANSFER_PLAN;
        static constexpr const char *WS_GATHERED_COMMAND_HEADER =
            DeviceMoERebalanceWorkspaceContract::WS_GATHERED_COMMAND_HEADER;
        static constexpr const char *WS_TRANSFER_SLOT_CLAIM_INDEX =
            DeviceMoERebalanceWorkspaceContract::WS_TRANSFER_SLOT_CLAIM_INDEX;
        static constexpr const char *WS_WAVE_STATE =
            DeviceMoERebalanceWorkspaceContract::WS_WAVE_STATE;
        static constexpr const char *WS_GATHERED_WAVE_STATE =
            DeviceMoERebalanceWorkspaceContract::WS_GATHERED_WAVE_STATE;
        static constexpr const char *WS_STATUS =
            DeviceMoERebalanceWorkspaceContract::WS_STATUS;
        static constexpr const char *WS_LOCAL_DIRECTORY =
            DeviceMoERebalanceWorkspaceContract::WS_LOCAL_DIRECTORY;
        static constexpr const char *WS_LOCAL_SOURCE_DESCRIPTORS =
            DeviceMoERebalanceWorkspaceContract::WS_LOCAL_SOURCE_DESCRIPTORS;
        static constexpr const char *WS_LOCAL_TRANSFER_PAYLOAD =
            DeviceMoERebalanceWorkspaceContract::WS_LOCAL_TRANSFER_PAYLOAD;
        static constexpr const char *WS_GATHERED_TRANSFER_PAYLOAD =
            DeviceMoERebalanceWorkspaceContract::WS_GATHERED_TRANSFER_PAYLOAD;
        static constexpr const char *WS_COPY_STATUS =
            DeviceMoERebalanceWorkspaceContract::WS_COPY_STATUS;
        static constexpr const char *WS_GATHERED_COPY_STATUS =
            DeviceMoERebalanceWorkspaceContract::WS_GATHERED_COPY_STATUS;
        static constexpr const char *WS_APPLY_STATUS =
            DeviceMoERebalanceWorkspaceContract::WS_APPLY_STATUS;

        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;

            ILocalTPContext *tp_ctx = nullptr;
            IMoERuntimeTable *moe_runtime_table = nullptr;
            int tp_device_idx = -1;
            DeviceMoERebalanceConfig config;
            DeviceMoEExpertDirectoryEntry *local_transfer_slots = nullptr;
            uint32_t local_transfer_slot_count = 0;
            uint64_t collective_payload_slot_bytes = 0;
            uint32_t collective_payload_slot_capacity = 0;
            std::string stage_name;
            std::string workspace_name;
            DeviceMoERebalanceStagePhase phase = DeviceMoERebalanceStagePhase::PlanCopyApply;
            DeviceMoERebalanceTransferMode transfer_mode = DeviceMoERebalanceTransferMode::ResidentOnly;
            int apply_layer_idx = -1;
            bool join_transfer_stream_after_copy = true;
            std::shared_ptr<DeviceMoERebalanceTransferState> transfer_state;
            /**
             * @brief Optional durable placement-publication authority.
             *
             * Only the standalone all-layer maintenance graph may bind this
             * arena. Current-batch LLEP and route-piggybacked apply stages must
             * leave it null because their placement is transient or cannot
             * publish a complete main/MTP runtime-table family.
             */
            std::shared_ptr<DeviceMoEOverlayEpochArena> overlay_epoch_arena;
        };

        static_assert(StageParamsRequired<Params>);

        static std::string workspaceBufferName(
            const char *base_name,
            const std::string &workspace_name);

        explicit MoEDeviceRebalanceStage(Params params);
        ~MoEDeviceRebalanceStage() override;

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::MOE_DEVICE_REBALANCE; }
        /**
         * @brief Report whether this phase launches a standalone LocalTP collective.
         *
         * Rebalance is phase-split: sideband-only pack/apply phases are local,
         * while state gathering and transfer-backed planning/transport phases
         * launch raw NCCL/RCCL operations. Capture policy must inspect this
         * concrete stage rather than classifying every rebalance phase alike.
         */
        bool isCollectiveStage() const override;
        std::string name() const override { return params_.stage_name.empty() ? "moe_device_rebalance" : params_.stage_name; }
        size_t estimatedFlops() const override { return 0; }
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        bool isGraphCapturable() const override;
        bool prepareGraphLaunch(IDeviceContext *ctx, void *stream) override;
        GraphLaunchPreparationPolicy graphLaunchPreparationPolicy() const override
        {
            return usesTransferSlotApply()
                       ? GraphLaunchPreparationPolicy::CaptureOnly
                       : GraphLaunchPreparationPolicy::None;
        }

        /**
         * @brief Report whether this stage owns the persistent request transaction.
         *
         * A phase-split graph may contain several stages that consume the same
         * workspace. Exactly the stage that runs planning owns the request
         * lifetime transition, preventing reset code from resetting one shared
         * transaction repeatedly merely because several consumers can name it.
         */
        bool ownsRequestTransactionState() const noexcept;

        /**
         * @brief Reset all request-owned transaction contents on an explicit stream.
         *
         * Model-lifetime workspace addresses remain stable for captured replay.
         * The controller, command headers, wave cursors, and plan counts are
         * reset together in stream order. The operation is asynchronous and
         * becomes visible through the orchestrator's reset-ready event.
         *
         * @param stream Exact request-reset stream; null is a fatal contract error.
         * @return true when the backend reset kernel was enqueued successfully.
         */
        bool resetRequestTransactionStateOnStream(void *stream);

        /**
         * @brief Reset this stage's private MoE launch metadata on hard invalidation.
         */
        void invalidateKernelDynamicState() override;
        StageDumpInfo buildDumpInfoImpl() const override;
        StageBufferRequirements getBufferRequirements() const override { return {}; }
        StageBufferContract bufferContract() const override { return {}; }
        CoherencePolicy coherencePolicy() const override { return CoherencePolicy::NONE; }

        WorkspaceRequirements getWorkspaceRequirements(int m, int n = 0, int k = 0) const override;
        void bindWorkspace(DeviceWorkspaceManager *workspace) override;
        void unbindWorkspace() override;
        bool hasWorkspace() const override { return bound_workspace_ != nullptr; }
        DeviceWorkspaceManager *getWorkspace() const override { return bound_workspace_; }

        const Params &getParams() const { return params_; }
        size_t traceHistogramLayerCount() const { return histogramLayerCount(); }
        size_t traceLocalHistogramEntries() const { return localHistogramEntries(); }
        size_t traceGatheredHistogramEntries() const { return gatheredHistogramEntries(); }
        /**
         * @return Exact prebound journal storage and canonical capacities.
         * This projects pointer metadata only; it never reads device-owned state.
         */
        DeviceMoERebalanceMovementJournalView movementJournalView() const;
        /**
         * @brief Return the completion records owned by this stage phase.
         *
         * Diagnostics and lifecycle validation must use this contract before
         * reading shared status workspace. A record named by the contract is
         * mandatory and malformed content is fatal. An unnamed record belongs
         * to another graph transaction and must not be attributed to this one.
         */
        DeviceMoERebalanceStatusPublicationContract
        statusPublicationContract() const;

    private:
        Params params_;
        DeviceWorkspaceManager *bound_workspace_ = nullptr;
        /**
         * @brief Rebalance-stage-owned controller and transfer launch state.
         *
         * Maintenance graphs use a transfer stream in addition to the compute
         * stream. Owning the backend object here prevents unrelated MoE stages
         * from changing its workspace or inherited stream while either graph
         * is being captured or replayed.
         */
        std::unique_ptr<IMoEKernel> owned_moe_kernel_;

        std::string localHistogramBufferName() const;
        std::string gatheredHistogramBufferName() const;
        std::string transferPlanBufferName() const;
        std::string transferPlanCountBufferName() const;
        std::string commandHeaderBufferName() const;
        std::string controllerStateBufferName() const;
        std::string placementPlanScratchBufferName() const;
        std::string llepLayerPlansBufferName() const;
        std::string gatheredTransferPlanBufferName() const;
        std::string gatheredCommandHeaderBufferName() const;
        std::string transferSlotClaimIndexBufferName() const;
        std::string waveStateBufferName() const;
        std::string gatheredWaveStateBufferName() const;
        std::string statusBufferName() const;
        std::string localDirectoryBufferName() const;
        std::string localSourceDescriptorsBufferName() const;
        std::string localTransferPayloadBufferName() const;
        std::string gatheredTransferPayloadBufferName() const;
        std::string copyStatusBufferName() const;
        std::string gatheredCopyStatusBufferName() const;
        std::string applyStatusBufferName() const;
        size_t localHistogramEntries() const;
        size_t gatheredHistogramEntries() const;
        size_t histogramLayerCount() const;
        size_t llepPlannerScratchCount() const;
        size_t localDirectoryEntries() const;
        size_t localSourceDescriptorEntries() const;
        size_t transferPlanEntries() const;
        size_t transferPlanCapacity() const;
        size_t payloadSlotCapacity() const;
        size_t commandBufferCount() const;
        size_t collectivePayloadSlotCount() const;
        size_t collectivePayloadLocalBytes() const;
        size_t collectivePayloadGatheredBytes() const;
        bool usesTransferSlotApply() const;
        bool usesCompactTransferSlots() const;
        bool usesFixedPayloadTransfer() const;
        bool usesCollectivePayloadLane() const;
        bool usesReadyWaveApply() const;
        bool collectsState() const;
        bool gathersStateInline() const;
        bool runsController() const;
        bool usesParallelLLEPPlanning() const;
        bool runsPlanning() const;
        bool runsApply() const;
        bool validateCommon(const char *context) const;
        std::string workspaceSuffix() const;
        /** @return Pointer-free capacity projected from this stage's parameters. */
        DeviceMoERebalanceWorkspaceCapacity workspaceCapacity() const noexcept;
        /** @return Complete canonical workspace binding for this stage. */
        DeviceMoERebalanceWorkspaceBinding workspaceBinding() const;
        bool requireMaterializedAsyncTransferState() const;
        DeviceMoERebalanceTransferState *transferState() const;
    };

} // namespace llaminar2
