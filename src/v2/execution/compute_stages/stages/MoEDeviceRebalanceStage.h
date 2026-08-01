/**
 * @file MoEDeviceRebalanceStage.h
 * @brief Graph-capturable MoE runtime-table rebalance publish/apply stage.
 */

#pragma once

#include "../IComputeStage.h"
#include "../StageParamsBase.h"
#include "../../../execution/moe/DeviceMoERebalanceController.h"
#include "../../../interfaces/IWorkspaceConsumer.h"

#include <memory>
#include <string>

namespace llaminar2
{
    class IBackend;
    class ILocalTPContext;
    class IMoEKernel;
    class IMoERuntimeTable;

    enum class DeviceMoERebalanceStagePhase
    {
        /**
         * Atomic maintenance transaction: collect and gather state, run the
         * controller, publish command metadata, prepare immutable payload
         * bytes, transport arrivals, and apply the ready wave before recording
         * one terminal completion edge.
         *
         * This is the required standalone maintenance-graph path. It prevents
         * the host from observing intermediate state or selecting a follow-up
         * graph while preserving explicit auxiliary-stream overlap inside the
         * captured transaction.
         */
        PlanCopyApply,

        /**
         * Graph-capturable collection phase for piggybacked rebalance metadata:
         * pack local histogram/directory state only. A following LocalTP
         * collective sideband all-gathers these compact buffers on an existing
         * compute collective.
         */
        CollectState,

        /**
         * Graph-capturable producer phase: pack histograms/directories, run the
         * device controller, enqueue transfer-slot copies on the context-owned
         * transfer stream, and record completion without joining the compute
         * stream.
         */
        PlanAndCopy,

        /**
         * Graph-capturable producer phase after a sideband collective has
         * already gathered local histogram/directory state. This runs the
         * device controller and transfer path without launching standalone
         * state all-gathers.
         */
        PlanAndCopyAfterSideband,

        /**
         * Pack this participant's requested expert payloads after command
         * buffers have been all-gathered as sidebands on an existing decode
         * collective. This phase must not launch a rebalance-specific TP
         * collective.
         */
        PackCollectivePayloadAfterSideband,

        /**
         * Unpack all-gathered collective payload sideband bytes into local
         * transfer slots and publish transfer completion. This phase must not
         * launch a rebalance-specific TP collective.
         */
        UnpackCollectivePayloadAfterSideband,

        /**
         * Graph-capturable consumer phase: wait for the transfer completion
         * event and apply arrived transfer slots to the mirrored runtime table.
         */
        Apply,

        /**
         * Graph-captured stream-join phase. Split producers enqueue transfer
         * work on the context-owned transfer stream without joining immediately;
         * a late JoinTransfer stage waits on the transfer completion event near
         * the end of the decode graph so transfer can overlap later compute
         * while stream capture still has an explicit rejoin edge.
         */
        JoinTransfer,
    };

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
         * @brief Materialize the persistent auxiliary stream and ordering events.
         *
         * This method is suitable for eager execution setup. Graph-captured
         * callers should use prepareForCapture(), which additionally orders the
         * public capture stream after work already queued on the transfer lane.
         *
         * @param device GPU that owns the stream and event resources.
         * @param name_suffix Stable name of the shared auxiliary stream lane.
         * @return true when all persistent resources are available.
         */
        bool ensure(DeviceId device, const std::string &name_suffix);

        /**
         * @brief Prepare this transaction's event edges for graph launch.
         *
         * Stream and event creation must happen before CUDA/HIP capture begins.
         * The shared auxiliary stream can retain work from an earlier eager
         * launch or graph replay. This method records this owner's terminal
         * event after that stream work and queues a wait on the public capture
         * stream, making the lifetime boundary explicit without a host or
         * device synchronization.
         *
         * @param device GPU that owns both streams.
         * @param name_suffix Stable name of the shared auxiliary stream lane.
         * @param capture_stream Explicit stream on which capture/replay will start.
         * @return true when resources exist and the device-side ordering edge was
         *         queued successfully.
         */
        bool prepareForCapture(
            DeviceId device,
            const std::string &name_suffix,
            void *capture_stream);

        void release();

        void *transferStream() const { return transfer_stream_; }
        void *computeReadyEvent() const { return compute_ready_event_; }
        void *transferDoneEvent() const { return transfer_done_event_; }

    private:
        IBackend *event_backend_ = nullptr;
        int event_device_ordinal_ = -1;
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
        static constexpr const char *WS_LOCAL_HISTOGRAM = "moe_rebalance_local_histogram";
        static constexpr const char *WS_GATHERED_HISTOGRAM = "moe_rebalance_gathered_histogram";
        static constexpr const char *WS_TRANSFER_PLAN = "moe_rebalance_transfer_plan";
        static constexpr const char *WS_TRANSFER_PLAN_COUNT = "moe_rebalance_transfer_plan_count";
        static constexpr const char *WS_COMMAND_HEADER = "moe_rebalance_command_header";
        static constexpr const char *WS_CONTROLLER_STATE = "moe_rebalance_controller_state";
        static constexpr const char *WS_GATHERED_TRANSFER_PLAN = "moe_rebalance_gathered_transfer_plan";
        static constexpr const char *WS_GATHERED_COMMAND_HEADER = "moe_rebalance_gathered_command_header";
        static constexpr const char *WS_WAVE_STATE = "moe_rebalance_wave_state";
        static constexpr const char *WS_GATHERED_WAVE_STATE = "moe_rebalance_gathered_wave_state";
        static constexpr const char *WS_STATUS = "moe_rebalance_status";
        static constexpr const char *WS_LOCAL_DIRECTORY = "moe_rebalance_local_directory";
        static constexpr const char *WS_LOCAL_SOURCE_DESCRIPTORS = "moe_rebalance_local_source_descriptors";
        static constexpr const char *WS_LOCAL_TRANSFER_PAYLOAD = "moe_rebalance_local_transfer_payload";
        static constexpr const char *WS_GATHERED_TRANSFER_PAYLOAD = "moe_rebalance_gathered_transfer_payload";
        static constexpr const char *WS_COPY_STATUS = "moe_rebalance_copy_status";
        static constexpr const char *WS_GATHERED_COPY_STATUS = "moe_rebalance_gathered_copy_status";
        static constexpr const char *WS_APPLY_STATUS = "moe_rebalance_apply_status";

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
        bool needsGraphLaunchPreparation() const override { return usesTransferSlotApply(); }

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
        std::string gatheredTransferPlanBufferName() const;
        std::string gatheredCommandHeaderBufferName() const;
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
        bool runsPlanning() const;
        bool runsApply() const;
        bool validateCommon(const char *context) const;
        std::string workspaceSuffix() const;
        bool ensureAsyncTransferState();
        DeviceMoERebalanceTransferState *transferState() const;
    };

} // namespace llaminar2
