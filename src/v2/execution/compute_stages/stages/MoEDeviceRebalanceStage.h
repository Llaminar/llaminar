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
    class IMoERuntimeTable;

    enum class DeviceMoERebalanceStagePhase
    {
        /**
         * Compatibility mode: plan, copy arrivals, wait for completion, and
         * apply them in one stage. Production decode graphs should prefer the
         * split PlanAndCopy + Apply phases so transfer can overlap with compute.
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
         * Graph-capturable maintenance collection phase: pack local histogram
         * state and all-gather it immediately, but do not run the controller.
         * The async maintenance graph uses this before Apply so a ready-wave
         * apply can reset live histograms without erasing the just-observed
         * planning window.
         */
        CollectAndGatherState,

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
         * Graph-capturable maintenance producer phase after histogram state has
         * already been gathered. This runs the device controller and writes
         * local command buffers, but deliberately stops before any command
         * metadata allgather or expert payload movement. The follow-up
         * MetadataAndPayload graph runs only when the completed probe requested
         * non-empty payload slots.
         */
        PlanProbeAfterSideband,

        /**
         * Graph-capturable maintenance follow-up phase. This consumes command
         * metadata prepared by PlanProbeAfterSideband, gathers/projects command
         * buffers across the rebalance domain, moves only the compact payload
         * bucket for the active wave, and publishes transfer completion.
         */
        GatherCommandsAndCopyPreparedPayload,

        /**
         * Graph-capturable maintenance payload phase. This consumes already
         * gathered/projected command metadata, moves the prepared payload bucket,
         * and publishes transfer completion. New maintenance replay should use
         * GatherCommandsAndCopyPreparedPayload so no-work probes do not pay the
         * command-buffer allgather.
         */
        CopyPreparedPayload,

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

    class DeviceMoERebalanceTransferState
    {
    public:
        ~DeviceMoERebalanceTransferState();

        bool ensure(DeviceId device, const std::string &name_suffix);
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
            uint64_t payload_edge_mask = 0;
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
        std::string name() const override { return params_.stage_name.empty() ? "moe_device_rebalance" : params_.stage_name; }
        size_t estimatedFlops() const override { return 0; }
        size_t estimatedMemoryBytes() const override;
        bool supportsBackend(ComputeBackendType backend) const override;
        bool isGraphCapturable() const override;
        bool prepareGraphLaunch(IDeviceContext *ctx, void *stream) override;
        bool needsGraphLaunchPreparation() const override { return usesTransferSlotApply(); }
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

    private:
        Params params_;
        DeviceWorkspaceManager *bound_workspace_ = nullptr;

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
